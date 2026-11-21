"""SzybkiS3Gen – szybka ścieżka S3Gen (tokeny mowy -> audio 24 kHz) dla Chatterboxa.

Opracowane w badaniu opóźnień 20.11 (prototyp: scratchpad/badanie_s3gen/szybki_s3gen.py), tu z podziałem
na etapy mel() i hift() – potrzebnym do zmiany tempa w melspektrogramie i do strumieniowania.
Samodzielny moduł (zależy tylko od torch/numpy/einops i obiektu S3Gen z pakietu chatterbox). Działa z obydwoma dekoderami:
  - standardowym s3gen.pt (CFM, domyślnie 10 kroków Eulera + CFG 0.7 – jak oryginał),
  - zdestylowanym meanflow z Chatterbox-Turbo (S3Gen(meanflow=True) + s3gen_meanflow.safetensors, 1-2 kroki, bez CFG).

Co robi inaczej niż S3Gen.inference (matematycznie to samo, poza dtype):
  1. trzy grafy CUDA na kubełek długości: enkoder przepływu, estymator CFM, wokoder HiFT
     (oryginał jest ograniczony narzutem uruchamiania ~1500 małych jąder na krok + 14 synchronizacji .item()),
  2. estymator (i enkoder) w fp16 – osadzenie czasu i stan ODE zostają w fp32,
  3. HiFT bez parametryzacji weight_norm (wagi zapieczone), generator sinusów i iSTFT bez CPU.
Wejścia dopełniane w prawo do kubełka; maski sprawiają, że wynik dla prawdziwych ramek się nie zmienia
(wyjątek: HiFT nie jest przyczynowy – ostatnie ramki mogą się minimalnie różnić; i tak są w ucinanym ogonie).

Użycie:
    s3 = S3Gen(); s3.load_state_dict(...); s3.to("cuda").eval()
    sz = SzybkiS3Gen(s3, n_krokow=4)                # std: np. 4 kroki z CFG; meanflow: n_krokow=2
    sz.rozgrzej(ref_dict, maks_tokenow=400)          # nagrywa grafy z góry (kilka s, raz przy starcie)
    wav = sz.inference(tokeny, ref_dict)             # (1, N) float32 na GPU, jak s3.inference(...)[0]
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"


# ============================================================================ estymator CFM

def _decoder_forward(est, x, mask, mu, t, spks, cond, r=None):
    """ConditionalDecoder.forward bez add_optional_chunk_mask (.item() = synchronizacja, nie do nagrania).
    static_chunk_size=0 => maska uwagi = maska paddingu, ta sama we wszystkich blokach.
    Osadzenie czasu liczone w fp32 (t*1000 w sinusach – w fp16/bf16 błąd fazy byłby duży)."""
    from einops import pack, rearrange, repeat
    from chatterbox.models.s3gen.decoder import mask_to_bias
    td = est.time_mlp.linear_1.weight.dtype
    t = est.time_mlp(est.time_embeddings(t.float()).to(td))
    if est.meanflow:
        r = est.time_mlp(est.time_embeddings(r.float()).to(td))
        t = est.time_embed_mixer(torch.cat([t, r], dim=1).to(est.time_embed_mixer.weight.dtype))
    t = t.to(x.dtype)
    x = pack([x, mu], "b * t")[0]
    x = pack([x, repeat(spks, "b c -> b c t", t=x.shape[-1])], "b * t")[0]
    x = pack([x, cond], "b * t")[0]
    bias = mask_to_bias(mask.bool(), x.dtype)
    hiddens = []
    for resnet, tbs, down in est.down_blocks:
        x = resnet(x, mask, t)
        x = rearrange(x, "b c t -> b t c").contiguous()
        for tb in tbs:
            x = tb(hidden_states=x, attention_mask=bias, timestep=t)
        x = rearrange(x, "b t c -> b c t").contiguous()
        hiddens.append(x)
        x = down(x * mask)
    for resnet, tbs in est.mid_blocks:
        x = resnet(x, mask, t)
        x = rearrange(x, "b c t -> b t c").contiguous()
        for tb in tbs:
            x = tb(hidden_states=x, attention_mask=bias, timestep=t)
        x = rearrange(x, "b t c -> b c t").contiguous()
    for resnet, tbs, up in est.up_blocks:
        skip = hiddens.pop()
        x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
        x = resnet(x, mask, t)
        x = rearrange(x, "b c t -> b t c").contiguous()
        for tb in tbs:
            x = tb(hidden_states=x, attention_mask=bias, timestep=t)
        x = rearrange(x, "b t c -> b c t").contiguous()
        x = up(x * mask)
    x = est.final_block(x, mask)
    return est.final_proj(x * mask) * mask


def _nagraj(fn, pool):
    """Rozgrzewka na bocznym strumieniu + nagranie grafu; zwraca (graf, wynik)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g, pool=pool):
        out = fn()
    return g, out


class _GrafyEstymatora:
    def __init__(self, est, B, dtype, meanflow, kubelek=64):
        self.est, self.B, self.dtype, self.meanflow, self.kub = est, B, dtype, meanflow, kubelek
        self.grafy, self.pool = {}, None

    def Tb(self, T):
        return int(math.ceil(T / self.kub) * self.kub)

    def _nowy(self, Tb):
        B, d = self.B, self.dtype
        st = {k: torch.zeros(B, 80, Tb, device=DEV, dtype=d) for k in ("x", "mu", "cond")}
        st["mask"] = torch.zeros(B, 1, Tb, device=DEV, dtype=d)
        st["mask"][:, :, :8] = 1
        st["spks"] = torch.zeros(B, 80, device=DEV, dtype=d)
        st["t"] = torch.zeros(B, device=DEV)
        st["r"] = torch.zeros(B, device=DEV)
        g, out = _nagraj(lambda: _decoder_forward(self.est, st["x"], st["mask"], st["mu"], st["t"], st["spks"],
                                                  st["cond"], st["r"] if self.meanflow else None), self.pool)
        self.pool = self.pool or g.pool()
        self.grafy[Tb] = (g, st, out)

    def ustaw(self, T, mu, spks, cond):
        Tb = self.Tb(T)
        if Tb not in self.grafy:
            self._nowy(Tb)
        g, st, out = self.grafy[Tb]
        for k in ("x", "mu", "cond", "mask", "spks"):
            st[k].zero_()
        st["mask"][:, :, :T] = 1
        st["mu"][0:1, :, :T] = mu
        st["spks"][0:1] = spks
        st["cond"][0:1, :, :T] = cond            # przy CFG druga połowa (bez warunku) zostaje zerowa
        self.akt = (g, st, out, T)

    def __call__(self, x, t, r):
        g, st, out, T = self.akt
        st["x"][:, :, :T] = x
        st["t"].fill_(0).add_(t)
        st["r"].fill_(0).add_(r)
        g.replay()
        return out[:, :, :T]


# ============================================================================ enkoder przepływu

class _GrafyEnkodera:
    """input_embedding + UpsampleConformerEncoder + encoder_proj, kubełki co 16 tokenów (z tokenami podpowiedzi).
    Po warstwie embed zerujemy dopełnienie, bo PreLookaheadLayer patrzy 3 tokeny w przód (w oryginale widzi
    tam zera z F.pad) – wynik dla prawdziwych tokenów jak bez dopełnienia."""

    def __init__(self, flow, dtype, kubelek=16):
        self.dtype, self.kub = dtype, kubelek
        self.emb = copy.deepcopy(flow.input_embedding).to(dtype).eval()
        self.enc = copy.deepcopy(flow.encoder).to(dtype).eval()
        self.proj = copy.deepcopy(flow.encoder_proj).to(dtype).eval()
        self.grafy, self.pool = {}, None

    def Nb(self, N):
        return int(math.ceil(N / self.kub) * self.kub)

    def _fn(self, st):
        enc = self.enc
        x = self.emb(st["tok"]) * st["mf"]
        xs, pos, _ = enc.embed(x, st["mb"])
        xs = enc.pre_lookahead_layer(xs * st["mf"])
        for layer in enc.encoders:
            xs = layer(xs, st["mb"], pos, st["mb"])[0]
        xs = xs.transpose(1, 2).contiguous()
        xs = F.interpolate(xs, scale_factor=float(enc.up_layer.stride), mode="nearest")
        xs = enc.up_layer.conv(F.pad(xs, (enc.up_layer.stride * 2, 0), value=0.0))
        xs = xs.transpose(1, 2).contiguous()
        xs, pos2, _ = enc.up_embed(xs, st["mb2"])
        for layer in enc.up_encoders:
            xs = layer(xs, st["mb2"], pos2, st["mb2"])[0]
        return self.proj(enc.after_norm(xs))

    def _nowy(self, Nb):
        st = dict(tok=torch.zeros(1, Nb, dtype=torch.long, device=DEV),
                  mf=torch.zeros(1, Nb, 1, dtype=self.dtype, device=DEV),
                  mb=torch.zeros(1, 1, Nb, dtype=torch.bool, device=DEV),
                  mb2=torch.zeros(1, 1, 2 * Nb, dtype=torch.bool, device=DEV))
        st["mf"][:, :8] = 1
        st["mb"][:, :, :8] = True
        st["mb2"][:, :, :16] = True
        g, out = _nagraj(lambda: self._fn(st), self.pool)
        self.pool = self.pool or g.pool()
        self.grafy[Nb] = (g, st, out)

    def __call__(self, tok):
        N = tok.shape[1]
        Nb = self.Nb(N)
        if Nb not in self.grafy:
            self._nowy(Nb)
        g, st, out = self.grafy[Nb]
        for k in ("tok", "mf", "mb", "mb2"):
            st[k].zero_()
        st["tok"][:, :N] = tok
        st["mf"][:, :N] = 1
        st["mb"][:, :, :N] = True
        st["mb2"][:, :, :2 * N] = True
        g.replay()
        return out[:, :2 * N]


# ============================================================================ HiFT

def _kopia_hift_bez_weight_norm(h):
    """Nowy HiFTGenerator (te same argumenty co w S3Token2Wav), wagi z oryginału, weight_norm zapieczony.
    (copy.deepcopy modułu z parametryzacjami współdzieli ich stan – usunięcie na kopii psuje oryginał.)"""
    from torch.nn.utils import parametrize
    from chatterbox.models.s3gen.hifigan import HiFTGenerator
    from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
    nowy = HiFTGenerator(sampling_rate=24000, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
                         source_resblock_kernel_sizes=[7, 7, 11],
                         source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                         f0_predictor=ConvRNNF0Predictor())
    nowy.load_state_dict(h.state_dict())
    nowy = nowy.to(DEV).eval()
    for m in nowy.modules():
        if parametrize.is_parametrized(m):
            for name in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)
    nowy.stft_window = nowy.stft_window.to(DEV)
    return nowy


def _hift(h, mels):
    """HiFTGenerator.inference bez CPU: f0 -> źródło (sinusy na GPU) -> dekoder -> iSTFT (irfft + fold)."""
    f0 = h.f0_predictor(mels).float()
    x0 = h.f0_upsamp(f0[:, None])                                        # [1, 1, L]
    sg = h.m_source.l_sin_gen
    H = sg.harmonic_num + 1
    F_mat = x0 * torch.arange(1, H + 1, device=DEV, dtype=torch.float32).view(1, H, 1) / sg.sampling_rate
    theta = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
    phase = (torch.rand(1, H, 1, device=DEV) * 2 - 1) * np.pi
    phase[:, 0] = 0
    uv = (x0 > sg.voiced_threshold).float()
    sine = sg.sine_amp * torch.sin(theta + phase) * uv + (uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3) * torch.randn_like(theta)
    s = h.m_source.l_tanh(h.m_source.l_linear(sine.transpose(1, 2))).transpose(1, 2)   # [1, 1, L]
    s_re, s_im = h._stft(s.squeeze(1))
    s_stft = torch.cat([s_re, s_im], dim=1)
    x = h.conv_pre(mels)
    for i in range(h.num_upsamples):
        x = h.ups[i](F.leaky_relu(x, h.lrelu_slope))
        if i == h.num_upsamples - 1:
            x = h.reflection_pad(x)
        x = x + h.source_resblocks[i](h.source_downs[i](s_stft))
        xs = h.resblocks[i * h.num_kernels](x)
        for j in range(1, h.num_kernels):
            xs = xs + h.resblocks[i * h.num_kernels + j](x)
        x = xs / h.num_kernels
    x = h.conv_post(F.leaky_relu(x))
    n_fft, hop = h.istft_params["n_fft"], h.istft_params["hop_len"]
    nf = n_fft // 2 + 1
    mag = torch.clip(torch.exp(x[:, :nf]), max=1e2)
    ph = torch.sin(x[:, nf:])
    fr = torch.fft.irfft(torch.complex(mag * torch.cos(ph), mag * torch.sin(ph)), n=n_fft, dim=1)
    win = h.stft_window
    fr = fr * win.view(1, -1, 1)
    Fr = fr.shape[-1]
    L = n_fft + hop * (Fr - 1)
    y = F.fold(fr, (1, L), (1, n_fft), stride=(1, hop)).view(1, L)
    env = F.fold((win * win).view(1, -1, 1).expand(1, n_fft, Fr), (1, L), (1, n_fft), stride=(1, hop)).view(1, L)
    y = y[:, n_fft // 2: L - n_fft // 2] / env[:, n_fft // 2: L - n_fft // 2]
    return torch.clamp(y, -h.audio_limit, h.audio_limit)


class _GrafyHiFT:
    def __init__(self, h, kubelek=32):
        self.h, self.kub = h, kubelek
        self.grafy, self.pool = {}, None

    def Tb(self, T):
        return int(math.ceil(T / self.kub) * self.kub)

    def _nowy(self, Tb):
        mel = torch.full((1, 80, Tb), -11.5, device=DEV)
        g, out = _nagraj(lambda: _hift(self.h, mel), self.pool)
        self.pool = self.pool or g.pool()
        self.grafy[Tb] = (g, mel, out)

    def __call__(self, mels):
        T = mels.shape[-1]
        Tb = self.Tb(T)
        if Tb not in self.grafy:
            self._nowy(Tb)
        g, mel, out = self.grafy[Tb]
        mel[:, :, :T] = mels
        mel[:, :, T:] = mels[:, :, T - 1:T]          # dopełnienie powieleniem ostatniej ramki
        g.replay()
        return out[:, :T * 480]


# ============================================================================ całość

class SzybkiS3Gen:
    def __init__(self, s3, n_krokow: int | None = None, cfg: float | None = None, dtype=torch.float16):
        self.s3 = s3
        self.meanflow = bool(getattr(s3, "meanflow", False))
        self.n = n_krokow or (2 if self.meanflow else 10)
        self.cfg = float(cfg if cfg is not None else (0.0 if self.meanflow else s3.flow.decoder.inference_cfg_rate))
        self.dtype = dtype
        est = copy.deepcopy(s3.flow.decoder.estimator).to(dtype).eval()
        est.time_mlp.float()
        if est.time_embed_mixer is not None:
            est.time_embed_mixer.float()
        self.g_est = _GrafyEstymatora(est, 2 if self.cfg > 0 else 1, dtype, self.meanflow)
        self.g_enc = _GrafyEnkodera(s3.flow, dtype)
        # kopia: oryginalny model zostaje nietknięty (nazwa inna niż metoda hift() – atrybut by ją zasłonił)
        self.model_hift = _kopia_hift_bez_weight_norm(s3.mel2wav)
        self.g_hift = _GrafyHiFT(self.model_hift)
        self._t = {}

    def _kroki(self, n):
        """Pary (t, r) dla n kroków Eulera (cosinusowy harmonogram jak w oryginale; meanflow – liniowy)."""
        if n not in self._t:
            t = torch.linspace(0, 1, n + 1, device=DEV)
            t = t if self.meanflow else 1 - torch.cos(t * 0.5 * torch.pi)
            self._t[n] = [(t[i:i + 1].clone(), t[i + 1:i + 2].clone()) for i in range(n)]
        return self._t[n]

    @torch.inference_mode()
    def rozgrzej(self, ref, maks_tokenow=400, min_tokenow=1):
        """Nagrywa grafy dla wszystkich kubełków od min_tokenow do maks_tokenow tokenów mowy (dla danej podpowiedzi)."""
        P = int(ref["prompt_token"].shape[1])
        mel1 = int(ref["prompt_feat"].shape[1])
        for n in range(min_tokenow, maks_tokenow + 1):
            N = P + n
            if self.g_enc.Nb(N) not in self.g_enc.grafy:
                self.g_enc._nowy(self.g_enc.Nb(N))
            T = 2 * N
            if self.g_est.Tb(T) not in self.g_est.grafy:
                self.g_est._nowy(self.g_est.Tb(T))
            T2 = 2 * N - mel1
            if T2 > 0 and self.g_hift.Tb(T2) not in self.g_hift.grafy:
                self.g_hift._nowy(self.g_hift.Tb(T2))
        torch.cuda.synchronize()

    @torch.inference_mode()
    def mel(self, speech_tokens, ref, n_krokow=None, szum=None, finalize=True):
        """Melspektrogram (1, 80, 2N) wygenerowanej części (bez podpowiedzi). `szum` – opcjonalny stały szum
        startowy (1, 80, >=T) indeksowany bezwzględną pozycją ramki (podpowiedź + mowa), wspólny dla całej
        kwestii. finalize=False (strumieniowanie): ostatnie pre_lookahead_len tokenów (3) to tylko
        „wyprzedzenie” enkodera – ich ramki odcinamy przed dekoderem (jak flow.inference w oryginale).
        `ref` może mieć podpowiedź wydłużoną o już oddaną mowę (strumieniowanie w trybie „prefix”)."""
        s3 = self.s3
        tokens = torch.atleast_2d(speech_tokens).to(DEV)
        flow = s3.flow
        prompt_feat = ref["prompt_feat"].to(DEV, torch.float32)
        tok = torch.cat([ref["prompt_token"].to(DEV).long(), tokens.long()], dim=1)
        h = self.g_enc(tok).float()                                             # (1, T, 80)
        if not finalize:
            h = h[:, : h.shape[1] - flow.pre_lookahead_len * flow.token_mel_ratio]
        T = h.shape[1]
        mel1 = prompt_feat.shape[1]
        spks = flow.spk_embed_affine_layer(F.normalize(torch.atleast_2d(ref["embedding"]).to(DEV, torch.float32), dim=1))
        cond = torch.zeros(1, 80, T, device=DEV)
        cond[:, :, :mel1] = prompt_feat.transpose(1, 2)
        mu = h.transpose(1, 2)
        if szum is not None and szum.shape[-1] >= T:
            z = szum[:, :, :T].clone()
        else:
            z = torch.randn(1, 80, T, device=DEV)
        d = self.dtype
        self.g_est.ustaw(T, mu.to(d), spks.to(d), cond.to(d))
        x = z
        for t, r in self._kroki(n_krokow or self.n):
            v = self.g_est(x.to(d), t, r).float()
            if self.cfg > 0:
                v = (1.0 + self.cfg) * v[0:1] - self.cfg * v[1:2]
            x = x + (r - t) * v
        return x[:, :, mel1:].clone()

    @torch.inference_mode()
    def hift(self, mels, poczatek=True):
        """Vocoder: melspektrogram -> dźwięk (1, N) float32 na GPU. Na początku kwestii (poczatek=True)
        wygaszenie jak w oryginale (bez „przecieku” z nagrania referencyjnego)."""
        wav = self.g_hift(mels.float()).clone()
        if poczatek:
            wav[:, :len(self.s3.trim_fade)] *= self.s3.trim_fade
        return wav

    @torch.inference_mode()
    def inference(self, speech_tokens, ref, zwroc_mel=False, n_krokow=None):
        """Jak s3.inference(...)[0]: tokeny -> (1, N) float32 na GPU."""
        mels = self.mel(speech_tokens, ref, n_krokow)
        wav = self.hift(mels)
        return (wav, mels) if zwroc_mel else wav

    def pamiec_mb(self):
        return torch.cuda.memory_allocated() / 1e6
