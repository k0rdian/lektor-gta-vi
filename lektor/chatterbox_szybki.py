"""Szybkie dekodowanie T3 Chatterboxa – wersja 2 (badanie opóźnień 20.11; zastąpiła wersję 1).

Pierwsza wersja (kopia: kopia-przed-kolejka-2026-11-20/lektor/chatterbox_szybki_v1_z_przerwij.py) miała
jeden graf kroku w fp32 i próbkowanie w Pythonie: ~5,5 ms na token. Ta: ~2,3 ms na token, pierwszy token
po ~4 ms (zamiast ~18 ms). Względem wersji 1:
  1. Próbkowanie w grafie CUDA: CFG, kara za powtórzenia (maska obecności tokenów), temperatura, min_p,
     softmax, losowanie, osadzenie wylosowanego tokenu i pozycji -> krok = jedno replay(); Python tylko
     odbiera tokeny. Arytmetyka jak w transformers (sprawdzone bitowo), losowanie = algorytm
     torch.multinomial dla 1 próbki (argmax(p / Exp(1)) z tego samego generatora) -> w trybie
     dokladny=True (fp32, te same operacje co stary dekoder) przy tym samym ziarnie wychodzą identyczne
     tokeny jak w starym dekoderze.
  2. Wagi warstw w fp16 (kopia ~1 GB), logity i próbkowanie w fp32. Odchylenie od czystego fp32 rzędu
     TF32, z którego aplikacja korzysta dziś (TV ~0,1% na token).
  3. Kubełki długości KV: osobny graf dla L = 128, 256, ... - uwaga czyta tylko pierwsze L pozycji.
  4. Fuzje: QKV i gate+up jednym GEMM-em, skala uwagi w Wq (0,125 = 2^-3, dokładnie), RoPE przez roll,
     reszta dodawana w GEMM-ie (addmm_), RMSNorm jednym jądrem (F.rms_norm).
  5. Prefill w grafie (dopełnienie do kubełka długości; przy uwadze przyczynowej dopełnienie nie zmienia
     prawdziwych pozycji - sprawdzone: logity i KV bitowo równe wersji eager) + osadzenie warunkowania
     głosu liczone raz na głos/ekspresję.
  6. generuj_strumien(): generator oddający tokeny zaraz po wylosowaniu (pod strumieniowy wokoder).

API zgodne ze starym: SzybkiT3(t3); przygotuj_graf(); generuj(**te same argumenty, przerwij=Event) -> (1, N).
Tylko jedna generacja naraz (wspólne bufory statyczne); porzucenie generatora w połowie jest bezpieczne.
"""
from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F


def _obroc_polowe(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class SzybkiT3:
    def __init__(self, t3, typ=torch.float16, maks_dl: int = 1024,
                 kubelki=(128, 256, 384, 512, 640, 768, 896, 1024),
                 kubelki_prefill=(32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320, 384, 448, 512),
                 dokladny: bool = False):
        self.t3 = t3
        llama = t3.tfmr
        cfg = llama.config
        self.llama = llama
        self.warstwy = llama.layers
        self.N = len(self.warstwy)
        self.H = cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        if cfg.num_key_value_heads != self.H:
            raise RuntimeError("SzybkiT3 obsługuje tylko model bez GQA")
        self.D = cfg.hidden_size
        self.I = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.B = 2                                       # CFG: warunkowa + bezwarunkowa
        self.dev = t3.device
        self.dokladny = dokladny                         # fp32, te same operacje co stary dekoder (testy)
        self.typ = torch.float32 if dokladny else typ
        self.maks_dl = maks_dl
        self.kubelki = [maks_dl] if dokladny else sorted({k for k in kubelki if k <= maks_dl} | {maks_dl})
        self.kubelki_prefill = [] if dokladny else sorted(k for k in kubelki_prefill if k <= maks_dl)
        self.skala = self.hd ** -0.5
        self.hp = t3.hp
        typ = self.typ
        with torch.inference_mode():
            self.w = []
            if not dokladny:
                for L in self.warstwy:
                    a, m = L.self_attn, L.mlp
                    self.w.append({
                        "ln1": L.input_layernorm.weight.detach().to(typ).contiguous(),
                        "ln2": L.post_attention_layernorm.weight.detach().to(typ).contiguous(),
                        "qkv": torch.cat([a.q_proj.weight * self.skala, a.k_proj.weight, a.v_proj.weight], 0).to(typ).contiguous(),
                        "o": a.o_proj.weight.detach().to(typ).contiguous(),
                        "gu": torch.cat([m.gate_proj.weight, m.up_proj.weight], 0).to(typ).contiguous(),
                        "d": m.down_proj.weight.detach().to(typ).contiguous(),
                    })
            self.ln_f = llama.norm.weight.detach().to(typ).contiguous()
            self.glowa = t3.speech_head.weight.detach()               # fp32
            self.emb = t3.speech_emb.weight.detach()                  # fp32
            self.pos = t3.speech_pos_emb.emb.weight.detach()          # fp32
            self.V = self.emb.shape[0]
            kszt = (self.B, self.H, maks_dl, self.hd)
            self.k = [torch.zeros(kszt, device=self.dev, dtype=typ) for _ in range(self.N)]
            self.v = [torch.zeros(kszt, device=self.dev, dtype=typ) for _ in range(self.N)]
            pozycje = torch.arange(maks_dl, device=self.dev)[None]
            if dokladny:     # jak stary dekoder: cos/sin w typie modelu, rotate_half
                cos, sin = llama.rotary_emb(torch.zeros(1, 1, self.D, device=self.dev, dtype=typ), pozycje)
                self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()
            else:            # roll(x, hd/2) * sin ze znakiem == rotate_half(x) * sin
                cos, sin = llama.rotary_emb(torch.zeros(1, 1, self.D, device=self.dev, dtype=torch.float32), pozycje)
                h2 = self.hd // 2
                znak = torch.cat([-torch.ones(h2, device=self.dev), torch.ones(h2, device=self.dev)])
                self.cos = cos[0].to(typ).contiguous()
                self.sin = (sin[0] * znak).to(typ).contiguous()
            self.indeksy = torch.arange(maks_dl, device=self.dev)
            self.zero_t = torch.zeros((), device=self.dev, dtype=typ)
            self.minf_t = torch.full((), float("-inf"), device=self.dev, dtype=typ)
            # stan kroku (bufory statyczne grafów)
            self.tok_buf = torch.zeros(1, 1, device=self.dev, dtype=torch.long)       # ostatni wylosowany token
            self.liczniki = torch.zeros(3, device=self.dev, dtype=torch.long)         # [poz. KV, nr poz. mowy, n]
            self.krok_plus = torch.tensor([1, 1, 0], device=self.dev, dtype=torch.long)
            self.n_plus = torch.tensor([0, 0, 1], device=self.dev, dtype=torch.long)
            self.start_12 = torch.tensor([1, 0], device=self.dev, dtype=torch.long)
            self.obecne = torch.zeros(1, self.V, device=self.dev, dtype=torch.bool)   # tokeny w ids (kara)
            self.wyj = torch.zeros(maks_dl, device=self.dev, dtype=torch.long)
            self.logity_buf = torch.zeros(self.B, self.V, device=self.dev, dtype=torch.float32)
            self.blad = torch.zeros(1, device=self.dev, dtype=torch.bool)             # nie-skończone logity (fp16)
            f32 = dict(device=self.dev, dtype=torch.float32)
            self.cfg_t = torch.zeros((), **f32)
            self.kara_t = torch.ones((), **f32)
            self.odwr_kara_t = torch.ones((), **f32)
            self.odwr_temp_t = torch.ones((), **f32)
            self.min_p_t = torch.zeros((), **f32)
            self.wyj_host = torch.zeros(maks_dl, dtype=torch.long).pin_memory()
            self.blad_host = torch.zeros(maks_dl, dtype=torch.bool).pin_memory()
            if self.kubelki_prefill:
                self.pf_wej = torch.zeros(self.B, self.kubelki_prefill[-1], self.D, device=self.dev, dtype=typ)
                self.pf_ost = torch.zeros(1, device=self.dev, dtype=torch.long)
        self.grafy = {}
        self.graf_probka = None
        self.pula = None
        self.zdarzenia = [torch.cuda.Event() for _ in range(4)]
        self._klucz_param = None
        self._klucz_cond = None

    # ------------------------------------------------------------------ sieć (ścieżka szybka)
    def _norma(self, x, w):
        return F.rms_norm(x, (self.D,), w, self.eps)

    def _rope(self, x, cos, sin):
        return x * cos + torch.roll(x, self.hd // 2, -1) * sin

    def _warstwa_krok(self, i, h, cos, sin, bias, L):
        """Jeden token; h (B, D) modyfikowane w miejscu."""
        W = self.w[i]
        B, H, hd = self.B, self.H, self.hd
        x = self._norma(h, W["ln1"])
        qkv = F.linear(x, W["qkv"]).view(B, 3, H, hd)
        qk = self._rope(qkv[:, :2], cos, sin)
        q, k, v = qk[:, 0].unsqueeze(2), qk[:, 1].unsqueeze(2), qkv[:, 2].unsqueeze(2)
        poz = self.liczniki[0:1]
        self.k[i].index_copy_(2, poz, k)
        self.v[i].index_copy_(2, poz, v)
        s = torch.matmul(q, self.k[i][:, :, :L].transpose(-1, -2))       # (B, H, 1, L); skala jest w Wq
        p = torch.softmax(s + bias, dim=-1)
        o = torch.matmul(p, self.v[i][:, :, :L]).view(B, self.D)
        h.addmm_(o, W["o"].t())
        x = self._norma(h, W["ln2"])
        gu = F.linear(x, W["gu"])
        a = F.silu(gu[:, : self.I])
        a.mul_(gu[:, self.I:])
        h.addmm_(a, W["d"].t())
        return h

    def _rdzen_prefill(self, h, T):
        """h (B*T, D) -> po wszystkich warstwach (w miejscu); zapisuje KV [0, T)."""
        B, H, hd = self.B, self.H, self.hd
        cos = self.cos[:T][None, None]
        sin = self.sin[:T][None, None]
        for i in range(self.N):
            W = self.w[i]
            x = self._norma(h, W["ln1"])
            qkv = F.linear(x, W["qkv"]).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)    # (3, B, H, T, hd)
            q = self._rope(qkv[0], cos, sin)
            k = self._rope(qkv[1], cos, sin)
            v = qkv[2]
            self.k[i][:, :, :T].copy_(k)
            self.v[i][:, :, :T].copy_(v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
            h.addmm_(o.transpose(1, 2).reshape(B * T, self.D), W["o"].t())
            x = self._norma(h, W["ln2"])
            gu = F.linear(x, W["gu"])
            a = F.silu(gu[:, : self.I])
            a.mul_(gu[:, self.I:])
            h.addmm_(a, W["d"].t())
        return h

    def _logity(self, h):
        return F.linear(self._norma(h, self.ln_f).float(), self.glowa)        # (B, V) fp32

    # ------------------------------------------------------------------ sieć (tryb dokładny = stary dekoder)
    def _warstwa_dokladna(self, i, h, cos, sin, zapisz, uwaga):
        L = self.warstwy[i]
        a = L.self_attn
        B, T, _ = h.shape
        x = L.input_layernorm(h)
        q = a.q_proj(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = a.k_proj(x).view(B, T, self.H, self.hd).transpose(1, 2)
        v = a.v_proj(x).view(B, T, self.H, self.hd).transpose(1, 2)
        q = q * cos + _obroc_polowe(q) * sin
        k = k * cos + _obroc_polowe(k) * sin
        K, V = zapisz(i, k, v)
        o = uwaga(q, K, V).transpose(1, 2).reshape(B, T, self.H * self.hd)
        h = h + a.o_proj(o)
        return h + L.mlp(L.post_attention_layernorm(h))

    def _prefill_dokladny(self, embeds):
        T0 = embeds.shape[1]
        cos, sin = self.cos[:T0][None, None], self.sin[:T0][None, None]

        def zapisz(i, k, v):
            self.k[i][:, :, :T0].copy_(k)
            self.v[i][:, :, :T0].copy_(v)
            return k, v

        def uwaga(q, K, V):
            return F.scaled_dot_product_attention(q, K, V, is_causal=True, scale=self.skala)
        h = embeds
        for i in range(self.N):
            h = self._warstwa_dokladna(i, h, cos, sin, zapisz, uwaga)
        return self.t3.speech_head(self.llama.norm(h[:, -1:]))[:, -1, :].float()

    # ------------------------------------------------------------------ próbkowanie (w grafie)
    def _probkuj(self, logity):
        """Jak pętla T3.inference: CFG, RepetitionPenalty, /temperatura, MinP, (TopP=1 nic nie robi), multinomial.
        Na CUDA x/c dla skalara c liczy się jako x*(1/c) - stąd odwrotności liczone w fp32."""
        war, bezw = logity[0:1, :], logity[1:2, :]
        l = war + self.cfg_t * (war - bezw)
        self.blad.logical_or_(~torch.isfinite(l).all().view(1))
        kara = torch.where(l < 0, l * self.kara_t, l * self.odwr_kara_t)
        l = torch.where(self.obecne, kara, l)
        l = l * self.odwr_temp_t
        p = torch.softmax(l, dim=-1)
        l = l.masked_fill(p < p.amax(dim=-1, keepdim=True) * self.min_p_t, float("-inf"))
        p = torch.softmax(l, dim=-1)
        nast = torch.argmax(p / torch.empty_like(p).exponential_(1), dim=-1, keepdim=True)   # == multinomial(p, 1)
        self.tok_buf.copy_(nast)
        self.obecne.index_fill_(1, nast.view(1), True)
        self.wyj.index_copy_(0, self.liczniki[2:3], nast.view(1))
        self.liczniki.add_(self.n_plus)

    # ------------------------------------------------------------------ krok i prefill w grafie
    def _krok(self, L: int):
        B = self.B
        poz = self.liczniki[0:1]
        x = F.embedding(self.tok_buf.view(1), self.emb) + F.embedding(self.liczniki[1:2], self.pos)   # (1, D) fp32
        if self.dokladny:
            h = x.view(1, 1, self.D).expand(B, 1, self.D).contiguous()
            cos = self.cos.index_select(0, poz)[None, None]
            sin = self.sin.index_select(0, poz)[None, None]
            maska = (self.indeksy <= poz).view(1, 1, 1, self.maks_dl)

            def zapisz(i, k, v):
                self.k[i].index_copy_(2, poz, k)
                self.v[i].index_copy_(2, poz, v)
                return self.k[i], self.v[i]

            def uwaga(q, K, V):
                w = torch.matmul(q, K.transpose(-1, -2)) * self.skala
                return torch.matmul(torch.softmax(w.masked_fill(~maska, float("-inf")), dim=-1), V)
            for i in range(self.N):
                h = self._warstwa_dokladna(i, h, cos, sin, zapisz, uwaga)
            logity = self.t3.speech_head(self.llama.norm(h))[:, -1, :].float()
        else:
            h = x.to(self.typ).expand(B, self.D).contiguous()
            cos = self.cos.index_select(0, poz)
            sin = self.sin.index_select(0, poz)
            bias = torch.where(self.indeksy[:L] <= poz, self.zero_t, self.minf_t)
            for i in range(self.N):
                h = self._warstwa_krok(i, h, cos, sin, bias, L)
            logity = self._logity(h)
        if self._nagrywam_wymuszone:
            self.logity_buf.copy_(logity)
        else:
            self._probkuj(logity)
        self.liczniki.add_(self.krok_plus)

    def _prefill_graf(self, Tb):
        B = self.B
        h = self.pf_wej[:, :Tb].reshape(B * Tb, self.D).clone()
        h = self._rdzen_prefill(h, Tb)
        self.logity_buf.copy_(self._logity(h.view(B, Tb, self.D).index_select(1, self.pf_ost).view(B, self.D)))

    def _nagraj(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        zapas = [t.clone() for t in (self.liczniki, self.tok_buf, self.obecne, self.wyj, self.blad)]
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        if self.pula is None:
            self.pula = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self.pula):     # grafy nigdy nie biegną równolegle - wspólna pula
            fn()
        torch.cuda.synchronize()
        for t, z in zip((self.liczniki, self.tok_buf, self.obecne, self.wyj, self.blad), zapas):
            t.copy_(z)
        return g

    _nagrywam_wymuszone = False

    def przygotuj_graf(self, wymuszone: bool = False):
        """Nagrywa grafy: próbkowanie, krok dla każdego kubełka KV, prefill dla każdego kubełka długości.
        wymuszone=True: dodatkowo grafy kroku bez próbkowania (logity do testów zgodności)."""
        with torch.inference_mode():
            self.liczniki.copy_(torch.tensor([0, 1, 0], device=self.dev))
            if self.graf_probka is None:
                self.graf_probka = self._nagraj(lambda: self._probkuj(self.logity_buf))
            for L in self.kubelki:
                if L not in self.grafy:
                    self.grafy[L] = self._nagraj(lambda L=L: self._krok(L))
            for Tb in self.kubelki_prefill:
                if ("pf", Tb) not in self.grafy:
                    self.grafy[("pf", Tb)] = self._nagraj(lambda Tb=Tb: self._prefill_graf(Tb))
            if wymuszone:
                self._nagrywam_wymuszone = True
                try:
                    for L in self.kubelki:
                        if ("w", L) not in self.grafy:
                            self.grafy[("w", L)] = self._nagraj(lambda L=L: self._krok(L))
                finally:
                    self._nagrywam_wymuszone = False
            for t in self.k + self.v:
                t.zero_()
            torch.cuda.synchronize()

    def _kubelek(self, poz: int) -> int:
        for L in self.kubelki:
            if poz < L:
                return L
        raise ValueError("poza buforem KV")

    def _uruchom_prefill(self, wejscie):
        """Logity pierwszego kroku -> logity_buf (graf z dopełnieniem, a dla bardzo długich tekstów eager)."""
        T0 = wejscie.shape[1]
        if self.kubelki_prefill and T0 <= self.kubelki_prefill[-1]:
            Tb = next(k for k in self.kubelki_prefill if k >= T0)
            self.pf_wej[:, :T0].copy_(wejscie)
            self.pf_ost.fill_(T0 - 1)
            self.grafy[("pf", Tb)].replay()
        elif self.dokladny:
            self.logity_buf.copy_(self._prefill_dokladny(wejscie))
        else:
            B = self.B
            h = self._rdzen_prefill(wejscie.to(self.typ).reshape(B * T0, self.D).contiguous(), T0)
            self.logity_buf.copy_(self._logity(h.view(B, T0, self.D)[:, -1]))

    # ------------------------------------------------------------------ wejście
    def _wejscie(self, t3_cond, text_tokens, cfg_weight):
        """[warunkowanie, tekst, BOS, BOS] jak w T3.inference (bitowo); warunkowanie liczone raz na głos."""
        t3 = self.t3
        hp = t3.hp
        # słaba referencja do obiektu T3Cond (id() mógłby zostać użyty ponownie po zwolnieniu obiektu)
        emocja = float(t3_cond.emotion_adv.view(-1)[0])
        if self._klucz_cond is None or self._klucz_cond[0]() is not t3_cond or self._klucz_cond[1] != emocja:
            self._cond = t3.prepare_conditioning(t3_cond)                             # (1, Lc, D)
            bos = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=self.dev)
            b = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
            self._bos2 = torch.cat([b, b], dim=1).expand(2, -1, -1)                 # (2, 2, D)
            self._klucz_cond = (weakref.ref(t3_cond), emocja)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.dev)
        te = t3.text_emb(text_tokens)
        if cfg_weight > 0.0:
            te[1].zero_()
        te = te + t3.text_pos_emb(text_tokens)
        return torch.cat([self._cond.expand(te.size(0), -1, -1), te, self._bos2], dim=1)

    def _ustaw_probkowanie(self, temperature, cfg_weight, repetition_penalty, min_p, top_p):
        if top_p < 1.0:
            raise NotImplementedError("top_p < 1 nie jest obsługiwane przez graf")   # aplikacja używa 1.0
        klucz = (float(temperature), float(cfg_weight), float(repetition_penalty), float(min_p))
        if self._klucz_param == klucz:
            return
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)                     # noqa: E731
        self.cfg_t.fill_(float(cfg_weight))
        self.kara_t.fill_(float(repetition_penalty))
        self.odwr_kara_t.fill_(float((f32(1.0) / f32(float(repetition_penalty))).item()))
        self.odwr_temp_t.fill_(float((f32(1.0) / f32(float(temperature))).item()))
        self.min_p_t.fill_(float(min_p))
        self._klucz_param = klucz

    # ------------------------------------------------------------------ API
    @torch.inference_mode()
    def generuj_strumien(self, *, t3_cond, text_tokens: torch.Tensor, max_new_tokens: int, temperature: float,
                         cfg_weight: float, repetition_penalty: float, min_p: float, top_p: float,
                         wyprzedzenie: int = 1, przerwij=None):
        """Oddaje tokeny mowy (int) zaraz po wylosowaniu; semantyka jak T3.inference: najwyżej
        max_new_tokens tokenów, EOS (jeśli wylosowany) oddawany jako ostatni.
        wyprzedzenie=1: GPU ma zawsze kolejny krok w kolejce (brak przestojów między krokami); po EOS
        jeden zbędny krok (~2-3 ms GPU) dokończy się w tle.
        przerwij: threading.Event – sprawdzany przed każdym tokenem (lektor porzucił kwestię)."""
        hp = self.hp
        wejscie = self._wejscie(t3_cond, text_tokens, cfg_weight)
        T0 = wejscie.shape[1]
        max_new_tokens = min(max_new_tokens, self.maks_dl - T0)
        if max_new_tokens < 8:
            raise ValueError("tekst za długi dla szybkiej ścieżki")
        self._ustaw_probkowanie(temperature, cfg_weight, repetition_penalty, min_p, top_p)
        self.liczniki[0:1].fill_(T0)
        self.liczniki[1:3].copy_(self.start_12)
        self.obecne.zero_()
        self.obecne[0, hp.start_speech_token] = True        # ids zaczynają się od BOS (też karany)
        self.blad.zero_()
        self._uruchom_prefill(wejscie)
        self.graf_probka.replay()                            # token 0
        stop = hp.stop_speech_token
        ev = self.zdarzenia
        nz = len(ev)
        wyprzedzenie = max(0, min(int(wyprzedzenie), nz - 1))

        def zlec(j):
            if j > 0:            # krok tworzący token j przetwarza token j-1 na pozycji KV T0+j-1
                self.grafy[self._kubelek(T0 + j - 1)].replay()
            self.wyj_host[j:j + 1].copy_(self.tok_buf.view(1), non_blocking=True)
            self.blad_host[j:j + 1].copy_(self.blad, non_blocking=True)
            ev[j % nz].record()

        zlec(0)
        zlecone = 1
        for j in range(max_new_tokens):
            if przerwij is not None and przerwij.is_set():
                return
            while zlecone < max_new_tokens and zlecone <= j + wyprzedzenie:
                zlec(zlecone)
                zlecone += 1
            ev[j % nz].synchronize()
            if bool(self.blad_host[j]):
                raise FloatingPointError("nie-skończone logity (fp16) - użyj ścieżki fp32")
            t = int(self.wyj_host[j])
            yield t
            if t == stop:
                break

    def generuj(self, *, wyprzedzenie: int = 1, przerwij=None, **kw) -> torch.Tensor:
        """Jak stary SzybkiT3.generuj: (1, N) tokenów mowy (N = 0, gdy przerwano przed pierwszym)."""
        tok = list(self.generuj_strumien(wyprzedzenie=wyprzedzenie, przerwij=przerwij, **kw))
        return torch.tensor(tok, dtype=torch.long, device=self.dev).view(1, -1)

    @torch.inference_mode()
    def logity_wymuszone(self, *, t3_cond, text_tokens, cfg_weight, tokeny):
        """Logity (n, 2, V) przed każdym tokenem wymuszonej sekwencji (test zgodności; wymaga
        przygotuj_graf(wymuszone=True))."""
        wejscie = self._wejscie(t3_cond, text_tokens, cfg_weight)
        T0 = wejscie.shape[1]
        self.liczniki.copy_(torch.tensor([T0, 1, 0], device=self.dev))
        self._uruchom_prefill(wejscie)
        wyn = [self.logity_buf.clone()]
        for j, t in enumerate(tokeny[:-1].tolist()):
            self.tok_buf.fill_(int(t))
            self.grafy[("w", self._kubelek(T0 + j))].replay()
            wyn.append(self.logity_buf.clone())
        return torch.stack(wyn)
