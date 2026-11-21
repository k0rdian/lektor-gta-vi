"""Strumieniowa synteza S3Gen dla Chatterboxa: tokeny mowy -> dźwięk kawałkami, podczas gdy model
językowy (T3) liczy dalsze tokeny. Dzięki temu głos rusza po pierwszych ~0,5 s wygenerowanej mowy,
a nie dopiero po całym zdaniu.

Opracowane w badaniu opóźnień 20.11 (prototyp: scratchpad/badanie_strumien/strumien_s3.py), tu tylko
ścieżka szybka (SzybkiS3Gen: grafy CUDA, fp16, dekoder standardowy albo meanflow).

Jedna wypowiedź (StrumienS3), kolejne wywołania przetworz():
  - flow (tokeny -> melspektrogram), tryb „prefix”: oddane już ramki mel (i ich tokeny) dopisujemy do
    podpowiedzi (nagrania referencyjnego głosu), więc dekoder liczy tylko ramki nowych tokenów, jako ciąg
    dalszy tego, co już zabrzmiało – tak model był uczony (warunek = początek nagrania). Ostatnie 3 tokeny
    to tylko „wyprzedzenie” enkodera (jak w oryginale przy finalize=False): utrwalamy je w następnym
    kawałku. Szum startowy dekodera jest stały dla wypowiedzi i indeksowany bezwzględną pozycją ramki.
    Tryb „rerun” (do porównań): za każdym razem całe flow od nowa, bierzemy tylko nowe ramki.
  - HiFT (mel -> dźwięk), tryb „kontekst”: vocoder liczy kawałek razem z kilkunastoma ramkami z lewej
    (pamięć splotów), pierwsze C ramek przenika (sin²/cos²) z zatrzymanym ogonem poprzedniego kawałka,
    a ostatnie C ramek zatrzymuje do następnego razu – na złączach nie ma trzasków.
  - znak wodny Perth (jak w oryginale) liczony na [kontekst z lewej | kawałek | ogon] i wycinany, więc
    jest ciągły mimo podziału na kawałki.
  - tempo: liniowa interpolacja melspektrogramu w czasie (ta sama formuła co F.interpolate w zwykłej
    ścieżce), strumieniowo – wysokość głosu się nie zmienia.
"""
from __future__ import annotations

import numpy as np
import torch

SR = 24000
HOP = 480            # próbek audio na ramkę mel (24 kHz / 50 Hz)
TOK = 960            # próbek audio na token mowy (25 Hz)
WYPRZEDZENIE = 3     # tokeny wyprzedzenia enkodera flow (pre_lookahead_len)
TOKEN_MOWY_MAKS = 6561   # tokeny >= to start/stop (drop_invalid_tokens)


def okna_przenikania(n: int):
    """sin² / cos²: suma = 1, stała głośność przy zgodnych sygnałach."""
    if n <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    w_in = (np.sin(0.5 * np.pi * t) ** 2).astype(np.float32)
    return w_in, (1.0 - w_in).astype(np.float32)


def granice(harmonogram, n_max=100000):
    """(pierwszy, kolejne...) -> kolejne granice tokenów, ostatnia wartość się powtarza.
    Np. (16, 30) -> 16, 46, 76, ...; (12, 24, 48) -> 12, 36, 84, 132, ..."""
    s, i = 0, 0
    while s < n_max:
        s += int(harmonogram[min(i, len(harmonogram) - 1)])
        yield s
        i += 1


class StrumienMelTempo:
    """Zmiana tempa w dziedzinie mel dla strumienia: wynik jak F.interpolate(mode="linear",
    align_corners=False) na całości. Klatka wyjściowa k potrzebuje ramek floor(p) i floor(p)+1,
    p = (k + 0,5)·tempo − 0,5 – oddajemy tylko te, dla których obie już są."""

    def __init__(self, tempo: float):
        self.tempo = float(tempo)
        self.buf = None
        self.baza = 0
        self.k = 0
        self.n_we = 0

    def dodaj(self, mel: torch.Tensor, koniec: bool) -> torch.Tensor:
        self.buf = mel if self.buf is None else torch.cat([self.buf, mel], dim=2)
        self.n_we += mel.shape[2]
        if self.n_we == 0:
            return mel[:, :, :0]
        if koniec:
            k_kon = max(1, round(self.n_we / self.tempo))
        else:
            k_kon = int(np.floor((self.n_we - 1 + 0.5) / self.tempo - 0.5)) + 1
            while k_kon > self.k and int(np.floor(max(0.0, (k_kon - 1 + 0.5) * self.tempo - 0.5))) + 1 > self.n_we - 1:
                k_kon -= 1
        if k_kon <= self.k:
            return self.buf[:, :, :0]
        k = torch.arange(self.k, k_kon, device=self.buf.device, dtype=torch.float64)
        p = ((k + 0.5) * self.tempo - 0.5).clamp(min=0)
        i0 = torch.floor(p).long().clamp(max=self.n_we - 1)
        i1 = (i0 + 1).clamp(max=self.n_we - 1)
        w = (p - i0.double()).float()[None, None]
        a = self.buf.index_select(2, i0 - self.baza)
        b = self.buf.index_select(2, i1 - self.baza)
        out = a + (b - a) * w
        self.k = k_kon
        zostaw = int(np.floor(max(0.0, (self.k + 0.5) * self.tempo - 0.5))) - self.baza
        if zostaw > 0 and not koniec:
            self.buf = self.buf[:, :, zostaw:]
            self.baza += zostaw
        return out


class StrumienS3:
    """Jedna wypowiedź: kolejne kawałki tokenów -> kolejne kawałki dźwięku (float32, 24 kHz)."""

    def __init__(self, sz, ref: dict, *, n_krokow: int, nakladka: int = 4, kontekst_hift: int = 12,
                 znak=None, znak_kontekst_s: float = 0.25, tempo: float = 1.0, flow_tryb: str = "prefix",
                 maks_tokenow: int = 1000):
        self.sz = sz
        self.s3 = sz.s3
        self.dev = torch.device("cuda")
        self.ptok = ref["prompt_token"].to(self.dev).long()
        self.pfeat = ref["prompt_feat"].to(self.dev).float()
        self.emb = ref["embedding"].to(self.dev).float()
        self.off = int(self.pfeat.shape[1] - 2 * self.ptok.shape[1])   # 1, gdy referencja nie była cięta co 40 ms
        if self.off not in (0, 1):
            raise ValueError(f"nietypowa podpowiedź głosu ({self.pfeat.shape[1]} ramek, {self.ptok.shape[1]} tokenów)")
        self.n_krokow = int(n_krokow)
        self.C = int(nakladka)
        self.Cv = int(kontekst_hift)
        self.w_in, self.w_out = okna_przenikania(self.C * HOP)
        self.znak = znak
        self.Lw = int(znak_kontekst_s * SR)
        self.tempo = float(tempo)
        self.flow_tryb = flow_tryb
        # stały szum startowy dekodera, indeksowany bezwzględną pozycją ramki (podpowiedź + mowa)
        self.szum = torch.randn(1, 80, self.pfeat.shape[1] + 2 * (maks_tokenow + 2 * WYPRZEDZENIE) + 64,
                                device=self.dev)
        self.fade = self.s3.trim_fade.float().cpu().numpy()
        self.k = 0                     # tyle tokenów ma już utrwalone (oddane do HiFT) ramki mel
        self.mel = None                # (1,80,F) utrwalone ramki mel wygenerowanej mowy
        self.mel_h = None              # historia mel podana do HiFT (po zmianie tempa)
        self.mel_tempo = StrumienMelTempo(self.tempo) if abs(self.tempo - 1.0) > 0.01 else None
        self.ogon = None               # zatrzymane (jeszcze nie oddane) audio do przenikania
        self.E = 0                     # tyle ramek mel_h jest już oddanych jako dźwięk
        self.wyemitowane = 0           # oddane próbki
        self.surowe = np.zeros(0, np.float32)   # ostatnie oddane audio przed znakiem wodnym (kontekst znaku)
        self.czasy = {"flow_ms": 0.0, "hift_ms": 0.0, "znak_ms": 0.0}

    # ------------------------------------------------------------------ flow
    def _flow(self, tokeny: list, do_k: int, koniec: bool) -> torch.Tensor:
        n_wej = len(tokeny) if koniec else do_k + WYPRZEDZENIE
        dev = self.dev
        if self.flow_tryb == "rerun":
            t = torch.tensor([tokeny[:n_wej]], device=dev)
            mel = self.sz.mel(t, {"prompt_token": self.ptok, "prompt_feat": self.pfeat, "embedding": self.emb},
                              n_krokow=self.n_krokow, szum=self.szum, finalize=koniec)
            start = 0 if self.k == 0 else 2 * self.k - self.off
            return mel[:, :, start:]
        if self.k == 0:
            ptok, pfeat = self.ptok, self.pfeat
        else:
            ptok = torch.cat([self.ptok, torch.tensor([tokeny[:self.k]], device=dev)], dim=1)
            pfeat = torch.cat([self.pfeat, self.mel.transpose(1, 2)], dim=1)
        t = torch.tensor([tokeny[self.k:n_wej]], device=dev)
        return self.sz.mel(t, {"prompt_token": ptok, "prompt_feat": pfeat, "embedding": self.emb},
                           n_krokow=self.n_krokow, szum=self.szum, finalize=koniec)

    # ------------------------------------------------------------------ HiFT
    def _hift(self, koniec: bool) -> np.ndarray:
        """HiFT na [E−Cv, G) z historii mel; oddaje [E, G−C) (na końcu – do końca), pierwsze C ramek
        (= poprzedni ogon) przenika z poprzednią wersją."""
        L = self.C * HOP
        G = self.mel_h.shape[2]
        if G <= self.E or (not koniec and G - self.E <= self.C):
            return np.zeros(0, np.float32)          # nic nowego do oddania (ogon czeka na następny kawałek)
        a = max(0, self.E - self.Cv)
        mowa = self.sz.g_hift(self.mel_h[:, :, a:G].float())
        mowa = mowa[0].float().cpu().numpy()[(self.E - a) * HOP:]
        if self.ogon is not None:
            n = min(L, len(mowa), len(self.ogon))
            mowa[:n] = mowa[:n] * self.w_in[:n] + self.ogon[:n] * self.w_out[:n]
        if koniec:
            self.ogon = None
            self.E = G
            return mowa
        self.ogon = mowa[-L:].copy()
        self.E = G - self.C
        return mowa[:-L]

    # ------------------------------------------------------------------ znak wodny
    def _znak_wodny(self, surowe: np.ndarray, koniec: bool) -> np.ndarray:
        if self.znak is None or len(surowe) == 0:
            return surowe
        lewy = self.surowe[-self.Lw:] if self.Lw else np.zeros(0, np.float32)
        prawy = self.ogon if (self.ogon is not None and not koniec) else np.zeros(0, np.float32)
        zapas = np.zeros(HOP, np.float32)          # iSTFT znaku obcina koniec do wielokrotności 10 ms
        seg = np.concatenate([lewy, surowe, prawy, zapas])
        w = np.asarray(self.znak.apply_watermark(seg, sample_rate=SR), dtype=np.float32)
        out = w[len(lewy): len(lewy) + len(surowe)]
        if len(out) < len(surowe):
            out = np.concatenate([out, surowe[len(out):]])
        return out

    # ------------------------------------------------------------------ jeden kawałek
    @torch.inference_mode()
    def przetworz(self, tokeny: list, do_k: int, koniec: bool) -> np.ndarray:
        """Utrwala tokeny [k, do_k) (przy koniec=True – wszystkie) i zwraca dźwięk gotowy do odtwarzania
        (float32, 24 kHz, ze znakiem wodnym; może być pusty). Bez koniec potrzebne są tokeny do
        do_k + 3 (wyprzedzenie)."""
        import time
        t0 = time.perf_counter()
        n = len(tokeny)
        if koniec:
            do_k = n
        elif n < do_k + WYPRZEDZENIE or do_k <= self.k:
            raise ValueError("za mało tokenów na kawałek")
        if do_k > self.k:
            nowy_mel = self._flow(tokeny, do_k, koniec)
        else:
            nowy_mel = torch.zeros(1, 80, 0, device=self.dev)
        t1 = time.perf_counter()
        self.mel = nowy_mel if self.mel is None else torch.cat([self.mel, nowy_mel], dim=2)
        mel_h = nowy_mel
        if koniec:                     # jak w zwykłej ścieżce: najwyżej 2·(n−1) ramek (ostatni token to szum)
            przed = self.mel.shape[2] - nowy_mel.shape[2]
            mel_h = nowy_mel[:, :, : max(0, 2 * (n - 1) - przed)]
        if self.mel_tempo is not None:
            mel_h = self.mel_tempo.dodaj(mel_h, koniec)
        self.mel_h = mel_h if self.mel_h is None else torch.cat([self.mel_h, mel_h], dim=2)
        surowe = self._hift(koniec) if self.mel_h.shape[2] else np.zeros(0, np.float32)
        t2 = time.perf_counter()
        if self.wyemitowane == 0 and len(surowe):       # wygaszenie początku jak w oryginale
            surowe = surowe.copy()
            m = min(len(self.fade), len(surowe))
            surowe[:m] *= self.fade[:m]
        audio = self._znak_wodny(surowe, koniec)
        if len(surowe):
            self.surowe = np.concatenate([self.surowe, surowe])[-max(self.Lw, 1):]
        self.wyemitowane += len(audio)
        self.k = do_k
        t3 = time.perf_counter()
        self.czasy["flow_ms"] += (t1 - t0) * 1000
        self.czasy["hift_ms"] += (t2 - t1) * 1000
        self.czasy["znak_ms"] += (t3 - t2) * 1000
        return audio
