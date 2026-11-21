"""Proces roboczy głosu Chatterbox (uruchamiany w .venv-chatterbox, osobno od aplikacji).

Aplikacja główna (.venv) nie ładuje PyTorcha – startuje ten skrypt jako podproces i rozmawia
z nim przez stdin/stdout:
  - żądanie:   jedna linia JSON na stdin, np. {"cmd": "syntezuj", "tekst": "..."}
  - odpowiedź: jedna linia JSON na stdout (np. {"ok": true, "sr": 24000, "n_bajtow": 96000}),
               a po niej – jeśli n_bajtow > 0 – surowe audio PCM int16 mono.
Wszystko, co biblioteki wypisują same z siebie, trafia na stderr (stdout jest zarezerwowany
na protokół). Proces kończy się, gdy aplikacja zamknie stdin (np. po zamknięciu okna).

Ręczne użycie (diagnostyka):
  .venv-chatterbox\\Scripts\\python.exe lektor\\chatterbox_worker.py --pobierz --test
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

BAZA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BAZA))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # modele są lokalnie – żadnych zapytań do sieci
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from lektor.chatterbox_pliki import KATALOG_MODELI, SR, brakujace_pliki, pobierz_modele  # noqa: E402


def info(msg: str):
    """Komunikat dla aplikacji (trafia do jej dziennika)."""
    sys.stderr.write("LEKTOR: " + msg.replace("\n", " ") + "\n")
    sys.stderr.flush()


def _pobierz_meanflow():
    """Dekoder meanflow z oficjalnego repozytorium Chatterbox-Turbo (ok. 1 GB, raz)."""
    import urllib.request
    PLIK_MEANFLOW.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLIK_MEANFLOW.with_suffix(".czesc")
    info("Pobieram szybki dekoder Chatterbox-Turbo (ok. 1 GB, jednorazowo)…")
    req = urllib.request.Request(URL_MEANFLOW, headers={"User-Agent": "lektor-gta"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            blok = r.read(1 << 22)
            if not blok:
                break
            f.write(blok)
    os.replace(tmp, PLIK_MEANFLOW)


# ---------------------------------------------------------------- model

# Tryby „Szybkość / jakość” (ustawienie cb_tryb). Dekoder: "std" = oryginalny s3gen.pt (CFM z CFG),
# "meanflow" = zdestylowany dekoder z Chatterbox-Turbo (2 kroki, bez CFG – 13× szybszy, zgodny z tokenami
# wielojęzycznymi). Strumieniowanie: dźwięk rusza po `pierwszy` tokenach mowy (25 tokenów = 1 s), dalej
# kawałkami po `kawalek` tokenów, a model językowy w tym czasie liczy dalej.
TRYBY = {
    "najszybszy":   {"dekoder": "meanflow", "kroki": 2, "pierwszy": 10, "kawalek": 20},
    "zrownowazony": {"dekoder": "meanflow", "kroki": 2, "pierwszy": 16, "kawalek": 30},
    "jakosc":       {"dekoder": "std", "kroki": 10, "pierwszy": 30, "kawalek": 50},
}
PLIK_MEANFLOW = KATALOG_MODELI / "turbo" / "s3gen_meanflow.safetensors"
URL_MEANFLOW = "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen_meanflow.safetensors"


class Silnik:
    def __init__(self):
        self.model = None
        self.wersja = None
        self.urzadzenie = None
        self.conds_domyslne = None
        self._cache_conds: dict = {}
        self.aktywny_glos = None
        self.szybki = None             # SzybkiT3 (graf CUDA) albo None = oryginalna pętla
        self.s3_szybkie: dict = {}     # dekoder ("std"/"meanflow") -> SzybkiS3Gen (grafy CUDA) albo None
        self._mf_s3 = None             # model S3Gen z dekoderem meanflow (ładowany przy pierwszym użyciu)

    # --- szybki S3Gen (grafy CUDA) ---
    def _s3gen_szybki(self, dekoder: str):
        """SzybkiS3Gen dla danego dekodera (tworzony przy pierwszym użyciu); None = oryginalna ścieżka."""
        if self.urzadzenie != "cuda" or os.environ.get("LEKTOR_BEZ_GRAFU") == "1":
            return None
        if dekoder in self.s3_szybkie:
            return self.s3_szybkie[dekoder]
        sz = None
        try:
            import gc
            import torch
            from lektor.chatterbox_s3gen_szybki import SzybkiS3Gen
            # tylko jeden szybki vocoder naraz: każdy to kopia wag + grafy CUDA dla wszystkich długości
            # (razem z drugim dekoderem pamięć karty potrafiła się skończyć – wtedy Windows przenosi ją
            # do RAM-u i wszystko zwalnia kilkudziesięciokrotnie). Powrót do trybu = ponowne nagranie grafów.
            for d in [d for d in self.s3_szybkie if d != dekoder]:
                del self.s3_szybkie[d]
            if dekoder != "meanflow":
                self._mf_s3 = None
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            if dekoder == "meanflow":
                if not PLIK_MEANFLOW.exists():
                    _pobierz_meanflow()
                from chatterbox.models.s3gen import S3Gen
                from safetensors.torch import load_file
                mf = S3Gen(meanflow=True)
                mf.load_state_dict(load_file(str(PLIK_MEANFLOW)), strict=True)
                self._mf_s3 = mf.to("cuda").eval()
                sz = SzybkiS3Gen(self._mf_s3, n_krokow=2, dtype=torch.float16)
            else:
                sz = SzybkiS3Gen(self.model.s3gen, dtype=torch.float16)
            if self.model.conds is not None:
                sz.rozgrzej(self.model.conds.gen, maks_tokenow=400)
            info(f"Szybki vocoder ({'meanflow' if dekoder == 'meanflow' else 'standardowy'}, grafy CUDA) "
                 f"gotowy w {time.perf_counter() - t0:.1f} s.")
        except Exception as e:
            sz = None
            info(f"Szybki vocoder {dekoder} niedostępny ({type(e).__name__}: {e}) – używam zwykłej ścieżki.")
        self.s3_szybkie[dekoder] = sz
        return sz

    def _wlacz_szybki(self):
        """Szybki dekoder T3 (graf CUDA); przy jakimkolwiek problemie zostaje oryginalna pętla."""
        if self.urzadzenie != "cuda":
            return
        try:
            from lektor.chatterbox_szybki import SzybkiT3
            t0 = time.perf_counter()
            sz = SzybkiT3(self.model.t3)
            sz.przygotuj_graf()
            self.szybki = sz
            info(f"Szybki dekoder (graf CUDA) gotowy w {time.perf_counter() - t0:.1f} s.")
        except Exception as e:
            self.szybki = None
            info(f"Szybki dekoder niedostępny ({type(e).__name__}: {e}) – używam zwykłej pętli.")

    def zaladuj(self, wersja: str = "v3", urzadzenie: str = "auto", szybki: bool = True,
                tryb: str | None = None) -> dict:
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.WARNING)
        t0 = time.perf_counter()
        import torch
        from chatterbox.models.tokenizers import tokenizer as tok
        from chatterbox.models.t3 import t3 as t3mod
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        # tylko polski: bez chińskiej segmentacji (która przy starcie łączy się z siecią)
        tok.ChineseCangjieConverter._load_cangjie_mapping = lambda self, model_dir=None: None
        tok.ChineseCangjieConverter._init_segmenter = lambda self: None
        t3mod.tqdm = lambda it, **k: it          # bez paska postępu przy każdej kwestii

        if urzadzenie in ("auto", "", None):
            urzadzenie = "cuda" if torch.cuda.is_available() else "cpu"
        if urzadzenie == "cuda" and not torch.cuda.is_available():
            info("CUDA niedostępna – używam procesora (będzie wolno).")
            urzadzenie = "cpu"
        if urzadzenie == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        brak = brakujace_pliki(wersja)
        if brak:
            raise FileNotFoundError(f"Brak plików modelu w modele/chatterbox: {', '.join(brak)}")
        self.model = ChatterboxMultilingualTTS.from_local(str(KATALOG_MODELI), urzadzenie, t3_model=wersja)
        self.conds_domyslne = self.model.conds
        self.wersja, self.urzadzenie = wersja, urzadzenie
        self.szybki = None
        if szybki and os.environ.get("LEKTOR_BEZ_GRAFU") != "1":
            self._wlacz_szybki()
        t_ladowania = time.perf_counter() - t0
        # rozgrzewka: pierwsza synteza ładuje jądra CUDA (kilkanaście sekund) – lepiej teraz niż w grze
        # (strumieniowo, czyli tą samą drogą co w grze: grafy T3, vocodera i znak wodny)
        t1 = time.perf_counter()
        self.syntezuj_strumien("Dzień dobry, sprawdzam głos. Wszystko działa?", "pl", 0.5, 0.5, 0.8, tryb=tryb,
                               wyslij=lambda audio, info: None)
        t_rozgrzewki = time.perf_counter() - t1
        nazwa_gpu = torch.cuda.get_device_name(0) if urzadzenie == "cuda" else "procesor"
        info(f"Chatterbox {wersja} gotowy na {nazwa_gpu}: ładowanie {t_ladowania:.1f} s, rozgrzewka {t_rozgrzewki:.1f} s.")
        return {"urzadzenie": urzadzenie, "gpu": nazwa_gpu, "sr": SR,
                "ladowanie_s": round(t_ladowania, 1), "rozgrzewka_s": round(t_rozgrzewki, 1)}

    def ustaw_glos(self, conds: str | None) -> dict:
        from chatterbox.mtl_tts import Conditionals
        if conds is None:
            self.model.conds = self.conds_domyslne
            self.aktywny_glos = None
            return {"glos": "wbudowany"}
        if conds not in self._cache_conds:
            mapa = "cpu" if self.urzadzenie == "cpu" else None
            self._cache_conds[conds] = Conditionals.load(conds, map_location=mapa or "cpu").to(self.urzadzenie)
        self.model.conds = self._cache_conds[conds]
        self.aktywny_glos = conds
        # grafy vocodera zależą od długości nagrania referencyjnego – dograj brakujące kubełki teraz
        for sz in self.s3_szybkie.values():
            if sz is not None:
                try:
                    sz.rozgrzej(self.model.conds.gen, maks_tokenow=400)
                except Exception as e:
                    info(f"Grafy vocodera dla nowego głosu: {e}")
        return {"glos": conds}

    def _ustaw_ekspresje(self, ekspresja: float):
        import torch
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        m = self.model
        if m.conds is None:
            raise RuntimeError("Brak głosu (conds)")
        if float(ekspresja) != float(m.conds.t3.emotion_adv[0, 0, 0].item()):
            c = m.conds.t3
            m.conds.t3 = T3Cond(speaker_emb=c.speaker_emb, cond_prompt_speech_tokens=c.cond_prompt_speech_tokens,
                                emotion_adv=float(ekspresja) * torch.ones(1, 1, 1)).to(device=m.device)

    def _tokeny_tekstu(self, tekst: str, jezyk: str):
        """Tekst -> tokeny dla T3 (2 kopie na potrzeby CFG) i limit tokenów mowy."""
        import torch
        import torch.nn.functional as F
        from chatterbox.mtl_tts import punc_norm
        m = self.model
        tekst = punc_norm(tekst)
        # ok. 25 tokenów na sekundę mowy; polski to ok. 12-15 znaków/s – zapas na wolne tempo
        max_tokenow = int(min(1000, max(120, len(tekst) * 25 / 7 + 60)))
        tokeny = m.tokenizer.text_to_tokens(tekst, language_id=jezyk).to(m.device)
        tokeny = torch.cat([tokeny, tokeny], dim=0)
        tokeny = F.pad(tokeny, (1, 0), value=m.t3.hp.start_text_token)
        tokeny = F.pad(tokeny, (0, 1), value=m.t3.hp.stop_text_token)
        return tokeny, max_tokenow

    def _generuj_mowe(self, tokeny, max_tokenow: int, cfg: float, temperatura: float, przerwij=None):
        """Tokeny mowy T3 (cała wypowiedź)."""
        from chatterbox.models.s3tokenizer import drop_invalid_tokens
        m = self.model
        argumenty = dict(t3_cond=m.conds.t3, text_tokens=tokeny, max_new_tokens=max_tokenow,
                         temperature=float(temperatura), cfg_weight=float(cfg),
                         repetition_penalty=1.2, min_p=0.05, top_p=1.0)
        mowa = None
        if self.szybki is not None:
            try:
                mowa = self.szybki.generuj(**argumenty, przerwij=przerwij)
            except (ValueError, FloatingPointError, NotImplementedError) as e:
                # bardzo długi tekst / nie-skończone logity fp16 / nieobsługiwane próbkowanie – zwykła pętla
                info(f"Szybki dekoder: {type(e).__name__}: {e} – ta kwestia zwykłą pętlą.")
                mowa = None
        if mowa is None:
            mowa = m.t3.inference(**argumenty)
        return drop_invalid_tokens(mowa[0]).to(m.device)

    def _mel(self, mowa, tryb: dict, szum=None):
        """Tokeny mowy -> melspektrogram (szybka ścieżka z grafami albo oryginalna)."""
        sz = self._s3gen_szybki(tryb["dekoder"])
        if sz is not None:
            try:
                return sz.mel(mowa, self.model.conds.gen, n_krokow=tryb["kroki"], szum=szum), sz
            except Exception as e:
                info(f"Szybki vocoder zawiódł ({type(e).__name__}: {e}) – wracam do zwykłej ścieżki.")
                self.s3_szybkie[tryb["dekoder"]] = None
        return self.model.s3gen.flow_inference(mowa, ref_dict=self.model.conds.gen, finalize=True), None

    def _vocoder(self, mel, sz, tempo: float = 1.0, poczatek: bool = True):
        """Melspektrogram -> dźwięk float32 (numpy). Tempo zmieniamy, skalując melspektrogram w czasie:
        vocoder bierze wysokość głosu z melspektrogramu, więc ton się nie zmienia (lepiej niż WSOLA)."""
        import numpy as np
        import torch.nn.functional as F
        if abs(float(tempo) - 1.0) > 0.01:
            mel = F.interpolate(mel.float(), size=max(2, round(mel.shape[-1] / float(tempo))), mode="linear",
                                align_corners=False)
        if sz is not None:
            wav = sz.hift(mel, poczatek=poczatek)
        else:
            s3 = self.model.s3gen
            wav, _ = s3.hift_inference(mel.to(dtype=s3.dtype), None)
            if poczatek:
                wav[:, :len(s3.trim_fade)] *= s3.trim_fade
        return np.asarray(wav.squeeze(0).float().cpu().numpy(), dtype=np.float32)

    def syntezuj(self, tekst: str, jezyk: str = "pl", ekspresja: float = 0.5, cfg: float = 0.5,
                 temperatura: float = 0.8, max_tokenow: int | None = None, tempo: float = 1.0, przerwij=None,
                 tryb: str | None = None):
        """Jak ChatterboxMultilingualTTS.generate, ale z limitem tokenów zależnym od długości
        tekstu (model, który się „zapętli”, nie produkuje 40 s bełkotu), tempem, przerywaniem
        i szybkim vocoderem (grafy CUDA, dekoder wg trybu)."""
        import numpy as np
        import torch

        t_tryb = TRYBY.get(tryb or "zrownowazony", TRYBY["zrownowazony"])
        self._ustaw_ekspresje(ekspresja)
        tokeny, limit = self._tokeny_tekstu(tekst, jezyk)
        max_tokenow = max_tokenow or limit
        m = self.model
        with torch.inference_mode():
            t0 = time.perf_counter()
            mowa = self._generuj_mowe(tokeny, max_tokenow, cfg, temperatura, przerwij)
            t_t3 = time.perf_counter() - t0
            n = int(mowa.shape[-1])
            if n < 2 or (przerwij is not None and przerwij.is_set()):
                return np.zeros(0, dtype=np.int16), {"t3_ms": round(t_t3 * 1000), "s3_ms": 0, "tokeny": n}
            t1 = time.perf_counter()
            mel, sz = self._mel(mowa, t_tryb)
            mel = mel[..., : 2 * (n - 1)]                     # ostatni token to ~40 ms szumu
            wav = self._vocoder(mel, sz, tempo, poczatek=True)
            t_s3 = time.perf_counter() - t1
            wav = m.watermarker.apply_watermark(wav, sample_rate=m.sr)
        audio = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        return (audio * 32767).astype(np.int16), {"t3_ms": round(t_t3 * 1000), "s3_ms": round(t_s3 * 1000),
                                                   "tokeny": n, "limit": max_tokenow}

    def syntezuj_strumien(self, tekst: str, jezyk: str, ekspresja: float, cfg: float, temperatura: float,
                          tempo: float = 1.0, tryb=None, wyslij=None, przerwij=None) -> dict:
        """Synteza kwestii strumieniowo: zdanie po zdaniu, a w zdaniu kawałek po kawałku – dźwięk rusza po
        pierwszych tokenach mowy (TRYBY[...]["pierwszy"]), a model językowy w tym czasie liczy dalej.
        Każdy gotowy kawałek od razu idzie do aplikacji: wyslij(audio_int16, info), gdzie
        info["koniec_zdania"] mówi, czy to ostatni kawałek zdania (tam aplikacja skraca ciszę)."""
        from lektor.glos import podziel_na_zdania
        t_tryb = TRYBY.get(tryb or "zrownowazony", TRYBY["zrownowazony"])
        czasy = {"t3_ms": 0, "s3_ms": 0, "tokeny": 0, "pierwszy_ms": None, "kawalkow": 0}
        t0 = time.perf_counter()
        for zdanie in podziel_na_zdania(tekst):
            if przerwij is not None and przerwij.is_set():
                break
            self._zdanie_strumien(zdanie, jezyk, ekspresja, cfg, temperatura, tempo, tryb, t_tryb, wyslij, przerwij,
                                  czasy, t0)
        czasy["t3_ms"], czasy["s3_ms"] = round(czasy["t3_ms"]), round(czasy["s3_ms"])
        return czasy

    def _zdanie_strumien(self, zdanie, jezyk, ekspresja, cfg, temperatura, tempo, tryb, t_tryb, wyslij, przerwij,
                         czasy, t0):
        import numpy as np
        import torch
        from lektor.chatterbox_strumien import TOKEN_MOWY_MAKS, WYPRZEDZENIE, StrumienS3, granice

        def przerwano():
            return przerwij is not None and przerwij.is_set()

        def nadaj(audio, koniec_zdania: bool, **dod):
            if przerwano():
                return
            if audio.dtype != np.int16:
                audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            if len(audio) == 0 and not koniec_zdania:
                return
            if czasy["pierwszy_ms"] is None and len(audio):
                czasy["pierwszy_ms"] = round((time.perf_counter() - t0) * 1000)
            czasy["kawalkow"] += 1
            wyslij(audio, {"koniec_zdania": koniec_zdania, **dod})

        def calym_zdaniem():
            """Bez strumienia (brak szybkich ścieżek albo awaria w trakcie): całe zdanie naraz."""
            audio, cz = self.syntezuj(zdanie, jezyk, ekspresja, cfg, temperatura, tempo=tempo, przerwij=przerwij,
                                      tryb=tryb)
            for k in ("t3_ms", "s3_ms", "tokeny"):
                czasy[k] += cz.get(k, 0)
            nadaj(audio, True, t3_ms=cz.get("t3_ms"), s3_ms=cz.get("s3_ms"), tokeny=cz.get("tokeny"))

        sz = self._s3gen_szybki(t_tryb["dekoder"]) if self.szybki is not None else None
        if sz is None:
            calym_zdaniem()
            return
        m = self.model
        self._ustaw_ekspresje(ekspresja)
        tokeny, limit = self._tokeny_tekstu(zdanie, jezyk)
        with torch.inference_mode():
            try:
                st = StrumienS3(sz, m.conds.gen, n_krokow=t_tryb["kroki"], nakladka=t_tryb.get("nakladka", 4),
                                kontekst_hift=t_tryb.get("kontekst_hift", 12), znak=m.watermarker, tempo=tempo,
                                maks_tokenow=limit)
            except ValueError as e:                     # nietypowa podpowiedź głosu
                info(f"Strumień niedostępny dla tego głosu ({e}) – zdaniami.")
                calym_zdaniem()
                return
            kolejne = granice((t_tryb["pierwszy"], t_tryb["kawalek"]))
            g = next(kolejne)
            mowa: list = []
            t_start = time.perf_counter()
            t_s3 = 0.0
            try:
                for t in self.szybki.generuj_strumien(
                        t3_cond=m.conds.t3, text_tokens=tokeny, max_new_tokens=limit,
                        temperature=float(temperatura), cfg_weight=float(cfg), repetition_penalty=1.2,
                        min_p=0.05, top_p=1.0, przerwij=przerwij):
                    if t >= TOKEN_MOWY_MAKS:            # start/stop – jak drop_invalid_tokens
                        continue
                    mowa.append(t)
                    if len(mowa) >= g + WYPRZEDZENIE:
                        t1 = time.perf_counter()
                        audio = st.przetworz(mowa, g, koniec=False)
                        t_s3 += time.perf_counter() - t1
                        nadaj(audio, False)
                        g = next(kolejne)
            except (FloatingPointError, ValueError, NotImplementedError) as e:
                if st.wyemitowane == 0 and not przerwano():
                    info(f"Szybki dekoder: {type(e).__name__}: {e} – to zdanie zwykłą pętlą.")
                    calym_zdaniem()
                else:
                    info(f"Szybki dekoder: {type(e).__name__}: {e} – reszta zdania pominięta.")
                return
            if przerwano():
                return
            t1 = time.perf_counter()
            czasy["t3_ms"] += (t1 - t_start - t_s3) * 1000      # generowanie tokenów bez wokodera w trakcie
            audio = st.przetworz(mowa, len(mowa), koniec=True) if len(mowa) >= 2 else np.zeros(0, np.float32)
            t_s3 += time.perf_counter() - t1
            czasy["s3_ms"] += t_s3 * 1000
            czasy["tokeny"] += len(mowa)
            nadaj(audio, True, tokeny=len(mowa))

    # ---------------------------------------------------------------- tworzenie głosu

    def utworz_glos(self, wejscie: str, katalog: str, od: float | None = None, do: float | None = None,
                    ekspresja: float = 0.5) -> dict:
        """Z dowolnego nagrania (mp3, wav, m4a, ogg, flac, mp4…) robi „model głosu”:
        conds.pt (warunkowanie Chatterboxa) + referencja.wav (wybrane ~10 s) + glos.json."""
        import numpy as np
        import soundfile as sf
        import torch
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.mtl_tts import Conditionals

        m = self.model
        katalog = Path(katalog)
        katalog.mkdir(parents=True, exist_ok=True)
        audio = wczytaj_audio(wejscie, SR)
        calosc_s = len(audio) / SR
        ostrzezenia = []
        if od is not None or do is not None:
            a = int(max(0.0, od or 0.0) * SR)
            b = int(min(calosc_s, do if do else calosc_s) * SR)
            if b - a < SR:
                raise ValueError("Wybrany fragment jest krótszy niż 1 s")
            audio = audio[a:b]
        if np.mean(np.abs(audio) > 0.99) > 0.001:
            ostrzezenia.append("Nagranie jest przesterowane (obcięte szczyty) – głos może brzmieć szorstko.")
        maska, prog_db = maska_mowy(audio, SR)
        mowa_s = float(maska.sum()) * 0.01
        if mowa_s < 3.0:
            raise ValueError(f"W nagraniu jest za mało mowy ({mowa_s:.1f} s). Potrzeba min. 5 s, najlepiej 15–60 s.")
        if mowa_s < 8.0:
            ostrzezenia.append(f"Mało mowy ({mowa_s:.0f} s) – głos będzie mniej podobny. Najlepiej 15–60 s czystej mowy.")
        a, b = najlepszy_fragment(maska, SR, dlugosc_s=10.0)
        ref = zageszczenie_ciszy(audio[a:b], SR)
        ref = normalizuj_glosnosc(ref)
        # zapas: Chatterbox bierze najwyżej 10 s do vocodera i 6 s do modelu językowego
        ref = ref[: 10 * SR]
        # całe tokeny (40 ms = 960 próbek): wtedy melspektrogram podpowiedzi ma dokładnie 2 ramki na token
        # (inaczej jedna ramka nadmiarowa przesuwa początek generowanej mowy względem podpowiedzi)
        ref = ref[: len(ref) // 960 * 960]
        sf.write(str(katalog / "referencja.wav"), ref, SR, subtype="PCM_16")

        import librosa
        ref16 = librosa.resample(ref, orig_sr=SR, target_sr=16000)
        with torch.inference_mode():
            gen = m.s3gen.embed_ref(ref, SR, device=m.device)
            plen = m.t3.hp.speech_cond_prompt_len
            prompt, _ = m.s3gen.tokenizer.forward([ref16[: 6 * 16000]], max_len=plen)
            prompt = torch.atleast_2d(prompt).to(m.device)
            # tożsamość głosu (embedding mówcy) liczona z całej mowy w nagraniu, nie tylko z 10 s
            segmenty = segmenty_mowy(audio, maska, SR, maks_s=90.0)
            seg16 = [librosa.resample(s, orig_sr=SR, target_sr=16000) for s in segmenty] or [ref16]
            emb = m.ve.embeds_from_wavs(seg16, sample_rate=16000, as_spk=True)
            emb = torch.from_numpy(np.asarray(emb, dtype=np.float32)).reshape(1, -1).to(m.device)
        t3c = T3Cond(speaker_emb=emb, cond_prompt_speech_tokens=prompt,
                     emotion_adv=float(ekspresja) * torch.ones(1, 1, 1)).to(device=m.device)
        conds = Conditionals(t3c, gen)
        conds.save(katalog / "conds.pt")
        self._cache_conds.pop(str(katalog / "conds.pt"), None)
        return {"nagranie_s": round(calosc_s, 1), "mowa_s": round(mowa_s, 1),
                "fragment_s": [round(a / SR, 1), round(b / SR, 1)], "referencja_s": round(len(ref) / SR, 1),
                "segmentow": len(segmenty), "ostrzezenia": ostrzezenia}


# ---------------------------------------------------------------- obróbka nagrania

def ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def wczytaj_audio(sciezka: str, sr: int):
    """Dowolny format (przez ffmpeg) -> mono float32 w zadanej częstotliwości."""
    import numpy as np
    exe = ffmpeg()
    if exe:
        with tempfile.TemporaryDirectory() as tmp:
            wyj = Path(tmp) / "audio.wav"
            r = subprocess.run([exe, "-nostdin", "-y", "-loglevel", "error", "-i", sciezka, "-vn",
                                "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", str(wyj)],
                               capture_output=True, timeout=300,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode == 0 and wyj.exists():
                import soundfile as sf
                a, _ = sf.read(str(wyj), dtype="float32")
                return np.ascontiguousarray(a.reshape(-1))
            blad = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            info(f"ffmpeg nie odczytał pliku ({blad[0]}) – próbuję librosą.")
    import librosa
    a, _ = librosa.load(sciezka, sr=sr, mono=True)
    return a.astype(np.float32)


def maska_mowy(audio, sr: int):
    """Prosty detektor mowy: ramki 10 ms z energią blisko poziomu głośnych fragmentów."""
    import numpy as np
    hop = sr // 100
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    ramki = audio[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(ramki ** 2, axis=1)) + 1e-9)
    prog = max(np.percentile(db, 95) - 30.0, -50.0)
    maska = db > prog
    # wygładzenie: krótkie przerwy (do 150 ms) w środku mowy też są mową
    wynik = maska.copy()
    przerwa = 0
    for i, v in enumerate(maska):
        if v:
            if 0 < przerwa <= 15:
                wynik[i - przerwa:i] = True
            przerwa = 0
        else:
            przerwa += 1
    return wynik, float(prog)


def najlepszy_fragment(maska, sr: int, dlugosc_s: float = 10.0) -> tuple[int, int]:
    """Okno ~10 s z największym udziałem mowy (próbki audio od-do)."""
    import numpy as np
    hop = sr // 100
    okno = int(dlugosc_s * 100 * 1.25)          # trochę dłużej – cisza zostanie skrócona
    if len(maska) <= okno:
        idx = np.flatnonzero(maska)
        a, b = (idx[0], idx[-1] + 1) if len(idx) else (0, len(maska))
        return a * hop, b * hop
    suma = np.convolve(maska.astype(np.float32), np.ones(okno, dtype=np.float32), mode="valid")
    start = int(np.argmax(suma))
    idx = np.flatnonzero(maska[start:start + okno])
    a, b = start + idx[0], start + idx[-1] + 1
    return max(0, a - 5) * hop, (b + 5) * hop


def zageszczenie_ciszy(audio, sr: int, maks_cisza_s: float = 0.35):
    """Skraca długie pauzy w referencji (więcej mowy w 10 s = lepsze podobieństwo)."""
    import numpy as np
    maska, _ = maska_mowy(audio, sr)
    hop = sr // 100
    czesci, cisza = [], 0
    maks = int(maks_cisza_s * 100)
    for i, v in enumerate(maska):
        fragment = audio[i * hop:(i + 1) * hop]
        if v:
            cisza = 0
            czesci.append(fragment)
        else:
            cisza += 1
            if cisza <= maks:
                czesci.append(fragment)
    return np.concatenate(czesci) if czesci else audio


def normalizuj_glosnosc(audio, cel_db: float = -20.0):
    import numpy as np
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    wynik = audio * (10 ** (cel_db / 20) / rms)
    szczyt = float(np.max(np.abs(wynik)) + 1e-9)
    if szczyt > 0.95:
        wynik *= 0.95 / szczyt
    return wynik.astype(np.float32)


def segmenty_mowy(audio, maska, sr: int, maks_s: float = 90.0) -> list:
    """Ciągłe fragmenty mowy (1,5-10 s) do policzenia embeddingu mówcy."""
    hop = sr // 100
    segmenty, start, razem = [], None, 0.0
    for i, v in enumerate(list(maska) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dl = (i - start) / 100
            if dl >= 1.5:
                for a in range(start, i, 1000):            # dłuższe tniemy na kawałki po 10 s
                    b = min(i, a + 1000)
                    if (b - a) / 100 >= 1.5:
                        segmenty.append(audio[a * hop:b * hop])
                        razem += (b - a) / 100
            start = None
        if razem >= maks_s:
            break
    return segmenty


# ---------------------------------------------------------------- serwer (protokół stdin/stdout)

def serwer():
    """Pętla poleceń. Osobny wątek czyta stdin, żeby polecenie „przerwij” (anulowanie bieżącej
    syntezy strumieniowej, gdy lektor porzucił kwestię) docierało w trakcie syntezy; pozostałe
    polecenia trafiają do kolejki i wykonują się po kolei."""
    import queue as _queue

    proto = os.fdopen(os.dup(1), "wb", buffering=0)   # prawdziwy stdout – tylko dla protokołu
    os.dup2(2, 1)                                     # wszystko inne na stderr
    sys.stdout = sys.stderr
    # Prawdziwy stdin (potok od aplikacji) czyta tylko wątek protokołu, z prywatnej kopii uchwytu.
    # Standardowe wejście procesu podmieniamy na NUL: w Windows czekający odczyt z potoku blokuje
    # każdą inną operację na tym uchwycie (np. gdy biblioteka przy ładowaniu modelu uruchamia
    # podproces, który dziedziczy stdin) – bez tego ładowanie modelu zawieszało się na zawsze.
    wejscie = os.fdopen(os.dup(0), "rb")
    nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nul, 0)
    os.close(nul)
    sys.stdin = open(os.devnull, "r", encoding="utf-8")
    silnik = Silnik()
    lock = threading.Lock()
    polecenia: _queue.Queue = _queue.Queue()
    przerwij = threading.Event()
    biezace = {"id": None}
    anulowane: set = set()          # „przerwij” dla syntezy, która jeszcze czeka w kolejce
    stan_lock = threading.Lock()

    def odpowiedz(obj: dict, dane: bytes = b""):
        obj["n_bajtow"] = len(dane)
        with lock:
            proto.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n" + dane)

    def czytaj():
        for surowa in wejscie:
            surowa = surowa.strip()
            if not surowa:
                continue
            try:
                z = json.loads(surowa)
            except ValueError:
                polecenia.put({"cmd": "_zly_json"})
                continue
            if z.get("cmd") == "przerwij":
                if z.get("id") is not None:
                    with stan_lock:
                        if z.get("id") == biezace["id"]:
                            przerwij.set()  # przerywa tylko tę syntezę, o którą chodzi
                        else:
                            anulowane.add(z.get("id"))
                continue
            polecenia.put(z)
        polecenia.put(None)                 # aplikacja zamknęła stdin – kończymy

    threading.Thread(target=czytaj, name="stdin", daemon=True).start()
    odpowiedz({"ok": True, "gotowy": True, "pid": os.getpid(), "protokol": 2})
    while True:
        z = polecenia.get()
        if z is None:
            break
        cmd = z.get("cmd")
        if cmd == "_zly_json":
            odpowiedz({"ok": False, "blad": "zły JSON"})
            continue
        try:
            if cmd == "syntezuj_strumien":
                with stan_lock:
                    przerwij.clear()
                    biezace["id"] = z.get("id")
                    if z.get("id") in anulowane:      # przerwana, zanim ruszyła
                        anulowane.discard(z.get("id"))
                        przerwij.set()
                    if len(anulowane) > 100:
                        anulowane.clear()
                t0 = time.perf_counter()
                try:
                    czasy = silnik.syntezuj_strumien(
                        z["tekst"], z.get("jezyk", "pl"), z.get("ekspresja", 0.5), z.get("cfg", 0.5),
                        z.get("temperatura", 0.8), tempo=float(z.get("tempo", 1.0)), tryb=z.get("tryb"),
                        wyslij=lambda audio, info: odpowiedz({"ok": True, "sr": SR, "koniec": False, **info},
                                                             audio.tobytes()),
                        przerwij=przerwij)
                finally:
                    with stan_lock:
                        biezace["id"] = None
                odpowiedz({"ok": True, "koniec": True, "przerwano": przerwij.is_set(),
                           "ms": round((time.perf_counter() - t0) * 1000), **czasy})
                continue
        except Exception as e:
            slad = traceback.format_exc()
            sys.stderr.write(slad)
            odpowiedz({"ok": False, "koniec": True, "blad": f"{type(e).__name__}: {e}", "slad": slad[-1500:]})
            continue
        try:
            if cmd == "ping":
                odpowiedz({"ok": True, "zaladowany": silnik.model is not None, "wersja": silnik.wersja})
            elif cmd == "zaladuj":
                if silnik.model is None or silnik.wersja != z.get("wersja", "v3"):
                    silnik.model = None
                    wynik = silnik.zaladuj(z.get("wersja", "v3"), z.get("urzadzenie", "auto"), tryb=z.get("tryb"))
                else:
                    if z.get("tryb"):     # zmiana trybu: przygotuj (i rozgrzej) jego vocoder od razu
                        silnik._s3gen_szybki(TRYBY.get(z["tryb"], TRYBY["zrownowazony"])["dekoder"])
                    wynik = {"urzadzenie": silnik.urzadzenie, "sr": SR, "juz": True}
                odpowiedz({"ok": True, **wynik})
            elif cmd == "glos":
                odpowiedz({"ok": True, **silnik.ustaw_glos(z.get("conds"))})
            elif cmd == "syntezuj":
                t0 = time.perf_counter()
                audio, czasy = silnik.syntezuj(z["tekst"], z.get("jezyk", "pl"), z.get("ekspresja", 0.5),
                                               z.get("cfg", 0.5), z.get("temperatura", 0.8))
                odpowiedz({"ok": True, "sr": SR, "ms": round((time.perf_counter() - t0) * 1000), **czasy},
                          audio.tobytes())
            elif cmd == "utworz_glos":
                wynik = silnik.utworz_glos(z["wejscie"], z["katalog"], z.get("od"), z.get("do"),
                                           z.get("ekspresja", 0.5))
                odpowiedz({"ok": True, **wynik})
            elif cmd == "koniec":
                odpowiedz({"ok": True})
                break
            else:
                odpowiedz({"ok": False, "blad": f"nieznane polecenie {cmd}"})
        except Exception as e:
            slad = traceback.format_exc()
            sys.stderr.write(slad)
            odpowiedz({"ok": False, "blad": f"{type(e).__name__}: {e}", "slad": slad[-1500:]})
    os._exit(0)


# ---------------------------------------------------------------- test szybkiego dekodera

def test_szybki(wersja: str) -> bool:
    """Porównuje szybki dekoder T3 (fp16, grafy CUDA, próbkowanie w grafie) z oryginalną pętlą
    Hugging Face: rozkład, z którego losowany jest każdy token (po CFG, karze za powtórzenia,
    temperaturze i min_p), przy tej samej wymuszonej sekwencji tokenów – średnia odległość TV
    (szansa, że wylosowany zostanie inny token) i zgodność najbardziej prawdopodobnego tokenu.
    Do tego: generuj == generuj_strumien przy tym samym ziarnie, przerwanie, czas na token."""
    import threading as _th
    import torch
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    from lektor.chatterbox_szybki import SzybkiT3

    CFG, TEMP, KARA, MIN_P = 0.5, 0.8, 1.2, 0.05
    ARG = dict(max_new_tokens=400, temperature=TEMP, cfg_weight=CFG, repetition_penalty=KARA, min_p=MIN_P,
               top_p=1.0)
    silnik = Silnik()
    silnik.zaladuj(wersja, szybki=False)
    m = silnik.model
    t3 = m.t3
    cond = m.conds.t3
    teksty = ["Słuchaj, nie mamy dużo czasu. Gliny będą tu za pięć minut.",
              "Jason, ty naprawdę myślisz, że on nas tak po prostu wypuści?",
              "Hej! Masz to?"]
    hf = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)

    def rozklad(log2, obecne):
        """(n, 2, V) logity -> rozkład próbkowania (n, V) jak w T3.inference."""
        war, bezw = log2[:, 0].float(), log2[:, 1].float()
        lg = war + CFG * (war - bezw)
        lg = torch.where(obecne, torch.where(lg < 0, lg * KARA, lg / KARA), lg) / TEMP
        p = torch.softmax(lg, -1)
        lg = lg.masked_fill(p < p.amax(-1, keepdim=True) * MIN_P, float("-inf"))
        return lg, torch.softmax(lg, -1)

    tv_wszystkie, top1_wszystkie, zgodne_strumien = [], [], 0
    with torch.inference_mode():
        sz = SzybkiT3(t3)
        sz.przygotuj_graf(wymuszone=True)
        for k, tekst in enumerate(teksty):
            tok, _ = silnik._tokeny_tekstu(tekst, "pl")
            torch.manual_seed(7 + k)
            mowa = sz.generuj(t3_cond=cond, text_tokens=tok, **ARG)[0]
            torch.manual_seed(7 + k)
            strumien = list(sz.generuj_strumien(t3_cond=cond, text_tokens=tok, **ARG))
            zgodne_strumien += int(mowa.tolist() == strumien)
            # oryginał (HF, fp32/TF32): ta sama sekwencja tokenów wymuszona krok po kroku
            poczatek = t3.hp.start_speech_token * torch.ones_like(tok[:, :1])
            embeds, _ = t3.prepare_input_embeds(t3_cond=cond, text_tokens=tok, speech_tokens=poczatek,
                                                cfg_weight=CFG)
            bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=m.device)
            bos_emb = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
            wyj = hf(inputs_embeds=torch.cat([embeds, torch.cat([bos_emb, bos_emb])], dim=1), past_key_values=None,
                     use_cache=True, output_hidden_states=True, return_dict=True)
            past = wyj.past_key_values
            ref = [wyj.logits[:, -1, :].float().clone()]
            for i, t in enumerate(mowa[:-1].tolist()):
                nast = torch.tensor([[t]], dtype=torch.long, device=m.device)
                emb = t3.speech_emb(nast) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
                wyj = hf(inputs_embeds=torch.cat([emb, emb]), past_key_values=past, output_hidden_states=True,
                         return_dict=True)
                past = wyj.past_key_values
                ref.append(wyj.logits[:, -1, :].float().clone())
            ref = torch.stack(ref)                                                    # (n, 2, V)
            szyb = sz.logity_wymuszone(t3_cond=cond, text_tokens=tok, cfg_weight=CFG, tokeny=mowa)
            n, V = ref.shape[0], ref.shape[-1]
            obecne = torch.zeros(n, V, dtype=torch.bool, device=m.device)
            obecne[:, t3.hp.start_speech_token] = True
            for j in range(1, n):
                obecne[j] = obecne[j - 1]
                obecne[j, int(mowa[j - 1])] = True
            l_ref, p_ref = rozklad(ref, obecne)
            l_sz, p_sz = rozklad(szyb, obecne)
            tv_wszystkie += (0.5 * (p_ref - p_sz).abs().sum(-1)).tolist()
            top1_wszystkie += (l_ref.argmax(-1) == l_sz.argmax(-1)).float().tolist()
        tv = sum(tv_wszystkie) / len(tv_wszystkie)
        top1 = sum(top1_wszystkie) / len(top1_wszystkie)
        print(f"Zgodność z oryginałem ({len(tv_wszystkie)} kroków): szansa innego tokenu (TV) {tv * 100:.2f}%, "
              f"ten sam najbardziej prawdopodobny token w {top1 * 100:.1f}% kroków.", flush=True)
        print(f"generuj == generuj_strumien przy tym samym ziarnie: {zgodne_strumien}/{len(teksty)}.", flush=True)

        # przerwanie: Event ustawiony w trakcie -> generator kończy się od razu
        tok, _ = silnik._tokeny_tekstu(teksty[0], "pl")
        stop = _th.Event()
        ile = 0
        for _ in sz.generuj_strumien(t3_cond=cond, text_tokens=tok, przerwij=stop, **ARG):
            ile += 1
            if ile == 5:
                stop.set()
        przerwanie_ok = ile == 5
        print(f"Przerwanie w trakcie: {'działa' if przerwanie_ok else f'NIE działa ({ile} tokenów)'}.", flush=True)

        # czas na token (cała pętla z odbiorem tokenów) i do pierwszego tokenu
        tok, _ = silnik._tokeny_tekstu("Słuchaj, nie mamy dużo czasu. Gliny będą tu za pięć minut, a ty jeszcze "
                                       "nawet nie odpaliłeś samochodu.", "pl")
        czasy, pierwsze = [], []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t_pierwszy, n = None, 0
            for _t in sz.generuj_strumien(t3_cond=cond, text_tokens=tok, **ARG):
                n += 1
                if t_pierwszy is None:
                    t_pierwszy = time.perf_counter()
            t_k = time.perf_counter()
            pierwsze.append((t_pierwszy - t0) * 1000)
            czasy.append((t_k - t_pierwszy) / max(1, n - 1) * 1000)
        t_sz = sorted(czasy)[1]
        # oryginał: krok pętli HF
        emb = t3.speech_emb(torch.tensor([[100]], device=m.device)) + t3.speech_pos_emb.get_fixed_embedding(1)
        past2 = past
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            wyj = hf(inputs_embeds=torch.cat([emb, emb]), past_key_values=past2, output_hidden_states=True,
                     return_dict=True)
            past2 = wyj.past_key_values
        torch.cuda.synchronize()
        t_hf = (time.perf_counter() - t0) / 30 * 1000
        print(f"Czas na token: oryginał {t_hf:.1f} ms, szybki dekoder {t_sz:.2f} ms ({t_hf / t_sz:.1f}× szybciej); "
              f"pierwszy token po {sorted(pierwsze)[1]:.1f} ms.", flush=True)
    return tv < 0.005 and top1 >= 0.97 and zgodne_strumien == len(teksty) and przerwanie_ok


# ---------------------------------------------------------------- tryb ręczny

def test_reczny(wersja: str):
    import numpy as np
    import soundfile as sf
    silnik = Silnik()
    print(f"Ładuję Chatterbox {wersja}…", flush=True)
    print(silnik.zaladuj(wersja), flush=True)
    zdania = ["Słuchaj, nie mamy dużo czasu. Gliny będą tu za pięć minut.",
              "Jason, ty naprawdę myślisz, że on nas tak po prostu wypuści?",
              "Witamy w Vice City. Tutaj nikt nie pyta, skąd masz pieniądze."]
    calosc = []
    for z in zdania:
        t0 = time.perf_counter()
        audio, czasy = silnik.syntezuj(z)
        dt = time.perf_counter() - t0
        print(f"  {dt:5.2f} s na {len(audio) / SR:4.1f} s mowy  {czasy}  | {z}", flush=True)
        calosc += [audio, np.zeros(int(0.5 * SR), dtype=np.int16)]
    (BAZA / "probki").mkdir(exist_ok=True)
    plik = BAZA / "probki" / f"chatterbox_{wersja}_test.wav"
    sf.write(str(plik), np.concatenate(calosc), SR, subtype="PCM_16")
    print(f"Zapisano {plik}", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    wersja = "v2" if "--v2" in args else "v3"
    if "--pobierz" in args:
        pobierz_modele(wersja, log=lambda m: print(m, flush=True),
                       postep=lambda p: print(f"\r  {p * 100:5.1f}%", end="", flush=True))
        print("Modele Chatterbox są na miejscu.", flush=True)
    if "--test-szybki" in args:
        ok = test_szybki(wersja)
        print("Szybki dekoder: " + ("ZGODNY z oryginałem." if ok else "NIEZGODNY – aplikacja użyje zwykłej pętli."),
              flush=True)
    if "--test" in args:
        test_reczny(wersja)
    if not args or "--serwer" in args:
        serwer()


if __name__ == "__main__":
    main()
