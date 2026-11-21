"""Głos lektora: Piper TTS (lokalnie, na procesorze) albo Chatterbox (GPU, osobny proces,
klonowanie głosu z nagrania) i odtwarzanie przez sounddevice."""
from __future__ import annotations

import collections
import json
import queue
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import chatterbox_pliki as cbp

KATALOG_GLOSOW = Path(__file__).resolve().parents[1] / "modele" / "piper"
URL_PACZKI = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{}.tar.bz2"

# Polskie głosy Pipera (pliki .onnx + .onnx.json z paczek sherpa-onnx na GitHubie)
GLOSY = {
    "meski": {"nazwa": "męski (WitoldG)", "plik": "pl_PL-meski_wg_glos-medium",
              "paczka": "vits-piper-pl_PL-meski_wg_glos-medium"},
    "jarvis": {"nazwa": "jarvis (WitoldG)", "plik": "pl_PL-jarvis_wg_glos-medium",
               "paczka": "vits-piper-pl_PL-jarvis_wg_glos-medium"},
    "darkman": {"nazwa": "darkman", "plik": "pl_PL-darkman-medium",
                "paczka": "vits-piper-pl_PL-darkman-medium"},
    "mc_speech": {"nazwa": "mc_speech", "plik": "pl_PL-mc_speech-medium",
                  "paczka": "vits-piper-pl_PL-mc_speech-medium"},
    "bass": {"nazwa": "bass (high, wolniejszy)", "plik": "pl_PL-bass-high",
             "paczka": "vits-piper-pl_PL-bass-high"},
}


def lista_glosow() -> list[dict]:
    """Wbudowane głosy Pipera + każdy inny głos Pipera z modele/piper + głosy Chatterboxa."""
    wynik = []
    znane = set()
    for klucz, g in GLOSY.items():
        onnx = KATALOG_GLOSOW / f"{g['plik']}.onnx"
        wynik.append({"id": klucz, "nazwa": "Piper · " + g["nazwa"], "pobrany": onnx.exists(), "silnik": "piper"})
        znane.add(g["plik"])
    if KATALOG_GLOSOW.exists():
        for onnx in sorted(KATALOG_GLOSOW.glob("*.onnx")):
            if onnx.stem not in znane and onnx.with_suffix(".onnx.json").exists():
                wynik.append({"id": "plik:" + onnx.stem, "nazwa": "Piper · " + onnx.stem + " (własny)",
                              "pobrany": True, "silnik": "piper"})
    if cbp.zainstalowany():
        for g in cbp.lista_glosow():
            g["silnik"] = "chatterbox"
            wynik.append(g)
    return wynik


def _sciezki(klucz: str) -> tuple[Path, Path]:
    if klucz.startswith("plik:"):
        stem = klucz[5:]
    else:
        stem = GLOSY[klucz]["plik"]
    onnx = KATALOG_GLOSOW / f"{stem}.onnx"
    return onnx, Path(str(onnx) + ".json")


def pobierz_glos(klucz: str, log=print, postep=None) -> tuple[Path, Path]:
    """Pobiera głos z GitHuba (jeśli go nie ma) i zwraca ścieżki modelu i konfiguracji."""
    onnx, cfg = _sciezki(klucz)
    if onnx.exists() and cfg.exists():
        return onnx, cfg
    if klucz.startswith("plik:"):
        raise FileNotFoundError(f"Brak pliku {onnx.name} lub {cfg.name} w modele/piper")
    g = GLOSY[klucz]
    KATALOG_GLOSOW.mkdir(parents=True, exist_ok=True)
    url = URL_PACZKI.format(g["paczka"])
    log(f"Pobieram głos Pipera „{g['nazwa']}” z GitHuba…")
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "glos.tar.bz2"
        req = urllib.request.Request(url, headers={"User-Agent": "lektor-gta"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tar_path, "wb") as f:
            razem = int(r.headers.get("Content-Length") or 0)
            pobrane = 0
            while True:
                blok = r.read(1 << 20)
                if not blok:
                    break
                f.write(blok)
                pobrane += len(blok)
                if postep and razem:
                    postep(pobrane / razem)
        with tarfile.open(tar_path, "r:bz2") as tar:
            for m in tar.getmembers():
                nazwa = Path(m.name).name
                if nazwa in (onnx.name, cfg.name) and m.isfile():
                    zrodlo = tar.extractfile(m)
                    if zrodlo is None:
                        continue
                    docelowy = KATALOG_GLOSOW / nazwa
                    with open(docelowy, "wb") as out:
                        out.write(zrodlo.read())
    if not (onnx.exists() and cfg.exists()):
        raise RuntimeError(f"W paczce {g['paczka']} nie znalazłem plików modelu")
    log(f"Głos „{g['nazwa']}” gotowy.")
    return onnx, cfg


class GlosPiper:
    """Załadowany głos Pipera; syntezuje zdanie po zdaniu (pierwsze audio jak najszybciej)."""

    def __init__(self, log=print):
        self.log = log
        self.voice = None
        self.klucz = None
        self.sample_rate = 22050
        self._lock = threading.Lock()

    def zaladuj(self, klucz: str, postep=None) -> None:
        from piper import PiperVoice, SynthesisConfig

        onnx, cfg = pobierz_glos(klucz, self.log, postep)
        t0 = time.perf_counter()
        voice = PiperVoice.load(str(onnx), config_path=str(cfg))
        # rozgrzewka: pierwsza synteza zawsze jest wolniejsza
        for _ in voice.synthesize("Dzień dobry.", syn_config=SynthesisConfig()):
            pass
        with self._lock:
            self.voice = voice
            self.klucz = klucz
            self.sample_rate = voice.config.sample_rate
        self.log(f"Załadowano głos {klucz} ({(time.perf_counter() - t0) * 1000:.0f} ms).")

    def syntezuj(self, tekst: str, predkosc: float = 1.0, przy_starcie=None):
        """Generator kawałków audio int16 (jeden kawałek = jedno zdanie)."""
        from piper import SynthesisConfig

        with self._lock:
            voice = self.voice
        if voice is None:
            raise RuntimeError("Głos nie jest załadowany")
        cfg = SynthesisConfig(length_scale=1.0 / max(0.5, min(2.0, predkosc)))
        for chunk in voice.synthesize(tekst, syn_config=cfg):
            yield chunk.audio_int16_array.reshape(-1), chunk.sample_rate


# ---------------------------------------------------------------- Chatterbox

def podziel_na_zdania(tekst: str, min_dl: int = 25, maks_dl: int = 220) -> list[str]:
    """Dzieli kwestię na kawałki do osobnej syntezy: pierwsze zdanie zabrzmi szybciej, a następne
    generują się, gdy lektor mówi poprzednie. Bardzo krótkie zdania („Co?”) łączymy z następnym."""
    czesci = [c for c in re.split(r"(?<=[.!?…])\s+", tekst.strip()) if c]
    wynik, bufor = [], ""
    for c in czesci:
        bufor = f"{bufor} {c}".strip() if bufor else c
        if len(bufor) >= min_dl:
            wynik.append(bufor)
            bufor = ""
    if bufor:
        if wynik and len(bufor) < 12:
            wynik[-1] += " " + bufor
        else:
            wynik.append(bufor)
    gotowe = []
    for z in wynik:
        while len(z) > maks_dl:                      # bardzo długie zdanie: tniemy na przecinku
            i = z.rfind(", ", 0, maks_dl)
            if i < min_dl:
                i = z.find(" ", maks_dl // 2)
                if i < 0:
                    break
            gotowe.append(z[:i + 1].strip())
            z = z[i + 1:].strip()
        if z:
            gotowe.append(z)
    return gotowe


def zmien_tempo(audio: np.ndarray, tempo: float, sr: int) -> np.ndarray:
    """Zmiana tempa mowy bez zmiany wysokości głosu (WSOLA). tempo > 1 = szybciej."""
    if abs(tempo - 1.0) < 0.02 or len(audio) < sr // 5:
        return audio
    x = audio.astype(np.float32)
    n = int(0.030 * sr)                 # okno 30 ms
    hop_wy = n // 2
    hop_we = hop_wy * tempo
    tol = int(0.012 * sr)               # szukanie najlepszego dopasowania ±12 ms
    okno = np.hanning(n).astype(np.float32)
    x = np.pad(x, (tol, n + 2 * tol))
    dl_wy = int(len(audio) / tempo) + 2 * n
    y = np.zeros(dl_wy, dtype=np.float32)
    suma_okien = np.zeros(dl_wy, dtype=np.float32)
    poz_we, poz_wy, poprz = 0.0, 0, None
    while True:
        idealna = int(round(poz_we)) + tol
        if idealna + n + tol > len(x) or poz_wy + n > dl_wy:
            break
        if poprz is None:
            start = idealna
        else:
            obszar = x[idealna - tol: idealna + tol + n]
            korelacja = np.correlate(obszar, poprz, mode="valid")
            start = idealna - tol + int(np.argmax(korelacja))
        y[poz_wy:poz_wy + n] += x[start:start + n] * okno
        suma_okien[poz_wy:poz_wy + n] += okno
        poprz = x[start + hop_wy: start + hop_wy + n]
        if len(poprz) < n:
            break
        poz_we += hop_we
        poz_wy += hop_wy
    suma_okien[suma_okien < 1e-3] = 1.0
    wynik = (y / suma_okien)[:max(1, int(len(audio) / tempo))]
    return np.clip(wynik, -32768, 32767).astype(np.int16)


class GlosChatterbox:
    """Chatterbox Multilingual (Resemble AI) w osobnym procesie z .venv-chatterbox.

    Proces startuje przy pierwszym wyborze głosu Chatterbox, ładuje model na GPU (kilka s)
    i robi rozgrzewkę (kilkanaście s – jednorazowo). Potem syntezuje zdanie po zdaniu."""

    def __init__(self, ustawienia, log=print):
        self.ust = ustawienia
        self.log = log
        self.voice = None              # klucz głosu, gdy gotowy
        self.klucz = None
        self.sample_rate = cbp.SR
        self.info: dict = {}
        self.ostatnie_czasy: dict = {}
        self._proc = None
        self._lock = threading.Lock()          # jedno żądanie (albo cały strumień kwestii) naraz
        self._lock_zapisu = threading.Lock()   # sam zapis do stdin – „przerwij” musi przejść w trakcie strumienia
        self._ogon_stderr: collections.deque = collections.deque(maxlen=80)
        self._model = None             # (wersja, urządzenie) załadowane w procesie
        self._nr = 0                   # numer żądania syntezy (do „przerwij”)
        self.skrypt = cbp.SKRYPT_WORKERA

    # --- proces ---
    def _start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not cbp.zainstalowany():
            raise RuntimeError("Chatterbox nie jest zainstalowany – uruchom instaluj_chatterbox.bat")
        import os
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen(
            [str(cbp.PYTHON_CHATTERBOX), "-u", str(self.skrypt), "--serwer"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cbp.BAZA), env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._model = None
        threading.Thread(target=self._czytaj_stderr, args=(self._proc,), name="chatterbox-stderr",
                         daemon=True).start()
        odp, _ = self._odbierz()
        if not odp.get("gotowy"):
            raise RuntimeError("Proces Chatterboxa nie wystartował poprawnie")

    def _czytaj_stderr(self, proc):
        for surowa in proc.stderr:
            linia = surowa.decode("utf-8", "replace").rstrip()
            if not linia:
                continue
            self._ogon_stderr.append(linia)
            if linia.startswith("LEKTOR: "):
                self.log("Chatterbox: " + linia[8:])

    def _odbierz(self) -> tuple[dict, bytes]:
        linia = self._proc.stdout.readline()
        if not linia:
            raise EOFError("proces nie odpowiada")
        odp = json.loads(linia)
        n = int(odp.get("n_bajtow") or 0)
        dane = bytearray()
        while len(dane) < n:
            kawalek = self._proc.stdout.read(n - len(dane))
            if not kawalek:
                raise EOFError("urwane dane audio")
            dane += kawalek
        return odp, bytes(dane)

    def _pisz(self, obj: dict):
        with self._lock_zapisu:
            self._proc.stdin.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
            self._proc.stdin.flush()

    def przerwij(self, nr: int):
        """Przerywa syntezę o danym numerze (z dowolnego wątku; nie czeka na trwający strumień).
        Proces sprawdza flagę co token, więc GPU zwalnia się od razu, a nie po całym zdaniu."""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                self._pisz({"cmd": "przerwij", "id": nr})
            except (OSError, ValueError):
                pass

    def _wyslij(self, obj: dict) -> tuple[dict, bytes]:
        with self._lock:
            self._start()
            try:
                self._pisz(obj)
                odp, dane = self._odbierz()
            except (OSError, EOFError, ValueError) as e:
                ogon = " | ".join(l for l in list(self._ogon_stderr)[-6:] if l.strip())
                self._zabij()
                raise RuntimeError(f"Proces Chatterboxa przestał działać ({e}). Ostatnie komunikaty: {ogon}")
        if not odp.get("ok"):
            raise RuntimeError(odp.get("blad") or "nieznany błąd Chatterboxa")
        return odp, dane

    def _zabij(self):
        p, self._proc = self._proc, None
        self._model = None
        self.voice = None
        if p is not None:
            try:
                p.kill()
            except Exception:
                pass

    def zamknij(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._pisz({"cmd": "koniec"})
                self._proc.wait(timeout=3)
            except Exception:
                pass
        self._zabij()

    # --- interfejs głosu ---
    def _zapewnij_model(self, etap=None, postep=None):
        wersja = str(self.ust.get("cb_model", "v3"))
        urz = str(self.ust.get("cb_urzadzenie", "auto"))
        tryb = str(self.ust.get("cb_tryb", "zrownowazony"))
        if cbp.brakujace_pliki(wersja):
            cbp.pobierz_modele(wersja, self.log, postep)
        if self._model != (wersja, urz, tryb) or self._proc is None or self._proc.poll() is not None:
            if etap:
                etap("ładuję Chatterbox na GPU (pierwszy raz do ~40 s)…")
            odp, _ = self._wyslij({"cmd": "zaladuj", "wersja": wersja, "urzadzenie": urz, "tryb": tryb})
            if not odp.get("juz"):
                self.info = {k: v for k, v in odp.items() if k not in ("ok", "n_bajtow")}
            self._model = (wersja, urz, tryb)

    def zaladuj(self, klucz: str, postep=None, etap=None) -> None:
        t0 = time.perf_counter()
        self.voice = None
        self._zapewnij_model(etap, postep)
        conds = cbp.sciezka_conds(klucz)
        self._wyslij({"cmd": "glos", "conds": str(conds) if conds else None})
        self.klucz = self.voice = klucz
        self.log(f"Załadowano głos Chatterbox {klucz[3:]} ({(time.perf_counter() - t0):.1f} s, "
                 f"{self.info.get('gpu', '')}).")

    def syntezuj(self, tekst: str, predkosc: float = 1.0, przy_starcie=None):
        """Strumień kawałków audio jednej kwestii (jedno żądanie do procesu, wiele ramek odpowiedzi):
        krotki (audio int16, sr, koniec_zdania).

        Proces sam dzieli tekst na zdania, a zdania na kawałki (strumieniowo, gdy model językowy
        jeszcze liczy) i oddaje dźwięk, gdy tylko kawałek jest gotowy; tempo zmienia
        w melspektrogramie (bez zmiany wysokości głosu). Gdy lektor porzuci kwestię (generator
        zamknięty przed końcem), wysyłamy „przerwij” i doczytujemy ramki do końca – inaczej
        następna odpowiedź zostałaby pomylona z resztkami tej. przy_starcie(nr) podaje numer
        żądania, żeby lektor mógł je przerwać z innego wątku (przerwij(nr))."""
        if self.voice is None:
            raise RuntimeError("Głos Chatterbox nie jest załadowany")
        with self._lock:
            self._start()
            self._nr += 1
            nr = self._nr
            if przy_starcie is not None:
                przy_starcie(nr)
            zadanie = {"cmd": "syntezuj_strumien", "id": nr, "tekst": tekst, "jezyk": "pl",
                       "ekspresja": float(self.ust.get("cb_ekspresja", 0.5)),
                       "cfg": float(self.ust.get("cb_cfg", 0.5)),
                       "temperatura": float(self.ust.get("cb_temperatura", 0.8)),
                       "tempo": float(predkosc), "tryb": self.ust.get("cb_tryb", "zrownowazony")}
            skonczone = False
            try:
                self._pisz(zadanie)
                while True:
                    odp, dane = self._odbierz()
                    if not odp.get("ok"):
                        skonczone = bool(odp.get("koniec", True))
                        raise RuntimeError(odp.get("blad") or "nieznany błąd Chatterboxa")
                    if dane:
                        # trzeci element: czy to ostatni kawałek zdania (proces strumieniuje też wewnątrz
                        # zdania – ciszę na końcu skracamy tylko na końcu zdania)
                        yield (np.frombuffer(dane, dtype=np.int16).copy(), int(odp.get("sr") or cbp.SR),
                               bool(odp.get("koniec_zdania", True)))
                    if odp.get("koniec"):
                        skonczone = True
                        self.ostatnie_czasy = {k: odp.get(k) for k in
                                               ("ms", "pierwszy_ms", "t3_ms", "s3_ms", "tokeny", "kawalkow")}
                        return
            except (OSError, EOFError, ValueError) as e:
                skonczone = True
                ogon = " | ".join(l for l in list(self._ogon_stderr)[-6:] if l.strip())
                self._zabij()
                raise RuntimeError(f"Proces Chatterboxa przestał działać ({e}). Ostatnie komunikaty: {ogon}")
            finally:
                if not skonczone and self._proc is not None:
                    try:
                        self._pisz({"cmd": "przerwij", "id": nr})
                        while True:
                            odp, _ = self._odbierz()
                            if odp.get("koniec") or not odp.get("ok"):
                                break
                    except Exception:
                        self._zabij()

    def utworz_glos(self, plik: Path, nazwa: str, od: float | None = None, do: float | None = None) -> dict:
        """Tworzy głos z nagrania (wymaga załadowanego modelu – załaduje go w razie potrzeby)."""
        self._zapewnij_model()
        nazwa = nazwa.strip() or plik.stem
        katalog = cbp.KATALOG_GLOSOW / cbp.slug(nazwa)
        odp, _ = self._wyslij({"cmd": "utworz_glos", "wejscie": str(plik), "katalog": str(katalog),
                               "od": od, "do": do, "ekspresja": float(self.ust.get("cb_ekspresja", 0.5))})
        meta = {"nazwa": nazwa, "utworzony": time.strftime("%Y-%m-%d %H:%M"), "zrodlo": plik.name,
                "model": self._model[0] if self._model else None}
        meta.update({k: v for k, v in odp.items() if k not in ("ok", "n_bajtow")})
        (katalog / "glos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        klucz = "cb:" + katalog.name
        if self.klucz == klucz:              # nadpisany aktywny głos – przeładuj warunkowanie
            self._wyslij({"cmd": "glos", "conds": str(katalog / "conds.pt")})
        self.log(f"Utworzono głos Chatterbox „{nazwa}”: {meta.get('mowa_s')} s mowy, "
                 f"referencja {meta.get('referencja_s')} s.")
        meta["id"] = klucz
        return meta


class Glos:
    """Wspólny „głos” dla lektora: wybiera Pipera albo Chatterboxa po kluczu (cb:… = Chatterbox).

    Dopóki nowo wybrany głos się ładuje, lektor mówi poprzednim (nie gubi kwestii)."""

    def __init__(self, ustawienia, log=print):
        self.log = log
        self.piper = GlosPiper(log)
        self.chatterbox = GlosChatterbox(ustawienia, log)
        self.klucz = None
        self.ostatni_silnik = None
        self.awaria = False             # proces Chatterboxa padł – aplikacja wczyta go ponownie
        self._t_ostrzezenia = 0.0

    def _silnik(self, klucz):
        return self.chatterbox if (klucz or "").startswith("cb:") else self.piper

    @property
    def voice(self):
        s = self._silnik(self.klucz)
        return s.voice if (s.voice is not None and s.klucz == self.klucz) else None

    @property
    def nazwa_silnika(self) -> str:
        return "Chatterbox" if (self.klucz or "").startswith("cb:") else "Piper"

    @property
    def sample_rate(self) -> int:
        return self._silnik(self.klucz).sample_rate

    def zaladuj(self, klucz: str, postep=None, etap=None) -> None:
        self.klucz = klucz
        if klucz.startswith("cb:"):
            if self.piper.voice is None:
                # zapasowy Piper (ok. 1 s): czyta, dopóki Chatterbox się ładuje (ok. 15 s po starcie)
                for zapas in ("jarvis", "meski"):
                    if _sciezki(zapas)[0].exists():
                        try:
                            self.piper.zaladuj(zapas)
                            break
                        except Exception:
                            continue
            self.chatterbox.zaladuj(klucz, postep, etap)
        else:
            self.piper.zaladuj(klucz, postep)

    def syntezuj(self, tekst: str, predkosc: float = 1.0, przy_starcie=None):
        s = self._silnik(self.klucz)
        if s.voice is None or s.klucz != self.klucz:
            inny = self.piper if s is self.chatterbox else self.chatterbox
            if inny.voice is not None:
                if time.perf_counter() - self._t_ostrzezenia > 10:
                    self.log("Wybrany głos jeszcze się ładuje – czytam poprzednim.")
                    self._t_ostrzezenia = time.perf_counter()
                s = inny
        self.ostatni_silnik = "Chatterbox" if s is self.chatterbox else "Piper"
        wydane = False
        try:
            for el in s.syntezuj(tekst, predkosc, przy_starcie=przy_starcie):
                wydane = True
                yield el
        except RuntimeError as e:
            # proces Chatterboxa padł, zanim zabrzmiało cokolwiek: tę kwestię czyta Piper,
            # a aplikacja wczytuje Chatterboxa od nowa (awaria=True – patrz Aplikacja._straznik)
            if s is not self.chatterbox or wydane or self.piper.voice is None:
                raise
            self.awaria = True
            self.log(f"{e} – tę kwestię czyta Piper, Chatterbox wczytuje się od nowa.")
            self.ostatni_silnik = "Piper"
            yield from self.piper.syntezuj(tekst, predkosc)

    def przerwij(self, nr: int):
        self.chatterbox.przerwij(nr)

    def utworz_glos(self, plik: Path, nazwa: str, od=None, do=None) -> dict:
        return self.chatterbox.utworz_glos(plik, nazwa, od, do)

    def zamknij(self):
        self.chatterbox.zamknij()


API_WASAPI = "Windows WASAPI"
API_BEZ_TRYBU_BLOKUJACEGO = ("Windows DirectSound", "Windows WDM-KS")   # DS: write() to atrapa; WDM-KS: wyłączność
NAZWY_MAPOWANIA = ("Mapowanie dźwięku Microsoft", "Microsoft Sound Mapper",
                   "Podstawowy sterownik dźwięku", "Primary Sound Driver")
LATENCJA_WASAPI = 0.04          # bufor ~50 ms: zapas na chwilowe przestoje Pythona (GIL), nie opóźnia startu


def _api(sd, nazwa):
    for i, a in enumerate(sd.query_hostapis()):
        if a["name"] == nazwa:
            return i, a
    return None, None


def urzadzenie_wasapi(sd, wybor) -> int | None:
    """Indeks urządzenia WASAPI dla wyboru z interfejsu: None = domyślne urządzenie Windows, nazwa
    urządzenia (zalecane – indeksy zmieniają się po podłączeniu sprzętu) albo stary indeks MME/DS."""
    i_api, api = _api(sd, API_WASAPI)
    if api is None:
        return None
    domyslne = api["default_output_device"] if api["default_output_device"] >= 0 else None
    if wybor in (None, "", -1):
        return domyslne
    if isinstance(wybor, str) and not wybor.lstrip("-").isdigit():
        nazwa = wybor.strip()
    else:
        try:
            d = sd.query_devices(int(wybor))
        except Exception:
            return domyslne
        if d["hostapi"] == i_api:
            return int(wybor)
        nazwa = d["name"].strip()
    if nazwa.startswith(NAZWY_MAPOWANIA):
        return domyslne
    wyjscia = [i for i in api["devices"] if sd.query_devices(i)["max_output_channels"] > 0]
    dokladne = [i for i in wyjscia if sd.query_devices(i)["name"] == nazwa]
    if dokladne:
        return dokladne[0]
    # nazwy MME są ucięte do 31 znaków – porównujemy też prefiks
    pasujace = [i for i in wyjscia if sd.query_devices(i)["name"].startswith(nazwa)]
    return pasujace[0] if len(pasujace) == 1 else domyslne


def x2_polifaza(audio: np.ndarray, m: int = 16) -> np.ndarray:
    """24 kHz -> 48 kHz filtrem sinc z oknem Kaisera (~1 ms na 4 s mowy, obrazy -69 dB; np.interp: -41 dB)."""
    j = np.arange(-m + 1, m + 1)
    ker = (np.sinc(j - 0.5) * np.kaiser(2 * m, 7.0)).astype(np.float32)
    ker /= ker.sum()
    x = audio.astype(np.float32)
    nieparz = np.correlate(np.pad(x, (m - 1, m)), ker, mode="valid")
    y = np.empty(2 * len(x), dtype=np.float32)
    y[0::2] = x
    y[1::2] = nieparz
    return np.clip(y, -32768, 32767).astype(np.int16)


def urzadzenia_audio() -> list[dict]:
    """Lista wyjść dla interfejsu: tylko WASAPI (bez duplikatów MME/DirectSound/WDM-KS).
    Identyfikatorem jest NAZWA urządzenia – indeksy PortAudio zmieniają się po podłączeniu sprzętu."""
    try:
        import sounddevice as sd
        i_api, api = _api(sd, API_WASAPI)
        if api is None:
            return []
        wynik = []
        for i in api["devices"]:
            d = sd.query_devices(i)
            if d["max_output_channels"] > 0:
                dopisek = " (domyślne)" if i == api["default_output_device"] else ""
                wynik.append({"id": d["name"], "nazwa": d["name"] + dopisek})
        return wynik
    except Exception:
        return []


class Odtwarzacz:
    """Odtwarza kawałki int16 przez WASAPI (tryb współdzielony, Windows sam zmienia częstotliwość):
    bufor ok. 50 ms zamiast ok. 93 ms przez MME. Zapasowo: WASAPI w natywnej częstotliwości
    (z przeliczeniem w aplikacji), potem MME, a bez karty dźwiękowej – „na niby”. DirectSound
    i WDM-KS nie są używane: PortAudio nie ma dla nich trybu blokującego (dźwięk by znikał).
    Kawałek da się przerwać w trakcie (krótkie wyciszenie)."""

    BLOK = 480                   # 20 ms przy 24 kHz – szybka reakcja na „ucisz”

    def __init__(self, log=print):
        self.log = log
        self._stream = None
        self._sr = None
        self._sr_wyjscia = None      # gdy wyjście ma inną częstotliwość niż mowa (przeliczamy w aplikacji)
        self._urzadzenie = None
        self.opis = ""
        self.atrapa = False
        self._t_atrapy = 0.0

    def _proby(self, sd, sr, wybor):
        proby = []
        dev_w = urzadzenie_wasapi(sd, wybor)
        if dev_w is not None:
            proby.append(("WASAPI", dict(device=dev_w, samplerate=sr, latency=LATENCJA_WASAPI,
                                         extra_settings=sd.WasapiSettings(auto_convert=True))))
            natywna = int(sd.query_devices(dev_w)["default_samplerate"])
            proby.append(("WASAPI (natywna częstotliwość)", dict(device=dev_w, samplerate=natywna,
                                                                 latency=LATENCJA_WASAPI)))
        dev = None
        if wybor not in (None, "", -1) and not isinstance(wybor, str):
            try:
                d = sd.query_devices(int(wybor))
                if sd.query_hostapis(d["hostapi"])["name"] not in API_BEZ_TRYBU_BLOKUJACEGO:
                    dev = int(wybor)
            except Exception:
                dev = None
        proby.append(("MME", dict(device=dev, samplerate=sr, latency="low")))
        return proby

    def _otworz(self, sr: int, urzadzenie, odswiez: bool = False):
        if self._sr == sr and self._urzadzenie == urzadzenie and not odswiez:
            if self._stream is not None:
                return
            # tryb „na niby” po błędzie – co jakiś czas próbujemy wrócić do prawdziwego wyjścia
            if self.atrapa and time.perf_counter() - self._t_atrapy < 10:
                return
        self.zamknij()
        self._t_atrapy = time.perf_counter()
        self._sr_wyjscia = None
        try:
            import sounddevice as sd
            if odswiez:              # nowa lista urządzeń (np. zmienione domyślne urządzenie Windows)
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception:
                    pass
            ostatni = None
            for nazwa, p in self._proby(sd, sr, urzadzenie):
                try:
                    s = sd.OutputStream(channels=1, dtype="int16", **p)
                    s.start()
                except Exception as e:
                    ostatni = e
                    continue
                self._stream = s
                if int(p["samplerate"]) != sr:
                    self._sr_wyjscia = int(p["samplerate"])
                dev = p["device"] if p["device"] is not None else sd.default.device[1]
                self.opis = f"{nazwa} · {sd.query_devices(dev)['name']} · {s.latency * 1000:.0f} ms"
                break
            if self._stream is None:
                raise ostatni or RuntimeError("brak wyjścia audio")
            self.atrapa = False
        except Exception as e:  # brak karty dźwiękowej – liczymy tylko czasy
            self.log(f"Nie mogę otworzyć wyjścia audio ({e}). Odtwarzam „na niby”.")
            self._stream = None
            self.atrapa = True
            self.opis = "na niby"
        self._sr, self._urzadzenie = sr, urzadzenie

    def _dopasuj(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if not self._sr_wyjscia or self._sr_wyjscia == sr:
            return audio, sr
        if self._sr_wyjscia == 2 * sr:
            return x2_polifaza(audio), self._sr_wyjscia
        n = int(len(audio) * self._sr_wyjscia / sr)
        x = np.linspace(0, len(audio) - 1, n)
        return np.interp(x, np.arange(len(audio)), audio.astype(np.float32)).astype(np.int16), self._sr_wyjscia

    def zamknij(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._sr = None

    def graj_kawalek(self, audio: np.ndarray, sr_wej: int, anuluj: threading.Event, glosnosc: float = 1.0,
                     urzadzenie=None, przy_pierwszym=None) -> bool:
        """Odtwarza jeden kawałek. Zwraca False, jeśli przerwano (z krótkim wyciszeniem)."""
        self._otworz(sr_wej, urzadzenie)
        audio, sr = self._dopasuj(audio, sr_wej)
        if glosnosc != 1.0:
            audio = np.clip(audio.astype(np.float32) * glosnosc, -32768, 32767).astype(np.int16)
        blok_n = max(1, int(self.BLOK * sr / 24000))
        for i in range(0, len(audio), blok_n):
            if anuluj.is_set():
                self._wycisz(audio[i:i + int(0.06 * sr)])
                return False
            blok = audio[i:i + blok_n]
            if przy_pierwszym is not None:
                przy_pierwszym()
                przy_pierwszym = None
            if self.atrapa or self._stream is None:
                time.sleep(len(blok) / sr)
                continue
            try:
                self._stream.write(blok.reshape(-1, 1))
            except Exception as e:
                # np. zmiana domyślnego urządzenia w Windows, uśpiony interfejs audio:
                # odświeżamy listę urządzeń i otwieramy wyjście od nowa
                self.log(f"Błąd wyjścia audio ({e}) – otwieram urządzenie ponownie.")
                self._otworz(sr_wej, urzadzenie, odswiez=True)
                if self._sr_wyjscia and self._sr_wyjscia != sr:
                    return True     # zmieniła się częstotliwość wyjścia – resztę kawałka pomijamy
        return not anuluj.is_set()

    def graj(self, kawalki, stop: threading.Event, glosnosc: float = 1.0,
             urzadzenie=None, przy_pierwszym=None) -> bool:
        """Odtwarza wszystkie kawałki po kolei. Zwraca True, jeśli doczytano do końca."""
        for audio, sr_wej in kawalki:
            if stop.is_set():
                return False
            if not self.graj_kawalek(audio, sr_wej, stop, glosnosc, urzadzenie, przy_pierwszym):
                return False
            przy_pierwszym = None
        return not stop.is_set()

    def _wycisz(self, ogon: np.ndarray):
        if self._stream is None or len(ogon) == 0:
            return
        rampa = np.linspace(1.0, 0.0, len(ogon), dtype=np.float32)
        try:
            self._stream.write((ogon.astype(np.float32) * rampa).astype(np.int16).reshape(-1, 1))
        except Exception:
            pass


def ma_dzwiek(audio: np.ndarray, sr: int, prog_db: float = -45.0) -> bool:
    """Czy w kawałku jest cokolwiek głośniejszego niż cisza (ramki 10 ms, próg względem pełnej skali)."""
    ramka = max(1, sr // 100)
    n = len(audio) // ramka
    if n == 0:
        return False
    x = audio[: n * ramka].astype(np.float32).reshape(n, ramka) / 32768.0
    return bool((20 * np.log10(np.sqrt((x * x).mean(axis=1)) + 1e-9) > prog_db).any())


def przytnij_cisze(audio: np.ndarray, sr: int, poczatek: bool = True, koniec_zostaw_s: float | None = None,
                   prog_db: float = -45.0, zapas_s: float = 0.02) -> np.ndarray:
    """Obcina ciszę na początku kawałka (szybszy start głosu) i – opcjonalnie – długą ciszę na
    końcu (zostawia `koniec_zostaw_s`). Próg liczony w ramkach 10 ms względem pełnej skali int16."""
    ramka = max(1, sr // 100)
    n = len(audio) // ramka
    if n == 0:
        return audio
    x = audio[: n * ramka].astype(np.float32).reshape(n, ramka) / 32768.0
    db = 20 * np.log10(np.sqrt((x * x).mean(axis=1)) + 1e-9)
    glosne = np.flatnonzero(db > prog_db)
    if len(glosne) == 0:
        return audio
    a = max(0, glosne[0] * ramka - int(zapas_s * sr)) if poczatek else 0
    b = len(audio)
    if koniec_zostaw_s is not None:
        b = min(len(audio), (glosne[-1] + 1) * ramka + int(koniec_zostaw_s * sr))
    return audio[a:b] if b > a else audio


def _ostatnia_glosna_ramka(audio: np.ndarray, sr: int, prog_db: float = -45.0) -> int | None:
    """Koniec (w próbkach) ostatniej ramki 10 ms głośniejszej niż cisza; None = sama cisza."""
    ramka = max(1, sr // 100)
    n = len(audio) // ramka
    if n == 0:
        return None
    x = audio[: n * ramka].astype(np.float32).reshape(n, ramka) / 32768.0
    glosne = np.flatnonzero(20 * np.log10(np.sqrt((x * x).mean(axis=1)) + 1e-9) > prog_db)
    return None if len(glosne) == 0 else int((glosne[-1] + 1) * ramka)


def rozdziel_cisze_na_koncu(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Kawałek ze środka zdania -> (dźwięk do zagrania, cisza z końca wstrzymana do następnego kawałka).
    Dopiero następny kawałek pokaże, czy to pauza w zdaniu (wraca w całości), czy koniec zdania."""
    b = _ostatnia_glosna_ramka(audio, sr)
    if b is None:
        return audio[:0], audio
    if b >= len(audio):
        return audio, None
    return audio[:b], audio[b:]


def przytnij_koniec(audio: np.ndarray, sr: int, zostaw_s: float) -> np.ndarray:
    """Koniec zdania: najwyżej `zostaw_s` ciszy po ostatnim głośnym fragmencie (też gdy to sama cisza)."""
    b = _ostatnia_glosna_ramka(audio, sr)
    return audio[: min(len(audio), (b or 0) + int(zostaw_s * sr))]


_KONIEC = object()


def _klucz_tts(tekst: str) -> str:
    """Postać tekstu, od której zależy brzmienie (małe litery, interpunkcja tylko .?!). Wstępną
    syntezę odtwarzamy wyłącznie, gdy potwierdzony napis brzmiałby identycznie – „będą” i „beda”
    to dla lektora co innego, choć porównanie linii (norm_linii) uznaje je za to samo.
    Brak kropki na końcu = kropka (synteza i tak ją dopisuje)."""
    t = " ".join(re.sub(r"[^\w\s?!.]", " ", tekst.lower()).split())
    t = re.sub(r"\s+([?!.])", r"\1", t)
    return t if (not t or t[-1] in ".?!") else t + "."


class _Kwestia:
    """Jedna kwestia w kolejce lektora."""

    def __init__(self, nr: int, tekst: str, meta: dict, wstepna: bool = False):
        self.nr = nr
        self.tekst = tekst
        self.meta = meta
        self.anulowana = threading.Event()
        # kwestia „wstępna” (synteza ruszyła przed potwierdzeniem napisu przez OCR) czeka
        # z odtwarzaniem, aż potok ją potwierdzi albo anuluje
        self.potwierdzona = threading.Event()
        if not wstepna:
            self.potwierdzona.set()
        self.t_wstepna = time.perf_counter() if wstepna else None
        self.nr_zadania = None         # numer żądania w procesie syntezy (natychmiastowe przerwanie)
        self.czasy: dict = {}
        self.tempo = 1.0
        self.zaleglosc_s = 0.0
        self.t_start: float | None = None
        self.silnik: str | None = None


class Lektor:
    """Kolejka kwestii: jeden wątek syntezuje (Piper/Chatterbox), drugi odtwarza.

    Dzięki dwóm wątkom następna kwestia generuje się, gdy poprzednia jeszcze brzmi, więc między
    kwestiami nie ma ciszy na syntezę. Polityki (ustawienie "polityka"):
    - "kolejka" (domyślna): czyta WSZYSTKIE kwestie po kolei, nic nie przepada (zachowany kontekst
      rozmowy). Gdy lektor zostaje w tyle, mówi szybciej (do "tempo_maks") i skraca ciszę na końcu
      kwestii. Tylko kwestie czekające dłużej niż "kolejka_maks_s" są pomijane (bezpiecznik).
    - "dokoncz": dokańcza bieżącą kwestię, a z zaległych zostaje tylko najnowsza.
    - "przerwij": nowa kwestia przerywa bieżącą.
    Każde anulowanie kwestii, której synteza właśnie trwa, natychmiast przerywa żądanie w procesie
    syntezy (po numerze), więc GPU nie liczy już niepotrzebnego zdania.
    """

    ZNAKI_NA_S = 14.0        # szacunek: tyle znaków tekstu na sekundę mowy przy tempie 1,0
    BUFOR_MAKS_S = 8.0       # najwyżej tyle sekund gotowego audio czeka na odtworzenie
    ZALEGLOSC_OD_S = 1.5     # od takiej zaległości lektor zaczyna przyspieszać
    PRZYROST_TEMPA = 0.06    # o tyle (względnie) rośnie tempo na każdą sekundę zaległości ponad próg
    CISZA_NA_KONCU_S = 0.3   # dłuższą ciszę na końcu kawałka skracamy do tylu sekund…
    CISZA_NADRABIANIE_S = 0.08   # …a gdy lektor nadrabia zaległości – do tylu
    MAKS_CZEKANIA_NA_POTWIERDZENIE_S = 3.0

    def __init__(self, glos, odtwarzacz: Odtwarzacz, ustawienia, log=print, po_kwestii=None):
        self.glos = glos
        self.odtwarzacz = odtwarzacz
        self.ust = ustawienia          # obiekt z metodą get(klucz, domyślna)
        self.log = log
        self.po_kwestii = po_kwestii   # callback(dict z metrykami)
        self._cv = threading.Condition()
        self._teksty: collections.deque = collections.deque()   # czekają na syntezę
        self._aktywne: list = []        # w syntezie albo z audio w buforze, jeszcze nie skończone
        self._audio: queue.Queue = queue.Queue()               # (kwestia, audio | _KONIEC | wyjątek, sr)
        self._buforowane_s = 0.0
        self._grana = None
        self._w_syntezie = None
        self._licznik = 0
        self._dziala = True
        self.mowi = False
        self.statystyki: collections.Counter = collections.Counter()
        self._watki = [threading.Thread(target=self._petla_syntezy, name="lektor-synteza", daemon=True),
                       threading.Thread(target=self._petla_odtwarzania, name="lektor-glos", daemon=True)]
        for w in self._watki:
            w.start()

    # --- anulowanie: zawsze przez _anuluj_k_locked (przerywa też trwającą syntezę) ---
    def _anuluj_k_locked(self, k: _Kwestia) -> bool:
        if k.anulowana.is_set():
            return False
        k.anulowana.set()
        if k is self._w_syntezie and k.nr_zadania is not None:
            self._przerwij_zadanie(k.nr_zadania)
        return True

    def _przerwij_zadanie(self, nr: int):
        przerwij = getattr(self.glos, "przerwij", None)
        if przerwij is not None:
            try:
                przerwij(nr)
                self.statystyki["przerwania_gpu"] += 1
            except Exception:
                pass

    def _anuluj_locked(self, zostaw_grana: bool, oprocz: _Kwestia | None = None) -> int:
        """Anuluje czekające i zsyntezowane kwestie (poza `oprocz`). Przy zostaw_grana zostaje też
        bieżąca: ta, którą słychać, a jeśli jeszcze nic nie brzmi – najstarsza aktywna."""
        biezaca = None
        if zostaw_grana:
            biezaca = self._grana or next((k for k in self._aktywne
                                           if not k.anulowana.is_set() and k is not oprocz), None)
        ile = 0
        for k in list(self._teksty) + list(self._aktywne):
            if k is biezaca or k is oprocz:
                continue
            if self._anuluj_k_locked(k) and k.potwierdzona.is_set():
                ile += 1
        self._teksty = collections.deque(x for x in self._teksty if x is oprocz)
        return ile

    def _przytnij_stare_locked(self):
        maks = float(self.ust.get("kolejka_maks_s", 30.0))
        teraz = time.perf_counter()
        while self._teksty and teraz - self._teksty[0].meta["t_zlecenia"] > maks:
            k = self._teksty.popleft()
            self._anuluj_k_locked(k)
            self.log(f"Pominięto kwestię czekającą ponad {maks:.0f} s: „{k.tekst[:60]}”.")

    def _zastosuj_polityke_locked(self, k: _Kwestia) -> bool:
        """Polityka dla nowej, potwierdzonej kwestii k. False = k to duplikat, nie trzeba jej czytać."""
        from .tekst import norm_linii, podobienstwo
        polityka = self.ust.get("polityka", "kolejka")
        if polityka == "przerwij":
            self._anuluj_locked(zostaw_grana=False, oprocz=k)
        elif polityka == "dokoncz":
            if self._anuluj_locked(zostaw_grana=True, oprocz=k):
                self.log("Pominięto zaległą kwestię (przyszła nowsza).")
        else:
            self._przytnij_stare_locked()
            # ta sama kwestia już czeka albo brzmi (kolejka bywa dłuższa niż okno powtórek potoku);
            # próg 98: ponowny odczyt tej samej linii przez OCR daje ~100, a różne kwestie różniące
            # się jednym słowem lub cyfrą („numer 0” / „numer 1”) – zwykle mniej niż 98
            n = norm_linii(k.tekst)
            for inna in list(self._teksty) + list(self._aktywne):
                if inna is not k and not inna.anulowana.is_set() and inna.potwierdzona.is_set() \
                        and podobienstwo(n, norm_linii(inna.tekst)) >= 98:
                    self.statystyki["duplikaty"] += 1
                    self.log(f"Pominięto duplikat w kolejce: „{k.tekst[:60]}”.")
                    return False
        return True

    # --- interfejs ---
    def powiedz(self, tekst: str, meta: dict | None = None):
        meta = dict(meta or {})
        meta.setdefault("t_zlecenia", time.perf_counter())
        with self._cv:
            self._licznik += 1
            k = _Kwestia(self._licznik, tekst, meta)
            if self._zastosuj_polityke_locked(k):
                self._teksty.append(k)
            self._cv.notify_all()

    def przygotuj(self, tekst: str, meta: dict | None = None) -> int:
        """Zaczyna syntezę od razu, ale odtworzy ją dopiero po potwierdz(). Zwraca klucz kwestii."""
        meta = dict(meta or {})
        meta.setdefault("t_zlecenia", time.perf_counter())
        with self._cv:
            self._przytnij_stare_locked()
            self._licznik += 1
            k = _Kwestia(self._licznik, tekst, meta, wstepna=True)
            self._teksty.append(k)
            self._cv.notify_all()
            return k.nr

    def _znajdz_locked(self, nr: int):
        for k in list(self._teksty) + list(self._aktywne):
            if k.nr == nr:
                return k
        return None

    def potwierdz(self, nr: int, tekst: str, meta: dict | None = None) -> bool:
        """Napis potwierdzony. Jeśli brzmiałby tak samo jak przygotowany – wstępna synteza zostaje
        (zysk na czasie); jeśli nie – jest przerywana i kwestia idzie normalnie (powiedz)."""
        meta = dict(meta or {})
        with self._cv:
            k = self._znajdz_locked(nr)
            if k is not None and not k.anulowana.is_set() and _klucz_tts(tekst) == _klucz_tts(k.tekst):
                for klucz in ("t_klatki", "ocr_ms", "surowy", "pominiete"):
                    if klucz in meta:
                        k.meta[klucz] = meta[klucz]
                k.potwierdzona.set()
                if self._zastosuj_polityke_locked(k):
                    self.statystyki["wstepne_trafione"] += 1
                else:
                    self._anuluj_k_locked(k)
                self._cv.notify_all()
                return True
            if k is not None:
                self._anuluj_k_locked(k)
                self.statystyki["wstepne_chybione"] += 1
        self.powiedz(tekst, meta)
        return False

    def anuluj(self, nr: int):
        """Napis okazał się inny / zniknął – wstępna synteza do kosza (GPU przerywa od razu)."""
        with self._cv:
            k = self._znajdz_locked(nr)
            if k is not None:
                self._anuluj_k_locked(k)
                if k in self._teksty:
                    self._teksty.remove(k)
                self._cv.notify_all()

    def ucisz(self):
        with self._cv:
            self._anuluj_locked(zostaw_grana=False)
            self._cv.notify_all()

    def pomin(self):
        """Pomija tylko kwestię, którą właśnie słychać (reszta kolejki zostaje)."""
        with self._cv:
            if self._grana is not None:
                self._anuluj_k_locked(self._grana)
            self._cv.notify_all()

    def zakoncz(self):
        self._dziala = False
        self.ucisz()
        with self._cv:
            self._cv.notify_all()

    def stan(self) -> dict:
        with self._cv:
            potwierdzone = [k for k in list(self._teksty) + list(self._aktywne)
                            if k.potwierdzona.is_set() and not k.anulowana.is_set()]
            czeka = sum(1 for k in potwierdzone if k is not self._grana)
            najstarsza = min((k.meta.get("t_klatki") or k.meta["t_zlecenia"] for k in potwierdzone), default=None)
            return {"mowi": self.mowi, "w_kolejce": czeka,
                    "zaleglosc_s": round(self._zaleglosc_s_locked(), 1),
                    "opoznienie_s": round(time.perf_counter() - najstarsza, 1) if najstarsza else 0.0,
                    "tempo": round(self._grana.tempo, 2) if self._grana is not None else None,
                    **dict(self.statystyki)}

    # --- tempo ---
    def _zaleglosc_s_locked(self, bez=None) -> float:
        """Ile sekund mowy czeka przed najnowszą kwestią: gotowe audio + szacunek dla tekstów."""
        baza = max(0.5, float(self.ust.get("predkosc", 1.0)))
        tekst = sum(len(k.tekst) for k in self._teksty if k is not bez and not k.anulowana.is_set())
        return self._buforowane_s + tekst / (self.ZNAKI_NA_S * baza)

    def _tempo_locked(self, k: _Kwestia) -> float:
        baza = float(self.ust.get("predkosc", 1.0))
        k.zaleglosc_s = self._zaleglosc_s_locked(bez=k)
        if not self.ust.get("przyspieszaj", True) or k.zaleglosc_s <= self.ZALEGLOSC_OD_S:
            return baza
        maks = max(baza, float(self.ust.get("tempo_maks", 1.5)))
        return min(maks, baza * (1.0 + self.PRZYROST_TEMPA * (k.zaleglosc_s - self.ZALEGLOSC_OD_S)))

    # --- wątek syntezy ---
    def _przy_starcie(self, k: _Kwestia):
        def f(nr):
            k.nr_zadania = nr
            if k.anulowana.is_set():            # anulowana, zanim proces dostał żądanie
                self._przerwij_zadanie(nr)
        return f

    def _petla_syntezy(self):
        while self._dziala:
            with self._cv:
                while self._dziala and (not self._teksty or self._buforowane_s > self.BUFOR_MAKS_S):
                    self._cv.wait(timeout=0.2)
                if not self._dziala:
                    return
                k = self._teksty.popleft()
                if k.anulowana.is_set():
                    continue
                k.tempo = self._tempo_locked(k)
                nadrabia = k.zaleglosc_s > self.ZALEGLOSC_OD_S
                self._aktywne.append(k)
                self._w_syntezie = k
            k.t_start = time.perf_counter()
            try:
                gen = self.glos.syntezuj(k.tekst, k.tempo, przy_starcie=self._przy_starcie(k))
                przed_glosem = True             # jeszcze nic nie zabrzmiało (cisza na początku = opóźnienie)
                wstrzymana = None               # cisza z końca kawałka w środku zdania (patrz niżej)
                try:
                    for i, el in enumerate(gen):
                        audio, sr = el[0], el[1]
                        koniec_zdania = bool(el[2]) if len(el) > 2 else True
                        if k.anulowana.is_set() or not self._dziala:
                            break
                        if i == 0:
                            k.czasy["tts_ms"] = (time.perf_counter() - k.t_start) * 1000
                            k.silnik = getattr(self.glos, "ostatni_silnik", None) or \
                                getattr(self.glos, "nazwa_silnika", None)
                        if przed_glosem and not ma_dzwiek(audio, sr):
                            continue                    # sam początkowy oddech/cisza – nie czekamy na niego
                        if wstrzymana is not None:      # cisza z końca poprzedniego kawałka tego zdania
                            audio = np.concatenate([wstrzymana, audio])
                            wstrzymana = None
                        # cisza na początku = opóźnienie; długą ciszę na końcu zdania skracamy (bardziej
                        # przy nadrabianiu). Kawałek ze środka zdania (strumień): ciszę z jego końca
                        # wstrzymujemy – pauza w zdaniu wróci z następnym kawałkiem, a koniec zdania
                        # zostanie skrócony jak zawsze, nawet gdy wypadł na granicy kawałków.
                        if przed_glosem:
                            audio = przytnij_cisze(audio, sr, poczatek=True)
                        if koniec_zdania:
                            audio = przytnij_koniec(audio, sr, self.CISZA_NADRABIANIE_S if nadrabia
                                                    else self.CISZA_NA_KONCU_S)
                        else:
                            audio, wstrzymana = rozdziel_cisze_na_koncu(audio, sr)
                        przed_glosem = False
                        if len(audio) == 0:
                            continue
                        with self._cv:
                            self._buforowane_s += len(audio) / sr
                        self._audio.put((k, audio, sr))
                finally:
                    gen.close()            # deterministycznie: „przerwij” + doczytanie, zwolnienie procesu
                self._audio.put((k, _KONIEC, None))
            except Exception as e:
                self._audio.put((k, e, None))
            finally:
                with self._cv:
                    if self._w_syntezie is k:
                        self._w_syntezie = None

    # --- wątek odtwarzania ---
    def _petla_odtwarzania(self):
        while self._dziala:
            try:
                k, el, sr = self._audio.get(timeout=0.2)
            except queue.Empty:
                continue
            if el is _KONIEC or isinstance(el, Exception):
                self._koniec_kwestii(k, el if isinstance(el, Exception) else None)
                continue
            dl = len(el) / sr
            try:
                if k.anulowana.is_set():
                    continue
                if not k.potwierdzona.is_set():
                    # wstępna kwestia (synteza ruszyła przed potwierdzeniem napisu) – czekamy na potok
                    while self._dziala and not (k.potwierdzona.is_set() or k.anulowana.is_set()):
                        if time.perf_counter() - (k.t_wstepna or 0) > self.MAKS_CZEKANIA_NA_POTWIERDZENIE_S:
                            with self._cv:
                                self._anuluj_k_locked(k)
                            self.statystyki["wstepne_wygasle"] += 1
                            self.log("Wstępnie przygotowana kwestia nie została potwierdzona – pomijam ją.")
                            break
                        k.potwierdzona.wait(0.005)
                    if k.anulowana.is_set():
                        continue
                with self._cv:
                    self._grana = k
                    self.mowi = True

                def pierwszy_dzwiek(k=k):
                    k.czasy.setdefault("t_audio", time.perf_counter())

                self.odtwarzacz.graj_kawalek(el, sr, k.anulowana, glosnosc=float(self.ust.get("glosnosc", 1.0)),
                                             urzadzenie=self.ust.get("urzadzenie_audio"),
                                             przy_pierwszym=pierwszy_dzwiek)
            except Exception as e:
                self.log(f"Błąd odtwarzania: {e}")
            finally:
                with self._cv:
                    self._buforowane_s = max(0.0, self._buforowane_s - dl)
                    self._cv.notify_all()

    def _koniec_kwestii(self, k: _Kwestia, blad: Exception | None):
        with self._cv:
            if k in self._aktywne:
                self._aktywne.remove(k)
            if self._grana is k:
                self._grana = None
            self.mowi = self._grana is not None
            self._cv.notify_all()
        if blad is not None:
            self.log(f"Błąd lektora: {blad}")
            return
        t_audio = k.czasy.get("t_audio")
        if t_audio is None:          # anulowana, zanim zabrzmiała – nie trafia do historii
            return
        if self.po_kwestii:
            meta = k.meta
            self.po_kwestii({
                "tekst": k.tekst,
                "surowy": meta.get("surowy", k.tekst),
                "pominiete": meta.get("pominiete") or [],
                "silnik": k.silnik or "Piper",
                "tts_ms": round(k.czasy.get("tts_ms", 0)),
                "ocr_ms": meta.get("ocr_ms"),
                "czekanie_ms": round(((k.t_start or t_audio) - meta["t_zlecenia"]) * 1000),
                "calosc_ms": round((t_audio - meta["t_klatki"]) * 1000) if meta.get("t_klatki") else None,
                "przerwana": k.anulowana.is_set(),
                "predkosc": round(k.tempo, 2),
                "zaleglosc_s": round(k.zaleglosc_s, 1),
                "wstepna": k.t_wstepna is not None,
                "czas": time.strftime("%H:%M:%S"),
            })
