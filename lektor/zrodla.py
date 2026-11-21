"""Źródła obrazu: karta przechwytująca (Elgato) lub kamera, ekran, plik wideo, pojedynczy obraz."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BACKENDY = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}


class Zrodlo:
    """Wspólna baza: wątek produkuje klatki, a my trzymamy tylko najnowszą.

    Błąd źródła (np. chwilowy błąd zrzutu ekranu w Windows, karta, która na moment zgubi
    sygnał) nie kończy już pracy na stałe: wątek próbuje ponownie co 0,5–5 s, a gdy klatki
    wrócą, czyści komunikat o błędzie."""

    opis = "źródło"

    def __init__(self, log=print):
        self.log = log
        self._klatka = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._dziala = False
        self._watek = None
        self._gen = 0
        self.fps = 0.0
        self.blad = ""

    def start(self):
        self._dziala = True
        self._gen += 1
        self._watek = threading.Thread(target=self._bezpieczna_petla, args=(self._gen,),
                                       name=type(self).__name__, daemon=True)
        self._watek.start()

    def stop(self):
        self._dziala = False
        if self._watek is not None:
            self._watek.join(timeout=2)

    def restart(self):
        """Uruchamia wątek źródła od nowa (gdy przestał podawać klatki, np. zawieszony sterownik).
        Stary wątek, nawet jeśli wisi, po wybudzeniu sam się zakończy (inny numer generacji)."""
        self.stop()
        self.start()

    def _trwa(self, gen: int) -> bool:
        return self._dziala and gen == self._gen

    def ostatnia_klatka(self):
        with self._lock:
            return self._klatka, self._ts

    def _podaj(self, klatka: np.ndarray, ts: float | None = None):
        with self._lock:
            self._klatka = klatka
            self._ts = ts if ts is not None else time.perf_counter()

    def _bezpieczna_petla(self, gen: int):
        licznik, t_licz = 0, time.perf_counter()
        przerwa = 0.5
        while self._trwa(gen):
            klatek = 0
            try:
                for _ in self._petla(gen):
                    klatek += 1
                    if self.blad:
                        self.log("Źródło obrazu znów działa.")
                        self.blad = ""
                    licznik += 1
                    teraz = time.perf_counter()
                    if teraz - t_licz >= 1.0:
                        self.fps = licznik / (teraz - t_licz)
                        licznik, t_licz = 0, teraz
                    if not self._trwa(gen):
                        break
                if not self._trwa(gen):
                    break
                raise RuntimeError("źródło przestało podawać klatki")
            except Exception as e:
                if not self._trwa(gen):
                    break
                opis = str(e) or type(e).__name__
                if opis != self.blad:
                    self.log(f"Błąd źródła obrazu: {opis} – ponawiam za {przerwa:.1f} s.")
                self.blad = opis
                self.fps = 0.0
            koniec = time.perf_counter() + przerwa
            while self._trwa(gen) and time.perf_counter() < koniec:
                time.sleep(0.05)
            przerwa = 0.5 if klatek > 30 else min(5.0, przerwa * 2)

    def _petla(self, gen: int):  # generator: yield po każdej klatce
        raise NotImplementedError


class ZrodloKamera(Zrodlo):
    """Karta przechwytująca widziana w Windows jako kamera (np. Elgato HD60 X)."""

    def __init__(self, indeks: int = 0, backend: str = "dshow", szer: int = 1920, wys: int = 1080,
                 fps: int = 60, log=print):
        super().__init__(log)
        self.indeks, self.backend, self.szer, self.wys, self.fps_cel = indeks, backend, szer, wys, fps
        self.opis = f"karta/kamera #{indeks} ({backend})"

    def _petla(self, gen: int):
        api = BACKENDY.get(self.backend, cv2.CAP_ANY) if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.indeks, api)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Nie mogę otworzyć urządzenia #{self.indeks}. Czy OBS/Elgato Studio go nie używa?")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.szer)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wys)
            cap.set(cv2.CAP_PROP_FPS, self.fps_cel)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log(f"Przechwytywanie: {w}×{h}, {cap.get(cv2.CAP_PROP_FPS):.0f} kl./s ({self.opis}).")
            bledy = 0
            while self._trwa(gen):
                ok, klatka = cap.read()
                if not ok or klatka is None:
                    bledy += 1
                    if bledy > 100:
                        raise RuntimeError("Karta przestała podawać obraz (sprawdź HDCP w PS5 i kabel).")
                    time.sleep(0.01)
                    continue
                bledy = 0
                self._podaj(klatka)
                yield
        finally:
            cap.release()


class ZrodloEkran(Zrodlo):
    """Zrzut ekranu (np. okno podglądu z OBS albo film z gry odtwarzany na monitorze)."""

    def __init__(self, monitor: int = 1, fps: int = 30, log=print):
        super().__init__(log)
        self.monitor, self.fps_cel = monitor, fps
        self.opis = f"ekran #{monitor}"

    def _petla(self, gen: int):
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[min(max(1, self.monitor), len(sct.monitors) - 1)]
            self.log(f"Przechwytuję ekran #{self.monitor}: {mon['width']}×{mon['height']}.")
            odstep = 1.0 / max(1, self.fps_cel)
            while self._trwa(gen):
                t = time.perf_counter()
                img = np.asarray(sct.grab(mon))[:, :, :3]  # BGRA -> BGR
                self._podaj(np.ascontiguousarray(img), t)
                yield
                reszta = odstep - (time.perf_counter() - t)
                if reszta > 0:
                    time.sleep(reszta)


class ZrodloPlik(Zrodlo):
    """Plik wideo odtwarzany w czasie rzeczywistym (zapętlony) – do powtarzalnych testów."""

    def __init__(self, sciezka: str, log=print):
        super().__init__(log)
        self.sciezka = sciezka
        self.opis = f"plik {Path(sciezka).name}"

    def _petla(self, gen: int):
        cap = cv2.VideoCapture(self.sciezka)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Nie mogę otworzyć pliku: {self.sciezka}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            odstep = 1.0 / fps
            self.log(f"Odtwarzam plik {Path(self.sciezka).name} ({fps:.0f} kl./s, w pętli).")
            t_nast = time.perf_counter()
            puste = 0
            while self._trwa(gen):
                ok, klatka = cap.read()
                if not ok:
                    puste += 1
                    if puste > 3:
                        raise RuntimeError(f"Nie mogę czytać klatek z pliku {Path(self.sciezka).name}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                puste = 0
                teraz = time.perf_counter()
                if t_nast > teraz:
                    time.sleep(t_nast - teraz)
                t_nast = max(t_nast + odstep, time.perf_counter() - odstep)
                self._podaj(klatka)
                yield
        finally:
            cap.release()


class ZrodloObraz(Zrodlo):
    """Jeden obraz (np. zrzut ekranu z napisem) podawany jak strumień – do strojenia OCR."""

    def __init__(self, obraz: np.ndarray, nazwa: str = "obraz", log=print):
        super().__init__(log)
        self.obraz = obraz
        self.opis = f"obraz {nazwa}"

    def _petla(self, gen: int):
        self.log(f"Obraz testowy: {self.obraz.shape[1]}×{self.obraz.shape[0]}.")
        while self._trwa(gen):
            self._podaj(self.obraz)
            yield
            time.sleep(0.05)


def wczytaj_obraz(sciezka_lub_bajty) -> np.ndarray:
    if isinstance(sciezka_lub_bajty, (bytes, bytearray)):
        dane = np.frombuffer(sciezka_lub_bajty, dtype=np.uint8)
    else:
        dane = np.fromfile(str(sciezka_lub_bajty), dtype=np.uint8)  # działa też z polskimi znakami w ścieżce
    img = cv2.imdecode(dane, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udało się odczytać obrazu")
    return img


def wykryj_kamery(backend: str = "dshow", maks: int = 6) -> list[dict]:
    """Sprawdza urządzenia 0..maks-1; na Windows dodaje nazwy z DirectShow (pygrabber)."""
    nazwy = []
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            nazwy = FilterGraph().get_input_devices()
        except Exception:
            nazwy = []
    api = BACKENDY.get(backend, cv2.CAP_ANY) if sys.platform == "win32" else cv2.CAP_ANY
    wynik = []
    for i in range(maks):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            ok, kl = cap.read()
            opis = nazwy[i] if i < len(nazwy) else f"urządzenie {i}"
            roz = f"{kl.shape[1]}×{kl.shape[0]}" if ok and kl is not None else "brak obrazu"
            wynik.append({"indeks": i, "nazwa": opis, "obraz": roz})
        cap.release()
    return wynik
