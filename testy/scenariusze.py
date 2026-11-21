"""Scenariusze testowe bez gry: sztuczne „nagrania” z napisami w stylu GTA VI, puszczane przez
prawdziwy OCR i prawdziwy potok lektora (z Twoimi ustawieniami z ustawienia.json).

Sprawdza dokładnie te sytuacje, które psuły odczyt:
  - zmienia się tylko druga linia (pierwsza zostaje) – lektor ma czytać tylko nową linię,
  - napisy przesuwają się w górę / dopisuje się druga linia,
  - przez obszar napisów przejeżdża samochód z tablicą rejestracyjną,
  - tablica stoi z boku napisu,
  - napis na chwilę „mignie” (zniknie na 0,4 s) – nie może być przeczytany drugi raz.

Uruchom:  .venv\\Scripts\\python.exe testy\\scenariusze.py            (wszystkie, ok. 2 min)
          .venv\\Scripts\\python.exe testy\\scenariusze.py tablica    (tylko wybrane)
          .venv\\Scripts\\python.exe testy\\scenariusze.py --mp4      (zapisz też filmy do testy\\scenariusze\\)

Filmy MP4 możesz potem odtworzyć w aplikacji: Źródło obrazu → Plik wideo → ścieżka do pliku.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BAZA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BAZA))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from lektor.ocr import Ocr  # noqa: E402
from lektor.potok import Potok  # noqa: E402
from lektor.tekst import norm_linii, podobienstwo, wczytaj_slownik  # noqa: E402
from lektor.zrodla import Zrodlo  # noqa: E402

W, H, FPS = 1920, 1080, 30
KATALOG_MP4 = BAZA / "testy" / "scenariusze"

A = "Słuchaj, nie mamy dużo czasu."
B = "Gliny będą tu za pięć minut."
C = "Wiesz, co jest najgorsze?"
D = "Że jeszcze rok temu myślałem, że wyjdę z tego czysty."
E = "Dwieście tysięcy w gotówce. Ani centa mniej."

# Scenariusz: lista odcinków (czas trwania s, [linie napisu], tablice), oczekiwane wypowiedzi.
# Tablica: (tekst, x_start, x_koniec, y) – x/y względne, ruch liniowy w czasie odcinka.
SCENARIUSZE = {
    "druga_linia": {
        "opis": "Pierwsza linia zostaje, zmienia się tylko druga (3 razy)",
        "odcinki": [(1.0, [], []), (3.0, [A, B], []), (3.0, [A, C], []), (3.0, [A, D], []), (2.0, [], [])],
        "oczekiwane": [A + " " + B, C, D],
    },
    "przewijanie": {
        "opis": "Napisy przesuwają się w górę: nowa linia pojawia się na dole",
        "odcinki": [(1.0, [], []), (3.0, [A, B], []), (3.0, [B, C], []), (3.0, [C, D], []), (2.0, [], [])],
        "oczekiwane": [A + " " + B, C, D],
    },
    "dopisywanie": {
        "opis": "Najpierw jedna linia, po chwili dopisuje się druga",
        "odcinki": [(1.0, [], []), (2.5, [A], []), (3.0, [A, B], []), (2.0, [], [])],
        "oczekiwane": [A, B],
    },
    "tablica": {
        "opis": "Przez obszar napisów przejeżdża auto z tablicą, potem pojawia się napis",
        "odcinki": [(1.0, [], []), (3.0, [], [("J26B7", 0.10, 0.75, 0.86)]),
                    (3.0, [C], [("KR 4521", 0.85, 0.30, 0.80)]), (2.0, [], [])],
        "oczekiwane": [C],
    },
    "tablica_obok": {
        "opis": "Zaparkowane auto: tablica stoi z boku, obok napisu",
        "odcinki": [(1.0, [], [("LS 42 KR", 0.76, 0.76, 0.86)]), (3.0, [E], [("LS 42 KR", 0.76, 0.76, 0.86)]),
                    (2.0, [], [("LS 42 KR", 0.76, 0.76, 0.86)])],
        "oczekiwane": [E],
    },
    "mrugniecie": {
        "opis": "Napis znika na 0,4 s i wraca (np. przejście kamery) – czytamy raz",
        "odcinki": [(1.0, [], []), (2.5, [A, B], []), (0.4, [], []), (2.5, [A, B], []), (2.0, [], [])],
        "oczekiwane": [A + " " + B],
    },
    "powrot_kwestii": {
        "opis": "Ta sama kwestia wraca po 7 s przerwy – to nowa wypowiedź, czytamy drugi raz",
        "odcinki": [(1.0, [], []), (2.5, [C], []), (7.0, [], []), (2.5, [C], []), (1.5, [], [])],
        "oczekiwane": [C, C],
    },
}


def czcionka(rozmiar: int):
    for nazwa in ("seguisb.ttf", "segoeui.ttf", "ArialNova-Bold.ttf", "arialbd.ttf", "arial.ttf",
                  "VerdanaPro-Bold.ttf", "SegUIVar.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nazwa, rozmiar)
        except OSError:
            continue
    return ImageFont.load_default(size=rozmiar)


FONT_NAPISU = czcionka(40)
FONT_TABLICY = czcionka(34)


def tlo(t: float) -> np.ndarray:
    """Ruchome „miasto”: gradient z przesuwającymi się pasami (kamera w ruchu)."""
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    r = 90 + 60 * np.sin(6.0 * x + 0.7 * t) + 30 * y
    g = 110 + 50 * np.sin(4.0 * x - 0.5 * t + 1.0) + 20 * y
    b = 130 + 60 * np.cos(5.0 * x + 0.3 * t) - 40 * y
    obraz = np.stack([b + 0 * y, g + 0 * y, r + 0 * y], axis=-1)
    pasy = (np.sin(40 * x + 3.0 * t) > 0.97).astype(np.float32) * 40
    obraz += pasy[..., None]
    return np.clip(obraz, 0, 255).astype(np.uint8)


_cache_napisow: dict = {}


def warstwa_napisu(linie: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, int]:
    """Napis jak w GTA VI: jasnoszary, bez obrysu, na półprzezroczystym ciemnym pasku."""
    if linie in _cache_napisow:
        return _cache_napisow[linie]
    wys_linii = 56
    obraz = Image.new("RGBA", (W, wys_linii * len(linie) + 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(obraz)
    pozycje = []
    for i, linia in enumerate(linie):        # najpierw paski, potem tekst (pasek nie zasłoni ogonków)
        szer = d.textlength(linia, font=FONT_NAPISU)
        x, y = (W - szer) / 2, 12 + i * wys_linii
        d.rectangle([x - 16, y - 4, x + szer + 16, y + wys_linii - 4], fill=(0, 0, 0, 120))
        pozycje.append((x, y, linia))
    for x, y, linia in pozycje:
        d.text((x, y), linia, font=FONT_NAPISU, fill=(186, 186, 186, 255))
    rgba = np.asarray(obraz)
    wynik = (rgba[..., :3][..., ::-1].copy(), rgba[..., 3:4].astype(np.float32) / 255.0, rgba.shape[0])
    _cache_napisow[linie] = wynik
    return wynik


_cache_tablic: dict = {}


def warstwa_tablicy(tekst: str) -> np.ndarray:
    """Biała tablica z ciemnymi znakami (jak na amerykańskich autach)."""
    if tekst not in _cache_tablic:
        obraz = Image.new("RGB", (190, 70), (235, 235, 230))
        d = ImageDraw.Draw(obraz)
        d.rectangle([2, 2, 187, 67], outline=(40, 40, 40), width=3)
        szer = d.textlength(tekst, font=FONT_TABLICY)
        d.text(((190 - szer) / 2, 14), tekst, font=FONT_TABLICY, fill=(20, 30, 90))
        _cache_tablic[tekst] = np.asarray(obraz)[..., ::-1].copy()
    return _cache_tablic[tekst]


def klatka(scen: dict, t: float) -> np.ndarray:
    obraz = tlo(t)
    start = 0.0
    for dl, linie, tablice in scen["odcinki"]:
        if start <= t < start + dl:
            postep = (t - start) / dl
            for tekst, xa, xb, y in tablice:
                tab = warstwa_tablicy(tekst)
                th, tw = tab.shape[:2]
                cx = int((xa + (xb - xa) * postep) * W)
                cy = int(y * H)
                x0, y0 = max(0, cx - tw // 2), max(0, cy - th // 2)
                x1, y1 = min(W, x0 + tw), min(H, y0 + th)
                obraz[y0:y1, x0:x1] = tab[:y1 - y0, :x1 - x0]
            if linie:
                rgb, alfa, wys = warstwa_napisu(tuple(linie))
                y0 = int(0.93 * H) - wys
                fragment = obraz[y0:y0 + wys].astype(np.float32)
                obraz[y0:y0 + wys] = (fragment * (1 - alfa) + rgb * alfa).astype(np.uint8)
            break
        start += dl
    return obraz


def dlugosc(scen: dict) -> float:
    return sum(o[0] for o in scen["odcinki"])


class ZrodloScenariusz(Zrodlo):
    """Odtwarza scenariusz w czasie rzeczywistym (30 kl./s), jak karta przechwytująca."""

    def __init__(self, scen: dict, log=print):
        super().__init__(log)
        self.scen = scen
        self.opis = "scenariusz"
        self.koniec = threading.Event()

    def _petla(self, gen: int):
        t0 = time.perf_counter()
        n = 0
        while self._trwa(gen):
            t = time.perf_counter() - t0
            if t >= dlugosc(self.scen):
                self.koniec.set()
                while self._trwa(gen):
                    time.sleep(0.05)
                    yield
                return
            self._podaj(klatka(self.scen, t))
            yield
            n += 1
            reszta = t0 + n / FPS - time.perf_counter()
            if reszta > 0:
                time.sleep(reszta)


class LektorNagrywajacy:
    """Zamiast mówić – zapisuje, co lektor by przeczytał (także ścieżka „synteza wstępna”:
    przygotuj przy 1. odczycie, potwierdz po stabilizacji, anuluj gdy napis się zmienił)."""

    def __init__(self):
        self.wypowiedzi: list[tuple[float, str, dict]] = []
        self.mowi = False
        self.wstepne: dict[int, str] = {}
        self.licznik = 0
        self.potwierdzone = 0
        self.anulowane = 0

    def powiedz(self, tekst, meta=None):
        self.wypowiedzi.append((time.perf_counter(), tekst, dict(meta or {})))

    def przygotuj(self, tekst, meta=None):
        self.licznik += 1
        self.wstepne[self.licznik] = tekst
        return self.licznik

    def potwierdz(self, nr, tekst, meta=None):
        przygotowany = self.wstepne.pop(nr, None)
        trafiona = przygotowany is not None and podobienstwo(norm_linii(przygotowany), norm_linii(tekst)) >= 95
        self.potwierdzone += trafiona
        self.powiedz(tekst, meta)
        return trafiona

    def anuluj(self, nr):
        if self.wstepne.pop(nr, None) is not None:
            self.anulowane += 1

    def ucisz(self):
        pass


class UstawieniaTestowe:
    def __init__(self, d: dict):
        self._d = d

    def get(self, k, domyslna=None):
        return self._d.get(k, domyslna)

    def aktualizuj(self, zmiany):
        self._d.update(zmiany)
        return set(zmiany)


def ustawienia_uzytkownika() -> dict:
    sys.path.insert(0, str(BAZA))
    from aplikacja import DOMYSLNE  # noqa: E402
    d = dict(DOMYSLNE)
    plik = BAZA / "ustawienia.json"
    if plik.exists():
        try:
            d.update(json.loads(plik.read_text(encoding="utf-8")))
        except ValueError:
            pass
    d["auto_poszerzanie"] = False       # test nie może zmieniać zapisanego obszaru
    d["roi"] = [0.16, 0.74, 0.84, 0.965]
    return d


def zgodne(powiedziane: str, oczekiwane: str) -> bool:
    """Podobne (drobne błędy OCR są OK), ale bez doczepionych słów (np. „KR” z tablicy)."""
    p, o = norm_linii(powiedziane), norm_linii(oczekiwane)
    if podobienstwo(p, o) < 85:
        return False
    slowa_o = o.split()
    for s in p.split():
        if len(s) >= 2 and not any(podobienstwo(s, x) >= 75 for x in slowa_o):
            return False
    return True


def uruchom(nazwa: str, scen: dict, ocr: Ocr, ust: dict, slownik) -> bool:
    lektor = LektorNagrywajacy()
    potok = Potok(ocr, lektor, UstawieniaTestowe(dict(ust)), lambda: [], log=lambda m: print("      · " + m))
    zrodlo = ZrodloScenariusz(scen)
    print(f"\n▶ {nazwa}: {scen['opis']} ({dlugosc(scen):.0f} s)")
    potok.start(zrodlo)
    zrodlo.koniec.wait(dlugosc(scen) + 10)
    time.sleep(0.8)
    potok.stop()
    powiedziane = [t for _, t, _ in lektor.wypowiedzi]
    oczekiwane = scen["oczekiwane"]
    for t in powiedziane:
        print(f"    lektor: „{t}”")
    print(f"    synteza wstępna: trafiona {lektor.potwierdzone}×, anulowana {lektor.anulowane}×, "
          f"wisząca {len(lektor.wstepne)}")
    if lektor.wstepne:
        print("  ✘ zostały niepotwierdzone/nieanulowane wstępne syntezy")
        return False
    ok = len(powiedziane) == len(oczekiwane) and all(zgodne(p, o) for p, o in zip(powiedziane, oczekiwane))
    if ok:
        print(f"  ✔ ZALICZONY ({len(powiedziane)} kwestii, zgodnie z oczekiwaniem)")
    else:
        print("  ✘ NIEZALICZONY – oczekiwano:")
        for o in oczekiwane:
            print(f"      „{o}”")
    return ok


def zapisz_mp4(nazwa: str, scen: dict) -> Path:
    KATALOG_MP4.mkdir(parents=True, exist_ok=True)
    plik = KATALOG_MP4 / f"{nazwa}.mp4"
    zapis = cv2.VideoWriter(str(plik), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(dlugosc(scen) * FPS)
    for i in range(n):
        zapis.write(klatka(scen, i / FPS))
    zapis.release()
    return plik


def main() -> int:
    argumenty = [a for a in sys.argv[1:] if not a.startswith("--")]
    wybrane = {k: v for k, v in SCENARIUSZE.items() if not argumenty or k in argumenty}
    if "--mp4" in sys.argv:
        for nazwa, scen in wybrane.items():
            print(f"Zapisuję {zapisz_mp4(nazwa, scen)}")
        if "--tylko-mp4" in sys.argv:
            return 0
    ust = ustawienia_uzytkownika()
    print(f"Model OCR: {ust['ocr_model']}, rozdzielczość {ust['ocr_rozdzielczosc']}, "
          f"stabilizacja {ust['stabilizacja']}, czułość {ust['czulosc']}")
    ocr = Ocr(print)
    ocr.zaladuj(ust["ocr_model"], int(ust["ocr_rozdzielczosc"]))
    slownik = wczytaj_slownik(BAZA / "slownik_wymowy.txt")
    wyniki = {n: uruchom(n, s, ocr, ust, slownik) for n, s in wybrane.items()}
    print("\n==== Podsumowanie")
    for n, ok in wyniki.items():
        print(f"  {'✔' if ok else '✘'} {n}")
    zal = sum(wyniki.values())
    print(f"{zal}/{len(wyniki)} scenariuszy zaliczonych.")
    return 0 if zal == len(wyniki) else 1


if __name__ == "__main__":
    raise SystemExit(main())
