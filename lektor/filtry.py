"""Odsiewanie tekstu, który nie jest napisem dialogowym.

Obszar napisów to dół ekranu, przez który przejeżdżają samochody (tablice rejestracyjne),
przechodzą postacie, migają szyldy i HUD. OCR jest na tyle czuły, że czyta też to – a lektor
potem odczytywał np. „J26B7”. Każde pole OCR przechodzi tu przez filtry; odrzucone dostaje
klucz "odrzucone" z powodem (podgląd rysuje je na czerwono, a „Odczytaj teraz” pokazuje powód).

Filtry (każdy da się wyłączyć w interfejsie):
- obce znaki   – chińskie/japońskie znaki, które PaddleOCR „widzi” w kratkach i liściach;
- tablica      – krótki kod z WIELKICH liter i cyfr (J26B7, ABC 1234);
- ruch         – napisy stoją w miejscu, a tablica jadącego auta się przesuwa: pole, które
                 między dwoma kolejnymi odczytami przesunęło się, nie jest napisem;
- poza środkiem – napisy GTA VI są wyśrodkowane; tekst z boku (szyld, tablica, HUD) odpada;
- ciemny tekst – (domyślnie wyłączony) ciemne litery na jasnym tle, czyli tablica, a nie napis.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .tekst import czysc_znaki, norm_linii, podobienstwo, wyglada_na_tablice

PROG_OBCYCH = 0.25          # udział obcych liter, od którego całe pole uznajemy za śmieć
TOLERANCJA_SRODKA = 0.12    # o ile (część szerokości klatki) środek linii może odbiegać od środka ekranu
RUCH_MIN_PX = 4.0           # przesunięcie (px klatki), od którego pole uznajemy za ruchome
RUCH_WSP_WYSOKOSCI = 0.15   # ...albo 15% wysokości pola, jeśli to więcej
RUCH_MAKS_ODSTEP = 1.0      # s – porównujemy tylko z odczytem sprzed najwyżej tylu sekund


def oznacz_pole(p: dict, wycinek: np.ndarray | None, min_pewnosc: float, opcje: dict) -> None:
    """Filtry „statyczne” jednego pola (współrzędne wycinka: x, y, w, h). Zmienia p w miejscu."""
    if p.get("odrzucone"):
        return
    czysty, obce = czysc_znaki(p["tekst"])
    if obce >= PROG_OBCYCH:
        p["odrzucone"] = "obce znaki"
        return
    p["tekst"] = czysty
    if not any(c.isalpha() for c in czysty):
        p["odrzucone"] = "bez liter"
        return
    if p["pewnosc"] < min_pewnosc:
        p["odrzucone"] = "niska pewność"
        return
    if opcje.get("filtr_tablic", True) and wyglada_na_tablice(czysty):
        p["odrzucone"] = "tablica rejestracyjna"
        return
    if opcje.get("filtr_ciemny", False) and wycinek is not None and ciemny_tekst(wycinek, p):
        p["odrzucone"] = "ciemny tekst na jasnym tle"


def ciemny_tekst(wycinek: np.ndarray, p: dict) -> bool:
    """Czy litery w polu są ciemne na jasnym tle (tablica, szyld), a nie jasne (napis GTA)."""
    h, w = wycinek.shape[:2]
    x0, y0 = max(0, int(p["x"])), max(0, int(p["y"]))
    x1, y1 = min(w, int(p["x"] + p["w"]) + 1), min(h, int(p["y"] + p["h"]) + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return False
    szary = cv2.cvtColor(wycinek[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(szary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    jasne = b > 0
    udzial = float(jasne.mean())
    if udzial <= 0.0 or udzial >= 1.0:
        return False
    # litery zajmują mniej miejsca niż tło – mniejsza klasa to „atrament”
    if udzial < 0.5:
        return False                       # jasne litery na ciemnym tle – tak wyglądają napisy
    atrament = float(szary[~jasne].mean())
    tlo = float(szary[jasne].mean())
    return atrament < 100 and tlo > 140


def oznacz_ruch(pola: list[dict], poprzednie: list[dict], W: int, H: int) -> None:
    """Odrzuca pola, które przesunęły się względem poprzedniego odczytu (współrzędne względne klatki).

    Przesunięcie liczymy jako najmniejsze z przesunięć lewej krawędzi, prawej krawędzi i środka:
    sztywno jadące pole przesuwa się całe, a napis, który się „dopisuje”, ma przynajmniej jedną
    z tych wartości nieruchomą (wyśrodkowany – środek, wyrównany do lewej – lewą krawędź)."""
    if not poprzednie:
        return
    for p in pola:
        if p.get("odrzucone"):
            continue
        n = norm_linii(p["tekst"])
        if len(n) < 2:
            continue
        najlepsze, wynik = None, 0.0
        for q in poprzednie:
            s = podobienstwo(n, norm_linii(q["tekst"]))
            if s > wynik:
                najlepsze, wynik = q, s
        if najlepsze is None or wynik < 75:
            continue
        q = najlepsze
        dl = abs(p["x0"] - q["x0"]) * W
        dp = abs(p["x1"] - q["x1"]) * W
        ds = abs((p["x0"] + p["x1"]) - (q["x0"] + q["x1"])) / 2 * W
        dy = abs((p["y0"] + p["y1"]) - (q["y0"] + q["y1"])) / 2 * H
        przesuniecie = math.hypot(min(dl, dp, ds), dy)
        wys = (p["y1"] - p["y0"]) * H
        if przesuniecie > max(RUCH_MIN_PX, RUCH_WSP_WYSOKOSCI * wys):
            p["odrzucone"] = "ruch"
            p["przesuniecie_px"] = round(przesuniecie, 1)


def grupuj_linie(pola: list[dict]) -> list[list[dict]]:
    """Grupuje pola (współrzędne względne x0..y1) w linie od góry; w linii – od lewej."""
    pola = sorted(pola, key=lambda p: (p["y0"] + p["y1"]) / 2)
    linie: list[list[dict]] = []
    for p in pola:
        srodek, wys = (p["y0"] + p["y1"]) / 2, p["y1"] - p["y0"]
        if linie:
            ost = linie[-1]
            sr = sum((q["y0"] + q["y1"]) / 2 for q in ost) / len(ost)
            w_ost = sum(q["y1"] - q["y0"] for q in ost) / len(ost)
            if abs(srodek - sr) < 0.5 * max(w_ost, wys):
                ost.append(p)
                continue
        linie.append([p])
    return [sorted(l, key=lambda q: q["x0"]) for l in linie]


def _skupiska(linia: list[dict], proporcja: float) -> list[list[dict]]:
    """Dzieli linię na skupiska pól rozdzielone dużą przerwą (np. napis + szyld z boku)."""
    wys = max(1e-6, sum(q["y1"] - q["y0"] for q in linia) / len(linia))
    skupiska = [[linia[0]]]
    for q in linia[1:]:
        prawa = max(r["x1"] for r in skupiska[-1])
        # przerwa w „wysokościach linii”; współrzędne są względne, więc szerokość mnożymy
        # przez proporcje klatki (W/H), żeby porównywać piksele z pikselami
        if (q["x0"] - prawa) * proporcja > 1.5 * wys:
            skupiska.append([q])
        else:
            skupiska[-1].append(q)
    return skupiska


def linie_napisu(pola: list[dict], tylko_srodek: bool = True,
                 tolerancja: float = TOLERANCJA_SRODKA, proporcja: float = 16 / 9) -> list[str]:
    """Składa linie z przyjętych pól. Przy `tylko_srodek` linie (skupiska pól) leżące daleko
    od środka ekranu są odrzucane – napisy GTA VI są wyśrodkowane."""
    przyjete = [p for p in pola if not p.get("odrzucone")]
    wynik = []
    for linia in grupuj_linie(przyjete):
        czesci = []
        for sk in _skupiska(linia, proporcja):
            x0, x1 = min(q["x0"] for q in sk), max(q["x1"] for q in sk)
            if tylko_srodek and abs((x0 + x1) / 2 - 0.5) > tolerancja:
                for q in sk:
                    q["odrzucone"] = "poza środkiem"
                continue
            czesci.append(" ".join(q["tekst"] for q in sk))
        tekst = " ".join(c for c in czesci if c).strip()
        if tekst:
            wynik.append(tekst)
    return wynik


def opis_odrzuconych(pola: list[dict]) -> list[dict]:
    return [{"tekst": p["tekst"], "powod": p["odrzucone"], "pewnosc": round(p["pewnosc"], 2)}
            for p in pola if p.get("odrzucone") and p.get("odrzucone") not in ("bez liter",)]
