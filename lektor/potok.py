"""Potok: klatka -> wycinek napisów -> wykrycie zmiany -> OCR -> stabilizacja -> lektor.

Zmiany po pierwszych testach z GTA VI (19.11):
- napisy GTA VI są jasnoszare (~175-190), bez obrysu, na półprzezroczystym pasku – stały próg
  jasności 200 ich nie widział i OCR w ogóle się nie uruchamiał. Domyślnie działa teraz maska
  automatyczna (cienkie, jasne kreski na tle lokalnym), a OCR i tak robi odczyt kontrolny co 0,6 s;
- ramki z OCR są zapisywane we współrzędnych całej klatki (nie „rozjeżdżają się” po zmianie
  obszaru) i znikają, gdy są stare;
- gdy linia napisu dotyka krawędzi wycinka, obszar jest poszerzany (napisy są wyśrodkowane,
  więc brakujący koniec jest lustrzanym odbiciem widocznego) i odczyt jest powtarzany;
- błąd OCR, zawieszone źródło obrazu czy wyjątek w pętli nie zatrzymują już potoku na stałe.

Zmiany 20.11 (noc):
- pamięć przeczytanych linii: gdy gra zostawia pierwszą linię, a zmienia tylko drugą (albo
  dopisuje/przesuwa linie), lektor czyta wyłącznie nowe linie – wcześniej czytał całość od nowa;
- filtry pól OCR (lektor/filtry.py): tablice rejestracyjne, ruchomy tekst, tekst poza środkiem,
  znaki CJK; odrzucone pola widać na podglądzie na czerwono;
- „pusto” potwierdza teraz OCR, a nie sama maska (maska potrafi chwilę nie widzieć napisu na
  jasnym tle – to też kończyło się ponownym czytaniem tej samej kwestii);
- bezpiecznik: identyczna kwestia nie zostanie powiedziana drugi raz w ciągu kilku sekund;
- stabilizacja „3” wymaga naprawdę 3 zgodnych odczytów (wcześniej działała jak „2”).
"""
from __future__ import annotations

import collections
import threading
import time
import traceback

import cv2
import numpy as np

from .filtry import linie_napisu, opis_odrzuconych, oznacz_pole, oznacz_ruch, RUCH_MAKS_ODSTEP
from .tekst import (PamiecLinii, norm_linii, normalizuj, podobienstwo, skladaj_linie,
                    wyglada_na_napis)

KONTROLA_GDY_TEKST = 0.6      # s – odczyt kontrolny, gdy w wycinku widać coś „tekstowego”
KONTROLA_GDY_PUSTO = 1.5      # s – odczyt kontrolny, gdy maska nic nie widzi (siatka bezpieczeństwa)
CHLODZENIE_PO_RUCHU = 0.2     # s – po odczycie bez nowego tekstu (ruszało się tylko tło)
BRAK_KLATEK_RESTART = 3.0     # s bez nowej klatki = źródło się zawiesiło
PROG_OBECNOSCI = 0.0006       # udział pikseli maski, poniżej którego wycinek uznajemy za pusty
PROG_TOPHAT = 40              # o ile kreska musi być jaśniejsza od lokalnego tła
_JADRO = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
MARGINES_POSZERZENIA = 0.02   # zapas (część szerokości klatki) przy poszerzaniu obszaru
GRANICE_ROI = (0.02, 0.35, 0.98, 0.995)   # dalej obszar sam się nie poszerzy
ZYCIE_RAMEK = 2.5             # s – po tym czasie ramki OCR znikają z podglądu
ZYCIE_LINII = 3.0             # s – tyle pamiętamy przeczytaną linię od chwili, gdy OCR ją ostatnio widział
OKNO_POWTORKI = 6.0           # s – w tym czasie ta sama kwestia nie zostanie powiedziana drugi raz
PROG_POWTORKI = 90            # podobieństwo (0-100), od którego kwestię uznajemy za tę samą


def normalizuj_roi(roi) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in roi)
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    return [x0, y0, x1, y1]


def wytnij_z_pozycja(klatka: np.ndarray, roi) -> tuple[np.ndarray, tuple[int, int]]:
    """Wycinek klatki i jego lewy górny róg w pikselach klatki."""
    h, w = klatka.shape[:2]
    x0, y0, x1, y1 = normalizuj_roi(roi)
    a = min(int(x0 * w), max(0, w - 8))
    c = min(int(y0 * h), max(0, h - 8))
    b = min(w, max(a + 8, int(x1 * w)))
    d = min(h, max(c + 8, int(y1 * h)))
    return klatka[c:d, a:b], (a, c)


def wytnij(klatka: np.ndarray, roi) -> np.ndarray:
    return wytnij_z_pozycja(klatka, roi)[0]


def maska_tekstu(wycinek: np.ndarray, jasnosc: int | None = None) -> np.ndarray:
    """Mała, binarna „sygnatura” tekstu – tania do porównywania klatka po klatce.

    jasnosc=None – tryb automatyczny: cienkie kreski wyraźnie jaśniejsze od swojego
    otoczenia (morfologiczny top-hat). Łapie szary tekst GTA VI na pasku, biały tekst
    z obrysem i tekst na jasnym niebie, a ignoruje duże jasne plamy (niebo, światła).
    jasnosc=liczba – stary tryb: piksele jaśniejsze niż próg.
    """
    szary = cv2.cvtColor(wycinek, cv2.COLOR_BGR2GRAY)
    h, w = szary.shape
    szer = 320
    maly = cv2.resize(szary, (szer, max(8, int(h * szer / max(1, w)))), interpolation=cv2.INTER_AREA)
    if jasnosc:
        return maly > int(jasnosc)
    return cv2.morphologyEx(maly, cv2.MORPH_TOPHAT, _JADRO) > PROG_TOPHAT


def poszerz_roi(roi, pola, poz, ksztalt_wycinka, ksztalt_klatki, min_pewnosc: float):
    """Gdy linia napisu dotyka krawędzi wycinka, zwraca poszerzony obszar (albo None).

    Napisy GTA VI są wyśrodkowane, więc jeśli linia jest ucięta z lewej, a jej prawy koniec
    jest w R, to lewy koniec jest w 1 - R. Tekst poza środkiem (HUD, nazwy ulic) nie
    poszerza obszaru, bo jego „lustrzany” koniec leży wewnątrz wycinka.
    """
    H, W = ksztalt_klatki[:2]
    h, w = ksztalt_wycinka[:2]
    ox, oy = poz
    x0, y0, x1, y1 = normalizuj_roi(roi)
    nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    prog_x, prog_y = max(2.0, 0.004 * w), max(2.0, 0.02 * h)
    for p in pola:
        if p["pewnosc"] < min_pewnosc or not any(c.isalpha() for c in p["tekst"]):
            continue
        lewo = (ox + p["x"]) / W
        prawo = (ox + p["x"] + p["w"]) / W
        gora = (oy + p["y"]) / H
        dol = (oy + p["y"] + p["h"]) / H
        przy_lewej = p["x"] <= prog_x
        przy_prawej = p["x"] + p["w"] >= w - prog_x
        if przy_lewej and przy_prawej:
            nx0, nx1 = min(nx0, x0 - 0.06), max(nx1, x1 + 0.06)
        elif przy_lewej:
            nx0 = min(nx0, 1.0 - prawo - MARGINES_POSZERZENIA)
        elif przy_prawej:
            nx1 = max(nx1, 1.0 - lewo + MARGINES_POSZERZENIA)
        # w pionie tylko dla linii wyglądających na napis (wyśrodkowanych albo właśnie uciętych)
        if przy_lewej or przy_prawej or abs((lewo + prawo) / 2 - 0.5) < 0.12:
            wys = dol - gora
            if p["y"] <= prog_y:
                ny0 = min(ny0, y0 - 1.2 * wys)
            if p["y"] + p["h"] >= h - prog_y:
                ny1 = max(ny1, y1 + 0.8 * wys)
    gx0, gy0, gx1, gy1 = GRANICE_ROI
    nowy = [max(gx0, min(nx0, x0)), max(gy0, min(ny0, y0)), min(gx1, max(nx1, x1)), min(gy1, max(ny1, y1))]
    if max(abs(a - b) for a, b in zip(nowy, (x0, y0, x1, y1))) < 0.003:
        return None
    return [round(v, 4) for v in nowy]


class _Stan:
    """Stan pętli odczytu (osobno, żeby dało się go łatwo zresetować)."""

    def __init__(self):
        self.ost_ts = 0.0
        self.ost_nowa = time.perf_counter()
        self.ost_restart = 0.0
        self.poprz_maska = None
        self.czeka = False            # napis się zmienił i czekamy na stabilny odczyt
        self.t_zmiany = 0.0
        self.kandydat = ""
        self.zgodnych = 0             # ile kolejnych odczytów zgadza się z kandydatem
        self.proby = 0
        self.ocr_ms_suma = 0.0
        self.t_pusto = None           # od kiedy maska nic nie widzi
        self.t_ocr_pusto = None       # od kiedy OCR nic nie czyta
        self.t_ost_ocr = 0.0
        self.chlodzenie_do = 0.0
        self.bledy_z_rzedu = 0
        self.maska_widzi_napis = False  # czy maska „widziała” ostatnio przeczytany napis
        self.wymus_odczyt = False     # maska przestała widzieć napis – niech OCR to potwierdzi
        self.poprz_pola: list[dict] = []   # pola poprzedniego odczytu (filtr ruchu)
        self.t_poprz_pola = 0.0
        self.wstepna = None           # klucz kwestii, której synteza ruszyła przed potwierdzeniem napisu
        self.wstepne_starty = 0       # ile razy w tym epizodzie stabilizacji ruszyła synteza wstępna


class Potok:
    MAKS_STARTOW_WSTEPNYCH = 2   # syntez wstępnych na jeden epizod stabilizacji napisu

    def __init__(self, ocr, lektor, ustawienia, slownik_fn, log=print, przeladuj_ocr=None,
                 diagnostyka=None):
        self.ocr = ocr
        self.lektor = lektor
        self.ust = ustawienia
        self.slownik_fn = slownik_fn   # zwraca aktualny słownik wymowy
        self.log = log
        self.przeladuj_ocr = przeladuj_ocr
        self.diagnostyka = diagnostyka  # opcjonalny zapis odczytów do folderu diagnostyka/
        self.zrodlo = None
        self._dziala = False
        self._watek = None
        # stan widoczny w interfejsie
        self.ostatni_odczyt = ""
        self.ostatni_ocr_ms = 0.0
        self.ostatnio_przeczytane = ""
        self.pamiec = PamiecLinii(ZYCIE_LINII)
        self._wypowiedziane: collections.deque = collections.deque(maxlen=20)   # (czas, tekst znormalizowany)
        self.odrzucone = []            # pola odrzucone przy ostatnim odczycie (opis dla interfejsu)
        self.pola_podgladu = {"ts": 0.0, "pola": []}   # ramki we współrzędnych względnych klatki
        self.ocr_na_sekunde = 0.0
        self.udzial_maski = 0.0
        self.t_ost_ocr = 0.0
        self.bledy_ocr = 0
        self.restarty_zrodla = 0
        self.pominiete_linie = 0       # ile linii nie przeczytano drugi raz (dla diagnostyki)
        self.wstepne_trafione = 0      # ile razy synteza ruszyła przed potwierdzeniem i się przydała
        self.stan = "zatrzymany"
        self._licz_ocr, self._t_licz = 0, time.perf_counter()
        self._bledy_log: dict[str, list] = {}

    # --- sterowanie ---
    def start(self, zrodlo):
        self.stop()
        self.zrodlo = zrodlo
        zrodlo.start()
        self._dziala = True
        self.wyczysc_ramki()
        self._watek = threading.Thread(target=self._petla, name="potok", daemon=True)
        self._watek.start()
        self.stan = "działa"

    def stop(self):
        self._dziala = False
        if self._watek is not None:
            self._watek.join(timeout=3)
            self._watek = None
        if self.zrodlo is not None:
            self.zrodlo.stop()
        self.stan = "zatrzymany"

    def wyczysc_ramki(self):
        self.pola_podgladu = {"ts": 0.0, "pola": []}

    def ramki_do_podgladu(self) -> list[dict]:
        p = self.pola_podgladu
        if not p["pola"] or time.perf_counter() - p["ts"] > ZYCIE_RAMEK:
            return []
        return p["pola"]

    def _prog_jasnosci(self):
        if self.ust.get("prog_auto", True):
            return None
        return int(self.ust.get("jasnosc_tekstu", 200))

    # --- OCR z poszerzaniem obszaru ---
    def _opcje_filtrow(self) -> dict:
        return {"filtr_tablic": bool(self.ust.get("filtr_tablic", True)),
                "filtr_ciemny": bool(self.ust.get("filtr_ciemny", False))}

    def _ocr(self, klatka: np.ndarray, roi) -> tuple[float, list[dict], list[float]]:
        """OCR wycinka z filtrami pól. Zwraca (ms, pola we współrzędnych względnych klatki – także
        odrzucone, z kluczem "odrzucone" – oraz użyty obszar)."""
        min_p = float(self.ust.get("min_pewnosc", 0.6))
        opcje = self._opcje_filtrow()
        H, W = klatka.shape[:2]
        roi = normalizuj_roi(roi)
        ms_suma = 0.0
        for proba in range(3):
            wyc, (ox, oy) = wytnij_z_pozycja(klatka, roi)
            _, ms, pola = self.ocr.rozpoznaj(wyc, min_p)
            ms_suma += ms
            for p in pola:
                oznacz_pole(p, wyc, min_p, opcje)
            if proba == 2 or not self.ust.get("auto_poszerzanie", True):
                break
            nowy = poszerz_roi(roi, [p for p in pola if not p.get("odrzucone")], (ox, oy),
                               wyc.shape, klatka.shape, min_p)
            if nowy is None:
                break
            roi = nowy
        pola_klatki = [{
            "tekst": p["tekst"], "pewnosc": p["pewnosc"], "odrzucone": p.get("odrzucone"),
            "x0": (ox + p["x"]) / W, "y0": (oy + p["y"]) / H,
            "x1": (ox + p["x"] + p["w"]) / W, "y1": (oy + p["y"] + p["h"]) / H,
        } for p in pola]
        return ms_suma, pola_klatki, roi

    def _linie(self, pola: list[dict], klatka: np.ndarray) -> list[str]:
        H, W = klatka.shape[:2]
        return linie_napisu(pola, bool(self.ust.get("tylko_srodek", True)), proporcja=W / max(1, H))

    def _zapamietaj_poszerzenie(self, roi_uzyty, linie):
        """Jeśli obszar trzeba było poszerzyć i odczyt się udał – zapisz go w ustawieniach."""
        zapisany = normalizuj_roi(self.ust.get("roi"))
        if not linie or max(abs(a - b) for a, b in zip(roi_uzyty, zapisany)) < 0.003:
            return
        self.ust.aktualizuj({"roi": [round(v, 4) for v in roi_uzyty]})
        x0, y0, x1, y1 = roi_uzyty
        self.log(f"Napis wychodził poza zaznaczony obszar – poszerzyłem go do x {x0:.2f}–{x1:.2f}, "
                 f"y {y0:.2f}–{y1:.2f} i zapisałem.")

    def odczytaj_teraz(self) -> dict:
        """Jednorazowy OCR bieżącej klatki (przycisk „Odczytaj teraz”)."""
        if self.zrodlo is None:
            raise RuntimeError("Najpierw uruchom źródło obrazu")
        klatka, _ = self.zrodlo.ostatnia_klatka()
        if klatka is None:
            raise RuntimeError("Brak klatki ze źródła")
        ms, pola, roi = self._ocr(klatka, self.ust.get("roi"))
        linie = self._linie(pola, klatka)
        self.pola_podgladu = {"ts": time.perf_counter(), "pola": pola}
        self._zapamietaj_poszerzenie(roi, linie)
        tekst = skladaj_linie(linie)
        nowe = self.pamiec.nowe(linie) if self.ust.get("bez_powtorek", True) else linie
        return {"linie": linie, "tekst": tekst, "ocr_ms": round(ms, 1), "roi": roi,
                "pola": [{"tekst": p["tekst"], "pewnosc": p["pewnosc"], "odrzucone": p.get("odrzucone")}
                         for p in pola],
                "juz_przeczytane": [l for l in linie if l not in nowe],
                "do_przeczytania": normalizuj(skladaj_linie(nowe), self.slownik_fn(),
                                              bool(self.ust.get("usun_imiona")))}

    # --- zgłaszanie błędów bez zalewania dziennika ---
    def _zglos_blad(self, co: str, e: Exception):
        klucz = f"{co}: {type(e).__name__}: {e}"
        teraz = time.perf_counter()
        wpis = self._bledy_log.get(klucz)
        if wpis and teraz - wpis[0] < 10:
            wpis[1] += 1
            return
        pominiete = wpis[1] if wpis else 0
        self._bledy_log[klucz] = [teraz, 0]
        dopisek = f" (powtórzył się jeszcze {pominiete}×)" if pominiete else ""
        self.log(f"{klucz}{dopisek}")
        if not wpis:  # przy pierwszym wystąpieniu – skąd dokładnie (do zgłoszenia)
            slad = traceback.extract_tb(e.__traceback__)[-2:]
            self.log("   miejsce: " + " <- ".join(f"{s.name}() {s.filename.split('/')[-1].split(chr(92))[-1]}:{s.lineno}"
                                                    for s in reversed(slad)))

    # --- pętla główna ---
    def _petla(self):
        s = _Stan()
        while self._dziala:
            try:
                self._krok(s)
            except Exception as e:     # pętla nie może umrzeć – inaczej lektor „głuchnie” na dobre
                self._zglos_blad("Błąd potoku", e)
                s.czeka = False
                self._anuluj_wstepna(s)
                time.sleep(0.2)

    def _krok(self, s: _Stan):
        zr = self.zrodlo
        klatka, ts = zr.ostatnia_klatka()
        teraz = time.perf_counter()
        if klatka is None or ts == s.ost_ts:
            if (teraz - s.ost_nowa > BRAK_KLATEK_RESTART and not zr.blad
                    and teraz - s.ost_restart > 2 * BRAK_KLATEK_RESTART):
                self.log(f"Źródło obrazu nie podaje nowych klatek od {teraz - s.ost_nowa:.0f} s – uruchamiam je ponownie.")
                s.ost_restart = teraz
                self.restarty_zrodla += 1
                zr.restart()
            time.sleep(0.003)
            return
        s.ost_ts, s.ost_nowa = ts, teraz

        roi = self.ust.get("roi")
        wyc, _ = wytnij_z_pozycja(klatka, roi)
        maska = maska_tekstu(wyc, self._prog_jasnosci())
        udzial = float(maska.mean())
        self.udzial_maski = udzial
        obecny = udzial >= PROG_OBECNOSCI
        zmiana = (s.poprz_maska is None or s.poprz_maska.shape != maska.shape or
                  float(np.count_nonzero(s.poprz_maska ^ maska)) / maska.size > float(self.ust.get("czulosc", 0.004)))
        s.poprz_maska = maska

        if obecny:
            s.t_pusto = None
        elif s.t_pusto is None:
            s.t_pusto = ts
        elif ts - s.t_pusto > 0.5 and s.maska_widzi_napis:
            # maska przestała widzieć napis – ale potrafi go chwilę nie widzieć na jasnym tle,
            # więc pamięć czyścimy dopiero, gdy OCR potwierdzi, że obszar jest pusty
            s.wymus_odczyt = True
            s.maska_widzi_napis = False

        if zmiana and obecny and not s.czeka and teraz >= s.chlodzenie_do:
            s.czeka, s.t_zmiany, s.kandydat, s.zgodnych, s.proby, s.ocr_ms_suma = True, ts, "", 0, 0, 0.0
        if (not s.czeka and not s.wymus_odczyt and
                teraz - s.t_ost_ocr < (KONTROLA_GDY_TEKST if obecny else KONTROLA_GDY_PUSTO)):
            return
        if self.ocr.silnik is None:      # OCR jeszcze się ładuje
            time.sleep(0.05)
            return
        s.wymus_odczyt = False

        try:
            ms, pola, roi_uzyty = self._ocr(klatka, roi)
        except Exception as e:
            s.bledy_z_rzedu += 1
            self.bledy_ocr += 1
            s.t_ost_ocr = teraz
            self._zglos_blad("Błąd OCR", e)
            if s.bledy_z_rzedu >= 3 and self.przeladuj_ocr is not None:
                self.log("OCR zawiódł kilka razy z rzędu – ładuję silnik OCR od nowa.")
                s.bledy_z_rzedu = 0
                try:
                    self.przeladuj_ocr()
                except Exception as e2:
                    self._zglos_blad("Ponowne ładowanie OCR", e2)
            time.sleep(0.3)
            return
        s.bledy_z_rzedu = 0
        s.t_ost_ocr = self.t_ost_ocr = time.perf_counter()
        self._licz_ocr += 1
        if s.t_ost_ocr - self._t_licz >= 1.0:
            self.ocr_na_sekunde = self._licz_ocr / (s.t_ost_ocr - self._t_licz)
            self._licz_ocr, self._t_licz = 0, s.t_ost_ocr

        # filtr ruchu: porównanie z poprzednim odczytem (napisy stoją, tablice jadą)
        H, W = klatka.shape[:2]
        if self.ust.get("filtr_tablic", True) and s.t_ost_ocr - s.t_poprz_pola <= RUCH_MAKS_ODSTEP:
            oznacz_ruch(pola, s.poprz_pola, W, H)
        s.poprz_pola = [p for p in pola if p.get("odrzucone") in (None, "ruch")]
        s.t_poprz_pola = s.t_ost_ocr
        linie = self._linie(pola, klatka)
        tekst = skladaj_linie(linie)
        self.ostatni_odczyt, self.ostatni_ocr_ms = tekst, ms
        self.pola_podgladu = {"ts": s.t_ost_ocr, "pola": pola}
        self.odrzucone = opis_odrzuconych(pola)
        if self.diagnostyka is not None and self.odrzucone:
            self.diagnostyka.odrzucone(klatka, roi_uzyty, pola)
        pewnosci = [p["pewnosc"] for p in pola if not p.get("odrzucone")]

        if not wyglada_na_napis(tekst, pewnosci):
            if s.czeka and s.proby == 0:
                s.chlodzenie_do = teraz + CHLODZENIE_PO_RUCHU   # ruszało się tylko tło
            s.czeka = False
            self._anuluj_wstepna(s)
            if s.t_ocr_pusto is None:
                s.t_ocr_pusto = ts
            # pusto: potwierdzone dwoma odczytami OCR albo OCR + maską
            if ts - s.t_ocr_pusto > 0.7 or (not obecny and s.t_pusto is not None and ts - s.t_pusto > 0.5):
                if self.ostatnio_przeczytane or len(self.pamiec):
                    self.ostatnio_przeczytane = ""
                    self.pamiec.wyczysc()
            return
        s.t_ocr_pusto = None
        bez_powtorek = bool(self.ust.get("bez_powtorek", True))
        if bez_powtorek:
            self.pamiec.odswiez(linie, s.t_ost_ocr)
            szczegoly = self.pamiec.szczegoly(linie)
        else:
            szczegoly = [(l, l) for l in linie]
        nowe = [n for _, n in szczegoly if n]
        nowy_tekst = skladaj_linie(nowe)
        duplikat = (podobienstwo(tekst, self.ostatnio_przeczytane) >= float(self.ust.get("prog_duplikatu", 90))
                    or not nowe or not wyglada_na_napis(nowy_tekst, pewnosci))
        if duplikat:
            if s.czeka and s.proby == 0:
                s.chlodzenie_do = teraz + CHLODZENIE_PO_RUCHU   # ruszało się tylko tło
            s.czeka = False
            s.maska_widzi_napis = s.maska_widzi_napis or obecny
            self._anuluj_wstepna(s)
            return

        # nowy tekst – stabilizacja: `wymagane` kolejnych zgodnych odczytów
        if not s.czeka:                  # zauważył go dopiero odczyt kontrolny
            s.czeka, s.t_zmiany, s.kandydat, s.zgodnych, s.proby, s.ocr_ms_suma = True, ts, "", 0, 0, 0.0
            s.wstepne_starty = 0
        s.proby += 1
        s.ocr_ms_suma += ms
        wymagane = max(1, int(self.ust.get("stabilizacja", 2)))
        if s.kandydat and podobienstwo(tekst, s.kandydat) >= 92:
            s.zgodnych += 1
        else:
            s.kandydat, s.zgodnych = tekst, 1
            # nowy kandydat: synteza rusza od razu, a lektor odtworzy ją dopiero po potwierdzeniu
            # (stabilizacja to kilka odczytów OCR, ok. 100-150 ms – tyle zyskujemy). Najwyżej
            # MAKS_STARTOW_WSTEPNYCH razy na epizod: gdy napis dalej się zmienia (np. pojawia się
            # stopniowo), kolejne starty tylko zajmowałyby GPU tekstem, który i tak się nie przyda.
            if s.wstepna is not None:
                self._anuluj_wstepna(s)          # synteza poprzedniego kandydata jest już nieaktualna
            if (wymagane > 1 and self.ust.get("synteza_wstepna", True) and hasattr(self.lektor, "przygotuj")
                    and s.wstepne_starty < self.MAKS_STARTOW_WSTEPNYCH):
                wstepny = normalizuj(nowy_tekst, self.slownik_fn(), bool(self.ust.get("usun_imiona")))
                if wstepny and not (bez_powtorek and self._niedawno_powiedziane(wstepny)):
                    s.wstepna = self.lektor.przygotuj(wstepny, {"t_klatki": s.t_zmiany, "surowy": tekst})
                    s.wstepne_starty += 1
        if s.zgodnych < wymagane and s.proby < wymagane + 4:
            return
        s.czeka = False
        self.ostatnio_przeczytane = tekst
        s.maska_widzi_napis = obecny
        self._zapamietaj_poszerzenie(roi_uzyty, linie)
        # pominięte: całe linie albo początki linii, które lektor już przeczytał
        pominiete = [l if n is None else l[:max(0, len(l) - len(n))].strip()
                     for l, n in szczegoly if n != l]
        pominiete = [p for p in pominiete if p]
        if bez_powtorek:
            self.pominiete_linie += len(pominiete)
            self.pamiec.dodaj(linie, s.t_ost_ocr)
        do_przeczytania = normalizuj(nowy_tekst, self.slownik_fn(), bool(self.ust.get("usun_imiona")))
        if not do_przeczytania:
            self._anuluj_wstepna(s)
            return
        if bez_powtorek and self._niedawno_powiedziane(do_przeczytania):
            self._anuluj_wstepna(s)
            self.log(f"Pominięto powtórkę: „{do_przeczytania}” (ta sama kwestia przed chwilą).")
            if self.diagnostyka is not None:
                self.diagnostyka.odczyt(klatka, roi_uzyty, pola, "powtorka", tekst, do_przeczytania, pominiete)
            return
        if pominiete:
            self.log(f"Bez powtórki: pominąłem już przeczytane {', '.join('„' + p + '”' for p in pominiete)}.")
        self._wypowiedziane.append((time.perf_counter(), norm_linii(do_przeczytania)))
        if self.diagnostyka is not None:
            self.diagnostyka.odczyt(klatka, roi_uzyty, pola, "przeczytano", tekst, do_przeczytania, pominiete)
        meta = {"t_klatki": s.t_zmiany, "ocr_ms": round(s.ocr_ms_suma), "surowy": tekst, "pominiete": pominiete}
        if s.wstepna is not None:
            klucz, s.wstepna = s.wstepna, None
            if self.lektor.potwierdz(klucz, do_przeczytania, meta):
                self.wstepne_trafione += 1
            return
        self.lektor.powiedz(do_przeczytania, meta)

    def _anuluj_wstepna(self, s: _Stan):
        if s.wstepna is not None:
            klucz, s.wstepna = s.wstepna, None
            try:
                self.lektor.anuluj(klucz)
            except Exception as e:
                self._zglos_blad("Anulowanie wstępnej syntezy", e)

    def _niedawno_powiedziane(self, tekst: str) -> bool:
        teraz = time.perf_counter()
        n = norm_linii(tekst)
        return any(teraz - t <= OKNO_POWTORKI and podobienstwo(n, w) >= PROG_POWTORKI
                   for t, w in self._wypowiedziane)
