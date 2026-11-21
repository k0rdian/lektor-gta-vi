#!/usr/bin/env python3
"""Lektor GTA VI v2 – aplikacja testowa.

Obraz (Elgato / ekran / plik / zrzut) -> PaddleOCR na silniku MNN -> Piper TTS albo Chatterbox
(klonowanie głosu z nagrania, GPU).
Uruchom:  python aplikacja.py   (interfejs otworzy się w przeglądarce)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import statistics
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BAZA = Path(__file__).resolve().parent
sys.path.insert(0, str(BAZA))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lektor import chatterbox_pliki as cbp  # noqa: E402
from lektor import zrodla  # noqa: E402
from lektor.diagnostyka import Diagnostyka  # noqa: E402
from lektor.glos import GLOSY, Glos, GlosPiper, Lektor, Odtwarzacz, lista_glosow, urzadzenia_audio  # noqa: E402
from lektor.ocr import MODELE_OCR, Ocr  # noqa: E402
from lektor.potok import Potok, wytnij  # noqa: E402
from lektor.tekst import normalizuj, wczytaj_slownik  # noqa: E402

PLIK_USTAWIEN = BAZA / "ustawienia.json"
PLIK_SLOWNIKA = BAZA / "slownik_wymowy.txt"
KATALOG_TESTOW = BAZA / "testy"
KATALOG_PROBEK = BAZA / "probki"
KATALOG_DIAGNOSTYKI = BAZA / "diagnostyka"
KATALOG_NAGRAN = BAZA / "nagrania"
FORMATY_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".wma", ".mp4", ".mkv",
                 ".webm", ".mov", ".avi"}

DOMYSLNE = {
    "zrodlo": "kamera",          # kamera | ekran | plik | obraz
    "kamera_indeks": 0,
    "kamera_backend": "dshow",   # dshow | msmf | auto
    "ekran_nr": 1,
    "plik_wideo": "",
    "obraz": "testy/klatka1.png",
    "roi": [0.12, 0.74, 0.88, 0.97],
    "ocr_model": "v6_small_mnn",
    "ocr_rozdzielczosc": 960,
    "min_pewnosc": 0.6,
    "czulosc": 0.004,
    "jasnosc_tekstu": 200,       # używane tylko przy wyłączonym progu automatycznym
    "prog_auto": True,           # automatyczne wykrywanie tekstu (szare napisy GTA VI)
    "auto_poszerzanie": True,    # poszerzaj obszar, gdy napis wychodzi poza ramkę
    "stabilizacja": 2,
    "prog_duplikatu": 90,
    "usun_imiona": False,
    "bez_powtorek": True,        # nie czytaj drugi raz linii, która już była przeczytana i wciąż jest na ekranie
    "synteza_wstepna": True,     # zaczynaj syntezę po 1. odczycie, graj po potwierdzeniu (stabilizacji)
    "filtr_tablic": True,        # odrzucaj tablice rejestracyjne i tekst, który się przesuwa
    "tylko_srodek": True,        # czytaj tylko linie wyśrodkowane (napisy GTA VI)
    "filtr_ciemny": False,       # odrzucaj ciemny tekst na jasnym tle (eksperymentalne)
    "zapis_diagnostyki": False,  # zapisuj wycinki z odczytami do folderu diagnostyka/
    "glos": "meski",             # klucz głosu Pipera albo cb:<nazwa> dla Chatterboxa
    "predkosc": 1.0,
    "glosnosc": 1.0,
    "urzadzenie_audio": None,
    "polityka": "kolejka",       # kolejka (czytaj wszystko po kolei) | dokoncz | przerwij
    "przyspieszaj": True,        # przy zaległościach mów szybciej (do tempo_maks)
    "tempo_maks": 1.5,           # najwyższe tempo przy nadrabianiu zaległości
    "kolejka_maks_s": 30.0,      # bezpiecznik: kwestia czekająca dłużej jest pomijana
    "cb_model": "v3",            # Chatterbox Multilingual: v3 | v2
    "cb_tryb": "zrownowazony",   # najszybszy | zrownowazony | jakosc (strumieniowanie, dekoder, precyzja)
    "cb_urzadzenie": "auto",     # auto | cuda | cpu
    "cb_ekspresja": 0.5,         # „exaggeration” – emocje/ekspresja
    "cb_cfg": 0.5,               # „cfg_weight” – niżej = wolniej i spokojniej, 0 = bez akcentu referencji
    "cb_temperatura": 0.8,
}


class Ustawienia:
    def __init__(self, plik: Path):
        self.plik = plik
        self._lock = threading.Lock()
        self._d = dict(DOMYSLNE)
        if plik.exists():
            try:
                self._d.update(json.loads(plik.read_text(encoding="utf-8")))
            except Exception:
                pass

    def get(self, klucz, domyslna=None):
        with self._lock:
            return self._d.get(klucz, domyslna)

    def wszystkie(self) -> dict:
        with self._lock:
            return dict(self._d)

    def aktualizuj(self, zmiany: dict) -> set:
        zmienione = set()
        with self._lock:
            for k, v in zmiany.items():
                if k in DOMYSLNE and self._d.get(k) != v:
                    self._d[k] = v
                    zmienione.add(k)
            if zmienione:
                self._zapisz(json.dumps(self._d, ensure_ascii=False, indent=2))
        return zmienione

    def _zapisz(self, tresc: str):
        """Zapis przez plik tymczasowy, z ponowieniami. Pulpit bywa synchronizowany przez OneDrive,
        który potrafi na chwilę zablokować plik – wtedy ustawienie i tak działa (jest w pamięci),
        a zapis na dysk ponawiamy przy następnej zmianie."""
        tmp = self.plik.with_suffix(".json.tmp")
        for proba in range(5):
            try:
                tmp.write_text(tresc, encoding="utf-8")
                os.replace(tmp, self.plik)
                return
            except OSError as e:
                blad = e
                time.sleep(0.05 * (proba + 1))
        print(f"Nie udało się zapisać {self.plik.name}: {blad}", flush=True)


class Nagrywarka:
    """Nagrywa obraz ze źródła do MP4 (30 kl./s, w czasie rzeczywistym), np. scenę z błędnym
    odczytem – potem można ją odtwarzać w kółko jako „Plik wideo” i stroić ustawienia.
    Przydaje się szczególnie z Elgato, którego nie da się jednocześnie używać w OBS."""

    FPS = 30
    MAKS_S = 300

    def __init__(self, katalog: Path, log):
        self.katalog = katalog
        self.log = log
        self.plik: Path | None = None
        self.t_start = 0.0
        self._dziala = False
        self._watek = None

    @property
    def nagrywa(self) -> bool:
        return self._dziala

    def stan(self) -> dict:
        return {"nagrywa": self._dziala, "plik": self.plik.name if self.plik else "",
                "sciezka": str(self.plik) if self.plik else "",
                "czas_s": round(time.perf_counter() - self.t_start) if self._dziala else 0}

    def start(self, zrodlo):
        if self._dziala:
            return
        klatka, _ = zrodlo.ostatnia_klatka()
        if klatka is None:
            raise RuntimeError("Brak obrazu ze źródła – najpierw kliknij Start")
        self.katalog.mkdir(exist_ok=True)
        self.plik = self.katalog / time.strftime("nagranie_%Y-%m-%d_%H%M%S.mp4")
        h, w = klatka.shape[:2]
        zapis = cv2.VideoWriter(str(self.plik), cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (w, h))
        if not zapis.isOpened():
            raise RuntimeError("Nie mogę utworzyć pliku nagrania")
        self._dziala = True
        self.t_start = time.perf_counter()
        self._watek = threading.Thread(target=self._petla, args=(zrodlo, zapis, (w, h)), daemon=True)
        self._watek.start()
        self.log(f"Nagrywam obraz do {self.plik.name} (najwyżej {self.MAKS_S // 60} min).")

    def stop(self):
        self._dziala = False
        if self._watek is not None:
            self._watek.join(timeout=3)
            self._watek = None

    def _petla(self, zrodlo, zapis, rozmiar):
        n = 0
        try:
            while self._dziala and time.perf_counter() - self.t_start < self.MAKS_S:
                cel = self.t_start + n / self.FPS
                reszta = cel - time.perf_counter()
                if reszta > 0:
                    time.sleep(reszta)
                klatka, _ = zrodlo.ostatnia_klatka()
                if klatka is not None:
                    if (klatka.shape[1], klatka.shape[0]) != rozmiar:
                        klatka = cv2.resize(klatka, rozmiar)
                    zapis.write(klatka)
                n += 1
        except Exception as e:
            self.log(f"Błąd nagrywania: {e}")
        finally:
            zapis.release()
            self._dziala = False
            if self.plik is not None:
                self.log(f"Zapisano nagranie {self.plik.name} ({n / self.FPS:.0f} s). "
                         f"Odtworzysz je jako „Plik wideo”: {self.plik}")


class Aplikacja:
    def __init__(self):
        self.ust = Ustawienia(PLIK_USTAWIEN)
        self.dziennik = collections.deque(maxlen=200)
        self.historia = collections.deque(maxlen=60)
        self.ocr = Ocr(self.log)
        self.glos = Glos(self.ust, self.log)
        self.odtwarzacz = Odtwarzacz(self.log)
        self.lektor = Lektor(self.glos, self.odtwarzacz, self.ust, self.log, self._po_kwestii)
        self._slownik, self._slownik_mtime = [], None
        self.diagnostyka = Diagnostyka(KATALOG_DIAGNOSTYKI, self.ust, self.log)
        self.potok = Potok(self.ocr, self.lektor, self.ust, self.slownik, self.log,
                           przeladuj_ocr=self.ocr.przeladuj, diagnostyka=self.diagnostyka)
        self.nagrywarka = Nagrywarka(KATALOG_NAGRAN, self.log)
        self.ladowanie = {"ocr": "", "glos": ""}
        self.tworzenie_glosu = ""
        self.klienci_podgladu = 0
        self.blad_startu = ""
        self._urzadzenia = None
        self._lock_glosu = threading.Lock()
        threading.Thread(target=self._straznik, name="straznik", daemon=True).start()

    def _straznik(self):
        """Gdy proces Chatterboxa padnie w trakcie gry, kwestię czyta zapasowy Piper, a tu
        Chatterbox wczytuje się od nowa w tle (najwyżej raz na 30 s)."""
        ostatnio = 0.0
        while True:
            time.sleep(2.0)
            try:
                if (self.glos.awaria and str(self.ust.get("glos")).startswith("cb:")
                        and not self.ladowanie["glos"].startswith(("ładuję", "pobieram"))
                        and time.perf_counter() - ostatnio > 30):
                    self.glos.awaria = False
                    ostatnio = time.perf_counter()
                    self.log("Proces Chatterboxa przestał działać – wczytuję go ponownie.")
                    self.zaladuj_glos()
            except Exception as e:
                self.log(f"Strażnik: {e}")

    # --- narzędzia ---
    def log(self, msg: str):
        linia = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.dziennik.append(linia)
        print(linia, flush=True)

    def slownik(self):
        try:
            mtime = PLIK_SLOWNIKA.stat().st_mtime
        except OSError:
            return []
        if mtime != self._slownik_mtime:
            self._slownik = wczytaj_slownik(PLIK_SLOWNIKA)
            self._slownik_mtime = mtime
        return self._slownik

    def _po_kwestii(self, w: dict):
        self.historia.appendleft(w)
        opis = f"„{w['tekst']}”  OCR {w['ocr_ms']} ms · {w.get('silnik', 'Piper')} {w['tts_ms']} ms"
        if w.get("calosc_ms") is not None:
            opis += f" · od zmiany napisu do głosu {w['calosc_ms']} ms"
        self.log(opis)

    # --- ładowanie modeli w tle ---
    def zaladuj_ocr(self, czekaj=False):
        def praca():
            self.ladowanie["ocr"] = "ładuję…"
            try:
                self.ocr.zaladuj(self.ust.get("ocr_model"), int(self.ust.get("ocr_rozdzielczosc")))
                self.ladowanie["ocr"] = ""
            except Exception as e:
                self.ladowanie["ocr"] = f"błąd: {e}"
                self.log(f"Nie udało się załadować OCR: {e}")
        if czekaj:
            praca()
        else:
            threading.Thread(target=praca, daemon=True).start()

    def zaladuj_glos(self, czekaj=False):
        def postep(p):
            self.ladowanie["glos"] = f"pobieram {p * 100:.0f}%"

        def etap(opis):
            self.ladowanie["glos"] = opis

        def praca():
            # jedno ładowanie naraz (szybkie przełączanie głosów w interfejsie nie może uruchomić
            # dwóch procesów Chatterboxa); po zakończeniu sprawdzamy, czy wybór się nie zmienił
            with self._lock_glosu:
                while True:
                    klucz = self.ust.get("glos")
                    self.ladowanie["glos"] = "ładuję…"
                    try:
                        self.glos.zaladuj(klucz, postep, etap)
                        self.ladowanie["glos"] = ""
                    except Exception as e:
                        self.ladowanie["glos"] = f"błąd: {e}"
                        self.log(f"Nie udało się załadować głosu: {e}")
                        return
                    if self.ust.get("glos") == klucz:
                        return
        if czekaj:
            praca()
        else:
            threading.Thread(target=praca, daemon=True).start()

    def utworz_glos(self, plik: Path, nazwa: str, od=None, do=None) -> dict:
        self.tworzenie_glosu = "tworzę głos (pierwszy raz ładuje model, do ~40 s)…"
        try:
            meta = self.glos.utworz_glos(plik, nazwa, od, do)
        finally:
            self.tworzenie_glosu = ""
        return meta

    def usun_glos(self, klucz: str):
        if not klucz.startswith("cb:") or klucz == cbp.DOMYSLNY_GLOS:
            raise ValueError("Można usunąć tylko własny głos Chatterbox")
        katalog = cbp.sciezka_conds(klucz).parent
        kosz = cbp.KATALOG_GLOSOW / "_usuniete"
        kosz.mkdir(parents=True, exist_ok=True)
        cel = kosz / f"{katalog.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(katalog), str(cel))          # do „kosza” w glosy/_usuniete, nie na zawsze
        self.log(f"Usunięto głos {katalog.name} (kopia w glosy/_usuniete/{cel.name}).")
        if self.ust.get("glos") == klucz:
            self.zmien_ustawienia({"glos": "meski"})

    # --- źródło i start ---
    def zbuduj_zrodlo(self):
        u = self.ust
        typ = u.get("zrodlo")
        if typ == "kamera":
            return zrodla.ZrodloKamera(int(u.get("kamera_indeks")), u.get("kamera_backend"), log=self.log)
        if typ == "ekran":
            return zrodla.ZrodloEkran(int(u.get("ekran_nr")), log=self.log)
        if typ == "plik":
            sciezka = str(u.get("plik_wideo") or "").strip().strip('"')
            if not sciezka:
                raise ValueError("Podaj ścieżkę do pliku wideo")
            return zrodla.ZrodloPlik(sciezka, log=self.log)
        if typ == "obraz":
            sciezka = Path(str(u.get("obraz") or ""))
            if not sciezka.is_absolute():
                sciezka = BAZA / sciezka
            return zrodla.ZrodloObraz(zrodla.wczytaj_obraz(sciezka), sciezka.name, log=self.log)
        raise ValueError(f"Nieznane źródło: {typ}")

    def start(self):
        self.blad_startu = ""
        if self.ocr.silnik is None and not self.ladowanie["ocr"].startswith("ładuję"):
            self.zaladuj_ocr(czekaj=True)
        if self.glos.voice is None and not self.ladowanie["glos"].startswith(("ładuję", "pobieram")):
            self.zaladuj_glos()
        try:
            self.potok.start(self.zbuduj_zrodlo())
            self.log(f"Start: {self.potok.zrodlo.opis}.")
        except Exception as e:
            self.blad_startu = str(e)
            self.log(f"Nie udało się wystartować: {e}")

    def stop(self):
        if self.nagrywarka.nagrywa:
            self.nagrywarka.stop()
        self.potok.stop()
        self.lektor.ucisz()
        self.log("Zatrzymano.")

    def zmien_ustawienia(self, zmiany: dict):
        zmienione = self.ust.aktualizuj(zmiany)
        if "roi" in zmienione:
            self.potok.wyczysc_ramki()      # stare ramki dotyczyły poprzedniego obszaru
        if zmienione & {"ocr_model", "ocr_rozdzielczosc"}:
            self.zaladuj_ocr()
        if "glos" in zmienione or (zmienione & {"cb_model", "cb_urzadzenie", "cb_tryb"}
                                   and str(self.ust.get("glos")).startswith("cb:")):
            self.zaladuj_glos()
        zrodlowe = {"zrodlo", "kamera_indeks", "kamera_backend", "ekran_nr", "plik_wideo", "obraz"}
        if zmienione & zrodlowe and self.potok.stan == "działa":
            self.start()
        return sorted(zmienione)

    # --- stan dla interfejsu ---
    def stan(self) -> dict:
        hist = list(self.historia)

        def mediana(klucz):
            v = [h[klucz] for h in hist[:30] if isinstance(h.get(klucz), (int, float))]
            return round(statistics.median(v)) if v else None

        zr = self.potok.zrodlo
        if self._urzadzenia is None:
            self._urzadzenia = urzadzenia_audio()
        return {
            "dziala": self.potok.stan == "działa",
            "zrodlo": zr.opis if zr else "",
            "fps": round(zr.fps, 1) if zr else 0,
            "blad_zrodla": (zr.blad if zr else "") or self.blad_startu,
            "ocr_na_s": round(self.potok.ocr_na_sekunde, 1),
            "odczyt": self.potok.ostatni_odczyt,
            "odczyt_ms": round(self.potok.ostatni_ocr_ms, 1),
            "diagnostyka": {
                "maska_proc": round(self.potok.udzial_maski * 100, 2),
                "ocr_wiek_s": round(time.perf_counter() - self.potok.t_ost_ocr, 1) if self.potok.t_ost_ocr else None,
                "bledy_ocr": self.potok.bledy_ocr,
                "restarty_zrodla": self.potok.restarty_zrodla,
                "klienci_podgladu": self.klienci_podgladu,
                "pominiete_linie": self.potok.pominiete_linie,
                "wstepne_trafione": self.potok.wstepne_trafione,
                "pamietane_linie": len(self.potok.pamiec),
                "zapisane_diagnostyki": self.diagnostyka.zapisane,
            },
            "odrzucone": self.potok.odrzucone[:6] if self.potok.t_ost_ocr and
                         time.perf_counter() - self.potok.t_ost_ocr < 3 else [],
            "mowi": self.lektor.mowi,
            "lektor": self.lektor.stan(),
            "atrapa_audio": self.odtwarzacz.atrapa,
            "wyjscie_audio": self.odtwarzacz.opis,
            "ladowanie": self.ladowanie,
            "ocr_gotowy": self.ocr.silnik is not None,
            "glos_gotowy": self.glos.voice is not None,
            "silnik_glosu": self.glos.nazwa_silnika,
            "chatterbox": {
                "zainstalowany": cbp.zainstalowany(),
                "info": self.glos.chatterbox.info,
                "czasy": self.glos.chatterbox.ostatnie_czasy,
                "tworzenie": self.tworzenie_glosu,
                "modele": [{"id": k, "nazwa": v, "pobrany": not cbp.brakujace_pliki(k)}
                           for k, v in cbp.NAZWY_MODELI.items()],
            },
            "nagrywanie": self.nagrywarka.stan(),
            "historia": hist[:25],
            "statystyki": {"calosc_ms": mediana("calosc_ms"), "ocr_ms": mediana("ocr_ms"),
                           "tts_ms": mediana("tts_ms"), "kwestii": len(hist)},
            "dziennik": list(self.dziennik)[-40:],
            "ustawienia": self.ust.wszystkie(),
            "listy": {
                "glosy": lista_glosow(),
                "modele_ocr": [{"id": k, "nazwa": v} for k, v in MODELE_OCR.items()],
                "audio": self._urzadzenia,
                "obrazy": sorted(p.name for p in KATALOG_TESTOW.glob("*.png")) +
                          sorted(p.name for p in KATALOG_TESTOW.glob("*.jpg")),
            },
        }

    def podglad_jpeg(self, szer: int = 960) -> bytes | None:
        zr = self.potok.zrodlo
        if zr is None:
            return None
        klatka, _ = zr.ostatnia_klatka()
        if klatka is None:
            return None
        h, w = klatka.shape[:2]
        mw, mh = szer, max(1, int(h * szer / w))
        maly = cv2.resize(klatka, (mw, mh), interpolation=cv2.INTER_AREA)
        # ramki rozpoznanego tekstu (zielone = przyjęte, żółte = poniżej progu pewności,
        # czerwone = odrzucone przez filtry: tablica, ruch, poza środkiem…);
        # zapisane we współrzędnych całej klatki, więc trafiają w tekst niezależnie od obszaru
        for p in self.potok.ramki_do_podgladu():
            powod = p.get("odrzucone")
            if powod in ("niska pewność", "bez liter"):
                kolor = (0, 200, 255)
            elif powod:
                kolor = (60, 60, 255)
            else:
                kolor = (80, 220, 80)
            a = (int(p["x0"] * mw), int(p["y0"] * mh))
            b = (int(p["x1"] * mw), int(p["y1"] * mh))
            cv2.rectangle(maly, a, b, kolor, 2)
            if powod and kolor == (60, 60, 255):
                cv2.putText(maly, _bez_ogonkow(powod), (a[0], max(12, a[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, kolor, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", maly, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return buf.tobytes() if ok else None


# ---------------------------------------------------------------- serwer HTTP

APP: Aplikacja | None = None


def _bez_ogonkow(t: str) -> str:
    """cv2.putText nie umie polskich znaków."""
    return t.translate(str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ"))


class Obsluga(BaseHTTPRequestHandler):
    """Serwer interfejsu. HTTP/1.1 z utrzymywaniem połączeń (keep-alive): na tym komputerze
    filtr sieciowy (AdGuard) zrywał ok. 13% nowych połączeń z 127.0.0.1 – stąd losowe
    „Failed to fetch” i zawieszony podgląd. Przeglądarka używa teraz kilku stałych połączeń
    zamiast otwierać nowe co 0,7 s (test: 180/180 zapytań bez błędu)."""

    server_version = "LektorGTA/2"
    protocol_version = "HTTP/1.1"
    timeout = 60                     # bezczynne połączenie zamykamy po minucie

    def log_message(self, fmt, *args):  # cisza w konsoli
        pass

    def _json(self, obj, kod=200):
        dane = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(kod)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(dane)))
        if kod >= 400:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(dane)

    def _cialo(self) -> bytes:
        """Treść żądania – czytana raz (przy keep-alive nieprzeczytane bajty zepsułyby
        następne żądanie na tym samym połączeniu)."""
        if getattr(self, "_wczytane", None) is None:
            n = int(self.headers.get("Content-Length") or 0)
            self._wczytane = self.rfile.read(n) if n else b""
        return self._wczytane

    def _plik(self, p: Path, typ: str):
        dane = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", typ)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(dane)))
        self.end_headers()
        self.wfile.write(dane)

    def _html(self, tresc: str):
        dane = tresc.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(dane)))
        self.end_headers()
        self.wfile.write(dane)

    def do_GET(self):
        url = urlparse(self.path)
        sciezka = url.path
        try:
            if sciezka.startswith("/glosy/") and sciezka.endswith("/referencja.wav"):
                nazwa = unquote(sciezka.split("/")[2])
                if "/" in nazwa or "\\" in nazwa or ".." in nazwa:
                    self.send_error(404)
                    return
                p = cbp.KATALOG_GLOSOW / nazwa / "referencja.wav"
                if not p.exists():
                    self.send_error(404)
                    return
                self._plik(p, "audio/wav")
                return
            if sciezka == "/diagnostyka":
                dzien = (parse_qs(url.query).get("dzien") or [None])[0]
                self._html(APP.diagnostyka.strona(dzien))
                return
            if sciezka.startswith("/diagnostyka/"):
                czesci = sciezka.split("/")
                p = APP.diagnostyka.plik(czesci[2], czesci[3]) if len(czesci) == 4 else None
                if p is None:
                    self.send_error(404)
                    return
                self._plik(p, "image/jpeg")
                return
            if sciezka in ("/", "/index.html"):
                dane = (BAZA / "ui" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(dane)))
                self.end_headers()
                self.wfile.write(dane)
            elif sciezka == "/api/stan":
                self._json(APP.stan())
            elif sciezka == "/podglad.jpg":
                jpg = APP.podglad_jpeg()
                if jpg is None:
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            elif sciezka == "/podglad.mjpg":
                self._mjpeg()
            elif sciezka == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True

    def _mjpeg(self):
        # strumień bez długości – po nim połączenie musi się zamknąć
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=klatka")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        APP.klienci_podgladu += 1
        t_bledu = 0.0
        try:
            while True:
                try:
                    jpg = APP.podglad_jpeg()
                except Exception as e:      # błąd rysowania jednej klatki nie może zerwać podglądu
                    if time.perf_counter() - t_bledu > 30:
                        APP.log(f"Błąd podglądu: {e}")
                        t_bledu = time.perf_counter()
                    jpg = None
                if jpg is not None:
                    self.wfile.write(b"--klatka\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.08)
        finally:
            APP.klienci_podgladu -= 1

    def _utworz_glos(self):
        """Wgrane nagranie (dowolny format audio/wideo) -> głos Chatterbox w glosy/<nazwa>/."""
        nazwa_pliku = Path(unquote(self.headers.get("X-Plik") or "nagranie.wav")).name
        nazwa = unquote(self.headers.get("X-Nazwa") or "").strip() or Path(nazwa_pliku).stem
        roz = Path(nazwa_pliku).suffix.lower()
        n = int(self.headers.get("Content-Length") or 0)

        def odrzuc(komunikat):
            # przeczytaj (i wyrzuć) wysyłany plik, żeby przeglądarka dostała komunikat, a nie zerwane połączenie
            zostalo = n if n <= 1024 * 1024 * 1024 else 0
            while zostalo > 0:
                blok = self.rfile.read(min(1 << 20, zostalo))
                if not blok:
                    break
                zostalo -= len(blok)
            raise ValueError(komunikat)

        if roz not in FORMATY_AUDIO:
            odrzuc(f"Nieobsługiwany format {roz or '(brak)'} – użyj np. mp3, wav, m4a, ogg, flac")

        def liczba(naglowek):
            v = (self.headers.get(naglowek) or "").replace(",", ".").strip()
            return float(v) if v else None

        try:
            od, do = liczba("X-Od"), liczba("X-Do")
        except ValueError:
            odrzuc("Fragment od/do musi być liczbą sekund")
        if n <= 0:
            raise ValueError("Pusty plik")
        if n > 1024 * 1024 * 1024:
            raise ValueError("Plik większy niż 1 GB – wytnij krótszy fragment (wystarczy 15–60 s mowy)")
        wgrane = cbp.KATALOG_GLOSOW / "_wgrane"
        wgrane.mkdir(parents=True, exist_ok=True)
        cel = wgrane / f"{time.strftime('%Y%m%d_%H%M%S')}_{cbp.slug(Path(nazwa_pliku).stem)}{roz}"
        with open(cel, "wb") as f:
            zostalo = n
            while zostalo > 0:
                blok = self.rfile.read(min(1 << 20, zostalo))
                if not blok:
                    break
                f.write(blok)
                zostalo -= len(blok)
        meta = APP.utworz_glos(cel, nazwa, od, do)
        self._json({"ok": True, **meta})

    def do_POST(self):
        sciezka = urlparse(self.path).path
        self._wczytane = None
        try:
            if sciezka != "/api/glos_chatterbox":     # nagranie głosu czyta się strumieniowo na dysk
                self._cialo()
            if sciezka == "/api/ustawienia":
                self._json({"zmienione": APP.zmien_ustawienia(json.loads(self._cialo() or b"{}"))})
            elif sciezka == "/api/start":
                APP.start()
                self._json({"ok": not APP.blad_startu, "blad": APP.blad_startu})
            elif sciezka == "/api/stop":
                APP.stop()
                self._json({"ok": True})
            elif sciezka == "/api/ucisz":
                APP.lektor.ucisz()
                self._json({"ok": True})
            elif sciezka == "/api/pomin":
                APP.lektor.pomin()
                self._json({"ok": True})
            elif sciezka == "/api/kamery":
                self._json({"kamery": zrodla.wykryj_kamery(APP.ust.get("kamera_backend"))})
            elif sciezka == "/api/test_glosu":
                dane = json.loads(self._cialo() or b"{}")
                tekst = normalizuj(dane.get("tekst", ""), APP.slownik(), bool(APP.ust.get("usun_imiona")))
                if APP.glos.voice is None:
                    APP.zaladuj_glos(czekaj=True)
                if APP.glos.voice is None:
                    raise RuntimeError(APP.ladowanie.get("glos") or "Głos nie jest załadowany")
                APP.lektor.powiedz(tekst, {"t_klatki": time.perf_counter(), "ocr_ms": 0, "surowy": tekst})
                self._json({"ok": True, "tekst": tekst})
            elif sciezka == "/api/odczytaj":
                self._json(APP.potok.odczytaj_teraz())
            elif sciezka == "/api/obraz":
                nazwa = Path(self.headers.get("X-Nazwa") or "wgrany.png").name
                img = zrodla.wczytaj_obraz(self._cialo())   # rzuci wyjątek, jeśli to nie obraz
                cel = KATALOG_TESTOW / ("wgrany_" + "".join(c for c in Path(nazwa).stem if c.isalnum() or c in "-_")[:40] + ".png")
                ok, buf = cv2.imencode(".png", img)
                cel.write_bytes(buf.tobytes())
                APP.ust.aktualizuj({"zrodlo": "obraz", "obraz": f"testy/{cel.name}"})
                APP.start()
                self._json({"ok": True, "plik": cel.name})
            elif sciezka == "/api/glos_chatterbox":
                self._utworz_glos()
            elif sciezka == "/api/usun_glos":
                APP.usun_glos(json.loads(self._cialo() or b"{}").get("id", ""))
                self._json({"ok": True})
            elif sciezka == "/api/nagrywanie":
                wlacz = json.loads(self._cialo() or b"{}").get("wlacz")
                if wlacz:
                    if APP.potok.zrodlo is None or APP.potok.stan != "działa":
                        raise RuntimeError("Najpierw uruchom źródło obrazu (Start)")
                    APP.nagrywarka.start(APP.potok.zrodlo)
                else:
                    APP.nagrywarka.stop()
                self._json({"ok": True, **APP.nagrywarka.stan()})
            else:
                self.send_error(404)
        except Exception as e:
            APP.log(f"Błąd: {e}")
            self._json({"ok": False, "blad": str(e), "slad": traceback.format_exc()[-600:]}, 500)


# ---------------------------------------------------------------- tryby wiersza poleceń

def tryb_probki():
    """Generuje próbki wszystkich wbudowanych głosów Pipera do folderu probki/."""
    import wave
    zdania = [z.strip() for z in (KATALOG_TESTOW / "zdania.txt").read_text(encoding="utf-8").splitlines() if z.strip()]
    slownik = wczytaj_slownik(PLIK_SLOWNIKA)
    KATALOG_PROBEK.mkdir(exist_ok=True)
    raport = ["Próbki głosów Pipera (te same 10 kwestii, po normalizacji i słowniku wymowy)", ""]
    for klucz, g in GLOSY.items():
        glos = GlosPiper(print)
        glos.zaladuj(klucz)
        sr = glos.sample_rate
        cisza = np.zeros(int(0.7 * sr), dtype=np.int16)
        calosc, czasy_pierwsze, synteza, audio_s = [], [], 0.0, 0.0
        for z in zdania:
            tekst = normalizuj(z, slownik)
            t0 = time.perf_counter()
            pierwszy = None
            for i, (a, _sr, *_) in enumerate(glos.syntezuj(tekst)):
                if i == 0:
                    pierwszy = time.perf_counter() - t0
                calosc.append(a)
            synteza += time.perf_counter() - t0
            czasy_pierwsze.append(pierwszy * 1000)
            calosc.append(cisza)
        audio = np.concatenate(calosc)
        audio_s = len(audio) / sr
        plik = KATALOG_PROBEK / f"piper_{klucz}.wav"
        with wave.open(str(plik), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio.tobytes())
        raport.append(f"{plik.name:24s} {g['nazwa']:24s} pierwsze zdanie: mediana "
                      f"{statistics.median(czasy_pierwsze):.0f} ms · synteza {synteza:.2f} s na {audio_s:.0f} s audio")
        print(raport[-1])
    (KATALOG_PROBEK / "czasy.txt").write_text("\n".join(raport) + "\n", encoding="utf-8")


def tryb_probki_chatterbox(klucz: str):
    """Te same 10 kwestii co próbki Pipera, ale głosem Chatterbox – do porównania (probki/)."""
    import wave
    ust = Ustawienia(PLIK_USTAWIEN)
    if not klucz.startswith("cb:"):
        klucz = "cb:" + klucz
    zdania = [z.strip() for z in (KATALOG_TESTOW / "zdania.txt").read_text(encoding="utf-8").splitlines() if z.strip()]
    slownik = wczytaj_slownik(PLIK_SLOWNIKA)
    glos = Glos(ust, print)
    t0 = time.perf_counter()
    glos.zaladuj(klucz, etap=print)
    print(f"Głos gotowy po {time.perf_counter() - t0:.1f} s ({glos.chatterbox.info.get('gpu', '')}).")
    calosc, pierwsze, synteza, sr = [], [], 0.0, cbp.SR
    for z in zdania:
        tekst = normalizuj(z, slownik)
        t = time.perf_counter()
        for i, (a, sr, *_) in enumerate(glos.syntezuj(tekst, float(ust.get("predkosc", 1.0)))):
            if i == 0:
                pierwsze.append((time.perf_counter() - t) * 1000)
            calosc.append(a)
        synteza += time.perf_counter() - t
        calosc.append(np.zeros(int(0.7 * sr), dtype=np.int16))
        print(f"  {pierwsze[-1]:6.0f} ms do pierwszego zdania | {z}")
    audio = np.concatenate(calosc)
    KATALOG_PROBEK.mkdir(exist_ok=True)
    plik = KATALOG_PROBEK / f"chatterbox_{klucz[3:]}.wav"
    with wave.open(str(plik), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    print(f"Zapisano {plik}: pierwsze zdanie mediana {statistics.median(pierwsze):.0f} ms, "
          f"synteza {synteza:.1f} s na {len(audio) / sr:.0f} s audio.")
    glos.zamknij()


def tryb_test_ocr(sciezki: list[str], model: str):
    ocr = Ocr(print)
    ocr.zaladuj(model)
    roi = DOMYSLNE["roi"]
    slownik = wczytaj_slownik(PLIK_SLOWNIKA)
    from lektor.tekst import skladaj_linie
    for s in sciezki:
        img = zrodla.wczytaj_obraz(s)
        wyc = wytnij(img, roi)
        ocr.rozpoznaj(wyc)                      # pierwszy przebieg na rozgrzewkę
        linie, ms, _ = ocr.rozpoznaj(wyc)
        tekst = skladaj_linie(linie)
        print(f"{Path(s).name}: {ms:.0f} ms | {tekst} | -> {normalizuj(tekst, slownik)}")


def main():
    global APP
    ap = argparse.ArgumentParser(description="Lektor GTA VI v2 – aplikacja testowa")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bez-przegladarki", action="store_true")
    ap.add_argument("--probki", action="store_true", help="wygeneruj próbki głosów do folderu probki/")
    ap.add_argument("--probki-chatterbox", metavar="GLOS",
                    help="próbki głosu Chatterbox (np. domyslny albo nazwa własnego głosu z folderu glosy/)")
    ap.add_argument("--test-ocr", nargs="+", metavar="OBRAZ", help="OCR na obrazach (bez interfejsu)")
    ap.add_argument("--model", default="v6_small_mnn")
    a = ap.parse_args()

    if a.probki:
        tryb_probki()
        return
    if a.probki_chatterbox:
        tryb_probki_chatterbox(a.probki_chatterbox)
        return
    if a.test_ocr:
        tryb_test_ocr(a.test_ocr, a.model)
        return

    APP = Aplikacja()
    APP.log("Lektor GTA VI v2 – start. Ładuję OCR i głos w tle…")
    APP.zaladuj_ocr()
    APP.zaladuj_glos()
    serwer = None
    for port in range(a.port, a.port + 10):
        try:
            serwer = ThreadingHTTPServer(("127.0.0.1", port), Obsluga)
            break
        except OSError:
            continue
    if serwer is None:
        raise SystemExit("Nie znalazłem wolnego portu")
    serwer.daemon_threads = True
    url = f"http://127.0.0.1:{serwer.server_address[1]}/"
    APP.log(f"Interfejs: {url}  (zamknij to okno, żeby wyłączyć aplikację)")
    if not a.bez_przegladarki:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        serwer.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        APP.nagrywarka.stop()
        APP.potok.stop()
        APP.lektor.zakoncz()
        APP.odtwarzacz.zamknij()
        APP.glos.zamknij()


if __name__ == "__main__":
    main()
