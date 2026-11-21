"""Chatterbox: ścieżki, lista plików modeli i pobieranie z Hugging Face.

Tylko biblioteka standardowa – moduł importuje zarówno aplikacja (.venv), jak i proces
roboczy Chatterboxa (.venv-chatterbox)."""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

BAZA = Path(__file__).resolve().parents[1]
KATALOG_MODELI = BAZA / "modele" / "chatterbox"
KATALOG_GLOSOW = BAZA / "glosy"
PYTHON_CHATTERBOX = BAZA / ".venv-chatterbox" / "Scripts" / "python.exe"
SKRYPT_WORKERA = BAZA / "lektor" / "chatterbox_worker.py"
REPO = "ResembleAI/chatterbox"
URL_PLIKU = "https://huggingface.co/" + REPO + "/resolve/main/{}"

WSPOLNE = ["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt"]
MODELE_T3 = {
    "v3": "t3_mtl23ls_v3.safetensors",   # Multilingual V3 (2026) – lepsze podobieństwo głosu, mniej „halucynacji”
    "v2": "t3_mtl23ls_v2.safetensors",   # poprzednia wersja, do porównania
}
NAZWY_MODELI = {
    "v3": "Multilingual V3 (zalecany)",
    "v2": "Multilingual V2 (starszy)",
}
SR = 24000                     # częstotliwość próbkowania audio Chatterboxa
DOMYSLNY_GLOS = "cb:domyslny"  # wbudowany głos z conds.pt (nagrany po angielsku – po polsku mówi z lekkim akcentem)


def pliki_modelu(wersja: str) -> list[str]:
    return WSPOLNE + [MODELE_T3.get(wersja, MODELE_T3["v3"])]


def brakujace_pliki(wersja: str) -> list[str]:
    return [p for p in pliki_modelu(wersja) if not (KATALOG_MODELI / p).exists()]


def pobierz_modele(wersja: str = "v3", log=print, postep=None) -> None:
    """Pobiera brakujące pliki modelu (ok. 3,2 GB dla v3) do modele/chatterbox.
    postep(ułamek 0..1) – łączny postęp pobierania."""
    brak = brakujace_pliki(wersja)
    if not brak:
        return
    KATALOG_MODELI.mkdir(parents=True, exist_ok=True)
    rozmiary = {}
    for nazwa in brak:
        req = urllib.request.Request(URL_PLIKU.format(nazwa), method="HEAD", headers={"User-Agent": "lektor-gta"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                rozmiary[nazwa] = int(r.headers.get("Content-Length") or 0)
        except Exception:
            rozmiary[nazwa] = 0
    razem = sum(rozmiary.values()) or 1
    pobrane_razem = 0
    log(f"Pobieram model Chatterbox {wersja}: {', '.join(brak)} (ok. {razem / 1e9:.1f} GB)…")
    for nazwa in brak:
        cel = KATALOG_MODELI / nazwa
        tmp = KATALOG_MODELI / (nazwa + ".czesc")
        req = urllib.request.Request(URL_PLIKU.format(nazwa), headers={"User-Agent": "lektor-gta"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                blok = r.read(1 << 22)
                if not blok:
                    break
                f.write(blok)
                pobrane_razem += len(blok)
                if postep:
                    postep(min(1.0, pobrane_razem / razem))
        os.replace(tmp, cel)
        log(f"Pobrano {nazwa} ({cel.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s).")


# ---------------------------------------------------------------- głosy użytkownika

def slug(nazwa: str) -> str:
    t = unicodedata.normalize("NFKD", nazwa.replace("ł", "l").replace("Ł", "L"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^A-Za-z0-9_-]+", "_", t).strip("_").lower()
    return t[:40] or "glos"


def lista_glosow() -> list[dict]:
    """Głosy Chatterboxa: wbudowany + utworzone przez użytkownika (folder glosy/)."""
    wynik = [{"id": DOMYSLNY_GLOS, "nazwa": "Chatterbox · wbudowany głos modelu (anglojęzyczny)",
              "pobrany": not brakujace_pliki("v3")}]
    if KATALOG_GLOSOW.exists():
        for d in sorted(KATALOG_GLOSOW.iterdir()):
            if (d / "conds.pt").exists():
                try:
                    meta = json.loads((d / "glos.json").read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                wynik.append({"id": "cb:" + d.name, "nazwa": "Chatterbox · " + meta.get("nazwa", d.name),
                              "pobrany": True, "wlasny": True, "info": meta})
    return wynik


def sciezka_conds(klucz: str) -> Path | None:
    """cb:domyslny -> None (wbudowany głos modelu), cb:<slug> -> glosy/<slug>/conds.pt"""
    if klucz == DOMYSLNY_GLOS:
        return None
    nazwa = klucz.split(":", 1)[1]
    if "/" in nazwa or "\\" in nazwa or ".." in nazwa:
        raise ValueError("Niepoprawna nazwa głosu")
    p = KATALOG_GLOSOW / nazwa / "conds.pt"
    if not p.exists():
        raise FileNotFoundError(f"Brak głosu {nazwa} (glosy/{nazwa}/conds.pt)")
    return p


def zainstalowany() -> bool:
    return PYTHON_CHATTERBOX.exists()
