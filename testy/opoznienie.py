"""Pomiar opóźnienia głosu (bez gry i bez dźwięku): ile mija od zlecenia kwestii do pierwszego
dźwięku lektora – przez prawdziwą ścieżkę aplikacji (Lektor -> Glos -> proces Chatterboxa/Piper).

Uruchom:  .venv\\Scripts\\python.exe testy\\opoznienie.py                 (głos i ustawienia z ustawienia.json)
          .venv\\Scripts\\python.exe testy\\opoznienie.py --glos jarvis   (inny głos, np. Piper)
          .venv\\Scripts\\python.exe testy\\opoznienie.py --tryb najszybszy   (tryb szybkości Chatterboxa)

Wynik: mediana i rozrzut czasu „do pierwszego dźwięku” dla 16 kwestii (od „Hej!” po długie zdania)
oraz test kolejki: 4 kwestie naraz – czy wszystkie zostały przeczytane i jakie było tempo.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BAZA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BAZA))

from lektor.glos import Glos, Lektor, Odtwarzacz  # noqa: E402
from lektor.tekst import normalizuj, wczytaj_slownik  # noqa: E402

DODATKOWE = ["Hej!", "Co ty robisz?", "Dobra, jedziemy.", "Nie teraz, Lucia.", "Masz to?", "Szybko, do samochodu!"]


class Ustawienia:
    def __init__(self, d: dict):
        self.d = d

    def get(self, k, domyslna=None):
        return self.d.get(k, domyslna)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glos", help="klucz głosu (np. jarvis albo cb:janusz); domyślnie z ustawień")
    ap.add_argument("--tryb", help="tryb szybkości Chatterboxa (cb_tryb), np. najszybszy / zrownowazony / jakosc")
    ap.add_argument("--powtorzen", type=int, default=1)
    a = ap.parse_args()

    from aplikacja import DOMYSLNE
    d = dict(DOMYSLNE)
    plik = BAZA / "ustawienia.json"
    if plik.exists():
        d.update(json.loads(plik.read_text(encoding="utf-8")))
    if a.glos:
        d["glos"] = a.glos if a.glos in ("meski", "jarvis", "darkman", "mc_speech", "bass") or a.glos.startswith(("cb:", "plik:")) else "cb:" + a.glos
    if a.tryb:
        d["cb_tryb"] = a.tryb
    d["glosnosc"] = 0.0
    ust = Ustawienia(d)
    slownik = wczytaj_slownik(BAZA / "slownik_wymowy.txt")
    zdania = [z.strip() for z in (BAZA / "testy" / "zdania.txt").read_text(encoding="utf-8").splitlines() if z.strip()]
    zdania += DODATKOWE

    glos = Glos(ust, log=lambda m: print("  · " + m))
    t0 = time.perf_counter()
    glos.zaladuj(d["glos"], etap=lambda e: print("  · " + e))
    print(f"Głos {d['glos']} gotowy po {time.perf_counter() - t0:.1f} s; tempo {d.get('predkosc')}, "
          f"tryb {d.get('cb_tryb', '-')}.")

    odtw = Odtwarzacz(log=lambda m: None)
    odtw.atrapa = True                      # bez dźwięku: odtwarzanie „na niby” w czasie rzeczywistym
    odtw._otworz = lambda sr, urz: None
    historia = []
    gotowe = threading.Event()

    def po_kwestii(w):
        historia.append(w)
        gotowe.set()

    lektor = Lektor(glos, odtw, ust, log=lambda m: print("  · " + m), po_kwestii=po_kwestii)

    # 1) pojedyncze kwestie: czas od zlecenia do pierwszego dźwięku
    czasy = []
    for _ in range(a.powtorzen):
        for z in zdania:
            gotowe.clear()
            tekst = normalizuj(z, slownik)
            lektor.powiedz(tekst, {"t_klatki": time.perf_counter()})
            gotowe.wait(60)
            w = historia[-1]
            czasy.append(w["calosc_ms"])
            print(f"  {w['calosc_ms']:5d} ms do pierwszego dźwięku | {w['silnik']:10s} | {tekst}")
    print(f"\nDO PIERWSZEGO DŹWIĘKU: mediana {statistics.median(czasy):.0f} ms, "
          f"min {min(czasy)} ms, maks {max(czasy)} ms ({len(czasy)} kwestii)")

    # 2) kolejka: 4 kwestie naraz (szybka wymiana zdań) – wszystkie mają zostać przeczytane
    historia.clear()
    seria = [normalizuj(z, slownik) for z in ["Hej, ty! Stój!", "Co jest? Czego chcesz?",
                                              "Oddawaj kluczyki, szybko!", "Dobra, dobra, spokojnie. Masz."]]
    t0 = time.perf_counter()
    for t in seria:
        lektor.powiedz(t, {"t_klatki": time.perf_counter()})
        time.sleep(0.4)
    while len(historia) < len(seria) and time.perf_counter() - t0 < 60:
        time.sleep(0.05)
    print("\nKOLEJKA (4 kwestie co 0,4 s):")
    for w in historia:
        print(f"  start po {w['calosc_ms']:5d} ms · tempo {w['predkosc']:.2f} · zaległość {w['zaleglosc_s']:.1f} s | {w['tekst']}")
    kompletna = [w["tekst"] for w in historia] == seria
    print(f"  wszystkie przeczytane po kolei: {'TAK' if kompletna else 'NIE'}; "
          f"koniec po {time.perf_counter() - t0:.1f} s")
    lektor.zakoncz()
    glos.zamknij()
    return 0 if kompletna else 1


if __name__ == "__main__":
    raise SystemExit(main())
