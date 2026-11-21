"""Test strumieniowej syntezy Chatterboxa na prawdziwym modelu (bez odtwarzania dźwięku).

Dla każdego trybu (najszybszy / zrownowazony / jakosc) syntezuje kilka kwestii przez
Silnik.syntezuj_strumien – tę samą drogę co aplikacja – i mierzy:
  - czas do pierwszego kawałka dźwięku (TTFA, w procesie syntezy, bez OCR i karty dźwiękowej),
  - czy dźwięk nadąża: dla każdego kawałka sprawdza, czy był gotowy, zanim skończyłby się
    poprzedni przy odtwarzaniu od pierwszego kawałka (przerwy = „zacięcia”),
  - liczbę kawałków i ich długości.
Zapisuje nagrania do probki/strumien_<tryb>.wav (do odsłuchu) i wyniki do probki/strumien_test.json.

Użycie (z katalogu projektu):
  .venv-chatterbox\\Scripts\\python.exe testy\\strumien_test.py [--glos janusz] [--tryby najszybszy,zrownowazony,jakosc]
                                                             [--tempo 1.0] [--powtorzen 1]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BAZA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BAZA))

KWESTIE = [
    "Hej! Masz to?",
    "Nie teraz.",
    "Słuchaj, nie mamy dużo czasu. Gliny będą tu za pięć minut.",
    "Jason, ty naprawdę myślisz, że on nas tak po prostu wypuści?",
    "Witamy w Vice City. Tutaj nikt nie pyta, skąd masz pieniądze.",
    "Dobra, wsiadaj. Jedziemy na lotnisko, zanim ktoś zauważy, że zniknęliśmy.",
    "Co ty robisz?! Zostaw to!",
    "Wiesz co? Zawsze wiedziałam, że skończymy w takim miejscu jak to, ale nie sądziłam, że tak szybko.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glos", default="janusz")
    ap.add_argument("--tryby", default="najszybszy,zrownowazony,jakosc")
    ap.add_argument("--tempo", type=float, default=1.0)
    ap.add_argument("--powtorzen", type=int, default=1)
    ap.add_argument("--ekspresja", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=0.4)
    ap.add_argument("--temperatura", type=float, default=0.75)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from lektor import chatterbox_worker as w
    from lektor.chatterbox_pliki import SR

    tryby = [t.strip() for t in a.tryby.split(",") if t.strip()]
    s = w.Silnik()
    t0 = time.perf_counter()
    print(s.zaladuj("v3", tryb=tryby[0]), flush=True)
    conds = BAZA / "glosy" / a.glos / "conds.pt"
    if conds.exists():
        s.ustaw_glos(str(conds))
    print(f"Model gotowy po {time.perf_counter() - t0:.1f} s; głos: {a.glos if conds.exists() else 'wbudowany'}",
          flush=True)
    wyniki = {}
    (BAZA / "probki").mkdir(exist_ok=True)
    for tryb in tryby:
        s._s3gen_szybki(w.TRYBY[tryb]["dekoder"])        # grafy tego trybu przed pomiarem
        s.syntezuj_strumien("Rozgrzewka trybu.", "pl", a.ekspresja, a.cfg, a.temperatura, tempo=a.tempo, tryb=tryb,
                            wyslij=lambda audio, info: None)
        pomiary, nagranie = [], []
        for powt in range(a.powtorzen):
            for tekst in KWESTIE:
                kawalki = []
                t_start = time.perf_counter()

                def wyslij(audio, info, kawalki=kawalki, t_start=t_start):
                    kawalki.append((time.perf_counter() - t_start, audio.copy(), dict(info)))

                czasy = s.syntezuj_strumien(tekst, "pl", a.ekspresja, a.cfg, a.temperatura, tempo=a.tempo,
                                            tryb=tryb, wyslij=wyslij)
                t_calosc = time.perf_counter() - t_start
                if not kawalki:
                    print(f"  [{tryb}] BRAK DŹWIĘKU: {tekst}", flush=True)
                    continue
                # odtwarzanie od chwili pierwszego kawałka: czy każdy następny zdążył?
                t_gram = kawalki[0][0]
                przerwy = 0.0
                for t_got, audio, _ in kawalki:
                    if t_got > t_gram:
                        przerwy += t_got - t_gram
                        t_gram = t_got
                    t_gram += len(audio) / SR
                audio = np.concatenate([k[1] for k in kawalki])
                p = {"tekst": tekst, "ttfa_ms": round(kawalki[0][0] * 1000, 1),
                     "calosc_ms": round(t_calosc * 1000), "audio_s": round(len(audio) / SR, 2),
                     "kawalkow": len(kawalki), "przerwy_ms": round(przerwy * 1000, 1),
                     "dlugosci_s": [round(len(k[1]) / SR, 2) for k in kawalki],
                     "konce_zdan": [bool(k[2].get("koniec_zdania")) for k in kawalki],
                     "t3_ms": czasy.get("t3_ms"), "s3_ms": czasy.get("s3_ms"), "tokeny": czasy.get("tokeny")}
                pomiary.append(p)
                if powt == 0:
                    nagranie += [audio, np.zeros(int(0.6 * SR), dtype=np.int16)]
                print(f"  [{tryb}] TTFA {p['ttfa_ms']:6.1f} ms | całość {p['calosc_ms']:5d} ms na {p['audio_s']:4.1f} s"
                      f" | kawałków {p['kawalkow']:2d} | przerwy {p['przerwy_ms']:5.1f} ms | {tekst[:50]}", flush=True)
        plik = BAZA / "probki" / f"strumien_{tryb}.wav"
        sf.write(str(plik), np.concatenate(nagranie), SR, subtype="PCM_16")
        ttfa = [p["ttfa_ms"] for p in pomiary]
        wyniki[tryb] = {"ttfa_mediana_ms": round(statistics.median(ttfa), 1), "ttfa_maks_ms": max(ttfa),
                        "przerwy_maks_ms": max(p["przerwy_ms"] for p in pomiary),
                        "rtf_mediana": round(statistics.median(p["calosc_ms"] / 1000 / max(p["audio_s"], 0.01)
                                                               for p in pomiary), 3),
                        "pomiary": pomiary, "nagranie": str(plik)}
        import torch
        wyniki[tryb]["vram_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        print(f"[{tryb}] TTFA mediana {wyniki[tryb]['ttfa_mediana_ms']} ms, maks {wyniki[tryb]['ttfa_maks_ms']} ms; "
              f"najdłuższa przerwa {wyniki[tryb]['przerwy_maks_ms']} ms; RTF {wyniki[tryb]['rtf_mediana']}; "
              f"pamięć karty {wyniki[tryb]['vram_gb']} GB -> {plik.name}", flush=True)
    (BAZA / "probki" / "strumien_test.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
