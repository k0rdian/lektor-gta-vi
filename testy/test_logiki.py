"""Testy logiki lektora bez gry i bez OCR: pamięć linii (bez powtórek) i filtry tablic/ruchu.

Uruchom:  .venv\\Scripts\\python.exe testy\\test_logiki.py     (albo testuj.bat)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lektor.filtry import linie_napisu, oznacz_pole, oznacz_ruch  # noqa: E402
from lektor.tekst import PamiecLinii, czysc_znaki, wyglada_na_napis, wyglada_na_tablice  # noqa: E402

A = "Słuchaj, nie mamy dużo czasu."
B = "Gliny będą tu za pięć minut."
C = "Wiesz, co jest najgorsze?"
D = "Że jeszcze rok temu myślałem, że wyjdę z tego czysty."


def pamiec_z(*linie, t=0.0) -> PamiecLinii:
    p = PamiecLinii(zycie=3.0)
    p.dodaj(list(linie), t)
    return p


# ---------------------------------------------------------------- pamięć linii

def test_zmienia_sie_tylko_druga_linia():
    p = pamiec_z(A, B)
    assert p.nowe([A, C]) == [C]


def test_trzecia_zmiana_drugiej_linii():
    p = pamiec_z(A, B)
    p.odswiez([A, C], 1.0)
    p.dodaj([A, C], 1.0)
    p.odswiez([A, D], 2.0)
    assert p.nowe([A, D]) == [D]


def test_napisy_przesuwaja_sie_w_gore():
    p = pamiec_z(A, B)
    assert p.nowe([B, C]) == [C]


def test_dopisana_druga_linia():
    p = pamiec_z(A)
    assert p.nowe([A, B]) == [B]


def test_szum_ocr_to_ta_sama_linia():
    p = pamiec_z(A)
    assert p.nowe(["Sluchaj, nie mamy duzo czasu"]) == []
    assert p.nowe(["Słuchaj nie mamy dużo czasu."]) == []


def test_ocr_skleil_dwie_linie():
    p = pamiec_z(A, B)
    assert p.nowe([A + " " + B]) == []


def test_ocr_skleil_stare_linie_z_nowa():
    p = pamiec_z(A, B)
    assert p.nowe([A + " " + B + " " + C]) == [C]


def test_ocr_skleil_linie_z_bledami():
    p = pamiec_z(A, B)
    assert p.nowe(["Sluchaj, nie mamy duzo czasu. Gliny beda tu za piec minut."]) == []


def test_ocr_rozbil_linie():
    dluga = "Jason, ty naprawdę myślisz, że on nas tak po prostu wypuści?"
    p = pamiec_z(dluga)
    assert p.nowe(["Jason, ty naprawdę myślisz,", "że on nas tak po prostu wypuści?"]) == []


def test_wczesniej_przeczytany_tylko_poczatek():
    p = pamiec_z("Słuchaj, nie mamy dużo")
    assert p.nowe(["Słuchaj, nie mamy dużo czasu."]) == ["czasu."]


def test_krotka_linia_nie_obcina_poczatku():
    p = pamiec_z("Nie.")
    assert p.nowe(["Nie wiem, co robić."]) == ["Nie wiem, co robić."]


def test_zupelnie_nowy_napis():
    p = pamiec_z(A, B)
    assert p.nowe([C, D]) == [C, D]


def test_zapomina_po_czasie():
    p = pamiec_z(A, t=0.0)
    p.odswiez([C], 4.0)
    assert p.nowe([A]) == [A]


def test_widoczna_linia_nie_wygasa():
    p = pamiec_z(A, t=0.0)
    p.odswiez([A, B], 2.5)
    p.odswiez([A, C], 5.0)
    assert p.nowe([A]) == []


def test_krotkie_linie_porownywane_dokladnie():
    p = pamiec_z("Co?")
    assert p.nowe(["No?"]) == ["No?"]
    assert p.nowe(["Co?"]) == []


# ---------------------------------------------------------------- tablice i śmieci OCR

def test_obce_znaki():
    czysty, obce = czysc_znaki("V AP 工 口.")
    assert obce >= 0.25, obce
    czysty, obce = czysc_znaki("Źródło? Żółć, jaźń, gęśla — Łódź!")
    assert obce == 0 and czysty == "Źródło? Żółć, jaźń, gęśla — Łódź!"


def test_tablice():
    for t in ("J26B7", "ABC 1234", "7XY-123", "LS 42 KR", "4X7B2"):
        assert wyglada_na_tablice(t), t
    for t in ("OK", "Jason", "GTA", "911", "Masz 5 minut.", "Nie!", "Vice City", "STOP"):
        assert not wyglada_na_tablice(t), t


def test_pole_tablicy_odrzucone():
    p = {"tekst": "J26B7", "pewnosc": 0.95, "x": 0, "y": 0, "w": 50, "h": 20}
    oznacz_pole(p, None, 0.55, {"filtr_tablic": True})
    assert p.get("odrzucone") == "tablica rejestracyjna"
    p = {"tekst": "J26B7", "pewnosc": 0.95, "x": 0, "y": 0, "w": 50, "h": 20}
    oznacz_pole(p, None, 0.55, {"filtr_tablic": False})
    assert not p.get("odrzucone")


def test_smieci_nie_wygladaja_na_napis():
    assert not wyglada_na_napis("J26B7", [0.9])
    assert not wyglada_na_napis("ll", [0.99])
    assert wyglada_na_napis(A, [0.9])


# ---------------------------------------------------------------- ruch i położenie

def pole(tekst, x0, y0, x1, y1, odrz=None):
    return {"tekst": tekst, "pewnosc": 0.9, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "odrzucone": odrz}


def test_ruchoma_tablica_odrzucona():
    poprz = [pole("ABCDE", 0.40, 0.85, 0.46, 0.88)]
    teraz = [pole("ABCDE", 0.43, 0.85, 0.49, 0.88)]       # 0.03 × 1920 = 58 px w bok
    oznacz_ruch(teraz, poprz, 1920, 1080)
    assert teraz[0]["odrzucone"] == "ruch"


def test_nieruchomy_napis_przyjety():
    poprz = [pole(A, 0.35, 0.85, 0.65, 0.89)]
    teraz = [pole(A, 0.3502, 0.8501, 0.6501, 0.8899)]       # drobne drganie ramki
    oznacz_ruch(teraz, poprz, 1920, 1080)
    assert not teraz[0]["odrzucone"]


def test_dopisujacy_sie_napis_to_nie_ruch():
    poprz = [pole("Słuchaj, nie mamy dużo", 0.38, 0.85, 0.62, 0.89)]
    teraz = [pole("Słuchaj, nie mamy dużo czasu.", 0.35, 0.85, 0.65, 0.89)]   # wyśrodkowany, urósł
    oznacz_ruch(teraz, poprz, 1920, 1080)
    assert not teraz[0]["odrzucone"]


def test_tekst_z_boku_odrzucony():
    pola = [pole(A, 0.35, 0.85, 0.65, 0.89), pole("Vespucci Beach", 0.70, 0.80, 0.82, 0.83)]
    assert linie_napisu(pola, True) == [A]
    assert pola[1]["odrzucone"] == "poza środkiem"


def test_szyld_w_tej_samej_linii_co_napis():
    pola = [pole(A, 0.35, 0.85, 0.65, 0.89), pole("Pizza", 0.80, 0.852, 0.86, 0.888)]
    assert linie_napisu(pola, True) == [A]


def test_bez_filtra_srodka():
    pola = [pole(A, 0.35, 0.85, 0.65, 0.89), pole("Vespucci Beach", 0.70, 0.80, 0.82, 0.83)]
    assert linie_napisu(pola, False) == ["Vespucci Beach", A]


def test_dwie_linie_wysrodkowane():
    pola = [pole(A, 0.36, 0.85, 0.64, 0.885), pole(B, 0.37, 0.90, 0.63, 0.935)]
    assert linie_napisu(pola, True) == [A, B]


# ---------------------------------------------------------------- lektor: kolejka, zdania, tempo

class _GlosTestowy:
    """Udaje syntezę: 1 znak tekstu = 4 ms audio przy tempie 1, synteza trwa 30 ms na zdanie."""
    nazwa_silnika = "Test"
    ostatni_silnik = "Test"

    def __init__(self):
        self.wywolania = []

    def syntezuj(self, tekst, tempo=1.0, przy_starcie=None):
        import time
        import numpy as np
        self.wywolania.append((tekst, round(tempo, 3)))
        if przy_starcie is not None:
            przy_starcie(len(self.wywolania))
        if "AWARIA" in tekst:
            raise RuntimeError("awaria syntezy")
        for zdanie in tekst.split(". "):
            time.sleep(0.03)
            n = int(len(zdanie) * 0.004 / tempo * 16000)
            yield (np.full(n, 3000, dtype=np.int16), 16000)


class _UstTestowe:
    def __init__(self, **d):
        self.d = {"polityka": "kolejka", "predkosc": 1.0, "przyspieszaj": True, "tempo_maks": 1.5, **d}

    def get(self, k, domyslna=None):
        return self.d.get(k, domyslna)


def _lektor_testowy(**ust):
    from lektor.glos import Lektor, Odtwarzacz
    odtw = Odtwarzacz(log=lambda m: None)
    odtw.atrapa = True                          # „na niby”: czeka tyle, ile trwa audio, bez dźwięku
    odtw._otworz = lambda sr, urz: None
    historia, dziennik = [], []
    glos = _GlosTestowy()
    lektor = Lektor(glos, odtw, _UstTestowe(**ust), log=dziennik.append, po_kwestii=historia.append)
    return lektor, glos, historia, dziennik


def _czekaj(warunek, limit=10.0):
    import time
    t0 = time.time()
    while not warunek():
        if time.time() - t0 > limit:
            raise AssertionError("przekroczono czas oczekiwania")
        time.sleep(0.02)


def test_kolejka_czyta_wszystko_po_kolei():
    lektor, glos, historia, dziennik = _lektor_testowy()
    teksty = [f"Kwestia numer {i}. Ma dwa zdania i trochę tekstu" for i in range(5)]
    for t in teksty:
        lektor.powiedz(t, {"t_klatki": 0})
    _czekaj(lambda: len(historia) == 5)
    lektor.zakoncz()
    assert [h["tekst"] for h in historia] == teksty            # nic nie przepadło, kolejność zachowana
    assert not any(h["przerwana"] for h in historia)
    tempa = [h["predkosc"] for h in historia]
    assert tempa[0] > 1.0 and tempa[-1] == 1.0, tempa          # przy zaległości szybciej, na końcu normalnie


def test_kolejka_bez_przyspieszania():
    lektor, glos, historia, dziennik = _lektor_testowy(przyspieszaj=False)
    for i in range(3):
        lektor.powiedz(f"Kwestia {i}. Trochę tekstu do przeczytania", {})
    _czekaj(lambda: len(historia) == 3)
    lektor.zakoncz()
    assert all(h["predkosc"] == 1.0 for h in historia)


def test_kolejka_synteza_wyprzedza_odtwarzanie():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy(przyspieszaj=False)
    lektor.powiedz("Pierwsza długa kwestia. " + "bla " * 60, {})
    lektor.powiedz("Druga kwestia", {})
    time.sleep(0.3)                     # pierwsza jeszcze brzmi (ok. 1 s), a druga już zsyntezowana
    assert len(glos.wywolania) == 2 and not historia
    _czekaj(lambda: len(historia) == 2)
    lektor.zakoncz()


def test_dokoncz_zostawia_najnowsza():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy(polityka="dokoncz")
    lektor.powiedz("Bieżąca kwestia. " + "bla " * 40, {})
    time.sleep(0.15)
    for i in range(3):
        lektor.powiedz(f"Zaległa {i}", {})
    _czekaj(lambda: len(historia) == 2)
    time.sleep(0.2)
    lektor.zakoncz()
    assert [h["tekst"] for h in historia] == ["Bieżąca kwestia. " + "bla " * 40, "Zaległa 2"]
    assert any("Pominięto zaległą" in d for d in dziennik)


def test_przerwij_przerywa_biezaca():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy(polityka="przerwij")
    lektor.powiedz("Długa kwestia. " + "bla " * 100, {})
    _czekaj(lambda: lektor.mowi)
    time.sleep(0.1)
    lektor.powiedz("Nowa kwestia", {})
    _czekaj(lambda: len(historia) == 2)
    lektor.zakoncz()
    assert historia[0]["przerwana"] and historia[1]["tekst"] == "Nowa kwestia" and not historia[1]["przerwana"]


def test_ucisz_czysci_kolejke():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy()
    for i in range(4):
        lektor.powiedz(f"Kwestia {i}. " + "bla " * 30, {})
    _czekaj(lambda: lektor.mowi)
    lektor.ucisz()
    time.sleep(0.5)
    lektor.zakoncz()
    assert len(historia) <= 1 and all(h["przerwana"] for h in historia)
    assert lektor.stan()["w_kolejce"] == 0


def test_blad_syntezy_nie_zatrzymuje_kolejki():
    lektor, glos, historia, dziennik = _lektor_testowy()
    lektor.powiedz("AWARIA", {})
    lektor.powiedz("Następna kwestia działa", {})
    _czekaj(lambda: len(historia) == 1)
    lektor.zakoncz()
    assert historia[0]["tekst"] == "Następna kwestia działa"
    assert any("awaria syntezy" in d for d in dziennik)


def test_wstepna_synteza_potwierdzona():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy()
    nr = lektor.przygotuj("Wstępna kwestia do sprawdzenia", {"t_klatki": 0})
    time.sleep(0.2)                      # synteza już ruszyła, ale nic nie gra przed potwierdzeniem
    assert len(glos.wywolania) == 1 and not lektor.mowi and not historia
    assert lektor.potwierdz(nr, "Wstępna kwestia do sprawdzenia.", {"t_klatki": 1.0})
    _czekaj(lambda: len(historia) == 1)
    lektor.zakoncz()
    assert len(glos.wywolania) == 1      # bez drugiej syntezy
    assert historia[0]["tekst"] == "Wstępna kwestia do sprawdzenia"


def test_wstepna_synteza_inny_tekst():
    lektor, glos, historia, dziennik = _lektor_testowy()
    nr = lektor.przygotuj("Ala ma ko", {})
    assert not lektor.potwierdz(nr, "Ala ma kota i psa", {})
    _czekaj(lambda: len(historia) == 1)
    lektor.zakoncz()
    assert [h["tekst"] for h in historia] == ["Ala ma kota i psa"]


def test_wstepna_synteza_inne_ogonki_to_inny_tekst():
    # „beda” (błąd OCR w 1. odczycie) i „będą” brzmią inaczej – wstępna synteza nie może zostać użyta
    lektor, glos, historia, dziennik = _lektor_testowy()
    nr = lektor.przygotuj("Gliny beda tu za pięć minut", {})
    assert not lektor.potwierdz(nr, "Gliny będą tu za pięć minut", {})
    _czekaj(lambda: len(historia) == 1)
    lektor.zakoncz()
    assert historia[0]["tekst"] == "Gliny będą tu za pięć minut"


def test_anulowanie_przerywa_synteze_na_gpu():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy()
    przerwane = []
    glos.przerwij = przerwane.append
    nr = lektor.przygotuj(". ".join(["Zdanie numer jeden"] * 30), {})   # synteza ok. 0,9 s
    _czekaj(lambda: len(glos.wywolania) == 1)
    time.sleep(0.1)
    lektor.anuluj(nr)
    lektor.zakoncz()
    assert przerwane == [1], przerwane            # żądanie nr 1 przerwane od razu, nie po całym zdaniu


def test_duplikat_w_kolejce_pominiety():
    lektor, glos, historia, dziennik = _lektor_testowy(przyspieszaj=False)
    lektor.powiedz("Pierwsza kwestia. " + "bla " * 30, {})
    lektor.powiedz("Ta sama kwestia", {})
    lektor.powiedz("Ta sama kwestia.", {})        # OCR odczytał drugi raz, a pierwsza jeszcze czeka
    _czekaj(lambda: len(historia) == 2)
    lektor.zakoncz()
    assert [h["tekst"] for h in historia] == ["Pierwsza kwestia. " + "bla " * 30, "Ta sama kwestia"]
    assert lektor.statystyki["duplikaty"] == 1


def test_pomin_pomija_tylko_biezaca():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy(przyspieszaj=False)
    lektor.powiedz("Długa kwestia. " + "bla " * 60, {})
    lektor.powiedz("Następna", {})
    _czekaj(lambda: lektor.mowi)
    time.sleep(0.05)
    lektor.pomin()
    _czekaj(lambda: len(historia) == 2)
    lektor.zakoncz()
    assert historia[0]["przerwana"] and historia[1]["tekst"] == "Następna" and not historia[1]["przerwana"]


def test_wstepna_synteza_anulowana_i_bez_potwierdzenia():
    import time
    lektor, glos, historia, dziennik = _lektor_testowy()
    lektor.MAKS_CZEKANIA_NA_POTWIERDZENIE_S = 0.3
    nr = lektor.przygotuj("Napis, który zniknął", {})
    lektor.anuluj(nr)
    lektor.przygotuj("Napis, którego nikt nie potwierdził", {})
    time.sleep(0.8)
    lektor.powiedz("Normalna kwestia", {})
    _czekaj(lambda: len(historia) == 1)
    lektor.zakoncz()
    assert [h["tekst"] for h in historia] == ["Normalna kwestia"]


def test_przycinanie_ciszy_na_poczatku():
    import numpy as np
    from lektor.glos import przytnij_cisze
    sr = 24000
    cisza = np.zeros(int(0.3 * sr), dtype=np.int16)
    mowa = (np.sin(np.arange(sr) * 0.05) * 8000).astype(np.int16)
    wynik = przytnij_cisze(np.concatenate([cisza, mowa, cisza]), sr, poczatek=True, koniec_zostaw_s=0.1)
    assert abs(len(wynik) / sr - (1.0 + 0.02 + 0.1)) < 0.02, len(wynik) / sr


def test_strumien_cisza_miedzy_kawalkami():
    """Kawałki strumienia (krotki z koniec_zdania): cisza przed pierwszym słowem przepada, pauza
    w środku zdania zostaje w całości (także na granicy kawałków), a cisza na końcu zdania jest
    skracana do 0,3 s, nawet gdy rozkłada się na dwa kawałki."""
    import numpy as np
    lektor, glos, historia, dziennik = _lektor_testowy(przyspieszaj=False)
    sr = 24000
    mowa = (np.sin(np.arange(int(0.5 * sr)) * 0.05) * 8000).astype(np.int16)

    def cisza(s):
        return np.zeros(int(round(s * sr)), dtype=np.int16)

    kawalki = [
        (cisza(0.3), sr, False),                                         # sam oddech – pomijany
        (np.concatenate([cisza(0.1), mowa, cisza(0.2)]), sr, False),     # początek: zostaje 0,02 s ciszy
        (np.concatenate([cisza(0.2), mowa, cisza(0.4)]), sr, False),     # pauza 0,2 + 0,2 s zostaje
        (cisza(0.5), sr, True),                                          # koniec: 0,4 + 0,5 s -> 0,3 s
    ]

    def syntezuj(tekst, tempo=1.0, przy_starcie=None):
        if przy_starcie is not None:
            przy_starcie(1)
        yield from kawalki

    glos.syntezuj = syntezuj
    zagrane = []

    def graj_kawalek(audio, sr_wej, anuluj, glosnosc=1.0, urzadzenie=None, przy_pierwszym=None):
        if przy_pierwszym is not None:
            przy_pierwszym()
        zagrane.append(audio.copy())

    lektor.odtwarzacz.graj_kawalek = graj_kawalek
    lektor.powiedz("Test strumienia.", {})
    _czekaj(lambda: len(historia) == 1)
    lektor.zakoncz()
    calosc = len(np.concatenate(zagrane)) / sr
    assert abs(calosc - (0.02 + 0.5 + 0.4 + 0.5 + 0.3)) < 0.015, calosc
    assert len(zagrane) == 3, [len(z) / sr for z in zagrane]


def test_podzial_na_zdania():
    from lektor.glos import podziel_na_zdania
    assert podziel_na_zdania("Co? Nie, nie, nie. To nie tak miało wyglądać!") == \
        ["Co? Nie, nie, nie. To nie tak miało wyglądać!"]
    assert podziel_na_zdania(A + " " + B) == [A, B]
    dlugie = ", ".join(["to jest bardzo długie zdanie bez kropki"] * 8) + "."
    czesci = podziel_na_zdania(dlugie)
    assert len(czesci) >= 2 and all(len(c) <= 220 for c in czesci)
    assert " ".join(czesci).replace("  ", " ") == dlugie


def test_zmiana_tempa_bez_zmiany_wysokosci():
    import numpy as np
    from lektor.glos import zmien_tempo
    sr = 24000
    t = np.arange(sr) / sr
    ton = (np.sin(2 * np.pi * 220 * t) * 12000).astype(np.int16)    # 1 s, 220 Hz
    szybciej = zmien_tempo(ton, 1.25, sr)
    assert abs(len(szybciej) / sr - 0.8) < 0.02, len(szybciej) / sr
    srodek = szybciej[2000:-2000].astype(np.float32)
    przejscia = np.count_nonzero(np.diff(np.signbit(srodek))) / 2 / (len(srodek) / sr)
    assert abs(przejscia - 220) < 6, przejscia                          # wysokość ta sama


def main() -> int:
    testy = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bledy = 0
    for nazwa, f in testy:
        try:
            f()
            print(f"  OK   {nazwa}")
        except Exception:
            bledy += 1
            print(f"  BŁĄD {nazwa}")
            traceback.print_exc()
    print(f"\n{len(testy) - bledy}/{len(testy)} testów zaliczonych.")
    return 1 if bledy else 0


if __name__ == "__main__":
    raise SystemExit(main())
