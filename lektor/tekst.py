"""Obróbka tekstu z OCR: składanie linii, normalizacja pod lektora, słownik wymowy,
pamięć przeczytanych linii (bez powtórek) i rozpoznawanie „nie-napisów” (tablice, obce znaki)."""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # awaryjnie, gdyby rapidfuzz nie był zainstalowany
    _fuzz = None

try:
    from num2words import num2words as _num2words
except ImportError:
    _num2words = None

DUZE = "A-ZĄĆĘŁŃÓŚŹŻ"
MALE = "a-ząćęłńóśźż"


def podobienstwo(a: str, b: str) -> float:
    """Podobieństwo dwóch tekstów w skali 0-100 (odporne na drobne błędy OCR)."""
    if not a or not b:
        return 0.0
    if _fuzz is not None:
        return float(_fuzz.ratio(a.lower(), b.lower()))
    import difflib
    return 100.0 * difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _czesciowe(krotszy: str, dluzszy: str) -> float:
    """Najlepsze dopasowanie krótszego tekstu do fragmentu dłuższego (0-100)."""
    if not krotszy or not dluzszy:
        return 0.0
    if _fuzz is not None:
        return float(_fuzz.partial_ratio(krotszy, dluzszy))
    if krotszy in dluzszy:
        return 100.0
    import difflib
    n = len(krotszy)
    return max((100.0 * difflib.SequenceMatcher(None, krotszy, dluzszy[i:i + n]).ratio()
                for i in range(0, max(1, len(dluzszy) - n + 1))), default=0.0)


def norm_linii(tekst: str) -> str:
    """Postać do porównywania linii: małe litery bez polskich znaków (OCR najczęściej myli
    właśnie ogonki: „będą” / „beda”), bez interpunkcji i nadmiarowych spacji."""
    t = unicodedata.normalize("NFKD", tekst.lower().replace("ł", "l"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


# ---------------------------------------------------------------- znaki spoza alfabetu łacińskiego

_DOZWOLONE_INNE = set("„”“‘’«»…–—€£°§·¡¿")


def czysc_znaki(tekst: str) -> tuple[str, float]:
    """Usuwa znaki, których nie ma w polskich napisach (chińskie, japońskie, symbole).

    PaddleOCR ma w słowniku tysiące znaków CJK i w liściach, kratkach czy na tablicach
    rejestracyjnych potrafi „zobaczyć” np. „工 口”. Zwraca (oczyszczony tekst, udział obcych liter
    wśród wszystkich liter) – duży udział oznacza, że całe pole to śmieć."""
    litery = obce = 0
    wynik = []
    for c in tekst:
        if c.isascii() or c in _DOZWOLONE_INNE:
            wynik.append(c)
            litery += c.isalpha()
            continue
        if c.isalpha():
            litery += 1
            if unicodedata.name(c, "").startswith("LATIN"):
                wynik.append(c)
            else:
                obce += 1
            continue
        if c.isspace():
            wynik.append(" ")
    czysty = re.sub(r"\s{2,}", " ", "".join(wynik)).strip()
    return czysty, (obce / litery if litery else 0.0)


def wyglada_na_tablice(tekst: str) -> bool:
    """Tablica rejestracyjna / numer: krótki ciąg WIELKICH liter i cyfr (np. J26B7, ABC 1234, 7XY-12).

    Kwestia dialogowa nigdy nie składa się wyłącznie z takiego kodu."""
    t = re.sub(r"[\s\-–.·•:_]", "", tekst)
    if not 3 <= len(t) <= 9 or not t.isascii() or not t.isalnum() or t.upper() != t:
        return False
    return any(c.isdigit() for c in t) and any(c.isalpha() for c in t)


# ---------------------------------------------------------------- pamięć przeczytanych linii

class PamiecLinii:
    """Linie napisów, które lektor już przeczytał i które wciąż są na ekranie.

    Gra często zostawia pierwszą linię, a zmienia tylko drugą (albo dopisuje nową linię pod
    starą, albo przesuwa napisy w górę). Bez tej pamięci każda taka zmiana kończyła się
    ponownym przeczytaniem całości – pierwsza linia padała 2-3 razy. Linia jest „pamiętana”,
    dopóki OCR ją widzi; po `zycie` sekundach niewidzenia zostaje zapomniana, więc ta sama
    kwestia pokazana ponownie po przerwie zostanie przeczytana."""

    PROG = 86          # podobieństwo linii (0-100), od którego uznajemy ją za tę samą
    PROG_KROTKIE = 100  # linie do 3 znaków muszą się zgadzać dokładnie
    MIN_PREFIKS = 12   # z ilu znaków musi mieć stara linia, żeby odciąć ją z początku nowej

    def __init__(self, zycie: float = 3.0):
        self.zycie = zycie
        self._linie: list[list] = []   # [znormalizowana, ostatnio widziana, oryginał]

    def __len__(self):
        return len(self._linie)

    def wyczysc(self):
        self._linie.clear()

    def linie(self) -> list[str]:
        return [l[2] for l in self._linie]

    def _okna(self) -> list[str]:
        """Pojedyncze linie i sklejenia 2-3 kolejnych (OCR czasem skleja dwie linie w jedną)."""
        n = [l[0] for l in self._linie]
        okna = list(n)
        for dl in (2, 3):
            okna += [" ".join(n[i:i + dl]) for i in range(0, len(n) - dl + 1)]
        return [o for o in okna if o]

    def _znana(self, n: str, okna: list[str]) -> bool:
        prog = self.PROG_KROTKIE if len(n) <= 3 else self.PROG
        for o in okna:
            if podobienstwo(n, o) >= prog:
                return True
            # linia rozbita przez OCR na dwie: długi kawałek starej linii
            if len(n) >= 12 and len(n) < len(o) and _czesciowe(n, o) >= 92:
                return True
        return False

    def _bez_prefiksu(self, linia: str, okna: list[str]) -> str | None:
        """Gdy nowa linia zaczyna się od już przeczytanej (np. OCR złapał wcześniej tylko jej
        początek albo skleił starą linię z nową), zwraca samą nową końcówkę.

        Początek musi pasować do starej linii wyraźnie lepiej niż cała nowa linia – inaczej
        (np. dwie stare linie sklejone przez OCR z drobnym błędem) odcięlibyśmy z niej
        ostatni wyraz i przeczytali go jako „nowy”."""
        slowa = linia.split()
        cala = norm_linii(linia)
        najlepsze_k, najlepszy = 0, 0.0
        for o in okna:
            if len(o) < self.MIN_PREFIKS:
                continue
            prog = max(90.0, podobienstwo(cala, o) + 3)
            for k in range(1, len(slowa)):
                pref = norm_linii(" ".join(slowa[:k]))
                if len(pref) < self.MIN_PREFIKS:
                    continue
                s = podobienstwo(pref, o)
                if s >= prog and s > najlepszy:
                    najlepsze_k, najlepszy = k, s
        return " ".join(slowa[najlepsze_k:]) if najlepsze_k else None

    def szczegoly(self, linie: list[str]) -> list[tuple[str, str | None]]:
        """Dla każdej linii: (linia, część do przeczytania albo None, gdy już była przeczytana)."""
        if not self._linie:
            return [(l, l if norm_linii(l) else None) for l in linie]
        okna = self._okna()
        wynik = []
        for linia in linie:
            n = norm_linii(linia)
            if not n:
                wynik.append((linia, None))
                continue
            if max(podobienstwo(n, o) for o in okna) >= 95:
                wynik.append((linia, None))        # praktycznie identyczna z przeczytaną
                continue
            # „stara linia + nowa końcówka” sprawdzamy przed zwykłym podobieństwem (inaczej np. 88%
            # podobieństwa całości uznałoby dłuższą linię za przeczytaną i końcówka by przepadła);
            # odcinamy kolejne znane początki, bo OCR potrafi skleić dwie stare linie z nową
            reszta = linia
            for _ in range(4):
                r = self._bez_prefiksu(reszta, okna)
                if r is None or len(norm_linii(r)) < 2:
                    break
                reszta = r
            if reszta != linia:
                nr = norm_linii(reszta)
                znana = len(nr) < 2 or self._znana(nr, okna) or \
                    max(podobienstwo(nr, o) for o in okna) >= 95
                wynik.append((linia, None if znana else reszta))
            elif self._znana(n, okna):
                wynik.append((linia, None))
            else:
                wynik.append((linia, linia))
        return wynik

    def nowe(self, linie: list[str]) -> list[str]:
        """Zwraca tylko to, czego lektor jeszcze nie czytał (całe linie albo ich nowe końcówki)."""
        return [nowa for _, nowa in self.szczegoly(linie) if nowa]

    def odswiez(self, linie: list[str], teraz: float | None = None):
        """Linie z pamięci, które OCR nadal widzi, „żyją” dalej; reszta wygasa po `zycie` s."""
        teraz = time.perf_counter() if teraz is None else teraz
        widziane = [norm_linii(l) for l in linie]
        calosc = " ".join(widziane)
        for wpis in self._linie:
            n = wpis[0]
            if any(podobienstwo(n, w) >= self.PROG for w in widziane) or \
                    (len(n) >= 8 and _czesciowe(n, calosc) >= 90):
                wpis[1] = teraz
        self._linie = [w for w in self._linie if teraz - w[1] <= self.zycie]

    def dodaj(self, linie: list[str], teraz: float | None = None):
        teraz = time.perf_counter() if teraz is None else teraz
        for linia in linie:
            n = norm_linii(linia)
            if not n:
                continue
            for wpis in self._linie:
                if podobienstwo(n, wpis[0]) >= self.PROG:
                    wpis[0], wpis[1], wpis[2] = n, teraz, linia
                    break
            else:
                self._linie.append([n, teraz, linia])
        if len(self._linie) > 12:
            self._linie = self._linie[-12:]


_SAMOGLOSKI = set("aeiouyąęóAEIOUYĄĘÓ")


def wyglada_na_napis(tekst: str, pewnosci=()) -> bool:
    """Odsiewa śmieci, które OCR czasem „widzi” w liściach czy światłach (np. „ll”, „W”).

    Kwestia musi mieć co najmniej 2 litery i samogłoskę, a bardzo krótka (do 4 liter)
    – także wysoką pewność odczytu."""
    litery = [c for c in tekst if c.isalpha()]
    if len(litery) < 2 or not any(c in _SAMOGLOSKI for c in litery):
        return False
    if len(litery) < 5 and pewnosci and max(pewnosci) < 0.8:
        return False
    return True


def skladaj_linie(linie: list[str]) -> str:
    """Skleja linie napisu w jedną kwestię (z obsługą przeniesień wyrazów)."""
    wynik = ""
    for linia in linie:
        linia = linia.strip()
        if not linia:
            continue
        if not wynik:
            wynik = linia
        elif wynik.endswith("-") and len(wynik) > 1 and wynik[-2].isalpha():
            wynik = wynik[:-1] + linia
        else:
            wynik = wynik + " " + linia
    return wynik


def wczytaj_slownik(sciezka: Path) -> list[tuple[re.Pattern, str]]:
    """Czyta plik 'słowo = wymowa' (jedna para w linii, # to komentarz)."""
    reguly: list[tuple[re.Pattern, str]] = []
    if not sciezka.exists():
        return reguly
    for linia in sciezka.read_text(encoding="utf-8").splitlines():
        linia = linia.strip()
        if not linia or linia.startswith("#") or "=" not in linia:
            continue
        zrodlo, cel = (x.strip() for x in linia.split("=", 1))
        if zrodlo:
            wzor = re.compile(r"(?<![\w])" + re.escape(zrodlo) + r"(?![\w])", re.IGNORECASE)
            reguly.append((wzor, cel))
    # dłuższe frazy najpierw (np. "Vice City" przed "Vice")
    reguly.sort(key=lambda r: -len(r[0].pattern))
    return reguly


def _liczby_na_slowa(t: str) -> str:
    if _num2words is None:
        return t

    def kwota(m: re.Match) -> str:
        liczba = int(m.group(1).replace(" ", "").replace(".", "").replace(",", ""))
        try:
            return _num2words(liczba, lang="pl") + " dolarów"
        except Exception:
            return m.group(0)

    # $200 000 / $1,500 / 200 000 $
    t = re.sub(r"\$\s?(\d{1,3}(?:[ .,]\d{3})+|\d+)", kwota, t)
    t = re.sub(r"(\d{1,3}(?:[ .,]\d{3})+|\d+)\s?\$", kwota, t)

    def procent(m: re.Match) -> str:
        try:
            return _num2words(int(m.group(1)), lang="pl") + " procent"
        except Exception:
            return m.group(0)

    t = re.sub(r"(\d+)\s?%", procent, t)

    def godzina(m: re.Match) -> str:
        try:
            g, mi = int(m.group(1)), int(m.group(2))
            return _num2words(g, lang="pl") + (" " + _num2words(mi, lang="pl") if mi else "")
        except Exception:
            return m.group(0)

    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", godzina, t)

    def liczba(m: re.Match) -> str:
        surowe = m.group(0).replace(" ", "")
        try:
            return _num2words(int(surowe), lang="pl")
        except Exception:
            return m.group(0)

    t = re.sub(r"\b\d{1,3}(?: \d{3})+\b", liczba, t)  # 200 000
    t = re.sub(r"\b\d+\b", liczba, t)
    return t


_SLOWO_IMIENIA = rf"(?:[{DUZE}][{MALE}]+|[{DUZE}]{{2,}})"
_IMIE_NA_POCZATKU = re.compile(rf"^{_SLOWO_IMIENIA}(?:[ \-]{_SLOWO_IMIENIA}){{0,2}}:\s+")


def normalizuj(tekst: str, slownik: list[tuple[re.Pattern, str]] | None = None,
               usun_imiona: bool = False) -> str:
    """Przygotowuje tekst z OCR do przeczytania przez lektora."""
    t = tekst.strip()
    if not t:
        return ""
    # Typowe pomyłki OCR w interpunkcji
    t = t.replace("…", "...").replace("„", "").replace("”", "").replace("“", "")
    t = t.replace("«", "").replace("»", "").replace('"', "")
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)          # spacja przed znakiem
    # Nazwa postaci na początku ("LUCIA: ..." albo "Lucia: ..."), gdy w grze
    # są włączone imiona mówiących
    if usun_imiona:
        t = _IMIE_NA_POCZATKU.sub("", t)
    # Myślnik dialogowy na początku i myślniki w środku zdania
    t = re.sub(r"^[\-–—]\s*", "", t)
    t = re.sub(r"\s+[\-–—]\s+", ", ", t)
    # TEKST KRZYCZANY wielkimi literami -> zwykły (Piper czyta go poprawniej)
    litery = [c for c in t if c.isalpha()]
    if len(litery) > 6 and sum(c.isupper() for c in litery) / len(litery) > 0.8:
        t = t.lower()
        t = t[:1].upper() + t[1:]
    t = _liczby_na_slowa(t)
    for wzor, cel in slownik or []:
        t = wzor.sub(cel, t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    # Wielokropek: na końcu i przed nowym zdaniem jak kropka, w środku jak przecinek
    t = re.sub(r"\.\.\.$", ".", t)
    t = re.sub(rf"\.\.\.(?=\s+[{DUZE}])", ".", t)
    t = re.sub(r"\.\.\.(?=\s)", ",", t)
    t = re.sub(r"\s*[-–—]+$", "", t)                # urwany wyraz/myślnik na końcu
    if t and t[-1] not in ".!?":
        t += "."
    return t
