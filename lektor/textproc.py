"""Czyszczenie tekstu z OCR, stabilizacja odczytów i wykrywanie powtórek.

Ten moduł nie zależy od Windowsa, więc da się go testować wszędzie.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Mapping

from rapidfuzz import fuzz

PL_VOWELS = set("aąeęioóuyAĄEĘIOÓUY")
SINGLE_LETTER_WORDS = set("aiouwzAIOUWZ")
JUNK_EDGES = " \t|_~•*·=<>^`"

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")
_LETTER_RUNS = re.compile(r"[^\W\d_]+")
_SPEAKER = re.compile(r"^\s*[A-ZĄĆĘŁŃÓŚŹŻ][\wĄĆĘŁŃÓŚŹŻąćęłńóśźż.'\- ]{0,24}:\s+")
_WORD_HYPHEN_END = re.compile(r"\w-$")


def count_letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def norm_key(text: str) -> str:
    """Klucz do porównań: małe litery, bez interpunkcji i podwójnych spacji."""
    return _WS.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def clean_lines(lines: Iterable[tuple[str, float]], min_height_ratio: float = 0.55) -> str:
    """Skleja linie z OCR w jedną kwestię.

    Odrzuca linie bez liter i linie dużo niższe od najwyższej (zwykle śmieci
    z tła), łączy wyrazy przeniesione myślnikiem.
    """
    items = [(" ".join(text.split()), float(height or 0)) for text, height in lines]
    items = [(text, height) for text, height in items if count_letters(text) >= 1]
    if not items:
        return ""
    tallest = max(height for _, height in items)
    if tallest > 0:
        items = [
            (text, height)
            for text, height in items
            if height <= 0 or height >= min_height_ratio * tallest
        ]
    out = ""
    for text, _ in items:
        text = text.strip(JUNK_EDGES)
        if not text:
            continue
        if out and _WORD_HYPHEN_END.search(out) and text[:1].islower():
            out = out[:-1] + text
        elif out:
            out = f"{out} {text}"
        else:
            out = text
    return _WS.sub(" ", out).strip(JUNK_EDGES)


def looks_like_text(text: str, min_letters: int = 2) -> bool:
    """Prosty test, czy odczyt wygląda na zdanie, a nie na szum z tła."""
    if not text:
        return False
    letters = count_letters(text)
    if letters < min_letters:
        return False
    non_space = sum(1 for ch in text if not ch.isspace())
    if letters / max(1, non_space) < 0.6:
        return False
    words = _LETTER_RUNS.findall(text)
    total = sum(len(word) for word in words)
    if total == 0:
        return False
    good = 0
    for word in words:
        if len(word) >= 2 and any(ch in PL_VOWELS for ch in word):
            good += len(word)
        elif len(word) == 1 and word in SINGLE_LETTER_WORDS:
            good += 1
    return good / total >= 0.6


def prepare_for_speech(
    text: str, replacements: Mapping[str, str] | None = None, strip_speaker: bool = False
) -> str:
    """Zamiany wymowy (np. „Vice City” → „Wajs Siti”) i opcjonalne usuwanie „Imię:”."""
    if strip_speaker:
        text = _SPEAKER.sub("", text, count=1)
    if replacements:
        for source in sorted(replacements, key=len, reverse=True):
            target = replacements[source]
            if source:
                text = re.sub(r"(?<!\w)" + re.escape(source), lambda _m, t=target: t, text)
    return _WS.sub(" ", text).strip()


# Znaki, które syntezator czyta na głos jako nazwy („średnik”, „ukośnik”, „tylda”…).
# W napisach z gry to prawie zawsze błąd OCR, np. ogonek albo kreska odczytane osobno.
_JUNK_MARKS = re.compile(r"[;|\\_~^*#@<>‹›\[\]{}`´¨°¸˛˙˝ˇ¯•·¦¬§©®™]|(?<!\d)\$(?!\d)|(?<!\d)(?<!\d )%")  # „500$”, „50%” zostają
_WORD_SEPARATORS = re.compile(r"(?<!\d)[&/=+](?!\d)")  # „km/h”, „Rock & Roll”; „1/2”, „2+2” zostają
_SPACE_BEFORE_MARK = re.compile(r"\s+([.,?!])")
_POLISH_LETTERS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def _drop_junk(match: re.Match) -> str:
    text, start, end = match.string, match.start(), match.end()
    left = text[start - 1] if start > 0 else " "
    right = text[end] if end < len(text) else " "
    return "" if left.isalpha() and right.isalpha() else " "  # „si;ę” → „się”, „tak ; nie” → „tak nie”


def strip_spoken_symbols(text: str) -> str:
    """Usuwa znaki, które lektor wymówiłby jako nazwy symboli."""
    text = _WORD_SEPARATORS.sub(" ", _JUNK_MARKS.sub(_drop_junk, text))
    return _SPACE_BEFORE_MARK.sub(r"\1", _WS.sub(" ", text)).strip()


def read_quality(text: str) -> int:
    """Ocena odczytu OCR: polskie litery na plus, znaki-śmieci mocno na minus."""
    return sum(ch in _POLISH_LETTERS for ch in text) - 3 * len(_JUNK_MARKS.findall(text))


# Znaki, przy których syntezator robi pauzę. Liczby „1,5” i godziny „12:30” zostają.
_COMMA = re.compile(r"(?<!\d),|,(?!\d)")
_PAUSE_MARKS = re.compile(r"[;:](?=\s|$)")
_DASH = re.compile(r"(^|\s)[-–—]+(?=\s|$|[^\W\d_])")  # „- Co?”, „ – ”, „-Nic”; nie „czarno-biały”
_ELLIPSIS = re.compile(r"\s*(?:\.{2,}|…)+\s*")
_QUOTES = re.compile(r"[\"„”“«»]")
_SPACE_BEFORE_END = re.compile(r"\s+([.?!])")
_DOUBLE_END = re.compile(r"([.?!])\.+")


def _ellipsis(match: re.Match) -> str:
    rest = match.string[match.end() :]
    if not rest:
        return "."  # wielokropek na końcu kwestii = zwykła kropka
    return ". " if rest[:1].isupper() else " "  # przed wielką literą kończy zdanie


def fluent_text(text: str, skip_commas: bool = True) -> str:
    """Tekst dla syntezatora: bez znaków, przy których lektor robi długie pauzy.

    Przecinki, średniki, dwukropki, myślniki, wielokropki w środku zdania i cudzysłowy
    znikają. Kropki, pytajniki i wykrzykniki zostają — od nich zależy intonacja.
    """
    text = strip_spoken_symbols(text)
    if not skip_commas or not text:
        return text
    text = _ELLIPSIS.sub(_ellipsis, text)
    text = _QUOTES.sub("", text)
    text = _DASH.sub(r"\1", text)
    text = _PAUSE_MARKS.sub(" ", text)
    text = _COMMA.sub(" ", text)
    text = _SPACE_BEFORE_END.sub(r"\1", _WS.sub(" ", text))
    text = _DOUBLE_END.sub(r"\1", text)
    return text.strip().lstrip(".?! ").strip()


class Stabilizer:
    """Przepuszcza tekst dopiero, gdy kilka kolejnych odczytów OCR się zgadza.

    Każdy napis jest zwracany raz, nawet jeśli wisi na ekranie kilka sekund.
    Spośród zgodnych odczytów wybiera najczystszy (z polskimi znakami, bez śmieci).
    """

    def __init__(self, required_reads: int = 2, similarity: float = 90) -> None:
        self.required = max(1, int(required_reads))
        self.similarity = similarity
        self.reset()

    def reset(self) -> None:
        self._candidate: str | None = None
        self._key = ""
        self._count = 0
        self._emitted = False
        self.first_seen: float | None = None  # kiedy pierwszy raz odczytano bieżący napis

    def feed(self, text: str, now: float | None = None) -> str | None:
        if not text:
            self.reset()
            return None
        key = norm_key(text)
        if not key:
            self.reset()
            return None
        if self._candidate is not None and fuzz.ratio(key, self._key) >= self.similarity:
            self._count += 1
            if read_quality(text) >= read_quality(self._candidate):
                self._candidate = text  # gorszy odczyt nie zastępuje lepszego
            self._key = key
        else:
            self._candidate, self._key = text, key
            self._count = 1
            self._emitted = False
            self.first_seen = now
        if self._count >= self.required and not self._emitted:
            self._emitted = True
            return self._candidate
        return None


class RepeatFilter:
    """Pilnuje, żeby lektor nie czytał drugi raz tej samej kwestii.

    - tę samą (lub prawie tę samą) kwestię w oknie czasowym pomija,
    - fragment kwestii już przeczytanej pomija (OCR złapał tylko część),
    - jeśli nowy odczyt to poprzednia kwestia plus dalszy ciąg, zwraca tylko dalszy ciąg.
    Napis, który wciąż wisi na ekranie, odświeża swój czas — nie wróci po upływie okna.
    """

    def __init__(self, window_s: float = 12.0, similarity: float = 88, min_letters: int = 2) -> None:
        self.window_s = float(window_s)
        self.similarity = similarity
        self.min_letters = min_letters
        self._history: deque[list] = deque()  # [klucz, tekst, czas]

    def clear(self) -> None:
        self._history.clear()

    def _purge(self, now: float) -> None:
        while self._history and now - self._history[0][2] > self.window_s:
            self._history.popleft()

    def _remember(self, key: str, flat: str, now: float) -> None:
        self._history.append([key, flat, now])
        while len(self._history) > 50:
            self._history.popleft()

    def _is_known(self, key: str, now: float) -> bool:
        for entry in reversed(self._history):
            old_key = entry[0]
            same = fuzz.ratio(key, old_key) >= self.similarity
            fragment = (
                len(key) <= len(old_key)
                and (len(key) >= 0.5 * len(old_key) or len(key) >= 12)
                and fuzz.partial_ratio(key, old_key) >= 92
            )
            if same or fragment:
                entry[2] = now
                return True
        return False

    def process(self, text: str, now: float) -> str | None:
        self._purge(now)
        key = norm_key(text)
        if not key:
            return None
        if self._is_known(key, now):
            return None
        flat = " ".join(text.split())
        if self._history:
            old_key, old_flat, _t = self._history[-1]
            if len(old_key) >= 8 and len(key) > len(old_key) + 5:
                alignment = fuzz.partial_ratio_alignment(old_flat.lower(), flat.lower())
                if alignment is not None and alignment.score >= 92 and alignment.dest_start <= 3:
                    tail = flat[alignment.dest_end :].lstrip(" ,.;:-–—" + JUNK_EDGES).rstrip(JUNK_EDGES)
                    self._remember(key, flat, now)
                    if looks_like_text(tail, max(self.min_letters, 4)) and any(
                        len(word) >= 3 for word in _LETTER_RUNS.findall(tail)
                    ):
                        return tail
                    return None
        self._remember(key, flat, now)
        return text
