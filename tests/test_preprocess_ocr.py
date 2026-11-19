"""Obróbka obrazu + prawdziwy OCR (Tesseract z polskim) na syntetycznych klatkach.

Na Windowsie aplikacja używa OCR Windowsa; tu Tesseract sprawdza, czy obraz
po obróbce jest czytelny dla silnika OCR.
"""

import pytest
from rapidfuzz import fuzz
from synth import background, draw_subtitle

from lektor import preprocess, textproc
from lektor.ocr import OcrError, TesseractOcr
from lektor.roi import to_pixels

ROI = (0.12, 0.70, 0.76, 0.28)
CASES = [
    "Musimy się stąd wynosić, zanim przyjedzie policja.",
    "Jason, nie panikuj.\nWszystko jest pod kontrolą.",
    "Zażółć gęślą jaźń, a potem jedź na plażę.",
]


@pytest.fixture(scope="module")
def ocr():
    try:
        return TesseractOcr("pl")
    except OcrError as exc:
        pytest.skip(str(exc))


def _read(ocr, frame, mode="outline"):
    height, width = frame.shape[:2]
    x, y, w, h = to_pixels(ROI, width, height)
    prepared = preprocess.prepare(frame[y : y + h, x : x + w], mode, 170, 110, preprocess.auto_scale(height))
    if not prepared.has_text:
        return "", prepared
    return textproc.clean_lines(ocr.recognize(prepared.image)), prepared


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (3840, 2160)])
@pytest.mark.parametrize("text", CASES)
def test_outline_mode_reads_polish_subtitles(ocr, size, text):
    frame = draw_subtitle(background(*size, t=1.3, seed=7), text)
    got, prepared = _read(ocr, frame)
    expected = " ".join(text.split())
    assert prepared.has_text
    assert fuzz.ratio(got, expected) >= 92, (got, expected)


def test_colored_names_are_kept(ocr):
    segments = [
        ("Powiedz ", (255, 255, 255)),
        ("Lucii", (93, 182, 229)),
        (", że ", (255, 255, 255)),
        ("Jason", (171, 237, 171)),
        (" już jedzie.", (255, 255, 255)),
    ]
    frame = draw_subtitle(background(1920, 1080, t=0.4, seed=3), "", segments=segments)
    got, _ = _read(ocr, frame)
    assert fuzz.ratio(got, "Powiedz Lucii, że Jason już jedzie.") >= 92, got


def test_empty_frame_gives_no_text(ocr):
    frame = background(1920, 1080, t=2.0, seed=11)
    got, _ = _read(ocr, frame)
    assert not textproc.looks_like_text(got), got


def test_outline_beats_raw_on_busy_background(ocr):
    text = "Zabierz torbę i spotkamy się przy przystani."
    frame = draw_subtitle(background(1920, 1080, t=0.9, seed=5), text)
    outline, _ = _read(ocr, frame, "outline")
    assert fuzz.ratio(outline, text) >= 92, outline
