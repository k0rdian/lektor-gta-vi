"""Regresja prawdziwego OCR Windowsa: polskie znaki i obraz zgłoszony przez użytkownika."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from synth import background, draw_subtitle
from lektor import preprocess, textproc
from lektor.ocr import WindowsOcr, OcrError

pytestmark = pytest.mark.skipif(os.name != "nt", reason="wymaga OCR Windowsa")
EXPECTED = "Myślałem, że ktoś taki stary jak ty będzie wiedział, kiedy się zamknąć."


@pytest.fixture(scope="module")
def polish_ocr():
    try:
        engine = WindowsOcr("pl")
    except OcrError as exc:
        pytest.skip(str(exc))
    yield engine
    engine.close()


@pytest.mark.parametrize("resize", [0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
def test_reported_subtitles_at_different_video_sizes(polish_ocr, resize):
    path = Path(__file__).parent / "data" / "polish_subtitles.png"
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA if resize < 1 else cv2.INTER_CUBIC)
    scale = preprocess.auto_scale(2160, crop=image)
    prepared = preprocess.prepare(image, scale=scale)
    assert prepared.has_text
    got = textproc.clean_lines(polish_ocr.recognize(prepared.image))
    assert got.casefold() == EXPECTED.casefold(), got


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (3840, 2160)])
@pytest.mark.parametrize("text", [
    "Zażółć gęślą jaźń, a potem jedź na plażę.",
    "Łódź, żółw, źródło, ćma, śnieg, koń, wąż i gęś.",
    "Musimy się stąd wynosić, zanim przyjedzie policja.",
])
def test_polish_diacritics_on_busy_background(polish_ocr, size, text):
    width, height = size
    frame = draw_subtitle(background(width, height, t=1.3, seed=7), text)
    crop = frame[round(height * .70):round(height * .98), round(width * .05):round(width * .95)]
    prepared = preprocess.prepare(crop, scale=preprocess.auto_scale(height, crop=crop))
    got = textproc.clean_lines(polish_ocr.recognize(prepared.image))
    assert got.casefold() == text.casefold(), got


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (3840, 2160)])
def test_more_tolerant_outline_does_not_read_empty_background(polish_ocr, size):
    width, height = size
    frame = background(width, height, t=1.3, seed=7)
    crop = frame[round(height * .70):round(height * .98), round(width * .05):round(width * .95)]
    prepared = preprocess.prepare(crop, scale=preprocess.auto_scale(height, crop=crop))
    got = textproc.clean_lines(polish_ocr.recognize(prepared.image)) if prepared.has_text else ""
    assert not textproc.looks_like_text(got), got
