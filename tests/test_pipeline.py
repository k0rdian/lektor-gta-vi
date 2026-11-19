"""Cały potok na syntetycznym filmie: źródło plikowe → obróbka → OCR → filtry → lektor."""

import json

import cv2
import pytest
from fakes import FakeAudio, FakeTTS
from rapidfuzz import fuzz
from synth import background, draw_subtitle, make_mp3

from lektor import config
from lektor.app import App
from lektor.ocr import OcrError, TesseractOcr

FPS = 10
TIMELINE = [
    (0.0, 0.6, None),
    (0.6, 3.2, "Musimy się stąd wynosić, zanim przyjedzie policja."),
    (3.2, 3.6, None),
    (3.6, 6.2, "Jason, nie panikuj.\nWszystko jest pod kontrolą."),
    (6.2, 8.2, "Witamy w Vice City."),
    (8.2, 8.6, None),
    (8.6, 10.2, "Witamy w Vice City."),  # ta sama kwestia tuż po — ma być pominięta
    (10.2, 11.0, None),
]
FLICKER = {15, 16, 17}  # napis znika na chwilę w środku pierwszej kwestii


def _make_video(path, width=1280, height=720):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (width, height))
    for i in range(int(11.0 * FPS)):
        t = i / FPS
        frame = background(width, height, t=t, seed=i)
        text = next((s for a, b, s in TIMELINE if a <= t < b), None)
        if text and i not in FLICKER:
            frame = draw_subtitle(frame, text)
        writer.write(frame)
    writer.release()


def test_end_to_end_reads_each_line_once(tmp_path):
    try:
        ocr = TesseractOcr("pl")
    except OcrError as exc:
        pytest.skip(str(exc))
    video = tmp_path / "gta.mp4"
    _make_video(video)
    (tmp_path / "roi.json").write_text(json.dumps({"file:gta.mp4": [0.12, 0.70, 0.76, 0.28]}), encoding="utf-8")

    cfg = config.load(None)
    cfg.source.type = "file"
    cfg.source.file = str(video)
    cfg.ocr.interval_ms = 100
    tts, audio = FakeTTS(make_mp3(tmp_path / "beep.mp3")), FakeAudio(0.3)
    app = App(cfg, tmp_path, ocr=ocr, tts=tts, audio_factory=lambda: audio, interactive=False)
    assert app.run() == 0

    spoken = [text for text, _ in tts.calls]
    assert len(spoken) == 3, spoken
    assert fuzz.ratio(spoken[0], "Musimy się stąd wynosić, zanim przyjedzie policja.") >= 92
    assert spoken[1].startswith("Dżejson") and fuzz.partial_ratio(spoken[1], "Wszystko jest pod kontrolą.") >= 92
    assert "Wajs Siti" in spoken[2]
    assert len(audio.played) == 3
