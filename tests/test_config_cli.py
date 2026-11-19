from pathlib import Path

import pytest
from synth import background, draw_subtitle

from lektor import config, log
from lektor import __main__ as cli

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_config_parses_without_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(log, "warn", warnings.append)
    cfg = config.load(ROOT / "config.toml")
    assert warnings == []
    assert cfg.tts.voice == "pl-PL-MarekNeural"
    assert cfg.source.type == "capture" and cfg.source.device == "HD60"
    assert cfg.replacements == {"Vice City": "Wajs Siti", "Jason": "Dżejson"}
    assert cfg.ocr.upscale == 0.0 and isinstance(cfg.ocr.upscale, float)
    assert cfg.text.repeat_window_s == 12.0 and isinstance(cfg.text.repeat_window_s, float)
    defaults = config.to_namespace(config.DEFAULTS)
    for section in ("source", "ocr", "text", "tts", "ducking", "hotkeys", "debug"):
        assert vars(getattr(cfg, section)).keys() == vars(getattr(defaults, section)).keys()


def test_bad_values_fall_back_to_defaults(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(log, "warn", warnings.append)
    path = tmp_path / "c.toml"
    path.write_text(
        '[ocr]\ninterval_ms = "szybko"\nbrightness = 160\n[tts]\nvolume = 2\n[nieznana]\nx = 1\n'
        '[source]\ndevice = 1\n[replacements]\n"GTA" = "Dżi Ti Ej"\n',
        encoding="utf-8",
    )
    cfg = config.load(path)
    assert cfg.ocr.interval_ms == 200 and cfg.ocr.brightness == 160
    assert cfg.tts.volume == 2.0 and cfg.source.device == 1
    assert cfg.replacements == {"GTA": "Dżi Ti Ej"}
    assert len(warnings) == 2


def test_broken_toml_gives_readable_error(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[ocr\nbrightness = ", encoding="utf-8")
    with pytest.raises(ValueError) as info:
        config.load(path)
    assert "c.toml" in str(info.value)


def test_cli_overrides():
    cfg = config.load(None)
    args = cli.build_parser().parse_args(
        ["--source", "file:C:/nagrania/gta.mp4", "--voice", "pl-PL-ZofiaNeural", "--preview", "--device", "0"]
    )
    cli.apply_overrides(cfg, args)
    assert cfg.source.type == "file" and cfg.source.file == "C:/nagrania/gta.mp4"
    assert cfg.tts.voice == "pl-PL-ZofiaNeural" and cfg.debug.preview and cfg.source.device == "0"


def test_cli_test_ocr_runs_all_modes(tmp_path, monkeypatch):
    pytest.importorskip("pytesseract")
    import cv2

    from lektor import roi

    image = draw_subtitle(background(1920, 1080, t=0.5, seed=2), "Gdzie jest mój samochód?")
    path = tmp_path / "zrzut ekranu ż.png"
    cv2.imencode(".png", image)[1].tofile(str(path))
    monkeypatch.setattr(roi, "select", lambda frame, current=None: (0.12, 0.70, 0.76, 0.28))
    monkeypatch.setattr(cli, "BASE_DIR", tmp_path)
    lines = []
    monkeypatch.setattr(log, "info", lines.append)
    cfg = config.load(None)
    cfg.ocr.engine = "tesseract"
    assert cli.cmd_test_ocr(cfg, path) == 0
    outline = next(line for line in lines if line.startswith("[outline]"))
    assert "Gdzie jest mój samochód?" in outline
    assert (tmp_path / "debug_samples" / "test-ocr-outline.png").exists()
