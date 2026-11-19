"""Wczytywanie config.toml i scalanie z wartościami domyślnymi."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import log

DEFAULTS: dict[str, Any] = {
    "source": {
        "type": "capture",  # capture | screen | file
        "device": "HD60",  # capture: fragment nazwy urządzenia albo jego numer
        "backend": "dshow",  # dshow | msmf
        "width": 1920,
        "height": 1080,
        "fps": 0,  # 0 = domyślne urządzenia
        "fourcc": "",  # np. "NV12", "YUY2", "MJPG"; pusto = domyślny format
        "monitor": 1,  # screen: numer monitora (1 = główny)
        "file": "",  # file: ścieżka do obrazka albo filmu (testy)
    },
    "ocr": {
        "engine": "windows",  # windows | tesseract
        "language": "pl",
        "interval_ms": 200,
        "preprocess": "outline",  # outline | bright | gray | raw
        "brightness": 170,
        "outline": 110,
        "upscale": 0.0,  # 0 = automatycznie
        "tesseract_cmd": "",
    },
    "text": {
        "min_letters": 2,
        "stable_reads": 2,
        "stable_similarity": 90,
        "repeat_window_s": 12.0,
        "repeat_similarity": 88,
        "strip_speaker_names": False,
        "ignore": [],
    },
    "replacements": {
        "Vice City": "Wajs Siti",
        "Jason": "Dżejson",
    },
    "tts": {
        "voice": "pl-PL-MarekNeural",
        "rate": 20,  # tempo bazowe w %
        "rate_step": 15,  # +% za każdą sekundę spóźnienia ponad catch_up_s
        "rate_max": 60,
        "catch_up_s": 1.5,  # od takiego spóźnienia lektor przyspiesza
        "skip_commas": True,  # bez pauz na przecinkach, myślnikach, wielokropkach
        "max_pause_ms": 180,  # dłuższe pauzy w środku kwestii są skracane; 0 = bez zmian
        "pitch": 0,
        "volume": 1.0,
        "output_device": "",
        "gapless": True,  # jeden stały strumień audio, bez przerw między kwestiami
        "max_delay_s": 0.0,  # zgodność ze starymi ustawieniami; kolejka nie ma limitu wieku
        "timeout_s": 10.0,
        "retries": 2,
        "greeting": "Lektor gotowy.",
    },
    "ducking": {
        "enabled": False,
        "process": "obs64.exe",
        "level": 0.35,
    },
    "hotkeys": {
        "enabled": True,
        "pause": "<ctrl>+<alt>+p",
        "repeat": "<ctrl>+<alt>+r",
        "skip": "<ctrl>+<alt>+k",
        "select_region": "<ctrl>+<alt>+w",
        "preview": "<ctrl>+<alt>+d",
        "quit": "<ctrl>+<alt>+q",
    },
    "debug": {
        "preview": False,
        "verbose": False,
        "save_samples": False,
    },
}

# Sekcje, które użytkownik podaje w całości (zastępują domyślne, nie są scalane).
FREEFORM_SECTIONS = {"replacements"}


def load(path: Path | None) -> SimpleNamespace:
    data = copy.deepcopy(DEFAULTS)
    if path is not None:
        if path.exists():
            try:
                with open(path, "rb") as handle:
                    user = tomllib.load(handle)
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f"Błąd w pliku {path.name}: {exc}") from exc
            _merge(data, user)
        else:
            log.warn(f"Nie znalazłem {path} — używam ustawień domyślnych.")
    return to_namespace(data)


def to_namespace(data: dict[str, Any]) -> SimpleNamespace:
    sections = {}
    for name, value in data.items():
        if name in FREEFORM_SECTIONS or not isinstance(value, dict):
            sections[name] = value
        else:
            sections[name] = SimpleNamespace(**value)
    return SimpleNamespace(**sections)


def _merge(base: dict[str, Any], user: dict[str, Any], prefix: str = "") -> None:
    for key, value in user.items():
        name = prefix + key
        if not prefix and key in FREEFORM_SECTIONS:
            if isinstance(value, dict):
                base[key] = {str(k): str(v) for k, v in value.items()}
            else:
                log.warn(f"Sekcja [{key}] powinna być tabelą — pomijam.")
            continue
        if key not in base:
            log.warn(f"Nieznane ustawienie „{name}” — pomijam.")
            continue
        current = base[key]
        if isinstance(current, dict):
            if isinstance(value, dict):
                _merge(current, value, name + ".")
            else:
                log.warn(f"„{name}” powinno być sekcją — pomijam.")
            continue
        base[key] = _coerce(current, value, name)


def _coerce(default: Any, value: Any, name: str) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(default, int):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value) if float(value).is_integer() else value
        if isinstance(value, str) and name == "source.device":
            return value
    elif isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(default, str):
        if isinstance(value, str):
            return value
        if isinstance(value, int) and name == "source.device":
            return value
    elif isinstance(default, list):
        if isinstance(value, list):
            return [str(v) for v in value]
    log.warn(f"Zła wartość „{name}” = {value!r} — zostawiam {default!r}.")
    return default
