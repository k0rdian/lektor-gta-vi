"""Proste, wątkowo bezpieczne logowanie do konsoli i do pliku lektor.log.

Szczegóły (każdy odczyt OCR, tekst wysłany do syntezatora) trafiają zawsze do pliku,
a do konsoli tylko w trybie --verbose — plik pozwala odtworzyć, co się działo.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_verbose = False
_file = None
_path: Path | None = None
_written = 0
MAX_BYTES = 5_000_000  # potem lektor.log → lektor.log.1


def setup(verbose: bool = False, path: Path | None = None) -> None:
    global _verbose
    _verbose = verbose
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    if path is not None:
        with _lock:
            _open(Path(path))


def _open(path: Path) -> None:
    global _file, _path, _written
    if _file is not None and _path == path:
        return
    _close()
    try:
        if path.exists() and path.stat().st_size > MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
        _file = open(path, "a", encoding="utf-8", buffering=1)
        _file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        _path, _written = path, 0
    except OSError:
        _file, _path = None, None


def _close() -> None:
    global _file, _path
    if _file is not None:
        try:
            _file.close()
        except OSError:
            pass
    _file, _path = None, None


def close() -> None:
    with _lock:
        _close()


def is_verbose() -> bool:
    return _verbose


def _emit(prefix: str, msg: str, console: bool = True) -> None:
    global _written
    line = f"[{time.strftime('%H:%M:%S')}] {prefix}{msg}"
    with _lock:
        if console:
            print(line, flush=True)
        if _file is not None:
            try:
                _file.write(line + "\n")
                _written += len(line) + 1
                if _written > MAX_BYTES and _path is not None:
                    path = _path
                    _close()
                    _open(path)
            except (OSError, ValueError):
                pass


def info(msg: str) -> None:
    _emit("", msg)


def warn(msg: str) -> None:
    _emit("UWAGA: ", msg)


def error(msg: str) -> None:
    _emit("BŁĄD: ", msg)


def debug(msg: str) -> None:
    """Szczegół: zawsze do pliku, do konsoli tylko z --verbose."""
    _emit("  · ", msg, console=_verbose)


def short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
