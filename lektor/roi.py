"""Zaznaczanie myszką i zapamiętywanie obszaru z napisami."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import log

# Domyślny obszar: dolny środek ekranu (x, y, szerokość, wysokość jako ułamki klatki).
DEFAULT_ROI = (0.12, 0.76, 0.76, 0.20)
WINDOW = "Lektor - zaznacz obszar napisow"
HINT = "Zaznacz mysza pas z napisami (z zapasem na 2 linie). ENTER = zapisz, ESC = anuluj"


def to_pixels(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    px = min(max(0, round(x * width)), width - 8)
    py = min(max(0, round(y * height)), height - 8)
    pw = min(max(8, round(w * width)), width - px)
    ph = min(max(8, round(h * height)), height - py)
    return px, py, pw, ph


def _screen_limits() -> tuple[int, int]:
    if os.name == "nt":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0) * 0.9), int(user32.GetSystemMetrics(1) * 0.85)
        except Exception:  # noqa: BLE001
            pass
    return 1600, 900


def select(frame: np.ndarray, current: tuple[float, float, float, float] | None = None):
    """Pokazuje klatkę i pozwala zaznaczyć prostokąt. Zwraca ułamki (x, y, w, h) albo None."""
    height, width = frame.shape[:2]
    max_w, max_h = _screen_limits()
    scale = min(1.0, max_w / width, max_h / height)
    view = frame if scale >= 1.0 else cv2.resize(
        frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )
    view = np.ascontiguousarray(view).copy()
    if current is not None:
        x, y, w, h = to_pixels(current, view.shape[1], view.shape[0])
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 215, 255), 2)
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(view, HINT, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    log.info("Zaznacz obszar z napisami w oknie, które się otworzyło (ENTER zapisuje).")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    x, y, w, h = cv2.selectROI(WINDOW, view, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)
    if w < 5 or h < 5:
        return None
    vh, vw = view.shape[:2]
    return (x / vw, y / vh, w / vw, h / vh)


class RoiStore:
    """Zapisane obszary w roi.json — osobno dla każdego źródła obrazu."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, list[float]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warn(f"Nie mogę odczytać {self.path.name} ({exc}) — zaczynam od zera.")

    def get(self, signature: str) -> tuple[float, float, float, float] | None:
        value = self._data.get(signature)
        if isinstance(value, list) and len(value) == 4:
            x, y, w, h = (float(v) for v in value)
            if 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1:
                return x, y, w, h
        return None

    def set(self, signature: str, roi: tuple[float, float, float, float]) -> None:
        self._data[signature] = [round(float(v), 5) for v in roi]
        try:
            self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.warn(f"Nie mogę zapisać {self.path.name}: {exc}")
