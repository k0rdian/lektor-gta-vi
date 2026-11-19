"""Syntetyczne klatki „z gry”: ruchliwe tło + napisy z obwódką jak w GTA."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def background(width: int, height: int, t: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = np.empty((height, width, 3), np.float32)
    base[..., 0] = 110 + 80 * np.sin(6 * x + t) * y
    base[..., 1] = 90 + 60 * np.cos(4 * y + 0.7 * t)
    base[..., 2] = 60 + 110 * x
    base += rng.normal(0, 16, (height, width, 3)).astype(np.float32)
    image = np.clip(base, 0, 255).astype(np.uint8)
    # jasne plamy (światła, niebo) — także pod napisami
    cx = int(width * (0.32 + 0.08 * np.sin(t)))
    cv2.circle(image, (cx, int(height * 0.86)), int(height * 0.06), (235, 235, 240), -1)
    cv2.rectangle(
        image, (int(width * 0.62), int(height * 0.80)), (int(width * 0.80), int(height * 0.96)), (205, 222, 228), -1
    )
    cv2.rectangle(image, (int(width * 0.05), int(height * 0.10)), (int(width * 0.45), int(height * 0.35)), (240, 240, 240), -1)
    return image


def draw_subtitle(image: np.ndarray, text: str, segments=None) -> np.ndarray:
    """Napis wyśrodkowany przy dole. segments = [(fragment, (R,G,B)), ...] dla kolorowych słów."""
    height, width = image.shape[:2]
    size = int(height * 0.032)
    font = _font(size)
    stroke = max(2, size // 12)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    lines = [[(line, (255, 255, 255))] for line in text.split("\n")] if segments is None else [segments]
    step = int(size * 1.3)
    y = int(height * 0.84) - (len(lines) - 1) * step
    for parts in lines:
        total = sum(draw.textlength(part, font=font) for part, _ in parts)
        x = (width - total) / 2
        for part, color in parts:
            draw.text((x, y), part, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += draw.textlength(part, font=font)
        y += step
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_mp3(path: Path, seconds: float = 0.4) -> bytes:
    if shutil.which("ffmpeg") is None:
        import pytest

        pytest.skip("brak ffmpeg — test dźwięku pominięty")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "24000", "-ac", "1", "-b:a", "48k", str(path)],
        check=True,
    )
    return path.read_bytes()
