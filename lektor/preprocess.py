"""Przygotowanie wycinka z napisami pod OCR.

Domyślny tryb "outline" zakłada typowe napisy z gier: jasne litery
(dowolnego koloru) z ciemną obwódką albo cieniem. Zostawia tylko jasne
kształty o rozmiarze liter, otoczone ciemnym pierścieniem, i zwraca
czarny tekst na białym tle — z takim obrazem OCR radzi sobie najlepiej.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MODES = ("outline", "bright", "gray", "raw")


@dataclass
class Prepared:
    image: np.ndarray  # obraz dla OCR (szary albo BGR)
    has_text: bool  # czy w ogóle warto wołać OCR
    components: int = 0


def auto_scale(frame_height: int, upscale: float = 0.0, crop: np.ndarray | None = None, brightness: int = 170) -> float:
    """Powiększenie wycinka: OCR lepiej czyta litery wysokie na 40+ px."""
    if upscale and upscale > 0:
        return float(upscale)
    base = 2.0 if frame_height <= 1200 else 1.0
    if base == 1.0 and crop is not None and crop.size:
        value = crop if crop.ndim == 2 else np.max(crop[:, :, :3], axis=2)
        _, _, stats, _ = cv2.connectedComponentsWithStats((value >= brightness).astype(np.uint8), connectivity=8)
        letter_height = _letter_height(stats, *value.shape)
        if letter_height and letter_height < 25:
            return 2.0
    return base


def _letter_height(stats: np.ndarray, height: int, width: int, scale: float = 1.0) -> float:
    """Wysokość korpusów liter; pomija drobne akcenty i duże plamy tła."""
    bodies = (
        (stats[:, cv2.CC_STAT_AREA] >= 3 * scale * scale)
        & (stats[:, cv2.CC_STAT_HEIGHT] >= max(8 * scale, 0.04 * height))
        & (stats[:, cv2.CC_STAT_HEIGHT] < 0.9 * height)
        & (stats[:, cv2.CC_STAT_WIDTH] < 0.2 * width)
    )
    bodies[0] = False
    heights = stats[bodies, cv2.CC_STAT_HEIGHT]
    return float(np.median(heights)) if heights.size else 0.0


def _pad(image: np.ndarray, pad: int, value) -> np.ndarray:
    if pad <= 0:
        return image
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


def prepare(
    crop: np.ndarray | None,
    mode: str = "outline",
    brightness: int = 170,
    outline: int = 110,
    scale: float = 2.0,
    pad: int = 12,
) -> Prepared:
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return Prepared(np.full((16, 16), 255, np.uint8), False)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    if scale and abs(scale - 1.0) > 1e-3:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if mode == "raw":
        return Prepared(_pad(crop, pad, (255, 255, 255)), True)
    if mode == "gray":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(255 - gray, None, 0, 255, cv2.NORM_MINMAX)
        return Prepared(_pad(gray, pad, 255), True)

    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    bright = (value >= brightness).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    height, width = value.shape
    if count <= 1:
        return Prepared(_pad(np.full((height, width), 255, np.uint8), pad, 255), False)

    comp_w = stats[:, cv2.CC_STAT_WIDTH]
    comp_h = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    s = max(scale, 1.0)
    keep = (
        (area >= 3 * s * s)
        & (comp_h >= max(2, round(2 * s)))
        & (comp_h <= 0.95 * height)
        & (comp_w <= 0.6 * width)
    )
    keep[0] = False

    if mode == "outline":
        # Rozmiar obwódki zależy też od samych liter: na ekranie 4K lub przy
        # powiększeniu filmu stałe 2 px trafiały wyłącznie w antyaliasing.
        letter_height = _letter_height(stats, height, width, s)
        radius = max(1, round(2 * s), round(letter_height / 18))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        grown = cv2.dilate(labels.astype(np.float32), kernel).astype(np.int32)
        ring = (labels == 0) & (grown > 0)
        ring_labels = grown[ring]
        sums = np.bincount(ring_labels, weights=value[ring].astype(np.float64), minlength=count)
        counts = np.bincount(ring_labels, minlength=count)
        mean_ring = np.where(counts > 0, sums / np.maximum(counts, 1), 255.0)
        dark_counts = np.bincount(ring_labels, weights=(value[ring] <= outline), minlength=count)
        dark_fraction = dark_counts / np.maximum(counts, 1)
        # Antyaliasing zostawia jasny brzeg wokół wypełnienia. Średnia jasność
        # pierścienia potrafi przez to usunąć całe cienkie „l” lub akcent,
        # mimo wyraźnej czarnej obwódki na blisko połowie jego obwodu.
        keep &= (mean_ring <= outline) | (dark_fraction >= 0.45)

    keep = _drop_blobs(keep, bright, labels, stats, scale)
    kept = int(keep.sum())
    mask = keep[labels]
    out = np.where(mask, 0, 255).astype(np.uint8)
    return Prepared(_pad(out, pad, 255), kept >= 2, kept)


def _drop_blobs(keep: np.ndarray, bright: np.ndarray, labels: np.ndarray, stats: np.ndarray, scale: float) -> np.ndarray:
    """Usuwa jasne plamy (światła, niebo, tarcze) — litery mają cienkie, podobnej grubości kreski.

    Grubość kreski = maksimum transformaty odległości w obrębie kształtu. Litery
    i kropki nad nimi mają ją zbliżoną; plamy są wyraźnie „grubsze” albo wyższe od tekstu.
    """
    candidates = np.flatnonzero(keep)
    if candidates.size < 3:
        return keep
    s = max(scale, 1.0)
    comp_x = stats[:, cv2.CC_STAT_LEFT]
    comp_y = stats[:, cv2.CC_STAT_TOP]
    comp_w = stats[:, cv2.CC_STAT_WIDTH]
    comp_h = stats[:, cv2.CC_STAT_HEIGHT]
    heights = comp_h[candidates]
    tall_enough = heights[heights >= 3 * s]
    ref_h = float(np.median(tall_enough if tall_enough.size else heights))

    dist = cv2.distanceTransform(bright, cv2.DIST_L2, 3)
    thickness = np.zeros(len(stats), np.float32)
    for i in candidates:
        x, y, w, h = comp_x[i], comp_y[i], comp_w[i], comp_h[i]
        inside = labels[y : y + h, x : x + w] == i
        thickness[i] = float(dist[y : y + h, x : x + w][inside].max())
    letters = candidates[comp_h[candidates] >= 0.6 * ref_h]
    ref_t = float(np.median(thickness[letters if letters.size else candidates]))

    result = keep.copy()
    result[candidates] = (thickness[candidates] <= 2.0 * ref_t + 1.0) & (comp_h[candidates] <= 2.5 * ref_h)
    return result
