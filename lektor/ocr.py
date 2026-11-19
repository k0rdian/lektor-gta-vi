"""Silniki OCR: wbudowany OCR Windowsa (domyślny) i opcjonalnie Tesseract."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import cv2
import numpy as np

Line = tuple[str, float]  # tekst linii, wysokość liter w pikselach

POLISH_OCR_HELP = """Windows nie ma OCR dla języka „{lang}” (dostępne: {available}).
Doinstaluj go w PowerShellu uruchomionym jako administrator:
    Get-WindowsCapability -Online | Where-Object Name -like 'Language.OCR*{tag}*' | Add-WindowsCapability -Online
albo: Ustawienia > Czas i język > Język i region > Dodaj język > polski.
Potem uruchom lektora ponownie."""

_OCR_TAGS = {"pl": "pl-PL", "en": "en-US", "de": "de-DE"}


class OcrError(RuntimeError):
    pass


async def _await(operation):
    return await operation


class WindowsOcr:
    """OCR wbudowany w Windows 10/11 (Windows.Media.Ocr) przez pakiety winrt."""

    name = "Windows OCR"

    def __init__(self, language: str = "pl") -> None:
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
            from winrt.windows.media.ocr import OcrEngine
            from winrt.windows.storage.streams import DataWriter
        except ImportError as exc:
            raise OcrError(
                "Brak pakietów winrt dla OCR Windowsa. Uruchom setup.bat "
                "(albo: .venv\\Scripts\\pip install winocr)."
            ) from exc
        try:  # zwykle już zrobione przy imporcie winrt; jawnie dla pewności
            from winrt.runtime import ApartmentType, init_apartment

            init_apartment(ApartmentType.MULTI_THREADED)
        except Exception:  # noqa: BLE001 — już zainicjowane albo inny tryb: nie szkodzi
            pass
        self._bitmap_cls = SoftwareBitmap
        self._pixel_format = BitmapPixelFormat.RGBA8
        self._writer_cls = DataWriter
        lang = Language(language)
        if not OcrEngine.is_language_supported(lang):
            available = ", ".join(item.language_tag for item in OcrEngine.available_recognizer_languages)
            raise OcrError(
                POLISH_OCR_HELP.format(
                    lang=language, available=available or "żaden", tag=_OCR_TAGS.get(language, language)
                )
            )
        engine = OcrEngine.try_create_from_language(lang)
        if engine is None:
            raise OcrError(f"Nie udało się uruchomić OCR Windowsa dla „{language}”.")
        self._engine = engine
        self._max_dim = int(OcrEngine.max_image_dimension) or 10000
        self._loop = asyncio.new_event_loop()

    def recognize(self, image: np.ndarray) -> list[Line]:
        if image.ndim == 2:
            rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        else:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        height, width = rgba.shape[:2]
        if max(height, width) > self._max_dim:
            factor = self._max_dim / max(height, width)
            rgba = cv2.resize(
                rgba, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_AREA
            )
            height, width = rgba.shape[:2]
        writer = self._writer_cls()
        writer.write_bytes(np.ascontiguousarray(rgba).tobytes())
        bitmap = self._bitmap_cls.create_copy_from_buffer(
            writer.detach_buffer(), self._pixel_format, width, height
        )
        result = self._loop.run_until_complete(_await(self._engine.recognize_async(bitmap)))
        lines: list[Line] = []
        for line in result.lines:
            heights = [word.bounding_rect.height for word in line.words]
            lines.append((line.text, float(max(heights)) if heights else 0.0))
        return lines

    def close(self) -> None:
        try:
            self._loop.close()
        except RuntimeError:
            pass


class TesseractOcr:
    """Alternatywa: Tesseract (wymaga instalacji programu i pakietu pytesseract)."""

    name = "Tesseract"
    LANGS = {"pl": "pol", "en": "eng", "de": "deu"}

    def __init__(self, language: str = "pl", command: str = "") -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrError(
                "Brak pytesseract: .venv\\Scripts\\pip install pytesseract "
                "(i zainstaluj Tesseract z językiem polskim)."
            ) from exc
        if command:
            pytesseract.pytesseract.tesseract_cmd = command
        elif os.name == "nt":
            default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if default.exists():
                pytesseract.pytesseract.tesseract_cmd = str(default)
        self._tess = pytesseract
        self._lang = self.LANGS.get(language, language)
        try:
            available = pytesseract.get_languages(config="")
        except Exception as exc:  # noqa: BLE001 — komunikat dla użytkownika
            raise OcrError(f"Nie mogę uruchomić Tesseracta: {exc}") from exc
        if self._lang not in available:
            raise OcrError(f"Tesseract nie ma języka „{self._lang}”. Dostępne: {', '.join(available)}")

    def recognize(self, image: np.ndarray) -> list[Line]:
        data = self._tess.image_to_data(
            image, lang=self._lang, config="--psm 6", output_type=self._tess.Output.DICT
        )
        groups: dict[tuple[int, int, int], list[tuple[int, str, int]]] = {}
        for i, word in enumerate(data["text"]):
            if not word or not word.strip():
                continue
            try:
                confidence = float(data["conf"][i])
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < 30:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append((data["left"][i], word, data["height"][i]))
        lines: list[Line] = []
        for key in sorted(groups):
            words = sorted(groups[key])
            lines.append((" ".join(w for _, w, _ in words), float(max(h for *_, h in words))))
        return lines

    def close(self) -> None:
        pass


def create(engine: str, language: str, tesseract_cmd: str = ""):
    engine = (engine or "windows").lower()
    if engine == "windows":
        return WindowsOcr(language)
    if engine == "tesseract":
        return TesseractOcr(language, tesseract_cmd)
    raise OcrError(f"Nieznany silnik OCR „{engine}” (dostępne: windows, tesseract).")
