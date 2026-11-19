"""Źródła obrazu: karta przechwytująca (OpenCV), ekran (mss) i plik (testy)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import log

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class SourceError(RuntimeError):
    pass


def imread(path: Path) -> np.ndarray | None:
    """cv2.imread nie lubi polskich znaków w ścieżkach na Windowsie — czytamy przez numpy."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class _LatestFrame:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._stamp = time.monotonic()

    def get(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._stamp


class FrameSource:
    signature = "source"
    finished = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def snapshot(self, timeout: float = 5.0) -> np.ndarray | None:
        raise NotImplementedError

    def frame_size(self) -> tuple[int, int] | None:
        """(szerokość, wysokość) pełnej klatki albo None, gdy jeszcze jej nie ma."""
        raise NotImplementedError

    def read_roi(self, roi_px: tuple[int, int, int, int]) -> np.ndarray | None:
        raise NotImplementedError


class _ThreadedSource(FrameSource):
    """Wspólna część dla źródeł czytanych w osobnym wątku."""

    def __init__(self) -> None:
        self._latest = _LatestFrame()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self, timeout: float = 5.0) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        while True:
            frame, _ = self._latest.get()
            if frame is not None:
                return frame.copy()
            if time.monotonic() >= deadline or self.finished:
                return None
            time.sleep(0.05)

    def frame_size(self) -> tuple[int, int] | None:
        frame, _ = self._latest.get()
        if frame is None:
            return None
        return frame.shape[1], frame.shape[0]

    def read_roi(self, roi_px: tuple[int, int, int, int]) -> np.ndarray | None:
        frame, _ = self._latest.get()
        if frame is None:
            return None
        x, y, w, h = roi_px
        return frame[y : y + h, x : x + w].copy()


def list_capture_devices() -> list[str]:
    """Nazwy urządzeń wideo DirectShow (kolejność = numery w OpenCV)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception as exc:  # noqa: BLE001 — pygrabber jest opcjonalny
        log.debug(f"pygrabber niedostępny: {exc}")
        return []
    try:
        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception as exc:  # noqa: BLE001
        log.debug(f"pygrabber: {exc}")
        return []


def resolve_capture_device(device: str | int, backend: str = "dshow") -> tuple[int, str]:
    if isinstance(device, int) or str(device).strip().isdigit():
        index = int(device)
        names = list_capture_devices() if backend == "dshow" else []
        return index, names[index] if index < len(names) else f"urządzenie {index}"
    names = list_capture_devices()
    if not names:
        raise SourceError(
            "Nie mogę pobrać listy urządzeń wideo. Wpisz w config.toml numer urządzenia, "
            "np. device = 0 (sprawdź: start.bat --list-devices)."
        )
    query = str(device).lower()
    for index, name in enumerate(names):
        if query in name.lower():
            return index, name
    listing = "; ".join(f"{i}: {n}" for i, n in enumerate(names))
    raise SourceError(f"Nie znalazłem urządzenia „{device}”. Dostępne: {listing}")


class CaptureSource(_ThreadedSource):
    """Karta przechwytująca (np. Elgato HD60 X) albo wirtualna kamera OBS."""

    RETRIEVE_EVERY_S = 0.05  # dekodujemy max ~20 klatek/s, reszta jest tylko odbierana

    def __init__(
        self,
        device: str | int = "HD60",
        backend: str = "dshow",
        width: int = 1920,
        height: int = 1080,
        fps: int = 0,
        fourcc: str = "",
    ) -> None:
        super().__init__()
        self.index, self.name = resolve_capture_device(device, backend)
        self.signature = f"capture:{self.name}"
        self._api = cv2.CAP_MSMF if backend == "msmf" else cv2.CAP_DSHOW
        self._want = (int(width or 0), int(height or 0), int(fps or 0), str(fourcc or ""))
        self._opened = threading.Event()
        self._error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        if not self._opened.wait(15.0):
            raise SourceError(f"Urządzenie „{self.name}” nie odpowiada.")
        if self._error:
            raise SourceError(self._error)

    def _open(self):
        cap = cv2.VideoCapture(self.index, self._api)
        if not cap.isOpened():
            cap.release()
            return None
        width, height, fps, fourcc = self._want
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc[:4].ljust(4)))
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        return cap

    def _run(self) -> None:
        cap = self._open()
        if cap is None:
            self._error = (
                f"Nie mogę otworzyć „{self.name}”. Jeśli kartę trzyma OBS albo Elgato Studio, "
                "zamknij je — albo włącz w OBS wirtualną kamerę i ustaw w config.toml "
                'device = "OBS Virtual Camera".'
            )
            self._opened.set()
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Obraz: {self.name} ({width}x{height})")
        self._opened.set()

        failures = 0
        last_retrieve = 0.0
        last_check = time.monotonic()
        black_since: float | None = None
        warned_black = False
        while not self._stop.is_set():
            if not cap.grab():
                failures += 1
                if failures == 100:
                    log.warn("Karta przestała wysyłać obraz — próbuję połączyć się ponownie…")
                if failures >= 100 and failures % 100 == 0:
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open() or cap
                time.sleep(0.01)
                continue
            if failures >= 100:
                log.info("Obraz z karty wrócił.")
            failures = 0
            now = time.monotonic()
            if now - last_retrieve < self.RETRIEVE_EVERY_S:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            last_retrieve = now
            self._latest.set(frame)
            if now - last_check >= 2.0:
                last_check = now
                if float(frame[::16, ::16].mean()) < 4.0:
                    black_since = black_since or now
                    if not warned_black and now - black_since > 6.0:
                        warned_black = True
                        log.warn(
                            "Obraz z karty jest czarny. Jeśli to PS5: Ustawienia > System > HDMI "
                            "> wyłącz „Włącz HDCP”. Sprawdź też, czy konsola jest włączona."
                        )
                else:
                    black_since = None
        cap.release()


class ScreenSource(FrameSource):
    """Wycinek ekranu (Remote Play, gra na PC, podgląd w OBS). Czyta w wątku głównym."""

    def __init__(self, monitor: int = 1) -> None:
        try:
            import mss
        except ImportError as exc:
            raise SourceError("Brak pakietu mss (uruchom setup.bat).") from exc
        self._mss = mss.mss()
        monitors = self._mss.monitors
        if monitor < 1 or monitor >= len(monitors):
            raise SourceError(f"Nie ma monitora {monitor}. Dostępne: 1–{len(monitors) - 1}.")
        self._monitor = dict(monitors[monitor])
        width, height = self._monitor["width"], self._monitor["height"]
        self.signature = f"screen:{monitor}:{width}x{height}"
        log.info(f"Obraz: monitor {monitor} ({width}x{height})")

    def stop(self) -> None:
        try:
            self._mss.close()
        except Exception:  # noqa: BLE001
            pass

    def _grab(self, region: dict) -> np.ndarray:
        shot = self._mss.grab(region)
        return np.asarray(shot)[:, :, :3].copy()

    def snapshot(self, timeout: float = 5.0) -> np.ndarray | None:
        return self._grab(self._monitor)

    def frame_size(self) -> tuple[int, int] | None:
        return self._monitor["width"], self._monitor["height"]

    def read_roi(self, roi_px: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = roi_px
        region = {
            "left": self._monitor["left"] + x,
            "top": self._monitor["top"] + y,
            "width": w,
            "height": h,
        }
        return self._grab(region)


class FileSource(_ThreadedSource):
    """Obrazek albo film z dysku — do testów bez konsoli."""

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise SourceError(f"Nie ma pliku {self.path}")
        self.loop = loop
        self.signature = f"file:{self.path.name}"
        self._image: np.ndarray | None = None
        if self.path.suffix.lower() in IMAGE_EXTENSIONS:
            self._image = imread(self.path)
            if self._image is None:
                raise SourceError(f"Nie mogę odczytać obrazu {self.path}")

    def start(self) -> None:
        if self._image is not None:
            self._latest.set(self._image)
            return
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise SourceError(f"Nie mogę otworzyć filmu {self.path}")
        log.info(f"Obraz: plik {self.path.name}")
        self._thread = threading.Thread(target=self._run, args=(cap,), name="file", daemon=True)
        self._thread.start()

    def _run(self, cap) -> None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / max(1.0, fps)
        next_time = time.monotonic()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self._latest.set(frame)
            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        cap.release()
        self.finished = True


def create_source(cfg) -> FrameSource:
    kind = str(cfg.type).lower()
    if kind == "capture":
        return CaptureSource(cfg.device, cfg.backend, cfg.width, cfg.height, cfg.fps, cfg.fourcc)
    if kind == "screen":
        return ScreenSource(int(cfg.monitor))
    if kind == "file":
        if not cfg.file:
            raise SourceError('Dla type = "file" podaj ścieżkę w source.file.')
        return FileSource(cfg.file)
    raise SourceError(f"Nieznane źródło „{cfg.type}” (dostępne: capture, screen, file).")
