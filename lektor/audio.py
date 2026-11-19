"""Dźwięk: dekodowanie MP3 z Edge TTS (miniaudio) i odtwarzanie (sounddevice).

Uwaga: sounddevice importujemy dopiero w wątku odtwarzacza. PortAudio
inicjuje COM w wątku, który go ładuje — trzymamy to z dala od wątku OCR.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import log

_HOSTAPI_PREFERENCE = ("MME", "Windows DirectSound", "Windows WASAPI")


class AudioError(RuntimeError):
    pass


def decode_mp3(data: bytes, samplerate: int, channels: int, volume: float = 1.0) -> np.ndarray:
    """MP3 → float32 w zadanej częstotliwości (miniaudio sam przepróbkowuje)."""
    import miniaudio

    decoded = miniaudio.decode(
        data,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=channels,
        sample_rate=samplerate,
    )
    pcm = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    if volume != 1.0:
        pcm = np.clip(pcm * float(volume), -1.0, 1.0)
    return pcm


def trim_silence(pcm: np.ndarray, samplerate: int, threshold: float = 0.01, keep_s: float = 0.03) -> np.ndarray:
    """Obcina ciszę z początku i końca — lektor odzywa się odrobinę szybciej."""
    if pcm.size == 0:
        return pcm
    level = np.abs(pcm) if pcm.ndim == 1 else np.abs(pcm).max(axis=1)
    loud = np.flatnonzero(level > threshold)
    if loud.size == 0:
        return pcm
    keep = int(keep_s * samplerate)
    start = max(0, int(loud[0]) - keep)
    end = min(len(level), int(loud[-1]) + keep * 3)
    return pcm[start:end]


def compress_pauses(pcm: np.ndarray, samplerate: int, max_pause_ms: float = 180, frame_ms: float = 10.0) -> np.ndarray:
    """Skraca ciszę wewnątrz kwestii (np. po kropce) do max_pause_ms; 0 = bez zmian.

    Cisza to ramki 10 ms dużo cichsze od mowy. Krótsze przerwy (zwarcia przy „p”, „t”, „k”,
    ciche spółgłoski) zostają nietknięte, a z każdej długiej pauzy zostaje jej początek i koniec.
    """
    if max_pause_ms is None or max_pause_ms <= 0 or pcm.size == 0 or samplerate <= 0:
        return pcm
    frame = max(1, int(samplerate * frame_ms / 1000))
    keep = max(8, int(round(max(80.0, float(max_pause_ms)) / frame_ms)))  # ramki ciszy, które zostają
    level = np.abs(pcm) if pcm.ndim == 1 else np.abs(pcm).max(axis=1)
    count = len(level) // frame
    if count <= keep:
        return pcm
    peaks = level[: count * frame].reshape(count, frame).max(axis=1)
    loud = float(np.percentile(peaks, 95))
    if loud <= 0:
        return pcm
    threshold = min(0.02, max(0.002, 0.012 * loud))  # ok. 38 dB poniżej głośnej mowy
    edges = np.diff(np.concatenate(([0], (peaks < threshold).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    head = keep // 2
    cuts = [((s + head) * frame, (e - (keep - head)) * frame) for s, e in zip(starts, ends) if e - s > keep]
    if not cuts:
        return pcm
    out = pcm.copy()
    fade = min(int(0.004 * samplerate), head * frame)
    pieces, position = [], 0
    for cut_start, cut_end in cuts:
        if fade > 0:  # łagodne sklejenie, żeby nie było trzasku
            ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            if out.ndim > 1:
                ramp = ramp[:, None]
            out[cut_start - fade : cut_start] *= ramp
            out[cut_end : cut_end + fade] *= ramp[::-1]
        pieces.append(out[position:cut_start])
        position = cut_end
    pieces.append(out[position:])
    return np.concatenate(pieces)


def list_output_devices() -> list[str]:
    import sounddevice as sd

    apis = sd.query_hostapis()
    lines = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_output_channels"] > 0:
            lines.append(f"{index}: {device['name']}  [{apis[device['hostapi']]['name']}]")
    return lines


def resolve_output_device(query) -> int | None:
    """Fragment nazwy → numer urządzenia. Preferuje MME (przyjmie każdą częstotliwość)."""
    import sounddevice as sd

    if query is None or str(query).strip() == "":
        return None
    if isinstance(query, int) or str(query).strip().isdigit():
        return int(query)
    devices = sd.query_devices()
    apis = sd.query_hostapis()
    needle = str(query).lower()
    matches = [
        index
        for index, device in enumerate(devices)
        if device["max_output_channels"] > 0 and needle in device["name"].lower()
    ]
    if not matches:
        log.warn(f"Nie znalazłem wyjścia audio „{query}” — używam domyślnego (lista: --list-devices).")
        return None

    def rank(index: int) -> int:
        name = apis[devices[index]["hostapi"]]["name"]
        return _HOSTAPI_PREFERENCE.index(name) if name in _HOSTAPI_PREFERENCE else len(_HOSTAPI_PREFERENCE)

    return min(matches, key=rank)


class AudioOut:
    """Odtwarza gotowe próbki; play() blokuje do końca albo do stop().

    Domyślnie przez jeden stały strumień: kolejne kwestie grają bez ponownego otwierania
    urządzenia, więc między nimi nie ma dodatkowej ciszy. Gdy strumień nie działa
    (sterownik, odłączone słuchawki), lektor wraca do zwykłego sd.play().
    """

    MAX_STREAM_FAILURES = 3

    def __init__(self, device_query: str | int = "", gapless: bool = True) -> None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise AudioError(f"Nie mogę załadować sounddevice/PortAudio: {exc}") from exc
        self._sd = sd
        self.device = resolve_output_device(device_query)
        try:
            info = sd.query_devices(self.device, "output") if self.device is not None else sd.query_devices(kind="output")
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"Brak wyjścia audio: {exc}") from exc
        self.name = str(info["name"])
        self.samplerate = int(info["default_samplerate"] or 48000)
        self.channels = 2 if int(info["max_output_channels"]) >= 2 else 1
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._buffer: np.ndarray | None = None
        self._position = 0
        self._stream = None
        self._failures = 0
        self._fallback_logged = False
        self.gapless = bool(gapless)
        if self.gapless:
            self._open_stream()

    # ------------------------------------------------------------ stały strumień
    def _open_stream(self) -> bool:
        if self._failures >= self.MAX_STREAM_FAILURES:
            if not self._fallback_logged:
                self._fallback_logged = True
                log.warn("Stały strumień audio zawodzi — do końca sesji odtwarzam po staremu (gapless = false).")
            return False
        try:
            stream = self._sd.OutputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=self._fill,
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001 — PortAudio/sterownik
            self._failures += 1
            log.warn(f"Stały strumień audio nie ruszył ({exc}) — odtwarzam po staremu.")
            return False
        self._stream = stream
        return True

    def _stream_alive(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:  # noqa: BLE001
            return False

    def _close_stream(self, drain: bool = False) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        for action in ("stop" if drain else "abort", "close"):
            try:
                getattr(stream, action)()
            except Exception:  # noqa: BLE001 — zamykamy i tak
                pass

    def _fill(self, outdata, frames, time_info, status) -> None:
        """Callback PortAudio (osobny wątek): następne próbki albo cisza."""
        try:
            with self._lock:
                buffer = self._buffer
                if buffer is None:
                    outdata.fill(0)
                    return
                start = self._position
                chunk = buffer[start : start + frames]
                count = len(chunk)
                outdata[:count] = chunk
                if count < frames:
                    outdata[count:] = 0
                self._position = start + count
                if self._position >= len(buffer):
                    self._buffer = None
                    self._done.set()
        except Exception:  # noqa: BLE001 — wyjątek w callbacku zatrzymałby strumień
            outdata.fill(0)

    def _as_frames(self, pcm: np.ndarray) -> np.ndarray:
        data = np.asarray(pcm, dtype=np.float32).reshape(len(pcm), -1)
        if data.shape[1] == 1 and self.channels > 1:
            data = np.repeat(data, self.channels, axis=1)
        elif data.shape[1] > self.channels:
            data = data[:, : self.channels]
        return np.ascontiguousarray(data)

    # ------------------------------------------------------------- odtwarzanie
    def play(self, pcm: np.ndarray, cancelled=None) -> bool:
        """True = dograne do końca, False = przerwane."""
        self._stop.clear()
        if cancelled is not None and cancelled():
            return False
        if len(pcm) == 0:
            return True
        if self.gapless and not self._stream_alive():
            self._close_stream()
            self._open_stream()
        if self._stream is None:
            return self._play_simple(pcm, cancelled)
        data = self._as_frames(pcm)
        with self._lock:
            self._done.clear()
            self._position = 0
            self._buffer = data
        deadline = time.monotonic() + len(data) / self.samplerate + 2.0
        finished = False
        while not finished:
            if self._done.wait(0.02):
                finished = True
            elif self._stop.is_set() or (cancelled is not None and cancelled()):
                break
            elif time.monotonic() > deadline or not self._stream_alive():
                self._failures += 1
                log.warn("Strumień audio przestał grać — otworzę go od nowa.")
                with self._lock:
                    self._buffer = None
                self._close_stream()
                break
        if not finished:
            with self._lock:
                self._buffer = None
        return finished

    def _play_simple(self, pcm: np.ndarray, cancelled=None) -> bool:
        """Stary sposób: osobne otwarcie urządzenia dla każdej kwestii."""
        sd = self._sd
        sd.play(pcm, self.samplerate, device=self.device)
        stream = sd.get_stream()
        deadline = time.monotonic() + len(pcm) / self.samplerate + 1.5
        finished = True
        while time.monotonic() < deadline:
            if self._stop.is_set() or (cancelled is not None and cancelled()):
                finished = False
                break
            if not stream.active:
                break
            time.sleep(0.02)
        sd.stop()
        return finished

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        """Zamyka strumień; to, co już jest w buforze karty, zdąży wybrzmieć."""
        with self._lock:
            self._buffer = None
        self._close_stream(drain=True)
