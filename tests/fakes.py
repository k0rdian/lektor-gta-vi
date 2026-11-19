"""Atrapy TTS i wyjścia audio do testów (bez internetu i karty dźwiękowej)."""

from __future__ import annotations

import asyncio
import threading
import time


class FakeTTS:
    name = "fake-tts"

    def __init__(self, mp3: bytes, delay: float = 0.05) -> None:
        self.mp3 = mp3
        self.delay = delay
        self.calls: list[tuple[str, int]] = []

    async def synthesize(self, text: str, rate: int = 0) -> bytes:
        self.calls.append((text, rate))
        await asyncio.sleep(self.delay)
        return self.mp3


class FakeAudio:
    name = "fake-audio"
    samplerate = 24000
    channels = 2

    def __init__(self, play_seconds: float = 0.4) -> None:
        self.play_seconds = play_seconds
        self.played: list[int] = []
        self._stop = threading.Event()

    def play(self, pcm, cancelled=None) -> bool:
        self._stop.clear()
        self.played.append(len(pcm))
        end = time.monotonic() + self.play_seconds
        while time.monotonic() < end:
            if self._stop.is_set() or (cancelled is not None and cancelled()):
                return False
            time.sleep(0.01)
        return True

    def stop(self) -> None:
        self._stop.set()
