"""Synteza mowy przez Edge TTS (domyślnie głos Marek) z ponawianiem prób."""

from __future__ import annotations

import asyncio

from . import log


class TTSError(RuntimeError):
    pass


def describe(exc: BaseException | None) -> str:
    text = repr(exc)
    if "403" in text:
        return (
            "Edge TTS odrzucił połączenie (403) — Microsoft zmienił zabezpieczenia. "
            "Zaktualizuj bibliotekę: .venv\\Scripts\\pip install -U edge-tts"
        )
    if isinstance(exc, TimeoutError):
        return "Edge TTS nie odpowiedział na czas — sprawdź internet."
    if exc is not None and type(exc).__name__ == "NoAudioReceived":
        return "Edge TTS nie zwrócił dźwięku (serwer zerwał połączenie albo zła nazwa głosu w config.toml)."
    return f"Edge TTS: {text}"


class EdgeTTS:
    name = "Edge TTS"

    def __init__(self, voice: str = "pl-PL-MarekNeural", pitch: int = 0, timeout_s: float = 10.0, retries: int = 2) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise TTSError("Brak pakietu edge-tts (uruchom setup.bat).") from exc
        self._edge = edge_tts
        self.voice = voice
        self.pitch = int(pitch)
        self.timeout = max(2.0, float(timeout_s))
        self.retries = max(0, int(retries))

    async def synthesize(self, text: str, rate: int = 0) -> bytes:
        last: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.wait_for(self._once(text, int(rate)), timeout=self.timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — edge_tts/aiohttp/timeout
                last = exc
                log.debug(f"Edge TTS, próba {attempt + 1}: {exc!r}")
                if attempt < self.retries:
                    await asyncio.sleep(0.3 * (attempt + 1))
        raise TTSError(describe(last))

    async def _once(self, text: str, rate: int) -> bytes:
        seconds = max(2, int(self.timeout))
        communicate = self._edge.Communicate(
            text,
            self.voice,
            rate=f"{rate:+d}%",
            pitch=f"{self.pitch:+d}Hz",
            connect_timeout=seconds,
            receive_timeout=seconds,
        )
        audio = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio.extend(chunk["data"])
        if not audio:
            raise self._edge.exceptions.NoAudioReceived("brak dźwięku")
        return bytes(audio)


async def list_voices(locale_prefix: str = "pl-") -> list[dict]:
    import edge_tts

    voices = await edge_tts.list_voices()
    return [voice for voice in voices if str(voice.get("Locale", "")).startswith(locale_prefix)]
