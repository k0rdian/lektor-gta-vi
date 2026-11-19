"""Ściszanie dźwięku wybranej aplikacji, gdy mówi lektor (jak w klasycznym lektorze).

Działa tylko dla programów grających przez PC (np. OBS z podsłuchem karty,
PS Remote Play, gra na PC). Wymaga pycaw. Pracuje we własnym wątku, bo
pycaw używa COM.
"""

from __future__ import annotations

import queue
import threading

from . import log


class Ducker:
    def __init__(self, process_name: str, level: float = 0.35) -> None:
        self._process = process_name.lower().strip()
        self._level = min(1.0, max(0.0, float(level)))
        self._commands: queue.Queue[str | None] = queue.Queue()
        self._ducked = False
        self._warned = False
        self._thread = threading.Thread(target=self._run, name="ducking", daemon=True)
        self._thread.start()

    def duck(self) -> None:
        if not self._ducked:
            self._ducked = True
            self._commands.put("duck")

    def release(self) -> None:
        if self._ducked:
            self._ducked = False
            self._commands.put("release")

    def close(self) -> None:
        self.release()
        self._commands.put(None)
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import comtypes

            try:
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            except OSError:
                pass
            from pycaw.pycaw import AudioUtilities
        except Exception as exc:  # noqa: BLE001
            log.warn(f"Ściszanie gry niedostępne ({exc}).")
            return
        saved: list[tuple[object, float]] = []
        while True:
            command = self._commands.get()
            if command is None:
                break
            try:
                if command == "duck":
                    saved.clear()
                    for session in AudioUtilities.GetAllSessions():
                        process = session.Process
                        if process is None or process.name().lower() != self._process:
                            continue
                        volume = session.SimpleAudioVolume
                        original = float(volume.GetMasterVolume())
                        saved.append((volume, original))
                        volume.SetMasterVolume(original * self._level, None)
                    if not saved and not self._warned:
                        self._warned = True
                        log.warn(f"Ściszanie: nie widzę dźwięku z „{self._process}” (sprawdź ducking.process).")
                elif command == "release":
                    for volume, original in saved:
                        volume.SetMasterVolume(original, None)
                    saved.clear()
            except Exception as exc:  # noqa: BLE001
                log.debug(f"Ściszanie: {exc}")
        for volume, original in saved:
            try:
                volume.SetMasterVolume(original, None)
            except Exception:  # noqa: BLE001
                pass
