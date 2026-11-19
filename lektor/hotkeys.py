"""Skróty klawiszowe: globalne (pynput) oraz pojedyncze litery w oknie konsoli."""

from __future__ import annotations

from collections.abc import Callable

from . import log


def pretty(combo: str) -> str:
    """'<ctrl>+<alt>+p' → 'Ctrl+Alt+P'."""
    parts = []
    for part in combo.split("+"):
        part = part.strip()
        if part.startswith("<") and part.endswith(">"):
            parts.append(part[1:-1].capitalize())
        else:
            parts.append(part.upper())
    return "+".join(parts)


class GlobalHotkeys:
    """Działają także wtedy, gdy okno lektora nie jest aktywne."""

    def __init__(self, mapping: dict[str, Callable[[], None]]) -> None:
        self._listener = None
        mapping = {combo: action for combo, action in mapping.items() if combo}
        if not mapping:
            return
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys(mapping)
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:  # noqa: BLE001
            self._listener = None
            log.warn(f"Globalne skróty nie działają ({exc}) — używaj liter w oknie konsoli.")

    @property
    def active(self) -> bool:
        return self._listener is not None

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass


class ConsoleKeys:
    """Pojedyncze litery, gdy aktywne jest okno konsoli (tylko Windows)."""

    def __init__(self, mapping: dict[str, Callable[[], None]]) -> None:
        self._mapping = {key.lower(): action for key, action in mapping.items()}
        try:
            import msvcrt

            self._msvcrt = msvcrt
        except ImportError:
            self._msvcrt = None

    def poll(self) -> None:
        if self._msvcrt is None:
            return
        while self._msvcrt.kbhit():
            char = self._msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                self._msvcrt.getwch()
                continue
            action = self._mapping.get(char.lower())
            if action is not None:
                action()

    def handle(self, char: str) -> bool:
        action = self._mapping.get(char.lower())
        if action is None:
            return False
        action()
        return True
