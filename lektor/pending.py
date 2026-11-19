"""Trwała kolejka: wpis znika dopiero po odtworzeniu całej kwestii."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Line:
    text: str
    detected_at: float
    system: bool = False
    key: str = ""
    detected_wall: float = 0.0


class PendingQueue:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._lines = {}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != 1 or not isinstance(data.get("pending"), list):
                raise ValueError(f"Nieprawidłowy plik kolejki: {self.path}")
            now, wall = time.monotonic(), time.time()
            for item in data["pending"]:
                key, text, detected_wall = str(item["id"]), str(item["text"]), float(item["detected_at"])
                self._lines[key] = Line(text, now - max(0, wall - detected_wall), key=key, detected_wall=detected_wall)

    def snapshot(self):
        with self._lock:
            return list(self._lines.values())

    def contains(self, line):
        with self._lock:
            return line.system or line.key in self._lines

    def add(self, text, detected_at):
        line = Line(text, detected_at, key=uuid.uuid4().hex, detected_wall=time.time() - (time.monotonic() - detected_at))
        with self._lock:
            updated = dict(self._lines)
            updated[line.key] = line
            self._save(updated)
            self._lines = updated
        return line

    def complete(self, line):
        if line.system:
            return
        with self._lock:
            if line.key not in self._lines:
                return
            updated = dict(self._lines)
            del updated[line.key]
            self._save(updated)
            self._lines = updated

    def clear(self):
        with self._lock:
            self._save({})
            self._lines = {}

    def _save(self, lines):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "pending": [
            {"id": line.key, "text": line.text, "detected_at": line.detected_wall}
            for line in lines.values()
        ]}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(10):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.05)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
