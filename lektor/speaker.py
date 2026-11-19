"""Kolejka FIFO bez pomijania zaległych kwestii, z zapisem przed syntezą.

Tempo zależy od spóźnienia: gdy lektor nie nadąża za napisami, czyta szybciej.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import log
from .audio import compress_pauses, decode_mp3, trim_silence
from .pending import Line, PendingQueue
from .textproc import fluent_text

# Ile znaków na sekundę czyta głos przy tempie 0 — startowo, potem uczy się z nagrań.
DEFAULT_PACE_CPS = 14.0
RATE_DROP = 10  # zwalnianie najwyżej o 10 punktów procentowych na kwestię
PACE_MEMORY_S = 6.0  # po takiej przerwie w czytaniu tempo wraca od razu do bazowego


@dataclass
class Clip:
    line: Line
    pcm: np.ndarray
    rate: int
    synth_ms: float
    generation: int = 0


class Speaker:
    def __init__(self, tts, tts_cfg, audio_factory: Callable[[], object], ducker=None, greeting: str = "", *, queue_path: Path | None = None, debug_dir: Path | None = None) -> None:
        self._tts, self._cfg, self._audio_factory = tts, tts_cfg, audio_factory
        self._debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._greeting, self.ducker = greeting, ducker
        self.pending = PendingQueue(queue_path)
        self._text_q: queue.Queue[Line] = queue.Queue()
        for line in self.pending.snapshot():
            self._text_q.put(line)
        self._clip_q: queue.Queue[Clip] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._audio_ready = threading.Event()
        self._playing = threading.Event()
        self._synthesizing = threading.Event()
        self._repeat = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._generation = 0
        self._play_generation = 0
        self._state_lock = threading.RLock()
        self._current: Clip | None = None
        self._current_started: float | None = None
        self._resumed_at = time.monotonic()
        self._pace_cps = DEFAULT_PACE_CPS
        self._last_rate: int | None = None
        self._last_rate_at = 0.0
        self._skip_keys = set()
        self._audio = None
        self._audio_error = None
        self.paused = False
        self.last_clip: Clip | None = None
        self.spoken = 0
        self.history: list[tuple[str, int]] = []

    def start(self) -> None:
        self._player = threading.Thread(target=self._player_loop, name="player", daemon=True)
        self._player.start()
        if not self._audio_ready.wait(15):
            raise RuntimeError("Wyjście audio nie wystartowało.")
        if self._audio_error is not None:
            raise self._audio_error
        self._resumed_at = time.monotonic()
        if self._greeting and not self.paused:
            self._text_q.put(Line(self._greeting, time.monotonic(), system=True))
        restored = self.backlog()
        if restored:
            log.info(f"Przywrócono zapisane kwestie: {restored}.")
        self._synth = threading.Thread(target=self._synth_loop, name="tts", daemon=True)
        self._synth.start()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()
        self._interrupt_audio()
        for thread in (getattr(self, "_synth", None), getattr(self, "_player", None)):
            if thread is not None:
                thread.join(timeout=3)

    def say(self, text: str, detected_at: float | None = None) -> None:
        if text and not self.paused:
            # Zapis na dysku kończy się przed dodaniem do kolejki wątku TTS.
            line = self.pending.add(text, detected_at if detected_at is not None else time.monotonic())
            self._text_q.put(line)

    def _interrupt_audio(self):
        self._play_generation += 1
        if self._audio is not None:
            self._audio.stop()

    def skip(self) -> None:
        with self._state_lock:
            if self._current is not None:
                line = self._current.line
                self.pending.complete(line)
                self._skip_keys.add(line.key)
            self._interrupt_audio()

    def repeat(self) -> None:
        self._repeat.set()

    def set_paused(self, paused: bool) -> None:
        with self._state_lock:
            self.paused = paused
            if paused:
                self._resume.clear()
                self._interrupt_audio()
            else:
                self._resumed_at = time.monotonic()  # czas pauzy nie liczy się do spóźnienia
                self._resume.set()

    def flush(self) -> None:
        """Usuń kolejkę wyłącznie na wyraźne polecenie, nigdy przy pauzie."""
        with self._state_lock:
            self.pending.clear()
            self._generation += 1
            self._repeat.clear()
            for items in (self._text_q, self._clip_q):
                while True:
                    try:
                        items.get_nowait()
                    except queue.Empty:
                        break
            self._interrupt_audio()

    def backlog(self) -> int:
        return len(self.pending.snapshot())

    def idle(self) -> bool:
        return self.backlog() == 0 and not self._playing.is_set() and not self._synthesizing.is_set() and self._clip_q.empty() and self._text_q.empty()

    @property
    def samplerate(self) -> int:
        return self._audio.samplerate if self._audio is not None else 0

    def _wait_running(self):
        while not self._stop.is_set():
            if self._resume.wait(0.1):
                return not self._stop.is_set()
        return False

    def _choose_rate(self, line: Line) -> int:
        """Tempo bazowe, a gdy lektor się spóźnia — o rate_step % szybciej za każdą sekundę.

        Spóźnienie liczymy dla najnowszej czekającej kwestii: ile minie od pojawienia się
        jej napisu, zanim lektor do niej dojdzie (to, co jeszcze gra, gotowe nagrania
        i kwestie przed nią). Zwalnia stopniowo, żeby tempo nie skakało.
        """
        cfg = self._cfg
        base = int(cfg.rate)
        top = max(base, int(cfg.rate_max))
        if line.system:
            return base
        now = time.monotonic()
        waiting = self.pending.snapshot()
        index = next((i for i, item in enumerate(waiting) if item.key == line.key), None)
        if index is None:
            return base
        newest = waiting[-1]
        lag = now - max(newest.detected_at, self._resumed_at) + self._ahead_seconds(now)
        lag += sum(self._estimate_seconds(item.text, base) for item in waiting[index:-1])
        rate = base + float(cfg.rate_step) * max(0.0, lag - float(getattr(cfg, "catch_up_s", 1.5)))
        if self._last_rate is not None and now - self._last_rate_at < PACE_MEMORY_S:
            rate = max(rate, self._last_rate - RATE_DROP)
        rate = int(round(min(top, max(base, rate))))
        self._last_rate, self._last_rate_at = rate, now
        return rate

    def _ahead_seconds(self, now: float) -> float:
        """Ile sekund nagrań zostało do odtworzenia przed kwestią, którą teraz syntezujemy."""
        samplerate = self.samplerate or 1
        with self._state_lock:
            current, started = self._current, self._current_started
        remaining = 0.0
        if current is not None:
            length = len(current.pcm) / samplerate
            remaining = length if started is None else max(0.0, length - (now - started))
        with self._clip_q.mutex:
            queued = list(self._clip_q.queue)
        return remaining + sum(len(clip.pcm) / samplerate for clip in queued)

    def _estimate_seconds(self, text: str, rate: int) -> float:
        return len(text) / (self._pace_cps * (1 + max(-50, rate) / 100))

    def _learn_pace(self, text: str, seconds: float, rate: int) -> None:
        """Mierzy, ile znaków na sekundę czyta głos (przeliczone na tempo 0)."""
        if len(text) < 15 or seconds < 0.5:
            return
        measured = len(text) / (seconds * (1 + max(-50, rate) / 100))
        self._pace_cps = 0.7 * self._pace_cps + 0.3 * min(30.0, max(6.0, measured))

    def _synth_loop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            while self._wait_running():
                try:
                    line = self._text_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._synthesizing.set()
                try:
                    while self.pending.contains(line) and self._wait_running():
                        if self._synthesize_one(loop, line):
                            break
                        # Przy błędzie ta sama kwestia zostaje pierwsza; nic nie znika.
                        if self._stop.wait(1.0):
                            break
                finally:
                    self._synthesizing.clear()
        finally:
            loop.close()

    def _synthesize_one(self, loop, line: Line) -> bool:
        generation = self._generation
        text = fluent_text(line.text, bool(getattr(self._cfg, "skip_commas", True)))
        if not text:  # same znaki interpunkcyjne — nie ma czego czytać
            try:
                self.pending.complete(line)
            except OSError as exc:
                log.error(f"Nie mogę zapisać stanu kolejki: {exc}")
                return False
            return True
        rate = self._choose_rate(line)
        started = time.perf_counter()
        try:
            mp3 = loop.run_until_complete(self._tts.synthesize(text, rate))
            samplerate = self._audio.samplerate
            pcm = trim_silence(decode_mp3(mp3, samplerate, self._audio.channels, self._cfg.volume), samplerate)
            pcm = compress_pauses(pcm, samplerate, getattr(self._cfg, "max_pause_ms", 0))
        except Exception as exc:
            log.error(f"Synteza: {exc}. Kwestia pozostaje w kolejce; ponowię próbę.")
            return False
        elapsed = (time.perf_counter() - started) * 1000
        self._learn_pace(text, len(pcm) / samplerate, rate)
        log.debug(f"synteza {rate:+d}% · {elapsed:.0f} ms · {len(pcm) / samplerate:.1f} s · do syntezatora: {text}")
        self._save_debug(text, mp3)
        if line.system:
            log.info(f"Edge TTS odpowiada ({elapsed:.0f} ms).")
        clip = Clip(line, pcm, rate, elapsed, generation)
        while not self._stop.is_set() and generation == self._generation and self.pending.contains(line):
            try:
                self._clip_q.put(clip, timeout=0.1)
                return True
            except queue.Full:
                pass
        return True

    def _save_debug(self, text: str, mp3: bytes) -> None:
        """Przy save_samples: dokładnie to, co wysłaliśmy do Edge TTS, i to, co odesłał."""
        if self._debug_dir is None:
            return
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            (self._debug_dir / f"{stamp}-4-syntezator.txt").write_text(text, encoding="utf-8")
            (self._debug_dir / f"{stamp}-5-edge.mp3").write_bytes(mp3)
        except OSError as exc:
            log.debug(f"Nie zapisałem próbki dźwięku: {exc}")

    def _player_loop(self) -> None:
        try:
            self._audio = self._audio_factory()
        except BaseException as exc:
            self._audio_error = exc
            self._audio_ready.set()
            return
        log.info(f"Dźwięk: {self._audio.name} ({self._audio.samplerate} Hz)")
        self._audio_ready.set()
        while self._wait_running():
            repeated = False
            if self._repeat.is_set():
                self._repeat.clear()
                clip = self.last_clip
                repeated = True
                if clip is None:
                    continue
            else:
                try:
                    clip = self._clip_q.get(timeout=0.1)
                except queue.Empty:
                    if self.ducker is not None:
                        self.ducker.release()
                    continue
            while self._wait_running():
                if clip.generation != self._generation or (not repeated and not self.pending.contains(clip.line)):
                    break
                try:
                    completed = self._play(clip, repeated)
                finally:
                    self._playing.clear()
                if completed:
                    break
                if self._stop.wait(0.1):
                    break
        if self.ducker is not None:
            self.ducker.release()
        close = getattr(self._audio, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 — zamykanie nie może wywrócić wyjścia
                log.debug(f"Zamykanie dźwięku: {exc!r}")

    def _play(self, clip: Clip, repeated: bool = False) -> bool:
        with self._state_lock:
            self._current = clip
            token = self._play_generation
        cancelled = lambda: self._stop.is_set() or self.paused or token != self._play_generation or clip.generation != self._generation
        if cancelled():
            with self._state_lock:
                self._current = None
            return False
        if self.ducker is not None:
            self.ducker.duck()
        delay = time.monotonic() - clip.line.detected_at
        log.info(
            f"   » {'powtarzam' if repeated else 'czyta'}: {log.short(clip.line.text)} "
            f"(opóźnienie {delay:.1f} s · tempo {clip.rate:+d}% · synteza {clip.synth_ms / 1000:.1f} s)"
        )
        self._playing.set()
        completed = False
        with self._state_lock:
            self._current_started = time.monotonic()
        try:
            completed = self._audio.play(clip.pcm, cancelled=cancelled)
        except Exception as exc:
            log.error(f"Odtwarzanie: {exc}. Kwestia pozostaje w kolejce.")
        finally:
            with self._state_lock:
                self._current = None
                self._current_started = None
        skipped = clip.line.key in self._skip_keys
        if skipped:
            self._skip_keys.discard(clip.line.key)
            self.last_clip = clip
            return True
        if not completed or cancelled():
            if self.ducker is not None:
                self.ducker.release()
            return False
        if not repeated:
            # Potwierdzenie dopiero po pełnym odtworzeniu. Awaria dysku nie
            # pozwala przejść dalej i zgubić wpisu z zapisu.
            while not self._stop.is_set():
                try:
                    self.pending.complete(clip.line)
                    break
                except OSError as exc:
                    log.error(f"Nie mogę zapisać stanu kolejki: {exc}")
                    self._stop.wait(1)
        if not clip.line.system:
            self.last_clip = clip
            if not repeated:
                self.spoken += 1
                self.history.append((clip.line.text, clip.rate))
        return True
