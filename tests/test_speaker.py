import time
import json

import numpy as np
import pytest
from fakes import FakeAudio, FakeTTS

from lektor import config
from lektor.speaker import Speaker


@pytest.fixture
def mp3(monkeypatch):
    # Testujemy rzeczywiste wątki i kolejność kolejki bez sieci, ffmpeg i głośników.
    monkeypatch.setattr("lektor.speaker.decode_mp3", lambda data, rate, channels, volume: np.ones((1000, channels), np.float32))
    return b"audio testowe"


def _wait_idle(speaker, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if speaker.idle():
            return
        time.sleep(0.05)
    raise AssertionError("lektor nie skończył na czas")


def test_reads_lines_in_order_and_speeds_up_on_backlog(mp3):
    cfg = config.load(None).tts
    tts, audio = FakeTTS(mp3), FakeAudio(0.4)
    speaker = Speaker(tts, cfg, lambda: audio)
    speaker.start()
    now = time.monotonic()
    for i in range(4):
        speaker.say(f"Kwestia numer {i}", now)
    _wait_idle(speaker)
    speaker.stop()
    assert [text for text, _ in speaker.history] == [f"Kwestia numer {i}" for i in range(4)]
    rates = [rate for _, rate in tts.calls]
    assert all(cfg.rate <= rate <= cfg.rate_max for rate in rates)
    assert max(rates) > cfg.rate  # przy zaległościach tempo rośnie
    assert max(rates) <= cfg.rate_max
    assert len(audio.played) == 4


def test_stale_lines_are_all_read_even_when_newer_wait(mp3):
    cfg = config.load(None).tts
    cfg.max_delay_s = 1  # nawet stary config nie może przywrócić pomijania
    tts, audio = FakeTTS(mp3), FakeAudio(0.2)
    speaker = Speaker(tts, cfg, lambda: audio)
    speaker.start()
    now = time.monotonic()
    speaker.say("Bardzo stara kwestia", now - 30)
    speaker.say("Świeża kwestia", now)
    _wait_idle(speaker)
    speaker.stop()
    assert [text for text, _ in speaker.history] == ["Bardzo stara kwestia", "Świeża kwestia"]


def test_pause_preserves_backlog_and_restarts_interrupted_line(mp3):
    cfg = config.load(None).tts
    tts, audio = FakeTTS(mp3, delay=0.2), FakeAudio(1.0)
    speaker = Speaker(tts, cfg, lambda: audio)
    speaker.start()
    now = time.monotonic()
    for i in range(3):
        speaker.say(f"Linia {i}", now)
    time.sleep(0.5)
    speaker.set_paused(True)
    speaker.say("Podczas pauzy", time.monotonic())
    time.sleep(0.2)
    assert speaker.backlog() == 3
    assert speaker.history == []
    speaker.set_paused(False)
    speaker.say("Po pauzie", time.monotonic())
    _wait_idle(speaker)
    speaker.stop()
    spoken = [text for text, _ in speaker.history]
    assert "Podczas pauzy" not in spoken
    assert spoken[-1] == "Po pauzie"
    assert spoken == ["Linia 0", "Linia 1", "Linia 2", "Po pauzie"]


def test_skip_and_repeat(mp3):
    cfg = config.load(None).tts
    tts, audio = FakeTTS(mp3), FakeAudio(2.0)
    speaker = Speaker(tts, cfg, lambda: audio)
    speaker.start()
    speaker.say("Długa kwestia", time.monotonic())
    time.sleep(0.6)
    speaker.skip()
    _wait_idle(speaker)
    assert len(audio.played) == 1
    audio.play_seconds = 0.2
    speaker.repeat()
    time.sleep(0.6)
    _wait_idle(speaker)
    speaker.stop()
    assert len(audio.played) == 2  # powtórzenie ostatniej


def _wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Nie osiągnięto oczekiwanego stanu")


def test_stop_keeps_current_and_waiting_lines_for_restart(mp3, tmp_path):
    path = tmp_path / "kolejka.json"
    cfg = config.load(None).tts
    first = Speaker(FakeTTS(mp3, delay=.01), cfg, lambda: FakeAudio(2), queue_path=path)
    first.start()
    expected = ["Pierwsza: zażółć gęślą jaźń.", "Druga kwestia.", "Trzecia kwestia."]
    for text in expected:
        first.say(text)
    _wait(first._playing.is_set)
    first.stop()
    assert [item["text"] for item in json.loads(path.read_text(encoding="utf-8"))["pending"]] == expected

    restored = Speaker(FakeTTS(mp3, delay=.01), cfg, lambda: FakeAudio(.02), queue_path=path)
    restored.set_paused(True)
    restored.start()
    time.sleep(.15)
    assert restored.history == []
    assert restored.backlog() == 3
    restored.set_paused(False)
    _wait_idle(restored)
    restored.stop()
    assert [text for text, _ in restored.history] == expected
    assert json.loads(path.read_text(encoding="utf-8"))["pending"] == []


def test_tts_error_retries_same_line_before_following_lines(mp3, tmp_path):
    from lektor.tts import TTSError

    class FailsOnce(FakeTTS):
        async def synthesize(self, text, rate=0):
            if not hasattr(self, "failed"):
                self.failed = True
                self.calls.append((text, rate))
                raise TTSError("chwilowy brak internetu")
            return await super().synthesize(text, rate)

    cfg = config.load(None).tts
    tts = FailsOnce(mp3, delay=.01)
    speaker = Speaker(tts, cfg, lambda: FakeAudio(.02), queue_path=tmp_path / "kolejka.json")
    speaker.start()
    expected = ["Najpierw ten tekst.", "Dopiero potem drugi."]
    for text in expected:
        speaker.say(text)
    _wait_idle(speaker)
    speaker.stop()
    assert [text for text, _ in speaker.history] == expected
    assert [text for text, _ in tts.calls] == [expected[0], *expected]


def test_audio_error_does_not_discard_line(mp3):
    class FailsOnce(FakeAudio):
        def play(self, pcm, cancelled=None):
            if not hasattr(self, "failed"):
                self.failed = True
                raise RuntimeError("chwilowy błąd wyjścia audio")
            return super().play(pcm, cancelled)

    audio = FailsOnce(.02)
    speaker = Speaker(FakeTTS(mp3, delay=.01), config.load(None).tts, lambda: audio)
    speaker.start()
    speaker.say("Nie zgub tej kwestii.")
    speaker.say("Oraz następnej.")
    _wait_idle(speaker)
    speaker.stop()
    assert [text for text, _ in speaker.history] == ["Nie zgub tej kwestii.", "Oraz następnej."]


def test_completed_lines_are_not_restored(mp3, tmp_path):
    path = tmp_path / "kolejka.json"
    speaker = Speaker(FakeTTS(mp3, delay=.01), config.load(None).tts, lambda: FakeAudio(.01), queue_path=path)
    speaker.start()
    speaker.say("To już zostało przeczytane.")
    _wait_idle(speaker)
    speaker.stop()
    restored = Speaker(FakeTTS(mp3), config.load(None).tts, lambda: FakeAudio(), queue_path=path)
    assert restored.backlog() == 0


def test_explicit_flush_cancels_inflight_synthesis_and_clears_saved_queue(mp3, tmp_path):
    path = tmp_path / "kolejka.json"
    speaker = Speaker(FakeTTS(mp3, delay=.2), config.load(None).tts, lambda: FakeAudio(.02), queue_path=path)
    speaker.start()
    speaker.say("To ma zostać usunięte na polecenie.")
    _wait(speaker._synthesizing.is_set)
    speaker.flush()
    _wait_idle(speaker)
    speaker.stop()
    assert speaker.history == []
    assert json.loads(path.read_text(encoding="utf-8"))["pending"] == []
