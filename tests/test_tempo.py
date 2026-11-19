"""Płynność i tempo lektora: bez pauz na przecinkach, krótkie pauzy, przyspieszanie przy spóźnieniu."""

import time

import numpy as np
import pytest
from fakes import FakeAudio, FakeTTS

from lektor import config
from lektor.audio import compress_pauses
from lektor.speaker import Clip, Speaker
from lektor.textproc import Stabilizer, fluent_text, read_quality, strip_spoken_symbols


# ------------------------------------------------------------- tekst bez pauz
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mówiłem ci, że się szkoliłem na Marudera?", "Mówiłem ci że się szkoliłem na Marudera?"),
        ("No, no. Wspominałeś. Przepraszam, ale spieszy mi się.", "No no. Wspominałeś. Przepraszam ale spieszy mi się."),
        ("Ja... nie wiem.", "Ja nie wiem."),
        ("Ja… Nie wiem!", "Ja. Nie wiem!"),
        ("Nie wiem...", "Nie wiem."),
        ("Co?...", "Co?"),
        ("- Co robisz? - Nic.", "Co robisz? Nic."),
        ("-Co robisz? -Nic.", "Co robisz? Nic."),
        ("Jason – mój brat – przyjechał.", "Jason mój brat przyjechał."),
        ("Słuchaj: jutro idziemy; jasne?", "Słuchaj jutro idziemy jasne?"),
        ("Powiedział „Wajs Siti” i wyszedł.", "Powiedział Wajs Siti i wyszedł."),
        ("Ok,ale szybko", "Ok ale szybko"),
        # liczby, godziny i wyrazy z łącznikiem zostają
        ("Kosztuje 1,5 miliona, serio.", "Kosztuje 1,5 miliona serio."),
        ("Spotkamy się o 12:30, dobra?", "Spotkamy się o 12:30 dobra?"),
        ("czarno-biały samochód", "czarno-biały samochód"),
        ("Temperatura -5 stopni.", "Temperatura -5 stopni."),
        ("...", ""),
    ],
)
def test_fluent_text_removes_pause_marks(text, expected):
    assert fluent_text(text) == expected


def test_fluent_text_can_keep_commas():
    assert fluent_text("No, no.  Tak.", skip_commas=False) == "No, no. Tak."


# ---------------------------------------- znaki, które lektor czytałby na głos
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Miejcie mnie na oku, dop;ki si; nie zawin;.", "Miejcie mnie na oku, dopki si nie zawin."),
        ("tak ; nie", "tak nie"),
        ("si|ę", "się"),
        ("Rock & Roll", "Rock Roll"),
        ("Jedziemy 120 km/h.", "Jedziemy 120 km h."),
        ("To 50% szans, pół na pół: 1/2.", "To 50% szans, pół na pół: 1/2."),
        ("Mam 500$ i 2+2=4.", "Mam 500$ i 2+2=4."),
        ("~Hej~ *ty* #tam", "Hej ty tam"),
        ("$wiat jest m%ój.", "wiat jest mój."),
        ("Szansa 50 % albo $100.", "Szansa 50 % albo $100."),
    ],
)
def test_symbols_read_aloud_by_tts_are_removed(text, expected):
    assert strip_spoken_symbols(text) == expected


def test_symbols_are_removed_even_when_commas_are_kept():
    assert fluent_text("Ty; tak, ty.", skip_commas=False) == "Ty tak, ty."
    assert fluent_text("Ty; tak, ty.") == "Ty tak ty."


def test_stabilizer_prefers_clean_polish_read_in_either_order():
    clean, dirty = "Miejcie mnie na oku, dopóki się nie zawinę.", "Miejcie mnie na oku, dopóki si; nie zawinę."
    for first, second in ((clean, dirty), (dirty, clean)):
        stabilizer = Stabilizer(required_reads=2, similarity=90)
        assert stabilizer.feed(first, 1.0) is None
        assert stabilizer.feed(second, 1.2) == clean
        assert stabilizer.first_seen == 1.0
    assert read_quality(clean) > read_quality(dirty)


def test_details_go_to_log_file_but_not_to_console(tmp_path, capsys):
    from lektor import log

    log.setup(verbose=False, path=tmp_path / "lektor.log")
    try:
        log.debug("OCR 30 ms: Miejcie mnie na oku")
        log.info("Napis: Miejcie mnie na oku")
    finally:
        log.close()
    console = capsys.readouterr().out
    written = (tmp_path / "lektor.log").read_text(encoding="utf-8")
    assert "OCR 30 ms" not in console and "Napis:" in console
    assert "OCR 30 ms: Miejcie mnie na oku" in written and "Napis: Miejcie mnie na oku" in written


# ------------------------------------------------------------ skracanie pauz
SR = 48000
_rng = np.random.default_rng(7)


def _voice(ms, amp=0.5):
    t = np.arange(int(SR * ms / 1000)) / SR
    return (amp * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def _silence(ms):
    return _rng.normal(0, 0.0005, int(SR * ms / 1000)).astype(np.float32)


def _hiss(ms, amp=0.03):  # cicha spółgłoska (s, f, h)
    return _rng.normal(0, amp / 3, int(SR * ms / 1000)).astype(np.float32)


def test_long_pauses_are_shortened_and_speech_is_kept():
    parts = [_voice(400), _silence(60), _voice(300), _hiss(120), _silence(300), _voice(500),
             _silence(800), _hiss(80, 0.015), _voice(400), _silence(150), _voice(200)]
    pcm = np.concatenate(parts)
    out = compress_pauses(pcm, SR, 180)
    # 300 ms → 180 ms i 800 ms → 180 ms; krótkie przerwy (60, 150 ms) i ciche spółgłoski zostają
    assert abs((len(pcm) - len(out)) / SR - 0.74) < 0.03
    assert np.isclose((out ** 2).sum(), (pcm ** 2).sum(), rtol=1e-3)


def test_pause_compression_handles_stereo_and_can_be_disabled():
    pcm = np.concatenate([_voice(300), _silence(700), _voice(300)])
    stereo = np.stack([pcm, pcm], axis=1)
    assert compress_pauses(stereo, SR, 180).shape == (len(compress_pauses(pcm, SR, 180)), 2)
    assert len(compress_pauses(pcm, SR, 0)) == len(pcm)
    assert len(compress_pauses(np.zeros(0, np.float32), SR, 180)) == 0


def test_pause_limit_never_cuts_consonant_closures():
    pcm = np.concatenate([_voice(200), _silence(70), _voice(200)])
    assert len(compress_pauses(pcm, SR, 10)) == len(pcm)  # najmniej 80 ms zostaje zawsze


# ------------------------------------------------------- tempo przy spóźnieniu
def _speaker(**tts):
    cfg = config.load(None).tts
    tts = {"rate": 20, "rate_step": 10, "rate_max": 60, "catch_up_s": 1.5, **tts}
    for key, value in tts.items():
        setattr(cfg, key, value)
    speaker = Speaker(FakeTTS(b""), cfg, lambda: FakeAudio())
    speaker._audio = FakeAudio()  # samplerate 24000; bez wątków
    return speaker, cfg


def _clip(seconds, rate=20):
    return Clip(line=None, pcm=np.zeros((int(24000 * seconds), 2), np.float32), rate=rate, synth_ms=0)


LONG_LINE = "Kwestia, która ma dokładnie czterdzieści 3."  # 43 znaki ≈ 2,5 s przy +20 %


def test_fresh_line_without_backlog_uses_base_rate():
    speaker, cfg = _speaker()
    line = speaker.pending.add("Krótka kwestia.", time.monotonic())
    assert speaker._choose_rate(line) == cfg.rate == 20


def test_rate_grows_with_audio_still_playing_ahead():
    speaker, cfg = _speaker()
    speaker._current, speaker._current_started = _clip(3.0), time.monotonic()
    line = speaker.pending.add("Następna kwestia.", time.monotonic())
    # spóźnienie 3,0 s − 1,5 s progu = 1,5 s → +15 %
    assert speaker._choose_rate(line) == pytest.approx(35, abs=1)


def test_rate_counts_lines_waiting_behind_and_respects_max():
    speaker, cfg = _speaker()
    now = time.monotonic()
    first = speaker.pending.add(LONG_LINE, now)
    speaker.pending.add(LONG_LINE, now)
    speaker.pending.add(LONG_LINE, now)
    # najnowsza czeka na dwie kwestie przed sobą: ok. 5 s → 20 + 10 × 3,5 = 55 %
    assert speaker._choose_rate(first) == pytest.approx(55, abs=2)
    for _ in range(3):
        speaker.pending.add(LONG_LINE, now)
    assert speaker._choose_rate(first) == cfg.rate_max == 60


def test_rate_slows_down_gradually_and_resets_after_quiet_time():
    speaker, cfg = _speaker()
    speaker._last_rate, speaker._last_rate_at = 60, time.monotonic()
    rates = []
    for _ in range(5):
        line = speaker.pending.add("Spokojna kwestia.", time.monotonic())
        rates.append(speaker._choose_rate(line))
        speaker.pending.complete(line)
    assert rates == [50, 40, 30, 20, 20]
    speaker._last_rate, speaker._last_rate_at = 60, time.monotonic() - 30
    line = speaker.pending.add("Po długiej ciszy.", time.monotonic())
    assert speaker._choose_rate(line) == 20


def test_time_in_pause_is_not_counted_as_delay():
    speaker, cfg = _speaker()
    line = speaker.pending.add("Kwestia sprzed pauzy.", time.monotonic() - 120)
    speaker.set_paused(True)
    speaker.set_paused(False)
    assert speaker._choose_rate(line) == cfg.rate


# --------------------------------------------------- tekst dla syntezatora
@pytest.fixture
def mp3(monkeypatch):
    monkeypatch.setattr("lektor.speaker.decode_mp3", lambda data, rate, channels, volume: np.ones((1000, channels), np.float32))
    return b"audio testowe"


def _wait_idle(speaker, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if speaker.idle():
            return
        time.sleep(0.05)
    raise AssertionError("lektor nie skończył na czas")


def test_synthesizer_gets_fluent_text_but_queue_keeps_original(mp3):
    tts = FakeTTS(mp3, delay=0.01)
    speaker = Speaker(tts, config.load(None).tts, lambda: FakeAudio(0.02))
    speaker.start()
    speaker.say("Tak, tak... jasne, Dżejson.")
    speaker.say("...")
    _wait_idle(speaker)
    speaker.stop()
    assert [text for text, _ in tts.calls] == ["Tak tak jasne Dżejson."]
    assert [text for text, _ in speaker.history] == ["Tak, tak... jasne, Dżejson."]
    assert speaker.backlog() == 0
