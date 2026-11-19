"""Części windowsowe na atrapach zgodnych z API prawdziwych paczek
(winrt 4.2, sounddevice 0.5, pygrabber 0.2, pycaw, pynput 1.8) — łapią literówki
i złe założenia, zanim kod trafi na Windowsa."""

import sys
import threading
import time
import types

import numpy as np
import pytest

from lektor import textproc


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


# ------------------------------------------------------------------ WinRT OCR
@pytest.fixture
def fake_winrt(monkeypatch):
    calls = {}

    class Language:
        def __init__(self, tag):
            self.language_tag = tag

    class Rect:
        def __init__(self, height):
            self.x, self.y, self.width, self.height = 0.0, 0.0, 10.0, height

    class Word:
        def __init__(self, text, height):
            self.text, self.bounding_rect = text, Rect(height)

    class OcrLine:
        def __init__(self, text, height):
            self.text = text
            self.words = [Word(w, height) for w in text.split()]

    class OcrResult:
        lines = [OcrLine("Musimy się stąd", 40.0), OcrLine("wynosić.", 41.0), OcrLine("ii", 7.0)]

    class AsyncOp:
        def __await__(self):
            async def done():
                return OcrResult()

            return done().__await__()

    class OcrEngineStatic(type):
        @property
        def available_recognizer_languages(cls):
            return [Language("en-US"), Language("pl-PL")]

        @property
        def max_image_dimension(cls):
            return 2600

    class OcrEngine(metaclass=OcrEngineStatic):
        supported = True

        @classmethod
        def is_language_supported(cls, language):
            calls["lang"] = language.language_tag
            return cls.supported

        @classmethod
        def try_create_from_language(cls, language):
            return cls()

        def recognize_async(self, bitmap):
            calls["bitmap"] = bitmap
            return AsyncOp()

    class BitmapPixelFormat:
        RGBA8 = 30

    class SoftwareBitmap:
        @classmethod
        def create_copy_from_buffer(cls, buffer, fmt, width, height):
            calls.update(size=(width, height), fmt=fmt, buflen=len(buffer))
            return object()

    class DataWriter:
        def __init__(self):
            self.data = b""

        def write_bytes(self, value):
            self.data += bytes(value)

        def detach_buffer(self):
            return self.data

    class ApartmentType:
        MULTI_THREADED = 1

    def init_apartment(kind):
        calls["apartment"] = kind

    modules = {
        "winrt": _module("winrt"),
        "winrt.runtime": _module("winrt.runtime", ApartmentType=ApartmentType, init_apartment=init_apartment),
        "winrt.windows": _module("winrt.windows"),
        "winrt.windows.globalization": _module("winrt.windows.globalization", Language=Language),
        "winrt.windows.graphics": _module("winrt.windows.graphics"),
        "winrt.windows.graphics.imaging": _module(
            "winrt.windows.graphics.imaging", BitmapPixelFormat=BitmapPixelFormat, SoftwareBitmap=SoftwareBitmap
        ),
        "winrt.windows.media": _module("winrt.windows.media"),
        "winrt.windows.media.ocr": _module("winrt.windows.media.ocr", OcrEngine=OcrEngine),
        "winrt.windows.storage": _module("winrt.windows.storage"),
        "winrt.windows.storage.streams": _module("winrt.windows.storage.streams", DataWriter=DataWriter),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls, OcrEngine


def test_windows_ocr_reads_lines_and_resizes(fake_winrt):
    calls, _ = fake_winrt
    from lektor.ocr import WindowsOcr

    engine = WindowsOcr("pl")
    lines = engine.recognize(np.full((3000, 900), 255, np.uint8))
    width, height = calls["size"]
    assert max(width, height) <= 2600
    assert calls["fmt"] == 30 and calls["buflen"] == width * height * 4
    assert calls["apartment"] == 1 and calls["lang"] == "pl"
    assert lines[0] == ("Musimy się stąd", 40.0)
    assert textproc.clean_lines(lines) == "Musimy się stąd wynosić."
    engine.close()


def test_windows_ocr_missing_polish_explains_fix(fake_winrt):
    _, engine_cls = fake_winrt
    engine_cls.supported = False
    from lektor.ocr import OcrError, WindowsOcr

    with pytest.raises(OcrError) as info:
        WindowsOcr("pl")
    message = str(info.value)
    assert "Language.OCR*pl-PL*" in message and "en-US, pl-PL" in message


# -------------------------------------------------------------- sounddevice
@pytest.fixture
def fake_sounddevice(monkeypatch):
    devices = [
        {"name": "Mapowanie dźwięku Microsoft - Output", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        {"name": "Głośniki (Focusrite USB Audio)", "hostapi": 2, "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "Głośniki (Focusrite USB Audio)", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        {"name": "Mikrofon (Focusrite USB Audio)", "hostapi": 0, "max_output_channels": 0, "default_samplerate": 44100.0},
    ]
    apis = ({"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"})
    class Played(list):
        state: dict

    played = Played()

    class Stream:
        def __init__(self, seconds):
            self.until = time.monotonic() + seconds

        @property
        def active(self):
            return time.monotonic() < self.until

    state = {}

    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            return devices
        if device is None:
            return devices[0]
        return devices[device]

    def play(data, samplerate=None, device=None):
        played.append((len(data), samplerate, device))
        state["stream"] = Stream(len(data) / samplerate)

    class OutputStream:
        """Jak sounddevice.OutputStream z callbackiem: wątek woła callback blok po bloku."""

        def __init__(self, samplerate=None, channels=None, dtype=None, device=None, callback=None, **kwargs):
            if state.get("stream_error"):
                raise RuntimeError("PortAudio: Unanticipated host error")
            assert dtype == "float32" and callable(callback)
            self.samplerate, self.channels, self.device, self.callback = samplerate, channels, device, callback
            self.active = False
            self.closed = False
            self.frames_out = 0  # ile niezerowych próbek trafiło do karty
            state.setdefault("streams", []).append(self)

        def start(self):
            self.active = True
            threading.Thread(target=self._run, daemon=True).start()

        def _run(self):
            block = max(1, self.samplerate // 100)
            while self.active:
                if state.get("stream_dies"):
                    self.active = False
                    return
                out = np.full((block, self.channels), np.nan, np.float32)
                self.callback(out, block, None, None)
                assert not np.isnan(out).any()  # callback wypełnia cały blok
                self.frames_out += int(np.count_nonzero(out[:, 0]))
                time.sleep(block / self.samplerate)

        def stop(self):
            self.active = False

        def abort(self):
            self.active = False

        def close(self):
            self.active = False
            self.closed = True

    module = _module(
        "sounddevice",
        query_devices=query_devices,
        query_hostapis=lambda index=None: apis,
        play=play,
        get_stream=lambda: state["stream"],
        stop=lambda ignore_errors=True: state.setdefault("stops", []).append(1),
        OutputStream=OutputStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    played.state = state
    return played


def test_audio_prefers_mme_and_plays(fake_sounddevice):
    from lektor.audio import AudioOut, list_output_devices, resolve_output_device

    assert resolve_output_device("focusrite") == 2  # MME, nie WASAPI
    assert resolve_output_device("") is None
    assert resolve_output_device("3") == 3
    assert any("[Windows WASAPI]" in line for line in list_output_devices())
    out = AudioOut("Focusrite", gapless=False)
    assert (out.device, out.samplerate, out.channels) == (2, 44100, 2)
    assert out.play(np.zeros((4410, 2), np.float32)) is True
    assert fake_sounddevice[-1] == (4410, 44100, 2)


def test_audio_gapless_stream_plays_lines_back_to_back(fake_sounddevice):
    from lektor.audio import AudioOut

    out = AudioOut("Focusrite")
    stream = fake_sounddevice.state["streams"][0]
    assert (stream.device, stream.samplerate, stream.channels) == (2, 44100, 2)
    started = time.monotonic()
    assert out.play(np.full((4410, 2), 0.5, np.float32)) is True
    assert out.play(np.full(4410, 0.25, np.float32)) is True  # mono → na oba kanały
    assert time.monotonic() - started < 1.0
    deadline = time.monotonic() + 1.0  # wątek atrapy dolicza ostatni blok chwilę po callbacku
    while stream.frames_out < 8820 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stream.frames_out == 8820
    assert len(fake_sounddevice.state["streams"]) == 1 and fake_sounddevice == []  # bez sd.play
    assert out.play(np.zeros((0, 2), np.float32)) is True  # pusta kwestia nie blokuje kolejki
    out.close()
    assert stream.closed


@pytest.mark.parametrize("gapless", [True, False])
def test_audio_stop_interrupts(fake_sounddevice, gapless):
    import threading

    from lektor.audio import AudioOut

    out = AudioOut("", gapless=gapless)
    threading.Timer(0.2, out.stop).start()
    started = time.monotonic()
    assert out.play(np.full((44100 * 3, 2), 0.5, np.float32)) is False
    assert time.monotonic() - started < 1.0
    if gapless:
        time.sleep(0.05)  # blok, który był w drodze, zdąży się doliczyć
        frames = fake_sounddevice.state["streams"][0].frames_out
        time.sleep(0.1)
        assert fake_sounddevice.state["streams"][0].frames_out == frames  # po stop() cisza


def test_audio_falls_back_to_sd_play_when_stream_fails(fake_sounddevice):
    from lektor.audio import AudioOut

    fake_sounddevice.state["stream_error"] = True
    out = AudioOut("Focusrite")
    assert out.play(np.zeros((4410, 2), np.float32)) is True
    assert fake_sounddevice[-1] == (4410, 44100, 2)


def test_audio_reopens_stream_that_stopped(fake_sounddevice):
    from lektor.audio import AudioOut

    out = AudioOut("")
    fake_sounddevice.state["stream_dies"] = True
    assert out.play(np.full((44100, 2), 0.5, np.float32)) is False  # karta zniknęła w trakcie
    fake_sounddevice.state["stream_dies"] = False
    assert out.play(np.full((4410, 2), 0.5, np.float32)) is True
    assert len(fake_sounddevice.state["streams"]) == 2


def test_decode_mp3_resamples_to_device_rate(tmp_path):
    from synth import make_mp3

    from lektor.audio import decode_mp3, trim_silence

    pcm = decode_mp3(make_mp3(tmp_path / "a.mp3", 0.5), 44100, 2, volume=0.5)
    assert pcm.ndim == 2 and pcm.shape[1] == 2
    assert abs(len(pcm) / 44100 - 0.5) < 0.1
    assert np.abs(pcm).max() <= 0.51
    padded = np.concatenate([np.zeros((22050, 2), np.float32), pcm, np.zeros((22050, 2), np.float32)])
    assert len(trim_silence(padded, 44100)) < len(padded) - 30000


# ---------------------------------------------------------------- pygrabber
def test_capture_device_lookup(monkeypatch):
    class FilterGraph:
        def get_input_devices(self):
            return ["Integrated Camera", "Game Capture HD60 X", "OBS Virtual Camera"]

    monkeypatch.setitem(sys.modules, "pygrabber", _module("pygrabber"))
    monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", _module("pygrabber.dshow_graph", FilterGraph=FilterGraph))
    from lektor.sources import SourceError, resolve_capture_device

    assert resolve_capture_device("HD60") == (1, "Game Capture HD60 X")
    assert resolve_capture_device("obs virtual") == (2, "OBS Virtual Camera")
    assert resolve_capture_device(0) == (0, "Integrated Camera")
    assert resolve_capture_device("2") == (2, "OBS Virtual Camera")
    with pytest.raises(SourceError) as info:
        resolve_capture_device("Cam Link")
    assert "1: Game Capture HD60 X" in str(info.value)


# -------------------------------------------------------------------- pycaw
def test_ducker_lowers_and_restores(monkeypatch):
    class Volume:
        def __init__(self):
            self.level = 0.8

        def GetMasterVolume(self):
            return self.level

        def SetMasterVolume(self, level, context):
            self.level = level

    class Process:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

    class Session:
        def __init__(self, name):
            self.Process = Process(name) if name else None
            self.SimpleAudioVolume = Volume()

    sessions = [Session("obs64.exe"), Session("Discord.exe"), Session(None)]

    class AudioUtilities:
        @staticmethod
        def GetAllSessions():
            return sessions

    monkeypatch.setitem(sys.modules, "comtypes", _module("comtypes", COINIT_MULTITHREADED=0, CoInitializeEx=lambda flags=None: None))
    monkeypatch.setitem(sys.modules, "pycaw", _module("pycaw"))
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", _module("pycaw.pycaw", AudioUtilities=AudioUtilities))
    from lektor.ducking import Ducker

    ducker = Ducker("OBS64.exe", level=0.25)
    ducker.duck()
    time.sleep(0.2)
    assert sessions[0].SimpleAudioVolume.level == pytest.approx(0.2)
    assert sessions[1].SimpleAudioVolume.level == pytest.approx(0.8)
    ducker.release()
    time.sleep(0.2)
    assert sessions[0].SimpleAudioVolume.level == pytest.approx(0.8)
    ducker.duck()
    ducker.close()  # zamknięcie też przywraca głośność
    assert sessions[0].SimpleAudioVolume.level == pytest.approx(0.8)


# ------------------------------------------------------------------- pynput
def test_hotkeys_register_with_pynput(monkeypatch):
    registered = {}

    class GlobalHotKeys:
        def __init__(self, mapping):
            registered.update(mapping)
            self.daemon = False

        def start(self):
            registered["_started"] = self.daemon

        def stop(self):
            registered["_stopped"] = True

    keyboard = _module("pynput.keyboard", GlobalHotKeys=GlobalHotKeys)
    monkeypatch.setitem(sys.modules, "pynput", _module("pynput", keyboard=keyboard))
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    from lektor.hotkeys import ConsoleKeys, GlobalHotkeys, pretty

    hits = []
    hk = GlobalHotkeys({"<ctrl>+<alt>+p": lambda: hits.append("p"), "": lambda: None})
    assert hk.active and registered["_started"] is True
    assert set(k for k in registered if not k.startswith("_")) == {"<ctrl>+<alt>+p"}
    registered["<ctrl>+<alt>+p"]()
    hk.stop()
    assert hits == ["p"] and registered["_stopped"]
    assert pretty("<ctrl>+<alt>+w") == "Ctrl+Alt+W"
    keys = ConsoleKeys({"q": lambda: hits.append("q")})
    assert keys.handle("Q") and hits[-1] == "q"
    keys.poll()  # bez msvcrt (Linux) — nic nie robi
