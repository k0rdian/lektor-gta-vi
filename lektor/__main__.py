"""Uruchamianie: python -m lektor [opcje]  (na Windowsie: start.bat [opcje])."""

import sys

# COM w trybie MTA dla bibliotek opartych na comtypes (pygrabber, pycaw) — tak samo
# jak WinRT/OCR. Musi być ustawione przed pierwszym importem comtypes.
sys.coinit_flags = 0  # type: ignore[attr-defined]

import argparse  # noqa: E402
import asyncio  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from . import config, log  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_TEXT = "Dzień dobry, tu lektor. Witamy w Vice City — Jason już na ciebie czeka."


def _set_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lektor",
        description="Lektor GTA VI — czyta napisy z ekranu głosem Edge TTS (Marek).",
    )
    parser.add_argument("--config", default=str(BASE_DIR / "config.toml"), help="ścieżka do config.toml")
    parser.add_argument("--select", action="store_true", help="zaznacz obszar napisów od nowa")
    parser.add_argument("--source", help="capture | screen | file:ŚCIEŻKA (nadpisuje config)")
    parser.add_argument("--device", help="karta: fragment nazwy albo numer urządzenia")
    parser.add_argument("--monitor", type=int, help="ekran: numer monitora")
    parser.add_argument("--youtube", action="store_true", help="test YouTube: wybierz monitor i obszar napisów")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-paused", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--voice", help="głos Edge TTS, np. pl-PL-ZofiaNeural")
    parser.add_argument("--preview", action="store_true", help="pokaż okno podglądu wycinka i obrazu dla OCR")
    parser.add_argument("--verbose", action="store_true", help="więcej informacji w konsoli")
    parser.add_argument("--list-devices", action="store_true", help="pokaż karty/kamery, monitory i wyjścia audio")
    parser.add_argument("--voices", action="store_true", help="pokaż polskie głosy Edge TTS")
    parser.add_argument("--test-voice", nargs="?", const=TEST_TEXT, metavar="TEKST", help="sprawdź głos i dźwięk")
    parser.add_argument("--test-ocr", metavar="OBRAZ", help="sprawdź OCR na zrzucie ekranu")
    return parser


def apply_overrides(cfg, args) -> None:
    if args.verbose:
        cfg.debug.verbose = True
    if args.preview:
        cfg.debug.preview = True
    if args.voice:
        cfg.tts.voice = args.voice
    if args.device is not None:
        cfg.source.device = args.device
    if args.monitor is not None:
        cfg.source.monitor = args.monitor
    if args.source:
        if args.source.lower().startswith("file:"):
            cfg.source.type = "file"
            cfg.source.file = args.source[5:]
        else:
            cfg.source.type = args.source.lower()


def cmd_list_devices(cfg) -> int:
    from .sources import list_capture_devices

    log.info("Urządzenia wideo (karta przechwytująca, wirtualna kamera OBS):")
    names = list_capture_devices()
    for index, name in enumerate(names):
        log.info(f"   {index}: {name}")
    if not names:
        log.info("   (brak albo nie udało się odczytać listy)")
    try:
        import mss

        with mss.mss() as screen:
            log.info("Monitory:")
            for index, mon in enumerate(screen.monitors[1:], start=1):
                log.info(f"   {index}: {mon['width']}x{mon['height']} (pozycja {mon['left']},{mon['top']})")
    except Exception as exc:  # noqa: BLE001
        log.warn(f"Monitory: {exc}")
    try:
        from .audio import list_output_devices

        log.info("Wyjścia audio (w config.toml wystarczy fragment nazwy):")
        for line in list_output_devices():
            log.info(f"   {line}")
    except Exception as exc:  # noqa: BLE001
        log.warn(f"Wyjścia audio: {exc}")
    return 0


def cmd_voices() -> int:
    from .tts import list_voices

    voices = asyncio.run(list_voices("pl-"))
    for voice in voices:
        log.info(f"   {voice.get('ShortName')}  ({voice.get('Gender')})")
    log.info("Głos zmienisz w config.toml: [tts] voice = \"...\"")
    return 0


def cmd_test_voice(cfg, text: str) -> int:
    from . import textproc
    from .audio import AudioOut, compress_pauses, decode_mp3, trim_silence
    from .tts import EdgeTTS

    tts = EdgeTTS(cfg.tts.voice, cfg.tts.pitch, cfg.tts.timeout_s, cfg.tts.retries)
    out = AudioOut(cfg.tts.output_device, gapless=cfg.tts.gapless)
    speech = textproc.prepare_for_speech(text, cfg.replacements, cfg.text.strip_speaker_names)
    speech = textproc.fluent_text(speech, cfg.tts.skip_commas)
    log.info(f"Głos: {cfg.tts.voice} · tempo {int(cfg.tts.rate):+d}% · wyjście: {out.name} ({out.samplerate} Hz)")
    log.info(f"Tekst: {speech}")
    started = time.perf_counter()
    mp3 = asyncio.run(tts.synthesize(speech, int(cfg.tts.rate)))
    elapsed = (time.perf_counter() - started) * 1000
    pcm = trim_silence(decode_mp3(mp3, out.samplerate, out.channels, cfg.tts.volume), out.samplerate)
    before = len(pcm) / out.samplerate
    pcm = compress_pauses(pcm, out.samplerate, cfg.tts.max_pause_ms)
    after = len(pcm) / out.samplerate
    log.info(f"Synteza: {elapsed:.0f} ms, nagranie: {after:.1f} s (skrócone pauzy: −{before - after:.1f} s) — odtwarzam…")
    out.play(pcm)
    out.close()
    log.info("Jeśli było słychać lektora, Edge TTS i dźwięk działają.")
    return 0


def cmd_test_ocr(cfg, path: Path) -> int:
    import cv2

    from . import preprocess, textproc
    from .ocr import create as create_ocr
    from .roi import select as select_roi
    from .roi import to_pixels
    from .sources import SourceError, imread

    image = imread(path)
    if image is None:
        raise SourceError(f"Nie mogę odczytać obrazu {path}")
    ocr = create_ocr(cfg.ocr.engine, cfg.ocr.language, cfg.ocr.tesseract_cmd)
    height, width = image.shape[:2]
    roi = select_roi(image)
    if roi is None:
        log.info("Bez zaznaczenia — czytam cały obraz.")
        crop = image
    else:
        x, y, w, h = to_pixels(roi, width, height)
        crop = image[y : y + h, x : x + w]
    scale = preprocess.auto_scale(height, cfg.ocr.upscale, crop, cfg.ocr.brightness)
    for mode in preprocess.MODES:
        prepared = preprocess.prepare(crop, mode, cfg.ocr.brightness, cfg.ocr.outline, scale)
        started = time.perf_counter()
        lines = ocr.recognize(prepared.image) if prepared.has_text else []
        elapsed = (time.perf_counter() - started) * 1000
        text = textproc.clean_lines(lines)
        mark = "  <- obecny tryb" if mode == cfg.ocr.preprocess else ""
        log.info(f"[{mode:7}] {elapsed:4.0f} ms: {text or '(nic)'}{mark}")
        out = BASE_DIR / "debug_samples"
        out.mkdir(exist_ok=True)
        cv2.imencode(".png", prepared.image)[1].tofile(str(out / f"test-ocr-{mode}.png"))
    log.info(f"Obrazy po obróbce zapisałem w {BASE_DIR / 'debug_samples'}.")
    ocr.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log.setup(verbose=args.verbose)
    try:
        cfg = config.load(Path(args.config))
    except ValueError as exc:
        log.error(str(exc))
        return 2
    apply_overrides(cfg, args)
    log.setup(verbose=cfg.debug.verbose)
    _set_dpi_awareness()

    maintenance = args.list_devices or args.voices or args.test_voice is not None or args.test_ocr
    if not maintenance and not args.worker:
        from .controls import run_panel

        return run_panel(cfg, args, list(argv if argv is not None else sys.argv[1:]))

    log.setup(verbose=cfg.debug.verbose, path=BASE_DIR / "lektor.log")
    try:
        log.debug(f"Python {sys.version.split()[0]} · argumenty: {' '.join(argv if argv is not None else sys.argv[1:])}")
        return _run(cfg, args)
    finally:
        log.close()


def _run(cfg, args) -> int:
    from .audio import AudioError
    from .ocr import OcrError
    from .sources import SourceError
    from .tts import TTSError

    try:
        if args.list_devices:
            return cmd_list_devices(cfg)
        if args.voices:
            return cmd_voices()
        if args.test_voice is not None:
            return cmd_test_voice(cfg, args.test_voice)
        if args.test_ocr:
            return cmd_test_ocr(cfg, Path(args.test_ocr))
        from .app import App

        app = App(cfg, BASE_DIR, force_select=args.select)
        if args.start_paused:
            app.paused = True
            cfg.tts.greeting = ""
        if args.worker:
            import threading

            def commands():
                for line in sys.stdin:
                    command = line.strip()
                    if command == "pause":
                        app.request_pause()
                    elif command == "select":
                        app.request_select()
                    elif command == "quit":
                        app.request_quit()
                app.request_quit()

            threading.Thread(target=commands, daemon=True).start()
        return app.run()
    except (OcrError, SourceError, AudioError, TTSError) as exc:
        log.error(str(exc))
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
