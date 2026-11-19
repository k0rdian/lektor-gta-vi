"""Główna pętla: obraz → wycinek z napisami → OCR → filtr tekstu → lektor."""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np

from . import log, preprocess, textproc
from .hotkeys import ConsoleKeys, GlobalHotkeys, pretty
from .ocr import create as create_ocr
from .roi import DEFAULT_ROI, RoiStore, to_pixels
from .roi import select as select_roi
from .sources import SourceError, create_source
from .speaker import Speaker

PREVIEW = "Lektor - podglad"


class App:
    def __init__(
        self,
        cfg,
        base_dir: Path,
        force_select: bool = False,
        *,
        ocr=None,
        source=None,
        tts=None,
        audio_factory=None,
        interactive: bool = True,
    ) -> None:
        self.cfg = cfg
        self.base_dir = Path(base_dir)
        self.force_select = force_select
        self.interactive = interactive  # False w testach: bez okien i skrótów
        self._ocr = ocr
        self._source = source
        self._tts = tts
        self._audio_factory = audio_factory
        self.speaker: Speaker | None = None
        self.roi = DEFAULT_ROI
        self.quit = False
        self.paused = False
        self.preview = bool(cfg.debug.preview) and interactive
        self._preview_open = False
        self._want_pause = False
        self._want_preview = False
        self._want_select = False
        self._console: ConsoleKeys | None = None
        self._ocr_failures = 0
        self.last_text = ""
        self.last_ocr_ms = 0.0
        self.detected: list[str] = []
        self._ignore = [re.compile(pattern, re.IGNORECASE) for pattern in cfg.text.ignore]

    # ------------------------------------------------------------------ akcje
    # Wołane także z wątku skrótów — ustawiają tylko flagi albo proste wywołania.
    def request_pause(self) -> None:
        self._want_pause = True

    def request_repeat(self) -> None:
        if self.speaker is not None:
            self.speaker.repeat()

    def request_skip(self) -> None:
        if self.speaker is not None:
            self.speaker.skip()

    def request_select(self) -> None:
        self._want_select = True

    def request_preview(self) -> None:
        self._want_preview = True

    def request_quit(self) -> None:
        self.quit = True

    # ------------------------------------------------------------------- start
    def run(self) -> int:
        cfg = self.cfg
        # OCR najpierw: WinRT ustawia COM w tym wątku jako MTA.
        ocr = self._ocr or create_ocr(cfg.ocr.engine, cfg.ocr.language, cfg.ocr.tesseract_cmd)
        log.info(f"OCR: {ocr.name} (język: {cfg.ocr.language})")
        source = self._source or create_source(cfg.source)
        hotkeys = None
        ducker = None
        try:
            source.start()
            frame = source.snapshot(timeout=10.0)
            if frame is None:
                raise SourceError("Nie dostaję obrazu ze źródła (sprawdź kabel i czy konsola jest włączona).")
            store = RoiStore(self.base_dir / "roi.json")
            saved = store.get(source.signature)
            if saved is not None:
                self.roi = saved
            if saved is None or self.force_select:
                picked = select_roi(frame, saved) if self.interactive else None
                if picked is not None:
                    self.roi = picked
                    store.set(source.signature, picked)
                    log.info("Zapisano obszar napisów.")
                elif self.interactive:
                    log.info("Anulowano wybór obszaru — lektor pozostaje wyłączony.")
                    return 0
                elif saved is None:
                    self.roi = DEFAULT_ROI
                    log.warn("Nie zaznaczono obszaru — używam domyślnego (dół ekranu). Zmienisz go skrótem.")

            tts = self._tts
            if tts is None:
                from .tts import EdgeTTS

                tts = EdgeTTS(cfg.tts.voice, cfg.tts.pitch, cfg.tts.timeout_s, cfg.tts.retries)
            log.info(f"Głos: {cfg.tts.voice}")
            if cfg.ducking.enabled and self.interactive:
                from .ducking import Ducker

                ducker = Ducker(cfg.ducking.process, cfg.ducking.level)
            greeting = cfg.tts.greeting if self._tts is None else ""
            self.speaker = Speaker(
                tts, cfg.tts, self._audio_factory or self._make_audio, ducker, greeting,
                queue_path=self.base_dir / "kolejka.json" if self.interactive else None,
                debug_dir=self.base_dir / "debug_samples" if cfg.debug.save_samples and self.interactive else None,
            )
            self.speaker.set_paused(self.paused)
            self.speaker.start()

            self._console = ConsoleKeys(
                {
                    "p": self.request_pause,
                    "r": self.request_repeat,
                    "k": self.request_skip,
                    "w": self.request_select,
                    "d": self.request_preview,
                    "q": self.request_quit,
                }
            )
            if self.interactive and cfg.hotkeys.enabled:
                hk = cfg.hotkeys
                hotkeys = GlobalHotkeys(
                    {
                        hk.pause: self.request_pause,
                        hk.repeat: self.request_repeat,
                        hk.skip: self.request_skip,
                        hk.select_region: self.request_select,
                        hk.preview: self.request_preview,
                        hk.quit: self.request_quit,
                    }
                )
            if self.interactive:
                self._print_help(hotkeys)
                log.info("Pauza — lektor milczy." if self.paused else "Wznowiono czytanie.")
            self._loop(ocr, source, store)
        except KeyboardInterrupt:
            log.info("Zamykam (Ctrl+C).")
        finally:
            if hotkeys is not None:
                hotkeys.stop()
            if self.speaker is not None:
                self.speaker.stop()
            if ducker is not None:
                ducker.close()
            source.stop()
            ocr.close()
            if self.interactive:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
        log.info(f"Koniec. Przeczytane kwestie: {self.speaker.spoken if self.speaker else 0}.")
        return 0

    def _make_audio(self):
        from .audio import AudioOut

        return AudioOut(self.cfg.tts.output_device, gapless=self.cfg.tts.gapless)

    def _print_help(self, hotkeys) -> None:
        hk = self.cfg.hotkeys
        log.info("Lektor działa — czyta napisy, które pojawią się w zaznaczonym obszarze.")
        if hotkeys is not None and hotkeys.active:
            log.info(
                f"Skróty: {pretty(hk.pause)} pauza · {pretty(hk.repeat)} powtórz · {pretty(hk.skip)} pomiń · "
                f"{pretty(hk.select_region)} nowy obszar · {pretty(hk.preview)} podgląd · {pretty(hk.quit)} wyjście"
            )
        log.info("W oknie konsoli wystarczą litery: p pauza, r powtórz, k pomiń, w obszar, d podgląd, q wyjście.")

    # --------------------------------------------------------------- pętla
    def _loop(self, ocr, source, store: RoiStore) -> None:
        cfg = self.cfg
        stabilizer = textproc.Stabilizer(cfg.text.stable_reads, cfg.text.stable_similarity)
        repeats = textproc.RepeatFilter(cfg.text.repeat_window_s, cfg.text.repeat_similarity, cfg.text.min_letters)
        interval = max(0.05, float(cfg.ocr.interval_ms) / 1000.0)
        next_tick = time.perf_counter()
        idle_ticks = 0
        while not self.quit:
            if self._console is not None:
                self._console.poll()
            self._apply_requests(source, store, stabilizer)

            if source.finished:  # film testowy się skończył — dokończ czytanie i wyjdź
                idle_ticks = idle_ticks + 1 if self.speaker.idle() else 0
                if idle_ticks >= 3:
                    break
                time.sleep(0.1)
                continue
            size = source.frame_size()
            if size is None:
                time.sleep(0.05)
                continue
            width, height = size
            detected_at = time.monotonic()
            crop = source.read_roi(to_pixels(self.roi, width, height))
            prepared = None
            if crop is not None and not self.paused:
                prepared = preprocess.prepare(
                    crop,
                    cfg.ocr.preprocess,
                    cfg.ocr.brightness,
                    cfg.ocr.outline,
                    preprocess.auto_scale(height, cfg.ocr.upscale, crop, cfg.ocr.brightness),
                )
                text = self._read_text(ocr, prepared)
                stable = stabilizer.feed(text, detected_at)
                if stable:
                    first_seen = stabilizer.first_seen if stabilizer.first_seen is not None else detected_at
                    self._on_subtitle(stable, repeats, first_seen, crop, prepared)
            if self.preview:
                self._show_preview(crop, prepared)

            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._sleep(delay)
            else:
                next_tick = time.perf_counter()

    def _read_text(self, ocr, prepared) -> str:
        if not prepared.has_text:
            self.last_text = ""
            return ""
        started = time.perf_counter()
        try:
            lines = ocr.recognize(prepared.image)
            self._ocr_failures = 0
        except Exception as exc:  # noqa: BLE001 — jeden zły odczyt nie może zatrzymać lektora
            self._ocr_failures += 1
            if self._ocr_failures in (1, 50):
                log.error(f"OCR: {exc}")
            return ""
        self.last_ocr_ms = (time.perf_counter() - started) * 1000
        text = textproc.clean_lines(lines)
        if not textproc.looks_like_text(text, self.cfg.text.min_letters):
            text = ""
        if text and text != self.last_text:
            log.debug(f"OCR {self.last_ocr_ms:.0f} ms: {text}")  # do lektor.log: co dokładnie odczytał OCR
        self.last_text = text
        return text

    def _on_subtitle(self, text: str, repeats, detected_at: float, crop, prepared) -> None:
        cfg = self.cfg
        if any(pattern.search(text) for pattern in self._ignore):
            log.debug(f"pomijam (ignore): {text}")
            return
        to_say = repeats.process(text, detected_at)
        if not to_say:
            log.debug(f"powtórka, pomijam: {text}")
            return
        speech = textproc.prepare_for_speech(to_say, cfg.replacements, cfg.text.strip_speaker_names)
        if not speech:
            return
        self.detected.append(to_say)
        log.info(f"Napis: {to_say}")
        self.speaker.say(speech, detected_at)
        if cfg.debug.save_samples:
            self._save_sample(to_say, crop, prepared)

    def _apply_requests(self, source, store: RoiStore, stabilizer) -> None:
        if self._want_pause:
            self._want_pause = False
            self.paused = not self.paused
            self.speaker.set_paused(self.paused)
            stabilizer.reset()
            log.info("Pauza — lektor milczy." if self.paused else "Wznowiono czytanie.")
        if self._want_preview:
            self._want_preview = False
            self.preview = (not self.preview) and self.interactive
            if not self.preview:
                self._close_preview()
            log.info("Podgląd włączony." if self.preview else "Podgląd wyłączony.")
        if self._want_select:
            self._want_select = False
            if not self.interactive:
                return
            self.paused = True
            self.speaker.set_paused(True)
            stabilizer.reset()
            log.info("Pauza — lektor milczy.")
            frame = source.snapshot(timeout=3.0)
            if frame is None:
                log.warn("Brak klatki do zaznaczenia obszaru.")
                return
            self._close_preview()
            picked = select_roi(frame, self.roi)
            if picked is not None:
                self.roi = picked
                store.set(source.signature, picked)
                stabilizer.reset()
                log.info("Zapisano nowy obszar napisów.")
            else:
                log.info("Anulowano — zostaje poprzedni obszar.")

    # ------------------------------------------------------------- podgląd
    def _show_preview(self, crop, prepared) -> None:
        if crop is None:
            return
        if self._preview_open and cv2.getWindowProperty(PREVIEW, cv2.WND_PROP_VISIBLE) < 1:
            self._preview_open = False
            self.preview = False
            log.info("Podgląd zamknięty (włączysz go ponownie klawiszem d).")
            return
        panels = [crop]
        if prepared is not None:
            image = prepared.image
            panels.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image)
        width = 960
        rows = []
        for panel in panels:
            h, w = panel.shape[:2]
            factor = width / max(1, w)
            interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
            rows.append(cv2.resize(panel, (width, max(1, int(h * factor))), interpolation=interp))
        status = np.zeros((30, width, 3), np.uint8)
        state = "PAUZA" if self.paused else "czyta"
        backlog = self.speaker.backlog() if self.speaker is not None else 0
        message = f"{state} | OCR {self.last_ocr_ms:.0f} ms | kolejka {backlog} | p pauza  w obszar  d zamknij"
        cv2.putText(status, message, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(PREVIEW, np.vstack(rows + [status]))
        self._preview_open = True

    def _close_preview(self) -> None:
        if self._preview_open:
            try:
                cv2.destroyWindow(PREVIEW)
                cv2.waitKey(1)
            except cv2.error:
                pass
            self._preview_open = False

    def _sleep(self, delay: float) -> None:
        if self._preview_open:
            key = cv2.waitKey(max(1, int(delay * 1000))) & 0xFF
            if key != 255 and self._console is not None:
                self._console.handle(chr(key))
        else:
            time.sleep(delay)

    def _save_sample(self, text: str, crop, prepared) -> None:
        folder = self.base_dir / "debug_samples"
        try:
            folder.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            if crop is not None:
                cv2.imencode(".png", crop)[1].tofile(str(folder / f"{stamp}-1-wycinek.png"))
            if prepared is not None:
                cv2.imencode(".png", prepared.image)[1].tofile(str(folder / f"{stamp}-2-dla-ocr.png"))
            (folder / f"{stamp}-3-tekst.txt").write_text(text, encoding="utf-8")
        except (OSError, cv2.error) as exc:
            log.debug(f"Nie zapisałem próbki: {exc}")
