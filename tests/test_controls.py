import io
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lektor import __main__ as cli, config
from lektor.app import App
from lektor.controls import stop_worker


def test_worker_starts_paused_without_spoken_greeting(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "BASE_DIR", tmp_path)  # lektor.log nie trafia do folderu aplikacji

    def run(app):
        assert app.paused is True
        assert app.cfg.tts.greeting == ""
        return 0

    monkeypatch.setattr(App, "run", run)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["--worker", "--start-paused"]) == 0


def test_cancel_selection_closes_source_without_starting_audio(tmp_path, monkeypatch):
    import lektor.app as module

    closed = []
    source = SimpleNamespace(
        signature="screen:test", start=lambda: None,
        snapshot=lambda **kwargs: np.zeros((100, 200, 3), dtype=np.uint8),
        stop=lambda: closed.append("source"),
    )
    ocr = SimpleNamespace(name="test", close=lambda: closed.append("ocr"))
    monkeypatch.setattr(module, "select_roi", lambda *args: None)
    app = App(config.load(None), tmp_path, ocr=ocr, source=source)
    assert app.run() == 0
    assert app.speaker is None
    assert closed == ["source", "ocr"]


@pytest.mark.skipif(os.name != "nt", reason="Windows process tree")
def test_stop_kills_worker_and_venv_child():
    import psutil

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        descendants = []
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            descendants = psutil.Process(process.pid).children(recursive=True)
            if descendants:
                break
            time.sleep(0.05)
        stop_worker(process)
        assert process.poll() is not None
        _, alive = psutil.wait_procs(descendants, timeout=3)
        assert not alive
    finally:
        stop_worker(process)
