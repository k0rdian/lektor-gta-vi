import json
import time
import subprocess
import sys

import pytest

from lektor.pending import PendingQueue


def test_failed_atomic_write_keeps_previous_queue(tmp_path, monkeypatch):
    path = tmp_path / "kolejka.json"
    queue = PendingQueue(path)
    line = queue.add("Zażółć gęślą jaźń.", time.monotonic())
    previous = path.read_bytes()

    def fail(*args):
        raise OSError("brak miejsca")

    monkeypatch.setattr("lektor.pending.os.replace", fail)
    with pytest.raises(OSError):
        queue.add("Nowa kwestia", time.monotonic())
    assert path.read_bytes() == previous
    assert [item.text for item in queue.snapshot()] == [line.text]
    with pytest.raises(OSError):
        queue.complete(line)
    assert path.read_bytes() == previous
    assert queue.contains(line)
    assert not list(tmp_path.glob("*.tmp"))


def test_recovered_detection_time_does_not_use_old_monotonic_clock(tmp_path):
    path = tmp_path / "kolejka.json"
    queue = PendingQueue(path)
    queue.add("Kwestia sprzed minuty.", time.monotonic() - 60)
    restored = PendingQueue(path).snapshot()
    assert 59 <= time.monotonic() - restored[0].detected_at <= 62
    assert json.loads(path.read_text(encoding="utf-8"))["pending"][0]["text"] == restored[0].text


def test_accepted_queue_survives_forced_process_termination(tmp_path):
    from lektor.controls import stop_worker

    path = tmp_path / "kolejka.json"
    script = (
        "import sys,time; from pathlib import Path; from lektor.pending import PendingQueue; "
        "q=PendingQueue(Path(sys.argv[1])); "
        "q.add('Zażółć gęślą jaźń.', time.monotonic()); "
        "q.add('Druga kwestia.', time.monotonic()); "
        "print('saved', flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(path)],
        stdout=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        assert process.stdout.readline().strip() == "saved"
    finally:
        stop_worker(process)
    assert [line.text for line in PendingQueue(path).snapshot()] == ["Zażółć gęślą jaźń.", "Druga kwestia."]
