from types import SimpleNamespace

from lektor import config
from lektor.app import App
from lektor.roi import RoiStore


def test_end_of_video_waits_for_backlog_longer_than_thirty_seconds(tmp_path, monkeypatch):
    import lektor.app as module

    elapsed = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    app = App(config.load(None), tmp_path, interactive=False)
    app.speaker = SimpleNamespace(idle=lambda: elapsed[0] >= 45)
    app._loop(None, SimpleNamespace(finished=True), RoiStore(tmp_path / "roi.json"))
    assert elapsed[0] >= 45
