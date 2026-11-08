"""Windows status readers can briefly block atomic replacement."""

from camera_designer import marble_preview_worker as worker


def test_atomic_status_retries_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    original = worker.os.replace
    calls = 0

    def replace(source, destination):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(13, "status reader holds destination")
        return original(source, destination)

    monkeypatch.setattr(worker.os, "replace", replace)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    worker._atomic_json(target, {"state": "exposing"})
    assert calls == 3
    assert target.read_text(encoding="utf-8") == '{"state": "exposing"}'
