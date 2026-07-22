from render_progress_toolbar import RenderProgressToolbar
import live_spectral_text_demo as demo


def test_progress_toolbar_uses_canonical_knobs_for_pause_and_cancel():
    toolbar = RenderProgressToolbar()
    knobs = {knob.name: knob for knob in toolbar.panel.knobs}

    assert len(knobs) == 6
    assert knobs["pause_render"].control_widget == "button"
    assert knobs["cancel_render"].control_widget == "button"
    assert knobs["pause_render"].choices == ["PAUSE", "RESUME"]


def test_progress_toolbar_routes_canonical_button_actions(monkeypatch):
    toolbar = RenderProgressToolbar()
    monkeypatch.setattr(
        toolbar._grid, "route_event", lambda _event, _rect: ("pause_render", 1)
    )
    assert toolbar.handle_event(object(), (0, 0, 100, 20)) == (
        "toggle-render-pause"
    )

    monkeypatch.setattr(
        toolbar._grid, "route_event", lambda _event, _rect: ("cancel_render", 1)
    )
    assert toolbar.handle_event(object(), (0, 0, 100, 20)) == "cancel-render"


def test_worker_forwards_pause_and_cancel_to_active_renderer():
    def render(*_args, **_kwargs):
        raise AssertionError("renderer should not start in this test")

    calls = []
    render.toggle_paused = lambda: calls.append("pause") or True
    render.cancel_current = lambda: calls.append("cancel") or True
    worker = demo.SpectralTextRenderWorker(render)
    try:
        assert worker.toggle_foreground_paused() is True
        assert worker.cancel_foreground() is True
        assert calls == ["pause", "cancel"]
    finally:
        worker.close()
