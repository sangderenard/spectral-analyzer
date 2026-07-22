from work_preview_toolbar import WorkPreviewToolbar
import live_spectral_text_demo as demo


def test_work_preview_tabs_are_canonical_panel_buttons():
    toolbar = WorkPreviewToolbar()
    knobs = {knob.name: knob for knob in toolbar.panel.knobs}

    assert toolbar.mode == "whole-work"
    assert set(knobs) == {"whole-work", "work-piece"}
    assert all(knob.control_widget == "button" for knob in knobs.values())
    assert toolbar.panel.payload["role"] == "view_tabs"


def test_work_preview_tab_routes_without_changing_render_settings(monkeypatch):
    toolbar = WorkPreviewToolbar()
    monkeypatch.setattr(
        toolbar._grid, "route_event", lambda _event, _rect: ("work-piece", 1)
    )

    assert toolbar.handle_event(object(), (0, 0, 200, 24)) == (
        "work-preview:work-piece"
    )
    assert toolbar.mode == "work-piece"


def test_work_piece_crop_uses_render_dimensions_then_fits_ui(monkeypatch):
    class Surface:
        def __init__(self, size):
            self.size = size
            self.last_rect = None

        def get_size(self):
            return self.size

        def subsurface(self, rect):
            self.last_rect = tuple(rect)
            return Surface((rect[2], rect[3]))

        def copy(self):
            return self

    source = Surface((1024, 1024))
    monkeypatch.setattr(
        demo,
        "_fit_panel_preview_texture",
        lambda texture, width, height: (texture.get_size(), width, height),
    )

    fitted = demo._active_work_texture(
        source, None, 320, 180, 256, 256, 1024, 1
    )

    assert source.last_rect == (0, 0, 256, 256)
    assert fitted == ((256, 256), 320, 180)
