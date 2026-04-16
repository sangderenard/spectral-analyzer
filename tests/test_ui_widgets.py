from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _FakeText:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_width(self) -> int:
        return max(8, len(self._text) * 7)

    def get_height(self) -> int:
        return 16


class _FakeFont:
    def render(self, text: str, *_args, **_kwargs) -> _FakeText:
        return _FakeText(text)

    def get_linesize(self) -> int:
        return 16


class _FakeRect:
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def centerx(self) -> int:
        return self.x + self.w // 2

    @property
    def centery(self) -> int:
        return self.y + self.h // 2

    def collidepoint(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom


class _FakeSurface:
    def __init__(self, size: tuple[int, int], *_args, **_kwargs) -> None:
        self.w, self.h = size
        self._clip = _FakeRect(0, 0, self.w, self.h)

    def fill(self, *_args, **_kwargs) -> None:
        return None

    def blit(self, *_args, **_kwargs) -> None:
        return None

    def get_clip(self) -> _FakeRect:
        return self._clip

    def set_clip(self, rect: _FakeRect) -> None:
        self._clip = rect


def _patch_widget_pygame(monkeypatch):
    import types
    import bass_viewer

    monkeypatch.setattr(bass_viewer.pygame, "Rect", _FakeRect)
    monkeypatch.setattr(bass_viewer.pygame, "Surface", _FakeSurface)
    monkeypatch.setattr(bass_viewer.pygame, "SRCALPHA", 0, raising=False)
    if not hasattr(bass_viewer.pygame, "draw"):
        monkeypatch.setattr(
            bass_viewer.pygame,
            "draw",
            types.SimpleNamespace(),
            raising=False,
        )
    monkeypatch.setattr(bass_viewer.pygame.draw, "rect", lambda *a, **k: None, raising=False)
    return bass_viewer


def test_scrollable_subpanel_list_click_toggles_selection_and_expansion(monkeypatch):
    bass_viewer = _patch_widget_pygame(monkeypatch)
    ModularSubpanelSpec = bass_viewer.ModularSubpanelSpec
    ScrollableSubpanelList = bass_viewer.ScrollableSubpanelList

    font = _FakeFont()
    surf = _FakeSurface((320, 280))
    widget = ScrollableSubpanelList("Modules", max_height=120)
    widget.set_subpanels([
        ModularSubpanelSpec(key="fft", title="FFT", summary_lines=["a", "b"]),
        ModularSubpanelSpec(key="fb", title="FB", summary_lines=["c"]),
    ])

    widget.render(surf, font, 0, 0, 320)
    first_header = widget._header_rects["fft"]
    hit = widget.handle_click(first_header.centerx, first_header.centery)

    assert hit == "fft"
    assert widget.selected_key == "fft"
    assert widget.subpanels[0].expanded is False


def test_scrollable_subpanel_list_scrolls_and_clamps(monkeypatch):
    bass_viewer = _patch_widget_pygame(monkeypatch)
    ModularSubpanelSpec = bass_viewer.ModularSubpanelSpec
    ScrollableSubpanelList = bass_viewer.ScrollableSubpanelList

    font = _FakeFont()
    surf = _FakeSurface((320, 320))
    widget = ScrollableSubpanelList("Modules", max_height=80)
    widget.set_subpanels([
        ModularSubpanelSpec(
            key=f"k{i}",
            title=f"Panel {i}",
            summary_lines=["line1", "line2", "line3"],
        )
        for i in range(6)
    ])

    widget.render(surf, font, 0, 0, 320)
    assert widget._content_h > widget.max_height

    assert widget.handle_scroll(3) is True
    assert widget.scroll_y > 0

    widget.handle_scroll(999)
    max_scroll = max(0, widget._content_h - widget.max_height)
    assert widget.scroll_y == max_scroll


def test_scrollable_subpanel_list_emits_add_and_remove_actions(monkeypatch):
    bass_viewer = _patch_widget_pygame(monkeypatch)
    ModularSubpanelSpec = bass_viewer.ModularSubpanelSpec
    ScrollableSubpanelList = bass_viewer.ScrollableSubpanelList
    SubpanelAddOption = bass_viewer.SubpanelAddOption

    font = _FakeFont()
    surf = _FakeSurface((360, 240))
    widget = ScrollableSubpanelList("Modules", max_height=80)
    widget.show_remove_button = True
    widget.set_add_options([SubpanelAddOption("fft", "FFT")])
    widget.set_subpanels([ModularSubpanelSpec(key="fft_1", title="FFT 1")])
    widget.selected_key = "fft_1"

    widget.render(surf, font, 0, 0, 360)
    add_rect = widget._add_button_rects["fft"]
    rem_rect = widget._remove_button_rect

    assert widget.handle_click(add_rect.centerx, add_rect.centery) == "add:fft"
    assert rem_rect is not None
    assert widget.handle_click(rem_rect.centerx, rem_rect.centery) == "remove:fft_1"


def test_source_panel_itinerary_add_remove_and_toggle():
    from bass_viewer import SourcePanel

    panel = SourcePanel(input_dir=ROOT, output_root=ROOT)
    assert [item["engine"] for item in panel._analysis_itinerary] == [
        "fft", "fb", "wavelet"
    ]

    panel._add_analysis_module("fft")
    fft_items = [item for item in panel._analysis_itinerary if item["engine"] == "fft"]
    assert len(fft_items) == 2

    panel._sync_itinerary_from_engine_toggle("fft", False)
    assert all(item["engine"] != "fft" for item in panel._analysis_itinerary)
    assert panel.include_cqt is False

    panel._sync_itinerary_from_engine_toggle("fft", True)
    assert any(item["engine"] == "fft" for item in panel._analysis_itinerary)
    assert panel.include_cqt is True


def test_source_panel_module_control_paths_update_engine_settings():
    from bass_viewer import SourcePanel

    panel = SourcePanel(input_dir=ROOT, output_root=ROOT)
    fft_key = next(item["key"] for item in panel._analysis_itinerary if item["engine"] == "fft")
    old_algo = panel.cqt_algorithm_idx
    old_hop = panel.hop_length

    assert panel._handle_module_control(("module", fft_key, "algorithm", "next")) is True
    assert panel.cqt_algorithm_idx != old_algo

    assert panel._handle_module_control(("module", fft_key, "hop", "next")) is True
    assert panel.hop_length != old_hop


def test_source_panel_build_analysis_itinerary_keeps_per_module_ranges():
    from bass_viewer import SourcePanel

    panel = SourcePanel(input_dir=ROOT, output_root=ROOT)
    fft_key = next(item["key"] for item in panel._analysis_itinerary if item["engine"] == "fft")
    fb_key = next(item["key"] for item in panel._analysis_itinerary if item["engine"] == "fb")
    panel._module_item(fft_key)["start_sec"] = 1.0
    panel._module_item(fft_key)["end_sec"] = 3.5
    panel._module_item(fft_key)["total_sec"] = 10.0
    panel._module_item(fb_key)["start_sec"] = 4.0
    panel._module_item(fb_key)["end_sec"] = 0.0
    panel._module_item(fb_key)["total_sec"] = 10.0

    itinerary = panel.build_analysis_itinerary()
    fft_run = next(run for run in itinerary.runs if run.key == fft_key)
    fb_run = next(run for run in itinerary.runs if run.key == fb_key)

    assert fft_run.time_range.start_sec == 1.0
    assert fft_run.time_range.end_sec == 3.5
    assert fb_run.time_range.as_fraction_span() == (0.4, 1.0)


def test_source_panel_folder_rows_include_inventory_timelines(tmp_path):
    from analysis_itinerary import AnalysisArtifact, AnalysisDatasetRecord, AnalysisInventory, AnalysisTimeRange
    from bass_viewer import SourcePanel

    analysis_dir = tmp_path / "demo_analysis"
    analysis_dir.mkdir()
    (analysis_dir / "filterbank").mkdir()
    inv = AnalysisInventory(datasets=[
        AnalysisDatasetRecord(
            dataset_key="fft:a1",
            engine="fft",
            algorithm="stft",
            settings_hash="a1",
            time_range=AnalysisTimeRange(start_sec=1.0, end_sec=3.0, total_sec=8.0),
            artifacts=[AnalysisArtifact(kind="npz", path="cqt_data_a1.npz")],
        )
    ])
    (analysis_dir / "analysis_inventory.json").write_text(inv.to_json(), encoding="utf-8")

    panel = SourcePanel(input_dir=str(tmp_path), output_root=str(tmp_path))
    panel._refresh_folders()
    rows = {item.key: item for item in panel.folder_list.items}

    assert str(analysis_dir) in rows
    assert rows[str(analysis_dir)].timeline_segments
    dataset_key = f"{analysis_dir}::dataset::fft:a1"
    assert dataset_key in rows
    assert rows[dataset_key].timeline_segments
