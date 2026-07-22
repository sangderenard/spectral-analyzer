from types import SimpleNamespace

import numpy as np
from PIL import Image

from camera_software import (
    WindowElementHarvestCache,
    build_window_element_scene_manifest,
    layout_program_ui,
    program_ui_manifest,
)


def _layout():
    return layout_program_ui(
        program_ui_manifest(), 360, 180, work_width=100, work_height=100
    )


def test_window_manifest_retains_panels_controls_text_and_nine_slice_children():
    manifest = build_window_element_scene_manifest(_layout())
    by_key = {item.element_key: item for item in manifest.elements}

    assert by_key["program-backdrop"].kind == "window"
    assert by_key["work-visual-pass"].kind == "window.button"
    assert by_key["camera-label"].kind == "window.text_run"
    patches = [
        item for item in manifest.elements
        if item.parent_key == "camera-panel"
        and item.kind == "window.panel_patch"
    ]
    assert len(patches) == 9
    assert all(item.style_object_key for item in patches)
    assert all(item.content_signature for item in manifest.elements)


def test_holistic_crop_cache_only_reuses_exact_element_and_condition(tmp_path):
    manifest = build_window_element_scene_manifest(_layout())
    root = manifest.root_rect_px
    image_path = tmp_path / "whole.png"
    linear_path = tmp_path / "whole.npy"
    Image.new("RGBA", (root[2], root[3]), (12, 34, 56, 255)).save(image_path)
    np.save(linear_path, np.ones((root[3], root[2], 3), np.float32))
    cache = WindowElementHarvestCache(str(tmp_path / "harvest.json"))
    condition = cache.condition_signature(
        camera_key="camera-A", lighting_key="flash-A",
        material_key="materials-A", reflection_boundary_key="room-A",
        color_transform_key="display-A",
    )

    records = cache.harvest(
        manifest,
        condition_signature=condition,
        display_image_path=str(image_path),
        linear_image_path=str(linear_path),
        output_directory=str(tmp_path / "plates"),
    )
    camera_panel = next(
        item for item in manifest.elements
        if item.element_key == "camera-panel"
    )

    assert records
    assert cache.find(camera_panel, condition) is not None
    assert cache.find(camera_panel, condition).linear_path.endswith("_linear.npy")
    assert cache.find(camera_panel, "different-condition") is None


def test_scroll_rows_body_text_and_two_progress_pies_join_the_manifest():
    class Rect:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h

    spec = SimpleNamespace(
        key="lens", title="LENS / APERTURE", summary_lines=["f/4", "82.5 mm"],
        expanded=True, enabled=True, accent_rgb=(90, 140, 205),
    )
    widget = SimpleNamespace(
        title="CAMERA LOADOUT", scroll_y=0, selected_key="lens",
        _viewport_rect=Rect(6, 32, 88, 90), _content_h=90,
        _header_rects={"lens": Rect(6, 32, 88, 22)},
        subpanels=[spec],
        control_table=SimpleNamespace(sorted_records=lambda: []),
    )
    browser = SimpleNamespace(
        widget=widget,
        progress_metrics={
            "convergence": 0.45, "priority_share": 0.25,
            "working": True, "convergence_velocity_per_pass": 1.2e-3,
        },
    )
    manifest = build_window_element_scene_manifest(
        _layout(), widgets={"camera-panel": browser}
    )

    kinds = [item.kind for item in manifest.elements]
    assert "window.list_row" in kinds
    assert "window.list_body_text" in kinds
    pies = [item for item in manifest.elements if item.kind == "window.progress_pie"]
    assert len(pies) == 2
    assert {item.metadata["value"] for item in pies} == {0.45, 0.25}
