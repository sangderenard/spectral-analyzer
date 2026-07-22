from types import SimpleNamespace

from camera_software.control_layout import program_ui_manifest
from exposure_control_toolbar import ExposureControlToolbar
from present_work_panel import PresentWorkPanel
from ray_trace_toolbar import RayTraceToolbar
from top_toolbar_rows import TopToolbarRows


class _Calibration:
    key = "glass"
    label = "Glass"
    description = "Measure spectral transmission through known glass."

    @staticmethod
    def work_asset_manifest():
        return {"scene": {"material": "bk7"}, "validator": "dispersion"}


def test_program_manifest_declares_a_pluggable_present_work_host():
    panel = next(
        item for item in program_ui_manifest().panels
        if item.name == "editor-text"
    )

    assert panel.label == "PRESENT WORK"
    assert panel.payload["role"] == "present_work_host"
    assert panel.payload["widget"] == "PresentWorkPanel"
    assert set(panel.payload["content_modules"]) == {
        "text-material", "calibration", "toolbar", "detail",
    }


def test_present_work_selection_switches_content_without_destroying_text():
    panel = PresentWorkPanel("editable material sample")
    assert panel.text_editor_active

    panel.show_calibration(_Calibration(), {"thickness_mm": 12.0})
    assert not panel.text_editor_active
    assert panel.document.kind == "calibration"
    assert any("scene.material: bk7" in line for line in panel.document.lines)
    assert any("thickness_mm: 12.0" in line for line in panel.document.lines)

    panel.show_work({
        "kind": "asset",
        "subtype": SimpleNamespace(
            kind="material", display_name="Amber panel material", text=""
        ),
    })
    assert panel.text_editor_active
    assert panel.text == "editable material sample"


def test_toolbar_rows_expand_into_the_present_work_form_and_route_changes():
    ray = RayTraceToolbar()
    exposure = ExposureControlToolbar()
    rows = TopToolbarRows(ray, exposure)
    panel = PresentWorkPanel("sample")

    panel.show_toolbar("exposure-toolbar", "EXPOSURE")
    advanced = rows.advanced_rows(panel.document.toolbar_row)
    assert {name for name, _label, _value in advanced} == {
        "allocation_mode", "grid_mode", "subdivision_axis",
        "locked_grid_columns", "locked_grid_rows", "work_width_px",
        "work_height_px", "final_edge_px",
    }

    before = exposure.settings.allocation_mode
    assert rows.handle_routed("allocation_mode", 1) == (
        "exposure-settings-changed"
    )
    assert exposure.settings.allocation_mode != before
