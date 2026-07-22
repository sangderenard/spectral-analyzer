import json
from types import SimpleNamespace

from bass_viewer import ScrollableSubpanelList
from camera_loadout_browser import (
    CameraLoadoutBrowser,
    camera_loadout_sections,
    selected_camera_context,
)


def _order():
    return {
        "schema_version": 1,
        "defaults": {
            "image": {
                "width": 800,
                "height": 600,
                "region": {"x": 100, "y": 50, "width": 400, "height": 300},
            },
            "camera": {
                "focal_mm": 90.0,
                "aperture_mm": 30.0,
                "position_m": [2.0, -1.0, 0.5],
                "target_m": [0.0, 0.0, 0.0],
                "focus_target_m": [0.1, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "lens_shift_x_mm": 0.012345678,
                "lens_tilt_y_deg": -0.00000125,
                "back_type": "6x7 roll-film back",
            },
            "exposure": {"time_s": 0.125, "iso": 64, "sensor_sweeps": 12},
        },
        "jobs": [{"id": "portrait", "token": "portrait"}],
    }


def test_camera_browser_uses_same_repository_scrolling_widget_as_work_browser():
    browser = CameraLoadoutBrowser()

    assert type(browser.widget) is ScrollableSubpanelList


def test_camera_hierarchy_contains_detailed_lens_focus_movement_back_and_crop():
    context = selected_camera_context({}, _order())
    sections = camera_loadout_sections(context)
    by_key = {section["key"]: section for section in sections}

    assert set(by_key) >= {
        "selection", "camera", "lens", "focus", "movements",
        "back", "sensor", "exposure", "output",
    }
    assert "f-number: f/3.000000" in by_key["lens"]["lines"]
    assert "lens shift X: +0.012345678 mm" in by_key["movements"]["lines"]
    assert "lens tilt Y: -0.000001250 deg" in by_key["movements"]["lines"]
    assert "back type: 6x7 roll-film back" in by_key["back"]["lines"]
    assert "crop origin: (100, 50) px" in by_key["sensor"]["lines"]
    # Unspecified physical backs resolve to the actual 56 mm square thick-lens
    # plate, not the old display-only 36x24 assumption.
    assert "active sensor: 28.000000000 x 28.000000000 mm" in by_key["sensor"]["lines"]


def test_completed_asset_prefers_solved_renderer_summary(tmp_path):
    scene_path = tmp_path / "scene_order.json"
    scene_path.write_text(json.dumps(_order()), encoding="utf-8")
    summary_path = tmp_path / "0000_cpp_summary.json"
    summary_path.write_text(json.dumps({
        "frame_config_summary": {
            "camera_solve_status": "ok",
            "camera_solve_error": 1.25e-9,
            "camera_coc_um": 0.75,
            "camera_sensor_adjust_mm": -0.000125,
            "camera_front_shift_mm": 1.5,
            "camera_front_tilt_deg": 0.25,
            "camera_mode_name": "thick lens",
            "thin_lens_planes": False,
        }
    }), encoding="utf-8")
    archived = SimpleNamespace(
        scene_path=str(scene_path),
        resolved_summary_path=str(summary_path),
        camera_spec={},
        exposure_spec={},
    )
    subtype = SimpleNamespace(text="portrait", scenes=(archived,))
    context = selected_camera_context({"kind": "asset", "subtype": subtype})
    by_key = {
        section["key"]: section for section in camera_loadout_sections(context)
    }

    assert context["has_solved_state"] is True
    assert "solve status: ok" in by_key["focus"]["lines"]
    assert "circle of confusion: 0.750000 um" in by_key["focus"]["lines"]
    assert "front tilt solved: 0.250000000 deg" in by_key["movements"]["lines"]


def test_sync_retains_expansion_state_when_selected_work_updates():
    browser = CameraLoadoutBrowser()
    browser.sync({}, fallback_order=_order())
    movement = next(
        spec for spec in browser.widget.subpanels if spec.key == "movements"
    )
    movement.expanded = True
    changed = _order()
    changed["defaults"]["camera"]["focal_mm"] = 100.0

    browser.sync({}, fallback_order=changed)

    movement = next(
        spec for spec in browser.widget.subpanels if spec.key == "movements"
    )
    assert movement.expanded is True


def test_camera_browser_tracks_selected_work_progress_metrics():
    browser = CameraLoadoutBrowser()

    browser.sync(
        {},
        progress_metrics={
            "convergence": 0.72,
            "convergence_velocity_per_pass": 1.25e-3,
            "priority_share": 0.25,
            "working": True,
            "status": "working",
        },
    )

    assert browser.progress_metrics["convergence"] == 0.72
    assert browser.progress_metrics["convergence_velocity_per_pass"] == 1.25e-3
    assert browser.progress_metrics["priority_share"] == 0.25
    assert browser.progress_metrics["working"] is True


def test_design_subtype_exposes_pose_animation_under_camera_hierarchy():
    animation = SimpleNamespace(
        name="rack focus in / hold / rack focus out",
        rack_frames=12,
        hold_frames=6,
        focus_location_count=7,
        replacement_frame_index=15,
        animation_key="pose-animation:demo",
    )
    subtype = SimpleNamespace(
        display_name="representative nine-slice square",
        pose_animations=(animation,),
    )

    context = selected_camera_context(
        {"kind": "design", "subtype": subtype}, _order()
    )
    sections = {
        section["key"]: section for section in camera_loadout_sections(context)
    }

    assert context["selection_label"] == "representative nine-slice square"
    assert "pose_animations" in sections
    assert "rack in: 12 frames" in sections["pose_animations"]["lines"]
    assert "focus locations: 7" in sections["pose_animations"]["lines"]
