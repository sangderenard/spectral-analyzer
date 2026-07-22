from types import SimpleNamespace

from bass_viewer import ScrollableSubpanelList
from render_work_browser import RenderWorkBrowser, artifact_convergence


def test_work_browser_directly_uses_repository_scrolling_widget():
    browser = RenderWorkBrowser()

    assert type(browser.widget) is ScrollableSubpanelList


def test_work_browser_lists_jobs_completed_assets_and_selects_preview(tmp_path):
    preview = tmp_path / "actually.png"
    preview.write_bytes(b"preview")
    request = SimpleNamespace(
        request_key="request-1",
        token_asset=SimpleNamespace(token="pending"),
        target_kind=SimpleNamespace(value="token_sequence"),
        refinement_pass=4,
    )
    artifact = SimpleNamespace(
        preview_path=str(preview), linear_path="", samples=64
    )
    subtype = SimpleNamespace(
        subtype_key="subtype-1",
        text="actually",
        kind="token",
        complete=True,
        artifacts=(artifact,),
    )
    bundle = SimpleNamespace(
        display_name="DejaVu Sans Mono / red ink",
        subtypes=(subtype,),
    )
    browser = RenderWorkBrowser()

    browser.sync(
        [request], [bundle], active=("style-1", "A", 7, True)
    )

    titles = [spec.title for spec in browser.widget.subpanels]
    assert titles[0:2] == ["VALIDATION / STARTUP", "CALIBRATION ROOMS"]
    assert "  FOCUS HALL CALIBRATION ROOM" in titles
    assert "  PRISM ROOM CALIBRATION ROOM" in titles
    assert "  DOUBLE SLIT CALIBRATION ROOM" in titles
    assert titles[-1] == "WORK / ASSETS"
    browser.set_work_assets_expanded(True)
    browser.sync([request], [bundle], active=("style-1", "A", 7, True))
    titles = [spec.title.strip() for spec in browser.widget.subpanels]
    assert titles[-4:] == [
        "WORK / ASSETS", "● WORKING  A", "QUEUED  pending", "DONE  actually",
    ]
    browser.widget.selected_key = "asset:subtype-1"
    assert browser.selected_preview_path == str(preview)
    assert browser.selected_payload["subtype"] is subtype


def test_active_pending_row_is_marked_working_and_priority_is_normalized():
    requests = [
        SimpleNamespace(
            request_key=f"request-{token}",
            token_asset=SimpleNamespace(token=token),
            target_kind=SimpleNamespace(value="atlas_glyph"),
            target_key=f"asset-{token}",
            refinement_pass=0,
        )
        for token in ("A", "B")
    ]
    browser = RenderWorkBrowser()
    browser.set_work_assets_expanded(True)

    browser.sync(requests, [], active=("style", "B", 0, True))

    titles = [spec.title for spec in browser.widget.subpanels]
    stripped = [title.strip() for title in titles]
    assert stripped[-3:] == ["WORK / ASSETS", "QUEUED  A", "● WORKING  B"]
    shares = [
        float(spec.payload["priority_share"])
        for spec in browser.widget.subpanels
        if spec.key.startswith("job:")
    ]
    assert sum(shares) == 1.0
    browser.widget.selected_key = "job:request-B"
    assert browser.selected_metrics["working"] is True
    assert browser.selected_metrics["status"] == "working"


def test_artifact_convergence_uses_retained_quality_evidence():
    developing = SimpleNamespace(
        samples=12,
        metadata={
            "atlas_quality": {
                "glyph_exposure_coverage": 1.0,
                "glyph_radiance_coverage": 1.0,
                "relative_rmse": 0.0,
                "p95_relative_delta": 0.0,
                "stable_hold": 1,
                "converged": False,
            }
        },
    )
    converged = SimpleNamespace(
        samples=13,
        metadata={"atlas_quality": {"converged": True}},
    )

    assert 0.8 < artifact_convergence(developing) < 1.0
    assert artifact_convergence(converged) == 1.0


def test_work_row_displays_convergence_velocity_in_scientific_notation():
    artifact = SimpleNamespace(
        preview_path="",
        linear_path="",
        samples=128,
        created_at_s=1.0,
        metadata={
            "checkpoint_epochs_per_exposure": 64,
            "atlas_quality": {
                "convergence_metric": 0.75,
                "convergence_velocity_per_pass": 3.125e-3,
            },
        },
    )
    subtype = SimpleNamespace(
        subtype_key="subtype-velocity",
        source_asset_key="asset",
        text="V",
        kind="glyph",
        complete=False,
        artifacts=(artifact,),
    )
    browser = RenderWorkBrowser()
    browser.set_work_assets_expanded(True)

    browser.sync(
        [],
        [SimpleNamespace(display_name="style", subtypes=(subtype,))],
    )

    row = next(
        spec for spec in browser.widget.subpanels
        if spec.key == "asset:subtype-velocity"
    )
    browser.widget.selected_key = "asset:subtype-velocity"
    assert "convergence velocity: +3.125e-03 pass^-1" in row.summary_lines
    assert "quality checkpoint: every 64 passes" in row.summary_lines
    assert browser.selected_metrics[
        "convergence_velocity_per_pass"
    ] == 3.125e-3


def test_layout_design_subtypes_and_pose_cache_appear_in_work_browser(tmp_path):
    from camera_software import (
        LayoutWorkProgressCache,
        build_layout_object_work_manifest,
        layout_program_ui,
        program_ui_manifest,
    )

    layout = layout_program_ui(
        program_ui_manifest(), 360, 180, work_width=100, work_height=100
    )
    manifest = build_layout_object_work_manifest(layout, str(tmp_path))
    cache = LayoutWorkProgressCache(manifest.progress_cache_path)
    browser = RenderWorkBrowser()
    browser.set_work_assets_expanded(True)

    browser.sync([], [], layout_work=(manifest, cache))

    design_rows = [
        spec for spec in browser.widget.subpanels
        if spec.key.startswith("design:")
    ]
    assert len(design_rows) == 2
    assert all(
        spec.title.strip().startswith("DESIGN UNRENDERED") for spec in design_rows
    )
    assert all(
        any("rack focus in / hold / rack focus out" in line
            for line in spec.summary_lines)
        for spec in design_rows
    )


def test_calibration_selector_defaults_to_real_physical_scene():
    browser = RenderWorkBrowser()

    assert browser.selected_calibration_mode.key == "color-science"
    assert browser.calibration_manifest["scene"]["real_camera_exposure"] is True
    assert browser.calibration_manifest["transport"]["lane_table"]["domain"] == "fixed_spectral"

    browser.select_calibration_mode("focus-hall")
    manifest = browser.calibration_manifest
    assert manifest["calibration_mode"] == "focus-hall"
    assert manifest["scene"]["scene_type"] == "focus_hall"
    assert len(manifest["scene"]["distance_cards"]) >= 8


def test_startup_spectral_results_are_primary_rows_before_collapsed_work():
    from camera_software import (
        run_calibration_validators,
        run_startup_spectral_sanity,
    )

    browser = RenderWorkBrowser()
    browser.set_startup_validation(
        run_startup_spectral_sanity((1, 3)),
        run_calibration_validators("camera-bootstrap"),
    )
    browser.sync([], [])

    keys = [spec.key for spec in browser.widget.subpanels]
    assert keys[:3] == [
        "validation-root", "validation:spectral:1", "validation:spectral:3",
    ]
    assert keys[-1] == "work-assets-root"
    assert browser.widget.selected_key == "validation-root"
    assert all(not key.startswith("job:") for key in keys)
    three_lane = next(
        spec for spec in browser.widget.subpanels
        if spec.key == "validation:spectral:3"
    )
    assert "native CPU lane precheck: passed" in three_lane.summary_lines
    assert "native GPU camera exposure: queued" in three_lane.summary_lines

    browser.set_startup_exposure(
        "spectral:3", "passed", preview_path="three-lane-camera.png"
    )
    browser.sync([], [])
    three_lane = next(
        spec for spec in browser.widget.subpanels
        if spec.key == "validation:spectral:3"
    )
    assert three_lane.payload["preview_path"] == "three-lane-camera.png"
    assert three_lane.payload["status"] == "passed"
