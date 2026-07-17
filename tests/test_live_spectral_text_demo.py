import threading
import time
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import live_spectral_text_demo as demo
import scene_orders as orders
from camera_software.progressive_exposure import (
    ExposureProgressEvent,
    ExposureProgressKind,
    SensorPixelSlice,
    SensorRegion,
)


def _resolved(text: str) -> dict:
    package = demo.build_paragraph_order(text, display_width=160, display_height=96)
    return orders.resolved_jobs(package, demo.JOB_ID)[0]


def test_live_program_defaults_to_four_complete_sensor_sweeps():
    args = demo._args([])
    assert args.sensor_sweeps == 4
    package = demo.build_paragraph_order("default sweep count")
    job = orders.resolved_jobs(package, demo.JOB_ID)[0]
    assert orders.order_runtime_settings(job)["sensor_sweeps"] == 4


def test_live_cli_accepts_reusable_raytrained_priority_model():
    args = demo._args(["--priority-model", "learned.npz"])
    assert args.priority_model == "learned.npz"


def test_live_renderer_never_trains_from_orthographic_preview():
    source = inspect.getsource(demo.make_subprocess_renderer)
    assert "train_scene_priority_network" not in source
    assert "SPECTRAL_PRIORITY_TRAINING_STEPS" not in source


def test_display_raster_and_text_face_focus_are_explicit():
    job = _resolved("intricate spectral letterforms")
    runtime = orders.order_runtime_settings(job)
    metadata = orders.composition_metadata(job)

    assert (runtime["width"], runtime["height"]) == (160, 96)
    assert runtime["region"] == {"x": 320, "y": 192, "width": 160, "height": 96}
    assert metadata["full_frame"] == {"width": 800, "height": 480}
    expected = (
        np.asarray(job["planes"][0]["normal"])
        * orders.resolved_glyph_depth(job)
        * (1.0 - float(job["geometry"]["embed_fraction"]))
    )
    assert np.allclose(job["camera"]["focus_target_m"], expected, atol=1.0e-12)
    assert np.isclose(
        metadata["camera"]["focus_distance_m"],
        orders.camera_focus_distance(job),
    )
    assert metadata["camera"]["focus_distance_m"] < np.linalg.norm(
        np.asarray(job["camera"]["target_m"]) - np.asarray(job["camera"]["position_m"])
    )


def test_live_font_depth_tracks_formatter_resolved_height_and_mesh_is_detailed():
    short = _resolved("Hi")
    long = _resolved(
        "A substantially longer paragraph forces the formatter to reduce its "
        "physical line height while preserving the authored extrusion ratio."
    )

    ratio = demo.DEFAULT_EXTRUSION_DEPTH_RATIO
    for job in (short, long):
        assert np.isclose(
            orders.resolved_glyph_depth(job) / orders.resolved_glyph_height(job),
            ratio,
        )
        assert job["geometry"]["outline_subdivisions"] == 4
        assert job["geometry"]["cap_grid"] == 64
    assert orders.resolved_glyph_height(long) < orders.resolved_glyph_height(short)
    assert orders.resolved_glyph_depth(long) < orders.resolved_glyph_depth(short)


def test_render_contract_reports_native_raster_and_coordinate_only_frame():
    summary = demo.render_contract_summary(
        120, 100, sensor_sweeps=4, scene_width=960, scene_height=600
    )
    assert "ui_scene=960x600" in summary
    assert "scan_region<=120x100" in summary
    assert "native_sensor=960x960" in summary
    assert "composition_frame=4800x3000" in summary
    assert "authored_sensor_sweeps=4" in summary
    assert "live_exposure=continuous" in summary
    assert "requested regions and pixel slices accumulate at their scene coordinates" in summary


def test_progress_preview_uses_presentation_only_auto_exposure_source():
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "preview_white" in source
    assert "linear_progress[..., :3]" in source
    assert "np.save" not in source[source.index("preview_white") - 500:source.index("preview_white") + 500]


def test_live_order_exposes_independent_sensor_refinement_sweeps():
    package = demo.build_paragraph_order(
        "refine",
        display_width=120,
        display_height=100,
        sensor_sweeps=4,
    )
    job = orders.resolved_jobs(package, demo.JOB_ID)[0]
    assert orders.order_runtime_settings(job)["sensor_sweeps"] == 4


def test_self_rendering_scene_places_editor_and_controls_on_distinct_sensor_regions():
    scene = demo.build_self_rendering_program_scene(
        "editable words", display_width=200, display_height=120, revision=3
    )
    expected = demo.program_ui_sensor_regions(200, 120)

    assert {item.object_id for item in scene.objects} == set(expected)
    assert len({item.placement.center_m for item in scene.objects}) == len(scene.objects)
    for item in scene.objects:
        assert item.products[0].sensor_region == expected[item.object_id]
        assert item.revision == (
            3 if item.object_id in {"editor-text", "status-text"}
            else demo.PROGRAM_STATIC_GEOMETRY_REVISION
        )
    # Perspective is shared: no object contract contains a camera.
    assert all(not hasattr(item, "camera") for item in scene.objects)


def test_static_geometry_change_invalidates_retained_sensor_evidence():
    previous = demo.build_self_rendering_program_scene(
        "old editor", display_width=200, display_height=120, revision=1
    )
    current = demo.build_self_rendering_program_scene(
        "new editor", display_width=200, display_height=120, revision=2
    )
    assert demo._static_scene_is_reusable(previous, current)

    previous_by_id = {item.object_id: item for item in previous.objects}
    stale_panel = replace(
        previous_by_id["camera-panel"],
        primitive=demo.DisplayPrimitive(
            demo.DisplayPrimitiveKind.BOX, label="CAMERA"
        ),
        revision=1,
    )
    stale = replace(
        previous,
        objects=tuple(
            stale_panel if item.object_id == "camera-panel" else item
            for item in previous.objects
        ),
    )
    assert not demo._static_scene_is_reusable(stale, current)


def test_ui_products_become_gpu_uv_requests_inside_photographed_region():
    scene = demo.build_self_rendering_program_scene(
        "work allocation", display_width=200, display_height=120
    )
    control = demo.build_ui_next_scan_control(
        scene,
        demo.program_display_region(200, 120),
        sequence=7,
        targeted_fraction=0.75,
    )

    assert control["sequence"] == 7
    assert control["targeted_fraction"] == 0.75
    assert len(control["uv_requests"]) == len(scene.objects)
    assert len(control["pixel_slice_requests"]) == 2
    assert set(control["metadata"]["object_ids"]) == {
        item.object_id for item in scene.objects
    }
    for request in control["uv_requests"]:
        u0, v0, u1, v1 = request["uv_bounds"]
        assert 0.0 <= u0 < u1 <= 1.0
        assert 0.0 <= v0 < v1 <= 1.0
        assert request["target_level"] >= 2


def test_ui_regions_are_split_into_bounded_scan_chunks_without_resizing_scene():
    scene = demo.build_self_rendering_program_scene(
        "chunked work", display_width=960, display_height=600
    )
    photographed = demo.program_display_region(960, 600)
    control = demo.build_ui_next_scan_control(
        scene,
        photographed,
        sequence=1,
        targeted_fraction=0.75,
        scan_width=100,
        scan_height=100,
    )

    assert scene.sensor_width == 960 * demo.SENSOR_CROP_SCALE
    assert scene.sensor_height == 600 * demo.SENSOR_CROP_SCALE
    assert len(control["uv_requests"]) > len(scene.objects)
    for request in control["uv_requests"]:
        u0, v0, u1, v1 = request["uv_bounds"]
        assert (u1 - u0) * photographed.width <= 100.0 + 1.0e-9
        assert (v1 - v0) * photographed.height <= 100.0 + 1.0e-9


def test_delta_blend_changes_only_arbitrary_dirty_sites():
    current = np.full((3, 4, 3), 0.8, np.float32)
    current_weight = np.full((3, 4), 2.0, np.float32)
    previous_sum = np.full((3, 4, 3), 0.2, np.float32)
    previous_weight = np.ones((3, 4), np.float32)
    selected = SensorPixelSlice(4, 3, (0, 6, 11))

    blended = demo._blend_delta_exposure(
        current, current_weight, previous_sum, previous_weight, selected
    )

    assert np.allclose(blended, current)
    current_weight[:] = 0.0
    blended = demo._blend_delta_exposure(
        current, current_weight, previous_sum, previous_weight, selected
    )
    assert np.allclose(blended.reshape(-1, 3)[[0, 6, 11]], 0.2)
    assert np.allclose(blended.reshape(-1, 3)[[1, 2, 3]], 0.8)


def test_self_rendering_order_has_one_camera_and_all_ui_text_geometry():
    scene = demo.build_self_rendering_program_scene(
        "same camera", display_width=160, display_height=96
    )
    job = orders.resolved_jobs(
        demo.build_display_scene_order(scene), demo.JOB_ID
    )[0]

    assert [item["id"] for item in job["objects"]] == [
        "editor-text",
        "camera-label",
        "work-label",
        "status-text",
    ]
    assert {
        item["id"]: item["token"] for item in job["objects"]
    } == {
        "editor-text": "same camera",
        "camera-label": "CAMERA",
        "work-label": "WORK VALUE",
        "status-text": "SPECTRAL EXPOSURE ACTIVE",
    }
    plane_ids = {item["id"] for item in job["planes"]}
    assert "display-surface-editor-text" in plane_ids
    assert "display-surface-camera-label" in plane_ids
    assert "display-surface-work-label" in plane_ids
    assert "display-surface-status-text" in plane_ids
    assert "display-surface-window-minimize" in plane_ids
    assert "display-icon-window-minimize-bar" in plane_ids
    assert "display-icon-window-close-forward" in plane_ids
    assert "display-icon-window-close-backward" in plane_ids
    assert job["camera"]["position_m"] == list(scene.camera.position_m)
    assert job["camera"]["target_m"] == list(scene.camera.target_m)
    assert orders.order_runtime_settings(job)["region"] == {
        "x": 320, "y": 192, "width": 160, "height": 96,
    }


def test_display_preview_never_enlarges_traced_pixels():
    assert demo._texture_display_rect(960, 600, 960, 600) == (0, 0, 960, 600)
    assert demo._texture_display_rect(960, 600, 1920, 1200) == (480, 300, 960, 600)
    assert demo._texture_display_rect(960, 600, 480, 300) == (0, 0, 480, 300)


def test_sensor_product_maps_into_shared_progressive_texture():
    photographed = demo.program_display_region(200, 120)
    products = demo.program_ui_sensor_regions(200, 120)

    editor_rect = demo._sensor_product_texture_rect(
        products["editor-text"], photographed, 200, 120
    )
    close_rect = demo._sensor_product_texture_rect(
        products["window-close"], photographed, 200, 120
    )
    assert editor_rect[0] == 0
    assert editor_rect[2] == 200
    assert editor_rect[1] > products["camera-panel"].y - photographed.y
    assert editor_rect[1] + editor_rect[3] <= photographed.height
    assert close_rect[0] + close_rect[2] <= photographed.width
    assert close_rect[1] < products["camera-label"].y - photographed.y


def test_sensor_product_maps_to_exact_ui_view_destination():
    photographed = demo.program_display_region(200, 120)
    product = demo.SensorRegion(
        photographed.x + 100, photographed.y + 10, 50, 20
    )

    assert demo._sensor_product_window_rect(
        product, photographed, 400, 240
    ) == (200, 20, 100, 40)


def test_hud_extends_canvas_and_scales_for_small_displays():
    large_hud = demo._hud_layout(960)
    small_hud = demo._hud_layout(80)
    assert small_hud["height"] < large_hud["height"]
    assert small_hud["editor_font_px"] < large_hud["editor_font_px"]
    assert small_hud["status_font_px"] < large_hud["status_font_px"]

    canvas, hud = demo._window_regions(80, 48 + small_hud["height"])
    assert canvas == (0, 0, 80, 48)
    assert hud == (0, 48, 80, small_hud["height"])
    assert canvas[1] + canvas[3] == hud[1]


@pytest.mark.parametrize(
    ("work_width", "work_height"),
    [(24, 24), (64, 48), (100, 100), (320, 180), (960, 600)],
)
def test_work_size_derives_scaling_program_frame(work_width, work_height):
    width, height = demo._initial_window_size(work_width, work_height)
    crop = demo.program_display_region(width, height)
    regions = demo.program_ui_sensor_regions(
        width, height, work_width=work_width, work_height=work_height
    )

    assert width >= 2 * work_width
    assert height > work_height
    assert regions["program-backdrop"] == crop
    assert regions["camera-panel"].width == work_width
    assert regions["camera-panel"].height == work_height
    assert regions["work-panel"].width == work_width
    assert regions["work-panel"].height == work_height
    assert regions["work-panel"].x == (
        regions["camera-panel"].x + regions["camera-panel"].width
    )
    assert regions["camera-label"].y < regions["camera-panel"].y
    assert regions["work-label"].y < regions["work-panel"].y
    assert (
        regions["camera-panel"].y
        >= regions["camera-label"].y
        + regions["camera-label"].height
        + max(1, regions["camera-label"].height // 2)
    )
    assert regions["editor-text"].y == (
        regions["camera-panel"].y + regions["camera-panel"].height
    )
    assert regions["status-text"].y == (
        regions["editor-text"].y + regions["editor-text"].height
    )
    assert regions["status-text"].y + regions["status-text"].height == (
        crop.y + crop.height
    )
    for region in regions.values():
        assert region.x >= crop.x
        assert region.y >= crop.y
        assert region.x + region.width <= crop.x + crop.width
        assert region.y + region.height <= crop.y + crop.height


def test_program_world_layout_matches_square_native_camera_frame():
    scene_width, scene_height = demo._initial_window_size(100, 100)
    crop = demo.program_display_region(scene_width, scene_height)
    full = demo._placement_for_sensor_region(
        crop, crop, thickness_m=0.01
    )

    assert np.isclose(full.size_m[0], demo.PROGRAM_SCENE_EXTENT_M)
    assert np.isclose(full.size_m[1], demo.PROGRAM_SCENE_EXTENT_M)


def test_work_sized_scene_has_exact_camera_and_work_sensor_products():
    scene_width, scene_height = demo._initial_window_size(100, 100)
    crop = demo.program_display_region(scene_width, scene_height)
    regions = demo.program_ui_sensor_regions(
        scene_width,
        scene_height,
        work_width=100,
        work_height=100,
    )
    scene = demo.build_self_rendering_program_scene(
        "one camera renders this whole UI",
        display_width=scene_width,
        display_height=scene_height,
        work_width=100,
        work_height=100,
    )

    assert crop.width == scene_width
    assert crop.height == scene_height
    assert regions["program-backdrop"] == crop
    assert regions["camera-panel"].x == crop.x
    assert regions["camera-panel"].width == 100
    assert regions["camera-panel"].height == 100
    assert regions["work-panel"].x == (
        regions["camera-panel"].x + regions["camera-panel"].width
    )
    assert regions["work-panel"].width == 100
    assert regions["work-panel"].height == 100
    assert regions["camera-panel"].y == regions["work-panel"].y
    assert regions["camera-label"].y < regions["camera-panel"].y
    assert regions["work-label"].y < regions["work-panel"].y
    assert regions["window-close"].y < regions["camera-label"].y
    assert regions["editor-text"].x == crop.x
    assert regions["editor-text"].y == (
        regions["camera-panel"].y + regions["camera-panel"].height
    )
    assert regions["editor-text"].width == crop.width
    assert regions["status-text"].x == crop.x
    assert regions["status-text"].y == (
        regions["editor-text"].y + regions["editor-text"].height
    )
    assert regions["status-text"].width == crop.width
    assert (
        regions["status-text"].y + regions["status-text"].height
        == crop.y + crop.height
    )
    for region in regions.values():
        assert region.x >= crop.x
        assert region.y >= crop.y
        assert region.x + region.width <= crop.x + crop.width
        assert region.y + region.height <= crop.y + crop.height
    assert scene.sensor_width == scene_width * demo.SENSOR_CROP_SCALE
    assert scene.sensor_height == scene_height * demo.SENSOR_CROP_SCALE
    work_label = regions["work-label"]
    for object_id in (
        "window-minimize", "window-maximize", "window-close",
    ):
        control_region = regions[object_id]
        assert (
            work_label.x + work_label.width <= control_region.x
            or control_region.x + control_region.width <= work_label.x
            or work_label.y + work_label.height <= control_region.y
            or control_region.y + control_region.height <= work_label.y
        )
    by_id = {item.object_id: item for item in scene.objects}
    assert by_id["editor-text"].horizontal_align == "left"
    assert by_id["editor-text"].vertical_align == "top"
    assert by_id["status-text"].horizontal_align == "left"
    assert by_id["status-text"].vertical_align == "bottom"
    assert by_id["camera-panel"].placement.thickness_m <= 0.002
    assert by_id["camera-label"].placement.thickness_m <= 0.001
    assert by_id["program-backdrop"].products[0].sensor_region == crop
    assert by_id["program-backdrop"].placement.size_m == (
        demo.PROGRAM_SCENE_EXTENT_M,
        demo.PROGRAM_SCENE_EXTENT_M,
    )
    job = orders.resolved_jobs(
        demo.build_display_scene_order(scene, sensor_sweeps=1), demo.JOB_ID
    )[0]
    assert orders.order_runtime_settings(job)["region"] == {
        "x": 2 * scene_width,
        "y": 2 * scene_height,
        "width": scene_width,
        "height": scene_height,
    }
    jobs_by_id = {
        item["id"]: item for item in job["objects"]
    }
    assert jobs_by_id["editor-text"]["geometry"]["horizontal_align"] == "left"
    assert jobs_by_id["editor-text"]["geometry"]["vertical_align"] == "top"
    assert jobs_by_id["status-text"]["geometry"]["horizontal_align"] == "left"
    assert jobs_by_id["status-text"]["geometry"]["vertical_align"] == "bottom"
    control = demo.build_ui_next_scan_control(
        scene,
        crop,
        sequence=1,
        targeted_fraction=0.75,
        scan_width=100,
        scan_height=100,
    )
    for request in control["uv_requests"]:
        u0, v0, u1, v1 = request["uv_bounds"]
        assert (u1 - u0) * crop.width <= 100.0 + 1.0e-9
        assert (v1 - v0) * crop.height <= 100.0 + 1.0e-9
    assert all(
        len(request["site_indices"]) <= 100 * 100
        for request in control["pixel_slice_requests"]
    )
    assert {obj.object_id for obj in scene.objects} == {
        "camera-panel",
        "program-backdrop",
        "work-panel",
        "camera-label",
        "work-label",
        "editor-text",
        "status-text",
        "window-minimize",
        "window-maximize",
        "window-close",
    }
    assert {
        item["id"]: item["token"] for item in job["objects"]
    } == {
        "editor-text": "one camera renders this whole UI",
        "camera-label": "CAMERA",
        "work-label": "WORK VALUE",
        "status-text": "SPECTRAL EXPOSURE ACTIVE",
    }


def test_authored_focus_reaches_native_camera_and_bdpt_descriptor():
    import exposure_render_demo as exposure

    job = _resolved("focused text")
    base = exposure._build_thick_lens_lab_tracer_scene()
    scene, _ = orders.compile_job(base, job)
    previous_job = exposure._ORDERED_SCENE_JOB
    try:
        exposure._ORDERED_SCENE_JOB = job
        cam, solved = exposure._build_thick_lens_lab_camera_package(
            scene, 160, 160, exposure.DEFAULT_OPTICS
        )
        expected_sensor_distance = orders.camera_focus_distance(job)
        assert np.isclose(solved.sanity_input.focus_distance_m, expected_sensor_distance)
        assert np.isclose(solved.sanity_report.planes.focal_plane_z_m, expected_sensor_distance)

        descriptor = exposure._build_camera_sensor_descriptor(
            exposure.DEFAULT_OPTICS, cam, 160, 160, solved=solved
        )
        assert descriptor is not None
        expected_lens_distance = expected_sensor_distance - cam.focal_m
        assert np.isclose(descriptor["focus_distance_m"], expected_lens_distance)
        assert descriptor["focus_distance_m"] > 0.0
    finally:
        exposure._ORDERED_SCENE_JOB = previous_job


def test_authored_sensor_focus_is_converted_to_physical_lab_conjugate():
    import exposure_render_demo as exposure
    import thick_lens_focus_lab as lab

    scene = lab.SceneConfig()
    sensor_focus_m = 3.75
    object_distance_m = exposure._lab_object_distance_for_sensor_focus(scene, sensor_focus_m)
    design = scene.optical_design
    focal_m = design.focal_length_range_m[0] + (
        design.focal_length_range_m[1] - design.focal_length_range_m[0]
    ) * design.zoom
    group_span_m = (
        (design.group_count - 1) * design.min_air_gap_m
        + design.group_count * design.group_thickness_m
    )
    image_distance_m = 1.0 / (1.0 / focal_m - 1.0 / object_distance_m)
    reconstructed = (
        object_distance_m + group_span_m + image_distance_m + design.sensor_clearance_m
    )

    assert object_distance_m < sensor_focus_m
    assert np.isclose(reconstructed, sensor_focus_m, atol=1.0e-12)


def test_physical_lab_focus_probe_corrects_finite_lens_error():
    import exposure_render_demo as exposure
    import thick_lens_focus_lab as lab

    sensor_focus_m = 3.75
    scene = exposure._ordered_lab_scene_for_sensor_focus(lab, sensor_focus_m)
    lab._scene_lenses(scene)
    physical_sensor_focus_m = scene.image_plate.x - scene.object_plane.x

    assert np.isclose(physical_sensor_focus_m, sensor_focus_m, atol=2.0e-6)


def test_paragraph_wraps_and_centers_inside_physical_text_box():
    job = _resolved(
        "A physically rendered paragraph wraps its words and remains centered "
        "inside the authored surface without touching its boundary."
    )
    layout = orders.layout_paragraph(job)
    contours = orders._paragraph_contours(job)
    points = np.concatenate([contour[:-1] for contour in contours], axis=0)
    box = np.asarray(layout["text_box_m"], np.float64)

    assert len(layout["lines"]) >= 2
    assert layout["block_height_m"] <= box[1] + 1.0e-10
    assert points[:, 0].min() >= -0.5 * box[0] - 1.0e-10
    assert points[:, 0].max() <= +0.5 * box[0] + 1.0e-10
    assert points[:, 1].min() >= -0.5 * box[1] - 1.0e-10
    assert points[:, 1].max() <= +0.5 * box[1] + 1.0e-10
    assert abs(float(points[:, 1].min() + points[:, 1].max())) < 0.02


def test_long_unbroken_word_is_hard_wrapped_and_fitted():
    job = _resolved("spectrophotometrically" * 5)
    layout = orders.layout_paragraph(job)

    assert len(layout["lines"]) > 1
    assert "".join(layout["lines"]) == job["token"]
    assert layout["block_height_m"] <= layout["text_box_m"][1] + 1.0e-10


def test_render_worker_keeps_inflight_revision_and_coalesces_pending_edits():
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[tuple[int, str]] = []

    def render(sequence, text, order, next_scan_control, progress_sink):
        calls.append((sequence, text))
        if sequence == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=f"{sequence}.png",
            linear_path=f"{sequence}.npy",
            manifest_path=f"{sequence}.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(render)
    try:
        worker.submit("one", {"value": 1})
        assert first_started.wait(timeout=2.0)
        worker.submit("two", {"value": 2})
        worker.submit("three", {"value": 3})
        release_first.set()

        deadline = time.monotonic() + 2.0
        latest = None
        while time.monotonic() < deadline:
            latest, busy, error, submitted = worker.snapshot()
            if latest is not None and latest.text == "three" and not busy:
                break
            time.sleep(0.01)
        assert error == ""
        assert submitted == 3
        assert latest is not None and latest.sequence == 3
        assert calls == [(1, "one"), (3, "three")]
    finally:
        release_first.set()
        worker.close()


def test_render_worker_drops_superseded_progress_and_resets_between_submissions():
    first_started = threading.Event()
    release_first = threading.Event()

    def event(sequence: int, event_sequence: int) -> ExposureProgressEvent:
        return ExposureProgressEvent(
            exposure_id=f"revision-{sequence:04d}",
            sequence=event_sequence,
            kind=ExposureProgressKind.PASS_AVAILABLE,
            region=SensorRegion(x=0, y=0, width=8, height=8),
            pass_index=event_sequence,
            completed_work=event_sequence,
            total_work=2,
        )

    def render(sequence, text, order, next_scan_control, progress_sink):
        if sequence == 1:
            progress_sink(event(1, 1))
            first_started.set()
            assert release_first.wait(timeout=2.0)
            progress_sink(event(1, 2))
        else:
            progress_sink(event(sequence, 1))
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=f"{sequence}.png",
            linear_path=f"{sequence}.npy",
            manifest_path=f"{sequence}.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(render)
    try:
        worker.submit("one", {})
        assert first_started.wait(timeout=2.0)
        latest, _ = worker.progress_snapshot()
        assert latest is not None and latest.exposure_id == "revision-0001"

        worker.submit("two", {})
        latest, layers = worker.progress_snapshot()
        assert latest is None
        assert layers == {}
        release_first.set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            latest, _ = worker.progress_snapshot()
            rendered, busy, error, _ = worker.snapshot()
            if latest is not None and rendered is not None and not busy:
                break
            time.sleep(0.01)
        assert error == ""
        assert latest is not None and latest.exposure_id == "revision-0002"
        assert latest.sequence == 1
    finally:
        release_first.set()
        worker.close()


def test_render_worker_forwards_current_progress_to_inventory_observer():
    observed = []

    def render(sequence, text, order, next_scan_control, progress_sink):
        progress_sink(ExposureProgressEvent(
            exposure_id="shared-scene",
            sequence=1,
            kind=ExposureProgressKind.LAYER_AVAILABLE,
            region=SensorRegion(0, 0, 8, 8),
            pass_index=1,
        ))
        return demo.RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path="image.png",
            linear_path="linear.npy",
            manifest_path="manifest.json",
            elapsed_s=0.01,
        )

    worker = demo.SpectralTextRenderWorker(render, progress_observer=observed.append)
    try:
        worker.submit("one", {})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            latest, busy, error, _ = worker.snapshot()
            if latest is not None and not busy:
                break
            time.sleep(0.01)
        assert error == ""
        assert len(observed) == 1
        assert observed[0].exposure_id == "shared-scene"
    finally:
        worker.close()
