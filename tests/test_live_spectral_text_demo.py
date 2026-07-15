import threading
import time

import numpy as np

import live_spectral_text_demo as demo
import scene_orders as orders


def _resolved(text: str) -> dict:
    package = demo.build_paragraph_order(text, display_width=160, display_height=96)
    return orders.resolved_jobs(package, demo.JOB_ID)[0]


def test_display_raster_and_text_face_focus_are_explicit():
    job = _resolved("intricate spectral letterforms")
    runtime = orders.order_runtime_settings(job)
    metadata = orders.composition_metadata(job)

    assert (runtime["width"], runtime["height"]) == (160, 96)
    assert runtime["region"] == {"x": 320, "y": 192, "width": 160, "height": 96}
    assert metadata["full_frame"] == {"width": 800, "height": 480}
    expected = np.asarray(job["planes"][0]["normal"]) * 0.009
    assert np.allclose(job["camera"]["focus_target_m"], expected, atol=1.0e-12)
    assert np.isclose(
        metadata["camera"]["focus_distance_m"],
        orders.camera_focus_distance(job),
    )
    assert metadata["camera"]["focus_distance_m"] < np.linalg.norm(
        np.asarray(job["camera"]["target_m"]) - np.asarray(job["camera"]["position_m"])
    )


def test_render_contract_reports_native_raster_and_coordinate_only_frame():
    summary = demo.render_contract_summary(120, 100, sensor_sweeps=4)
    assert "output=120x100" in summary
    assert "native_sensor=120x120" in summary
    assert "composition_frame=600x500" in summary
    assert "crop=(240,200,120,100)" in summary
    assert "sensor_sweeps=4" in summary
    assert "coordinates only, not a rendered raster" in summary


def test_live_order_exposes_independent_sensor_refinement_sweeps():
    package = demo.build_paragraph_order(
        "refine",
        display_width=120,
        display_height=100,
        sensor_sweeps=4,
    )
    job = orders.resolved_jobs(package, demo.JOB_ID)[0]
    assert orders.order_runtime_settings(job)["sensor_sweeps"] == 4


def test_display_preview_never_enlarges_traced_pixels():
    assert demo._texture_display_rect(960, 600, 960, 600) == (0, 0, 960, 600)
    assert demo._texture_display_rect(960, 600, 1920, 1200) == (480, 300, 960, 600)
    assert demo._texture_display_rect(960, 600, 480, 300) == (0, 0, 480, 300)


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

    def render(sequence, text, order):
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
