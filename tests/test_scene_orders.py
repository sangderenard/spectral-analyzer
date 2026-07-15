import numpy as np

import scene_orders as orders


ORDER = "configs/scene_orders/glyph_a_red_gloss.json"


def test_example_order_resolves_and_consumes_runtime_settings():
    package = orders.load_order(ORDER)
    job = orders.resolved_jobs(package, "glyph_A")[0]
    runtime = orders.order_runtime_settings(job)

    assert job["token"] == "A"
    assert runtime["width"] == 64 and runtime["height"] == 64
    assert runtime["focal_mm"] == 35.0
    assert runtime["aperture_mm"] == 25.0


def test_circular_glyph_is_solid_non_degenerate_and_half_embedded():
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    plane = job["planes"][0]
    tri = orders._glyph_triangles(job, plane)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    assert tri.shape[0] > 500
    assert np.all(area2 > 1.0e-14)
    assert np.isclose(tri[..., 0].min(), -0.016)
    assert np.isclose(tri[..., 0].max(), +0.016)
    # Circular profile bulges laterally between its two planar caps.
    front_back = np.isclose(np.abs(tri[..., 0]), 0.016)
    cap_extent = np.abs(tri[..., 1][front_back]).max()
    assert np.abs(tri[..., 1]).max() > cap_extent


def test_compiler_replaces_subject_emitters_but_keeps_rig_lighting():
    import exposure_render_demo as exposure

    base = exposure._build_thick_lens_lab_tracer_scene()
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    scene, report = orders.compile_job(base, job)

    # The lab's default orbiter subject carries in-frame emissive demo spheres
    # tagged as both object and source triangles.  A scene order replaces the
    # whole subject, so those emitters must be gone from the compiled scene…
    base_objects = np.asarray(base.camera_tri_groups["object"], np.int64)
    base_sources = np.asarray(base.src_tri_idx, np.int64)
    subject_emitters = np.intersect1d(base_objects, base_sources)
    assert subject_emitters.size > 0
    assert report.removed_subject_emitters == int(subject_emitters.size)
    assert scene.src_tri_idx.size == base.src_tri_idx.size - subject_emitters.size
    # …while photographic rig lighting (ring light, side-room flash) and its
    # authored power are fully retained.
    keep = ~np.isin(base_sources, subject_emitters)
    assert np.isclose(
        float(np.sum(np.asarray(scene.src_emit_W))),
        float(np.sum(np.asarray(base.src_emit_W)[keep])),
    )
    assert float(np.sum(np.asarray(scene.src_emit_W))) > 0.0
    # No compiled emitter may live in the authored subject group.
    assert np.intersect1d(
        np.asarray(scene.src_tri_idx, np.int64),
        np.asarray(scene.camera_tri_groups["object"], np.int64),
    ).size == 0
    assert scene.camera_tri_groups["sensor"].size == base.camera_tri_groups["sensor"].size
    assert scene.camera_tri_groups["aperture_blocker"].size == base.camera_tri_groups["aperture_blocker"].size
    assert scene.camera_tri_groups["thin_lens"].size == base.camera_tri_groups["thin_lens"].size
    assert report.glyph_triangles > 500
    assert report.plane_triangles == 12
    mats = scene.mat_buf.reshape(scene.mat_n_mats, 32, 12)
    matte, red = mats[-2], mats[-1]
    assert np.all(matte[:8, 2] > 0.0)
    assert np.all(red[:8, 2] > 0.0)
    # Renderer band order is long/red wavelength to short/blue wavelength.
    # The baked curve is a display-transfer metamer: long-wavelength bands
    # must dominate and the round trip through band_to_display_rgb weights
    # must reproduce the authored red-dominant colour.
    assert red[0, 2] == np.max(red[:8, 2])
    assert red[0, 2] > 20.0 * np.max(red[3:8, 2])
    display = orders._display_rgb_weights(np.linspace(700.0, 400.0, 8)).T @ red[:8, 2]
    assert display[0] > 10.0 * display[1]
    assert display[0] > 10.0 * display[2]


def test_multi_glyph_token_renders_as_laid_out_text():
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    job = dict(job)
    job["token"] = "AXE"
    plane = job["planes"][0]
    tri = orders._glyph_triangles(job, plane)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    assert np.all(area2 > 1.0e-14)
    # Three glyphs laid out by real font metrics: wider than tall, and wider
    # than any single glyph of the same cap height.
    height = float(job.get("geometry", {}).get("height_m", 0.2))
    width = float(tri[..., 1].max() - tri[..., 1].min())
    assert np.isclose(float(tri[..., 2].max() - tri[..., 2].min()), height, rtol=0.35)
    assert width > 1.5 * height


def test_empty_token_is_rejected():
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    job = dict(job)
    job["token"] = "   "
    try:
        orders._validate_resolved_job(job)
    except ValueError as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("blank token must be rejected")


def test_spatial_camera_maps_world_subject_into_stationary_optical_rig():
    import exposure_render_demo as exposure

    base = exposure._build_thick_lens_lab_tracer_scene()
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    job = dict(job)
    job["camera"] = dict(job["camera"], position_m=[2.0, -1.0, 0.5],
                         target_m=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
    scene, report = orders.compile_job(base, job)

    sensor = np.asarray(scene.camera_tri_groups["sensor"], np.int64)
    sensor_center = np.asarray(scene.verts)[sensor].reshape(-1, 3).mean(axis=0)
    base_sensor = np.asarray(base.camera_tri_groups["sensor"], np.int64)
    base_sensor_center = np.asarray(base.verts)[base_sensor].reshape(-1, 3).mean(axis=0)
    assert np.allclose(sensor_center, base_sensor_center, atol=1.0e-12)
    assert report.camera_position_m == (2.0, -1.0, 0.5)

    object_ids = np.asarray(scene.camera_tri_groups["object"], np.int64)
    pose = orders.camera_pose(job)
    expected_plane = orders._world_to_canonical_camera(
        orders._box_plane_triangles(job["planes"][0]), pose, base_sensor_center
    )
    actual_plane = np.asarray(scene.verts)[object_ids[:12]].reshape(-1, 3, 3)
    assert np.allclose(actual_plane, expected_plane, atol=1.0e-12)

    base_lens = np.asarray(base.camera_tri_groups["thin_lens"], np.int64)
    moved_lens = np.asarray(scene.camera_tri_groups["thin_lens"], np.int64)
    assert np.array_equal(np.asarray(scene.verts)[moved_lens], np.asarray(base.verts)[base_lens])


def test_sensor_region_maps_to_physical_tile_and_composition_coordinates():
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    job = dict(job)
    job["image"] = {
        "width": 800, "height": 600,
        "region": {"x": 600, "y": 50, "width": 100, "height": 150},
    }
    tile = orders.sensor_tile(job, sensor_w_m=0.036, sensor_h_m=0.024)
    metadata = orders.composition_metadata(job)

    assert np.isclose(tile["sensor_w_m"], 0.0045)
    assert np.isclose(tile["sensor_h_m"], 0.006)
    assert np.isclose(tile["right_offset_m"], 0.01125)
    assert np.isclose(tile["up_offset_m"], 0.007)
    assert metadata["origin"] == "top-left"
    assert metadata["full_frame"] == {"width": 800, "height": 600}
    assert metadata["region"] == {"x": 600, "y": 50, "width": 100, "height": 150}


def test_invalid_sensor_region_and_incomplete_camera_pose_are_rejected():
    job = orders.resolved_jobs(orders.load_order(ORDER), "glyph_A")[0]
    bad_region = dict(job)
    bad_region["image"] = {"width": 64, "height": 64,
                           "region": {"x": 60, "y": 0, "width": 8, "height": 8}}
    bad_camera = dict(job)
    bad_camera["camera"] = dict(job["camera"], position_m=[1.0, 0.0, 0.0])
    for candidate in (bad_region, bad_camera):
        try:
            orders._validate_resolved_job(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid spatial order must be rejected")


def test_native_square_accumulator_returns_exact_rectangular_tile_shape_and_dtype():
    import exposure_render_demo as exposure

    raw = np.arange(16 * 16 * 3, dtype=np.float64).reshape(16, 16, 3)
    tile = exposure._native_sensor_tile_to_display(raw, width=12, height=7)
    assert tile.shape == (7, 12, 3)
    assert tile.dtype == raw.dtype
    assert exposure._native_sensor_schedule_shape(width=12, height=7) == (12, 12)


def test_native_cpp_linear_output_is_not_the_display_curve():
    import exposure_render_demo as exposure

    display = np.full((7, 12, 3), 0.75, dtype=np.float32)
    raw = np.arange(display.size, dtype=np.float32).reshape(display.shape)
    backend = type("Backend", (), {"_native_sensor_linear_image": raw})()
    linear = exposure._native_sensor_linear_output(backend, display)

    assert linear.dtype == raw.dtype
    assert np.array_equal(linear, raw)
    assert not np.shares_memory(linear, raw)
    assert not np.array_equal(linear, display)
