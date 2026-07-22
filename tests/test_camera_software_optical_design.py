import numpy as np

from camera_software import (
    OpticalDesignSpec,
    effective_focal_length,
    image_distance_for_object,
    paraxial_system_matrix,
    solve_four_group_zoom_surrogate,
)


def test_four_group_solver_returns_ordered_controllable_train():
    spec = OpticalDesignSpec(
        focal_length_range_m=(0.060, 0.140),
        zoom=0.55,
        focus_distance_m=1.5,
        f_number=2.8,
        entrance_x_m=0.9,
        sensor_x_m=2.4,
    )
    solved = solve_four_group_zoom_surrogate(spec)

    xs = [g.x_m for g in solved.groups]
    assert len(solved.groups) == 4
    assert xs == sorted(xs)
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert solved.aperture_radius_m > 0.0
    # A compound rear train may form a virtual exit pupil beyond the sensor.
    # Backward transport carries that finite virtual target explicitly.
    assert np.isfinite(solved.exit_pupil_x_m)
    assert solved.exit_pupil_radius_m > 0.0
    assert np.isclose(xs[0], spec.entrance_x_m)
    assert xs[-1] <= spec.sensor_x_m - spec.sensor_clearance_m + 1.0e-9
    assert solved.assembly_front_x_m == spec.entrance_x_m
    assert solved.assembly_back_x_m == spec.sensor_x_m
    assert abs(solved.effective_focal_length_m - spec.target_focal_length_m) < 5.0e-4
    assert np.isfinite(solved.sensor_error_m)
    assert abs(solved.sensor_error_m) < 0.15
    assert all((b.x_m - 0.5 * b.thickness_m) - (a.x_m + 0.5 * a.thickness_m) >= spec.min_air_gap_m - 1.0e-9
               for a, b in zip(solved.groups, solved.groups[1:]))

    m = paraxial_system_matrix(solved.groups)
    assert np.allclose(m, solved.matrix)
    assert np.isfinite(effective_focal_length(m))


def test_four_group_solver_can_apply_to_thick_lens_scene_and_parametric_chain():
    from thick_lens_focus_lab import SceneConfig, _compound_lens_from_scene, _scene_lenses

    scene = SceneConfig()
    scene.optical_design = OpticalDesignSpec(
        focal_length_range_m=(0.070, 0.120),
        zoom=0.25,
        focus_distance_m=2.0,
        entrance_x_m=0.88,
        sensor_x_m=scene.image_plate.x,
    )

    lenses = _scene_lenses(scene)
    lens = _compound_lens_from_scene(scene)
    profiles = lens.boundary_teleport_profiles(n_azimuth=8, verify=True)

    assert len(lenses) == 4
    assert len(lens.registered_faces()) == 9
    assert profiles["front"].target_cone_half_angle_rad > 0.0
    assert 0.0 <= profiles["front"].verified_transmission_fraction <= 1.0


def test_thick_lens_scene_defaults_to_semantic_optical_design():
    from camera_software import SolvedOpticalTrain
    from thick_lens_focus_lab import SceneConfig, _compound_lens_from_scene, _scene_lenses

    scene = SceneConfig()
    lenses = _scene_lenses(scene)
    optics = _compound_lens_from_scene(scene)

    assert len(lenses) == 4
    assert isinstance(scene.optical_design, SolvedOpticalTrain)
    assert np.isclose(scene.image_plate.sensor_half_w, 0.028)
    assert np.isclose(scene.image_plate.sensor_half_h, 0.028)
    assert abs(scene.optical_design.spec.target_focal_length_m - 0.0825) < 1.0e-12
    assert np.isfinite(optics.f_eff) and optics.f_eff > 0.0
    assert scene.exit_pupil_radius > 0.0
    assert scene.aperture_model == "geometry"


def test_thick_lens_scene_preserves_wave_aperture_mode_flag():
    from thick_lens_focus_lab import SceneConfig, _scene_lenses

    scene = SceneConfig()
    scene.aperture_model = "wave3d"
    _scene_lenses(scene)

    assert scene.aperture_model == "wave3d"


def test_scene_view_fit_tracks_generated_mesh_extents():
    from thick_lens_focus_lab import SceneConfig, _auto_fit_scene_view_to_mesh

    scene = SceneConfig()
    tri_arr = np.array([
        [[-0.20, -0.31, 0.0], [-0.10, 0.33, 0.0], [-0.15, 0.0, 0.29]],
        [[2.80, 0.0, -0.36], [2.90, 0.0, 0.34], [2.85, 0.35, 0.0]],
    ], dtype=np.float64)
    _auto_fit_scene_view_to_mesh(scene, tri_arr)

    pts = tri_arr.reshape(-1, 3)
    assert scene.x_min <= float(np.min(pts[:, 0]))
    assert scene.x_max >= float(np.max(pts[:, 0]))
    assert scene.view_radius >= float(np.max(np.abs(pts[:, 1:3])))


def test_default_scene_imports_subject_without_legacy_stage():
    from material_db import MaterialDatabase
    from thick_lens_focus_lab import SceneConfig, _import_subject_scene

    scene = SceneConfig()
    db = MaterialDatabase()
    tris = []
    mats = []
    source_ids = []
    object_ids = []

    _import_subject_scene(scene, db, tris, mats, source_ids, object_ids)

    assert scene.include_legacy_stage is False
    assert len(tris) > 0
    assert len(object_ids) == len(tris)
    assert len(mats) == len(tris)
    assert len(set(mats)) > 1
    assert any(name.startswith("subject_") for name in db._order)


def test_imported_subject_preserves_depth_for_camera_view():
    from material_db import MaterialDatabase
    from thick_lens_focus_lab import SceneConfig, _import_subject_scene
    import test_basic_gl_cpp_window as subject_mod

    scene = SceneConfig()
    db = MaterialDatabase()
    tris = []
    mats = []
    source_ids = []
    object_ids = []
    _import_subject_scene(scene, db, tris, mats, source_ids, object_ids)

    subject_db, subject_idx = subject_mod.register_materials()
    verts8, *_ = subject_mod.scene_for_phase(
        subject_idx,
        scene.subject_time_s,
        scene_mode=scene.subject_scene_mode,
    )
    src = np.asarray(verts8[:, 0:3], dtype=np.float64)
    pts = np.asarray(tris, dtype=np.float64).reshape(-1, 3)

    assert scene.subject_depth_scale is None
    assert np.isclose(float(np.ptp(pts[:, 0])), float(np.ptp(src[:, 2])) * scene.subject_scale)
    assert np.isclose(float(np.ptp(pts[:, 1])), float(np.ptp(src[:, 1])) * scene.subject_scale)
    assert np.isclose(float(np.ptp(pts[:, 2])), float(np.ptp(src[:, 0])) * scene.subject_scale)


def test_default_solved_lens_mesh_radii_are_geometrically_valid():
    from thick_lens_focus_lab import SceneConfig, _lens_is_valid, _scene_lenses

    scene = SceneConfig()
    lenses = _scene_lenses(scene)

    assert len(lenses) == 4
    assert all(_lens_is_valid(lens) for lens in lenses)
    assert 0.025 <= max(lens.aperture_radius for lens in lenses) <= 0.120
    assert all((b.x_front - a.x_back) > 0.0 for a, b in zip(lenses, lenses[1:]))


def test_solver_respects_explicit_hardware_group_parameters():
    spec = OpticalDesignSpec(
        entrance_x_m=0.40,
        sensor_x_m=0.95,
        lock_focal_length=False,
        lock_group_positions=True,
        lock_group_powers=True,
        group_positions_m=(0.42, 0.51, 0.63, 0.78),
        group_powers_dpt=(14.0, -6.0, 9.0, 5.0),
        group_aperture_radii_m=(0.031, 0.028, 0.030, 0.026),
        group_thicknesses_m=(0.010, 0.011, 0.012, 0.013),
        group_radius_front_m=(1.0, -1.1, 1.2, 1.3),
        group_radius_back_m=(1.4, -1.5, 1.6, 1.7),
        group_iors=(1.50, 1.55, 1.60, 1.52),
    )
    solved = solve_four_group_zoom_surrogate(spec)

    assert np.allclose([g.x_m for g in solved.groups], spec.group_positions_m)
    assert np.allclose([g.power for g in solved.groups], spec.group_powers_dpt)
    assert np.allclose([g.aperture_radius_m for g in solved.groups], spec.group_aperture_radii_m)
    assert np.allclose([g.thickness_m for g in solved.groups], spec.group_thicknesses_m)
    assert np.allclose([g.radius_front_m for g in solved.groups], spec.group_radius_front_m)
    assert np.allclose([g.radius_back_m for g in solved.groups], spec.group_radius_back_m)
    assert np.allclose([g.ior for g in solved.groups], spec.group_iors)


def test_image_distance_for_object_is_finite_for_solution():
    spec = OpticalDesignSpec(focus_distance_m=1.2)
    solved = solve_four_group_zoom_surrogate(spec)
    obj_x = solved.groups[0].x_m - spec.focus_distance_m
    img_d = image_distance_for_object(solved.groups, obj_x)

    assert np.isfinite(img_d)
    assert img_d > 0.0


def test_scene_solves_exact_compound_focus_to_fixed_sensor_and_focal_length():
    from thick_lens_focus_lab import SceneConfig, _compound_lens_from_scene, _paraxial_image_x, _scene_lenses

    scene = SceneConfig()
    lenses = _scene_lenses(scene)

    optics = _compound_lens_from_scene(scene)
    exact_focus_x = _paraxial_image_x(optics, scene.object_plane.x)

    assert np.isfinite(exact_focus_x)
    assert abs(float(scene.image_plate.x) - float(scene.optical_design.spec.sensor_x_m)) < 1.0e-12
    assert abs(float(scene.image_plate.x) - exact_focus_x) < 5.0e-6
    assert abs(float(optics.f_eff) - scene.optical_design.spec.target_focal_length_m) < 5.0e-4
    assert all(
        (b.x_front - a.x_back) >= scene.optical_design.spec.min_air_gap_m - 1.0e-9
        for a, b in zip(lenses, lenses[1:])
    )
