import numpy as np

from camera_designer.camera_preset import simple_doublet_preset
from camera_designer.compound_optics import CompoundLens, RayBundle, TerminationReason
from camera_designer.lens_assembly import LensAssemblySpec


def test_compound_lens_batch_transfer_matches_scalar_trace():
    lens = CompoundLens.from_preset(simple_doublet_preset())
    origins = np.array(
        [
            [0.0, 0.0000, 0.0],
            [0.0, 0.0008, 0.0],
            [0.0, 0.0016, 0.0],
        ],
        dtype=np.float64,
    )
    directions = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (3, 1))

    batch = lens.evaluate_bundle(RayBundle(origins, directions))

    assert batch.status.tolist() == [TerminationReason.PASSED.value] * 3
    for i in range(origins.shape[0]):
        scalar = lens.trace(origins[i], directions[i])
        assert batch.status[i] == scalar.reason.value
        assert np.allclose(batch.origins[i], scalar.intercepts[-1])
        assert np.allclose(batch.directions[i], scalar.direction)


def test_lens_assembly_transfer_uses_installed_optics():
    lens = CompoundLens.from_preset(simple_doublet_preset())
    assembly = LensAssemblySpec()
    assembly.set_optics(lens, mode=LensAssemblySpec.MODE_PARAMETRIC)

    origins = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    result = assembly.evaluate_transfer(RayBundle(origins, directions))
    payload = assembly.build_parametric_payload()

    assert assembly.mode == LensAssemblySpec.MODE_PARAMETRIC
    assert result.status[0] == TerminationReason.PASSED.value
    assert int(payload[1]) == len(lens.elements)


def test_side_bundle_sampling_and_failure_short_circuit():
    lens = CompoundLens.from_preset(simple_doublet_preset())

    cone = lens.side_cone("front")
    bundle = lens.sample_side_bundle(
        "front",
        n_spatial=4,
        n_directions=5,
        wavelengths_um=[0.50, 0.60],
    )

    assert cone.side == "front"
    assert cone.half_angle_rad > 0.0
    assert bundle.origins.shape == (40, 3)
    assert bundle.directions.shape == (40, 3)
    assert bundle.wavelengths.shape == (40,)

    filtered, result, mask = lens.drop_terminated(bundle)
    assert mask.shape == (40,)
    assert filtered.origins.shape[0] == int(np.count_nonzero(mask))
    assert np.all(result.status[mask] == TerminationReason.PASSED.value)


def test_back_side_bundle_uses_reverse_axis():
    lens = CompoundLens.from_preset(simple_doublet_preset())
    bundle = lens.sample_side_bundle("back", n_spatial=1, n_directions=1)
    result = lens.evaluate_bundle(bundle)

    assert bundle.directions[0, 0] < 0.0
    assert result.status[0] == TerminationReason.PASSED.value


def test_compound_lens_builds_transfer_lut_payload():
    lens = CompoundLens.from_preset(simple_doublet_preset())
    payload, n_src = lens.build_transfer_lut(n_u=8, n_v=8, n_directions=4)

    assert payload.dtype == np.float32
    assert payload[0] == np.float32(14950.0)
    assert int(payload[1]) == 8
    assert int(payload[2]) == 8
    assert int(payload[3]) == 2
    assert int(payload[4]) == 2
    assert n_src > 0
    cells = payload[16:].reshape(8, 8, 2, 2, 7)
    assert np.count_nonzero(cells[..., 4]) > 0


def test_lens_assembly_lut_bake_uses_parametric_compound_lens():
    class FakeEndpoint:
        def __init__(self):
            self.preset = simple_doublet_preset()

    assembly = LensAssemblySpec()
    assembly.bake_lut(FakeEndpoint(), tracer=None, n_rays=64, n_grid=8, verbose=False)

    assert assembly.mode == LensAssemblySpec.MODE_LUT
    assert assembly.optics is not None
    assert assembly._transfer_grid is not None
    assert assembly._baked_ep is None
    assert assembly._transfer_grid[0] == np.float32(14950.0)
    assert assembly._transfer_grid_noodles > 0


def test_vignetting_profile_declares_face_cones_from_either_side():
    lens = CompoundLens.from_preset(simple_doublet_preset())

    front = lens.vignetting_profile_from_point((0.0, 0.0, 0.0), side="front")
    back = lens.vignetting_profile_from_point((0.08, 0.0, 0.0), side="back")

    assert len(front.faces) == len(lens.registered_faces())
    assert front.limiting_face is not None
    assert front.cutoff_half_angle_rad > 0.0
    assert front.limiting_face.q_matrix.shape == (3, 3)
    assert np.allclose(front.limiting_face.center, front.limiting_face.face.center)

    assert back.limiting_face is not None
    assert back.limiting_face.center_direction[0] < 0.0
    assert back.cutoff_half_angle_rad > 0.0


def test_lens_assembly_profiles_point_pair():
    lens = CompoundLens.from_preset(simple_doublet_preset())
    assembly = LensAssemblySpec()
    assembly.set_optics(lens, mode=LensAssemblySpec.MODE_PARAMETRIC)

    profiles = assembly.profile_field_pair(
        object_point=(0.0, 0.0, 0.0),
        image_point=(0.08, 0.0, 0.0),
    )

    assert set(profiles) == {"front", "back"}
    assert profiles["front"].limiting_face is not None
    assert profiles["back"].limiting_face is not None
    assert profiles["front"].boundary is not None
    assert profiles["back"].boundary is not None


def test_boundary_teleport_profiles_describe_compound_interfaces():
    lens = CompoundLens.from_preset(simple_doublet_preset())
    profiles = lens.boundary_teleport_profiles(n_azimuth=8, verify=True)

    front = profiles["front"]
    back = profiles["back"]

    assert front.source_side == "front"
    assert front.target_side == "back"
    assert back.source_side == "back"
    assert back.target_side == "front"
    assert front.q_matrix.shape == (3, 3)
    assert front.edge_to_edge_half_angle_rad >= front.target_cone_half_angle_rad
    assert back.edge_to_edge_half_angle_rad >= back.target_cone_half_angle_rad
    assert 0.0 <= front.verified_transmission_fraction <= 1.0
    assert 0.0 <= back.verified_transmission_fraction <= 1.0


def test_thick_lens_scene_builds_matching_parametric_faces():
    from thick_lens_focus_lab import LensConfig, SceneConfig, _compound_lens_from_scene

    scene = SceneConfig()
    scene.optical_design = None
    scene.lens_stack = [
        LensConfig(
            center_x=0.5,
            thickness=0.04,
            aperture_radius=0.025,
            radius_front=0.08,
            radius_back=0.09,
            ior=1.55,
        )
    ]
    scene.exit_pupil_x = 0.7
    scene.exit_pupil_radius = 0.012

    lens = _compound_lens_from_scene(scene)
    faces = lens.registered_faces()

    assert len(faces) == 3
    assert np.isclose(faces[0].x_pos, scene.lens_stack[0].x_front)
    assert np.isclose(faces[1].x_pos, scene.lens_stack[0].x_back)
    assert np.isclose(faces[2].x_pos, scene.exit_pupil_x)
    assert np.isclose(faces[2].radius, scene.exit_pupil_radius)
