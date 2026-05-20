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
