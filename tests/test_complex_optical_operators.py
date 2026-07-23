from pathlib import Path

import numpy as np
import pytest

from camera_software.complex_optical_operators import (
    COMPLEX_OPTICAL_OPERATOR_SCHEMA,
    ComplexOperatorStateBlock,
    ComplexOpticalOperator,
    ComplexSourceMode,
    JonesOperator,
    PhaseSpaceJacobian,
    RigidFieldMap,
    TransverseBasis,
    basis_change,
    canonical_operator_contract,
    dielectric_interface,
    compile_planar_reflection_interface,
    estimate_compound_lens_phase_space,
    optical_phase,
)
from camera_designer.camera_preset import simple_doublet_preset
from camera_designer.compound_optics import CompoundLens
from camera_designer.emitter_profile import PolarizationMode, PolarizationState
from camera_software.optical_transport_graph import compile_compound_lens_graph


ROOT = Path(__file__).resolve().parents[1]


def test_transverse_basis_is_orthonormal_and_right_handed():
    basis = TransverseBasis.from_direction((0.2, -0.3, 1.0))
    frame = np.stack((basis.s, basis.p, basis.k))
    assert frame @ frame.T == pytest.approx(np.eye(3), abs=1.0e-12)
    assert np.cross(basis.s, basis.p) == pytest.approx(basis.k)


def test_basis_change_preserves_physical_field_and_round_trips():
    k = np.asarray((0.0, 0.0, 1.0))
    source = TransverseBasis.from_direction(k, reference=(0.0, 1.0, 0.0))
    destination = TransverseBasis(
        s=source.p, p=-source.s, k=source.k
    )
    field = np.asarray((1.2 + 0.3j, -0.2 + 0.7j))
    changed = basis_change(source, destination).apply(field)
    restored = basis_change(destination, source).apply(changed)
    assert restored == pytest.approx(field)
    assert np.vdot(changed, changed).real == pytest.approx(
        np.vdot(field, field).real
    )


@pytest.mark.parametrize("angle_deg", [0.0, 20.0, 45.0, 60.0])
def test_power_normalized_fresnel_conserves_each_polarization(angle_deg):
    angle = np.deg2rad(angle_deg)
    direction = (np.cos(angle), np.sin(angle), 0.0)
    result = dielectric_interface(direction, (-1.0, 0.0, 0.0), 1.0, 1.5)
    assert not result.total_internal_reflection
    for lane in range(2):
        reflected = abs(result.reflection.matrix[lane, lane]) ** 2
        transmitted = abs(result.transmission.matrix[lane, lane]) ** 2
        assert reflected + transmitted == pytest.approx(1.0, abs=2.0e-12)


def test_brewster_angle_zeros_p_reflection_and_tir_is_unitary():
    brewster = np.arctan2(1.5, 1.0)
    at_brewster = dielectric_interface(
        (np.cos(brewster), np.sin(brewster), 0.0),
        (-1.0, 0.0, 0.0),
        1.0,
        1.5,
    )
    assert abs(at_brewster.reflection.matrix[1, 1]) < 1.0e-12

    angle = np.deg2rad(60.0)
    tir = dielectric_interface(
        (np.cos(angle), np.sin(angle), 0.0),
        (-1.0, 0.0, 0.0),
        1.5,
        1.0,
    )
    assert tir.total_internal_reflection
    assert tir.transmission is None
    assert np.abs(np.diag(tir.reflection.matrix)) == pytest.approx((1.0, 1.0))


def test_jones_composition_and_reciprocal_reverse_are_explicit():
    retarder = JonesOperator.diagonal(1.0, 1.0j)
    rotation = JonesOperator(np.asarray(((0.0, -1.0), (1.0, 0.0))))
    field = np.asarray((1.0 + 0.0j, 0.25 - 0.5j))
    assert rotation.compose(retarder).apply(field) == pytest.approx(
        rotation.apply(retarder.apply(field))
    )
    assert rotation.reciprocal_reverse().matrix == pytest.approx(
        rotation.matrix.T
    )


def test_planar_mirror_compiles_exact_pentaprism_field_turn():
    incoming = np.asarray((1.0, 0.0, 0.0))
    outgoing = np.asarray((1.0, 1.0, 0.0))/np.sqrt(2.0)
    normal = (incoming-outgoing)/np.linalg.norm(incoming-outgoing)
    source = TransverseBasis.from_direction(incoming)
    destination = TransverseBasis.from_direction(outgoing)

    artifact = compile_planar_reflection_interface(
        source,
        destination,
        normal,
        (450e-9, 550e-9, 650e-9),
        material_name="aluminum_mirror",
        n_incident=1.5168,
    )

    assert artifact.coordinate_map is RigidFieldMap.FLIP_X
    assert artifact.jones.shape == (3, 2, 2)
    assert np.all(np.sum(np.abs(artifact.jones)**2, axis=1) < 1.0)
    contract = artifact.graph_parameters()
    assert contract["resampling"] == "none-exact-signed-permutation"
    assert contract["allocation"] == "cold-only"


def test_emitter_polarization_becomes_incoherent_jones_modes_without_faking_it():
    linear = PolarizationState(
        mode=PolarizationMode.LINEAR,
        angle_deg=90.0,
        degree_of_polarization=0.6,
    )
    assert linear.jones_vector == pytest.approx((0.0, 1.0))
    modes = linear.coherent_mode_decomposition()
    assert tuple(weight for weight, _field in modes) == pytest.approx((0.8, 0.2))
    assert abs(np.vdot(modes[0][1], modes[1][1])) < 1.0e-12

    unpolarized = PolarizationState(mode=PolarizationMode.UNPOLARIZED)
    unpolarized_modes = unpolarized.coherent_mode_decomposition()
    assert tuple(weight for weight, _field in unpolarized_modes) == pytest.approx(
        (0.5, 0.5)
    )

    radial = PolarizationState(mode=PolarizationMode.RADIAL)
    azimuthal = PolarizationState(mode=PolarizationMode.AZIMUTHAL)
    assert radial.jones_vector_at(np.pi / 2.0) == pytest.approx((0.0, 1.0))
    assert azimuthal.jones_vector_at(np.pi / 2.0) == pytest.approx((-1.0, 0.0))


def test_canonical_phase_space_maps_compose_and_remain_symplectic():
    distance_over_n = 0.4
    power = 2.5
    free = np.block([
        [np.eye(2), distance_over_n * np.eye(2)],
        [np.zeros((2, 2)), np.eye(2)],
    ])
    lens = np.block([
        [np.eye(2), np.zeros((2, 2))],
        [-power * np.eye(2), np.eye(2)],
    ])
    composed = PhaseSpaceJacobian(lens).compose(PhaseSpaceJacobian(free))
    assert composed.determinant == pytest.approx(1.0)
    assert composed.symplectic_residual < 1.0e-12
    identity = composed.compose(composed.inverse())
    assert identity.matrix == pytest.approx(np.eye(4), abs=1.0e-12)


def test_finite_difference_retains_full_signed_map_not_only_determinant():
    exact = np.asarray([
        [1.0, 0.2, 0.4, 0.0],
        [-0.1, 1.0, 0.0, 0.4],
        [-2.0, 0.0, 0.2, 0.0],
        [0.0, -2.0, 0.0, 0.2],
    ])
    offset = np.asarray((0.1, -0.2, 0.3, 0.4))
    estimated = PhaseSpaceJacobian.finite_difference(
        lambda state: exact @ state + offset,
        (0.2, 0.1, -0.3, 0.05),
        step=(1.0e-6, 2.0e-6, 1.0e-6, 2.0e-6),
    )
    assert estimated.matrix == pytest.approx(exact, abs=5.0e-11)
    assert estimated.determinant == pytest.approx(np.linalg.det(exact))


def test_ray_field_gain_rejects_caustic_and_phase_uses_reference_opl():
    magnifier = np.diag((2.0, 3.0, 0.5, 1.0 / 3.0))
    jacobian = PhaseSpaceJacobian(magnifier)
    assert jacobian.configuration_amplitude_gain() == pytest.approx(
        1.0 / np.sqrt(6.0)
    )
    caustic = PhaseSpaceJacobian(np.diag((0.0, 1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="caustic"):
        caustic.configuration_amplitude_gain()

    frequency = 532.0e12
    wavelength = 299_792_458.0 / frequency
    assert optical_phase(
        frequency, 10.0 + 0.25 * wavelength,
        reference_optical_path_m=10.0,
    ) == pytest.approx(1.0j, abs=2.0e-8)


def test_operator_state_block_is_contiguous_fixed_stride_and_indexed():
    state = ComplexOperatorStateBlock()
    basis_id = state.add_basis(TransverseBasis.from_direction((1.0, 0.0, 0.0)))
    operator_id = state.add_operator(ComplexOpticalOperator())
    frozen = state.freeze()
    assert basis_id == 0 and operator_id == 0
    assert frozen["schema"] == COMPLEX_OPTICAL_OPERATOR_SCHEMA
    assert frozen["bases"].shape == (1, 8)
    assert frozen["operators"].shape == (1, 24)
    assert frozen["bases"].flags.c_contiguous
    assert frozen["operators"].flags.c_contiguous
    assert frozen["basis_stride_bytes"] == 32
    assert frozen["operator_stride_bytes"] == 96
    assert frozen["source_modes"].shape == (0, 12)
    assert frozen["source_mode_stride_bytes"] == 48


def test_source_modes_are_fixed_stride_and_keep_coherence_outside_rays():
    state = ComplexOperatorStateBlock()
    basis_id = state.add_basis(TransverseBasis.from_direction((1.0, 0.0, 0.0)))
    operator_id = state.add_operator(ComplexOpticalOperator())
    handles = state.add_polarization_modes(
        PolarizationState(
            mode=PolarizationMode.LINEAR,
            angle_deg=30.0,
            degree_of_polarization=0.25,
        ),
        basis_id=basis_id,
        operator_id=operator_id,
        coherence_seed=0x1_0000_0000,
    )
    frozen = state.freeze()
    assert handles == (0, 1)
    assert frozen["source_modes"].shape == (2, 12)
    assert frozen["source_modes"].dtype == np.uint32
    assert frozen["source_modes"].flags.c_contiguous
    assert frozen["source_mode_stride_bytes"] == 48
    first = frozen["source_modes"][0]
    assert first[5] == 0
    assert first[6] == 1
    assert first[7] == basis_id
    assert first[8] == operator_id


def test_source_mode_block_installer_uses_tags_without_repacking():
    from camera_software.complex_optical_operators import (
        ComplexSourceMode,
        install_source_mode_block,
    )

    class Tracer:
        def configure_complex_source_modes(
            self, tags, indices, words, bases, operators,
        ):
            self.tags = tags.copy()
            self.indices = indices.copy()
            self.words = words.copy()
            self.bases = bases.copy()
            self.operators = operators.copy()

    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator())
    block.add_source_mode(ComplexSourceMode(
        jones=np.asarray([1.0, 1.0j]),
        power_weight=0.5,
        coherence_id=0x12345678ABCDEF01,
        basis_id=3,
        operator_id=7,
    ))
    tracer = Tracer()
    receipt = install_source_mode_block(
        tracer, np.asarray([0xFFEEDDCCBBAA0099], np.uint64), block,
    )

    assert receipt == {
        "source_binding_count": 1,
        "source_binding_stride_bytes": 16,
        "source_mode_count": 1,
        "source_mode_stride_bytes": 48,
    }
    assert tracer.tags.tolist() == [0xFFEEDDCCBBAA0099]
    assert tracer.indices.tolist() == [0]
    assert tracer.words.shape == (1, 12)


def test_source_mode_block_installer_deduplicates_shared_modes():
    from camera_software.complex_optical_operators import (
        ComplexSourceMode,
        install_source_mode_block,
    )

    class Tracer:
        def configure_complex_source_modes(
            self, tags, indices, words, bases, operators,
        ):
            self.tags = tags.copy()
            self.indices = indices.copy()
            self.words = words.copy()

    block = ComplexOperatorStateBlock()
    block.add_basis(TransverseBasis.from_direction((0.0, 0.0, 1.0)))
    block.add_operator(ComplexOpticalOperator())
    block.add_source_mode(ComplexSourceMode(
        jones=np.asarray([1.0, 0.0]),
        power_weight=1.0,
        coherence_id=9,
        basis_id=0,
    ))
    tracer = Tracer()
    receipt = install_source_mode_block(
        tracer,
        np.asarray([11, 22, 33, 44], np.uint64),
        block,
        mode_indices=np.zeros(4, np.uint32),
    )

    assert receipt["source_binding_count"] == 4
    assert receipt["source_mode_count"] == 1
    assert tracer.indices.tolist() == [0, 0, 0, 0]
    assert tracer.words.shape == (1, 12)


def test_cpu_gpu_operator_abi_matches_and_declares_no_binding():
    cpu = (
        ROOT / "csrc/include/complex_optical_operators.h"
    ).read_text(encoding="utf-8")
    gpu = (
        ROOT / "csrc/shaders/complex_optical_operators.glsl.inc"
    ).read_text(encoding="utf-8")
    assert "sizeof(PackedTransverseBasisGpu) == 32" in cpu
    assert "sizeof(PackedOperatorGpu) == 96" in cpu
    assert "sizeof(PackedSourceModeGpu) == 48" in cpu
    assert "COMPLEX_BASIS_WORDS 8u" in gpu
    assert "COMPLEX_OPERATOR_WORDS 24u" in gpu
    assert "COMPLEX_SOURCE_MODE_WORDS 12u" in gpu
    assert "layout(std430" not in gpu
    assert "binding =" not in gpu


def test_optical_graph_publishes_the_canonical_operator_contract():
    preset = simple_doublet_preset()
    lens = CompoundLens.from_preset(
        preset, wavelengths_um=preset.wavelengths, axial_scale=-1.0
    )
    contract = compile_compound_lens_graph(
        lens, lane_count=len(preset.wavelengths)
    ).contract()
    assert contract["complex_operator_contract"] == canonical_operator_contract()
    assert contract["operator_state_block"] == {
        "schema": COMPLEX_OPTICAL_OPERATOR_SCHEMA,
        "basis_count": 1,
        "operator_count": 1,
        "source_mode_count": 0,
        "basis_stride_bytes": 32,
        "operator_stride_bytes": 96,
        "source_mode_stride_bytes": 48,
        "ownership": "compiled-graph-persistent",
    }
    exact = next(
        node for node in contract["nodes"]
        if node["key"] == "camera.exact-compound-lens"
    )
    assert exact["parameters"]["jones_transport"] == "indexed-2x2-complex"
    assert exact["parameters"]["differential_transport"] == (
        "indexed-canonical-4x4"
    )


def test_native_exact_lens_returns_signed_spectral_symplectic_maps():
    preset = simple_doublet_preset()
    lens = CompoundLens.from_preset(
        preset, wavelengths_um=preset.wavelengths, axial_scale=-1.0
    )
    entry_x = min(element.x_pos for element in lens.elements) - 1.0e-4
    origins = np.asarray((
        (entry_x, 0.0, 0.0),
        (entry_x, 0.001, -0.0005),
    ))
    directions = np.asarray(((1.0, 0.0, 0.0), (1.0, 0.002, -0.001)))
    blue = estimate_compound_lens_phase_space(
        lens, origins, directions,
        spectral_lane=0, q_step_m=1.0e-7, p_step=1.0e-7,
    )
    red = estimate_compound_lens_phase_space(
        lens, origins, directions,
        spectral_lane=len(preset.wavelengths) - 1,
        q_step_m=1.0e-7, p_step=1.0e-7,
    )
    assert blue.backend == "native-cpp"
    assert np.all(blue.valid) and np.all(red.valid)
    assert blue.determinants == pytest.approx((1.0, 1.0), abs=2.0e-7)
    assert red.determinants == pytest.approx((1.0, 1.0), abs=2.0e-7)
    assert np.max(blue.symplectic_residuals) < 2.0e-9
    assert np.max(red.symplectic_residuals) < 2.0e-9
    # The differential map uses the exact lane's refractive-index table.
    assert not np.allclose(blue.matrices, red.matrices, rtol=1.0e-5)


def test_native_exact_lens_differential_map_is_reciprocal():
    preset = simple_doublet_preset()
    lens = CompoundLens.from_preset(
        preset, wavelengths_um=preset.wavelengths, axial_scale=-1.0
    )
    entry_x = min(element.x_pos for element in lens.elements) - 1.0e-6
    origin = np.asarray(((entry_x, 0.0002, -0.0001),))
    direction = np.asarray(((1.0, 0.002, -0.001),))
    lane = len(preset.wavelengths) // 2
    forward = estimate_compound_lens_phase_space(
        lens, origin, direction, spectral_lane=lane,
        q_step_m=1.0e-8, p_step=1.0e-8,
    )
    traced = lens.trace(origin[0], direction[0])
    backward = estimate_compound_lens_phase_space(
        lens,
        traced.origin[None, :],
        (-traced.direction)[None, :],
        spectral_lane=lane,
        q_step_m=1.0e-8,
        p_step=1.0e-8,
    )
    assert forward.valid[0] and backward.valid[0]
    time_reverse = np.diag((1.0, 1.0, -1.0, -1.0))
    expected = (
        time_reverse
        @ np.linalg.inv(forward.matrices[0])
        @ time_reverse
    )
    assert backward.matrices[0] == pytest.approx(expected, abs=3.0e-5)
