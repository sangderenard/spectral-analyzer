import numpy as np
import pytest

from camera_software.optical_transport_graph import (
    OpticalExecutionDomain,
    wave_context_nodes,
)
from camera_software.physical_aperture import (
    APERTURE_PAYLOAD_MAGIC,
    APERTURE_PAYLOAD_VALUES,
    AperturePattern,
    LivePhysicalAperture,
)


def _iris():
    return LivePhysicalAperture.iris(
        "bench.iris",
        blade_count=8,
        opening_radius_m=2.0e-3,
        assembly_radius_m=8.0e-3,
        thickness_m=0.5e-3,
        rotation_rad=0.125,
    )


def test_live_aperture_has_one_geometry_and_wave_authority():
    aperture = _iris()
    vertices, normals = aperture.triangle_mesh(z_center_m=0.015)
    payload = aperture.wave_payload((0.0, 0.0, 2.0))

    assert vertices.shape == (8 * 12, 9)
    assert normals.shape == (8 * 12, 3)
    assert np.ptp(vertices.reshape(-1, 3)[:, 2]) == pytest.approx(
        aperture.thickness_m
    )
    assert payload.shape == (APERTURE_PAYLOAD_VALUES,)
    assert payload[6] == APERTURE_PAYLOAD_MAGIC
    assert int(payload[8]) == int(AperturePattern.IRIS_POLYGON)
    assert payload[16] == pytest.approx(aperture.thickness_m)
    assert aperture.graph_parameters()["ideal_mask"] is False


def test_wave_graph_retains_live_aperture_contract_and_fixed_payload():
    aperture = _iris()
    nodes, _ = wave_context_nodes(
        "blade-interaction",
        1,
        center_m=(0.0, 0.0, 0.0),
        radius_m=0.01,
        longitudinal_step_m=0.5e-3,
        longitudinal_steps=8,
        aperture_material=aperture,
    )
    arena = next(
        node for node in nodes
        if node.domain is OpticalExecutionDomain.T4_WAVE_ARENA
    )
    assert arena.parameters["aperture_material"]["key"] == "bench.iris"
    assert arena.parameters["aperture_material"]["ideal_mask"] is False
    assert len(arena.parameters["aperture_payload"]) == APERTURE_PAYLOAD_VALUES


def test_repeated_crt_patterns_share_the_live_wave_abi():
    for pattern in (
        AperturePattern.SHADOW_MASK,
        AperturePattern.SLOT_MASK,
        AperturePattern.APERTURE_GRILLE,
    ):
        aperture = LivePhysicalAperture(
            key=f"crt.{pattern.name.lower()}",
            pattern=pattern,
            opening_x_m=0.08e-3,
            opening_y_m=0.16e-3,
            pitch_x_m=0.28e-3,
            pitch_y_m=0.42e-3,
            assembly_radius_m=0.01,
            thickness_m=0.12e-3,
            material_name="invar",
            material_n_real=2.5,
            material_n_imag=3.5,
        )
        payload = aperture.wave_payload()
        assert len(payload) == APERTURE_PAYLOAD_VALUES
        assert int(payload[8]) == int(pattern)


def test_native_material_operator_is_reciprocal_and_not_an_ideal_mask():
    kernels = pytest.importorskip("_spectral_kernels")
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf

    wavelength = np.asarray([550.0e-9], np.float64)
    frequency = 299_792_458.0 / wavelength
    zero = np.zeros((1, 1), np.float64)
    mat_idx, mat_buf, mat_count = per_tri_spectral_to_mat_buf(
        zero, zero, zero, frequency
    )
    tracer = kernels.RayTracer(
        1,
        np.asarray([[0., 0., 0., 1., 0., 0., 1., 1., 0.]], np.float64),
        np.asarray([[0., 0., 1.]], np.float64),
        mat_idx, mat_buf, int(mat_count), frequency,
        299_792_458.0, np.zeros(1, np.float64),
    )
    aperture = LivePhysicalAperture.iris(
        "reciprocal.iris",
        blade_count=8,
        opening_radius_m=4.0e-6,
        assembly_radius_m=14.0e-6,
        thickness_m=0.2e-6,
        material_n_real=1.45,
        material_n_imag=0.02,
    )
    payload = aperture.wave_payload()
    forward_re = np.ones((1, 32, 32), np.float32)
    forward_im = np.zeros_like(forward_re)
    reverse_re, reverse_im = forward_re.copy(), forward_im.copy()
    forward = tracer.t4_apply_aperture_material(
        1, 32, 32, 1.0e-6, aperture.thickness_m, 1,
        wavelength, payload, forward_re, forward_im,
    )
    tracer.t4_apply_aperture_material(
        1, 32, 32, 1.0e-6, aperture.thickness_m, -1,
        wavelength, payload, reverse_re, reverse_im,
    )

    # The clear center remains untouched. A blade sample has finite,
    # non-binary complex transmission and reciprocal phase.
    assert forward_re[0, 16, 16] == pytest.approx(1.0)
    blade = (16, 26)
    forward_value = complex(
        forward_re[(0, *blade)], forward_im[(0, *blade)]
    )
    reverse_value = complex(
        reverse_re[(0, *blade)], reverse_im[(0, *blade)]
    )
    assert 0.0 < abs(forward_value) < 1.0
    assert reverse_value == pytest.approx(forward_value.conjugate(), rel=1.0e-5)
    assert forward["absorbed_power"] > 0.0
    assert forward["ideal_mask"] is False
