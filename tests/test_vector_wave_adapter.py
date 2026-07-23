import numpy as np
import pytest

from camera_designer.emitter_profile import PolarizationMode, PolarizationState
from camera_software.physical_aperture import LivePhysicalAperture
from camera_software.vector_wave_adapter import JonesFieldState
from wave_transform_visual_demo import _calibration_tracer


def _uniform(size=16):
    return np.ones((size, size), np.complex64)


def test_jones_field_stokes_respects_linear_circular_and_unpolarized_modes():
    linear = JonesFieldState.from_scalar_field(
        _uniform(),
        PolarizationState(
            mode=PolarizationMode.LINEAR,
            angle_deg=45.0,
        ),
    )
    i, q, u, v = linear.stokes()
    assert q == pytest.approx(np.zeros_like(q), abs=1.0e-6)
    assert u == pytest.approx(i, abs=1.0e-6)
    assert v == pytest.approx(np.zeros_like(v), abs=1.0e-6)

    circular = JonesFieldState.from_scalar_field(
        _uniform(),
        PolarizationState(
            mode=PolarizationMode.CIRCULAR,
            handedness=1,
        ),
    )
    i, q, u, v = circular.stokes()
    assert q == pytest.approx(np.zeros_like(q), abs=1.0e-6)
    assert u == pytest.approx(np.zeros_like(u), abs=1.0e-6)
    assert v == pytest.approx(i, abs=1.0e-6)

    unpolarized = JonesFieldState.from_scalar_field(
        _uniform(),
        PolarizationState(mode=PolarizationMode.UNPOLARIZED),
    )
    i, q, u, v = unpolarized.stokes()
    assert unpolarized.mode_count == 2
    assert q == pytest.approx(np.zeros_like(q), abs=1.0e-6)
    assert u == pytest.approx(np.zeros_like(u), abs=1.0e-6)
    assert v == pytest.approx(np.zeros_like(v), abs=1.0e-6)
    assert unpolarized.total_power() == pytest.approx(float(i.sum()))


def test_radial_field_has_spatial_jones_structure():
    radial = JonesFieldState.from_scalar_field(
        _uniform(17),
        PolarizationState(mode=PolarizationMode.RADIAL),
    )
    i, q, u, v = radial.stokes()
    center = 8
    assert q[center, -1] == pytest.approx(i[center, -1], abs=1.0e-6)
    assert q[-1, center] == pytest.approx(-i[-1, center], abs=1.0e-6)
    assert np.max(np.abs(v)) < 1.0e-6


def test_native_isotropic_aperture_preserves_circular_jones_relation():
    tracer = _calibration_tracer(532.0e-9)
    wavelength = np.asarray([532.0e-9], np.float64)
    state = JonesFieldState.from_scalar_field(
        _uniform(16),
        PolarizationState(
            mode=PolarizationMode.CIRCULAR,
            handedness=1,
        ),
    )
    aperture = LivePhysicalAperture.iris(
        "vector-test",
        blade_count=7,
        opening_radius_m=3.0e-6,
        assembly_radius_m=10.0e-6,
        thickness_m=0.15e-6,
        material_n_real=2.9,
        material_n_imag=3.0,
    )
    material, accounting = state.apply_isotropic_material_native(
        tracer,
        pitch_m=0.75e-6,
        distance_m=aperture.thickness_m,
        direction_sign=1,
        wavelengths_m=wavelength,
        payload=aperture.wave_payload(),
    )
    propagated = material.propagate_native(
        tracer,
        pitch_m=0.75e-6,
        distance_m=30.0e-6,
        direction_sign=1,
        wavelengths_m=wavelength,
    )
    i, q, u, v = propagated.stokes()
    illuminated = i > float(i.max()) * 1.0e-7
    assert np.max(np.abs(q[illuminated])) < float(i.max()) * 2.0e-5
    assert np.max(np.abs(u[illuminated])) < float(i.max()) * 2.0e-5
    assert v[illuminated] == pytest.approx(i[illuminated], rel=2.0e-5)
    assert accounting["ideal_mask"] is False
    assert accounting["absorbed_power"] > 0.0
