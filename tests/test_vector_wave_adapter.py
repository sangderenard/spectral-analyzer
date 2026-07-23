import math

import numpy as np
import pytest

from camera_designer.emitter_profile import PolarizationMode, PolarizationState
from camera_software.physical_aperture import LivePhysicalAperture
from camera_software.vector_wave_adapter import JonesFieldState, PaddedWaveDomain
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


@pytest.mark.parametrize(
    "quality,solve_size,steps,ratio",
    [
        ("balanced", 128, 4, 4.0),
        ("high", 256, 8, 16.0),
        ("bake", 512, 16, 64.0),
    ],
)
def test_padded_wave_domain_quality_controls_hidden_investment(
    quality, solve_size, steps, ratio,
):
    domain = PaddedWaveDomain.for_quality((64, 64), quality)
    field = domain.uniform_scalar_field()
    assert domain.solve_shape == (solve_size, solve_size)
    assert domain.propagation_steps == steps
    assert domain.investment_ratio == ratio
    assert domain.absorber_cells > 0
    assert field.shape == (1, solve_size, solve_size)
    np.testing.assert_array_equal(
        domain.crop(field), np.ones((1, 64, 64), np.complex64)
    )


def test_open_propagation_substeps_and_uses_native_absorber():
    class RecordingTracer:
        def __init__(self):
            self.steps = []
            self.absorbers = []

        def t4_angular_spectrum_step(
            self, bands, width, height, pitch, distance, direction,
            wavelengths, re, im,
        ):
            self.steps.append((bands, width, height, distance, direction))

        def t4_apply_absorbing_border(
            self, bands, width, height, cells, strength, fraction, re, im,
        ):
            self.absorbers.append((cells, strength, fraction))
            border = float(
                np.sum(re[:, :cells] ** 2)
                + np.sum(im[:, :cells] ** 2)
            )
            return {
                "field_power": float(np.sum(re*re + im*im)),
                "border_power": border,
                "absorbed_power": 0.0,
            }

    domain = PaddedWaveDomain.for_quality((8, 8), "balanced")
    state = JonesFieldState.from_scalar_field(
        domain.uniform_scalar_field(),
        PolarizationState(mode=PolarizationMode.LINEAR),
    )
    tracer = RecordingTracer()
    propagated, telemetry = state.propagate_open_native(
        tracer,
        pitch_m=0.75e-6,
        distance_m=20.0e-6,
        direction_sign=1,
        wavelengths_m=np.asarray([532.0e-9]),
        domain=domain,
    )

    expected_calls = domain.propagation_steps * 2
    assert len(tracer.steps) == expected_calls
    assert len(tracer.absorbers) == expected_calls
    assert sum(call[3] for call in tracer.steps) == pytest.approx(40.0e-6)
    assert all(
        call[2] == pytest.approx(1.0 / domain.propagation_steps)
        for call in tracer.absorbers
    )
    assert propagated.shape == domain.solve_shape
    assert telemetry["investment_ratio"] == domain.investment_ratio


def test_padded_open_domain_reduces_periodic_box_error():
    tracer = _calibration_tracer(532.0e-9)
    wavelength = np.asarray([532.0e-9], np.float64)
    pitch = 0.75e-6
    visible_size = 16
    polarization = PolarizationState(mode=PolarizationMode.LINEAR)

    def solve(domain):
        if domain is None:
            solve_size = visible_size
            scalar = np.ones((solve_size, solve_size), np.complex64)
        else:
            solve_size = domain.solve_shape[0]
            scalar = domain.uniform_scalar_field()
        aperture = LivePhysicalAperture.iris(
            "periodic-error-test",
            blade_count=9,
            opening_radius_m=0.76*pitch,
            assembly_radius_m=0.51*math.hypot(
                (solve_size-1)*pitch, (solve_size-1)*pitch
            ),
            thickness_m=0.10e-6,
            material_n_real=2.9,
            material_n_imag=3.0,
        )
        state = JonesFieldState.from_scalar_field(scalar, polarization)
        material, _ = state.apply_isotropic_material_native(
            tracer,
            pitch_m=pitch,
            distance_m=aperture.thickness_m,
            direction_sign=1,
            wavelengths_m=wavelength,
            payload=aperture.wave_payload(),
        )
        if domain is None:
            propagated = material.propagate_native(
                tracer,
                pitch_m=pitch,
                distance_m=90.0e-6,
                direction_sign=1,
                wavelengths_m=wavelength,
            )
            intensity = propagated.stokes()[0]
        else:
            propagated, _ = material.propagate_open_native(
                tracer,
                pitch_m=pitch,
                distance_m=90.0e-6,
                direction_sign=1,
                wavelengths_m=wavelength,
                domain=domain,
            )
            intensity = domain.crop(propagated.stokes()[0])
        return intensity / float(np.max(intensity))

    periodic = solve(None)
    balanced = solve(
        PaddedWaveDomain.for_quality(
            (visible_size, visible_size), "balanced"
        )
    )
    bake_reference = solve(
        PaddedWaveDomain.for_quality(
            (visible_size, visible_size), "bake"
        )
    )
    periodic_error = float(np.sqrt(np.mean(
        (periodic-bake_reference)**2
    )))
    balanced_error = float(np.sqrt(np.mean(
        (balanced-bake_reference)**2
    )))

    assert balanced_error < 0.6*periodic_error
