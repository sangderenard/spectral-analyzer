from __future__ import annotations

import numpy as np
import pytest

from ray_tracer_bridge import (
    RTS_APERTURE,
    RTS_CARDIOID,
    RTS_FIGURE8,
    RTS_HYPERCARDIOID,
    RTS_OMNI,
    _highband_weight,
    apply_mic_transfer_frequency_domain,
    make_microphone_receiver,
    mic_pattern_from_pressure_velocity,
)


def test_microphone_receiver_maps_coevolver_cardioid_def() -> None:
    mic_def = {
        "pos": np.array([0.0, 0.0, 0.135], dtype=np.float32),
        "axis": np.array([0.0, 0.0, -2.0], dtype=np.float32),
        "polar_a": 0.5,
        "polar_b": 0.5,
    }

    rec = make_microphone_receiver(mic_def)

    assert rec["kind"] == "microphone"
    assert rec["polar_type"] == RTS_CARDIOID
    np.testing.assert_allclose(rec["pos"], [0.0, 0.0, 0.135])
    np.testing.assert_allclose(rec["axis"], [0.0, 0.0, -1.0])


@pytest.mark.parametrize(
    "polar_a, polar_b, pattern, rts_type",
    [
        (1.0, 0.0, "omni", RTS_OMNI),
        (0.5, 0.5, "cardioid", RTS_CARDIOID),
        (0.0, 1.0, "figure8", RTS_FIGURE8),
        (0.25, 0.75, "hypercardioid", RTS_HYPERCARDIOID),
    ],
)
def test_mic_pressure_velocity_patterns_are_exact_presets(
    polar_a: float, polar_b: float, pattern: str, rts_type: int
) -> None:
    assert mic_pattern_from_pressure_velocity(polar_a, polar_b) == pattern
    rec = make_microphone_receiver(
        {"pos": [1.0, 2.0, 3.0], "axis": [0.0, 1.0, 0.0], "polar_a": polar_a, "polar_b": polar_b}
    )
    assert rec["polar_type"] == rts_type


def test_microphone_receiver_uses_aperture_type_when_radius_is_positive() -> None:
    rec = make_microphone_receiver(
        pos=[0.0, 0.0, 0.1],
        axis=[0.0, 0.0, -1.0],
        pattern="cardioid",
        aperture_r=0.012,
    )

    assert rec["polar_type"] == RTS_APERTURE
    assert rec["aperture_r"] == pytest.approx(0.012)


def test_custom_pressure_velocity_mix_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        mic_pattern_from_pressure_velocity(0.7, 0.3)


def test_microphone_receiver_rejects_missing_axis() -> None:
    with pytest.raises(ValueError, match="axis"):
        make_microphone_receiver(
            {"pos": [0.0, 0.0, 0.1], "polar_a": 0.5, "polar_b": 0.5}
        )


def test_microphone_receiver_rejects_zero_axis() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        make_microphone_receiver(
            {
                "pos": [0.0, 0.0, 0.1],
                "axis": [0.0, 0.0, 0.0],
                "polar_a": 0.5,
                "polar_b": 0.5,
            }
        )


def test_highband_weight_crosses_over_smoothly() -> None:
    freq = np.array([375.0, 1500.0, 6000.0])
    w = _highband_weight(freq, crossover_hz=1500.0, order=4)

    assert w[0] < 0.01
    assert w[1] == pytest.approx(0.5)
    assert w[2] > 0.99


def test_apply_mic_transfer_frequency_domain_preserves_shape_and_gain() -> None:
    sr = 8000.0
    n = 256
    src = np.ones((1, n), dtype=np.float32)
    freq_hz = np.array([0.0, sr / 2.0], dtype=np.float64)
    H = np.ones((1, 1, 2), dtype=np.complex128)

    y = apply_mic_transfer_frequency_domain(src, H, freq_hz, sr)

    assert y.shape == (1, n)
    np.testing.assert_allclose(y[0], src[0], atol=1e-5)


def test_apply_mic_transfer_rejects_source_count_mismatch() -> None:
    src = np.ones((2, 64), dtype=np.float32)
    H = np.ones((1, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="source count"):
        apply_mic_transfer_frequency_domain(src, H, np.array([100.0, 1000.0]), 8000.0)


def test_apply_mic_transfer_rejects_incomplete_frequency_coverage() -> None:
    src = np.ones((1, 64), dtype=np.float32)
    H = np.ones((1, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="full FFT range"):
        apply_mic_transfer_frequency_domain(src, H, np.array([100.0, 1000.0]), 8000.0)
