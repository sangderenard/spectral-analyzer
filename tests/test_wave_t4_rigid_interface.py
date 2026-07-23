from __future__ import annotations

import numpy as np
import pytest

from wave_transform_visual_demo import _calibration_tracer


def _apply(
    lanes: int,
    coordinate_map: int,
    jones: np.ndarray,
    s: np.ndarray,
    p: np.ndarray,
):
    tracer = _calibration_tracer(550.0e-9)
    result = tracer.t4_apply_rigid_field_interface(
        lanes,
        int(s.shape[2]),
        int(s.shape[1]),
        coordinate_map,
        np.ascontiguousarray(jones.real, np.float32),
        np.ascontiguousarray(jones.imag, np.float32),
        np.ascontiguousarray(s.real, np.float32),
        np.ascontiguousarray(s.imag, np.float32),
        np.ascontiguousarray(p.real, np.float32),
        np.ascontiguousarray(p.imag, np.float32),
    )
    return (
        np.asarray(result[0])+1j*np.asarray(result[1]),
        np.asarray(result[2])+1j*np.asarray(result[3]),
    )


@pytest.mark.parametrize("lanes", [1, 3, 4, 8, 16, 32])
def test_rigid_interface_identity_is_exact_for_every_lane_width(lanes: int):
    rng = np.random.default_rng(1000+lanes)
    shape = (lanes, 4, 4)
    s = rng.normal(size=shape)+1j*rng.normal(size=shape)
    p = rng.normal(size=shape)+1j*rng.normal(size=shape)
    jones = np.repeat(
        np.eye(2, dtype=np.complex64)[None], lanes, axis=0
    )

    out_s, out_p = _apply(lanes, 0, jones, s, p)

    np.testing.assert_array_equal(out_s, s.astype(np.complex64))
    np.testing.assert_array_equal(out_p, p.astype(np.complex64))


def test_rigid_interface_applies_coordinate_flip_and_complex_jones_matrix():
    s = np.arange(12, dtype=np.float32).reshape(1, 3, 4).astype(np.complex64)
    p = (20+np.arange(12, dtype=np.float32)).reshape(1, 3, 4)
    p = p.astype(np.complex64)
    jones = np.asarray([[[0.0, 1j], [1.0, 0.0]]], np.complex64)

    out_s, out_p = _apply(1, 2, jones, s, p)

    np.testing.assert_array_equal(out_s, 1j*p[:, ::-1, :])
    np.testing.assert_array_equal(out_p, s[:, ::-1, :])


def test_rigid_interface_unitary_operator_preserves_discrete_field_power():
    rng = np.random.default_rng(44)
    s = rng.normal(size=(4, 8, 8))+1j*rng.normal(size=(4, 8, 8))
    p = rng.normal(size=(4, 8, 8))+1j*rng.normal(size=(4, 8, 8))
    inv_sqrt_two = 1.0/np.sqrt(2.0)
    jones = np.repeat(np.asarray([[
        [inv_sqrt_two, 1j*inv_sqrt_two],
        [1j*inv_sqrt_two, inv_sqrt_two],
    ]], np.complex64), 4, axis=0)

    out_s, out_p = _apply(4, 7, jones, s, p)
    input_power = np.sum(
        np.abs(s.astype(np.complex64))**2
        + np.abs(p.astype(np.complex64))**2
    )
    output_power = np.sum(np.abs(out_s)**2+np.abs(out_p)**2)

    assert output_power == pytest.approx(input_power, rel=3.0e-7)


def test_rigid_interface_rejects_transpose_for_rectangular_grid():
    s = np.zeros((1, 3, 4), np.complex64)
    jones = np.eye(2, dtype=np.complex64)[None]

    with pytest.raises(RuntimeError, match="rejected"):
        _apply(1, 4, jones, s, s)
