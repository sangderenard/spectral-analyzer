from __future__ import annotations

import numpy as np
import pytest


def test_scene_order_preserves_explicit_calibration_spectral_records():
    from scene_orders import _material_payload

    authored = {
        "bands": [
            {
                "center_hz": 5.0e14,
                "bandwidth_hz": 1.0e12,
                "reflectance": 0.04,
                "transmittance": 0.94,
                "diffuse_frac": 0.0,
                "emission": 0.0,
                "reemission": 0.0,
                "ior_real": 1.52,
                "ior_imag": 0.0,
            }
        ]
    }

    payload = _material_payload(authored, "plane")

    assert payload["bands"] == authored["bands"]
    assert payload["bands"] is not authored["bands"]

from material_db import MAX_SPECTRAL_BANDS, MaterialDatabase
from spectral_material import Material, SpectralBand


def test_transport_mat_buf_resolves_lobes_by_frequency_not_row_number() -> None:
    db = MaterialDatabase()
    red_hz = 3.0e8 / 700.0e-9
    blue_hz = 3.0e8 / 450.0e-9
    db.register(
        "two_lobes",
        Material(
            name="two_lobes",
            domain="em_optical",
            spectral_bands=[
                SpectralBand(
                    center_hz=red_hz,
                    bandwidth_hz=red_hz * 0.01,
                    reflectance=0.8,
                    ior_real=1.4,
                ),
                SpectralBand(
                    center_hz=blue_hz,
                    bandwidth_hz=blue_hz * 0.01,
                    reflectance=0.2,
                    ior_real=1.8,
                ),
            ],
        ),
    )

    # Reverse the authored-lobe order deliberately. Transport rows must follow
    # this active grid, not whichever lobe happened to be stored first.
    grid = np.array([blue_hz, red_hz], np.float64)
    rows = db.build_mat_buf(freq_hz=grid).reshape(1, MAX_SPECTRAL_BANDS, 12)

    assert rows[0, 0, 0] == pytest.approx(blue_hz)
    assert rows[0, 1, 0] == pytest.approx(red_hz)
    assert rows[0, 0, 2] == pytest.approx(0.2, rel=1.0e-4)
    assert rows[0, 1, 2] == pytest.approx(0.8, rel=1.0e-4)
    assert rows[0, 0, 7] == pytest.approx(1.8, rel=1.0e-4)
    assert rows[0, 1, 7] == pytest.approx(1.4, rel=1.0e-4)


def test_transport_mat_buf_expands_pbr_only_material_across_active_grid() -> None:
    db = MaterialDatabase()
    db.register_from_mat16(
        "legacy",
        np.array(
            [0.4, 0.25, 0.35, 0.4, 0.25, 0.35, 0.2, 0.4, 0.6, 1.5, 1.0],
            np.float32,
        ),
    )
    grid = np.array([4.0e14, 5.0e14, 6.0e14], np.float64)
    rows = db.build_mat_buf(freq_hz=grid).reshape(1, MAX_SPECTRAL_BANDS, 12)

    assert np.all(rows[0, :3, 2] > 0.0)
    assert np.allclose(rows[0, :3, 7], 1.5)
    assert np.allclose(rows[0, :3, 0], grid)


def test_transport_mat_buf_rejects_invalid_grid() -> None:
    db = MaterialDatabase()
    db.register("plain", Material(name="plain", domain="em_optical"))
    with pytest.raises(ValueError, match="finite positive"):
        db.build_mat_buf(freq_hz=np.array([0.0], np.float64))
