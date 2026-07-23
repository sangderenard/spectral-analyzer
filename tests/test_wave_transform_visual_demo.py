from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wave_transform_visual_demo import (
    _aperture_sweep_radius,
    _remove_piston_phase,
    render_sequence,
    run_aperture_live,
    run_live,
    run_transport_live,
)


def test_visual_demo_uses_production_transform_and_writes_sequence(tmp_path):
    result = render_sequence(tmp_path, size=16, frames=2, scale=1)
    assert len(result["frames"]) == 3
    assert result["max_roundtrip_error"] < 2.0e-5
    for value in (*result["frames"], result["summary"]):
        path = Path(value)
        assert path.is_file()
        with Image.open(path) as image:
            assert image.width > 0
            assert image.height > 0


def test_live_mode_validates_before_opening_a_context():
    with pytest.raises(ValueError, match="power of two"):
        run_live(size=15)
    with pytest.raises(ValueError, match="cycle_steps"):
        run_live(cycle_steps=0)
    with pytest.raises(ValueError, match="fps"):
        run_live(fps=0)
    with pytest.raises(ValueError, match="panel_size"):
        run_transport_live(panel_size=32)
    with pytest.raises(ValueError, match="power of two"):
        run_aperture_live(size=15)
    with pytest.raises(ValueError, match="polarization"):
        run_aperture_live(size=16, polarization_mode="invented")
    with pytest.raises(ValueError, match="quality"):
        run_aperture_live(size=16, quality="reckless")


def test_relative_phase_gauge_removes_only_global_piston():
    y, x = np.mgrid[-1.0:1.0:9j, -1.0:1.0:9j]
    base = np.exp(-(x*x+y*y))*np.exp(1j*(0.7*x-0.35*y+0.2*x*y))
    rotated = base*np.exp(1j*2.17)
    base_relative, _ = _remove_piston_phase(base)
    rotated_relative, piston = _remove_piston_phase(rotated)

    np.testing.assert_allclose(
        rotated_relative, base_relative, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(np.abs(rotated_relative), np.abs(rotated))
    assert np.isfinite(piston)


def test_aperture_sweep_reaches_pinhole_and_clear_field_extremes():
    pitch = 0.75e-6
    minimum, assembly, sweep_min = _aperture_sweep_radius(0.0, 64, pitch)
    maximum, _, sweep_max = _aperture_sweep_radius(np.pi, 64, pitch)
    field_radius = 0.5*63*pitch

    assert minimum == pytest.approx(0.76*pitch)
    assert maximum == pytest.approx(1.51*field_radius)
    assert maximum < assembly
    assert sweep_min == pytest.approx(0.0)
    assert sweep_max == pytest.approx(1.0)
