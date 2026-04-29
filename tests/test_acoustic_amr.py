import math

import numpy as np
import pytest

from acoustic_amr import (
    AMR_APERTURE,
    AMR_GUITAR_INTERIOR,
    AMR_MESH_INTERSECTION,
    AcousticAMRFDTD,
    build_guitar_amr_grid,
)


def _ellipse_outline(n=32, rx=0.18, ry=0.24):
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([rx * np.cos(theta), ry * np.sin(theta)]).astype(np.float64)


def _small_grid(**kwargs):
    return build_guitar_amr_grid(
        _ellipse_outline(),
        body_h=0.08,
        dx=0.08,
        pad_cells=2,
        n_pml=1,
        soundhole=(0.0, 0.0, 0.045),
        max_refinement_level=2,
        **kwargs,
    )


def test_amr_importance_policy_refines_contents():
    grid = _small_grid()

    assert np.any(grid.importance == AMR_APERTURE)
    assert np.all(grid.cell_levels[grid.importance == AMR_APERTURE] == 2)

    assert np.any(grid.importance == AMR_GUITAR_INTERIOR)
    assert np.all(grid.cell_levels[grid.importance == AMR_GUITAR_INTERIOR] >= 1)

    assert np.any(grid.importance == AMR_MESH_INTERSECTION)
    assert np.all(grid.cell_levels[grid.importance == AMR_MESH_INTERSECTION] >= 1)

    low = grid.importance == 0
    assert np.any(low)
    assert grid.base_dx == pytest.approx(0.08)
    assert grid.min_dx == pytest.approx(0.02)


def test_amr_topology_is_balanced_and_positive():
    grid = _small_grid()
    jumps = np.abs(grid.cell_levels[grid.face_cell_neg] - grid.cell_levels[grid.face_cell_pos])
    assert int(jumps.max()) <= 1
    assert np.all(grid.cell_volumes > 0.0)
    assert np.all(grid.face_area > 0.0)
    assert np.all(grid.face_distance > 0.0)


def test_amr_requires_soundhole_and_balancing():
    with pytest.raises(ValueError, match="soundhole"):
        build_guitar_amr_grid(
            _ellipse_outline(),
            body_h=0.08,
            dx=0.08,
            pad_cells=2,
            n_pml=1,
            soundhole=None,
            max_refinement_level=2,
        )

    with pytest.raises(ValueError, match="balance_refinement=True"):
        _small_grid(balance_refinement=False)


def test_amr_c_pressure_stepper_is_required_and_moves_pressure():
    pytest.importorskip("_spectral_kernels")
    grid = _small_grid()
    solver = AcousticAMRFDTD(grid)
    solver.inject_pressure(np.array([0.0, 0.0, 0.04]), 1.0)
    p0 = solver.pressure.copy()
    solver.step(2)
    assert solver.dt > 0.0
    assert np.linalg.norm(solver.pressure - p0) > 0.0
