"""Regression tests for the AMR co-evolver.

Tests cover:
  1.  AMR descriptor classification (aperture/interior/mesh cells at expected levels)
  2.  AMR topology balance (no level jump > 1 across any face)
  3.  AMR FDTD CFL stability (dt > 0, dt < 100 ms)
  4.  AMR aperture openness (interior ↔ exterior connection)
  5.  Mic conic importance projection (multiple intermediate levels, aperture unchanged)
  6.  AMR coevolver pluck smoke test (runs without error, produces audio)
  7.  Bridge force propagates through plate to AMR pressure
  8.  Plate motion radiates through aperture
  9.  Sealed aperture reduces exterior pressure
  10. Mic pressure + velocity sampling nonzero after pluck
  11. Legacy uniform path still works (no AMR regression)
  12. AMR construction fails loudly with incomplete bridge mappings
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from acoustic_amr import (
    AMR_AIR,
    AMR_APERTURE,
    AMR_GUITAR_INTERIOR,
    AMR_MESH_INTERSECTION,
    AMR_PLATE,
    AMR_PML,
    AMR_WALL,
    AcousticAMRGrid,
    build_amr_coevolver_descriptor,
    build_bridge_plate_mapping,
    build_plate_amr_mappings,
    build_guitar_amr_grid,
    default_importance_policy,
    project_mic_importance_cone,
    validate_amr_grid,
)


# ---------------------------------------------------------------------------
# Helper: check if C extension is available
# ---------------------------------------------------------------------------

def _c_ext_missing() -> bool:
    try:
        import _spectral_kernels  # noqa: F401
        return False
    except ImportError:
        return True


# ---------------------------------------------------------------------------
# Helpers — tiny guitar body
# ---------------------------------------------------------------------------

def _simple_outline(cx: float = 0.25, cy: float = 0.2, rx: float = 0.18, ry: float = 0.15, n: int = 32) -> np.ndarray:
    """Elliptical guitar-body outline for tests."""
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(theta), cy + ry * np.sin(theta)], axis=1)


def _make_grid(
    *,
    dx: float = 0.04,
    pad_cells: int = 3,
    n_pml: int = 2,
    max_level: int = 2,
    body_h: float = 0.10,
    soundhole: tuple[float, float, float] = (0.25, 0.24, 0.04),
) -> AcousticAMRGrid:
    outline = _simple_outline()
    return build_guitar_amr_grid(
        outline,
        body_h,
        dx=dx,
        pad_cells=pad_cells,
        n_pml=n_pml,
        soundhole=soundhole,
        max_refinement_level=max_level,
        balance_refinement=True,
    )


def _plate_params(grid: AcousticAMRGrid, body_h: float = 0.10):
    """Return minimal plate parameters for a small test plate."""
    origin = np.array([grid.bounds_min[0] + grid.base_dx, grid.bounds_min[1] + grid.base_dx, 0.0], dtype=np.float32)
    plate_Nx = max(4, int((grid.bounds_max[0] - grid.bounds_min[0] - 2 * grid.base_dx) / (grid.base_dx * 0.5)))
    plate_Ny = max(4, int((grid.bounds_max[1] - grid.bounds_min[1] - 2 * grid.base_dx) / (grid.base_dx * 0.5)))
    plate_dx = float(grid.base_dx * 0.5)
    return plate_Nx, plate_Ny, plate_dx, origin


# ---------------------------------------------------------------------------
# Test 1 — AMR descriptor classification
# ---------------------------------------------------------------------------

def test_amr_classification_levels():
    grid = _make_grid(max_level=3)
    policy = default_importance_policy(3)
    # Aperture cells must be at the maximum level.
    aperture_levels = grid.cell_levels[grid.importance == AMR_APERTURE]
    assert aperture_levels.size > 0, "No aperture cells"
    assert np.all(aperture_levels == policy.aperture), \
        f"Aperture cells not at level {policy.aperture}: {np.unique(aperture_levels)}"
    # Interior cells must be >= guitar_interior level.
    interior_levels = grid.cell_levels[grid.importance == AMR_GUITAR_INTERIOR]
    assert interior_levels.size > 0, "No guitar-interior cells"
    assert np.all(interior_levels >= policy.guitar_interior), "Interior cells below required level"
    # Mesh-intersection cells must be >= mesh_intersections level.
    mesh_levels = grid.cell_levels[grid.importance == AMR_MESH_INTERSECTION]
    if mesh_levels.size > 0:
        assert np.all(mesh_levels >= policy.mesh_intersections), "Mesh cells below required level"


# ---------------------------------------------------------------------------
# Test 2 — AMR topology balance
# ---------------------------------------------------------------------------

def test_amr_topology_balance():
    grid = _make_grid()
    lneg = grid.cell_levels[grid.face_cell_neg]
    lpos = grid.cell_levels[grid.face_cell_pos]
    max_jump = int(np.abs(lneg.astype(np.int32) - lpos.astype(np.int32)).max())
    assert max_jump <= 1, f"AMR level jump of {max_jump} detected (max allowed: 1)"


# ---------------------------------------------------------------------------
# Test 3 — AMR FDTD CFL stability
# ---------------------------------------------------------------------------

def test_amr_cfl_stability():
    grid = _make_grid()
    c = 343.0
    # CFL dt = 0.77 * min_dx / (c * sqrt(3))
    dt = 0.77 * float(grid.min_dx) / (c * math.sqrt(3.0))
    assert dt > 0.0, "CFL dt is not positive"
    assert dt < 0.1, f"CFL dt {dt:.4f} s is unreasonably large (>=100 ms)"


# ---------------------------------------------------------------------------
# Test 4 — AMR aperture openness
# ---------------------------------------------------------------------------

def test_amr_aperture_openness():
    grid = _make_grid()
    # validate_amr_grid raises if aperture does not connect interior to exterior.
    validate_amr_grid(grid)  # should not raise


# ---------------------------------------------------------------------------
# Test 5 — Mic conic importance projection
# ---------------------------------------------------------------------------

def test_mic_conic_projection_levels():
    grid = _make_grid(max_level=3)
    cx, cy, hr = grid.soundhole
    body_h = 0.10
    mic_pos = np.array([cx, cy, body_h + 0.30], dtype=np.float64)

    # Sample aperture perimeter points.
    theta = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    perim = np.stack([
        cx + hr * np.cos(theta),
        cy + hr * np.sin(theta),
        np.full(16, body_h),
    ], axis=1)

    grid2 = project_mic_importance_cone(grid, mic_pos, perim, intermediate_levels=2)

    # Aperture cells must be unchanged.
    aperture_levels_orig = grid.cell_levels[grid.importance == AMR_APERTURE]
    aperture_levels_new  = grid2.cell_levels[grid2.importance == AMR_APERTURE]
    assert np.array_equal(aperture_levels_orig, aperture_levels_new), \
        "project_mic_importance_cone modified aperture cell levels"

    # There must be cells with intermediate levels between min and max.
    max_level = int(grid.max_refinement_level)
    non_aperture = grid2.importance != AMR_APERTURE
    levels_non_ap = grid2.cell_levels[non_aperture]
    unique_levels = np.unique(levels_non_ap)
    # At least 2 distinct non-zero intermediate levels present among non-aperture cells.
    intermediate = unique_levels[(unique_levels > 0) & (unique_levels < max_level)]
    assert len(intermediate) >= 2, \
        f"Expected >= 2 distinct intermediate non-aperture levels; got {intermediate}"

    # Topology balance must hold after projection.
    lneg = grid2.cell_levels[grid2.face_cell_neg]
    lpos = grid2.cell_levels[grid2.face_cell_pos]
    max_jump = int(np.abs(lneg.astype(np.int32) - lpos.astype(np.int32)).max())
    assert max_jump <= 1, f"Level jump after cone projection: {max_jump}"


# ---------------------------------------------------------------------------
# Test 6 — AMR coevolver pluck smoke test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_amr_coevolver_pluck_smoke():
    """Full round-trip: build descriptor → create coevolver → pluck → check audio."""
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import (
        CoEvolverStringDef, CoEvolverPickupDef, CoEvolverMicDef,
        coevolver_create_amr, coevolver_destroy,
        coevolver_pluck, coevolver_step_block,
    )

    grid = _make_grid(max_level=2)
    plate_Nx, plate_Ny, plate_dx, plate_origin = _plate_params(grid)
    cx, cy, hr = grid.soundhole
    bridge_xyz = np.array([[cx, cy - hr * 0.3, 0.0]], dtype=np.float64)

    desc = build_amr_coevolver_descriptor(
        grid,
        plate_Nx=plate_Nx,
        plate_Ny=plate_Ny,
        plate_dx=plate_dx,
        plate_origin=plate_origin,
        body_h=0.10,
        plate_mass_density=8.5,
        plate_stiffness_D=1.5,
        plate_alpha_M=5.0,
        plate_beta_K=1e-5,
        bridge_src_xyz=bridge_xyz,
        c=343.0,
        rho_air=1.21,
    )

    # Minimal single guitar string.
    str_def = CoEvolverStringDef()
    str_def.length = 0.65
    str_def.tension = 70.0
    str_def.linear_density = 5e-4
    str_def.damping = 2.0
    str_def.n_nodes = 48
    str_def.bridge_flat_idx = int(desc["bridge_plate_idx"][0])

    mic_def = CoEvolverMicDef()
    mic_def.pos_x = float(cx)
    mic_def.pos_y = float(cy)
    mic_def.pos_z = 0.10 + 0.20
    mic_def.axis_x = 0.0
    mic_def.axis_y = 0.0
    mic_def.axis_z = 1.0
    mic_def.polar_a = 0.5
    mic_def.polar_b = 0.5

    st = coevolver_create_amr(
        1, [str_def], 0, [], 1, [mic_def], desc, 44100.0, 8
    )
    assert st is not None, "coevolver_create_amr returned NULL"
    try:
        rc = coevolver_pluck(st, 0, 0.5, 0.001)
        assert rc == 0, f"coevolver_pluck returned {rc}"
        n_samples = 256
        out = np.zeros((1, n_samples), dtype=np.float32)
        rc = coevolver_step_block(st, n_samples, out)
        assert rc == 0, f"coevolver_step_block returned {rc}"
        assert np.any(np.abs(out) > 1e-12), "No audio output after pluck"
    finally:
        coevolver_destroy(st)


# ---------------------------------------------------------------------------
# Test 7 — Bridge force → plate → AMR pressure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_bridge_force_reaches_pressure():
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import amr_create, amr_setup_plate, amr_inject_bridge_drive, amr_step, amr_get_pressure, amr_destroy, amr_set_bridge_plate_sources

    grid = _make_grid(max_level=2)
    plate_Nx, plate_Ny, plate_dx, plate_origin = _plate_params(grid)
    cx, cy, hr = grid.soundhole
    bridge_xyz = np.array([[cx, cy - hr * 0.3, 0.0]], dtype=np.float64)

    desc = build_amr_coevolver_descriptor(
        grid, plate_Nx=plate_Nx, plate_Ny=plate_Ny, plate_dx=plate_dx,
        plate_origin=plate_origin, body_h=0.10,
        plate_mass_density=8.5, plate_stiffness_D=1.5,
        plate_alpha_M=5.0, plate_beta_K=1e-5,
        bridge_src_xyz=bridge_xyz,
    )

    st = amr_create(
        desc["n_cells"],
        desc["cell_centers"], desc["cell_volumes"], desc["open_volume_frac"],
        desc["cell_types"],
        desc["n_faces"],
        desc["face_cell_neg"], desc["face_cell_pos"],
        desc["face_area"], desc["face_open_frac"], desc["face_distance"],
        desc["c"], desc["rho_air"], desc["min_dx"],
    )
    assert st is not None

    try:
        amr_setup_plate(
            st,
            desc["plate_Nx"], desc["plate_Ny"], desc["plate_dx"], desc["plate_origin"],
            desc["plate_active"],
            desc["plate_mass_density"], desc["plate_stiffness_D"],
            desc["plate_alpha_M"], desc["plate_beta_K"],
            desc["n_active_plate"],
            desc["plate_active_flat_idx"],
            desc["plate_face_above_starts"], desc["plate_face_above_idx"], desc["plate_face_above_wgt"],
            desc["plate_face_below_starts"], desc["plate_face_below_idx"], desc["plate_face_below_wgt"],
            desc["plate_cell_above"], desc["plate_cell_below"],
        )
        amr_set_bridge_plate_sources(st, desc["n_bridge_plate"], desc["bridge_plate_idx"], desc["bridge_plate_wgt"])

        drives = np.ones(desc["n_bridge_plate"], dtype=np.float32) * 0.1
        amr_inject_bridge_drive(st, drives, 1.0)
        for _ in range(20):
            amr_step(st, 1)

        p = amr_get_pressure(st)
        assert np.any(np.abs(p) > 1e-15), "Pressure field remained zero after bridge force injection"
    finally:
        amr_destroy(st)


# ---------------------------------------------------------------------------
# Test 8 — Plate motion → aperture radiation (interior pressure transfers out)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_plate_aperture_radiation():
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import amr_create, amr_setup_plate, amr_inject_plate_force, amr_step, amr_get_pressure, amr_destroy

    grid = _make_grid(max_level=2)
    plate_Nx, plate_Ny, plate_dx, plate_origin = _plate_params(grid)
    cx, cy, hr = grid.soundhole
    bridge_xyz = np.array([[cx, cy - hr * 0.3, 0.0]], dtype=np.float64)
    desc = build_amr_coevolver_descriptor(
        grid, plate_Nx=plate_Nx, plate_Ny=plate_Ny, plate_dx=plate_dx,
        plate_origin=plate_origin, body_h=0.10,
        plate_mass_density=8.5, plate_stiffness_D=1.5,
        plate_alpha_M=5.0, plate_beta_K=1e-5, bridge_src_xyz=bridge_xyz,
    )

    st = amr_create(
        desc["n_cells"],
        desc["cell_centers"], desc["cell_volumes"], desc["open_volume_frac"],
        desc["cell_types"],
        desc["n_faces"],
        desc["face_cell_neg"], desc["face_cell_pos"],
        desc["face_area"], desc["face_open_frac"], desc["face_distance"],
        desc["c"], desc["rho_air"], desc["min_dx"],
    )
    assert st is not None
    try:
        amr_setup_plate(
            st,
            desc["plate_Nx"], desc["plate_Ny"], desc["plate_dx"], desc["plate_origin"],
            desc["plate_active"],
            desc["plate_mass_density"], desc["plate_stiffness_D"],
            desc["plate_alpha_M"], desc["plate_beta_K"],
            desc["n_active_plate"],
            desc["plate_active_flat_idx"],
            desc["plate_face_above_starts"], desc["plate_face_above_idx"], desc["plate_face_above_wgt"],
            desc["plate_face_below_starts"], desc["plate_face_below_idx"], desc["plate_face_below_wgt"],
            desc["plate_cell_above"], desc["plate_cell_below"],
        )
        n_active = int(desc["n_active_plate"])
        plate_idx = desc["plate_active_flat_idx"][:min(5, n_active)]
        wgt = np.ones(len(plate_idx), dtype=np.float32) / len(plate_idx)
        amr_inject_plate_force(st, len(plate_idx), plate_idx, wgt, 0.5)
        for _ in range(40):
            amr_step(st, 1)

        p = amr_get_pressure(st)
        centers = grid.cell_centers
        body_h = 0.10
        exterior = centers[:, 2] > body_h + grid.min_dx
        p_ext = np.abs(p[exterior])
        assert np.any(p_ext > 1e-15), "No exterior pressure after plate excitation (aperture radiation failed)"
    finally:
        amr_destroy(st)


# ---------------------------------------------------------------------------
# Test 9 — Sealed aperture reduces exterior pressure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_sealed_aperture_reduces_exterior():
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import amr_create, amr_setup_plate, amr_inject_plate_force, amr_step, amr_get_pressure, amr_destroy

    grid = _make_grid(max_level=2)
    plate_Nx, plate_Ny, plate_dx, plate_origin = _plate_params(grid)
    cx, cy, hr = grid.soundhole
    bridge_xyz = np.array([[cx, cy - hr * 0.3, 0.0]], dtype=np.float64)
    desc = build_amr_coevolver_descriptor(
        grid, plate_Nx=plate_Nx, plate_Ny=plate_Ny, plate_dx=plate_dx,
        plate_origin=plate_origin, body_h=0.10,
        plate_mass_density=8.5, plate_stiffness_D=1.5,
        plate_alpha_M=5.0, plate_beta_K=1e-5, bridge_src_xyz=bridge_xyz,
    )

    def _run_and_get_ext_pressure(open_frac_override: float) -> float:
        face_open = desc["face_open_frac"].copy()
        if open_frac_override == 0.0:
            face_open[:] = 0.0
        st = amr_create(
            desc["n_cells"],
            desc["cell_centers"], desc["cell_volumes"], desc["open_volume_frac"],
            desc["cell_types"],
            desc["n_faces"],
            desc["face_cell_neg"], desc["face_cell_pos"],
            desc["face_area"], face_open, desc["face_distance"],
            desc["c"], desc["rho_air"], desc["min_dx"],
        )
        assert st is not None
        try:
            amr_setup_plate(
                st,
                desc["plate_Nx"], desc["plate_Ny"], desc["plate_dx"], desc["plate_origin"],
                desc["plate_active"],
                desc["plate_mass_density"], desc["plate_stiffness_D"],
                desc["plate_alpha_M"], desc["plate_beta_K"],
                desc["n_active_plate"],
                desc["plate_active_flat_idx"],
                desc["plate_face_above_starts"], desc["plate_face_above_idx"], desc["plate_face_above_wgt"],
                desc["plate_face_below_starts"], desc["plate_face_below_idx"], desc["plate_face_below_wgt"],
                desc["plate_cell_above"], desc["plate_cell_below"],
            )
            n_active = int(desc["n_active_plate"])
            plate_idx = desc["plate_active_flat_idx"][:min(5, n_active)]
            wgt = np.ones(len(plate_idx), dtype=np.float32) / len(plate_idx)
            amr_inject_plate_force(st, len(plate_idx), plate_idx, wgt, 0.5)
            for _ in range(40):
                amr_step(st, 1)
            p = amr_get_pressure(st)
            body_h = 0.10
            exterior = grid.cell_centers[:, 2] > body_h + grid.min_dx
            return float(np.abs(p[exterior]).mean())
        finally:
            amr_destroy(st)

    p_open   = _run_and_get_ext_pressure(1.0)
    p_sealed = _run_and_get_ext_pressure(0.0)
    assert p_open > p_sealed, \
        f"Sealing the aperture did not reduce exterior pressure (open={p_open:.3e}, sealed={p_sealed:.3e})"


# ---------------------------------------------------------------------------
# Test 10 — Mic pressure + velocity sampling nonzero
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_mic_sampling_nonzero():
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import (
        CoEvolverStringDef, CoEvolverPickupDef, CoEvolverMicDef,
        coevolver_create_amr, coevolver_destroy,
        coevolver_pluck, coevolver_step_block, coevolver_get_mic,
    )

    grid = _make_grid(max_level=2)
    plate_Nx, plate_Ny, plate_dx, plate_origin = _plate_params(grid)
    cx, cy, hr = grid.soundhole
    bridge_xyz = np.array([[cx, cy - hr * 0.3, 0.0]], dtype=np.float64)

    desc = build_amr_coevolver_descriptor(
        grid, plate_Nx=plate_Nx, plate_Ny=plate_Ny, plate_dx=plate_dx,
        plate_origin=plate_origin, body_h=0.10,
        plate_mass_density=8.5, plate_stiffness_D=1.5,
        plate_alpha_M=5.0, plate_beta_K=1e-5, bridge_src_xyz=bridge_xyz,
    )

    str_def = CoEvolverStringDef()
    str_def.length = 0.65
    str_def.tension = 70.0
    str_def.linear_density = 5e-4
    str_def.damping = 2.0
    str_def.n_nodes = 48
    str_def.bridge_flat_idx = int(desc["bridge_plate_idx"][0])

    # Two mics: cardioid + velocity-sensitive.
    mic0 = CoEvolverMicDef()
    mic0.pos_x, mic0.pos_y, mic0.pos_z = float(cx), float(cy), 0.10 + 0.20
    mic0.axis_x, mic0.axis_y, mic0.axis_z = 0.0, 0.0, 1.0
    mic0.polar_a, mic0.polar_b = 0.5, 0.5

    mic1 = CoEvolverMicDef()
    mic1.pos_x, mic1.pos_y, mic1.pos_z = float(cx), float(cy), 0.10 + 0.20
    mic1.axis_x, mic1.axis_y, mic1.axis_z = 0.0, 0.0, 1.0
    mic1.polar_a, mic1.polar_b = 0.0, 1.0  # pure velocity

    st = coevolver_create_amr(1, [str_def], 0, [], 2, [mic0, mic1], desc, 44100.0, 8)
    assert st is not None
    try:
        coevolver_pluck(st, 0, 0.5, 0.001)
        n_samples = 512
        out = np.zeros((2, n_samples), dtype=np.float32)
        coevolver_step_block(st, n_samples, out)
        assert np.any(np.abs(out[0]) > 1e-12), "Mic 0 (cardioid) produced no output"
        assert np.any(np.abs(out[1]) > 1e-12), "Mic 1 (velocity) produced no output"
    finally:
        coevolver_destroy(st)


# ---------------------------------------------------------------------------
# Test 11 — Legacy uniform path regression
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_legacy_uniform_coevolver_still_works():
    """Ensure coevolver_create (uniform FDTD) still runs after AMR additions."""
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import (
        CoEvolverStringDef, CoEvolverMicDef,
        coevolver_create, coevolver_destroy,
        coevolver_pluck, coevolver_step_block,
    )

    str_def = CoEvolverStringDef()
    str_def.length = 0.65
    str_def.tension = 70.0
    str_def.linear_density = 5e-4
    str_def.damping = 2.0
    str_def.n_nodes = 48
    str_def.bridge_cell_flat = 0  # arbitrary for minimal test

    mic_def = CoEvolverMicDef()
    mic_def.pos_x, mic_def.pos_y, mic_def.pos_z = 0.1, 0.1, 0.3
    mic_def.axis_x, mic_def.axis_y, mic_def.axis_z = 0.0, 0.0, 1.0
    mic_def.polar_a, mic_def.polar_b = 1.0, 0.0

    Nx, Ny, Nz = 12, 10, 8
    dx = 0.02
    st = coevolver_create(
        1, [str_def], 0, [], 1, [mic_def],
        Nx, Ny, Nz, dx, np.zeros(3, dtype=np.float32),
        343.0, 1.21, 44100.0, 8,
    )
    assert st is not None, "coevolver_create returned NULL (legacy path broken)"
    try:
        coevolver_pluck(st, 0, 0.5, 0.001)
        out = np.zeros((1, 256), dtype=np.float32)
        rc = coevolver_step_block(st, 256, out)
        assert rc == 0
        assert np.any(np.abs(out) > 1e-12), "Legacy coevolver produced no audio"
    finally:
        coevolver_destroy(st)


# ---------------------------------------------------------------------------
# Test 12 — AMR construction fails with incomplete bridge mappings
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _c_ext_missing(),
    reason="_spectral_kernels C extension not built",
)
def test_amr_construction_fails_no_bridge():
    pytest.importorskip("_spectral_kernels")
    from _spectral_kernels import (
        CoEvolverStringDef, CoEvolverMicDef,
        coevolver_create_amr,
    )

    grid = _make_grid(max_level=2)
    plate_Nx, plate_Ny, plate_dx, plate_origin = _plate_params(grid)
    cx, cy, hr = grid.soundhole
    bridge_xyz = np.array([[cx, cy - hr * 0.3, 0.0]], dtype=np.float64)

    desc = build_amr_coevolver_descriptor(
        grid, plate_Nx=plate_Nx, plate_Ny=plate_Ny, plate_dx=plate_dx,
        plate_origin=plate_origin, body_h=0.10,
        plate_mass_density=8.5, plate_stiffness_D=1.5,
        plate_alpha_M=5.0, plate_beta_K=1e-5, bridge_src_xyz=bridge_xyz,
    )
    # Zero out bridge mapping — construction must fail loudly (return None / raise).
    desc["n_bridge_plate"] = 0
    desc["bridge_plate_idx"] = np.empty(0, dtype=np.int32)
    desc["bridge_plate_wgt"] = np.empty(0, dtype=np.float32)

    str_def = CoEvolverStringDef()
    str_def.length = 0.65
    str_def.tension = 70.0
    str_def.linear_density = 5e-4
    str_def.damping = 2.0
    str_def.n_nodes = 48
    str_def.bridge_flat_idx = 0

    mic_def = CoEvolverMicDef()
    mic_def.pos_x, mic_def.pos_y, mic_def.pos_z = float(cx), float(cy), 0.10 + 0.20
    mic_def.axis_x, mic_def.axis_y, mic_def.axis_z = 0.0, 0.0, 1.0
    mic_def.polar_a, mic_def.polar_b = 1.0, 0.0

    st = coevolver_create_amr(1, [str_def], 0, [], 1, [mic_def], desc, 44100.0, 8)
    assert st is None, "coevolver_create_amr should return NULL when bridge mapping is empty"


if __name__ == "__main__":
    # Quick self-check without pytest infrastructure.
    test_amr_classification_levels()
    print("test_amr_classification_levels: PASS")
    test_amr_topology_balance()
    print("test_amr_topology_balance: PASS")
    test_amr_cfl_stability()
    print("test_amr_cfl_stability: PASS")
    test_amr_aperture_openness()
    print("test_amr_aperture_openness: PASS")
    test_mic_conic_projection_levels()
    print("test_mic_conic_projection_levels: PASS")
    print("Pure-Python tests passed. C extension tests skipped (run via pytest).")
