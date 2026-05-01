"""test_amr_stability.py — minimal AMR coevolver stability stress-test.

Builds a tiny uniform Cartesian box (1500 cells, ~4100 faces) with a
Kirchhoff plate and one guitar-like string, plucks it, and runs the
simulation in blocks while reporting peak pressure.  Loads in seconds
so you can iterate quickly on solver parameters or reproduce instability.

Usage
-----
    python test_amr_stability.py
    python test_amr_stability.py --blocks 500 --block-samples 256 --amplitude 0.05
    python test_amr_stability.py --gradient-order 8   # use Fornberg stencil

Flags
-----
  --blocks N          How many step-blocks to run            (default 200)
  --block-samples N   Audio samples per block                (default 256)
  --amplitude A       Initial string pluck amplitude (m)     (default 0.01)
  --gradient-order N  Velocity gradient order: 2 or 8        (default 8)
  --no-pluck          Skip the pluck; useful to check idle stability
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Physics extension
# ---------------------------------------------------------------------------
try:
    from _spectral_kernels import AcousticCoEvolver
    _HAS_PHYSICS = True
except ImportError:
    print(
        "[test_amr_stability] _spectral_kernels unavailable — rebuild:\n"
        "  cmake -S csrc -B csrc_build && cmake --build csrc_build --config Release"
    )
    sys.exit(1)

from acoustic_amr import (
    AMR_AIR,
    AMR_PLATE,
    AcousticAMRGrid,
    BORDER_ANECHOIC,
    BorderConditionSpec,
    build_amr_coevolver_descriptor,
)

# ---------------------------------------------------------------------------
# Grid builder — uniform Cartesian box
# ---------------------------------------------------------------------------

def _cidx(ix: int, iy: int, iz: int, Ny: int, Nz: int) -> int:
    return ix * (Ny * Nz) + iy * Nz + iz


def build_box_grid(
    Nx: int = 15,
    Ny: int = 10,
    Nz: int = 10,
    dx: float = 0.02,
    plate_iz: int = 5,
) -> AcousticAMRGrid:
    """Uniform Nx×Ny×Nz Cartesian grid.

    Cells at iz==plate_iz → AMR_PLATE, all others → AMR_AIR.
    face_neg is always the cell with the lower coordinate value so that
    (cc[face_pos] - cc[face_neg]) · axis > 0 for every face — the invariant
    the fixed amr_build_dir_neighbors relies on.
    """
    nc = Nx * Ny * Nz
    ix_g, iy_g, iz_g = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
    )
    ix_f = ix_g.ravel()
    iy_f = iy_g.ravel()
    iz_f = iz_g.ravel()

    cell_centers = np.stack(
        [(ix_f + 0.5) * dx, (iy_f + 0.5) * dx, (iz_f + 0.5) * dx], axis=1
    ).astype(np.float64)

    cell_types = np.where(iz_f == plate_iz, AMR_PLATE, AMR_AIR).astype(np.uint8)
    cell_volumes      = np.full(nc, dx ** 3, dtype=np.float64)
    open_vol_frac     = np.ones(nc,          dtype=np.float64)
    cell_levels       = np.zeros(nc,         dtype=np.int16)
    cell_half_sizes   = np.full(nc, dx / 2,  dtype=np.float64)
    importance        = np.ones(nc,          dtype=np.uint8)

    # Build faces — three sweeps, one per axis direction.
    fn_lists: list[np.ndarray] = []
    fp_lists: list[np.ndarray] = []
    fa_lists: list[np.ndarray] = []

    for axis in range(3):
        if axis == 0:
            shape_n = (Nx - 1, Ny, Nz)
            def make_neg(a, b, c): return _cidx(a,   b, c, Ny, Nz)
            def make_pos(a, b, c): return _cidx(a+1, b, c, Ny, Nz)
        elif axis == 1:
            shape_n = (Nx, Ny - 1, Nz)
            def make_neg(a, b, c): return _cidx(a, b,   c, Ny, Nz)
            def make_pos(a, b, c): return _cidx(a, b+1, c, Ny, Nz)
        else:
            shape_n = (Nx, Ny, Nz - 1)
            def make_neg(a, b, c): return _cidx(a, b, c,   Ny, Nz)
            def make_pos(a, b, c): return _cidx(a, b, c+1, Ny, Nz)

        nf_ax = shape_n[0] * shape_n[1] * shape_n[2]
        fn = np.empty(nf_ax, dtype=np.int32)
        fp = np.empty(nf_ax, dtype=np.int32)
        k = 0
        for a in range(shape_n[0]):
            for b in range(shape_n[1]):
                for c in range(shape_n[2]):
                    fn[k] = make_neg(a, b, c)
                    fp[k] = make_pos(a, b, c)
                    k += 1
        fn_lists.append(fn)
        fp_lists.append(fp)
        fa_lists.append(np.full(nf_ax, axis, dtype=np.int32))

    face_neg  = np.concatenate(fn_lists)
    face_pos  = np.concatenate(fp_lists)
    face_axis = np.concatenate(fa_lists)
    nf = len(face_neg)

    face_area          = np.full(nf, dx ** 2, dtype=np.float64)
    face_open_fraction = np.ones(nf,          dtype=np.float64)
    face_distance      = np.full(nf, dx,      dtype=np.float64)

    bounds_min = np.zeros(3,                           dtype=np.float64)
    bounds_max = np.array([Nx * dx, Ny * dx, Nz * dx], dtype=np.float64)

    return AcousticAMRGrid(
        base_dx=dx,
        min_dx=dx,
        max_refinement_level=0,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        cell_centers=cell_centers,
        cell_half_sizes=cell_half_sizes,
        cell_levels=cell_levels,
        cell_types=cell_types,
        importance=importance,
        cell_volumes=cell_volumes,
        open_volume_fraction=open_vol_frac,
        face_cell_neg=face_neg,
        face_cell_pos=face_pos,
        face_axis=face_axis,
        face_area=face_area,
        face_open_fraction=face_open_fraction,
        face_distance=face_distance,
        soundhole=(Nx * dx * 0.5, Ny * dx * 0.5, 0.0),  # zero radius — no hole
        metadata={},
    )


# ---------------------------------------------------------------------------
# Coevolver builder
# ---------------------------------------------------------------------------

def build_test_coevolver(
    gradient_order: int = 8,
    sample_rate: float = 44100.0,
) -> AcousticCoEvolver:
    """Build a minimal AMR AcousticCoEvolver from a uniform box grid.

    Grid   : 15×10×10, dx=2cm  →  1500 cells, ~4100 faces
    Plate  : iz=5  (body_h = 0.11 m),  15×10 nodes at dx=2cm
    String : 1 string, E-like (70 N, 0.8 g/m), horizontal above plate
    Mic    : inside cavity, near centre
    """
    dx      = 0.02        # cell / plate node spacing (m)
    Nx, Ny, Nz = 15, 10, 10
    plate_iz   = 5
    body_h     = (plate_iz + 0.5) * dx   # = 0.11 m

    # Plate spans x ∈ [0, Nx*dx], y ∈ [0, Ny*dx]
    # Nodes start at cell-centre offset so each node sits exactly on a cell centre.
    plate_origin = np.array([0.5 * dx, 0.5 * dx, body_h], dtype=np.float32)
    plate_Nx, plate_Ny = Nx, Ny
    plate_dx = dx

    # Centre of the plate in world XY
    cx = 0.5 * Nx * dx   # 0.15 m
    cy = 0.5 * Ny * dx   # 0.10 m

    # Bridge saddle slightly toward the tail end of the plate (like real guitar)
    bridge_xyz = np.array([[cx, cy * 0.65, body_h]], dtype=np.float32)

    print(f"[test_amr] building box grid {Nx}×{Ny}×{Nz} dx={dx*100:.0f}cm "
          f"→ {Nx*Ny*Nz} cells, body_h={body_h:.3f}m", flush=True)
    t0 = time.perf_counter()
    grid = build_box_grid(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, plate_iz=plate_iz)
    print(f"[test_amr]   grid built in {time.perf_counter()-t0:.2f}s  "
          f"({grid.n_cells} cells, {grid.n_faces} faces)", flush=True)

    # Spruce-like plate material
    plate_mass_density = 1.8    # kg/m²  (rho_s*h, spruce 600 kg/m³ × 3 mm)
    plate_stiffness_D  = 0.30   # N·m    (bending stiffness)
    plate_alpha_M      = 2.0    # s⁻¹   (Rayleigh mass damping)
    plate_beta_K       = 1e-5   # s     (Rayleigh stiffness damping)

    t1 = time.perf_counter()
    desc = build_amr_coevolver_descriptor(
        grid=grid,
        plate_Nx=plate_Nx,
        plate_Ny=plate_Ny,
        plate_dx=plate_dx,
        plate_origin=plate_origin,
        body_h=body_h,
        plate_mass_density=plate_mass_density,
        plate_stiffness_D=plate_stiffness_D,
        plate_alpha_M=plate_alpha_M,
        plate_beta_K=plate_beta_K,
        bridge_src_xyz=bridge_xyz,
        c=343.0,
        rho_air=1.21,
        gradient_order=gradient_order,
        border_spec=BorderConditionSpec(mode=BORDER_ANECHOIC, sigma_order=3.0),
        n_pml=0,
    )
    print(f"[test_amr]   descriptor built in {time.perf_counter()-t1:.2f}s  "
          f"n_active_plate={desc['n_active_plate']}", flush=True)

    # Single guitar-E-like string, horizontal, just above the plate
    string_z = body_h + dx           # one cell above the plate
    n_segs   = 8
    xs       = np.linspace(cx - 0.06, cx + 0.06, n_segs + 1, dtype=np.float32)
    path_xyz = np.stack(
        [xs, np.full(n_segs + 1, cy, dtype=np.float32),
         np.full(n_segs + 1, string_z, dtype=np.float32)], axis=1
    )
    string_defs = [
        {
            "path_xyz":        path_xyz,
            "tension_N":       70.0,
            "linear_mass_kgm": 0.0008,
            "damping":         0.002,
            "stiffness_EI":    1e-5,
        }
    ]

    # No magnetic pickups — acoustic only
    pickup_defs = []

    # One omnidirectional mic inside the cavity
    mic_defs = [
        {
            "pos":  np.array([cx, cy, body_h + 2 * dx], dtype=np.float32),
            "gain": 1.0,
        }
    ]

    t2 = time.perf_counter()
    co = AcousticCoEvolver.from_amr_desc(
        desc, string_defs, pickup_defs, mic_defs,
        sample_rate=sample_rate,
        modal_stride=16,
    )
    print(f"[test_amr]   coevolver created in {time.perf_counter()-t2:.2f}s", flush=True)
    print(f"[test_amr]   total setup: {time.perf_counter()-t0:.2f}s", flush=True)
    return co


# ---------------------------------------------------------------------------
# Pytest-compatible test functions
# ---------------------------------------------------------------------------

def _run_blocks(co: AcousticCoEvolver, n_blocks: int, block_samples: int) -> float:
    """Step co-evolver for n_blocks and return peak pressure seen (Pa)."""
    p_max_seen = 0.0
    for _ in range(n_blocks):
        co.step(block_samples)
        p_field = co.get_pressure_field()
        p_max_seen = max(p_max_seen, float(np.max(np.abs(p_field))))
    return p_max_seen


def test_coevolver_builds() -> None:
    """Coevolver creation succeeds and returns a live object."""
    co = build_test_coevolver()
    assert co is not None


def test_idle_stays_silent() -> None:
    """Without excitation, pressure stays at zero."""
    co = build_test_coevolver()
    p_max = _run_blocks(co, n_blocks=20, block_samples=256)
    assert p_max < 1e-6, f"idle pressure {p_max:.3g} Pa — expected near zero"


def test_pluck_stable() -> None:
    """A realistic pluck stays below the divergence threshold for 200 blocks."""
    co = build_test_coevolver()
    co.pluck_string(0, 0.15, 0.01)
    p_max = _run_blocks(co, n_blocks=200, block_samples=256)
    assert np.isfinite(p_max), "pressure diverged to NaN/Inf"
    assert p_max < 2000.0, f"pressure exceeded CFL limit: {p_max:.3g} Pa"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks",         type=int,   default=200,  help="step blocks to run")
    ap.add_argument("--block-samples",  type=int,   default=256,  help="audio samples per block")
    ap.add_argument("--amplitude",      type=float, default=0.01, help="pluck amplitude (m)")
    ap.add_argument("--gradient-order", type=int,   default=8,    choices=[2, 8])
    ap.add_argument("--no-pluck",       action="store_true",      help="skip excitation")
    args = ap.parse_args()

    co = build_test_coevolver(gradient_order=args.gradient_order)

    if not args.no_pluck:
        print(f"\n[test_amr] plucking string 0 at pos=0.15 amplitude={args.amplitude}", flush=True)
        co.pluck_string(0, 0.15, args.amplitude)

    print(
        f"\n[test_amr] running {args.blocks} blocks × {args.block_samples} samples "
        f"(gradient_order={args.gradient_order}) ...\n",
        flush=True,
    )
    print(f"{'block':>6}  {'time_s':>7}  {'p_max_Pa':>10}  {'mic_rms':>10}  status")
    print("-" * 54)

    t_run = time.perf_counter()

    for blk in range(args.blocks):
        try:
            co.step(args.block_samples)
        except RuntimeError as exc:
            msg = str(exc)
            elapsed = time.perf_counter() - t_run
            print(f"\n[test_amr] EXCEPTION at block {blk} ({elapsed:.1f}s): {msg}")
            sys.exit(2)

        elapsed = time.perf_counter() - t_run

        # Sample mic output and compute RMS
        try:
            mic_out = co.get_mic_output(0, args.block_samples)
            mic_rms = float(np.sqrt(np.mean(mic_out.astype(np.float64) ** 2)))
        except Exception:
            mic_rms = float("nan")

        # Print every 10 blocks, plus the first and last
        if blk == 0 or (blk + 1) % 10 == 0 or blk == args.blocks - 1:
            # Use the uniform pressure field scatter to get p_max
            try:
                p_field = co.get_pressure_field()
                p_max = float(np.max(np.abs(p_field)))
                stable = "OK" if np.isfinite(p_max) and p_max < 2000.0 else "UNSTABLE"
            except Exception:
                p_max = float("nan")
                stable = "ERROR"

            print(
                f"{blk+1:>6}  {elapsed:>7.2f}  {p_max:>10.3g}  {mic_rms:>10.3g}  {stable}",
                flush=True,
            )

            if stable != "OK":
                print(f"\n[test_amr] stopping — simulation diverged at block {blk+1}")
                break

    total = time.perf_counter() - t_run
    audio_s = args.blocks * args.block_samples / 44100.0
    print(f"\n[test_amr] done: {total:.2f}s wall, {audio_s:.2f}s audio, "
          f"realtime factor {audio_s/max(total, 1e-9):.2f}×")


if __name__ == "__main__":
    main()
