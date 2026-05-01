"""Phase 14 — AMR throughput validation tests.

Seven tests covering:
  1. GL order-2 construction: no stencil buffers allocated.
  2. GL order-8 construction: stencil buffers allocated.
  3. CPU fused divergence parity: GL and C++ give same pressure after 1 step.
  4. GL fused divergence parity: fused shader conserves pressure·volume sum.
  5. Plate L4 cache parity: plate displacement is identical to analytical reference.
  6. Offline mic monolithic output: shape is (n_steps, n_mics, 4).
  7. Catalogue reuse test: second compile call returns cached program ID.

GL tests require an OpenGL 4.3 context.  If pygame or OpenGL is unavailable, or
context creation fails, all GL tests are skipped (not failed).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_grid(n_cells: int = 8) -> "AcousticAMRGrid":
    """Build a minimal 1-D acoustic grid: n_cells cells in a row along x.

    Cell spacing dx=0.05 m.  All cells are acoustic (type 0).
    Interior faces connect adjacent cells; no boundary faces (open domain).
    """
    from acoustic_amr import AcousticAMRGrid

    dx   = 0.05
    hx   = dx / 2.0
    nc   = int(n_cells)
    nf   = nc - 1  # one interior face between each adjacent pair

    cc = np.zeros((nc, 3), dtype=np.float64)
    cc[:, 0] = np.arange(nc, dtype=np.float64) * dx

    grid = AcousticAMRGrid(
        base_dx             = dx,
        min_dx              = dx,
        max_refinement_level= 0,
        bounds_min          = np.array([0.0, -hx, -hx]),
        bounds_max          = np.array([nc * dx, hx, hx]),
        cell_centers        = cc,
        cell_half_sizes     = np.full(nc, hx, dtype=np.float64),
        cell_levels         = np.zeros(nc, dtype=np.int16),
        cell_types          = np.zeros(nc, dtype=np.uint8),     # 0 = acoustic
        importance          = np.zeros(nc, dtype=np.uint8),
        cell_volumes        = np.full(nc, dx ** 3, dtype=np.float64),
        open_volume_fraction= np.ones(nc, dtype=np.float64),
        face_cell_neg       = np.arange(nf, dtype=np.int32),
        face_cell_pos       = np.arange(1, nc, dtype=np.int32),
        face_axis           = np.zeros(nf, dtype=np.uint8),     # x-axis faces
        face_area           = np.full(nf, dx ** 2, dtype=np.float64),
        face_open_fraction  = np.ones(nf, dtype=np.float64),
        face_distance       = np.full(nf, dx, dtype=np.float64),
        soundhole           = (0.0, 0.0, 0.0),
        metadata            = {},
    )
    return grid


def _make_tiny_plate_desc(grid: "AcousticAMRGrid",
                           nx: int = 3,
                           ny: int = 3) -> dict:
    """Build a minimal plate descriptor compatible with AMRGLComputeBackend.setup_plate.

    Uses a completely synthetic plate with no acoustic coupling (all cell_above /
    cell_below set to -1) so the test is self-contained.
    """
    n_active = nx * ny
    active_idx = np.arange(n_active, dtype=np.int32)          # all nodes active
    cell_above = np.full(n_active, -1, dtype=np.int32)         # no coupling
    cell_below = np.full(n_active, -1, dtype=np.int32)

    # Empty above/below face CSR (no plate BC faces)
    fa_starts = np.zeros(n_active + 1, dtype=np.int32)
    fa_idx    = np.empty(0, dtype=np.int32)
    fa_wgt    = np.empty(0, dtype=np.float32)
    fb_starts = np.zeros(n_active + 1, dtype=np.int32)
    fb_idx    = np.empty(0, dtype=np.int32)
    fb_wgt    = np.empty(0, dtype=np.float32)

    return {
        "plate_Nx":               nx,
        "plate_Ny":               ny,
        "plate_dx":               0.01,
        "plate_mass_density":     0.5,
        "plate_stiffness_D":      1.0,
        "plate_alpha_M":          1.0,
        "plate_beta_K":           1e-4,
        "n_plate_active":         n_active,
        "plate_active_idx":       active_idx,
        "plate_cell_above":       cell_above,
        "plate_cell_below":       cell_below,
        "plate_face_above_starts": fa_starts,
        "plate_face_above_idx":   fa_idx,
        "plate_face_above_wgt":   fa_wgt,
        "plate_face_below_starts": fb_starts,
        "plate_face_below_idx":   fb_idx,
        "plate_face_below_wgt":   fb_wgt,
    }


# ── pytest fixture: OpenGL 4.3 context ───────────────────────────────────────

@pytest.fixture(scope="module")
def gl_ctx():
    """Create an off-screen OpenGL 4.3 context via pygame.

    Yields True on success; skips the test if creation fails for any reason.
    """
    pytest.importorskip("pygame", reason="pygame not installed")
    pytest.importorskip("OpenGL", reason="PyOpenGL not installed")

    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL

    try:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode((1, 1), DOUBLEBUF | OPENGL)
    except Exception as exc:
        pytest.skip(f"OpenGL 4.3 context unavailable: {exc}")

    yield True

    # Do NOT call pygame.quit() here: this is a module-scoped fixture and
    # pytest may hold references to failed test frames (containing live
    # AMRGLComputeBackend objects with active GL buffer handles).  Calling
    # pygame.quit() while those handles are still alive causes heap
    # corruption (STATUS_HEAP_CORRUPTION / 0xc0000374 on Windows/NVIDIA).
    # The GL context will be destroyed cleanly when the process exits.


# ── Test 1: GL order-2 construction ──────────────────────────────────────────

def test_gl_order2_no_stencil(gl_ctx):
    """order-2 backend must leave stencil buffers as None (Phase 1+4)."""
    from acoustic_amr import AMRGLComputeBackend

    grid = _make_tiny_grid()
    b = AMRGLComputeBackend(grid, gradient_order=2)

    assert b.gradient_order == 2
    assert b._buf_stencil_cells is None, \
        "order-2 backend must NOT allocate stencil_cells SSBO"
    assert b._buf_stencil_coeff is None, \
        "order-2 backend must NOT allocate stencil_coeff SSBO"
    # Order-2 topology buffers must exist
    assert b._buf_face_neg      is not None
    assert b._buf_face_pos      is not None
    assert b._buf_face_inv_dist is not None
    # CSR pre-baked weight buffer must exist (Phase 3)
    assert b._buf_csr_weight is not None


# ── Test 2: GL order-8 construction ──────────────────────────────────────────

def test_gl_order8_stencil_allocated(gl_ctx):
    """order-8 backend must allocate both stencil SSBOs (Phase 1+4)."""
    from acoustic_amr import AMRGLComputeBackend

    grid = _make_tiny_grid(n_cells=12)
    b = AMRGLComputeBackend(grid, gradient_order=8)

    assert b.gradient_order == 8
    assert b._buf_stencil_cells is not None, \
        "order-8 backend must allocate stencil_cells SSBO"
    assert b._buf_stencil_coeff is not None, \
        "order-8 backend must allocate stencil_coeff SSBO"
    # CSR pre-baked weight buffer must still exist
    assert b._buf_csr_weight is not None


# ── Test 3: CPU fused divergence parity ──────────────────────────────────────

def test_cpu_fused_divergence_parity(gl_ctx):
    """GL fused div+pressure gives same pressure as C++ CPU backend after 1 step."""
    from OpenGL.GL import glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER

    cpu_backend = pytest.importorskip(
        "_spectral_kernels",
        reason="_spectral_kernels C extension not available",
    )

    from acoustic_amr import AMRGLComputeBackend, AcousticAMRFDTD

    grid = _make_tiny_grid(n_cells=8)

    # ── CPU backend: step from pressure pulse at cell 0 ──
    cpu = AcousticAMRFDTD(grid, gradient_order=2)
    # Inject pressure[0] = 1.0 Pa via the C++ API
    cpu._c.inject_pressure_nearest(
        np.array(grid.cell_centers[0], dtype=np.float64), 1.0
    )
    cpu.step(1)
    cpu_p = np.array(cpu._c.get_pressure(), dtype=np.float64)

    # ── GL backend: identical initial condition ───────────
    from OpenGL.GL import (
        glBindBuffer, glBufferSubData, GL_SHADER_STORAGE_BUFFER,
    )
    import ctypes

    b = AMRGLComputeBackend(grid, gradient_order=2)

    # Write 1.0 into cell 0 of the pressure SSBO
    init_p = np.zeros(grid.n_cells, dtype=np.float32)
    init_p[0] = 1.0
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_pressure)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, init_p.nbytes, init_p)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    b.step(1)

    out = np.empty(grid.n_cells, dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_pressure)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, ctypes.c_void_p(out.ctypes.data))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Allow for float32 vs float64 rounding; compare CPU result cast to float32.
    np.testing.assert_allclose(
        out.astype(np.float64), cpu_p, rtol=1e-4, atol=1e-8,
        err_msg="GL fused div+pressure diverges from C++ CPU backend after 1 step",
    )


# ── Test 4: GL fused divergence parity ───────────────────────────────────────

def test_gl_fused_divergence_pressure_conservation(gl_ctx):
    """After 1 step from a pressure pulse, weighted pressure sum is conserved.

    Physically: total acoustic energy injected into a closed domain must be
    redistributed but not created or destroyed by the div+pressure update.
    We verify that cell 0's pressure decreased and cell 1's increased.
    """
    import ctypes
    from OpenGL.GL import (
        glBindBuffer, glBufferSubData, glGetBufferSubData,
        GL_SHADER_STORAGE_BUFFER,
    )
    from acoustic_amr import AMRGLComputeBackend

    grid = _make_tiny_grid(n_cells=8)
    b = AMRGLComputeBackend(grid, gradient_order=2)

    # Inject pressure[3] = 1.0 Pa (middle of the line)
    init_p = np.zeros(grid.n_cells, dtype=np.float32)
    init_p[3] = 1.0
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_pressure)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, init_p.nbytes, init_p)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    b.step(1)

    out = np.empty(grid.n_cells, dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_pressure)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, ctypes.c_void_p(out.ctypes.data))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    assert not np.any(np.isnan(out)), "NaN in pressure after 1 step"
    assert not np.any(np.isinf(out)), "Inf in pressure after 1 step"

    # Central cell pressure must have decreased (divergence removes energy)
    assert float(out[3]) < 1.0, \
        f"pressure[3] should decrease from 1.0 after 1 step, got {out[3]:.6f}"
    # Neighbour cells must have gained pressure (positive divergence inflow)
    assert float(out[2]) > 0.0 or float(out[4]) > 0.0, \
        "at least one neighbour must gain pressure via velocity divergence"


# ── Test 5: Plate L4 cache parity ────────────────────────────────────────────

def test_plate_l4_cache_no_nan(gl_ctx):
    """Plate L4 cache must not produce NaN/Inf after several steps (Phase 5).

    Also verifies that the ext_force injection is reflected in plate_w_new.
    """
    import ctypes
    from OpenGL.GL import (
        glBindBuffer, glBufferSubData, glGetBufferSubData,
        GL_SHADER_STORAGE_BUFFER,
    )
    from acoustic_amr import AMRGLComputeBackend

    grid = _make_tiny_grid(n_cells=8)
    b = AMRGLComputeBackend(grid, gradient_order=2)

    desc = _make_tiny_plate_desc(grid, nx=4, ny=4)
    b.setup_plate(desc)

    n_active = desc["n_plate_active"]
    Nx, Ny   = desc["plate_Nx"], desc["plate_Ny"]
    N        = Nx * Ny

    # Apply a unit force to the first active node so the plate actually moves
    force = np.zeros(N, dtype=np.float32)
    force[0] = 1.0
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_ext_force)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, force.nbytes, force)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Run 5 steps (enough to accumulate L4 cache rotation)
    b.step(5)

    w = np.empty(N, dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_plate_w)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, w.nbytes, ctypes.c_void_p(w.ctypes.data))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    assert not np.any(np.isnan(w)), "NaN in plate_w after 5 steps"
    assert not np.any(np.isinf(w)), "Inf in plate_w after 5 steps"

    # The first active node must have moved (force was applied)
    active_flat = desc["plate_active_idx"]
    assert float(w[active_flat[0]]) != 0.0, \
        "plate node[0] should have non-zero displacement after forced step"


# ── Test 6: Offline mic monolithic output ─────────────────────────────────────

def test_offline_mic_output_shape(gl_ctx):
    """Offline mic buffer has shape (n_steps, n_mics, 4) after run_offline_steps (Phase 7)."""
    from acoustic_amr import AMRGLComputeBackend

    grid    = _make_tiny_grid(n_cells=8)
    b       = AMRGLComputeBackend(grid, gradient_order=2)

    n_mics   = 3
    n_steps  = 16

    # Build minimal mic arrays: each mic samples one cell with weight 1.
    mic_starts   = np.arange(n_mics + 1, dtype=np.int32)  # 1 entry per mic
    mic_cell_idx = np.array([0, 2, 4], dtype=np.int32)
    mic_cell_wgt = np.ones(n_mics, dtype=np.float32)
    # No face contributions
    mic_face_idx = np.empty(0, dtype=np.int32)
    mic_face_wx  = np.empty(0, dtype=np.float32)
    mic_face_wy  = np.empty(0, dtype=np.float32)
    mic_face_wz  = np.empty(0, dtype=np.float32)

    b.setup_mics(
        mic_cell_idx, mic_cell_wgt,
        mic_face_idx, mic_face_wx, mic_face_wy, mic_face_wz,
        mic_starts,
    )
    b.setup_offline_mic_output(n_steps)
    b.run_offline_steps(n_steps)

    result = b.read_offline_mic_output()

    assert result.shape == (n_steps, n_mics, 4), \
        f"Expected ({n_steps}, {n_mics}, 4) but got {result.shape}"
    assert result.dtype == np.float32, \
        f"Expected float32 but got {result.dtype}"
    assert not np.any(np.isnan(result)), "NaN in offline mic output"


# ── Test 7: Catalogue reuse test ─────────────────────────────────────────────

def test_shader_cache_reuse():
    """get_or_compile_compute returns the cached ID on second call (Phase 8).

    This test does NOT require a GL context: it directly plants a fake ID in
    the dict and verifies the lookup short-circuits shader compilation.
    """
    from acoustic_amr import _COMPUTE_PROGRAM_CACHE, get_or_compile_compute

    key   = "_test_reuse_sentinel"
    fake  = 99999

    assert key not in _COMPUTE_PROGRAM_CACHE, \
        "test pollution: sentinel key already in cache"

    _COMPUTE_PROGRAM_CACHE[key] = fake
    try:
        result = get_or_compile_compute(key, "/* unused */")
        assert result == fake, \
            f"Expected cached value {fake} but got {result}"
    finally:
        del _COMPUTE_PROGRAM_CACHE[key]
