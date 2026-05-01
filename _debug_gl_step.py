"""Debug: trace GL buffer values step by step."""
import numpy as np, os
os.environ['SPECTRAL_HEADLESS_BATCH'] = '1'

from test_amr_throughput import _make_tiny_grid
grid = _make_tiny_grid(8)

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.set_mode((1,1), DOUBLEBUF|OPENGL)

from OpenGL.GL import (
    glBindBuffer, glBufferSubData, glGetBufferSubData,
    GL_SHADER_STORAGE_BUFFER, glGenBuffers, glGetError, GL_NO_ERROR,
)

from acoustic_amr import AMRGLComputeBackend
b = AMRGLComputeBackend(grid, gradient_order=2)

def read_buf(buf, n, dtype=np.float32):
    out = np.empty(n, dtype=dtype)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, out)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return out

n = grid.n_cells
nf = grid.n_faces

print("=== Buffer IDs ===")
print(f"  _buf_pressure  = {b._buf_pressure}")
print(f"  _buf_velocity  = {b._buf_velocity}")
print(f"  _buf_inv_denom = {b._buf_inv_denom}")
print(f"  _buf_csr_weight= {b._buf_csr_weight}")
print(f"  _buf_p_damp    = {b._buf_p_damp}")

print("\n=== Initial buffer values ===")
print("  pressure :", read_buf(b._buf_pressure,  n))
print("  velocity :", read_buf(b._buf_velocity,  nf))
print("  inv_denom:", read_buf(b._buf_inv_denom, n))
print("  p_damp   :", read_buf(b._buf_p_damp,    n))
print("  csr_wgt  :", read_buf(b._buf_csr_weight, 14))  # 8 cells * 2 entries = 14 (7 faces * 2)

# Write pressure pulse at cell 0
init_p = np.zeros(n, dtype=np.float32)
init_p[0] = 1.0
glBindBuffer(GL_SHADER_STORAGE_BUFFER, b._buf_pressure)
glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, init_p.nbytes, init_p)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
print("\n=== After writing pressure[0]=1.0 ===")
print("  pressure :", read_buf(b._buf_pressure, n))

# Now manually run step 1 with instrumentation
from OpenGL.GL import (
    glUseProgram, glDispatchCompute, glMemoryBarrier, glBindBufferBase,
    GL_SHADER_STORAGE_BARRIER_BIT,
)

# Velocity step
L = 256
n_faces = b._n_faces
n_cells = b._n_cells

glUseProgram(b._prog_vel)
b._bind_base(b._buf_pressure,    0)
b._bind_base(b._buf_velocity,    1)
b._bind_base(b._buf_face_neg,    2)
b._bind_base(b._buf_face_pos,    3)
b._bind_base(b._buf_face_inv_dist, 4)
b._bind_base(b._buf_face_v_damp,   5)

from acoustic_amr import _uniform_f, _uniform_i
_uniform_f(b._prog_vel, "dt_over_rho", b._dt_over_rho)
_uniform_i(b._prog_vel, "n_faces",     n_faces)
_uniform_i(b._prog_vel, "plate_bc_enabled", 0)
print(f"\ndt_over_rho = {b._dt_over_rho:.6e}")
print(f"bulk        = {b._bulk:.6e}")

import math
glDispatchCompute(math.ceil(n_faces / L), 1, 1)
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

print("\n=== After velocity step ===")
print("  velocity :", read_buf(b._buf_velocity, nf))
print("  pressure :", read_buf(b._buf_pressure, n))  # should be unchanged

# Divergence+pressure step
glUseProgram(b._prog_div_pres)
b._bind_base(b._buf_velocity,    0)
b._bind_base(b._buf_pressure,    1)
b._bind_base(b._buf_csr_starts,  2)
b._bind_base(b._buf_csr_idx,     3)
b._bind_base(b._buf_csr_weight,  4)
b._bind_base(b._buf_inv_denom,   5)
b._bind_base(b._buf_p_damp,      6)
b._bind_base(b._buf_p_src_coeff, 7)
_uniform_f(b._prog_div_pres, "bulk",    b._bulk)
_uniform_i(b._prog_div_pres, "n_cells", n_cells)
glDispatchCompute(math.ceil(n_cells / L), 1, 1)
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

print("\n=== After div+pressure step ===")
print("  pressure :", read_buf(b._buf_pressure, n))

err = glGetError()
print(f"\nGL error code: {err} ({'none' if err == GL_NO_ERROR else 'ERROR!'})")
