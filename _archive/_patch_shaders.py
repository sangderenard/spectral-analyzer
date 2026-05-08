"""Patch acoustic_amr.py shader constants (Phases 1, 2, 3, 4, 5, 6)."""
import re

with open('acoustic_amr.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ── Phase 1 + 4 + 6: Replace _VELOCITY_UPDATE_GLSL with branchless order-8
# plus add new _VELOCITY_UPDATE_ORDER2_GLSL  ─────────────────────────────────
vel_start = content.index('_VELOCITY_UPDATE_GLSL = """')
div_start = content.index('\n_DIVERGENCE_CSR_GLSL = """')
old_vel_block = content[vel_start:div_start]

new_vel_block = r'''_VELOCITY_UPDATE_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* 8th-order (STENCIL_SW=4) Fornberg gradient for velocity update.
 * Phase 4: face_s_cells padding uses index 0 with coefficient 0 (branchless).
 * Phase 6: plate_owner[f] >= 0 triggers plate-BC early-out.
 */
layout(std430, binding = 0) buffer PressureBuf   { float pressure[];     };
layout(std430, binding = 1) buffer VelocityBuf   { float velocity[];     };
layout(std430, binding = 2) buffer StencilCells  { int   face_s_cells[]; };
layout(std430, binding = 3) buffer StencilCoeff  { float face_s_coeff[]; };
layout(std430, binding = 4) buffer FaceVDampBuf  { float face_v_damp[];  };
layout(std430, binding = 6) buffer PlateOwnerBuf { int   plate_owner[];  };
layout(std430, binding = 7) buffer PlateSignBuf  { float plate_sign[];   };
layout(std430, binding = 8) buffer PlateWgtBuf   { float plate_wgt[];    };
layout(std430, binding = 9) buffer ActiveIdxBuf  { int   active_idx[];   };
layout(std430, binding = 10) buffer PlateWBuf    { float plate_w[];      };
layout(std430, binding = 11) buffer PlateWpBuf   { float plate_w_prev[]; };

uniform float dt_over_rho;
uniform int   n_faces;
uniform float plate_inv_dt;
uniform int   plate_bc_enabled;

void main() {
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(n_faces)) return;

    if (plate_bc_enabled != 0) {
        int owner = plate_owner[f];
        if (owner >= 0) {
            int flat = active_idx[owner];
            float v_plt = (plate_w[flat] - plate_w_prev[flat]) * plate_inv_dt;
            velocity[f] = plate_sign[f] * v_plt * plate_wgt[f];
            return;
        }
    }

    const int base = int(f) * 8;
    float grad_p = 0.0;
    for (int k = 0; k < 8; ++k) {
        int ci = face_s_cells[base + k];
        grad_p += face_s_coeff[base + k] * pressure[ci];
    }
    velocity[f] = (velocity[f] - dt_over_rho * grad_p) * face_v_damp[f];
}
"""

_VELOCITY_UPDATE_ORDER2_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* 2nd-order central-difference gradient for velocity update.
 * Only two pressure cell gathers per face — no Fornberg stencil, no loop.
 * Phase 6: plate_owner[f] >= 0 triggers plate-BC early-out.
 */
layout(std430, binding = 0) buffer PressureBuf   { float pressure[];     };
layout(std430, binding = 1) buffer VelocityBuf   { float velocity[];     };
layout(std430, binding = 2) buffer FaceNegBuf    { int   face_neg[];     };
layout(std430, binding = 3) buffer FacePosBuf    { int   face_pos[];     };
layout(std430, binding = 4) buffer FaceInvDBuf   { float face_inv_d[];   };
layout(std430, binding = 5) buffer FaceVDampBuf  { float face_v_damp[];  };
layout(std430, binding = 6) buffer PlateOwnerBuf { int   plate_owner[];  };
layout(std430, binding = 7) buffer PlateSignBuf  { float plate_sign[];   };
layout(std430, binding = 8) buffer PlateWgtBuf   { float plate_wgt[];    };
layout(std430, binding = 9) buffer ActiveIdxBuf  { int   active_idx[];   };
layout(std430, binding = 10) buffer PlateWBuf    { float plate_w[];      };
layout(std430, binding = 11) buffer PlateWpBuf   { float plate_w_prev[]; };

uniform float dt_over_rho;
uniform int   n_faces;
uniform float plate_inv_dt;
uniform int   plate_bc_enabled;

void main() {
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(n_faces)) return;

    if (plate_bc_enabled != 0) {
        int owner = plate_owner[f];
        if (owner >= 0) {
            int flat = active_idx[owner];
            float v_plt = (plate_w[flat] - plate_w_prev[flat]) * plate_inv_dt;
            velocity[f] = plate_sign[f] * v_plt * plate_wgt[f];
            return;
        }
    }

    float grad_p = (pressure[face_pos[f]] - pressure[face_neg[f]]) * face_inv_d[f];
    velocity[f] = (velocity[f] - dt_over_rho * grad_p) * face_v_damp[f];
}
"""'''

content = content[:vel_start] + new_vel_block + content[div_start:]
print("Phase 1+4+6 velocity shaders: OK")

# ── Phase 2 + 3: Replace _DIVERGENCE_CSR_GLSL and _PRESSURE_UPDATE_GLSL
# with _DIVERGENCE_PRESSURE_GLSL (fused, pre-baked CSR weights) ────────────
div_start2 = content.index('_DIVERGENCE_CSR_GLSL = """')
plate_start = content.index('\n_PLATE_STEP_GLSL = """')
old_div_pres = content[div_start2:plate_start]

new_div_pres = r'''_DIVERGENCE_PRESSURE_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

/* Fused divergence + pressure update (Phases 2 and 3).
 * csr_face_weight[k] = ±face_flux_coef[face] pre-baked at setup time.
 * Eliminates the intermediate div_flux buffer, one dispatch, one barrier.
 *
 * p_new = p * P_damp - bulk * div/V * P_src_coeff
 * For non-PML cells P_damp=1, P_src_coeff=1 => standard leapfrog.
 */
layout(std430, binding = 0) buffer VelocityBuf    { float velocity[];        };
layout(std430, binding = 1) buffer PressureBuf    { float pressure[];        };
layout(std430, binding = 2) buffer CellStartsBuf  { int   cell_starts[];     };
layout(std430, binding = 3) buffer CsrIdxBuf      { int   csr_face_idx[];    };
layout(std430, binding = 4) buffer CsrWeightBuf   { float csr_face_weight[]; };
layout(std430, binding = 5) buffer InvDenomBuf    { float cell_inv_denom[];  };
layout(std430, binding = 6) buffer PDampBuf       { float P_damp[];          };
layout(std430, binding = 7) buffer PSrcBuf        { float P_src_coeff[];     };

uniform float bulk;
uniform int   n_cells;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= uint(n_cells)) return;

    float d = 0.0;
    int k0 = cell_starts[c];
    int k1 = cell_starts[c + 1];
    for (int k = k0; k < k1; ++k) {
        d += csr_face_weight[k] * velocity[csr_face_idx[k]];
    }
    pressure[c] = pressure[c] * P_damp[c]
                - bulk * d * cell_inv_denom[c] * P_src_coeff[c];
}
"""'''

content = content[:div_start2] + new_div_pres + content[plate_start:]
print("Phase 2+3 fused divergence-pressure shader: OK")

# ── Phase 5: Add L4 cache to _PLATE_STEP_GLSL ─────────────────────────────
# Replace the second 13-point biharmonic stencil with a cached L4wp read.
old_plate_step_start = content.index('_PLATE_STEP_GLSL = """')
plate_commit_start = content.index('\n_PLATE_COMMIT_GLSL = """')
old_plate_step = content[old_plate_step_start:plate_commit_start]

new_plate_step = r'''_PLATE_STEP_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PlateWBuf      { float plate_w[];       };
layout(std430, binding = 1) buffer PlateWpBuf     { float plate_w_prev[];  };
layout(std430, binding = 2) buffer PlateWnBuf     { float plate_w_new[];   };
layout(std430, binding = 3) buffer ExtForceBuf    { float ext_force[];     };
layout(std430, binding = 4) buffer ActiveIdxBuf   { int   active_idx[];    };
layout(std430, binding = 5) buffer PressureBuf    { float pressure[];      };
layout(std430, binding = 6) buffer CellAboveBuf   { int   cell_above[];    };
layout(std430, binding = 7) buffer CellBelowBuf   { int   cell_below[];    };
layout(std430, binding = 8) buffer PlateL4PrevBuf { float plate_L4_prev[]; };
layout(std430, binding = 9) buffer PlateL4NewBuf  { float plate_L4_new[];  };

uniform int   N_active;
uniform int   Nx;
uniform int   Ny;
uniform float dx2;       /* dx^2 */
uniform float dx4;       /* 1/dx^4 */
uniform float coeff_D0;  /* D*(1 + bK/dt) */
uniform float coeff_Dp;  /* D*(bK/dt)     */
uniform float damp_fwd;
uniform float damp_bwd;
uniform float dt2_inv_rh; /* dt^2 / (rho_h * damp_fwd) */

float W(int ii, int jj) {
    if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny) return 0.0;
    return plate_w[ii * Ny + jj];
}

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;

    int flat = active_idx[n];
    int i    = flat / Ny;
    int j    = flat % Ny;

    /* Acoustic load */
    float F_acou = 0.0;
    int ca = cell_above[n];
    int cb = cell_below[n];
    if (ca >= 0) F_acou += pressure[ca];
    if (cb >= 0) F_acou -= pressure[cb];
    F_acou *= dx2;

    /* 13-point biharmonic of current w */
    float L4w =
          W(i-2,j) + W(i+2,j) + W(i,j-2) + W(i,j+2)
        + 2.0*(W(i-1,j-1)+W(i-1,j+1)+W(i+1,j-1)+W(i+1,j+1))
        - 8.0*(W(i-1,j)+W(i+1,j)+W(i,j-1)+W(i,j+1))
        + 20.0*W(i,j);
    L4w *= dx4;

    /* Phase 5: read cached L4(w_prev) instead of recomputing it */
    float L4wp = plate_L4_prev[flat];

    float w_c = plate_w[flat];
    float w_p = plate_w_prev[flat];
    float rhs = F_acou + ext_force[flat] - coeff_D0*L4w + coeff_Dp*L4wp;
    plate_w_new[flat] = (damp_bwd*(2.0*w_c - w_p) + dt2_inv_rh*rhs) / damp_fwd;

    /* Cache L4w for use as L4wp next step */
    plate_L4_new[flat] = L4w;
}
"""'''

content = content[:old_plate_step_start] + new_plate_step + content[plate_commit_start:]
print("Phase 5 plate step with L4 cache: OK")

# ── Phase 5: Update _PLATE_COMMIT_GLSL to rotate L4 buffers ──────────────
old_commit_start = content.index('_PLATE_COMMIT_GLSL = """')
plate_bc_start = content.index('\n_PLATE_BC_GLSL = """')
old_commit = content[old_commit_start:plate_bc_start]

new_commit = r'''_PLATE_COMMIT_GLSL = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PlateWBuf      { float plate_w[];      };
layout(std430, binding = 1) buffer PlateWpBuf     { float plate_w_prev[]; };
layout(std430, binding = 2) buffer PlateWnBuf     { float plate_w_new[];  };
layout(std430, binding = 3) buffer ExtForceBuf    { float ext_force[];    };
layout(std430, binding = 4) buffer ActiveIdxBuf   { int   active_idx[];   };
layout(std430, binding = 8) buffer PlateL4PrevBuf { float plate_L4_prev[];};
layout(std430, binding = 9) buffer PlateL4NewBuf  { float plate_L4_new[]; };

uniform int N_active;

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= uint(N_active)) return;
    int flat = active_idx[n];
    plate_w_prev[flat] = plate_w[flat];
    plate_w[flat]      = plate_w_new[flat];
    ext_force[flat]    = 0.0;
    /* Phase 5: rotate L4 cache so next step reads correct L4(w_prev) */
    plate_L4_prev[flat] = plate_L4_new[flat];
}
"""'''

content = content[:old_commit_start] + new_commit + content[plate_bc_start:]
print("Phase 5 plate commit with L4 rotation: OK")

with open('acoustic_amr.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("All shader replacements written successfully.")
