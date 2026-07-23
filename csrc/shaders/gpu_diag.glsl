/**
 * gpu_diag.glsl  —  Shader-side diagnostic write helper  (GLSL 430).
 *
 * Opt-in design: no existing shader is modified.  To enable in a shader:
 *
 *   Step 1 — pick a free SSBO binding slot for that shader:
 *       #define DIAG_BINDING  9      // must not collide with existing bindings
 *
 *   Step 2 — declare the pass ID:
 *       #define DIAG_PASS_ID  GPU_DIAG_PASS_T5
 *
 *   Step 3 — include this file:
 *       #include "gpu_diag.glsl"
 *
 *   Step 4 — call at fault sites (macros require both DIAG_BINDING + DIAG_PASS_ID):
 *       if (idx >= n_items)  { DIAG_OOB(idx);  return; }
 *       if (isnan(val))      { DIAG_NAN(0u);   return; }
 *       if (denom == 0.0)    { DIAG_ZDIV(0u);  return; }
 *
 * C++ reads the buffer after each dispatch via gpu_diag_flush_ssbo() (see
 * gpu_diag.h for the layout constants).  Zero the buffer before each dispatch
 * by uploading GPU_DIAG_BUF_BYTES of zeros to the binding.
 *
 * DiagBuf SSBO layout  (all uint32, std430)
 * ──────────────────────────────────────────
 *   [0]              entry_count       (atomicAdd counter; 0 = clean)
 *   [1]              overflow_flags    (GPU_DIAG_ERR_OVERFLOW set if cap hit)
 *   [2..3]           reserved
 *   [4 + i*8 + 0]   pass_id           (GPU_DIAG_PASS_*)
 *   [4 + i*8 + 1]   wg_x              (gl_WorkGroupID.x)
 *   [4 + i*8 + 2]   wg_y              (gl_WorkGroupID.y)
 *   [4 + i*8 + 3]   wg_z              (gl_WorkGroupID.z)
 *   [4 + i*8 + 4]   inv_x             (gl_LocalInvocationID.x)
 *   [4 + i*8 + 5]   inv_y             (gl_LocalInvocationID.y)
 *   [4 + i*8 + 6]   error_code        (GPU_DIAG_ERR_* bitfield)
 *   [4 + i*8 + 7]   extra             (pass-specific payload, e.g. bad index)
 *
 * Note: only one invocation per workgroup needs to write — use the DIAG_*
 * macros at the point of detection rather than broadcasting the write.
 */

#ifndef GPU_DIAG_GLSL
#define GPU_DIAG_GLSL

/* ── Pass IDs ── must match GpuPassId enum in gpu_diag.h ─────────────────── */
#define GPU_DIAG_PASS_T1_BVH     1u
#define GPU_DIAG_PASS_T2_REFINE  2u
#define GPU_DIAG_PASS_T3_MAT     3u
#define GPU_DIAG_PASS_T4_FFT_FORWARD 4u
#define GPU_DIAG_PASS_T4_TRANSFER    5u
#define GPU_DIAG_PASS_T4_FFT_INVERSE 6u
#define GPU_DIAG_PASS_T5         7u
#define GPU_DIAG_PASS_UV_BLIT    8u
#define GPU_DIAG_PASS_TILE_OVL   9u
#define GPU_DIAG_PASS_COH_ACC   10u
#define GPU_DIAG_PASS_EMIT_RED  11u

/* ── Error code flags (bitfield — combine with |) ────────────────────────── */
#define GPU_DIAG_ERR_OOB       0x0001u   /* array index out of bounds    */
#define GPU_DIAG_ERR_NAN       0x0002u   /* NaN in a value being written */
#define GPU_DIAG_ERR_INF       0x0004u   /* +/-Inf detected              */
#define GPU_DIAG_ERR_ZERO_DIV  0x0008u   /* zero denominator             */
#define GPU_DIAG_ERR_OVERFLOW  0x0010u   /* DiagBuf entry cap exceeded   */

#define GPU_DIAG_MAX_ENTRIES 64u

/* ── DiagBuf SSBO — only declared when DIAG_BINDING is defined ───────────── */
#ifdef DIAG_BINDING

layout(std430, binding = DIAG_BINDING) buffer DiagBuf {
    uint diag_data[];
};

/**
 * Write one diagnostic entry.  Silently drops entries beyond GPU_DIAG_MAX_ENTRIES;
 * the overflow flag in diag_data[1] is set so the C++ reader knows truncation occurred.
 */
void gpu_diag_write(uint pass_id, uint error_code, uint extra) {
    uint idx = atomicAdd(diag_data[0], 1u);
    if (idx >= GPU_DIAG_MAX_ENTRIES) {
        atomicOr(diag_data[1], GPU_DIAG_ERR_OVERFLOW);
        return;
    }
    uint base = 4u + idx * 8u;
    diag_data[base + 0u] = pass_id;
    diag_data[base + 1u] = gl_WorkGroupID.x;
    diag_data[base + 2u] = gl_WorkGroupID.y;
    diag_data[base + 3u] = gl_WorkGroupID.z;
    diag_data[base + 4u] = gl_LocalInvocationID.x;
    diag_data[base + 5u] = gl_LocalInvocationID.y;
    diag_data[base + 6u] = error_code;
    diag_data[base + 7u] = extra;
}

/* ── Convenience macros (require DIAG_PASS_ID to also be defined) ─────────── */
#ifdef DIAG_PASS_ID
#define DIAG_OOB(extra)  gpu_diag_write(DIAG_PASS_ID, GPU_DIAG_ERR_OOB,      uint(extra))
#define DIAG_NAN(extra)  gpu_diag_write(DIAG_PASS_ID, GPU_DIAG_ERR_NAN,      uint(extra))
#define DIAG_INF(extra)  gpu_diag_write(DIAG_PASS_ID, GPU_DIAG_ERR_INF,      uint(extra))
#define DIAG_ZDIV(extra) gpu_diag_write(DIAG_PASS_ID, GPU_DIAG_ERR_ZERO_DIV, uint(extra))
#endif /* DIAG_PASS_ID */

#endif /* DIAG_BINDING */

#endif /* GPU_DIAG_GLSL */
