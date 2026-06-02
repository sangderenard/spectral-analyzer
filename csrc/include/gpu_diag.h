#pragma once
/**
 * gpu_diag.h  —  Standardized GPU dispatch error harness  (C++ side).
 *
 * Include this header AFTER gl_compute.h (needs GLenum, glGetError).
 *
 * Quick reference
 * ───────────────
 *   GLC_DISPATCH_CHECKED("label", wg_x, wg_y, wg_z);
 *     Drop-in for bare glc_DispatchCompute(x,y,z).  On GL error the macro
 *     prints a structured stderr line and continues; it does NOT abort.
 *     Callers that need to abort on error should use gpu_check_dispatch()
 *     directly and test the return value.
 *
 *   GLenum err = gpu_check_dispatch("label", wg_x, wg_y, wg_z);
 *     Same check, returns the raw error code so the caller can branch.
 *
 * Log format (stderr, only on error)
 * ───────────────────────────────────
 *   [GPU-ERR] #<seq>  t=<wall_s>s  <label>  wg=(<x>,<y>,<z>)  0x<code> <name>
 *
 *   seq    : monotonic dispatch counter (identifies ordering across passes)
 *   wall_s : seconds since first dispatch (correlates with heartbeat logs)
 *   label  : caller-supplied tag (e.g. "T1:BVH", "T4:BPM-m0")
 *   wg     : workgroup count actually dispatched
 *   code   : raw GL error hex
 *   name   : human-readable GL error constant
 *
 * Thread safety
 * ─────────────
 * gpu_check_dispatch() is thread-safe (function-local atomic counter,
 * fprintf/fflush are per-call).  Multiple GPU-facing threads will interleave
 * cleanly in stderr output.
 *
 * GLSL side
 * ─────────
 * Shaders can opt in to the DiagBuf mechanism via csrc/shaders/gpu_diag.glsl.
 * That file is a standalone GLSL #include — no shader is changed by default.
 * See that file's header for the opt-in protocol (DIAG_BINDING, DIAG_PASS_ID).
 *
 * Pass IDs (GpuPassId enum below) must stay in sync with the #define values
 * in gpu_diag.glsl.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

/* ── Pass IDs ── must match #define GPU_DIAG_PASS_* in gpu_diag.glsl ────── */
enum class GpuPassId : uint32_t {
    T1_BVH_INTERSECT =  1u,
    T2_REFINE        =  2u,
    T3_MATERIAL      =  3u,
    T4_BPM_M0        =  4u,
    T4_BPM_M1        =  5u,
    T4_BPM_M2        =  6u,
    T5_CONNECT       =  7u,
    UV_BLIT          =  8u,
    TILE_OVERLAP     =  9u,
    COHERENT_ACCUM   = 10u,
    EMITTER_REDUCE   = 11u,
};

/* ── Shader-side error code flags ── must match gpu_diag.glsl ────────────── */
static constexpr uint32_t GPU_DIAG_ERR_OOB      = 0x0001u;  /* out-of-bounds index  */
static constexpr uint32_t GPU_DIAG_ERR_NAN      = 0x0002u;  /* NaN detected         */
static constexpr uint32_t GPU_DIAG_ERR_INF      = 0x0004u;  /* +/-Inf detected      */
static constexpr uint32_t GPU_DIAG_ERR_ZERO_DIV = 0x0008u;  /* zero denominator     */
static constexpr uint32_t GPU_DIAG_ERR_OVERFLOW = 0x0010u;  /* DiagBuf slot cap hit */

/* ── DiagBuf layout constants (shared with gpu_diag.glsl) ──────────────── */
static constexpr int GPU_DIAG_MAX_ENTRIES = 64;
static constexpr int GPU_DIAG_HEADER_U32  =  4;   /* reserved header slots    */
static constexpr int GPU_DIAG_ENTRY_U32   =  8;   /* uint32 fields per entry  */
static constexpr int GPU_DIAG_BUF_U32     = GPU_DIAG_HEADER_U32
                                           + GPU_DIAG_MAX_ENTRIES * GPU_DIAG_ENTRY_U32;
static constexpr int GPU_DIAG_BUF_BYTES   = GPU_DIAG_BUF_U32 * 4;

/* DiagBuf entry layout (each entry = 8 x uint32):
 *   [0]  pass_id          (GpuPassId)
 *   [1]  wg_x             (gl_WorkGroupID.x)
 *   [2]  wg_y             (gl_WorkGroupID.y)
 *   [3]  wg_z             (gl_WorkGroupID.z)
 *   [4]  inv_x            (gl_LocalInvocationID.x)
 *   [5]  inv_y            (gl_LocalInvocationID.y)
 *   [6]  error_code       (GPU_DIAG_ERR_* bitfield)
 *   [7]  extra            (pass-specific payload, e.g. bad index)
 *
 * Header (4 x uint32 at offset 0):
 *   [0]  entry_count      (atomicAdd counter; 0 = clean)
 *   [1]  overflow_flags   (set if entry_count exceeded GPU_DIAG_MAX_ENTRIES)
 *   [2]  reserved
 *   [3]  reserved
 */

/* ── GL error code → human-readable string ─────────────────────────────── */
static inline const char* gpu_gl_err_str(unsigned err) {
    switch (err) {
        case 0x0500: return "GL_INVALID_ENUM";
        case 0x0501: return "GL_INVALID_VALUE";
        case 0x0502: return "GL_INVALID_OPERATION";
        case 0x0505: return "GL_OUT_OF_MEMORY";
        case 0x0506: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case 0x0507: return "GL_CONTEXT_LOST (TDR reset)";
        case 0x8031: return "GL_TABLE_TOO_LARGE";
        default:     return "GL_UNKNOWN_ERROR";
    }
}

/* ── Core C++ check ─────────────────────────────────────────────────────── */
/**
 * Call immediately after glc_DispatchCompute (and any preceding glFlush if
 * TDR detection is desired).  Increments the global dispatch sequence counter
 * regardless of success — use the seq number to correlate logs from different
 * passes.
 *
 * Returns GL_NO_ERROR on success, the error code otherwise.
 */
static inline GLenum gpu_check_dispatch(
        const char* label,
        unsigned wg_x, unsigned wg_y, unsigned wg_z)
{
    /* Monotonic dispatch sequence number — thread-safe, initialised once. */
    static std::atomic<uint64_t> seq{0};
    const uint64_t n = seq.fetch_add(1u, std::memory_order_relaxed);

    using Clock = std::chrono::steady_clock;
    static const Clock::time_point t0 = Clock::now();

    /* Drain ALL pending GL errors so none leak into the next dispatch check.
     * glGetError() returns one error per call; loop until the queue is empty. */
    GLenum first_err = GL_NO_ERROR;
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        if (first_err == GL_NO_ERROR) first_err = err;
        const float t_s = std::chrono::duration<float>(Clock::now() - t0).count();
        fprintf(stderr,
            "[GPU-ERR] #%llu  t=%.3fs  %-22s  wg=(%u,%u,%u)  0x%04X %s\n",
            (unsigned long long)n, t_s,
            label, wg_x, wg_y, wg_z,
            (unsigned)err, gpu_gl_err_str((unsigned)err));
        fflush(stderr);
    }
    return first_err;
}

/* ── Convenience macro ──────────────────────────────────────────────────── */
/**
 * GLC_DISPATCH_CHECKED(label, x, y, z)
 *
 * Drop-in replacement for bare:  glc_DispatchCompute(x, y, z);
 *
 * Dispatches then immediately calls gpu_check_dispatch() and discards the
 * return value.  Use gpu_check_dispatch() directly when you need to test the
 * error code (e.g. to set a gpu_ok flag and break out of a batch loop).
 */
#define GLC_DISPATCH_CHECKED(label, x, y, z)              \
    do {                                                    \
        glc_DispatchCompute((x), (y), (z));                \
        gpu_check_dispatch((label),                        \
            (unsigned)(x), (unsigned)(y), (unsigned)(z)); \
    } while(0)
