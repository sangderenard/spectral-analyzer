/**
 * serial_kernel.h — C ABI contract for the spectral serial daemon kernels.
 *
 * All exported symbols use extern "C" so both ctypes (Python DLL loading) and
 * pybind11 can bind them.  The Python side is free to hold opaque state handles
 * and pass raw data pointers from torch tensors without any Python GIL.
 *
 * Conventions
 * -----------
 * - All complex values are interleaved double pairs: [re0, im0, re1, im1, ...]
 *   This matches torch.complex128 contiguous storage exactly.
 * - "state handle" = opaque void* returned by *_create() and passed to every
 *   subsequent call.  Freed by *_destroy().
 * - Functions return 0 on success, negative error codes on failure.
 *
 * Error codes
 * -----------
 *   0  SK_OK
 *  -1  SK_ERR_NULL_STATE   — state handle is NULL
 *  -2  SK_ERR_DIM_MISMATCH — tensor dimension doesn't match compiled state
 *  -3  SK_ERR_DIVERGED     — iterative solver did not converge
 */

#pragma once

#ifdef _WIN32
#  ifdef BUILDING_ROUTER_STEP_DLL
     /* Building router_step_c.dll — export all SK_API symbols. */
#    define SK_API __declspec(dllexport)
#  elif defined(SK_IMPORT)
     /* External consumer linking against router_step.dll — import symbols. */
#    define SK_API __declspec(dllimport)
#  else
     /* Building the pybind11 module (or any other direct compilation) —
        no decoration: the translation unit contains the definitions. */
#    define SK_API
#  endif
#else
#  define SK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Compile-time constants (mirrored in Python) ─────────────────────────── */
#define SK_OK              0
#define SK_ERR_NULL_STATE -1
#define SK_ERR_DIM_MISMATCH -2
#define SK_ERR_DIVERGED   -3

/* ── RouterStep daemon ───────────────────────────────────────────────────── */

/**
 * RouterStepState: opaque handle.
 *
 * Internal layout (Eigen-backed):
 *   M         — N×N complex128 solve matrix  ((I − W_lin)⁻¹)
 *   W_lin     — N×N complex128 zero-delay weight matrix (kept for sat iteration)
 *   ring_W[]  — list of N×N delay weight matrices
 *   ring_buf[]— (B, d, N) ring buffers, one per unique delay length
 *   ring_pos[]— write position per ring
 *   B         — batch size
 *   N         — node count
 */
typedef struct RouterStepState RouterStepState;

/**
 * Allocate and initialise a RouterStep daemon.
 *
 * @param N               Number of graph nodes.
 * @param B               Batch size (independent instances sharing M).
 * @param M_re, M_im      N×N row-major arrays for the solve matrix.
 * @param W_re, W_im      N×N row-major arrays for the zero-delay weight matrix.
 * @param n_delays        Number of distinct delay groups.
 * @param delay_lengths   Array of n_delays integers (samples per group).
 * @param ring_W_re/im    Flat array: n_delays × N × N row-major matrices.
 * @param max_iterations  Fixed-point iteration cap for saturating edges.
 * @param convergence_eps Convergence threshold (|ΔX|_∞).
 * @param infinity_threshold  Magnitude above which a node is considered blown.
 * @return Opaque handle; NULL on allocation failure.
 */
SK_API RouterStepState* router_step_create(
    int    N,
    int    B,
    const double* M_re,
    const double* M_im,
    const double* W_re,
    const double* W_im,
    int    n_delays,
    const int*    delay_lengths,
    const double* ring_W_re,
    const double* ring_W_im,
    int    max_iterations,
    double convergence_eps,
    double infinity_threshold
);

/** Free state allocated by router_step_create. Safe to call with NULL. */
SK_API void router_step_destroy(RouterStepState* state);

/**
 * Advance one sample through the router.
 *
 * @param state   Daemon handle from router_step_create.
 * @param src     Input signal: interleaved complex128, shape (B, N) row-major.
 *                (Pointer into a torch.complex128 tensor's data_ptr().)
 * @param out     Output signal: same shape/layout.  May alias src.
 * @return SK_OK, SK_ERR_NULL_STATE, or SK_ERR_DIVERGED.
 *
 * Thread safety: not thread-safe; the caller (Python) must serialise calls
 * per state handle.  Multiple independent state handles are fine in parallel.
 */
SK_API int router_step_step(
    RouterStepState* state,
    const double*    src,   /* interleaved (B, N) complex128 */
    double*          out    /* interleaved (B, N) complex128 */
);

/**
 * Advance T samples in a tight serial loop (the daemon's core use case).
 *
 * src[t] and out[t] are at byte offset t * B * N * 2 * sizeof(double)
 * within the respective arrays, which must have at least T × B × N complex
 * elements.  The loop is fully inline — no Python callback overhead per step.
 *
 * @param state   Daemon handle.
 * @param src     Input buffer: interleaved complex128, shape (T, B, N).
 * @param out     Output buffer: same shape.
 * @param T       Number of samples to process.
 * @return SK_OK or an error code from the first failing step.
 */
SK_API int router_step_run(
    RouterStepState* state,
    const double*    src,
    double*          out,
    int              T
);

/**
 * Resize the batch dimension without recompiling M.
 * Reallocates ring buffers; zeros all delay state.
 */
SK_API int router_step_resize_batch(RouterStepState* state, int new_B);

/** Zero all ring-buffer delay state. */
SK_API void router_step_reset(RouterStepState* state);

/** Read diagnostic fields after a step. */
SK_API void router_step_diagnostics(
    const RouterStepState* state,
    int*    last_convergence_iters,
    int*    last_saturated
);

#ifdef __cplusplus
} /* extern "C" */
#endif
