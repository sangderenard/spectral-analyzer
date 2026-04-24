/**
 * transforms_api.h — Shared C enum and in-place transform dispatch.
 *
 * CTransformID values are the sole source of truth for the C-registered
 * transform set; keep in sync with CTransformID in c_transforms.py.
 *
 * All transforms operate on the complex magnitude while preserving phase,
 * matching _apply_saturation() in graph_solver.py.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CT_IDENTITY  = 0,
    CT_TANH      = 1,
    CT_SOFTCLIP  = 2,
    CT_HARDCLIP  = 3,
} CTransformID;

/**
 * Apply a registered transform in-place to n interleaved complex128 elements.
 * Each element occupies two doubles: buf[2*i] = real, buf[2*i+1] = imag.
 *
 * @param id    Which transform to apply.
 * @param knee  Scale / knee parameter (matches TensorEdge.saturation_knee).
 *              Clamped to 1e-30 internally; safe to pass 0.
 * @param buf   2*n doubles to transform in place.
 * @param n     Number of complex elements.
 */
void ct_apply(CTransformID id, double knee, double* buf, int n);

#ifdef __cplusplus
}
#endif
