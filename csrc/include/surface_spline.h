/**
 * surface_spline.h — Eigen-based quadratic surface spline fitter.
 *
 * For each triangle in one or more TriGroups this module computes a
 * per-triangle POLY_BARY coefficient block:
 *
 *   delta(u,v) = c0 + cu*u + cv*v + cuu*u*u + cuv*u*v + cvv*v*v
 *
 * that lifts the flat triangle toward its smooth neighbourhood.  The result
 * is a dense array of float64[6] blocks — one per triangle — ready to pass
 * directly as the `parametric_payload` in TriGroupDesc.
 *
 * Algorithm (per triangle T)
 * --------------------------
 *   1. Collect a neighbourhood N(T): the triangle itself plus any triangle
 *      sharing at least one vertex (1-ring).
 *   2. For each neighbour, sample its centroid in T's barycentric frame and
 *      compute the signed normal-axis displacement δ = (centroid - T.v0) · T.n.
 *   3. Build the 6-column Vandermonde matrix V from (1, u, v, u², uv, v²)
 *      of each sampled point.
 *   4. Solve the least-squares system  V c = δ  via Eigen's ColPivHouseholder
 *      QR, yielding c = [c0 cu cv cuu cuv cvv].
 *   5. Optional regularisation: Tikhonov ridge λ added to the normal equations.
 *
 * The full mesh fit is embarrassingly parallel across triangles and is
 * dispatched through a ThreadPool.
 *
 * C API (extern "C")
 * ------------------
 * See declarations below.  Python bindings are in pybind_kernels.cpp.
 */
#pragma once

#include "serial_kernel.h"  /* SK_OK, SK_ERR_* */

/* surface_spline functions are compiled directly into _spectral_kernels —
 * they are never in a standalone DLL, so we do not need dllexport/import. */
#ifndef SS_API
#  ifdef __GNUC__
#    define SS_API __attribute__((visibility("default")))
#  else
#    define SS_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fit POLY_BARY surface spline coefficients for a triangle mesh.
 *
 * Parameters
 * ----------
 * n_verts         : number of vertices.
 * verts_xyz       : (n_verts, 3) float64, row-major.
 * n_tris          : number of triangles.
 * tri_indices     : (n_tris, 3) int32, each row is [i0, i1, i2].
 * tri_subset      : (n_subset,) int32 — triangles to fit; NULL = fit all.
 * n_subset        : length of tri_subset; 0 when tri_subset is NULL (fit all).
 * ridge_lambda    : Tikhonov regularisation strength (0.0 = none).
 * n_threads       : worker count (0 = hardware_concurrency).
 * out_coeffs      : caller-allocated float64[n_tris * 6] — filled in order of
 *                   the original triangle indices (not the subset order).
 *                   Triangles NOT in the subset receive zero coefficients.
 *
 * Returns SK_OK on success, SK_ERR_NULL_STATE if pointers are null.
 */
SS_API int surface_spline_fit(
    int            n_verts,
    const double*  verts_xyz,
    int            n_tris,
    const int*     tri_indices,
    const int*     tri_subset,
    int            n_subset,
    double         ridge_lambda,
    int            n_threads,
    double*        out_coeffs
);

/**
 * Convenience: compute per-triangle smooth normal from the fitted POLY_BARY
 * coefficients and barycentric (u, v).  Useful for visualisation / debug.
 *
 * tri_idx     : which triangle (index into verts/tris used during fit).
 * u, v        : barycentric coords inside that triangle.
 * coeffs_6    : the float64[6] block for tri_idx.
 * verts_xyz   : original vertex buffer (3-element strides).
 * tri_row     : int[3] — the triangle's vertex indices.
 * out_normal  : float64[3] — receives the perturbed unit normal.
 */
SS_API int surface_spline_eval_normal(
    double         u,
    double         v,
    const double*  coeffs_6,
    const double*  verts_xyz,
    const int*     tri_row,
    double*        out_normal
);

#ifdef __cplusplus
} /* extern "C" */
#endif
