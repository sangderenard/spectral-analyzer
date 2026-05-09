"""
surface_spline_utils.py — convenience wrappers for the C++ surface spline fitter.

Converts triangle groups into POLY_BARY parametric surfaces ready to be passed
to ``RayTracer.register_tri_group(..., parametric_surface=...)``.

Typical usage
-------------
::

    from surface_spline_utils import parameterize_mesh, parameterize_tri_groups
    import numpy as np

    # From a raw mesh:
    coeffs = parameterize_mesh(verts, faces, n_threads=8)
    # coeffs.shape == (n_tris, 6)

    # From a list of TriangleGroup objects (bdpt_integrator.TriangleGroup):
    fitted = parameterize_tri_groups(verts, faces, groups, n_threads=8)
    # fitted is the same list with .parametric_surface populated.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

try:
    import _spectral_kernels as _sk
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


def _require_sk() -> None:
    if not _HAS_SK:
        raise RuntimeError(
            "surface_spline_utils: _spectral_kernels extension is not built; "
            "run `cmake --build csrc_build --target _spectral_kernels` first."
        )


def parameterize_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    tri_subset: Optional[np.ndarray] = None,
    ridge_lambda: float = 0.0,
    n_threads: int = 0,
) -> np.ndarray:
    """Fit per-triangle POLY_BARY displacement coefficients for a triangle mesh.

    Parameters
    ----------
    verts        : (N, 3) float64 vertex positions.
    faces        : (T, 3) int32  vertex-index triples.
    tri_subset   : (K,) int32 or None.  Only fit these triangle indices.
    ridge_lambda : Tikhonov regularisation on curvature terms (0 = off).
    n_threads    : worker threads (0 = hardware_concurrency).

    Returns
    -------
    (T, 6) float64 — per-triangle [c0, cu, cv, cuu, cuv, cvv].
    """
    _require_sk()

    verts_f64 = np.asarray(verts, dtype=np.float64)
    faces_i32 = np.asarray(faces, dtype=np.int32)

    if verts_f64.ndim != 2 or verts_f64.shape[1] != 3:
        raise ValueError("verts must be (N, 3) float64")
    if faces_i32.ndim != 2 or faces_i32.shape[1] != 3:
        raise ValueError("faces must be (T, 3) int32")

    subset_arg = (
        np.asarray(tri_subset, dtype=np.int32) if tri_subset is not None else None
    )

    coeffs: np.ndarray = _sk.surface_spline_fit(
        verts_f64,
        faces_i32,
        tri_subset=subset_arg,
        ridge_lambda=ridge_lambda,
        n_threads=n_threads,
    )
    return coeffs  # shape (T, 6)


def parameterize_tri_groups(
    verts: np.ndarray,
    faces: np.ndarray,
    groups,                          # List[bdpt_integrator.TriangleGroup]
    ridge_lambda: float = 0.0,
    n_threads: int = 0,
) -> list:
    """Fit POLY_BARY coefficients and inject them into a list of TriangleGroup objects.

    Each group's ``triangle_indices`` (list of global triangle indices) selects
    which triangles to fit.  After fitting, ``group.parametric_surface`` is set to:

    .. code-block:: python

        {
            "kind":   TRI_PARAM_SURFACE_POLY_BARY,  # 1
            "coeffs": np.ndarray(shape=(len(group.triangle_indices), 6), dtype=float64)
        }

    Parameters
    ----------
    verts        : (N, 3) float64 — global vertex buffer shared by all groups.
    faces        : (T, 3) int32  — global face buffer.
    groups       : list of ``TriangleGroup``-like objects with a
                   ``triangle_indices`` attribute (iterable of int).
    ridge_lambda : Tikhonov regularisation (0 = off).
    n_threads    : worker threads (0 = hardware_concurrency).

    Returns
    -------
    The same ``groups`` list, mutated in-place, for convenience.
    """
    _require_sk()

    try:
        TRI_PARAM_SURFACE_POLY_BARY = _sk.TRI_PARAM_SURFACE_POLY_BARY  # type: ignore[attr-defined]
    except AttributeError:
        TRI_PARAM_SURFACE_POLY_BARY = 1

    verts_f64 = np.asarray(verts, dtype=np.float64)
    faces_i32 = np.asarray(faces, dtype=np.int32)

    if verts_f64.ndim != 2 or verts_f64.shape[1] != 3:
        raise ValueError("verts must be (N, 3) float64")
    if faces_i32.ndim != 2 or faces_i32.shape[1] != 3:
        raise ValueError("faces must be (T, 3) int32")

    n_tris = faces_i32.shape[0]

    # Collect all triangles across all groups, batch-fit, then slice back.
    all_tri_indices: list[int] = []
    for g in groups:
        all_tri_indices.extend(int(i) for i in g.triangle_indices)

    if not all_tri_indices:
        return groups

    subset = np.array(sorted(set(all_tri_indices)), dtype=np.int32)

    # Fit the whole mesh but only work on the relevant triangles.
    all_coeffs = parameterize_mesh(
        verts_f64, faces_i32,
        tri_subset=subset,
        ridge_lambda=ridge_lambda,
        n_threads=n_threads,
    )  # shape (n_tris, 6)

    for g in groups:
        tri_ids = np.array([int(i) for i in g.triangle_indices], dtype=np.int32)
        if tri_ids.size == 0:
            continue
        # Clamp to valid range defensively.
        valid = (tri_ids >= 0) & (tri_ids < n_tris)
        if not valid.all():
            tri_ids = tri_ids[valid]
        group_coeffs = all_coeffs[tri_ids]  # (K, 6)
        g.parametric_surface = {
            "kind":   TRI_PARAM_SURFACE_POLY_BARY,
            "coeffs": group_coeffs,
        }

    return groups


def eval_spline_normals(
    u: float,
    v: float,
    coeffs_6: np.ndarray,
    verts: np.ndarray,
    tri_row: np.ndarray,
) -> np.ndarray:
    """Return the perturbed unit normal at barycentric (u,v).

    Parameters
    ----------
    u, v      : barycentric coordinates.
    coeffs_6  : float64[6] — the POLY_BARY block for the triangle.
    verts     : (N, 3) float64.
    tri_row   : int[3] — vertex indices of the triangle.

    Returns
    -------
    float64[3] — perturbed unit normal.
    """
    _require_sk()
    return _sk.surface_spline_eval_normal(
        float(u),
        float(v),
        np.asarray(coeffs_6, dtype=np.float64),
        np.asarray(verts,    dtype=np.float64),
        np.asarray(tri_row,  dtype=np.int32),
    )
