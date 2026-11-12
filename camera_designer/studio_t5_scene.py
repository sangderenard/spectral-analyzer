"""Lossless Studio-scene inputs for the existing native BDPT/T5 tracer.

The Studio scene builder already uses MaterialDatabase. This bridge preserves
its triangle order and spectral material buffer rather than reconstructing
materials from display RGB or adding a second lighting interpretation.
"""

from __future__ import annotations

import numpy as np

from mat_flags import MAT_FLAG_EMISSIVE, MAX_SPECTRAL_BANDS


def native_t5_scene_inputs(scene: tuple, wavelengths_um) -> dict:
    """Extract exact native RayTracer constructor arrays from a Studio scene."""
    geom, shade, mat_buf, triangles, _contexts, _ctx_map, bounds, _sources = scene
    tri = np.asarray(triangles, np.float64)
    geom = np.asarray(geom, np.float32)
    shade = np.asarray(shade, np.float32)
    materials = np.asarray(mat_buf, np.float32)
    wavelengths = np.asarray(wavelengths_um, np.float64).reshape(-1)
    if (tri.ndim != 3 or tri.shape[1:] != (3, 3) or geom.shape != (len(tri), 16)
            or shade.shape != (len(tri), 16) or materials.ndim != 2
            or materials.shape[1] != 12 or len(materials) % MAX_SPECTRAL_BANDS
            or not 1 <= len(wavelengths) <= MAX_SPECTRAL_BANDS):
        raise ValueError("Studio scene does not match native T5 geometry/material ABI")
    if not (np.isfinite(tri).all() and np.isfinite(materials).all()
            and np.isfinite(wavelengths).all() and np.all(wavelengths > 0)):
        raise ValueError("Studio T5 scene contains nonfinite geometry or spectrum")
    mat_idx = shade[:, 14].view(np.int32).copy()
    n_mats = len(materials) // MAX_SPECTRAL_BANDS
    if np.any(mat_idx < 0) or np.any(mat_idx >= n_mats):
        raise ValueError("Studio triangle material index is outside native MatBuf")
    flags = geom[:, 15].view(np.uint32)
    emitted = materials.reshape(n_mats, MAX_SPECTRAL_BANDS, 12)[mat_idx, :len(wavelengths), 5]
    emitter_mask = ((flags & MAT_FLAG_EMISSIVE) != 0) & np.any(emitted > 0, axis=1)
    emitter_ids = np.ascontiguousarray(np.flatnonzero(emitter_mask), np.int32)
    if not len(emitter_ids):
        raise ValueError("native T5 needs authored emissive Studio triangles")
    return {
        "verts": np.ascontiguousarray(tri.reshape(-1, 9), np.float64),
        "normals": np.ascontiguousarray(geom[:, 12:15], np.float64),
        "mat_idx": np.ascontiguousarray(mat_idx, np.int32),
        "mat_buf": np.ascontiguousarray(materials, np.float32),
        "mat_n_mats": n_mats,
        "freq_hz": np.ascontiguousarray(299792458.0 / (wavelengths * 1e-6), np.float64),
        "emitter_tri_ids": emitter_ids,
        "bounds_min": np.ascontiguousarray(bounds[0], np.float32),
        "bounds_max": np.ascontiguousarray(bounds[1], np.float32),
    }
