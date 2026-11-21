"""Lossless Studio-scene inputs for the existing native BDPT/T5 tracer.

The Studio scene builder already uses MaterialDatabase. This bridge preserves
its triangle order and spectral material buffer rather than reconstructing
materials from display RGB or adding a second lighting interpretation.
"""

from __future__ import annotations

import numpy as np

from mat_flags import MAT_FLAG_EMISSIVE, MAX_SPECTRAL_BANDS
from .studio_region_gpu import region_records_from_geometry, validate_region_records


def native_t5_scene_inputs(scene: tuple, wavelengths_um, spheres=None) -> dict:
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
    # Keep the authenticated Studio boundary identity when lowering the scene.
    # The native T3 consumer must resolve these tags before it can claim media
    # parity; geometry and a material index alone lose bubble/rod/dye ownership.
    boundary_tags = np.ascontiguousarray(geom[:, [3, 7]], np.float32)
    if not np.isfinite(boundary_tags).all():
        raise ValueError("Studio boundary tags must be finite")
    region_records = region_records_from_geometry(geom)
    region_ids = np.full(len(geom), -1, np.int32)
    if region_records is None:
        region_records = np.empty((0, 4), np.float32)
    else:
        region_records = validate_region_records(region_records, geom, shade, n_mats)
        if len(region_records) > 0 and region_records[0, 2] == 1:
            raise ValueError("native T3 does not yet resolve axial rod bundles")
        tagged = boundary_tags[:, 0] <= -1
        region_ids[tagged] = ((-boundary_tags[tagged, 0] - 1) // 4).astype(np.int32)
        for region_id, record in enumerate(region_records):
            if record[0] == 4:
                standalone = (boundary_tags[:, 0] == 1) & (boundary_tags[:, 1] == record[1] + 2)
                region_ids[standalone] = region_id
    exact_spheres = np.empty((0, 5), np.float64)
    if spheres is not None:
        sphere_rows = np.asarray(spheres, np.float64)
        if sphere_rows.ndim != 2 or sphere_rows.shape[1] != 8 or not np.isfinite(sphere_rows).all():
            raise ValueError("exact Studio spheres require the validated (N,8) scene records")
        exact_spheres = np.ascontiguousarray(sphere_rows[:, :5], np.float64)
        owners = exact_spheres[:, 4]
        if (np.any(exact_spheres[:, 3] <= 0) or np.any(owners != np.floor(owners))
                or np.any(owners < 0) or np.any(owners >= len(tri))
                or len(set(owners.astype(int))) != len(owners)):
            raise ValueError("exact Studio sphere owner identity is invalid")
        for owner, center in zip(owners.astype(int), exact_spheres[:, :3]):
            if np.any(tri[owner] != center):
                raise ValueError("exact Studio sphere owner must be the degenerate scene triangle")
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
        "boundary_tags": boundary_tags,
        "region_records": np.ascontiguousarray(region_records, np.float32),
        "native_region_media": np.ascontiguousarray(region_records[:, :2], np.int32),
        "tri_region_ids": region_ids,
        "exact_spheres": exact_spheres,
        "tri_flags": np.ascontiguousarray(flags, np.uint32),
    }


def create_native_t5_tracer(scene: tuple, wavelengths_um, spheres=None):
    """Install the verified Studio scene on the existing native T1–T5 tracer.

    This is the CPU transport path while native GPU T1/T3 lacks the authored
    region traversal. The renderer must still configure its camera and T5
    schedule on the returned tracer before submitting work.
    """
    import _spectral_kernels as native

    inputs = native_t5_scene_inputs(scene, wavelengths_um, spheres)
    tracer = native.RayTracer(
        len(inputs["verts"]), inputs["verts"], inputs["normals"],
        inputs["mat_idx"], inputs["mat_buf"], inputs["mat_n_mats"],
        inputs["freq_hz"], 299792458.0,
        np.zeros(len(inputs["freq_hz"]), np.float64),
    )
    try:
        flags = inputs["tri_flags"]
        edges = np.r_[0, np.flatnonzero(flags[1:] != flags[:-1]) + 1, len(flags)]
        for start, end in zip(edges[:-1], edges[1:]):
            tracer.set_tri_ior(int(start), int(end - start), int(flags[start]))
        if len(inputs["native_region_media"]):
            tracer.set_studio_regions(inputs["native_region_media"], inputs["tri_region_ids"])
        if len(inputs["exact_spheres"]):
            tracer.set_studio_spheres(inputs["exact_spheres"])
    except Exception:
        tracer.stop_pipeline()
        raise
    return tracer
