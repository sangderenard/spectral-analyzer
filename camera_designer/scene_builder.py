"""camera_designer/scene_builder.py
=====================================
Convert a CameraPreset + scene description into a PyRayTracer instance with
registered scale-context spheres, ready for use in the camera designer station.

All tracing is done exclusively by the C extension `_spectral_kernels.RayTracer`.

Public API
----------
    tracer, ctx_map = build_tracer(preset, lights, scene_object, wavelengths_um)

Parameters
----------
preset          : CameraPreset
lights          : list of dicts:
                      { "pos": [x,y,z], "color": [r,g,b], "power": float }
scene_object    : dict describing the central scene object:
                      { "type": "sphere", "pos": [x,y,z], "radius": float }
                  or  { "type": "mesh",  "verts": (N,3), "normals": (N,3) }
wavelengths_um  : sequence of wavelengths in micrometres; default (0.45, 0.55, 0.65)

Returns
-------
tracer          : _spectral_kernels.RayTracer
ctx_map         : dict  label -> context_id  (one entry per lens element context)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera_preset import CameraPreset, EmitterSpec, ProjectorBackSpec
from .optical_material import OpticalMaterial, MATERIAL_CATALOG
from .optical_volume import OpticalVolume, _glass_spec_to_material
from .parametric_surfaces import (
    ConicSurface,
    FlatSurface,
    ParametricSurface,
    SphericalSurface,
)

__all__ = ["build_tracer", "build_gpu_scene"]

# GLSL MAT_FLAG constants (must match the defines in _GPU_RAY_FIELD_CS)
_MAT_FLAG_EMISSIVE   = np.uint32(1)
_MAT_FLAG_REACTIVE   = np.uint32(2)
_MAT_FLAG_ABSORBER   = np.uint32(4)
_MAT_FLAG_NO_SHADOW  = np.uint32(8)
_MAT_FLAG_MANIFOLD   = np.uint32(16)
_MAT_FLAG_TRANSMISSIVE = np.uint32(64)

_C_LIGHT = 2.99792458e8  # m/s


# ─────────────────────────────────────────────────────────────────────────────
# Surface triangulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _triangulate_disk(
    z_func,           # callable(r) -> z offset from z_axis
    z_axis: float,    # z position of the surface vertex
    r_inner: float,
    r_outer: float,
    n_rings: int = 8,
    n_sectors: int = 24,
    outward_normal_z: float = 1.0,   # +1 or -1 to flip face normal
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (verts, normals) arrays for a disk-like surface.

    verts  : float64 (N_tri, 9)  — three consecutive xyz per triangle
    normals: float64 (N_tri, 3)  — one normal per triangle (flat shading)
    """
    rings = np.linspace(r_inner, r_outer, n_rings + 1)
    # sample (r, theta) grid
    thetas = np.linspace(0.0, 2.0 * math.pi, n_sectors, endpoint=False)

    pts: List[np.ndarray] = []
    for r in rings:
        for th in thetas:
            x = r * math.cos(th)
            y = r * math.sin(th)
            z = z_axis + z_func(r)
            pts.append(np.array([x, y, z], np.float64))

    pts_arr = np.array(pts)  # (n_rings+1) * n_sectors, 3

    def idx(ring_i, sector_i):
        return ring_i * n_sectors + (sector_i % n_sectors)

    tri_verts = []
    tri_normals = []
    for ri in range(n_rings):
        for si in range(n_sectors):
            p0 = pts_arr[idx(ri,   si    )]
            p1 = pts_arr[idx(ri+1, si    )]
            p2 = pts_arr[idx(ri+1, si + 1)]
            p3 = pts_arr[idx(ri,   si + 1)]
            # two triangles: (p0,p1,p2) and (p0,p2,p3)
            for a, b, c in [(p0, p1, p2), (p0, p2, p3)]:
                e1 = b - a
                e2 = c - a
                n = np.cross(e1, e2)
                nl = np.linalg.norm(n)
                if nl < 1e-30:
                    continue
                n = n / nl
                if outward_normal_z < 0:
                    n = -n
                row = np.concatenate([a, b, c])
                tri_verts.append(row)
                tri_normals.append(n)

    return np.array(tri_verts, np.float64), np.array(tri_normals, np.float64)


def _triangulate_surface(
    surf: ParametricSurface,
    n_rings: int = 8,
    n_sectors: int = 24,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Triangulate a ParametricSurface.

    Returns (verts, normals, z_pos, r_outer).
    verts  : (N_tri, 9) float64
    normals: (N_tri, 3) float64
    """
    if isinstance(surf, FlatSurface):
        z_pos   = surf.z_pos
        r_inner = getattr(surf, "r_min", 0.0)
        r_outer = surf.r_max

        def z_func(r):
            return 0.0

        verts, normals = _triangulate_disk(
            z_func, z_pos, r_inner, r_outer, n_rings, n_sectors)
        return verts, normals, z_pos, r_outer

    elif isinstance(surf, SphericalSurface):
        R       = surf.R
        r_inner = getattr(surf, "r_min", 0.0)
        r_outer = surf.r_max
        # z_vertex is 0 in local surface coords — find it from attribute
        z_vertex = float(getattr(surf, "z_vertex", getattr(surf, "z_pos", 0.0)))

        def z_func(r):
            # sag formula: z = R - sqrt(R^2 - r^2), with sign from R
            if abs(R) < 1e-12:
                return 0.0
            disc = R * R - r * r
            if disc < 0.0:
                disc = 0.0
            return R - math.copysign(math.sqrt(disc), R)

        verts, normals = _triangulate_disk(
            z_func, z_vertex, r_inner, r_outer, n_rings, n_sectors,
            outward_normal_z=1.0 if R > 0 else -1.0)
        return verts, normals, z_vertex, r_outer

    elif isinstance(surf, ConicSurface):
        r_outer = float(surf.r_max)
        r_inner = float(surf.r_min)

        def z_func(r):
            sag = float(surf._sag(r * r))
            return sag if math.isfinite(sag) else 0.0

        verts, normals = _triangulate_disk(
            z_func, 0.0, r_inner, r_outer, n_rings, n_sectors,
            outward_normal_z=1.0,
        )
        return verts, normals, 0.0, r_outer

    else:
        # Generic fallback: sample with the surface's own intersect() on a grid
        r_outer = float(getattr(surf, "r_max", 0.020))
        r_inner = float(getattr(surf, "r_min", 0.0))
        z_axis  = float(getattr(surf, "z_pos",
                        getattr(surf, "z_vertex", 0.0)))
        rd = np.array([0., 0., -1.], np.float64)

        def z_func(r):
            for angle in [0.0]:
                ro = np.array([r * math.cos(angle), r * math.sin(angle),
                                z_axis + 1.0], np.float64)
                t, hit, _ = surf.intersect(ro, rd)
                if math.isfinite(t) and t < 1e29:
                    return hit[2] - z_axis
            return 0.0

        verts, normals = _triangulate_disk(
            z_func, z_axis, r_inner, r_outer, n_rings, n_sectors)
        return verts, normals, z_axis, r_outer


def _triangulate_sphere_scene_object(
    center: np.ndarray,
    radius: float,
    n_lat: int = 12,
    n_lon: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (verts, normals) for a sphere (outward normals)."""
    verts_list = []
    norms_list = []
    for i in range(n_lat):
        lat0 = math.pi * (-0.5 + i / n_lat)
        lat1 = math.pi * (-0.5 + (i + 1) / n_lat)
        for j in range(n_lon):
            lon0 = 2.0 * math.pi * (j / n_lon)
            lon1 = 2.0 * math.pi * ((j + 1) / n_lon)

            def pt(la, lo):
                return center + radius * np.array([
                    math.cos(la) * math.cos(lo),
                    math.cos(la) * math.sin(lo),
                    math.sin(la)], np.float64)

            p00 = pt(lat0, lon0)
            p10 = pt(lat1, lon0)
            p11 = pt(lat1, lon1)
            p01 = pt(lat0, lon1)

            for a, b, c in [(p00, p10, p11), (p00, p11, p01)]:
                e1 = b - a
                e2 = c - a
                n  = np.cross(e1, e2)
                nl = np.linalg.norm(n)
                if nl < 1e-30:
                    continue
                verts_list.append(np.concatenate([a, b, c]))
                norms_list.append(n / nl)

    return np.array(verts_list, np.float64), np.array(norms_list, np.float64)


def _triangulate_cylinder_wall(
    r: float,
    z_back: float,
    z_front: float,
    n_sectors: int = 32,
    n_div_z:   int = 4,
    r_outer: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a cylindrical wall shell (lens barrel).

    Always emits the inner bore surface at radius ``r`` with inward-facing
    normals (toward the optical axis) so stray rays escaping radially are
    absorbed.  When ``r_outer > r`` an outer skin at ``r_outer`` with
    outward-facing normals is also emitted, blocking rays incident from
    outside the barrel.

    Returns (verts, normals)  shapes (N,9) and (N,3)  float64.
    """
    thetas = np.linspace(0.0, 2.0 * math.pi, n_sectors, endpoint=False)
    zs     = np.linspace(z_back, z_front, n_div_z + 1)

    verts_list: List[np.ndarray] = []
    norms_list: List[np.ndarray] = []

    # Emit one ring of quads for a given radius; inward=True flips normals inward.
    def _emit_ring(radius: float, inward: bool) -> None:
        for zi in range(n_div_z):
            z0, z1 = float(zs[zi]), float(zs[zi + 1])
            for si in range(n_sectors):
                th0 = float(thetas[si])
                th1 = float(thetas[(si + 1) % n_sectors])
                p00 = np.array([radius * math.cos(th0), radius * math.sin(th0), z0], np.float64)
                p10 = np.array([radius * math.cos(th1), radius * math.sin(th1), z0], np.float64)
                p11 = np.array([radius * math.cos(th1), radius * math.sin(th1), z1], np.float64)
                p01 = np.array([radius * math.cos(th0), radius * math.sin(th0), z1], np.float64)
                th_mid = (th0 + th1) * 0.5
                if inward:
                    nv = np.array([-math.cos(th_mid), -math.sin(th_mid), 0.0], np.float64)
                else:
                    nv = np.array([ math.cos(th_mid),  math.sin(th_mid), 0.0], np.float64)
                for a, b, c in [(p00, p11, p10), (p00, p01, p11)]:
                    e1 = b - a; e2 = c - a
                    ncheck = np.cross(e1, e2)
                    if np.dot(ncheck, nv) < 0.0:
                        a, c = c, a
                    nl = np.linalg.norm(np.cross(b - a, c - a))
                    if nl < 1e-30:
                        continue
                    verts_list.append(np.concatenate([a, b, c]))
                    norms_list.append(nv.copy())

    _emit_ring(r, inward=True)
    if r_outer > r + 1e-9:
        _emit_ring(r_outer, inward=False)

    if not verts_list:
        return np.empty((0, 9), np.float64), np.empty((0, 3), np.float64)
    return np.array(verts_list, np.float64), np.array(norms_list, np.float64)


def _triangulate_box_shell_walls(
    x_half:   float,
    y_half:   float,
    z_front:  float,
    z_back:   float,
    port_r:   float = 0.0,
    n_div:    int   = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Six outward-facing walls of a rectangular camera body shell.

    The front wall (at z_front) has a circular port of radius ``port_r``
    cut out of it (for the lens mount opening).  All normals point *outward*
    so that external stray rays hit the front face of each wall.

    Returns (verts, normals) float64.
    """
    verts_list: List[np.ndarray] = []
    norms_list: List[np.ndarray] = []
    bmin = np.array([-x_half, -y_half, z_back],  np.float64)
    bmax = np.array([ x_half,  y_half, z_front], np.float64)

    # faces: (fixed_axis, sign)  sign=+1 → max-face, outward_n = +axis
    faces = [(0, +1.0), (0, -1.0), (1, +1.0), (1, -1.0), (2, +1.0), (2, -1.0)]
    for fixed_axis, sign in faces:
        fixed_val  = bmax[fixed_axis] if sign > 0 else bmin[fixed_axis]
        outward_n  = np.zeros(3, np.float64)
        outward_n[fixed_axis] = sign
        u_axis = (fixed_axis + 1) % 3
        v_axis = (fixed_axis + 2) % 3
        u0, u1 = bmin[u_axis], bmax[u_axis]
        v0, v1 = bmin[v_axis], bmax[v_axis]

        is_front_face = (fixed_axis == 2 and sign > 0)

        for ui in range(n_div):
            for vi in range(n_div):
                ua = u0 + (u1 - u0) * ui       / n_div
                ub = u0 + (u1 - u0) * (ui + 1) / n_div
                va = v0 + (v1 - v0) * vi       / n_div
                vb = v0 + (v1 - v0) * (vi + 1) / n_div

                def _pt(u, v, _fa=fixed_axis, _fv=fixed_val, _ua=u_axis, _va=v_axis):
                    p = np.zeros(3, np.float64)
                    p[_fa] = _fv; p[_ua] = u; p[_va] = v
                    return p

                p00, p10 = _pt(ua, va), _pt(ub, va)
                p11, p01 = _pt(ub, vb), _pt(ua, vb)

                # Cull sub-quads whose centre lies inside the lens mount port
                if is_front_face and port_r > 0.0:
                    cu = (ua + ub) * 0.5
                    cv = (va + vb) * 0.5
                    if math.hypot(cu, cv) < port_r:
                        continue

                # Winding for outward normals
                if sign > 0:
                    tris_pts = [(p00, p10, p11), (p00, p11, p01)]
                else:
                    tris_pts = [(p00, p11, p10), (p00, p01, p11)]

                for a, b, c in tris_pts:
                    e1 = b - a; e2 = c - a
                    nv = np.cross(e1, e2); nl = np.linalg.norm(nv)
                    if nl < 1e-30:
                        continue
                    verts_list.append(np.concatenate([a, b, c]))
                    norms_list.append(outward_n.copy())

    if not verts_list:
        return np.empty((0, 9), np.float64), np.empty((0, 3), np.float64)
    return np.array(verts_list, np.float64), np.array(norms_list, np.float64)


def _triangulate_box(
    bmin: np.ndarray,   # (3,) world-space minimum corner
    bmax: np.ndarray,   # (3,) world-space maximum corner
    n_div: int = 4,     # subdivisions per face edge (for 3D accumulation coverage)
) -> Tuple[np.ndarray, np.ndarray]:
    """Six inward-facing walls of an axis-aligned box.

    Returns (verts, normals) with all normals pointing toward the box interior.
    n_div subdivides each face into n_div×n_div quads so the 3D accumulation
    grid sees spatially resolved arrival samples rather than just 2 giant tris.
    """
    verts_list: List[np.ndarray] = []
    norms_list: List[np.ndarray] = []

    # (fixed_axis, sign): sign=-1 → min face, inward_n = +axis
    #                     sign=+1 → max face, inward_n = -axis
    faces = [
        (0, -1.0), (0,  1.0),   # ±X
        (1, -1.0), (1,  1.0),   # ±Y
        (2, -1.0), (2,  1.0),   # ±Z
    ]
    for fixed_axis, sign in faces:
        fixed_val  = bmax[fixed_axis] if sign > 0 else bmin[fixed_axis]
        inward_n   = np.zeros(3, np.float64)
        inward_n[fixed_axis] = -sign
        u_axis = (fixed_axis + 1) % 3
        v_axis = (fixed_axis + 2) % 3
        u0, u1 = bmin[u_axis], bmax[u_axis]
        v0, v1 = bmin[v_axis], bmax[v_axis]
        for ui in range(n_div):
            for vi in range(n_div):
                ua = u0 + (u1 - u0) * ui       / n_div
                ub = u0 + (u1 - u0) * (ui + 1) / n_div
                va = v0 + (v1 - v0) * vi       / n_div
                vb = v0 + (v1 - v0) * (vi + 1) / n_div

                def _pt(u, v, _fa=fixed_axis, _fv=fixed_val, _ua=u_axis, _va=v_axis):
                    p = np.zeros(3, np.float64)
                    p[_fa] = _fv
                    p[_ua] = u
                    p[_va] = v
                    return p

                p00, p10 = _pt(ua, va), _pt(ub, va)
                p11, p01 = _pt(ub, vb), _pt(ua, vb)

                # Winding chosen so cross(e1,e2) aligns with inward_n.
                # sign < 0: CCW gives +axis  /  sign > 0: reverse → -axis
                if sign < 0:
                    tris_pts = [(p00, p10, p11), (p00, p11, p01)]
                else:
                    tris_pts = [(p00, p11, p10), (p00, p01, p11)]

                for a, b, c in tris_pts:
                    e1 = b - a; e2 = c - a
                    nv = np.cross(e1, e2)
                    nl = np.linalg.norm(nv)
                    if nl < 1e-30:
                        continue
                    verts_list.append(np.concatenate([a, b, c]))
                    norms_list.append(inward_n.copy())

    if not verts_list:
        return np.empty((0, 9), np.float64), np.empty((0, 3), np.float64)
    return np.array(verts_list, np.float64), np.array(norms_list, np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Material → reflectance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresnel_reflectance(n1: float, n2: float) -> float:
    """Unpolarised normal-incidence Fresnel reflectance between n1 and n2."""
    r = (n1 - n2) / (n1 + n2)
    return r * r


def _surface_reflectances(
    mat_in: OpticalMaterial,
    mat_out: OpticalMaterial,
    wavelengths_um: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (refl_re, refl_im) arrays of shape (n_bands,) for an interface.

    For a glass-air interface the reflectance is ~Fresnel; the imaginary part
    encodes the extinction of the outgoing medium (absorption within the glass).
    """
    n_bands = len(wavelengths_um)
    refl_re = np.zeros(n_bands, np.float64)
    refl_im = np.zeros(n_bands, np.float64)
    for b, wl in enumerate(wavelengths_um):
        n1 = mat_in.n_at(wl)
        n2 = mat_out.n_at(wl)
        refl_re[b] = _fresnel_reflectance(n1, n2)
        # Small imaginary component from outgoing medium extinction
        refl_im[b] = mat_out.k * 0.01
    return refl_re, refl_im


def _sensor_reflectances(n_bands: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sensor is an absorber: reflectance ≈ 0 (slight real part for stability)."""
    return np.full(n_bands, 0.02, np.float64), np.zeros(n_bands, np.float64)


def _mirror_reflectances(n_bands: int) -> Tuple[np.ndarray, np.ndarray]:
    """Near-perfect mirror: |R| = 0.95 real."""
    return np.full(n_bands, 0.95, np.float64), np.zeros(n_bands, np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Shared geometry collector
# ─────────────────────────────────────────────────────────────────────────────

# Abstract flag tokens used in ior_specs — kept independent of C extension consts
_GEO_FLAG_TRANSMISSIVE      = "transmissive"
_GEO_FLAG_APERTURE_STOP     = "aperture_stop"
# Aperture-blade geometry (subset of APERTURE_STOP) — same material flag,
# but tagged separately so register_into() can find the BLOCKER tris.
_GEO_FLAG_APERTURE_BLADE    = "aperture_blade"
# Sensor surface (subset of APERTURE_STOP) — same material flag (absorber),
# but tagged separately so register_into() can find the SENSOR tris.
_GEO_FLAG_SENSOR            = "sensor"
# Emissive surface: emits light into the ray tracer (LED, flash, projector).
# Triangulated as a disc; also synthesises a scene-light dict when enabled.
_GEO_FLAG_EMITTER           = "emitter"
# Projector back: large backlit emitter panel just behind the sensor plane.
# diff_all is shaped per-band by ProjectorBackSpec.spectral_weights to model
# the sensor QE response as a spectral transmission filter.
_GEO_FLAG_PROJECTOR_BACK    = "projector_back"
# Confinement box walls: dark-tinted glass exterior, high-reflectance interior.
# Flagged MAT_FLAG_EMISSIVE so GLSL records surface arrivals (vector field
# for outer simulation injection).
_GEO_FLAG_CONFINEMENT_WALL  = "confinement_wall"


def _element_surface_frame(element) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotation, translation) for one authored optical surface."""

    shift = tuple(getattr(element, "shift_xy_m", (0.0, 0.0)))
    tilt = tuple(getattr(element, "tilt_xy_deg", (0.0, 0.0)))
    tx = math.radians(float(tilt[0]))
    ty = math.radians(float(tilt[1]))
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], np.float64)
    ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], np.float64)
    rotation = ry @ rx
    translation = np.array([
        float(shift[0]), float(shift[1]), float(getattr(element, "z_vertex", 0.0))
    ], np.float64)
    return rotation, translation


def _apply_element_surface_frame(
    verts: np.ndarray, normals: np.ndarray, element,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform local tessellation and normals into camera coordinates."""

    rotation, translation = _element_surface_frame(element)
    points = np.asarray(verts, np.float64).reshape(-1, 3)
    transformed = points @ rotation.T + translation
    transformed_normals = np.asarray(normals, np.float64) @ rotation.T
    lengths = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
    transformed_normals /= np.maximum(lengths, 1.0e-15)
    return transformed.reshape(-1, 9), transformed_normals


def _collect_optical_geometry(
    preset: CameraPreset,
    scene_object: Optional[dict],
    wavelengths_um: Sequence[float],
    confinement_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> tuple:
    """Triangulate a CameraPreset + optional scene object into raw geometry arrays.

    Returns
    -------
    verts_all     : (N_tri, 9)  float64 — three vertices per triangle, flattened
    normals_all   : (N_tri, 3)  float64 — one outward normal per triangle
    refl_re_all   : (N_tri, n_bands) float64 — Fresnel reflectance (real part)
    refl_im_all   : (N_tri, n_bands) float64 — Fresnel reflectance (imaginary)
    diff_all      : (N_tri,)    float64 — diffuse albedo per triangle
    ior_specs     : list of (tri_start, n_tris, n_in, n_out, flag_str)
                    flag_str ∈ {_GEO_FLAG_TRANSMISSIVE, _GEO_FLAG_APERTURE_STOP}
    context_specs : list of (label, center_xyz, radius, n_re, n_im, dt_m)
    mat_groups    : list of (tri_start, n_tris, mat_in_rgb, mat_out_rgb,
                             albedo_rgb, ior, opacity, flag_str)
                    — per-group material metadata for GPU buffer packing
    """
    wavelengths_um = list(wavelengths_um)
    n_bands = len(wavelengths_um)
    air = MATERIAL_CATALOG.get("air", OpticalMaterial())

    all_verts:   List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    all_refl_re: List[np.ndarray] = []
    all_refl_im: List[np.ndarray] = []
    all_diff:    List[np.ndarray] = []

    context_specs: List[Tuple[str, np.ndarray, float, float, float, float]] = []
    ior_specs:     List[tuple] = []
    mat_groups:    List[tuple] = []
    lens_material_specs: List[tuple] = []
    _tri_cursor: int = 0

    # ── Lens elements ─────────────────────────────────────────────────────
    elements = sorted(
        preset.lens_group.elements,
        key=lambda e: getattr(e, "z_vertex", getattr(e, "z_pos", 0.0)),
        reverse=True,
    )
    medium_before = air
    for el in elements:
        surf = el.surface
        label = getattr(el, "label", None) or "lens"
        verts, normals, z_local, r_outer = _triangulate_surface(
            surf, n_rings=32, n_sectors=128,
        )
        if len(verts) == 0:
            continue
        # ParametricSurface coordinates are local to their LensElement.  The
        # element's z_vertex is the authoritative camera-space placement (and
        # is what the analytic/bake path already uses).  Losing this transform
        # piled every tessellated lens surface around z=0 on top of the sensor.
        verts, normals = _apply_element_surface_frame(verts, normals, el)
        rotation, translation = _element_surface_frame(el)
        local_vertex = np.array([0.0, 0.0, float(z_local)], np.float64)
        world_vertex = rotation @ local_vertex + translation
        z_vertex = float(world_vertex[2])

        glass_name = None
        if hasattr(el, "glass_out") and el.glass_out is not None:
            glass_name = getattr(el.glass_out, "name", None)
        # Catalog names take the fast path.  Custom/edited GlassSpec instances
        # must retain their authored Sellmeier data instead of silently turning
        # into air when their name is not in the small built-in catalog.
        mat_out = (MATERIAL_CATALOG.get(glass_name)
                   if glass_name else None)
        if mat_out is None:
            mat_out = _glass_spec_to_material(getattr(el, "glass_out", None))
        # ``glass_out`` is the medium after this boundary when traversing the
        # authored lens front-to-back.  The incident medium is therefore the
        # previous element's glass, not air at every surface.
        mat_in = medium_before

        refl_re, refl_im = _surface_reflectances(mat_in, mat_out, wavelengths_um)
        diffusion = np.full(len(verts), mat_out.scatter_albedo or 0.05, np.float64)

        n_new = len(verts)
        all_verts.append(verts)
        all_normals.append(normals)
        all_refl_re.append(np.tile(refl_re, (n_new, 1)))
        all_refl_im.append(np.tile(refl_im, (n_new, 1)))
        all_diff.append(diffusion)

        lam_mean = float(np.mean(wavelengths_um))
        n_in = float(mat_in.n_at(lam_mean))
        n_out = float(mat_out.n_at(lam_mean))
        is_interface = abs(n_out - n_in) > 1.0e-8
        flag_str = _GEO_FLAG_TRANSMISSIVE if is_interface else ""
        if flag_str:
            ior_specs.append((_tri_cursor, n_new, n_in, n_out, flag_str))

        # MatBuf needs a transmissive material row even at a glass->air exit.
        # Use the non-air side as the boundary record; explicit boundary media
        # installed below determine the actual Snell n1/n2 pair.
        boundary_mat = mat_out if n_out > 1.0 + 1.0e-8 else mat_in

        # Glass tint RGB (wavelength-independent approximation: flat 0.9 white glass)
        glass_rgb  = (0.9, 0.9, 0.9)
        mat_groups.append((
            _tri_cursor, n_new,
            (1.0, 1.0, 1.0),   # mat_in: air
            glass_rgb,          # mat_out: glass
            glass_rgb,          # albedo
            float(boundary_mat.n_at(lam_mean)),
            0.0 if flag_str else 1.0,  # opacity: 0 = fully transparent glass
            flag_str,
        ))
        lens_material_specs.append((
            _tri_cursor, n_new, label, boundary_mat, mat_in, mat_out,
        ))

        lam_avg_m = np.mean(wavelengths_um) * 1e-6
        dt_m = lam_avg_m / 20.0
        n_re = boundary_mat.n_at(lam_mean)
        n_im = boundary_mat.k
        context_specs.append((label,
                               np.asarray(world_vertex, np.float64),
                               r_outer * 2.0, n_re, n_im, dt_m))
        _tri_cursor += n_new
        medium_before = mat_out

    # ── Aperture stop — per-blade physical geometry ───────────────────────
    from .parametric_surfaces import ApertureStop as _ApertureStop

    ap = preset.aperture_stop
    if ap is not None and isinstance(ap, _ApertureStop):
        z_ap  = float(ap.z_pos)
        r_in  = float(ap.r_inner)
        r_out = float(ap.r_outer)
        n_bl  = int(ap.n_blades) if ap.n_blades >= 3 else 0
        a_rot = float(ap.aperture_rot)

        blade_verts_list:   List[np.ndarray] = []
        blade_normals_list: List[np.ndarray] = []

        if n_bl == 0:
            if r_in > 1e-9:
                bv, bn = _triangulate_disk(lambda r: 0.0, z_ap, 0.0, r_in,
                                           n_rings=24, n_sectors=256)
                blade_verts_list.append(bv); blade_normals_list.append(bn)
            bv, bn = _triangulate_disk(lambda r: 0.0, z_ap, r_out, r_out * 2.0,
                                       n_rings=12, n_sectors=256)
            blade_verts_list.append(bv); blade_normals_list.append(bn)
        else:
            # Each opaque blade occupies the region OUTSIDE one edge of the
            # clear N-gon.  The old center-out wedges inverted the aperture and
            # left only a pinhole.  These material quads meet at the inscribed
            # opening and extend into the surrounding barrel, so rays interact
            # with real blade geometry rather than a perfect aperture mask.
            clear_poly = ap.blade_polygon_xy()
            blade_outer = r_out * 2.0
            all_blade_verts: List[np.ndarray] = []
            all_blade_normals: List[np.ndarray] = []
            for bi in range(n_bl):
                q0 = clear_poly[bi]
                q1 = clear_poly[(bi + 1) % n_bl]
                a0 = math.atan2(float(q0[1]), float(q0[0]))
                a1 = math.atan2(float(q1[1]), float(q1[0]))
                p0 = np.array([q0[0], q0[1], z_ap], np.float64)
                p1 = np.array([q1[0], q1[1], z_ap], np.float64)
                o0 = np.array([blade_outer * math.cos(a0),
                               blade_outer * math.sin(a0), z_ap], np.float64)
                o1 = np.array([blade_outer * math.cos(a1),
                               blade_outer * math.sin(a1), z_ap], np.float64)
                for a, b_pt, c in ((p0, o0, o1), (p0, o1, p1)):
                    nv = np.cross(b_pt - a, c - a)
                    nl = np.linalg.norm(nv)
                    if nl < 1e-30:
                        continue
                    all_blade_verts.append(np.concatenate([a, b_pt, c]))
                    all_blade_normals.append(nv / nl)
            if all_blade_verts:
                blade_verts_list.append(np.array(all_blade_verts,   np.float64))
                blade_normals_list.append(np.array(all_blade_normals, np.float64))
            if r_in > 1e-9:
                bv, bn = _triangulate_disk(
                    lambda r: 0.0, z_ap, 0.0, r_in,
                    n_rings=12, n_sectors=128,
                )
                blade_verts_list.append(bv); blade_normals_list.append(bn)

        for bv, bn in zip(blade_verts_list, blade_normals_list):
            if len(bv) == 0: continue
            n_new = len(bv)
            zero_r = np.zeros(n_bands, np.float64)
            all_verts.append(bv);   all_normals.append(bn)
            all_refl_re.append(np.tile(zero_r, (n_new, 1)))
            all_refl_im.append(np.tile(zero_r, (n_new, 1)))
            all_diff.append(np.zeros(n_new, np.float64))
            ior_specs.append((_tri_cursor, n_new, 1.0, 1.0, _GEO_FLAG_APERTURE_STOP))
            mat_groups.append((
                _tri_cursor, n_new,
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                1.0, 1.0, _GEO_FLAG_APERTURE_STOP,
            ))
            _tri_cursor += n_new

        lam_avg_m      = float(np.mean(wavelengths_um)) * 1e-6
        iris_dt_m      = lam_avg_m / 20.0
        context_specs.append((
            f"iris_{z_ap:.6f}",
            np.array([0.0, 0.0, z_ap], np.float64),
            r_out * 3.0, 1.0, 0.0, iris_dt_m,
        ))

    # ── Lens tube shell (inner barrel wall + end collars) ────────────────
    # A matte-black cylinder snapped around the widest lens element, running
    # from the front of the lens assembly back to the aperture/mount flange.
    # The inner bore wall uses inward normals so sideways-escaping rays are
    # absorbed; the annular end collars block light from entering the barrel
    # off-axis at the front and leaking out the back.
    _tube_mat     = MATERIAL_CATALOG.get("tube", MATERIAL_CATALOG["black_anodize"])
    _tube_tint_rgb = _tube_mat.tint[:3]
    _absorber_r   = np.zeros(n_bands, np.float64)   # zero reflectance = perfect absorber

    # Compute tube extents from the lens element list
    _el_r_maxes = [float(getattr(e.surface, "r_max", 0.0)) for e in elements]
    _el_z_vals  = [float(getattr(e, "z_vertex",
                          getattr(e.surface, "z_pos", 0.0))) for e in elements]
    if _el_r_maxes and _el_z_vals:
        _tube_r_inner = max(_el_r_maxes)              # bore = widest clear aperture
        _tube_r_outer = _tube_r_inner + 0.004         # 4 mm wall thickness
        _tube_z_front = max(_el_z_vals) + 0.008       # 8 mm front collar overhang
        _tube_z_back  = float(getattr(preset.aperture_stop, "z_pos",
                               min(_el_z_vals) if _el_z_vals else 0.0))

        if _tube_z_front > _tube_z_back + 1e-4:
            # Bore wall: inner (inward normals) + outer skin (outward normals)
            _tube_bv, _tube_bn = _triangulate_cylinder_wall(
                _tube_r_inner, _tube_z_back, _tube_z_front, n_sectors=160, n_div_z=48,
                r_outer=_tube_r_outer)
            # Front annular collar (blocks off-axis entry at the front of the barrel)
            _tube_fc_v, _tube_fc_n = _triangulate_disk(
                lambda _r: 0.0, _tube_z_front,
                _tube_r_inner, _tube_r_outer, n_rings=8, n_sectors=160,
                outward_normal_z=1.0)
            # Back annular collar (light-seal at the mount flange)
            _tube_bc_v, _tube_bc_n = _triangulate_disk(
                lambda _r: 0.0, _tube_z_back,
                _tube_r_inner, _tube_r_outer, n_rings=8, n_sectors=160,
                outward_normal_z=-1.0)

            for _tv, _tn in [(_tube_bv, _tube_bn),
                              (_tube_fc_v, _tube_fc_n),
                              (_tube_bc_v, _tube_bc_n)]:
                if len(_tv) == 0:
                    continue
                _n_new = len(_tv)
                all_verts.append(_tv);    all_normals.append(_tn)
                all_refl_re.append(np.tile(_absorber_r, (_n_new, 1)))
                all_refl_im.append(np.tile(_absorber_r, (_n_new, 1)))
                all_diff.append(np.zeros(_n_new, np.float64))
                mat_groups.append((
                    _tri_cursor, _n_new,
                    _tube_tint_rgb, _tube_tint_rgb, _tube_tint_rgb,
                    1.7, 1.0, _GEO_FLAG_APERTURE_STOP,   # MAT_FLAG_ABSORBER
                ))
                _tri_cursor += _n_new

    # ── Camera body box shell ─────────────────────────────────────────────
    # Matte-black rectangular housing.  A circular port of radius = lens bore
    # is cut in the front face for the lens mount opening.  The body extends
    # from the aperture/mount z-plane rearward to behind the sensor.
    _body_mat      = MATERIAL_CATALOG.get("body", MATERIAL_CATALOG["blackened_steel"])
    _body_tint_rgb = _body_mat.tint[:3]

    _bs = preset.body
    _bx = float(_bs.width  * 0.5)
    _by = float(_bs.height * 0.5)
    _bz = float(_bs.depth  * 0.5)
    _sensor_z = float(getattr(preset.sensor, "z_pos",
                       getattr(preset.sensor, "z_vertex", 0.0))) if preset.sensor else 0.0
    _body_z_front = float(getattr(preset.aperture_stop, "z_pos",
                           float(_bz)))  # align front face to mount/aperture z
    _body_z_back  = _body_z_front - float(_bs.depth)
    _body_port_r  = _tube_r_inner if (_el_r_maxes and _el_z_vals) else 0.0

    _body_bv, _body_bn = _triangulate_box_shell_walls(
        _bx, _by, _body_z_front, _body_z_back,
        port_r=_body_port_r, n_div=32)
    if len(_body_bv) > 0:
        _n_new = len(_body_bv)
        all_verts.append(_body_bv);   all_normals.append(_body_bn)
        all_refl_re.append(np.tile(_absorber_r, (_n_new, 1)))
        all_refl_im.append(np.tile(_absorber_r, (_n_new, 1)))
        all_diff.append(np.full(_n_new, 0.05, np.float64))
        mat_groups.append((
            _tri_cursor, _n_new,
            _body_tint_rgb, _body_tint_rgb, _body_tint_rgb,
            1.58, 1.0, _GEO_FLAG_APERTURE_STOP,   # MAT_FLAG_ABSORBER
        ))
        _tri_cursor += _n_new

    # ── Sensor plane ─────────────────────────────────────────────────────
    sensor = preset.sensor
    if sensor is not None:
        s_verts, s_normals, _, _ = _triangulate_surface(sensor, n_rings=32, n_sectors=128)
        if len(s_verts) > 0:
            n_new = len(s_verts)
            refl_re, refl_im = _sensor_reflectances(n_bands)
            all_verts.append(s_verts);    all_normals.append(s_normals)
            all_refl_re.append(np.tile(refl_re, (n_new, 1)))
            all_refl_im.append(np.tile(refl_im, (n_new, 1)))
            all_diff.append(np.full(n_new, 0.9, np.float64))
            mat_groups.append((
                _tri_cursor, n_new,
                (0.25, 0.25, 0.30), (0.0, 0.0, 0.0), (0.2, 0.2, 0.25),
                1.5, 1.0,
                _GEO_FLAG_APERTURE_STOP,   # MAT_FLAG_ABSORBER — sensor kills rays
            ))
            _tri_cursor += n_new

        # Sensor-bay back wall: a deep-black flat absorber 2 mm behind the sensor.
        # Catches any rays transmitted through or scattered past the sensor face.
        _back_mat      = MATERIAL_CATALOG.get("back", MATERIAL_CATALOG["flocking_black"])
        _back_tint_rgb = _back_mat.tint[:3]
        _snr_z  = float(getattr(sensor, "z_pos", getattr(sensor, "z_vertex", 0.0)))
        _snr_r  = float(getattr(sensor, "r_max", 0.020))
        _back_v, _back_n = _triangulate_disk(
            lambda _r: 0.0, _snr_z - 0.002, 0.0, _snr_r * 1.05,
            n_rings=16, n_sectors=128, outward_normal_z=1.0)
        if len(_back_v) > 0:
            _n_new = len(_back_v)
            all_verts.append(_back_v);    all_normals.append(_back_n)
            all_refl_re.append(np.tile(np.zeros(n_bands, np.float64), (_n_new, 1)))
            all_refl_im.append(np.tile(np.zeros(n_bands, np.float64), (_n_new, 1)))
            all_diff.append(np.zeros(_n_new, np.float64))
            mat_groups.append((
                _tri_cursor, _n_new,
                _back_tint_rgb, _back_tint_rgb, _back_tint_rgb,
                1.6, 1.0, _GEO_FLAG_APERTURE_STOP,   # MAT_FLAG_ABSORBER
            ))
            _tri_cursor += _n_new

    # ── Scene object ──────────────────────────────────────────────────
    # ── Emitter discs ────────────────────────────────────────────────
    # Each EmitterSpec becomes a filled disc triangulated in world-space.
    # When enabled the disc carries _GEO_FLAG_EMITTER so the GPU shader marks
    # it MAT_FLAG_EMISSIVE; disabled emitters are geometry-only (black body).
    for _em in getattr(preset, 'emitters', []):
        _em_pos = np.asarray(_em.pos, np.float64)
        _em_nrm = np.asarray(_em.normal, np.float64)
        _em_nrm_len = np.linalg.norm(_em_nrm)
        if _em_nrm_len > 1e-12:
            _em_nrm = _em_nrm / _em_nrm_len
        else:
            _em_nrm = np.array([0., 0., -1.], np.float64)
        _em_r = float(_em.radius)
        _em_flag = _GEO_FLAG_EMITTER if _em.enabled else _GEO_FLAG_APERTURE_STOP

        # Build two tangent vectors perpendicular to the normal
        _up = np.array([0., 1., 0.], np.float64)
        if abs(_em_nrm @ _up) > 0.9:
            _up = np.array([1., 0., 0.], np.float64)
        _t1 = np.cross(_em_nrm, _up); _t1 /= np.linalg.norm(_t1)
        _t2 = np.cross(_em_nrm, _t1)

        _n_seg = 24
        _em_angles = np.linspace(0, 2 * math.pi, _n_seg, endpoint=False)
        _em_vlist: List[np.ndarray] = []
        _em_nlist: List[np.ndarray] = []
        for _ai in range(_n_seg):
            _a0, _a1 = _em_angles[_ai], _em_angles[(_ai + 1) % _n_seg]
            _p0 = _em_pos + _em_r * (math.cos(_a0) * _t1 + math.sin(_a0) * _t2)
            _p1 = _em_pos + _em_r * (math.cos(_a1) * _t1 + math.sin(_a1) * _t2)
            # Fan triangle: centre, p0, p1  (winding matches normal direction)
            tri = np.array([list(_em_pos), list(_p0), list(_p1)], np.float64).reshape(1, 9)
            _em_vlist.append(tri)
            _em_nlist.append(_em_nrm.reshape(1, 3))
        if _em_vlist:
            _ev  = np.concatenate(_em_vlist,  axis=0)
            _en  = np.concatenate(_em_nlist,  axis=0)
            _n_new = len(_ev)
            _em_col = tuple(float(c) for c in _em.color)
            # Emitters: perfect Lambertian emitter (diff=1) or black when disabled
            _em_diff_val = 1.0 if _em.enabled else 0.0
            all_verts.append(_ev)
            all_normals.append(_en)
            all_refl_re.append(np.zeros((_n_new, n_bands), np.float64))
            all_refl_im.append(np.zeros((_n_new, n_bands), np.float64))
            all_diff.append(np.full(_n_new, _em_diff_val, np.float64))
            mat_groups.append((
                _tri_cursor, _n_new,
                _em_col, _em_col, _em_col,
                1.0,          # IOR = air
                1.0,          # opacity
                _em_flag,
            ))
            _tri_cursor += _n_new

    # ── Projector back ────────────────────────────────────────────────────
    # A large disc emitter centred on the optical axis at z = sensor.z_pos
    # minus z_offset.  The disc faces +Z (toward the lens group) and emits
    # a spectrally-filtered white source shaped by the sensor QE weights.
    # Even when disabled the surface is present as a black absorber so the
    # inspector can see where the panel sits in the cross-section view.
    _pb = getattr(preset, 'projector_back', None)
    _pb_extra: dict = {}   # maps tri_cursor_start → (weights, power) for GPU packing
    if _pb is not None:
        _pb_r = preset.sensor.r_max * float(_pb.radius_scale)
        _pb_z = preset.sensor.z_pos - float(_pb.z_offset)
        _pb_pos = np.array([0.0, 0.0, _pb_z], np.float64)
        _pb_nrm = np.array([0.0, 0.0, 1.0], np.float64)   # emit toward +Z
        _pb_t1  = np.array([1.0, 0.0, 0.0], np.float64)
        _pb_t2  = np.array([0.0, 1.0, 0.0], np.float64)
        _pb_segs = 32
        _pb_angles = np.linspace(0, 2 * math.pi, _pb_segs, endpoint=False)
        _pb_vlist: List[np.ndarray] = []
        _pb_nlist: List[np.ndarray] = []
        for _ai in range(_pb_segs):
            _a0 = _pb_angles[_ai]
            _a1 = _pb_angles[(_ai + 1) % _pb_segs]
            _p0 = _pb_pos + _pb_r * (math.cos(_a0) * _pb_t1 + math.sin(_a0) * _pb_t2)
            _p1 = _pb_pos + _pb_r * (math.cos(_a1) * _pb_t1 + math.sin(_a1) * _pb_t2)
            tri = np.array([list(_pb_pos), list(_p0), list(_p1)], np.float64).reshape(1, 9)
            _pb_vlist.append(tri)
            _pb_nlist.append(_pb_nrm.reshape(1, 3))
        if _pb_vlist:
            _pb_ev = np.concatenate(_pb_vlist, axis=0)
            _pb_en = np.concatenate(_pb_nlist, axis=0)
            _pb_n_new = len(_pb_ev)
            _pb_weights = _pb.resolved_weights(n_bands) if _pb.enabled else [0.0] * n_bands
            _pb_power   = float(_pb.power) if _pb.enabled else 0.0
            _pb_diff_mean = float(np.mean(_pb_weights)) * _pb_power
            _pb_col = tuple(float(c) for c in _pb.color)
            _pb_flag = _GEO_FLAG_PROJECTOR_BACK if _pb.enabled else _GEO_FLAG_APERTURE_STOP
            _pb_extra[_tri_cursor] = (_pb_weights, _pb_power)
            all_verts.append(_pb_ev)
            all_normals.append(_pb_en)
            all_refl_re.append(np.zeros((_pb_n_new, n_bands), np.float64))
            all_refl_im.append(np.zeros((_pb_n_new, n_bands), np.float64))
            all_diff.append(np.full(_pb_n_new, _pb_diff_mean, np.float64))
            mat_groups.append((
                _tri_cursor, _pb_n_new,
                _pb_col, _pb_col, _pb_col,
                1.0, 1.0,   # IOR=air, opacity=1
                _pb_flag,
            ))
            _tri_cursor += _pb_n_new

    # ── Scene object ──────────────────────────────────────────────────────
    if scene_object is not None:
        obj_type = scene_object.get("type", "sphere")
        if obj_type == "qr_target":
            from camera_software.qr_optical_validator import triangulate_qr_target

            for is_black, qv, qn in triangulate_qr_target(scene_object):
                if len(qv) == 0:
                    continue
                n_new = len(qv)
                level = 0.006 if is_black else 0.92
                spectral_r = np.full(n_bands, level, np.float64)
                rgb = (level, level, level)
                all_verts.append(qv); all_normals.append(qn)
                all_refl_re.append(np.tile(spectral_r, (n_new, 1)))
                all_refl_im.append(np.zeros((n_new, n_bands), np.float64))
                all_diff.append(np.full(n_new, 0.96, np.float64))
                mat_groups.append((
                    _tri_cursor, n_new, rgb, rgb, rgb, 1.0, 1.0, "",
                ))
                _tri_cursor += n_new
            sv = np.empty((0, 9), np.float64)
            sn = np.empty((0, 3), np.float64)
        elif obj_type == "sphere":
            obj_pos = np.asarray(scene_object.get("pos", [0., 0., 1.0]), np.float64)
            obj_r   = float(scene_object.get("radius", 0.05))
            sv, sn  = _triangulate_sphere_scene_object(obj_pos, obj_r)
        elif obj_type == "mesh":
            raw_v = np.asarray(scene_object["verts"],   np.float64)
            raw_n = np.asarray(scene_object["normals"], np.float64)
            if raw_v.shape[1] == 3:
                n_tri_obj = raw_v.shape[0] // 3
                sv = raw_v[:n_tri_obj*3].reshape(n_tri_obj, 9)
                sn = raw_n[:n_tri_obj*3:3]
            else:
                sv = raw_v; sn = raw_n
        else:
            sv = np.empty((0, 9), np.float64)
            sn = np.empty((0, 3), np.float64)

        if len(sv) > 0:
            n_new = len(sv)
            half_r = np.full(n_bands, 0.5, np.float64)
            all_verts.append(sv);   all_normals.append(sn)
            all_refl_re.append(np.tile(half_r, (n_new, 1)))
            all_refl_im.append(np.tile(np.zeros(n_bands, np.float64), (n_new, 1)))
            all_diff.append(np.full(n_new, 0.8, np.float64))
            mat_groups.append((
                _tri_cursor, n_new,
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.6, 0.6, 0.6),
                1.5, 1.0, "",
            ))
            _tri_cursor += n_new

    # ── Confinement box walls ─────────────────────────────────────────────
    # Six inward-facing walls bounding the entire optical bench.
    # Material:
    #   Interior (mat_in): near-perfect mirror — R=0.90, diff=0.05, abs=0.05
    #   Exterior (mat_out): dark borosilicate tint — R=0.02, diff=0.02, abs=0.95
    # Flagged MAT_FLAG_EMISSIVE so every interior surface arrival is recorded
    # into the 3D accumulation texture.  The outer simulation reads this back
    # as a directional vector field and re-injects it as an emitter.
    if confinement_box is not None:
        cb_min = np.asarray(confinement_box[0], np.float64)
        cb_max = np.asarray(confinement_box[1], np.float64)
        bv, bn = _triangulate_box(cb_min, cb_max, n_div=16)
        if len(bv) > 0:
            n_new    = len(bv)
            wall_r   = np.full(n_bands, 0.90, np.float64)   # mirror interior for C tracer
            all_verts.append(bv)
            all_normals.append(bn)
            all_refl_re.append(np.tile(wall_r, (n_new, 1)))
            all_refl_im.append(np.tile(np.zeros(n_bands, np.float64), (n_new, 1)))
            all_diff.append(np.full(n_new, 0.05, np.float64))
            mat_groups.append((
                _tri_cursor, n_new,
                # mat_in  (R=reflectivity, G=diffusion, B=absorption) — mirror interior
                (0.90, 0.05, 0.05),
                # mat_out — dark borosilicate exterior absorbs incoming light
                (0.02, 0.02, 0.95),
                # albedo — near-black so no false-colour emission
                (0.03, 0.03, 0.03),
                1.52, 1.0,   # IOR=1.52 (borosilicate), opacity=1.0
                _GEO_FLAG_CONFINEMENT_WALL,
            ))
            _tri_cursor += n_new

    # ── Stack ─────────────────────────────────────────────────────────────
    if not all_verts:
        dummy_v = np.array([[[-1, -1, -10], [1, -1, -10], [0, 1, -10]]],
                           np.float64).reshape(1, 9)
        dummy_n = np.array([[0.0, 0.0, 1.0]], np.float64)
        all_verts.append(dummy_v);   all_normals.append(dummy_n)
        all_refl_re.append(np.zeros((1, n_bands), np.float64))
        all_refl_im.append(np.zeros((1, n_bands), np.float64))
        all_diff.append(np.zeros(1, np.float64))

    verts_all   = np.concatenate(all_verts,   axis=0)
    normals_all = np.concatenate(all_normals, axis=0)
    refl_re_all = np.concatenate(all_refl_re, axis=0)
    refl_im_all = np.concatenate(all_refl_im, axis=0)
    diff_all    = np.concatenate(all_diff,    axis=0)

    return (verts_all, normals_all, refl_re_all, refl_im_all, diff_all,
            ior_specs, context_specs, mat_groups, lens_material_specs)


# ─────────────────────────────────────────────────────────────────────────────
# Main builder — C extension tracer
# ─────────────────────────────────────────────────────────────────────────────

def build_tracer(
    preset: CameraPreset,
    lights: List[dict],
    scene_object: Optional[dict] = None,
    wavelengths_um: Sequence[float] = (0.450, 0.550, 0.650),
    confinement_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    exact_lens_transport: bool = True,
    allow_mesh_lens_transport_for_diagnostics: bool = False,
) -> Tuple[object, Dict[str, int]]:
    """Build a _spectral_kernels.RayTracer from a CameraPreset and scene.

    Returns
    -------
    tracer  : _spectral_kernels.RayTracer instance
    ctx_map : dict label -> context_id
    """
    if not exact_lens_transport and not allow_mesh_lens_transport_for_diagnostics:
        raise RuntimeError(
            "mesh-only camera optics are diagnostic-only; pass "
            "allow_mesh_lens_transport_for_diagnostics=True explicitly"
        )
    try:
        from _spectral_kernels import RayTracer as _CRayTracer
    except ImportError as exc:
        raise RuntimeError(
            "_spectral_kernels extension not found — rebuild the C extension first."
        ) from exc
    # Material flags have one Python/C++/GLSL source of truth.  The retired
    # RT_TRI_FLAG_* extension constants were a stale parallel ABI and are no
    # longer exported by the native module.
    from mat_flags import MAT_FLAG_TRANSMISSIVE, MAT_FLAG_APERTURE_STOP

    wavelengths_um = list(wavelengths_um)
    n_bands   = len(wavelengths_um)
    freq_hz   = np.array([_C_LIGHT / (wl * 1e-6) for wl in wavelengths_um], np.float64)
    speed_m_s = _C_LIGHT

    (verts_all, normals_all, refl_re_all, refl_im_all, diff_all,
     ior_specs, context_specs, _mat_groups, lens_material_specs) = _collect_optical_geometry(
         preset, scene_object, wavelengths_um, confinement_box)

    # Native compound-lens transport is canonical along scene X.  Designer
    # authoring uses optical Z, so rotate the complete authored scene once at
    # the backend boundary: (x,y,z)_designer -> (-z,x,y)_transport.  The sign
    # makes ordinary world->sensor travel +X, matching the production payload.
    points = verts_all.reshape(-1, 3).copy()
    points = points[:, [2, 0, 1]]
    points[:, 0] *= -1.0
    verts_all = np.ascontiguousarray(points.reshape(-1, 9), np.float64)
    normals_all = np.ascontiguousarray(normals_all[:, [2, 0, 1]], np.float64)
    normals_all[:, 0] *= -1.0
    context_specs = [
        (
            label,
            np.asarray(
                [-float(center[2]), float(center[0]), float(center[1])],
                np.float64,
            ),
            radius, n_re, n_im, dt_m,
        )
        for label, center, radius, n_re, n_im, dt_m in context_specs
    ]

    n_tri    = len(verts_all)
    atmo_abs = np.zeros(n_bands, np.float64)

    # Bake the per-group transmissive IOR (n_out) into the per-tri ior_real
    # bands. The new RayTracer has no separate set_tri_ior(n_in, n_out) call —
    # the surface IOR lives in the unified MatBuf, so we fold ior_specs into
    # the material data here before packing.
    ior_real_bands = np.ones((n_tri, n_bands), np.float64)
    ior_imag_bands = np.zeros((n_tri, n_bands), np.float64)
    transmittance_bands = np.zeros((n_tri, n_bands), np.float64)
    for tri_start, n_tris_seg, _n_in, n_out, _flag_str in ior_specs:
        if n_tris_seg > 0:
            sl = slice(tri_start, tri_start + n_tris_seg)
            ior_real_bands[sl, :] = float(n_out)
            if _flag_str == _GEO_FLAG_TRANSMISSIVE:
                # The current transport ABI enters Snell/Fresnel handling only
                # for explicitly transmissive MatBuf rows.  IOR alone describes
                # a boundary; it must not silently turn an opaque material into
                # glass.  These authored camera surfaces are glass by contract.
                transmittance_bands[sl, :] = np.clip(
                    1.0 - np.hypot(refl_re_all[sl], refl_im_all[sl]),
                    0.0, 1.0,
                )
    for (tri_start, n_tris_seg, _label, optical_mat,
         _medium_before, _medium_after) in lens_material_specs:
        sl = slice(tri_start, tri_start + n_tris_seg)
        ior_real_bands[sl, :] = np.asarray(
            [optical_mat.n_at(wl) for wl in wavelengths_um], np.float64)
        ior_imag_bands[sl, :] = float(optical_mat.k)

    from ray_tracer_bridge import per_tri_spectral_to_mat_buf as _per_tri_to_mat_buf
    mat_idx_arr, mat_buf_arr, mat_n_mats_int = _per_tri_to_mat_buf(
        refl_re_all.astype(np.float64, copy=False),
        refl_im_all.astype(np.float64, copy=False),
        np.tile(diff_all.astype(np.float64, copy=False)[:, None], (1, n_bands)),
        freq_hz,
        transmittance_bands=transmittance_bands,
        ior_real_bands=ior_real_bands,
        ior_imag_bands=ior_imag_bands,
    )

    tracer = _CRayTracer(
        n_tri,
        verts_all.astype(np.float64, copy=False),
        normals_all.astype(np.float64, copy=False),
        mat_idx_arr,
        mat_buf_arr,
        int(mat_n_mats_int),
        freq_hz,
        speed_m_s,
        atmo_abs,
    )

    # Install exact compound-medium adjacency.  Triangle normals point toward
    # +Z, so the positive side is the medium before the boundary in the
    # front-to-back prescription and the negative side is the medium after it.
    # Mat indices are fixed-stride material records and are therefore valid as
    # persistent medium identifiers in both CPU and GPU transport.
    medium_indices: dict[tuple, int] = {}

    def _medium_key(material: OpticalMaterial) -> tuple:
        return (
            material.name, material.n_d, tuple(material.B), tuple(material.C),
            material.k, material.scatter_albedo,
        )

    for (tri_start, n_tris_seg, _label, boundary_mat,
         medium_pos, medium_neg) in lens_material_specs:
        boundary_idx = int(mat_idx_arr[tri_start])
        boundary_n = float(boundary_mat.n_at(float(np.mean(wavelengths_um))))
        if boundary_n > 1.0 + 1.0e-8:
            medium_indices[_medium_key(boundary_mat)] = boundary_idx

        def _resolved_medium_index(material: OpticalMaterial) -> int:
            if float(material.n_at(float(np.mean(wavelengths_um)))) <= 1.0 + 1.0e-8:
                return -1
            return medium_indices.get(_medium_key(material), boundary_idx)

        tracer.set_tri_boundary_media(
            tri_start, n_tris_seg,
            _resolved_medium_index(medium_pos),
            _resolved_medium_index(medium_neg),
        )

    # Map abstract flag tokens to C extension constants
    _flag_map = {
        _GEO_FLAG_TRANSMISSIVE:  MAT_FLAG_TRANSMISSIVE,
        _GEO_FLAG_APERTURE_STOP: MAT_FLAG_APERTURE_STOP,
    }
    for tri_start, n_tris, _n_in, _n_out, flag_str in ior_specs:
        if n_tris > 0 and flag_str in _flag_map:
            tracer.set_tri_ior(tri_start, n_tris, _flag_map[flag_str])

    if exact_lens_transport:
        from .compound_optics import CompoundLens
        from .lens_assembly import LensAssemblySpec
        from camera_software.camera_build import RebuiltCameraArtifact

        surface_ids = {
            str(label): np.arange(
                int(tri_start), int(tri_start) + int(n_tris), dtype=np.int32
            )
            for (
                tri_start, n_tris, label, _material,
                _medium_before, _medium_after,
            ) in lens_material_specs
        }
        surface_radius = {
            str(element.label): float(getattr(element.surface, "R", 0.0))
            for element in preset.lens_group.elements
        }
        group_names = sorted({
            label.rsplit("_", 1)[0]
            for label in surface_ids
            if label.endswith("_front") or label.endswith("_back")
        })
        lens_surface_groups = []
        for group in group_names:
            front_label = f"{group}_front"
            back_label = f"{group}_back"
            if front_label not in surface_ids or back_label not in surface_ids:
                continue
            lens_surface_groups.append((
                surface_ids[front_label],
                surface_ids[back_label],
                surface_radius.get(front_label, 0.0),
                surface_radius.get(back_label, 0.0),
            ))
        if not lens_surface_groups:
            if not allow_mesh_lens_transport_for_diagnostics:
                raise RuntimeError(
                    "camera transport refused mesh-only lens geometry; "
                    "exact parametric front/back proxy groups were not found"
                )
        else:
            optics = CompoundLens.from_preset(
                preset,
                wavelengths_um=wavelengths_um,
                axial_scale=-1.0,
            )
            assembly = LensAssemblySpec()
            assembly.set_optics(optics, mode=LensAssemblySpec.MODE_PARAMETRIC)
            artifact = RebuiltCameraArtifact.create(
                scene_config={"coordinate_system": "canonical-optical-x"},
                lens_assembly=assembly,
                lens_surface_groups=lens_surface_groups,
                scene_lenses=(),
                wavelengths_nm=np.asarray(wavelengths_um, np.float64) * 1000.0,
                sensor_center=(
                    -float(preset.sensor.z_pos), 0.0, 0.0,
                ),
                diffraction_model="disabled_for_geometric_fast_preview",
            )
            artifact.register_exact_transport(tracer, verts_all.reshape(-1, 3, 3))
            if int(getattr(assembly, "_entrance_gid", -1)) < 0:
                raise RuntimeError(
                    "camera transport refused zero exact parametric groups"
                )
            print(
                f"[camera-transport] exact required; {artifact.describe()}",
                flush=True,
            )

    ctx_map: Dict[str, int] = {}
    for label, center, radius, n_re, n_im, dt_m in context_specs:
        n_substeps = min(max(10, int(math.ceil(radius * 2.0 / dt_m))), 50000)
        cid = tracer.add_scale_context(
            pos=center, radius=radius, scale_type=1,
            dt_m=dt_m, n_substeps=n_substeps, n_real=n_re, n_imag=n_im,
        )
        ctx_map[label] = cid

    return tracer, ctx_map


# ─────────────────────────────────────────────────────────────────────────────
# GPU scene builder — packs geometry into the _gpu_ray_field() SSBO format
# ─────────────────────────────────────────────────────────────────────────────

def build_gpu_scene(
    preset: CameraPreset,
    lights: List[dict],
    scene_object: Optional[dict] = None,
    wavelengths_um: Sequence[float] = (0.450, 0.550, 0.650),
    confinement_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> tuple:
    """Pack a CameraPreset into the GPU SSBO format consumed by _gpu_ray_field().

    Returns
    -------
    packed_geom  : (N_tri, 16) float32   — TriGeomBuf rows (binding 0)
    packed_shade : (N_tri, 16) float32   — TriShadeBuf rows (binding 9)
    mat_buf      : (N_mat * MAX_SPECTRAL_BANDS, 12) float32 — MatBuf (binding 10)
    bvh_tris     : (N_tri, 3, 3) float32  — vertex triples for _build_gpu_bvh()
    context_buf  : (N_ctx, 8)   float32
                   ScaleContext SSBO rows: [cx,cy,cz,r, dt_m,n_re,n_im, type=1]
    ctx_map      : dict label -> row-index into context_buf
    bounds       : (bmin, bmax)  each float32 (3,)
    source_buf   : (N_src, 9)   float32  [px,py,pz, dx,dy,dz, model_int, model_param, amp_weight]
    """
    from mat_flags import (
        MAT_FLAG_EMISSIVE, MAT_FLAG_ABSORBER, MAT_FLAG_TRANSMISSIVE,
    )
    wavelengths_um = list(wavelengths_um)

    (verts_all, normals_all, _refl_re, _refl_im, _diff,
     ior_specs, context_specs, mat_groups, lens_material_specs) = _collect_optical_geometry(
         preset, scene_object, wavelengths_um, confinement_box)

    # ── Build per-triangle material arrays ────────────────────────────────
    n_tri = len(verts_all)
    mat_in_rgb  = np.zeros((n_tri, 3), np.float32)
    mat_out_rgb = np.zeros((n_tri, 3), np.float32)
    albedo_rgb  = np.zeros((n_tri, 3), np.float32)
    ior_col     = np.ones(n_tri, np.float32)
    opac_col    = np.ones(n_tri, np.float32)
    flags_u32   = np.zeros(n_tri, np.uint32)

    # Map abstract geometry tokens to MAT_FLAG bit values from mat_flags
    # (the single Python source of truth for both backends).
    _glsl_flag = {
        _GEO_FLAG_TRANSMISSIVE:     np.uint32(MAT_FLAG_TRANSMISSIVE),
        _GEO_FLAG_APERTURE_STOP:    np.uint32(MAT_FLAG_ABSORBER),
        _GEO_FLAG_EMITTER:          np.uint32(MAT_FLAG_EMISSIVE),
        _GEO_FLAG_PROJECTOR_BACK:   np.uint32(MAT_FLAG_EMISSIVE),
        _GEO_FLAG_CONFINEMENT_WALL: np.uint32(MAT_FLAG_EMISSIVE),  # records arrivals
    }

    for tri_start, n_tris, mi_rgb, mo_rgb, alb_rgb, ior, opa, flag_str in mat_groups:
        sl = slice(tri_start, tri_start + n_tris)
        mat_in_rgb[sl]  = mi_rgb
        mat_out_rgb[sl] = mo_rgb
        albedo_rgb[sl]  = alb_rgb
        ior_col[sl]     = ior
        opac_col[sl]    = opa
        if flag_str in _glsl_flag:
            flags_u32[sl] = _glsl_flag[flag_str]

    # Reinterpret uint32 flags as float32 bits (same bit pattern the GLSL reads)
    flags_f32 = np.frombuffer(flags_u32.tobytes(), np.float32)

    # ── Pack the shared split GPU contract ───────────────────────────────
    # TriGeom is the hot BVH row; TriShade is fetched only after a hit.  MatBuf
    # is the same fixed-stride spectral material block used by the production
    # ray pipeline.  Keep these three blocks separate: recombining the old
    # 32-float triangle row would undo the cache-line migration.
    tris = verts_all.reshape(-1, 3, 3).astype(np.float32)
    nrm  = normals_all.astype(np.float32)

    packed_geom = np.zeros((n_tri, 16), np.float32)
    packed_geom[:, 0:3]   = tris[:, 0, :]
    packed_geom[:, 4:7]   = tris[:, 1, :] - tris[:, 0, :]
    packed_geom[:, 8:11]  = tris[:, 2, :] - tris[:, 0, :]
    packed_geom[:, 12:15] = nrm
    packed_geom[:, 15]    = flags_f32

    raw16 = np.zeros((n_tri, 16), np.float32)
    raw16[:, 0:3] = mat_in_rgb
    raw16[:, 3:6] = mat_out_rgb
    raw16[:, 6:9] = albedo_rgb
    raw16[:, 9]   = ior_col
    raw16[:, 10]  = opac_col
    raw16[:, 11]  = flags_f32
    raw16[:, 12:15] = -1.0

    from material_db import MaterialDatabase
    mat_db = MaterialDatabase.instance()
    freq_hz = np.asarray([_C_LIGHT / (wl * 1e-6) for wl in wavelengths_um],
                         dtype=np.float64)

    # Resolve only unique authored rows.  A camera shell contains tens of
    # thousands of triangles but normally fewer than a dozen materials; doing
    # content registration per triangle made every small bench edit needlessly
    # Python-bound.
    unique_rows, inverse = np.unique(raw16, axis=0, return_inverse=True)
    unique_indices = np.fromiter(
        (mat_db.ensure_mat16(row) for row in unique_rows),
        dtype=np.int32,
        count=len(unique_rows),
    )
    mat_idx = unique_indices[inverse]

    # Lens glass gets a named spectral row sampled directly from its Sellmeier
    # model.  Geometry remains compact (one uint material index per triangle),
    # while the shared fixed-stride MatBuf retains wavelength-specific n + ik
    # for the current transport grid and future complex-field consumption.
    if freq_hz.size == 1:
        bandwidths = np.asarray([max(float(freq_hz[0]) * 0.01, 1.0)], np.float64)
    else:
        bandwidths = np.asarray([
            max(float(np.min(np.abs(np.delete(freq_hz, i) - f))) * 0.10, 1.0)
            for i, f in enumerate(freq_hz)
        ], np.float64)
    for (tri_start, n_tris_seg, _label, optical_mat,
         _medium_before, _medium_after) in lens_material_specs:
        signature = (
            optical_mat.name, optical_mat.n_d, tuple(optical_mat.B),
            tuple(optical_mat.C), optical_mat.k, optical_mat.scatter_albedo,
            tuple(float(wl) for wl in wavelengths_um),
        )
        material_key = "camera_optical:" + repr(signature)
        if material_key not in mat_db:
            bands = []
            for wl, f, bw in zip(wavelengths_um, freq_hz, bandwidths):
                n_real = float(optical_mat.n_at(float(wl)))
                fresnel = float(_fresnel_reflectance(1.0, n_real))
                diffuse = float(np.clip(optical_mat.scatter_albedo, 0.0, 1.0))
                bands.append({
                    "center_hz": float(f),
                    "bandwidth_hz": float(bw),
                    "reflectance": fresnel,
                    "transmittance": float(np.clip(1.0 - fresnel - diffuse, 0.0, 1.0)),
                    "diffuse_frac": diffuse,
                    "emission": 0.0,
                    "reemission": 0.0,
                    "ior_real": n_real,
                    "ior_imag": float(optical_mat.k),
                })
            mat_db.register(material_key, {
                "name": optical_mat.name,
                "domain": "em_optical",
                "albedo": [0.9, 0.9, 0.9],
                "roughness": float(np.clip(optical_mat.roughness, 0.0, 1.0)),
                "metallic": 1.0 if optical_mat.is_mirror else 0.0,
                "transmission": 0.0 if optical_mat.is_opaque else 1.0,
                "ior": float(optical_mat.n_d),
                "opacity": 1.0 if optical_mat.is_opaque else 0.0,
                "spectral_bands": bands,
            })
        sl = slice(tri_start, tri_start + n_tris_seg)
        mat_idx[sl] = mat_db.index_of(material_key)

    mat_buf = np.ascontiguousarray(mat_db.build_mat_buf(freq_hz=freq_hz),
                                   dtype=np.float32)
    mat_idx_bits = np.frombuffer(mat_idx.astype(np.uint32).tobytes(), np.float32)

    packed_shade = np.zeros((n_tri, 16), np.float32)
    packed_shade[:, 0:3]   = mat_in_rgb
    packed_shade[:, 3]     = ior_col
    packed_shade[:, 4:7]   = mat_out_rgb
    packed_shade[:, 7]     = opac_col
    packed_shade[:, 8:11]  = albedo_rgb
    packed_shade[:, 12:14] = -1.0
    packed_shade[:, 14]    = mat_idx_bits

    # Confinement walls: enable the GLSL emissive splat so every surface arrival
    # is deposited into the 3D volume texture.
    # Preserve the legacy surface-arrival splat gate.  Source launch itself is
    # handled by source_buf; this strength is only used when a traced ray hits
    # one of the specially tagged surfaces.
    for _ts, _nt, *_, _fs in mat_groups:
        if _fs == _GEO_FLAG_CONFINEMENT_WALL:
            _sl = slice(_ts, _ts + _nt)
            packed_shade[_sl, 11] = 1.0
            packed_shade[_sl, 12] = 1.0
        elif _fs == _GEO_FLAG_EMITTER:
            _sl = slice(_ts, _ts + _nt)
            packed_shade[_sl, 11] = 1.0
            packed_shade[_sl, 12] = 1.0
        elif _fs == _GEO_FLAG_PROJECTOR_BACK:
            # diff_all already carries band-averaged power × mean_weight.
            # Enable the emissive splat gate identically to regular emitters;
            # per-band spectral curve modulation is future GPU shader work.
            _sl = slice(_ts, _ts + _nt)
            packed_shade[_sl, 11] = 1.0
            packed_shade[_sl, 12] = 1.0

    packed_geom = np.ascontiguousarray(packed_geom)
    packed_shade = np.ascontiguousarray(packed_shade)

    bvh_tris = tris  # (N_tri, 3, 3) float32 — ready for _build_gpu_bvh()

    # ── ScaleContext SSBO: (N_ctx, 8) float32 ─────────────────────────────
    # [cx,cy,cz,r, dt_m, n_re, n_im, scale_type=1.0]
    ctx_map: Dict[str, int] = {}
    if context_specs:
        context_buf = np.zeros((len(context_specs), 8), np.float32)
        for i, (label, center, radius, n_re, n_im, dt_m) in enumerate(context_specs):
            context_buf[i, 0:3] = center.astype(np.float32)
            context_buf[i, 3]   = float(radius)
            context_buf[i, 4]   = float(dt_m)
            context_buf[i, 5]   = float(n_re)
            context_buf[i, 6]   = float(n_im)
            context_buf[i, 7]   = 1.0   # RT_SCALE_WAVE
            ctx_map[label] = i
    else:
        context_buf = np.zeros((0, 8), np.float32)

    # ── Scene bounds ──────────────────────────────────────────────────────
    all_pts = tris.reshape(-1, 3)
    bmin = all_pts.min(axis=0)
    bmax = all_pts.max(axis=0)

    # ── Pre-baked ray buffer: (N_rays, 12) float32 ───────────────────────
    # Each row is a fully-sampled ray: [pos_xyz, amp_w, dir_xyz, 0,
    #                                    freq_hz, phase, energy, 0]
    # Direction and phase are baked CPU-side from each source's directional
    # model and PhaseState — the GPU PASS_FORWARD just reads and traces them.
    _PREBAKE_N = 4096
    _WL_UM = np.asarray(wavelengths_um, np.float64)
    from .ray_order import RayOrder as _RayOrder
    from .emitter_profile import DirectionalModel as _DM, PhaseState as _PS, \
        CoherenceModel as _CM, AngularDistribution as _AD

    # Collect EmitterSpec sources
    _spec_list: list = []
    for lgt in lights:
        if "_spec" in lgt:
            _spec_list.append(lgt["_spec"])
    for em in getattr(preset, 'emitters', []):
        if em.enabled:
            _spec_list.append(em)

    if _spec_list:
        _order = _RayOrder.from_emitter_specs(
            _spec_list, wavelengths_um=_WL_UM, n_spatial_samples=1,
        )
        source_buf = _order.bake_rays(_PREBAKE_N, seed=42)
    else:
        # Legacy plain-dict lights and/or projector_back only — build a minimal
        # RayOrder via from_lights_list, then bake from it.
        _legacy_lights = [lgt for lgt in lights if "_spec" not in lgt]
        _pb = getattr(preset, 'projector_back', None)
        if _pb is not None and _pb.enabled:
            _pb_z = preset.sensor.z_pos - float(_pb.z_offset)
            _legacy_lights = list(_legacy_lights) + [{
                "pos": [0.0, 0.0, float(_pb_z)],
                "dir": [0.0, 0.0, 1.0],
                "power": float(_pb.power),
                "label": "projector_back",
            }]
        if not _legacy_lights:
            _legacy_lights = [{"pos": [0., 0., 5.], "dir": [0., 0., -1.],
                                "power": 1.0, "label": "default"}]
        _order = _RayOrder.from_lights_list(_legacy_lights, wavelengths_um=_WL_UM)
        source_buf = _order.bake_rays(_PREBAKE_N, seed=42)

    return packed_geom, packed_shade, mat_buf, bvh_tris, context_buf, ctx_map, (bmin, bmax), source_buf


# ─────────────────────────────────────────────────────────────────────────────
# Emitter light synthesis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _emitter_lights_from_preset(preset: CameraPreset) -> List[dict]:
    """Convert enabled EmitterSpec and ProjectorBackSpec to light dicts.

    Each enabled EmitterSpec becomes one directional light (disc centre,
    emission normal, EmitterSpec.power).

    A ProjectorBackSpec, when enabled, contributes one large axial source at
    the back panel position facing +Z (into the lens group).  Its power is
    scaled by the mean spectral weight so spectrally-narrow presets don't
    blind the tracer with full white power.
    """
    lights: List[dict] = []
    for em in getattr(preset, 'emitters', []):
        if em.enabled:
            lights.append({
                "pos":   list(em.pos),
                "dir":   list(em.normal),
                "power": float(em.power),
                "color": list(em.color),
            })
    pb = getattr(preset, 'projector_back', None)
    if pb is not None and pb.enabled:
        _pb_z   = preset.sensor.z_pos - float(pb.z_offset)
        _n_wl   = len(getattr(preset, 'wavelengths', [0.55]))
        _w_mean = float(np.mean(pb.resolved_weights(_n_wl)))
        lights.append({
            "pos":   [0.0, 0.0, _pb_z],
            "dir":   [0.0, 0.0,  1.0],   # into lens
            "power": float(pb.power) * _w_mean,
            "color": list(pb.color),
        })
    return lights


# ─────────────────────────────────────────────────────────────────────────────
# Source helpers — convert light descriptors to PyRayTracer source arrays
# ─────────────────────────────────────────────────────────────────────────────

def lights_to_sources(
    lights: List[dict],
    aim_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert light dicts to (src_pos, src_dir, src_directivity) float64 arrays.

    aim_point : if given, all lights point toward this world-space position;
                otherwise they use each light's own ``"dir"`` key or -Z.

    Raises ValueError if ``lights`` is empty — callers must not invoke the
    tracer with zero sources.
    """
    if not lights:
        raise ValueError(
            "lights_to_sources: empty lights list — no sources to trace. "
            "Provide at least one EmitterSpec or placed light before spawning rays."
        )

    n = len(lights)
    pos_arr  = np.zeros((n, 3), np.float64)
    dir_arr  = np.zeros((n, 3), np.float64)
    dirp_arr = np.zeros(n, np.float64)

    for i, lgt in enumerate(lights):
        p = np.asarray(lgt.get("pos", [0., 0., 5.0]), np.float64)
        pos_arr[i] = p

        if aim_point is not None:
            d = aim_point - p
        else:
            d = np.array([0., 0., -1.0], np.float64)
        nl = np.linalg.norm(d)
        dir_arr[i] = d / nl if nl > 1e-12 else np.array([0., 0., -1.])

        # directivity: power ≥ 1 means more focused; treat "power" as emission
        # directivity exponent (1.0 = cosine, 2.0 = tight spot)
        dirp_arr[i] = max(0.5, float(lgt.get("power", 1.0)))

    return pos_arr, dir_arr, dirp_arr
