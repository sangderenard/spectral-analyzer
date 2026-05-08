"""ray_tracer_bridge.py
=======================
Python integration layer between CavityScene room geometry and the
_spectral_kernels.RayTracer C extension.

Converts PolygonalRoom / CircularRoom / MeshRoom geometry into flat numpy
arrays, derives frequency bands from source profiles or a log-spaced default,
computes frequency-dependent complex reflectances (Kramers-Kronig consistent
to first order), and calls the C tracer.

Typical usage
-------------
    from ray_tracer_bridge import trace_cavity_scene
    from opengl_widget import RayAccumulatorWidget

    segs, meta = trace_cavity_scene(scene, n_rays=512, max_bounces=10, n_bands=12)

    widget = RayAccumulatorWidget(
        n_sources=meta['n_sources'],
        n_bands=meta['n_bands'],
    )
    widget.update_segments(segs)
    widget.set_playhead(0.0)   # advance each frame: t = elapsed / max_path_length * c

The returned segment buffer has shape (N_seg, 12), float32:
    columns: x0,y0,z0, x1,y1,z1, src_id, bounce, band, amplitude, phase, path_len
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # avoid circular; cavity_engine types used by annotation strings only

try:
    from _spectral_kernels import RayTracer as _CRayTracer
    from _spectral_kernels import FieldSolver as _CFieldSolver
    _HAS_C_TRACER = True
    _HAS_FIELD_SOLVER = True
except ImportError:
    _CRayTracer = None
    _CFieldSolver = None
    _HAS_C_TRACER = False
    _HAS_FIELD_SOLVER = False

try:
    from spectral_material import (
        Material            as _SpectralMaterial,
        WallBand            as _WallBand,
        parse_wall_bands    as _parse_wall_bands,
        wall_band_at        as _wall_band_at,
        materials_to_tracer_mat_props as _materials_to_tracer_mat_props,
    )
    _HAS_SPECTRAL_MAT = True
except ImportError:
    _SpectralMaterial        = None
    _WallBand                = None
    _parse_wall_bands        = None
    _wall_band_at            = None
    _materials_to_tracer_mat_props = None
    _HAS_SPECTRAL_MAT = False


# ---------------------------------------------------------------------------
# Room triangulation
# ---------------------------------------------------------------------------

def _triangulate_polygonal_room(room) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Triangulate a PolygonalRoom into flat triangle arrays.

    Coordinate convention: vertices_xy = [(x, z), ...] in the XZ horizontal
    plane; Y is up.  Floor at Y=0, ceiling at Y=height.

    Returns
    -------
    verts        : (N_tri, 3, 3) float64 — triangle vertex triples
    normals      : (N_tri, 3)    float64 — outward unit normals
    mat_props    : (N_tri, 3)    float64 — [reflectivity, diffusion, absorption]
    n_room_tris  : int — number of room-shell triangles before baffles start;
                   verts[n_room_tris:] contains only the baffle panel triangles.
    """
    vxy = np.asarray(room.vertices_xy, dtype=np.float64)   # (N, 2): x, z
    N   = len(vxy)
    h   = float(room.height)

    # Build 3D floor vertices (Y=0) and ceiling vertices (Y=height).
    floor_v = np.column_stack([vxy[:, 0], np.zeros(N),    vxy[:, 1]])
    ceil_v  = np.column_stack([vxy[:, 0], np.full(N, h),  vxy[:, 1]])

    tris, normals_out, mat_props = [], [], []

    def _push_tri(v0, v1, v2, inward_n_hint, ref, diff, abso):
        """Append a triangle with normal forced to agree with inward_n_hint."""
        e1 = v1 - v0
        e2 = v2 - v0
        n  = np.cross(e1, e2)
        nm = np.linalg.norm(n)
        if nm < 1e-12:
            return  # degenerate
        n /= nm
        if np.dot(n, inward_n_hint) < 0.0:
            n = -n
            v1, v2 = v2, v1   # flip winding
        tris.append(np.array([v0, v1, v2], dtype=np.float64))
        normals_out.append(n)
        mat_props.append([ref, diff, abso])

    up   = np.array([0.0,  1.0, 0.0])
    down = np.array([0.0, -1.0, 0.0])

    # Floor: fan from vertex 0, normals pointing up.
    if getattr(room, 'closed_floor', True):
        for i in range(1, N - 1):
            _push_tri(floor_v[0], floor_v[i], floor_v[i + 1], up,
                      room.floor_reflectivity, room.floor_diffusion,
                      room.floor_absorption)

    # Ceiling: fan from vertex 0, normals pointing down.
    if getattr(room, 'closed_roof', True):
        for i in range(1, N - 1):
            _push_tri(ceil_v[0], ceil_v[i + 1], ceil_v[i], down,
                      room.roof_reflectivity, room.roof_diffusion,
                      room.roof_absorption)

    # Walls: each polygon edge extruded into a quad → 2 triangles.
    for i in range(N):
        j  = (i + 1) % N
        v00 = floor_v[i]
        v10 = floor_v[j]
        v01 = ceil_v[i]
        v11 = ceil_v[j]

        # Inward normal: perpendicular to edge in XZ plane, pointing toward
        # room interior.  For a CCW polygon (viewed from above), the inward
        # normal rotates 90° left of the edge direction.
        edge_xz  = np.array([v10[0] - v00[0], 0.0, v10[2] - v00[2]])
        inward_n = np.array([-edge_xz[2], 0.0, edge_xz[0]])
        in_norm  = np.linalg.norm(inward_n)
        if in_norm > 1e-9:
            inward_n /= in_norm

        _push_tri(v00, v10, v11, inward_n,
                  room.wall_reflectivity, room.wall_diffusion,
                  room.wall_absorption)
        _push_tri(v00, v11, v01, inward_n,
                  room.wall_reflectivity, room.wall_diffusion,
                  room.wall_absorption)

    # Record split point: everything appended after this is a baffle panel.
    n_room_tris = len(tris)

    # Optional baffles / panels.
    for panel in getattr(room, 'baffles', []):
        _add_baffle(panel, tris, normals_out, mat_props)

    return (np.array(tris,       dtype=np.float64),
            np.array(normals_out, dtype=np.float64),
            np.array(mat_props,   dtype=np.float64),
            n_room_tris)


def _add_baffle(panel, tris, normals_out, mat_props) -> None:
    """Convert a CavityPanel into triangles and append to the lists.

    When the panel has no material_mask the quad is emitted as two triangles
    (existing behaviour, unchanged).

    When a float32 material_mask (H, W) is present the panel quad is
    subdivided into an H×W grid.  Each cell is either:
      - mask == 0  → open (no triangles, physically absent from geometry)
      - mask  > 0  → solid; material props are scaled by the mask density so
                     intermediate values continuously vary reflectivity,
                     diffusion, and absorption across the panel surface.

    panel.half_size controls the physical extent of the quad (default 1.0 m).
    """
    p = np.asarray(panel.point,  dtype=np.float64)
    n = np.asarray(panel.normal, dtype=np.float64)
    nm = np.linalg.norm(n)
    if nm < 1e-9:
        return
    n /= nm

    # Build a local frame perpendicular to n.
    up  = np.array([0.0, 1.0, 0.0])
    ax1 = np.cross(n, up if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    ax1_nm = np.linalg.norm(ax1)
    if ax1_nm < 1e-9:
        return
    ax1 /= ax1_nm
    ax2 = np.cross(n, ax1)

    _default_half = float(getattr(panel, 'half_size', 1.0))
    half_h = float(getattr(panel, 'half_h', None) or _default_half)  # along ax1
    half_w = float(getattr(panel, 'half_w', None) or _default_half)  # along ax2
    ref  = getattr(panel, 'reflectivity', 0.7)
    diff = getattr(panel, 'diffusion',    0.35)
    abso = getattr(panel, 'absorption',   0.15)

    mask = getattr(panel, 'material_mask', None)

    def _push_tri(v0, v1, v2, r, d, a):
        e1 = v1 - v0;  e2 = v2 - v0
        tn = np.cross(e1, e2)
        tnm = np.linalg.norm(tn)
        if tnm < 1e-12:
            return
        tn /= tnm
        if np.dot(tn, n) < 0.0:
            tn = -tn;  v1, v2 = v2, v1
        tris.append(np.array([v0, v1, v2], dtype=np.float64))
        normals_out.append(tn)
        mat_props.append([r, d, a])

    outline_pts = getattr(panel, 'outline_pts', None)
    if outline_pts is not None and abs(float(n[2])) > 0.9:
        _add_outline_plate_baffle(panel, np.asarray(outline_pts, dtype=np.float64),
                                  tris, normals_out, mat_props)
        return

    if mask is None:
        # No mask — single quad, two triangles.
        v00 = p - ax1 * half_h - ax2 * half_w
        v10 = p + ax1 * half_h - ax2 * half_w
        v11 = p + ax1 * half_h + ax2 * half_w
        v01 = p - ax1 * half_h + ax2 * half_w
        _push_tri(v00, v10, v11, ref, diff, abso)
        _push_tri(v00, v11, v01, ref, diff, abso)
        return

    # Mask present — subdivide into grid cells.
    H, W = mask.shape
    # Precompute grid corner positions along each axis.
    u_edges = np.linspace(-half_h, half_h, W + 1)  # W+1 column edges along ax1
    v_edges = np.linspace(-half_w, half_w, H + 1)  # H+1 row edges along ax2

    for row in range(H):
        for col in range(W):
            density = float(mask[row, col])
            if density <= 0.0:
                continue  # Open — physically absent, no geometry here.

            # Four corners of this cell in 3-D.
            u0, u1 = u_edges[col],     u_edges[col + 1]
            v0, v1 = v_edges[row],     v_edges[row + 1]
            c00 = p + ax1 * u0 + ax2 * v0
            c10 = p + ax1 * u1 + ax2 * v0
            c11 = p + ax1 * u1 + ax2 * v1
            c01 = p + ax1 * u0 + ax2 * v1

            # Scale material properties by mask density.
            r = ref  * density
            d = diff * density
            a = abso * density

            _push_tri(c00, c10, c11, r, d, a)
            _push_tri(c00, c11, c01, r, d, a)


def _outline_ray_radius(outline: np.ndarray, cx: float, cy: float, theta: float) -> float:
    dx, dy = math.cos(theta), math.sin(theta)
    best = None
    n = len(outline)
    for i in range(n):
        x0, y0 = float(outline[i, 0]) - cx, float(outline[i, 1]) - cy
        x1, y1 = float(outline[(i + 1) % n, 0]) - cx, float(outline[(i + 1) % n, 1]) - cy
        ex, ey = x1 - x0, y1 - y0
        det = dx * (-ey) - dy * (-ex)
        if abs(det) < 1e-10:
            continue
        t = (x0 * (-ey) - y0 * (-ex)) / det
        u = (dx * y0 - dy * x0) / det
        if t > 0.0 and -1e-6 <= u <= 1.0 + 1e-6:
            best = t if best is None else max(best, t)
    return float(best if best is not None else 0.0)


def _add_outline_plate_baffle(panel, outline: np.ndarray, tris, normals_out, mat_props) -> None:
    """Triangulate guitar plates as smooth outline geometry, not mask squares."""
    p = np.asarray(panel.point, dtype=np.float64)
    n = np.asarray(panel.normal, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)
    ref = float(getattr(panel, 'reflectivity', 0.7))
    diff = float(getattr(panel, 'diffusion', 0.35))
    abso = float(getattr(panel, 'absorption', 0.15))
    soundhole = getattr(panel, 'soundhole', None)

    def _push(v0, v1, v2):
        tn = np.cross(v1 - v0, v2 - v0)
        tnm = np.linalg.norm(tn)
        if tnm < 1e-12:
            return
        tn /= tnm
        if np.dot(tn, n) < 0.0:
            tn = -tn
            v1, v2 = v2, v1
        tris.append(np.array([v0, v1, v2], dtype=np.float64))
        normals_out.append(tn)
        mat_props.append([ref, diff, abso])

    def _push_skirt_quad(a_xy, b_xy, z0, z1):
        a0 = np.array([float(a_xy[0]), float(a_xy[1]), z0], dtype=np.float64)
        b0 = np.array([float(b_xy[0]), float(b_xy[1]), z0], dtype=np.float64)
        b1 = np.array([float(b_xy[0]), float(b_xy[1]), z1], dtype=np.float64)
        a1 = np.array([float(a_xy[0]), float(a_xy[1]), z1], dtype=np.float64)
        e = b0 - a0
        mid = 0.5 * (a0 + b0)
        cen = np.array([float(outline[:, 0].mean()), float(outline[:, 1].mean()), z0])
        radial = mid - cen
        tn = np.cross(e, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(tn) > 1e-12 and np.dot(tn, radial) < 0.0:
            _push(a0, b1, b0)
            _push(a0, a1, b1)
        else:
            _push(a0, b0, b1)
            _push(a0, b1, a1)

    def _add_seam_glue():
        eps = 0.003
        z0 = z - eps
        z1 = z + eps
        for ii in range(len(outline)):
            _push_skirt_quad(outline[ii], outline[(ii + 1) % len(outline)], z0, z1)

    z = float(p[2])
    if soundhole is None:
        cen = np.array([float(outline[:, 0].mean()), float(outline[:, 1].mean()), z])
        pts = [np.array([float(x), float(y), z]) for x, y in outline]
        for i in range(len(pts)):
            _push(cen, pts[i], pts[(i + 1) % len(pts)])
        _add_seam_glue()
        return

    cx, cy, hr = (float(soundhole[0]), float(soundhole[1]), float(soundhole[2]))
    n_theta = 192
    n_radial = 18
    verts = []
    for ri in range(n_radial + 1):
        frac = ri / float(n_radial)
        for ai in range(n_theta):
            th = 2.0 * math.pi * ai / float(n_theta)
            outer = _outline_ray_radius(outline, cx, cy, th)
            r = hr + frac * max(0.0, outer - hr)
            verts.append(np.array([cx + r * math.cos(th), cy + r * math.sin(th), z]))
    for ri in range(n_radial):
        row = ri * n_theta
        nxt = (ri + 1) * n_theta
        for ai in range(n_theta):
            a = row + ai
            b = row + ((ai + 1) % n_theta)
            c = nxt + ((ai + 1) % n_theta)
            d = nxt + ai
            _push(verts[a], verts[d], verts[c])
            _push(verts[a], verts[c], verts[b])
    _add_seam_glue()


def extract_scene_geometry_with_materials(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a scene and return instrument body geometry plus materials.

    Does NOT require the C RayTracer extension — pure Python triangulation.
    Returns only the baffle/panel triangles (instrument body), not the room walls.

    Returns
    -------
    geo_verts_flat : (N_tri, 9) float32 — triangle vertices flattened (v0+v1+v2)
    geo_normals    : (N_tri, 3) float32 — per-triangle outward normals
    geo_materials  : (N_tri, M) float32 — material columns. The first three
                     are [interior reflectivity, diffusion, absorption].
                     Columns 3:6, when present, describe the exterior face.
    """
    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    if room is None:
        return (np.zeros((0, 9), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 6), dtype=np.float32))

    n_room_tris = 0
    if hasattr(room, 'vertices_xy'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(room)
    elif hasattr(room, 'radius'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(
            _circular_room_to_polygon(room))
    elif hasattr(room, 'triangles'):
        verts, normals, mat_props = _triangulate_mesh_room(room)
    else:
        return (np.zeros((0, 9), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 6), dtype=np.float32))

    if len(verts) == 0:
        return (np.zeros((0, 9), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 6), dtype=np.float32))

    unified = _watertight_guitar_body_from_baffles(room)
    if unified is not None:
        verts, normals, mat_props = unified
        n_room_tris = 0

    # Show only the baffle/instrument panels, not the room shell.
    if n_room_tris > 0 and n_room_tris < len(verts):
        disp_verts = verts[n_room_tris:]
        disp_norms = normals[n_room_tris:]
        disp_mats = mat_props[n_room_tris:]
    else:
        disp_verts = verts
        disp_norms = normals
        disp_mats = mat_props

    n = len(disp_verts)
    geo_verts_flat = disp_verts.reshape(n, 9).astype(np.float32) if n else np.zeros((0, 9), dtype=np.float32)
    geo_normals    = disp_norms.astype(np.float32) if n else np.zeros((0, 3), dtype=np.float32)
    geo_materials  = disp_mats.astype(np.float32) if n else np.zeros((0, 6), dtype=np.float32)
    return geo_verts_flat, geo_normals, geo_materials


def extract_scene_geometry(scene) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a scene and return the instrument body geometry for display."""
    geo_verts_flat, geo_normals, _ = extract_scene_geometry_with_materials(scene)
    return geo_verts_flat, geo_normals


def _scene_trace_geometry(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a scene into trace-ready world-space triangles and normals."""
    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    if room is None:
        raise TypeError(f"Scene {type(scene).__name__} has neither 'room' nor 'geometry'")

    n_room_tris = 0
    if hasattr(room, 'vertices_xy'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(room)
    elif hasattr(room, 'radius'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(
            _circular_room_to_polygon(room))
    elif hasattr(room, 'triangles'):
        verts, normals, mat_props = _triangulate_mesh_room(room)
    else:
        raise TypeError(f"Unsupported room type: {type(room).__name__}")

    verts, normals, mat_props = _drop_absorptive_helper_shell(
        room, verts, normals, mat_props, n_room_tris)
    if len(verts) == 0:
        raise ValueError("Room triangulated to zero triangles")
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        np.asarray(mat_props, dtype=np.float64),
    )


def _coerce_trace_geometry(
        scene=None,
        verts: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
):
    if verts is None:
        return _scene_trace_geometry(scene)
    tri_verts = np.asarray(verts, dtype=np.float64)
    if tri_verts.ndim != 3 or tri_verts.shape[1:] != (3, 3):
        raise ValueError("verts must have shape (N, 3, 3)")
    if normals is None:
        e1 = tri_verts[:, 1, :] - tri_verts[:, 0, :]
        e2 = tri_verts[:, 2, :] - tri_verts[:, 0, :]
        tri_normals = np.cross(e1, e2)
        nm = np.linalg.norm(tri_normals, axis=1, keepdims=True)
        tri_normals = tri_normals / np.where(nm > 1e-12, nm, 1.0)
    else:
        tri_normals = np.asarray(normals, dtype=np.float64)
        if tri_normals.shape != (len(tri_verts), 3):
            raise ValueError("normals must have shape (N, 3)")
    return tri_verts, tri_normals, None


def _trace_float_dtype(*values) -> np.dtype:
    dtype = None
    for value in values:
        arr = np.asarray(value)
        if arr.dtype.kind == 'f':
            dtype = arr.dtype if dtype is None else np.result_type(dtype, arr.dtype)
    return np.dtype(np.float64 if dtype is None else dtype)


def _normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= eps:
        raise ValueError("Zero-length vector is not a valid ray direction")
    return v / n


def _camera_basis(forward, up, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = _normalize_vec(np.asarray(forward, dtype=dtype))
    upv = np.asarray(up, dtype=dtype)
    upv = upv - fwd * float(np.dot(upv, fwd))
    if np.linalg.norm(upv) <= 1e-12:
        fallback = np.array([0.0, 1.0, 0.0], dtype=dtype)
        if abs(float(np.dot(fwd, fallback))) > 0.95:
            fallback = np.array([1.0, 0.0, 0.0], dtype=dtype)
        upv = fallback - fwd * float(np.dot(fallback, fwd))
    upv = _normalize_vec(upv)
    right = _normalize_vec(np.cross(fwd, upv))
    upv = _normalize_vec(np.cross(right, fwd))
    return fwd, right, upv


def _raycast_triangles(
        verts: np.ndarray,
        normals: np.ndarray,
        origins,
        directions,
    owner_tags=None,
        max_distance: float = math.inf,
        eps: float = 1e-9,
        batch_size: int = 128,
) -> dict:
    """Intersect one or more rays against triangle soup with Moller-Trumbore."""
    dtype = _trace_float_dtype(origins, directions, verts)
    tris = np.asarray(verts, dtype=dtype)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError("verts must have shape (N, 3, 3)")
    norms = np.asarray(normals, dtype=dtype)
    if norms.shape != (len(tris), 3):
        raise ValueError("normals must have shape (N, 3)")
    tags = None
    if owner_tags is not None:
        tags = np.asarray(owner_tags)
        if len(tags) != len(tris):
            raise ValueError("owner_tags must have length N")

    ray_o = np.asarray(origins, dtype=dtype)
    ray_d = np.asarray(directions, dtype=dtype)
    single = ray_o.ndim == 1
    if single:
        ray_o = ray_o[None, :]
    if ray_d.ndim == 1:
        ray_d = ray_d[None, :]
    if ray_o.shape != ray_d.shape or ray_o.shape[1] != 3:
        raise ValueError("origins and directions must both have shape (R, 3) or (3,)")

    for i in range(len(ray_d)):
        ray_d[i] = _normalize_vec(ray_d[i], eps=eps)

    v0 = tris[:, 0, :]
    e1 = tris[:, 1, :] - v0
    e2 = tris[:, 2, :] - v0
    max_dist = float(max_distance)

    n_rays = len(ray_o)
    hit_t = np.full(n_rays, np.inf, dtype=dtype)
    hit_idx = np.full(n_rays, -1, dtype=np.int32)

    for start in range(0, n_rays, max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), n_rays)
        ro = ray_o[start:end]
        rd = ray_d[start:end]
        pvec = np.cross(rd[:, None, :], e2[None, :, :])
        det = np.sum(e1[None, :, :] * pvec, axis=2)
        det_ok = np.abs(det) > eps
        inv_det = np.zeros_like(det)
        inv_det[det_ok] = 1.0 / det[det_ok]

        tvec = ro[:, None, :] - v0[None, :, :]
        u = np.sum(tvec * pvec, axis=2) * inv_det
        qvec = np.cross(tvec, e1[None, :, :])
        v = np.sum(rd[:, None, :] * qvec, axis=2) * inv_det
        t = np.sum(e2[None, :, :] * qvec, axis=2) * inv_det

        valid = det_ok
        valid &= u >= -eps
        valid &= v >= -eps
        valid &= (u + v) <= (1.0 + eps)
        valid &= t > eps
        if math.isfinite(max_dist):
            valid &= t <= max_dist

        t_valid = np.where(valid, t, np.inf)
        batch_hit_idx = np.argmin(t_valid, axis=1)
        batch_hit_t = t_valid[np.arange(end - start), batch_hit_idx]
        miss = ~np.isfinite(batch_hit_t)
        batch_hit_idx = batch_hit_idx.astype(np.int32, copy=False)
        batch_hit_idx[miss] = -1
        hit_t[start:end] = batch_hit_t
        hit_idx[start:end] = batch_hit_idx

    hit_mask = hit_idx >= 0
    hit_pos = np.full((n_rays, 3), np.nan, dtype=dtype)
    hit_normals = np.full((n_rays, 3), np.nan, dtype=dtype)
    if np.any(hit_mask):
        hit_pos[hit_mask] = ray_o[hit_mask] + ray_d[hit_mask] * hit_t[hit_mask, None]
        hit_normals[hit_mask] = norms[hit_idx[hit_mask]]

    if tags is None:
        hit_owner = None
    else:
        if tags.dtype == object:
            hit_owner = np.empty(n_rays, dtype=object)
            hit_owner[:] = None
            if np.any(hit_mask):
                hit_owner[hit_mask] = tags[hit_idx[hit_mask]]
        else:
            hit_owner = np.full(n_rays, -1, dtype=tags.dtype)
            if np.any(hit_mask):
                hit_owner[hit_mask] = tags[hit_idx[hit_mask]]

    result = {
        'origin': ray_o[0] if single else ray_o,
        'direction': ray_d[0] if single else ray_d,
        'hit': bool(hit_mask[0]) if single else hit_mask,
        'distance': (float(hit_t[0]) if hit_mask[0] else None) if single else hit_t,
        'position': hit_pos[0] if single else hit_pos,
        'normal': hit_normals[0] if single else hit_normals,
        'triangle_index': (int(hit_idx[0]) if hit_mask[0] else -1) if single else hit_idx,
    }
    if hit_owner is not None:
        result['owner_tag'] = hit_owner[0] if single else hit_owner
    return result


def _sample_cone_directions(
        direction,
        cone_angle_rad: float,
        n_rays: int,
        up=(0.0, 1.0, 0.0),
        seed: int = 42,
) -> np.ndarray:
    dtype = _trace_float_dtype(direction, up)
    if n_rays <= 0:
        raise ValueError("n_rays must be >= 1")
    fwd, right, upv = _camera_basis(direction, up, dtype)
    half_angle = max(0.0, float(cone_angle_rad))
    dirs = np.empty((n_rays, 3), dtype=dtype)
    dirs[0] = fwd
    if n_rays == 1 or half_angle <= 1e-12:
        if n_rays > 1:
            dirs[1:] = fwd
        return dirs

    rng = np.random.default_rng(seed)
    cos_max = math.cos(half_angle)
    for i in range(1, n_rays):
        u1 = float(rng.random())
        u2 = float(rng.random())
        cos_theta = 1.0 - u1 * (1.0 - cos_max)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * math.pi * u2
        dirs[i] = _normalize_vec(
            fwd * cos_theta
            + right * (math.cos(phi) * sin_theta)
            + upv * (math.sin(phi) * sin_theta)
        )
    return dirs


def trace_single_ray(
        scene,
        origin,
        direction,
        max_distance: float = math.inf,
        verts: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        owner_tags=None,
) -> dict:
    """Trace one no-bounce ray into the scene and return the nearest hit."""
    tri_verts, tri_normals, _ = _coerce_trace_geometry(scene, verts=verts, normals=normals)
    return _raycast_triangles(
        tri_verts,
        tri_normals,
        origin,
        direction,
        owner_tags=owner_tags,
        max_distance=max_distance,
    )


def trace_cone_rays(
        scene,
        origin,
        direction,
        cone_angle_rad: float = 0.02,
        n_rays: int = 7,
        up=(0.0, 1.0, 0.0),
        seed: int = 42,
        max_distance: float = math.inf,
        verts: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        owner_tags=None,
) -> dict:
    """Trace a small no-bounce ray cone and return per-ray hits plus summary stats."""
    tri_verts, tri_normals, _ = _coerce_trace_geometry(scene, verts=verts, normals=normals)
    dirs = _sample_cone_directions(direction, cone_angle_rad, n_rays, up=up, seed=seed)
    dtype = _trace_float_dtype(origin, dirs)
    origins = np.broadcast_to(np.asarray(origin, dtype=dtype), dirs.shape).copy()
    hits = _raycast_triangles(
        tri_verts,
        tri_normals,
        origins,
        dirs,
        owner_tags=owner_tags,
        max_distance=max_distance,
    )
    hit_mask = np.asarray(hits['hit'], dtype=bool)
    distances = np.asarray(hits['distance'])
    finite = distances[np.isfinite(distances)]
    hits['n_rays'] = int(n_rays)
    hits['n_hits'] = int(hit_mask.sum())
    hits['hit_ratio'] = float(hit_mask.mean())
    hits['nearest_distance'] = float(finite.min()) if len(finite) else None
    hits['farthest_distance'] = float(finite.max()) if len(finite) else None
    return hits


def trace_depth_map(
        scene,
        cam_pos,
        cam_fwd,
        cam_up=(0.0, 1.0, 0.0),
        fov_rad: float = 1.0,
        width: int = 512,
        height: int = 384,
        max_distance: float = math.inf,
        batch_size: int = 256,
        verts: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        owner_tags=None,
) -> tuple[np.ndarray, dict]:
    """Trace a no-bounce pinhole depth map from a camera pose."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    tri_verts, tri_normals, _ = _coerce_trace_geometry(scene, verts=verts, normals=normals)
    dtype = _trace_float_dtype(cam_pos, cam_fwd, cam_up)
    cam_pos_arr = np.asarray(cam_pos, dtype=dtype)
    fwd, right, upv = _camera_basis(cam_fwd, cam_up, dtype)

    aspect = float(width) / float(height)
    tan_half = math.tan(0.5 * float(fov_rad))
    xs = ((np.arange(width, dtype=dtype) + 0.5) / float(width) * 2.0 - 1.0) * aspect * tan_half
    ys = (1.0 - (np.arange(height, dtype=dtype) + 0.5) / float(height) * 2.0) * tan_half
    grid_x, grid_y = np.meshgrid(xs, ys)
    dirs = (
        fwd[None, None, :]
        + grid_x[:, :, None] * right[None, None, :]
        + grid_y[:, :, None] * upv[None, None, :]
    )
    dirs = dirs.reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(cam_pos_arr, dirs.shape).copy()

    hits = _raycast_triangles(
        tri_verts,
        tri_normals,
        origins,
        dirs,
        owner_tags=owner_tags,
        max_distance=max_distance,
        batch_size=batch_size,
    )
    hit_mask = np.asarray(hits['hit'], dtype=bool).reshape(height, width)
    distances = np.asarray(hits['distance'], dtype=dtype).reshape(height, width)
    triangle_index = np.asarray(hits['triangle_index'], dtype=np.int32).reshape(height, width)
    position = np.asarray(hits['position'], dtype=dtype).reshape(height, width, 3)
    normal = np.asarray(hits['normal'], dtype=dtype).reshape(height, width, 3)
    depth_map = distances.copy()
    depth_map[~hit_mask] = np.inf

    meta = {
        'hit_mask': hit_mask,
        'triangle_index': triangle_index,
        'position_map': position,
        'normal_map': normal,
        'camera_forward': fwd,
        'camera_right': right,
        'camera_up': upv,
    }
    if 'owner_tag' in hits:
        meta['owner_tag_map'] = np.asarray(hits['owner_tag'], dtype=object).reshape(height, width)
    return depth_map, meta


def _drop_absorptive_helper_shell(room, verts, normals, mat_props, n_room_tris):
    """For standalone body scenes, remove the dummy enclosing PolygonalRoom."""
    unified = _watertight_guitar_body_from_baffles(room)
    if unified is not None:
        return unified
    has_baffles = bool(getattr(room, 'baffles', []))
    wall_ref = float(getattr(room, 'wall_reflectivity', 1.0))
    wall_abs = float(getattr(room, 'wall_absorption', 0.0))
    if has_baffles and n_room_tris > 0 and wall_ref <= 1e-9 and wall_abs >= 0.999:
        return verts[n_room_tris:], normals[n_room_tris:], mat_props[n_room_tris:]
    return verts, normals, mat_props


def _watertight_guitar_body_from_baffles(room):
    baffles = list(getattr(room, 'baffles', []) or [])
    top = next((p for p in baffles
                if getattr(p, 'key', '') == 'str_top_plate'
                and getattr(p, 'outline_pts', None) is not None), None)
    if top is None:
        return None

    outline = np.asarray(getattr(top, 'outline_pts'), dtype=np.float64)
    if len(outline) < 8:
        return None
    body_h = float(getattr(top, 'point', (0.0, 0.0, 0.06))[2])
    soundhole = getattr(top, 'soundhole', (0.0, 0.05, 0.028))
    cx, cy, hr = float(soundhole[0]), float(soundhole[1]), float(soundhole[2])
    top_ref = float(getattr(top, 'reflectivity', 0.7))
    top_diff = float(getattr(top, 'diffusion', 0.25))
    top_abso = float(getattr(top, 'absorption', 0.15))
    unfinished_softwood = (
        min(max(top_ref * 0.86, 0.46), 0.68),
        max(top_diff, 0.62),
        max(top_abso, 0.28),
    )
    rib_softwood = (
        min(max(top_ref * 0.92, 0.50), 0.72),
        max(top_diff, 0.54),
        max(top_abso * 0.85, 0.22),
    )
    back_softwood = (
        min(max(top_ref * 0.88, 0.48), 0.70),
        max(top_diff, 0.58),
        max(top_abso, 0.25),
    )
    enamel_finish = (0.88, 0.10, 0.055)

    n_theta = 192
    n_radial = 18
    tris: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    props: list[list[float]] = []

    def _push(v0, v1, v2, hint=None, interior_mat=unfinished_softwood,
              exterior_mat=enamel_finish):
        v0 = np.asarray(v0, dtype=np.float64)
        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)
        tn = np.cross(v1 - v0, v2 - v0)
        tnm = np.linalg.norm(tn)
        if tnm < 1e-12:
            return
        tn /= tnm
        if hint is not None and np.dot(tn, hint) < 0.0:
            v1, v2 = v2, v1
            tn = -tn
        tris.append(np.array([v0, v1, v2], dtype=np.float64))
        norms.append(tn)
        props.append([*interior_mat, *exterior_mat])

    outer_top = []
    inner_top = []
    radial_rows = []
    for ri in range(n_radial + 1):
        frac = ri / float(n_radial)
        row = []
        for ai in range(n_theta):
            th = 2.0 * math.pi * ai / float(n_theta)
            outer = _outline_ray_radius(outline, cx, cy, th)
            r = hr + frac * max(0.0, outer - hr)
            p = np.array([cx + r * math.cos(th), cy + r * math.sin(th), body_h])
            row.append(p)
        radial_rows.append(row)
    inner_top = radial_rows[0]
    outer_top = radial_rows[-1]
    outer_back = [np.array([p[0], p[1], 0.0], dtype=np.float64) for p in outer_top]

    # Top annulus, with soundhole open.
    for ri in range(n_radial):
        row = radial_rows[ri]
        nxt = radial_rows[ri + 1]
        for ai in range(n_theta):
            a = row[ai]
            b = row[(ai + 1) % n_theta]
            c = nxt[(ai + 1) % n_theta]
            d = nxt[ai]
            _push(a, d, c, np.array([0.0, 0.0, 1.0]), unfinished_softwood)
            _push(a, c, b, np.array([0.0, 0.0, 1.0]), unfinished_softwood)

    # Inner face of soundboard (3 mm plate thickness) — normals point into body cavity.
    _plate_thick = 0.003
    for ri in range(n_radial):
        row = radial_rows[ri]
        nxt = radial_rows[ri + 1]
        for ai in range(n_theta):
            _a = np.array([row[ai][0],               row[ai][1],               body_h - _plate_thick], dtype=np.float64)
            _b = np.array([row[(ai+1)%n_theta][0],   row[(ai+1)%n_theta][1],   body_h - _plate_thick], dtype=np.float64)
            _c = np.array([nxt[(ai+1)%n_theta][0],   nxt[(ai+1)%n_theta][1],   body_h - _plate_thick], dtype=np.float64)
            _d = np.array([nxt[ai][0],               nxt[ai][1],               body_h - _plate_thick], dtype=np.float64)
            _push(_a, _c, _d, np.array([0.0, 0.0, -1.0]), unfinished_softwood)
            _push(_a, _b, _c, np.array([0.0, 0.0, -1.0]), unfinished_softwood)

    # Side walls using the exact same outer ring as the top/back caps.
    cen2 = np.array([float(outline[:, 0].mean()), float(outline[:, 1].mean())])
    for ai in range(n_theta):
        a0 = outer_back[ai]
        b0 = outer_back[(ai + 1) % n_theta]
        b1 = outer_top[(ai + 1) % n_theta]
        a1 = outer_top[ai]
        mid = 0.5 * (a0 + b0)
        hint2 = mid[:2] - cen2
        hint = np.array([hint2[0], hint2[1], 0.0], dtype=np.float64)
        _push(a0, b0, b1, hint, rib_softwood)
        _push(a0, b1, a1, hint, rib_softwood)

    # Back cap, closed.
    back_c = np.array([cen2[0], cen2[1], 0.0], dtype=np.float64)
    for ai in range(n_theta):
        _push(back_c, outer_back[(ai + 1) % n_theta], outer_back[ai],
              np.array([0.0, 0.0, -1.0]), back_softwood)

    return (np.asarray(tris, dtype=np.float64),
            np.asarray(norms, dtype=np.float64),
            np.asarray(props, dtype=np.float64))


def _circular_room_to_polygon(room):
    """Return a duck-typed object exposing vertices_xy from a CircularRoom."""
    n_seg = getattr(room, 'n_segments', 24)
    r     = float(room.radius)
    angles = [2 * math.pi * i / n_seg for i in range(n_seg)]
    vxy = [(r * math.cos(a), r * math.sin(a)) for a in angles]

    class _FlatRoom:
        vertices_xy        = vxy
        height             = room.height
        wall_reflectivity  = room.wall_reflectivity
        wall_diffusion     = room.wall_diffusion
        wall_absorption    = room.wall_absorption
        floor_reflectivity = room.floor_reflectivity
        floor_diffusion    = room.floor_diffusion
        floor_absorption   = room.floor_absorption
        roof_reflectivity  = room.roof_reflectivity
        roof_diffusion     = room.roof_diffusion
        roof_absorption    = room.roof_absorption
        baffles            = getattr(room, 'baffles', [])

    return _FlatRoom()


def _triangulate_mesh_room(room) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts_list, normals_list, mat_props_list = [], [], []
    verts_src = np.asarray(room.vertices, dtype=np.float64)  # (M, 3)
    for tri in room.triangles:
        idx = tri.vertex_indices
        v0, v1, v2 = verts_src[idx[0]], verts_src[idx[1]], verts_src[idx[2]]
        e1 = v1 - v0;  e2 = v2 - v0
        tn = np.cross(e1, e2)
        tnm = np.linalg.norm(tn)
        if tnm < 1e-12:
            continue
        tn /= tnm
        verts_list.append([v0, v1, v2])
        normals_list.append(tn)
        mat_key = getattr(tri, 'material_key', None)
        mat     = room.materials.get(mat_key) if mat_key else None
        ref  = getattr(mat, 'reflectivity', 0.7)
        diff = getattr(mat, 'diffusion',    0.35)
        abso = getattr(mat, 'absorption',   0.15)
        mat_props_list.append([ref, diff, abso])
    return (np.array(verts_list,   dtype=np.float64),
            np.array(normals_list,  dtype=np.float64),
            np.array(mat_props_list, dtype=np.float64))


# ---------------------------------------------------------------------------
# Source emission spectrum model
# ---------------------------------------------------------------------------
#
# Analogous to the 4-parameter surface material system, every acoustic source
# can carry an EmissionSpectrum made of one or more EmissionBands.
#
# GPU pipeline integration
# ------------------------
# Source tuples fed to the forward ray shader have the form:
#     (pos_f32, dir_f32, n_rays, spec_f32)
# where spec_f32 is a 4-element float32 array of per-band energy fractions
# mapping to hardware bands B0–B3 (55-220 / 220-440 / 440-880 / 880+ Hz).
#
# EmissionSpectrum.band_weights(band_edges_hz) returns a normalised 4-element
# array suitable for spec_f32.  EmissionSpectrum.allocate_rays(n_total) splits
# n_total rays across the four hardware bands in proportion to their weights.
#
# Film interaction
# ----------------
# Recorded signal per band:
#     recorded[b] = emission_weight[b] * film_filter[b]
# EmissionSpectrum drives the numerator; FilmParams.band_filters drives the
# denominator.  Together they model source chromaticity × detector sensitivity.

# Hardware band edges shared with the GPU forward shader.
_GPU_BAND_EDGES_HZ = (55.0, 220.0, 440.0, 880.0, float('inf'))


@dataclass
class EmissionBand:
    """One Gaussian lobe in frequency space — 4 parameters, like a surface material.

    center_hz    : peak frequency of the emission lobe
    bandwidth_hz : standard deviation of the Gaussian (determines spread)
    amplitude    : peak weight of this lobe relative to others
    phase_sigma  : std-dev of random phase offsets drawn per ray (radians);
                   0 = all rays coherent, π = fully incoherent
    """
    center_hz:    float
    bandwidth_hz: float
    amplitude:    float = 1.0
    phase_sigma:  float = 0.0

    def power_at(self, freq_hz: float) -> float:
        """Gaussian power spectral density at freq_hz."""
        x = (freq_hz - self.center_hz) / max(self.bandwidth_hz, 1e-6)
        return self.amplitude * math.exp(-0.5 * x * x)


@dataclass
class EmissionSpectrum:
    """Composite emission spectrum for an acoustic source.

    A collection of EmissionBands whose combined power spectral density is
    evaluated by summing each band's Gaussian lobe.

    ray_allocation : 'proportional' — split n_rays proportionally to band power
                     'uniform'      — equal rays across non-zero bands
    """
    bands:          list = field(default_factory=list)   # list[EmissionBand]
    ray_allocation: str  = 'proportional'

    # ------------------------------------------------------------------
    # Core spectral queries
    # ------------------------------------------------------------------

    def power_at(self, freq_hz: float) -> float:
        """Total power spectral density at a given frequency."""
        return sum(b.power_at(freq_hz) for b in self.bands)

    def band_weights(
            self,
            band_edges_hz: tuple = _GPU_BAND_EDGES_HZ,
            n_sample: int = 32,
    ) -> np.ndarray:
        """Integrate power into hardware bands via Gaussian quadrature (mid-point rule).

        Returns a normalised float32 array of length len(band_edges_hz)-1 whose
        elements sum to 1.0.  This array maps directly onto the spec_f32 slot of
        a GPU source tuple and onto uSrcSpectrum in the forward ray shader.

        band_edges_hz : monotone increasing sequence of n+1 bin edges (Hz).
                        Defaults to the four GPU hardware band boundaries.
        n_sample      : number of log-spaced sample points per band for
                        numerical integration.
        """
        n_bands = len(band_edges_hz) - 1
        weights = np.zeros(n_bands, dtype=np.float64)

        for i in range(n_bands):
            lo = band_edges_hz[i]
            hi = min(band_edges_hz[i + 1], 20000.0)
            if hi <= lo:
                continue
            # log-spaced sample points inside the band
            f_samples = np.geomspace(lo, hi, n_sample)
            psd = np.array([self.power_at(f) for f in f_samples])
            # trapezoidal integration in log-frequency
            log_f = np.log(f_samples)
            weights[i] = float(np.trapz(psd, log_f))

        total = weights.sum()
        if total > 0.0:
            weights /= total
        else:
            weights[:] = 1.0 / n_bands   # flat fallback for silent spectrum

        return weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Ray distribution
    # ------------------------------------------------------------------

    def allocate_rays(self, n_total: int) -> np.ndarray:
        """Distribute n_total rays across hardware bands.

        Returns an int32 array of length 4 (for the default GPU bands) that
        sums to n_total.  Used when building (pos, dir, n_rays, spec_f32)
        tuples — the caller may split one logical source into four per-band
        sub-sources or use the fractions directly via spec_f32.

        ray_allocation=='proportional' : multinomial draw from band_weights.
        ray_allocation=='uniform'      : equal share among bands with weight>0.
        """
        w = self.band_weights()
        n = len(w)

        if self.ray_allocation == 'uniform':
            active = (w > 0).sum()
            base   = n_total // max(active, 1)
            counts = np.where(w > 0, base, 0).astype(np.int32)
        else:
            counts = np.zeros(n, dtype=np.int32)
            for i in range(n - 1):
                counts[i] = int(round(float(w[i]) * n_total))

        # Assign remainder to the dominant band.
        counts[-1] = n_total - int(counts[:-1].sum())
        counts[-1] = max(0, counts[-1])
        return counts

    def sample_phases(self, n_rays: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw n_rays initial phase offsets (radians) from the emission model.

        Phase spread is the amplitude-weighted mean phase_sigma across bands.
        Returns float32 array of shape (n_rays,) in [−π, π].
        """
        if rng is None:
            rng = np.random.default_rng()

        total_amp = sum(b.amplitude for b in self.bands) or 1.0
        sigma = sum(b.phase_sigma * b.amplitude for b in self.bands) / total_amp

        if sigma <= 0.0:
            return np.zeros(n_rays, dtype=np.float32)

        phases = rng.normal(0.0, sigma, size=n_rays).astype(np.float32)
        phases = (phases + math.pi) % (2.0 * math.pi) - math.pi
        return phases

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def white(cls) -> 'EmissionSpectrum':
        """Flat power across all GPU hardware bands."""
        edges = _GPU_BAND_EDGES_HZ
        bands = []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = min(edges[i + 1], 20000.0)
            center = math.sqrt(lo * hi)           # geometric mean
            bw     = (hi - lo) * 0.5
            bands.append(EmissionBand(center_hz=center, bandwidth_hz=bw, amplitude=1.0))
        return cls(bands=bands)

    @classmethod
    def tonal(cls, center_hz: float, q_factor: float = 8.0,
              phase_sigma: float = 0.1) -> 'EmissionSpectrum':
        """Narrow Gaussian centred on center_hz with quality factor Q."""
        bw = center_hz / max(q_factor, 0.1)
        return cls(bands=[EmissionBand(
            center_hz=center_hz, bandwidth_hz=bw,
            amplitude=1.0, phase_sigma=phase_sigma,
        )])

    @classmethod
    def harmonic_series(
            cls,
            fundamental_hz: float,
            n_harmonics:    int   = 6,
            rolloff:        float = 0.7,
            q_factor:       float = 12.0,
            phase_sigma:    float = 0.3,
    ) -> 'EmissionSpectrum':
        """Stack of n_harmonics tonal bands at integer multiples of fundamental_hz.

        amplitude rolls off as rolloff**k for the k-th harmonic.
        """
        bands = []
        for k in range(1, n_harmonics + 1):
            f  = fundamental_hz * k
            bw = f / max(q_factor, 0.1)
            bands.append(EmissionBand(
                center_hz=f, bandwidth_hz=bw,
                amplitude=rolloff ** (k - 1),
                phase_sigma=phase_sigma,
            ))
        return cls(bands=bands)

    @classmethod
    def noise_band(cls, center_hz: float, octave_width: float = 1.0,
                   phase_sigma: float = math.pi) -> 'EmissionSpectrum':
        """Broad noise emission centred at center_hz spanning ±octave_width octaves.

        phase_sigma = π → fully incoherent (white noise character).
        """
        bw = center_hz * (2.0 ** (octave_width * 0.5) - 2.0 ** (-octave_width * 0.5))
        return cls(bands=[EmissionBand(
            center_hz=center_hz, bandwidth_hz=bw,
            amplitude=1.0, phase_sigma=phase_sigma,
        )])


# Named emission presets — drop-in replacements for the spec_f32 slot.
EMISSION_PRESETS: dict[str, EmissionSpectrum] = {
    'white':           EmissionSpectrum.white(),
    'low_fundamental': EmissionSpectrum.harmonic_series(110.0,  n_harmonics=4, rolloff=0.6),
    'mid_fundamental': EmissionSpectrum.harmonic_series(220.0,  n_harmonics=6, rolloff=0.7),
    'high_fundamental':EmissionSpectrum.harmonic_series(440.0,  n_harmonics=8, rolloff=0.75),
    'bass_noise':      EmissionSpectrum.noise_band(80.0,   octave_width=2.0),
    'mid_noise':       EmissionSpectrum.noise_band(440.0,  octave_width=2.0),
    'treble_noise':    EmissionSpectrum.noise_band(3500.0, octave_width=2.0),
    'open_a_string':   EmissionSpectrum.harmonic_series(110.0,  n_harmonics=12, rolloff=0.65, q_factor=20.0),
    'open_e_string':   EmissionSpectrum.harmonic_series(82.4,   n_harmonics=12, rolloff=0.60, q_factor=20.0),
    'open_g_string':   EmissionSpectrum.harmonic_series(196.0,  n_harmonics=10, rolloff=0.70, q_factor=18.0),
    'open_b_string':   EmissionSpectrum.harmonic_series(246.9,  n_harmonics=10, rolloff=0.72, q_factor=18.0),
    'incoherent_full': EmissionSpectrum(
        bands=[EmissionBand(c, c * 0.5, 1.0, math.pi)
               for c in (110.0, 330.0, 660.0, 1320.0)],
        ray_allocation='uniform',
    ),
}


# ---------------------------------------------------------------------------
# Frequency bands
# ---------------------------------------------------------------------------

def build_frequency_bands(scene, n_bands: int = 12) -> np.ndarray:
    """Build a log-spaced frequency array for the given scene.

    Prefers to bracket the scene's source band profiles if available;
    otherwise uses 80 Hz – 8 kHz.
    """
    explicit: list[float] = []
    for src in getattr(scene, 'sources', []):
        for band in getattr(src, 'band_profiles', []):
            hz = getattr(band, 'center_hz', None)
            if hz is not None and hz > 0.0:
                explicit.append(float(hz))

    if len(explicit) >= 2:
        f_min = max(20.0, min(explicit) * 0.5)
        f_max = min(20000.0, max(explicit) * 2.0)
    else:
        f_min, f_max = 80.0, 8000.0

    return np.geomspace(f_min, f_max, n_bands, dtype=np.float64)


# ---------------------------------------------------------------------------
# Complex reflectance model
# ---------------------------------------------------------------------------

def build_complex_reflectances(
        mat_props: np.ndarray,
        freq_hz:   np.ndarray,
        f_ref:     float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frequency-dependent complex reflectances.

    Magnitude model:  r(f) = r0 * max(0, 1 - absorption * sqrt(f / f_ref))
      — reflectivity drops as sqrt(f) at high frequencies (viscous losses).

    Phase model (Kramers-Kronig first-order approximation):
      The causal response of a locally reacting wall has a phase lead that
      scales as +arctan(absorption * ln(f / f_ref) / π).  This gives a small
      imaginary part that preserves causality and creates phase dispersion
      between bands — the spectral equivalent of room mode structure.

    Returns
    -------
    refl_re, refl_im : (N_tri, N_bands) float64
    """
    n_tri   = len(mat_props)
    n_bands = len(freq_hz)
    f_ratio = freq_hz / f_ref                  # (N_bands,)
    log_f   = np.log(np.maximum(f_ratio, 1e-9))

    refl_re = np.empty((n_tri, n_bands), dtype=np.float64)
    refl_im = np.empty((n_tri, n_bands), dtype=np.float64)

    for i in range(n_tri):
        r0   = float(mat_props[i, 0])
        abso = float(mat_props[i, 2])

        # Frequency-dependent magnitude.
        r_f = r0 * np.maximum(0.0, 1.0 - abso * np.sqrt(np.maximum(f_ratio, 0.0)))

        # KK-consistent phase: small imaginary part.
        phi_f = np.arctan(abso * log_f / math.pi)

        refl_re[i, :] = r_f * np.cos(phi_f)
        refl_im[i, :] = r_f * np.sin(phi_f)

    return refl_re, refl_im


# ---------------------------------------------------------------------------
# Phase 2 unified-MatBuf adapter
# ---------------------------------------------------------------------------
#
# The C++ ``RayTracer`` and the GLSL compute path both index a single flat
# ``mat_buf`` SSBO of shape ``(N_mat * MAX_SPECTRAL_BANDS, 12) float32`` —
# row ``mat_idx[tri] * MAX_SPECTRAL_BANDS + band`` carries the band record:
#
#     0..3 : center_hz, bandwidth_hz, reflectance_mag, transmittance
#     4..7 : diffuse_frac, emission, reemission, ior_real
#     8..11: ior_imag, _pad, _pad, _pad
#
# The tracer derives the per-band reflectance phase from the complex IOR
# (``r_F = (1 - n) / (1 + n)``) and multiplies it by ``reflectance_mag``.
# To reproduce a target complex reflectance ``r = re + j·im`` exactly, set
# ``n = (1 - r) / (1 + r)`` and ``reflectance_mag = |r|``: the tracer's
# ``r_F`` then equals ``r`` and the recombined ``r_used = |r| · exp(i·arg r)``
# matches the input.
#
# This helper bakes that transform plus per-tri byte-keyed deduplication so
# legacy callers that hand us per-tri spectral arrays can hand the new ctor
# the ``(mat_idx, mat_buf, mat_n_mats)`` triplet it expects.

from material_db import MAX_SPECTRAL_BANDS as _MAX_SPECTRAL_BANDS

_MAT_BUF_BAND_FLOATS = 12
_MAT_BUF_MAT_FLOATS  = _MAT_BUF_BAND_FLOATS * _MAX_SPECTRAL_BANDS


def per_tri_spectral_to_mat_buf(
        refl_re:            np.ndarray,
        refl_im:            np.ndarray,
        diffusion_bands:    np.ndarray,
        freq_hz:            np.ndarray,
        *,
        emission_bands:     Optional[np.ndarray] = None,
        reemission_bands:   Optional[np.ndarray] = None,
        transmittance_bands: Optional[np.ndarray] = None,
        bandwidth_hz:       Optional[np.ndarray] = None,
        ior_real_bands:     Optional[np.ndarray] = None,
        ior_imag_bands:     Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack per-triangle spectral surface data into the unified MatBuf format.

    Parameters
    ----------
    refl_re, refl_im     : (N_tri, n_bands) float — complex surface reflectance
    diffusion_bands      : (N_tri, n_bands) float OR (N_tri,) float
    freq_hz              : (n_bands,) float
    emission_bands,
    reemission_bands,
    transmittance_bands  : optional (N_tri, n_bands) float, default zeros
    bandwidth_hz         : optional (n_bands,) float — defaults to adjacent gap
    ior_real_bands,
    ior_imag_bands       : optional (N_tri, n_bands) float overrides; when not
                           supplied the IOR is derived from the target complex
                           reflectance via ``n = (1 - r) / (1 + r)``.

    Returns
    -------
    mat_idx     : (N_tri,) int32
    mat_buf     : (N_unique * MAX_SPECTRAL_BANDS, 12) float32
    mat_n_mats  : int — number of unique materials after byte-deduplication
    """
    refl_re = np.asarray(refl_re, np.float64)
    refl_im = np.asarray(refl_im, np.float64)
    if refl_re.shape != refl_im.shape:
        raise ValueError("refl_re / refl_im must have identical shape")
    if refl_re.ndim != 2:
        raise ValueError("refl_re must be (N_tri, n_bands)")
    n_tri, n_bands = refl_re.shape
    if n_bands > _MAX_SPECTRAL_BANDS:
        raise ValueError(
            f"n_bands={n_bands} exceeds MAX_SPECTRAL_BANDS={_MAX_SPECTRAL_BANDS}")
    freq_hz = np.asarray(freq_hz, np.float64).ravel()
    if freq_hz.size != n_bands:
        raise ValueError("freq_hz length must equal n_bands")

    # ── Broadcast / default the optional per-band arrays ────────────────────
    diff = np.asarray(diffusion_bands, np.float64)
    if diff.ndim == 1:
        diff = np.tile(diff[:, None], (1, n_bands))
    if diff.shape != (n_tri, n_bands):
        raise ValueError("diffusion_bands shape mismatch")

    def _opt(arr, name):
        if arr is None:
            return np.zeros((n_tri, n_bands), np.float64)
        a = np.asarray(arr, np.float64)
        if a.shape != (n_tri, n_bands):
            raise ValueError(f"{name} shape mismatch (need {(n_tri, n_bands)})")
        return a

    emis    = _opt(emission_bands,     "emission_bands")
    reemis  = _opt(reemission_bands,   "reemission_bands")
    transm  = _opt(transmittance_bands, "transmittance_bands")

    # Bandwidth: nearest-neighbour spacing, with edges mirrored.
    if bandwidth_hz is None:
        if n_bands == 1:
            bw = np.array([max(1.0, freq_hz[0] * 0.5)], np.float64)
        else:
            diffs = np.diff(freq_hz)
            bw = np.empty(n_bands, np.float64)
            bw[1:-1] = 0.5 * (np.abs(diffs[:-1]) + np.abs(diffs[1:]))
            bw[0]    = abs(diffs[0])
            bw[-1]   = abs(diffs[-1])
    else:
        bw = np.asarray(bandwidth_hz, np.float64).ravel()
        if bw.size != n_bands:
            raise ValueError("bandwidth_hz length must equal n_bands")

    # ── Reflectance magnitude + Fresnel-consistent IOR per band ─────────────
    mag = np.hypot(refl_re, refl_im)
    if ior_real_bands is None or ior_imag_bands is None:
        r_c = refl_re + 1j * refl_im
        denom = 1.0 + r_c
        denom = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
        n_c = (1.0 - r_c) / denom
        ior_re = np.real(n_c) if ior_real_bands is None else \
                 np.asarray(ior_real_bands, np.float64)
        ior_im = np.imag(n_c) if ior_imag_bands is None else \
                 np.asarray(ior_imag_bands, np.float64)
    else:
        ior_re = np.asarray(ior_real_bands, np.float64)
        ior_im = np.asarray(ior_imag_bands, np.float64)
    if ior_re.shape != (n_tri, n_bands) or ior_im.shape != (n_tri, n_bands):
        raise ValueError("ior_*_bands shape mismatch")

    # ── Build the per-tri (MAX_BANDS, 12) record ────────────────────────────
    rec = np.zeros((n_tri, _MAX_SPECTRAL_BANDS, 12), np.float32)
    rec[:, :n_bands, 0]  = freq_hz[None, :].astype(np.float32)
    rec[:, :n_bands, 1]  = bw[None, :].astype(np.float32)
    rec[:, :n_bands, 2]  = mag.astype(np.float32)
    rec[:, :n_bands, 3]  = transm.astype(np.float32)
    rec[:, :n_bands, 4]  = diff.astype(np.float32)
    rec[:, :n_bands, 5]  = emis.astype(np.float32)
    rec[:, :n_bands, 6]  = reemis.astype(np.float32)
    rec[:, :n_bands, 7]  = ior_re.astype(np.float32)
    rec[:, :n_bands, 8]  = ior_im.astype(np.float32)

    # ── Byte-keyed dedup ────────────────────────────────────────────────────
    flat = np.ascontiguousarray(rec.reshape(n_tri, -1))   # (N_tri, MAX*12) f32
    keys = flat.view(np.uint8).reshape(n_tri, -1)
    seen: dict[bytes, int] = {}
    mat_idx = np.empty(n_tri, np.int32)
    unique_rows: list[np.ndarray] = []
    for i in range(n_tri):
        kb = keys[i].tobytes()
        idx = seen.get(kb)
        if idx is None:
            idx = len(unique_rows)
            seen[kb] = idx
            unique_rows.append(flat[i])
        mat_idx[i] = idx
    if unique_rows:
        mat_buf = np.ascontiguousarray(
            np.stack(unique_rows, axis=0).reshape(-1, 12), np.float32)
    else:
        mat_buf = np.zeros((0, 12), np.float32)
    return mat_idx, mat_buf, len(unique_rows)


def _per_tri_mat_buf_from_legacy(
        refl_re, refl_im, diffusion_bands, freq_hz,
        *, emission_bands=None, reemission_bands=None):
    """Convenience wrapper for the trace_*_scene call sites in this module."""
    return per_tri_spectral_to_mat_buf(
        refl_re, refl_im, diffusion_bands, freq_hz,
        emission_bands=emission_bands,
        reemission_bands=reemission_bands,
    )


def assign_wall_band_materials(
        verts:    np.ndarray,   # (n_tri, 3, 3) float64 triangle vertices
        wall_bands: list,       # list[WallBand] (sorted by top, from station.yaml)
        wall_z_min: float = 0.0,
        wall_z_max: float = 1.0,
        fallback_material: Optional[object] = None,
) -> list:
    """Assign a SpectralMaterial to each triangle based on its Z-centroid height.

    Maps triangle centroids' Z coordinate to a normalised wall height, then
    looks up the appropriate WallBand material.  Non-wall triangles (those
    outside [wall_z_min, wall_z_max]) receive ``fallback_material`` or the
    first/last band material.

    Parameters
    ----------
    verts           : (n_tri, 3, 3) float64 — triangle vertices
    wall_bands      : list[WallBand]         — sorted by WallBand.top
    wall_z_min      : float — Z of the bottom of the first wall band
    wall_z_max      : float — Z of the top of the last wall band
    fallback_material : Material | None      — used for out-of-range triangles

    Returns
    -------
    materials : list[Material], length n_tri
    """
    if not _HAS_SPECTRAL_MAT or not wall_bands:
        return []
    n_tri = len(verts)
    # Triangle Z-centroids
    centroids_z = verts.reshape(n_tri, 3, 3)[:, :, 2].mean(axis=1)   # (n_tri,)
    z_range = max(wall_z_max - wall_z_min, 1e-6)
    materials = []
    for z in centroids_z:
        t = float(z - wall_z_min) / z_range
        mat = _wall_band_at(wall_bands, t)
        if mat is None:
            mat = fallback_material or wall_bands[-1].material
        materials.append(mat)
    return materials


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Approximate CIE 1931 spectral locus colours for acoustic/spectral bands.
# Maps a log-frequency range to perceptual hue (red→violet over 20–20k Hz),
# then tone-maps magnitude with exposure/gamma.
def spectral_bands_to_rgb(
        flux:    np.ndarray,   # (n_tri, n_bands) float32 irradiance
        freq_hz: np.ndarray,   # (n_bands,) float64 centre frequencies
        exposure: float = 3.0,
        gamma:    float = 0.45,
) -> np.ndarray:
    """Convert per-triangle per-band irradiance into a linear RGB array.

    Each frequency band is mapped to a perceptual colour using a simple
    spectral→RGB approximation (visible-light analogy for audio bands):
      20 Hz → 20 kHz spans red → violet over log frequency.

    Returns
    -------
    rgb : float32 (n_tri, 3) in [0, 1], gamma-corrected
    """
    n_tri, n_bands = flux.shape
    # Log-frequency position in [0, 1].
    f0, f1 = 20.0, 20000.0
    t = np.clip((np.log10(freq_hz) - np.log10(f0)) /
                (np.log10(f1) - np.log10(f0)), 0.0, 1.0).astype(np.float32)  # (n_bands,)

    # Simple spectral locus hue wheel (red=0, green=0.33, blue=0.67, violet=1.0).
    # Uses a piecewise sinusoidal approximation.
    hue = t  # 0 → 1 from red to violet
    r_band = np.clip(np.cos(np.pi * (hue - 0.0)) * 1.5, 0.0, 1.0)
    g_band = np.clip(np.sin(np.pi * hue) * 1.2, 0.0, 1.0)
    b_band = np.clip(np.cos(np.pi * (hue - 1.0)) * 1.5, 0.0, 1.0)
    # shape: (n_bands,)

    # Weighted sum of spectral colour by irradiance.
    w = flux  # (n_tri, n_bands)
    rgb = np.stack([
        (w * r_band).sum(axis=1),
        (w * g_band).sum(axis=1),
        (w * b_band).sum(axis=1),
    ], axis=1).astype(np.float32)  # (n_tri, 3)

    # Exposure + gamma tone-map.
    rgb = rgb * float(exposure)
    peak = rgb.max()
    if peak > 1e-9:
        rgb /= peak
    rgb = np.clip(rgb ** float(gamma), 0.0, 1.0)
    return rgb


# ---------------------------------------------------------------------------
# Global scene field integration  (sensor-independent, GL shader seed data)
# ---------------------------------------------------------------------------

@dataclass
class SceneFieldIntegration:
    """Area-weighted spectral integrals of the ray-traced light/surface field.

    These values are independent of sensor or camera placement; they describe
    the total energy state of the scene and are intended as seed data for
    OpenGL shaders (upload via glUniform1fv / glUniform3fv).

    Attributes
    ----------
    n_bands          : int
    freq_hz          : (n_bands,)  float64 — band centre frequencies
    surface_power    : (n_bands,)  float32 — Σ(flux_i × area_i) per band
    surface_direct   : (n_bands,)  float32 — direct-illumination component
    surface_indirect : (n_bands,)  float32 — reflected/bounced component
    transport_power  : (n_bands,)  float32 — total in-flight amplitude per band
    total_power      : (n_bands,)  float32 — surface_power + transport_power
    rgb              : (3,)        float32 — spectral-weighted scene tint
    peak_band        : int         — index of the dominant band in surface_power
    """
    n_bands:          int
    freq_hz:          np.ndarray   # (n_bands,) float64
    surface_power:    np.ndarray   # (n_bands,) float32
    surface_direct:   np.ndarray   # (n_bands,) float32
    surface_indirect: np.ndarray   # (n_bands,) float32
    transport_power:  np.ndarray   # (n_bands,) float32
    total_power:      np.ndarray   # (n_bands,) float32
    rgb:              np.ndarray   # (3,)       float32
    peak_band:        int

    def as_uniform_vec(self) -> np.ndarray:
        """Flat float32 array for a single contiguous GL uniform upload.

        Layout (total length = 4 * n_bands + 4):
            surface_power    [0          : n_bands  ]
            surface_direct   [n_bands    : 2*n_bands]
            surface_indirect [2*n_bands  : 3*n_bands]
            transport_power  [3*n_bands  : 4*n_bands]
            rgb              [4*n_bands  : 4*n_bands+3]
            peak_band        [4*n_bands+3]            (as float)
        """
        return np.concatenate([
            self.surface_power,
            self.surface_direct,
            self.surface_indirect,
            self.transport_power,
            self.rgb,
            np.array([float(self.peak_band)], dtype=np.float32),
        ])


def integrate_scene_fields(
        segs:             np.ndarray,   # (N_seg, 12) float32
        surface_flux:     np.ndarray,   # (n_tri, n_bands) float32
        surface_direct:   np.ndarray,   # (n_tri, n_bands) float32
        surface_indirect: np.ndarray,   # (n_tri, n_bands) float32
        tri_verts:        np.ndarray,   # (n_tri, 3, 3) or (n_tri, 9) float32/64
        freq_hz:          np.ndarray,   # (n_bands,) float64
) -> SceneFieldIntegration:
    """Build a sensor-independent SceneFieldIntegration from ray trace outputs.

    Computes area-weighted spectral power integrals across *all* surfaces
    (not just the display subset) and accumulates in-flight ray amplitude per
    band from the segment buffer.  The result is independent of camera/sensor
    placement and is suitable for seeding OpenGL lighting shaders.

    Parameters
    ----------
    segs             : full segment buffer from trace_cavity_scene
    surface_flux     : (n_tri, n_bands) total per-triangle irradiance
    surface_direct   : (n_tri, n_bands) direct-illumination irradiance
    surface_indirect : (n_tri, n_bands) reflected irradiance
    tri_verts        : (n_tri, 3, 3) or (n_tri, 9) triangle vertex coords
    freq_hz          : (n_bands,) band centre frequencies
    """
    n_tri, n_bands = surface_flux.shape

    # Triangle areas via cross-product.
    verts3 = tri_verts.reshape(n_tri, 3, 3).astype(np.float64)
    e1     = verts3[:, 1] - verts3[:, 0]
    e2     = verts3[:, 2] - verts3[:, 0]
    areas  = (0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)).astype(np.float32)  # (n_tri,)

    # Area-weighted integrals across all surface triangles.
    sp = (surface_flux     * areas[:, None]).sum(axis=0).astype(np.float32)
    sd = (surface_direct   * areas[:, None]).sum(axis=0).astype(np.float32)
    si = (surface_indirect * areas[:, None]).sum(axis=0).astype(np.float32)

    # In-flight energy: accumulate segment amplitudes into per-band bins.
    # Segment column layout: x0,y0,z0, x1,y1,z1, src_id, bounce, band, amp, phase, path_len
    tp = np.zeros(n_bands, dtype=np.float32)
    if len(segs) > 0:
        band_idx = np.clip(segs[:, 8].astype(np.int32), 0, n_bands - 1)
        amps     = np.abs(segs[:, 9]).astype(np.float32)
        np.add.at(tp, band_idx, amps)

    total = sp + tp

    # Spectral-weighted RGB tint (same hue mapping as spectral_bands_to_rgb).
    f0, f1 = 20.0, 20000.0
    t  = np.clip((np.log10(freq_hz) - np.log10(f0)) /
                 (np.log10(f1) - np.log10(f0)), 0.0, 1.0).astype(np.float32)
    r_b = np.clip(np.cos(np.pi * t) * 1.5,         0.0, 1.0)
    g_b = np.clip(np.sin(np.pi * t) * 1.2,         0.0, 1.0)
    b_b = np.clip(np.cos(np.pi * (t - 1.0)) * 1.5, 0.0, 1.0)

    sp_n  = sp / (sp.max() + 1e-12)
    rgb   = np.array([(sp_n * r_b).sum(),
                      (sp_n * g_b).sum(),
                      (sp_n * b_b).sum()], dtype=np.float32)
    peak_rgb = rgb.max()
    if peak_rgb > 1e-9:
        rgb /= peak_rgb

    peak_band = int(sp.argmax()) if sp.max() > 1e-12 else 0

    return SceneFieldIntegration(
        n_bands          = n_bands,
        freq_hz          = freq_hz,
        surface_power    = sp,
        surface_direct   = sd,
        surface_indirect = si,
        transport_power  = tp,
        total_power      = total,
        rgb              = rgb,
        peak_band        = peak_band,
    )


def trace_cavity_scene(
        scene,
        n_rays:        int   = 256,
        max_bounces:   int   = 8,
        n_bands:       int   = 12,
        min_amplitude: float = 0.005,
        speed_m_s:     float = 343.0,
        seed:          int   = 42,
        out_cap:       int   = -1,
) -> tuple[np.ndarray, dict]:
    """Trace a CavityScene and return a float32 segment buffer.

    Parameters
    ----------
    scene         : CavityScene from cavity_engine
    n_rays        : Rays per source
    max_bounces   : Maximum reflections per ray
    n_bands       : Number of frequency bands
    min_amplitude : Stop tracing when |A|_max < this
    speed_m_s     : Speed of sound (m/s)
    seed          : RNG seed for Monte Carlo diffuse scatter
    out_cap       : Maximum segments to allocate (-1 = auto)

    Returns
    -------
    segs : (N_seg, 12) float32
        Segment buffer sorted by path_length, ready for RayAccumulatorWidget.
    meta : dict
        n_sources, n_bands, freq_hz, max_path_length, bbox_min, bbox_max
    """
    if not _HAS_C_TRACER:
        raise RuntimeError(
            "_spectral_kernels.RayTracer not available — rebuild the C extension "
            "with `cmake -S csrc -B csrc_build && cmake --build csrc_build --config Release`"
        )

    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    if room is None:
        raise TypeError(f"Scene {type(scene).__name__} has neither 'room' nor 'geometry'")

    # Triangulate room geometry.
    n_room_tris = 0
    if hasattr(room, 'vertices_xy'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(room)
    elif hasattr(room, 'radius'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(
            _circular_room_to_polygon(room))
    elif hasattr(room, 'triangles'):
        verts, normals, mat_props = _triangulate_mesh_room(room)
    else:
        raise TypeError(f"Unsupported room type: {type(room).__name__}")

    verts, normals, mat_props = _drop_absorptive_helper_shell(
        room, verts, normals, mat_props, n_room_tris)

    n_tri = len(verts)
    if n_tri == 0:
        raise ValueError("Room triangulated to zero triangles")

    # Frequency bands.
    freq_hz = build_frequency_bands(scene, n_bands)

    # Complex reflectances.
    # Use per-triangle SpectralMaterial objects when the room or scene
    # carries a 'spectral_materials' list (one Material per triangle).
    # This replaces the scalar-reflectance fallback with physically-based
    # per-band Gaussian curves that support emission and re-emission.
    _spec_mats = getattr(scene, 'spectral_materials', None) \
              or getattr(room, 'spectral_materials', None)
    if _HAS_SPECTRAL_MAT and _spec_mats is not None and len(_spec_mats) == n_tri:
        refl_re, refl_im, diffusion_bands, emission_bands, reemission_bands = \
            _materials_to_tracer_mat_props(_spec_mats, np.asarray(freq_hz, np.float64))
    else:
        refl_re, refl_im = build_complex_reflectances(mat_props, freq_hz)
        diffusion_bands  = np.tile(mat_props[:, 1:2], (1, n_bands))
        emission_bands   = np.zeros((n_tri, n_bands), dtype=np.float64)
        reemission_bands = np.zeros((n_tri, n_bands), dtype=np.float64)

    # Atmospheric absorption: ~0.01 Np/m at 1 kHz, f^1.5 scaling.
    atmo_abs = 0.01 * (freq_hz / 1000.0) ** 1.5

    # Source positions and directions.
    sources = getattr(scene, 'sources', [])
    if sources:
        src_pos  = np.array([s.position          for s in sources], dtype=np.float64)
        src_dir  = np.array([s.direction          for s in sources], dtype=np.float64)
        src_dexp = np.array([s.directivity_power  for s in sources], dtype=np.float64)
    else:
        # Default: one omni-directional source at the room centroid.
        all_v   = verts.reshape(-1, 3)
        centroid = (all_v.min(axis=0) + all_v.max(axis=0)) * 0.5
        src_pos  = (centroid + np.array([0.1, 0.2, 0.0]))[None, :]
        src_dir  = np.array([[0.0, 0.0, 1.0]])
        src_dexp = np.array([0.0])   # omni

    n_sources = len(src_pos)

    # Ensure direction vectors are normalised.
    norms = np.linalg.norm(src_dir, axis=1, keepdims=True)
    src_dir = src_dir / np.where(norms > 1e-9, norms, 1.0)

    # Segment capacity: fixed vis budget, never scales with n_rays.
    if out_cap <= 0:
        out_cap = 2_000_000

    # Build tracer and run.
    # Pass diffusion as the dominant per-band diffuse fraction (mean across bands
    # for tracers that only accept a 1-D diffusion array; full per-band array
    # is stored in meta for consumers that can use it).
    mat_idx, mat_buf, mat_n_mats = per_tri_spectral_to_mat_buf(
        refl_re, refl_im, diffusion_bands, freq_hz,
        emission_bands=emission_bands,
        reemission_bands=reemission_bands,
    )
    tracer = _CRayTracer(
        n_tri      = n_tri,
        verts      = verts.reshape(n_tri, 9),         # (n_tri, 9) ≡ (n_tri, 3, 3)
        normals    = normals,
        mat_idx    = mat_idx,
        mat_buf    = mat_buf,
        mat_n_mats = mat_n_mats,
        freq_hz    = freq_hz,
        speed_m_s  = speed_m_s,
        atmo_abs   = atmo_abs,
    )

    # Use trace_surface (preferred) if available — returns segs + per-triangle flux.
    _has_surface = hasattr(tracer, 'trace_surface')
    if _has_surface:
        result = tracer.trace_surface(
            src_pos          = src_pos,
            src_dir          = src_dir,
            src_directivity  = src_dexp,
            n_rays           = n_rays,
            max_bounces      = max_bounces,
            min_amplitude    = min_amplitude,
            seed             = seed,
            out_cap          = out_cap,
        )
        segs             = result['segs']          # (N_segs, 12) float32
        surface_direct   = result['direct']        # (n_tri, n_bands) float32
        surface_indirect = result['indirect']      # (n_tri, n_bands) float32
    else:
        segs = tracer.trace(
            src_pos          = src_pos,
            src_dir          = src_dir,
            src_directivity  = src_dexp,
            n_rays           = n_rays,
            max_bounces      = max_bounces,
            min_amplitude    = min_amplitude,
            seed             = seed,
            out_cap          = out_cap,
        )
        surface_direct   = np.zeros((n_tri, n_bands), dtype=np.float32)
        surface_indirect = np.zeros((n_tri, n_bands), dtype=np.float32)

    # Metadata.
    if len(segs) > 0:
        all_pts   = np.vstack([segs[:, :3], segs[:, 3:6]])
        bbox_min  = all_pts.min(axis=0).astype(np.float32)
        bbox_max  = all_pts.max(axis=0).astype(np.float32)
        max_path  = float(segs[:, 11].max())
    else:
        bbox_min = bbox_max = np.zeros(3, dtype=np.float32)
        max_path = 1.0

    # Per-triangle surface illumination.
    # Prefer tracer-owned flux (trace_surface); fall back to nearest-centroid.
    if _has_surface:
        surface_flux = surface_direct + surface_indirect  # (n_tri, n_bands)
    else:
        # Legacy nearest-centroid fallback.
        surface_flux = np.zeros((n_tri, n_bands), dtype=np.float32)
        if len(segs) > 0 and n_tri > 0:
            tri_centroids = verts.reshape(n_tri, 3, 3).mean(axis=1).astype(np.float32)
            p1  = segs[:, 3:6].astype(np.float32)
            amp = segs[:, 9].astype(np.float32)
            BATCH = 4096
            for start in range(0, len(p1), BATCH):
                end     = min(start + BATCH, len(p1))
                diff    = p1[start:end, None, :] - tri_centroids[None, :, :]
                nearest = (diff * diff).sum(axis=2).argmin(axis=1)
                np.add.at(surface_flux[:, 0], nearest, amp[start:end])
            surface_direct   = surface_flux
            surface_indirect = np.zeros_like(surface_flux)

    # Scalar (per-triangle, band-summed, normalised to [0,1]).
    surface_scalar = surface_flux.sum(axis=1).astype(np.float32)
    peak = surface_scalar.max()
    if peak > 1e-9:
        surface_scalar /= peak

    # Spectral RGB for surface colour.
    surface_rgb = spectral_bands_to_rgb(surface_flux, freq_hz)

    # For the visual display, show only baffle panels (the instrument body) when
    # they exist.  The absorptive enclosure box is irrelevant for visualisation
    # and its 1 m scale would swamp the camera framing of the tiny instrument.
    if n_room_tris > 0 and n_room_tris < n_tri:
        disp_verts    = verts[n_room_tris:]
        disp_norms    = normals[n_room_tris:]
        disp_scalar   = surface_scalar[n_room_tris:]
        disp_direct   = surface_direct[n_room_tris:]
        disp_indirect = surface_indirect[n_room_tris:]
        disp_flux     = surface_flux[n_room_tris:]
        disp_rgb      = surface_rgb[n_room_tris:]
    else:
        disp_verts    = verts
        disp_norms    = normals
        disp_scalar   = surface_scalar
        disp_direct   = surface_direct
        disp_indirect = surface_indirect
        disp_flux     = surface_flux
        disp_rgb      = surface_rgb

    n_disp = len(disp_verts)
    geo_verts_flat = disp_verts.reshape(n_disp, 9).astype(np.float32) if n_disp else np.zeros((0, 9), dtype=np.float32)
    geo_normals    = disp_norms.astype(np.float32) if n_disp else np.zeros((0, 3), dtype=np.float32)

    # Global scene field integration — sensor-independent, covers ALL triangles
    # (room shell + baffles) so the energy budget is complete.  Suitable as
    # seed data for OpenGL shaders via meta['scene_field'].as_uniform_vec().
    scene_field = integrate_scene_fields(
        segs             = segs,
        surface_flux     = surface_flux,      # full n_tri, not display subset
        surface_direct   = surface_direct,
        surface_indirect = surface_indirect,
        tri_verts        = verts,             # (n_tri, 3, 3) float64
        freq_hz          = freq_hz,
    )

    meta = {
        'n_sources':        n_sources,
        'n_bands':          n_bands,
        'freq_hz':          freq_hz,
        'max_path_length':  max_path,
        'bbox_min':         bbox_min,
        'bbox_max':         bbox_max,
        'geo_verts_flat':   geo_verts_flat,
        'geo_normals':      geo_normals,
        # Ray-traced surface illumination (tracer-owned, preferred):
        'surface_flux':     disp_flux,     # (n_disp_tri, n_bands) float32
        'surface_direct':   disp_direct,   # (n_disp_tri, n_bands) float32
        'surface_indirect': disp_indirect, # (n_disp_tri, n_bands) float32
        'surface_rgb':      disp_rgb,      # (n_disp_tri, 3)       float32
        'surface_scalar':   disp_scalar,   # (n_disp_tri,)         float32
        # Legacy alias:
        'surface_illum':    disp_scalar,
        # Spectral material emission / re-emission (full n_tri, not just display):
        'emission_bands':   emission_bands.astype(np.float32),   # (n_tri, n_bands)
        'reemission_bands': reemission_bands.astype(np.float32), # (n_tri, n_bands)
        'diffusion_bands':  diffusion_bands.astype(np.float32),  # (n_tri, n_bands)
        # Global field integration (sensor-independent, GL shader seed):
        'scene_field':      scene_field,   # SceneFieldIntegration
    }

    return segs.astype(np.float32), meta


# ---------------------------------------------------------------------------
# Convenience: build RayAccumulatorWidget pre-loaded with a scene trace
# ---------------------------------------------------------------------------

def make_ray_accumulator_widget(
        scene,
        n_rays:        int   = 256,
        max_bounces:   int   = 8,
        n_bands:       int   = 12,
        min_amplitude: float = 0.005,
        speed_m_s:     float = 343.0,
        seed:          int   = 42,
        bounce_decay:  float = 1.6,
        point_size:    float = 12.0,
        amp_scale:     float = 3.0,
):
    """Trace scene and return a fully-loaded RayAccumulatorWidget.

    Convenience wrapper — equivalent to::

        segs, meta = trace_cavity_scene(scene, ...)
        widget = RayAccumulatorWidget(n_sources=..., n_bands=...)
        widget.update_segments(segs)

    Call ``widget.set_playhead(t)`` each frame with ``t`` in [0, 1] where
    1.0 shows all rays.  Map audio playback position to::

        t = playback_sample / sample_rate * speed_m_s / meta['max_path_length']
    """
    from opengl_widget import RayAccumulatorWidget

    segs, meta = trace_cavity_scene(
        scene,
        n_rays=n_rays, max_bounces=max_bounces, n_bands=n_bands,
        min_amplitude=min_amplitude, speed_m_s=speed_m_s, seed=seed,
    )

    widget = RayAccumulatorWidget(
        n_sources    = meta['n_sources'],
        n_bands      = meta['n_bands'],
        bounce_decay = bounce_decay,
        point_size   = point_size,
        amp_scale    = amp_scale,
    )
    widget.update_segments(segs)

    return widget, meta


# ---------------------------------------------------------------------------
# Field solver bridge
# ---------------------------------------------------------------------------

# Polar type constants matching RtsPolarType enum in rt_field_solver.h
RTS_OMNI          = 0
RTS_CARDIOID      = 1
RTS_FIGURE8       = 2
RTS_HYPERCARDIOID = 3
RTS_APERTURE      = 4


def _build_rts_scene(
        verts:     np.ndarray,
        normals:   np.ndarray,
        mat_props: np.ndarray,
        freq_hz:   np.ndarray,
        medium_n_re: float = 1.0,
        medium_n_im: float = 0.0,
) -> dict:
    """Convert triangulated room arrays into a scene dict for FieldSolver.

    Uses the same KK-consistent complex reflectance model as the ray tracer
    but repackages it as the normalised impedance (Z/Z_air) that rts_solve
    expects for acoustic mode, or as ñ for EM mode.

    For acoustic mode the surface impedance Z/Z_air is derived from the
    reflectance r via the plane-wave relation:
        r = (Z - 1) / (Z + 1)   →   Z = (1 + r) / (1 - r)

    mat_diffusion is kept as a separate per-material scalar.

    Returns
    -------
    dict with keys matching PyFieldSolver constructor expectations.
    """
    n_tri   = len(verts)
    n_bands = len(freq_hz)

    # KK reflectances: shape (n_tri, n_bands)
    refl_re, refl_im = build_complex_reflectances(mat_props, freq_hz)

    # Each triangle is its own material so n_mats == n_tri.
    # Derive normalised impedance from reflectance.
    r_c  = refl_re + 1j * refl_im   # complex reflectance (n_tri, n_bands)
    denom = 1.0 - r_c
    # Guard near-zero denominators (perfect reflector)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
    Z_c  = (1.0 + r_c) / denom      # normalised impedance (n_tri, n_bands)

    # Assign each triangle its own material index.
    mat_idx  = np.arange(n_tri, dtype=np.int32)

    return {
        'verts':         verts.reshape(n_tri, 9).astype(np.float64),
        'normals':       normals.astype(np.float64),
        'mat_idx':       mat_idx,
        'n_mats':        n_tri,
        'mat_n_re':      Z_c.real.astype(np.float64),
        'mat_n_im':      Z_c.imag.astype(np.float64),
        'mat_diffusion': mat_props[:, 1].astype(np.float64),
        'medium_n_re':   float(medium_n_re),
        'medium_n_im':   float(medium_n_im),
    }


def _build_rts_scene_em(
        verts:     np.ndarray,
        normals:   np.ndarray,
        mat_props: np.ndarray,
        freq_hz:   np.ndarray,
        medium_n_re: float = 1.0,
        medium_n_im: float = 0.0,
) -> dict:
    """Scene dict for EM mode.

    For EM, mat_n = ñ (complex refractive index).  We approximate ñ from
    the acoustic/optical reflectance using Fresnel normal incidence:
        r = (1 - ñ) / (1 + ñ)   →   ñ = (1 - r) / (1 + r)

    For surfaces that are not dielectrics you can override with measured ñ
    values by bypassing this helper entirely and constructing the dict manually.
    """
    n_tri   = len(verts)
    n_bands = len(freq_hz)

    refl_re, refl_im = build_complex_reflectances(mat_props, freq_hz)
    r_c  = refl_re + 1j * refl_im
    denom = 1.0 + r_c
    denom = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
    n_c  = (1.0 - r_c) / denom      # complex refractive index

    mat_idx = np.arange(n_tri, dtype=np.int32)

    return {
        'verts':         verts.reshape(n_tri, 9).astype(np.float64),
        'normals':       normals.astype(np.float64),
        'mat_idx':       mat_idx,
        'n_mats':        n_tri,
        'mat_n_re':      n_c.real.astype(np.float64),
        'mat_n_im':      n_c.imag.astype(np.float64),
        'mat_diffusion': mat_props[:, 1].astype(np.float64),
        'medium_n_re':   float(medium_n_re),
        'medium_n_im':   float(medium_n_im),
    }


def make_omni_receiver(pos, axis=(0.0, 1.0, 0.0)) -> dict:
    """Convenience: omnidirectional point receiver."""
    return {
        'pos':        np.asarray(pos,  dtype=np.float64),
        'axis':       np.asarray(axis, dtype=np.float64),
        'polar_type': RTS_OMNI,
        'aperture_r': 0.0,
        'pol_s':      np.zeros(3, dtype=np.float64),
        'pol_p':      np.zeros(3, dtype=np.float64),
    }


def make_cardioid_receiver(pos, axis) -> dict:
    """Cardioid receiver pointed along `axis`."""
    a = np.asarray(axis, dtype=np.float64)
    nm = np.linalg.norm(a)
    if nm > 1e-9:
        a /= nm
    return {
        'pos':        np.asarray(pos, dtype=np.float64),
        'axis':       a,
        'polar_type': RTS_CARDIOID,
        'aperture_r': 0.0,
        'pol_s':      np.zeros(3, dtype=np.float64),
        'pol_p':      np.zeros(3, dtype=np.float64),
    }


def make_aperture_receiver(pos, axis, aperture_r: float = 0.05) -> dict:
    """Aperture receiver (disc integral, polar-weighted)."""
    a = np.asarray(axis, dtype=np.float64)
    nm = np.linalg.norm(a)
    if nm > 1e-9:
        a /= nm
    return {
        'pos':        np.asarray(pos, dtype=np.float64),
        'axis':       a,
        'polar_type': RTS_APERTURE,
        'aperture_r': float(aperture_r),
        'pol_s':      np.zeros(3, dtype=np.float64),
        'pol_p':      np.zeros(3, dtype=np.float64),
    }


def make_em_receiver(pos, axis, pol_s, pol_p) -> dict:
    """EM receiver with explicit s/p detector polarisation axes."""
    def _n(v):
        v = np.asarray(v, dtype=np.float64)
        nm = np.linalg.norm(v)
        return v / nm if nm > 1e-9 else v
    return {
        'pos':        np.asarray(pos, dtype=np.float64),
        'axis':       _n(axis),
        'polar_type': RTS_OMNI,
        'aperture_r': 0.0,
        'pol_s':      _n(pol_s),
        'pol_p':      _n(pol_p),
    }


def _normalised_vec3(v, *, name: str = "vector") -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    nm = float(np.linalg.norm(arr))
    if nm > 1e-9:
        return arr / nm
    raise ValueError(f"{name} must be a non-zero 3-vector")


def _mic_pattern_to_receiver_type(pattern: str, aperture_r: float) -> int:
    p = str(pattern or "cardioid").strip().lower().replace("-", "_")
    if aperture_r > 0.0:
        return RTS_APERTURE
    if p in ("omni", "omnidirectional"):
        return RTS_OMNI
    if p in ("cardioid", "cardioid_pressure_velocity"):
        return RTS_CARDIOID
    if p in ("figure8", "figure_8", "bidirectional"):
        return RTS_FIGURE8
    if p in ("hypercardioid", "hyper"):
        return RTS_HYPERCARDIOID
    raise ValueError(
        "Unsupported mic pattern "
        f"{pattern!r}; expected omni, cardioid, figure8, hypercardioid, or aperture"
    )


def mic_pattern_from_pressure_velocity(polar_a: float, polar_b: float) -> str:
    """Map a pressure/velocity mic mix to the nearest ray-solver pattern.

    The co-evolver mic model stores first-order microphone response as
    ``polar_a * P + polar_b * rho*c*v_n``.  The ray solver has named polar
    patterns, so this helper accepts only the exact presets we can represent
    without inventing a response that the C solver cannot honour.
    """
    a = float(polar_a)
    b = float(polar_b)
    if abs(a - 1.0) < 1e-6 and abs(b) < 1e-6:
        return "omni"
    if abs(a - 0.5) < 1e-6 and abs(b - 0.5) < 1e-6:
        return "cardioid"
    if abs(a) < 1e-6 and abs(b - 1.0) < 1e-6:
        return "figure8"
    if abs(a - 0.25) < 1e-6 and abs(b - 0.75) < 1e-6:
        return "hypercardioid"
    raise ValueError(
        "Mic polar mix cannot be represented exactly by the ray solver: "
        f"polar_a={a}, polar_b={b}. Use one of omni (1,0), cardioid (0.5,0.5), "
        "figure8 (0,1), or hypercardioid (0.25,0.75)."
    )


def make_microphone_receiver(
        mic_def: dict | None = None,
        *,
        pos=None,
        axis=None,
        pattern: str | None = None,
        aperture_r: float = 0.0,
) -> dict:
    """Build a receiver dict for the high-frequency microphone ray path.

    This is the microphone-side half of the bidirectional high-band plan:
    the receiver has an orientation, polar pattern, and optional aperture
    radius, and ``solve_highband_mic_transfer`` connects source ray subpaths
    to this receiver coherently with shadow rays.

    ``mic_def`` may be a co-evolver mic descriptor with ``pos``, ``axis``,
    ``polar_a`` and ``polar_b``.  Exact co-evolver presets map to the matching
    ray-solver pattern.  Required fields are not defaulted: malformed mic
    descriptors fail immediately.
    """
    if mic_def is not None:
        if pos is None:
            if "pos" not in mic_def:
                raise ValueError("mic_def must contain 'pos'")
            pos = mic_def["pos"]
        if axis is None:
            if "axis" not in mic_def:
                raise ValueError("mic_def must contain 'axis'")
            axis = mic_def["axis"]
        if pattern is None:
            if "polar_a" not in mic_def or "polar_b" not in mic_def:
                raise ValueError("mic_def must contain exact 'polar_a' and 'polar_b'")
            pattern = mic_pattern_from_pressure_velocity(
                float(mic_def["polar_a"]),
                float(mic_def["polar_b"]),
            )
        if aperture_r == 0.0:
            aperture_r = float(mic_def.get("aperture_r", 0.0))

    if pos is None:
        raise ValueError("make_microphone_receiver requires pos or mic_def['pos']")
    if axis is None:
        raise ValueError("make_microphone_receiver requires axis or mic_def['axis']")
    if pattern is None:
        raise ValueError("make_microphone_receiver requires pattern or mic_def polar coefficients")

    rec = {
        'pos':        np.asarray(pos, dtype=np.float64).reshape(3),
        'axis':       _normalised_vec3(axis, name="microphone axis"),
        'polar_type': _mic_pattern_to_receiver_type(pattern, float(aperture_r)),
        'aperture_r': float(aperture_r),
        'pol_s':      np.zeros(3, dtype=np.float64),
        'pol_p':      np.zeros(3, dtype=np.float64),
        'kind':       'microphone',
        'pattern':    str(pattern),
    }
    return rec


def _highband_weight(freq_hz: np.ndarray, crossover_hz: float, order: int = 4) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=np.float64)
    fc = max(float(crossover_hz), 1e-9)
    p = max(1, int(order))
    r = (np.maximum(f, 0.0) / fc) ** p
    return r / (1.0 + r)


def solve_highband_mic_transfer(
        scene,
        mic_defs:       list[dict] | None = None,
        receivers:      list[dict] | None = None,
        *,
        n_rays:         int   = 2048,
        max_bounces:    int   = 10,
        n_bands:        int   = 16,
        min_amplitude:  float = 0.0005,
        speed_m_s:      float = 343.0,
        seed:           int   = 42,
        crossover_hz:   float = 1500.0,
        crossover_order:int   = 4,
        aperture_r:     float = 0.012,
        freq_hz:        np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve the upper-frequency microphone transfer matrix.

    This function intentionally targets the mic path, not pickups.  Low and
    low-mid pressure still come from the co-evolved FDTD microphone.  Above
    ``crossover_hz``, this ray transfer adds source-to-mic geometric detail:
    source directivity, specular/diffuse bounces, occlusion, per-band phase,
    atmospheric loss, microphone polar response, and aperture averaging.

    Implementation note: the current C ``FieldSolver`` uses a bidirectional
    estimator in the practical acoustic-rendering sense: source subpaths are
    traced through the scene, and every bounce is coherently connected to each
    microphone sample by an occlusion-tested shadow ray.  That gives the mic
    its own directional/area acceptance instead of treating it as a passive
    afterthought of source broadcasting.

    Returns
    -------
    H_high : complex128 ndarray, shape (n_sources, n_mics, n_bands)
        High-pass-weighted transfer.  Multiply/convolve source spectra through
        this and add it to the low-band FDTD mic signal.
    meta : dict
        Includes ``freq_hz``, ``highband_weight`` and the receiver descriptors.
    """
    if receivers is None:
        mic_defs = list(mic_defs or [])
        if not mic_defs:
            raise ValueError("solve_highband_mic_transfer requires mic_defs or receivers")
        receivers = [
            make_microphone_receiver(m, aperture_r=float(m.get("aperture_r", aperture_r)))
            for m in mic_defs
        ]
    else:
        receivers = list(receivers)

    sources = getattr(scene, 'sources', None)
    if not sources:
        raise ValueError(
            "solve_highband_mic_transfer requires explicit scene.sources; "
            "implicit centroid sources are not valid for the mic high-band path"
        )
    if not receivers:
        raise ValueError("solve_highband_mic_transfer requires at least one microphone receiver")

    H, meta = solve_transfer_matrix(
        scene,
        receivers=receivers,
        mode='acoustic',
        n_rays=n_rays,
        max_bounces=max_bounces,
        n_bands=n_bands,
        min_amplitude=min_amplitude,
        speed_m_s=speed_m_s,
        seed=seed,
        schroeder_hz=0.0,
        freq_hz=freq_hz,
    )

    weight = _highband_weight(meta['freq_hz'], crossover_hz, crossover_order)
    H_high = H * weight.reshape(1, 1, -1)
    meta = dict(meta)
    meta.update({
        'path':              'mic_highband_bidirectional',
        'crossover_hz':      float(crossover_hz),
        'crossover_order':   int(crossover_order),
        'highband_weight':   weight.astype(np.float64),
        'receivers':         receivers,
        'estimator':         'source_subpaths_with_receiver_shadow_connections',
    })
    return H_high, meta


def apply_mic_transfer_frequency_domain(
        source_signals: np.ndarray,
        H_mic:          np.ndarray,
        freq_hz:        np.ndarray,
        sample_rate:    float,
        n_out:          int | None = None,
) -> np.ndarray:
    """Apply a band-sampled mic transfer matrix to source signals.

    ``H_mic`` is interpolated over FFT bins and summed across sources.  This is
    meant for the high-band ray add-on; the returned signal can be added to the
    co-evolver/FDTD mic output after level calibration.
    """
    x = np.asarray(source_signals)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("source_signals must have shape (n_sources, n_samples)")

    H = np.asarray(H_mic, dtype=np.complex128)
    if H.ndim != 3:
        raise ValueError("H_mic must have shape (n_sources, n_mics, n_bands)")
    if H.shape[0] != x.shape[0]:
        raise ValueError(
            f"H_mic source count {H.shape[0]} does not match source_signals {x.shape[0]}"
        )

    n = int(n_out or x.shape[1])
    fft_n = max(n, x.shape[1])
    bins = np.fft.rfftfreq(fft_n, d=1.0 / float(sample_rate))
    band_f = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    if len(band_f) != H.shape[2]:
        raise ValueError("freq_hz length must match H_mic band count")
    if np.any(np.diff(band_f) <= 0.0):
        raise ValueError("freq_hz must be strictly increasing")
    if band_f[0] > bins[0] or band_f[-1] < bins[-1]:
        raise ValueError(
            "freq_hz must cover the full FFT range [0, Nyquist] for deterministic "
            "mic transfer application; no extrapolation is performed"
        )

    X = np.fft.rfft(x, n=fft_n, axis=1)
    Y = np.zeros((H.shape[1], len(bins)), dtype=np.complex128)
    for si in range(H.shape[0]):
        for mi in range(H.shape[1]):
            h_re = np.interp(bins, band_f, H[si, mi].real)
            h_im = np.interp(bins, band_f, H[si, mi].imag)
            Y[mi] += X[si] * (h_re + 1j * h_im)

    y = np.fft.irfft(Y, n=fft_n, axis=1)[:, :n]
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Integrators: bulk IR accumulation and image rendering
# ---------------------------------------------------------------------------

def integrate_cavity_ir(
        scene,
        receivers,
        sample_rate:   float = 44100.0,
        n_samples:     int   = 4096,
        n_rays:        int   = 512,
        max_bounces:   int   = 12,
        n_bands:       int   = 12,
        min_amplitude: float = 0.001,
        speed_m_s:     float = 343.0,
        seed:          int   = 42,
) -> tuple[np.ndarray, dict]:
    """Trace a CavityScene and accumulate per-receiver impulse responses.

    Parameters
    ----------
    scene      : CavityScene
    receivers  : list — each element is either a (3,) array-like position, or
                 a dict with keys ``'pos'`` (3,) and optionally ``'aperture_r'``
                 (float, default 0.5 m).
    sample_rate: IR sample rate in Hz.
    n_samples  : number of IR time bins.
    n_rays     : rays per source.
    max_bounces: max reflections per ray.
    n_bands    : number of frequency bands.
    min_amplitude: ray cutoff threshold.
    speed_m_s  : wave propagation speed.
    seed       : RNG seed.

    Returns
    -------
    ir  : complex64 ndarray, shape (n_src, n_rec, n_bands, n_samples)
          Coherent complex impulse response per (source, receiver, band).
    meta: dict — freq_hz, n_sources, n_bands, sample_rate, n_samples
    """
    if not _HAS_C_TRACER:
        raise RuntimeError(
            "_spectral_kernels.RayTracer not available — rebuild the C extension")

    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    if room is None:
        raise TypeError(f"Scene {type(scene).__name__} has neither 'room' nor 'geometry'")

    n_room_tris = 0
    if hasattr(room, 'vertices_xy'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(room)
    elif hasattr(room, 'radius'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(
            _circular_room_to_polygon(room))
    elif hasattr(room, 'triangles'):
        verts, normals, mat_props = _triangulate_mesh_room(room)
    else:
        raise TypeError(f"Unsupported room type: {type(room).__name__}")

    verts, normals, mat_props = _drop_absorptive_helper_shell(
        room, verts, normals, mat_props, n_room_tris)

    n_tri = len(verts)
    if n_tri == 0:
        raise ValueError("Room triangulated to zero triangles")

    freq_hz = build_frequency_bands(scene, n_bands)
    refl_re, refl_im = build_complex_reflectances(mat_props, freq_hz)
    atmo_abs = 0.01 * (freq_hz / 1000.0) ** 1.5

    sources = getattr(scene, 'sources', [])
    if sources:
        src_pos  = np.array([s.position         for s in sources], dtype=np.float64)
        src_dir  = np.array([s.direction         for s in sources], dtype=np.float64)
        src_dexp = np.array([s.directivity_power for s in sources], dtype=np.float64)
    else:
        all_v   = verts.reshape(-1, 3)
        centroid = (all_v.min(axis=0) + all_v.max(axis=0)) * 0.5
        src_pos  = (centroid + np.array([0.1, 0.2, 0.0]))[None, :]
        src_dir  = np.array([[0.0, 0.0, 1.0]])
        src_dexp = np.array([0.0])

    norms = np.linalg.norm(src_dir, axis=1, keepdims=True)
    src_dir = src_dir / np.where(norms > 1e-9, norms, 1.0)

    # Unpack receivers.
    rec_pos_list = []
    rec_apr_list = []
    for r in receivers:
        if isinstance(r, dict):
            rec_pos_list.append(np.asarray(r['pos'], dtype=np.float64))
            rec_apr_list.append(float(r.get('aperture_r', 0.5)))
        else:
            rec_pos_list.append(np.asarray(r, dtype=np.float64))
            rec_apr_list.append(0.5)

    rec_pos_arr = np.array(rec_pos_list, dtype=np.float64)   # (n_rec, 3)
    rec_apr_arr = np.array(rec_apr_list, dtype=np.float64)   # (n_rec,)

    mat_idx, mat_buf, mat_n_mats = per_tri_spectral_to_mat_buf(
        refl_re, refl_im, mat_props[:, 1], freq_hz,
    )
    tracer = _CRayTracer(
        n_tri      = n_tri,
        verts      = verts.reshape(n_tri, 9),
        normals    = normals,
        mat_idx    = mat_idx,
        mat_buf    = mat_buf,
        mat_n_mats = mat_n_mats,
        freq_hz    = freq_hz,
        speed_m_s  = speed_m_s,
        atmo_abs   = atmo_abs,
    )

    ir_re, ir_im = tracer.integrate_ir(
        src_pos          = src_pos,
        src_dir          = src_dir,
        src_directivity  = src_dexp,
        rec_pos          = rec_pos_arr,
        rec_aperture_r   = rec_apr_arr,
        speed_m_s        = speed_m_s,
        sample_rate      = sample_rate,
        n_samples        = n_samples,
        n_rays           = n_rays,
        max_bounces      = max_bounces,
        min_amplitude    = min_amplitude,
        seed             = seed,
    )

    ir = np.empty(ir_re.shape, dtype=np.complex64)
    ir.real[:] = ir_re
    ir.imag[:] = ir_im

    meta = {
        'freq_hz':     freq_hz,
        'n_sources':   len(src_pos),
        'n_bands':     len(freq_hz),
        'sample_rate': sample_rate,
        'n_samples':   n_samples,
    }
    return ir, meta


def integrate_cavity_image(
        scene,
        cam_pos,
        cam_fwd,
        cam_up        = (0.0, 1.0, 0.0),
        fov_rad:   float = 1.0,
        width:     int   = 512,
        height:    int   = 384,
        n_rays:    int   = 512,
        max_bounces: int = 12,
        n_bands:   int   = 12,
        min_amplitude: float = 0.001,
        speed_m_s: float = 343.0,
        seed:      int   = 42,
) -> tuple[np.ndarray, dict]:
    """Trace a CavityScene and render an acoustic energy image.

    Each ray hit is projected through a pinhole camera and its per-band
    amplitude |A[b]| is splatted into the image buffer.  The result shows
    where acoustic energy concentrates in the scene as seen from the camera.

    Parameters
    ----------
    scene      : CavityScene
    cam_pos    : (3,) camera world position.
    cam_fwd    : (3,) camera look direction.
    cam_up     : (3,) up hint (orthogonalised internally).
    fov_rad    : full vertical field of view in radians.
    width, height : image resolution in pixels.
    n_rays     : rays per source.
    max_bounces: max reflections.
    n_bands    : number of frequency bands.
    min_amplitude: ray cutoff.
    speed_m_s  : wave speed (unused except for scene build, kept for symmetry).
    seed       : RNG seed.

    Returns
    -------
    image : float32 ndarray, shape (n_bands, height, width)
            Accumulated |amplitude| per pixel per band.
            Sum over bands for a scalar energy map, or slice [b] for one band.
    meta  : dict — freq_hz, n_bands
    """
    if not _HAS_C_TRACER:
        raise RuntimeError(
            "_spectral_kernels.RayTracer not available — rebuild the C extension")

    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    if room is None:
        raise TypeError(f"Scene {type(scene).__name__} has neither 'room' nor 'geometry'")

    n_room_tris = 0
    if hasattr(room, 'vertices_xy'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(room)
    elif hasattr(room, 'radius'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(
            _circular_room_to_polygon(room))
    elif hasattr(room, 'triangles'):
        verts, normals, mat_props = _triangulate_mesh_room(room)
    else:
        raise TypeError(f"Unsupported room type: {type(room).__name__}")

    verts, normals, mat_props = _drop_absorptive_helper_shell(
        room, verts, normals, mat_props, n_room_tris)

    n_tri = len(verts)
    if n_tri == 0:
        raise ValueError("Room triangulated to zero triangles")

    freq_hz = build_frequency_bands(scene, n_bands)
    refl_re, refl_im = build_complex_reflectances(mat_props, freq_hz)
    atmo_abs = 0.01 * (freq_hz / 1000.0) ** 1.5

    sources = getattr(scene, 'sources', [])
    if sources:
        src_pos  = np.array([s.position         for s in sources], dtype=np.float64)
        src_dir  = np.array([s.direction         for s in sources], dtype=np.float64)
        src_dexp = np.array([s.directivity_power for s in sources], dtype=np.float64)
    else:
        all_v   = verts.reshape(-1, 3)
        centroid = (all_v.min(axis=0) + all_v.max(axis=0)) * 0.5
        src_pos  = (centroid + np.array([0.1, 0.2, 0.0]))[None, :]
        src_dir  = np.array([[0.0, 0.0, 1.0]])
        src_dexp = np.array([0.0])

    norms = np.linalg.norm(src_dir, axis=1, keepdims=True)
    src_dir = src_dir / np.where(norms > 1e-9, norms, 1.0)

    mat_idx, mat_buf, mat_n_mats = per_tri_spectral_to_mat_buf(
        refl_re, refl_im, mat_props[:, 1], freq_hz,
    )
    tracer = _CRayTracer(
        n_tri      = n_tri,
        verts      = verts.reshape(n_tri, 9),
        normals    = normals,
        mat_idx    = mat_idx,
        mat_buf    = mat_buf,
        mat_n_mats = mat_n_mats,
        freq_hz    = freq_hz,
        speed_m_s  = speed_m_s,
        atmo_abs   = atmo_abs,
    )

    image = tracer.integrate_image(
        src_pos          = src_pos,
        src_dir          = src_dir,
        src_directivity  = src_dexp,
        cam_pos          = np.asarray(cam_pos, dtype=np.float64),
        cam_fwd          = np.asarray(cam_fwd, dtype=np.float64),
        cam_up           = np.asarray(cam_up,  dtype=np.float64),
        fov_rad          = fov_rad,
        width            = width,
        height           = height,
        n_rays           = n_rays,
        max_bounces      = max_bounces,
        min_amplitude    = min_amplitude,
        seed             = seed,
    )

    meta = {
        'freq_hz': freq_hz,
        'n_bands': len(freq_hz),
    }
    return image, meta


def solve_transfer_matrix(
        scene,
        receivers:      list,
        mode:           str            = 'acoustic',
        n_rays:         int            = 512,
        max_bounces:    int            = 8,
        n_bands:        int            = 12,
        min_amplitude:  float          = 0.001,
        speed_m_s:      float          = 343.0,
        seed:           int            = 42,
        schroeder_hz:   float          = 0.0,
        medium_n_re:    float          = 1.0,
        medium_n_im:    float          = 0.0,
        freq_hz:        np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute the complex transfer matrix H[source → receiver, freq].

    Parameters
    ----------
    scene        : CavityScene — room + sources
    receivers    : list of receiver dicts from make_*_receiver helpers
    mode         : 'acoustic' or 'em'
    n_rays       : rays per source (per polarisation for EM)
    max_bounces  : maximum reflections
    n_bands      : number of frequency bands
    min_amplitude: ray amplitude cutoff
    speed_m_s    : wave propagation speed
    seed         : RNG seed for diffuse scatter
    schroeder_hz : hybrid crossover (0 = ray-only; auto sets ~1900*sqrt(T60/V))
    medium_n_re  : real part of medium refractive index
    medium_n_im  : imaginary part (propagation loss in medium)

    Returns
    -------
    H    : complex128 ndarray
           acoustic — shape (n_src, n_rec, n_bands)
           em       — shape (n_src, n_rec, n_bands, 2, 2)
    meta : dict with keys: freq_hz, n_sources, n_bands, mode
    """
    if not _HAS_FIELD_SOLVER:
        raise RuntimeError(
            "_spectral_kernels.FieldSolver not available — rebuild with "
            "`cmake -S csrc -B csrc_build && cmake --build csrc_build --config Release`"
        )

    room = getattr(scene, 'room', None) or getattr(scene, 'geometry', None)
    if room is None:
        raise TypeError(f"Scene {type(scene).__name__} has neither 'room' nor 'geometry'")

    # Triangulate room geometry.
    n_room_tris = 0
    if hasattr(room, 'vertices_xy'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(room)
    elif hasattr(room, 'radius'):
        verts, normals, mat_props, n_room_tris = _triangulate_polygonal_room(
            _circular_room_to_polygon(room))
    elif hasattr(room, 'triangles'):
        verts, normals, mat_props = _triangulate_mesh_room(room)
    else:
        raise TypeError(f"Unsupported room type: {type(room).__name__}")

    verts, normals, mat_props = _drop_absorptive_helper_shell(
        room, verts, normals, mat_props, n_room_tris)

    n_tri = len(verts)
    if n_tri == 0:
        raise ValueError("Room triangulated to zero triangles")

    if freq_hz is not None:
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
    else:
        freq_hz = build_frequency_bands(scene, n_bands)

    # Build scene dict.
    if mode == 'em':
        rts_scene = _build_rts_scene_em(verts, normals, mat_props, freq_hz,
                                        medium_n_re, medium_n_im)
    else:
        rts_scene = _build_rts_scene(verts, normals, mat_props, freq_hz,
                                     medium_n_re, medium_n_im)

    # Source arrays.
    sources = getattr(scene, 'sources', [])
    if sources:
        src_pos  = np.array([s.position         for s in sources], dtype=np.float64)
        src_dir  = np.array([s.direction         for s in sources], dtype=np.float64)
        src_dexp = np.array([s.directivity_power for s in sources], dtype=np.float64)
    else:
        all_v    = verts.reshape(-1, 3)
        centroid = (all_v.min(axis=0) + all_v.max(axis=0)) * 0.5
        src_pos  = (centroid + np.array([0.1, 0.2, 0.0]))[None, :]
        src_dir  = np.array([[0.0, 0.0, 1.0]])
        src_dexp = np.array([0.0])

    n_sources = len(src_pos)
    norms = np.linalg.norm(src_dir, axis=1, keepdims=True)
    src_dir = src_dir / np.where(norms > 1e-9, norms, 1.0)

    # Build solver.
    solver = _CFieldSolver(
        scene      = rts_scene,
        receivers  = receivers,
        freq_hz    = freq_hz,
        speed_m_s  = speed_m_s,
        mode       = mode,
    )

    # Solve.
    if mode == 'em':
        # Default source polarisation: real x-axis, imaginary y-axis
        # so each source emits two orthogonal linear polarisations.
        n_src = len(src_pos)
        src_pol_re = np.tile([1.0, 0.0, 0.0], (n_src, 1)).astype(np.float64)
        src_pol_im = np.tile([0.0, 1.0, 0.0], (n_src, 1)).astype(np.float64)
        H = solver.solve_em(
            src_pos, src_dir, src_dexp,
            src_pol_re, src_pol_im,
            n_rays=n_rays, max_bounces=max_bounces,
            min_amplitude=min_amplitude, seed=seed,
            schroeder_hz=schroeder_hz,
        )
    else:
        H = solver.solve_acoustic(
            src_pos, src_dir, src_dexp,
            n_rays=n_rays, max_bounces=max_bounces,
            min_amplitude=min_amplitude, seed=seed,
            schroeder_hz=schroeder_hz,
        )

    meta = {
        'freq_hz':   freq_hz,
        'n_sources': n_sources,
        'n_bands':   len(freq_hz),
        'mode':      mode,
    }
    return H, meta
