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
    refl_re, refl_im = build_complex_reflectances(mat_props, freq_hz)

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
    tracer = _CRayTracer(
        n_tri     = n_tri,
        verts     = verts.reshape(n_tri, 9),          # (n_tri, 9) ≡ (n_tri, 3, 3)
        normals   = normals,
        refl_re   = refl_re,
        refl_im   = refl_im,
        diffusion = mat_props[:, 1],
        freq_hz   = freq_hz,
        speed_m_s = speed_m_s,
        atmo_abs  = atmo_abs,
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

    tracer = _CRayTracer(
        n_tri     = n_tri,
        verts     = verts.reshape(n_tri, 9),
        normals   = normals,
        refl_re   = refl_re,
        refl_im   = refl_im,
        diffusion = mat_props[:, 1],
        freq_hz   = freq_hz,
        speed_m_s = speed_m_s,
        atmo_abs  = atmo_abs,
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

    tracer = _CRayTracer(
        n_tri     = n_tri,
        verts     = verts.reshape(n_tri, 9),
        normals   = normals,
        refl_re   = refl_re,
        refl_im   = refl_im,
        diffusion = mat_props[:, 1],
        freq_hz   = freq_hz,
        speed_m_s = speed_m_s,
        atmo_abs  = atmo_abs,
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
