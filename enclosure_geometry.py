"""enclosure_geometry.py
========================
Geometry builders for all glass-enclosure shapes used by the room system.

All builders return **float32 (-1, 6)** arrays with interleaved
``[x, y, z, nx, ny, nz]`` suitable for the Phong VAO layout::

    [(attrib_loc=0, n_components=3, stride=24, offset=0),
     (attrib_loc=1, n_components=3, stride=24, offset=12)]

Wireframe builders return **float32 (-1, 3)** ``[x, y, z]`` pairs
(GL_LINES: each consecutive pair of rows is one line segment).

Shape catalogue
---------------
* ``build_rectangular_jar``  — 4 flat walls + bevel corners + top cap
                               (same algorithm as glass_room.py original)
* ``build_cylindrical_jar``  — open-bottom cylindrical shell + top annular cap
* ``build_spherical_jar``    — full UV-sphere shell, smooth outward normals
* ``build_cylindrical_pedestal`` — solid cylinder base (for sphere / cyl jars)
* ``build_glass_tablet_rect``    — two parallel flat rectangular panes
* ``build_glass_tablet_polar``   — two parallel circular disk panes
* ``build_enclosure_wireframe``  — dispatcher returning lines for any shape
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_π = math.pi
_2π = 2.0 * _π


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quad(p00, p01, p10, p11) -> list:
    """Two triangles from four [x,y,z,nx,ny,nz] vertices (CCW winding)."""
    return [p00, p10, p11, p00, p11, p01]


def _norm3(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _vn(pos, n) -> list:
    """Pack position + normal into a 6-element list."""
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    nx, ny, nz = float(n[0]),   float(n[1]),   float(n[2])
    return [px, py, pz, nx, ny, nz]


def _pack(rows: list) -> np.ndarray:
    return np.array(rows, np.float32).reshape(-1, 6)


def _pack3(rows: list) -> np.ndarray:
    return np.array(rows, np.float32).reshape(-1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Rectangular bell-jar
#     (re-implementation of the original glass_room.py logic as a clean function)
# ─────────────────────────────────────────────────────────────────────────────

def build_rectangular_jar(
    wb_min: Sequence,
    wb_max: Sequence,
    thickness: float = 0.020,
    bevel_segs: int = 6,
) -> np.ndarray:
    """Rectangular glass bell-jar: 4 flat walls + rounded vertical edges + top cap.

    Parameters
    ----------
    wb_min, wb_max : sequence of 3 floats
        World-space AABB of the *interior* volume.
    thickness : float
        Wall / bevel radius in metres.
    bevel_segs : int
        Number of arc segments per vertical corner bevel.

    Returns
    -------
    float32 array, shape (-1, 6)  [x,y,z,nx,ny,nz]
    """
    t  = float(thickness)
    r  = t
    xi, xa = float(wb_min[0]), float(wb_max[0])   # X: left/right interior
    yi, ya = float(wb_min[1]), float(wb_max[1])   # Y: depth interior
    zi, za = float(wb_min[2]), float(wb_max[2])   # Z: height interior
    xo0, xo1 = xi - t, xa + t   # outer X bounds
    yo0, yo1 = yi - t, ya + t   # outer Y (depth) bounds
    zo0, zo1 = zi - t, za + t   # outer Z (height) bounds

    rows: list = []

    # ── flat X-walls (left/right): constant X, spanning Y (depth) and Z ──────
    for cx_wall, nx in [(xo0, -1.0), (xo1, 1.0)]:
        rows += _quad(
            _vn([cx_wall, yi + r, zo0], [nx, 0, 0]),
            _vn([cx_wall, ya - r, zo0], [nx, 0, 0]),
            _vn([cx_wall, yi + r, zo1], [nx, 0, 0]),
            _vn([cx_wall, ya - r, zo1], [nx, 0, 0]),
        )
    # ── flat Y-walls (front/back): constant Y, spanning X and Z ─────────────
    for cy_wall, ny in [(yo0, -1.0), (yo1, 1.0)]:
        rows += _quad(
            _vn([xi + r, cy_wall, zo0], [0, ny, 0]),
            _vn([xa - r, cy_wall, zo0], [0, ny, 0]),
            _vn([xi + r, cy_wall, zo1], [0, ny, 0]),
            _vn([xa - r, cy_wall, zo1], [0, ny, 0]),
        )

    # ── vertical corner bevels (arc in XY plane, spanning Z) ─────────────────
    corners = [
        (xi, yi, (-1, -1)),
        (xa, yi, ( 1, -1)),
        (xa, ya, ( 1,  1)),
        (xi, ya, (-1,  1)),
    ]
    for (cx, cy_c, (sx, sy)) in corners:
        a_start = math.atan2(sy, sx) + _π / 2
        for k in range(bevel_segs):
            a0 = a_start + k       * _π / (2 * bevel_segs)
            a1 = a_start + (k + 1) * _π / (2 * bevel_segs)
            nx0, ny0 = math.cos(a0), math.sin(a0)
            nx1, ny1 = math.cos(a1), math.sin(a1)
            bx0, by0 = cx + r * nx0, cy_c + r * ny0
            bx1, by1 = cx + r * nx1, cy_c + r * ny1
            rows += _quad(
                _vn([bx0, by0, zo0], [nx0, ny0, 0]),
                _vn([bx1, by1, zo0], [nx1, ny1, 0]),
                _vn([bx0, by0, zo1], [nx0, ny0, 0]),
                _vn([bx1, by1, zo1], [nx1, ny1, 0]),
            )

    # ── top cap (flat at z = zo1, normal = (0,0,+1)) ─────────────────────────
    def top_strip(x0, x1, y0, y1):
        rows.extend(_quad(
            _vn([x0, y0, zo1], [0, 0, 1]),
            _vn([x1, y0, zo1], [0, 0, 1]),
            _vn([x0, y1, zo1], [0, 0, 1]),
            _vn([x1, y1, zo1], [0, 0, 1]),
        ))

    # Centre rectangle
    top_strip(xi + r, xa - r, yi - r, ya + r)
    # Side strips (X extend)
    top_strip(xo0, xi + r, yi - r, ya + r)
    top_strip(xa - r, xo1, yi - r, ya + r)
    # Front/back strips (Y extend)
    top_strip(xi + r, xa - r, yi - r - t, yi - r)
    top_strip(xi + r, xa - r, ya + r, ya + r + t)
    # Corner arcs on top cap
    for (cx, cy_c, (sx, sy)) in corners:
        a_start = math.atan2(sy, sx) + _π / 2
        for k in range(bevel_segs):
            a0 = a_start + k       * _π / (2 * bevel_segs)
            a1 = a_start + (k + 1) * _π / (2 * bevel_segs)
            p_c  = _vn([cx,                    cy_c,                    zo1], [0, 0, 1])
            p_a0 = _vn([cx + r * math.cos(a0), cy_c + r * math.sin(a0), zo1], [0, 0, 1])
            p_a1 = _vn([cx + r * math.cos(a1), cy_c + r * math.sin(a1), zo1], [0, 0, 1])
            rows += [p_c, p_a0, p_a1]

    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cylindrical jar
# ─────────────────────────────────────────────────────────────────────────────

def build_cylindrical_jar(
    center: Sequence,
    radius: float,
    height: float,
    thickness: float = 0.018,
    lon_segs: int = 28,
) -> np.ndarray:
    """Open-bottom cylindrical glass shell with top annular cap.

    The cylinder's axis is +Z.  The base opening sits at ``center[2]``;
    the top cap is at ``center[2] + height``.

    Parameters
    ----------
    center : (cx, cy, cz)
        Centre of the bottom opening circle.
    radius : float
        Outer radius of the glass cylinder.
    height : float
        Interior height.
    thickness : float
        Nominal glass-wall thickness; used for the annular cap width.
    lon_segs : int
        Number of longitudinal divisions around the circumference.

    Returns
    -------
    float32 array, shape (-1, 6)
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r, h, t = float(radius), float(height), float(thickness)
    rows: list = []

    # Side surface — outer cylinder, outward radial normals (in XY plane)
    for i in range(lon_segs):
        a0 = _2π * i       / lon_segs
        a1 = _2π * (i + 1) / lon_segs
        n0x, n0y = math.cos(a0), math.sin(a0)
        n1x, n1y = math.cos(a1), math.sin(a1)
        rows += _quad(
            _vn([cx + r * n0x, cy + r * n0y, cz    ], [n0x, n0y, 0]),
            _vn([cx + r * n1x, cy + r * n1y, cz    ], [n1x, n1y, 0]),
            _vn([cx + r * n0x, cy + r * n0y, cz + h], [n0x, n0y, 0]),
            _vn([cx + r * n1x, cy + r * n1y, cz + h], [n1x, n1y, 0]),
        )

    # Top annular cap — horizontal ring, normal = +Z
    r_in  = max(0.0, r - t)
    r_out = r + t
    top_z = cz + h
    for i in range(lon_segs):
        a0 = _2π * i       / lon_segs
        a1 = _2π * (i + 1) / lon_segs
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        rows += _quad(
            _vn([cx + r_out * c0, cy + r_out * s0, top_z], [0, 0, 1]),
            _vn([cx + r_out * c1, cy + r_out * s1, top_z], [0, 0, 1]),
            _vn([cx + r_in  * c0, cy + r_in  * s0, top_z], [0, 0, 1]),
            _vn([cx + r_in  * c1, cy + r_in  * s1, top_z], [0, 0, 1]),
        )

    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Spherical jar
# ─────────────────────────────────────────────────────────────────────────────

def build_spherical_jar(
    center: Sequence,
    radius: float,
    thickness: float = 0.020,
    lat_segs: int = 18,
    lon_segs: int = 28,
) -> np.ndarray:
    """Full UV-sphere glass shell with smooth outward normals.

    Parameters
    ----------
    center : (cx, cy, cz)
        Geometric centre of the sphere.
    radius : float
        Outer radius.
    lat_segs, lon_segs : int
        Latitude and longitude division counts.

    Returns
    -------
    float32 array, shape (-1, 6)
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = float(radius)
    rows: list = []

    def vtx(th: float, ph: float) -> list:
        # Z-up: polar axis is +Z (cos(th) on Z)
        nx = math.sin(th) * math.cos(ph)
        ny = math.sin(th) * math.sin(ph)
        nz = math.cos(th)
        return [cx + r * nx, cy + r * ny, cz + r * nz, nx, ny, nz]

    for lat in range(lat_segs):
        th0 = _π * lat       / lat_segs
        th1 = _π * (lat + 1) / lat_segs
        for lon in range(lon_segs):
            ph0 = _2π * lon       / lon_segs
            ph1 = _2π * (lon + 1) / lon_segs
            v00 = vtx(th0, ph0)
            v01 = vtx(th0, ph1)
            v10 = vtx(th1, ph0)
            v11 = vtx(th1, ph1)
            if lat == 0:
                rows += [v00, v10, v11]
            elif lat == lat_segs - 1:
                rows += [v10, v01, v00]
            else:
                rows += [v00, v10, v11, v00, v11, v01]

    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Cylindrical pedestal  (solid)
# ─────────────────────────────────────────────────────────────────────────────

def build_cylindrical_pedestal(
    center: Sequence,
    radius: float,
    height: float,
    lon_segs: int = 24,
) -> np.ndarray:
    """Solid cylinder used as a pedestal base for sphere / cylinder jars.

    The cylinder runs from ``center[2]`` (bottom) to ``center[2]+height`` (top).

    Returns
    -------
    float32 array, shape (-1, 6)
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r, h = float(radius), float(height)
    rows: list = []

    for i in range(lon_segs):
        a0 = _2π * i       / lon_segs
        a1 = _2π * (i + 1) / lon_segs
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)

        # Side quad — outward normal in XY plane
        am = (a0 + a1) * 0.5
        nm = (math.cos(am), math.sin(am), 0.0)
        rows += _quad(
            _vn([cx + r * c0, cy + r * s0, cz    ], nm),
            _vn([cx + r * c1, cy + r * s1, cz    ], nm),
            _vn([cx + r * c0, cy + r * s0, cz + h], nm),
            _vn([cx + r * c1, cy + r * s1, cz + h], nm),
        )

        # Bottom fan triangle — normal = −Z
        rows += [
            _vn([cx,           cy,           cz], [0, 0, -1]),
            _vn([cx + r * c1,  cy + r * s1,  cz], [0, 0, -1]),
            _vn([cx + r * c0,  cy + r * s0,  cz], [0, 0, -1]),
        ]

        # Top fan triangle — normal = +Z
        rows += [
            _vn([cx,           cy,           cz + h], [0, 0, 1]),
            _vn([cx + r * c0,  cy + r * s0,  cz + h], [0, 0, 1]),
            _vn([cx + r * c1,  cy + r * s1,  cz + h], [0, 0, 1]),
        ]

    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Rectangular glass tablet  (two parallel flat panes)
# ─────────────────────────────────────────────────────────────────────────────

def build_glass_tablet_rect(
    center: Sequence,
    normal: Sequence,
    right:  Sequence,
    width:  float,
    height: float,
    gap:    float   = 0.012,
    thickness: float = 0.006,
) -> np.ndarray:
    """Two parallel flat rectangular glass panes separated by *gap*.

    The panes are camera-facing at construction time: *normal* points toward
    the viewer.  Both faces of each pane are rendered (for glass transparency).

    Parameters
    ----------
    center  : world-space midpoint between the two panes
    normal  : unit vector pointing from back pane to front pane (viewer-facing)
    right   : unit vector pointing to the right in the pane's plane
    width, height : pane dimensions in metres
    gap     : separation between the two inner surfaces
    thickness : glass-plate thickness (each plate)

    Returns
    -------
    float32 array, shape (-1, 6)
    """
    n  = _norm3(normal)
    rr = _norm3(right)
    up = _norm3(np.cross(rr, n))   # right × normal = up  (right-hand rule)

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    hw, hh = width * 0.5, height * 0.5
    # Offsets of the two plate centres along the normal
    d_front = gap * 0.5 + thickness * 0.5
    d_back  = -(gap * 0.5 + thickness * 0.5)
    rows: list = []

    for d, face_n_sign in [(d_front, 1.0), (d_back, -1.0)]:
        pc = np.array([cx, cy, cz]) + d * n
        fn = n * face_n_sign

        # Four corners of this plate
        bl = pc - hw * rr - hh * up
        br = pc + hw * rr - hh * up
        tl = pc - hw * rr + hh * up
        tr = pc + hw * rr + hh * up

        # Front face (facing viewer)
        rows += _quad(
            _vn(bl, fn), _vn(br, fn),
            _vn(tl, fn), _vn(tr, fn),
        )
        # Back face (facing away from viewer)
        bn = -fn
        rows += _quad(
            _vn(br, bn), _vn(bl, bn),
            _vn(tr, bn), _vn(tl, bn),
        )

    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Polar glass tablet  (two parallel circular disk panes)
# ─────────────────────────────────────────────────────────────────────────────

def build_glass_tablet_polar(
    center:    Sequence,
    normal:    Sequence,
    right:     Sequence,
    radius:    float,
    gap:       float   = 0.010,
    thickness: float   = 0.005,
    segs:      int     = 32,
) -> np.ndarray:
    """Two parallel circular glass disk panes separated by *gap*.

    Each disk is tessellated as a centre-point fan.

    Returns
    -------
    float32 array, shape (-1, 6)
    """
    n  = _norm3(normal)
    rr = _norm3(right)
    up = _norm3(np.cross(rr, n))

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = float(radius)
    d_front = gap * 0.5 + thickness * 0.5
    d_back  = -(gap * 0.5 + thickness * 0.5)
    rows: list = []

    for d, face_n_sign in [(d_front, 1.0), (d_back, -1.0)]:
        pc = np.array([cx, cy, cz]) + d * n
        fn = n * face_n_sign
        bn = -fn

        ctr = _vn(pc, fn)
        ring = []
        for k in range(segs + 1):
            a = _2π * k / segs
            edge = pc + r * (math.cos(a) * rr + math.sin(a) * up)
            ring.append(_vn(edge, fn))

        # Front-face fan
        for k in range(segs):
            rows += [ctr, ring[k], ring[k + 1]]

        # Back-face fan (reversed winding)
        ctr_b = _vn(pc, bn)
        ring_b = [_vn(v[:3], bn) for v in ring]
        for k in range(segs):
            rows += [ctr_b, ring_b[k + 1], ring_b[k]]

    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Wireframe helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_wireframe_box(wb_min: Sequence, wb_max: Sequence) -> np.ndarray:
    """12 GL_LINES edges of an AABB.  Returns float32 (-1, 3) [x,y,z]."""
    xi, yi, zi = float(wb_min[0]), float(wb_min[1]), float(wb_min[2])
    xa, ya, za = float(wb_max[0]), float(wb_max[1]), float(wb_max[2])
    edges = [
        (xi,yi,zi),(xa,yi,zi), (xa,yi,zi),(xa,ya,zi),
        (xa,ya,zi),(xi,ya,zi), (xi,ya,zi),(xi,yi,zi),
        (xi,yi,za),(xa,yi,za), (xa,yi,za),(xa,ya,za),
        (xa,ya,za),(xi,ya,za), (xi,ya,za),(xi,yi,za),
        (xi,yi,zi),(xi,yi,za), (xa,yi,zi),(xa,yi,za),
        (xi,ya,zi),(xi,ya,za), (xa,ya,zi),(xa,ya,za),
    ]
    return _pack3(edges)


def build_wireframe_cylinder(
    center: Sequence, radius: float, height: float, lon_segs: int = 24
) -> np.ndarray:
    """Top and bottom circles + vertical struts.  Returns float32 (-1, 3)."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r, h = float(radius), float(height)
    edges: list = []
    for i in range(lon_segs):
        a0 = _2π * i       / lon_segs
        a1 = _2π * (i + 1) / lon_segs
        bx0, by0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        bx1, by1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        # bottom circle (at z=cz)
        edges += [[bx0, by0, cz], [bx1, by1, cz]]
        # top circle (at z=cz+h)
        edges += [[bx0, by0, cz + h], [bx1, by1, cz + h]]
        # vertical strut every 4th segment
        if i % 4 == 0:
            edges += [[bx0, by0, cz], [bx0, by0, cz + h]]
    return _pack3(edges)


def build_wireframe_sphere(
    center: Sequence, radius: float, segs: int = 24
) -> np.ndarray:
    """Three great circles (XY, YZ, XZ planes).  Returns float32 (-1, 3)."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = float(radius)
    edges: list = []
    for plane in range(3):
        for i in range(segs):
            a0 = _2π * i       / segs
            a1 = _2π * (i + 1) / segs
            if plane == 0:    # horizontal circle in XY
                p0 = [cx + r * math.cos(a0), cy + r * math.sin(a0), cz]
                p1 = [cx + r * math.cos(a1), cy + r * math.sin(a1), cz]
            elif plane == 1:  # vertical circle in XZ
                p0 = [cx + r * math.cos(a0), cy, cz + r * math.sin(a0)]
                p1 = [cx + r * math.cos(a1), cy, cz + r * math.sin(a1)]
            else:              # vertical circle in YZ
                p0 = [cx, cy + r * math.cos(a0), cz + r * math.sin(a0)]
                p1 = [cx, cy + r * math.cos(a1), cz + r * math.sin(a1)]
            edges += [p0, p1]
    return _pack3(edges)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  High-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_enclosure_mesh(obj) -> np.ndarray:
    """Build triangle mesh for a ``PlacedEnclosure`` data object.

    Parameters
    ----------
    obj : PlacedEnclosure
        Provides ``shape``, ``dims``, and ``pos`` / ``yaw_deg``.

    Returns
    -------
    float32 array, shape (-1, 6)  [x,y,z,nx,ny,nz]  in world space,
    or an empty array on unknown shape.
    """
    from placed_object import PlacedEnclosure  # local import to avoid circularity

    s  = obj.shape
    d  = obj.dims
    cx, cy, cz = float(obj.pos[0]), float(obj.pos[1]), float(obj.pos[2])

    if s == "rect":
        w  = float(d.get("width_m",  1.0))
        dp = float(d.get("depth_m",  1.0))
        h  = float(d.get("height_m", 1.2))
        t  = float(d.get("glass_thickness_m", 0.018))
        bs = int(  d.get("bevel_segs", 6))
        # Z-up: cx=X, cy=Y_depth centre, cz=Z_height base (floor level)
        wb_min = np.array([cx - w / 2, cy - dp / 2, cz    ])
        wb_max = np.array([cx + w / 2, cy + dp / 2, cz + h])
        return build_rectangular_jar(wb_min, wb_max, t, bs)

    if s == "cyl":
        r  = float(d.get("radius_m", 0.85))
        h  = float(d.get("height_m", 1.60))
        t  = float(d.get("glass_thickness_m", 0.018))
        ls = int(  d.get("lon_segs", 28))
        return build_cylindrical_jar([cx, cy, cz], r, h, t, ls)

    if s == "sphere":
        r  = float(d.get("radius_m", 1.10))
        t  = float(d.get("glass_thickness_m", 0.020))
        la = int(  d.get("lat_segs", 18))
        lo = int(  d.get("lon_segs", 28))
        pr = float(d.get("pedestal_radius_m", 0.30))
        ph = float(d.get("pedestal_height_m", 0.90))
        sphere_z = cz + ph + r   # Z-up: sphere centre Z above pedestal top
        mesh_s = build_spherical_jar([cx, cy, sphere_z], r, t, la, lo)
        mesh_p = build_cylindrical_pedestal([cx, cy, cz], pr, ph)
        return np.concatenate([mesh_s, mesh_p], axis=0)

    if s == "tablet_rect":
        w  = float(d.get("width_m",  1.80))
        h  = float(d.get("height_m", 1.00))
        gp = float(d.get("gap_m",    0.012))
        t  = float(d.get("glass_thickness_m", 0.006))
        yaw = math.radians(float(obj.yaw_deg))
        # normal faces +Y rotated by yaw around +Z
        normal = np.array([ math.sin(yaw), math.cos(yaw), 0.0])
        right  = np.array([ math.cos(yaw),-math.sin(yaw), 0.0])
        return build_glass_tablet_rect([cx, cy, cz], normal, right, w, h, gp, t)

    if s == "tablet_polar":
        r  = float(d.get("radius_m", 0.75))
        gp = float(d.get("gap_m",    0.010))
        t  = float(d.get("glass_thickness_m", 0.005))
        sg = int(  d.get("segs", 32))
        yaw = math.radians(float(obj.yaw_deg))
        normal = np.array([ math.sin(yaw), math.cos(yaw), 0.0])
        right  = np.array([ math.cos(yaw),-math.sin(yaw), 0.0])
        return build_glass_tablet_polar([cx, cy, cz], normal, right, r, gp, t, sg)

    # Fallback — empty array
    return np.zeros((0, 6), np.float32)


def build_enclosure_wireframe(obj) -> np.ndarray:
    """Build GL_LINES wireframe for a ``PlacedEnclosure``.

    Returns float32 (-1, 3) [x,y,z].
    """
    s = obj.shape
    d = obj.dims
    cx, cy, cz = float(obj.pos[0]), float(obj.pos[1]), float(obj.pos[2])

    if s == "rect":
        w  = float(d.get("width_m",  1.0))
        dp = float(d.get("depth_m",  1.0))
        h  = float(d.get("height_m", 1.2))
        t  = float(d.get("glass_thickness_m", 0.018))
        wb_min = np.array([cx - w/2, cy - dp/2, cz    ])
        wb_max = np.array([cx + w/2, cy + dp/2, cz + h])
        return build_wireframe_box(wb_min, wb_max)

    if s in ("cyl",):
        r = float(d.get("radius_m", 0.85))
        h = float(d.get("height_m", 1.60))
        ls = int(d.get("lon_segs", 28))
        return build_wireframe_cylinder([cx, cy, cz], r, h, min(ls, 24))

    if s == "sphere":
        r  = float(d.get("radius_m", 1.10))
        ph = float(d.get("pedestal_height_m", 0.90))
        sphere_z = cz + ph + r
        return build_wireframe_sphere([cx, cy, sphere_z], r)

    if s in ("tablet_rect", "tablet_polar"):
        # Simple box outline around the tablet volume
        # Z-up: X=width, Y=gap (depth), Z=height
        if s == "tablet_rect":
            w = float(d.get("width_m", 1.80))
            h = float(d.get("height_m", 1.00))
            gp = float(d.get("gap_m", 0.012))
            wb_min = np.array([cx - w/2, cy - gp, cz - h/2])
            wb_max = np.array([cx + w/2, cy + gp, cz + h/2])
        else:
            r  = float(d.get("radius_m", 0.75))
            gp = float(d.get("gap_m", 0.010))
            wb_min = np.array([cx - r, cy - gp, cz - r])
            wb_max = np.array([cx + r, cy + gp, cz + r])
        return build_wireframe_box(wb_min, wb_max)

    return np.zeros((0, 3), np.float32)
