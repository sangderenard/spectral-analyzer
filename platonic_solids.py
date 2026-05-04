"""platonic_solids.py
====================
Factory functions returning DECMesh instances for the five platonic solids
and a handful of derived shapes useful in the fabricator palette.

All meshes are centred at the origin and scaled to circumradius ≈ 1.0
(i.e. every vertex is at distance ≈ 1 from the origin).

Available
---------
  tetrahedron()
  cube()
  octahedron()
  icosahedron()
  dodecahedron()      (computed as dual of icosahedron)
  triangular_prism()
  square_pyramid()
  triangular_bipyramid()
    wedge()
"""
from __future__ import annotations

import math
from itertools import product as _iproduct

import numpy as np

from dec_mesh import DECMesh, _fix_winding


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_circumradius(verts: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(verts, axis=1).max()
    return verts / r if r > 1e-12 else verts


def _build(verts: np.ndarray, faces: list) -> DECMesh:
    verts  = np.asarray(verts, np.float64)
    verts  = _normalise_circumradius(verts)
    faces  = _fix_winding(verts, faces)
    return DECMesh.from_raw(verts, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Platonic solids
# ─────────────────────────────────────────────────────────────────────────────

def tetrahedron() -> DECMesh:
    """Regular tetrahedron.  4 triangular faces."""
    verts = np.array([
        ( 1,  1,  1),
        ( 1, -1, -1),
        (-1,  1, -1),
        (-1, -1,  1),
    ], np.float64)
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ]
    return _build(verts, faces)


def cube() -> DECMesh:
    """Regular cube (hexahedron).  6 square faces."""
    # Vertices: all ±1 combinations
    verts = np.array(list(_iproduct([-1, 1], repeat=3)), np.float64)
    # Index by bit pattern: 0=(−,−,−) ... 7=(+,+,+)
    # face[i] lists verts with that axis fixed
    faces = [
        [0, 2, 3, 1],   # z = −1  (bottom)
        [4, 5, 7, 6],   # z = +1  (top)
        [0, 1, 5, 4],   # y = −1  (front)
        [2, 6, 7, 3],   # y = +1  (back)
        [0, 4, 6, 2],   # x = −1  (left)
        [1, 3, 7, 5],   # x = +1  (right)
    ]
    return _build(verts, faces)


def octahedron() -> DECMesh:
    """Regular octahedron.  8 triangular faces."""
    verts = np.array([
        ( 1,  0,  0),   # 0 +X
        (-1,  0,  0),   # 1 −X
        ( 0,  1,  0),   # 2 +Y
        ( 0, -1,  0),   # 3 −Y
        ( 0,  0,  1),   # 4 +Z
        ( 0,  0, -1),   # 5 −Z
    ], np.float64)
    faces = [
        [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],   # top cap
        [5, 2, 0], [5, 1, 2], [5, 0, 3], [5, 3, 1],   # bottom cap
    ]
    return _build(verts, faces)


def icosahedron() -> DECMesh:
    """Regular icosahedron.  20 triangular faces."""
    φ = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        ( 0, -1, -φ), ( 0,  1, -φ), ( 0, -1,  φ), ( 0,  1,  φ),
        (-1, -φ,  0), ( 1, -φ,  0), (-1,  φ,  0), ( 1,  φ,  0),
        (-φ,  0, -1), ( φ,  0, -1), (-φ,  0,  1), ( φ,  0,  1),
    ], np.float64)
    # 20 faces — winding fixed by _fix_winding
    faces = [
        [0, 1, 9], [0, 9, 5], [0, 5, 4], [0, 4, 8], [0, 8, 1],
        [1, 6, 9], [9, 6, 7], [9, 7,11], [9,11, 5], [5,11, 4],
        [4,11,10], [4,10, 8], [8,10, 6], [8, 6, 1], [1, 7, 6],
        [2, 3,11], [2,11,10], [2,10, 3], [3,10, 6], [3, 6, 7],
    ]
    return _build(verts, faces)


def dodecahedron() -> DECMesh:
    """Regular dodecahedron.  12 pentagonal faces.
    Computed as the dual of the icosahedron: each icosahedron vertex
    becomes a pentagonal face whose corners are the incentres of the
    adjacent icosahedron faces.
    """
    ico = icosahedron()
    raw_verts = ico.verts        # (12,3) — one per icos vertex
    raw_faces = ico.faces        # 20 triangular faces

    # Build adjacency: for each ico vertex, which faces contain it?
    V_adj: list[list[int]] = [[] for _ in range(len(raw_verts))]
    for fi, face in enumerate(raw_faces):
        for vi in face:
            V_adj[vi].append(fi)

    # Dodecahedron vertices = icosahedron face centres (normalised)
    dod_verts = np.array([raw_verts[f].mean(axis=0) for f in raw_faces], np.float64)
    dod_verts = dod_verts / np.linalg.norm(dod_verts, axis=1, keepdims=True)

    # Dodecahedron faces = pentagon of face-centre indices around each ico vertex
    dod_faces = []
    for vi, adj_faces in enumerate(V_adj):
        if len(adj_faces) != 5:
            continue   # shouldn't happen for icosahedron
        # Sort adjacent face indices by angle around the ico vertex
        centre = raw_verts[vi]
        up     = _perpendicular(centre)
        right  = np.cross(up, centre)
        angles = [math.atan2(np.dot(dod_verts[fi] - centre, right),
                             np.dot(dod_verts[fi] - centre, up))
                  for fi in adj_faces]
        ordered = [adj_faces[k] for k in sorted(range(5), key=lambda k: angles[k])]
        dod_faces.append(ordered)

    return _build(dod_verts, dod_faces)


def _perpendicular(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to v."""
    v = v / np.linalg.norm(v)
    perp = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return np.cross(v, perp) / np.linalg.norm(np.cross(v, perp))


# ─────────────────────────────────────────────────────────────────────────────
# Derived shapes
# ─────────────────────────────────────────────────────────────────────────────

def triangular_prism() -> DECMesh:
    """Triangular prism: 2 triangular caps + 3 rectangular sides (5 faces)."""
    r = 1.0
    h = 0.8
    angles = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    bot = [(r * math.cos(a), r * math.sin(a), -h / 2) for a in angles]
    top = [(r * math.cos(a), r * math.sin(a),  h / 2) for a in angles]
    verts = np.array(bot + top, np.float64)
    # b0,b1,b2 = 0,1,2;  t0,t1,t2 = 3,4,5
    faces = [
        [0, 2, 1],           # bottom cap (CCW looking down)
        [3, 4, 5],           # top cap
        [0, 1, 4, 3],        # side 0-1
        [1, 2, 5, 4],        # side 1-2
        [2, 0, 3, 5],        # side 2-0
    ]
    return _build(verts, faces)


def square_pyramid() -> DECMesh:
    """Square pyramid: 1 square base + 4 triangular faces (5 faces)."""
    s = 1.0
    h = 1.2
    verts = np.array([
        (-s, -s, 0), ( s, -s, 0),
        ( s,  s, 0), (-s,  s, 0),
        ( 0,  0, h),
    ], np.float64)
    faces = [
        [0, 3, 2, 1],   # base (facing −Z)
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ]
    return _build(verts, faces)


def triangular_bipyramid() -> DECMesh:
    """Triangular bipyramid: 6 triangular faces (5 vertices)."""
    r = 1.0
    h = 1.0
    angles = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    eq  = [(r * math.cos(a), r * math.sin(a), 0.0) for a in angles]
    verts = np.array(eq + [(0, 0, h), (0, 0, -h)], np.float64)
    # equatorial: 0,1,2;  top: 3;  bottom: 4
    faces = [
        [0, 1, 3], [1, 2, 3], [2, 0, 3],   # upper
        [1, 0, 4], [2, 1, 4], [0, 2, 4],   # lower
    ]
    return _build(verts, faces)


def wedge() -> DECMesh:
    """Right wedge (triangular prism) with rectangular footprint and sloped top."""
    x0, x1 = -1.0, 1.0
    y0, y1 = -1.0, 1.0
    z0 = -1.0
    z_top_front = -0.2
    z_top_back = 1.0
    verts = np.array([
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z_top_front),
        (x1, y0, z_top_front),
        (x1, y1, z_top_back),
        (x0, y1, z_top_back),
    ], np.float64)
    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return _build(verts, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, callable] = {
    "tetrahedron":        tetrahedron,
    "cube":               cube,
    "octahedron":         octahedron,
    "icosahedron":        icosahedron,
    "dodecahedron":       dodecahedron,
    "triangular_prism":   triangular_prism,
    "square_pyramid":     square_pyramid,
    "triangular_bipyramid": triangular_bipyramid,
    "wedge":              wedge,
}


def get(solid_id: str) -> DECMesh:
    """Return a DECMesh by palette ID string."""
    fn = _REGISTRY.get(solid_id)
    if fn is None:
        raise KeyError(f"Unknown solid: {solid_id!r}.  "
                       f"Available: {list(_REGISTRY)}")
    return fn()


def available() -> list[str]:
    return list(_REGISTRY)
