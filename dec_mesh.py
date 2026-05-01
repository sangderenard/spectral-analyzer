"""dec_mesh.py
=============
Discrete Exterior Calculus mesh representation.

0-cells: vertices (positions in R³)
1-cells: edges (oriented vertex pairs; each undirected edge appears once)
2-cells: faces (ordered vertex index lists; any convex polygon)

All faces carry a triangle decomposition (fan from vertex 0) for GL rendering.

The boundary operators ∂₁ and ∂₂ are implicit in the edge / face adjacency
tables and computed lazily on demand.

Key operations
--------------
  DECMesh.from_raw(verts, faces)   — build from numpy arrays
  mesh.attach(other, self_face, other_face) — rigid attach face-to-face
  mesh.with_symmetry(mode, axis, count)    — replicate with symmetry
  mesh.gl_triangles()                      — (N,6) float32 pos+norm for GL
  mesh.gl_edges()                          — (E,2,3) float32 for wireframe
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rotation_from_to(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix mapping unit vector v_from → unit vector v_to."""
    v_from = _norm(np.asarray(v_from, np.float64))
    v_to   = _norm(np.asarray(v_to,   np.float64))
    axis   = np.cross(v_from, v_to)
    s      = np.linalg.norm(axis)
    c      = np.dot(v_from, v_to)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        # Anti-parallel: 180° rotation around any perp axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax   = _norm(np.cross(v_from, perp))
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    ax = axis / s
    K  = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    return np.eye(3) + K * s + K @ K * (1.0 - c)


def _rotation_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    R = np.eye(3)
    R[0, 0] =  c;  R[0, 1] = -s
    R[1, 0] =  s;  R[1, 1] =  c
    return R


def _rotation_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = _norm(np.asarray(axis, np.float64))
    c  = math.cos(angle_rad)
    s  = math.sin(angle_rad)
    K  = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    return np.eye(3) + K * s + K @ K * (1.0 - c)


# ─────────────────────────────────────────────────────────────────────────────
# DECMesh
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DECMesh:
    """Discrete Exterior Calculus mesh.

    verts     : (V, 3)  float64  — vertex positions
    edges     : (E, 2)  int32   — oriented vertex pairs (each edge once)
    faces     : list of list[int]  — face vertex index sequences (F faces)
    tris      : (T, 3)  int32   — triangle decomposition of all faces
    tri_face  : (T,)    int32   — which face each triangle belongs to
    """
    verts:    np.ndarray
    edges:    np.ndarray
    faces:    list
    tris:     np.ndarray
    tri_face: np.ndarray

    # ── cached derived data ───────────────────────────────────────────────────
    _face_normals: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _face_centers: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, verts: np.ndarray, faces: list) -> "DECMesh":
        """Build a DECMesh from vertex positions and face index lists."""
        verts = np.asarray(verts, np.float64)
        edges, tris, tri_face = _compute_edges_and_tris(verts, faces)
        return cls(verts=verts, edges=edges, faces=list(faces),
                   tris=tris, tri_face=tri_face)

    # ── geometry queries ─────────────────────────────────────────────────────

    def face_normal(self, i: int) -> np.ndarray:
        return self.face_normals()[i]

    def face_center(self, i: int) -> np.ndarray:
        return self.face_centers()[i]

    def face_normals(self) -> np.ndarray:
        if self._face_normals is None:
            self._face_normals = _compute_face_normals(self.verts, self.faces)
        return self._face_normals

    def face_centers(self) -> np.ndarray:
        if self._face_centers is None:
            self._face_centers = np.array(
                [self.verts[f].mean(axis=0) for f in self.faces])
        return self._face_centers

    # ── GL data ───────────────────────────────────────────────────────────────

    def gl_triangles(self, color: Optional[np.ndarray] = None) -> np.ndarray:
        """(T*3, 6) float32  pos_xyz + norm_xyz for GL_TRIANGLES.
        Each triangle gets a flat face normal."""
        normals = self.face_normals()
        rows = []
        for ti, tri in enumerate(self.tris):
            fi = int(self.tri_face[ti])
            n  = normals[fi].astype(np.float32)
            for vi in tri:
                p = self.verts[vi].astype(np.float32)
                rows.append((*p, *n))
        return np.array(rows, np.float32)

    def gl_edges(self) -> np.ndarray:
        """(E, 2, 3) float32  two endpoints per edge for GL_LINES."""
        out = np.empty((len(self.edges), 2, 3), np.float32)
        for i, (a, b) in enumerate(self.edges):
            out[i, 0] = self.verts[a]
            out[i, 1] = self.verts[b]
        return out

    def gl_face_highlight(self, face_idx: int) -> np.ndarray:
        """(T*3, 6) float32 triangles for one highlighted face (GL_TRIANGLES)."""
        normals = self.face_normals()
        n = normals[face_idx].astype(np.float32)
        rows = []
        for ti, tri in enumerate(self.tris):
            if int(self.tri_face[ti]) == face_idx:
                for vi in tri:
                    p = self.verts[vi].astype(np.float32)
                    rows.append((*p, *n))
        return np.array(rows, np.float32) if rows else np.zeros((0, 6), np.float32)

    # ── attachment ────────────────────────────────────────────────────────────

    def attach(self, other: "DECMesh", self_face: int, other_face: int,
               edge_align: bool = True) -> "DECMesh":
        """Rigidly attach `other` so its face `other_face` meets `self_face`.

        The two face centres are coincident and normals are anti-parallel.
        If edge_align=True, the pieces are also rotated around the shared normal
        to align their longest face edges.
        Returns a new DECMesh with both pieces merged.
        """
        nA = self.face_normal(self_face)
        cA = self.face_center(self_face)
        nB = other.face_normal(other_face)
        cB = other.face_center(other_face)

        # Rotation: map nB → -nA
        R = _rotation_from_to(nB, -nA)

        # Optional: align edge directions around the face normal
        if edge_align:
            fA = self.faces[self_face]
            fB = other.faces[other_face]
            eA = _norm(self.verts[fA[1]]  - self.verts[fA[0]])
            eB = _norm(other.verts[fB[1]] - other.verts[fB[0]])
            eB_rot = R @ eB
            cos_t  = np.clip(np.dot(eA, eB_rot), -1, 1)
            sin_t  = np.dot(np.cross(eB_rot, eA), -nA)
            extra  = _rotation_axis_angle(-nA, math.atan2(sin_t, cos_t))
            R      = extra @ R

        # Transform other.verts: centre at cB, apply R, place at cA
        moved = (R @ (other.verts - cB).T).T + cA

        # Merge
        V = len(self.verts)
        new_verts  = np.concatenate([self.verts, moved])
        new_edges  = np.concatenate([self.edges, other.edges + V])
        new_faces  = self.faces + [[v + V for v in f] for f in other.faces]
        new_tris   = np.concatenate([self.tris, other.tris + V])
        new_tf     = np.concatenate([self.tri_face,
                                     other.tri_face + len(self.faces)])
        return DECMesh(verts=new_verts, edges=new_edges,
                       faces=new_faces, tris=new_tris, tri_face=new_tf)

    def with_symmetry(self, mode: str, axis: str, count: int) -> "DECMesh":
        """Return a new mesh with `count` symmetry copies.

        mode : 'radial' — rotational copies around axis
               'bilateral' — one mirror across a plane containing axis
        axis : 'x' | 'y' | 'z'
        count: number of rotational copies (used for radial only)
        """
        ax_vec = {'x': np.array([1.,0.,0.]), 'y': np.array([0.,1.,0.]),
                  'z': np.array([0.,0.,1.])}.get(axis, np.array([0.,0.,1.]))

        if mode == 'radial':
            copies = [self]
            for k in range(1, max(2, count)):
                angle = 2 * math.pi * k / count
                R = _rotation_axis_angle(ax_vec, angle)
                rotated_v = (R @ self.verts.T).T
                V = sum(len(m.verts) for m in copies)
                F = sum(len(m.faces) for m in copies)
                copies.append(DECMesh(
                    verts    = rotated_v,
                    edges    = self.edges.copy(),
                    faces    = [list(f) for f in self.faces],
                    tris     = self.tris.copy(),
                    tri_face = self.tri_face.copy()))
            return _merge_many(copies)

        if mode == 'bilateral':
            # Mirror across the plane perpendicular to ax_vec
            mirror = np.eye(3) - 2 * np.outer(ax_vec, ax_vec)
            mv     = (mirror @ self.verts.T).T
            V      = len(self.verts)
            mirrored = DECMesh(
                verts    = mv,
                edges    = self.edges.copy(),
                faces    = [list(reversed(f)) for f in self.faces],  # flip winding
                tris     = self.tris[:, ::-1].copy(),
                tri_face = self.tri_face.copy())
            return _merge_many([self, mirrored])

        return self   # mode == 'none'

    def transform(self, R: np.ndarray, t: np.ndarray = None) -> "DECMesh":
        """Apply rotation R (3×3) and optional translation t (3,)."""
        new_v = (R @ self.verts.T).T
        if t is not None:
            new_v = new_v + t
        return DECMesh(verts=new_v, edges=self.edges.copy(),
                       faces=[list(f) for f in self.faces],
                       tris=self.tris.copy(), tri_face=self.tri_face.copy())

    def centred(self) -> "DECMesh":
        return self.transform(np.eye(3), -self.verts.mean(axis=0))

    def scaled(self, s: float) -> "DECMesh":
        return DECMesh(verts=self.verts * s, edges=self.edges.copy(),
                       faces=[list(f) for f in self.faces],
                       tris=self.tris.copy(), tri_face=self.tri_face.copy())

    # ── ray intersection (for face picking) ───────────────────────────────────

    def ray_intersect_face(self, ray_o: np.ndarray, ray_d: np.ndarray
                           ) -> tuple[int, float]:
        """Return (face_index, t) for nearest hit, or (-1, inf)."""
        best_fi, best_t = -1, float('inf')
        for ti, tri in enumerate(self.tris):
            t = _ray_tri(ray_o, ray_d,
                         self.verts[tri[0]], self.verts[tri[1]], self.verts[tri[2]])
            if 0 < t < best_t:
                best_t = t
                best_fi = int(self.tri_face[ti])
        return best_fi, best_t


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_edges_and_tris(verts: np.ndarray, faces: list
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_set: dict[tuple, int] = {}
    tris_list, tri_face_list = [], []

    for fi, face in enumerate(faces):
        # Triangle fan from vertex 0
        for k in range(1, len(face) - 1):
            tris_list.append([face[0], face[k], face[k + 1]])
            tri_face_list.append(fi)
        # Collect edges (unordered pairs → store as sorted tuple)
        for k in range(len(face)):
            a, b = face[k], face[(k + 1) % len(face)]
            key = (min(a, b), max(a, b))
            edge_set.setdefault(key, len(edge_set))

    edges = np.array(list(edge_set.keys()), np.int32) if edge_set else np.zeros((0, 2), np.int32)
    tris  = np.array(tris_list, np.int32)    if tris_list else np.zeros((0, 3), np.int32)
    tf    = np.array(tri_face_list, np.int32) if tri_face_list else np.zeros(0, np.int32)
    return edges, tris, tf


def _compute_face_normals(verts: np.ndarray, faces: list) -> np.ndarray:
    normals = np.zeros((len(faces), 3), np.float64)
    for i, face in enumerate(faces):
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        nn = np.linalg.norm(n)
        normals[i] = n / nn if nn > 1e-12 else n
    return normals


def _fix_winding(verts: np.ndarray, faces: list) -> list:
    """Ensure all face normals point away from the vertex centroid."""
    centroid = verts.mean(axis=0)
    fixed = []
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n   = np.cross(v1 - v0, v2 - v0)
        ctr = verts[np.array(face)].mean(axis=0)
        if np.dot(n, ctr - centroid) < 0:
            fixed.append(list(reversed(face)))
        else:
            fixed.append(list(face))
    return fixed


def _merge_many(meshes: list) -> DECMesh:
    """Combine a list of DECMesh objects into one (no vertex welding)."""
    if len(meshes) == 1:
        return meshes[0]
    acc = meshes[0]
    for m in meshes[1:]:
        V = len(acc.verts)
        F = len(acc.faces)
        acc = DECMesh(
            verts    = np.concatenate([acc.verts, m.verts]),
            edges    = np.concatenate([acc.edges, m.edges + V]),
            faces    = acc.faces + [[v + V for v in f] for f in m.faces],
            tris     = np.concatenate([acc.tris, m.tris + V]),
            tri_face = np.concatenate([acc.tri_face, m.tri_face + F]))
    return acc


def _ray_tri(o, d, v0, v1, v2) -> float:
    """Möller–Trumbore ray-triangle intersection.  Returns t > 0 or inf."""
    eps  = 1e-9
    e1   = v1 - v0
    e2   = v2 - v0
    h    = np.cross(d, e2)
    a    = np.dot(e1, h)
    if abs(a) < eps:
        return float('inf')
    f = 1.0 / a
    s = o - v0
    u = f * np.dot(s, h)
    if not (0.0 <= u <= 1.0):
        return float('inf')
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return float('inf')
    t = f * np.dot(e2, q)
    return t if t > eps else float('inf')
