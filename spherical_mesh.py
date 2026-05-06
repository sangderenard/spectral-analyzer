"""Small spherical-coordinate mesh helpers for shader/rasterizer tests.

The utilities here keep parametric spherical work mechanical: generate a
spherical vertex distribution, attach equirectangular UVs, and expand indexed
triangles into the flat triangle payloads used by the basic C/GL tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SphereMesh:
    vertices: np.ndarray   # (N, 3) float32
    normals: np.ndarray    # (N, 3) float32
    uvs: np.ndarray        # (N, 2) float32, equirectangular
    indices: np.ndarray    # (T, 3) int32, outward orientation

    def triangles(self, center=(0.0, 0.0, 0.0), radius: float = 1.0) -> np.ndarray:
        c = np.asarray(center, dtype=np.float32).reshape(1, 1, 3)
        return (self.vertices[self.indices] * float(radius) + c).astype(np.float32)

    def flat_vertices(self, center=(0.0, 0.0, 0.0), radius: float = 1.0,
                      *, include_uv: bool = True) -> np.ndarray:
        tris = self.triangles(center=center, radius=radius).reshape(-1, 3)
        nrms = self.normals[self.indices].reshape(-1, 3).astype(np.float32)
        if not include_uv:
            return np.ascontiguousarray(np.concatenate([tris, nrms], axis=1), dtype=np.float32)
        uv = self.uvs[self.indices].reshape(-1, 2).astype(np.float32)
        return np.ascontiguousarray(np.concatenate([tris, nrms, uv], axis=1), dtype=np.float32)


def spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return unit XYZ from longitude theta and latitude phi, both radians."""
    cp = np.cos(phi)
    return np.stack([cp * np.cos(theta), np.sin(phi), cp * np.sin(theta)], axis=-1)


def cartesian_to_equirect_uv(xyz: np.ndarray) -> np.ndarray:
    p = np.asarray(xyz, dtype=np.float32)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    q = p / np.where(n > 1e-8, n, 1.0)
    theta = np.arctan2(q[..., 2], q[..., 0])
    phi = np.arcsin(np.clip(q[..., 1], -1.0, 1.0))
    u = (theta + math.pi) / (2.0 * math.pi)
    v = (phi + 0.5 * math.pi) / math.pi
    return np.stack([u, v], axis=-1).astype(np.float32)


def uv_sphere(longitudes: int = 64, latitudes: int = 32) -> SphereMesh:
    """Indexed UV sphere with duplicated seam vertices and outward triangles."""
    lon = max(3, int(longitudes))
    lat = max(2, int(latitudes))
    verts = []
    uvs = []
    for j in range(lat + 1):
        v = j / float(lat)
        phi = (v - 0.5) * math.pi
        for i in range(lon + 1):
            u = i / float(lon)
            theta = u * 2.0 * math.pi - math.pi
            verts.append(spherical_to_cartesian(np.array(theta), np.array(phi)))
            uvs.append((u, v))
    vertices = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    normals = vertices.copy()
    uv_arr = np.asarray(uvs, dtype=np.float32)
    idx = []
    row = lon + 1
    for j in range(lat):
        for i in range(lon):
            a = j * row + i
            b = a + 1
            c = (j + 1) * row + i
            d = c + 1
            if j > 0:
                idx.append((a, c, b))
            if j < lat - 1:
                idx.append((b, c, d))
    return SphereMesh(vertices, normals, uv_arr, np.asarray(idx, dtype=np.int32))


def equirect_texture(width: int, height: int,
                     fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Build an RGBA8 texture from fn(U, V) returning RGB or RGBA in [0, 1]."""
    w = max(1, int(width))
    h = max(1, int(height))
    u = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
    v = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
    U, V = np.meshgrid(u, v, indexing="xy")
    out = np.asarray(fn(U, V), dtype=np.float32)
    if out.shape[-1] == 3:
        alpha = np.ones((*out.shape[:2], 1), dtype=np.float32)
        out = np.concatenate([out, alpha], axis=-1)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
