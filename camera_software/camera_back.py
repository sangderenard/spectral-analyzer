"""camera_software/camera_back.py
---------------------------------
UV-parameterized camera back (sensor / film / plate) surfaces.

A CameraBack binds a physical sensor surface — flat, cylindrical,
hemispherical (retina), or anything else — to a rectangular pixel buffer.
Ray-cast results from an arbitrary ``LensManifold`` are projected onto the
surface geometry and splatted into the buffer by inverse UV lookup.

Class hierarchy
---------------
CameraBack          — abstract base; owns the pixel buffer and accumulator
  FlatBack          — standard flat rectangular sensor / film plane
  CylindricalBack   — curved panoramic film
  SphericalBack     — dome / hemispherical retinal surface
  ManifoldBack      — routes rays through a LensManifold then to any back

The UV coordinate space is [0, 1]² in both U and V, where (0.5, 0.5) is
the optical centre of the surface.  The pixel buffer is indexed as
(row=V*H, col=U*W).

Mesh generation
---------------
Every subclass implements ``mesh_verts_uvs(subdivisions)`` returning:
  positions  (V, 3) float64 — surface 3D positions  (sensor-local)
  uvs        (V, 2) float64 — per-vertex UV in [0, 1]²
  triangles  (T, 3) int32   — triangle indices

The mesh is suitable for:
  - Rendering the sensor geometry inside the scene
  - GPU texture blit (render the pixel buffer onto the physical surface)
  - CPU barycentric inverse-UV lookup

All arithmetic uses the native dtype of incoming data; no dtype coercion.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = ["CameraBack", "FlatBack", "CylindricalBack", "SphericalBack",
           "ManifoldBack", "LargeFormatPlateBack"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm3(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-15, n, 1.0)


def _sphere_grid(n_lat: int, n_lon: int,
                 radius: float,
                 z_offset: float = 0.0,
                 cap_fraction: float = 1.0) -> tuple:
    """Return (positions, normals, uvs, tris) for a sphere / spherical cap.

    cap_fraction=1.0 → full sphere; cap_fraction=0.5 → hemisphere.
    z_offset shifts the sphere centre along +Z.
    """
    positions, normals, uvs, tris = [], [], [], []
    for i in range(n_lat + 1):
        theta = math.pi * cap_fraction * (i / n_lat)   # polar angle
        for j in range(n_lon + 1):
            phi = 2.0 * math.pi * (j / n_lon)           # azimuth
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta) + z_offset
            positions.append([x, y, z])
            normals.append(_norm3(np.array([[x, y, z - z_offset]])).ravel().tolist())
            uvs.append([j / n_lon, i / n_lat])
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * (n_lon + 1) + j
            b = a + 1
            c = a + (n_lon + 1)
            d = c + 1
            tris += [[a, b, d], [a, d, c]]
    return (np.array(positions, np.float64),
            np.array(normals,   np.float64),
            np.array(uvs,       np.float64),
            np.array(tris,      np.int32))


def _disk_grid(n_rings: int, n_spokes: int, radius: float) -> tuple:
    """Return (positions, uvs, tris) for a flat disk."""
    positions, uvs, tris = [[0.0, 0.0, 0.0]], [[0.5, 0.5]], []
    for i in range(1, n_rings + 1):
        r = radius * (i / n_rings)
        for j in range(n_spokes):
            phi = 2.0 * math.pi * (j / n_spokes)
            x, y = r * math.cos(phi), r * math.sin(phi)
            positions.append([x, y, 0.0])
            uvs.append([(x / radius + 1.0) * 0.5, (y / radius + 1.0) * 0.5])
    for j in range(n_spokes):
        a = 0
        b = 1 + j
        c = 1 + (j + 1) % n_spokes
        tris.append([a, b, c])
    for i in range(n_rings - 1):
        for j in range(n_spokes):
            a = 1 + i * n_spokes + j
            b = 1 + i * n_spokes + (j + 1) % n_spokes
            c = 1 + (i + 1) * n_spokes + j
            d = 1 + (i + 1) * n_spokes + (j + 1) % n_spokes
            tris += [[a, b, d], [a, d, c]]
    return (np.array(positions, np.float64),
            np.array(uvs,       np.float64),
            np.array(tris,      np.int32))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CameraBack:
    """UV-parameterized sensor surface with pixel-buffer accumulation.

    Subclass and implement:
        ray_to_uv(ray_dirs)         — (N,3) directions → (N,2) UV in [0,1]²
        mesh_verts_uvs(subdivisions)— triangulated mesh for rendering / lookup
    """

    def __init__(self, res_w: int = 512, res_h: int = 512,
                 n_channels: int = 4) -> None:
        self.res_w = int(res_w)
        self.res_h = int(res_h)
        self.n_channels = int(n_channels)
        self._buf   = np.zeros((self.res_h, self.res_w, self.n_channels),
                               np.float64)
        self._count = np.zeros((self.res_h, self.res_w), np.float64)
        self._tex_handle: int = 0

    # ------------------------------------------------------------------
    # Interface

    def ray_to_uv(self, ray_dirs: np.ndarray) -> np.ndarray:
        """Map (N, 3) ray directions → (N, 2) UV in [0, 1]².  Override."""
        raise NotImplementedError

    def uv_to_position(self, uvs: np.ndarray) -> np.ndarray:
        """Batch back-cast: (N, 2) pixel UV → (N, 3) sensor-local position.

        Pure parametric evaluation — no mesh, no barycentric lookup, exact
        float64 arithmetic.  Override in each subclass.
        """
        raise NotImplementedError

    def uv_to_ray_dir(self, uvs: np.ndarray) -> np.ndarray:
        """Batch back-cast: (N, 2) pixel UV → (N, 3) unit ray direction.

        Returns the normalised direction from each surface position pointing
        toward the scene (the reverse of ray_to_uv).  Override in each
        subclass.
        """
        raise NotImplementedError

    def mesh_verts_uvs(self, subdivisions: int = 32) -> tuple:
        """Return (positions (V,3), uvs (V,2), triangles (T,3)).  Override."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Accumulation

    def accumulate(self, ray_dirs: np.ndarray,
                   values: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> None:
        """Splat per-ray *values* into the pixel buffer.

        Parameters
        ----------
        ray_dirs : (N, 3) float64 — ray directions at the sensor surface
        values   : (N, C) or (N,) — radiance / energy per ray
        weights  : (N,) or None   — per-ray weight (uniform 1 if None)
        """
        uvs = self.ray_to_uv(np.asarray(ray_dirs, np.float64))
        self._splat(uvs, values, weights)

    def _splat(self, uvs: np.ndarray,
               values: np.ndarray,
               weights: Optional[np.ndarray]) -> None:
        """Internal: given (N,2) UV and values, scatter-add into buffer."""
        px = np.clip((uvs[:, 0] * self.res_w).astype(np.intp),
                     0, self.res_w - 1)
        py = np.clip((uvs[:, 1] * self.res_h).astype(np.intp),
                     0, self.res_h - 1)
        vals = np.asarray(values, np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
        w = (np.asarray(weights, np.float64)
             if weights is not None
             else np.ones(len(uvs), np.float64))
        nc = min(self.n_channels, vals.shape[1])
        for c in range(nc):
            np.add.at(self._buf[:, :, c], (py, px), vals[:, c] * w)
        np.add.at(self._count, (py, px), w)

    def to_rect(self) -> np.ndarray:
        """Return (H, W, C) float64 mean-accumulated image."""
        denom = np.maximum(self._count[:, :, None], 1e-30)
        return self._buf / denom

    def vertex_colors_rgba(self, uvs: np.ndarray,
                           exposure_scale: float = 1.0) -> np.ndarray:
        """Return (N, 4) float32 per-vertex RGBA tint sampled from the buffer.

        RGB is the mean accumulated radiance at each UV position.  Alpha is
        the normalised accumulated weight at that pixel, scaled by
        *exposure_scale*.  Unexposed sub-pixels are fully transparent;
        well-exposed ones become opaque with the average arriving color —
        making the sensor mesh behave as a positive transparency film.
        """
        img = self.to_rect()                                    # (H, W, C) float64
        uv  = np.asarray(uvs, np.float64)
        px  = np.clip((uv[:, 0] * self.res_w  - 0.5).astype(np.intp),
                      0, self.res_w  - 1)
        py  = np.clip((uv[:, 1] * self.res_h - 0.5).astype(np.intp),
                      0, self.res_h - 1)
        rgba = np.zeros((len(uv), 4), np.float32)
        for c in range(min(self.n_channels, 3)):
            rgba[:, c] = img[py, px, c].astype(np.float32)
        max_cnt = float(np.max(self._count)) if np.any(self._count > 0) else 1.0
        rgba[:, 3] = np.clip(
            self._count[py, px].astype(np.float32)
            / max(max_cnt, 1e-9) * float(exposure_scale),
            0.0, 1.0)
        return rgba

    def clear(self) -> None:
        self._buf[:] = 0.0
        self._count[:] = 0.0
        self._tex_handle = 0

    # ------------------------------------------------------------------
    # GL upload

    def to_gl_texture(self) -> int:
        """Upload accumulated buffer to an RGBA32F GL texture; return handle."""
        from OpenGL.GL import (
            glGenTextures, glBindTexture, glTexImage2D, glTexSubImage2D,
            glTexParameteri,
            GL_TEXTURE_2D, GL_RGBA32F, GL_RGBA, GL_FLOAT, GL_NEAREST,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE,
        )
        img = self.to_rect()
        # Ensure 4-channel float32 for GL
        if img.shape[2] >= 4:
            img4 = img[:, :, :4].astype(np.float32)
        else:
            img4 = np.ones((self.res_h, self.res_w, 4), np.float32)
            img4[:, :, :img.shape[2]] = img[:, :, :img.shape[2]].astype(np.float32)
        img4 = np.ascontiguousarray(img4)

        if not self._tex_handle:
            self._tex_handle = int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, self._tex_handle)
            for p, v in [(GL_TEXTURE_MIN_FILTER, GL_NEAREST),
                         (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                         (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
                         (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)]:
                glTexParameteri(GL_TEXTURE_2D, p, v)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
                         self.res_w, self.res_h, 0,
                         GL_RGBA, GL_FLOAT, img4)
        else:
            glBindTexture(GL_TEXTURE_2D, self._tex_handle)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            self.res_w, self.res_h,
                            GL_RGBA, GL_FLOAT, img4)
        return self._tex_handle

    def destroy_gl(self) -> None:
        if self._tex_handle:
            from OpenGL.GL import glDeleteTextures
            glDeleteTextures([self._tex_handle])
            self._tex_handle = 0

    # ------------------------------------------------------------------
    # Mesh with pos+norm for scene rendering (6-float32 per vertex)

    def mesh_pn_f32(self, subdivisions: int = 32) -> tuple:
        """Return (verts (V,6) float32 pos+norm, tris (T,3) int32) for GL."""
        pos, uvs, tris = self.mesh_verts_uvs(subdivisions)
        # Flat surface normal: estimate from cross products at each vertex
        # (safe even for degenerate meshes)
        norms = np.zeros_like(pos)
        t = tris
        v0 = pos[t[:, 0]]; v1 = pos[t[:, 1]]; v2 = pos[t[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)           # face normals (unnormalised)
        for i in range(3):
            np.add.at(norms, t[:, i], fn)
        n = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(n > 1e-15, n, 1.0)
        verts_f32 = np.concatenate([pos.astype(np.float32),
                                    norms.astype(np.float32)], axis=1)
        return verts_f32, tris.astype(np.int32)


# ---------------------------------------------------------------------------
# FlatBack
# ---------------------------------------------------------------------------

class FlatBack(CameraBack):
    """Standard flat rectangular sensor / film plane.

    The surface lies in the XY plane at z=0 in sensor-local space.
    Ray direction (dx, dy, dz) is projected to (dx/dz, dy/dz) and mapped to
    UV via the field-of-view tangents.
    """

    def __init__(self, res_w: int = 512, res_h: int = 512,
                 n_channels: int = 4,
                 fov_tan_x: float = 1.0,
                 fov_tan_y: float = 1.0) -> None:
        super().__init__(res_w, res_h, n_channels)
        self.fov_tan_x = float(fov_tan_x)
        self.fov_tan_y = float(fov_tan_y)

    def ray_to_uv(self, ray_dirs: np.ndarray) -> np.ndarray:
        rd = np.asarray(ray_dirs, np.float64)
        safe_z = np.where(rd[:, 2] > 1e-9, rd[:, 2], 1e-9)
        fx = rd[:, 0] / safe_z / max(self.fov_tan_x, 1e-9)
        fy = rd[:, 1] / safe_z / max(self.fov_tan_y, 1e-9)
        u = np.clip((fx + 1.0) * 0.5, 0.0, 1.0)
        v = np.clip((fy + 1.0) * 0.5, 0.0, 1.0)
        return np.stack([u, v], axis=1)

    def uv_to_position(self, uvs: np.ndarray) -> np.ndarray:
        uv = np.asarray(uvs, np.float64)
        x = (2.0 * uv[:, 0] - 1.0) * self.fov_tan_x
        y = (2.0 * uv[:, 1] - 1.0) * self.fov_tan_y
        z = np.zeros(len(uv), np.float64)
        return np.stack([x, y, z], axis=1)

    def uv_to_ray_dir(self, uvs: np.ndarray) -> np.ndarray:
        """Perspective back-cast: UV → unit direction toward scene (+Z)."""
        uv = np.asarray(uvs, np.float64)
        x = (2.0 * uv[:, 0] - 1.0) * self.fov_tan_x
        y = (2.0 * uv[:, 1] - 1.0) * self.fov_tan_y
        z = np.ones(len(uv), np.float64)
        return _norm3(np.stack([x, y, z], axis=1))

    def mesh_verts_uvs(self, subdivisions: int = 1) -> tuple:
        n = max(1, subdivisions)
        xs = np.linspace(-1.0, 1.0, n + 1)
        ys = np.linspace(-1.0, 1.0, n + 1)
        positions, uvs = [], []
        for y in ys:
            for x in xs:
                positions.append([x, y, 0.0])
                uvs.append([(x + 1.0) * 0.5, (y + 1.0) * 0.5])
        tris = []
        w = n + 1
        for i in range(n):
            for j in range(n):
                a = i * w + j
                b = a + 1
                c = a + w
                d = c + 1
                tris += [[a, b, d], [a, d, c]]
        return (np.array(positions, np.float64),
                np.array(uvs,       np.float64),
                np.array(tris,      np.int32))


# ---------------------------------------------------------------------------
# CylindricalBack
# ---------------------------------------------------------------------------

class CylindricalBack(CameraBack):
    """Curved cylindrical film — panoramic / IMAX emulation.

    The cylinder is centred on the optical axis (+Z), open at both ends.
    The horizontal field angle maps uniformly around the arc; vertical maps
    linearly to height.
    """

    def __init__(self, res_w: int = 512, res_h: int = 512,
                 n_channels: int = 4,
                 radius: float = 1.0,
                 half_height: float = 0.5,
                 half_arc_angle: float = math.pi / 2) -> None:
        super().__init__(res_w, res_h, n_channels)
        self.radius          = float(radius)
        self.half_height     = float(half_height)
        self.half_arc_angle  = float(half_arc_angle)   # radians, half-angle

    def ray_to_uv(self, ray_dirs: np.ndarray) -> np.ndarray:
        rd = np.asarray(ray_dirs, np.float64)
        # Horizontal: atan2(x, z) → azimuth
        az = np.arctan2(rd[:, 0], np.where(rd[:, 2] > 1e-9, rd[:, 2], 1e-9))
        u  = np.clip((az / self.half_arc_angle + 1.0) * 0.5, 0.0, 1.0)
        # Vertical: y/z → elevation, mapped to height
        el = rd[:, 1] / np.maximum(np.sqrt(rd[:, 0]**2 + rd[:, 2]**2), 1e-9)
        v  = np.clip((el / (self.half_height / self.radius) + 1.0) * 0.5,
                     0.0, 1.0)
        return np.stack([u, v], axis=1)

    def uv_to_position(self, uvs: np.ndarray) -> np.ndarray:
        uv  = np.asarray(uvs, np.float64)
        phi = (2.0 * uv[:, 0] - 1.0) * self.half_arc_angle
        x   = self.radius * np.sin(phi)
        z   = self.radius * np.cos(phi)
        y   = (2.0 * uv[:, 1] - 1.0) * self.half_height
        return np.stack([x, y, z], axis=1)

    def uv_to_ray_dir(self, uvs: np.ndarray) -> np.ndarray:
        """Inward-facing unit direction from cylindrical surface toward axis."""
        pos = self.uv_to_position(uvs)
        # Inward radial: negate x and z, keep y unchanged, then normalise
        d = np.stack([-pos[:, 0],
                       np.zeros(len(pos), np.float64),
                      -pos[:, 2]], axis=1)
        return _norm3(d)

    def mesh_verts_uvs(self, subdivisions: int = 32) -> tuple:
        n_az  = max(4, subdivisions)
        n_v   = max(2, subdivisions // 2)
        positions, uvs = [], []
        for iv in range(n_v + 1):
            vf = iv / n_v
            y  = self.half_height * (2.0 * vf - 1.0)
            for ia in range(n_az + 1):
                af  = ia / n_az
                phi = self.half_arc_angle * (2.0 * af - 1.0)
                x   = self.radius * math.sin(phi)
                z   = self.radius * math.cos(phi)
                positions.append([x, y, z])
                uvs.append([af, vf])
        tris = []
        w = n_az + 1
        for iv in range(n_v):
            for ia in range(n_az):
                a = iv * w + ia
                b = a + 1
                c = a + w
                d = c + 1
                tris += [[a, b, d], [a, d, c]]
        return (np.array(positions, np.float64),
                np.array(uvs,       np.float64),
                np.array(tris,      np.int32))


# ---------------------------------------------------------------------------
# SphericalBack
# ---------------------------------------------------------------------------

class SphericalBack(CameraBack):
    """Hemispherical / dome back — canonic retinal surface.

    The hemisphere opens in the +Z direction (toward the scene).  Ray
    directions are mapped to UV via their angular deviation from the +Z
    optical axis.  Optionally applies a cortical magnification function (CMF)
    log-polar transform that compresses the periphery and expands the fovea.

    Parameters
    ----------
    radius            : metres — physical radius of the spherical surface
    max_field_angle   : radians — half-angle of the FOV boundary (π/2 = 90°)
    cortical_magnification
                      : bool — apply CMF log-polar UV warp
    cmf_k, cmf_e      : CMF parameters (Schwartz 1977)
                          M^-1(r) = k * log(1 + r_deg / e)
    """

    def __init__(self, res_w: int = 512, res_h: int = 512,
                 n_channels: int = 4,
                 radius: float = 10.5e-3,
                 max_field_angle: float = math.pi / 2,
                 cortical_magnification: bool = False,
                 cmf_k: float = 0.065,
                 cmf_e: float = 15.0) -> None:
        super().__init__(res_w, res_h, n_channels)
        self.radius             = float(radius)
        self.max_field_angle    = float(max_field_angle)
        self.cortical_magnification = bool(cortical_magnification)
        self.cmf_k = float(cmf_k)
        self.cmf_e = float(cmf_e)
        # Precompute CMF normalisation denominator
        self._cmf_norm = (self.cmf_k *
                          math.log1p(math.degrees(max_field_angle) / self.cmf_e))

    def _uv_to_angles(self, u: np.ndarray, v: np.ndarray) -> tuple:
        """Inverse of ``_angles_to_uv``: (N,) UV ∈ [0,1] → (az, el) radians."""
        if self.cortical_magnification:
            # Reconstruct scaled polar coords from the [0,1] UV
            cu = 2.0 * u - 1.0          # ∈ [-1, 1]
            cv = 2.0 * v - 1.0
            r_scaled = np.sqrt(cu ** 2 + cv ** 2)   # ∈ [0, 1] (≈ r_norm)
            phi = np.arctan2(cv, cu)
            # Forward: r_norm_out = cmf_k * log1p(r_deg/e) / cmf_norm
            # Inverse: r_deg = e * (exp(r_scaled * cmf_norm / cmf_k) - 1)
            r_norm_full = np.clip(r_scaled, 0.0, 1.0) * self._cmf_norm
            r_deg = self.cmf_e * np.expm1(r_norm_full / max(self.cmf_k, 1e-15))
            r_rad = np.radians(r_deg)
            az = r_rad * np.cos(phi)
            el = r_rad * np.sin(phi)
        else:
            max_tan = math.tan(self.max_field_angle * 0.5)
            az = np.arctan((2.0 * u - 1.0) * max_tan)
            el = np.arctan((2.0 * v - 1.0) * max_tan)
        return az, el

    def _angles_to_uv(self, az: np.ndarray, el: np.ndarray) -> tuple:
        """(N,) az/el in radians → (u, v) each (N,) in [0, 1]."""
        if self.cortical_magnification:
            r_rad  = np.sqrt(az ** 2 + el ** 2)
            r_deg  = np.degrees(r_rad)
            r_norm = self.cmf_k * np.log1p(r_deg / self.cmf_e)
            r_norm = r_norm / max(self._cmf_norm, 1e-15)
            phi    = np.arctan2(el, az)
            u      = (r_norm * np.cos(phi) + 1.0) * 0.5
            v      = (r_norm * np.sin(phi) + 1.0) * 0.5
        else:
            max_tan = math.tan(self.max_field_angle * 0.5)
            u = (np.tan(np.clip(az, -self.max_field_angle, self.max_field_angle))
                 / max(max_tan, 1e-9) + 1.0) * 0.5
            v = (np.tan(np.clip(el, -self.max_field_angle, self.max_field_angle))
                 / max(max_tan, 1e-9) + 1.0) * 0.5
        return (np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0))

    def ray_to_uv(self, ray_dirs: np.ndarray) -> np.ndarray:
        rd    = np.asarray(ray_dirs, np.float64)
        denom = np.maximum(rd[:, 2], 1e-9)
        az    = np.arctan2(rd[:, 0], denom)
        el    = np.arctan2(rd[:, 1], denom)
        u, v  = self._angles_to_uv(az, el)
        return np.stack([u, v], axis=1)

    def uv_to_position(self, uvs: np.ndarray) -> np.ndarray:
        """Batch back-cast: (N, 2) UV → (N, 3) point on the spherical surface."""
        uv = np.asarray(uvs, np.float64)
        az, el = self._uv_to_angles(uv[:, 0], uv[:, 1])
        sin_az = np.sin(az);  cos_az = np.cos(az)
        sin_el = np.sin(el);  cos_el = np.cos(el)
        x = self.radius * sin_az * cos_el
        y = self.radius * sin_el
        z = self.radius * cos_az * cos_el
        return np.stack([x, y, z], axis=1)

    def uv_to_ray_dir(self, uvs: np.ndarray) -> np.ndarray:
        """Inward unit direction from the dome surface toward the scene origin."""
        pos = self.uv_to_position(uvs)
        return _norm3(-pos)

    def mesh_verts_uvs(self, subdivisions: int = 32) -> tuple:
        """Hemispherical mesh, surface normals pointing inward (toward scene)."""
        n_lat = max(4, subdivisions)
        n_lon = max(8, subdivisions * 2)
        positions, uvs_out = [], []
        tris = []
        # θ ∈ [0, max_field_angle] from optical axis (+Z)
        for i in range(n_lat + 1):
            theta = self.max_field_angle * (i / n_lat)
            for j in range(n_lon + 1):
                phi = 2.0 * math.pi * (j / n_lon) - math.pi
                x = self.radius * math.sin(theta) * math.cos(phi)
                y = self.radius * math.sin(theta) * math.sin(phi)
                z = self.radius * math.cos(theta)
                positions.append([x, y, z])
                # UV via the same angle→UV mapping as ray_to_uv
                az_v = math.atan2(x, max(z, 1e-15))
                el_v = math.atan2(y, max(z, 1e-15))
                u_v, v_v = self._angles_to_uv(
                    np.array([az_v]), np.array([el_v]))
                uvs_out.append([float(u_v[0]), float(v_v[0])])
        for i in range(n_lat):
            for j in range(n_lon):
                a = i * (n_lon + 1) + j
                b = a + 1
                c = a + (n_lon + 1)
                d = c + 1
                tris += [[a, b, d], [a, d, c]]
        return (np.array(positions, np.float64),
                np.array(uvs_out,   np.float64),
                np.array(tris,      np.int32))


# ---------------------------------------------------------------------------
# ManifoldBack
# ---------------------------------------------------------------------------

class ManifoldBack(CameraBack):
    """Routes scene rays through a LensManifold before projecting to a back.

    The manifold maps aperture UV + scene direction → retinal output direction;
    the underlying ``CameraBack`` then converts that output direction to a
    pixel UV.  This is the full eye-model pipeline:

        scene ray (dir) ──► LensManifold.interpolate(uv_ap) ──► out_dir
                                                                       │
                                                          SphericalBack.ray_to_uv
                                                                       │
                                                                pixel (px, py)

    The ``CameraBack`` instance passed as *back* is used for all buffer
    operations; ``ManifoldBack`` delegates its ``_buf`` / ``_count`` to it.
    """

    def __init__(self, manifold, back: CameraBack, k: int = 8) -> None:
        # Delegate buffer storage to the underlying back so callers can use
        # either object interchangeably.
        super().__init__(back.res_w, back.res_h, back.n_channels)
        # Redirect our buffer to the underlying back's buffer
        self._buf   = back._buf
        self._count = back._count
        self.manifold = manifold
        self.back     = back
        self.k        = int(k)

    # -- CameraBack interface ------------------------------------------------

    def ray_to_uv(self, ray_dirs: np.ndarray) -> np.ndarray:
        """Map scene directions through the manifold → back UV.

        The aperture position is fixed at the pupil centre (0, 0) for a
        simple forward query.  Use ``accumulate_scene`` for full aperture
        integration.
        """
        rd      = np.asarray(ray_dirs, np.float64)
        uv_ap   = np.zeros((len(rd), 2), np.float64)   # pupil centre
        out_dir = self.manifold.interpolate(uv_ap, k=self.k)   # (N, 3)
        return self.back.ray_to_uv(out_dir)

    def uv_to_position(self, uvs: np.ndarray) -> np.ndarray:
        """Delegates to the underlying back — parametric, no mesh needed."""
        return self.back.uv_to_position(uvs)

    def uv_to_ray_dir(self, uvs: np.ndarray) -> np.ndarray:
        """Delegates to the underlying back."""
        return self.back.uv_to_ray_dir(uvs)

    def mesh_verts_uvs(self, subdivisions: int = 32) -> tuple:
        return self.back.mesh_verts_uvs(subdivisions)

    def to_rect(self) -> np.ndarray:
        return self.back.to_rect()

    def clear(self) -> None:
        self.back.clear()
        self._buf   = self.back._buf
        self._count = self.back._count

    # -- Extended accumulation with aperture ----------------------------------

    def accumulate_scene(self,
                         scene_dirs:    np.ndarray,
                         uv_aperture:   np.ndarray,
                         values:        np.ndarray,
                         weights:       Optional[np.ndarray] = None) -> None:
        """Accumulate scene rays routed through the manifold with full aperture.

        Parameters
        ----------
        scene_dirs   : (N, 3) float64 — scene-space ray directions
        uv_aperture  : (N, 2) float64 — normalised pupil position [-1,1] for
                                         each ray (from CylinderCoords.disk_sample)
        values       : (N, C) or (N,) — per-ray radiance / energy
        weights      : (N,) or None
        """
        out_dirs = self.manifold.interpolate(
            np.asarray(uv_aperture, np.float64), k=self.k)   # (N, 3)
        uvs = self.back.ray_to_uv(out_dirs)
        self._splat(uvs, values, weights)


# ---------------------------------------------------------------------------
# LargeFormatPlateBack
# ---------------------------------------------------------------------------

class LargeFormatPlateBack(FlatBack):
    """Large-format analog plate positioned behind the digital sensor.

    This is the second light-receiving plane inside a box camera.  It sits
    at ``z_offset_m`` behind the digital sensor plane (negative Z in
    sensor-local space) and is physically wider by ``scale_factor``, matching
    the diverging light cone that passes through the semi-transparent digital
    sensor.

    Like all CameraBack subclasses, it accumulates radiance via the standard
    ``accumulate()`` / ``_splat()`` path.  ``vertex_colors_rgba()`` returns a
    per-vertex RGBA tint that can be applied to the triangulated mesh for
    live rendering as a positive transparency (a digital positive film).

    The mesh from ``mesh_verts_uvs(subdivisions)`` has its vertices pushed
    back by ``z_offset_m`` so the geometry sits at the correct depth inside
    the camera body.  Use ``subdivisions=res_w`` (or ``res_h``) to generate a
    pixel-aligned sub-pixel mesh.

    Parameters
    ----------
    res_w, res_h    : pixel resolution of the accumulation buffer
    n_channels      : number of colour channels (4 = RGBA)
    fov_tan_x/y     : field-of-view tangents inherited from the digital sensor
    z_offset_m      : depth offset behind the sensor plane (metres, > 0)
    scale_factor    : FOV tangent multiplier; > 1 captures a wider cone
    """

    def __init__(self, res_w: int = 512, res_h: int = 512,
                 n_channels: int = 4,
                 fov_tan_x: float = 1.0,
                 fov_tan_y: float = 1.0,
                 z_offset_m: float = 0.010,
                 scale_factor: float = 1.3) -> None:
        super().__init__(res_w, res_h, n_channels,
                         fov_tan_x=fov_tan_x * scale_factor,
                         fov_tan_y=fov_tan_y * scale_factor)
        self.z_offset_m  = float(z_offset_m)
        self.scale_factor = float(scale_factor)

    def mesh_verts_uvs(self, subdivisions: int = 1) -> tuple:
        """Pixel-aligned plate mesh shifted back by ``z_offset_m``."""
        pos, uvs, tris = super().mesh_verts_uvs(subdivisions)
        pos = pos.copy()
        pos[:, 2] -= self.z_offset_m
        return pos, uvs, tris


# ---------------------------------------------------------------------------
# Eye-mesh builder
# ---------------------------------------------------------------------------

def build_eye_mesh(eye,
                   subdivisions: int = 32) -> dict:
    """Build renderable meshes for all eye surfaces.

    Parameters
    ----------
    eye           : EyeGeometry instance
    subdivisions  : tessellation density (32 is fine for display)

    Returns
    -------
    dict mapping surface name → ``(verts (V, 6) float32 pos+norm, tris (T, 3) int32)``

    Surfaces returned
    -----------------
    ``sclera``        — outer sphere (full globe, radius = retina_r * 1.25)
    ``cornea_cap``    — anterior corneal surface (spherical cap)
    ``lens_ant``      — crystalline lens anterior face (spherical cap)
    ``lens_post``     — crystalline lens posterior face (spherical cap, flipped)
    ``retina``        — inner retinal hemisphere (inward-facing dome)
    ``pupil``         — iris aperture disk at pupil_z
    """
    ax = np.array([0., 0., 1.], np.float64)   # optical axis

    def _sphere_cap_pn(radius: float, z_centre: float,
                       cap_angle: float, inward: bool,
                       n: int = subdivisions) -> tuple:
        """Spherical cap mesh; inward=True → normals point toward axis."""
        pos_out, _, _, tris = _sphere_grid(n, n * 2, abs(radius), z_centre,
                                           cap_fraction=cap_angle / math.pi)
        norms = pos_out.copy()
        norms[:, 2] -= z_centre
        nm = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(nm > 1e-15, nm, 1.0)
        if inward:
            norms = -norms
        verts = np.concatenate([pos_out.astype(np.float32),
                                 norms.astype(np.float32)], axis=1)
        return verts, tris.astype(np.int32)

    meshes = {}

    # ── Outer sclera (full sphere) ──────────────────────────────────────
    sclera_r = eye.retina_r * 1.25
    sclera_z = eye.retina_centre_z
    sp, _, _, st = _sphere_grid(subdivisions, subdivisions * 2,
                                 sclera_r, sclera_z, cap_fraction=1.0)
    sn = sp.copy(); sn[:, 2] -= sclera_z
    sn_m = np.linalg.norm(sn, axis=1, keepdims=True)
    sn = sn / np.where(sn_m > 1e-15, sn_m, 1.0)
    meshes['sclera'] = (
        np.concatenate([sp.astype(np.float32), sn.astype(np.float32)], axis=1),
        st.astype(np.int32))

    # ── Cornea cap (anterior surface, outward normals) ───────────────────
    cornea_cap_angle = math.asin(
        min(eye.pupil_r * 2.0 / abs(eye.cornea_r_curv), 0.99))
    meshes['cornea_cap'] = _sphere_cap_pn(
        eye.cornea_r_curv, eye.cornea_z - eye.cornea_r_curv,
        cornea_cap_angle, inward=False, n=subdivisions // 2)

    # ── Crystalline lens anterior ────────────────────────────────────────
    lens_cap_angle = math.asin(
        min(eye.pupil_r * 1.5 / abs(eye.lens_ant_r_curv), 0.99))
    meshes['lens_ant'] = _sphere_cap_pn(
        eye.lens_ant_r_curv, eye.lens_ant_z - eye.lens_ant_r_curv,
        lens_cap_angle, inward=False, n=subdivisions // 2)

    # ── Crystalline lens posterior ───────────────────────────────────────
    meshes['lens_post'] = _sphere_cap_pn(
        eye.lens_post_r_curv, eye.lens_post_z - eye.lens_post_r_curv,
        lens_cap_angle, inward=True, n=subdivisions // 2)

    # ── Retina hemisphere (inward-facing, toward lens) ───────────────────
    ret_p, _, _, ret_t = _sphere_grid(
        subdivisions, subdivisions * 2,
        eye.retina_r, eye.retina_centre_z, cap_fraction=0.5)
    # Normals point toward centre (inward)
    ret_n = ret_p.copy(); ret_n[:, 2] -= eye.retina_centre_z
    ret_nm = np.linalg.norm(ret_n, axis=1, keepdims=True)
    ret_n = -ret_n / np.where(ret_nm > 1e-15, ret_nm, 1.0)   # inward
    meshes['retina'] = (
        np.concatenate([ret_p.astype(np.float32),
                         ret_n.astype(np.float32)], axis=1),
        ret_t.astype(np.int32))

    # ── Pupil disk ────────────────────────────────────────────────────────
    pd_pos, pd_uv, pd_tri = _disk_grid(
        max(4, subdivisions // 4), max(16, subdivisions), eye.pupil_r)
    pd_pos = pd_pos.copy()
    pd_pos[:, 2] = eye.pupil_z   # move to pupil plane
    pupil_norm = np.tile([0., 0., 1.], (len(pd_pos), 1)).astype(np.float32)
    meshes['pupil'] = (
        np.concatenate([pd_pos.astype(np.float32), pupil_norm], axis=1),
        pd_tri.astype(np.int32))

    return meshes
