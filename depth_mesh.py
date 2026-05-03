"""depth_mesh.py
================
Parametric depth-map-driven mesh.

Stock base shapes
-----------------
  rect    — flat rectangular grid, W × H metres.
            Grid has (grid_u+1) × (grid_v+1) vertices.
            Canonical axes: axis_u (columns) × axis_v (rows) determine the
            plane; the base normal is cross(axis_u, axis_v).

  cyl     — cylindrical surface, radius R, length L along the axis_v direction,
            sweep_deg ≤ 360° around that axis.
            (grid_u+1) columns (angle), (grid_v+1) rows (length).

  sphere  — UV sphere, radius R.
            (grid_u+1) columns (longitude), (grid_v+1) rows (latitude,
            row 0 = south pole, row grid_v = north pole).

Coordinate convention (matches room_geometry.py)
-------------------------------------------------
  +X right, +Y depth (into room), +Z up.

Depth map
---------
  A 2-D numpy array [grid_v+1, grid_u+1] of float values, any range;
  values are mapped to [0, 1] internally for display but used raw for
  displacement:

      vertex_pos += base_normal * depth_map[row, col] * depth_scale

  Pass ``depth_map=None`` (default) for no displacement — flat surface.

Origin vertex
-------------
  Named string or integer vertex index.  After generation all vertices are
  translated so the chosen vertex lies at (0, 0, 0) in local space.

  Named anchors for rect:
      "center"          grid centre (interpolated)
      "bottom_left"     col=0, row=0
      "bottom_right"    col=grid_u, row=0
      "top_left"        col=0, row=grid_v
      "top_right"       col=grid_u, row=grid_v
      "bottom_center"   col=grid_u//2, row=0
      "top_center"      col=grid_u//2, row=grid_v
      "left_center"     col=0, row=grid_v//2
      "right_center"    col=grid_u, row=grid_v//2

  Named anchors for cyl / sphere:
      "bottom_center"   axis end at v=0 (bottom centre for cyl)
      "top_center"      axis end at v=grid_v (top centre for cyl)
      "bottom_seam"     col=0, row=0  (seam at angle=0)
      "top_seam"        col=0, row=grid_v
      "bottom_pole"     south-pole vertex (sphere only)
      "top_pole"        north-pole vertex (sphere only)

  Integer: 0-based flat vertex index (row * (grid_u+1) + col).

GL upload
---------
  After building, call ``build_gl()`` to upload to GPU.
  Draw with ``draw(MVP, MV, prog, light_uniforms_fn)`` where
  ``light_uniforms_fn(prog)`` sets any material / light uniforms on the
  already-bound program.
"""
from __future__ import annotations

import ctypes
import math
from typing import Optional, Union

import numpy as np

try:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
        GL_TRIANGLES,
        glBindBuffer, glBindVertexArray, glBufferData, glDrawArrays,
        glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays,
        glUniformMatrix4fv, glGetUniformLocation, GL_TRUE,
        glVertexAttribPointer, glDeleteBuffers, glDeleteVertexArrays,
        glBufferSubData,
    )
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm3(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def _triangle_normal(p0, p1, p2) -> np.ndarray:
    return _norm3(_cross(p1 - p0, p2 - p0))


# ─────────────────────────────────────────────────────────────────────────────
# Per-shape canonical vertex grid builders
# Returns: pos[V, U, 3] float64,  base_normals[V, U, 3] float64
# ─────────────────────────────────────────────────────────────────────────────

def _build_rect_grid(width: float, height: float,
                     axis_u: np.ndarray, axis_v: np.ndarray,
                     grid_u: int, grid_v: int):
    """Flat rectangular grid in the plane spanned by axis_u × axis_v."""
    au = _norm3(np.asarray(axis_u, np.float64))
    av = _norm3(np.asarray(axis_v, np.float64))
    base_n = _norm3(_cross(au, av))

    us = np.linspace(0.0, 1.0, grid_u + 1)
    vs = np.linspace(0.0, 1.0, grid_v + 1)
    pos = np.zeros((grid_v + 1, grid_u + 1, 3), np.float64)
    nrm = np.zeros((grid_v + 1, grid_u + 1, 3), np.float64)
    for vi, vf in enumerate(vs):
        for ui, uf in enumerate(us):
            pos[vi, ui] = au * (uf * width) + av * (vf * height)
            nrm[vi, ui] = base_n
    return pos, nrm


def _build_cyl_grid(radius: float, length: float,
                    sweep_deg: float,
                    axis_v: np.ndarray,
                    grid_u: int, grid_v: int,
                    start_angle_deg: float = 0.0):
    """Cylindrical shell swept around axis_v.
    Columns (U) sweep around the axis; rows (V) go along the axis.
    start_angle_deg offsets the seam from angle=0 so the arc can be centred
    in any direction.  Outward normals point radially away from the axis.
    """
    av = _norm3(np.asarray(axis_v, np.float64))
    # Build two perpendicular axes for the radial plane.
    # Prefer [1,0,0] unless av is nearly parallel to it.
    if abs(av[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    ax = _norm3(np.cross(tmp, av))  # 'right' in the radial plane
    ay = _norm3(np.cross(av, ax))   # 'up' in the radial plane

    start_rad = math.radians(start_angle_deg)
    angles = np.linspace(start_rad, start_rad + math.radians(sweep_deg), grid_u + 1)
    lengths = np.linspace(0.0, length, grid_v + 1)

    pos = np.zeros((grid_v + 1, grid_u + 1, 3), np.float64)
    nrm = np.zeros((grid_v + 1, grid_u + 1, 3), np.float64)
    for vi, lf in enumerate(lengths):
        for ui, theta in enumerate(angles):
            radial = ax * math.cos(theta) + ay * math.sin(theta)
            pos[vi, ui] = radial * radius + av * lf
            nrm[vi, ui] = radial
    return pos, nrm


def _build_sphere_grid(radius: float, grid_u: int, grid_v: int,
                       lat_min_deg: float = -90.0, lat_max_deg: float = 90.0,
                       lon_min_deg: float = 0.0,   lon_max_deg: float = 360.0):
    """UV sphere (or a cap / sector thereof).
    Row 0 = south-most latitude, row grid_v = north-most latitude.
    Col 0..grid_u spans lon_min..lon_max (longitude).
    Outward normals are the unit position vectors.

    Default behaviour (lat -90..+90, lon 0..360) builds a full UV sphere.
    Restrict lat_min/lat_max to produce an upper/lower cap.
    Restrict lon_min/lon_max to produce a longitudinal sector.
    """
    lons = np.linspace(math.radians(lon_min_deg),
                       math.radians(lon_max_deg), grid_u + 1)
    lats = np.linspace(math.radians(lat_min_deg),
                       math.radians(lat_max_deg), grid_v + 1)

    pos = np.zeros((grid_v + 1, grid_u + 1, 3), np.float64)
    nrm = np.zeros((grid_v + 1, grid_u + 1, 3), np.float64)
    for vi, lat in enumerate(lats):
        for ui, lon in enumerate(lons):
            p = np.array([
                math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat),
            ])
            pos[vi, ui] = p * radius
            nrm[vi, ui] = p
    return pos, nrm


# ─────────────────────────────────────────────────────────────────────────────
# Origin anchor resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_origin(pos: np.ndarray,
                    origin: Union[str, int],
                    shape: str,
                    grid_u: int, grid_v: int) -> np.ndarray:
    """Return the world position of the chosen origin vertex."""
    G = grid_u + 1  # cols

    def _at(vi, ui):
        return pos[vi, ui].copy()

    if isinstance(origin, int):
        vi, ui = divmod(origin, G)
        return _at(vi, ui)

    named = {
        # rect / universal corners
        "bottom_left":    (0,         0),
        "bottom_right":   (0,         grid_u),
        "top_left":       (grid_v,    0),
        "top_right":      (grid_v,    grid_u),
        "bottom_center":  (0,         grid_u // 2),
        "top_center":     (grid_v,    grid_u // 2),
        "left_center":    (grid_v // 2, 0),
        "right_center":   (grid_v // 2, grid_u),
        # cyl / sphere
        "bottom_seam":    (0,         0),
        "top_seam":       (grid_v,    0),
        "bottom_pole":    (0,         0),
        "top_pole":       (grid_v,    0),
    }

    if origin == "center":
        # Bilinear centre of the grid
        mid_v = (grid_v) / 2.0
        mid_u = (grid_u) / 2.0
        vi0, ui0 = int(mid_v), int(mid_u)
        vi1, ui1 = min(vi0 + 1, grid_v), min(ui0 + 1, grid_u)
        fv, fu = mid_v - vi0, mid_u - ui0
        p = (pos[vi0, ui0] * (1 - fu) * (1 - fv)
             + pos[vi0, ui1] * fu * (1 - fv)
             + pos[vi1, ui0] * (1 - fu) * fv
             + pos[vi1, ui1] * fu * fv)
        return p

    if origin in named:
        vi, ui = named[origin]
        return _at(vi, ui)

    raise ValueError(f"Unknown origin anchor: {origin!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Smooth normals via vertex averaging
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_normals_from_positions(pos: np.ndarray) -> np.ndarray:
    """Compute per-vertex averaged normals from a (V, U, 3) position grid."""
    V, U = pos.shape[:2]
    normals = np.zeros((V, U, 3), np.float64)
    for vi in range(V):
        for ui in range(U):
            n_acc = np.zeros(3, np.float64)
            count = 0
            # Accumulate normals from all neighbouring triangles sharing this vertex
            for dv, du in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
                r0, c0 = vi + dv, ui + du
                r1, c1 = r0 + 1, c0 + 1
                if 0 <= r0 < V - 1 and 0 <= c0 < U - 1:
                    p00 = pos[r0, c0]
                    p10 = pos[r1, c0]
                    p01 = pos[r0, c1]
                    p11 = pos[r1, c1]
                    # Two triangles: (00,10,11) and (00,11,01)
                    n1 = _triangle_normal(p00, p10, p11)
                    n2 = _triangle_normal(p00, p11, p01)
                    if np.all(np.isfinite(n1)):
                        n_acc += n1; count += 1
                    if np.all(np.isfinite(n2)):
                        n_acc += n2; count += 1
            if count:
                normals[vi, ui] = _norm3(n_acc)
    return normals


# ─────────────────────────────────────────────────────────────────────────────
# Triangle soup builder
# ─────────────────────────────────────────────────────────────────────────────

def _grid_to_triangles(pos: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Convert (V, U, 3) position+normal grids to (N, 6) float32 triangle soup."""
    V, U = pos.shape[:2]
    tris = []
    for vi in range(V - 1):
        for ui in range(U - 1):
            p00, n00 = pos[vi,   ui],   nrm[vi,   ui]
            p10, n10 = pos[vi+1, ui],   nrm[vi+1, ui]
            p01, n01 = pos[vi,   ui+1], nrm[vi,   ui+1]
            p11, n11 = pos[vi+1, ui+1], nrm[vi+1, ui+1]
            # CCW winding facing base_normal direction
            for p, n in [(p00, n00), (p10, n10), (p11, n11)]:
                tris.append((*p, *n))
            for p, n in [(p00, n00), (p11, n11), (p01, n01)]:
                tris.append((*p, *n))
    return np.array(tris, np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DepthMesh
# ─────────────────────────────────────────────────────────────────────────────

class DepthMesh:
    """Parametric mesh built from a stock base shape, optionally displaced by a depth map.

    Parameters
    ----------
    shape : str
        "rect" | "cyl" | "sphere"
    size : tuple
        rect  → (width, height) in metres
        cyl   → (radius, length)
        sphere → (radius,)          # single-element tuple or scalar
    grid_u, grid_v : int
        Subdivision counts along U (columns) and V (rows).
    depth_map : np.ndarray or None
        2-D array [grid_v+1, grid_u+1] of displacement values.
        Each vertex is pushed along its base normal by value * depth_scale.
    depth_scale : float
        Metres per unit of depth_map value.  Default 1.0.
    origin : str or int
        Origin vertex anchor.  See module docstring for named anchors.
        Default "bottom_left".
    axis_u : array-like
        (rect only) U-axis direction.  Default [1, 0, 0].
    axis_v : array-like
        (rect / cyl) V-axis direction.  Default [0, 0, 1].
    sweep_deg : float
        (cyl only) Angular sweep.  Default 360.0.
    smooth_normals : bool
        Recompute per-vertex normals from triangle soup after displacement.
        Recommended when depth_map is non-uniform.  Default True.
    """

    def __init__(self,
                 shape: str = "rect",
                 size=(1.0, 1.0),
                 grid_u: int = 1,
                 grid_v: int = 1,
                 depth_map: Optional[np.ndarray] = None,
                 depth_scale: float = 1.0,
                 origin: Union[str, int] = "bottom_left",
                 axis_u=(1.0, 0.0, 0.0),
                 axis_v=(0.0, 0.0, 1.0),
                 sweep_deg: float = 360.0,
                 smooth_normals: bool = True,
                 # ── extended shape control ──────────────────────────────
                 start_angle_deg: float = 0.0,
                 flip_normals: bool = False,
                 lat_min_deg: float = -90.0,
                 lat_max_deg: float = 90.0,
                 lon_min_deg: float = 0.0,
                 lon_max_deg: float = 360.0):
        """
        Additional parameters
        --------------------
        start_angle_deg : float
            (cyl only) Starting angle for the sweep in degrees.  Use this to
            centre the arc in any radial direction.  Default 0 = seam at the
            canonical 'ax' direction for the chosen axis_v.
        flip_normals : bool
            Negate all base normals after grid construction.  Use True for
            concave (inside-facing) surfaces such as curved walls viewed from
            within.  Default False.
        lat_min_deg, lat_max_deg : float
            (sphere only) Latitude range in degrees.  Default -90..+90 = full
            sphere.  Restrict to build a cap, e.g. lat_min=0 for the upper
            hemisphere.
        lon_min_deg, lon_max_deg : float
            (sphere only) Longitude range.  Default 0..360 = full longitude.
        """
        self.shape           = shape.lower()
        self.size            = size
        self.grid_u          = int(grid_u)
        self.grid_v          = int(grid_v)
        self.depth_map       = depth_map
        self.depth_scale     = float(depth_scale)
        self.origin          = origin
        self.axis_u          = np.asarray(axis_u, np.float64)
        self.axis_v          = np.asarray(axis_v, np.float64)
        self.sweep_deg       = float(sweep_deg)
        self.smooth_normals  = smooth_normals
        self.start_angle_deg = float(start_angle_deg)
        self.flip_normals    = bool(flip_normals)
        self.lat_min_deg     = float(lat_min_deg)
        self.lat_max_deg     = float(lat_max_deg)
        self.lon_min_deg     = float(lon_min_deg)
        self.lon_max_deg     = float(lon_max_deg)

        # Built data
        self._vertex_data: Optional[np.ndarray] = None   # (N, 6) float32
        self._origin_offset: Optional[np.ndarray] = None  # (3,) float64

        # GL handles
        self._vao = self._vbo = None
        self._n_verts: int = 0
        self._gl_ready: bool = False

        self.build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self):
        """(Re)generate mesh CPU data from current parameters."""
        s = self.shape
        sz = self.size

        if s == "rect":
            w = float(sz[0]) if len(sz) >= 1 else 1.0
            h = float(sz[1]) if len(sz) >= 2 else 1.0
            pos, nrm = _build_rect_grid(w, h, self.axis_u, self.axis_v,
                                        self.grid_u, self.grid_v)
        elif s == "cyl":
            r = float(sz[0]) if len(sz) >= 1 else 0.5
            l = float(sz[1]) if len(sz) >= 2 else 1.0
            pos, nrm = _build_cyl_grid(r, l, self.sweep_deg, self.axis_v,
                                       self.grid_u, self.grid_v,
                                       start_angle_deg=self.start_angle_deg)
        elif s == "sphere":
            r = float(sz[0]) if hasattr(sz, '__len__') else float(sz)
            pos, nrm = _build_sphere_grid(r, self.grid_u, self.grid_v,
                                          lat_min_deg=self.lat_min_deg,
                                          lat_max_deg=self.lat_max_deg,
                                          lon_min_deg=self.lon_min_deg,
                                          lon_max_deg=self.lon_max_deg)
        else:
            raise ValueError(f"Unknown shape: {self.shape!r}")

        # Optionally invert normals for inside-facing (concave) surfaces
        if self.flip_normals:
            nrm = -nrm

        # Apply depth map displacement along base normals
        if self.depth_map is not None:
            dm = np.asarray(self.depth_map, np.float64)
            # Resize depth map to grid if needed
            if dm.shape != (self.grid_v + 1, self.grid_u + 1):
                dm = self._resize_depth_map(dm)
            for vi in range(self.grid_v + 1):
                for ui in range(self.grid_u + 1):
                    pos[vi, ui] += nrm[vi, ui] * dm[vi, ui] * self.depth_scale
            if self.smooth_normals:
                nrm = _smooth_normals_from_positions(pos)

        # Compute origin offset and translate
        origin_pos = _resolve_origin(pos, self.origin, s, self.grid_u, self.grid_v)
        pos -= origin_pos[np.newaxis, np.newaxis, :]
        self._origin_offset = origin_pos.copy()

        self._vertex_data = _grid_to_triangles(pos, nrm)
        self._n_verts = len(self._vertex_data)

    def _resize_depth_map(self, dm: np.ndarray) -> np.ndarray:
        """Bilinear resize of depth map to (grid_v+1, grid_u+1)."""
        from scipy.ndimage import zoom
        target_h = self.grid_v + 1
        target_w = self.grid_u + 1
        zy = target_h / dm.shape[0]
        zx = target_w / dm.shape[1]
        return zoom(dm, (zy, zx), order=1)

    # ── Depth map update ──────────────────────────────────────────────────────

    def set_depth_map(self, depth_map: Optional[np.ndarray],
                      rebuild_gl: bool = True):
        """Update depth map and rebuild mesh.  If GL is ready, re-uploads."""
        self.depth_map = depth_map
        self.build()
        if rebuild_gl and self._gl_ready:
            self._upload_gl()

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def build_gl(self):
        """Upload vertex data to GPU.  Call once after GL context is ready."""
        if not _HAS_GL:
            return
        if self._vertex_data is None or self._n_verts == 0:
            return
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertex_data.nbytes,
                     self._vertex_data.tobytes(), GL_DYNAMIC_DRAW)
        stride = 6 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(12))
        glBindVertexArray(0)
        self._gl_ready = True

    def _upload_gl(self):
        """Re-upload vertex data after a rebuild.  Assumes VAO/VBO exist."""
        if not _HAS_GL or not self._gl_ready:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._vertex_data.nbytes,
                     self._vertex_data.tobytes(), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def destroy_gl(self):
        if self._gl_ready and _HAS_GL:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
        self._vao = self._vbo = None
        self._gl_ready = False

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, MVP: np.ndarray, MV: np.ndarray,
             prog: int, uniforms_fn=None):
        """Draw the mesh using the given compiled program.

        Parameters
        ----------
        MVP, MV : np.ndarray (4, 4) float32
            Combined model-view-projection and model-view matrices.
            These should already incorporate the station's world transform
            and any local offset.
        prog : int
            Compiled GL program handle.
        uniforms_fn : callable or None
            ``uniforms_fn(prog)`` — called after binding prog, before draw.
            Should set material/light uniforms on the already-bound program.
        """
        if not self._gl_ready or self._n_verts == 0:
            return
        from OpenGL.GL import glUseProgram, glUniformMatrix4fv, glBindVertexArray, glDrawArrays
        glUseProgram(prog)
        loc_mvp = glGetUniformLocation(prog, b'uMVP')
        if loc_mvp >= 0:
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, MVP.astype(np.float32))
        loc_mv = glGetUniformLocation(prog, b'uMV')
        if loc_mv >= 0:
            glUniformMatrix4fv(loc_mv, 1, GL_TRUE, MV.astype(np.float32))
        if uniforms_fn is not None:
            uniforms_fn(prog)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._n_verts)
        glBindVertexArray(0)
        glUseProgram(0)

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def flat_wall(cls,
                  width: float = 2.0,
                  height: float = 3.0,
                  grid_u: int = 8,
                  grid_v: int = 8,
                  depth_map: Optional[np.ndarray] = None,
                  depth_scale: float = 0.1,
                  origin: Union[str, int] = "bottom_left") -> "DepthMesh":
        """Vertical wall panel in the XZ plane (normal −Y, facing player)."""
        return cls(shape="rect",
                   size=(width, height),
                   grid_u=grid_u, grid_v=grid_v,
                   depth_map=depth_map, depth_scale=depth_scale,
                   origin=origin,
                   axis_u=(1.0, 0.0, 0.0),
                   axis_v=(0.0, 0.0, 1.0))

    @classmethod
    def flat_floor(cls,
                   width: float = 2.0,
                   depth: float = 2.0,
                   grid_u: int = 4,
                   grid_v: int = 4,
                   depth_map: Optional[np.ndarray] = None,
                   depth_scale: float = 0.02,
                   origin: Union[str, int] = "bottom_left") -> "DepthMesh":
        """Horizontal floor tile in the XY plane (normal +Z, facing up)."""
        return cls(shape="rect",
                   size=(width, depth),
                   grid_u=grid_u, grid_v=grid_v,
                   depth_map=depth_map, depth_scale=depth_scale,
                   origin=origin,
                   axis_u=(1.0, 0.0, 0.0),
                   axis_v=(0.0, 1.0, 0.0))

    @classmethod
    def from_yaml(cls, cfg: dict) -> "DepthMesh":
        """Construct a DepthMesh from a YAML config dict.

        Expected keys (all optional, with defaults):
          shape       : "rect" | "cyl" | "sphere"    default: "rect"
          size        : [w, h] or [r, l] or [r]       default: [1.0, 1.0]
          grid_u      : int                            default: 8
          grid_v      : int                            default: 8
          depth_scale : float                          default: 0.1
          origin      : str or int                     default: "bottom_left"
          axis_u      : [x, y, z]                      default: [1,0,0]
          axis_v      : [x, y, z]                      default: [0,0,1]
          sweep_deg   : float                          default: 360.0
          smooth_normals: bool                         default: true
          depth_map_path: str or null                  default: null
        """
        shape  = cfg.get("shape", "rect")
        size   = tuple(cfg.get("size", [1.0, 1.0]))
        gu     = int(cfg.get("grid_u", 8))
        gv     = int(cfg.get("grid_v", 8))
        dscale = float(cfg.get("depth_scale", 0.1))
        origin = cfg.get("origin", "bottom_left")
        au     = cfg.get("axis_u", [1.0, 0.0, 0.0])
        av     = cfg.get("axis_v", [0.0, 0.0, 1.0])
        sweep  = float(cfg.get("sweep_deg", 360.0))
        smooth = bool(cfg.get("smooth_normals", True))

        dm = None
        dm_path = cfg.get("depth_map_path")
        if dm_path:
            dm = _load_depth_map(dm_path)

        return cls(shape=shape, size=size, grid_u=gu, grid_v=gv,
                   depth_map=dm, depth_scale=dscale, origin=origin,
                   axis_u=au, axis_v=av, sweep_deg=sweep,
                   smooth_normals=smooth)

    # ── Properties ────────────────────────────────────────────────────────────

    # ── Shape description (for ray tracing / collision) ──────────────────────

    def shape_description(self) -> dict:
        """Return an analytical description of the base shape.

        Ray-tracing and collision systems can use this dict instead of
        the triangle mesh for exact intersection tests.

        Returned keys common to all shapes
        -----------------------------------
        type         : 'plane' | 'cylinder' | 'sphere'
        flip_normals : bool  — True when normals point inward (concave surface)
        origin_offset: [x,y,z]  — translation applied to put the origin vertex
                                   at (0,0,0) in local space

        Additional keys by type
        -----------------------
        plane
            normal  : [nx,ny,nz]  outward (pre-flip) normal direction
            size    : [width, height]  metres
        cylinder
            radius      : float  metres
            length      : float  metres (along cylinder axis)
            axis        : [ax,ay,az]  unit cylinder axis direction
            sweep_deg   : float  arc span in degrees
            start_angle_deg : float  start angle of the sweep
        sphere
            radius       : float  metres
            lat_min_deg  : float
            lat_max_deg  : float
            lon_min_deg  : float
            lon_max_deg  : float
        """
        sz = self.size
        base: dict = {
            'flip_normals':  self.flip_normals,
            'origin_offset': (self._origin_offset.tolist()
                              if self._origin_offset is not None else [0, 0, 0]),
        }
        if self.shape == 'rect':
            au = _norm3(self.axis_u)
            av = _norm3(self.axis_v)
            base.update({
                'type':   'plane',
                'normal': _norm3(_cross(au, av)).tolist(),
                'size':   [float(sz[0]), float(sz[1])],
            })
        elif self.shape == 'cyl':
            base.update({
                'type':            'cylinder',
                'radius':          float(sz[0]),
                'length':          float(sz[1]),
                'axis':            _norm3(self.axis_v).tolist(),
                'sweep_deg':       self.sweep_deg,
                'start_angle_deg': self.start_angle_deg,
            })
        elif self.shape == 'sphere':
            base.update({
                'type':        'sphere',
                'radius':      float(sz[0]) if hasattr(sz, '__len__') else float(sz),
                'lat_min_deg': self.lat_min_deg,
                'lat_max_deg': self.lat_max_deg,
                'lon_min_deg': self.lon_min_deg,
                'lon_max_deg': self.lon_max_deg,
            })
        else:
            base['type'] = 'unknown'
        return base

    # ── Wall convenience constructor ──────────────────────────────────────────

    @classmethod
    def wall(cls,
             shape_type: str = "flat",
             width: float = 2.0,
             height: float = 3.0,
             grid_u: int = 16,
             grid_v: int = 24,
             depth_map: Optional[np.ndarray] = None,
             depth_scale: float = 0.0,
             origin: Union[str, int] = "bottom_left",
             arc_radius: float = 6.0,
             smooth_normals: bool = True) -> "DepthMesh":
        """Build a wall mesh for the given shape.

        shape_type
        ----------
        flat
            Rectangular flat panel. axis_u=+X, axis_v=+Z, outward normal=−Y
            (faces the operator standing in front of the console).

        arc
            Concave horizontal arc (cylinder axis +Z). The wall curves
            left/right around the operator, like a planetarium screen.
            arc_radius is measured from the operator position to the wall
            surface. Minimum safe radius enforced via min_wall_radius().

        vertical_arc
            Concave vertical arc (cylinder axis +X, along the station width).
            The wall transitions into an arching ceiling — a barrel-vault or
            cove shape. Same minimum-radius constraint as arc.

        spherical
            Concave sphere cap. The sphere is centred at the operator position;
            the cap covers the back wall and the overhead area. Lon 45°–135°
            (centred on +Y = into the room) and lat 0°–90° (above horizon).

        Parameters
        ----------
        arc_radius : float
            Metres. Used for arc / vertical_arc / spherical shapes. Values
            smaller than min_wall_radius(width, 0.7) are silently clamped.
        """
        st = shape_type.lower()

        if st == "flat":
            return cls(shape="rect",
                       size=(width, height),
                       grid_u=grid_u, grid_v=grid_v,
                       depth_map=depth_map, depth_scale=depth_scale,
                       origin=origin,
                       axis_u=(1.0, 0.0, 0.0),
                       axis_v=(0.0, 0.0, 1.0),
                       smooth_normals=smooth_normals)

        # Clamp radius to safe minimum
        r_min = min_wall_radius(width, 0.7)
        r = max(float(arc_radius), r_min)

        if st == "arc":
            # Concave horizontal cylinder (axis +Z), swept left/right.
            # With axis_v=(0,0,1): radial at theta=180° is [0,+1,0] (+Y).
            # Centre the arc there so the wall faces the operator (-Y inward).
            half_sin = min(width / (2.0 * r), 1.0)
            sweep = math.degrees(2.0 * math.asin(half_sin))
            start = 180.0 - sweep / 2.0
            return cls(shape="cyl",
                       size=(r, height),
                       grid_u=grid_u, grid_v=grid_v,
                       depth_map=depth_map, depth_scale=depth_scale,
                       origin=origin,
                       axis_v=(0.0, 0.0, 1.0),
                       sweep_deg=sweep,
                       start_angle_deg=start,
                       flip_normals=True,
                       smooth_normals=smooth_normals)

        if st == "vertical_arc":
            # Concave vertical cylinder (axis +X, along station width).
            # Arc in the YZ plane, from wall to ceiling.
            # With axis_v=(1,0,0): ax=[0,0,-1], ay=[0,1,0].
            # theta=90° → radial=[0,1,0] (+Y); theta=180° → radial=[0,0,1] (+Z).
            # Centre the arc at theta=90° (wall/ceiling junction): start at 90 - sweep/2.
            half_sin = min(height / (2.0 * r), 1.0)
            sweep = math.degrees(2.0 * math.asin(half_sin))
            start = 90.0 - sweep / 2.0
            return cls(shape="cyl",
                       size=(r, width),
                       grid_u=grid_v, grid_v=grid_u,   # v=along-width, u=along-arc
                       depth_map=depth_map, depth_scale=depth_scale,
                       origin=origin,
                       axis_v=(1.0, 0.0, 0.0),
                       sweep_deg=sweep,
                       start_angle_deg=start,
                       flip_normals=True,
                       smooth_normals=smooth_normals)

        if st == "spherical":
            # Concave sphere cap centred at operator position.
            # Lon 45°–135° (centred on +Y = into the room, the wall direction).
            # Lat 0°–90° (above horizon = upper hemisphere).
            # Width determines the lon range, height the lat range.
            half_lon_sin = min(width / (2.0 * r), 1.0)
            lon_half = math.degrees(math.asin(half_lon_sin))
            half_lat_sin = min(height / (2.0 * r), 1.0)
            lat_half = math.degrees(math.asin(half_lat_sin))
            lon_center = 90.0   # +Y direction in standard sphere coords
            lat_center = 45.0   # 45° above horizon
            return cls(shape="sphere",
                       size=(r,),
                       grid_u=grid_u, grid_v=grid_v,
                       depth_map=depth_map, depth_scale=depth_scale,
                       origin=origin,
                       lon_min_deg=lon_center - lon_half,
                       lon_max_deg=lon_center + lon_half,
                       lat_min_deg=max(0.0, lat_center - lat_half),
                       lat_max_deg=min(90.0, lat_center + lat_half),
                       flip_normals=True,
                       smooth_normals=smooth_normals)

        raise ValueError(
            f"Unknown wall shape_type {shape_type!r}. "
            "Expected: flat | arc | vertical_arc | spherical")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vertex_data(self) -> Optional[np.ndarray]:
        """(N, 6) float32 triangle soup: [x, y, z, nx, ny, nz]."""
        return self._vertex_data

    @property
    def n_verts(self) -> int:
        return self._n_verts

    def __repr__(self) -> str:
        return (f"DepthMesh(shape={self.shape!r}, size={self.size}, "
                f"grid={self.grid_u}×{self.grid_v}, "
                f"n_verts={self._n_verts}, gl={self._gl_ready})")


# ─────────────────────────────────────────────────────────────────────────────
# Depth map I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_depth_map(path: str) -> np.ndarray:
    """Load a depth map from a file.
    Supported formats:
      .npy / .npz  — numpy arrays (first array for .npz)
      .png / .jpg  — grayscale image normalized to [0, 1] float64
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float64)
    if ext == ".npz":
        d = np.load(path)
        key = list(d.keys())[0]
        return d[key].astype(np.float64)
    # Image path — treat as grayscale heightmap
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.asarray(img, np.float64) / 255.0
    except ImportError:
        import struct, zlib
        # Minimal PNG loader fallback for 8-bit grayscale
        raise RuntimeError(
            f"PIL (Pillow) required to load image depth maps: {path}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Wall radius utility
# ─────────────────────────────────────────────────────────────────────────────

def min_wall_radius(station_width: float, station_depth: float,
                    margin: float = 0.5) -> float:
    """Minimum safe arc radius for a curved wall behind a duty station.

    The curved surface must not intrude into the space the operator needs to
    stand and work.  Uses the half-diagonal of (station_width/2, station_depth
    + margin), which represents the far corner of the operating footprint.

    Parameters
    ----------
    station_width : float
        Full width (X extent) of the station including side wings, in metres.
    station_depth : float
        Depth (Y extent) of the station body, in metres.
    margin : float
        Additional clearance behind the station, in metres.  Default 0.5 m.

    Returns
    -------
    float
        Minimum radius in metres.  Arc/sphere radii smaller than this value
        would clip into the operating space.
    """
    half_w = station_width / 2.0
    return math.sqrt(half_w ** 2 + (station_depth + margin) ** 2)


def flat_depth_map(grid_u: int, grid_v: int, value: float = 0.0) -> np.ndarray:
    """Convenience: create a flat (uniform) depth map."""
    return np.full((grid_v + 1, grid_u + 1), value, np.float64)


def gaussian_bump(grid_u: int, grid_v: int,
                  cx: float = 0.5, cy: float = 0.5,
                  sigma: float = 0.25,
                  amplitude: float = 1.0) -> np.ndarray:
    """Convenience: depth map with a Gaussian bump.
    cx, cy in [0,1]; sigma in [0,1] fraction of grid size.
    """
    us = np.linspace(0.0, 1.0, grid_u + 1)
    vs = np.linspace(0.0, 1.0, grid_v + 1)
    U, V = np.meshgrid(us, vs)
    return amplitude * np.exp(-((U - cx)**2 + (V - cy)**2) / (2 * sigma**2))
