"""glass_room.py
================
GlassRoom — geometry builder for the simulation bell-jar enclosure.

Provides:
  - ``GlassRoom`` — config-driven class with methods to build the belljar
    (4 side walls + top cap with rounded corners) and the opaque skirt
    (floor-to-belljar base with rounded corners).
  - ``_build_sim_belljar_world`` / ``_build_sim_skirt_world`` — the original
    free functions, preserved here so existing call-sites in demo_pluck_gl.py
    still work while new code uses the class-based interface.

Both the class methods and the free functions return interleaved float32 arrays
shaped ``(-1, 6)`` with columns ``[x, y, z, nx, ny, nz]``, ready for a VAO
with a 24-byte stride (``layout(location=0)`` pos, ``layout(location=1)`` norm).

Config keys (from configs/duty_stations/simulator/glass_room.yaml)
------------------------------------------------------------------
glass_thickness_m : float   wall thickness = bevel radius   (default 0.02)
bevel_segs        : int     segments per corner arc          (default 6)
show_wireframe    : bool    draw inner box wireframe?         (default True)
wireframe_color   : list    RGBA for wireframe line           (default blue)
glass_color       : list    RGBA for glass surfaces           (default blue-clear)
skirt_color       : list    RGBA for opaque skirt             (default dark)
ambient           : float   Phong ambient factor              (default 0.22)
spec_strength     : float   Phong specular strength           (default 0.65)
shininess         : float   Phong shininess exponent          (default 128)
grain             : float   surface grain strength            (default 0.06)
render_belljar    : bool    draw belljar?                     (default True)
render_skirt      : bool    draw skirt?                       (default True)
render_wireframe  : bool    draw wireframe overlay?           (default True)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (backward-compatible with demo_pluck_gl.py)
# ─────────────────────────────────────────────────────────────────────────────

_GLASS_THICKNESS_M = 0.02   # default 2 cm glass wall thickness


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builders (free functions — preserved for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sim_belljar_world(
    wb_min: np.ndarray,
    wb_max: np.ndarray,
    thickness: float = _GLASS_THICKNESS_M,
    bevel_segs: int = 6,
) -> np.ndarray:
    """Square glass bell-jar (4 side walls + top cap, open at bottom) with
    rounded vertical corners in world space.

    Each outer vertical edge is replaced by a quarter-cylinder arc of
    *bevel_segs* segments, radius = *thickness*.  The flat walls are trimmed
    to meet the arcs at their tangent points.  The top cap is a fan-tessellated
    rounded rectangle.

    Returns an interleaved float32 array shaped (-1, 6) with columns
    [x, y, z, nx, ny, nz], suitable for a VAO with
    ``[(0, 3, 24, 0), (1, 3, 24, 12)]`` attributes (stride 24 bytes).
    """
    t  = float(thickness)
    r  = t                               # bevel radius = wall thickness
    xi, xa = float(wb_min[0]), float(wb_max[0])
    yi, ya = float(wb_min[1]), float(wb_max[1])
    zi, za = float(wb_min[2]), float(wb_max[2])
    xo0, xo1 = xi - t, xa + t
    yo0, yo1 = yi - t, ya + t
    zt = za + t

    rows: list = []

    def qf(a, b, c, d, n):
        n3 = list(n)
        for tri in ((a, b, c), (a, c, d)):
            for v in tri:
                rows.append(list(v) + n3)

    def tf(a, b, c, n):
        n3 = list(n)
        for v in (a, b, c):
            rows.append(list(v) + n3)

    # Corner arc centres and angle ranges:
    #  SW: (xo0+r, yo0+r)  π → 3π/2
    #  SE: (xo1-r, yo0+r)  3π/2 → 2π
    #  NE: (xo1-r, yo1-r)  0 → π/2
    #  NW: (xo0+r, yo1-r)  π/2 → π
    corners = [
        (xo0 + r, yo0 + r, math.pi,           3 * math.pi / 2),
        (xo1 - r, yo0 + r, 3 * math.pi / 2,   2 * math.pi),
        (xo1 - r, yo1 - r, 0.0,               math.pi / 2),
        (xo0 + r, yo1 - r, math.pi / 2,       math.pi),
    ]

    def arc_pts_2d(cx, cy, a0, a1, n):
        return [
            (cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cy + r * math.sin(a0 + (a1 - a0) * i / n))
            for i in range(n + 1)
        ]

    # Outer side walls — flat sections between arc tangent points
    qf([xo0, yo0+r, zi], [xo0, yo1-r, zi], [xo0, yo1-r, zt], [xo0, yo0+r, zt], [-1, 0, 0])
    qf([xo1, yo1-r, zi], [xo1, yo0+r, zi], [xo1, yo0+r, zt], [xo1, yo1-r, zt], [+1, 0, 0])
    qf([xo1-r, yo0, zi], [xo0+r, yo0, zi], [xo0+r, yo0, zt], [xo1-r, yo0, zt], [0, -1, 0])
    qf([xo0+r, yo1, zi], [xo1-r, yo1, zi], [xo1-r, yo1, zt], [xo0+r, yo1, zt], [0, +1, 0])

    # Outer corner arc columns
    for cx, cy, a0, a1 in corners:
        pts = arc_pts_2d(cx, cy, a0, a1, bevel_segs)
        for i in range(bevel_segs):
            px0, py0 = pts[i]
            px1, py1 = pts[i + 1]
            amid = a0 + (a1 - a0) * (i + 0.5) / bevel_segs
            nx, ny = math.cos(amid), math.sin(amid)
            qf([px0, py0, zi], [px1, py1, zi], [px1, py1, zt], [px0, py0, zt], [nx, ny, 0])

    # Top cap — fan-tessellated rounded rectangle
    boundary_xy: list = []
    for cx, cy, a0, a1 in corners:
        pts = arc_pts_2d(cx, cy, a0, a1, bevel_segs)
        boundary_xy.extend(pts[:-1])
    n_b = len(boundary_xy)
    cx_cap = (xo0 + xo1) * 0.5
    cy_cap = (yo0 + yo1) * 0.5
    cap_n = [0, 0, 1]
    for i in range(n_b):
        px0, py0 = boundary_xy[i]
        px1, py1 = boundary_xy[(i + 1) % n_b]
        tf([cx_cap, cy_cap, zt], [px0, py0, zt], [px1, py1, zt], cap_n)

    # Inner side faces (visible from inside the jar)
    qf([xi, ya, zi], [xi, yi, zi], [xi, yi, za], [xi, ya, za], [+1., 0., 0.])
    qf([xa, yi, zi], [xa, ya, zi], [xa, ya, za], [xa, yi, za], [-1., 0., 0.])
    qf([xa, yi, zi], [xi, yi, zi], [xi, yi, za], [xa, yi, za], [0., +1., 0.])
    qf([xi, ya, zi], [xa, ya, zi], [xa, ya, za], [xi, ya, za], [0., -1., 0.])
    qf([xi, ya, za], [xa, ya, za], [xa, yi, za], [xi, yi, za], [0., 0., -1.])

    return np.ascontiguousarray(np.asarray(rows, np.float32).reshape(-1, 6))


def _build_sim_skirt_world(
    wb_min: np.ndarray,
    wb_max: np.ndarray,
    thickness: float = _GLASS_THICKNESS_M,
    bevel_segs: int = 6,
) -> np.ndarray:
    """Opaque skirt: four outer walls from the floor (Z = 0) up to the open
    bottom of the bell jar (Z = *wb_min*[2]), with rounded vertical corners.

    No centre floor panel is added, so the guitar base inside is never clipped.
    Returns the same interleaved (pos3|norm3) format as
    :func:`_build_sim_belljar_world`, or an empty array when the sim region
    already starts at or below the stage floor.
    """
    floor_z = 0.0
    z_top   = float(wb_min[2])
    if z_top <= floor_z + 1e-4:
        return np.zeros((0, 6), np.float32)

    t  = float(thickness)
    r  = t
    xi, xa = float(wb_min[0]), float(wb_max[0])
    yi, ya = float(wb_min[1]), float(wb_max[1])
    xo0, xo1 = xi - t, xa + t
    yo0, yo1 = yi - t, ya + t

    rows: list = []

    def qf(a, b, c, d, n):
        n3 = list(n)
        for tri in ((a, b, c), (a, c, d)):
            for v in tri:
                rows.append(list(v) + n3)

    corners = [
        (xo0 + r, yo0 + r, math.pi,           3 * math.pi / 2),
        (xo1 - r, yo0 + r, 3 * math.pi / 2,   2 * math.pi),
        (xo1 - r, yo1 - r, 0.0,               math.pi / 2),
        (xo0 + r, yo1 - r, math.pi / 2,       math.pi),
    ]

    def arc_pts_2d(cx, cy, a0, a1, n):
        return [
            (cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cy + r * math.sin(a0 + (a1 - a0) * i / n))
            for i in range(n + 1)
        ]

    # Flat wall sections
    qf([xo0, yo0+r, floor_z], [xo0, yo1-r, floor_z], [xo0, yo1-r, z_top], [xo0, yo0+r, z_top], [-1, 0, 0])
    qf([xo1, yo1-r, floor_z], [xo1, yo0+r, floor_z], [xo1, yo0+r, z_top], [xo1, yo1-r, z_top], [+1, 0, 0])
    qf([xo1-r, yo0, floor_z], [xo0+r, yo0, floor_z], [xo0+r, yo0, z_top], [xo1-r, yo0, z_top], [0, -1, 0])
    qf([xo0+r, yo1, floor_z], [xo1-r, yo1, floor_z], [xo1-r, yo1, z_top], [xo0+r, yo1, z_top], [0, +1, 0])

    # Floor cap — solid bottom panel at Z=0 facing downward
    qf([xo0+r, yo0, floor_z], [xo1-r, yo0, floor_z],
       [xo1-r, yo1, floor_z], [xo0+r, yo1, floor_z], [0, 0, -1])

    # Corner arc columns
    for cx, cy, a0, a1 in corners:
        pts = arc_pts_2d(cx, cy, a0, a1, bevel_segs)
        for i in range(bevel_segs):
            px0, py0 = pts[i]
            px1, py1 = pts[i + 1]
            amid = a0 + (a1 - a0) * (i + 0.5) / bevel_segs
            nx, ny = math.cos(amid), math.sin(amid)
            qf([px0, py0, floor_z], [px1, py1, floor_z],
               [px1, py1, z_top], [px0, py0, z_top], [nx, ny, 0])

    return np.ascontiguousarray(np.asarray(rows, np.float32).reshape(-1, 6))


def _build_sim_wireframe_world(
    wb_min: np.ndarray,
    wb_max: np.ndarray,
) -> np.ndarray:
    """24 position-only vertices (12 GL_LINES edges) tracing the inner sim box.

    Returns a flat float32 array shaped (-1, 3).
    """
    x0, y0, z0 = float(wb_min[0]), float(wb_min[1]), float(wb_min[2])
    x1, y1, z1 = float(wb_max[0]), float(wb_max[1]), float(wb_max[2])
    edges = [
        [x0, y0, z0], [x1, y0, z0],   [x1, y0, z0], [x1, y1, z0],
        [x1, y1, z0], [x0, y1, z0],   [x0, y1, z0], [x0, y0, z0],
        [x0, y0, z1], [x1, y0, z1],   [x1, y0, z1], [x1, y1, z1],
        [x1, y1, z1], [x0, y1, z1],   [x0, y1, z1], [x0, y0, z1],
        [x0, y0, z0], [x0, y0, z1],   [x1, y0, z0], [x1, y0, z1],
        [x1, y1, z0], [x1, y1, z1],   [x0, y1, z0], [x0, y1, z1],
    ]
    return np.asarray(edges, np.float32).reshape(-1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Class-based interface
# ─────────────────────────────────────────────────────────────────────────────

class GlassRoom:
    """Config-driven builder for the simulation enclosure geometry.

    Usage
    -----
    ::

        room = GlassRoom(cfg)          # cfg from glass_room.yaml
        jar  = room.belljar(wb_min, wb_max)    # (-1,6) float32
        skirt = room.skirt(wb_min, wb_max)     # (-1,6) float32
        wire  = room.wireframe(wb_min, wb_max) # (-1,3) float32

    All returned arrays are freshly allocated on each call; they are not
    cached, so you can freely pass them to ``glBufferData``.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.thickness  = float(cfg.get("glass_thickness_m", _GLASS_THICKNESS_M))
        self.bevel_segs = int(cfg.get("bevel_segs", 6))

        # Render-control flags
        self.render_belljar   = bool(cfg.get("render_belljar",   True))
        self.render_skirt     = bool(cfg.get("render_skirt",     True))
        self.render_wireframe = bool(cfg.get("render_wireframe", True))
        self.show_wireframe   = bool(cfg.get("show_wireframe",   True))

        # Visual parameters
        gc = cfg.get("glass_color",     [0.15, 0.55, 0.90, 0.28])
        sc = cfg.get("skirt_color",     [0.10, 0.12, 0.16, 0.85])
        wc = cfg.get("wireframe_color", [0.20, 0.80, 1.00, 0.60])
        self.glass_color     = tuple(float(v) for v in gc[:4])
        self.skirt_color     = tuple(float(v) for v in sc[:4])
        self.wireframe_color = tuple(float(v) for v in wc[:4])
        self.wireframe_width = float(cfg.get("wireframe_width", 1.4))

        self.ambient      = float(cfg.get("ambient",      0.22))
        self.spec_strength = float(cfg.get("spec_strength", 0.65))
        self.shininess    = float(cfg.get("shininess",    128.0))
        self.grain        = float(cfg.get("grain",        0.06))

    # ── Geometry ──────────────────────────────────────────────────────────────

    def belljar(self, wb_min: np.ndarray, wb_max: np.ndarray) -> np.ndarray:
        """Build the glass bell-jar geometry for the given world-space box."""
        return _build_sim_belljar_world(
            wb_min, wb_max,
            thickness=self.thickness,
            bevel_segs=self.bevel_segs,
        )

    def skirt(self, wb_min: np.ndarray, wb_max: np.ndarray) -> np.ndarray:
        """Build the opaque base skirt for the given world-space box."""
        return _build_sim_skirt_world(
            wb_min, wb_max,
            thickness=self.thickness,
            bevel_segs=self.bevel_segs,
        )

    def wireframe(self, wb_min: np.ndarray, wb_max: np.ndarray) -> np.ndarray:
        """Build the inner-box wireframe edge list."""
        return _build_sim_wireframe_world(wb_min, wb_max)

    # ── Convenience: build all three at once ──────────────────────────────────

    def build_all(
        self, wb_min: np.ndarray, wb_max: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(belljar, skirt, wireframe)`` honouring render-control flags.

        A component is ``None`` if its render flag is False.
        """
        jar   = self.belljar(wb_min, wb_max)   if self.render_belljar   else None
        sk    = self.skirt(wb_min, wb_max)      if self.render_skirt     else None
        wire  = self.wireframe(wb_min, wb_max)  if (self.render_wireframe
                                                    and self.show_wireframe) else None
        return jar, sk, wire

    def __repr__(self) -> str:
        return (f"GlassRoom(thickness={self.thickness}, "
                f"bevel_segs={self.bevel_segs})")


# ─────────────────────────────────────────────────────────────────────────────
# Enclosure — config-driven dispatch to enclosure_geometry.py
# ─────────────────────────────────────────────────────────────────────────────

try:
    from enclosure_geometry import (
        build_enclosure_mesh    as _build_enc_mesh,
        build_enclosure_wireframe as _build_enc_wire,
    )
    from placed_object import PlacedEnclosure as _PlacedEnclosure
    _HAS_ENCLOSURE_GEOMETRY = True
except ImportError:
    _HAS_ENCLOSURE_GEOMETRY = False


class Enclosure:
    """Config-driven enclosure that dispatches to enclosure_geometry.py.

    This is a thin wrapper around a :class:`placed_object.PlacedEnclosure`
    data-class.  It provides the same ``.mesh()`` / ``.wireframe()`` interface
    as :class:`GlassRoom`, but works for all enclosure shapes
    (rect, cylinder, sphere, tablet_rect, tablet_polar).

    Parameters
    ----------
    placed_enc : PlacedEnclosure
        The data-class instance that holds shape, dims, position, and colors.

    Examples
    --------
    ::

        enc = Enclosure(my_placed_enc)
        mesh_verts = enc.mesh()          # float32(-1,6)  [x,y,z,nx,ny,nz]
        wire_verts = enc.wireframe()     # float32(-1,3)  [x,y,z]
    """

    def __init__(self, placed_enc: "_PlacedEnclosure"):
        if not _HAS_ENCLOSURE_GEOMETRY:
            raise ImportError(
                "enclosure_geometry.py / placed_object.py are required for Enclosure")
        self._obj = placed_enc

    # ── Geometry ──────────────────────────────────────────────────────────────

    def mesh(self) -> np.ndarray:
        """Return the solid / glass mesh as float32(-1,6) ``[x,y,z,nx,ny,nz]``."""
        return _build_enc_mesh(self._obj)

    def wireframe(self) -> np.ndarray:
        """Return the wireframe edge list as float32(-1,3) ``[x,y,z]``."""
        return _build_enc_wire(self._obj)

    # ── Properties (delegated to PlacedEnclosure) ─────────────────────────────

    @property
    def shape(self) -> str:
        return self._obj.shape

    @property
    def glass_color(self) -> tuple:
        return self._obj.glass_color

    @property
    def wireframe_color(self) -> tuple:
        return self._obj.wireframe_color

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        shape: str,
        dims: dict,
        pos,
        yaw_deg: float = 0.0,
        glass_color=(0.15, 0.55, 0.90, 0.28),
        skirt_color=(0.10, 0.12, 0.16, 0.85),
        wireframe_color=(0.20, 0.80, 1.00, 0.60),
        label: str = "",
        obj_id: str = "",
        simulator: "dict | None" = None,
    ) -> "Enclosure":
        """Build an Enclosure from individual parameters (no YAML needed).

        Parameters
        ----------
        shape : str
            One of ``'rect'``, ``'cylinder'``, ``'sphere'``,
            ``'tablet_rect'``, ``'tablet_polar'``.
        dims : dict
            Shape-specific dimensions; see enclosure_geometry.py docstring.
        pos : array-like (3,)
            World-space [x, y, z] position.
        yaw_deg : float
            Rotation about the Y axis in degrees.
        glass_color, skirt_color, wireframe_color : tuple (4,)
            RGBA colors (floats 0–1).
        label : str
            Human-readable name.
        obj_id : str
            Unique identifier string.
        simulator : dict or None
            Optional simulator config attached to the enclosure.
        """
        pos_arr = np.asarray(pos, dtype=np.float64)
        enc = _PlacedEnclosure(
            obj_id=obj_id or f"enc_{shape}",
            label=label or shape,
            pos=pos_arr,
            yaw_deg=float(yaw_deg),
            shape=shape,
            dims=dims,
            glass_color=tuple(float(v) for v in glass_color[:4]),
            skirt_color=tuple(float(v) for v in skirt_color[:4]),
            wireframe_color=tuple(float(v) for v in wireframe_color[:4]),
            simulator=simulator,
        )
        return cls(enc)

    def __repr__(self) -> str:
        return f"Enclosure(shape={self._obj.shape!r}, pos={self._obj.pos})"
