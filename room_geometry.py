"""room_geometry.py
===================
Procedural geometry for the walk-through room environment.

The room is a simple box: floor, ceiling, and four walls.  All surfaces
carry interleaved [x, y, z, nx, ny, nz] vertex data (stride 24, float32).

Coordinate convention
---------------------
* +X right, +Y depth (forward into room), +Z up.
* The room occupies  −width/2 ≤ x ≤ +width/2 (centred at x=0)
                      0 ≤ y ≤ depth   (entrance at y=0, back wall at y=depth)
                      0 ≤ z ≤ height  (floor at z=0, ceiling at z=height)

Public API
----------
``build_room_mesh(cfg)``
    Build all room surfaces from a config dict (room.yaml contents).
    Returns a dict with keys ``floor``, ``ceiling``, ``walls``, ``grid_lines``.

``build_room_floor_grid(cfg)``
    Build the decorative floor-grid line overlay.
    Returns float32 (-1, 3) [x,y,z] for GL_LINES.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quad(p00, p01, p10, p11) -> list:
    """Two CCW triangles from four [x,y,z,nx,ny,nz] rows."""
    return [p00, p10, p11, p00, p11, p01]


def _vn(pos: Sequence, n: Sequence) -> list:
    return [float(pos[0]), float(pos[1]), float(pos[2]),
            float(n[0]),   float(n[1]),   float(n[2])]


def _pack(rows: list) -> np.ndarray:
    return np.array(rows, np.float32).reshape(-1, 6)


def _pack3(rows: list) -> np.ndarray:
    return np.array(rows, np.float32).reshape(-1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Surface builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_floor(x0: float, x1: float, y0: float, y1: float, z: float) -> np.ndarray:
    """Single flat rectangle at height z, normal (0,0,+1)."""
    n = [0.0, 0.0, 1.0]
    rows = _quad(
        _vn([x0, y0, z], n), _vn([x1, y0, z], n),
        _vn([x0, y1, z], n), _vn([x1, y1, z], n),
    )
    return _pack(rows)


def _build_ceiling(x0: float, x1: float, y0: float, y1: float, z: float) -> np.ndarray:
    """Single flat rectangle at height z, normal (0,0,−1)."""
    n = [0.0, 0.0, -1.0]
    rows = _quad(
        _vn([x0, y1, z], n), _vn([x1, y1, z], n),
        _vn([x0, y0, z], n), _vn([x1, y0, z], n),
    )
    return _pack(rows)


def _build_xwall(x: float, y0: float, y1: float, z0: float, z1: float,
                 nx: float) -> np.ndarray:
    """Vertical wall with constant x, spanning Y (depth) and Z (height), outward normal (nx,0,0)."""
    n = [nx, 0.0, 0.0]
    rows = _quad(
        _vn([x, y0, z0], n), _vn([x, y1, z0], n),
        _vn([x, y0, z1], n), _vn([x, y1, z1], n),
    )
    return _pack(rows)


def _build_ywall(y: float, x0: float, x1: float, z0: float, z1: float,
                 ny: float) -> np.ndarray:
    """Vertical wall with constant y, spanning X (width) and Z (height), outward normal (0,ny,0)."""
    n = [0.0, ny, 0.0]
    rows = _quad(
        _vn([x0, y, z0], n), _vn([x1, y, z0], n),
        _vn([x0, y, z1], n), _vn([x1, y, z1], n),
    )
    return _pack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_room_mesh(cfg: dict) -> Dict[str, np.ndarray]:
    """Build all room surfaces.

    Parameters
    ----------
    cfg : dict
        Contents of ``configs/room_station/room.yaml``.

    Returns
    -------
    dict with keys:
        ``floor``   — float32 (-1,6)
        ``ceiling`` — float32 (-1,6)
        ``walls``   — float32 (-1,6)  (all 4 walls concatenated)
    """
    dims = cfg.get("dimensions", {})
    W = float(dims.get("width_m",  12.0))
    H = float(dims.get("height_m",  4.0))
    D = float(dims.get("depth_m",  10.0))
    t = float(dims.get("wall_thickness_m", 0.25))

    x0, x1 = -W / 2.0, W / 2.0   # X: left/right
    y0, y1 = 0.0, D               # Y: depth (entrance → back wall)
    z0, z1 = 0.0, H               # Z: height (floor → ceiling)

    floor   = _build_floor(x0, x1, y0, y1, z0)
    ceiling = _build_ceiling(x0, x1, y0, y1, z1)

    w_left  = _build_xwall(x0, y0, y1, z0, z1, -1.0)
    w_right = _build_xwall(x1, y0, y1, z0, z1,  1.0)
    w_near  = _build_ywall(y0, x0, x1, z0, z1, -1.0)
    w_far   = _build_ywall(y1, x0, x1, z0, z1,  1.0)
    walls   = np.concatenate([w_left, w_right, w_near, w_far], axis=0)

    return {
        "floor":   floor,
        "ceiling": ceiling,
        "walls":   walls,
    }


def build_room_floor_grid(cfg: dict) -> np.ndarray:
    """Build a GL_LINES grid overlay on the floor.

    Returns float32 (-1, 3) [x,y,z] — each consecutive pair of rows is
    one line segment (suitable for GL_LINES).

    Returns an empty array if ``floor_grid.enabled`` is false.
    """
    grid_cfg = cfg.get("floor_grid", {})
    if not grid_cfg.get("enabled", True):
        return np.zeros((0, 3), np.float32)

    dims    = cfg.get("dimensions", {})
    W       = float(dims.get("width_m",  12.0))
    D       = float(dims.get("depth_m",  10.0))
    spacing = float(grid_cfg.get("spacing_m", 1.0))
    z       = 0.001    # tiny lift off the floor to avoid z-fighting
    x0, x1  = -W / 2.0, W / 2.0
    y0, y1  = 0.0, D

    lines: list = []

    # Lines parallel to X axis (constant y/depth)
    y = y0
    while y <= y1 + 1e-6:
        lines += [[x0, y, z], [x1, y, z]]
        y += spacing

    # Lines parallel to Y axis (constant x)
    x = x0
    while x <= x1 + 1e-6:
        lines += [[x, y0, z], [x, y1, z]]
        x += spacing

    if not lines:
        return np.zeros((0, 3), np.float32)
    return np.array(lines, np.float32).reshape(-1, 3)


def room_bounds(cfg: dict):
    """Return (x0,x1, y0,y1, z0,z1) from a room config.

    In Z-up convention: y-axis is depth, z-axis is height.
    Returns (x0,x1, y0,y1=depth, z0,z1=height).
    """
    dims = cfg.get("dimensions", {})
    W = float(dims.get("width_m",  12.0))
    H = float(dims.get("height_m",  4.0))
    D = float(dims.get("depth_m",  10.0))
    return (-W/2, W/2, 0.0, D, 0.0, H)  # x0,x1, y0,y1(depth), z0,z1(height)
