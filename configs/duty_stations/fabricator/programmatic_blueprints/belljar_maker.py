from __future__ import annotations
from typing import Any

import numpy as np

from dec_mesh import DECMesh


def _box_faces(w: float, d: float, h: float, z0: float = 0.0,
               center_xy: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, list[list[int]]]:
    cx, cy = center_xy
    hx = 0.5 * float(w)
    hy = 0.5 * float(d)
    z1 = float(z0) + float(h)
    verts = np.array([
        [cx - hx, cy - hy, z0],
        [cx + hx, cy - hy, z0],
        [cx + hx, cy + hy, z0],
        [cx - hx, cy + hy, z0],
        [cx - hx, cy - hy, z1],
        [cx + hx, cy - hy, z1],
        [cx + hx, cy + hy, z1],
        [cx - hx, cy + hy, z1],
    ], np.float64)
    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return verts, faces


def _merge_raw(parts: list[tuple[np.ndarray, list[list[int]]]]) -> tuple[np.ndarray, list[list[int]]]:
    v_all = []
    f_all: list[list[int]] = []
    v_off = 0
    for verts, faces in parts:
        v_all.append(verts)
        for f in faces:
            f_all.append([int(v + v_off) for v in f])
        v_off += int(len(verts))
    if not v_all:
        return np.zeros((0, 3), np.float64), []
    return np.vstack(v_all), f_all


def _coerce(params: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def factory(params: dict[str, Any]) -> DECMesh:
    width = max(0.20, _coerce(params, "width_m", 1.20))
    depth = max(0.20, _coerce(params, "depth_m", 1.20))
    height = max(0.20, _coerce(params, "height_m", 1.40))
    wall = max(0.01, min(0.20, _coerce(params, "wall_thickness_m", 0.05)))
    skirt_h = max(0.00, _coerce(params, "skirt_height_m", 0.12))

    outer_v, outer_f = _box_faces(width, depth, height, z0=0.0)

    # Inner cavity as open-bottom wall set + top, approximating belljar geometry.
    iw = max(0.05, width - 2.0 * wall)
    idp = max(0.05, depth - 2.0 * wall)
    ih = max(0.05, height - wall)
    inner_v, _inner_full = _box_faces(iw, idp, ih, z0=wall)
    # keep all inner faces except the bottom cap to leave jar open below
    inner_f = [
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]

    parts = [(outer_v, outer_f), (inner_v, inner_f)]

    if skirt_h > 1e-6:
        sw = width + 2.0 * wall
        sd = depth + 2.0 * wall
        skirt_v, skirt_f = _box_faces(sw, sd, skirt_h, z0=-skirt_h)
        parts.append((skirt_v, skirt_f))

    verts, faces = _merge_raw(parts)
    return DECMesh.from_raw(verts, faces)


BLUEPRINT = {
    "id": "belljar_maker",
    "label": "Belljar Maker",
    "knobspec": [
        {"name": "width_m", "label": "Width", "dtype": "float", "default": 1.20, "low": 0.20, "high": 4.0, "step": 0.05, "fmt": ".2f"},
        {"name": "depth_m", "label": "Depth", "dtype": "float", "default": 1.20, "low": 0.20, "high": 4.0, "step": 0.05, "fmt": ".2f"},
        {"name": "height_m", "label": "Height", "dtype": "float", "default": 1.40, "low": 0.20, "high": 5.0, "step": 0.05, "fmt": ".2f"},
        {"name": "wall_thickness_m", "label": "Wall", "dtype": "float", "default": 0.05, "low": 0.01, "high": 0.20, "step": 0.01, "fmt": ".2f"},
        {"name": "skirt_height_m", "label": "Skirt", "dtype": "float", "default": 0.12, "low": 0.00, "high": 1.0, "step": 0.02, "fmt": ".2f"},
    ],
    "factory": factory,
}
