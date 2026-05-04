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
    off = 0
    for verts, faces in parts:
        v_all.append(verts)
        for f in faces:
            f_all.append([int(v + off) for v in f])
        off += int(len(verts))
    if not v_all:
        return np.zeros((0, 3), np.float64), []
    return np.vstack(v_all), f_all


def _coerce(params: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def factory(params: dict[str, Any]) -> DECMesh:
    span = max(0.20, _coerce(params, "span_m", 0.44))
    outer_d = max(0.02, _coerce(params, "outer_diameter_m", 0.08))
    inner_d = max(0.00, min(outer_d * 0.92, _coerce(params, "inner_diameter_m", 0.04)))
    rod_d = max(0.01, _coerce(params, "rod_diameter_m", 0.05))
    z = max(0.02, _coerce(params, "z_offset_m", 0.16))

    parts: list[tuple[np.ndarray, list[list[int]]]] = []

    # Left hollow connector represented as an outer shell and inner void body.
    outer_v, outer_f = _box_faces(span, outer_d, outer_d, z0=z - 0.5 * outer_d, center_xy=(-0.18, 0.0))
    inner_v, inner_f = _box_faces(span * 0.94, inner_d, inner_d, z0=z - 0.5 * inner_d, center_xy=(-0.18, 0.0))
    parts.append((outer_v, outer_f))
    if inner_d > 1e-6:
        parts.append((inner_v, inner_f))

    # Right solid connector.
    rod_v, rod_f = _box_faces(span, rod_d, rod_d, z0=z - 0.5 * rod_d, center_xy=(0.18, 0.0))
    parts.append((rod_v, rod_f))

    verts, faces = _merge_raw(parts)
    return DECMesh.from_raw(verts, faces)


BLUEPRINT = {
    "id": "duty_station_parametric_connector",
    "label": "Duty Station Parametric Connector",
    "knobspec": [
        {"name": "span_m", "label": "Span", "dtype": "float", "default": 0.44, "low": 0.20, "high": 1.20, "step": 0.02, "fmt": ".2f"},
        {"name": "outer_diameter_m", "label": "Pipe OD", "dtype": "float", "default": 0.08, "low": 0.02, "high": 0.20, "step": 0.01, "fmt": ".2f"},
        {"name": "inner_diameter_m", "label": "Pipe ID", "dtype": "float", "default": 0.04, "low": 0.00, "high": 0.16, "step": 0.01, "fmt": ".2f"},
        {"name": "rod_diameter_m", "label": "Rod D", "dtype": "float", "default": 0.05, "low": 0.01, "high": 0.20, "step": 0.01, "fmt": ".2f"},
        {"name": "z_offset_m", "label": "Z Offset", "dtype": "float", "default": 0.16, "low": 0.02, "high": 0.60, "step": 0.01, "fmt": ".2f"},
    ],
    "factory": factory,
}
