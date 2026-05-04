from __future__ import annotations

from typing import Any

import numpy as np

from dec_mesh import DECMesh


def _coerce(params: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def _coerce_bool(params: dict[str, Any], key: str, default: bool) -> bool:
    try:
        return bool(params.get(key, default))
    except Exception:
        return bool(default)


def _box_faces(
    w: float,
    d: float,
    h: float,
    z0: float = 0.0,
    center_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, list[list[int]]]:
    cx, cy = center_xy
    hx = 0.5 * float(w)
    hy = 0.5 * float(d)
    z1 = float(z0) + float(h)
    verts = np.array(
        [
            [cx - hx, cy - hy, z0],
            [cx + hx, cy - hy, z0],
            [cx + hx, cy + hy, z0],
            [cx - hx, cy + hy, z0],
            [cx - hx, cy - hy, z1],
            [cx + hx, cy - hy, z1],
            [cx + hx, cy + hy, z1],
            [cx - hx, cy + hy, z1],
        ],
        np.float64,
    )
    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return verts, faces


def _cylinder_faces(
    radius: float,
    height: float,
    z0: float = 0.0,
    center_xy: tuple[float, float] = (0.0, 0.0),
    segments: int = 16,
) -> tuple[np.ndarray, list[list[int]]]:
    cx, cy = center_xy
    seg = max(8, int(segments))
    r = max(1e-6, float(radius))
    z1 = float(z0) + float(height)

    bottom = []
    top = []
    for i in range(seg):
        a = (2.0 * np.pi * float(i)) / float(seg)
        x = cx + r * float(np.cos(a))
        y = cy + r * float(np.sin(a))
        bottom.append([x, y, z0])
        top.append([x, y, z1])

    verts = np.asarray(bottom + top, dtype=np.float64)
    faces: list[list[int]] = []

    # caps
    faces.append(list(range(seg)))
    faces.append(list(range(2 * seg - 1, seg - 1, -1)))

    # sides
    for i in range(seg):
        j = (i + 1) % seg
        faces.append([i, j, seg + j, seg + i])

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


def factory(params: dict[str, Any]) -> DECMesh:
    width = max(0.30, _coerce(params, "width_m", 1.40))
    depth = max(0.30, _coerce(params, "depth_m", 0.90))
    height = max(0.40, _coerce(params, "height_m", 1.00))

    include_top = _coerce_bool(params, "include_top", True)
    top_thickness = max(0.01, _coerce(params, "top_thickness_m", 0.08))

    base_style = str(params.get("base_style", "cylinder_pillars"))
    base_inset = max(0.00, _coerce(params, "base_inset_m", 0.10))

    leg_size = max(0.03, _coerce(params, "leg_size_m", 0.08))
    pillar_radius = max(0.03, _coerce(params, "pillar_radius_m", 0.09))
    pillar_count = int(round(_coerce(params, "pillar_count", 2)))
    pillar_count = 4 if pillar_count >= 3 else 2

    box_wall = max(0.01, _coerce(params, "box_wall_thickness_m", 0.05))
    box_bottom = _coerce_bool(params, "box_has_bottom", False)

    # Material knobs are intentionally carried as parameters for workflow/UI
    # but do not alter geometry in this blueprint.
    _ = params.get("top_material", "glass")
    _ = params.get("base_material", "metal")
    _ = params.get("inner_face_material", "smokey_glass")
    _ = params.get("outer_face_material", "glass")

    top_h = top_thickness if include_top else 0.0
    base_h = max(0.02, height - top_h)
    z_top = base_h

    parts: list[tuple[np.ndarray, list[list[int]]]] = []

    if include_top:
        parts.append(_box_faces(width, depth, top_thickness, z0=z_top))

    if base_style == "full_slab":
        parts.append(_box_faces(width, depth, base_h, z0=0.0))

    elif base_style == "hollow_box":
        hw = 0.5 * width
        hd = 0.5 * depth
        t = min(box_wall, hw - 0.02, hd - 0.02)
        t = max(0.01, t)

        # north/south walls
        parts.append(_box_faces(width, t, base_h, z0=0.0, center_xy=(0.0, hd - 0.5 * t)))
        parts.append(_box_faces(width, t, base_h, z0=0.0, center_xy=(0.0, -hd + 0.5 * t)))
        # east/west walls
        inner_d = max(0.04, depth - 2.0 * t)
        parts.append(_box_faces(t, inner_d, base_h, z0=0.0, center_xy=(hw - 0.5 * t, 0.0)))
        parts.append(_box_faces(t, inner_d, base_h, z0=0.0, center_xy=(-hw + 0.5 * t, 0.0)))

        if box_bottom:
            parts.append(_box_faces(width - 2.0 * t, depth - 2.0 * t, t, z0=0.0, center_xy=(0.0, 0.0)))

    elif base_style == "square_legs":
        lx = max(0.05, 0.5 * width - base_inset)
        ly = max(0.05, 0.5 * depth - base_inset)
        leg_centers = [
            (-lx, -ly),
            (-lx, ly),
            (lx, -ly),
            (lx, ly),
        ]
        for cxy in leg_centers:
            parts.append(_box_faces(leg_size, leg_size, base_h, z0=0.0, center_xy=cxy))

    else:  # cylinder_pillars
        if pillar_count == 4:
            px = max(0.06, 0.5 * width - base_inset)
            py = max(0.06, 0.5 * depth - base_inset)
            centers = [(-px, -py), (-px, py), (px, -py), (px, py)]
        else:
            px = max(0.06, 0.25 * width)
            centers = [(-px, 0.0), (px, 0.0)]

        for cxy in centers:
            parts.append(_cylinder_faces(pillar_radius, base_h, z0=0.0, center_xy=cxy, segments=16))

    verts, faces = _merge_raw(parts)
    return DECMesh.from_raw(verts, faces)


BLUEPRINT = {
    "id": "table_maker",
    "label": "Table Maker",
    "knobspec": [
        {"name": "width_m", "label": "Width", "dtype": "float", "default": 1.40, "low": 0.30, "high": 6.00, "step": 0.05, "fmt": ".2f"},
        {"name": "depth_m", "label": "Depth", "dtype": "float", "default": 0.90, "low": 0.30, "high": 6.00, "step": 0.05, "fmt": ".2f"},
        {"name": "height_m", "label": "Height", "dtype": "float", "default": 1.00, "low": 0.40, "high": 2.50, "step": 0.02, "fmt": ".2f"},
        {"name": "include_top", "label": "Include Top", "dtype": "bool", "default": True},
        {"name": "top_thickness_m", "label": "Top Thickness", "dtype": "float", "default": 0.08, "low": 0.01, "high": 0.40, "step": 0.01, "fmt": ".2f"},
        {"name": "base_style", "label": "Base Style", "dtype": "choice", "default": "cylinder_pillars", "choices": ["cylinder_pillars", "square_legs", "full_slab", "hollow_box"]},
        {"name": "pillar_count", "label": "Pillar Count", "dtype": "int", "default": 2, "low": 2, "high": 4, "step": 2, "fmt": ".0f"},
        {"name": "pillar_radius_m", "label": "Pillar Radius", "dtype": "float", "default": 0.09, "low": 0.03, "high": 0.40, "step": 0.01, "fmt": ".2f"},
        {"name": "leg_size_m", "label": "Leg Size", "dtype": "float", "default": 0.08, "low": 0.03, "high": 0.30, "step": 0.01, "fmt": ".2f"},
        {"name": "base_inset_m", "label": "Base Inset", "dtype": "float", "default": 0.10, "low": 0.00, "high": 1.20, "step": 0.01, "fmt": ".2f"},
        {"name": "box_wall_thickness_m", "label": "Box Wall", "dtype": "float", "default": 0.05, "low": 0.01, "high": 0.30, "step": 0.01, "fmt": ".2f"},
        {"name": "box_has_bottom", "label": "Box Has Bottom", "dtype": "bool", "default": False},
        {"name": "top_material", "label": "Top Material", "dtype": "choice", "default": "glass", "choices": ["glass", "smokey_glass", "metal", "wood", "composite"]},
        {"name": "base_material", "label": "Base Material", "dtype": "choice", "default": "metal", "choices": ["metal", "wood", "composite", "glass", "smokey_glass"]},
        {"name": "inner_face_material", "label": "Inner Face Material", "dtype": "choice", "default": "smokey_glass", "choices": ["smokey_glass", "glass", "metal", "wood", "composite"]},
        {"name": "outer_face_material", "label": "Outer Face Material", "dtype": "choice", "default": "glass", "choices": ["glass", "smokey_glass", "metal", "wood", "composite"]},
    ],
    "factory": factory,
}
