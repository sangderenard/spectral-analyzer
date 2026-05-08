from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np


HULL_WALL_TYPES = ("flat", "cylindrical")
HULL_CORNER_TYPES = ("flat", "spherical")


def choice_index(value: Any, choices: Iterable[str], default: int = 0) -> int:
    opts = tuple(choices)
    if str(value) in opts:
        return opts.index(str(value))
    try:
        idx = int(value)
    except Exception:
        idx = int(default)
    return max(0, min(idx, len(opts) - 1))


def _rect_cell_poly(gx: int, gy: int) -> list[tuple[float, float]]:
    return [
        (float(gx), float(gy)),
        (float(gx + 1), float(gy)),
        (float(gx + 1), float(gy + 1)),
        (float(gx), float(gy + 1)),
    ]


def _normalise_floor_type(floor_plan: Optional[dict], state: dict) -> str:
    if isinstance(floor_plan, dict) and floor_plan.get("floor_type"):
        return str(floor_plan.get("floor_type"))
    raw = state.get("floor_type", "rect")
    names = ("rect", "polar", "polar_rect", "arc")
    try:
        idx = int(raw)
        return names[idx] if 0 <= idx < len(names) else "rect"
    except Exception:
        return str(raw)


def _descriptor_common(state: dict) -> dict:
    voxel_side = max(1e-6, float(state.get("voxel_side_m", state.get("cell_size_m", 1.0)) or 1.0))
    wall_height = max(voxel_side, float(state.get("wall_height", 3.0) or 3.0))
    level_min = int(math.floor(float(state.get("wall_level_min", 0) or 0)))
    level_count = max(1, int(math.ceil(wall_height / voxel_side)))
    return {
        "coordinate_space": "room_plan_meters",
        "voxel_side_m": voxel_side,
        "level_range": (level_min, level_min + level_count - 1),
        "z_range_m": (level_min * voxel_side, level_min * voxel_side + wall_height),
        "height_m": wall_height,
        "occlusion_role": "structure",
    }


def _finalise_descriptors(footprints: list[dict], state: dict) -> list[dict]:
    common = _descriptor_common(state)
    out: list[dict] = []
    for raw in footprints:
        fp = dict(common)
        fp.update(raw)
        fp.setdefault("occluded_cells", [])
        fp.setdefault("occluded_voxels", [
            (int(cell[0]), int(cell[1]), level)
            for cell in fp.get("occluded_cells", []) or []
            if isinstance(cell, (list, tuple)) and len(cell) >= 2
            for level in range(int(fp["level_range"][0]), int(fp["level_range"][1]) + 1)
        ])
        out.append(fp)
    return out


def _wall_segment_polygon(
    p0: tuple[float, float],
    p1: tuple[float, float],
    inward: tuple[float, float],
    thickness: float,
) -> list[tuple[float, float]]:
    ix, iy = inward
    return [
        p0,
        p1,
        (p1[0] + ix * thickness, p1[1] + iy * thickness),
        (p0[0] + ix * thickness, p0[1] + iy * thickness),
    ]


def _radial_boundary_segments(floor_plan: dict, state: dict) -> list[dict]:
    ft = str(floor_plan.get("floor_type", "polar"))
    radius = max(1.0, float(floor_plan.get("floor_radius", state.get("floor_radius", 8.0)) or 8.0))
    if ft == "polar_rect":
        segs = max(2, int(state.get("floor_polar_rays", floor_plan.get("angular_segments", 24)) or 24))
        return [
            {
                "kind": "radial_chord",
                "surface": "outer",
                "segment": i,
                "polyline": [
                    (radius * math.cos(math.tau * i / segs), radius * math.sin(math.tau * i / segs)),
                    (radius * math.cos(math.tau * (i + 1) / segs), radius * math.sin(math.tau * (i + 1) / segs)),
                ],
            }
            for i in range(segs)
        ]

    cells = [
        c for c in floor_plan.get("cells", [])
        if isinstance(c, dict) and int(c.get("ring", -1)) == int(floor_plan.get("radial_segments", 1)) - 1
    ]
    out: list[dict] = []
    for c in cells:
        corners = c.get("tile_corners") or c.get("corners")
        if not (isinstance(corners, list) and len(corners) >= 4):
            continue
        pts = [(float(p[0]), float(p[1])) for p in corners[:4] if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(pts) < 4:
            continue
        outer = sorted(pts, key=lambda p: math.hypot(p[0], p[1]))[-2:]
        outer.sort(key=lambda p: math.atan2(p[1], p[0]))
        out.append({
            "kind": "radial_chord",
            "surface": "outer",
            "ring": int(c.get("ring", 0)),
            "segment": int(c.get("segment", 0)),
            "polyline": outer,
        })
    return out


def build_wall_footprints(
    floor_plan: Optional[dict],
    state: dict,
    floor_result: Optional[np.ndarray] = None,
) -> list[dict]:
    """Describe wall footprints and occlusion volumes without changing tile layout.

    The descriptors are intentionally geometry records rather than renderer
    records.  Current callers mostly consume 2D plan polygons and cells, but
    every footprint also carries a voxel side length, level range, and z range
    so later non-planar floor and ceiling occlusion can share the same contract.
    """
    floor_plan = floor_plan or {}
    ft = _normalise_floor_type(floor_plan, state)
    footprints: list[dict] = []

    if ft == "rect":
        width = int(floor_plan.get("width", state.get("room_width_cells", 8)) or 8)
        depth = int(floor_plan.get("height", state.get("room_depth_cells", 8)) or 8)
        wall_t = max(0.05, float(state.get("wall_thickness", 0.16) or 0.16))
        edge_defs = [
            ("hull_n", "north", (0.0, float(depth)), (float(width), float(depth)), (0.0, -1.0)),
            ("hull_s", "south", (float(width), 0.0), (0.0, 0.0), (0.0, 1.0)),
            ("hull_e", "east", (float(width), 0.0), (float(width), float(depth)), (-1.0, 0.0)),
            ("hull_w", "west", (0.0, float(depth)), (0.0, 0.0), (1.0, 0.0)),
        ]
        for prefix, surface, p0, p1, inward in edge_defs:
            htype = HULL_WALL_TYPES[choice_index(state.get(f"{prefix}_type", 0), HULL_WALL_TYPES)]
            amount = max(0.0, float(state.get(f"{prefix}_amount", 0.0) or 0.0))
            kind = "rect_wall" if htype == "flat" else "cylindrical_wall"
            footprints.append({
                "kind": kind,
                "surface": surface,
                "amount": amount,
                "polyline": [p0, p1],
                "polygon": _wall_segment_polygon(p0, p1, inward, max(wall_t, amount if htype == "cylindrical" else wall_t)),
                "occluded_cells": [],
            })

        if floor_result is not None:
            for gy in range(int(floor_result.shape[0])):
                for gx in range(int(floor_result.shape[1])):
                    if int(floor_result[gy, gx]) == 7:
                        footprints.append({
                            "kind": "wall_occlusion_cell",
                            "surface": "hull",
                            "cell": (int(gx), int(gy)),
                            "polygon": _rect_cell_poly(gx, gy),
                            "occluded_cells": [(int(gx), int(gy))],
                        })

        corner_defs = [
            ("hull_ne", "north_east", float(width), float(depth)),
            ("hull_nw", "north_west", 0.0, float(depth)),
            ("hull_se", "south_east", float(width), 0.0),
            ("hull_sw", "south_west", 0.0, 0.0),
        ]
        for prefix, surface, cx, cy in corner_defs:
            ctype = HULL_CORNER_TYPES[choice_index(state.get(f"{prefix}_type", 0), HULL_CORNER_TYPES)]
            radius = max(0.0, float(state.get(f"{prefix}_radius", 0.0) or 0.0))
            if ctype != "spherical" or radius <= 0.0:
                continue
            footprints.append({
                "kind": "spherical_corner_wall",
                "surface": surface,
                "center": (cx, cy),
                "radius": radius,
                "vertical_radius": radius,
                "occluding": True,
            })
        return _finalise_descriptors(footprints, state)

    if ft in ("polar", "polar_rect", "arc"):
        radius = max(1.0, float(floor_plan.get("floor_radius", state.get("floor_radius", 8.0)) or 8.0))
        wall_mode = HULL_WALL_TYPES[choice_index(state.get("hull_n_type", 0), HULL_WALL_TYPES)]
        kind = "cylindrical_radial_wall" if wall_mode == "cylindrical" else "straight_chord_wall"
        segments = _radial_boundary_segments(floor_plan, state)
        for seg in segments:
            seg = dict(seg)
            seg["kind"] = kind if kind == "straight_chord_wall" else "cylindrical_radial_wall"
            seg["radius"] = radius
            seg["occluded_cells"] = []
            footprints.append(seg)
        for prefix in ("hull_ne", "hull_nw", "hull_se", "hull_sw"):
            ctype = HULL_CORNER_TYPES[choice_index(state.get(f"{prefix}_type", 0), HULL_CORNER_TYPES)]
            radius_v = max(0.0, float(state.get(f"{prefix}_radius", 0.0) or 0.0))
            if ctype == "spherical" and radius_v > 0.0:
                footprints.append({
                    "kind": "spherical_radial_wedge_wall",
                    "surface": "outer",
                    "footprint": "radius" if kind == "cylindrical_radial_wall" else "ngon_chords",
                    "radius": radius,
                    "vertical_radius": radius_v,
                    "segments": [dict(s) for s in segments],
                    "occluded_cells": [],
                })
                break
    return _finalise_descriptors(footprints, state)


def wall_occlusion_cells(footprints: Iterable[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for fp in footprints:
        for cell in fp.get("occluded_cells", []) or []:
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                cells.add((int(cell[0]), int(cell[1])))
    return cells


def wall_occlusion_voxels(footprints: Iterable[dict]) -> set[tuple[int, int, int]]:
    voxels: set[tuple[int, int, int]] = set()
    for fp in footprints:
        raw_voxels = fp.get("occluded_voxels", []) or []
        for voxel in raw_voxels:
            if isinstance(voxel, (list, tuple)) and len(voxel) >= 3:
                voxels.add((int(voxel[0]), int(voxel[1]), int(voxel[2])))
        if raw_voxels:
            continue
        lr = fp.get("level_range", (0, 0))
        try:
            z0, z1 = int(lr[0]), int(lr[1])
        except Exception:
            z0, z1 = 0, 0
        for cell in fp.get("occluded_cells", []) or []:
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                for level in range(z0, z1 + 1):
                    voxels.add((int(cell[0]), int(cell[1]), int(level)))
    return voxels
