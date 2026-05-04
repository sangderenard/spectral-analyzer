from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np

from dec_mesh import DECMesh

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _yaml = None
    _HAS_YAML = False


def _box_mesh(width: float, depth: float, height: float,
              center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> DECMesh:
    hx = max(1e-6, float(width)) * 0.5
    hy = max(1e-6, float(depth)) * 0.5
    hz = max(1e-6, float(height)) * 0.5
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))

    verts = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], np.float64)

    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return DECMesh.from_raw(verts, faces)


def _wedge_mesh(width: float, depth: float, height: float,
                top_tilt_deg: float = 14.0,
                center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> DECMesh:
    """Rectangular footprint with a top plane that rises toward +Y (operator-facing wedge)."""
    hx = max(1e-6, float(width)) * 0.5
    hy = max(1e-6, float(depth)) * 0.5
    h = max(1e-6, float(height))
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))

    tilt = math.radians(float(top_tilt_deg))
    dz = max(0.0, float(depth) * math.tan(tilt))
    z_front = cz - 0.5 * h
    z_back = cz + 0.5 * h + dz

    verts = np.array([
        [cx - hx, cy - hy, cz - 0.5 * h],
        [cx + hx, cy - hy, cz - 0.5 * h],
        [cx + hx, cy + hy, cz - 0.5 * h],
        [cx - hx, cy + hy, cz - 0.5 * h],
        [cx - hx, cy - hy, z_front],
        [cx + hx, cy - hy, z_front],
        [cx + hx, cy + hy, z_back],
        [cx - hx, cy + hy, z_back],
    ], np.float64)

    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return DECMesh.from_raw(verts, faces)


def _load_duty_station_cfg(path: str) -> dict:
    if not _HAS_YAML or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _yaml.safe_load(handle) or {}
    except Exception:
        return {}


def duty_station_primitive_meshes(station_cfg_path: str = "configs/meshes/duty_station.yaml") -> Dict[str, DECMesh]:
    cfg = _load_duty_station_cfg(station_cfg_path)
    cons = cfg.get("console", {}) if isinstance(cfg, dict) else {}
    scr = cfg.get("screen", {}) if isinstance(cfg, dict) else {}
    wings = cfg.get("side_wings", {}) if isinstance(cfg, dict) else {}
    wall = cfg.get("back_wall", {}) if isinstance(cfg, dict) else {}

    cons_w = float(cons.get("width", 1.40))
    cons_d = float(cons.get("depth", 0.62))
    cons_h = float(cons.get("height", 0.88))
    cons_tilt = float(cons.get("top_tilt_deg", 14.0))

    scr_w = float(scr.get("width", 1.10))
    scr_h = float(scr.get("height", 0.72))
    scr_t = 0.05

    wing_enabled = bool(wings.get("enabled", True))
    wing_w = float(wings.get("width", 0.14))
    wing_h = float(wings.get("height", 0.52))
    wing_d = max(0.08, cons_d * 0.90)

    wall_enabled = bool(wall.get("enabled", False))
    wall_w = float(wall.get("width", cons_w + (2.0 * wing_w if wing_enabled else 0.0)))
    wall_h = float(wall.get("height", 3.20))
    wall_t = float(wall.get("thickness", 0.05))

    out: Dict[str, DECMesh] = {
        "duty_console_body": _wedge_mesh(cons_w, cons_d, cons_h, top_tilt_deg=cons_tilt),
        "duty_screen_panel": _box_mesh(scr_w, scr_t, scr_h),
    }

    if wing_enabled:
        out["duty_left_wing"] = _box_mesh(wing_w, wing_d, wing_h)
        out["duty_right_wing"] = _box_mesh(wing_w, wing_d, wing_h)

    if wall_enabled:
        out["duty_back_wall"] = _box_mesh(wall_w, wall_t, wall_h)

    floor_w = max(cons_w + 2.0 * (wing_w if wing_enabled else 0.0) + 0.20, 0.80)
    floor_d = max(cons_d + 0.18, 0.80)
    floor_t = 0.10
    out["duty_floor_tile"] = _box_mesh(floor_w, floor_d, floor_t)
    out["duty_left_hollow_pipe"] = _box_mesh(0.36, 0.08, 0.08)
    out["duty_right_solid_rod"] = _box_mesh(0.38, 0.05, 0.05)
    out["duty_under_wing_plug"] = _box_mesh(0.06, 0.06, 0.14)
    return out


def duty_station_catalog_entries(station_cfg_path: str = "configs/meshes/duty_station.yaml") -> List[dict]:
    meshes = duty_station_primitive_meshes(station_cfg_path)
    entries: List[dict] = []
    labels = {
        "duty_console_body": "Duty Console Body",
        "duty_screen_panel": "Duty Screen Panel",
        "duty_left_wing": "Duty Left Wing",
        "duty_right_wing": "Duty Right Wing",
        "duty_back_wall": "Duty Back Wall",
        "duty_floor_tile": "Duty Floor Tile",
        "duty_left_hollow_pipe": "Duty Left Hollow Pipe",
        "duty_right_solid_rod": "Duty Right Solid Rod",
        "duty_under_wing_plug": "Duty Under-Wing Plug",
    }
    colors = {
        "duty_console_body": [0.20, 0.35, 0.58],
        "duty_screen_panel": [0.18, 0.66, 0.86],
        "duty_left_wing": [0.24, 0.56, 0.78],
        "duty_right_wing": [0.24, 0.56, 0.78],
        "duty_back_wall": [0.36, 0.42, 0.54],
        "duty_floor_tile": [0.26, 0.30, 0.38],
        "duty_left_hollow_pipe": [0.24, 0.62, 0.86],
        "duty_right_solid_rod": [0.62, 0.72, 0.82],
        "duty_under_wing_plug": [0.28, 0.58, 0.82],
    }

    for pid, mesh in meshes.items():
        bmin = mesh.verts.min(axis=0)
        bmax = mesh.verts.max(axis=0)
        ext = bmax - bmin
        subtitle = f"{ext[0]:.2f}m x {ext[1]:.2f}m x {ext[2]:.2f}m"
        entries.append({
            "id": pid,
            "label": labels.get(pid, pid.replace("_", " ").title()),
            "subtitle": subtitle,
            "kind": "duty_primitive",
            "icon": "[]",
            "color_rgb": colors.get(pid, [0.4, 0.4, 0.4]),
        })
    return entries
