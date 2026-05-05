from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Any

from dec_mesh import DECMesh


def _safe_module_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "fabricator_auto_" + "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in stem)


def _load_sibling_module(filename: str) -> ModuleType:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(_safe_module_name(path), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sibling blueprint module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_WORKBENCH_BP = getattr(_load_sibling_module("table_belljar_workbench.py"), "BLUEPRINT")


def factory(_params: dict[str, Any]) -> DECMesh:
    # Auto-sized: 0.25m x 0.25m footprint, total 1.0m tall, table + belljar.
    params = {
        "include_belljar": True,
        "auto_fit_belljar_to_table": False,
        "belljar_clearance_m": 0.0,
        "belljar_offset_x_m": 0.0,
        "belljar_offset_y_m": 0.0,
        "include_left_plug": False,
        "include_right_plug": False,
        "table__width_m": 0.25,
        "table__depth_m": 0.25,
        "table__height_m": 0.24,
        "table__include_top": True,
        "table__top_thickness_m": 0.02,
        "table__base_style": "hollow_box",
        "table__box_wall_thickness_m": 0.02,
        "table__box_has_bottom": False,
        "table__base_inset_m": 0.01,
        "belljar__width_m": 0.23,
        "belljar__depth_m": 0.23,
        "belljar__height_m": 0.76,
        "belljar__wall_thickness_m": 0.01,
        "belljar__skirt_height_m": 0.0,
    }
    return _WORKBENCH_BP["factory"](params)


BLUEPRINT = {
    "id": "network_contact_module_auto",
    "label": "Network Contact Module (Auto)",
    "knobspec": [],
    "factory": factory,
}
