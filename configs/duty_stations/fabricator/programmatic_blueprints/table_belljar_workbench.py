from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Any

import numpy as np

from dec_mesh import DECMesh


def _safe_module_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "fabricator_comp_" + "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in stem)


def _load_sibling_module(filename: str) -> ModuleType:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    name = _safe_module_name(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load blueprint module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_child_blueprint(filename: str) -> dict[str, Any]:
    mod = _load_sibling_module(filename)
    bp = getattr(mod, "BLUEPRINT", None)
    if not isinstance(bp, dict):
        raise RuntimeError(f"Missing BLUEPRINT dict in {filename}")
    fac = bp.get("factory")
    if not callable(fac):
        raise RuntimeError(f"BLUEPRINT.factory is not callable in {filename}")
    return bp


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


def _extract_prefixed_params(params: dict[str, Any], prefix: str, child_bp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in child_bp.get("knobspec", []):
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", ""))
        if not name:
            continue
        full = f"{prefix}{name}"
        out[name] = params.get(full, spec.get("default"))
    return out


def _translate_mesh(mesh: DECMesh, dx: float, dy: float, dz: float) -> DECMesh:
    moved = np.array(mesh.verts, dtype=np.float64, copy=True)
    moved[:, 0] += float(dx)
    moved[:, 1] += float(dy)
    moved[:, 2] += float(dz)
    return DECMesh.from_raw(moved, [list(f) for f in mesh.faces])


def _merge_meshes(meshes: list[DECMesh]) -> DECMesh:
    if not meshes:
        return DECMesh.from_raw(np.zeros((0, 3), np.float64), [])
    verts_all: list[np.ndarray] = []
    faces_all: list[list[int]] = []
    off = 0
    for m in meshes:
        verts_all.append(np.asarray(m.verts, np.float64))
        for f in m.faces:
            faces_all.append([int(v + off) for v in f])
        off += int(len(m.verts))
    return DECMesh.from_raw(np.vstack(verts_all), faces_all)


_TABLE_BP = _get_child_blueprint("table_maker.py")
_BELLJAR_BP = _get_child_blueprint("belljar_maker.py")
_CONNECTOR_BP = _get_child_blueprint("duty_station_parametric_connector.py")


def factory(params: dict[str, Any]) -> DECMesh:
    table_params = _extract_prefixed_params(params, "table__", _TABLE_BP)
    belljar_params = _extract_prefixed_params(params, "belljar__", _BELLJAR_BP)
    plug_params = _extract_prefixed_params(params, "plug__", _CONNECTOR_BP)

    include_belljar = _coerce_bool(params, "include_belljar", True)
    auto_fit_belljar = _coerce_bool(params, "auto_fit_belljar_to_table", True)
    include_left_plug = _coerce_bool(params, "include_left_plug", False)
    include_right_plug = _coerce_bool(params, "include_right_plug", False)

    belljar_clearance = max(0.0, _coerce(params, "belljar_clearance_m", 0.01))
    belljar_offset_x = _coerce(params, "belljar_offset_x_m", 0.0)
    belljar_offset_y = _coerce(params, "belljar_offset_y_m", 0.0)

    plug_side_clearance = max(0.0, _coerce(params, "plug_side_clearance_m", 0.06))
    plug_y_offset = _coerce(params, "plug_y_offset_m", 0.0)
    plug_z_offset = _coerce(params, "plug_z_offset_m", 0.02)

    table_mesh = _TABLE_BP["factory"](table_params)

    table_w = max(0.30, float(table_params.get("width_m", 1.40)))
    table_d = max(0.30, float(table_params.get("depth_m", 0.90)))
    table_h = max(0.40, float(table_params.get("height_m", 1.00)))
    table_top_z = table_h

    pieces: list[DECMesh] = [table_mesh]

    if include_belljar:
        if auto_fit_belljar:
            belljar_params["width_m"] = max(0.20, min(4.0, table_w * 0.92))
            belljar_params["depth_m"] = max(0.20, min(4.0, table_d * 0.92))
        belljar_mesh = _BELLJAR_BP["factory"](belljar_params)
        belljar_mesh = _translate_mesh(
            belljar_mesh,
            belljar_offset_x,
            belljar_offset_y,
            table_top_z + belljar_clearance,
        )
        pieces.append(belljar_mesh)

    span = max(0.20, float(plug_params.get("span_m", 0.44)))
    plug_anchor_z = table_top_z + plug_z_offset

    if include_left_plug:
        left_mesh = _CONNECTOR_BP["factory"](plug_params)
        left_x = -(0.5 * table_w + plug_side_clearance + 0.5 * span)
        left_mesh = _translate_mesh(left_mesh, left_x, plug_y_offset, plug_anchor_z)
        pieces.append(left_mesh)

    if include_right_plug:
        right_mesh = _CONNECTOR_BP["factory"](plug_params)
        right_x = (0.5 * table_w + plug_side_clearance + 0.5 * span)
        right_mesh = _translate_mesh(right_mesh, right_x, plug_y_offset, plug_anchor_z)
        pieces.append(right_mesh)

    return _merge_meshes(pieces)


def _prefix_knobs(child_bp: dict[str, Any], prefix: str, group: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in child_bp.get("knobspec", []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", ""))
        if not name:
            continue
        spec = dict(raw)
        spec["name"] = f"{prefix}{name}"
        spec["label"] = f"{raw.get('label', name)}"
        spec["group"] = group
        spec["group_collapsible"] = True
        spec["group_default_expanded"] = True
        out.append(spec)
    return out


BLUEPRINT = {
    "id": "table_belljar_workbench",
    "label": "Table Belljar Workbench",
    "knobspec": [
        {
            "name": "include_belljar",
            "label": "Include Belljar",
            "dtype": "bool",
            "default": True,
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "auto_fit_belljar_to_table",
            "label": "Auto Fit Belljar To Table",
            "dtype": "bool",
            "default": True,
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "belljar_clearance_m",
            "label": "Belljar Clearance",
            "dtype": "float",
            "default": 0.01,
            "low": 0.0,
            "high": 1.0,
            "step": 0.01,
            "fmt": ".2f",
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "belljar_offset_x_m",
            "label": "Belljar Offset X",
            "dtype": "float",
            "default": 0.0,
            "low": -2.0,
            "high": 2.0,
            "step": 0.01,
            "fmt": ".2f",
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "belljar_offset_y_m",
            "label": "Belljar Offset Y",
            "dtype": "float",
            "default": 0.0,
            "low": -2.0,
            "high": 2.0,
            "step": 0.01,
            "fmt": ".2f",
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "include_left_plug",
            "label": "Include Left Duty Plug",
            "dtype": "bool",
            "default": False,
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "include_right_plug",
            "label": "Include Right Duty Plug",
            "dtype": "bool",
            "default": False,
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "plug_side_clearance_m",
            "label": "Plug Side Clearance",
            "dtype": "float",
            "default": 0.06,
            "low": 0.0,
            "high": 1.0,
            "step": 0.01,
            "fmt": ".2f",
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "plug_y_offset_m",
            "label": "Plug Y Offset",
            "dtype": "float",
            "default": 0.0,
            "low": -2.0,
            "high": 2.0,
            "step": 0.01,
            "fmt": ".2f",
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
        {
            "name": "plug_z_offset_m",
            "label": "Plug Z Offset",
            "dtype": "float",
            "default": 0.02,
            "low": -1.0,
            "high": 2.0,
            "step": 0.01,
            "fmt": ".2f",
            "group": "Composite Layout",
            "group_collapsible": True,
            "group_default_expanded": True,
        },
    ] + _prefix_knobs(_TABLE_BP, "table__", "Table Blueprint")
      + _prefix_knobs(_BELLJAR_BP, "belljar__", "Belljar Blueprint")
      + _prefix_knobs(_CONNECTOR_BP, "plug__", "Duty Plug Blueprint"),
    "factory": factory,
}
