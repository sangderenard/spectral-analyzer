from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from types import ModuleType
from typing import Any, Callable, Optional

from dec_mesh import DECMesh


@dataclass
class ProgrammaticKnobSpec:
    name: str
    label: str
    dtype: str = "float"      # float | int | bool | choice
    default: Any = 0.0
    low: float = 0.0
    high: float = 1.0
    step: float = 0.1
    choices: list[str] | None = None
    fmt: str = ".3f"


@dataclass
class ProgrammaticBlueprint:
    id: str
    label: str
    knobspec: list[ProgrammaticKnobSpec]
    factory: Callable[[dict[str, Any]], DECMesh]
    source_path: str


def _safe_module_name(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    out = []
    for ch in stem:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "fabricator_prog_" + "".join(out)


def _load_module(path: str) -> Optional[ModuleType]:
    try:
        name = _safe_module_name(path)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _coerce_knobspec(raw: Any) -> list[ProgrammaticKnobSpec]:
    out: list[ProgrammaticKnobSpec] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, ProgrammaticKnobSpec):
            out.append(item)
            continue
        if isinstance(item, dict):
            out.append(ProgrammaticKnobSpec(
                name=str(item.get("name", "")),
                label=str(item.get("label", item.get("name", ""))),
                dtype=str(item.get("dtype", "float")),
                default=item.get("default", 0.0),
                low=float(item.get("low", 0.0)),
                high=float(item.get("high", 1.0)),
                step=float(item.get("step", 0.1)),
                choices=[str(v) for v in item.get("choices", [])] if isinstance(item.get("choices", []), list) else None,
                fmt=str(item.get("fmt", ".3f")),
            ))
    return [k for k in out if k.name]


def _extract_blueprint(mod: ModuleType, source_path: str) -> Optional[ProgrammaticBlueprint]:
    # Contract A: module function get_blueprint() -> dict/object
    if hasattr(mod, "get_blueprint") and callable(getattr(mod, "get_blueprint")):
        try:
            obj = mod.get_blueprint()
        except Exception:
            obj = None
        if isinstance(obj, dict):
            bid = str(obj.get("id", ""))
            lbl = str(obj.get("label", bid))
            ks = _coerce_knobspec(obj.get("knobspec", []))
            fac = obj.get("factory")
            if bid and callable(fac):
                return ProgrammaticBlueprint(bid, lbl, ks, fac, source_path)

    # Contract B: module-level BLUEPRINT dict
    bp = getattr(mod, "BLUEPRINT", None)
    if isinstance(bp, dict):
        bid = str(bp.get("id", ""))
        lbl = str(bp.get("label", bid))
        ks = _coerce_knobspec(bp.get("knobspec", []))
        fac = bp.get("factory")
        if bid and callable(fac):
            return ProgrammaticBlueprint(bid, lbl, ks, fac, source_path)

    # Contract C: module-level attributes
    bid = str(getattr(mod, "BLUEPRINT_ID", ""))
    if bid:
        lbl = str(getattr(mod, "BLUEPRINT_LABEL", bid))
        ks = _coerce_knobspec(getattr(mod, "KNOBSPEC", []))
        fac = getattr(mod, "factory", None)
        if callable(fac):
            return ProgrammaticBlueprint(bid, lbl, ks, fac, source_path)

    return None


def load_programmatic_blueprints(folder: str) -> list[ProgrammaticBlueprint]:
    out: list[ProgrammaticBlueprint] = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        low = str(name).lower()
        if not low.endswith(".py"):
            continue
        if low.startswith("_"):
            continue
        path = os.path.join(folder, name)
        mod = _load_module(path)
        if mod is None:
            continue
        bp = _extract_blueprint(mod, path)
        if bp is None:
            continue
        out.append(bp)
    out.sort(key=lambda b: (b.label.lower(), b.id.lower()))
    return out


def default_knob_values(bp: ProgrammaticBlueprint) -> dict[str, Any]:
    vals: dict[str, Any] = {}
    for k in bp.knobspec:
        vals[k.name] = k.default
    return vals


def build_mesh(bp: ProgrammaticBlueprint, values: dict[str, Any]) -> DECMesh:
    return bp.factory(dict(values))
