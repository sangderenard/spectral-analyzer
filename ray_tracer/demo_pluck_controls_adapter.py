"""Adapters for migrating demo_pluck_gl control definitions.

This module converts legacy tuple-based control definitions used in
demo_pluck_gl into KnobSpec descriptors and station hierarchy nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .station_specs import StationControlNode


def _resolve_knob_spec() -> Any:
    try:
        from signal_generator_v2 import KnobSpec
        return KnobSpec
    except Exception:
        @dataclass
        class KnobSpec:  # type: ignore[no-redef]
            name: str
            label: str
            dtype: str = "float"
            default: Any = None
            low: float = 0.0
            high: float = 1.0
            step: float = 0.0
            unit: str = ""
            choices: list[str] | None = None
            is_log: bool = False
            group: str = ""
            fmt: str = ".3g"
            source_class: str = ""
            rebuild_layout: bool = False
            visible_when: tuple[str, str] | None = None

        return KnobSpec


def slider_defs_to_knobs(
    defs: Iterable[tuple[Any, ...]],
    *,
    group: str,
    source_class: str,
) -> list[Any]:
    KnobSpec = _resolve_knob_spec()
    out: list[Any] = []
    for row in defs:
        if len(row) < 7:
            continue
        key, label, lo, hi, default, is_log, live = row[:7]
        dtype = "float"
        if isinstance(default, int) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            dtype = "int"
        knob = KnobSpec(
            str(key),
            str(label),
            dtype,
            default,
            float(lo),
            float(hi),
            0.0,
            "",
            [],
            bool(is_log),
            group,
            ".4g",
            source_class,
            bool(not live),
            None,
        )
        out.append(knob)
    return out


def knobs_to_station_nodes(
    knobs: Iterable[Any],
    *,
    parent_key: str,
    raised_material: str = "control_raised",
) -> list[StationControlNode]:
    nodes: list[StationControlNode] = []
    for idx, knob in enumerate(knobs):
        key = f"{parent_key}.{getattr(knob, 'name', idx)}"
        label = str(getattr(knob, "label", getattr(knob, "name", key)))
        nodes.append(
            StationControlNode(
                key=key,
                label=label,
                knob=knob,
                children=[],
                depth_mode="raise",
                material_slot=raised_material,
                payload={"knob_name": getattr(knob, "name", key)},
            )
        )
    return nodes
