"""
knob_manifest.py — Export KnobSpec metadata for all major engine classes.

Produces a JSON manifest per class listing every configurable field with
its type, bounds, choices, group, and visibility condition.  Agent prompts
can embed or reference these manifests to enumerate valid fields and values
without ever reading the source code.

Usage
-----
    # Export all manifests to agent_prompts/manifests/
    python knob_manifest.py

    # Or from code:
    from knob_manifest import export_all_manifests
    manifests = export_all_manifests("agent_prompts/manifests")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Single-knob serializer
# ---------------------------------------------------------------------------

def _knob_to_dict(k) -> dict:
    return {
        "name":         k.name,
        "label":        k.label,
        "dtype":        k.dtype,
        "default":      k.default,
        "low":          k.low,
        "high":         k.high,
        "step":         k.step,
        "unit":         k.unit,
        "choices":      list(k.choices),
        "is_log":       k.is_log,
        "group":        k.group,
        "fmt":          k.fmt,
        "visible_when": list(k.visible_when) if k.visible_when else None,
    }


# ---------------------------------------------------------------------------
# Class-level manifest builder
# ---------------------------------------------------------------------------

def export_knob_manifest(cls, instance_type: str, description: str = "") -> dict:
    """
    Export all KnobSpec entries from cls.knobs() as a JSON-serializable dict.

    The result has two views of the same data:
    - ``fields``: flat list in declaration order (use for completeness)
    - ``groups``: dict[group_name → list[field]] (use for UI/agent structure)
    """
    knobs = cls.knobs()
    groups: Dict[str, List[dict]] = {}
    fields: List[dict] = []
    for k in knobs:
        d = _knob_to_dict(k)
        fields.append(d)
        g = k.group or "_ungrouped"
        groups.setdefault(g, []).append(d)

    return {
        "instance_type": instance_type,
        "class_name":    cls.__name__,
        "description":   description,
        "field_count":   len(fields),
        "groups":        groups,
        "fields":        fields,
    }


# ---------------------------------------------------------------------------
# Export all known manifests
# ---------------------------------------------------------------------------

def export_all_manifests(output_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Build manifests for every class that advertises a knobs() classmethod.

    Parameters
    ----------
    output_dir:
        If provided, each manifest is written to
        ``{output_dir}/{instance_type}_manifest.json``.

    Returns
    -------
    dict mapping instance_type → manifest dict
    """
    from analytic_driver import AnalyticVoice, LFODefinition

    specs = [
        (AnalyticVoice, "voice",
         "Single analytic voice: oscillator, envelope, chirp, harmonics, FM/AM, granular emission"),
        (LFODefinition, "lfo",
         "Low-frequency oscillator: rate, shape, depth, phase offset"),
    ]

    try:
        from routing_engine import RoutingGraph
        specs.append((RoutingGraph, "routing_graph",
                       "Signal routing graph: feedback decay, latency compensation, edge weights"))
    except Exception:
        pass

    try:
        from granular_engine import GrainPopulationSpec
        specs.append((GrainPopulationSpec, "grain_population",
                       "Granular synthesis population: density, duration, pitch spread, coherence"))
    except Exception:
        pass

    manifests: Dict[str, dict] = {}
    for cls, itype, desc in specs:
        manifests[itype] = export_knob_manifest(cls, itype, desc)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for itype, manifest in manifests.items():
            path = os.path.join(output_dir, f"{itype}_manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

    return manifests


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    default_out = os.path.join(os.path.dirname(__file__), "agent_prompts", "manifests")
    out = sys.argv[1] if len(sys.argv) > 1 else default_out
    manifests = export_all_manifests(out)
    print(f"Exported {len(manifests)} manifests to {out}/")
    for itype, m in manifests.items():
        print(f"  {itype}: {m['field_count']} fields in "
              f"{len(m['groups'])} groups")
