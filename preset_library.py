"""
preset_library.py — Instance save/load/clone system for all major engine types.

Every saveable element (voice, lfo, dynamics_program, …) is stored as an
InstanceWrapper: a thin JSON envelope around the engine object's to_dict()
payload, plus human-readable metadata and a semantic summary.

Usage
-----
    from preset_library import get_library, InstanceWrapper
    lib = get_library()

    # Save a voice
    from analytic_driver import AnalyticVoice
    voice = AnalyticVoice(); voice.label = "My Lead"
    wrapper = InstanceLibrary.from_engine_object(
        voice, instance_id="voice_my_lead_v1", label="My Lead",
        tags=["lead", "pure"], summary={"role": "lead", "register": "all"}
    )
    lib.save(wrapper)

    # Load it back
    w = lib.load("voice", "voice_my_lead_v1")
    voice2 = lib.to_engine_object(w)

    # Clone and modify
    w2 = lib.clone("voice", "voice_my_lead_v1", "voice_my_lead_v2", "My Lead v2")

    # Search
    results = lib.search(instance_type="voice", tags=["lead"])
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ENGINE_VERSION = "score-0.1"

VALID_TYPES = frozenset({
    "voice",          # flat AnalyticVoice payload
    "lfo",            # flat LFODefinition payload
    "dynamics_program",
    "improv_program",
    "rhythm_page",
    "routing_graph",
    "grain_population",
    "patch",          # full AnalyticPatch
    "micropatch",     # voice + lfos + fm/am wiring + optional routing edges
})

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")


# ---------------------------------------------------------------------------
# InstanceWrapper
# ---------------------------------------------------------------------------

@dataclass
class InstanceWrapper:
    """JSON-serializable envelope for any saved engine element."""
    instance_type:  str
    instance_id:    str
    label:          str               = ""
    description:    str               = ""
    tags:           List[str]         = field(default_factory=list)
    summary:        Dict[str, Any]    = field(default_factory=dict)
    engine_version: str               = ENGINE_VERSION
    created_at:     str               = ""
    payload:        Dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "instance_type":  self.instance_type,
            "instance_id":    self.instance_id,
            "label":          self.label,
            "description":    self.description,
            "tags":           list(self.tags),
            "summary":        dict(self.summary),
            "engine_version": self.engine_version,
            "created_at":     self.created_at,
            "payload":        dict(self.payload),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceWrapper":
        return cls(
            instance_type  = d.get("instance_type", ""),
            instance_id    = d.get("instance_id", ""),
            label          = d.get("label", ""),
            description    = d.get("description", ""),
            tags           = list(d.get("tags", [])),
            summary        = dict(d.get("summary", {})),
            engine_version = d.get("engine_version", ENGINE_VERSION),
            created_at     = d.get("created_at", ""),
            payload        = dict(d.get("payload", {})),
        )


# ---------------------------------------------------------------------------
# InstanceLibrary
# ---------------------------------------------------------------------------

class InstanceLibrary:
    """File-based library: one JSON file per instance, organized by type."""

    def __init__(self, library_dir: str = PRESET_DIR):
        self.library_dir = library_dir
        os.makedirs(library_dir, exist_ok=True)
        for t in VALID_TYPES:
            os.makedirs(os.path.join(library_dir, t), exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path(self, instance_type: str, instance_id: str) -> str:
        return os.path.join(self.library_dir, instance_type, f"{instance_id}.json")

    def _validate_type(self, instance_type: str) -> None:
        if instance_type not in VALID_TYPES:
            raise ValueError(
                f"Unknown instance_type: {instance_type!r}. "
                f"Valid types: {sorted(VALID_TYPES)}"
            )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def save(self, wrapper: InstanceWrapper, overwrite: bool = False) -> str:
        """Persist wrapper to disk. Returns the file path written."""
        self._validate_type(wrapper.instance_type)
        if not wrapper.created_at:
            wrapper.created_at = datetime.datetime.utcnow().isoformat()
        path = self._path(wrapper.instance_type, wrapper.instance_id)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"Instance already exists: {wrapper.instance_id}. "
                f"Pass overwrite=True to replace."
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapper.to_dict(), f, indent=2)
        return path

    def load(self, instance_type: str, instance_id: str) -> InstanceWrapper:
        """Load a single wrapper by type and id."""
        self._validate_type(instance_type)
        path = self._path(instance_type, instance_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No instance found: {instance_type}/{instance_id}"
            )
        with open(path, encoding="utf-8") as f:
            return InstanceWrapper.from_dict(json.load(f))

    def exists(self, instance_type: str, instance_id: str) -> bool:
        return os.path.exists(self._path(instance_type, instance_id))

    def delete(self, instance_type: str, instance_id: str) -> None:
        path = self._path(instance_type, instance_id)
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self, instance_type: str, source_id: str, new_id: str,
              new_label: str = "") -> InstanceWrapper:
        """
        Deep-copy source_id → new_id.
        The clone is saved immediately and returned.
        """
        src = self.load(instance_type, source_id)
        dst = InstanceWrapper(
            instance_type  = src.instance_type,
            instance_id    = new_id,
            label          = new_label or f"{src.label} (copy)",
            description    = src.description,
            tags           = list(src.tags),
            summary        = dict(src.summary),
            engine_version = src.engine_version,
            payload        = dict(src.payload),
        )
        self.save(dst)
        return dst

    # ------------------------------------------------------------------
    # List / search
    # ------------------------------------------------------------------

    def list(self, instance_type: Optional[str] = None) -> List[InstanceWrapper]:
        """Return all wrappers, optionally filtered by type."""
        results: List[InstanceWrapper] = []
        types = [instance_type] if instance_type else sorted(VALID_TYPES)
        for t in types:
            t_dir = os.path.join(self.library_dir, t)
            if not os.path.isdir(t_dir):
                continue
            for fname in sorted(os.listdir(t_dir)):
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(t_dir, fname), encoding="utf-8") as f:
                        results.append(InstanceWrapper.from_dict(json.load(f)))
                except Exception:
                    pass
        return results

    def search(self, instance_type: Optional[str] = None,
               tags: Optional[List[str]] = None,
               query: str = "") -> List[InstanceWrapper]:
        """Filter by type, tags (any match), and/or a text query."""
        results = self.list(instance_type)
        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set.intersection(r.tags)]
        if query:
            q = query.lower()
            results = [
                r for r in results
                if q in r.label.lower()
                or q in r.description.lower()
                or q in r.instance_id.lower()
            ]
        return results

    # ------------------------------------------------------------------
    # Engine object bridge
    # ------------------------------------------------------------------

    def to_engine_object(self, wrapper: InstanceWrapper):
        """Deserialize a wrapper's payload back to the native engine object."""
        t = wrapper.instance_type
        p = wrapper.payload
        if t == "voice":
            from analytic_driver import AnalyticVoice
            return AnalyticVoice.from_dict(p)
        if t == "lfo":
            from analytic_driver import LFODefinition
            return LFODefinition.from_dict(p)
        if t == "dynamics_program":
            from dynamics_engine import DynamicsProgram
            return DynamicsProgram.from_dict(p)
        if t == "improv_program":
            from improv_engine import ImprovProgram
            return ImprovProgram.from_dict(p)
        if t == "routing_graph":
            from routing_engine import RoutingGraph
            return RoutingGraph.from_dict(p)
        raise NotImplementedError(f"to_engine_object not implemented for {t!r}")

    @staticmethod
    def from_engine_object(
        obj,
        instance_id:  str,
        label:        str = "",
        description:  str = "",
        tags:         Optional[List[str]] = None,
        summary:      Optional[Dict[str, Any]] = None,
    ) -> "InstanceWrapper":
        """Wrap a live engine object into an InstanceWrapper (does not save)."""
        from analytic_driver import AnalyticVoice, LFODefinition
        _dp = _ip = _rg = None
        try:
            from dynamics_engine import DynamicsProgram as _dp
        except ImportError:
            pass
        try:
            from improv_engine import ImprovProgram as _ip
        except ImportError:
            pass
        try:
            from routing_engine import RoutingGraph as _rg
        except ImportError:
            pass

        if isinstance(obj, AnalyticVoice):
            itype = "voice"
        elif isinstance(obj, LFODefinition):
            itype = "lfo"
        elif _dp and isinstance(obj, _dp):
            itype = "dynamics_program"
        elif _ip and isinstance(obj, _ip):
            itype = "improv_program"
        elif _rg and isinstance(obj, _rg):
            itype = "routing_graph"
        else:
            raise TypeError(f"Unsupported object type: {type(obj).__name__}")

        return InstanceWrapper(
            instance_type = itype,
            instance_id   = instance_id,
            label         = label,
            description   = description,
            tags          = tags or [],
            summary       = summary or {},
            payload       = obj.to_dict(),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_library: Optional[InstanceLibrary] = None


def get_library(library_dir: str = PRESET_DIR) -> InstanceLibrary:
    """Return (or create) the global InstanceLibrary singleton."""
    global _library
    if _library is None or _library.library_dir != library_dir:
        _library = InstanceLibrary(library_dir)
    return _library
