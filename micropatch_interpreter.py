"""
micropatch_interpreter.py — Load micropatch (and flat voice) wrappers into a live AnalyticPatch.

Handles both payload shapes:
  - "voice" preset  : flat AnalyticVoice.to_dict() directly in payload
  - "micropatch"    : { source: {...}, lfos: [...], fm: {...}, am: {...}, edges: [...] }

Resolves preset_id references against the InstanceLibrary before constructing objects.
Wires FM/AM source_key, adds all keys to the RoutingGraph, and connects voice → __mix__
unless payload.edges overrides the topology.

Usage
-----
    from micropatch_interpreter import load_micropatch, apply_set_fields
    from preset_library import get_library

    lib = get_library()
    wrapper = lib.load("micropatch", "lead_bright_chirp_01")
    result = load_micropatch(wrapper, patch, library=lib)
    # result: {"voice_key": "abc123", "lfo_keys": ["def456"], "key_map": {...}}

    # Apply field edits from agent output
    voice = patch.voices[-1]
    apply_set_fields(voice, {
        "chirp.chirp_type":    "exponential",
        "chirp.f_delta_start": -80.0,
        "adsr.release":        0.15,
    })
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analytic_driver import AnalyticPatch
    from preset_library import InstanceLibrary, InstanceWrapper


# ---------------------------------------------------------------------------
# LFO shape name normalizer
# (spec addendum used lowercase/alternate names; map to actual TitleCase names)
# ---------------------------------------------------------------------------

_LFO_SHAPE_MAP: Dict[str, str] = {
    "sine":        "Sine",
    "triangle":    "Triangle",
    "sawtooth":    "Sawtooth",
    "saw_up":      "Sawtooth",   # no direction distinction in engine
    "saw_down":    "Sawtooth",
    "square":      "Square",
    # TitleCase pass-through (already correct)
    "Sine":        "Sine",
    "Triangle":    "Triangle",
    "Sawtooth":    "Sawtooth",
    "Square":      "Square",
}
_VALID_LFO_SHAPES = {"Sine", "Triangle", "Sawtooth", "Square"}

# Bucket B shapes — valid to declare, silently mapped to Sine
_BUCKET_B_SHAPES = {"noise", "sample_hold"}


def _normalize_lfo_shape(raw: str) -> str:
    normalized = _LFO_SHAPE_MAP.get(raw, raw)
    if normalized not in _VALID_LFO_SHAPES:
        return "Sine"
    return normalized


# ---------------------------------------------------------------------------
# resolve_source_dict
# ---------------------------------------------------------------------------

def _resolve_source_dict(
    source: Any,
    library: Optional["InstanceLibrary"],
) -> dict:
    """
    Accept either:
      - a full AnalyticVoice dict  (has "freq_hz" or "label" or "key" etc.)
      - {"preset_id": "voice_..._v1"}  →  look up in library and return its payload

    Returns a flat AnalyticVoice dict ready for AnalyticVoice.from_dict().
    """
    if not isinstance(source, dict):
        return {}

    preset_id = source.get("preset_id")
    if preset_id:
        if library is None:
            raise ValueError(
                f"micropatch references preset_id={preset_id!r} "
                f"but no library was provided."
            )
        ref = library.load("voice", preset_id)
        return dict(ref.payload)

    return source


# ---------------------------------------------------------------------------
# _make_lfo
# ---------------------------------------------------------------------------

def _make_lfo(entry: dict, library: Optional["InstanceLibrary"]):
    """
    Build a LFODefinition from a micropatch lfos[] entry.

    Handles:
      - inline definition  (rate_hz, shape, phase_offset, depth)
      - {"preset_id": "lfo_slow_sine_v1"}
      - legacy spec names: phase → phase_offset, amplitude → depth
    """
    from analytic_driver import LFODefinition

    # Preset reference?
    preset_id = entry.get("preset_id")
    if preset_id:
        if library is None:
            raise ValueError(
                f"micropatch LFO references preset_id={preset_id!r} "
                f"but no library was provided."
            )
        ref = library.load("lfo", preset_id)
        params = dict(ref.payload)
    else:
        params = entry

    lfo = LFODefinition()
    lfo.rate_hz      = float(params.get("rate_hz", 1.0))
    lfo.shape        = _normalize_lfo_shape(params.get("shape", "Sine"))
    # Accept both spec-addendum name ("phase") and actual name ("phase_offset")
    lfo.phase_offset = float(params.get("phase_offset", params.get("phase", 0.0)))
    # Accept both spec-addendum name ("amplitude") and actual name ("depth")
    lfo.depth        = float(params.get("depth", params.get("amplitude", 1.0)))

    label = params.get("label", entry.get("label", "LFO"))
    lfo.label = label

    return lfo


# ---------------------------------------------------------------------------
# load_micropatch
# ---------------------------------------------------------------------------

def load_micropatch(
    wrapper: "InstanceWrapper",
    patch: "AnalyticPatch",
    library: Optional["InstanceLibrary"] = None,
) -> dict:
    """
    Integrate a micropatch (or flat voice) wrapper into a live AnalyticPatch.

    Supports both payload shapes:
      - instance_type="voice"     : payload is a flat AnalyticVoice.to_dict()
      - instance_type="micropatch": payload has {source, lfos, fm, am, edges}

    Parameters
    ----------
    wrapper  : InstanceWrapper from preset_library
    patch    : live AnalyticPatch to integrate into
    library  : InstanceLibrary used to resolve preset_id references

    Returns
    -------
    dict with keys:
        "voice_key"  : str  — the voice's actual key in patch.voices
        "lfo_keys"   : list[str] — keys of any LFODefinitions added
        "key_map"    : dict  — {local_id → actual_key} for routing resolution
    """
    from analytic_driver import AnalyticVoice, LFODefinition, ModRouting

    result: dict = {"voice_key": None, "lfo_keys": [], "key_map": {}}

    itype   = wrapper.instance_type
    payload = wrapper.payload

    # ----------------------------------------------------------------
    # Determine source dict
    # ----------------------------------------------------------------
    if itype == "voice":
        # Flat voice preset — payload IS the AnalyticVoice dict
        source_dict = _resolve_source_dict(payload, library)
        lfos_spec   = []
        fm_spec     = None
        am_spec     = None
        edges_spec  = []
        feedback_spec = None

    elif itype in ("micropatch", "patch"):
        source_raw    = payload.get("source", {})
        source_dict   = _resolve_source_dict(source_raw, library)
        lfos_spec     = payload.get("lfos", [])
        fm_spec       = payload.get("fm")
        am_spec       = payload.get("am")
        edges_spec    = payload.get("edges", [])
        feedback_spec = payload.get("feedback")

    else:
        raise ValueError(
            f"load_micropatch: unsupported instance_type={itype!r}. "
            f"Expected 'voice' or 'micropatch'."
        )

    # ----------------------------------------------------------------
    # Build AnalyticVoice
    # ----------------------------------------------------------------
    if not source_dict:
        raise ValueError("load_micropatch: no source dict found in payload.")

    # Allow agents to write "key": "auto" to request fresh key generation
    if source_dict.get("key") == "auto":
        source_dict = dict(source_dict)
        source_dict["key"] = uuid.uuid4().hex[:8]

    voice = AnalyticVoice.from_dict(source_dict)
    patch.voices.append(voice)

    result["voice_key"]         = voice.key
    result["key_map"]["source"] = voice.key

    # ----------------------------------------------------------------
    # Build LFODefinitions
    # ----------------------------------------------------------------
    lfo_by_id: Dict[str, LFODefinition] = {}

    for entry in lfos_spec:
        local_id = entry.get("id", uuid.uuid4().hex[:4])
        lfo = _make_lfo(entry, library)
        patch.lfos.append(lfo)
        result["lfo_keys"].append(lfo.key)
        result["key_map"][local_id] = lfo.key
        lfo_by_id[local_id] = lfo

    # ----------------------------------------------------------------
    # Wire FM
    # ----------------------------------------------------------------
    if fm_spec:
        lfo_key = _resolve_modrouting_key(fm_spec, lfo_by_id, library, result["key_map"])
        if lfo_key:
            depth_hz = float(fm_spec.get("depth_hz", 2.0))
            voice.fm = ModRouting(source_key=lfo_key, depth_hz=depth_hz, depth_amp=0.0)

    # ----------------------------------------------------------------
    # Wire AM
    # ----------------------------------------------------------------
    if am_spec:
        lfo_key = _resolve_modrouting_key(am_spec, lfo_by_id, library, result["key_map"])
        if lfo_key:
            depth_amp = float(am_spec.get("depth_amp", 0.2))
            voice.am = ModRouting(source_key=lfo_key, depth_hz=0.0, depth_amp=depth_amp)

    # ----------------------------------------------------------------
    # Routing graph
    # ----------------------------------------------------------------
    mixer_key = "__mix__"
    if patch.mixers:
        mixer_key = patch.mixers[0].key  # typically "__mix__"

    patch.routing.add_node(voice.key)
    for lfo in lfo_by_id.values():
        patch.routing.add_node(lfo.key)

    if feedback_spec:
        _apply_feedback(patch.routing, feedback_spec)

    if edges_spec:
        _apply_edges(patch.routing, edges_spec, result["key_map"], mixer_key)
    else:
        # Default: voice → mixer at weight 1.0 (if not already wired)
        already = any(
            e.src_key == voice.key and e.dst_key == mixer_key
            for e in patch.routing.edges
        )
        if not already:
            patch.routing.set_weight(voice.key, mixer_key, 1.0)
            patch.routing.add_node(mixer_key)

    return result


# ---------------------------------------------------------------------------
# Modulation routing key resolver
# ---------------------------------------------------------------------------

def _resolve_modrouting_key(
    spec: dict,
    lfo_by_id: Dict[str, Any],
    library: Optional["InstanceLibrary"],
    key_map: dict,
) -> Optional[str]:
    """
    Resolve the actual engine key for an fm/am spec dict.

    Accepts:
      {"lfo_id": "lfo1", ...}       — local LFO id defined in payload.lfos
      {"preset_id": "lfo_slow_v1"} — library LFO to load on the fly
      {"source_key": "abc123"}      — direct key (already resolved)
    """
    from analytic_driver import LFODefinition

    # Direct key reference (already in engine format)
    if "source_key" in spec:
        return spec["source_key"]

    # Local LFO id
    if "lfo_id" in spec:
        lid = spec["lfo_id"]
        if lid in lfo_by_id:
            return lfo_by_id[lid].key
        if lid in key_map:
            return key_map[lid]

    # Library preset reference — load and add to patch implicitly
    if "preset_id" in spec and library is not None:
        from preset_library import get_library
        ref = library.load("lfo", spec["preset_id"])
        lfo = LFODefinition.from_dict(ref.payload)
        # Note: caller must add to patch.lfos if desired; here we just return key
        return lfo.key

    return None


# ---------------------------------------------------------------------------
# Edge application
# ---------------------------------------------------------------------------

def _apply_edges(routing, edges_spec: list, key_map: dict, mixer_key: str) -> None:
    """
    Apply payload.edges to the RoutingGraph.

    Translates local IDs ("source", "lfo1", "out") to actual engine keys.
    Special local IDs:
      "source" → voice key
      "out"    → mixer_key (__mix__)
      "__mix__" → mixer_key directly
    """
    for edge in edges_spec:
        src_local = edge.get("src", edge.get("from", ""))
        dst_local = edge.get("dst", edge.get("to",   ""))
        weight    = float(edge.get("w",    edge.get("gain",  1.0)))
        angle     = float(edge.get("angle", 0.0))
        delay     = float(edge.get("delay", 0.0))

        src_key = _resolve_local_id(src_local, key_map, mixer_key)
        dst_key = _resolve_local_id(dst_local, key_map, mixer_key)

        if src_key and dst_key:
            routing.add_node(src_key)
            routing.add_node(dst_key)
            e = routing._find_edge(src_key, dst_key)
            if e is not None:
                e.weight    = weight
                e.angle_rad = angle
                e.delay_s   = delay
            else:
                from routing_engine import RoutingEdge
                routing.edges.append(RoutingEdge(src_key, dst_key, weight, angle, delay))


def _resolve_local_id(local_id: str, key_map: dict, mixer_key: str) -> Optional[str]:
    if local_id in ("out", "__mix__"):
        return mixer_key
    if local_id in key_map:
        return key_map[local_id]
    # Already an engine key (8-char hex or similar)
    return local_id if local_id else None


# ---------------------------------------------------------------------------
# Feedback config application
# ---------------------------------------------------------------------------

def _apply_feedback(routing, spec: dict) -> None:
    fb = routing.feedback
    if "enabled" in spec:
        fb.enabled = bool(spec["enabled"])
    if "delay_s" in spec:
        fb.delay_s = float(spec["delay_s"])
    if "decay" in spec:
        fb.decay = float(spec["decay"])
    if "max_iterations" in spec:
        fb.max_iterations = int(spec["max_iterations"])
    if "ringdown_mode" in spec:
        fb.ringdown_mode = str(spec["ringdown_mode"])
    if "ringdown_max_s" in spec:
        fb.ringdown_max_s = float(spec["ringdown_max_s"])
    if "ringdown_threshold" in spec:
        fb.ringdown_threshold = float(spec["ringdown_threshold"])


# ---------------------------------------------------------------------------
# apply_set_fields
# ---------------------------------------------------------------------------

def apply_set_fields(obj: Any, fields: Dict[str, Any]) -> List[str]:
    """
    Apply dot-path field updates to any engine object.

    Paths correspond to knob `name` values from the manifest
    (e.g. "chirp.f_delta_start", "adsr.attack", "manifold_type").

    Parameters
    ----------
    obj    : AnalyticVoice or any nested object with dot-accessible attributes
    fields : {"dot.path": value, ...}

    Returns
    -------
    list of paths that could not be applied (unknown or type-error)
    """
    failed = []
    for path, value in fields.items():
        try:
            _set_nested(obj, path.split("."), value)
        except (AttributeError, TypeError, ValueError) as exc:
            failed.append(f"{path}: {exc}")
    return failed


def _set_nested(obj: Any, parts: List[str], value: Any) -> None:
    """Traverse dot-path parts on obj and set the final attribute."""
    for part in parts[:-1]:
        obj = getattr(obj, part)
    attr = parts[-1]
    # Preserve type of existing value where possible
    existing = getattr(obj, attr, None)
    if existing is not None and not isinstance(existing, (str, bool)):
        try:
            if isinstance(existing, int):
                value = int(value)
            elif isinstance(existing, float):
                value = float(value)
        except (TypeError, ValueError):
            pass
    setattr(obj, attr, value)


# ---------------------------------------------------------------------------
# Convenience: wrap_voice
# ---------------------------------------------------------------------------

def wrap_voice(
    voice,
    instance_id: str,
    label:       str = "",
    description: str = "",
    tags:        Optional[List[str]] = None,
    summary:     Optional[dict] = None,
) -> "InstanceWrapper":
    """
    Wrap a live AnalyticVoice as a flat 'voice' InstanceWrapper (no modulation).
    Use InstanceLibrary.from_engine_object() for the full version with type detection.
    """
    from preset_library import InstanceWrapper
    return InstanceWrapper(
        instance_type = "voice",
        instance_id   = instance_id,
        label         = label or voice.label,
        description   = description,
        tags          = tags or [],
        summary       = summary or {},
        payload       = voice.to_dict(),
    )


def wrap_micropatch(
    voice,
    instance_id:  str,
    label:        str = "",
    description:  str = "",
    tags:         Optional[List[str]] = None,
    summary:      Optional[dict] = None,
    lfos:         Optional[list] = None,
) -> "InstanceWrapper":
    """
    Wrap a live AnalyticVoice (plus optional LFODefinitions) as a 'micropatch' wrapper.

    Parameters
    ----------
    lfos : list of LFODefinition objects whose keys are referenced by voice.fm / voice.am
    """
    from preset_library import InstanceWrapper

    lfos_payload = []
    if lfos:
        for i, lfo in enumerate(lfos):
            d = lfo.to_dict()
            d["id"] = d.get("key", f"lfo{i}")
            lfos_payload.append(d)

    fm_payload = None
    if voice.fm:
        fm_payload = {
            "source_key": voice.fm.source_key,
            "depth_hz":   voice.fm.depth_hz,
        }

    am_payload = None
    if voice.am:
        am_payload = {
            "source_key": voice.am.source_key,
            "depth_amp":  voice.am.depth_amp,
        }

    payload = {
        "source": voice.to_dict(),
        "lfos":   lfos_payload,
        "fm":     fm_payload,
        "am":     am_payload,
        "edges":  [],
    }

    return InstanceWrapper(
        instance_type = "micropatch",
        instance_id   = instance_id,
        label         = label or voice.label,
        description   = description,
        tags          = tags or [],
        summary       = summary or {},
        payload       = payload,
    )
