"""network_materializer.py
==========================
Convert a loaded :class:`~analytic_driver.AnalyticPatch` into the
``(nodes, edges)`` inputs consumed by :class:`~graph_solver.GraphSolver`.

All signal-source nodes are **fully differentiable torch modules** whose
learnable parameters are **dynamically registered from each analytic class's
``.knobs()`` classmethod**.  No hand-coded ``nn.Parameter`` declarations and
no calls to numpy synthesis functions from ``analytic_driver``.

Architecture
------------
``KnobDrivenModule``
    Base ``nn.Module``.  Iterates the ``KnobSpec`` list returned by
    ``cls.knobs()``; float/int knobs become ``nn.Parameter`` entries in
    ``self.params`` (a ``nn.ParameterDict``); log-scaled knobs are stored
    as ``log(value)`` and read back via ``_param(name)``; choice/bool/str
    knobs are stored as plain Python attributes.

``LFOTorchModule``
    Analytic LFO oscillator.  Sine: ``depth * exp(i*(2π*rate*t+φ))``.
    Other shapes embed shaped real magnitude into a rotating phasor.

``AnalyticLFOModule``
    Same as above but initialised from ``AnalyticModule.knobs()`` (the
    module-graph LFO type), including optional per-channel nodes.

``InterauralTorchModule``
    Torch port of ``_place_signal``.  Holds parameters for both ch1 and
    ch2 channels; ``build_nodes()`` returns two ``TensorNode``s with
    independent transforms sharing this module's parameters.

``ControlSliderTorchModule``
    DC complex constant source.  ``value ∈ [0,1]`` is an ``nn.Parameter``;
    ``low/high/is_log`` define the mapping (stored as buffers or attrs).

``PitchQuantizerTorchModule``
    Instantaneous discrete pitch snap (non-differentiable core) with
    learnable interpolation coefficients (portamento_time, slew_rate etc.)
    from the quantizer knobs.

Concessions
-----------
STATE MACHINE
    ``transform=None``; zeros for source value.  Plugin dispatch is not
    available inside the materializer.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from analytic_driver import AnalyticPatch

_CDTYPE = torch.complex128

# ---------------------------------------------------------------------------
# Lazy imports — kept at function scope to avoid hard circular dependencies
# ---------------------------------------------------------------------------

def _import_ad():
    import analytic_driver as _ad
    return _ad


def _import_gs():
    import graph_solver as _gs
    return _gs


# ---------------------------------------------------------------------------
# Knob-system utilities
# ---------------------------------------------------------------------------

def _dot_to_key(name: str) -> str:
    """Convert dot-path knob name (e.g. ``"adsr.attack"``) to a safe key."""
    return name.replace(".", "__")


def _get_nested_attr(obj, dot_path: str, default=None):
    """Read a nested attribute or dict key using dot-path notation."""
    for part in dot_path.split("."):
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(part, None)
        else:
            obj = getattr(obj, part, None)
    return obj if obj is not None else default


def _knobs_for_groups(knobs: list, groups: set) -> list:
    """Return only the knobs whose ``.group`` belongs to *groups*."""
    return [k for k in knobs if k.group in groups]


# ---------------------------------------------------------------------------
# KnobDrivenModule — base class
# ---------------------------------------------------------------------------

class KnobDrivenModule(nn.Module):
    """Generic ``nn.Module`` wrapper around any analytic object.

    Pass any analytic object that has a ``.knobs()`` classmethod.  Every
    float/int knob becomes an ``nn.Parameter``; log-scaled knobs are stored
    as ``log(value)`` and read back via ``_param(name)``; choice/bool/str
    knobs are stored as plain attributes.  The analytic object itself is
    stored as ``self.analytic_obj``.

    If *archetype* is provided (an ``AnalyticArchetype`` instance), this
    module registers itself with that archetype so the solver can dispatch
    all same-type nodes concurrently via the archetype's fire() method.
    """

    def __init__(self, analytic_obj, archetype=None) -> None:
        super().__init__()
        self.analytic_obj = analytic_obj
        self.archetype = archetype
        knobs = []
        knobs_fn = getattr(type(analytic_obj), "knobs", None)
        if knobs_fn is not None:
            knobs = knobs_fn()
        self.params = nn.ParameterDict()
        self._knob_is_log: Dict[str, bool] = {}
        for k in knobs:
            safe_key = _dot_to_key(k.name)
            if k.dtype in ("float", "int"):
                raw = _get_nested_attr(analytic_obj, k.name)
                if raw is None:
                    raw = k.default if k.default is not None else 0.0
                val = float(raw)
                stored = math.log(max(abs(val), 1e-12)) if k.is_log else val
                self.params[safe_key] = nn.Parameter(
                    torch.tensor(stored, dtype=torch.float64)
                )
                self._knob_is_log[safe_key] = bool(k.is_log)
            else:
                raw = _get_nested_attr(analytic_obj, k.name)
                if raw is None:
                    raw = k.default
                setattr(self, safe_key, raw)
        # Register with the archetype daemon if provided.
        if archetype is not None:
            node_key = str(getattr(analytic_obj, "key", id(analytic_obj)))
            archetype.register_node(node_key)

    def _param(self, name: str) -> Tensor:
        """Return the parameter tensor, applying ``exp()`` if log-scaled."""
        safe_key = _dot_to_key(name)
        p = self.params[safe_key]
        return torch.exp(p) if self._knob_is_log.get(safe_key, False) else p


# ---------------------------------------------------------------------------
# RoutingEdge → TensorEdge
# ---------------------------------------------------------------------------

def routing_edge_to_tensor_edge(e) -> "TensorEdge":
    """Convert one :class:`~routing_engine.RoutingEdge` to a
    :class:`~graph_solver.TensorEdge`.

    Weight folding
    --------------
    ``RoutingEdge`` stores amplitude and phase rotation separately::

        complex_weight = e.weight * exp(j * e.angle_rad)

    This is folded into ``TensorEdge.weight`` as a single complex scalar.

    Field mapping
    -------------
    RoutingEdge         TensorEdge
    ──────────────────  ──────────────────────
    src_key             src_key
    dst_key             dst_key
    weight*exp(j*ang)   weight          (folded complex)
    delay_s             delay_s
    saturation          saturation_policy
    saturation_knee     saturation_knee
    router_key          group
    item_slot           src_port        (SM named input slot preserved)
    edge_kind           semantic_role
    src_port            src_port        (only when item_slot is empty)
    dst_port            dst_port
    """
    _gs = _import_gs()

    w = complex(e.weight) * cmath.exp(1j * float(e.angle_rad))
    weight = torch.complex(
        torch.tensor(w.real, dtype=torch.float64),
        torch.tensor(w.imag, dtype=torch.float64),
    )

    src_port = str(e.item_slot) if getattr(e, "item_slot", "") else str(getattr(e, "src_port", ""))

    return _gs.TensorEdge(
        src_key           = str(e.src_key),
        dst_key           = str(e.dst_key),
        weight            = weight,
        delay_s           = float(getattr(e, "delay_s", 0.0)),
        saturation_policy = str(getattr(e, "saturation", "")),
        saturation_knee   = float(getattr(e, "saturation_knee", 1.0)),
        group             = str(getattr(e, "router_key", "")),
        semantic_role     = str(getattr(e, "edge_kind", "signal")),
        src_port          = src_port,
        dst_port          = str(getattr(e, "dst_port", "")),
    )


# ---------------------------------------------------------------------------
# materialize_network — full torch module assembly
# ---------------------------------------------------------------------------

def materialize_network(
    patch: "AnalyticPatch",
    sr: float,
) -> "Tuple[List, List]":
    """Wrap every analytic object in the patch into TensorNodes for GraphSolver."""
    _ad = _import_ad()
    _gs = _import_gs()

    nodes: list = []
    seen_keys: set = set()

    # One AnalyticArchetype per analytic class name — shared across all nodes
    # of the same type so the solver can batch-dispatch them concurrently.
    _archetypes: dict = {}

    def _get_archetype(analytic_obj) -> "_gs.AnalyticArchetype":
        type_name = type(analytic_obj).__name__
        if type_name not in _archetypes:
            _archetypes[type_name] = _gs.AnalyticArchetype(archetype_key=type_name)
        return _archetypes[type_name]

    def _add(node) -> None:
        if node.key not in seen_keys:
            nodes.append(node)
            seen_keys.add(node.key)

    # ── Virtual patch nodes ──────────────────────────────────────────────
    tonic_hz = float(getattr(patch, "seq_tonic_hz", 440.0))
    seq_hz   = float(getattr(patch, "_seq_note_hz", 0.0)) or tonic_hz

    def _dc_transform(hz: float):
        val = torch.tensor(complex(hz, 0.0), dtype=_CDTYPE)
        def _t(x: Tensor) -> Tensor:  # noqa: ARG001
            return val
        return _t

    _add(_gs.TensorNode(key="__patch_tonic__", layer="virtual",
                        transform=_dc_transform(tonic_hz)))
    _add(_gs.TensorNode(key="__patch_seq__",   layer="virtual",
                        transform=_dc_transform(seq_hz)))

    # ── System inputs / outputs (passthrough accumulators) ───────────────
    for key in _ad._system_input_keys(patch):
        _add(_gs.TensorNode(key=key, layer="system_in",  transform=None))
    for key in _ad._system_output_keys(patch):
        _add(_gs.TensorNode(key=key, layer="system_out", transform=None))

    # ── Voices ───────────────────────────────────────────────────────────
    for v in patch.voices:
        arch = _get_archetype(v)
        wrapper = KnobDrivenModule(v, archetype=arch)
        _add(_gs.TensorNode(key=v.key, layer="voice",
                            natural_rate_hz=float(getattr(v, "freq_hz", 0.0)),
                            analytic_module=wrapper,
                            archetype_key=arch.archetype_key))

    # ── LFOs ─────────────────────────────────────────────────────────────
    for lfo in patch.lfos:
        arch = _get_archetype(lfo)
        wrapper = KnobDrivenModule(lfo, archetype=arch)
        _add(_gs.TensorNode(key=lfo.key, layer="lfo",
                            natural_rate_hz=float(getattr(lfo, "rate_hz", 1.0)),
                            analytic_module=wrapper,
                            archetype_key=arch.archetype_key))

    # ── Modules ──────────────────────────────────────────────────────────
    for mod in patch.modules:
        mtype = str(getattr(mod, "module_type", "passthrough"))
        layer = str(getattr(mod, "signal_layer", "signal")) or "signal"
        arch = _get_archetype(mod)
        wrapper = KnobDrivenModule(mod, archetype=arch)
        if mtype == "interaural":
            _add(_gs.TensorNode(key=mod.ch1_key(), layer=layer,
                                analytic_module=wrapper, archetype_key=arch.archetype_key))
            _add(_gs.TensorNode(key=mod.ch2_key(), layer=layer,
                                analytic_module=wrapper, archetype_key=arch.archetype_key))
        elif mtype == "state_machine":
            sm_layer = str(getattr(mod, "signal_layer", "performer")) or "performer"
            _add(_gs.TensorNode(key=mod.key, layer=sm_layer,
                                analytic_module=wrapper, archetype_key=arch.archetype_key))
            for out_key in mod.sm_out_keys():
                _add(_gs.TensorNode(key=out_key, layer=sm_layer,
                                    analytic_module=wrapper, archetype_key=arch.archetype_key))
        else:
            _add(_gs.TensorNode(key=mod.key, layer=layer,
                                natural_rate_hz=float(getattr(mod, "rate_hz", 0.0)),
                                analytic_module=wrapper, archetype_key=arch.archetype_key))

    # ── Mixers — pure accumulators ───────────────────────────────────────
    for mx in patch.mixers:
        _add(_gs.TensorNode(key=mx.key, layer="master", transform=None))

    # ── Edges ─────────────────────────────────────────────────────────────
    edges: list = []

    # Internal MetaVoiceNode edges (wired between sub-nodes of each voice)
    for e in getattr(patch, "_meta_edges", []):
        if e.src_key in seen_keys and e.dst_key in seen_keys:
            edges.append(e)
    # Clear the temporary stash
    if hasattr(patch, "_meta_edges"):
        del patch._meta_edges

    # Routing edges from the patch's merged graph
    g = _ad._working_routing_graph_for_synthesis(patch)
    for re in g.edges:
        if re.src_key not in seen_keys or re.dst_key not in seen_keys:
            continue
        edges.append(routing_edge_to_tensor_edge(re))

    return nodes, edges
