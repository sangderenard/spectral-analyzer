"""routing_engine.py

Standalone analytic-signal routing engine.

RoutingEdge, FeedbackConfig, RoutingGraph, solve_routing_complex, and RoutingMixer
are all independent of the GUI / AnalyticPatch system and work with any numpy
complex array signal source — from signal_generator_v2, sequence_engine, tf_ray_engine,
or a plain oscillator.

Quickstart
----------
    from routing_engine import RoutingGraph, RoutingMixer
    import numpy as np

    g = RoutingGraph()
    g.add_node("osc1")
    g.add_node("osc2")
    g.set_weight("osc1", "osc2", 0.5)    # osc1 feeds 50% into osc2's input
    g.set_weight("osc2", "osc1", -0.3)   # osc2 feeds -30% back into osc1
    g.set_delay_s("osc2", "osc1", 0.002) # 2 ms feedback delay
    g.feedback.enabled = True
    g.feedback.decay = 0.05              # 5% amplitude decay per feedback cycle

    # source signals: complex128 (analytic) or real — any generator output works
    sources = {"osc1": osc1_array, "osc2": osc2_array}
    mixer = RoutingMixer(g, sample_rate=48_000)
    out = mixer.mix(sources)             # {"osc1": ..., "osc2": ...} routed outputs

    # Convenience: sum + real-project to float32 for audio output
    audio = mixer.mix_and_project(sources)

Signal model
------------
Each edge (src → dst) contributes:

    x_dst[t] += weight * exp(i * angle_rad) * x_src[t − delay_samples]

Instantaneous edges (delay_s == 0) are gathered into a complex weight matrix W
and solved algebraically once:

    X = (I − W)⁻¹ @ Src

Delayed edges use a causal chunk-based solver with chunk size equal to the
minimum nonzero delay across all edges.  The instantaneous matrix is still
pre-inverted once and reused per chunk.

The feedback.decay knob multiplies all edge weights by (1 − decay), providing
a global energy-loss control.  When feedback.enabled is False, decay is not applied.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Signal layer hierarchy — ordered from source to final output.
#
# Exact causal flow:
#
#   voice        (ROUTER → driver)
#     Voice oscillators.  Voice routers converge their outputs onto drivers.
#
#   driver       (ROUTER → performer SM)
#     Driver outputs carry per-note identity (item_slot on the routing edge)
#     so the performer SM knows which driver belongs to which instrument.
#
#   performer    (STATE MACHINE)
#     Receives driver signals keyed by item_slot.  Internally owns and
#     maintains all instrument states.  Emits per-instrument outputs
#     (aperture pressure, body resonance, etc.).
#
#   instrument   (SHARED ROUTING — feeds instrument mixer AND room SM)
#     Instrument outputs participate in two paths simultaneously:
#       1. instrument mixer — aperture/pickup outputs for DI, electronic
#          routing, effects chains.
#       2. room SM — acoustic propagation → mic stream(s).
#     The routing edges from instrument nodes decide which path(s) are used.
#
#   instrument mixer   (MIXER)
#     Collects instrument aperture/pickup signals for electronic routing
#     and effects.  Output goes to the master mixer.
#
#   room         (STATE MACHINE — not a mixer)
#     Receives instrument outputs, simulates acoustic propagation, emits
#     1 or N microphone stream(s).  Mic streams go directly to the master
#     mixer.  There is no room mixer.
#
#   master mixer / mastering level mixer   (MIXER — top level)
#     Receives room SM mic stream(s) and instrument mixer output.
#     Final stereo projection is applied here.
#
# SM_LAYERS: layers whose nodes are AnalyticModule state machines.
# MIXER_LAYERS: layers whose nodes are AnalyticMixer instances.
# An empty string means "unspecified" (valid for backward-compat nodes).
SIGNAL_LAYERS: tuple = ("voice", "driver", "performer", "instrument", "room", "master")
SM_LAYERS:     tuple = ("performer", "room")
MIXER_LAYERS:  tuple = ("voice_router", "instrument", "master")


@dataclass
class RoutingEdge:
    """Directed edge: applies weight * exp(i*angle_rad) to src, optionally delayed.

    router_key
    ----------
    Every signal edge is owned by exactly one router instance.  The router
    that owns the edge is the only one that includes it in its solve.  Empty
    string means unowned / legacy (included by any solver that encounters it).

    item_slot
    ---------
    When an edge feeds a state-machine (SM) module, setting *item_slot* to a
    non-empty string causes the SM to receive the signal under that name rather
    than under the opaque src_key.  This preserves performer/instrument identity
    all the way to the SM boundary without collapsing the signal to an anonymous
    mix:

        edge.item_slot = "violin_1"   # SM receives inputs["violin_1"]
        edge.item_slot = ""           # SM receives inputs[src_key]  (legacy)
    """
    src_key:          str   = ""
    dst_key:          str   = ""
    weight:           float = 0.0   # amplitude coefficient, typically in [-2, 2]
    angle_rad:        float = 0.0   # phase rotation applied to src before mixing
    delay_s:          float = 0.0   # per-edge propagation delay; 0 = synchronous
    item_slot:        str   = ""    # named SM input slot; empty = use src_key
    router_key:       str   = ""    # owning router instance key; empty = legacy
    saturation:       str   = ""    # "tanh" | "hardclip" | "softclip" | "" = none
    saturation_knee:  float = 1.0   # magnitude at which saturation engages


@dataclass
class ParamEdge:
    """Control edge: routes a scalar parameter derived from any signal node to
    any node's sub-parameter (knob).

    Causality is unconditionally enforced: the value sampled at time t is
    delivered to the destination parameter at t + delay_samples (minimum 1).
    This prevents circular instantaneous control dependencies and matches
    physical reality — a control signal cannot arrive before it is emitted.

    Extractors
    ----------
    magnitude  |z|          — always positive; useful for energy/density control
    real       Re(z)        — signed; follows the analytic real part
    imag       Im(z)        — signed quadrature
    phase      arg(z)       — [-π, π]; useful for pitch-tracking
    energy     |z|²         — squared magnitude; heavier weighting of loud parts
    rms        running RMS  — smoothed magnitude (128-sample window)

    param_path
    ----------
    Dot-separated path from the destination node key to the target sub-parameter,
    e.g. "chirp.f_delta_start" or "envelope.attack_s".  Empty = node-level scalar.
    """
    src_key:       str   = ""
    dst_key:       str   = ""
    weight:        float = 1.0
    extractor:     str   = "magnitude"
    delay_samples: int   = 0          # 0 = instantaneous (acyclic paths only)
    param_path:    str   = ""         # sub-parameter target within dst node

    def __post_init__(self) -> None:
        if self.delay_samples < 0:
            self.delay_samples = 0

    def to_dict(self) -> dict:
        d: dict = {"src": self.src_key, "dst": self.dst_key,
                   "w": self.weight, "extractor": self.extractor,
                   "delay_samples": self.delay_samples}
        if self.param_path:
            d["param_path"] = self.param_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ParamEdge":
        return cls(
            src_key=str(d.get("src", "")),
            dst_key=str(d.get("dst", "")),
            weight=float(d.get("w", 1.0)),
            extractor=str(d.get("extractor", "magnitude")),
            delay_samples=max(1, int(d.get("delay_samples", 1))),
            param_path=str(d.get("param_path", "")),
        )


@dataclass
class FeedbackConfig:
    """Global feedback solve parameters."""
    enabled:        bool  = False
    delay_s:        float = 0.0   # legacy global-delay field (per-edge delay preferred)
    decay:          float = 0.0   # 0–1: global amplitude loss per feedback cycle
    max_iterations: int   = 1000  # hard cap for nonlinear fixed-point iteration
    # Ringdown policy — controls how long the routing graph is allowed to run
    # after all source signals have gone silent.
    #   "none"             – no ringdown; buffer ends at note duration
    #   "fixed"            – extend by ringdown_max_s seconds
    #   "decay_to_silence" – extend until energy falls below ringdown_threshold
    #                        (capped at ringdown_max_s)
    ringdown_mode:      str   = "decay_to_silence"
    ringdown_max_s:     float = 4.0    # hard cap on tail extension (seconds)
    ringdown_threshold: float = 1e-4   # amplitude threshold for decay_to_silence


@dataclass
class RoutingGraph:
    """Directed weighted graph of signal nodes with per-edge delay and phase rotation.

    Nodes
    -----
    Register signal sources with add_node().  The order of registration sets
    the canonical index used internally by the solver.

    Edges
    -----
    set_weight / set_angle_rad / set_delay_s create or update edges.
    A zero-weight edge with non-zero angle or delay is permitted.

    Serialisation
    -------------
    to_dict() / from_dict() give a JSON-compatible round-trip.
    """
    nodes:       list = field(default_factory=list)   # list[str] — ordered node keys
    edges:       list = field(default_factory=list)   # list[RoutingEdge]
    param_edges: list = field(default_factory=list)   # list[ParamEdge] — scalar extraction rules
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    # When True, source nodes are pre-synthesised starting before t=0 so that
    # after their routing delays their signal arrives at every destination on
    # time (latency / PDC compensation).  Feedback cycles remain causal.
    latency_compensation: bool = False

    # Per-node type annotation — {node_key: type_str}.
    # Type values from SIGNAL_LAYERS or custom strings.
    node_types: dict = field(default_factory=dict)

    # Per-node router-type participation — {node_key: [router_type, ...]}.
    # source_router_types: router types in which this node provides signal (source).
    # sink_router_types:   router types in which this node receives signal (sink).
    # A node absent from these dicts participates in all routers it appears in
    # (legacy / untyped behaviour).
    node_source_router_types: dict = field(default_factory=dict)
    node_sink_router_types:   dict = field(default_factory=dict)

    # Backward-compat alias — populated by from_dict for old patches.
    node_layers: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    @classmethod
    def knobs(cls) -> list:
        """KnobSpec descriptors for the routing graph's global controls.

        These are surfaced in the PartialPanel when a mixer node is active so
        the user can adjust feedback/delay settings and toggle latency
        compensation without leaving the editor.
        """
        try:
            from signal_generator_v2 import KnobSpec  # type: ignore
        except Exception:
            from dataclasses import dataclass as _kdc, field as _kfield
            @_kdc
            class KnobSpec:  # type: ignore
                name: str = ""; label: str = ""; dtype: str = "float"
                default: object = None; low: float = 0.0; high: float = 1.0
                step: float = 0.0; unit: str = ""; choices: list = _kfield(default_factory=list)
                is_log: bool = False; group: str = ""; fmt: str = ".3g"
                rebuild_layout: bool = False; visible_when: object = None
        _RINGDOWN_MODES = ["none", "fixed", "decay_to_silence"]
        return [
            KnobSpec("latency_compensation", "Latency comp.",    "bool",  False,  0, 1, 0, "", [], False, "Routing"),
            KnobSpec("feedback.enabled",     "Feedback",         "bool",  False,  0, 1, 0, "", [], False, "Routing"),
            KnobSpec("feedback.delay_s",     "FB delay",         "float", 0.0, 0.0, 2.0, 0, "s",  [], False, "Routing", ".4f"),
            KnobSpec("feedback.decay",       "FB decay",         "float", 0.0, 0.0, 1.0, 0, "",   [], False, "Routing", ".3f"),
            KnobSpec("feedback.ringdown_mode",      "Ringdown mode",   "choice", "decay_to_silence", 0, 2, 1, "", _RINGDOWN_MODES, False, "Routing"),
            KnobSpec("feedback.ringdown_max_s",     "Ringdown max",    "float",  4.0, 0.1, 20.0, 0, "s", [], False, "Routing", ".2f"),
            KnobSpec("feedback.ringdown_threshold", "Ringdown thresh", "float",  1e-4, 1e-6, 0.1, 0, "", [], False, "Routing", ".2e"),
        ]

    def add_node(
        self,
        key: str,
        layer: str = "",
        node_type: str = "",
        source_router_types: "list[str] | tuple[str, ...] | None" = None,
        sink_router_types:   "list[str] | tuple[str, ...] | None" = None,
    ) -> None:
        """Register a node.

        node_type            — signal role (from SIGNAL_LAYERS or custom).
        source_router_types  — router types that may use this node as a source.
                               None = unrestricted (legacy behaviour).
        sink_router_types    — router types that may use this node as a sink.
                               None = unrestricted.
        layer                — backward-compat alias for node_type; ignored when
                               node_type is also provided.
        """
        if key not in self.nodes:
            self.nodes.append(key)
        nt = node_type or layer
        if nt:
            self.node_types[key] = nt
            self.node_layers[key] = nt          # keep compat alias in sync
        if source_router_types is not None:
            self.node_source_router_types[key] = list(source_router_types)
        if sink_router_types is not None:
            self.node_sink_router_types[key] = list(sink_router_types)

    def node_keys(self) -> list:
        """Return the ordered list of registered node keys."""
        return list(self.nodes)

    def set_node_type(self, key: str, node_type: str) -> None:
        """Set the type for *key*, registering the node if absent."""
        self.add_node(key)
        self.node_types[key] = str(node_type)
        self.node_layers[key] = str(node_type)

    def get_node_type(self, key: str) -> str:
        """Return the type for *key*, or empty string."""
        return self.node_types.get(key, self.node_layers.get(key, ""))

    def set_node_participation(
        self,
        key: str,
        source_router_types: "list[str] | None" = None,
        sink_router_types:   "list[str] | None" = None,
    ) -> None:
        """Explicitly declare which router types this node participates in."""
        self.add_node(key)
        if source_router_types is not None:
            self.node_source_router_types[key] = list(source_router_types)
        if sink_router_types is not None:
            self.node_sink_router_types[key] = list(sink_router_types)

    def get_node_source_router_types(self, key: str) -> "list[str]":
        return list(self.node_source_router_types.get(key, []))

    def get_node_sink_router_types(self, key: str) -> "list[str]":
        return list(self.node_sink_router_types.get(key, []))

    def is_source_in_router(self, key: str, router_type: str) -> bool:
        """True if *key* is allowed as a source in *router_type* routers."""
        allowed = self.node_source_router_types.get(key)
        return allowed is None or router_type in allowed

    def is_sink_in_router(self, key: str, router_type: str) -> bool:
        """True if *key* is allowed as a sink in *router_type* routers."""
        allowed = self.node_sink_router_types.get(key)
        return allowed is None or router_type in allowed

    # Backward-compat shims for code that used node_layers directly
    def set_node_layer(self, key: str, layer: str) -> None:
        self.add_node(key, node_type=layer)

    def get_node_layer(self, key: str) -> str:
        return self.get_node_type(key)

    def nodes_in_layer(self, layer: str) -> list:
        return [k for k in self.nodes if self.get_node_type(k) == layer]

    def nodes_for_router_source(self, router_type: str) -> list:
        """Return all node keys that can act as sources in *router_type* routers."""
        return [k for k in self.nodes if self.is_source_in_router(k, router_type)]

    def nodes_for_router_sink(self, router_type: str) -> list:
        """Return all node keys that can act as sinks in *router_type* routers."""
        return [k for k in self.nodes if self.is_sink_in_router(k, router_type)]

    def ensure_defaults(self, keys: list, mix_key: str = "__mix__") -> None:
        """For every key in *keys* that has no edge going to *mix_key*, add one
        with weight 1.0.  Idempotent — existing edges are not touched."""
        existing_to_mix = {e.src_key for e in self.edges if e.dst_key == mix_key}
        for k in keys:
            if k != mix_key and k not in existing_to_mix:
                self.edges.append(RoutingEdge(k, mix_key, 1.0))
        for k in keys:
            self.add_node(k)
        self.add_node(mix_key)

    # ------------------------------------------------------------------
    # Edge accessors — get/set weight, angle, delay
    # ------------------------------------------------------------------

    def _find_edge(self, src_key: str, dst_key: str) -> Optional[RoutingEdge]:
        for e in self.edges:
            if e.src_key == src_key and e.dst_key == dst_key:
                return e
        return None

    def get_weight(self, src_key: str, dst_key: str) -> float:
        e = self._find_edge(src_key, dst_key)
        return e.weight if e is not None else 0.0

    def set_weight(self, src_key: str, dst_key: str, weight: float) -> None:
        e = self._find_edge(src_key, dst_key)
        if e is not None:
            e.weight = weight
        elif abs(weight) > 1e-12:
            self.edges.append(RoutingEdge(src_key, dst_key, weight, 0.0, 0.0))

    def get_angle_rad(self, src_key: str, dst_key: str) -> float:
        e = self._find_edge(src_key, dst_key)
        return e.angle_rad if e is not None else 0.0

    def set_angle_rad(self, src_key: str, dst_key: str, angle: float) -> None:
        e = self._find_edge(src_key, dst_key)
        if e is not None:
            e.angle_rad = angle
        elif abs(angle) > 1e-9:
            self.edges.append(RoutingEdge(src_key, dst_key, 0.0, angle, 0.0))

    def get_delay_s(self, src_key: str, dst_key: str) -> float:
        e = self._find_edge(src_key, dst_key)
        return e.delay_s if e is not None else 0.0

    def set_delay_s(self, src_key: str, dst_key: str, delay: float) -> None:
        e = self._find_edge(src_key, dst_key)
        if e is not None:
            e.delay_s = delay
        elif delay > 1e-9:
            self.edges.append(RoutingEdge(src_key, dst_key, 0.0, 0.0, delay))

    def prune(self) -> None:
        """Remove edges whose weight, angle, and delay are all effectively zero."""
        self.edges = [
            e for e in self.edges
            if abs(e.weight) > 1e-12 or abs(e.angle_rad) > 1e-9 or e.delay_s > 1e-9
        ]

    def prune_keys(self, valid_keys: "list | set") -> None:
        """Remove edges whose src_key or dst_key is not in *valid_keys*.

        Call this after a voice/LFO/param-node is deleted, or after loading a
        patch that may reference nodes that no longer exist, to avoid ghost
        edges in the routing grid.
        """
        vk = set(valid_keys)
        self.edges = [
            e for e in self.edges
            if e.src_key in vk and e.dst_key in vk
        ]
        self.param_edges = [
            pe for pe in self.param_edges
            if pe.src_key in vk and pe.dst_key in vk
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def _edge_dict(e: RoutingEdge) -> dict:
            d: dict = {"src": e.src_key, "dst": e.dst_key, "w": e.weight,
                       "angle": e.angle_rad, "delay": e.delay_s}
            if e.item_slot:
                d["item_slot"] = e.item_slot
            if e.router_key:
                d["router_key"] = e.router_key
            if e.saturation:
                d["saturation"] = e.saturation
                d["saturation_knee"] = e.saturation_knee
            return d
        result: dict = {
            "nodes": list(self.nodes),
            "edges": [_edge_dict(e) for e in self.edges],
            "param_edges": [e.to_dict() for e in self.param_edges],
            "feedback": {
                "enabled":            self.feedback.enabled,
                "delay_s":            self.feedback.delay_s,
                "decay":              self.feedback.decay,
                "max_iterations":     self.feedback.max_iterations,
                "ringdown_mode":      self.feedback.ringdown_mode,
                "ringdown_max_s":     self.feedback.ringdown_max_s,
                "ringdown_threshold": self.feedback.ringdown_threshold,
            },
            "latency_compensation": self.latency_compensation,
        }
        if self.node_types:
            result["node_types"] = dict(self.node_types)
        if self.node_source_router_types:
            result["node_source_router_types"] = {
                k: list(v) for k, v in self.node_source_router_types.items()
            }
        if self.node_sink_router_types:
            result["node_sink_router_types"] = {
                k: list(v) for k, v in self.node_sink_router_types.items()
            }
        # Emit node_layers for backward compat with older readers
        if self.node_layers:
            result["node_layers"] = dict(self.node_layers)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "RoutingGraph":
        g = cls()
        g.nodes = list(d.get("nodes", []))
        g.param_edges = [ParamEdge.from_dict(pe) for pe in d.get("param_edges", [])]
        for ed in d.get("edges", []):
            g.edges.append(RoutingEdge(
                src_key=str(ed.get("src", "")),
                dst_key=str(ed.get("dst", "")),
                weight=float(ed.get("w", 0.0)),
                angle_rad=float(ed.get("angle", 0.0)),
                delay_s=float(ed.get("delay", 0.0)),
                item_slot=str(ed.get("item_slot", "")),
                router_key=str(ed.get("router_key", "")),
                saturation=str(ed.get("saturation", "")),
                saturation_knee=float(ed.get("saturation_knee", 1.0)),
            ))
        # node_types (new) — fall back to legacy node_layers
        raw_types = d.get("node_types") or d.get("node_layers", {})
        g.node_types  = {str(k): str(v) for k, v in raw_types.items()}
        g.node_layers = dict(g.node_types)   # keep compat alias in sync
        g.node_source_router_types = {
            str(k): [str(x) for x in v]
            for k, v in d.get("node_source_router_types", {}).items()
        }
        g.node_sink_router_types = {
            str(k): [str(x) for x in v]
            for k, v in d.get("node_sink_router_types", {}).items()
        }
        fb = d.get("feedback", {})
        g.feedback = FeedbackConfig(
            enabled=bool(fb.get("enabled", False)),
            delay_s=float(fb.get("delay_s", 0.0)),
            decay=float(fb.get("decay", 0.0)),
            max_iterations=int(fb.get("max_iterations", 1000)),
            ringdown_mode=str(fb.get("ringdown_mode", "decay_to_silence")),
            ringdown_max_s=float(fb.get("ringdown_max_s", 4.0)),
            ringdown_threshold=float(fb.get("ringdown_threshold", 1e-4)),
        )
        g.latency_compensation = bool(d.get("latency_compensation", False))
        return g


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

# Soft-clip: linear through any physically meaningful range, only prevents
# float overflow.  Half-width at ~1e200 — the sigmoid curve only begins to
# deviate from identity at magnitudes that are astronomically beyond any
# sane signal level (~10^245 dB).  Complex variant applies independently to
# magnitude while preserving phase exactly.
_SOFTCLIP_KNEE: float = 1e200

def _softclip_complex(X: np.ndarray) -> np.ndarray:
    """In-place magnitude-preserving soft-clip for complex128 arrays.

    |z| < _SOFTCLIP_KNEE  →  z is returned unchanged (bit-identical).
    |z| >= _SOFTCLIP_KNEE →  magnitude is tanh-compressed while phase is preserved.
    """
    mag = np.abs(X)
    mask = mag >= _SOFTCLIP_KNEE
    if not np.any(mask):
        return X
    # For the (extremely rare) overflow-range samples: compress magnitude
    # via  m_out = knee * tanh(m_in / knee).  This is identity for m << knee.
    m_in = mag[mask]
    m_out = _SOFTCLIP_KNEE * np.tanh(m_in / _SOFTCLIP_KNEE)
    # Preserve phase: scale by m_out/m_in
    scale = np.ones_like(m_in)
    nz = m_in > 0
    scale[nz] = m_out[nz] / m_in[nz]
    X[mask] *= scale
    return X


def _safe_inverse(mat: np.ndarray) -> np.ndarray:
    """Return a stable inverse, adding a tiny diagonal regularizer on failure."""
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        jitter = np.eye(mat.shape[0], dtype=mat.dtype) * 1e-6
        return np.linalg.inv(mat + jitter)


def _find_wccs(node_keys: list, edges: list, coupled_pairs: list) -> list:
    """Union-find weakly connected components over the routing edge graph.

    coupled_pairs: list of lists of node keys that must land in the same
    component regardless of whether a routing edge connects them (e.g.
    interaural ch1/ch2 coupled-transform pairs).

    Returns a list of components; each component is a sorted list of
    node-key indices into node_keys.
    """
    n = len(node_keys)
    if n == 0:
        return []
    ki = {k: i for i, k in enumerate(node_keys)}
    parent = list(range(n))

    def _find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a: int, b: int) -> None:
        a, b = _find(a), _find(b)
        if a != b:
            parent[b] = a

    for e in edges:
        si = ki.get(e.src_key)
        di = ki.get(e.dst_key)
        if si is not None and di is not None:
            _union(si, di)

    for pair in coupled_pairs:
        idxs = [ki[k] for k in pair if k in ki]
        for j in range(1, len(idxs)):
            _union(idxs[0], idxs[j])

    groups: dict = {}
    for i in range(n):
        r = _find(i)
        if r not in groups:
            groups[r] = []
        groups[r].append(i)
    return list(groups.values())


def solve_routing_complex(
    Src: np.ndarray,        # (N, T) complex128 — independent source signals
    edges: list,            # list[RoutingEdge]
    node_keys: list,        # canonical key list, len == N
    sr: float,
    global_decay: float = 1.0,
    node_transforms: "dict | None" = None,
    coupled_transforms: "dict | None" = None,
    max_iterations: int = 1000,
    convergence_eps: float = 1e-12,
) -> np.ndarray:            # (N, T) complex128 — solved node outputs
    """Solve the analytic routing system — unified iterative graph solver.

    Every node in the graph is solved together.  Linear nodes (no transform)
    and nonlinear nodes (with a transform callable) participate in the same
    iteration.  The solver iterates the full graph until convergence or
    *max_iterations*, whichever comes first.

    Algorithm
    ---------
    1.  Build the instantaneous weight matrix W and per-delay weight matrices.
    2.  Pre-advance (negative-delay) edges are folded into Src.
    3.  For graphs with NO nonlinear transforms and NO delayed edges:
        exact solve via X = (I − W)⁻¹ @ Src — converged in one step.
    4.  For graphs WITH nonlinear transforms:
        Iterative fixed-point.  Each iteration:
          a.  Evaluate the linear contribution:  X_lin = M @ (Src_eff + delayed_contrib)
              where M = (I − W)⁻¹ and Src_eff includes transform-node outputs
              from the previous iteration.
          b.  Apply each node transform to its row:  X[i] = f_i(X_lin[i])
              (identity for nodes without a transform).
          c.  Soft-clip the entire X to prevent float overflow (linear through
              any sane range; see ``_softclip_complex``).
          d.  Check convergence: max|X_new − X_old| < eps.  If converged, stop.
        The 1000-iteration guard exists only for genuinely undamped nonlinear
        feedback cycles that cannot converge.
    5.  For delayed edges: causal chunk-by-chunk processing.  Within each chunk,
        the same iterative convergence loop runs.

    Parameters
    ----------
    global_decay
        All edge weights are multiplied by this scalar before the solve.
    node_transforms
        dict mapping node_key → callable(complex128[T]) → complex128[T].
        The callable receives the node's accumulated input and returns its
        output. Must return complex128 and preserve the analytic signal.
    coupled_transforms
        dict mapping tuple[str, ...] → callable(dict[str, complex128[T]])
                                        → dict[str, complex128[T]].
        For transforms that couple multiple nodes (e.g. interaural spatial
        that reads ch1+ch2 inputs and writes back two new complex signals).
        The callable receives {key: row} for every key in the tuple and must
        return a dict with the same keys holding the transformed rows.
        Applied after node_transforms in every convergence iteration.
    max_iterations
        Hard cap on convergence iterations.  Only reached for undamped
        nonlinear feedback.
    convergence_eps
        Fixed-point convergence threshold on max|X_new − X_old|.

    Signal model
    -------------
    Each edge contributes:
        x_dst[t] += weight * exp(i * angle_rad) * x_src[t − delay_samples]

    The solver never truncates, approximates, or projects the complex signal.
    The only non-identity operation on the signal outside of edge weights and
    node transforms is the soft-clip guard at ~1e200 magnitude.
    """
    ki = {k: i for i, k in enumerate(node_keys)}
    N, T = Src.shape

    # ── WCC partitioning: solve independent subgraphs concurrently ────
    # coupled_transforms node-pairs must land in the same component even when
    # no explicit routing edge connects them (e.g. interaural ch1/ch2).
    _coupled_pairs = [list(kt) for kt in (coupled_transforms or {})]
    _components = _find_wccs(node_keys, edges, _coupled_pairs)
    if len(_components) > 1:
        _transform_keys = set(node_transforms or {})
        _coupled_keys: set = set()
        for _kt in (coupled_transforms or {}):
            _coupled_keys.update(_kt)

        # Nodes that are isolated AND have no transform collapse to X[i] = Src[i].
        # All others go into the work list for a recursive per-component solve.
        X = Src.copy()
        _work: list = []
        for _comp in _components:
            if len(_comp) == 1:
                _key = node_keys[_comp[0]]
                if _key not in _transform_keys and _key not in _coupled_keys:
                    continue  # trivially isolated — X[i] = Src[i] already
            _work.append(_comp)

        def _solve_component(_comp_indices: list) -> tuple:
            _comp_set = {node_keys[_i] for _i in _comp_indices}
            _sub_keys  = [node_keys[_i] for _i in _comp_indices]
            _sub_Src   = Src[_comp_indices]
            _sub_edges = [_e for _e in edges
                          if _e.src_key in _comp_set and _e.dst_key in _comp_set]
            _sub_nt = ({_k: _v for _k, _v in (node_transforms or {}).items()
                        if _k in _comp_set} or None)
            _sub_ct = ({_kt: _fn for _kt, _fn in (coupled_transforms or {}).items()
                        if all(_k in _comp_set for _k in _kt)} or None)
            return _comp_indices, solve_routing_complex(
                _sub_Src, _sub_edges, _sub_keys, sr, global_decay,
                _sub_nt, _sub_ct, max_iterations, convergence_eps,
            )

        if len(_work) > 1:
            _n_workers = min(len(_work), 8)
            with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
                _futs = [_pool.submit(_solve_component, _comp) for _comp in _work]
                for _fut in _futs:
                    _idxs, _result = _fut.result()
                    X[_idxs] = _result
        elif _work:
            _idxs, _result = _solve_component(_work[0])
            X[_idxs] = _result
        return X
    # ── Single component (or empty): fall through to the unified solver ─

    transforms = node_transforms or {}

    # Index set of nodes that have a nonlinear single-node transform
    transform_idx: set = set()
    for key in transforms:
        ti = ki.get(key)
        if ti is not None:
            transform_idx.add(ti)

    # Coupled transform groups: list of (keys_tuple, index_list, callable)
    coupled_groups: list = []
    for keys_tuple, fn in (coupled_transforms or {}).items():
        idxs = [ki[k] for k in keys_tuple if k in ki]
        if len(idxs) == len(keys_tuple):
            coupled_groups.append((keys_tuple, idxs, fn))

    # ── Build weight matrices ──────────────────────────────────────────
    W_inst: np.ndarray = np.zeros((N, N), dtype=np.complex128)
    delay_groups: Dict[int, np.ndarray] = {}
    adv_groups: Dict[int, np.ndarray] = {}

    for e in edges:
        si = ki.get(e.src_key)
        di = ki.get(e.dst_key)
        if si is None or di is None:
            continue
        cw = complex(
            e.weight * global_decay * math.cos(e.angle_rad),
            e.weight * global_decay * math.sin(e.angle_rad),
        )
        d_samp = int(round(e.delay_s * sr))
        if d_samp == 0:
            W_inst[di, si] += cw
        elif d_samp > 0:
            if d_samp not in delay_groups:
                delay_groups[d_samp] = np.zeros((N, N), dtype=np.complex128)
            delay_groups[d_samp][di, si] += cw
        else:
            adv = -d_samp
            if adv not in adv_groups:
                adv_groups[adv] = np.zeros((N, N), dtype=np.complex128)
            adv_groups[adv][di, si] += cw

    # Fold pre-advance (negative-delay) contributions into Src
    if adv_groups:
        Src = Src.copy()
        for adv, Wadv in adv_groups.items():
            if adv < T:
                Src[:, :T - adv] += Wadv @ Src[:, adv:]

    # Pre-invert the instantaneous linear part
    IW = np.eye(N, dtype=np.complex128) - W_inst
    M = _safe_inverse(IW)

    has_transforms = len(transform_idx) > 0 or len(coupled_groups) > 0

    # ── Helper: apply all transforms (single-node and coupled) ─────────
    def _apply_transforms(X: np.ndarray) -> np.ndarray:
        for key, fn in transforms.items():
            ti = ki.get(key)
            if ti is not None:
                X[ti] = fn(X[ti])
        for keys_tuple, idxs, fn in coupled_groups:
            rows_in = {k: X[i] for k, i in zip(keys_tuple, idxs)}
            rows_out = fn(rows_in)
            for k, i in zip(keys_tuple, idxs):
                if k in rows_out:
                    X[i] = rows_out[k]
        return X

    # ── No delays, no transforms: exact linear solve ──────────────────
    if not delay_groups and not has_transforms:
        return M @ Src

    nl_idx = sorted(transform_idx | {i for _, idxs, _ in coupled_groups for i in idxs})
    li_idx = [i for i in range(N) if i not in nl_idx]

    if has_transforms and nl_idx and li_idx:
        W_ll = W_inst[np.ix_(li_idx, li_idx)]
        W_ln = W_inst[np.ix_(li_idx, nl_idx)]
        W_nl = W_inst[np.ix_(nl_idx, li_idx)]
        W_nn = W_inst[np.ix_(nl_idx, nl_idx)]
        Ainv = _safe_inverse(np.eye(len(li_idx), dtype=np.complex128) - W_ll)
        rhs_n_const = W_nl @ Ainv
        W_eff = W_nn + W_nl @ Ainv @ W_ln
        local_of_global = {gi: i for i, gi in enumerate(nl_idx)}
        reduced_transform_idx = [local_of_global[i] for i in sorted(transform_idx)]
        reduced_single_transforms = {
            local_of_global[gi]: transforms[node_keys[gi]]
            for gi in sorted(transform_idx)
        }
        reduced_coupled_groups: list = []
        for keys_tuple, idxs, fn in coupled_groups:
            reduced_coupled_groups.append(
                (keys_tuple, [local_of_global[i] for i in idxs], fn)
            )

        def _apply_reduced_transforms(X_n: np.ndarray) -> np.ndarray:
            for ti in reduced_transform_idx:
                X_n[ti] = reduced_single_transforms[ti](X_n[ti])
            for keys_tuple, idxs, fn in reduced_coupled_groups:
                rows_in = {k: X_n[i] for k, i in zip(keys_tuple, idxs)}
                rows_out = fn(rows_in)
                for k, i in zip(keys_tuple, idxs):
                    if k in rows_out:
                        X_n[i] = rows_out[k]
            return X_n

        def _solve_reduced(rhs: np.ndarray) -> np.ndarray:
            rhs_l = rhs[li_idx]
            rhs_n = rhs[nl_idx]
            rhs_eff = rhs_n + rhs_n_const @ rhs_l
            M_eff = _safe_inverse(np.eye(len(nl_idx), dtype=np.complex128) - W_eff)
            X_n = M_eff @ rhs_eff
            _apply_reduced_transforms(X_n)
            _softclip_complex(X_n)
            for _iter in range(max_iterations):
                X_prev = X_n.copy()
                X_input_n = rhs_eff + W_eff @ X_n
                X_n = X_input_n.copy()
                _apply_reduced_transforms(X_n)
                _softclip_complex(X_n)
                if np.max(np.abs(X_n - X_prev)) < convergence_eps:
                    break
            X = np.zeros((N, rhs.shape[1]), dtype=np.complex128)
            X[nl_idx] = X_n
            X[li_idx] = Ainv @ (rhs_l + W_ln @ X_n)
            return X
    else:
        def _solve_reduced(rhs: np.ndarray) -> np.ndarray:
            X = M @ rhs
            _apply_transforms(X)
            _softclip_complex(X)
            for _iter in range(max_iterations):
                X_prev = X.copy()
                X_input = rhs + W_inst @ X
                X = X_input.copy()
                for ti in transform_idx:
                    X[ti] = transforms[node_keys[ti]](X_input[ti])
                for keys_tuple, idxs, fn in coupled_groups:
                    rows_in = {k: X_input[i] for k, i in zip(keys_tuple, idxs)}
                    rows_out = fn(rows_in)
                    for k, i in zip(keys_tuple, idxs):
                        if k in rows_out:
                            X[i] = rows_out[k]
                _softclip_complex(X)
                if np.max(np.abs(X - X_prev)) < convergence_eps:
                    break
            return X

    # ── No delays, with transforms: iterative fixed-point ─────────────
    if not delay_groups:
        return _solve_reduced(Src)

    # ── Delayed edges: causal chunk solver ─────────────────────────────
    d_min = min(delay_groups.keys())
    X = np.zeros((N, T), dtype=np.complex128)

    for t0 in range(0, T, d_min):
        t1 = min(t0 + d_min, T)
        chunk_len = t1 - t0

        rhs = Src[:, t0:t1].copy()
        for d, Wd in delay_groups.items():
            t_past = t0 - d
            if t_past >= 0:
                rhs += Wd @ X[:, t_past: t_past + chunk_len]

        if not has_transforms:
            X[:, t0:t1] = M @ rhs
        else:
            X[:, t0:t1] = _solve_reduced(rhs)

    return X


# ---------------------------------------------------------------------------
# Ringdown utilities
# ---------------------------------------------------------------------------

def estimate_ringdown_samples(
    edges: list,
    sr: float,
    global_decay: float,
    fb: "FeedbackConfig",
) -> int:
    """Return the number of extra zero-source samples needed for the routing
    graph to drain to silence after all inputs have stopped.

    The estimate is based on the worst-case delayed feedback path:
    a signal bouncing through an edge with effective weight ``w`` and
    delay ``d`` samples decays as ``w^k`` after ``k`` bounces.  We solve
    for the smallest ``k`` such that ``w^k <= threshold`` and return ``k*d``.

    Policy (``fb.ringdown_mode``):
    --------------------------------
    ``"none"``             — always returns 0; use when you deliberately want
                             hard truncation (useful for rhythmically tight
                             dry sounds with no feedback).
    ``"fixed"``            — always extends by exactly ``fb.ringdown_max_s``
                             seconds.  Good when you want a predictable tail
                             budget regardless of settings.
    ``"decay_to_silence"`` — analytically derives the minimum tail needed for
                             the loudest feedback path to drop below
                             ``fb.ringdown_threshold``.  Capped at
                             ``fb.ringdown_max_s`` seconds.
    """
    if fb.ringdown_mode == "none":
        return 0

    max_n = int(fb.ringdown_max_s * sr)
    if max_n <= 0:
        return 0

    if fb.ringdown_mode == "fixed":
        return max_n

    # "decay_to_silence": find the dominant delayed feedback path
    delayed = [(e.delay_s, abs(e.weight) * global_decay)
               for e in edges if e.delay_s > 1e-9 and abs(e.weight) > 1e-12]
    if not delayed:
        return 0

    max_w = max(w for _, w in delayed)
    if max_w <= 0.0:
        return 0
    if max_w >= 1.0:
        # Sustained or divergent — cap at the max budget
        return max_n

    max_d_samp = max(int(round(d * sr)) for d, _ in delayed)
    if max_d_samp <= 0:
        return 0

    threshold = max(fb.ringdown_threshold, 1e-12)
    bounces = math.ceil(math.log(threshold) / math.log(max_w))
    return min(max_n, bounces * max_d_samp)


# ---------------------------------------------------------------------------
# Latency compensation (pre-delay / look-ahead)
# ---------------------------------------------------------------------------

def compute_latency_compensation(
    edges: list,
    node_keys: list,
    sr: float,
) -> dict:
    """Compute the pre-roll needed for each source node so that, after
    propagating through routing delays, its signal arrives at every
    destination on time (at t=0 from the destination's perspective).

    Algorithm
    ---------
    Only *feedforward* paths (acyclic portions of the graph) contribute to
    the lead-time budget.  Feedback cycles remain causal — a feedback edge
    (a→b) is detected if b can reach a; those edges are excluded from the
    longest-path computation.

    We compute the *longest-delay path* from each node to every other node
    using a topological-sort longest-path on the feedforward DAG.  For each
    source node the required pre-roll is:

        lead[src] = max over all downstream nodes d of: delay(src → d via longest path)

    Returns
    -------
    dict[str, int]
        {node_key: lead_samples} for every node that needs pre-roll.
        Nodes with no outgoing delays get lead=0 (not included in the dict).
    """
    ki = {k: i for i, k in enumerate(node_keys)}
    N = len(node_keys)

    # Build adjacency as sample-delay values (direct edges only)
    # and reachability for cycle detection
    direct_delay = {}  # (si, di) → delay_samples  (positive delays only)
    for e in edges:
        si = ki.get(e.src_key)
        di = ki.get(e.dst_key)
        if si is None or di is None or e.delay_s <= 1e-9:
            continue  # zero or negative delay: no latency to compensate
        d_samp = int(round(e.delay_s * sr))
        if d_samp <= 0:
            continue
        key_pair = (si, di)
        direct_delay[key_pair] = max(direct_delay.get(key_pair, 0), d_samp)

    if not direct_delay:
        return {}

    # Reachability matrix via BFS (all edges, including zero-delay, for cycle check)
    reachable = [[False] * N for _ in range(N)]
    adj: list[list[int]] = [[] for _ in range(N)]
    for e in edges:
        si = ki.get(e.src_key)
        di = ki.get(e.dst_key)
        if si is not None and di is not None:
            adj[si].append(di)
    for start in range(N):
        stack = list(adj[start])
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            reachable[start][cur] = True
            stack.extend(adj[cur])

    # Remove feedback edges (si→di where di can reach si)
    ff_edges = {(si, di): d for (si, di), d in direct_delay.items()
                if not reachable[di][si]}

    if not ff_edges:
        return {}

    # Longest-path on the feedforward DAG using dynamic programming.
    # dist[i][j] = max total delay (samples) from node i to node j via ff edges.
    # Start with direct edges; then relax iteratively (Bellman-Ford style for DAG).
    dist = [[0] * N for _ in range(N)]
    for (si, di), d in ff_edges.items():
        dist[si][di] = max(dist[si][di], d)

    # Relax N-1 times (longest path in DAG has at most N-1 hops)
    for _ in range(N - 1):
        changed = False
        for (si, di), d in ff_edges.items():
            for j in range(N):
                if dist[di][j] > 0 or di == j:
                    new_d = d + (dist[di][j] if di != j else 0)
                    if new_d > dist[si][j]:
                        dist[si][j] = new_d
                        changed = True
        if not changed:
            break

    # Lead time for each source = max delay to any downstream node
    result = {}
    for i, key in enumerate(node_keys):
        lead = max(dist[i])
        if lead > 0:
            result[key] = lead
    return result


def solve_routing_with_ringdown(
    Src: np.ndarray,         # (N, T) complex128 — source signals (note duration only)
    edges: list,
    node_keys: list,
    sr: float,
    global_decay: float,
    fb: "FeedbackConfig",
    node_transforms: "dict | None" = None,
    coupled_transforms: "dict | None" = None,
) -> tuple:                  # (X_full: (N, T+ringdown_n), note_len: int)
    """Run ``solve_routing_complex`` with a zero-padded tail for ringdown.

    Returns
    -------
    X_full : np.ndarray, shape (N, T + ringdown_n)
        Full solved output including the tail.  Indices ``[:, :note_len]``
        correspond to the original note; ``[:, note_len:]`` are the tail.
    note_len : int
        Original T (= ``Src.shape[1]``).  Callers that need to know where
        the note ended (e.g. for accumulation timing) use this.
    """
    note_len = Src.shape[1]
    ringdown_n = estimate_ringdown_samples(edges, sr, global_decay, fb)
    if ringdown_n > 0:
        tail = np.zeros((Src.shape[0], ringdown_n), dtype=np.complex128)
        Src_ext = np.concatenate([Src, tail], axis=1)
    else:
        Src_ext = Src
    X_full = solve_routing_complex(Src_ext, edges, node_keys, sr, global_decay,
                                   node_transforms=node_transforms,
                                   coupled_transforms=coupled_transforms)
    return X_full, note_len


# ---------------------------------------------------------------------------
# Parametric routing — extract scalar time series from complex signal nodes
# ---------------------------------------------------------------------------

def extract_param_series(arr: np.ndarray, extractor: str) -> np.ndarray:
    """Convert a complex128 time series to a float64 scalar series.

    Parameters
    ----------
    arr       complex128 array of length T (one node's routing output)
    extractor see ParamEdge docstring for valid values
    """
    if extractor == "real":
        return arr.real.astype(np.float64)
    if extractor == "imag":
        return arr.imag.astype(np.float64)
    if extractor == "phase":
        return np.angle(arr)
    if extractor == "energy":
        return (arr.real ** 2 + arr.imag ** 2).astype(np.float64)
    if extractor == "rms":
        mag = np.abs(arr)
        win = 128
        kernel = np.ones(win, dtype=np.float64) / win
        return np.sqrt(np.convolve(mag ** 2, kernel, mode="same"))
    # default: "magnitude"
    return np.abs(arr)


def solve_param_routing(
    signal_map:     Dict[str, np.ndarray],    # node_key → complex128 array (routing output)
    param_edges:    list,                      # list[ParamEdge]
    param_defaults: Dict[str, float],          # param_node_key → default scalar value
    param_bounds:   Dict[str, tuple],          # param_node_key → (low, high)
) -> Dict[str, np.ndarray]:                   # param_node_key → float64 time series
    """Accumulate scalar parameter time series from ParamEdge extraction rules.

    Each ParamEdge reads from a signal node in *signal_map*, extracts a scalar
    float64 series using the specified extractor, scales by weight, and adds it
    to the destination param node's accumulator.

    Param nodes not referenced by any edge return their default value as a
    constant time series.  All results are clamped to their declared bounds.
    """
    T = max((len(v) for v in signal_map.values()), default=0)
    result: Dict[str, np.ndarray] = {
        key: np.full(T, default, dtype=np.float64)
        for key, default in param_defaults.items()
    }

    for e in param_edges:
        if e.src_key not in signal_map:
            continue
        extracted = extract_param_series(signal_map[e.src_key], e.extractor) * e.weight
        dst = e.dst_key
        if dst not in result:
            result[dst] = np.zeros(T, dtype=np.float64)
        L = min(len(extracted), len(result[dst]))
        result[dst][:L] += extracted[:L]

    for key, (lo, hi) in param_bounds.items():
        if key in result:
            result[key] = np.clip(result[key], lo, hi)

    return result


# ---------------------------------------------------------------------------
# High-level mixer
# ---------------------------------------------------------------------------

class RoutingMixer:
    """High-level API: apply a RoutingGraph to named complex signal arrays.

    Usage
    -----
    1.  Build a RoutingGraph with add_node / set_weight / set_angle_rad / set_delay_s.
    2.  Pass it with a sample_rate to RoutingMixer.
    3.  Call mix(sources) where sources is {key: ndarray (complex or real, shape (T,))}.
        Returns {key: complex128 ndarray} — one entry per registered node.

    The mixer is stateless (no internal audio buffer).  Call mix() as many
    times as needed with different source dicts; the RoutingGraph can be
    mutated between calls.

    Auto-registration
    -----------------
    Any source key not in graph.nodes is automatically registered before the
    solve so it is included in the output dict.

    Signal compatibility
    --------------------
    Real (float) inputs are promoted to complex128 before the solve.
    The solver preserves the analytic nature of complex inputs.
    """

    def __init__(self, graph: RoutingGraph, sample_rate: float = 48_000.0) -> None:
        self._graph = graph
        self._sr = float(sample_rate)

    @property
    def graph(self) -> RoutingGraph:
        return self._graph

    @property
    def sample_rate(self) -> float:
        return self._sr

    def mix(self, sources: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply routing graph to named source arrays.

        Parameters
        ----------
        sources:
            Mapping of node_key → 1-D numpy array (complex128 or any real dtype).
            Arrays may differ in length; shorter ones are zero-padded to match
            the longest.  Keys not in graph.nodes are auto-added.

        Returns
        -------
        dict[str, np.ndarray]
            Routed output for every node in the graph (complex128, shape (T,)).
            The output for each node is the result of the routing solve — it
            includes the node's own source signal plus all routed contributions.
        """
        # Collect all keys (graph nodes + any extra source keys)
        keys = list(self._graph.nodes)
        for k in sources:
            if k not in keys:
                keys.append(k)

        T = max((len(np.asarray(v)) for v in sources.values()), default=0)
        if T == 0:
            return {k: np.zeros(0, dtype=np.complex128) for k in keys}

        N = len(keys)
        Src = np.zeros((N, T), dtype=np.complex128)
        for i, k in enumerate(keys):
            if k in sources:
                arr = np.asarray(sources[k], dtype=np.complex128)
                L = min(len(arr), T)
                Src[i, :L] = arr[:L]

        g = self._graph
        global_decay = (
            max(0.0, 1.0 - float(g.feedback.decay))
            if g.feedback.enabled else 1.0
        )

        X = solve_routing_complex(Src, g.edges, keys, self._sr, global_decay)
        return {k: X[i] for i, k in enumerate(keys)}

    def mix_to_mono(self, sources: Dict[str, np.ndarray]) -> np.ndarray:
        """Route all sources and return the element-wise sum of all node outputs.

        The result is a complex128 array of length T.  Take .real for a
        mono audio signal or use a projection from signal_generator_v2.
        """
        out = self.mix(sources)
        if not out:
            return np.zeros(0, dtype=np.complex128)
        arrays = list(out.values())
        total = arrays[0].copy()
        for a in arrays[1:]:
            total = total + a
        return total

    def mix_and_project(
        self,
        sources: Dict[str, np.ndarray],
        output_key: Optional[str] = None,
    ) -> np.ndarray:
        """Route sources and return a float32 real projection for audio output.

        Parameters
        ----------
        output_key:
            If given, project only that node's output signal.
            If None (default), sum all node outputs first.
        """
        out = self.mix(sources)
        if output_key is not None:
            sig = out.get(output_key, np.zeros(0, dtype=np.complex128))
        else:
            if not out:
                return np.zeros(0, dtype=np.float32)
            arrays = list(out.values())
            sig = arrays[0].copy()
            for a in arrays[1:]:
                sig = sig + a
        return sig.real.astype(np.float32)

    def mix_stereo_quadrature(
        self,
        sources: Dict[str, np.ndarray],
        output_key: Optional[str] = None,
    ):
        """Route sources and return (L, R) float32 stereo quadrature pair.

        L = real part, R = imaginary part of the analytic routed signal.
        No phase folding — each unique phase state maps to a distinct (L, R) point.
        """
        out = self.mix(sources)
        if output_key is not None:
            sig = out.get(output_key, np.zeros(0, dtype=np.complex128))
        else:
            if not out:
                z = np.zeros(0, dtype=np.float32)
                return z, z
            arrays = list(out.values())
            sig = arrays[0].copy()
            for a in arrays[1:]:
                sig = sig + a
        L = sig.real.astype(np.float32)
        R = sig.imag.astype(np.float32)
        return L, R
