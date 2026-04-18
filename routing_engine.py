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
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoutingEdge:
    """Directed edge: applies weight * exp(i*angle_rad) to src, optionally delayed."""
    src_key:   str   = ""
    dst_key:   str   = ""
    weight:    float = 0.0   # amplitude coefficient, typically in [-2, 2]
    angle_rad: float = 0.0   # phase rotation applied to src before mixing (radians)
    delay_s:   float = 0.0   # per-edge propagation delay; 0 = synchronous


@dataclass
class ParamEdge:
    """Extraction edge: converts a routed complex signal into a scalar parameter time series.

    After the routing solve, for each param node p:

        float_series(t) = Σ_edges  weight * extract(X[src_key](t), extractor)

    This operates on the *already-solved* signal arrays — the heavy routing math is
    done by the existing signal solver; ParamEdge only describes the readout.

    Extractors
    ----------
    magnitude  |z|          — always positive; useful for energy/density control
    real       Re(z)        — signed; follows the analytic real part
    imag       Im(z)        — signed quadrature
    phase      arg(z)       — [-π, π]; useful for pitch-tracking
    energy     |z|²         — squared magnitude; heavier weighting of loud parts
    rms        running RMS  — smoothed magnitude (128-sample window)
    """
    src_key:   str   = ""
    dst_key:   str   = ""           # key of the destination ParamNode
    weight:    float = 1.0
    extractor: str   = "magnitude"  # see docstring

    def to_dict(self) -> dict:
        return {"src": self.src_key, "dst": self.dst_key,
                "w": self.weight, "extractor": self.extractor}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamEdge":
        return cls(src_key=str(d.get("src", "")), dst_key=str(d.get("dst", "")),
                   weight=float(d.get("w", 1.0)), extractor=str(d.get("extractor", "magnitude")))


@dataclass
class FeedbackConfig:
    """Global feedback solve parameters."""
    enabled:        bool  = False
    delay_s:        float = 0.0   # legacy global-delay field (per-edge delay preferred)
    decay:          float = 0.0   # 0–1: global amplitude loss per feedback cycle
    max_iterations: int   = 8     # Neumann truncation depth for instantaneous fallback
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

    def add_node(self, key: str) -> None:
        """Register a node key if not already present."""
        if key not in self.nodes:
            self.nodes.append(key)

    def node_keys(self) -> list:
        """Return the ordered list of registered node keys."""
        return list(self.nodes)

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
        return {
            "nodes": list(self.nodes),
            "edges": [
                {"src": e.src_key, "dst": e.dst_key, "w": e.weight,
                 "angle": e.angle_rad, "delay": e.delay_s}
                for e in self.edges
            ],
            "param_edges": [e.to_dict() for e in self.param_edges],
            "feedback": {
                "enabled":           self.feedback.enabled,
                "delay_s":           self.feedback.delay_s,
                "decay":             self.feedback.decay,
                "max_iterations":    self.feedback.max_iterations,
                "ringdown_mode":     self.feedback.ringdown_mode,
                "ringdown_max_s":    self.feedback.ringdown_max_s,
                "ringdown_threshold": self.feedback.ringdown_threshold,
            },
            "latency_compensation": self.latency_compensation,
        }

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
            ))
        fb = d.get("feedback", {})
        g.feedback = FeedbackConfig(
            enabled=bool(fb.get("enabled", False)),
            delay_s=float(fb.get("delay_s", 0.0)),
            decay=float(fb.get("decay", 0.0)),
            max_iterations=int(fb.get("max_iterations", 8)),
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


def solve_routing_complex(
    Src: np.ndarray,        # (N, T) complex128 — independent source signals
    edges: list,            # list[RoutingEdge]
    node_keys: list,        # canonical key list, len == N
    sr: float,
    global_decay: float = 1.0,
    node_transforms: "dict | None" = None,
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
        The callable receives the node's accumulated input (the sum of all
        routed contributions plus its independent source), and returns the
        node's output.  The output is what downstream nodes see.  The
        callable must preserve the analytic signal — return complex128,
        modify only the components it is responsible for, leave the rest
        intact.
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
    transforms = node_transforms or {}

    # Index set of nodes that have a nonlinear transform
    transform_idx: set = set()
    for key in transforms:
        ti = ki.get(key)
        if ti is not None:
            transform_idx.add(ti)

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
    try:
        M = np.linalg.inv(IW)
    except np.linalg.LinAlgError:
        M = np.linalg.inv(IW + np.eye(N, dtype=np.complex128) * 1e-6)

    has_transforms = len(transform_idx) > 0

    # ── Helper: apply all node transforms to X in-place ────────────────
    def _apply_transforms(X: np.ndarray) -> np.ndarray:
        for key, fn in transforms.items():
            ti = ki.get(key)
            if ti is not None:
                X[ti] = fn(X[ti])
        return X

    # ── No delays, no transforms: exact linear solve ──────────────────
    if not delay_groups and not has_transforms:
        return M @ Src

    # ── No delays, with transforms: iterative fixed-point ─────────────
    if not delay_groups:
        X = M @ Src                       # initial estimate (linear solve)
        _apply_transforms(X)
        _softclip_complex(X)

        for _iter in range(max_iterations):
            X_prev = X.copy()
            # Full iterative step: for each node, compute accumulated input
            # (Src + W @ X_current), then apply transform (identity for linear
            # nodes, nonlinear callable for transform nodes).
            X_input = Src + W_inst @ X    # accumulated input for each node
            X_new = X_input.copy()
            for ti in transform_idx:
                X_new[ti] = transforms[node_keys[ti]](X_input[ti])
            _softclip_complex(X_new)

            delta = np.max(np.abs(X_new - X_prev))
            X = X_new
            if delta < convergence_eps:
                break
        return X

    # ── Delayed edges: causal chunk solver ─────────────────────────────
    # Chunk size = minimum nonzero delay.  Within each chunk, the
    # instantaneous system (including nonlinear transforms) is iterated
    # to convergence.
    d_min = min(delay_groups.keys())
    X = np.zeros((N, T), dtype=np.complex128)

    for t0 in range(0, T, d_min):
        t1 = min(t0 + d_min, T)
        chunk_len = t1 - t0

        # RHS = Src slice + delayed contributions from already-computed chunks
        rhs = Src[:, t0:t1].copy()
        for d, Wd in delay_groups.items():
            t_past = t0 - d
            if t_past >= 0:
                rhs += Wd @ X[:, t_past: t_past + chunk_len]

        if not has_transforms:
            # Pure linear chunk: exact solve
            X[:, t0:t1] = M @ rhs
        else:
            # Iterative solve within this chunk
            X_chunk = M @ rhs             # initial linear estimate
            _apply_transforms(X_chunk)
            _softclip_complex(X_chunk)

            for _iter in range(max_iterations):
                X_prev = X_chunk.copy()
                X_input = rhs + W_inst @ X_chunk
                X_chunk = X_input.copy()
                for ti in transform_idx:
                    X_chunk[ti] = transforms[node_keys[ti]](X_input[ti])
                _softclip_complex(X_chunk)

                delta = np.max(np.abs(X_chunk - X_prev))
                if delta < convergence_eps:
                    break
            X[:, t0:t1] = X_chunk

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
                                   node_transforms=node_transforms)
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
