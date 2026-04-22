"""Complex graph solver over arbitrary node tensors.

Core rules:
* one step() call resolves one dt sample
* all arithmetic is torch.complex128
* nodes receive whole tensors and may transform them directly
* edges contribute by ordinary complex multiplication with broadcast semantics
* zero-delay linear regions solve exactly
* nonlinear/cyclic SCCs resolve by fixed-point iteration
* delayed edges carry state across samples
* mixer-grid flow and patch-panel flow are one network surface per sample
"""
from __future__ import annotations

import concurrent.futures
import cmath
import contextlib
import math
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set

# ── Profiling ─────────────────────────────────────────────────────────────────
# Set to True to enable per-span timing instrumentation around solver internals.
# In normal mode every span() call is a zero-cost contextlib.nullcontext().
PROFILE_TIMING: bool = True

# How often (seconds) the background reporter prints accumulated statistics.
# Set to 0 to disable the background reporter entirely.
PROFILE_REPORT_INTERVAL_S: float = 5.0


@dataclass
class _NameStats:
    """Per-name aggregate statistics, updated inline by TimingBank.span()."""
    count:    int   = 0
    total_ms: float = 0.0   # wall-clock including all descendant spans
    self_ms:  float = 0.0   # wall-clock MINUS direct children (actual work here)
    mn_ms:    float = 1e18  # min per-call total_ms
    mx_ms:    float = 0.0   # max per-call total_ms
    depth:    int   = 0     # call depth at first observation (for indentation)
    order:    int   = 0     # insertion order for stable column display


class TimingBank:
    """Lightweight nested-span timing registry with O(1) per-span overhead.

    Self-time is computed inline using a thread-local child-accumulator stack:
    each span subtracts the elapsed time of any direct child spans so that
    ``self_ms`` reflects only work done *in this span*, not its callees.  The
    shared lock is acquired only for an O(1) stat-struct update at span exit —
    never during the body of the span or during I/O.

    Usage::

        with _T.span("outer"):
            with _T.span("outer.inner"):
                do_work()

        _T.report()   # print summary to stdout (non-destructive snapshot)
        _T.reset()    # clear accumulated stats

    When ``PROFILE_TIMING`` is ``False`` every ``span()`` call is a
    zero-cost ``contextlib.nullcontext()`` — no allocation, no overhead.
    """

    def __init__(self) -> None:
        self._stats: Dict[str, _NameStats] = {}
        self._next_order: int = 0
        self._lock = threading.Lock()
        self._local = threading.local()   # call-stack per thread

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time the enclosed block under *name*.

        Thread-safe.  Nests correctly across any call depth.  Child elapsed
        time is accumulated into the parent frame so ``self_ms`` is always
        accurate without a second pass over the records.
        """
        if not PROFILE_TIMING:
            yield
            return
        # Thread-local stack: each frame = [children_elapsed_ms_so_far]
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        stack: list = self._local.stack
        frame: list = [0.0]
        stack.append(frame)
        t_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t_start) * 1_000.0
            stack.pop()
            self_ms = elapsed - frame[0]           # subtract children
            if stack:
                stack[-1][0] += elapsed            # tell parent about us
            depth = len(stack)                     # 0 = top-level
            with self._lock:
                s = self._stats.get(name)
                if s is None:
                    s = _NameStats(depth=depth, order=self._next_order)
                    self._next_order += 1
                    self._stats[name] = s
                s.count    += 1
                s.total_ms += elapsed
                s.self_ms  += self_ms
                s.mn_ms     = min(s.mn_ms, elapsed)
                s.mx_ms     = max(s.mx_ms, elapsed)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, *, title: str = "TimingBank report", file=None) -> None:
        """Print a summary to *file* (default stdout).  Non-destructive snapshot."""
        if file is None:
            file = sys.stdout
        with self._lock:
            snap = {k: _NameStats(v.count, v.total_ms, v.self_ms, v.mn_ms, v.mx_ms, v.depth, v.order)
                    for k, v in self._stats.items()}
        if not snap:
            print(f"[{title}] — no timing data", file=file)
            return
        self._format_stats(snap, title=title, file=file)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        with self._lock:
            self._stats.clear()
            self._next_order = 0

    # ------------------------------------------------------------------
    # Background reporter
    # ------------------------------------------------------------------

    def start_reporter(self, interval_s: float = 5.0, title: str = "GraphSolver profile") -> None:
        """Start a daemon thread that prints statistics every *interval_s* seconds.

        All console I/O happens in the daemon — the solve thread is never
        touched.  Stats are drained after each report so each window covers
        only the preceding interval.  Safe to call multiple times (no-op if
        already running).
        """
        if interval_s <= 0:
            return
        with self._lock:
            if getattr(self, "_reporter_running", False):
                return
            self._reporter_running = True
        self._reporter_title    = title
        self._reporter_interval = interval_s

        def _loop() -> None:
            import io
            while self._reporter_running:
                time.sleep(self._reporter_interval)
                if not self._reporter_running:
                    break
                # Drain under lock; do all formatting/I/O outside it.
                with self._lock:
                    snap = {k: _NameStats(v.count, v.total_ms, v.self_ms, v.mn_ms, v.mx_ms, v.depth, v.order)
                            for k, v in self._stats.items()}
                    self._stats.clear()
                    self._next_order = 0
                if not snap:
                    continue
                buf = io.StringIO()
                self._format_stats(snap, title=self._reporter_title, file=buf)
                sys.stderr.write(buf.getvalue())   # single syscall, no interleaving
                sys.stderr.flush()

        t = threading.Thread(target=_loop, daemon=True, name="TimingBank-reporter")
        t.start()

    def stop_reporter(self) -> None:
        """Signal the reporter thread to stop after its current sleep."""
        self._reporter_running = False

    # ------------------------------------------------------------------
    # Internal formatter
    # ------------------------------------------------------------------

    def _format_stats(self, snap: Dict[str, "_NameStats"], *, title: str, file) -> None:
        if not snap:
            return
        # Display order: sort by (depth, insertion order) so parents precede children.
        ordered = sorted(snap.keys(), key=lambda k: (snap[k].depth, snap[k].order))
        max_name_w = max(len(n) + snap[n].depth * 2 for n in ordered)
        name_col   = max(max_name_w, 36)
        sep        = "─" * (name_col + 72)
        lines = [f"\n{sep}", f" {title}", sep]
        hdr   = "span".ljust(name_col)
        lines.append(
            f" {hdr}  {'calls':>6}  {'self ms':>10}  {'self avg':>9}"
            f"  {'total ms':>10}  {'tot avg':>9}"
        )
        lines.append(sep)
        for name in ordered:
            s      = snap[name]
            indent = "  " * s.depth
            label  = (indent + name).ljust(name_col)
            s_avg  = s.self_ms  / s.count
            t_avg  = s.total_ms / s.count
            lines.append(
                f" {label}  {s.count:>6}  {s.self_ms:>10.3f}  {s_avg:>9.3f}"
                f"  {s.total_ms:>10.3f}  {t_avg:>9.3f}"
            )
        lines.append(sep + "\n")
        print("\n".join(lines), file=file)


# Module-level singleton — import and instrument from anywhere::
#
#   from graph_solver import _T, PROFILE_TIMING
#   with _T.span("my_span"):
#       ...
#   _T.report()
_T: TimingBank = TimingBank()

# Auto-start the background reporter if profiling is enabled.
if PROFILE_TIMING and PROFILE_REPORT_INTERVAL_S > 0:
    _T.start_reporter(interval_s=PROFILE_REPORT_INTERVAL_S, title="GraphSolver live profile")

import torch
import torch.nn as nn
from torch import Tensor

_CDTYPE = torch.complex128


def _canonical_complex(x: Tensor | complex | float | int) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(_CDTYPE)
    return torch.tensor(x, dtype=_CDTYPE)


def _broadcast_shape(shapes: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    out: tuple[int, ...] = ()
    for shape in shapes:
        out = torch.broadcast_shapes(out, tuple(shape))
    return out


def _broadcast_to(x: Tensor | complex | float | int, shape: tuple[int, ...], device: torch.device) -> Tensor:
    t = _canonical_complex(x).to(device)
    return torch.broadcast_to(t, shape) if shape else t


def _apply_saturation(policy: str, x: Tensor, knee: float) -> Tensor:
    policy = str(policy or "")
    if not policy:
        return x
    mag = x.abs()
    phase = torch.exp(1j * x.angle())
    k = max(float(knee), 1e-12)
    if policy == "tanh":
        out_mag = k * torch.tanh(mag / k)
    elif policy == "hardclip":
        out_mag = torch.clamp(mag, max=k)
    elif policy == "softclip":
        out_mag = (k * mag) / (k + mag)
    else:
        return x
    return out_mag * phase


def _add_broadcast(acc: Optional[Tensor], value: Tensor) -> Tensor:
    if acc is None:
        return value
    shape = torch.broadcast_shapes(tuple(acc.shape), tuple(value.shape))
    return torch.broadcast_to(acc, shape) + torch.broadcast_to(value, shape)


def _node_tag(keys: "Sequence[str]", max_keys: int = 3) -> str:
    """Build a short, dot-safe profiling tag from one or more node keys.

    Dots in key names are replaced with '/' so the tag can be appended to a
    dotted span name without creating spurious hierarchy levels.  When there
    are more than *max_keys* keys the count is appended instead.
    """
    safe = [k.replace(".", "/") for k in keys[:max_keys]]
    tag = "+".join(safe) if safe else "?"
    if len(keys) > max_keys:
        tag += f"+…{len(keys)}"
    return tag


def _interp_payload(x0: Tensor, x1: Tensor, t: float, mode: str) -> Tensor:
    if mode == "linear":
        return x0 + t * (x1 - x0)
    mag0 = x0.abs()
    mag1 = x1.abs()
    ph0 = x0.angle()
    ph1 = x1.angle()
    dph = ((ph1 - ph0 + math.pi) % (2.0 * math.pi)) - math.pi
    return (mag0 + t * (mag1 - mag0)) * torch.exp(1j * (ph0 + t * dph))


def _tarjan_sccs(node_keys: list[str], edges: Iterable["TensorEdge"], sample_rate: float) -> list[list[int]]:
    n = len(node_keys)
    ki = {k: i for i, k in enumerate(node_keys)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for edge in edges:
        if edge.delay_steps(sample_rate) != 0:
            continue
        si = ki.get(edge.src_key)
        di = ki.get(edge.dst_key)
        if si is None or di is None:
            continue
        adj[si].append(di)

    idx_ctr = [0]
    index = [-1] * n
    lowlink = [-1] * n
    stack: list[int] = []
    on_stack = [False] * n
    out: list[list[int]] = []

    def _dfs(v: int) -> None:
        index[v] = lowlink[v] = idx_ctr[0]
        idx_ctr[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj[v]:
            if index[w] == -1:
                _dfs(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            out.append(scc)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 1000))
    for v in range(n):
        if index[v] == -1:
            _dfs(v)
    return out


def _is_control_role(role: str, roles: Sequence[str]) -> bool:
    role = str(role or "")
    return role in set(str(r) for r in roles)


@dataclass(frozen=True)
class NetworkClock:
    """One network-level cycle clock with a dt port."""

    dt: float = 0.0
    sample_index: int = 0
    port_key: str = "__dt__"
    layer: str = ""

    def as_tensor(self, device: torch.device) -> Tensor:
        return torch.tensor(complex(float(self.dt), 0.0), dtype=_CDTYPE, device=device)


@dataclass(frozen=True)
class NodeLayerPresence:
    """Minimal layer visibility for one node."""

    input_layers: tuple[str, ...] = ()
    output_layers: tuple[str, ...] = ()
    parameter_inputs: bool = False
    parameter_outputs: bool = False


@dataclass(frozen=True)
class MixerNodeSpec:
    """Matrix mixdown dimensions: n_in signal channels → n_out signal channels."""

    n_in_channels: int = 1
    n_out_channels: int = 1


@dataclass(frozen=True)
class PortShapeSpec:
    """Formal shape advertisement for one semantic port form."""

    key: str = ""
    dims: tuple[str, ...] = ()
    batch_dims: tuple[int, ...] = ()
    channel_dims: tuple[int, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class PortSliceBinding:
    """Describe which slice of a port payload is acted on."""

    key: str = ""
    src_slice: str = ""
    dst_slice: str = ""
    weight_slice: str = ""


@dataclass(frozen=True)
class WeightShapeBinding:
    """Describe a weight form and how it correlates to port data slices."""

    key: str = ""
    weight_dims: tuple[str, ...] = ()
    src_port_key: str = ""
    dst_port_key: str = ""
    slice_bindings: tuple[PortSliceBinding, ...] = ()
    semantic_role: str = ""


@dataclass(frozen=True)
class KnobBinding:
    """Programmatic parameter-link origin, usually not user-exposed on the patch panel."""

    key: str = ""
    param_path: str = ""
    semantic_role: str = ""
    patch_visible: bool = False


@dataclass(frozen=True)
class SemanticPortContract:
    """Formal semantic port advertisement carried by a node archetype."""

    key: str = ""
    label: str = ""
    direction: str = "in"   # in | out
    domain: str = "signal"  # signal | control | parameter | mixed
    semantic_role: str = ""
    allowed_shapes: tuple[PortShapeSpec, ...] = ()
    weight_bindings: tuple[WeightShapeBinding, ...] = ()
    knob_bindings: tuple[KnobBinding, ...] = ()
    icl_meta_sim: str = ""
    mixer_grid_capable: bool = False


@dataclass(frozen=True)
class DataPort:
    """Named semantic port on a router-side patch panel."""

    key: str
    direction: str = "in"  # in | out
    semantic_label: str = ""
    data_label: str = ""


@dataclass(frozen=True)
class PortSet:
    """Named addressed port family for router nodes."""

    key: str
    ports: tuple[DataPort, ...] = ()
    retains_identity: bool = True


@dataclass(frozen=True)
class RouterNodeSpec:
    """Structured dispersal by address through named port sets."""

    port_sets: tuple[PortSet, ...] = ()
    retains_identity: bool = True
    addressed: bool = True


@dataclass(frozen=True)
class NodeArchetype:
    """Base authored node archetype."""

    mixer_layer_participations: tuple[str, ...] = ()
    semantic_ports: tuple[SemanticPortContract, ...] = ()


@dataclass(frozen=True)
class RouterArchetype(NodeArchetype):
    """Router archetype: addressed dispersal with identity-preserving port sets."""

    router: RouterNodeSpec = field(default_factory=RouterNodeSpec)


class AnalyticArchetype(nn.Module):
    """Shared dispatch daemon for TensorNodes of the same analytic type.

    One instance is created per analytic class (e.g. one for all
    ``LFODefinition`` nodes, one for all ``AnalyticVoice`` nodes).  Every
    TensorNode of the type registers itself; the solver enqueues their
    inputs, fires all archetypes concurrently within each topological wave,
    then scatters results back into the output dict.

    Per-tick protocol
    -----------------
    1. ``register_node(key)`` — once per node at graph build time.
    2. ``enqueue(key, x, knob_module)`` — solver stages each node's
       accumulated input alongside a reference to its ``KnobDrivenModule``.
    3. ``fire(forward_fn)`` — called in a dedicated worker thread.  Gathers
       inputs, stacks a batch tensor, calls ``forward_fn`` once (or identity
       if ``None``), scatters outputs to ``_results``, updates ``state_bank``.
    4. ``get_result(key)`` / ``clear_results()`` — solver reads then clears.

    ``forward_fn`` signature (once analytic internals are wired)::

        forward_fn(
            x_batch:      Tensor[B, ...],
            knob_modules: list[KnobDrivenModule],
            state:        Optional[Tensor[B, S]],
            keys:         list[str],
        ) -> (y_batch: Tensor[B, ...], new_state: Optional[Tensor[B, S]])

    Until then ``forward_fn=None`` → identity passthrough.

    State bank
    ----------
    ``state_bank[N, S]`` holds per-node sub-dt state, SM state, and any
    unfinished-business cache.  It is addressed by row index (``node_index``
    mapping).  All state for the entire archetype lives in one contiguous
    tensor so scatter/gather is a single indexed assignment.
    """

    def __init__(
        self,
        archetype_key: str,
        state_width: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.archetype_key = str(archetype_key)
        self.state_width = int(state_width)
        self.device = device or torch.device("cpu")
        self._node_keys: list[str] = []
        self.node_index: Dict[str, int] = {}
        self._lock = threading.Lock()  # guards registration only
        if state_width > 0:
            self.register_buffer(
                "state_bank",
                torch.zeros(0, state_width, dtype=_CDTYPE, device=self.device),
            )
        else:
            self.state_bank: Optional[Tensor] = None
        # Per-tick staging — never shared between threads
        self._pending: list[tuple[str, Tensor, Any]] = []
        self._results: Dict[str, Tensor] = {}

    def register_node(self, node_key: str) -> int:
        """Register *node_key*, return its row index.  Thread-safe."""
        with self._lock:
            if node_key in self.node_index:
                return self.node_index[node_key]
            idx = len(self._node_keys)
            self._node_keys.append(node_key)
            self.node_index[node_key] = idx
            if self.state_width > 0 and self.state_bank is not None:
                new_row = torch.zeros(
                    1, self.state_width, dtype=_CDTYPE, device=self.device
                )
                self.state_bank = torch.cat([self.state_bank, new_row], dim=0)
            return idx

    def enqueue(self, node_key: str, x: Tensor, knob_module: Any = None) -> None:
        """Stage one node's accumulated input for this tick."""
        self._pending.append((node_key, x, knob_module))

    def fire(self, forward_fn: Optional[Callable] = None) -> None:
        """Drain the queue: batch inputs, call forward_fn once, scatter results.

        Runs in a dedicated worker thread — one thread per archetype per
        topological wave.  Safe because each ``AnalyticArchetype`` owns its
        own ``_pending``, ``_results``, and ``state_bank``.
        """
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        keys = [k for k, _, _ in pending]
        xs   = [x for _, x, _ in pending]
        mods = [m for _, _, m in pending]

        shapes = [tuple(x.shape) for x in xs]
        common = _broadcast_shape(shapes) if shapes else ()
        x_batch = torch.stack(
            [_broadcast_to(x, common, self.device) for x in xs], dim=0
        )  # [B, ...]

        if forward_fn is not None:
            indices = [self.node_index[k] for k in keys]
            state_batch = (
                self.state_bank[indices]
                if self.state_bank is not None and self.state_bank.shape[0] > 0
                else None
            )
            result = forward_fn(x_batch, mods, state_batch, keys)
            y_batch, new_state = result if isinstance(result, tuple) else (result, None)
            if new_state is not None and self.state_bank is not None:
                idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    self.state_bank[idx_t] = new_state.detach()
        else:
            y_batch = x_batch  # identity until forward is wired

        for i, key in enumerate(keys):
            self._results[key] = y_batch[i]

    def get_result(self, node_key: str) -> Optional[Tensor]:
        return self._results.get(node_key)

    def clear_results(self) -> None:
        self._results.clear()


class DebugArchetype(AnalyticArchetype):
    """Drop-in debug archetype that accepts any node key.

    Forward: applies a trivial complex rotation (``e^{i * π/64}``) to the
    batched input so that the archetype path is provably distinct from an
    identity passthrough and shows up in the output as a tiny phase shift.
    Also accumulates per-key hit counts for post-solve inspection.

    Usage::

        arch = DebugArchetype()
        arch.register_node("any_key")   # register every node you want
        solver = GraphSolver(nodes, edges, archetypes={"debug": arch})
    """

    # Fixed phase rotation applied to every batch element.
    # Stored as a Python complex so it is safe at class-definition / import time.
    _ROTATION: complex = cmath.exp(1j * math.pi / 64)

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(archetype_key="debug", state_width=0, device=device)
        self.hit_counts: Dict[str, int] = {}

    def fire(self, forward_fn: Optional[Callable] = None) -> None:  # type: ignore[override]
        """Batch all pending inputs, apply phase rotation, scatter results."""
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        keys = [k for k, _, _ in pending]
        xs   = [x for _, x, _ in pending]

        shapes = [tuple(x.shape) for x in xs]
        common = _broadcast_shape(shapes) if shapes else ()
        x_batch = torch.stack(
            [_broadcast_to(x, common, self.device) for x in xs], dim=0
        )

        rot = _canonical_complex(self._ROTATION).to(self.device)
        y_batch = x_batch * rot

        for i, key in enumerate(keys):
            self._results[key] = y_batch[i]
            self.hit_counts[key] = self.hit_counts.get(key, 0) + 1


@dataclass(frozen=True)
class MixerArchetype(NodeArchetype):
    """Mixer archetype: matrix mixdown plus physical presence."""

    mixer: MixerNodeSpec = field(default_factory=MixerNodeSpec)


@dataclass(frozen=True)
class PatchPanelContract:
    """Minimal contract label for cross-layer patch-panel links."""

    key: str
    semantic_role: str = ""
    src_port_set: str = ""
    dst_port_set: str = ""


# ── Channel / Level-grid structural types ─────────────────────────────────────

@dataclass(frozen=True)
class ChannelSpec:
    """One advertised channel on a level's routing grid axis.

    Metanodes declare their channels; the level grid only shows channels
    that have advertised participation on that level.

    ``direction="out"`` → this channel is a source, appears on the row axis.
    ``direction="in"``  → this channel is a destination, appears on the col axis.

    Labels and owner_key are metadata for the user-facing mixer table;
    they carry no routing semantics in the solver.
    """
    key: str
    label: str = ""
    direction: str = "out"   # "out" | "in"
    level_key: str = ""      # which level this channel participates on
    owner_key: str = ""      # metanode that declared this channel
    domain: str = "signal"   # signal | parameter | control | clock


@dataclass(frozen=True)
class EncoderNodeSpec:
    """Spec for an encoder metanode.

    An encoder owns an ``nn.Module`` interior and an ``Optimizer``.  It fires
    at solve-start, reads its input channels from the solver state, runs its
    forward pass, and writes parameter values to target nodes via param edges.
    Nothing in the fixed network is trainable by default; only params
    reachable from an encoder's output edges are updated.
    """
    input_channel_keys: tuple[str, ...] = ()
    output_param_keys: tuple[str, ...] = ()
    fire_phase: str = "at_solve_start"   # "at_solve_start" | "before_start"


@dataclass(frozen=True)
class LossNodeSpec:
    """Spec for a loss metanode: computes loss and calls backward at solve end."""
    prediction_key: str = ""
    reference_key: str = ""
    loss_fn_name: str = "complex_l2"


@dataclass(frozen=True)
class StepNodeSpec:
    """Spec for a step metanode: calls optimizer.step() at solve end (after loss)."""
    encoder_keys: tuple[str, ...] = ()


@dataclass
class RouterContract:
    """One routing rule declared on a :class:`RouterNode`.

    A contract is a **routing rule**.  :meth:`RouterNode.add_edges`
    (or :meth:`LevelGridRouter.add_edges`) appends one :class:`TensorEdge`
    per ``(src, dst)`` pair from this contract into whatever edge list you
    pass it.  The edge list is the live state of the network — modify it
    freely outside of an active solve.

    The edge list is the network.  Add, remove, or replace edges in it\n    freely \u2014 outside of an active solve \u2014 at any time.  The solver reads
    whatever is in the list when ``step()`` runs.

    Field guide
    ---------------
    ``src_keys``
        One or more node keys whose output feeds this contract.  Each
        ``(src, dst)`` pair becomes a separate ``TensorEdge``.
    ``dst_keys``
        One or more destination node keys.
    ``gain``
        Complex scalar weight on the materialised edges.  Default ``1+0j``.
    ``transform``
        Optional per-edge function applied by :meth:`TensorEdge.apply` after
        the weight multiply.  Mapped to ``TensorEdge.saturation_fn``.
        If you need a multi-input transform (e.g. bus compression across
        several sources) you must insert an intermediate :class:`TensorNode`
        with the appropriate ``transform`` callable — that is the correct
        expression of that topology in the graph.
    ``semantic_role``
        Free-text label propagated to materialised edges for diagnostics and
        patch-panel export.  No routing semantics.
    ``notes``
        Free prose for the modder.  Ignored at every stage.

    Example — 30 % wet send from two voices to a reverb bus
    --------------------------------------------------------
    ::

        RouterContract(
            src_keys=("voice_1_out", "voice_2_out"),
            dst_keys=("reverb_send",),
            gain=0.3 + 0j,
            semantic_role="wet_send",
        )
        # Materialises four TensorEdges:
        #   voice_1_out → reverb_send  weight=0.3
        #   voice_2_out → reverb_send  weight=0.3

    Example — parameter injection with a halving transform
    -------------------------------------------------------
    ::

        RouterContract(
            src_keys=("encoder_out",),
            dst_keys=("reverb_decay", "delay_feedback"),
            transform=lambda x: x * 0.5,
            semantic_role="param_injection",
        )
        # Materialises two TensorEdges each carrying the halving fn.
    """

    src_keys: tuple[str, ...] = ()
    dst_keys: tuple[str, ...] = ()
    gain: complex = 1.0 + 0.0j
    transform: Optional[Callable[[Tensor], Tensor]] = None
    semantic_role: str = ""
    notes: str = ""


@dataclass(frozen=True)
class TensorNode:
    """One node in the solve graph.

    Lifecycle flags and hooks
    -------------------------
    All nodes participate in the normal continuous solve through ``transform``
    (the ``fire_continuously`` path).  Four additional boundary hooks let a
    node run Python-side logic at orchestration boundaries:

    ``fire_before_start`` / ``hook_before_start``
        Once before the network epoch begins.  No solver statefulness
        guaranteed.  Default hook: no-op.

    ``fire_at_solve_start`` / ``hook_at_solve_start``
        On the first solver tick of each epoch, in addition to the
        continuous transform.  Network IS stateful.

    ``fire_at_solve_end`` / ``hook_at_solve_end``
        Once after the last solver tick of each epoch while the network
        remains fully stateful.  Use for loss calc / optimizer step.

    ``fire_after_end`` / ``hook_after_end``
        Once after the epoch fully closes.  No statefulness guaranteed.
        Use for checkpointing, I/O.

    ``fire_continuously`` is implicit: it IS ``transform``.  If ``transform``
    is ``None`` the node is a pure accumulator; setting
    ``fire_continuously=False`` records that intent without affecting the
    transform field.
    """
    key: str
    layer: str = ""
    transform: Optional[Callable[[Tensor], Tensor]] = None
    natural_rate_hz: float = 0.0
    layer_presence: NodeLayerPresence = field(default_factory=NodeLayerPresence)
    archetype: NodeArchetype = field(default_factory=NodeArchetype)
    # ── lifecycle flags ───────────────────────────────────────────────
    fire_before_start: bool = False
    fire_at_solve_start: bool = False
    fire_continuously: bool = True   # informational; actual hook is transform
    fire_at_solve_end: bool = False
    fire_after_end: bool = False
    # ── lifecycle hooks (None → no-op) ────────────────────────────────
    hook_before_start: Optional[Callable[[], None]] = None
    hook_at_solve_start: Optional[Callable[[], None]] = None
    hook_at_solve_end: Optional[Callable[[], None]] = None
    hook_after_end: Optional[Callable[[], None]] = None
    # ── channel declarations ──────────────────────────────────────────
    # Declare ChannelSpec entries here so that GraphSolver auto-builds a
    # LevelGridRouter for the corresponding level_key.  No manual LevelGrid
    # construction required; the solver creates one per unique level_key
    # and exposes it via solver.level_grid_routers[level_key].
    channels: tuple["ChannelSpec", ...] = ()
    # ── analytic module back-reference ────────────────────────────────
    # The analytic object (AnalyticVoice, LFODefinition, AnalyticModule,
    # ControlSlider, etc.) this node was built from.  None for virtual nodes.
    analytic_module: Optional[Any] = None
    # ── archetype dispatch key ─────────────────────────────────────────
    # Matches an AnalyticArchetype.archetype_key registered on the solver.
    # When set, the solver routes this node through the shared archetype
    # daemon instead of calling node.transform directly.
    archetype_key: str = ""


@dataclass(frozen=True)
class TensorEdge:
    src_key: str
    dst_key: str
    weight: Tensor | complex | float | int
    delay_s: float = 0.0
    delay_samples: int = 0
    analog_complex_delay: Tensor | complex | float | int = 1.0 + 0.0j
    coefficient_set: str = ""
    coefficient_names: tuple[str, ...] = ()
    crosstalk_weight: Tensor | complex | float | int = 0.0 + 0.0j
    saturation_policy: str = ""
    saturation_knee: float = 1.0
    saturation_fn: Optional[Callable[[Tensor], Tensor]] = None
    group: str = ""
    semantic_role: str = ""
    src_port_set: str = ""
    dst_port_set: str = ""
    src_port: str = ""
    dst_port: str = ""
    contract_key: str = ""
    contract_semantic_role: str = ""
    src_mask: Tensor | complex | float | int = 1.0 + 0.0j
    dst_mask: Tensor | complex | float | int = 1.0 + 0.0j
    activity_mask: Tensor | complex | float | int = 1.0 + 0.0j
    activity_contract: str = ""
    src_addresses: tuple[str, ...] = ()
    dst_addresses: tuple[str, ...] = ()

    def delay_steps(self, sample_rate: float) -> int:
        if int(self.delay_samples) > 0:
            return int(self.delay_samples)
        return max(0, int(round(float(self.delay_s) * float(sample_rate))))

    def apply(self, payload: Tensor) -> Tensor:
        out = _canonical_complex(self.weight).to(payload.device) * payload
        out = out * _canonical_complex(self.analog_complex_delay).to(payload.device)
        out = out * _canonical_complex(self.src_mask).to(payload.device)
        out = out * _canonical_complex(self.dst_mask).to(payload.device)
        out = out * _canonical_complex(self.activity_mask).to(payload.device)
        crosstalk = _canonical_complex(self.crosstalk_weight).to(payload.device)
        if torch.any(crosstalk != 0):
            out = out + crosstalk * payload
        if self.saturation_fn is not None:
            out = _canonical_complex(self.saturation_fn(out)).to(payload.device)
        else:
            out = _apply_saturation(self.saturation_policy, out, self.saturation_knee)
        return out


@dataclass(frozen=True)
class EdgeGroup:
    """Named edge family for authored grouping."""

    key: str
    semantic_role: str = ""
    edge_refs: tuple[tuple[str, str], ...] = ()


class LevelGrid(nn.Module):
    """Routing gain table for one level.

    The grid is ``channels_src × channels_dst``.  Metanodes advertise their
    channels on a level; only those channels appear on the grid's axes.
    A cell ``gains[i, j]`` being nonzero implicitly defines an edge from
    ``channels_src[i]`` to ``channels_dst[j]`` with that complex gain.

    The ``trainable_mask`` buffer is the whitelist.  A cell is learnable only
    if explicitly marked.  The containing optimizer must be given
    ``self.trainable_cells()`` to actually update those weights — nothing is
    trained by default.

    Labels and owner keys on ``ChannelSpec`` entries are metadata for the
    user-facing mixer table; the solver sees only keys and gains.

    Usage
    -----
    Build the grid, set gains, whitelist cells you want to learn.  Call
    :meth:`add_edges` to append the current nonzero-gain cells as
    ``TensorEdge`` objects into any edge list.  The edge list is the live
    state of the network; add, remove, or replace entries whenever you like
    (outside of an active solve).
    """

    def __init__(
        self,
        level_key: str,
        channels_src: Sequence[ChannelSpec],
        channels_dst: Sequence[ChannelSpec],
        gains: Optional[Tensor] = None,
        trainable_mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.level_key = level_key
        self.channels_src: tuple[ChannelSpec, ...] = tuple(channels_src)
        self.channels_dst: tuple[ChannelSpec, ...] = tuple(channels_dst)
        n_src = len(self.channels_src)
        n_dst = len(self.channels_dst)
        if gains is None:
            gains = torch.zeros(n_src, n_dst, dtype=_CDTYPE)
        self.gains = nn.Parameter(gains.to(_CDTYPE))
        tm = trainable_mask if trainable_mask is not None else torch.zeros(n_src, n_dst, dtype=torch.bool)
        self.register_buffer("trainable_mask", tm)

    # ── index helpers ─────────────────────────────────────────────────────────

    def _src_idx(self, key: str) -> int:
        return next(i for i, c in enumerate(self.channels_src) if c.key == key)

    def _dst_idx(self, key: str) -> int:
        return next(i for i, c in enumerate(self.channels_dst) if c.key == key)

    # ── structural mutation (no-grad) ─────────────────────────────────────────

    def set_gain(self, src_key: str, dst_key: str, value: complex) -> None:
        """Set a gain cell by channel key.  No-grad structural edit."""
        with torch.no_grad():
            self.gains[self._src_idx(src_key), self._dst_idx(dst_key)] = value

    def whitelist(self, src_key: str, dst_key: str) -> None:
        """Mark a cell as learnable (opts it into ``trainable_cells()``)."""
        self.trainable_mask[self._src_idx(src_key), self._dst_idx(dst_key)] = True

    # ── optimizer integration ─────────────────────────────────────────────────

    def trainable_cells(self) -> List[Tensor]:
        """Return 0-dim Parameter views for whitelisted cells.

        Pass the result to your optimizer so only those cells are stepped:

            opt = Adam(grid.trainable_cells(), lr=1e-3)
        """
        return [
            self.gains[i, j]
            for i in range(len(self.channels_src))
            for j in range(len(self.channels_dst))
            if bool(self.trainable_mask[i, j])
        ]

    # ── edge materialisation ──────────────────────────────────────────────────

    def add_edges(self, edges: List[TensorEdge]) -> None:
        """Append one ``TensorEdge`` into *edges* for every nonzero gain cell.

        *edges* is whatever edge list you are working with — it is the live
        state of the network.  Call this whenever you want the current grid
        gains reflected in that list.  The ``weight`` field of each appended
        edge is a 0-dim view into ``self.gains`` so autograd flows through
        whitelisted cells.  Zero-gain cells produce no edge.
        """
        for i, ch_src in enumerate(self.channels_src):
            for j, ch_dst in enumerate(self.channels_dst):
                g = self.gains[i, j]
                if g.abs().item() != 0.0:
                    edges.append(
                        TensorEdge(
                            src_key=ch_src.key,
                            dst_key=ch_dst.key,
                            weight=g,
                            semantic_role="level_grid",
                            contract_key=self.level_key,
                        )
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Router infrastructure
# ─────────────────────────────────────────────────────────────────────────────

#: Informational level key for routers that logically belong to the
#: end-of-graph position.  Has no runtime effect; used only for grouping
#: and diagnostics.
TERMINAL_LEVEL_KEY: str = "__terminal__"


class RouterNode(nn.Module):
    """Edge-authoring tool for contract-based routing.

    A helper for declaring contract-based routing rules and appending them to
    an edge list.

    The edge list is the network
    ----------------------------
    The solver's edge list is a plain mutable list.  It is the live state of
    the network — visible in the UI, editable at any time outside of an active
    solve.  Nodes can carry any callables they want.  Nothing enforces when
    you read or write the list; the solver simply reads whatever is in it at
    the start of each ``step()`` call.

    ``RouterNode`` is a convenience, not a gatekeeper.  It knows how to
    translate a list of :class:`RouterContract` objects into
    :class:`TensorEdge` entries and append them wherever you point it.
    You could skip it entirely and append ``TensorEdge`` objects directly.

    Quick subclassing guide
    -----------------------
    **Contract-based router** (recommended)::

        class MyVoiceSendRouter(RouterNode):
            def __init__(self):
                super().__init__(
                    level_key="voice_layer",
                    contracts=[
                        RouterContract(
                            src_keys=("v1_out", "v2_out"),
                            dst_keys=("reverb_in",),
                            gain=0.25 + 0j,
                            semantic_role="reverb_send",
                        ),
                    ],
                    node_key="voice_send_router",
                )

        router = MyVoiceSendRouter()
        router.add_edges(my_edge_list)   # append to whichever list you like

    **Override add_edges** for computed or conditional topology::

        class MyConditionalRouter(RouterNode):
            def add_edges(self, edges: List[TensorEdge]) -> None:
                super().add_edges(edges)            # base contracts
                if self.some_flag:
                    edges.append(TensorEdge("a", "b", 1.0))

    Parameters
    ----------
    level_key:
        Grouping label.  No effect on the solve; for organisation and
        diagnostics only.
    contracts:
        Routing rules appended by :meth:`add_edges`.
    node_key:
        Key for the optional :class:`TensorNode` returned by :meth:`build_node`.
        Defaults to ``"__router_<level_key>__"``.
    layer:
        Layer string forwarded to the ``TensorNode``.  Defaults to
        ``level_key``.
    """

    def __init__(
        self,
        level_key: str,
        contracts: Sequence["RouterContract"] = (),
        *,
        node_key: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.level_key = str(level_key)
        self.contracts: List[RouterContract] = list(contracts)
        self._node_key = node_key or f"__router_{level_key}__"
        self._layer = layer if layer is not None else level_key

    # ------------------------------------------------------------------
    def add_edges(self, edges: List["TensorEdge"]) -> None:
        """Append one ``TensorEdge`` per contract ``(src, dst)`` pair into *edges*.

        *edges* is any edge list — pass whatever list represents the current
        network state.  Each appended edge carries ``contract.gain`` as its
        weight and ``contract.transform`` as its ``saturation_fn``.

        Two contracts targeting the same ``(src, dst)`` produce two independent
        edges in the list; the solver sums them naturally.
        """
        for contract in self.contracts:
            for sk in contract.src_keys:
                for dk in contract.dst_keys:
                    edges.append(TensorEdge(
                        src_key=sk,
                        dst_key=dk,
                        weight=contract.gain,
                        saturation_fn=contract.transform,
                        semantic_role=contract.semantic_role,
                        contract_key=self._node_key,
                        contract_semantic_role=contract.semantic_role,
                    ))

    # ------------------------------------------------------------------
    def build_node(self) -> "TensorNode":
        """Return a passive ``TensorNode`` that anchors this router in the graph.

        The node has ``transform=None`` (pure accumulator) and does not
        participate in the linear solve.  Add it to the ``nodes`` list passed
        to :class:`GraphSolver` only when you want the router to appear in
        graph diagnostics or to be addressable as an edge endpoint.
        """
        return TensorNode(
            key=self._node_key,
            layer=self._layer,
            transform=None,
        )


class LevelGridRouter(RouterNode):
    """Edge-authoring tool backed by a :class:`LevelGrid` gain table.

    A :class:`RouterNode` backed by a :class:`LevelGrid` gain table.

    Use this when you want a matrix of named channel-to-channel gains to
    control which edges exist and at what weight.  The gain table is the
    direct source of truth for those edges — call :meth:`add_edges` on
    whatever edge list you are editing to reflect the current gain state.

    1.  Instantiate with source/destination :class:`ChannelSpec` lists.
    2.  Set gains: ``router.grid.set_gain(src_key, dst_key, value)``.
        Whitelist cells for learning: ``router.grid.whitelist(src_key, dst_key)``.
    3.  Call ``router.add_edges(my_edge_list)`` to append the nonzero cells.
        Do this again whenever the gain table changes and you want the edge
        list to reflect it (after removing any stale entries first).

    Because gain cells are ``nn.Parameter`` views, whitelisted edges carry
    live gradient connectivity automatically.
    """

    def __init__(
        self,
        level_key: str,
        channels_src: Sequence["ChannelSpec"],
        channels_dst: Sequence["ChannelSpec"],
        *,
        node_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            level_key=level_key,
            contracts=[],
            node_key=node_key or f"__lgrid_{level_key}__",
            layer=level_key,
        )
        self.grid = LevelGrid(
            level_key=level_key,
            channels_src=list(channels_src),
            channels_dst=list(channels_dst),
        )

    # ------------------------------------------------------------------
    def add_edges(self, edges: List["TensorEdge"]) -> None:
        """Delegate to :meth:`LevelGrid.add_edges`.

        Appends one :class:`TensorEdge` per nonzero gain cell into *edges*.
        Weight fields are live Parameter views.
        """
        self.grid.add_edges(edges)


class DefaultTerminalRouter(RouterNode):
    """No-op edge-authoring stub.

    Produces no edges.  Exists only as a concrete no-op subclass for cases
    where a router object is syntactically required but no additional edges
    are needed.  The solver does not register or consult this (or any router)
    at runtime.
    """

    def __init__(self) -> None:
        super().__init__(
            level_key=TERMINAL_LEVEL_KEY,
            contracts=[],
            node_key="__terminal_router__",
            layer="",
        )


@dataclass(frozen=True)
class SolveLayer:
    """Ordered layer entry for the per-dt solve."""

    key: str
    kind: str = "signal"  # signal | parameter
    accepts_from_any_layer: bool = False
    emits_to_any_layer: bool = False


@dataclass(frozen=True)
class LayerSolvePlan:
    """Minimal sequential layer solve skeleton with a trailing control phase."""

    signal_layers: tuple[str, ...]
    parameter_layer: SolveLayer
    control_feedback_roles: tuple[str, ...] = ("control", "param", "parameter", "control_feedback")
    control_feedback_iterations: int = 1
    cycle_entire_layer_stack: bool = True
    one_network_per_sample: bool = True


@dataclass
class SCCSpec:
    scc_id: int
    node_indices: list[int]
    node_keys: list[str]
    is_linear: bool
    natural_rate_hz: float = 0.0


@dataclass
class CondensedGraph:
    sccs: list[SCCSpec]
    node_to_scc: dict[int, int]
    node_keys: list[str]
    tier: int


class CyclicTensorBlock(nn.Module):
    """Fixed-point cyclic solver over one SCC."""

    def __init__(
        self,
        node_keys: Sequence[str],
        internal_edges: Sequence[TensorEdge],
        node_map: Dict[str, TensorNode],
        *,
        default_k_max: int,
        tol: float,
        sample_rate: float,
        use_z_cache: bool,
        interp_mode: str,
        timing_offset: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if interp_mode not in {"none", "linear", "polar"}:
            raise ValueError(f"Unsupported interp_mode {interp_mode!r}")

        self.node_keys = list(node_keys)
        self.internal_edges = list(internal_edges)
        self.node_map = node_map
        self.tol = float(tol)
        self.sample_rate = float(sample_rate)
        self.use_z_cache = bool(use_z_cache)
        self.interp_mode = str(interp_mode)
        self.timing_offset = float(timing_offset)
        self.device = device

        natural_rate_hz = max(float(node_map[key].natural_rate_hz) for key in self.node_keys)
        self.K_max = (
            max(int(default_k_max), math.ceil(self.sample_rate / natural_rate_hz))
            if natural_rate_hz > 0.0 else int(default_k_max)
        )

        self._cache: dict[str, Tensor] = {}
        self._x_prev: dict[str, Tensor] = {}

    def _infer_payload_shape(self, src_map: Dict[str, Tensor]) -> tuple[int, ...]:
        shapes: list[tuple[int, ...]] = []
        for value in src_map.values():
            shapes.append(tuple(value.shape))
        for value in self._cache.values():
            shapes.append(tuple(value.shape))
        return _broadcast_shape(shapes)

    def _solve_linear_seed(self, src_map: Dict[str, Tensor], payload_shape: tuple[int, ...]) -> Dict[str, Tensor]:
        m = len(self.node_keys)
        M = torch.zeros(payload_shape + (m, m), dtype=_CDTYPE, device=self.device)
        eye = torch.eye(m, dtype=_CDTYPE, device=self.device)
        M[...] = eye
        b = torch.zeros(payload_shape + (m,), dtype=_CDTYPE, device=self.device)
        index = {key: i for i, key in enumerate(self.node_keys)}

        for key, tensor in src_map.items():
            b[..., index[key]] = _broadcast_to(tensor, payload_shape, self.device)
        for edge in self.internal_edges:
            di = index[edge.dst_key]
            si = index[edge.src_key]
            coeff = _broadcast_to(edge.apply(torch.ones(payload_shape, dtype=_CDTYPE, device=self.device)), payload_shape, self.device)
            M[..., di, si] = M[..., di, si] - coeff

        solved = torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
        return {key: solved[..., i] for i, key in enumerate(self.node_keys)}

    def _apply_once(self, z_map: Dict[str, Tensor], src_map: Dict[str, Tensor], payload_shape: tuple[int, ...]) -> Dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for key in self.node_keys:
            acc = _broadcast_to(src_map.get(key, 0.0j), payload_shape, self.device)
            for edge in self.internal_edges:
                if edge.dst_key != key:
                    continue
                acc = acc + edge.apply(_broadcast_to(z_map.get(edge.src_key, 0.0j), payload_shape, self.device))
            transform = self.node_map[key].transform
            mapped = _canonical_complex(transform(acc) if transform is not None else acc).to(self.device)
            out[key] = _broadcast_to(mapped, payload_shape, self.device)
        return out

    # ------------------------------------------------------------------
    # IFT backward helpers
    # ------------------------------------------------------------------

    def _flatten_z(self, z_map: Dict[str, Tensor]) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
        """Flatten node payload dict to one 1D complex128 tensor."""
        parts: list[Tensor] = []
        shapes: list[tuple[str, tuple[int, ...]]] = []
        for key in self.node_keys:
            t = _canonical_complex(z_map.get(key, 0j)).to(self.device)
            shapes.append((key, tuple(t.shape)))
            parts.append(t.reshape(-1))
        flat = torch.cat(parts) if parts else torch.zeros(0, dtype=_CDTYPE, device=self.device)
        return flat, shapes

    def _unflatten_z(self, flat: Tensor, shapes: list[tuple[str, tuple[int, ...]]]) -> Dict[str, Tensor]:
        """Unflatten 1D complex128 tensor back to node payload dict."""
        result: dict[str, Tensor] = {}
        offset = 0
        for key, shape in shapes:
            n = math.prod(shape) if shape else 1
            chunk = flat[offset : offset + n]
            result[key] = chunk.reshape(shape) if shape else chunk.reshape(())
            offset += n
        return result

    def _ift_remat(
        self,
        z_star: Dict[str, Tensor],
        src_map: Dict[str, Tensor],
        payload_shape: tuple[int, ...],
    ) -> Dict[str, Tensor]:
        """Re-materialise the fixed point with IFT-corrected backward.

        Forward value: identical to z_star (z* = F(z*, src; params)).
        Backward: the gradient flowing through this tensor is corrected by
        the adjoint Picard solve  v = (I - J_F(z*)^T)^{-1} @ dL/dz*
        before propagating to parameters.  This is the Implicit Function
        Theorem gradient — exact, O(1) compute-graph depth.

        Mechanism
        ---------
        1. Run ONE step z_remat = F(z*_det, src; params) with grad enabled.
           z_remat == z* numerically (fixed-point property), but the
           computation graph links z_remat to differentiable params/transforms.
        2. Register a hook on the flattened z_remat tensor.  The hook
           intercepts dL/dz* and replaces it with v via adjoint Picard,
           using F evaluated at z* with grad on z (params detached).
        3. Return the unflattened z_remat.  Downstream autograd routes
           v^T @ J_F(params) to the parameter leaves — the IFT formula.
        """
        # Build flat z* for use in the adjoint closure (no grad needed here).
        z_star_flat, shapes = self._flatten_z({k: v.detach() for k, v in z_star.items()})

        # Detached src for the adjoint-only forward pass.
        src_det = {
            k: (v.detach() if isinstance(v, Tensor) else v)
            for k, v in src_map.items()
        }

        K_max = self.K_max
        tol = self.tol
        node_keys = self.node_keys
        internal_edges = self.internal_edges
        node_map = self.node_map
        device = self.device

        def F_flat_adjoint(z_flat: Tensor) -> Tensor:
            """F(z; src_det, params_det) — grad tracks z only, not params."""
            z_dict = self._unflatten_z(z_flat, shapes)
            out: dict[str, Tensor] = {}
            for key in node_keys:
                acc = _broadcast_to(src_det.get(key, 0j), payload_shape, device)
                for edge in internal_edges:
                    if edge.dst_key != key:
                        continue
                    w = edge.weight.detach() if isinstance(edge.weight, Tensor) else edge.weight
                    src_val = _broadcast_to(z_dict.get(edge.src_key, 0j), payload_shape, device)
                    acc = acc + _canonical_complex(w).to(device) * src_val
                transform = node_map[key].transform
                if transform is not None:
                    # Detach transform parameters for adjoint; only z tracks grad.
                    if isinstance(transform, nn.Module):
                        with torch.no_grad():
                            mapped = _canonical_complex(transform(acc.detach())).to(device)
                        # Straight-through: treat transform as identity for adjoint J_F(z)
                        out[key] = _broadcast_to(acc, payload_shape, device)
                    else:
                        out[key] = _broadcast_to(
                            _canonical_complex(transform(acc)).to(device), payload_shape, device
                        )
                else:
                    out[key] = _broadcast_to(acc, payload_shape, device)
            flat_out, _ = self._flatten_z(out)
            return flat_out

        def ift_hook(grad: Tensor) -> Tensor:
            """Replace dL/dz* with (I - J_F^T)^{-1} @ dL/dz* via adjoint Picard."""
            with torch.enable_grad():
                z_g = z_star_flat.detach().requires_grad_(True)
                F_val = F_flat_adjoint(z_g)

            v = grad.clone()
            for _ in range(K_max):
                jfT_v = torch.autograd.grad(
                    F_val, z_g, grad_outputs=v,
                    retain_graph=True, allow_unused=True,
                )[0]
                if jfT_v is None:
                    break
                v_new = grad + jfT_v
                if (v_new - v).abs().max().item() < tol:
                    v = v_new
                    break
                v = v_new
            return v

        # One forward step at z* with live params for the parameter grad path.
        z_remat = self._apply_once(
            {k: v.detach() for k, v in z_star.items()},
            src_map,
            payload_shape,
        )

        # Flatten, attach hook, unflatten.
        z_remat_flat, _ = self._flatten_z(z_remat)
        if z_remat_flat.requires_grad:
            z_remat_flat.register_hook(ift_hook)

        return self._unflatten_z(z_remat_flat, shapes)

    # ------------------------------------------------------------------

    def step(self, src_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        payload_shape = self._infer_payload_shape(src_map)
        if self.use_z_cache and self._cache:
            z_map = {key: _broadcast_to(self._cache[key], payload_shape, self.device) for key in self.node_keys}
        else:
            z_map = self._solve_linear_seed(src_map, payload_shape)

        # ── Picard / sub-dt iteration (always no_grad) ─────────────────
        if self.interp_mode == "none":
            with torch.no_grad():
                for _ in range(self.K_max):
                    z_new = self._apply_once(z_map, src_map, payload_shape)
                    residual = max(
                        (z_new[key] - z_map[key]).abs().max().item()
                        for key in self.node_keys
                    )
                    z_map = z_new
                    if residual < self.tol:
                        break
        else:
            prev_src = {
                key: _broadcast_to(self._x_prev.get(key, 0.0j), payload_shape, self.device)
                for key in self.node_keys
            }
            curr_src = {
                key: _broadcast_to(src_map.get(key, 0.0j), payload_shape, self.device)
                for key in self.node_keys
            }
            with torch.no_grad():
                for k in range(self.K_max):
                    t = min(max((k + self.timing_offset) / max(self.K_max, 1), 0.0), 1.0)
                    interp_src = {
                        key: _interp_payload(prev_src[key], curr_src[key], t, self.interp_mode)
                        for key in self.node_keys
                    }
                    z_map = self._apply_once(z_map, interp_src, payload_shape)
            self._x_prev = {key: curr_src[key].detach().clone() for key in self.node_keys}

        # ── Cache (detached — cache is purely state, never a grad path) ─
        if self.use_z_cache:
            self._cache = {key: z_map[key].detach().clone() for key in self.node_keys}

        # ── IFT re-materialisation when grad is required ────────────────
        if torch.is_grad_enabled():
            z_map = self._ift_remat(z_map, src_map, payload_shape)

        return z_map

    def reset(self) -> None:
        self._cache = {}
        self._x_prev = {}


class GraphSolver(nn.Module):
    """Directed complex graph solver over arbitrary node tensors."""

    def __init__(
        self,
        nodes: Sequence[TensorNode],
        edges: Sequence[TensorEdge],
        *,
        edge_groups: Sequence[EdgeGroup] = (),
        layer_order: Sequence[str] = (),
        parameter_layer_key: str = "param",
        control_feedback_iterations: int = 1,
        network_clock: Optional[NetworkClock] = None,
        sample_rate: float = 48_000.0,
        default_k_max: int = 64,
        convergence_eps: float = 1e-10,
        use_z_cache: bool = True,
        interp_mode: str = "none",
        timing_offset: float = 0.0,
        device: torch.device = torch.device("cpu"),
        archetypes: Optional[Dict[str, "AnalyticArchetype"]] = None,
    ) -> None:
        super().__init__()
        self.nodes = list(nodes)
        self.edges = [
            TensorEdge(
                e.src_key,
                e.dst_key,
                _canonical_complex(e.weight).to(device),
                e.delay_s,
                e.delay_samples,
                _canonical_complex(e.analog_complex_delay).to(device),
                e.coefficient_set,
                e.coefficient_names,
                _canonical_complex(e.crosstalk_weight).to(device),
                e.saturation_policy,
                e.saturation_knee,
                e.saturation_fn,
                e.group,
                e.semantic_role,
                e.src_port_set,
                e.dst_port_set,
                e.src_port,
                e.dst_port,
                e.contract_key,
                e.contract_semantic_role,
                _canonical_complex(e.src_mask).to(device),
                _canonical_complex(e.dst_mask).to(device),
                _canonical_complex(e.activity_mask).to(device),
                e.activity_contract,
                e.src_addresses,
                e.dst_addresses,
            )
            for e in edges
        ]
        self.edge_groups = list(edge_groups)
        self.parameter_layer_key = str(parameter_layer_key)
        self.network_clock = network_clock or NetworkClock()
        self.sample_rate = float(sample_rate)
        self.default_k_max = int(default_k_max)
        self.convergence_eps = float(convergence_eps)
        self.use_z_cache = bool(use_z_cache)
        self.interp_mode = str(interp_mode)
        self.timing_offset = float(timing_offset)
        self.device = device

        self.node_keys = [node.key for node in self.nodes]
        self.node_map = {node.key: node for node in self.nodes}
        self.node_index = {node.key: i for i, node in enumerate(self.nodes)}
        derived_layers = [str(node.layer or "") for node in self.nodes if str(node.layer or "")]
        ordered_layers = tuple(str(layer) for layer in layer_order) if layer_order else tuple(dict.fromkeys(derived_layers))
        self.solve_plan = LayerSolvePlan(
            signal_layers=ordered_layers,
            parameter_layer=SolveLayer(
                key=self.parameter_layer_key,
                kind="parameter",
                accepts_from_any_layer=True,
                emits_to_any_layer=True,
            ),
            control_feedback_iterations=max(1, int(control_feedback_iterations)),
        )

        self.meta_edges = [edge for edge in self.edges if edge.activity_contract or edge.src_addresses or edge.dst_addresses]
        self.network_links = tuple(self.edges)
        raw_sccs = _tarjan_sccs(self.node_keys, self.edges, self.sample_rate)
        node_to_scc: dict[int, int] = {}
        sccs: list[SCCSpec] = []
        for scc_id, raw in enumerate(reversed(raw_sccs)):
            keys = [self.node_keys[i] for i in raw]
            is_linear = all(self.node_map[key].transform is None for key in keys)
            natural_rate_hz = max(float(self.node_map[key].natural_rate_hz) for key in keys)
            for ni in raw:
                node_to_scc[ni] = scc_id
            sccs.append(SCCSpec(
                scc_id=scc_id,
                node_indices=list(raw),
                node_keys=keys,
                is_linear=is_linear,
                natural_rate_hz=natural_rate_hz,
            ))
        tier = 1 if all(scc.is_linear for scc in sccs) else (3 if len(sccs) == 1 else 2)
        self.condensed = CondensedGraph(sccs, node_to_scc, list(self.node_keys), tier)
        self.sccs_by_layer: dict[str, list[SCCSpec]] = {layer: [] for layer in self.solve_plan.signal_layers}
        self.control_feedback_sccs: list[SCCSpec] = []
        for scc in self.condensed.sccs:
            node_layers = {str(self.node_map[key].layer or "") for key in scc.node_keys}
            assigned = False
            for layer in self.solve_plan.signal_layers:
                if layer in node_layers:
                    self.sccs_by_layer.setdefault(layer, []).append(scc)
                    assigned = True
                    break
            linked_to_patch = any(
                edge.contract_key and (edge.src_key in scc.node_keys or edge.dst_key in scc.node_keys)
                for edge in self.edges
            )
            if any(
                _is_control_role(edge.semantic_role, self.solve_plan.control_feedback_roles)
                for edge in self.edges
                if edge.src_key in scc.node_keys or edge.dst_key in scc.node_keys
            ) or linked_to_patch:
                self.control_feedback_sccs.append(scc)
            elif any(self.node_map[key].layer_presence.parameter_inputs or self.node_map[key].layer_presence.parameter_outputs for key in scc.node_keys):
                self.control_feedback_sccs.append(scc)
            elif not assigned and scc not in self.control_feedback_sccs:
                self.control_feedback_sccs.append(scc)

        self._first_tick: bool = True
        self.zero_delay_edges = [edge for edge in self.edges if edge.delay_steps(self.sample_rate) == 0]
        self.delayed_edges_by_steps: dict[int, list[TensorEdge]] = {}
        for edge in self.edges:
            d = edge.delay_steps(self.sample_rate)
            if d > 0:
                self.delayed_edges_by_steps.setdefault(d, []).append(edge)

        self.cyclic_blocks = nn.ModuleDict()
        for scc in self.condensed.sccs:
            if scc.is_linear:
                continue
            internal_edges = [
                edge for edge in self.zero_delay_edges
                if edge.src_key in scc.node_keys and edge.dst_key in scc.node_keys
            ]
            self.cyclic_blocks[str(scc.scc_id)] = CyclicTensorBlock(
                scc.node_keys,
                internal_edges,
                self.node_map,
                default_k_max=self.default_k_max,
                tol=self.convergence_eps,
                sample_rate=self.sample_rate,
                use_z_cache=self.use_z_cache,
                interp_mode=self.interp_mode,
                timing_offset=self.timing_offset,
                device=self.device,
            )

        self._delay_buffers: dict[int, list[dict[str, Tensor]]] = {
            d: [{} for _ in range(d)] for d in self.delayed_edges_by_steps
        }
        self._delay_pos: dict[int, int] = {d: 0 for d in self.delayed_edges_by_steps}
        self._last_outputs: dict[str, Tensor] = {}

        # Collect AnalyticArchetype instances from analytic_module fields.
        # The archetype is looked up on each node's KnobDrivenModule (.archetype).
        self.archetypes: Dict[str, "AnalyticArchetype"] = dict(archetypes) if archetypes else {}
        for node in self.nodes:
            am = node.analytic_module
            if am is not None:
                arch = getattr(am, "archetype", None)
                if isinstance(arch, AnalyticArchetype):
                    ak = arch.archetype_key
                    if ak not in self.archetypes:
                        self.archetypes[ak] = arch
        # Also accept archetypes referenced by archetype_key on nodes that have
        # no analytic_module (e.g. nodes registered with a DebugArchetype).
        for ak, arch in list(self.archetypes.items()):
            for node in self.nodes:
                if node.archetype_key == ak and node.key not in arch.node_index:
                    arch.register_node(node.key)

    def _fire_pending_archetypes(
        self,
        pending: Dict[str, "AnalyticArchetype"],
        outputs: Dict[str, Tensor],
    ) -> None:
        """Fire all pending archetypes concurrently, then scatter results."""
        if not pending:
            return
        with _T.span("solver.fire_archetypes"):
            with _T.span("solver.fire_archetypes.thread_pool"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as pool:
                    def _timed_fire(ak: str, arch) -> None:
                        with _T.span(f"solver.archetype.{ak.replace('.', '/')}"):
                            arch.fire()
                    futures = {pool.submit(_timed_fire, ak, arch): arch for ak, arch in pending.items()}
                    concurrent.futures.wait(futures)
            with _T.span("solver.fire_archetypes.scatter"):
                for arch in pending.values():
                    outputs.update(arch._results)
                    arch.clear_results()

    def _dispatch_layer_sccs(
        self,
        layer: str,
        base_src: Dict[str, Optional[Tensor]],
        outputs: Dict[str, Tensor],
    ) -> None:
        """Walk a layer's SCCs in topological order, dispatching archetype-eligible
        singleton SCCs concurrently within each causal wave.

        A *wave flush* is triggered whenever a pending (staged) node key appears
        as a zero-delay source for the next SCC.  This preserves topological
        correctness while maximising concurrency across independent archetypes.
        """
        pending: Dict[str, AnalyticArchetype] = {}  # archetype_key -> arch
        pending_keys: set[str] = set()               # node keys staged but not yet resolved

        def _flush() -> None:
            with _T.span("solver.dispatch.wave_flush"):
                self._fire_pending_archetypes(pending, outputs)
            pending.clear()
            pending_keys.clear()

        for scc in self.sccs_by_layer.get(layer, ()):
            # Flush if any input to this SCC comes from a still-pending node.
            if pending_keys and any(
                edge.src_key in pending_keys
                for edge in self.zero_delay_edges
                if edge.dst_key in scc.node_keys
            ):
                _flush()

            local_src = self._build_local_src(scc, base_src, outputs)

            # Archetype-eligible: singleton SCC with a registered archetype key.
            if len(scc.node_keys) == 1:
                key = scc.node_keys[0]
                ak = self.node_map[key].archetype_key
                if ak and ak in self.archetypes:
                    x = local_src.get(key, torch.zeros((), dtype=_CDTYPE, device=self.device))
                    am = self.node_map[key].analytic_module
                    self.archetypes[ak].enqueue(key, x, am)
                    pending[ak] = self.archetypes[ak]
                    pending_keys.add(key)
                    continue

            # Not archetype-eligible — solve immediately (blocks).
            with _T.span("solver.dispatch.solve_scc"):
                self._solve_scc(scc, base_src, outputs)

        # Final flush for any remaining staged nodes.
        if pending:
            _flush()

    def _solve_linear_region(self, node_keys: Sequence[str], src_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        m = len(node_keys)
        payload_shape = _broadcast_shape(
            [tuple(value.shape) for value in src_map.values()] +
            [tuple(self._last_outputs[key].shape) for key in node_keys if key in self._last_outputs]
        )
        M = torch.zeros(payload_shape + (m, m), dtype=_CDTYPE, device=self.device)
        M[...] = torch.eye(m, dtype=_CDTYPE, device=self.device)
        b = torch.zeros(payload_shape + (m,), dtype=_CDTYPE, device=self.device)
        index = {key: i for i, key in enumerate(node_keys)}

        for key, tensor in src_map.items():
            b[..., index[key]] = _broadcast_to(tensor, payload_shape, self.device)
        for edge in self.zero_delay_edges:
            if edge.src_key not in index or edge.dst_key not in index:
                continue
            di = index[edge.dst_key]
            si = index[edge.src_key]
            coeff = _broadcast_to(edge.apply(torch.ones(payload_shape, dtype=_CDTYPE, device=self.device)), payload_shape, self.device)
            M[..., di, si] = M[..., di, si] - coeff

        solved = torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
        return {key: solved[..., i] for i, key in enumerate(node_keys)}

    def _build_local_src(
        self,
        scc: SCCSpec,
        base_src: Dict[str, Optional[Tensor]],
        outputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        local_src: dict[str, Tensor] = {}
        for key in scc.node_keys:
            base = base_src.get(key)
            if base is not None:
                local_src[key] = _add_broadcast(local_src.get(key), base)
        for edge in self.zero_delay_edges:
            if edge.dst_key not in scc.node_keys or edge.src_key not in outputs:
                continue
            contrib = edge.apply(outputs[edge.src_key])
            local_src[edge.dst_key] = _add_broadcast(local_src.get(edge.dst_key), contrib)
        return local_src

    def _solve_scc(
        self,
        scc: SCCSpec,
        base_src: Dict[str, Optional[Tensor]],
        outputs: Dict[str, Tensor],
    ) -> None:
        with _T.span("solver.scc"):
            local_src = self._build_local_src(scc, base_src, outputs)
            node_tag = _node_tag(scc.node_keys)
            if str(scc.scc_id) in self.cyclic_blocks:
                with _T.span(f"solver.scc.cyclic.{node_tag}"):
                    solved = self.cyclic_blocks[str(scc.scc_id)].step(local_src)
            else:
                with _T.span(f"solver.scc.linear.{node_tag}"):
                    solved = self._solve_linear_region(scc.node_keys, local_src)
            outputs.update(solved)

    def step(self, ext: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with _T.span("solver.step"):
            unknown = sorted(set(ext) - set(self.node_keys))
            if unknown:
                raise KeyError(f"Unknown node inputs: {unknown}")

            if self._first_tick:
                self._first_tick = False
                for node in self.nodes:
                    if node.fire_at_solve_start and node.hook_at_solve_start is not None:
                        node.hook_at_solve_start()

            with _T.span("solver.step.inject"):
                base_src: dict[str, Optional[Tensor]] = {key: None for key in self.node_keys}
                if self.network_clock.port_key in base_src:
                    base_src[self.network_clock.port_key] = _add_broadcast(
                        base_src[self.network_clock.port_key],
                        self.network_clock.as_tensor(self.device),
                    )
                for key, value in ext.items():
                    base_src[key] = _add_broadcast(base_src[key], _canonical_complex(value).to(self.device))

            with _T.span("solver.step.delay_read"):
                for delay, edges in self.delayed_edges_by_steps.items():
                    snapshot = self._delay_buffers[delay][self._delay_pos[delay]]
                    for edge in edges:
                        src_val = snapshot.get(edge.src_key)
                        if src_val is None:
                            continue
                        contrib = edge.apply(src_val)
                        base_src[edge.dst_key] = _add_broadcast(base_src[edge.dst_key], contrib)

            src_map = {key: value for key, value in base_src.items() if value is not None}
            if self.condensed.tier == 1:
                with _T.span("solver.step.solve"):
                    with _T.span("solver.step.solve.linear_tier1"):
                        outputs = self._solve_linear_region(self.node_keys, src_map)
            else:
                with _T.span("solver.step.solve"):
                    outputs: dict[str, Tensor] = {}
                    for _ in range(self.solve_plan.control_feedback_iterations):
                        with _T.span("solver.step.solve.layers"):
                            for layer in self.solve_plan.signal_layers:
                                with _T.span("solver.step.solve.layers.dispatch"):
                                    self._dispatch_layer_sccs(layer, base_src, outputs)
                        with _T.span("solver.step.solve.control_feedback"):
                            for scc in self.control_feedback_sccs:
                                self._solve_scc(scc, base_src, outputs)
                        if not self.solve_plan.cycle_entire_layer_stack:
                            break

            with _T.span("solver.step.delay_write"):
                for delay in self.delayed_edges_by_steps:
                    pos = self._delay_pos[delay]
                    self._delay_buffers[delay][pos] = {key: value.detach().clone() for key, value in outputs.items()}
                    self._delay_pos[delay] = (pos + 1) % delay

            self._last_outputs = {key: value.detach().clone() for key, value in outputs.items()}
            return outputs

    def reset(self) -> None:
        self._first_tick = True
        for delay in self.delayed_edges_by_steps:
            self._delay_buffers[delay] = [{} for _ in range(delay)]
            self._delay_pos[delay] = 0
        for block in self.cyclic_blocks.values():
            block.reset()
        self._last_outputs = {}

    # ------------------------------------------------------------------
    # Lifecycle boundary dispatch
    # ------------------------------------------------------------------

    def dispatch_before_start(self) -> None:
        """Call before the epoch begins (no network statefulness guaranteed)."""
        for node in self.nodes:
            if node.fire_before_start and node.hook_before_start is not None:
                node.hook_before_start()

    def dispatch_at_solve_end(self) -> None:
        """Call after the last tick of the epoch (network still stateful).

        Intended for loss meta nodes: fire_at_solve_end=True,
        hook_at_solve_end=lambda: loss_node.compute_and_step().
        """
        for node in self.nodes:
            if node.fire_at_solve_end and node.hook_at_solve_end is not None:
                node.hook_at_solve_end()

    def dispatch_after_end(self) -> None:
        """Call after the epoch fully closes (no statefulness guaranteed)."""
        for node in self.nodes:
            if node.fire_after_end and node.hook_after_end is not None:
                node.hook_after_end()
