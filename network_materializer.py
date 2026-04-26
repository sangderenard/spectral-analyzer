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
import copy
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from analytic_driver import AnalyticPatch

_CDTYPE = torch.complex128


def _block_len(x: Tensor) -> int:
    """Return how many time samples a tensor-shaped payload represents."""
    if x.dim() >= 3:
        return max(1, int(x.shape[1]))
    return max(1, int(x.numel()))


def _shape_like(x: Tensor, value: Tensor) -> Tensor:
    """Broadcast scalar source output to the payload shape the solver supplied."""
    return torch.ones_like(x.to(_CDTYPE)) * value.to(_CDTYPE)


def _time_series_like(x: Tensor, series: Tensor) -> Tensor:
    """Broadcast a length-T series across ``x`` while preserving B/data/C axes."""
    z = x.to(_CDTYPE)
    t = _block_len(z)
    y = series.to(device=z.device)
    if y.dim() == 1:
        y = y.reshape(1, int(y.shape[0]), 1)
    elif y.dim() == 2:
        y = y.unsqueeze(-1)
    elif y.dim() == 0:
        y = y.reshape(1, 1, 1)
    if y.shape[1] < t:
        pad = y[:, -1:, ...].expand(*y.shape[:1], t - int(y.shape[1]), *y.shape[2:])
        y = torch.cat([y, pad], dim=1)
    y = y[:, :t, ...].to(_CDTYPE)
    if z.dim() >= 3:
        return torch.broadcast_to(y, torch.broadcast_shapes(tuple(y.shape), tuple(z.shape)))
    return y.reshape(tuple(z.shape) or ())


def _flatten_time_lanes(z: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Return lanes as ``(L, T)`` plus original shape for B,T,...,C tensors."""
    if z.dim() < 3:
        return z.reshape(1, -1), tuple(z.shape)
    moved = z.movedim(1, -1)
    return moved.reshape(-1, z.shape[1]), tuple(z.shape)


def _unflatten_time_lanes(lanes: Tensor, original_shape: tuple[int, ...]) -> Tensor:
    if len(original_shape) < 3:
        return lanes.reshape(original_shape)
    time_n = original_shape[1]
    outer_shape = original_shape[:1] + original_shape[2:]
    moved = lanes.reshape(*outer_shape, time_n)
    return moved.movedim(-1, 1)


def _block_faculty_for(analytic_obj):
    fn = getattr(type(analytic_obj), "block_faculty", None)
    return fn() if callable(fn) else None


def _archetype_key_for(analytic_obj) -> str:
    return str(getattr(type(analytic_obj), "ARCHETYPE_KEY", "") or "")


@dataclass
class CompiledGraph:
    solver: object
    output_node_keys: list[str]
    module_registry: dict[str, nn.Module]


def compile_nodes(
    nodes: "List",
    edges: "List",
    sample_rate: float,
    *,
    output_node_keys: "Optional[List[str]]" = None,
) -> CompiledGraph:
    """Construct a GraphSolver runtime bundle from pre-built nodes and edges.

    This is the permanent entry point for direct node construction — no
    AnalyticPatch involved.  Callers build TensorNodes and TensorEdges
    directly and hand them off here.

    The shared EdgeFifoBank is collected from whichever edge carries a
    ``fifo_bank`` attribute (all edges in a graph must share one bank
    instance).  If no edge carries one, a fresh bank is created.
    """
    _gs = _import_gs()

    try:
        from edge_fifo_bank import bank_for_edges
        edge_bank = None
        for e in edges:
            b = getattr(e, "fifo_bank", None)
            if b is not None:
                edge_bank = b
                break
        fifo_bank = bank_for_edges(edges, stride=1, bank=edge_bank)
    except Exception:
        fifo_bank = None

    solver = _gs.GraphSolver(nodes=nodes, edges=edges, sample_rate=float(sample_rate), fifo_bank=fifo_bank)

    node_keys = {node.key for node in nodes}
    if output_node_keys is not None:
        out_keys = [k for k in output_node_keys if k in node_keys]
    else:
        out_keys = [node.key for node in nodes if getattr(node, "layer", "") == "master"]
        if not out_keys:
            out_keys = list(node_keys)

    module_registry = {
        node.key: node.analytic_module
        for node in nodes
        if isinstance(getattr(node, "analytic_module", None), nn.Module)
    }
    return CompiledGraph(
        solver=solver,
        output_node_keys=out_keys,
        module_registry=module_registry,
    )

# ---------------------------------------------------------------------------
# Lazy imports — kept at function scope to avoid hard circular dependencies
# ---------------------------------------------------------------------------

def _import_ad():
    import analytic_driver as _ad
    return _ad


def _import_gs():
    import graph_solver as _gs
    return _gs


def _import_vgn():
    import voice_graph_node as _vgn
    return _vgn


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
        # Pick up the block-processing faculty from the analytic object if it
        # declares one.  Both classmethod and staticmethod forms are accepted.
        _faculty_fn = getattr(type(analytic_obj), "block_faculty", None)
        self.block_faculty = _faculty_fn() if callable(_faculty_fn) else None

    def _param(self, name: str) -> Tensor:
        """Return the parameter tensor, applying ``exp()`` if log-scaled."""
        safe_key = _dot_to_key(name)
        p = self.params[safe_key]
        return torch.exp(p) if self._knob_is_log.get(safe_key, False) else p


class LFOTorchNode(nn.Module):
    """Stateful one-sample LFO source for GraphSolver materialization."""

    def __init__(self, lfo, sample_rate: float) -> None:
        super().__init__()
        self.shape = str(getattr(lfo, "shape", "Sine"))
        self.rate_hz = nn.Parameter(torch.tensor(float(getattr(lfo, "rate_hz", 1.0)), dtype=torch.float64))
        self.depth = nn.Parameter(torch.tensor(float(getattr(lfo, "depth", 1.0)), dtype=torch.float64))
        self.phase_offset = nn.Parameter(torch.tensor(float(getattr(lfo, "phase_offset", 0.0)), dtype=torch.float64))
        self.register_buffer("_sample_rate", torch.tensor(float(sample_rate), dtype=torch.float64))
        self.register_buffer("_sample_idx", torch.tensor(0, dtype=torch.int64))

    def reset(self) -> None:
        self._sample_idx.zero_()

    def _wave(self, sample_offsets: Tensor) -> Tensor:
        t = sample_offsets.to(torch.float64) / self._sample_rate
        ph = 2.0 * math.pi * self.rate_hz * t + self.phase_offset
        cycle = torch.remainder(ph / (2.0 * math.pi), 1.0)
        if self.shape == "Triangle":
            val = 2.0 * torch.abs(2.0 * cycle - 1.0) - 1.0
        elif self.shape == "Sawtooth":
            val = 2.0 * cycle - 1.0
        elif self.shape == "Square":
            val = torch.sign(torch.sin(ph))
        else:
            val = torch.sin(ph)
        return (self.depth * val).to(_CDTYPE)

    def forward_block(self, n_samples: int, *, device: Optional[torch.device] = None) -> Tensor:
        n = max(1, int(n_samples))
        dev = device or self._sample_idx.device
        start = self._sample_idx.to(device=dev)
        offsets = start + torch.arange(n, dtype=torch.int64, device=dev)
        self._sample_idx.add_(n)
        return self._wave(offsets)

    def forward(self, x: Tensor) -> Tensor:  # noqa: ARG002
        return _time_series_like(x, self.forward_block(_block_len(x), device=x.device))


def materialize_lfo(lfo, sr: float) -> Tuple["object", nn.Module]:
    """Return the TensorNode plus backing module for one LFODefinition."""
    _gs = _import_gs()
    module = LFOTorchNode(lfo, sr)
    node = _gs.TensorNode(
        key=str(getattr(lfo, "key", "")),
        layer="lfo",
        transform=module.forward,
        natural_rate_hz=float(getattr(lfo, "rate_hz", 1.0)),
        analytic_module=module,
        archetype_key=_archetype_key_for(lfo),
        block_faculty=_block_faculty_for(lfo) or _gs.BlockFaculty(),
        fire_at_solve_start=True,
        hook_at_solve_start=module.reset,
    )
    return node, module


class ParamTorchNode(nn.Module):
    """Scalar control extractor for ParamNode graph values."""

    def __init__(self, param_node) -> None:
        super().__init__()
        self.extractor = str(getattr(param_node, "extractor", "magnitude"))
        self.register_buffer("default_value", torch.tensor(float(getattr(param_node, "default_value", 0.0)), dtype=torch.float64))
        self.register_buffer("low", torch.tensor(float(getattr(param_node, "low", 0.0)), dtype=torch.float64))
        self.register_buffer("high", torch.tensor(float(getattr(param_node, "high", 1.0)), dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        z = x.to(_CDTYPE)
        if z.numel() == 0:
            return z
        if bool(torch.all(z.abs() == 0.0).item()):
            val = torch.ones_like(z.real) * self.default_value
        elif self.extractor == "real":
            val = z.real.to(torch.float64)
        elif self.extractor == "imag":
            val = z.imag.to(torch.float64)
        elif self.extractor == "phase":
            val = torch.angle(z).to(torch.float64)
        elif self.extractor == "energy":
            val = (z.abs() ** 2).to(torch.float64)
        else:
            val = z.abs().to(torch.float64)
        lo = torch.minimum(self.low, self.high)
        hi = torch.maximum(self.low, self.high)
        return val.clamp(lo, hi).to(_CDTYPE)


class ControlSliderTorchNode(nn.Module):
    """DC control source for one ControlSlider."""

    def __init__(self, slider) -> None:
        super().__init__()
        self.is_log = bool(getattr(slider, "is_log", False))
        self.value = nn.Parameter(torch.tensor(float(getattr(slider, "value", 0.5)), dtype=torch.float64))
        self.register_buffer("low", torch.tensor(float(getattr(slider, "low", 0.0)), dtype=torch.float64))
        self.register_buffer("high", torch.tensor(float(getattr(slider, "high", 1.0)), dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:  # noqa: ARG002
        v = self.value.clamp(0.0, 1.0)
        if self.is_log and float(self.low.item()) > 0.0 and float(self.high.item()) > 0.0:
            out = self.low * torch.pow(self.high / self.low, v)
        else:
            out = self.low + v * (self.high - self.low)
        return _shape_like(x, out)


class LFOChannelTorchNode(nn.Module):
    """Stateful one-sample LFO source for AnalyticModule lfo_channels."""

    def __init__(self, channel: dict, sample_rate: float) -> None:
        super().__init__()
        self.shape = str(channel.get("shape", "Sine"))
        self.rate_hz = nn.Parameter(torch.tensor(float(channel.get("rate_hz", 1.0)), dtype=torch.float64))
        self.amplitude = nn.Parameter(torch.tensor(float(channel.get("amplitude", 1.0)), dtype=torch.float64))
        self.phase_offset = nn.Parameter(torch.tensor(float(channel.get("phase_offset", 0.0)), dtype=torch.float64))
        self.register_buffer("_sample_rate", torch.tensor(float(sample_rate), dtype=torch.float64))
        self.register_buffer("_sample_idx", torch.tensor(0, dtype=torch.int64))

    def reset(self) -> None:
        self._sample_idx.zero_()

    def _wave(self, sample_offsets: Tensor) -> Tensor:
        t = sample_offsets.to(torch.float64) / self._sample_rate
        ph = 2.0 * math.pi * self.rate_hz * t + self.phase_offset
        cycle = torch.remainder(ph / (2.0 * math.pi), 1.0)
        if self.shape == "Triangle":
            val = 2.0 * torch.abs(2.0 * cycle - 1.0) - 1.0
        elif self.shape == "Sawtooth":
            val = 2.0 * cycle - 1.0
        elif self.shape == "Square":
            val = torch.sign(torch.sin(ph))
        else:
            val = torch.sin(ph)
        return (self.amplitude * val).to(_CDTYPE)

    def forward_block(self, n_samples: int, *, device: Optional[torch.device] = None) -> Tensor:
        n = max(1, int(n_samples))
        dev = device or self._sample_idx.device
        start = self._sample_idx.to(device=dev)
        offsets = start + torch.arange(n, dtype=torch.int64, device=dev)
        self._sample_idx.add_(n)
        return self._wave(offsets)

    def forward(self, x: Tensor) -> Tensor:  # noqa: ARG002
        return _time_series_like(x, self.forward_block(_block_len(x), device=x.device))


class PitchQuantizerTorchNode(nn.Module):
    """Stateful one-sample wrapper around QuantizerHandle."""

    def __init__(self, module, tuning, sample_rate: float) -> None:
        super().__init__()
        from analytic_model import make_quantizer_handle

        self.handle = make_quantizer_handle(module, tuning)
        self.dt = 1.0 / max(float(sample_rate), 1.0)

    def reset(self) -> None:
        self.handle.reset()

    def forward(self, x: Tensor) -> Tensor:
        z = x.to(_CDTYPE)
        lanes, original_shape = _flatten_time_lanes(z)
        outs: list[np.ndarray] = []
        import copy as _copy
        for lane_idx, lane in enumerate(lanes):
            handle = self.handle if lane_idx == 0 else _copy.deepcopy(self.handle)
            hz_in = lane.real.detach().cpu().numpy()
            outs.append(handle.process_series(hz_in, hz_in, domain="hz", dt=self.dt))
        real_lanes = torch.as_tensor(np.stack(outs, axis=0), dtype=torch.float64, device=z.device)
        real = _unflatten_time_lanes(real_lanes, original_shape)
        return torch.complex(real, z.imag.to(torch.float64)).to(_CDTYPE)


class InterauralTorchNode(nn.Module):
    """Complex128 one-channel port of the legacy interaural placement math."""

    def __init__(self, module, channel: int) -> None:
        super().__init__()
        suffix = "" if channel == 1 else "_ch2"
        self.azimuth = nn.Parameter(torch.tensor(float(getattr(module, f"iau_azimuth{suffix}", 0.0)), dtype=torch.float64))
        self.elevation = nn.Parameter(torch.tensor(float(getattr(module, f"iau_elevation{suffix}", 0.0)), dtype=torch.float64))
        self.distance = nn.Parameter(torch.tensor(float(getattr(module, f"iau_distance{suffix}", 0.0)), dtype=torch.float64))
        self.width = nn.Parameter(torch.tensor(float(getattr(module, f"iau_width{suffix}", 0.0)), dtype=torch.float64))
        self.channel = int(channel)

    def forward(self, x: Tensor) -> Tensor:
        z = x.to(_CDTYPE)
        dist_gain = 1.0 / (1.0 + self.distance.clamp(0.0, 1.0) * 4.0)
        az = self.azimuth.clamp(-1.0, 1.0)
        el = self.elevation.clamp(-1.0, 1.0)
        half_w = self.width.clamp(0.0, 1.0) * (math.pi / 4.0)
        theta_c = (az + 1.0) * (math.pi / 4.0)
        el_tilt = el * (math.pi / 8.0)
        theta = theta_c - half_w + el_tilt if self.channel == 1 else theta_c + half_w - el_tilt
        return z * dist_gain.to(_CDTYPE) * torch.exp(1j * theta.to(_CDTYPE))


class MixerTorchNode(nn.Module):
    """Torch identity mixer node.

    The graph solver performs edge accumulation before calling this transform,
    so the mixer kernel's job is to keep the accumulated complex payload in
    torch space and preserve scalar or block shape.
    """

    def __init__(self, mixer) -> None:
        super().__init__()
        self.projection_active = bool(getattr(mixer, "projection_active", True))

    def forward(self, x: Tensor) -> Tensor:
        return x.to(_CDTYPE)


class PrecomputedSeriesTorchNode(nn.Module):
    """Torch source backed by a precomputed real-valued control series."""

    def __init__(self, series, *, dtype: torch.dtype = _CDTYPE) -> None:
        super().__init__()
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros((1, 1, 1), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, arr.shape[0], 1)
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.as_tensor(arr, dtype=torch.float64)
        self.register_buffer("series", tensor)
        self.register_buffer("_sample_idx", torch.tensor(0, dtype=torch.int64))
        self.dtype = dtype

    def reset(self) -> None:
        self._sample_idx.zero_()

    def forward_block(self, n_samples: int, *, device: Optional[torch.device] = None) -> Tensor:
        n = max(1, int(n_samples))
        dev = device or self.series.device
        idx0 = int(self._sample_idx.item())
        idx = torch.arange(idx0, idx0 + n, dtype=torch.long, device=dev)
        idx = torch.clamp(idx, 0, int(self.series.shape[1]) - 1)
        self._sample_idx.add_(n)
        real = self.series.to(device=dev)[:, idx, ...]
        return real.to(_CDTYPE if self.dtype.is_complex else self.dtype)

    def forward(self, x: Tensor) -> Tensor:  # noqa: ARG002
        y = self.forward_block(_block_len(x), device=x.device)
        return _time_series_like(x, y)


class GranularVoiceNode(nn.Module):
    """Thin graph wrapper around GranularClusterDriver with one-sample output."""

    def __init__(self, voice, sample_rate: float, duration: float) -> None:
        super().__init__()
        self.voice = voice
        self.sample_rate = float(sample_rate)
        self.duration = float(duration)
        self.register_buffer("_sample_idx", torch.tensor(0, dtype=torch.int64))
        self._buffer: Optional[Tensor] = None

    def reset(self) -> None:
        self._sample_idx.zero_()
        self._buffer = None

    def _render(self) -> Tensor:
        from analytic_synth_legacy import _compute_envelope, _ensure_granular
        from granular_engine import GranularClusterDriver

        spec = _ensure_granular(self.voice)
        if spec is None:
            n = max(1, int(round(self.duration * self.sample_rate)))
            return torch.zeros(n, dtype=_CDTYPE)
        if hasattr(spec, "center_frequency_hz"):
            spec.center_frequency_hz = float(getattr(self.voice, "freq_hz", spec.center_frequency_hz))
        seed = int(getattr(spec, "editor_seed", 0))
        driver = GranularClusterDriver(spec, sr=self.sample_rate, rng_seed=seed)
        raw = np.asarray(driver.synthesize(self.duration), dtype=np.complex128)
        env = _compute_envelope(self.voice, len(raw), self.duration)
        rendered = (raw * env).astype(np.complex128, copy=False)
        pre_delay = max(0, int(round(float(getattr(self.voice, "pre_delay", 0.0)) * self.sample_rate)))
        if pre_delay:
            rendered[:pre_delay] = 0.0
        return torch.as_tensor(rendered, dtype=_CDTYPE)

    def forward(self, x: Tensor) -> Tensor:  # noqa: ARG002
        if self._buffer is None:
            self._buffer = self._render()
        idx = int(self._sample_idx.item())
        n = _block_len(x)
        self._sample_idx.add_(n)
        if idx >= int(self._buffer.numel()):
            if x.dim() > 0:
                return torch.zeros(tuple(x.shape), dtype=_CDTYPE, device=x.device)
            return torch.zeros((), dtype=_CDTYPE, device=self._buffer.device)
        if n == 1:
            return _shape_like(x, self._buffer[idx].to(device=x.device))
        out = torch.zeros(n, dtype=_CDTYPE, device=x.device)
        stop = min(idx + n, int(self._buffer.numel()))
        span = max(0, stop - idx)
        if span:
            out[:span] = self._buffer[idx:stop].to(device=x.device)
        return _time_series_like(x, out)


def _param_target_voice_port(attr: str) -> str:
    attr = str(attr)
    if attr.startswith("chirp."):
        return "chirp_mod_in"
    if attr.startswith("fm."):
        return "fm_in"
    if attr.startswith("am."):
        return "am_in"
    return "env_mod_in"


def _voice_port_node_key(voice_key: str, port: str) -> str:
    suffix = {
        "fm_in": "fm",
        "am_in": "am",
        "env_mod_in": "env_mod",
        "chirp_mod_in": "chirp_mod",
        "pitch_in": "pitch_in",
    }[port]
    return f"{voice_key}_{suffix}"


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
        saturation_policy = str(getattr(e, "saturation", "")),
        saturation_knee   = float(getattr(e, "saturation_knee", 1.0)),
        group             = str(getattr(e, "router_key", "")),
        semantic_role     = str(getattr(e, "edge_kind", "signal")),
        src_port          = src_port,
        dst_port          = str(getattr(e, "dst_port", "")),
    )


def _prune_orphan_tensor_nodes(nodes: list, edges: list) -> "Tuple[List, List]":
    """Drop only edges whose endpoints were not materialized.

    The Phase 3 materializer is the stable graph surface for the full patch,
    including currently-unrouted LFOs, controls, modules, and system ports.
    Those nodes must remain available for later routing edits and recompiles.
    """
    valid_keys = {node.key for node in nodes}
    edges = [
        edge for edge in edges
        if edge.src_key in valid_keys and edge.dst_key in valid_keys
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# materialize_network — full torch module assembly
# ---------------------------------------------------------------------------

def materialize_network(
    patch: "AnalyticPatch",
    sr: float,
    *,
    demo_batch_size: int = 1,
) -> "Tuple[List, List]":
    """Wrap every analytic object in the patch into TensorNodes for GraphSolver."""
    _ad = _import_ad()
    _gs = _import_gs()

    nodes: list = []
    seen_keys: set = set()
    patch.__dict__.pop("_meta_edges", None)

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
            return _shape_like(x, val.to(device=x.device))
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

    routing_edges = list(getattr(_ad._working_routing_graph_for_synthesis(patch), "edges", []))
    voice_runtime_keys: dict[str, list[str]] = {}

    # ── Voices ───────────────────────────────────────────────────────────
    for v in patch.voices:
        if str(getattr(v, "emission_mode", "single")) == "granular":
            voice_runtime_keys[str(v.key)] = [str(v.key)]
            module = GranularVoiceNode(v, sr, float(getattr(patch, "duration", 1.0)))
            _add(_gs.TensorNode(
                key=v.key,
                layer="voice",
                transform=module.forward,
                natural_rate_hz=float(getattr(v, "freq_hz", 0.0)),
                analytic_module=module,
                archetype_key=_archetype_key_for(v),
                block_faculty=_block_faculty_for(v) or _gs.BlockFaculty(),
                fire_at_solve_start=True,
                hook_at_solve_start=module.reset,
            ))
            continue
        _vgn = _import_vgn()
        poly_count = max(1, int(getattr(v, "polyphony_count", 1)))
        runtime_keys = [str(v.key)] if poly_count == 1 else [
            f"{v.key}__poly{i + 1}" for i in range(poly_count)
        ]
        voice_runtime_keys[str(v.key)] = runtime_keys
        _add(_gs.TensorNode(key=v.key, layer="voice", transform=None))

        base_occupied: set[str] = set()
        if getattr(getattr(v, "fm", None), "source_key", ""):
            base_occupied.add("fm_in")
        if getattr(getattr(v, "am", None), "source_key", ""):
            base_occupied.add("am_in")
        original_helper_map = {
            f"{v.key}_fm": "fm_in",
            f"{v.key}_am": "am_in",
            f"{v.key}_env_mod": "env_mod_in",
            f"{v.key}_chirp_mod": "chirp_mod_in",
            f"{v.key}_pitch_in": "pitch_in",
        }
        for edge in routing_edges:
            port = original_helper_map.get(str(getattr(edge, "dst_key", "")))
            if port:
                base_occupied.add(port)
        for pn in getattr(patch, "param_nodes", []):
            for target in getattr(pn, "targets", []):
                if str(target.get("voice_key", "")) == str(v.key):
                    base_occupied.add(_param_target_voice_port(str(target.get("attr", ""))))
        for i, runtime_key in enumerate(runtime_keys):
            rv = copy.copy(v)
            rv.key = runtime_key
            if poly_count > 1:
                rv.phase_origin = float(getattr(v, "phase_origin", 0.0)) + (2.0 * math.pi * i / poly_count)
            vnode = _vgn.MetaVoiceNode.from_voice(
                rv,
                sample_rate=sr,
                duration=float(getattr(patch, "duration", 1.0)),
            )
            voice_nodes, voice_edges = vnode.build_nodes(occupied_inputs=tuple(sorted(base_occupied)))
            for node in voice_nodes:
                _add(node)
            for edge in voice_edges:
                if edge.src_key in seen_keys and edge.dst_key in seen_keys:
                    patch.__dict__.setdefault("_meta_edges", []).append(edge)
            patch.__dict__.setdefault("_meta_edges", []).append(
                _gs.TensorEdge(
                    src_key=f"{runtime_key}_out",
                    dst_key=v.key,
                    weight=complex(1.0, 0.0),
                    semantic_role="voice_poly_sum" if poly_count > 1 else "voice_alias",
                )
            )
    # ── LFOs ─────────────────────────────────────────────────────────────
    for lfo in patch.lfos:
        node, _module = materialize_lfo(lfo, sr)
        _add(node)

    # ── Modules ──────────────────────────────────────────────────────────
    for mod in patch.modules:
        mtype = str(getattr(mod, "module_type", "passthrough"))
        layer = str(getattr(mod, "signal_layer", "signal")) or "signal"
        if mtype == "interaural":
            ch1 = InterauralTorchNode(mod, channel=1)
            ch2 = InterauralTorchNode(mod, channel=2)
            _add(_gs.TensorNode(key=mod.ch1_key(), layer=layer,
                                transform=ch1.forward, analytic_module=ch1,
                                archetype_key=_archetype_key_for(mod),
                                block_faculty=_block_faculty_for(mod) or _gs.BlockFaculty()))
            _add(_gs.TensorNode(key=mod.ch2_key(), layer=layer,
                                transform=ch2.forward, analytic_module=ch2,
                                archetype_key=_archetype_key_for(mod),
                                block_faculty=_block_faculty_for(mod) or _gs.BlockFaculty()))
        elif mtype == "state_machine":
            wrapper = KnobDrivenModule(mod)
            sm_layer = str(getattr(mod, "signal_layer", "performer")) or "performer"
            _add(_gs.TensorNode(key=mod.key, layer=sm_layer,
                                transform=None, analytic_module=wrapper,
                                archetype_key=_archetype_key_for(mod),
                                block_faculty=wrapper.block_faculty or _gs.BlockFaculty()))
            for out_key in mod.sm_out_keys():
                _add(_gs.TensorNode(key=out_key, layer=sm_layer,
                                    transform=None, analytic_module=wrapper,
                                    archetype_key=_archetype_key_for(mod),
                                    block_faculty=wrapper.block_faculty or _gs.BlockFaculty()))
        elif mtype == "lfo":
            if getattr(mod, "lfo_channels", None):
                _add(_gs.TensorNode(key=mod.key, layer=layer, transform=None))
                for i, ch in enumerate(mod.lfo_channels):
                    lfo_mod = LFOChannelTorchNode(dict(ch), sr)
                    ch_key = mod.lfo_ch_key(i)
                    _add(_gs.TensorNode(
                        key=ch_key,
                        layer=layer,
                        transform=lfo_mod.forward,
                        natural_rate_hz=float(ch.get("rate_hz", 1.0)),
                        analytic_module=lfo_mod,
                        archetype_key=_archetype_key_for(mod),
                        block_faculty=_block_faculty_for(mod) or _gs.BlockFaculty(),
                        fire_at_solve_start=True,
                        hook_at_solve_start=lfo_mod.reset,
                    ))
                    patch.__dict__.setdefault("_meta_edges", []).append(
                        _gs.TensorEdge(
                            src_key=ch_key,
                            dst_key=mod.key,
                            weight=complex(1.0, 0.0),
                            semantic_role="module_lfo_channel_sum",
                        )
                    )
            else:
                node, _module = materialize_lfo(mod, sr)
                _add(_gs.TensorNode(
                    key=mod.key,
                    layer=layer,
                    transform=node.transform,
                    natural_rate_hz=node.natural_rate_hz,
                    analytic_module=node.analytic_module,
                    archetype_key=_archetype_key_for(mod),
                    block_faculty=_block_faculty_for(mod) or _gs.BlockFaculty(),
                    fire_at_solve_start=True,
                    hook_at_solve_start=getattr(node.analytic_module, "reset", None),
                ))
        elif mtype == "pitch_quantizer":
            qmod = PitchQuantizerTorchNode(mod, getattr(patch, "tuning", None), sr)
            _add(_gs.TensorNode(
                key=mod.key,
                layer=layer,
                transform=qmod.forward,
                analytic_module=qmod,
                archetype_key=_archetype_key_for(mod),
                block_faculty=_block_faculty_for(mod) or _gs.BlockFaculty(),
                fire_at_solve_start=True,
                hook_at_solve_start=qmod.reset,
            ))
        else:
            _add(_gs.TensorNode(key=mod.key, layer=layer, transform=None))

    # ── Control sliders — DC sources ─────────────────────────────────────
    for surface in getattr(patch, "controls", []):
        for slider in getattr(surface, "sliders", []):
            cmod = ControlSliderTorchNode(slider)
            _add(_gs.TensorNode(
                key=slider.key,
                layer="control",
                transform=cmod.forward,
                analytic_module=cmod,
                archetype_key=_archetype_key_for(slider),
                block_faculty=_block_faculty_for(slider) or _gs.BlockFaculty(),
            ))

    # ── Mixers — torch identity kernels over the accumulated signal ─────
    for mx in patch.mixers:
        mmod = MixerTorchNode(mx)
        _add(_gs.TensorNode(
            key=mx.key,
            layer="master",
            transform=mmod.forward,
            analytic_module=mmod,
            archetype_key=_archetype_key_for(mx),
            block_faculty=_block_faculty_for(mx) or _gs.BlockFaculty(),
        ))

    # ── Param nodes — scalar control extractors ─────────────────────────
    for pn in getattr(patch, "param_nodes", []):
        module = ParamTorchNode(pn)
        _add(_gs.TensorNode(
            key=pn.key,
            layer="param",
            transform=module.forward,
            analytic_module=module,
            archetype_key=_archetype_key_for(pn),
            block_faculty=_block_faculty_for(pn) or _gs.BlockFaculty(),
        ))

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
    for voice in patch.voices:
        fm = getattr(voice, "fm", None)
        runtime_keys = voice_runtime_keys.get(str(voice.key), [str(voice.key)])
        if fm is not None and getattr(fm, "source_key", "") in seen_keys:
            for runtime_key in runtime_keys:
                if f"{runtime_key}_fm" in seen_keys:
                    edges.append(_gs.TensorEdge(
                        src_key=str(fm.source_key),
                        dst_key=f"{runtime_key}_fm",
                        weight=complex(1.0, 0.0),
                        semantic_role="lfo_fm_source",
                    ))
        am = getattr(voice, "am", None)
        if am is not None and getattr(am, "source_key", "") in seen_keys:
            for runtime_key in runtime_keys:
                if f"{runtime_key}_am" in seen_keys:
                    edges.append(_gs.TensorEdge(
                        src_key=str(am.source_key),
                        dst_key=f"{runtime_key}_am",
                        weight=complex(1.0, 0.0),
                        semantic_role="lfo_am_source",
                    ))
    for pn in getattr(patch, "param_nodes", []):
        if pn.key not in seen_keys:
            continue
        for target in getattr(pn, "targets", []):
            voice_key = str(target.get("voice_key", ""))
            attr = str(target.get("attr", ""))
            if not voice_key:
                continue
            port = _param_target_voice_port(attr)
            for runtime_key in voice_runtime_keys.get(voice_key, [voice_key]):
                dst_key = _voice_port_node_key(runtime_key, port)
                if dst_key in seen_keys:
                    edges.append(_gs.TensorEdge(
                        src_key=pn.key,
                        dst_key=dst_key,
                        weight=complex(1.0, 0.0),
                        semantic_role="param_series_injection",
                        dst_port=port,
                    ))
    original_voice_helper_ports: dict[str, tuple[str, str]] = {}
    for voice_key, runtime_keys in voice_runtime_keys.items():
        for port in ("fm_in", "am_in", "env_mod_in", "chirp_mod_in", "pitch_in"):
            original_voice_helper_ports[_voice_port_node_key(voice_key, port)] = (voice_key, port)

    for re in g.edges:
        if re.src_key not in seen_keys:
            continue
        dst_key = str(re.dst_key)
        helper = original_voice_helper_ports.get(dst_key)
        if helper is not None:
            voice_key, port = helper
            for runtime_key in voice_runtime_keys.get(voice_key, [voice_key]):
                runtime_dst = _voice_port_node_key(runtime_key, port)
                if runtime_dst not in seen_keys:
                    continue
                te = routing_edge_to_tensor_edge(re)
                edges.append(replace(te, dst_key=runtime_dst, dst_port=port))
            continue
        if dst_key not in seen_keys:
            continue
        edges.append(routing_edge_to_tensor_edge(re))

    return _prune_orphan_tensor_nodes(nodes, edges)


def compile_network(
    patch: "AnalyticPatch",
    sr: Optional[float] = None,
    *,
    demo_batch_size: int = 1,
    extra_nodes: "Optional[List]" = None,
    extra_edges: "Optional[List]" = None,
) -> CompiledGraph:
    """DEPRECATED: Use compile_nodes with direct node construction instead.

    Materialize *patch* and wrap it in a GraphSolver runtime bundle.

    ``extra_nodes`` and ``extra_edges`` are appended after materialization so
    that pre-built TensorNodes (e.g. ScoreSequencerNode, AudioProjectorNode)
    can participate in the same solver without special-casing in the
    materializer.  Every node is equal; this is the seam that lets the caller
    compose arbitrary nodes alongside patch-derived ones.
    """
    _ad = _import_ad()
    sample_rate = float(sr if sr is not None else getattr(patch, "preview_sr", 48_000.0))
    nodes, edges = materialize_network(
        patch,
        sample_rate,
        demo_batch_size=max(1, int(demo_batch_size)),
    )

    if extra_nodes:
        nodes = list(nodes) + list(extra_nodes)
    if extra_edges:
        edges = list(edges) + list(extra_edges)

    node_keys = {node.key for node in nodes}
    output_keys = [key for key in _ad._system_output_keys(patch) if key in node_keys]
    if not output_keys:
        output_keys = [getattr(mx, "key", "") for mx in getattr(patch, "mixers", []) if getattr(mx, "key", "") in node_keys]

    return compile_nodes(nodes, edges, sample_rate, output_node_keys=output_keys or None)
