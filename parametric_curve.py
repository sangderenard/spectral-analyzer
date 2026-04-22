"""parametric_curve.py
─────────────────────────────────────────────────────────────────────────────
Core data model, evaluation engine, and envelope rule tree.

No UI dependencies.  Import from here; the editor is a thin wrapper.

Structure
─────────
• ControlPoint      — (t, v, theta) knot in [0,1]², Catmull-Rom tension,
                      optional break_after for chain discontinuities.
• TimeMarker        — named vertical divider; regions are the spans between
                      consecutive markers.
• RegionEffect      — per-region modifier on top of the base spline.
• GateEvent         — one note-on / note-off record for time-warp logic.
• TimeWarpCoordinator — maps absolute time → curve t_norm via gate history.
• ParametricCurve   — top-level object with polar Catmull-Rom chains,
                      segment database (t-indexed, no Newton), JSON I/O.

Envelope rule tree
──────────────────
• RuleNode          — one declarative transition rule (trigger × region →
                      blend_mode, crossfade_m).
• EnvelopeRuleTree  — tree of RuleNode objects.  Given a ParametricCurve and
                      a trigger event, it instantiates a Hermite blend spline
                      over [t_event-m, t_event+m] using the analytic polynomial
                      derivatives of the curve — no Newton, no discretization.

Piecewise builder
─────────────────
• PiecewiseEnvelopeBuilder — builds a logic-gated, monotonic piecewise
                      Callable[[float], float] from TimeMarker regions.
                      ``from_curve(curve)`` auto-populates from markers.
                      ``build()`` returns a snapshot callable.

Engine
──────
• ParametricCurveEngine — fully-prepped bundle containing:
      · rule_tree      : EnvelopeRuleTree
      · segment_dict   : {label: Callable} per-region snapshot
      · curve          : base ParametricCurve
      · make_blend()   : compose cached segments with weights/mode/clamp
      · interpret()    : gate-history → bespoke piecewise callable (cached)
      · evaluate()     : cached callable + additive layers at one t_norm
      · _cache         : durable {key: Callable} store, max_cache LRU limit
      Factory: ``ParametricCurveEngine.from_defaults()`` or ``.from_blank()``

Audio helpers
─────────────
• render_envelope_audio(curve, events, freq_hz, gain, dur, tau_up, sr)
      Renders a sustain-warped envelope × oscillator buffer from a gate-event
      log.  Returns preview tensors (or None if deps unavailable).
• render_piecewise_audio(env_fn, chirp_fn, gate_history, amp_curve,
                         chirp_curve, freq_hz, gain, dur, sr, oversample)
      Renders audio at sr×oversample then box-filters back to sr while
      preserving the full complex128 analytic field end-to-end.
      Returns (audio_c128, amp_c128, chirp_c128) tensors.

Usage
─────
    from parametric_curve import (
        ParametricCurve, ControlPoint, TimeMarker, RegionEffect,
        GateEvent, TimeWarpCoordinator,
        RuleNode, EnvelopeRuleTree,
        PiecewiseEnvelopeBuilder,
        ParametricCurveEngine,
        default_envelope, default_chirp, default_blank,
        render_envelope_audio, render_piecewise_audio,
    )

    engine = ParametricCurveEngine.from_defaults(name="voice")
    fn     = engine.interpret(gate_history)      # → Callable[[float], float]
    v      = engine.evaluate(t_norm, key)        # callable + additive layers
    blend  = engine.make_blend(weights={"attack": 0.7, "decay": 0.3})
"""
from __future__ import annotations

import json
import math
import os
import dataclasses as _dc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

try:
    import sounddevice as _sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False


# ─────────────────────────────────────────────────────────────────────────────
# Low-level Catmull-Rom helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SegCoeffs:
    """Hermite polynomial for one Catmull-Rom segment.
    v(u) = c0 + c1*u + c2*u² + c3*u³,  u = (t - t0) / (t1 - t0)
    """
    t0: float
    t1: float
    c0: float
    c1: float
    c2: float
    c3: float


def _build_cr_chain(pts: "List[ControlPoint]") -> "List[_SegCoeffs]":
    """Build Catmull-Rom coefficients for a chain of >= 2 points."""
    n = len(pts)
    if n < 2:
        return []
    segs: List[_SegCoeffs] = []
    for i in range(n - 1):
        t0 = pts[i].t
        t1 = pts[i + 1].t
        if t1 - t0 < 1e-14:
            continue
        pm1 = pts[max(0,     i - 1)]
        p0  = pts[i]
        p1  = pts[i + 1]
        p2  = pts[min(n - 1, i + 2)]
        tens = (p0.tension + p1.tension) * 0.5
        m0 = tens * (p1.v - pm1.v)
        m1 = tens * (p2.v - p0.v)
        d  = p1.v - p0.v
        c0 = p0.v
        c1 = m0
        c2 = 3.0 * d - 2.0 * m0 - m1
        c3 = -2.0 * d + m0 + m1
        segs.append(_SegCoeffs(t0, t1, c0, c1, c2, c3))
    return segs


def _split_into_chains(
    points: "List[ControlPoint]",
) -> "List[List[ControlPoint]]":
    """Split sorted control points at break_after flags."""
    if not points:
        return []
    chains: List[List[ControlPoint]] = []
    cur: List[ControlPoint] = [points[0]]
    for pt in points[1:]:
        cur.append(pt)
        if cur[-2].break_after:
            chains.append(cur[:-1])
            cur = [pt]
    chains.append(cur)
    return [c for c in chains if c]


def _eval_segs_vectorized(
    segs: "List[_SegCoeffs]",
    t_norm: torch.Tensor,
    fill_left: float,
    fill_right: float,
) -> torch.Tensor:
    """Evaluate a list of consecutive segments over flat t_norm [M]."""
    if not segs:
        return torch.full_like(t_norm, fill_left)
    dtype = t_norm.dtype
    t0s = torch.tensor([s.t0 for s in segs], dtype=dtype)
    t1s = torch.tensor([s.t1 for s in segs], dtype=dtype)
    c0s = torch.tensor([s.c0 for s in segs], dtype=dtype)
    c1s = torch.tensor([s.c1 for s in segs], dtype=dtype)
    c2s = torch.tensor([s.c2 for s in segs], dtype=dtype)
    c3s = torch.tensor([s.c3 for s in segs], dtype=dtype)
    idx = torch.searchsorted(t0s.contiguous(), t_norm.contiguous(), right=True) - 1
    idx = idx.clamp(0, len(segs) - 1)
    seg_t0 = t0s[idx]
    seg_t1 = t1s[idx]
    span   = (seg_t1 - seg_t0).clamp(min=1e-14)
    u = ((t_norm - seg_t0) / span).clamp(0.0, 1.0)
    out = c0s[idx] + c1s[idx] * u + c2s[idx] * (u * u) + c3s[idx] * (u * u * u)
    out = torch.where(t_norm < t0s[0],  torch.tensor(fill_left,  dtype=dtype), out)
    out = torch.where(t_norm > t1s[-1], torch.tensor(fill_right, dtype=dtype), out)
    return out


def _eval_all_chains(
    chains: "List[List[_SegCoeffs]]",
    t_norm: torch.Tensor,
    gap_fill: float = 0.0,
) -> torch.Tensor:
    """Evaluate all chains and composite them; gaps between chains → gap_fill."""
    if not chains:
        return torch.full_like(t_norm, gap_fill)
    result = torch.full_like(t_norm, gap_fill)
    for segs in chains:
        if not segs:
            continue
        t_start = segs[0].t0
        t_end   = segs[-1].t1
        mask = (t_norm >= t_start) & (t_norm <= t_end)
        if not mask.any():
            continue
        fill_r = segs[-1].c0 + segs[-1].c1 + segs[-1].c2 + segs[-1].c3
        vals = _eval_segs_vectorized(segs, t_norm, segs[0].c0, fill_r)
        result = torch.where(mask, vals, result)
    return result


def _apply_slew(signal: torch.Tensor, slew_samples: int) -> torch.Tensor:
    """One-pole IIR slew on the last axis.  Works on [N] or [B, N]."""
    if slew_samples <= 0:
        return signal
    alpha = math.exp(-1.0 / max(slew_samples, 1))
    out = signal.clone()
    n = out.shape[-1]
    for i in range(1, n):
        out[..., i] = alpha * out[..., i - 1] + (1.0 - alpha) * out[..., i]
    return out


def _apply_activation(x: torch.Tensor, mode: str, drive: float) -> torch.Tensor:
    """Apply a named nonlinear activation to x ∈ [0,1], returning [0,1].

    Modes: none | tanh | sigmoid | softplus | elu
    drive is a temperature/scale parameter.  mode="none" or drive<=0 → identity.
    """
    if mode == "none" or drive <= 0.0:
        return x
    if mode == "tanh":
        denom = math.tanh(drive)
        if abs(denom) < 1e-30:
            return x
        return torch.tanh(drive * x) / denom
    elif mode == "sigmoid":
        lo   = 1.0 / (1.0 + math.exp( drive * 0.5))
        hi   = 1.0 / (1.0 + math.exp(-drive * 0.5))
        span = hi - lo
        if span < 1e-30:
            return x
        return (torch.sigmoid(drive * (x - 0.5)) - lo) / span
    elif mode == "softplus":
        denom = math.log1p(math.exp(drive))
        if denom < 1e-30:
            return x
        return torch.log1p(torch.exp(drive * x)) / denom
    elif mode == "elu":
        scaled  = drive * (2.0 * x - 1.0)
        elu_out = torch.where(scaled >= 0, scaled, torch.expm1(scaled))
        lo      = math.expm1(-drive)
        hi      = float(drive)
        span    = hi - lo
        if span < 1e-30:
            return x
        return (elu_out - lo) / span
    return x


def _unwrap_angles(angles: "List[float]") -> "List[float]":
    """Unwrap a list of angles (radians) so consecutive differences are in (-π, π]."""
    if not angles:
        return []
    out = [angles[0]]
    for a in angles[1:]:
        diff = a - out[-1]
        diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
        out.append(out[-1] + diff)
    return out


def _build_cr_chain_for_values(
    pts: "List[ControlPoint]",
    values: "List[float]",
) -> "List[_SegCoeffs]":
    """Build a Catmull-Rom coefficient chain using external `values` instead of pts[i].v.

    The segment structure (t0, t1, tension) comes from `pts`; the interpolated
    scalar values come from `values` (must have len == len(pts)).
    """
    n = len(pts)
    if n < 2:
        return []
    segs: List[_SegCoeffs] = []
    for i in range(n - 1):
        t0 = pts[i].t
        t1 = pts[i + 1].t
        if t1 - t0 < 1e-14:
            continue
        vm1 = values[max(0,     i - 1)]
        v0  = values[i]
        v1  = values[i + 1]
        v2  = values[min(n - 1, i + 2)]
        tens = (pts[i].tension + pts[i + 1].tension) * 0.5
        m0 = tens * (v1 - vm1)
        m1 = tens * (v2 - v0)
        d  = v1 - v0
        c0 = v0
        c1 = m0
        c2 = 3.0 * d - 2.0 * m0 - m1
        c3 = -2.0 * d + m0 + m1
        segs.append(_SegCoeffs(t0, t1, c0, c1, c2, c3))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Public data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ControlPoint:
    """A knot in the Catmull-Rom piecewise spline.

    t, v        normalized coordinates in [0, 1].  v is interpreted as the
                polar magnitude r at this knot.
    theta       phase angle (radians) at this knot.  Default 0 = real axis.
    tension     Catmull-Rom tangent scale: 0 = linear, 0.5 = standard, 1 = loose.
    break_after True → chain discontinuity after this point; gap = gap_fill value.
    """
    t: float
    v: float
    theta: float = 0.0
    tension: float = 0.5
    break_after: bool = False

    def to_dict(self) -> dict:
        return {"t": self.t, "v": self.v, "theta": self.theta,
                "tension": self.tension, "break_after": self.break_after}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlPoint":
        return cls(t=float(d["t"]), v=float(d["v"]),
                   theta=float(d.get("theta", 0.0)),
                   tension=float(d.get("tension", 0.5)),
                   break_after=bool(d.get("break_after", False)))


@dataclass
class TimeMarker:
    """A labeled vertical dividing line.

    Markers partition the curve into regions.  When a marker slides, all
    control points inside the two adjacent regions are proportionally rescaled
    to preserve local shape.

    t       normalized x ∈ (0, 1)
    label   arbitrary user string (shown in editor; used as region name)
    pinned  if True the marker cannot be dragged
    """
    t: float
    label: str = ""
    pinned: bool = False

    def to_dict(self) -> dict:
        return {"t": self.t, "label": self.label, "pinned": self.pinned}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeMarker":
        return cls(t=float(d["t"]), label=str(d.get("label", "")),
                   pinned=bool(d.get("pinned", False)))


# Region modes
_REGION_MODES = ("normal", "silence", "hold", "sustain", "loop", "mirror", "additive", "gate")

# Activation function names
_ACTIVATION_MODES = ("none", "tanh", "sigmoid", "softplus", "elu")

# Per-mode background tint in the editor
_REGION_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "normal":   (0.18, 0.22, 0.28, 1.0),
    "silence":  (0.12, 0.12, 0.12, 1.0),
    "hold":     (0.22, 0.18, 0.28, 1.0),
    "loop":     (0.18, 0.28, 0.22, 1.0),
    "mirror":   (0.28, 0.22, 0.18, 1.0),
    "additive": (0.22, 0.28, 0.18, 1.0),
    "gate":     (0.28, 0.18, 0.18, 1.0),
    "sustain":  (0.18, 0.28, 0.38, 1.0),
}


@dataclass
class RegionEffect:
    """Modifier applied in the span between two TimeMarkers (or curve edges).

    Modes
    ─────
    normal    base Catmull-Rom only
    silence   force output to 0
    hold      clamp output to the value at the left boundary
    loop      tile sub_curve repeatedly within the region
    mirror    tile sub_curve with alternating reversal
    additive  add sub_curve onto base
    gate      binarise: v > gate_threshold → 1.0, else 0.0
    sustain   timing annotation for playback warp (no shape change here)
    """
    mode: str = "normal"
    sub_curve: Optional["ParametricCurve"] = None
    lfo_ref: Optional[str] = None
    lfo_depth: float = 0.0
    loop_count: int = 0
    gate_threshold: float = 0.5

    def to_dict(self) -> dict:
        d: dict = {
            "mode":            self.mode,
            "lfo_ref":         self.lfo_ref,
            "lfo_depth":       self.lfo_depth,
            "loop_count":      self.loop_count,
            "gate_threshold":  self.gate_threshold,
        }
        if self.sub_curve is not None:
            d["sub_curve"] = self.sub_curve.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RegionEffect":
        sub: Optional["ParametricCurve"] = None
        if d.get("sub_curve"):
            sub = ParametricCurve.from_dict(d["sub_curve"])
        return cls(
            mode=str(d.get("mode", "normal")),
            sub_curve=sub,
            lfo_ref=d.get("lfo_ref"),
            lfo_depth=float(d.get("lfo_depth", 0.0)),
            loop_count=int(d.get("loop_count", 0)),
            gate_threshold=float(d.get("gate_threshold", 0.5)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gate / time-warp machinery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateEvent:
    """One note-on / note-off record for use with TimeWarpCoordinator.

    t_on      absolute onset time in seconds
    t_off     absolute release time in seconds; None = note still held
    velocity  [0,1] — available to downstream callers, not used by warp logic
    """
    t_on:     float
    t_off:    "float | None" = None
    velocity: float          = 1.0


@dataclass
class TimeWarpCoordinator:
    """Maps absolute-time query points → curve-normalized t_norm ∈ [0,1].

    Given a list of GateEvent records and a vector of absolute-time query
    points, every point is resolved through three independent policy layers:

    retrigger_mode
        "retrigger"  — curve phase resets to 0 at every note-on
        "legato"     — curve continues from wherever it was at previous note-off
        "free"       — gate history is ignored; t_abs / curve_duration → t_norm

    release_mode
        "tail"       — curve keeps advancing at its natural rate after note-off
        "freeze"     — phase locks to the note-off position until next note-on
        "reset"      — phase snaps to 0.0 immediately after note-off

    loop_mode
        "none"       — play once; clamp t_norm at 1.0 if gate exceeds duration
        "loop"       — wrap: t_norm = phase % 1.0
        "ping_pong"  — bounce: t_norm alternates 0→1→0 as phase increases

    curve_duration
        Nominal duration in seconds for one full pass through the curve.
        None → auto-stretch the curve to match each gate's span.
    """
    retrigger_mode: str          = "retrigger"
    release_mode:   str          = "tail"
    loop_mode:      str          = "none"
    curve_duration: "float | None" = None

    def warp(
        self,
        t_abs: torch.Tensor,
        gates: "list[GateEvent]",
    ) -> torch.Tensor:
        """Map absolute times → curve t_norm [0,1] given the gate history."""
        if not gates:
            if self.curve_duration:
                return (t_abs / max(self.curve_duration, 1e-14)).clamp(0.0, 1.0)
            return t_abs.clamp(0.0, 1.0)

        G    = len(gates)
        _INF = float("inf")

        t_on_list  = [g.t_on for g in gates]
        t_off_list = [g.t_off if g.t_off is not None else _INF for g in gates]

        cdur_list = []
        for g in gates:
            if self.curve_duration is not None:
                cdur_list.append(max(self.curve_duration, 1e-14))
            elif g.t_off is not None:
                cdur_list.append(max(g.t_off - g.t_on, 1e-14))
            else:
                cdur_list.append(1.0)

        phase_off_list = [0.0] * G
        if self.retrigger_mode == "legato":
            for i in range(1, G):
                gate_span = max(t_off_list[i - 1] - t_on_list[i - 1], 0.0)
                if t_off_list[i - 1] == _INF:
                    gate_span = 0.0
                phase_off_list[i] = phase_off_list[i - 1] + gate_span / cdur_list[i - 1]
                if self.loop_mode == "none":
                    phase_off_list[i] = min(phase_off_list[i], 1.0)

        t_on_t  = torch.tensor(t_on_list,    dtype=torch.float64)
        t_off_t = torch.tensor(t_off_list,   dtype=torch.float64)
        cdur_t  = torch.tensor(cdur_list,    dtype=torch.float64)
        ph_t    = torch.tensor(phase_off_list, dtype=torch.float64)

        g_idx = (
            torch.searchsorted(t_on_t.contiguous(), t_abs.contiguous(), right=True) - 1
        ).clamp(0, G - 1)

        t_on_m  = t_on_t[g_idx]
        t_off_m = t_off_t[g_idx]
        cdur_m  = cdur_t[g_idx]
        ph_m    = ph_t[g_idx]

        t_in_gate = (t_abs - t_on_m).clamp(min=0.0)
        t_raw = ph_m + t_in_gate / cdur_m

        past_off = t_abs > t_off_m
        if self.release_mode == "freeze":
            t_off_phase = ph_m + (t_off_m - t_on_m).clamp(min=0.0) / cdur_m
            t_raw = torch.where(past_off, t_off_phase, t_raw)
        elif self.release_mode == "reset":
            t_raw = torch.where(past_off, torch.zeros_like(t_raw), t_raw)

        if self.loop_mode == "loop":
            t_norm = t_raw % 1.0
        elif self.loop_mode == "ping_pong":
            phase2 = t_raw % 2.0
            t_norm = torch.where(phase2 <= 1.0, phase2, 2.0 - phase2)
        else:
            t_norm = t_raw.clamp(0.0, 1.0)

        return t_norm

    def warp_with_curve(
        self,
        t_abs:  torch.Tensor,
        gates:  "list[GateEvent]",
        curve:  "ParametricCurve",
        tau_up: float = 0.38,
    ) -> torch.Tensor:
        """Map absolute times → curve t_norm through a continuous sustain warp.

        The sustain transformation stays parametric: it builds a piecewise time
        map from the gate events and the curve's sustain spans, then evaluates
        that map at the requested wall-clock query points.

        For a sustain span [lo, hi] with midpoint mid=(lo+hi)/2:
          • while the note is held and transport enters [lo, mid), phase follows
            the exact asymptote
                dphi/dt = (mid - phi) / ((mid - lo) * curve_duration)
            so transport cleanly approaches mid without any sample stepping
          • when the note is released from inside a sustain span, transport
            exits that span through a finite cubic Hermite bridge to hi, with
            tau_up only shaping the bridge duration
          • outside sustain handling, transport is linear in absolute time

        retrigger_mode, release_mode, and loop_mode are still honoured. If the
        curve has no sustain regions this is identical to calling warp().
        """
        sus_spans = sorted(curve.sustain_regions(), key=lambda span: span[0])
        if not sus_spans:
            return self.warp(t_abs, gates)

        t_flat = t_abs.reshape(-1).to(torch.float64)
        if t_flat.numel() == 0:
            return t_abs.clone()

        if self.retrigger_mode == "free" or not gates:
            dur = max(self.curve_duration or 1.0, 1e-14)
            return (t_flat / dur).clamp(0.0, 1.0).reshape(t_abs.shape)

        dur = max(self.curve_duration or 1.0, 1e-14)
        inv_dur = 1.0 / dur
        eps = 1e-12
        shaped_spans = [(float(lo), 0.5 * (float(lo) + float(hi)), float(hi))
                        for lo, hi in sus_spans]

        t_sorted, order = torch.sort(t_flat)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), dtype=order.dtype, device=order.device)
        raw_phase = torch.zeros_like(t_sorted)

        ev_list: "list[tuple[float, str]]" = []
        for gate in gates:
            ev_list.append((float(gate.t_on), "on"))
            if gate.t_off is not None:
                ev_list.append((float(gate.t_off), "off"))
        ev_list.sort(key=lambda item: item[0])

        def _apply_loop_mode(phase_t: torch.Tensor) -> torch.Tensor:
            if self.loop_mode == "loop":
                return phase_t % 1.0
            if self.loop_mode == "ping_pong":
                phase2 = phase_t % 2.0
                return torch.where(phase2 <= 1.0, phase2, 2.0 - phase2)
            return phase_t.clamp(0.0, 1.0)

        def _nearby_spans(phase: float) -> "list[tuple[float, float, float]]":
            if self.loop_mode == "none":
                return shaped_spans
            cycle = math.floor(phase)
            out: "list[tuple[float, float, float]]" = []
            for k in range(cycle - 1, cycle + 3):
                shift = float(k)
                for lo, mid, hi in shaped_spans:
                    out.append((shift + lo, shift + mid, shift + hi))
            out.sort(key=lambda span: (span[0], span[2]))
            return out

        def _next_span_from(phase: float) -> "tuple[float, float, float] | None":
            for lo, mid, hi in _nearby_spans(phase):
                if hi > phase + eps:
                    return (lo, mid, hi)
            return None

        def _span_containing(phase: float) -> "tuple[float, float, float] | None":
            for lo, mid, hi in _nearby_spans(phase):
                if lo - eps <= phase < hi - eps:
                    return (lo, mid, hi)
            return None

        def _eval_cubic(seg: _SegCoeffs, t_query: torch.Tensor) -> torch.Tensor:
            dt = max(seg.t1 - seg.t0, 1e-14)
            u = ((t_query - seg.t0) / dt).clamp(0.0, 1.0)
            return seg.c0 + u * (seg.c1 + u * (seg.c2 + u * seg.c3))

        def _eval_cubic_scalar(seg: _SegCoeffs, t_query: float) -> float:
            dt = max(seg.t1 - seg.t0, 1e-14)
            u = max(0.0, min(1.0, (t_query - seg.t0) / dt))
            return seg.c0 + u * (seg.c1 + u * (seg.c2 + u * seg.c3))

        q_idx = 0
        q_count = int(t_sorted.numel())

        def _fill_segment(t0: float, t1: float, fn: Any, include_end: bool = False) -> None:
            nonlocal q_idx
            if t1 < t0 - eps:
                return
            lo = max(
                q_idx,
                int(torch.searchsorted(
                    t_sorted,
                    torch.tensor(t0, dtype=torch.float64),
                    right=False,
                ).item()),
            )
            hi = int(torch.searchsorted(
                t_sorted,
                torch.tensor(t1, dtype=torch.float64),
                right=include_end,
            ).item())
            if hi <= lo:
                return
            q_t = t_sorted[lo:hi]
            raw_phase[lo:hi] = fn(q_t)
            q_idx = hi

        def _linear_end(phase: float, interval_end: float) -> float:
            if self.loop_mode == "none" and phase < 1.0 - eps:
                return min(interval_end, cursor + (1.0 - phase) * dur)
            return interval_end

        cursor = min(float(t_sorted[0]), ev_list[0][0] if ev_list else float(t_sorted[0]), 0.0)
        t_limit = float(t_sorted[-1])
        ev_idx = 0
        has_started = False
        note_held = False
        phase = 0.0
        freeze_phase: "float | None" = None
        active_bridge: "_SegCoeffs | None" = None

        while cursor <= t_limit + eps and q_idx < q_count:
            while ev_idx < len(ev_list) and ev_list[ev_idx][0] <= cursor + eps:
                _, kind = ev_list[ev_idx]
                ev_idx += 1

                if kind == "on":
                    if self.retrigger_mode == "retrigger" or not has_started:
                        phase = 0.0
                    note_held = True
                    has_started = True
                    freeze_phase = None
                    active_bridge = None
                    continue

                note_held = False
                if not has_started:
                    continue

                if self.release_mode == "freeze":
                    freeze_phase = phase
                    active_bridge = None
                elif self.release_mode == "reset":
                    phase = 0.0
                    has_started = False
                    freeze_phase = 0.0
                    active_bridge = None
                else:
                    freeze_phase = None
                    span = _span_containing(phase)
                    if span is None:
                        active_bridge = None
                        continue
                    lo, mid, hi = span
                    remaining = hi - phase
                    if remaining <= eps:
                        active_bridge = None
                        continue
                    if phase < mid - eps:
                        half = max(mid - lo, 1e-14)
                        m0 = max(0.0, (mid - phase) / (half * dur))
                    else:
                        second = max(hi - mid, 1e-14)
                        m0 = max(0.0, min(inv_dur, (phase - mid) / (second * dur)))
                    m1 = 0.0 if (self.loop_mode == "none" and hi >= 1.0 - eps) else inv_dur
                    t_natural = remaining * dur
                    slow_factor = max(0.0, 1.0 - min(1.0, m0 * dur))
                    t_bridge = t_natural + min(max(tau_up, 0.0), 2.0 * t_natural) * slow_factor
                    t_bridge = min(max(t_bridge, 1e-9), max(3.0 * t_natural, 1e-9))
                    active_bridge = _hermite_seg_coeffs(
                        cursor,
                        cursor + t_bridge,
                        phase,
                        m0,
                        hi,
                        m1,
                    )

            next_event = ev_list[ev_idx][0] if ev_idx < len(ev_list) else float("inf")
            interval_end = min(next_event, t_limit)
            if interval_end < cursor + eps:
                if next_event < float("inf"):
                    cursor = next_event
                    continue
                break

            while cursor < interval_end - eps and q_idx < q_count:
                include_end = interval_end >= t_limit - eps

                if freeze_phase is not None and not note_held:
                    seg_end = interval_end
                    freeze_value = float(freeze_phase)
                    _fill_segment(cursor, seg_end,
                                  lambda q_t, val=freeze_value: torch.full_like(q_t, val),
                                  include_end=include_end)
                    phase = freeze_value
                    cursor = seg_end
                    continue

                if not has_started:
                    seg_end = interval_end
                    _fill_segment(cursor, seg_end,
                                  lambda q_t: torch.zeros_like(q_t),
                                  include_end=include_end)
                    phase = 0.0
                    cursor = seg_end
                    continue

                if active_bridge is not None and not note_held and cursor < active_bridge.t1 - eps:
                    seg_end = min(interval_end, active_bridge.t1)
                    seg = active_bridge
                    _fill_segment(cursor, seg_end,
                                  lambda q_t, seg=seg: _eval_cubic(seg, q_t),
                                  include_end=seg_end >= t_limit - eps)
                    phase = _eval_cubic_scalar(seg, seg_end)
                    cursor = seg_end
                    if cursor >= seg.t1 - eps:
                        phase = seg.c0 + seg.c1 + seg.c2 + seg.c3
                        active_bridge = None
                    continue

                if note_held:
                    next_span = _next_span_from(phase)
                    if next_span is not None and phase < next_span[0] - eps:
                        seg_end = min(interval_end, _linear_end(phase, cursor + (next_span[0] - phase) * dur))
                        phase0 = phase
                        t0 = cursor
                        _fill_segment(cursor, seg_end,
                                      lambda q_t, phase0=phase0, t0=t0: phase0 + (q_t - t0) * inv_dur,
                                      include_end=seg_end >= t_limit - eps)
                        phase = min(1.0, phase0 + (seg_end - t0) * inv_dur) if self.loop_mode == "none" else phase0 + (seg_end - t0) * inv_dur
                        cursor = seg_end
                        continue
                    if next_span is not None and phase < next_span[1] - eps:
                        lo, mid, _ = next_span
                        hold_tau = max((mid - lo) * dur, 1e-14)
                        phase0 = phase
                        t0 = cursor
                        seg_end = interval_end
                        _fill_segment(
                            cursor,
                            seg_end,
                            lambda q_t, phase0=phase0, t0=t0, mid=mid, hold_tau=hold_tau:
                                mid - (mid - phase0) * torch.exp(-(q_t - t0) / hold_tau),
                            include_end=include_end,
                        )
                        phase = mid - (mid - phase0) * math.exp(-(seg_end - t0) / hold_tau)
                        cursor = seg_end
                        continue

                seg_end = _linear_end(phase, interval_end)
                phase0 = phase
                t0 = cursor
                _fill_segment(cursor, seg_end,
                              lambda q_t, phase0=phase0, t0=t0: phase0 + (q_t - t0) * inv_dur,
                              include_end=seg_end >= t_limit - eps)
                phase = min(1.0, phase0 + (seg_end - t0) * inv_dur) if self.loop_mode == "none" else phase0 + (seg_end - t0) * inv_dur
                cursor = seg_end

            if cursor >= next_event - eps and next_event < float("inf"):
                cursor = next_event
                continue
            if interval_end >= t_limit - eps:
                break
            cursor = interval_end

        if q_idx < q_count:
            tail_value = torch.full((q_count - q_idx,), float(phase), dtype=torch.float64)
            raw_phase[q_idx:] = tail_value

        return _apply_loop_mode(raw_phase)[inverse].reshape(t_abs.shape)


# ─────────────────────────────────────────────────────────────────────────────
# NoteStateMachine — per-voice gate tracker for rule-based tick evaluation
# ─────────────────────────────────────────────────────────────────────────────

class NoteStateMachine:
    """Per-voice gate tracker driving rule-based ParametricCurve evaluation.

    Tracks gate state (IDLE / HELD / RELEASED) and resets to IDLE when a
    release completes (t_norm >= 1.0).  One tick → one complex128 evaluation
    with no look-ahead and no pre-discretization.

    The state machine is intentionally NOT an nn.Module — it holds no learnable
    parameters.  It is owned as a plain Python attribute by the oscillator.

    Usage in the per-sample solver loop
    ────────────────────────────────────
        state = NoteStateMachine()
        state.gate_on(t_abs=0.0, velocity=0.9)
        for sample_idx in range(n_samples):
            t_norm = sample_idx / (sr * note_duration)
            z = state.evaluate(curve, torch.tensor(t_norm, dtype=torch.float64))
            # z is complex128 — carry it unbroken into synthesis
    """

    IDLE     = "idle"
    HELD     = "held"
    RELEASED = "released"

    def __init__(self) -> None:
        self._state:           str             = self.IDLE
        self._t_on:            Optional[float] = None
        self._t_off:           Optional[float] = None
        self._velocity:        float           = 1.0
        self._pending_trigger: Optional[str]   = None
        self._blend_entry:     Any             = None  # _SegEntry | None

    # ── gate events ───────────────────────────────────────────────────────────

    def gate_on(self, t_abs: float, velocity: float = 1.0) -> None:
        """Record a note-on.  A retrigger event is queued if already HELD."""
        trigger_retrigger = self._state == self.HELD
        self._t_on        = float(t_abs)
        self._t_off       = None
        self._velocity    = max(0.0, min(1.0, float(velocity)))
        self._state       = self.HELD
        self._blend_entry = None
        if trigger_retrigger:
            self._pending_trigger = "retrigger"
        # gate_on from IDLE / RELEASED does not queue a rule event —
        # the curve simply starts fresh from t_norm = 0.

    def gate_off(self, t_abs: float) -> None:
        """Record a note-off.  Queues a 'release' event if currently HELD."""
        if self._state == self.HELD:
            self._t_off           = float(t_abs)
            self._state           = self.RELEASED
            self._pending_trigger = "release"
            self._blend_entry     = None

    def hard_stop(self) -> None:
        """Immediate silence — queues 'hard_stop' and clears to IDLE."""
        self._pending_trigger = "hard_stop"
        self._state           = self.IDLE
        self._blend_entry     = None

    def reset(self) -> None:
        """Hard-clear all state (voice steal, engine reset)."""
        self._state           = self.IDLE
        self._t_on            = None
        self._t_off           = None
        self._velocity        = 1.0
        self._pending_trigger = None
        self._blend_entry     = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def state(self) -> str:
        return self._state

    @property
    def gate_event(self) -> "Optional[GateEvent]":
        """Current GateEvent for TimeWarpCoordinator, or None if IDLE."""
        if self._t_on is None:
            return None
        return GateEvent(t_on=self._t_on, t_off=self._t_off, velocity=self._velocity)

    # ── per-tick evaluation ───────────────────────────────────────────────────

    def _consume_pending(self, curve: "ParametricCurve", t_norm: float) -> None:
        """Instantiate a rule-tree blend for the pending event at this t_norm."""
        if self._pending_trigger is None:
            return
        trigger               = self._pending_trigger
        self._pending_trigger = None
        region                = curve._region_label_at(t_norm)
        if trigger in ("retrigger", "release", "hard_stop"):
            entries = curve.rule_tree.instantiate(curve, t_norm, trigger, region)
            if entries:
                self._blend_entry = entries[0]

    def evaluate(
        self,
        curve: "ParametricCurve",
        t_norm_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return the complex128 curve value for this single solver tick.

        Steps per tick
        ──────────────
        1. Consume any pending gate event → instantiate blend from the rule tree
           at this exact t_norm; no look-ahead.
        2. If a blend entry is active and t_norm falls within its window,
           evaluate the Hermite/constant-power blend and convert to complex128.
        3. Otherwise evaluate the base curve via evaluate_normalized.
        4. Auto-clear to IDLE when fully released (state == RELEASED and
           t_norm >= 1.0).

        No pre-discretization.  No look-ahead.  One call → one complex128 scalar.

        Parameters
        ----------
        curve:
            The ParametricCurve to evaluate.
        t_norm_tensor:
            float64 scalar tensor in [0, 1] representing the current normalised
            time for this tick.  Shape () expected.

        Returns
        -------
        complex128 scalar tensor, shape ().
        """
        t_val = float(t_norm_tensor.item() if hasattr(t_norm_tensor, "item") else t_norm_tensor)
        t_val = max(0.0, min(1.0, t_val))

        self._consume_pending(curve, t_val)

        if (
            self._blend_entry is not None
            and self._blend_entry.t0 <= t_val <= self._blend_entry.t1
        ):
            z_py = self._blend_entry.z_at_t(t_val)
            z = torch.tensor(z_py, dtype=torch.complex128)
        else:
            if self._blend_entry is not None and t_val > self._blend_entry.t1:
                self._blend_entry = None
            t_tensor = torch.tensor(t_val, dtype=torch.float64)
            z = curve.evaluate_normalized(t_tensor)  # complex128 scalar

        # Auto-clear after full release
        if self._state == self.RELEASED and t_val >= 1.0:
            self.reset()

        return z


# ─────────────────────────────────────────────────────────────────────────────
# Y-axis scale helpers (display ↔ physical value conversion)
# ─────────────────────────────────────────────────────────────────────────────

_TANH_K = 2.5   # sharpness of tanh Y-axis scale


def _y_to_physical(
    v_norm: float, v_lo: float, v_hi: float, scale: str
) -> float:
    """Map normalized [0,1] → physical [v_lo, v_hi] through named scale."""
    if scale == "log" and v_lo > 0.0 and v_hi > 0.0:
        return math.exp(math.log(v_lo) + v_norm * (math.log(v_hi) - math.log(v_lo)))
    if scale == "tanh":
        t     = 2.0 * v_norm - 1.0   # [-1, 1]
        denom = math.tanh(_TANH_K)
        w     = math.tanh(_TANH_K * t) / denom if abs(denom) > 1e-30 else t
        return v_lo + (w + 1.0) * 0.5 * (v_hi - v_lo)
    return v_lo + v_norm * (v_hi - v_lo)


def _y_to_physical_t(
    v_norm: torch.Tensor, v_lo: float, v_hi: float, scale: str
) -> torch.Tensor:
    """Torch-tensor version of _y_to_physical."""
    if scale == "log" and v_lo > 0.0 and v_hi > 0.0:
        return torch.exp(
            math.log(v_lo) + v_norm * (math.log(v_hi) - math.log(v_lo))
        )
    if scale == "tanh":
        t     = 2.0 * v_norm - 1.0
        denom = math.tanh(_TANH_K)
        w     = torch.tanh(_TANH_K * t) / denom if abs(denom) > 1e-30 else t
        return v_lo + (w + 1.0) * 0.5 * (v_hi - v_lo)
    return v_lo + v_norm * (v_hi - v_lo)


def _y_from_physical(
    v_phys: float, v_lo: float, v_hi: float, scale: str
) -> float:
    """Inverse of _y_to_physical — physical → normalized [0,1]."""
    if scale == "log" and v_lo > 0.0 and v_hi > 0.0:
        v_phys = max(v_lo, min(v_hi, v_phys))
        lo, hi = math.log(v_lo), math.log(v_hi)
        return (math.log(v_phys) - lo) / max(hi - lo, 1e-30)
    if scale == "tanh":
        span = v_hi - v_lo
        if abs(span) < 1e-30:
            return 0.5
        w = max(-1.0 + 1e-9, min(1.0 - 1e-9, 2.0 * (v_phys - v_lo) / span - 1.0))
        denom = math.tanh(_TANH_K)
        t = math.atanh(w * denom) / _TANH_K if abs(denom) > 1e-30 else w
        return (t + 1.0) * 0.5
    span = v_hi - v_lo
    return (v_phys - v_lo) / span if abs(span) > 1e-30 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ParametricCurve — the unified object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParametricCurve:
    """Fully abstract 1-D parametric curve with complex polar output.

    z(t) = r(t) · exp(i·θ(t))

    r(t) is the Catmull-Rom spline through control-point v values (magnitude).
    θ(t) is the Catmull-Rom spline through control-point theta values (unwrapped).

    The segment database (build_segment_db) stores per-segment callables
    z_at_t(t_norm) → complex evaluated directly from the cubic polynomials
    in O(1) with no Newton-Raphson.
    """
    points:          List[ControlPoint]      = field(default_factory=list)
    markers:         List[TimeMarker]        = field(default_factory=list)
    regions:         Dict[int, RegionEffect] = field(default_factory=dict)
    v_lo:            float = 0.0
    v_hi:            float = 1.0
    slew_samples:    int   = 0
    activation:        str   = "none"
    activation_drive:  float = 1.0
    name:              str   = "curve"
    y_scale:           str   = "linear"   # "linear" | "log" | "tanh"
    warp_coordinator: "TimeWarpCoordinator | None" = field(
        default=None, repr=False, compare=False)
    _cache_key:    Any = field(default=None, repr=False, compare=False)
    _r_chains:     Any = field(default=None, repr=False, compare=False)
    _theta_chains: Any = field(default=None, repr=False, compare=False)
    _seg_db:       Any = field(default=None, repr=False, compare=False)
    rule_tree:     Any = field(default=None, repr=False, compare=False)

    # ── initialisation ───────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Lazily initialise rule_tree after the full module is loaded."""
        if self.rule_tree is None:
            # EnvelopeRuleTree is defined later in this module; by the time
            # any ParametricCurve *instance* is created the module is fully
            # loaded, so the forward reference is safe here.
            self.rule_tree = EnvelopeRuleTree.default()

    # ── cache management ──────────────────────────────────────────────────────

    def _invalidate(self) -> None:
        self._cache_key    = None
        self._r_chains     = None
        self._theta_chains = None
        self._seg_db       = None

    def _region_label_at(self, t_norm: float) -> str:
        """Return the TimeMarker label of the region that contains t_norm.

        Regions are the spans between consecutive markers.  The label of the
        marker to the *left* of t_norm names the span.  Returns ``"any"`` when
        the curve has no markers so that rule-tree lookups using ``"any"`` still
        match.
        """
        sorted_markers = sorted(self.markers, key=lambda m: m.t)
        if not sorted_markers:
            return "any"
        for i, m in enumerate(sorted_markers):
            if t_norm < m.t:
                # Before the first marker → use first marker's label as prefix
                return sorted_markers[i - 1].label if i > 0 else ("pre_" + sorted_markers[0].label)
        return sorted_markers[-1].label

    def _ensure_baked(self) -> "Tuple[list, list]":
        """Build and cache the polar Catmull-Rom chains.

        Returns (r_chains, theta_chains), each a list of _SegCoeffs lists.
        r_chains encode r(t); theta_chains encode θ(t) (unwrapped).
        """
        key = (
            tuple((p.t, p.v, p.theta, p.tension, p.break_after) for p in self.points),
            tuple((m.t, m.label) for m in self.markers),
            tuple(sorted((k, v.mode) for k, v in self.regions.items())),
        )
        if self._cache_key == key and self._r_chains is not None:
            return self._r_chains, self._theta_chains

        sorted_pts = sorted(self.points, key=lambda p: p.t)
        raw_chains = _split_into_chains(sorted_pts)

        r_chains: list = []
        theta_chains: list = []
        for chain in raw_chains:
            if len(chain) < 2:
                continue
            r_segs     = _build_cr_chain(chain)
            thetas_uw  = _unwrap_angles([p.theta for p in chain])
            theta_segs = _build_cr_chain_for_values(chain, thetas_uw)
            r_chains.append(r_segs)
            theta_chains.append(theta_segs)

        self._cache_key    = key
        self._r_chains     = r_chains
        self._theta_chains = theta_chains
        self._seg_db       = None
        return r_chains, theta_chains

    # ── segment database (t-indexed, no Newton) ───────────────────────────────

    def build_segment_db(self) -> list:
        """Return (and cache) the segment database.

        Each entry is a named tuple  _SegEntry(t0, t1, z_at_t)  where:

            t0, t1      : float   — segment span in global normalized time [0, 1]
            z_at_t(t)   : callable  float → complex
                          Evaluates z = r(t)·exp(i·θ(t)) directly from the cubic
                          polynomial coefficients.  O(1), no Newton.

        Rebuilt only when the control-point geometry changes.
        """
        import cmath as _cmath
        from collections import namedtuple

        if self._seg_db is not None:
            return self._seg_db

        r_chains, theta_chains = self._ensure_baked()
        if not r_chains or not theta_chains:
            self._seg_db = []
            return self._seg_db

        segs_r: list = []
        segs_t: list = []
        for rc, tc in zip(r_chains, theta_chains):
            segs_r.extend(rc)
            segs_t.extend(tc)

        if not segs_r:
            self._seg_db = []
            return self._seg_db

        _SegEntry = namedtuple('_SegEntry', ['t0', 't1', 'z_at_t'])

        def _make_entry(sr, st):
            cr0, cr1, cr2, cr3 = sr.c0, sr.c1, sr.c2, sr.c3
            ct0, ct1, ct2, ct3 = st.c0, st.c1, st.c2, st.c3
            seg_t0 = sr.t0
            seg_t1 = sr.t1
            seg_dt = seg_t1 - seg_t0

            def z_at_t(t_norm: float,
                       _t0=seg_t0, _dt=seg_dt,
                       _cr0=cr0, _cr1=cr1, _cr2=cr2, _cr3=cr3,
                       _ct0=ct0, _ct1=ct1, _ct2=ct2, _ct3=ct3) -> complex:
                """Return z = r·exp(iθ) at global normalized time t_norm ∈ [t0, t1].

                Direct cubic evaluation — O(1).  t_norm outside [t0,t1] is clamped.
                """
                u_l = (t_norm - _t0) / _dt if abs(_dt) > 1e-30 else 0.5
                u_l = max(0.0, min(1.0, u_l))
                r = _cr0 + u_l * (_cr1 + u_l * (_cr2 + u_l * _cr3))
                θ = _ct0 + u_l * (_ct1 + u_l * (_ct2 + u_l * _ct3))
                return r * _cmath.exp(1j * θ)

            return _SegEntry(t0=seg_t0, t1=seg_t1, z_at_t=z_at_t)

        self._seg_db = [_make_entry(segs_r[i], segs_t[i]) for i in range(len(segs_r))]
        return self._seg_db

    # ── analytic polar value + derivative ────────────────────────────────────

    def _eval_polar_with_deriv(
        self, t_norm: float
    ) -> "tuple[float, float, float, float]":
        """Return (r, dr/dt_norm, θ, dθ/dt_norm) at t_norm ∈ [0,1].

        Uses the baked Catmull-Rom polynomial coefficients directly.
        Derivatives are exact analytic values via the chain rule.
        Used by EnvelopeRuleTree to construct Hermite blend splines at
        transition points — no Newton, no discretization.
        """
        r_chains, theta_chains = self._ensure_baked()
        if not r_chains or not theta_chains:
            return (0.0, 0.0, 0.0, 0.0)

        segs_r: list = []
        segs_t: list = []
        for rc, tc in zip(r_chains, theta_chains):
            segs_r.extend(rc)
            segs_t.extend(tc)

        if not segs_r:
            return (0.0, 0.0, 0.0, 0.0)

        # Find the segment containing t_norm (last segment wins for t=t1)
        sr = segs_r[0]
        st = segs_t[0]
        for i in range(len(segs_r)):
            if segs_r[i].t0 <= t_norm <= segs_r[i].t1:
                sr = segs_r[i]
                st = segs_t[i]
                break

        dt = sr.t1 - sr.t0
        if abs(dt) < 1e-30:
            u_l = 0.5
        else:
            u_l = max(0.0, min(1.0, (t_norm - sr.t0) / dt))

        r    = sr.c0 + u_l * (sr.c1 + u_l * (sr.c2 + u_l * sr.c3))
        θ    = st.c0 + u_l * (st.c1 + u_l * (st.c2 + u_l * st.c3))
        drdU = sr.c1 + u_l * (2.0 * sr.c2 + u_l * 3.0 * sr.c3)
        dθdU = st.c1 + u_l * (2.0 * st.c2 + u_l * 3.0 * st.c3)
        # Chain rule: dv/dt_norm = (dv/du_l) / (dt_norm/du_l) = drdU / dt
        dr_dt = drdU / dt if abs(dt) > 1e-30 else 0.0
        dθ_dt = dθdU / dt if abs(dt) > 1e-30 else 0.0

        return (r, dr_dt, θ, dθ_dt)

    # ── sustain region query ──────────────────────────────────────────────────

    def sustain_regions(self) -> "list[tuple[float, float]]":
        """Return (t_lo, t_hi) for every region whose mode is 'sustain'."""
        sorted_m = sorted(self.markers, key=lambda m: m.t)
        bounds   = [0.0] + [m.t for m in sorted_m] + [1.0]
        out = []
        for ri in range(len(bounds) - 1):
            eff = self.regions.get(ri, RegionEffect())
            if eff.mode == "sustain":
                out.append((bounds[ri], bounds[ri + 1]))
        return out

    def _region_label_at(self, t_norm: float) -> str:
        """Return the TimeMarker label of the region containing t_norm.

        The label is the marker at the LEFT boundary of the containing region;
        if the region has no left marker (first region) the label is \"start\".
        Used by EnvelopeRuleTree to select the matching RuleNode.
        """
        sorted_m = sorted(self.markers, key=lambda m: m.t)
        bounds   = [0.0] + [m.t for m in sorted_m] + [1.0]
        labels   = ["start"] + [m.label for m in sorted_m]
        for ri in range(len(bounds) - 1):
            if bounds[ri] <= t_norm <= bounds[ri + 1]:
                return labels[ri]
        return labels[-1]

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate_normalized(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Evaluate the complex polar signal at t_norm ∈ [0,1].

        Returns z(t) = r(t)·exp(i·θ(t)) as complex128.
        Region effects, activation, and slew are applied.
        """
        orig_shape = t_norm.shape
        t64 = t_norm.reshape(-1).to(torch.float64)
        r_chains, theta_chains = self._ensure_baked()

        r     = _eval_all_chains(r_chains, t64, gap_fill=0.0).clamp(0.0, 1.0)
        theta = _eval_all_chains(theta_chains, t64, gap_fill=0.0)

        sorted_markers = sorted(self.markers, key=lambda m: m.t)
        bounds = [0.0] + [m.t for m in sorted_markers] + [1.0]
        for ri in range(len(bounds) - 1):
            t_lo = bounds[ri]
            t_hi = bounds[ri + 1]
            eff  = self.regions.get(ri, RegionEffect())
            if eff.mode == "normal":
                continue
            mask = (t64 >= t_lo) & (t64 <= t_hi)
            if not mask.any():
                continue
            span    = max(t_hi - t_lo, 1e-14)
            t_local = ((t64 - t_lo) / span).clamp(0.0, 1.0)

            if eff.mode == "silence":
                r = torch.where(mask, torch.zeros_like(r), r)
            elif eff.mode == "hold":
                first_idx = mask.nonzero(as_tuple=False)
                hold_v = r[first_idx[0, 0]] if first_idx.numel() > 0 else torch.tensor(0.0, dtype=torch.float64)
                r = torch.where(mask, hold_v.expand_as(r), r)
            elif eff.mode == "gate":
                gated = (r > eff.gate_threshold).to(torch.float64)
                r = torch.where(mask, gated, r)
            elif eff.mode == "loop" and eff.sub_curve is not None:
                reps   = max(eff.loop_count, 1)
                tile_t = (t_local * reps) % 1.0
                sub_v  = eff.sub_curve.evaluate_normalized(tile_t).real
                r = torch.where(mask, sub_v, r)
            elif eff.mode == "mirror" and eff.sub_curve is not None:
                reps    = max(eff.loop_count, 1)
                phase   = t_local * reps
                rep_idx = phase.long()
                tile_t  = phase % 1.0
                tile_t  = torch.where((rep_idx % 2) == 1, 1.0 - tile_t, tile_t)
                sub_v   = eff.sub_curve.evaluate_normalized(tile_t).real
                r       = torch.where(mask, sub_v, r)
            elif eff.mode == "sustain":
                pass  # timing annotation only; shape unchanged
            elif eff.mode == "additive" and eff.sub_curve is not None:
                sub_v = eff.sub_curve.evaluate_normalized(t_local).real
                added = (r + sub_v * eff.lfo_depth).clamp(0.0, 1.0)
                r = torch.where(mask, added, r)

        z = r.to(torch.complex128) * torch.exp(1j * theta.to(torch.complex128))
        if self.activation != "none":
            mag     = z.abs()
            mag_act = _apply_activation(mag, self.activation, self.activation_drive)
            z = z * (mag_act / mag.clamp(min=1e-30))
        if self.slew_samples > 0:
            z = _apply_slew(z, self.slew_samples)
        return z.reshape(orig_shape)

    # ── Y-axis scale conversion ───────────────────────────────────────────────

    def to_physical(
        self, v_norm: "Union[float, torch.Tensor]"
    ) -> "Union[float, torch.Tensor]":
        """Map normalized curve value [0,1] → physical value in [v_lo, v_hi].

        Uses self.y_scale for the mapping ("linear", "log", or "tanh").
        """
        if isinstance(v_norm, torch.Tensor):
            return _y_to_physical_t(v_norm, self.v_lo, self.v_hi, self.y_scale)
        return _y_to_physical(float(v_norm), self.v_lo, self.v_hi, self.y_scale)

    def from_physical(
        self, v_phys: "Union[float, torch.Tensor]"
    ) -> "Union[float, torch.Tensor]":
        """Inverse of to_physical — physical → normalized [0,1]."""
        if isinstance(v_phys, torch.Tensor):
            return torch.tensor(
                [_y_from_physical(float(x), self.v_lo, self.v_hi, self.y_scale)
                 for x in v_phys.reshape(-1)],
                dtype=v_phys.dtype,
            ).reshape(v_phys.shape)
        return _y_from_physical(float(v_phys), self.v_lo, self.v_hi, self.y_scale)

    def bake(self) -> "Tuple[list, list]":
        """Ensure the polar CR chains are built and return (r_chains, theta_chains)."""
        return self._ensure_baked()

    def evaluate_complex(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Full complex polar signal at t_norm ∈ [0,1].  Returns complex128."""
        return self.evaluate_normalized(t_norm)

    def __call__(
        self,
        t: "Union[float, torch.Tensor]",
        gate_history: "list[GateEvent] | None" = None,
        warp: "TimeWarpCoordinator | None"      = None,
        *,
        apply_activation: bool = True,
        apply_slew:       bool = True,
    ) -> torch.Tensor:
        """Universal callable interface.

        Simple form (no gate_history): t is t_norm ∈ [0,1].
        Gated form: t is absolute time in seconds; a TimeWarpCoordinator maps
        t → t_norm based on gate boundaries and retrigger/legato/loop policy.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)

        if gate_history is None:
            return self.evaluate_normalized(t)

        coordinator = warp or self.warp_coordinator or TimeWarpCoordinator()
        t_flat = t.reshape(-1).to(torch.float64)
        t_norm = coordinator.warp(t_flat, gate_history)
        return self.evaluate_normalized(t_norm).reshape(t.shape)

    def evaluate(self, t_ax: torch.Tensor, dur: float) -> torch.Tensor:
        """Evaluate in real-world time.
        t_ax   absolute time in seconds, shape [N] or [B, N]
        dur    total duration in seconds
        Returns same shape/dtype, values in [v_lo, v_hi].
        """
        t_norm = t_ax / max(dur, 1e-14)
        v_norm = self.evaluate_normalized(t_norm)
        return self.v_lo + v_norm * (self.v_hi - self.v_lo)

    # ── point editing helpers ─────────────────────────────────────────────────

    def add_point(self, t: float, v: float, tension: float = 0.5) -> int:
        pt = ControlPoint(t=float(t), v=float(v), tension=float(tension))
        self.points.append(pt)
        self.points.sort(key=lambda p: p.t)
        self._invalidate()
        return next(i for i, p in enumerate(self.points) if p is pt)

    def remove_point(self, index: int) -> None:
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self._invalidate()

    def move_point(self, index: int, t: float, v: float) -> None:
        if 0 <= index < len(self.points):
            self.points[index].t = float(t)
            self.points[index].v = float(v)
            self.points.sort(key=lambda p: p.t)
            self._invalidate()

    def toggle_break(self, index: int) -> None:
        if 0 <= index < len(self.points):
            self.points[index].break_after = not self.points[index].break_after
            self._invalidate()

    # ── marker editing helpers ────────────────────────────────────────────────

    def add_marker(self, t: "Union[float, TimeMarker]", label: str = "") -> int:
        if isinstance(t, TimeMarker):
            label = label or t.label
            t = t.t
        t  = max(0.001, min(0.999, float(t)))
        mk = TimeMarker(t=t, label=label)
        self.markers.append(mk)
        self.markers.sort(key=lambda m: m.t)
        return next(i for i, m in enumerate(self.markers) if m is mk)

    def remove_marker(self, index: int) -> None:
        if 0 <= index < len(self.markers):
            self.markers.pop(index)
            new_regions: Dict[int, RegionEffect] = {}
            for ri, eff in self.regions.items():
                if ri < index:
                    new_regions[ri] = eff
                elif ri > index:
                    new_regions[ri - 1] = eff
            self.regions = new_regions

    def slide_marker(self, index: int, new_t: float) -> None:
        """Slide marker[index] to new_t, proportionally rescaling adjacent points."""
        if index < 0 or index >= len(self.markers):
            return
        if self.markers[index].pinned:
            return
        sorted_idx = sorted(range(len(self.markers)), key=lambda i: self.markers[i].t)
        mk_pos     = sorted_idx.index(index)
        bounds     = [0.0] + [self.markers[i].t for i in sorted_idx] + [1.0]
        old_t      = bounds[mk_pos + 1]
        t_left_lo  = bounds[mk_pos]
        t_right_hi = bounds[mk_pos + 2]
        new_t = max(t_left_lo + 1e-4, min(t_right_hi - 1e-4, float(new_t)))
        for pt in self.points:
            if t_left_lo < pt.t < old_t:
                u = (pt.t - t_left_lo) / max(old_t  - t_left_lo,  1e-14)
                pt.t = t_left_lo + u * (new_t - t_left_lo)
            elif old_t < pt.t < t_right_hi:
                u = (pt.t - old_t) / max(t_right_hi - old_t, 1e-14)
                pt.t = new_t + u * (t_right_hi - new_t)
        self.markers[index].t = new_t
        self.points.sort(key=lambda p: p.t)
        self._invalidate()

    def region_index_at(self, t_norm: float) -> int:
        sorted_t = sorted(m.t for m in self.markers)
        for i, mt in enumerate(sorted_t):
            if t_norm < mt:
                return i
        return len(sorted_t)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":             self.name,
            "v_lo":             self.v_lo,
            "v_hi":             self.v_hi,
            "y_scale":          self.y_scale,
            "slew_samples":     self.slew_samples,
            "activation":       self.activation,
            "activation_drive": self.activation_drive,
            "points":           [p.to_dict() for p in self.points],
            "markers":          [m.to_dict() for m in self.markers],
            "regions":          {str(k): v.to_dict() for k, v in self.regions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParametricCurve":
        pts  = [ControlPoint.from_dict(x) for x in d.get("points",  [])]
        mks  = [TimeMarker.from_dict(x)   for x in d.get("markers", [])]
        regs = {int(k): RegionEffect.from_dict(v)
                for k, v in d.get("regions", {}).items()}
        return cls(
            name=str(d.get("name", "curve")),
            v_lo=float(d.get("v_lo", 0.0)),
            v_hi=float(d.get("v_hi", 1.0)),
            y_scale=str(d.get("y_scale", "linear")),
            slew_samples=int(d.get("slew_samples", 0)),
            activation=str(d.get("activation", "none")),
            activation_drive=float(d.get("activation_drive", 1.0)),
            points=pts, markers=mks, regions=regs,
        )

    def save(self, folder: str) -> str:
        """Write <folder>/<name>.json.  Returns the written path."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ParametricCurve":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def load_library(cls, folder: str) -> "Dict[str, ParametricCurve]":
        """Load all *.json files in folder → {stem: curve}."""
        lib: Dict[str, ParametricCurve] = {}
        if not os.path.isdir(folder):
            return lib
        for fn in os.listdir(folder):
            if fn.endswith(".json"):
                try:
                    c = cls.load(os.path.join(folder, fn))
                    lib[os.path.splitext(fn)[0]] = c
                except Exception:
                    pass
        return lib


# ─────────────────────────────────────────────────────────────────────────────
# Factory defaults
# ─────────────────────────────────────────────────────────────────────────────

def default_envelope(name: str = "default_envelope") -> ParametricCurve:
    """Classic attack-decay-sustain-release shape, fully labeled."""
    c = ParametricCurve(name=name, v_lo=0.0, v_hi=1.0)
    c.add_point(0.00, 0.0)
    c.add_point(0.05, 1.0)
    c.add_point(0.15, 0.7)
    c.add_point(0.85, 0.7)
    c.add_point(1.00, 0.0)
    c.add_marker(0.05,  "attack_end")
    c.add_marker(0.15,  "decay_end")
    c.add_marker(0.85,  "sustain_end")
    return c


def default_chirp(name: str = "default_chirp") -> ParametricCurve:
    """Neutral frequency-deviation curve — flat line at 0.5 = 0 Hz deviation.

    v_lo=-200, v_hi=200 so normalized 0.5 maps to exactly 0 Hz offset.
    Users can reshape this to add pitch sweeps, vibrato anchors, etc.
    """
    c = ParametricCurve(name=name, v_lo=-200.0, v_hi=200.0)
    c.add_point(0.0, 0.5)
    c.add_point(1.0, 0.5)
    return c


def default_blank(name: str = "blank") -> ParametricCurve:
    """Flat mid-value curve — free-form starting point."""
    c = ParametricCurve(name=name, v_lo=0.0, v_hi=1.0)
    c.add_point(0.0, 0.5)
    c.add_point(1.0, 0.5)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Audio render helper (sustain-warp simulation + oscillator synthesis)
# ─────────────────────────────────────────────────────────────────────────────

def _events_to_gates(events: "list[tuple[str, float]]") -> "list[GateEvent]":
    """Convert a flat [("on"|"off", t), ...] event list into GateEvent objects.

    Each contiguous (on, off) pair becomes one GateEvent.  A trailing "on"
    with no matching "off" produces a GateEvent with t_off=None (note held).
    """
    sorted_ev = sorted(events, key=lambda e: e[1])
    gates: "list[GateEvent]" = []
    pending_on: "float | None" = None
    for kind, t in sorted_ev:
        if kind == "on":
            if pending_on is not None:
                gates.append(GateEvent(t_on=pending_on, t_off=None))
            pending_on = t
        else:
            if pending_on is not None:
                gates.append(GateEvent(t_on=pending_on, t_off=t))
                pending_on = None
    if pending_on is not None:
        gates.append(GateEvent(t_on=pending_on, t_off=None))
    return gates


def render_envelope_audio(
    curve: ParametricCurve,
    events: "list[tuple[str, float]]",
    freq_hz: float,
    gain: float,
    dur: float,
    tau_up: float = 0.38,
    sr: int = 44100,
    chirp_curve: "ParametricCurve | None" = None,
) -> "tuple[Any, Any, Any]":
    """Render a sustain-warped envelope × oscillator buffer.

    Parameters
    ----------
    curve       : ParametricCurve — amplitude envelope
    events      : list of ("on"|"off", t_offset_seconds) sorted by t_offset
    freq_hz     : carrier oscillator frequency
    gain        : output amplitude scale
    dur         : envelope duration in seconds
    tau_up      : sustain rate recovery time constant (seconds)
    sr          : sample rate
    chirp_curve : optional ParametricCurve for frequency deviation (Hz)

    Returns
    -------
    (audio, amp_env_norm, chirp_env_norm)
        audio          : float32 ndarray — real part of complex signal
        amp_env_norm   : float64 ndarray — amplitude envelope [0,1]
        chirp_env_norm : float64 ndarray — chirp envelope [0,1] as stored
    Returns (None, None, None) if numpy is unavailable.

    The sustain transport is a continuous query-time warp built from the gate
    events and sustain spans. The curve itself is still evaluated directly in
    t-norm space via evaluate_normalized — no Newton, no hidden grid state.
    """
    if not (_HAS_SD and _HAS_NP):
        return None, None, None

    import numpy as np

    gates = _events_to_gates(events)
    if not gates:
        return None, None, None

    # Compute render span: from 0 to (last note-off or note-on) + one full dur
    last_gate  = gates[-1]
    t_end      = last_gate.t_off if last_gate.t_off is not None else last_gate.t_on
    render_end = t_end + dur
    n_max      = min(int(render_end * sr) + 1, sr * 60)

    t_abs = torch.linspace(0.0, float(n_max - 1) / sr, n_max, dtype=torch.float64)

    coordinator = TimeWarpCoordinator(
        retrigger_mode="retrigger",
        release_mode="tail",
        loop_mode="none",
        curve_duration=dur,
    )
    fracs_t = coordinator.warp_with_curve(t_abs, gates, curve, tau_up=tau_up)

    # Trim to the first sample where t_norm has reached 1.0 (envelope done)
    done = (fracs_t >= 1.0).nonzero(as_tuple=False)
    n_total = int(done[0, 0]) + 1 if done.numel() > 0 else n_max
    fracs_t = fracs_t[:n_total]
    amp_env = curve.evaluate_normalized(fracs_t).real.clamp(0.0, 1.0).to(torch.float64)

    # Chirp: evaluate at the same warped t_norm fracs
    if chirp_curve is not None:
        chirp_raw = chirp_curve.evaluate_normalized(fracs_t).real.clamp(0.0, 1.0).to(torch.float64)
        chirp_hz  = chirp_curve.to_physical(chirp_raw)
    else:
        chirp_raw = torch.zeros(n_total, dtype=torch.float64)
        chirp_hz  = torch.zeros(n_total, dtype=torch.float64)

    f_inst = torch.full((n_total,), float(freq_hz), dtype=torch.float64) + chirp_hz
    phase  = torch.cumsum(2.0 * math.pi * f_inst / float(sr), dim=0)
    sig    = torch.exp(1j * phase.to(torch.complex128))

    out_c  = amp_env.to(torch.complex128) * sig * float(gain)
    audio  = out_c.real.to(torch.float32).numpy().astype(_np.float32)
    amp_np = amp_env.numpy().astype(_np.float64)
    chr_np = chirp_raw.numpy().astype(_np.float64)

    return audio, amp_np, chr_np


# ─────────────────────────────────────────────────────────────────────────────
# Torch-design voice synthesizer (cumsum phase integration)
# ─────────────────────────────────────────────────────────────────────────────

def render_voice_audio(
    amp_curve:   "ParametricCurve",
    chirp_curve: "ParametricCurve",
    freq_hz:     float,
    gain:        float,
    dur:         float,
    sr:          int = 44100,
) -> "tuple[Any, Any, Any]":
    """Render audio from two envelopes using torch cumsum phase integration.

    This is the "new torch design" oscillator — identical to the synthesis
    path in _synthesize_voice (analytic_driver.py) but driven by two
    ParametricCurve objects instead of an AnalyticVoice.

    Parameters
    ----------
    amp_curve   : amplitude envelope (t_norm → [0,1])
    chirp_curve : frequency-deviation envelope; the normalized [0,1] output
                  is scaled by (v_hi − v_lo) and offset by v_lo to obtain
                  Hz deviation from freq_hz.
    freq_hz     : carrier base frequency
    gain        : output amplitude scale
    dur         : duration in seconds
    sr          : sample rate

    Returns
    -------
    (audio, amp_env, chirp_env_norm)
        audio         : float32 ndarray (n,) — real part of complex signal
        amp_env       : float64 ndarray (n,) — amplitude envelope [0,1]
        chirp_env_norm: float64 ndarray (n,) — chirp envelope [0,1] as stored
    All three are as-rendered at sample resolution — no re-computation needed.
    Returns (None, None, None) if numpy is unavailable.
    """
    if not _HAS_NP:
        return None, None, None

    n = max(1, int(sr * dur))
    t_nrm = torch.linspace(0.0, 1.0, n, dtype=torch.float64)

    # Amplitude envelope: real part of complex polar signal, clamped [0,1]
    amp_env: torch.Tensor = amp_curve.evaluate_normalized(t_nrm).real.clamp(0.0, 1.0)

    # Chirp envelope: normalize then map to Hz deviation via the curve's y_scale
    chirp_raw: torch.Tensor = chirp_curve.evaluate_normalized(t_nrm).real.clamp(0.0, 1.0)
    chirp_hz: torch.Tensor = chirp_curve.to_physical(chirp_raw)   # respects y_scale

    # Instantaneous frequency — float64 throughout, no dtype coercion
    f_inst = torch.full((n,), float(freq_hz), dtype=torch.float64) + chirp_hz

    # Torch cumsum phase integration — the "new torch design" oscillator
    phase = torch.cumsum(2.0 * math.pi * f_inst / float(sr), dim=0)
    sig   = torch.exp(1j * phase.to(torch.complex128))             # complex128

    # Apply amplitude envelope and gain; take real part as the audio output
    out_c  = amp_env.to(torch.complex128) * sig * float(gain)
    audio  = out_c.real.to(torch.float32).numpy().astype(_np.float32)
    amp_np = amp_env.numpy().astype(_np.float64)
    chr_np = chirp_raw.numpy().astype(_np.float64)   # normalized [0,1] for display

    return audio, amp_np, chr_np


# ─────────────────────────────────────────────────────────────────────────────
# Hermite blend spline helper
# ─────────────────────────────────────────────────────────────────────────────

def _hermite_seg_coeffs(
    t0: float, t1: float,
    v0: float, dv0: float,
    v1: float, dv1: float,
) -> _SegCoeffs:
    """Fit a cubic Hermite polynomial through the given boundary conditions.

    The polynomial is in local parameter u_l = (t - t0) / (t1 - t0) ∈ [0,1]:
        v(u_l) = c0 + c1·u_l + c2·u_l² + c3·u_l³

    Boundary conditions (tangents scaled from t_norm-space to u_l-space):
        v(0) = v0,  v'(u=0) = dv0 · dt   (c1 = dv0 * dt)
        v(1) = v1,  v'(u=1) = dv1 · dt
    """
    dt = t1 - t0
    m0 = dv0 * dt
    m1 = dv1 * dt
    d  = v1 - v0
    c0 = v0
    c1 = m0
    c2 = 3.0 * d - 2.0 * m0 - m1
    c3 = -2.0 * d + m0 + m1
    return _SegCoeffs(t0=t0, t1=t1, c0=c0, c1=c1, c2=c2, c3=c3)


# ─────────────────────────────────────────────────────────────────────────────
# Envelope Rule Tree
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleNode:
    """One rule in the envelope transition tree.

    trigger     event type the rule responds to:
                  "retrigger"  — new note fires while one is active
                  "release"    — note-off (normal decay, staccato cut)
                  "hard_stop"  — immediate silence (bow lift, valve close)
                  "any"        — matches any trigger

    region      label of the curve region where this rule applies, or "any"
                (matched against TimeMarker labels in the active curve)

    blend_mode  how to connect outgoing z(t−m) → incoming z(t+m):
                  "hermite"        — cubic Hermite in r and θ separately;
                                     boundary conditions from analytic derivatives
                  "hard_cut"       — instantaneous cut; crossfade_m ignored
                  "constant_power" — equal-power crossfade on |z|, θ interpolated

    crossfade_m half-width of the blend region in t_norm units
    priority    higher priority nodes are evaluated first when multiple match
    children    child RuleNode objects (further specializations)
    """
    trigger:     str
    region:      str
    blend_mode:  str                  = "hermite"
    crossfade_m: float                = 0.05
    priority:    int                  = 0
    children:    "List[RuleNode]"     = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trigger":     self.trigger,
            "region":      self.region,
            "blend_mode":  self.blend_mode,
            "crossfade_m": self.crossfade_m,
            "priority":    self.priority,
            "children":    [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuleNode":
        children = [RuleNode.from_dict(c) for c in d.get("children", [])]
        return cls(
            trigger=str(d["trigger"]),
            region=str(d["region"]),
            blend_mode=str(d.get("blend_mode", "hermite")),
            crossfade_m=float(d.get("crossfade_m", 0.05)),
            priority=int(d.get("priority", 0)),
            children=children,
        )


@dataclass
class EnvelopeRuleTree:
    """Declarative rule tree for note transition and affectation logic.

    Transitions
    ───────────
    Rules define how to blend the envelope curve across event boundaries
    (retrigger, release, hard_stop).  Each matching rule specifies a blend
    mode and a crossfade half-width m.  When an event fires at t_event:

        outgoing endpoint:  t_a = t_event − m  → (r₀, ṙ₀, θ₀, θ̇₀)
        incoming endpoint:  t_b = t_event + m  → (r₁, ṙ₁, θ₁, θ̇₁)

    A cubic Hermite is solved for r and θ separately over [t_a, t_b] using
    the analytic polynomial derivatives from the curve — no Newton, no grid.
    The resulting _SegEntry can be spliced into the active playback sequence.

    Affectations
    ────────────
    Dynamics, tremolo, vibrato, and other performance overlays are separate
    ParametricCurve objects loaded from JSON files.  They are applied
    multiplicatively over score-marked regions by the caller.  The rule tree
    does not manage them — it only manages event-triggered transitions.

    Usage
    ─────
        tree = EnvelopeRuleTree()
        tree.add_rule(RuleNode("retrigger", "any",     blend_mode="hermite",        crossfade_m=0.04))
        tree.add_rule(RuleNode("release",   "sustain", blend_mode="hermite",        crossfade_m=0.06))
        tree.add_rule(RuleNode("hard_stop", "any",     blend_mode="hard_cut",       crossfade_m=0.0))
        tree.add_rule(RuleNode("release",   "attack",  blend_mode="constant_power", crossfade_m=0.03,
                               priority=10))  # overrides the generic release rule in attack region

        # Instantiate the blend at a retrigger event at t=0.35:
        entries = tree.instantiate(curve, t_event=0.35, trigger="retrigger", region_label="sustain")
    """
    rules: "List[RuleNode]" = field(default_factory=list)

    def add_rule(self, node: "RuleNode") -> None:
        """Add a rule to the top level of the tree."""
        self.rules.append(node)

    def _collect_matching(
        self,
        node: "RuleNode",
        trigger: str,
        region_label: str,
    ) -> "List[RuleNode]":
        """Recursively collect nodes matching (trigger, region_label)."""
        matches: list = []
        t_match = (node.trigger == "any" or node.trigger == trigger)
        r_match = (node.region  == "any" or node.region  == region_label)
        if t_match and r_match:
            matches.append(node)
        for child in node.children:
            matches.extend(self._collect_matching(child, trigger, region_label))
        return matches

    def matching_rules(
        self,
        trigger: str,
        region_label: str,
    ) -> "List[RuleNode]":
        """Return all rules matching (trigger, region_label), sorted by priority desc."""
        found: list = []
        for root_node in self.rules:
            found.extend(self._collect_matching(root_node, trigger, region_label))
        found.sort(key=lambda n: n.priority, reverse=True)
        return found

    def build_blend_entry(
        self,
        curve: "ParametricCurve",
        t_event: float,
        node: "RuleNode",
    ) -> Any:
        """Construct a blend _SegEntry from Hermite cubics at the transition point.

        Evaluates the curve's analytic polynomial derivatives at the two crossfade
        endpoints, then fits cubic Hermite splines for r and θ separately.

        For "hard_cut":       returns a zero-width cut entry at t_event.
        For "hermite":        cubic Hermite in r and θ with analytic boundary tangents.
        For "constant_power": equal-power crossfade on |z|, θ interpolated linearly.

        The returned _SegEntry has only (t0, t1, z_at_t) — no Newton, no x-lookup.
        """
        import cmath as _cmath
        from collections import namedtuple as _nt

        _SE = _nt('_SegEntry', ['t0', 't1', 'z_at_t'])
        m = max(node.crossfade_m, 0.0)

        if node.blend_mode == "hard_cut" or m < 1e-12:
            # Instantaneous cut: sample incoming side only
            t_snap = max(0.0, min(1.0, t_event))
            r_in, _, θ_in, _ = curve._eval_polar_with_deriv(t_snap)
            _r = r_in; _θ = θ_in

            def _z_cut(t_norm: float, __r=_r, __θ=_θ) -> complex:
                return __r * _cmath.exp(1j * __θ)

            return _SE(t0=t_snap, t1=max(t_snap + 1e-12, t_snap), z_at_t=_z_cut)

        t_a = max(0.0, min(1.0, t_event - m))
        t_b = max(0.0, min(1.0, t_event + m))

        r0, dr0, θ0, dθ0 = curve._eval_polar_with_deriv(t_a)
        r1, dr1, θ1, dθ1 = curve._eval_polar_with_deriv(t_b)

        if node.blend_mode == "constant_power":
            _r0 = r0; _r1 = r1; _θ0 = θ0; _θ1 = θ1
            _ta = t_a; _dt = t_b - t_a

            def _z_cp(t_norm: float,
                      _R0=_r0, _R1=_r1, _T0=_θ0, _T1=_θ1,
                      _t0=_ta, _dtv=_dt) -> complex:
                u = (t_norm - _t0) / _dtv if abs(_dtv) > 1e-30 else 0.5
                u = max(0.0, min(1.0, u))
                angle = u * math.pi * 0.5
                r = _R0 * math.cos(angle) + _R1 * math.sin(angle)
                θ = _T0 + u * (_T1 - _T0)
                return r * _cmath.exp(1j * θ)

            return _SE(t0=t_a, t1=t_b, z_at_t=_z_cp)

        # Default: hermite — solve cubic Hermite for r and θ separately
        sr = _hermite_seg_coeffs(t_a, t_b, r0, dr0, r1, dr1)
        st = _hermite_seg_coeffs(t_a, t_b, θ0, dθ0, θ1, dθ1)

        cr0, cr1, cr2, cr3 = sr.c0, sr.c1, sr.c2, sr.c3
        ct0, ct1, ct2, ct3 = st.c0, st.c1, st.c2, st.c3
        _ta = t_a; _dtv = t_b - t_a

        def _z_hermite(t_norm: float,
                       _t0=_ta, _dt=_dtv,
                       _cr0=cr0, _cr1=cr1, _cr2=cr2, _cr3=cr3,
                       _ct0=ct0, _ct1=ct1, _ct2=ct2, _ct3=ct3) -> complex:
            u_l = (t_norm - _t0) / _dt if abs(_dt) > 1e-30 else 0.5
            u_l = max(0.0, min(1.0, u_l))
            r = _cr0 + u_l * (_cr1 + u_l * (_cr2 + u_l * _cr3))
            θ = _ct0 + u_l * (_ct1 + u_l * (_ct2 + u_l * _ct3))
            return r * _cmath.exp(1j * θ)

        return _SE(t0=t_a, t1=t_b, z_at_t=_z_hermite)

    def instantiate(
        self,
        curve: "ParametricCurve",
        t_event: float,
        trigger: str,
        region_label: str,
    ) -> list:
        """Return blend _SegEntry objects for the highest-priority matching rule.

        If no rule matches, returns an empty list (pass-through, no blend).

        The returned entries span [t_event−m, t_event+m] and can be spliced
        into the caller's active playback segment sequence.

        Children of the matched node are also instantiated in priority order
        and appended, allowing layered overlays at a single transition point.
        """
        rules = self.matching_rules(trigger, region_label)
        if not rules:
            return []

        node = rules[0]  # highest priority
        entries = [self.build_blend_entry(curve, t_event, node)]

        # Instantiate children that also match this trigger/region
        for child in node.children:
            c_match = (
                (child.trigger == "any" or child.trigger == trigger)
                and (child.region == "any" or child.region == region_label)
            )
            if c_match:
                entries.append(self.build_blend_entry(curve, t_event, child))

        return entries

    @classmethod
    def default(cls) -> "EnvelopeRuleTree":
        """Return a sensible default rule tree for interactive envelope editing."""
        tree = cls()
        tree.add_rule(RuleNode("retrigger", "any",     blend_mode="hermite",        crossfade_m=0.04))
        tree.add_rule(RuleNode("release",   "sustain", blend_mode="hermite",        crossfade_m=0.06))
        tree.add_rule(RuleNode("hard_stop", "any",     blend_mode="hard_cut",       crossfade_m=0.0))
        tree.add_rule(RuleNode("release",   "attack",  blend_mode="constant_power", crossfade_m=0.03,
                               priority=10))
        return tree

    def to_dict(self) -> dict:
        return {"rules": [r.to_dict() for r in self.rules]}

    @classmethod
    def from_dict(cls, d: dict) -> "EnvelopeRuleTree":
        rules = [RuleNode.from_dict(r) for r in d.get("rules", [])]
        tree = cls()
        tree.rules = rules
        return tree

    def save(self, folder: str) -> str:
        """Write <folder>/rule_tree.json.  Returns the written path."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "rule_tree.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "EnvelopeRuleTree":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ─────────────────────────────────────────────────────────────────────────────
# PiecewiseEnvelopeBuilder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _EnvPiece:
    """One piece of a piecewise parametric envelope."""
    t_lo:  float
    t_hi:  float
    fn:    Any   # Callable[[float], float]
    gate:  Any   # Optional[Callable[[], bool]] — None = always active
    label: str = ""


class PiecewiseEnvelopeBuilder:
    """Build a piecewise, logic-gated, monotonic parametric callable.

    Each piece covers a sub-interval of t_norm ∈ [0, 1] and may carry an
    optional *gate* predicate.  At evaluation time pieces are scanned in
    ascending t_lo order; the first piece whose interval contains t_norm
    AND whose gate (if any) returns True is used.  If no piece matches, a
    *fallback* callable is consulted (default: 0.0).

    Monotonicity guarantee
    ──────────────────────
    Pieces are kept sorted by t_lo so the lookup always scans forward — the
    resulting callable never backtracks in domain.

    Logic gating
    ────────────
    A gate is a zero-argument callable that returns bool.  Typical use:

        held = lambda: note_is_held          # live flag closure
        builder.add_piece(0.3, 0.7, sustain_fn, gate=held, label="sustain")

    When the gate returns False the piece is invisible to the evaluator
    (skipped), and the next matching piece or the fallback is used instead.

    Factory convenience
    ───────────────────
    Use ``PiecewiseEnvelopeBuilder.from_curve(curve)`` to build a fully
    populated builder directly from a ParametricCurve's TimeMarker regions.

    Usage::

        builder = PiecewiseEnvelopeBuilder()
        builder.add_piece(0.0, 0.1, attack_fn,  label="attack")
        builder.add_piece(0.1, 0.3, decay_fn,   label="decay")
        builder.add_piece(0.3, 0.7, sustain_fn, gate=lambda: note_held,
                          label="sustain")
        builder.add_piece(0.7, 1.0, release_fn, label="release")
        builder.set_fallback(curve.evaluate_scalar)
        env_fn = builder.build()
        value  = env_fn(0.45)
    """

    def __init__(self) -> None:
        self._pieces:   "List[_EnvPiece]" = []
        self._fallback: Any = None          # Optional[Callable[[float], float]]

    # ── builder API ───────────────────────────────────────────────────────────

    def add_piece(
        self,
        t_lo:  float,
        t_hi:  float,
        fn:    Any,   # Callable[[float], float]
        *,
        gate:  Any = None,   # Optional[Callable[[], bool]]
        label: str = "",
    ) -> "PiecewiseEnvelopeBuilder":
        """Add a piece covering t_norm ∈ [t_lo, t_hi].

        *fn(t_norm)* must return a float.
        *gate*, if provided, is called at evaluation time; if it returns
        False the piece is skipped (fallback is tried instead).
        Returns *self* for method chaining.
        """
        if t_hi <= t_lo:
            return self
        self._pieces.append(_EnvPiece(
            t_lo=float(t_lo), t_hi=float(t_hi),
            fn=fn, gate=gate, label=label,
        ))
        return self

    def set_fallback(self, fn: Any) -> "PiecewiseEnvelopeBuilder":
        """Set the fallback callable used when no piece matches t_norm."""
        self._fallback = fn
        return self

    def pieces(self) -> "List[_EnvPiece]":
        """Return a sorted copy of the current piece list."""
        return sorted(self._pieces, key=lambda p: p.t_lo)

    def piece_labels(self) -> "List[str]":
        """Return the label of each piece in t_lo order."""
        return [p.label for p in self.pieces()]

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self) -> Any:  # Callable[[Tensor | float], Tensor[complex128]]
        """Assemble and return the vectorized piecewise callable.

        The returned callable accepts a 1-D (or scalar) torch.Tensor of
        t_norm values and returns a complex128 tensor of the same length,
        preserving the full complex field from ``evaluate_normalized``.

        Piece dispatch is performed with boolean masks — no Python loop over
        samples.  Gate predicates are still zero-arg callables evaluated once
        per call (not per sample).

        Scalar float inputs are accepted and return a scalar complex128 tensor.
        """
        sorted_pieces = sorted(self._pieces, key=lambda p: p.t_lo)
        fallback      = self._fallback
        n_pieces      = len(sorted_pieces)

        def _piecewise(t_norm: "Union[float, torch.Tensor]") -> torch.Tensor:
            if not isinstance(t_norm, torch.Tensor):
                t_norm = torch.tensor([float(t_norm)], dtype=torch.float64)
            scalar_in = (t_norm.ndim == 0)
            t = t_norm.reshape(-1).to(torch.float64)
            n = t.shape[0]

            # Fill with fallback first, then overwrite with matching pieces.
            if fallback is not None:
                fb = fallback(t)
                if isinstance(fb, torch.Tensor):
                    result = fb.to(torch.complex128)
                else:
                    result = torch.full((n,), complex(float(fb)), dtype=torch.complex128)
            else:
                result = torch.zeros(n, dtype=torch.complex128)

            assigned = torch.zeros(n, dtype=torch.bool)
            for p in sorted_pieces:
                if p.gate is not None and not p.gate():
                    continue
                mask = (~assigned) & (t >= p.t_lo) & (t <= p.t_hi)
                if not mask.any():
                    continue
                val = p.fn(t[mask])
                if isinstance(val, torch.Tensor):
                    result[mask] = val.to(torch.complex128)
                else:
                    result[mask] = complex(float(val))
                assigned[mask] = True

            return result.squeeze(0) if scalar_in else result

        _piecewise.__doc__ = (
            f"PiecewiseEnvelopeCallable  "
            f"pieces={n_pieces}  "
            f"labels={[p.label for p in sorted_pieces]}"
        )
        return _piecewise

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_curve(
        cls,
        curve:         "ParametricCurve",
        *,
        blend_history: "Optional[List[dict]]" = None,
        gate_fn:       Any = None,  # Optional[Callable[[], bool]] for sustain-region gating
        suffix_strip:  "Optional[List[str]]" = None,
    ) -> "PiecewiseEnvelopeBuilder":
        """Build a PiecewiseEnvelopeBuilder from a ParametricCurve.

        Regions are derived from the curve's TimeMarker objects (same logic
        as the segment-sequence report).  Each region becomes one piece whose
        ``fn`` evaluates the curve at the given t_norm.
        """
        _SUFFIXES = suffix_strip or ("_end", "_start", "_begin", "_out", "_in")

        def _phase_name(raw: str, fallback: str) -> str:
            label = (raw or "").strip()
            if not label:
                return fallback
            for suf in _SUFFIXES:
                if label.endswith(suf):
                    stem = label[: -len(suf)].strip()
                    return stem if stem else fallback
            return label

        sorted_m = sorted(curve.markers, key=lambda m: m.t)
        bounds   = [0.0] + [m.t for m in sorted_m] + [1.0]
        n_reg    = len(bounds) - 1

        def _curve_eval(t_norm: "Union[float, torch.Tensor]", _c: "ParametricCurve" = curve) -> torch.Tensor:
            if not isinstance(t_norm, torch.Tensor):
                t_norm = torch.tensor([float(t_norm)], dtype=torch.float64)
            return _c.evaluate_normalized(t_norm.to(torch.float64))

        builder = cls()
        builder.set_fallback(_curve_eval)

        for i in range(n_reg):
            t_lo, t_hi = bounds[i], bounds[i + 1]
            fallback_lbl = f"seg_{i + 1}"
            if i < len(sorted_m):
                lbl = _phase_name(sorted_m[i].label, fallback_lbl)
            else:
                if sorted_m:
                    stem = _phase_name(sorted_m[-1].label, "")
                    lbl  = f"{stem}_tail" if stem else fallback_lbl
                else:
                    lbl = fallback_lbl

            def _make_fn(lo: float, hi: float) -> Any:
                def _fn(t_norm: "Union[float, torch.Tensor]", _c=curve) -> torch.Tensor:
                    return _curve_eval(t_norm, _c)
                return _fn

            builder.add_piece(
                t_lo,
                t_hi,
                _make_fn(t_lo, t_hi),
                gate=gate_fn,
                label=lbl,
            )

        for b in (blend_history or []):
            c_lo  = float(b["c_lo"])
            c_hi  = float(b["c_hi"])
            blbl  = f"{b['trigger']}/{b['blend_mode']}"
            builder.add_piece(
                c_lo,
                c_hi,
                lambda t, _c=curve: _curve_eval(t, _c),
                gate=None,
                label=blbl,
            )

        return builder


def normalize_channel_complex_signals(
    signals: "Dict[str, Union[torch.Tensor, List[complex], List[float], tuple]]",
    *,
    time_stretch: bool = False,
) -> "Dict[str, torch.Tensor]":
    """Normalize a set of 1-D channel signals to a shared complex128 length.

    Signals in the same logical channel should share a display timeline.  When
    lengths differ, the default behaviour is zero-padding so every signal ends
    at the same sample.  With ``time_stretch=True`` shorter signals are
    linearly resampled to the channel's maximum length instead.

    All outputs are 1-D ``torch.complex128`` tensors.  No component splitting
    is required for the normalization step.
    """
    prepared: dict[str, torch.Tensor] = {}
    max_len = 0
    for key, sig in signals.items():
        ten = torch.as_tensor(sig, dtype=torch.complex128).reshape(-1)
        prepared[str(key)] = ten
        max_len = max(max_len, int(ten.numel()))

    if max_len <= 0:
        return {key: torch.zeros(0, dtype=torch.complex128) for key in prepared}

    out: dict[str, torch.Tensor] = {}
    for key, ten in prepared.items():
        cur_len = int(ten.numel())
        if cur_len == max_len:
            out[key] = ten.clone()
            continue
        if cur_len <= 0:
            out[key] = torch.zeros(max_len, dtype=torch.complex128)
            continue
        if not time_stretch or cur_len == 1:
            pad = torch.zeros(max_len - cur_len, dtype=torch.complex128)
            out[key] = torch.cat((ten, pad), dim=0)
            continue

        pos = torch.linspace(0.0, float(cur_len - 1), max_len, dtype=torch.float64)
        i0 = torch.floor(pos).to(torch.int64)
        i1 = torch.clamp(i0 + 1, max=cur_len - 1)
        w1 = (pos - i0.to(torch.float64)).to(torch.complex128)
        w0 = (1.0 - (pos - i0.to(torch.float64))).to(torch.complex128)
        out[key] = ten[i0] * w0 + ten[i1] * w1

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ParametricCurveEngine — fully-prepped engine bundle
# ─────────────────────────────────────────────────────────────────────────────

class ParametricCurveEngine:
    """Fully assembled parametric curve engine.

    A ParametricCurveEngine bundles every capability needed to manufacture,
    cache, blend, and interpret parametric envelopes in a single object:

        rule_tree      — EnvelopeRuleTree: declarative transition/blend rules
        segment_dict   — {label: Callable[[float], float]}: per-region callables
        curve          — ParametricCurve: the base monotonic reference curve
        make_blend()   — factory: arbitrarily compose cached segments
        interpret()    — factory: note-event history → bespoke piecewise callable
        _cache         — durable {key: Callable} store, survives across calls
        additive layers— stacked post-process callables for dynamic modification

    One-sample-cycle model
    ──────────────────────
    In a per-sample solve loop calling code can call ``interpret(gates)`` to
    retrieve (or build) a bespoke envelope callable, then evaluate it at the
    current t_norm with ``evaluate(t_norm, key)``.  The additive layer stack
    allows dynamics/vibrato/tremolo to be applied without touching the cached
    callables.  The cached callables remain durably addressable by key for as
    long as the engine lives (or until ``evict`` / ``flush_cache`` is called).

    Complex phase space
    ───────────────────
    The underlying ParametricCurve operates in the complex plane.  All
    callables returned by this engine evaluate the real part of the curve's
    complex trajectory — ensuring monotonic, fluent traversal of whatever
    phase-space geometry the curve encodes.
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        curve:       "ParametricCurve",
        rule_tree:   "Optional[EnvelopeRuleTree]" = None,
        *,
        chirp_curve: "Optional[ParametricCurve]"  = None,
        max_cache:   int = 256,
    ) -> None:
        self.curve        = curve
        self.chirp_curve  = chirp_curve
        self.rule_tree    = rule_tree if rule_tree is not None else EnvelopeRuleTree.default()
        self._max_cache   = max_cache
        self._cache:      "Dict[str, Any]"                  = {}
        self._builders:   "Dict[str, PiecewiseEnvelopeBuilder]" = {}
        self._additive:   "List[Tuple[Any, float]]"         = []
        self._seg_dict:   "Dict[str, Any]"                  = {}
        self._rebuild_segments()
        self._dirty: bool = False

    # ── segment dictionary ────────────────────────────────────────────────────

    def _rebuild_segments(self) -> None:
        """Rebuild _seg_dict from the curve's current markers."""
        builder = PiecewiseEnvelopeBuilder.from_curve(self.curve)
        self._seg_dict = {
            label: piece.fn
            for label, piece in zip(builder.piece_labels(), builder.pieces())
        }

    @property
    def segment_dict(self) -> "Dict[str, Any]":
        """Per-region callables keyed by label — read-only snapshot."""
        self._ensure_fresh()
        return dict(self._seg_dict)

    # ── dirty tracking ────────────────────────────────────────────────────────

    def _ensure_fresh(self) -> None:
        """Rebuild segments and flush cache if the scene has been mutated."""
        if self._dirty:
            self._rebuild_segments()
            self.flush_cache()
            self._dirty = False

    def mark_dirty(self) -> None:
        """Manually mark the engine dirty after direct mutation of ``self.curve``.

        Call this whenever you mutate ``self.curve.points``, ``self.curve.markers``,
        or ``self.curve.regions`` outside of the engine's own accessor methods.
        The next call to any output method will trigger a rebuild.
        """
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """True if the engine has pending scene mutations not yet recomputed."""
        return self._dirty

    # ── scene accessors — control points ─────────────────────────────────────

    @property
    def points(self) -> "List[ControlPoint]":
        """Read-only snapshot of the curve's control points.

        Mutate through ``add_point`` / ``move_point`` / ``remove_point`` so
        the engine can track dirtiness and invalidate caches.
        """
        return list(self.curve.points)

    def add_point(
        self,
        t:           float,
        v:           float,
        theta:       float = 0.0,
        *,
        tension:     float = 0.5,
        break_after: bool  = False,
    ) -> int:
        """Append a ControlPoint, keep list sorted by t.  Returns new index."""
        cp = ControlPoint(t=float(t), v=float(v), theta=float(theta),
                          tension=float(tension), break_after=break_after)
        self.curve.points.append(cp)
        self.curve.points.sort(key=lambda p: p.t)
        self._dirty = True
        return self.curve.points.index(cp)

    def move_point(
        self,
        index: int,
        *,
        t:     "Optional[float]" = None,
        v:     "Optional[float]" = None,
        theta: "Optional[float]" = None,
    ) -> None:
        """Move an existing ControlPoint in-place.  Marks dirty."""
        cp = self.curve.points[index]
        if t     is not None: cp.t     = float(t)
        if v     is not None: cp.v     = float(v)
        if theta is not None: cp.theta = float(theta)
        if t is not None:
            self.curve.points.sort(key=lambda p: p.t)
        self._dirty = True

    def remove_point(self, index: int) -> "ControlPoint":
        """Remove ControlPoint at *index*.  Marks dirty.  Returns removed point."""
        cp = self.curve.points.pop(index)
        self._dirty = True
        return cp

    # ── scene accessors — time markers ────────────────────────────────────────

    @property
    def markers(self) -> "List[TimeMarker]":
        """Read-only snapshot of the curve's time markers."""
        return list(self.curve.markers)

    def add_marker(self, t: float, label: str = "", *, pinned: bool = False) -> int:
        """Append a TimeMarker, keep list sorted by t.  Returns new index."""
        m = TimeMarker(t=float(t), label=label, pinned=pinned)
        self.curve.markers.append(m)
        self.curve.markers.sort(key=lambda x: x.t)
        self._dirty = True
        return self.curve.markers.index(m)

    def remove_marker(self, label_or_index: "Union[str, int]") -> "TimeMarker":
        """Remove a TimeMarker by label string or list index.  Marks dirty."""
        if isinstance(label_or_index, int):
            m = self.curve.markers.pop(label_or_index)
            self._dirty = True
            return m
        for i, m in enumerate(self.curve.markers):
            if m.label == label_or_index:
                self.curve.markers.pop(i)
                self._dirty = True
                return m
        raise KeyError(f"No marker with label {label_or_index!r}")

    def move_marker(self, label_or_index: "Union[str, int]", t: float) -> None:
        """Move an existing TimeMarker to *t*.  Marks dirty."""
        if isinstance(label_or_index, int):
            self.curve.markers[label_or_index].t = float(t)
        else:
            for m in self.curve.markers:
                if m.label == label_or_index:
                    m.t = float(t)
                    break
            else:
                raise KeyError(f"No marker with label {label_or_index!r}")
        self.curve.markers.sort(key=lambda x: x.t)
        self._dirty = True

    # ── scene accessors — region effects ─────────────────────────────────────

    @property
    def region_effects(self) -> "Dict[int, RegionEffect]":
        """Read-only copy of the curve's region-effect dict {region_index: effect}."""
        return dict(self.curve.regions)

    def set_region_effect(self, region_index: int, effect: "RegionEffect") -> None:
        """Assign a RegionEffect to a region by its integer index.  Marks dirty."""
        self.curve.regions[int(region_index)] = effect
        self._dirty = True

    def clear_region_effect(self, region_index: int) -> None:
        """Remove the RegionEffect at *region_index* if present.  Marks dirty."""
        self.curve.regions.pop(int(region_index), None)
        self._dirty = True

    # ── blend factory ─────────────────────────────────────────────────────────

    def make_blend(
        self,
        weights:  "Optional[Dict[str, float]]"  = None,
        mode:     str                            = "mean",
        segments: "Optional[List[str]]"          = None,
        clamp:    "Tuple[float, float]"          = (0.0, 1.0),
    ) -> "Any":  # Callable[[float], float]
        """Return a callable that blends segment callables at any t_norm.

        The callable is a snapshot — it captures the segment functions at
        call time and is unaffected by subsequent ``update_curve`` calls.

        Parameters
        ──────────
        weights  : {segment_label: weight}.  Missing labels default to 1.0.
                   Pass None for uniform weighting across all selected segments.
        mode     : 'mean'           — normalised weighted average
                   'additive'       — weighted sum (no normalisation)
                   'constant_power' — equal-power mix: sqrt(Σ(w·v)²)
        segments : list of segment labels to include.  None → all.
        clamp    : (lo, hi) output clamp.  Pass (-inf, inf) to disable.
        """
        self._ensure_fresh()
        seg_keys = segments if segments is not None else list(self._seg_dict.keys())
        snap     = {k: self._seg_dict[k] for k in seg_keys if k in self._seg_dict}
        if not snap:
            return lambda t: 0.0

        _weights = {k: float((weights or {}).get(k, 1.0)) for k in snap}
        _mode    = mode
        _lo, _hi = float(clamp[0]), float(clamp[1])

        def _blend(t_norm: float) -> float:
            pairs = [(snap[k](t_norm), _weights[k]) for k in snap]
            if _mode == "additive":
                v = sum(w * val for val, w in pairs)
            elif _mode == "constant_power":
                v = math.sqrt(sum((w * val) ** 2 for val, w in pairs))
            else:  # mean
                total_w = sum(w for _, w in pairs) or 1.0
                v = sum(w * val for val, w in pairs) / total_w
            return max(_lo, min(_hi, v))

        _blend.__doc__ = (
            f"BlendCallable  mode={_mode}  segments={list(snap.keys())}"
        )
        return _blend

    # ── note-event interpreter ────────────────────────────────────────────────

    def _gate_key(self, gate_history: "List[GateEvent]") -> str:
        parts = []
        for g in gate_history:
            off = f"{g.t_off:.6f}" if g.t_off is not None else "x"
            parts.append(f"{g.t_on:.6f},{off}")
        return ":".join(parts)

    def _region_at(self, t_norm: float) -> str:
        """Return the segment label for the region containing t_norm."""
        sorted_m = sorted(self.curve.markers, key=lambda m: m.t)
        bounds   = [0.0] + [m.t for m in sorted_m] + [1.0]
        labels   = list(self._seg_dict.keys())
        for i in range(len(bounds) - 1):
            if bounds[i] <= t_norm <= bounds[i + 1]:
                return labels[i] if i < len(labels) else f"seg_{i + 1}"
        return "tail"

    def interpret(
        self,
        gate_history:  "List[GateEvent]",
        blend_history: "Optional[List[dict]]" = None,
        *,
        gate_fn:       "Any"             = None,
        cache_key:     "Optional[str]"   = None,
        force_rebuild: bool              = False,
    ) -> "Any":  # Callable[[float], float]
        """Interpret gate/blend history → bespoke piecewise Callable[[float], float].

        The rule tree is consulted at every note-on (trigger='retrigger') and
        note-off (trigger='release') boundary.  Matching rules inject Hermite
        or constant-power blend segments at the exact transition t_norm so the
        envelope traverses the phase-space curve fluently through every event.

        The resulting callable is cached under ``cache_key`` (auto-derived from
        gate timing when None) and retrievable via ``engine[key]`` or
        ``engine.cached(key)`` at any future point in the solve cycle.

        Parameters
        ──────────
        gate_history  : recorded GateEvent list
        blend_history : optional blend-event dicts from the editor session
        gate_fn       : zero-arg predicate → bool; attached to every region
                        piece for live sustain gating
        cache_key     : explicit key; auto-generated if None
        force_rebuild : bypass cache and rebuild unconditionally
        """
        self._ensure_fresh()
        key = cache_key or self._gate_key(gate_history)
        if not force_rebuild and key in self._cache:
            return self._cache[key]

        builder = PiecewiseEnvelopeBuilder.from_curve(
            self.curve,
            blend_history=blend_history,
            gate_fn=gate_fn,
        )

        # Inject rule-tree blend entries at each gate boundary
        for gate in gate_history:
            events: "List[Tuple[float, str]]" = [(gate.t_on, "retrigger")]
            if gate.t_off is not None:
                events.append((gate.t_off, "release"))
            for t_ev, trigger in events:
                region  = self._region_at(t_ev)
                entries = self.rule_tree.instantiate(self.curve, t_ev, trigger, region)
                for entry in entries:
                    t0, t1   = float(entry.t0), float(entry.t1)
                    fn_snap  = entry.z_at_t
                    def _entry_fn(
                        t: "Union[float, torch.Tensor]",
                        _f=fn_snap,
                    ) -> torch.Tensor:
                        val = _f(t)
                        return (
                            val.to(torch.complex128)
                            if isinstance(val, torch.Tensor)
                            else torch.as_tensor(val, dtype=torch.complex128)
                        )
                    builder.add_piece(
                        t0, t1,
                        _entry_fn,
                        label=f"{trigger}@{t_ev:.3f}",
                    )

        fn = builder.build()
        self._cache[key]    = fn
        self._builders[key] = builder

        # LRU-lite eviction when over budget
        while len(self._cache) > self._max_cache:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            self._builders.pop(oldest, None)

        return fn

    # ── additive effect layers ────────────────────────────────────────────────

    def add_layer(self, fn: "Any", weight: float = 1.0) -> None:
        """Add an additive post-process callable.

        ``fn(t_norm) → float`` is applied on top of any cached callable by
        ``evaluate()``.  Layers do NOT mutate cached callables — they are a
        separate post-process stack that can be cleared independently.
        """
        self._additive.append((fn, float(weight)))

    def clear_layers(self) -> None:
        """Remove all additive layers."""
        self._additive.clear()

    def evaluate(self, t_norm: float, key: str) -> float:
        """Evaluate a cached callable then sum additive layers.

        Returns 0.0 for unknown keys rather than raising.
        """
        fn = self._cache.get(key)
        v  = fn(t_norm) if fn is not None else 0.0
        for layer_fn, w in self._additive:
            v += w * layer_fn(t_norm)
        return v

    # ── cache management ──────────────────────────────────────────────────────

    def cached(self, key: str) -> "Optional[Any]":
        """Return the cached callable for *key*, or None."""
        return self._cache.get(key)

    def cache_keys(self) -> "List[str]":
        """Return all current cache keys in insertion order."""
        return list(self._cache.keys())

    def builder_for(self, key: str) -> "Optional[PiecewiseEnvelopeBuilder]":
        """Return the PiecewiseEnvelopeBuilder that produced *key*, or None."""
        return self._builders.get(key)

    def evict(self, key: str) -> None:
        """Remove one entry from the callable cache."""
        self._cache.pop(key, None)
        self._builders.pop(key, None)

    def flush_cache(self) -> None:
        """Clear the entire callable and builder cache."""
        self._cache.clear()
        self._builders.clear()

    def __getitem__(self, key: str) -> "Any":
        """``engine[key]`` — retrieve a cached callable (raises KeyError)."""
        fn = self._cache.get(key)
        if fn is None:
            raise KeyError(key)
        return fn

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    # ── curve hot-swap ────────────────────────────────────────────────────────

    def update_curve(self, curve: "ParametricCurve") -> None:
        """Replace the base amplitude curve, rebuild segments, flush cache."""
        self.curve = curve
        self._rebuild_segments()
        self.flush_cache()

    def update_chirp(self, curve: "ParametricCurve") -> None:
        """Replace the chirp curve (cache is NOT flushed — chirp is external)."""
        self.chirp_curve = curve

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "curve":     self.curve.to_dict(),
            "chirp":     self.chirp_curve.to_dict() if self.chirp_curve else None,
            "rule_tree": self.rule_tree.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        d:          dict,
        *,
        max_cache:  int = 256,
    ) -> "ParametricCurveEngine":
        curve = ParametricCurve.from_dict(d["curve"])
        chirp = ParametricCurve.from_dict(d["chirp"]) if d.get("chirp") else None
        rt    = EnvelopeRuleTree.from_dict(d["rule_tree"]) if d.get("rule_tree") else None
        return cls(curve, rt, chirp_curve=chirp, max_cache=max_cache)

    # ── factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_defaults(
        cls,
        *,
        name:      str = "engine",
        rule_tree: "Optional[EnvelopeRuleTree]" = None,
        max_cache: int = 256,
    ) -> "ParametricCurveEngine":
        """Engine with built-in default_envelope + default_chirp curves."""
        return cls(
            default_envelope(name),
            rule_tree,
            chirp_curve=default_chirp(f"{name}_chirp"),
            max_cache=max_cache,
        )

    @classmethod
    def from_blank(
        cls,
        *,
        name:      str = "engine",
        rule_tree: "Optional[EnvelopeRuleTree]" = None,
        max_cache: int = 256,
    ) -> "ParametricCurveEngine":
        """Engine with flat (blank) amplitude and chirp curves."""
        return cls(
            default_blank(name),
            rule_tree,
            chirp_curve=default_blank(f"{name}_chirp"),
            max_cache=max_cache,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Piecewise-callable oscillator renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_piecewise_audio(
    env_fn:        "Any",                    # Callable[[float], float] — amplitude
    chirp_fn:      "Any",                    # Callable[[float], float] — chirp (norm [0,1])
    gate_history:  "List[GateEvent]",
    amp_curve:     "ParametricCurve",        # needed for sustain-region warp
    chirp_curve:   "ParametricCurve",        # needed for to_physical() Hz conversion
    freq_hz:       float,
    gain:          float,
    dur:           float,
    sr:            int   = 44100,
    oversample:    int   = 4,
    tau_up:        float = 0.38,
) -> "tuple[Any, Any, Any]":
    """Render audio sample-by-sample using piecewise complex callables.

    The amplitude envelope and chirp envelope are evaluated through the
    ``PiecewiseEnvelopeBuilder`` callables built from the recorded session.
    Both share the same ``TimeWarpCoordinator``-warped t_norm axis so they
    stay synchronized without collapsing the complex field.

    To maximise blending quality the oscillator runs at ``oversample × sr``
    internally and is box-filtered back down to ``sr`` before return.

    Parameters
    ----------
    env_fn       : piecewise amplitude callable, t_norm → [0, 1]
    chirp_fn     : piecewise chirp callable, t_norm → [0, 1]
    gate_history : list of GateEvent recorded during the session
    amp_curve    : ParametricCurve used for sustain-region time-warp
    chirp_curve  : ParametricCurve used only for ``.to_physical()`` Hz scaling
    freq_hz      : carrier base frequency (Hz)
    gain         : output amplitude scale
    dur          : envelope duration in seconds
    sr           : output sample rate
    oversample   : internal oversampling factor (1 = no oversampling)
    tau_up       : sustain-rate recovery time constant (seconds)

    Returns
    -------
    (audio, amp_env_norm, chirp_env_norm)
        audio          : complex128 tensor at ``sr``
        amp_env_norm   : complex128 tensor at ``sr``
        chirp_env_norm : complex128 tensor at ``sr``
    Returns (None, None, None) when gate_history is empty.
    """
    if not gate_history:
        return None, None, None

    oversample = max(1, int(oversample))
    sr_over    = sr * oversample

    # Normalize gate times so the first note-on is at t=0 in the render window.
    # gate_history carries absolute session-clock times; t_abs starts from 0.0,
    # so without this shift the warp coordinator advances play_elapsed for the
    # entire pre-note gap before the first key-press, producing phantom envelope
    # traversal and causing the first-note trim to fire prematurely.
    raw_gates = list(gate_history)
    t_offset  = raw_gates[0].t_on
    gates = [
        GateEvent(
            t_on      = g.t_on  - t_offset,
            t_off     = (g.t_off - t_offset) if g.t_off is not None else None,
            velocity  = g.velocity,
        )
        for g in raw_gates
    ]

    last_gate  = gates[-1]
    t_end      = last_gate.t_off if last_gate.t_off is not None else last_gate.t_on
    render_end = t_end + dur
    n_over     = min(int(render_end * sr_over) + oversample, sr_over * 60)

    t_abs = torch.linspace(0.0, float(n_over - 1) / sr_over, n_over,
                           dtype=torch.float64)

    coordinator = TimeWarpCoordinator(
        retrigger_mode="retrigger",
        release_mode="tail",
        loop_mode="none",
        curve_duration=dur,
    )
    fracs_t = coordinator.warp_with_curve(t_abs, gates,
                                          amp_curve,
                                          tau_up=tau_up)

    # Use the full render window — do NOT trim at the first fracs_t >= 1.0.
    # The old trim cut the render at first_note_on + dur, silently dropping
    # every subsequent note in a multi-note session.  The render window
    # (n_over) already ends at last_gate.t_off + dur, which is the correct
    # stopping point.
    n_total = n_over

    # Evaluate piecewise callables on the full oversampled t-axis as a tensor
    # broadcast.  Both callables return the native complex128 field from
    # evaluate_normalized — no scalar projection.
    amp_over  = env_fn(fracs_t)    # complex128, shape (n_total,)
    chrp_over = chirp_fn(fracs_t)  # complex128, shape (n_total,)

    # Convert complex chirp field → complex Hz via curve's physical mapping.
    # The imaginary component of the chirp encodes frequency-domain phase deviation;
    # it propagates through to_physical unchanged (linear/log/tanh on complex tensor).
    chirp_hz = chirp_curve.to_physical(chrp_over)

    # Build complex instantaneous frequency and integrate complex phase.
    # Imaginary part of f_inst acts as per-sample damping/growth (exp(-Im*t) envelope).
    f_inst = freq_hz + chirp_hz                                         # complex128
    phase  = torch.cumsum(2.0 * math.pi * f_inst / float(sr_over), dim=0)  # complex128
    sig    = torch.exp(1j * phase)                                      # complex128 oscillator
    out_c  = amp_over * sig * float(gain)                               # complex128 output

    # Downsample from sr_over back to sr using reshape+mean (box filter).
    # out_c, amp_over, chrp_over remain complex128 through the downsample.
    if oversample > 1:
        trim      = (n_total // oversample) * oversample
        audio     = out_c[:trim].reshape(-1, oversample).mean(dim=1)
        amp_down  = amp_over[:trim].reshape(-1, oversample).mean(dim=1)   # complex128
        chrp_down = chrp_over[:trim].reshape(-1, oversample).mean(dim=1)  # complex128
        osc_down  = sig[:trim].reshape(-1, oversample).mean(dim=1)        # complex128 raw oscillator
    else:
        audio     = out_c
        amp_down  = amp_over   # complex128
        chrp_down = chrp_over  # complex128
        osc_down  = sig         # complex128 raw oscillator (pre-amplitude)

    return audio, amp_down, chrp_down, osc_down
