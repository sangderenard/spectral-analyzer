"""voice_graph_node.py — Differentiable voice + mixer nodes for graph_solver.

Solver semantics — READ THIS FIRST
------------------------------------
This is an **instantaneous-sample solver**, not a windowed or block processor.
Every node transform is called once per solver tick and operates on a single
sample moment.  There are no time-axis loops inside any transform.

    Tensor dimensions in this file
    --------------------------------
    B   — batch: simultaneous *instances* of the same node being solved in
            parallel (e.g. B voices of the same type triggered at once).
            NEVER a time window.  B=1 is the common single-instance case.
    C   — channel: node-internal routing dimension only (e.g. harmonic
            partials inside VoiceTorchOscillator).  Never crosses node
            boundaries as a public edge dimension.

    An edge tensor shape is ``()`` (scalar) or ``(B,)``.
    If you see a loop over samples anywhere in a transform, it is a bug.

Architecture
------------
Each ``MetaVoiceNode`` is a compound ``nn.Module`` that owns a
``VoiceTorchOscillator`` and exposes a ``knobs()`` classmethod that mirrors
``AnalyticVoice.knobs()`` exactly.  The oscillator is a full differentiable
re-implementation of ``_synthesize_voice`` in torch.complex128 so that every
voice parameter is reachable by autograd.

Envelope and chirp use piecewise **Catmull-Rom splines** over learnable
control-point values (``nn.Parameter``) with fixed (buffered) normalised
time positions.  When a live driver signal arrives at the
``{key}_env_mod`` or ``{key}_chirp_mod`` port, it is **added** to the
spline output at that tick: the spline provides the base curve and the
driver offsets it.  No signal → bare spline output.  The envelope
result is clamped ≥ 0 after the sum.  Each chirp section carries a
string label that identifies its semantic context (e.g. “onset”,
“sustain”, “release”) — the labels are metadata only, never learned.

Compound node / subnetwork pattern
-----------------------------------
A ``MetaVoiceNode.build_nodes()`` call returns the ``TensorNode`` /
``TensorEdge`` entries that expose this voice’s ports to a ``GraphSolver``.
Internal ports:

    {key}_out        — primary signal output (oscillator transform)
    {key}_fm         — FM accumulator (weighted edges from other nodes)
    {key}_am         — AM accumulator (weighted edges from other nodes)
    {key}_env_mod    — instantaneous driver signal carried by the envelope
                       spline this tick
    {key}_chirp_mod  — instantaneous driver signal carried by the chirp
                       spline this tick

FM and AM nodes carry a one-sample delay edge into the output node by
default, matching the causal constraint that modulator output is available
before the carrier computes.  Set ``causal_mod_delay=False`` to remove the
delay (useful when the solver is Picard-iterating the full SCC).

Knob integration
----------------
``VoiceTorchOscillator.knobs()`` returns the *exact same* ``KnobSpec`` list
as ``AnalyticVoice.knobs()``.  ``MetaVoiceNode.knobs()`` delegates to it
unchanged.  Parameter names map 1-to-1:

    KnobSpec.name           → nn.Parameter attribute name on the oscillator
    adsr.attack / adsr.decay / …  → flat attrs: adsr_attack, adsr_decay, …
    env_knots               → EnvelopeSpline.knot_values (Parameter)
    chirp.f_delta_start/end → ChirpSpline.knot_values[0] / knot_values[-1]
    harmonic_brightness     → harmonic_brightness (Parameter, unconstrained)
    harmonic_warp_strength  → harmonic_warp_strength (Parameter)
    fm.depth_hz             → fm_depth_hz (Parameter)
    am.depth_amp            → am_depth_amp (Parameter)

VoiceTrainer
------------
Gradient accumulation across ``accum_n`` *individual solver ticks* before
each ``optimizer.step()``.  Each tick is one sample moment; the trainer
calls ``MetaVoiceNode.set_sample(idx)`` to advance absolute time before
each tick.  There is no batched-window synthesis path in the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from graph_solver import (
    ChannelSpec,
    EncoderNodeSpec,
    GraphSolver,
    LevelGrid,
    LossNodeSpec,
    MixerArchetype,
    MixerNodeSpec,
    NodeArchetype,
    NodeLayerPresence,
    SemanticPortContract,
    StepNodeSpec,
    TensorEdge,
    TensorNode,
    _CDTYPE,
)

from parametric_curve import (
    ComplexSignalAggregator,
    ParametricCurve,
    NoteStateMachine,
    default_envelope,
    default_chirp,
)

try:
    from signal_generator_v2 import KnobSpec
except Exception:
    from dataclasses import dataclass as _kdc, field as _kfield

    @_kdc
    class KnobSpec:  # type: ignore[no-redef]
        name: str = ""
        label: str = ""
        dtype: str = "float"
        default: object = None
        low: float = 0.0
        high: float = 1.0
        step: float = 0.0
        unit: str = ""
        choices: list = _kfield(default_factory=list)
        is_log: bool = False
        group: str = ""
        fmt: str = ".3g"
        source_class: str = ""
        rebuild_layout: bool = False
        visible_when: object = None


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable ADSR envelope (no spline — inline, timing params have grads)
# ──────────────────────────────────────────────────────────────────────────────

def adsr_envelope_diff(
    t_norm: Tensor,   # scalar () or (B,) — current normalised time, one moment
    attack: Tensor,   # scalar float64 Parameter
    decay: Tensor,
    sustain: Tensor,
    release: Tensor,
    peak: Tensor,
) -> Tensor:
    """Differentiable piecewise-linear ADSR evaluated at one sample moment.

    ``t_norm`` is the **current normalised time** of the solver tick, shape
    ``()`` or ``(B,)`` for B simultaneous voice instances.  It is NOT a
    time-series window; there is no iteration over samples here.

    Gradients flow to all five scalar Parameters.  The hard thresholds
    (indicator functions) are not differentiable at exactly t=t1/t2/t3, but
    gradients exist everywhere else and the optimiser learns correctly.

    Phase layout (normalised 0-1 time)
    -----------------------------------
    [0, t1]    attack  — ramp from 0 to *peak*
    [t1, t2]   decay   — ramp from *peak* to *sustain*
    [t2, t3]   sustain — flat at *sustain*
    [t3, 1]    release — ramp from *sustain* to 0
    """
    eps = torch.tensor(1e-4, dtype=t_norm.dtype, device=t_norm.device)
    dur = torch.tensor(1.0, dtype=t_norm.dtype, device=t_norm.device)
    a = attack.clamp(min=eps)
    dk = decay.clamp(min=eps)
    r = release.clamp(min=eps)

    t1 = a.clamp(max=dur - eps)
    t2 = (a + dk).clamp(max=dur - eps)
    t3 = (dur - r).clamp(min=t2 + eps, max=dur)

    # Phase ramp values
    r_atk = (t_norm / t1.clamp(min=eps)).clamp(0.0, 1.0)
    r_dcy = ((t_norm - t1) / (t2 - t1).clamp(min=eps)).clamp(0.0, 1.0)
    r_rel = ((t_norm - t3) / (dur - t3).clamp(min=eps)).clamp(0.0, 1.0)

    v_attack = peak * r_atk
    v_decay = peak + (sustain - peak) * r_dcy
    v_sustain = sustain * torch.ones_like(t_norm)
    v_release = sustain * (1.0 - r_rel)

    # Hard phase gates — non-diff at boundaries but fine for optimisation
    g_atk = (t_norm <= t1).to(t_norm.dtype)
    g_dcy = ((t_norm > t1) & (t_norm <= t2)).to(t_norm.dtype)
    g_sus = ((t_norm > t2) & (t_norm <= t3)).to(t_norm.dtype)
    g_rel = (t_norm > t3).to(t_norm.dtype)

    return (
        v_attack * g_atk
        + v_decay * g_dcy
        + v_sustain * g_sus
        + v_release * g_rel
    ).clamp(min=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Catmull-Rom spline evaluation — differentiable w.r.t. knot_values
# ──────────────────────────────────────────────────────────────────────────────

def catmull_rom_eval(
    t: Tensor,            # () or (B,) float64 — current time, one moment per instance
    knot_times: Tensor,   # (K,) float64 buffer, strictly monotone in [0, 1]
    knot_values: Tensor,  # (K,) float64 parameter
) -> Tensor:
    """Evaluate a piecewise Catmull-Rom spline at the current sample moment.

    ``t`` is the **current normalised time** of the solver tick — shape ``()``
    for a single instance or ``(B,)`` for B simultaneous instances.  It is
    NOT a time-series array; this function evaluates one point (or B points)
    of the spline, not a window.

    ``knot_times`` must be a non-learnable buffer; gradients flow only through
    ``knot_values``.  Phantom knots at boundaries are clamped to the nearest
    real knot so that the spline is C1 and value-preserving at t=0 and t=1.

    Parameters
    ----------
    t:
        Current normalised time in [0, 1].  Shape ``()`` or ``(B,)``.
    knot_times:
        Monotone knot positions in [0, 1], shape (K,).  K ≥ 2.
    knot_values:
        Knot amplitudes, shape (K,).  These are the learnable parameters.

    Returns
    -------
    Tensor of shape matching ``t``, same dtype.
    """
    t = t.clamp(0.0, 1.0)
    K = knot_values.shape[0]

    # Segment index: largest i such that knot_times[i] <= t
    # t shape: () or (B,) — one time point per instance, not a window
    # Unsqueeze for broadcast against (K,) knot_times
    t_q = t.unsqueeze(-1) if t.dim() > 0 else t.unsqueeze(0).unsqueeze(-1)  # (..., 1)
    seg_idx = (t_q >= knot_times.unsqueeze(0)).long().sum(-1) - 1
    seg_idx = seg_idx.squeeze(0) if t.dim() == 0 else seg_idx
    seg_idx = seg_idx.clamp(0, K - 2)

    # Normalised position within segment → u ∈ [0, 1]
    t0 = knot_times[seg_idx]                          # () or (B,)
    t1 = knot_times[(seg_idx + 1).clamp(max=K - 1)]  # () or (B,)
    dt = (t1 - t0).clamp(min=1e-12)
    u = ((t - t0) / dt).clamp(0.0, 1.0)              # () or (B,)
    u2 = u * u
    u3 = u2 * u

    # Four control points with boundary clamping (phantom knots mirror the edge)
    i0 = (seg_idx - 1).clamp(0, K - 1)
    i1 = seg_idx.clamp(0, K - 1)
    i2 = (seg_idx + 1).clamp(0, K - 1)
    i3 = (seg_idx + 2).clamp(0, K - 1)

    p0 = knot_values[i0]
    p1 = knot_values[i1]
    p2 = knot_values[i2]
    p3 = knot_values[i3]

    # Standard centripetal Catmull-Rom (tension = 0.5)
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: build ParametricCurve objects from AnalyticVoice fields
# ──────────────────────────────────────────────────────────────────────────────

def _norm_v(physical: float, v_lo: float, v_hi: float) -> float:
    span = v_hi - v_lo
    return (physical - v_lo) / span if abs(span) > 1e-30 else 0.5


def _parametric_from_voice_envelope(voice: object) -> ParametricCurve:
    """Build a ParametricCurve envelope from an AnalyticVoice instance.

    Tries knot-based envelope first (``voice.active_knots()``), falls back to
    ADSR scalar parameters.  Returns a default envelope if neither is available.
    """
    piecewise = getattr(voice, "piecewise_env", None)
    if piecewise is not None and getattr(piecewise, "curve", None) is not None:
        return piecewise.curve

    knots = None
    if hasattr(voice, "active_knots"):
        try:
            knots = voice.active_knots()
        except Exception:
            knots = None

    c = default_envelope(name="voice_envelope")
    c.points.clear()

    if knots and len(knots) >= 2:
        for t, v in knots:
            c.add_point(float(t), float(v))
        return c

    adsr    = getattr(voice, "adsr", None)
    attack  = float(getattr(adsr, "attack",  0.005)) if adsr else 0.005
    decay   = float(getattr(adsr, "decay",   0.04))  if adsr else 0.04
    sustain = float(getattr(adsr, "sustain", 0.75))  if adsr else 0.75
    release = float(getattr(adsr, "release", 0.08))  if adsr else 0.08
    peak    = float(getattr(adsr, "peak",    1.0))   if adsr else 1.0
    a  = min(attack,  1.0)
    dk = min(decay,   1.0 - a)
    r  = min(release, 1.0 - a - dk)
    ts = a + dk
    te = max(1.0 - r, ts + 0.001)
    for t, v in [(0.0, 0.0), (a, peak), (ts, sustain), (te, sustain), (1.0, 0.0)]:
        c.add_point(t, v)
    return c


def _parametric_from_voice_chirp(voice: object) -> ParametricCurve:
    """Build a ParametricCurve chirp from an AnalyticVoice instance.

    Maps the ChirpSpec's f_delta_start / f_delta_end into the curve's v_lo/v_hi
    range.  Returns a flat default chirp if no chirp spec is present.
    """
    piecewise = getattr(voice, "piecewise_env", None)
    if piecewise is not None and getattr(piecewise, "chirp_curve", None) is not None:
        return piecewise.chirp_curve

    chirp_spec = getattr(voice, "chirp", None)
    if chirp_spec is None:
        return default_chirp(name="voice_chirp")

    ctype = getattr(chirp_spec, "chirp_type", "none")
    f0    = float(getattr(chirp_spec, "f_delta_start", 0.0))
    f1    = float(getattr(chirp_spec, "f_delta_end",   0.0))
    v_lo  = min(f0, f1, -200.0)
    v_hi  = max(f0, f1,  200.0)

    c      = default_chirp(name="voice_chirp")
    c.v_lo = v_lo
    c.v_hi = v_hi
    c.points.clear()

    n     = 5
    times = [i / (n - 1) for i in range(n)]
    if ctype == "none":
        vals = [0.5] * n
    elif ctype == "linear":
        vals = [_norm_v(f0 + (f1 - f0) * t, v_lo, v_hi) for t in times]
    else:
        vals = ([_norm_v(f0, v_lo, v_hi)]
                + [0.5] * (n - 2)
                + [_norm_v(f1, v_lo, v_hi)])
    for t, v in zip(times, vals):
        c.add_point(t, v)
    return c


# NOTE: EnvelopeSpline and ChirpSpline removed — ParametricCurve + NoteStateMachine
# is the only accepted path for envelope and chirp.

# ──────────────────────────────────────────────────────────────────────────────
# VoiceTorchOscillator — ParametricCurve-driven voice, complex128 throughout
# ──────────────────────────────────────────────────────────────────────────────

class VoiceTorchOscillator(nn.Module):
    """Differentiable voice oscillator driven exclusively by ParametricCurve objects.

    Envelope and chirp are owned as ``ParametricCurve`` instances with an
    independent ``NoteStateMachine`` each.  At every solver tick the state
    machine resolves the rule tree for that single moment — no pre-discretization,
    no look-ahead, no Catmull-Rom knot Parameters.

    Synthesis equation (all terms resolved before final emission)
    -----------------------------------------
        out = amp_total · exp(i · phase_total)

    where:
        phase_total = cumsum(2π · (f_base + chirp_hz) / sr) + env_phase + fm_phase
        amp_total   = exp(log_amplitude) · env_mag · fm_mag

    The curve outputs are folded into the primitive magnitude/phase state
    before the final complex sample is emitted. Chirp contributes directly
    to instantaneous frequency, and the envelope's complex field contributes
    magnitude/phase at that same depth rather than as a trailing multiply.

    Learnable parameters
    --------------------
    log_freq              log(freq_hz)
    log_amplitude         log(amplitude)
    phase_origin          initial phase in radians
    semitone_offset       fine pitch shift in semitones
    harmonic_brightness   amplitude roll-off exponent per partial
    harmonic_warp_strength  stretches harmonic ratios
    fm_depth_hz           FM modulation depth (complex exp factor)
    am_depth_amp          (reserved)

    Non-learnable buffers
    ---------------------
    _sample_rate, _duration, _sample_idx, _running_phase

    Python-only attributes (not Parameters or buffers)
    ---------------------------------------------------
    envelope_curve     ParametricCurve
    chirp_curve        ParametricCurve
    _note_state        NoteStateMachine for envelope_curve
    _chirp_state       NoteStateMachine for chirp_curve
    pitch_input_mode   "hz" or "midi" — how the {key}_pitch_in port is interpreted

    Pitch input port (``{key}_pitch_in``)
    --------------------------------------
    When an upstream graph node drives the ``{key}_pitch_in`` port, the
    oscillator ignores its own ``log_freq`` parameter and derives the base
    frequency from the incoming value instead.  ``pitch_input_mode`` selects
    the interpretation:

        "hz"   — the real part of the port value is treated as Hz directly.
        "midi" — the real part is a MIDI note number.  Converted via::

                    f = tuning_ref_hz * 2 ** ((note - tuning_ref_note) / 12)

    ``tuning_ref_hz`` (default 440.0) and ``tuning_ref_note`` (default 69.0)
    are learnable ``nn.Parameter`` s so that A440 tuning and octave transposition
    can be optimised or configured as knobs.  ``pitch_input_mode`` is a plain
    Python toggle (not a ``Parameter``) and can be set at any time.
    """

    def __init__(
        self,
        *,
        freq_hz:                float = 440.0,
        amplitude:              float = 1.0,
        phase_origin:           float = 0.0,
        semitone_offset:        float = 0.0,
        harmonic_brightness:    float = 1.0,
        harmonic_warp_strength: float = 0.0,
        fm_depth_hz:            float = 0.0,
        am_depth_amp:           float = 0.0,
        manifold_type:          str   = "pure",
        harmonic_count:         int   = 8,
        envelope_curve: "Optional[ParametricCurve]" = None,
        chirp_curve:    "Optional[ParametricCurve]" = None,
        loop_enabled:   bool  = False,
        loop_start:     float = 0.1,
        loop_end:       float = 0.9,
        emission_mode:  str   = "single",
        pre_delay:      float = 0.0,
        sample_rate:    float = 48_000.0,
        duration:       float = 1.0,
        pitch_input_mode: str   = "hz",
        tuning_ref_hz:    float = 440.0,
        tuning_ref_note:  float = 69.0,
    ) -> None:
        super().__init__()

        # ── learnable scalars ─────────────────────────────────────────
        self.log_freq = nn.Parameter(
            torch.tensor(math.log(max(freq_hz, 1e-3)), dtype=torch.float64)
        )
        self.log_amplitude = nn.Parameter(
            torch.tensor(math.log(max(amplitude, 1e-9)), dtype=torch.float64)
        )
        self.phase_origin = nn.Parameter(
            torch.tensor(phase_origin, dtype=torch.float64)
        )
        self.semitone_offset = nn.Parameter(
            torch.tensor(semitone_offset, dtype=torch.float64)
        )
        self.harmonic_brightness = nn.Parameter(
            torch.tensor(harmonic_brightness, dtype=torch.float64)
        )
        self.harmonic_warp_strength = nn.Parameter(
            torch.tensor(harmonic_warp_strength, dtype=torch.float64)
        )
        self.fm_depth_hz = nn.Parameter(
            torch.tensor(fm_depth_hz, dtype=torch.float64)
        )
        self.am_depth_amp = nn.Parameter(
            torch.tensor(am_depth_amp, dtype=torch.float64)
        )

        # ── pitch input mode + tuning reference ───────────────────────
        # pitch_input_mode is a plain Python attribute — not learnable,
        # switchable at runtime without touching the parameter graph.
        if pitch_input_mode not in ("hz", "midi"):
            raise ValueError(
                f"pitch_input_mode must be 'hz' or 'midi', got {pitch_input_mode!r}"
            )
        self.pitch_input_mode: str = pitch_input_mode
        # tuning_ref_hz and tuning_ref_note are learnable so they can be
        # optimised or exposed as knobs (e.g. set A=432 Hz or transpose octave).
        self.tuning_ref_hz = nn.Parameter(
            torch.tensor(max(tuning_ref_hz, 1.0), dtype=torch.float64)
        )
        self.tuning_ref_note = nn.Parameter(
            torch.tensor(float(tuning_ref_note), dtype=torch.float64)
        )

        # ── non-learnable buffers ─────────────────────────────────────
        self.register_buffer(
            "_sample_rate", torch.tensor(sample_rate, dtype=torch.float64)
        )
        self.register_buffer(
            "_duration", torch.tensor(max(duration, 1e-9), dtype=torch.float64)
        )
        self.register_buffer("_sample_idx",    torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_running_phase", torch.tensor(0.0, dtype=torch.float64))

        # ── ParametricCurve objects and their state machines ──────────
        # These are plain Python attributes — not Parameters, not buffers.
        self.envelope_curve: ParametricCurve = (
            envelope_curve if envelope_curve is not None
            else default_envelope("voice_envelope")
        )
        self.chirp_curve: ParametricCurve = (
            chirp_curve if chirp_curve is not None
            else default_chirp("voice_chirp")
        )
        self._note_state:  NoteStateMachine = NoteStateMachine()
        self._chirp_state: NoteStateMachine = NoteStateMachine()

        # ── mode flags ────────────────────────────────────────────────
        self.manifold_type:      str   = manifold_type
        self.harmonic_count:     int   = max(1, int(harmonic_count))
        self.loop_enabled:       bool  = bool(loop_enabled)
        self.loop_start:         float = float(loop_start)
        self.loop_end:           float = float(loop_end)
        self.emission_mode:      str   = emission_mode
        self.pre_delay:          float = float(pre_delay)
        # Single-sample runtime cache: reuse work across repeated calls at the
        # same sample instant (for example redundant fixed-point passes).
        self._scalar_cache_key: Optional[tuple[float, int]] = None
        self._scalar_cache_t_norm: Optional[Tensor] = None
        self._scalar_cache_env: Optional[Tensor] = None
        self._scalar_cache_chirp: Optional[Tensor] = None
        self._scalar_cache_amp: Optional[Tensor] = None
        self._scalar_cache_base_sig: Optional[Tensor] = None

    def _invalidate_scalar_cache(self) -> None:
        self._scalar_cache_key = None
        self._scalar_cache_t_norm = None
        self._scalar_cache_env = None
        self._scalar_cache_chirp = None
        self._scalar_cache_amp = None
        self._scalar_cache_base_sig = None

    # ------------------------------------------------------------------
    # Gate control — drives both curve state machines
    # ------------------------------------------------------------------

    def gate_on(self, t_abs: float, velocity: float = 1.0) -> None:
        """Fire a note-on into both envelope and chirp state machines."""
        self._invalidate_scalar_cache()
        self._note_state.gate_on(t_abs, velocity)
        self._chirp_state.gate_on(t_abs, velocity)

    def gate_off(self, t_abs: float) -> None:
        """Fire a note-off into both envelope and chirp state machines."""
        self._invalidate_scalar_cache()
        self._note_state.gate_off(t_abs)
        self._chirp_state.gate_off(t_abs)

    # ------------------------------------------------------------------
    # Knobs
    # ------------------------------------------------------------------

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        try:
            from analytic_driver import AnalyticVoice as _AV
            return _AV.knobs()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_voice(
        cls,
        voice: object,
        sample_rate: float = 48_000.0,
        duration:    float = 1.0,
    ) -> "VoiceTorchOscillator":
        """Construct from an ``AnalyticVoice`` instance."""
        return cls(
            freq_hz=float(getattr(voice, "freq_hz", 440.0)),
            amplitude=float(getattr(voice, "amplitude", 1.0)),
            phase_origin=float(getattr(voice, "phase_origin", 0.0)),
            semitone_offset=float(getattr(voice, "semitone_offset", 0.0)),
            harmonic_brightness=float(getattr(voice, "harmonic_brightness", 1.0)),
            harmonic_warp_strength=float(getattr(voice, "harmonic_warp_strength", 0.0)),
            fm_depth_hz=float(getattr(getattr(voice, "fm", None), "depth_hz", 0.0)),
            am_depth_amp=float(getattr(getattr(voice, "am", None), "depth_amp", 0.0)),
            manifold_type=str(getattr(voice, "manifold_type", "pure")),
            harmonic_count=int(getattr(voice, "harmonic_count", 8)),
            envelope_curve=_parametric_from_voice_envelope(voice),
            chirp_curve=_parametric_from_voice_chirp(voice),
            loop_enabled=bool(getattr(voice, "loop_enabled", False)),
            loop_start=float(getattr(voice, "loop_start", 0.1)),
            loop_end=float(getattr(voice, "loop_end", 0.9)),
            emission_mode=str(getattr(voice, "emission_mode", "single")),
            pre_delay=float(getattr(voice, "pre_delay", 0.0)),
            sample_rate=sample_rate,
            duration=duration,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def tick(self) -> None:
        self._invalidate_scalar_cache()
        self._sample_idx = self._sample_idx + 1  # type: ignore[assignment]

    def set_sample(self, idx: int) -> None:
        self._invalidate_scalar_cache()
        self._sample_idx.fill_(idx)

    def reset(self) -> None:
        self._invalidate_scalar_cache()
        self._sample_idx.zero_()
        self._running_phase.zero_()
        self._note_state.reset()
        self._chirp_state.reset()

    # ------------------------------------------------------------------
    # Harmonic manifold synthesis
    # ------------------------------------------------------------------

    def _synthesize_harmonics(
        self,
        phase_cumsum: Tensor,  # (N,) float64 cumulative phase of fundamental
        f_base:       Tensor,  # (N,) or () float64 base frequency
        *,
        amplitude_scale: Optional[Tensor] = None,
    ) -> Tensor:
        """Harmonic manifold — returns complex128 (N,)."""
        hc   = self.harmonic_count
        bri  = self.harmonic_brightness
        warp = self.harmonic_warp_strength
        sr   = self._sample_rate

        sig      = torch.zeros(phase_cumsum.shape, dtype=_CDTYPE, device=phase_cumsum.device)
        norm_acc = torch.zeros((), dtype=torch.float64, device=phase_cumsum.device)

        for k in range(1, hc + 1):
            k_f     = torch.tensor(float(k), dtype=torch.float64, device=phase_cumsum.device)
            h_ratio = k_f + warp * (k_f - 1.0)
            h_amp   = 1.0 / (k_f ** bri)
            if amplitude_scale is not None:
                h_amp = h_amp * amplitude_scale
            h_phase_incr = 2.0 * math.pi * f_base * h_ratio / sr
            h_phase = torch.cumsum(h_phase_incr, dim=0) + k_f * self.phase_origin
            sig      = sig + h_amp.to(_CDTYPE) * torch.exp(1j * h_phase.to(_CDTYPE))
            norm_acc = norm_acc + h_amp

        return sig / norm_acc.clamp(min=1e-12).to(_CDTYPE)

    # ------------------------------------------------------------------
    # forward — one tick at a time, complex128 throughout
    # ------------------------------------------------------------------

    def forward(
        self,
        x:     Tensor,
        t_abs: Optional[Tensor] = None,
        *,
        env_mod:   Optional[Tensor] = None,
        chirp_mod: Optional[Tensor] = None,
        pitch_in:  Optional[Tensor] = None,
    ) -> Tensor:
        """Synthesize the voice for this solver tick.

        Called once per tick.  Returns one complex128 sample (or N for the
        batch path when ``t_abs`` has N elements).

        Parameters
        ----------
        x:
            Accumulated complex128 input from FM graph edges this tick.
            Applied as ``carrier *= exp(i · fm_depth_hz · x)``.  Pass zeros
            when no FM edges are connected.
        t_abs:
            Absolute time in seconds, shape () or (N,).
            ``None`` → derived from ``_sample_idx``.
        env_mod:
            Instantaneous complex128 modulator from ``{key}_env_mod`` port.
            Multiplied into env_z: ``env_z *= env_mod``.
        chirp_mod:
            Instantaneous complex128 modulator from ``{key}_chirp_mod`` port.
            Multiplied into chirp_z: ``chirp_z *= chirp_mod``.
        pitch_in:
            Optional complex128 value from ``{key}_pitch_in`` port.  When
            non-zero, **overrides** ``log_freq``; the real part is the control
            value and is interpreted according to ``pitch_input_mode``:

                "hz"   — real part is the desired frequency in Hz.
                "midi" — real part is a MIDI note number, converted via::

                              f = tuning_ref_hz * 2 ** ((note - tuning_ref_note) / 12)

            ``semitone_offset`` is still applied on top regardless of mode.
            When ``pitch_in`` is ``None`` or identically zero, ``log_freq``
            is used unchanged.

        Returns
        -------
        complex128 scalar () or (N,).
        """
        sr            = self._sample_rate
        dur           = self._duration
        if t_abs is None:
            t_abs = (self._sample_idx.to(torch.float64) / sr).unsqueeze(0)

        t_abs = t_abs.to(torch.float64)
        N = t_abs.shape[0]

        scalar_key: Optional[tuple[float, int]] = None
        if N == 1:
            scalar_t = float(t_abs.reshape(-1)[0].item())
            scalar_key = (scalar_t, int(self._sample_idx.item()))
            if self._scalar_cache_key != scalar_key:
                self._invalidate_scalar_cache()
                self._scalar_cache_key = scalar_key
                self._scalar_cache_t_norm = (t_abs / dur).clamp(0.0, 1.0)
            t_norm = self._scalar_cache_t_norm
            assert t_norm is not None
        else:
            t_norm = (t_abs / dur).clamp(0.0, 1.0)

        # ── envelope and chirp: envelope stays complex, chirp becomes Hz ─────
        if N == 1:
            if self._scalar_cache_env is None:
                self._scalar_cache_env = self._note_state.evaluate(self.envelope_curve, t_norm[0])
            if self._scalar_cache_chirp is None:
                self._scalar_cache_chirp = self._chirp_state.evaluate(self.chirp_curve, t_norm[0])
            env_z = self._scalar_cache_env
            chirp_z = self._scalar_cache_chirp
        else:
            # Batch path: direct curve evaluation (no state-machine routing)
            env_z   = self.envelope_curve.evaluate_normalized(t_norm)  # (N,) complex128
            chirp_z = self.chirp_curve.evaluate_normalized(t_norm)     # (N,) complex128

        # Resolve curve/modulator state first so it can be injected into the
        # primitive magnitude/phase state before sample emission.
        if env_mod is not None:
            env_z = env_z * env_mod.to(torch.complex128)
        chirp_hz = self.chirp_curve.to_physical(
            chirp_z.real.to(torch.float64).clamp(0.0, 1.0)
        ).to(torch.float64)
        if chirp_mod is not None:
            chirp_hz = chirp_hz + chirp_mod.real.to(torch.float64)

        # ── carrier frequency ─────────────────────────────────────────
        # pitch_in (from upstream port) takes precedence over log_freq.
        # The real part is the control value; we cross complex→real here
        # at the control boundary (not inside synthesis).
        semitone_shift = 2.0 ** (self.semitone_offset / 12.0)
        pitch_active = pitch_in is not None and pitch_in.real.abs().max().item() > 0.0
        if pitch_active:
            p = pitch_in.real.to(torch.float64).reshape(())
            if self.pitch_input_mode == "midi":
                f_base = self.tuning_ref_hz * torch.pow(
                    torch.tensor(2.0, dtype=torch.float64, device=p.device),
                    (p - self.tuning_ref_note) / 12.0,
                ) * semitone_shift
            else:  # "hz"
                f_base = p.clamp(min=1e-3) * semitone_shift
        else:
            f_base = torch.exp(self.log_freq) * semitone_shift

        # FM from graph edge remains complex; decompose it once into the same
        # magnitude/phase state instead of stacking a later carrier multiply.
        fm_factor: Optional[Tensor] = None
        if x is not None:
            x_c = x.to(_CDTYPE)
            if x_c.dim() == 0:
                x_c = x_c.unsqueeze(0)
            if x_c.shape[0] == N:
                fm_factor = torch.exp(1j * self.fm_depth_hz.to(_CDTYPE) * x_c)

        # ── global amplitude ──────────────────────────────────────────
        if scalar_key is not None and self._scalar_cache_amp is not None:
            A = self._scalar_cache_amp
        else:
            A = torch.exp(self.log_amplitude).to(_CDTYPE)
            if scalar_key is not None:
                self._scalar_cache_amp = A

        total_factor = A * env_z.to(_CDTYPE)
        if fm_factor is not None:
            total_factor = total_factor * fm_factor
        synth = ComplexSignalAggregator(float(sr.item()), N, device=t_abs.device)
        synth.add_frequency_hz(chirp_hz.reshape(-1))
        synth.add_complex_factor(total_factor.reshape(-1))
        f_inst = synth.frequency_series(f_base).reshape(-1)
        phase_cumsum = synth.phase_series(f_base, phase_origin=self.phase_origin).reshape(-1)
        amp_total = synth.amplitude.reshape(-1)

        mt = self.manifold_type
        if mt in ("harmonic", "harmonic_warp") and self.harmonic_count > 1:
            out = self._synthesize_harmonics(
                phase_cumsum,
                f_inst,
                amplitude_scale=amp_total,
            )
        else:
            out = amp_total.to(_CDTYPE) * torch.exp(1j * phase_cumsum.to(_CDTYPE))

        # ── pre-delay ─────────────────────────────────────────────────
        if self.pre_delay > 0.0:
            mask = t_abs < self.pre_delay
            if mask.any():
                out = out.clone()
                out[mask] = torch.zeros(1, dtype=_CDTYPE, device=out.device)

        if N == 1:
            return out.squeeze(0)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# MetaVoiceNode — compound nn.Module that builds TensorNode / TensorEdge entries
# ──────────────────────────────────────────────────────────────────────────────

_VOICE_SIGNAL_INPUT_PORTS = frozenset({"fm_in", "am_in", "env_mod_in", "chirp_mod_in"})
_VOICE_PARAMETER_INPUT_PORTS = frozenset({"pitch_in"})


@dataclass(frozen=True)
class VoicePortOccupancy:
    """Binary occupied-port view for one voice node."""

    signal_inputs: frozenset[str] = frozenset()
    parameter_inputs: frozenset[str] = frozenset()

    @classmethod
    def from_ports(cls, ports: Sequence[str] | None = None) -> "VoicePortOccupancy":
        signal_inputs: set[str] = set()
        parameter_inputs: set[str] = set()
        unknown: list[str] = []
        for raw_port in ports or ():
            port = str(raw_port or "")
            if not port:
                continue
            if port in _VOICE_SIGNAL_INPUT_PORTS:
                signal_inputs.add(port)
            elif port in _VOICE_PARAMETER_INPUT_PORTS:
                parameter_inputs.add(port)
            else:
                unknown.append(port)
        if unknown:
            raise KeyError(f"Unknown voice input port(s): {sorted(unknown)}")
        return cls(
            signal_inputs=frozenset(signal_inputs),
            parameter_inputs=frozenset(parameter_inputs),
        )

    def has_input(self, port_key: str) -> bool:
        port = str(port_key or "")
        return port in self.signal_inputs or port in self.parameter_inputs


def _coerce_voice_port_occupancy(
    occupancy: VoicePortOccupancy | Sequence[str] | None,
) -> VoicePortOccupancy:
    if isinstance(occupancy, VoicePortOccupancy):
        return occupancy
    return VoicePortOccupancy.from_ports(occupancy)


def _prune_orphan_tensor_graph(
    nodes: Sequence[TensorNode],
    edges: Sequence[TensorEdge],
    *,
    keep_keys: Sequence[str] = (),
) -> Tuple[List[TensorNode], List[TensorEdge]]:
    keep = {str(key) for key in keep_keys}
    active_keys = set(keep)
    for edge in edges:
        active_keys.add(str(edge.src_key))
        active_keys.add(str(edge.dst_key))
    pruned_nodes = [node for node in nodes if node.key in active_keys]
    valid_keys = {node.key for node in pruned_nodes}
    pruned_edges = [
        edge for edge in edges
        if edge.src_key in valid_keys and edge.dst_key in valid_keys
    ]
    return pruned_nodes, pruned_edges


class MetaVoiceNode(nn.Module):
    """Compound voice node: owns a ``VoiceTorchOscillator`` and exposes ports.

    INSTANTANEOUS SOLVER NOTE
    --------------------------
    Every transform registered with ``build_nodes()`` is called **once per
    solver tick** and produces exactly one output value (or B values for B
    simultaneous instances of this node).  The ``_t_window`` stored here is
    a scalar time (or (B,) per-instance time) for the current tick — it is
    NOT a sample buffer or a window.

    Calling ``build_nodes()`` returns the ``TensorNode`` / ``TensorEdge``
    entries that should be added to a ``GraphSolver``.  The returned nodes are:

        ``{key}_out``       — primary signal output (oscillator transform)
        ``{key}_fm``        — FM accumulator (pass-through, no transform)
        ``{key}_am``        — AM accumulator (pass-through, no transform)
        ``{key}_env_mod``   — envelope spline modulation input (capture node)
        ``{key}_chirp_mod`` — chirp spline modulation input (capture node)

    FM/AM edges carry a one-sample delay by default (causal constraint).
    env_mod/chirp_mod edges carry zero weight (ordering only; value is
    captured as a side-effect and passed into the oscillator, not summed
    into the output node’s accumulator).

    Parameters
    ----------
    key:
        Unique string identifier for this voice node.
    oscillator:
        ``VoiceTorchOscillator`` instance owned by this node.
    layer:
        Solver layer tag (e.g. ``"signal"``).
    causal_mod_delay:
        If ``True`` (default), FM/AM accumulator → output edges carry a
        one-sample delay so that the oscillator’s SCC is acyclic.
        Set ``False`` only when the solver is iterating the cyclic SCC.
    """

    def __init__(
        self,
        key: str,
        oscillator: VoiceTorchOscillator,
        *,
        layer: str = "signal",
        causal_mod_delay: bool = True,
    ) -> None:
        super().__init__()
        self.key = key
        self.oscillator = oscillator
        self.layer = layer
        self.causal_mod_delay = bool(causal_mod_delay)

        # Per-step time window (set by set_time_window / set_sample before step)
        self._t_window: Optional[Tensor] = None
        # Live driver modulation signals (set by side-effect accumulator nodes)
        self._env_mod:   Optional[Tensor] = None
        self._chirp_mod: Optional[Tensor] = None
        # Live pitch input from upstream port (set by pitch_in capture node)
        self._pitch_in:  Optional[Tensor] = None

    # ------------------------------------------------------------------
    def set_sample(self, idx: int) -> None:
        """Point the oscillator at a specific sample index."""
        self.oscillator.set_sample(idx)
        sr = float(self.oscillator._sample_rate.item())
        self._t_window = torch.tensor([idx / sr], dtype=torch.float64)

    def set_time_window(self, t_start: float, n_samples: int, sr: float) -> None:
        """Set a multi-sample time window for batch synthesis."""
        t = torch.arange(n_samples, dtype=torch.float64) / sr + t_start
        self._t_window = t

    def tick(self) -> None:
        """Advance the oscillator's sample counter and update ``_t_window``."""
        self.oscillator.tick()
        sr = float(self.oscillator._sample_rate.item())
        idx = int(self.oscillator._sample_idx.item())
        self._t_window = torch.tensor([idx / sr], dtype=torch.float64)

    def reset(self) -> None:
        """Reset oscillator state."""
        self.oscillator.reset()
        self._t_window = None
        self._env_mod = None
        self._chirp_mod = None
        self._pitch_in = None

    def gate_on(self, t_abs: float, velocity: float = 1.0) -> None:
        """Fire a note-on into the oscillator's state machines."""
        self.oscillator.gate_on(t_abs, velocity)

    def gate_off(self, t_abs: float) -> None:
        """Fire a note-off into the oscillator's state machines."""
        self.oscillator.gate_off(t_abs)

    # ------------------------------------------------------------------
    def _make_transform(self) -> Callable[[Tensor], Tensor]:
        """Return the ``TensorNode.transform`` closure for the output node.

        The closure captures ``self`` so that updates to ``_t_window`` and
        oscillator parameters are visible on every call.
        """
        node = self

        def _transform(x: Tensor) -> Tensor:
            t = node._t_window
            if t is None:
                return node.oscillator.forward(
                    x,
                    env_mod=node._env_mod,
                    chirp_mod=node._chirp_mod,
                    pitch_in=node._pitch_in,
                )
            return node.oscillator.forward(
                x, t,
                env_mod=node._env_mod,
                chirp_mod=node._chirp_mod,
                pitch_in=node._pitch_in,
            )

        return _transform

    # ------------------------------------------------------------------
    def build_nodes(
        self,
        *,
        occupied_inputs: VoicePortOccupancy | Sequence[str] | None = None,
    ) -> Tuple[List[TensorNode], List[TensorEdge]]:
        """Return ``(nodes, edges)`` to register in a ``GraphSolver``.

        Nodes
        -----
        ``{key}_out``      — oscillator output (drives the synthesis transform)
        ``{key}_fm``       — FM signal accumulator (identity passthrough)
        ``{key}_am``       — AM signal accumulator (identity passthrough)
        ``{key}_env_mod``  — envelope modulation input: the active envelope spline
                             carries this incoming signal (spline × signal.real)
        ``{key}_chirp_mod``— chirp modulation input: the chirp spline carries
                             this incoming signal (spline × signal.real)

        Internal edges
        --------------
        ``{key}_fm``        → ``{key}_out``  (delay=1 if causal_mod_delay; weight=1)
        ``{key}_am``        → ``{key}_out``  (delay=1 if causal_mod_delay; weight=1)
        ``{key}_env_mod``   → ``{key}_out``  (delay=0; weight=0 — ordering only)
        ``{key}_chirp_mod`` → ``{key}_out``  (delay=0; weight=0 — ordering only)

        Only ports listed in ``occupied_inputs`` are materialised. This keeps
        the emitted subgraph aligned with the actual connectome instead of
        advertising dormant helper nodes to the solver.
        """
        k = self.key
        occupancy = _coerce_voice_port_occupancy(occupied_inputs)
        transform = self._make_transform()

        archetype = NodeArchetype(
            semantic_ports=(
                SemanticPortContract(
                    key="signal_out",
                    label="Signal",
                    direction="out",
                    domain="signal",
                    semantic_role="voice_signal",
                ),
                SemanticPortContract(
                    key="fm_in",
                    label="FM",
                    direction="in",
                    domain="signal",
                    semantic_role="fm_modulator",
                ),
                SemanticPortContract(
                    key="am_in",
                    label="AM",
                    direction="in",
                    domain="signal",
                    semantic_role="am_modulator",
                ),
                SemanticPortContract(
                    key="env_mod_in",
                    label="Env Mod",
                    direction="in",
                    domain="signal",
                    semantic_role="envelope_modulator",
                ),
                SemanticPortContract(
                    key="chirp_mod_in",
                    label="Chirp Mod",
                    direction="in",
                    domain="signal",
                    semantic_role="chirp_modulator",
                ),
            ),
        )

        out_node = TensorNode(
            key=f"{k}_out",
            layer=self.layer,
            transform=transform,
            archetype=archetype,
            layer_presence=NodeLayerPresence(
                output_layers=(self.layer,),
                parameter_inputs=True,
            ),
        )
        fm_node = TensorNode(
            key=f"{k}_fm",
            layer=self.layer,
            transform=None,  # pure accumulator
            archetype=NodeArchetype(),
            layer_presence=NodeLayerPresence(
                input_layers=(self.layer,),
            ),
        )
        am_node = TensorNode(
            key=f"{k}_am",
            layer=self.layer,
            transform=None,
            archetype=NodeArchetype(),
            layer_presence=NodeLayerPresence(
                input_layers=(self.layer,),
            ),
        )

        node_self = self

        def _env_mod_capture(x: Tensor) -> Tensor:
            # x is the solver's accumulated input to this node.  When no driver
            # edge is wired here, x is the zero-initialised accumulator state.
            # Treat all-zero as "not connected" so the oscillator uses the bare
            # envelope unmodified.  A driver sending genuine zero is equivalent.
            node_self._env_mod = x if x.abs().max().item() > 0.0 else None
            return x

        def _chirp_mod_capture(x: Tensor) -> Tensor:
            node_self._chirp_mod = x if x.abs().max().item() > 0.0 else None
            return x

        def _pitch_in_capture(x: Tensor) -> Tensor:
            # Store non-zero pitch values; zero means "not driven this tick".
            # Upstream nodes sending genuine zero Hz / MIDI-0 are indistinguishable
            # from "no connection" — that edge case is intentionally elided.
            node_self._pitch_in = x if x.real.abs().max().item() > 0.0 else None
            return x

        env_mod_node = TensorNode(
            key=f"{k}_env_mod",
            layer=self.layer,
            transform=_env_mod_capture,
            archetype=NodeArchetype(),
            layer_presence=NodeLayerPresence(input_layers=(self.layer,)),
        )
        chirp_mod_node = TensorNode(
            key=f"{k}_chirp_mod",
            layer=self.layer,
            transform=_chirp_mod_capture,
            archetype=NodeArchetype(),
            layer_presence=NodeLayerPresence(input_layers=(self.layer,)),
        )
        pitch_in_node = TensorNode(
            key=f"{k}_pitch_in",
            layer=self.layer,
            transform=_pitch_in_capture,
            archetype=NodeArchetype(
                semantic_ports=(
                    SemanticPortContract(
                        key="pitch_in",
                        label="Pitch In",
                        direction="in",
                        domain="control",
                        semantic_role="pitch_source",
                    ),
                ),
            ),
            layer_presence=NodeLayerPresence(
                input_layers=(self.layer,),
                parameter_outputs=True,
            ),
        )

        delay_samples = 1 if self.causal_mod_delay else 0
        fm_edge = TensorEdge(
            src_key=f"{k}_fm",
            dst_key=f"{k}_out",
            weight=complex(1.0, 0.0),
            delay_samples=delay_samples,
            semantic_role="fm_modulator",
            src_port="signal_out",
            dst_port="fm_in",
        )
        am_edge = TensorEdge(
            src_key=f"{k}_am",
            dst_key=f"{k}_out",
            weight=complex(1.0, 0.0),
            delay_samples=delay_samples,
            semantic_role="am_modulator",
            src_port="signal_out",
            dst_port="am_in",
        )
        # Zero-weight edges: ensure capture nodes run before output node in
        # topological order; weight=0 means no contribution to x.
        env_mod_edge = TensorEdge(
            src_key=f"{k}_env_mod",
            dst_key=f"{k}_out",
            weight=complex(0.0, 0.0),
            delay_samples=0,
            semantic_role="envelope_modulator",
            src_port="signal_out",
            dst_port="env_mod_in",
        )
        chirp_mod_edge = TensorEdge(
            src_key=f"{k}_chirp_mod",
            dst_key=f"{k}_out",
            weight=complex(0.0, 0.0),
            delay_samples=0,
            semantic_role="chirp_modulator",
            src_port="signal_out",
            dst_port="chirp_mod_in",
        )
        pitch_in_edge = TensorEdge(
            src_key=f"{k}_pitch_in",
            dst_key=f"{k}_out",
            weight=complex(0.0, 0.0),
            delay_samples=0,
            semantic_role="pitch_source",
            src_port="pitch_in",
            dst_port="signal_out",
        )

        nodes: list[TensorNode] = [out_node]
        edges: list[TensorEdge] = []
        if occupancy.has_input("fm_in"):
            nodes.append(fm_node)
            edges.append(fm_edge)
        if occupancy.has_input("am_in"):
            nodes.append(am_node)
            edges.append(am_edge)
        if occupancy.has_input("env_mod_in"):
            nodes.append(env_mod_node)
            edges.append(env_mod_edge)
        if occupancy.has_input("chirp_mod_in"):
            nodes.append(chirp_mod_node)
            edges.append(chirp_mod_edge)
        if occupancy.has_input("pitch_in"):
            nodes.append(pitch_in_node)
            edges.append(pitch_in_edge)
        return _prune_orphan_tensor_graph(nodes, edges, keep_keys=(out_node.key,))

    # ------------------------------------------------------------------
    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        """Knob manifest — delegates to ``VoiceTorchOscillator.knobs()``."""
        return VoiceTorchOscillator.knobs()

    @classmethod
    def from_voice(
        cls,
        voice: object,
        *,
        sample_rate: float = 48_000.0,
        duration: float = 1.0,
        layer: str = "signal",
        causal_mod_delay: bool = True,
    ) -> "MetaVoiceNode":
        """Construct from an ``AnalyticVoice`` dataclass instance."""
        key = str(getattr(voice, "key", "voice"))
        osc = VoiceTorchOscillator.from_voice(voice, sample_rate=sample_rate, duration=duration)
        return cls(key, osc, layer=layer, causal_mod_delay=causal_mod_delay)


# ──────────────────────────────────────────────────────────────────────────────
# MixerSumNode — passive accumulator (gains live on the level grid)
# ──────────────────────────────────────────────────────────────────────────────

class MixerSumNode(nn.Module):
    """Passive accumulator mixer node.

    The mixer has no learnable parameters of its own.  Gain weights live on
    the *level grid* (``LevelGrid``), not here.  This node is a pure signal
    sink: ``TensorEdge`` contributions from all source channels accumulate
    into the node's ``acc`` before ``transform`` is called.  Since
    ``transform=None``, ``GraphSolver`` passes the accumulated sum straight
    through.
    """

    def __init__(self, key: str, layer: str = "signal") -> None:
        super().__init__()
        self.key = key
        self.layer = layer

    def build_node(self) -> TensorNode:
        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=None,
            archetype=MixerArchetype(
                mixer=MixerNodeSpec(n_in_channels=1, n_out_channels=1),
                semantic_ports=(
                    SemanticPortContract(
                        key="mix_in",
                        label="Mix input",
                        direction="in",
                        domain="signal",
                        semantic_role="mix_source",
                        mixer_grid_capable=True,
                    ),
                    SemanticPortContract(
                        key="mix_out",
                        label="Mix output",
                        direction="out",
                        domain="signal",
                        semantic_role="mix_result",
                    ),
                ),
            ),
        )

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Encoder / Loss / Step metanodes
# ──────────────────────────────────────────────────────────────────────────────

class EncoderMetaNode(nn.Module):
    """Graph-resident encoder: reads input channels, emits parameter updates.

    Fires at solve-start via ``hook_at_solve_start``.  Reads accumulated
    values for ``input_keys`` from the solver state, runs ``encoder.forward``,
    and calls each setter in ``param_setters`` with the result.

    The encoder owns its ``optimizer``; ``StepMetaNode`` drives the step.
    Nothing in the fixed network is trainable unless reachable from an
    encoder's output edges.

    Parameters
    ----------
    key:
        Unique graph key.
    encoder_module:
        ``nn.Module`` mapping stacked input values to a flat output tensor.
    optimizer:
        Optimizer over ``encoder_module.parameters()``.
    input_keys:
        Ordered list of solver state keys read each tick.
    param_setters:
        ``{label: callable(output_tensor)}`` — each setter writes to its
        target parameter in the fixed network.
    layer:
        Layer string for the underlying ``TensorNode``.
    """

    def __init__(
        self,
        key: str,
        encoder_module: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        input_keys: List[str],
        param_setters: Dict[str, Callable[[Tensor], None]],
        layer: str = "encoder",
    ) -> None:
        super().__init__()
        self.key = key
        self.encoder = encoder_module
        self.optimizer = optimizer
        self.input_keys = list(input_keys)
        self.param_setters = param_setters
        self.layer = layer

    def build_node(self, get_state: Callable[[], Dict[str, Tensor]]) -> TensorNode:
        def _hook() -> None:
            state = get_state()
            vals = [state[k] for k in self.input_keys if k in state]
            if not vals:
                return
            x = torch.stack(vals)
            out = self.encoder(x)
            for setter in self.param_setters.values():
                setter(out)

        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=None,
            fire_at_solve_start=True,
            hook_at_solve_start=_hook,
        )

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return []


class LossMetaNode(nn.Module):
    """Accumulates predictions over the full solve, computes loss once, calls backward.

    One complete solve = all N sample ticks.  Each tick the node's transform
    appends the current prediction to ``self._preds``.  After the last tick
    (``hook_after_end``), loss is computed over the entire accumulated buffer
    and ``backward()`` is called exactly once.  ``reference`` must be a
    Tensor of the same length as the solve (shape ``(N,)``, complex128).

    Parameters
    ----------
    key:
        Unique graph key.
    prediction_key:
        Solver state key for the per-tick predicted signal (e.g. ``"mix_out"``).
    reference:
        Full-solve reference tensor, shape ``(N,)``, complex128.  Must cover
        the same number of ticks as the solve duration.
    loss_fn:
        ``(pred_seq, ref_seq) → scalar Tensor`` where both args are ``(N,)``.
        Defaults to mean complex L2 over the full sequence.
    layer:
        Layer string for the underlying ``TensorNode``.
    """

    def __init__(
        self,
        key: str,
        *,
        prediction_key: str,
        reference: Tensor,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        layer: str = "loss",
    ) -> None:
        super().__init__()
        self.key = key
        self.prediction_key = prediction_key
        self.register_buffer("reference", reference.to(_CDTYPE))
        if loss_fn is None:
            def _default(pred: Tensor, ref: Tensor) -> Tensor:
                diff = pred.to(_CDTYPE) - ref.to(_CDTYPE)
                return (diff * diff.conj()).real.mean()
            self.loss_fn: Callable[[Tensor, Tensor], Tensor] = _default
        else:
            self.loss_fn = loss_fn
        self.layer = layer
        self.last_loss: Optional[float] = None
        self._preds: List[Tensor] = []

    def reset(self) -> None:
        """Clear the prediction buffer; call before each solve pass."""
        self._preds = []

    def build_node(self, get_state: Callable[[], Dict[str, Tensor]]) -> TensorNode:
        """Return the TensorNode for this loss node.

        The ``transform`` collects one prediction per tick.
        ``hook_after_end`` fires once after ALL ticks, computes full-sequence
        loss, and calls ``backward()`` — never per-tick.
        """
        preds = self._preds
        prediction_key = self.prediction_key

        def _collect(acc: Tensor) -> Tensor:
            # transform receives the accumulated value at this node; we only
            # care about the prediction node's value, fetched from solver state
            state = get_state()
            v = state.get(prediction_key)
            if v is not None:
                preds.append(v.detach() if not v.requires_grad else v)
            return acc  # pass through unchanged (this node is a passive tap)

        def _backward_hook() -> None:
            if not preds:
                return
            pred_seq = torch.stack(preds)          # (N,) complex128
            ref = self.reference[: len(preds)]     # match actual ticks run
            loss = self.loss_fn(pred_seq, ref)
            self.last_loss = float(loss.detach())
            loss.backward()
            preds.clear()

        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=_collect,
            fire_after_end=True,
            hook_after_end=_backward_hook,
        )

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return []


class StepMetaNode(nn.Module):
    """Calls ``optimizer.step()`` + ``zero_grad()`` once after the full solve.

    Must be inserted *after* ``LossMetaNode`` in the node list so the
    dispatch order (insertion order) guarantees ``backward()`` runs first.

    Parameters
    ----------
    key:
        Unique graph key.
    optimizer:
        The encoder's optimizer.  Stepped exactly once per epoch.
    layer:
        Layer string for the underlying ``TensorNode``.
    """

    def __init__(
        self,
        key: str,
        optimizer: torch.optim.Optimizer,
        *,
        layer: str = "step",
    ) -> None:
        super().__init__()
        self.key = key
        self._optimizer = optimizer
        self.layer = layer

    def build_node(self) -> TensorNode:
        opt = self._optimizer

        def _step_hook() -> None:
            opt.step()
            opt.zero_grad()

        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=None,
            fire_after_end=True,
            hook_after_end=_step_hook,
        )

    @classmethod
    def knobs(cls) -> List[KnobSpec]:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build a complete voice + mixer network from a list of AnalyticVoice
# ──────────────────────────────────────────────────────────────────────────────

def build_voice_mixer_network(
    voices: Sequence[object],
    sample_rate: float = 48_000.0,
    duration: float = 1.0,
    layer: str = "signal",
    mixer_key: str = "mix_out",
    causal_mod_delay: bool = True,
    device: torch.device = torch.device("cpu"),
    voice_port_occupancy: Optional[Mapping[str, VoicePortOccupancy | Sequence[str]]] = None,
) -> Tuple["GraphSolver", Dict[str, MetaVoiceNode], MixerSumNode]:
    """Build a ``GraphSolver`` containing one ``MetaVoiceNode`` per voice
    and a single ``MixerSumNode`` that accumulates them.

    Gain weights live on a ``LevelGrid``, not on the mixer node itself.
    Each voice output is wired to the mixer with unit gain via a plain
    ``TensorEdge``; replace these with ``LevelGrid.to_edges()`` when
    per-voice gain control is needed.

    ``voice_port_occupancy`` optionally declares which helper input ports
    should be materialised for each voice key. Unlisted ports are omitted,
    so the solver only sees the occupied voice connectome.

    Returns
    -------
    solver:
        The constructed ``GraphSolver`` (ready to call ``step()`` on).
    voice_nodes:
        Mapping from voice key → ``MetaVoiceNode`` (owns oscillator + state).
    mixer_node:
        The ``MixerSumNode`` instance.
    """
    all_tnodes: List[TensorNode] = []
    all_tedges: List[TensorEdge] = []
    voice_nodes: Dict[str, MetaVoiceNode] = {}

    mixer_nn = MixerSumNode(mixer_key, layer=layer)
    all_tnodes.append(mixer_nn.build_node())

    for v in voices:
        vnode = MetaVoiceNode.from_voice(
            v,
            sample_rate=sample_rate,
            duration=duration,
            layer=layer,
            causal_mod_delay=causal_mod_delay,
        )
        occupancy = None if voice_port_occupancy is None else voice_port_occupancy.get(vnode.key)
        tnodes, tedges = vnode.build_nodes(occupied_inputs=occupancy)
        all_tnodes.extend(tnodes)
        all_tedges.extend(tedges)
        voice_nodes[vnode.key] = vnode

        all_tedges.append(
            TensorEdge(
                src_key=f"{vnode.key}_out",
                dst_key=mixer_key,
                weight=complex(1.0, 0.0),
                semantic_role="mix_source",
            )
        )

    all_tnodes, all_tedges = _prune_orphan_tensor_graph(
        all_tnodes,
        all_tedges,
        keep_keys=(mixer_key,),
    )
    solver = GraphSolver(
        all_tnodes,
        all_tedges,
        sample_rate=sample_rate,
        device=device,
    )
    return solver, voice_nodes, mixer_nn


# ──────────────────────────────────────────────────────────────────────────────
# VoiceTrainer — one backward + one step per full epoch
# ──────────────────────────────────────────────────────────────────────────────

class VoiceTrainer:
    """Train one voice's parameters to match a reference complex waveform.

    One epoch = tick through all N samples, accumulate the full-sequence loss,
    call ``loss.backward()`` once, then ``optimizer.step()`` once.  No
    sub-epoch accumulation windows.
    """

    def __init__(
        self,
        solver: "GraphSolver",
        voice_nodes: Dict[str, MetaVoiceNode],
        optimizer: torch.optim.Optimizer,
        *,
        target_key: str = "mix_out",
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        self.solver = solver
        self.voice_nodes = voice_nodes
        self.optimizer = optimizer
        self.target_key = target_key

        if loss_fn is None:
            def _default_loss(pred: Tensor, ref: Tensor) -> Tensor:
                diff = pred.to(_CDTYPE) - ref.to(_CDTYPE)
                return (diff * diff.conj()).real.mean()
            self.loss_fn: Callable[[Tensor, Tensor], Tensor] = _default_loss
        else:
            self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    def _set_all_samples(self, idx: int) -> None:
        for vnode in self.voice_nodes.values():
            vnode.set_sample(idx)

    # ------------------------------------------------------------------
    def fit(
        self,
        reference: Tensor,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """Run the training loop.

        One pass through the reference = one epoch = one backward = one step.

        Parameters
        ----------
        reference:
            Complex128 tensor of shape (N,).
        n_steps:
            Number of full passes (epochs).  Defaults to 1.
        verbose:
            Print per-epoch loss if True.
        callback:
            Optional callable ``(epoch_idx, loss)`` called after each step.

        Returns
        -------
        List of per-epoch losses (one entry per epoch).
        """
        reference = reference.to(_CDTYPE)
        N = reference.shape[0]
        n_steps = max(1, int(n_steps or 1))
        losses: List[float] = []

        for epoch in range(n_steps):
            self.solver.reset()
            for vnode in self.voice_nodes.values():
                vnode.reset()

            self.optimizer.zero_grad()
            epoch_loss = torch.zeros((), dtype=torch.float64)

            for i in range(N):
                self._set_all_samples(i)
                outputs = self.solver.step({})
                pred = outputs.get(self.target_key)
                if pred is None:
                    raise KeyError(
                        f"Target key {self.target_key!r} not in solver outputs. "
                        f"Available: {list(outputs.keys())}"
                    )
                epoch_loss = epoch_loss + self.loss_fn(pred, reference[i])

            # One backward and one step for the entire epoch
            epoch_loss.backward()
            self.optimizer.step()

            loss_val = float(epoch_loss.detach().item())
            losses.append(loss_val)
            if verbose:
                print(f"[VoiceTrainer] epoch={epoch}  loss={loss_val:.6e}")
            if callback is not None:
                callback(epoch, loss_val)

        return losses
