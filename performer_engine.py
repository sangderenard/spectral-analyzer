"""performer_engine.py — Torch driver synthesis engine.

Synthesises ALL D driver slots across the entire patch in one batched
torch operation.

Ownership hierarchy
-------------------
  Performer  →  Instrument (body)  →  Driver(s)  →  Voice components

Each driver slot is one (instrument-driver, voice-component) pair.  Performer
timing and chirp humanization are baked into the per-slot DriverConfig tensors
at build time (pre_delay_samples, chirp offsets).  Performers themselves are
state-machine objects and have no representation here.

Voice components (transient, air, body, signal, …) are individual routing-
graph nodes.  Each driver slot maps to exactly one voice node (voice_idx).
Multiple slots on the same physical driver share an instrument_idx, which is
used for intra-instrument sympathetic coupling.

Pipeline
--------
1. ``driver_synthesis_step``   – chirp + FM + harmonics + envelope  → (D, T)
   ``scatter_to_voices``       – accumulate slots per voice node     → (V, T)
2. FM/AM cross-voice resolution via ``multi_level_driver_step`` topo sort
3. Intra-instrument sympathetic coupling (external, keyed by instrument_idx)
4. Instrument body cavity (external, via cavity_engine)
5. Room solve (external, via cavity_engine)

Convention: ``h_amps`` must be pre-normalised so that amplitude=1 produces
peak ≈ 1 for a single-harmonic voice.  For harmonic voices::

    h_amps[k] = (1/k^brightness) / Σ_k (1/k^brightness)

For pure voices: ``h_amps = [1.0, 0, ...]``, ``h_ratios = [1.0, 0, ...]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

_2PI = 2.0 * math.pi

# ── Chirp type codes ─────────────────────────────────────────────────────────
CHIRP_NONE = 0
CHIRP_LINEAR = 1
CHIRP_EXPONENTIAL = 2
CHIRP_POWER = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriverConfig:
    """Static config tensors for D driver note-lines.

    Built from patch + performer map.  Updated on note activation /
    deactivation or parameter changes.

    All tensors live on *device* with float64 / complex128 precision.
    """

    D: int                         # total driver slot count
    H: int                         # max harmonics across all drivers
    K: int                         # max envelope knots
    V: int                         # voice count (routing graph node count)
    device: torch.device

    # -- Per-driver (D,) float64 --
    f0: Tensor                     # base frequency (Hz)
    amplitude: Tensor              # linear amplitude
    phase_origin: Tensor           # initial phase (rad)
    pre_delay_samples: Tensor      # (D,) int64 — leading silence in samples
    note_duration: Tensor          # (D,) seconds (total note length)
    active: Tensor                 # (D,) bool — slot currently sounding

    # -- Chirp (D,) --
    chirp_type: Tensor             # int64 — CHIRP_NONE / LINEAR / EXP / POWER
    chirp_f_start: Tensor          # Hz deviation at note start
    chirp_f_end: Tensor            # Hz deviation at note end
    chirp_tau: Tensor              # exponential time constant (s)
    chirp_power: Tensor            # power-law exponent

    # -- Harmonics (D, H) --
    h_ratios: Tensor               # frequency multipliers (0-padded)
    h_amps: Tensor                 # amplitude weights (0-padded, pre-normalised)
    n_harmonics: Tensor            # (D,) int

    # -- Envelope (D, K) --
    env_t: Tensor                  # knot times (s, ascending, right-padded with 1e30)
    env_v: Tensor                  # knot values (right-padded with last real value)
    env_n: Tensor                  # (D,) int — actual knot count

    # -- Mapping --
    voice_idx: Tensor              # (D,) int — which voice routing node
    instrument_idx: Tensor         # (D,) int — which instrument (for sympathetic coupling)

    # -- FM / AM wiring --
    fm_source_voice: Tensor        # (D,) int — FM modulator voice (-1 = none)
    fm_depth_hz: Tensor            # (D,) FM depth in Hz
    am_source_voice: Tensor        # (D,) int — AM modulator voice (-1 = none)
    am_depth: Tensor               # (D,) AM modulation depth

    # -- Optional packed parametric curves --
    parametric_env_packed: Any | None = None
    parametric_env_row: Tensor | None = None     # (D,) int64 — row in packed batch, -1 = none
    parametric_chirp_packed: Any | None = None
    parametric_chirp_row: Tensor | None = None   # (D,) int64 — row in packed batch, -1 = none


@dataclass
class DriverState:
    """Mutable state carried across chunks for D drivers."""

    phase_acc: Tensor              # (D, H) accumulated phase (rad)
    t_pos: Tensor                  # (D,) absolute time position (s)


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def init_driver_state(cfg: DriverConfig) -> DriverState:
    """Zero-initialise driver state.

    Phase accumulators are seeded with per-harmonic phase origins::

        phase_acc[d, h] = (h + 1) * phase_origin[d]

    where *h* is 0-based (harmonic index 1, 2, … H).
    """
    dev = cfg.device
    h_k = torch.arange(1, cfg.H + 1, dtype=torch.float64, device=dev)  # (H,)
    phase_acc = h_k.unsqueeze(0) * cfg.phase_origin.unsqueeze(1)         # (D, H)
    return DriverState(
        phase_acc=phase_acc,
        t_pos=torch.zeros(cfg.D, dtype=torch.float64, device=dev),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Envelope — piecewise linear via searchsorted
# ═══════════════════════════════════════════════════════════════════════════════

def _envelope(
    env_t: Tensor, env_v: Tensor, K: int, t_note: Tensor,
) -> Tensor:
    """Piecewise-linear envelope for all drivers.

    Parameters
    ----------
    env_t   : (D, K) float64 — knot times (ascending, right-padded with 1e30)
    env_v   : (D, K) float64 — knot values (right-padded with last real value)
    K       : int — max knots (second dim of env_t / env_v)
    t_note  : (D, T) float64 — time within note (s), clamped ≥ 0

    Returns
    -------
    (D, T) float64 — envelope values ≥ 0
    """
    idx = torch.searchsorted(env_t.contiguous(), t_note.contiguous())   # (D, T)
    idx = idx.clamp(1, K - 1)
    idx_lo = idx - 1
    t0 = env_t.gather(1, idx_lo)                                        # (D, T)
    t1 = env_t.gather(1, idx)
    v0 = env_v.gather(1, idx_lo)
    v1 = env_v.gather(1, idx)
    frac = ((t_note - t0) / (t1 - t0).clamp(min=1e-15)).clamp(0.0, 1.0)
    return (v0 + (v1 - v0) * frac).clamp(min=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Chirp
# ═══════════════════════════════════════════════════════════════════════════════

def _chirp(cfg: DriverConfig, t_abs: Tensor) -> Tensor:
    """Compute chirp frequency deviation for all drivers.  (D, T) float64."""
    D, T_len = t_abs.shape
    dev = cfg.device
    delta = torch.zeros(D, T_len, dtype=torch.float64, device=dev)
    fs, fe = cfg.chirp_f_start, cfg.chirp_f_end

    # Linear: lerp start→end over note_duration
    m = cfg.chirp_type == CHIRP_LINEAR
    if m.any():
        dur = cfg.note_duration[m].unsqueeze(1).clamp(min=1e-15)
        frac = (t_abs[m] / dur).clamp(0.0, 1.0)
        delta[m] = fs[m].unsqueeze(1) + (fe[m] - fs[m]).unsqueeze(1) * frac

    # Exponential: start·exp(-t/τ) + end·(1 − exp(-t/τ))
    m = cfg.chirp_type == CHIRP_EXPONENTIAL
    if m.any():
        tau = cfg.chirp_tau[m].unsqueeze(1).clamp(min=1e-9)
        decay = torch.exp(-t_abs[m] / tau)
        delta[m] = fs[m].unsqueeze(1) * decay + fe[m].unsqueeze(1) * (1.0 - decay)

    # Power: start·(1 − (t/dur)^p) + end·(t/dur)^p
    m = cfg.chirp_type == CHIRP_POWER
    if m.any():
        dur = cfg.note_duration[m].unsqueeze(1).clamp(min=1e-15)
        p = cfg.chirp_power[m].unsqueeze(1).clamp(min=1e-3)
        tau_n = (t_abs[m] / dur).clamp(min=0.0).pow(p)
        delta[m] = fs[m].unsqueeze(1) * (1.0 - tau_n) + fe[m].unsqueeze(1) * tau_n

    return delta


def _parametric_curve_series(
    packed: Any,
    row_map: Tensor,
    t_norm: Tensor,
    *,
    physical: bool,
) -> Tensor:
    """Evaluate selected packed parametric curves for drivers with valid rows."""
    D, T = t_norm.shape
    dev = t_norm.device
    out = torch.zeros(D, T, dtype=torch.float64, device=dev)
    if packed is None or row_map is None or row_map.numel() == 0:
        return out

    valid = row_map >= 0
    if not bool(valid.any()):
        return out

    t_sel = t_norm[valid]
    eval_out = packed.evaluate(
        t_sel,
        clamp_r=True,
        apply_activation=True,
        apply_slew=True,
        physical=physical,
    )
    if eval_out.ndim == 1:
        eval_out = eval_out.unsqueeze(0)
    out[valid] = eval_out.real.to(torch.float64)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Core batch synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def driver_synthesis_step(
    cfg: DriverConfig,
    state: DriverState,
    chunk_T: int,
    sr: float,
    *,
    fm_mod: Optional[Tensor] = None,
    am_mod: Optional[Tensor] = None,
    active_override: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, DriverState]:
    """Synthesise *chunk_T* samples for all D drivers in one batch.

    Parameters
    ----------
    cfg             : DriverConfig
    state           : DriverState
    chunk_T         : int — samples per chunk
    sr              : float — sample rate
    fm_mod          : (D, chunk_T) float64, optional — Hz added to f_inst
    am_mod          : (D, chunk_T) float64, optional — depth added to amplitude
    active_override : (D,) bool, optional — override cfg.active for this call

    Returns
    -------
    driver_out  : (D, chunk_T) complex128 — per-driver signals
    voice_out   : (V, chunk_T) complex128 — per-voice sums
    new_state   : DriverState
    """
    D, H, V = cfg.D, cfg.H, cfg.V
    dev = cfg.device
    T = chunk_T
    cdtype = torch.complex128

    if D == 0:
        empty_d = torch.zeros(0, T, dtype=cdtype, device=dev)
        empty_v = torch.zeros(max(V, 0), T, dtype=cdtype, device=dev)
        return empty_d, empty_v, DriverState(
            phase_acc=state.phase_acc.clone(), t_pos=state.t_pos.clone(),
        )

    active = active_override if active_override is not None else cfg.active

    # ── Time axis ─────────────────────────────────────────────────────────
    local_t = torch.arange(T, dtype=torch.float64, device=dev) / sr      # (T,)
    t_abs = state.t_pos.unsqueeze(1) + local_t.unsqueeze(0)               # (D, T)

    # ── Instantaneous frequency ───────────────────────────────────────────
    f_inst = cfg.f0.unsqueeze(1).expand(D, T).clone()                     # (D, T)
    f_inst += _chirp(cfg, t_abs)
    t_norm = (t_abs / cfg.note_duration.unsqueeze(1).clamp(min=1e-15)).clamp(0.0, 1.0)
    if cfg.parametric_chirp_packed is not None and cfg.parametric_chirp_row is not None:
        f_inst += _parametric_curve_series(
            cfg.parametric_chirp_packed,
            cfg.parametric_chirp_row,
            t_norm,
            physical=True,
        )
    if fm_mod is not None:
        f_inst = f_inst + fm_mod

    # ── Phase increment per sample per harmonic ───────────────────────────
    dphi_base = (_2PI / sr) * f_inst                                       # (D, T)
    dphi = dphi_base.unsqueeze(1) * cfg.h_ratios.unsqueeze(2)             # (D, H, T)

    # Cumulative phase with carry from previous chunk
    phase = torch.cumsum(dphi, dim=-1) + state.phase_acc.unsqueeze(2)     # (D, H, T)

    # ── Harmonic synthesis ────────────────────────────────────────────────
    #   sig = Σ_h  h_amp[h] · exp(j · phase[h])
    sig = (
        cfg.h_amps.unsqueeze(2) * torch.exp(1j * phase.to(cdtype))
    ).sum(dim=1)                                                           # (D, T)

    # ── Amplitude + AM + envelope ─────────────────────────────────────────
    amp = cfg.amplitude.unsqueeze(1).expand(D, T)                          # (D, T)
    if am_mod is not None:
        amp = amp * (1.0 + am_mod)
    env = _envelope(cfg.env_t, cfg.env_v, cfg.K, t_abs.clamp(min=0.0))   # (D, T)
    if cfg.parametric_env_packed is not None and cfg.parametric_env_row is not None:
        env_param = _parametric_curve_series(
            cfg.parametric_env_packed,
            cfg.parametric_env_row,
            t_norm,
            physical=False,
        ).clamp(min=0.0)
        param_mask = cfg.parametric_env_row >= 0
        if bool(param_mask.any()):
            env[param_mask] = env_param[param_mask]

    out = (amp * env).to(cdtype) * sig                                    # (D, T)

    # ── Pre-delay mask ────────────────────────────────────────────────────
    sample_global = (
        (state.t_pos * sr).to(torch.int64).unsqueeze(1)
        + torch.arange(T, dtype=torch.int64, device=dev).unsqueeze(0)
    )                                                                      # (D, T)
    out = out * (sample_global >= cfg.pre_delay_samples.unsqueeze(1))

    # ── Activity mask ─────────────────────────────────────────────────────
    out = out * active.unsqueeze(1)

    # ── Scatter to voice level ────────────────────────────────────────────
    voice_out = torch.zeros(V, T, dtype=cdtype, device=dev)
    voice_out.scatter_add_(
        0, cfg.voice_idx.unsqueeze(1).expand(D, T), out,
    )

    # ── Update state ──────────────────────────────────────────────────────
    new_state = DriverState(
        phase_acc=phase[:, :, -1],                                         # (D, H)
        t_pos=state.t_pos + T / sr,                                        # (D,)
    )

    return out, voice_out, new_state


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter utilities
# ═══════════════════════════════════════════════════════════════════════════════

def scatter_to_voices(
    driver_out: Tensor, voice_idx: Tensor, V: int,
) -> Tensor:
    """Sum driver outputs per voice node.  ``(D, T) → (V, T)``."""
    D, T = driver_out.shape
    out = torch.zeros(V, T, dtype=driver_out.dtype, device=driver_out.device)
    out.scatter_add_(0, voice_idx.unsqueeze(1).expand(D, T), driver_out)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FM / AM modulation builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_fm_mod(
    cfg: DriverConfig, voice_signals: Tensor, sr: float,
) -> Optional[Tensor]:
    """Build ``(D, T)`` FM modulation from resolved voice-level signals.

    Extracts instantaneous frequency from each driver's FM source voice,
    normalises by mean absolute frequency, and scales by ``fm_depth_hz``.
    Returns *None* when no driver has FM wired.
    """
    has_fm = cfg.fm_source_voice >= 0
    if not has_fm.any():
        return None
    D = cfg.D
    T = voice_signals.shape[1]
    dev = cfg.device

    fm = torch.zeros(D, T, dtype=torch.float64, device=dev)
    src_idx = cfg.fm_source_voice[has_fm].clamp(min=0)
    src = voice_signals[src_idx]                                          # (n, T)

    # Instantaneous frequency: angle(z[n] · conj(z[n−1])) · sr / 2π
    pd = torch.angle(src[:, 1:] * src[:, :-1].conj())
    fi = torch.cat([pd[:, :1], pd], dim=1) * (sr / _2PI)
    f_mean = fi.abs().mean(dim=1, keepdim=True).clamp(min=1.0)
    fm[has_fm] = cfg.fm_depth_hz[has_fm].unsqueeze(1) * (fi / f_mean)
    return fm


def build_am_mod(
    cfg: DriverConfig, voice_signals: Tensor,
) -> Optional[Tensor]:
    """Build ``(D, T)`` AM modulation from resolved voice-level signals.

    Uses peak-normalised magnitude envelope of each driver's AM source voice
    scaled by ``am_depth``.  Returns *None* when no driver has AM wired.
    """
    has_am = cfg.am_source_voice >= 0
    if not has_am.any():
        return None
    D = cfg.D
    T = voice_signals.shape[1]
    dev = cfg.device

    am = torch.zeros(D, T, dtype=torch.float64, device=dev)
    src_idx = cfg.am_source_voice[has_am].clamp(min=0)
    mag = voice_signals[src_idx].abs().to(torch.float64)                  # (n, T)
    peak = mag.amax(dim=1, keepdim=True).clamp(min=1e-12)
    am[has_am] = cfg.am_depth[has_am].unsqueeze(1) * (mag / peak)
    return am


# ═══════════════════════════════════════════════════════════════════════════════
# FM / AM topological ordering
# ═══════════════════════════════════════════════════════════════════════════════

def fm_am_topo_levels(cfg: DriverConfig) -> list[list[int]]:
    """Voice-level topological sort for FM and AM dependencies.

    Returns a list of voice-index lists.  Level 0 has no dependencies.
    Raises ``ValueError`` on cycles.
    """
    V = cfg.V
    deps: dict[int, set[int]] = {v: set() for v in range(V)}

    for attr in ("fm_source_voice", "am_source_voice"):
        src_tensor: Tensor = getattr(cfg, attr)
        has = src_tensor >= 0
        if not has.any():
            continue
        s_list = src_tensor[has].tolist()
        d_list = cfg.voice_idx[has].tolist()
        for s, d in zip(s_list, d_list):
            if int(s) != int(d):
                deps[int(d)].add(int(s))

    levels: list[list[int]] = []
    remaining = set(range(V))
    resolved: set[int] = set()

    while remaining:
        level = sorted(v for v in remaining if deps[v].issubset(resolved))
        if not level:
            raise ValueError("FM/AM dependency cycle detected among voices")
        levels.append(level)
        resolved.update(level)
        remaining -= set(level)

    return levels


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-level driver step (handles FM / AM topo ordering)
# ═══════════════════════════════════════════════════════════════════════════════

def multi_level_driver_step(
    cfg: DriverConfig,
    state: DriverState,
    chunk_T: int,
    sr: float,
) -> tuple[Tensor, Tensor, DriverState]:
    """Full driver synthesis with FM/AM topological resolution.

    Voices are partitioned into dependency levels.  Each level's drivers are
    synthesised with FM/AM built from previously-resolved voice signals.

    Returns ``(driver_out, voice_out, new_state)`` — same shape contract as
    ``driver_synthesis_step``.
    """
    levels = fm_am_topo_levels(cfg)

    if len(levels) <= 1:
        # No cross-voice dependencies — single pass
        return driver_synthesis_step(cfg, state, chunk_T, sr)

    D, V, T = cfg.D, cfg.V, chunk_T
    dev = cfg.device
    cdtype = torch.complex128

    all_out = torch.zeros(D, T, dtype=cdtype, device=dev)
    voice_out = torch.zeros(V, T, dtype=cdtype, device=dev)
    final_phase = state.phase_acc.clone()

    for level_voices in levels:
        # FM / AM from already-resolved voice signals
        fm = build_fm_mod(cfg, voice_out, sr)
        am = build_am_mod(cfg, voice_out)

        # Mask: only this level's drivers are active
        level_set = set(level_voices)
        level_mask = torch.tensor(
            [int(cfg.voice_idx[d].item()) in level_set for d in range(D)],
            dtype=torch.bool, device=dev,
        )
        active_this = cfg.active & level_mask

        d_out, v_out, ns = driver_synthesis_step(
            cfg, state, chunk_T, sr,
            fm_mod=fm, am_mod=am, active_override=active_this,
        )

        all_out[level_mask] = d_out[level_mask]
        voice_out += v_out
        final_phase[level_mask] = ns.phase_acc[level_mask]

    new_state = DriverState(
        phase_acc=final_phase,
        t_pos=state.t_pos + T / sr,
    )
    return all_out, voice_out, new_state


