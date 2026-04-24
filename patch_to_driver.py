"""patch_to_driver.py — Bridge: AnalyticPatch → DriverConfig.

Converts the high-level AnalyticPatch (voices, performers, routing) into the
flat tensor config consumed by ``performer_engine.py``.

Typical usage::

    from patch_to_driver import build_driver_config
    from performer_engine import init_driver_state, multi_level_driver_step

    cfg, performers, driver_list = build_driver_config(patch, performers, device, sr)
    state = init_driver_state(cfg)
    # per-chunk synthesis loop:
    driver_out, voice_out, state = multi_level_driver_step(cfg, state, chunk_T, sr)

Driver slots
------------
One slot is created for every (performer, voice) pair where the voice key
appears in ``PerformerPlacement.source_voice_keys``.  Muted voices are skipped.
When a performer has an empty ``source_voice_keys`` list it is expanded to all
non-muted voices (keeps compatibility with simple single-performer patches that
pre-date the placement solver).

If the *performers* list is empty (placement solver has not been run), a single
synthetic performer is created for every non-muted voice, positioned at the
origin with no delays or gain offsets.

Ownership
---------
Performers own instruments 1:1.  Each driver slot bakes in the performer's
timing humanization (geometric_delay_ms, humanization_ms) into
pre_delay_samples.  Performers are state-machine objects and carry no tensor
representation in DriverConfig.

``instrument_idx`` identifies which instrument (excitation point) each slot
belongs to, used by the sympathetic coupling step.

Frequency resolution
--------------------
Each voice's ``note_tracking`` field determines which base frequency is used:

* ``"note"``  — *note_hz* (the currently sounding note, defaults to ``seq_tonic_hz``)
* ``"root"``  — ``patch.seq_tonic_hz`` (tonal centre / scale root)
* ``"free"``  — ``voice.freq_hz`` (voice's own absolute frequency)

``voice.semitone_offset`` is applied after tracking resolution in all cases.

Envelope
--------
``voice.active_knots()`` returns ``[[t_frac, v], …]`` with *t_frac* ∈ [0, 1].
These are converted to absolute seconds by multiplying by *note_duration*
(``patch.duration``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from parametric_curve import ParametricCurve
from performer_engine import (
    DriverConfig,
    CHIRP_NONE,
    CHIRP_LINEAR,
    CHIRP_EXPONENTIAL,
    CHIRP_POWER,
)

# ── Chirp-type string → integer code ─────────────────────────────────────────
_CHIRP_CODE: dict[str, int] = {
    "none":        CHIRP_NONE,
    "linear":      CHIRP_LINEAR,
    "exponential": CHIRP_EXPONENTIAL,
    "power":       CHIRP_POWER,
}

# ── Synthetic performer dataclass (used when performers list is empty) ────────

class _SyntheticPerformer:
    """Minimal stand-in for PerformerPlacement when the placement solver has
    not been run.  Only the fields accessed by build_driver_config are set."""

    __slots__ = (
        "key", "label", "chair_key",
        "source_voice_keys",
        "body_type",
        "x", "y",
        "geometric_delay_ms", "humanization_ms",
        "phase_offset_rad",
        "gain_db",
    )

    def __init__(self, voice_key: str, voice_label: str, body_type: str) -> None:
        self.key               = voice_key
        self.label             = voice_label
        self.chair_key         = "default"
        self.source_voice_keys = [voice_key]
        self.body_type         = body_type
        self.x                 = 0.0
        self.y                 = 0.0
        self.geometric_delay_ms = 0.0
        self.humanization_ms   = 0.0
        self.phase_offset_rad  = 0.0
        self.gain_db           = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_f0(voice, note_hz: float, root_hz: float) -> float:
    """Compute instantaneous f0 from voice tracking mode + semitone offset."""
    semitones = float(voice.semitone_offset)
    tracking  = getattr(voice, "note_tracking", "note")
    if tracking == "free":
        base = float(voice.freq_hz)
    elif tracking == "root":
        base = root_hz
    else:  # "note" (default)
        base = note_hz
    return base * (2.0 ** (semitones / 12.0))


def _build_harmonics(voice) -> tuple[list[float], list[float]]:
    """Return ``(h_ratios, h_amps)`` for *voice*'s manifold type.

    Both lists have length ``max(1, voice.harmonic_count)``.
    ``h_amps`` is pre-normalised so the sum is 1.0 (convention: amplitude=1
    produces peak ≈ 1 when all harmonics are in phase).
    """
    manifold   = getattr(voice, "manifold_type", "pure")
    H          = max(1, int(getattr(voice, "harmonic_count", 1)))
    brightness = max(0.0, float(getattr(voice, "harmonic_brightness", 1.0)))

    if manifold == "pure":
        return [1.0] + [0.0] * (H - 1), [1.0] + [0.0] * (H - 1)

    # Harmonic / harmonic_warp
    warp = float(getattr(voice, "harmonic_warp_strength", 0.0))
    ratios: list[float] = []
    for k in range(1, H + 1):
        if manifold == "harmonic_warp" and warp != 0.0:
            # Stretch: ratio_k = k * (1 + warp*(k-1)/max(H-1,1))
            r = float(k) * (1.0 + warp * (k - 1) / max(H - 1, 1))
        else:
            r = float(k)
        ratios.append(r)

    # Amplitude rolloff: amp_k = 1/k^brightness, then normalise
    raw: list[float] = [
        (1.0 / (k ** brightness) if brightness > 0.0 else 1.0)
        for k in range(1, H + 1)
    ]
    total = sum(raw)
    if total > 0.0:
        raw = [a / total for a in raw]

    return ratios, raw


def _build_env_knots(voice, note_duration: float) -> tuple[list[float], list[float]]:
    """Return ``(env_t_secs, env_v)`` scaled to *note_duration* seconds.

    ``voice.active_knots()`` returns ``[[t_frac, v], …]`` with t_frac ∈ [0,1].
    """
    knots = voice.active_knots()  # [[t_frac, v], …]
    dur   = max(note_duration, 1e-9)
    t_secs = [float(k[0]) * dur for k in knots]
    vals   = [float(k[1])       for k in knots]
    return t_secs, vals


def _build_parametric_driver_batches(
    voices: list,
    driver_list: list[tuple[int, int]],
    device: torch.device,
) -> tuple[object | None, Tensor, object | None, Tensor]:
    """Pack per-driver parametric env/chirp curves for the batched driver engine."""
    env_curves: list[ParametricCurve] = []
    chirp_curves: list[ParametricCurve] = []
    env_row: list[int] = []
    chirp_row: list[int] = []

    for _, vi in driver_list:
        voice = voices[vi]
        piecewise = getattr(voice, "piecewise_env", None)
        if piecewise is not None:
            env_row.append(len(env_curves))
            env_curves.append(piecewise.curve)
            chirp_row.append(len(chirp_curves))
            chirp_curves.append(piecewise.chirp_curve)
        else:
            env_row.append(-1)
            chirp_row.append(-1)

    packed_env = (
        ParametricCurve.pack_batch(env_curves, device=device)
        if env_curves else None
    )
    packed_chirp = (
        ParametricCurve.pack_batch(chirp_curves, device=device)
        if chirp_curves else None
    )
    return (
        packed_env,
        torch.tensor(env_row, dtype=torch.int64, device=device),
        packed_chirp,
        torch.tensor(chirp_row, dtype=torch.int64, device=device),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def build_driver_config(
    patch,
    performers: list,
    device: torch.device,
    sr: float,
    *,
    note_hz: Optional[float] = None,
    muted_voice_keys: Optional[set] = None,
) -> tuple[DriverConfig, list, list]:
    """Convert *patch* + *performers* into a flat DriverConfig tensor bundle.

    Parameters
    ----------
    patch             : AnalyticPatch — source of voices, tuning, and duration.
    performers        : list[PerformerPlacement] — placement solver output.  If
                        empty, one synthetic performer is created per non-muted voice.
    device            : torch target device.
    sr                : sample rate (Hz) — used to convert pre_delay to samples.
    note_hz           : current note frequency (Hz).  Defaults to
                        ``patch.seq_tonic_hz`` when *None*.
    muted_voice_keys  : optional set of voice keys to treat as muted regardless
                        of ``voice.muted``.  Used to honour solo-exclusion logic
                        without mutating voice objects.

    Returns
    -------
    (DriverConfig, performers, driver_list)
    """
    if note_hz is None:
        note_hz = float(patch.seq_tonic_hz)
    root_hz       = float(patch.seq_tonic_hz)
    note_duration = float(patch.duration)
    _muted_keys: set = set(muted_voice_keys) if muted_voice_keys else set()

    # ── Voice index map ──────────────────────────────────────────────────────
    voices: list = list(patch.voices)
    V             = len(voices)
    voice_key_to_idx: dict[str, int] = {v.key: i for i, v in enumerate(voices)}

    def _is_muted(v) -> bool:
        return getattr(v, "muted", False) or v.key in _muted_keys

    # ── Ensure non-empty performer list ─────────────────────────────────────
    if not performers:
        performers = [
            _SyntheticPerformer(
                v.key,
                getattr(v, "label", v.key),
                getattr(v, "body_type", "direct"),
            )
            for v in voices
            if not _is_muted(v)
        ]

    P = len(performers)
    if P == 0 or V == 0:
        return _empty_config(V, device), [], []

    # ── Enumerate driver slots ───────────────────────────────────────────────
    # Each slot is a (performer_index, voice_index) pair.
    driver_list: list[tuple[int, int]] = []
    for p_idx, perf in enumerate(performers):
        vkeys = list(getattr(perf, "source_voice_keys", []))
        if not vkeys:
            # Expand to all non-muted voices
            vkeys = [v.key for v in voices if not getattr(v, "muted", False)]
        for vk in vkeys:
            vi = voice_key_to_idx.get(vk)
            if vi is None:
                continue  # dangling reference — voice removed since placement solve
            if _is_muted(voices[vi]):
                continue
            driver_list.append((p_idx, vi))

    D = len(driver_list)
    if D == 0:
        return _empty_config(V, device), [], []

    packed_env, param_env_row, packed_chirp, param_chirp_row = _build_parametric_driver_batches(
        voices, driver_list, device,
    )

    # ── Determine padded dimensions ──────────────────────────────────────────
    H_per: list[int] = []
    K_per: list[int] = []
    for _, vi in driver_list:
        v  = voices[vi]
        mt = getattr(v, "manifold_type", "pure")
        H_per.append(
            max(1, int(getattr(v, "harmonic_count", 1)))
            if mt in ("harmonic", "harmonic_warp") else 1
        )
        K_per.append(len(v.active_knots()))
    H = max(H_per)
    K = max(K_per)

    # ── Per-driver lists (populated below) ───────────────────────────────────
    f0_list:             list[float] = []
    amplitude_list:      list[float] = []
    phase_origin_list:   list[float] = []
    pre_delay_samp_list: list[int]   = []
    note_dur_list:       list[float] = []
    active_list:         list[bool]  = []
    chirp_type_list:     list[int]   = []
    chirp_fs_list:       list[float] = []
    chirp_fe_list:       list[float] = []
    chirp_tau_list:      list[float] = []
    chirp_pow_list:      list[float] = []
    h_ratios_list:       list[list[float]] = []
    h_amps_list:         list[list[float]] = []
    n_harmonics_list:    list[int]   = []
    env_t_list:          list[list[float]] = []
    env_v_list:          list[list[float]] = []
    env_n_list:          list[int]   = []
    voice_idx_list:      list[int]   = []
    instrument_idx_list: list[int]   = []
    fm_src_list:         list[int]   = []
    fm_depth_list:       list[float] = []
    am_src_list:         list[int]   = []
    am_depth_list:       list[float] = []

    for p_idx, vi in driver_list:
        v    = voices[vi]
        perf = performers[p_idx]

        # ── Frequency ────────────────────────────────────────────────────────
        f0_list.append(_resolve_f0(v, note_hz, root_hz))

        # ── Amplitude (pure voice; performer gain applied at scatter) ────────
        amplitude_list.append(float(v.amplitude))

        # ── Phase origin (per harmonic seed in init_driver_state) ────────────
        phase_origin_list.append(float(v.phase_origin))

        # ── Pre-delay = voice pre_delay + performer geometric + humanization ─
        pre_delay_s = (
            float(getattr(v,    "pre_delay",          0.0))
            + float(getattr(perf, "geometric_delay_ms", 0.0)) * 1e-3
            + float(getattr(perf, "humanization_ms",    0.0)) * 1e-3
        )
        pre_delay_samp_list.append(int(round(pre_delay_s * sr)))

        note_dur_list.append(note_duration)
        active_list.append(True)

        # ── Chirp ─────────────────────────────────────────────────────────────
        chirp = getattr(v, "chirp", None)
        chirp_type_list.append(
            _CHIRP_CODE.get(getattr(chirp, "chirp_type", "none"), CHIRP_NONE)
        )
        chirp_fs_list.append(float(getattr(chirp, "f_delta_start", 0.0)) if chirp else 0.0)
        chirp_fe_list.append(float(getattr(chirp, "f_delta_end",   0.0)) if chirp else 0.0)
        chirp_tau_list.append(float(getattr(chirp, "tau",           0.5)) if chirp else 0.5)
        chirp_pow_list.append(float(getattr(chirp, "chirp_power",   1.0)) if chirp else 1.0)

        # ── Harmonics ─────────────────────────────────────────────────────────
        h_ratios, h_amps = _build_harmonics(v)
        n_h = len(h_ratios)
        h_ratios_list.append(h_ratios + [0.0] * (H - n_h))
        h_amps_list.append(h_amps   + [0.0] * (H - n_h))
        n_harmonics_list.append(n_h)

        # ── Envelope ──────────────────────────────────────────────────────────
        env_t_secs, env_v = _build_env_knots(v, note_duration)
        n_k      = len(env_t_secs)
        last_v   = env_v[-1] if env_v else 0.0
        env_t_list.append(env_t_secs + [1e30]  * (K - n_k))
        env_v_list.append(env_v      + [last_v] * (K - n_k))
        env_n_list.append(n_k)

        # ── Mapping ───────────────────────────────────────────────────────────
        voice_idx_list.append(vi)
        instrument_idx_list.append(p_idx)

        # ── FM wiring ─────────────────────────────────────────────────────────
        fm = getattr(v, "fm", None)
        if fm and getattr(fm, "source_key", ""):
            fm_vi = voice_key_to_idx.get(fm.source_key, -1)
        else:
            fm_vi = -1
        fm_src_list.append(fm_vi)
        fm_depth_list.append(float(fm.depth_hz) if fm else 0.0)

        # ── AM wiring ─────────────────────────────────────────────────────────
        am = getattr(v, "am", None)
        if am and getattr(am, "source_key", ""):
            am_vi = voice_key_to_idx.get(am.source_key, -1)
        else:
            am_vi = -1
        am_src_list.append(am_vi)
        am_depth_list.append(float(am.depth_amp) if am else 0.0)

    # ── Tensor constructors ──────────────────────────────────────────────────
    def _ft(lst):  return torch.tensor(lst, dtype=torch.float64, device=device)
    def _it(lst):  return torch.tensor(lst, dtype=torch.int64,   device=device)
    def _bt(lst):  return torch.tensor(lst, dtype=torch.bool,    device=device)
    def _ft2(lst): return torch.tensor(lst, dtype=torch.float64, device=device)

    cfg = DriverConfig(
        D=D, H=H, K=K, V=V, device=device,
        f0                  = _ft(f0_list),
        amplitude           = _ft(amplitude_list),
        phase_origin        = _ft(phase_origin_list),
        pre_delay_samples   = _it(pre_delay_samp_list),
        note_duration       = _ft(note_dur_list),
        active              = _bt(active_list),
        chirp_type          = _it(chirp_type_list),
        chirp_f_start       = _ft(chirp_fs_list),
        chirp_f_end         = _ft(chirp_fe_list),
        chirp_tau           = _ft(chirp_tau_list),
        chirp_power         = _ft(chirp_pow_list),
        h_ratios            = _ft2(h_ratios_list),
        h_amps              = _ft2(h_amps_list),
        n_harmonics         = _it(n_harmonics_list),
        env_t               = _ft2(env_t_list),
        env_v               = _ft2(env_v_list),
        env_n               = _it(env_n_list),
        voice_idx           = _it(voice_idx_list),
        instrument_idx      = _it(instrument_idx_list),
        fm_source_voice     = _it(fm_src_list),
        fm_depth_hz         = _ft(fm_depth_list),
        am_source_voice     = _it(am_src_list),
        am_depth            = _ft(am_depth_list),
        parametric_env_packed = packed_env,
        parametric_env_row    = param_env_row,
        parametric_chirp_packed = packed_chirp,
        parametric_chirp_row    = param_chirp_row,
    )

    return cfg, performers, driver_list


# ── Empty-config helper ──────────────────────────────────────────────────────

def _empty_config(V: int, device: torch.device) -> DriverConfig:
    """Return a zero-slot DriverConfig when there are no active driver slots."""
    def _ft():   return torch.zeros(0, dtype=torch.float64, device=device)
    def _it():   return torch.zeros(0, dtype=torch.int64,   device=device)
    def _bt():   return torch.zeros(0, dtype=torch.bool,    device=device)
    def _ft2(h): return torch.zeros(0, h, dtype=torch.float64, device=device)

    return DriverConfig(
        D=0, H=1, K=5, V=V, device=device,
        f0=_ft(), amplitude=_ft(), phase_origin=_ft(),
        pre_delay_samples=_it(), note_duration=_ft(), active=_bt(),
        chirp_type=_it(), chirp_f_start=_ft(), chirp_f_end=_ft(),
        chirp_tau=_ft(), chirp_power=_ft(),
        h_ratios=_ft2(1), h_amps=_ft2(1), n_harmonics=_it(),
        env_t=_ft2(5), env_v=_ft2(5), env_n=_it(),
        voice_idx=_it(), instrument_idx=_it(),
        fm_source_voice=_it(), fm_depth_hz=_ft(),
        am_source_voice=_it(), am_depth=_ft(),
        parametric_env_packed=None, parametric_env_row=_it(),
        parametric_chirp_packed=None, parametric_chirp_row=_it(),
    )
