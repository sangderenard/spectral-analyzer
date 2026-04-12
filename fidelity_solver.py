"""Fidelity solver — optimise CQT / FB / CWT parameters for minimum loss.

Supports:
  • Standalone CLI with full report
  • Programmatic API for integration into bass_viewer
  • Per-parameter locks (invariants): any value can be frozen
  • Per-octave schedules: BPO / hop / Q-scale / sigma solved as a vector, not a scalar
  • ScheduleVec: callable per-octave plan implementing OctaveBPOFunc / OctaveHopFunc
  • Dynamic limit discovery — solver explores beyond hardcoded UI ranges
  • CQT engine "algorithm" choice: librosa (pseudoinverse) or nsgt (painless frame)

Parameter modes (per engine, per param):
  Global   — solver optimises one scalar value
  Schedule — solver optimises one value per octave (uses bpo_func / hop_func)
  Locked   — value is fixed, solver never touches it
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution

from torch_cqt_new import (
    _CQT_WINDOWS,
    OctaveBPOFunc,
    OctaveFloatFunc,
    OctaveHopFunc,
    fidelity_curve,
    fidelity_curve_cwt,
    fidelity_curve_fb,
)
from torch_nsgt import fidelity_curve_nsgt

# ═══════════════════════════════════════════════════════════════════════════
# Schedule — per-octave callable plan
# ═══════════════════════════════════════════════════════════════════════════


class ScheduleVec:
    """Per-octave schedule that implements OctaveBPOFunc / OctaveHopFunc.

    Callable as ``schedule(octave_index, base_value) → int | float``.
    Out-of-range indices return ``base_value``, so it degrades gracefully
    when a different octave count is used at inference vs solve time.
    """

    # Params whose schedules should be treated as integers
    _INT_PARAMS: frozenset[str] = frozenset(
        {"bins_per_octave", "hop_length", "bands_per_octave",
         "scales_per_octave"})

    def __init__(self, values: list[float], param_name: str = ""):
        self._values: list[float] = list(values)
        self._int = param_name in self._INT_PARAMS or not values

    def __call__(self, octave_idx: int, base_value: float) -> int | float:
        v = (self._values[octave_idx]
             if 0 <= octave_idx < len(self._values)
             else float(base_value))
        return int(round(v)) if self._int else float(v)

    # ---- introspection ----------------------------------------------------

    @property
    def n_octaves(self) -> int:
        return len(self._values)

    def to_list(self) -> list[float]:
        return list(self._values)

    def scalar_approx(self) -> float:
        """Geometric mean — scalar summary for display / backward compat."""
        if not self._values:
            return 1.0
        pos = [max(v, 1e-9) for v in self._values]
        return math.exp(sum(math.log(v) for v in pos) / len(pos))

    def __repr__(self) -> str:
        return f"ScheduleVec({[round(v,2) for v in self._values]})"

    # ---- serialisation ---------------------------------------------------

    def to_json(self) -> list[float]:
        return self._values

    @classmethod
    def from_json(cls, data: list[float], param_name: str = "") -> "ScheduleVec":
        return cls(data, param_name)


def _n_octaves_for(fmin: float, fmax: float) -> int:
    """Number of octaves spanned by [fmin, fmax]."""
    fmin = max(fmin, 1e-9)
    fmax = max(fmax, fmin * 1.001)
    return max(1, int(math.ceil(math.log2(fmax / fmin))))


# ═══════════════════════════════════════════════════════════════════════════
# Parameter descriptors
# ═══════════════════════════════════════════════════════════════════════════


# Params that support per-octave scheduling, per engine.
# hop_length is excluded from CWT: it is always 1 (continuous analysis)
# and is not exposed in the viewer UI, so scheduling it is meaningless.
SCHEDULABLE: dict[str, frozenset[str]] = {
    "cqt": frozenset({"bins_per_octave", "hop_length", "filter_scale"}),
    "fb":  frozenset({"bands_per_octave", "hop_length"}),
    "cwt": frozenset({"scales_per_octave", "sigma"}),
}


@dataclass
class ParamDesc:
    """Descriptor for a single tuneable parameter."""
    name: str
    engine: str                       # "cqt" | "fb" | "cwt"
    kind: str                         # "int" | "float" | "log_float" | "choice"
    low: float                        # lower bound (or index 0 for choice)
    high: float                       # upper bound (or len-1 for choice)
    default: float                    # initial / unlocked default
    locked: bool = False              # if True, solver won't touch it
    locked_value: float | None = None # explicit lock value (None → use default)
    choices: list[Any] | None = None  # for kind="choice"
    per_octave: bool = False          # if True, solve N values instead of 1
    n_octaves: int = 1                # number of octave slots (used when per_octave)

    @property
    def value(self) -> float:
        if self.locked and self.locked_value is not None:
            return self.locked_value
        return self.default

    # Number of solver variables contributed by this param
    @property
    def n_vars(self) -> int:
        return self.n_octaves if self.per_octave else 1

    def decode(self, x: float) -> Any:
        """Convert a continuous optimiser variable back to native type."""
        if self.kind == "int":
            return int(round(x))
        elif self.kind == "log_float":
            return float(10.0 ** x)
        elif self.kind == "choice":
            idx = int(round(np.clip(x, 0, len(self.choices) - 1)))
            return self.choices[idx]
        return float(x)

    def decode_schedule(self, xs: np.ndarray) -> ScheduleVec:
        """Decode a vector of solver variables into a ScheduleVec."""
        return ScheduleVec([self.decode(float(x)) for x in xs], self.name)

    def encode(self, v: Any) -> float:
        """Convert a native value to continuous optimiser variable."""
        if self.kind == "log_float":
            return math.log10(max(v, 1e-30))
        if self.kind == "choice":
            try:
                return float(self.choices.index(v))
            except ValueError:
                return 0.0
        return float(v)

    def bounds(self) -> tuple[float, float]:
        if self.kind == "log_float":
            return (math.log10(max(self.low, 1e-30)),
                    math.log10(max(self.high, 1e-30)))
        return (self.low, self.high)


# ═══════════════════════════════════════════════════════════════════════════
# Engine configuration — all tuneable params for each engine
# ═══════════════════════════════════════════════════════════════════════════


_FILTER_TYPES = ["Linkwitz-Riley 4", "Butterworth 4", "Butterworth 8"]
_CWT_WAVELETS = ["morlet", "morse", "ricker"]
_CQT_ALGORITHMS = ["librosa", "nsgt"]
_CQT_WINDOW_CHOICES = list(_CQT_WINDOWS)
_DWT_WAVELETS = [
    "haar", "db1", "db2", "db4", "db8", "sym2", "sym4", "coif1", "coif2"
]


def build_param_registry(
    sr: int = 44100,
    engines: list[str] | None = None,
    schedule_params: set[tuple[str, str]] | None = None,
    octave_counts: dict[str, int] | None = None,
) -> list[ParamDesc]:
    """Build the full set of tuneable parameters for requested engines.

    Parameters
    ----------
    schedule_params : set of (engine, param_name)
        Params to solve as per-octave schedules instead of scalars.
        Must be a subset of ``SCHEDULABLE[engine]``.
    octave_counts : dict
        ``{engine: n_octaves}`` — how many octave slots each schedule spans.
        Computed from fmin/fmax if not supplied.
    """
    if engines is None:
        engines = ["cqt", "fb", "cwt", "dwt"]
        if "dwt" in engines:
            params.extend([
                ParamDesc("wavelet", "dwt", "choice",
                          0, len(_DWT_WAVELETS) - 1, 0,
                          choices=_DWT_WAVELETS),
                ParamDesc("levels", "dwt", "int", 1, 12, 6),
            ])
    if schedule_params is None:
        schedule_params = set()
    if octave_counts is None:
        octave_counts = {}

    params: list[ParamDesc] = []
    nyquist = sr / 2.0

    def _sched(engine: str, name: str, p: ParamDesc) -> ParamDesc:
        """Mark param as per-octave if requested."""
        if (engine, name) in schedule_params and name in SCHEDULABLE.get(engine, set()):
            p.per_octave = True
            p.n_octaves = octave_counts.get(engine, 1)
        return p

    if "cqt" in engines:
        params.extend([
            ParamDesc("algorithm", "cqt", "choice",
                      0, len(_CQT_ALGORITHMS) - 1, 0,
                      choices=_CQT_ALGORITHMS),
            _sched("cqt", "bins_per_octave",
                   ParamDesc("bins_per_octave", "cqt", "int", 4, 2400, 48)),
            _sched("cqt", "hop_length",
                   ParamDesc("hop_length", "cqt", "int", 16, 8192, 512)),
            ParamDesc("fmin", "cqt", "log_float",
                      0.5, nyquist * 0.99, 16.35),
            ParamDesc("fmax", "cqt", "log_float",
                      20.0, nyquist, min(20000.0, nyquist)),
            _sched("cqt", "filter_scale",
                   ParamDesc("filter_scale", "cqt", "float", 0.1, 10.0, 1.0)),
            ParamDesc("window", "cqt", "choice",
                      0, len(_CQT_WINDOW_CHOICES) - 1, 0,
                      choices=_CQT_WINDOW_CHOICES),
        ])

    if "fb" in engines:
        params.extend([
            _sched("fb", "bands_per_octave",
                   ParamDesc("bands_per_octave", "fb", "int", 1, 96, 12)),
            _sched("fb", "hop_length",
                   ParamDesc("hop_length", "fb", "int", 16, 8192, 512)),
            ParamDesc("fmin", "fb", "log_float",
                      0.5, nyquist * 0.99, 16.35),
            ParamDesc("fmax", "fb", "log_float",
                      20.0, nyquist, min(20000.0, nyquist)),
            ParamDesc("filter_type", "fb", "choice",
                      0, len(_FILTER_TYPES) - 1, 0,
                      choices=_FILTER_TYPES),
        ])

    if "cwt" in engines:
        params.extend([
            _sched("cwt", "scales_per_octave",
                   ParamDesc("scales_per_octave", "cwt", "int", 1, 240, 12)),
            ParamDesc("hop_length", "cwt", "int", 1, 512, 1),
            ParamDesc("fmin", "cwt", "log_float",
                      1e-3, nyquist * 0.5, 0.1),
            ParamDesc("fmax", "cwt", "log_float",
                      1.0, nyquist, sr / 4.0),
            _sched("cwt", "sigma",
                   ParamDesc("sigma", "cwt", "float", 1.0, 60.0, 6.0)),
            ParamDesc("wavelet", "cwt", "choice",
                      0, len(_CWT_WAVELETS) - 1, 0,
                      choices=_CWT_WAVELETS),
        ])

    return params


# ═══════════════════════════════════════════════════════════════════════════
# Solver result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    """Outcome of a solver run."""
    params: dict[str, dict[str, Any]]   # engine → {param: scalar/ScheduleVec}
    loss_total: float                   # aggregate objective
    loss_red: float                     # peak catastrophic
    loss_green: float                   # peak degraded
    loss_blue: float                    # peak inherent
    per_engine: dict[str, dict]         # engine → fidelity_curve result
    locked: dict[str, dict[str, Any]]   # engine → {param: locked_value}
    bounds_used: dict[str, dict[str, tuple[float, float]]]  # actual bounds
    schedules: dict[str, dict[str, ScheduleVec]] = field(default_factory=dict)
    # ^ engine → {param_name: ScheduleVec} — only populated for schedule params


# ═══════════════════════════════════════════════════════════════════════════
# Core solver
# ═══════════════════════════════════════════════════════════════════════════


def _decode_vals(
    x: np.ndarray,
    all_params: list[ParamDesc],
) -> dict[str, dict[str, Any]]:
    """Decode solver vector x into {engine: {param: value|ScheduleVec}}."""
    vals: dict[str, dict[str, Any]] = {}
    xi = 0
    for p in all_params:
        eng = p.engine
        if eng not in vals:
            vals[eng] = {}
        if p.locked:
            vals[eng][p.name] = p.decode(p.encode(p.value))
        elif p.per_octave:
            n = p.n_octaves
            vals[eng][p.name] = p.decode_schedule(x[xi: xi + n])
            xi += n
        else:
            vals[eng][p.name] = p.decode(x[xi])
            xi += 1
    return vals


def _extract_funcs(
    ev: dict[str, Any],
) -> tuple[OctaveBPOFunc | None, OctaveHopFunc | None,
           OctaveFloatFunc | None, OctaveFloatFunc | None]:
    """Pull per-octave callables from decoded engine values.

    Returns (bpo_func, hop_func, filter_scale_func, sigma_func).
    Each is None when the parameter was solved as a global scalar.
    """
    bpo_raw = (ev.get("bins_per_octave")
               or ev.get("bands_per_octave")
               or ev.get("scales_per_octave"))
    hop_raw = ev.get("hop_length")
    fs_raw  = ev.get("filter_scale")
    sig_raw = ev.get("sigma")

    bpo_func          = bpo_raw if isinstance(bpo_raw, ScheduleVec) else None
    hop_func          = hop_raw if isinstance(hop_raw, ScheduleVec) else None
    filter_scale_func = fs_raw  if isinstance(fs_raw,  ScheduleVec) else None
    sigma_func        = sig_raw if isinstance(sig_raw,  ScheduleVec) else None
    return bpo_func, hop_func, filter_scale_func, sigma_func


def _scalar(v: Any, default: float) -> float:
    """Extract scalar from possibly-ScheduleVec value."""
    if isinstance(v, ScheduleVec):
        return v.scalar_approx()
    if v is None:
        return default
    return float(v)


def _int_scalar(v: Any, default: int) -> int:
    return max(1, int(round(_scalar(v, float(default)))))


def _evaluate(
    x: np.ndarray,
    free_params: list[ParamDesc],
    all_params: list[ParamDesc],
    sr: int,
    engines: list[str],
    signal_length: int | None = None,
) -> float:
    """Objective function: weighted sum of worst-case loss across engines."""
    vals = _decode_vals(x, all_params)
    total_loss = 0.0


    for eng in engines:
        ev = vals.get(eng, {})
        bpo_func, hop_func, filter_scale_func, sigma_func = _extract_funcs(ev)
        try:
            if eng == "cqt":
                algo = str(ev.get("algorithm", "librosa"))
                if algo == "nsgt":
                    fc = fidelity_curve_nsgt(
                        sr,
                        fmin=_scalar(ev.get("fmin"), 16.35),
                        fmax=_scalar(ev.get("fmax"), 20000.0),
                        bins_per_octave=_int_scalar(ev.get("bins_per_octave"), 48),
                        Ls=signal_length or sr,
                        window=str(ev.get("window", "hann")),
                    )
                    rgb = fc.get("loss_rgb", np.zeros((1, 3)))
                    pr = fc.get("painless")
                    if pr is not None and not pr.is_painless:
                        total_loss += 50.0
                    cond = fc.get("condition_number", 1.0)
                    if cond > 1.0:
                        total_loss += 0.1 * math.log2(max(cond, 1.0))
                else:
                    fc = fidelity_curve(
                        sr,
                        hop_length=_int_scalar(ev.get("hop_length"), 512),
                        fmin=_scalar(ev.get("fmin"), 16.35),
                        fmax=_scalar(ev.get("fmax"), 20000.0),
                        bins_per_octave=_int_scalar(ev.get("bins_per_octave"), 48),
                        filter_scale=_scalar(ev.get("filter_scale"), 1.0),
                        window=str(ev.get("window", "hann")),
                        bpo_func=bpo_func,
                        hop_func=hop_func,
                        filter_scale_func=filter_scale_func,
                    )
                    rgb = fc.get("loss_rgb", np.zeros((1, 3)))
            elif eng == "fb":
                fc = fidelity_curve_fb(
                    sr,
                    bands_per_octave=_int_scalar(ev.get("bands_per_octave"), 12),
                    fmin=_scalar(ev.get("fmin"), 16.35),
                    fmax=_scalar(ev.get("fmax"), 20000.0),
                    hop_length=_int_scalar(ev.get("hop_length"), 512),
                    filter_type=str(ev.get("filter_type", "Linkwitz-Riley 4")),
                    bpo_func=bpo_func,
                    hop_func=hop_func,
                )
                rgb = fc.get("loss_rgb", np.zeros((1, 3)))
            elif eng == "cwt":
                fc = fidelity_curve_cwt(
                    sr,
                    fmin=_scalar(ev.get("fmin"), 0.1),
                    fmax=_scalar(ev.get("fmax"), sr / 4.0),
                    scales_per_octave=_int_scalar(ev.get("scales_per_octave"), 12),
                    hop_length=_int_scalar(ev.get("hop_length"), 1),
                    wavelet=str(ev.get("wavelet", "morlet")),
                    sigma=_scalar(ev.get("sigma"), 6.0),
                    bpo_func=bpo_func,
                    hop_func=hop_func,
                    sigma_func=sigma_func,
                )
                rgb = fc.get("loss_rgb", np.zeros((1, 3)))
            elif eng == "dwt":
                # DWT: no fidelity_curve_dwt yet, so use a placeholder loss
                # TODO: implement real DWT fidelity curve
                # For now, penalize for non-default wavelet/levels as a stub
                wavelet = str(ev.get("wavelet", "db4"))
                levels = _int_scalar(ev.get("levels"), 6)
                # Placeholder: lower loss for longer wavelet support, more levels
                support_len = {
                    "haar": 2, "db1": 2, "db2": 4, "db4": 8, "db8": 16,
                    "sym2": 4, "sym4": 8, "coif1": 6, "coif2": 12
                }.get(wavelet, 8)
                # Lower support_len = worse time resolution, higher levels = more freq resolution
                # Loss is lower for higher levels and longer support
                loss = 1.0 / (levels * support_len)
                rgb = np.array([[loss, loss * 0.5, loss * 0.2]])
            else:
                continue
        except Exception:
            return 1e6  # infeasible config

        if rgb.shape[0] == 0:
            return 1e6

        peak_r = float(rgb[:, 0].max())
        peak_g = float(rgb[:, 1].max())
        peak_b = float(rgb[:, 2].max())
        mean_r = float(rgb[:, 0].mean())
        mean_g = float(rgb[:, 1].mean())
        mean_b = float(rgb[:, 2].mean())

        eng_loss = (10.0 * peak_r + 3.0 * peak_g + 1.0 * peak_b +
                    5.0 * mean_r + 1.5 * mean_g + 0.5 * mean_b)
        total_loss += eng_loss

    return total_loss


def solve(
    sr: int = 44100,
    engines: list[str] | None = None,
    locks: dict[str, dict[str, Any]] | None = None,
    schedule_params: set[tuple[str, str]] | None = None,
    signal_length: int | None = None,
    max_iter: int = 200,
    seed: int | None = 42,
    callback: Callable[[Any], None] | None = None,
) -> SolverResult:
    """Run the fidelity solver.

    Parameters
    ----------
    sr : int
        Sample rate.
    engines : list of str
        Which engines to optimise. Default: all enabled.
    locks : dict
        ``{engine: {param_name: locked_value}}``.
        Locked params are frozen at the given value.
    schedule_params : set of (engine, param_name)
        Params to solve as per-octave schedules.  Must be in
        ``SCHEDULABLE[engine]``.  Mutually exclusive with locked.
    max_iter : int
        Maximum solver iterations (differential_evolution *maxiter*).
    seed : int or None
        RNG seed for reproducibility.
    callback : callable, optional
        Called each generation with the current OptimizeResult.

    Returns
    -------
    SolverResult
    """
    if engines is None:
        engines = ["cqt", "fb", "cwt"]
    if locks is None:
        locks = {}
    if schedule_params is None:
        schedule_params = set()

    # Compute octave count per engine from locked/default fmin+fmax
    octave_counts: dict[str, int] = {}
    # CAFLS defaults: CQT starts at anchor f_a ≈ B0; CWT ends at anchor;
    # FB spans full range (subsonic to Nyquist).
    _cafls_anchor = 30.87  # B0 ≈ f_a
    _fmin_defaults = {"cqt": _cafls_anchor, "fb": 0.1, "cwt": 0.1, "dwt": 16.35}
    _fmax_defaults = {"cqt": min(20000.0, sr / 2.0),
                      "fb": min(20000.0, sr / 2.0),
                      "cwt": _cafls_anchor,
                      "dwt": min(20000.0, sr / 2.0)}
    for eng in engines:
        if eng == "dwt":
            # DWT: octaves not meaningful, set to 1
            octave_counts[eng] = 1
        else:
            fmin = float(locks.get(eng, {}).get("fmin", _fmin_defaults.get(eng, 16.35)))
            fmax = float(locks.get(eng, {}).get("fmax", _fmax_defaults.get(eng, sr / 2.0)))
            octave_counts[eng] = _n_octaves_for(fmin, fmax)

    all_params = build_param_registry(sr, engines,
                                      schedule_params=schedule_params,
                                      octave_counts=octave_counts)

    # Apply locks (schedule params cannot also be locked)
    for p in all_params:
        eng_locks = locks.get(p.engine, {})
        if p.name in eng_locks and not p.per_octave:
            p.locked = True
            p.locked_value = eng_locks[p.name]

    # Build free-parameter bounds (per-octave params contribute n_octaves bounds)
    free_params = [p for p in all_params if not p.locked]
    bounds: list[tuple[float, float]] = []
    for p in free_params:
        b = p.bounds()
        bounds.extend([b] * p.n_vars)

    if not bounds:
        x0 = np.array([])
        best_cost = _evaluate(x0, free_params, all_params, sr, engines, signal_length)
    else:
        result = differential_evolution(
            _evaluate,
            bounds=bounds,
            args=(free_params, all_params, sr, engines, signal_length),
            maxiter=max_iter,
            seed=seed,
            tol=1e-6,
            polish=True,
            callback=callback,
            init="sobol",
        )
        x0 = result.x
        best_cost = result.fun

    # Decode final values
    final_vals = _decode_vals(x0, all_params)

    # Compute final fidelity for each engine
    per_engine: dict[str, dict] = {}
    peak_r = peak_g = peak_b = 0.0

    for eng in engines:
        ev = final_vals.get(eng, {})
        bpo_func, hop_func, filter_scale_func, sigma_func = _extract_funcs(ev)
        try:
            if eng == "cqt":
                algo = str(ev.get("algorithm", "librosa"))
                if algo == "nsgt":
                    fc = fidelity_curve_nsgt(
                        sr,
                        fmin=_scalar(ev.get("fmin"), 16.35),
                        fmax=_scalar(ev.get("fmax"), 20000.0),
                        bins_per_octave=_int_scalar(ev.get("bins_per_octave"), 48),
                        Ls=signal_length or sr,
                        window=str(ev.get("window", "hann")),
                    )
                else:
                    fc = fidelity_curve(
                        sr,
                        hop_length=_int_scalar(ev.get("hop_length"), 512),
                        fmin=_scalar(ev.get("fmin"), 16.35),
                        fmax=_scalar(ev.get("fmax"), 20000.0),
                        bins_per_octave=_int_scalar(ev.get("bins_per_octave"), 48),
                        filter_scale=_scalar(ev.get("filter_scale"), 1.0),
                        window=str(ev.get("window", "hann")),
                        bpo_func=bpo_func,
                        hop_func=hop_func,
                        filter_scale_func=filter_scale_func,
                    )
            elif eng == "fb":
                fc = fidelity_curve_fb(
                    sr,
                    bands_per_octave=_int_scalar(ev.get("bands_per_octave"), 12),
                    fmin=_scalar(ev.get("fmin"), 16.35),
                    fmax=_scalar(ev.get("fmax"), 20000.0),
                    hop_length=_int_scalar(ev.get("hop_length"), 512),
                    filter_type=str(ev.get("filter_type", "Linkwitz-Riley 4")),
                    bpo_func=bpo_func,
                    hop_func=hop_func,
                )
            elif eng == "cwt":
                fc = fidelity_curve_cwt(
                    sr,
                    fmin=_scalar(ev.get("fmin"), 0.1),
                    fmax=_scalar(ev.get("fmax"), sr / 4.0),
                    scales_per_octave=_int_scalar(ev.get("scales_per_octave"), 12),
                    hop_length=_int_scalar(ev.get("hop_length"), 1),
                    wavelet=str(ev.get("wavelet", "morlet")),
                    sigma=_scalar(ev.get("sigma"), 6.0),
                    bpo_func=bpo_func,
                    hop_func=hop_func,
                    sigma_func=sigma_func,
                )
            elif eng == "dwt":
                # DWT: no fidelity_curve_dwt yet, so just record params
                wavelet = str(ev.get("wavelet", "db4"))
                levels = _int_scalar(ev.get("levels"), 6)
                fc = {"wavelet": wavelet, "levels": levels,
                      "loss_rgb": np.array([[0.0, 0.0, 0.0]])}
            else:
                continue
        except Exception:
            fc = {}
        per_engine[eng] = fc
        rgb = fc.get("loss_rgb", np.zeros((1, 3)))
        if rgb.shape[0]:
            peak_r = max(peak_r, float(rgb[:, 0].max()))
            peak_g = max(peak_g, float(rgb[:, 1].max()))
            peak_b = max(peak_b, float(rgb[:, 2].max()))

    # Record locked params
    locked_out: dict[str, dict[str, Any]] = {}
    for p in all_params:
        if p.locked:
            locked_out.setdefault(p.engine, {})[p.name] = p.decode(p.encode(p.value))

    # Record bounds actually used
    bounds_out: dict[str, dict[str, tuple[float, float]]] = {}
    for p in all_params:
        bounds_out.setdefault(p.engine, {})
        if not p.locked:
            bounds_out[p.engine][p.name] = (p.low, p.high)

    # Extract ScheduleVec results
    schedules_out: dict[str, dict[str, ScheduleVec]] = {}
    for eng, ev in final_vals.items():
        for pname, pval in ev.items():
            if isinstance(pval, ScheduleVec):
                schedules_out.setdefault(eng, {})[pname] = pval

    return SolverResult(
        params=final_vals,
        loss_total=best_cost,
        loss_red=peak_r,
        loss_green=peak_g,
        loss_blue=peak_b,
        per_engine=per_engine,
        locked=locked_out,
        bounds_used=bounds_out,
        schedules=schedules_out,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════


def format_report(result: SolverResult) -> str:
    """Human-readable text report of solver results."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  FIDELITY SOLVER REPORT")
    lines.append("=" * 72)
    lines.append("")

    # Overall scores
    lines.append(f"  Total objective:  {result.loss_total:.4f}")
    lines.append(f"  Peak RED   (catastrophic): {result.loss_red:.2%}")
    lines.append(f"  Peak GREEN (degraded):     {result.loss_green:.2%}")
    lines.append(f"  Peak BLUE  (inherent):     {result.loss_blue:.2%}")
    lines.append("")

    for eng, vals in sorted(result.params.items()):
        lines.append(f"  ── {eng.upper()} ──")
        locked_vals = result.locked.get(eng, {})
        sched_vals  = result.schedules.get(eng, {})
        for pname, pval in sorted(vals.items()):
            if pname in locked_vals:
                tag = " [LOCKED]"
            elif pname in sched_vals:
                tag = f" [SCHEDULE ×{sched_vals[pname].n_octaves}]"
            else:
                tag = ""
            if isinstance(pval, ScheduleVec):
                approx = pval.scalar_approx()
                sched_str = " ".join(f"{v:.3g}" for v in pval.to_list())
                lines.append(f"    {pname:24s} ≈ {approx:.4g}  [{sched_str}]{tag}")
            elif isinstance(pval, float):
                lines.append(f"    {pname:24s} = {pval:.6g}{tag}")
            else:
                lines.append(f"    {pname:24s} = {pval}{tag}")

        # Per-engine loss summary
        fc = result.per_engine.get(eng, {})
        rgb = fc.get("loss_rgb", np.zeros((1, 3)))
        if rgb.shape[0] > 0:
            lines.append(f"    peak  R={float(rgb[:,0].max()):.2%}  "
                         f"G={float(rgb[:,1].max()):.2%}  "
                         f"B={float(rgb[:,2].max()):.2%}")
            lines.append(f"    mean  R={float(rgb[:,0].mean()):.2%}  "
                         f"G={float(rgb[:,1].mean()):.2%}  "
                         f"B={float(rgb[:,2].mean()):.2%}")
            freqs = fc.get("freqs", np.array([]))
            if len(freqs):
                lines.append(f"    freq range: {freqs[0]:.3f} — {freqs[-1]:.1f} Hz  "
                             f"({len(freqs)} bins)")
        lines.append("")

    # Dynamic bounds
    if result.bounds_used:
        lines.append("  ── DYNAMIC BOUNDS (solver search space) ──")
        for eng, bdict in sorted(result.bounds_used.items()):
            for pname, (lo, hi) in sorted(bdict.items()):
                lines.append(f"    {eng}.{pname:24s}: [{lo:.6g}, {hi:.6g}]")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, ScheduleVec):
        return {"__schedule__": obj.to_json()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def format_json(result: SolverResult) -> str:
    """Machine-readable JSON report."""
    obj = {
        "loss_total": result.loss_total,
        "loss_red": result.loss_red,
        "loss_green": result.loss_green,
        "loss_blue": result.loss_blue,
        "params": result.params,
        "schedules": {
            eng: {k: v.to_json() for k, v in sdict.items()}
            for eng, sdict in result.schedules.items()
        },
        "locked": result.locked,
        "bounds_used": {
            eng: {k: list(v) for k, v in bdict.items()}
            for eng, bdict in result.bounds_used.items()
        },
    }
    return json.dumps(obj, indent=2, default=_json_default)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fidelity solver — find optimal CQT/FB/CWT parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Lock parameters with --lock ENGINE.PARAM=VALUE, e.g.:
              --lock cqt.fmin=32.7 --lock cwt.sigma=6 --lock fb.filter_type="Linkwitz-Riley 4"
        """),
    )
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate (default: 44100)")
    parser.add_argument("--engines", nargs="+", default=["cqt", "fb", "cwt"],
                        choices=["cqt", "fb", "cwt"],
                        help="Which engines to optimise")
    parser.add_argument("--lock", action="append", default=[],
                        metavar="ENGINE.PARAM=VALUE",
                        help="Lock a parameter (repeatable)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum solver iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of text report")

    args = parser.parse_args(argv)

    # Parse locks
    locks: dict[str, dict[str, Any]] = {}
    for spec in args.lock:
        if "=" not in spec:
            parser.error(f"Lock must be ENGINE.PARAM=VALUE, got: {spec!r}")
        key, val_str = spec.split("=", 1)
        if "." not in key:
            parser.error(f"Lock key must be ENGINE.PARAM, got: {key!r}")
        eng, pname = key.split(".", 1)
        # Try to parse as number
        try:
            val: Any = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str.strip('"').strip("'")
        if eng not in locks:
            locks[eng] = {}
        locks[eng][pname] = val

    import time as _time
    t0 = _time.monotonic()
    gen_count = [0]

    def _cb(xk, convergence=0):
        gen_count[0] += 1
        elapsed = _time.monotonic() - t0
        print(f"  gen {gen_count[0]:3d}  conv={convergence:.6f}  "
              f"elapsed={elapsed:.1f}s", file=sys.stderr)

    result = solve(
        sr=args.sr,
        engines=args.engines,
        locks=locks,
        max_iter=args.max_iter,
        seed=args.seed,
        callback=_cb,
    )
    elapsed = _time.monotonic() - t0

    if args.json:
        print(format_json(result))
    else:
        report = format_report(result)
        print(report)
        print(f"  Solved in {elapsed:.1f}s ({gen_count[0]} generations)")


if __name__ == "__main__":
    main()
