"""Signal processing tools.

*Tools* are pre/post transform operations on time-domain audio.  They differ
from *engines* (CQT, NSGT, STFT — torch_cqt_new.py, torch_nsgt.py) which
perform the irreversible audio-to-spectrogram mapping and are formally
specified, scheduled, and produce persisted artefacts.

Tools are composable, mostly stateless, and sit on either side of an engine:

    audio → [pre-filter] → [heterodyne] → [spline detrend] → ENGINE → spectrogram

Available tools
---------------
HeterodyneShift
    Single-sideband carrier mix.  Shifts a band of interest by moving its
    geometric centre frequency, preserving relative frequency relationships
    within the band.  Used to lift sub-bass / infrasound content into the
    analysis window, or fold high-frequency content down.

SplineDetrender
    Scientifically rigorous spline detrending.  Controlled by a
    ``SplineDetrenderConfig`` that exposes fit target, trend resolution,
    spline order, application mode, edge handling, robust IRLS weighting,
    gain compensation, and continuous blend.

    Key use cases:
    - DC removal          → fit_target="waveform", trend_mode="knots", n_knots=1
    - Linear detrend      → fit_target="waveform", trend_mode="knots", n_knots=2
    - Envelope cleaning   → fit_target="hilbert",  trend_mode="hz",    cutoff_hz=2.0
    - Robust detrend      → robust=True (ignores transients via Tukey IRLS)

Each tool is exposed both as a plain function and as a class that implements
``SignalTool.process(signal, sr) → ndarray``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class SignalTool:
    """Abstract base for all signal processing tools."""

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Heterodyne
# ──────────────────────────────────────────────────────────────────────────────

def heterodyne_shift(
    signal: np.ndarray,
    sr: int,
    *,
    freq_lo: float,
    freq_hi: float,
    scale: float,
) -> np.ndarray:
    """Shift a band by moving its geometric centre via analytic heterodyning.

    Single-sideband (SSB) mix: only the positive-frequency half is shifted,
    so no mirror sideband is created and there is no carrier tone in the
    output.

    Parameters
    ----------
    signal  : 1-D float audio array.
    sr      : sample rate (Hz).
    freq_lo : lower edge of the band being shifted (Hz).
    freq_hi : upper edge of the band being shifted (Hz).
    scale   : frequency scale factor.  1.0 = no shift, 2.0 = one octave up.
    """
    from scipy.signal import hilbert

    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0 or sr <= 0:
        return np.zeros(0, dtype=np.float64)
    scale = max(float(scale), 1e-9)
    lo = max(0.0, float(freq_lo))
    hi = max(lo + 1e-6, float(freq_hi))
    src_center = math.sqrt(lo * hi) if lo > 0.0 else 0.5 * hi
    target_center = src_center * scale
    shift_hz = target_center - src_center
    if abs(shift_hz) < 1e-9:
        return x.copy()
    analytic = hilbert(x)
    t = np.arange(x.size, dtype=np.float64) / float(sr)
    carrier = np.exp(1j * 2.0 * math.pi * shift_hz * t)
    return np.asarray(np.real(analytic * carrier), dtype=np.float64)


class HeterodyneShift(SignalTool):
    """Object form of :func:`heterodyne_shift`."""

    def __init__(self, *, freq_lo: float, freq_hi: float, scale: float) -> None:
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.scale = scale

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        return heterodyne_shift(
            signal, sr,
            freq_lo=self.freq_lo,
            freq_hi=self.freq_hi,
            scale=self.scale,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Spline detrender — configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SplineDetrenderConfig:
    """Full configuration for the spline detrender.

    Fit targets
    -----------
    ``"waveform"``
        Fit directly to raw signal samples.  Removes DC offset and very slow
        baseline drift.  1 knot = pure mean subtraction.  2 knots = linear
        detrend (removes tilt + offset).

    ``"rms_env"``
        Fit to the windowed RMS energy envelope.  Tracks how loud the signal
        is over time.  Good for removing amplitude drift.

    ``"hilbert_env"``
        Fit to the instantaneous amplitude |analytic(x)|.  The most precise
        amplitude envelope; higher computational cost.

    ``"peak_env"``
        Fit to the windowed peak absolute value.  Tracks the peak excursion.

    ``"median_env"``
        Fit to the windowed median absolute value.  Robust to transients;
        useful when impulses should not steer the trend.

    Trend control modes
    -------------------
    ``"knots"``
        Directly set the number of interior knots via ``n_knots``.
        1 knot = mean removal (DC block), 2 = linear detrend.
        Uses LSQUnivariateSpline for exact knot placement.

    ``"hz"``
        Set the effective lowpass cutoff in Hz via ``cutoff_hz``.
        Converted to knots by the Nyquist rule: n_knots = ⌈2·fc·T⌉.
        Most intuitive for audio work.

    ``"seg"``
        One knot every ``segment_sec`` seconds.  Intuitive for specifying
        how slowly the trend is allowed to vary.

    ``"smooth"``
        Raw scipy ``UnivariateSpline`` smoothing factor ``s``.  Controls the
        fit as a trade-off between smoothness and closeness.  Useful when the
        other modes don't give enough fine control, but requires calibration.

    Application modes
    -----------------
    ``"subtract"``  detrended = signal − trend
    ``"divide"``    detrended = signal / max(trend, soft_floor)
    ``"normalize"`` divide then rescale to original RMS (preserves loudness)

    Edge handling
    -------------
    ``"natural"``  scipy default: spline extrapolates beyond data.
    ``"clamped"``  Force zero slope at both boundaries by duplicating the
                   boundary value just outside the signal.
    ``"mirror"``   Reflect signal at both ends before fitting; crop after.
                   Suppresses Gibbs-like ringing at boundaries.

    Robust fitting
    --------------
    When ``robust=True``, IRLS (iteratively reweighted least squares) with
    Tukey's biweight M-estimator is applied.  Samples whose residual exceeds
    ~4.7 × MAD are progressively down-weighted, so transients and onsets do
    not distort the trend estimate.
    """

    # -- Fit target --
    fit_target: str = "waveform"
    # "waveform" | "rms_env" | "hilbert_env" | "peak_env" | "median_env"

    # -- Trend control --
    trend_mode: str = "knots"
    # "knots" | "hz" | "seg" | "smooth"

    n_knots: int = 2             # interior knots (trend_mode="knots")
    cutoff_hz: float = 0.5       # effective LP cutoff (trend_mode="hz")
    segment_sec: float = 2.0     # knot spacing in seconds (trend_mode="seg")
    smoothing: float = 1e5       # UnivariateSpline s (trend_mode="smooth")

    # -- Polynomial order --
    spline_order: int = 3        # 1 = linear, 3 = cubic, 5 = quintic

    # -- Application mode --
    mode: str = "subtract"       # "subtract" | "divide" | "normalize"

    # -- Edge handling --
    edge: str = "natural"        # "natural" | "clamped" | "mirror"

    # -- Robust IRLS --
    robust: bool = False
    robust_iters: int = 4        # IRLS iterations
    robust_c: float = 4.685      # Tukey biweight constant (× 1.4826 × MAD)

    # -- Post-processing --
    gain_comp: bool = False      # restore original RMS after detrend
    blend: float = 1.0           # 0.0 = bypass, 1.0 = full detrend
    soft_floor: float = 1e-6     # minimum absolute divisor (divide/normalize)

    # -- Envelope estimation (rms_env / peak_env / median_env) --
    env_window_ms: float = 50.0  # window length in ms
    env_hop_frac: float = 0.25   # hop as fraction of window (0.25 = 75% overlap)

    # -- Decimation for waveform/hilbert fits --
    max_fit_points: int = 8192

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SplineDetrenderConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ──────────────────────────────────────────────────────────────────────────────
# Spline detrender — internals
# ──────────────────────────────────────────────────────────────────────────────

def _n_knots_from_config(cfg: SplineDetrenderConfig, n_samples: int, sr: int) -> int:
    """Resolve cfg.trend_mode to an integer knot count."""
    if cfg.trend_mode == "knots":
        return max(1, int(cfg.n_knots))
    duration = n_samples / max(sr, 1)
    if cfg.trend_mode == "hz":
        # Nyquist: n_knots = ceil(2 · fc · T)
        return max(1, math.ceil(2.0 * cfg.cutoff_hz * duration))
    if cfg.trend_mode == "seg":
        return max(1, math.ceil(duration / max(cfg.segment_sec, 1e-6)))
    # "smooth" → not used for LSQ; caller checks trend_mode separately
    return max(1, int(cfg.n_knots))


def _estimate_envelope(
    x: np.ndarray,
    sr: int,
    cfg: SplineDetrenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_env, y_env) in normalised time [0,1] for envelope fit targets."""
    n = len(x)
    win_n = max(4, int(cfg.env_window_ms * sr / 1000.0))
    hop_n = max(1, int(win_n * cfg.env_hop_frac))

    centers: list[float] = []
    values: list[float] = []
    i = 0
    while i + win_n <= n:
        frame = x[i: i + win_n]
        center = (i + win_n / 2.0) / n          # normalised position
        if cfg.fit_target == "rms_env":
            val = math.sqrt(float(np.mean(frame ** 2)))
        elif cfg.fit_target == "peak_env":
            val = float(np.max(np.abs(frame)))
        else:  # "median_env"
            val = float(np.median(np.abs(frame)))
        centers.append(center)
        values.append(val)
        i += hop_n

    # Ensure we have a point near the right boundary
    if not centers or centers[-1] < 0.99:
        tail = x[max(0, n - win_n):]
        if cfg.fit_target == "rms_env":
            val = math.sqrt(float(np.mean(tail ** 2)))
        elif cfg.fit_target == "peak_env":
            val = float(np.max(np.abs(tail)))
        else:
            val = float(np.median(np.abs(tail)))
        centers.append(1.0)
        values.append(val)

    return np.array(centers, dtype=np.float64), np.array(values, dtype=np.float64)


def _apply_edge(
    t: np.ndarray,
    y: np.ndarray,
    cfg: SplineDetrenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend (t, y) according to edge handling before spline fitting."""
    if cfg.edge == "natural":
        return t, y

    dt = t[1] - t[0] if len(t) > 1 else 1e-4
    pad = max(1, len(t) // 10)

    if cfg.edge == "clamped":
        # Duplicate boundary values just outside [0,1] to force near-zero slope.
        t_l = np.array([t[0] - 2 * dt, t[0] - dt])
        y_l = np.array([y[0], y[0]])
        t_r = np.array([t[-1] + dt, t[-1] + 2 * dt])
        y_r = np.array([y[-1], y[-1]])
        return (np.concatenate([t_l, t, t_r]),
                np.concatenate([y_l, y, y_r]))

    if cfg.edge == "mirror":
        # Reflect signal at both ends.
        n_pad = min(pad, len(t) - 1)
        t_l = t[0] - (t[1: n_pad + 1] - t[0])[::-1]
        y_l = y[1: n_pad + 1][::-1]
        t_r = t[-1] + (t[-1] - t[-(n_pad + 1): -1])[::-1]
        y_r = y[-(n_pad + 1): -1][::-1]
        return (np.concatenate([t_l, t, t_r]),
                np.concatenate([y_l, y, y_r]))

    return t, y


def _fit_spline(
    t_fit: np.ndarray,
    y_fit: np.ndarray,
    cfg: SplineDetrenderConfig,
    n_samples: int,
    sr: int,
) -> "Any":
    """Fit a spline to (t_fit, y_fit) and return a callable f(t) → value."""
    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

    k = max(1, min(5, cfg.spline_order))

    # Need at least k+1 points to fit order-k spline.
    if len(t_fit) < k + 2:
        # Degenerate: return a constant (mean).
        mean_val = float(np.mean(y_fit)) if len(y_fit) > 0 else 0.0
        return lambda t: np.full_like(t, mean_val, dtype=np.float64)

    def _make_spl(t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
        if cfg.trend_mode == "smooth":
            kw = {"s": float(cfg.smoothing)}
            if w is not None:
                kw["w"] = w
            return UnivariateSpline(t, y, k=k, **kw)
        else:
            # LSQ with evenly spaced interior knots in the interior of t.
            n_k = _n_knots_from_config(cfg, n_samples, sr)
            n_k = min(n_k, max(1, len(t) - k - 1))  # scipy constraint
            knots = np.linspace(t[0], t[-1], n_k + 2)[1:-1]
            # Ensure knots are strictly interior to data range.
            eps = (t[-1] - t[0]) * 1e-6
            knots = knots[(knots > t[0] + eps) & (knots < t[-1] - eps)]
            if len(knots) == 0:
                # Fallback: smoothing spline.
                return UnivariateSpline(t, y, k=k, s=float(cfg.smoothing),
                                       w=w)
            try:
                return LSQUnivariateSpline(t, y, knots, k=k, w=w)
            except Exception:
                return UnivariateSpline(t, y, k=k, s=float(cfg.smoothing), w=w)

    if not cfg.robust:
        return _make_spl(t_fit, y_fit)

    # IRLS with Tukey's biweight.
    weights = np.ones(len(t_fit), dtype=np.float64)
    spl = _make_spl(t_fit, y_fit, weights)
    for _ in range(max(1, cfg.robust_iters)):
        residuals = np.abs(y_fit - spl(t_fit))
        mad = float(np.median(residuals))
        scale = cfg.robust_c * 1.4826 * (mad + 1e-12)
        r = residuals / scale
        weights = np.where(r < 1.0, (1.0 - r ** 2) ** 2, 0.0)
        weights = np.maximum(weights, 1e-6)
        spl = _make_spl(t_fit, y_fit, weights)
    return spl


# ──────────────────────────────────────────────────────────────────────────────
# Spline detrender — public API
# ──────────────────────────────────────────────────────────────────────────────

def spline_detrend(
    signal: np.ndarray,
    sr: int,
    config: SplineDetrenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and remove a slowly-varying trend from *signal*.

    Parameters
    ----------
    signal : 1-D float audio array.
    sr     : sample rate (Hz).
    config : :class:`SplineDetrenderConfig` controlling all aspects of the
             estimation and removal.

    Returns
    -------
    detrended : signal with trend removed (float64, same length as input).
    trend     : estimated slow-varying baseline (float64, same length).
                For envelope fit targets this is the full-resolution spline
                envelope (not the raw windowed values).
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 4:
        return x.copy(), np.zeros(n, dtype=np.float64)

    original_rms = math.sqrt(float(np.mean(x ** 2))) if config.gain_comp else 0.0

    # ── 1.  Build the series the spline is fitted to ─────────────────────────
    t_full = np.linspace(0.0, 1.0, n)

    if config.fit_target == "waveform":
        step = max(1, n // config.max_fit_points)
        t_fit_raw = t_full[::step]
        y_fit_raw = x[::step]

    elif config.fit_target == "hilbert_env":
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(x))
        step = max(1, n // config.max_fit_points)
        t_fit_raw = t_full[::step]
        y_fit_raw = envelope[::step]

    else:
        # rms_env, peak_env, median_env
        t_fit_raw, y_fit_raw = _estimate_envelope(x, sr, config)

    # ── 2.  Apply edge handling ───────────────────────────────────────────────
    t_fit, y_fit = _apply_edge(t_fit_raw, y_fit_raw, config)

    # ── 3.  Fit spline ────────────────────────────────────────────────────────
    spl = _fit_spline(t_fit, y_fit, config, n, sr)

    # ── 4.  Evaluate trend at every sample ───────────────────────────────────
    trend = np.asarray(spl(t_full), dtype=np.float64)

    # ── 5.  Apply mode ────────────────────────────────────────────────────────
    sf = max(float(config.soft_floor), 1e-15)
    if config.mode == "divide":
        denom = np.where(np.abs(trend) >= sf, trend, np.sign(trend) * sf)
        denom = np.where(denom == 0.0, sf, denom)
        detrended = x / denom
    elif config.mode == "normalize":
        denom = np.where(np.abs(trend) >= sf, trend, np.sign(trend) * sf)
        denom = np.where(denom == 0.0, sf, denom)
        detrended = x / denom
        d_rms = math.sqrt(float(np.mean(detrended ** 2)))
        if d_rms > 1e-12:
            detrended *= original_rms / d_rms
    else:  # "subtract"
        detrended = x - trend

    # ── 6.  Optional gain compensation ───────────────────────────────────────
    if config.gain_comp and config.mode == "subtract":
        d_rms = math.sqrt(float(np.mean(detrended ** 2)))
        if d_rms > 1e-12:
            detrended = detrended * (original_rms / d_rms)

    # ── 7.  Blend ─────────────────────────────────────────────────────────────
    blend = max(0.0, min(1.0, float(config.blend)))
    if blend < 1.0:
        detrended = blend * detrended + (1.0 - blend) * x

    return detrended.astype(np.float64), trend.astype(np.float64)


class SplineDetrender(SignalTool):
    """Object form of :func:`spline_detrend`.

    Pass a :class:`SplineDetrenderConfig` to control all aspects of the
    estimation.
    """

    def __init__(self, config: SplineDetrenderConfig | None = None) -> None:
        self.config = config or SplineDetrenderConfig()

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        detrended, _ = spline_detrend(signal, sr, self.config)
        return detrended
