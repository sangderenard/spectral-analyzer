#!/usr/bin/env python3
"""
Musical Spectral Analyzer — CQT edition

What it does
------------
- Loads a WAV or MP3 file (mono or stereo)
- Computes a Constant-Q Transform (CQT): every bin is exactly 1 cent wide
  (1/100 semitone, 1/1200 octave), giving identical frequency resolution in
  musical terms at every pitch across the full spectrum.
- The CQT window length scales inversely with frequency, so low notes get
  long windows (fine pitch / coarser time) and high notes get short windows
  (coarser pitch / finer time) — matching the physics of musical sound.
- All three spectrogram images share the same CQT matrix; views are just
  frequency-range slices with different display-row densities for readability.
- Produces:
    cqt_data.npz            — compressed CQT tensors + metadata
    frame_metrics.csv / detected_bulges.csv / summary.txt

  Run bass_plot.py on the output directory to render spectrogram images.

Dependencies
------------
pip install numpy scipy librosa torch
pip install miniaudio   # MP3 support, no ffmpeg needed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks
from shard_budget import ShardBudget, ShardHandle
from analysis_itinerary import (
    AnalysisArtifact,
    AnalysisDatasetRecord,
    AnalysisInventory,
    AnalysisTimeRange,
)

try:
    import soundfile as _sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False
    from scipy.io import wavfile


EPS = 1e-12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _parse_schedule_arg(raw: str | None, *, integer: bool) -> list[int] | list[float] | None:
    """Parse a JSON list CLI schedule argument."""
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("schedule arguments must be JSON lists")
    if integer:
        return [int(round(float(v))) for v in data]
    return [float(v) for v in data]


def _schedule_hashable(values: Sequence[int | float] | None) -> list[int | float] | None:
    if values is None:
        return None
    return [float(v) if isinstance(v, float) else int(v) for v in values]


# ---------------------------------------------------------------------------
# Settings token — human-readable, decodable fingerprint of the parameters
# that shape the CQT output.  Format:
#
#   {algo}-h{hop}-b{bpo}-q{q}-f{fmin_x100}-x{fmax_int}-w{win}-c{cd}s{sd}[-{crc4}]
#
# The CRC suffix is only appended when non-default optional params are present
# (schedules, heterodyne, bandpass, composite, trim_pad) so the common case
# stays short and the full canonical dict still provides uniqueness guarantees.
#
# decode_settings_token() reverses the main fields without needing any stored
# settings file — the token IS the settings summary for reconstruction.
# ---------------------------------------------------------------------------

_ALGO_ENCODE: dict[str, str] = {
    "librosa": "lbr", "nsgt": "nsg", "stft": "stf",
}
_ALGO_DECODE: dict[str, str] = {v: k for k, v in _ALGO_ENCODE.items()}

_WIN_ENCODE: dict[str, str] = {
    "hann": "hn", "blackman": "bk", "blackmanharris": "bh",
    "hamming": "hm", "boxcar": "bx", "nuttall": "nu",
    "flattop": "ft", "bartlett": "bt", "cosine": "cs",
}
_WIN_DECODE: dict[str, str] = {v: k for k, v in _WIN_ENCODE.items()}

_DTYPE_BITS: dict[str, str] = {"float16": "16", "float32": "32", "float64": "64"}
_DTYPE_FROM_BITS: dict[str, str] = {v: k for k, v in _DTYPE_BITS.items()}


def _settings_canonical(
    cfg: "AnalysisConfig",
    composite_spec: str | None = None,
    gap_seconds: float = 0.5,
    trim_pad: bool = False,
) -> dict:
    """Return the canonical dict of all transform-affecting settings."""
    canonical: dict = {
        "fft_algorithm": cfg.fft_algorithm,
        "hop_length": cfg.hop_length,
        "stft_n_fft": cfg.stft_n_fft,
        "bins_per_octave": cfg.bins_per_octave,
        "cqt_filter_scale": round(cfg.cqt_filter_scale, 4),
        "cqt_window": cfg.cqt_window,
        "cqt_bpo_schedule": _schedule_hashable(cfg.cqt_bpo_schedule),
        "cqt_hop_schedule": _schedule_hashable(cfg.cqt_hop_schedule),
        "cqt_filter_scale_schedule": _schedule_hashable(cfg.cqt_filter_scale_schedule),
        "cqt_fmin": round(cfg.cqt_fmin, 4),
        "cqt_fmax": round(cfg.cqt_fmax, 4),
        "bass_min": round(cfg.bass_min, 4),
        "bass_max": round(cfg.bass_max, 4),
        "kick_min": round(cfg.kick_min, 4),
        "kick_max": round(cfg.kick_max, 4),
        "bulge_min": round(cfg.bulge_min, 4),
        "bulge_max": round(cfg.bulge_max, 4),
        "edge_threshold_db": round(cfg.edge_threshold_db, 4),
        "peak_prominence_db": round(cfg.peak_prominence_db, 4),
        "envelope_smooth_ms": round(cfg.envelope_smooth_ms, 4),
        "cqt_compute_dtype": cfg.cqt_compute_dtype,
        "cqt_save_dtype": cfg.cqt_save_dtype,
    }
    if composite_spec:
        canonical["composite"] = composite_spec
        canonical["gap_seconds"] = round(gap_seconds, 4)
    if trim_pad:
        canonical["trim_pad"] = True
    if cfg.prefilter_bp_fmin is not None:
        canonical["prefilter_bp_fmin"] = round(cfg.prefilter_bp_fmin, 4)
    if cfg.prefilter_bp_fmax is not None:
        canonical["prefilter_bp_fmax"] = round(cfg.prefilter_bp_fmax, 4)
    if cfg.heterodyne_hz != 0.0:
        canonical["heterodyne_hz"] = round(cfg.heterodyne_hz, 6)
    if cfg.postfilter_bp_fmin is not None:
        canonical["postfilter_bp_fmin"] = round(cfg.postfilter_bp_fmin, 4)
    if cfg.postfilter_bp_fmax is not None:
        canonical["postfilter_bp_fmax"] = round(cfg.postfilter_bp_fmax, 4)
    if cfg.spline_detrend is not None:
        canonical["spline_detrend"] = cfg.spline_detrend.to_dict()
    return canonical


def settings_token(
    cfg: "AnalysisConfig",
    composite_spec: str | None = None,
    gap_seconds: float = 0.5,
    trim_pad: bool = False,
) -> str:
    """Return a human-readable, decodable token identifying the analysis settings.

    The token encodes the core transform parameters directly in its text so
    they can be recovered without a stored settings file.  A 4-char CRC suffix
    is added only when non-default optional parameters (schedules, heterodyne,
    bandpass, composite, trim_pad) are present.

    Examples
    --------
    Default settings:
        lbr-h512-b1200-q1000-f1635-x20000-whn-c64s32
    Custom hop + filter scale + fmin:
        lbr-h256-b1200-q800-f2000-x20000-whn-c32s32
    With heterodyne (extras → CRC appended):
        lbr-h512-b1200-q1000-f1635-x20000-whn-c64s32-3a7f
    """
    algo = _ALGO_ENCODE.get(cfg.fft_algorithm, cfg.fft_algorithm[:3])
    hop  = cfg.hop_length
    bpo  = cfg.bins_per_octave
    q    = int(round(cfg.cqt_filter_scale * 1000))
    fmin = int(round(cfg.cqt_fmin * 100))
    fmax = int(round(cfg.cqt_fmax))
    win  = _WIN_ENCODE.get(cfg.cqt_window, cfg.cqt_window[:2])
    cd   = _DTYPE_BITS.get(cfg.cqt_compute_dtype, cfg.cqt_compute_dtype)
    sd   = _DTYPE_BITS.get(cfg.cqt_save_dtype, cfg.cqt_save_dtype)

    token = f"{algo}-h{hop}-b{bpo}-q{q}-f{fmin}-x{fmax}-w{win}-c{cd}s{sd}"

    has_extras = bool(
        cfg.cqt_bpo_schedule or cfg.cqt_hop_schedule or
        cfg.cqt_filter_scale_schedule or cfg.heterodyne_hz != 0.0 or
        cfg.prefilter_bp_fmin is not None or cfg.prefilter_bp_fmax is not None or
        cfg.postfilter_bp_fmin is not None or cfg.postfilter_bp_fmax is not None or
        cfg.spline_detrend is not None or
        composite_spec or trim_pad or cfg.stft_n_fft is not None
    )
    if has_extras:
        canonical = _settings_canonical(cfg, composite_spec, gap_seconds, trim_pad)
        blob = json.dumps(canonical, sort_keys=True).encode()
        crc = hashlib.sha256(blob).hexdigest()[:4]
        token = f"{token}-{crc}"

    return token


def decode_settings_token(token: str) -> dict:
    """Decode a settings token back to its encoded parameter values.

    Returns a dict with whatever fields the token encodes.  Fields controlled
    by the CRC (schedules, heterodyne, bandpass, composite) are not recoverable
    from the token alone — the dict will be missing them.  The returned values
    are sufficient to reconstruct the CQT grid geometry (sr aside).

    Raises ValueError for tokens that cannot be parsed.
    """
    parts = token.split("-")
    if len(parts) < 8:
        raise ValueError(f"token too short to decode: {token!r}")

    result: dict = {}
    algo_code = parts[0]
    result["fft_algorithm"] = _ALGO_DECODE.get(algo_code, algo_code)

    for part in parts[1:]:
        if not part:
            continue
        if part.startswith("h") and part[1:].isdigit():
            result["hop_length"] = int(part[1:])
        elif part.startswith("b") and part[1:].isdigit():
            result["bins_per_octave"] = int(part[1:])
        elif part.startswith("q") and part[1:].isdigit():
            result["cqt_filter_scale"] = int(part[1:]) / 1000.0
        elif part.startswith("f") and part[1:].isdigit():
            result["cqt_fmin"] = int(part[1:]) / 100.0
        elif part.startswith("x") and part[1:].isdigit():
            result["cqt_fmax"] = float(part[1:])
        elif part.startswith("w"):
            result["cqt_window"] = _WIN_DECODE.get(part[1:], part[1:])
        elif part.startswith("c") and "s" in part[1:]:
            cd, sd = part[1:].split("s", 1)
            result["cqt_compute_dtype"] = _DTYPE_FROM_BITS.get(cd, f"float{cd}")
            result["cqt_save_dtype"]    = _DTYPE_FROM_BITS.get(sd, f"float{sd}")
        # 4-char hex CRC suffix — informational only, not decoded

    return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    fft_algorithm: str = "librosa"  # "librosa" | "nsgt" | "stft"
    hop_length: int = 512           # CQT hop in samples (~11.6 ms at 44.1 kHz)
    stft_n_fft: int | None = None   # explicit STFT FFT size when fft_algorithm='stft'
    bins_per_octave: int = 1200     # 1 cent per bin
    cqt_filter_scale: float = 1.0   # librosa/torch CQT filter_scale (Q multiplier)
    cqt_window: str = "hann"
    cqt_bpo_schedule: list[int] | None = None
    cqt_hop_schedule: list[int] | None = None
    cqt_filter_scale_schedule: list[float] | None = None
    cqt_grid_hop_length: int | None = None  # derived common frame step for scheduled CQT
    cqt_fmin: float = 16.35        # lowest CQT bin — C0, lowest piano/organ note
    cqt_fmax: float = 20000.0      # highest CQT bin (capped at Nyquist)
    # compute_dtype controls the transform math itself.  Save dtype is separate
    # and only affects how the result is encoded on disk.
    cqt_compute_dtype: str = "float64"   # "float32" or "float64"
    # save_dtype: on-disk dtype for CQT real/imag arrays.  Analysis always runs
    # at cfg.cqt_compute_dtype; this only controls the encoding written to disk.
    # float16 requires a stored scale factor (cqt_scale) so magnitudes are
    # exactly restored on load.  float32 is the default; float64 for archival.
    cqt_save_dtype: str = "float32"      # "float32", "float16", or "float64"
    # fmin can go as low as ~1 Hz without breaking librosa; the only hard
    # constraint is fmin > 0 and the signal must contain at least a few
    # periods of the lowest frequency (at 5 Hz that's just 0.2 s of audio).
    # 0 Hz is undefined (log2(0)).  Values below ~10 Hz are infrasound —
    # real physical energy, not audible but present in electronic music.
    bass_view_max_freq: float = 250.0
    treble_view_min_freq: float = 2000.0
    bass_min: float = 20.0
    bass_max: float = 180.0
    kick_min: float = 35.0
    kick_max: float = 140.0
    bulge_min: float = 30.0
    bulge_max: float = 120.0
    edge_threshold_db: float = -50.0
    peak_prominence_db: float = 1.5
    envelope_smooth_ms: float = 80.0
    image_dpi: int = 160
    seconds_per_inch: float = 10.0  # time axis density — 10 s/in gives readable overview
    color_gamma: float = 2.0        # perceptual gamma on normalised dB before colormap:
                                    # > 1 pushes noise floor toward black, strong
                                    # fundamentals/harmonics dominate the colour range.
                                    # < 1 expands quiet detail at the cost of noise.
    resample_taps: int = 64         # anti-alias filter length for octave decimation
    # Pre-analysis bandpass filter.  Applied to the raw audio (after region
    # slice / compositing) before the heterodyne mix.  None = no filtering on
    # that edge.
    prefilter_bp_fmin: float | None = None
    prefilter_bp_fmax: float | None = None
    # Heterodyne carrier (Hz, signed).  The audio is mixed with
    #   cos(2π * heterodyne_hz * t)
    # before the transform.  Positive → content shifts UP (brings seismic /
    # subsonic energy into the analysis window); negative → shifts DOWN (folds
    # high-frequency content into a lower analysis band).  0.0 = off.
    # Display axis: physical_freq = analysis_freq − heterodyne_hz
    heterodyne_hz: float = 0.0
    # Post-heterodyne bandpass filter.  Applied after the carrier mix and
    # before the CQT/STFT transform.  Useful to reject the DSB mirror image.
    # None = no filtering on that edge.
    postfilter_bp_fmin: float | None = None
    postfilter_bp_fmax: float | None = None
    # Pre/post filter implementation back-end.
    # "fir" → frequency-domain Linkwitz-Riley via filter_engine (zero-phase, HQ default).
    # "iir" → time-domain Butterworth via sosfilt (causal streaming, fast).
    filter_mode: str = "fir"
    # Spline detrender applied after the heterodyne mix.  None = disabled.
    # Configured via SplineDetrenderConfig from signal_tools.
    spline_detrend: "SplineDetrenderConfig | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a WAV or MP3 file using a Constant-Q Transform with 1-cent frequency resolution."
    )
    parser.add_argument("--shash", type=str, default=None, help="Settings hash for STFT streaming (internal use)")
    parser.add_argument("wav_path", help="Path to input audio file (.wav or .mp3)")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--hop-length", type=int, default=512,
                        help="CQT hop length in samples (default 512)")
    parser.add_argument("--stft-n-fft", type=int, default=None,
                        help="Explicit STFT FFT size when --fft-algorithm=stft. "
                             "Must be a power of two; bypasses legacy CQT-derived sizing.")
    parser.add_argument("--fft-algorithm", type=str, default="librosa",
                        choices=["librosa", "nsgt", "stft"],
                        help="FFT-engine algorithm label used for versioning "
                             "and on-disk bookkeeping.")
    parser.add_argument("--bins-per-octave", type=int, default=1200,
                        help="CQT bins per octave; 1200 = 1 cent/bin (default 1200)")
    parser.add_argument("--cqt-filter-scale", type=float, default=1.0,
                        help="CQT filter_scale / Q multiplier (default 1.0).")
    parser.add_argument("--cqt-window", type=str, default="hann",
                        help="CQT window family (hann, hamming, blackman, blackmanharris).")
    parser.add_argument("--cqt-bpo-schedule", type=str, default=None,
                        help="JSON list of per-octave CQT bins-per-octave values.")
    parser.add_argument("--cqt-hop-schedule", type=str, default=None,
                        help="JSON list of per-octave CQT decimated hop values.")
    parser.add_argument("--cqt-filter-scale-schedule", type=str, default=None,
                        help="JSON list of per-octave CQT filter_scale values.")
    parser.add_argument("--cqt-fmin", type=float, default=16.35,
                        help="Lowest CQT frequency in Hz (default 16.35 = C0). "
                             "Can go as low as ~1 Hz; 0 is undefined. "
                             "Try 5 for infrasonic sub-bass, 16.35 for full musical range.")
    parser.add_argument("--cqt-fmax", type=float, default=20000.0)
    parser.add_argument("--bass-min", type=float, default=20.0)
    parser.add_argument("--bass-max", type=float, default=180.0)
    parser.add_argument("--kick-min", type=float, default=35.0)
    parser.add_argument("--kick-max", type=float, default=140.0)
    parser.add_argument("--bulge-min", type=float, default=30.0)
    parser.add_argument("--bulge-max", type=float, default=120.0)
    parser.add_argument("--bass-view-max-freq", type=float, default=250.0)
    parser.add_argument("--treble-view-min-freq", type=float, default=2000.0)
    parser.add_argument("--edge-threshold-db", type=float, default=-50.0,
                        help="Per-frame threshold for edge frequency tracking (dB re frame peak).")
    parser.add_argument("--peak-prominence-db", type=float, default=1.5,
                        help="Prominence for bulge peak detection.")
    parser.add_argument("--envelope-smooth-ms", type=float, default=80.0,
                        help="Smoothing window for low-end envelope (ms).")
    parser.add_argument("--seconds-per-inch", type=float, default=10.0,
                        help="Time axis density in seconds per inch (default 10.0). "
                             "Figure width is set automatically from audio duration. "
                             "Lower = more zoomed in, higher = more compressed.")
    parser.add_argument("--color-gamma", type=float, default=2.0,
                        help="Perceptual gamma applied to normalised dB before colormap "
                             "(default 2.0). > 1 pushes noise floor to black and lets "
                             "strong fundamentals/harmonics dominate; < 1 lifts quiet detail.")
    parser.add_argument("--composite", type=str, default=None,
                        help="Composite spec: extract multiple time regions with padding "
                             "between them.  Format: 'pad:start:end:pad[;...]' where pad is "
                             "z(zero)/r(reflect)/n(none) and times are seconds or L-X "
                             "(length minus X).  Use 'default' for z:15:25:z;z:L-25:L-15:z.")
    parser.add_argument("--gap-seconds", type=float, default=0.5,
                        help="Duration of each half-gap between composite regions in seconds "
                             "(default 0.5). Full visible gap is twice this value.")
    parser.add_argument("--trim-pad", action="store_true", default=False,
                        help="Trim padding regions from the edges of spectrogram images "
                             "(only affects display, not analysis).")
    parser.add_argument("--resample-taps", type=int, default=64,
                        help="Anti-alias filter length for octave decimation (default 64).")
    parser.add_argument("--cqt-precision", type=str, default=None,
                        choices=["64x32", "64x16", "32x16", "64x64"],
                        help="CQT precision shorthand: <compute_bits>x<save_bits>. "
                             "64x32 = float64 compute, float32 save. "
                             "64x16 = float64 compute, float16 save (~50%% smaller). "
                             "32x16 = float32 compute (half VRAM), float16 save. "
                             "64x64 = float64 compute and save (archival).")
    parser.add_argument("--cqt-compute-precision", type=str, default=None,
                        choices=["32", "64"],
                        help="CQT compute precision in bits. "
                             "Affects transform math only, not NPZ encoding. Default 64.")
    parser.add_argument("--cqt-save-precision", type=str, default=None,
                        choices=["16", "32", "64"],
                        help="CQT save precision in bits. "
                             "Affects NPZ encoding only. Default 32.")
    parser.add_argument("--start-time", type=float, default=None,
                        help="Start time in seconds for partial analysis. "
                             "If omitted, analysis starts from the beginning.")
    parser.add_argument("--end-time", type=float, default=None,
                        help="End time in seconds for partial analysis. "
                             "If omitted, analysis runs to the end.")
    parser.add_argument("--prefilter-bp-fmin", type=float, default=None,
                        help="Bandpass pre-filter lower cutoff (Hz). "
                             "Applied to audio before the transform. "
                             "Omit for no high-pass filtering.")
    parser.add_argument("--prefilter-bp-fmax", type=float, default=None,
                        help="Bandpass pre-filter upper cutoff (Hz). "
                             "Applied to audio before the transform. "
                             "Omit for no low-pass filtering.")
    parser.add_argument("--postfilter-bp-fmin", type=float, default=None,
                        help="Post-heterodyne bandpass filter lower cutoff (Hz). "
                             "Applied after the heterodyne mix, before the transform. "
                             "Omit for no high-pass post-filtering.")
    parser.add_argument("--postfilter-bp-fmax", type=float, default=None,
                        help="Post-heterodyne bandpass filter upper cutoff (Hz). "
                             "Applied after the heterodyne mix, before the transform. "
                             "Omit for no low-pass post-filtering.")
    parser.add_argument("--heterodyne-hz", type=float, default=0.0,
                        help="Heterodyne carrier frequency (Hz, signed). "
                             "Positive: mix signal UP (seismic/subsonic → analysis range). "
                             "Negative: mix signal DOWN (high-freq → lower analysis range). "
                             "Physical freq = analysis_freq − heterodyne_hz. Default 0 (off).")
    parser.add_argument("--filter-mode", type=str, default="fir",
                        choices=["fir", "iir"],
                        help="Filter implementation: fir (freq-domain LR, zero-phase, HQ) or "
                             "iir (Butterworth sosfilt, causal streaming, fast). Default: fir.")
    # --- Spline detrender ---
    parser.add_argument("--spline-detrend", action="store_true", default=False,
                        help="Enable spline detrender after heterodyne.")
    parser.add_argument("--spline-fit-target", type=str, default="waveform",
                        choices=["waveform", "rms_env", "hilbert_env", "peak_env", "median_env"],
                        help="What to fit the spline to (default: waveform).")
    parser.add_argument("--spline-trend-mode", type=str, default="knots",
                        choices=["knots", "hz", "seg", "smooth"],
                        help="How to specify trend coarseness (default: knots).")
    parser.add_argument("--spline-n-knots", type=int, default=2,
                        help="Interior knot count (trend-mode=knots). 1=DC removal, 2=linear.")
    parser.add_argument("--spline-cutoff-hz", type=float, default=0.5,
                        help="Equivalent LP cutoff Hz (trend-mode=hz).")
    parser.add_argument("--spline-segment-sec", type=float, default=2.0,
                        help="Knot spacing in seconds (trend-mode=seg).")
    parser.add_argument("--spline-smoothing", type=float, default=1e5,
                        help="UnivariateSpline s factor (trend-mode=smooth).")
    parser.add_argument("--spline-order", type=int, default=3,
                        choices=[1, 3, 5],
                        help="Spline polynomial order: 1=linear, 3=cubic, 5=quintic.")
    parser.add_argument("--spline-mode", type=str, default="subtract",
                        choices=["subtract", "divide", "normalize"],
                        help="subtract/divide/normalize (default: subtract).")
    parser.add_argument("--spline-edge", type=str, default="natural",
                        choices=["natural", "clamped", "mirror"],
                        help="Edge handling (default: natural).")
    parser.add_argument("--spline-robust", action="store_true", default=False,
                        help="IRLS robust fitting (Tukey biweight).")
    parser.add_argument("--spline-robust-iters", type=int, default=4,
                        help="IRLS iteration count (default: 4).")
    parser.add_argument("--spline-gain-comp", action="store_true", default=False,
                        help="Restore original RMS after detrend.")
    parser.add_argument("--spline-blend", type=float, default=1.0,
                        help="Blend 0=bypass..1=full detrend (default: 1.0).")
    parser.add_argument("--spline-soft-floor", type=float, default=1e-6,
                        help="Minimum divisor for divide/normalize modes.")
    parser.add_argument("--spline-env-window-ms", type=float, default=50.0,
                        help="Window length ms for envelope fit targets.")
    return parser.parse_args()


def _build_spline_config(args: "argparse.Namespace") -> "SplineDetrenderConfig | None":
    """Construct a SplineDetrenderConfig from parsed CLI args, or None if disabled."""
    if not args.spline_detrend:
        return None
    from signal_tools import SplineDetrenderConfig
    return SplineDetrenderConfig(
        fit_target=args.spline_fit_target,
        trend_mode=args.spline_trend_mode,
        n_knots=args.spline_n_knots,
        cutoff_hz=args.spline_cutoff_hz,
        segment_sec=args.spline_segment_sec,
        smoothing=args.spline_smoothing,
        spline_order=args.spline_order,
        mode=args.spline_mode,
        edge=args.spline_edge,
        robust=args.spline_robust,
        robust_iters=args.spline_robust_iters,
        gain_comp=args.spline_gain_comp,
        blend=args.spline_blend,
        soft_floor=args.spline_soft_floor,
        env_window_ms=args.spline_env_window_ms,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ram_available_mb() -> float:
    """Return available system RAM in MB.  Returns inf when psutil is absent."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return float("inf")


def _backpressure_wait(
    label: str = "",
    floor_mb: float | None = None,
    interval_s: float = 0.05,
) -> None:
    """Block until system RAM is above *floor_mb*.

    The floor defaults to SPECTRAL_RAM_FLOOR_MB (env, default 512 MB).
    Runs a tight loop sleeping *interval_s* seconds between checks.
    No-op when psutil is unavailable.

    Call this before any allocation that could grow unbounded:
    - before each sosfilt call in the onset loop
    - before each shard write in _save_cqt_stream_bundle
    - before allocating any large intermediate array
    """
    import time as _time
    if floor_mb is None:
        floor_mb = max(
            0.0,
            float(os.environ.get("SPECTRAL_RAM_FLOOR_MB", "512") or 512))
    if floor_mb <= 0:
        return
    avail = _ram_available_mb()
    if avail == float("inf") or avail >= floor_mb:
        return
    tag = f" [{label}]" if label else ""
    print(f"\r  RAM backpressure{tag}: {avail:.0f} MB free < {floor_mb:.0f} MB floor — waiting",
          end="", flush=True)
    import gc as _gc
    while True:
        _gc.collect()
        avail = _ram_available_mb()
        if avail >= floor_mb:
            print()  # newline after the waiting message
            return
        _time.sleep(interval_s)


def _stream_write_npy(path: str, arr: np.ndarray) -> None:
    """Write *arr* to .npy in row-chunks with RAM backpressure.

    Chunks are sized by SPECTRAL_STREAM_WRITE_CHUNK_MB (default 64 MB).
    Before each chunk write we check available RAM against SPECTRAL_RAM_FLOOR_MB
    and block until headroom is restored.  This lets the disk I/O pipeline
    naturally throttle computation — if writes are slow, RAM stays bounded.

    On Windows, open_memmap(mode="w+") raises OSError/EINVAL when the target
    file already exists and is still mapped (e.g. from a previous failed run).
    We therefore write into a sibling .tmp file, fully close the memmap, then
    atomically replace the target so the final file is never partially written.
    """
    arr_np = np.asarray(arr)
    tmp_path = path + ".tmp"
    # Remove any leftover temp file from a previous crash.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    mm = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=arr_np.dtype, shape=arr_np.shape)
    if arr_np.ndim < 2 or arr_np.shape[0] <= 1:
        mm[...] = arr_np
        mm.flush()
        del mm
        os.replace(tmp_path, path)
        return

    target_mb = max(
        4,
        int(os.environ.get("SPECTRAL_STREAM_WRITE_CHUNK_MB", "64") or 64),
    )
    row_bytes = int(np.prod(arr_np.shape[1:])) * arr_np.dtype.itemsize
    rows_per_chunk = max(1, (target_mb * 1024 * 1024) // max(row_bytes, 1))
    n_rows = arr_np.shape[0]
    for r0 in range(0, n_rows, rows_per_chunk):
        _backpressure_wait(label=os.path.basename(path))
        r1 = min(n_rows, r0 + rows_per_chunk)
        mm[r0:r1, ...] = arr_np[r0:r1, ...]
        mm.flush()
    del mm
    # On Windows, os.replace fails with PermissionError if any process still
    # holds a handle to the destination file (e.g. a lingering memmap).  Try
    # the atomic replace first; if denied, remove the target explicitly then
    # rename — the tiny window of non-atomicity is acceptable here.
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        import gc as _gc
        _gc.collect()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        os.rename(tmp_path, path)


def _save_cqt_stream_bundle(
    outdir: str,
    shash: str,
    save_dict: dict[str, np.ndarray],
    pre_sharded: "frozenset[str] | None" = None,
) -> tuple[str, str]:
    """Persist transform payload as permanent .npy shards + JSON manifest.

    Arrays whose keys appear in *pre_sharded* are already on disk at their
    canonical path inside the stream directory (written directly during
    computation) and will be recorded in the manifest without being re-written.

    All other arrays are written via _stream_write_npy which:
      - chunks by SPECTRAL_STREAM_WRITE_CHUNK_MB (default 64 MB)
      - blocks between chunks when RAM < SPECTRAL_RAM_FLOOR_MB
    """
    stream_dir_name = f"cqt_data_{shash}.stream"
    stream_dir = os.path.join(outdir, stream_dir_name)
    os.makedirs(stream_dir, exist_ok=True)
    arrays: dict[str, str] = {}
    # Register pre-sharded arrays in the manifest without re-writing them.
    if pre_sharded:
        for key in sorted(pre_sharded):
            fname = f"{key}.npy"
            if os.path.isfile(os.path.join(stream_dir, fname)):
                arrays[key] = fname
    for key, val in save_dict.items():
        if pre_sharded and key in pre_sharded:
            continue  # already on disk at the canonical permanent path
        _backpressure_wait(label=key)
        fname = f"{key}.npy"
        _stream_write_npy(os.path.join(stream_dir, fname), np.asarray(val))
        arrays[key] = fname
    manifest_name = f"cqt_data_{shash}.stream.json"
    manifest_path = os.path.join(outdir, manifest_name)
    manifest = {
        "format": "stream_v1",
        "settings_hash": shash,
        "stream_dir": stream_dir_name,
        "arrays": arrays,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_name, stream_dir


# ---------------------------------------------------------------------------
# Resume detection — skip recomputing stages whose shard files exist
# ---------------------------------------------------------------------------

# Keys that must be present for a valid CQT resume (before onset).
_CQT_RESUME_KEYS = [
    "freqs", "times", "real_left", "imag_left",
    "sr", "bins_per_octave", "hop_length", "cqt_grid_hop_length", "is_stereo",
]


def _stream_dir_for(outdir: str, shash: str) -> str:
    return os.path.join(outdir, f"cqt_data_{shash}.stream")


def _alloc_shards(
    stream_dir: str,
    n_bins: int,
    n_frames: int,
    dtype: np.dtype,
    *,
    channels: "tuple[str, ...]" = ("left",),
) -> "dict[str, np.memmap]":
    """Pre-allocate permanent memmap shard files for real+imag per channel.

    Must be called before any compute function.  The returned dict maps
    e.g. ``"real_left"`` and ``"imag_left"`` to open writable np.memmap
    objects backed by their canonical .npy files in *stream_dir*.

    Compute functions receive this dict and write directly into these
    memmaps as output arrives — no intermediate RAM accumulation.
    """
    os.makedirs(stream_dir, exist_ok=True)
    shards: dict[str, np.memmap] = {}
    for ch in channels:
        for part in ("real", "imag"):
            key = f"{part}_{ch}"
            path = os.path.join(stream_dir, f"{key}.npy")
            try:
                shards[key] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=dtype, shape=(n_bins, n_frames))
            except OSError as e:
                try:
                    usage = shutil.disk_usage(os.path.abspath(stream_dir))
                    print(f"[DiskError] Cannot allocate shard {key}: {e}")
                    print(f"  Path: {path}")
                    print(f"  Size needed: {n_bins * n_frames * dtype.itemsize / 1024 / 1024:.1f} MB")
                    print(f"  Free space: {usage.free/1024/1024/1024:.2f} GB")
                except Exception:
                    pass
                raise
    return shards


class _ShardPipeline:
    """Double-buffered GPU-compute → memmap shard pipeline.

    Two buffer slots (0/1) alternate: the CUDA compute stream fills one, the
    CUDA copy stream drains it into pinned RAM, and a background I/O thread
    writes pinned RAM to the shard memmaps.  The HDD -> CPU -> VRAM -> CPU ->
    HDD conveyor stays intact while slot reuse provides backpressure.

    Backpressure is structural: a slot must be returned by the I/O thread
    before the GPU can reuse it, bounding peak pinned RAM to 2 × one chunk.

    Parameters
    ----------
    n_fft, hop, n_bins, n_frames : STFT geometry
    max_chunk   : initial maximum frames per chunk (resized on OOM)
    win_t       : window tensor already resident on *device*
    device      : torch.device
    torch_dtype : torch.float32 or torch.float64
    real_mm, imag_mm : writable np.memmap shards, shape (≥n_bins, ≥n_frames)
    """

    def __init__(
        self,
        n_fft: int,
        hop: int,
        n_bins: int,
        bin_lo: int,
        bin_hi: int,
        n_frames: int,
        max_chunk: int,
        win_t: "torch.Tensor",
        device: "torch.device",
        torch_dtype: "torch.dtype",
        real_mm: "np.memmap",
        imag_mm: "np.memmap",
    ) -> None:
        import queue as _q, threading as _t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = n_bins
        self.bin_lo = bin_lo
        self.bin_hi = bin_hi
        self.n_frames = n_frames
        self.max_chunk = max_chunk
        self.win_t = win_t
        self.device = device
        self.torch_dtype = torch_dtype
        self.real_mm = real_mm
        self.imag_mm = imag_mm
        self._err: "BaseException | None" = None
        self._use_cuda = device.type == "cuda"
        self._cplx = (torch.complex128 if torch_dtype == torch.float64
                      else torch.complex64)
        self._real_np = np.float64 if torch_dtype == torch.float64 else np.float32
        self._real_torch = torch.float64 if torch_dtype == torch.float64 else torch.float32

        # Two CUDA streams: one for compute kernels, one for DMA transfers.
        if self._use_cuda:
            self._cs = torch.cuda.Stream(device=device)
            self._ds = torch.cuda.Stream(device=device)
        else:
            self._cs = self._ds = None

        # Slot pool and I/O queue.  _free_q holds slot indices ready for GPU;
        # _write_q holds (slot, n_frames, frame_offset) pending disk write.
        self._free_q: "queue.Queue[int]" = _q.Queue()
        self._write_q: "queue.Queue" = _q.Queue()
        self._thread_cls = _t.Thread

        # GPU/CPU buffers and CUDA events — allocated on first use.
        self._gpu: "list[dict]" = []
        self._cpu: "list[tuple]" = []
        self._events: "list" = []
        self._inflight: "list[object | None]" = [None, None]
        self._io_thread: "_t.Thread | None" = None

        self._alloc(max_chunk)

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _alloc(self, max_chunk: int) -> None:
        """Allocate/reallocate all GPU and CPU buffers for *max_chunk* frames."""
        # Generous input headroom: max signal slice + padding on both sides.
        max_in = max_chunk * self.hop + self.n_fft + 4096
        self._gpu = [
            {
                "x":  torch.empty(max_in, device=self.device, dtype=self.torch_dtype),
            }
            for _ in range(2)
        ]
        self._cpu = [
            (
                torch.empty((self.n_bins, max_chunk), dtype=self._real_torch,
                            pin_memory=self._use_cuda),
                torch.empty((self.n_bins, max_chunk), dtype=self._real_torch,
                            pin_memory=self._use_cuda),
            )
            for _ in range(2)
        ]
        self._events = (
            [torch.cuda.Event() for _ in range(2)] if self._use_cuda
            else [None, None]
        )
        self._inflight = [None, None]
        # Repopulate slot pool.
        while not self._free_q.empty():
            try: self._free_q.get_nowait()
            except Exception: pass
        for i in range(2):
            self._free_q.put(i)
        self._start_io()

    # ------------------------------------------------------------------
    # I/O thread lifecycle
    # ------------------------------------------------------------------

    def _start_io(self) -> None:
        self._io_thread = self._thread_cls(
            target=self._io_worker, daemon=True, name="shard-io")
        self._io_thread.start()

    def _stop_io(self) -> None:
        """Send sentinel and join — all queued writes complete before return."""
        if self._io_thread and self._io_thread.is_alive():
            self._write_q.put(None)
            self._io_thread.join()

    def _io_worker(self) -> None:
        while True:
            item = self._write_q.get()
            if item is None:
                return
            slot, n, off = item
            try:
                ev = self._events[slot]
                if ev is not None:
                    ev.synchronize()           # wait for DMA completion
                pr, pi = self._cpu[slot]
                self.real_mm[:self.n_bins, off:off + n] = pr[:, :n].numpy()
                self.imag_mm[:self.n_bins, off:off + n] = pi[:, :n].numpy()
            except Exception as exc:
                self._err = exc
            finally:
                self._inflight[slot] = None
                self._free_q.put(slot)         # return slot to free pool

    # ------------------------------------------------------------------
    # Per-chunk compute and DMA
    # ------------------------------------------------------------------

    def _compute(self, slot: int, np_slice: "np.ndarray",
                 pad_left: int, pad_right: int, pad_mode: str,
                 chunk_size: int, torchcqt_pad: "callable") -> "tuple[int, torch.Tensor]":
        """Run STFT for one chunk into GPU buffer *slot*."""
        import contextlib
        g = self._gpu[slot]
        ctx = (torch.cuda.stream(self._cs) if self._use_cuda
               else contextlib.nullcontext())
        with ctx:
            if pad_left > 0 or pad_right > 0:
                _t = torchcqt_pad(
                    torch.as_tensor(np_slice, dtype=self.torch_dtype),
                    pad_left, pad_right, mode=pad_mode)
                _l = int(_t.shape[-1])
                g["x"][:_l].copy_(_t, non_blocking=True)
                del _t
            else:
                _l = int(np_slice.shape[-1])
                g["x"][:_l].copy_(
                    torch.as_tensor(np_slice, dtype=self.torch_dtype),
                    non_blocking=True)
            spec = torch.stft(
                g["x"][:_l],
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=self.win_t,
                center=False,
                onesided=True,
                return_complex=True,
            )
            actual = min(int(spec.shape[-1]), chunk_size)
            if actual != int(spec.shape[-1]):
                spec = spec[:, :actual]
        return actual, spec

    def _kick_dma(self, slot: int, spec: "torch.Tensor", actual: int) -> None:
        """Async D2H: copy complex STFT lanes into pinned CPU buffers."""
        import contextlib
        pr, pi = self._cpu[slot]
        ctx = (torch.cuda.stream(self._ds) if self._use_cuda
               else contextlib.nullcontext())
        with ctx:
            if self._use_cuda:
                # Copy stream must wait for compute stream to finish writing spec.
                self._ds.wait_stream(self._cs)
            self._inflight[slot] = spec
            pr[:, :actual].copy_(
                spec.real[self.bin_lo:self.bin_hi, :actual], non_blocking=True)
            pi[:, :actual].copy_(
                spec.imag[self.bin_lo:self.bin_hi, :actual], non_blocking=True)
            ev = self._events[slot]
            if ev is not None:
                ev.record(self._ds)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, x_arr: "np.ndarray", pad: int, pad_mode: str,
            torchcqt_pad: "callable") -> None:
        """Stream the full STFT of *x_arr* into the shard memmaps."""
        import gc, contextlib
        total_len = int(x_arr.shape[-1])
        start_frame = 0

        while start_frame < self.n_frames:
            if self._err:
                raise self._err
            chunk_size = min(self.max_chunk, self.n_frames - start_frame)
            chunk_start = start_frame * self.hop - pad
            chunk_end   = chunk_start + chunk_size * self.hop + self.n_fft - self.hop
            pad_left  = max(0, -chunk_start)
            pad_right = max(0, chunk_end - total_len)
            np_slice  = x_arr[..., max(0, chunk_start):min(total_len, chunk_end)]

            # Acquire a free slot — blocks if both are in-flight (backpressure).
            slot = self._free_q.get()
            try:
                actual, spec = self._compute(
                    slot, np_slice, pad_left, pad_right,
                    pad_mode, chunk_size, torchcqt_pad)
                self._kick_dma(slot, spec, actual)
            except (RuntimeError, MemoryError) as e:
                self._inflight[slot] = None
                self._free_q.put(slot)          # return slot before any re-raise
                is_oom = (isinstance(e, MemoryError)
                          or "out of memory" in str(e).lower())
                if not is_oom or self.max_chunk <= 1:
                    raise
                # OOM: no write is queued yet — safe to drain and restart.
                self._stop_io()
                gc.collect()
                if self._use_cuda:
                    torch.cuda.empty_cache()
                self.max_chunk = max(1, self.max_chunk // 2)
                print(f"[STFT OOM] Reduced to {self.max_chunk} frames, "
                      "reallocating buffers...")
                self._alloc(self.max_chunk)
                continue                        # retry same chunk

            self._write_q.put((slot, actual, start_frame))
            start_frame += actual

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Wait for all in-flight writes, then fsync the memmaps."""
        self._stop_io()
        if self._err:
            raise self._err
        self.real_mm.flush()
        self.imag_mm.flush()

    def __enter__(self) -> "_ShardPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()


def _shard_exists(stream_dir: str, key: str) -> bool:
    p = os.path.join(stream_dir, f"{key}.npy")
    return os.path.isfile(p) and os.path.getsize(p) > 0


def _resume_stage(outdir: str, shash: str) -> str:
    """Return resume stage for the given output directory and settings hash.

    Possible return values
    ----------------------
    "none"       — no prior analysis; run everything from scratch.
    "save_only"  — CQT shards exist but bundle manifest was never written
                   (crash during _save_cqt_stream_bundle).  Skip compute;
                   only redo the save step.
    "cqt"        — CQT shards exist and manifest is present; can skip
                   audio → CQT, start from onset.
    "full"       — All shards including onset exist; skip to metrics/reporting.
    """
    sd = _stream_dir_for(outdir, shash)
    if not os.path.isdir(sd):
        return "none"

    # Detect stereo from the existing shard if possible.
    is_stereo_shard = os.path.join(sd, "is_stereo.npy")
    if _shard_exists(sd, "is_stereo"):
        is_stereo = bool(np.load(is_stereo_shard))
    else:
        is_stereo = os.path.isfile(os.path.join(sd, "real_right.npy"))

    required_cqt = list(_CQT_RESUME_KEYS)
    if is_stereo:
        required_cqt += ["real_right", "imag_right"]

    if not all(_shard_exists(sd, k) for k in required_cqt):
        # Metadata shards are missing.  If the raw CQT arrays are present the
        # run crashed after compute but before _save_cqt_stream_bundle wrote
        # the scalar/vector metadata.  The grid can be reconstructed from the
        # array shape + settings, so signal a special stage rather than "none".
        has_raw = _shard_exists(sd, "real_left") and _shard_exists(sd, "imag_left")
        return "shards_raw" if has_raw else "none"

    # CQT shards are all present.  Check whether the bundle manifest was
    # written.  If not, the previous run crashed during _save_cqt_stream_bundle
    # and we only need to redo that step ("save_only").
    manifest_json = os.path.join(outdir, f"cqt_data_{shash}.stream.json")
    if not os.path.isfile(manifest_json):
        return "save_only"

    return "cqt"


def _load_shard(stream_dir: str, key: str) -> np.ndarray:
    return np.load(os.path.join(stream_dir, f"{key}.npy"), allow_pickle=False)


def _reconstruct_stream_metadata(
    stream_dir: str,
    cfg: "AnalysisConfig",
    sr: int,
    is_stereo: bool,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reconstruct and write missing metadata shards from raw CQT arrays.

    Called when a stream dir has real_left/imag_left but no freqs/times/sr/etc.
    (i.e. ``_resume_stage`` returned ``"shards_raw"``).

    Approach: the raw array shape ``(n_bins, n_frames)`` is self-describing.
    freqs is reconstructed from ``cfg.cqt_fmin``, ``cfg.bins_per_octave`` and
    ``n_bins``.  times is reconstructed from ``grid_hop``, ``sr`` and
    ``n_frames``.  All derived .npy files are written idempotently (existing
    files are never overwritten) so the function is safe to call even when
    some metadata was already written.

    Returns (freqs, times) as numpy arrays, suitable for downstream use.
    """
    import librosa as _lib

    raw_path = os.path.join(stream_dir, "real_left.npy")
    mm = np.load(raw_path, mmap_mode="r")
    n_bins, n_frames = mm.shape
    del mm  # release mmap immediately — we only needed the shape

    grid_hop = int(cfg.cqt_grid_hop_length or cfg.hop_length)
    freqs = _lib.cqt_frequencies(
        n_bins, fmin=cfg.cqt_fmin, bins_per_octave=cfg.bins_per_octave
    ).astype(np.float64)
    times = (np.arange(n_frames, dtype=np.float64) * grid_hop / sr)

    meta: "dict[str, np.ndarray]" = {
        "sr":                np.int32(sr),
        "is_stereo":         np.bool_(is_stereo),
        "bins_per_octave":   np.int32(cfg.bins_per_octave),
        "hop_length":        np.int32(cfg.hop_length),
        "cqt_grid_hop_length": np.int32(grid_hop),
        "freqs":             freqs,
        "times":             times,
    }
    for key, val in meta.items():
        fsp = os.path.join(stream_dir, f"{key}.npy")
        if not os.path.isfile(fsp):
            try:
                np.save(fsp, val)
                print(f"  [reconstruct] wrote {key}.npy  (shape={np.asarray(val).shape})")
            except Exception as _e:
                print(f"  [reconstruct] warn: could not write {key}.npy: {_e}")
    return freqs, times


def _load_cqt_from_stream(outdir: str, shash: str) -> dict:
    """Load CQT complex arrays and metadata from an existing stream directory."""
    sd = _stream_dir_for(outdir, shash)
    is_stereo = bool(_load_shard(sd, "is_stereo"))
    sr = int(_load_shard(sd, "sr"))
    bins_per_octave = int(_load_shard(sd, "bins_per_octave"))
    hop_length = int(_load_shard(sd, "hop_length"))
    cqt_grid_hop_length = int(_load_shard(sd, "cqt_grid_hop_length"))
    freqs = _load_shard(sd, "freqs")
    times = _load_shard(sd, "times")
    real_left = _load_shard(sd, "real_left")
    imag_left = _load_shard(sd, "imag_left")
    cqt_L = real_left + 1j * imag_left
    del real_left, imag_left
    cqt_R = None
    if is_stereo:
        real_right = _load_shard(sd, "real_right")
        imag_right = _load_shard(sd, "imag_right")
        cqt_R = real_right + 1j * imag_right
        del real_right, imag_right
    return dict(sr=sr, bins_per_octave=bins_per_octave, hop_length=hop_length,
                cqt_grid_hop_length=cqt_grid_hop_length, freqs=freqs, times=times,
                cqt_L=cqt_L, cqt_R=cqt_R, is_stereo=is_stereo)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _to_float64_stereo(data: np.ndarray, sr: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Normalise raw audio → (sr, left, right) as float64.  Clamps peaks > 1."""
    if np.issubdtype(data.dtype, np.integer):
        max_abs = max(abs(np.iinfo(data.dtype).min), abs(np.iinfo(data.dtype).max))
        data = data.astype(np.float64) / max_abs
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 2:
        left, right = data[:, 0], data[:, 1]
    else:
        left = right = data
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-12)
    if peak > 1.0:
        left = left / peak
        right = right / peak
    return sr, left, right


def load_audio_stereo(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """Load any supported audio file → (sr, left, right) as float64.

    Uses soundfile (libsndfile) as the primary backend — supports WAV, FLAC,
    OGG/Vorbis, AIFF, W64, RF64, and many others.  Falls back to miniaudio
    for MP3 when soundfile is unavailable, or scipy for WAV-only.
    """
    if _HAS_SOUNDFILE:
        data, sr = _sf.read(path, dtype="float64", always_2d=False)
        return _to_float64_stereo(data, sr)
    # Fallback chain
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        try:
            import miniaudio
        except ImportError:
            raise ImportError(
                "Install soundfile (pip install soundfile) for broad codec support, "
                "or miniaudio (pip install miniaudio) for MP3.")
        decoded = miniaudio.decode_file(
            path,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=2,
        )
        interleaved = np.frombuffer(decoded.samples, dtype=np.float32).astype(np.float64)
        left = interleaved[0::2].copy()
        right = interleaved[1::2].copy()
        return _to_float64_stereo(np.column_stack([left, right]), decoded.sample_rate)
    # WAV-only fallback
    sr, data = wavfile.read(path)
    return _to_float64_stereo(data, sr)


def stream_filtered_channel_to_mmap(
    path: str,
    ch_idx: int,
    s0: int,
    s1: int,
    cfg: "AnalysisConfig",
    sr: int,
    out_path: str,
    chunk_samples: int = 2_000_000,
    pad_before: int = 0,
    pad_after: int = 0,
) -> "tuple[np.ndarray, int]":
    """Stream one channel of *path* through the filter chain and write to *out_path*.

    Reads soundfile blocks of *chunk_samples* samples at a time.  Never holds
    more than one block + filter state in RAM.  The IIR filter (pre, post) uses
    ``scipy.signal.sosfilt`` with state carry across blocks so there is no phase
    discontinuity at block boundaries.  Heterodyne is computed from the absolute
    sample position — also continuous across blocks.

    *pad_before* / *pad_after*: number of REAL file samples to include BEFORE
    s0 and AFTER s1 in the output mmap.  These give the CQT's analysis windows
    genuine signal content instead of zeros when the analysis region is sliced
    from the middle of a longer file.  The actual padding added is clamped to
    the available file content (it will be less than requested at the file
    boundaries).

    Returns ``(mmap, actual_pad_before)`` where *actual_pad_before* is the
    number of pre-padding samples actually prepended (≤ pad_before).  The
    caller must pass this as ``signal_offset`` to ``cqt()`` so that frame-0
    aligns to sample *s0* rather than the start of the padded mmap.
    """
    from scipy.signal import butter, sosfilt_zi
    from filter_engine import build_streaming_filter, FIR, IIR
    _fmode = IIR if cfg.filter_mode == "iir" else FIR

    with _sf.SoundFile(path) as _probe:
        total_frames = _probe.frames

    # Extend read range with real file content as CQT padding.
    s0_actual = max(0, s0 - pad_before)
    s1_actual = min(total_frames, s1 + pad_after)
    actual_pad_before = s0 - s0_actual   # ≤ pad_before
    n_samples = s1_actual - s0_actual

    # Allocate the output mmap.
    dst = np.lib.format.open_memmap(out_path, mode="w+",
                                     dtype=np.float64, shape=(n_samples,))

    # --- Build filter objects (unified FIR/IIR via filter_engine) ---
    nyq = sr / 2.0

    pre_filter = None
    if cfg.prefilter_bp_fmin is not None or cfg.prefilter_bp_fmax is not None:
        pre_filter = build_streaming_filter(
            sr, cfg.prefilter_bp_fmin, cfg.prefilter_bp_fmax,
            filter_mode=_fmode, order=8)

    post_filter = None
    if cfg.postfilter_bp_fmin is not None or cfg.postfilter_bp_fmax is not None:
        post_filter = build_streaming_filter(
            sr, cfg.postfilter_bp_fmin, cfg.postfilter_bp_fmax,
            filter_mode=_fmode, order=8)

    het_hz = float(cfg.heterodyne_hz) if cfg.heterodyne_hz else 0.0

    # Carrier-removal high-pass: stays IIR (physics-derived, not user-configurable).
    carrier_hp_filter = None
    if het_hz != 0.0:
        _hp_f = max(abs(het_hz) * 0.99, 1.0)
        if _hp_f < nyq * 0.9999:
            from filter_engine import build_streaming_filter as _bsf, IIR as _IIR
            carrier_hp_filter = _bsf(sr, _hp_f, None, filter_mode=_IIR, order=8)

    write_pos = 0
    with _sf.SoundFile(path) as sf_file:
        sf_file.seek(s0_actual)
        read_pos = 0
        while read_pos < n_samples:
            n_read = min(chunk_samples, n_samples - read_pos)
            block = sf_file.read(n_read, dtype="float64", always_2d=True)
            if len(block) == 0:
                break
            if block.ndim == 2 and block.shape[1] > ch_idx:
                ch = block[:, ch_idx].copy()
            else:
                ch = block.reshape(-1).copy()

            if pre_filter is not None:
                ch = pre_filter.process(ch)

            if het_hz != 0.0:
                # Absolute sample position within the FILE (not within s0_actual).
                abs_start = s0_actual + read_pos
                t_chunk = (np.arange(len(ch), dtype=np.float64) + abs_start) / sr
                # Complex single-sideband (SSB) heterodyne:
                #   Re[analytic(x) * exp(j*2π*f_c*t)]
                # shifts near-DC content at f_in → f_c + f_in without a mirror
                # sideband at f_c - f_in and without carrier leakage at f_c.
                # A simple real cos multiplication produces double-sideband with
                # carrier, making it impossible to separate lifted content from
                # the carrier tone during playback.
                from scipy.signal import hilbert as _sp_hilbert
                ch_analytic = _sp_hilbert(ch)   # ch + j·H{ch}
                phasor = np.exp(1j * 2.0 * np.pi * het_hz * t_chunk)
                ch = np.real(ch_analytic * phasor)

            if carrier_hp_filter is not None:
                # Carrier-removal: high-pass tuned to the carrier frequency,
                # rejecting sub-carrier residual from imperfect Hilbert edges.
                ch = carrier_hp_filter.process(ch)

            if post_filter is not None:
                ch = post_filter.process(ch)

            dst[write_pos: write_pos + len(ch)] = ch
            write_pos += len(ch)
            read_pos += len(block)

    # Spline detrend applied to the full channel after streaming.
    # Requires the complete signal, so it runs post-loop on the mmap.
    if cfg.spline_detrend is not None and write_pos > 0:
        from signal_tools import spline_detrend as _spline_detrend
        _full = dst[:write_pos].copy()
        _detrended, _ = _spline_detrend(_full, sr, cfg.spline_detrend)
        dst[:write_pos] = _detrended

    dst.flush()
    # Re-open read-only so the caller gets an immutable view.
    mmap_ro = np.lib.format.open_memmap(out_path, mode="r", dtype=np.float64,
                                         shape=(n_samples,))
    return mmap_ro, actual_pad_before


# ---------------------------------------------------------------------------
# Processed-audio playback helpers
# ---------------------------------------------------------------------------

def _write_wav_stereo_f64(
    out_path: str,
    left: "np.ndarray",
    right: "np.ndarray",
    sr: int,
) -> None:
    """Write peak-normalised stereo 16-bit PCM WAV from two float64 arrays."""
    import struct
    n = len(left)
    peak = max(float(np.abs(left).max()), float(np.abs(right).max()))
    scale = 32767.0 / peak if peak > 1e-9 else 1.0
    l16 = (left  * scale).clip(-32768, 32767).astype(np.int16)
    r16 = (right * scale).clip(-32768, 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = l16
    interleaved[1::2] = r16
    raw = interleaved.tobytes()
    with open(out_path, "wb") as _wfh:
        _wfh.write(b"RIFF")
        _wfh.write(struct.pack("<I", 36 + len(raw)))
        _wfh.write(b"WAVEfmt ")
        _wfh.write(struct.pack("<IHHIIHH", 16, 1, 2, sr, sr * 4, 4, 16))
        _wfh.write(b"data")
        _wfh.write(struct.pack("<I", len(raw)))
        _wfh.write(raw)


def _save_processed_audio_streaming(
    wav_path: str,
    s0: int,
    s1: int,
    is_stereo: bool,
    sr: int,
    cfg: "AnalysisConfig",
    out_path: str,
) -> None:
    """Stream both channels through the filter/heterodyne chain and save as WAV.

    Only called when at least one of heterodyne_hz, prefilter_bp, or
    postfilter_bp is active.  The resulting WAV covers [s0, s1) from the
    source file with all transforms applied so that spacebar playback in the
    viewer matches the spectrogram.  Sample 0 corresponds to s0/sr seconds
    in the original timeline.
    """
    import gc as _gc
    import tempfile as _tf2
    tmps: "list[str]" = []
    left_arr: "np.ndarray | None" = None
    right_arr: "np.ndarray | None" = None
    try:
        _fd, _tmp_l = _tf2.mkstemp(suffix=".proc_L.npy")
        os.close(_fd)
        tmps.append(_tmp_l)
        mmap_l, _ = stream_filtered_channel_to_mmap(
            wav_path, 0, s0, s1, cfg, sr, _tmp_l, pad_before=0, pad_after=0)
        left_arr = np.array(mmap_l[: s1 - s0], copy=True)
        del mmap_l
        _gc.collect()

        if is_stereo:
            _fd, _tmp_r = _tf2.mkstemp(suffix=".proc_R.npy")
            os.close(_fd)
            tmps.append(_tmp_r)
            mmap_r, _ = stream_filtered_channel_to_mmap(
                wav_path, 1, s0, s1, cfg, sr, _tmp_r, pad_before=0, pad_after=0)
            right_arr = np.array(mmap_r[: s1 - s0], copy=True)
            del mmap_r
            _gc.collect()
        else:
            right_arr = left_arr

        _write_wav_stereo_f64(out_path, left_arr, right_arr, sr)
    except Exception as _pe:
        print(f"  [warn] could not save processed audio: {_pe}")
        try:
            os.unlink(out_path)
        except Exception:
            pass
    finally:
        del left_arr, right_arr
        _gc.collect()
        for _t in tmps:
            try:
                os.unlink(_t)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Constant-Q Transform
# ---------------------------------------------------------------------------

def _stft_n_fft(n_samples: int, sr: int, cfg: "AnalysisConfig") -> int:
    """Compute the n_fft that _compute_stft will use, without running it."""
    if cfg.stft_n_fft:
        n_fft = int(cfg.stft_n_fft)
        n_fft = max(256, n_fft)
        n_fft = int(2 ** round(math.log2(n_fft)))
        return min(n_fft, max(256, int(2 ** math.ceil(math.log2(max(n_samples, 256))))))
    bpo = max(1, int(cfg.bins_per_octave))
    alpha = ((2.0 ** (2.0 / bpo) - 1.0) /
             max(2.0 ** (2.0 / bpo) + 1.0, 1e-12))
    q = max(float(cfg.cqt_filter_scale), 1e-6) / max(alpha, 1e-12)
    target_len = q * sr / max(float(cfg.cqt_fmin), 1.0)
    target_len = max(256.0, min(131072.0, target_len))
    n_fft = int(2 ** math.ceil(math.log2(target_len)))
    return min(n_fft, max(256, int(2 ** math.ceil(math.log2(max(n_samples, 256))))))


def _estimate_max_cqt_pad_samples(cfg: "AnalysisConfig", sr: int) -> int:
    """Return the maximum CQT half-window size in ORIGINAL sample rate samples.

    This is the physically correct pre/post padding needed when slicing a
    region from the middle of a recording: the lowest-frequency CQT bin has
    the widest analysis window, and every frame near the region boundary
    needs *pad_samples* of real signal content on each side to avoid
    zero/reflection artifacts.

    For subsonic analysis (cqt_fmin ≪ 1 Hz) this can be very large — e.g.
    fmin=0.01 Hz at 44100 Hz → ~152 M samples (≈ 57 min).  The caller
    clamps to available file content; files shorter than 2×pad are fully
    padded with whatever content is available.
    """
    bpo = max(1, int(cfg.bins_per_octave))
    r = 2.0 ** (1.0 / bpo)
    alpha = (r - 1.0) / (r + 1.0)          # constant-Q fractional bandwidth
    Q = max(float(cfg.cqt_filter_scale), 1e-6) / max(alpha, 1e-30)
    fmin = max(float(cfg.cqt_fmin), 1e-12)
    # CQT window at fmin: n_fft ≈ Q * sr / fmin.  Half-window is the pad.
    half_window = int(math.ceil(Q * sr / fmin / 2.0))
    return half_window


def estimate_transform_shape(
    n_samples: int, sr: int, cfg: "AnalysisConfig"
) -> "tuple[int, int]":
    """Return (n_bins, n_frames) pre-compute, suitable for shard pre-allocation.

    For STFT, both dimensions are exact.
    For CQT, n_bins is exact from config; n_frames is a safe upper bound
    (actual may be slightly less; the authoritative length is len(times) in
    the ``times.npy`` shard written after compute).
    """
    if cfg.fft_algorithm == "stft":
        n_fft = _stft_n_fft(n_samples, sr, cfg)
        pad = n_fft // 2
        hop = cfg.hop_length
        n_frames = max(1, 1 + (n_samples + 2 * pad - n_fft) // hop)
        n_bins_full = n_fft // 2 + 1
        freqs_all = np.linspace(0.0, float(sr) / 2.0, n_bins_full, dtype=np.float64)
        f_lo = max(0.0, float(cfg.cqt_fmin))
        f_hi = min(float(cfg.cqt_fmax), float(sr) / 2.0)
        b_lo = max(0, int(np.searchsorted(freqs_all, f_lo, side="left")))
        b_hi = min(n_bins_full, max(b_lo + 1, int(np.searchsorted(freqs_all, f_hi, side="right"))))
        n_bins = max(1, b_hi - b_lo)
        return n_bins, n_frames
    # CQT / NSGT path
    fmax = min(cfg.cqt_fmax, sr / 2.0 * 0.95)
    if cfg.cqt_bpo_schedule:
        n_bins = sum(cfg.cqt_bpo_schedule)
    else:
        n_bins = max(1, int(math.floor(
            cfg.bins_per_octave * math.log2(max(fmax / max(cfg.cqt_fmin, 1e-9), 1.0)))))
    # Safe upper bound: typical CQT produces slightly fewer frames than this.
    n_frames = max(1, int(math.ceil(n_samples / cfg.hop_length)) + 4)
    return n_bins, n_frames


def _compute_stft(
    x: np.ndarray,
    sr: int,
    cfg: "AnalysisConfig",
    shards: "dict[str, np.memmap]",
    zero_front: bool = False,
) -> "tuple[np.ndarray, np.ndarray]":
    """STFT path for fft_algorithm='stft'.

    Writes real+imag output directly into the pre-allocated shard memmaps
    ``shards['real_left']`` and ``shards['imag_left']``.  Returns only
    (freqs, times); no complex or power arrays are held in RAM.
    """
    import torch
    from scipy.signal import get_window as _get_window

    real_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float64 if real_dt == np.float64 else torch.float32

    n_fft = _stft_n_fft(len(x), sr, cfg)
    hop = cfg.hop_length

    # Window — fall back to hann if the name is not recognised
    try:
        win_np = _get_window(cfg.cqt_window, n_fft).astype(real_dt)
    except Exception:
        win_np = np.hanning(n_fft).astype(real_dt)
    win_t = torch.as_tensor(win_np, device=device, dtype=torch_dtype)

    x_arr = np.asarray(x, dtype=real_dt)
    pad = n_fft // 2
    pad_mode = "constant" if (zero_front or int(x_arr.shape[-1]) <= pad) else "reflect"
    from torch_cqt import _pad as torchcqt_pad

    total_len = x_arr.shape[-1]
    n_frames = max(1, 1 + (total_len + 2 * pad - n_fft) // hop)
    n_bins_full = n_fft // 2 + 1
    freqs_all = np.linspace(0.0, float(sr) / 2.0, n_bins_full, dtype=real_dt)
    f_lo = max(0.0, float(cfg.cqt_fmin))
    f_hi = min(float(cfg.cqt_fmax), float(sr) / 2.0)
    b_lo = max(0, int(np.searchsorted(freqs_all, f_lo, side="left")))
    b_hi = min(n_bins_full, max(b_lo + 1, int(np.searchsorted(freqs_all, f_hi, side="right"))))
    n_bins = max(1, b_hi - b_lo)

    real_mm = shards["real_left"]
    imag_mm = shards["imag_left"]
    print(f"  n_fft={n_fft}  hop={hop}  n_bins={n_bins} / {n_bins_full}  n_frames={n_frames}"
          f"  keep=[{b_lo}:{b_hi}]  pad={pad}  pad_mode='{pad_mode}'")
    import sys; sys.stdout.flush()

    # Budget for the double-buffered STFT conveyor:
    # VRAM:
    # two slot-local input staging buffers, two in-flight complex outputs, and
    # one chunk of transient FFT workspace margin.
    _itemsize = np.dtype(real_dt).itemsize
    _bytes_per_frame = (2 * hop + n_fft + 4 * n_bins_full) * _itemsize
    if device.type == "cuda":
        _free_vram, _ = torch.cuda.mem_get_info(device)
        _budget = max(64 * 1024 * 1024, int(_free_vram * 0.70))
    else:
        _budget = 512 * 1024 * 1024
    _vram_chunk_frames = max(1, _budget // _bytes_per_frame)

    # Host pinned RAM:
    # two slots × (real + imag) staging buffers sized to the retained band.
    _host_bytes_per_frame = 4 * n_bins * _itemsize
    _avail_ram_mb = _ram_available_mb()
    if math.isfinite(_avail_ram_mb):
        _host_budget = int(max(1024, min(8192, _avail_ram_mb * 0.50)) * 1024 * 1024)
    else:
        _host_budget = 8 * 1024 * 1024 * 1024
    _host_chunk_frames = max(1, _host_budget // max(1, _host_bytes_per_frame))

    max_chunk_frames = max(1, min(n_frames, _vram_chunk_frames, _host_chunk_frames))
    print(f"[STFT] VRAM budget {_budget // 1024**2} MB, host pinned budget {_host_budget // 1024**2} MB"
          f" -> initial chunk {max_chunk_frames} frames")
    import sys as _sys; _sys.stdout.flush()

    with _ShardPipeline(
        n_fft, hop, n_bins, b_lo, b_hi, n_frames, max_chunk_frames,
        win_t, device, torch_dtype, real_mm, imag_mm,
    ) as pipe:
        pipe.run(x_arr, pad, pad_mode, torchcqt_pad)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    freqs = freqs_all[b_lo:b_hi]
    times = np.arange(n_frames, dtype=real_dt) * real_dt(hop / sr)
    cfg.cqt_grid_hop_length = int(hop)
    return freqs, times


def compute_cqt(
    audio_path: str,
    ch_idx: int,
    s0: int,
    s1: int,
    sr: int,
    cfg: "AnalysisConfig",
    shards: "dict[str, np.memmap]",
    zero_front: bool = False,
    tmp_dir: "str | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute CQT/STFT and write directly into pre-allocated shard memmaps.

    Audio is streamed from *audio_path* one block at a time — the file is
    never loaded entirely into RAM.  *ch_idx* selects the channel (0=L, 1=R).
    *s0*/*s1* bound the sample range to analyse.

    *shards* must be pre-allocated by the caller via ``_alloc_shards`` before
    this function is called.  The keys ``"real_left"`` and ``"imag_left"``
    (and ``"real_right"``/``"imag_right"`` for the right channel) must be
    present and backed by writable memmap files in the stream directory.

    Returns (freqs, times) only — no complex or power arrays in RAM.
    The shards on disk are the sole output.
    """
    import gc
    import tempfile as _tempfile

    if cfg.fft_algorithm == "stft":
        # STFT path: fall back to loading the channel into RAM for now.
        # (streaming STFT is a separate effort.)
        if _HAS_SOUNDFILE:
            with _sf.SoundFile(audio_path) as _f:
                _f.seek(s0)
                _blk = _f.read(s1 - s0, dtype="float64", always_2d=True)
            x = _blk[:, ch_idx] if _blk.ndim == 2 and _blk.shape[1] > ch_idx else _blk.reshape(-1)
        else:
            sr_tmp, data = wavfile.read(audio_path)
            x = _to_float64_stereo(data, sr_tmp)[1 + ch_idx][s0:s1]
        return _compute_stft(x, sr, cfg, shards, zero_front=zero_front)

    from torch_cqt_new import (
        ExplicitOctaveSchedule,
        _common_hop_and_strides,
        cqt as _torch_cqt,
    )

    if cfg.cqt_fmin <= 0:
        raise ValueError(
            f"--cqt-fmin must be > 0 Hz (got {cfg.cqt_fmin}); try 5 for infrasonic sub-bass.")
    fmax = min(cfg.cqt_fmax, sr / 2.0 * 0.95)
    sched_octaves = max(
        [len(v) for v in (
            cfg.cqt_bpo_schedule,
            cfg.cqt_hop_schedule,
            cfg.cqt_filter_scale_schedule,
        ) if v],
        default=0,
    )
    if cfg.cqt_bpo_schedule:
        n_bins = sum(cfg.cqt_bpo_schedule)
    else:
        n_bins = int(math.floor(cfg.bins_per_octave * math.log2(fmax / cfg.cqt_fmin)))
    hop = cfg.hop_length
    bpo_func = (ExplicitOctaveSchedule(cfg.cqt_bpo_schedule, integer=True)
                if cfg.cqt_bpo_schedule else None)
    hop_func = (ExplicitOctaveSchedule(cfg.cqt_hop_schedule, integer=True)
                if cfg.cqt_hop_schedule else None)
    fs_func = (ExplicitOctaveSchedule(cfg.cqt_filter_scale_schedule)
               if cfg.cqt_filter_scale_schedule else None)
    n_octaves = (sched_octaves if sched_octaves > 0
                 else max(1, int(math.ceil(math.log2(max(fmax / cfg.cqt_fmin, 1.001))))))

    real_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float64 if real_dt == np.float64 else torch.float32

    resample_kw = {
        "lowpass_filter_width": cfg.resample_taps,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 14.769656459379492,
    }

    _sink = shards["_sink"]

    # ── Compute physics-based padding: real file content before/after [s0, s1) ──
    # The lowest-frequency CQT bin needs up to half its analysis window of real
    # signal on each side of the region.  We fetch those samples from disk so
    # the CQT never sees zeros at the region boundaries.
    max_pad = _estimate_max_cqt_pad_samples(cfg, sr)
    if max_pad > 0 and (s0 > 0 or True):  # always try; clamps at file boundary
        _pad_str = (f"{max_pad/sr:.1f}s" if max_pad < sr * 3600
                    else f"{max_pad/sr/3600:.2f}h")
        print(f"  CQT padding: requesting {_pad_str} of real signal on each side "
              f"(fmin={cfg.cqt_fmin} Hz)")

    # ── Stream the filtered audio to a temporary memmap ──
    _td = tmp_dir or _tempfile.gettempdir()
    _tmp_fd, _tmp_path = _tempfile.mkstemp(suffix=".filtered.npy", dir=_td)
    os.close(_tmp_fd)
    try:
        filtered_mmap, actual_pad_before = stream_filtered_channel_to_mmap(
            audio_path, ch_idx, s0, s1, cfg, sr, _tmp_path,
            pad_before=max_pad, pad_after=max_pad)

        if actual_pad_before < max_pad:
            print(f"  [info] pre-padding clamped to {actual_pad_before/sr:.3f}s "
                  f"(analysis starts at file position {s0/sr:.3f}s)")

        freqs_np, n_frames = _torch_cqt(
            filtered_mmap, sr,
            hop_length=hop,
            fmin=cfg.cqt_fmin,
            n_bins=n_bins,
            bins_per_octave=cfg.bins_per_octave,
            filter_scale=cfg.cqt_filter_scale,
            window=cfg.cqt_window,
            bpo_func=bpo_func,
            hop_func=hop_func,
            filter_scale_func=fs_func,
            pad_mode="constant" if zero_front else "reflect",
            device=device,
            dtype=torch_dtype,
            resample_kw=resample_kw,
            shards={"_sink": _sink},
            signal_n_samples=s1 - s0,          # true analysis length (not padded)
            signal_offset=actual_pad_before,    # where [s0,s1) starts in the mmap
        )
    finally:
        # Release mmap reference and delete the temp file.
        try:
            del filtered_mmap
        except Exception:
            pass
        gc.collect()
        try:
            os.unlink(_tmp_path)
        except Exception:
            pass

    freqs = freqs_np.astype(real_dt, copy=False)

    if cfg.cqt_hop_schedule:
        dec_hops = cfg.cqt_hop_schedule[:n_octaves]
        orig_hops: list[int] = []
        scale_val = 1
        for hop_oct in dec_hops:
            orig_hops.append(int(hop_oct) * scale_val)
            if int(hop_oct) % 2 == 0:
                scale_val *= 2
        grid_hop, _ = _common_hop_and_strides(orig_hops)
    else:
        grid_hop = hop
    cfg.cqt_grid_hop_length = int(grid_hop)
    times = np.arange(n_frames, dtype=real_dt) * real_dt(grid_hop / sr)
    return freqs, times


def _compute_cqt_array(
    x: np.ndarray,
    sr: int,
    cfg: "AnalysisConfig",
    shards: "dict[str, np.memmap]",
    zero_front: bool = False,
) -> "tuple[np.ndarray, np.ndarray]":
    """CQT/STFT on a pre-built numpy array (used for composite mode only).

    The array must already have all filters / heterodyne applied.
    The full array IS converted to a VRAM tensor; only use this path when the
    array is unavoidably in RAM (e.g. composite builds it from scratch).
    """
    if cfg.fft_algorithm == "stft":
        return _compute_stft(x, sr, cfg, shards, zero_front=zero_front)

    from torch_cqt_new import (
        ExplicitOctaveSchedule,
        _common_hop_and_strides,
        cqt as _torch_cqt,
    )

    if cfg.cqt_fmin <= 0:
        raise ValueError(f"--cqt-fmin must be > 0 Hz (got {cfg.cqt_fmin})")
    fmax = min(cfg.cqt_fmax, sr / 2.0 * 0.95)
    sched_octaves = max(
        [len(v) for v in (cfg.cqt_bpo_schedule, cfg.cqt_hop_schedule,
                          cfg.cqt_filter_scale_schedule) if v], default=0)
    if cfg.cqt_bpo_schedule:
        n_bins = sum(cfg.cqt_bpo_schedule)
    else:
        n_bins = int(math.floor(cfg.bins_per_octave * math.log2(fmax / cfg.cqt_fmin)))
    hop = cfg.hop_length
    bpo_func = ExplicitOctaveSchedule(cfg.cqt_bpo_schedule, integer=True) \
        if cfg.cqt_bpo_schedule else None
    hop_func = ExplicitOctaveSchedule(cfg.cqt_hop_schedule, integer=True) \
        if cfg.cqt_hop_schedule else None
    fs_func = ExplicitOctaveSchedule(cfg.cqt_filter_scale_schedule) \
        if cfg.cqt_filter_scale_schedule else None
    n_octaves = sched_octaves if sched_octaves > 0 else \
        max(1, int(math.ceil(math.log2(max(fmax / cfg.cqt_fmin, 1.001)))))

    real_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float64 if real_dt == np.float64 else torch.float32
    y = torch.as_tensor(np.asarray(x), device=device, dtype=torch_dtype)

    resample_kw = {"lowpass_filter_width": cfg.resample_taps,
                   "resampling_method": "sinc_interp_kaiser",
                   "beta": 14.769656459379492}

    freqs_np, n_frames = _torch_cqt(
        y, sr, hop_length=hop, fmin=cfg.cqt_fmin, n_bins=n_bins,
        bins_per_octave=cfg.bins_per_octave, filter_scale=cfg.cqt_filter_scale,
        window=cfg.cqt_window, bpo_func=bpo_func, hop_func=hop_func,
        filter_scale_func=fs_func,
        pad_mode="constant" if zero_front else "reflect",
        device=device, dtype=torch_dtype, resample_kw=resample_kw,
        shards={"_sink": shards["_sink"]},
    )
    del y
    freqs = freqs_np.astype(real_dt, copy=False)
    if cfg.cqt_hop_schedule:
        dec_hops = cfg.cqt_hop_schedule[:n_octaves]
        orig_hops: list[int] = []
        scale_val = 1
        for hop_oct in dec_hops:
            orig_hops.append(int(hop_oct) * scale_val)
            if int(hop_oct) % 2 == 0:
                scale_val *= 2
        grid_hop, _ = _common_hop_and_strides(orig_hops)
    else:
        grid_hop = hop
    cfg.cqt_grid_hop_length = int(grid_hop)
    times = np.arange(n_frames, dtype=real_dt) * real_dt(grid_hop / sr)
    return freqs, times


# ---------------------------------------------------------------------------
# Time-domain onset enhancement
# ---------------------------------------------------------------------------

def compute_onset_envelope(
    x: np.ndarray, sr: int, freqs: np.ndarray, hop: int, n_frames: int,
    compute_dtype: str = "float64",
) -> np.ndarray:
    """Compute a per-bin, per-frame onset strength from short-window time-domain
    energy, to sharpen transient edges that the CQT smears.

    For each CQT frequency bin, we bandpass-filter the signal around that
    frequency (±1 semitone) using IIR filters, compute a short-window RMS
    envelope (window = 2 * hop samples for sub-frame resolution), then take
    the half-wave-rectified first difference as onset strength.

    CPU RAM backpressure
    --------------------
    Each sosfilt call allocates ~signal_len × dtype_bytes of RAM.  To prevent
    runaway allocation when many semitones are processed in sequence, we
    explicitly delete intermediates and invoke gc after each band.  An
    optional env var SPECTRAL_ONSET_RAM_MB caps working-set size by pausing
    between bands when psutil reports available RAM below that threshold.

    Returns onset strength matrix (n_bins, n_frames), values ≥ 0, normalised
    per-bin to [0, 1].  This can be additively blended with the CQT power
    to enhance transient clarity without altering steady-state tones.
    """
    import gc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dt = np.float64 if compute_dtype == "float64" else np.float32
    torch_dtype = torch.float64 if real_dt == np.float64 else torch.float32
    n_bins = len(freqs)
    onset = np.zeros((n_bins, n_frames), dtype=real_dt)

    # Short RMS window: 2× hop gives one sub-division per CQT frame.
    rms_win = 2 * hop
    midi_vals = 12.0 * np.log2(np.clip(freqs, 1e-9, None) / 440.0) + 69.0
    semitone_ids = np.round(midi_vals).astype(int)
    unique_semitones = np.unique(semitone_ids)

    from scipy.signal import butter, sosfilt
    for semi in unique_semitones:
        _backpressure_wait(label=f"onset_semi_{semi}")

        bin_indices = np.flatnonzero(semitone_ids == semi)
        center_freq = 440.0 * 2.0 ** ((semi - 69.0) / 12.0)
        f_lo = center_freq * 2.0 ** (-1.0 / 12.0)
        f_hi = center_freq * 2.0 ** (1.0 / 12.0)
        nyq = sr / 2.0
        if f_hi >= nyq or f_lo <= 0:
            continue
        try:
            sos = butter(4, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
        except ValueError:
            continue

        filtered = sosfilt(sos, x).astype(real_dt, copy=False)

        # RMS envelope via GPU conv1d on squared signal.
        fsig = torch.from_numpy(filtered ** 2).to(device=device, dtype=torch_dtype)
        del filtered  # release CPU copy immediately — backpressure on RAM
        box = torch.ones(1, 1, rms_win, device=device, dtype=torch_dtype) / rms_win
        rms_sq = torch.nn.functional.conv1d(
            fsig.view(1, 1, -1), box, padding=rms_win // 2
        ).view(-1)
        del fsig, box

        frame_indices = torch.arange(n_frames, device=device) * hop
        frame_indices = frame_indices.clamp(max=rms_sq.shape[0] - 1)
        rms_frames = rms_sq[frame_indices].cpu().numpy()
        del rms_sq, frame_indices

        onset_str = np.maximum(np.diff(rms_frames, prepend=rms_frames[0]), 0.0)
        peak = onset_str.max()
        if peak > 0:
            onset_str /= peak
        for bi in bin_indices:
            onset[bi] = onset_str
        del onset_str, rms_frames

        # Explicit GC pass so the next sosfilt starts clean.
        gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return onset


# ---------------------------------------------------------------------------
# Musical scale annotation
# ---------------------------------------------------------------------------

def midi_to_note_name(n: int) -> str:
    octave = (n // 12) - 1
    return f"{NOTE_NAMES[n % 12]}{octave}"


def semitone_mask_for_freqs(freqs: np.ndarray, bins_per_octave: int) -> Tuple[np.ndarray, List[str]]:
    """Return a bool mask where each CQT bin falls on a 12-TET semitone,
    and the corresponding note name strings (empty string for non-semitone bins).

    A bin is "on a semitone" when it is within half a bin-width of a semitone
    boundary in cent space.
    """
    midi_frac = 12.0 * np.log2(np.clip(freqs, 1e-9, None) / 440.0) + 69.0
    cents_from_semitone = (midi_frac * 100.0) % 100.0
    if len(midi_frac) > 1:
        cents_per_bin = np.empty_like(midi_frac)
        cents_per_bin[0] = abs((midi_frac[1] - midi_frac[0]) * 100.0)
        cents_per_bin[-1] = abs((midi_frac[-1] - midi_frac[-2]) * 100.0)
        if len(midi_frac) > 2:
            cents_per_bin[1:-1] = abs((midi_frac[2:] - midi_frac[:-2]) * 50.0)
    else:
        cents_per_bin = np.full_like(midi_frac, 1200.0 / bins_per_octave)
    on_semitone = (
        np.minimum(cents_from_semitone, 100.0 - cents_from_semitone)
        < (cents_per_bin / 2.0))
    midi_rounded = np.rint(midi_frac).astype(int)
    names: List[str] = [""] * len(freqs)
    for i in np.flatnonzero(on_semitone):
        n = midi_rounded[i]
        names[i] = f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"
    return on_semitone, names


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def band_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= fmax)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    kernel = np.ones(width, dtype=x.dtype) / np.asarray(width, dtype=x.dtype)
    return np.convolve(x, kernel, mode="same").astype(x.dtype, copy=False)


def weighted_centroid(freqs: np.ndarray, power_band: np.ndarray) -> np.ndarray:
    real_dt = np.result_type(freqs.dtype, power_band.dtype, np.float32)
    eps = real_dt.type(EPS)
    return ((freqs[:, None] * power_band).sum(axis=0)
            / (power_band.sum(axis=0) + eps)).astype(real_dt, copy=False)


def peak_frequency(freqs: np.ndarray, power_band: np.ndarray) -> np.ndarray:
    return freqs[np.argmax(power_band, axis=0)]


def rolloff_pair(
    freqs: np.ndarray, power_band: np.ndarray, low_q: float = 0.15, high_q: float = 0.85
) -> Tuple[np.ndarray, np.ndarray]:
    csum = np.cumsum(power_band, axis=0)
    total = csum[-1, :] + np.result_type(power_band.dtype, np.float32).type(EPS)
    lo = np.clip(np.argmax(csum >= (total * low_q)[None, :], axis=0), 0, len(freqs) - 1)
    hi = np.clip(np.argmax(csum >= (total * high_q)[None, :], axis=0), 0, len(freqs) - 1)
    return freqs[lo], freqs[hi]


def track_edge_frequencies(
    freqs: np.ndarray, power: np.ndarray, edge_threshold_db: float
) -> Tuple[np.ndarray, np.ndarray]:
    real_dt = np.result_type(freqs.dtype, power.dtype, np.float32)
    rel_ratio = real_dt.type(10.0) ** real_dt.type(edge_threshold_db / 10.0)
    threshold = (np.max(power, axis=0) + real_dt.type(EPS)) * rel_ratio
    active = power >= threshold[None, :]
    any_active = active.any(axis=0)
    low = np.full(power.shape[1], np.nan, dtype=real_dt)
    high = np.full(power.shape[1], np.nan, dtype=real_dt)
    low_idx = np.argmax(active, axis=0)
    high_idx = active.shape[0] - 1 - np.argmax(active[::-1], axis=0)
    low[any_active] = freqs[low_idx[any_active]]
    high[any_active] = freqs[high_idx[any_active]]
    return low, high


def db_power(power: np.ndarray) -> np.ndarray:
    real_dt = np.result_type(power.dtype, np.float32)
    eps = real_dt.type(EPS)
    return real_dt.type(10.0) * np.log10(np.maximum(power, eps))


    def _norm_gamma(db: np.ndarray) -> np.ndarray:
        n = np.clip((db - vmin) / (vmax - vmin + EPS), 0.0, 1.0)
        return n ** color_gamma

    nL = _norm_gamma(db_L)   # (n_rows, n_cols) in [0,1]
    nR = _norm_gamma(db_R)

    # Build RGB tensor: R=max, G=right, B=left.
    # Flip rows so low frequency is at the bottom.
    R = np.maximum(nL, nR)[::-1]
    G = nR[::-1]
    B = nL[::-1]
    img_data = np.stack([R, G, B], axis=-1)  # (n_rows, n_cols, 3)

    # Red overlay proportional to pad influence — smooth gradient.
    # Intensity reflects the fraction of each CQT window that overlaps with
    # composite padding.  Low-frequency bins have longer windows so the red
    # gradient extends further from pad boundaries at the bottom of the image.
    if sub_influence is not None and np.any(sub_influence > 0):
        inf = sub_influence[::-1]  # flip to match image row order
        img_data[:, :, 0] = np.maximum(img_data[:, :, 0], inf * 0.3)
        img_data[:, :, 1] *= (1.0 - 0.7 * inf)
        img_data[:, :, 2] *= (1.0 - 0.7 * inf)

    # Convert to uint8.
    img_u8 = (np.clip(img_data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # --- Margins and labels ---
    # Font: try to load a monospace system font; fall back to default.
    try:
        font = ImageFont.truetype("consola.ttf", 11)
        font_title = ImageFont.truetype("consola.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("cour.ttf", 11)
            font_title = ImageFont.truetype("cour.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
            font_title = font

    # Build semitone labels.
    semi_pos = np.flatnonzero(sub_semi)
    # In image coords, row i in data → y = (n_rows - 1 - i) in the data strip.
    semi_labels = []
    for i in semi_pos:
        y_img = n_rows - 1 - i
        lbl = f"{sub_names[i]}  {sub_freqs[i]:.1f} Hz"
        semi_labels.append((y_img, lbl))

    # Time footer labels: place ticks at nice intervals.
    t0, t1 = float(times[0]), float(times[-1])
    dur = t1 - t0
    if dur > 0:
        tick_step = _nice_time_step(dur, n_cols)
        t_first = math.ceil(t0 / tick_step) * tick_step
        time_ticks = []
        t = t_first
        while t <= t1:
            frac = (t - t0) / dur
            px = int(frac * (n_cols - 1))
            time_ticks.append((px, f"{t:.1f}"))
            t += tick_step
    else:
        time_ticks = [(0, f"{t0:.1f}")]

    # Measure label widths for margins.
    max_lbl_w = 0
    for _, lbl in semi_labels:
        bbox = font.getbbox(lbl)
        max_lbl_w = max(max_lbl_w, bbox[2] - bbox[0])
    margin_left = max_lbl_w + 12
    margin_right = max_lbl_w + 12
    margin_top = 22
    margin_bottom = 22
    total_w = margin_left + n_cols + margin_right
    total_h = margin_top + n_rows + margin_bottom

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    # Paste spectrogram data.
    spec_img = Image.fromarray(img_u8, "RGB")
    canvas.paste(spec_img, (margin_left, margin_top))

    draw = ImageDraw.Draw(canvas)

    # Title (centered in top margin).
    tbbox = font_title.getbbox(title)
    tw = tbbox[2] - tbbox[0]
    draw.text(((total_w - tw) // 2, 2), title, fill=(220, 220, 220), font=font_title)

    # Y-axis labels on both sides.
    for y_img, lbl in semi_labels:
        y_canvas = margin_top + y_img
        bbox = font.getbbox(lbl)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        # Left axis: right-aligned.
        draw.text((margin_left - lw - 4, y_canvas - lh // 2),
                  lbl, fill=(180, 180, 180), font=font)
        # Right axis: left-aligned.
        draw.text((margin_left + n_cols + 4, y_canvas - lh // 2),
                  lbl, fill=(180, 180, 180), font=font)
        # Horizontal guide line across spectrogram.
        draw.line([(margin_left, y_canvas), (margin_left + n_cols - 1, y_canvas)],
                  fill=(60, 60, 60), width=1)

    # Time axis labels in the footer.
    for px, lbl in time_ticks:
        x_canvas = margin_left + px
        bbox = font.getbbox(lbl)
        lw = bbox[2] - bbox[0]
        draw.text((x_canvas - lw // 2, margin_top + n_rows + 3),
                  lbl, fill=(180, 180, 180), font=font)
        # Small tick mark.
        draw.line([(x_canvas, margin_top + n_rows),
                   (x_canvas, margin_top + n_rows + 2)],
                  fill=(120, 120, 120), width=1)

    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def analyze(
    freqs: np.ndarray, times: np.ndarray, power: np.ndarray, cfg: AnalysisConfig
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n_frames = power.shape[1]
    real_dt = np.result_type(freqs.dtype, power.dtype, np.float32)
    bass_mask = band_mask(freqs, cfg.bass_min, cfg.bass_max)
    kick_mask = band_mask(freqs, cfg.kick_min, cfg.kick_max)
    bulge_mask = band_mask(freqs, cfg.bulge_min, cfg.bulge_max)

    # When CQT fmin is above the bass/kick/bulge range (e.g. Hybrid mode
    # or narrow partial analysis), produce zeroed metrics instead of crashing.
    has_bass = np.any(bass_mask)
    has_kick = np.any(kick_mask)
    has_bulge = np.any(bulge_mask)

    zeros = np.zeros(n_frames, dtype=real_dt)
    nans = np.full(n_frames, np.nan, dtype=real_dt)

    if has_bass:
        bass_power = power[bass_mask, :]
        bass_energy_db = db_power(bass_power.sum(axis=0))
        bass_delta = np.diff(bass_power, axis=1, prepend=bass_power[:, :1])
        bass_flux_db = db_power(
            np.maximum(bass_delta, real_dt.type(0.0)).sum(axis=0)
            + real_dt.type(EPS)
        )
    else:
        bass_energy_db = zeros.copy()
        bass_flux_db = zeros.copy()

    if has_kick:
        kick_freqs = freqs[kick_mask]
        kick_power = power[kick_mask, :]
        kick_centroid_hz = weighted_centroid(kick_freqs, kick_power)
        kick_peak_freq_hz = peak_frequency(kick_freqs, kick_power)
        kick_rolloff_low_hz, kick_rolloff_high_hz = rolloff_pair(
            kick_freqs, kick_power, 0.15, 0.85)
        kick_energy_db = db_power(kick_power.sum(axis=0))
    else:
        kick_centroid_hz = nans.copy()
        kick_peak_freq_hz = nans.copy()
        kick_rolloff_low_hz = nans.copy()
        kick_rolloff_high_hz = nans.copy()
        kick_energy_db = zeros.copy()

    if has_bulge:
        bulge_power = power[bulge_mask, :]
        bulge_energy_db = db_power(bulge_power.sum(axis=0))
        dt = np.median(np.diff(times)) if len(times) > 1 else 0.01
        smooth_width = max(1, int(round(
            (cfg.envelope_smooth_ms / 1000.0) / max(dt, 1e-6))))
        bulge_energy_db_smooth = moving_average(bulge_energy_db, smooth_width)
    else:
        bulge_energy_db_smooth = zeros.copy()

    lowest_rep_freq_hz, highest_rep_freq_hz = track_edge_frequencies(
        freqs, power, cfg.edge_threshold_db)

    peaks_idx, _ = find_peaks(bulge_energy_db_smooth,
                              prominence=cfg.peak_prominence_db)

    return {
        "bass_energy_db": bass_energy_db,
        "kick_energy_db": kick_energy_db,
        "bulge_energy_db": bulge_energy_db_smooth,
        "kick_centroid_hz": kick_centroid_hz,
        "kick_peak_freq_hz": kick_peak_freq_hz,
        "kick_rolloff_low_hz": kick_rolloff_low_hz,
        "kick_rolloff_high_hz": kick_rolloff_high_hz,
        "lowest_rep_freq_hz": lowest_rep_freq_hz,
        "highest_rep_freq_hz": highest_rep_freq_hz,
        "bass_flux_db": bass_flux_db,
    }, peaks_idx


# ---------------------------------------------------------------------------
# CSV / TXT output
# ---------------------------------------------------------------------------

def write_metrics_csv(out_path: str, times: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    fieldnames = ["time_s"] + list(metrics.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(times):
            row: Dict = {"time_s": float(t)}
            for k, v in metrics.items():
                row[k] = "" if np.isnan(v[i]) else float(v[i])
            writer.writerow(row)


def write_peaks_csv(
    out_path: str, times: np.ndarray, metrics: Dict[str, np.ndarray], peaks_idx: np.ndarray
) -> None:
    fieldnames = ["time_s", "bulge_energy_db", "kick_peak_freq_hz",
                  "kick_centroid_hz", "kick_rolloff_low_hz", "kick_rolloff_high_hz"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in peaks_idx:
            writer.writerow({
                "time_s": float(times[i]),
                "bulge_energy_db": float(metrics["bulge_energy_db"][i]),
                "kick_peak_freq_hz": float(metrics["kick_peak_freq_hz"][i]),
                "kick_centroid_hz": float(metrics["kick_centroid_hz"][i]),
                "kick_rolloff_low_hz": float(metrics["kick_rolloff_low_hz"][i]),
                "kick_rolloff_high_hz": float(metrics["kick_rolloff_high_hz"][i]),
            })


def write_summary_txt(
    out_path: str, wav_path: str, sr: int, x: np.ndarray,
    times: np.ndarray, metrics: Dict[str, np.ndarray],
    peaks_idx: np.ndarray, cfg: AnalysisConfig,
) -> None:
    lines = [
        f"Input file: {wav_path}",
        f"Sample rate: {sr} Hz",
        f"Duration: {len(x) / sr:.3f} s",
        f"CQT frames: {len(times)}",
        f"CQT bins/octave: {cfg.bins_per_octave}  ({1200 / cfg.bins_per_octave:.2f} cents/bin)",
        f"Hop length: {cfg.cqt_grid_hop_length or cfg.hop_length} samples  "
        f"({(cfg.cqt_grid_hop_length or cfg.hop_length) / sr * 1000:.1f} ms)",
        f"CQT Q scale: {cfg.cqt_filter_scale:.3f}",
        f"CQT window: {cfg.cqt_window}",
        "",
        "Band configuration:",
        f"  Bass band:  {cfg.bass_min:.1f} - {cfg.bass_max:.1f} Hz",
        f"  Kick band:  {cfg.kick_min:.1f} - {cfg.kick_max:.1f} Hz",
        f"  Bulge band: {cfg.bulge_min:.1f} - {cfg.bulge_max:.1f} Hz",
        f"  Edge threshold: {cfg.edge_threshold_db:.1f} dB relative to frame peak",
        "",
        "Summary statistics:",
        f"  Mean bass energy:              {np.nanmean(metrics['bass_energy_db']):.3f} dB",
        f"  Mean bulge energy:             {np.nanmean(metrics['bulge_energy_db']):.3f} dB",
        f"  Mean kick centroid:            {np.nanmean(metrics['kick_centroid_hz']):.3f} Hz",
        f"  Mean kick peak frequency:      {np.nanmean(metrics['kick_peak_freq_hz']):.3f} Hz",
        f"  Mean kick low rolloff (15%%):  {np.nanmean(metrics['kick_rolloff_low_hz']):.3f} Hz",
        f"  Mean kick high rolloff (85%%): {np.nanmean(metrics['kick_rolloff_high_hz']):.3f} Hz",
        f"  Mean lowest represented freq:  {np.nanmean(metrics['lowest_rep_freq_hz']):.3f} Hz",
        f"  Mean highest represented freq: {np.nanmean(metrics['highest_rep_freq_hz']):.3f} Hz",
        f"  Detected bulge peaks:          {len(peaks_idx)}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Versions manifest — tracks all analysis versions in a folder
# ---------------------------------------------------------------------------

MANIFEST_FILE = "versions.json"


def load_manifest(analysis_dir: str) -> dict:
    """Load the versions manifest, returning ``{"versions": {...}}."""
    path = os.path.join(analysis_dir, MANIFEST_FILE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"versions": {}}


def save_manifest(analysis_dir: str, manifest: dict) -> None:
    path = os.path.join(analysis_dir, MANIFEST_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _update_manifest(outdir: str, shash: str, cfg: "AnalysisConfig",
                     args: argparse.Namespace) -> None:
    """Add or update a version entry in the manifest."""
    manifest = load_manifest(outdir)
    import datetime
    manifest["versions"][shash] = {
        "npz": f"cqt_data_{shash}.npz",
        "mipmap": f"mipmap_cache_{shash}.npz",
        "created": datetime.datetime.now().isoformat(),
        "settings": {
            "hop_length": cfg.hop_length,
            "stft_n_fft": cfg.stft_n_fft,
            "fft_algorithm": cfg.fft_algorithm,
            "cqt_grid_hop_length": cfg.cqt_grid_hop_length,
            "bins_per_octave": cfg.bins_per_octave,
            "cqt_filter_scale": cfg.cqt_filter_scale,
            "cqt_window": cfg.cqt_window,
            "cqt_bpo_schedule": cfg.cqt_bpo_schedule,
            "cqt_hop_schedule": cfg.cqt_hop_schedule,
            "cqt_filter_scale_schedule": cfg.cqt_filter_scale_schedule,
            "cqt_fmin": cfg.cqt_fmin,
            "cqt_fmax": cfg.cqt_fmax,
            "bass_min": cfg.bass_min,
            "bass_max": cfg.bass_max,
            "kick_min": cfg.kick_min,
            "kick_max": cfg.kick_max,
            "bulge_min": cfg.bulge_min,
            "bulge_max": cfg.bulge_max,
            "edge_threshold_db": cfg.edge_threshold_db,
            "peak_prominence_db": cfg.peak_prominence_db,
            "envelope_smooth_ms": cfg.envelope_smooth_ms,
        },
    }
    save_manifest(outdir, manifest)


def _write_analysis_inventory(
    outdir: str,
    shash: str,
    cfg: "AnalysisConfig",
    wav_path: str,
    total_duration_s: float,
    start_time_s: float,
    end_time_s: float,
) -> None:
    inv_path = os.path.join(outdir, "analysis_inventory.json")
    if os.path.isfile(inv_path):
        try:
            with open(inv_path, "r", encoding="utf-8") as f:
                inv = AnalysisInventory.from_dict(json.load(f))
        except Exception:
            inv = AnalysisInventory()
    else:
        inv = AnalysisInventory()

    settings = {
        "fft_algorithm": cfg.fft_algorithm,
        "hop_length": cfg.hop_length,
        "stft_n_fft": cfg.stft_n_fft,
        "cqt_grid_hop_length": cfg.cqt_grid_hop_length,
        "bins_per_octave": cfg.bins_per_octave,
        "cqt_filter_scale": cfg.cqt_filter_scale,
        "cqt_window": cfg.cqt_window,
        "cqt_bpo_schedule": cfg.cqt_bpo_schedule,
        "cqt_hop_schedule": cfg.cqt_hop_schedule,
        "cqt_filter_scale_schedule": cfg.cqt_filter_scale_schedule,
        "cqt_fmin": cfg.cqt_fmin,
        "cqt_fmax": cfg.cqt_fmax,
        "cqt_compute_dtype": cfg.cqt_compute_dtype,
        "cqt_save_dtype": cfg.cqt_save_dtype,
        "wav_path": wav_path,
        "region_start": start_time_s,
        "region_end": end_time_s,
        "region_total": total_duration_s,
    }
    dataset = AnalysisDatasetRecord(
        dataset_key=f"fft:{shash}",
        engine="fft",
        algorithm=str(cfg.fft_algorithm),
        run_key=shash,
        settings_hash=shash,
        folder=outdir,
        status="materialized",
        label=f"FFT {str(cfg.fft_algorithm).upper()}",
        time_range=AnalysisTimeRange(
            start_sec=start_time_s,
            end_sec=end_time_s,
            total_sec=total_duration_s,
        ).clamped(),
        settings=settings,
        artifacts=[
            AnalysisArtifact(kind="npz", path=f"cqt_data_{shash}.npz", settings_hash=shash, folder=outdir),
            AnalysisArtifact(kind="npz_legacy", path="cqt_data.npz", settings_hash=shash, folder=outdir),
            AnalysisArtifact(kind="versions_manifest", path=MANIFEST_FILE, settings_hash=shash, folder=outdir),
        ],
    )
    inv.upsert_dataset(dataset)
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inv.to_dict(), f, indent=2, sort_keys=True)


def delete_version(analysis_dir: str, shash: str) -> bool:
    """Delete a single analysis version.

    Returns ``True`` if the folder was removed (last version deleted).
    """
    manifest = load_manifest(analysis_dir)
    entry = manifest["versions"].pop(shash, None)
    if entry is None:
        return False

    # Remove associated files
    for fkey in ("npz", "mipmap"):
        fname = entry.get(fkey, "")
        fpath = os.path.join(analysis_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    if manifest["versions"]:
        # Still have versions — update legacy symlink to point to the
        # most recently created remaining version.
        save_manifest(analysis_dir, manifest)
        latest = max(manifest["versions"].items(),
                     key=lambda kv: kv[1].get("created", ""))
        latest_npz = latest[1]["npz"]
        legacy = os.path.join(analysis_dir, "cqt_data.npz")
        if os.path.isfile(legacy) or os.path.islink(legacy):
            os.remove(legacy)
        try:
            os.symlink(latest_npz, legacy)
        except OSError:
            import shutil
            shutil.copy2(os.path.join(analysis_dir, latest_npz), legacy)
        return False
    else:
        # Last version — remove the entire folder
        import shutil
        shutil.rmtree(analysis_dir)
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    def _emit_progress(frac: float, label: str) -> None:
        frac = max(0.0, min(1.0, float(frac)))
        print(f"[PROGRESS] {frac:.4f} {label}", flush=True)

    _emit_progress(0.01, "init")
    cfg = AnalysisConfig(
        fft_algorithm=str(args.fft_algorithm).strip().lower(),
        hop_length=args.hop_length,
        stft_n_fft=args.stft_n_fft,
        bins_per_octave=args.bins_per_octave,
        cqt_filter_scale=args.cqt_filter_scale,
        cqt_window=str(args.cqt_window).strip().lower(),
        cqt_bpo_schedule=_parse_schedule_arg(
            args.cqt_bpo_schedule, integer=True),
        cqt_hop_schedule=_parse_schedule_arg(
            args.cqt_hop_schedule, integer=True),
        cqt_filter_scale_schedule=_parse_schedule_arg(
            args.cqt_filter_scale_schedule, integer=False),
        cqt_fmin=args.cqt_fmin,
        cqt_fmax=args.cqt_fmax,
        bass_view_max_freq=args.bass_view_max_freq,
        treble_view_min_freq=args.treble_view_min_freq,
        bass_min=args.bass_min,
        bass_max=args.bass_max,
        kick_min=args.kick_min,
        kick_max=args.kick_max,
        bulge_min=args.bulge_min,
        bulge_max=args.bulge_max,
        edge_threshold_db=args.edge_threshold_db,
        peak_prominence_db=args.peak_prominence_db,
        envelope_smooth_ms=args.envelope_smooth_ms,
        seconds_per_inch=args.seconds_per_inch,
        color_gamma=args.color_gamma,
        resample_taps=args.resample_taps,
        prefilter_bp_fmin=args.prefilter_bp_fmin,
        prefilter_bp_fmax=args.prefilter_bp_fmax,
        heterodyne_hz=float(args.heterodyne_hz or 0.0),
        postfilter_bp_fmin=args.postfilter_bp_fmin,
        postfilter_bp_fmax=args.postfilter_bp_fmax,
        filter_mode=args.filter_mode,
        spline_detrend=_build_spline_config(args),
    )
    # Parse precision controls.  --cqt-precision is a shorthand; the explicit
    # flags override it when provided.
    _prec_map = {
        "64x32": ("float64", "float32"),
        "64x16": ("float64", "float16"),
        "32x16": ("float32", "float16"),
        "64x64": ("float64", "float64"),
    }
    if args.cqt_precision:
        cfg.cqt_compute_dtype, cfg.cqt_save_dtype = _prec_map[args.cqt_precision]
    if args.cqt_compute_precision:
        cfg.cqt_compute_dtype = f"float{args.cqt_compute_precision}"
    if args.cqt_save_precision:
        cfg.cqt_save_dtype = f"float{args.cqt_save_precision}"
    print(f"  CQT precision: compute={cfg.cqt_compute_dtype}  save={cfg.cqt_save_dtype}")

    wav_path = os.path.abspath(args.wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    outdir = args.outdir or os.path.join(os.path.dirname(wav_path), f"{base_name}_analysis")
    ensure_dir(outdir)

    # --- Compute settings hash early so resume can be checked before audio load ---
    shash = settings_token(cfg, args.composite,
                          getattr(args, 'gap_seconds', 0.5),
                          args.trim_pad)
    # --shash override: treat an existing stream directory as authoritative data.
    # Passed by the viewer's last-resort resume path when no matching settings
    # file can be found — the shard grid is self-describing, so we skip hash
    # recomputation and point directly at the known stream directory.
    if getattr(args, 'shash', None):
        shash = args.shash
        print(f"  [shash override] Using supplied hash: {shash}")

    # --- Write itinerary file: tracks which stages are complete ---
    itinerary_path = os.path.join(outdir, f"itinerary_{shash}.json")

    def _load_itinerary() -> dict:
        if os.path.isfile(itinerary_path):
            try:
                with open(itinerary_path) as _f:
                    return json.load(_f)
            except Exception:
                pass
        return {}

    def _mark_stage(stage: str) -> None:
        it = _load_itinerary()
        it[stage] = True
        with open(itinerary_path, "w") as _f:
            json.dump(it, _f, indent=2)

    def _stage_done(stage: str) -> bool:
        return bool(_load_itinerary().get(stage, False))

    # --- Check resume stage ---
    resume = _resume_stage(outdir, shash)
    if resume != "none":
        print(f"  Resume detected (hash={shash}, stage={resume})")

    # --- shards_raw: raw CQT arrays present but metadata shards missing ---
    # This happens when the run crashed after compute_cqt but before
    # _save_cqt_stream_bundle wrote sr.npy / freqs.npy / times.npy etc.
    # Strategy: reconstruct the missing metadata from array shape + settings,
    # write the derived .npy files, then fall through as "save_only".
    if resume == "shards_raw":
        _sr_dir = _stream_dir_for(outdir, shash)
        # 1. Try to get sr / is_stereo from the stream-internal settings snapshot.
        _snap_path = os.path.join(_sr_dir, f"settings_{shash}.json")
        _snap_sr: "int | None" = None
        _snap_stereo: "bool | None" = None
        if os.path.isfile(_snap_path):
            try:
                with open(_snap_path, encoding="utf-8") as _snap_fh:
                    _snap = json.load(_snap_fh)
                _snap_sr = int(_snap["sr"])
                _snap_stereo = bool(_snap["is_stereo"])
                print(f"  [shards_raw] got sr={_snap_sr}, is_stereo={_snap_stereo} "
                      f"from stream settings snapshot")
            except Exception as _se:
                print(f"  [shards_raw] warn: could not read snapshot: {_se}")
        # 2. Fall back to loading the audio file for sr / is_stereo.
        if _snap_sr is None:
            print("  [shards_raw] no snapshot found; loading audio for sr/is_stereo…")
            _emit_progress(0.05, "load_audio")
            _raw_sr, _raw_L, _raw_R = load_audio_stereo(wav_path)
            _snap_sr = _raw_sr
            _snap_stereo = not np.array_equal(_raw_L, _raw_R)
            del _raw_L, _raw_R
        # 3. Reconstruct missing shards from array shape + cfg.
        _r_freqs, _r_times = _reconstruct_stream_metadata(
            _sr_dir, cfg, _snap_sr, _snap_stereo
        )
        # 4. Promote to save_only — the existing block below will load
        #    everything it needs from the now-complete shard directory.
        resume = "save_only"
        print("  [shards_raw] metadata reconstructed; continuing as save_only")

    # --- save_only fast path: shards exist but bundle write never completed ---
    # Load all needed values from existing shards so we can skip straight to
    # the save section without reading or re-processing the audio file.
    if resume == "save_only":
        print("  save_only resume: skipping audio load and CQT compute.")
        _shard_dir = _stream_dir_for(outdir, shash)
        sr = int(_load_shard(_shard_dir, "sr"))
        is_stereo = bool(_load_shard(_shard_dir, "is_stereo"))
        freqs = _load_shard(_shard_dir, "freqs")
        times = _load_shard(_shard_dir, "times")
        cfg.cqt_grid_hop_length = int(_load_shard(_shard_dir, "cqt_grid_hop_length"))
        _grid_hop = cfg.cqt_grid_hop_length or cfg.hop_length
        # Approximate full duration from last CQT frame + one hop.
        total_duration_s = float(times[-1]) + _grid_hop / sr if len(times) > 0 else 0.0
        region_start_s = 0.0
        region_end_s = total_duration_s
        composite_regions = None
        use_zero_front = False
        x_mono = None  # audio not loaded; not needed for save_only
    else:
        # --- Probe audio metadata without loading the full file ---
        _emit_progress(0.05, "probe_audio")
        if _HAS_SOUNDFILE:
            with _sf.SoundFile(wav_path) as _sf_probe:
                sr = _sf_probe.samplerate
                _total_frames = _sf_probe.frames
                _n_channels = _sf_probe.channels
            if _n_channels >= 2:
                # Read a small probe to test whether channels are identical.
                with _sf.SoundFile(wav_path) as _sf_probe:
                    _probe = _sf_probe.read(min(10000, _total_frames),
                                            dtype="float64", always_2d=True)
                is_stereo = not np.array_equal(_probe[:, 0], _probe[:, 1])
                del _probe
            else:
                is_stereo = False
        else:
            # soundfile not available — fall back to loading the full file.
            sr, x_left, x_right = load_audio_stereo(wav_path)
            _total_frames = len(x_left)
            is_stereo = not np.array_equal(x_left, x_right)

        total_duration_s = (_total_frames / sr) if sr > 0 else 0.0
        region_start_s = 0.0
        region_end_s = total_duration_s
        s0_analysis = 0
        s1_analysis = _total_frames

        # --- Time-region slicing ---
        if args.start_time is not None or args.end_time is not None:
            if args.start_time is not None:
                s0_analysis = max(0, min(_total_frames,
                                         int(args.start_time * sr)))
            if args.end_time is not None:
                s1_analysis = max(s0_analysis,
                                  min(_total_frames, int(args.end_time * sr)))
            region_start_s = s0_analysis / sr
            region_end_s   = s1_analysis / sr
            print(f"  Partial analysis: {s0_analysis/sr:.3f}s - "
                  f"{s1_analysis/sr:.3f}s "
                  f"({s1_analysis - s0_analysis} samples of {_total_frames})")

        # --- Composite (requires full arrays in RAM — exception to streaming) ---
        composite_regions = None
        use_zero_front = False
        x_mono = None
        x_left_composite = None   # only set for composite path
        x_right_composite = None

        if args.composite is not None:
            # Composite builds new time-rearranged arrays — must load all.
            if _HAS_SOUNDFILE:
                sr, x_left_composite, x_right_composite = load_audio_stereo(wav_path)
                x_left_composite  = x_left_composite[s0_analysis:s1_analysis]
                x_right_composite = x_right_composite[s0_analysis:s1_analysis]
            else:
                x_left_composite  = x_left[s0_analysis:s1_analysis]
                x_right_composite = x_right[s0_analysis:s1_analysis]
            x_mono_composite = (x_left_composite + x_right_composite) * 0.5
            from composite import parse_composite_spec, build_composite
            audio_len_s = len(x_mono_composite) / sr
            segments = parse_composite_spec(args.composite, audio_len_s)
            x_left_composite, regions_L = build_composite(
                x_left_composite, sr, segments, cfg.hop_length, args.gap_seconds)
            x_right_composite, _ = build_composite(
                x_right_composite, sr, segments, cfg.hop_length, args.gap_seconds)
            x_mono = (x_left_composite + x_right_composite) * 0.5
            composite_regions = regions_L
            s0_analysis = 0
            s1_analysis = len(x_left_composite)
            for ri in composite_regions:
                print(f"  Region {ri.seg_idx}: original "
                      f"{ri.original_start_s:.3f}s - {ri.original_end_s:.3f}s "
                      f"-> composite {ri.composite_start_sample/sr:.3f}s - "
                      f"{ri.composite_end_sample/sr:.3f}s")
            use_zero_front = True

            # Apply pre-filter / heterodyne / post-filter to composite arrays
            # (normal path handles filters inside stream_filtered_channel_to_mmap;
            # composite must do it explicitly here on its in-RAM arrays).
            if cfg.prefilter_bp_fmin is not None or cfg.prefilter_bp_fmax is not None:
                from scipy.signal import butter, sosfiltfilt
                nyq = sr / 2.0
                lo = cfg.prefilter_bp_fmin
                hi = cfg.prefilter_bp_fmax
                if lo is not None: lo = max(lo, 0.5)
                if hi is not None: hi = min(hi, nyq * 0.9999)
                if lo is not None and hi is not None:
                    sos = butter(8, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
                elif lo is not None:
                    sos = butter(8, lo / nyq, btype="highpass", output="sos")
                else:
                    sos = butter(8, hi / nyq, btype="lowpass", output="sos")
                x_left_composite  = sosfiltfilt(sos, x_left_composite.astype(np.float64))
                x_right_composite = sosfiltfilt(sos, x_right_composite.astype(np.float64))
                x_mono = ((x_left_composite + x_right_composite) * 0.5)
                print(f"  Prefilter bandpass: "
                      f"{f'{lo:.2f} Hz' if lo is not None else 'DC'} "
                      f"– {f'{hi:.2f} Hz' if hi is not None else 'Nyquist'}")

            if cfg.heterodyne_hz != 0.0:
                from scipy.signal import hilbert as _sp_hilbert, butter as _sp_butter, sosfiltfilt as _sp_sosfiltfilt
                _hlen = len(x_left_composite)
                t_vec = np.arange(_hlen, dtype=np.float64) / sr
                phasor = np.exp(1j * 2.0 * np.pi * cfg.heterodyne_hz * t_vec)
                # Complex SSB: Re[analytic(x) * exp(j*2π*f_c*t)]
                x_left_composite = np.real(
                    _sp_hilbert(x_left_composite.astype(np.float64)) * phasor)
                x_right_composite = np.real(
                    _sp_hilbert(x_right_composite.astype(np.float64)) * phasor)
                # Carrier-removal high-pass (zero-phase on in-RAM arrays)
                _chr_nyq = sr / 2.0
                _chr_f = max(abs(cfg.heterodyne_hz) * 0.99, 1.0)
                if _chr_f < _chr_nyq * 0.9999:
                    _chr_sos = _sp_butter(8, _chr_f / _chr_nyq,
                                          btype="highpass", output="sos")
                    x_left_composite  = _sp_sosfiltfilt(_chr_sos, x_left_composite)
                    x_right_composite = _sp_sosfiltfilt(_chr_sos, x_right_composite)
                x_mono = (x_left_composite + x_right_composite) * 0.5
                print(f"  Heterodyne (complex SSB): {cfg.heterodyne_hz:+.3f} Hz")

            if cfg.spline_detrend is not None:
                from signal_tools import spline_detrend as _spline_detrend
                x_left_composite, _ = _spline_detrend(x_left_composite, sr, cfg.spline_detrend)
                x_right_composite, _ = _spline_detrend(x_right_composite, sr, cfg.spline_detrend)
                x_mono = (x_left_composite + x_right_composite) * 0.5
                _sc = cfg.spline_detrend
                print(f"  Spline detrend: target={_sc.fit_target} "
                      f"trend={_sc.trend_mode} mode={_sc.mode}"
                      f"{' robust' if _sc.robust else ''}")

            if cfg.postfilter_bp_fmin is not None or cfg.postfilter_bp_fmax is not None:
                from scipy.signal import butter, sosfiltfilt
                nyq = sr / 2.0
                lo = cfg.postfilter_bp_fmin
                hi = cfg.postfilter_bp_fmax
                if lo is not None: lo = max(lo, 0.5)
                if hi is not None: hi = min(hi, nyq * 0.9999)
                if lo is not None and hi is not None:
                    sos = butter(8, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
                elif lo is not None:
                    sos = butter(8, lo / nyq, btype="highpass", output="sos")
                else:
                    sos = butter(8, hi / nyq, btype="lowpass", output="sos")
                x_left_composite  = sosfiltfilt(sos, x_left_composite.astype(np.float64))
                x_right_composite = sosfiltfilt(sos, x_right_composite.astype(np.float64))
                x_mono = ((x_left_composite + x_right_composite) * 0.5)
                print(f"  Post-filter bandpass: "
                      f"{f'{lo:.2f} Hz' if lo is not None else 'DC'} "
                      f"– {f'{hi:.2f} Hz' if hi is not None else 'Nyquist'}")

        elif not _HAS_SOUNDFILE:
            # soundfile unavailable — x_left/x_right already loaded; use old path
            x_left_composite  = x_left[s0_analysis:s1_analysis]
            x_right_composite = x_right[s0_analysis:s1_analysis]
            x_mono = (x_left_composite + x_right_composite) * 0.5

        else:
            # Normal streaming path — print filter info; filters run inside compute_cqt
            nyq = sr / 2.0
            if cfg.prefilter_bp_fmin is not None or cfg.prefilter_bp_fmax is not None:
                lo = cfg.prefilter_bp_fmin
                hi = cfg.prefilter_bp_fmax
                if lo is not None: lo = max(lo, 0.5)
                if hi is not None: hi = min(hi, nyq * 0.9999)
                print(f"  Prefilter bandpass: "
                      f"{f'{lo:.2f} Hz' if lo is not None else 'DC'} "
                      f"– {f'{hi:.2f} Hz' if hi is not None else 'Nyquist'}")
            if cfg.heterodyne_hz != 0.0:
                print(f"  Heterodyne (complex SSB): {cfg.heterodyne_hz:+.3f} Hz  "
                      f"(physical = analysis "
                      f"{'-' if cfg.heterodyne_hz >= 0 else '+'}"
                      f"{abs(cfg.heterodyne_hz):.3f} Hz)"
                      f"  carrier-removal HP at {max(abs(cfg.heterodyne_hz)*0.99, 1.0):.3f} Hz auto-applied")
            if cfg.postfilter_bp_fmin is not None or cfg.postfilter_bp_fmax is not None:
                lo = cfg.postfilter_bp_fmin
                hi = cfg.postfilter_bp_fmax
                if lo is not None: lo = max(lo, 0.5)
                if hi is not None: hi = min(hi, nyq * 0.9999)
                print(f"  Post-filter bandpass: "
                      f"{f'{lo:.2f} Hz' if lo is not None else 'DC'} "
                      f"– {f'{hi:.2f} Hz' if hi is not None else 'Nyquist'}")
            if cfg.spline_detrend is not None:
                _sc = cfg.spline_detrend
                print(f"  Spline detrend: target={_sc.fit_target} "
                      f"trend={_sc.trend_mode} mode={_sc.mode} "
                      f"order={_sc.spline_order} edge={_sc.edge}"
                      f"{' robust' if _sc.robust else ''}  (applied post-stream)")

    # --- CQT / STFT computation or resume from shards ---
    # ShardBudget is the single authority for all shard I/O.  It owns the
    # mmap files, the async write worker (double-buffer), and the read path.
    _shard_dir = _stream_dir_for(outdir, shash)
    os.makedirs(_shard_dir, exist_ok=True)

    # Write a self-describing settings snapshot INSIDE the stream directory.
    # This makes every analysis attempt discoverable even if it crashes before
    # the root-level analysis_settings.json is written or updated — the viewer
    # can glob for cqt_data_*.stream/ and read the embedded file to show all
    # attempts regardless of which hash was last used in the UI.
    _stream_settings_path = os.path.join(_shard_dir, f"settings_{shash}.json")
    if not os.path.isfile(_stream_settings_path):
        try:
            _canonical_snap = {
                "settings_hash": shash,
                "wav_path": wav_path,
                "fft_algorithm": cfg.fft_algorithm,
                "hop_length": cfg.hop_length,
                "bins_per_octave": cfg.bins_per_octave,
                "cqt_filter_scale": round(cfg.cqt_filter_scale, 4),
                "cqt_fmin": round(cfg.cqt_fmin, 4),
                "cqt_fmax": round(cfg.cqt_fmax, 4),
                "cqt_compute_dtype": cfg.cqt_compute_dtype,
                "cqt_save_dtype": cfg.cqt_save_dtype,
                # sr and is_stereo are needed by shards_raw reconstruction; they
                # are known at this point for all code paths (loaded from shard
                # for resume paths, freshly measured for new runs).
                "sr": int(sr),
                "is_stereo": bool(is_stereo),
            }
            if cfg.cqt_bpo_schedule:
                _canonical_snap["cqt_bpo_schedule"] = cfg.cqt_bpo_schedule
            if cfg.cqt_hop_schedule:
                _canonical_snap["cqt_hop_schedule"] = cfg.cqt_hop_schedule
            if cfg.heterodyne_hz != 0.0:
                _canonical_snap["heterodyne_hz"] = round(cfg.heterodyne_hz, 6)
            if cfg.prefilter_bp_fmin is not None:
                _canonical_snap["prefilter_bp_fmin"] = round(cfg.prefilter_bp_fmin, 4)
            if cfg.prefilter_bp_fmax is not None:
                _canonical_snap["prefilter_bp_fmax"] = round(cfg.prefilter_bp_fmax, 4)
            if cfg.spline_detrend is not None:
                _canonical_snap["spline_detrend"] = cfg.spline_detrend.to_dict()
            with open(_stream_settings_path, "w", encoding="utf-8") as _ssf:
                json.dump(_canonical_snap, _ssf, indent=2, sort_keys=True)
        except Exception as _e:
            print(f"  [warn] could not write stream settings snapshot: {_e}")

    # Front-load all static recovery metadata into the stream dir BEFORE any
    # mmap allocation or compute.  Philosophy: anything that helps reconstruct
    # the run parameters or identify the grid should be durable on disk first.
    # sr and is_stereo are known at this point for every code path:
    #   - save_only / cqt resume: loaded from existing shard files above
    #   - fresh compute: set by load_audio_stereo() in the else branch
    # Files are only written when absent so resume paths don't overwrite them.
    _front_static = {
        "sr":              np.int32(sr),
        "is_stereo":       np.bool_(is_stereo),
        "bins_per_octave": np.int32(cfg.bins_per_octave),
        "hop_length":      np.int32(cfg.hop_length),
    }
    for _fsk, _fsv in _front_static.items():
        _fsp = os.path.join(_shard_dir, f"{_fsk}.npy")
        if not os.path.isfile(_fsp):
            try:
                np.save(_fsp, _fsv)
            except Exception as _fse:
                print(f"  [warn] front-load {_fsk}: {_fse}")
    del _front_static, _fsk, _fsv, _fsp

    # --- Save post-processed audio for spacebar playback ---
    # When heterodyne or band-pass filtering is active the raw audio does not
    # match what the spectrogram shows.  Write a WAV of the processed signal
    # (covering the exact analysis region) so AudioPlayer can play it back.
    # Only attempted on a fresh or "cqt" resume run (s0_analysis / s1_analysis
    # are known); save_only resume skips because audio probing was skipped.
    _has_processing = (
        cfg.heterodyne_hz != 0.0
        or cfg.prefilter_bp_fmin is not None
        or cfg.prefilter_bp_fmax is not None
        or cfg.postfilter_bp_fmin is not None
        or cfg.postfilter_bp_fmax is not None
        or cfg.spline_detrend is not None
    )
    _proc_audio_path = os.path.join(outdir, f"processed_audio_{shash}.wav")
    if _has_processing and resume != "save_only" and not os.path.isfile(_proc_audio_path):
        print("  Saving processed audio for spacebar playback...")
        if args.composite is not None and x_left_composite is not None:
            # Composite path: in-RAM arrays already have all transforms applied.
            _rc = (x_right_composite
                   if (is_stereo and x_right_composite is not None)
                   else x_left_composite)
            try:
                _write_wav_stereo_f64(
                    _proc_audio_path,
                    np.asarray(x_left_composite, dtype=np.float64),
                    np.asarray(_rc, dtype=np.float64),
                    sr,
                )
                print(f"  Processed audio saved: {os.path.basename(_proc_audio_path)}")
            except Exception as _pe:
                print(f"  [warn] processed audio save failed: {_pe}")
        elif _HAS_SOUNDFILE:
            # Streaming path: re-stream both channels through the filter chain.
            _save_processed_audio_streaming(
                wav_path, s0_analysis, s1_analysis, is_stereo, sr, cfg, _proc_audio_path)
            if os.path.isfile(_proc_audio_path):
                print(f"  Processed audio saved: {os.path.basename(_proc_audio_path)}")
        # else: soundfile unavailable and no composite — filters not applied to
        # arrays on that path either, so there is nothing post-processed to save.

    _budget = ShardBudget(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if resume in ("cqt", "save_only"):
        if resume == "save_only":
            print("Loading CQT from stream shards (save_only resume, memory-mapped)...")
        else:
            print("Loading CQT from stream shards (resume, memory-mapped)...")
        # For "cqt" resume these were not yet set; for "save_only" they were
        # already loaded above — overwrite is harmless and keeps the branch clean.
        sr = int(_load_shard(_shard_dir, "sr"))
        freqs = _load_shard(_shard_dir, "freqs")
        times = _load_shard(_shard_dir, "times")
        is_stereo = bool(_load_shard(_shard_dir, "is_stereo"))
        cfg.cqt_grid_hop_length = int(_load_shard(_shard_dir, "cqt_grid_hop_length"))
        n_frames_act = len(times)
        # Open existing shards read-only through the budget.  No raw memmaps.
        _channels = ("left", "right") if is_stereo else ("left",)
        handle = _budget.load_shards(_shard_dir, n_frames_act, _channels)
    else:
        real_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
        # Use array-based path for composite (arrays built in RAM) or if soundfile
        # is unavailable.  Otherwise use the chunk-streaming path.
        _use_array_path = (x_left_composite is not None) or (not _HAS_SOUNDFILE)

        if is_stereo:
            n_samples_est = (len(x_left_composite) if _use_array_path
                             else s1_analysis - s0_analysis)
            n_bins_est, n_frames_est = estimate_transform_shape(
                n_samples_est, sr, cfg)
            _channels = ("left", "right")
            _write_handle = _budget.open_shards(
                _shard_dir, n_bins_est, n_frames_est, np.dtype(real_dt), _channels
            )

            _emit_progress(0.10, "cqt_left")
            print("Computing CQT (left)...")
            if _use_array_path:
                freqs, times = _compute_cqt_array(
                    x_left_composite, sr, cfg,
                    {"_sink": _budget.make_sink(_write_handle, "left")},
                    zero_front=use_zero_front,
                )
                del x_left_composite
            else:
                freqs, times = compute_cqt(
                    wav_path, 0, s0_analysis, s1_analysis, sr, cfg,
                    {"_sink": _budget.make_sink(_write_handle, "left")},
                    zero_front=use_zero_front,
                    tmp_dir=_shard_dir,
                )
            n_frames_act = len(times)

            # Front-load CQT shape metadata immediately after the first channel
            # completes.  If the right channel or any later step crashes, these
            # files make the shard grid discoverable and the run resumable.
            _fl_ghop = np.int32(cfg.cqt_grid_hop_length or cfg.hop_length)
            for _flk, _flv in [("freqs", freqs), ("times", times),
                                ("cqt_grid_hop_length", _fl_ghop)]:
                _flp = os.path.join(_shard_dir, f"{_flk}.npy")
                if not os.path.isfile(_flp):
                    try:
                        np.save(_flp, _flv)
                    except Exception as _fle:
                        print(f"  [warn] front-load {_flk}: {_fle}")
            del _fl_ghop, _flk, _flv, _flp

            _emit_progress(0.38, "cqt_right")
            print("Computing CQT (right)...")
            if _use_array_path:
                _compute_cqt_array(
                    x_right_composite, sr, cfg,
                    {"_sink": _budget.make_sink(_write_handle, "right")},
                    zero_front=use_zero_front,
                )
                del x_right_composite
            else:
                compute_cqt(
                    wav_path, 1, s0_analysis, s1_analysis, sr, cfg,
                    {"_sink": _budget.make_sink(_write_handle, "right")},
                    zero_front=use_zero_front,
                    tmp_dir=_shard_dir,
                )

        else:
            n_samples_est = (len(x_mono) if _use_array_path
                             else s1_analysis - s0_analysis)
            n_bins_est, n_frames_est = estimate_transform_shape(
                n_samples_est, sr, cfg)
            _channels = ("left",)
            _write_handle = _budget.open_shards(
                _shard_dir, n_bins_est, n_frames_est, np.dtype(real_dt), _channels
            )

            _emit_progress(0.10, "cqt_mono")
            print("Computing CQT (mono)...")
            if _use_array_path:
                freqs, times = _compute_cqt_array(
                    x_mono, sr, cfg,
                    {"_sink": _budget.make_sink(_write_handle, "left")},
                    zero_front=use_zero_front,
                )
            else:
                freqs, times = compute_cqt(
                    wav_path, 0, s0_analysis, s1_analysis, sr, cfg,
                    {"_sink": _budget.make_sink(_write_handle, "left")},
                    zero_front=use_zero_front,
                    tmp_dir=_shard_dir,
                )
            n_frames_act = len(times)

            # Front-load CQT shape metadata immediately after mono compute.
            _fl_ghop = np.int32(cfg.cqt_grid_hop_length or cfg.hop_length)
            for _flk, _flv in [("freqs", freqs), ("times", times),
                                ("cqt_grid_hop_length", _fl_ghop)]:
                _flp = os.path.join(_shard_dir, f"{_flk}.npy")
                if not os.path.isfile(_flp):
                    try:
                        np.save(_flp, _flv)
                    except Exception as _fle:
                        print(f"  [warn] front-load {_flk}: {_fle}")
            del _fl_ghop, _flk, _flv, _flp

        # Flush the write worker before closing w+ handles (Windows requires
        # all open w+ memmaps to be released before os.replace can overwrite
        # them; budget.flush() ensures the worker has finished all disk ops).
        _budget.flush()
        _write_handle.close()
        del _write_handle

        # Re-open shards read-only through the budget.  No raw memmaps.
        _channels = ("left", "right") if is_stereo else ("left",)
        handle = _budget.load_shards(_shard_dir, n_frames_act, _channels)

    # --- Onset enhancement (QUARANTINED — not wired to UI yet) ---
    # _emit_progress(0.60, "onset")
    # print("Computing time-domain onset enhancement...")
    # onset = compute_onset_envelope(
    #     x_mono, sr, freqs, cfg.cqt_grid_hop_length or cfg.hop_length,
    #     len(times), compute_dtype=cfg.cqt_compute_dtype)
    # power_db = db_power(power_mono)
    # per_bin_range = (np.max(power_db, axis=1) - np.min(power_db, axis=1))[:, None]
    # real_dt = power_mono.dtype.type
    # boost_db = onset * np.clip(per_bin_range, real_dt(0.0), real_dt(6.0))
    # boost_lin = real_dt(10.0) ** (boost_db / real_dt(10.0))
    # power_mono = (power_mono * boost_lin).astype(power_mono.dtype, copy=False)

    # --- Build pad mask (per-frame boolean: True = pad region) ---
    pad_mask = np.ones(len(times), dtype=bool)  # assume all pad until proven real
    if composite_regions is not None:
        for ri in composite_regions:
            t_start = ri.composite_start_sample / sr
            t_end = ri.composite_end_sample / sr
            pad_mask &= ~((times >= t_start) & (times < t_end))
    else:
        pad_mask[:] = False  # no compositing → no pad

    # --- Build 2-D pad influence map (n_bins, n_frames) ---
    # For each CQT bin b at frequency f, the analysis window has half-width
    # h_b = ceil(Q·sr / f) / 2 samples.  The influence value at (b, frame)
    # is the fraction of that window overlapping composite-generated padding
    # (reflect / zero / inter-region gap).  This produces a smooth gradient
    # that is wider at low frequencies and narrower at high frequencies.
    pad_influence = None
    if composite_regions is not None:
        real_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
        total_samples = len(x_mono)
        sample_is_pad = np.ones(total_samples, dtype=real_dt)
        for ri in composite_regions:
            s0 = max(0, ri.composite_start_sample)
            s1 = min(total_samples, ri.composite_end_sample)
            sample_is_pad[s0:s1] = 0.0
        cumsum = np.empty(total_samples + 1, dtype=real_dt)
        cumsum[0] = 0.0
        np.cumsum(sample_is_pad, out=cumsum[1:])

        alpha = ((2.0 ** (2.0 / cfg.bins_per_octave) - 1.0)
                 / (2.0 ** (2.0 / cfg.bins_per_octave) + 1.0))
        Q_val = cfg.cqt_filter_scale / max(alpha, 1e-30)
        kernel_half = np.ceil(Q_val * sr / freqs / 2.0).astype(np.int64)
        grid_hop = cfg.cqt_grid_hop_length or cfg.hop_length
        frame_centers = (np.arange(len(times)) * grid_hop).astype(np.int64)

        n_bins = len(freqs)
        n_frames = len(times)
        pad_influence = np.empty((n_bins, n_frames), dtype=real_dt)
        chunk = 500
        for b0 in range(0, n_bins, chunk):
            b1 = min(b0 + chunk, n_bins)
            h = kernel_half[b0:b1, None]            # (chunk, 1)
            fc = frame_centers[None, :]              # (1, n_frames)
            starts = np.clip(fc - h, 0, total_samples)
            ends = np.clip(fc + h, 0, total_samples)
            wlen = (ends - starts).astype(real_dt)
            wsum = cumsum[ends] - cumsum[starts]
            pad_influence[b0:b1] = (
                wsum / np.maximum(wlen, real_dt(1.0))
            ).astype(real_dt, copy=False)
        del cumsum, sample_is_pad

    semitone_mask, note_names = semitone_mask_for_freqs(freqs, cfg.bins_per_octave)

    cents_per_bin = 1200.0 / cfg.bins_per_octave
    n_semitones = int(semitone_mask.sum())
    print(
        f"  {len(freqs)} bins  ({cents_per_bin:.2f} cents/bin)  "
        f"{n_semitones} semitones  "
        f"{len(times)} frames  "
        f"hop={(cfg.cqt_grid_hop_length or cfg.hop_length) / sr * 1000:.1f} ms"
        f"  {'stereo' if is_stereo else 'mono'}"
    )

    # --- Bass/kick/bulge metrics (QUARANTINED — not wired to UI yet) ---
    # _emit_progress(0.72, "analyze")
    # metrics, peaks_idx = analyze(freqs, times, power_mono, cfg)

    # Auto-trim: find the lowest frequency bin that actually carries energy
    # above the noise floor.  Streamed through the budget in column-chunks;
    # no full 2-D power array is ever in RAM.
    _at_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
    _bin_peak_pwr = np.full(handle.shape[0], EPS, dtype=_at_dt)
    if is_stereo:
        for (_cr, _ci, _, _), (_crR, _ciR, _, _) in zip(
            _budget.iter_frame_chunks(handle, "left",  out_dtype=_at_dt),
            _budget.iter_frame_chunks(handle, "right", out_dtype=_at_dt),
        ):
            _pw = (_cr ** 2 + _ci ** 2 + _crR ** 2 + _ciR ** 2) * _at_dt(0.5)
            np.maximum(_bin_peak_pwr, np.max(_pw, axis=1), out=_bin_peak_pwr)
            del _cr, _ci, _crR, _ciR, _pw
    else:
        for _cr, _ci, _, _ in _budget.iter_frame_chunks(handle, "left", out_dtype=_at_dt):
            np.maximum(_bin_peak_pwr, np.max(_cr ** 2 + _ci ** 2, axis=1), out=_bin_peak_pwr)
            del _cr, _ci
    bin_peak_db = db_power(_bin_peak_pwr)
    del _bin_peak_pwr
    noise_floor_db = float(np.median(bin_peak_db))
    active_bins = bin_peak_db > (noise_floor_db + 6.0)
    if np.any(active_bins):
        auto_fmin = float(freqs[np.flatnonzero(active_bins)[0]])
        auto_fmin = max(auto_fmin * 2.0 ** (-1.0 / 12.0), float(freqs[0]))
    else:
        auto_fmin = float(freqs[0])
    print(f"  Auto-trim: lowest active frequency ~= {auto_fmin:.1f} Hz")

    # x_mono is None for save_only resume (audio was never loaded).
    duration_s = (len(x_mono) / sr) if x_mono is not None else total_duration_s
    fig_width_in = max(8.0, duration_s / cfg.seconds_per_inch)
    print(f"  Duration {duration_s:.1f} s -> figure width {fig_width_in:.1f} in "
          f"@ {cfg.seconds_per_inch} s/in  (gamma={cfg.color_gamma})")

    fmax_actual = float(freqs[-1])
    bass_fmax = min(cfg.bass_view_max_freq, fmax_actual)
    treble_fmin = min(cfg.treble_view_min_freq, fmax_actual)

    # --- Save compressed CQT tensors + plotting metadata ---
    # shash was computed early (before audio load) for resume detection.
    versioned_npz = f"cqt_data_{shash}.npz"
    if resume in ("cqt", "save_only"):
        print(f"Finalizing saved CQT tensors from stream shards ({versioned_npz}) ...")
    else:
        print(f"Saving streamed CQT tensors ({versioned_npz}) ...")
    _SAVE_DTYPE_MAP = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    save_np_dtype = _SAVE_DTYPE_MAP.get(cfg.cqt_save_dtype, np.float32)

    # For float16 we must normalize to [-1, 1] before encoding; float16's
    # maximum representable value is only 65504, and CQT coefficients can
    # exceed that for loud signals.  The scale is stored in the NPZ so the
    # viewer can restore exact magnitudes for iCQT reconstruction.
    # Shards are file-backed via the budget handle.  All reads go through
    # budget.iter_frame_chunks() / scan_abs_max() — no full array in RAM.
    # When save_np_dtype differs from compute dtype, budget.convert_shards()
    # rewrites them in-place (streaming); after that they are always
    # registered as pre_sharded regardless of dtype match.
    _real_comp_dt = np.float64 if cfg.cqt_compute_dtype == "float64" else np.float32
    _dtype_match = (save_np_dtype == _real_comp_dt)

    if save_np_dtype == np.float16:
        # Scan the absolute maximum across all channels in column-chunks.
        # Replaces float(np.abs(cqt_L_real).max()) which materialised the
        # entire shard in RAM.
        _scan_channels = tuple(handle.channels)
        cqt_scale = _budget.scan_abs_max(handle, _scan_channels) or 1.0
    else:
        cqt_scale = None

    def _to_aux_save(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=save_np_dtype)

    save_dict: dict = dict(
        freqs=_to_aux_save(freqs),
        times=_to_aux_save(times),
        pad_mask=pad_mask,
        pad_influence=(_to_aux_save(pad_influence)
                       if pad_influence is not None
                       else np.empty((0, 0), dtype=save_np_dtype)),
        semitone_mask=semitone_mask,
        note_names=np.array(note_names, dtype="U12"),
        sr=np.int32(sr),
        bins_per_octave=np.int32(cfg.bins_per_octave),
        hop_length=np.int32(cfg.hop_length),
        stft_n_fft=np.int32(cfg.stft_n_fft or 0),
        fft_algorithm=np.array(cfg.fft_algorithm),
        cqt_grid_hop_length=np.int32(cfg.cqt_grid_hop_length or cfg.hop_length),
        cqt_filter_scale=save_np_dtype(cfg.cqt_filter_scale),
        cqt_window=np.array(cfg.cqt_window),
        cqt_fmin=save_np_dtype(cfg.cqt_fmin),
        cqt_fmax=save_np_dtype(cfg.cqt_fmax),
        color_gamma=save_np_dtype(cfg.color_gamma),
        image_dpi=np.int32(cfg.image_dpi),
        seconds_per_inch=save_np_dtype(cfg.seconds_per_inch),
        auto_fmin=save_np_dtype(auto_fmin),
        fmax_actual=save_np_dtype(fmax_actual),
        bass_fmax=save_np_dtype(bass_fmax),
        treble_fmin=save_np_dtype(treble_fmin),
        trim_pad=np.bool_(args.trim_pad),
        is_stereo=np.bool_(is_stereo),
        bulge_min=save_np_dtype(cfg.bulge_min),
        bulge_max=save_np_dtype(cfg.bulge_max),
        wav_path=np.array(wav_path),
        settings_hash=np.array(shash),
    )
    # If the save dtype differs from compute dtype, convert the shards in-place
    # via the budget (streaming, row-chunk by row-chunk, atomic tmp→rename).
    # After conversion they are in final save dtype and can be registered as
    # pre_sharded — no second write pass is needed.
    # budget.convert_shards() closes the source mmaps in handle and re-opens
    # the converted files read-only, so handle remains valid after this call.
    if not _dtype_match:
        _budget.convert_shards(
            handle,
            out_dtype=save_np_dtype,
            scale=cqt_scale if save_np_dtype == np.float16 else None,
        )
    _pre_sharded_keys: set[str] = {"real_left", "imag_left"}
    if is_stereo:
        _pre_sharded_keys.update({"real_right", "imag_right"})

    # Only store the right channel when the source is genuinely stereo.
    # The viewer aliases right = left on load when real_right is absent.
    # (stereo + dtype mismatch is handled above; stereo + dtype match: flag already set)

    # For float16 saves, persist the scale factor so the viewer can restore
    # exact magnitudes for display and iCQT reconstruction.
    if cqt_scale is not None:
        save_dict["cqt_scale"] = np.float64(cqt_scale)
    if cfg.cqt_bpo_schedule:
        save_dict["cqt_bpo_schedule"] = np.asarray(
            cfg.cqt_bpo_schedule, dtype=np.int32)
    if cfg.cqt_hop_schedule:
        save_dict["cqt_hop_schedule"] = np.asarray(
            cfg.cqt_hop_schedule, dtype=np.int32)
    if cfg.cqt_filter_scale_schedule:
        save_dict["cqt_filter_scale_schedule"] = np.asarray(
            cfg.cqt_filter_scale_schedule, dtype=save_np_dtype)
    # Heterodyne and prefilter metadata — stored so the viewer can correctly
    # label the frequency axis with physical (pre-heterodyne) frequencies.
    save_dict["heterodyne_hz"] = np.float64(cfg.heterodyne_hz)
    if cfg.prefilter_bp_fmin is not None:
        save_dict["prefilter_bp_fmin"] = np.float64(cfg.prefilter_bp_fmin)
    if cfg.prefilter_bp_fmax is not None:
        save_dict["prefilter_bp_fmax"] = np.float64(cfg.prefilter_bp_fmax)
    _emit_progress(0.80, "save_stream")
    manifest_name, _saved_stream_dir = _save_cqt_stream_bundle(
        outdir, shash, save_dict,
        pre_sharded=frozenset(_pre_sharded_keys) if _pre_sharded_keys else None,
    )
    # Legacy _ctmp_* cleanup (was used by an older shard helper; no longer created).
    for _ctag in (["left"] + (["right"] if is_stereo else [])):
        for _cpart in ("real", "imag"):
            _ctmp = os.path.join(_shard_dir, f"_ctmp_{_ctag}_{_cpart}.npy")
            if os.path.isfile(_ctmp):
                try:
                    os.remove(_ctmp)
                except OSError:
                    pass
    # Tiny pointer NPZ: keeps legacy discovery path while data lives in shards.
    np.savez(
        os.path.join(outdir, versioned_npz),
        stream_manifest=np.array(manifest_name),
        settings_hash=np.array(shash),
        fft_algorithm=np.array(cfg.fft_algorithm),
        sr=np.int32(sr),
        bins_per_octave=np.int32(cfg.bins_per_octave),
        wav_path=np.array(wav_path),
        is_stereo=np.bool_(is_stereo),
    )
    # Also write a legacy symlink / copy so old code can find cqt_data.npz
    legacy_path = os.path.join(outdir, "cqt_data.npz")
    if os.path.isfile(legacy_path) or os.path.islink(legacy_path):
        os.remove(legacy_path)
    try:
        os.symlink(versioned_npz, legacy_path)
    except OSError:
        shutil.copy2(os.path.join(outdir, versioned_npz), legacy_path)
    # Close the budget: flushes the write worker and releases all mmap handles.
    handle.close()
    _budget.close()
    del handle, _budget
    _mark_stage("cqt_saved")

    # --- Update versions manifest + modern inventory ---
    _update_manifest(outdir, shash, cfg, args)
    _write_analysis_inventory(
        outdir,
        shash,
        cfg,
        wav_path,
        total_duration_s,
        region_start_s,
        region_end_s,
    )

    # Composite audio is NOT saved as WAV/FLAC — the complex CQT shards in
    # the stream directory are the sole audio representation on disk.
    # Use iCQT reconstruction in the viewer if playback is needed.

    # --- Write CSV / TXT outputs (QUARANTINED — metrics not computed) ---
    # _emit_progress(0.94, "write_reports")
    # write_metrics_csv(os.path.join(outdir, "frame_metrics.csv"), times, metrics)
    # write_peaks_csv(os.path.join(outdir, "detected_bulges.csv"), times, metrics, peaks_idx)
    # write_summary_txt(os.path.join(outdir, "summary.txt"), wav_path, sr, x_mono,
    #                   times, metrics, peaks_idx, cfg)
    # _mark_stage("reports_written")

    _emit_progress(1.00, "complete")
    print(f"\nAnalysis complete. Outputs written to: {outdir}")
    print(f"  {versioned_npz}  stream manifest pointer (settings hash: {shash})")
    print(f"  itinerary_{shash}.json  stage completion log (for resume)")
    print("  cqt_data.npz            (points to latest version)")
    print("\nRun bass_plot.py to generate spectrogram images from the saved data.")


if __name__ == "__main__":
    main()
