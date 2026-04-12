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
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

try:
    import soundfile as _sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False
    from scipy.io import wavfile


EPS = 1e-12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# Settings hash — deterministic fingerprint of the parameters that shape
# the CQT output so that multiple analyses can coexist in one folder.
# ---------------------------------------------------------------------------

def settings_hash(cfg: "AnalysisConfig", composite_spec: str | None = None,
                  gap_seconds: float = 0.5, trim_pad: bool = False) -> str:
    """Return an 8-char hex digest uniquely identifying the analysis settings.

    Only the parameters that change the CQT tensor output are included.
    Display-only parameters (image_dpi, seconds_per_inch, color_gamma) are
    intentionally excluded.
    """
    canonical = {
        "hop_length": cfg.hop_length,
        "bins_per_octave": cfg.bins_per_octave,
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
    }
    if composite_spec:
        canonical["composite"] = composite_spec
        canonical["gap_seconds"] = round(gap_seconds, 4)
    if trim_pad:
        canonical["trim_pad"] = True
    blob = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    hop_length: int = 512           # CQT hop in samples (~11.6 ms at 44.1 kHz)
    bins_per_octave: int = 1200     # 1 cent per bin
    cqt_fmin: float = 16.35        # lowest CQT bin — C0, lowest piano/organ note
    cqt_fmax: float = 20000.0      # highest CQT bin (capped at Nyquist)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a WAV or MP3 file using a Constant-Q Transform with 1-cent frequency resolution."
    )
    parser.add_argument("wav_path", help="Path to input audio file (.wav or .mp3)")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--hop-length", type=int, default=512,
                        help="CQT hop length in samples (default 512)")
    parser.add_argument("--bins-per-octave", type=int, default=1200,
                        help="CQT bins per octave; 1200 = 1 cent/bin (default 1200)")
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
    parser.add_argument("--start-time", type=float, default=None,
                        help="Start time in seconds for partial analysis. "
                             "If omitted, analysis starts from the beginning.")
    parser.add_argument("--end-time", type=float, default=None,
                        help="End time in seconds for partial analysis. "
                             "If omitted, analysis runs to the end.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


# ---------------------------------------------------------------------------
# Constant-Q Transform
# ---------------------------------------------------------------------------

def compute_cqt(x: np.ndarray, sr: int, cfg: AnalysisConfig, zero_front: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CQT power spectrogram via torch_cqt_new.

    Delegates to the production-tested ``torch_cqt_new.cqt`` engine —
    the single authoritative CQT implementation in this codebase.

    Returns:
      freqs       – center Hz for each bin,    shape (n_bins,)
      times       – center time per frame,     shape (n_frames,)
      power       – magnitude squared,         shape (n_bins, n_frames)  float32
      cqt_complex – complex CQT coefficients,  shape (n_bins, n_frames)  complex64
    """
    from torch_cqt_new import cqt as _torch_cqt

    if cfg.cqt_fmin <= 0:
        raise ValueError(f"--cqt-fmin must be > 0 Hz (got {cfg.cqt_fmin}); try 5 for infrasonic sub-bass.")
    fmax = min(cfg.cqt_fmax, sr / 2.0 * 0.95)
    n_bins = int(math.floor(cfg.bins_per_octave * math.log2(fmax / cfg.cqt_fmin)))
    hop = cfg.hop_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.as_tensor(np.asarray(x), device=device)

    resample_kw = {
        "lowpass_filter_width": cfg.resample_taps,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 14.769656459379492,
    }

    C, freqs_t = _torch_cqt(
        y, sr,
        hop_length=hop,
        fmin=cfg.cqt_fmin,
        n_bins=n_bins,
        bins_per_octave=cfg.bins_per_octave,
        pad_mode="constant" if zero_front else "reflect",
        device=device,
        resample_kw=resample_kw,
    )

    freqs = freqs_t.cpu().numpy()
    cqt_np = C.cpu().numpy()

    power = (np.abs(cqt_np) ** 2).clip(min=EPS).astype(np.float32)
    cqt_complex = cqt_np.astype(np.complex64)

    n_frames = cqt_complex.shape[-1]
    times = np.arange(n_frames) * (hop / sr)
    return freqs, times, power, cqt_complex


# ---------------------------------------------------------------------------
# Time-domain onset enhancement
# ---------------------------------------------------------------------------

def compute_onset_envelope(
    x: np.ndarray, sr: int, freqs: np.ndarray, hop: int, n_frames: int,
) -> np.ndarray:
    """Compute a per-bin, per-frame onset strength from short-window time-domain
    energy, to sharpen transient edges that the CQT smears.

    For each CQT frequency bin, we bandpass-filter the signal around that
    frequency (±1 semitone), compute a short-window RMS envelope (window =
    2 * hop samples for sub-frame resolution), then take the half-wave-rectified
    first difference as onset strength.

    Returns onset strength matrix (n_bins, n_frames), values ≥ 0, normalised
    per-bin to [0, 1].  This can be additively blended with the CQT power
    to enhance transient clarity without altering steady-state tones.
    """
    device = torch.device("cuda")
    n_bins = len(freqs)
    onset = np.zeros((n_bins, n_frames), dtype=np.float32)
    sig = torch.from_numpy(x.astype(np.float32)).to(device)

    # Short RMS window: 2× hop gives one sub-division per CQT frame.
    rms_win = 2 * hop
    # Process in semitone-wide groups (100 bins at 1200 bpo) to keep
    # the bandpass filter count manageable.  Bins within a semitone share
    # the same bandpass, and we spread the result across them.
    semitone_cents = 100
    bins_per_semitone = max(1, int(round(1200.0 / (1200.0 / len(freqs) * (1200.0 / 1200.0)))))
    # Simpler: group by semitone index.
    midi_vals = 12.0 * np.log2(np.clip(freqs, 1e-9, None) / 440.0) + 69.0
    semitone_ids = np.round(midi_vals).astype(int)
    unique_semitones = np.unique(semitone_ids)

    for semi in unique_semitones:
        bin_indices = np.flatnonzero(semitone_ids == semi)
        center_freq = 440.0 * 2.0 ** ((semi - 69.0) / 12.0)
        # Bandpass: ±1 semitone.
        f_lo = center_freq * 2.0 ** (-1.0 / 12.0)
        f_hi = center_freq * 2.0 ** (1.0 / 12.0)
        nyq = sr / 2.0
        if f_hi >= nyq or f_lo <= 0:
            continue
        # Design bandpass on CPU (tiny), apply on GPU via FFT.
        from scipy.signal import butter, sosfilt
        try:
            sos = butter(4, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
        except ValueError:
            continue
        # Apply filter on CPU (sosfilt isn't on GPU, but it's a tiny 8th-order
        # IIR on a 1-D signal — negligible time vs the FFT convolutions).
        filtered = sosfilt(sos, x).astype(np.float32)

        # RMS envelope via GPU conv1d with a squared signal.
        fsig = torch.from_numpy(filtered ** 2).to(device)
        box = torch.ones(1, 1, rms_win, device=device) / rms_win
        rms_sq = torch.nn.functional.conv1d(
            fsig.view(1, 1, -1), box, padding=rms_win // 2
        ).view(-1)
        # Hop-sample to CQT frame grid.
        frame_indices = torch.arange(n_frames, device=device) * hop
        frame_indices = frame_indices.clamp(max=rms_sq.shape[0] - 1)
        rms_frames = rms_sq[frame_indices].cpu().numpy()
        # Half-wave rectified first difference = onset strength.
        onset_str = np.maximum(np.diff(rms_frames, prepend=rms_frames[0]), 0.0)
        # Normalise per-semitone.
        peak = onset_str.max()
        if peak > 0:
            onset_str /= peak
        # Write to all bins in this semitone.
        for bi in bin_indices:
            onset[bi] = onset_str

    del sig
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
    cents_per_bin = 1200.0 / bins_per_octave
    midi_frac = 12.0 * np.log2(np.clip(freqs, 1e-9, None) / 440.0) + 69.0
    cents_from_semitone = (midi_frac * 100.0) % 100.0
    on_semitone = np.minimum(cents_from_semitone, 100.0 - cents_from_semitone) < (cents_per_bin / 2.0)
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
    return np.convolve(x, np.ones(width) / width, mode="same")


def weighted_centroid(freqs: np.ndarray, power_band: np.ndarray) -> np.ndarray:
    return (freqs[:, None] * power_band).sum(axis=0) / (power_band.sum(axis=0) + EPS)


def peak_frequency(freqs: np.ndarray, power_band: np.ndarray) -> np.ndarray:
    return freqs[np.argmax(power_band, axis=0)]


def rolloff_pair(
    freqs: np.ndarray, power_band: np.ndarray, low_q: float = 0.15, high_q: float = 0.85
) -> Tuple[np.ndarray, np.ndarray]:
    csum = np.cumsum(power_band, axis=0)
    total = csum[-1, :] + EPS
    lo = np.clip(np.argmax(csum >= (total * low_q)[None, :], axis=0), 0, len(freqs) - 1)
    hi = np.clip(np.argmax(csum >= (total * high_q)[None, :], axis=0), 0, len(freqs) - 1)
    return freqs[lo], freqs[hi]


def track_edge_frequencies(
    freqs: np.ndarray, power: np.ndarray, edge_threshold_db: float
) -> Tuple[np.ndarray, np.ndarray]:
    rel_ratio = 10.0 ** (edge_threshold_db / 10.0)
    threshold = (np.max(power, axis=0) + EPS) * rel_ratio
    active = power >= threshold[None, :]
    any_active = active.any(axis=0)
    low = np.full(power.shape[1], np.nan)
    high = np.full(power.shape[1], np.nan)
    low_idx = np.argmax(active, axis=0)
    high_idx = active.shape[0] - 1 - np.argmax(active[::-1], axis=0)
    low[any_active] = freqs[low_idx[any_active]]
    high[any_active] = freqs[high_idx[any_active]]
    return low, high


def db_power(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, EPS))


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
    bass_mask = band_mask(freqs, cfg.bass_min, cfg.bass_max)
    kick_mask = band_mask(freqs, cfg.kick_min, cfg.kick_max)
    bulge_mask = band_mask(freqs, cfg.bulge_min, cfg.bulge_max)

    # When CQT fmin is above the bass/kick/bulge range (e.g. Hybrid mode
    # or narrow partial analysis), produce zeroed metrics instead of crashing.
    has_bass = np.any(bass_mask)
    has_kick = np.any(kick_mask)
    has_bulge = np.any(bulge_mask)

    if not has_bass:
        print("  Warning: bass band outside CQT range — metrics zeroed")
    if not has_kick:
        print("  Warning: kick band outside CQT range — metrics zeroed")
    if not has_bulge:
        print("  Warning: bulge band outside CQT range — metrics zeroed")

    zeros = np.zeros(n_frames, dtype=np.float64)
    nans = np.full(n_frames, np.nan, dtype=np.float64)

    if has_bass:
        bass_power = power[bass_mask, :]
        bass_energy_db = db_power(bass_power.sum(axis=0))
        bass_delta = np.diff(bass_power, axis=1, prepend=bass_power[:, :1])
        bass_flux_db = db_power(np.maximum(bass_delta, 0.0).sum(axis=0) + EPS)
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
        f"Hop length: {cfg.hop_length} samples  ({cfg.hop_length / sr * 1000:.1f} ms)",
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
    """Load the versions manifest, returning ``{"versions": {...}}``."""
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
            "bins_per_octave": cfg.bins_per_octave,
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
    cfg = AnalysisConfig(
        hop_length=args.hop_length,
        bins_per_octave=args.bins_per_octave,
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
    )

    wav_path = os.path.abspath(args.wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    outdir = args.outdir or os.path.join(os.path.dirname(wav_path), f"{base_name}_analysis")
    ensure_dir(outdir)

    # --- Load stereo ---
    sr, x_left, x_right = load_audio_stereo(wav_path)
    is_stereo = not np.array_equal(x_left, x_right)
    # Mono mix for analysis (metrics, onset).
    x_mono = (x_left + x_right) * 0.5

    # --- Time-region slicing (partial analysis) ---
    if args.start_time is not None or args.end_time is not None:
        total_samples = len(x_left)
        s0 = 0
        s1 = total_samples
        if args.start_time is not None:
            s0 = max(0, min(total_samples, int(args.start_time * sr)))
        if args.end_time is not None:
            s1 = max(s0, min(total_samples, int(args.end_time * sr)))
        if s1 > s0:
            x_left = x_left[s0:s1]
            x_right = x_right[s0:s1]
            x_mono = x_mono[s0:s1]
            print(f"  Partial analysis: {s0/sr:.3f}s – {s1/sr:.3f}s "
                  f"({s1-s0} samples of {total_samples})")

    # --- Compositing ---
    composite_regions = None
    if args.composite is not None:
        from composite import parse_composite_spec, build_composite
        audio_len_s = len(x_mono) / sr
        segments = parse_composite_spec(args.composite, audio_len_s)
        x_left, regions_L = build_composite(
            x_left, sr, segments, cfg.hop_length, args.gap_seconds)
        x_right, _ = build_composite(
            x_right, sr, segments, cfg.hop_length, args.gap_seconds)
        x_mono = (x_left + x_right) * 0.5
        composite_regions = regions_L
        for ri in composite_regions:
            print(f"  Region {ri.seg_idx}: original "
                  f"{ri.original_start_s:.3f}s – {ri.original_end_s:.3f}s "
                  f"→ composite {ri.composite_start_sample/sr:.3f}s – "
                  f"{ri.composite_end_sample/sr:.3f}s")

    use_zero_front = composite_regions is not None

    # --- CQT for both channels ---
    print("Computing CQT (left)...")
    freqs, times, power_L, cqt_L = compute_cqt(x_left, sr, cfg, zero_front=use_zero_front)
    print("Computing CQT (right)...")
    _, _, power_R, cqt_R = compute_cqt(x_right, sr, cfg, zero_front=use_zero_front)

    # Mono power for analysis / onset (average).
    power_mono = (power_L + power_R) * 0.5

    # --- Onset enhancement (on mono) ---
    print("Computing time-domain onset enhancement...")
    onset = compute_onset_envelope(x_mono, sr, freqs, cfg.hop_length, len(times))
    power_db = db_power(power_mono)
    per_bin_range = (np.max(power_db, axis=1) - np.min(power_db, axis=1))[:, None]
    boost_db = onset * np.clip(per_bin_range, 0, 6.0)
    boost_lin = 10.0 ** (boost_db / 10.0)
    power_L = power_L * boost_lin
    power_R = power_R * boost_lin
    power_mono = (power_L + power_R) * 0.5

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
        total_samples = len(x_mono)
        sample_is_pad = np.ones(total_samples, dtype=np.float32)
        for ri in composite_regions:
            s0 = max(0, ri.composite_start_sample)
            s1 = min(total_samples, ri.composite_end_sample)
            sample_is_pad[s0:s1] = 0.0
        cumsum = np.empty(total_samples + 1, dtype=np.float64)
        cumsum[0] = 0.0
        np.cumsum(sample_is_pad, out=cumsum[1:])

        Q_val = 1.0 / (2.0 ** (1.0 / cfg.bins_per_octave) - 1.0)
        kernel_half = np.ceil(Q_val * sr / freqs / 2.0).astype(np.int64)
        frame_centers = (np.arange(len(times)) * cfg.hop_length).astype(np.int64)

        n_bins = len(freqs)
        n_frames = len(times)
        pad_influence = np.empty((n_bins, n_frames), dtype=np.float32)
        chunk = 500
        for b0 in range(0, n_bins, chunk):
            b1 = min(b0 + chunk, n_bins)
            h = kernel_half[b0:b1, None]            # (chunk, 1)
            fc = frame_centers[None, :]              # (1, n_frames)
            starts = np.clip(fc - h, 0, total_samples)
            ends = np.clip(fc + h, 0, total_samples)
            wlen = (ends - starts).astype(np.float64)
            wsum = cumsum[ends] - cumsum[starts]
            pad_influence[b0:b1] = (wsum / np.maximum(wlen, 1.0)).astype(np.float32)
        del cumsum, sample_is_pad

    semitone_mask, note_names = semitone_mask_for_freqs(freqs, cfg.bins_per_octave)

    cents_per_bin = 1200.0 / cfg.bins_per_octave
    n_semitones = int(semitone_mask.sum())
    print(
        f"  {len(freqs)} bins  ({cents_per_bin:.2f}¢/bin)  "
        f"{n_semitones} semitones  "
        f"{len(times)} frames  "
        f"hop={cfg.hop_length / sr * 1000:.1f} ms"
        f"  {'stereo' if is_stereo else 'mono'}"
    )

    metrics, peaks_idx = analyze(freqs, times, power_mono, cfg)

    # Auto-trim: find the lowest frequency bin that actually carries energy
    # above the noise floor.
    bin_peak_db = db_power(np.max(power_mono, axis=1))
    noise_floor_db = float(np.median(bin_peak_db))
    active_bins = bin_peak_db > (noise_floor_db + 6.0)
    if np.any(active_bins):
        auto_fmin = float(freqs[np.flatnonzero(active_bins)[0]])
        auto_fmin = max(auto_fmin * 2.0 ** (-1.0 / 12.0), float(freqs[0]))
    else:
        auto_fmin = float(freqs[0])
    print(f"  Auto-trim: lowest active frequency ≈ {auto_fmin:.1f} Hz")

    duration_s = len(x_mono) / sr
    fig_width_in = max(8.0, duration_s / cfg.seconds_per_inch)

    fmax_actual = float(freqs[-1])
    bass_fmax = min(cfg.bass_view_max_freq, fmax_actual)
    treble_fmin = min(cfg.treble_view_min_freq, fmax_actual)

    print(f"  Duration {duration_s:.1f} s → figure width {fig_width_in:.1f} in "
          f"@ {cfg.seconds_per_inch} s/in  (γ={cfg.color_gamma})")

    # --- Save compressed CQT tensors + plotting metadata ---
    shash = settings_hash(cfg, args.composite,
                          getattr(args, 'gap_seconds', 0.5),
                          args.trim_pad)
    versioned_npz = f"cqt_data_{shash}.npz"
    print(f"Saving compressed CQT tensors ({versioned_npz}) ...")
    save_dict = dict(
        freqs=freqs.astype(np.float32),
        times=times.astype(np.float32),
        power_left=power_L.astype(np.float32),
        power_right=power_R.astype(np.float32),
        real_left=cqt_L.real.astype(np.float32),
        imag_left=cqt_L.imag.astype(np.float32),
        real_right=cqt_R.real.astype(np.float32),
        imag_right=cqt_R.imag.astype(np.float32),
        phase_left=np.angle(cqt_L).astype(np.float32),
        phase_right=np.angle(cqt_R).astype(np.float32),
        onset=onset.astype(np.float32),
        pad_mask=pad_mask,
        pad_influence=pad_influence if pad_influence is not None else np.empty((0, 0), dtype=np.float32),
        semitone_mask=semitone_mask,
        note_names=np.array(note_names, dtype="U12"),
        peaks_idx=peaks_idx.astype(np.int64),
        # Config scalars
        sr=np.int32(sr),
        bins_per_octave=np.int32(cfg.bins_per_octave),
        hop_length=np.int32(cfg.hop_length),
        color_gamma=np.float32(cfg.color_gamma),
        image_dpi=np.int32(cfg.image_dpi),
        seconds_per_inch=np.float32(cfg.seconds_per_inch),
        auto_fmin=np.float32(auto_fmin),
        fmax_actual=np.float32(fmax_actual),
        bass_fmax=np.float32(bass_fmax),
        treble_fmin=np.float32(treble_fmin),
        trim_pad=np.bool_(args.trim_pad),
        is_stereo=np.bool_(is_stereo),
        bulge_min=np.float32(cfg.bulge_min),
        bulge_max=np.float32(cfg.bulge_max),
        wav_path=np.array(wav_path),
        settings_hash=np.array(shash),
    )
    # Add metrics arrays
    for k, v in metrics.items():
        save_dict[f"metrics_{k}"] = v.astype(np.float32)
    np.savez_compressed(os.path.join(outdir, versioned_npz), **save_dict)
    # Also write a legacy symlink / copy so old code can find cqt_data.npz
    legacy_path = os.path.join(outdir, "cqt_data.npz")
    if os.path.isfile(legacy_path) or os.path.islink(legacy_path):
        os.remove(legacy_path)
    try:
        os.symlink(versioned_npz, legacy_path)
    except OSError:
        import shutil
        shutil.copy2(os.path.join(outdir, versioned_npz), legacy_path)
    del cqt_L, cqt_R  # free memory after saving

    # --- Update versions manifest ---
    _update_manifest(outdir, shash, cfg, args)

    # --- Save composited audio for viewer playback ---
    if composite_regions is not None:
        stereo = np.column_stack([x_left, x_right])
        stereo_i16 = (stereo * 32767).clip(-32768, 32767).astype(np.int16)
        composite_wav = os.path.join(outdir, "composite_audio.wav")
        wavfile.write(composite_wav, sr, stereo_i16)
        print(f"  composite_audio.wav     composited playback audio ({len(x_left)/sr:.3f} s)")
        del stereo, stereo_i16

    # --- Write CSV / TXT outputs ---
    write_metrics_csv(os.path.join(outdir, "frame_metrics.csv"), times, metrics)
    write_peaks_csv(os.path.join(outdir, "detected_bulges.csv"), times, metrics, peaks_idx)
    write_summary_txt(os.path.join(outdir, "summary.txt"), wav_path, sr, x_mono,
                      times, metrics, peaks_idx, cfg)

    print(f"\nAnalysis complete. Outputs written to: {outdir}")
    print(f"  {versioned_npz}  compressed tensors (settings hash: {shash})")
    print("  cqt_data.npz            (points to latest version)")
    print("  frame_metrics.csv")
    print("  detected_bulges.csv")
    print("  summary.txt")
    print("\nRun bass_plot.py to generate spectrogram images from the saved data.")


if __name__ == "__main__":
    main()
