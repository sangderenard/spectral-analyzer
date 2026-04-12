"""TF-plane surgery: affine transformation of rectangular time-frequency regions.

Two engines:

**STFT engine** (``tf_transplant``):
  Best for moderate frequency ratios (< ~4×) within the audible range.
  High temporal resolution, proper IF phase vocoder.

**CWT engine** (``cwt_transplant``):
  Best for large frequency ratios and subsonic→sonic work (whale song,
  seismic, infrasound).  Log-frequency axis means a 20× frequency ratio
  is a 4.3-octave *translation* — no interpolation stretching.  Natural
  adaptive resolution: long windows at low frequencies, short at high.

**Dispatcher** (``transplant``):
  Automatically selects the appropriate engine based on frequency ratio
  and region boundaries.

Both engines accept the same TFRegion + affine matrix interface.
Both return ``TransplantResult`` carrying the signal, the engine used,
and an **uncertainty envelope** — because the Heisenberg–Gabor limit
means there is no single authoritative reconstruction; the envelope
quantifies the time-frequency smearing inherent in the transform.

All operations are pure torch — GPU-ready, dtype-preserving.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from crossover import lr_decompose, lr_reconstruct


# ═══════════════════════════════════════════════════════════════════════════
# Public data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TFRegion:
    """A rectangle in time-frequency space.

    Coordinates are in physical units (seconds, Hz).
    """
    t0: float           # start time (seconds)
    t1: float           # end time (seconds)
    f0: float           # low frequency (Hz)
    f1: float           # high frequency (Hz)

    def __post_init__(self):
        if self.t0 > self.t1:
            self.t0, self.t1 = self.t1, self.t0
        if self.f0 > self.f1:
            self.f0, self.f1 = self.f1, self.f0


@dataclass
class TransplantResult:
    """Result of a TF transplant operation with uncertainty metadata.

    The ``signal`` is the best-effort reconstruction.  The
    ``uncertainty_envelope`` quantifies how much temporal smearing
    the transform introduces at each sample — larger values mean the
    local time-frequency content is less precisely localised.

    For the STFT engine this is derived from the window length.
    For the CWT engine it is the wavelet time-spread σ_t(f) averaged
    over the destination frequency range, sampled at each output sample.
    """
    signal: torch.Tensor
    engine: Literal["stft", "cwt"]
    uncertainty_envelope: torch.Tensor | None = None
    dst_region: TFRegion | None = None
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════


def rejection_margin(order: int, rejection_db: float = 60.0) -> float:
    """Frequency ratio from crossover point to ``rejection_db`` attenuation.

    For an LR transfer function ``H(f) = 1/(1 + (f/fc)^order)``, returns
    the factor ``k`` such that ``H(k·fc) <= 10^(-rejection_db/20)``.

    Place the crossover at ``f_edge / k`` (lowpass) or ``f_edge * k``
    (highpass) to guarantee ``rejection_db`` of isolation at ``f_edge``.
    """
    eps = 10.0 ** (-rejection_db / 20.0)
    return (1.0 / eps - 1.0) ** (1.0 / order)


@dataclass
class WorkingRange:
    """Signal-aware frequency and time limits for TF operations.

    All limits are derived from the physical signal parameters — sample rate
    and duration — so the viewer never tries to work outside what the data
    can actually resolve.
    """
    sr: int
    duration: float
    nyquist: float              # sr / 2
    f_min_cwt: float            # lowest CWT-resolvable frequency
    f_min_stft: float           # lowest STFT-resolvable frequency (at n_fft_max)
    f_min: float                # practical floor = min(f_min_cwt, f_min_stft)
    f_max: float                # = nyquist
    n_octaves: float            # log2(f_max / f_min)
    n_octaves_subsonic: float   # octaves below 20 Hz (0 if f_min >= 20)
    stft_regime: TFRegion       # region where STFT engine is appropriate
    cwt_regime: TFRegion        # region where CWT engine is appropriate


def working_range(
    sr: int,
    duration: float,
    *,
    min_cycles_cwt: float = 4.0,
    n_fft_max: int = 2 ** 18,
    cwt_sigma: float = 6.0,
) -> WorkingRange:
    """Compute the resolvable frequency range from signal parameters.

    Parameters
    ----------
    sr : int
        Sample rate (Hz).
    duration : float
        Signal length (seconds).
    min_cycles_cwt : float
        Minimum number of wavelet cycles to consider a frequency resolved.
        The CWT floor is ``min_cycles_cwt / duration``.
    n_fft_max : int
        Largest STFT window the system will use.  The STFT floor is
        ``sr / n_fft_max``.
    cwt_sigma : float
        Morlet σ parameter.  Used for the CWT floor estimate:
        ``cwt_sigma / (π * duration)`` — requires the wavelet's
        Gaussian envelope to fit within the signal.

    Returns
    -------
    WorkingRange
        Physical limits on what the signal can resolve.
    """
    nyquist = sr / 2.0

    # CWT: need the wavelet envelope to fit in the signal
    # For Morlet: effective support ≈ 2 * sigma / (2π f) seconds
    # Require that < duration → f > sigma / (π * duration)
    f_min_cwt = max(cwt_sigma / (math.pi * duration), min_cycles_cwt / duration)

    # STFT: frequency resolution = sr / n_fft → lowest bin centre
    f_min_stft = sr / n_fft_max

    f_min = min(f_min_cwt, f_min_stft)
    f_max = nyquist

    if f_min >= f_max:
        f_min = f_max / 2  # degenerate: at least one octave

    n_octaves = math.log2(f_max / f_min) if f_min > 0 else 0.0
    n_octaves_subsonic = max(0.0, math.log2(20.0 / f_min)) if f_min < 20 else 0.0

    stft_regime = TFRegion(t0=0, t1=duration, f0=max(f_min_stft, 20.0), f1=f_max)
    cwt_regime = TFRegion(t0=0, t1=duration, f0=f_min_cwt, f1=min(f_max, 200.0))

    return WorkingRange(
        sr=sr,
        duration=duration,
        nyquist=nyquist,
        f_min_cwt=f_min_cwt,
        f_min_stft=f_min_stft,
        f_min=f_min,
        f_max=f_max,
        n_octaves=n_octaves,
        n_octaves_subsonic=n_octaves_subsonic,
        stft_regime=stft_regime,
        cwt_regime=cwt_regime,
    )


def tf_transplant(
    signal: torch.Tensor,
    sr: int,
    src: TFRegion,
    affine: torch.Tensor,
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
    min_bins_in_region: int = 128,
    n_fft_max: int = 2 ** 18,
    crossover_order: int = 8,
    rejection_db: float = 60.0,
    dampen_source: bool = False,
    fade_bins: int = 4,
) -> torch.Tensor:
    """Apply an affine transformation to a rectangular TF region.

    Parameters
    ----------
    signal : (n_samples,) or (..., n_samples)
        Input waveform.
    sr : int
        Sample rate.
    src : TFRegion
        Source rectangle in (seconds, Hz).
    affine : Tensor (2, 3)
        Affine matrix mapping normalised source coords ``[t_norm, f_norm, 1]``
        to normalised destination coords ``[t', f']``.
        Identity = ``[[1,0,0],[0,1,0]]`` → no change.
    n_fft : int or None
        STFT window size.  If None (default), auto-computed so the
        source region spans at least *min_bins_in_region* frequency bins.
    hop_length : int or None
        STFT hop.  Default ``n_fft // 4``.
    min_bins_in_region : int
        Target minimum number of STFT bins across the source frequency
        range (used when n_fft is auto-computed).
    n_fft_max : int
        Upper bound on auto-computed n_fft.
    crossover_order : int
        LR crossover steepness for band isolation.
    rejection_db : float
        Required stopband rejection (dB) for the crossover margin.
    dampen_source : bool
        If True, the source region is faded to silence.
    fade_bins : int
        Width of the raised-cosine fade at all TF region edges.

    Returns
    -------
    Tensor  (..., n_samples)
        Modified signal with the TF region transplanted.
    """
    # ── Degenerate source region → pass through unchanged ──
    if src.f1 - src.f0 < 1.0 or src.t1 - src.t0 < 1e-6:
        return signal

    # ── Auto-size n_fft for maximum resolution in the working region ──
    f_span = src.f1 - src.f0
    if n_fft is None:
        # Target: min_bins_in_region bins across f_span
        # bins_in_span = f_span / (sr / n_fft) = f_span * n_fft / sr
        # ⇒ n_fft = min_bins_in_region * sr / f_span
        n_fft_needed = int(math.ceil(min_bins_in_region * sr / f_span))
        n_fft = 1 << math.ceil(math.log2(max(n_fft_needed, 256)))
        n_fft = min(n_fft, n_fft_max)
    # Ensure n_fft doesn't exceed twice the signal length (reflect padding limit)
    max_for_signal = signal.shape[-1] * 2
    if n_fft > max_for_signal:
        n_fft = 1 << int(math.log2(max(max_for_signal, 256)))
    if hop_length is None:
        hop_length = n_fft // 4

    nyq = sr / 2.0
    N = signal.shape[-1]
    lead = signal.shape[:-1]
    flat = signal.reshape(-1, N) if lead else signal.unsqueeze(0)
    B = flat.shape[0]
    dtype = signal.dtype
    device = signal.device
    cdtype = _to_complex(dtype)

    affine = affine.to(dtype=dtype, device=device)

    # ── Determine destination region from affine ──
    dst = _affine_dst_region(src, affine, nyq)

    # ── Compute the affine output bounding box in source-normalised coords ──
    #    (needed for the inverse-map grid)
    corners = torch.tensor(
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=affine.dtype, device=affine.device,
    )
    mapped = corners @ affine.T  # (4, 2)  in source-normalised space
    mapped_t_min = float(mapped[:, 0].min())
    mapped_t_max = float(mapped[:, 0].max())
    mapped_f_min = float(mapped[:, 1].min())
    mapped_f_max = float(mapped[:, 1].max())

    # ── Crossover isolation with epsilon margin ──
    margin = rejection_margin(crossover_order, rejection_db)
    f_lo = min(src.f0, dst.f0)
    f_hi = max(src.f1, dst.f1)
    xo_lo = f_lo / margin  # crossover placed here; at f_lo it's down by rejection_db
    xo_hi = f_hi * margin  # at f_hi it's down by rejection_db
    xo = [f for f in [xo_lo, xo_hi] if 0 < f < nyq]
    if not xo:
        xo = [f for f in [f_lo, f_hi] if 0 < f < nyq]

    if not xo:
        return signal

    bands = lr_decompose(flat, xo, sr, order=crossover_order)
    # bands: (B, n_bands, N)
    # The working band is the one(s) that overlap [f_lo, f_hi].
    # With two crossovers xo_lo < xo_hi, the middle band (index 1) is the
    # working band; bands 0 and 2 pass through untouched.
    n_bands = bands.shape[-2]
    if n_bands == 3:
        work_idx = 1
    elif n_bands == 2:
        # Only one crossover survived clipping — figure out which side
        work_idx = 0 if xo[0] >= f_hi else 1 if xo[0] <= f_lo else 0
    else:
        work_idx = 0

    work_band = bands[:, work_idx, :]  # (B, N)

    # ── STFT the working band ──
    window = torch.hann_window(n_fft, dtype=dtype, device=device)
    spec = _stft_hann(work_band, n_fft, hop_length, window)
    # spec: (B, n_bins, n_frames) complex
    n_bins = spec.shape[-2]
    n_frames = spec.shape[-1]

    # ── Convert physical TF coordinates to bin/frame indices ──
    src_t0_fr = max(0, int(src.t0 * sr / hop_length))
    src_t1_fr = min(n_frames, int(math.ceil(src.t1 * sr / hop_length)))
    src_f0_bn = max(0, int(src.f0 / nyq * n_bins))
    src_f1_bn = min(n_bins, int(math.ceil(src.f1 / nyq * n_bins)))

    dst_t0_fr = max(0, int(dst.t0 * sr / hop_length))
    dst_t1_fr = min(n_frames, int(math.ceil(dst.t1 * sr / hop_length)))
    dst_f0_bn = max(0, int(dst.f0 / nyq * n_bins))
    dst_f1_bn = min(n_bins, int(math.ceil(dst.f1 / nyq * n_bins)))

    dst_h = dst_f1_bn - dst_f0_bn
    dst_w = dst_t1_fr - dst_t0_fr
    if dst_h <= 0 or dst_w <= 0:
        return signal

    # ── Build sampling grid ──
    # For each pixel (i, j) in the destination rectangle, we need to find
    # the continuous (f, t) coordinates in the source spectrogram.
    #
    # The affine maps source-normalised [0,1]² → output coords.
    # The destination bounding box spans [mapped_t_min, mapped_t_max] ×
    # [mapped_f_min, mapped_f_max] in the affine's output coordinate system.
    # We generate a grid in that system, then inverse-map to source [0,1]².

    inv_affine = _affine_2x3_inverse(affine)

    # Destination pixel grid → affine output coordinates
    dst_t_aff = torch.linspace(mapped_t_min, mapped_t_max, dst_w,
                               dtype=dtype, device=device)
    dst_f_aff = torch.linspace(mapped_f_min, mapped_f_max, dst_h,
                               dtype=dtype, device=device)
    grid_f, grid_t = torch.meshgrid(dst_f_aff, dst_t_aff, indexing="ij")
    # (dst_h, dst_w) — these are in the affine's output coordinate system

    ones = torch.ones_like(grid_t)
    coords = torch.stack([grid_t, grid_f, ones], dim=-1)  # (H, W, 3)
    src_coords = torch.matmul(coords, inv_affine.T)  # (H, W, 2)
    src_t_norm = src_coords[..., 0]  # [0,1] in source time
    src_f_norm = src_coords[..., 1]  # [0,1] in source freq

    # Normalised source coords → continuous spectrogram pixel coords
    src_t_px = src_t_norm * (src_t1_fr - src_t0_fr - 1) + src_t0_fr
    src_f_px = src_f_norm * (src_f1_bn - src_f0_bn - 1) + src_f0_bn

    # ── Sample source spectrogram via bilinear interpolation ──
    sampled = _bilinear_complex(spec, src_f_px, src_t_px)
    # sampled: (B, dst_h, dst_w) complex

    # ── Phase vocoder correction ──
    dst_f_px = torch.linspace(dst_f0_bn, dst_f1_bn - 1, dst_h,
                              dtype=dtype, device=device).unsqueeze(-1)
    sampled = _phase_correct(sampled, spec, src_f_px, src_t_px, dst_f_px,
                             hop_length, n_fft)

    # ── Modify the spectrogram ──
    out_spec = spec.clone()

    if dampen_source:
        mask = _smooth_mask(n_bins, n_frames,
                            src_f0_bn, src_f1_bn,
                            src_t0_fr, src_t1_fr,
                            fade_bins=fade_bins,
                            device=device, dtype=dtype)
        out_spec = out_spec * (1.0 - mask).unsqueeze(0)

    # Add transformed content at destination with fade-in at edges
    dst_mask = _smooth_mask(n_bins, n_frames,
                            dst_f0_bn, dst_f0_bn + dst_h,
                            dst_t0_fr, dst_t0_fr + dst_w,
                            fade_bins=fade_bins,
                            device=device, dtype=dtype)
    dst_window = dst_mask[dst_f0_bn:dst_f0_bn + dst_h,
                          dst_t0_fr:dst_t0_fr + dst_w]  # (dst_h, dst_w)
    out_spec[:, dst_f0_bn:dst_f0_bn + dst_h,
             dst_t0_fr:dst_t0_fr + dst_w] += sampled * dst_window.unsqueeze(0)

    # ── iSTFT the modified working band ──
    mod_band = _istft_hann(out_spec, n_fft, hop_length, window, length=N)

    # ── Reconstruct: replace the working band, keep others pristine ──
    out_bands = bands.clone()
    out_bands[:, work_idx, :] = mod_band
    result = lr_reconstruct(out_bands)

    if lead:
        result = result.reshape(*lead, N)
    else:
        result = result.squeeze(0)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CWT-based transplant — for large frequency ratios and subsonic work
# ═══════════════════════════════════════════════════════════════════════════


def cwt_transplant(
    signal: torch.Tensor,
    sr: int,
    src: TFRegion,
    affine: torch.Tensor,
    *,
    scales_per_octave: int = 48,
    wavelet: str = "morlet",
    wavelet_kw: dict | None = None,
    epsilon: float | None = None,
    hop_length: int = 1,
    dampen_source: bool = False,
    batch_size: int = 64,
    n_fft_max: int = 2 ** 20,
    fade_scales: int = 4,
) -> TransplantResult:
    """CWT-based TF transplant for arbitrary frequency ratios.

    Uses the Continuous Wavelet Transform where frequency is a
    log-spaced axis — a multiplicative frequency shift becomes a
    *translation* in scale index space.  This is ideal for subsonic→sonic
    casting (e.g. 1–20 Hz → 20–400 Hz) where the STFT engine would
    have only a handful of bins and frames.

    The CWT provides natural adaptive resolution: long windows for low
    frequencies (resolving individual cycles of 1 Hz signals) and short
    windows at higher frequencies.  The price is that the inverse CWT is
    *not* perfectly reconstructing in general — reconstruction quality
    depends on the wavelet and scale coverage.  The returned
    ``TransplantResult.uncertainty_envelope`` quantifies the local
    temporal smearing (Heisenberg σ_t) at each output sample.

    Parameters
    ----------
    signal : (n_samples,) or (..., n_samples)
    sr : int
    src : TFRegion
    affine : Tensor (2, 3)
        Same convention as ``tf_transplant``.
    scales_per_octave : int
        CWT frequency resolution.  48 = 1/8-tone.  Higher is better
        for fractional-octave shifts but costs more memory.
    wavelet, wavelet_kw, epsilon
        Forwarded to ``cwt()`` / ``icwt()``.
    hop_length : int
        CWT hop in samples.  1 = every sample (full resolution).
    dampen_source : bool
        Zero out the source region in the scalogram before reconstruction.
    batch_size : int
        Batch size for OOM-halving CWT loops.
    n_fft_max : int
        Max FFT size for CWT (triggers auto-resample above this).
    fade_scales : int
        Number of scale rows for raised-cosine fade at region edges.

    Returns
    -------
    TransplantResult
    """
    from torch_cqt_new import cwt, icwt

    N = signal.shape[-1]
    dtype = signal.dtype
    device = signal.device
    nyq = sr / 2.0

    affine = affine.to(dtype=dtype, device=device)

    if src.f1 - src.f0 < 1e-6 or src.t1 - src.t0 < 1e-6:
        return TransplantResult(
            signal=signal, engine="cwt",
            uncertainty_envelope=torch.zeros_like(signal),
            dst_region=src,
            metadata={"reason": "degenerate_region"},
        )

    # ── Determine destination region ──
    dst = _affine_dst_region(src, affine, nyq)

    # ── CWT frequency range: cover union of src + dst with margin ──
    f_lo = max(0.05, min(src.f0, dst.f0) * 0.5)
    f_hi = min(nyq * 0.95, max(src.f1, dst.f1) * 2.0)
    if f_hi <= f_lo:
        return TransplantResult(
            signal=signal, engine="cwt",
            uncertainty_envelope=torch.zeros_like(signal),
            dst_region=dst,
            metadata={"reason": "invalid_freq_range"},
        )

    # ── Forward CWT ──
    sig_flat = signal.reshape(-1)
    W, freqs = cwt(
        sig_flat, sr,
        fmin=f_lo, fmax=f_hi,
        scales_per_octave=scales_per_octave,
        hop_length=hop_length,
        wavelet=wavelet,
        wavelet_kw=wavelet_kw,
        epsilon=epsilon,
        device=device, dtype=dtype,
        batch_size=batch_size,
        n_fft_max=n_fft_max,
    )
    # W: (n_scales, n_frames) complex, freqs: (n_scales,) high-to-low

    n_scales, n_frames = W.shape[-2], W.shape[-1]
    freqs_hz = freqs.to(dtype=dtype, device=device)

    # ── Map physical frequencies to scale indices ──
    # freqs is high-to-low, so log(freq) decreases with index.
    log_freqs = torch.log2(freqs_hz)  # (n_scales,) decreasing

    def _freq_to_scale_idx(f_hz: float) -> float:
        """Continuous scale index for a physical frequency."""
        if f_hz <= 0:
            return float(n_scales - 1)
        lf = math.log2(f_hz)
        # Linear interpolation in log-freq space
        # log_freqs[0] = highest freq, log_freqs[-1] = lowest
        if n_scales < 2:
            return 0.0
        idx = (float(log_freqs[0]) - lf) / (
            float(log_freqs[0]) - float(log_freqs[-1])
        ) * (n_scales - 1)
        return max(0.0, min(float(n_scales - 1), idx))

    # ── Source rectangle in scalogram coordinates ──
    src_s0 = _freq_to_scale_idx(src.f1)  # higher freq → lower index
    src_s1 = _freq_to_scale_idx(src.f0)  # lower freq → higher index
    src_t0 = max(0.0, src.t0 * sr / max(hop_length, 1))
    src_t1 = min(float(n_frames), src.t1 * sr / max(hop_length, 1))

    # ── Destination rectangle in scalogram coordinates ──
    dst_s0 = _freq_to_scale_idx(dst.f1)
    dst_s1 = _freq_to_scale_idx(dst.f0)
    dst_t0 = max(0.0, dst.t0 * sr / max(hop_length, 1))
    dst_t1 = min(float(n_frames), dst.t1 * sr / max(hop_length, 1))

    # Integer bounds for destination patch
    ds0 = max(0, int(math.floor(dst_s0)))
    ds1 = min(n_scales, int(math.ceil(dst_s1)))
    dt0 = max(0, int(math.floor(dst_t0)))
    dt1 = min(n_frames, int(math.ceil(dst_t1)))
    dst_ns = ds1 - ds0
    dst_nt = dt1 - dt0

    if dst_ns <= 0 or dst_nt <= 0:
        return TransplantResult(
            signal=signal, engine="cwt",
            uncertainty_envelope=torch.zeros_like(signal),
            dst_region=dst,
            metadata={"reason": "empty_destination"},
        )

    # ── Build sampling grid in scalogram space ──
    # The affine maps normalised source [0,1]² to output coords.
    # We work in scalogram coordinates: (scale_idx, frame_idx).
    inv_affine = _affine_2x3_inverse(affine)

    # Compute the affine output bounding box
    corners = torch.tensor(
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=dtype, device=device,
    )
    mapped = corners @ affine.T  # (4, 2)

    # Destination grid spans the mapped bounding box
    dst_t_lin = torch.linspace(
        float(mapped[:, 0].min()), float(mapped[:, 0].max()),
        dst_nt, dtype=dtype, device=device,
    )
    dst_s_lin = torch.linspace(
        float(mapped[:, 1].min()), float(mapped[:, 1].max()),
        dst_ns, dtype=dtype, device=device,
    )
    grid_s, grid_t = torch.meshgrid(dst_s_lin, dst_t_lin, indexing="ij")
    ones = torch.ones_like(grid_t)
    coords = torch.stack([grid_t, grid_s, ones], dim=-1)
    src_coords = torch.matmul(coords, inv_affine.T)
    src_t_norm = src_coords[..., 0]  # [0,1] in source time
    src_s_norm = src_coords[..., 1]  # [0,1] in source freq (scale)

    # Normalised → continuous scalogram coordinates
    src_t_px = src_t_norm * (src_t1 - src_t0 - 1) + src_t0
    src_s_px = src_s_norm * (src_s1 - src_s0 - 1) + src_s0

    # ── Sample source scalogram via bilinear interpolation ──
    # _bilinear_complex expects (B, F, T); W is (n_scales, n_frames)
    W_batch = W.unsqueeze(0)  # (1, n_scales, n_frames)
    sampled = _bilinear_complex(W_batch, src_s_px, src_t_px)
    # sampled: (1, dst_ns, dst_nt) complex

    # ── Phase correction for CWT ──
    # In CWT, each scale row k has center frequency freqs[k].
    # After shifting, a coefficient at source scale s with freq f_s
    # is placed at destination scale d with freq f_d.
    # The phase must rotate by the ratio f_d / f_s per sample.
    src_freqs_interp = _interp_1d(freqs_hz, src_s_px)  # (dst_ns, dst_nt)
    dst_freqs_at_grid = _interp_1d(
        freqs_hz,
        torch.linspace(ds0, ds1 - 1, dst_ns, dtype=dtype, device=device)
            .unsqueeze(-1).expand(-1, dst_nt),
    )  # (dst_ns, dst_nt)

    freq_ratio = dst_freqs_at_grid / src_freqs_interp.clamp(min=1e-10)
    # Phase correction: accumulated phase difference per hop
    omega_src = 2.0 * math.pi * src_freqs_interp / sr  # rads/sample
    omega_dst = omega_src * freq_ratio
    # Cumulative correction across time frames
    frame_idx = torch.arange(dst_nt, dtype=dtype, device=device)
    hop_samples = max(hop_length, 1)
    phase_correction = (omega_dst - omega_src) * hop_samples * frame_idx.unsqueeze(0)
    rotation = torch.polar(
        torch.ones_like(phase_correction),
        phase_correction,
    )
    sampled = sampled * rotation.unsqueeze(0)

    # ── Fade mask for smooth edges ──
    s_fade = _1d_fade(dst_ns, fade_scales, dtype=dtype, device=device)
    t_fade = _1d_fade(dst_nt, fade_scales, dtype=dtype, device=device)
    fade_mask = s_fade.unsqueeze(-1) * t_fade.unsqueeze(0)  # (dst_ns, dst_nt)

    # ── Modify the scalogram ──
    W_mod = W.clone()

    if dampen_source:
        # Fade out source region
        ss0 = max(0, int(math.floor(src_s0)))
        ss1 = min(n_scales, int(math.ceil(src_s1)))
        st0 = max(0, int(math.floor(src_t0)))
        st1 = min(n_frames, int(math.ceil(src_t1)))
        src_ns = ss1 - ss0
        src_nt = st1 - st0
        if src_ns > 0 and src_nt > 0:
            src_fade = _1d_fade(src_ns, fade_scales, dtype=dtype, device=device)
            src_tfade = _1d_fade(src_nt, fade_scales, dtype=dtype, device=device)
            damp_mask = src_fade.unsqueeze(-1) * src_tfade.unsqueeze(0)
            W_mod[ss0:ss1, st0:st1] *= (1.0 - damp_mask)

    # Paste transformed content
    W_mod[ds0:ds1, dt0:dt1] = (
        W_mod[ds0:ds1, dt0:dt1] + sampled.squeeze(0) * fade_mask
    )

    # ── Inverse CWT ──
    reconstructed = icwt(
        W_mod, freqs_hz, sr,
        hop_length=hop_length,
        wavelet=wavelet,
        wavelet_kw=wavelet_kw,
        epsilon=epsilon,
        device=device, dtype=dtype,
        batch_size=batch_size,
        length=N,
    )

    # ── Uncertainty envelope ──
    uncertainty = _cwt_uncertainty_envelope(
        freqs_hz, dst, sr, N, hop_length,
        wavelet=wavelet,
        wavelet_kw=wavelet_kw,
        dtype=dtype, device=device,
    )

    result_sig = reconstructed.to(dtype)
    if signal.shape != result_sig.shape:
        result_sig = result_sig.reshape(signal.shape)

    return TransplantResult(
        signal=result_sig,
        engine="cwt",
        uncertainty_envelope=uncertainty,
        dst_region=dst,
        metadata={
            "n_scales": n_scales,
            "n_frames": n_frames,
            "scales_per_octave": scales_per_octave,
            "src_scale_range": (src_s0, src_s1),
            "dst_scale_range": (dst_s0, dst_s1),
            "freq_ratio": float(freq_ratio.mean()),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher — auto-selects STFT or CWT engine
# ═══════════════════════════════════════════════════════════════════════════


def transplant(
    signal: torch.Tensor,
    sr: int,
    src: TFRegion,
    affine: torch.Tensor,
    *,
    engine: Literal["auto", "stft", "cwt"] = "auto",
    dampen_source: bool = False,
    # STFT-specific
    n_fft: int | None = None,
    hop_length: int | None = None,
    min_bins_in_region: int = 128,
    n_fft_max: int = 2 ** 18,
    crossover_order: int = 8,
    rejection_db: float = 60.0,
    fade_bins: int = 4,
    # CWT-specific
    scales_per_octave: int = 48,
    wavelet: str = "morlet",
    wavelet_kw: dict | None = None,
    epsilon: float | None = None,
    cwt_hop: int = 1,
    cwt_batch_size: int = 64,
    cwt_n_fft_max: int = 2 ** 20,
    fade_scales: int = 4,
) -> TransplantResult:
    """Dispatcher: auto-select STFT or CWT engine for TF transplant.

    Selection criteria (when ``engine="auto"``):

    - **STFT** if both source and destination are above 20 Hz AND the
      maximum frequency ratio between any source and destination
      frequency is below 4×.  STFT gives superior temporal resolution
      and exact reconstruction.

    - **CWT** otherwise.  Required for subsonic content, large ratios,
      or crossing the 20 Hz boundary.  Provides adaptive resolution
      and honest uncertainty reporting, but reconstruction is
      approximate (bounded by the uncertainty envelope).

    Returns ``TransplantResult`` with ``.engine`` indicating which
    path was taken.
    """
    nyq = sr / 2.0
    dst = _affine_dst_region(src, affine, nyq)

    if engine == "auto":
        all_above_20 = src.f0 >= 20 and dst.f0 >= 20
        max_freq = max(src.f1, dst.f1, 1e-10)
        min_freq = max(min(src.f0, dst.f0), 1e-10)
        ratio = max_freq / min_freq
        use_stft = all_above_20 and ratio < 4.0
    else:
        use_stft = (engine == "stft")

    if use_stft:
        sig = tf_transplant(
            signal, sr, src, affine,
            n_fft=n_fft, hop_length=hop_length,
            min_bins_in_region=min_bins_in_region,
            n_fft_max=n_fft_max,
            crossover_order=crossover_order,
            rejection_db=rejection_db,
            dampen_source=dampen_source,
            fade_bins=fade_bins,
        )
        # STFT uncertainty: uniform σ_t = n_fft / (2 * sr) seconds
        eff_nfft = n_fft
        if eff_nfft is None:
            f_span = max(src.f1 - src.f0, 1.0)
            needed = int(math.ceil(min_bins_in_region * sr / f_span))
            eff_nfft = 1 << math.ceil(math.log2(max(needed, 256)))
            eff_nfft = min(eff_nfft, n_fft_max)
        sigma_t_sec = eff_nfft / (2.0 * sr)
        unc = torch.full_like(sig, sigma_t_sec)
        return TransplantResult(
            signal=sig, engine="stft",
            uncertainty_envelope=unc,
            dst_region=dst,
            metadata={"n_fft": eff_nfft, "sigma_t_sec": sigma_t_sec},
        )
    else:
        return cwt_transplant(
            signal, sr, src, affine,
            scales_per_octave=scales_per_octave,
            wavelet=wavelet,
            wavelet_kw=wavelet_kw,
            epsilon=epsilon,
            hop_length=cwt_hop,
            dampen_source=dampen_source,
            batch_size=cwt_batch_size,
            n_fft_max=cwt_n_fft_max,
            fade_scales=fade_scales,
        )


def affine_identity(device: torch.device | None = None,
                    dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """2×3 identity (no transformation)."""
    return torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=dtype, device=device)


def affine_translate(dt: float = 0.0, df: float = 0.0,
                     device: torch.device | None = None,
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Shift in normalised TF coords: dt in [−1,1] time, df in [−1,1] freq."""
    return torch.tensor([[1, 0, dt], [0, 1, df]], dtype=dtype, device=device)


def affine_scale(st: float = 1.0, sf: float = 1.0,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Scale in time and frequency (centred on the source rectangle)."""
    # Scale about centre (0.5, 0.5) → translate, scale, translate back
    # [s  0  0.5(1-s)]
    # [0  s  0.5(1-s)]
    return torch.tensor([
        [st, 0, 0.5 * (1 - st)],
        [0, sf, 0.5 * (1 - sf)],
    ], dtype=dtype, device=device)


def affine_rotate(angle_rad: float,
                  device: torch.device | None = None,
                  dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Rotate about the centre of the normalised source rectangle."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # Rotate about (0.5, 0.5):
    tx = 0.5 * (1 - c + s)
    ty = 0.5 * (1 - c - s)
    return torch.tensor([[c, -s, tx], [s, c, ty]], dtype=dtype, device=device)


def affine_skew(kt: float = 0.0, kf: float = 0.0,
                device: torch.device | None = None,
                dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Skew/shear in time (kt) and frequency (kf)."""
    return torch.tensor([
        [1, kt, -0.5 * kt],
        [kf, 1, -0.5 * kf],
    ], dtype=dtype, device=device)


def affine_compose(*matrices: torch.Tensor) -> torch.Tensor:
    """Compose multiple 2×3 affine matrices left-to-right.

    ``affine_compose(A, B)`` means: apply A first, then B.
    """
    if not matrices:
        return affine_identity()
    result = _to_3x3(matrices[0])
    for m in matrices[1:]:
        result = _to_3x3(m) @ result
    return result[:2, :]


# ═══════════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════════


def _to_complex(dtype: torch.dtype) -> torch.dtype:
    return {
        torch.float16: torch.complex32,
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }.get(dtype, torch.complex64)


def _to_3x3(m: torch.Tensor) -> torch.Tensor:
    """Pad a 2×3 affine to 3×3 for matrix multiplication."""
    row = torch.tensor([[0, 0, 1]], dtype=m.dtype, device=m.device)
    return torch.cat([m, row], dim=0)


def _affine_2x3_inverse(m: torch.Tensor) -> torch.Tensor:
    """Invert a 2×3 affine matrix."""
    m33 = _to_3x3(m)
    return torch.linalg.inv(m33)[:2, :]


def _affine_dst_region(
    src: TFRegion,
    affine: torch.Tensor,
    nyq: float,
) -> TFRegion:
    """Compute the bounding box of the source rectangle after affine transform."""
    corners = torch.tensor([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=affine.dtype, device=affine.device)
    mapped = (corners @ affine.T)  # (4, 2)

    # Normalised → physical
    t_range = src.t1 - src.t0
    f_range = src.f1 - src.f0
    dst_t0 = src.t0 + float(mapped[:, 0].min()) * t_range
    dst_t1 = src.t0 + float(mapped[:, 0].max()) * t_range
    dst_f0 = src.f0 + float(mapped[:, 1].min()) * f_range
    dst_f1 = src.f0 + float(mapped[:, 1].max()) * f_range

    # Clamp to valid ranges
    dst_t0 = max(0.0, dst_t0)
    dst_f0 = max(0.0, dst_f0)
    dst_f1 = min(nyq, dst_f1)

    return TFRegion(t0=dst_t0, t1=dst_t1, f0=dst_f0, f1=dst_f1)


def _stft_hann(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Centred STFT with Hann window.  (B, N) → (B, n_bins, n_frames)."""
    pad = n_fft // 2
    x_pad = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
    return torch.stft(
        x_pad, n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=False, normalized=False,
        onesided=True, return_complex=True,
    )


def _istft_hann(
    S: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse STFT with Hann window.  (B, n_bins, n_frames) → (B, N)."""
    return torch.istft(
        S, n_fft, hop_length=hop_length,
        win_length=n_fft, window=window,
        center=True, normalized=False,
        onesided=True, length=length,
    )


def _bilinear_complex(
    spec: torch.Tensor,
    f_coords: torch.Tensor,
    t_coords: torch.Tensor,
) -> torch.Tensor:
    """Bilinear interpolation of a complex spectrogram.

    Parameters
    ----------
    spec : (B, F, T) complex
    f_coords : (H, W) continuous frequency bin indices
    t_coords : (H, W) continuous time frame indices

    Returns
    -------
    (B, H, W) complex  — interpolated values.
    """
    B, F, T = spec.shape
    H, W = f_coords.shape

    f0 = f_coords.long().clamp(0, F - 2)
    t0 = t_coords.long().clamp(0, T - 2)
    f1 = f0 + 1
    t1 = t0 + 1

    wf = (f_coords - f0.to(f_coords.dtype)).clamp(0, 1)
    wt = (t_coords - t0.to(t_coords.dtype)).clamp(0, 1)

    # Gather corners: (B, H, W) for each of 4 corners
    def _gather(fi, ti):
        fi_c = fi.clamp(0, F - 1).unsqueeze(0).expand(B, -1, -1)
        ti_c = ti.clamp(0, T - 1).unsqueeze(0).expand(B, -1, -1)
        # Advanced indexing
        b_idx = torch.arange(B, device=spec.device).view(B, 1, 1).expand_as(fi_c)
        return spec[b_idx, fi_c, ti_c]

    v00 = _gather(f0, t0)
    v01 = _gather(f0, t1)
    v10 = _gather(f1, t0)
    v11 = _gather(f1, t1)

    wf = wf.unsqueeze(0)
    wt = wt.unsqueeze(0)

    top = v00 * (1 - wt) + v01 * wt
    bot = v10 * (1 - wt) + v11 * wt
    return top * (1 - wf) + bot * wf


def _phase_correct(
    sampled: torch.Tensor,
    src_spec: torch.Tensor,
    src_f_px: torch.Tensor,
    src_t_px: torch.Tensor,
    dst_f_px: torch.Tensor,
    hop_length: int,
    n_fft: int,
) -> torch.Tensor:
    """Instantaneous-frequency phase vocoder correction.

    Estimates the true instantaneous frequency from inter-frame phase
    differences of the source spectrogram, interpolates it at the
    continuous source coordinates, scales by the destination/source
    frequency ratio, and propagates phase cumulatively.

    Parameters
    ----------
    sampled : (B, H, W) complex — bilinear-interpolated STFT coefficients
    src_spec : (B, F, T) complex — source spectrogram for IF estimation
    src_f_px : (H, W) — continuous source frequency bin coordinates
    src_t_px : (H, W) — continuous source time frame coordinates
    dst_f_px : (H, 1) or (H, W) — destination frequency bins
    hop_length : int
    n_fft : int

    Returns
    -------
    (B, H, W) complex — phase-corrected coefficients.
    """
    rdtype = sampled.real.dtype
    device = sampled.device
    B, F, T = src_spec.shape

    # ── 1. Instantaneous frequency field from source spectrogram ──
    bin_idx = torch.arange(F, dtype=rdtype, device=device)
    omega_k = 2.0 * math.pi * bin_idx / n_fft  # rads/sample per bin

    if T >= 2:
        expected = omega_k * hop_length  # expected phase advance per hop
        phase = torch.angle(src_spec)  # (B, F, T)
        dphi = phase[:, :, 1:] - phase[:, :, :-1]  # (B, F, T-1)
        dev = dphi - expected.view(1, F, 1)
        dev = dev - (2.0 * math.pi) * torch.round(dev / (2.0 * math.pi))
        IF_interior = omega_k.view(1, F, 1) + dev / hop_length  # (B, F, T-1)
        IF_frame0 = omega_k.view(1, F, 1).expand(B, -1, 1)
        IF_field = torch.cat([IF_frame0, IF_interior], dim=-1)  # (B, F, T)
    else:
        IF_field = omega_k.view(1, F, 1).expand(B, -1, max(T, 1))

    # ── 2. Interpolate IF at continuous source coordinates ──
    IF_interp = _bilinear_complex(
        IF_field.to(src_spec.dtype), src_f_px, src_t_px,
    ).real  # (B, H, W)

    # ── 3. Scale IF by destination/source frequency ratio ──
    src_f = src_f_px.clamp(min=0.5)
    dst_f = dst_f_px.clamp(min=0.5)
    ratio = dst_f / src_f
    IF_dst = IF_interp * ratio.unsqueeze(0)  # (B, H, W)

    # ── 4. Cumulative phase propagation ──
    phase_advance = IF_dst * hop_length
    cum = torch.cumsum(phase_advance, dim=-1)  # (B, H, W)
    seed = torch.angle(sampled[:, :, :1])  # (B, H, 1)
    synth_phase = seed + cum - cum[:, :, :1]

    # ── 5. Reconstruct: original magnitude × synthesised phase ──
    mag = sampled.abs()
    return torch.polar(mag, synth_phase.to(rdtype))


def _smooth_mask(
    n_bins: int,
    n_frames: int,
    f0: int,
    f1: int,
    t0: int,
    t1: int,
    fade_bins: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Smooth Tukey-style mask for a rectangular TF region.

    Returns (n_bins, n_frames) with 1.0 inside the region,
    smooth fade at edges, 0.0 outside.
    """
    mask = torch.zeros(n_bins, n_frames, dtype=dtype, device=device)

    # Frequency mask (raised cosine fade)
    f_mask = torch.zeros(n_bins, dtype=dtype, device=device)
    f_mask[f0:f1] = 1.0
    if fade_bins > 0:
        for i in range(fade_bins):
            alpha = 0.5 * (1 - math.cos(math.pi * i / fade_bins))
            if f0 + i < n_bins:
                f_mask[f0 + i] = alpha
            if f1 - 1 - i >= 0:
                f_mask[f1 - 1 - i] = alpha

    # Time mask
    t_mask = torch.zeros(n_frames, dtype=dtype, device=device)
    t_mask[t0:t1] = 1.0
    fade_frames = max(1, fade_bins)
    if fade_frames > 0:
        for i in range(fade_frames):
            alpha = 0.5 * (1 - math.cos(math.pi * i / fade_frames))
            if t0 + i < n_frames:
                t_mask[t0 + i] = alpha
            if t1 - 1 - i >= 0:
                t_mask[t1 - 1 - i] = alpha

    return f_mask.unsqueeze(-1) * t_mask.unsqueeze(0)


# ── CWT helpers ──────────────────────────────────────────────────────────


def _interp_1d(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """1-D linear interpolation of ``values`` at continuous ``indices``.

    ``values``: (N,) — the lookup table.
    ``indices``: arbitrary shape — continuous indices into ``values``.
    Returns the same shape as ``indices``.
    """
    N = values.shape[0]
    idx0 = indices.long().clamp(0, N - 2)
    idx1 = idx0 + 1
    w = (indices - idx0.to(indices.dtype)).clamp(0, 1)
    return values[idx0] * (1 - w) + values[idx1] * w


def _1d_fade(
    length: int,
    fade: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Raised-cosine fade-in/out mask of given *length*."""
    mask = torch.ones(length, dtype=dtype, device=device)
    f = min(fade, length // 2)
    if f > 0:
        ramp = 0.5 * (1 - torch.cos(
            torch.linspace(0, math.pi, f, dtype=dtype, device=device)))
        mask[:f] = ramp
        mask[-f:] = ramp.flip(0)
    return mask


def _cwt_uncertainty_envelope(
    freqs: torch.Tensor,
    dst: TFRegion,
    sr: int,
    n_samples: int,
    hop_length: int,
    wavelet: str = "morlet",
    wavelet_kw: dict | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the temporal uncertainty σ_t(n) for the destination region.

    For a Morlet wavelet with parameter σ, the time-domain Gaussian
    envelope has standard deviation::

        σ_t(f) = σ / (2π f)     [in seconds]

    This quantifies the irreducible temporal smearing: a CWT coefficient
    at frequency f localises the signal to within ±σ_t(f).  At 1 Hz with
    σ=6, that's ≈ 0.95 s of blur.  At 100 Hz, it's ≈ 9.5 ms.

    The returned envelope is the *average* σ_t across the destination
    frequency range, sampled at each output sample.   Where the destination
    region is not active (outside [dst.t0, dst.t1]) the envelope is zero.

    Returns
    -------
    (n_samples,) float tensor — σ_t in seconds at each sample.
    """
    wkw = dict(wavelet_kw or {})
    sigma = wkw.get("sigma", 6.0)

    # Average σ_t over destination frequencies
    f_lo = max(dst.f0, 1e-10)
    f_hi = max(dst.f1, 1e-10)
    # Integral of σ/(2πf) from f_lo to f_hi = σ/(2π) * ln(f_hi/f_lo)
    # Divided by (f_hi - f_lo) for the average? No — for uniform weight
    # in log-freq it's: mean = σ/(2π) * ln(f_hi/f_lo) / log_span
    # But the natural measure on a log-freq grid is per-octave, so:
    if f_hi > f_lo:
        log_span = math.log(f_hi / f_lo)
        # Average of 1/f in [f_lo, f_hi] on a linear scale = ln(f_hi/f_lo)/(f_hi-f_lo)
        # Average of σ/(2πf) = σ/(2π) * ln(f_hi/f_lo) / (f_hi - f_lo)
        # But on a LOG scale (which is how CWT bins are distributed):
        # average of σ/(2πf) over log-spaced bins = σ/(2π) * (1/f_lo - 1/f_hi) / log_span
        # Hmm — simpler: just take geometric mean frequency
        f_geo = math.sqrt(f_lo * f_hi)
        sigma_t_avg = sigma / (2.0 * math.pi * f_geo)
    else:
        sigma_t_avg = sigma / (2.0 * math.pi * max(f_lo, 1e-10))

    # Build envelope: nonzero only in the destination time range
    envelope = torch.zeros(n_samples, dtype=dtype, device=device)
    t0_samp = max(0, int(dst.t0 * sr))
    t1_samp = min(n_samples, int(math.ceil(dst.t1 * sr)))
    if t1_samp > t0_samp:
        # Taper at edges rather than a hard box
        region_len = t1_samp - t0_samp
        fade = min(32, region_len // 4)
        window = _1d_fade(region_len, fade, dtype=dtype, device=device)
        envelope[t0_samp:t1_samp] = sigma_t_avg * window

    return envelope
