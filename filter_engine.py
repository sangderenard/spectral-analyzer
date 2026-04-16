"""Unified filter engine.

All filter operations in this project route through here.  Two back-ends
are available and selected via the ``filter_mode`` parameter:

    "fir"  — Frequency-domain Linkwitz-Riley (crossover.py).
             Zero-phase, perfectly-reconstructing, GPU-accelerated.
             HQ default.  Requires torch.

    "iir"  — Time-domain IIR (scipy Butterworth 8th-order, sosfilt).
             Causal with state carry across blocks; lower memory per
             sample but non-linear phase near the transition band.
             Fast fallback, numpy-only.

Public surface
--------------
apply_bandpass(signal, sr, fmin, fmax, *, filter_mode, order)
    Single-shot bandpass / HP / LP on a complete 1-D numpy array.

build_streaming_filter(sr, fmin, fmax, *, filter_mode, order)
    Returns a StreamingFilter that can be applied block-by-block with
    state carry.  FIR mode: overlap-save convolution.  IIR mode: sosfilt_zi.

decompose_bands(signal, sr, crossovers, *, filter_mode, order, dtype)
    Split a 1-D numpy array into N+1 perfectly-reconstructing subbands.

FilterMode
    Literal type alias; convenience constants FIR / IIR.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

FilterMode = Literal["fir", "iir"]
FIR: FilterMode = "fir"
IIR: FilterMode = "iir"

_DEFAULT_ORDER = 8


# ──────────────────────────────────────────────────────────────────────────────
# Single-shot helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_bandpass(
    signal: np.ndarray,
    sr: int,
    fmin: float | None,
    fmax: float | None,
    *,
    filter_mode: FilterMode = FIR,
    order: int = _DEFAULT_ORDER,
) -> np.ndarray:
    """Apply a bandpass / HP / LP filter to *signal* and return the result.

    Parameters
    ----------
    signal   : 1-D ndarray, any real dtype.
    sr       : sample rate in Hz.
    fmin     : highpass edge Hz, or None for LP-only.
    fmax     : lowpass edge Hz, or None for HP-only.
    filter_mode : ``"fir"`` (freq-domain LR, zero-phase) or
                  ``"iir"`` (Butterworth sosfiltfilt, zero-phase offline).
    order    : filter order (must be even for LR).
    """
    if fmin is None and fmax is None:
        return signal.copy()
    src_dtype = signal.dtype
    if filter_mode == FIR:
        return _fir_apply_bandpass(signal, sr, fmin, fmax, order, src_dtype)
    else:
        return _iir_apply_bandpass(signal, sr, fmin, fmax, order, src_dtype)


def decompose_bands(
    signal: np.ndarray,
    sr: int,
    crossovers: list[float],
    *,
    filter_mode: FilterMode = FIR,
    order: int = _DEFAULT_ORDER,
) -> list[np.ndarray]:
    """Split *signal* into ``len(crossovers)+1`` perfectly-reconstructing subbands.

    The output list sums back to *signal* to machine precision in FIR mode,
    and to within the IIR cascade error in IIR mode.

    Parameters
    ----------
    crossovers : sorted interior crossover frequencies (Hz).
    """
    src_dtype = signal.dtype
    if filter_mode == FIR:
        return _fir_decompose(signal, sr, crossovers, order, src_dtype)
    else:
        return _iir_decompose(signal, sr, crossovers, order, src_dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming filter (block-by-block with state carry)
# ──────────────────────────────────────────────────────────────────────────────

class StreamingFilter:
    """Stateful single-channel filter that processes one block at a time.

    Usage::

        f = build_streaming_filter(sr, fmin, fmax, filter_mode="iir")
        for block in blocks:
            out = f.process(block)
    """

    def process(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _IIRStreamingFilter(StreamingFilter):
    def __init__(self, sos: np.ndarray, zi: np.ndarray) -> None:
        self._sos = sos
        self._zi = zi

    def process(self, block: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt
        out, self._zi = sosfilt(self._sos, block, zi=self._zi)
        return out


class _FIRStreamingFilter(StreamingFilter):
    """Overlap-save convolution with a windowed-sinc FIR kernel.

    The kernel is built once from the crossover LR frequency-domain
    response evaluated at the required frequencies, then converted to
    a time-domain impulse response via IFFT.  This gives us a causal
    (linear-phase, fixed-delay) FIR that matches the crossover.py
    frequency shape exactly.
    """

    def __init__(
        self,
        sr: int,
        fmin: float | None,
        fmax: float | None,
        order: int,
    ) -> None:
        self._sr = sr
        self._order = order
        self._fmin = fmin
        self._fmax = fmax

        # Build impulse response
        h = _fir_kernel(sr, fmin, fmax, order)
        self._h = h
        self._n_taps = len(h)
        # Overlap-save tail buffer
        self._tail = np.zeros(self._n_taps - 1, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        import numpy.fft as _fft
        h = self._h
        tail = self._tail
        n_tap = self._n_taps
        x = np.concatenate([tail, block.astype(np.float64)])
        # Linear convolution via FFT, take only the causal output
        n_fft = int(2 ** math.ceil(math.log2(len(x) + n_tap - 1)))
        X = _fft.rfft(x, n=n_fft)
        H = _fft.rfft(h, n=n_fft)
        y = _fft.irfft(X * H, n=n_fft)
        # Overlap-save: output starts at index (n_tap-1)
        out = y[n_tap - 1: n_tap - 1 + len(block)]
        self._tail = x[-(n_tap - 1):]
        return out.astype(block.dtype)


def build_streaming_filter(
    sr: int,
    fmin: float | None,
    fmax: float | None,
    *,
    filter_mode: FilterMode = FIR,
    order: int = _DEFAULT_ORDER,
) -> StreamingFilter:
    """Build a stateful block-by-block filter.

    In ``"iir"`` mode: returns a ``sosfilt_zi``-backed filter (causal, non-linear phase).
    In ``"fir"`` mode: returns an overlap-save FIR (linear-phase, fixed group delay).
    """
    if filter_mode == IIR:
        from scipy.signal import butter, sosfilt_zi
        nyq = sr / 2.0
        lo = max(fmin, 0.5) if fmin is not None else None
        hi = min(fmax, nyq * 0.9999) if fmax is not None else None
        if lo is not None and hi is not None:
            sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        elif lo is not None:
            sos = butter(order, lo / nyq, btype="highpass", output="sos")
        else:
            assert hi is not None
            sos = butter(order, hi / nyq, btype="lowpass", output="sos")
        zi = sosfilt_zi(sos) * 0.0
        return _IIRStreamingFilter(sos, zi)
    else:
        return _FIRStreamingFilter(sr, fmin, fmax, order)


# ──────────────────────────────────────────────────────────────────────────────
# FIR back-end
# ──────────────────────────────────────────────────────────────────────────────

def _fir_kernel(
    sr: int,
    fmin: float | None,
    fmax: float | None,
    order: int,
) -> np.ndarray:
    """Build a real-valued windowed LR frequency-response impulse response.

    The frequency-domain LR mask (same shape as crossover.py uses) is
    evaluated on a dense grid and returned via IFFT as a time-domain
    kernel.  The kernel length is chosen so the transition-band rolloff
    is fully resolved.
    """
    nyq = sr / 2.0
    lo = max(fmin, 0.0) if fmin is not None else 0.0
    hi = min(fmax, nyq) if fmax is not None else nyq

    # Length: at least 4 × sr/min_cutoff_hz, min 64 taps
    fc_ref = lo if lo > 0 else hi
    n_taps = max(64, int(4 * sr / max(fc_ref, 1.0)))
    # Next odd power of 2 + 1 for linear-phase symmetry
    n_taps = int(2 ** math.ceil(math.log2(n_taps))) + 1

    n_fft = n_taps * 4
    freqs = np.linspace(0, nyq, n_fft // 2 + 1, dtype=np.float64)

    # Build gain mask: product of LR LP and HP masks
    mask = np.ones(len(freqs), dtype=np.float64)
    if fmax is not None and hi < nyq:
        r_hi = (freqs / max(hi, 1e-9)) ** order
        mask *= 1.0 / (1.0 + r_hi)
    if fmin is not None and lo > 0:
        r_lo = (freqs / max(lo, 1e-9)) ** order
        hp_mask = r_lo / (1.0 + r_lo)   # HP = 1 − LP
        mask *= hp_mask

    # IFFT → time-domain kernel
    h_full = np.fft.irfft(mask, n=n_fft)
    # Shift to causal (linear-phase): rotate by n_fft//2
    h_full = np.roll(h_full, n_fft // 2)
    # Window to n_taps
    h = h_full[:n_taps] * np.blackman(n_taps)
    h /= h.sum() if h.sum() != 0 else 1.0
    return h.astype(np.float64)


def _fir_apply_bandpass(
    signal: np.ndarray,
    sr: int,
    fmin: float | None,
    fmax: float | None,
    order: int,
    src_dtype: np.dtype,
) -> np.ndarray:
    """Zero-phase FIR via frequency-domain LR mask (single-shot, not streaming)."""
    import torch
    from crossover import lr_decompose

    nyq = sr / 2.0
    lo = max(fmin, 0.0) if fmin is not None else None
    hi = min(fmax, nyq) if fmax is not None else None

    # Build crossover list: hp edge, lp edge — extract the middle band
    xo: list[float] = []
    band_idx = 0
    if lo is not None and lo > 0:
        xo.append(lo)
        band_idx = 1
    if hi is not None and hi < nyq:
        xo.append(hi)

    if not xo:
        return signal.copy()

    x_t = torch.as_tensor(signal.astype(np.float64), dtype=torch.float64)
    bands = lr_decompose(x_t, xo, sr, order=order)

    if lo is not None and lo > 0 and hi is not None and hi < nyq:
        # bandpass: bands[0]=LP below lo, bands[1]=target band, bands[2]=HP above hi
        # but lr_decompose gives n_bands = len(xo)+1 = 2 or 3
        # With both edges: [below_lo, between, above_hi] → bands[1]
        result = bands[1].numpy()
    elif lo is not None and lo > 0:
        # HP: bands[0]=below, bands[1]=above
        result = bands[1].numpy()
    else:
        # LP: bands[0]=below hi, bands[1]=above hi
        result = bands[0].numpy()

    return result.astype(src_dtype)


def _fir_decompose(
    signal: np.ndarray,
    sr: int,
    crossovers: list[float],
    order: int,
    src_dtype: np.dtype,
) -> list[np.ndarray]:
    import torch
    from crossover import lr_decompose

    if not crossovers:
        return [signal.copy()]

    x_t = torch.as_tensor(signal.astype(np.float64), dtype=torch.float64)
    bands = lr_decompose(x_t, crossovers, sr, order=order)
    return [bands[i].numpy().astype(src_dtype) for i in range(bands.shape[0])]


# ──────────────────────────────────────────────────────────────────────────────
# IIR back-end
# ──────────────────────────────────────────────────────────────────────────────

def _iir_apply_bandpass(
    signal: np.ndarray,
    sr: int,
    fmin: float | None,
    fmax: float | None,
    order: int,
    src_dtype: np.dtype,
) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    lo = max(fmin, 0.5) if fmin is not None else None
    hi = min(fmax, nyq * 0.9999) if fmax is not None else None
    x = signal.astype(np.float64)
    if lo is not None and hi is not None:
        sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    elif lo is not None:
        sos = butter(order, lo / nyq, btype="highpass", output="sos")
    else:
        assert hi is not None
        sos = butter(order, hi / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x).astype(src_dtype)


def _iir_decompose(
    signal: np.ndarray,
    sr: int,
    crossovers: list[float],
    order: int,
    src_dtype: np.dtype,
) -> list[np.ndarray]:
    """IIR LR decomposition matching the existing _compute_lr cascade."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    half = order // 2
    sig = signal.astype(np.float64)
    xo = sorted(crossovers)
    bands: list[np.ndarray] = []
    prev_lp2: np.ndarray | None = None
    for i, fc in enumerate(xo):
        wn = max(fc / nyq, 1e-4)
        wn = min(wn, 0.9999)
        sos = butter(half, wn, btype="low", output="sos")
        curr_lp2 = sosfilt(sos, sosfilt(sos, sig))
        sub = curr_lp2 if prev_lp2 is None else curr_lp2 - prev_lp2
        bands.append(sub.astype(src_dtype))
        prev_lp2 = curr_lp2
    # HP residual
    hp = sig if prev_lp2 is None else sig - prev_lp2
    bands.append(hp.astype(src_dtype))
    return bands
