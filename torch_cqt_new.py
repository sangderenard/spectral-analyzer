"""Pure-torch CQT / iCQT — 1:1 transliteration of librosa's algorithm.

Every function maps to a specific librosa function. Comments reference
the librosa source directly. Zero librosa/numpy/scipy calls at runtime.

Forward:  cqt(y, sr, ...) → C  complex (..., n_bins, n_frames)
Inverse:  icqt(C, sr, ...) → y  float   (..., n_samples)
"""
from __future__ import annotations
import gc
import math
import os
import tempfile
import threading
import time
import warnings
import numpy as np
import torch
import torchaudio.functional as TAF
from dataclasses import dataclass, field
from math import gcd
from typing import Callable, Protocol, Sequence, TypedDict, runtime_checkable

# ── Progress observation ─────────────────────────────────────────────────


@dataclass
class CQTProgress:
    """Thread-safe read-only snapshot of CQT computation progress.

    All attributes are updated atomically from the compute thread.
    Read from any thread at any time.
    """
    # Octave-level
    n_octaves: int = 0
    octave: int = 0                  # current octave loop index (0 = top)
    octave_n_filters: int = 0        # filters in this octave
    octave_n_fft: int = 0            # n_fft for this octave
    octave_n_frames: int = 0         # STFT frames for this octave
    octave_freq_lo: float = 0.0
    octave_freq_hi: float = 0.0

    # Filter build progress
    filters_done: int = 0            # filters built so far this octave
    filter_batch_size: int = 0       # current sub-batch K (after halving)

    # Streaming matmul progress
    filter_batches_done: int = 0     # filter-batches completed (matmul)
    filter_batches_total: int = 0
    frame_batches_done: int = 0      # frame-batches in current filter-batch
    frame_batches_total: int = 0
    frame_batch_size: int = 0        # current frame sub-batch (after halving)

    # Timing
    octave_start_time: float = 0.0
    total_start_time: float = 0.0

    # Lock for atomic updates from compute thread
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        """Return a plain dict copy — safe to read from any thread."""
        with self._lock:
            return {
                'n_octaves': self.n_octaves,
                'octave': self.octave,
                'octave_n_filters': self.octave_n_filters,
                'octave_n_fft': self.octave_n_fft,
                'octave_n_frames': self.octave_n_frames,
                'octave_freq_lo': self.octave_freq_lo,
                'octave_freq_hi': self.octave_freq_hi,
                'filters_done': self.filters_done,
                'filter_batch_size': self.filter_batch_size,
                'filter_batches_done': self.filter_batches_done,
                'filter_batches_total': self.filter_batches_total,
                'frame_batches_done': self.frame_batches_done,
                'frame_batches_total': self.frame_batches_total,
                'frame_batch_size': self.frame_batch_size,
                'octave_start_time': self.octave_start_time,
                'total_start_time': self.total_start_time,
            }

    def elapsed_octave(self) -> float:
        return time.monotonic() - self.octave_start_time

    def elapsed_total(self) -> float:
        return time.monotonic() - self.total_start_time

    def summary(self) -> str:
        """One-line human-readable status."""
        s = self.snapshot()
        pct_filters = 100 * s['filters_done'] / max(1, s['octave_n_filters'])
        pct_matmul = 100 * s['filter_batches_done'] / max(1, s['filter_batches_total'])
        return (
            f"oct {s['octave']}/{s['n_octaves']} "
            f"[{s['octave_freq_lo']:.1f}-{s['octave_freq_hi']:.1f} Hz] "
            f"n_fft={s['octave_n_fft']} "
            f"filters={s['filters_done']}/{s['octave_n_filters']} ({pct_filters:.0f}%) "
            f"matmul={s['filter_batches_done']}/{s['filter_batches_total']} ({pct_matmul:.0f}%) "
            f"K={s['filter_batch_size']} "
            f"elapsed={self.elapsed_octave():.1f}s"
        )


@runtime_checkable
class CQTProgressCallback(Protocol):
    """Optional callback invoked after each sub-batch completes."""
    def __call__(self, progress: CQTProgress) -> None: ...


# ── Constants ────────────────────────────────────────────────────────────

# librosa.note_to_hz("C1")
_C1_HZ = 32.70319566257483

# librosa.filters.WINDOW_BANDWIDTHS (subset mirrored locally)
_WINDOW_BANDWIDTHS: dict[str, float] = {
    "hann": 1.50018310546875,
    "hamming": 1.3629455320350348,
    "blackman": 1.7269681554262326,
    "blackmanharris": 2.0045975283585014,
}
_HANN_BANDWIDTH = _WINDOW_BANDWIDTHS["hann"]
_CQT_WINDOWS: tuple[str, ...] = tuple(_WINDOW_BANDWIDTHS.keys())


# ── Dtype helpers ────────────────────────────────────────────────────────

_FLOAT_TO_COMPLEX: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
_COMPLEX_TO_FLOAT: dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def _to_complex(dtype: torch.dtype) -> torch.dtype:
    """Map a real dtype to its complex counterpart."""
    if dtype in _FLOAT_TO_COMPLEX:
        return _FLOAT_TO_COMPLEX[dtype]
    if dtype.is_complex:
        return dtype
    raise ValueError(f"No complex counterpart for {dtype}")


def _to_float(dtype: torch.dtype) -> torch.dtype:
    """Map a complex dtype to its real counterpart."""
    if dtype in _COMPLEX_TO_FLOAT:
        return _COMPLEX_TO_FLOAT[dtype]
    if dtype.is_floating_point:
        return dtype
    raise ValueError(f"No float counterpart for {dtype}")


def _canonical_cqt_window(window: str) -> str:
    key = str(window).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "bh": "blackmanharris",
        "blackharris": "blackmanharris",
    }
    key = aliases.get(key, key)
    if key not in _WINDOW_BANDWIDTHS:
        raise ValueError(
            f"Unsupported CQT window {window!r}. "
            f"Expected one of {sorted(_WINDOW_BANDWIDTHS)}")
    return key


def _window_bandwidth(window: str) -> float:
    return _WINDOW_BANDWIDTHS[_canonical_cqt_window(window)]


# ── Pad ──────────────────────────────────────────────────────────────────

def _pad(
    x: torch.Tensor,
    pad_left: int,
    pad_right: int,
    mode: str = "reflect",
    value: float = 0.0,
) -> torch.Tensor:
    """Pad *x* along its last dimension, supporting arbitrary leading dims."""
    if pad_left == 0 and pad_right == 0:
        return x

    L = x.shape[-1]

    if mode == "constant":
        return torch.nn.functional.pad(x, (pad_left, pad_right),
                                       mode="constant", value=value)

    if mode in ("reflect", "replicate", "circular"):
        orig_shape = x.shape
        flat = x.reshape(-1, L).unsqueeze(1)  # (B, 1, L)

        if mode == "reflect" and L <= 1:
            val = flat[..., :1].expand(*flat.shape[:-1], pad_left + L + pad_right)
            return val.squeeze(1).reshape(*orig_shape[:-1], -1)

        if mode == "reflect" and (pad_left >= L or pad_right >= L):
            result = flat
            cur_L = L
            left_remaining = pad_left
            right_remaining = pad_right
            while right_remaining > 0:
                chunk = min(right_remaining, cur_L - 1)
                result = torch.nn.functional.pad(
                    result, (0, chunk), mode="reflect")
                cur_L = result.shape[-1]
                right_remaining -= chunk
            while left_remaining > 0:
                chunk = min(left_remaining, cur_L - 1)
                result = torch.nn.functional.pad(
                    result, (chunk, 0), mode="reflect")
                cur_L = result.shape[-1]
                left_remaining -= chunk
            return result.squeeze(1).reshape(*orig_shape[:-1], -1)

        padded = torch.nn.functional.pad(flat, (pad_left, pad_right), mode=mode)
        return padded.squeeze(1).reshape(*orig_shape[:-1], -1)

    raise ValueError(f"Unsupported pad mode: {mode!r}")


# ══════════════════════════════════════════════════════════════════════════
# Pure-torch replacements for librosa helper functions
# ══════════════════════════════════════════════════════════════════════════

def _cqt_frequencies(
    n_bins: int, fmin: float, bins_per_octave: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Torch clone of ``librosa.cqt_frequencies`` (tuning=0).

    Returns (n_bins,) *dtype* on *device*.
    """
    return fmin * (2.0 ** (torch.arange(n_bins, dtype=dtype,
                                        device=device) / bins_per_octave))


def _relative_bandwidth(freqs: torch.Tensor) -> torch.Tensor:
    """Torch clone of ``librosa.filters._relative_bandwidth``.

    freqs : (n_bins,) float64
    Returns alpha : (n_bins,) float64, same device.
    """
    logf = torch.log2(freqs)
    bpo = torch.empty_like(freqs)
    bpo[0] = 1.0 / (logf[1] - logf[0])
    bpo[-1] = 1.0 / (logf[-1] - logf[-2])
    if len(freqs) > 2:
        bpo[1:-1] = 2.0 / (logf[2:] - logf[:-2])
    alpha = (2.0 ** (2.0 / bpo) - 1.0) / (2.0 ** (2.0 / bpo) + 1.0)
    return alpha


def _wavelet_lengths(
    freqs: torch.Tensor,
    sr: float,
    filter_scale: float,
    window_bandwidth: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Torch clone of ``librosa.filters.wavelet_lengths`` (gamma=0).

    Returns lengths : (n_bins,) float64, same device as freqs.
    """
    Q = filter_scale / alpha           # (n_bins,)
    lengths = Q * sr / freqs           # gamma=0 → denominator is just freqs
    return lengths


# ══════════════════════════════════════════════════════════════════════════
# Window functions — matching scipy.signal.get_window(..., fftbins=True)
# ══════════════════════════════════════════════════════════════════════════

def _periodic_window(
    n: int,
    window: str,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Periodic window matching ``scipy.signal.get_window(..., fftbins=True)``.

    This is the window family used by librosa's CQT wavelet builder.
    """
    if n <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    key = _canonical_cqt_window(window)
    k = torch.arange(n, dtype=dtype, device=device)
    phase = 2.0 * math.pi * k / float(n)
    cos1 = torch.cos(phase)
    if key == "hann":
        return 0.5 - 0.5 * cos1
    if key == "hamming":
        return 0.54 - 0.46 * cos1
    if key == "blackman":
        return 0.42 - 0.5 * cos1 + 0.08 * torch.cos(2.0 * phase)
    if key == "blackmanharris":
        return (0.35875
                - 0.48829 * cos1
                + 0.14128 * torch.cos(2.0 * phase)
                - 0.01168 * torch.cos(3.0 * phase))
    raise AssertionError(f"Unhandled CQT window {window!r}")


def _periodic_window_bank(
    col: torch.Tensor,
    n_vals: torch.Tensor,
    window: str,
) -> torch.Tensor:
    """Vectorized periodic window bank for varying integer lengths."""
    key = _canonical_cqt_window(window)
    phase = 2.0 * math.pi * col / n_vals
    cos1 = torch.cos(phase)
    ones = torch.ones_like(cos1)
    if key == "hann":
        out = 0.5 - 0.5 * cos1
    elif key == "hamming":
        out = 0.54 - 0.46 * cos1
    elif key == "blackman":
        out = 0.42 - 0.5 * cos1 + 0.08 * torch.cos(2.0 * phase)
    elif key == "blackmanharris":
        out = (0.35875
               - 0.48829 * cos1
               + 0.14128 * torch.cos(2.0 * phase)
               - 0.01168 * torch.cos(3.0 * phase))
    else:
        raise AssertionError(f"Unhandled CQT window {window!r}")
    return torch.where(n_vals <= 1.0, ones, out)


# ══════════════════════════════════════════════════════════════════════════
# librosa.filters.wavelet  →  _wavelet
# librosa.__vqt_filter_fft  →  _vqt_filter_fft
# ══════════════════════════════════════════════════════════════════════════

def _wavelet(
    freqs: torch.Tensor,
    sr: float,
    window: str = "hann",
    filter_scale: float = 1.0,
    alpha: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch clone of ``librosa.filters.wavelet(..., pad_fft=True, norm=1)``.

    Returns
    -------
    basis   : (n_filters, n_fft)  complex on device
    lengths : (n_filters,)  float on device — fractional filter lengths
    """
    if device is None:
        device = freqs.device
    cdtype = _to_complex(dtype)

    lengths = _wavelet_lengths(freqs, sr, filter_scale, _HANN_BANDWIDTH, alpha)

    n_filters = len(freqs)
    filters_list: list[torch.Tensor] = []

    for i in range(n_filters):
        ilen = lengths[i].item()  # scalar float
        freq = freqs[i].item()

        # ── librosa.util.phasor(arange(-ilen//2, ilen//2) * 2π f / sr) ──
        t = torch.arange(-ilen // 2, ilen // 2,
                         dtype=dtype, device=device)
        sig = torch.exp(1j * 2.0 * math.pi * freq / sr * t)

        # ── __float_window(window)(len(sig)) ──
        # floor(ilen) samples of real window, zero-pad to ceil(ilen)
        n_min = int(math.floor(ilen))
        n_max = int(math.ceil(ilen))
        win = _periodic_window(n_min, window, device, dtype=dtype)
        if n_max > n_min:
            win = _pad(win, 0, n_max - n_min, mode="constant")
        # Set any samples beyond floor(ilen) to zero (librosa does window[n_min:] = 0)
        if n_max > n_min:
            win[n_min:] = 0.0
        # Align to sig length
        win = win[:len(sig)]

        sig = sig * win

        # ── librosa.util.normalize(sig, norm=1)  → L1 ──
        sig = sig / torch.abs(sig).sum()

        filters_list.append(sig)

    # ── pad_fft=True: pad to next power of 2 ──
    max_len = int(lengths.max().item())
    n_fft = int(2 ** math.ceil(math.log2(max_len)))

    # ── librosa.util.pad_center: center each filter in n_fft ──
    basis = torch.zeros((n_filters, n_fft),
                        dtype=cdtype, device=device)
    for i, filt in enumerate(filters_list):
        flen = filt.shape[0]
        lpad = (n_fft - flen) // 2
        basis[i, lpad:lpad + flen] = filt

    return basis, lengths


def _vqt_filter_fft(
    sr: float,
    freqs: torch.Tensor,
    window: str,
    filter_scale: float,
    hop_length: int,
    alpha: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Torch clone of ``librosa.__vqt_filter_fft``.

    Returns
    -------
    fft_basis : (n_filters, n_fft//2+1)  complex on device
    n_fft     : int
    lengths   : (n_filters,)  float on device
    """
    # Step 1: build time-domain basis via wavelet()
    basis, lengths = _wavelet(
        freqs, sr, window=window, filter_scale=filter_scale,
        alpha=alpha, device=device, dtype=dtype)

    n_fft = basis.shape[1]

    # Step 2: ensure n_fft >= 2^(1 + ceil(log2(hop_length)))
    min_fft = int(2.0 ** (1 + math.ceil(math.log2(hop_length))))
    if n_fft < min_fft:
        n_fft = min_fft

    # Step 3: re-normalize basis w.r.t. FFT window length
    # librosa: ``basis *= lengths[:, np.newaxis] / float(n_fft)``
    basis = basis * (lengths / float(n_fft)).unsqueeze(1)

    # Step 4: FFT and keep non-negative frequencies
    fft_basis = torch.fft.fft(basis, n=n_fft, dim=1)[:, :n_fft // 2 + 1]

    return fft_basis, n_fft, lengths


# ══════════════════════════════════════════════════════════════════════════
# STFT / iSTFT — torch only
# ══════════════════════════════════════════════════════════════════════════

def _stft(y: torch.Tensor, n_fft: int, hop_length: int,
          pad_mode: str = "constant") -> torch.Tensor:
    """Centered STFT with ones window — matches ``librosa.stft(window='ones')``.

    Input:  (..., L)
    Output: (..., n_fft//2+1, n_frames)
    """
    window = torch.ones(n_fft, dtype=y.dtype, device=y.device)
    pad = n_fft // 2
    y_padded = _pad(y, pad, pad, mode=pad_mode)

    lead = y_padded.shape[:-1]
    flat = y_padded.reshape(-1, y_padded.shape[-1]) if lead else y_padded
    D = torch.stft(flat, n_fft, hop_length=hop_length,
                   win_length=n_fft, window=window,
                   center=False, normalized=False,
                   onesided=True, return_complex=True)
    if lead:
        D = D.reshape(*lead, D.shape[-2], D.shape[-1])
    return D.to(_to_complex(y.dtype))


def _istft(D: torch.Tensor, n_fft: int, hop_length: int,
           length: int | None = None) -> torch.Tensor:
    """Inverse STFT with ones window — matches ``librosa.istft(window='ones')``.

    Input:  (..., n_fft//2+1, n_frames)
    Output: (..., L)
    """
    window = torch.ones(n_fft, dtype=_to_float(D.dtype), device=D.device)

    lead = D.shape[:-2]
    flat = D.reshape(-1, D.shape[-2], D.shape[-1]) if lead else D
    y = torch.istft(flat, n_fft, hop_length=hop_length,
                    win_length=n_fft, window=window,
                    center=True, normalized=False,
                    onesided=True, length=length)
    if lead:
        y = y.reshape(*lead, y.shape[-1])
    return y


# ══════════════════════════════════════════════════════════════════════════
# librosa.core.__cqt_response  →  _cqt_response
# ══════════════════════════════════════════════════════════════════════════

def _cqt_response(
    y: torch.Tensor,
    n_fft: int,
    hop_length: int,
    fft_basis: torch.Tensor,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """``fft_basis @ stft(y)`` — torch clone of ``librosa.__cqt_response``."""
    D = _stft(y, n_fft, hop_length, pad_mode=pad_mode)
    return torch.matmul(fft_basis, D)


# ══════════════════════════════════════════════════════════════════════════
# Fully-batched streaming CQT — all operations GPU-vectorized + batched
# ══════════════════════════════════════════════════════════════════════════


def _compute_n_fft(
    freqs: torch.Tensor,
    sr: float,
    filter_scale: float,
    hop_length: int,
    alpha: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Compute n_fft and wavelet lengths without building any filters."""
    lengths = _wavelet_lengths(freqs, sr, filter_scale, _HANN_BANDWIDTH, alpha)
    max_len = int(lengths.max().item())
    n_fft = int(2 ** math.ceil(math.log2(max(1, max_len))))
    min_fft = int(2.0 ** (1 + math.ceil(math.log2(hop_length))))
    if n_fft < min_fft:
        n_fft = min_fft
    return n_fft, lengths


def _optimal_tile(
    n_filters: int,
    n_frames: int,
    n_fft: int,
    bytes_complex: int,
    vram_budget: int,
) -> tuple[int, int]:
    """Compute the GPU-optimal (K_filters, T_frames) tile for the matmul.

    For ``filter_bank (K × F) @ frames_rfft (F × T) → output (K × T)``
    the working-set memory is:
        W = (K + T) × F × bc + K × T × bc   (bc = bytes per complex element)

    With F = n_fft//2+1 and a fixed budget M, the AM-GM optimum with K = T is:
        K² + 2·K·F = M/bc   →   K = −F + √(F² + M/bc)

    Both K and T are clamped to [1, n_filters] and [1, n_frames] respectively.
    The result is the *starting* tile; OOM-halving in the caller will shrink it
    further if VRAM is more constrained than the budget suggests.
    """
    F = n_fft // 2 + 1
    bc = bytes_complex
    discriminant = F * F + vram_budget / bc
    K_opt = max(1, int(-F + math.sqrt(discriminant)))
    K_opt = min(K_opt, n_filters)
    # Given K, solve T: K×T×bc + T×F×bc + K×F×bc = M → T = (M/bc − K×F) / (F + K)
    denom = F + K_opt
    T_opt = max(1, int((vram_budget / bc - K_opt * F) // max(denom, 1)))
    T_opt = min(T_opt, n_frames)
    return K_opt, T_opt


def _vram_budget(device: torch.device, fraction: float = 0.6) -> int:
    """Return available VRAM budget in bytes for one tile allocation.

    Uses *fraction* of free VRAM on CUDA; falls back to 512 MB on CPU.
    """
    if device.type == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info(device)
            return max(64 * 1024 * 1024, int(free * fraction))
        except Exception:
            return 512 * 1024 * 1024
    return 512 * 1024 * 1024


def _try_or_halve(fn, batch_size, *args, **kwargs):
    """Call fn(batch_size, *args, **kwargs). On CUDA OOM, halve batch_size and retry.

    Returns (result, final_batch_size).
    """
    while True:
        try:
            return fn(batch_size, *args, **kwargs), batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise  # can't halve further
            batch_size = max(1, batch_size // 2)


def _build_wavelet_sub(
    K: int,
    freqs: torch.Tensor,
    lengths: torch.Tensor,
    n_fft: int,
    sr: float,
    window: str,
    device: torch.device,
    dtype: torch.dtype,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
) -> torch.Tensor:
    """Build wavelet filters in sub-batches of size K with OOM halving.

    Returns (B, n_fft//2+1) complex on *device*.
    """
    cdtype = _to_complex(dtype)
    B = len(freqs)
    n_freq_bins = n_fft // 2 + 1
    fft_basis = torch.empty(B, n_freq_bins, dtype=cdtype, device=device)
    freq_idx = torch.arange(n_freq_bins, dtype=dtype, device=device)

    b_start = 0
    while b_start < B:
        b_end = min(b_start + K, B)
        k = b_end - b_start

        try:
            sub_freqs = freqs[b_start:b_end]
            sub_lengths = lengths[b_start:b_end]
            int_lens = sub_lengths.long()
            sub_max = int(int_lens.max().item())

            col = torch.arange(sub_max, dtype=dtype, device=device).unsqueeze(0)
            half = (sub_lengths / 2.0).unsqueeze(1)
            t = col - half
            mask = col < int_lens.unsqueeze(1).to(dtype)

            phase = (2.0 * math.pi / sr) * sub_freqs.unsqueeze(1) * t
            sig = torch.polar(torch.ones_like(phase), phase)
            sig = sig * mask
            del phase

            n_vals = int_lens.unsqueeze(1).to(dtype)
            win = _periodic_window_bank(col, n_vals, window)
            win = win * mask
            sig = sig * win
            del t, win, col, mask, n_vals

            l1 = sig.abs().sum(dim=1, keepdim=True).clamp_min(1e-30)
            sig = sig / l1
            del l1

            sig = sig * (sub_lengths / float(n_fft)).unsqueeze(1)

            X = torch.fft.fft(sig, n=n_fft, dim=1)[:, :n_freq_bins]
            del sig

            lpads = (n_fft - int_lens) // 2
            shift_phase = (-2.0 * math.pi / n_fft) * lpads.unsqueeze(1).to(dtype) * freq_idx.unsqueeze(0)
            shift = torch.polar(torch.ones_like(shift_phase), shift_phase)
            X = X * shift
            del shift, shift_phase

            fft_basis[b_start:b_end] = X.to(cdtype)
            del X

            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Progress update
            if progress is not None:
                with progress._lock:
                    progress.filters_done = b_end
                    progress.filter_batch_size = K
                if on_progress is not None:
                    on_progress(progress)

            b_start = b_end  # success — advance

        except torch.cuda.OutOfMemoryError:
            # Free whatever partial tensors exist
            torch.cuda.empty_cache()
            if K <= 1:
                raise  # truly impossible
            K = max(1, K // 2)
            if progress is not None:
                with progress._lock:
                    progress.filter_batch_size = K
            # don't advance b_start — retry same chunk with smaller K

    del freq_idx
    return fft_basis


def _build_wavelet_batch(
    freqs: torch.Tensor,
    lengths: torch.Tensor,
    n_fft: int,
    sr: float,
    window: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 64,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
) -> torch.Tensor:
    """Build wavelet filters in adaptive sub-batches with OOM halving.

    Starts with *batch_size* filters at a time. On OOM, halves and retries.
    Fully vectorized within each sub-batch.

    Returns (B, n_fft//2+1) complex on *device*.
    """
    return _build_wavelet_sub(
        batch_size, freqs, lengths, n_fft, sr, window, device, dtype,
        progress=progress, on_progress=on_progress)


# ══════════════════════════════════════════════════════════════════════════
# Helpers for mmap-based streaming signal input
# ══════════════════════════════════════════════════════════════════════════

def _make_temp_npy() -> str:
    """Return a unique temp file path for a .npy mmap (not yet created)."""
    fd, path = tempfile.mkstemp(suffix=".cqt_tmp.npy")
    os.close(fd)
    return path


def _decimate_mmap_2x(
    src: np.ndarray,
    out_path: str,
    device: torch.device,
    resample_kw: "ResampleKW | None" = None,
    chunk_samples: int = 2_000_000,
) -> np.ndarray:
    """Decimate *src* 2× into a new numpy memmap at *out_path*.

    Processes *src* in chunks to avoid loading it entirely into RAM.
    Returns the open writable memmap (caller must keep reference alive).
    Edge artifacts from stateless chunk resampling are ~64 samples per
    chunk boundary — negligible for chunks >> 64 samples.
    """
    n_src = len(src)
    # torchaudio 2:1 resample gives ceil(n / 2) output samples
    n_dst = (n_src + 1) // 2
    dst = np.lib.format.open_memmap(out_path, mode="w+",
                                     dtype=src.dtype, shape=(n_dst,))
    kw = dict(resample_kw) if resample_kw else {}
    write_pos = 0
    read_pos = 0
    while read_pos < n_src:
        end = min(n_src, read_pos + chunk_samples)
        chunk_np = np.asarray(src[read_pos:end])
        chunk_t = torch.as_tensor(chunk_np, device=device,
                                   dtype=torch.float64 if chunk_np.dtype == np.float64
                                   else torch.float32)
        dec_t = TAF.resample(chunk_t, 2, 1, **kw)
        n_out = dec_t.shape[-1]
        dst[write_pos: write_pos + n_out] = dec_t.cpu().numpy()
        write_pos += n_out
        read_pos = end
        del chunk_t, dec_t
    dst.flush()
    return dst


def _cqt_response_streaming(
    y: "torch.Tensor | None",
    n_fft: int,
    hop_length: int,
    freqs: torch.Tensor,
    lengths: torch.Tensor,
    my_sr: float,
    full_sr: float,
    window: str,
    filter_scale: float,
    pad_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 0,  # 0 = uncapped: VRAM budget alone governs tile size
    shard_sink=None,          # callable(f_global_start, f_global_end, t_out_start, t_out_end, tile_np)
    sink_filter_offset: int = 0,
    sink_stride: int = 1,
    sink_common_cols: int | None = None,
    apply_scale: bool = False,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
    signal_mmap: "np.ndarray | None" = None,  # 1D numpy array; when set, y must be None
    signal_chunk_frames: int = 2048,           # output frames per VRAM chunk (mmap path)
    signal_offset: int = 0,                    # sample offset into signal_mmap where signal starts
) -> None:
    """Batch CQT with OOM-catch-and-halve on every allocation.

    Batches filters and frames independently. On any CUDA OOM the
    responsible batch size is halved and that chunk is retried.
    No size estimation — the GPU itself is the arbiter.

    Every completed tile is passed immediately to *shard_sink*:
        shard_sink(f_global_start, f_global_end, t_out_start, t_out_end, tile_np)
    where *tile_np* is a numpy array of shape (n_filters_tile, n_frames_tile)
    with complex dtype.  The sink writes real/imag to disk and handles
    backpressure — no tensor is retained in RAM beyond the current tile.

    When *signal_mmap* is provided, the signal is loaded into VRAM in
    chunks of *signal_chunk_frames* output frames at a time.  The full
    signal is never resident in VRAM simultaneously.
    """
    cdtype = _to_complex(dtype)
    n_filters = len(freqs)
    sr_scale = math.sqrt(full_sr / my_sr)

    # Determine signal length and set up the full-tensor path if needed.
    if signal_mmap is not None:
        sig_n = len(signal_mmap)
        sig_padded = None
        frames_view = None
    else:
        sig = y.reshape(-1)
        sig_n = sig.shape[-1]
        pad_amt = n_fft // 2
        if pad_mode == "constant":
            sig_padded = torch.nn.functional.pad(sig, (pad_amt, pad_amt))
        else:
            sig_padded = _pad(sig, pad_amt, pad_amt, mode=pad_mode)
        frames_view = sig_padded.unfold(0, n_fft, hop_length)

    n_frames = max(1, 1 + sig_n // hop_length)
    n_freq_bins = n_fft // 2 + 1

    # --- Compute optimal starting tile size from available VRAM ---
    # For the matmul filter_bank(K×F) @ frames_rfft(F×T) → (K×T):
    #   Working set = (K+T)×F×bc + K×T×bc  (bc = bytes per complex).
    # The AM-GM optimum with K=T gives K = −F + √(F²+M/bc).
    # This is the *starting* size; OOM halving below handles tighter budgets.
    _bc = 8 if dtype == torch.float32 else 16  # bytes per complex element
    _budget = _vram_budget(device)
    _K_opt, _T_opt = _optimal_tile(n_filters, n_frames, n_fft, _bc, _budget)
    # batch_size=0 means uncapped — VRAM budget is the sole constraint.
    # Only apply the cap when an explicit positive limit was requested.
    f_bs = _K_opt if batch_size <= 0 else min(batch_size, _K_opt)
    _t_bs_init = _T_opt if batch_size <= 0 else min(batch_size, _T_opt)

    # --- Filter loop with OOM halving ---
    f_start = 0
    while f_start < n_filters:
        f_end = min(f_start + f_bs, n_filters)

        try:
            fft_rows = _build_wavelet_batch(
                freqs[f_start:f_end], lengths[f_start:f_end],
                n_fft, my_sr, window, device, dtype, batch_size=f_bs,
                progress=progress, on_progress=on_progress)
            fft_rows = fft_rows * sr_scale

            if progress is not None:
                with progress._lock:
                    progress.filter_batches_total = math.ceil(n_filters / max(1, f_bs))
                    progress.frame_batches_total = math.ceil(n_frames / max(1, batch_size))

            # --- Frame loop (supports both full-tensor and mmap-chunk paths) ---
            t_bs = _t_bs_init  # start from VRAM-optimal T, halve on OOM
            t_start = 0
            while t_start < n_frames:
                if signal_mmap is not None:
                    # ── Mmap path: load one signal chunk into VRAM ──
                    t_chunk_end = min(t_start + signal_chunk_frames, n_frames)
                    n_chunk_f = t_chunk_end - t_start

                    # Signal samples for frames [t_start, t_chunk_end):
                    #   frame k spans mmap[signal_offset + k*hop - n_fft//2 :
                    #                      signal_offset + k*hop + n_fft//2]
                    # signal_offset shifts all reads so that frame 0 is centred
                    # on the true analysis start (not the pre-padding start).
                    # Reads that fall before sample 0 or after sig_n-1 are
                    # zero-filled — but with adequate pre/post padding those
                    # zero regions are never reached.
                    half = n_fft // 2
                    raw_s = signal_offset + t_start * hop_length - half
                    raw_e = signal_offset + (t_chunk_end - 1) * hop_length + half  # exclusive
                    act_s = max(0, raw_s)
                    act_e = min(sig_n, raw_e)
                    left_pad  = max(0, -raw_s)
                    right_pad = max(0, raw_e - sig_n)
                    total_samp = left_pad + (act_e - act_s) + right_pad
                    # total_samp == (n_chunk_f - 1)*hop + n_fft  ✓

                    chunk_np = np.empty(total_samp, dtype=signal_mmap.dtype)
                    if left_pad > 0:
                        chunk_np[:left_pad] = 0.0
                    if act_e > act_s:
                        chunk_np[left_pad: left_pad + (act_e - act_s)] = \
                            signal_mmap[act_s:act_e]
                    if right_pad > 0:
                        chunk_np[left_pad + (act_e - act_s):] = 0.0

                    chunk_t = torch.as_tensor(chunk_np, device=device, dtype=dtype)
                    del chunk_np
                    frames_chunk = chunk_t.unfold(0, n_fft, hop_length)
                    # frames_chunk shape: (n_chunk_f, n_fft)  ✓

                    # Inner frame sub-batching with OOM halving
                    tc = 0
                    while tc < n_chunk_f:
                        tc_end = min(tc + t_bs, n_chunk_f)
                        try:
                            F = torch.fft.rfft(
                                frames_chunk[tc:tc_end], n=n_fft, dim=1,
                            ).to(cdtype)
                            resp_batch = torch.matmul(fft_rows, F.T)
                            if apply_scale:
                                resp_batch = resp_batch / lengths[
                                    f_start:f_end
                                ].unsqueeze(-1).sqrt()

                            g_t0 = t_start + tc
                            g_t1 = t_start + tc_end
                            dst_f0 = sink_filter_offset + f_start
                            dst_f1 = sink_filter_offset + f_end
                            if sink_stride <= 1:
                                dst_t1 = min(sink_common_cols or n_frames, g_t1)
                                n_out_cols = dst_t1 - g_t0
                                if n_out_cols > 0 and shard_sink is not None:
                                    tile = resp_batch[..., :n_out_cols].cpu().numpy()
                                    shard_sink(dst_f0, dst_f1, g_t0, dst_t1, tile)
                                    del tile
                            else:
                                dst_t0 = g_t0 * sink_stride
                                dst_t1 = min(
                                    sink_common_cols or (n_frames * sink_stride),
                                    g_t1 * sink_stride)
                                if dst_t1 > dst_t0 and shard_sink is not None:
                                    expanded = resp_batch.repeat_interleave(
                                        sink_stride, dim=-1)
                                    expanded = expanded[..., :dst_t1 - dst_t0]
                                    tile = expanded.cpu().numpy()
                                    del expanded
                                    shard_sink(dst_f0, dst_f1, dst_t0, dst_t1, tile)
                                    del tile
                            del resp_batch, F

                            if progress is not None:
                                with progress._lock:
                                    progress.frame_batches_done += 1
                                    progress.frame_batch_size = t_bs
                                if on_progress is not None:
                                    on_progress(progress)
                            tc = tc_end  # success

                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if t_bs <= 1:
                                raise
                            t_bs = max(1, t_bs // 2)
                            # retry same tc

                    del chunk_t, frames_chunk
                    t_start = t_chunk_end  # advance by whole chunk

                else:
                    # ── Full-tensor path (existing behaviour) ──
                    t_end = min(t_start + t_bs, n_frames)
                    try:
                        F = torch.fft.rfft(
                            frames_view[t_start:t_end], n=n_fft, dim=1,
                        ).to(cdtype)

                        resp_batch = torch.matmul(fft_rows, F.T)
                        if apply_scale:
                            resp_batch = resp_batch / lengths[
                                f_start:f_end
                            ].unsqueeze(-1).sqrt()

                        dst_f0 = sink_filter_offset + f_start
                        dst_f1 = sink_filter_offset + f_end
                        if sink_stride <= 1:
                            dst_t1 = min(sink_common_cols or n_frames, t_end)
                            n_out_cols = dst_t1 - t_start
                            if n_out_cols > 0 and shard_sink is not None:
                                tile = resp_batch[..., :n_out_cols].cpu().numpy()
                                shard_sink(dst_f0, dst_f1, t_start, dst_t1, tile)
                                del tile
                        else:
                            dst_t0 = t_start * sink_stride
                            dst_t1 = min(sink_common_cols or (n_frames * sink_stride),
                                         t_end * sink_stride)
                            if dst_t1 > dst_t0 and shard_sink is not None:
                                expanded = resp_batch.repeat_interleave(sink_stride, dim=-1)
                                expanded = expanded[..., :dst_t1 - dst_t0]
                                tile = expanded.cpu().numpy()
                                del expanded
                                shard_sink(dst_f0, dst_f1, dst_t0, dst_t1, tile)
                                del tile
                        del resp_batch
                        del F

                        if progress is not None:
                            with progress._lock:
                                progress.frame_batches_done += 1
                                progress.frame_batch_size = t_bs
                            if on_progress is not None:
                                on_progress(progress)

                        t_start = t_end  # success — advance

                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if t_bs <= 1:
                            raise
                        t_bs = max(1, t_bs // 2)
                        # retry same t_start

            del fft_rows
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if progress is not None:
                with progress._lock:
                    progress.filter_batches_done += 1
                    progress.frame_batches_done = 0
                if on_progress is not None:
                    on_progress(progress)

            f_start = f_end  # success — advance

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if f_bs <= 1:
                raise
            f_bs = max(1, f_bs // 2)
            # retry same f_start

    del sig_padded, frames_view


# ══════════════════════════════════════════════════════════════════════════
# Resampling — torchaudio polyphase Kaiser-windowed sinc filter
# ══════════════════════════════════════════════════════════════════════════


class ResampleKW(TypedDict, total=False):
    """Keyword arguments forwarded to ``torchaudio.functional.resample``.

    All fields are optional — omitted keys use torchaudio defaults.

    lowpass_filter_width : int
        Number of zero-crossings ("taps") on each side of the sinc
        kernel.  Default 6.  Crank to 64-256+ for extreme quality;
        higher = sharper cutoff & better stopband but slower.
    rolloff : float
        Anti-aliasing cutoff as fraction of Nyquist.  Default 0.99.
        Lower values trade bandwidth for less aliasing.
    resampling_method : str
        ``"sinc_interp_hann"`` (default) or ``"sinc_interp_kaiser"``.
        Kaiser gives a tuneable sidelobe/mainlobe trade-off via *beta*.
    beta : float | None
        Kaiser window β.  Only used when *resampling_method* is
        ``"sinc_interp_kaiser"``.  Higher β = narrower mainlobe,
        lower sidelobes.  14.769656459379492 ≈ soxr "VHQ".
        None lets torchaudio choose (≈ 14.77).
    """

    lowpass_filter_width: int
    rolloff: float
    resampling_method: str
    beta: float | None


#: Preset that approximates soxr "Very High Quality".
RESAMPLE_VHQ: ResampleKW = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}


def _resample(y: torch.Tensor, orig_sr: int, target_sr: int,
              scale: bool = False, *,
              resample_kw: ResampleKW | None = None) -> torch.Tensor:
    """Resample along last dimension via torchaudio polyphase sinc filter.

    Parameters
    ----------
    scale : bool
        If True, apply librosa's energy scaling: ``y /= sqrt(ratio)``
        where ``ratio = target_sr / orig_sr``.
    resample_kw : ResampleKW, optional
        Extra keyword arguments forwarded to
        ``torchaudio.functional.resample``.  See :class:`ResampleKW`.
    """
    if orig_sr == target_sr:
        return y

    kw = dict(resample_kw) if resample_kw else {}
    y_out = TAF.resample(y, orig_sr, target_sr, **kw)

    if scale:
        ratio = target_sr / orig_sr
        y_out = y_out / math.sqrt(ratio)

    return y_out


# ══════════════════════════════════════════════════════════════════════════
# librosa.__trim_stack  →  _trim_stack
# ══════════════════════════════════════════════════════════════════════════

def _trim_stack(
    cqt_resp: list[torch.Tensor],
    n_bins: int,
) -> torch.Tensor:
    """Torch clone of ``librosa.__trim_stack``.

    cqt_resp[0] = top octave, cqt_resp[-1] = bottom octave.
    Output is ordered low-to-high: (..., n_bins, n_frames).
    """
    max_col = min(c.shape[-1] for c in cqt_resp)
    lead = cqt_resp[0].shape[:-2]
    shape = list(lead) + [n_bins, max_col]
    cqt_out = torch.empty(shape, dtype=cqt_resp[0].dtype,
                          device=cqt_resp[0].device)

    end = n_bins
    for c_i in cqt_resp:
        n_oct = c_i.shape[-2]
        if end < n_oct:
            cqt_out[..., :end, :] = c_i[..., -end:, :max_col]
        else:
            cqt_out[..., end - n_oct:end, :] = c_i[..., :max_col]
        end -= n_oct

    return cqt_out



# ══════════════════════════════════════════════════════════════════════════
# Per-octave variable BPO / hop helpers
# ══════════════════════════════════════════════════════════════════════════

# Type aliases for per-octave override callables.
#   octave_index: int — 0 = top (highest-frequency) octave
#   base_value:   int — the global default (bins_per_octave or hop_length)
# Must return an int.
OctaveBPOFunc = Callable[[int, int], int]
OctaveHopFunc = Callable[[int, int], int]
OctaveFloatFunc = Callable[[int, float], float]


class ExplicitOctaveSchedule:
    """Serializable explicit per-octave schedule callable."""

    def __init__(self, values: Sequence[int | float], *,
                 integer: bool = False) -> None:
        self._values = [int(v) if integer else float(v) for v in values]
        self._integer = integer
        self.n_octaves = len(self._values)

    def __call__(self, octave_idx: int,
                 base_value: int | float) -> int | float:
        if 0 <= octave_idx < len(self._values):
            val = self._values[octave_idx]
        else:
            val = base_value
        return int(round(val)) if self._integer else float(val)

    def to_list(self) -> list[int | float]:
        return list(self._values)


def _schedule_n_octaves(*funcs: object) -> int | None:
    """Return the first explicit schedule length advertised by a callable."""
    for func in funcs:
        n_octaves = getattr(func, "n_octaves", None)
        if isinstance(n_octaves, int) and n_octaves > 0:
            return n_octaves
    return None


def _common_hop_and_strides(hops_in_samples: Sequence[int]) -> tuple[int, list[int]]:
    """Compute a common time-grid step and per-octave strides."""
    common = 0
    for hop in hops_in_samples:
        common = hop if common == 0 else gcd(common, int(hop))
    common = max(common, 1)
    strides = [max(1, int(hop) // common) for hop in hops_in_samples]
    return common, strides


def _expand_octave_time_grid(
    octave_resp: torch.Tensor,
    stride: int,
    common_cols: int,
) -> torch.Tensor:
    """Sample-and-hold expand an octave response onto a common time grid."""
    if stride <= 1:
        return octave_resp[..., :common_cols]
    n_src = 1 + max(0, (common_cols - 1) // stride)
    trimmed = octave_resp[..., :n_src]
    expanded = trimmed.repeat_interleave(stride, dim=-1)
    return expanded[..., :common_cols]


def _build_variable_freq_grid(
    fmin: float,
    n_octaves: int,
    base_bpo: int,
    bpo_func: OctaveBPOFunc,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    """Build a non-uniform frequency grid with per-octave BPO.

    Returns
    -------
    freqs : (total_bins,) Tensor — frequencies low-to-high.
    bpo_per_oct : list[int] — BPO for each octave (index 0 = top octave).
    """
    # Compute BPO for each octave (0 = top, n_octaves-1 = bottom)
    bpo_per_oct = [bpo_func(i, base_bpo) for i in range(n_octaves)]

    # Build frequency list octave by octave, bottom-to-top
    # Octave j spans [fmin * 2^j, fmin * 2^(j+1))
    all_freqs: list[float] = []
    for j_from_bottom in range(n_octaves):
        oct_idx = n_octaves - 1 - j_from_bottom  # top-down index
        bpo = bpo_per_oct[oct_idx]
        oct_lo = fmin * (2.0 ** j_from_bottom)
        for k in range(bpo):
            f = oct_lo * (2.0 ** (k / bpo))
            all_freqs.append(f)

    freqs = torch.tensor(all_freqs, dtype=dtype, device=device)
    return freqs, bpo_per_oct


def _alpha_from_bpo(bpo: int) -> float:
    """Relative bandwidth for a constant-Q octave with *bpo* bins."""
    bpo = max(int(bpo), 1)
    num = 2.0 ** (2.0 / bpo) - 1.0
    den = 2.0 ** (2.0 / bpo) + 1.0
    return num / den


def _dwt_equivalent_filter_len(dec_len: int, level: int) -> int:
    """Equivalent analysis-filter length in original-rate samples."""
    dec_len = max(int(dec_len), 1)
    level = max(int(level), 1)
    return 1 + (dec_len - 1) * ((1 << level) - 1)


def fidelity_curve(
    sr: int,
    hop_length: int = 512,
    fmin: float = 16.35,
    fmax: float = 20000.0,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    window: str = "hann",
    bpo_func: OctaveBPOFunc | None = None,
    hop_func: OctaveHopFunc | None = None,
    filter_scale_func: OctaveFloatFunc | None = None,
) -> dict[str, np.ndarray]:
    """Compute the time-frequency fidelity curve for given CQT parameters.

    Pure math — no audio needed.  Returns arrays suitable for plotting.

    Parameters
    ----------
    sr : int
        Sample rate.
    hop_length : int
        Base hop length (may be overridden per-octave by *hop_func*).
    fmin, fmax : float
        Frequency range.
    bins_per_octave : int
        Base BPO (may be overridden per-octave by *bpo_func*).
    filter_scale : float
        CQT filter scale (Q multiplier).  May be overridden per-octave by
        *filter_scale_func*.
    window : str
        Window family used by the CQT basis.  Affects the effective
        frequency-domain main-lobe width.
    bpo_func : callable(octave_index, base_bpo) → int, optional
        Per-octave BPO override.
    hop_func : callable(octave_index, base_hop) → int, optional
        Per-octave hop override.
    filter_scale_func : callable(octave_index, base_filter_scale) → float, optional
        Per-octave filter scale override.  Same signature as hop_func.

    Returns
    -------
    dict with keys:
        freqs          : (N,) float — center frequencies Hz
        delta_t        : (N,) float — time resolution in seconds
        delta_f        : (N,) float — frequency resolution in Hz
        uncertainty    : (N,) float — Δt·Δf product (Gabor limit = 1/(4π))
        gabor_limit    : float — theoretical minimum Δt·Δf
        octave_index   : (N,) int — which octave each bin belongs to
        bpo_per_octave : list[int] — actual BPO per octave
        hop_per_octave : list[int] — actual hop per octave
        confidence     : (N,) float — composite [0,1] confidence
        conf_temporal  : (N,) float — envelope Nyquist margin (clamped 0–1)
        conf_nyquist   : (N,) float — proximity to Nyquist (clamped 0–1)
    """
    window = _canonical_cqt_window(window)
    window_bw = _window_bandwidth(window)
    n_octaves = max(1, int(math.ceil(math.log2(max(fmax / fmin, 1.001)))))

    # Per-octave BPO
    if bpo_func is not None:
        bpo_list = [bpo_func(i, bins_per_octave) for i in range(n_octaves)]
    else:
        bpo_list = [bins_per_octave] * n_octaves

    # Per-octave hop (account for octave decimation).
    # Build in top-down order (i=0 = top octave = full sr).
    # The base hop decimates by 2× per octave going downward, so octave i
    # sees decimated_hop = hop_length // 2^i (when evenly divisible).
    hop_list: list[int] = []
    for i in range(n_octaves):
        decimated_hop = hop_length >> i if hop_length % (1 << i) == 0 else hop_length // max(1, 2 ** i)
        if hop_func is not None:
            h_oct = hop_func(i, decimated_hop)
        else:
            h_oct = decimated_hop
        hop_list.append(h_oct)

    # Per-octave filter scale
    if filter_scale_func is not None:
        fs_list: list[float] = [filter_scale_func(i, filter_scale)
                                 for i in range(n_octaves)]
    else:
        fs_list = [filter_scale] * n_octaves

    freqs_list: list[float] = []
    dt_list: list[float] = []
    df_list: list[float] = []
    oct_idx_list: list[int] = []
    frame_dt_list: list[float] = []
    q_list: list[float] = []
    support_dt_list: list[float] = []

    for j_bottom in range(n_octaves):
        oct_top_idx = n_octaves - 1 - j_bottom  # 0 = top
        bpo = bpo_list[oct_top_idx]
        hop = hop_list[oct_top_idx]
        fs  = fs_list[oct_top_idx]
        oct_lo = fmin * (2.0 ** j_bottom)

        # Effective sr after decimation
        eff_sr = sr / (2.0 ** oct_top_idx) if oct_top_idx > 0 else sr

        for k in range(bpo):
            f = oct_lo * (2.0 ** (k / bpo))
            if f > fmax:
                break
            alpha = _alpha_from_bpo(bpo)
            Q = fs / max(alpha, 1e-30)
            delta_f = window_bw * f / max(Q, 1e-30)

            # Match the actual basis geometry: the CQT wavelet length is
            # Q * sr / f samples at the octave's effective sample rate.
            filter_support = Q / max(f, 1e-30)
            frame_spacing = hop / eff_sr
            delta_t = max(filter_support, frame_spacing)

            freqs_list.append(f)
            dt_list.append(delta_t)
            df_list.append(delta_f)
            oct_idx_list.append(oct_top_idx)
            q_list.append(Q)
            frame_dt_list.append(frame_spacing)
            support_dt_list.append(filter_support)

    freqs_arr = np.array(freqs_list)
    dt_arr = np.array(dt_list)
    df_arr = np.array(df_list)
    q_arr = np.array(q_list)
    frame_dt_arr = np.array(frame_dt_list)
    support_dt_arr = np.array(support_dt_list)
    unc_arr = dt_arr * df_arr
    gabor = 1.0 / (4.0 * math.pi)

    # ── Confidence metrics ──
    nyquist = sr / 2.0
    # Temporal confidence:  can the hop rate track each bin's envelope?
    # envelope_sr = 1/Δt, required Nyquist = 2·Δf
    # margin = envelope_sr / (2·Δf),  clamped to [0, 1]
    env_sr = 1.0 / np.maximum(frame_dt_arr, 1e-30)
    conf_temporal = np.minimum(1.0, env_sr / (2.0 * np.maximum(df_arr, 1e-30)))

    # Nyquist confidence: how much of the filter sits below sr/2?
    # If f + Δf/2 > nyquist the upper side-band is cut.
    headroom = (nyquist - freqs_arr) / np.maximum(df_arr, 1e-30)
    conf_nyquist = np.clip(headroom, 0.0, 1.0)

    # Composite
    confidence = conf_temporal * conf_nyquist

    # ── Three-tier RGB loss map ──────────────────────────────────────────
    # Each bin → (R, G, B) in [0,1].  Dark = good.  Bright = bad.
    #
    # BLUE — inherent Gabor overhead (the physics tax, always nonzero)
    #   How far above the Gabor limit is Δt·Δf?
    #   Use log scale: ratio 1→0, 10→0.5, 100→1.0
    gabor_ratio = unc_arr / gabor            # always ≥ 1 in theory
    tier_blue = np.clip(
        np.log2(np.maximum(gabor_ratio, 1.0)) / np.log2(100.0),
        0.0, 1.0).astype(np.float32)
    #
    # GREEN — degraded (losing detail, but data still exists)
    #   temporal aliasing onset: envelope_sr can't track envelope BW
    temporal_loss = 1.0 - conf_temporal
    #   temporal smearing: actual basis support > 50 ms → transient blur
    filter_len_sec = support_dt_arr
    smear_loss = np.clip((filter_len_sec - 0.05) / 0.45, 0.0, 1.0)
    #   spectral broadening: Δf/f exceeds a semitone (1/12 octave)
    relative_bw = df_arr / np.maximum(freqs_arr, 1e-30)
    semitone_bw = 2.0 ** (1.0 / 12) - 1.0   # ~0.0595
    half_oct_bw = 2.0 ** 0.5 - 1.0           # ~0.4142
    spectral_loss = np.clip(
        (relative_bw - semitone_bw) / (half_oct_bw - semitone_bw), 0.0, 1.0)
    tier_green = np.maximum(temporal_loss, np.maximum(smear_loss, spectral_loss))
    #
    # RED — catastrophic (hard data loss / unrecoverable)
    #   Nyquist truncation: upper sideband amputated
    nyquist_loss = 1.0 - conf_nyquist
    #   Severe temporal aliasing: margin < 0.5 → wholesale aliasing
    severe_alias = np.clip(1.0 - 2.0 * conf_temporal, 0.0, 1.0)
    tier_red = np.maximum(nyquist_loss, severe_alias)
    #
    # Stack into (N, 3) float32 array — ready for direct pixel use.
    loss_rgb = np.stack([tier_red, tier_green, tier_blue], axis=-1).astype(np.float32)

    return {
        "freqs": freqs_arr,
        "delta_t": dt_arr,
        "delta_f": df_arr,
        "uncertainty": unc_arr,
        "gabor_limit": gabor,
        "octave_index": np.array(oct_idx_list, dtype=np.int32),
        "bpo_per_octave": bpo_list,
        "hop_per_octave": hop_list,
        "filter_scale_per_octave": fs_list,
        "window": window,
        "Q": q_arr,
        "frame_spacing": frame_dt_arr,
        "filter_support": support_dt_arr,
        "confidence": confidence,
        "conf_temporal": conf_temporal,
        "conf_nyquist": conf_nyquist,
        "loss_rgb": loss_rgb,
    }


def _loss_rgb_from_metrics(
    freqs: np.ndarray,
    delta_t: np.ndarray,
    delta_f: np.ndarray,
    sr: int,
    filter_scale_or_Q: np.ndarray | float,
    sampling_delta_t: np.ndarray | None = None,
) -> np.ndarray:
    """Shared three-tier RGB loss computation used by all fidelity functions.

    Parameters are per-bin arrays of equal length.  Returns (N, 3) float32.
    """
    gabor = 1.0 / (4.0 * math.pi)
    unc = delta_t * delta_f
    nyquist = sr / 2.0

    # BLUE — Gabor overhead (log scale)
    gabor_ratio = unc / gabor
    tier_blue = np.clip(
        np.log2(np.maximum(gabor_ratio, 1.0)) / np.log2(100.0),
        0.0, 1.0).astype(np.float32)

    # GREEN — degraded
    sample_dt = delta_t if sampling_delta_t is None else sampling_delta_t
    env_sr = 1.0 / np.maximum(sample_dt, 1e-30)
    conf_temporal = np.minimum(1.0, env_sr / (2.0 * np.maximum(delta_f, 1e-30)))
    temporal_loss = 1.0 - conf_temporal
    Q_arr = np.asarray(filter_scale_or_Q, dtype=np.float64)
    if Q_arr.ndim == 0:
        Q_arr = np.full_like(freqs, float(Q_arr))
    # Smear loss: based on Q (cycle count), not absolute time.
    # A filter at 1 Hz lasting 8 seconds (Q=8) is fine — that's physics.
    # But Q > 50 means the filter is overresolving frequency at the cost
    # of temporal detail.  Q < 20 → 0,  Q > 200 → 1.
    smear_loss = np.clip((Q_arr - 20.0) / 180.0, 0.0, 1.0)
    relative_bw = delta_f / np.maximum(freqs, 1e-30)
    semitone_bw = 2.0 ** (1.0 / 12) - 1.0
    half_oct_bw = 2.0 ** 0.5 - 1.0
    spectral_loss = np.clip(
        (relative_bw - semitone_bw) / (half_oct_bw - semitone_bw), 0.0, 1.0)
    tier_green = np.maximum(temporal_loss, np.maximum(smear_loss, spectral_loss))

    # RED — catastrophic
    headroom = (nyquist - freqs) / np.maximum(delta_f, 1e-30)
    conf_nyquist = np.clip(headroom, 0.0, 1.0)
    nyquist_loss = 1.0 - conf_nyquist
    severe_alias = np.clip(1.0 - 2.0 * conf_temporal, 0.0, 1.0)
    tier_red = np.maximum(nyquist_loss, severe_alias)

    return np.stack([tier_red, tier_green, tier_blue], axis=-1).astype(np.float32)


def fidelity_curve_fb(
    sr: int,
    bands_per_octave: int = 12,
    fmin: float = 16.35,
    fmax: float = 20000.0,
    hop_length: int = 512,
    filter_type: str = "Linkwitz-Riley 4",
    bpo_func: OctaveBPOFunc | None = None,
    hop_func: OctaveHopFunc | None = None,
) -> dict[str, np.ndarray]:
    """Fidelity / loss-map for a filter-bank decomposition.

    This models the filter bank as a zero-phase residual split followed by
    Hilbert-envelope sampling.  Frequency resolution is set by band width,
    and time resolution is limited by the envelope hop and the band-limited
    modulation Nyquist criterion.  It does not assume causal ring-down.

    Parameters
    ----------
    bpo_func : callable(octave_index, base_bpo) → int, optional
        Per-octave bands-per-octave override.
    hop_func : callable(octave_index, base_hop) → int, optional
        Per-octave hop override.

    Returns dict with keys: freqs, delta_t, delta_f, loss_rgb,
    octave_index, bpo_per_octave, hop_per_octave.
    """
    fmax = min(fmax, sr / 2.0)
    n_octaves = max(1, int(math.ceil(math.log2(max(fmax / fmin, 1.001)))))

    # Per-octave BPO and hop
    bpo_list = [bpo_func(i, bands_per_octave) if bpo_func else bands_per_octave
                for i in range(n_octaves)]
    hop_list = [hop_func(i, hop_length) if hop_func else hop_length
                for i in range(n_octaves)]

    filter_orders = {
        "Linkwitz-Riley 4": 4,
        "Butterworth 4": 4,
        "Butterworth 8": 8,
    }
    if filter_type in filter_orders:
        filter_order = filter_orders[filter_type]
    elif filter_type.startswith("Linkwitz-Riley"):
        filter_order = max(2, int(filter_type.split()[-1]))
    elif filter_type.startswith("Butterworth"):
        filter_order = max(2, int(filter_type.split()[-1]))
    else:
        filter_order = 4

    freqs_list: list[float] = []
    dt_list: list[float] = []
    df_list: list[float] = []
    oct_idx_list: list[int] = []
    frame_dt_list: list[float] = []

    for j in range(n_octaves):
        bpo = bpo_list[j]
        hop = hop_list[j]
        oct_lo = fmin * (2.0 ** j)

        bw_ratio = 2.0 ** (1.0 / max(bpo, 1)) - 1.0
        hop_t = hop / sr

        for k in range(bpo):
            f = oct_lo * (2.0 ** (k / bpo))
            if f > fmax:
                break
            delta_f = f * bw_ratio
            delta_t = max(1.0 / max(2.0 * delta_f, 1e-30), hop_t)

            freqs_list.append(f)
            df_list.append(delta_f)
            dt_list.append(delta_t)
            oct_idx_list.append(j)
            frame_dt_list.append(hop_t)

    freqs = np.array(freqs_list, dtype=np.float64)
    delta_t = np.array(dt_list, dtype=np.float64)
    delta_f = np.array(df_list, dtype=np.float64)
    frame_dt = np.array(frame_dt_list, dtype=np.float64)

    Q_arr = freqs / np.maximum(delta_f, 1e-30)
    loss_rgb = _loss_rgb_from_metrics(
        freqs, delta_t, delta_f, sr, Q_arr, sampling_delta_t=frame_dt)
    return {
        "freqs": freqs,
        "delta_t": delta_t,
        "delta_f": delta_f,
        "loss_rgb": loss_rgb,
        "octave_index": np.array(oct_idx_list, dtype=np.int32),
        "bpo_per_octave": bpo_list,
        "hop_per_octave": hop_list,
        "frame_spacing": frame_dt,
        "filter_order": np.full_like(freqs, filter_order, dtype=np.float64),
        "filter_type": np.array([filter_type] * len(freqs), dtype=object),
    }


def fidelity_curve_dwt(
    sr: int,
    wavelet: str = "db4",
    level: int = 6,
    extension: str = "symmetric",
) -> dict[str, np.ndarray]:
    """Fidelity / loss-map for a discrete wavelet transform.

    The DWT is modelled as a dyadic analysis tree with one approximation
    band plus one detail band per decomposition level.  The per-level time
    support comes from the wavelet analysis filter length after dyadic
    upsampling through the cascade.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    level = max(1, int(level))

    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("fidelity_curve_dwt requires pywt") from exc

    wave = pywt.Wavelet(wavelet)
    dec_len = int(getattr(wave, "dec_len", 2))
    nyq = sr / 2.0

    freqs_list: list[float] = []
    dt_list: list[float] = []
    df_list: list[float] = []
    band_lo_list: list[float] = []
    band_hi_list: list[float] = []
    level_list: list[int] = []
    approx_mask: list[bool] = []
    names: list[str] = []
    support_dt_list: list[float] = []
    spacing_dt_list: list[float] = []

    approx_hi = nyq / (2.0 ** level)
    approx_bw = max(approx_hi, 1e-30)
    approx_support = _dwt_equivalent_filter_len(dec_len, level) / sr
    approx_spacing = (2.0 ** level) / sr
    freqs_list.append(max(approx_hi / 2.0, 1e-6))
    dt_list.append(max(approx_support, approx_spacing))
    df_list.append(approx_bw)
    band_lo_list.append(0.0)
    band_hi_list.append(approx_hi)
    level_list.append(level)
    approx_mask.append(True)
    names.append(f"A{level}")
    support_dt_list.append(approx_support)
    spacing_dt_list.append(approx_spacing)

    for lev in range(level, 0, -1):
        f_lo = nyq / (2.0 ** lev)
        f_hi = nyq / (2.0 ** (lev - 1))
        f_hi = min(f_hi, nyq)
        if f_hi <= f_lo:
            continue
        bw = max(f_hi - f_lo, 1e-30)
        support = _dwt_equivalent_filter_len(dec_len, lev) / sr
        spacing = (2.0 ** lev) / sr
        freqs_list.append(math.sqrt(f_lo * f_hi))
        dt_list.append(max(support, spacing))
        df_list.append(bw)
        band_lo_list.append(f_lo)
        band_hi_list.append(f_hi)
        level_list.append(lev)
        approx_mask.append(False)
        names.append(f"D{lev}")
        support_dt_list.append(support)
        spacing_dt_list.append(spacing)

    freqs = np.array(freqs_list, dtype=np.float64)
    delta_t = np.array(dt_list, dtype=np.float64)
    delta_f = np.array(df_list, dtype=np.float64)
    q_arr = freqs / np.maximum(delta_f, 1e-30)
    spacing_dt = np.array(spacing_dt_list, dtype=np.float64)
    loss_rgb = _loss_rgb_from_metrics(
        freqs, delta_t, delta_f, sr, q_arr, sampling_delta_t=spacing_dt)

    return {
        "freqs": freqs,
        "delta_t": delta_t,
        "delta_f": delta_f,
        "loss_rgb": loss_rgb,
        "band_lo": np.array(band_lo_list, dtype=np.float64),
        "band_hi": np.array(band_hi_list, dtype=np.float64),
        "level_index": np.array(level_list, dtype=np.int32),
        "is_approximation": np.array(approx_mask, dtype=bool),
        "band_name": np.array(names, dtype=object),
        "filter_support": np.array(support_dt_list, dtype=np.float64),
        "coefficient_spacing": spacing_dt,
        "dec_len": np.full_like(freqs, dec_len, dtype=np.float64),
        "wavelet": np.array([wavelet] * len(freqs), dtype=object),
        "extension": np.array([extension] * len(freqs), dtype=object),
    }


def fidelity_curve_cwt(
    sr: int,
    fmin: float = 0.1,
    fmax: float | None = None,
    scales_per_octave: int = 12,
    hop_length: int = 1,
    wavelet: str = "morlet",
    sigma: float = 6.0,
    bpo_func: OctaveBPOFunc | None = None,
    hop_func: OctaveHopFunc | None = None,
    sigma_func: OctaveFloatFunc | None = None,
) -> dict[str, np.ndarray]:
    """Fidelity / loss-map for a Continuous Wavelet Transform.

    The CWT's time-frequency trade-off is governed by the mother wavelet.
    For Morlet:  Δf = f / (σ√2),  Δt = σ / (2πf).
    For Ricker:  Δf ≈ f / 2,      Δt ≈ 1 / f.
    For Morse:   approximated as Morlet with σ=3 (broader).

    Parameters
    ----------
    bpo_func : callable(octave_index, base_spo) → int, optional
        Per-octave scales-per-octave override.
    hop_func : callable(octave_index, base_hop) → int, optional
        Per-octave hop override.
    sigma_func : callable(octave_index, base_sigma) → float, optional
        Per-octave Morlet σ override.  Only applied when wavelet == "morlet".

    Returns dict with keys: freqs, delta_t, delta_f, loss_rgb,
    octave_index, bpo_per_octave, hop_per_octave, sigma_per_octave.
    """
    if fmax is None:
        fmax = sr / 4.0
    fmax = min(fmax, sr / 2.0)
    fmin = max(fmin, 1e-3)

    n_octaves = max(1, math.ceil(math.log2(max(fmax / fmin, 1.001))))

    # Per-octave scales and hop
    spo_list = [bpo_func(i, scales_per_octave) if bpo_func else scales_per_octave
                for i in range(n_octaves)]
    hop_list = [hop_func(i, hop_length) if hop_func else hop_length
                for i in range(n_octaves)]
    sigma_list: list[float] = [sigma_func(i, sigma) if sigma_func else sigma
                                for i in range(n_octaves)]

    sqrt2 = math.sqrt(2.0)
    two_pi = 2.0 * math.pi

    freqs_list: list[float] = []
    dt_list: list[float] = []
    df_list: list[float] = []
    oct_idx_list: list[int] = []
    frame_dt_list: list[float] = []

    for j in range(n_octaves):
        spo = spo_list[j]
        hop = hop_list[j]
        sig = sigma_list[j]
        oct_lo = fmin * (2.0 ** j)
        hop_t = hop / sr

        for k in range(spo):
            f = oct_lo * (2.0 ** (k / spo))
            if f > fmax:
                break

            if wavelet == "morlet":
                delta_f = f / (sig * sqrt2)
                delta_t_filter = sig / (two_pi * max(f, 1e-30))
            elif wavelet == "ricker":
                delta_f = f / 2.0
                delta_t_filter = 1.0 / max(f, 1e-30)
            else:  # morse / fallback
                eff_sigma = 3.0
                delta_f = f / (eff_sigma * sqrt2)
                delta_t_filter = eff_sigma / (two_pi * max(f, 1e-30))

            delta_t = max(delta_t_filter, hop_t)

            freqs_list.append(f)
            df_list.append(delta_f)
            dt_list.append(delta_t)
            oct_idx_list.append(j)
            frame_dt_list.append(hop_t)

    freqs = np.array(freqs_list, dtype=np.float64)
    delta_t = np.array(dt_list, dtype=np.float64)
    delta_f = np.array(df_list, dtype=np.float64)
    frame_dt = np.array(frame_dt_list, dtype=np.float64)

    Q_arr = freqs / np.maximum(delta_f, 1e-30)
    loss_rgb = _loss_rgb_from_metrics(
        freqs, delta_t, delta_f, sr, Q_arr, sampling_delta_t=frame_dt)
    return {
        "freqs": freqs,
        "delta_t": delta_t,
        "delta_f": delta_f,
        "loss_rgb": loss_rgb,
        "octave_index": np.array(oct_idx_list, dtype=np.int32),
        "bpo_per_octave": spo_list,
        "hop_per_octave": hop_list,
        "sigma_per_octave": sigma_list,
        "frame_spacing": frame_dt,
    }


# ══════════════════════════════════════════════════════════════════════════
# Forward CQT — librosa.vqt (gamma=0)
# ══════════════════════════════════════════════════════════════════════════

def cqt(
    y: torch.Tensor,
    sr: int,
    hop_length: int = 512,
    fmin: float | None = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    window: str = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    resample_kw: ResampleKW | None = None,
    batch_size: int = 0,  # 0 = uncapped: VRAM budget alone governs tile size
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
    n_fft_max: int = 2**20,
    bpo_func: OctaveBPOFunc | None = None,
    hop_func: OctaveHopFunc | None = None,
    filter_scale_func: OctaveFloatFunc | None = None,
    shards: "dict | None" = None,  # {"real": np.memmap, "imag": np.memmap} — written directly, nothing returned
    signal_n_samples: "int | None" = None,  # true analysis length when mmap contains pre/post padding
    signal_offset: int = 0,  # sample offset INTO the mmap where the true signal starts
) -> "tuple[np.ndarray, int] | tuple[np.ndarray, int]":
    """Torch clone of ``librosa.cqt`` / ``librosa.vqt`` (gamma=0).

    Parameters
    ----------
    y : torch.Tensor (..., n_samples) float
    sr : int — sample rate
    dtype : torch.dtype, optional
        Float dtype for computation (e.g. torch.float32, torch.float64).
        Defaults to the input tensor's dtype.
    resample_kw : ResampleKW, optional
        Keyword arguments forwarded to ``torchaudio.functional.resample``.
        Use ``RESAMPLE_VHQ`` for soxr-VHQ-equivalent quality, or pass
        custom ``lowpass_filter_width`` / ``rolloff`` / ``beta`` values.
    n_fft_max : int
        Auto-resample gate. When an octave's n_fft exceeds this,
        the signal is pre-resampled down (2× repeatedly) until n_fft
        fits, while preserving Nyquist above the octave's fmax.
        Default 2**20 (~1M). Set to 0 to disable.
    bpo_func : callable(octave_index, base_bpo) → int, optional
        Per-octave bins-per-octave override.  ``octave_index`` 0 is the
        top (highest-frequency) octave.  When provided, ``n_bins`` is
        ignored and the total bin count is the sum of per-octave BPOs.
    hop_func : callable(octave_index, base_hop) → int, optional
        Per-octave hop-length override.  ``octave_index`` 0 is the top
        octave.  Receives the *current* decimated hop.
    filter_scale_func : callable(octave_index, base_filter_scale) → float, optional
        Per-octave filter-scale override.  ``octave_index`` 0 is the top
        octave.
    window : str, optional
        Window family for the CQT basis.  One of ``hann``, ``hamming``,
        ``blackman``, or ``blackmanharris``.

    Returns
    -------
    C     : complex (..., n_bins, n_frames) on *device*
    freqs : float (n_bins,) on *device*
    """
    _y_is_mmap = isinstance(y, np.ndarray)
    if not _y_is_mmap and not isinstance(y, torch.Tensor):
        y = torch.as_tensor(np.asarray(y))
    if device is None:
        device = y.device if isinstance(y, torch.Tensor) else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        if isinstance(y, torch.Tensor):
            dtype = y.dtype if y.dtype.is_floating_point else torch.float64
        else:
            dtype = torch.float64 if y.dtype == np.float64 else torch.float32
    if fmin is None:
        fmin = _C1_HZ
    window = _canonical_cqt_window(window)

    if _y_is_mmap:
        y_t = None                          # never load full array into VRAM
        y_n_samples = int(np.asarray(y).reshape(-1).shape[0])
    else:
        y_t = y.to(device=device, dtype=dtype)
        y_n_samples = int(y_t.shape[-1])

    schedule_octaves = _schedule_n_octaves(
        bpo_func, hop_func, filter_scale_func)

    # ── Per-octave BPO schedule ──
    if bpo_func is not None:
        # Variable BPO mode: compute octave count from fmin/fmax derived
        # from n_bins and base bpo, then build non-uniform grid.
        if schedule_octaves is not None:
            n_octaves = schedule_octaves
        else:
            fmax_target = fmin * 2.0 ** (n_bins / bins_per_octave)
            n_octaves = max(1, int(math.ceil(math.log2(
                max(fmax_target / fmin, 1.001)))))
        freqs, bpo_per_oct = _build_variable_freq_grid(
            fmin, n_octaves, bins_per_octave, bpo_func, device, dtype)
        n_bins = len(freqs)
        # Per-octave filter counts (top-down order, matching bpo_per_oct)
        filters_per_oct = list(bpo_per_oct)
    else:
        n_octaves = (schedule_octaves
                     if schedule_octaves is not None
                     else int(math.ceil(n_bins / bins_per_octave)))
        n_filters = min(bins_per_octave, n_bins)
        freqs = _cqt_frequencies(n_bins, fmin, bins_per_octave, device,
                                 dtype=dtype)
        bpo_per_oct = [bins_per_octave] * n_octaves
        filters_per_oct = [n_filters] * n_octaves
        # Last (bottom) octave may have fewer bins
        remainder = n_bins - n_filters * (n_octaves - 1)
        if remainder > 0 and n_octaves > 1:
            filters_per_oct[-1] = remainder

    if filter_scale_func is not None:
        fs_per_oct = [float(filter_scale_func(i, filter_scale))
                      for i in range(n_octaves)]
    else:
        fs_per_oct = [float(filter_scale)] * n_octaves

    # ── Guardrails ──
    fmax = freqs[-1].item()
    nyquist = sr / 2.0
    if fmax > nyquist:
        n_above = int((freqs > nyquist).sum().item())
        warnings.warn(
            f"fmax={fmax:.1f} Hz exceeds Nyquist={nyquist:.1f} Hz "
            f"(sr={sr}). {n_above}/{n_bins} bins are above Nyquist "
            f"and will contain aliased energy. "
            f"Raise sr to at least {int(math.ceil(fmax * 2))} "
            f"or reduce n_bins/fmin.",
            stacklevel=2,
        )
    # Check signal length vs decimation depth
    sig_len = y_n_samples  # works for both mmap and tensor paths
    deepest_sr = sr / (2 ** (n_octaves - 1)) if n_octaves > 1 else sr
    deepest_len = sig_len / (2 ** (n_octaves - 1)) if n_octaves > 1 else sig_len
    bottom_bpo = bpo_per_oct[-1]
    alpha_bottom = _alpha_from_bpo(bottom_bpo)
    Q_approx = fs_per_oct[-1] / max(alpha_bottom, 1e-30)
    wavelet_approx = Q_approx * deepest_sr / max(fmin, 1e-30)
    nfft_approx = 2 ** math.ceil(math.log2(max(1, wavelet_approx)))
    if deepest_len < nfft_approx:
        warnings.warn(
            f"Signal length ({sig_len} samples, {sig_len/sr:.2f}s) is short "
            f"for {n_octaves} octaves: bottom octave will have ~{deepest_len:.0f} "
            f"samples after decimation vs n_fft={nfft_approx}. "
            f"Bottom-octave bins will be heavily zero-padded. "
            f"Use a longer signal or fewer octaves for accurate low-frequency bins.",
            stacklevel=2,
        )
    # Memory estimate
    bytes_per_complex = 8 if dtype == torch.float32 else 16
    est_mem = sum(
        fp * (nfft_approx // 2 + 1) * bytes_per_complex
        for fp in filters_per_oct
    )
    if est_mem > 1e9:
        warnings.warn(
            f"Filter bank will require ~{est_mem / 1e9:.1f} GB. "
            f"This is expected for variable-BPO × "
            f"{n_octaves} octaves. Consider float32 dtype to halve memory.",
            stacklevel=2,
        )

    if n_bins == 1:
        r = 2.0 ** (1.0 / bins_per_octave)
        alpha = torch.tensor([(r ** 2 - 1) / (r ** 2 + 1)],
                             dtype=dtype, device=device)
    else:
        alpha = _relative_bandwidth(freqs)

    # ── Build per-octave slice schedule (top-down) ──
    # Slices into the freq/alpha/lengths arrays, grouped by octave.
    # freqs is ordered low-to-high.  Octave i=0 is top (highest freq).
    oct_slices: list[slice] = []
    end = n_bins
    for i in range(n_octaves):
        nf = filters_per_oct[i]
        start = end - nf
        oct_slices.append(slice(start, end))
        end = start

    lengths = torch.empty_like(freqs)
    for i, sl in enumerate(oct_slices):
        lengths[sl] = _wavelet_lengths(
            freqs[sl], sr, fs_per_oct[i], _HANN_BANDWIDTH, alpha[sl])

    # ── Progress init ──
    if progress is not None:
        with progress._lock:
            progress.n_octaves = n_octaves
            progress.total_start_time = time.monotonic()

    # ── Pre-pass: compute strides and common_cols analytically (no signal tensor) ──
    # All sample counts are tracked with integer arithmetic; no VRAM allocation.
    # _compute_n_fft does a GPU sync (.max().item()) on the per-octave freq
    # tensors (tiny), which is necessary and cheap.
    # Also caches (n_fft, lengths, n_extra_dec) per octave for the main loop.
    #
    # When signal_n_samples is provided (padded mmap path), use the TRUE
    # analysis length for frame-count arithmetic, not the padded mmap length.
    _plan_n = int(signal_n_samples if signal_n_samples is not None
                  else y_n_samples)  # signal sample count at current decimation level
    _plan_sr = float(sr)
    _plan_hop = hop_length
    _plan_scale = 1
    _plan_orig_hops: list[int] = []
    _plan_frames: list[int] = []
    _oct_plan: list[tuple[int, torch.Tensor, int]] = []  # (n_fft, lengths, n_extra_dec)
    for _pi in range(n_octaves):
        _psl = oct_slices[_pi]
        if hop_func is not None:
            _plan_hop = hop_func(_pi, _plan_hop)
        _pfreqs = freqs[_psl]
        _palpha = alpha[_psl]
        _pfs = fs_per_oct[_pi]
        _plan_orig_hops.append(int(_plan_hop * _plan_scale))
        _pn_fft, _poct_lengths = _compute_n_fft(_pfreqs, _plan_sr, _pfs,
                                                  _plan_hop, _palpha)
        _plan_oct_n = _plan_n       # signal sample count at this octave's rate
        _plan_oct_sr = _plan_sr
        _plan_oct_hop = _plan_hop
        _extra_dec = 0
        if n_fft_max > 0:
            _pfmax = float(_pfreqs[-1].item())
            while _pn_fft > n_fft_max:
                if _plan_oct_sr / 4.0 < _pfmax:
                    break
                if _plan_oct_hop % 2 != 0:
                    break
                _plan_oct_sr /= 2.0
                _plan_oct_hop //= 2
                _plan_oct_n = (_plan_oct_n + 1) // 2   # ceil(n/2) — matches torchaudio
                _extra_dec += 1
                _pn_fft, _poct_lengths = _compute_n_fft(_pfreqs, _plan_oct_sr, _pfs,
                                                          _plan_oct_hop, _palpha)
        _oct_plan.append((_pn_fft, _poct_lengths, _extra_dec))
        _psig_len = max(_plan_oct_n, _pn_fft)
        _plan_frames.append(max(1, 1 + _psig_len // _plan_oct_hop))
        if _plan_hop % 2 == 0:
            _plan_hop //= 2
            _plan_sr /= 2.0
            _plan_scale *= 2
            _plan_n = (_plan_n + 1) // 2    # ceil(n/2) — matches torchaudio

    common_hop, strides = _common_hop_and_strides(_plan_orig_hops)
    common_cols = min(
        (_nf - 1) * _st + 1
        for _nf, _st in zip(_plan_frames, strides)
    )

    # Build per-tile shard sink from the shards dict.
    # The sink is called with (f_global_start, f_global_end, t_out_start, t_out_end, tile_np)
    # where tile_np is complex numpy.  It writes real+imag directly to the
    # pre-allocated memmap shards — no tensor is retained beyond the tile.
    if shards is not None:
        _shard_sink = shards["_sink"]
        _C_acc = None
    else:
        # No shards: accumulate tiles in-memory, return (C_tensor, freqs_np)
        _np_cdtype = np.complex64 if dtype == torch.float32 else np.complex128
        _C_acc = np.zeros((n_bins, common_cols), dtype=_np_cdtype)

        def _shard_sink(f0: int, f1: int, t0: int, t1: int,
                        tile: np.ndarray) -> None:
            _C_acc[f0:f1, t0:t1] = tile

    # ── Iterate down the octaves ──
    if _y_is_mmap:
        _mmap_1d = np.asarray(y).reshape(-1)
        _mmap_sr = float(sr)
        _mmap_hop = hop_length
        _mmap_scale = 1
        # _cur_mmap is the signal at the current (progressively decimated) rate.
        # _cur_mmap_owned: True when we allocated it and must delete it at cleanup.
        _cur_mmap = _mmap_1d
        _cur_mmap_owned = False
        _cur_mmap_path: str | None = None
        _all_temp_paths: list[str] = []
        _all_temp_mmaps: list[np.ndarray] = []
        # Track the offset (in SAMPLES at the current decimation level) where
        # the true analysis signal starts inside the padded mmap.  This is
        # halved by ceil-division each time the mmap is decimated 2×.
        _cur_offset: int = signal_offset
        try:
            for i in range(n_octaves):
                sl = oct_slices[i]
                if hop_func is not None:
                    _mmap_hop = hop_func(i, _mmap_hop)

                freqs_oct = freqs[sl]
                filter_scale_oct = fs_per_oct[i]
                n_fft, oct_lengths, n_extra = _oct_plan[i]

                # ── Auto-resample gate: apply n_extra additional 2× decimations ──
                oct_mmap = _cur_mmap
                oct_sr = _mmap_sr
                oct_hop = _mmap_hop
                oct_offset = _cur_offset  # offset into oct_mmap at this level
                oct_extra_paths: list[str] = []
                oct_extra_mmaps: list[np.ndarray] = []
                for _ in range(n_extra):
                    _tp = _make_temp_npy()
                    _all_temp_paths.append(_tp)
                    _dm = _decimate_mmap_2x(oct_mmap, _tp, device, resample_kw)
                    _all_temp_mmaps.append(_dm)
                    oct_extra_paths.append(_tp)
                    oct_extra_mmaps.append(_dm)
                    oct_mmap = _dm
                    oct_sr /= 2.0
                    oct_hop //= 2
                    oct_offset = (oct_offset + 1) // 2  # ceil — matches torchaudio decimation

                if progress is not None:
                    with progress._lock:
                        progress.octave = i
                        progress.octave_n_filters = len(freqs_oct)
                        progress.octave_n_fft = n_fft
                        progress.octave_freq_lo = freqs_oct[0].item()
                        progress.octave_freq_hi = freqs_oct[-1].item()
                        progress.filters_done = 0
                        progress.filter_batches_done = 0
                        progress.frame_batches_done = 0
                        progress.octave_start_time = time.monotonic()
                    if on_progress is not None:
                        on_progress(progress)

                oct_n = len(oct_mmap)
                if oct_n < n_fft:
                    warnings.warn(
                        f"Octave {i} mmap length {oct_n} < n_fft={n_fft}; "
                        f"freq [{freqs_oct[0].item():.1f}, {freqs_oct[-1].item():.1f}] Hz "
                        f"will be zero-padded.",
                        stacklevel=2,
                    )

                _cqt_response_streaming(
                    None, n_fft, oct_hop, freqs_oct, oct_lengths,
                    oct_sr, float(sr), window, filter_scale_oct, pad_mode, device,
                    dtype, batch_size,
                    shard_sink=_shard_sink,
                    sink_filter_offset=sl.start,
                    sink_stride=strides[i],
                    sink_common_cols=common_cols,
                    apply_scale=scale,
                    progress=progress,
                    on_progress=on_progress,
                    signal_mmap=oct_mmap,
                    signal_offset=oct_offset,
                )

                # Decimate for next octave (equivalent to my_y = _resample(my_y, 2→1))
                if _mmap_hop % 2 == 0:
                    _mmap_hop //= 2
                    _mmap_sr /= 2.0
                    _mmap_scale *= 2
                    _cur_offset = (_cur_offset + 1) // 2  # ceil — matches torchaudio
                    if i < n_octaves - 1:
                        _tp = _make_temp_npy()
                        _all_temp_paths.append(_tp)
                        _dm = _decimate_mmap_2x(_cur_mmap, _tp, device, resample_kw)
                        _all_temp_mmaps.append(_dm)
                        # Replace current mmap (we keep the old path in _all_temp_paths
                        # for final cleanup — don't delete early, Windows locks files).
                        _cur_mmap = _dm
                        _cur_mmap_owned = True
                        _cur_mmap_path = _tp
                else:
                    if i < n_octaves - 1:
                        warnings.warn(
                            f"hop_length becomes odd ({_mmap_hop}) at octave {i}; "
                            f"remaining {n_octaves - 1 - i} lower octave(s) will NOT "
                            f"be decimated. Use hop_length divisible by "
                            f"2**{n_octaves-1}={2**(n_octaves-1)} for full decimation.",
                            stacklevel=2,
                        )
        finally:
            # Release all mmap references so Windows can delete the files.
            _cur_mmap = None
            for _mm in _all_temp_mmaps:
                del _mm
            _all_temp_mmaps.clear()
            import gc as _gc
            _gc.collect()
            for _tp in _all_temp_paths:
                try:
                    os.unlink(_tp)
                except Exception:
                    pass

    else:
        # ── Tensor path (existing behaviour) ──
        my_y, my_sr, my_hop = y_t, float(sr), hop_length
        my_scale = 1

        for i in range(n_octaves):
            sl = oct_slices[i]

            # Per-octave hop override
            if hop_func is not None:
                my_hop = hop_func(i, my_hop)

            freqs_oct = freqs[sl]
            filter_scale_oct = fs_per_oct[i]

            # Use n_fft / lengths cached from the analytical pre-pass.
            n_fft, oct_lengths, n_extra = _oct_plan[i]

            # ── Auto-resample gate (apply pre-computed count — no while loop) ──
            oct_y = my_y
            oct_sr = my_sr
            oct_hop = my_hop
            for _ in range(n_extra):
                oct_sr /= 2.0
                oct_hop //= 2
                oct_y = _resample(oct_y, orig_sr=2, target_sr=1,
                                  scale=True, resample_kw=resample_kw)

            if progress is not None:
                with progress._lock:
                    progress.octave = i
                    progress.octave_n_filters = len(freqs_oct)
                    progress.octave_n_fft = n_fft
                    progress.octave_freq_lo = freqs_oct[0].item()
                    progress.octave_freq_hi = freqs_oct[-1].item()
                    progress.filters_done = 0
                    progress.filter_batches_done = 0
                    progress.frame_batches_done = 0
                    progress.octave_start_time = time.monotonic()
                if on_progress is not None:
                    on_progress(progress)

            # Pad signal if shorter than n_fft
            if oct_y.shape[-1] < n_fft:
                pad_ratio = n_fft / oct_y.shape[-1]
                warnings.warn(
                    f"Octave loop i={i} (sr_eff={oct_sr:.0f}): signal "
                    f"length {oct_y.shape[-1]} < n_fft={n_fft} "
                    f"({pad_ratio:.1f}× zero-pad). "
                    f"Freq range [{freqs_oct[0].item():.1f}, "
                    f"{freqs_oct[-1].item():.1f}] Hz will be degraded.",
                    stacklevel=2,
                )
                my_y_padded = _pad(oct_y, 0, n_fft - oct_y.shape[-1],
                                   mode="constant")
            else:
                my_y_padded = oct_y

            _cqt_response_streaming(
                my_y_padded, n_fft, oct_hop, freqs_oct, oct_lengths,
                oct_sr, float(sr), window, filter_scale_oct, pad_mode, device,
                dtype, batch_size,
                shard_sink=_shard_sink,
                sink_filter_offset=sl.start,
                sink_stride=strides[i],
                sink_common_cols=common_cols,
                apply_scale=scale,
                progress=progress,
                on_progress=on_progress,
            )

            # librosa: downsample for next octave
            if my_hop % 2 == 0:
                my_hop //= 2
                my_sr /= 2.0
                my_scale *= 2
                my_y = _resample(my_y, orig_sr=2, target_sr=1, scale=True,
                                 resample_kw=resample_kw)
            else:
                if i < n_octaves - 1:
                    warnings.warn(
                        f"hop_length becomes odd ({my_hop}) at octave loop "
                        f"i={i}; remaining {n_octaves - 1 - i} lower octave(s) "
                        f"will NOT be decimated (no 2× downsampling). "
                        f"Use a hop_length divisible by 2**{n_octaves-1} "
                        f"(={2**(n_octaves-1)}) for full decimation.",
                        stacklevel=2,
                    )

    freqs_np = freqs.cpu().numpy()
    if _C_acc is not None:
        return torch.as_tensor(_C_acc, device=device), freqs
    return freqs_np, common_cols



# ══════════════════════════════════════════════════════════════════════════
# Continuous Wavelet Transform — subsonic / scientific TF analysis
# ══════════════════════════════════════════════════════════════════════════

# ── Complex-log bandwidth ────────────────────────────────────────────────
#
# Standard constant-Q: σ is the same at every scale.  As center
# frequency f → 0 the wavelet becomes infinitely long.
#
# Complex-log bandwidth regularises this via the analytic continuation
# of the logarithm:
#
#     σ(f) = σ₀ · |ln(f + iε)| / |ln(f)|
#
# • f >> ε  →  |ln(f + iε)| ≈ |ln f|        →  σ ≈ σ₀  (constant-Q)
# • f → 0   →  |ln(iε)| = |ln ε + iπ/2|     →  σ → σ₀·|ln ε + iπ/2| / |ln f|
#                                                   → 0 (wider BW, shorter wavelet)
#
# One parameter ε (Hz) controls the transition.  A natural choice is
# the lowest physical frequency of interest (e.g. 0.01 Hz for seismic,
# C0 = 16.35 Hz for musical).


def _complex_log_sigma(
    freqs: torch.Tensor,
    sigma_0: float,
    epsilon: float,
) -> torch.Tensor:
    """Compute per-scale sigma via complex-log bandwidth.

    Parameters
    ----------
    freqs : Tensor (n_scales,)
        Center frequencies in Hz, positive.
    sigma_0 : float
        Base sigma (constant-Q value at high frequencies).
    epsilon : float
        Regularisation frequency (Hz).  Controls where the transition
        from constant-Q to widening bandwidth begins.

    Returns
    -------
    sigmas : Tensor (n_scales,) same dtype/device as *freqs*.
    """
    # |ln(f + iε)| = sqrt(ln²|f+iε| + atan2²(ε, f))
    #              = sqrt( (0.5·ln(f²+ε²))² + atan2(ε, f)² )
    f = freqs.clamp(min=1e-30)
    mag_sq = f * f + epsilon * epsilon
    ln_mag = 0.5 * torch.log(mag_sq)
    angle = torch.atan2(
        torch.full_like(f, epsilon),
        f,
    )
    complex_ln_abs = torch.sqrt(ln_mag * ln_mag + angle * angle)

    # |ln(f)|  — real log of the frequency
    real_ln_abs = torch.abs(torch.log(f))
    # Avoid division by zero at f = 1 Hz (where ln f = 0)
    real_ln_abs = real_ln_abs.clamp(min=1e-10)

    ratio = (complex_ln_abs / real_ln_abs).clamp(max=1.0)
    return sigma_0 * ratio


# ── Mother wavelet spectral definitions ──────────────────────────────────
#
# Each wavelet is defined *in the frequency domain* as a function of
# (omega, center_freq, sr).  Working in frequency domain means:
#   1. No n_fft-length time-domain buffers
#   2. Natural convolution via pointwise multiply
#   3. Analytic (one-sided spectrum) wavelets are trivial
#
# All wavelets return a 1-D complex tensor on the given device/dtype.


def _morlet_freq(
    omega: torch.Tensor,
    center_freq: float,
    sr: float,
    sigma: float = 6.0,
) -> torch.Tensor:
    """Morlet wavelet in frequency domain.

    ψ̂(ω) = π^{-1/4} · exp(-σ²/2 · (ω/ω₀ - 1)²)

    *sigma* controls the time-frequency trade-off: higher = better
    frequency resolution, longer wavelet in time.  Default 6.0 gives
    ~6 oscillations under the Gaussian envelope, good for most TF work.
    Analytic: positive frequencies only, zero for ω < 0.
    """
    w0 = 2.0 * math.pi * center_freq / sr
    ratio = omega / w0
    gauss = torch.exp(-0.5 * (sigma * (ratio - 1.0)) ** 2)
    # Analytic: zero out negative frequencies (omega <= 0)
    gauss = gauss * (omega > 0).to(gauss.dtype)
    norm = math.pi ** (-0.25)
    return (norm * gauss).to(omega.dtype)


def _morse_freq(
    omega: torch.Tensor,
    center_freq: float,
    sr: float,
    beta: float = 4.0,
    gamma: float = 3.0,
    **_unused: Any,
) -> torch.Tensor:
    """Generalized Morse wavelet in frequency domain.

    ψ̂(ω) = a_β,γ · ω^β · exp(-ω^γ)   for ω > 0, else 0.

    Scaled so peak is at *center_freq*.  (β,γ) tune the trade-off:
      - β/γ = time-bandwidth product, higher = more oscillations
      - γ controls skewness; γ=3 ≈ Morlet-like, γ=1 = Cauchy-like
    """
    w0 = 2.0 * math.pi * center_freq / sr
    # Peak of ω^β exp(-ω^γ) is at ω_peak = (β/γ)^(1/γ)
    w_peak = (beta / gamma) ** (1.0 / gamma)
    # Scale so that the peak aligns with w0:
    # ψ(ω/s) peaks at ω = s·ω_peak, want s·ω_peak = w0 → s = w0/ω_peak
    # In substituted form: ω_scaled = ω·(ω_peak/w0)
    w_scaled = omega * (w_peak / w0)
    pos = (omega > 0).to(omega.dtype)
    # Clamp to avoid 0^β when beta < 1
    ws_safe = torch.clamp(w_scaled * pos, min=1e-30)
    raw = (ws_safe ** beta) * torch.exp(-(ws_safe ** gamma)) * pos
    # Normalize to unit peak
    peak_val = (w_peak ** beta) * math.exp(-(w_peak ** gamma))
    return (raw / max(peak_val, 1e-30)).to(omega.dtype)


def _ricker_freq(
    omega: torch.Tensor,
    center_freq: float,
    sr: float,
) -> torch.Tensor:
    """Ricker (Mexican Hat) wavelet in frequency domain.

    ψ̂(ω) = (2/√3) · π^{-1/4} · ω² · exp(-ω²/2)

    Real-valued wavelet (symmetric spectrum).  Scaled so peak energy
    is at *center_freq*.  Good for detecting edges/transients in
    subsonic signals.
    """
    # Peak of ω² exp(-ω²/2) is at ω = √2
    w_peak = math.sqrt(2.0)
    w0 = 2.0 * math.pi * center_freq / sr
    # ω_scaled = ω·(ω_peak/w0) so peak lands at ω = w0
    w_scaled = omega * (w_peak / w0)
    norm = 2.0 / math.sqrt(3.0) * math.pi ** (-0.25)
    raw = norm * (w_scaled ** 2) * torch.exp(-0.5 * w_scaled ** 2)
    return raw.to(omega.dtype)


# Registry of mother wavelets
_CWT_WAVELETS: dict[str, Callable] = {
    "morlet": _morlet_freq,
    "morse": _morse_freq,
    "ricker": _ricker_freq,
}


def _build_cwt_wavelets_sub(
    K: int,
    center_freqs: torch.Tensor,
    n_fft: int,
    sr: float,
    device: torch.device,
    dtype: torch.dtype,
    wavelet_fn: Callable,
    wavelet_kw: dict,
    per_sigma: torch.Tensor | None = None,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
) -> torch.Tensor:
    """Build CWT wavelet bank in sub-batches with OOM halving.

    If *per_sigma* is given (n_scales,), each wavelet gets its own sigma
    override (complex-log bandwidth).

    Returns (n_scales, n_fft) complex on *device*.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    cdtype = _to_complex(dtype)
    B = len(center_freqs)
    # Full-spectrum omega for frequency-domain wavelets
    omega = 2.0 * math.pi * torch.arange(n_fft, dtype=dtype, device=device) / n_fft

    fft_basis = torch.empty(B, n_fft, dtype=cdtype, device=device)

    b_start = 0
    while b_start < B:
        b_end = min(b_start + K, B)

        try:
            sub_freqs = center_freqs[b_start:b_end]
            # Build each wavelet in the sub-batch
            rows = []
            for j in range(b_end - b_start):
                cf = float(sub_freqs[j].item())
                kw = dict(wavelet_kw)
                if per_sigma is not None:
                    kw["sigma"] = float(per_sigma[b_start + j].item())
                psi = wavelet_fn(omega, cf, sr, **kw)
                rows.append(psi.to(cdtype))
            fft_basis[b_start:b_end] = torch.stack(rows)
            del rows

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if progress is not None:
                with progress._lock:
                    progress.filters_done = b_end
                    progress.filter_batch_size = K
                if on_progress is not None:
                    on_progress(progress)

            b_start = b_end

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if K <= 1:
                raise
            K = max(1, K // 2)
            if progress is not None:
                with progress._lock:
                    progress.filter_batch_size = K

    del omega
    return fft_basis


def _cwt_convolve_streaming(
    y: torch.Tensor,
    n_fft: int,
    hop_length: int,
    center_freqs: torch.Tensor,
    sr: float,
    pad_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    wavelet_fn: Callable,
    wavelet_kw: dict,
    per_sigma: torch.Tensor | None = None,
    batch_size: int = 64,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
) -> torch.Tensor:
    """CWT via full-signal FFT convolution (ssqueezepy algorithm).

    1. Pad signal symmetrically to n_fft (power-of-2)
    2. FFT the entire padded signal once
    3. For each scale: multiply spectrum by wavelet, IFFT
    4. Unpad to original signal length, subsample by hop_length

    Returns (n_scales, n_frames) complex.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    cdtype = _to_complex(dtype)
    n_scales = len(center_freqs)

    sig = y.reshape(-1)
    N = sig.shape[0]

    # ── Symmetric pad to n_fft (n_fft >= N, power of 2) ──
    n1 = (n_fft - N) // 2
    n2 = n_fft - N - n1
    if pad_mode == "constant":
        sig_padded = torch.nn.functional.pad(sig, (n1, n2))
    else:
        sig_padded = _pad(sig, n1, n2, mode=pad_mode)

    # ── Single FFT of entire padded signal ──
    xh = torch.fft.fft(sig_padded.to(dtype)).to(cdtype)

    # ── Frequency axis for wavelet construction ──
    omega = (2.0 * math.pi * torch.arange(n_fft, dtype=dtype, device=device)
             / n_fft)

    # Output dimensions
    n_time = N  # unpadded signal length
    n_frames = max(1, 1 + (n_time - 1) // hop_length) if hop_length > 1 else n_time
    out = torch.empty(n_scales, n_frames, dtype=cdtype, device=device)

    # ── Scale loop with OOM halving ──
    bs = min(batch_size, n_scales)
    s_start = 0
    while s_start < n_scales:
        s_end = min(s_start + bs, n_scales)

        try:
            # Build wavelet batch at full signal length
            rows = []
            for j in range(s_start, s_end):
                cf = float(center_freqs[j].item())
                kw = dict(wavelet_kw)
                if per_sigma is not None:
                    kw["sigma"] = float(per_sigma[j].item())
                psi = wavelet_fn(omega, cf, sr, **kw)
                rows.append(psi.to(cdtype))
            psi_batch = torch.stack(rows)  # (chunk, n_fft)

            # Multiply spectra and IFFT — the entire ssqueezepy forward
            Wx_padded = torch.fft.ifft(psi_batch * xh.unsqueeze(0), dim=-1)

            # Unpad: keep only the original signal's time range
            Wx_unpad = Wx_padded[:, n1:n1 + N]

            # Subsample by hop_length
            if hop_length > 1:
                out[s_start:s_end] = Wx_unpad[:, ::hop_length]
            else:
                out[s_start:s_end] = Wx_unpad

            del rows, psi_batch, Wx_padded, Wx_unpad

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if progress is not None:
                with progress._lock:
                    progress.filter_batches_done += 1
                    progress.frame_batches_done = 0
                if on_progress is not None:
                    on_progress(progress)

            s_start = s_end

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)

    del xh, omega, sig_padded
    return out


def cwt(
    y: torch.Tensor,
    sr: int,
    freqs: torch.Tensor | np.ndarray | None = None,
    fmin: float = 0.1,
    fmax: float | None = None,
    n_scales: int = 128,
    scales_per_octave: int | None = None,
    hop_length: int = 1,
    wavelet: str = "morlet",
    wavelet_kw: dict | None = None,
    pad_mode: str = "reflect",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    batch_size: int = 64,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
    n_fft_max: int = 2 ** 20,
    epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous Wavelet Transform for subsonic / scientific TF analysis.

    Computes a complex-valued scalogram using analytic mother wavelets.
    Designed for frequencies below C0 (16.35 Hz) but works at any range.

    Parameters
    ----------
    y : Tensor (..., n_samples)
        Input signal.
    sr : int
        Sample rate in Hz.
    freqs : Tensor or ndarray, optional
        Explicit center frequencies (Hz), highest-first.
        If given, overrides fmin/fmax/n_scales/scales_per_octave.
    fmin : float
        Lowest center frequency (Hz).  Default 0.1 Hz.
    fmax : float or None
        Highest center frequency.  Default ``sr / 4`` (half Nyquist).
    n_scales : int
        Number of scales (if freqs not given and scales_per_octave not given).
    scales_per_octave : int or None
        If given, overrides n_scales.  Logarithmically spaced.
    hop_length : int
        Hop in samples between output frames.  Default 1 (every sample).
    wavelet : str
        Mother wavelet: ``"morlet"`` (complex, default),
        ``"morse"`` (complex, tuneable), ``"ricker"`` (real).
    wavelet_kw : dict or None
        Extra kwargs passed to the wavelet function (e.g. ``sigma``
        for Morlet, ``beta``/``gamma`` for Morse).
    pad_mode : str
        Padding mode for signal edges.  Default ``"reflect"``.
    device, dtype : torch.device, torch.dtype
        Computation device and dtype.  Defaults to input's.
    batch_size : int
        Initial batch size for OOM-halving loops.
    progress, on_progress : CQTProgress, CQTProgressCallback
        Progress observation (same API as cqt).
    n_fft_max : int
        Maximum FFT size before auto-resampling.  Default 2^20.
    epsilon : float, optional
        Complex-log bandwidth regularisation.  When set, the Morlet sigma
        is replaced by a per-scale sigma via
        ``σ(f) = σ₀ · |ln(f + iε)| / |ln(f)|``.
        At high f this is ≈ σ₀ (constant-Q).  Near DC the effective sigma
        shrinks → wider bandwidth → shorter wavelet → tractable FFT sizes.
        Only affects wavelets that accept a ``sigma`` keyword (Morlet).

    Returns
    -------
    (W, freqs) : (Tensor, Tensor)
        W : complex (..., n_scales, n_frames)
        freqs : float Tensor (n_scales,) — center frequencies, high-to-low.
    """
    if device is None:
        device = y.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if dtype is None:
        dtype = y.dtype if y.is_floating_point() else torch.float32
    y_t = y.to(device=device, dtype=dtype)

    if wavelet not in _CWT_WAVELETS:
        raise ValueError(
            f"Unknown wavelet {wavelet!r}. "
            f"Available: {sorted(_CWT_WAVELETS)}")
    wavelet_fn = _CWT_WAVELETS[wavelet]
    wkw = dict(wavelet_kw or {})

    # ── Build frequency grid ──
    if freqs is not None:
        if isinstance(freqs, np.ndarray):
            f_grid = torch.from_numpy(freqs).to(dtype=dtype, device=device)
        else:
            f_grid = freqs.to(dtype=dtype, device=device)
    else:
        if fmax is None:
            fmax = float(sr) / 4.0
        if fmax <= fmin:
            raise ValueError(f"fmax={fmax} must be > fmin={fmin}")

        if scales_per_octave is not None:
            n_octaves_span = math.log2(fmax / fmin)
            n_scales = max(1, int(round(n_octaves_span * scales_per_octave)))

        # Log-spaced from fmax down to fmin (high-to-low like CQT)
        f_grid = torch.logspace(
            math.log10(fmax), math.log10(fmin),
            steps=n_scales, dtype=dtype, device=device,
        )

    actual_n_scales = len(f_grid)

    # ── Complex-log bandwidth (per-scale sigma) ──
    sigma = wkw.get("sigma", 6.0)
    per_sigma: torch.Tensor | None = None
    if epsilon is not None:
        per_sigma = _complex_log_sigma(f_grid, sigma, epsilon)
        # Longest wavelet is at highest sigma (highest freq ≈ σ₀),
        # but compute from actual max to be safe.
        eff_sigma = float(per_sigma.max().item())
    else:
        eff_sigma = sigma

    # ── Determine n_fft ──
    # Must be >= signal length AND >= wavelet support (ssqueezepy: pad to
    # next power of 2 of the signal, ensuring wavelet is resolved).
    sig = y_t.reshape(-1)
    N = sig.shape[0]
    lowest_f = float(f_grid[-1].item())
    support_samples = int(2.0 * eff_sigma * sr / (2.0 * math.pi * lowest_f))
    n_fft = 2 ** math.ceil(math.log2(max(64, N, support_samples)))

    # ── Auto-resample gate ──
    work_sr = float(sr)
    work_hop = hop_length
    work_y = sig

    if n_fft_max > 0:
        while n_fft > n_fft_max:
            # Highest freq in grid must be below new Nyquist
            new_sr = work_sr / 2.0
            if float(f_grid[0].item()) >= new_sr / 2.0:
                break  # can't resample further without aliasing
            work_y = _resample(
                work_y.unsqueeze(0),
                orig_sr=int(work_sr), target_sr=int(new_sr),
                scale=False,
            ).squeeze(0)
            work_sr = new_sr
            work_hop = max(1, work_hop // 2)
            # Recompute n_fft for new sr
            N_work = work_y.shape[0]
            support_samples = int(2.0 * eff_sigma * work_sr / (2.0 * math.pi * lowest_f))
            n_fft = 2 ** math.ceil(math.log2(max(64, N_work, support_samples)))

    if progress is not None:
        with progress._lock:
            progress.n_octaves = 1
            progress.total_start_time = time.monotonic()
            progress.filters_total = actual_n_scales
            progress.filter_batches_total = math.ceil(
                actual_n_scales / max(1, batch_size))

    # ── Compute CWT ──
    W = _cwt_convolve_streaming(
        work_y, n_fft, work_hop, f_grid, work_sr,
        pad_mode, device, dtype,
        wavelet_fn, wkw,
        batch_size=batch_size,
        progress=progress, on_progress=on_progress,
        per_sigma=per_sigma,
    )

    return W, f_grid


# ══════════════════════════════════════════════════════════════════════════
# Inverse CQT — librosa.icqt
# ══════════════════════════════════════════════════════════════════════════

def icqt(
    C: torch.Tensor,
    sr: int,
    hop_length: int = 512,
    fmin: float | None = None,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    window: str = "hann",
    bpo_func: OctaveBPOFunc | None = None,
    hop_func: OctaveHopFunc | None = None,
    filter_scale_func: OctaveFloatFunc | None = None,
    scale: bool = True,
    length: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    resample_kw: ResampleKW | None = None,
) -> torch.Tensor:
    """Torch clone of ``librosa.icqt``.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Float dtype for computation. Defaults to the input tensor's
        real component dtype.
    resample_kw : ResampleKW, optional
        Keyword arguments forwarded to ``torchaudio.functional.resample``.

    Returns
    -------
    y : float tensor (..., n_samples) on *device*.
    """
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(np.asarray(C))
    if device is None:
        device = C.device
    if dtype is None:
        dtype = _to_float(C.dtype) if C.dtype.is_complex else (
            C.dtype if C.dtype.is_floating_point else torch.float64)
    cdtype = _to_complex(dtype)
    if fmin is None:
        fmin = _C1_HZ
    window = _canonical_cqt_window(window)

    C_t = C.to(device=device, dtype=cdtype)

    n_bins = C_t.shape[-2]
    if bpo_func is not None:
        n_octaves = (_schedule_n_octaves(bpo_func, hop_func, filter_scale_func)
                     or int(math.ceil(n_bins / bins_per_octave)))
        freqs, bpo_per_oct = _build_variable_freq_grid(
            fmin, n_octaves, bins_per_octave, bpo_func, device, dtype)
        if len(freqs) != n_bins:
            raise ValueError(
                "Scheduled iCQT requires a frequency grid with exactly "
                f"{n_bins} bins, got {len(freqs)}")
        filters_per_oct = list(bpo_per_oct)
    else:
        n_octaves = int(math.ceil(n_bins / bins_per_octave))
        freqs = _cqt_frequencies(n_bins, fmin, bins_per_octave, device,
                                 dtype=dtype)
        n_filters = min(bins_per_octave, n_bins)
        filters_per_oct = [n_filters] * n_octaves
        remainder = n_bins - n_filters * (n_octaves - 1)
        if remainder > 0 and n_octaves > 1:
            filters_per_oct[-1] = remainder

    if n_bins == 1:
        r = 2.0 ** (1.0 / bins_per_octave)
        alpha = torch.tensor([(r ** 2 - 1) / (r ** 2 + 1)],
                             dtype=dtype, device=device)
    else:
        alpha = _relative_bandwidth(freqs)

    oct_slices: list[slice] = []
    end = n_bins
    for n_oct_filters in filters_per_oct:
        start = end - n_oct_filters
        oct_slices.append(slice(start, end))
        end = start

    fs_per_oct = [float(filter_scale_func(i, filter_scale))
                  if filter_scale_func is not None else float(filter_scale)
                  for i in range(n_octaves)]

    lengths = torch.empty_like(freqs)
    for i, sl in enumerate(oct_slices):
        lengths[sl] = _wavelet_lengths(
            freqs[sl], sr, fs_per_oct[i], _HANN_BANDWIDTH, alpha[sl])

    # Trim CQT frames if length is given
    if length is not None:
        n_frames = int(math.ceil((length + lengths.max().item()) / hop_length))
        C_t = C_t[..., :n_frames]

    # librosa: ``C_scale = np.sqrt(lengths)``
    C_scale = lengths.sqrt()

    # Build per-octave sr/hop schedule
    srs: list[float] = []
    hops: list[int] = []
    orig_hops: list[int] = []
    my_sr = float(sr)
    my_hop = hop_length
    my_scale = 1
    for i in range(n_octaves):
        if hop_func is not None:
            my_hop = hop_func(i, my_hop)
        srs.append(my_sr)
        hops.append(my_hop)
        orig_hops.append(int(my_hop * my_scale))
        if my_hop % 2 == 0:
            my_sr *= 0.5
            my_hop //= 2
            my_scale *= 2

    _, strides = _common_hop_and_strides(orig_hops)

    y: torch.Tensor | None = None

    for i, (my_sr, my_hop) in enumerate(zip(srs, hops)):
        sl = oct_slices[i]
        filter_scale_oct = fs_per_oct[i]
        stride = strides[i]

        fft_basis, n_fft, oct_lengths = _vqt_filter_fft(
            my_sr, freqs[sl], window, filter_scale_oct, my_hop,
            alpha=alpha[sl], device=device, dtype=dtype)

        # librosa: ``inv_basis = fft_basis.conjugate().T.todense()``
        inv_basis = fft_basis.conj().T  # (n_fft//2+1, n_oct_filters)

        # librosa: ``freq_power = 1 / np.sum(util.abs2(inv_basis), axis=0)``
        freq_power = 1.0 / (inv_basis.abs() ** 2).sum(dim=0)

        # librosa: ``freq_power *= n_fft / lengths[sl]``
        freq_power = freq_power * (n_fft / oct_lengths)

        # librosa einsum back-projection
        C_oct = C_t[..., sl, :]
        if stride > 1:
            C_oct = C_oct[..., ::stride]
        if scale:
            weighted = (C_scale[sl] * freq_power).unsqueeze(-1) * C_oct
        else:
            weighted = freq_power.unsqueeze(-1) * C_oct

        D_oct = torch.matmul(inv_basis.to(cdtype),
                             weighted.to(cdtype))

        y_oct = _istft(D_oct, n_fft, my_hop)

        # Upsample to full sr
        resample_factor = int(sr / my_sr)
        if resample_factor > 1:
            y_oct = _resample(y_oct, orig_sr=1, target_sr=resample_factor,
                              scale=False, resample_kw=resample_kw)

        if y is None:
            y = y_oct
        else:
            n = min(y.shape[-1], y_oct.shape[-1])
            y[..., :n] = y[..., :n] + y_oct[..., :n]

    assert y is not None

    if length is not None:
        if y.shape[-1] > length:
            y = y[..., :length]
        elif y.shape[-1] < length:
            y = _pad(y, 0, length - y.shape[-1], mode="constant")

    return y


# ══════════════════════════════════════════════════════════════════════════
# Inverse CWT — ssqueezepy one-integral algorithm
# ══════════════════════════════════════════════════════════════════════════


def _adm_ssq(wavelet_fn: Callable, wavelet_kw: dict, sr: float,
             ref_freq: float = 1.0) -> float:
    """Compute ssq admissibility constant: ∫ conj(ψ̂(ω)) / ω  dω,  ω > 0.

    Uses the same wavelet function as the forward CWT so the normalization
    is self-consistent.  Numerically integrates on a dense grid.
    """
    # Evaluate our wavelet on a fine grid of angular frequencies.
    # Our wavelet is psi(omega, center_freq, sr, **kw) defined on
    # omega = 2*pi*k/N.  For the admissibility integral we need the
    # *mother wavelet* (scale=1).  Our parametrization at ref_freq
    # with a synthetic sr gives the mother wavelet.
    #
    # Mother wavelet: psih(w) with peak at w = 2*pi*ref_freq/sr.
    # We want to integrate psih(w)/w from 0+ to inf.  Since the
    # wavelet decays as a Gaussian, we only need a finite range.
    N_int = 65536
    # Build omega as if it were an FFT grid of length N_int with
    # sr chosen so the wavelet peak lands at a well-sampled location.
    # Use sr = ref_freq * 2 * pi / 1.0 so w0 = 1.0 → peak at omega=1
    # But simpler: just use a fine linspace and call the wavelet.
    sigma = wavelet_kw.get("sigma", 6.0)
    w0 = 2.0 * math.pi * ref_freq / sr
    # Wavelet is concentrated around w0 with width ~ w0/sigma
    # Integrate from 0+ to 6*w0 (well beyond the Gaussian tail)
    w_max = max(10.0 * w0, 6.0 * sigma * w0)
    w = torch.linspace(1e-12, w_max, N_int, dtype=torch.float64)
    psi = wavelet_fn(w, ref_freq, sr, **wavelet_kw).to(torch.float64)
    # conj(psi)/w — psi is real for Morlet/Morse/Ricker in freq domain
    integrand = psi / w
    Cpsi = float(torch.trapezoid(integrand, w).item())
    return max(Cpsi, 1e-30)


def _icwt_reconstruct(
    W: torch.Tensor,
    freqs: torch.Tensor,
    n_fft: int,
    hop_length: int,
    sr: float,
    wavelet_fn: Callable,
    wavelet_kw: dict,
    pad_mode: str = "reflect",
    batch_size: int = 64,
    per_sigma: torch.Tensor | None = None,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse CWT — ssqueezepy one-integral algorithm (Eq 2.6 of [1]).

        x(n) = (2 / Cψ) · dj · Σ_s Re{W[s, n]}

    For L1-normalized wavelets on a log-frequency grid, the inverse is a
    straight sum of Re(Wx) across scales, times a scalar.  No per-scale
    weighting, no overlap-add.

    References
    ----------
    [1] Daubechies, Lu, Wu — Synchrosqueezed Wavelet Transforms (2011)
    [2] ssqueezepy: https://github.com/OverLordGoldDragon/ssqueezepy

    Parameters
    ----------
    W : complex Tensor (n_scales, n_frames)
    freqs : Tensor (n_scales,) — center frequencies Hz, high-to-low.
    n_fft : int — (unused, kept for API compat)
    hop_length : int — hop in samples.
    sr : float — sample rate.
    wavelet_fn, wavelet_kw — wavelet used in forward.
    pad_mode, batch_size, per_sigma — standard params.
    length : int or None — desired output length.

    Returns
    -------
    y : real Tensor (n_samples,)
    """
    device = W.device
    dtype = _to_float(W.dtype) if W.dtype.is_complex else W.dtype

    n_scales, n_frames = W.shape[-2], W.shape[-1]
    freqs_d = freqs.to(device=device, dtype=dtype)

    # ── nv (voices per octave) from the frequency grid ──
    if n_scales > 1:
        log2_ratios = torch.log2(freqs_d[:-1] / freqs_d[1:])
        nv = float(1.0 / log2_ratios.mean().item())
    else:
        nv = 1.0
    dj = math.log(2.0) / nv  # ln(2) / nv

    # ── Admissibility constant, computed from our wavelet ──
    Cpsi = _adm_ssq(wavelet_fn, wavelet_kw, sr,
                     ref_freq=float(freqs_d[n_scales // 2].item()))

    # ── One-integral inverse: x = (2/Cpsi) * dj * Σ Re{W} ──
    # (L1-norm, log scales → no per-scale weighting)
    coeff = (2.0 / Cpsi) * dj

    # Batched sum over scales (OOM halving)
    y_frames = torch.zeros(n_frames, dtype=dtype, device=device)
    bs = min(batch_size, n_scales)
    s_start = 0
    while s_start < n_scales:
        s_end = min(s_start + bs, n_scales)
        try:
            y_frames += W[s_start:s_end].real.to(dtype).sum(dim=0)
            s_start = s_end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)

    y_frames = coeff * y_frames

    # ── hop > 1: upsample back ──
    if hop_length == 1:
        if length is not None:
            y_frames = y_frames[:length]
        return y_frames

    out_len = length if length is not None else ((n_frames - 1) * hop_length + 1)
    y = torch.nn.functional.interpolate(
        y_frames.unsqueeze(0).unsqueeze(0),
        size=out_len,
        mode="linear",
        align_corners=True,
    ).squeeze()
    return y


def icwt(
    W: torch.Tensor,
    freqs: torch.Tensor,
    sr: int,
    hop_length: int = 1,
    wavelet: str = "morlet",
    wavelet_kw: dict | None = None,
    pad_mode: str = "reflect",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    batch_size: int = 64,
    n_fft_max: int = 2 ** 20,
    epsilon: float | None = None,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse Continuous Wavelet Transform.

    Reconstructs a time-domain signal from a complex CWT scalogram.

    Parameters
    ----------
    W : complex Tensor (n_scales, n_frames)
        CWT scalogram (output of ``cwt()``).
    freqs : Tensor (n_scales,)
        Center frequencies in Hz, as returned by ``cwt()``.
    sr : int
        Sample rate of the original signal.
    hop_length : int
        Hop length used in the forward CWT.
    wavelet : str
        Wavelet name (must match forward pass).
    wavelet_kw : dict, optional
        Wavelet keyword arguments (must match forward pass).
    pad_mode : str
        Padding mode (must match forward pass).
    epsilon : float, optional
        Complex-log bandwidth epsilon (must match forward pass).
    length : int, optional
        Desired output length in samples.

    Returns
    -------
    y : real Tensor (n_samples,)
    """
    if device is None:
        device = W.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if dtype is None:
        dtype = _to_float(W.dtype) if W.dtype.is_complex else (
            W.dtype if W.dtype.is_floating_point else torch.float32)

    if wavelet not in _CWT_WAVELETS:
        raise ValueError(
            f"Unknown wavelet {wavelet!r}. "
            f"Available: {sorted(_CWT_WAVELETS)}")
    wavelet_fn = _CWT_WAVELETS[wavelet]
    wkw = dict(wavelet_kw or {})

    f_grid = freqs.to(device=device, dtype=dtype)

    # Per-sigma only needed for Cpsi if epsilon was used
    per_sigma: torch.Tensor | None = None
    if epsilon is not None:
        sigma = wkw.get("sigma", 6.0)
        per_sigma = _complex_log_sigma(f_grid, sigma, epsilon)

    return _icwt_reconstruct(
        W, f_grid, 0, hop_length, float(sr),
        wavelet_fn, wkw,
        pad_mode=pad_mode,
        batch_size=batch_size,
        per_sigma=per_sigma,
        length=length,
    )


# ══════════════════════════════════════════════════════════════════════════
# Phase vocoder — audify subsonic/infrasound to audible range
# ══════════════════════════════════════════════════════════════════════════


def audify(
    y: torch.Tensor,
    sr: int,
    *,
    shift_factor: float | None = None,
    target_hz: float = 440.0,
    fmin: float | None = None,
    fmax: float | None = None,
    target_sr: int = 44100,
    scales_per_octave: int = 24,
    wavelet: str = "morlet",
    wavelet_kw: dict | None = None,
    epsilon: float | None = None,
    hop_length: int = 1,
    batch_size: int = 64,
    n_fft_max: int = 2 ** 20,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    progress: CQTProgress | None = None,
    on_progress: CQTProgressCallback | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Phase-vocoder: shift subsonic/infrasound up to audible range.

    Analyses the signal with CWT at its native frequencies, then
    reconstructs using wavelets centred at shifted (audible) frequencies.
    Temporal structure is preserved — a 60-second infrasound recording
    becomes a 60-second audible rendering.

    Parameters
    ----------
    y : Tensor (n_samples,)
        Input signal at its native sample rate.
    sr : int
        Native sample rate (e.g. 20 Hz for CTBTO/IMS data).
    shift_factor : float, optional
        Multiplicative frequency shift.  All CWT centre frequencies are
        multiplied by this factor for resynthesis.  If not given, it is
        computed automatically so that the *median* analysis frequency
        maps to ``target_hz``.
    target_hz : float
        When ``shift_factor`` is None, the median analysis frequency is
        mapped to this value.  Default 440 Hz (concert A).
    fmin : float, optional
        Lowest analysis frequency.  Default: ``sr / (signal_length / 4)``
        (≈ 4 cycles of the lowest resolvable frequency).
    fmax : float, optional
        Highest analysis frequency.  Default: ``sr / 4`` (half Nyquist).
    target_sr : int
        Output sample rate.  Default 44100.
    scales_per_octave : int
        Frequency resolution for the CWT.  Default 24 (quarter-tone).
    wavelet, wavelet_kw, epsilon, hop_length, batch_size, n_fft_max
        Forwarded to ``cwt()`` and ``icwt()``.
    device, dtype
        Computation device and float dtype.
    progress, on_progress
        Progress observation.

    Returns
    -------
    (audio, out_sr, freqs_analysis, freqs_audible)
        audio : real Tensor (n_samples_out,) — audible signal at *target_sr*.
        out_sr : int — output sample rate (= target_sr).
        freqs_analysis : Tensor (n_scales,) — original centre freqs (Hz).
        freqs_audible : Tensor (n_scales,) — shifted centre freqs (Hz).
    """
    if device is None:
        device = y.device if isinstance(y, torch.Tensor) else torch.device("cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if dtype is None:
        if isinstance(y, torch.Tensor) and y.is_floating_point():
            dtype = y.dtype
        else:
            dtype = torch.float32

    y_t = y.to(device=device, dtype=dtype) if isinstance(y, torch.Tensor) else \
        torch.as_tensor(y, dtype=dtype, device=device)

    sig = y_t.reshape(-1)
    n_samples = sig.shape[0]

    # ── Default frequency range ──
    if fmin is None:
        # At least 4 full cycles in the signal
        duration = n_samples / float(sr)
        fmin = max(1e-6, 4.0 / duration)
    if fmax is None:
        fmax = float(sr) / 4.0
    if fmax <= fmin:
        raise ValueError(
            f"fmax={fmax:.4g} must be > fmin={fmin:.4g}. "
            f"Signal too short or sr too low?")

    # ── Forward CWT at native frequencies ──
    W, freqs_analysis = cwt(
        sig, sr,
        fmin=fmin, fmax=fmax,
        scales_per_octave=scales_per_octave,
        hop_length=hop_length,
        wavelet=wavelet,
        wavelet_kw=wavelet_kw,
        pad_mode="reflect",
        device=device, dtype=dtype,
        batch_size=batch_size,
        n_fft_max=n_fft_max,
        epsilon=epsilon,
        progress=progress,
        on_progress=on_progress,
    )

    # ── Compute shift factor ──
    if shift_factor is None:
        median_f = float(freqs_analysis[len(freqs_analysis) // 2].item())
        if median_f < 1e-30:
            raise ValueError("Median analysis frequency is ~0; can't auto-shift")
        shift_factor = target_hz / median_f

    freqs_audible = freqs_analysis * shift_factor

    # ── Verify audible frequencies are below Nyquist of target_sr ──
    max_audible = float(freqs_audible[0].item())
    if max_audible >= target_sr / 2.0:
        # Reduce shift_factor to stay below Nyquist
        safe_max = target_sr / 2.0 * 0.9  # 90% of Nyquist
        shift_factor = safe_max / float(freqs_analysis[0].item())
        freqs_audible = freqs_analysis * shift_factor
        max_audible = float(freqs_audible[0].item())

    # ── Reconstruct at audible frequencies ──
    # We need the same wavelet shape but centred at audible freqs,
    # with n_fft appropriate for target_sr.
    wkw = dict(wavelet_kw or {})
    sigma = wkw.get("sigma", 6.0)

    # Per-sigma for audible freqs (epsilon relative to audible range)
    audible_per_sigma: torch.Tensor | None = None
    if epsilon is not None:
        # Scale epsilon by the same shift factor so the transition
        # region maps proportionally
        audible_epsilon = epsilon * shift_factor
        audible_per_sigma = _complex_log_sigma(
            freqs_audible, sigma, audible_epsilon)
        eff_sigma = float(audible_per_sigma.max().item())
    else:
        eff_sigma = sigma

    # n_fft for the audible reconstruction
    lowest_audible = float(freqs_audible[-1].item())
    support = int(2.0 * eff_sigma * target_sr / (2.0 * math.pi * lowest_audible))
    n_fft_audible = 2 ** math.ceil(math.log2(max(64, support)))

    # Clamp n_fft to n_fft_max (with auto-resample if needed)
    work_sr_out = float(target_sr)
    if n_fft_max > 0:
        while n_fft_audible > n_fft_max:
            new_sr = work_sr_out / 2.0
            if max_audible >= new_sr / 2.0:
                break
            work_sr_out = new_sr
            support = int(
                2.0 * eff_sigma * work_sr_out
                / (2.0 * math.pi * lowest_audible))
            n_fft_audible = 2 ** math.ceil(math.log2(max(64, support)))

    # Hop length for audible output — preserve frame count
    # Analysis produced n_frames from original signal.
    # For reconstruction, hop must give the same n_frames
    # at the audible n_fft size.
    n_frames = W.shape[-1]
    # Output length in working sr samples
    out_samples_working = int(n_samples * work_sr_out / float(sr))
    audible_hop = max(1, out_samples_working // max(1, n_frames))

    wavelet_fn = _CWT_WAVELETS[wavelet]

    # Build audible wavelet bank
    psi_audb = _build_cwt_wavelets_sub(
        batch_size, freqs_audible, n_fft_audible, work_sr_out,
        device, dtype, wavelet_fn, wkw,
        per_sigma=audible_per_sigma,
    )  # (n_scales, n_fft_audible) complex

    # Energy normalisation
    psi_energy = (psi_audb.abs() ** 2).sum(dim=0).clamp(min=1e-30)

    cdtype = _to_complex(dtype)
    win = torch.hann_window(n_fft_audible, dtype=dtype, device=device)

    out_len = n_fft_audible + (n_frames - 1) * audible_hop
    audio = torch.zeros(out_len, dtype=dtype, device=device)
    norm_buf = torch.zeros(out_len, dtype=dtype, device=device)

    s_bs = batch_size
    for t in range(n_frames):
        c_t = W[..., t].to(cdtype)

        # Frequency-domain synthesis: Σ_k c_k · ψ_audible_k(ω) / energy(ω)
        X_freq = torch.zeros(n_fft_audible, dtype=cdtype, device=device)
        s_start = 0
        bs = s_bs
        while s_start < W.shape[-2]:
            s_end = min(s_start + bs, W.shape[-2])
            try:
                X_freq += (psi_audb[s_start:s_end]
                           * c_t[s_start:s_end].unsqueeze(-1)).sum(dim=0)
                s_start = s_end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)

        X_freq = X_freq / psi_energy
        x_t = torch.fft.ifft(X_freq).real * win

        idx = t * audible_hop
        audio[idx:idx + n_fft_audible] += x_t
        norm_buf[idx:idx + n_fft_audible] += win * win

    norm_buf = norm_buf.clamp(min=1e-30)
    audio = audio / norm_buf

    # Trim padding
    pad_amt = n_fft_audible // 2
    audio = audio[pad_amt:]
    if audio.shape[0] > pad_amt:
        audio = audio[:-pad_amt]

    # Upsample to target_sr if we downsampled during auto-resample
    if abs(work_sr_out - target_sr) > 0.5:
        audio = _resample(
            audio.unsqueeze(0),
            orig_sr=int(work_sr_out), target_sr=target_sr,
            scale=False,
        ).squeeze(0)

    return audio, target_sr, freqs_analysis, freqs_audible
