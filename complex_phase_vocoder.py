"""complex_phase_vocoder.py

Pure-complex phase vocoder for use by analytic_driver.

Input contract
--------------
  Z      : complex ndarray  (n_bins, n_frames) — the caller's analytic spectral
            representation (CQT, iCWT output, or any bin × frame complex matrix).
            The dtype is preserved.  No real signal is ever produced or consumed.

  freqs  : float ndarray  (n_bins,)
            Centre frequency (Hz) of each bin.  Used to compute the nominal
            per-frame phase advance for each bin so that the instantaneous-
            frequency tracker is calibrated correctly.

  hop    : int — frame-to-frame sample hop in the original representation.

  sr     : int — sample rate (Hz).

Public API
----------
  time_stretch(Z, freqs, hop, sr, rate, *, device=None) -> complex ndarray
      Resample along the frame axis by *rate* (>1 = slower, <1 = faster).
      Preserves phase coherence via IF-based unwrapping.

  pitch_shift(Z, freqs, bpo, semitones, *, device=None) -> complex ndarray
      Shift along the bin axis by *semitones*.  Magnitude is interpolated
      (linear); phase is carried with the interpolated bin.
      *bpo* = bins-per-octave of the representation.

  transform(Z, freqs, hop, sr, bpo,
            stretch_rate=1.0, pitch_semitones=0.0, *, device=None)
      Compose time-stretch then pitch-shift in one call.

All operations work on numpy arrays.  When torch is available and *device*
is supplied (or CUDA is present), the inner loops run on-device and the
result is returned as a numpy array with the original dtype.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def _np_cdtype(arr: np.ndarray) -> np.dtype:
    """Return the complex dtype matching *arr* (complex64 → complex64, else complex128)."""
    if arr.dtype == np.complex64:
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


def _float_dtype(cdtype: np.dtype) -> np.dtype:
    return np.dtype(np.float32) if cdtype == np.complex64 else np.dtype(np.float64)


# ---------------------------------------------------------------------------
# Nominal phase advance per bin per frame
# ---------------------------------------------------------------------------

def _nominal_phase_advance(freqs: np.ndarray, hop: int, sr: int) -> np.ndarray:
    """phi_adv[k] = 2π × freqs[k] × hop / sr  (radians per frame)."""
    return (2.0 * math.pi * hop / sr) * np.asarray(freqs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core: IF-phase-vocoder time-stretch (complex in, complex out)
# ---------------------------------------------------------------------------

def time_stretch(
    Z: np.ndarray,
    freqs: np.ndarray,
    hop: int,
    sr: int,
    rate: float,
    *,
    device=None,
) -> np.ndarray:
    """Time-stretch *Z* by *rate* using instantaneous-frequency phase tracking.

    Parameters
    ----------
    Z      : complex (n_bins, n_frames)
    freqs  : float   (n_bins,)
    hop    : int     — sample hop of the representation
    sr     : int     — sample rate
    rate   : float   — stretch factor; >1 = slower (more output frames)
    device : torch.device or None — if given (and torch available) runs on GPU

    Returns
    -------
    Z_out : complex ndarray (n_bins, n_out_frames), same dtype as *Z*
    """
    rate = max(float(rate), 1e-9)
    cdtype = _np_cdtype(Z)
    n_bins, n_frames = Z.shape
    if n_frames <= 1:
        return Z.copy()

    phi_adv = _nominal_phase_advance(freqs, hop, sr).astype(
        np.float32 if cdtype == np.complex64 else np.float64
    )  # (n_bins,)

    try:
        import torch
        return _time_stretch_torch(Z, phi_adv, rate, cdtype, device)
    except Exception:
        return _time_stretch_numpy(Z, phi_adv, rate, cdtype)


def _time_stretch_numpy(
    Z: np.ndarray,
    phi_adv: np.ndarray,
    rate: float,
    cdtype: np.dtype,
) -> np.ndarray:
    n_bins, n_frames = Z.shape
    fdtype = _float_dtype(cdtype)

    # output time positions in input-frame coordinates
    out_steps = np.arange(0.0, float(n_frames - 1), rate, dtype=np.float64)
    n_out = len(out_steps)
    if n_out == 0:
        return np.zeros((n_bins, 0), dtype=cdtype)

    out = np.empty((n_bins, n_out), dtype=cdtype)
    phase_acc = np.angle(Z[:, 0]).astype(np.float64)  # (n_bins,)

    for col_idx, t in enumerate(out_steps):
        i0 = int(math.floor(t))
        i1 = min(i0 + 1, n_frames - 1)
        frac = t - i0

        C0 = Z[:, i0]
        C1 = Z[:, i1]

        mag = (1.0 - frac) * np.abs(C0) + frac * np.abs(C1)

        dp = np.angle(C1) - np.angle(C0) - phi_adv
        dp -= 2.0 * math.pi * np.round(dp / (2.0 * math.pi))
        true_adv = phi_adv + dp

        phase_acc = phase_acc + true_adv
        out[:, col_idx] = (mag * np.exp(1j * phase_acc)).astype(cdtype)

    return out


def _time_stretch_torch(
    Z: np.ndarray,
    phi_adv: np.ndarray,
    rate: float,
    cdtype: np.dtype,
    device,
) -> np.ndarray:
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_bins, n_frames = Z.shape
    torch_cdtype = torch.complex64 if cdtype == np.complex64 else torch.complex128
    torch_fdtype = torch.float32   if cdtype == np.complex64 else torch.float64

    Z_t = torch.as_tensor(Z, dtype=torch_cdtype, device=device)  # (n_bins, n_frames)
    phi_t = torch.as_tensor(phi_adv, dtype=torch_fdtype, device=device)  # (n_bins,)

    time_steps = torch.arange(0.0, float(n_frames - 1), rate,
                               dtype=torch_fdtype, device=device)
    n_out = time_steps.shape[0]
    if n_out == 0:
        return np.zeros((n_bins, 0), dtype=cdtype)

    out_parts: list[torch.Tensor] = []
    phase_acc = torch.angle(Z_t[:, 0])  # (n_bins,)

    chunk_w = n_out  # start greedy; halve on OOM
    start = 0
    while start < n_out:
        end = min(start + chunk_w, n_out)
        try:
            ts = time_steps[start:end]                               # (K,)
            i0 = ts.floor().long().clamp(max=n_frames - 2)          # (K,)
            frac = ts - i0.to(torch_fdtype)                         # (K,)

            C0 = Z_t[:, i0]          # (n_bins, K)
            C1 = Z_t[:, i0 + 1]

            mag = (1.0 - frac) * C0.abs() + frac * C1.abs()

            dp = torch.angle(C1) - torch.angle(C0) - phi_t.unsqueeze(1)
            dp = dp - 2.0 * math.pi * torch.round(dp / (2.0 * math.pi))

            true_adv = phi_t.unsqueeze(1) + dp                     # (n_bins, K)
            cum = torch.cumsum(true_adv, dim=1)                     # (n_bins, K)
            phase_block = phase_acc.unsqueeze(1) + cum              # (n_bins, K)

            chunk_out = torch.polar(mag, phase_block)               # (n_bins, K) complex
            phase_acc = phase_block[:, -1]

            out_parts.append(chunk_out)
            start = end
        except (RuntimeError, MemoryError) as exc:
            if chunk_w <= 1:
                raise
            oom = isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
            if not oom:
                raise
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            chunk_w = max(1, chunk_w // 2)

    result = torch.cat(out_parts, dim=1)  # (n_bins, n_out)
    return result.cpu().numpy().astype(cdtype)


# ---------------------------------------------------------------------------
# Core: bin-axis pitch shift (no real signal ever)
# ---------------------------------------------------------------------------

def pitch_shift(
    Z: np.ndarray,
    freqs: np.ndarray,
    bpo: float,
    semitones: float,
    *,
    device=None,
) -> np.ndarray:
    """Shift pitch by *semitones* via interpolation along the bin axis.

    Parameters
    ----------
    Z        : complex (n_bins, n_frames)
    freqs    : float   (n_bins,) — centre Hz for each bin
    bpo      : float   — bins per octave of the representation
    semitones: float   — semitones to shift (positive = up)
    device   : unused (kept for API symmetry)

    Returns
    -------
    Z_out : complex ndarray (n_bins, n_frames), same dtype as *Z*
    """
    if abs(semitones) < 1e-9:
        return Z.copy()
    cdtype = _np_cdtype(Z)
    n_bins, n_frames = Z.shape
    bin_shift = semitones * (bpo / 12.0)

    # Source bin indices (float) in the original grid for each output bin
    src_bins = np.arange(n_bins, dtype=np.float64) - bin_shift
    i0 = np.floor(src_bins).astype(np.int64)
    frac = src_bins - i0.astype(np.float64)     # (n_bins,)

    # Clamp to valid range; out-of-range bins get zero
    valid = (i0 >= 0) & (i0 < n_bins - 1)
    i0 = np.clip(i0, 0, n_bins - 2)
    frac = np.clip(frac, 0.0, 1.0)

    # Linear interpolation of magnitude; carry phase from nearest source
    mag0 = np.abs(Z[i0, :])                     # (n_bins, n_frames)
    mag1 = np.abs(Z[i0 + 1, :])
    mag_out = (1.0 - frac[:, None]) * mag0 + frac[:, None] * mag1

    # Phase from the dominant source bin (nearest neighbour for phase)
    nearest = np.where(frac <= 0.5, i0, i0 + 1)
    phase_out = np.angle(Z[nearest, :])          # (n_bins, n_frames)

    Z_out = (mag_out * np.exp(1j * phase_out)).astype(cdtype)

    # Zero out bins that mapped outside the original range
    Z_out[~valid, :] = 0.0

    return Z_out


# ---------------------------------------------------------------------------
# Composed transform
# ---------------------------------------------------------------------------

def transform(
    Z: np.ndarray,
    freqs: np.ndarray,
    hop: int,
    sr: int,
    bpo: float,
    stretch_rate: float = 1.0,
    pitch_semitones: float = 0.0,
    *,
    device=None,
) -> np.ndarray:
    """Apply time-stretch then pitch-shift to the complex spectral array *Z*.

    Parameters
    ----------
    Z              : complex (n_bins, n_frames)
    freqs          : float   (n_bins,) — centre Hz per bin
    hop            : int     — frame hop in samples
    sr             : int     — sample rate
    bpo            : float   — bins per octave
    stretch_rate   : float   — >1 slower, <1 faster
    pitch_semitones: float   — semitones to transpose
    device         : torch.device or None

    Returns
    -------
    Z_out : complex ndarray, same dtype as *Z*
    """
    Z_out = Z
    if abs(stretch_rate - 1.0) > 1e-9:
        Z_out = time_stretch(Z_out, freqs, hop, sr, stretch_rate, device=device)
    if abs(pitch_semitones) > 1e-9:
        Z_out = pitch_shift(Z_out, freqs, bpo, pitch_semitones, device=device)
    return Z_out
