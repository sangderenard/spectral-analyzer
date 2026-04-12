"""Perfectly-reconstructing Linkwitz-Riley crossover filter bank.

Frequency-domain, zero-phase, GPU-accelerated implementation.
Sum of all output bands == input signal to machine precision.

Binary-tree topology — at each 2-way split:
    LP(fc) = X · 1/(1 + (f/fc)^order)        Linkwitz-Riley lowpass
    HP(fc) = X − LP                           residual → exact complement

This guarantees the leaf-band sum telescopes to the root at every
level of the tree, so  Σ bands == original  by construction.

    ┌─ LP(f₀) ─────────────── band 0  (DC … f₀)
    ├─ LP(f₁) ─┤
    │           └─ HP(f₀) ──── band 1  (f₀ … f₁)
  X ─┤
    │           ┌─ LP(f₂) ──── band 2  (f₁ … f₂)
    └─ HP(f₁) ─┤
                └─ HP(f₂) ──── band 3  (f₂ … Nyq)
"""
from __future__ import annotations

import math
import torch


# ── Public API ───────────────────────────────────────────────────────────


def lr_decompose(
    signal: torch.Tensor,
    crossovers: list[float] | torch.Tensor,
    sr: int,
    order: int = 8,
) -> torch.Tensor:
    """Split *signal* into perfectly-reconstructing frequency bands.

    Parameters
    ----------
    signal : (..., n_samples)
        Input waveform.  Any leading batch dimensions are preserved.
    crossovers : list[float] or Tensor
        Crossover frequencies in Hz (any order, duplicates OK).
        Values outside (0, Nyquist) are silently dropped.
        N valid crossovers → N+1 output bands.
    sr : int
        Sample rate in Hz.
    order : int
        Linkwitz-Riley order (positive even integer).
        Rolloff is ``order × 6  dB/oct``.
        Typical: 4 → 24 dB/oct,  8 → 48 dB/oct,  16 → 96 dB/oct.

    Returns
    -------
    Tensor (..., n_bands, n_samples)
        ``bands.sum(dim=-2) == signal`` to machine precision.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"order must be a positive even integer, got {order}")

    nyq = sr / 2.0
    if isinstance(crossovers, torch.Tensor):
        xo = sorted(set(crossovers.tolist()))
    else:
        xo = sorted(set(float(f) for f in crossovers))
    xo = [f for f in xo if 0 < f < nyq]

    N = signal.shape[-1]
    if not xo or N == 0:
        return signal.unsqueeze(-2)

    # Zero-pad to next power of 2 for FFT speed
    n_fft = 1 << math.ceil(math.log2(max(N, 2)))
    if n_fft > N:
        pad = signal.new_zeros(*signal.shape[:-1], n_fft - N)
        padded = torch.cat([signal, pad], dim=-1)
    else:
        padded = signal

    X = torch.fft.rfft(padded)

    # Frequency axis in the signal's own dtype (preserves precision)
    n_bins = X.shape[-1]
    freqs = torch.linspace(0, nyq, n_bins,
                           device=signal.device, dtype=signal.dtype)

    # Balanced binary-tree split
    bands_spec = _tree_split(X, freqs, xo, order)

    # IRFFT → time domain, strip padding
    bands = torch.stack(
        [torch.fft.irfft(B, n=n_fft)[..., :N] for B in bands_spec],
        dim=-2,
    )
    return bands


def lr_reconstruct(bands: torch.Tensor) -> torch.Tensor:
    """Sum bands back to a signal.  Inverse of :func:`lr_decompose`."""
    return bands.sum(dim=-2)


def lr_band_response(
    crossovers: list[float],
    sr: int,
    n_points: int = 4096,
    order: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the magnitude response of each band.

    Useful for plotting or verifying the crossover shape.

    Returns
    -------
    freqs : (n_points,)
        Frequency axis from 0 to Nyquist.
    responses : (n_bands, n_points)
        Per-band gain in [0, 1].  ``responses.sum(dim=0)`` is identically 1.
    """
    nyq = sr / 2.0
    xo = sorted(set(float(f) for f in crossovers))
    xo = [f for f in xo if 0 < f < nyq]

    freqs = torch.linspace(0, nyq, n_points)
    unity = torch.ones(n_points)

    if not xo:
        return freqs, unity.unsqueeze(0)

    responses = _tree_response(unity, freqs, xo, order)
    return freqs, torch.stack(responses)


def log_crossovers(
    fmin: float,
    fmax: float,
    bands_per_octave: int = 1,
) -> list[float]:
    """Generate log-spaced crossover frequencies between *fmin* and *fmax*.

    Parameters
    ----------
    fmin, fmax : float
        Outer edges of the desired band range (Hz).
    bands_per_octave : int
        How many bands per octave (1 → octave bands, 3 → ⅓-octave, …).

    Returns
    -------
    list[float]
        Interior crossover frequencies (not including fmin/fmax themselves).
        Pass directly to :func:`lr_decompose`.
    """
    if fmin <= 0 or fmax <= fmin or bands_per_octave < 1:
        return []
    n_octaves = math.log2(fmax / fmin)
    n_bands = max(1, round(n_octaves * bands_per_octave))
    if n_bands <= 1:
        return []
    ratio = (fmax / fmin) ** (1.0 / n_bands)
    return [fmin * ratio ** i for i in range(1, n_bands)]


# ── Internals ────────────────────────────────────────────────────────────


def _tree_split(
    X: torch.Tensor,
    freqs: torch.Tensor,
    xo: list[float],
    order: int,
) -> list[torch.Tensor]:
    """Balanced binary tree of LR 2-way frequency-domain splits."""
    if not xo:
        return [X]

    mid = len(xo) // 2
    fc = xo[mid]

    # LR lowpass gain:  1 / (1 + (f/fc)^order)
    r = (freqs / fc).pow(order)
    lp = X / (1.0 + r)
    hp = X - lp                 # residual guarantees perfect reconstruction

    left = _tree_split(lp, freqs, xo[:mid], order)
    right = _tree_split(hp, freqs, xo[mid + 1:], order)
    return left + right


def _tree_response(
    parent: torch.Tensor,
    freqs: torch.Tensor,
    xo: list[float],
    order: int,
) -> list[torch.Tensor]:
    """Magnitude response variant of :func:`_tree_split` (real-valued)."""
    if not xo:
        return [parent]

    mid = len(xo) // 2
    fc = xo[mid]

    r = (freqs / fc).pow(order)
    lp = parent / (1.0 + r)
    hp = parent - lp

    left = _tree_response(lp, freqs, xo[:mid], order)
    right = _tree_response(hp, freqs, xo[mid + 1:], order)
    return left + right
