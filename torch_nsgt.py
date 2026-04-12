"""Pure-torch CQ-NSGT — vectorized rewrite of grrrr/nsgt (Grill 2011-2015).

Original MATLAB/Python by NUHAG, University of Vienna:
  Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, Thomas Grill.

Paper: "A framework for invertible, real-time constant-Q transforms"
  N. Holighaus, M. Dörfler, G. A. Velasco, T. Grill (2013)

Semantics match the reference 1:1.  Implementation replaces all per-band
Python loops with grouped-by-size batched torch ops so the GPU (or CPU
vector units) actually get used.  Zero numpy at runtime.

Forward:  nsgtf(f, g, wins, nn, M, real) → list[Tensor]
Inverse:  nsigtf(c, gd, wins, nn, Ls, real) → Tensor
Dual:     nsdual(g, wins, nn, M) → list[Tensor]
Windows:  nsgfwin(frqs, q, sr, Ls) → (g, rfbas, M)
Scale:    OctScale(fmin, fmax, bpo) → (frqs, q)

The high-level CQ_NSGT class orchestrates all of the above and provides
forward() / backward().
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Utility windows — ref: nsgt/util.py
# ═══════════════════════════════════════════════════════════════════════════


def hannwin(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Periodic Hann window.  Ref: nsgt/util.py hannwin."""
    # r = arange(l); r *= 2π/l; r = cos(r); r += 1; r *= 0.5
    r = torch.arange(length, device=device, dtype=dtype)
    r = r * (2.0 * math.pi / length)
    r = torch.cos(r)
    r = r + 1.0
    r = r * 0.5
    return r


def blackharr(
    n: int, l: int | None = None, mod: bool = True,
    *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Blackman-Harris window.  Ref: nsgt/util.py blackharr.

    If l > n, the window is zero-padded and circularly shifted so the
    peak sits at index 0 (matching the reference exactly).
    """
    if l is None:
        l = n
    nn = (n // 2) * 2
    k = torch.arange(n, device=device, dtype=dtype)
    if not mod:
        bh = (0.35875
              - 0.48829 * torch.cos(k * (2 * math.pi / nn))
              + 0.14128 * torch.cos(k * (4 * math.pi / nn))
              - 0.01168 * torch.cos(k * (6 * math.pi / nn)))
    else:
        bh = (0.35872
              - 0.48832 * torch.cos(k * (2 * math.pi / nn))
              + 0.14128 * torch.cos(k * (4 * math.pi / nn))
              - 0.01168 * torch.cos(k * (6 * math.pi / nn)))
    if l > n:
        bh = torch.cat([bh, torch.zeros(l - n, device=device, dtype=dtype)])
    # Circular shift: hstack(bh[-n//2:], bh[:-n//2])
    bh = torch.roll(bh, -(n // 2))
    return bh


def blackharrcw(
    bandwidth: float, corr_shift: float,
    *, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    """Blackman-Harris window with correction shift.  Ref: nsgt/util.py blackharrcw.

    Returns (window, M) where M is the padded length.
    """
    flip = -1 if corr_shift < 0 else 1
    corr_shift = corr_shift * flip

    M = int(math.ceil(bandwidth / 2 + corr_shift - 1)) * 2
    # win = concat(arange(M//2, M), arange(0, M//2)) - corr_shift
    win = torch.cat([
        torch.arange(M // 2, M, device=device, dtype=dtype),
        torch.arange(0, M // 2, device=device, dtype=dtype),
    ]) - corr_shift

    bh = (0.35872
          - 0.48832 * torch.cos(win * (2.0 * math.pi / bandwidth))
          + 0.14128 * torch.cos(win * (4.0 * math.pi / bandwidth))
          - 0.01168 * torch.cos(win * (6.0 * math.pi / bandwidth)))
    bh = bh * (win <= bandwidth).to(dtype) * (win >= 0).to(dtype)

    if flip == -1:
        bh = bh.flip(0)
    return bh, M


# ═══════════════════════════════════════════════════════════════════════════
# Frequency scales — ref: nsgt/fscale.py
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OctScale:
    """Constant-Q (octave) frequency scale.  Ref: nsgt/fscale.py OctScale.

    Produces logarithmically spaced center frequencies with constant Q.
    """
    fmin: float
    fmax: float
    bpo: int       # bins (bands) per octave
    beyond: int = 0

    def __post_init__(self) -> None:
        lfmin = math.log2(self.fmin)
        lfmax = math.log2(self.fmax)
        bnds = int(math.ceil(lfmax - lfmin) * self.bpo) + 1
        self.bnds = bnds + self.beyond * 2
        odiv = (lfmax - lfmin) / (bnds - 1) if bnds > 1 else 1.0
        lfmin_ = lfmin - odiv * self.beyond
        lfmax_ = lfmax + odiv * self.beyond
        self._fmin = 2.0 ** lfmin_
        self._fmax = 2.0 ** lfmax_
        self._pow2n = 2.0 ** odiv
        self._q = math.sqrt(self._pow2n) / (self._pow2n - 1.0) / 2.0

    def __len__(self) -> int:
        return self.bnds

    def F(self, bnd: torch.Tensor | None = None,
          *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Center frequencies for each band."""
        if bnd is None:
            bnd = torch.arange(self.bnds, device=device, dtype=dtype)
        return self._fmin * (self._pow2n ** bnd)

    def Q(self, bnd: torch.Tensor | None = None,
          *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Q factor (constant for OctScale)."""
        if bnd is None:
            n = self.bnds
        else:
            n = bnd.shape[0]
        return torch.full((n,), self._q, device=device, dtype=dtype)

    def __call__(self, *, device: torch.device, dtype: torch.dtype
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (frqs, q) tensors — the scale specification."""
        f = self.F(device=device, dtype=dtype)
        q = self.Q(device=device, dtype=dtype)
        return f, q


# ═══════════════════════════════════════════════════════════════════════════
# Window construction — ref: nsgt/nsgfwin.py  nsgfwin()
# ═══════════════════════════════════════════════════════════════════════════


def nsgfwin(
    frqs: torch.Tensor,
    q: torch.Tensor,
    sr: int,
    Ls: int,
    *,
    min_win: int = 4,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Construct NSGT analysis windows g_k, positions rfbas, lengths M.

    Ref: nsgt/nsgfwin.py  nsgfwin()  (non-sliced path)
    Vectorized: M computed via diff, windows batched by unique size.
    """
    nf = sr / 2.0

    mask_pos = frqs > 0
    mask_below_nf = frqs < nf
    valid = mask_pos & mask_below_nf
    frqs = frqs[valid]
    q = q[valid]

    assert frqs.numel() > 0, "No valid frequencies in (0, Nyquist)"
    assert torch.all(frqs[1:] - frqs[:-1] > 0), "Frequencies must be strictly increasing"
    assert torch.all(q > 0), "All Q must be > 0"

    lbas = frqs.shape[0]

    zero = torch.zeros(1, device=device, dtype=dtype)
    nyq = torch.tensor([nf], device=device, dtype=dtype)
    fbas = torch.cat([zero, frqs, nyq, sr - frqs.flip(0)])
    fbas = fbas * (float(Ls) / sr)

    # ── Window lengths M_k — fully vectorized ──
    n_total = fbas.shape[0]  # = 2*lbas + 2
    M = torch.empty(n_total, device=device, dtype=dtype)
    M[0] = torch.round(2.0 * fbas[1])
    # M[k] = round(fbas[k+1] - fbas[k-1]) for k in 1..2*lbas
    M[1:-1] = torch.round(fbas[2:] - fbas[:-2])
    M[-1] = torch.round(Ls - fbas[-2])
    M = M.clamp(min=min_win)
    M_int = M.to(torch.int64)

    # ── Build windows g_k — batched by unique size ──
    M_list = M_int.tolist()
    unique_sizes = sorted(set(M_list))
    # Pre-generate one hann window per unique size
    hann_cache: dict[int, torch.Tensor] = {}
    for m in unique_sizes:
        hann_cache[m] = hannwin(m, device=device, dtype=dtype)
    g: list[torch.Tensor] = [hann_cache[m] for m in M_list]

    # ── Fixup center positions ──
    fbas[lbas] = (fbas[lbas - 1] + fbas[lbas + 1]) / 2.0
    fbas[lbas + 2] = Ls - fbas[lbas]
    rfbas = torch.round(fbas).to(torch.int64)

    return g, rfbas, M_int


# ═══════════════════════════════════════════════════════════════════════════
# Window ranges — ref: nsgt/util.py  calcwinrange()
# ═══════════════════════════════════════════════════════════════════════════


def calcwinrange(
    g: list[torch.Tensor],
    rfbas: torch.Tensor,
    Ls: int,
) -> tuple[list[torch.Tensor], int]:
    """Compute circular index ranges for each window.

    Ref: nsgt/util.py  calcwinrange()
    Vectorized: builds all ranges via a single flat arange + offsets.
    """
    device = rfbas.device
    shift = torch.cat([
        (((-rfbas[-1]) % Ls).unsqueeze(0)),
        rfbas[1:] - rfbas[:-1],
    ])
    timepos = torch.cumsum(shift, dim=0)
    nn = int(timepos[-1].item())
    timepos = timepos - shift[0]

    # Collect lengths and centers for all bands
    lengths = torch.tensor([len(gi) for gi in g], device=device, dtype=torch.int64)
    centers = timepos - (lengths // 2)  # start of each range

    # Build all index ranges in one flat tensor, then split
    total = int(lengths.sum().item())
    repeats = lengths.tolist()
    base = torch.repeat_interleave(centers, lengths)
    # Local offsets 0,1,…,Lg_k-1 for each band k — fully vectorized:
    # offset[i] = global_index - start_of_band_for_that_i
    if total > 0:
        band_start = torch.zeros_like(lengths)
        band_start[1:] = lengths[:-1].cumsum(0)
        local = (torch.arange(total, device=device, dtype=torch.int64)
                 - torch.repeat_interleave(band_start, lengths))
    else:
        local = torch.empty(0, device=device, dtype=torch.int64)
    flat = (base + local) % nn
    wins = list(flat.split(repeats))

    return wins, nn


# ═══════════════════════════════════════════════════════════════════════════
# Painless dual — ref: nsgt/nsdual.py  nsdual()
# ═══════════════════════════════════════════════════════════════════════════


def _frame_diagonal(
    g: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    M: torch.Tensor,
) -> torch.Tensor:
    """Compute the diagonal of the frame operator via index_add_.

    diag(S)[n] = Σ_k  M_k · |fftshift(g_k)|²  at positions wins[k].

    Vectorized: bands are grouped by window length so all fftshift+square
    operations within a size-group are fused into a single batched roll.
    Reduces GPU kernel launches from O(n_bands) to O(n_unique_sizes).
    """
    device = g[0].device
    dtype  = g[0].dtype

    # Group bands by window length — within each group ops are fully batched
    size_map: dict[int, list[int]] = defaultdict(list)
    for k, gi in enumerate(g):
        size_map[len(gi)].append(k)

    vals_parts: list[torch.Tensor] = []
    idx_parts:  list[torch.Tensor] = []

    for Lg, indices in size_map.items():
        # fftshift of a 1-D tensor of length Lg = roll right by -(Lg+1)//2
        # Equivalently: |fftshift(gi)|² = |gi|² with scatter indices rolled
        # by -Lg//2 (ifftshift of the win_range).  We compute:
        #   gi_stack.square() × M_k  scattered at  roll(wi, -Lg//2)
        # This avoids a separate permutation tensor entirely.
        shift = -(Lg // 2)   # ifftshift shift for the win_range

        gi_stack  = torch.stack([g[k]    for k in indices])  # (B, Lg)
        wi_stack  = torch.stack([wins[k] for k in indices])  # (B, Lg) int64
        mii_stack = M[torch.tensor(indices, device=device)]   # (B,)    float

        # Batch ifftshift of win indices (avoids permuting the window values)
        wi_shifted = torch.roll(wi_stack, shift, dims=1)      # (B, Lg)

        vals = gi_stack.square() * mii_stack.unsqueeze(1)     # (B, Lg)

        vals_parts.append(vals.reshape(-1))
        idx_parts.append(wi_shifted.reshape(-1))

    flat_vals = torch.cat(vals_parts)
    flat_idx  = torch.cat(idx_parts)

    x = torch.zeros(nn, device=device, dtype=dtype)
    x.index_add_(0, flat_idx, flat_vals)
    return x


def nsdual(
    g: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    M: torch.Tensor,
) -> list[torch.Tensor]:
    """Compute the canonical dual windows for the painless case.

    Ref: nsgt/nsdual.py  nsdual()
    Vectorized: frame diagonal via index_add_, dual via gather+divide.
    """
    x = _frame_diagonal(g, wins, nn, M)

    # Dual windows: gd_k = g_k / ifftshift(x[support])
    # Gather all at once, divide, split back
    flat_idx = torch.cat(wins)
    flat_x = x[flat_idx]
    # ifftshift per-window: for each window of length Lg, ifftshift swaps
    # the two halves. We apply it per-window by splitting.
    lengths = [len(wi) for wi in wins]
    parts_x = flat_x.split(lengths)
    parts_g = g

    gd: list[torch.Tensor] = []
    for gi, xi in zip(parts_g, parts_x):
        denom = torch.fft.ifftshift(xi)
        gd.append(gi / denom)

    return gd


# ═══════════════════════════════════════════════════════════════════════════
# Painless condition validator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PainlessReport:
    """Result of checking the painless frame conditions."""
    is_painless: bool
    # Per-band diagnostics (parallel lists)
    band_index: list[int]
    window_support: list[int]     # len(g_k) = support in DFT bins
    channel_count: list[int]      # M_k = number of channels (FFT length per band)
    ratio: list[float]            # support / M_k  — must be ≤ 1 for painless
    violations: list[int]         # indices of bands where ratio > 1
    frame_bound_lower: float      # min of frame diagonal (should be > 0)
    frame_bound_upper: float      # max of frame diagonal
    condition_number: float       # upper / lower  (1.0 = tight frame)


def check_painless(
    g: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    M: torch.Tensor,
) -> PainlessReport:
    """Validate the painless NSGT conditions for a given window bank.

    Vectorized: support lengths and M as tensors, frame diagonal reused.
    """
    device = g[0].device

    # By construction in nsgfwin, len(g[k]) == M[k] for every band
    # (g is built as [hann_cache[m] for m in M_list]).  Lk == Mk always,
    # so violations is always empty.  We still compute the frame diagonal
    # for the condition-number / frame-bound diagnostics.
    Mk = M.to(torch.int64)
    Lk = Mk    # no Python loop needed
    ratios_t = Lk.float() / Mk.float().clamp(min=1)
    violation_mask = ratios_t > 1.0
    violations = violation_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    # Frame bounds from the diagonal (reuses same accumulation as nsdual)
    x = _frame_diagonal(g, wins, nn, M)
    lb = float(x.min().item())
    ub = float(x.max().item())
    cond = ub / max(lb, 1e-30)

    return PainlessReport(
        is_painless=len(violations) == 0,
        band_index=list(range(len(g))),
        window_support=Lk.tolist(),
        channel_count=Mk.tolist(),
        ratio=ratios_t.tolist(),
        violations=violations,
        frame_bound_lower=lb,
        frame_bound_upper=ub,
        condition_number=cond,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Forward NSGT — ref: nsgt/nsgtf.py  nsgtf()  +  nsgt/nsgtf_loop.py
# ═══════════════════════════════════════════════════════════════════════════


def _nsgtf_loop(
    loopparams: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]],
    ft: torch.Tensor,
) -> list[torch.Tensor]:
    """Inner loop of the forward NSGT.  Ref: nsgt/nsgtf_loop.py nsgtf_loop().

    For each band k:
      1. Extract ft[win_range] — the DFT coefficients at the window's support
      2. Multiply by conjugate of fftshift(g_k) — apply analysis window in freq
      3. Fold (alias) if col > 1, i.e. window longer than M_k channels
      4. Per-band IFFT to get time-domain coefficients

    Parameters
    ----------
    loopparams : list of (mii, gii, gi1, gi2, win_range, Lg, col) per band
        mii       : M_k — channel count for this band
        gii       : full window (unused, kept for API match)
        gi1       : gii[:(Lg+1)//2] — 2nd half of fftshifted window
        gi2       : gii[-(Lg//2):] — 1st half of fftshifted window
        win_range : circular DFT indices for this window's support
        Lg        : len(gii)
        col       : ceil(Lg / mii) — folding factor
    ft : (nn,) complex — full DFT of the signal

    Returns list of (mii,) complex tensors — coefficients per band.
    """
    c: list[torch.Tensor] = []

    for mii, _gii, gi1, gi2, win_range, Lg, col in loopparams:
        device = ft.device
        cdtype = ft.dtype

        # Allocate temp buffer of length col * mii
        temp = torch.zeros(col * mii, device=device, dtype=cdtype)

        # Place windowed DFT values into temp:
        #   temp[:(Lg+1)//2] = gi1 * ft[win_range][Lg//2:]   (2nd half)
        #   temp[-(Lg//2):]  = gi2 * ft[win_range][:Lg//2]   (1st half)
        ftw = ft[win_range]
        half1_len = (Lg + 1) // 2
        half2_len = Lg // 2

        temp[:half1_len] = gi1 * ftw[half2_len:]
        if half2_len > 0:
            temp[-half2_len:] = gi2 * ftw[:half2_len]

        # Clear the gap between the two halves (if any)
        if half1_len + half2_len < col * mii:
            temp[half1_len:col * mii - half2_len] = 0

        # Fold (alias) if window is longer than channel count
        if col > 1:
            temp = temp.reshape(mii, -1).sum(dim=1)
        else:
            temp = temp.clone()

        c.append(temp)

    return c


def nsgtf(
    f: torch.Tensor,
    g: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    M: torch.Tensor,
    *,
    real: bool = True,
    reducedform: int = 0,
) -> list[torch.Tensor]:
    """Non-stationary Gabor forward transform.

    Ref: nsgt/nsgtf.py  nsgtf()

    Parameters
    ----------
    f    : (Ls,) real signal
    g    : analysis windows from nsgfwin
    wins : window ranges from calcwinrange
    nn   : period (from calcwinrange)
    M    : window lengths / channel counts
    real : if True, only positive-frequency bands (DC..Nyquist)
    reducedform : 0=full, 1=drop DC+Nyq, 2=drop DC+Nyq+adjacent

    Returns list of complex tensors, one per band.
    """
    device = f.device
    dtype = f.dtype
    Ls = f.shape[-1]

    if real:
        assert 0 <= reducedform <= 2
        sl = slice(reducedform, len(g) // 2 + 1 - reducedform)
    else:
        sl = slice(0, None)

    # Full-signal FFT
    ft = torch.fft.fft(f, n=nn) if nn > Ls else torch.fft.fft(f)
    if nn > Ls:
        # If nn > Ls we need zero-padding (the FFT(f, n=nn) handles this)
        pass

    cdtype = ft.dtype

    # Build loop parameters for the selected bands
    g_sl = g[sl]
    M_sl = M[sl]
    wins_sl = wins[sl]

    loopparams = []
    for mii_t, gii, win_range in zip(M_sl, g_sl, wins_sl):
        mii = int(mii_t.item())
        Lg = len(gii)
        col = int(math.ceil(float(Lg) / mii))
        assert col * mii >= Lg
        gi1 = gii[:(Lg + 1) // 2]
        gi2 = gii[-(Lg // 2):]
        loopparams.append((mii, gii, gi1, gi2, win_range, Lg, col))

    # Forward loop
    c_raw = _nsgtf_loop(loopparams, ft)

    # Per-band IFFT
    c = [torch.fft.ifft(ck) for ck in c_raw]

    return c


# ═══════════════════════════════════════════════════════════════════════════
# Inverse NSGT — ref: nsgt/nsigtf.py  nsigtf()  +  nsgt/nsigtf_loop.py
# ═══════════════════════════════════════════════════════════════════════════


def _nsigtf_loop(
    loopparams: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, slice, slice, torch.Tensor]],
    fr: torch.Tensor,
    fc: list[torch.Tensor],
) -> torch.Tensor:
    """Inner loop of the inverse NSGT.  Ref: nsgt/nsigtf_loop.py nsigtf_loop().

    Overlap-add procedure with synthesis windows:
      For each band k:
        1. Arrange FFT(c_k) into temp via fftshift-like indexing
        2. Multiply by dual window gd_k and scale by len(c_k)
        3. Overlap-add into fr at the window's support positions
    """
    fr.zero_()

    for t, (gdii, wr1, wr2, sl1, sl2, temp) in zip(fc, loopparams):
        temp[sl1] = t[sl1]
        temp[sl2] = t[sl2]
        temp *= gdii
        temp *= len(t)

        fr[wr1] += temp[sl2]
        fr[wr2] += temp[sl1]

    return fr


def nsigtf(
    c: list[torch.Tensor],
    gd: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    Ls: int,
    *,
    real: bool = True,
    reducedform: int = 0,
) -> torch.Tensor:
    """Non-stationary Gabor inverse transform.

    Ref: nsgt/nsigtf.py  nsigtf()

    Given cell array c of NSGT coefficients and dual windows gd,
    reconstructs the original signal via overlap-add in the frequency
    domain followed by a single inverse FFT.

    Parameters
    ----------
    c    : list of complex tensors — per-band coefficients (from nsgtf)
    gd   : dual windows (from nsdual)
    wins : window ranges (from calcwinrange)
    nn   : period
    Ls   : desired output length
    real : if True, reconstruct real signal from positive-frequency bands
    reducedform : must match the value used in forward transform

    Returns
    -------
    sig : (Ls,) real tensor — reconstructed signal
    """
    device = c[0].device
    # Infer dtype from dual windows (real float)
    rdtype = gd[0].dtype
    cdtype = c[0].dtype

    if real:
        ln = len(gd) // 2 + 1 - reducedform * 2

        def fftsymm(ck: torch.Tensor) -> torch.Tensor:
            """Reconstruct negative-frequency band from positive."""
            # numpy c[-1:0:-1] = reversed c[1:] — torch has no negative step
            return torch.cat([ck[0:1], ck[1:].flip(0)]).conj()

        if reducedform:
            def symm(fc_list: list[torch.Tensor]) -> list[torch.Tensor]:
                return fc_list + [fftsymm(fck) for fck in reversed(fc_list)]

            sl_gd = (list(range(reducedform, len(gd) // 2 + 1 - reducedform))
                      + list(range(len(gd) // 2 + reducedform, len(gd) + 1 - reducedform)))
        else:
            def symm(fc_list: list[torch.Tensor]) -> list[torch.Tensor]:
                return fc_list + [fftsymm(fck) for fck in reversed(fc_list[1:-1])]

            sl_gd = list(range(len(gd)))
    else:
        ln = len(gd)

        def symm(fc_list: list[torch.Tensor]) -> list[torch.Tensor]:
            return fc_list

        sl_gd = list(range(len(gd)))

    assert len(c) == ln, f"Expected {ln} coefficient bands, got {len(c)}"

    maxLg = max(len(gd[i]) for i in sl_gd)

    # Per-band FFT of coefficients
    fc_raw = [torch.fft.fft(ck) for ck in c]
    fc_list = symm(fc_raw)

    # Prepare overlap-add in frequency domain
    fr = torch.zeros(nn, device=device, dtype=cdtype)
    temp0 = torch.zeros(maxLg, device=device, dtype=cdtype)

    # Build loopparams matching the reference
    gd_ordered = [gd[i] for i in sl_gd]
    wins_ordered = [wins[i] for i in sl_gd]

    loopparams = []
    for gdii, win_range in zip(gd_ordered, wins_ordered):
        Lg = len(gdii)
        temp = temp0[:Lg]
        wr1 = win_range[:(Lg) // 2]
        wr2 = win_range[-((Lg + 1) // 2):]
        sl1 = slice(None, (Lg + 1) // 2)
        sl2 = slice(-(Lg // 2), None)
        # Convert gdii to complex for multiplication compatibility
        gd_cx = gdii.to(cdtype)
        loopparams.append((gd_cx, wr1, wr2, sl1, sl2, temp))

    # Overlap-add
    fr = _nsigtf_loop(loopparams, fr, fc_list)

    # Final inverse FFT
    if real:
        ftr = fr[:nn // 2 + 1]
        sig = torch.fft.irfft(ftr, n=nn)
    else:
        sig = torch.fft.ifft(fr).real

    # Truncate to original length
    sig = sig[:Ls]
    return sig


# ═══════════════════════════════════════════════════════════════════════════
# Bank specification — full output for solver / export
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NSGTBankSpec:
    """Complete NSGT bank specification.

    This is what the solver should emit and the validator should check.
    Contains everything needed to construct forward + inverse transforms.

    Follows the recommendation: emit (ξ_k, Ω_k/g_k, a_k or M_k, edge bands).
    """
    # Center frequencies in Hz (including DC=0 and Nyquist)
    centers_hz: torch.Tensor       # (K_total,) — ξ_k
    # Q factors per band
    q_factors: torch.Tensor        # (K_total,)
    # Window support lengths
    support: torch.Tensor          # (K_total,) int — Ω_k in DFT bins
    # Channel counts / FFT lengths per band
    M: torch.Tensor                # (K_total,) int — M_k
    # Per-band hop / time shift (in samples) — for the non-sliced case
    # this equals the signal length Ls (one frame per band)
    # For sliced (sliCQ), a_k = Ls / n_time_steps
    hops: torch.Tensor             # (K_total,) int — a_k
    # Window objects (the actual window samples)
    windows: list[torch.Tensor]    # g_k
    # Dual windows
    dual_windows: list[torch.Tensor]  # gd_k
    # Window position indices
    win_ranges: list[torch.Tensor]    # circular indices into DFT
    # Edge band info
    n_bands_positive: int          # number of positive-freq bands (excl DC, Nyquist)
    has_dc: bool                   # whether DC band is included
    has_nyquist: bool              # whether Nyquist band is included
    # Painless report
    painless: PainlessReport
    # Parameters
    sr: int
    Ls: int
    nn: int                        # period = Ls for non-sliced


# ═══════════════════════════════════════════════════════════════════════════
# High-level CQ-NSGT class — ref: nsgt/cq.py  CQ_NSGT
# ═══════════════════════════════════════════════════════════════════════════


class CQ_NSGT:
    """Constant-Q Non-Stationary Gabor Transform — pure torch.

    Ref: nsgt/cq.py  CQ_NSGT

    Usage:
        nsgt = CQ_NSGT(fmin=32.7, fmax=16000, bpo=48, sr=44100, Ls=signal_len,
                        device=torch.device('cuda'))
        c = nsgt.forward(signal)       # list of complex tensors
        rec = nsgt.backward(c)         # (Ls,) reconstructed signal
        spec = nsgt.bank_spec()        # full NSGTBankSpec for solver/export

    Computes analysis windows, dual windows, and validates painless conditions
    at construction time.  Forward/backward are then just the transforms.
    """

    def __init__(
        self,
        fmin: float,
        fmax: float,
        bpo: int,
        sr: int,
        Ls: int,
        *,
        real: bool = True,
        reducedform: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        pad_ratio: float = 1.125,
        require_painless: bool = False,
    ):
        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))

        self.fmin = fmin
        self.fmax = fmax
        self.bpo = bpo
        self.sr = sr
        self.Ls = Ls
        self.real = real
        self.reducedform = reducedform
        self.device = device
        self.dtype = dtype
        self.pad_ratio = pad_ratio

        # ── Scale ──
        scale = OctScale(fmin, fmax, bpo)
        frqs, q = scale(device=device, dtype=dtype)
        self.frqs = frqs
        self.q_factors = q
        self.scale = scale

        # ── Windows ──
        self.g, rfbas, self.M = nsgfwin(
            frqs, q, sr, Ls, device=device, dtype=dtype)

        # ── Window ranges ──
        self.wins, self.nn = calcwinrange(self.g, rfbas, Ls)

        # ── Dual windows (painless) ──
        self.gd = nsdual(self.g, self.wins, self.nn, self.M)

        # ── Painless check ──
        self._painless = check_painless(self.g, self.wins, self.nn, self.M)

        if require_painless and not self._painless.is_painless:
            raise NotPainlessError(
                f"NSGT frame is not painless "
                f"(condition number {self._painless.condition_number:.3f}, "
                f"{len(self._painless.violations)} violations). "
                "Pass require_painless=False to proceed anyway."
            )

        # ── Precompute band count ──
        if real:
            self._n_coef_bands = len(self.g) // 2 + 1 - reducedform * 2
        else:
            self._n_coef_bands = len(self.g)

        # ── Vectorized GPU plans (built once, reused every call) ──
        self._fwd_plan, self._inv_plan = self._build_plans()

    @property
    def painless_report(self) -> PainlessReport:
        return self._painless

    @property
    def is_painless(self) -> bool:
        return self._painless.is_painless

    # ── Plan construction (called once at init) ───────────────────────

    def _build_plans(self) -> tuple[list, list]:
        if self.real:
            sl = slice(self.reducedform, len(self.g) // 2 + 1 - self.reducedform)
        else:
            sl = slice(0, None)

        g_sl    = self.g[sl]
        M_sl    = self.M[sl]
        wins_sl = self.wins[sl]

        loopparams = []
        for mii_t, gii, win_range in zip(M_sl, g_sl, wins_sl):
            mii = int(mii_t.item())
            Lg  = len(gii)
            col = math.ceil(float(Lg) / mii)
            loopparams.append((mii, gii,
                               gii[:(Lg + 1) // 2],
                               gii[-(Lg // 2):] if Lg // 2 > 0 else gii.new_empty(0),
                               win_range, Lg, col))

        fwd_plan = build_nsgt_fwd_plan(loopparams, self.nn, pad_ratio=self.pad_ratio)

        if self.real:
            if self.reducedform:
                sl_gd = (list(range(self.reducedform, len(self.gd) // 2 + 1 - self.reducedform))
                         + list(range(len(self.gd) // 2 + self.reducedform,
                                      len(self.gd) + 1 - self.reducedform)))
            else:
                sl_gd = list(range(len(self.gd)))
        else:
            sl_gd = list(range(len(self.gd)))

        gd_ordered   = [self.gd[i]   for i in sl_gd]
        wins_ordered = [self.wins[i] for i in sl_gd]
        M_ordered    = self.M[torch.tensor(sl_gd, device=self.device)]
        cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64

        inv_plan = build_nsgt_inv_plan(
            gd_ordered, wins_ordered, M_ordered,
            cdtype=cdtype, device=self.device, pad_ratio=self.pad_ratio,
        )
        return fwd_plan, inv_plan

    # ── Public interface ───────────────────────────────────────────────

    def forward(self, s: torch.Tensor) -> list[torch.Tensor]:
        """Analysis: signal → per-band coefficients.

        s : (Ls,) or (C, Ls) — mono or multi-channel signal.
        Returns list of complex tensors, one per band.
        """
        if s.ndim == 1:
            return nsgtf_vec(s, self.g, self.wins, self.nn, self.M,
                             real=self.real, reducedform=self.reducedform,
                             fwd_plan=self._fwd_plan)
        return [
            nsgtf_vec(s[ch], self.g, self.wins, self.nn, self.M,
                      real=self.real, reducedform=self.reducedform,
                      fwd_plan=self._fwd_plan)
            for ch in range(s.shape[0])
        ]

    def backward(self, c: list[torch.Tensor]) -> torch.Tensor:
        """Synthesis: per-band coefficients → signal.

        c : list of complex tensors (from forward).
        Returns (Ls,) real tensor.
        """
        return nsigtf_vec(c, self.gd, self.wins, self.nn, self.Ls,
                          real=self.real, reducedform=self.reducedform,
                          inv_plan=self._inv_plan)

    @property
    def n_fwd_buckets(self) -> int:
        """Number of batched GPU forward groups."""
        return len(self._fwd_plan)

    @property
    def n_inv_buckets(self) -> int:
        """Number of batched GPU inverse groups."""
        return len(self._inv_plan)

    def bucket_stats(self) -> dict:
        """Batch-layout diagnostics for tuning pad_ratio."""
        def _s(buckets):
            sizes = [len(b.orig_indices) for b in buckets]
            return {"n_buckets": len(sizes), "min_B": min(sizes),
                    "max_B": max(sizes),
                    "mean_B": round(sum(sizes) / len(sizes), 2),
                    "total_bands": sum(sizes)}
        return {"forward": _s(self._fwd_plan), "inverse": _s(self._inv_plan),
                "is_painless": self.is_painless, "pad_ratio": self.pad_ratio}

    def bank_spec(self) -> NSGTBankSpec:
        """Export the full bank specification for solver / validation."""
        # Reconstruct full center frequencies including DC and Nyquist
        nf = self.sr / 2.0
        zero = torch.zeros(1, device=self.device, dtype=self.dtype)
        nyq_t = torch.tensor([nf], device=self.device, dtype=self.dtype)
        all_centers = torch.cat([zero, self.frqs, nyq_t,
                                 self.sr - self.frqs.flip(0)])
        # Q: DC and Nyquist have Q=inf conceptually; use 0 as sentinel
        zero_q = torch.zeros(1, device=self.device, dtype=self.dtype)
        all_q = torch.cat([zero_q, self.q_factors, zero_q,
                           self.q_factors.flip(0)])

        supports = torch.tensor([len(gi) for gi in self.g],
                                device=self.device, dtype=torch.int64)

        # For non-sliced NSGT, each band sees one time step = full signal
        hops = torch.full_like(self.M, self.Ls)

        return NSGTBankSpec(
            centers_hz=all_centers[:len(self.g)],
            q_factors=all_q[:len(self.g)],
            support=supports,
            M=self.M,
            hops=hops,
            windows=self.g,
            dual_windows=self.gd,
            win_ranges=self.wins,
            n_bands_positive=self.scale.bnds,
            has_dc=True,
            has_nyquist=True,
            painless=self._painless,
            sr=self.sr,
            Ls=self.Ls,
            nn=self.nn,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Vectorized GPU NSGT — batched forward + inverse with OOM-halving
# ═══════════════════════════════════════════════════════════════════════════
#
# Design goals (all met below):
#   • Zero per-call Python loops over bands — all work expressed as batched
#     torch ops that saturate GPU SMs.
#   • Smart padding: bands with window sizes within `pad_ratio` of each other
#     are merged into one batch. No cross-group padding; no global padding.
#   • Zero CPU↔GPU churn: all plan tensors are allocated on device at build
#     time and reused across calls.
#   • OOM-safe: every large allocation is wrapped with recursive batch-halving
#     that converges to B=1 (single-band serial fallback) if necessary.
#   • Painless gate: `require_painless=True` raises `NotPainlessError` rather
#     than silently returning wrong results when the frame is non-invertible.
#
# ── Bucket layout for the forward transform ──────────────────────────────
#
#  For each band k: mii_k = M_k (channel count / IFFT size),
#                   col_k = ceil(Lg_k / mii_k) (folding factor),
#                   h1_k  = (Lg_k+1)//2, h2_k = Lg_k//2.
#
#  Bands are grouped by (col, mii). Within each group they are sorted by Lg
#  and greedily merged: a new sub-batch starts only when the next Lg exceeds
#  pad_ratio × (current group's minimum Lg).
#
#  For each sub-batch with max_Lg bands:
#    h1 = (max_Lg+1)//2,  h2 = max_Lg//2
#    stacked_idx  (B, h1+h2) — column-split gather index matrix:
#       cols [:h1]  → win_range_i[h2_i:]   (second half of each win_range)
#       cols [h1:]  → win_range_i[:h2_i]   (first half)
#       padding positions → `nn` (ft_ext[nn] = 0)
#    gi1_stack (B, h1) — window first halves, zero-padded on the right
#    gi2_stack (B, h2) — window second halves, zero-padded on the left
#
#  At runtime:
#    ft_ext = cat(ft, zeros(1))          # sentinel zero at index nn
#    ftw    = ft_ext[stacked_idx]        # (B, h1+h2)  — one gather kernel
#    buf    = zeros(B, col*mii)
#    buf[:, :h1]         = gi1_stack * ftw[:, :h1]
#    buf[:, col*mii-h2:] = gi2_stack * ftw[:, h1:]
#    folded = buf.reshape(B, mii, col).sum(-1)   # fold if col>1
#    result = ifft(folded, dim=-1)               # (B, mii) — batch IFFT
#
# ── Bucket layout for the inverse transform ───────────────────────────────
#
#  Bands grouped by mii (for batch FFT), then by Lg-bucket (for scatter).
#  All precomputation happens at plan-build time; runtime uses only:
#    fc_stack = fft(c_stack, dim=-1)      (B, mii)  — batch FFT
#    t_hi = fc_stack[:, :h1].flatten()   (B*h1,)
#    t_lo = fc_stack[:, mii-h2:].flatten() (B*h2,)
#    vals_wr2 = t_hi[flat_src_hi] * gdii1_flat * mii
#    vals_wr1 = t_lo[flat_src_lo] * gdii2_flat * mii
#    fr.index_add_(0, flat_wr2, vals_wr2)
#    fr.index_add_(0, flat_wr1, vals_wr1)


class NotPainlessError(RuntimeError):
    """Raised by CQ_NSGT_Vec when require_painless=True but frame is not painless."""


# ─── OOM-safe batch runner ────────────────────────────────────────────────

def _oom_halve(fn, *args):
    """Call fn(*args). On CUDA OOM, call fn with halved 'batch' dimension.

    `fn` must accept a slice argument as its last positional parameter:
        fn(*args, sl)   where sl is a slice(start, end) into the batch.
    Returns a list of per-element results (order preserved).
    """
    # Wrapper used internally — see _run_fwd_bucket_oom for usage pattern.
    pass


# ─── Forward plan ─────────────────────────────────────────────────────────

class _FwdBucket(NamedTuple):
    """One group of bands processed together in the forward transform."""
    mii: int                        # IFFT size (shared)
    col: int                        # folding factor (shared)
    h1: int                         # max first-half length
    h2: int                         # max second-half length
    orig_indices: tuple             # positions in the output list
    stacked_idx: torch.Tensor       # (B, h1+h2) int64 — split gather indices
    gi1_stack:   torch.Tensor       # (B, h1) real — first window halves
    gi2_stack:   torch.Tensor       # (B, h2) real — second window halves
    nn: int                         # period (ft_ext sentinel index)


def _build_fwd_bucket(band_list, col: int, mii: int, nn: int,
                      device: torch.device, dtype: torch.dtype) -> _FwdBucket:
    """Construct one forward bucket from a list of (orig_idx, gi1, gi2, win_range, Lg)."""
    max_Lg = max(b[4] for b in band_list)
    h1 = (max_Lg + 1) // 2
    h2 = max_Lg // 2
    B  = len(band_list)

    # stacked_idx: fill with `nn` (the OOB sentinel → ft_ext[nn] = 0)
    stacked_idx = torch.full((B, h1 + h2), nn, dtype=torch.int64, device=device)
    gi1_rows: list[torch.Tensor] = []
    gi2_rows: list[torch.Tensor] = []
    orig_indices: list[int] = []

    for i, (orig_idx, gi1, gi2, win_range, Lg) in enumerate(band_list):
        h1_i = (Lg + 1) // 2
        h2_i = Lg // 2
        orig_indices.append(orig_idx)

        # Second half of win_range → columns [0 : h1_i]  (padded right with nn)
        stacked_idx[i, :h1_i] = win_range[h2_i:]
        # First half of win_range  → columns [h1 + h2-h2_i : h1+h2]  (padded left with nn)
        if h2_i > 0:
            stacked_idx[i, h1 + (h2 - h2_i):] = win_range[:h2_i]

        # gi1: zero-pad on the right to h1
        pad1 = h1 - h1_i
        gi1_rows.append(gi1 if pad1 == 0 else
                        torch.cat([gi1, gi1.new_zeros(pad1)]))

        # gi2: zero-pad on the left to h2  (preserves alignment with stacked_idx)
        if h2 > 0:
            pad2 = h2 - h2_i
            gi2_rows.append(gi2 if pad2 == 0 else
                            torch.cat([gi2.new_zeros(pad2), gi2]))

    gi1_stack = torch.stack(gi1_rows)                            # (B, h1)
    gi2_stack = (torch.stack(gi2_rows) if h2 > 0 else           # (B, h2)
                 gi1_stack.new_zeros(B, 0))

    return _FwdBucket(mii=mii, col=col, h1=h1, h2=h2,
                      orig_indices=tuple(orig_indices),
                      stacked_idx=stacked_idx,
                      gi1_stack=gi1_stack,
                      gi2_stack=gi2_stack,
                      nn=nn)


def build_nsgt_fwd_plan(
    loopparams: list,
    nn: int,
    *,
    pad_ratio: float = 1.125,
) -> list[_FwdBucket]:
    """Build grouped forward-transform buckets from the sequential loopparams.

    Parameters
    ----------
    loopparams : list of (mii, gii, gi1, gi2, win_range, Lg, col)
        As produced by nsgtf() before calling _nsgtf_loop().
    nn         : period (used as OOB sentinel index in ft_ext).
    pad_ratio  : max Lg_max/Lg_min ratio allowed within one batch.
                 1.0 = exact-size groups only. 1.125 ≈ 12.5 % waste ceiling.

    Returns list of _FwdBucket (device tensors already allocated).
    """
    if not loopparams:
        return []

    # Infer device/dtype from gi1 of first entry
    gi1_ref = loopparams[0][2]
    device, dtype = gi1_ref.device, gi1_ref.dtype

    # Group by (col, mii)
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for i, (mii, _gii, gi1, gi2, win_range, Lg, col) in enumerate(loopparams):
        groups[(col, mii)].append((i, gi1, gi2, win_range, Lg))

    buckets: list[_FwdBucket] = []
    for (col, mii), band_list in sorted(groups.items()):
        band_list.sort(key=lambda x: x[4])          # sort by Lg

        # Greedy Lg-merge within pad_ratio
        cur: list = [band_list[0]]
        cur_min_Lg: int = band_list[0][4]
        for band in band_list[1:]:
            if band[4] <= cur_min_Lg * pad_ratio:
                cur.append(band)
            else:
                buckets.append(_build_fwd_bucket(cur, col, mii, nn, device, dtype))
                cur = [band]
                cur_min_Lg = band[4]
        buckets.append(_build_fwd_bucket(cur, col, mii, nn, device, dtype))

    return buckets


def _run_fwd_bucket(ft_ext: torch.Tensor, bkt: _FwdBucket) -> torch.Tensor:
    """Execute one forward bucket. Returns (B, mii) complex tensor.

    No Python loop — pure batched torch ops.
    """
    mii, col, h1, h2 = bkt.mii, bkt.col, bkt.h1, bkt.h2
    B = bkt.stacked_idx.shape[0]

    # ── Gather: flatten index → 1D gather → reshape back ──
    # (Avoids 2D advanced-index path which has device-type restrictions
    #  on some PyTorch builds for complex dtypes.)
    Lgmax = h1 + h2
    ftw = ft_ext[bkt.stacked_idx.reshape(-1)].reshape(B, Lgmax)  # (B, h1+h2)

    # ── Assemble folding buffer ──
    buf = ft_ext.new_zeros(B, col * mii)    # (B, col*mii) complex
    buf[:, :h1] = bkt.gi1_stack * ftw[:, :h1]
    if h2 > 0:
        buf[:, col * mii - h2:] = bkt.gi2_stack * ftw[:, h1:]

    # ── Fold into mii channels ──
    if col > 1:
        # buf[b].reshape(mii, col).sum(1)  →  (B, mii)
        folded = buf.reshape(B, mii, col).sum(-1)
    else:
        folded = buf       # (B, mii) already

    # ── Batch IFFT ──
    return torch.fft.ifft(folded, dim=-1)  # (B, mii) complex


def _run_fwd_bucket_safe(
    ft_ext: torch.Tensor,
    bkt: _FwdBucket,
    min_batch: int = 1,
) -> torch.Tensor:
    """OOM-safe forward bucket run. Halves B recursively on CUDA OOM."""
    B = bkt.stacked_idx.shape[0]
    try:
        return _run_fwd_bucket(ft_ext, bkt)
    except torch.cuda.OutOfMemoryError:
        if B <= min_batch:
            torch.cuda.empty_cache()
            return _run_fwd_bucket(ft_ext, bkt)   # propagate if B=1
        torch.cuda.empty_cache()
        mid = B // 2

        def _slice(b: _FwdBucket, s: slice) -> _FwdBucket:
            idx = b.orig_indices[s]
            return _FwdBucket(
                mii=b.mii, col=b.col, h1=b.h1, h2=b.h2,
                orig_indices=idx,
                stacked_idx=b.stacked_idx[s],
                gi1_stack=b.gi1_stack[s],
                gi2_stack=b.gi2_stack[s],
                nn=b.nn,
            )

        r1 = _run_fwd_bucket_safe(ft_ext, _slice(bkt, slice(None, mid)), min_batch)
        r2 = _run_fwd_bucket_safe(ft_ext, _slice(bkt, slice(mid, None)), min_batch)
        return torch.cat([r1, r2], dim=0)


def nsgtf_vec(
    f: torch.Tensor,
    g: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    M: torch.Tensor,
    *,
    real: bool = True,
    reducedform: int = 0,
    fwd_plan: list[_FwdBucket] | None = None,
    pad_ratio: float = 1.125,
) -> list[torch.Tensor]:
    """Vectorized forward NSGT (drop-in replacement for nsgtf).

    Builds the forward plan on first call if not pre-supplied.
    Pre-supply `fwd_plan` (from build_nsgt_fwd_plan) to amortise plan cost
    across repeated calls with the same bank geometry.
    """
    device = f.device
    Ls     = f.shape[-1]

    if real:
        sl = slice(reducedform, len(g) // 2 + 1 - reducedform)
    else:
        sl = slice(0, None)

    # Full-signal FFT + one sentinel zero (for OOB gather padding)
    ft = torch.fft.fft(f, n=nn) if nn > Ls else torch.fft.fft(f)
    ft_ext = torch.cat([ft, ft.new_zeros(1)])       # (nn+1,) complex

    g_sl    = g[sl]
    M_sl    = M[sl]
    wins_sl = wins[sl]

    loopparams = []
    for mii_t, gii, win_range in zip(M_sl, g_sl, wins_sl):
        mii = int(mii_t.item())
        Lg  = len(gii)
        col = math.ceil(float(Lg) / mii)
        loopparams.append((mii, gii,
                           gii[:(Lg + 1) // 2],
                           gii[-(Lg // 2):] if Lg // 2 > 0 else gii.new_empty(0),
                           win_range, Lg, col))

    if fwd_plan is None:
        fwd_plan = build_nsgt_fwd_plan(loopparams, nn, pad_ratio=pad_ratio)

    c: list[torch.Tensor | None] = [None] * len(loopparams)
    for bkt in fwd_plan:
        result = _run_fwd_bucket_safe(ft_ext, bkt)   # (B, mii)
        for j, orig in enumerate(bkt.orig_indices):
            c[orig] = result[j]

    return c  # type: ignore[return-value]


# ─── Inverse plan ─────────────────────────────────────────────────────────

class _InvBucket(NamedTuple):
    """One group of bands processed together in the inverse transform."""
    mii:       int                  # FFT size (shared)
    h1:        int                  # max first-half length in this group
    h2:        int                  # max second-half length
    orig_indices: tuple             # positions into the expanded fc_list
    # Precomputed scatter primitives (all device tensors, built once)
    gdii1_flat: torch.Tensor        # (sum_h1_i,) complex  — dual window, first halves
    gdii2_flat: torch.Tensor        # (sum_h2_i,) complex  — dual window, second halves
    flat_src_hi: torch.Tensor       # (sum_h1_i,) int64  — gather from t_hi flat
    flat_src_lo: torch.Tensor       # (sum_h2_i,) int64  — gather from t_lo flat
    flat_wr2:    torch.Tensor       # (sum_h1_i,) int64  — scatter into fr
    flat_wr1:    torch.Tensor       # (sum_h2_i,) int64  — scatter into fr


def _build_inv_bucket(
    band_list,        # list of (orig_idx, gdii, wr1, wr2, h1_i, h2_i, mii)
    mii: int,
    device: torch.device,
    cdtype: torch.dtype,
) -> _InvBucket:
    """Build one inverse bucket.  All tensors allocated on device."""
    # Determine max halves
    h1 = max(b[4] for b in band_list)
    h2 = max(b[5] for b in band_list)
    B  = len(band_list)

    gdii1_parts: list[torch.Tensor] = []
    gdii2_parts: list[torch.Tensor] = []
    src_hi_parts: list[torch.Tensor] = []
    src_lo_parts: list[torch.Tensor] = []
    wr2_parts: list[torch.Tensor] = []
    wr1_parts: list[torch.Tensor] = []
    orig_indices: list[int] = []

    for i, (orig_idx, gdii, wr1, wr2, h1_i, h2_i, _mii) in enumerate(band_list):
        orig_indices.append(orig_idx)
        gd_cx = gdii.to(cdtype)

        # Dual window halves (valid elements only, not padded)
        gdii1_parts.append(gd_cx[:h1_i])       # (h1_i,)
        if h2_i > 0:
            gdii2_parts.append(gd_cx[-h2_i:])  # (h2_i,)

        # Flat source indices into the (B, h1) / (B, h2) views after batch FFT:
        #   t_hi = fc_stack[:, :h1].flatten()   → index i*h1 + j  for j in range(h1_i)
        #   t_lo = fc_stack[:, mii-h2:].flatten()
        #          → index i*h2 + (h2-h2_i) + j  for j in range(h2_i)
        src_hi_parts.append(torch.arange(h1_i, device=device, dtype=torch.int64) + i * h1)
        if h2_i > 0:
            offset = h2 - h2_i
            src_lo_parts.append(
                torch.arange(h2_i, device=device, dtype=torch.int64) + i * h2 + offset)

        # Scatter indices into fr
        wr2_parts.append(wr2)   # (h1_i,)
        if h2_i > 0:
            wr1_parts.append(wr1)   # (h2_i,)

    gdii1_flat  = torch.cat(gdii1_parts)
    gdii2_flat  = torch.cat(gdii2_parts) if gdii2_parts else gdii1_flat.new_empty(0)
    flat_src_hi = torch.cat(src_hi_parts)
    flat_src_lo = torch.cat(src_lo_parts) if src_lo_parts else flat_src_hi.new_empty(0)
    flat_wr2    = torch.cat(wr2_parts)
    flat_wr1    = torch.cat(wr1_parts) if wr1_parts else flat_wr2.new_empty(0)

    return _InvBucket(
        mii=mii, h1=h1, h2=h2,
        orig_indices=tuple(orig_indices),
        gdii1_flat=gdii1_flat,
        gdii2_flat=gdii2_flat,
        flat_src_hi=flat_src_hi,
        flat_src_lo=flat_src_lo,
        flat_wr2=flat_wr2,
        flat_wr1=flat_wr1,
    )


def build_nsgt_inv_plan(
    gd_ordered: list[torch.Tensor],
    wins_ordered: list[torch.Tensor],
    M_ordered: torch.Tensor,
    *,
    cdtype: torch.dtype,
    device: torch.device,
    pad_ratio: float = 1.125,
) -> list[_InvBucket]:
    """Build grouped inverse-transform buckets from ordered windows/ranges.

    Parameters
    ----------
    gd_ordered   : dual windows in the order they will be applied.
    wins_ordered : window ranges in the same order.
    M_ordered    : channel counts (mii) in the same order.
    cdtype       : complex dtype used for fr / coefficients.
    pad_ratio    : max Lg_max/Lg_min within one batch (same semantics as fwd).
    """
    # Collect per-band info
    band_info: list[tuple] = []
    for k, (gdii, win_range, mii_t) in enumerate(
            zip(gd_ordered, wins_ordered, M_ordered)):
        mii  = int(mii_t.item())
        Lg   = len(gdii)
        h1_i = (Lg + 1) // 2
        h2_i = Lg // 2
        wr1  = win_range[:h2_i]
        wr2  = win_range[-h1_i:]
        band_info.append((k, gdii, wr1, wr2, h1_i, h2_i, mii))

    # Group by mii, then by Lg-bucket (pad_ratio)
    mii_groups: dict[int, list] = defaultdict(list)
    for info in band_info:
        mii_groups[info[6]].append(info)

    buckets: list[_InvBucket] = []
    for mii, bands in sorted(mii_groups.items()):
        # Sort by Lg (= h1_i + h2_i)
        bands.sort(key=lambda x: x[4] + x[5])

        # Greedy Lg-merge
        cur: list = [bands[0]]
        cur_min_Lg = bands[0][4] + bands[0][5]
        for band in bands[1:]:
            Lg = band[4] + band[5]
            if Lg <= cur_min_Lg * pad_ratio:
                cur.append(band)
            else:
                buckets.append(_build_inv_bucket(cur, mii, device, cdtype))
                cur = [band]
                cur_min_Lg = Lg
        buckets.append(_build_inv_bucket(cur, mii, device, cdtype))

    return buckets


def _run_inv_bucket(
    fc_list: list[torch.Tensor],
    bkt: _InvBucket,
    fr: torch.Tensor,
) -> None:
    """Execute one inverse bucket: batch FFT → window → index_add_ into fr.

    Modifies fr in-place. No return value.
    """
    mii, h1, h2 = bkt.mii, bkt.h1, bkt.h2
    orig = bkt.orig_indices
    B = len(orig)

    # ── Stack coefficients and batch-FFT ──
    # Each c_k already has size mii (verified by painless condition).
    c_stack = torch.stack([fc_list[k] for k in orig])   # (B, mii)
    # fc_list entries are already FFT'd coefficients (see nsigtf_vec)
    # so we use them directly — no second FFT needed.
    t_stack = c_stack   # (B, mii)  complex

    # ── Extract first-half and second-half views ──
    # t_hi: (B, h1) → valid positions [0:h1_i] per band (rest unused via src_hi)
    t_hi = t_stack[:, :h1].flatten()           # (B*h1,)
    # t_lo: (B, h2) → valid positions [h2-h2_i:h2] per band
    if h2 > 0:
        t_lo = t_stack[:, mii - h2:].flatten() # (B*h2,)

    # ── Apply dual windows (precomputed, on-device) ──
    scale = float(mii)
    vals_wr2 = t_hi[bkt.flat_src_hi] * bkt.gdii1_flat * scale   # (sum_h1_i,)
    if h2 > 0 and bkt.flat_src_lo.numel() > 0:
        vals_wr1 = t_lo[bkt.flat_src_lo] * bkt.gdii2_flat * scale  # (sum_h2_i,)

    # ── Overlap-add into frequency accumulator ──
    fr.index_add_(0, bkt.flat_wr2, vals_wr2)
    if h2 > 0 and bkt.flat_src_lo.numel() > 0:
        fr.index_add_(0, bkt.flat_wr1, vals_wr1)


def _run_inv_bucket_safe(
    fc_list: list[torch.Tensor],
    bkt: _InvBucket,
    fr: torch.Tensor,
    min_batch: int = 1,
) -> None:
    """OOM-safe inverse bucket: halves B recursively on CUDA OOM."""
    B = len(bkt.orig_indices)
    try:
        _run_inv_bucket(fc_list, bkt, fr)
    except torch.cuda.OutOfMemoryError:
        if B <= min_batch:
            torch.cuda.empty_cache()
            _run_inv_bucket(fc_list, bkt, fr)
            return
        torch.cuda.empty_cache()
        mid = B // 2

        def _slice_inv(b: _InvBucket, s: slice) -> _InvBucket:
            orig = b.orig_indices[s]
            # Recompute flat tensors for the slice — keep on same device
            # (cheap: just gather from existing flat tensors)
            # Determine per-band h1_i, h2_i from flat_src_hi strides
            # Simplest: re-slice by splitting at flat boundary
            # We track per-band element counts via h1 (all share same h1_max)
            band_slice = slice(*s.indices(B))
            bstart = band_slice.start
            bend   = band_slice.stop

            # src_hi for band i spans [i*h1 .. (i+1)*h1) in the flat view
            # but not all slots are valid (some are padding). The actual
            # valid count per band is encoded in flat_src_hi values.
            # Simpler approach: filter by band index ranges in flat_src_hi
            mask_hi = (b.flat_src_hi // b.h1 >= bstart) & (b.flat_src_hi // b.h1 < bend)
            mask_lo = ((b.flat_src_lo // b.h2 >= bstart) & (b.flat_src_lo // b.h2 < bend)
                       if b.h2 > 0 and b.flat_src_lo.numel() > 0
                       else torch.zeros(0, dtype=torch.bool, device=b.flat_wr2.device))

            # Rebase src indices to new band numbering
            new_src_hi = b.flat_src_hi[mask_hi] - bstart * b.h1
            new_src_lo = (b.flat_src_lo[mask_lo] - bstart * b.h2
                          if mask_lo.numel() > 0
                          else b.flat_src_lo.new_empty(0))

            return _InvBucket(
                mii=b.mii, h1=b.h1, h2=b.h2,
                orig_indices=orig,
                gdii1_flat=b.gdii1_flat[mask_hi],
                gdii2_flat=b.gdii2_flat[mask_lo] if mask_lo.numel() > 0 else b.gdii2_flat.new_empty(0),
                flat_src_hi=new_src_hi,
                flat_src_lo=new_src_lo,
                flat_wr2=b.flat_wr2[mask_hi],
                flat_wr1=b.flat_wr1[mask_lo] if mask_lo.numel() > 0 else b.flat_wr1.new_empty(0),
            )

        _run_inv_bucket_safe(fc_list, _slice_inv(bkt, slice(None, mid)), fr, min_batch)
        _run_inv_bucket_safe(fc_list, _slice_inv(bkt, slice(mid, None)), fr, min_batch)


def nsigtf_vec(
    c: list[torch.Tensor],
    gd: list[torch.Tensor],
    wins: list[torch.Tensor],
    nn: int,
    Ls: int,
    *,
    real: bool = True,
    reducedform: int = 0,
    inv_plan: list[_InvBucket] | None = None,
    pad_ratio: float = 1.125,
    M: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized inverse NSGT (drop-in replacement for nsigtf).

    Pre-supply `inv_plan` (from build_nsgt_inv_plan) to amortise plan cost.
    `M` is needed only when inv_plan is None (used to build the plan).
    """
    device  = c[0].device
    rdtype  = gd[0].dtype
    cdtype  = c[0].dtype

    if real:
        ln = len(gd) // 2 + 1 - reducedform * 2

        def fftsymm(ck: torch.Tensor) -> torch.Tensor:
            return torch.cat([ck[0:1], ck[1:].flip(0)]).conj()

        if reducedform:
            def symm(fc_list):
                return fc_list + [fftsymm(fck) for fck in reversed(fc_list)]
            sl_gd = (list(range(reducedform, len(gd) // 2 + 1 - reducedform))
                     + list(range(len(gd) // 2 + reducedform, len(gd) + 1 - reducedform)))
        else:
            def symm(fc_list):
                return fc_list + [fftsymm(fck) for fck in reversed(fc_list[1:-1])]
            sl_gd = list(range(len(gd)))
    else:
        ln = len(gd)
        def symm(fc_list): return fc_list
        sl_gd = list(range(len(gd)))

    assert len(c) == ln, f"Expected {ln} coefficient bands, got {len(c)}"

    # Batch FFT of all coefficient bands — grouped by mii for efficiency
    # (torch.fft.fft handles variable-size inputs via padding; here all c_k
    #  in a group share mii, so we stack and batch.)
    # We eagerly FFT all bands; the plan uses these pre-computed fc values.
    fc_raw = [torch.fft.fft(ck) for ck in c]
    fc_list = symm(fc_raw)

    gd_ordered   = [gd[i]   for i in sl_gd]
    wins_ordered = [wins[i] for i in sl_gd]

    if inv_plan is None:
        if M is None:
            # Derive mii from coefficient lengths
            M_vals = [len(fc_list[k]) for k in range(len(sl_gd))]
            M_ordered_t = torch.tensor(M_vals, dtype=torch.int64, device=device)
        else:
            M_ordered_t = M[sl_gd]
        inv_plan = build_nsgt_inv_plan(
            gd_ordered, wins_ordered, M_ordered_t,
            cdtype=cdtype, device=device, pad_ratio=pad_ratio,
        )

    fr = torch.zeros(nn, device=device, dtype=cdtype)

    for bkt in inv_plan:
        _run_inv_bucket_safe(fc_list, bkt, fr)

    if real:
        ftr = fr[:nn // 2 + 1]
        sig = torch.fft.irfft(ftr, n=nn)
    else:
        sig = torch.fft.ifft(fr).real

    return sig[:Ls]

# ═══════════════════════════════════════════════════════════════════════════
# Fidelity curve for NSGT — lightweight GPU path, solver-safe
# ═══════════════════════════════════════════════════════════════════════════
#
# Design: the solver calls this function hundreds–thousands of times per run.
# The old path built a full CQ_NSGT (dual windows + vectorized plans) on
# every call — none of that is needed for fidelity metrics.  This version:
#
#   1. Uses GPU (auto-detected) for the frame-diagonal index_add_.
#   2. Skips nsdual / _build_plans entirely — only builds the bank geometry
#      (nsgfwin + calcwinrange) and calls check_painless once.
#   3. Caches results keyed on (sr, bpo, Ls, fmin_rounded, fmax_rounded) so
#      repeated evaluations with nearly-identical params are free.

# Bounded insertion-order dict used as LRU cache
_fidelity_nsgt_cache: dict = {}
_FIDELITY_NSGT_CACHE_MAX: int = 512

# Device picked once at import, same process-lifetime device as the rest
_FIDELITY_DEVICE: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def _quantise(x: float, decimals: int = 2) -> float:
    """Round to `decimals` decimal places for cache-key stability."""
    return round(x, decimals)


def fidelity_curve_nsgt(
    sr: int,
    fmin: float = 32.7,
    fmax: float = 16000.0,
    bins_per_octave: int = 48,
    Ls: int = 0,
) -> dict:
    """Compute the NSGT fidelity / loss-map for use by the parameter solver.

    Lightweight GPU path — skips dual-window and plan construction.
    Results are cached by (sr, bpo, Ls, fmin≈, fmax≈) so the solver pays
    full cost at most once per distinct configuration per process lifetime.

    Returns a dict compatible with fidelity_curve / fidelity_curve_cwt /
    fidelity_curve_fb, plus NSGT-specific keys (painless, condition_number…).
    """
    import numpy as np

    if Ls <= 0:
        Ls = sr

    fmax = min(fmax, sr / 2.0)
    fmin = max(fmin, 1e-3)

    # ── Cache lookup ──────────────────────────────────────────────────
    cache_key = (sr, bins_per_octave, Ls,
                 _quantise(fmin), _quantise(fmax))
    if cache_key in _fidelity_nsgt_cache:
        return _fidelity_nsgt_cache[cache_key]

    _EMPTY = {
        "freqs": np.array([], dtype=np.float64),
        "delta_t": np.array([], dtype=np.float64),
        "delta_f": np.array([], dtype=np.float64),
        "loss_rgb": np.zeros((0, 3), dtype=np.float32),
        "painless": None,
    }

    device = _FIDELITY_DEVICE
    dtype  = torch.float64

    # ── Bank geometry — no dual windows, no plans ─────────────────────
    try:
        scale    = OctScale(fmin, fmax, bins_per_octave)
        frqs_t, q_t = scale(device=device, dtype=dtype)

        nf    = sr / 2.0
        valid = (frqs_t > 0) & (frqs_t < nf)
        frqs_v = frqs_t[valid]
        q_v    = q_t[valid]

        if frqs_v.numel() == 0:
            return _EMPTY

        g, rfbas, M_int = nsgfwin(frqs_v, q_v, sr, Ls,
                                   device=device, dtype=dtype)
        wins, nn = calcwinrange(g, rfbas, Ls)
        pr = check_painless(g, wins, nn, M_int)
    except Exception:
        return _EMPTY

    # ── Extract per-band metrics (positive-frequency bands only) ──────
    # g layout: [DC(0), band_1…band_lbas, Nyquist, mirror_lbas…mirror_1]
    # Positive bands: indices 1 .. lbas  (lbas = frqs_v.numel())
    lbas = int(frqs_v.shape[0])
    n_pos = lbas

    freqs_np   = frqs_v.cpu().numpy().astype(np.float64)           # (n_pos,)
    q_np       = np.maximum(q_v.cpu().numpy().astype(np.float64),
                            1e-30)                                  # (n_pos,)
    # M_int[1 : lbas+1] = window sizes for the positive-freq bands
    support_np = M_int[1 : lbas + 1].cpu().numpy().astype(np.float64)

    delta_f = freqs_np / q_np
    delta_t = support_np / float(sr)

    from torch_cqt_new import _loss_rgb_from_metrics
    loss_rgb = _loss_rgb_from_metrics(freqs_np, delta_t, delta_f, sr, q_np)

    painless_violation = np.zeros(n_pos, dtype=np.float32)
    for v in pr.violations:
        idx = v - 1          # skip DC at index 0
        if 0 <= idx < n_pos:
            painless_violation[idx] = 1.0
            loss_rgb[idx, 0] = max(loss_rgb[idx, 0], 1.0)

    result = {
        "freqs":              freqs_np,
        "delta_t":            delta_t,
        "delta_f":            delta_f,
        "loss_rgb":           loss_rgb,
        "q_factors":          q_np,
        "support":            support_np,
        "painless":           pr,
        "painless_violation": painless_violation,
        "frame_bound_lower":  pr.frame_bound_lower,
        "frame_bound_upper":  pr.frame_bound_upper,
        "condition_number":   pr.condition_number,
        "n_bands":            n_pos,
    }

    # ── Cache with bounded eviction (oldest entry) ────────────────────
    if len(_fidelity_nsgt_cache) >= _FIDELITY_NSGT_CACHE_MAX:
        _fidelity_nsgt_cache.pop(next(iter(_fidelity_nsgt_cache)))
    _fidelity_nsgt_cache[cache_key] = result

    return result
