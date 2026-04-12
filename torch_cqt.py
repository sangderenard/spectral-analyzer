"""Pure-torch CQT / iCQT — 1:1 transliteration of librosa's algorithm.

Every function maps to a specific librosa function. Comments reference
the librosa source directly.

Forward:  cqt(y, sr, ...) → C  complex (..., n_bins, n_frames)
Inverse:  icqt(C, sr, ...) → y  float   (..., n_samples)
"""
from __future__ import annotations
import math
import numpy as np
import torch
import librosa


# ── Pad ──────────────────────────────────────────────────────────────────

def _pad(
    x: torch.Tensor,
    pad_left: int,
    pad_right: int,
    mode: str = "reflect",
    value: float = 0.0,
) -> torch.Tensor:
    """Pad *x* along its last dimension, supporting arbitrary leading dims.

    Works for 1-D through N-D tensors.  For ``mode="reflect"`` handles the
    edge case where ``x.shape[-1] <= pad`` by applying reflect in a loop.

    Parameters
    ----------
    x : Tensor  (..., L)
    pad_left, pad_right : int
    mode : "reflect" | "replicate" | "constant" | "circular"
    value : fill value when mode == "constant"
    """
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
# librosa.filters.wavelet  →  _wavelet
# librosa.__vqt_filter_fft  →  _vqt_filter_fft
# ══════════════════════════════════════════════════════════════════════════

def _wavelet(
    freqs: np.ndarray,
    sr: float,
    filter_scale: float = 1.0,
    norm: float = 1.0,
    window: str = "hann",
    alpha: np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Torch clone of ``librosa.filters.wavelet(..., pad_fft=True, norm=1)``.

    Returns
    -------
    filters : (n_filters, n_fft)  complex128
    lengths : (n_filters,)  numpy float64 — fractional filter lengths
    """
    if device is None:
        device = torch.device("cpu")

    # librosa.filters.wavelet_lengths
    lengths, _ = librosa.filters.wavelet_lengths(
        freqs=freqs, sr=sr, filter_scale=filter_scale,
        window=window, alpha=alpha)

    n_filters = len(freqs)
    filters_list: list[torch.Tensor] = []

    for ilen, freq in zip(lengths, freqs):
        # ── librosa.util.phasor(arange(-ilen//2, ilen//2) * 2π f / sr) ──
        t = torch.arange(-ilen // 2, ilen // 2,
                         dtype=torch.float64, device=device)
        sig = torch.exp(1j * 2.0 * math.pi * freq / sr * t)

        # ── __float_window(window)(len(sig)) ──
        # floor(ilen) samples of real window, zero-pad to ceil(ilen) if fractional
        n_min = int(math.floor(ilen))
        n_max = int(math.ceil(ilen))
        win = torch.hann_window(n_min, periodic=False,
                                dtype=torch.float64, device=device)
        if n_max > n_min:
            win = _pad(win, 0, n_max - n_min, mode="constant")
        # Truncate / align to sig length
        win = win[:len(sig)]

        sig = sig * win

        # ── librosa.util.normalize(sig, norm=1)  → L1 ──
        sig = sig / torch.abs(sig).sum()

        filters_list.append(sig)

    # ── pad_fft=True: pad to next power of 2 ──
    max_len = int(max(lengths))
    n_fft = int(2 ** math.ceil(math.log2(max_len)))

    # ── librosa.util.pad_center: center each filter in n_fft ──
    basis = torch.zeros((n_filters, n_fft),
                        dtype=torch.complex128, device=device)
    for i, filt in enumerate(filters_list):
        flen = filt.shape[0]
        lpad = (n_fft - flen) // 2
        basis[i, lpad:lpad + flen] = filt

    return basis, lengths


def _vqt_filter_fft(
    sr: float,
    freqs: np.ndarray,
    filter_scale: float,
    norm: float,
    hop_length: int,
    window: str = "hann",
    alpha: np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int, np.ndarray]:
    """Torch clone of ``librosa.__vqt_filter_fft``.

    Returns
    -------
    fft_basis : (n_filters, n_fft//2+1)  complex128
    n_fft     : int
    lengths   : (n_filters,)  numpy float64
    """
    if device is None:
        device = torch.device("cpu")

    # Step 1: build time-domain basis via wavelet()
    basis, lengths = _wavelet(
        freqs, sr, filter_scale=filter_scale, norm=norm,
        window=window, alpha=alpha, device=device)

    n_fft = basis.shape[1]

    # Step 2: ensure n_fft >= 2^(1 + ceil(log2(hop_length)))
    # (librosa: ``if n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length)))``)
    min_fft = int(2.0 ** (1 + math.ceil(math.log2(hop_length))))
    if n_fft < min_fft:
        n_fft = min_fft

    # Step 3: re-normalize basis w.r.t. FFT window length
    # librosa: ``basis *= lengths[:, np.newaxis] / float(n_fft)``
    lengths_t = torch.from_numpy(lengths).to(device=device, dtype=torch.float64)
    basis = basis * (lengths_t / float(n_fft)).unsqueeze(1)

    # Step 4: FFT and keep non-negative frequencies
    fft_basis = torch.fft.fft(basis, n=n_fft, dim=1)[:, :n_fft // 2 + 1]

    return fft_basis, n_fft, lengths


# ══════════════════════════════════════════════════════════════════════════
# librosa.core.spectrum.stft / istft  →  _stft / _istft
# ══════════════════════════════════════════════════════════════════════════

def _stft(y: torch.Tensor, n_fft: int, hop_length: int,
          pad_mode: str = "constant",
          dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Centered STFT with ones window — matches ``librosa.stft(window='ones')``.

    librosa.__cqt_response calls ``stft(y, n_fft=n_fft, hop_length=hop_length,
    window='ones', pad_mode=mode)``.

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
    return D.to(dtype)


def _istft(D: torch.Tensor, n_fft: int, hop_length: int,
           length: int | None = None,
           dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Inverse STFT with ones window — matches ``librosa.istft(window='ones')``.

    Input:  (..., n_fft//2+1, n_frames)
    Output: (..., L)
    """
    window = torch.ones(n_fft, dtype=dtype, device=D.device)

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
    """Torch clone of ``librosa.__cqt_response``.

    Computes ``fft_basis @ stft(y)``.

    Input:  y (..., L),  fft_basis (n_filters, n_fft//2+1)
    Output: (..., n_filters, n_frames)
    """
    D = _stft(y, n_fft, hop_length, pad_mode=pad_mode)
    # D is (..., n_fft//2+1, n_frames)
    # fft_basis is (n_filters, n_fft//2+1)
    # We need (..., n_filters, n_frames) = fft_basis @ D
    # For batched: use einsum or matmul on last two dims
    return torch.matmul(fft_basis, D)


# ══════════════════════════════════════════════════════════════════════════
# librosa.resample  →  _resample
# ══════════════════════════════════════════════════════════════════════════

def _resample(y: torch.Tensor, orig_sr: int, target_sr: int,
              scale: bool = False) -> torch.Tensor:
    """Resample along last dimension.

    Input/Output: (..., L) → (..., L')

    Parameters
    ----------
    scale : bool
        If True, scale output by target_sr/orig_sr (matches
        ``librosa.resample(scale=True)``).
    """
    if orig_sr == target_sr:
        return y
    ratio = target_sr / orig_sr
    if ratio == int(ratio):
        ratio_int = int(ratio)
        L = y.shape[-1]
        lead = y.shape[:-1]
        up = torch.zeros(*lead, L * ratio_int, dtype=y.dtype, device=y.device)
        up[..., ::ratio_int] = y
        filt_half = 32 * ratio_int
        filt_len = 2 * filt_half + 1
        t = torch.arange(-filt_half, filt_half + 1,
                         dtype=torch.float64, device=y.device)
        h = torch.sinc(t / ratio_int) * torch.hann_window(
            filt_len, periodic=False, dtype=torch.float64,
            device=y.device)
        h = h / h.sum() * ratio_int
        flat = up.reshape(-1, 1, up.shape[-1])
        filtered = torch.nn.functional.conv1d(
            flat, h.view(1, 1, -1), padding=filt_half)
        out = filtered.reshape(*lead, -1)
    elif 1 / ratio == int(1 / ratio):
        # Downsample by integer factor
        down = int(1 / ratio)
        filt_half = 32 * down
        filt_len = 2 * filt_half + 1
        t = torch.arange(-filt_half, filt_half + 1,
                         dtype=torch.float64, device=y.device)
        h = torch.sinc(t / down) * torch.hann_window(
            filt_len, periodic=False, dtype=torch.float64,
            device=y.device)
        h = h / h.sum()
        lead = y.shape[:-1]
        flat = y.reshape(-1, 1, y.shape[-1])
        filtered = torch.nn.functional.conv1d(flat, h.view(1, 1, -1),
                                               padding=filt_half)
        out = filtered[..., ::down].reshape(*lead, -1)
    else:
        # FFT resample fallback
        n_out = int(round(y.shape[-1] * ratio))
        Y = torch.fft.rfft(y, dim=-1)
        new_nf = n_out // 2 + 1
        if new_nf <= Y.shape[-1]:
            Y2 = Y[..., :new_nf] * (n_out / y.shape[-1])
        else:
            Y2 = torch.zeros(*Y.shape[:-1], new_nf,
                              dtype=Y.dtype, device=Y.device)
            Y2[..., :Y.shape[-1]] = Y * (n_out / y.shape[-1])
        out = torch.fft.irfft(Y2, n=n_out, dim=-1)

    if scale:
        out = out * (target_sr / orig_sr)
    return out


# ══════════════════════════════════════════════════════════════════════════
# librosa.__trim_stack  →  _trim_stack
# ══════════════════════════════════════════════════════════════════════════

def _trim_stack(
    cqt_resp: list[torch.Tensor],
    n_bins: int,
) -> torch.Tensor:
    """Torch clone of ``librosa.__trim_stack``.

    cqt_resp[0] = top octave, cqt_resp[-1] = bottom octave.
    Output is ordered low-to-high: (n_bins, n_frames).
    """
    max_col = min(c.shape[-1] for c in cqt_resp)
    # Grab leading dimensions from first response
    lead = cqt_resp[0].shape[:-2]
    shape = list(lead) + [n_bins, max_col]
    dtype = cqt_resp[0].dtype
    device = cqt_resp[0].device
    cqt_out = torch.empty(shape, dtype=dtype, device=device)

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
# librosa.vqt (with gamma=0 → cqt)  →  cqt
# ══════════════════════════════════════════════════════════════════════════

def cqt(
    y: np.ndarray | torch.Tensor,
    sr: int,
    hop_length: int = 512,
    fmin: float | None = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    norm: float = 1.0,
    window: str = "hann",
    scale: bool = True,
    pad_mode: str = "constant",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Torch clone of ``librosa.vqt`` (with gamma=0 → CQT).

    Returns
    -------
    C     : complex torch tensor (..., n_bins, n_frames) on *device*
    freqs : numpy array (n_bins,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fmin is None:
        fmin = librosa.note_to_hz("C1")

    if isinstance(y, np.ndarray):
        y_t = torch.from_numpy(y).to(device=device, dtype=torch.float64)
    else:
        y_t = y.to(device=device, dtype=torch.float64)

    n_octaves = int(math.ceil(n_bins / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    freqs = librosa.cqt_frequencies(
        n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    if n_bins == 1:
        alpha = np.atleast_1d(
            (2 ** (2 / bins_per_octave) - 1) / (2 ** (2 / bins_per_octave) + 1))
    else:
        alpha = librosa.filters._relative_bandwidth(freqs=freqs)

    lengths, _ = librosa.filters.wavelet_lengths(
        freqs=freqs, sr=sr, window=window,
        filter_scale=filter_scale, alpha=alpha)

    # ── Iterate down the octaves (librosa.vqt main loop) ──
    my_y, my_sr, my_hop = y_t, float(sr), hop_length
    vqt_resp: list[torch.Tensor] = []

    for i in range(n_octaves):
        # Slice out the current octave of filters
        # librosa: i==0 → sl=slice(-n_filters, None)
        #          else → sl=slice(-n_filters*(i+1), -n_filters*i)
        if i == 0:
            sl = slice(-n_filters, None)
        else:
            sl = slice(-n_filters * (i + 1), -n_filters * i)

        freqs_oct = freqs[sl]
        alpha_oct = alpha[sl]

        # librosa: fft_basis, n_fft, _ = __vqt_filter_fft(...)
        fft_basis, n_fft, _ = _vqt_filter_fft(
            my_sr, freqs_oct, filter_scale, norm, my_hop,
            window=window, alpha=alpha_oct, device=device)

        # librosa: ``fft_basis[:] *= np.sqrt(sr / my_sr)``
        fft_basis = fft_basis * math.sqrt(sr / my_sr)

        # Pad signal if it's too short for STFT
        if my_y.shape[-1] < n_fft:
            my_y_padded = _pad(my_y, 0, n_fft - my_y.shape[-1],
                               mode="constant")
        else:
            my_y_padded = my_y

        # librosa: ``__cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode)``
        vqt_resp.append(
            _cqt_response(my_y_padded, n_fft, my_hop, fft_basis,
                          pad_mode=pad_mode))

        # librosa: downsample for next octave
        # ``my_y = audio.resample(my_y, orig_sr=2, target_sr=1,
        #                         res_type=res_type, scale=True)``
        if my_hop % 2 == 0:
            my_hop //= 2
            my_sr /= 2.0
            my_y = _resample(my_y, orig_sr=2, target_sr=1, scale=True)

    # librosa: ``V = __trim_stack(vqt_resp, n_bins, dtype)``
    V = _trim_stack(vqt_resp, n_bins)

    # librosa: ``if scale: V /= np.sqrt(lengths)``
    if scale:
        lengths_t = torch.from_numpy(lengths).to(device=device, dtype=torch.float64)
        # expand_to: shape for broadcasting over (..., n_bins, n_frames)
        V = V / lengths_t.unsqueeze(-1).sqrt()

    return V, freqs


# ══════════════════════════════════════════════════════════════════════════
# librosa.icqt  →  icqt
# ══════════════════════════════════════════════════════════════════════════

def icqt(
    C: torch.Tensor | np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float | None = None,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    norm: float = 1.0,
    window: str = "hann",
    scale: bool = True,
    length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Torch clone of ``librosa.icqt``.

    Returns float64 tensor (..., n_samples) on *device*.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fmin is None:
        fmin = librosa.note_to_hz("C1")

    if isinstance(C, np.ndarray):
        C_t = torch.from_numpy(C).to(device=device)
    else:
        C_t = C.to(device)
    if not C_t.is_complex():
        C_t = C_t.to(torch.complex128)

    n_bins = C_t.shape[-2]
    n_octaves = int(math.ceil(n_bins / bins_per_octave))

    freqs = librosa.cqt_frequencies(
        n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    if n_bins == 1:
        alpha = np.atleast_1d(
            (2 ** (2 / bins_per_octave) - 1) / (2 ** (2 / bins_per_octave) + 1))
    else:
        alpha = librosa.filters._relative_bandwidth(freqs=freqs)

    lengths, _ = librosa.filters.wavelet_lengths(
        freqs=freqs, sr=sr, window=window,
        filter_scale=filter_scale, alpha=alpha)

    # librosa: trim CQT frames if length is given
    if length is not None:
        n_frames = int(math.ceil((length + max(lengths)) / hop_length))
        C_t = C_t[..., :n_frames]

    # librosa: ``C_scale = np.sqrt(lengths)``
    C_scale = torch.from_numpy(np.sqrt(lengths)).to(device=device,
                                                     dtype=torch.float64)

    # librosa: build per-octave sr/hop schedule
    # Assume the top octave is at the full rate
    srs: list[float] = [float(sr)]
    hops: list[int] = [hop_length]
    for _ in range(n_octaves - 1):
        if hops[0] % 2 == 0:
            srs.insert(0, srs[0] * 0.5)
            hops.insert(0, hops[0] // 2)
        else:
            srs.insert(0, srs[0])
            hops.insert(0, hops[0])

    y: torch.Tensor | None = None

    for i, (my_sr, my_hop) in enumerate(zip(srs, hops)):
        n_filters = min(bins_per_octave, n_bins - bins_per_octave * i)
        sl = slice(bins_per_octave * i, bins_per_octave * i + n_filters)

        # librosa: fft_basis, n_fft, _ = __vqt_filter_fft(my_sr, freqs[sl], ...)
        fft_basis, n_fft, _ = _vqt_filter_fft(
            my_sr, freqs[sl], filter_scale, norm, my_hop,
            window=window, alpha=alpha[sl], device=device)

        # librosa: ``inv_basis = fft_basis.conjugate().T.todense()``
        inv_basis = fft_basis.conj().T  # (n_fft//2+1, n_filters)

        # librosa: ``freq_power = 1 / np.sum(util.abs2(inv_basis), axis=0)``
        freq_power = 1.0 / (torch.abs(inv_basis) ** 2).sum(dim=0)

        # librosa: ``freq_power *= n_fft / lengths[sl]``
        oct_len = torch.from_numpy(lengths[sl]).to(device=device,
                                                    dtype=torch.float64)
        freq_power = freq_power * (n_fft / oct_len)

        # librosa:
        #   if scale:
        #     D_oct = einsum("fc,c,c,...ct->...ft",
        #                    inv_basis, C_scale[sl], freq_power, C[...,sl,:])
        #   else:
        #     D_oct = einsum("fc,c,...ct->...ft",
        #                    inv_basis, freq_power, C[...,sl,:])
        if scale:
            weighted = (C_scale[sl] * freq_power).unsqueeze(-1) * C_t[..., sl, :]
        else:
            weighted = freq_power.unsqueeze(-1) * C_t[..., sl, :]

        D_oct = torch.matmul(inv_basis.to(torch.complex128),
                             weighted.to(torch.complex128))

        # librosa: ``y_oct = istft(D_oct, window="ones", hop_length=my_hop)``
        y_oct = _istft(D_oct, n_fft, my_hop)

        # librosa: ``y_oct = audio.resample(y_oct, orig_sr=1,
        #                                   target_sr=sr//my_sr, scale=False)``
        resample_factor = int(sr / my_sr)
        if resample_factor > 1:
            y_oct = _resample(y_oct, orig_sr=1, target_sr=resample_factor,
                              scale=False)

        # librosa: accumulate
        if y is None:
            y = y_oct
        else:
            n = min(y.shape[-1], y_oct.shape[-1])
            y[..., :n] = y[..., :n] + y_oct[..., :n]

    assert y is not None

    # librosa: ``y = util.fix_length(y, size=length)``
    if length is not None:
        if y.shape[-1] > length:
            y = y[..., :length]
        elif y.shape[-1] < length:
            y = _pad(y, 0, length - y.shape[-1], mode="constant")

    return y
