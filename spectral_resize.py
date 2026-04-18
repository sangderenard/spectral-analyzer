#!/usr/bin/env python3
"""
spectral_resize.py — Frequency-aware spectrogram image resizer.

Uses a layered semi-nearest-neighbor algorithm that:
  • Enlarging  — NN block expansion blended with Lanczos at sub-pixel
                 boundaries, retaining digital grit / pixel accuracy.
  • Shrinking  — Exact fractional-area box filter per axis, preserving
                 moiré and interference patterns without blurring.

Each spatial axis is resized independently in an intelligently sorted
order (most-aggressive reduction first; largest expansion first for
upscaling) so each pass works on the most regular grid possible.

When an analytical JSON sidecar (produced by bass_viewer's export) is
supplied (auto-detected or explicit), frequency / time metadata is
embedded in the output sidecar and used to sharpen octave-boundary
edges on the frequency axis.

Usage
-----
    python spectral_resize.py input.png --width 3840 --height 2160
    python spectral_resize.py input.png -W 1920 -H 1080 --output out.png
    python spectral_resize.py input.png -W 7680 -H 4320 --sidecar data.json
    python spectral_resize.py input.png -W 800 -H 600 --blend-nn 0.80

Arguments
---------
    input           16-bit (or 8-bit) PNG to resize.
    -W / --width    Target width  in pixels.
    -H / --height   Target height in pixels.
    --sidecar PATH  JSON sidecar (auto-detected if omitted).
    --output PATH   Output PNG path (default: <base>_<W>x<H>.png beside input).
    --blend-nn FRAC Fraction of NN weight at sub-pixel boundary for upscaling.
                    0 = pure Lanczos, 1 = pure NN.  Default 0.72.
    --no-sidecar    Skip sidecar auto-detection / writing.
    --verbose       Extra diagnostic output.

Dependencies
------------
    numpy, scipy, Pillow
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom as _scipy_zoom


# ---------------------------------------------------------------------------
# PNG I/O  (16-bit round-trip, matching bass_viewer's _write_png_u16)
# ---------------------------------------------------------------------------

def _read_png_any(path: str) -> tuple[np.ndarray, int]:
    """Return (arr uint16 HxWx3, bit_depth).  Promotes 8-bit to uint16."""
    from PIL import Image
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA", "L", "I;16", "I;16B"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.dtype == np.uint8:
        return arr[..., :3].astype(np.uint16) * 257, 8  # 255 → 65535
    if arr.dtype == np.uint16:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr[..., :3], 16
    # float or int32 fallback
    arr = arr.astype(np.float64)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-9) * 65535.0
    return arr.astype(np.uint16)[..., :3], 16


def _write_png_u16(path: str, img_u16: np.ndarray) -> None:
    """Write uint16 RGB image as 16-bit PNG (no Pillow — mirrors bass_viewer)."""
    arr = np.asarray(img_u16)
    if arr.dtype != np.uint16:
        raise ValueError("img_u16 must be uint16")
    h, w, c = arr.shape
    assert c == 3
    color_type = 2  # RGB
    arr_be = np.ascontiguousarray(arr.astype(">u2", copy=False))
    rows = [b"\x00" + arr_be[y].tobytes() for y in range(h)]
    idat = zlib.compress(b"".join(rows), level=6)

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)
    blob = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(blob)


def _write_png_u8(path: str, img_u8: np.ndarray) -> None:
    """Write uint8 RGB image as 8-bit PNG (no Pillow — mirrors bass_viewer)."""
    arr = np.asarray(img_u8)
    if arr.dtype != np.uint8:
        raise ValueError("img_u8 must be uint8")
    h, w, c = arr.shape
    assert c == 3
    rows = [b"\x00" + arr[y].tobytes() for y in range(h)]
    idat = zlib.compress(b"".join(rows), level=6)

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # bit_depth=8, color_type=2 (RGB)
    blob = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(blob)


# ---------------------------------------------------------------------------
# Core 1-D resamplers
# ---------------------------------------------------------------------------

def _area_downscale_1d(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Exact fractional-area box filter along *axis*.

    Each output bin accumulates the exact fractional coverage of source
    bins, preserving interference / moiré without windowing artefacts.
    Works in float64 to avoid precision loss; caller converts back.
    """
    src = arr.shape[axis]
    if target == src:
        return arr
    assert target < src, "Use _semi_nn_lanczos_1d for upscaling"

    # Pre-move working axis to axis 0 for simpler indexing
    arr_f = np.moveaxis(arr.astype(np.float64, copy=False), axis, 0)
    rest = arr_f.shape[1:]
    out = np.empty((target,) + rest, dtype=np.float64)

    for o in range(target):
        src_lo = o * src / target
        src_hi = (o + 1) * src / target
        i0 = int(src_lo)
        i1 = int(math.ceil(src_hi))
        n = i1 - i0

        w = np.ones(n, dtype=np.float64)
        # fractional left edge
        w[0] = (i0 + 1) - src_lo
        # fractional right edge (may overlap with left if n==1)
        w[-1] = min(w[-1], src_hi - (i1 - 1))
        if n == 1:
            w[0] = src_hi - src_lo

        chunk = arr_f[i0:i1]                          # shape: (n, *rest)
        w_bc = w.reshape((n,) + (1,) * len(rest))
        out[o] = np.sum(chunk * w_bc, axis=0) / w.sum()

    return np.moveaxis(out, 0, axis)


def _semi_nn_lanczos_1d(arr: np.ndarray, target: int, axis: int,
                        blend_nn: float = 0.72) -> np.ndarray:
    """Semi-nearest-neighbor upscaler with Lanczos blend at sub-pixel edges.

    For each output pixel:
      • Compute sub-pixel distance to the nearest source pixel centre.
      • Interpolate between NN sample and a cubic-spline (Lanczos-like)
        sample.  Weight = blend_nn at pixel centres → pure NN grit;
        weight decreases smoothly toward block edges → Lanczos smoothing.

    ``blend_nn`` is the NN fraction at maximum sub-pixel offset (0.5 px).
    At pixel centres (0 offset) the result is always pure NN.
    """
    src_sz = arr.shape[axis]
    if target == src_sz:
        return arr
    assert target > src_sz, "Use _area_downscale_1d for downscaling"

    arr_f = np.asarray(arr, dtype=np.float64)

    # ---- NN reference ----
    coords_nn = (np.arange(target, dtype=np.float64) + 0.5) * (src_sz / target) - 0.5
    idx_nn = np.clip(np.round(coords_nn).astype(np.int64), 0, src_sz - 1)
    nn_out = np.take(arr_f, idx_nn, axis=axis)

    # ---- Cubic-spline (Lanczos-like) reference via scipy zoom ----
    zoom_factors = [1.0] * arr_f.ndim
    zoom_factors[axis] = target / src_sz
    lz_out = _scipy_zoom(arr_f, zoom_factors, order=3, prefilter=True,
                         mode="nearest")
    # scipy zoom may differ by ±1 pixel in size; trim/pad to exact target
    lz_out = _match_size(lz_out, target, axis)

    # ---- Per-output-position blend weight ----
    # subpix ∈ [0, 0.5]: 0 at pixel centres, 0.5 at block boundaries
    subpix = np.abs(coords_nn - np.round(coords_nn))          # [0, 0.5]
    # alpha_lz: 0 at centre (pure NN) → (1-blend_nn) at max offset
    alpha_lz = (subpix * 2.0) * (1.0 - blend_nn)             # [0, 1-blend_nn]

    shape_bc = [1] * arr_f.ndim
    shape_bc[axis] = target
    alpha_lz = alpha_lz.reshape(shape_bc)

    return nn_out * (1.0 - alpha_lz) + lz_out * alpha_lz


def _match_size(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Trim or zero-pad *arr* along *axis* to exactly *target* elements."""
    cur = arr.shape[axis]
    if cur == target:
        return arr
    if cur > target:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target)
        return arr[tuple(slices)]
    # pad
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - cur)
    return np.pad(arr, pad_width, mode="edge")


# ---------------------------------------------------------------------------
# Frequency-aware octave-boundary sharpening (when sidecar present)
# ---------------------------------------------------------------------------

def _octave_boundary_mask(freq_axis_hz: list[float], sharpness: float = 1.5
                          ) -> np.ndarray:
    """Return a per-row sharpness multiplier boosted at octave boundaries.

    The multiplier is 1.0 everywhere except at rows that straddle an
    exact octave (2x) frequency ratio, where it rises to *sharpness*.
    This gives the frequency axis a subtle clarifying push at the
    musically most important boundaries without affecting the rest.
    """
    freqs = np.asarray(freq_axis_hz, dtype=np.float64)
    if freqs.size < 2:
        return np.ones(len(freq_axis_hz), dtype=np.float64)

    mask = np.ones(len(freqs), dtype=np.float64)
    # Detect rows where the frequency crosses an octave boundary
    # (i.e., floor(log2(f)) changes between adjacent rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2f = np.where(freqs > 0, np.log2(np.maximum(freqs, 1e-9)), 0.0)
    octave_floor = np.floor(log2f).astype(np.int32)
    crossings = np.where(np.diff(octave_floor) != 0)[0]
    if crossings.size == 0:
        return mask
    # Gaussian bump of width ~2 rows centred on each crossing
    sigma = 1.2
    for c in crossings:
        for r in range(max(0, c - 3), min(len(freqs), c + 4)):
            bump = math.exp(-0.5 * ((r - c) / sigma) ** 2)
            mask[r] = max(mask[r], 1.0 + (sharpness - 1.0) * bump)
    return mask


def _apply_sharpness_mask(arr_f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a per-row sharpness multiplier around the row mean.

    Pixels deviate from the row mean; the deviation is amplified by the
    mask value, retaining mean luminance while sharpening edge contrast.
    """
    # arr_f: (H, W, C) float64
    row_mean = arr_f.mean(axis=1, keepdims=True)          # (H, 1, C)
    deviation = arr_f - row_mean
    m = mask[:, np.newaxis, np.newaxis]                    # (H, 1, 1)
    return row_mean + deviation * m


# ---------------------------------------------------------------------------
# Top-level resize
# ---------------------------------------------------------------------------

def resize_spectral(
    img_u16: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    blend_nn: float = 0.72,
    freq_axis_hz: list[float] | None = None,
    octave_sharpen: float = 1.5,
    verbose: bool = False,
) -> np.ndarray:
    """Resize *img_u16* (H×W×3 uint16) to *target_h* × *target_w*.

    Each axis is resized independently.  Axis processing order:
      • Mixed (one up, one down): downscale axis first.
      • Both downscale: most-aggressive reduction first.
      • Both upscale: largest expansion first.

    Returns uint16 array of shape (target_h, target_w, 3).
    """
    src_h, src_w = img_u16.shape[:2]
    scale_x = target_w / src_w
    scale_y = target_h / src_h

    if verbose:
        print(f"  src {src_w}×{src_h}  →  {target_w}×{target_h}"
              f"  (scale_x={scale_x:.4f}  scale_y={scale_y:.4f})")

    # ---- Build ordered list of (axis_name, axis_int, scale, target_size) ----
    axes = [
        ("x", 1, scale_x, target_w),
        ("y", 0, scale_y, target_h),
    ]

    def _sort_key(item: tuple) -> tuple:
        _, _, scale, _ = item
        if scale < 1.0:
            return (0, scale)        # downscale first, most aggressive first
        return (1, -scale)           # upscale second, largest first

    axes.sort(key=_sort_key)

    # Work in float64 throughout; convert at the very end
    current = img_u16.astype(np.float64, copy=True)

    for name, axis, scale, tgt_sz in axes:
        if scale == 1.0:
            continue
        if verbose:
            direction = "↑" if scale > 1.0 else "↓"
            print(f"  {direction} resizing {name}-axis  "
                  f"{current.shape[axis]} → {tgt_sz}  (×{scale:.4f})")
        if scale < 1.0:
            current = _area_downscale_1d(current, tgt_sz, axis)
        else:
            current = _semi_nn_lanczos_1d(current, tgt_sz, axis, blend_nn)

    # ---- Frequency-aware octave sharpening (freq axis = axis 0) ----
    if freq_axis_hz is not None and octave_sharpen > 1.0:
        # freq_axis_hz is relative to the *output* image (already resized)
        out_freq_axis = _interpolate_axis_metadata(freq_axis_hz, target_h)
        mask = _octave_boundary_mask(out_freq_axis, sharpness=octave_sharpen)
        if verbose:
            n_crossings = int(np.sum(np.diff(np.floor(
                np.log2(np.maximum(np.asarray(out_freq_axis), 1e-9))).astype(int)) != 0))
            print(f"  octave sharpening: {n_crossings} boundary crossings")
        current = _apply_sharpness_mask(current, mask)

    # Downscale from internal float64 range (0–65535) to uint8 (0–255)
    out = np.clip(np.round(current / 257.0), 0, 255).astype(np.uint8)
    return out


def _interpolate_axis_metadata(values: list[float], target_len: int) -> list[float]:
    """Resample a metadata axis (freq or time) to match a new pixel count."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0] * target_len
    src_coords = np.linspace(0.0, 1.0, arr.size)
    dst_coords = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_coords, src_coords, arr).tolist()


# ---------------------------------------------------------------------------
# Sidecar load / write
# ---------------------------------------------------------------------------

def _load_sidecar(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_sidecar(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _derive_output_sidecar(
    src: dict[str, Any],
    new_w: int,
    new_h: int,
    out_image_file: str,
) -> dict[str, Any]:
    """Build a sidecar for the resized output from the source sidecar."""
    import copy
    out = copy.deepcopy(src)
    out["image_file"] = os.path.basename(out_image_file)
    out["image_shape"] = [new_h, new_w]
    out["bit_depth"] = 8
    out["encoding"] = "uint8"
    out["scale_factor"] = 255
    out["derived_from"] = src.get("image_file", "")
    out["resize_note"] = "Resized via spectral_resize.py (semi-NN + area-avg)"

    # Re-interpolate per-pixel metadata axes to the new dimensions
    if "freq_axis_hz" in src:
        out["freq_axis_hz"] = _interpolate_axis_metadata(src["freq_axis_hz"], new_h)
    if "time_axis_s" in src:
        out["time_axis_s"] = _interpolate_axis_metadata(src["time_axis_s"], new_w)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frequency-aware spectral image resizer (semi-NN + area-avg).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input PNG (16-bit or 8-bit).")
    p.add_argument("-W", "--width", type=int, required=True, help="Target width in pixels.")
    p.add_argument("-H", "--height", type=int, required=True, help="Target height in pixels.")
    p.add_argument("--sidecar", default=None,
                   help="JSON sidecar path (auto-detected if omitted).")
    p.add_argument("--output", default=None,
                   help="Output PNG path.")
    p.add_argument("--blend-nn", type=float, default=0.72,
                   help="NN fraction at sub-pixel boundary for upscaling (default 0.72).")
    p.add_argument("--octave-sharpen", type=float, default=1.5,
                   help="Sharpness multiplier at octave boundaries (default 1.5, 1.0 = off).")
    p.add_argument("--no-sidecar", action="store_true",
                   help="Disable sidecar auto-detection and writing.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    in_path = os.path.abspath(args.input)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")

    # ---- Auto-detect sidecar ----
    sidecar_data: dict[str, Any] | None = None
    sidecar_path_in: str | None = None
    if not args.no_sidecar:
        if args.sidecar:
            sidecar_path_in = os.path.abspath(args.sidecar)
        else:
            candidate = os.path.splitext(in_path)[0] + ".json"
            if os.path.isfile(candidate):
                sidecar_path_in = candidate
        if sidecar_path_in:
            sidecar_data = _load_sidecar(sidecar_path_in)
            if args.verbose:
                print(f"  loaded sidecar: {sidecar_path_in}")

    # ---- Build output path ----
    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        stem = Path(in_path).stem
        out_dir = os.path.dirname(in_path)
        out_path = os.path.join(out_dir, f"{stem}_{args.width}x{args.height}.png")

    # ---- Load image ----
    img_u16, src_bit_depth = _read_png_any(in_path)
    src_h, src_w = img_u16.shape[:2]
    if args.verbose:
        print(f"Loaded: {in_path}  ({src_w}×{src_h}, {src_bit_depth}-bit)")

    # ---- Resize ----
    freq_axis: list[float] | None = None
    if sidecar_data and "freq_axis_hz" in sidecar_data:
        freq_axis = sidecar_data["freq_axis_hz"]

    result = resize_spectral(
        img_u16,
        args.width,
        args.height,
        blend_nn=args.blend_nn,
        freq_axis_hz=freq_axis,
        octave_sharpen=args.octave_sharpen,
        verbose=args.verbose,
    )

    # ---- Write output PNG ----
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _write_png_u8(out_path, result)
    print(f"Saved: {out_path}  ({args.width}×{args.height})")

    # ---- Write output sidecar ----
    if sidecar_data is not None and not args.no_sidecar:
        out_sidecar = _derive_output_sidecar(sidecar_data, args.width, args.height, out_path)
        sidecar_out_path = os.path.splitext(out_path)[0] + ".json"
        _write_sidecar(sidecar_out_path, out_sidecar)
        if args.verbose:
            print(f"  sidecar: {sidecar_out_path}")


if __name__ == "__main__":
    main()
