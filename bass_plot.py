#!/usr/bin/env python3
"""
Data Texture Generator — produces 16-bit RGBA data textures from saved .npz data.

Usage
-----
    python bass_plot.py <analysis_dir_or_npz>  [--outdir DIR]

Reads ``cqt_data.npz`` produced by ``bass_analysis.py`` and generates 16-bit
per-channel RGBA PNG data textures for OpenGL consumption.

Outputs
-------
    cqt_left_complex.png    R=real_L,  G=imag_L,  B=0,              A=content_mask
    cqt_right_complex.png   R=real_R,  G=imag_R,  B=0,              A=content_mask
    cqt_left_polar.png      R=mag_L,   G=phase_L, B=0,              A=content_mask
    cqt_right_polar.png     R=mag_R,   G=phase_R, B=0,              A=content_mask
    cqt_mag_stereo.png      R=mag_L,   G=mag_R,   B=similarity_mag, A=content_mask
    cqt_phase_stereo.png    R=phase_L, G=phase_R, B=similarity_pha, A=content_mask
    onset.png               R=onset,   G=0,       B=0,              A=content_mask
    axis_freq.png           frequency axis (split left|right at center)
    axis_time.png           time axis (split top|bottom at center)
    texture_meta.json       normalization parameters for shader decoding

Row 0 of each texture corresponds to the lowest frequency bin.
Alpha encodes content mask: 65535 = real content, 0 = pure padding.

Axis textures are 8-bit RGBA with transparent background.  The frequency
axis is split vertically at the center (left half = right-aligned labels
for display left of the spectrogram, right half = left-aligned labels for
the right side).  The time axis is split horizontally at the center.

Dependencies
------------
pip install numpy Pillow
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPS = 1e-12


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 16-bit RGBA data textures from saved CQT data."
    )
    parser.add_argument("input",
                        help="Path to analysis directory or cqt_data.npz file")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: same as input)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 16-bit RGBA PNG writer (no dependencies beyond stdlib)
# ---------------------------------------------------------------------------

def write_rgba16_png(path: str, data: np.ndarray) -> None:
    """Write a 16-bit-per-channel RGBA PNG.

    Parameters
    ----------
    path : str
        Output file path.
    data : ndarray, shape (H, W, 4), dtype uint16
        RGBA pixel data.
    """
    h, w = data.shape[:2]

    def _chunk(ctype: bytes, cdata: bytes) -> bytes:
        raw = ctype + cdata
        crc = struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
        return struct.pack(">I", len(cdata)) + raw + crc

    # IHDR: bit_depth=16, color_type=6 (RGBA)
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0)

    # Vectorised buffer: big-endian conversion + prepend filter byte per row.
    be = data.astype(">u2")
    row_bytes = be.view(np.uint8).reshape(h, -1)       # (h, w*8) view
    scanlines = np.empty((h, 1 + row_bytes.shape[1]), dtype=np.uint8)
    scanlines[:, 0] = 0                                 # filter: None
    scanlines[:, 1:] = row_bytes
    idat = zlib.compress(scanlines, 6)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", idat))
        f.write(_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _to_u16(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map float array from [vmin, vmax] to uint16 [0, 65535]."""
    span = vmax - vmin
    if span < EPS:
        return np.zeros_like(arr, dtype=np.uint16)
    normed = (arr - vmin) / span
    return (np.clip(normed, 0.0, 1.0) * 65535 + 0.5).astype(np.uint16)


def _build_rgba(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                a: np.ndarray) -> np.ndarray:
    """Stack four uint16 2-D arrays into (H, W, 4) RGBA."""
    return np.stack([r, g, b, a], axis=-1)


# ---------------------------------------------------------------------------
# Axis texture helpers
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _nice_time_step(duration: float, n_px: int) -> float:
    """Choose a round tick step giving ~1 tick per 80-120 pixels."""
    target_count = max(1, n_px // 100)
    raw = duration / target_count
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    for nice in [1.0, 2.0, 5.0, 10.0]:
        step = nice * mag
        if duration / step <= target_count * 1.5:
            return step
    return mag * 10.0


def render_freq_axis(
    freqs: np.ndarray,
    semitone_mask: np.ndarray,
    note_names: list[str],
    half_width: int = 200,
) -> Image.Image:
    """Render a split frequency-axis texture.

    Left half: right-aligned labels (for display left of the spectrogram).
    Right half: left-aligned labels (for display right of the spectrogram).
    Row 0 corresponds to the lowest frequency bin.
    """
    n_bins = len(freqs)
    width = half_width * 2
    img = Image.new("RGBA", (width, n_bins), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(11)

    for i in np.flatnonzero(semitone_mask):
        y = n_bins - 1 - i          # row 0 = lowest freq, PIL y 0 = top
        name = note_names[i] if i < len(note_names) else ""
        if not name:
            continue
        freq = freqs[i]
        label = f"{name}  {freq:.1f} Hz"
        bbox = font.getbbox(label)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]

        # Left half — right-aligned
        draw.line([(half_width - 6, y), (half_width - 1, y)],
                  fill=(100, 100, 100, 255))
        draw.text((half_width - 8 - lw, y - lh // 2), label,
                  fill=(200, 200, 200, 255), font=font)

        # Right half — left-aligned
        draw.line([(half_width, y), (half_width + 5, y)],
                  fill=(100, 100, 100, 255))
        draw.text((half_width + 8, y - lh // 2), label,
                  fill=(200, 200, 200, 255), font=font)

    return img


def render_time_axis(
    times: np.ndarray,
    n_frames: int,
    half_height: int = 32,
) -> Image.Image:
    """Render a split time-axis texture.

    Top half: labels at the bottom edge (for display above the spectrogram).
    Bottom half: labels at the top edge (for display below the spectrogram).
    """
    height = half_height * 2
    img = Image.new("RGBA", (n_frames, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(11)

    t0, t1 = float(times[0]), float(times[-1])
    dur = t1 - t0
    if dur <= 0:
        return img

    tick_step = _nice_time_step(dur, n_frames)
    t_first = math.ceil(t0 / tick_step) * tick_step
    t = t_first
    while t <= t1:
        frac = (t - t0) / dur
        x = int(frac * (n_frames - 1))
        label = f"{t:.1f}"
        bbox = font.getbbox(label)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]

        # Top half — labels hang from bottom edge
        draw.line([(x, half_height - 4), (x, half_height - 1)],
                  fill=(100, 100, 100, 255))
        draw.text((x - lw // 2, half_height - 5 - lh), label,
                  fill=(200, 200, 200, 255), font=font)

        # Bottom half — labels sit on top edge
        draw.line([(x, half_height), (x, half_height + 3)],
                  fill=(100, 100, 100, 255))
        draw.text((x - lw // 2, half_height + 5), label,
                  fill=(200, 200, 200, 255), font=font)

        t += tick_step

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    inp = args.input
    if os.path.isdir(inp):
        npz_path = os.path.join(inp, "cqt_data.npz")
    else:
        npz_path = inp
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Cannot find {npz_path}")

    outdir = args.outdir or os.path.dirname(npz_path)
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading {npz_path} ...")
    d = np.load(npz_path, allow_pickle=True)

    real_L = d["real_left"]
    imag_L = d["imag_left"]
    real_R = d["real_right"]
    imag_R = d["imag_right"]
    phase_L = d["phase_left"]
    phase_R = d["phase_right"]

    n_bins, n_frames = real_L.shape
    is_stereo = bool(d["is_stereo"])

    print(f"  {n_bins} bins x {n_frames} frames  "
          f"{'stereo' if is_stereo else 'mono'}")

    # --- Derived quantities ---
    mag_L = np.sqrt(real_L ** 2 + imag_L ** 2)
    mag_R = np.sqrt(real_R ** 2 + imag_R ** 2)

    # Magnitude similarity: 1 when equal, 0 when one channel is silent.
    mag_peak = np.maximum(mag_L, mag_R)
    mag_similarity = np.where(mag_peak < EPS, 1.0,
                              np.minimum(mag_L, mag_R) / mag_peak)

    # Phase similarity: 1 when in-phase, 0 when anti-phase.
    phase_similarity = (1.0 + np.cos(phase_L - phase_R)) / 2.0

    # --- Content mask (alpha) ---
    pi = d["pad_influence"]
    if pi.size > 0:
        alpha = ((1.0 - pi) * 65535 + 0.5).astype(np.uint16)
    else:
        alpha = np.full((n_bins, n_frames), 65535, dtype=np.uint16)

    # --- Onset ---
    onset = d["onset"] if "onset" in d else None

    # --- Shared normalization ranges ---
    # Real / imag: symmetric so midpoint 32768 = 0.
    ri_abs_max = float(max(
        np.abs(real_L).max(), np.abs(real_R).max(),
        np.abs(imag_L).max(), np.abs(imag_R).max(),
    ))
    ri_min, ri_max = -ri_abs_max, ri_abs_max

    # Magnitude: shared [0, max].
    mag_max_val = float(max(mag_L.max(), mag_R.max()))

    # Phase: fixed [-pi, pi].
    ph_min, ph_max = -float(np.pi), float(np.pi)

    zeros = np.zeros((n_bins, n_frames), dtype=np.uint16)

    print("Generating data textures ...")
    meta: dict = {
        "format": "rgba16_png",
        "bit_depth": 16,
        "orientation": "row0_lowest_freq",
        "alpha": "content_mask (65535=real content, 0=pure padding)",
        "n_bins": n_bins,
        "n_frames": n_frames,
        "is_stereo": is_stereo,
        "textures": {},
    }

    # 1) Left complex
    name = "cqt_left_complex"
    r = _to_u16(real_L, ri_min, ri_max)
    g = _to_u16(imag_L, ri_min, ri_max)
    write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                     _build_rgba(r, g, zeros, alpha))
    meta["textures"][name] = {
        "R": {"field": "real_left", "min": ri_min, "max": ri_max},
        "G": {"field": "imag_left", "min": ri_min, "max": ri_max},
        "B": None,
    }
    print(f"  {name}.png")

    # 2) Right complex
    name = "cqt_right_complex"
    r = _to_u16(real_R, ri_min, ri_max)
    g = _to_u16(imag_R, ri_min, ri_max)
    write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                     _build_rgba(r, g, zeros, alpha))
    meta["textures"][name] = {
        "R": {"field": "real_right", "min": ri_min, "max": ri_max},
        "G": {"field": "imag_right", "min": ri_min, "max": ri_max},
        "B": None,
    }
    print(f"  {name}.png")

    # 3) Left polar
    name = "cqt_left_polar"
    r = _to_u16(mag_L, 0.0, mag_max_val)
    g = _to_u16(phase_L, ph_min, ph_max)
    write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                     _build_rgba(r, g, zeros, alpha))
    meta["textures"][name] = {
        "R": {"field": "mag_left", "min": 0.0, "max": mag_max_val},
        "G": {"field": "phase_left", "min": ph_min, "max": ph_max},
        "B": None,
    }
    print(f"  {name}.png")

    # 4) Right polar
    name = "cqt_right_polar"
    r = _to_u16(mag_R, 0.0, mag_max_val)
    g = _to_u16(phase_R, ph_min, ph_max)
    write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                     _build_rgba(r, g, zeros, alpha))
    meta["textures"][name] = {
        "R": {"field": "mag_right", "min": 0.0, "max": mag_max_val},
        "G": {"field": "phase_right", "min": ph_min, "max": ph_max},
        "B": None,
    }
    print(f"  {name}.png")

    # 5) Stereo magnitude comparison
    name = "cqt_mag_stereo"
    r = _to_u16(mag_L, 0.0, mag_max_val)
    g = _to_u16(mag_R, 0.0, mag_max_val)
    b = _to_u16(mag_similarity, 0.0, 1.0)
    write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                     _build_rgba(r, g, b, alpha))
    meta["textures"][name] = {
        "R": {"field": "mag_left", "min": 0.0, "max": mag_max_val},
        "G": {"field": "mag_right", "min": 0.0, "max": mag_max_val},
        "B": {"field": "mag_similarity", "min": 0.0, "max": 1.0},
    }
    print(f"  {name}.png")

    # 6) Stereo phase comparison
    name = "cqt_phase_stereo"
    r = _to_u16(phase_L, ph_min, ph_max)
    g = _to_u16(phase_R, ph_min, ph_max)
    b = _to_u16(phase_similarity, 0.0, 1.0)
    write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                     _build_rgba(r, g, b, alpha))
    meta["textures"][name] = {
        "R": {"field": "phase_left", "min": ph_min, "max": ph_max},
        "G": {"field": "phase_right", "min": ph_min, "max": ph_max},
        "B": {"field": "phase_similarity", "min": 0.0, "max": 1.0},
    }
    print(f"  {name}.png")

    # 7) Onset
    if onset is not None:
        name = "onset"
        r = _to_u16(onset, 0.0, 1.0)
        write_rgba16_png(os.path.join(outdir, f"{name}.png"),
                         _build_rgba(r, zeros, zeros, alpha))
        meta["textures"][name] = {
            "R": {"field": "onset", "min": 0.0, "max": 1.0},
            "G": None,
            "B": None,
        }
        print(f"  {name}.png")
    else:
        print("  (onset not found in npz, skipping onset texture)")

    # --- Axis textures (8-bit RGBA, transparent background) ---
    print("Generating axis textures ...")
    freqs = d["freqs"]
    times = d["times"]
    semitone_mask = d["semitone_mask"]
    note_names = list(d["note_names"])

    freq_half_w = 200
    freq_img = render_freq_axis(freqs, semitone_mask, note_names,
                                half_width=freq_half_w)
    freq_img.save(os.path.join(outdir, "axis_freq.png"))
    print(f"  axis_freq.png  ({freq_img.width}x{freq_img.height})")

    time_half_h = 32
    time_img = render_time_axis(times, n_frames, half_height=time_half_h)
    time_img.save(os.path.join(outdir, "axis_time.png"))
    print(f"  axis_time.png  ({time_img.width}x{time_img.height})")

    meta["axes"] = {
        "freq": {
            "file": "axis_freq.png",
            "width": freq_img.width,
            "height": freq_img.height,
            "split": "vertical_center",
            "left_half": "right-aligned labels (display left of spectrogram)",
            "right_half": "left-aligned labels (display right of spectrogram)",
            "fmin": float(freqs[0]),
            "fmax": float(freqs[-1]),
        },
        "time": {
            "file": "axis_time.png",
            "width": time_img.width,
            "height": time_img.height,
            "split": "horizontal_center",
            "top_half": "labels at bottom edge (display above spectrogram)",
            "bottom_half": "labels at top edge (display below spectrogram)",
            "t_start": float(times[0]),
            "t_end": float(times[-1]),
        },
    }

    # --- Write metadata ---
    meta_path = os.path.join(outdir, "texture_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  texture_meta.json")

    print(f"\nData textures written to: {outdir}")


if __name__ == "__main__":
    main()
