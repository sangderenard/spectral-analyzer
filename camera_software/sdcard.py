"""camera_software/sdcard.py
============================
SD-card I/O and tonemapping for the camera system.

All file output from the ray tracer flows through here so that:
  - Raw linear-light float arrays are preserved as compressed .npz files.
  - Tonemapped 8-bit PNGs are derived from the raw arrays.
  - Neither operation is scattered across the application.

Directory layout (default root: ``camera/sdcard/`` inside the repo)
--------------------------------------------------------------------
    camera/sdcard/
        raw/   ← compressed .npz archives, one per exposure
        png/   ← tonemapped .png images

Tonemappers
-----------
tonemap_log1p(linear, white_percentile=99.0)
    Mirrors the C++ ``reduce_endpoints_to_sensor_image`` curve:
    log1p-compand followed by a mild Reinhard knee and slight
    luminance desaturation.  Suitable for BDPT/spectral output.

tonemap_percentile(linear, percentile=99.9, disc_mask=False)
    Simple percentile-normalise to [0, 1].  When disc_mask=True the
    normalisation is computed only over the inscribed circular disc
    (physical sensor aperture) so corner artefacts do not drive the
    white point.

SDCard
------
Lightweight class that owns the on-disk path and provides:
    save_raw(stamp, tag, arr)
        Save arr as ``raw/{stamp}_{tag}.npz`` (np.savez_compressed).
    save_png(stamp, tag, arr, tonemap="log1p", disc_mask=False)
        Tonemap arr and write ``png/{stamp}_{tag}.png``.
    save_exposure(stamp, images)
        Convenience: iterate a {tag: arr} dict and call save_raw +
        save_png for every non-empty array.
"""
from __future__ import annotations

import math
import os
import struct
import zlib
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

__all__ = [
    "tonemap_reinhard",
    "tonemap_log1p",
    "tonemap_percentile",
    "TONEMAP_MODES",
    "SDCard",
    "DEFAULT_SDCARD_ROOT",
]

# Repo-relative default: camera/sdcard/ two levels up from this file.
DEFAULT_SDCARD_ROOT: Path = Path(__file__).resolve().parent.parent / "camera" / "sdcard"


# ---------------------------------------------------------------------------
# Tonemappers
# ---------------------------------------------------------------------------

def tonemap_reinhard(
    linear: np.ndarray,
    white_percentile: float = 99.0,
) -> np.ndarray:
    """Extended Reinhard tonemapper: ``y = x / (1 + x)``.

    Highlights are gently compressed toward white rather than clipped,
    so over-exposed areas retain visible detail.  Black maps exactly to
    black (0 → 0), so shadow structure is fully preserved.

    The linear array is normalised by its *white_percentile* value
    before the Reinhard curve is applied so the curve midpoint sits at
    a meaningful scene brightness rather than an arbitrary photon count.

    Input: any float (H,W,3) in linear light.
    Output: float32 in [0, 1)  (Reinhard never reaches 1.0 exactly,
            but np.clip ensures the range is safe for 8-bit output).
    """
    arr = np.asarray(linear, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"tonemap_reinhard expects (H,W,3), got {arr.shape}")

    white = float(np.percentile(arr, float(white_percentile)))
    if white < 1e-8:
        white = float(arr.max())
    if white < 1e-8:
        white = 1.0

    x = np.maximum(arr, 0.0) / white
    y = x / (1.0 + x)           # extended Reinhard
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def tonemap_log1p(
    linear: np.ndarray,
    white_percentile: float = 99.0,
) -> np.ndarray:
    """Log1p compander + Reinhard knee + luminance desaturation.

    Mirrors the C++ curve in ``reduce_endpoints_to_sensor_image``.
    Input is any float (H,W,3) array in linear light.
    Output is float32 in [0, 1].

    The native dtype of *linear* is read but the computation is
    performed in float64 for precision; output is float32.
    """
    arr = np.asarray(linear, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"tonemap_log1p expects (H,W,3), got {arr.shape}")

    white = float(np.percentile(arr, float(white_percentile)))
    if white < 1e-8:
        white = float(arr.max())
    if white < 1e-8:
        white = 1.0

    white_scale = white
    log_denom = math.log1p(6.0)

    x = np.maximum(arr, 0.0) / white_scale
    y = np.log1p(x * 6.0) / log_denom
    y = y / (1.0 + 0.18 * y)
    y = np.clip(y, 0.0, 1.0)

    # Slight luminance desaturation (mirrors C++ 0.90/0.10 blend)
    luma = (0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2])[..., np.newaxis]
    y = np.clip(0.90 * y + 0.10 * luma, 0.0, 1.0)

    return y.astype(np.float32)


def tonemap_percentile(
    linear: np.ndarray,
    percentile: float = 99.9,
    disc_mask: bool = False,
) -> np.ndarray:
    """Percentile-normalise to [0, 1].

    When *disc_mask* is True the white-point is computed only over
    the inscribed circular disc (physical sensor aperture), preventing
    corner artefacts from collapsing the valid image to black.

    Input: any float (H,W,3).
    Output: float32 in [0, 1] with out-of-disc pixels zeroed when
    disc_mask is True.
    """
    arr = np.asarray(linear, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"tonemap_percentile expects (H,W,3), got {arr.shape}")

    if disc_mask:
        h, w = arr.shape[:2]
        cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
        r2 = (min(h, w) * 0.5) ** 2
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r2  # (H,W)
        disc_vals = arr[mask]
        peak = float(np.percentile(disc_vals, percentile)) if disc_vals.size else 0.0
        if peak <= 0.0:
            peak = float(disc_vals.max()) if disc_vals.size else 0.0
        if peak <= 0.0:
            peak = 1.0
        result = arr.copy()
        result[~mask] = 0.0
        result = np.clip(result / peak, 0.0, 1.0)
    else:
        peak = float(np.percentile(arr, percentile))
        if peak <= 0.0:
            peak = float(arr.max())
        if peak <= 0.0:
            peak = 1.0
        result = np.clip(arr / peak, 0.0, 1.0)

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# PNG encoder (PIL or stdlib fallback)
# ---------------------------------------------------------------------------

def _write_png(arr_f32: np.ndarray, path: Union[str, Path]) -> None:
    """Write a float32 (H,W,3) image in [0,1] as an 8-bit PNG."""
    u8 = (np.clip(arr_f32, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image as _PILImage
        _PILImage.fromarray(u8, mode="RGB").save(str(path))
        return
    except ImportError:
        pass
    # Minimal stdlib-only PNG writer (no external deps)
    h, w = u8.shape[:2]

    def _chunk(tag: bytes, data: bytes) -> bytes:
        header = struct.pack(">I", len(data)) + tag + data
        return header + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + u8[y].tobytes() for y in range(h))
    body = (
        _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(rows, 6))
        + _chunk(b"IEND", b"")
    )
    with open(str(path), "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + body)


# ---------------------------------------------------------------------------
# SDCard
# ---------------------------------------------------------------------------

# Registry of all available tonemappers.  Keys are the string names accepted
# by SDCard.save_png / save_exposure.  Add new entries here to extend.
TONEMAP_MODES: Dict[str, Callable] = {
    "reinhard":   tonemap_reinhard,
    "log1p":      tonemap_log1p,
    "percentile": tonemap_percentile,
}

DEFAULT_TONEMAP: str = "reinhard"

# Back-compat alias kept for any callers that imported the private name.
_TONEMAP_FNS = TONEMAP_MODES


class SDCard:
    """Virtual SD card — saves raw .npz and tonemapped .png to disk.

    Parameters
    ----------
    root :
        Directory that plays the role of the SD card root.
        Defaults to ``camera/sdcard/`` at the repo root.
        Sub-directories ``raw/`` and ``png/`` are created on first write.
    """

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        self._root = Path(root) if root is not None else DEFAULT_SDCARD_ROOT
        self._raw_dir = self._root / "raw"
        self._png_dir = self._root / "png"

    def _ensure_dirs(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._png_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Low-level writers
    # ------------------------------------------------------------------

    def save_raw(self, stamp: str, tag: str, arr: np.ndarray) -> Path:
        """Save *arr* as a compressed .npz file.

        The array's native dtype is preserved exactly.

        Returns the path written.
        """
        self._ensure_dirs()
        path = self._raw_dir / f"{stamp}_{tag}.npz"
        np.savez_compressed(str(path), image=arr)
        return path

    def save_png(
        self,
        stamp: str,
        tag: str,
        arr: np.ndarray,
        tonemap: str = DEFAULT_TONEMAP,
        disc_mask: bool = False,
    ) -> Path:
        """Tonemap *arr* and write a PNG.

        Parameters
        ----------
        tonemap :
            Name of the tonemapper to use.  Must be a key in
            ``TONEMAP_MODES``.  Defaults to ``DEFAULT_TONEMAP``
            (currently ``"reinhard"``).  Available modes:

            ``"reinhard"``   — extended Reinhard; highlights compress
                               smoothly, blacks preserved exactly.
            ``"log1p"``      — log1p compander with Reinhard knee;
                               mirrors the C++ reduction curve.
            ``"percentile"`` — simple percentile normalise.
        disc_mask :
            When True, the white-point is computed only over the
            inscribed circular disc (physical sensor aperture), then
            corners are zeroed.  Applies regardless of *tonemap*.

        Returns the path written.
        """
        self._ensure_dirs()
        path = self._png_dir / f"{stamp}_{tag}.png"
        fn = TONEMAP_MODES.get(tonemap, tonemap_reinhard)
        if disc_mask:
            # Build an inscribed-disc mask, zero corners, then tonemap.
            src = np.asarray(arr, dtype=np.float64)
            h, w = src.shape[:2]
            cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
            r2 = (min(h, w) * 0.5) ** 2
            yy, xx = np.ogrid[:h, :w]
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r2
            src[~mask] = 0.0
            mapped = fn(src)
            mapped[~mask] = 0.0
        else:
            mapped = fn(arr)
        _write_png(mapped, path)
        return path

    # ------------------------------------------------------------------
    # Exposure-level convenience
    # ------------------------------------------------------------------

    def save_exposure(
        self,
        stamp: str,
        images: Dict[str, Optional[np.ndarray]],
        tonemap: str = DEFAULT_TONEMAP,
        disc_tags: Optional[set] = None,
    ) -> Dict[str, str]:
        """Save raw + PNG for every non-empty array in *images*.

        Parameters
        ----------
        stamp :
            Timestamp / identifier string used in every filename.
        images :
            ``{tag: array_or_None}`` mapping.  Tags whose array is None
            or all-zero are skipped.
        tonemap :
            Tonemapper to use for PNG output (default ``"log1p"``).
        disc_tags :
            Set of tag names for which disc masking should be applied
            (e.g. ``{"bdpt"}`` for the circular physical sensor).

        Returns
        -------
        dict  ``{tag: "raw path | png path"}`` for every tag saved.
        """
        if disc_tags is None:
            disc_tags = set()
        saved: Dict[str, str] = {}
        for tag, arr in images.items():
            if arr is None:
                continue
            if not (arr.ndim == 3 and arr.shape[2] == 3 and np.any(arr > 0)):
                print(f"[sdcard] {tag} empty — skipped", flush=True)
                continue
            raw_path = self.save_raw(stamp, tag, arr)
            png_path = self.save_png(
                stamp, tag, arr,
                tonemap=tonemap,
                disc_mask=(tag in disc_tags),
            )
            saved[tag] = f"{raw_path} | {png_path}"
            print(f"[sdcard] {tag} → {raw_path.name}  {png_path.name}", flush=True)
        return saved
