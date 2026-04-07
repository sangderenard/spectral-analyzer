"""Audio compositing for CQT analysis.

Builds a single analysis-ready signal from multiple time regions of an audio
file, with configurable padding at each boundary and FFT-hop-aligned edges.

Composite spec format (for --composite CLI arg)::

    pad:start:end:pad [; pad:start:end:pad ...]

Padding types
    z   zero-pad  (silence — clean spectral separation)
    r   reflect   (mirror the region edge — smooth spectral transition)
    n   none      (hard cut — no padding at this boundary)

Time expressions
    15      plain seconds from the start of the audio
    L-25    audio length minus 25 seconds
    L       the full audio length

Presets
    default     z:15:25:z;z:L-25:L-15:z

Example
    --composite "z:10:20:z; z:L-20:L-10:z"
    Extracts 10–20 s and the last 10 s of the track, with zero pads on every
    boundary, suitable for direct comparison of beginning vs. ending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One time region to extract, with a padding type on each side."""
    start_s: float
    end_s: float
    pad_before: str   # "z", "r", or "n"
    pad_after: str    # "z", "r", or "n"


@dataclass
class RegionInfo:
    """Where a real-data region sits inside the composited signal."""
    seg_idx: int
    original_start_s: float
    original_end_s: float
    composite_start_sample: int
    composite_end_sample: int


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_time_expr(expr: str, audio_len_s: float) -> float:
    """Parse a time expression: plain seconds or ``L-X``."""
    expr = expr.strip()
    if expr.upper().startswith("L"):
        rest = expr[1:].strip()
        if not rest:
            return audio_len_s
        return audio_len_s + float(rest)   # "L-25" → L + (-25)
    return float(expr)


def _normalize_pad(p: str) -> str:
    p = p.strip().lower()
    if p in ("z", "zero"):
        return "z"
    if p in ("r", "reflect"):
        return "r"
    if p in ("n", "none"):
        return "n"
    raise ValueError(f"Unknown pad type '{p}'; expected z / r / n "
                     f"(zero / reflect / none)")


def parse_composite_spec(spec: str, audio_len_s: float) -> List[Segment]:
    """Parse a composite spec string into a list of :class:`Segment`."""
    if spec.strip().lower() == "default":
        spec = "z:15:25:z;z:L-25:L-15:z"

    segments: List[Segment] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split(":")]
        if len(fields) != 4:
            raise ValueError(
                f"Each segment needs 4 colon-separated fields "
                f"(pad:start:end:pad), got {len(fields)}: '{part}'"
            )
        pb, t0_s, t1_s, pa = fields
        segments.append(Segment(
            start_s=parse_time_expr(t0_s, audio_len_s),
            end_s=parse_time_expr(t1_s, audio_len_s),
            pad_before=_normalize_pad(pb),
            pad_after=_normalize_pad(pa),
        ))
    if not segments:
        raise ValueError("Composite spec produced no segments.")
    return segments


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _hop_align(n: int, hop: int) -> int:
    """Round *n* up to the next multiple of *hop*."""
    return ((n + hop - 1) // hop) * hop


def _make_pad(pad_type: str, region: np.ndarray,
              length: int, side: str) -> np.ndarray:
    """Build a padding array of *length* samples.

    Parameters
    ----------
    pad_type : "z", "r", or "n"
    region   : the real-data slice this pad adjoins
    length   : desired pad length in samples (may be 0)
    side     : "before" or "after" — which edge of *region* to mirror for "r"
    """
    if length <= 0 or pad_type == "n":
        return np.empty(0, dtype=region.dtype)

    if pad_type == "z":
        return np.zeros(length, dtype=region.dtype)

    if pad_type == "r":
        if side == "before":
            src = region[1:length + 1]
        else:
            rlen = len(region)
            src = region[max(rlen - length - 1, 0):rlen - 1]
        # If the region is shorter than the requested pad, tile it.
        if len(src) == 0:
            return np.zeros(length, dtype=region.dtype)
        if len(src) < length:
            reps = (length // len(src)) + 1
            src = np.tile(src, reps)
        return src[:length][::-1].copy()

    raise ValueError(f"Unknown pad type: {pad_type}")


def build_composite(
    x: np.ndarray,
    sr: int,
    segments: List[Segment],
    hop: int,
    gap_seconds: float = 0.5,
) -> Tuple[np.ndarray, List[RegionInfo]]:
    """Build a composited signal from *segments* of *x*.

    Each segment is extracted from the original audio, hop-length–aligned,
    then surrounded by the requested padding type.  Between consecutive
    segments the trailing pad of one and the leading pad of the next form
    the inter-region gap.  All pad and region lengths are rounded up to the
    next ``hop`` boundary so that CQT frame edges land cleanly on real-data
    boundaries.

    The returned signal is meant to be passed directly to
    ``compute_cqt(..., zero_front=True)`` — ``compute_cqt`` still adds its
    own outer CQT-kernel pads for correct low-frequency support, but uses
    zero (not reflect) padding at the front so the leading silence is
    preserved.

    Parameters
    ----------
    x            : 1-D float64 original audio
    sr           : sample rate
    segments     : parsed composite spec (from :func:`parse_composite_spec`)
    hop          : CQT hop length
    gap_seconds  : duration of each *half*-gap between regions (seconds).
                   The visible gap between two regions is ``2 × gap_seconds``.

    Returns
    -------
    composite : 1-D float32 signal
    regions   : metadata list — one entry per segment
    """
    # Pad lengths, hop-aligned.
    # Outer (leading / trailing) pads = same as inner for visual consistency.
    pad_len = _hop_align(max(int(gap_seconds * sr), hop * 2), hop)

    pieces: List[np.ndarray] = []
    regions: List[RegionInfo] = []
    cursor = 0  # running sample position in the composited signal

    for i, seg in enumerate(segments):
        s0 = max(0, min(int(seg.start_s * sr), len(x)))
        s1 = max(s0, min(int(seg.end_s * sr), len(x)))
        if s1 <= s0:
            raise ValueError(
                f"Segment {i} is empty: {seg.start_s:.3f} s – {seg.end_s:.3f} s "
                f"resolved to samples {s0}–{s1} (audio length "
                f"{len(x)/sr:.3f} s)")
        region = x[s0:s1].astype(np.float32)

        # Hop-align the region length so frame boundaries land on data edges.
        aligned_len = _hop_align(len(region), hop)
        if aligned_len > len(region):
            region = np.pad(region, (0, aligned_len - len(region)))

        # Padding before this segment.
        pad_before = _make_pad(seg.pad_before, region, pad_len, "before")
        pieces.append(pad_before)
        cursor += len(pad_before)

        # The real data.
        region_start = cursor
        pieces.append(region)
        cursor += len(region)
        region_end = cursor

        # Padding after this segment.
        pad_after = _make_pad(seg.pad_after, region, pad_len, "after")
        pieces.append(pad_after)
        cursor += len(pad_after)

        regions.append(RegionInfo(
            seg_idx=i,
            original_start_s=seg.start_s,
            original_end_s=seg.end_s,
            composite_start_sample=region_start,
            composite_end_sample=region_end,
        ))

    composite = np.concatenate(pieces)
    return composite, regions
