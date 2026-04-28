"""Complex-domain resonator core — string definitions and coupling matrix.

Provides:
- ResonatorString / StringCouplingConfig dataclasses
- build_string_coupling_matrix(): score-level harmonic + spatial coupling weights
  used to route sympathetic atom injections between driver FIFOs.

Physical string-body simulation (FDTD, two-way plate coupling, pickup/mic output)
is handled entirely by the C AcousticCoEvolver (acoustic_coevolver.h).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ResonatorString:
    """One complex resonant mode or string channel."""

    key: str
    fundamental_hz: float
    decay_s: float = 1.25
    drive_gain: float = 1.0
    phase_offset_rad: float = 0.0
    x: float = 0.0
    y: float = 0.0
    harmonic_weight: float = 1.0


@dataclass
class StringCouplingConfig:
    """Controls how string-to-string influence is estimated."""

    harmonic_sigma: float = 0.08
    distance_sigma: float = 1.25
    harmonic_strength: float = 0.7
    distance_strength: float = 0.3
    base_strength: float = 0.18
    max_coupling: float = 0.45


def _safe_decay(decay_s: float, sr: float) -> float:
    if decay_s <= 0.0:
        return 0.0
    return math.exp(-1.0 / max(decay_s * sr, 1.0))


def _distance(a: ResonatorString, b: ResonatorString) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _nearest_harmonic_distance_ratio(f_a: float, f_b: float, max_harmonic: int = 16) -> float:
    """Return a small ratio when two strings line up on nearby harmonics."""
    if f_a <= 0.0 or f_b <= 0.0:
        return 1.0
    best = abs(math.log(max(f_a, 1e-9) / max(f_b, 1e-9)))
    for ha in range(1, max_harmonic + 1):
        for hb in range(1, max_harmonic + 1):
            ratio = (f_a * ha) / max(f_b * hb, 1e-9)
            best = min(best, abs(math.log(max(ratio, 1e-9))))
    return best


def build_string_coupling_matrix(
    strings: list[ResonatorString],
    config: StringCouplingConfig | None = None,
) -> np.ndarray:
    """Construct a complex coupling matrix from harmonic and distance affinity."""

    cfg = config or StringCouplingConfig()
    n = len(strings)
    mat = np.zeros((n, n), dtype=np.complex128)
    if n == 0:
        return mat

    for i, dst in enumerate(strings):
        for j, src in enumerate(strings):
            if i == j:
                continue
            harmonic_metric = _nearest_harmonic_distance_ratio(
                float(dst.fundamental_hz),
                float(src.fundamental_hz),
            )
            harmonic_affinity = math.exp(
                -(harmonic_metric / max(cfg.harmonic_sigma, 1e-6)) ** 2
            )
            distance_affinity = math.exp(
                -(_distance(dst, src) / max(cfg.distance_sigma, 1e-6)) ** 2
            )
            strength = cfg.base_strength * (
                cfg.harmonic_strength * harmonic_affinity
                + cfg.distance_strength * distance_affinity
            )
            strength *= float(dst.harmonic_weight) * float(src.harmonic_weight)
            strength = min(cfg.max_coupling, max(0.0, strength))
            # A tiny phase tilt favors near-harmonic bloom instead of flat summation.
            phase = 0.15 * harmonic_metric
            mat[i, j] = strength * complex(math.cos(phase), math.sin(phase))
    return mat
