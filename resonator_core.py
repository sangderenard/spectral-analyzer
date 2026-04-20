"""Complex-domain resonator core for coupled string/body influence.

The core model keeps every signal in the analytic / complex domain and solves a
small coupled resonator system:

    y_i[n] = decay_i * e^{j*omega_i} * y_i[n-1]
             + drive_gain_i * x_i[n]
             + sum_j coupling[i, j] * y_j[n-1]

where each resonator channel can represent a string, tine, bar, or any narrow
body mode driven by one or more performers.  Coupling coefficients can be
constructed from:

- harmonic proximity between fundamentals
- spatial / physical string distance
- explicit user coefficients

This is intended to become the shared kernel behind:
- performer-local instrument resonators
- grouped section resonators
- room / audience capture stages
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass
class ResonatorSolveResult:
    """Solved resonator signals plus metadata used to build the coupling."""

    output: np.ndarray
    coupling_matrix: np.ndarray
    metadata: dict = field(default_factory=dict)


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


def solve_coupled_strings(
    drive: np.ndarray,
    strings: list[ResonatorString],
    *,
    sample_rate: float,
    coupling_matrix: np.ndarray | None = None,
    coupling_config: StringCouplingConfig | None = None,
) -> ResonatorSolveResult:
    """Solve a coupled string resonator in the complex domain.

    Parameters
    ----------
    drive:
        Complex driver array of shape ``(n_strings, n_samples)``.
    strings:
        Resonator definitions corresponding to the first axis of ``drive``.
    sample_rate:
        Solve sample rate in Hz.
    coupling_matrix:
        Optional precomputed complex coupling matrix.  When omitted, one is
        estimated from harmonic proximity and string spacing.
    coupling_config:
        Used only when ``coupling_matrix`` is omitted.
    """

    drive = np.asarray(drive, dtype=np.complex128)
    if drive.ndim != 2:
        raise ValueError("drive must have shape (n_strings, n_samples)")
    n_strings, n_samples = drive.shape
    if len(strings) != n_strings:
        raise ValueError("len(strings) must match drive.shape[0]")

    coupling = (
        np.asarray(coupling_matrix, dtype=np.complex128)
        if coupling_matrix is not None
        else build_string_coupling_matrix(strings, coupling_config)
    )
    if coupling.shape != (n_strings, n_strings):
        raise ValueError("coupling_matrix must have shape (n_strings, n_strings)")

    out = np.zeros((n_strings, n_samples), dtype=np.complex128)
    prev = np.zeros(n_strings, dtype=np.complex128)
    decay_factors = np.array(
        [
            _safe_decay(float(s.decay_s), float(sample_rate))
            * complex(
                math.cos(2.0 * math.pi * float(s.fundamental_hz) / max(sample_rate, 1.0) + float(s.phase_offset_rad)),
                math.sin(2.0 * math.pi * float(s.fundamental_hz) / max(sample_rate, 1.0) + float(s.phase_offset_rad)),
            )
            for s in strings
        ],
        dtype=np.complex128,
    )
    drive_gains = np.array([float(s.drive_gain) for s in strings], dtype=np.float64)

    for n in range(n_samples):
        coupled = coupling @ prev
        current = decay_factors * prev + drive[:, n] * drive_gains + coupled
        out[:, n] = current
        prev = current

    metadata = {
        "string_keys": [s.key for s in strings],
        "fundamentals_hz": [float(s.fundamental_hz) for s in strings],
    }
    return ResonatorSolveResult(output=out, coupling_matrix=coupling, metadata=metadata)

