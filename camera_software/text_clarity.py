"""Incremental confidence test for progressively exposed text images."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClarityReport:
    layer_count: int
    drift: float
    edge_energy: float
    noise_rse_p90: float
    has_signal: bool
    accepted: bool


class MonteCarloClarityDiscriminator:
    """Judge cumulative images from the variance of independent layer increments.

    This class does not denoise or alter transport. It keeps Welford moments of
    each newly exposed layer and reports confidence only over the currently
    informative (high-signal) sensor region.
    """

    def __init__(self, *, min_layers: int = 8, max_drift: float = 0.008,
                 max_noise_rse_p90: float = 0.25):
        self.min_layers = max(2, int(min_layers))
        self.max_drift = float(max_drift)
        self.max_noise_rse_p90 = float(max_noise_rse_p90)
        self._previous_cumulative: np.ndarray | None = None
        self._previous_normalized: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._count = 0

    def update(self, cumulative_rgb: np.ndarray) -> ClarityReport:
        rgb = np.asarray(cumulative_rgb, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("clarity input must have shape (height, width, >=3)")
        cumulative = np.mean(np.maximum(rgb[..., :3], 0.0), axis=2)
        increment = (cumulative if self._previous_cumulative is None
                     else cumulative - self._previous_cumulative)
        self._previous_cumulative = cumulative.copy()
        self._count += 1
        if self._mean is None:
            self._mean = increment.copy()
            self._m2 = np.zeros_like(increment)
        else:
            delta = increment - self._mean
            self._mean += delta / float(self._count)
            self._m2 += delta * (increment - self._mean)

        white = float(np.percentile(cumulative, 99.5)) if cumulative.size else 0.0
        has_signal = bool(white > 1.0e-20)
        normalized = (cumulative / white) if has_signal else np.zeros_like(cumulative)
        drift = (float("inf") if self._previous_normalized is None or not has_signal
                 else float(np.mean(np.abs(normalized - self._previous_normalized))))
        self._previous_normalized = normalized.copy()
        edge = (float(np.mean(np.abs(np.diff(normalized, axis=0)))
                      + np.mean(np.abs(np.diff(normalized, axis=1))))
                if min(normalized.shape) > 1 else 0.0)

        noise_rse_p90 = float("inf")
        if self._count >= 2 and self._mean is not None and self._m2 is not None:
            variance = np.maximum(self._m2 / float(self._count - 1), 0.0)
            stderr = np.sqrt(variance / float(self._count))
            signal_floor = max(float(np.percentile(self._mean, 75.0)), 1.0e-12)
            informative = self._mean > signal_floor
            if np.any(informative):
                relative = stderr[informative] / np.maximum(
                    np.abs(self._mean[informative]), signal_floor
                )
                noise_rse_p90 = float(np.percentile(relative, 90.0))

        accepted = bool(
            self._count >= self.min_layers
            and has_signal
            and edge > 1.0e-4
            and drift <= self.max_drift
            and noise_rse_p90 <= self.max_noise_rse_p90
        )
        return ClarityReport(
            layer_count=self._count,
            drift=drift,
            edge_energy=edge,
            noise_rse_p90=noise_rse_p90,
            has_signal=has_signal,
            accepted=accepted,
        )
