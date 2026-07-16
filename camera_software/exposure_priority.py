"""Task-driven tile priority contracts for adaptive spectral exposure.

This module decides *where* another valid sample epoch is valuable.  It never
changes ray generation, spectral PDFs, radiance, or sensor accumulation.  The
small learned model is deliberately an injected scorer so GPU inference can
replace the reference policy without coupling model code to the renderer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .progressive_exposure import SensorRegion


@dataclass(frozen=True)
class TileEvidence:
    """Completed-epoch evidence for one sensor tile."""

    region: SensorRegion
    sample_count: int
    variance: float
    ambiguity: float = 0.0
    requested_priority: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        for name in ("variance", "ambiguity", "requested_priority"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


class LearnedTilePriority(Protocol):
    """Replaceable scorer; production implementations may run on the GPU."""

    def score(self, evidence: Sequence[TileEvidence]) -> np.ndarray:
        """Return one finite, non-negative marginal-value score per tile."""


@dataclass(frozen=True)
class PriorityPolicyConfig:
    uncertainty_weight: float = 1.0
    ambiguity_weight: float = 1.0
    learned_weight: float = 1.0
    requested_weight: float = 4.0
    firm_request: float = 0.8
    priority_quantum: float = 0.05

    def __post_init__(self) -> None:
        if self.priority_quantum <= 0.0:
            raise ValueError("priority_quantum must be positive")
        if not 0.0 <= self.firm_request <= 1.0:
            raise ValueError("firm_request must be in [0, 1]")
        for name in (
            "uncertainty_weight", "ambiguity_weight",
            "learned_weight", "requested_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PrioritizedTile:
    evidence: TileEvidence
    priority: float
    learned_priority: float
    standard_error: float
    locality_key: int


def _part1by1(value: int) -> int:
    value &= 0x0000FFFF
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value


def sensor_region_morton_key(region: SensorRegion) -> int:
    """Morton key of a region centre, used only as a locality tie-break."""

    x = int(region.x + region.width // 2)
    y = int(region.y + region.height // 2)
    return _part1by1(x) | (_part1by1(y) << 1)


class LocalityAwarePriorityPolicy:
    """Combine confidence, task ambiguity, learned value, ROI, and locality.

    Priority is authoritative across score bands.  Morton order is consulted
    only inside a small quantized band, so locality cannot suppress a firmly
    requested or clearly ambiguous remote tile.
    """

    def __init__(
        self,
        config: PriorityPolicyConfig | None = None,
        learned_model: LearnedTilePriority | None = None,
    ) -> None:
        self.config = config or PriorityPolicyConfig()
        self.learned_model = learned_model

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        peak = float(np.max(values)) if values.size else 0.0
        return values / peak if peak > 0.0 else np.zeros_like(values)

    def rank(self, evidence: Sequence[TileEvidence]) -> list[PrioritizedTile]:
        items = tuple(evidence)
        if not items:
            return []
        cfg = self.config
        learned = (
            np.asarray(self.learned_model.score(items), dtype=np.float64)
            if self.learned_model is not None else np.zeros(len(items), np.float64)
        )
        if learned.shape != (len(items),):
            raise ValueError("learned priority model must return one score per tile")
        if not np.all(np.isfinite(learned)) or np.any(learned < 0.0):
            raise ValueError("learned priority scores must be finite and non-negative")

        uncertainty = np.asarray([
            math.sqrt(item.variance / max(1, item.sample_count)) for item in items
        ], dtype=np.float64)
        ambiguity = self._normalise(np.asarray([item.ambiguity for item in items]))
        learned_n = self._normalise(learned)
        uncertainty_n = self._normalise(uncertainty)

        ranked: list[PrioritizedTile] = []
        for index, item in enumerate(items):
            priority = (
                cfg.uncertainty_weight * float(uncertainty_n[index])
                + cfg.ambiguity_weight * float(ambiguity[index])
                + cfg.learned_weight * float(learned_n[index])
                + cfg.requested_weight * item.requested_priority
            )
            ranked.append(PrioritizedTile(
                evidence=item,
                priority=float(priority),
                learned_priority=float(learned[index]),
                standard_error=float(uncertainty[index]),
                locality_key=sensor_region_morton_key(item.region),
            ))

        quantum = cfg.priority_quantum
        return sorted(
            ranked,
            key=lambda tile: (
                0 if tile.evidence.requested_priority >= cfg.firm_request else 1,
                -int(math.floor(tile.priority / quantum)),
                tile.locality_key,
            ),
        )
