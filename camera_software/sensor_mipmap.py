"""Reference ABI for a sparse, recursively refined spectral sensor mipmap.

The production implementation belongs on the GPU.  This reference model makes
the storage and rollup invariants executable before shaders depend on them:

* sampling and subdivision are independent decisions;
* samples retain global sensor UV and explicit node lineage;
* direct observations never share storage with descendant-derived rollups;
* rollups average child estimates by physical area, never by sample count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SensorUvBounds:
    u0: float
    v0: float
    u1: float
    v1: float

    def __post_init__(self) -> None:
        values = (self.u0, self.v0, self.u1, self.v1)
        if not all(np.isfinite(values)):
            raise ValueError("sensor UV bounds must be finite")
        if not (0.0 <= self.u0 < self.u1 <= 1.0 and 0.0 <= self.v0 < self.v1 <= 1.0):
            raise ValueError("sensor UV bounds must be ordered inside [0, 1]")

    @property
    def area(self) -> float:
        return (self.u1 - self.u0) * (self.v1 - self.v0)

    def contains(self, u: float, v: float) -> bool:
        # The upper edge is inclusive only at the physical sensor boundary.
        u_ok = self.u0 <= u < self.u1 or (self.u1 == 1.0 and u == 1.0)
        v_ok = self.v0 <= v < self.v1 or (self.v1 == 1.0 and v == 1.0)
        return bool(u_ok and v_ok)

    def subdivide_3x3(self) -> tuple["SensorUvBounds", ...]:
        du = (self.u1 - self.u0) / 3.0
        dv = (self.v1 - self.v0) / 3.0
        children = []
        for row in range(3):
            for column in range(3):
                u0 = self.u0 + column * du
                v0 = self.v0 + row * dv
                u1 = self.u1 if column == 2 else self.u0 + (column + 1) * du
                v1 = self.v1 if row == 2 else self.v0 + (row + 1) * dv
                children.append(SensorUvBounds(u0, v0, u1, v1))
        return tuple(children)


@dataclass
class WeightedSpectralMoments:
    n_bands: int
    sum: np.ndarray = field(init=False)
    sum_sq: np.ndarray = field(init=False)
    weight: float = 0.0
    weight_sq: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if self.n_bands <= 0:
            raise ValueError("n_bands must be positive")
        self.sum = np.zeros(self.n_bands, dtype=np.float64)
        self.sum_sq = np.zeros(self.n_bands, dtype=np.float64)

    @property
    def valid(self) -> bool:
        return self.weight > 0.0

    @property
    def mean(self) -> np.ndarray:
        if not self.valid:
            return np.zeros(self.n_bands, dtype=np.float64)
        return self.sum / self.weight

    @property
    def effective_count(self) -> float:
        if self.weight_sq <= 0.0:
            return 0.0
        return self.weight * self.weight / self.weight_sq

    def add(self, spectrum: np.ndarray, weight: float = 1.0) -> None:
        value = np.asarray(spectrum, dtype=np.float64)
        if value.shape != (self.n_bands,) or not np.all(np.isfinite(value)):
            raise ValueError("spectrum must be one finite value per band")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("sample weight must be finite and positive")
        self.sum += weight * value
        self.sum_sq += weight * value * value
        self.weight += float(weight)
        self.weight_sq += float(weight * weight)
        self.count += 1


@dataclass(frozen=True)
class SpectralEstimate:
    mean: np.ndarray
    evidence: float
    valid: bool


@dataclass(frozen=True)
class SensorMipSample:
    node_id: int
    global_uv: tuple[float, float]
    spectrum: np.ndarray
    weight: float


@dataclass
class SensorMipNode:
    node_id: int
    parent_id: int | None
    child_slot: int | None
    level: int
    bounds: SensorUvBounds
    direct: WeightedSpectralMoments
    children: tuple[int, ...] = ()
    rolled: SpectralEstimate | None = None
    completed_epochs: int = 0

    @property
    def subdivided(self) -> bool:
        return bool(self.children)


class SparseSensorMipmap:
    """Executable reference for the GPU sparse 9-way sensor hierarchy."""

    def __init__(self, n_bands: int, *, maximum_depth: int) -> None:
        if maximum_depth < 0:
            raise ValueError("maximum_depth must be non-negative")
        self.n_bands = int(n_bands)
        self.maximum_depth = int(maximum_depth)
        self.nodes: dict[int, SensorMipNode] = {}
        self.samples: list[SensorMipSample] = []
        self._next_node_id = 0
        self.root_id = self._allocate(None, None, 0, SensorUvBounds(0.0, 0.0, 1.0, 1.0))

    def _allocate(
        self,
        parent_id: int | None,
        child_slot: int | None,
        level: int,
        bounds: SensorUvBounds,
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes[node_id] = SensorMipNode(
            node_id=node_id,
            parent_id=parent_id,
            child_slot=child_slot,
            level=level,
            bounds=bounds,
            direct=WeightedSpectralMoments(self.n_bands),
        )
        return node_id

    def complete_work(self, node_id: int, *, subdivide: bool = False) -> tuple[int, ...]:
        """Complete one node epoch and optionally materialize its 3x3 children."""
        node = self.nodes[node_id]
        node.completed_epochs += 1
        if not subdivide or node.level >= self.maximum_depth:
            return ()
        if node.children:
            return node.children
        node.children = tuple(
            self._allocate(node_id, slot, node.level + 1, bounds)
            for slot, bounds in enumerate(node.bounds.subdivide_3x3())
        )
        return node.children

    def leaf_at_uv(self, u: float, v: float) -> SensorMipNode:
        """Return the deepest currently materialized node containing a UV."""
        u, v = float(u), float(v)
        node = self.nodes[self.root_id]
        if not node.bounds.contains(u, v):
            raise ValueError("sensor UV lies outside [0, 1]")
        while node.children:
            node = next(
                self.nodes[child_id]
                for child_id in node.children
                if self.nodes[child_id].bounds.contains(u, v)
            )
        return node

    def refine_uv(self, u: float, v: float, *, target_level: int) -> SensorMipNode:
        """Descend only the lineage needed to distinguish one continuous UV."""
        if target_level < 0 or target_level > self.maximum_depth:
            raise ValueError("target_level lies outside the configured hierarchy")
        node = self.leaf_at_uv(u, v)
        while node.level < target_level:
            children = self.complete_work(node.node_id, subdivide=True)
            node = next(
                self.nodes[child_id]
                for child_id in children
                if self.nodes[child_id].bounds.contains(float(u), float(v))
            )
        return node

    def add_sample(
        self,
        node_id: int,
        global_uv: tuple[float, float],
        spectrum: np.ndarray,
        *,
        weight: float = 1.0,
    ) -> None:
        node = self.nodes[node_id]
        u, v = map(float, global_uv)
        if not node.bounds.contains(u, v):
            raise ValueError("sample UV lies outside its scheduled sensor node")
        node.direct.add(spectrum, weight)
        self.samples.append(SensorMipSample(
            node_id=node_id,
            global_uv=(u, v),
            spectrum=np.asarray(spectrum, dtype=np.float64).copy(),
            weight=float(weight),
        ))
        self._recompute_ancestors(node_id)

    def direct_estimate(self, node_id: int) -> SpectralEstimate:
        moments = self.nodes[node_id].direct
        return SpectralEstimate(
            mean=moments.mean.copy(),
            evidence=moments.effective_count,
            valid=moments.valid,
        )

    def resolved_estimate(self, node_id: int) -> SpectralEstimate:
        """Fuse retained direct evidence with an independent complete rollup."""
        node = self.nodes[node_id]
        direct = self.direct_estimate(node_id)
        rolled = node.rolled
        if rolled is None or not rolled.valid:
            return direct
        if not direct.valid:
            return rolled
        total = direct.evidence + rolled.evidence
        if total <= 0.0:
            return direct
        return SpectralEstimate(
            mean=(direct.mean * direct.evidence + rolled.mean * rolled.evidence) / total,
            evidence=total,
            valid=True,
        )

    def _recompute_ancestors(self, node_id: int) -> None:
        parent_id = self.nodes[node_id].parent_id
        while parent_id is not None:
            parent = self.nodes[parent_id]
            estimates = [self.resolved_estimate(child_id) for child_id in parent.children]
            if len(estimates) == 9 and all(estimate.valid for estimate in estimates):
                areas = np.asarray([
                    self.nodes[child_id].bounds.area / parent.bounds.area
                    for child_id in parent.children
                ], dtype=np.float64)
                mean = sum(
                    area * estimate.mean for area, estimate in zip(areas, estimates)
                )
                # Evidence is limited by the least-resolved stratum: repeatedly
                # sampling one child cannot claim complete parent confidence.
                evidence = min(estimate.evidence for estimate in estimates)
                parent.rolled = SpectralEstimate(mean=mean, evidence=evidence, valid=True)
            else:
                parent.rolled = None
            parent_id = parent.parent_id
