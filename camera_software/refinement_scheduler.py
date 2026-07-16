"""Top-k ordering for an unconditionally recursive sensor frontier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .sensor_mipmap import SensorUvBounds, SparseSensorMipmap


def _part1by1(value: int) -> int:
    value &= 0x0000FFFF
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value


def _locality_key(bounds: SensorUvBounds) -> int:
    u = int(round((bounds.u0 + bounds.u1) * 0.5 * 65535.0))
    v = int(round((bounds.v0 + bounds.v1) * 0.5 * 65535.0))
    return _part1by1(u) | (_part1by1(v) << 1)


@dataclass(frozen=True)
class RefinementWork:
    node_id: int
    bounds: SensorUvBounds
    subdivision_level: int
    sample_index: int
    priority: float = 0.0


class RecursiveSensorWorkScheduler:
    """Order a sparse frontier without controlling whether subdivision occurs.

    Completing any nonterminal node always creates and enqueues all nine
    children. At maximum depth the same node is queued for a new independent
    sample epoch. Learned scores only determine top-k removal order.
    """

    def __init__(self, n_bands: int, *, maximum_depth: int) -> None:
        self.mipmap = SparseSensorMipmap(n_bands, maximum_depth=maximum_depth)
        root = self.mipmap.nodes[self.mipmap.root_id]
        self._pending: dict[int, RefinementWork] = {
            root.node_id: RefinementWork(root.node_id, root.bounds, root.level, 0)
        }

    def set_priorities(self, priorities: Mapping[int, float]) -> None:
        for node_id, score in priorities.items():
            if node_id not in self._pending:
                continue
            value = float(score)
            if value < 0.0:
                raise ValueError("work priority must be non-negative")
            previous = self._pending[node_id]
            self._pending[node_id] = RefinementWork(
                node_id=previous.node_id,
                bounds=previous.bounds,
                subdivision_level=previous.subdivision_level,
                sample_index=previous.sample_index,
                priority=value,
            )

    def top_k(self, count: int) -> list[RefinementWork]:
        if count <= 0:
            raise ValueError("top-k count must be positive")
        ordered = sorted(
            self._pending.values(),
            key=lambda work: (-work.priority, _locality_key(work.bounds), work.node_id),
        )[:count]
        for work in ordered:
            del self._pending[work.node_id]
        return ordered

    def next(self) -> RefinementWork | None:
        work = self.top_k(1)
        return work[0] if work else None

    def complete(self, work: RefinementWork) -> tuple[RefinementWork, ...]:
        child_ids = self.mipmap.complete_work(work.node_id)
        if child_ids:
            created = tuple(
                RefinementWork(
                    node_id=child_id,
                    bounds=self.mipmap.nodes[child_id].bounds,
                    subdivision_level=self.mipmap.nodes[child_id].level,
                    sample_index=0,
                )
                for child_id in child_ids
            )
        else:
            node = self.mipmap.nodes[work.node_id]
            created = (RefinementWork(
                node_id=node.node_id,
                bounds=node.bounds,
                subdivision_level=node.level,
                sample_index=work.sample_index + 1,
                priority=work.priority,
            ),)
        self._pending.update((item.node_id, item) for item in created)
        return created

    def __len__(self) -> int:
        return len(self._pending)


# Compatibility name for callers that imported the earlier placeholder.
HierarchicalRefinementScheduler = RecursiveSensorWorkScheduler
