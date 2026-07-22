"""Need-weighted scheduling for resumable spectral render jobs.

The scheduler is transport-neutral: one completed packet may be a recursive
sensor epoch, a spatial BDPT page, or any other checkpointable unit. It chooses
the next logical job after every packet so one exposure cannot monopolize the
physical tracer.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np


SCHEDULER_SCHEMA_VERSION = 1


def _finite_nonnegative(path: str) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        value = np.asarray(np.load(path, allow_pickle=False), np.float64)
    except (OSError, ValueError):
        return None
    if value.size == 0:
        return None
    return np.maximum(np.where(np.isfinite(value), value, 0.0), 0.0)


@dataclass
class RenderPacketJob:
    """Persistent state for one logical exposure."""

    job_id: str
    target_packets: int
    weight: float = 1.0
    completed_packets: int = 0
    deficit: float = 0.0
    dispatch_count: int = 0
    last_dispatch_tick: int = -1
    failures: int = 0
    priority_map_path: str = ""
    sensor_sum_path: str = ""
    exposure_weight_path: str = ""
    linear_accumulation_path: str = ""
    message: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.job_id = str(self.job_id)
        self.target_packets = int(self.target_packets)
        self.completed_packets = int(self.completed_packets)
        self.weight = float(self.weight)
        if not self.job_id:
            raise ValueError("render job id must be non-empty")
        if self.target_packets <= 0:
            raise ValueError("target_packets must be positive")
        if self.completed_packets < 0:
            raise ValueError("completed_packets must be non-negative")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("render job weight must be finite and positive")

    @property
    def remaining_packets(self) -> int:
        return max(0, self.target_packets - self.completed_packets)

    @property
    def done(self) -> bool:
        return self.remaining_packets == 0

    def evidence_need(self) -> float:
        """Return a bounded need estimate from the latest sensor evidence."""

        priority = _finite_nonnegative(self.priority_map_path)
        priority_need = 0.5
        if priority is not None:
            peak = float(np.percentile(priority, 99.0))
            if peak > 0.0:
                priority_need = float(np.clip(
                    np.percentile(priority, 90.0) / peak, 0.0, 1.0
                ))

        weight = _finite_nonnegative(self.exposure_weight_path)
        coverage_need = 1.0
        if weight is not None:
            positive = weight[weight > 0.0]
            uncovered = float(np.count_nonzero(weight <= 0.0)) / float(weight.size)
            if positive.size:
                reference = max(float(np.percentile(positive, 75.0)), 1.0e-12)
                under = float(np.mean(np.clip(1.0 - weight / reference, 0.0, 1.0)))
                coverage_need = max(uncovered, under)

        return float(np.clip(
            0.65 * priority_need + 0.35 * coverage_need, 0.0, 1.0
        ))

    def effective_need(self) -> float:
        """Scheduling weight with a floor that prevents starvation."""

        development_need = self.remaining_packets / float(self.target_packets)
        return self.weight * (
            0.20 + 0.55 * development_need + 0.25 * self.evidence_need()
        )


class NeedWeightedPacketScheduler:
    """Deficit scheduler that re-evaluates scene need after every packet."""

    def __init__(self, jobs: Iterable[RenderPacketJob]) -> None:
        ordered = list(jobs)
        if not ordered:
            raise ValueError("at least one render job is required")
        ids = [job.job_id for job in ordered]
        if len(set(ids)) != len(ids):
            raise ValueError("render job ids must be unique")
        self.jobs = {job.job_id: job for job in ordered}
        self.order = tuple(ids)
        self.tick = 0

    @property
    def done(self) -> bool:
        return all(job.done for job in self.jobs.values())

    def next_job(self) -> RenderPacketJob | None:
        ready = [
            self.jobs[job_id] for job_id in self.order
            if not self.jobs[job_id].done
        ]
        if not ready:
            return None
        issued: dict[str, float] = {}
        for job in ready:
            credit = max(1.0e-9, float(job.effective_need()))
            issued[job.job_id] = credit
            job.deficit += credit
        selected = max(
            ready,
            key=lambda job: (
                job.deficit,
                -job.last_dispatch_tick,
                -self.order.index(job.job_id),
            ),
        )
        selected.deficit -= sum(issued.values())
        selected.dispatch_count += 1
        selected.last_dispatch_tick = self.tick
        self.tick += 1
        return selected

    def complete_packet(
        self,
        job_id: str,
        *,
        packets: int = 1,
        priority_map_path: str = "",
        sensor_sum_path: str = "",
        exposure_weight_path: str = "",
        linear_accumulation_path: str = "",
        message: str = "",
    ) -> RenderPacketJob:
        job = self.jobs[str(job_id)]
        amount = max(1, int(packets))
        job.completed_packets = min(
            job.target_packets, job.completed_packets + amount
        )
        for name, value in (
            ("priority_map_path", priority_map_path),
            ("sensor_sum_path", sensor_sum_path),
            ("exposure_weight_path", exposure_weight_path),
            ("linear_accumulation_path", linear_accumulation_path),
        ):
            if value:
                setattr(job, name, os.path.abspath(value))
        job.message = str(message)
        return job

    def fail_packet(self, job_id: str, message: str) -> RenderPacketJob:
        job = self.jobs[str(job_id)]
        job.failures += 1
        job.message = str(message)
        return job

    def to_payload(self) -> dict:
        return {
            "schema_version": SCHEDULER_SCHEMA_VERSION,
            "tick": self.tick,
            "order": list(self.order),
            "jobs": [asdict(self.jobs[job_id]) for job_id in self.order],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "NeedWeightedPacketScheduler":
        if int(payload.get("schema_version", -1)) != SCHEDULER_SCHEMA_VERSION:
            raise ValueError("unsupported render scheduler schema")
        scheduler = cls(RenderPacketJob(**item) for item in payload["jobs"])
        scheduler.tick = max(0, int(payload.get("tick", 0)))
        return scheduler

    @classmethod
    def load(cls, path: str) -> "NeedWeightedPacketScheduler":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_payload(json.load(handle))

    def save(self, path: str) -> None:
        final_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        temporary = final_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.to_payload(), handle, indent=2, sort_keys=True)
        os.replace(temporary, final_path)
