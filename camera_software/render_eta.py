"""Whole-render progress and ETA derived from native T5 page logging."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Any, Mapping


_T5_PASS_RE = re.compile(r"t5_fired(?:->|→)([0-9]+)")
_CONNECT_RE = re.compile(r"active_pairs=([0-9]+)/([0-9]+)")
_INTENTS_RE = re.compile(r"\bintents=([0-9]+)")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


@dataclass(frozen=True)
class RenderEtaSnapshot:
    bundle_index: int
    bundle_count: int
    page_index: int
    pages_per_bundle: int
    completed_units: float
    total_units: int
    elapsed_s: float
    eta_s: float | None

    @property
    def fraction(self) -> float:
        return min(1.0, max(0.0, self.completed_units / max(1, self.total_units)))

    @property
    def percent(self) -> float:
        return 100.0 * self.fraction

    def message(self) -> str:
        return (
            f"bundle {self.bundle_index}/{self.bundle_count} "
            f"T5 {self.page_index}/{self.pages_per_bundle} | "
            f"overall {self.percent:.2f}% | "
            f"elapsed {format_duration(self.elapsed_s)} | "
            f"ETA {format_duration(self.eta_s)}"
        )


class RenderEtaTracker:
    """Track bundles plus the native T5 pages hidden inside one epoch."""

    # Default active vertex arena and conservative usable path fraction. This
    # predicts the native primary-page split until an actual `intents=` line
    # makes the denominator measured rather than estimated.
    _DEFAULT_VERTEX_CAP = 6_875_000
    _USABLE_NUMERATOR = 25
    _USABLE_DENOMINATOR = 32

    def __init__(
        self,
        runtime: Mapping[str, Any],
        *,
        started_at: float | None = None,
    ) -> None:
        self.runtime = dict(runtime)
        self.bundle_count = max(1, int(self.runtime.get("epoch_bundle_count", 1)))
        self.camera_rays_per_bundle = max(
            1,
            int(self.runtime.get("max_sensor_epochs", 1))
            * int(self.runtime.get("sensor_top_k", 1))
            * int(self.runtime.get("sensor_samples_per_node", 1)),
        )
        bounces = max(1, int(self.runtime.get("max_bounces", 8)))
        estimated_primary_page = max(
            1,
            self._DEFAULT_VERTEX_CAP * self._USABLE_NUMERATOR
            // (self._USABLE_DENOMINATOR * bounces),
        )
        self.primary_rays_per_page = estimated_primary_page
        self.pages_per_bundle = max(
            1, math.ceil(self.camera_rays_per_bundle / estimated_primary_page)
        )
        self.bundle_index = 1
        self.page_index = 0
        self.connect_fraction = 0.0
        self.started_at = time.perf_counter() if started_at is None else float(started_at)
        self._pause_started_at: float | None = None
        self._paused_total_s = 0.0

    def set_paused(self, paused: bool) -> None:
        now = time.perf_counter()
        if paused and self._pause_started_at is None:
            self._pause_started_at = now
        elif not paused and self._pause_started_at is not None:
            self._paused_total_s += now - self._pause_started_at
            self._pause_started_at = None

    def begin_bundle(self, bundle_index: int) -> RenderEtaSnapshot:
        self.bundle_index = max(1, min(self.bundle_count, int(bundle_index)))
        self.page_index = 0
        self.connect_fraction = 0.0
        return self.snapshot()

    def observe(self, line: str) -> tuple[RenderEtaSnapshot | None, bool]:
        text = str(line)
        changed = False
        pass_completed = False
        intent_match = _INTENTS_RE.search(text)
        if intent_match:
            measured = max(1, int(intent_match.group(1)))
            # Diagnostics can sample flash or camera pages. The smaller
            # first-generation page is the limiting camera work quantum.
            self.primary_rays_per_page = min(
                self.primary_rays_per_page, measured
            )
            self.pages_per_bundle = max(
                1,
                math.ceil(
                    self.camera_rays_per_bundle / self.primary_rays_per_page
                ),
            )
            changed = True
        connect_match = _CONNECT_RE.search(text)
        if connect_match:
            completed = int(connect_match.group(1))
            total = max(1, int(connect_match.group(2)))
            self.connect_fraction = min(0.999999, completed / total)
            changed = True
        pass_match = _T5_PASS_RE.search(text)
        if pass_match:
            self.page_index = max(self.page_index, int(pass_match.group(1)))
            self.connect_fraction = 0.0
            changed = True
            pass_completed = True
        return (self.snapshot() if changed else None), pass_completed

    def complete_bundle(self, bundle_index: int) -> RenderEtaSnapshot:
        self.bundle_index = max(1, min(self.bundle_count, int(bundle_index)))
        self.page_index = self.pages_per_bundle
        self.connect_fraction = 0.0
        return self.snapshot()

    def snapshot(self) -> RenderEtaSnapshot:
        now = time.perf_counter()
        current_pause = (
            0.0 if self._pause_started_at is None
            else now - self._pause_started_at
        )
        elapsed = max(
            0.0,
            now - self.started_at - self._paused_total_s - current_pause,
        )
        completed = (
            (self.bundle_index - 1) * self.pages_per_bundle
            + min(self.pages_per_bundle, self.page_index + self.connect_fraction)
        )
        total = self.bundle_count * self.pages_per_bundle
        eta = None
        if completed > 0.0 and completed < total and elapsed > 0.0:
            eta = elapsed * (total - completed) / completed
        elif completed >= total:
            eta = 0.0
        return RenderEtaSnapshot(
            bundle_index=self.bundle_index,
            bundle_count=self.bundle_count,
            page_index=min(self.page_index, self.pages_per_bundle),
            pages_per_bundle=self.pages_per_bundle,
            completed_units=completed,
            total_units=total,
            elapsed_s=elapsed,
            eta_s=eta,
        )


__all__ = [
    "RenderEtaSnapshot", "RenderEtaTracker", "format_duration",
]
