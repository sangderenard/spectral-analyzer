"""Transport-neutral progress contracts for long-running camera exposures.

The ray tracer owns physical accumulation.  Consumers receive immutable events
and may build previews, mip pyramids, or remote telemetry without changing the
accumulator or its sampling policy.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterable

import numpy as np


PROGRESS_SCHEMA_VERSION = 1
PROGRESS_LINE_PREFIX = "SPECTRAL_EXPOSURE_PROGRESS "


class ExposureProgressKind(str, Enum):
    STARTED = "started"
    PASS_AVAILABLE = "pass_available"
    LAYER_AVAILABLE = "layer_available"
    REGION_AVAILABLE = "region_available"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SensorRegion:
    """Integer sensor-bin bounds; sampling inside the bounds remains continuous."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("sensor region requires non-negative origin and positive extent")


@dataclass(frozen=True)
class ExposureProgressEvent:
    exposure_id: str
    sequence: int
    kind: ExposureProgressKind
    region: SensorRegion
    pass_index: int = 0
    zoom_level: int = 0
    subdivision_level: int = 0
    sensor_node_id: int | None = None
    parent_sensor_node_id: int | None = None
    global_uv_bounds: tuple[float, float, float, float] | None = None
    completed_work: int = 0
    total_work: int = 0
    linear_accumulation_path: str = ""
    sample_count_path: str = ""
    preview_path: str = ""
    priority_map_path: str = ""
    orthographic_reference_path: str = ""
    message: str = ""
    schema_version: int = PROGRESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.exposure_id:
            raise ValueError("exposure_id must be non-empty")
        for name in (
            "sequence", "pass_index", "zoom_level", "subdivision_level",
            "completed_work", "total_work",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.total_work and self.completed_work > self.total_work:
            raise ValueError("completed_work cannot exceed total_work")
        for name in ("sensor_node_id", "parent_sensor_node_id"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative when present")
        if self.global_uv_bounds is not None:
            u0, v0, u1, v1 = map(float, self.global_uv_bounds)
            if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
                raise ValueError("global_uv_bounds must be ordered inside [0, 1]")

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    def to_line(self) -> str:
        return PROGRESS_LINE_PREFIX + json.dumps(
            self.to_payload(), separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "ExposureProgressEvent":
        data = dict(payload)
        version = int(data.get("schema_version", -1))
        if version != PROGRESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported exposure progress schema {version}")
        data["kind"] = ExposureProgressKind(str(data["kind"]))
        data["region"] = SensorRegion(**dict(data["region"]))
        if data.get("global_uv_bounds") is not None:
            data["global_uv_bounds"] = tuple(map(float, data["global_uv_bounds"]))
        return cls(**data)

    @classmethod
    def from_line(cls, line: str) -> "ExposureProgressEvent | None":
        if not str(line).startswith(PROGRESS_LINE_PREFIX):
            return None
        return cls.from_payload(json.loads(str(line)[len(PROGRESS_LINE_PREFIX):]))


ProgressSink = Callable[[ExposureProgressEvent], None]


class ExposureProgressBroker:
    """Thread-safe latest-state broker, independent of UI and renderer threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: ExposureProgressEvent | None = None
        self._layers: dict[tuple[int, int, int | None], ExposureProgressEvent] = {}

    def publish(self, event: ExposureProgressEvent) -> None:
        with self._lock:
            if self._latest is not None and event.exposure_id != self._latest.exposure_id:
                self._layers.clear()
            if self._latest is not None and event.exposure_id == self._latest.exposure_id:
                if event.sequence <= self._latest.sequence:
                    raise ValueError("progress event sequence must increase within an exposure")
            self._latest = event
            if event.kind in (
                ExposureProgressKind.PASS_AVAILABLE,
                ExposureProgressKind.LAYER_AVAILABLE,
                ExposureProgressKind.REGION_AVAILABLE,
                ExposureProgressKind.COMPLETED,
            ):
                self._layers[
                    (event.zoom_level, event.subdivision_level, event.sensor_node_id)
                ] = event

    def reset(self) -> None:
        """Forget the previous exposure before a newly submitted one starts."""
        with self._lock:
            self._latest = None
            self._layers.clear()

    def snapshot(self) -> tuple[
        ExposureProgressEvent | None,
        dict[tuple[int, int, int | None], ExposureProgressEvent],
    ]:
        with self._lock:
            return self._latest, dict(self._layers)


class LinearProgressArtifactWriter:
    """Persist presentation snapshots atomically, then announce availability.

    The artifact is a transport/readback boundary for consumers that cannot
    share the renderer's GPU context. It is never the authoritative exposure
    accumulator and must not participate in transport or refinement decisions.
    """

    def __init__(self, root: str, line_sink: Callable[[str], None] = print,
                 retain_last: int = 0) -> None:
        self.root = os.path.abspath(root)
        self._line_sink = line_sink
        self._retain_last = max(0, int(retain_last))
        self._published_paths: list[tuple[str, str, str]] = []
        os.makedirs(self.root, exist_ok=True)

    def publish(
        self,
        event: ExposureProgressEvent,
        linear_accumulation: np.ndarray,
        sample_count: np.ndarray | None = None,
        priority_map: np.ndarray | None = None,
    ) -> ExposureProgressEvent:
        array = np.asarray(linear_accumulation)
        if array.ndim != 3 or array.shape[2] < 3 or not np.issubdtype(array.dtype, np.floating):
            raise ValueError("linear accumulation must be floating HxWxC sensor data")
        node_suffix = "" if event.sensor_node_id is None else f"_n{event.sensor_node_id:08d}"
        name = (
            f"pass_{event.pass_index:06d}_z{event.zoom_level:02d}_"
            f"s{event.subdivision_level:02d}{node_suffix}.npy"
        )
        final_path = os.path.join(self.root, name)
        temp_path = final_path + ".tmp"
        count_path = ""
        priority_path = ""
        if sample_count is not None:
            counts = np.asarray(sample_count)
            if counts.shape != array.shape[:2] or not np.issubdtype(counts.dtype, np.integer):
                raise ValueError("sample count must be an integer array matching the sensor raster")
            count_name = (
                f"counts_{event.pass_index:06d}_z{event.zoom_level:02d}_"
                f"s{event.subdivision_level:02d}{node_suffix}.npy"
            )
            count_path = os.path.join(self.root, count_name)
            count_temp_path = count_path + ".tmp"
            with open(count_temp_path, "wb") as handle:
                np.save(handle, np.ascontiguousarray(counts), allow_pickle=False)
            os.replace(count_temp_path, count_path)
        if priority_map is not None:
            priority = np.asarray(priority_map)
            if priority.shape != array.shape[:2] or not np.issubdtype(priority.dtype, np.floating):
                raise ValueError("priority map must be a floating array matching the sensor raster")
            priority_name = (
                f"priority_{event.pass_index:06d}_z{event.zoom_level:02d}_"
                f"s{event.subdivision_level:02d}{node_suffix}.npy"
            )
            priority_path = os.path.join(self.root, priority_name)
            priority_temp_path = priority_path + ".tmp"
            with open(priority_temp_path, "wb") as handle:
                np.save(handle, np.ascontiguousarray(priority), allow_pickle=False)
            os.replace(priority_temp_path, priority_path)
        with open(temp_path, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        os.replace(temp_path, final_path)
        announced = ExposureProgressEvent(
            **{
                **event.__dict__,
                "linear_accumulation_path": final_path,
                "sample_count_path": count_path,
                "priority_map_path": priority_path,
            }
        )
        self._line_sink(announced.to_line())
        self._published_paths.append((final_path, count_path, priority_path))
        if self._retain_last > 0:
            while len(self._published_paths) > self._retain_last:
                stale_linear, stale_count, stale_priority = self._published_paths.pop(0)
                for stale_path in (stale_linear, stale_count, stale_priority):
                    if not stale_path:
                        continue
                    try:
                        os.remove(stale_path)
                    except OSError:
                        pass
        return announced


def parse_progress_lines(lines: Iterable[str]) -> Iterable[ExposureProgressEvent]:
    for line in lines:
        event = ExposureProgressEvent.from_line(line.rstrip("\r\n"))
        if event is not None:
            yield event
