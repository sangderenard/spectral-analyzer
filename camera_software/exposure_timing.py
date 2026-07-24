"""Deterministic simulated-time exposure slicing for camera software.

This module is intentionally small: it gives camera code a stable frame/slice
identity, authored shutter/emitter/sensor states, and exact per-slice
completion accounting without borrowing time from the display loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
)


_UINT64_MAX = (1 << 64) - 1
_ResultT = TypeVar("_ResultT")
CAMERA_SLICE_JOB_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class CameraCausalityKey:
    """Portable identity for one camera-authored unit of physical time.

    ``causality_id`` is monotonic within the camera namespace.  Program
    generation is separate so a reset/reprogram can invalidate old work
    without pretending that an integer counter is itself simulated time.
    """

    camera_id: str
    program_generation: int
    causality_id: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera causality keys require a non-empty camera_id")
        for name, value in (
            ("program_generation", self.program_generation),
            ("causality_id", self.causality_id),
        ):
            if not 0 <= int(value) <= _UINT64_MAX:
                raise ValueError(f"{name} must fit an unsigned 64-bit integer")

    def to_wire(self) -> Dict[str, str | int]:
        return {
            "camera_id": self.camera_id,
            "program_generation": int(self.program_generation),
            "causality_id": int(self.causality_id),
        }

    @staticmethod
    def from_wire(value: Mapping[str, Any]) -> "CameraCausalityKey":
        return CameraCausalityKey(
            camera_id=str(value["camera_id"]),
            program_generation=int(value["program_generation"]),
            causality_id=int(value["causality_id"]),
        )


def _clamp01(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except Exception:
        f = float(default)
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _shutter_mode_code(mode: Any) -> int:
    mode_s = str(mode or "open").lower()
    mode_map = {
        "open": 0,
        "closed": 1,
        "cap": 1,
        "lens_cap": 1,
        "iris": 2,
        "sliding_x": 3,
        "sliding_y": 4,
    }
    return int(mode_map.get(mode_s, 0))


@dataclass(frozen=True)
class ExposureSlice:
    """One deterministic camera exposure interval.

    The slice describes simulated time and the camera program state for both
    sensor sweep and mounted-light emission. Transport completion barriers will
    later key off this identity instead of loose global dispatch counters.
    """

    frame_id: int
    slice_id: int
    t0: float
    t1: float
    shutter_mode: int
    shutter_open: float
    shutter_center_u: float
    shutter_center_v: float
    shutter_softness: float
    exposure_weight: float
    flash_weight: float
    sensor_weight: float
    profile: str = "steady"
    scene_version_id: Optional[int] = None
    flash_active: bool = True
    sensor_integrating: bool = True
    camera_id: str = "camera"
    program_generation: int = 0
    causality_id: int = 0

    @property
    def dt(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))

    @property
    def causality_key(self) -> CameraCausalityKey:
        return CameraCausalityKey(
            camera_id=str(self.camera_id),
            program_generation=int(self.program_generation),
            causality_id=int(self.causality_id),
        )

    def sensor_submit_args(self) -> Dict[str, float | int]:
        """Arguments accepted by ray_pipeline_submit_sensor_sweep."""
        return {
            "shutter_mode": int(self.shutter_mode),
            "shutter_open": float(self.shutter_open),
            "shutter_center_u": float(self.shutter_center_u),
            "shutter_center_v": float(self.shutter_center_v),
            "shutter_softness": float(self.shutter_softness),
            "exposure_weight": float(self.sensor_weight),
        }


@dataclass(frozen=True)
class ExposureFrame:
    """Camera-owned simulated-time exposure frame."""

    frame_id: int
    t0: float
    t1: float
    slices: tuple[ExposureSlice, ...] = ()
    camera_id: str = "camera"
    program_generation: int = 0

    @property
    def dt(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))

    @property
    def causality_keys(self) -> tuple[CameraCausalityKey, ...]:
        return tuple(item.causality_key for item in self.slices)


class CameraExposureScheduler:
    """Builds and owns deterministic exposure slices for the active frame."""

    def __init__(
        self,
        *,
        camera_id: str = "camera",
        program_generation: int = 0,
        next_causality_id: int = 0,
    ) -> None:
        if not camera_id:
            raise ValueError("camera_id must be non-empty")
        if not 0 <= int(program_generation) <= _UINT64_MAX:
            raise ValueError("program_generation must fit uint64")
        if not 0 <= int(next_causality_id) <= _UINT64_MAX:
            raise ValueError("next_causality_id must fit uint64")
        self.camera_id = str(camera_id)
        self.program_generation = int(program_generation)
        self.next_causality_id = int(next_causality_id)
        self.scene_time_s: float = 0.0
        self.next_frame_id: int = 0
        self.active_frame: Optional[ExposureFrame] = None
        self._authoring_lock = threading.RLock()

    def begin_frame(self,
                    scene: Any,
                    dt_s: Optional[float] = None,
                    *,
                    t0_s: Optional[float] = None) -> ExposureFrame:
        """Start a new exposure frame from the current camera/scene settings."""
        with self._authoring_lock:
            if t0_s is not None:
                self.scene_time_s = float(t0_s)
            frame = self._build_frame(scene, dt_s)
            self.active_frame = frame
            self.next_frame_id = int(frame.frame_id) + 1
            return frame

    def ensure_frame(self,
                     scene: Any,
                     dt_s: Optional[float] = None,
                     *,
                     t0_s: Optional[float] = None) -> ExposureFrame:
        if self.active_frame is None:
            return self.begin_frame(scene, dt_s, t0_s=t0_s)
        return self.active_frame

    def finish_frame(self) -> None:
        """Advance simulated time to the end of the active frame."""
        with self._authoring_lock:
            if self.active_frame is not None:
                self.scene_time_s = max(
                    self.scene_time_s, float(self.active_frame.t1)
                )
            self.active_frame = None

    def reset(self, scene_time_s: float = 0.0, *, new_program: bool = True) -> None:
        """Reset camera time without ever reusing a causality id.

        By default this begins a new program generation, so results still in
        flight from the old program are unambiguously stale.
        """

        with self._authoring_lock:
            if new_program:
                if self.program_generation >= _UINT64_MAX:
                    raise OverflowError("camera program generation exhausted uint64")
                self.program_generation += 1
            self.scene_time_s = float(scene_time_s)
            self.next_frame_id = 0
            self.active_frame = None

    def _allocate_causality_id(self) -> int:
        if self.next_causality_id > _UINT64_MAX:
            raise OverflowError("camera causality id exhausted uint64")
        result = self.next_causality_id
        self.next_causality_id += 1
        return result

    def _build_frame(self, scene: Any, dt_s: Optional[float]) -> ExposureFrame:
        burst = getattr(scene, "camera_light_burst", None)
        plate = getattr(scene, "image_plate", None)

        stages = int(max(1, getattr(burst, "stages", 1)))
        exposure_time = float(getattr(burst, "exposure_time_s", 0.010))
        # burst.exposure_time_s is authoritative — the scene-coordinator's dt_s
        # (driven by the display loop) must NOT override the camera program.
        # dt_s is only a fallback for when no burst is configured or its
        # exposure_time_s is zero/negative.
        if exposure_time <= 0.0 and dt_s is not None:
            exposure_time = float(max(0.0, dt_s))
        if exposure_time <= 0.0:
            exposure_time = 0.010

        duty = _clamp01(getattr(burst, "duty_cycle", 1.0), 1.0)
        energy = max(0.0, float(getattr(burst, "energy_scale", 1.0)))
        enabled = bool(getattr(burst, "enabled", True))
        profile = str(getattr(burst, "profile", "steady") or "steady")
        active_flash_slices = (
            min(stages, max(1, int(math.ceil(stages * duty))))
            if enabled and energy > 0.0 and duty > 0.0
            else 0
        )
        total_flash_weight = energy * duty if active_flash_slices else 0.0
        active_flash_weight = (
            total_flash_weight / float(active_flash_slices)
            if active_flash_slices
            else 0.0
        )

        mode = _shutter_mode_code(getattr(plate, "shutter_mode", "open"))
        open_f = _clamp01(getattr(plate, "shutter_open", 1.0), 1.0)
        sensor_integrating = mode != 1 and open_f > 0.0
        sensor_weight = (1.0 / float(stages)) if sensor_integrating else 0.0
        cu_base = _clamp01(getattr(plate, "shutter_center_u", 0.5), 0.5)
        cv_base = _clamp01(getattr(plate, "shutter_center_v", 0.5), 0.5)
        softness = max(0.0, float(getattr(plate, "shutter_softness", 0.0)))

        frame_id = int(self.next_frame_id)
        t0 = float(self.scene_time_s)
        slice_dt = exposure_time / float(stages)
        slices: List[ExposureSlice] = []
        for slice_id in range(stages):
            flash_active = slice_id < active_flash_slices
            cu = cu_base
            cv = cv_base
            if stages > 1 and mode == 3:
                half = 0.5 * open_f
                lo = half
                hi = 1.0 - half
                phase = (float(slice_id) + 0.5) / float(stages)
                cu = float(lo + (hi - lo) * phase)
            elif stages > 1 and mode == 4:
                half = 0.5 * open_f
                lo = half
                hi = 1.0 - half
                phase = (float(slice_id) + 0.5) / float(stages)
                cv = float(lo + (hi - lo) * phase)

            s0 = t0 + slice_dt * float(slice_id)
            s1 = t0 + slice_dt * float(slice_id + 1)
            slices.append(ExposureSlice(
                frame_id=frame_id,
                slice_id=int(slice_id),
                t0=s0,
                t1=s1,
                shutter_mode=mode,
                shutter_open=open_f,
                shutter_center_u=cu,
                shutter_center_v=cv,
                shutter_softness=softness,
                exposure_weight=sensor_weight,
                flash_weight=active_flash_weight if flash_active else 0.0,
                sensor_weight=sensor_weight,
                profile=profile,
                flash_active=flash_active,
                sensor_integrating=sensor_integrating,
                camera_id=self.camera_id,
                program_generation=self.program_generation,
                causality_id=self._allocate_causality_id(),
            ))

        return ExposureFrame(
            frame_id=frame_id,
            t0=t0,
            t1=t0 + exposure_time,
            slices=tuple(slices),
            camera_id=self.camera_id,
            program_generation=self.program_generation,
        )


@dataclass(frozen=True)
class CameraSliceJob:
    """Order-independent work envelope suitable for a local or remote worker.

    The scene itself may live in a content-addressed cache; ``scene_version_id``
    names the exact state a worker must resolve before executing this slice.
    """

    exposure_slice: ExposureSlice
    scene_version_id: Optional[int] = None
    solver_revision: str = ""

    def __post_init__(self) -> None:
        embedded = self.exposure_slice.scene_version_id
        if (
            embedded is not None
            and self.scene_version_id is not None
            and int(embedded) != int(self.scene_version_id)
        ):
            raise ValueError(
                "job scene version conflicts with its exposure slice"
            )

    @property
    def key(self) -> CameraCausalityKey:
        return self.exposure_slice.causality_key

    @staticmethod
    def from_slice(
        exposure_slice: ExposureSlice, *, solver_revision: str = ""
    ) -> "CameraSliceJob":
        return CameraSliceJob(
            exposure_slice=exposure_slice,
            scene_version_id=exposure_slice.scene_version_id,
            solver_revision=str(solver_revision),
        )

    def to_wire(self) -> Dict[str, Any]:
        """Return a canonical JSON-compatible network job descriptor."""

        return {
            "schema_version": CAMERA_SLICE_JOB_SCHEMA_VERSION,
            "exposure_slice": asdict(self.exposure_slice),
            "scene_version_id": self.scene_version_id,
            "solver_revision": self.solver_revision,
        }

    @staticmethod
    def from_wire(value: Mapping[str, Any]) -> "CameraSliceJob":
        version = int(value.get("schema_version", -1))
        if version != CAMERA_SLICE_JOB_SCHEMA_VERSION:
            raise ValueError(f"unsupported camera slice job schema {version}")
        slice_value = value.get("exposure_slice")
        if not isinstance(slice_value, Mapping):
            raise ValueError("camera slice job is missing exposure_slice")
        scene_version = value.get("scene_version_id")
        return CameraSliceJob(
            exposure_slice=ExposureSlice(**dict(slice_value)),
            scene_version_id=(
                None if scene_version is None else int(scene_version)
            ),
            solver_revision=str(value.get("solver_revision", "")),
        )

    @property
    def descriptor_digest(self) -> str:
        """SHA-256 of the canonical work descriptor, excluding scene payload."""

        encoded = json.dumps(
            self.to_wire(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CameraSliceResult(Generic[_ResultT]):
    """One attempt's immutable, content-identified result for a slice job."""

    key: CameraCausalityKey
    payload_digest: str
    payload: _ResultT
    job_digest: str = ""
    worker_id: str = ""
    attempt_id: str = ""

    def __post_init__(self) -> None:
        if not self.payload_digest:
            raise ValueError("slice results require a content digest")


class CameraResultCollector(Generic[_ResultT]):
    """Accept arbitrary completion order and release canonical camera order.

    Retried network jobs are idempotent when their digest matches. A conflicting
    result for the same causality key is rejected instead of silently changing
    the exposure. In the ordinary one-worker ordered path every submit releases
    immediately and the pending map never grows beyond one entry.
    """

    def __init__(
        self,
        expected: (
            ExposureFrame
            | Iterable[CameraCausalityKey]
            | Iterable[CameraSliceJob]
        ),
        *,
        max_pending: Optional[int] = None,
        on_commit: Optional[
            Callable[[CameraSliceResult[_ResultT]], None]
        ] = None,
    ) -> None:
        expected_job_digests: Dict[CameraCausalityKey, str] = {}
        if isinstance(expected, ExposureFrame):
            keys = expected.causality_keys
        else:
            expected_items = tuple(expected)
            if all(isinstance(item, CameraSliceJob) for item in expected_items):
                jobs = tuple(expected_items)
                keys = tuple(job.key for job in jobs)
                expected_job_digests = {
                    job.key: job.descriptor_digest for job in jobs
                }
            elif all(
                isinstance(item, CameraCausalityKey)
                for item in expected_items
            ):
                keys = tuple(expected_items)
            else:
                raise TypeError(
                    "expected must contain only causality keys or slice jobs"
                )
        if len(set(keys)) != len(keys):
            raise ValueError("expected camera causality keys must be unique")
        if max_pending is not None and int(max_pending) < 1:
            raise ValueError("max_pending must be positive")
        self._expected = tuple(keys)
        self._expected_set = set(keys)
        self._expected_job_digests = expected_job_digests
        self._max_pending = (
            None if max_pending is None else int(max_pending)
        )
        self._on_commit = on_commit
        self._next_commit = 0
        self._pending: Dict[
            CameraCausalityKey, CameraSliceResult[_ResultT]
        ] = {}
        self._accepted_digests: Dict[CameraCausalityKey, str] = {}
        self._lock = threading.Lock()

    @property
    def complete(self) -> bool:
        with self._lock:
            return self._next_commit == len(self._expected)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def committed_count(self) -> int:
        with self._lock:
            return self._next_commit

    def submit(
        self, result: CameraSliceResult[_ResultT]
    ) -> tuple[CameraSliceResult[_ResultT], ...]:
        """Accept one result and return the newly contiguous commit sequence."""

        key = result.key
        with self._lock:
            if key not in self._expected_set:
                raise ValueError("result does not belong to this camera exposure")
            expected_job_digest = self._expected_job_digests.get(key)
            if (
                expected_job_digest is not None
                and result.job_digest != expected_job_digest
            ):
                raise ValueError(
                    "result was not produced from the authored slice job"
                )
            prior_digest = self._accepted_digests.get(key)
            if prior_digest is not None:
                if prior_digest != result.payload_digest:
                    raise RuntimeError(
                        "conflicting results received for one causality key"
                    )
                return ()
            if (
                self._max_pending is not None
                and len(self._pending) >= self._max_pending
            ):
                raise BufferError("camera result reorder window is full")

            self._pending[key] = result
            self._accepted_digests[key] = result.payload_digest
            ready: List[CameraSliceResult[_ResultT]] = []
            while self._next_commit < len(self._expected):
                next_key = self._expected[self._next_commit]
                next_result = self._pending.get(next_key)
                if next_result is None:
                    break
                # Run the optional reducer while holding the collector's
                # ordering lock. A concurrent submit can therefore never make
                # detector commits observable in a different order.
                if self._on_commit is not None:
                    self._on_commit(next_result)
                self._pending.pop(next_key)
                ready.append(next_result)
                self._next_commit += 1
            return tuple(ready)


class CausalMultireaderQueue(Generic[_ResultT]):
    """Retain each committed slice until every registered system releases it.

    Registration is prospective: a newly registered reader participates in
    slices published after registration, but cannot retroactively hold a slice
    it never observed. Readers may acknowledge different slices in any order;
    retirement (and therefore camera/dt advance) remains a contiguous prefix.

    Connect this to :class:`CameraResultCollector` with
    ``on_commit=queue.publish``. This deliberately separates unordered solver
    completion from the stronger statement that every dependent system has
    consumed the resulting causal state.
    """

    def __init__(
        self,
        expected: ExposureFrame | Iterable[CameraCausalityKey],
        *,
        on_retire: Optional[
            Callable[[CameraSliceResult[_ResultT]], None]
        ] = None,
    ) -> None:
        keys = (
            expected.causality_keys
            if isinstance(expected, ExposureFrame)
            else tuple(expected)
        )
        if len(set(keys)) != len(keys):
            raise ValueError("expected camera causality keys must be unique")
        self._expected = tuple(keys)
        self._published_count = 0
        self._retired_count = 0
        self._readers: set[str] = set()
        self._entries: Dict[
            CameraCausalityKey, CameraSliceResult[_ResultT]
        ] = {}
        self._holds: Dict[CameraCausalityKey, set[str]] = {}
        self._on_retire = on_retire
        self._lock = threading.Lock()

    @property
    def published_count(self) -> int:
        with self._lock:
            return self._published_count

    @property
    def retired_count(self) -> int:
        with self._lock:
            return self._retired_count

    @property
    def advance_complete(self) -> bool:
        with self._lock:
            return self._retired_count == len(self._expected)

    def is_retired(self, key: CameraCausalityKey) -> bool:
        """Return whether ``key`` belongs to the contiguous retired prefix."""

        with self._lock:
            try:
                index = self._expected.index(key)
            except ValueError as exc:
                raise ValueError(
                    "causal key does not belong to this queue"
                ) from exc
            return index < self._retired_count

    def register(self, reader_id: str) -> None:
        reader = str(reader_id)
        if not reader:
            raise ValueError("reader_id must be non-empty")
        with self._lock:
            if reader in self._readers:
                raise ValueError("causal reader is already registered")
            self._readers.add(reader)

    def unregister(self, reader_id: str, *, release_holds: bool = False) -> None:
        """Remove a reader, requiring an explicit policy for outstanding holds."""

        reader = str(reader_id)
        with self._lock:
            if reader not in self._readers:
                raise ValueError("causal reader is not registered")
            held_keys = tuple(
                key for key, readers in self._holds.items()
                if reader in readers
            )
            if held_keys and not release_holds:
                raise RuntimeError(
                    "reader still holds causal slices; acknowledge them or "
                    "explicitly release_holds"
                )
            self._readers.remove(reader)
            if release_holds:
                for key in held_keys:
                    self._holds[key].discard(reader)
                self._retire_ready_locked()

    def publish(self, result: CameraSliceResult[_ResultT]) -> None:
        """Publish exactly the next camera-ordered result to all current readers."""

        with self._lock:
            if self._published_count >= len(self._expected):
                raise ValueError("all expected causal slices are already published")
            expected_key = self._expected[self._published_count]
            if result.key != expected_key:
                raise ValueError("causal queue publication must follow camera order")
            self._entries[result.key] = result
            self._holds[result.key] = set(self._readers)
            self._published_count += 1
            self._retire_ready_locked()

    def pending_for(
        self, reader_id: str
    ) -> tuple[CameraSliceResult[_ResultT], ...]:
        """Return retained published results this reader has not acknowledged."""

        reader = str(reader_id)
        with self._lock:
            if reader not in self._readers:
                raise ValueError("causal reader is not registered")
            return tuple(
                self._entries[key]
                for key in self._expected[
                    self._retired_count:self._published_count
                ]
                if reader in self._holds.get(key, ())
            )

    def acknowledge(
        self, reader_id: str, key: CameraCausalityKey
    ) -> tuple[CameraSliceResult[_ResultT], ...]:
        """Release one reader hold and return newly retired prefix entries."""

        reader = str(reader_id)
        with self._lock:
            if reader not in self._readers:
                raise ValueError("causal reader is not registered")
            if key not in self._entries:
                if key in self._expected[:self._retired_count]:
                    return ()
                raise ValueError("causal slice has not been published")
            holds = self._holds[key]
            if reader not in holds:
                return ()
            holds.remove(reader)
            return self._retire_ready_locked()

    def _retire_ready_locked(
        self,
    ) -> tuple[CameraSliceResult[_ResultT], ...]:
        retired: List[CameraSliceResult[_ResultT]] = []
        while self._retired_count < self._published_count:
            key = self._expected[self._retired_count]
            if self._holds.get(key):
                break
            result = self._entries[key]
            if self._on_retire is not None:
                self._on_retire(result)
            retired.append(result)
            self._entries.pop(key)
            self._holds.pop(key)
            self._retired_count += 1
        return tuple(retired)


@dataclass(frozen=True)
class SceneSnapshot:
    """Scene frame delivered to the camera for one simulated interval."""

    snapshot_id: int
    t0: float
    t1: float
    sample_time: float
    scene: Any
    exposure_frame: Optional[ExposureFrame] = None
    delta_metric: float = 0.0
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))


class SceneFrameProvider:
    """Supplies scene snapshots cut to camera/coordinator time intervals."""

    @property
    def current_scene(self) -> Any:
        raise NotImplementedError

    def snapshot(self,
                 t0: float,
                 t1: float,
                 sample_time: float,
                 *,
                 exposure_frame: Optional[ExposureFrame] = None) -> SceneSnapshot:
        raise NotImplementedError

    def reset(self, scene_time_s: float = 0.0) -> None:
        _ = scene_time_s


class MutableSceneFrameProvider(SceneFrameProvider):
    """Scene provider for the current thick-lens static-BVH world.

    The provider owns the mutable scene object for now. A future provider can
    return immutable mesh/BVH frame handles without changing the camera-side
    contract.
    """

    def __init__(self,
                 scene: Any,
                 *,
                 time_attr: str = "subject_time_s",
                 label: str = "scene",
                 scene_advance: Optional[Callable[[Any, float, float], None]] = None) -> None:
        self.scene = scene
        self.time_attr = str(time_attr)
        self.label = str(label)
        self.scene_advance = scene_advance
        self.next_snapshot_id = 0
        self.last_sample_time: Optional[float] = None

    @property
    def current_scene(self) -> Any:
        return self.scene

    def snapshot(self,
                 t0: float,
                 t1: float,
                 sample_time: float,
                 *,
                 exposure_frame: Optional[ExposureFrame] = None) -> SceneSnapshot:
        sample = float(sample_time)
        if self.time_attr and hasattr(self.scene, self.time_attr):
            setattr(self.scene, self.time_attr, sample)
        if self.scene_advance is not None:
            self.scene_advance(self.scene, sample, max(0.0, float(t1) - float(t0)))
        prev = self.last_sample_time
        delta = 0.0 if prev is None else abs(sample - float(prev))
        self.last_sample_time = sample
        snap = SceneSnapshot(
            snapshot_id=int(self.next_snapshot_id),
            t0=float(t0),
            t1=float(t1),
            sample_time=sample,
            scene=self.scene,
            exposure_frame=exposure_frame,
            delta_metric=delta,
            label=self.label,
            metadata={"provider": type(self).__name__},
        )
        self.next_snapshot_id += 1
        return snap

    def reset(self, scene_time_s: float = 0.0) -> None:
        self.next_snapshot_id = 0
        self.last_sample_time = None
        if self.time_attr and hasattr(self.scene, self.time_attr):
            setattr(self.scene, self.time_attr, float(scene_time_s))


@dataclass(frozen=True)
class SceneCameraStep:
    """One outer simulated-time step containing scene and camera time."""

    step_id: int
    t0: float
    t1: float
    scene_dt: float
    camera_dt: float
    exposure_frame: ExposureFrame
    snapshot: Optional[SceneSnapshot] = None

    @property
    def dt(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))


class SceneCameraCoordinator:
    """Coordinates scene snapshots and camera exposure frames.

    The coordinator is the owner of scene time for camera capture. Callers hand
    it a dt, and it returns a camera exposure frame plus the scene snapshot that
    belongs to that frame.
    """

    def __init__(self,
                 scene_provider: SceneFrameProvider,
                 camera_scheduler: Optional[CameraExposureScheduler] = None,
                 *,
                 scene_time_s: float = 0.0) -> None:
        self.scene_provider = scene_provider
        self.camera_scheduler = camera_scheduler or CameraExposureScheduler()
        self.scene_time_s = float(scene_time_s)
        self.next_step_id = 0
        self.last_step: Optional[SceneCameraStep] = None
        self.last_snapshot: Optional[SceneSnapshot] = None

    def begin_step(self,
                   dt_s: float,
                   *,
                   scene_dt_s: Optional[float] = None,
                   camera_dt_s: Optional[float] = None,
                   sample_time_s: Optional[float] = None,
                   force_new_camera_frame: bool = False) -> SceneCameraStep:
        dt = max(0.0, float(dt_s))
        scene_dt = max(0.0, float(scene_dt_s)) if scene_dt_s is not None else dt
        camera_dt = max(0.0, float(camera_dt_s)) if camera_dt_s is not None else dt
        t0 = float(self.scene_time_s)
        t1 = t0 + scene_dt

        scene = self.scene_provider.current_scene
        if force_new_camera_frame or self.camera_scheduler.active_frame is None:
            frame = self.camera_scheduler.begin_frame(scene, camera_dt, t0_s=t0)
        else:
            frame = self.camera_scheduler.ensure_frame(scene, camera_dt, t0_s=t0)

        sample = float(sample_time_s) if sample_time_s is not None else t1
        snapshot = self.scene_provider.snapshot(t0, t1, sample, exposure_frame=frame)
        self.scene_time_s = t1

        step = SceneCameraStep(
            step_id=int(self.next_step_id),
            t0=t0,
            t1=t1,
            scene_dt=scene_dt,
            camera_dt=camera_dt,
            exposure_frame=frame,
            snapshot=snapshot,
        )
        self.next_step_id += 1
        self.last_step = step
        self.last_snapshot = snapshot
        return step

    def ensure_step(self, dt_s: float, **kwargs: Any) -> SceneCameraStep:
        if self.camera_scheduler.active_frame is not None and self.last_step is not None:
            return self.last_step
        return self.begin_step(dt_s, **kwargs)

    def finish_frame(self) -> None:
        self.camera_scheduler.finish_frame()
        self.scene_time_s = max(self.scene_time_s, self.camera_scheduler.scene_time_s)

    def reset(self, scene_time_s: float = 0.0) -> None:
        self.scene_time_s = float(scene_time_s)
        self.next_step_id = 0
        self.last_step = None
        self.last_snapshot = None
        self.camera_scheduler.reset(scene_time_s)
        self.scene_provider.reset(scene_time_s)


@dataclass(frozen=True)
class CameraTimelineSlice:
    """Camera program state at one em-timescale time slice.

    Wraps ExposureSlice with explicit camera-program booleans so the
    pipeline can enforce ordering without inspecting shutter weights.
    """

    frame_id: int
    slice_id: int
    t0: float
    t1: float
    flash_active: bool
    shutter_open_fraction: float
    sensor_integrating: bool
    exposure_slice: "ExposureSlice"
    scene_version_id: Optional[int] = None

    @property
    def dt(self) -> float:
        return max(0.0, self.t1 - self.t0)


@dataclass(frozen=True)
class CameraTimeline:
    """Ordered em-timescale slice sequence for one camera exposure.

    prerequisite_flash_slice_for_sensor maps each sensor slice_id to the
    flash slice_id whose records must have materialized before that sensor
    slice may be admitted. Emitter activity comes from the scheduler-authored
    ExposureSlice; the timeline does not reinterpret duty cycle.
    """

    frame_id: int
    flash_slices: tuple          # tuple[int, ...]
    sensor_slices: tuple         # tuple[int, ...]
    prerequisite_flash_slice_for_sensor: Dict[int, int]
    slices: tuple                # tuple[CameraTimelineSlice, ...]

    @staticmethod
    def from_exposure_frame(frame: "ExposureFrame") -> "CameraTimeline":
        """Compile the scheduler-authored frame without reinterpreting it."""

        n = len(frame.slices)
        if n == 0:
            return CameraTimeline(
                frame_id=frame.frame_id,
                flash_slices=(),
                sensor_slices=(),
                prerequisite_flash_slice_for_sensor={},
                slices=(),
            )
        flash_slice_ids = tuple(
            i for i, exposure_slice in enumerate(frame.slices)
            if bool(getattr(exposure_slice, "flash_active", True))
        )
        sensor_slice_ids = tuple(
            i for i, exposure_slice in enumerate(frame.slices)
            if bool(getattr(exposure_slice, "sensor_integrating", True))
        )
        prereqs: Dict[int, int] = {}
        for sensor_slice_id in sensor_slice_ids:
            eligible = tuple(
                flash_slice_id for flash_slice_id in flash_slice_ids
                if flash_slice_id <= sensor_slice_id
            )
            prereqs[sensor_slice_id] = eligible[-1] if eligible else -1
        flash_slice_set = set(flash_slice_ids)
        sensor_slice_set = set(sensor_slice_ids)
        timeline_slices = []
        for i, es in enumerate(frame.slices):
            timeline_slices.append(CameraTimelineSlice(
                frame_id=frame.frame_id,
                slice_id=i,
                t0=es.t0,
                t1=es.t1,
                flash_active=(i in flash_slice_set),
                shutter_open_fraction=float(es.shutter_open),
                sensor_integrating=(i in sensor_slice_set),
                exposure_slice=es,
            ))
        return CameraTimeline(
            frame_id=frame.frame_id,
            flash_slices=flash_slice_ids,
            sensor_slices=sensor_slice_ids,
            prerequisite_flash_slice_for_sensor=prereqs,
            slices=tuple(timeline_slices),
        )


@dataclass(frozen=True)
class CameraSliceState:
    """Observable lifecycle of one camera slice.

    Work-dispatch flags describe estimator/runtime progress. The remaining
    flags describe increasingly strong physical commit boundaries.
    """

    emission_work_dispatched: bool = False
    sensor_probe_work_dispatched: bool = False
    transport_products_materialized: bool = False
    reception_closed: bool = False
    detector_integration_committed: bool = False


class ExposureBarrier:
    """Tracks computational work and physical commit for one exposure.

    Emission and sensor-probe work may be dispatched in either order. Neither
    dispatch is itself a physical sensor event. A detector commit becomes
    eligible only after transport materialization and reception closure.

    The legacy flash/sensor method names remain narrow compatibility aliases.
    """

    def __init__(self, timeline: CameraTimeline) -> None:
        self.timeline = timeline
        self._flash_dispatched_through: int = -1
        self.flash_submitted_through: int = -1   # confirmed materialized
        self.sensor_submitted_through: int = -1
        self._flash_dispatched: set[int] = set()
        self._flash_materialized: set[int] = set()
        self._sensor_submitted: set[int] = set()
        self._emission_work_dispatched: set[int] = set()
        self._transport_materialized: set[int] = set()
        self._sensor_probe_work_dispatched: set[int] = set()
        self._reception_closed: set[int] = set()
        self._detector_committed: set[int] = set()
        self.on_flash_materialized = None
        self.on_transport_materialized = None
        self.on_reception_closed = None
        self.on_detector_committed = None

    def _slice(self, slice_id: int) -> CameraTimelineSlice:
        slice_id = int(slice_id)
        if not 0 <= slice_id < len(self.timeline.slices):
            raise ValueError("camera slice id is outside the authored timeline")
        result = self.timeline.slices[slice_id]
        if int(result.slice_id) != slice_id:
            raise RuntimeError("camera timeline slice ids are not dense and stable")
        return result

    @staticmethod
    def _contiguous_through(expected: tuple, completed: set[int]) -> int:
        through = -1
        for slice_id in expected:
            if slice_id not in completed:
                break
            through = int(slice_id)
        return through

    def state(self, slice_id: int) -> CameraSliceState:
        slice_id = int(self._slice(slice_id).slice_id)
        return CameraSliceState(
            emission_work_dispatched=(
                slice_id in self._emission_work_dispatched
            ),
            sensor_probe_work_dispatched=(
                slice_id in self._sensor_probe_work_dispatched
            ),
            transport_products_materialized=(
                slice_id in self._transport_materialized
            ),
            reception_closed=slice_id in self._reception_closed,
            detector_integration_committed=(
                slice_id in self._detector_committed
            ),
        )

    def record_emission_work_dispatched(
        self, slice_id: int, submitted: int = 1
    ) -> None:
        """Record light-side estimator dispatch, including a zero-ray marker."""

        slice_id = int(self._slice(slice_id).slice_id)
        if int(submitted) < 0:
            raise ValueError("submitted emission count must be non-negative")
        self._emission_work_dispatched.add(slice_id)
        if slice_id in self.timeline.flash_slices:
            self._flash_dispatched.add(slice_id)
            self._flash_dispatched_through = self._contiguous_through(
                self.timeline.flash_slices, self._flash_dispatched
            )

    def record_sensor_probe_work_dispatched(
        self, slice_id: int, submitted: int = 1
    ) -> None:
        """Record nonphysical sensor-side importance-query dispatch."""

        timeline_slice = self._slice(slice_id)
        slice_id = int(timeline_slice.slice_id)
        if not timeline_slice.sensor_integrating:
            if int(submitted) == 0:
                return
            raise ValueError(
                "cannot dispatch sensor probes for a non-integrating slice"
            )
        if int(submitted) < 0:
            raise ValueError("submitted sensor-probe count must be non-negative")
        self._sensor_probe_work_dispatched.add(slice_id)
        self._sensor_submitted.add(slice_id)
        self.sensor_submitted_through = self._contiguous_through(
            self.timeline.sensor_slices, self._sensor_submitted
        )

    def confirm_transport_products_materialized(self, slice_id: int) -> None:
        """Confirm all admitted transport work for the slice terminated."""

        slice_id = int(self._slice(slice_id).slice_id)
        if slice_id not in self._emission_work_dispatched:
            raise RuntimeError(
                "cannot materialize transport before emission work is declared"
            )
        newly_materialized = slice_id not in self._transport_materialized
        self._transport_materialized.add(slice_id)
        if slice_id in self.timeline.flash_slices:
            self._flash_materialized.add(slice_id)
            self.flash_submitted_through = self._contiguous_through(
                self.timeline.flash_slices, self._flash_materialized
            )
        if newly_materialized:
            if self.on_transport_materialized is not None:
                self.on_transport_materialized(slice_id)
            if (
                slice_id in self.timeline.flash_slices
                and self.on_flash_materialized is not None
            ):
                self.on_flash_materialized(slice_id)

    def reception_may_close(self, slice_id: int) -> bool:
        timeline_slice = self._slice(slice_id)
        slice_id = int(timeline_slice.slice_id)
        if slice_id not in self._transport_materialized:
            return False
        return (
            not timeline_slice.sensor_integrating
            or slice_id in self._sensor_probe_work_dispatched
        )

    def confirm_reception_closed(self, slice_id: int) -> None:
        """Confirm coherent/stochastic reduction completed for the slice."""

        slice_id = int(self._slice(slice_id).slice_id)
        if not self.reception_may_close(slice_id):
            raise RuntimeError(
                "cannot close reception before transport and sensor probes"
            )
        newly_closed = slice_id not in self._reception_closed
        self._reception_closed.add(slice_id)
        if newly_closed and self.on_reception_closed is not None:
            self.on_reception_closed(slice_id)

    def confirm_detector_integration_committed(self, slice_id: int) -> None:
        """Commit the physical detector contribution for one integrating slice."""

        timeline_slice = self._slice(slice_id)
        slice_id = int(timeline_slice.slice_id)
        if not timeline_slice.sensor_integrating:
            raise ValueError(
                "cannot commit detector integration for a non-integrating slice"
            )
        if slice_id not in self._reception_closed:
            raise RuntimeError(
                "cannot commit detector integration before reception closes"
            )
        newly_committed = slice_id not in self._detector_committed
        self._detector_committed.add(slice_id)
        if newly_committed and self.on_detector_committed is not None:
            self.on_detector_committed(slice_id)

    def record_flash_dispatched(self, slice_id: int, submitted: int = 1) -> None:
        """Compatibility alias for camera-flash emission work."""

        slice_id = int(slice_id)
        if int(submitted) < 0:
            raise ValueError("submitted flash count must be non-negative")
        if slice_id not in self.timeline.flash_slices:
            if submitted == 0:
                return
            raise ValueError("cannot dispatch flash for a non-flash camera slice")
        self.record_emission_work_dispatched(slice_id, submitted)
        if submitted == 0:
            # Nothing was emitted; nothing to wait for — confirm immediately.
            self.confirm_transport_products_materialized(slice_id)

    def confirm_flash_materialized(self, slice_id: int) -> None:
        """Compatibility alias for transport materialization."""

        slice_id = int(self._slice(slice_id).slice_id)
        if slice_id not in self.timeline.flash_slices:
            raise ValueError("cannot materialize flash for a non-flash camera slice")
        self.confirm_transport_products_materialized(slice_id)

    def sensor_may_submit(self, slice_id: int) -> bool:
        """Legacy materialization gate; sensor-probe dispatch need not use it."""

        prereq = self.timeline.prerequisite_flash_slice_for_sensor.get(slice_id, -1)
        if prereq < 0:
            return True
        return prereq in self._flash_materialized

    def record_sensor_submitted(self, slice_id: int) -> None:
        """Compatibility alias for sensor-probe work dispatch."""

        self.record_sensor_probe_work_dispatched(slice_id)

    @property
    def all_flash_dispatched(self) -> bool:
        if not self.timeline.flash_slices:
            return True
        return all(
            slice_id in self._flash_dispatched
            for slice_id in self.timeline.flash_slices
        )

    @property
    def all_sensor_submitted(self) -> bool:
        if not self.timeline.sensor_slices:
            return True
        return all(
            slice_id in self._sensor_submitted
            for slice_id in self.timeline.sensor_slices
        )

    @property
    def exposure_complete(self) -> bool:
        for timeline_slice in self.timeline.slices:
            slice_id = int(timeline_slice.slice_id)
            if timeline_slice.sensor_integrating:
                if slice_id not in self._detector_committed:
                    return False
            elif slice_id not in self._transport_materialized:
                return False
        return True


class SceneCameraClock:
    """Outer deterministic clock for a scene observed by a camera.

    The scene container owns this clock. Each call advances simulated scene
    time, optionally lets the scene update itself, and ensures the camera has
    an exposure frame whose slices belong to the same interval.
    """

    def __init__(self, scene_time_s: float = 0.0) -> None:
        self.scene_time_s: float = float(scene_time_s)
        self.next_step_id: int = 0
        self.last_step: Optional[SceneCameraStep] = None

    def advance(self,
                scene: Any,
                camera_scheduler: CameraExposureScheduler,
                dt_s: float,
                *,
                scene_dt_s: Optional[float] = None,
                camera_dt_s: Optional[float] = None,
                scene_advance: Optional[Callable[[Any, float, float], None]] = None,
                start_new_camera_frame: bool = False) -> SceneCameraStep:
        """Advance scene time and ensure a camera exposure frame.

        Parameters
        ----------
        scene
            Scene object or config. If it exposes ``subject_time_s``, this
            method updates it to the end of the scene interval.
        camera_scheduler
            The camera-owned exposure scheduler.
        dt_s
            Caller-provided simulated step. This is never wall-clock sleeping.
        scene_dt_s, camera_dt_s
            Optional split. Defaults to the same `dt_s` for both: the camera
            observes the same scene interval. Later callers can decouple these
            for slow-motion, high-speed capture, or paused scene capture.
        scene_advance
            Optional callback `(scene, t1, dt)` for richer scene containers.
        start_new_camera_frame
            Force a new exposure frame even if the camera has one active.
        """
        dt = max(0.0, float(dt_s))
        scene_dt = max(0.0, float(scene_dt_s)) if scene_dt_s is not None else dt
        camera_dt = max(0.0, float(camera_dt_s)) if camera_dt_s is not None else dt

        t0 = float(self.scene_time_s)
        t1 = t0 + scene_dt
        self.scene_time_s = t1

        if hasattr(scene, "subject_time_s"):
            setattr(scene, "subject_time_s", t1)
        if scene_advance is not None:
            scene_advance(scene, t1, scene_dt)

        if start_new_camera_frame or camera_scheduler.active_frame is None:
            frame = camera_scheduler.begin_frame(scene, camera_dt, t0_s=t0)
        else:
            frame = camera_scheduler.ensure_frame(scene, camera_dt, t0_s=t0)

        step = SceneCameraStep(
            step_id=int(self.next_step_id),
            t0=t0,
            t1=t1,
            scene_dt=scene_dt,
            camera_dt=camera_dt,
            exposure_frame=frame,
            snapshot=None,
        )
        self.next_step_id += 1
        self.last_step = step
        return step

    def reset(self, scene_time_s: float = 0.0) -> None:
        self.scene_time_s = float(scene_time_s)
        self.next_step_id = 0
        self.last_step = None
