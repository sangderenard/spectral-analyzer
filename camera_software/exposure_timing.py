"""Deterministic simulated-time exposure slicing for camera software.

This module is intentionally small: it gives camera code a stable frame/slice
identity and shutter/flash timing contract before the ray pipeline grows more
complete per-slice completion barriers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


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

    @property
    def dt(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))

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
    slices: List[ExposureSlice] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))


class CameraExposureScheduler:
    """Builds and owns deterministic exposure slices for the active frame."""

    def __init__(self) -> None:
        self.scene_time_s: float = 0.0
        self.next_frame_id: int = 0
        self.active_frame: Optional[ExposureFrame] = None

    def begin_frame(self,
                    scene: Any,
                    dt_s: Optional[float] = None,
                    *,
                    t0_s: Optional[float] = None) -> ExposureFrame:
        """Start a new exposure frame from the current camera/scene settings."""
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
        if self.active_frame is not None:
            self.scene_time_s = max(self.scene_time_s, float(self.active_frame.t1))
        self.active_frame = None

    def reset(self, scene_time_s: float = 0.0) -> None:
        self.scene_time_s = float(scene_time_s)
        self.next_frame_id = 0
        self.active_frame = None

    def _build_frame(self, scene: Any, dt_s: Optional[float]) -> ExposureFrame:
        burst = getattr(scene, "camera_light_burst", None)
        plate = getattr(scene, "image_plate", None)

        stages = int(max(1, getattr(burst, "stages", 1)))
        exposure_time = float(getattr(burst, "exposure_time_s", 0.010))
        if dt_s is not None:
            exposure_time = float(max(0.0, dt_s))
        if exposure_time <= 0.0:
            exposure_time = 0.010

        duty = _clamp01(getattr(burst, "duty_cycle", 1.0), 1.0)
        energy = max(0.0, float(getattr(burst, "energy_scale", 1.0)))
        enabled = bool(getattr(burst, "enabled", True))
        profile = str(getattr(burst, "profile", "steady") or "steady")
        per_slice_energy = (energy * duty / float(stages)) if enabled else 0.0

        mode = _shutter_mode_code(getattr(plate, "shutter_mode", "open"))
        open_f = _clamp01(getattr(plate, "shutter_open", 1.0), 1.0)
        cu_base = _clamp01(getattr(plate, "shutter_center_u", 0.5), 0.5)
        cv_base = _clamp01(getattr(plate, "shutter_center_v", 0.5), 0.5)
        softness = max(0.0, float(getattr(plate, "shutter_softness", 0.0)))

        frame_id = int(self.next_frame_id)
        t0 = float(self.scene_time_s)
        slice_dt = exposure_time / float(stages)
        slices: List[ExposureSlice] = []
        for slice_id in range(stages):
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
                exposure_weight=per_slice_energy,
                flash_weight=per_slice_energy,
                sensor_weight=per_slice_energy,
                profile=profile,
            ))

        return ExposureFrame(
            frame_id=frame_id,
            t0=t0,
            t1=t0 + exposure_time,
            slices=slices,
        )


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
