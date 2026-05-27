# Camera Simulated-Time Exposure Plan

## Direction

Move exposure orchestration into camera software as a deterministic simulated-time harness. The caller provides a scene/camera `dt`; camera software subdivides that interval into exposure slices, shutter sweeps, flash-program slices, and connector eligibility barriers.

This is not wall-clock pacing. Wall time may be used for UI responsiveness and scheduling fairness, but optical transport time is a simulation value owned by the camera and scene container.

## Goals

1. Give the camera a real scene-time boundary: `advance(dt)` or equivalent receives a time slice from the containing scene/application.
2. Let the scene container advance mesh/object animation for the same interval, including orbiting subjects, before or during camera sub-slices as needed.
3. Make the sensor sweep and camera-mounted flash/light programs consume matching simulated-time slices.
4. Replace loose `dispatched` latches with slice-completion barriers that mean record production for that slice is closed enough for T5/MIS.
5. Keep BDPT/MIS transport stage-deployable across CPU and GPU operators; the harness coordinates stages, it does not make a one-side-only path.

## Exposure Slice Model

Each camera frame is divided into exposure slices:

- `frame_id`: monotonically increasing camera frame.
- `slice_id`: monotonically increasing within a frame.
- `t0`, `t1`: simulated scene time covered by this slice.
- `dt`: `t1 - t0`.
- `shutter_mask`: sensor area active during the slice.
- `flash_program_state`: camera-light emission behavior for the slice.
- `scene_sample_state`: geometry/material time state used by transport for the slice.

The first implementation can keep geometry frozen per slice. Later moving geometry can be sampled at `t0`, midpoint, or with per-ray time offsets once the intersection path supports it.

## Stage Completion Barriers

Current `signal_flash_dispatched()` and `signal_sensor_dispatched()` are too weak because submission is not completion. The intended barrier states are:

1. Flash slice submitted.
2. Flash slice transport closed.
3. Sensor slice submitted.
4. Sensor slice transport closed.
5. T5/MIS slice eligible.
6. T5/MIS slice complete.

T5 should connect a named slice, not a loose global record pool. A slice becomes eligible only when both light-side and sensor-side transport are closed for that slice.

## Record Identity

Short term:

- Add frame/slice ownership to the camera software harness.
- Keep per-slice pending record pools until T5 consumes that slice.
- Do not consume light records globally before the matching sensor slice is closed.

Medium term:

- Add a compact `slice_id` or `time_offset` field to transport records if SSBO budget permits.
- Prefer `slice_id` first; it fixes ordering and barrier correctness without requiring moving-geometry evaluation.
- Add physical time offsets only when lens/material/scene evaluation can honor them.

## Flash Program

Camera-mounted lights should be camera software actors. For each exposure slice, the flash program receives:

- `frame_id`, `slice_id`
- `t0`, `dt`
- shutter/open state
- planned burst envelope
- emitter profile/material state

It submits only the energy belonging to that slice and reports completion only after its transport records are closed.

## Sensor Program

The sensor back owns raw formation. For each exposure slice, it receives:

- active shutter region
- sensor sweep stage
- exposure weight
- slice identity

It submits discrete sensor sweep work and reports completion only after sensor-side transport records are closed.

## Immediate Fixes Enabled

1. T5 no longer fires on partial record pools.
2. Light records are retained until the matching sensor slice is closed and connected.
3. CPU and GPU material paths share the same completion semantics.
4. Progress HUD can report actual slice states instead of ambiguous global counters.
5. Scene animation can return through deterministic camera time instead of ad hoc render-loop behavior.

## Implementation Order

1. Introduce camera-software exposure slice data structures and a no-op deterministic scheduler.
2. Route current shutter-stage calculation through the scheduler.
3. Change flash/sensor latch names and semantics from dispatch counters to slice completion.
4. Make T5 consume specific closed slices and retain unmatched records by slice.
5. Expose slice progress in HUD.
6. Add optional record `slice_id` or `time_offset` only after the barrier model is correct.

## Current Structural Landing

Implemented first:

- `camera_software.exposure_timing.ExposureSlice`
- `camera_software.exposure_timing.ExposureFrame`
- `camera_software.exposure_timing.CameraExposureScheduler`

The lab now routes its existing exposure stage count, shutter sweep arguments, and per-stage flash/sensor weight through this scheduler. This intentionally preserves current transport behavior while creating the frame/slice identity needed for the next migration: replacing loose dispatch counters with per-slice completion barriers.

Added next:

- `camera_software.exposure_timing.SceneCameraStep`
- `camera_software.exposure_timing.SceneCameraClock`
- `camera_software.exposure_timing.SceneSnapshot`
- `camera_software.exposure_timing.SceneFrameProvider`
- `camera_software.exposure_timing.MutableSceneFrameProvider`
- `camera_software.exposure_timing.SceneCameraCoordinator`

`CameraItem` now carries a camera-owned `exposure_scheduler` property, and `CameraContext` exposes it to camera software modules. The thick lens harness now uses an outer scene-camera coordinator, which advances camera capture time by explicit steps, requests a scene snapshot for that step, and ensures camera exposure slices are allocated from the same simulated-time interval.

The thick lens subject scene is currently BVH-backed static geometry, so live animation is applied by controlled scene rebuilds keyed from coordinator snapshots. The reusable orbiter scene's `scene_for_phase()` is evaluated at `scene.subject_time_s`; the harness rebuilds when the pipeline is between exposure stages and sufficiently idle. This is the correct structural handoff until moving geometry has a native pipeline representation.

The current direction is now coordinator-first: the coordinator owns scene time,
the scene provider delivers an explicit `SceneSnapshot`, and the camera exposure
scheduler binds that snapshot to an `ExposureFrame`. Display polling may observe
this state, but it must not advance scene time or create new transport time on
its own.

In the thick lens demo, `ForwardCppLensBench.begin_scene_camera_step()` now asks
the `SceneCameraCoordinator` for the active step. Repeated HUD/render-loop calls
reuse the active exposure frame until transport marks it complete. When the
subject animation requires a BVH rebuild, the rebuild consumes the stored
`SceneSnapshot` and then re-arms the new bench at the same step interval, so the
image path remains tied to the requested scene frame instead of an incidental
global time value.
