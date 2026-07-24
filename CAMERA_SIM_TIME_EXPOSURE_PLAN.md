# Camera Simulated-Time Exposure Plan

## Direction

Move exposure orchestration into camera software as a deterministic simulated-time harness. The caller provides a scene/camera `dt`; camera software subdivides that interval into exposure slices, shutter sweeps, flash-program slices, and connector eligibility barriers.

This is not wall-clock pacing. Wall time may be used for UI responsiveness and scheduling fairness, but optical transport time is a simulation value owned by the camera and scene container.

More precisely, the caller or simulation manager submits a requested state time
or interval; it does not dictate the camera's integration timestep. The camera
interprets that request through its authored electromechanical program and
returns a finite ordered slice budget. The optical/wave solver then maintains
physical-time advancement inside each admitted slice, including exact arrival
timestamps and persistent delayed state. Completion returns to the camera,
which alone advances shutter, flash, sensor-integration, transfer, and readout
eligibility.

Pluck/game time is meta time. Nodus scheduling time is execution time. Camera
program time is electromechanical simulated time. Wave-solver time is optical
physical time. They exchange revisioned requests, bounded slice budgets,
timestamped work, and completion events; they are not one global tick.

The system is intentionally multirate. Camera-scale intervals must not be
marched at an optical carrier timestep, and optical delay/phase must not be
quantized onto a display, game, or component-refresh clock.

## Goals

1. Give the camera a real scene-time boundary: `advance(dt)` or equivalent receives a time slice from the containing scene/application.
2. Let the scene container advance mesh/object animation for the same interval, including orbiting subjects, before or during camera sub-slices as needed.
3. Make the sensor sweep and camera-mounted flash/light programs consume matching simulated-time slices.
4. Replace loose `dispatched` latches with slice-completion barriers that mean record production for that slice is closed enough for T5/MIS.
5. Keep BDPT/MIS transport stage-deployable across CPU and GPU operators; the harness coordinates stages, it does not make a one-side-only path.

## Exposure Slice Model

Each camera frame is divided into exposure slices:

- `camera_id`: stable namespace for the authoring camera.
- `program_generation`: camera-script revision; old-generation work is stale.
- `causality_id`: camera-authored monotonic `uint64` work/commit identity.
- `frame_id`: monotonically increasing camera frame.
- `slice_id`: monotonically increasing within a frame.
- `t0`, `t1`: simulated scene time covered by this slice.
- `dt`: `t1 - t0`.
- `shutter_mask`: sensor area active during the slice.
- `flash_program_state`: camera-light emission behavior for the slice.
- `scene_sample_state`: geometry/material time state used by transport for the slice.

The first implementation can keep geometry frozen per slice. Later moving geometry can be sampled at `t0`, midpoint, or with per-ray time offsets once the intersection path supports it.

The causality ID is not time. It keys the immutable descriptor that contains
time, camera program state, and scene revision. Keeping those concepts separate
allows reruns, stale-result rejection, and exact timestamps without comparing
floating-point time as an identity.

## Unordered and Distributed Execution

Authored order and execution order are deliberately separate:

1. The camera authors immutable `CameraSliceJob` envelopes in causal order.
   The versioned JSON-compatible descriptor has a canonical SHA-256 digest, so
   a returned attempt can prove which exact camera/scene/solver request it ran.
2. Any local process, GPU worker, or network worker may execute any job.
3. A `CameraResultCollector` accepts results in arbitrary order, deduplicates
   retries by causality key plus content digest, and rejects conflicting or
   stale results.
4. Results become visible for physical commit in canonical camera order. This
   makes floating-point/reception reduction reproducible rather than dependent
   on network arrival order.
5. In the ordinary ordered single-worker path, every submission commits
   immediately and the reorder map remains empty.

Each concurrently active job must own an isolated mutable solver context (or a
future explicitly partitioned native context arena). Merely tagging records is
not sufficient if global queues, pending pools, sensor accumulation, or T5
latches remain shared. A single worker may efficiently reuse one native context
sequentially after the prior job reaches its closure boundary.

Native T5 counters therefore remain local attempt-completion mechanics. The
camera causality key is the portable identity above native tracer instances.
True same-instance interleaving is a later native context-table/arena feature,
not a precondition for shared rendering across processes or machines.

## Multi-reader Advance Gate

Canonical result availability is still not permission to advance time. A
`CausalMultireaderQueue` multicasts each committed slice to all systems
registered at publication time and retains it while any reader holds it.
Readers may process and acknowledge slices in any order, but retirement remains
a contiguous camera-authored prefix. Camera/dt advance is permitted only across
that fully released prefix.

Registration is prospective so a late reader cannot retroactively claim data it
never observed. Disconnecting a reader with outstanding holds is an error unless
the coordinator explicitly selects a failure policy that releases those holds.
This is the protocol the Turing managed-dt adapter should consume: solver
completion publishes causal state; wave, exposure, detector, storage/readout,
and other registered managers release their leases; the time report commits
only after the shared gate retires the slice.

## Stage Completion Barriers

`signal_flash_dispatched()` and `signal_sensor_dispatched()` are estimator-work
markers, not physical camera events. Light-side paths and backward sensor-side
importance probes may be constructed concurrently. The intended slice states
are:

1. Emission-side work dispatched, including an explicit zero-ray family.
2. Sensor-probe work dispatched when the sensor is integrating.
3. All admitted transport products materialized.
4. Coherent/stochastic reception closed.
5. Detector integration committed.
6. Later electromechanical transfer/read state completed when authored.

Native T5 already supplies the correct local causal latch: both work families
must be declared and every resulting path generation must reach zero in-flight
work before connection/reduction begins. T5 should ultimately connect a named
slice rather than a loose global record pool. Returning from a successful T5
join is the current BDPT reception-closed and detector-committed boundary; it
is not permission to advance any unrelated camera slice.

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
6. Disabling the camera flash no longer disables natural scene emitters or
   sensor-side importance sampling.
7. A native T5 watchdog/failure return cannot be mistaken for a committed
   detector result.

## Implementation Order

1. Introduce camera-software exposure slice data structures and a no-op deterministic scheduler.
2. Route current shutter-stage calculation through the scheduler.
3. Change flash/sensor latch names and semantics from dispatch counters to slice completion.
4. Make T5 consume specific closed slices and retain unmatched records by slice.
5. Expose slice progress in HUD.
6. Add optional record `slice_id` or `time_offset` only after the barrier model is correct.

## Current Structural Landing

Implemented first:

- `camera_software.exposure_timing.CameraCausalityKey`
- `camera_software.exposure_timing.ExposureSlice`
- `camera_software.exposure_timing.ExposureFrame`
- `camera_software.exposure_timing.CameraExposureScheduler`
- `camera_software.exposure_timing.CameraSliceJob`
- `camera_software.exposure_timing.CameraSliceResult`
- `camera_software.exposure_timing.CameraResultCollector`
- `camera_software.exposure_timing.CausalMultireaderQueue`

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

The qualified Turing boundary for the next integration tranche is
`src.common.ManagedTimeRuntime`. Each camera slice becomes a
`TimeWindowRequest`: camera-program generation maps to `generation`, slice
sequence maps to monotonically increasing `request_id`, `t0`/`t1` map to
`t_start`/`t_end`, and every authored shutter, flash, or sensor transition maps
to an exact `event_times` entry. The camera advances its script only after an
exact `TimeAdvanceReport`; failed windows leave both solver state and committed
Turing time unchanged.

That tranche now has an executable first seam in
`camera_software.managed_time_bridge.CameraManagedTimeBridge`. The adapter
constructs Turing's canonical `TimeWindowRequest` lazily rather than copying
its controller or request contract. Turing's `ManagedTimeRuntime.advance`
accepts a domain-neutral optional commit gate; a rejected gate restores the
entire managed window. `CausalMultireaderQueue.is_retired()` supplies the
current Python release oracle. The next Nodus adapter should contribute its
FIFO/process quiescence to the same gate.

The cross-repository lineage and verified landmark map is maintained in
`VITRUVIAN_REPOSITORY_ATLAS.md`. Read it before proposing a replacement graph,
time, tensor, compiler, or execution subsystem.

The Nodus half of this seam is now executable for a qualified typed FIFO edge.
`ManagedProcessState` composes solver and Nodus checkpoints; the FIFO
transaction snapshot includes writer and reader state; and the commit predicate
observes true all-reader quiescence. The integration test intentionally rejects
one exact-landed camera window, verifies solver and FIFO rollback, reruns it,
drains both readers, and commits the authored boundary.

The current `ExposureBarrier` now records the estimator/physical split
explicitly through `CameraSliceState`:

- emission work and sensor-probe work may be dispatched in either order;
- transport materialization is required before reception closure;
- reception closure is required before detector integration commit;
- exposure completion means every integrating slice committed its detector
  contribution, not merely that work was placed on queues;
- legacy flash/sensor method names remain compatibility aliases while callers
  migrate to the explicit lifecycle.

The camera scheduler now authors namespaced, generation-scoped monotonic
causality IDs without reusing IDs after reset. Distributed results can finish
in any order and retry idempotently, while conflicting duplicate results and
old-program results are rejected. The multireader queue separates "solver
result exists" from "all dependent systems consumed it", providing the explicit
advance gate needed by the managed-dt integration.

The thick-lens BDPT path now assigns independent weights to camera-flash and
natural-emitter families. The camera scheduler authors discrete flash-active
states, while natural emission follows sensor integration time even in a
no-flash exposure. A successful native `join_t5()` is checked before the Python
camera barrier commits reception and detector state.
