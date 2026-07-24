# Native optical transaction coverage

Date: 2026-07-23

This ledger defines what a camera/Turing managed window may mutate and still
claim exact rollback. It is a correctness boundary, not a list of everything
that eventually ought to be copied. A subsystem is eligible for a scientific
window only when it is:

1. immutable for the complete camera-program generation;
2. covered by a registered `ManagedProcessState` participant;
3. derived from covered state and invalidated by generation on rollback; or
4. explicitly outside the transaction as a lossy presentation product.

Quiescence alone does not restore state. A native handle is not covered merely
because its Python owner is checkpointed.

## Landed participants

| State owner | Mutable state | Mechanism | Evidence |
|---|---|---|---|
| Turing `ManagedTimeRuntime` | committed time, request sequence, dt controller | whole-window snapshot/restore | focused Turing managed-time tests |
| optical solver Python state | participant-defined solver values | `copy_shallow()` / `restore()` | managed bridge and cross-repository test |
| Nodus `EdgeTensorFifo` | storage, slot tags, write sequence, writer binding, every reader cursor | versioned quiescent transaction snapshot shared by runtime and table ABIs | native CTests, byte-identical ABI parity, camera/Turing rerun test |
| native T4 wave arenas | split-complex field blocks, active fields and lanes, lane wavelength/index state, progress, link generation/terminal telemetry, boundary telemetry | opaque same-process `RayPipelineWaveCheckpoint`; idle-only, topology/configuration checked | deterministic native restore/replay test |
| native CPU pipeline queues | T1/T2/T3/T4/T5 inputs, ordinary output, wave exits, BDPT side-data, T5-ready channel | reusable non-destructive `PipelineQueue<T>` snapshots composed by idle-only `RayPipelineQueueCheckpoint`; GPU pipelines rejected | native queue unit test and rejected-output retraction/replay test |
| Python causal reception queue | reader acknowledgements and retirement | Python behavioral oracle | camera timing tests |

`NativeWaveStateParticipant` and `NativeQueueStateParticipant` cover only
their named rows. Neither must be registered under a misleading name such as
`native_pipeline` until the remaining rows are covered.

## Cold state that may be generation-pinned

These values need not be copied per window if mutation is prohibited until the
camera program is superseded:

- ray-tracer triangles, BVH, material buffers, spectral LUT definitions;
- scale-context and wave-arena topology;
- T4 FFT plans, geometry, boundary configuration, and aperture material;
- compiled optical assembly and lens/aperture graph;
- complex source basis/operator tables;
- Nodus FIFO shape/type/capacity and registered reader set;
- camera electromechanical program and scene snapshot identity.

Every handle into cold state needs a generation. Restore rejects Nodus reader
set drift and T4 arena topology/configuration drift now. Equivalent checks are
still required for other native arenas.

## Uncovered CPU-visible state

| Owner | Examples | Required next mechanism |
|---|---|---|
| BDPT process state | side-data queues, pending vectors, reusable light stash, T5-ready channel, overflow and in-flight counters, next subpath id | one BDPT transaction participant with deterministic record ordering |
| sensor state | RGB/near-miss accumulators, priority map, epoch counts, running peaks | copy-on-write sensor generations or bounded tile checkpoints |
| ray-tracer accumulators | UV image accumulators, BSSRDF illumination accumulators, camera field grid/strikes, event telemetry | explicit accumulator participant; cold scene state must remain separate |
| legacy stateful ray scheduler | live ray pool, amplitude pool, geometric/context queues | checkpoint or prohibit this API inside managed camera windows |
| adaptive execution state | stage batch sizes/fractions, performance counters | decide which values affect scientific ordering; checkpoint deterministic controls and classify pure telemetry separately |

A T4 trace emits records to `Q_out`. The native replay qualification now proves
that composing wave and queue checkpoints retracts the rejected record before
retry. The full native ray pipeline remains incomplete because accumulators
and counters outside the queues are not yet covered.

## Uncovered GPU-resident state

The GL dispatch object contains persistent T1-T5 SSBOs, indirect counters,
sensor RGB/mipmap hierarchy, BDPT/T5 records and backlog, VCM tables, display
accumulators, publication generations, and pending GPU jobs.

Bulk readback per timestep is not the desired architecture. Preferred order:

1. allocate generation-owned logical state for each admitted camera slice;
2. publish output generations only after the Turing commit gate succeeds;
3. on rollback, abandon the rejected generation without making it visible;
4. use copy-on-write or ping-pong buffers for state that must seed a retry;
5. retain bounded readback snapshots only as a qualification oracle.

Presentation textures may be lossy and need not roll back pixel-for-pixel, but
their generation registry must never adopt a product from a rejected camera
causality generation.

## Commit eligibility

For scientific mode, the camera commit predicate is the conjunction of:

- exact Turing landing at the authored slice boundary;
- all Nodus processes quiescent with no forbidden drop/overflow;
- every lossless reader released the causality slice;
- every enabled native participant checkpointed and restorable;
- no uncovered native subsystem was mutated;
- coherent reductions reached their deterministic completion criteria;
- sensor/exposure products belong to the same camera causality and program
  generation.

Realtime preview is a separate mode. It may skip these guarantees but must
report time slip, dropped work, and non-scientific status.

## Next implementation order

1. add sensor and BDPT participants, including pending/stash vectors and
   deterministic counters;
2. cover ray-tracer UV/illumination accumulators and legacy live-ray pools;
3. join them with the landed wave and queue participants into an accurately named CPU
   pipeline participant;
4. add GPU generation/COW state and a commit-time publication fence;
5. only then let the full native optical callback execute inside a scientific
   managed window.
