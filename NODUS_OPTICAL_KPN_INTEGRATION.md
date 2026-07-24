# Nodus KPN integration boundary for optical transport

The rendering and optical work exists to become a first-class computational
pillar that people can author, connect, and operate through Nodus. The optical
engine must therefore not grow a second general execution manager. Nodus at
`C:\dev\Powershell\nodus` is the eventual host authority for graph authorship,
branch-frontier FIFO transport, process scheduling, persistence, and resource
coordination. The standalone optical programs are focused development and
calibration hosts for the exact same pillar API, not a competing application
architecture.

This document records the inspected implementation state and the narrow
boundary required by the optical engine. It does not vendor Nodus code and does
not assign a license to either repository.

The complete repository ownership, current Pluck execution path, target
Nodus-driven process loop, data boundaries, and return-to-bench gates are in
`OPTICAL_ENGINE_REPOSITORY_INTEGRATION.md`.
The fresh-context implementation brief for the first coherent fan-in process
is `COHERENT_RECEPTION_POOL_HANDOFF.md`.

## Implemented Nodus facilities

Nodus `EdgeTensorFifo` is implemented, not merely proposed. The implementation
is in `include/inl/table_abi_core.inl`, its public operations are surfaced by
`include/inl/table_abi_edge_io.inl` and `include/table_abi.h`, and execution is
coordinated by `include/thread_manager.h` and `src/thread_manager.cpp`:

- one atomically bound writer per edge;
- fixed typed shape, element size, layout, dtype, slot capacity, and overwrite
  policy configured before work;
- backend-owned storage with host and device transfer hooks;
- fixed 64-entry reader table with independent monotonic sequences;
- release publication through per-slot sequence tags and acquire consumption;
- blocking and nonblocking reads/writes;
- batched pointer publish/consume;
- explicit rejected-full versus overwrite/drop behavior;
- authoritative reader-minimum tracking in the FIFO, with a
  coordinator-facing mirror now factored into domain-neutral
  `EdgeReaderRegistry`;
- sparse-delta and accumulated sparse-delta products;
- serializable FIFO state.

Nodus also supplies:

- typed sparse graphs whose connections are port-to-port;
- abstract tensor handles and backend dispatch;
- tensor pools intended to avoid allocator churn;
- heavily developed parallel scatter/gather and dyadic binning kernels;
- scheduled and free-spinning module execution.

## Important present limits

The current `ThreadManager` is a canvas/module coordinator, not yet a clean
domain-independent KPN library target.

- Its public module contracts name canvas tables, stages, and UI-oriented
  module indices.
- Acyclic graphs receive Kahn topological scheduling and a parallel ready
  frontier.
- Cyclic graphs currently receive an ASAP/ALAP slack ordering and execute once
  sequentially per submitted tick. That is useful scheduling, but it is not an
  exact zero-delay scattering solve, a delayed resonator worklist, or a
  convergence-certified optical SCC solve.
- `canvas_tables` still packages the larger graph manager with UI, SDL,
  typography, and canvas behavior. The optical engine no longer links that
  pillar merely to obtain an edge queue.

The first physical extraction step landed on 2026-07-23:

- `include/edge_reader_registry.h` now owns the thread-safe external reader-slot
  mirror without naming tables, canvas, stages, tools, or optical policy;
- each `ThreadManager` owns one registry, so edge-id namespaces from unrelated
  jobs do not collide in a permanent process-global registry;
- the manager's legacy methods delegate to it, preserving the current ABI;
- unregister now recomputes the frontier from surviving readers' actual
  sequences instead of retaining a stale aggregate minimum;
- `tests/edge_reader_registry_test.cpp` qualifies independent edges,
  multi-reader minima, removal, and invalid-slot updates.

The transport extraction then landed:

- the existing `EdgeTensorFifo` moved, without a parallel implementation, to
  the self-contained `include/edge_tensor_fifo.h`;
- its three redundant `ThreadManager` minimum lookups were removed; the
  FIFO's internal 64-reader atomic table is the transport/backpressure
  authority;
- the versioned transaction format moved to
  `include/edge_tensor_fifo_transaction.h`, shared by table and runtime ABIs;
- `nodus_runtime.dll` exposes a one-edge domain-neutral C ABI without table,
  canvas, rope, typography, PNG, or stage dependencies;
- `cmake/auto_sources.cmake` excludes runtime-only sources from the legacy
  canvas glob so the dependency boundary cannot silently collapse;
- the Spectral bridge prefers this ABI and retains a legacy-table fallback;
- native runtime tests, byte-for-byte legacy snapshot parity, two-reader
  backpressure, and the camera/Turing rollback-rerun test pass.

## Required extraction in Nodus

The runtime-only target now contains item 2 and the edge-level portion of item
8 below. The remaining graph/process extraction still requires:

1. typed sparse port graph;
2. `EdgeTensorFifo`; **landed**
3. abstract tensor handle/backend and pool;
4. scatter/gather and dyadic binning operations;
5. a domain-neutral process callback contract;
6. deterministic ready-frontier scheduling;
7. explicit SCC policy callbacks rather than a built-in optical algorithm;
8. quiescence, residual-work, overflow, and drop accounting; **FIFO-level
   quiescence/drop accounting landed; process-level work remains**

The optical engine supplies SCC physics:

- zero-delay linear scattering: direct `(I-S)x=b` or equivalent compiled solve;
- positive-delay cycle: timestamped worklist within a declared time or residual
  window;
- nonlinear/time-varying cycle: explicit time evolution;
- coherent join: frequency/coherence/arrival-keyed complex accumulation.

Nodus supplies composition, lifecycle, execution, typed storage, backpressure,
and parallel frontier mechanics. It must not decide optical coherence or
convergence. The optical engine remains independently loadable so Nodus,
standalone calibration programs, and automated render workers all invoke one
physics authority.

The compiled optical schedule now distinguishes ordinary branch-frontier work,
timestamped acyclic edges, positive-delay cyclic worklists, and zero-delay SCC
solves. A global or LCM component clock may schedule controls and commensurate
process bands, but it never quantizes optical carrier phase or replaces exact
arrival/OPL metadata.

## Optical token contract

`csrc/include/optical_branch_abi.h` defines the fixed 96-byte scheduling token
that a Nodus edge would carry.

The token contains only:

- a handle to pipeline-owned contiguous state;
- the state generation required to reject a stale handle;
- node/product and lineage identity;
- coherence/sample/lane identity;
- high/low arrival time and optical path;
- exact continuous frequency where applicable;
- stochastic PDF and residual-power bound;
- validity and solve-mode flags.

It never contains an N-lane complex field. Wave arenas, complex-ray state
blocks, and compiled operators remain owned by the optical pipeline.
`kAbiVersion` is negotiated once in the typed Nodus edge/schema descriptor;
it is not repeated in every 96-byte token.

`csrc/include/optical_reception_abi.h` now defines a separate fixed 96-byte
reception-process token. Contribution and completion are explicit mutually
exclusive roles. The token carries a generation-checked handle into
pipeline-owned `OpticalReceptionKey` storage plus hot routing, contribution,
camera-causality, coherence, arrival, solve, and stale-state fields. Local
handles are never network identities. `camera_software/optical_reception_tokens.py`
is the bounded Python behavioral oracle, and
`camera_software/nodus_runtime_bridge.py` qualifies the byte contract against
the current Nodus C ABI.

Nodus now has a distinct versioned quiescent transaction snapshot for Turing
rollback. It includes configuration, storage, slot tags, write sequence, bound
writer, and every active reader key/sequence. Restore rejects configuration or
reader-set drift and synchronizes the coordinator-facing reader registry
through the legacy `ThreadManager` bridge. This is not the older persistence
snapshot. The scheduler must hold the process frontier quiescent while
capturing or restoring it.

The current Python qualification bridge implements `copy_shallow()`/`restore()`
with this ABI and reports quiescence only when every registered reader reaches
the write frontier. `ManagedProcessState` checkpoints it with solver state.
A real camera/Turing/Nodus test proves gate rejection restores solver time and
both reader cursors before the same request reruns and commits exactly.

## Adoption order

1. Keep current pipeline-native linear and cold fan-out paths.
2. Qualify `OpticalBranchToken` round-trip through an extracted Nodus FIFO.
   **Landed:** the exact 96-byte token passes the extracted runtime, legacy
   parity, two-reader backpressure, and transaction restore gates.
3. Compare shuffled versus ordered acyclic execution.
4. Add coherent fan-in as an optical accumulator process.
5. Add delayed-cycle worklists.
6. Add exact linear SCC callbacks.
7. Adopt the extracted Nodus manager only after overflow, quiescence,
   determinism, and throughput gates pass.

Today Pluck's `MultiEngineRenderGraph` calls the optical pillar's
`submit()`/`poll()` adapter from its frame loop. That is the migration seam,
not evidence that Nodus is already driving optical work. The first Nodus
integration should replace the process-loop driver behind the pillar while
preserving its request and preview-product surfaces.

No optical result may be silently dropped. A bounded FIFO may apply
backpressure, grow during cold configuration, or terminate with a failed solve
report; overwrite/drop is permitted only for explicitly lossy preview products.

Turing now exposes the domain-neutral transaction seam needed by this runtime:
`ManagedTimeRuntime.advance(..., commit_gate=...)`. Nodus FIFO/process
quiescence should be combined with optical reception and camera-reader release
at that boundary. A false gate rolls back the complete managed window; it is
not a polling tick or permission to discard queued work.
