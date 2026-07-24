# Coherent reception pool: fresh-agent implementation handoff

Date: 2026-07-23  
Repository: `C:\Users\alber\Downloads\spectral-analyzer`  
Branch at handoff: `nogodsnomasters`  
Related repository: `C:\dev\Powershell\nodus`

## Continuation status: camera/Turing seam landed

On 2026-07-23 the first executable cross-repository time boundary landed:

- `camera_software/managed_time_bridge.py` maps `CameraSliceJob` to Turing's
  canonical `TimeWindowRequest` without copying Turing timestep logic;
- Turing `ManagedTimeRuntime.advance()` now accepts an optional domain-neutral
  precommit gate;
- gate rejection after exact solver landing restores the complete time window,
  controller, committed clock, and request sequence;
- `CausalMultireaderQueue.is_retired()` is the present Python release oracle;
- a live test using both repositories proved two exact event boundaries,
  two-reader refusal with whole-window rollback, and exact commit after release.

The next process-runtime tranche should adapt Nodus quiescence and reader
release to this gate. Do not build another coordinator.

That tranche has now landed for one qualified FIFO edge:

- the 96-byte `OpticalReceptionToken` carries explicit contribution/completion
  roles and a generation-checked handle into bounded rich-key storage;
- Nodus transports it losslessly to independent readers;
- Nodus transaction snapshots cover storage, writer binding, and reader
  cursors and restore the coordinator-facing reader minima;
- `ManagedProcessState` checkpoints solver and process runtime together;
- the end-to-end camera/Turing/Nodus test proves rejection, complete rollback,
  rerun, reader drain, and exact commit.

The remaining production work is widening this proven transaction boundary
from the extracted FIFO and T4 wave state to the complete native optical
pipeline.

The first Nodus extraction cut is also landed:

- `C:\dev\Powershell\nodus\include\edge_reader_registry.h` separates exact,
  thread-safe external reader-slot accounting from the canvas/module
  `ThreadManager`;
- `ThreadManager` delegates its existing reader API to one owned registry, so
  existing callers retain their ABI and unrelated runtime contexts do not
  share an edge-id namespace;
- the FIFO's internal atomic reader table remains authoritative; the registry
  is a coordinator mirror, not another queue;
- slow-reader removal now exposes the surviving reader's real sequence instead
  of retaining a stale aggregate minimum;
- the standalone registry regression passes.

That honest runtime extraction is now complete for one edge:

- `EdgeTensorFifo` is physically self-contained in
  `include/edge_tensor_fifo.h`;
- `edge_tensor_fifo_transaction.h` is the single snapshot authority used by
  legacy tables and `nodus_runtime.dll`;
- Spectral prefers the runtime-only ABI, with an explicit legacy fallback;
- extracted/legacy snapshot bytes and reader traces match exactly;
- native ABI, overflow rejection, two-reader backpressure, and the golden
  camera/Turing rollback transaction pass.

The larger Nodus sparse graph/process manager is not extracted yet.

Native transaction widening has begun with a real idle-only T4 checkpoint.
`RayPipelineWaveCheckpoint` copies/restores arena field blocks, spectral lanes,
progress, link generations, and boundary telemetry and rejects topology drift.
`NativeWaveStateParticipant` adapts it to `ManagedProcessState`, and a native
test proves deterministic restore/replay. This is deliberately not called a
complete pipeline checkpoint: ray-tracer UV/illumination state, BDPT
pending/stash vectors and counters, sensor accumulators, legacy live-ray
pools, and GPU-resident buffers still need their own participants or
generation/copy-on-write checkpoints.

CPU pipeline queues are now the second native layer: generic reusable queue
snapshots feed `RayPipelineQueueCheckpoint`, covering T1-T5 inputs, ordinary
outputs, wave exits, BDPT side-data queues, and the T5 channel at an idle
CPU-only frontier. `NativeQueueStateParticipant` exposes it to managed state.
The native replay test proves a rejected T4 traversal's `Q_out` product is
retracted before retry. Pending/stash vectors, accumulators, counters, and GPU
state remain outside this participant.

The exact covered/uncovered state ledger and safe widening order are in
`NATIVE_OPTICAL_TRANSACTION_COVERAGE.md`. Do not place a full native trace
inside a rejected scientific window until every enabled accumulator and GPU
product joins the landed wave and queue participants.

Read `VITRUVIAN_REPOSITORY_ATLAS.md` for the larger successor/predecessor map
before replacing any graph, time, tensor, compilation, rendering, or game-host
subsystem.

Nodus is expected to load Pluck's renderer pillars whenever its own rasterizer
is not chosen. The existing seam is now known: `pluck_render_graph.py` supplies
the engine-neutral `EnginePillar`/`MultiEngineRenderGraph` lifecycle, while
`globals_renderer.py` routes the established document, OpenGL, CPU raster, and
progressive ray-trace backends with independent cadence and nonblocking sticky
results. Treat Nodus as loader/runtime host for these pillars. Do not collapse
them into its rasterizer, double-tick host-driven pillars, or create a second
document submission stream.

## Read this first

This task is part of a larger effort to make the repository's optical engine a
first-class computational pillar operated through Nodus. The immediate goal is
not to build another solver, another graph engine, or an attractive
demonstration. It is to add the physically correct staged accumulation boundary
required for coherent optical fan-in.

The core idea is:

```text
transport/scatter
        ↓
coherent reception pool
        ↓
completion/closure
        ↓
complex field reduction
        ↓
material or optical operator
        ↓
all resulting scattering products
        ↓
next transport frontier
```

A material/interface must not necessarily react to the first crossing that
arrives. Contributions that describe one coherent incident state must be
assembled first. Only the completed incident state is passed through the
material/operator.

This document is self-contained, but the following documents define the
surrounding contracts and should be read before editing:

1. `OPTICAL_ENGINE_REPOSITORY_INTEGRATION.md`
2. `NODUS_OPTICAL_KPN_INTEGRATION.md`
3. `OPTICAL_TRANSPORT_GRAPH.md`
4. `COMPLEX_OPTICAL_OPERATOR_CONTRACT.md`
5. `WAVE_T4_REPLACEMENT_DECISION.md`
6. `OPTICAL_COMPONENT_ARENA.md`

## Architectural position

### Camera program owns the electromechanical time budget

The coherent reception pool, optical arrival timestamps, solver epochs, Nodus
ready scheduling, and a host/display frame tick are not the camera clock.

Pluck time is meta-organization and game/application time. It may cause a new
scientific state request to be authored, cancelled, superseded, displayed, or
given more compute resources. It is not simulation time and must never be
converted implicitly into an optical or camera integration step.

The intended end-product protocol is:

1. A simulation manager receives an immutable state request with a requested
   observation time or interval, scene revision, camera program, requested
   products, and scientific accuracy policy.
2. The camera owns interpretation of that request as an electromechanical
   program. It produces a bounded budget of ordered slices describing shutter,
   aperture, mirror, flash/emitter, sensor-reset, sensor-integration, and
   sensor-read states.
3. The camera dispatches the energetic events and timetable transitions
   belonging to those slices.
4. The optical/wave solver owns physical-time advancement within the admitted
   slice budget. It processes exact event timestamps, propagation/group delay,
   coherent arrival windows, persistent field state, and declared delayed-loop
   horizons.
5. Slice completion returns to the camera. The camera decides whether the next
   electromechanical state or sensor read is eligible.
6. The simulation manager observes progress/products and schedules resources;
   it does not reinterpret camera or optical time.

The camera's slice budget is therefore both a physical program and a scheduling
envelope. A future sensor may schedule reset, rolling exposure, charge
transfer, nondestructive reads, ADC windows, or readout rows as distinct
camera-owned states. The solver must not assume that every slice contains one
flash followed immediately by one full-frame sensor sweep.

This is a multirate system. Camera mechanics may evolve on microsecond to
second scales while optical carrier phase and propagation delay occupy much
smaller scales. Scientific execution must not obtain a common tick by taking
an LCM of those physical rates or march the entire camera exposure at an
optical carrier timestep. The camera supplies macro slice boundaries and event
budgets; the wave solver advances its persistent state through physically
necessary micro-events, analytic propagation, arrival timestamps, or declared
time-domain steps. Accuracy and residual policies determine resolution.

The thick-lens camera design already defines the camera-owned time contracts in
`camera_software/exposure_timing.py`:

- `CameraExposureScheduler` authors deterministic exposure frames and slices;
- `CameraTimeline` turns an exposure into ordered flash and sensor states;
- `ExposureBarrier` enforces flash-materialization prerequisites before sensor
  submission;
- `SceneCameraClock` / `SceneCameraCoordinator` bind the scene snapshot to the
  simulated interval observed by the camera.

These pieces must converge behind one authoritative camera-program process
before general Nodus execution. That process owns:

1. camera program identity and generation;
2. exposure frame and slice identity;
3. simulated `t0` / `t1`, independent of wall/display time;
4. the immutable scene and camera state for the slice;
5. the finite slice budget and its electromechanical state;
6. flash/energetic-event dispatch and confirmation that optical products
   materialized;
7. permission for sensor integration, transfer, read, and accumulation;
8. completion/cancellation/supersession of the slice;
9. advancement to the next scripted camera state.

Every optical reception key participating in a camera exposure must therefore
include compact camera-program generation, exposure-frame, and exposure-slice
identity. `arrival_epoch` remains physical optical time within that program
slice. `solve_epoch` remains numerical/refinement identity. Neither may replace
the camera slice.

The authority and request flow is:

```text
Pluck/game/meta time
    -> immutable scientific state request(time or interval)
        -> simulation manager
            -> camera electromechanical slice budget
                -> energetic events + shutter/sensor timetable
                    -> Nodus resource/process scheduling
                        -> wave solver physical-time advancement
                            -> optical arrival/reception/closure
                                -> camera slice completion/read eligibility
                                    -> published exposure product
```

Pluck's frame `tick()` is a host progress opportunity only. A compiled optical
component refresh rate or GraphSolver clock is a subordinate numerical cadence
only. Neither may advance the camera program, close an exposure, or select a
different scripted state.

Nodus owns execution readiness, queueing, backpressure, and resources. The
simulation manager owns request lifecycle. The camera owns its physical
electromechanical timetable. The wave solver owns optical physical-time
advancement. These are cooperating authorities, not aliases for one clock.

### Existing Turing multirate coordinator

Do not design the camera/wave time coordinator without studying the existing
implementation in:

```text
C:\dev\Powershell\Turing\src\common\dt_system
```

The proven design and behavior currently include:

- `SuperstepPlan` / `SuperstepResult`: a requested exact outer window,
  controller proposal, result, and next-step recommendation;
- `STController`: PI-smoothed proposals, CFL/error penalties, engine-provided
  limits, and conservative growth policy;
- `step_with_dt_control_used`: snapshot/restore rollback, rejected-step
  metrics, halved retries, and explicit persistent-failure reporting;
- `run_superstep`: exact landing on an outer time window using adaptive
  microsteps;
- nested `RoundNode` contracts and per-node attempted-dt/metric telemetry;
- engine registrations with localized controllers and causal timestep hints.

This maps well onto the camera contract:

```text
camera slice duration
    -> SuperstepPlan.round_max
wave solver preferred microstep
    -> SuperstepPlan.dt_init / dt_next
optical accuracy and conservation diagnostics
    -> engine-specific Metrics/Targets
persistent wave state checkpoint
    -> snapshot/restore
failed scientific advance
    -> rollback + smaller-dt rerun
accepted exact slice landing
    -> camera slice completion
```

This is not presently a standalone timestep library. Even `dt.py` and
`dt_controller.py` import Turing's `AbstractTensor` and use its tensor
construction, comparison, logarithm, exponential, clamp, and type-restoration
semantics. The graph-facing layer additionally depends on Turing's
`StateTable`, state-table archive/integrator, realtime allocator, NetworkX
process adapter, and Transmogrifier ILP scheduler.

Architectural ownership is resolved: Turing is the universal computational
accounting substrate below Nodus. Its `AbstractTensor`, state accounting, and
managed-dt system are intentional cross-engine dependencies. Nodus owns graph
authorship, process execution, queues, and resources while consuming Turing
accounting/time services. The camera and optical engine consume both through
explicit adapters and retain authority over their physical semantics.

Do not copy a few Python files and call that an extraction. Do not fork the dt
controller into the optical repository or reimplement it privately in Nodus.
Finish and qualify the real Turing dt system, then expose a narrow stable
adapter that preserves its tensor/state ownership without placing Python object
traffic in the native optical hot path.

The Turing worktree is separate and dirty; its cleanup must be an intentionally
scoped cross-repository tranche that preserves unrelated changes.

The scoped Turing tranche now repairs those migration seams:

- `src.common.ManagedTimeRuntime` is the narrow revisioned absolute-time
  boundary;
- `TimeWindowRequest` carries request identity, generation, absolute start/end,
  initial microstep, and exact authored event times;
- stale, replayed, reordered, and discontinuous requests are rejected;
- scientific `MetaLoopRunner` execution preserves nested adaptive
  `RoundNode` supersteps instead of flattening them into one scheduler pass;
- rejected microsteps restore graph state, engine checkpoints, controller
  state, and the shared `StateTable`;
- terminal failure restores the complete requested window;
- named `Metrics.error_channels` / `Targets.error_limits` support optical
  residuals without disguising them as fluid errors;
- result telemetry retains attempted and accepted timesteps, rejection count,
  and exact event-boundary landings;
- the bisection solver now evaluates candidates transactionally against an
  explicit `StateTable` and commits only the selected timestep;
- causal-ceiling violations reject without advancing in scientific mode, while
  explicitly approximate realtime mode reports clipping and time slip.

The camera adapter should translate each admitted camera-program slice into a
`TimeWindowRequest`. It should use camera program generation as the Turing
generation, use a monotonically increasing slice request ID, put shutter/flash/
sensor transitions in `event_times`, and treat `TimeAdvanceReport` completion
as permission to advance the camera state machine. The wave solver remains the
managed state's physical advance callback and owns its optical error channels.

Turing's implementation notes and remaining native-state limitations are in:

```text
C:\dev\Powershell\Turing\docs\managed_time_runtime.md
```

Realtime mode remains intentionally non-scientific and must not be used as the
camera exposure completion authority.

### Nodus owns general execution

Nodus owns:

- graph authorship and composition;
- typed FIFOs;
- ready-frontier scheduling;
- process lifecycle;
- predecessor/completion credits;
- backpressure;
- quiescence;
- persistence;
- cross-engine resource coordination.

The relevant implemented Nodus machinery is in:

- `C:\dev\Powershell\nodus\include\inl\table_abi_core.inl`
- `C:\dev\Powershell\nodus\include\inl\table_abi_edge_io.inl`
- `C:\dev\Powershell\nodus\include\table_abi.h`
- `C:\dev\Powershell\nodus\include\thread_manager.h`
- `C:\dev\Powershell\nodus\src\thread_manager.cpp`

Do not infer Nodus capability solely from its stale preliminary FIFO document.
Its `EdgeTensorFifo` and lock-free publication/consumption paths exist.

Do not modify the Nodus worktree casually. At this handoff it has substantial
unrelated changes. A later Nodus integration should extract a runtime-only
target instead of linking its complete UI/canvas/SDL stack.

### The optical engine owns physical reduction

The optical engine owns:

- frequency and spectral-lane semantics;
- coherence identity;
- transverse basis and mode compatibility;
- Jones fields;
- optical path and group delay;
- complex accumulation;
- material/interface operators;
- power conservation;
- solve accuracy and residual criteria;
- zero-delay and delayed-cycle policies;
- persistent ray/field state.

Nodus may deliver tokens and completion credits to the reception process. It
must not decide whether two optical contributions are coherent or how their
fields are transformed into a shared basis.

### Current hosts are migration seams

Today Pluck's Python frame loop calls:

```text
MultiEngineRenderGraph
    -> OpticalEnginePillar.tick()
    -> OpticalEngineBackend.submit()/poll()
```

That is a working host adapter, not the final general process runtime. The
coherent reception implementation must be independently usable by the optical
pipeline now and later callable as a Nodus process.

The same file also defines `ExternalEnginePillar`, whose `tick()` intentionally
does nothing because the Pluck host calls that engine at its established point.
That is the lowest-impact bridge for document/OpenGL/ray renderers: Nodus
selects and accounts for the process, while the adapter preserves existing
renderer ownership until a backend is deliberately converted to graph-driven
execution. `GlobalChannelDispatcher` remains the concrete backend router and
drop-if-busy/latest-result policy, not functionality to duplicate in Nodus.

## Current implemented frontier

The current uncommitted optical tranche already provides:

- typed optical scattering products;
- geometric length, OPL, group delay, and phase-reference contracts;
- branch, join, and all-edge SCC schedule metadata;
- explicit schedule distinctions among:
  - ready-frontier work;
  - timestamped edges;
  - delayed cyclic work;
  - zero-delay SCC work;
- a fixed 96-byte `OpticalBranchToken`;
- `state_handle` plus `state_generation`;
- native persistent T4 field fan-out;
- Jones operators on exact rigid field links;
- tested fan-out power conservation;
- tested link-order invariance;
- native rejection of fan-in and cycles;
- component-level T4 execution locks preventing opposing lock inversion;
- preservation of the cached linear fast path for nonbranching components.

The native fan-out test verifies:

```text
amplitude 0.6 -> power 0.36
amplitude 0.8 -> power 0.64
total power 1.00
```

The current native authority rejects fan-in because coherent accumulation does
not exist yet. Do not remove that rejection until the new accumulator is
installed and qualified.

## Important files

### Python contracts and graph compilation

- `camera_software/optical_transport_contracts.py`
  - `OpticalScatteringProductSpec`
  - `OpticalTimingSpec`
  - `OpticalAccuracySpec`
  - `OpticalSolveReport`
- `camera_software/optical_transport_graph.py`
  - `OpticalTransportGraphSpec`
  - `CompiledOpticalSchedule`
  - branch/join/SCC compilation
  - `install_optical_graph()`
- `camera_software/complex_optical_operators.py`
  - transverse basis
  - Jones algebra
  - field-interface compilation
- `camera_software/optical_components.py`
  - component ports and engine contracts

### Native pipeline

- `csrc/include/optical_branch_abi.h`
  - fixed scheduling token
- `csrc/include/ray_pipeline.h`
  - `WaveArena`
  - `WaveFieldLink`
  - persistent state/snapshots
- `csrc/kernels/ray_tracer.cpp`
  - wave-context link registration
  - arena transfer
  - branch traversal
  - T4 completion/publication
- `csrc/bindings/pybind_kernels.cpp`
  - Python/native inspection boundary

### Current tests

- `tests/test_optical_transport_graph.py`
- `tests/test_wave_t4_pipeline.py`
- `tests/test_optical_components.py`
- `tests/test_complex_optical_operators.py`
- `tests/test_complex_transport_schema.py`
- `tests/test_graph_solver_delays.py`
- `tests/test_routing_layers.py`

## What is a reception pool?

A reception pool is persistent, bounded state associated with one compiled
optical receive port or join process.

It receives contributions identified by a physical accumulation key. The
minimum conceptual key is:

```text
engine state generation
destination node
destination port
scattering product
frequency or exact spectral lane
coherence identity
arrival epoch/window
transverse basis or compatible mode
solve epoch
```

Some of these values may be represented indirectly by a compiled handle. Do
not put arbitrary strings or dynamic Python objects into the native hot key.

Contributions within one physically compatible coherent key add as complex
fields:

```text
E_total = sum(E_i)
```

Contributions with different coherence identities do not add as amplitudes.
They remain separate coherent modes and may be combined only through derived
intensity, Stokes, detector, or statistical products.

The reception pool is not:

- a general-purpose message broker;
- a Python dictionary of field arrays;
- a per-pixel heap object;
- a replacement for Nodus FIFOs;
- a global frame barrier;
- permission to accumulate different frequencies into one Jones vector;
- permission to silently resample incompatible field grids or bases.

## Staged execution semantics

### Acyclic deterministic graph

For a compiled acyclic graph, pool closure can be exact.

The compiler knows the possible predecessor products for a receive port.
Every predecessor sends:

1. zero or more contribution tokens;
2. one completion credit for the relevant solve key.

The pool closes when:

- every expected predecessor has completed;
- every accepted contribution has been reduced;
- no outstanding queue item for that key remains.

Arrival order must not affect the physical result beyond declared
floating-point tolerance.

After closure:

1. convert compatible contributions into the declared destination basis;
2. sum complex amplitudes;
3. measure input power and residual;
4. apply the material/parametric/field operator exactly once;
5. publish every named scattering product;
6. publish completion credits for downstream pools;
7. retire or recycle the closed pool generation.

### Stochastic ray batches

Stochastic path selection has different closure semantics.

A finite estimator batch may be accumulated with:

- selection PDF;
- transport weight;
- sample identity;
- coherence identity when meaningful;
- batch/epoch identity.

Closing a stochastic batch produces an estimate for that batch. Later batches
refine the estimate; they do not retroactively become a single deterministic
field event.

Do not mix a stochastic estimator batch and deterministic coherent fan-in
under an undocumented common reduction.

### Zero-delay cycle

A zero-delay linear SCC cannot be made correct by repeatedly cycling tokens.
The optical engine must assemble and solve the scattering system, for example:

```text
(I - S) x = b
```

The reception-pool work should expose the assembled port/operator data needed
by that later SCC process. It does not need to complete the exact SCC solver in
the first vertical slice.

### Positive-delay cycle

A delayed loop uses arrival timestamps and a declared termination policy:

- time window;
- maximum branch depth;
- residual-power threshold;
- explicit simulation horizon.

The pool closes one arrival epoch/window at a time. It must not wait for “all
future crossings,” which is not finite.

### Mixed zero/positive-delay SCC

A mixed SCC may require an algebraic solve inside one arrival epoch plus
time-domain evolution between epochs. The compiled schedule already reports
both zero-delay-cycle and positive-delay-edge facts. Do not collapse them into
one generic feedback mode.

## Proposed first implementation slice

Implement only acyclic deterministic coherent fan-in first.

The slice should support:

- a compiled join node with two or more deterministic predecessors;
- fixed-band and continuous-frequency lanes;
- compatible Jones S/P state;
- exact basis conversion;
- explicit predecessor completion credits;
- deterministic complex summation;
- one downstream material/operator invocation;
- power and unresolved-work reporting;
- no silent drops;
- no cycles;
- no implicit grid resampling.

Recommended calibration graph:

```text
coherent source
    -> lossless splitter
        -> path A: phase 0
        -> path B: configurable phase phi
    -> coherent reception pool
    -> detector
```

Expected detector behavior:

```text
phi = 0      -> constructive interference
phi = pi     -> destructive interference
phi sweep    -> sinusoidal intensity
```

Also test two distinct coherence identities. They must add in intensity rather
than interfere.

## Suggested native data design

Do not commit to these exact names without inspecting surrounding ABI style,
but preserve these properties.

### Cold descriptor

One compiled receive-port descriptor should contain:

- stable join/pool id;
- destination node/port ids;
- expected predecessor product ids;
- exact lane specialization;
- representation kind;
- basis/operator handles;
- pool capacity;
- closure policy;
- accuracy policy;
- storage offsets;
- generation.

### Hot contribution token

Reuse or extend the fixed scheduling sidecar by handle. A token should identify:

- pool/join destination;
- predecessor product;
- state handle and generation;
- frequency/lane;
- coherence;
- arrival/OPL;
- solve epoch;
- contribution validity;
- completion-credit versus field-contribution role.

Do not place an N-lane field into the token.

If `OpticalBranchToken` needs new flags, preserve:

- 96-byte fixed stride if practical;
- standard-layout and trivially-copyable guarantees;
- explicit ABI version negotiation;
- stale-generation rejection.

Do not repurpose fields ambiguously. If the current token cannot express a
completion credit cleanly, define a versioned role/flag contract rather than
encoding one in an impossible frequency or negative PDF.

### Persistent state

The pool state should be:

- cold-sized;
- solid and contiguous;
- aligned;
- owned by the optical pipeline;
- stable for one engine generation;
- reset/recycled without allocation;
- capable of exact lane specializations;
- addressable through compact handles;
- safe from hot hash-table lookup.

Candidate organization:

```text
pool descriptors
pool generation/state words
predecessor completion bitsets or counters
lane/coherence slot descriptors
complex S/P accumulation planes or compact complex-ray states
power/residual/accounting rows
```

Field-sized reception requires more care than complex-ray reception. Start
with the smallest representation that proves coherent join semantics without
lying about field-grid compatibility. If using persistent T4 fields, combine
only exactly matching grids and declared rigid basis maps.

## Determinism and numerical reduction

Complex floating-point addition is not bitwise associative. “Order
independent” therefore needs an explicit numerical policy.

For the first slice:

- assign a stable predecessor/product order at cold compilation;
- store contributions by predecessor slot;
- reduce in that stable order after closure;
- use complex128 CPU reference calculations;
- use complex64 only where the production field ABI requires it;
- compare shuffled arrival order against stable reduction output;
- report the tolerance used.

Do not use atomic complex accumulation and then claim strict reproducibility.
Atomics may become a fast explicitly nondeterministic preview specialization
later.

If many contributions accumulate into one scalar/Jones state, consider pairwise
or compensated reduction. For full fields, tree reduction may be preferable,
but its order must be stable for deterministic bake mode.

## Conservation and closure accounting

Every closed pool should report:

- accepted contribution count;
- expected predecessor count;
- completed predecessor count;
- input coherent power by mode;
- reduced field power;
- absorbed power;
- emitted output-product power;
- residual power;
- rejected incompatible contributions;
- stale-generation contributions;
- dropped contributions;
- unresolved products;
- maximum numerical error.

Scientific mode must fail on a nonzero unapproved drop.

Be careful with interference: the power of the coherent sum is not generally
the sum of input powers. Constructive/destructive cross terms are physical.
Conservation must be checked across the complete lossless scattering network,
including complementary output ports, not by demanding that one recombination
port retain the sum of all incident powers.

## Basis and compatibility rules

Before summation, contributions must agree on:

- frequency within the declared exact identity/tolerance;
- coherence identity;
- arrival window;
- field normalization;
- representation;
- transverse sample grid;
- physical port plane;
- polarization handedness;
- basis or an exact declared basis transform.

Use the repository convention:

```text
s x p = k
Jones column = [E_s, E_p]^T
```

Use existing `TransverseBasis`, `JonesOperator`, and rigid-field interface
machinery. Do not invent a second convention inside the pool.

If a contribution requires interpolation rather than an exact signed
permutation/basis operation, reject it until an explicit resampling operator is
authored and qualified.

## Interaction with native T4

Current T4 fan-out recursively transfers source fields to preallocated
destination arenas. Fan-in is rejected both by:

- `install_optical_graph()`; and
- native wave-context link registration.

The first coherent-pool implementation does not automatically justify removing
both checks.

Only permit native T4 fan-in after:

1. the graph compiler lowers the join to a reception process;
2. destination storage is not overwritten by each predecessor;
3. predecessor completion is explicit;
4. basis/grid compatibility is validated;
5. complex reduction occurs once after closure;
6. the destination marches only after the reduction;
7. reverse-direction behavior has a declared reciprocal contract;
8. conservation and arrival-order tests pass.

The connected-component execution lock prevents corruption/deadlock. It does
not make sequential writes a coherent sum.

## Interaction with Nodus

The initial optical implementation may use an in-repository reference frontier,
but its process surface must map directly onto Nodus:

```text
input EdgeTensorFifo(s)
    -> optical reception callback
    -> persistent optical pool state
    -> output EdgeTensorFifo(s)
```

Nodus integration should later provide:

- typed token edges;
- predecessor completion credits;
- ready scheduling;
- pool-process activation;
- backpressure;
- quiescence;
- cancellation/supersession by engine generation;
- accounting.

The optical callback provides:

- key validation;
- stale-generation rejection;
- basis conversion;
- coherent reduction;
- closure validation;
- scattering;
- optical reports.

Do not vendor Nodus code into `spectral-analyzer`. Do not create another
general thread manager while waiting for the runtime extraction.

## Ordered implementation plan

### 1. Re-audit current state

- run `git status --short`;
- inspect all current uncommitted optical changes;
- confirm the canonical native module is loaded from the repository root;
- read the related architecture documents;
- do not overwrite unrelated changes.

### 2. Freeze a reference contract

Add Python reference types for:

- reception key;
- predecessor declaration;
- contribution;
- completion credit;
- closed-pool report.

Define deterministic equality/ordering and serialization. Add validation tests
before native work.

### 3. Extend the compiled schedule

For each join:

- assign a stable pool id;
- list expected predecessor product ids in stable order;
- attach coherence/arrival/representation policy;
- distinguish deterministic coherent join from stochastic/incoherent merge;
- reject unsupported mixed policies.

Do not infer coherence policy from node names.

### 4. Build a CPU reference accumulator

Implement:

- generation validation;
- contribution staging by predecessor;
- completion credits;
- stable-order complex reduction;
- exact basis conversion;
- close-once lifecycle;
- accounting and solve report.

Use this as the correctness oracle.

### 5. Add the native persistent pool block

Cold-allocate descriptors, slots, counters, and complex state. No hot allocation
or dynamic solver construction.

Expose bounded inspection through pybind for tests. Do not make production
progress depend on Python readback.

### 6. Lower one acyclic graph join

Replace the current fan-in rejection only for a graph that has an installed
qualified reception descriptor.

Route every predecessor to its assigned slot. March or scatter the destination
only after closure.

### 7. Qualify deterministic interference

Test:

- constructive phase;
- destructive phase;
- phase sweep;
- shuffled arrival order;
- distinct coherence identities;
- fixed lanes;
- continuous one-lane and multi-lane cohorts;
- stale generation;
- missing completion;
- duplicate completion;
- queue/full/drop behavior;
- incompatible basis/grid rejection.

### 8. Define the Nodus adapter ABI

Without modifying the dirty Nodus tree, specify or implement the narrow adapter
needed to:

- register the token schema/version;
- bind pool process callbacks;
- round-trip contribution and completion tokens;
- observe backpressure/quiescence;
- reject stale state generations.

Actual Nodus edits should occur only in an intentionally scoped clean change.

### 9. Add cycles later

Do not add delayed or zero-delay cyclic execution merely because acyclic joins
work. They are separate acceptance gates and separate optical processes.

## Required tests and acceptance gates

Minimum acceptance for acyclic coherent fan-in:

1. Two equal in-phase amplitudes produce the expected coherent field.
2. Equal opposite-phase amplitudes cancel at the selected recombination port.
3. Complementary lossless output ports conserve total network power.
4. Shuffled arrival order produces the same result within declared tolerance.
5. Different coherence ids do not interfere.
6. Different frequencies do not enter the same complex sum.
7. Basis conversion agrees with the existing CPU Jones oracle.
8. Incompatible grids fail; no implicit resampler appears.
9. A pool cannot close before every expected predecessor credit arrives.
10. Duplicate contributions/completions follow an explicit error policy.
11. Stale state generation is rejected.
12. No scientific contribution is silently dropped.
13. Fixed lane widths `1,3,4,8,16,32` retain exact storage specialization.
14. Continuous multi-lane metadata survives.
15. Ordinary compact ray/SSBO strides do not grow.
16. Unrelated linear T4 chains retain their cached fast path.
17. Existing native and broader optical regression suites remain green.
18. Build uses `csrc_build`.

## Baseline validation commands

```powershell
cmake --build csrc_build --config Release --target _spectral_kernels -j 8

python -m pytest `
  tests/test_wave_t4_pipeline.py `
  tests/test_optical_transport_graph.py `
  tests/test_optical_components.py -q

python -m pytest `
  tests/test_complex_transport_schema.py `
  tests/test_complex_optical_operators.py `
  tests/test_vector_wave_adapter.py `
  tests/test_wave_t4_rigid_interface.py `
  tests/test_camera_build.py `
  tests/test_exact_camera_transport_contract.py `
  tests/test_camera_designer_bridge.py `
  tests/test_camera_station_gateway.py `
  tests/test_camera_designer_gpu_bench.py `
  tests/test_gpu_preview.py `
  tests/test_pluck_render_graph.py `
  tests/test_surface_scan_preview.py -q

python -m pytest `
  tests/test_graph_solver_delays.py `
  tests/test_routing_layers.py -q
```

At this handoff the corresponding groups passed:

- 94 focused optical graph/component/native T4 tests;
- 88 broader camera/station/preview/complex integration tests;
- 66 graph delay/routing tests.

Live prism-room and optics-bench visual acceptance remains outstanding.

## Non-negotiable invariants

1. The camera station must not own a tracer.
2. Fast preview is a backend mode, not a station renderer.
3. T4 uses the pipeline and owns persistent contiguous state.
4. Do not allocate or instantiate solver state in hot dispatch.
5. Do not widen every ordinary ray with N-lane complex data.
6. Preserve stable handles and explicit state generations.
7. Preserve exact lane widths `1,3,4,8,16,32`.
8. Preserve continuous-frequency semantics independently of lane width.
9. Physical apertures remain material geometry, not masks.
10. Parametric lenses remain authoritative wherever applicable.
11. A Jones field without a basis is incomplete.
12. Unpolarized light is not one Jones vector.
13. A scalar determinant is not a full optical tangent map.
14. A delayed cycle is not a zero-delay solve.
15. A component lock is not coherent accumulation.
16. A demo may not invent a private scientific engine.
17. Unsupported physics must fail explicitly.
18. Nodus owns general process scheduling; optical code owns physics.
19. Never silently drop scientific optical products.
20. Do not modify or clean unrelated dirty worktrees.

## Likely first code review questions

Before accepting an implementation, ask:

- What exact event closes the pool?
- How are missing and duplicate predecessor completions detected?
- Where is state generation checked?
- Which values form the coherence key?
- How are bases reconciled?
- What prevents cross-frequency accumulation?
- What is deterministic about reduction order?
- What is the power/conservation accounting domain?
- What happens on queue pressure?
- Which allocations occur after execution starts?
- Does any ordinary ray or SSBO stride grow?
- Can unrelated components still run concurrently?
- Does a live host need Python readback to advance?
- How does this callback map to a Nodus FIFO process?
- Which code still rejects fan-in, and why is removing it now safe?

If those questions do not have precise answers, the implementation is not
ready to replace the present fan-in rejection.

## Desired endpoint before returning to the optics bench

The first compelling integrated proof should be a small top-down laser table:

```text
coherent source
    -> physical aperture
    -> exact parametric lens
    -> beam splitter
        -> path A with mirror
        -> path B with variable OPL
    -> coherent reception/recombination
    -> sensor
```

It should publish:

- actual transported ray/field state;
- path and OPL;
- phase;
- intensity;
- coherence/pool state;
- predecessor completion;
- residual/error;
- graph frontier;
- resulting sensor product.

The picture is an acceptance product, not the architecture. The real success is
that the same compiled optical processes can run in a standalone calibration
host, through Pluck's optics bench, and eventually under Nodus without changing
their physics authority.
