# Vitruvian repository atlas

Date audited: 2026-07-23

This project cannot be understood one repository at a time. The repositories
are a lineage: work commonly stopped when one layer exposed a problem whose
solution became the organizing principle of the next repository. A broken or
unfinished host therefore does not imply that its central mechanism is absent.
Before implementing a new manager, graph, tensor abstraction, compiler,
timestep controller, queue, renderer, or game host, search this atlas and the
listed repositories.

This is a road map, not a claim that every historical checkout is currently
healthy. Entries distinguish verified mechanisms from archaeological leads.

## The body

```text
game / field experience / camera and scientific instruments
    spectral-analyzer optical pillar
        Nodus authored process graph and KPN runtime
            Turing computation, accounting, managed time, translation
                AbstractTensor backend-neutral values
                    Transmogrifier symbolic graph construction/lowering
                        native CPU/GPU/GLSL and, eventually, physical media
```

This is not a strict package dependency stack yet. It is an architectural
direction and an ownership map. In particular:

- the camera owns its electromechanical program and physical slice budget;
- Turing accounts for exact scientific time, retries, and rollback;
- Nodus owns graph execution, typed queues, readers, backpressure, and
  resources;
- each scientific engine owns its physical state, convergence, and products;
- Pluck/game time is a host opportunity, never camera or solver time.

## Active integration spine

### Spectral Analyzer

Path:

```text
C:\Users\alber\Downloads\spectral-analyzer
```

Role: scientific continuous-spectrum optical engine, camera program, coherent
reception, light-table experiment description, emitters, sensors, lens chains,
and the current thick-lens development host.

Start here:

- `OPTICAL_ENGINE_REPOSITORY_INTEGRATION.md`
- `NODUS_OPTICAL_KPN_INTEGRATION.md`
- `CAMERA_SIM_TIME_EXPOSURE_PLAN.md`
- `COHERENT_RECEPTION_POOL_HANDOFF.md`
- `OPTICAL_TRANSPORT_GRAPH.md`
- `camera_software/exposure_timing.py`
- `camera_software/coherent_reception.py`
- `camera_software/managed_time_bridge.py`
- `csrc/include/optical_branch_abi.h`

Verified on 2026-07-23:

- fixed 96-byte native optical scheduling token;
- fixed 96-byte coherent reception contribution/completion token backed by a
  bounded generation-owned rich-key table;
- camera-generation plus monotonic-causality identity;
- unordered job execution with canonical commit order;
- multi-reader causal release gate;
- bounded deterministic coherent reception;
- branch/join/delayed-cycle/zero-delay-SCC schedule distinctions;
- camera-to-Turing managed-time adapter.
- real two-reader reception-token transport through the Nodus DLL, including
  slowest-reader lossless backpressure.

Do not add a private optical graph manager or timestep controller here.

### Nodus

Path:

```text
C:\dev\Powershell\nodus
```

Role: graph authorship, typed port-to-port transport, independent readers,
backpressure, process readiness, lifecycle, persistence, and resource
coordination.

Nodus is also the eventual loader/host for Pluck renderer pillars when a graph
does not select Nodus's own rasterizer. Do not interpret "Nodus subsumes the
graph/runtime" as "rewrite every renderer inside Nodus." The Pluck document
renderer(s), OpenGL renderer, and ray tracer remain independently meaningful
execution pillars and should meet Nodus through loadable process/backend
contracts.

The current boundary has now been audited. `pluck_render_graph.py` defines the
engine-neutral `EnginePillar` contract (`submit`, `tick`, `describe`,
`shutdown`), `ExternalEnginePillar` for host-driven engines, and
`MultiEngineRenderGraph` for registration and scheduling. `globals_renderer.py`
is the existing backend router: it selects CPU/GL document rendering,
`BaseRasterizer`, the host GL renderer, or progressive `RayTracer`; gives 2D and
3D independent cadence; and uses bounded nonblocking CPU workers whose result
slot remains readable while a later submission is dropped. The shared
`DocRenderer` backend is deliberately reused rather than fed through a second
submission stream. These are the first Nodus loader-adapter targets. Preserve
their product ownership and host-driven execution points; do not add a parallel
renderer coordinator.

Start here:

- `include/inl/table_abi_core.inl` (`EdgeTensorFifo`)
- `include/inl/table_abi_edge_io.inl`
- `include/table_abi.h`
- `include/thread_manager.h`
- `src/thread_manager.cpp`

Verified on 2026-07-23:

- the FIFO is implemented, not a proposal;
- fixed typed shape/layout/dtype and backend storage;
- one bound writer and up to 64 independently sequenced readers;
- acquire/release publication;
- reject-full and explicit overwrite behavior;
- Kahn scheduling and parallel ready frontiers;
- the exact optical 96-byte token round-trips through the built DLL;
- publication remains blocked until the slowest lossless reader advances.

Current boundary: the FIFO transport is now extracted into
`nodus_runtime.dll`; the larger sparse graph/process manager remains packaged
with canvas/UI/SDL code. Nodus cycle ordering is not an optical SCC solver; the
optical engine supplies that policy.

The first dependency cut is now explicit in
`include/edge_reader_registry.h`: coordinator-facing reader slots and exact
per-edge minima no longer live inside the canvas-heavy `ThreadManager`.
`ThreadManager` owns and delegates to one registry per manager. This is only a
mirror for scheduling/diagnostics; `EdgeTensorFifo`'s internal atomic reader
table remains the authority for consumption and backpressure. Do not create a
second FIFO or a process-global edge namespace while completing the extraction.

`EdgeTensorFifo` now lives in a self-contained runtime header and its versioned
transaction utility is shared by both ABIs. Spectral defaults to the extracted
one-edge runtime. A parity test proves the extracted and legacy table paths
produce identical transaction bytes and lossless reader traces.

Nodus now exposes a separate quiescent transaction snapshot for managed
scientific rollback. It preserves FIFO configuration, storage, slot publication
tags, write sequence, bound writer, and every active reader key/sequence;
restore rejects configuration or reader-set changes and resynchronizes
the coordinator-facing registry through the legacy ThreadManager bridge. The
older persistence snapshot remains a different format and must not be
substituted. Capture/restore requires the process frontier to be quiescent.

Repository warning: the checkout has substantial unrelated dirty work and at
least one sparse-graph test currently crashes. Broad Nodus test results are not
a reliable map of which mechanisms are sound. Qualify the exact ABI seam being
adopted.

### Turing

Path:

```text
C:\dev\Powershell\turing
```

Role: universal computational accounting below Nodus: backend-neutral tensors,
state identity, managed scientific time, rollback/rerun metrics, graph
translation, compilation, and execution down to the cassette survival
computer.

Start here:

- `README.md`
- `MODULES.md`
- `docs/managed_time_runtime.md`
- `src/common/dt_system/time_runtime.py`
- `src/common/dt_system/dt_graph.py`
- `src/common/tensors/abstraction.py`
- `src/turing_machine/survival_computer.py`
- `src/compiler`
- `src/transmogrifier`

Verified on 2026-07-23:

- absolute revisioned `TimeWindowRequest`;
- exact authored event boundaries;
- adaptive microsteps, rejection, rollback, and rerun;
- nested scientific rounds and named error channels;
- transactional bisection;
- optional precommit gate which rolls back the entire exact-landed window if
  surrounding process/reception/readers are not quiescent;
- working compile → tape IR → cassette → `TapeMachine` survival-computer path;
- NumPy, Torch, JAX, and pure-Python `AbstractTensor` backends;
- graph translation/compiler tests are substantially operational.

Current boundary: correctness-first Python state copies do not yet cover every
native/GPU allocation. Native states need generation checkpoints or
copy-on-write handles. Turing's managed time is the real dependency; do not
copy its controller files into optics or Nodus.

Repository warning: this checkout is dirty. FluxSpring is presently broken in
`src/common/tensors/autoautograd/fluxspring/demo_spectral_routing.py` by an
incomplete temporary graph-memory edit. FluxSpring is not a prerequisite for
the camera/Nodus/Turing integration.

### Transmogrifier

Paths:

```text
C:\dev\Powershell\transmogrifier
C:\dev\Powershell\turing\src\transmogrifier
```

Role: convert described symbolic mathematics, including SymPy expressions,
into executable process graphs; schedule and lower those graphs toward Python,
C/CFFI, GLSL, and other backends.

The Turing copy is the more advanced line currently inspected. Landmarks:

- `graph/graph_express2.py` (`ProcessGraph`, `build_from_expression`)
- `graph/graph_deep_compiler.py`

Use this first as a cold graph-construction and lowering system. Do not force it
into the first optical hot-path bridge. Its importance is that an optical
experiment eventually need not remain hand-authored Python: symbolic program,
graph IR, process deployment, and backend compilation can converge here.

The standalone directory is historical and is not independently versioned as
a clean repository.

### AbstractTensor

Primary current path:

```text
C:\dev\Powershell\turing\src\common\tensors
```

Role: values and operations which survive translation across execution
backends. This is the route by which Python prototypes can eventually be
subsumed without making NumPy, Torch, GLSL, or native CPU representation the
architecture.

Start with `abstraction.py`. Treat accelerator backends individually: the
registry and several major backends work, while some experimental native
backends remain partial.

## Adjacent descendants and ancestors

### AMP

Path:

```text
C:\dev\Powershell\amp
```

Role: native signal-process graphs, oscillator/driver authority, continuous
time-base handoff, and low-latency physical/audio scheduling.

Road sign:

```text
docs/kpn_continuous_timebase_design.md
```

This is relevant whenever optical scheduling starts to conflate authored
physical authority, process cadence, and resampling. AMP contains a separate
physical-domain exploration of authority handoff and continuous KPN timing; it
should inform, not replace, Turing's scientific transaction boundary.

### Early LLM project field

Path:

```text
C:\Apache24\htdocs\AI Projects
```

This is a large archaeological field of early LLM-assisted experiments,
including `game`, `FontMapper`, geometry/visual experiments, neural systems,
stories, and interfaces. Do not bulk-modernize it. Search it when a supposedly
new presentation, game, ASCII, visual, or learned-mapping mechanism appears.

Verified landmarks:

- `FontMapper`: an evolutionary series of learned glyph/bitmap mappings,
  image-to-ASCII representation, font compatibility, rendering, and distributed
  task experiments. Later work is under `FontMapper\FM16`.
- `game`: early plot/game/runtime experiments including `game.py`,
  `renderer.py`, `objects.py`, `ngon.py`, `mappedtensor.py`, and
  `physics_engine.py`.

The possible separate “Adam” project and the exact current Pluck checkout were
not located in the bounded 2026-07-23 audit. Do not conclude they are absent;
search user project drives deliberately when those lineages become relevant.

## Current executable cross-repository seam

The first real bridge landed on 2026-07-23:

```text
CameraSliceJob
    -> CameraManagedTimeBridge
        -> Turing TimeWindowRequest
            -> adaptive exact managed advance
                -> optional process/reception/multireader commit gate
                    -> TimeAdvanceReport
                        -> camera may advance
```

The next transport seam also landed:

```text
rich OpticalReceptionKey
    -> bounded generation-owned local key handle
        -> 96-byte OpticalReceptionToken
            -> typed lossless Nodus FIFO
                -> independent readers + slowest-reader backpressure
```

The first complete transactional vertical slice is also qualified:

```text
camera-authored TimeWindowRequest
    -> Turing checkpoints solver + Nodus FIFO
        -> optical token publication and reader progress
            -> camera commit gate rejects held work
                -> solver, writer, storage, and reader cursors roll back
                    -> same request reruns
                        -> all readers drain
                            -> exact camera-time commit
```

Turing's commit gate is domain-neutral. It sees only approval or rejection.
Spectral Analyzer supplies the camera/optical meaning. A future Nodus adapter
will supply FIFO and process-quiescence meaning through the same gate.

Live qualification:

```text
camera window: [0.0, 0.2]
authored internal events: 0.05, 0.15
registered readers: wave, detector
first attempt: exact solver landing, gate rejection, whole-window rollback
second attempt after both releases: exact commit at 0.2
```

## Rules for continuation agents

1. Read this atlas and the four Spectral Analyzer integration documents before
   proposing architecture.
2. Search successor and predecessor repositories before implementing a general
   subsystem.
3. Separate a mechanism's health from its containing demo's health.
4. Preserve dirty worktrees. Make narrow changes and inspect overlapping diffs.
5. Keep ownership explicit: camera time, Turing accounting, Nodus execution,
   engine physics, host/game meta time.
6. Use Python implementations as behavioral oracles while migrating hot paths;
   do not delete them merely because a native seam exists.
7. Never silently drop a scientific product. Backpressure, bounded growth, or
   explicit failure are acceptable; accidental overwrite is not.
8. State and token generation must invalidate stale native handles.
9. Arbitrary job execution order is allowed; physical commit order and
   coherent reduction remain authored and deterministic.
10. Record what was verified, what merely exists, and what remains archaeology.

## Next road

The next narrow tranche is not another coordinator:

1. widen the landed one-edge `nodus_runtime` target to the sparse port graph,
   process callbacks, and deterministic ready-frontier manager;
2. bind a coarse optical process callback without moving field arrays through
   Python objects;
3. extend transaction coverage beyond the qualified FIFO edge, idle-only T4
   wave arena, and CPU T1-T5 queues to sensor/BDPT accumulators and GPU-resident
   state mutated inside a window;
4. prove shuffled single-machine and multi-worker execution against the Python
   causal/reception oracles;
5. add native/GPU state generations or copy-on-write checkpoints;
6. only then move graph construction/lowering toward Transmogrifier and
   AbstractTensor.

For renderer loading, adapt Nodus to Pluck's already-landed `EnginePillar` and
`MultiEngineRenderGraph` boundary. Preserve `GlobalChannelDispatcher` as the
current backend-selection and asynchronous-delivery implementation. First prove
one external pillar can be registered, selected, and polled through a Nodus
process without double-ticking its host-owned renderer; only then generalize
the loader.

The authoritative state-by-state rollback ledger is
`NATIVE_OPTICAL_TRANSACTION_COVERAGE.md`. Read it before registering a native
handle with `ManagedProcessState`; quiescence is not equivalent to coverage.
