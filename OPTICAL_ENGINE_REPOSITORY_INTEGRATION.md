# Optical engine repository and process-loop integration

Status: architectural boundary plus current implementation inventory  
Primary repository: `C:\Users\alber\Downloads\spectral-analyzer`  
Nodus repository: `C:\dev\Powershell\nodus`

## Purpose

The optical engine is a first-class computational pillar intended to be
authored and operated through Nodus. Camera stations, the Pluck environment,
standalone calibration programs, automated render workers, and future optics
benches are clients of that same pillar. None of those clients may acquire a
private tracer or substitute a local scientific solver.

Nodus and the optical engine have different authorities:

- Nodus owns graph authorship, process lifecycle, typed queues, backpressure,
  ready-frontier scheduling, persistence, and cross-engine resource
  coordination.
- The optical engine owns components, materials, transport representations,
  optical operators, coherence, timing, conservation, numerical state,
  convergence, and result provenance.

Nodus schedules optical processes. It does not implement optical physics.
The optical engine uses Nodus process services. It does not grow a second
general execution manager.

## Repository ownership

### `spectral-analyzer`: optical physics and current hosts

This repository owns the canonical optical implementation:

- `camera_software/optical_components.py`
  defines loadable physical components, ports, controls, engine capabilities,
  material roles, and cold compilation.
- `camera_software/optical_transport_graph.py`
  compiles component transport into typed T1/T2/T3/T4 nodes, named scattering
  products, rigid field interfaces, branch/join metadata, and all-edge SCC
  regions.
- `camera_software/optical_transport_contracts.py`
  defines product timing, deterministic versus stochastic selection,
  accuracy, residual, dropped-work, and solve-report contracts.
- `camera_software/complex_optical_operators.py`
  owns transverse bases, Jones operators, canonical tangent maps, coherent
  source modes, and rigid exact-grid field interfaces.
- `csrc/kernels/ray_tracer.cpp` and `csrc/include/ray_pipeline.h`
  own the native spectral ray pipeline and persistent T4 wave arenas.
- `csrc/include/optical_branch_abi.h`
  defines the fixed-stride scheduling sidecar intended for a Nodus edge.
- `camera_software/optical_engine_backend.py`
  is the current engine-owned interactive execution service.
- `camera_software/gpu_preview.py`
  publishes immutable OpenGL texture products and performs optional
  asynchronous capture.

This repository also contains current hosts and adapters:

- `live_spectral_text_demo.py` is an optical-engine frontend despite its
  historical text-demo name.
- `wave_transform_visual_demo.py` is a calibration and inspection client of
  production kernels.
- `demo_pluck_gl.py`, `pluck_render_graph.py`, and
  `camera_designer_station.py` are the current Pluck/station integration.
- `camera_designer/` authors camera bodies, compound lenses, emitters, and
  exact parametric lens transport consumed by the engine.

Hosts submit immutable requests and display published products. They do not
own transport algorithms.

### `nodus`: eventual host and execution authority

Nodus supplies the general process substrate. The relevant implemented code is
not the stale preliminary FIFO document:

- `include/inl/table_abi_core.inl` implements `EdgeTensorFifo`;
- `include/inl/table_abi_edge_io.inl` and `include/table_abi.h` expose its
  operations;
- `include/thread_manager.h` and `src/thread_manager.cpp` implement scheduled
  and free-spinning module execution;
- the sparse graph, tensor abstraction, pools, scatter/gather, and dyadic
  kernels supply the broader dataflow substrate.

The current Nodus build couples these facilities to canvas, SDL, typography,
and other coordinator code. Direct optical integration should follow a
runtime-only extraction with a narrow C/C++ ABI. The optical repository must
not copy the manager or link the entire presentation stack merely to obtain
one FIFO.

Nodus currently has substantial unrelated working-tree changes. Optical work
must not edit, clean, reset, or vendor that tree as an incidental step.

### `fftfree`: source project and vendored snapshot

The original transform project is `C:\dev\Powershell\fftfree`. The authorized
snapshot in `third_party/fftfree` is the optical repository's reproducible
native CPU transform dependency.

It is a transform executor, not a wave solver. T4 retains ownership of:

- field state;
- angular-spectrum transfer functions;
- padding and absorbing boundaries;
- polarization;
- spectral lanes;
- pipeline scheduling.

Production builds should consume the reviewed vendored adapter rather than
depending on the mutable external checkout. Provenance is recorded in
`third_party/fftfree/VENDORED_FROM.md`. Licensing remains intentionally
unassigned until the project owner finalizes repository licenses.

### USD and PXR

USD is the scene/component exchange format. The repository's material system
remains canonical because it carries engine-specific physics such as
structural-colour/Maxwell artifact declarations that generic USD materials do
not fully express. USD material networks may be imported or exported through
declared translations.

PXR is an optional reference USD parser/runtime, not the definition of the
internal optical ABI. A host without PXR can still consume the repository's
resolved manifests and compiled component contracts. No shader-hot operation
depends on parsing USD.

## Three graph layers that must not be confused

### 1. Optical physical graph

`OpticalTransportGraphSpec` describes physical transport:

- exact parametric elements;
- material interactions;
- persistent wave regions;
- named reflected/transmitted/diffracted/detected products;
- optical path and group delay;
- coherence and stochastic-selection semantics;
- branches, joins, and cyclic regions.

The optical engine compiles and validates this layer. Nodus must treat its
physics processes and state handles as opaque typed work.

### 2. Current in-repository mathematical graph

`graph_solver.py` is presently used to validate and exercise the complex
network surface, including tensor operations and some SCC mathematics. It is
not the eventual outer application scheduler and must not become a second
Nodus.

Useful exact/vectorized optical mathematics may remain as optical process
implementations. General lifecycle, queue, and resource management should
move behind the Nodus runtime boundary.

### 3. Current Pluck pillar coordinator

`pluck_render_graph.py` is a small Python coordinator that lets Pluck register
presentation, acoustic, and optical pillars. It currently supplies request
coalescing and calls the optical backend's `submit()`/`poll()` pair once per
Pluck frame.

This is a working host adapter and migration seam, not the final general
execution engine. Nodus should eventually drive the process loops behind this
pillar contract. Pluck may retain the pillar-facing API so game code does not
need to know whether execution is local, Nodus-managed, or delegated.

## Execution today

The present interactive path is:

```text
camera station / Pluck / frontend
    -> EngineRequest
    -> MultiEngineRenderGraph
    -> OpticalEnginePillar.tick()
    -> OpticalEngineBackend.submit()/poll()
    -> camera manifest + exact compound lens + scene materials
    -> native RayTracer and optional installed T4 graph
    -> SurfaceScanPreview or ProjectorBackPreview
    -> RayPipelinePreviewBridge
    -> PreviewProductRegistry
    -> GLPreviewCompositor / asynchronous capture
```

Important current limitations:

- `OpticalEngineBackend` supports `fast_preview` only.
- A submitted revision currently builds a new tracer; durable session/cache
  reuse across compatible revisions is not complete.
- `OpticalEnginePillar` marks a request complete after the first published
  generation. Continuous refinement requires a declared persistent-job policy
  rather than relying on an accidental endless poll.
- Pluck's Python frame loop, not Nodus, currently calls `tick()`.
- Native T4 supports acyclic field fan-out but intentionally rejects field
  fan-in and cycles.

These are migration facts, not permissions to add station-owned alternatives.

## Target Nodus-driven process loop

Nodus should operate a coarse engine/process loop, not schedule each shader
instruction or ordinary ray bounce as a generic graph node.

### Cold revision loop

1. A host publishes an immutable request containing bench identity, scene
   revision, camera/component manifests, requested products, accuracy, and
   backend requirements.
2. Nodus applies latest-revision supersession only where the request contract
   permits it. Accepted scientific work is never silently discarded.
3. An optical compile process resolves USD/manifests, canonical materials,
   exact parametric elements, T4 regions, products, timing, and required engine
   specializations.
4. A resource process acquires or reuses the compatible long-lived optical
   session: GL owner, compiled shaders, SSBO arenas, scene/material caches,
   FFT plans, T4 state, and output products.
5. The compiled revision is atomically published to the session. Hot state is
   not resized while work is in flight.

### Hot work loop

1. Nodus places a fixed `OpticalBranchToken` or a coarse job token on a typed
   edge.
2. The target optical process resolves `state_handle` into pipeline-owned
   contiguous state.
3. T1/T2/T3/T4 execute through their existing specialized pipeline machinery.
   Nodus does not unpack N-lane fields or reproduce the shader pipeline.
4. Acyclic scattering products publish new fixed tokens to the ready
   frontier. Queue capacity, backpressure, and quiescence are Nodus concerns;
   selection PDF, power, coherence, and timing are optical concerns.
5. A presentation process publishes immutable shared-texture descriptions.
   Display consumers do not force field readback.
6. A capture process may asynchronously retain selected generations. No file
   I/O sits in the transport hot path.

### Join and cycle processes

The optical engine supplies domain processes selected by the compiled SCC
contract:

- coherent join: accumulate by frequency, coherence, arrival, basis, and
  product identity before continuing;
- incoherent join: combine intensity/statistics without inventing phase;
- zero-delay linear SCC: solve the compiled scattering system directly;
- delayed cycle: process timestamped tokens within a declared time or
  residual-power window;
- nonlinear/time-varying cycle: explicit time evolution.

Nodus invokes and schedules these processes, provides their queues, and
observes quiescence. It does not choose the coherence key or stopping
criterion.

## Data boundaries

### Host request

`EngineRequest` is immutable, revisioned, content-addressable host intent. It
is suitable for Pluck, a standalone frontend, an automated worker, or a future
Nodus adapter.

### Physical graph

`OpticalTransportGraphSpec` is the cold physical execution contract. It is not
a per-frame UI graph and not a tensor FIFO payload.

### Scheduling token

`OpticalBranchToken` is a fixed, aligned 96-byte trivially copyable record.
It carries:

- state handle;
- state generation for stale-handle rejection;
- lineage/coherence/sample/lane identity;
- node and product identity;
- high/low arrival time and OPL;
- continuous frequency;
- selection PDF;
- residual-power bound;
- validity/solve flags.

It never carries the N-lane field itself.
The token's fixed layout version is negotiated in the typed edge/schema
descriptor rather than repeated in every token.

### Persistent optical state

Ray sidecars, source/operator tables, wave arenas, FFT workspaces, and compiled
operators are solid engine-owned blocks with stable offsets. A Nodus token
references them for the lifetime of one published engine generation.

Handle generation and stale-handle rejection must be explicit before tokens
cross a persistent or interprocess boundary.

### Presentation product

`PreviewTextureProduct` describes an immutable texture generation and its
provenance. The GL share group provides the zero-copy path inside one process.
Cross-process deployment will require an explicitly selected shared-resource
or staged-transfer mechanism; a raw GL name is not portable across arbitrary
processes.

## Scheduling semantics corrected during the branch audit

The compiled optical schedule distinguishes:

- `requires_branch_frontier`: an acyclic branch or join needs ready work;
- `has_timestamped_edges`: at least one product carries positive group delay;
- `requires_cycle_solver`: at least one all-edge SCC is cyclic;
- `requires_timed_worklist`: a cyclic SCC contains positive-delay transport;
- `zero_delay_cycle_region_ids`: SCCs containing an algebraic zero-delay
  cycle.

A delayed acyclic edge does not by itself require a timed worklist. A
zero-delay cycle cannot be solved by repeatedly ticking it as if it were a
delay. A mixed SCC may require both an algebraic zero-delay treatment and
time-evolution policy.

Native T4 link registration enforces the subset it can execute today:

- forward fan-out is permitted;
- each destination has one forward producer;
- cycles are rejected;
- compatible grid/lane/basis rules remain mandatory.

The graph installer and native C/C++ authority now enforce the same boundary.
Each connected field component also receives one cold component-level
execution lock. A traversal may hold a source arena while it transfers into a
destination, so independently locking arenas could otherwise deadlock under
opposing traffic. Unrelated field components retain independent locks and can
run concurrently. Components without branching retain the original linear
cached traversal even when some other component branches.

This serialization prevents state corruption; it is not a substitute for
coherent counter-propagating accumulation. A deterministic experiment that
injects mutually coherent fields from several ports must wait for the declared
join/SCC process rather than relying on sequential arena mutation.

## Correctness audit at this integration cut

The audit corrected four concrete boundary problems:

1. Native link registration previously accepted a fan-in or cycle and failed
   only later when the pipeline was built. Registration now rejects unsupported
   topology immediately; construction retains a defensive validation pass.
2. The graph schedule previously called every branch/delay/cycle a "timed
   worklist." It now distinguishes ready-frontier, timestamp, delayed-cycle,
   and zero-delay-SCC requirements.
3. SCC delay telemetry previously included edges leaving the SCC. It now
   measures internal edges only and separately identifies internal zero-delay
   cycles.
4. Arena-local locks alone permitted lock-order inversion under opposing
   traversal. Connected components now share a cold execution lock, while
   unrelated components remain concurrent and unrelated linear chains retain
   their cached fast path.

Validated at this cut:

- native extension rebuilt from `csrc_build`;
- 94 optical graph/component/native T4 tests passed;
- 88 broader complex-operator, exact-camera, station, preview, surface-scan,
  and Pluck-pillar tests passed;
- native fan-out conserves power and is invariant to link registration order;
- stochastic products require a selection PDF;
- deterministic products reject a selection PDF;
- dropped work and excess final residual fail the solve-report contract.

Not yet correctness-qualified:

- coherent fan-in;
- deterministic mutually coherent multi-port injection;
- zero-delay or delayed optical cycles;
- `OpticalBranchToken` round-trip through an extracted Nodus FIFO;
- long-lived tracer/cache reuse across revisions;
- live prism-room and optics-bench visual acceptance after the branch changes.

## Integration phases

### Phase A: qualify the current optical pillar

- keep the current Pluck adapter;
- complete live prism, camera-station, projector, and aperture acceptance;
- measure tracer construction, shader compilation, scene upload, dispatch,
  publication, and capture separately;
- retain the native fan-out conservation/order tests.

### Phase B: extract Nodus runtime services

- create a runtime-only Nodus library/target;
- expose typed sparse graph, `EdgeTensorFifo`, tensor handles/pools,
  ready-frontier scheduling, SCC policy callbacks, quiescence, and accounting;
- add ABI/version negotiation;
- do not pull UI or SDL dependencies into the optical engine.

### Phase C: token/FIFO qualification

- round-trip `OpticalBranchToken` without reinterpretation or stride changes;
- test full, backpressure, cancellation, restart, stale handle, and explicit
  lossy-preview behavior;
- compare ordered and shuffled acyclic execution;
- require identical physical products within declared numerical tolerances.

### Phase D: Nodus-driven optical process loop

- adapt `OpticalEnginePillar` to submit into the Nodus runtime;
- retain the existing frontend/station request and product APIs;
- reuse long-lived optical sessions across compatible revisions;
- allow Nodus to drive coarse process progress independently of Pluck's frame
  rate;
- publish progress and textures back through the existing registry.

### Phase E: return to the optics bench

- load canonical camera/component manifests into the bell jar;
- let Nodus author and schedule the optical process graph;
- expose fast preview, exact parametric paths, T4 slices, accumulation,
  residual, and queue/SCC diagnostics as selectable layers;
- keep the station a request/editor/presentation surface;
- prove a branched laser-table or pentaprism chain before claiming general
  mixed ray/wave composition.

## Acceptance gates

Nodus-driven optics is ready for the bench only when:

1. the same request produces the same compiled physical graph in standalone
   and Pluck hosts;
2. no station/frontend constructs a private tracer;
3. ordinary compact ray and exact lane strides remain unchanged;
4. fixed and continuous modes retain their distinct semantics;
5. branch power, PDF, coherence, OPL, and arrival metadata survive FIFO
   round-trip;
6. rejected/full/dropped work is reported and scientific work is never
   silently overwritten;
7. zero-delay and delayed cycles select different declared processes;
8. GL publication does not impose synchronous readback;
9. compatible revisions reuse the GPU owner and immutable caches;
10. shutdown, supersession, and stale handles cannot mutate a newer engine
    generation;
11. focused numerical tests and live optics-bench acceptance both pass.

## Build and validation

Native optical code is built from this repository's actual build directory:

```powershell
cmake --build csrc_build --config Release --target _spectral_kernels -j 8
```

The mutable external Nodus and fftfree checkouts are not production build
directories for `spectral-analyzer`. Nodus integration should first be tested
through its extracted runtime target; FFT execution uses the reviewed vendored
snapshot.

Related documents:

- `COHERENT_RECEPTION_POOL_HANDOFF.md`
- `NODUS_OPTICAL_KPN_INTEGRATION.md`
- `OPTICAL_TRANSPORT_GRAPH.md`
- `COMPLEX_OPTICAL_OPERATOR_CONTRACT.md`
- `WAVE_T4_REPLACEMENT_DECISION.md`
- `WAVE_ENGINE_ACTION_PLAN.md`
- `OPTICAL_COMPONENT_ARENA.md`
- `MAXWELL_PATCH_CONTEXT.md`
