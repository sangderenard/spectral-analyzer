# Nodus-side tool brief: optical KPN integration

Status: handoff brief for an agent working directly in `C:\dev\Powershell\nodus`  
Written from: `C:\Users\alber\Downloads\spectral-analyzer` (branch `nogodsnomasters`)  
Date: 2026-07-24

## Scope of this document

This is **not** an inventory of Nodus. Nodus's own capabilities, gaps, and
internal design are for the agent working inside that repository to assess
directly against its own source. This document states only:

1. what already exists and is verified on the spectral-analyzer side;
2. what was recently found to be fake/self-contained work masquerading as
   Nodus integration, and was removed;
3. the two concrete things this repository needs from Nodus next.

Read these primary documents before making design decisions; do not treat
this brief as a replacement for them:

- `NODUS_OPTICAL_KPN_INTEGRATION.md` — implemented Nodus facilities, present
  limits, required extraction list, optical token contract, adoption order.
- `OPTICAL_ENGINE_REPOSITORY_INTEGRATION.md` — repository ownership split,
  current execution path, target Nodus-driven process loop.
- `OPTICAL_TRANSPORT_GRAPH.md` — the optical physical graph's own port/
  execution-domain contract (T1/T2/T3/T4), which Nodus must schedule as
  opaque typed work and must not reimplement.
- `COHERENT_RECEPTION_POOL_HANDOFF.md` — the coherent-reception-pool
  architecture, camera/Turing/Nodus time authority split, and current
  transaction/checkpoint state.
- `COMPLEX_OPTICAL_OPERATOR_CONTRACT.md`, `OPTICAL_COMPONENT_ARENA.md`,
  `WAVE_T4_REPLACEMENT_DECISION.md` — the physics contracts each solver tool
  must respect (Jones operators, component registry, T4 band specializations).
- `NATIVE_OPTICAL_TRANSACTION_COVERAGE.md` — exact covered/uncovered native
  transaction state.

## What already exists and is verified (spectral-analyzer side)

- `csrc/include/optical_branch_abi.h` — fixed 96-byte `OpticalBranchToken`
  scheduling sidecar (handle + generation, node/product/lineage identity,
  coherence/sample/lane identity, arrival time, OPL, frequency, PDF, residual
  bound, validity/solve-mode flags). Never carries an N-lane complex field.
- `csrc/include/optical_reception_abi.h` — fixed 96-byte
  `OpticalReceptionToken` for the coherent-reception process (contribution vs.
  completion roles, generation-checked handle into `OpticalReceptionKey`
  storage).
- `csrc/include/nodus_optical_fifo.h` — a thin RAII wrapper over the real
  extracted `nodus_runtime_abi.h` C ABI, sized for `OpticalBranchToken`. This
  is genuine plumbing to the real transport, not a demo. It is currently
  unused in this tree (see "what was removed" below) but is the correct
  primitive for a *shared* graph edge, not a private per-node buffer.
- `camera_software/nodus_runtime_bridge.py` /
  `camera_software/optical_reception_tokens.py` — the Python qualification
  bridge and bounded behavioral oracle for the reception token, tested against
  the real Nodus C ABI in `tests/test_nodus_reception_fifo_integration.py`.
- Native T4 wave arena: persistent per-arena contiguous state, band
  specializations exactly `1, 3, 4, 8, 16, 32`, fixed-band and continuous-
  cohort modes, fan-out with tested power conservation and link-order
  invariance, native rejection of fan-in/cycles (rejection must stay until a
  real coherent accumulator lands).
- Native transaction/checkpoint participants: `RayPipelineWaveCheckpoint`
  (idle-only T4 field/arena/generation restore) and
  `RayPipelineQueueCheckpoint` (T1-T5 input/output/wave-exit/BDPT side queues),
  both adapted to `ManagedProcessState`. A real camera/Turing/Nodus test
  proves rejected-window rollback and exact commit for one FIFO edge plus
  this native state.
- `csrc/kernels/ray_tracer.cpp`: `spectral_wave_t4_tick(void* pipeline)` is a
  real, already-exported C ABI entry point — a genuine plugin DLL could
  `GetProcAddress` it and drive the T4 wave solver one tick at a time. This is
  correct groundwork and was left in place.
- Component/graph compiler layers that any Nodus tool node must defer to,
  never reimplement: `camera_software/optical_components.py` (loadable
  components — `aperture.iris`, `lens.default-camera`, `mirror.plane`,
  `pentaprism.finder` — each cold-compiling to T1/T2/T3/T4 artifacts),
  `camera_software/optical_transport_graph.py` (compiles component transport
  into typed T1-T4 nodes, named products, branch/join/SCC metadata),
  `camera_software/complex_optical_operators.py` (Jones operators, transverse
  bases, tangent maps).

## What was fake and has just been removed from this tree

An earlier attempt built three "Nodus ITool" DLLs
(`cohort_gate_tool.cpp`, `wave_solver_tool.cpp`, `field_edge_out_tool.cpp`
under the now-deleted `csrc/nodus_optical_tools/`) plus a loader smoke test
(`tool_load_harness.cpp`) and a self-looping FIFO inside
`ray_tracer.cpp`. All of it has been deleted or excised because:

- each of the three tools allocated its **own private** `OpticalBranchFifo`
  in `initialize()` instead of receiving a Nodus-owned shared graph edge —
  cohort_gate's outbox and wave_solver's inbox were two different objects
  that nothing ever connected;
- the host ABI's real graph-edge hooks (`edge_publish`/`edge_consume` in
  `optical_engine_host_abi.h`) were declared but never called by any tool —
  dead capability papering over the private-FIFO shortcut;
- the build deliberately skipped `REGISTER_TOOL`/`ToolRegistrar`, so Nodus's
  own scheduler had no way to discover these "nodes" at all;
- `tool_load_harness.cpp` loaded one DLL at a time with a null host and
  ticked it in a bare `for` loop from `main()` — proof that a DLL loads, not
  proof of graph execution;
- in `ray_tracer.cpp`, `RayPipelineState` owned a `wave_kpn_fifo` that a
  single call site published a token into and then immediately consumed from
  itself (writer id == reader id == 1, same object, same call), while the
  adjacent comment claimed "a live dependency, not a demo." It was a
  synchronous self-loop and provably a no-op on the ray's own fields.

None of this needs to be resurrected or referenced. It is recoverable from
git history if anyone ever wants to see the shape of the mistake, but it is
not a starting point. Do not build the next attempt by patching these files
back in — build fresh against the real Nodus graph/edge/registration APIs.

## Prong 1 — a way to drive a headless KPN graph from Python

The end state: an authored/compiled Nodus graph (nodes = registered ITool
instances, edges = real Nodus-owned typed FIFOs) must be:

1. **launchable with no GUI.** Nodus's current `ThreadManager`/`table_abi`
   stack is coupled to canvas/SDL/typography per
   `NODUS_OPTICAL_KPN_INTEGRATION.md`'s "Important present limits" section.
   Figure out (or build, if it does not exist) a runtime-only process
   entry point that: loads a serialized graph description, constructs and
   registers its tool nodes, wires its edges, and runs the scheduler to
   quiescence or to an external stop signal — without initializing any
   window, renderer, or canvas table. The one precedent for a runtime-only
   extraction is the single-edge `nodus_runtime.dll` described in that same
   document (table/canvas/SDL/rope/typography/PNG/stage-free). The same
   discipline needs to extend to whole-graph process execution, not just one
   edge.
2. **savable/loadable as a graph, not just hand-assembled in code.** There
   needs to be an authoring→save→headless-load path: author or compile a
   graph once (topology, node types, edge types/capacities/backpressure
   policy, per-node configuration), persist it, and have the headless
   runtime reconstruct the exact same graph from that saved form. Check
   what Nodus already has for graph serialization before inventing a new
   format — this repository does not know that state and is not assuming
   Nodus has nothing.
3. **exposed as one module with declared in/out ports**, so the entire
   compiled graph can be embedded as a single black-box unit (e.g. from
   Python, or from another process/engine) instead of requiring the caller
   to know internal topology. This means: a small number of named external
   input edges the host feeds tokens into, and named external output edges
   the host drains, with the internal graph doing whatever fan-out/fan-in/
   solving it needs in between. This is the same "opaque typed work" boundary
   `OPTICAL_ENGINE_REPOSITORY_INTEGRATION.md` already describes for the hot
   work loop — it needs to become an actual callable module boundary, not
   just a description.
4. **drivable from Python, running in C, with hooks fired back into Python.**
   Concretely: a Python extension (ctypes bridge like
   `camera_software/nodus_runtime_bridge.py` already does for one FIFO, or a
   proper pybind11 module) that can: create/load a headless graph process,
   push tokens onto its declared input ports, pull tokens off its declared
   output ports, and **register Python callables that the C runtime invokes**
   for events a synchronous poll loop would miss or delay — at minimum
   quiescence, per-node completion/product-ready, overflow/drop, and
   rejected/rolled-back transaction. Model the callback boundary the same
   way `OpticalEngineHost` in the (now-removed) `optical_engine_host_abi.h`
   modeled the engine→tool boundary: a stable, versioned, trivially-copyable
   C function-pointer table, no STL/C++ classes crossing the boundary. Do not
   put Python object traffic in the scheduler's hot path — hooks fire, they
   do not block the graph on Python's GIL for ordinary token flow.

Nothing found in this repository already provides (1)-(4) for Nodus as a
whole graph; only the single-edge FIFO ABI is confirmed extracted. Verify
that against current Nodus source before assuming any part is missing or
present — this repo's search only confirms what does *not* exist on the
spectral-analyzer side.

## Prong 2 — the mixed solver tools optical systems actually need

A real optical graph is not one solver; it is several different physics
domains cooperating through typed ports, per
`OPTICAL_TRANSPORT_GRAPH.md`'s "Execution domains" and
`OPTICAL_COMPONENT_ARENA.md`'s `--component-engine` set
(`ray`, `parametric`, `wave`, `hybrid`, `maxwell`). Each domain below needs to
become a **real, registered** Nodus tool that references the existing engine
implementation through a host vtable (the way `optical_engine_host_abi.h`'s
`OpticalEngineHost` was designed to work) — it must never reimplement the
physics and must never fabricate a private edge the way the deleted tools
did.

| Tool node | Physics domain | Existing implementation to reference (do not reimplement) |
| --- | --- | --- |
| T1 geometric discovery | ray/BVH transport boundary discovery | native ray pipeline in `csrc/kernels/ray_tracer.cpp` / `csrc/include/ray_pipeline.h` |
| T2 parametric segment | closed-form/compiled network transport (exact compound lens, propagation spans, mirrors/beam splitters, cached Jones/modal operators, rechannel/basis-change) | fused exact-conic payload + `camera_software/optical_transport_graph.py` T2 lowering |
| T3 material interaction | Fresnel/conductor/dielectric per-band scattering, BDPT emission | T3 material handlers in `csrc/kernels/ray_tracer.cpp` (`ray_material.comp.glsl` GPU mirror) |
| T4 wave arena | persistent complex-field propagation, band-specialized `1,3,4,8,16,32`, fixed and continuous-cohort modes | `WAVE_T4_REPLACEMENT_DECISION.md` implementation, `wave_t4.h`, `spectral_wave_t4_tick()` |
| Maxwell patch artifact | localized full-Maxwell scattering compiled to a cached operator | patch compiler described in `MAXWELL_PATCH_CONTEXT.md` (verify current status before assuming complete) |
| Cohort gate | release-conditional accumulator: exact-count (closed graph) or time-gap guard (open/unknown arrival order) | logic itself is sound (see git history of the deleted `cohort_gate_tool.cpp` for the reference algorithm); re-implement as a registered tool over a real shared edge |
| Coherent join / reception pool | frequency+coherence+arrival-keyed complex accumulation before a material may react | **not yet built** — this is the actual subject of `COHERENT_RECEPTION_POOL_HANDOFF.md`; the `OpticalReceptionToken` ABI and Python oracle exist, the process itself does not |
| Incoherent join | combine intensity/statistics without inventing phase | not yet built |
| Zero-delay linear SCC solve | direct `(I-S)x=b` or equivalent compiled solve for zero-delay cycles | `GraphSolver`'s existing SCC path per `OPTICAL_TRANSPORT_GRAPH.md`; native T4 currently *rejects* fan-in/cycles until this lands — do not remove that rejection prematurely |
| Delayed-cycle worklist | timestamped worklist within a declared time/residual window | not yet built as a Nodus process; conceptually scoped in `NODUS_OPTICAL_KPN_INTEGRATION.md` |
| Nonlinear/time-varying cycle | explicit time evolution | not yet built |
| Camera slice / Turing time-window source | electromechanical slice budget → `TimeWindowRequest` → gated advance | `camera_software/managed_time_bridge.py`, `camera_software/exposure_timing.py`, Turing's `ManagedTimeRuntime`/`SuperstepPlan` (see `COHERENT_RECEPTION_POOL_HANDOFF.md`) |
| Transaction/quiescence checkpoint participant | snapshot/restore of edge + solver + queue state on gate rejection | native `RayPipelineWaveCheckpoint` / `RayPipelineQueueCheckpoint` + `ManagedProcessState`; needs a Nodus-side participant that actually holds the graph quiescent during capture/restore |

Requirements that apply to every row above, without exception:

- **Register for real.** Use Nodus's actual `REGISTER_TOOL`/`ToolRegistrar` (or
  whatever the current equivalent is) so `ThreadManager`'s real ready-frontier
  scheduler discovers and drives the node. A tool that only exports
  `create_tool`/`destroy_tool` and gets loaded by hand is not a graph member.
- **Share edges, never own them.** A node receives its input/output edges
  from the graph (Nodus-owned `EdgeTensorFifo` instances), it does not
  construct its own inbox/outbox to stay "runnable in isolation." If a node
  genuinely needs to be unit-testable alone, test it by injecting a real
  Nodus edge object into it, not by giving it a private one at `initialize()`.
- **Carry the existing fixed tokens.** `OpticalBranchToken` (96 B) for
  scheduling/branch work, `OpticalReceptionToken` (96 B) for the reception
  pool. Neither ever carries an N-lane complex field or other pipeline-owned
  state directly — only a generation-checked handle into it.
- **Respect the SCC/timing taxonomy.** Ordinary branch-frontier work,
  timestamped acyclic edges, positive-delay cyclic worklists, and zero-delay
  SCC solves are four distinct schedules, not one generic tick. Nodus
  provides the scheduling primitives; the optical engine supplies which
  policy applies per compiled region.

## Acceptance bar (so this doesn't repeat)

Before calling any of the above "integrated," demonstrate — with a real,
runnable example, not a comment claiming it — at least:

1. two distinct registered tool nodes exchanging `OpticalBranchToken`s over
   **one** shared Nodus-owned edge, scheduled by `ThreadManager`'s actual
   frontier (not a hand-written loop calling `tick()` in sequence);
2. that same graph loaded and run to quiescence with **no GUI/canvas/SDL**
   initialized;
3. that same graph driven from **Python** — tokens pushed in on a declared
   input port, tokens observed coming out a declared output port, and at
   least one Python-registered hook fired by the C runtime (e.g. on
   quiescence).

If any of those three cannot be shown concretely, it is not done yet —
say so plainly rather than approximating it with a private FIFO or a
loader smoke test.
