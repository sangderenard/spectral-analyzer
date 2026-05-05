# Controls / Actions / Shaders / FIFO / Naive-Graph Subsystem

This document records the work consolidated into the root `controls.py` and
`naive_graph.py` modules during the recent session, then lays out an action
plan for importing the **torch-based `GraphSolver`** (the analytic driver's
generic-object-passing, cyclic-reduction graph solver) on top of the same
hierarchy, and finally collects notes on the cost/complexity of offloading a
graph computation to C through a Kahn Process Network (KPN) with a
bidirectional C↔Python action API.

The other (non-torch) solver in the workspace is intentionally **out of
scope** for this plan and is not considered anywhere below.

---

## 1. What was built

### 1.1 Canonical control hierarchy — [controls.py](controls.py)

A single, exclusive module through which any control anywhere in the
application is expressed and any action is registered/executed.

#### 1.1.1 OwnerScope and the five subordinate peer axes

Every owner (`GLOBAL` and each `OBJECT`; `ROOM` is purely organizational
and owns nothing) has the same five subordinate peer axes attached as
peer `ControlNode`s:

```
global  (OwnerScope.GLOBAL)
  ├─ controls/   (CONTROLS)   — KnobSpec / Panel descriptors only
  ├─ actions/    (ACTIONS)    — triangle-group action sites only
  ├─ shaders/    (SHADERS)    — custom shader nodes only
  ├─ inputs/     (INPUTS)     — FIFO edge endpoints (network in)
  ├─ outputs/    (OUTPUTS)    — FIFO edge endpoints (network out)
  └─ room/<id>/  (ROOM)
       └─ object/<id>/  (OBJECT)
            ├─ controls/   ├─ actions/   ├─ shaders/
            ├─ inputs/     └─ outputs/
```

The shader-vs-control hierarchy is unambiguous by construction: no axis
can carry the wrong kind of node, and no axis owns another.

#### 1.1.2 Producers (the only sanctioned way to attach nodes)

* `slider_defs_to_knobs(...)` / `knobs_to_object_nodes(object_id, ...)`
* `register_triangle_group_action(object_id, triangle_group, ...)`
* `register_shader_node(owner_id, shader_id, ...)`
* `register_input_endpoint(owner_id, endpoint_id, capacity=...)`
* `register_output_endpoint(owner_id, endpoint_id, capacity=...)`

#### 1.1.3 Action registry + dispatcher

* `ActionRegistry` resolves callable / script-path+hook-name pairs.
* `ActionDispatcher` runs a daemon worker thread; `enqueue_action(...)`
  is the single entry point. Falls back to synchronous dispatch when the
  worker isn't running. Exceptions in user code are caught so the worker
  never dies. Started before the main loop, stopped in `finally:`.

### 1.2 FIFO endpoints — `Fifo` + `FifoContext`

Each input/output endpoint owns a `Fifo` (bounded `deque` with its **own
RLock**, capacity 0 = unbounded; oldest-drop on overflow). Every shader
hook and the future compute coordinator reach FIFOs through a per-owner
`FifoContext`:

```python
fifos.input("radar").drain()
fifos.output("trace").push(payload)
```

Naming convention is intentionally simple: each FIFO is referenced by
its raw `endpoint_id` within its owner's scope; `inputs` and `outputs`
are separate dicts so the same name can appear on both axes without
ambiguity. `ControlGraph.fifo_context(owner_id)` builds the view.

### 1.3 Shader subsystem — bottom-up frame walk

* `ShaderSpec` carries `shader_id`, `targets`, `flip_slots`,
  `min_period_s` (cadence/rest), and four optional callables:
  * `pre_hook(spec, node, fifos, frame_index, dt) -> dict | None` —
    runs immediately before `run` on the same thread; returned dict is
    merged into the kwargs of `run` (sanctioned channel for passing
    extra buffers).
  * `run(spec, node, fifos, frame_index, dt, **extra) -> dict | None` —
    main-thread (so it may touch the GL context); returned
    `{target_id: payload}` is written into the matching `ShaderFlipBuffer`.
  * `post_hook(spec, node, fifos, produced, frame_index, dt)` — typical
    use is publishing finished payloads onto output FIFOs.
  * `compute_tick(spec, node, fifos, **kwargs)` — **never** called by
    the shader walker; reserved for the future computational-graph
    coordinator (fire via `dispatch_compute_tick(shader_id, **kw)`).
* `ShaderFlipBuffer(target_id, slots=2, ...)` — opaque ping/pong
  storage. `write()` rotates the slot, `latest()` reads the most recent.
* `ShaderFrameWalker.tick(frame_index, dt) -> ShaderFrameResult`:
  bottom-up post-order over every `SHADERS` axis (deepest objects first,
  then global). For each due shader, runs `pre → run → post` with the
  same `FifoContext`; populates `finalized_targets` (the set of global
  buffer ids the final draw must gate fragments against) and
  `flip_buffers` (latest payload per target — forwarded even when the
  shader is resting).
* Wired into [demo_pluck_gl.py](demo_pluck_gl.py) just before the final
  draw's `glClear`, fail-soft (a misbehaving shader can't kill the
  frame loop).

### 1.4 Naive graph controller — [naive_graph.py](naive_graph.py)

Backbone for any unoptimized inter-object network. Connections are edges
between the FIFO endpoints registered in `controls.py`.

* `Address(owner_id, endpoint_id)` (string `"owner/endpoint"` and
  2-tuple forms accepted).
* `Edge(src, dsts)` — single-source-to-many-destinations routing rule.
* `Message(payload, remaining: set[Address], deadline_monotonic?)` —
  dropped the moment its remaining-set is empty (last receiver picked
  it up) or its deadline passes.
* `NaiveGraphController.tick()` is the dict-based switch:
  1. Hand off any subnet whose handler is registered (a duty station
     can take over an entire subnet; handler errors are swallowed so
     they cannot kill the controller tick).
  2. Drain every output FIFO (skipping handled subnets), look up edges
     per source, build one `Message` per item addressed to the union of
     edge destinations. Output items on edges-less ports are drained
     anyway and counted as `unrouted` (no unbounded growth).
  3. Walk the inbox once: deliver each unexpired payload into every
     remaining recipient's input FIFO (per-`Fifo` lock keeps this safe
     across threads). Empty-`remaining` messages are dropped; partials
     are carried to the front of the inbox for next tick.
* `submit(payload, addresses, timeout_s=None)` — direct send for
  scripted / cabling sends that bypass output FIFOs.
* `register_subnet_handler(subnet_id, handler)` — exactly the "special
  duty station that takes over the role entirely" path.
* Singleton: `get_naive_graph_controller()`.

### 1.5 Thread-safety summary

* `ControlGraph` and `ActionRegistry` use a single RLock for hierarchy /
  registry mutation.
* `ActionDispatcher` uses a daemon worker thread + `queue.Queue`.
* Each `Fifo` owns its own RLock — the only lock needed for FIFO I/O.
* `NaiveGraphController` uses its own RLock only for registry/inbox
  mutation; FIFO I/O happens through the per-`Fifo` locks.
* Net effect: the main-thread shader tick, the action dispatcher, the
  naive graph controller tick, a subnet handler thread, and the future
  compute-graph coordinator can all run concurrently. Contention is at
  the finest possible granularity — the specific endpoint they happen
  to share.

---

## 2. Action plan: import the torch-based `GraphSolver`

Goal: make the analytic driver's generic-object-passing, cyclic-reduction
graph solver a first-class **subnet handler** under the naive-graph
controller — the SCC/cyclic-reduction torch solver lives in
[graph_solver.py](graph_solver.py), the materializer that builds it
lives in [network_materializer.py](network_materializer.py), and the
runtime entry point in
[analytic_runtime.py](analytic_runtime.py) is `render_patch_graph(...)`
which calls `solver.run_schedule({}, n_frames=n)` /
`solver.step({})`.

### 2.1 Survey of what we are importing (read-only)

| Concern | Where it lives |
|---|---|
| Generic node payload (whole tensor in / whole tensor out, complex128) | `TensorNode`, `TensorEdge` in [graph_solver.py](graph_solver.py#L1217-L1349) |
| Cyclic / SCC fixed-point reduction | `_tarjan_sccs`, `CondensedGraph`, `CyclicTensorBlock`, `_solve_scc`, `_solve_lateral_group`, `_solve_linear_region` in [graph_solver.py](graph_solver.py#L1671-L2700) |
| Schedule path (full KPN window) vs single-step path | `GraphSolver.run_schedule(...)` and `GraphSolver.step(...)` in [graph_solver.py](graph_solver.py#L2592-L2700) |
| Materializer: build `(nodes, edges)` from a patch | `compile_network` / `compile_nodes` in [network_materializer.py](network_materializer.py#L1017) |
| Existing runtime entry point | `render_patch_graph(...)` in [analytic_runtime.py](analytic_runtime.py#L192) |

What we are **not** importing: the other (non-torch) solver in the
workspace. Anywhere a choice exists, pick the torch path.

### 2.2 Mapping `GraphSolver` onto the new hierarchy

The torch solver becomes an *owner-level subnet handler* under
`NaiveGraphController`. The mapping is one-to-one and intentionally
shallow (no rewiring of the solver internals):

1. **Owner per analytic patch.** Each `AnalyticPatch` (or each compiled
   sub-patch) becomes an `OBJECT` under `GLOBAL` (or under a `ROOM` if
   the patch belongs to a room). The owner id is the patch id.
2. **One input endpoint per `TensorNode` external source key.** For
   every node key that `solver.step(ext)` accepts in `ext`, register an
   input endpoint with the same `endpoint_id`. The endpoint's `Fifo`
   becomes the queue the rest of the application pushes external
   tensors into.
3. **One output endpoint per "product" key.** For every key in
   `_resolve_product_keys(compiled, patch, product_keys)`, register an
   output endpoint with the same `endpoint_id`. The post-step worker
   pushes the per-tick or per-window product into this `Fifo`.
4. **Subnet tag.** All endpoints belonging to one compiled patch get
   the same `subnet_id` (e.g. `f"graph_solver:{patch_id}"`). This is
   the tag the controller will hand off to the handler.

### 2.3 The subnet handler — wiring the solver into `tick()`

`SubnetHandler` signature is `handler(controller, subnet_id, now)`.
Build a small adapter class:

```python
class TorchGraphSolverSubnet:
    def __init__(self, patch, *, subnet_id, sample_rate, mode="step"):
        from network_materializer import compile_network
        self.compiled = compile_network(patch, sample_rate)
        self.solver = self.compiled.solver
        self.solver.reset()
        self.solver.dispatch_before_start()
        self.product_keys = _resolve_product_keys(self.compiled, patch, None)
        self.subnet_id = subnet_id
        self.mode = mode  # "step" or "schedule"

    def __call__(self, controller, subnet_id, now):
        # 1. Drain inputs from this owner's FIFO context.
        ctx = get_control_graph().fifo_context(self.subnet_id_owner)
        ext = {}
        for key, fifo in ctx.inputs.items():
            items = fifo.drain()
            if items:
                # Generic-object passing: pass the latest item or stack.
                ext[key] = items[-1] if self.mode == "step" else _stack(items)

        # 2. Run one step or one schedule window under torch.no_grad().
        with torch.no_grad():
            if self.mode == "schedule":
                outputs = self.solver.run_schedule(ext, n_frames=self.window)
            else:
                outputs = self.solver.step(ext)

        # 3. Publish products to outputs.
        for key in self.product_keys:
            value = outputs.get(key)
            if value is not None and ctx.has_output(key):
                ctx.output(key).push(value)
```

Then registration is a one-liner:

```python
ng_ctrl = get_naive_graph_controller()
ng_ctrl.register_subnet_handler(
    f"graph_solver:{patch_id}",
    TorchGraphSolverSubnet(patch, subnet_id=..., sample_rate=sr).__call__,
)
```

### 2.4 Cyclic-reduction guarantees we keep

* `_tarjan_sccs` decomposes once at construction time inside
  `GraphSolver.__init__`. Re-imports are not needed; the SCC plan is a
  property of the compiled solver. As long as we hold a single
  `TorchGraphSolverSubnet` per patch, the cyclic reduction (SCC →
  `CyclicTensorBlock` fixed-point with IFT-corrected backward) runs
  exactly once per `step()`/`run_schedule()` — no work done at the
  controls layer.
* If the patch topology changes, throw the subnet away and re-register
  it. There is no incremental-recompile path in `GraphSolver`, so we
  do not invent one here.

### 2.5 Generic-object passing

`Fifo.push/pop/drain` is payload-agnostic. The pre-step adapter is the
only place that converts FIFO items to the tensor shape the solver
expects (`_canonical_complex(...)`, `_canonical_schedule_tensor(...)`).
Three policies — all already supported by the solver:

* **Latest sample** (`"step"` mode): take `items[-1]` (or a chosen
  reducer) per input port. One call to `solver.step(ext)`.
* **Window stack** (`"schedule"` mode): stack `items` into the leading
  time dim using `_canonical_schedule_tensor`. One call to
  `solver.run_schedule(ext, n_frames=len(items))`.
* **Pass-through** (already-tensor payloads): skip conversion; the
  caller is responsible for dtype/device. (Per project convention we
  do **not** force float32/float64 dtype; we propagate whatever the
  producer pushed.)

### 2.6 Cadence and threading

* Default home for the handler is the naive-graph tick (single-thread,
  same cadence as the rest of message passing).
* If the solver is heavy, move that one subnet onto its own thread:
  the handler enqueues onto a dedicated `queue.Queue`, a worker thread
  pulls and runs `solver.step` / `run_schedule`, and pushes results
  back into the output FIFOs. Per-`Fifo` RLocks make this safe with no
  controller-level coordination.
* `solver` calls must stay inside `torch.no_grad()` for the runtime
  path (training paths still use `step` directly, but those are not
  triggered by the controller).

### 2.7 Action plan, ordered

1. **Adapter module** `torch_graph_subnet.py`:
   * `TorchGraphSolverSubnet` (class above).
   * Helper `register_torch_graph_subnet(patch, *, subnet_id, sample_rate, mode, window=None)`
     that:
     * calls `compile_network(patch, sample_rate)`,
     * registers one input endpoint per accepted `ext` key on the
       patch's owner,
     * registers one output endpoint per resolved product key,
     * tags every endpoint with `subnet_id`,
     * registers the handler on the naive controller.
2. **Owner attachment**: pick an `owner_id` convention (`patch.id` or
   `patch.name`) and call `get_control_graph().attach_object(owner_id)`
   so the five peer axes exist before endpoints are added.
3. **Replace direct `render_patch_graph` calls with controller-driven
   ones** anywhere they are in the live runtime (search shows
   `render_patch_graph` is the existing entry point in
   [analytic_runtime.py](analytic_runtime.py)). Hard cutover, no
   shim/aliases (per project preference).
4. **Smoke test** parity with `render_patch_graph`: feed the same
   `ext` dict via input FIFOs, tick the controller, drain product
   FIFOs, compare tensors element-wise (allclose at `_CDTYPE`
   tolerance) for at least one patch with a cyclic SCC and at least
   one tier-1 linear patch.
5. **Optional dedicated-thread variant** (`mode="step_threaded"`): a
   subclass that runs the solver on its own worker; only swap it in if
   the inline cost of `solver.step` measured in step 4 hurts the
   controller's tick budget.
6. **Schedule-window cadence**: when `mode="schedule"`, expose
   `window` and only fire `solver.run_schedule` when `len(items) >=
   window` on every input port (or after a deadline). Otherwise fall
   back to `step` to keep latency bounded.

### 2.8 Risk register (specific to this import)

* **Implicit `_CDTYPE` (complex128) requirement.** The solver casts
  every input through `_canonical_complex(...).to(self.device)`.
  Generic objects pushed through FIFOs that are not coercible to
  complex128 will raise inside the adapter, not inside user code.
  Surface those as `unrouted` rather than crashing the tick.
* **`fifo_bank` already inside `GraphSolver`.** `network_materializer`
  attaches an `EdgeFifoBank` to the solver. That bank is internal to
  the solver and **must not** be confused with our FIFO endpoints.
  Keep them strictly disjoint: solver-internal `fifo_bank` for
  edge-history, controls-level `Fifo` for inter-owner messaging.
* **No incremental retopo.** `_tarjan_sccs` and `CondensedGraph` are
  built once. Topology changes ⇒ rebuild the subnet. Document this in
  the adapter's docstring; do not paper over it.
* **`run_schedule` allocates `(1, n, 1)` tensors per node.** Pick
  `window` carefully when `mode="schedule"`; otherwise stay on
  `mode="step"`.

---

## 3. Notes: offloading a graph computation to C through a KPN with
   a bidirectional C↔Python action API

This is forward-looking; nothing here is being implemented now. The
point is to record the cost surfaces so the choice can be made cleanly
when the time comes.

### 3.1 Why a KPN at all

A Kahn Process Network is a perfect match for this codebase: every
process is a deterministic function from input streams to output
streams over unbounded FIFOs — exactly the semantics our `Fifo` already
provides. The controls hierarchy already names the ports; the naive
controller already routes between them. A C-side KPN would replace
just the *execution* of one or more subnets with a native runtime,
keeping the Python side as the editor / orchestrator / UI.

### 3.2 What the C side has to be

* A KPN scheduler (round-robin or demand-driven) with bounded FIFO
  channels per port.
* Per-process state owned by C — the C runtime, not Python, holds the
  process workspace between fires (otherwise crossing the FFI per fire
  destroys throughput).
* A small message-passing core: pop input FIFOs, call the process,
  push output FIFOs, repeat. No global locks; one mutex per channel
  (mirrors our per-`Fifo` RLock).

### 3.3 The bidirectional C↔Python action API

Two directions, each with its own cost profile:

1. **Python → C (control plane).** Cheap if rare: one `cffi`/`ctypes`
   call per topology mutation (`add_process`, `add_channel`,
   `set_process_param`, `start`, `stop`). Marshalling cost is
   amortized; this is not where wall-clock time is lost.
2. **C → Python (action firing).** This is the expensive direction.
   Every time a C-side process wants to issue an action against the
   action registry it must:
   * Acquire the GIL (or use `Py_AddPendingCall` from outside it).
   * Marshal the action key + payload into Python objects.
   * Call back into our `enqueue_action(...)`.
   The cost of one round trip is O(microseconds); doing it per audio
   sample is unfeasible. Doing it per *event* (note on/off, state
   transition, debug hook) is fine.

### 3.4 Pragmatic shape of the bridge

* **Bulk transport**, not per-item callbacks. Use a shared-memory
  ring buffer for each `Fifo` whose other end is in C. Python writes
  raw bytes; C reads raw bytes. The serializer is fixed per channel
  (declared at registration time) so no per-item type tag is needed.
* **One callback type, batched**. Expose a single C → Python entry
  point `flush_pending_actions(action_key[], payload_bytes[])` that
  the C scheduler calls at most once per its own tick. Python-side
  this turns into a loop of `enqueue_action(...)` calls — still
  cheap because our dispatcher is already a queue.
* **No Python objects in C-owned state.** The C process function
  signature should be `void process(state*, in_channels[], out_channels[],
  events_out*)`. Anything else either re-introduces the GIL on the hot
  path or owns memory neither side can free deterministically.

### 3.5 Complexity hot spots, ranked

1. **GIL discipline.** Any callback path from C that holds the GIL
   blocks every other Python thread (UI, action dispatcher, naive
   controller). The bridge must release the GIL around the scheduler
   loop and re-acquire only at action-flush time.
2. **Lifecycle / ownership of channel buffers.** Decide once and
   document: C owns the ring buffer memory; Python gets a `memoryview`
   for the lifetime of the channel. Closing the channel from either
   side must be a single explicit call; never rely on GC.
3. **Backpressure.** Our Python `Fifo` drops oldest on overflow. A C
   KPN with bounded channels typically *blocks* the producer, which is
   the whole point of KPN determinism. Pick one policy per channel at
   registration time and surface it through the same `capacity` knob
   the Python side already has — but be explicit that "0 = unbounded"
   on the C side means "block on full" rather than "grow forever".
4. **Topology updates while running.** The Python control plane will
   want to add/remove processes and rewire channels during play. The
   C scheduler needs a quiescence point ("pause", "swap topology",
   "resume") rather than fine-grained mutation. Mirror what the torch
   solver already forces us to do (rebuild on retopology).
5. **Determinism vs. real-time.** Pure KPN is deterministic but not
   real-time; scheduling is order-of-firing only. If audio output
   needs sample-accurate timing, the C scheduler has to expose a
   wall-clock-driven mode, which breaks pure-KPN determinism. Pick
   per-subnet, document, do not mix.
6. **Debuggability.** A C KPN that runs across a thousand fires
   between Python observations is invisible from a Python debugger.
   Plan a tracing channel from day one (one extra output channel per
   process, written into a ring of `(process_id, fire_index, ts,
   short_event)` records). The naive-controller stats counters are
   the right precedent: cheap, always on, queryable.

### 3.6 Migration path (when we get there)

1. Stand the C KPN runtime up as a *single subnet handler* on the
   naive controller (mirror of §2.3): one subnet per C-side network,
   exactly like the torch `GraphSolver` adapter.
2. The handler's `__call__` empties Python output FIFOs into the C
   ring buffers and copies the C output ring buffers into Python
   input FIFOs. Nothing else changes on the Python side.
3. Action callbacks come back through the batched
   `flush_pending_actions` route into our existing
   `ActionDispatcher` queue.
4. Only after that bridge is solid: move individual `TorchGraphSolverSubnet`
   instances over to C-backed equivalents one at a time, keeping the
   torch path as the canonical reference implementation for parity
   testing.

The point of this ordering: the controls/actions/FIFO/naive-graph
layers we built today are *already* the right Python-side seam for the
C bridge. Nothing here needs to change to support it.
