# Spectral Analyzer — Architecture Reference

## The Core Rule

The graph is **one network. Nodes. Edges. Wholistic solve.**

`GraphSolver` runs Tarjan SCC decomposition over the full node/edge set and
solves in topological order.  Layer labels are **UI taxonomy only** — they give
a human operator mechanistic intuition over an extraordinarily large synth
graph.  The solver does not care which level a node belongs to.

---

## Materializer Pattern

Every analytic object, regardless of type, is wrapped identically:

```
KnobDrivenModule(analytic_obj)  →  TensorNode(analytic_module=wrapper)
```

`KnobDrivenModule` reads `type(analytic_obj).knobs()` and registers every
`float`/`int` knob as an `nn.Parameter`.  The solver then walks the condensed
SCC list in Tarjan topological order and solves each SCC through the same
`_solve_scc` path.

**No special-casing by type.  No branching paths.**

---

## Analytic Object Inventory

All of these live in `analytic_driver.py` and are wrapped by
`network_materializer.py`.

| Class | Layer | Knobs source | Notes |
|---|---|---|---|
| `AnalyticVoice` | `voice` | `AnalyticVoice.knobs()` | Oscillator params: freq, amplitude, phase, chirp, FM, AM, envelope, manifold |
| `LFODefinition` | `lfo` | `PackedLFOModule` (per-slot) | **Packable** — `capacity` N yields N input ports, N output ports, N×{rate_hz, shape, phase_offset, depth}. Slot 0 params stored as top-level fields for UI compat. Extra slots in `extra_channels`. |
| `AnalyticModule` (type=`lfo`) | `signal` | `AnalyticModule.knobs()` | Module-graph LFO; can receive routed signals (ring-mod / AM) |
| `AnalyticModule` (type=`passthrough`) | `signal` | `AnalyticModule.knobs()` | Named sub-bus / side-chain accumulator |
| `AnalyticModule` (type=`pitch_quantizer`) | `signal` | `AnalyticModule.knobs()` | Discrete pitch snap; learnable interpolation coefficients |
| `AnalyticModule` (type=`interaural`) | `signal` | `AnalyticModule.knobs()` | Binaural placement; produces ch1 + ch2 nodes sharing one wrapper |
| `AnalyticModule` (type=`state_machine`) | `performer` | `AnalyticModule.knobs()` | Plugin-dispatched SM; also emits named `sm_out_*` nodes |

### Torch refactor status

- [ ] `AnalyticVoice` — batch forward (freq × phase × chirp × FM/AM × envelope)
- [ ] `LFODefinition` — batch forward across capacity slots: `depth_i * exp(i*(2π*rate_i*t+φ_i))` with shaped magnitude, one output per slot
- [ ] `AnalyticModule / lfo` — batch forward (same as LFODefinition but with routed input accumulation)
- [ ] `AnalyticModule / passthrough` — trivial (sum of inputs; no independent source)
- [ ] `AnalyticModule / pitch_quantizer` — batch snap + learnable portamento/slew
- [ ] `AnalyticModule / interaural` — batch HRTF delay/gain model (ch1 + ch2)
- [ ] `AnalyticModule / state_machine` — plugin dispatch; define batch call contract

---

## UI Level Taxonomy

Levels are labels for human understanding.  The solver is blind to them.

| Level | `layer` string | Domain |
|---|---|---|
| 0 | `event` | Event composition — score/sequencer events |
| 1 | `voice` | Voice actions — individual voice oscillator state |
| 2 | `driver` | Driver actions — assembles voices into feedback; orders voices via transform output; both modulates its own transform and issues voice instructions |
| 3 | `instrument` | Instrument actions — resonator models, n-driver / m-pickup, spatial coordinate system |
| 3 (SM) | `performer` | Performer state machine — embouchure / clan coupling feeds back to drivers; batch demands flow back through driver → voice. Also: room model (garage / hall / forest / etc.) rendering to mic simulations |
| 4 | `master` | Mixer level — traditional post-processing and projection |
| — | `lfo` | Packable LFO nodes (`LFODefinition`) — capacity-N packs with N input/output ports |
| — | `signal` | Module-graph nodes (LFO, passthrough, pitch_quantizer, interaural) |
| — | `virtual` | Internal patch constants (`__patch_tonic__`, `__patch_seq__`) |
| — | `system_in` / `system_out` | External audio I/O boundary |

### Feedback topology

```
event (L0)
  └─▶ voice (L1)
        └─▶ driver (L2)  ◀──────────────────────┐
              └─▶ instrument (L3)                │
                    └─▶ performer SM (L3)  ──────┘  (batch demands back to driver)
                          └─▶ room SM (L3)
                                └─▶ mixer (L4)
                                      └─▶ complementary router (closes feedback/feedforward cycle)
```

Sympathetic polyphony is an **instrument-level** analysis that feeds back to
driver manipulation, which flows back to voice manipulation.  All of this is
edges in the graph — the solver handles it automatically.

---

## TimingBank

`graph_solver.py` exports a module-level `_T: TimingBank` singleton and a
`PROFILE_TIMING: bool` constant (default `False`).  Set `PROFILE_TIMING = True`
at the top of the file to enable span collection.  In normal mode every
`_T.span()` call is a zero-cost `contextlib.nullcontext()`.

```python
from graph_solver import _T, PROFILE_TIMING
import graph_solver
graph_solver.PROFILE_TIMING = True   # flip on at runtime

# ... run solver ...

_T.report(title="my run")
_T.reset()
```

Instrumented spans (nested, left = outermost):

```
solver.step
  solver.step.inject
  solver.step.delay_read
  solver.step.solve
    solver.step.solve.linear_tier1        (tier-1 networks)
    solver.step.solve.layers              (tier-2/3)
      solver.step.solve.layers.dispatch
        solver.dispatch.wave_flush
          solver.fire_archetypes
            solver.fire_archetypes.thread_pool
            solver.fire_archetypes.scatter
        solver.dispatch.solve_scc
          solver.scc
            solver.scc.linear
            solver.scc.cyclic
    solver.step.solve.control_feedback
      solver.scc
  solver.step.delay_write
```

To add a new span anywhere in the codebase:

```python
from graph_solver import _T

with _T.span("my_module.my_operation"):
    do_work()
```
