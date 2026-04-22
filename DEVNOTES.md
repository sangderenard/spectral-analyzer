# Development Notes — Spectral Analyzer / Analytic Driver

---

## 2026-04-20 T14:31 — Global Solver: Condensed-DAG Architecture

**Author:** Claude Sonnet 4.6 (session collaboration)

### Appreciation

Reading the full routing stack — `routing_engine.py`, `routing_solve_torch.py`,
`sm_node.py` — I want to note what is already correct here before describing
what's missing, because the foundation is genuinely well-designed.

The analytic signal discipline (`torch.complex128` throughout, no silent
narrowing) is load-bearing in a way most signal processing systems get wrong:
real projections collapse phase information irreversibly, and this system
treats the analytic representation as the invariant, not the exception.
The `(I−W)⁻¹ @ Src` resolvent as the canonical linear solve, the
`_softclip_complex` magnitude-preserving overflow guard, the `CompiledRouter`
per-sample ring-buffer architecture — these are all the right ideas.

What excited me most in today's design conversation was the sub-dt cycling
insight: that unconverged cyclic state is not an error to be suppressed but
information to be carried. The cache is *the contract*, not an optimisation.
This is the right way to think about it. A nonlinear feedback cycle with a
natural period shorter than `dt` isn't aliasing if the solver is honest about
what it didn't finish — it just needs a place to park the residual and resume
next sample. That's a simple invariant once named, but I haven't seen it
stated explicitly in the signal processing literature, possibly because
block-based solvers don't expose per-sample iteration at this granularity.

The Lagrangian framing (residual potential `V(z) = ½‖z−F(z)‖²`, fixed point
as stationary point of `L = ½‖ż‖² − V`) is a useful analysis instrument even
though it doesn't appear at runtime. It gives a spectral stability certificate
(`ρ(J_F(z*)) < 1`) that can be checked at graph-build time, and it
characterises the convergence rate in terms the physics of the modelled system
can speak to. An EM junction and a waveguide tube junction are both just `F`
with different Jacobians; the Lagrangian makes that structural similarity
visible.

The implicit function theorem autodiff is the clean answer to the gradient
problem. Unrolled fixed-point iteration gives depth-K compute graphs and
gradient explosion; the IFT gives exact gradients at O(1) depth by solving
the adjoint system instead. This is what makes the solver machine-optimisable
without the parameter learning fighting the iteration structure.

### What was implemented today

`graph_solver.py` — new module, builds on the existing `routing_solve_torch`
primitives:

- `SCCNode` / `CondensedDAG` — Tarjan SCC decomposition of the zero-delay
  subgraph, topological ordering of the condensed DAG, precomputed at
  build time.

- `build_condensed_dag()` — pure graph function, no tensors. Classifies each
  SCC as linear (resolvent) or nonlinear/cyclic (fixed-point). Stores
  inter-SCC dependency edges.

- `FixedPointSolve` — `torch.autograd.Function` with IFT backward. Forward
  runs Picard to convergence with no grad tracking. Backward solves the
  adjoint system `(I−J_F^T)v = grad_output` via the same Picard iteration
  on the transpose, then returns `v^T · ∂F/∂θ` as the parameter gradient.
  Exact gradient, O(1) compute graph depth relative to iteration count.

- `CyclicBlock` — `nn.Module` per nonlinear SCC. Holds `W_scc` (local routing
  matrix) and `z_cache` (sub-dt carry-over buffer). Fixed-point loop uses
  masked convergence (no Python `break`) so the static `K_max` bound makes
  it `torch.compile`-friendly. `K_max` scales with `natural_rate_hz` when
  declared.

- `GraphSolver` — `nn.Module`, condensed-DAG executor. One `step()` call per
  sample. Routes external inputs, accumulates delayed contributions from ring
  buffers, walks SCCs in topological order (resolvent or `CyclicBlock`),
  commits results to ring buffers. Reuses `CompiledRouter`'s delay-line
  ring-buffer design.

### Invariants established

1. All signal tensors `torch.complex128` — no silent narrowing.
2. SCC decomposition precomputed at build time, never per-sample.
3. `CyclicBlock` owns cycle resolution — nothing upstream breaks cycles.
4. IFT backward — never backprop through iteration loops.
5. `z_cache` is first-class state — unconverged residual is carried, not
   discarded.
6. `K_max` is a hard limit — solver never hangs.
7. Physics lives in `node_transforms` callables — `GraphSolver` is
   domain-blind.

### Next steps

- Phase 2: `StateMachineNode` integration — `GraphSolver` needs to call
  `sm.step()` at SCC boundaries and honour the `.in` / `.out` / `.ctrl`
  slot contract from `sm_node.py`.
- Phase 3: `torch.compile` verification pass — test that `CyclicBlock.step()`
  compiles cleanly with static `K_max` unrolling.
- Phase 4: IFT backward integration test — verify gradients match finite
  differences on a small nonlinear cycle.
- Phase 5: `natural_rate_hz` scheduling — expose this as a knob on SCC nodes
  so the audio physics state machines can declare their natural timescale.
