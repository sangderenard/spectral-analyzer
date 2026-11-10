# Wave continuation energy audit

## Result

The existing native T4 entry/exit adapters already transfer discrete field
power. The incomplete part is **field -> ordinary ray -> later field**:
the later arena starts again from the original source mode instead of consuming
the previous exit's Jones state. No production transport code was changed by
this audit. The complete coupling is not implemented or qualified.

This is not evidence that wave coupling caused the reported color issue.
If an exposure did not enable wave contexts, this T4 continuation defect cannot
be the cause of its color output.

## Measured native reproductions

`tests/test_wave_t4_continuation.py` drives the existing native CPU pipeline.
It does not implement a wave solver or change renderer/exposure configuration.

- Two compatible linked arenas retain a source power weight exactly once.
  The linked controls pass for weights 0.25 and 0.5.
- Two separated vacuum arenas retain relative powers `[0.25, 0.0625]` for a
  source weight of 0.25, instead of `[0.25, 0.25]`.
- The equivalent weight-0.5 case produces `[0.5, 0.25]`, not `[0.5, 0.5]`.
- A unitary S/P-swap interface correctly changes an S source into P inside
  linked fields. After a gap carried by an ordinary ray, the next arena resets
  to S. The P power fractions are `[1, 0]`, instead of `[1, 1]`.

The power tests divide by matching unit-weight controls to isolate source
weighting from the ordinary propagation and numerical exterior. The Jones
test shows why merely compensating the scalar energy would not fix coupling.

The three missing continuation gates are strict expected failures. Running
with `--runxfail` exposes the actual assertion failures and their values.
Do not describe the expected-failure cases as passing physical acceptance.

## Existing implementation to extend

- `csrc/kernels/ray_tracer.cpp`, `pipeline_source_jones`: resolves an immutable
  source-mode record by ray tag, applies its power weight/operator, and rotates
  the source basis into the current arena. It has no previous-exit consumer.
- `wave_arena_seed`, `wave_arena_seed_continuous_cohort`, and
  `wave_arena_seed_fixed_cohort`: call that same source resolver on entry.
- `wave_arena_extract`: produces the scalar representative and optional
  `WaveExitStateRecord` per lane. The S/P reduction and basis are available,
  but no mutable continuation is installed from them.
- `RayPipelineState::complex_source_state`: one immutable cold allocation.
  `Q_wave_exit_states` is an opt-in diagnostic output queue, not a registry.
- `csrc/include/optical_branch_abi.h`: the existing `OpticalBranchToken`
  already declares `state_handle` and `state_generation`. These tokens are
  not routed alongside ordinary T1/T2/T3 intents.
- `OPTICAL_TRANSPORT_GRAPH.md`, "Observable native wave boundaries": explicitly
  requires cold-allocated contiguous continuation state and compact handles.
  It rules out per-ray heap payloads and hot unordered lookup.

## Requirements before a production fix can be called complete

1. Route the existing branch handle/generation contract through ordinary
   transport, including CPU/GPU packing, branching, and reused submissions.
   A mutable cache keyed only by root tag is not sufficient: branches share
   that tag and can have different Jones states.
2. Allocate bounded continuation state cold, with explicit capacity failure,
   generation checks, lifetime/reclamation, and checkpoint participation.
   Ordinary rays must not acquire an N-lane Jones payload.
3. Consume exit S/P, basis, coherence, frequency/PDF, and the scalar phase
   reference once. Across an unchanged homogeneous ray segment, the retained
   Jones pair can use the scalar propagation ratio; that formula must not be
   extended silently across material interactions.
4. Compose physical Jones/basis operators through intervening exact T2 and T3
   boundaries. An unpolarized scalar Fresnel factor is not that composition.
5. Preserve the complete field through compatible field links. The declared
   first-moment ray reduction is not invertible and cannot recover a diffracted
   spatial field when a later operator needs it.
6. Qualify both directions, fixed/continuous spectra, all exact lane widths,
   multiple roots, branch fanout, repeated submissions, capacity exhaustion,
   rollback, and CPU/GPU parity. Remove the strict xfails only when the
   corresponding native paths actually satisfy the tests.

## Verification

Focused CPU-only rerun:

```powershell
python -m pytest -q tests/test_wave_t4_continuation.py tests/test_wave_t4_pipeline.py tests/test_wave_t4_rigid_interface.py tests/test_complex_optical_operators.py tests/test_complex_transport_schema.py tests/test_vector_wave_adapter.py -k 'not gpu' -rx
```

Result: **60 passed, 4 deselected, 3 xfailed** in 7.03 seconds.

```powershell
python -m pytest -q tests/test_wave_t4_continuation.py --runxfail --tb=short
```

Result: **2 passed, 3 failed**, precisely the energy and Jones assertions
listed above. No native rebuild or GPU exposure run was performed for the new
regression file. The existing five-file baseline suite had previously passed
62 tests; it included one GPU T1 test, so that earlier run is not CPU-only.
