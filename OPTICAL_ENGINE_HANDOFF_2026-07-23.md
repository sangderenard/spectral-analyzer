# Optical engine handoff — 2026-07-23

## Current committed frontier

The branch is `nogodsnomasters`. The two most recent implementation commits
are:

- `2b4ae78 bring projector back through reciprocal optics`
- `9dd5e52 formalize Jones and differential optical operators`

The repository was clean before this handoff document was added.

## What is working

### Projector/camera reciprocity

- Projector mode uses the physical ordering:
  `projector back -> spectral transmissive sensor scrim -> reciprocal exact
  compound lens -> scene`.
- The sensor remains real geometry. Only the opaque sensor-bay rear wall is
  removed in projector mode.
- The sensor scrim has per-lane transmission, color, and diffusion.
- Ordinary camera mode retains its absorbing sensor and rear wall.
- The projector source is a full physical plane using the common
  `EmitterProfile` contract, not a point-light shortcut.
- Reciprocal exact-lens traversal has its own intent flag. It does not acquire
  sensor/BDPT ownership or sensor-splat semantics.
- Projector requests are scheduled by the persistent optical backend. The
  camera station does not own a tracer.

### Complex/Jones operator layer

- Complex lane schema v2 remains exactly 64 bytes.
- Former reserved lane words 14 and 15 are stable `basis_id` and
  `operator_id` handles.
- Full state lives in graph-owned contiguous tables:
  - 32-byte right-handed transverse-basis records;
  - 96-byte combined Jones/canonical differential operator records.
  - 48-byte coherent source-mode records containing normalized Jones
    amplitudes, power, 64-bit coherence identity, and stable basis/operator
    handles.
- CPU reference, C++ ABI, and GLSL ABI agree on:
  - `s x p = k`;
  - Jones column order `[E_s, E_p]`;
  - basis changes;
  - complex 2x2 operator application and composition;
  - power-normalized dielectric Fresnel scattering;
  - canonical phase-space coordinates `[q_s,q_p,n*k_s,n*k_p]`;
  - signed 4x4 tangent maps;
  - reference-OPL carrier phase;
  - caustic-safe ray-to-field amplitude conversion.
- Unpolarized and partially polarized sources decompose into incoherent
  orthogonal Jones modes. Linear angle, circular handedness, radial, and
  azimuthal polarization now resolve correctly.
- The threaded native exact-lens helper returns spectral-lane-specific full
  4x4 maps, signed determinants, symplectic residuals, and validity.

### Existing T4 capability relevant to the next step

The native wave arena already owns four persistent field families:

```text
forward s
forward p
backward s
backward p
```

It marches every active component. The missing vector behavior is localized
at the remaining exit adapter: source-tagged production entry now resolves
the persistent source/basis/operator block and seeds s/p, while current exit
still reduces combined s/p power to one scalar ray amplitude. Untagged legacy
rays deliberately retain the s-only specialization.

A reusable `JonesFieldState` reference adapter now drives both components
through the production native aperture-material and angular-spectrum kernels.
It preserves independent coherent modes and produces Stokes/analyzer
intensity only after propagation. This qualifies the vector kernel path while
leaving the production ray-pipeline boundary honestly unfinished.

### Physical-aperture vector demo

`wave_transform_visual_demo.py --aperture-live` now provides:

- linear, circular, radial, azimuthal, partial, and unpolarized sources;
- independent coherent-mode propagation for partial/unpolarized light;
- Stokes I/Q/U/V, spatial polarization, and rotatable analyzer views;
- coherent s/p material and propagated phase views;
- forward and reverse vector propagation through real finite material blades;
- relative-phase display with piston removed, or absolute phase on demand.
- padded open-boundary solve tiers:
  - balanced: 2x width, 4x samples, 4 propagation substeps;
  - high: 4x width, 16x samples, 8 propagation substeps;
  - bake: 8x width, 64x samples, 16 propagation substeps;
- the production exact-lane absorbing exterior after every substep, with only
  the central requested field published to the OpenGL panels.
- live switching between perceptual fixed bands and stratified
  continuous-frequency cohorts at exact widths `1,3,4,8,16,32`.

The blade material is isotropic, so it does not invent polarization conversion.
The demo is a qualification and visualization client of shared production T4
kernels, not a station-owned or demo-owned solver.

## Verified numerical properties

- Transverse bases are orthonormal and right-handed.
- Basis changes preserve field power and round-trip.
- Lossless Fresnel reflection plus power-normalized transmission equals one
  separately for s and p.
- p reflection vanishes at Brewster incidence.
- Total internal reflection has unit magnitude with complex phase.
- Canonical maps compose, invert, and retain signed off-diagonal structure.
- Default exact-lens maps have determinant approximately +1 and very small
  symplectic residual.
- Exact-lens maps differ by spectral lane.
- Forward and backward exact-lens maps satisfy time-reversal reciprocity within
  finite-difference tolerance.
- Ray-to-field conversion rejects a singular configuration map as a caustic.

The final focused groups passed 134 tests in total. The canonical native
extension was rebuilt successfully from `csrc_build`.

## Architectural invariants

Do not weaken these while continuing:

1. The station and UI submit optical work; they never acquire a private tracer.
2. T4 uses the ray pipeline and also owns persistent contiguous arena state.
3. Do not put an N-lane Jones/differential payload into every ordinary ray.
4. Carry a stable handle through compact ray state; resolve full state only at
   an operator or representation boundary.
5. Exact `1,3,4,8,16,32` lane specializations remain deliberately tight.
6. Continuous spectral cohorts remain distinct from fixed lanes.
7. Physical apertures are finite material geometry, never ideal masks.
8. A scalar determinant does not replace a full differential map.
9. A lone Jones vector does not represent unpolarized light.
10. A caustic is routed to a wave/uniform treatment, not hidden by clamping.
11. Build native code from `csrc_build`.

## Honest incompleteness

- Source modes, transverse bases, and Jones/differential operators are
  installed in one cold, aligned native `RayPipelineState` allocation. Compact
  16-byte tag bindings index deduplicated 48-byte source modes, so many rays
  can share a mode without widening ordinary CPU/GPU intent records.
- Ordinary T3 material boundaries do not yet apply the canonical Jones
  interface operators.
- Exact T2 computes full tangent maps through the native helper, but the hot
  transport path does not yet compose those maps into an indexed operator
  record.
- T4 entry is Jones-complete for configured source records in fixed and
  continuous modes; the legacy no-record specialization remains s-only.
- T4 exit still collapses vector field state to scalar complex ray amplitude.
- Full Jones projector textures remain unwired. Programmatic fixed and
  continuous Jones launches are wired through the shared source-mode API.
- Finite projector pupil-fill and condenser/relay source optics remain after
  the vector boundary is correct.

## Ordered next implementation plan

### 1. Pipeline-owned complex source state

- Completed: one persistent allocation containing source bindings, deduplicated
  source modes, bases, and operators is owned by `RayPipelineState`.
- Lower each emitter's coherent-mode decomposition into this block once per
  immutable source revision.
- Store spectral/coherence identity, local transverse basis, Jones amplitudes,
  and operator handle.
- Carry only a stable handle through T1/T2/T3. Reuse existing correlation
  identity or a verified compact handle; do not enlarge all ray records with
  vectors.
- Define explicit lifetime, generation, invalid-handle, and child-copy rules.

Ray tags are the stable handles. Configuration is accepted only with no work
in flight; replacement atomically publishes a new immutable block. Tags and
continuous-frequency state are copied unchanged by child CPU/GPU intents.

Acceptance:

- linear/circular/elliptical source reaches a wave port with both components;
- unpolarized modes retain separate coherence identities;
- ordinary scalar rays have no additional dynamic allocation.

### 2. Jones-complete ray-to-T4 entry

Completed for configured source records in both fixed-band and continuous
cohort modes.

- Construct the incidence/arena basis from ray direction and port frame.
- Apply the indexed basis-change/Jones operator.
- Seed both persistent T4 transverse fields.
- Apply reference-OPL carrier phase once per coherent cohort.
- Preserve fixed-lane and continuous-cohort specialization behavior.

Acceptance:

- s-only and p-only plane waves remain separated;
- a 45-degree linear source deposits equal s/p power;
- circular input retains quadrature phase;
- entry power equals seeded field power within tolerance.

### 3. Jones-complete T4 exit

- Extract complex first moments per component without merging polarization.
- Return a complex-ray sidecar handle or retain the full field when one-ray
  reduction is invalid.
- Compose the exit basis and differential operator.
- Route singular/caustic reductions to a declared wave continuation.

Acceptance:

- entry -> homogeneous arena -> exit preserves Jones state and total power;
- forward/backward round-trip is reciprocal;
- scalar calibration mode remains an explicit specialization.

### 4. Physical interface operators

- Apply dielectric/conductor Jones operators in T3 using the same material
  spectral data as scalar transport.
- Carry reflected and transmitted bases independently.
- Add absorption and complex-index conductor qualification.
- Do not replace stochastic transport weights or BDPT PDFs with Jones
  amplitudes; maintain their separate measures.

Acceptance:

- Fresnel CPU/GLSL parity;
- Brewster/TIR tests through actual T3;
- energy accounting with absorption;
- reverse operator reciprocity.

### 5. Hot exact-T2 differential composition

- Move the canonical tangent-map calculation/composition into the exact T2
  event path or a compiled differential specialization.
- Write/update indexed operator records outside ordinary ray payload.
- Connect signed map data to camera PDFs, ray/field gain, and diagnostics.
- Retain the current threaded finite-difference helper as the qualification
  oracle; do not make eight extra traces per production ray the default.

Acceptance:

- hot/finite-difference map agreement;
- symplectic residual threshold;
- spectral-lane parity;
- no material regression in T2 throughput.

### 6. Return to source optics

After vector boundaries pass, implement finite projector pupil quadrature,
condenser/relay elements, Jones field textures, and continuous-frequency
source sidecars. Then the projection bench can examine physically meaningful
light-source optics rather than a scalar center-pupil approximation.

## Restart commands

```powershell
git status --short
git log -5 --oneline
cmake --build csrc_build --config Release --target _spectral_kernels -j 8
python -m pytest tests/test_complex_optical_operators.py tests/test_complex_transport_schema.py tests/test_optical_transport_graph.py -q
python -m pytest tests/test_wave_t4_pipeline.py tests/test_camera_designer_gpu_bench.py -q
```

Primary contract:

- `COMPLEX_OPTICAL_OPERATOR_CONTRACT.md`

Related architecture:

- `WAVE_T4_REPLACEMENT_DECISION.md`
- `OPTICAL_TRANSPORT_GRAPH.md`
