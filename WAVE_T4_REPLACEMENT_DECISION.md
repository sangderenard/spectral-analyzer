# Wave/T4 replacement decision

Date: 2026-07-22  
Implementation cut: 2026-07-23, checkpoint `5c459e4`

## Decision

T4 becomes the sole stateful optical complex-field subsystem. It owns its
arenas, exact band-specialized storage, propagation state, source coherence,
boundaries, checkpoints, progressive display, and ray/field handoff.

The ordinary spectral ray pipeline may submit work to T4 and consume completed
work, but it must not contain a second field solver. Python may configure and
inspect T4; it must not perform production propagation.

The production field representation is a bidirectional, two-component vector
angular-spectrum field. Every exact lane width uses that same permanent ABI.
This reduced Helmholtz engine reconstructs the longitudinal component from
transversality and uses explicit polarized interface scattering.
It is deliberately not described as a volumetric full-vector Maxwell solver.

Localized full-Maxwell work is a separate patch compiler described in
`MAXWELL_PATCH_CONTEXT.md`. It produces cached complex polarized scattering
artifacts consumed by both ray transport and T4; it never runs implicitly in a
camera-scale hot propagation loop. The ordered implementation program is in
`WAVE_ENGINE_ACTION_PLAN.md`.

## Existing code disposition

### Retire from production

- `camera_designer/wave_tube.py`: separate NumPy ADI-BPM, UV-average seed,
  swallowed native registration failure, and per-forward-frame solve hook.
- The ideal polygon aperture mask as a production optical element. Apertures
  are physical blade/baffle geometry with spectral material properties.
- The optical/EM authority of `rt_field_solver.cpp`. Its acoustic transfer
  work is a separate concern and must be split or preserved deliberately.
- The disconnected `field_grid_step_helmholtz_regular` marcher. `FieldGrid`
  allocation, injection, tiling, and display/capture storage remain shared
  infrastructure.
- `coherent_accumulate.comp.glsl` as an alleged diffraction solver. It deposits
  ray crossings coherently but does not propagate a field.

### Removed at the implementation cut

- The standalone NumPy wave tube, public scalar marcher, matching compute
  shader, and their double-slit runner were removed together after checkpoint
  `5c459e4`. They are recoverable from Git history, not callable production
  alternatives.

### Preserve as isolated validation oracles

- Direct CPU Rayleigh-Sommerfeld propagation for small scalar reference cases;
  its documentation must not claim a GPU backend that is absent.
- A future slit acceptance scene must use physical blade geometry and the
  production arena rather than resurrecting a separate numerical runner.

### Preserve as shared infrastructure

- Camera triangle geometry, directional media, material database, parametric
  lens transport, spectral lane metadata, and BDPT transport contracts.
- `FieldGrid` complex storage and tile access.
- Field display texture publication, but not its current ray-hit accumulator as
  a substitute for a propagated field.
- Acoustic FDTD, AMR, pressure backend, and co-evolution engines.

## Band specializations

The supported band counts are exactly `1, 3, 4, 8, 16, 32`.

Each count receives an intentionally compiled backend specialization. Runtime
selection occurs once when an arena is constructed. Storage is sized for the
selected count; no 32-lane allocation is used for a smaller field.

Band count is storage width, not spectral semantics. Every specialization has
two modes: fixed bands and continuous cohorts. A continuous one-lane field
carries one sampled frequency, PDF, and coherence identity. A continuous
multi-lane field carries that metadata independently for every lane so several
stratified continuous samples can propagate concurrently without becoming a
fixed discretized spectrum. Coherent samples may share a coherence identity;
independent identities combine only as intensity.

- C++ uses `WaveArenaState<B>`-equivalent exact layouts.
- GLSL sources share algorithm includes and are compiled with a fixed
  `WAVE_BANDS` definition.
- Band loops are statically unrolled/generated.
- C++, GLSL, and Torch share generated ABI/layout/constants and calibration
  vectors. Backend-specific numerical code remains readable and is required to
  pass parity tests.

### Hot-state allocation invariant

T4 remains a stage of the existing ray pipeline: the pipeline schedules T4,
hands work into it, and consumes its completed output. T4 simultaneously owns
one persistent, solid contiguous block of memory for its numerical system
state. The pipeline supplies that state block to the selected T4 backend
through a buffer, and the backend updates it in place across dispatches.
Its offsets are computed when the arena is created and remain stable for the
arena lifetime. A hot march may change indices and overwrite state slots; it
must not resize the state or instantiate a solver.

The persistent block is not a detached solver and it does not bypass the
pipeline. It is also not permission to repack unrelated pipeline storage. Ray
intents and results cross the normal pipeline stage boundary; values required
for continuing T4 numerical state live in the persistent block.

## Propagation model

The target production model is bidirectional vector angular-spectrum
propagation with complex material/interface scattering and explicit
forward/backward field states. It replaces the current one-way paraxial arena
assumption.

- Each lane carries two transverse complex electric-field components; the
  longitudinal component follows from the plane-wave transversality condition.
- Free-space and homogeneous-medium propagation use the spectral `kz` relation.
- Complex refractive index supplies phase and absorption.
- Physical interfaces split reflected and transmitted fields through explicit
  Jones/modal scattering operators with declared basis conventions.
- Independent source/coherence modes are accumulated incoherently; samples
  belonging to one coherent mode are accumulated as complex amplitude.
- Torch is the differentiable/reference backend. GLSL is the production live
  backend. CPU code is the deterministic validation backend.

Literal wavelength-resolved 3-D Maxwell/FDTD across a 6x6 camera is outside
this contract: its memory and spatial resolution are infeasible at visible
wavelengths. Full-Maxwell treatment is localized to authored patches and
compiled into scattering artifacts. Vector/high-NA behavior belongs in the
ordinary representation and must not be hidden behind scalar claims.

## Boundary strategy

Physical and numerical boundaries are distinct.

- Aperture blades, baffles, lens rims, mirrors, walls, and absorbers are actual
  scene geometry and spectral materials. They are never ideal field masks.
- The numerical exterior uses a padded guard region plus a smooth complex
  absorbing layer. FFT wraparound is not allowed to reach the useful region.
- Longitudinal ends are explicit incoming/outgoing ports.
- A closed experiment places reflective physical geometry inside the numerical
  absorber. An open experiment places the absorber outside the authored scene.
- Every run reports border incident power, absorbed power, reflected leakage,
  and residual energy. A PML finite-difference edge backend remains an option
  if the padded complex absorber cannot satisfy the reflection gates.

### Live physical-aperture ABI

`camera_software.physical_aperture.LivePhysicalAperture` is the shared live
authoring object for iris blades and repeated mask/grille assemblies. It emits:

- finite-thickness material triangles for ray and OpenGL camera geometry;
- a cold optical-graph contract naming pattern, dimensions, material, and the
  explicit `ideal_mask: false` invariant;
- one fixed 19-double `APTR` v1 payload copied into arena-owned configuration.

The present T4 consumer is a localized scalar split-step complex-index volume
operator. It provides phase, reciprocal attenuation, diffraction after
propagation, and material power accounting. It does **not** yet provide the
required reflected field, vector polarization coupling, oblique path-length
correction, or sharp-interface Fresnel/Jones operator. Those omissions remain
release gates; the thin material operator must not be relabelled as completion
of a fully physical blade boundary.

## Stateful ownership

T4 is split into independently testable responsibilities:

1. `WaveSceneCompiler`: consumes the same authored geometry/material contract
   as the camera tracer and produces arena-local material/interface data.
2. `WaveArenaState`: owns exact-band fields, material slices, ports, borders,
   iteration counters, convergence, and checkpoints.
3. `WaveBackend`: CPU, GLSL, or Torch implementation behind one lifecycle.
4. `WaveDisplayPublisher`: publishes progressive complex-field-derived images
   without readback controlling solver progress.
5. `WaveScheduler`: queues arena work and performs explicit ray/field handoff.

## Surface-scan raster shunt

A separate deterministic preview stage is approved. It is not T4 and not a
path integrator.

For each output pixel it generates the center-site camera sample, transports it
through the existing physical/parametric lens contract, intersects the first
ordinary scene surface, and evaluates directly visible material radiance plus
scan/debug channels. It terminates there: no stochastic bounce tree, BDPT, or
wave propagation.

The stage owns two output surfaces. While one is displayed, the other is being
written; publication swaps them only after a complete generation. This makes
it useful for calibration, framing, material inspection, and rapid parameter
dialing while retaining lens distortion, vignetting, focus, and occlusion.

The first implementation is `submit_surface_scan()` plus
`surface_scan_resolve.comp.glsl`. Center-site camera intents are cached until
the camera pose or sensor resolution changes, traverse the normal pipeline and
parametric lens stage, then shunt the first ordinary surface to an RGBA32F
material/scan-light resolver. Misses remain alpha-zero. Two shared textures use
build, pending, and adopted ownership; scan subbatches accumulate into one back
generation and publish only when the otherwise-idle scan submission completes.
BDPT record capacity is zero for this path, so it does not reserve or fill the
large path-connection record store.

GPU T1 now resolves the nearest arena-boundary or triangle event and emits a
packed `WaveIntent` tail through its already-bound hit SSBO. Arena-enabled
surface scans therefore follow the same T1→T4→T1 pipeline route instead of
silently bypassing field volumes or requiring CPU T1.

## Complex transport sidecar ABI

`complex_transport.h` and `complex_transport.glsl.inc` define the versioned
CPU/GPU boundary schema. Ordinary geometric rays retain their existing compact
pipeline representation. A ray crossing a wave port gains one 48-byte CPU
sidecar containing frequency, spectral PDF, coherence/sample identity, optical
path slot, transport Jacobian, source lane, and validity flags. Amplitudes stay
on the ray and are not duplicated by CPU scheduling.

T4 compacts compatible samples into exact `1, 3, 4, 8, 16, 32` lane blocks.
The complete boundary lane reserves scalar/Jones complex amplitudes and packs
to four std430 `vec4` slots (64 bytes) on GPU. Phase-sensitive frequency and
optical-path values use float-float high/low words without requiring shader
fp64. A producing GPU stage writes wave records into a tail section of an SSBO
it already binds; T4 rebinds that storage after the producer completes. This
avoids a ninth simultaneous compute-SSBO binding and avoids charging every
non-wave ray for an N-lane payload.

The generic SSBO rechanneler is the preferred single-source CPU/GLSL mechanism
for the dense, fixed WaveIntent-to-complex-lane reshape after selection or
cohort indices exist. It is suitable for a hot dedicated pre-pass: descriptors
are built and shaders compiled during setup, not in the dispatch loop. It does
not itself perform predicates, prefix compaction, sorting, or cohort grouping;
the producing stage or scheduler must supply a dense record range or scatter
indices first. Although its source documentation and tests define canonical T5
repacks, the current production ray pipeline does not invoke it; T5 presently
uses dedicated sort/pack/scatter shaders. Wave integration should make the
rechanneler an actual pipeline facility rather than duplicating its remapping
logic by hand.

The orphaned `t0_intent_pack.comp.glsl` raw-object experiment is separately and
explicitly retired: it is unwired, cannot dereference Eigen amplitude storage,
and predates the current intent layout. It is not the SSBO rechanneler.

### Jones and differential operators

`COMPLEX_OPTICAL_OPERATOR_CONTRACT.md` defines the canonical boundary math.
Complex transport schema v2 keeps the lane at 64 bytes and assigns its former
reserved words to stable basis/operator indices. Persistent arena-owned tables
hold right-handed transverse bases, complex 2x2 Jones operators, and signed
canonical 4x4 tangent maps.

The reference and shader ABIs now agree on basis rotation, Jones application,
power-normalized dielectric Fresnel scattering, canonical phase-space
coordinates, reference-OPL carrier phase, and caustic-safe ray/field gain. The
native exact-lens helper returns full spectral-lane-specific 4x4 maps plus
determinant and symplectic residual. Production T4 entry now resolves a stable
ray tag through a compact binding into the single-allocation, deduplicated
source/basis/operator state and seeds both s and p for fixed bands and
continuous cohorts. Legacy rays with no source record remain explicitly
s-only. T4 still reduces s/p output to a scalar ray; replacing that exit
adapter is an explicit remaining gate.

A shared `JonesFieldState` reference adapter now keeps coherent source modes
separate, drives both s and p components through the production native T4
aperture-material and angular-spectrum kernels, and accumulates only
intensities into Stokes/analyzer views. The physical-aperture live demo uses
that adapter to qualify vector transport. This demonstrates that the existing
T4 kernels can carry the state; it does not claim that the scalar production
arena-to-ray adapter has been replaced.

The aperture qualification client now surrounds its visible crop with a
power-of-two hidden solve domain and applies the production numerical exterior
after each propagation substep. Its balanced/high/bake tiers deliberately
trade 4x/16x/64x visible-sample memory for boundary distance and convergence.
This removes the former unpadded calibration FFT's square periodic-box
signature without introducing a demo-private propagation or absorber.

## Removal gates

Legacy production paths are deleted only after the replacement passes:

- CPU/GLSL parity for every supported band specialization.
- Plane wave and Gaussian-beam propagation.
- Physical two-blade slit and finite-thickness baffle diffraction.
- Dielectric slab Fresnel transmission/reflection and absorption.
- Prism dispersion and thick-lens focus.
- Forward/backward reciprocity.
- Open-boundary reflection/leakage limits.
- Mirror-box energy accounting and convergence.
- Live display/checkpoint continuation without changing the solution.
