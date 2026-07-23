# Optical transport graph integration contract

Status: compiler/ABI scaffold with native T4 installation  
Schema: `optical-transport-graph-v1`

## Purpose

The optical transport graph gives the existing `GraphSolver` authority over
modular optical topology without moving numerical solvers into the graph
runtime. It connects the existing pipeline systems:

- T1 discovers geometric transport boundaries.
- T2 executes compiled parametric/network segments.
- T3 performs ordinary material interaction and ray continuation.
- T4 owns persistent spatial wave fields.
- Maxwell patch compilers produce cached scattering artifacts used by T2/T4.

`MultiEngineRenderGraph` remains the outer engine/job coordinator. It is not
the optical module graph.

## Non-negotiable ownership

The graph owns:

- named ports and products;
- topology, branching, and bidirectional connectivity;
- representation and lane compatibility;
- causal delays and SCC scheduling;
- cold compilation and artifact identity;
- selection of an execution domain for each module.

The graph does not own:

- conic intersection or Snell/Fresnel mathematics;
- ray/BVH traversal;
- FFT or spatial field propagation;
- Maxwell discretization;
- hot allocation;
- texture presentation.

The shader pipeline never interprets arbitrary Python graph nodes. Cold
compilation lowers graph segments to compact artifacts and schedules consumed
by specialized pipeline runners.

## Execution domains

### Pipeline port

A typed boundary between the ordinary ray pipeline and a compiled segment.
Ports may carry ordinary rays, complex-ray sidecars, transverse field planes,
or cached polarized scattering operators.

### T2 parametric

Local algebraic or compiled network transport. The first authoritative module
is `fused-exact-compound-lens`, whose artifact is the existing magic-14949
exact-conic payload. Graph adoption must not replace it with a generic shader
interpreter or tessellated refraction.

Future T2 modules include:

- exact propagation spans;
- individual conic/material boundaries;
- physical aperture and baffle boundaries;
- mirrors and beam splitters;
- cached Jones/modal operators;
- Maxwell scattering artifact applications;
- explicit rechannel and basis-change operations.

Adjacent compatible nodes may be fused at cold compile time.

### T4 wave arena

Spatial field propagation with one persistent, solid, contiguous state block
per arena. State is allocated/resized outside the hot dispatch sequence.

The production default is:

- two transverse complex field components;
- bidirectional vector angular-spectrum Helmholtz propagation;
- FFT execution for homogeneous spans;
- padded absorbing borders;
- explicit Jones/modal scattering at physical interfaces.

Available backend declarations include the production angular-spectrum
backend, reserved split-step spans, and Maxwell-patch artifacts. A declaration
selects an existing backend; it does not put that backend's numerical
implementation in the graph.

## FFT and border policy

`angular-spectrum-fft` is the default propagation style, not a declaration of
periodic physical boundaries. The default border is `padded-absorbing`:

1. map the physical port into a padded field plane;
2. apply the declared absorbing edge treatment;
3. propagate with energy-normalized FFT operators;
4. crop/publish the declared physical port.

Periodic borders require explicit authoring. Raw FFT wraparound is never an
implicit fallback.

Sharp material interfaces, physical blades, and baffles are boundary
operators/geometry. Split-step material phase is reserved for genuinely smooth
inhomogeneous spans.

## Shared port ABI

Every port declares:

- representation;
- execution domain;
- exact lane count (`1, 3, 4, 8, 16, 32`);
- one or two complex field components;
- forward/backward/bidirectional behavior;
- persistent-state requirement;
- representation normalization and basis through its module contract.

The generated native ABI must additionally preserve:

- frequency high/low;
- optical path high/low;
- local s/p basis and handedness;
- PDF and phase-space Jacobian;
- coherence, sample, source, and lane identity;
- medium, material, port, and artifact identity;
- validity and transition flags.

Ordinary geometric rays remain compact. Extended complex state is attached
through the existing sidecar/rechannel mechanism only when a graph port
requires it.

## Representation transitions

Representation changes must be explicit adapter modules or explicitly named
link adapters. Required adapters include:

- ray to complex-ray sidecar;
- complex-ray bundle to transverse field;
- transverse field to outgoing complex-ray distribution;
- material/Jones basis change;
- fixed-lane rechannel;
- Maxwell scattering artifact to ray or field result.

Direct complex-ray-to-field links without an adapter are invalid.

The initial wave context is therefore:

```text
complex ray
  -> complex-ray-to-transverse-field
  -> persistent T4 arena
  -> transverse-field-to-complex-ray
  -> complex ray
```

## Branching and bidirectional systems

Mirrors, beam splitters, interferometers, folded lenses, and center-obscured
systems are graph branches. Each product edge retains its semantic role
(`reflected`, `transmitted`, `diffracted-order-N`, and so on).

Zero-delay feedback retains `GraphSolver`'s exact/vectorized SCC path. Declared
causal delay uses its FIFO/delayed schedule. Spatial wave marching remains T4
state and must not be emulated by graph sample delay.

## Cold compilation products

`compile_optical_graph()` currently produces:

- a `GraphSolver` topology used for structural/SCC validation;
- T2 payloads keyed by graph node;
- T4 arena descriptors keyed by graph node;
- a serializable contract for diagnostics and future native code generation.

`RebuiltCameraArtifact.compile_transport_graph()` opt-in compiles the current
camera without adding graph/Torch startup cost to ordinary rendering.

The current compound-lens lowering must preserve its source payload exactly.
This equivalence is a regression gate.

`install_optical_graph()` now performs the first graph-to-pipeline lowering:

- it requires the authoritative camera builder to confirm that the exact T2
  artifact was registered; it never installs a surrogate thin/thick lens;
- it clears non-graph scale contexts when requested and installs each physical
  graph T4 region through the existing native scale-context ABI;
- it retains borrowed axis payload storage for the installation lifetime;
- it returns node-to-context receipts and identifies native
  `wave_arena_stats()` as the transition-telemetry authority.
- direct T4-field-to-T4-field graph edges lower to native persistent field
  links. Forward transport follows source to destination and backward
  transport follows the reciprocal edge. Cold pipeline construction rejects
  branching, cycles, lane/grid mismatch, basis/index mismatch, or
  noncoincident planes; it never inserts an implicit resampler.

The installed production code identifies itself as `AngularSpectrum`.
`angular-spectrum-fft` graph nodes lower directly to the exact-lane native
arena. The arena owns padded forward/backward S/P complex planes in one
persistent contiguous block. The current native transform executor is CPU
radix-2; GPU-routed work uses that same kernel on the GL owner thread until the
staged GLSL FFT plan is attached. Unsupported split-step and Maxwell
declarations fail explicitly.

The shared `transport.complex-accumulation` 3D texture is resolved from ray-hit
accumulation. It is an honest complex-ray/path layer, but it is not a readout
of the persistent T4 state block.

The separate `wave.arena-state` 2D texture is now resolved from one selected
band and direction of that persistent block. Hue encodes phase and value
encodes fourth-root field power; alpha is zero only where field power is zero.
It is a presentation product, while `wave_arena_field_snapshot()` is the
bounded raw complex-field calibration/capture API. The display resolve is
opt-in through the optical request's `wave_arena` product so an unobserved
arena does not pay CPU staging or texture-upload cost.

## Ray/field boundary contract

An installed T4 region is an oriented plane-to-plane patch. `radius_m` is its
transverse half-extent; its longitudinal extent is exactly
`longitudinal_step_m * longitudinal_steps`. CPU and GPU T1 intersect that same
oriented box, so routing geometry cannot silently disagree with the distance
advanced by the angular-spectrum kernel.

The current scalar entry adapter deposits each complex-ray lane as a compact
three-cell Gaussian reconstruction kernel. The kernel has unit discrete L2
power and carries the ray's transverse phase ramp, preserving input power,
phase, position, and paraxial direction. This is explicitly an adapter kernel:
a geometric ray contains no authored beam waist.

The current exit adapter emits one representative complex ray per lane using
the field's power centroid and global phase-correlation direction. Its complex
amplitude magnitude is the square root of total lane power. This preserves
power but intentionally discards higher spatial moments; graph contracts call
it `power-preserving-first-moment-ray`. Systems requiring the complete exit
field must connect it to another field/operator port rather than use this
reduction.

That complete-field connection now exists for identity-compatible ports. It
copies active S/P complex planes directly between already-allocated contiguous
state blocks and marches the destination before any ray extraction. There is
no hot allocation, and only the terminal field port emits a representative
ray. `linked_transfers`, `next_forward`, `next_backward`, and boundary
`linked_from_arena`/`linked_to_arena` telemetry make the route observable.

`wave_arena_stats()` exposes entry/exit world and local coordinates,
directions, input/seeded/propagated/output power, adapter identity, and the
transition generation without adding per-ray state to ordinary pipeline
records.

## Projector-back source plane

Reverse projection uses the camera's existing large back plane; it is not a
point light and does not instantiate another tracer. `ProjectorBackSpec`
lowers to the common `EmitterProfile` contract and may therefore author
spectrum, coherence, carrier phase, angular distribution, polarization, and
an emissive/Jones texture. Fixed-lane scalar complex launch is currently
implemented. Full Jones launch and continuous-frequency source sidecars remain
explicit follow-on ABI work.

The physical order is:

```text
projector back source plane
  -> spectrally colored transmissive sensor scrim
  -> reciprocal exact compound lens
  -> optional scene-side T4 regions
  -> projected scene field
```

Projector mode retains the sensor geometry as an authored transmissive scrim
with per-lane transmission and diffusion. It removes only the opaque
sensor-bay back wall. Ordinary camera mode retains the absorbing sensor and
back wall.

`compile_projector_back_graph()` declares the reverse route through the same
magic-14949 T2 payload. Native intents use a dedicated reverse-optics flag,
separate from the sensor/BDPT-backward bit, so a physical projector source
does not acquire sensor ownership or sensor splat semantics. The source plane
is sampled across its full area and aimed through the analytically derived
exit pupil. Current pupil quadrature uses the center site; finite pupil-fill
and illumination-optics modules are the next source-optics layer.

## Adoption gates

1. **Contract scaffold**
   - typed nodes, links, domains, representations, lanes, FFT/border policies;
   - exact payload equivalence;
   - branching and invalid-transition tests.
2. **Native graph installation** *(first vertical slice complete)*
   - physical T4 placement and axis lower into native scale contexts;
   - exact T2 registration is externally attested, not duplicated;
   - native transition telemetry proves graph-declared T4 generations.
3. **Generated native ABI**
   - generate C++/GLSL/Torch layout constants from one schema;
   - round-trip every supported lane width.
4. **T2 schedule runner**
   - execute a short compiled opcode/fused-segment schedule;
   - retain the exact compound-lens fused operation;
   - prohibit hot allocation and generic per-node interpretation.
5. **T4 transition runner**
   - use graph port metadata to seed/extract persistent arenas;
   - compatible full-field arena links are complete on the native CPU
     executor; replace readback-steered arena routing with GPU-resident cohort
     assembly.
   - bounded raw field snapshots and an opt-in shared display texture are
     complete; direct GPU FFT state-to-display resolution remains.
6. **Scientific equivalence**
   - compare graph-compiled camera transport against the current exact camera;
   - pass reciprocity, energy, phase, polarization, prism, aperture, and lens
     calibration gates.
7. **Authority transition**
   - make graph compilation the canonical camera/bench assembly path only
     after equivalence and performance gates pass;
   - retire duplicate assembly logic after live acceptance.

## Current limitations

- The graph contract is Python-side and not yet generated into native layouts.
- T2 consumes the exact lens payload directly; it does not yet consume a
  multi-operation graph schedule.
- T4 descriptors construct the vector angular-spectrum arena, but the native
  GPU FFT executor is not implemented; the production CPU kernel is used.
- Persistent field links currently support identity-compatible ports only.
  Basis rotations, refractive interfaces, grid changes, and branch operators
  require explicit compiled field operators and remain intentionally rejected.
- The live T4 texture is currently staged from CPU-resident production state.
  It is never generated unless requested, but will become a direct GPU resolve
  when the GLSL FFT executor owns the arena state.
- Ray/field adapters are declared but not yet implemented as production GPU
  kernels. The CPU entry/exit adapter currently maps legacy scalar ray
  amplitude to S polarization; full Jones sidecar mapping remains.
- Projector launch currently maps its authored frequency to an exact fixed
  lane. Continuous-frequency source launch needs the planned frequency
  sidecar; it is not represented by an arbitrary nearest-lane approximation.
- Projector pupil quadrature currently aims each source-plane site at the exit
  pupil center. Finite pupil-fill, condenser/relay optics, and textured Jones
  field sampling remain light-source-optics work.
- T4 state is not yet published as a zero-copy OpenGL texture; the existing
  complex volume is ray-hit accumulation.
- Maxwell artifact nodes are reserved contract space, not an implemented patch
  compiler.

These limitations must remain visible. Demonstrations must not label graph
topology validation as a completed wave solve.
