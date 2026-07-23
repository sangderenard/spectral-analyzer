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

Available backend declarations may include scalar FFT, split-step FFT,
ADI-reference, and Maxwell-patch modes. A declaration selects an existing
backend; it does not put that backend's numerical implementation in the graph.

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

The installed production code currently identifies itself as
`LegacyAdiCalibration`. Therefore only graph nodes explicitly declaring
`adi-reference` can be installed today. An `angular-spectrum-fft` declaration
is rejected rather than silently executed by ADI. The FFT declaration remains
the intended production default, not a claim about the present native kernel.

The shared `transport.complex-accumulation` 3D texture is currently resolved
from ray-hit accumulation. It is an honest complex-ray/path layer, but it is
not a readout of the persistent T4 state block. A distinct T4 state texture is
required before a UI or capture may label an image as a wave-arena solution.

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
   - replace readback-steered arena routing with GPU-resident cohort assembly.
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
- T4 descriptors can construct the current native ADI-reference arena, but
  the planned vector angular-spectrum FFT backend is not implemented.
- Ray/field adapters are declared but not yet implemented as production GPU
  kernels.
- T4 state is not yet published as a zero-copy OpenGL texture; the existing
  complex volume is ray-hit accumulation.
- Maxwell artifact nodes are reserved contract space, not an implemented patch
  compiler.

These limitations must remain visible. Demonstrations must not label graph
topology validation as a completed wave solve.
