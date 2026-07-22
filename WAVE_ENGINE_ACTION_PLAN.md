# Wave engine action plan

Date: 2026-07-22  
Baseline before the current wave/T1 work: `c4cb29f`

## Destination

There are two electromagnetic engines with one shared transport contract:

1. **Ordinary wave context:** a bidirectional, two-component vector
   angular-spectrum Helmholtz engine. This is the production T4 engine for
   camera-scale diffraction, interference, polarization, lens propagation,
   apertures, and authored material interfaces.
2. **Maxwell patch context:** a localized microscopic solver/compiler for
   structural colour, metasurfaces, gratings, and other features for which a
   bulk index or ordinary interface law is insufficient. It produces a cached
   polarized scattering artifact; it is not a whole-camera propagation mode.

Both use the existing pipeline. T4 additionally owns a persistent contiguous
state block supplied to its backend as buffers. Neither engine allocates a
solver or resizes state in the hot dispatch sequence.

## Why this ordinary engine is the choice

The ordinary engine is not generic BPM and not a scalar-only stepping scheme.
It represents each lane as two transverse complex electric-field components.
In homogeneous spans, an angular-spectrum step diagonalizes propagation into
plane waves and applies the exact Helmholtz longitudinal phase for every
supported spatial frequency. The longitudinal field follows from
transversality. At physical interfaces, forward and backward fields are
coupled by explicit Jones/modal scattering operators.

This gives the useful middle ground:

- substantially more correct at high numerical aperture than scalar BPM;
- exact homogeneous propagation up to sampling and boundary error;
- natural diffraction, defocus, evanescent-policy, and polarization handling;
- efficient FFT-heavy GPU execution with fixed lane specializations;
- compatible with rays at ports and with localized Maxwell-derived patches;
- no claim that a camera-sized visible-light volume is being solved by full
  volumetric Maxwell discretization.

Scalar fields remain a deliberate fast specialization for scalar-valid
calibrations. They do not define the permanent ABI.

## Ordered action plan

### 0. Freeze and establish the live baseline

- Stop adding numerical behavior until the current accumulated ray/T1/T4 work
  passes live application acceptance.
- Exercise ordinary prism/no-arena, continuous 1/4/16/32 lanes, arena entry and
  exit, surface scan, full flash/sensor BDPT, visible publication, VRAM, and
  throughput.
- Record stale diagnostics, legacy paths still reached, and any CPU readback
  that gates GPU progress.

Exit gate: the present ray renderer is visibly correct and its before/after
cost is measured, not inferred from focused regression tests.

### 1. Seal the shared complex transport contract

- Keep ordinary rays compact; attach complex data only at wave-capable ports.
- Promote the existing reserved `s/p` amplitudes into the required two-
  component boundary representation. Define the local polarization basis and
  handedness unambiguously.
- Preserve frequency high/low, optical path high/low, PDF, Jacobian, coherence
  identity, sample identity, source lane, and validity flags.
- Add medium/port/material/artifact identity without inflating every geometric
  ray. Use compact sidecars and post-selection rechanneling.
- Generate the C++/GLSL/Torch layouts and constants from one schema.

Exit gate: CPU/GPU/Torch round trips preserve phase, basis, identity, and all
exact lane widths `1, 3, 4, 8, 16, 32`.

### 2. Build the T4 ownership shell before replacing its mathematics

- Finish `WaveSceneCompiler`, `WaveArenaState`, `WaveBackend`,
  `WaveDisplayPublisher`, and `WaveScheduler` as independently testable parts.
- Keep one solid persistent state allocation per arena with stable offsets.
- Move arena compaction/cohort assembly GPU-side so production progress is not
  controlled by per-bounce counter or payload readback.
- Preserve fixed-band and continuous-cohort semantics in every specialization.

Exit gate: the legacy ADI calibration backend and a no-op/reference backend can
run through the same state lifecycle without hot allocation.

### 3. Implement the vector angular-spectrum backend

- Store forward/backward transverse complex fields in the arena state block.
- Use padded FFT planes, exact `kz` propagation in homogeneous spans, explicit
  propagating/evanescent policy, and energy-normalized transforms.
- Compile scene geometry into ordered ports, homogeneous spans, smooth-volume
  slices where needed, and physical material interfaces.
- Apply Fresnel/Jones/modal interface scattering; never use ideal aperture
  masks for authored blades or baffles.
- Use split-step material phase only for genuinely smooth inhomogeneous spans,
  not as a substitute for sharp-interface scattering.
- Publish progressive intensity, phase, polarization, residual, and boundary
  diagnostics without readback steering the solve.

Exit gate: plane/Gaussian propagation, physical double slit, dielectric slab,
prism, thick lens, reciprocity, and open-boundary reflection gates pass.

### 4. Establish the Maxwell patch compiler contract

- Add `MaxwellPatchContext` as an offline/calibration job built from localized
  USD microgeometry, complex tensor material laws, excitation ports/modes,
  boundary conditions, frequency/angle sampling, and differentiable
  parameters.
- Permit specialized backends (for example periodic modal methods or
  unstructured full-wave methods) behind the context; do not encode a solver
  algorithm into the authored material.
- Compile a versioned artifact containing a complex polarized bidirectional
  scattering operator, interpolation domain, normalization, error estimates,
  passivity/reciprocity evidence, and complete provenance.
- Make compilation differentiable where the selected backend supports it. The
  existing metric-aware parallel Laplace--Beltrami infrastructure may supply
  mesh/discretization machinery, but it qualifies only after compatible
  vector curl/divergence and material/interface operators are demonstrated.

Exit gate: an independent verifier reproduces the artifact's calibration
cases and rejects artifacts outside their declared domain.

### 5. Connect Maxwell artifacts to materials and ordinary T4

- A material may carry a cold `maxwell_patch` authoring declaration. The hot
  material tensors remain unchanged.
- Scene compilation resolves that declaration to a cached artifact ID and a
  compact hot lookup table used at the affected surface/volume patch.
- Ordinary rays and T4 fields query the same artifact contract. Rays sample
  its polarized directional distribution; T4 applies its coherent modal/Jones
  operator. Detector accumulation remains coherence-correct.
- Missing, stale, or out-of-domain artifacts fail loudly or use an explicitly
  authored bulk fallback. They never silently become RGB texture colour.

Exit gate: the same structural-colour patch agrees under ray sampling, ordinary
wave illumination, and direct Maxwell verification within declared error.

### 6. Retire legacy authority and optimize

- Remove legacy optical wave paths only after the documented removal gates.
- Keep acoustic grid/pressure infrastructure independent of optical removal.
- Profile FFT batching, lane occupancy, arena scheduling, material-artifact
  locality, and publication. Cache immutable scene transforms and compiled
  scattering tables; overlap ingestion, GPU work, publication, and checkpoint
  output.
- Specialize and unroll exact lane widths without allocating 32 lanes for a
  smaller mode. Continuous multi-lane work retains per-lane complex metadata.

Exit gate: scientific gates, live acceptance, memory ceilings, and throughput
budgets are all recorded for every production backend.

