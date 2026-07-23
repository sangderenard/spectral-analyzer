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

## Graph and presentation authority

Do not introduce another optical solver or another general graph beside the
systems already responsible for those jobs:

- `GraphSolver` plus `CompiledGraph` is the complex network/module graph used
  by audio, controls, and future parametric EM network regions. It owns tensor
  network causality and SCC scheduling; it does not own spatial wave marching.
- `MultiEngineRenderGraph` coordinates independently persistent engines. It
  brokers requests and products; it does not reproduce their algorithms.
- `RayPipelinePreviewBridge`, `PreviewProductRegistry`, and
  `GLPreviewCompositor` are the GPU-resident presentation path. Producers
  publish immutable shared texture descriptions; viewers never invent a
  substitute field merely to fill a tab.
- A native `transport.complex-accumulation` product is complex amplitude
  carried by rays and accumulated in a volume. It is useful diagnostic data,
  but it is explicitly not a T4 wave solution.
- A product may be labelled as a wave field only when it is published by a
  `WaveBackend` operating through the persistent `WaveArenaState` lifecycle.

Standalone demonstrations must be thin hosts or request clients of these
authorities. Reference numerical kernels may exist as calibration backends,
but must identify their backend and must not masquerade as production engine
integration.

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

The concrete cross-system graph ABI and staged lowering gates are documented in
[`OPTICAL_TRANSPORT_GRAPH.md`](OPTICAL_TRANSPORT_GRAPH.md). Its initial
`optical-transport-graph-v1` scaffold validates topology through `GraphSolver`,
retains the existing exact compound lens as a fused T2 artifact, and declares
explicit T4 wave entry/arena/exit nodes. It changes no numerical solver
authority.

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

### 0A. Give rendering a durable process lifetime

- Replace repeated exposure subprocess initialization with a small local render
  service. The UI rasterizer submits immutable scene/content revisions and
  receives progress plus double-buffered image publications.
- The service owns the long-lived GL context, compiled shader variants, SSBO
  arenas, scene/material caches, T4 state, checkpoints, and output writers.
- Keep job messages versioned and content-addressed so reconnect/restart is
  deterministic. A UI disconnect must not destroy accepted work; a service
  crash must not corrupt the last completed checkpoint.
- Use bounded queues and explicit cancellation/revision supersession. Do not
  turn the server into a second renderer or let network serialization enter a
  shader hot loop.
- Start with loopback-only transport and a narrow authenticated control surface.
  The useful boundary is process lifetime and persistence, not remote exposure.

Exit gate: successive UI renders reuse one verified GPU owner and its caches,
while restart/resume reproduces the same manifest-addressed result.

### 1. Seal the shared complex transport contract

- Use the graph engine's `complex-network-context-v1` surface for parametric
  and network regions: named complex entry ports, named complex products,
  persistent state, explicit sample rate, and declared causal delays.  This is
  a context boundary, not an audio projection; complex128 tensors and arbitrary
  batch/time/channel shapes remain intact.
- Keep zero-delay networks on their exact/vectorized SCC path.  Acyclic delayed
  channels remain vectorized shifts over the work window; only delayed
  feedback requires causal rollout.  A context must report which schedule it
  selected so a wave/parametric transition cannot silently change latency.
- Define the wave-region adapter against the same entry/product vocabulary.
  The adapter is responsible for basis, normalization, port geometry, lane
  identity, and ray/field conversion; the graph solver must not pretend that a
  scalar complex port is already a transverse Maxwell field.
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

Exit gate (completed 2026-07-23): one padded, exact-lane, bidirectional S/P
state block runs through the pipeline lifecycle without hot allocation. The
discarded calibration backend is available only through Git history.

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

Transform execution is backend-owned under the shared field ABI:

- Torch-owned CUDA contexts should use `torch.fft`/cuFFT rather than a custom
  transform;
- GL-owned contexts require the staged GLSL FFT so fields do not cross an API
  boundary or synchronize through Python;
- the vendored `third_party/fftfree` complex Cooley--Tukey engine is the
  planned vectorized native CPU executor/reference because it already supports
  in-place batched axes, explicit strides, preplanned workspace, 2D transpose,
  caller-owned dispatch, and transform telemetry;
- Stockham and mixed-radix variants remain opt-in until isolated parity and
  no-hot-allocation gates pass. CQT/VQT/NSGT and wavelet engines remain
  frequency-analysis operators, not substitutes for the spatial 2D DFT.

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
