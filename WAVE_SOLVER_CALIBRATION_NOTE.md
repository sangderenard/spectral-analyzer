# Wave Solver Calibration Note

## Current capability

The repository already contains executable optical-wave foundations. They are
not yet a complete wave camera, but they are sufficient for an immediately
testable calibration workflow.

- The native extension exposes an exact scalar Rayleigh-Sommerfeld plane-to-
  plane propagator.
- The native extension exposes a CPU ADI Crank-Nicolson beam-propagation
  method (BPM) stepper.
- `csrc/shaders/ray_wave_bpm.comp.glsl` implements the corresponding GPU BPM
  step.
- `camera_designer/wave_tube.py` contains an independent NumPy BPM reference
  and a ray/field handoff prototype.
- Ray transport already carries complex amplitude and optical path length.
- Complex field capture, coherent plane accumulation, aperture masks, and
  field-display shaders already exist.

An audit-time double-slit test confirmed that the compiled CPU BPM and NumPy
reference agree to approximately `4.8e-4` relative L2 error, preserve power in
the bounded test, and produce symmetric interference fringes. The compiled
Rayleigh-Sommerfeld propagator also produced a finite symmetric fringe field.

## Known integration gaps

The live exposure renderer must not advertise a production wave camera yet.

- `CAMERA_MODE_WAVE_ASSEMBLY` remains explicitly marked as a stub.
- `WaveTube.register()` uses a stale `center=` keyword for a binding that now
  requires `pos=`, and silently discards the resulting exception.
- GPU T4 uses `arena.nz` as its transverse `ny` dimension.
- CPU T4 seeds, marches, extracts, and requeues a ray; GPU T4 currently only
  dispatches a field step and retires the incoming work item.
- CPU/GPU band limits disagree (32 arena bands versus 16 shader bands).
- Default wave-context payloads do not provide the camera optical axis, so the
  arena falls back to world `+Z`.
- Exact parametric camera transport clears surrogate wave contexts to prevent
  a second, invalid steering operation.
- The compiled 3-D field-grid Helmholtz marcher is not exposed to Python or
  called by the exposure renderer.
- The EM `FieldSolver` is a coherent ray/Jones transfer solver; it is not a
  spatial transmissive-glass wave solver.

## Shared durable calibration lifecycle

Every calibration room starts in the Calibration Rooms hierarchy. Launching a
room creates or reopens a configuration-addressed record under Work / Assets.
The record owns its manifest, launch history, status, result paths, and a
stable filesystem home. A repeated compatible ray calibration continues its
retained sensor sum/weight; an interrupted wave calibration continues its
complex-field step checkpoint. Cancel means checkpoint, not delete. Double
Slit is one room using this shared lifecycle, not a special durability case.

## Double Slit Room

The first wave calibration should isolate propagation from glass and Monte
Carlo transport. A coherent source illuminates a physical double-slit aperture,
and the resulting complex field is propagated to a sensor plane independently
by the CPU and GPU BPM implementations.

Every work revision must retain:

- The exact work manifest and solver parameters.
- The entry aperture field.
- CPU and GPU complex output fields (`real` and `imaginary`).
- Per-band intensity images.
- A CPU band strip and GPU band strip using the same normalization.
- A combined comparison image.
- Symmetry, power, fringe-spacing, finite-value, and CPU/GPU agreement metrics.
- Checkpoint/progress state sufficient to resume uncompleted bands or steps.

The analytical first-order fringe spacing is

`fringe_spacing = wavelength * propagation_distance / slit_separation`.

The room is launched from the calibration hierarchy. Once launched, its work
record belongs to the durable work/assets system: it remains visible after
completion, can be selected for inspection, and an interrupted revision can
resume without erasing completed band products.

The verified 200×200 example uses five bands from 450–650 nm and 100 steps.
Its CPU/GPU intensity fields agree to about `2.9e-5` relative L2. The GPU
record names the NVIDIA renderer, OpenGL version, exact shader path, and shader
SHA-256, so a missing GPU execution cannot silently masquerade as CPU output.

## Retained evidence suitable for shimmer experiments

Ray exposures already retain the unbiased ingredients needed to visualize
how integration changes without modifying the actual exposure:

- linear sensor sum and exposure weight;
- normalized linear image and sample-count raster;
- requested-priority/noise maps;
- the current and preceding progress layers (foreground currently retains
  three layers; bounded atlas work retains two);
- convergence deltas and refinement-pass history.

Wave calibration additionally retains entry, CPU, and GPU complex fields as
separate real/imaginary arrays plus per-band intensity. This makes phase motion
available for diagnostic visualization even though the detector product is
still `|E|²`.

A truthful Monte Carlo "twinkle" should be a presentation layer driven by the
arrival delta, not random glitter painted over the image. For two consecutive
unbiased estimates `L_prev` and `L_now`, compute a robustly normalized signed
delta and briefly brighten positive arrivals while cooling negative revisions;
decay that overlay over a few display frames. Weight it by the retained
uncertainty or priority map so converged regions settle down naturally. The
saved sum, weight, and final developed image remain untouched. For wave work,
a separate diagnostic shimmer can animate phase hue while intensity controls
brightness; that should be explicitly labelled "phase view," because real
sensor intensity does not visibly oscillate at optical frequency.

## Route toward a wave camera

After the Double Slit Room is trustworthy, the same field contract can be
placed at the camera pupil:

1. Accumulate a phase-bearing complex field at the physical aperture.
2. Apply the real aperture/blade mask and lens OPL-derived phase.
3. Propagate pupil to sensor with Rayleigh-Sommerfeld or BPM.
4. Convert to intensity only at detection: `I = |E|^2`.
5. Sum unrelated temporal/spatial coherence groups incoherently.

Full prism wave transport additionally requires a spatial refractive-index
operator `n(x,y,z)` or explicit wave interface/phase-screen operators. The
current homogeneous BPM is already useful for diffraction, interference,
defocus, Airy structure, and pupil effects, but it is not that prism solver.
