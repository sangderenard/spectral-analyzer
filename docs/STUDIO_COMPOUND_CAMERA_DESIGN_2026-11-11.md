# Compound cameras in the Studio sensor path — 2026-11-11

## Current measured state

The Studio profile selector in `turing/examples/meowdoku/marble_lens_contract.py`
admits authored analytical LensSpec profiles and lists compound prescriptions
as references. Its fixed camera still launches thin-lens rays in
`demo_pluck_gl.SensorAccumulator`: sensor pixel plus a sampled aperture point
determines a world ray, and the same compute shader traces the scene. A
separate, older manifold branch substitutes one precomputed **direction per
pixel** while leaving origin at camera eye. That branch cannot represent
pupil samples, off-axis origin, spectral dispersion, pupil rejection, or lens
throughput; it is not a compound-camera adapter.

The exact source prescription and parametric region evaluator already exist in
`camera_designer.compound_optics.CompoundLens`, `LensAssemblySpec`, native T2,
and `camera_software.optical_transport_graph.compile_compound_lens_graph`.
The compiled graph is currently opt-in; `camera_build.py` says the ordinary
pipeline does not consume its schedule directly. The existing LUT bake is a
front-side position/direction table. It omits wavelength, throughput and
backward sensor-side sampling, so it cannot be substituted for the eight-band
Studio camera without extending that same contract and verifying it.

Fresh bounded reruns of the existing 2026-11-09 trial drivers, with only output
paths changed:

| Path | Work | Wall time | Evidence |
|---|---:|---:|---|
| Native exact CPU BDPT/VCM | 64×64, 8 pupil samples, 32,768 sensor rays, 2,784 forward rays | 31.993 s (5.982 s setup, 25.609 s active T5) | finite, 2,127 lit pixels; T5 fired, no upstream overflow; diagnostic connection queue truncated |
| Camera software geometric light table through production compound lens | 128×128, 30,000 rays | 2.722 s | finite, 3,865 lit pixels, recognizable chart; RGB passthrough, no BDPT or spectral sensor physics |
| Existing Studio thin-lens fixed:8 saved job (different scene) | 128×128, 327,680 sensor sites | 5.062 s elapsed, 2.422 s setup | complete HDR/PNG, two sensors; different scene and integrator, so no speed ratio is justified |

Rerun artifacts: `exposures/compound_camera_trial_20261111/` and
`exposures/compound_camera_table_20261111/`. The exact output is still
uncalibrated magenta; finite output does not certify color, exposure or
convergence. Neither rerun used the GPU compound path or wave regions.

## Camera transfer boundary

### Correction after tracing the native Stage 5 path

The 31.993 s trial does **not** measure conic traversal as the bottleneck.
Native T5 is the bidirectional all-pairs connection estimator and took 25.609 s
of active time. The Studio photo path already uses a different connection
strategy: its sensor compute shader reads the prepared forward field, samples
the actual emitter population with NEE and visibility checks, and continues
camera paths through scene materials. That program does not call native T5.
The direct area-NEE view is derived from the same original source records used
by the forward pass. Swapping the complete BDPT demo into Studio would also
swap its expensive light-connection algorithm, which is unnecessary for the
camera optical experiment. The lens is the component to integrate.

The authoritative exact GPU traversal is `parametric_lens_teleport` in
`csrc/shaders/ray_refine.comp.glsl`, keyed by the `14949` PLENS payload from
`CompoundLens.build_gpu_payload()`. The native CPU Stage 2 handler uses the
same payload. Both reverse the face order for sensor-to-scene rays and select
an index table by active wavelength. The existing `lens_optics.cpp` batch
Jacobian helper traverses exact conics too, but its local `trace_payload` only
returns geometric exits; it is not a complete replacement for T2's optical
amplitude, Fresnel/event and OPL behavior. Transfer-profile truth must come
from the active native T2 path or a common extracted kernel with parity
against it, not from an unrelated thin-lens fit.

Keep the existing Studio pixel/time/film integrals and scene shader. Replace
only its camera-ray construction for an explicitly selected compound profile.
For each sensor pixel, aperture sample, wavelength and exposure slice, the
camera component must return:

- world-space exit origin and unit direction;
- transmitted or rejected status, transmission/throughput and sampling PDF;
- optical path length (and differential/phase-space data when wave or focus
  diagnostics require it);
- prescription hash, focus/zoom/stop state and coordinate transform.

The sample's lens-side ray starts at the physical film gate, samples the
**exit pupil**, and traverses the existing compound model backward through
the actual stop and parametric surfaces. The result enters the existing Studio
scene integrator as a world ray. This places compound optics inside each
sensor/aperture sample without invoking the CPU T5 all-pairs connection stage.
The current analytical thin-lens ray is the comparison lane and a first-order
proposal, not a substitute result for rejected or aberrated exact rays.

The current fixed:8 Studio shader carries eight film layers along one common
geometric ray. A dispersive compound lens cannot use that common ray: each
band has its own exact exit origin and direction. The fast transport choice is
to stratify wavelength at camera launch, trace one band-correct world path
per sample, and deposit into its own existing film layer with the wavelength
proposal-density factor. That factor is Monte Carlo sampling normalization,
not a shutter clock or a count of rays arriving as exposure. A quality mode
can evaluate all eight exits and scene paths per pixel/aperture sample at
greater cost. Both retain the Studio dt-slice exposure contract.

For a faster view, derive a sensor-side transfer product from the canonical
compound prescription. Key it by prescription, focus/zoom/stop, film pose and
spectral basis. Interpolate only within verified cells and retain pupil mask,
exit origin/direction, throughput and PDF. The existing front-only LUT needs
those fields and wavelength-aware backward evaluation before it can serve this
role. A cell with an invalid or high-error interpolation must use the exact
parametric evaluator or be declared unavailable; silently routing it through
the analytical thin-lens ray would change the optical model. Preserve the
exact T2 path for a selectable reference and for constructing/validating the
transfer product.

Profile construction should batch exact native T2 traversals over film
position, pupil coordinates and each wavelength. Store the exit ray, stop and
vignetting disposition, per-band throughput, OPL and local phase-space
derivatives. The existing native `estimate_lens_phase_space_jacobians` already
computes the 4×4 derivative with a spectral-lane parameter; it can guide
adaptive cell refinement. Subdivide near stop edges, discontinuities,
caustics and large held-out residuals. A fitted local transfer is admitted
only after held-out exact rays meet position, angle, throughput and energy
limits; the first tables are offline profiling artifacts, not a new optical
law. Lens state changes choose or rebuild a versioned table at the exposure
slice barrier, so a stale profile never enters a captured moment.

This is a proposed integration contract, not an installed renderer. GPU speed
is unmeasured. The current Studio GL shader and native T2 do not share a
per-sample buffer ABI, and CPU readback/upload may erase any speed gain. The
first implementation experiment should therefore measure camera-ray
generation separately from scene traversal and transfer, using one identical
scene, camera state, film response, seed and sample budget for analytical,
exact, and cached compound lanes. Report setup, camera transform, scene trace,
transfer, and total per slice. Include off-axis and stop-edge rays, full eight
bands, focus changes, invalid-cell counts, geometric and radiometric error.

## Acceptance for the next experiment

The source changes belong at these existing seams:

| Existing owner | Integration change |
|---|---|
| `CompoundLens.build_gpu_payload()` and `ray_refine.comp.glsl` | Remain the exact spectral conic authority; expose bounded sensor-side transfer observations from that traversal. |
| `camera_software` optical graph and lens assembly | Compile/profile a versioned reciprocal transfer keyed by actual lens and film state; reuse the native spectral Jacobian helper for adaptive refinement. |
| `demo_pluck_gl.SensorAccumulator` camera-ray launch | Choose analytical, exact compound, or verified transfer; carry exit origin, direction, band, throughput and PDF into its existing scene/NEE/field path. |
| `marble_lens_contract.py` | Admit compound profiles only after their full-field and state-domain checks, retaining authored thin LensSpec entries as analytical profiles. |

The physical housing already has a separate visibility control. A camera ray
emerging from the compound front must not encounter the same lens a second
time as ordinary scene geometry. Multi-camera lanes keep independent camera
state and accumulation but can share immutable profile data for identical
prescriptions and states.

1. Build the compound sensor-side transfer from the existing `CompoundLens`
   prescription; record source hash and all state. Do not infer missing YAML
   surfaces from the authored thin profiles.
2. Compare exact compound rays with the transfer output at held-out sensor,
   pupil and wavelength coordinates. Check pupil rejection, angular/position
   error, transmission, PDF and energy before rendering a scene.
3. Feed the validated rays to the current Studio sensor integrator, retaining
   its time slices, state barriers, materials and disjoint sensor lanes.
4. Time one matched full image against the current thin-lens camera. A target
   near its throughput is plausible only after camera transfer is GPU-resident
   or amortized; the present CPU table timing does not prove that target.

The published [pbrt realistic-camera implementation](https://pbr-book.org/3ed-2018/Camera_Models/Realistic_Cameras)
supports precomputed exit-pupil bounds for efficient film-side sampling. This
design uses that principle while keeping this repository's exact compound
prescription and parametric-region evaluator as the optical authority.
