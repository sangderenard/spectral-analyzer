# Thick-lens spectral render acceptance

This document is the authoritative entry point for the native thick-lens image
path. Older BDPT handoff notes describe intermediate implementations and must
not be treated as current run instructions.

## Intended command

```powershell
python exposure_render_demo.py `
  --width 64 --height 64 --frames 1 `
  --backend cpp --integrator bdpt --scene-mode thick-lens-lab `
  --gpu-resident --rgb-source sensor --no-window `
  --no-convergence --no-convergence-drive-batches `
  --total-rays 800000 --rays-per-batch 200000 `
  --bdpt-native-packages 4 --save-files `
  --out-dir exposures/thick_lens_acceptance_64
```

Saving does not alter resolution or enable oversampling. Oversampling is used
only when explicitly requested with `--output-oversample`.

## Fidelity invariants

- Full compound-lens geometry remains enabled.
- Glass dispersion uses the configured measured Sellmeier curves.
- The requested physical aperture schedule is not silently reduced.
- Native camera and emitter paths sample wavelengths across the existing ray
  schedule. Each path uses that wavelength's material and refractive data.
- Fresnel reflection/transmission is sampled stochastically as one path event,
  avoiding exponential duplication at every lens surface.
- T5 completion is measured against the full camera-by-light work domain.
  A budget-limited subset must be reported as partial/deferred, never complete.
- RGB display conversion uses one common exposure scale so spectral channel
  ratios are preserved.
- A partial/deferred T5 pass and an empty sensor buffer are not successful
  renders.

## Required evidence

A passing run must report all of the following:

- the requested aperture sample count in the `workload` line;
- no vertex, spectral, PDF, optical, or connection record overflow;
- nonzero camera and light vertex populations;
- `PASS COMPLETE`, with zero deferred pairs;
- a finite sensor result with nonzero lit pixels;
- `usable image`, followed by emitted PNG, 16-bit PNG, NPY, and JSON files.

Image quality is not established merely by being non-black. Coverage, focus,
color balance, clipping, and stability across seeds must be inspected after the
transport conditions above pass.

## Latest verified result (2026-07-10)

The command above completed four cap-safe packages in 165.4 seconds. Every T5
pass reported `PASS COMPLETE`, with about 27.4 million sampled pairs per
package, no deferred work, no record overflow, and 77/4096 lit sensor pixels.
The clean output is `exposures/thick_lens_acceptance_64/0000_cpp.png`.

This is a successful finite spectral sensor output, but it is **not yet the
high-quality acceptance image**: visual inspection resolves the colored
emissive objects while diffuse/basic scene geometry remains black. The open
quality gate is therefore narrowed to T5 non-emissive endpoint/material
contribution, not camera sampling, aperture fidelity, output saving, overflow,
or incomplete scheduling.

## T5 rejection audit (2026-07-11)

`--profile` now enables counters in the real T5 shader. A 16x16, one-package
full-lens run measured 6,926,624 candidate pairs: 695,699 retained valid
non-delta endpoints and throughput, 655,883 passed MIS, 81,577 shared a sampled
spectral band, 1,843 had nonzero response at both endpoint materials, and 14
were unoccluded. This explains the sparse image without weakening the camera.

The rejection counters are diagnostic only. Endpoint eligibility remains
matched to the CPU reference; delta-strategy validity is decided by the
connection-PDF and MIS stages rather than by an extra GPU-only endpoint filter.
Profile scatter logging uses one queue fence and summary instead of one message
and forced fence per 65,536-record chunk.

The audit found that an experimental native path had silently reduced the
165,637-vertex light population to 16 sampled vertices and then reported that
reduced domain as `PASS COMPLETE`. The 16x16 profile consequently evaluated
only 6.9 million of roughly 71.6 billion camera-light vertex pairs. That false
completion path is removed: budgeted work is again accounted against the full
domain and unevaluated work remains partial/deferred.

## Verified computation failure map (2026-07-11)

The current trace has established the following sequence with measured
evidence rather than scene assumptions:

1. Camera and emitter launches create spectral subpaths and T3 records them.
2. Thick-lens traversal commonly consumes the remaining bounce budget exactly
   when a camera path reaches ordinary scene geometry. The old budget-terminal
   branch left that final vertex with `pdf_flags=0`, causing T5 to reject its
   endpoint BSDF. Final-bounce vertices now retain their authored diffuse/GGX
   endpoint lobes without spawning a fabricated child edge.
3. The score prepass previously multiplied independent all-band beta sums.
   It could therefore rank a camera/light block highest even when the two
   blocks shared no wavelength. A bounded trace spent 8,388,608 pair tests on
   such a block and produced exactly zero spectral candidates. Scoring now uses
   the per-band beta dot product required by T5.
4. With those corrections, the same 10,000,000-pair diagnostic budget produced
   about 65--70 thousand nonzero spectral connections, 37--41 thousand visible
   connections, and 178--186 positive raw sensor pixels out of 256.
5. This is still not an accepted image. The full domain is roughly 71 billion
   pairs, while the bounded scheduler admits one coarse 8192x1024 unit and
   defers more than 99.98 percent of the work. Its few high-weight samples
   dominate the tone-mapped result, which remains isolated points. The saved
   diagnostic is `exposures/bdpt_trace_20260711/0000_cpp.png` and is explicitly
   partial, not a quality result.

The weight trace subsequently isolated a concrete PDF-measure failure. Fresnel
reflection/refraction records are marked `BDPT_PDF_FLAG_DELTA_SPECULAR` and
carry a discrete probability mass, but both the GPU scatter pass and CPU mirror
converted that mass with `cos(theta)/distance^2` as though it were a continuous
solid-angle density. Those artificial inverse-square factors multiplied across
the closely spaced compound-lens surfaces and produced enormous MIS
denominators. The correction retains the forward/reverse Fresnel probability
mass for delta events and applies area conversion only to continuous lobes.

A profile-only pre/post sensor snapshot now separates terminal camera-path
splats from T5 connection energy. Before the delta-measure correction, one
8,388,608-pair unit added approximately `1.2e-23` RGB energy. After the
correction, independent 10M-budget runs added approximately `0.68--0.73`, with
179--184 of 256 pixels receiving positive T5 energy and no negative accumulator
changes: a recovery of roughly 23 orders of magnitude. A 20M budget admitted
two units and added `1.16`, with 187/256 T5-lit pixels. The exact stochastic
path population varies between runs, so this is evidence of work scaling, not
a deterministic convergence ratio.

The remaining exhaustive coordination cost was also real. A representative
single-package 16x16 run contained about 432k camera vertices and 166k light
vertices: 71.9 billion Cartesian pairs in 8,639 coarse units. At the measured
profile rate of about 0.83--0.97 million pairs/second, exhaustive evaluation is
roughly 20--24 GPU-hours.

Budgeted native passes now use an explicit stratified Monte Carlo estimate of
the complete light-vertex domain. Every camera vertex receives the same number
of randomized, stratified light-vertex samples; contributions are multiplied by
`n_light_vertices / n_light_samples`, the inverse marginal selection
probability. This is not the removed 16-vertex truncation: all light vertices
remain eligible, all camera batches are processed, and the log/status says
`SAMPLED-ESTIMATE` rather than `COMPLETE`. `t5_pair_budget=0` still selects the
exhaustive domain. Variance/convergence remains a quality question, but omitted
light vertices are no longer silently treated as zero Monte Carlo mass.

At 16x16 and a 10M budget, the sampled pass evaluated 23 stratified light
samples for each of 432,522 camera vertices (9,948,006 pairs), used inverse
probability 7,216.43, and added about 761.9 RGB energy across 190/256 pixels.
At 64x64 and 20M per package, four packages completed in 125.7 seconds, each
using 11 samples per camera vertex; the final native sensor report was
3,011/4,096 lit pixels.

The saved PNG path had a separate display-encoding failure: tone-mapped values
were linear display RGB but the custom 8/16-bit PNG writer wrote them directly
without an sRGB transfer function or metadata. Linear value 0.012 therefore
became byte value 3 instead of an sRGB value near 29, hiding the continuous
shape that was present in the saved array. Both PNG writers now encode linear
display RGB to sRGB and write an `sRGB` chunk; the `.npy` data remains linear.
The current completed diagnostic is
`exposures/bdpt_sampled_64_20m_srgb/0000_cpp.png`. It shows the continuous
shaded object and is a successful non-black image, but visible Monte Carlo grain
means it is not yet the final high-quality acceptance render.

An additional audit counted MIS alternative cuts: about 89.9% were adjacent to
a delta vertex in the sampled thick-lens paths. This is recorded as a diagnostic,
not removed: an alternative cut along an already sampled, delta-consistent path
edge can be a valid strategy. No blanket delta endpoint rejection is authorized
by this count.

## Specular transport extension

The native path implements spectral BDPT vertex connection: sampled delta
lens/specular events remain valid internal path edges, GGX endpoints have
continuous BSDF/PDF evaluation, and a delta endpoint correctly has zero PDF for
an arbitrary new connection direction. It now also contains a spectral
vertex-merging estimator. It does not contain a specular-manifold Newton solve;
VCM is the selected general-purpose extension.

A 16x16 sampled profile on 2026-07-14 measured 9,950,950 pair tests. Of those,
6,961,985 presented a delta camera endpoint and 916,174 a delta light endpoint;
7,355,301 pairs retained both spectral betas, but only 333,368 produced both
continuous connection PDFs. This is a measured sampling limitation, not a
reason to pretend delta surfaces are diffuse or to modify the lens mechanics.

The implemented native extension is spectral vertex connection and merging
(VCM):

- A GPU linked-cell hash indexes eligible light vertices while retaining their
  complete subpath/PDF/spectral records.
- Nearby light vertices merge into non-delta camera vertices using the uniform
  surface-area kernel. Light paths may contain any preceding delta chain, which
  captures caustic paths that ordinary vertex connection samples poorly.
- Keep exact-delta endpoints out of arbitrary connection/merge BSDF evaluation;
  they remain constraints within sampled paths.
- Evaluate material response per wavelength with the existing diffuse/GGX and
  Fresnel contracts. No Phong/raster approximation is permitted.
- The balance-heuristic denominator includes every valid VC cut and VM weld for
  the extended path. The VM density includes `pi*r^2`; in the renderer's
  deferred-normalization convention this supplies the normalized uniform-disk
  kernel without applying a second inverse-area factor.
- Use VCM MIS weights against the existing vertex-connection strategies so the
  extra pass does not double count energy.  As of 2026-07-14 this is enforced
  symmetrically: the connection shader's balance-heuristic denominator now
  also contains the weld-technique density `prefix[m] x suffix[m] x pi r^2`
  for every non-delta interior chain vertex whenever a merge pass runs in the
  same T5 pass (`T5GpuParams.vm_area`, 0 when VCM is off — restoring exact
  connection-only MIS and CPU/GPU parity).  The merge shader's cut-technique
  enumeration was aligned to the connection shader's documented delta-cut
  policy so both estimators integrate against the same technique set.
- A progressive merge-radius schedule and logs expose sample count, radius,
  merged candidates, accepted merges, MIS energy, and variance in logs.  The
  radius and hash grid are prepared BEFORE the connection dispatch of the same
  pass so both estimators share one radius.
- Accumulate merging in linear sensor space before the common exposure and sRGB
  encoding stages.

The native 32x32 validation on 2026-07-14 indexed 83,805 merge-eligible light
vertices and queried 1,521,286 non-delta camera receivers. The hash visited
6,285,828 candidates, found 124,027 compatible surface neighborhoods, produced
94 positive spectral merges, and splatted 50 camera receivers. The complete
run took 18.11 seconds for the native package (29.4 seconds including Python
startup, camera construction, scene/cache work, profiling, and file output).
The connection-only A/B run took 17.27 seconds for its package. This establishes
that VCM is active and locally indexed rather than another Cartesian pass.

The saved validation is `exposures/vcm_validate_32/0000_cpp.png`. It remains a
small, grainy convergence diagnostic, not the final high-quality acceptance
image. VCM improves access to specular-chain paths but does not manufacture
sample count or remove ordinary Monte Carlo variance. The current intended
command accepts `--vcm-radius-mm` (default 2), `--vcm-radius-alpha` (default
0.7), and `--no-vcm` for controlled A/B testing.

No camera or thick-lens mechanics were changed for VCM. Lens traversal,
per-band IOR, Fresnel probability masses, aperture clipping, camera launches,
material reflectance, diffuse/GGX response, and exact-delta endpoint rules use
the existing paths. In particular, VCM never converts a delta endpoint into a
finite-width glossy event: a preceding delta chain contributes only after its
light path lands on a compatible non-delta surface.

The intended 64x64, four-package command subsequently completed in 127.03
seconds and saved `exposures/vcm_accept_64_4pkg/0000_cpp.png`. All four passes
reported `SAMPLED-ESTIMATE`; the VCM radius progressed from 2.000 mm to 1.625
mm, and the final GPU sensor had 3,064/4,096 lit pixels. The saved image is a
continuous, spatially coherent shaded object with localized emitter/specular
highlights rather than a fabricated array of point lights. Grain remains
visible, so this is a successful renderer/output acceptance image, not a claim
of noise convergence.

Reproduction command:

```powershell
python exposure_render_demo.py --integrator bdpt --scene-mode thick-lens-lab `
  --width 64 --height 64 --total-rays 10000 --rays-per-batch 10000 --frames 1 `
  --no-window --gpu-resident --save-files `
  --out-dir exposures/vcm_accept_64_4pkg --bdpt-native-packages 4 `
  --t5-pair-budget 20000000
```

Estimator references: Georgiev et al., *Light Transport Simulation with Vertex
Connection and Merging* (2012),
https://cgg.mff.cuni.cz/~jaroslav/papers/2012-vcm/2012-vcm-paper.pdf; and
Georgiev, *Implementing Vertex Connection and Merging* (revision 1, 2012),
https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev1.pdf.

## CPU/GPU BDPT parity audit (2026-07-14)

The transport and connection work is not confined to one backend. T1-T4 have
CPU and GPU implementations, and T5 vertex connection has a forced-CPU audit
path plus the normal GPU shader. VCM merging remains GPU-only and is therefore
not covered by the parity fixture below.

The audit found and corrected several real divergences in CPU T5: cross-band
scalar beta multiplication, missing endpoint BSDFs, opaque treatment of glass,
diffuse/GGX PDF selection instead of summation, and a nearest-pixel rather than
normalized 2x2 tent splat. The CPU path now follows the GPU's same-band spectral
product, authored diffuse/GGX endpoint response, per-band Fresnel transmission,
balance density, and sensor reconstruction.

It also found two backend-independent failures. A fast ray batch could finish
before the submission markers were signaled, permanently losing the T5 pass;
the completion and submission transitions now share an idempotent scheduler,
with an authoritative idle reconciliation for GPU-resident bounce generations.
More importantly, the camera measurement endpoint was evaluated as the black
sensor mesh's BRDF, erasing every connection. `BDPT_PDF_FLAG_SENSOR` now marks
that endpoint in both host and native GPU packing, and both material evaluators
treat it as a unit measurement response whose directional support is already in
the camera PDF/MIS chain. This does not alter lens traversal or camera launch
mechanics.

Regression command:

```powershell
python tests/t5_cpu_gpu_parity.py
```

The fixture clears terminal/direct accumulation before T5, so a zero or copied
pre-connection image cannot pass. It runs a diffuse scene and a second scene
with a transmissive boundary and wavelength-dependent IOR. The combined run
completed in 30.5 seconds. Diffuse GPU/CPU relative L2 was
`3.8714355e-08` (`63.99999630` vs `63.99999613` image sum); glass relative L2
was `1.5588332e-07` (`58.61502665` vs `58.61501974`). Both had positive isolated
T5 output and zero vertex, spectral, PDF, optical, and connection overflow.

## Public camera diagnostics

Both `exposure_render_demo.py` and `thick_lens_focus_lab.py` now print the exact
loaded `_spectral_kernels` path and require the `sensor-endpoint-v1` ABI. They
fail loudly rather than silently using a stale native module.

The supported headless wrapper is:

```powershell
.\run_camera_diagnostic.ps1 -Preset quick
.\run_camera_diagnostic.ps1 -Preset evidence
# Larger square sensor; native planner automatically adds cap-safe work units.
.\run_camera_diagnostic.ps1 -Preset evidence -Resolution 256
```

`quick` is a 16x16 one-package health render. On 2026-07-14, after the spatial
work scheduler correction, it completed in 25.9 seconds, produced 186/256 lit
transport pixels, and wrote native sensor evidence
and the completed T5 latch to JSON. `evidence` is the 64x64 four-package render;
the corrected spatial run completed in 151.7 seconds, produced 3,009/4,096 lit
saved pixels, and saved finite RGB channel sums of 94.950, 79.806, and 48.982.
Its image is `exposures/camera_diagnostic_evidence_64/0000_cpp.png`.

Native GPU-resident sensor values are currently normalized display RGB, not a
radiometrically calibrated joule accumulator. Reports therefore use
`measurement_status=native_sensor_nonradiometric` and do not mislabel the
unused legacy accumulator's zero as a failed image.

## Large-image spatial work contract

The GPU-resident path accumulates every work unit into one persistent global
sensor image. A work unit is not a separately normalized output tile: it retains
the full-resolution film coordinates, shutter weighting, aperture locations,
spectral band selection, thick-lens traversal, and sensor splat. Only the set of
camera paths resident for one T5 pass is bounded.

The sensor sweep scheduler was corrected on 2026-07-14:

- Its order is immutable for a given image resolution. Previously the order
  depended on the current submission length, so the shorter final submission
  could change the order under an existing offset and duplicate/omit samples.
- Pixels are ordered in fixed 32x32 spatial Morton work tiles and all aperture
  samples for a pixel remain adjacent. Python work-unit boundaries are restricted
  to whole pixels.
- Scheduler storage is now one record per pixel. It no longer materializes and
  sorts one record per pixel *per aperture sample*. At 1024x1024 with 192 aperture
  samples this removes a roughly 201-million-entry startup schedule; the global
  float RGB sensor itself is 12 MiB.
- The work planner caps a unit at 200,000 primary camera rays. It prints image
  pixels, aperture samples, total schedule rays, work-unit count, pixels per unit,
  primary rays per unit, and persistent sensor memory before tracing.

`tests/test_native_bdpt_work_plan.py` verifies exact contiguous coverage of a
1024x1024x192 schedule, whole-pixel boundaries, the primary-ray cap, and output
memory accounting. A forced three-unit 16x16 run then exercised the real native
pipeline with a deliberately short last unit: 86 + 86 + 84 pixels, three T5
passes, one final image, and 184/256 lit pixels. Its output is
`exposures/camera_tiled_16_3unit/0000_cpp.png`.

This makes larger images memory-safe with respect to the sensor schedule and
per-pass BDPT record caps; it does not make them cheap. Runtime still scales with
pixel count, 192 pupil samples, camera-path depth, and sampled T5/VCM work. Each
work unit currently launches a fresh statistically independent flash field and
rebuilds its light sort/VCM hash. That costs time but does not multiply persistent
memory, and avoids silently reusing one correlated light realization. Persistent
light-field reuse is a future optimization requiring explicit estimator and MIS
validation; it must not be introduced as a camera shortcut.

## Sensor orientation and authored-colour fidelity (2026-07-14)

Two output-boundary corrections landed after the first glyph renders; neither
touches transport, lens mechanics, sampling, or MIS:

- The native sensor accumulator is Y-major: element `[iy, iz]` maps `iy` along
  world +Y (camera right) and `iz` along world +Z (camera up), so a direct
  readback displays the scene rotated 90 degrees.  `exposure_render_demo`'s
  `_native_sensor_to_display` now reorients readbacks to display convention
  (row 0 = scene up, columns toward camera right), including the thick-lens
  image inversion.  The `glyph_A` order now renders upright.
- `scene_orders` baked authored linear RGB through fixed Gaussian primaries
  centred at 700/540/455 nm.  The display transfer edge-attenuates 700 nm to
  0.3 and carries green weight at the red basis tail, so authored reds rendered
  dim and yellow-shifted.  Band curves are now solved by non-negative least
  squares against the exact `band_to_display_rgb` weights with a mild
  smoothness prior: authored red round-trips to display `1 : 0.036 : 0.02`,
  and neutral/white round-trips exactly (the flat-spectrum warm cast of the
  display transfer is compensated in the authored curve, not by re-balancing
  the renderer).  Remaining warmth on the glyph in
  `exposures/scene_order_renders_v5/glyph_A/0000_cpp.png` tracks the authored
  warm flash/orb illuminant (measured illuminant G/R 0.40 against glyph-region
  G/R 0.32) and is physical, not a display fault.

## Log vocabulary

- `workload`: requested sensor/aperture schedule and primary rays.
- `records`: transport records retained for T5.
- `scheduler`: pairs selected and deferred under the current budget.
- `PASS COMPLETE`: all work for the pass was evaluated.
- `PASS PARTIAL-DEFERRED`: not a finished exposure.
- `sensor result`: finite/lit-pixel statistics after native readback.
- `FAILED`: no usable sensor output; black or previous-frame display is only a
  diagnostic fallback.

## Historical notes

Declarative subject/batch authoring is documented separately in
`SCENE_ORDER_PIPELINE.md`. Its compiler retains this document's validated
camera/lens/flash transport and replaces only explicitly replaceable subject
geometry.

`BDPT_GPU_T5_WORK_REPORT_2026_06_07.md`,
`NEXT_AGENT_BDPT_T5_BIG_THREE_ACTION_PLAN.md`, and `WORKPLAN_and_AUDIT.md`
are historical design/audit records. They contain useful rationale but refer to
line numbers, shader behavior, and limitations that have since changed.
