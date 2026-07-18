# Spectral Rasterizer Handoff - 2026-07-15

## Objective

Build a physically faithful spectral rasterizer that can render any requested
part of a photographic scene for layout construction. Layout describes world
geometry, materials, lights, camera, focus, sensor region, and requested raster.
The ray tracer produces linear sensor evidence. Camera software interprets that
evidence into a display image.

The intended ownership chain is:

```text
scene order -> world/layout compiler -> spectral BDPT/VCM + thick lens
            -> linear sensor accumulation -> camera software -> display raster
```

Text is the first demanding layout primitive, not the final scope.

## Implemented Path

- `scene_orders.py` validates and resolves declarative scene jobs.
- Scene jobs support explicit world camera position, target, up vector, and a
  separate focus target.
- World geometry is transformed into the native camera-local X-axis frame; the
  retained physical lens/sensor/light rig is not rotated.
- Paragraphs use real font contours, metric wrapping, hard long-word wrapping,
  fit-to-box, centering, triangulated caps, and closed extrusion sidewalls.
- The live paragraph scene uses an oversized quiet backplate and retains only
  the photographic rig emitters from the base scene.
- Requested rectangular output receives a complete square native schedule and
  exact rectangular readback. Native sensor orientation is corrected at the
  Python boundary.
- The live pygame UI has one render owner, coalesces pending edits, never
  cancels an in-flight exposure, never enlarges traced pixels, and keeps its HUD
  outside the traced canvas.
- Native subprocess output is teed to both the terminal and revision log.
- Native `RayTracer.get_sensor_image_linear()` exposes unnormalized linear
  accumulation separately from the diagnostic percentile/log preview.
- Every memory-safe spatial work unit is executed before sign-off.
- `exposure.sensor_sweeps` repeats complete sensor schedules with independent
  progressive pupil samples. `--bdpt-native-packages` only partitions spatial
  work and is not a quality axis.
- Explicit `--bdpt-native-sweeps` overrides the authored count for diagnostics.
- Multi-sweep raw readback is averaged using a divisor passed explicitly from
  `ExposureSession` to `CppExposureBackend`; dtype is preserved.

## Spectral Fidelity

The active image formation path remains the native spectral tracer:

- Eight spectral bands in the current scene.
- Spectral material payloads.
- Dispersive thick-lens interfaces.
- Physical aperture/pupil sampling.
- Light and camera BDPT subpaths.
- Sampled T5 camera-light connections.
- Optional spectral VCM merging.
- Additive native sensor accumulation.

There is no raster text fallback, synthetic text compositing, post blur,
denoising, or sharpening in the transport path.

## Artifact Semantics

- `0000_cpp_linear.npy`: authoritative linear sensor RGB accumulation, averaged
  across complete sensor sweeps. Preserve this unchanged.
- `0000_cpp.png`: native diagnostic preview. It uses frame-relative percentile
  normalization and a logarithmic curve. Do not use it to judge exposure or
  calibrated color.
- `0000_cpp_camera.png`: new camera-software interpretation of the linear sensor
  array for the live text path.

The live viewer has been changed to load `0000_cpp_camera.png`, not the native
diagnostic preview.

## Camera Output Work

`camera_software.sensor_back` now owns a fixed-reference linear-sensor output
pipeline:

- Fixed `sensor_white_level`.
- Explicit exposure compensation in EV.
- Fixed RGB white-balance gains.
- Optional 3x3 color matrix.
- Fixed `linear` or `reinhard` tone curve.
- sRGB transfer encoding.
- Output clipping only when selected by the profile.
- No frame maximum or percentile participates in the transform.
- Floating-point input dtype is preserved and input arrays are not modified.

The old `_ColorScienceHook` per-frame peak normalization was removed. ADU input
is referenced to sensor black/white levels, electron input to full-well, and
raw linear RGB to the profile's fixed white level.

The live CLI now exposes:

```text
--camera-white-level VALUE
--camera-exposure-ev STOPS
--camera-white-balance R G B
--camera-tone-curve linear|reinhard
```

Direct camera tests pass. Existing live tests pass. A fresh expensive native
live render has not yet been completed after this final camera-output wiring;
that is the first takeover smoke test.

## Measured Refinement Result

Controlled 120x100 output, 120x120 native sensor, 32 aperture samples:

- One complete sweep: three spatial units.
- Four complete sweeps: twelve spatial cycles.
- Four-sweep run recorded 2,227,200 rays in 449.08 seconds.
- Corrected four-sweep mean luma stayed within 1.4% of one sweep.
- High-frequency luma variation fell to 0.487x.
- High-frequency chromatic variation fell to 0.654x.

The luma result closely matches the independent-sampling expectation
`1 / sqrt(4) = 0.5`. Progressive sensor sweeps therefore genuinely reduce
estimator variance. Remaining stable softness/color belongs to scene geometry,
material, illumination, optics, spectral response, or camera interpretation.

## Current Scene Facts

- Live text material albedo is warm beige `[0.82, 0.76, 0.62]`; a yellow cast is
  partly authored.
- Live paragraph profile is `straight`, not circular or bulged.
- Depth is 12 mm with 25% embedding, leaving 9 mm exposed; sidewalls can still
  carry colored illumination.
- The ordered physical lab lens is focused at the paragraph's actual
  approximately 3.73 m sensor-to-text conjugate.
- The compound lens has approximately 137.6 mm effective focal length and a
  20.62 mm iris, approximately f/6.7. Scene-order 35/25 metadata is not the
  final physical lens's EFL/iris ratio.

## Main Quality Limits

1. T5 sampled connections dominate runtime. Each spatial cycle evaluates about
   19.3 million sampled pairs from a roughly 79 billion pair full domain in the
   120x100 diagnostic.
2. Sensor sweeps are coarse full-image refinements; accumulation is not yet
   persistent or resumable by tile/sample range.
3. The native accumulator is square, so rectangular crops trace unnecessary
   rows or columns.
4. There is no per-pixel variance/convergence buffer. Legacy global SNR is not
   valid evidence for native non-radiometric sensor RGB.
5. The scene schema is still specialized: arbitrary imported meshes, textures,
   alpha cutouts, instancing, general lights, and CSG are not complete.
6. Camera output has explicit controls but no measured camera characterization
   profile yet. Identity white balance and matrix are defaults, not calibration.
7. A pre-existing Pylance diagnostic remains in `exposure_render_demo.py`:
   unresolved `OpticalAssembly` annotation. It is unrelated to this work.

## Any-Purpose Rasterizer Landscape

The next architecture should define sample identity as:

```text
(scene version, camera, film region, pixel, sample index, wavelength, time)
```

This is the prerequisite for deterministic resume, distributed execution,
reproducible A/B comparisons, and adaptive sampling without duplicate work.

Priority sequence:

1. Complete one fresh live native render and verify raw, diagnostic, and camera
   PNG artifacts plus terminal camera-profile reporting.
2. Add persistent tiled accumulation with explicit sample ranges and checkpoint
   metadata.
3. Make native scheduling rectangular and crop-aware while proving equivalence
   to cropping a complete sensor exposure.
4. Add per-pixel sample count, first moment, second moment, luminance variance,
   and chromatic variance buffers.
5. Add unbiased adaptive allocation based on those buffers.
6. Generalize scene orders to meshes, instances, textures, alpha geometry,
   curves/paths, arbitrary emitters, and reusable assets.
7. Establish neutral calibration scenes for exposure, spectral color, PSF by
   wavelength/field position, slanted-edge MTF, energy conservation, and
   crop/full-frame equivalence.
8. Profile and improve T5 light-vertex sampling only with estimator-bias and
   variance evidence.
9. Add distributed tile/sample workers after deterministic sample identity and
   checkpoint merging are established.

## Validation Commands

Focused Python suite:

```powershell
python -m pytest tests/test_camera_sensor_output.py tests/test_native_bdpt_work_plan.py tests/test_live_spectral_text_demo.py tests/test_scene_orders.py -q
```

Native Release build when C++ changes require it:

```powershell
cmake --build csrc_build --config Release
```

Live smoke at a small raster:

```powershell
python live_spectral_text_demo.py --display-width 120 --display-height 100 --sensor-sweeps 1
```

For a bounded diagnostic, invoke `exposure_render_demo.py` with
`--no-convergence-drive-batches`; the default live policy can scale aperture
samples to 192 and execute 14 spatial units at 120x100.

## Evidence and Repository Hygiene

Generated render directories under `exposures/` are intentionally excluded
from the handoff commit. They total tens of MiB and are reproducible. Important
paths from this session include:

- `exposures/text_focus_diagnostic/render_complete`
- `exposures/text_focus_diagnostic/render_refined_4x_true`
- `exposures/spatial_tile_render_v2`
- `exposures/live_text_acceptance`

Do not commit generated field captures, `.npy` evidence, render logs, or PNGs
unless a later task deliberately establishes a compact golden-test fixture.

## Reusable UI asset slice (2026-07-17)

`camera_software/render_assets.py` now defines content-addressed extruded-token
assets, rotating-stage light-field conditions, an atomic render catalog,
character-atlas fallback, and whole-page/constituent-part bake plans. The live
display scene uses this token asset contract when producing its existing scene
order. See `LIGHT_FIELD_ASSET_ARCHITECTURE.md` for ownership and the next
executor integration boundary.

The production default has since been narrowed to the accepted fixed head-on
red-ink-on-black-slate scene. Missing unique characters are queued before
exact token sequences, captures retain a padded neighboring-light region, and
rotation or camera/light action recording is optional rather than required.
The live render owner now follows a hard production ladder: fixed-width
upper/lowercase alphabet cells, word tokens, complete editor/fixed-UI token
strings, then the total UI scene. Later tiers cannot be scheduled or composed
before earlier tiers converge. Every prepared atlas process restores the
asset's accumulated native sensor sum and weight, then executes one much
larger epoch by default: up to 1,024 selected sensor nodes across 64 recursive
refinement submissions, versus the former 128 × 8. The configurable per-epoch
ray load amortizes procedural setup without changing the convergence unit.
Covered frames become immediately composable but remain scheduled
least-refined-first until successive image-delta checks converge.
