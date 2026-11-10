# Automatically derived lens contracts

The user's requested architecture is preserved: Studio remains a shader built
from selected Spectral components. This change adds a cold contract exporter;
it does not replace Studio with the full native renderer, modify the original
lenses, change focus guards, restart a worker, or install a UI selection.

## Source collections

`configs/lenses/*.yaml` contains six authored analytical profiles. Each has
focal range, minimum focus distance, maximum aperture, radial-distortion
coefficients, vignetting and four transmission values. These are not glass
surface prescriptions. The macro profile's authored minimum is 0.23 m. The
YAMLs do not specify the focus datum, transmission wavelengths, distortion
coordinate convention, maximum focus distance or smallest aperture.

`thick_lens_focus_lab._LENS_PRESETS_6x6` contains eleven requests to the existing
four-group surrogate solver. Their labels are not guaranteed measurements of
the generated glass. The exporter runs the existing solver and exact tracer;
it does not synthesize a new optical design from the label.

`camera_designer.camera_preset.PRESET_REGISTRY` contains two complete surface
presets, the doublet and schematic eye. They are exported using the existing
SceneBuilder `CompoundLens.from_preset(axial_scale=-1)` conversion, with their
entire serialized prescription, source hash and 587 nm reference. Neither is
silently converted from its nominal focal-length label. The full inventory is
19 contracts across these three collections.

## Implementation

- `camera_software/lens_contracts.py`: `profile_contract`,
  `profile_shader_parameters`, `compound_contract`, `camera_preset_contract`,
  `export_collection`.
- `tests/test_lens_contracts.py`: source retention, hashes, strict units and
  controls, invalid profiles, duplicates, independent exact-ray checks,
  separated principal planes, and non-mutating focus-actuator checks.
- Run `python -m camera_software.lens_contracts --include-bench-presets
  --include-camera-presets
  --output exposures/dot_lens_contracts_20261109/lens_contracts.json`.

The collection is deterministic for the source inputs. YAML source-byte hashes
and resolved compound-payload hashes identify stale contracts. Every authored
YAML field is retained. Invalid or duplicate profiles produce explicit errors.

## Three physical controls

- Focus distance is in metres. YAML minimums remain authored, not measured;
  their reference datum remains explicitly unknown. Compound actuator probes
  use the first refracting vertex, not the sensor or front of the housing.
- Zoom is focal length in metres. A prime has equal minimum/maximum and no
  adjustable zoom. A solved zoom contract currently represents one sampled
  zoom setting, not a certified whole zoom interval.
- Aperture is f-number. The approximate shader input is entrance-pupil radius
  `f / (2 N)`. Authored maximum opening is preserved. Unspecified minimum
  opening is not invented. Compound pupil estimates retain their existing
  paraxial provenance and are not full-field pupil calibration.

`profile_shader_parameters` explicitly returns an analytical rectilinear
approximation with a lens-plane focus convention. It does not silently claim
the source's unspecified focus datum or turn a fisheye label into a calibrated
180-degree mapping. Distortion, vignetting and four-band transmission are
retained but not applied until their conventions are resolved.

## Measured versus requested properties

Compound contracts include the existing reduced-angle ABCD matrix, both
principal planes, effective focal lengths, back focal distance and pupil
estimates. Nine independent exact conic ray probes check near-axis agreement.
The probe thresholds are 10 nm height and 1 microradian reduced-angle error.
Scalar prescriptions do not invent a reference wavelength.

Preset acceptance separately checks the generated EFL against the solver's
0.5 mm tolerance and the actual sensor-focus residual against 5 micrometres.
Passing the paraxial equivalence check alone cannot approve a mislabeled lens.
At the current default 56 x 56 mm sensor and 1 m design setting, the accepted
named presets are 80/1.4, 55-110 at its midpoint, and 120Macro. Their derived
EFLs are approximately 79.828, 82.297 and 120.109 mm respectively.

Examples of rejected results in this configuration: the nominal 65Macro
produces about 139.004 mm EFL; the nominal 4.5 mm microscope request produces
about 1785.758 mm. The existing solver reports an unavailable group-spacing
envelope. The contracts preserve the actual result and reject the nominal
claim instead of substituting the label.

The two registry conversions are also not approved as positive air-camera
equivalents. The current doublet conversion derives -20.663 mm EFL despite
its 50 mm label; its small-ray/ABCD comparison passes, so this is not repaired
by changing the label in a contract. The eye derives -20.091 mm image-side
EFL, has image medium index 1.336 and a curved retina, and fails the current
near-axis comparison. These are conversion/prescription investigation leads,
not evidence that their intended physical designs are inherently negative.
Their existing source and transport are unchanged.

## Focus validation and close focus

The exporter copies the existing rear-group travel bounds, searches a bracket
for each sampled focus request, then verifies sensor residual and four actual
near-axis rays. An end-stop fallback or a conjugate-equation pole is not a
successful focus. The live scene and original prescription are not mutated.

The sampled distances are 5, 10, 20, 50, 100, 300, 1000, 5000 and 10000 mm.
The three accepted presets pass the 300, 1000, 5000 and 10000 mm samples in
this configuration. None of these generated setups passes the 5-20 mm samples.
This is not a universal simulator limitation. It is also not a continuous
focus-range certification: intervals between samples, full pupil aberration,
field curvature, distortion, color and zoom/focus coupling still need testing.

The existing Studio 20 mm UI/API restriction and 10 mm analytical CPU/GLSL
clamps are separate from these physical lens limits. They are unchanged by
this contract export. Removing them is not proof of a physical macro lens.

## Integration boundary

The next adapter should consume explicit lens contracts and sensor dimensions
in the current composed Studio shader. Use the existing exposure scheduler
and keep source hashes beside captures. Retain existing modes while exposing
the three controls. Reject unavailable settings with their measured reason;
do not hide failures behind a generic success flag. Approximate profiles and
accepted compound-derived states must remain visibly distinguishable.

Current generated evidence is in `exposures/dot_lens_contracts_20261109/`.
Fifteen focused contract tests pass. The final combined regression passes all
40 tests in 16.47 seconds, including existing optical-design and compound-transfer tests.
This is a contract-generation and calibration foundation, not a deployed UI or
a claim of optical equivalence over the full image field.
