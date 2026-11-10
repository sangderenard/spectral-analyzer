# Native emissive geometry in Studio

## Implemented component

Studio no longer reduces a luminous panel to its centre or constructs a
separate lamp description. `prepare_photo_geometry` builds the authored
geometry and shared MaterialDatabase MatBuf, then
`studio_native_emitters.native_emissive_sources` calls the original
`RayTracer.submit_emissive_triangles` sampler. Native root records supply
surface positions, cosine-hemisphere directions, wavelengths and spectral
weights. The fast Studio shader remains the renderer.

The existing native sampler groups triangles by UV/material domain. Its
default UV areas are equal even when world areas differ. This adapter uses
the existing API once per triangle, with complete spectral cycles, to avoid
that domain-allocation defect without introducing another sampler. Its
budget is a target, not an exact root count. Actual counts are recorded.
The original grouped native API still needs a separate general repair.

`bdpt_records.py` shares the native record layouts with the thick-lens lab
and spectral regression instead of loading the lab UI for sampling.

## Weighting and image estimator

The original forward SourceRec12 keeps native beta in packet.z. Its unused
direction.w carries the authored scene triangle identity. Camera records
are a separate buffer derived from those same roots. They carry the real
face normal, the same MatBuf emission spectrum, and world area divided by
the number of retained geometry samples on the triangle. The camera
selects uniformly from these roots and applies the reciprocal selection
probability once, both positive cosines, inverse squared distance and the
Lambertian 1/pi factor. Directions sampled for forward transport are not
mistaken for emitter surface normals.

Emission is one-sided. Native MatBuf emission is evaluated before tagged
glass-boundary continuation, so luminous transmitting surfaces remain
visible. Wavelengths are matched by actual frequency; native and camera
three-band array order differ.

The camera estimator retains this direct-emitter partition:
- Direct and specular/delta camera emitter hits contribute emission.
- Area next-event estimation owns the next emitter after a diffuse event.
  That corresponding BSDF-sampled emitter hit is suppressed.

Correction at 2026-11-10T01:11 UTC: both existing voxel-field image reads
(surface energy and volume marching) were restored at the user's direction.
They were wrongly disabled as part of the source integration. Native area
sources and the corrected binary remain; no centroid source was restored.
The restored field remains an additive existing estimator. Its normalization,
overlap with camera estimates, fixed packet cutoff and integer splats are
not newly validated by restoring the reads. Do not describe the combined
image as an unbiased or energy-calibrated estimator.

This is component integration, not a claim that the full native T5 BDPT
algorithm or a validated photon-merging estimator now runs inside Studio.

## Verification so far

- 54 focused CPU tests passed with the corrected native binary: real source
  roots, unequal areas, zero emission bands, sample-count energy invariance,
  camera geometry, dynamic scene compatibility, spectral handling and
  existing bounce/NEE contracts.
- The actual GPU area NEE function matched numerical integration on the
  RTX 3060. Increasing source samples 120 to 1,200 did not increase light
  output. Doubling source distance gave approximately one-quarter output.
  Back-facing source output was zero. Sensor, region sensor and forward
  shader variants linked.
- A saved 95,217-triangle Studio scene initialized 4,134 native roots and
  completed a 32x32 worker smoke in 2.72 seconds, including 2.17 seconds
  preparation. It reached two samples per pixel. Its sparse glass image
  is operational evidence only, not a quality/convergence benchmark.
- Exact underlight geometry: 5,004 distinct native panel origins, all launching
  upward; 1,211 rays escaped the grid/housing and 3,793 were blocked. The
  old centroid baseline escaped 0 of 5,000. This tests the underlight
  fixture only, not table/marble/whole-room occlusion or image convergence.
  Receipt: `repo-cache/underlight-check-20261110/native-results.json`.
- Independent review verified 4,098 positive source roots stayed identical
  when only a nonemissive subject or camera/housing moved.

Receipts: `build/dot-native-area-20261110/` and
`tests/studio_native_emitter_gpu_check.py`.

## Native build and release gate

Production promotion completed at 2026-11-10T00:48:26Z without a server restart.
The loaded production module passed all 54 focused CPU tests. Its SHA-256 is
`5fe18913f4892c94b2967ae3778f75bf2775766208f2bbabe45533625890796b`;
the verified old binary is retained under the receipt's rollback directory.

The old binary's duplicate emission sequence increment biases sparse
spectra as well as skipping lanes for even band counts. The existing
one-line source correction was built and tested separately. Production
binary promotion is a distinct release step; see the release receipt.
The adapter checks every triangle's per-band integrated beta and fails
explicitly if the loaded native generator violates its energy contract.
Zero-source scenes dispatch zero forward work and remain dark.

## Limits

Native surface positions are fixed deterministic quadrature samples, not
fresh independent Monte Carlo samples every frame. Binary shadow queries
do not solve refracted two-point connections through glass. The camera
still shares one geometric direction among its fixed spectral bands and
retains existing throughput/bounce cutoffs. Rare glass/caustic paths can
need many more samples than diffuse surfaces. MatBuf emission remains the
repository's authored spectral model, not a certified absolute radiometric
calibration. A converged full-scene image and complete transport parity
remain separate acceptance tests.
