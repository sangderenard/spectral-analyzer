# Next Agent Action Plan: BDPT/T5 Big Three

This handoff starts after commit `c5e0a05` and the follow-up uncommitted work
that made T5 evaluate per-band material response instead of producing a scalar
grey image.  The current local changes add:

- `csrc/shaders/ray_material_eval.glsl.inc`
- material-band binding for T5 at SSBO binding 8
- light-side `mat_id` packing for T5 vertices
- per-band T5 endpoint response using camera and light spectra
- pre-scatter BDPT spectral records from T3 so T5 evaluates the actual
  connection direction

The goal is still the same: T5 must be useful as real BDPT/MIS, not a fallback
diffuse compositor.  The three non-negotiable improvement areas are below.

## 1. Shared Material Evaluation

### Current State

T5 now uses `ray_material_eval.glsl.inc` to evaluate endpoint response by band.
This currently covers:

- diffuse reflectance response
- GGX BRDF response
- emission vertices as already-carried radiance
- material band lookup through `mat_bands`

This is a meaningful repair but not a complete unification.  T3 still owns most
sampling logic directly in `ray_material.comp.glsl`, while T5 owns a new eval
include injected by the shader loader.

### Required Work

1. Move common material accessors into shared includes.
   - Extract `mat_band_field`, reflectance, transmittance, emission, diffusion,
     IOR, GGX alpha, and GGX helpers from `ray_material.comp.glsl`.
   - Make both T3 and T5 include the same accessors.
   - Avoid two subtly different definitions of material truth.

2. Split sampling from evaluation.
   - T3 needs sampling helpers: choose diffuse/GGX/delta/refraction event and
     emit sampled outgoing ray plus PDF.
   - T5 needs evaluation helpers: evaluate response for a specific incoming and
     outgoing direction.
   - The two must share the same math and flags.

3. Add transmissive/Fresnel evaluation.
   - Current T5 endpoint eval covers diffuse and GGX but not a full
     transmission/refraction response.
   - Add per-band Fresnel/transmittance evaluation for valid non-delta
     transmissive events.
   - Keep true delta reflection/refraction out of arbitrary pair connections.

4. Add validation counters.
   - Count evaluated endpoint responses by material id and PDF flag.
   - Count zero response by reason: no material, no reflectance, wrong
     hemisphere, diffuse/GGX disabled, delta arbitrary connection.
   - Expose totals in profile logs.

### Acceptance Criteria

- T5 and T3 compile from shared material-access/eval helpers.
- Colored diffuse objects render with distinct colors in T5.
- GGX materials show directional highlight structure.
- Emissive materials contribute colored radiance instead of grey scalar power.
- Debug logs prove nonzero endpoint responses across multiple material ids and
  bands.

## 2. Fully Packed T5 Vertex Data

### Current State

T5 packed rows now carry enough data for the new endpoint evaluation:

- position, normal, incoming direction
- material id
- per-band beta magnitudes
- PDF flags
- optical block and Jacobian
- area-measure edge PDFs

The most fragile part is still the spectral side-record scatter.  Pack initializes
a grey fallback from scalar throughput, and `bdpt_scatter_t5.comp.glsl` is
expected to overwrite it with real per-band spectral records.

### Required Work

1. Prove spectral scatter coverage.
   - Add GPU-side counters for:
     - T5 rows with fallback-only beta
     - T5 rows overwritten by spectral side records
     - nonzero band count per row
     - material id distribution for camera and light rows
   - Log the counters when `--profile` is enabled.

2. Remove or gate grey fallback.
   - Once scatter coverage is proven, replace uniform fallback with zero or
     make fallback visible only under an explicit debug mode.
   - A silent grey fallback hides broken spectral data.

3. Confirm all T5 paths agree on layout.
   - GPU-native pack path.
   - CPU-prepacked GPU T5 job path.
   - Any tile-local T5 path.
   - `ray_pipeline.h` comments must match actual offsets.

4. Pack additional fields needed for transmission/delta-chain work.
   - medium before/after if required
   - front/back surface orientation if required
   - sampled outgoing direction for deterministic delta-chain matching
   - event classification beyond the current PDF flags if necessary

5. Preserve optical side records.
   - Verify optical blocking and Jacobians survive for both sensor and light
     families.
   - Add counters for optical-blocked vertices and non-unit Jacobians.

### Acceptance Criteria

- Profile output can state how many T5 rows have real spectral side data.
- No material color bug can be hidden by scalar fallback.
- Layout comments and shader offsets match.
- Camera and light vertex rows both report material ids, spectral bands, PDFs,
  and optical state.

## 3. Delta-Chain Handling

### Current State

Delta/specular vertices are accepted as valid path-chain records.  Arbitrary
T5 pair connection through a delta lobe still returns zero PDF/response.  That
is intentional and must remain true unless the connection direction exactly
matches the deterministic sampled event.

Direct terminal emissive splat handles sensor paths that actually hit emissive
surfaces, including after propagated specular/refraction chains.

### Required Work

1. Implement deterministic delta-chain connection.
   - Starting from a candidate endpoint, follow already-sampled delta edges in
     the packed chain.
   - Only connect when the deterministic chain reaches a non-delta endpoint,
     emitter, or the target direction matches within a strict angular tolerance.
   - Do not invent finite PDFs for arbitrary delta directions.

2. Add delta diagnostics.
   - Count packed delta vertices.
   - Count accepted propagated delta-chain terminal hits.
   - Count rejected arbitrary delta-pair attempts.
   - Count deterministic-chain attempts rejected by direction mismatch,
     optical block, missing next vertex, or visibility.

3. Handle mirror/glass/prism cases deliberately.
   - Mirror: deterministic reflection chain should carry colored/specular
     radiance when it reaches an emitter or connectable non-delta endpoint.
   - Glass/liquids: refraction chain must carry medium/Fresnel state and
     spectral transmittance.
   - Prism separation: per-band IOR/transmission must remain band-resolved
     through the chain.

4. Test visible emitters through delta chains.
   - Direct sensor terminal splat should light visible emissive geometry through
     glass/mirrors when the propagated path hits it.
   - Deterministic chain connection should handle cases where the terminal is
     not directly splatted but the sampled chain reaches a connectable target.

### Acceptance Criteria

- Arbitrary delta pair connections remain rejected.
- Propagated mirror/glass paths are not discarded by T5 validity filters.
- Emissive objects seen through glass/mirrors are visible and colored.
- At least one controlled mirror/glass/prism scene produces a logged nonzero
  delta-chain contribution path.

## Immediate Debug Sequence

Run the current renderer first and classify the image:

1. If colors appear but specular remains absent:
   - focus on GGX response math, roughness/alpha packing, and PDF flags.

2. If everything is still grey:
   - inspect spectral scatter counters first.
   - check that T5 rows are not still using fallback scalar beta.
   - check `mat_id` values in both light and camera rows.

3. If emissives are grey or too dim:
   - inspect light-side per-band beta records.
   - verify emission spectra are present before T5 connection.
   - verify terminal splat and T5 do not double-multiply or fail to multiply
     emission.

4. If image goes black:
   - check endpoint response zero counters.
   - confirm `t5_n_mats` uniform and material SSBO binding 8 are set before all
     T5 dispatch paths.
   - confirm T3 pre-scatter spectral records are reaching scatter-T5.

## Suggested Next Patch Order

1. Add T5 debug counters for material id, spectral overwrite, and endpoint
   response reasons.
2. Run one frame with `--profile` and record the counters.
3. Fix any missing spectral scatter/material id payload issue.
4. Move common material accessors into the shared include and make T3 use it.
5. Add transmission/Fresnel eval.
6. Add deterministic delta-chain connection.
7. Remove or explicitly gate grey fallback in pack.

The key rule for the next agent: do not add visual fallback composites.  Every
visible improvement must come from BDPT records, T5 material evaluation,
terminal sensor emission, or deterministic delta-chain transport.
