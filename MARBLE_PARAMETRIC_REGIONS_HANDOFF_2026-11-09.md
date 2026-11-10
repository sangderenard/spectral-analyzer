# Marble Studio: original parametric-region contracts and photo integration

## Implemented and validated local source

The current source implements exact saved spherical optical regions in both
existing Studio GPU directions. It directly reuses the original
`SphericalSurface.glsl_intercept_fn` closed-form kernel, not the native
triangle-local `SDF_SPHERE` approximation. No renderer replacement, service
restart, deployment, or interruption of a user exposure was performed.
General native triangle-group and spline kinds remain separate unsupported
contracts. The historical source audit below explains the distinction.

### Implementation contract

* Turing `marble_object.js` saves explicit version-1 `design.analytic_spheres`
  descriptors for eligible new saves, with original triangle membership,
  center, and radius. The saved source digest covers these descriptors.
  Old saves stay valid in mesh mode; exact mode asks to save them again.
* `marble_spheres.py` obtains descriptors only from the authenticated saved
  design after the unchanged MRP1 geometry hash proof. It checks the full
  v1 outer shell, uniform material, radial agreement, outward winding, and
  declared bubble/medium membership. Region identity is never inferred
  from shared material IDs; unsupported coatings are rejected.
* Initial support: uniform nonemissive uncoated spherical marbles without
  painted bands, and their declared air bubbles. Opaque spheres are allowed.
  Dice, emissive spherical area sources, painted/coated shells, arbitrary
  splines, and native triangle-local refinement are not presented as exact.
* `studio_photo_geometry.py` prepares the same CPU scene for admission and
  rendering. `PhotoJobs._validate_geometry` runs it with an 8-second timeout
  before replacing the active GPU job. Measured local fixture preflight was
  about 2.6 seconds. Final scene counts include base scene and fixtures;
  over-budget requests reject rather than silently decimating geometry.
* `append_saved_studio_mesh_to_gpu_scene(return_index_map=True)` preserves
  original membership through filtering. `studio_sphere_regions.py` removes
  only declared spherical faces, retaining one degenerate owner record
  per region for its existing material and positive/negative media. SSBO 12
  stores center/radius/owner, leaving Auto diagnostics on binding 11.
* Original closed-form roots compete against all remaining triangles in
  `nearest_filtered`, including visibility and origin-medium queries.
  Every spherical hit uses its analytic outward normal. Both directions
  return to traversal after each interface: no whole-marble teleport and
  no skipped colored interior geometry.
* `sphere_coverage=mesh|exact` and integer `triangle_budget=1000..1000000`
  are persisted. Legacy omission means mesh and 250000. Status/effective
  config report input/effective triangle counts and analytic sphere count
  in `geometry_stats`. Final RGB HDR includes `geometry_stats_json`.
* This changes optical coverage only. Saved mechanical geometry, volume
  integrals, mass receipts, and material/film physics stay unchanged.

### Actual validation receipt

Final focused CPU/source suites: 101 passed and 2 explicitly skipped GPU
cases across sphere regions, saved scene bridge, camera transport, spectral
basis, Auto probe, media, and film/worker integration. The skipped GPU cases
were run directly in a real context as recorded below. Turing sphere-admission,
geometry-policy/lifecycle, and route suites: 55 passed. The UI worker also
ran 71 focused lifecycle/boundary tests. No broad build or test discovery.

Both complete production GLSL programs passed offline glslang compilation.
Actual RTX 3060 forward and sensor probes were bit-identical for outside,
inside-glass, inside-bubble, tangent, miss, and coarse-mesh-silhouette rays;
four nested medium crossings; TIR retaining glass; analytic radial normals;
and a nearer ordinary interior triangle. Tests ran at radii 1 m and 0.01 m.
Opt-in regression: `tests/test_studio_sphere_regions_gpu.py` (direct function
invocation bypasses this repository's pytest GL stubs).

Actual 128 px / 16-epoch photographs completed for exact fixed:3+Auto, exact
fixed:1, and saved mesh fixed:3. The authenticated nine-sphere fixture had
53010 input triangles, 24466 effective triangles plus 9 analytic spheres;
mesh retained 53010. RGB HDR values were finite. Auto dispatched four 64-ray
batches at 1000 depth and correctly reported sparse coverage, retaining its
200 render cap instead of claiming convergence. Existing bubble GPU
interface and BVH suites passed again unchanged.

Local receipts, fixture, images, and bounded test scripts are in
`repo-cache/exact-spheres-20261109/`. These are verification artifacts, not
published user exposures. UI enablement/publication is coordinated by the
parent task; source integration alone does not claim deployment.

## Historical initial audit (before the implementation above)

Everything below records the original source audit and planned acceptance
criteria. Its missing-contract statements describe the earlier state, not
the implemented exact-sphere path above.

### Initial result

Inspected the current working sources on 2026-11-09, rather than relying on
the older inventory or the 2026-11-07 handoff. The Studio photograph still
uses the forward and sensor GLSL programs in `demo_pluck_gl.py`. The original
native tracer's parametric machinery exists, but these programs do not
consume its triangle-group contract. There is also a separate scale-context
contract; it must not be mistaken for analytic object intersections.

No renderer, photo request format, shader, service, or live exposure was
changed by this investigation. No continuous-surface toggle is implemented.
This document is the bounded integration design and the exact engineering
blocker, not a claim that support has shipped.

## Original mechanisms, with their actual semantics

### 1. Triangle-group surface refinement

Authoritative inputs are `TriGroupDesc` in `csrc/include/triangle_groups.h`
and `RayTracer.register_tri_group` in `csrc/bindings/pybind_kernels.cpp`.
Membership is an explicit list of triangle indices, not a material ID.
The group carries `parametric_surface_kind` and an owned payload.

* `POLY_BARY` (1) uses six float64 coefficients for a quadratic displacement
  in a triangle's barycentric coordinates.
* `SDF_SADDLE` (2) uses `[amplitude_m, neighborhood_margin_uv]`.
* `SDF_SPHERE` (3) uses `[radius_m, neighborhood_margin_uv]`.
* `NEURAL_ASSEMBLY` (4) is deliberately disabled/fail-closed in the native
  optical pipeline until it supplies reversible optical metadata.
* `PARAMETRIC_LENS` (5) uses a float32 assembly payload, magic 14949.

Important correction to the earlier handoff: **SDF_SPHERE is not an exact
world-space sphere primitive.** `apply_parametric_surface_point` in
`csrc/kernels/ray_tracer.cpp` evaluates

    cu = u - 1/3; cv = v - 1/3
    delta = (cu*cu + cv*cv) / (2*radius)

and a Jacobian normal in each hit triangle's local frame. The final point is
`hit_pos + warped_normal * delta`. This is a post-triangle-hit refinement;
it does not solve the incoming ray against a shared sphere center/radius.
It cannot recover rays missing a coarse triangle silhouette, and generally
does not leave the refined point on the original ray. This is the existing
contract, not a reason to silently redefine kind 3.

There is also an observed source-level CPU/GPU parity difference: the CPU
averages nine neighborhood samples when margin > 0, while
`ray_refine.comp.glsl::refine_sphere` evaluates only the central sample.
`upload_param_groups_data` copies the radius but not the second payload
value into its small GPU payload. At barycentric centroid the GPU delta is
zero; the CPU expression is nonzero for a positive margin. No native
runtime parity experiment was run here, so this is a source-derived
finding, not a measurement of the installed binary.

### 2. Exact parametric compound-lens transport

`camera_designer/lens_assembly.py::_register_parametric` registers front
and back proxy groups. `CompoundLens.build_gpu_payload()` is the source
of the physical surface train. CPU `apply_parametric_lens_from_f32` and GPU
`ray_refine.comp.glsl::parametric_lens_teleport` seek the actual conic
surfaces, apply Snell/Fresnel, accumulate optical path length, and emit
optical event metadata. Interior proxy groups absorb to prevent duplicate
intersection after traversal. Reverse rays use the canonical payload in
reverse order.

This is a sequential optical-assembly transport contract. Do not reuse
the whole-assembly teleport as an arbitrary marble volume: it would skip
the existing ribbons, confetti, bubbles, and other surfaces between entry
and exit. A marble boundary must return control to nearest-hit traversal
after each interface so those interiors remain observable.

The existing closed-form CPU geometry references also include
`parametric_surface.py::ConicSurface.intersect_ray`/`normal_at_point` and
`camera_designer/parametric_surfaces.py::SphericalSurface`. These are useful
oracles for an exact spherical boundary, but are not automatically called
by the current Studio GPU photo programs.

### 3. Scale-context regions

`RtScaleContext` in `csrc/include/ray_tracer.h` is a sphere-triggered medium
or specialized-handler region. It carries `context_kind` separately from
`scale_type`, plus an opaque payload. Native
`dispatch_scale_context_entry` handles the kind. `SPLINE_SURFACE` currently
only returns an entry flag, with no spline intersection implementation.

The header's claim that kind dispatch is mirrored into both photo GLSL
programs is not true of the current source:

* `build_gpu_scene` packs only eight floats per context:
  `[cx,cy,cz,r,dt_m,n_real,n_imag,scale_type]`.
* `_GPU_RAY_FIELD_CS` reads those two vec4s to split propagation spans.
  It has no `context_kind` or payload-dispatch field.
* `_GPU_SENSOR_CS` has no ScaleContext buffer or equivalent region-kind
  dispatch.

Thus supplying a wave-context sphere is not supplying a glass surface,
and enabling a kind-4 region cannot honestly promise spline intersections.

## Where the saved-object contract currently stops

1. Turing `examples/meowdoku/marble_photo.py::decode_upload` verifies the
   uploaded MRP1 geometry hash against the immutable saved object. It gets
   the PBR16 table from that object, not from an untrusted replacement.
2. `MarblePhotoJobs.start` writes only `vertices`, `material_ids`, and `pbr`
   into the NPZ. The config carries digest, scale, camera, fixtures, etc.,
   but no generator/design, region ownership, or parametric payload.
3. Spectral `append_saved_studio_mesh_to_gpu_scene` removes degenerate
   triangles and appends `TriGeom`/`TriShade` records. It does not return
   the original-to-retained triangle index mapping needed for groups.
4. Both photo shaders' `nearest_hit` return a triangle index from
   Moller-Trumbore traversal. The subsequent material lookup and normal
   load assume that triangle representation.

The saved object's generator/design already exists in
`examples/meowdoku/marble_objects.py`, and `marble_mesh.js::catEyeMarble`
separates `glass` from `interior`. But the photo bridge discards that
ownership. Material ID is not a safe replacement for shell ownership:
materials can be shared, edited, copied for enamel, or used by interiors.

## Integration work, in dependency order

1. **Preserve authored region identity.** Extend the saved design/photo
   handoff with versioned geometry semantics derived from the authenticated
   generator/design and explicit triangle membership. Include mode and
   payload in the existing digest. Retain the current geometry hash check.
   Preserve membership through degenerate-triangle filtering and scene
   concatenation. Do not infer regions from PBR IDs or coordinates alone.

2. **Choose and preserve the actual original contract.** For existing
   triangle-local parametric regions, carry the original enum/payload and
   use the original surface evaluator. First resolve and test the observed
   neighborhood-payload parity gap. Describe this mode as parametric
   refinement, not a continuous spherical marble.

3. **For the requested continuous marble option, use an exact boundary.**
   Adapt the original parametric surface equation and ray/normal contract,
   rather than reusing the misleading SDF_SPHERE name. Return true distance,
   outward normal, material/medium identity, and primitive/region identity.
   Compare against all remaining scene triangles before choosing the hit.
   Remove only the corresponding shell triangles from geometric traversal.
   Retain triangles deliberately in triangle mode and retain all interior
   geometry in either mode. Proxy bounds must enclose the actual surface,
   or rays outside a coarse silhouette will still be missed.

4. **Use identical geometry in both GPU photo directions and visibility.**
   The forward field, sensor path, direct-light visibility, subsequent
   dielectric bounces, and source sampling when the region is emissive must
   see the same surface contract. Do not add a normal-only adjustment after
   `nearest_hit`, or an analytic hit in just the camera pass. Reuse the
   existing MatBuf optics and film pipeline, without changing transport
   bases, bounce budgets, or exposure normalization as a side effect.

5. **Bound the initial supported designs honestly.** A uniform-material
   spherical shell is the smallest exact case. Preserve its physical
   saved radius and center. Partial enamel and painted bands are assigned
   by cell centers/column rounding in the current mesh generator, so an
   exact continuous material field needs an explicit matching convention.
   Reject unsupported combinations rather than substitute one material.
   Dice, arbitrary spline regions, and neural transport are separate cases.

6. **Expose the toggle after transport validation.** Persist mode, echo it
   in effective config/status/HDR metadata, and invalidate incompatible
   jobs/caches. Leave triangulated mode available for its intended optical
   artifacts. An existing exposure must finish or be explicitly cancelled
   by its owner before integration/restart.

## Required acceptance evidence

* Existing triangle-mode payloads and exposures remain unchanged.
* Outside, inside, tangent, miss, and near-surface rays match the original
  float64 parametric oracle. Verify both roots, front/back identity, and
  `hit = origin + t*direction`, not only normal appearance.
* A low-resolution shell exhibits round silhouette and continuous
  refraction only in the exact mode, including rays outside the mesh's
  inscribed silhouette. Record numerical hit and normal error separately
  from image comparisons.
* Nearest interior and background triangles still win when appropriate.
  No teleport skips colored inclusions. No original shell double-hit.
* Snell reciprocity, normal-incidence Fresnel, exit refraction, total
  internal reflection, and preserved optical path length are tested.
* Forward/sensor/visibility queries agree for the same rays and materials;
  test fixed:1 and fixed:3 without film retuning to hide differences.
* Invalid kind/payload, stale group indices, unsupported coatings/shapes,
  and modified saved geometry fail clearly before GPU work begins.
* Run a separate bounded GPU case, then compare saved exposures at the
  requested practical resolution/budget. Do not infer performance from
  the native tracer's historical benchmark or from a CPU unit test.

## Verification performed

Current Spectral HEAD: `cff63425be11e9a1b97987f623b729a7bd2c2f7d`, with existing
uncommitted film/color work preserved. Source inspection included the
native registrar, CPU evaluator, GPU refiner/uploader, lens registration,
scale dispatch, saved-object bridge, scene adapter, and both photo shaders.

    python -m pytest -q tests/test_saved_studio_gpu_scene.py tests/test_exact_camera_transport_contract.py

Result: **8 passed in 0.55s**. These are existing CPU/source-contract tests,
not end-to-end continuous-surface rendering tests. No new GPU job, native
build, Wasm build, deployment, cancellation, or service restart was run.

The exact blocker is missing cross-pipeline geometry/region identity and
intersection support, plus unresolved semantics/parity for some existing
kinds. There was no access denial or pending permission request.
