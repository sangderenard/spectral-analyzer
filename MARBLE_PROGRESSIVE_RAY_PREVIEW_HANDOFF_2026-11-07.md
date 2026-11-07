# Marble designer progressive Spectral preview — 2026-11-07

## Current result

The existing Spectral GLSL sensor now has a validated **256×256 monochrome
thin-lens glass/checker still** on the local RTX 3060. Three BK7 closed spheres
visibly bend a checker; the same geometry made opaque leaves a dark occlusion.
The production scene builder also accepts arbitrary saved Studio triangles
and their exact edited PBR16 table through
`append_saved_studio_mesh_to_gpu_scene()`. This has been GPU-tested with both
the three-sphere control and a final 97,280-triangle Studio cat-eye mesh.
Meowdoku now has an explicit saved-object photo panel and a background HTTP
worker route beside the interactive Pluck Phong view. A temporary local HTTP
server completed the full upload/status/image path. The live site deployment,
same-camera Phong registration, and physical caustic photo remain separate
gates. Do not display a synthetic or zero-valued image as a ray-traced preview.

## Existing faculties and the relevant route

`RAY_TRACER_INVENTORY.md` distinguishes the GLSL forward field and backward
sensor tracer, the C++ `_spectral_kernels.RayTracer`, lens baking, and a Numpy
geometric picker. The Numpy picker is single-hit geometry only; it cannot
produce glass refraction or caustics. `exposure_render_demo.py` provides a
headless still renderer (`--backend cpp --no-window --save-files`) with the
native `RayTracer` and `MaterialDatabase.build_mat_buf()`. Its current scene
builders cover fixed demonstrations and the thick-lens lab, whose authored
subject override is text geometry. They do not ingest an arbitrary marble
triangle mesh. The GLSL path is OpenGL compute, not a WebGL compute shader
that can run in the existing browser page.

The GLSL `SensorAccumulator` has a bounded progressive schedule:
`pump_forward(budget_ms=5.0)` advances light rays from a rotating dispatch
ring, and `tick()` advances the backward camera pass. Its shader has an
**analytical thin-lens** path: each sensor pixel aims at a focus point,
samples a finite circular or blade-shaped aperture, and offsets the ray
origin to the sampled pupil. It also accepts focal length/FOV, focus distance,
chromatic aberration, tilt-shift, Scheimpflug tilt, and a lens-manifold path.
`Renderer.tick_sensor()` currently resets on a display cadence; a still
preview should use its continuous-exposure mode (`sensor_fps=0`) and reset
only on a scene revision. This path needs a server-side OpenGL compute
context and the existing BVH/SSBO scene packing, not browser WebGL.

The installed native module at
`C:/dev/Powershell/spectral-analyzer/_spectral_kernels.cp311-win_amd64.pyd`
has the modern sensor pipeline (`submit_emissive_triangles`,
`submit_sensor_sweep`, `join_t5`, `get_sensor_image_linear`). It does not expose
the older `bidirectional_packed` method. `exposure_render_demo.py` contains
the current lifecycle in `CppExposureBackend.run_thick_lens_native_bdpt()`.

Use the sanctioned fixed **one-line** 587.5618 nm transport contract for the
first black-and-white optical preview. The user accepts a monochrome still
beside the Phong view. `camera_software/calibration_modes.py` already uses
that line for its focus-hall camera exposure. `SPECTRAL_TRANSPORT_ABI.md`
explicitly allows fixed lines for this work. `RayTraceSettings` validates
`transport_mode="fixed", lane_count=1` (`fixed:1`). This reduces fixed-lane
per-ray side data and material work versus 3/8/16/32 fixed lanes while
retaining actual geometric refraction at the reference wavelength. It does
not show chromatic dispersion. For later color, existing three-line contracts
are 450/550/650 nm and BK7 486.1327/587.5618/656.2725 nm (`fixed:3`).
The transport contract and exact frequency set belong in the scene digest.

## Bounded measurements

### GLSL proof, 2026-11-07

`demo_pluck_gl._ensure_worker_gl_context()` created an OpenGL 4.3 NVIDIA RTX
3060 context in the server process. The canonical `build_gpu_scene()` packed
the 4×4 authored emissive checker target, physical camera geometry, one fixed
587.5618 nm material lane, and three closed BK7 spheres. The actual
`SensorAccumulator` analytical thin-lens shader used a 5 mm aperture, 23 cm
focus distance, 20° FOV, four bounces, and a 256×256 film. The film eye was
placed at z=7 cm in front of the demo doublet, so the **analytical thin lens**
formed the image while the physical demo lens stayed behind the eye. This
avoids compounding two lens models in the preview proof.

128 bounded epochs dispatched 1,048,576 forward ray jobs and 2,097,152 sensor
sites in 8–9 seconds. The no-sphere target is a readable checker. The glass
spheres show shifted and inverted bright squares within their silhouettes;
the opaque spheres show a dark blocking shape. At equal exposure the central
ROI mean sensor value was 8.20 through glass versus 0.223 through opaque
(~36.8×). These are direct sensor signals, not a synthetic image blend. The
GPU adapter path using the PBR-derived material records also completed in
8.3 seconds with the same refracted checker. Images are local diagnostic
artifacts under `build/marble_ray_probe/`:

* `gl_lens_qr_epoch127.png` — empty checker.
* `gl_lens_sphere_qr_epoch127.png` — three transparent spheres.
* `gl_lens_opaque_qr_epoch127.png` — same opaque geometry.

The previous forward-field-only target appeared as sparse bright clusters:
the backward sensor shader sampled forward volume energy but did not add
direct radiance at an authored emissive surface. `demo_pluck_gl.py` now uses
the existing `MAT_FLAG_EMISSIVE`/`TriShade` emission contract at the sensor
hit. `scene_builder.py` adds `checker_target` with optional `emissive` white
squares. Both were tested on the GPU and committed as `de1cff4`.

The stills prove refraction, not caustics. Existing `SensorAccumulator`
surface emission is scalar for this one-line preview and its alpha channel
is a presence mask, while RGB accumulates exposure. No spectral color,
photo-calibrated exposure, or physical surface caustic measurement is implied.

### Saved Studio mesh and worker proof

`camera_designer.marble_preview_worker` is a real separate-process CLI using
the same scene builder, BVH builder, GPU forward field, and thin-lens backward
sensor. The saved mesh proof came from the production
`marble_mesh.js:catEyeMarble()` at Studio's 96 rings × 128 segments and three
closed ribbons: 97,280 triangles (72,960 interior, 24,320 shell). It was
serialized to a 606,652-byte NPZ and photographed by the worker with a
2 mm aperture, 11 cm focus, 36° FOV, and a 256² sensor. The full marble,
inclusions, and refracted checker are visible at
`build/marble_ray_probe/worker_marble_sync/preview-000357.png`.
Scene preparation took 2.0 seconds and 3.0 seconds of exposure dispatched
2,924,544 forward ray jobs and 5,849,088 sensor sites. The worker published
intermediate stills at roughly one-second intervals in this smoke. The final
status has digest `bbeea44ec1e2d1b733f74fed09955ba074f90178d03b2e13d4f14f567283224a`.

Worker input is an immutable `.npz` with `vertices` float32 `(N,3,3)` in
Studio **radius units**, `material_ids` int32 `(N,)`, and `pbr` float32
`(M,16)` containing the exact edited `MaterialDatabase` PBR table rows. The
client must concatenate final `lastGeometry.interior` and `.glass` faces in
their current winding and send each face's stable material ID. Geometric face
normals are derived from that winding; the current Spectral sensor does not
interpolate Studio's per-vertex normals. A JSON config supplies `scale_m`,
`object_center_m`, `camera` (`eye_m`, `target_m`, `resolution`, `fov_degrees`,
`aperture_radius_m`, `focus_distance_m`, optional blades/rotation), optional
`lights`, and `checker` (`center_m`, `width_m`, `checks`). Defaults are in the
worker source. The worker hashes the NPZ bytes plus canonical JSON into its
scene digest. For a saved design, the server can hold the immutable NPZ/JSON
and reuse the matching process/output until that saved revision changes.

Invoke from the Spectral repository root, for example:

```text
python -m camera_designer.marble_preview_worker --payload DESIGN.npz --config DESIGN.json --out PHOTO_DIR --seconds 600 --native-module-dir C:\dev\Powershell\spectral-analyzer
```

`status.json` is atomically replaced and names the latest
`preview-XXXXXX.png`, also written atomically. It carries digest, state,
elapsed/preparation seconds, forward ray jobs, sensor sites, transport, and
visible pixel count. An HTTP server should terminate the worker subprocess
on revision change and reject any image/status with a stale digest. The CLI
enforces at most 600 seconds of **exposure** and syncs with the GPU every
eight epochs to keep wall-clock progress honest. It restarts the 32-bit
forward light field every 4,096 epochs while keeping the sensor exposure,
preventing voxel counter overflow during a ten-minute photo. The worker's
JSON image paths are basenames for the owning server to serve.

### Temporary saved-object HTTP proof

The Meowdoku `MRP1` endpoint accepted the **actual final 97,280-triangle
Studio mesh** in a temporary saved-object store; it did not touch the live
library or server. Its binary request carried all face positions, all
per-vertex normals, and material IDs. The server recomputed the saved
object's 76-byte-per-face geometry SHA-256, loaded that object's exact PBR16
table, wrote the worker NPZ/JSON and returned HTTP 202 with photo digest
`6faecacbfea578a2bfefc89b1b003f4f5567e3ca73c9992cefc4afa114e189d9`.
Repeated `GET /api/marble_photo?digest=...` advanced to `complete` without a
worker error. The worker reported 4.015 seconds of preparation, **2.0 seconds
of photo exposure**, 1,761,280 forward ray jobs and 3,522,560 sensor sites.
`GET /api/marble_photo/image?digest=...&name=preview-000215.png` returned HTTP
200 with a valid 18,708-byte PNG, preserved at
`build/marble_ray_probe/http_actual_saved_marble.png`. The image shows the
glass outline, dyed ribbons and checker squares bent through the shell.
No 10-minute exposure was run. The 600-second cap is a requested exposure
limit after preparation, not a measured quality or compile time.

The first photo is fixed one-line monochrome at 587.5618 nm. Its default
stage light is fixed unless the client sends a `lights` config. The photo
camera is a separate thin-lens pose from Studio's WebGL Phong camera and is
shown in a separate panel; same-pixel registration or blending over Phong is
not claimed. The current Spectral sensor shades geometric face normals,
while the uploaded smooth per-vertex normals only verify saved geometry.
Caustics, color dispersion and full material/light parity are unverified.

### Earlier native probes

The probe used a single emissive quad, an optional closed octahedral glass
solid, three optical frequency bands, a 16×16 sensor, and four bounces.

* Native `integrate_image_into_packed`, 3,000 forward rays, completed in
  0.38 seconds. The bare emitter produced zero image signal; adding the glass
  produced a tiny forward surface-hit spot (`sum=79.64`, `max=2.19` in the
  raw three-band accumulator). That is not a camera view of a solid with
  background distortion or credible caustics.
* Native BDPT with 300 emitter rays per UV domain and 256 sensor pixels
  entered both T5 latch submissions and did not finish within 20 seconds. It
  was interrupted. One bounded retry added the demo's
  `begin_sensor_batching()`/`end_sensor_batching()` lifecycle and had the
  same outcome. There was no completed image to inspect.

The C++ `ray_tracer_integrate_image_parallel_impl` behind
`integrate_image_into_packed` enumerates rays from light `src_pos` and uses
the camera vectors only to project surface hits. It does not launch
camera-origin rays through glass to a lit checkerboard, so the 0.38-second
forward result cannot establish distortion. The native sensor sweep or GLSL
backward sensor is required.

The native camera path supports a physical sensor pose, separate aperture
plane pose, aperture samples per sensor pixel, and a thin-lens scale-context
transform with a focal-length payload. Merely increasing aperture samples
does **not** install the thin-lens transform; the current
`exposure_render_demo.py` lens-context registration and solved camera rig
must be followed. The test probe set a sensor and aperture pose but did not
register a thin-lens scale context; its stalled T5 result is therefore a
pipeline diagnostic, not a thin-lens image evaluation.

`ray_tracer.cpp` says T5 starts only after the GPU dispatch loop drains all
in-flight paths. `join_t5` has a 30-second **idle** watchdog but no finite
busy-path deadline. Before using it in a server worker, diagnose the native
dispatch state, path counts, and overflow queues with a bounded process and a
known-good Spectral scene. A process boundary is necessary so a request can
be canceled even when the native call does not return.
Log `in_flight_count()`, `pipeline_stats()`, `get_bdpt_latch_state()`, and
`get_bdpt_overflow()` while the worker is running. The C++
`ray_pipeline_gpu_dispatch_state` diagnostic is not exposed by the installed
Python binding; expose it only if those measurements point to GPU dispatch.

For thin-lens image formation, the GLSL `SensorAccumulator` is the shortest
existing route with bounded work: its analytical shader samples a finite
aperture, focuses each pixel ray at `uFocusDist`, and supports blade count,
aperture rotation, chromatic aberration, tilt-shift, and Scheimpflug tilt.
Its `rows_per_frame` bounds backward pixel dispatch; the forward pump has
`budget_ms`. `SensorAccumulator` owns the sensor texture and the camera
settings; `Renderer.sync_sensor_camera()` supplies pose/focus, and
`Renderer.tick_sensor()` calls the two passes. In native C++,
`PhysicalCameraRig` and `_build_camera_rig_mesh()` establish sensor/aperture
geometry, `configure_sensor_image()` and `configure_sensor_pose()` install
the film and pupil, `submit_sensor_sweep(aperture_samples=...)` samples pupil
rays, and `_configure_default_wave_contexts()` registers the separate
`SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM` with focal length. Aperture sampling
alone is not that lens transform. The current standalone probe did not build
the solved native camera package, so its T5 stall is a pipeline diagnostic,
not a thin-lens optical comparison.

## Scene bridge contract

The designer's final geometry is `lastGeometry` from `catEyeMarble()` or
`diceSolid()`. It contains `interior` and `glass` face arrays in normalized
radius units. Every tuple is `[a,b,c,materialId,[na,nb,nc]]`. Dice relief,
bevel, and face changes are already in those final triangles; do not
regenerate an older sphere on the server. Send triangles as binary typed
arrays (positions, normals, integer material IDs) rather than stringifying
100,000 face objects. Include the real scale, case orientation/position,
camera pose, floor/wall geometry and material IDs, and the stage-light
settings. The current stage uses five key samples and two fills. Preserve the
thin-lens camera: position/orientation, focal length, focus distance, aperture
radius and shape/blades, sensor plane and resolution, per-pixel lens samples,
and any chromatic aberration or tilt. The ray image must be registered to the
same view as the Phong scene before blending; otherwise the overlay is
misleading even if its optical paths are real.

The browser's optical table truth is `tables.{pbr,phong,enamel,texstack}.data`
and stable IDs from `assets.material_index`. `web_export.py` builds those IDs
by registering `PHONG_MATERIALS` and `MATERIAL_CATALOG` into the canonical
Spectral `MaterialDatabase`. The server bridge must use those same sources
and `MaterialDatabase.build_mat_buf(freq_hz)` for native spectral-band rows.
It must apply edited PBR parameters and verify their spectral mapping before
claiming parity; directly passing the exported PBR float texture as a native
`MatBuf` would be incorrect. A material edit must invalidate accumulation.

There is a concrete glass mismatch to resolve. `build_mat_buf()` maps
a PBR-only material's **albedo luminance** to spectral reflection and clips
transmission to the remaining budget. Meowdoku `marble_clear_glass` has high
PBR albedo (~0.93–1.0), so that fallback reflects nearly all rays despite
its 0.96 visual transmission slider. Spectral's existing
`configs/materials/borosilicate_glass.yaml` correctly authors low surface
scatter/reflection and scalar IOR. Its Abbe/Cauchy metadata is **not** used
automatically by `MaterialDatabase.build_mat_buf()`; the calibrated BK7
`camera_designer.optical_material` response would need a deliberate per-line
spectral-band mapping for future dispersion. Use the authored low-scatter
glass optical response for the one-line preview
and distinguish appearance tint from physical surface scatter; do not claim
caustics from the PBR-only fallback. Verify any later three-line material
response and Sellmeier/Cauchy mapping before release.

The client should hash the complete scene revision: mesh parameters and
triangles, material tables, camera/roll pose, lights, and floor/wall options.
It must also include every thin-lens setting and sampling policy.
Include `fixed:1` and 587.5618 nm in the first preview's digest; never mix
samples from different frequency contracts.
The server should key a single background accumulation job by that digest,
publish progressively increasing sample counts and images only under that
same digest, and cancel the previous worker when the digest changes. The
client must discard responses from earlier revisions. A Python server may
host this process, including a local GPU, but a GPU is not sufficient evidence
that the current BDPT job completes cheaply.

## Next gate

Run the live or dev Studio page end to end after export and server restart:
save a design, choose an explicit 30/180/600-second photo exposure, watch
intermediate images, then edit the design and verify cancellation and digest
invalidation. The temporary HTTP/GPU run proves the code path but does not
prove the deployed page. Do not eagerly compile Wasm for this preview.

For optical fidelity, align the thin-lens photo camera with the Studio WebGL
camera before offering a registered overlay, use Studio's current stage
light controls rather than the worker's fixed default, and compare exact
material responses at the chosen wavelength. The current sensor has one
geometric face normal per triangle; a per-vertex normal interpolation path
would be needed to match all Studio Phong smooth transitions. Test a saved
die with bevel/relief and a designed inclusion, then verify its optical
view against the Phong shape.

The existing sampler has pupil sampling and a row/center-priority mode;
record the ray/sample allocation and identical exposure controls for glass,
opaque, and empty comparisons. Prefer source aiming and the authored
emissive checker over more uniform rays that miss the useful scene. Do not
claim caustics until a controlled lit-floor pattern or energy concentration
appears in the photo.

Separately, diagnose the native BDPT stall on a small known-good scene with
an external process timeout and collect `get_bdpt_latch_state()`,
`get_bdpt_overflow()`, `pipeline_stats()`, and image statistics. Fix the
lifecycle or budget in Spectral itself before using native BDPT as preview.
