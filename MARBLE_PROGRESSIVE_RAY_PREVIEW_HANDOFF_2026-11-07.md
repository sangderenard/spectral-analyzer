# Marble designer progressive Spectral preview — 2026-11-07

## Current result

The existing Spectral renderers have the required refraction and camera/light
transport machinery. There is **no validated cheap still-image path for the
current Meowdoku marble scene yet**. The Meowdoku designer still renders with
its Pluck Phong shader. Do not display a synthetic or zero-valued image as a
ray-traced preview.

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

There is a concrete glass mismatch to resolve first. `build_mat_buf()` maps
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

First run the existing GLSL `SensorAccumulator` on a small **known-good**
glass/checker scene in a server-owned OpenGL compute context, fixed at one
587.5618 nm line, a nonzero aperture, finite focus distance, low sensor
resolution, bounded forward `budget_ms` and backward `rows_per_frame`, and
continuous exposure. Confirm nonzero, changing refraction behind the glass
before adding marble geometry. The current demo does not expose a headless
still-image service for that path; its scene SSBO builder and sensor texture
readback must be used directly, without a new tracer.

In parallel, diagnose the native BDPT stall on a small known-good scene with
an external process timeout and collect `get_bdpt_latch_state()`,
`get_bdpt_overflow()`, `pipeline_stats()`, and image statistics. Fix the
lifecycle or budget in Spectral itself before using native BDPT as preview.
Then add the marble scene adapter and compare a clear shell, a dyed
inclusion, and checkerboard refraction against the Phong view.
Only after a nonzero, timely optical image is demonstrated should Meowdoku
add background job endpoints and an overlay. Keep the Phong scene interactive
throughout accumulation and label the ray sample count and transport mode.
