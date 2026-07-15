# Declarative scene-order pipeline

`scene_orders.py` is the subject-authoring layer above the validated native
thick-lens BDPT renderer. It compiles JSON orders into `TracerScene` geometry
while retaining the lab scene's sensor, aperture blocker, lens assembly,
camera enclosure, ring light, and side-room flash. The base scene's entire
subject/object group is replaced, including any emissive demo geometry in that
group. It does not substitute a raster camera or a simplified lighting
renderer.

The first completed contract supports:

- A package containing one or more uniquely named jobs.
- One token/string per job, resolved through a font family or explicit font
  file.
- Compound font contours and counters/holes, triangulated front and back caps,
  closed side walls, and non-degenerate mesh validation.
- Straight or segmented circular-bulge extrusion profiles.
- Signed depth relative to an embedding plane via `embed_fraction`; `0.5`
  places half the extrusion behind the plane face.
- Any number of finite, arbitrarily oriented, thick planes.
- Separate order-authored spectral materials. Linear RGB is baked into the
  renderer's eight visible frequency bands; roughness and metallic remain in
  the unified material rows used by native T3/T5.
- Image width/height, focal length, aperture diameter, ISO, exposure time, T5
  pair budget, and flash intensity scale.
- Independent world-space camera `position_m`, `target_m`, and `up`, with
  planes and token geometry remaining in authored world coordinates.
- Rectangular `image.region` sensor scans. The renderer traces only that
  physical sub-sensor area and returns an image with the region's exact pixel
  dimensions instead of returning the full frame.
- A `composition_manifest.json` that records the full-frame dimensions,
  top-left tile rectangle, coordinate space, and normalized camera frame.
- Sequential batch processing into one output directory per job.

Unknown job, camera, flash, plane, or geometry fields are rejected. This is
intentional: an order must never appear successful while a requested physical
setting was silently ignored.

## Commands

The supplied examples include a circularly extruded bold `A` and a complete
`AXE` token, each embedded halfway into a matte-black backplate with glossy red
material:

```powershell
python render_scene_order.py configs/scene_orders/glyph_a_red_gloss.json --validate

python render_scene_order.py configs/scene_orders/glyph_a_red_gloss.json `
  --compile-only --out-dir exposures/scene_order_compile

python render_scene_order.py configs/scene_orders/glyph_a_red_gloss.json `
  --render --out-dir exposures/scene_order_renders

python render_scene_order.py configs/scene_orders/token_axe_red_gloss.json `
  --render --out-dir exposures/scene_order_renders

python render_scene_order.py configs/scene_orders/spatial_tile_axe.json `
  --render --out-dir exposures/spatial_tile_render
```

For a multi-job package, omit `--job` to process every job or repeat it to
select a subset:

```powershell
python render_scene_order.py package.json --render `
  --job glyph_A --job glyph_B --out-dir exposures/font_batch
```

Each rendered job stores its resolved request as
`resolved_scene_order.json`, its compositor-facing `composition_manifest.json`,
the tile PNG, 16-bit PNG, linear NumPy image, and scientific summary. For
example, a region `{x: 208, y: 156, width: 96, height: 72}` is returned as a
96x72 image for placement at pixel `(208, 156)` in the declared full frame.

## Spatial camera model

Scene orders use metres in a right-handed world coordinate system. Plane
centres, token embedding, camera position, and camera target are all authored
in that shared space. The native thick-lens kernels currently evaluate their
compound optical assembly on a canonical X axis, so compilation applies one
rigid world-to-camera change of basis to subject geometry. This is physically
equivalent to moving and aiming the camera: distances, angles, lens mechanics,
spectral transport, and material data are unchanged. It also avoids claiming
that axis-specific native lens handlers can consume an arbitrarily rotated rig.

`camera.target_m` controls aim. `camera.focus_target_m` independently identifies
the world-space plane that must be sharp. The compiler projects that point onto
the viewing axis; the thick-lens camera package carries the resulting
sensor-to-plane distance. Ordered scenes also rebuild the physical compound
lens and sensor conjugate for that distance; focus metadata alone is not
sufficient because the native sensor sweep traces through the actual glass and
does not consume `focus_distance_m`. The live paragraph places this target on
the exposed glyph face rather than inferring focus from aggregate scene bounds.
Focused base-camera meshes are cached by sensor-to-plane distance so subsequent
text revisions reuse the same optical assembly.

`image.width` and `image.height` declare the eventual composition frame.
`image.region` uses top-left pixel coordinates within that frame. Its fractional
extent maps directly onto the physical sensor width and height, so transport is
spent on the requested area rather than on unseen full-frame pixels. The native
sensor accumulator is internally square. Rectangular scans use a square native
schedule that covers the complete accumulator, then explicitly resample to the
requested output dimensions during readback while preserving the linear
array's native dtype. A rectangular schedule must not be submitted to that
square accumulator: doing so leaves a hard, stair-stepped unexposed region that
can be mistaken for an aperture vignette.

## Live spectral text

`live_spectral_text_demo.py` is an editable paragraph window backed by the same
native thick-lens BDPT path:

```powershell
python live_spectral_text_demo.py
```

The default display contract is an exact 960x600 ray-traced texture. It can be
set to the intended hardware raster explicitly:

```powershell
python live_spectral_text_demo.py --display-width 1920 --display-height 1080
```

Those values are source pixels, not a window scaling hint. The live order keeps
the established central sensor crop by declaring a virtual full-sensor frame
five times larger on each axis, while `image.region.width` and `height` equal
the requested display raster exactly. For a 960x600 output, the current square
native accumulator therefore executes a 960x960 schedule and readback returns
960x600. Increasing resolution increases real transport work; it does not
interpolate a smaller exposure.

Keyboard input updates an immediate text variable. After one second without an
edit, a dedicated worker snapshots the newest paragraph into a scene order and
runs one complete exposure. An in-flight exposure is never interrupted or
partially published. Further edits coalesce into one newest pending revision,
and the UI keeps displaying the last completed ray-traced texture until the
next revision is finished. The pygame preview never enlarges completed source
pixels: it displays at 1:1 when space permits, letterboxes larger windows, and
only downsamples when the window is smaller. The editor/status HUD occupies a
separate strip that extends the initial window below the requested canvas; it
never overlays the traced image. Narrow requested displays use smaller fonts,
padding, fewer editor lines, and compact status wording.

The parent process reports the requested output raster, actual square native
sensor raster, composition coordinates, revision submissions, and completion
times directly to the launching terminal. Child renderer output is line-buffered
and teed to both that terminal and `revision_NNNN/render.log`; native aperture
sample counts, camera-ray schedules, T5 work, and convergence messages are no
longer hidden inside the log file.

`exposure.sensor_sweeps` is the progressive image-quality axis. Each sweep
repeats the complete sensor schedule with disjoint progressive pupil samples;
the linear sensor accumulations are averaged after readback in their native
dtype. Runtime scales approximately linearly with this count. In contrast,
`--bdpt-native-packages` only sets a minimum number of memory-safe spatial
partitions inside each sweep and does not add samples. An explicit
`--bdpt-native-sweeps` value overrides the scene-authored count for controlled
diagnostics, and terminal reporting prints both values plus every sweep/unit.

There is no bloom, blur, denoising, sharpening, or convolution stage in the
saved-image path. The previously reused lab camera was physically solved for
its default one-metre scene rather than the paragraph's approximately 3.73 m
sensor distance, so aim and focus metadata did not make the glass focus the
text. The corrected complete mesh places the authored glyph plane at the
requested axial distance. Its exact compound-lens EFL is approximately 137.6 mm
and its 20.62 mm iris is approximately f/6.7, so a wide-open aperture is not the
primary blur source in that build.

Native `get_sensor_image()` is a display preview: it normalizes by the frame's
99th-percentile positive exposure and applies `log10(1 + 9x)`. That curve can
visually lift faint transport around bright text. `get_sensor_image_linear()`
now exposes the unnormalized sensor accumulation; `_linear.npy` uses that raw
array while the PNG retains the existing display preview for direct comparison.

The current live paragraph uses a straight 12 mm extrusion, not a circular or
bulged profile. Its warm beige `[0.82, 0.76, 0.62]` albedo intentionally favors
red and green over blue, so a yellow cast is partly authored rather than solely
a chromatic lens artifact.

Paragraph geometry uses font metrics for greedy word wrapping, hard wraps long
unbroken words, shares one fitted physical line scale, and centers the final
contour bounds horizontally and vertically inside `geometry.text_box_m`. The
background plane is deliberately oversized and nearly diffuse so no finite
plane edge or strong material highlight competes with the text window.

## Evidence

The current accepted render is
`exposures/scene_order_renders_v7/token_AXE/0000_cpp.png`. It is a 64x64 native
four-unit exposure of a 4,458-triangle closed multi-glyph token and authored
backplate. Compilation removed 2,640 subject-emitter triangles inherited from
the orbiter demo and retained the 144 rig-light emitter triangles. The final
native pass reported 2,561/4,096 lit sensor pixels and completed in
approximately 113.6 seconds. The image reads `AXE` left-to-right and contains
only the authored token and background as subject geometry.

The spatial-tile acceptance render is
`exposures/spatial_tile_render_v2/remote_AXE_tile/0000_cpp.png`. Its authored
camera is at `[3, -2, 1]` metres aimed at the world origin, while the local
backplate and token are authored at the origin. The returned PNG and linear
array are both 96x72; the linear array remains `float32`, and the composition
manifest places the tile at `(208, 156)` in a 512x384 frame.

The initial live-paragraph transport acceptance render is
`exposures/live_text_acceptance/live_paragraph/0000_cpp.png`. It is an exact
80x48 `float32` tile containing a five-line, 7,008-triangle paragraph extrusion.
All 3,840 output pixels and all 256 border samples are exposed, with no empty
row or column. This confirms that the earlier stepped edge was rectangular
schedule under-coverage, not an aperture silhouette or physical vignette. Its
80x48 raster is coverage evidence only; it is not the live demo's current
detail-resolution target.

The progressive-refinement diagnostic at 120x100 executed four complete
120x120 native sweeps: 12 spatial cycles and 2,227,200 recorded rays in 449.08
seconds. Against the one-sweep raw linear result, the corrected four-sweep
average preserved mean luma within 1.4%, reduced high-frequency luma variation
to 0.487x, and reduced high-frequency chromatic variation to 0.654x. The luma
result closely matches the 0.5x standard-deviation reduction expected from four
independent samples. This confirms that additional sensor sweeps genuinely
refine estimator noise; stable residual softness and color remain properties of
the authored material, geometry, illumination, and physical camera transport.

The first attempted render was correctly rejected as evidence: inline PBR/RGB
fields alone produced zero native spectral reflectance. The compiler now emits
explicit spectral bands, and tests require nonzero matte and glyph reflectance,
a red-dominant band order, non-degenerate geometry, exact half embedding, and
preservation of camera and photographic-rig groups while rejecting inherited
subject emitters.

## Current boundary

This is a working vertical slice, not yet a universal asset system. Arbitrary
imported meshes, CSG, per-flash position/direction/shape, compound lens element
orders, font layout/kerning controls, and resumable batch queues are not
implemented yet. Those fields are rejected rather than guessed. The next
extensions should add these as explicit schema versions while continuing to
compile into the same `TracerScene` and exposure contracts.

New subject types must declare their ownership group so replacement cannot
silently retain geometry from the base demonstration scene.
