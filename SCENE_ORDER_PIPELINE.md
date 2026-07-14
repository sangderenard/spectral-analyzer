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
```

For a multi-job package, omit `--job` to process every job or repeat it to
select a subset:

```powershell
python render_scene_order.py package.json --render `
  --job glyph_A --job glyph_B --out-dir exposures/font_batch
```

Each rendered job stores its resolved request as
`resolved_scene_order.json`, followed by the normal exposure PNG, 16-bit PNG,
linear NumPy image, and scientific summary.

## Evidence

The current accepted render is
`exposures/scene_order_renders_v7/token_AXE/0000_cpp.png`. It is a 64x64 native
four-unit exposure of a 4,458-triangle closed multi-glyph token and authored
backplate. Compilation removed 2,640 subject-emitter triangles inherited from
the orbiter demo and retained the 144 rig-light emitter triangles. The final
native pass reported 2,561/4,096 lit sensor pixels and completed in
approximately 113.6 seconds. The image reads `AXE` left-to-right and contains
only the authored token and background as subject geometry.

The first attempted render was correctly rejected as evidence: inline PBR/RGB
fields alone produced zero native spectral reflectance. The compiler now emits
explicit spectral bands, and tests require nonzero matte and glyph reflectance,
a red-dominant band order, non-degenerate geometry, exact half embedding, and
preservation of camera and photographic-rig groups while rejecting inherited
subject emitters.

## Current boundary

This is a working vertical slice, not yet a universal asset system. Arbitrary
imported meshes, CSG, per-flash position/direction/shape, compound lens element
orders, camera pose/aim, font layout/kerning controls, and resumable batch queues
are not implemented yet. Those fields are rejected rather than guessed. The
next extensions should add these as explicit schema versions while continuing
to compile into the same `TracerScene` and exposure contracts.

