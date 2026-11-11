# Canonical secondary field material shader

The sole material equation implementation remains `csrc/shaders/base_material.frag.glsl`.
Turing project IR preserves its conditional branches when exporting to WebGL2.
Compile the secondary with `PLUCK_TRANSMISSION`, `PLUCK_PREMULTIPLIED_GLASS`,
and `PLUCK_SECONDARY_FIELD`. The existing optional shadow define still works.

## Render contract

When disabled, the ordinary shader with `uLinearOutput=false` retains the old
nonlinear display and scalar field behavior. When enabled:

1. Render primary opaque geometry into floating-point linear premultiplied RGBA
   with depth, `uLinearOutput=true`, and scalar `uFieldGain=0`.
   The dedicated `PLUCK_FIELD_PRIMARY` program is used only while the field is active.
   Delivered mask bits replace direct diffuse (2), body scattering/remission (1),
   and transmission (4), since the recorded field includes zero-bounce segments.
   Unmatched components retain ordinary direct lighting. Authored emission and
   ambient remain once, outside these source-component ownership gates.
2. Render secondary pass 0, body scattering/remission, then pass 1, reflectance,
   with the same primary geometry, depth EQUAL, no depth writes, additive ONE/ONE.
   Both return zero alpha, and independently require delivered material eligibility.
3. Render secondary pass 2 for all retained transparent geometry, back-to-front,
   depth-tested against opaque geometry with no depth writes. It includes primary
   sources, emission, and field response in the same draw. Blend ONE/ONE_MINUS_SRC_ALPHA.
   Do not discard transparent geometry merely because its field material mask is zero.
4. Resolve with secondary pass 3 and `uFieldResolvedColor`: the canonical
   `pluck_display` applies exposure, ACES, and sRGB once. The source color is already
   color-compensated. No material table accesses occur in this pass.

Do not run field contributions over an already tone-mapped image, and do not draw
all glass primaries before their secondary contributions: both lose correct composition.
Background clear RGB must be scene-linear (invert the existing display background if
preserving its appearance). Passes 0/1 inherit the primary's identical transform,
culling, viewport and depth convention. Use `uRenderPass=0` with host-owned mesh lists. Pass 2 must retain even
high-opacity transmissive meshes: conservatively classifying them as transparent
in the host and then discarding them by an alpha threshold in the shader loses geometry.

## Directional bundle bindings

- `uFieldRadiance`: RGB float 3D atlas, typically texture unit 11.
- `uFieldDirection`: RGBA float 3D atlas of measured weighted propagation XYZ
  and nonnegative scalar weight, typically texture unit 12. Legacy RGB mean
  direction textures have implicit weight one and use nearest sampling.
- `uFieldMaterialMask`: R32UI 2D texture indexed by material ID, typically unit 13.
  Addressing supports multiple rows. Out-of-range IDs receive no field light.
  Class bits are body scattering=1, reflectance=2, glass=4.
- `uFieldResolvedColor`: floating-point RGBA 2D texture, typically unit 14.
- `uFieldGridDepth`: depth of one bin slab, not the atlas's total depth.
- `uFieldBinCount`: integer six or twelve (default six for old native hosts).
- `uFieldAtlasColumns`: integer one or two; bin modulo columns selects X tile,
  integer bin divided by columns selects Z slab.
- `uFieldSpatialLinear`: enable only with supported floating-point linear
  filtering and weighted moment textures; false retains exact cell addressing.
- `uFieldFromView`: affine view-to-normalized-field transform; out-of-bounds contributes zero.
- `uFieldViewFromWorld`: world-to-view direction rotation, not the position transform.
- `uFieldGain`: nonnegative response multiplier on the secondary only.

Both atlases use the same X/Z tiling. Twelve directions fit in64x32x192 rather
than requiring a depth384 texture. Nearest mode uses integer `texelFetch` at
`min(floor(coord*size),size-1)` plus the tile offset. Linear mode clamps each
coordinate to the tile's first/last texel center before hardware filtering;
neighboring angular tiles never bleed. Reconstruct mean direction by dividing
filtered weighted XYZ by filtered weight. Do not interpolate unit directions
independently of their measured energy.
The texture named radiance holds the server's fixed3 track-length throughput
density converted to linear RGB, not calibrated radiance. Its estimator is
`track-length-throughput-v1`; direction is its energy-weighted world propagation
mean. The light vector is the normalized negative mean rotated into view space.
Mean length scales confidence, preventing cancelled directions from inventing light.

The host must verify the native material-ID mapping before uploading eligibility.
Native ray-scene triangle IDs do not automatically identify browser mesh triangles.
Depth visibility AND this material mask gates the opaque work; transparent layers
are separately retained and composed rather than collapsed into a nearest-ID buffer.

## Phong override

`uSecondaryPhongSpecular=false` selects field-driven Phong specular only for
materials whose delivered mask includes reflectance bit 2. Compile the Studio primary
with `PLUCK_FIELD_PRIMARY`, bind `uFieldMaterialMask` at unit 13 even when inactive,
and set `uFieldReplaceSpecular=true` only while field replacement is active and the
override is off. Keep `uEnableSpecular` at the user's setting in both programs.
The primary suppresses direct specular only for eligible materials; secondary transparent
layers retain direct-source specular on unmatched rows. The specular override is independent
of diffuse ownership, so retaining direct specular never restores duplicated direct diffuse. When the override is true,
retain direct Phong and suppress field specular. Diffuse, body scattering and transmission
remain field-driven. Existing material shininess/strength and enamel tables are reused.
The primary define is deliberately opt-in so old native hosts do not gain a new integer
sampler binding requirement merely by updating the canonical shader source.

## Limits and validation

Six or twelve bins are a bounded angular approximation, not a complete directional radiance
function. Material volumetric response means existing thickness/translucence/remit
transport at a visible surface, not free-space volume ray marching. Glass uses existing
straight-through RGB transmission and Fresnel, not traced refraction/nested media.

`tests/test_studio_secondary_field.py` checks project-IR export and actual ES linking
with/without shadows, and runs an isolated hidden OpenGL pixel probe. The probe checks
all six bins, rotation/direction/coherence, masks, depth visibility, opacity-zero glass,
sorted two-layer attenuation, body scattering, final display resolution, and bitwise
legacy no-field parity. It does not establish production/browser integration by itself.

## Receiver category opt-ins

Subject, Stage, and Room selection belongs to actual geometry, not material IDs.
The Studio host and photo material-engine contract share `STUDIO_SCENE_PART_GROUP`.
Tables and light stands/fixtures are Stage; room surfaces, backdrop/holders, gobos,
and shop lights are Room. Subject geometry is explicit; particle/cage ownership
is unknown and retains ordinary shading. The same material row can appear in any group.

The host sets `uFieldCategoryDisabled` per draw. Its zero/default value preserves
existing hosts; true makes the delivered material mask zero. Disabled opaque draws
retain direct primary shading and contribute nothing in secondary passes 0/1.
Disabled transparent layers still draw in pass 2 with ordinary primary shading.
Resolve runs before category gating. Scalar legacy reception is gated per geometry too.
When all category checkboxes are unchecked, the field pipeline is bypassed entirely.

Mixed meshes retain per-face lineage. The existing global transparent index sort is
unchanged; adjacent triangles with the same category are emitted as contiguous runs
within that exact order. Categories are never rendered as separate whole sorted lists.
Changing receiver categories does not invalidate the scene, rebake rays, or mutate
shared material tables. Category defaults are opt-in/unchecked in the UI.

## Bounded directional ROI ownership

The v5 directional field may cover only the subject ROI even while all scene
transport remains present. Both primary and secondary must use the directional
bounds for `uFieldFromView`, with `uFieldRestrictToBounds=true`. The shared
material-mask helper returns zero outside the inclusive [0,1] cube, preserving
ordinary direct diffuse, specular, transmission and body response outside the ROI.
Otherwise primary suppression would leave selected Stage/Room geometry dark where
no directional samples exist. The guard defaults false for older hosts.

The host passes view/unit/origin to `bindMaterialMask`; the receiver uploads the
same directional ROI transform as `bindSecondary`. Fallback mask binding resets
the guard to false. Scalar legacy field reception keeps its own global bounds.
Shader tests exercise all six exterior faces, inclusive boundaries, interior
replacement, transparent transmission, body response, and the old zero-default path.
