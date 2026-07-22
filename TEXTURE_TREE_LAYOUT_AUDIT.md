# 2D Layout to Ray-Traced Window-Element Audit

## Implementation status (2026-07-20)

The audit below records the pre-integration state that motivated this work.
The blocking seams it identified are now addressed in the live program:

- `camera_software/window_elements.py` exports one semantic
  `WindowElementSceneManifest` before draw calls erase hierarchy. It retains
  program panels, nine-slice patches, actions, authored monofont text, live
  scroll-list rows/body text/controls, and the two progress pies with named
  sensor-space rectangles, clipping, state, style references, dependencies,
  transforms, and content signatures.
- `layout_panel_composition_trace()` is now the single nine-slice geometry
  oracle used by both the raster compositor and layout-object work manifest.
  Repeated square patches remain visibly tiled, including clipped partial
  terminal tiles; the old rectangular-sample-lattice reconstruction is gone.
- The semantic window manifest is published as JSON and as a retained OpenUSD
  layer in every layout package. Live scroll rows, body text, and progress pies
  are also instantiated as window-element objects in the ray-scene order, so a
  final holistic exposure contains the two lists rather than photographing
  only the empty hosts.
- `WindowElementHarvestCache` automatically harvests eligible unchanged
  panels/window branches from a completed holistic exposure in both display
  and linear form. Lookup is exact on element content/bounds plus camera,
  lighting, material, reflection-boundary, and color-transform condition, so a
  prior plate becomes reusable if that complete condition returns.
- The render-object library and `HierarchicalObjectResolver` provide the shared
  primitive → glyph/token → parametric panel subtype → scene → holistic static
  texture ladder, with independent alphabet and panel provider sign-offs.

One physical boundary remains intentional: a harvested camera-space plate is
never treated as an ordinary relightable albedo. It is an exact-condition
camera plate; a mismatch descends to constituent scene objects or live dynamic
fallbacks.

The camera seam is likewise explicit. Scene orders now carry canonical,
human-readable camera/lens/sensor/film-stage/flash/spectral manifests plus
facet compatibility keys. Exact camera geometry is cached independently of
lighting, and the optical prescription/spectral payload has its own coarser
cache identity. A previously prepared focus is a direct disk-cache reuse. A
new rear-group focus position still requires geometry/BVH construction because
the current native tracer has no safe BVH-refit API; the compatibility record
must not be interpreted as permission to reuse stale triangles.

## Original audit snapshot

## Intended ownership

The repository's retained 2D UI is the authoring oracle. It resolves hierarchy,
rectangles, clipping, scroll position, z order, control state, labels, and hit
regions. The ray tracer must receive those resolved details as retained window
element style objects. A raster is evidence or a fallback product; it is not the
canonical definition of the window.

The desired resolution ladder is:

1. primitive style objects (panel, border, glyph, icon, gauge, scrollbar),
2. token and control objects,
3. widget and panel assemblies,
4. the complete window scene,
5. a holistic camera-space exposure,
6. condition-indexed crops harvested from that exposure.

Every tier must preserve links to its children and source authoring nodes.

## What exists

### Retained 2D authoring

`bass_viewer.py` already contains most of the correct retained identity model:

- `DirtyUITree` holds parent/child nodes and dirty propagation.
- `UIAtom` holds a stable key, parent key, logical and motion rectangles, kind,
  z index, takeover state, visible rectangle, and optional atom surfaces.
- `Panel.snapshot_gl_atoms()` can split a rendered panel into a background and
  independently movable atom surfaces.
- `PanelItemMap` registers item rectangles as retained atoms.
- `ScrollableSubpanelList` separately retains hierarchical control paths,
  headers, body rectangles, scroll state, selection, and render callbacks.

`camera_software/control_layout.py` is a second, semantic layout path. It turns
the `controls.Panel` tree into `ControlLayoutElement` records with parent keys,
sibling order, routes, values, and rectangles.

### Existing panel rasterizer

`camera_software/layout_panel.py` produces a correct deterministic nine-slice
RGBA image at a requested size. It already clips partial terminal tiles because
Pillow composites the final source tile against the finite target image.

It does not expose a composition trace. After rendering, callers cannot ask
which source patch produced a pixel, which layer owns it, what color encoding
the source used, or which higher-level content was placed over the panel.

### Existing ray-scene bridge

`build_self_rendering_program_scene()` maps manifest host rectangles into
physical planes and text geometry. Panel planes retain object/subtype keys and
nine-slice parameters. `build_display_scene_order()` emits those references in
the ray-tracer scene order.

This bridge currently reconstructs panel border/crop calculations separately
from the 2D compositor. That duplication can drift. One stale path still names
the operation `square_uv_to_rectangular_sample_lattice`, while the work manifest
now correctly specifies repeated square tiles with clipped terminal tiles.

### Existing texture tree

The render library has strong identities for font styles, glyph/token subtypes,
parametric panel subtypes, archived scenes, progressive artifacts, and final
interface exposures. The layout work cache independently resolves a developing
panel still or rack-focus hold pose into a representative-square replacement.

There is not yet one node/artifact protocol shared by glyphs, window elements,
panels, and final interface crops. Selection and invalidation are consequently
implemented in several places.

### Actual live composition order

The displayed window is assembled in this order:

1. nine-slice fallback panels,
2. the available holistic ray texture,
3. cached monofont label/editor rasters when the holistic texture is stale,
4. active work/preview textures,
5. the camera and work `ScrollableSubpanelList` widgets rendered by pygame.

Therefore the current holistic ray render is not yet a holistic UI render. The
scroll lists and their content are painted afterward and are absent from a
harvested ray crop.

## Blocking seams

### 1. No common window-element manifest

`DirtyUITree`, `ScrollableSubpanelList`, and `ControlLayoutDesign` describe
related hierarchies in different shapes. The ray-scene bridge only understands
the last of these plus the top-level program layout.

### 2. Draw calls erase semantics

Once a widget calls `pygame.draw.*`, `font.render()`, and `blit()`, a surface no
longer says which pixels are a title, border, selection state, progress gauge,
or scrollbar. Pixel inspection cannot reliably reconstruct those meanings.

### 3. Coordinate spaces are implicit

The code uses panel-local rectangles, scrolled content rectangles, window
rectangles, absolute sensor rectangles, photographed-crop-local rectangles,
normalized UVs, and world placements. Several conversions are correct, but the
space and transform are not carried with each node.

### 4. Color and alpha contracts differ

Nine-slice panels are display-referred uint8 RGBA. Token composition is linear
RGB, then camera processed. Pygame widget surfaces use display RGB and mixed
alpha/colorkey behavior. Ray results retain both linear and display products.
A bridge must label encoding, alpha mode, and intended material use explicitly.

### 5. Cache identity is split

Panel progress uses cache keys, text uses asset/condition keys, widget redraws
use dirty flags and ad-hoc signatures, and final exposures use render keys. A
single unchanged element cannot yet be followed through every tier.

### 6. A camera crop is not an albedo texture

A crop harvested from a holistic exposure already contains lighting,
reflections, lens response, occlusion, and tone mapping. Applying it to a lit
surface would double those effects. It is valid as a camera-space replacement
plate under an exact matching condition, or as explicitly unlit/emissive
evidence for diagnostics. It is not a reusable physical surface material.

## Required bridge contract

Add a renderer-neutral `WindowElementSceneManifest` produced during 2D layout,
before semantic information is lost to draw calls. Each element needs:

- stable `element_key`, `parent_key`, sibling order, and z index;
- element kind and state subtype;
- layout, clip, visible, and motion rectangles with named coordinate spaces;
- the exact transform chain from local pixels to sensor pixels;
- style object key/subtype and parameter values;
- authored text/icon/value where applicable;
- compositing mode, opacity, color encoding, and alpha mode;
- child keys and dependency keys;
- content revision/signature and dirty region;
- fallback raster reference, when one exists;
- readiness and provenance for every supplied representation.

The ray bridge should consume this manifest without recomputing layout. It
should instantiate one retained scene object per semantic window element and
preserve the same keys in OpenUSD and render-library records.

## Window element style objects

The initial style-object vocabulary should be deliberately small:

- `window.panel` and `window.panel_patch`
- `window.border`
- `window.text_run` and `window.glyph`
- `window.icon`
- `window.button`
- `window.list_header`, `window.list_body`, and `window.list_row`
- `window.scroll_track` and `window.scroll_thumb`
- `window.progress_pie`
- `window.image_host`

State such as selected, disabled, working, developing, and converged belongs in
the subtype/parameter set, not in an unrelated bitmap identity.

## Harvest rules

After a genuinely holistic exposure, harvest every eligible element/panel into
both linear and display crops. Retain all prior crops; never discard one merely
because conditions changed.

A harvested camera-space plate is selectable only when all of these recur:

- element content signature,
- element bounds and clip signature,
- camera/lens/sensor/crop signature,
- lighting and exposure signature,
- material signature,
- surrounding occlusion/reflection-boundary signature,
- output color-transform signature.

Selection should prefer the largest exact cached branch: whole window, then
panel/widget, then token/control, then glyph/primitive. Any mismatch descends
the tree instead of stretching or approximately reusing an incompatible crop.

## Recommended implementation order

1. Define `WindowElementSceneManifest` and coordinate/color contracts.
2. Add exporters for `DirtyUITree`/`UIAtom`, `ScrollableSubpanelList`, and
   `ControlLayoutDesign`; do not alter their layout decisions.
3. Make the existing nine-slice compositor emit the same element/patch trace
   consumed by the ray bridge, removing duplicated border/crop math.
4. Teach the ray scene builder to instantiate window element style objects from
   the manifest and include the two live scroll widgets.
5. Only then enable automatic holistic panel harvesting and resolver selection.
6. Treat harvested crops as camera-space plates and add exact-condition lookup,
   including reuse when a former condition returns.

This order makes the first harvested panel truthful: it will contain the same
widget hierarchy and content that the user actually saw.
