# Ink-on-slate UI asset architecture

## Production default

Production begins with one deliberately narrow asset family:

- bold DejaVu Sans Mono character or token geometry in fixed-width image cells;
- circular glossy-red extrusion, half submerged in a flat matte-black slate;
- the accepted `glossy_red` and `matte_black` material values from the current
  scene-order examples;
- the existing head-on lab camera and retained camera ring-light rig;
- one 128x128 frame with a centered 80x80 content region, leaving 24 pixels on
  every side for reflected and spilled light.

`build_ink_on_slate_order()` emits ordinary schema-version-1 scene orders. It
does not author camera motion, light motion, a rotating stage, or an action
script. Characters use the accepted 0.2 m height and 0.032 m depth. Token
sequences use the accepted 0.13 m height and 0.028 m depth. Both use the
10-segment circular profile, 0.055 bulge, and 50% embedding.

`plan_ink_atlas_bake()` accepts characters, tokens, and complete token strings.
It checks the persistent catalog and enforces hard production dependencies:
the fixed upper/lowercase alphabet and required punctuation first, exact word
tokens second, and complete UI/editor token strings third. A later tier is
absent from the queue—not merely deprioritized—until the earlier tier
converges. The fixed production result is an
`IMAGE`; light-field or recorded-action products remain permitted by the
general catalog but are not required to get production started.

The first reusable asset boundary lives in
`camera_software/render_assets.py`. It separates asset identity and bake
planning from scene compilation and CPU/GPU exposure execution.

## Ownership

```text
DisplaySceneSpec
  ├─ whole page ───────────────┐
  ├─ constituent object parts ─┼─> BakePlan ─> CPU or GPU executor
  └─ text token assets ────────┘                   │
          │                                        v
          ├─ exact sequence lookup         RenderAssetCatalog
          └─ CharacterAtlas fallback               │
                                                   v
                                        page / part / token reuse
```

`ExtrudedTokenAsset` owns the font source identity, token sequence, physical
layout/extrusion recipe, material identity, and stable cache key. Font identity
includes a digest of the resolved font bytes when available, so a family name
cannot silently alias different installed font data. Its
`scene_order_object()` method is the compatibility adapter used by the current
scene-order compiler.

The optional `RotatingStageSpec` expands control values into explicit
`LightFieldCondition` values. A condition includes view azimuth/elevation,
light rig, material variant, and frame. These values participate in every
artifact lookup; changing a lighting or material condition cannot accidentally
reuse an incompatible field. Its default is one fixed head-on camera-ring
condition; rotation is opt-in.

`plan_display_scene_bake()` creates independent requests for:

- the complete page;
- every enabled constituent object;
- exact token sequences;
- unique character-atlas entries needed while an exact sequence is absent.

Requests are content-addressed and deduplicated. Completed requests are
committed through `RenderAssetCatalog.complete()`. The catalog is an atomic,
versioned JSON manifest and records linear evidence separately from preview and
composition paths.

## Rich object library

`RenderAssetCatalog` remains the low-level evidence and resume index. The
`RenderObjectLibrary` is organized around reusable visual styles, rather than
making every glyph a separate library object:

```text
style object
  subtype sets
    glyph
      "A", "B", ... (partial sets are valid)
    token
      "CAMERA", "Actual light", ...
  each subtype
    condition (view/light/material/frame)
      scene + image + sprite + light-field/resume evidence
```

A style identity contains the font source identity, material recipe/revision,
and stage family. Text content is a subtype. Consequently one DejaVu Sans Mono
red-ink object contains the currently completed alphabet and can steadily gain
punctuation, other Unicode glyphs, words, and whole sequences without pretending
that the set is complete.

Every subtype archives the exact `scene_order.json` and digest; condition,
camera, stage, lighting, material, exposure, geometry, and font specifications;
the reusable scene object template; renderer summary/context/log; and every
cached component in the render directory, including native sensor and resumable
epoch files.

All evidence lives canonically below:

```text
render_objects/<style>/subtypes/<glyph-or-token>/<subtype>/conditions/<condition>/
```

New jobs render directly there. Migration copies every old atlas component into
that hierarchy, validates it, and repoints `RenderAssetCatalog`; old paths are
migration inputs and are not part of the object API.

The library resolves `scene`, `light_field`, and `image` views for a selected
subtype. `compose_sprite()` chooses exact whole-token sprites and then glyph
fallbacks. `compose_scene()` makes the same choice but returns scene-order 3-D
objects, with placement offsets, ready to insert into a subsequent ray trace.
Missing token and glyph subtype requests remain explicit in either composition.

Missing historical stage details use the single fixed default condition and are
marked as inferred. Unknown camera-simulator internals are not presented as
exact; the resolved renderer summary remains attached as surviving provenance.

### Terminal interface after-render

The bakery has a final tier above reusable objects and composed scenes:

```text
interface assembly
  revision history
    after-render
      exact whole-interface scene
      final shared-camera image and linear sensor evidence
      diagnostics, priority map, log, and composition manifest
      references to every known style subtype used by each UI scene object
```

A foreground render is registered only after all UI elements have been placed
in the shared 3-D scene and the camera exposure has completed. Its canonical
location is
`render_interfaces/<interface>/after_renders/<revision-render>/`. The original
working revision is not the browser identity; `select_interface_after_render()`
returns the latest terminal render by default or a specified historical render.
This is the authoritative “what the complete interface looked like after ray
tracing” tier, while component styles and subtype scenes remain available for
editing and subsequent traces.

## Atlas behavior

Sequence glyphs win when present. If a sequence is not ready, `CharacterAtlas`
returns cached characters in the same font, extrusion, material, and
light-field condition. Missing unique characters become atlas bake requests.
Whitespace needs no rendered cell. This permits immediately changing text from
cached pieces while the physically superior whole sequence and page renders
continue in the background.

The atlas is not a 2-D font rasterizer. Its entries are rendered products of
the same extruded 3-D asset contract as complete token sequences.

## Background-independent sprite convention

Completed padded atlas captures also produce `raytraced_sprite.npz`. The
sprite is not represented by ordinary straight RGBA, because a glossy glyph
can brighten or shadow slate pixels outside its geometric silhouette. It uses:

```text
output = premultiplied_rgb
       + (1 - alpha) * destination
       + signed_additive_rgb
```

`alpha` comes from the exact authored font outline and extrusion silhouette,
not from recognizing red pixels or treating black as transparent. A smooth
source-background field is fitted outside a dilated silhouette. The remaining
signed residual records neighboring reflection, glow, chromatic spill, and
shadow. The layers reconstruct the source capture without thresholding faint
effects.

This is preferable to a special “low-alpha background material.” Alpha is not
a physical spectral material property, and making the slate transparent during
transport would remove the very reflected/spilled light the padded capture is
meant to preserve. The geometric matte plus signed light layer works with red
ink today and other foreground materials later.

`CachedTokenStringComposer` looks for an exact cached string, then exact word
tokens, then individual character sprites. It reports missing tokens and
characters while composing everything currently available. The live editor
uses that result as a temporary editor-region replacement until the exact
full-page revision is ready. Existing catalog records with linear evidence but
without a sprite are upgraded lazily without rerunning transport.

## Execution boundary

The plan does not choose CPU or GPU. An executor consumes `BakeRequest`,
applies its `LightFieldCondition` to the physical camera/stage/light rig,
compiles selected scene objects through `scene_orders.py`, and runs the
existing exposure path. On success it calls `RenderAssetCatalog.complete()`.
This keeps cache keys, scheduling, and UI controls identical across CPU and GPU
implementations.

The live demo now authors text through `ExtrudedTokenAsset` and uses one render
owner for every production tier. Startup gives the renderer to the fixed-width
alphabet cells. It proceeds to word tokens, then the editor and fixed UI token
strings (`CAMERA`, `WORK VALUE`, and `SPECTRAL EXPOSURE ACTIVE`). The total UI
scene is not submitted until those string orders converge.
`InkAtlasSubprocessRenderer` runs that request through the unchanged
scene-order renderer. Each prepared process executes 64 one-submission epochs
by default. Every epoch selects up to 1,024 sensor nodes, publishes the current
linear image to the work panel, and atomically checkpoints its sensor sum and
exposure weight. The total burst retains the same 64-submission ray budget,
while font, scene, lens, pipeline, and upload preparation remain amortized by
the single prepared process. If interrupted, the next process restores the
latest compatible epoch and executes only the unfinished portion of that
burst. After the burst it overwrites the developing linear frame, sprite,
manifest, and catalog record.
Covered developing sprites may be composed immediately, while they remain
queued least-refined-first until the body and neighboring-light region stays
within the convergence tolerances across one complete image-to-image
comparison. That comparison spans two substantial render bursts. The catalog
keeps the accumulated sensor sum, exposure weight, and refinement pass after
completion. A later asset browser can opt an object back into work by passing
an absolute `refinement_targets[asset_key]` value (normally its current pass
plus the user's requested number of passes); ordinary planning continues to
leave completed objects alone.
