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
scene-order renderer. Each prepared process executes one convergence epoch by
default, but that epoch selects up to 1,024 sensor nodes for each of 64
recursive refinement submissions. This is roughly 64 times the former
per-epoch sensor-ray work (128 nodes × 8 submissions), amortizing font, scene,
lens, pipeline, and upload preparation without changing the meaning of an
epoch. It adds that epoch to the asset's restored sensor sum and weight, then
overwrites its developing linear frame, sprite, manifest, and catalog record.
Covered developing sprites may be composed immediately, while they remain
queued least-refined-first until the body and neighboring-light region stays
within the convergence tolerances across successive epochs.
