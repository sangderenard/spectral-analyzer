# UV-baked emission/depth/remission texture stack — action plan

> Captured from design discussion.  Goal: lift the flat per-material emission
> contract to a per-texel UV contract **without** breaking the existing
> prebake-only / no-runtime-spectral / no-feedback-solve invariants.

## Why flat emission fails

1. **Spatial collapse.** Whole triangle group reads one scalar emission.
   Real diffusers want radiance that peaks where the source sits and falls
   off with cos·(1/r²).  Per-mat emission can't ramp across one face.
2. **Tonal collapse.** Saturated RGB hitting a non-tone-mapped sensor model
   is hard-clip, not activation.  Real receptors compress (log/sigmoid),
   then opponent-stage contrast-stretch.  Without that every emitter looks
   like the same paint chip.

A UV-baked 4-channel emission texture solves (1) cheaply.  (2) is a pixel-
shader fix that's mostly orthogonal but the texture stack feeds it the
right *spatial* inputs.

## Channel layout — single RGBA8 fetch

| Chan | Name             | Meaning                                          |
|------|------------------|--------------------------------------------------|
| R    | `direct`         | gain on directional/specular-coupled lobe        |
| G    | `diffusion`      | gain on Lambertian/diffuse lobe                  |
| B    | `saturation_adj` | post-mix chroma multiplier (0.5 = identity)      |
| A    | `dim`            | pre-multiply on whole emission (per-texel mask)  |

`R + G ≤ 1` is **NOT** required — both are independent gains.  An LED dot
with fresnel rim wants both ~0.7.

## Hot-loop fragment math (5 lines)

```glsl
vec4 e = texture(emit_uv, uv);                        // 1 fetch
vec3 emit    = mat.emit_rgb * e.a;                    // dim
vec3 direct  = emit * e.r * spec_lobe(N, V, L);       // direct couple
vec3 diffuse = emit * e.g;                            // lambertian
vec3 mixed   = direct + diffuse;
vec3 chroma  = mix(vec3(luminance(mixed)), mixed,
                   0.5 + e.b);                        // saturation
out_emit = chroma;
```

Cost: 1 fetch, 1 dot, 1 mix, 2 muls.  Negligible.

## Branch-free default

Compile-time `#define HAS_EMIT_UV` permutation, OR bind a 1×1 default
texture `(R=0, G=0, B=0.5, A=1.0)` so the math runs unchanged and
reproduces current behavior.  Prefer the permutation for hot paths.

## Depth: thickness, not parallax

Avoid POM/parallax (TBN, derivatives, rabbit hole).  Single 8-bit channel
= "depth behind surface in mm".  Use only as Beer-Lambert attenuator on
`direct`:

```glsl
float d = texture(depth_uv, uv).r * mat.depth_scale_mm;
direct *= exp(-mat.k_extinction * d);
```

One fetch, one exp.  Yields "bulb recessed in diffuser" without UV reproj.
True parallax stays gated behind a separate `#define HAS_PARALLAX`.

## Remission as UV mask, NOT per-texel FIR

Per-texel temporal state would be `W·H·K bytes per remissive material` —
unacceptable.  Keep the FIR per-material (already implemented in
`RemissionFeedbackEngine`); add a `remit_mask` UV texture that gates
**where on the surface** the per-mat emission lands:

```glsl
out_emit += remit_color * fir_output * texture(remit_mask, uv).a;
```

One fetch, one mul.  Gives "ruby afterglow concentrated in the cabochon
center" with zero per-texel temporal state.

## Triangle → texture binding

**Choice: 2D texture array, one layer per material.**

- Triangle → mat_id → array layer.
- One indirection, no branch.
- All emission/depth/remit_mask atlases parallel-allocated, sized to the
  largest UV map; smaller maps live in a corner with opaque-default fill.
- Preferred over single-atlas + uv_scale/offset (mip-bleed at boundaries).

## mat16 → mat24 expansion

8 new floats; all sentinel-able (`-1` = none); all optional:

| Idx | Field                 | Notes                               |
|-----|-----------------------|-------------------------------------|
| 16  | `emit_uv_layer`       | array layer (-1 = none)             |
| 17  | `remit_mask_layer`    | array layer (-1 = none)             |
| 18  | `depth_layer`         | array layer (-1 = none)             |
| 19  | `depth_scale_mm`      | Beer-Lambert depth scale            |
| 20  | `direct_lobe_power`   | spec exponent for `e.r` path        |
| 21  | `saturation_neutral`  | typically 0.5; per-mat override     |
| 22  | _pad                  |                                     |
| 23  | _pad                  |                                     |

Shader permutation key = `(has_emit_uv, has_remit_mask, has_depth)` =
**8 variants**.  Cheap to ship, easy to debug.

## Hard invariant (do NOT break)

Per-texel runtime spectral resolution is forbidden.  Whatever the UV
authoring pipeline produces must be **pre-baked sRGB** (or pre-baked
spectrum-coefficient triples if we ever go there).  CIE integration
happens once, offline.  Hot loop never touches `EmissionProfileDatabase`.

## Build sequence (each stage independently shippable & revertible)

1. **Plumbing.** Add 1×1 default texture-array binding to both C and GL
   renderers.  No shader logic change.  Proves binding works.
2. **Emission UV.** Add `HAS_EMIT_UV` permutation + 5-line fragment math.
   Test on one material with a hand-painted UV.
3. **Depth-as-thickness.** Add `HAS_DEPTH_UV` permutation + Beer-Lambert
   factor.  Single fetch + exp.
4. **Remission mask.** Add `HAS_REMIT_MASK` permutation; wire to existing
   `RemissionFeedbackEngine` output.
5. **(Optional later.)** True parallax / POM behind `HAS_PARALLAX`.

## Not in scope (parking lot)

- True POM / relief mapping.
- Per-texel emission temporal state.
- Per-fragment spectral resolution.
- Anisotropic UV filtering tweaks (use trilinear default).
