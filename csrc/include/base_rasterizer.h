/**
 * base_rasterizer.h — Tile-parallel software rasterizer.
 *
 * API surface: extern "C" for ctypes loading from Python.
 *
 * Material layout (float32 arrays) mirrors material_db.py exactly:
 *
 *   PBRBaseRecord  (binding=10 equivalent)  — 16 floats/material
 *     [0..2]  albedo.rgb
 *     [3]     roughness
 *     [4]     metallic
 *     [5]     transmission
 *     [6]     ior
 *     [7]     opacity
 *     [8..10] emission.rgb
 *     [11..15] reserved
 *
 *   PhongRecord    (binding=11 equivalent)  —  8 floats/material
 *     [0]     ambient
 *     [1]     spec_strength
 *     [2]     shininess
 *     [3]     _reserved (grain baked to 0 — no grain texture in base renderer)
 *     [4..6]  inner_color.rgb
 *     [7]     _pad
 *
 *   EnamelRecord   (binding=14 equivalent)  —  8 floats/material
 *     [0]     thickness_nm   (0 = no enamel)
 *     [1]     ior_real
 *     [2]     ior_imag
 *     [3]     roughness
 *     [4..6]  color.rgb
 *     [7]     _pad
 *
 *   TextureStackRecord (cold optional chunk) — 16 floats/material
 *     [0]     emit_uv_layer      (-1 = none)
 *     [1]     color_uv_layer     (-1 = none)
 *     [2]     depth_uv_layer     (-1 = none)
 *     [3]     remit_uv_layer     (-1 = none)
 *     [4]     depth_scale_mm
 *     [5]     thickness_scale_mm
 *     [6]     depth_bias_mm
 *     [7]     thickness_bias_mm
 *     [8]     emit_gain
 *     [9]     color_blend
 *     [10]    direct_lobe_power
 *     [11]    model_flags_or_indices
 *     [12]    remit_gain
 *     [13]    remit_attack
 *     [14]    remit_decay
 *     [15]    translucence_gain
 *
 * Geometry input:
 *   verts_view — (n_tris * 3, 6) float32: [x,y,z, nx,ny,nz] per vertex in
 *                VIEW space (after MV, before projection).
 *   mat_ids    — (n_tris,) int32: per-triangle material index.
 *   mvp        — (4,4) float32 column-major MVP matrix (for projection).
 *
 * Illumination:
 *   light_v      — vec3 light direction in VIEW space (unit).
 *   scene_rgb    — vec3 environment tint.
 *   scene_indirect — float [0,1] indirect fill ratio.
 *
 * Output: RGBA8 buffer (height * width * 4 bytes, row-major).
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque state handle ─────────────────────────────────────────────────── */
typedef struct BaseRasterizerState BaseRasterizerState;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

BaseRasterizerState* br_create(int width, int height, int tile_size);
void                 br_destroy(BaseRasterizerState* st);

/* ── Framebuffer ─────────────────────────────────────────────────────────── */

/* Fill colour buffer + depth buffer.  r,g,b,a in [0,1]. */
void br_clear(BaseRasterizerState* st, float r, float g, float b, float a);

/* ── Material tables ─────────────────────────────────────────────────────── */

/* Each call stores a pointer to the caller-owned array.  Arrays must remain
   valid for the lifetime of any br_render() call.  Pass NULL to disable the
   corresponding chunk (defaults/zeros are used). */
void br_set_pbr_chunk   (BaseRasterizerState* st, const float* data, int n_materials);
void br_set_phong_chunk (BaseRasterizerState* st, const float* data, int n_materials);
void br_set_enamel_chunk(BaseRasterizerState* st, const float* data, int n_materials);
void br_set_texture_stack_chunk(BaseRasterizerState* st, const float* data, int n_materials);

/* Copy an RGBA8 emission texture array into the rasterizer.  Data is tightly
   packed as layers × height × width × 4 bytes. */
void br_set_emit_uv_texture_array(BaseRasterizerState* st,
                                  const uint8_t* data,
                                  int width,
                                  int height,
                                  int layers);
void br_set_color_uv_texture_array(BaseRasterizerState* st,
                                   const uint8_t* data,
                                   int width,
                                   int height,
                                   int layers);
void br_set_depth_uv_texture_array(BaseRasterizerState* st,
                                   const uint8_t* data,
                                   int width,
                                   int height,
                                   int layers);
void br_set_remit_uv_texture_array(BaseRasterizerState* st,
                                   const uint8_t* data,
                                   int width,
                                   int height,
                                   int layers);

/* ── Scene illumination ──────────────────────────────────────────────────── */

/* All three floats arrays are vec3 (3 elements). */
void br_set_scene(BaseRasterizerState* st,
                  const float* light_v,
                  const float* scene_rgb,
                  float        scene_indirect);

/**
 * Push an explicit array of N directional lights.  Each light has its OWN
 * colour and direction — the rasterizer does NOT average them.  This is the
 * preferred path for multi-emitter scenes (the legacy `br_set_scene` builds
 * a single mixed light, which destroys per-emitter colour identity).
 *
 * dirs   : (n_lights * 3) float32 — view-space unit direction per light.
 * colors : (n_lights * 3) float32 — linear sRGB per light (no clamp).
 * intens : (n_lights,)    float32 — scalar gain per light.
 *
 * Pass n_lights = 0 to clear the multi-light array and fall back to the
 * legacy single-light synthesised from `br_set_scene` fields.
 * Maximum supported lights: 8 (excess silently truncated).
 */
void br_set_lights(BaseRasterizerState* st,
                   int          n_lights,
                   const float* dirs,
                   const float* colors,
                   const float* intens);

/* ── Object groups (the ONLY way to declare what emits light) ────────────── */
/*
 * The host application already knows what an "object" is.  Every triangle in
 * a draw call belongs to exactly one such object, has exactly one material,
 * and has a known model-view transform.  The renderer learns about objects
 * via this API and caches one cluster-light per emissive object internally.
 *
 * Per-frame protocol:
 *   The host calls `br_set_groups()` with the ENTIRE current partition of
 *   triangles into groups (the n_groups groups must cover all n_tris in the
 *   subsequent br_render() call).  Each group carries a `dirty` bitmask:
 *
 *     BR_DIRTY_GEOM (1) — triangles in this group changed (positions/topology
 *                          or this group is brand-new).  Engine does a full
 *                          recompute of the cached centroid+area for it.
 *     BR_DIRTY_MV   (2) — only the model-view transform changed.  Engine
 *                          re-transforms the cached centroid via
 *                          mv_new * mv_old.inverse() — a single 4×4 mul.
 *     BR_DIRTY_EMIT (4) — emission of this group's material changed.  Engine
 *                          re-reads pbr.emission for the cached entry.
 *     0             — no recompute.  Cache is reused verbatim.
 *
 *   The cache persists across br_render() calls.  Groups omitted from a
 *   subsequent call are dropped from the cache.
 */

#define BR_DIRTY_GEOM 1
#define BR_DIRTY_MV   2
#define BR_DIRTY_EMIT 4

typedef struct BRGroup {
    int   group_id;        /* host-stable identity                          */
    int   mat_id;          /* material index (PBR/phong/enamel chunks)      */
    int   tri_offset;      /* first triangle in verts_view that is ours     */
    int   tri_count;       /* number of triangles owned                     */
    float mv[16];          /* column-major view transform of group's frame  */
    int   dirty;           /* bitmask of BR_DIRTY_*                         */
} BRGroup;

void br_set_groups(BaseRasterizerState* st,
                   int             n_groups,
                   const BRGroup*  groups);

/* Cap the number of cluster-lights emitted per frame from the group cache.
   Clamped internally to [1, SceneParams::MAX_LIGHTS]. */
void br_set_max_lights(BaseRasterizerState* st, int max_lights);
void br_set_specular_enabled(BaseRasterizerState* st, int enabled);
void br_set_emission_direct_enabled(BaseRasterizerState* st, int enabled);

/* ── Render ──────────────────────────────────────────────────────────────── */

/**
 * Project, bin, and shade n_tris triangles into the framebuffer.
 *
 * verts_view : (n_tris * 3, 6) float32 — [x,y,z,nx,ny,nz] per vertex,
 *              in VIEW space.
 * mat_ids    : (n_tris,) int32 — material index per triangle.
 * mvp        : (16,) float32 — column-major 4×4 projection-only (or MVP)
 *              matrix.  For correct perspective divide supply the full MVP.
 */
void br_render(BaseRasterizerState* st,
               const float* verts_view,
               const int*   mat_ids,
               int          n_tris,
               const float* mvp);

/* Variant whose vertex rows are [x,y,z, nx,ny,nz, u,v]. */
void br_render_textured(BaseRasterizerState* st,
                        const float* verts_view_uv,
                        const int*   mat_ids,
                        int          n_tris,
                        const float* mvp);

/* ── Readback ────────────────────────────────────────────────────────────── */

/* Copy the colour buffer to out_rgba (height * width * 4 uint8, RGBA).
   Applies sRGB gamma correction. */
void br_readback_u8(const BaseRasterizerState* st, uint8_t* out_rgba);

/* Raw linear float readback (height * width * 4 float32, RGBA [0,1]).
   Use when you plan to upload to a GL texture for tone-mapping yourself. */
void br_readback_f32(const BaseRasterizerState* st, float* out_rgba);

/* O(1) readback pointer to internal linear float color buffer.
   Layout is tightly packed [height][width][4] in row-major order.
   Pointer is valid until the rasterizer is destroyed or resized. */
const float* br_readback_f32_ptr(const BaseRasterizerState* st);

/* Query dimensions. */
int br_width (const BaseRasterizerState* st);
int br_height(const BaseRasterizerState* st);

#ifdef __cplusplus
}
#endif
