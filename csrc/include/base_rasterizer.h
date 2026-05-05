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

/* ── Scene illumination ──────────────────────────────────────────────────── */

/* All three floats arrays are vec3 (3 elements). */
void br_set_scene(BaseRasterizerState* st,
                  const float* light_v,
                  const float* scene_rgb,
                  float        scene_indirect);

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

/* ── Readback ────────────────────────────────────────────────────────────── */

/* Copy the colour buffer to out_rgba (height * width * 4 uint8, RGBA).
   Applies sRGB gamma correction. */
void br_readback_u8(const BaseRasterizerState* st, uint8_t* out_rgba);

/* Raw linear float readback (height * width * 4 float32, RGBA [0,1]).
   Use when you plan to upload to a GL texture for tone-mapping yourself. */
void br_readback_f32(const BaseRasterizerState* st, float* out_rgba);

/* Query dimensions. */
int br_width (const BaseRasterizerState* st);
int br_height(const BaseRasterizerState* st);

#ifdef __cplusplus
}
#endif
