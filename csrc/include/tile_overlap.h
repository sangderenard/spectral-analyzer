/**
 * tile_overlap.h — Parallel tile-item overlap detection.
 *
 * Algorithm: Separating Axis Theorem (SAT) for exact convex-quad vs AABB
 * overlap per (tile, item) pair, run in parallel across all tiles via the
 * shared Eigen thread pool.
 *
 * For a bounded floor mesh (hundreds of tiles, tens of items) the brute-force
 * O(N_tiles × N_items × 6_axes) dot-product sweep is trivially cheap and
 * avoids any discretisation error from rasterisation.
 *
 * API surface: extern "C" for ctypes loading from Python.
 *
 * Tile corners layout (float32, 8 floats per tile):
 *   [0,1]  corner 0 (x,y)  — inner-left   (world metres)
 *   [2,3]  corner 1 (x,y)  — inner-right
 *   [4,5]  corner 2 (x,y)  — outer-right
 *   [6,7]  corner 3 (x,y)  — outer-left
 *   Any winding order; SAT does not require CCW.
 *
 * Item rect layout (float32, 4 floats per item):
 *   [0]  x    — left edge in world metres
 *   [1]  y    — bottom edge in world metres
 *   [2]  w    — width  (metres, positive)
 *   [3]  h    — height (metres, positive)
 *
 * Output status codes (int8_t):
 *   TO_CLEAR     1  — tile present, no items overlap
 *   TO_OCCUPIED  2  — exactly one item overlaps this tile
 *   TO_COLLISION 3  — two or more items overlap this tile
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Status codes ────────────────────────────────────────────────────────── */
#define TO_CLEAR     1
#define TO_OCCUPIED  2
#define TO_COLLISION 3

/* ── Opaque state handle ─────────────────────────────────────────────────── */
typedef struct TileOverlapState TileOverlapState;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

TileOverlapState* to_create(void);
void              to_destroy(TileOverlapState* st);

/* ── Input: tile polygons ────────────────────────────────────────────────── */

/* corners : flat [n_tiles * 8] float32 — 4 corners × (x,y) per tile.
   ids     : [n_tiles] int32 — caller-stable tile IDs echoed in results.
   Replaces any previously submitted tile list. */
void to_set_tiles(TileOverlapState* st,
                  const float* corners,
                  const int*   ids,
                  int          n_tiles);

/* ── Input: item footprints (AABB) ──────────────────────────────────────── */

/* rects : flat [n_items * 4] float32 — (x, y, w, h) per item in world metres.
   ids   : [n_items] int32 — caller-stable item IDs.
   Replaces any previously submitted item list. */
void to_set_items(TileOverlapState* st,
                  const float* rects,
                  const int*   ids,
                  int          n_items);

/* ── Compute ─────────────────────────────────────────────────────────────── */

/* Run SAT overlap detection in parallel across all tiles.
   out_status : caller-allocated [n_tiles] int8_t  — TO_CLEAR / TO_OCCUPIED / TO_COLLISION.
   out_counts : caller-allocated [n_tiles] int32   — number of items overlapping each tile.
   out_ids    : caller-allocated [n_tiles] int32   — tile IDs in submission order.
   All three arrays must be pre-allocated to n_tiles elements. */
void to_compute(TileOverlapState* st,
                int8_t* out_status,
                int*    out_counts,
                int*    out_ids);

/* ── Query ───────────────────────────────────────────────────────────────── */

int to_n_tiles(const TileOverlapState* st);
int to_n_items(const TileOverlapState* st);

#ifdef __cplusplus
}
#endif
