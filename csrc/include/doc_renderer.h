/**
 * doc_renderer.h — Document-hierarchy texture renderer.
 *
 * Architecture:
 *   Producers call dr_submit_node() to drop a (id, rect, payload) triple into
 *   a thread-safe FIFO.  A background worker pops work items, checks whether
 *   re-rendering is actually necessary ("pointlessness filter"), renders each
 *   node into its own RGBA tile, then marks the composite dirty.
 *
 *   dr_composite() blends all live tiles into a flat RGBA output buffer sized
 *   (renderer_height × renderer_width × 4 bytes) ready for GL texture upload
 *   or direct CPU blit.
 *
 * Glyph atlas:
 *   The atlas is opaque to C callers.  Python (or any external code) uploads
 *   pre-rendered glyph RGBA data via dr_load_glyph_atlas_rgba().  The atlas
 *   layout is: 16 columns × N rows, each cell glyph_w × glyph_h pixels,
 *   covering ASCII 32–127 (glyph_index = codepoint − 32).
 *   Until the atlas is loaded, text tiles render without glyphs (background
 *   and geometry only).
 *
 * Primitive atlas:
 *   Pre-baked primitive tiles (filled rect, rounded-rect outline, icon slots)
 *   uploaded via dr_load_primitive_atlas_rgba().  Primitive index stored in
 *   DocNodePayload.icon_id.  Unused when icon_id < 0.
 *
 * Control hierarchy mapping:
 *   Each KnobSpec / Panel node the Python side wants to render becomes a
 *   DocNodePayload with an appropriate DocNodeType.  Parent panels submit
 *   themselves first (lower id), then recurse into children (higher ids).
 *   The compositor respects hierarchy order: ancestors first, then siblings
 *   by explicit sibling order.
 *
 * API surface: extern "C" for ctypes loading from Python.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

/* Export macro: dllexport when building the DLL, dllimport when consuming it,
   plain symbol on non-MSVC (GCC/Clang export everything by default). */
#if defined(_MSC_VER) || defined(__MINGW32__)
#  ifdef BUILDING_DOC_RENDERER_DLL
#    define DR_API __declspec(dllexport)
#  else
#    define DR_API __declspec(dllimport)
#  endif
#else
#  define DR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Node geometry ───────────────────────────────────────────────────────── */

typedef struct {
    int x, y, w, h;   /* pixel rect in the output texture (top-left origin) */
} DocNodeRect;

/* ── Node type ───────────────────────────────────────────────────────────── */

typedef enum {
    DR_NODE_KNOB_FLOAT      = 0,   /* continuous float knob / slider       */
    DR_NODE_KNOB_INT        = 1,   /* integer knob / stepper               */
    DR_NODE_KNOB_ENUM       = 2,   /* enum / choice dropdown               */
    DR_NODE_KNOB_BOOL       = 3,   /* toggle / checkbox                    */
    DR_NODE_PANEL_HEADER    = 4,   /* panel title bar                      */
    DR_NODE_PANEL_BODY      = 5,   /* panel background fill                */
    DR_NODE_TEXT_LABEL      = 6,   /* static label / annotation            */
    DR_NODE_PRIM_RECT       = 7,   /* solid filled rectangle               */
    DR_NODE_PRIM_ROUNDRECT  = 8,   /* rounded-rectangle outline or fill    */
    DR_NODE_PRIM_ICON       = 9,   /* icon tile from primitive atlas       */
} DocNodeType;

/* ── Node payload ────────────────────────────────────────────────────────── */

#define DR_LABEL_MAX  64
#define DR_VALUE_MAX  128

typedef struct {
    DocNodeType type;

    char  label[DR_LABEL_MAX];   /* display label (UTF-8 truncated to ASCII) */
    char  value_str[DR_VALUE_MAX]; /* formatted current value or choice text */

    /* Colours: r,g,b,a each in [0,1] */
    float bg_rgba[4];
    float fg_rgba[4];
    float border_rgba[4];
    float accent_rgba[4];       /* value-bar / highlight colour            */

    float corner_radius;        /* px; 0 = sharp corners                   */
    float font_scale;           /* 1.0 = nominal atlas glyph size          */
    float value_norm;           /* [0,1] normalised knob position          */
    int   icon_id;              /* primitive-atlas slot; <0 = none         */
    int   border_px;            /* border thickness in pixels; 0 = none    */
    int   flags;                /* reserved                                */
} DocNodePayload;

/* ── Opaque renderer handle ──────────────────────────────────────────────── */

typedef struct DocRendererState DocRendererState;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

DocRendererState* dr_create(int width, int height);
void              dr_destroy(DocRendererState* st);

/* ── Atlas upload ────────────────────────────────────────────────────────── */

/* rgba must be (atlas_h × atlas_w × 4) bytes, row-major, RGBA8.
   Each glyph cell is glyph_w × glyph_h.  Atlas is 16 cells wide.
   Covers ASCII 32–127 starting at index 0 (space). */
void dr_load_glyph_atlas_rgba(DocRendererState* st,
                               const uint8_t* rgba,
                               int atlas_w, int atlas_h,
                               int glyph_w, int glyph_h);

/* Primitive atlas: arbitrary RGBA sheet.  Each icon is prim_w × prim_h.
   The sheet is prim_cols wide (in icon units).  icon_id = row*prim_cols+col. */
void dr_load_primitive_atlas_rgba(DocRendererState* st,
                                   const uint8_t* rgba,
                                   int atlas_w, int atlas_h,
                                   int prim_w, int prim_h,
                                   int prim_cols);

/* ── Node submission (FIFO drop-off) ─────────────────────────────────────── */

/* Drop a node into the render FIFO.  Returns immediately (non-blocking).
   If a node with the same id already has an identical payload and has been
   rendered, the item is silently discarded (pointlessness filter).
   Calling with a NULL payload removes the node from the live set. */
void dr_submit_node(DocRendererState* st,
                    uint64_t          node_id,
                    DocNodeRect       rect,
                    const DocNodePayload* payload);

/* Parent-aware submission.  parent_id=0 means root-level.  sibling_order
   controls order within the parent.  Negative sibling_order is treated as 0;
   ties are resolved by node_id for deterministic hierarchy order. */
void dr_submit_node_ex(DocRendererState* st,
                       uint64_t          node_id,
                       uint64_t          parent_id,
                       int               sibling_order,
                       DocNodeRect       rect,
                       const DocNodePayload* payload);

/* Force a node dirty (next submission will re-render regardless of payload). */
void dr_mark_dirty(DocRendererState* st, uint64_t node_id);

/* Remove all nodes and clear composite buffer. */
void dr_clear(DocRendererState* st);

/* ── Synchronous flush ───────────────────────────────────────────────────── */

/* Block until the FIFO is empty and all pending tiles are rendered.
   Safe to call from the main thread before dr_composite(). */
void dr_flush(DocRendererState* st);

/* ── Composite ───────────────────────────────────────────────────────────── */

/* Write the composited RGBA8 image into out_rgba (must be width*height*4 bytes).
   Blends all rendered tiles in hierarchy order (ancestor chain + sibling order).
   Does NOT call dr_flush(); caller decides when to flush. */
void dr_composite(DocRendererState* st, uint8_t* out_rgba);

/* Returns 1 if the composite is dirty since the last dr_composite() call. */
int dr_composite_dirty(DocRendererState* st);

/* ── Introspection ───────────────────────────────────────────────────────── */

int  dr_width (DocRendererState* st);
int  dr_height(DocRendererState* st);

/* Number of nodes currently in the live set. */
int  dr_node_count(DocRendererState* st);

/* Number of work items queued but not yet rendered. */
int  dr_queue_depth(DocRendererState* st);

#ifdef __cplusplus
}
#endif
