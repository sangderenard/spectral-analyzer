/**
 * doc_renderer.cpp — Document-hierarchy texture renderer (C backend).
 *
 * Design:
 *   - One background worker thread per DocRendererState.
 *   - Thread-safe FIFO: std::deque<DocWorkItem> protected by mutex + condvar.
 *   - Each node stores its last-rendered payload hash; if a new submission
 *     hashes identically the item is dropped before queuing ("pointlessness
 *     filter").
 *   - Worker renders tiles into per-node RGBA8 buffers.
 *   - dr_composite() alpha-blends all live tiles into the output buffer using
 *     hierarchical painter order (ancestor chain + sibling order).
 *   - Glyph atlas and primitive atlas are plain RGBA8 sheets loaded at
 *     runtime by the Python wrapper.
 */

#include "doc_renderer.h"

#include <Eigen/Core>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static inline uint8_t f2u8(float v) {
    int i = static_cast<int>(v * 255.0f + 0.5f);
    return static_cast<uint8_t>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

static inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

/* FNV-1a 64-bit hash over arbitrary bytes */
static uint64_t fnv1a(const uint8_t* data, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; ++i)
        h = (h ^ data[i]) * 1099511628211ULL;
    return h;
}

/* ── Per-node live record ─────────────────────────────────────────────────── */

struct DocNode {
    uint64_t         id          = 0;
    DocNodeRect      rect        = {};
    DocNodePayload   payload     = {};
    uint64_t         payload_hash = 0;     /* hash of last-rendered payload */
    std::vector<uint8_t> tile;             /* RGBA8, rect.w × rect.h × 4    */
    uint64_t         parent_id    = 0;     /* 0 = root-level node            */
    int              sibling_order = -1;   /* order inside parent; <0 = 0    */
    bool             rendered    = false;
    bool             dirty       = false;
};

/* ── Work item ───────────────────────────────────────────────────────────── */

struct DocWorkItem {
    uint64_t       node_id;
    uint64_t       parent_id;
    int            sibling_order;
    uint64_t       expected_hash; /* hash at submission time; stale = skip  */
    DocNodeRect    rect;
    DocNodePayload payload;
};

/* ── Glyph atlas state ────────────────────────────────────────────────────── */

struct GlyphAtlas {
    std::vector<uint8_t> rgba;
    int atlas_w  = 0, atlas_h  = 0;
    int glyph_w  = 8, glyph_h  = 8;
    bool loaded  = false;

    /* Sample glyph pixel (r,g,b,a) for ASCII codepoint c at pixel (px,py).
       Returns {0,0,0,0} if atlas not loaded or out of range. */
    void sample(int c, int px, int py, uint8_t out[4]) const {
        if (!loaded || px < 0 || py < 0 || px >= glyph_w || py >= glyph_h) {
            out[0] = out[1] = out[2] = out[3] = 0;
            return;
        }
        int idx = c - 32;
        if (idx < 0 || idx >= 96) { out[0]=out[1]=out[2]=out[3]=0; return; }
        int col = idx % 16;
        int row = idx / 16;
        int ax  = col * glyph_w + px;
        int ay  = row * glyph_h + py;
        if (ax >= atlas_w || ay >= atlas_h) { out[0]=out[1]=out[2]=out[3]=0; return; }
        const uint8_t* p = rgba.data() + (ay * atlas_w + ax) * 4;
        out[0]=p[0]; out[1]=p[1]; out[2]=p[2]; out[3]=p[3];
    }
};

/* ── Primitive atlas state ───────────────────────────────────────────────── */

struct PrimAtlas {
    std::vector<uint8_t> rgba;
    int atlas_w = 0, atlas_h = 0;
    int prim_w  = 0, prim_h  = 0;
    int prim_cols = 1;
    bool loaded  = false;

    void sample(int icon_id, int px, int py, uint8_t out[4]) const {
        if (!loaded || icon_id < 0 || px < 0 || py < 0 ||
            px >= prim_w || py >= prim_h) {
            out[0]=out[1]=out[2]=out[3]=0; return;
        }
        int col = icon_id % prim_cols;
        int row = icon_id / prim_cols;
        int ax  = col * prim_w + px;
        int ay  = row * prim_h + py;
        if (ax >= atlas_w || ay >= atlas_h) { out[0]=out[1]=out[2]=out[3]=0; return; }
        const uint8_t* p = rgba.data() + (ay * atlas_w + ax) * 4;
        out[0]=p[0]; out[1]=p[1]; out[2]=p[2]; out[3]=p[3];
    }
};

/* ── Main state ──────────────────────────────────────────────────────────── */

struct DocRendererState {
    int  width, height;

    /* Node map: id → DocNode  (access under nodes_mtx) */
    std::unordered_map<uint64_t, DocNode> nodes;
    std::vector<uint64_t> order;   /* live node ids for compositing          */
    std::mutex nodes_mtx;

    /* Atlases (access under atlas_mtx) */
    GlyphAtlas glyph_atlas;
    PrimAtlas  prim_atlas;
    std::mutex atlas_mtx;

    /* FIFO (access under fifo_mtx) */
    std::deque<DocWorkItem> fifo;
    std::mutex              fifo_mtx;
    std::condition_variable fifo_cv;
    std::atomic<bool>       running{true};

    /* Composite dirty flag */
    std::atomic<bool> composite_dirty{false};

    /* Worker thread */
    std::thread worker;

    DocRendererState(int w, int h) : width(w), height(h) {
        worker = std::thread([this]{ _worker_loop(); });
    }

    ~DocRendererState() {
        running.store(false);
        fifo_cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    /* ── Tile renderer ─────────────────────────────────────────────────── */

    /* Write text into a tile buffer starting at (tx, ty).
       Returns x position after last glyph. */
    int _draw_text(uint8_t* tile, int tw, int th,
                   int tx, int ty,
                   const char* text,
                   const uint8_t fg[4],
                   float font_scale,
                   const GlyphAtlas& ga) const {
        if (!ga.loaded || !text) return tx;
        int gw = static_cast<int>(ga.glyph_w * font_scale + 0.5f);
        int gh = static_cast<int>(ga.glyph_h * font_scale + 0.5f);
        if (gw < 1) gw = 1;
        if (gh < 1) gh = 1;

        for (const char* p = text; *p; ++p) {
            int c = static_cast<unsigned char>(*p);
            if (c < 32 || c > 127) c = 32;
            /* Sample each pixel of the scaled glyph */
            for (int gy = 0; gy < gh && (ty + gy) < th; ++gy) {
                for (int gx = 0; gx < gw && (tx + gx) < tw; ++gx) {
                    if (tx + gx < 0) continue;
                    int src_px = static_cast<int>(gx * ga.glyph_w / (float)gw);
                    int src_py = static_cast<int>(gy * ga.glyph_h / (float)gh);
                    uint8_t gs[4];
                    ga.sample(c, src_px, src_py, gs);
                    if (gs[3] == 0) continue;
                    /* Modulate glyph colour by fg */
                    float a  = gs[3] / 255.0f;
                    float ga_a_fg = a * (fg[3] / 255.0f);
                    uint8_t* dst = tile + ((ty + gy) * tw + (tx + gx)) * 4;
                    float da = dst[3] / 255.0f;
                    float oa = ga_a_fg + da * (1.0f - ga_a_fg);
                    if (oa > 0.0f) {
                        dst[0] = f2u8(((fg[0]/255.0f)*ga_a_fg + (dst[0]/255.0f)*da*(1.0f-ga_a_fg))/oa);
                        dst[1] = f2u8(((fg[1]/255.0f)*ga_a_fg + (dst[1]/255.0f)*da*(1.0f-ga_a_fg))/oa);
                        dst[2] = f2u8(((fg[2]/255.0f)*ga_a_fg + (dst[2]/255.0f)*da*(1.0f-ga_a_fg))/oa);
                        dst[3] = f2u8(oa);
                    }
                }
            }
            tx += gw + 1;
        }
        return tx;
    }

    /* Fill a rectangle within the tile with a solid colour. */
    static void _fill_rect(uint8_t* tile, int tw, int th,
                            int rx, int ry, int rw, int rh,
                            const uint8_t col[4]) {
        for (int y = ry; y < ry + rh && y < th; ++y) {
            if (y < 0) continue;
            for (int x = rx; x < rx + rw && x < tw; ++x) {
                if (x < 0) continue;
                uint8_t* d = tile + (y * tw + x) * 4;
                d[0] = col[0]; d[1] = col[1]; d[2] = col[2]; d[3] = col[3];
            }
        }
    }

    /* Draw a 1-px border rect within the tile. */
    static void _stroke_rect(uint8_t* tile, int tw, int th,
                              int rx, int ry, int rw, int rh,
                              int thickness,
                              const uint8_t col[4]) {
        for (int t = 0; t < thickness; ++t) {
            int x0=rx+t, y0=ry+t, x1=rx+rw-1-t, y1=ry+rh-1-t;
            for (int x=x0; x<=x1; ++x) {
                if (y0>=0&&y0<th&&x>=0&&x<tw) { uint8_t*d=tile+(y0*tw+x)*4; d[0]=col[0];d[1]=col[1];d[2]=col[2];d[3]=col[3]; }
                if (y1>=0&&y1<th&&x>=0&&x<tw) { uint8_t*d=tile+(y1*tw+x)*4; d[0]=col[0];d[1]=col[1];d[2]=col[2];d[3]=col[3]; }
            }
            for (int y=y0; y<=y1; ++y) {
                if (y<0||y>=th) continue;
                if (x0>=0&&x0<tw) { uint8_t*d=tile+(y*tw+x0)*4; d[0]=col[0];d[1]=col[1];d[2]=col[2];d[3]=col[3]; }
                if (x1>=0&&x1<tw) { uint8_t*d=tile+(y*tw+x1)*4; d[0]=col[0];d[1]=col[1];d[2]=col[2];d[3]=col[3]; }
            }
        }
    }

    /* Render a single node tile.  Must hold atlas_mtx (shared-read) before
       calling, OR call with copies of the atlas already captured. */
    void _render_tile(DocNode& node) {
        const int tw = node.rect.w;
        const int th = node.rect.h;
        node.tile.assign(static_cast<size_t>(tw * th * 4), 0);
        uint8_t* tile = node.tile.data();
        const DocNodePayload& p = node.payload;

        uint8_t bg[4]     = { f2u8(p.bg_rgba[0]),     f2u8(p.bg_rgba[1]),     f2u8(p.bg_rgba[2]),     f2u8(p.bg_rgba[3])     };
        uint8_t fg[4]     = { f2u8(p.fg_rgba[0]),     f2u8(p.fg_rgba[1]),     f2u8(p.fg_rgba[2]),     f2u8(p.fg_rgba[3])     };
        uint8_t border[4] = { f2u8(p.border_rgba[0]), f2u8(p.border_rgba[1]), f2u8(p.border_rgba[2]), f2u8(p.border_rgba[3]) };
        uint8_t accent[4] = { f2u8(p.accent_rgba[0]), f2u8(p.accent_rgba[1]), f2u8(p.accent_rgba[2]), f2u8(p.accent_rgba[3]) };

        /* ── Background fill ── */
        _fill_rect(tile, tw, th, 0, 0, tw, th, bg);

        /* ── Border ── */
        int bp = p.border_px > 0 ? p.border_px : 0;
        if (bp > 0)
            _stroke_rect(tile, tw, th, 0, 0, tw, th, bp, border);

        /* Usable inner area after border */
        int ix = bp + 2, iy = bp + 2;
        int iw = tw - bp*2 - 4, ih = th - bp*2 - 4;
        if (iw < 1 || ih < 1) { node.rendered = true; return; }

        /* ── Acquire atlases ── */
        GlyphAtlas ga_copy;
        PrimAtlas  pa_copy;
        {
            std::lock_guard<std::mutex> lk(atlas_mtx);
            ga_copy = glyph_atlas;
            pa_copy = prim_atlas;
        }

        const float fs = p.font_scale > 0.0f ? p.font_scale : 1.0f;
        const int   gh = ga_copy.loaded ? static_cast<int>(ga_copy.glyph_h * fs + 0.5f) : 8;

        switch (p.type) {

        case DR_NODE_PANEL_BODY:
            /* Already filled bg; nothing more */
            break;

        case DR_NODE_PANEL_HEADER: {
            /* Slightly brighter header bar at top */
            uint8_t hdr[4] = {
                static_cast<uint8_t>(std::min(255, (int)bg[0] + 30)),
                static_cast<uint8_t>(std::min(255, (int)bg[1] + 30)),
                static_cast<uint8_t>(std::min(255, (int)bg[2] + 30)),
                bg[3]
            };
            int bar_h = std::min(ih, gh + 4);
            _fill_rect(tile, tw, th, ix, iy, iw, bar_h, hdr);
            _draw_text(tile, tw, th, ix + 2, iy + 2, p.label, fg, fs, ga_copy);
            break;
        }

        case DR_NODE_TEXT_LABEL:
            _draw_text(tile, tw, th, ix, iy, p.label, fg, fs, ga_copy);
            break;

        case DR_NODE_KNOB_FLOAT:
        case DR_NODE_KNOB_INT: {
            /* Label at top */
            int ty2 = iy;
            _draw_text(tile, tw, th, ix, ty2, p.label, fg, fs, ga_copy);
            ty2 += gh + 2;

            /* Value bar */
            int bar_h    = std::max(3, ih / 5);
            int bar_full = static_cast<int>(iw * clamp01(p.value_norm) + 0.5f);
            uint8_t track[4] = { static_cast<uint8_t>(bg[0]/2), static_cast<uint8_t>(bg[1]/2), static_cast<uint8_t>(bg[2]/2), 220 };
            _fill_rect(tile, tw, th, ix, ty2, iw, bar_h, track);
            if (bar_full > 0)
                _fill_rect(tile, tw, th, ix, ty2, bar_full, bar_h, accent);

            /* Value text below bar */
            ty2 += bar_h + 2;
            _draw_text(tile, tw, th, ix, ty2, p.value_str, fg, fs * 0.85f, ga_copy);
            break;
        }

        case DR_NODE_KNOB_BOOL: {
            /* Checkbox square + label */
            int box = std::min(ih, gh + 2);
            uint8_t box_col[4];
            if (p.value_norm > 0.5f) {
                box_col[0]=accent[0]; box_col[1]=accent[1]; box_col[2]=accent[2]; box_col[3]=accent[3];
            } else {
                box_col[0]=static_cast<uint8_t>(bg[0]/2);
                box_col[1]=static_cast<uint8_t>(bg[1]/2);
                box_col[2]=static_cast<uint8_t>(bg[2]/2);
                box_col[3]=200;
            }
            _fill_rect(tile, tw, th, ix, iy, box, box, box_col);
            _stroke_rect(tile, tw, th, ix, iy, box, box, 1, border);
            _draw_text(tile, tw, th, ix + box + 3, iy + 1, p.label, fg, fs, ga_copy);
            break;
        }

        case DR_NODE_KNOB_ENUM: {
            /* Label + value text in a pill */
            _draw_text(tile, tw, th, ix, iy, p.label, fg, fs, ga_copy);
            int ty2 = iy + gh + 2;
            uint8_t pill[4] = { static_cast<uint8_t>(std::min(255,(int)bg[0]+40)),
                                 static_cast<uint8_t>(std::min(255,(int)bg[1]+40)),
                                 static_cast<uint8_t>(std::min(255,(int)bg[2]+40)),
                                 220 };
            _fill_rect(tile, tw, th, ix, ty2, iw, gh + 4, pill);
            _draw_text(tile, tw, th, ix + 2, ty2 + 2, p.value_str, fg, fs, ga_copy);
            break;
        }

        case DR_NODE_PRIM_RECT:
            _fill_rect(tile, tw, th, ix, iy, iw, ih, accent);
            break;

        case DR_NODE_PRIM_ROUNDRECT:
            /* Filled with accent; simple approximation (no actual rounding in sw) */
            _fill_rect(tile, tw, th, ix, iy, iw, ih, accent);
            if (bp > 0) _stroke_rect(tile, tw, th, ix, iy, iw, ih, 1, border);
            break;

        case DR_NODE_PRIM_ICON:
            if (pa_copy.loaded && p.icon_id >= 0) {
                /* Blit scaled icon */
                for (int py2 = 0; py2 < ih && py2 < pa_copy.prim_h; ++py2) {
                    for (int px2 = 0; px2 < iw && px2 < pa_copy.prim_w; ++px2) {
                        uint8_t ps[4];
                        pa_copy.sample(p.icon_id, px2, py2, ps);
                        if (ps[3] == 0) continue;
                        uint8_t* d = tile + ((iy+py2)*tw + (ix+px2))*4;
                        float sa = ps[3]/255.0f, da2 = d[3]/255.0f;
                        float oa = sa + da2*(1.0f-sa);
                        if (oa > 0.0f) {
                            d[0]=f2u8((ps[0]/255.0f*sa + d[0]/255.0f*da2*(1.0f-sa))/oa);
                            d[1]=f2u8((ps[1]/255.0f*sa + d[1]/255.0f*da2*(1.0f-sa))/oa);
                            d[2]=f2u8((ps[2]/255.0f*sa + d[2]/255.0f*da2*(1.0f-sa))/oa);
                            d[3]=f2u8(oa);
                        }
                    }
                }
            }
            break;
        }

        node.rendered = true;
    }

    /* ── Worker loop ───────────────────────────────────────────────────── */

    void _worker_loop() {
        while (running.load()) {
            DocWorkItem item;
            {
                std::unique_lock<std::mutex> lk(fifo_mtx);
                fifo_cv.wait(lk, [this]{ return !fifo.empty() || !running.load(); });
                if (!running.load() && fifo.empty()) break;
                if (fifo.empty()) continue;
                item = fifo.front();
                fifo.pop_front();
            }

            /* Pointlessness check: is this work item stale? */
            {
                std::lock_guard<std::mutex> lk(nodes_mtx);
                auto it = nodes.find(item.node_id);
                if (it == nodes.end()) continue;          /* node removed   */
                DocNode& node = it->second;
                if (node.payload_hash != item.expected_hash && !node.dirty)
                    continue;                              /* superseded     */
                if (node.rendered && !node.dirty &&
                    node.payload_hash == item.expected_hash)
                    continue;                              /* already done   */

                /* Copy payload into node and render */
                node.rect    = item.rect;
                node.payload = item.payload;
                node.parent_id = item.parent_id;
                node.sibling_order = item.sibling_order;
                node.dirty   = false;

                _render_tile(node);
            }

            composite_dirty.store(true);
        }
    }
};

/* ── extern "C" implementation ───────────────────────────────────────────── */

extern "C" {

DocRendererState* dr_create(int width, int height) {
    return new DocRendererState(width, height);
}

void dr_destroy(DocRendererState* st) {
    delete st;
}

void dr_load_glyph_atlas_rgba(DocRendererState* st,
                               const uint8_t* rgba,
                               int atlas_w, int atlas_h,
                               int glyph_w, int glyph_h) {
    std::lock_guard<std::mutex> lk(st->atlas_mtx);
    GlyphAtlas& ga = st->glyph_atlas;
    ga.atlas_w = atlas_w;  ga.atlas_h = atlas_h;
    ga.glyph_w = glyph_w;  ga.glyph_h = glyph_h;
    ga.rgba.assign(rgba, rgba + static_cast<size_t>(atlas_w * atlas_h * 4));
    ga.loaded = true;
}

void dr_load_primitive_atlas_rgba(DocRendererState* st,
                                   const uint8_t* rgba,
                                   int atlas_w, int atlas_h,
                                   int prim_w, int prim_h,
                                   int prim_cols) {
    std::lock_guard<std::mutex> lk(st->atlas_mtx);
    PrimAtlas& pa = st->prim_atlas;
    pa.atlas_w = atlas_w;  pa.atlas_h = atlas_h;
    pa.prim_w  = prim_w;   pa.prim_h  = prim_h;
    pa.prim_cols = prim_cols;
    pa.rgba.assign(rgba, rgba + static_cast<size_t>(atlas_w * atlas_h * 4));
    pa.loaded = true;
}

void dr_submit_node(DocRendererState* st,
                    uint64_t          node_id,
                    DocNodeRect       rect,
                    const DocNodePayload* payload) {
    dr_submit_node_ex(st, node_id, 0, -1, rect, payload);
}

void dr_submit_node_ex(DocRendererState* st,
                       uint64_t          node_id,
                       uint64_t          parent_id,
                       int               sibling_order,
                       DocNodeRect       rect,
                       const DocNodePayload* payload) {
    if (!payload) {
        /* Removal request */
        std::lock_guard<std::mutex> lk(st->nodes_mtx);
        auto it = st->nodes.find(node_id);
        if (it != st->nodes.end()) {
            /* Remove from order list */
            auto& ord = st->order;
            ord.erase(std::remove(ord.begin(), ord.end(), node_id), ord.end());
            st->nodes.erase(it);
            st->composite_dirty.store(true);
        }
        return;
    }

    uint64_t h = fnv1a(reinterpret_cast<const uint8_t*>(payload),
                       sizeof(DocNodePayload));

    DocWorkItem item;
    item.node_id       = node_id;
    item.parent_id     = parent_id;
    item.sibling_order = sibling_order;
    item.expected_hash = h;
    item.rect          = rect;
    item.payload       = *payload;

    {
        std::lock_guard<std::mutex> lk(st->nodes_mtx);
        auto& node = st->nodes[node_id];
        if (node.id == 0) {
            /* New node: register in the live set. */
            node.id = node_id;
            st->order.push_back(node_id);
        }
        node.parent_id = parent_id;
        node.sibling_order = sibling_order;

        /* Pointlessness filter: same payload + already rendered → skip */
        if (node.rendered && !node.dirty && node.payload_hash == h) return;

        node.payload_hash = h;
    }

    {
        std::lock_guard<std::mutex> lk(st->fifo_mtx);
        st->fifo.push_back(std::move(item));
    }
    st->fifo_cv.notify_one();
}

void dr_mark_dirty(DocRendererState* st, uint64_t node_id) {
    std::lock_guard<std::mutex> lk(st->nodes_mtx);
    auto it = st->nodes.find(node_id);
    if (it != st->nodes.end())
        it->second.dirty = true;
}

void dr_clear(DocRendererState* st) {
    {
        std::lock_guard<std::mutex> lk(st->fifo_mtx);
        st->fifo.clear();
    }
    {
        std::lock_guard<std::mutex> lk(st->nodes_mtx);
        st->nodes.clear();
        st->order.clear();
    }
    st->composite_dirty.store(true);
}

void dr_flush(DocRendererState* st) {
    /* Keep polling until FIFO is empty. */
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(st->fifo_mtx);
            if (st->fifo.empty()) break;
        }
        std::this_thread::yield();
    }
}

void dr_composite(DocRendererState* st, uint8_t* out_rgba) {
    const int ow = st->width;
    const int oh = st->height;
    const size_t total = static_cast<size_t>(ow * oh * 4);
    std::memset(out_rgba, 0, total);

    std::lock_guard<std::mutex> lk(st->nodes_mtx);

    /* Hierarchical painter's algorithm.  A node paints after its ancestors,
       and sibling_order controls order inside each parent.  Submission
       sequence is not part of the draw order. */
    std::vector<uint64_t> sorted = st->order;
    auto build_key = [&](uint64_t id) {
        std::vector<uint64_t> chain;
        std::unordered_set<uint64_t> seen;
        uint64_t cur = id;
        while (cur != 0 && seen.insert(cur).second) {
            auto it = st->nodes.find(cur);
            if (it == st->nodes.end()) break;
            chain.push_back(cur);
            uint64_t parent = it->second.parent_id;
            if (parent == 0 || st->nodes.find(parent) == st->nodes.end())
                break;
            cur = parent;
        }
        std::reverse(chain.begin(), chain.end());
        std::vector<uint64_t> key;
        key.reserve(chain.size() * 2);
        for (uint64_t nid : chain) {
            const DocNode& n = st->nodes.at(nid);
            uint64_t local = (n.sibling_order >= 0)
                ? static_cast<uint64_t>(n.sibling_order)
                : static_cast<uint64_t>(0);
            key.push_back(local);
            key.push_back(nid);
        }
        return key;
    };
    std::stable_sort(sorted.begin(), sorted.end(),
        [&](uint64_t a, uint64_t b){
            return build_key(a) < build_key(b);
        });

    for (uint64_t id : sorted) {
        auto it = st->nodes.find(id);
        if (it == st->nodes.end()) continue;
        const DocNode& node = it->second;
        if (!node.rendered || node.tile.empty()) continue;

        const int nx = node.rect.x, ny = node.rect.y;
        const int nw = node.rect.w, nh = node.rect.h;

        for (int row = 0; row < nh; ++row) {
            int dy = ny + row;
            if (dy < 0 || dy >= oh) continue;
            for (int col = 0; col < nw; ++col) {
                int dx = nx + col;
                if (dx < 0 || dx >= ow) continue;

                const uint8_t* src = node.tile.data() + (row * nw + col) * 4;
                uint8_t*       dst = out_rgba + (dy * ow + dx) * 4;

                float sa = src[3] / 255.0f;
                if (sa <= 0.0f) continue;
                float da = dst[3] / 255.0f;
                float oa = sa + da * (1.0f - sa);
                if (oa <= 0.0f) continue;
                dst[0] = f2u8((src[0]/255.0f * sa + dst[0]/255.0f * da*(1.0f-sa)) / oa);
                dst[1] = f2u8((src[1]/255.0f * sa + dst[1]/255.0f * da*(1.0f-sa)) / oa);
                dst[2] = f2u8((src[2]/255.0f * sa + dst[2]/255.0f * da*(1.0f-sa)) / oa);
                dst[3] = f2u8(oa);
            }
        }
    }

    st->composite_dirty.store(false);
}

int dr_composite_dirty(DocRendererState* st) {
    return st->composite_dirty.load() ? 1 : 0;
}

int dr_width (DocRendererState* st) { return st->width;  }
int dr_height(DocRendererState* st) { return st->height; }

int dr_node_count(DocRendererState* st) {
    std::lock_guard<std::mutex> lk(st->nodes_mtx);
    return static_cast<int>(st->nodes.size());
}

int dr_queue_depth(DocRendererState* st) {
    std::lock_guard<std::mutex> lk(st->fifo_mtx);
    return static_cast<int>(st->fifo.size());
}

} /* extern "C" */
