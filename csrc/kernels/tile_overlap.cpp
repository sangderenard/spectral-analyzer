#define _USE_MATH_DEFINES
/**
 * tile_overlap.cpp — Parallel tile-item SAT overlap detection.
 *
 * Design:
 *   - Separating Axis Theorem (SAT) for convex quad vs AABB — exact, no grid.
 *   - Static thread pool identical to base_rasterizer.cpp (tile-stealing).
 *   - Outer parallel loop: one job per tile.  Each job iterates all items.
 *   - For a bounded room (~500 tiles × ~100 items × 6 SAT axes) this is
 *     ~300k dot products total — fast enough that no BVH is warranted.
 */

#include "tile_overlap.h"

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/* ═══════════════════════════════════════════════════════════════════════════
 * Thread pool  (same pattern as base_rasterizer.cpp / acoustic_amr.cpp)
 * ═══════════════════════════════════════════════════════════════════════════ */
namespace {

struct OvlPool {
    int n;
    std::vector<std::thread> workers;
    std::mutex               mx;
    std::condition_variable  cv_wake, cv_done;
    std::function<void(int)> job;
    int              job_n  = 0;
    int              active = 0;
    bool             quit   = false;
    std::atomic<int> next{0};

    explicit OvlPool(int nt) : n(nt) {
        workers.reserve(nt);
        for (int i = 0; i < nt; ++i)
            workers.emplace_back(&OvlPool::loop, this);
    }

    ~OvlPool() {
        { std::unique_lock<std::mutex> lk(mx); quit = true; }
        cv_wake.notify_all();
        for (auto& t : workers) t.join();
    }

    void loop() {
        for (;;) {
            std::function<void(int)> fn;
            int local_n;
            {
                std::unique_lock<std::mutex> lk(mx);
                cv_wake.wait(lk, [this]{ return quit || next.load() < job_n; });
                if (quit) return;
                ++active; fn = job; local_n = job_n;
            }
            int id;
            while ((id = next.fetch_add(1, std::memory_order_relaxed)) < local_n)
                fn(id);
            {
                std::lock_guard<std::mutex> lk(mx);
                --active;
                if (next.load() >= job_n && active == 0)
                    cv_done.notify_one();
            }
        }
    }

    void run(int total, std::function<void(int)> fn) {
        if (total <= 0) return;
        {
            std::unique_lock<std::mutex> lk(mx);
            job = std::move(fn); job_n = total; active = 0;
            next.store(0, std::memory_order_relaxed);
        }
        cv_wake.notify_all();
        std::unique_lock<std::mutex> lk(mx);
        cv_done.wait(lk, [this]{ return next.load() >= job_n && active == 0; });
    }
};

static OvlPool*    g_pool = nullptr;
static std::once_flag g_once;

static OvlPool& ovl_pool() {
    std::call_once(g_once, []{
        int nt = static_cast<int>(std::thread::hardware_concurrency());
        if (nt < 1) nt = 1;
        g_pool = new OvlPool(nt);
    });
    return *g_pool;
}

template<typename F>
static void parallel_for(int n, F&& fn) {
    ovl_pool().run(n, [&](int i){ fn(i); });
}

} // namespace

/* ═══════════════════════════════════════════════════════════════════════════
 * SAT: convex quad vs axis-aligned bounding box
 *
 * corners[0..7] = 4 × (x,y), any winding order.
 * rect: (rx, ry, rx+rw, ry+rh).
 *
 * Tests 6 potential separating axes:
 *   4 quad edge normals  +  2 AABB principal axes (x, y).
 * Returns true iff the polygons overlap (no separating axis found).
 * ═══════════════════════════════════════════════════════════════════════════ */
static bool sat_quad_aabb(const float* c,
                          float rx, float ry, float rw, float rh) {
    const float eps = 1e-7f;

    /* AABB corners for projection */
    const float bx[4] = { rx,      rx + rw, rx + rw, rx      };
    const float by[4] = { ry,      ry,      ry + rh, ry + rh };

    /* ── 4 quad edge normals ─────────────────────────────────────────────── */
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) & 3;
        /* Edge vector i→j; outward normal = (-dy, dx) */
        float nx = -(c[j * 2 + 1] - c[i * 2 + 1]);
        float ny =   c[j * 2    ] - c[i * 2    ];
        if (std::abs(nx) < 1e-9f && std::abs(ny) < 1e-9f) continue;

        float qmin =  1e30f, qmax = -1e30f;
        for (int k = 0; k < 4; ++k) {
            float d = nx * c[k * 2] + ny * c[k * 2 + 1];
            if (d < qmin) qmin = d;
            if (d > qmax) qmax = d;
        }

        float bmin =  1e30f, bmax = -1e30f;
        for (int k = 0; k < 4; ++k) {
            float d = nx * bx[k] + ny * by[k];
            if (d < bmin) bmin = d;
            if (d > bmax) bmax = d;
        }

        if (qmax < bmin - eps || bmax < qmin - eps) return false;
    }

    /* ── X axis (AABB normal) ────────────────────────────────────────────── */
    {
        float qmin = c[0], qmax = c[0];
        for (int k = 1; k < 4; ++k) {
            if (c[k * 2] < qmin) qmin = c[k * 2];
            if (c[k * 2] > qmax) qmax = c[k * 2];
        }
        if (qmax < rx - eps || rx + rw < qmin - eps) return false;
    }

    /* ── Y axis (AABB normal) ────────────────────────────────────────────── */
    {
        float qmin = c[1], qmax = c[1];
        for (int k = 1; k < 4; ++k) {
            if (c[k * 2 + 1] < qmin) qmin = c[k * 2 + 1];
            if (c[k * 2 + 1] > qmax) qmax = c[k * 2 + 1];
        }
        if (qmax < ry - eps || ry + rh < qmin - eps) return false;
    }

    return true;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * State
 * ═══════════════════════════════════════════════════════════════════════════ */
struct TileOverlapState {
    /* Tile polygons: flat [n_tiles * 8] float */
    std::vector<float> tile_corners;
    std::vector<int>   tile_ids;

    /* Item AABBs: flat [n_items * 4] float (x,y,w,h) */
    std::vector<float> item_rects;
    std::vector<int>   item_ids;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

TileOverlapState* to_create(void) {
    return new TileOverlapState();
}

void to_destroy(TileOverlapState* st) {
    delete st;
}

void to_set_tiles(TileOverlapState* st,
                  const float* corners,
                  const int*   ids,
                  int          n_tiles) {
    if (!st || n_tiles <= 0) return;
    st->tile_corners.assign(corners, corners + (size_t)n_tiles * 8);
    st->tile_ids.assign(ids, ids + n_tiles);
}

void to_set_items(TileOverlapState* st,
                  const float* rects,
                  const int*   ids,
                  int          n_items) {
    if (!st) return;
    if (n_items <= 0) {
        st->item_rects.clear();
        st->item_ids.clear();
        return;
    }
    st->item_rects.assign(rects, rects + (size_t)n_items * 4);
    st->item_ids.assign(ids, ids + n_items);
}

void to_compute(TileOverlapState* st,
                int8_t* out_status,
                int*    out_counts,
                int*    out_ids) {
    if (!st) return;

    const int nt = static_cast<int>(st->tile_ids.size());
    const int ni = static_cast<int>(st->item_ids.size());

    const float* TC = st->tile_corners.data();
    const float* IR = st->item_rects.data();
    const int*   TI = st->tile_ids.data();

    parallel_for(nt, [&](int t) {
        const float* c = TC + (size_t)t * 8;
        int cnt = 0;

        for (int i = 0; i < ni; ++i) {
            const float* r = IR + (size_t)i * 4;
            if (sat_quad_aabb(c, r[0], r[1], r[2], r[3]))
                ++cnt;
        }

        out_counts[t] = cnt;
        out_ids[t]    = TI[t];
        out_status[t] = (cnt == 0) ? TO_CLEAR
                      : (cnt == 1) ? TO_OCCUPIED
                                   : TO_COLLISION;
    });
}

int to_n_tiles(const TileOverlapState* st) {
    return st ? static_cast<int>(st->tile_ids.size()) : 0;
}

int to_n_items(const TileOverlapState* st) {
    return st ? static_cast<int>(st->item_ids.size()) : 0;
}

} // extern "C"
