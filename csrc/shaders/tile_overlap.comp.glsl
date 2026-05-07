#version 430 core
/* tile_overlap.comp.glsl — GPU-parallel tile-item SAT overlap detection.
 *
 * Mirrors tile_overlap.cpp exactly: Separating Axis Theorem for convex quad
 * vs AABB across every (tile, item) pair.
 *
 * Dispatch: glDispatchCompute(n_tiles, 1, 1)
 *   workgroup = one tile
 *   local threads = items (up to LOCAL_SIZE; strides if n_items > LOCAL_SIZE)
 *
 * Each thread tests a subset of items against the tile and atomically
 * accumulates into shared s_count.  Thread 0 writes the final status.
 *
 * SSBO layout:
 *   binding=0  tile_corners  — n_tiles * 8 float  (4 corners × xy, metres)
 *   binding=1  item_rects    — n_items * 4 float  (x, y, w, h, metres)
 *   binding=2  out_counts    — n_tiles   * 1 int   (overlap item count per tile)
 *   binding=3  out_status    — n_tiles   * 1 int   (TO_CLEAR/OCCUPIED/COLLISION)
 *
 * Status constants match tile_overlap.h:
 *   TO_CLEAR     = 1
 *   TO_OCCUPIED  = 2
 *   TO_COLLISION = 3
 */

#define LOCAL_SIZE 64
layout(local_size_x = LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

/* ── SSBOs ─────────────────────────────────────────────────────────────── */
layout(std430, binding = 0) readonly buffer TileCornersBuf {
    float tile_corners[];
};
layout(std430, binding = 1) readonly buffer ItemRectsBuf {
    float item_rects[];
};
layout(std430, binding = 2) writeonly buffer OutCounts {
    int out_counts[];
};
layout(std430, binding = 3) writeonly buffer OutStatus {
    int out_status[];
};

/* ── Uniforms ──────────────────────────────────────────────────────────── */
uniform int n_tiles;
uniform int n_items;

/* ── Shared overlap counter ────────────────────────────────────────────── */
shared int s_count;

/* ══════════════════════════════════════════════════════════════════════════
 * SAT: convex quad (8 floats) vs AABB (x,y,w,h)
 *
 * Tests 6 potential separating axes:
 *   4 quad edge normals  (outward = (-dy, dx))
 *   2 AABB principal axes (x and y)
 *
 * Returns true iff polygons overlap.
 * ══════════════════════════════════════════════════════════════════════════ */
bool sat_quad_aabb(float c[8], float rx, float ry, float rw, float rh) {
    const float eps = 1e-7;

    float bx[4] = float[4](rx,      rx + rw, rx + rw, rx);
    float by_[4] = float[4](ry,      ry,      ry + rh, ry + rh);

    /* 4 quad edge normals */
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) & 3;
        float nx = -(c[j * 2 + 1] - c[i * 2 + 1]);
        float ny =   c[j * 2    ] - c[i * 2    ];
        if (abs(nx) < 1e-9 && abs(ny) < 1e-9) continue;

        float qmin =  1e30, qmax = -1e30;
        for (int k = 0; k < 4; ++k) {
            float d = nx * c[k * 2] + ny * c[k * 2 + 1];
            qmin = min(qmin, d); qmax = max(qmax, d);
        }

        float bmin =  1e30, bmax = -1e30;
        for (int k = 0; k < 4; ++k) {
            float d = nx * bx[k] + ny * by_[k];
            bmin = min(bmin, d); bmax = max(bmax, d);
        }

        if (qmax < bmin - eps || bmax < qmin - eps) return false;
    }

    /* X axis */
    {
        float qmin = min(min(c[0], c[2]), min(c[4], c[6]));
        float qmax = max(max(c[0], c[2]), max(c[4], c[6]));
        if (qmax < rx - eps || rx + rw < qmin - eps) return false;
    }

    /* Y axis */
    {
        float qmin = min(min(c[1], c[3]), min(c[5], c[7]));
        float qmax = max(max(c[1], c[3]), max(c[5], c[7]));
        if (qmax < ry - eps || ry + rh < qmin - eps) return false;
    }

    return true;
}

/* ── Main ──────────────────────────────────────────────────────────────── */
void main() {
    int tile_idx = int(gl_WorkGroupID.x);
    if (tile_idx >= n_tiles) return;

    /* Initialise shared counter once per workgroup */
    if (gl_LocalInvocationID.x == 0u)
        s_count = 0;
    barrier();
    memoryBarrierShared();

    /* Load this tile's 8 corner floats into local registers */
    int tb = tile_idx * 8;
    float tc[8];
    for (int i = 0; i < 8; ++i)
        tc[i] = tile_corners[tb + i];

    /* Each thread strides over items */
    int local_count = 0;
    for (int i = int(gl_LocalInvocationID.x); i < n_items; i += LOCAL_SIZE) {
        int ib = i * 4;
        float rx = item_rects[ib    ];
        float ry = item_rects[ib + 1];
        float rw = item_rects[ib + 2];
        float rh = item_rects[ib + 3];
        if (sat_quad_aabb(tc, rx, ry, rw, rh))
            local_count++;
    }

    if (local_count > 0)
        atomicAdd(s_count, local_count);

    barrier();
    memoryBarrierShared();

    /* Thread 0 writes result */
    if (gl_LocalInvocationID.x == 0u) {
        int cnt = s_count;
        out_counts[tile_idx] = cnt;
        out_status[tile_idx] = (cnt == 0) ? 1   /* TO_CLEAR     */
                             : (cnt == 1) ? 2   /* TO_OCCUPIED  */
                                          : 3;  /* TO_COLLISION */
    }
}
