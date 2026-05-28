#pragma once
/* ────────────────────────────────────────────────────────────────────────────
 * t5_hash_grid.h  —  Modular BDPT T5 spatial hash grid
 *
 * Two operating modes share the same build path:
 *
 *   CPU mode  — sorted flat array + std::lower_bound, replaces the
 *               unordered_multimap that was walking a linked-list chain per
 *               cell.  Build once, query per camera vertex via cpu_range().
 *
 *   GPU mode  — packed SSBOs for t5_hash_connect.comp.glsl.  The shader
 *               runs ONE invocation per camera vertex (flattened).  All
 *               invocations execute in parallel on the GPU; each independently
 *               looks up (2·rc+1)³ cells in O(1) per cell via an open-
 *               addressing power-of-two hash table, then accumulates the
 *               geometry-weighted contribution into the pixel buffer.
 *
 * ── GPU SSBO binding map (t5_hash_connect.comp.glsl) ─────────────────────
 *
 *   binding 0  T5LightVertBuf  — packed T5LightGpuVert, stride=T5_LGV_STRIDE
 *                                light vertices in cell-sorted order
 *   binding 1  T5HashBuf       — open-addressing hash table, 4 uint32 per slot
 *                                  [0]  cell_key_hi   (0xFFFFFFFF = empty)
 *                                  [1]  cell_key_lo
 *                                  [2]  vert_start    (index into binding 0)
 *                                  [3]  vert_count
 *   binding 2  T5CamVertBuf    — packed T5CamGpuVert,   stride=T5_CGV_STRIDE
 *                                flattened camera vertices in any order
 *   binding 3  T5PixelBuf      — float pixel accumulator  (3 × res²)
 *                                channels in R-major order: R[res²] G[res²] B[res²]
 *                                written via float-CAS (no GL_EXT_shader_atomic_float)
 *   binding 4  T5ParamsBuf     — T5GpuParams std430 block
 *
 * ── Packed vertex layouts ─────────────────────────────────────────────────
 *
 *   T5LightGpuVert  (T5_LGV_STRIDE = 12 floats = 48 bytes per vertex)
 *     [0..2]   pos xyz
 *     [3..5]   normal xyz
 *     [6]      throughput_scalar
 *     [7]      uintBitsToFloat(flags)
 *     [8]      uintBitsToFloat(subpath_id)
 *     [9]      uintBitsToFloat( (vertex_index & 0xFFFF) | (stream << 16) )
 *     [10]     beta_lum   (precomputed scalar luminance — |throughput_scalar|
 *                          or sum |beta_re| across spectral bands)
 *     [11]     0.0f  (pad)
 *
 *   T5CamGpuVert  (T5_CGV_STRIDE = 14 floats = 56 bytes per vertex)
 *     [0..2]   pos xyz
 *     [3..5]   normal xyz
 *     [6]      throughput_scalar
 *     [7]      uintBitsToFloat(flags)
 *     [8]      uintBitsToFloat(subpath_id)
 *     [9]      uintBitsToFloat(vertex_index)
 *     [10]     beta_lum
 *     [11]     sensor_origin_y
 *     [12]     sensor_origin_z
 *     [13]     0.0f  (pad)
 *
 * ────────────────────────────────────────────────────────────────────────── */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

/* ── Packed vertex strides ───────────────────────────────────────────────── */
static constexpr int T5_LGV_STRIDE = 12; /* light vertex, floats */
static constexpr int T5_CGV_STRIDE = 14; /* camera vertex, floats */
static constexpr int T5_HASH_SLOT  =  4; /* uint32 per hash table slot */
static constexpr uint32_t T5_EMPTY = 0xFFFFFFFFu; /* empty hash slot marker */

/* ── GPU params block (std430, binding 4) ───────────────────────────────── */
struct T5GpuParams {
    float    cell_size;        /* metres per grid cell */
    float    inv_cell_size;    /* 1 / cell_size */
    float    min_geom;         /* geometry-term floor */
    float    sensor_half_w;    /* sensor half-width  (Y axis, metres) */
    uint32_t n_light_verts;    /* entries in T5LightVertBuf */
    uint32_t n_cam_verts;      /* entries in T5CamVertBuf  */
    uint32_t hash_table_mask;  /* table_size - 1 (table_size is power-of-two) */
    int32_t  radius_cells;     /* search radius (cells per axis, ≥ 1) */
    int32_t  sensor_res;       /* pixel grid side (res × res image) */
    float    sensor_half_h;    /* sensor half-height (Z axis, metres) */
    float    _pad[2];
};
static_assert(sizeof(T5GpuParams) == 48, "T5GpuParams layout mismatch");

/* ────────────────────────────────────────────────────────────────────────────
 * T5HashGrid  —  owns all CPU and GPU buffers for the T5 connection pass.
 * ────────────────────────────────────────────────────────────────────────── */
struct T5HashGrid {

    /* ── Cell encoding ──────────────────────────────────────────────────── */
    static uint64_t encode_cell(int gx, int gy, int gz) noexcept {
        /* 21 bits per axis, biased +2^20 for negative coords. */
        return (static_cast<uint64_t>(gx + (1 << 20)) << 42)
             | (static_cast<uint64_t>(gy + (1 << 20)) << 21)
             |  static_cast<uint64_t>(gz + (1 << 20));
    }

    static std::pair<uint32_t, uint32_t> split_key(uint64_t k) noexcept {
        return { static_cast<uint32_t>(k >> 32), static_cast<uint32_t>(k) };
    }

    /* ── CPU sorted storage ─────────────────────────────────────────────── *
     * sorted_kvs: pairs (cell_key, original_vert_idx) sorted ascending by
     * cell_key.  cpu_range(key) returns [begin, end) slice indices via two
     * lower_bound calls — single allocation, contiguous iteration per cell.  */
    std::vector<std::pair<uint64_t, uint32_t>> sorted_kvs;

    /* Range of indices in sorted_kvs whose key == cell_key.
     * Returns {0,0} if the cell is absent.  The caller iterates
     * sorted_kvs[lo].second .. sorted_kvs[hi-1].second. */
    std::pair<size_t, size_t> cpu_range(uint64_t cell_key) const noexcept {
        auto lo = std::lower_bound(sorted_kvs.begin(), sorted_kvs.end(),
                                   std::make_pair(cell_key, (uint32_t)0));
        if (lo == sorted_kvs.end() || lo->first != cell_key)
            return {0, 0};
        auto hi = std::upper_bound(lo, sorted_kvs.end(),
                                   std::make_pair(cell_key,
                                       std::numeric_limits<uint32_t>::max()));
        return { static_cast<size_t>(lo - sorted_kvs.begin()),
                 static_cast<size_t>(hi - sorted_kvs.begin()) };
    }

    /* ── GPU buffers ────────────────────────────────────────────────────── *
     * These are sized and ready to upload directly via glBufferData.        */

    /* Light vertices sorted in cell order (T5_LGV_STRIDE floats each). */
    std::vector<float>    gpu_light_verts;   /* size = n_light_verts × T5_LGV_STRIDE */

    /* Open-addressing hash table: T5_HASH_SLOT uint32s per slot.
     * table_size is always a power of two.  Empty slot: slot[0] = T5_EMPTY. */
    std::vector<uint32_t> gpu_hash_table;    /* size = table_size × T5_HASH_SLOT */

    /* Camera vertices (T5_CGV_STRIDE floats each). */
    std::vector<float>    gpu_cam_verts;     /* size = n_cam_verts × T5_CGV_STRIDE */

    /* Pixel accumulator initialised to 0.  Shader writes via float CAS.
     * Layout: R[res²] then G[res²] then B[res²], stored as uint32 bit-casts
     * (GPU writes float via CAS; host reads back and reinterpret_casts).    */
    std::vector<uint32_t> gpu_pixel_accum;   /* size = 3 × res² */

    T5GpuParams           gpu_params{};

    /* ── Build (call once before dispatch) ──────────────────────────────── *
     *
     * light_packed: pre-packed T5LightGpuVert records,  n × T5_LGV_STRIDE floats.
     * light_pos:    pointer to a (n × 3) float array of world positions —
     *               used to compute cell keys for the hash table.
     * cell_size:    grid cell side length (metres).
     * n_light:      number of light vertices.
     *
     * The build:
     *   1. Computes a cell key for every light vertex from its position.
     *   2. Sorts vertex indices by cell key.
     *   3. Rebuilds gpu_light_verts in sorted order (contiguous cell access).
     *   4. Builds an open-addressing hash table mapping cell_key → (start, count).
     *   5. Copies light_packed-in-sorted-order into gpu_light_verts.
     *   6. sorted_kvs is populated for the CPU query path.               */
    void build_light(const float* light_packed,  /* n × T5_LGV_STRIDE floats */
                     uint32_t     n_light,
                     float        cell_size)
    {
        if (n_light == 0) {
            sorted_kvs.clear();
            gpu_light_verts.clear();
            gpu_hash_table.clear();
            return;
        }
        const float ics = 1.0f / cell_size;

        /* 1. Compute (cell_key, original_idx) for every light vertex. */
        sorted_kvs.resize(n_light);
        for (uint32_t i = 0; i < n_light; ++i) {
            const float* p = light_packed + (size_t)i * T5_LGV_STRIDE; /* pos at [0..2] */
            const int gx = static_cast<int>(std::floor(p[0] * ics));
            const int gy = static_cast<int>(std::floor(p[1] * ics));
            const int gz = static_cast<int>(std::floor(p[2] * ics));
            sorted_kvs[i] = { encode_cell(gx, gy, gz), i };
        }

        /* 2. Sort by cell key. */
        std::sort(sorted_kvs.begin(), sorted_kvs.end());

        /* 3. Rebuild gpu_light_verts in sorted order. */
        gpu_light_verts.resize((size_t)n_light * T5_LGV_STRIDE);
        for (uint32_t s = 0; s < n_light; ++s) {
            const uint32_t orig = sorted_kvs[s].second;
            std::memcpy(gpu_light_verts.data() + (size_t)s * T5_LGV_STRIDE,
                        light_packed           + (size_t)orig * T5_LGV_STRIDE,
                        T5_LGV_STRIDE * sizeof(float));
        }
        /* sorted_kvs[s].second retains the ORIGINAL index into the caller's
         * light_verts_flat array throughout — do NOT remap.  The GPU hash
         * table stores sorted positions (cell_start) which index into
         * gpu_light_verts (cell-sorted order) independently.              */

        /* 4. Count unique cells and size the hash table. */
        uint32_t n_unique = 0;
        {
            uint64_t prev = ~sorted_kvs[0].first; /* guaranteed != first key */
            for (const auto& kv : sorted_kvs)
                if (kv.first != prev) { ++n_unique; prev = kv.first; }
        }

        /* Table size ≥ 4 × n_unique, rounded up to power-of-two. */
        uint32_t table_size = 4;
        while (table_size < n_unique * 4u) table_size <<= 1;
        const uint32_t mask = table_size - 1u;

        gpu_hash_table.assign((size_t)table_size * T5_HASH_SLOT, 0u);
        /* Mark all slots empty. */
        for (uint32_t s = 0; s < table_size; ++s)
            gpu_hash_table[(size_t)s * T5_HASH_SLOT + 0] = T5_EMPTY;

        /* 5. Insert unique cells: iterate sorted_kvs, detect cell boundaries. */
        uint64_t prev_key = ~sorted_kvs[0].first;
        uint32_t cell_start = 0;
        for (uint32_t s = 0; s <= n_light; ++s) {
            const uint64_t cur_key = (s < n_light) ? sorted_kvs[s].first : ~prev_key;
            if (cur_key != prev_key && s > 0) {
                /* Flush the just-closed cell [cell_start, s). */
                const auto [khi, klo] = split_key(prev_key);
                uint32_t h = (khi * 2654435761u ^ klo * 2246822519u) & mask;
                while (gpu_hash_table[(size_t)h * T5_HASH_SLOT] != T5_EMPTY) {
                    h = (h + 1u) & mask;
                }
                gpu_hash_table[(size_t)h * T5_HASH_SLOT + 0] = khi;
                gpu_hash_table[(size_t)h * T5_HASH_SLOT + 1] = klo;
                gpu_hash_table[(size_t)h * T5_HASH_SLOT + 2] = cell_start;
                gpu_hash_table[(size_t)h * T5_HASH_SLOT + 3] = s - cell_start;
                cell_start = s;
            }
            prev_key = cur_key;
        }

        gpu_params.hash_table_mask = mask;
    }

    /* Pack camera vertices for GPU upload.
     * cam_packed: n × T5_CGV_STRIDE floats, in any order.              */
    void set_cam_verts(const float* cam_packed, uint32_t n_cam) {
        gpu_cam_verts.assign(cam_packed, cam_packed + (size_t)n_cam * T5_CGV_STRIDE);
    }

    /* Initialise the pixel accumulator (zeroed, stored as uint32 bit-casts). */
    void init_pixel_accum(int sensor_res) {
        const size_t pix = static_cast<size_t>(sensor_res) * sensor_res;
        gpu_pixel_accum.assign(3 * pix, 0u);
    }

    /* Convenience: CPU lookup using sorted_kvs (replaces unordered_multimap
     * equal_range in the CPU worker).  Returns sorted indices into the
     * gpu_light_verts array (which is in sorted cell order).             */
    std::pair<uint32_t, uint32_t> cpu_cell(uint64_t cell_key) const noexcept {
        auto [lo, hi] = cpu_range(cell_key);
        return { static_cast<uint32_t>(lo), static_cast<uint32_t>(hi) };
    }

    uint32_t n_light_verts() const noexcept {
        return static_cast<uint32_t>(
            gpu_light_verts.size() / static_cast<size_t>(T5_LGV_STRIDE));
    }
    uint32_t n_cam_verts() const noexcept {
        return static_cast<uint32_t>(
            gpu_cam_verts.size() / static_cast<size_t>(T5_CGV_STRIDE));
    }
};
