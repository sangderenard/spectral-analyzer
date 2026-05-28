#version 430 core
/* ────────────────────────────────────────────────────────────────────────────
 * t5_hash_connect.comp.glsl  —  GPU BDPT T5 spatial hash connection pass
 *
 * Dispatch: glDispatchCompute(ceil(n_cam_verts / 64), 1, 1)
 *
 * Each invocation = ONE flattened camera vertex.  All invocations run
 * simultaneously.  Each invocation independently:
 *
 *   1. Reads its camera vertex from T5CamVertBuf.
 *   2. Computes the integer grid cell (gx0, gy0, gz0) for that vertex.
 *   3. Iterates the (2·rc+1)³ neighbor cells.
 *   4. For each cell: O(1) lookup in the open-addressing hash table.
 *   5. For each light vertex in that cell: computes geometry term.
 *   6. Accumulates the geometry-weighted luminance contribution to the
 *      correct pixel in T5PixelBuf via a float CAS spin loop.
 *
 * MIS weighting and spectral colourisation are deferred to the CPU post-pass
 * (the same accum_pixel_color path already in ray_tracer.cpp).  The GPU
 * computes the dominant geometry × luminance factor, which is the bottleneck.
 *
 * ── SSBO binding map ───────────────────────────────────────────────────────
 *
 *  binding  name            access    description
 *  -------  --------------  --------  ----------------------------------------
 *    0      T5LightVertBuf  readonly  Light vertices, cell-sorted.
 *                                     Stride = T5_LGV_STRIDE = 12 floats.
 *                                       [0..2]  pos xyz
 *                                       [3..5]  normal xyz
 *                                       [6]     throughput_scalar
 *                                       [7]     uintBitsToFloat(flags)
 *                                       [8]     uintBitsToFloat(subpath_id)
 *                                       [9]     uintBitsToFloat(vert_info)
 *                                       [10]    beta_lum
 *                                       [11]    0 (pad)
 *    1      T5HashBuf       readonly  Open-addressing hash table.
 *                                     4 uint32 per slot:
 *                                       [0]  cell_key_hi  (0xFFFFFFFF = empty)
 *                                       [1]  cell_key_lo
 *                                       [2]  vert_start
 *                                       [3]  vert_count
 *    2      T5CamVertBuf    readonly  Camera vertices (flattened, any order).
 *                                     Stride = T5_CGV_STRIDE = 14 floats.
 *                                       [0..2]  pos xyz
 *                                       [3..5]  normal xyz
 *                                       [6]     throughput_scalar
 *                                       [7]     uintBitsToFloat(flags)
 *                                       [8]     uintBitsToFloat(subpath_id)
 *                                       [9]     uintBitsToFloat(vert_index)
 *                                       [10]    beta_lum
 *                                       [11]    sensor_origin_y
 *                                       [12]    sensor_origin_z
 *                                       [13]    0 (pad)
 *    3      T5PixelBuf      coherent  Pixel accumulator.
 *                                     3 × res² uint32 bit-cast floats.
 *                                     Layout: R[res²] G[res²] B[res²].
 *                                     Written via float CAS (no ext needed).
 *    4      T5ParamsBuf     readonly  T5GpuParams std430 block (see below).
 *
 * ── T5GpuParams (std430, binding 4) ───────────────────────────────────────
 *   float  cell_size        metres per grid cell
 *   float  inv_cell_size    1 / cell_size
 *   float  min_geom         geometry-term floor
 *   float  sensor_half_w    sensor half-width  (Y axis, metres)
 *   uint   n_light_verts
 *   uint   n_cam_verts
 *   uint   hash_table_mask  table_size − 1
 *   int    radius_cells     search radius (cells per axis)
 *   int    sensor_res       pixel grid side (res×res)
 *   float  _pad[3]
 * ──────────────────────────────────────────────────────────────────────── */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

/* ── Stride constants (must match t5_hash_grid.h) ──────────────────────── */
#define T5_LGV_STRIDE   12
#define T5_CGV_STRIDE   14
#define T5_HASH_SLOT     4
#define T5_EMPTY        0xFFFFFFFFu

/* NOTE: BDPT_FLAG_CONNECTABLE is intentionally NOT checked here.
 * The C++ packing already pre-filters to connectable vertices only,
 * so every vertex present in the SSBOs is implicitly connectable.
 * flags[7] holds raw material flags (e.g. MAT_FLAG_APERTURE_STOP=0x80)
 * which have nothing to do with connection eligibility.              */

/* ── SSBO declarations ──────────────────────────────────────────────────── */
layout(std430, binding = 0) readonly buffer T5LightVertBuf {
    float light_verts[];   /* n_light_verts × T5_LGV_STRIDE */
};

layout(std430, binding = 1) readonly buffer T5HashBuf {
    uint hash_slots[];     /* table_size × T5_HASH_SLOT */
};

layout(std430, binding = 2) readonly buffer T5CamVertBuf {
    float cam_verts[];     /* n_cam_verts × T5_CGV_STRIDE */
};

layout(std430, binding = 3) coherent buffer T5PixelBuf {
    uint pixel_accum[];    /* 3 × sensor_res² uint32 bit-cast floats */
};

layout(std430, binding = 4) readonly buffer T5ParamsBuf {
    float  cell_size;
    float  inv_cell_size;
    float  min_geom;
    float  sensor_half_w;
    uint   n_light_verts;
    uint   n_cam_verts;
    uint   hash_table_mask;
    int    radius_cells;
    int    sensor_res;
    float  sensor_half_h;
    float  _pad0;
    float  _pad1;
};

/* ── Float atomic add via CAS spin-loop ─────────────────────────────────── *
 * Avoids GL_EXT_shader_atomic_float.  Standard on all GL 4.3+ hardware.    */
void atomic_add_float(uint idx, float val) {
    uint expected = pixel_accum[idx];
    for (int i = 0; i < 64; ++i) {
        float fexp = uintBitsToFloat(expected);
        uint  desired = floatBitsToUint(fexp + val);
        uint  actual  = atomicCompSwap(pixel_accum[idx], expected, desired);
        if (actual == expected) return;
        expected = actual;
    }
}

/* ── Open-addressing cell lookup ────────────────────────────────────────── *
 * Returns (start, count) for a given 64-bit cell key, or (0, 0) if absent. */
uvec2 hash_lookup(uint key_hi, uint key_lo) {
    uint h = (key_hi * 2654435761u ^ key_lo * 2246822519u) & hash_table_mask;
    for (uint probe = 0u; probe <= hash_table_mask; ++probe) {
        uint base = h * uint(T5_HASH_SLOT);
        uint skey_hi = hash_slots[base + 0u];
        if (skey_hi == T5_EMPTY) return uvec2(0u, 0u);   /* empty slot */
        if (skey_hi == key_hi && hash_slots[base + 1u] == key_lo)
            return uvec2(hash_slots[base + 2u], hash_slots[base + 3u]);
        h = (h + 1u) & hash_table_mask;
    }
    return uvec2(0u, 0u);
}

/* ── Cell key encode (must match T5HashGrid::encode_cell in C++) ────────── *
 * key64 = uint64(ax) << 42 | uint64(ay) << 21 | uint64(az)                *
 *   ax,ay,az are 21-bit biased coords (coord + 2^20).                      *
 *   key_hi = key64 >> 32  =  (ax << 10) | (ay >> 11)                      *
 *   key_lo = key64 & 0xFFFFFFFF  =  ((ay & 0x7FF) << 21) | (az & 0x1FFFFF)*/
uvec2 encode_cell(int gx, int gy, int gz) {
    uint ax = uint(gx + (1 << 20));
    uint ay = uint(gy + (1 << 20));
    uint az = uint(gz + (1 << 20));
    uint key_hi = (ax << 10u) | (ay >> 11u);
    uint key_lo = ((ay & 0x7FFu) << 21u) | (az & 0x1FFFFFu);
    return uvec2(key_hi, key_lo);
}

/* ── Main ────────────────────────────────────────────────────────────────── */
void main() {
    const uint gid = gl_GlobalInvocationID.x;
    if (gid >= n_cam_verts) return;

    /* ── Load camera vertex ─────────────────────────────────────────────── */
    const uint cb = gid * uint(T5_CGV_STRIDE);
    const vec3  c_pos    = vec3(cam_verts[cb+0u], cam_verts[cb+1u], cam_verts[cb+2u]);
    const vec3  c_norm   = vec3(cam_verts[cb+3u], cam_verts[cb+4u], cam_verts[cb+5u]);
    const float c_ts     = cam_verts[cb+6u];
    const uint  c_flags  = floatBitsToUint(cam_verts[cb+7u]);
    const float c_beta   = cam_verts[cb+10u];
    const float c_soy    = cam_verts[cb+11u];
    const float c_soz    = cam_verts[cb+12u];

    /* ── Connectable check ────────────────────────────────────────────────
     * Pre-filtered by C++ packing — every vertex in the SSBO is connectable.
     * Nothing to check here; c_flags holds material flags, not vertex state. */
    if (c_beta < 1e-15f) return;

    /* ── Pixel index ────────────────────────────────────────────────────── */
    const float inv_w = float(sensor_res) / (2.0f * sensor_half_w);
    const float inv_h = float(sensor_res) / (2.0f * sensor_half_h);
    const int iy = int((c_soy + sensor_half_w) * inv_w);
    const int iz = int((c_soz + sensor_half_h) * inv_h);
    if (iy < 0 || iy >= sensor_res || iz < 0 || iz >= sensor_res) return;
    const uint px = uint(iy * sensor_res + iz);
    const uint pix = uint(sensor_res) * uint(sensor_res);

    /* ── Camera vertex cell ─────────────────────────────────────────────── */
    const int gx0 = int(floor(c_pos.x * inv_cell_size));
    const int gy0 = int(floor(c_pos.y * inv_cell_size));
    const int gz0 = int(floor(c_pos.z * inv_cell_size));

    float lum_accum = 0.0f;

    /* ── Iterate (2·rc+1)³ neighbor cells ──────────────────────────────── */
    for (int dix = -radius_cells; dix <= radius_cells; ++dix) {
        for (int diy = -radius_cells; diy <= radius_cells; ++diy) {
            for (int diz = -radius_cells; diz <= radius_cells; ++diz) {

                uvec2 cell_key = encode_cell(gx0 + dix, gy0 + diy, gz0 + diz);
                uvec2 range    = hash_lookup(cell_key.x, cell_key.y);
                const uint lv_start = range.x;
                const uint lv_count = range.y;
                if (lv_count == 0u) continue;

                /* ── Iterate light vertices in this cell ──────────────── */
                for (uint li = lv_start; li < lv_start + lv_count; ++li) {
                    const uint lb = li * uint(T5_LGV_STRIDE);
                    const vec3  l_pos  = vec3(light_verts[lb+0u],
                                              light_verts[lb+1u],
                                              light_verts[lb+2u]);
                    const vec3  l_norm = vec3(light_verts[lb+3u],
                                              light_verts[lb+4u],
                                              light_verts[lb+5u]);
                    const uint  l_flags = floatBitsToUint(light_verts[lb+7u]);
                    const float l_beta  = light_verts[lb+10u];

                    if (l_beta < 1e-15f) continue;  /* pre-filtered, but guard zero beta */

                    /* ── Geometry term ──────────────────────────────────── */
                    const vec3  dv    = l_pos - c_pos;
                    const float dist2 = dot(dv, dv);
                    if (dist2 < 1e-12f) continue;
                    const float dist  = sqrt(dist2);
                    const vec3  wc    = dv / dist;
                    const float cos_c = abs(dot(c_norm, wc));
                    const float cos_l = abs(dot(l_norm, -wc));
                    const float geom  = cos_c * cos_l / dist2;
                    if (geom < min_geom) continue;

                    /* ── Visibility (BVH shadow ray) ─────────────────────── */
                    if (shadow_occluded(c_pos, l_pos)) continue;

                    /* ── Accumulate (geometry × beta product) ───────────── *
                     * MIS weight = 1.0 here; CPU post-pass applies correct  *
                     * power-heuristic weights via accum_pixel_color.        */
                    lum_accum += c_beta * l_beta * geom;
                }
            }
        }
    }

    /* ── Write to pixel buffer (neutral 1/3 split across channels) ──────── *
     * The CPU post-pass spectral recolour replaces this; for now all three  *
     * channels receive equal weight so the image is greyscale-correct.      */
    if (lum_accum > 0.0f) {
        const float third = lum_accum * (1.0f / 3.0f);
        atomic_add_float(px,          third);  /* R */
        atomic_add_float(px + pix,    third);  /* G */
        atomic_add_float(px + 2u*pix, third);  /* B */
    }
}
