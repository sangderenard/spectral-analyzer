/*
 * ray_bvh_intersect.comp.glsl — GPU compute T1: BVH ray-triangle intersection
 *                                + amplitude propagation.
 *
 * One invocation per input RayIntent.  Outputs a RefinedHit-layout record
 * (same flat struct that T2 and T3 consume) into HitBuf.
 *
 * Flat buffer layouts (must match GlPipelineDispatch packing):
 *
 *  IntentBuf  (INTENT_STRIDE = 22 + 2*MAX_GPU_BANDS floats = 86 with MAX_GPU_BANDS=32):
 *    [0..2]   pos xyz
 *    [3..5]   dir xyz
 *    [6]      path_len
 *    [7]      medium_mat_idx  (intBitsToFloat)
 *    [8]      interaction_flags (uintBitsToFloat)
 *    [9]      src_id          (intBitsToFloat)
 *    [10]     bounce          (intBitsToFloat)
 *    [11]     bounces_left    (intBitsToFloat)
 *    [12]     min_amplitude
 *    [13]     tag_lo          (uintBitsToFloat)
 *    [14]     tag_hi          (uintBitsToFloat)
 *    [15]     color/spectral metadata (low byte color flag; uintBitsToFloat)
 *    [16]     priority
 *    [17]     sensor_origin_y
 *    [18]     sensor_origin_z
 *    [19]     continuous frequency_hz
 *    [20]     spectral PDF
 *    [21]     bdpt_subpath_id (uintBitsToFloat)
 *    [22..53] amp_re[MAX_GPU_BANDS]
 *    [54..85] amp_im[MAX_GPU_BANDS]
 *
 *  HitBuf  (REFINED_HIT_STRIDE = 29 + 2*MAX_GPU_BANDS floats = 93 with MAX_GPU_BANDS=32, written by this shader):
 *    [0..2]   refined_pos xyz  (= hit_pos, T2 may update for parametric)
 *    [3..5]   refined_n xyz    (= oriented tri normal, T2 may update)
 *    [6..8]   incoming_dir xyz
 *    [9..11]  seg_start xyz    (= intent.pos)
 *    [12]     path_len         (= intent.path_len + t_hit)
 *    [13]     path_at_seg_start (= intent.path_len)
 *    [14]     hit_tri         (intBitsToFloat)
 *    [15]     mat_idx         (intBitsToFloat)
 *    [16]     color_flag      (uintBitsToFloat)
 *    [17]     bounce          (intBitsToFloat)
 *    [18]     bounces_left    (intBitsToFloat)
 *    [19]     min_amplitude
 *    [20]     src_id          (intBitsToFloat)
 *    [21]     tag_lo          (uintBitsToFloat)
 *    [22]     tag_hi          (uintBitsToFloat)
 *    [23]     sensor_origin_y
 *    [24]     sensor_origin_z
 *    [25]     medium_mat_idx  (intBitsToFloat)
 *    [26..57] amp_re[MAX_GPU_BANDS]  (after amplitude propagation)
 *    [58..89] amp_im[MAX_GPU_BANDS]
 *    [26+2*MAX_GPU_BANDS=90]  bdpt_subpath_id (uintBitsToFloat; 0=untracked)
 *
 *  BvhBuf  (BVH_NODE_STRIDE = 10 floats):
 *    [0..2] lo xyz   [3..5] hi xyz
 *    [6] left (int)  [7] right (int)  [8] tri_start (int)  [9] tri_end (int)
 *    left == -1 → leaf node
 *
 *  TriIdBuf  (int per entry): BVH permutation → triangle index
 *
 *  TriFullBuf  (TRI_FULL_STRIDE = 16 floats):
 *    [0..2]  v0 xyz
 *    [3..5]  edge1 xyz  (v1-v0)
 *    [6..8]  edge2 xyz  (v2-v0)
 *    [9..11] normal xyz
 *    [12]    flags         (intBitsToFloat)
 *    [13]    mat_idx       (intBitsToFloat)
 *    [14]    medium_pos_mat_idx (intBitsToFloat)
 *    [15]    medium_neg_mat_idx (intBitsToFloat)
 *
 *  MatBandBuf  (MAT_FULL_BANDS=32 bands × 12 floats per material):
 *    mat_band(m,b,field) = mat_bands[(m * 32 + b) * 12 + field]
 *    field 7 = n_real, field 8 = n_imag
 *
 *  SceneBandBuf  (2 × n_bands + 4 × n_arenas floats):
 *    [0..nb-1]     k_real[b]    (= 2π*freq[b]/c)
 *    [nb..2nb-1]   atmo_abs[b]
 *    [2nb..]        arena center xyz + radius
 *
 *  CounterBuf (24 uint words; shared pipeline control):
 *    [0]=hit_count  [1]=miss_count  [2]=wave_count
 *    [8]=live intent count  [22]=wave-tail overflow count
 *
 *  WaveIntent tail (starts at wave_base_floats in HitBuf):
 *    [0] arena_id, [1..3] entry position, [4..6] direction,
 *    [7] path_len, [8] medium, [9] interaction_flags, [10] src_id,
 *    [11] bounce, [12] bounces_left, [13] min_amplitude,
 *    [14..15] tag lo/hi, [16] packed color/spectral metadata,
 *    [17] priority, [18..19] sensor origin, [20] frequency, [21] PDF,
 *    [22] bdpt_subpath_id, then n_bands real and n_bands imaginary values.
 */
#version 430 core

layout(local_size_x = 64) in;

/* ── Binding constants ──────────────────────────────────────────────────── */
#define MAX_GPU_BANDS     32
#define INTENT_STRIDE     (22 + 2*MAX_GPU_BANDS)
#define HIT_STRIDE        (29 + 2*MAX_GPU_BANDS)
#define BVH_NODE_STRIDE   10
#define TRI_FULL_STRIDE   16
#define MAT_BAND_STRIDE   12
#define MAT_FULL_BANDS    32      /* MAX_SPECTRAL_BANDS in C++ */

#define T_SELF            1e-4    /* self-intersection guard                   */
#define BVH_STACK_SIZE    64      /* max BVH depth                             */
#define EPS               1e-10

/* Triangle flags — must match mat_flags_generated.h */
#define MAT_FLAG_EMISSIVE         1u
#define MAT_FLAG_REACTIVE         2u
#define MAT_FLAG_ABSORBER         4u
#define MAT_FLAG_TRANSMISSIVE    64u
#define MAT_FLAG_APERTURE_STOP  128u

/* ── SSBOs ──────────────────────────────────────────────────────────────── */

layout(std430, binding = 0) readonly buffer IntentBuf  { float intents[];  };
layout(std430, binding = 1) coherent  buffer HitBuf    { float hits[];     };
layout(std430, binding = 2) coherent  buffer CounterBuf{ uint  counters[]; }; /* [0]=hit [1]=miss [2]=wave */
layout(std430, binding = 3) readonly buffer BvhBuf     { float bvh[];      };
layout(std430, binding = 4) readonly buffer TriIdBuf   { int   tri_ids[];  };
layout(std430, binding = 5) readonly buffer TriFullBuf { float trifull[];  };
layout(std430, binding = 6) readonly buffer MatBandBuf { float mat_bands[];};
layout(std430, binding = 7) readonly buffer SceneBandBuf{ float scene_bands[];};
/* bdpt_subpath_id is written to hit[26+2*MAX_GPU_BANDS] (as uintBitsToFloat) — no extra binding needed. */

/* ── Uniforms ───────────────────────────────────────────────────────────── */
/* n_intents is now read from counters[8] (written by post_t3_prep for bounce>0
 * and by the C++ host for bounce 0).  The uniform is kept as a compile-time
 * fallback (value -1) so old pipelined paths that do not set counters[8]
 * continue to work via the counters[0]-based guard in T3. */
uniform int   n_intents;  /* unused at runtime — counters[8] is the live source */
uniform int   n_bands;
uniform int   n_mats;
uniform int   n_arenas;
uniform int   n_tris;
uniform int   wave_base_floats;
uniform int   wave_stride;
uniform int   wave_capacity;

/* ── Helpers: load intent field ─────────────────────────────────────────── */

float intent_f(int base, int i)  { return intents[base + i]; }
int   intent_i(int base, int i)  { return floatBitsToInt(intents[base + i]); }
uint  intent_u(int base, int i)  { return floatBitsToUint(intents[base + i]); }

/* ── Helpers: write hit field ───────────────────────────────────────────── */

void hit_wf(int base, int i, float v)  { hits[base + i] = v; }
void hit_wi(int base, int i, int v)    { hits[base + i] = intBitsToFloat(v); }
void hit_wu(int base, int i, uint v)   { hits[base + i] = uintBitsToFloat(v); }

/* ── BVH helpers ────────────────────────────────────────────────────────── */

vec3 bvh_lo(int n) {
    int b = n * BVH_NODE_STRIDE;
    return vec3(bvh[b], bvh[b+1], bvh[b+2]);
}
vec3 bvh_hi(int n) {
    int b = n * BVH_NODE_STRIDE;
    return vec3(bvh[b+3], bvh[b+4], bvh[b+5]);
}
int bvh_left(int n)      { return floatBitsToInt(bvh[n * BVH_NODE_STRIDE + 6]); }
int bvh_right(int n)     { return floatBitsToInt(bvh[n * BVH_NODE_STRIDE + 7]); }
int bvh_tri_start(int n) { return floatBitsToInt(bvh[n * BVH_NODE_STRIDE + 8]); }
int bvh_tri_end(int n)   { return floatBitsToInt(bvh[n * BVH_NODE_STRIDE + 9]); }

/* AABB slab intersection; returns t_enter < t_exit or t_exit < 0 for miss. */
float aabb_hit(vec3 lo, vec3 hi, vec3 org, vec3 inv_dir, float t_max) {
    vec3 t0 = (lo - org) * inv_dir;
    vec3 t1 = (hi - org) * inv_dir;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    float t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_exit < T_SELF || t_enter > t_exit || t_enter > t_max) return -1.0;
    /* Clamp to 0: negative t_enter means the ray origin is inside the box.
     * The node still intersects the ray — do not skip it. */
    return max(t_enter, 0.0);
}

/* Möller-Trumbore ray-triangle intersection. Returns t or -1 on miss. */
float moller_trumbore(vec3 v0, vec3 e1, vec3 e2, vec3 org, vec3 dir, out float u, out float v) {
    vec3 h  = cross(dir, e2);
    float a = dot(e1, h);
    if (abs(a) < 1e-9) { u = v = 0.0; return -1.0; }
    float f  = 1.0 / a;
    vec3  s  = org - v0;
    u        = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { v = 0.0; return -1.0; }
    vec3  q  = cross(s, e1);
    v        = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) return -1.0;
    float t  = f * dot(e2, q);
    return (t > T_SELF) ? t : -1.0;
}

/* ── Material band accessors ─────────────────────────────────────────────── */

float mat_n_real_gpu(int mat, int b) {
    if (mat < 0 || mat >= n_mats) return 1.0;
    int off = (mat * MAT_FULL_BANDS + b) * MAT_BAND_STRIDE + 7;
    float n = mat_bands[off];
    return (n >= 1.0) ? n : 1.0;
}

float mat_n_imag_gpu(int mat, int b) {
    if (mat < 0 || mat >= n_mats) return 0.0;
    int off = (mat * MAT_FULL_BANDS + b) * MAT_BAND_STRIDE + 8;
    return mat_bands[off];
}

float mat_field_at_frequency(int mat, int lane, float frequency_hz, int field) {
    if (mat < 0 || mat >= n_mats || frequency_hz <= 0.0)
        return mat_bands[(max(mat, 0) * MAT_FULL_BANDS + lane) * MAT_BAND_STRIDE + field];
    int last = 0;
    while (last + 1 < MAT_FULL_BANDS
           && mat_bands[(mat * MAT_FULL_BANDS + last + 1) * MAT_BAND_STRIDE] > 0.0) last++;
    if (last == 0) return mat_bands[(mat * MAT_FULL_BANDS) * MAT_BAND_STRIDE + field];
    bool increasing = mat_bands[(mat * MAT_FULL_BANDS + last) * MAT_BAND_STRIDE]
                    >= mat_bands[(mat * MAT_FULL_BANDS) * MAT_BAND_STRIDE];
    int lo = 0;
    while (lo + 1 < last) {
        float next_f = mat_bands[(mat * MAT_FULL_BANDS + lo + 1) * MAT_BAND_STRIDE];
        if ((increasing && next_f >= frequency_hz) || (!increasing && next_f <= frequency_hz)) break;
        lo++;
    }
    int hi = min(last, lo + 1);
    int a = (mat * MAT_FULL_BANDS + lo) * MAT_BAND_STRIDE;
    int b = (mat * MAT_FULL_BANDS + hi) * MAT_BAND_STRIDE;
    float denom = mat_bands[b] - mat_bands[a];
    float t = abs(denom) > 1e-20 ? clamp((frequency_hz - mat_bands[a]) / denom, 0.0, 1.0) : 0.0;
    return mix(mat_bands[a + field], mat_bands[b + field], t);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

void main() {
    uint gid = gl_GlobalInvocationID.x;
    /* counters[8] = n_intents: written by C++ host before every T1 dispatch.
     * gpu-resident bounce>0: post_t3_prep updates it to nc after T3. */
    if (gid >= counters[8]) return;

    int ib = int(gid) * INTENT_STRIDE;

    /* Load intent */
    vec3 pos   = vec3(intent_f(ib, 0), intent_f(ib, 1), intent_f(ib, 2));
    vec3 dir   = vec3(intent_f(ib, 3), intent_f(ib, 4), intent_f(ib, 5));
    float path_len = intent_f(ib, 6);
    int   medium_mat = intent_i(ib, 7);
    uint  iflags     = intent_u(ib, 8);
    int   src_id     = intent_i(ib, 9);
    int   bounce     = intent_i(ib, 10);
    int   bounces_left = intent_i(ib, 11);
    float min_amp    = intent_f(ib, 12);
    uint  tag_lo     = intent_u(ib, 13);
    uint  tag_hi     = intent_u(ib, 14);
    /* Low byte is the ordinary color flag. Upper bits are a zero-cost GPU
     * sidecar: bits 8..13 source lane, bit 14 continuous-resolved. */
    uint  color_meta = intent_u(ib, 15);
    uint  color_flag = color_meta & 0xffu;
    float priority   = intent_f(ib, 16);
    float sensor_oy  = intent_f(ib, 17);
    float sensor_oz  = intent_f(ib, 18);
    float spectral_frequency_hz = intent_f(ib, 19);
    float spectral_pdf = intent_f(ib, 20);
    uint  bdpt_sid   = intent_u(ib, 21);

    float amp_re[MAX_GPU_BANDS];
    float amp_im[MAX_GPU_BANDS];
    for (int b = 0; b < MAX_GPU_BANDS; ++b) {
        amp_re[b] = (b < n_bands) ? intent_f(ib, 22 + b)              : 0.0;
        amp_im[b] = (b < n_bands) ? intent_f(ib, 22 + MAX_GPU_BANDS + b) : 0.0;
    }

    /* Normalise direction defensively */
    float dir_len = length(dir);
    if (dir_len < 1e-9) {
        atomicAdd(counters[1], 1u);
        return;
    }
    dir = dir / dir_len;

    /* ── Iterative BVH traversal ─────────────────────────────────────────── */
    vec3 inv_dir = vec3(1.0) / dir;

    int  stack[BVH_STACK_SIZE];
    int  sp       = 0;
    int  hit_tri  = -1;
    float best_t   = 1e30;
    float best_u   = 0.0;
    float best_v   = 0.0;

    /* Seed the nearest-event distance with the nearest arena boundary. BVH
     * traversal replaces it only when authored geometry is closer. */
    int wave_arena_id = -1;
    for (int ai = 0; ai < n_arenas; ++ai) {
        int ab = 2 * n_bands + 4 * ai;
        vec3 center = vec3(scene_bands[ab], scene_bands[ab+1], scene_bands[ab+2]);
        vec3 oc = pos - center;
        float radius = scene_bands[ab+3];
        float qb = dot(oc, dir);
        float qc = dot(oc, oc) - radius * radius;
        float disc = qb * qb - qc;
        if (disc < 0.0) continue;
        float root = sqrt(disc);
        float candidate = -qb - root;
        if (candidate <= T_SELF) candidate = -qb + root;
        if (candidate > T_SELF && candidate < best_t) {
            best_t = candidate;
            wave_arena_id = ai;
        }
    }

    stack[sp++] = 0;  /* root node */

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;

        vec3 lo = bvh_lo(node);
        vec3 hi = bvh_hi(node);
        float t_aabb = aabb_hit(lo, hi, pos, inv_dir, best_t);
        if (t_aabb < 0.0) continue;

        int left  = bvh_left(node);
        int right = bvh_right(node);

        if (left == -1) {
            /* Leaf: test all triangles */
            int ts = bvh_tri_start(node);
            int te = bvh_tri_end(node);
            for (int k = ts; k < te; ++k) {
                int tri_idx = tri_ids[k];
                int tb = tri_idx * TRI_FULL_STRIDE;
                vec3 v0 = vec3(trifull[tb],   trifull[tb+1], trifull[tb+2]);
                vec3 e1 = vec3(trifull[tb+3], trifull[tb+4], trifull[tb+5]);
                vec3 e2 = vec3(trifull[tb+6], trifull[tb+7], trifull[tb+8]);
                float u, v;
                float t = moller_trumbore(v0, e1, e2, pos, dir, u, v);
                if (t > T_SELF && t < best_t) {
                    best_t   = t;
                    best_u   = u;
                    best_v   = v;
                    hit_tri  = tri_idx;
                }
            }
        } else {
            /* Interior: push children */
            if (sp < BVH_STACK_SIZE - 1) stack[sp++] = left;
            if (sp < BVH_STACK_SIZE - 1) stack[sp++] = right;
        }
    }

    /* ── Miss ───────────────────────────────────────────────────────────── */
    if (hit_tri < 0 && wave_arena_id < 0) {
        atomicAdd(counters[1], 1u);
        return;
    }

    /* ── Hit: amplitude propagation ─────────────────────────────────────── */
    float tot_len = path_len + best_t;
    vec3  hit_pos = pos + best_t * dir;

    int nb = min(n_bands, MAX_GPU_BANDS);
    for (int b = 0; b < nb; ++b) {
        float n_re = (medium_mat >= 0) ? mat_field_at_frequency(medium_mat, b, spectral_frequency_hz, 7) : 1.0;
        float n_im = (medium_mat >= 0) ? mat_field_at_frequency(medium_mat, b, spectral_frequency_hz, 8) : 0.0;
        float k_real_b = spectral_frequency_hz > 0.0
            ? 6.283185307179586 * spectral_frequency_hz / 299792458.0
            : scene_bands[b];
        float atmo_b   = scene_bands[n_bands + b];
        float k_med    = k_real_b * n_re;
        float alpha    = atmo_b + k_real_b * n_im;
        float atten    = exp(-alpha * best_t);
        /* Backward rays (color_flag==1): no geometric spreading (importance sampling) */
        float spread   = ((color_flag & 1u) != 0u) ? 1.0 : 1.0 / (1.0 + tot_len);
        float amp_scale = atten * spread;
        float phase     = -k_med * best_t;
        float cos_p     = cos(phase);
        float sin_p     = sin(phase);
        float re = amp_re[b];
        float im = amp_im[b];
        amp_re[b] = amp_scale * (re * cos_p - im * sin_p);
        amp_im[b] = amp_scale * (re * sin_p + im * cos_p);
    }

    /* ── Wave-port event ─────────────────────────────────────────────────── */
    if (hit_tri < 0 && wave_arena_id >= 0) {
        uint wave_idx = atomicAdd(counters[2], 1u);
        if (wave_idx >= uint(wave_capacity)) {
            atomicAdd(counters[22], 1u);
            return;
        }
        int wb = wave_base_floats + int(wave_idx) * wave_stride;
        hits[wb +  0] = uintBitsToFloat(uint(wave_arena_id));
        hits[wb +  1] = hit_pos.x; hits[wb +  2] = hit_pos.y; hits[wb +  3] = hit_pos.z;
        hits[wb +  4] = dir.x;     hits[wb +  5] = dir.y;     hits[wb +  6] = dir.z;
        hits[wb +  7] = tot_len;
        hits[wb +  8] = intBitsToFloat(medium_mat);
        hits[wb +  9] = uintBitsToFloat(iflags | (1u << 1u));
        hits[wb + 10] = intBitsToFloat(src_id);
        hits[wb + 11] = intBitsToFloat(bounce);
        hits[wb + 12] = intBitsToFloat(bounces_left);
        hits[wb + 13] = min_amp;
        hits[wb + 14] = uintBitsToFloat(tag_lo);
        hits[wb + 15] = uintBitsToFloat(tag_hi);
        hits[wb + 16] = uintBitsToFloat(color_meta);
        hits[wb + 17] = priority;
        hits[wb + 18] = sensor_oy;
        hits[wb + 19] = sensor_oz;
        hits[wb + 20] = spectral_frequency_hz;
        hits[wb + 21] = spectral_pdf;
        hits[wb + 22] = uintBitsToFloat(bdpt_sid);
        for (int b = 0; b < nb; ++b) {
            hits[wb + 23 + b]      = amp_re[b];
            hits[wb + 23 + nb + b] = amp_im[b];
        }
        return;
    }

    /* ── Triangle data ───────────────────────────────────────────────────── */
    int tb = hit_tri * TRI_FULL_STRIDE;
    vec3 tri_n = vec3(trifull[tb+9], trifull[tb+10], trifull[tb+11]);
    int  tri_flags   = floatBitsToInt(trifull[tb + 12]);
    int  tri_mat     = floatBitsToInt(trifull[tb + 13]);

    /* Orient normal toward incoming ray (front-face convention) */
    if (dot(dir, tri_n) > 0.0) tri_n = -tri_n;

    /* ── Write hit record to HitBuf ─────────────────────────────────────── */
    uint out_idx = atomicAdd(counters[0], 1u);
    int ob = int(out_idx) * HIT_STRIDE;

    hit_wf(ob, 0, hit_pos.x);  hit_wf(ob, 1, hit_pos.y);  hit_wf(ob, 2, hit_pos.z);
    hit_wf(ob, 3, tri_n.x);    hit_wf(ob, 4, tri_n.y);    hit_wf(ob, 5, tri_n.z);
    hit_wf(ob, 6, dir.x);      hit_wf(ob, 7, dir.y);      hit_wf(ob, 8, dir.z);
    hit_wf(ob, 9,  pos.x);     hit_wf(ob, 10, pos.y);     hit_wf(ob, 11, pos.z);
    hit_wf(ob, 12, tot_len);
    hit_wf(ob, 13, path_len);
    hit_wi(ob, 14, hit_tri);
    hit_wi(ob, 15, tri_mat);
    hit_wu(ob, 16, color_meta);
    hit_wi(ob, 17, bounce);
    hit_wi(ob, 18, bounces_left);
    hit_wf(ob, 19, min_amp);
    hit_wi(ob, 20, src_id);
    hit_wu(ob, 21, tag_lo);
    hit_wu(ob, 22, tag_hi);
    hit_wf(ob, 23, sensor_oy);
    hit_wf(ob, 24, sensor_oz);
    hit_wi(ob, 25, medium_mat);
    hit_wf(ob, 26, spectral_frequency_hz);
    hit_wf(ob, 27, spectral_pdf);

    for (int b = 0; b < MAX_GPU_BANDS; ++b) {
        hit_wf(ob, 28 + b,              amp_re[b]);
        hit_wf(ob, 28 + MAX_GPU_BANDS + b, amp_im[b]);
    }
    hit_wu(ob, 28 + 2*MAX_GPU_BANDS, bdpt_sid);

}
