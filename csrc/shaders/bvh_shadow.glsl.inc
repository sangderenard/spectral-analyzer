/* bvh_shadow.glsl.inc — Shared BVH shadow-ray helper for GPU compute shaders.
 *
 * This file is injected as a preamble (after the #version line) into any
 * compute shader that needs shadow/visibility queries against the scene BVH.
 *
 * The caller must #define three binding-point macros BEFORE this file is
 * injected:
 *
 *   #define SHADOW_BVH_BINDING       N   // BvhBuf     (float[],  10 floats/node)
 *   #define SHADOW_TRI_ID_BINDING    N   // TriIdBuf   (int[])
 *   #define SHADOW_TRI_FULL_BINDING  N   // TriFullBuf (float[], 16 floats/tri)
 *
 * Layout constants (must match C++ BVHNode / Triangle packing):
 *
 *   BvhBuf per node  (BVH_NODE_STRIDE = 10):
 *     [0..2] lo xyz   [3..5] hi xyz
 *     [6] left(int)   [7] right(int)   [8] tri_start(int)   [9] tri_end(int)
 *     left == -1 → leaf node
 *
 *   TriFullBuf per tri  (TRI_FULL_STRIDE = 16):
 *     [0..2]  v0 xyz
 *     [3..5]  edge1 xyz  (v1 - v0)
 *     [6..8]  edge2 xyz  (v2 - v0)
 *     [9..11] normal xyz
 *     [12]    flags         (intBitsToFloat)
 *     [13]    mat_idx       (intBitsToFloat)
 *     [14]    medium_pos_mat_idx
 *     [15]    medium_neg_mat_idx
 *
 * Transmissive triangles (MAT_FLAG_TRANSMISSIVE = 64u) are transparent to
 * shadow rays — the function skips them so that glass lens elements do not
 * incorrectly occlude BDPT connections between a camera vertex on one side
 * of a lens and a light vertex on the other.
 *
 * Usage:
 *   bool occluded = shadow_occluded(cam_pos, light_pos);
 *   if (occluded) continue;
 */

#ifndef BVH_NODE_STRIDE
#  define BVH_NODE_STRIDE  10
#endif
#ifndef TRI_FULL_STRIDE
#  define TRI_FULL_STRIDE  16
#endif
#ifndef SHADOW_T_SELF
#  define SHADOW_T_SELF    1e-4
#endif
#ifndef SHADOW_STACK_SIZE
#  define SHADOW_STACK_SIZE 64
#endif

/* Triangle flags — must match mat_flags_generated.h */
#ifndef SHADOW_MAT_FLAG_TRANSMISSIVE
#  define SHADOW_MAT_FLAG_TRANSMISSIVE  64u
#endif
#ifndef SHADOW_MAT_FLAG_APERTURE_STOP
#  define SHADOW_MAT_FLAG_APERTURE_STOP 128u
#endif

/* ── SSBO declarations ──────────────────────────────────────────────────── */

layout(std430, binding = SHADOW_BVH_BINDING)      readonly buffer ShadowBvhBuf     { float shadow_bvh[];      };
layout(std430, binding = SHADOW_TRI_ID_BINDING)   readonly buffer ShadowTriIdBuf   { int   shadow_tri_ids[];  };
layout(std430, binding = SHADOW_TRI_FULL_BINDING) readonly buffer ShadowTriFullBuf { float shadow_trifull[];  };

/* ── BVH node accessors ─────────────────────────────────────────────────── */

vec3 _shd_bvh_lo(int n)       { int b = n * BVH_NODE_STRIDE; return vec3(shadow_bvh[b],   shadow_bvh[b+1], shadow_bvh[b+2]); }
vec3 _shd_bvh_hi(int n)       { int b = n * BVH_NODE_STRIDE; return vec3(shadow_bvh[b+3], shadow_bvh[b+4], shadow_bvh[b+5]); }
int  _shd_bvh_left(int n)     { return floatBitsToInt(shadow_bvh[n * BVH_NODE_STRIDE + 6]); }
int  _shd_bvh_right(int n)    { return floatBitsToInt(shadow_bvh[n * BVH_NODE_STRIDE + 7]); }
int  _shd_bvh_tri_start(int n){ return floatBitsToInt(shadow_bvh[n * BVH_NODE_STRIDE + 8]); }
int  _shd_bvh_tri_end(int n)  { return floatBitsToInt(shadow_bvh[n * BVH_NODE_STRIDE + 9]); }

/* ── AABB slab test ─────────────────────────────────────────────────────── */

bool _shd_aabb_hit(vec3 lo, vec3 hi, vec3 org, vec3 inv_dir, float t_max) {
    vec3 t0 = (lo - org) * inv_dir;
    vec3 t1 = (hi - org) * inv_dir;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    float t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);
    return (t_exit >= SHADOW_T_SELF && t_enter <= t_exit && t_enter <= t_max);
}

/* ── Möller-Trumbore (shadow variant — no u/v output needed) ────────────── */

float _shd_mt(vec3 v0, vec3 e1, vec3 e2, vec3 org, vec3 dir) {
    vec3  h = cross(dir, e2);
    float a = dot(e1, h);
    if (abs(a) < 1e-9) return -1.0;
    float f = 1.0 / a;
    vec3  s = org - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return -1.0;
    vec3  q = cross(s, e1);
    float v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) return -1.0;
    float t = f * dot(e2, q);
    return (t > SHADOW_T_SELF) ? t : -1.0;
}

/* ── shadow_occluded(p0, p1) ────────────────────────────────────────────── *
 * Returns true if any opaque geometry lies strictly between p0 and p1.      *
 * Transmissive triangles (glass lens elements) are skipped so that BDPT     *
 * connections are not falsely blocked by refractive surfaces.               *
 * Aperture-stop triangles are treated as opaque — they are physical blockers.*/

int shadow_first_blocker_except(
    vec3 p0, vec3 p1, int skip_tri0, int skip_tri1,
    out float blocker_t, out float segment_dist
) {
    vec3  d    = p1 - p0;
    float dist = length(d);
    blocker_t = -1.0;
    segment_dist = dist;
    if (dist < 2.0 * SHADOW_T_SELF) return -1;      /* degenerate segment */
    vec3 dir     = d / dist;
    vec3 inv_dir = vec3(1.0) / dir;
    float t_max  = dist - SHADOW_T_SELF;             /* don't hit p1 itself */

    int stack[SHADOW_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  /* root */

    while (sp > 0) {
        int node = stack[--sp];
        if (node < 0) continue;

        if (!_shd_aabb_hit(_shd_bvh_lo(node), _shd_bvh_hi(node),
                           p0, inv_dir, t_max)) continue;

        int left = _shd_bvh_left(node);
        if (left == -1) {
            /* Leaf — test each triangle */
            int ts = _shd_bvh_tri_start(node);
            int te = _shd_bvh_tri_end(node);
            for (int k = ts; k < te; ++k) {
                int ti  = shadow_tri_ids[k];
                if (ti == skip_tri0 || ti == skip_tri1) continue;
                int tb  = ti * TRI_FULL_STRIDE;
                uint tf = floatBitsToUint(shadow_trifull[tb + 12]);
                /* Skip transmissive surfaces — glass does not block visibility for
                 * BDPT connections.  The refractive bend (Snell) and attenuation
                 * (Fresnel) of a segment crossing glass is accounted for separately
                 * by the optical-connection path in t5_full_connect, which solves
                 * for the true refracted route rather than the straight segment. */
                if ((tf & SHADOW_MAT_FLAG_TRANSMISSIVE) != 0u) continue;
                vec3 v0 = vec3(shadow_trifull[tb],   shadow_trifull[tb+1], shadow_trifull[tb+2]);
                vec3 e1 = vec3(shadow_trifull[tb+3], shadow_trifull[tb+4], shadow_trifull[tb+5]);
                vec3 e2 = vec3(shadow_trifull[tb+6], shadow_trifull[tb+7], shadow_trifull[tb+8]);
                float t = _shd_mt(v0, e1, e2, p0, dir);
                if (t > SHADOW_T_SELF && t < t_max) {
                    blocker_t = t;
                    return ti;
                }
            }
        } else {
            if (sp + 1 < SHADOW_STACK_SIZE) {
                stack[sp++] = left;
                stack[sp++] = _shd_bvh_right(node);
            }
        }
    }
    return -1;
}

bool shadow_occluded_except(vec3 p0, vec3 p1, int skip_tri0, int skip_tri1) {
    float blocker_t;
    float segment_dist;
    return shadow_first_blocker_except(
        p0, p1, skip_tri0, skip_tri1, blocker_t, segment_dist
    ) >= 0;
}

bool shadow_occluded(vec3 p0, vec3 p1) {
    return shadow_occluded_except(p0, p1, -1, -1);
}
