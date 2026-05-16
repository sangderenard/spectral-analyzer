/*
 * ray_refine.comp.glsl — GPU compute T2: parametric surface refinement.
 *
 * One invocation per HitRecord (REFINED_HIT_STRIDE layout, written by T1).
 * For most triangles (TRI_PARAM_SURFACE_NONE) this is a no-op — T1 already
 * emits the correct oriented normal.  For parametric surfaces this shader
 * updates hit_pos and hit_n in-place using the surface's height-field Jacobian.
 *
 * Parametric kinds (must match C++ triangle_groups.h):
 *   0 = NONE       — passthrough (flip normal if needed — already done by T1)
 *   1 = POLY_BARY  — polynomial height: delta = c0+cu*u+cv*v+cuu*uu+cuv*uv+cvv*vv
 *   2 = SDF_SADDLE — (falls back to CPU; mark bit set in counter[3])
 *   3 = SDF_SPHERE — spherical cap: delta = k*(cu²+cv²), k=0.5/radius
 *
 * Flat buffer layouts (same bindings as T1 but shared HitBuf is now input):
 *
 *  HitBuf  (REFINED_HIT_STRIDE = 58 floats): in-place update of [0..5]
 *
 *  TriFullBuf  (TRI_FULL_STRIDE = 16 floats):
 *    [0..2] v0   [3..5] edge1  [6..8] edge2  [9..11] normal
 *    [12] flags  [13] mat_idx  [14] medium_pos  [15] medium_neg
 *
 *  TriParamGroupBuf  (int per triangle): parametric group_id, -1 if none.
 *
 *  GroupKindBuf  (GROUP_STRIDE = 1 int): kind code per group.
 *
 *  GroupPayloadBuf  (GROUP_PAYLOAD_STRIDE = 16 floats per group):
 *    SDF_SPHERE:  [0]=radius  [1]=margin  (originally float64 but stored as float32)
 *    POLY_BARY:   [0..5] = {c0, cu, cv, cuu, cuv, cvv}
 *
 *  CounterBuf:
 *    [0] = hit count  (from T1, used as n_hits here)
 *    [3] = cpu_refine count  (incremented for complex-kind fallbacks)
 */
#version 430 core

layout(local_size_x = 64) in;

/* ── Layout constants ───────────────────────────────────────────────────── */
#define MAX_GPU_BANDS         16
#define HIT_STRIDE            58
#define TRI_FULL_STRIDE       16
#define GROUP_PAYLOAD_STRIDE  16

/* Parametric kind codes */
#define PARAM_NONE        0
#define PARAM_POLY_BARY   1
#define PARAM_SDF_SADDLE  2
#define PARAM_SDF_SPHERE  3

#define EPS               1e-9

/* ── SSBOs ──────────────────────────────────────────────────────────────── */

layout(std430, binding = 0) coherent  buffer HitBuf           { float hits[];      };
layout(std430, binding = 1) readonly  buffer TriFullBuf        { float trifull[];   };
layout(std430, binding = 2) readonly  buffer TriParamGroupBuf  { int   tri_group[]; };
layout(std430, binding = 3) readonly  buffer GroupKindBuf      { int   group_kind[]; };
layout(std430, binding = 4) readonly  buffer GroupPayloadBuf   { float group_payload[]; };
layout(std430, binding = 5) coherent  buffer CounterBuf        { uint  counters[];  };

/* ── Uniforms ───────────────────────────────────────────────────────────── */
/* n_hits is now read from counters[0] written by T1 so the CPU can dispatch
 * T2 immediately after T1 without an intermediate GPU→CPU readback. */
uniform int n_tris;
uniform int n_groups;

/* ── Helpers ────────────────────────────────────────────────────────────── */

float hit_f(int base, int i) { return hits[base + i]; }
void  hit_wf(int base, int i, float v) { hits[base + i] = v; }

/* ── SDF_SPHERE height-field + Jacobian normal ──────────────────────────── */
/*
 * Surface parameterised in barycentric UV:
 *   S(u,v) = v0 + u*edge1 + v*edge2 + N0 * delta(u,v)
 *   delta(u,v) = k * (cu² + cv²),   cu = u - 1/3,  cv = v - 1/3
 *   k = 0.5 / radius
 * Normal via Jacobian:
 *   Su = edge1 + N0 * d(delta)/du = edge1 + N0 * 2k*cu
 *   Sv = edge2 + N0 * d(delta)/dv = edge2 + N0 * 2k*cv
 *   N  = normalize(Su × Sv)
 */
void refine_sphere(in vec3 v0, in vec3 e1, in vec3 e2, in vec3 n0,
                    float radius,
                    inout vec3 pos, inout vec3 n)
{
    float k = 0.5 / ((abs(radius) < EPS) ? 0.12 : radius);

    /* Recover barycentric UV from hit_pos */
    vec3  bq   = pos - v0;
    float d11  = dot(e1, e1);
    float d12  = dot(e1, e2);
    float d22  = dot(e2, e2);
    float d1q  = dot(e1, bq);
    float d2q  = dot(e2, bq);
    float den  = d11 * d22 - d12 * d12 + 1e-30;
    float u    = (d22 * d1q - d12 * d2q) / den;
    float v    = (d11 * d2q - d12 * d1q) / den;

    float cu   = u - (1.0 / 3.0);
    float cv   = v - (1.0 / 3.0);
    float delta= k * (cu * cu + cv * cv);
    float dzdu = 2.0 * k * cu;
    float dzdv = 2.0 * k * cv;

    vec3 Su    = e1 + n0 * dzdu;
    vec3 Sv    = e2 + n0 * dzdv;
    vec3 wn    = cross(Su, Sv);
    float wn_l = length(wn);
    if (wn_l < EPS)
        wn = n0;
    else
        wn = wn / wn_l;

    pos = pos + wn * delta;
    n   = wn;
}

/* ── POLY_BARY height-field + Jacobian normal ─────────────────────────── */
void refine_poly(in vec3 v0, in vec3 e1, in vec3 e2, in vec3 n0,
                  float c0, float cu_c, float cv_c,
                  float cuu, float cuv, float cvv,
                  inout vec3 pos, inout vec3 n)
{
    vec3  bq   = pos - v0;
    float d11  = dot(e1, e1);
    float d12  = dot(e1, e2);
    float d22  = dot(e2, e2);
    float d1q  = dot(e1, bq);
    float d2q  = dot(e2, bq);
    float den  = d11 * d22 - d12 * d12 + 1e-30;
    float u    = (d22 * d1q - d12 * d2q) / den;
    float v    = (d11 * d2q - d12 * d1q) / den;

    float delta = c0 + cu_c*u + cv_c*v + cuu*u*u + cuv*u*v + cvv*v*v;
    float dzdu  = cu_c + 2.0*cuu*u + cuv*v;
    float dzdv  = cv_c + 2.0*cvv*v + cuv*u;

    vec3 Su   = e1 + n0 * dzdu;
    vec3 Sv   = e2 + n0 * dzdv;
    vec3 wn   = cross(Su, Sv);
    float wn_l = length(wn);
    if (wn_l < EPS)
        wn = n0;
    else
        wn = wn / wn_l;

    pos = pos + wn * delta;
    n   = wn;
}

/* ── Main ───────────────────────────────────────────────────────────────── */

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= int(counters[0])) return;  /* counters[0] = T1 hit count */

    int hb = int(gid) * HIT_STRIDE;

    /* Read refined_pos and refined_n (fields set by T1) */
    vec3 pos = vec3(hit_f(hb, 0), hit_f(hb, 1), hit_f(hb, 2));
    vec3 n   = vec3(hit_f(hb, 3), hit_f(hb, 4), hit_f(hb, 5));
    int  hit_tri = floatBitsToInt(hit_f(hb, 14));

    /* Bounds check */
    if (hit_tri < 0 || hit_tri >= n_tris) return;

    /* Look up parametric group for this triangle */
    int gid_param = tri_group[hit_tri];
    if (gid_param < 0 || gid_param >= n_groups) {
        /* NONE kind: normal already oriented by T1, nothing to do */
        return;
    }

    int kind = group_kind[gid_param];

    if (kind == PARAM_NONE) {
        /* Normal already oriented by T1 */
        return;
    }

    if (kind == PARAM_SDF_SADDLE) {
        /* Complex kind: flag for CPU fallback */
        atomicAdd(counters[3], 1u);
        return;
    }

    /* Load triangle geometry */
    int tb = hit_tri * TRI_FULL_STRIDE;
    vec3 v0 = vec3(trifull[tb],   trifull[tb+1], trifull[tb+2]);
    vec3 e1 = vec3(trifull[tb+3], trifull[tb+4], trifull[tb+5]);
    vec3 e2 = vec3(trifull[tb+6], trifull[tb+7], trifull[tb+8]);
    vec3 n0 = vec3(trifull[tb+9], trifull[tb+10], trifull[tb+11]);

    /* Read incoming_dir for final normal orientation */
    vec3 inc_dir = vec3(hit_f(hb, 6), hit_f(hb, 7), hit_f(hb, 8));

    int   pb = gid_param * GROUP_PAYLOAD_STRIDE;

    if (kind == PARAM_SDF_SPHERE) {
        float radius = group_payload[pb];
        refine_sphere(v0, e1, e2, n0, radius, pos, n);
    } else if (kind == PARAM_POLY_BARY) {
        float c0  = group_payload[pb + 0];
        float cu_c = group_payload[pb + 1];
        float cv_c = group_payload[pb + 2];
        float cuu = group_payload[pb + 3];
        float cuv = group_payload[pb + 4];
        float cvv = group_payload[pb + 5];
        refine_poly(v0, e1, e2, n0, c0, cu_c, cv_c, cuu, cuv, cvv, pos, n);
    } else {
        /* Unknown kind: CPU fallback */
        atomicAdd(counters[3], 1u);
        return;
    }

    /* Re-orient refined normal toward incoming ray */
    if (dot(inc_dir, n) > 0.0) n = -n;

    /* Write back updated pos and normal */
    hit_wf(hb, 0, pos.x); hit_wf(hb, 1, pos.y); hit_wf(hb, 2, pos.z);
    hit_wf(hb, 3, n.x);   hit_wf(hb, 4, n.y);   hit_wf(hb, 5, n.z);
}
