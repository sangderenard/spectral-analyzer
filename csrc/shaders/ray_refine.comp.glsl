/*
 * ray_refine.comp.glsl — GPU compute T2: parametric surface refinement.
 *
 * One invocation per HitRecord (REFINED_HIT_STRIDE layout, written by T1).
 * For most triangles (TRI_PARAM_SURFACE_NONE) this is a no-op — T1 already
 * emits the correct oriented normal.  For parametric surfaces this shader
 * updates hit_pos and hit_n in-place using the surface's height-field Jacobian.
 *
 * Parametric kinds (must match C++ triangle_groups.h):
 *   0 = NONE            — passthrough (flip normal if needed — already done by T1)
 *   1 = POLY_BARY       — polynomial height: delta = c0+cu*u+cv*v+cuu*uu+cuv*uv+cvv*vv
 *   2 = SDF_SADDLE      — (falls back to CPU; mark bit set in counter[3])
 *   3 = SDF_SPHERE      — spherical cap: delta = k*(cu²+cv²), k=0.5/radius
 *   4 = NEURAL_ASSEMBLY — deferred absorber; MLP transport is disabled until
 *                          it supplies reversible optical metadata
 *
 * Flat buffer layouts (same bindings as T1 but shared HitBuf is now input):
 *
 *  HitBuf  (REFINED_HIT_STRIDE = 27+2*MAX_GPU_BANDS floats): in-place update of [0..8], [12], [16], [26..26+2*MAX_GPU_BANDS-1]
 *    [26+2*MAX_GPU_BANDS] = bdpt_subpath_id, carried through untouched from T1 to T3
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
 *    SDF_SPHERE:       [0]=radius  [1]=margin
 *    POLY_BARY:        [0..5] = {c0, cu, cv, cuu, cuv, cvv}
 *    NEURAL_ASSEMBLY:  [0]=int offset into NeuralPayBuf (bit-cast via floatBitsToInt)
 *
 *  NeuralPayBuf  (binding 6): still present because PARAMETRIC_LENS shares
 *    this transport payload buffer.  NEURAL_ASSEMBLY no longer consumes it.
 *
 *  CounterBuf:
 *    [0] = hit count  (from T1, used as n_hits here)
 *    [3] = cpu_refine count  (incremented for complex-kind fallbacks)
 *    [4] = BDPT optical-event count
 *
 * NEURAL_ASSEMBLY policy:
 *   T2 sets bit 3 of color_flag (HitBuf[16]) to absorb the ray.  The intended
 *   future implementation must emit OPL, Jacobian, eta/cosines, and PDFs like
 *   the parametric lens path before it is allowed back into the camera pipeline.
 */
#version 430 core

layout(local_size_x = 64) in;

/* ── Layout constants ───────────────────────────────────────────────────── */
#define MAX_GPU_BANDS         32
#define HIT_STRIDE            (27 + 2*MAX_GPU_BANDS)  /* 91 with MAX_GPU_BANDS=32 */
#define TRI_FULL_STRIDE       16
#define GROUP_PAYLOAD_STRIDE  16
#define MAX_NEURAL_DIM        512   /* max hidden_dim supported */

/* Parametric kind codes */
#define PARAM_NONE             0
#define PARAM_POLY_BARY        1
#define PARAM_SDF_SADDLE       2
#define PARAM_SDF_SPHERE       3
#define PARAM_NEURAL_ASSEMBLY  4
#define PARAM_PARAMETRIC_LENS  5

/* Parametric lens payload layout (mirrors CompoundLens.build_gpu_payload) */
#define PLENS_MAGIC       14949.0
#define PLENS_N_SURF      1    /* header offset: n_surfaces */
#define PLENS_HOOD_R      2
#define PLENS_HOOD_XF     3    /* hood x_front */
#define PLENS_HOOD_XR     4    /* hood x_rim */
#define PLENS_HEADER      8    /* floats before first surface record */
#define PLENS_SURF_STRIDE 8
#define PLENS_MAX_SURF    24   /* max surfaces supported in shader */

/* Per-surface record field offsets within PLENS_SURF_STRIDE */
#define PLENS_S_XPOS   0
#define PLENS_S_RCURV  1
#define PLENS_S_NBEF   2
#define PLENS_S_NAFT   3
#define PLENS_S_APR    4
#define PLENS_S_CONIK  5
#define PLENS_S_FLAGS  6   /* bit 0 = is_stop */

/* Neural payload header layout offsets */
#define NPAY_N_LAYERS    1
#define NPAY_IN_DIM      2
#define NPAY_HIDDEN_DIM  3
#define NPAY_OUT_DIM     4
#define NPAY_Z_EXIT      6
#define NPAY_BND_C0      7   /* acceptance boundary constant term */
#define NPAY_BND_C1      8   /* acceptance boundary quadratic coeff */
#define NPAY_R_LENS     10   /* physical lens radius */
#define NPAY_AXIS_IDX   13
#define NPAY_IN_MEAN    16   /* 5 floats */
#define NPAY_IN_SCALE   21   /* 5 floats */
#define NPAY_OUT_MEAN   26   /* 6 floats */
#define NPAY_OUT_SCL    32   /* 6 floats */
#define NPAY_LAYER_OFF  38

#define NEURAL_MAGIC    14948.0
#define TWO_PI          6.283185307179586
#define SPEED_LIGHT     299792458.0

#define EPS             1e-9

/* ── SSBOs ──────────────────────────────────────────────────────────────── */

layout(std430, binding = 0) coherent  buffer HitBuf           { float hits[];       };
layout(std430, binding = 1) readonly  buffer TriFullBuf        { float trifull[];    };
layout(std430, binding = 2) readonly  buffer TriParamGroupBuf  { int   tri_group[];  };
layout(std430, binding = 3) readonly  buffer GroupKindBuf      { int   group_kind[]; };
layout(std430, binding = 4) readonly  buffer GroupPayloadBuf   { float group_payload[]; };
layout(std430, binding = 5) coherent  buffer CounterBuf        { uint  counters[];   };
layout(std430, binding = 6) readonly  buffer NeuralPayBuf      { float npay[];       };
layout(std430, binding = 7) coherent  buffer BdptOutputBuf     { float bdpt_out[];   };

/* ── Uniforms ───────────────────────────────────────────────────────────── */
uniform int   n_tris;
uniform int   n_groups;
uniform int   n_bands;
uniform float freq_hz[MAX_GPU_BANDS];
uniform int   bdpt_max_optical;
uniform int   bdpt_optical_base;

#define BDPT_OPTICAL_STRIDE        28
#define BDPT_OPT_REFRACTION         0u
#define BDPT_OPT_REFLECTION         1u
#define BDPT_OPT_TIR                2u
#define BDPT_OPT_APERTURE_CLIP      3u
#define BDPT_OPT_VIGNETTE_CLIP      4u
#define BDPT_OPT_ABSORPTION         5u

/* ── Helpers ────────────────────────────────────────────────────────────── */

float hit_f(int base, int i) { return hits[base + i]; }
void  hit_wf(int base, int i, float v) { hits[base + i] = v; }

uint hit_u(int base, int i) { return floatBitsToUint(hits[base + i]); }
int  hit_i(int base, int i) { return floatBitsToInt(hits[base + i]); }

float fresnel_R(float cos_i, float cos_t, float n1, float n2)
{
    float rs_num = n1*cos_i - n2*cos_t;
    float rs_den = n1*cos_i + n2*cos_t;
    float rp_num = n1*cos_t - n2*cos_i;
    float rp_den = n1*cos_t + n2*cos_i;
    float rs = (abs(rs_den) > EPS) ? rs_num / rs_den : 1.0;
    float rp = (abs(rp_den) > EPS) ? rp_num / rp_den : 1.0;
    return clamp(0.5 * (rs*rs + rp*rp), 0.0, 1.0);
}

void emit_bdpt_optical(int hb, uint reason, uint element_index, uint flags,
                       vec3 pos, vec3 nrm, vec3 dir_in, vec3 dir_out,
                       float cos_i, float cos_t, float eta_i, float eta_t,
                       float fresnel_r, float transmittance, float throughput,
                       float opl, float geom_len, float aperture_r,
                       float transverse_r, float dist_past_aperture,
                       float phase_space_j)
{
    uint sid = hit_u(hb, 26 + 2*MAX_GPU_BANDS);
    if (sid == 0u || bdpt_max_optical <= 0) return;
    uint slot = atomicAdd(counters[4], 1u);
    if (int(slot) >= bdpt_max_optical) return;

    uint vi = uint(max(0, hit_i(hb, 17))) & 0xFFFFu;
    uint cflag = hit_u(hb, 16);
    uint stream = (cflag == 1u) ? 1u : 0u;
    uint packed_ve = vi | ((element_index & 0xFFFFu) << 16);
    uint packed_rf = (reason & 0xFFu) | ((stream & 0xFFu) << 8) | ((flags & 0xFFFFu) << 16);

    uint ob = uint(bdpt_optical_base) + slot * uint(BDPT_OPTICAL_STRIDE);
    bdpt_out[ob +  0] = uintBitsToFloat(sid);
    bdpt_out[ob +  1] = uintBitsToFloat(packed_ve);
    bdpt_out[ob +  2] = uintBitsToFloat(packed_rf);
    bdpt_out[ob +  3] = pos.x;       bdpt_out[ob +  4] = pos.y;       bdpt_out[ob +  5] = pos.z;
    bdpt_out[ob +  6] = nrm.x;       bdpt_out[ob +  7] = nrm.y;       bdpt_out[ob +  8] = nrm.z;
    bdpt_out[ob +  9] = dir_in.x;    bdpt_out[ob + 10] = dir_in.y;    bdpt_out[ob + 11] = dir_in.z;
    bdpt_out[ob + 12] = dir_out.x;   bdpt_out[ob + 13] = dir_out.y;   bdpt_out[ob + 14] = dir_out.z;
    bdpt_out[ob + 15] = cos_i;
    bdpt_out[ob + 16] = cos_t;
    bdpt_out[ob + 17] = eta_i;
    bdpt_out[ob + 18] = eta_t;
    bdpt_out[ob + 19] = fresnel_r;
    bdpt_out[ob + 20] = transmittance;
    bdpt_out[ob + 21] = throughput;
    bdpt_out[ob + 22] = opl;
    bdpt_out[ob + 23] = geom_len;
    bdpt_out[ob + 24] = aperture_r;
    bdpt_out[ob + 25] = transverse_r;
    bdpt_out[ob + 26] = dist_past_aperture;
    bdpt_out[ob + 27] = phase_space_j;
}

void absorb_with_optical_event(int hb, uint reason, uint element_index,
                               vec3 pos, vec3 dir_in, float eta_i, float eta_t,
                               float opl, float geom_len, float aperture_r,
                               float transverse_r, float dist_past_aperture)
{
    emit_bdpt_optical(hb, reason, element_index, 0u,
                      pos, vec3(0.0), dir_in, vec3(0.0),
                      0.0, 0.0, eta_i, eta_t,
                      reason == BDPT_OPT_TIR ? 1.0 : 0.0,
                      0.0, 0.0,
                      opl, geom_len, aperture_r, transverse_r,
                      dist_past_aperture, 1.0);
    uint cf = hit_u(hb, 16);
    hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
}

float vec3_comp(vec3 v, int i) {
    if (i == 0) return v.x;
    if (i == 1) return v.y;
    return v.z;
}

void vec3_set_comp(inout vec3 v, int i, float f) {
    if (i == 0) v.x = f;
    else if (i == 1) v.y = f;
    else v.z = f;
}

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

/* ── NEURAL_ASSEMBLY deferred absorber ─────────────────────────────────── */
/*
 * The old MLP teleport is intentionally disabled.  It must not re-enter the
 * camera pipeline until it can emit reversible optical metadata equivalent to
 * the parametric lens path.  Current behavior is fail-closed absorption.
 */
void neural_assembly_teleport(int hb, int pay_off)
{
    /* Deferred: MLP transport is disabled until it can emit reversible optical
     * data equivalent to the parametric lens path.  Fail closed by absorbing
     * the ray instead of performing an opaque teleport. */
    uint cflag_disabled = floatBitsToUint(hit_f(hb, 16));
    hit_wf(hb, 16, uintBitsToFloat(cflag_disabled | 8u));
}

/* ── Parametric lens exact algebraic teleport ───────────────────────────── */
/*
 * Evaluates the exact conic intersection + vector Snell's law for every
 * surface in the assembly, accumulates OPL, and either teleports the ray
 * (sets bit 4) or absorbs it (sets bit 3).  No iterative refinement —
 * each surface is one closed-form quadratic solve.
 *
 * Payload built by CompoundLens.build_gpu_payload(); physical parameters
 * live there exclusively.  The mesh is only consulted by T1 for BVH.
 *
 * Conic intersection quadratic  A·t²+B·t+C = 0  (exact for all conics):
 *   c  = 1/R,  kp = 1+k,  (ox,oy,oz) = origin − (x_v,0,0)
 *   A  = c·(dy²+dz² + kp·dx²)
 *   B  = 2·(c·(oy·dy+oz·dz+kp·ox·dx) − dx)
 *   C  = c·(oy²+oz²+kp·ox²) − 2·ox
 */
void parametric_lens_teleport(int hb, int pay_off)
{
    if (abs(npay[pay_off + 0] - PLENS_MAGIC) > 1.0) return;

    int   n_surf = int(npay[pay_off + PLENS_N_SURF]);
    float hood_r = npay[pay_off + PLENS_HOOD_R];
    float hood_xf= npay[pay_off + PLENS_HOOD_XF];
    float hood_xr= npay[pay_off + PLENS_HOOD_XR];

    if (n_surf < 1 || n_surf > PLENS_MAX_SURF) {
        absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, 0u,
                                  vec3(hit_f(hb, 0), hit_f(hb, 1), hit_f(hb, 2)),
                                  normalize(vec3(hit_f(hb, 6), hit_f(hb, 7), hit_f(hb, 8))),
                                  1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    vec3 ray_pos = vec3(hit_f(hb, 0), hit_f(hb, 1), hit_f(hb, 2));
    vec3 ray_dir = normalize(vec3(hit_f(hb, 6), hit_f(hb, 7), hit_f(hb, 8)));
    bool is_backward = (hit_u(hb, 16) == 1u);

    /* Lens hood: project to hood opening plane and check radius */
    if (!is_backward && hood_r > 0.0 && abs(ray_dir.x) > EPS) {
        float t_hood = (hood_xf - ray_pos.x) / ray_dir.x;
        if (t_hood < 0.0) {   /* hood is in front; t negative = forward projection */
            vec3 p_hood = ray_pos + t_hood * ray_dir;
            if (sqrt(p_hood.y*p_hood.y + p_hood.z*p_hood.z) > hood_r) {
                absorb_with_optical_event(hb, BDPT_OPT_VIGNETTE_CLIP, 0u,
                                          p_hood, ray_dir, 1.0, 1.0,
                                          0.0, abs(t_hood), hood_r,
                                          sqrt(p_hood.y*p_hood.y + p_hood.z*p_hood.z),
                                          sqrt(p_hood.y*p_hood.y + p_hood.z*p_hood.z) - hood_r);
                return;
            }
        }
    }

    float opl = 0.0;

    for (int si = 0; si < n_surf; si++) {
        int s = is_backward ? (n_surf - 1 - si) : si;
        int   sb     = pay_off + PLENS_HEADER + s * PLENS_SURF_STRIDE;
        float x_v    = npay[sb + PLENS_S_XPOS];
        float R      = npay[sb + PLENS_S_RCURV];
        float n_bef  = is_backward ? npay[sb + PLENS_S_NAFT] : npay[sb + PLENS_S_NBEF];
        float n_aft  = is_backward ? npay[sb + PLENS_S_NBEF] : npay[sb + PLENS_S_NAFT];
        float ap_r   = npay[sb + PLENS_S_APR];
        float k      = npay[sb + PLENS_S_CONIK];
        bool  is_stop= (int(npay[sb + PLENS_S_FLAGS]) & 1) != 0;
        float seg_t = 0.0;
        float seg_opl = 0.0;

        /* ── Intersect ray with this surface ─────────────────────────────── */
        /* s=0: entry seek from T1 BVH proxy-mesh hit.  Allow t >= -0.01 so the
         * ray can roll back up to 1 cm to the exact conic entrance vertex.
         * s>0: previous Snell step left ray just in front; require t > 2e-7. */
        {
            float t_min = (si == 0) ? -0.01 : 2e-7;
            float t;
            if (abs(R) < EPS) {
                if (abs(ray_dir.x) < EPS) {
                    absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                              ray_pos, ray_dir, n_bef, n_aft,
                                              0.0, 0.0, ap_r, 0.0, 0.0);
                    return;
                }
                t = (x_v - ray_pos.x) / ray_dir.x;
                if (t < t_min) {
                    absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                              ray_pos, ray_dir, n_bef, n_aft,
                                              0.0, 0.0, ap_r, 0.0, 0.0);
                    return;
                }
            } else {
                float c  = 1.0 / R;
                float kp = 1.0 + k;
                float ox = ray_pos.x - x_v;
                float oy = ray_pos.y;
                float oz = ray_pos.z;
                float dx = ray_dir.x, dy = ray_dir.y, dz = ray_dir.z;

                float A  = c * (dy*dy + dz*dz + kp * dx*dx);
                float B  = 2.0 * (c * (oy*dy + oz*dz + kp * ox*dx) - dx);
                float C  = c * (oy*oy + oz*oz + kp * ox*ox) - 2.0 * ox;

                float t1, t2;
                if (abs(A) < EPS) {
                    if (abs(B) < EPS) {
                        absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                                  ray_pos, ray_dir, n_bef, n_aft,
                                                  0.0, 0.0, ap_r, 0.0, 0.0);
                        return;
                    }
                    t = -C / B;
                    if (t < t_min) {
                        absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                                  ray_pos, ray_dir, n_bef, n_aft,
                                                  0.0, 0.0, ap_r, 0.0, 0.0);
                        return;
                    }
                } else {
                    float disc = B*B - 4.0*A*C;
                    if (disc < 0.0) {
                        absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                                  ray_pos, ray_dir, n_bef, n_aft,
                                                  0.0, 0.0, ap_r, 0.0, 0.0);
                        return;
                    }
                    float sq = sqrt(disc);
                    t1 = (-B - sq) / (2.0 * A);
                    t2 = (-B + sq) / (2.0 * A);
                    if (si == 0) {
                        /* Entry seek: pick root with smallest |t| that is >= t_min. */
                        bool v1 = t1 >= t_min, v2 = t2 >= t_min;
                        if (v1 && v2) {
                            t = (abs(t1) <= abs(t2)) ? t1 : t2;
                        } else if (v1) {
                            t = t1;
                        } else if (v2) {
                            t = t2;
                        } else {
                            absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                                      ray_pos, ray_dir, n_bef, n_aft,
                                                      0.0, 0.0, ap_r, 0.0, 0.0);
                            return;
                        }
                    } else {
                        float x1 = ray_pos.x + t1 * dx;
                        float x2 = ray_pos.x + t2 * dx;
                        if (t1 > 2e-7 && abs(x1 - x_v) <= abs(x2 - x_v)) {
                            t = t1;
                        } else if (t2 > 2e-7) {
                            t = t2;
                        } else {
                            absorb_with_optical_event(hb, BDPT_OPT_ABSORPTION, uint(s),
                                                      ray_pos, ray_dir, n_bef, n_aft,
                                                      0.0, 0.0, ap_r, 0.0, 0.0);
                            return;
                        }
                    }
                }
            }
            seg_t = t;
            seg_opl = n_bef * t;
            opl     += seg_opl;
            ray_pos += t * ray_dir;
        }

        /* ── Aperture / stop check ───────────────────────────────────────── */
        float r_tr = sqrt(ray_pos.y*ray_pos.y + ray_pos.z*ray_pos.z);
        if (ap_r > 0.0 && r_tr > ap_r) {
            absorb_with_optical_event(hb, BDPT_OPT_APERTURE_CLIP, uint(s),
                                      ray_pos, ray_dir, n_bef, n_aft,
                                      seg_opl, abs(seg_t), ap_r, r_tr, r_tr - ap_r);
            return;
        }
        if (is_stop) continue;   /* stop passed — no refraction, advance to next */

        /* ── Exact surface normal (gradient of conic equation) ───────────── */
        vec3 surf_n;
        if (abs(R) < EPS) {
            surf_n = vec3(1.0, 0.0, 0.0);
        } else {
            float c  = 1.0 / R;
            float kp = 1.0 + k;
            float dx = ray_pos.x - x_v;
            /* ∂f/∂x = −1 + (1+k)·c·(x−xv),  ∂f/∂y = c·y,  ∂f/∂z = c·z */
            surf_n = normalize(vec3(-1.0 + kp*c*dx,  c*ray_pos.y,  c*ray_pos.z));
        }
        if (dot(ray_dir, surf_n) > 0.0) surf_n = -surf_n;

        /* ── Exact vector Snell's law ─────────────────────────────────────── */
        float cos_i  = -dot(ray_dir, surf_n);
        float eta    = n_bef / n_aft;
        float sin2_t = eta * eta * max(0.0, 1.0 - cos_i*cos_i);
        if (sin2_t > 1.0) {
            /* TIR */
            emit_bdpt_optical(hb, BDPT_OPT_TIR, uint(s), 0u,
                              ray_pos, surf_n, ray_dir, vec3(0.0),
                              cos_i, 0.0, n_bef, n_aft,
                              1.0, 0.0, 0.0,
                              seg_opl, abs(seg_t), ap_r, r_tr, 0.0, 1.0);
            uint cf = hit_u(hb, 16);
            hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
            return;
        }
        float cos_t = sqrt(1.0 - sin2_t);
        vec3 dir_before = ray_dir;
        ray_dir = normalize(eta * ray_dir + (eta * cos_i - cos_t) * surf_n);
        float Rf = fresnel_R(cos_i, cos_t, n_bef, n_aft);
        float J = (cos_i > EPS && cos_t > EPS) ? (eta * eta) * (cos_t / cos_i) : 0.0;
        emit_bdpt_optical(hb, BDPT_OPT_REFRACTION, uint(s), 0u,
                          ray_pos, surf_n, dir_before, ray_dir,
                          cos_i, cos_t, n_bef, n_aft,
                          Rf, 1.0 - Rf, 1.0 - Rf,
                          seg_opl, abs(seg_t), ap_r, r_tr, 0.0, J);
    }

    /* ── Apply accumulated OPL phase to all spectral bands ──────────────── */
    int nb = min(n_bands, MAX_GPU_BANDS);
    for (int b = 0; b < nb; b++) {
        float freq = freq_hz[b];
        if (freq <= 0.0) continue;
        float phase = TWO_PI * freq * opl / SPEED_LIGHT;
        float ph_c  = cos(phase), ph_s = sin(phase);
        float ar = hit_f(hb, 26 + b);
        float ai = hit_f(hb, 26 + MAX_GPU_BANDS + b);
        hit_wf(hb, 26 + b,                ar*ph_c - ai*ph_s);
        hit_wf(hb, 26 + MAX_GPU_BANDS + b, ar*ph_s + ai*ph_c);
    }

    /* ── Write exit state and set bit 4 (teleport) ───────────────────────── */
    vec3 exit_pos = ray_pos + ray_dir * 2.0e-7;
    hit_wf(hb, 0, exit_pos.x); hit_wf(hb, 1, exit_pos.y); hit_wf(hb, 2, exit_pos.z);
    hit_wf(hb, 3, ray_dir.x);  hit_wf(hb, 4, ray_dir.y);  hit_wf(hb, 5, ray_dir.z);
    hit_wf(hb, 6, ray_dir.x);  hit_wf(hb, 7, ray_dir.y);  hit_wf(hb, 8, ray_dir.z);
    hit_wf(hb, 12, hit_f(hb, 12) + opl);
    uint cflag = floatBitsToUint(hit_f(hb, 16));
    hit_wf(hb, 16, uintBitsToFloat(cflag | 4u));
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

    /* Read incoming_dir */
    vec3 inc_dir = vec3(hit_f(hb, 6), hit_f(hb, 7), hit_f(hb, 8));

    int pb = gid_param * GROUP_PAYLOAD_STRIDE;

    /* ── NEURAL_ASSEMBLY: deferred ───────────────────────────────────────
     * Disabled until it can provide the same reversible optical metadata as
     * the parametric lens path.  Absorb instead of performing an opaque MLP
     * teleport inside the basic camera pipeline. */
    if (kind == PARAM_NEURAL_ASSEMBLY) {
        uint cflag_na = floatBitsToUint(hit_f(hb, 16));
        hit_wf(hb, 16, uintBitsToFloat(cflag_na | 8u));
        return;
    }

    /* ── PARAMETRIC_LENS: exact algebraic conic teleport ─────────────────── */
    if (kind == PARAM_PARAMETRIC_LENS) {
        int pay_off = floatBitsToInt(group_payload[pb]);
        if (pay_off >= 0) {
            parametric_lens_teleport(hb, pay_off);
        }
        {
            uint cflag_pl = floatBitsToUint(hit_f(hb, 16));
            if ((cflag_pl & (4u | 8u)) == 0u)
                hit_wf(hb, 16, uintBitsToFloat(cflag_pl | 8u));
        }
        return;
    }

    /* Load triangle geometry */
    int tb = hit_tri * TRI_FULL_STRIDE;
    vec3 v0 = vec3(trifull[tb],   trifull[tb+1], trifull[tb+2]);
    vec3 e1 = vec3(trifull[tb+3], trifull[tb+4], trifull[tb+5]);
    vec3 e2 = vec3(trifull[tb+6], trifull[tb+7], trifull[tb+8]);
    vec3 n0 = vec3(trifull[tb+9], trifull[tb+10], trifull[tb+11]);

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
