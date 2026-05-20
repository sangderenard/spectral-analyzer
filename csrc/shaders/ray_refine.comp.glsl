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
 *   4 = NEURAL_ASSEMBLY — MLP teleport: entrance→exit surface in one shot
 *
 * Flat buffer layouts (same bindings as T1 but shared HitBuf is now input):
 *
 *  HitBuf  (REFINED_HIT_STRIDE = 58 floats): in-place update of [0..8], [12], [16], [26..57]
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
 *  NeuralPayBuf  (binding 6): concatenated float32 neural payloads.
 *    Magic header at [off+0]=14948.0, then layout per neural_assembly.py:
 *      [0]     MAGIC_NEURAL (14948.0)
 *      [1..4]  n_layers, input_dim(5), hidden_dim, output_dim(6)
 *      [5]     z_ent   (scene axial coord of entrance)
 *      [6]     z_exit  (scene axial coord of exit)
 *      [7..12] reserved
 *      [13]    axis_idx (0=X, 2=Z)
 *      [14..15] reserved
 *      [16..20] in_mean[5]
 *      [21..25] in_scale[5]
 *      [26..31] out_mean[6]
 *      [32..37] out_scl[6]
 *      [38..]   layer weights (W row-major then bias for each layer)
 *
 *  CounterBuf:
 *    [0] = hit count  (from T1, used as n_hits here)
 *    [3] = cpu_refine count  (incremented for complex-kind fallbacks)
 *
 * NEURAL_ASSEMBLY passthrough protocol:
 *   T2 sets bit 4 of color_flag (HitBuf[16]) to signal T3 to re-emit the
 *   ray without Snell/Fresnel physics.  T3 checks this flag and forwards
 *   the ray with the updated pos/dir and original medium.
 */
#version 430 core

layout(local_size_x = 64) in;

/* ── Layout constants ───────────────────────────────────────────────────── */
#define MAX_GPU_BANDS         16
#define HIT_STRIDE            58
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

/* ── Uniforms ───────────────────────────────────────────────────────────── */
uniform int   n_tris;
uniform int   n_groups;
uniform int   n_bands;
uniform float freq_hz[MAX_GPU_BANDS];

/* ── Helpers ────────────────────────────────────────────────────────────── */

float hit_f(int base, int i) { return hits[base + i]; }
void  hit_wf(int base, int i, float v) { hits[base + i] = v; }

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

/* ── NEURAL_ASSEMBLY MLP teleport ──────────────────────────────────────── */
/*
 * Runs the trained MLP forward pass to teleport the ray from the entrance
 * surface (hit point) to the exit surface, accumulating phase on each band.
 * On success writes back updated pos, incoming_dir, hit_n, path_len, amp
 * and sets bit 4 of color_flag to signal T3 to skip Snell/Fresnel physics.
 */
void neural_assembly_teleport(int hb, int pay_off)
{
    /* Validate magic */
    if (abs(npay[pay_off] - NEURAL_MAGIC) > 1.0) return;

    int n_layers   = int(npay[pay_off + NPAY_N_LAYERS]);
    int input_dim  = int(npay[pay_off + NPAY_IN_DIM]);
    int hidden_dim = int(npay[pay_off + NPAY_HIDDEN_DIM]);
    int output_dim = int(npay[pay_off + NPAY_OUT_DIM]);
    float z_exit   = npay[pay_off + NPAY_Z_EXIT];
    int axis_idx   = int(npay[pay_off + NPAY_AXIS_IDX]);

    if (input_dim != 5 || output_dim != 6) return;
    if (n_layers < 2 || hidden_dim < 1 || hidden_dim > MAX_NEURAL_DIM) return;

    int nb = min(n_bands, MAX_GPU_BANDS);
    if (nb <= 0) return;

    /* Coordinate axis setup: ax=optical axis, t0/t1=transverse */
    int ax = (axis_idx >= 0 && axis_idx <= 2) ? axis_idx : 2;
    int t0 = (ax == 0) ? 1 : 0;
    int t1 = (ax == 2) ? 1 : 2;

    vec3 pos     = vec3(hit_f(hb, 0), hit_f(hb, 1), hit_f(hb, 2));
    vec3 inc_dir = vec3(hit_f(hb, 6), hit_f(hb, 7), hit_f(hb, 8));

    float pos_t0 = vec3_comp(pos, t0);
    float pos_t1 = vec3_comp(pos, t1);
    float inc_t0 = vec3_comp(inc_dir, t0);
    float inc_t1 = vec3_comp(inc_dir, t1);
    float dir_z_in = abs(vec3_comp(inc_dir, ax));

    if (dir_z_in < 1e-9) return;

    /* Parametric acceptance boundary: c0 + c1*(r/r_lens)^2
     * Shared gate for both LUT and MLP paths — reject outside the acceptance
     * cone and mark for silent absorption by T3 (bit 3 = 8u of color_flag). */
    {
        float bnd_c0 = npay[pay_off + NPAY_BND_C0];
        float bnd_c1 = npay[pay_off + NPAY_BND_C1];
        float r_lens = npay[pay_off + NPAY_R_LENS];
        if ((bnd_c0 != 0.0 || bnd_c1 != 0.0) && r_lens > 0.0) {
            float r_in_bnd = sqrt(pos_t0 * pos_t0 + pos_t1 * pos_t1);
            float r_norm   = r_in_bnd / r_lens;
            if (dir_z_in < bnd_c0 + bnd_c1 * r_norm * r_norm) {
                uint cflag_rej = floatBitsToUint(hit_f(hb, 16));
                hit_wf(hb, 16, uintBitsToFloat(cflag_rej | 8u));
                return;
            }
        }
    }

    /* Canonical cylindrical frame */
    float theta_hit  = atan(pos_t1, pos_t0);
    float r_in       = sqrt(pos_t0 * pos_t0 + pos_t1 * pos_t1);
    float cos_t      = cos(theta_hit);
    float sin_t      = sin(theta_hit);
    float dir_r_in   =  inc_t0 * cos_t + inc_t1 * sin_t;
    float dir_phi_in = -inc_t0 * sin_t + inc_t1 * cos_t;

    /* Normalization block offsets in npay */
    int in_mean_o  = pay_off + NPAY_IN_MEAN;
    int in_scl_o   = pay_off + NPAY_IN_SCALE;
    int out_mean_o = pay_off + NPAY_OUT_MEAN;
    int out_scl_o  = pay_off + NPAY_OUT_SCL;

    float buf0[MAX_NEURAL_DIM];
    float buf1[MAX_NEURAL_DIM];

    float ref_opl = 0.0;
    float ref_ex  = 0.0, ref_ey  = 0.0;
    float ref_dx  = 0.0, ref_dy  = 0.0, ref_dz = 0.0;
    bool  ref_valid = false;

    for (int b = 0; b < nb; b++) {
        float freq = freq_hz[b];
        if (freq <= 0.0) continue;
        float wavelength_um = SPEED_LIGHT / freq * 1.0e6;

        /* Normalize input */
        float raw_in[5];
        raw_in[0] = r_in;
        raw_in[1] = dir_r_in;
        raw_in[2] = dir_phi_in;
        raw_in[3] = dir_z_in;
        raw_in[4] = wavelength_um;
        for (int i = 0; i < 5; i++) {
            float scl = max(abs(npay[in_scl_o + i]), 1e-12);
            buf0[i] = (raw_in[i] - npay[in_mean_o + i]) / scl;
        }

        /* MLP forward pass */
        int w_off = pay_off + NPAY_LAYER_OFF;
        for (int l = 0; l < n_layers; l++) {
            int in_d  = (l == 0) ? 5 : hidden_dim;
            int out_d = (l == n_layers - 1) ? 6 : hidden_dim;
            bool relu = (l < n_layers - 1);
            int bv_off = w_off + out_d * in_d;
            for (int j = 0; j < out_d; j++) {
                float acc = npay[bv_off + j];
                for (int i = 0; i < in_d; i++)
                    acc += npay[w_off + j * in_d + i] * buf0[i];
                buf1[j] = (relu && acc < 0.0) ? 0.0 : acc;
            }
            for (int i = 0; i < out_d; i++) buf0[i] = buf1[i];
            w_off = bv_off + out_d;
        }

        /* Denormalize output */
        float r_out       = buf0[0] * npay[out_scl_o + 0] + npay[out_mean_o + 0];
        float delta_phi   = buf0[1] * npay[out_scl_o + 1] + npay[out_mean_o + 1];
        float dir_r_out   = buf0[2] * npay[out_scl_o + 2] + npay[out_mean_o + 2];
        float dir_phi_out = buf0[3] * npay[out_scl_o + 3] + npay[out_mean_o + 3];
        float dir_z_out   = buf0[4] * npay[out_scl_o + 4] + npay[out_mean_o + 4];
        float opl         = buf0[5] * npay[out_scl_o + 5] + npay[out_mean_o + 5];

        /* Reconstruct exit position and direction */
        float theta_out = theta_hit + delta_phi;
        float ex = r_out * cos(theta_out);
        float ey = r_out * sin(theta_out);

        float out_dx = dir_r_out * cos_t - dir_phi_out * sin_t;
        float out_dy = dir_r_out * sin_t + dir_phi_out * cos_t;
        float out_dz = dir_z_out;
        float dnorm  = sqrt(out_dx*out_dx + out_dy*out_dy + out_dz*out_dz);
        if (dnorm < 1e-12) continue;
        out_dx /= dnorm; out_dy /= dnorm; out_dz /= dnorm;

        /* Apply OPL phase rotation to this band's amplitude */
        float phase  = TWO_PI * freq * opl / SPEED_LIGHT;
        float ph_cos = cos(phase);
        float ph_sin = sin(phase);
        float ar = hit_f(hb, 26 + b);
        float ai = hit_f(hb, 26 + MAX_GPU_BANDS + b);
        hit_wf(hb, 26 + b,                ar * ph_cos - ai * ph_sin);
        hit_wf(hb, 26 + MAX_GPU_BANDS + b, ar * ph_sin + ai * ph_cos);

        /* First valid band defines pos/dir reference */
        if (!ref_valid) {
            ref_opl = opl;
            ref_ex  = ex;  ref_ey  = ey;
            ref_dx  = out_dx; ref_dy = out_dy; ref_dz = out_dz;
            ref_valid = true;
        }
    }

    if (!ref_valid) return;

    /* Reconstruct exit position and direction in scene space */
    vec3 new_pos;
    vec3_set_comp(new_pos, ax, z_exit);
    vec3_set_comp(new_pos, t0, ref_ex);
    vec3_set_comp(new_pos, t1, ref_ey);

    vec3 new_dir;
    vec3_set_comp(new_dir, ax, ref_dz);
    vec3_set_comp(new_dir, t0, ref_dx);
    vec3_set_comp(new_dir, t1, ref_dy);
    new_dir = normalize(new_dir);

    /* Nudge past exit surface to avoid self-intersection */
    new_pos = new_pos + new_dir * 2.0e-7;

    /* Update path_len */
    float new_path_len = hit_f(hb, 12) + ref_opl;

    /* Write back: pos, hit_n (= new_dir, any value — T3 skips physics),
     * incoming_dir (= new_dir so T3 uses correct direction in re-emit). */
    hit_wf(hb, 0, new_pos.x); hit_wf(hb, 1, new_pos.y); hit_wf(hb, 2, new_pos.z);
    hit_wf(hb, 3, new_dir.x); hit_wf(hb, 4, new_dir.y); hit_wf(hb, 5, new_dir.z);
    hit_wf(hb, 6, new_dir.x); hit_wf(hb, 7, new_dir.y); hit_wf(hb, 8, new_dir.z);
    hit_wf(hb, 12, new_path_len);

    /* Set bit 4 of color_flag: neural passthrough — T3 re-emits without Snell/Fresnel */
    uint cflag = floatBitsToUint(hit_f(hb, 16));
    hit_wf(hb, 16, uintBitsToFloat(cflag | 4u));
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
        uint cf = floatBitsToUint(hit_f(hb, 16));
        hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
        return;
    }

    vec3 ray_pos = vec3(hit_f(hb, 0), hit_f(hb, 1), hit_f(hb, 2));
    vec3 ray_dir = normalize(vec3(hit_f(hb, 6), hit_f(hb, 7), hit_f(hb, 8)));

    /* Lens hood: project to hood opening plane and check radius */
    if (hood_r > 0.0 && abs(ray_dir.x) > EPS) {
        float t_hood = (hood_xf - ray_pos.x) / ray_dir.x;
        if (t_hood < 0.0) {   /* hood is in front; t negative = forward projection */
            vec3 p_hood = ray_pos + t_hood * ray_dir;
            if (sqrt(p_hood.y*p_hood.y + p_hood.z*p_hood.z) > hood_r) {
                uint cf = floatBitsToUint(hit_f(hb, 16));
                hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
                return;
            }
        }
    }

    float opl = 0.0;

    for (int s = 0; s < n_surf; s++) {
        int   sb     = pay_off + PLENS_HEADER + s * PLENS_SURF_STRIDE;
        float x_v    = npay[sb + PLENS_S_XPOS];
        float R      = npay[sb + PLENS_S_RCURV];
        float n_bef  = npay[sb + PLENS_S_NBEF];
        float n_aft  = npay[sb + PLENS_S_NAFT];
        float ap_r   = npay[sb + PLENS_S_APR];
        float k      = npay[sb + PLENS_S_CONIK];
        bool  is_stop= (int(npay[sb + PLENS_S_FLAGS]) & 1) != 0;

        /* ── Intersect ray with this surface ─────────────────────────────── */
        if (s > 0) {
            /* s == 0: ray is already on the entrance surface (T1 hit point) */
            float t;
            if (abs(R) < EPS) {
                /* Flat surface: t = (x_v − ray_pos.x) / ray_dir.x */
                if (abs(ray_dir.x) < EPS) {
                    uint cf = floatBitsToUint(hit_f(hb, 16));
                    hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
                    return;
                }
                t = (x_v - ray_pos.x) / ray_dir.x;
            } else {
                /* Exact conic quadratic */
                float c  = 1.0 / R;
                float kp = 1.0 + k;
                float ox = ray_pos.x - x_v;
                float oy = ray_pos.y;
                float oz = ray_pos.z;
                float dx = ray_dir.x, dy = ray_dir.y, dz = ray_dir.z;

                float A  = c * (dy*dy + dz*dz + kp * dx*dx);
                float B  = 2.0 * (c * (oy*dy + oz*dz + kp * ox*dx) - dx);
                float C  = c * (oy*oy + oz*oz + kp * ox*ox) - 2.0 * ox;

                float disc, t1, t2;
                if (abs(A) < EPS) {
                    if (abs(B) < EPS) {
                        uint cf = floatBitsToUint(hit_f(hb, 16));
                        hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
                        return;
                    }
                    t = -C / B;
                } else {
                    disc = B*B - 4.0*A*C;
                    if (disc < 0.0) {
                        uint cf = floatBitsToUint(hit_f(hb, 16));
                        hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
                        return;
                    }
                    float sq = sqrt(disc);
                    t1 = (-B - sq) / (2.0 * A);
                    t2 = (-B + sq) / (2.0 * A);
                    float x1 = ray_pos.x + t1 * dx;
                    float x2 = ray_pos.x + t2 * dx;
                    if (t1 > 2e-7 && abs(x1 - x_v) <= abs(x2 - x_v)) {
                        t = t1;
                    } else if (t2 > 2e-7) {
                        t = t2;
                    } else {
                        uint cf = floatBitsToUint(hit_f(hb, 16));
                        hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
                        return;
                    }
                }
                if (t <= 2e-7) {
                    uint cf = floatBitsToUint(hit_f(hb, 16));
                    hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
                    return;
                }
            }
            opl     += n_bef * t;
            ray_pos += t * ray_dir;
        }

        /* ── Aperture / stop check ───────────────────────────────────────── */
        float r_tr = sqrt(ray_pos.y*ray_pos.y + ray_pos.z*ray_pos.z);
        if (ap_r > 0.0 && r_tr > ap_r) {
            uint cf = floatBitsToUint(hit_f(hb, 16));
            hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
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
            uint cf = floatBitsToUint(hit_f(hb, 16));
            hit_wf(hb, 16, uintBitsToFloat(cf | 8u));
            return;
        }
        float cos_t = sqrt(1.0 - sin2_t);
        ray_dir = normalize(eta * ray_dir + (eta * cos_i - cos_t) * surf_n);
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

    /* ── NEURAL_ASSEMBLY: full MLP teleport (writes back and returns) ─── */
    if (kind == PARAM_NEURAL_ASSEMBLY) {
        int pay_off = floatBitsToInt(group_payload[pb]);
        if (pay_off >= 0) {
            neural_assembly_teleport(hb, pay_off);
        }
        /* Force absorb if teleport didn't fire (bit 4) and boundary didn't already
         * mark absorb (bit 3).  Covers: no-payload interior absorbers (pay_off<0),
         * magic mismatch, degenerate geometry, and any other early-return path.
         * T3 silently drops rays with bit 3 set — they contribute zero energy. */
        {
            uint cflag_na = floatBitsToUint(hit_f(hb, 16));
            if ((cflag_na & (4u | 8u)) == 0u)
                hit_wf(hb, 16, uintBitsToFloat(cflag_na | 8u));
        }
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
