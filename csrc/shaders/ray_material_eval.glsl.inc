/* ray_material_eval.glsl.inc — canonical shared material evaluation library.
 *
 * Injected as a preamble into T5 connect and any future shader that needs
 * consistent material evaluation.  T3 (ray_material.comp.glsl) implements the
 * same math independently; this file is the authoritative reference so that
 * T5's BRDF evaluation and scatter-PDF evaluation always agree with T3.
 *
 * Host shader must provide (or the loader injects via #define block):
 *   MAT_EVAL_MAX_BANDS   (default 32)
 *   MAT_EVAL_BAND_STRIDE (default 12)
 *   MAT_EVAL_MAT_BINDING (default 8)
 *
 * The host shader must also declare `uniform int t5_n_mats;` when using the
 * material accessor functions.
 *
 * MatBand layout (MAT_BAND_STRIDE = 12 floats, row = mat * MAX_BANDS + band):
 *   [0]  center_hz      [1]  bandwidth_hz    [2]  reflectance_mag
 *   [3]  transmittance  [4]  diffuse_frac    [5]  emission
 *   [6]  reemission     [7]  ior_real        [8]  ior_imag
 *   [9]  reactive_shift [10] ggx_alpha       [11] metallic
 */
#ifndef RAY_MATERIAL_EVAL_GLSL_INC
#define RAY_MATERIAL_EVAL_GLSL_INC

#ifndef MAT_EVAL_MAX_BANDS
#define MAT_EVAL_MAX_BANDS 32
#endif
#ifndef MAT_EVAL_BAND_STRIDE
#define MAT_EVAL_BAND_STRIDE 12
#endif
#ifndef MAT_EVAL_MAT_BINDING
#define MAT_EVAL_MAT_BINDING 8
#endif

#ifndef BDPT_PDF_FLAG_DELTA_SPECULAR
#define BDPT_PDF_FLAG_DELTA_SPECULAR  (1u << 16)
#endif
#ifndef BDPT_PDF_FLAG_DIFFUSE
#define BDPT_PDF_FLAG_DIFFUSE         (1u << 18)
#endif
#ifndef BDPT_PDF_FLAG_GGX
#define BDPT_PDF_FLAG_GGX             (1u << 20)
#endif
#ifndef BDPT_PDF_FLAG_EMISSION
#define BDPT_PDF_FLAG_EMISSION        (1u << 21)
#endif

#define MEVAL_PI 3.14159265358979323846f

layout(std430, binding = MAT_EVAL_MAT_BINDING) readonly buffer MatEvalBandBuf {
    float mat_eval_bands[];
};

uniform int t5_n_mats;

/* ── Material accessors ──────────────────────────────────────────────────── */

float meval_band_field(int mat, int b, int field, float fallback) {
    if (mat < 0 || mat >= t5_n_mats) return fallback;
    if (b < 0 || b >= MAT_EVAL_MAX_BANDS) return fallback;
    return mat_eval_bands[(mat * MAT_EVAL_MAX_BANDS + b) * MAT_EVAL_BAND_STRIDE + field];
}

float meval_reflectance(int mat, int b) {
    return max(0.0f, meval_band_field(mat, b, 2, 0.0f));
}

float meval_transmittance(int mat, int b) {
    return max(0.0f, meval_band_field(mat, b, 3, 0.0f));
}

float meval_diffuse_p(int mat) {
    return clamp(meval_band_field(mat, 0, 4, 0.0f), 0.0f, 1.0f);
}

float meval_emission(int mat, int b) {
    return max(0.0f, meval_band_field(mat, b, 5, 0.0f));
}

float meval_ior_real(int mat, int b) {
    float n = meval_band_field(mat, b, 7, 1.0f);
    return (n > 1.0e-10f) ? n : 1.0f;
}

float meval_ggx_alpha(int mat) {
    return clamp(meval_band_field(mat, 0, 10, 0.0f), 0.0f, 1.0f);
}

/* ── xorshift32 PRNG (matches T3's rand_next) ────────────────────────────── */
float meval_rand_next(inout uint s) {
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    return float(s) * (1.0f / 4294967296.0f);
}

/* ── Pure math: GGX microfacet ───────────────────────────────────────────── *
 * Matches T3's ggx_D / ggx_G1 / ggx_pdf_solid_angle / ggx_sample_reflection */

float meval_ggx_D(float alpha, float NoH) {
    float a  = max(alpha, 1.0e-4f);
    float a2 = a * a;
    float d  = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (MEVAL_PI * d * d);
}

float meval_ggx_G1(float alpha, float NoV) {
    if (NoV <= 0.0f) return 0.0f;
    float a  = max(alpha, 1.0e-4f);
    float a2 = a * a;
    return (2.0f * NoV) / (NoV + sqrt(a2 + (1.0f - a2) * NoV * NoV));
}

/* GGX BRDF value (no cosine factor — caller multiplies by NoL if needed). */
float meval_ggx_brdf(vec3 n, vec3 in_dir, vec3 out_dir, float alpha, float refl) {
    vec3  V   = normalize(-in_dir);
    vec3  L   = normalize(out_dir);
    float NoV = max(0.0f, dot(n, V));
    float NoL = max(0.0f, dot(n, L));
    if (NoV <= 0.0f || NoL <= 0.0f) return 0.0f;
    vec3  H   = V + L;
    float h2  = dot(H, H);
    if (h2 <= 1.0e-20f) return 0.0f;
    H *= inversesqrt(h2);
    float NoH = max(0.0f, dot(n, H));
    float VoH = max(0.0f, dot(V, H));
    if (NoH <= 0.0f || VoH <= 1.0e-12f) return 0.0f;
    float D = meval_ggx_D(alpha, NoH);
    float G = meval_ggx_G1(alpha, NoV) * meval_ggx_G1(alpha, NoL);
    return refl * D * G / max(4.0f * NoV * NoL, 1.0e-12f);
}

/* GGX scatter PDF in solid angle domain — matches T3 ggx_pdf_solid_angle exactly. */
float meval_ggx_pdf_sa(vec3 n, vec3 in_dir, vec3 out_dir, float alpha) {
    vec3  V   = normalize(-in_dir);
    vec3  L   = normalize(out_dir);
    float NoV = max(0.0f, dot(n, V));
    float NoL = max(0.0f, dot(n, L));
    if (NoV <= 0.0f || NoL <= 0.0f) return 0.0f;
    vec3  H   = V + L;
    float h2  = dot(H, H);
    if (h2 <= 1.0e-20f) return 0.0f;
    H *= inversesqrt(h2);
    float NoH = max(0.0f, dot(n, H));
    float VoH = max(0.0f, dot(V, H));
    if (NoH <= 0.0f || VoH <= 1.0e-12f) return 0.0f;
    return meval_ggx_D(alpha, NoH) * meval_ggx_G1(alpha, NoV) / (4.0f * NoV);
}

/* GGX visible-normal importance-sampling — matches T3 ggx_sample_reflection. */
vec3 meval_ggx_sample_reflection(vec3 n, vec3 in_dir, float alpha, inout uint rng_state) {
    float u1 = meval_rand_next(rng_state);
    float u2 = meval_rand_next(rng_state);
    float a  = max(alpha, 1.0e-4f);
    vec3  up = abs(n.z) < 0.999f ? vec3(0,0,1) : vec3(1,0,0);
    vec3  t  = normalize(cross(up, n));
    vec3  b  = cross(n, t);
    vec3  V  = normalize(-in_dir);
    vec3  Vl = vec3(dot(V,t), dot(V,b), dot(V,n));
    if (Vl.z <= 1.0e-8f) return normalize(in_dir - 2.0f * dot(in_dir, n) * n);
    vec3  Vh = normalize(vec3(a * Vl.x, a * Vl.y, Vl.z));
    float lsq = Vh.x*Vh.x + Vh.y*Vh.y;
    vec3  T1  = (lsq > 1.0e-20f)
        ? vec3(-Vh.y, Vh.x, 0.0f) * inversesqrt(lsq)
        : vec3(1.0f, 0.0f, 0.0f);
    vec3  T2  = cross(Vh, T1);
    float r   = sqrt(max(0.0f, u1));
    float phi = 6.28318530718f * u2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - t1*t1)) + s * t2;
    vec3  Nh  = t1*T1 + t2*T2 + sqrt(max(0.0f, 1.0f - t1*t1 - t2*t2)) * Vh;
    vec3  Hl  = normalize(vec3(a * Nh.x, a * Nh.y, max(0.0f, Nh.z)));
    vec3  H   = normalize(t * Hl.x + b * Hl.y + n * Hl.z);
    if (dot(H, V) < 0.0f) H = -H;
    return normalize(in_dir - 2.0f * dot(in_dir, H) * H);
}

/* ── Pure math: Fresnel / Snell ─────────────────────────────────────────── *
 * Matches T3's fresnel_R / snell_refract exactly.                            */

float meval_fresnel_R(float cos_i, float cos_t, float n1, float n2) {
    float rs_num = n1*cos_i - n2*cos_t;
    float rs_den = n1*cos_i + n2*cos_t;
    float rp_num = n1*cos_t - n2*cos_i;
    float rp_den = n1*cos_t + n2*cos_i;
    float Rs = (abs(rs_den) < 1.0e-9f) ? 1.0f : (rs_num/rs_den)*(rs_num/rs_den);
    float Rp = (abs(rp_den) < 1.0e-9f) ? 1.0f : (rp_num/rp_den)*(rp_num/rp_den);
    return clamp(0.5f*(Rs + Rp), 0.0f, 1.0f);
}

bool meval_snell_refract(vec3 dir, vec3 n, float n1, float n2, out vec3 refracted) {
    float cos_i  = max(0.0f, -dot(dir, n));
    float sin2_t = (n1/n2)*(n1/n2)*(1.0f - cos_i*cos_i);
    if (sin2_t > 1.0f) { refracted = vec3(0.0f); return false; }
    float cos_t  = sqrt(max(0.0f, 1.0f - sin2_t));
    refracted    = normalize((n1/n2)*dir + ((n1/n2)*cos_i - cos_t)*n);
    return true;
}

/* ── Pure math: cosine-weighted hemisphere sampling ─────────────────────── *
 * Matches T3's cosine_hemisphere.                                            */

vec3 meval_cosine_hemisphere(vec3 n, inout uint rng_state) {
    float u1  = meval_rand_next(rng_state);
    float u2  = meval_rand_next(rng_state);
    float r   = sqrt(u1);
    float phi = 6.28318530718f * u2;
    float x   = r * cos(phi);
    float y   = r * sin(phi);
    float z   = sqrt(max(0.0f, 1.0f - u1));
    vec3  up  = abs(n.z) < 0.9999f ? vec3(0,0,1) : vec3(1,0,0);
    vec3  t   = normalize(cross(up, n));
    vec3  bv  = cross(n, t);
    return normalize(t*x + bv*y + n*z);
}

/* ── Scatter PDF (solid-angle domain) ───────────────────────────────────── *
 * Mirrors T3's per-lobe PDF computation exactly.  Use this anywhere T5 needs *
 * the PDF at which T3 would have sampled a given outgoing direction.          *
 * Returns 0 for delta-specular vertices (arbitrary connection rejected).     */

float meval_scatter_pdf_sa(vec3 n, vec3 in_dir, vec3 out_dir,
                           float diffuse_p, float ggx_alpha, uint pdf_flags) {
    if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return 0.0f;

    if ((pdf_flags & BDPT_PDF_FLAG_EMISSION) != 0u) {
        float cos_out = max(0.0f, dot(n, normalize(out_dir)));
        return cos_out / MEVAL_PI;
    }

    float spec_p = max(0.0f, 1.0f - diffuse_p);

    if ((pdf_flags & BDPT_PDF_FLAG_DIFFUSE) != 0u) {
        if (diffuse_p <= 0.0f) return 0.0f;
        float cos_out = max(0.0f, dot(n, normalize(out_dir)));
        return diffuse_p * cos_out / MEVAL_PI;
    }

    if ((pdf_flags & BDPT_PDF_FLAG_GGX) != 0u) {
        if (spec_p <= 0.0f || ggx_alpha <= 1.0e-3f) return 0.0f;
        return spec_p * meval_ggx_pdf_sa(n, in_dir, out_dir, ggx_alpha);
    }

    return 0.0f;
}

/* ── Endpoint BRDF evaluation ───────────────────────────────────────────── *
 * Returns the BRDF value (no cosine factor) at a path endpoint for the       *
 * given connection direction out_dir, modulated by the authored reflectance  *
 * at this band.  Delta-specular vertices always return 0 — arbitrary pair    *
 * connections through specular lobes are not valid BDPT strategies.          *
 *                                                                             *
 * For emission vertices the radiance is already carried in the spectral beta;*
 * return 1.0 so the caller's multiplication leaves the beta unchanged.       */

float meval_endpoint_response(int mat, int band, uint pdf_flags,
                              vec3 n, vec3 in_dir, vec3 out_dir) {
    if (mat < 0) return 1.0f;

    float cos_out = max(0.0f, dot(n, normalize(out_dir)));
    if (cos_out <= 0.0f) return 0.0f;

    if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return 0.0f;

    if ((pdf_flags & BDPT_PDF_FLAG_EMISSION) != 0u) return 1.0f;

    float refl = meval_reflectance(mat, band);
    if (refl <= 0.0f) return 0.0f;

    if ((pdf_flags & BDPT_PDF_FLAG_DIFFUSE) != 0u) {
        float dp = meval_diffuse_p(mat);
        if (dp <= 0.0f) return 0.0f;
        return refl * dp / MEVAL_PI;
    }

    if ((pdf_flags & BDPT_PDF_FLAG_GGX) != 0u) {
        float spec_p = max(0.0f, 1.0f - meval_diffuse_p(mat));
        float alpha  = meval_ggx_alpha(mat);
        if (spec_p <= 0.0f || alpha <= 1.0e-3f) return 0.0f;
        return spec_p * meval_ggx_brdf(n, in_dir, out_dir, alpha, refl);
    }

    /* No recognised lobe flag — conservatively zero rather than guessing. */
    return 0.0f;
}

#endif /* RAY_MATERIAL_EVAL_GLSL_INC */
