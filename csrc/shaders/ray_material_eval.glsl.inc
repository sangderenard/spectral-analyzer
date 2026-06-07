/* Shared spectral material evaluation for ray_material/T5 shaders.
 *
 * Host shader preamble must provide:
 *   MAT_EVAL_MAX_BANDS
 *   MAT_EVAL_BAND_STRIDE
 *   MAT_EVAL_MAT_BINDING
 * and uniform:
 *   int t5_n_mats
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

layout(std430, binding = MAT_EVAL_MAT_BINDING) readonly buffer MatEvalBandBuf {
    float mat_eval_bands[];
};

uniform int t5_n_mats;

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

float meval_ggx_D(float alpha, float NoH) {
    float a = max(alpha, 1.0e-4f);
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (3.14159265358979323846f * d * d);
}

float meval_ggx_G1(float alpha, float NoV) {
    if (NoV <= 0.0f) return 0.0f;
    float a = max(alpha, 1.0e-4f);
    float a2 = a * a;
    return (2.0f * NoV) / (NoV + sqrt(a2 + (1.0f - a2) * NoV * NoV));
}

float meval_ggx_brdf(vec3 n, vec3 in_dir, vec3 out_dir, float alpha, float refl) {
    vec3 V = normalize(-in_dir);
    vec3 L = normalize(out_dir);
    float NoV = max(0.0f, dot(n, V));
    float NoL = max(0.0f, dot(n, L));
    if (NoV <= 0.0f || NoL <= 0.0f) return 0.0f;
    vec3 H = V + L;
    float h2 = dot(H, H);
    if (h2 <= 1.0e-20f) return 0.0f;
    H *= inversesqrt(h2);
    float NoH = max(0.0f, dot(n, H));
    float VoH = max(0.0f, dot(V, H));
    if (NoH <= 0.0f || VoH <= 1.0e-12f) return 0.0f;

    float D = meval_ggx_D(alpha, NoH);
    float G = meval_ggx_G1(alpha, NoV) * meval_ggx_G1(alpha, NoL);
    return refl * D * G / max(4.0f * NoV * NoL, 1.0e-12f);
}

float meval_endpoint_response(int mat, int band, uint pdf_flags,
                              vec3 n, vec3 in_dir, vec3 out_dir) {
    if (mat < 0) return 1.0f;

    float cos_out = max(0.0f, dot(n, normalize(out_dir)));
    if (cos_out <= 0.0f) return 0.0f;

    if ((pdf_flags & BDPT_PDF_FLAG_EMISSION) != 0u) {
        return 1.0f;
    }

    float refl = meval_reflectance(mat, band);
    if (refl <= 0.0f) return 0.0f;

    if ((pdf_flags & BDPT_PDF_FLAG_DIFFUSE) != 0u) {
        return refl * meval_diffuse_p(mat) / 3.14159265358979323846f;
    }

    if ((pdf_flags & BDPT_PDF_FLAG_GGX) != 0u) {
        float spec_p = max(0.0f, 1.0f - meval_diffuse_p(mat));
        float alpha = meval_ggx_alpha(mat);
        if (spec_p <= 0.0f || alpha <= 1.0e-3f) return 0.0f;
        return spec_p * meval_ggx_brdf(n, in_dir, out_dir, alpha, refl);
    }

    if ((pdf_flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) {
        return 0.0f;
    }

    return refl / 3.14159265358979323846f;
}

#endif
