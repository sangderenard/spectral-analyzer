#version 430 core
/**
 * base_material.frag.glsl
 *
 * Fragment shader for the base material Phong renderer.
 *
 * Material data is read from three SSBOs (std430) that mirror the ctypes
 * structures in material_db.py exactly:
 *
 *   PBRBaseRecord   binding=10  16 floats/material
 *   PhongRecord     binding=11   8 floats/material
 *   EnamelRecord    binding=14   8 floats/material
 *
 * Per-material YAML properties consumed here
 * ------------------------------------------
 *   albedo_rgb       → pbr.albedo        (vec3, linear sRGB)
 *   smoothness       → 1 - pbr.roughness (controls specular breadth & strength)
 *   reflectivity     → derived into phong.spec_strength via _pbr_to_phong_record
 *   diffusion        → pbr.roughness     (diffuse / specular weight split)
 *   absorption       → baked into phong.spec_strength (less is more absorbed)
 *   ior              → pbr.ior           (Schlick Fresnel F0)
 *   opacity          → pbr.opacity       (fragment alpha; drives GL blending)
 *   emission_rgb     → pbr.emission      (additive self-emission)
 *   enamel.*         → EnamelRecord      (thin-film coat; thickness_nm=0 → skip)
 *     enamel.thickness_nm → iridescence visibility
 *     enamel.ior_real     → Fresnel gloss
 *     enamel.color_rgb    → tint
 *
 * Scene uniforms (uploaded once per frame by base_gl_renderer.py):
 *   uLightV           — light direction in VIEW space (unit vec3)
 *   uSceneRgb         — environment spectral tint (vec3)
 *   uSceneIndirectRatio — indirect fill fraction [0,1]
 */

/* ── Material SSBOs ───────────────────────────────────────────────────────── */

// PBRBaseRecord  (16 floats)
// [0..2]  albedo.rgb
// [3]     roughness
// [4]     metallic
// [5]     transmission
// [6]     ior
// [7]     opacity
// [8..10] emission.rgb
// [11..15] reserved
layout(std430, binding = 10) readonly buffer PBRChunk   { float pbr[];   };

// PhongRecord  (8 floats)
// [0] ambient  [1] spec_strength  [2] shininess  [3] _reserved
// [4..6] inner_color.rgb  [7] _pad
layout(std430, binding = 11) readonly buffer PhongChunk { float phong[]; };

// EnamelRecord  (8 floats)
// [0] thickness_nm  [1] ior_real  [2] ior_imag  [3] roughness
// [4..6] color.rgb  [7] _pad
layout(std430, binding = 14) readonly buffer EnamelChunk { float enamel[]; };

/* ── Accessors ────────────────────────────────────────────────────────────── */

#define PBR_STRIDE    16
#define PHONG_STRIDE   8
#define ENAM_STRIDE    8

vec3  mat_albedo   (int id) { int b=id*PBR_STRIDE;   return vec3(pbr[b],   pbr[b+1], pbr[b+2]); }
float mat_roughness(int id) { return pbr[id*PBR_STRIDE+3]; }
float mat_metallic (int id) { return pbr[id*PBR_STRIDE+4]; }
float mat_trans    (int id) { return pbr[id*PBR_STRIDE+5]; }
float mat_ior      (int id) { return pbr[id*PBR_STRIDE+6]; }
float mat_opacity  (int id) { return pbr[id*PBR_STRIDE+7]; }
vec3  mat_emission (int id) { int b=id*PBR_STRIDE+8; return vec3(pbr[b],   pbr[b+1], pbr[b+2]); }

float ph_ambient  (int id) { return phong[id*PHONG_STRIDE+0]; }
float ph_specstr  (int id) { return phong[id*PHONG_STRIDE+1]; }
float ph_shininess(int id) { return phong[id*PHONG_STRIDE+2]; }
vec3  ph_inner    (int id) { int b=id*PHONG_STRIDE+4; return vec3(phong[b],phong[b+1],phong[b+2]); }

float en_thick (int id) { return enamel[id*ENAM_STRIDE+0]; }
float en_ior   (int id) { return enamel[id*ENAM_STRIDE+1]; }
vec3  en_color (int id) { int b=id*ENAM_STRIDE+4; return vec3(enamel[b],enamel[b+1],enamel[b+2]); }

/* ── Inputs / uniforms ────────────────────────────────────────────────────── */

in  vec3     vNormV;
in  vec3     vPosV;
flat in int  vMatId;

out vec4 FragColor;

uniform vec3  uLightV;
uniform vec3  uSceneRgb;
uniform float uSceneIndirectRatio;

/* ── Helpers ──────────────────────────────────────────────────────────────── */

// Schlick Fresnel approximation (scalar F0, scalar cosTheta)
float schlick(float f0, float cosT) {
    float t = 1.0 - max(0.0, cosT);
    float t2 = t * t;
    return f0 + (1.0 - f0) * (t2 * t2 * t);
}

// Three-cosine thin-film iridescence (OPD in nm, reference period 550 nm)
// Maps optical path difference to a visible-spectrum RGB fringe colour.
vec3 thin_film_fringe(float opd_nm) {
    float f = opd_nm * (2.0 * 3.14159265 / 550.0);
    return vec3(
        0.5 + 0.5 * cos(f),
        0.5 + 0.5 * cos(f - 2.09439510),
        0.5 + 0.5 * cos(f - 4.18879020)
    );
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

void main() {
    int id = vMatId;

    // Geometry
    vec3 N = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3 L = normalize(uLightV);
    vec3 V = normalize(-vPosV);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    // Material parameters
    vec3  albedo   = mat_albedo   (id);
    float rough    = mat_roughness(id);
    float metallic = mat_metallic (id);
    float ior      = mat_ior      (id);
    float opacity  = mat_opacity  (id);
    vec3  emission = mat_emission (id);

    float ambient  = ph_ambient  (id);
    float specstr  = ph_specstr  (id);
    float shini    = ph_shininess(id);
    vec3  inner    = ph_inner    (id);

    // Front / back face base colour
    vec3 base = gl_FrontFacing ? albedo : inner;

    // Schlick Fresnel — F0 derived from IOR (dielectric) blended with albedo (conductor)
    float ior_f0 = (ior - 1.0) / (ior + 1.0);
    ior_f0 *= ior_f0;
    vec3 F0 = mix(vec3(ior_f0), albedo, metallic);
    // Fresnel at view angle
    float F_scal = schlick(ior_f0, NdotV);
    vec3  F_vec  = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // Diffuse + specular
    float diff = NdotL;
    float spec = pow(NdotH, max(shini, 1.0));

    // Scene-tinted ambient (indirect fill in shadowed regions)
    vec3  amb_light  = ambient * mix(vec3(1.0), uSceneRgb, 0.55);
    float shadowFill = uSceneIndirectRatio * 0.28 * (1.0 - diff);

    // Specular colour: neutral warm tint blended with scene
    vec3 spec_col = mix(vec3(1.0, 0.93, 0.70), uSceneRgb, 0.35);

    vec3 col = base * (amb_light + (0.78 + shadowFill) * diff)
             + spec_col * F_vec * (specstr * spec);

    // Rim lighting driven by indirect fill ratio
    float rim = pow(1.0 - NdotV, 3.0);
    col += base * rim * mix(0.14, 0.22, uSceneIndirectRatio);

    // Self-emission
    col += emission;

    // ── Enamel thin-film coating ──────────────────────────────────────────
    // Skipped entirely if thickness_nm == 0 (most materials).
    float enam_thick = en_thick(id);
    if (enam_thick > 0.0) {
        float enam_ior_v  = en_ior  (id);
        vec3  enam_c      = en_color(id);

        // Optical path difference (nm): 2 * n_film * d * cos(theta_t)
        // cos(theta_t) ≈ NdotV for near-normal incidence (Snell simplified)
        float opd_nm  = 2.0 * enam_ior_v * enam_thick * NdotV;

        // Iridescence fringe — most visible for thickness 100–800 nm
        vec3  fringe   = thin_film_fringe(opd_nm);
        float fringe_w = clamp(enam_thick / 800.0, 0.0, 0.45);

        vec3 tinted = col * enam_c * (vec3(1.0) + fringe * fringe_w);

        // Extra gloss from enamel layer's own Fresnel peak
        float eF0 = (enam_ior_v - 1.0) / (enam_ior_v + 1.0);
        eF0 *= eF0;
        float eF  = schlick(eF0, NdotV);
        tinted   += spec_col * (eF * 0.28 * spec);

        col = mix(col, tinted, 0.35);
    }

    // ── Opacity ────────────────────────────────────────────────────────────
    // For transmission > 0 the material is dielectric; blend alpha down.
    // The GL blend state (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) handles compositing.
    float alpha = opacity * (1.0 - mat_trans(id) * 0.8);

    FragColor = vec4(col, alpha);
}
