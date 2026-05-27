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
 *   uNumLights        — active light count (0..MAX_LIGHTS)
 *   uLightPos [N]     — view-space position per emitting surface group
 *   uLightColor[N]    — linear sRGB colour per light (no clamp)
 *   uLightIntensity[N]— scalar gain per light
 *   uLightGroupId[N]  — emitting group id; matching fragments skip that light
 *
 * There is NO ambient, NO scene tint, NO proxy bounce, NO hardcoded sun.
 * Every photon comes from a real emitter pushed in by the host.  A non-
 * emissive surface in a scene with zero lights renders pure black — that
 * is physically correct for this engine.
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

// TextureStackRecord (16 floats)
// [0] emit_uv_layer  [1] color_uv_layer  [2] depth_uv_layer  [3] remit_uv_layer
// [4] depth_scale_mm [5] thickness_scale_mm [6] depth_bias_mm [7] thickness_bias_mm
// [8] emit_gain [9] color_blend [10] direct_lobe_power [11] model_flags_or_indices
// [12] remit_gain [13] remit_attack [14] remit_decay [15] translucence_gain
layout(std430, binding = 15) readonly buffer TextureStackChunk { float texstack[]; };

/* ── Accessors ────────────────────────────────────────────────────────────── */

#define PBR_STRIDE    16
#define PHONG_STRIDE   8
#define ENAM_STRIDE    8
#define TEXSTACK_STRIDE 16

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
float tx_emit_layer(int id) { return texstack[id*TEXSTACK_STRIDE+0]; }
float tx_color_layer(int id) { return texstack[id*TEXSTACK_STRIDE+1]; }
float tx_depth_layer(int id) { return texstack[id*TEXSTACK_STRIDE+2]; }
float tx_remit_layer(int id) { return texstack[id*TEXSTACK_STRIDE+3]; }
float tx_depth_scale_mm(int id) { return texstack[id*TEXSTACK_STRIDE+4]; }
float tx_thickness_scale_mm(int id) { return texstack[id*TEXSTACK_STRIDE+5]; }
float tx_depth_bias_mm(int id) { return texstack[id*TEXSTACK_STRIDE+6]; }
float tx_thickness_bias_mm(int id) { return texstack[id*TEXSTACK_STRIDE+7]; }
float tx_emit_gain(int id) { return texstack[id*TEXSTACK_STRIDE+8]; }
float tx_color_blend(int id) { return texstack[id*TEXSTACK_STRIDE+9]; }
float tx_direct_lobe_power(int id) { return texstack[id*TEXSTACK_STRIDE+10]; }
float tx_model_flags(int id) { return texstack[id*TEXSTACK_STRIDE+11]; }
float tx_remit_gain    (int id) { return texstack[id*TEXSTACK_STRIDE+12]; }
float tx_bulb_radius_mm(int id) { return texstack[id*TEXSTACK_STRIDE+13]; }
// [14] remit_decay — reserved for temporal FIR, not yet sampled
float tx_translucence_gain(int id) { return texstack[id*TEXSTACK_STRIDE+15]; }

/* ── Inputs / uniforms ────────────────────────────────────────────────────── */

in  vec3     vNormV;
in  vec3     vPosV;
in  vec3     vPosObj;
flat in int  vMatId;
flat in int  vGroupId;
flat in int  vCullImmune;
in  vec2     vUv;
in  vec3     vVelocityW;   // world-space surface velocity (m/s) for optical effects

out vec4 FragColor;

// Multi-light array — every light is a real emitter.
#define MAX_LIGHTS 100
uniform int   uNumLights;
uniform vec3  uLightPos      [MAX_LIGHTS];   // view-space emitter position
uniform vec3  uLightColor    [MAX_LIGHTS];   // linear sRGB
uniform float uLightIntensity[MAX_LIGHTS];
uniform int   uLightGroupId  [MAX_LIGHTS];
uniform float uLightCalibration = 1.0;
uniform mat3  uCatCcmMatrix = mat3(1.0);

// ── UV emission/depth/remit texture stack (Stage 2 of the action plan) ─────
// Channel layout per UV_EMISSION_STACK_PLAN.md:
//   R = direct gain    G = diffusion gain    B = saturation_adj    A = dim
// Default 1×1×1 texel (0, 1, 0.5, 1) makes this an exact identity for any
// caller that does not bind aUv / does not author an emission map: the
// 'diffuse' branch passes the raw mat.emission through unchanged, the
// 'direct' branch contributes nothing, and the saturation mix is identity.
uniform sampler2DArray uEmitUv;
uniform sampler2DArray uColorUv;
uniform sampler2DArray uDepthUv;
uniform sampler2DArray uRemitUv;
uniform bool uEnableSpecular = true;
uniform bool uEnableEmissionDirect = false;
uniform int  uRenderPass = 0;  // 0=all, 1=opaque-only (alpha>=0.85), 2=transparent-only
uniform sampler3D uFieldVolume;
uniform float uFieldGain = 0.0;

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
    // Use triangle orientation for front/back classification; per-fragment
    // normal-sign tests can flicker on curved shells and create black speckle.
    bool front_facing = gl_FrontFacing;
    vec3 N = normalize(front_facing ? vNormV : -vNormV);
    vec3 V = normalize(-vPosV);
    float NdotV = max(dot(N, V), 0.0);
    // Selective back-face culling in shader: cull by default, keep both
    // sides only when the triangle is explicitly marked cull-immune.
    if (vCullImmune == 0 && !front_facing) {
        discard;
    }

    // Material parameters
    vec3  albedo   = mat_albedo   (id);
    float metallic = mat_metallic (id);
    float ior      = mat_ior      (id);
    float opacity  = mat_opacity  (id);
    vec3  emission = mat_emission (id);

    // Physics profile — selects equation family for this material
    //   0 = standard dielectric/conductor (default)
    //   1 = emissive lobe: emission cone shaped by NdotV^direct_lobe_power
    //   2 = translucent SSS: thickness_mm drives wrap lighting (4× scale)
    //   3 = frosted scatter: scatter_mask (depth_uv.A) softens specular
    int   profile_id = int(tx_model_flags(id)) & 0xFF;
    float lobe_p     = max(1.0, tx_direct_lobe_power(id));

    // Color UV override
    float color_layer = tx_color_layer(id);
    if (color_layer >= 0.0) {
        vec4 c_uv = texture(uColorUv, vec3(vUv, color_layer));
        albedo = mix(albedo, c_uv.rgb, clamp(c_uv.a * tx_color_blend(id), 0.0, 1.0));
    }

    // Depth UV — all four channels read up-front so profile 3 can pre-modify specular
    //   R = depth offset      G = thickness       B = translucence mask   A = scatter mask
    float depth_layer = tx_depth_layer(id);
    float depth_mm = 0.0, thickness_mm = 0.0;
    float translucence_mask = 1.0, scatter_mask = 0.0;
    if (depth_layer >= 0.0) {
        vec4 d_uv = texture(uDepthUv, vec3(vUv, depth_layer));
        depth_mm          = d_uv.r * tx_depth_scale_mm(id)     + tx_depth_bias_mm(id);
        thickness_mm      = d_uv.g * tx_thickness_scale_mm(id) + tx_thickness_bias_mm(id);
        translucence_mask = d_uv.b;
        scatter_mask      = d_uv.a;
    }

    float specstr  = ph_specstr  (id);
    float shini    = ph_shininess(id);
    vec3  inner    = ph_inner    (id);

    // Profile 3: frosted scatter — reduce specular sharpness before the lights loop
    if (profile_id == 3 && scatter_mask > 0.001) {
        shini   = max(1.0,  shini   * (1.0 - scatter_mask * 0.88));
        specstr = max(0.0,  specstr * (1.0 - scatter_mask * 0.70));
    }

    // Front / back face base colour
    vec3 base = front_facing ? albedo : inner;

    // Schlick Fresnel — F0 derived from IOR (dielectric) blended with albedo (conductor)
    float ior_f0 = (ior - 1.0) / (ior + 1.0);
    ior_f0 *= ior_f0;
    vec3 F0 = mix(vec3(ior_f0), albedo, metallic);
    vec3  F_vec  = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // Direct illumination from real emitters only.
    vec3 col = vec3(0.0);
    vec3 spec_col_acc = vec3(0.0);
    float spec_acc = 0.0;
    int n = min(uNumLights, MAX_LIGHTS);
    for (int i = 0; i < n; ++i) {
        if (uLightGroupId[i] == vGroupId) {
            continue;
        }
        vec3  Lvec  = uLightPos[i] - vPosV;
        float Lr2   = max(dot(Lvec, Lvec), 1e-8);
        vec3  L     = Lvec * inversesqrt(Lr2);
        float emitter_r2 = max(uLightIntensity[i], 0.0) * 0.0795774715; // area / (4*pi)
        float atten_r2 = max(Lr2 + emitter_r2, 1e-8);
        vec3  Lcol  = uLightColor[i] * ((uLightIntensity[i] * uLightCalibration) / atten_r2);
        vec3  H     = normalize(L + V);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float spec  = pow(NdotH, max(shini, 1.0));

        col += base * Lcol * NdotL;

        if (uEnableSpecular) {
            col += Lcol * F_vec * (specstr * spec);
        }

        spec_col_acc += Lcol * spec;
        spec_acc     += spec;
    }

    // ── Translucence (profiles 0/1: minimal; profile 2 = SSS: 4× scale) ─
    float t_scale = (profile_id == 2) ? 0.08 : 0.02;
    float translucence = max(0.0, thickness_mm) * tx_translucence_gain(id) * translucence_mask;
    col += base * translucence * t_scale;

    // ── Self-emission via UV texture-pack ────────────────────────────────
    float emit_layer = tx_emit_layer(id);
    vec4  e_uv     = (emit_layer >= 0.0)
                   ? texture(uEmitUv, vec3(vUv, emit_layer))
                   : vec4(0.0, 1.0, 0.5, 1.0);
    vec3  emit_dim = emission * (e_uv.a * tx_emit_gain(id));
    vec3  spec_dir = (spec_acc > 0.0) ? (spec_col_acc / spec_acc) : vec3(0.0);

    // Profile 1: forward-emission cone shaped by NdotV^lobe_p
    // Profiles 0/2/3: standard specular-coupled direct emission
    vec3 emit_dir;
    if (profile_id == 1) {
        float beam = pow(max(0.0, NdotV), lobe_p);
        emit_dir = emit_dim * (e_uv.r * beam);
    } else {
        emit_dir = uEnableEmissionDirect ? emit_dim * e_uv.r * spec_dir : vec3(0.0);
    }

    vec3  emit_dif = emit_dim * e_uv.g;
    vec3  emit_mix = emit_dir + emit_dif;
    float emit_lum = dot(emit_mix, vec3(0.2126, 0.7152, 0.0722));
    vec3  emit_out = mix(vec3(emit_lum), emit_mix, 0.5 + e_uv.b);

    // depth_uv.A = bulb_radius for profile 1. Raw value, no relationship to
    // other parameters. R²/(R²+d²): full emission at d=0, half at d=bulb_r.
    if (profile_id == 1 && scatter_mask > 0.0) {
        float d = max(0.0, depth_mm);
        emit_out *= (scatter_mask * scatter_mask) / (scatter_mask * scatter_mask + d * d);
    }

    col += emit_out;

    // Profile 3: frosted scatter — add diffuse halo from emission luminance
    if (profile_id == 3 && scatter_mask > 0.001) {
        col += base * (scatter_mask * emit_lum * 0.40 + scatter_mask * 0.025);
        col = max(col, vec3(0.0));
    }

    // ── Enamel thin-film coating ──────────────────────────────────────────
    vec3  spec_col = spec_col_acc;
    float spec     = spec_acc;

    float enam_thick = en_thick(id);
    if (enam_thick > 0.0) {
        float enam_ior_v  = en_ior  (id);
        vec3  enam_c      = en_color(id);

        float opd_nm  = 2.0 * enam_ior_v * enam_thick * NdotV;
        vec3  fringe   = thin_film_fringe(opd_nm);
        float fringe_w = clamp(enam_thick / 800.0, 0.0, 0.45);

        vec3 tinted = col * enam_c * (vec3(1.0) + fringe * fringe_w);

        float eF0 = (enam_ior_v - 1.0) / (enam_ior_v + 1.0);
        eF0 *= eF0;
        float eF  = schlick(eF0, NdotV);
        tinted   += spec_col * (eF * 0.28 * spec);

        col = mix(col, tinted, 0.35);
    }

    // 3-D field volume — sample energy density at this fragment's scene position
    // and add as glow.  vPosObj is already in [0,1]³ (= field texture UVW space).
    if (uFieldGain > 0.0) {
        col += texture(uFieldVolume, vPosObj).rgb * uFieldGain;
    }

    // Color compensation equation: C' = CAT_CCM * C.
    col = uCatCcmMatrix * col;

    // ── Opacity ────────────────────────────────────────────────────────────
    float alpha = opacity;
    if (uRenderPass == 1 && alpha < 0.85) discard;
    if (uRenderPass == 2 && alpha >= 0.85) discard;

    vec3 lin = max(col, vec3(0.0));
    vec3 s1 = lin * 12.92;
    vec3 s2 = 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055;
    vec3 srgb = mix(s1, s2, step(vec3(0.0031308), lin));
    FragColor = vec4(srgb, alpha);
}
