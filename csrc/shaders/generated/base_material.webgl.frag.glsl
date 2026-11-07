#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
precision highp sampler3D;
precision highp isampler2D;
precision highp usampler2D;

// Turing WebGL storage transport generated from project IR.
uniform highp sampler2D turing_ssbo_pbr;
float turing_ssbo_load_pbr(int index) {
    ivec2 extent = textureSize(turing_ssbo_pbr, 0);
    int safe_index = max(index, 0);
    ivec2 coordinate = ivec2(safe_index % extent.x, safe_index / extent.x);
    return texelFetch(turing_ssbo_pbr, coordinate, 0).r;
}

uniform highp sampler2D turing_ssbo_phong;
float turing_ssbo_load_phong(int index) {
    ivec2 extent = textureSize(turing_ssbo_phong, 0);
    int safe_index = max(index, 0);
    ivec2 coordinate = ivec2(safe_index % extent.x, safe_index / extent.x);
    return texelFetch(turing_ssbo_phong, coordinate, 0).r;
}

uniform highp sampler2D turing_ssbo_enamel;
float turing_ssbo_load_enamel(int index) {
    ivec2 extent = textureSize(turing_ssbo_enamel, 0);
    int safe_index = max(index, 0);
    ivec2 coordinate = ivec2(safe_index % extent.x, safe_index / extent.x);
    return texelFetch(turing_ssbo_enamel, coordinate, 0).r;
}

uniform highp sampler2D turing_ssbo_texstack;
float turing_ssbo_load_texstack(int index) {
    ivec2 extent = textureSize(turing_ssbo_texstack, 0);
    int safe_index = max(index, 0);
    ivec2 coordinate = ivec2(safe_index % extent.x, safe_index / extent.x);
    return texelFetch(turing_ssbo_texstack, coordinate, 0).r;
}

uniform highp sampler2D turing_ssbo_thermal_k;
float turing_ssbo_load_thermal_k(int index) {
    ivec2 extent = textureSize(turing_ssbo_thermal_k, 0);
    int safe_index = max(index, 0);
    ivec2 coordinate = ivec2(safe_index % extent.x, safe_index / extent.x);
    return texelFetch(turing_ssbo_thermal_k, coordinate, 0).r;
}

#line 2
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
 *   uLightSpotDir[N]  — view-space unit direction a SPOT light faces; zero
 *                       (the default, never set) = an omnidirectional point light
 *   uLightSpotCone[N] — (cos inner half-angle, cos outer half-angle): full
 *                       light inside the inner cone, smoothly none past the outer
 *
 * Optional variant PLUCK_TRANSMISSION (a #define the host adds, compiled as a
 * second program; without it this file is the original shader, byte for byte
 * in behaviour). It turns on the three material fields the tables always carried
 * but the base program never read:
 *   transmission (PBR record, float 5) lights on the far side of the surface reach the eye
 *                           through it: tinted by albedo^thickness (Beer-Lambert,
 *                           albedo = per-mm transmittance, path >= 1 mm), times
 *                           (1 - Fresnel), a diffuse part (roughness) plus a
 *                           forward lobe (1 - roughness) at the Phong shininess.
 *                           Alpha is capped at 1 - 0.8 * transmission (the
 *                           pre-archive rule) so a transmissive surface never
 *                           hides what is behind it more than that.
 *   translucence (texstack 15)  thickness_mm * gain * mask * (0.02, profile 2: 0.08)
 *                           is the fraction of back-lighting that scatters
 *                           through the body (base-tinted), per light behind the
 *                           surface, instead of a constant light-free glow.
 *   remit (texstack 3,12,13,14) re-emission of absorbed light: steady state
 *                           gain * mask(uRemitUv.a at layer) * albedo * absorbed
 *                           irradiance. Attack/decay (texstack 13, 14) shape only
 *                           the transient; the base renderers keep no per-material
 *                           state texture, so they do not enter here.
 * Light leaving the body this way is composited additively over what is behind
 * (alpha is raised just enough to carry it), not scaled by the surface's alpha.
 *
 * There is no global scene tint or hardcoded sun. The material's bounded
 * Phong ambient term is treated as an authored indirect-light floor so shadow
 * structure remains legible; all direct illumination comes from host emitters.
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
// project IR lowered std430 binding 10 (PBRChunk.pbr) to turing_ssbo_pbr

// PhongRecord  (8 floats)
// [0] ambient  [1] spec_strength  [2] shininess  [3] _reserved
// [4..6] inner_color.rgb  [7] _pad
// project IR lowered std430 binding 11 (PhongChunk.phong) to turing_ssbo_phong

// EnamelRecord  (8 floats)
// [0] thickness_nm  [1] ior_real  [2] ior_imag  [3] roughness
// [4..6] color.rgb  [7] _pad
// project IR lowered std430 binding 14 (EnamelChunk.enamel) to turing_ssbo_enamel

// TextureStackRecord (16 floats)
// [0] emit_uv_layer  [1] color_uv_layer  [2] depth_uv_layer  [3] remit_uv_layer
// [4] depth_scale_mm [5] thickness_scale_mm [6] depth_bias_mm [7] thickness_bias_mm
// [8] emit_gain [9] color_blend [10] direct_lobe_power [11] model_flags_or_indices (low byte: profile; bit 8: emission through the colour layer)
// [12] remit_gain [13] remit_attack [14] remit_decay [15] translucence_gain
// project IR lowered std430 binding 15 (TextureStackChunk.texstack) to turing_ssbo_texstack

// ThermalChunk (1 float per thermal id): a surface temperature in kelvin,
// indexed by the vertex's own aGroupId. When uThermalEmission is on and
// the id is > 0, the fragment self-emits a Planck blackbody at that
// temperature (three visible bands) -- the host uploads kelvins, nothing
// else; colour, brightness and the visible floor are all evaluated here.
// project IR lowered std430 binding 16 (ThermalChunk.thermal_k) to turing_ssbo_thermal_k
uniform bool  uThermalEmission;
uniform float uThermalGain;    // red-band radiance at uThermalRefK, in scene light units
uniform float uThermalRefK;
uniform float uThermalFloorK;   // below this the visible-band radiance is negligible

float planck_rel(float lambda_um, float t_k) {
    float x = 14387.77 / (lambda_um * t_k);
    if (x > 80.0) return 0.0;
    return 1.0 / (pow(lambda_um, 5.0) * (exp(x) - 1.0));
}

vec3 thermal_emission(float t_k) {
    if (t_k < uThermalFloorK) return vec3(0.0);
    float ref = planck_rel(0.610, uThermalRefK);
    vec3 e = vec3(planck_rel(0.610, t_k), planck_rel(0.550, t_k), planck_rel(0.465, t_k)) / max(ref, 1e-30);
    return min(e * uThermalGain, vec3(1000.0));
}

/* ── Accessors ────────────────────────────────────────────────────────────── */

#define PBR_STRIDE    16
#define PHONG_STRIDE   8
#define ENAM_STRIDE    8
#define TEXSTACK_STRIDE 16

vec3  mat_albedo   (int id) { int b=id*PBR_STRIDE;   return vec3(turing_ssbo_load_pbr(int(b)),   turing_ssbo_load_pbr(int(b+1)), turing_ssbo_load_pbr(int(b+2))); }
float mat_roughness(int id) { return turing_ssbo_load_pbr(int(id*PBR_STRIDE+3)); }
float mat_metallic (int id) { return turing_ssbo_load_pbr(int(id*PBR_STRIDE+4)); }
float mat_trans    (int id) { return turing_ssbo_load_pbr(int(id*PBR_STRIDE+5)); }
float mat_ior      (int id) { return turing_ssbo_load_pbr(int(id*PBR_STRIDE+6)); }
float mat_opacity  (int id) { return turing_ssbo_load_pbr(int(id*PBR_STRIDE+7)); }
vec3  mat_emission (int id) { int b=id*PBR_STRIDE+8; return vec3(turing_ssbo_load_pbr(int(b)),   turing_ssbo_load_pbr(int(b+1)), turing_ssbo_load_pbr(int(b+2))); }

float ph_ambient  (int id) { return turing_ssbo_load_phong(int(id*PHONG_STRIDE+0)); }
float ph_specstr  (int id) { return turing_ssbo_load_phong(int(id*PHONG_STRIDE+1)); }
float ph_shininess(int id) { return turing_ssbo_load_phong(int(id*PHONG_STRIDE+2)); }
vec3  ph_inner    (int id) { int b=id*PHONG_STRIDE+4; return vec3(turing_ssbo_load_phong(int(b)),turing_ssbo_load_phong(int(b+1)),turing_ssbo_load_phong(int(b+2))); }

float en_thick (int id) { return turing_ssbo_load_enamel(int(id*ENAM_STRIDE+0)); }
float en_ior   (int id) { return turing_ssbo_load_enamel(int(id*ENAM_STRIDE+1)); }
vec3  en_color (int id) { int b=id*ENAM_STRIDE+4; return vec3(turing_ssbo_load_enamel(int(b)),turing_ssbo_load_enamel(int(b+1)),turing_ssbo_load_enamel(int(b+2))); }
float tx_emit_layer(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+0)); }
float tx_color_layer(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+1)); }
float tx_depth_layer(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+2)); }
float tx_remit_layer(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+3)); }
float tx_depth_scale_mm(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+4)); }
float tx_thickness_scale_mm(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+5)); }
float tx_depth_bias_mm(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+6)); }
float tx_thickness_bias_mm(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+7)); }
float tx_emit_gain(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+8)); }
float tx_color_blend(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+9)); }
float tx_direct_lobe_power(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+10)); }
float tx_model_flags(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+11)); }
float tx_remit_gain    (int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+12)); }
float tx_bulb_radius_mm(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+13)); }
// [14] remit_decay — reserved for temporal FIR, not yet sampled
float tx_translucence_gain(int id) { return turing_ssbo_load_texstack(int(id*TEXSTACK_STRIDE+15)); }

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
uniform vec3  uLightSpotDir  [MAX_LIGHTS];   // view-space spot direction (zero: point light)
uniform vec2  uLightSpotCone [MAX_LIGHTS];   // cos(inner), cos(outer) half-angles
uniform float uLightCalibration;
uniform mat3  uCatCcmMatrix;
uniform float uExposure;

// Optional first-key depth shadow. Native hosts that do not define
// PLUCK_SHADOW_MAP retain the original ABI; WebGL adapters can opt in and
// provide the depth pre-pass without forking the material equations.
#ifdef PLUCK_SHADOW_MAP
uniform sampler2DArray uShadowMaps;
uniform bool uShadowEnabled;
uniform int uShadowLightCount;
uniform mat4 uShadowMatrices[8];
uniform vec2 uShadowTexelSize;
uniform float uShadowBias;
#endif

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
uniform bool uEnableSpecular;
uniform bool uEnableEmissionDirect;
uniform int  uRenderPass;  // 0=all, 1=opaque-only (alpha>=0.85), 2=transparent-only
uniform sampler3D uFieldVolume;
uniform float uFieldGain;

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

#ifdef PLUCK_SHADOW_MAP
float light_shadow_visibility(int light_index, vec3 N, vec3 L) {
    if (!uShadowEnabled || light_index >= uShadowLightCount) return 1.0;
    vec4 shadow_coord = uShadowMatrices[light_index] * vec4(vPosObj, 1.0);
    vec3 projected = shadow_coord.xyz / max(shadow_coord.w, 1e-6);
    projected = projected * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        projected.x <= 0.0 || projected.x >= 1.0 ||
        projected.y <= 0.0 || projected.y >= 1.0) return 1.0;
    float bias = max(uShadowBias * (1.0 - max(dot(N, L), 0.0)), uShadowBias * 0.25);
    float visible = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float depth = texture(uShadowMaps, vec3(
                projected.xy + vec2(float(x), float(y)) * uShadowTexelSize,
                float(light_index))).r;
            visible += projected.z - bias <= depth ? 1.0 : 0.0;
        }
    }
    return 0.18 + 0.82 * visible / 9.0;
}
#endif

#ifdef PLUCK_TRANSMISSION
// The same exposure / ACES / sRGB encode as the end of main(), for light that
// leaves the body and is composited on its own.
vec3 pluck_display(vec3 c) {
    vec3 lin = max(c, vec3(0.0)) * max(uExposure, 0.0);
    lin = clamp((lin * (2.51 * lin + 0.03)) /
                (lin * (2.43 * lin + 0.59) + 0.14), 0.0, 1.0);
    vec3 s1 = lin * 12.92;
    vec3 s2 = 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055;
    return mix(s1, s2, step(vec3(0.0031308), lin));
}
#endif

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
    // the colour layer as a FILM over the emission (model flag bit 8, see below): white where no layer applies
    vec3 film = vec3(1.0);
    if (color_layer >= 0.0) {
        vec4 c_uv = texture(uColorUv, vec3(vUv, color_layer));
        float c_w = clamp(c_uv.a * tx_color_blend(id), 0.0, 1.0);
        albedo = mix(albedo, c_uv.rgb, c_w);
        film = mix(vec3(1.0), c_uv.rgb, c_w);
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

#ifdef PLUCK_TRANSMISSION
    float m_trans   = clamp(mat_trans(id), 0.0, 1.0);
    float m_rough   = clamp(mat_roughness(id), 0.0, 1.0);
    float tl_scale  = (profile_id == 2) ? 0.08 : 0.02;
    float tl_frac   = min(max(0.0, thickness_mm) * tx_translucence_gain(id) * translucence_mask * tl_scale, 1.0);
    float remit_g   = max(0.0, tx_remit_gain(id));
    bool  use_tr    = (m_trans > 0.0) || (tl_frac > 0.0) || (remit_g > 0.0);
    vec3  tr_acc    = vec3(0.0);   // radiance from lights behind the surface, through the body
    vec3  sss_acc   = vec3(0.0);   // back-lighting scattered by the body
    vec3  remit_acc = vec3(0.0);   // irradiance the body absorbs and re-emits
#endif

    // Direct illumination from real emitters only.
    vec3 col = base * clamp(ph_ambient(id), 0.0, 0.12);
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
        // Spot cone: a light with a direction lights only the fragments inside
        // its cone, fading across the edge between the inner and outer angles.
        if (dot(uLightSpotDir[i], uLightSpotDir[i]) > 0.25) {
            float cos_a = dot(-L, normalize(uLightSpotDir[i]));
            Lcol *= smoothstep(uLightSpotCone[i].y, uLightSpotCone[i].x, cos_a);
        }
        vec3  H     = normalize(L + V);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float spec  = pow(NdotH, max(shini, 1.0));
        float visibility = 1.0;
#ifdef PLUCK_SHADOW_MAP
        visibility = light_shadow_visibility(i, N, L);
#endif

        col += base * Lcol * NdotL * visibility;

        if (uEnableSpecular) {
            col += Lcol * F_vec * (specstr * spec) * visibility;
        }

        spec_col_acc += Lcol * spec * visibility;
        spec_acc     += spec * visibility;

#ifdef PLUCK_TRANSMISSION
        if (use_tr) {
            float back = max(-dot(N, L), 0.0);
            if (back > 0.0) {
                float fwd = pow(max(dot(-L, V), 0.0), max(shini, 1.0));
                tr_acc  += Lcol * ((back * m_rough + fwd * (1.0 - m_rough)) * m_trans);
                sss_acc += Lcol * back;
            }
            if (remit_g > 0.0) {
                remit_acc += Lcol * abs(dot(N, L));
            }
        }
#endif
    }

#ifdef PLUCK_TRANSMISSION
    // ── Light leaving the body: transmission, translucence, remit ───────────
    vec3 col_t = vec3(0.0);
    if (use_tr) {
        vec3 T = pow(max(albedo, vec3(1e-3)), vec3(max(thickness_mm, 1.0)));
        col_t = T * (vec3(1.0) - F_vec) * tr_acc;
        col_t += base * sss_acc * tl_frac;
        if (remit_g > 0.0) {
            float remit_layer = tx_remit_layer(id);
            float remit_mask = (remit_layer >= 0.0) ? texture(uRemitUv, vec3(vUv, remit_layer)).a : 1.0;
            col_t += albedo * remit_acc * (remit_g * remit_mask);
        }
    }
#else
    // ── Translucence (profiles 0/1: minimal; profile 2 = SSS: 4× scale) ─
    float t_scale = (profile_id == 2) ? 0.08 : 0.02;
    float translucence = max(0.0, thickness_mm) * tx_translucence_gain(id) * translucence_mask;
    col += base * translucence * t_scale;
#endif

    // ── Self-emission via UV texture-pack ────────────────────────────────
    float emit_layer = tx_emit_layer(id);
    vec4  e_uv     = (emit_layer >= 0.0)
                   ? texture(uEmitUv, vec3(vUv, emit_layer))
                   : vec4(0.0, 1.0, 0.5, 1.0);
    vec3  emit_dim = emission * (e_uv.a * tx_emit_gain(id));
    // Model flag bit 8 (value 256, above the profile byte): EMISSION THROUGH THE FILM -- the surface's own light leaves
    // through its colour layer, tinted by it (a backlit transparency).  Off for every material that does not set it.
    if (((int(tx_model_flags(id)) >> 8) & 1) == 1) emit_dim *= film;
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

    // ── Thermal self-emission off the ThermalChunk SSBO ──────────────────
    if (uThermalEmission && vGroupId > 0) {
        col += thermal_emission(turing_ssbo_load_thermal_k(int(vGroupId)));
    }

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
#ifdef PLUCK_TRANSMISSION
    if (m_trans > 0.0) alpha = min(opacity, 1.0 - 0.8 * m_trans);
#endif
    if (uRenderPass == 1 && alpha < 0.85) discard;
    if (uRenderPass == 2 && alpha >= 0.85) discard;

    vec3 lin = max(col, vec3(0.0)) * max(uExposure, 0.0);
    // ACES approximation keeps energetic emitters colourful while preserving
    // nearby surface gradients instead of clipping entire regions to white.
    lin = clamp((lin * (2.51 * lin + 0.03)) /
                (lin * (2.43 * lin + 0.59) + 0.14), 0.0, 1.0);
    vec3 s1 = lin * 12.92;
    vec3 s2 = 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055;
    vec3 srgb = mix(s1, s2, step(vec3(0.0031308), lin));
#ifdef PLUCK_TRANSMISSION
    if (use_tr) {
        // out = alpha * surface + light_t + (1 - alpha) * behind: raise alpha only
        // as far as the body's own light needs, and fold the light into colour.
        vec3 st = pluck_display(uCatCcmMatrix * col_t);
        float a_t = max(alpha, min(1.0, max(st.r, max(st.g, st.b))));
        srgb = min((srgb * alpha + st) / max(a_t, 1e-4), vec3(1.0));
        alpha = a_t;
    }
#endif
    FragColor = vec4(srgb, alpha);
}
