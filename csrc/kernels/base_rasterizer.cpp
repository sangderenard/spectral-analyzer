#define _USE_MATH_DEFINES
/**
 * base_rasterizer.cpp — Tile-parallel software rasterizer.
 *
 * Design choices:
 *   - Eigen::Matrix4f / Vector4f for perspective projection (SIMD via AVX).
 *   - Static thread pool (same pattern as acoustic_amr.cpp) persists for the
 *     process lifetime.
 *   - Tiles are 16×16 pixels by default.  Each pixel belongs to exactly one
 *     tile, so the framebuffer write phase has zero contention.
 *   - Depth buffer is per-rasterizer-state (flat float array); no atomics are
 *     needed because tiles partition the screen.
 *   - Phong shading consumes the same PhongRecord / PBRBaseRecord / EnamelRecord
 *     float arrays that material_db.py exports for the SSBO path.
 *   - Schlick Fresnel from IOR — no grain texture (this is a materials renderer,
 *     not a guitar prop renderer).
 *   - Thin-film enamel: iridescence via three-cosine OPD approximation + gloss.
 *   - sRGB gamma correction on readback (not inline — avoids repeated pow in
 *     the inner loop when the caller reads back only once per frame).
 */

#include "base_rasterizer.h"

#include <Eigen/Dense>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Array8f = Eigen::Array<float, 8, 1>;
using Mask8   = Eigen::Array<bool, 8, 1>;

/* ═══════════════════════════════════════════════════════════════════════════
 * Thread pool  (identical pattern to acoustic_amr.cpp)
 * ═══════════════════════════════════════════════════════════════════════════ */
namespace {

struct RastThreadPool {
    int n;
    std::vector<std::thread> workers;
    std::mutex               mx;
    std::condition_variable  cv_wake, cv_done;
    std::function<void(int)> job;       // single-tile callback
    int              job_n  = 0;
    int              active = 0;
    bool             quit   = false;
    std::atomic<int> next_tile{0};      // dynamic tile-stealing counter

    explicit RastThreadPool(int n_threads) : n(n_threads) {
        workers.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i)
            workers.emplace_back(&RastThreadPool::loop, this);
    }

    ~RastThreadPool() {
        { std::unique_lock<std::mutex> lk(mx); quit = true; }
        cv_wake.notify_all();
        for (auto& t : workers) t.join();
    }

    void loop() {
        for (;;) {
            std::function<void(int)> fn;
            int local_job_n;
            {
                std::unique_lock<std::mutex> lk(mx);
                cv_wake.wait(lk, [this]{ return quit || next_tile.load() < job_n; });
                if (quit) return;
                ++active;
                fn = job;
                local_job_n = job_n;
            }
            // Each thread steals one tile at a time until the queue is drained.
            int tile_id;
            while ((tile_id = next_tile.fetch_add(1, std::memory_order_relaxed))
                   < local_job_n) {
                fn(tile_id);
            }
            {
                std::lock_guard<std::mutex> lk(mx);
                --active;
                if (next_tile.load() >= job_n && active == 0)
                    cv_done.notify_one();
            }
        }
    }

    void run(int total, std::function<void(int)> fn) {
        if (total <= 0) return;
        {
            std::unique_lock<std::mutex> lk(mx);
            job = std::move(fn); job_n = total; active = 0;
            next_tile.store(0, std::memory_order_relaxed);
        }
        cv_wake.notify_all();
        std::unique_lock<std::mutex> lk(mx);
        cv_done.wait(lk, [this]{
            return next_tile.load() >= job_n && active == 0;
        });
    }
};

static RastThreadPool* g_pool = nullptr;
static std::once_flag   g_pool_once;

static RastThreadPool& rast_pool() {
    std::call_once(g_pool_once, [] {
        int nt = static_cast<int>(std::thread::hardware_concurrency());
        if (nt < 1) nt = 1;
        g_pool = new RastThreadPool(nt);
        std::fprintf(stderr, "[rast_pool] tile thread pool: %d threads\n", nt);
        std::fflush(stderr);
    });
    return *g_pool;
}

template<typename F>
static void parallel_for(int n, F&& fn) {
    if (n <= 0) return;
    rast_pool().run(n, [&](int tile_id){ fn(tile_id); });
}

} // namespace thread pool

/* ═══════════════════════════════════════════════════════════════════════════
 * Material accessors  (match material_db.py ctypes layout)
 * ═══════════════════════════════════════════════════════════════════════════ */

static const int PBR_STRIDE   = 16;
static const int PHONG_STRIDE =  8;
static const int ENAM_STRIDE  =  8;
static const int TEXSTACK_STRIDE = 16;

// PBR accessors
static inline Vector3f pbr_albedo  (const float* pbr, int id)
    { const float* b = pbr + id*PBR_STRIDE; return {b[0],b[1],b[2]}; }
static inline float    pbr_roughness(const float* pbr, int id) { return pbr[id*PBR_STRIDE+3]; }
static inline float    pbr_metallic (const float* pbr, int id) { return pbr[id*PBR_STRIDE+4]; }
static inline float    pbr_trans    (const float* pbr, int id) { return pbr[id*PBR_STRIDE+5]; }
static inline float    pbr_ior      (const float* pbr, int id) { return pbr[id*PBR_STRIDE+6]; }
static inline float    pbr_opacity  (const float* pbr, int id) { return pbr[id*PBR_STRIDE+7]; }
static inline Vector3f pbr_emission (const float* pbr, int id)
    { const float* b = pbr + id*PBR_STRIDE+8; return {b[0],b[1],b[2]}; }

static inline float tex_emit_layer(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+0]; }
static inline float tex_color_layer(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+1]; }
static inline float tex_depth_layer(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+2]; }
static inline float tex_remit_layer(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+3]; }
static inline float tex_depth_scale_mm(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+4]; }
static inline float tex_thickness_scale_mm(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+5]; }
static inline float tex_depth_bias_mm(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+6]; }
static inline float tex_thickness_bias_mm(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+7]; }
static inline float tex_emit_gain(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+8]; }
static inline float tex_color_blend(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+9]; }
static inline float tex_direct_lobe_power(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+10]; }
static inline float tex_model_flags(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+11]; }
static inline float tex_remit_gain    (const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+12]; }
static inline float tex_bulb_radius_mm(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+13]; }
// [14] remit_decay — reserved for temporal FIR, not yet sampled
static inline float tex_translucence_gain(const float* tx, int id) { return tx[id*TEXSTACK_STRIDE+15]; }

// Phong accessors
static inline float    ph_ambient   (const float* ph, int id) { return ph[id*PHONG_STRIDE+0]; }
static inline float    ph_specstr   (const float* ph, int id) { return ph[id*PHONG_STRIDE+1]; }
static inline float    ph_shininess (const float* ph, int id) { return ph[id*PHONG_STRIDE+2]; }
static inline Vector3f ph_inner     (const float* ph, int id)
    { const float* b = ph + id*PHONG_STRIDE+4; return {b[0],b[1],b[2]}; }

// Enamel accessors
static inline float    en_thick  (const float* en, int id) { return en[id*ENAM_STRIDE+0]; }
static inline float    en_ior    (const float* en, int id) { return en[id*ENAM_STRIDE+1]; }
static inline Vector3f en_color  (const float* en, int id)
    { const float* b = en + id*ENAM_STRIDE+4; return {b[0],b[1],b[2]}; }

/* ═══════════════════════════════════════════════════════════════════════════
 * Shading  (Phong + Schlick Fresnel + enamel thin-film)
 * ═══════════════════════════════════════════════════════════════════════════ */

struct SceneParams {
    // ── Multi-light array (real per-emitter colour & direction) ─────────
    // Lights are *directional* (parallel rays, intensity is unitless gain).
    // Each cluster of co-located emitters in the calling Python becomes one
    // entry: e.g. emerald cluster → (dir = normalised mean orbit-position,
    // colour = emerald's prebaked sRGB, intensity = 1.0).  The rasterizer
    // never averages colours — each light contributes independently.
    //
    // There are NO other light sources.  No ambient.  No proxy bounce.  No
    // hardcoded sun.  A non-emissive surface in a scene with zero lights
    // renders pure black — that is physically correct.
    // Compile-time ceiling for the fixed-size scene arrays.  The runtime
    // cap is BaseRasterizerState::max_lights and may be set lower (or up
    // to this ceiling) via br_set_max_lights().
    static constexpr int MAX_LIGHTS = 100;
    int      n_lights = 0;
    Vector3f light_positions[MAX_LIGHTS]; // view-space positions (point lights)
    Vector3f light_colors   [MAX_LIGHTS]; // linear RGB, no clamp
    float    light_intensities[MAX_LIGHTS] = {0};
};

struct MaterialSample {
    Vector3f albedo;
    float    roughness;
    float    metallic;
    float    ior;
    float    opacity;
    Vector3f emission;

    float    ambient;
    float    spec_strength;
    float    shininess;
    Vector3f inner_color;

    float    enamel_thickness;
    float    enamel_ior;
    Vector3f enamel_color;
};

struct GeomTerms {
    Vector3f N;
    Vector3f V;
    Vector3f base;
    float    NdotV;
};

struct FresnelTerms {
    Vector3f F0;
    Vector3f F;
};

// All four channels of depth_uv, after scale/bias application.
struct DepthSample {
    float depth_mm          = 0.0f;
    float thickness_mm      = 0.0f;
    float translucence_mask = 1.0f; // depth_uv.B: spatial SSS weight [0,1]
    float scatter_mask      = 0.0f; // depth_uv.A: frost / scatter mask [0,1]
};

/* Three-cosine thin-film iridescence (period 550 nm in OPD space). */
static inline Vector3f thin_film_fringe(float opd_nm) {
    constexpr float TWO_PI = 6.28318530718f;
    float f = opd_nm * (TWO_PI / 550.0f);
    return {
        0.5f + 0.5f * std::cos(f),
        0.5f + 0.5f * std::cos(f - 2.09439510239f),
        0.5f + 0.5f * std::cos(f - 4.18879020479f)
    };
}

/* Schlick Fresnel from scalar F0. */
static inline float schlick(float f0, float cos_theta) {
    float t = 1.0f - std::max(0.0f, cos_theta);
    float t2 = t * t;
    return f0 + (1.0f - f0) * (t2 * t2 * t);
}

static inline float pow_unit_fast_math(float x, float exponent) {
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return std::exp2(exponent * std::log2(x));
}

static inline Array8f pow_unit_fast_math(const Array8f& x, float exponent) {
    const Array8f clamped = x.max(Array8f::Constant(1e-20f)).min(Array8f::Ones());
    Array8f y = (clamped.log() * Array8f::Constant(exponent)).exp();
    return (x <= Array8f::Zero()).select(Array8f::Zero(),
           (x >= Array8f::Ones()).select(Array8f::Ones(), y));
}

static inline MaterialSample kernel_sample_material(
    int mat_id,
    const float* pbr_data,
    const float* phong_data,
    const float* enamel_data)
{
    MaterialSample m;
    m.albedo           = pbr_data   ? pbr_albedo    (pbr_data,   mat_id) : Vector3f(0.5f,0.5f,0.5f);
    m.roughness        = pbr_data   ? pbr_roughness (pbr_data,   mat_id) : 0.5f;
    m.metallic         = pbr_data   ? pbr_metallic  (pbr_data,   mat_id) : 0.0f;
    m.ior              = pbr_data   ? pbr_ior       (pbr_data,   mat_id) : 1.5f;
    m.opacity          = pbr_data   ? pbr_opacity   (pbr_data,   mat_id) : 1.0f;
    m.emission         = pbr_data   ? pbr_emission  (pbr_data,   mat_id) : Vector3f(0.0f,0.0f,0.0f);

    m.ambient          = phong_data ? ph_ambient    (phong_data, mat_id) : 0.18f;
    m.spec_strength    = phong_data ? ph_specstr    (phong_data, mat_id) : 0.05f;
    m.shininess        = phong_data ? ph_shininess  (phong_data, mat_id) : 32.0f;
    m.inner_color      = phong_data ? ph_inner      (phong_data, mat_id) : m.albedo * 0.55f;

    m.enamel_thickness = enamel_data ? en_thick(enamel_data, mat_id) : 0.0f;
    m.enamel_ior       = enamel_data ? en_ior  (enamel_data, mat_id) : 1.52f;
    m.enamel_color     = enamel_data ? en_color(enamel_data, mat_id) : Vector3f(1.0f,1.0f,1.0f);
    return m;
}

static inline GeomTerms kernel_geometry_terms(
    const Vector3f& pos_v,
    const Vector3f& norm_v,
    bool front_facing,
    const MaterialSample& m,
    const SceneParams& /*sc*/)
{
    GeomTerms g;
    g.N = front_facing ? norm_v : (-norm_v);
    {
        float n2 = g.N.squaredNorm();
        if (n2 > 1e-12f) g.N *= 1.0f / std::sqrt(n2);
    }
    {
        float v2 = pos_v.squaredNorm();
        g.V = (v2 > 1e-12f) ? Vector3f(-pos_v * (1.0f / std::sqrt(v2)))
                            : Vector3f::Zero();
    }
    g.NdotV = std::max(0.0f, g.N.dot(g.V));
    g.base  = front_facing ? m.albedo : m.inner_color;
    return g;
}

static inline FresnelTerms kernel_fresnel_terms(
    const MaterialSample& m,
    float NdotV)
{
    FresnelTerms f;
    float ior_f0 = (m.ior - 1.0f) / (m.ior + 1.0f);
    ior_f0 *= ior_f0;
    f.F0 = (1.0f - m.metallic) * Vector3f(ior_f0, ior_f0, ior_f0)
         +  m.metallic        * m.albedo;
    float x  = 1.0f - NdotV;
    float x2 = x * x;
    float x5 = x2 * x2 * x;  // pow(x, 5) without log/exp
    f.F  = f.F0 + (Vector3f::Ones() - f.F0) * x5;
    return f;
}

/* Direct illumination from real emitters only.
 *
 * For each emitter-derived light: add Lambertian diffuse + Phong specular,
 * both tinted by the emitter's true colour.  After the loop add the
 * surface's own emission (self-light).  Nothing else.  No ambient.  No
 * proxy bounce.  No hardcoded sun.  No metallic albedo substitution.
 *
 * If n_lights == 0 the surface contributes only its own emission.  A
 * non-emissive surface in a dark scene renders black — correct physics.
 */
static inline Vector3f kernel_shade_lights(
    const MaterialSample& m,
    const Vector3f& pos_v,        // fragment position in view space
    const GeomTerms& g,
    const FresnelTerms& f,
    const SceneParams& sc,
    bool enable_specular,
    Vector3f* spec_color_out,
    float* spec_strength_out)
{
    Vector3f col = Vector3f::Zero();
    Vector3f spec_color_acc = Vector3f::Zero();
    float    spec_strength_acc = 0.0f;

    for (int i = 0; i < sc.n_lights; ++i) {
        // Per-fragment L from positional light.  This is what makes the
        // illumination track the emitter mesh in space — directional-only
        // lights collapse the geometry to a single point at the camera and
        // produce axis-flipped shading on receivers far from the origin.
        Vector3f Lvec = sc.light_positions[i] - pos_v;
        float    Lr2  = std::max(1e-8f, Lvec.squaredNorm());
        float    inv_Lr = 1.0f / std::sqrt(Lr2);
        Vector3f L    = Lvec * inv_Lr;
        // Inverse-square falloff (intensity already carries emitter area).
        Vector3f Lcol = sc.light_colors[i] * (sc.light_intensities[i] / Lr2);
        Vector3f LV = L + g.V;
        float lv2 = LV.squaredNorm();
        Vector3f H = (lv2 > 1e-12f) ? Vector3f(LV * (1.0f / std::sqrt(lv2))) : g.V;
        float NdotL = std::max(0.0f, g.N.dot(L));
        float NdotH = std::max(0.0f, g.N.dot(H));
        float spec  = pow_unit_fast_math(NdotH, std::max(1.0f, m.shininess));

        // Lambertian diffuse: surface albedo * incoming radiance * cos(theta)
        col += g.base.cwiseProduct(Lcol) * NdotL;

        // Phong specular: emitter colour modulated by Fresnel reflectance.
        if (enable_specular) {
            col += Lcol.cwiseProduct(f.F) * (m.spec_strength * spec);
        }

        spec_color_acc    += Lcol * spec;
        spec_strength_acc += spec;
    }

    if (spec_color_out)    *spec_color_out    = spec_color_acc;
    if (spec_strength_out) *spec_strength_out = spec_strength_acc;

    return col;
}

static inline Vector3f kernel_emission_uv(
    const MaterialSample& m,
    const Vector4f& e_uv,
    float emit_gain,
    const Vector3f& spec_col,
    float spec,
    bool enable_emission_direct,
    float direct_lobe_power,
    int   profile_id,
    float NdotV)
{
    Vector3f emit_dim = m.emission * (e_uv.w() * emit_gain);
    Vector3f emit_dir = Vector3f::Zero();
    if (profile_id == 1) {
        // Emissive lobe: forward-emission cone shaped by NdotV^lobe_power
        float lobe_p = std::max(1.0f, direct_lobe_power);
        float beam   = pow_unit_fast_math(std::max(0.0f, NdotV), lobe_p);
        emit_dir = emit_dim * (e_uv.x() * beam);
    } else if (enable_emission_direct && spec > 0.0f) {
        Vector3f spec_dir = spec_col * (1.0f / spec);
        emit_dir = emit_dim.cwiseProduct(spec_dir) * e_uv.x();
    }
    Vector3f emit_dif = emit_dim * e_uv.y();
    Vector3f emit_mix = emit_dir + emit_dif;
    float lum = emit_mix.dot(Vector3f(0.2126f, 0.7152f, 0.0722f));
    float sat = 0.5f + e_uv.z();
    return Vector3f::Constant(lum) * (1.0f - sat) + emit_mix * sat;
}

static inline Vector3f kernel_enamel_coat(
    const Vector3f& in_col,
    const MaterialSample& m,
    const GeomTerms& g,
    const Vector3f& spec_col,
    float spec)
{
    if (m.enamel_thickness <= 0.0f) {
        return in_col;
    }
    float opd_nm = 2.0f * m.enamel_ior * m.enamel_thickness * g.NdotV;
    Vector3f fringe = thin_film_fringe(opd_nm);
    float fringe_w  = std::min(0.45f, m.enamel_thickness / 800.0f);
    Vector3f tinted = in_col.cwiseProduct(m.enamel_color).cwiseProduct(
        Vector3f::Ones() + fringe * fringe_w);

    float eF0 = (m.enamel_ior - 1.0f) / (m.enamel_ior + 1.0f);
    eF0 *= eF0;
    float eF = schlick(eF0, g.NdotV);
    tinted += Vector3f(spec_col) * (eF * 0.28f * spec);
    return tinted * 0.35f + in_col * 0.65f;
}

/**
 * Evaluate Phong BRDF for one fragment.
 *
 * All vectors in VIEW space (L, V, N unit vectors).
 * Returns linear RGB (no gamma).  Alpha is returned via *alpha_out.
 */
static Vector3f shade(
    const Vector3f& pos_v,
    const Vector3f& norm_v,
    bool            front_facing,
    int             mat_id,
    const float*    pbr_data,
    const float*    phong_data,
    const float*    enamel_data,
    const SceneParams& sc,
    bool            enable_specular,
    bool            enable_emission_direct,
    const Vector4f& emit_uv,
    const Vector4f& color_uv,
    float           color_blend,
    float           emit_gain,
    float           direct_lobe_power,
    int             profile_id,
    const DepthSample& depth_s,
    float           translucence_gain,
    float           bulb_radius_mm,
    float*          alpha_out)
{
    MaterialSample m = kernel_sample_material(mat_id, pbr_data, phong_data, enamel_data);
    *alpha_out = m.opacity;
    float color_a = std::max(0.0f, std::min(1.0f, color_uv.w() * color_blend));
    m.albedo = m.albedo * (1.0f - color_a) + color_uv.head<3>() * color_a;

    // Profile 3: frosted scatter — pre-reduce specular sharpness in frosted regions
    if (profile_id == 3 && depth_s.scatter_mask > 0.001f) {
        m.shininess    *= std::max(0.0f, 1.0f - depth_s.scatter_mask * 0.88f);
        m.spec_strength *= std::max(0.0f, 1.0f - depth_s.scatter_mask * 0.70f);
    }

    GeomTerms g = kernel_geometry_terms(pos_v, norm_v, front_facing, m, sc);
    FresnelTerms f = kernel_fresnel_terms(m, g.NdotV);

    Vector3f spec_col(0.0f, 0.0f, 0.0f);
    float spec = 0.0f;
    Vector3f col = kernel_shade_lights(m, pos_v, g, f, sc, enable_specular, &spec_col, &spec);

    // Translucence: profile 2 (SSS) uses 4× stronger coupling
    float t_scale = (profile_id == 2) ? 0.08f : 0.02f;
    float translucence = std::max(0.0f, depth_s.thickness_mm) * translucence_gain
                       * depth_s.translucence_mask;
    col += g.base * (translucence * t_scale);

    // Bulb radius: Lorentzian falloff of emission by surface depth.
    // R² / (R² + d²) → 1 when depth≈0, 0.5 at depth=R, smoothly corners-dark.
    float bulb_factor = 1.0f;
    if (bulb_radius_mm > 0.0f) {
        float d = std::max(0.0f, depth_s.depth_mm);
        bulb_factor = (bulb_radius_mm * bulb_radius_mm)
                    / (bulb_radius_mm * bulb_radius_mm + d * d);
    }

    // Emission UV with profile-dependent lobe
    Vector3f emit_rgb = kernel_emission_uv(m, emit_uv, emit_gain, spec_col, spec,
                                           enable_emission_direct,
                                           direct_lobe_power, profile_id, g.NdotV)
                      * bulb_factor;
    col += emit_rgb;

    // Profile 3: frosted scatter halo — diffuse glow proportional to emission luminance
    if (profile_id == 3 && depth_s.scatter_mask > 0.001f) {
        float emit_lum = emit_rgb.dot(Vector3f(0.2126f, 0.7152f, 0.0722f));
        col += g.base * (depth_s.scatter_mask * emit_lum * 0.40f
                       + depth_s.scatter_mask * 0.025f);
        col = col.cwiseMax(Vector3f::Zero());
    }

    col = kernel_enamel_coat(col, m, g, spec_col, spec);
    return col;
}

struct PacketShadeInput {
    Array8f pos_x, pos_y, pos_z;
    Array8f nrm_x, nrm_y, nrm_z;
    Mask8   front_facing;
    Array8f emit_r, emit_g, emit_b, emit_a;
    Array8f color_r, color_g, color_b, color_a;
    Array8f depth_mm, thickness_mm, translucence_mask, scatter_mask;
};

struct PacketShadeOutput {
    Array8f col_r, col_g, col_b;
    Array8f alpha;
};

[[maybe_unused]] static inline PacketShadeOutput shade_packet8(
    const PacketShadeInput& in,
    int             mat_id,
    const float*    pbr_data,
    const float*    phong_data,
    const float*    enamel_data,
    const SceneParams& sc,
    bool            enable_specular,
    bool            enable_emission_direct,
    float           color_blend,
    float           emit_gain,
    float           direct_lobe_power,
    int             profile_id,
    float           translucence_gain,
    float           bulb_radius_mm)
{
    PacketShadeOutput out;

    const MaterialSample m0 = kernel_sample_material(mat_id, pbr_data, phong_data, enamel_data);
    const float color_a_min = 0.0f;
    const float color_a_max = 1.0f;

    const Array8f color_a = (in.color_a * Array8f::Constant(color_blend)).max(Array8f::Constant(color_a_min)).min(Array8f::Constant(color_a_max));
    const Array8f albedo_x = Array8f::Constant(m0.albedo.x()) * (Array8f::Ones() - color_a) + in.color_r * color_a;
    const Array8f albedo_y = Array8f::Constant(m0.albedo.y()) * (Array8f::Ones() - color_a) + in.color_g * color_a;
    const Array8f albedo_z = Array8f::Constant(m0.albedo.z()) * (Array8f::Ones() - color_a) + in.color_b * color_a;

    float shininess = m0.shininess;
    float spec_strength = m0.spec_strength;
    if (profile_id == 3) {
        const float scatter_max = in.scatter_mask.maxCoeff();
        if (scatter_max > 0.001f) {
            shininess *= std::max(0.0f, 1.0f - scatter_max * 0.88f);
            spec_strength *= std::max(0.0f, 1.0f - scatter_max * 0.70f);
        }
    }

    Array8f Nx = in.front_facing.select(in.nrm_x, -in.nrm_x);
    Array8f Ny = in.front_facing.select(in.nrm_y, -in.nrm_y);
    Array8f Nz = in.front_facing.select(in.nrm_z, -in.nrm_z);
    const Array8f n2 = Nx * Nx + Ny * Ny + Nz * Nz;
    const Mask8 n_ok = (n2 > Array8f::Constant(1e-12f));
    const Array8f inv_n = n_ok.select(Array8f::Ones() / n2.sqrt(), Array8f::Zero());
    Nx *= inv_n;
    Ny *= inv_n;
    Nz *= inv_n;

    const Array8f v2 = in.pos_x * in.pos_x + in.pos_y * in.pos_y + in.pos_z * in.pos_z;
    const Mask8 v_ok = (v2 > Array8f::Constant(1e-12f));
    const Array8f inv_v = v_ok.select(Array8f::Ones() / v2.sqrt(), Array8f::Zero());
    const Array8f Vx = -in.pos_x * inv_v;
    const Array8f Vy = -in.pos_y * inv_v;
    const Array8f Vz = -in.pos_z * inv_v;

    const Array8f NdotV = (Nx * Vx + Ny * Vy + Nz * Vz).max(Array8f::Zero());
    const Array8f base_x = in.front_facing.select(albedo_x, Array8f::Constant(m0.inner_color.x()));
    const Array8f base_y = in.front_facing.select(albedo_y, Array8f::Constant(m0.inner_color.y()));
    const Array8f base_z = in.front_facing.select(albedo_z, Array8f::Constant(m0.inner_color.z()));

    float ior_f0 = (m0.ior - 1.0f) / (m0.ior + 1.0f);
    ior_f0 *= ior_f0;
    const float F0x = (1.0f - m0.metallic) * ior_f0 + m0.metallic * m0.albedo.x();
    const float F0y = (1.0f - m0.metallic) * ior_f0 + m0.metallic * m0.albedo.y();
    const float F0z = (1.0f - m0.metallic) * ior_f0 + m0.metallic * m0.albedo.z();
    const Array8f x = Array8f::Ones() - NdotV;
    const Array8f x2 = x * x;
    const Array8f x5 = x2 * x2 * x;
    const Array8f Fx = Array8f::Constant(F0x) + (Array8f::Ones() - Array8f::Constant(F0x)) * x5;
    const Array8f Fy = Array8f::Constant(F0y) + (Array8f::Ones() - Array8f::Constant(F0y)) * x5;
    const Array8f Fz = Array8f::Constant(F0z) + (Array8f::Ones() - Array8f::Constant(F0z)) * x5;

    Array8f col_x = Array8f::Zero();
    Array8f col_y = Array8f::Zero();
    Array8f col_z = Array8f::Zero();
    Array8f spec_col_x = Array8f::Zero();
    Array8f spec_col_y = Array8f::Zero();
    Array8f spec_col_z = Array8f::Zero();
    Array8f spec_acc = Array8f::Zero();

    const float shininess_clamped = std::max(1.0f, shininess);
    for (int i = 0; i < sc.n_lights; ++i) {
        const Array8f Lvec_x = Array8f::Constant(sc.light_positions[i].x()) - in.pos_x;
        const Array8f Lvec_y = Array8f::Constant(sc.light_positions[i].y()) - in.pos_y;
        const Array8f Lvec_z = Array8f::Constant(sc.light_positions[i].z()) - in.pos_z;
        const Array8f Lr2 = (Lvec_x * Lvec_x + Lvec_y * Lvec_y + Lvec_z * Lvec_z).max(Array8f::Constant(1e-8f));
        const Array8f inv_Lr = Array8f::Ones() / Lr2.sqrt();
        const Array8f Lx = Lvec_x * inv_Lr;
        const Array8f Ly = Lvec_y * inv_Lr;
        const Array8f Lz = Lvec_z * inv_Lr;

        const Array8f Lcol_scale = Array8f::Constant(sc.light_intensities[i]) / Lr2;
        const Array8f Lcol_x = Array8f::Constant(sc.light_colors[i].x()) * Lcol_scale;
        const Array8f Lcol_y = Array8f::Constant(sc.light_colors[i].y()) * Lcol_scale;
        const Array8f Lcol_z = Array8f::Constant(sc.light_colors[i].z()) * Lcol_scale;

        const Array8f LVx = Lx + Vx;
        const Array8f LVy = Ly + Vy;
        const Array8f LVz = Lz + Vz;
        const Array8f lv2 = LVx * LVx + LVy * LVy + LVz * LVz;
        const Mask8 h_ok = (lv2 > Array8f::Constant(1e-12f));
        const Array8f inv_h = h_ok.select(Array8f::Ones() / lv2.sqrt(), Array8f::Zero());
        const Array8f Hx = h_ok.select(LVx * inv_h, Vx);
        const Array8f Hy = h_ok.select(LVy * inv_h, Vy);
        const Array8f Hz = h_ok.select(LVz * inv_h, Vz);

        const Array8f NdotL = (Nx * Lx + Ny * Ly + Nz * Lz).max(Array8f::Zero());
        const Array8f NdotH = (Nx * Hx + Ny * Hy + Nz * Hz).max(Array8f::Zero());
        const Array8f spec = pow_unit_fast_math(NdotH, shininess_clamped);

        col_x += base_x * Lcol_x * NdotL;
        col_y += base_y * Lcol_y * NdotL;
        col_z += base_z * Lcol_z * NdotL;

        if (enable_specular) {
            const Array8f s = Array8f::Constant(spec_strength) * spec;
            col_x += Lcol_x * Fx * s;
            col_y += Lcol_y * Fy * s;
            col_z += Lcol_z * Fz * s;
        }

        spec_col_x += Lcol_x * spec;
        spec_col_y += Lcol_y * spec;
        spec_col_z += Lcol_z * spec;
        spec_acc += spec;
    }

    const float t_scale = (profile_id == 2) ? 0.08f : 0.02f;
    const Array8f translucence = in.thickness_mm.max(Array8f::Zero())
                               * Array8f::Constant(translucence_gain)
                               * in.translucence_mask;
    col_x += base_x * (translucence * Array8f::Constant(t_scale));
    col_y += base_y * (translucence * Array8f::Constant(t_scale));
    col_z += base_z * (translucence * Array8f::Constant(t_scale));

    Array8f bulb_factor = Array8f::Ones();
    if (bulb_radius_mm > 0.0f) {
        const Array8f d = in.depth_mm.max(Array8f::Zero());
        const float r2 = bulb_radius_mm * bulb_radius_mm;
        bulb_factor = Array8f::Constant(r2) / (Array8f::Constant(r2) + d * d);
    }

    const Array8f emit_dim_scale = in.emit_a * Array8f::Constant(emit_gain);
    const Array8f emit_dim_x = Array8f::Constant(m0.emission.x()) * emit_dim_scale;
    const Array8f emit_dim_y = Array8f::Constant(m0.emission.y()) * emit_dim_scale;
    const Array8f emit_dim_z = Array8f::Constant(m0.emission.z()) * emit_dim_scale;

    Array8f emit_dir_x = Array8f::Zero();
    Array8f emit_dir_y = Array8f::Zero();
    Array8f emit_dir_z = Array8f::Zero();
    if (profile_id == 1) {
        const float lobe_p = std::max(1.0f, direct_lobe_power);
        const Array8f beam = pow_unit_fast_math(NdotV.max(Array8f::Zero()), lobe_p);
        emit_dir_x = emit_dim_x * (in.emit_r * beam);
        emit_dir_y = emit_dim_y * (in.emit_r * beam);
        emit_dir_z = emit_dim_z * (in.emit_r * beam);
    } else if (enable_emission_direct) {
        const Mask8 has_spec = (spec_acc > Array8f::Zero());
        const Array8f inv_spec = has_spec.select(Array8f::Ones() / spec_acc, Array8f::Zero());
        const Array8f spec_dir_x = spec_col_x * inv_spec;
        const Array8f spec_dir_y = spec_col_y * inv_spec;
        const Array8f spec_dir_z = spec_col_z * inv_spec;
        emit_dir_x = emit_dim_x * spec_dir_x * in.emit_r;
        emit_dir_y = emit_dim_y * spec_dir_y * in.emit_r;
        emit_dir_z = emit_dim_z * spec_dir_z * in.emit_r;
    }

    const Array8f emit_dif_x = emit_dim_x * in.emit_g;
    const Array8f emit_dif_y = emit_dim_y * in.emit_g;
    const Array8f emit_dif_z = emit_dim_z * in.emit_g;
    const Array8f emit_mix_x = emit_dir_x + emit_dif_x;
    const Array8f emit_mix_y = emit_dir_y + emit_dif_y;
    const Array8f emit_mix_z = emit_dir_z + emit_dif_z;
    const Array8f emit_lum = emit_mix_x * Array8f::Constant(0.2126f)
                           + emit_mix_y * Array8f::Constant(0.7152f)
                           + emit_mix_z * Array8f::Constant(0.0722f);
    const Array8f sat = Array8f::Constant(0.5f) + in.emit_b;
    Array8f emit_out_x = emit_lum * (Array8f::Ones() - sat) + emit_mix_x * sat;
    Array8f emit_out_y = emit_lum * (Array8f::Ones() - sat) + emit_mix_y * sat;
    Array8f emit_out_z = emit_lum * (Array8f::Ones() - sat) + emit_mix_z * sat;
    emit_out_x *= bulb_factor;
    emit_out_y *= bulb_factor;
    emit_out_z *= bulb_factor;
    col_x += emit_out_x;
    col_y += emit_out_y;
    col_z += emit_out_z;

    if (profile_id == 3) {
        const Mask8 frost = (in.scatter_mask > Array8f::Constant(0.001f));
        const Array8f add = in.scatter_mask * emit_lum * Array8f::Constant(0.40f)
                          + in.scatter_mask * Array8f::Constant(0.025f);
        const Array8f next_x = (col_x + base_x * add).max(Array8f::Zero());
        const Array8f next_y = (col_y + base_y * add).max(Array8f::Zero());
        const Array8f next_z = (col_z + base_z * add).max(Array8f::Zero());
        col_x = frost.select(next_x, col_x);
        col_y = frost.select(next_y, col_y);
        col_z = frost.select(next_z, col_z);
    }

    if (m0.enamel_thickness > 0.0f) {
        const Array8f opd_nm = Array8f::Constant(2.0f * m0.enamel_ior * m0.enamel_thickness) * NdotV;
        constexpr float TWO_PI = 6.28318530718f;
        const Array8f phase = opd_nm * Array8f::Constant(TWO_PI / 550.0f);
        const Array8f fringe_r = Array8f::Constant(0.5f) + Array8f::Constant(0.5f) * phase.cos();
        const Array8f fringe_g = Array8f::Constant(0.5f) + Array8f::Constant(0.5f) * (phase - Array8f::Constant(2.09439510239f)).cos();
        const Array8f fringe_b = Array8f::Constant(0.5f) + Array8f::Constant(0.5f) * (phase - Array8f::Constant(4.18879020479f)).cos();
        const float fringe_w = std::min(0.45f, m0.enamel_thickness / 800.0f);

        Array8f tinted_x = col_x * Array8f::Constant(m0.enamel_color.x())
                         * (Array8f::Ones() + fringe_r * Array8f::Constant(fringe_w));
        Array8f tinted_y = col_y * Array8f::Constant(m0.enamel_color.y())
                         * (Array8f::Ones() + fringe_g * Array8f::Constant(fringe_w));
        Array8f tinted_z = col_z * Array8f::Constant(m0.enamel_color.z())
                         * (Array8f::Ones() + fringe_b * Array8f::Constant(fringe_w));

        float eF0 = (m0.enamel_ior - 1.0f) / (m0.enamel_ior + 1.0f);
        eF0 *= eF0;
        const Array8f t = (Array8f::Ones() - NdotV.max(Array8f::Zero()));
        const Array8f t2 = t * t;
        const Array8f eF = Array8f::Constant(eF0)
                         + (Array8f::Ones() - Array8f::Constant(eF0)) * (t2 * t2 * t);
        const Array8f enamel_spec = eF * Array8f::Constant(0.28f) * spec_acc;
        tinted_x += spec_col_x * enamel_spec;
        tinted_y += spec_col_y * enamel_spec;
        tinted_z += spec_col_z * enamel_spec;

        col_x = tinted_x * Array8f::Constant(0.35f) + col_x * Array8f::Constant(0.65f);
        col_y = tinted_y * Array8f::Constant(0.35f) + col_y * Array8f::Constant(0.65f);
        col_z = tinted_z * Array8f::Constant(0.35f) + col_z * Array8f::Constant(0.65f);
    }

    out.col_r = col_x;
    out.col_g = col_y;
    out.col_b = col_z;
    out.alpha = Array8f::Constant(m0.opacity);
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Projected triangle
 * ═══════════════════════════════════════════════════════════════════════════ */

struct ProjVertex {
    float sx, sy;   // screen-space x, y (pixel coordinates)
    float inv_w;    // 1 / clip_w  (for perspective-correct interp)
    float ndc_z;    // clip-space z / w, OpenGL convention [-1, +1]
    bool  valid;    // finite and in front of the eye plane
    Vector3f pos_v; // view-space position
    Vector3f nrm_v; // view-space normal
    Vector2f uv;    // texture-stack UV, optional
};

struct ScreenTri {
    ProjVertex v[3];
    int   mat_id;
    float bbox_x0, bbox_y0, bbox_x1, bbox_y1;  // screen bbox
    bool  valid;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * State
 * ═══════════════════════════════════════════════════════════════════════════ */

struct BaseRasterizerState {
    int width, height, tile_size;
    int tiles_x, tiles_y, n_tiles;

    // Framebuffer: linear float RGBA
    std::vector<float> cbuf;  // width * height * 4
    std::vector<float> zbuf;  // width * height (NDC depth, smaller = closer)

    // Material chunk pointers (owned by the caller — no copy)
    const float* pbr_data    = nullptr;
    const float* phong_data  = nullptr;
    const float* enamel_data = nullptr;
    const float* texstack_data = nullptr;
    int n_materials           = 0;
    int n_texstack_materials   = 0;
    std::vector<uint8_t> emit_uv_data;
    int emit_uv_w = 0;
    int emit_uv_h = 0;
    int emit_uv_layers = 0;
    std::vector<uint8_t> color_uv_data;
    int color_uv_w = 0;
    int color_uv_h = 0;
    int color_uv_layers = 0;
    std::vector<uint8_t> depth_uv_data;
    int depth_uv_w = 0;
    int depth_uv_h = 0;
    int depth_uv_layers = 0;
    std::vector<uint8_t> remit_uv_data;
    int remit_uv_w = 0;
    int remit_uv_h = 0;
    int remit_uv_layers = 0;

    SceneParams scene;
    // Runtime cap on the number of cluster-lights emitted per frame.
    // Defaults to SceneParams::MAX_LIGHTS; clamped to [1, MAX_LIGHTS].
    int max_lights = SceneParams::MAX_LIGHTS;
    bool enable_specular = true;
    bool enable_emission_direct = false;

    // ── Object-group cache for emissive cluster lights ────────────────
    //
    // One entry per host-declared group.  Centroid is stored in the
    // view-space frame established by `last_mv`; on a BR_DIRTY_MV-only
    // update we transport the centroid via `mv_new * last_mv.inverse()`,
    // skipping any per-triangle work.  On BR_DIRTY_GEOM we recompute
    // centroid + area_sum from the group's slice of the current
    // verts_view buffer.  Clean groups consume zero work per frame.
    struct GroupCache {
        int      mat_id     = -1;
        int      tri_offset = 0;
        int      tri_count  = 0;
        Vector3f centroid_v = Vector3f::Zero();
        float    area_sum   = 0.0f;
        Matrix4f last_mv    = Matrix4f::Identity();
        bool     have_geom  = false;   // false until first DIRTY_GEOM resolved
    };
    std::unordered_map<int, GroupCache> group_cache;

    // Per-frame group descriptor list (caller-owned copy).  Empty list ⇒
    // no lights at all (legacy "draw whatever, no illumination" path).
    std::vector<BRGroup> groups;

    // Per-tile triangle lists (rebuilt each render call)
    std::vector<std::vector<int>> tile_tris;

    // Projected triangles for the current draw
    std::vector<ScreenTri> screen_tris;
};

static inline Vector4f sample_rgba8_array(
    const std::vector<uint8_t>& data,
    int width,
    int height,
    int layers,
    int layer,
    const Vector2f& uv,
    const Vector4f& fallback)
{
    if (data.empty() || width <= 0 || height <= 0 || layers <= 0 ||
        layer < 0 || layer >= layers) {
        return fallback;
    }

    int tx = static_cast<int>(uv.x() * static_cast<float>(width));
    int ty = static_cast<int>(uv.y() * static_cast<float>(height));

    // U wraps. Texture widths are currently powers of two, so this is the hot
    // path; keep modulo fallback for any future odd-sized texture.
    if ((width & (width - 1)) == 0) {
        tx &= (width - 1);
    } else {
        tx %= width;
        if (tx < 0) tx += width;
    }
    ty = std::max(0, std::min(height - 1, ty));

    const size_t layer_off = static_cast<size_t>(layer)
                           * static_cast<size_t>(height) * static_cast<size_t>(width) * 4u;
    const uint8_t* p = data.data() + layer_off
                     + (static_cast<size_t>(ty) * static_cast<size_t>(width)
                     +  static_cast<size_t>(tx)) * 4u;
    return Vector4f(p[0], p[1], p[2], p[3]) * 0.00392156862745098f;
}

static inline Vector4f sample_emit_uv(
    const BaseRasterizerState* st,
    int mat_id,
    const Vector2f& uv)
{
    if (!st || !st->texstack_data || mat_id < 0 || mat_id >= st->n_texstack_materials) {
        return Vector4f(0.0f, 1.0f, 0.5f, 1.0f);
    }
    int layer = (int)std::lround(tex_emit_layer(st->texstack_data, mat_id));
    return sample_rgba8_array(st->emit_uv_data, st->emit_uv_w, st->emit_uv_h,
                              st->emit_uv_layers, layer, uv,
                              Vector4f(0.0f, 1.0f, 0.5f, 1.0f));
}

static inline Vector4f sample_color_uv(
    const BaseRasterizerState* st,
    int mat_id,
    const Vector2f& uv)
{
    if (!st || !st->texstack_data || mat_id < 0 || mat_id >= st->n_texstack_materials) {
        return Vector4f(1.0f, 1.0f, 1.0f, 0.0f);
    }
    int layer = (int)std::lround(tex_color_layer(st->texstack_data, mat_id));
    return sample_rgba8_array(st->color_uv_data, st->color_uv_w, st->color_uv_h,
                              st->color_uv_layers, layer, uv,
                              Vector4f(1.0f, 1.0f, 1.0f, 0.0f));
}

static inline DepthSample sample_depth_uv4(
    const BaseRasterizerState* st,
    int mat_id,
    const Vector2f& uv)
{
    DepthSample ds;
    if (!st || !st->texstack_data || st->depth_uv_data.empty() ||
        mat_id < 0 || mat_id >= st->n_texstack_materials ||
        st->depth_uv_w <= 0 || st->depth_uv_h <= 0 || st->depth_uv_layers <= 0) {
        return ds;
    }
    int layer = (int)std::lround(tex_depth_layer(st->texstack_data, mat_id));
    if (layer < 0 || layer >= st->depth_uv_layers) {
        return ds;
    }
    Vector4f d = sample_rgba8_array(st->depth_uv_data, st->depth_uv_w, st->depth_uv_h,
                                    st->depth_uv_layers, layer, uv, Vector4f::Zero());
    ds.depth_mm          = d.x() * tex_depth_scale_mm(st->texstack_data, mat_id)
                         + tex_depth_bias_mm(st->texstack_data, mat_id);
    ds.thickness_mm      = d.y() * tex_thickness_scale_mm(st->texstack_data, mat_id)
                         + tex_thickness_bias_mm(st->texstack_data, mat_id);
    ds.translucence_mask = d.z();  // depth_uv.B raw [0,1]
    ds.scatter_mask      = d.w();  // depth_uv.A raw [0,1]
    return ds;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

BaseRasterizerState* br_create(int width, int height, int tile_size) {
    auto* st = new BaseRasterizerState();
    st->width     = width;
    st->height    = height;
    st->tile_size = tile_size > 0 ? tile_size : 16;
    st->tiles_x   = (width  + st->tile_size - 1) / st->tile_size;
    st->tiles_y   = (height + st->tile_size - 1) / st->tile_size;
    st->n_tiles   = st->tiles_x * st->tiles_y;
    st->cbuf.assign(static_cast<size_t>(width * height * 4), 0.0f);
    st->zbuf.assign(static_cast<size_t>(width * height),     1.0f);
    st->tile_tris.resize(static_cast<size_t>(st->n_tiles));
    return st;
}

void br_destroy(BaseRasterizerState* st) { delete st; }

void br_clear(BaseRasterizerState* st, float r, float g, float b, float a) {
    const int n = st->width * st->height;
    // Vectorized fill: map the framebuffer as a 4 x N matrix and broadcast
    // the clear color across all columns in one expression.
    Eigen::Map<Eigen::Matrix<float, 4, Eigen::Dynamic>> cmap(
        st->cbuf.data(), 4, n);
    cmap.colwise() = Vector4f(r, g, b, a);
    std::fill(st->zbuf.begin(), st->zbuf.end(), 1.0f);
}

void br_set_pbr_chunk(BaseRasterizerState* st, const float* data, int n_materials) {
    st->pbr_data    = data;
    st->n_materials = n_materials;
}
void br_set_phong_chunk(BaseRasterizerState* st, const float* data, int n_materials) {
    st->phong_data  = data;
    (void)n_materials;
}
void br_set_enamel_chunk(BaseRasterizerState* st, const float* data, int n_materials) {
    st->enamel_data = data;
    (void)n_materials;
}
void br_set_texture_stack_chunk(BaseRasterizerState* st, const float* data, int n_materials) {
    st->texstack_data = data;
    st->n_texstack_materials = n_materials;
}
void br_set_emit_uv_texture_array(BaseRasterizerState* st,
                                  const uint8_t* data,
                                  int width,
                                  int height,
                                  int layers) {
    if (!st) return;
    if (!data || width <= 0 || height <= 0 || layers <= 0) {
        st->emit_uv_data.clear();
        st->emit_uv_w = st->emit_uv_h = st->emit_uv_layers = 0;
        return;
    }
    size_t n = (size_t)width * (size_t)height * (size_t)layers * 4u;
    st->emit_uv_data.assign(data, data + n);
    st->emit_uv_w = width;
    st->emit_uv_h = height;
    st->emit_uv_layers = layers;
}
void br_set_color_uv_texture_array(BaseRasterizerState* st,
                                   const uint8_t* data,
                                   int width,
                                   int height,
                                   int layers) {
    if (!st) return;
    if (!data || width <= 0 || height <= 0 || layers <= 0) {
        st->color_uv_data.clear();
        st->color_uv_w = st->color_uv_h = st->color_uv_layers = 0;
        return;
    }
    size_t n = (size_t)width * (size_t)height * (size_t)layers * 4u;
    st->color_uv_data.assign(data, data + n);
    st->color_uv_w = width;
    st->color_uv_h = height;
    st->color_uv_layers = layers;
}
void br_set_depth_uv_texture_array(BaseRasterizerState* st,
                                   const uint8_t* data,
                                   int width,
                                   int height,
                                   int layers) {
    if (!st) return;
    if (!data || width <= 0 || height <= 0 || layers <= 0) {
        st->depth_uv_data.clear();
        st->depth_uv_w = st->depth_uv_h = st->depth_uv_layers = 0;
        return;
    }
    size_t n = (size_t)width * (size_t)height * (size_t)layers * 4u;
    st->depth_uv_data.assign(data, data + n);
    st->depth_uv_w = width;
    st->depth_uv_h = height;
    st->depth_uv_layers = layers;
}
void br_set_remit_uv_texture_array(BaseRasterizerState* st,
                                   const uint8_t* data,
                                   int width,
                                   int height,
                                   int layers) {
    if (!st) return;
    if (!data || width <= 0 || height <= 0 || layers <= 0) {
        st->remit_uv_data.clear();
        st->remit_uv_w = st->remit_uv_h = st->remit_uv_layers = 0;
        return;
    }
    size_t n = (size_t)width * (size_t)height * (size_t)layers * 4u;
    st->remit_uv_data.assign(data, data + n);
    st->remit_uv_w = width;
    st->remit_uv_h = height;
    st->remit_uv_layers = layers;
}

void br_set_scene(BaseRasterizerState* /*st*/,
                  const float* /*light_v*/,
                  const float* /*scene_rgb*/,
                  float        /*scene_indirect*/) {
    // No-op.  This entry point survived only as an ABI shim; it accepted
    // an artificial scene tint and indirect-fill knob that this engine
    // refuses on principle.  All illumination is now configured exclusively
    // via br_set_lights() with real emitter colours and directions.
}

void br_set_lights(BaseRasterizerState* /*st*/,
                   int          /*n_lights*/,
                   const float* /*dirs*/,
                   const float* /*colors*/,
                   const float* /*intens*/) {
    // No-op.  Light state is no longer carried in the rasterizer; every
    // frame, br_render() rebuilds the active light set from the EMISSIVE
    // MATERIALS attached to the rendered triangles.  Materials emit;
    // nothing else does.  This entry point survives only as an ABI shim
    // for ctypes/pybind callers that may still invoke it.
}

/* ── Object groups ──────────────────────────────────────────────────────── */
//
// The host's scene graph already has the partition we need: every triangle
// belongs to one object, every object has one material and one model-view
// transform.  We accept that partition verbatim and never re-discover it.
//
// A frame consists of (a) br_set_groups() with the full group list, then
// (b) br_render() with the matching triangle soup.  Cached per-group
// (centroid, area_sum) entries persist across frames.  A clean group does
// no work; a BR_DIRTY_MV-only group does one 4×4 transform of its cached
// centroid; only BR_DIRTY_GEOM walks triangles.
void br_set_groups(BaseRasterizerState* st,
                   int             n_groups,
                   const BRGroup*  groups)
{
    if (!st) return;
    st->groups.assign(groups, groups + std::max(0, n_groups));
}

void br_set_max_lights(BaseRasterizerState* st, int max_lights)
{
    if (!st) return;
    if (max_lights < 1) max_lights = 1;
    if (max_lights > SceneParams::MAX_LIGHTS) max_lights = SceneParams::MAX_LIGHTS;
    st->max_lights = max_lights;
}

void br_set_specular_enabled(BaseRasterizerState* st, int enabled)
{
    if (!st) return;
    st->enable_specular = enabled != 0;
}

void br_set_emission_direct_enabled(BaseRasterizerState* st, int enabled)
{
    if (!st) return;
    st->enable_emission_direct = enabled != 0;
}

/* ── Project helpers ─────────────────────────────────────────────────────── */

// Vertex projection is performed in bulk inside br_render() as a single
// proj(4×4) * positions(4 × 3N) GEMM — see the projection block there.

/* ── Render ──────────────────────────────────────────────────────────────── */

static void br_render_impl(BaseRasterizerState* st,
                           const float* verts_view,
                           const int*   mat_ids,
                           int          n_tris,
                           const float* mvp,
                           int          vertex_stride)
{
    if (!st || n_tris <= 0) return;
    const int VS = vertex_stride >= 8 ? 8 : 6;
    const int tri_stride = VS * 3;

    const float half_w = st->width  * 0.5f;
    const float half_h = st->height * 0.5f;
    const int   W      = st->width;
    const int   H      = st->height;
    const int   TS     = st->tile_size;
    const int   TX     = st->tiles_x;

    // Parse 4×4 column-major matrix
    Matrix4f proj;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            proj(r, c) = mvp[c*4 + r];

    // Project all vertices in one GEMM: clip(4 × 3N) = proj(4×4) * pos(4 × 3N).
    // Building the homogeneous position matrix in a contiguous buffer lets
    // Eigen's vectorized matmul amortize the projection cost across all
    // vertices instead of doing 3N separate 4×1 matvecs.
    const int n_verts = 3 * n_tris;
    Eigen::Matrix<float, 4, Eigen::Dynamic> positions(4, n_verts);
    for (int i = 0; i < n_tris; ++i) {
        const float* base = verts_view + i * tri_stride;
        positions(0, 3*i+0) = base[0];   positions(1, 3*i+0) = base[1];
        positions(2, 3*i+0) = base[2];   positions(3, 3*i+0) = 1.0f;
        positions(0, 3*i+1) = base[VS+0];   positions(1, 3*i+1) = base[VS+1];
        positions(2, 3*i+1) = base[VS+2];   positions(3, 3*i+1) = 1.0f;
        positions(0, 3*i+2) = base[2*VS+0]; positions(1, 3*i+2) = base[2*VS+1];
        positions(2, 3*i+2) = base[2*VS+2]; positions(3, 3*i+2) = 1.0f;
    }
    Eigen::Matrix<float, 4, Eigen::Dynamic> clip_all = proj * positions;

    st->screen_tris.resize(static_cast<size_t>(n_tris));
    for (int i = 0; i < n_tris; ++i) {
        ScreenTri& st_tri = st->screen_tris[i];
        const float* base = verts_view + i * tri_stride;
        for (int k = 0; k < 3; ++k) {
            ProjVertex& pv = st_tri.v[k];
            const float* vv = base + k * VS;
            pv.pos_v = Vector3f(vv[0], vv[1], vv[2]);
            pv.nrm_v = Vector3f(vv[3], vv[4], vv[5]);
            pv.uv = (VS >= 8) ? Vector2f(vv[6], vv[7]) : Vector2f(0.0f, 0.0f);
            pv.inv_w = 0.0f;
            pv.ndc_z = 1.0f;
            pv.valid = false;

            const float cx = clip_all(0, 3*i+k);
            const float cy = clip_all(1, 3*i+k);
            const float cz = clip_all(2, 3*i+k);
            const float cw = clip_all(3, 3*i+k);
            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(cz) || !std::isfinite(cw) || cw <= 1e-6f) {
                pv.sx = pv.sy = -1e9f;
                continue;
            }
            const float inv_w = 1.0f / cw;
            const float ndcx  = cx * inv_w;
            const float ndcy  = cy * inv_w;
            const float ndcz  = cz * inv_w;
            if (!std::isfinite(ndcx) || !std::isfinite(ndcy) || !std::isfinite(ndcz)) {
                pv.sx = pv.sy = -1e9f;
                continue;
            }
            pv.sx    = (ndcx + 1.0f) * half_w;
            pv.sy    = (1.0f - ndcy) * half_h;  // flip Y
            pv.inv_w = inv_w;
            pv.ndc_z = ndcz;
            pv.valid = true;
        }
        st_tri.mat_id = mat_ids[i];

        if (!st_tri.v[0].valid || !st_tri.v[1].valid || !st_tri.v[2].valid) {
            st_tri.valid = false;
            continue;
        }

        // Cheap homogeneous frustum reject after projection.  This rasterizer
        // does not clip triangles against the near/far planes yet, so reject
        // triangles that would require clipping; otherwise one vertex near
        // w=0 can stretch a primitive across the whole frame.
        float min_z = std::min({st_tri.v[0].ndc_z, st_tri.v[1].ndc_z, st_tri.v[2].ndc_z});
        float max_z = std::max({st_tri.v[0].ndc_z, st_tri.v[1].ndc_z, st_tri.v[2].ndc_z});
        if (min_z < -1.0f || max_z > 1.0f) {
            st_tri.valid = false;
            continue;
        }

        // Back-face cull in screen space
        float ex0 = st_tri.v[1].sx - st_tri.v[0].sx;
        float ey0 = st_tri.v[1].sy - st_tri.v[0].sy;
        float ex1 = st_tri.v[2].sx - st_tri.v[0].sx;
        float ey1 = st_tri.v[2].sy - st_tri.v[0].sy;
        float signed_area = ex0 * ey1 - ex1 * ey0;
        // Cull back faces (positive signed area in screen space = back)
        if (signed_area >= 0.0f) { st_tri.valid = false; continue; }
        st_tri.valid = true;

        float minx = std::min({st_tri.v[0].sx, st_tri.v[1].sx, st_tri.v[2].sx});
        float miny = std::min({st_tri.v[0].sy, st_tri.v[1].sy, st_tri.v[2].sy});
        float maxx = std::max({st_tri.v[0].sx, st_tri.v[1].sx, st_tri.v[2].sx});
        float maxy = std::max({st_tri.v[0].sy, st_tri.v[1].sy, st_tri.v[2].sy});
        st_tri.bbox_x0 = std::max(0.0f, std::floor(minx));
        st_tri.bbox_y0 = std::max(0.0f, std::floor(miny));
        st_tri.bbox_x1 = std::min((float)(W-1), std::ceil(maxx));
        st_tri.bbox_y1 = std::min((float)(H-1), std::ceil(maxy));
    }

    // Clear tile lists
    for (auto& tl : st->tile_tris) tl.clear();

    // Bin triangles into tiles
    for (int i = 0; i < n_tris; ++i) {
        const ScreenTri& tri = st->screen_tris[i];
        if (!tri.valid) continue;
        int tx0 = static_cast<int>(tri.bbox_x0) / TS;
        int ty0 = static_cast<int>(tri.bbox_y0) / TS;
        int tx1 = static_cast<int>(tri.bbox_x1) / TS;
        int ty1 = static_cast<int>(tri.bbox_y1) / TS;
        tx0 = std::max(0, tx0);
        ty0 = std::max(0, ty0);
        tx1 = std::min(st->tiles_x - 1, tx1);
        ty1 = std::min(st->tiles_y - 1, ty1);
        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx)
                st->tile_tris[ty * TX + tx].push_back(i);
    }

    // Parallel tile rasterization
    const int n_tiles = st->n_tiles;
    const auto& screen_tris = st->screen_tris;
    const auto& tile_tris   = st->tile_tris;
    float* cbuf = st->cbuf.data();
    float* zbuf = st->zbuf.data();
    const float* pbr_data    = st->pbr_data;
    const float* phong_data  = st->phong_data;
    const float* enamel_data = st->enamel_data;

    // ── Update emissive-group cache, then derive scene lights ──────────
    //
    // The host has declared the partition via br_set_groups().  We never
    // cluster by mat_id, never walk all triangles, never average across
    // materials.  For each declared group we touch only the work that its
    // dirty bitmask demands:
    //
    //   BR_DIRTY_GEOM → parallel reduction over the group's tri slice in
    //                   verts_view to compute (centroid_v, area_sum) using
    //                   Eigen's vectorised cross-product.
    //   BR_DIRTY_MV   → one Matrix4f * Vector4f to transport the cached
    //                   centroid into the new view frame.
    //   BR_DIRTY_EMIT → no geometry work; emission is re-fetched at the
    //                   light-emission step below.
    //   0             → nothing.
    //
    // The cache lives in `st->group_cache` and persists across renders.
    // Groups absent from the current st->groups list are dropped.
    SceneParams scene;
    {
        // Drop stale cache entries (groups no longer declared).
        {
            std::unordered_map<int, BaseRasterizerState::GroupCache> kept;
            kept.reserve(st->groups.size());
            for (const BRGroup& g : st->groups) {
                auto it = st->group_cache.find(g.group_id);
                if (it != st->group_cache.end())
                    kept.emplace(g.group_id, std::move(it->second));
            }
            st->group_cache.swap(kept);
        }

        // Update each declared group according to its dirty flags.
        for (const BRGroup& g : st->groups) {
            BaseRasterizerState::GroupCache& c = st->group_cache[g.group_id];
            c.mat_id     = g.mat_id;
            c.tri_offset = g.tri_offset;
            c.tri_count  = g.tri_count;

            // Parse mv (column-major).
            Matrix4f mv;
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row)
                    mv(row, col) = g.mv[col*4 + row];

            const bool need_geom = (g.dirty & BR_DIRTY_GEOM) || !c.have_geom;
            const bool need_mv   = (g.dirty & BR_DIRTY_MV) && !need_geom;

            if (need_geom) {
                // Parallel area-weighted centroid reduction over the group's
                // triangle slice using Eigen Map for vectorised arithmetic.
                int t0 = std::max(0, g.tri_offset);
                int t1 = std::min(n_tris, g.tri_offset + g.tri_count);
                int tn = std::max(0, t1 - t0);

                if (tn > 0) {
                    // Each thread accumulates its own (wpos, wsum) then we
                    // reduce.  parallel_for splits over chunks of triangles.
                    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency());
                    std::vector<Vector3f> wpos_acc(n_threads, Vector3f::Zero());
                    std::vector<float>    wsum_acc(n_threads, 0.0f);

                    parallel_for(n_threads, [&](int tid) {
                        int chunk = (tn + n_threads - 1) / n_threads;
                        int s = t0 + tid * chunk;
                        int e = std::min(t0 + (tid + 1) * chunk, t1);
                        Vector3f wp = Vector3f::Zero();
                        float    ws = 0.0f;
                        for (int i = s; i < e; ++i) {
                            const float* base = verts_view + i * tri_stride;
                            // Eigen::Map for SIMD-friendly vec3 loads.
                            Eigen::Map<const Vector3f> p0(base + 0);
                            Eigen::Map<const Vector3f> p1(base + VS);
                            Eigen::Map<const Vector3f> p2(base + 2*VS);
                            Vector3f e1 = p1 - p0;
                            Vector3f e2 = p2 - p0;
                            float area = 0.5f * e1.cross(e2).norm();
                            if (area <= 0.0f) continue;
                            Vector3f cc = (p0 + p1 + p2) * (1.0f / 3.0f);
                            wp += cc * area;
                            ws += area;
                        }
                        wpos_acc[tid] = wp;
                        wsum_acc[tid] = ws;
                    });

                    Vector3f wp_total = Vector3f::Zero();
                    float    ws_total = 0.0f;
                    for (int t = 0; t < n_threads; ++t) {
                        wp_total += wpos_acc[t];
                        ws_total += wsum_acc[t];
                    }
                    if (ws_total > 1e-8f) {
                        c.centroid_v = wp_total / ws_total;
                        c.area_sum   = ws_total;
                        c.last_mv    = mv;
                        c.have_geom  = true;
                    } else {
                        c.area_sum  = 0.0f;
                        c.have_geom = false;
                    }
                } else {
                    c.area_sum  = 0.0f;
                    c.have_geom = false;
                }
            } else if (need_mv) {
                // Cheap path: re-transport cached centroid via mv_new * mv_old.inverse().
                if (c.have_geom) {
                    Matrix4f delta = mv * c.last_mv.inverse();
                    Eigen::Vector4f c4(c.centroid_v.x(), c.centroid_v.y(), c.centroid_v.z(), 1.0f);
                    Eigen::Vector4f cn = delta * c4;
                    c.centroid_v = cn.head<3>();
                    c.last_mv    = mv;
                }
            }
        }

        // Emit lights from the cache.  Up to MAX_LIGHTS strongest entries.
        struct Cand { Vector3f pos; Vector3f col; float inten; };
        std::vector<Cand> cands;
        cands.reserve(st->group_cache.size());

        const int n_mat = st->n_materials;
        for (auto& kv : st->group_cache) {
            const auto& c = kv.second;
            if (!c.have_geom || c.area_sum <= 1e-8f) continue;
            if (!pbr_data || c.mat_id < 0 || c.mat_id >= n_mat) continue;

            Vector3f col = pbr_emission(pbr_data, c.mat_id);
            float col_mag = col.norm();
            if (col_mag < 1e-6f) continue;     // non-emissive group → no light

            // Emit POSITIONAL light: store the cluster centroid in view
            // space and let the per-fragment shader compute L = normalize(
            // centroid - pos_v).  Directional approximation produced an
            // apparent axis flip on receivers far from the camera origin.
            cands.push_back({c.centroid_v, col, c.area_sum});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b){ return a.inten > b.inten; });
        int nl = std::min((int)cands.size(),
                          std::min(st->max_lights, SceneParams::MAX_LIGHTS));
        scene.n_lights = nl;
        for (int i = 0; i < nl; ++i) {
            scene.light_positions[i]   = cands[i].pos;
            scene.light_colors[i]      = cands[i].col;
            scene.light_intensities[i] = cands[i].inten;
        }
    }

    parallel_for(n_tiles, [&](int tile_id) {
        {
            const auto& tlist = tile_tris[tile_id];
            if (tlist.empty()) return;

            int ttx = tile_id % TX;
            int tty = tile_id / TX;
            int px0 = ttx * TS;
            int py0 = tty * TS;
            int px1 = std::min(px0 + TS, W);
            int py1 = std::min(py0 + TS, H);

            for (int ti : tlist) {
                const ScreenTri& tri = screen_tris[ti];

                // Edge function coefficients (A, B, C) for each edge
                // Edge i–j: A*(x - xi) + B*(y - yi), sign convention
                float ax[3], ay[3], ac[3];
                for (int k = 0; k < 3; ++k) {
                    int k1 = (k + 1) % 3;
                    ax[k] = -(tri.v[k1].sy - tri.v[k].sy);
                    ay[k] =   tri.v[k1].sx - tri.v[k].sx;
                    ac[k] =  -ax[k] * tri.v[k].sx - ay[k] * tri.v[k].sy;
                }

                // Total area (for barycentric normalisation)
                float area_inv = 1.0f /
                    (ax[0] * tri.v[2].sx + ay[0] * tri.v[2].sy + ac[0] + 1e-9f);
                // Correct sign: we culled back-faces so area should be negative
                // area_inv will carry the sign; barycentric coords stay positive

                float color_blend       = 1.0f;
                float emit_gain         = 1.0f;
                float direct_lobe_power = 1.0f;
                int   profile_id        = 0;
                float translucence_gain = 0.0f;
                float bulb_radius_mm    = 0.0f;
                if (st->texstack_data && tri.mat_id >= 0 && tri.mat_id < st->n_texstack_materials) {
                    color_blend       = tex_color_blend(st->texstack_data, tri.mat_id);
                    emit_gain         = tex_emit_gain(st->texstack_data, tri.mat_id);
                    direct_lobe_power = tex_direct_lobe_power(st->texstack_data, tri.mat_id);
                    profile_id        = static_cast<int>(tex_model_flags(st->texstack_data, tri.mat_id)) & 0xFF;
                    translucence_gain = tex_translucence_gain(st->texstack_data, tri.mat_id);
                    bulb_radius_mm    = tex_bulb_radius_mm(st->texstack_data, tri.mat_id);
                }

                for (int py = py0; py < py1; ++py) {
                    float cy = py + 0.5f;
                    // Bail out early if this row is entirely outside the bbox.
                    if (cy < tri.bbox_y0 || cy > tri.bbox_y1) continue;

                    // Row-start edge values: compute once, then step by ax[k] per pixel.
                    float row_e0 = ax[0] * (px0 + 0.5f) + ay[0] * cy + ac[0];
                    float row_e1 = ax[1] * (px0 + 0.5f) + ay[1] * cy + ac[1];
                    float row_e2 = ax[2] * (px0 + 0.5f) + ay[2] * cy + ac[2];

                    for (int px = px0; px < px1; ++px,
                                                  row_e0 += ax[0],
                                                  row_e1 += ax[1],
                                                  row_e2 += ax[2]) {
                        float cx = px + 0.5f;

                        // Quick bbox reject (x-axis only; y already checked).
                        if (cx < tri.bbox_x0 || cx > tri.bbox_x1) continue;

                        // area_inv < 0 after back-face culling, so inside pixels
                        // satisfy e0/e1/e2 <= 0.
                        float e0 = row_e0;
                        float e1 = row_e1;
                        float e2 = row_e2;
                        if (e0 > 0.0f || e1 > 0.0f || e2 > 0.0f) continue;

                        float lam0 = e1 * area_inv;
                        float lam1 = e2 * area_inv;
                        float lam2 = 1.0f - lam0 - lam1;

                        float depth = lam0 * tri.v[0].ndc_z
                                    + lam1 * tri.v[1].ndc_z
                                    + lam2 * tri.v[2].ndc_z;
                        if (!std::isfinite(depth)) continue;

                        int idx = py * W + px;
                        if (depth < -1.0f || depth > 1.0f) continue;
                        if (depth >= zbuf[idx]) continue;
                        zbuf[idx] = depth;

                        float iw = lam0 * tri.v[0].inv_w
                                 + lam1 * tri.v[1].inv_w
                                 + lam2 * tri.v[2].inv_w;
                        float w  = (iw > 1e-8f) ? 1.0f / iw : 1.0f;

                        Vector3f pos_v =
                            (tri.v[0].pos_v * (lam0 * tri.v[0].inv_w)
                           + tri.v[1].pos_v * (lam1 * tri.v[1].inv_w)
                           + tri.v[2].pos_v * (lam2 * tri.v[2].inv_w)) * w;
                        Vector3f nrm_v =
                            (tri.v[0].nrm_v * (lam0 * tri.v[0].inv_w)
                           + tri.v[1].nrm_v * (lam1 * tri.v[1].inv_w)
                           + tri.v[2].nrm_v * (lam2 * tri.v[2].inv_w)) * w;
                        Vector2f uv =
                            (tri.v[0].uv * (lam0 * tri.v[0].inv_w)
                           + tri.v[1].uv * (lam1 * tri.v[1].inv_w)
                           + tri.v[2].uv * (lam2 * tri.v[2].inv_w)) * w;

                        // Skip per-pixel normalize() of nrm_v and pos_v:
                        // kernel_geometry_terms() renormalizes N, and the
                        // sign of dot(nrm_v, -pos_v) is invariant under any
                        // positive scaling.
                        bool front = nrm_v.dot(-pos_v) >= 0.0f;
                        Vector4f emit_uv  = sample_emit_uv(st, tri.mat_id, uv);
                        Vector4f color_uv = sample_color_uv(st, tri.mat_id, uv);
                        DepthSample depth_sample = sample_depth_uv4(st, tri.mat_id, uv);

                        float alpha;
                        Vector3f col = shade(pos_v, nrm_v, front,
                                             tri.mat_id,
                                             pbr_data, phong_data, enamel_data,
                                             scene,
                                             st->enable_specular,
                                             st->enable_emission_direct,
                                             emit_uv,
                                             color_uv,
                                             color_blend,
                                             emit_gain,
                                             direct_lobe_power,
                                             profile_id,
                                             depth_sample,
                                             translucence_gain,
                                             bulb_radius_mm,
                                             &alpha);

                        float* p = cbuf + idx * 4;
                        if (alpha >= 0.9999f) {
                            p[0] = col.x();
                            p[1] = col.y();
                            p[2] = col.z();
                            p[3] = 1.0f;
                        } else {
                            float inv_a = 1.0f - alpha;
                            p[0] = col.x() * alpha + p[0] * inv_a;
                            p[1] = col.y() * alpha + p[1] * inv_a;
                            p[2] = col.z() * alpha + p[2] * inv_a;
                            p[3] = std::min(1.0f, p[3] + alpha);
                        }
                    }
                }
            }
        }
    });
}

void br_render(BaseRasterizerState* st,
               const float* verts_view,
               const int*   mat_ids,
               int          n_tris,
               const float* mvp)
{
    br_render_impl(st, verts_view, mat_ids, n_tris, mvp, 6);
}

void br_render_textured(BaseRasterizerState* st,
                        const float* verts_view_uv,
                        const int*   mat_ids,
                        int          n_tris,
                        const float* mvp)
{
    br_render_impl(st, verts_view_uv, mat_ids, n_tris, mvp, 8);
}

/* ── Readback ────────────────────────────────────────────────────────────── */

static inline float linear_to_srgb(float c) {
    c = std::max(0.0f, std::min(1.0f, c));
    return (c <= 0.0031308f) ? c * 12.92f
                              : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

void br_readback_u8(const BaseRasterizerState* st, uint8_t* out) {
    const float* c = st->cbuf.data();
    int n = st->width * st->height;
    for (int i = 0; i < n; ++i) {
        out[4*i+0] = static_cast<uint8_t>(linear_to_srgb(c[4*i+0]) * 255.0f + 0.5f);
        out[4*i+1] = static_cast<uint8_t>(linear_to_srgb(c[4*i+1]) * 255.0f + 0.5f);
        out[4*i+2] = static_cast<uint8_t>(linear_to_srgb(c[4*i+2]) * 255.0f + 0.5f);
        out[4*i+3] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, c[4*i+3])) * 255.0f + 0.5f);
    }
}

void br_readback_f32(const BaseRasterizerState* st, float* out) {
    std::memcpy(out, st->cbuf.data(), static_cast<size_t>(st->width * st->height * 4) * sizeof(float));
}

const float* br_readback_f32_ptr(const BaseRasterizerState* st) {
    if (!st) return nullptr;
    return st->cbuf.data();
}

int br_width (const BaseRasterizerState* st) { return st->width;  }
int br_height(const BaseRasterizerState* st) { return st->height; }

} // extern "C"
