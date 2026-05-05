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
    static constexpr int MAX_LIGHTS = 32;
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
        float spec  = std::pow(NdotH, std::max(1.0f, m.shininess));

        // Lambertian diffuse: surface albedo * incoming radiance * cos(theta)
        col += g.base.cwiseProduct(Lcol) * NdotL;

        // Phong specular: emitter colour modulated by Fresnel reflectance.
        col += Lcol.cwiseProduct(f.F) * (m.spec_strength * spec);

        spec_color_acc    += Lcol * spec;
        spec_strength_acc += spec;
    }

    if (spec_color_out)    *spec_color_out    = spec_color_acc;
    if (spec_strength_out) *spec_strength_out = spec_strength_acc;

    // Surface's own emission (real — the material radiates).
    col += m.emission;
    return col;
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
    const Vector3f& pos_v,  // fragment position in view space
    const Vector3f& norm_v, // interpolated normal in view space
    bool            front_facing,
    int             mat_id,
    const float*    pbr_data,
    const float*    phong_data,
    const float*    enamel_data,
    const SceneParams& sc,
    float*          alpha_out)
{
    MaterialSample m = kernel_sample_material(mat_id, pbr_data, phong_data, enamel_data);
    *alpha_out = m.opacity;

    GeomTerms g = kernel_geometry_terms(pos_v, norm_v, front_facing, m, sc);
    FresnelTerms f = kernel_fresnel_terms(m, g.NdotV);

    Vector3f spec_col(0.0f, 0.0f, 0.0f);
    float spec = 0.0f;
    Vector3f col = kernel_shade_lights(m, pos_v, g, f, sc, &spec_col, &spec);
    col = kernel_enamel_coat(col, m, g, spec_col, spec);
    return col;
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
    int n_materials           = 0;

    SceneParams scene;
    // Runtime cap on the number of cluster-lights emitted per frame.
    // Defaults to SceneParams::MAX_LIGHTS; clamped to [1, MAX_LIGHTS].
    int max_lights = SceneParams::MAX_LIGHTS;

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

/* ── Project helpers ─────────────────────────────────────────────────────── */

// Vertex projection is performed in bulk inside br_render() as a single
// proj(4×4) * positions(4 × 3N) GEMM — see the projection block there.

/* ── Render ──────────────────────────────────────────────────────────────── */

void br_render(BaseRasterizerState* st,
               const float* verts_view,
               const int*   mat_ids,
               int          n_tris,
               const float* mvp)
{
    if (!st || n_tris <= 0) return;

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
        const float* base = verts_view + i * 18;
        positions(0, 3*i+0) = base[0];   positions(1, 3*i+0) = base[1];
        positions(2, 3*i+0) = base[2];   positions(3, 3*i+0) = 1.0f;
        positions(0, 3*i+1) = base[6];   positions(1, 3*i+1) = base[7];
        positions(2, 3*i+1) = base[8];   positions(3, 3*i+1) = 1.0f;
        positions(0, 3*i+2) = base[12];  positions(1, 3*i+2) = base[13];
        positions(2, 3*i+2) = base[14];  positions(3, 3*i+2) = 1.0f;
    }
    Eigen::Matrix<float, 4, Eigen::Dynamic> clip_all = proj * positions;

    st->screen_tris.resize(static_cast<size_t>(n_tris));
    for (int i = 0; i < n_tris; ++i) {
        ScreenTri& st_tri = st->screen_tris[i];
        const float* base = verts_view + i * 18;  // 3 verts × 6 floats
        for (int k = 0; k < 3; ++k) {
            ProjVertex& pv = st_tri.v[k];
            const float* vv = base + k * 6;
            pv.pos_v = Vector3f(vv[0], vv[1], vv[2]);
            pv.nrm_v = Vector3f(vv[3], vv[4], vv[5]);
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
                            const float* base = verts_view + i * 18;
                            // Eigen::Map for SIMD-friendly vec3 loads.
                            Eigen::Map<const Vector3f> p0(base + 0);
                            Eigen::Map<const Vector3f> p1(base + 6);
                            Eigen::Map<const Vector3f> p2(base + 12);
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

                        // Quick bbox reject (x-axis only; y already checked)
                        if (cx < tri.bbox_x0 || cx > tri.bbox_x1) continue;

                        // Edge function test (values maintained incrementally)
                        float e0 = row_e0;
                        float e1 = row_e1;
                        float e2 = row_e2;
                        // All must have the same sign as area
                        // area_inv < 0 (back-face culled), so we want e0,e1,e2 < 0
                        if (e0 > 0.0f || e1 > 0.0f || e2 > 0.0f) continue;

                        // Barycentric (perspective-correct)
                        float lam0 = e1 * area_inv;  // weight for v[0]
                        float lam1 = e2 * area_inv;  // weight for v[1]
                        float lam2 = 1.0f - lam0 - lam1;

                        // Perspective-correct depth and attributes
                        float iw = lam0 * tri.v[0].inv_w
                                 + lam1 * tri.v[1].inv_w
                                 + lam2 * tri.v[2].inv_w;
                        float w  = (iw > 1e-8f) ? 1.0f / iw : 1.0f;

                        // Window depth is linearly interpolated after
                        // perspective division.  OpenGL convention: -1 is
                        // the near plane and +1 is the far plane.
                        float depth = lam0 * tri.v[0].ndc_z
                                    + lam1 * tri.v[1].ndc_z
                                    + lam2 * tri.v[2].ndc_z;
                        if (!std::isfinite(depth)) continue;

                        int idx = py * W + px;
                        if (depth < -1.0f || depth > 1.0f) continue;
                        if (depth >= zbuf[idx]) continue;  // depth test (smaller = closer)
                        zbuf[idx] = depth;

                        // Interpolate view-space position and normal
                        Vector3f pos_v =
                            (tri.v[0].pos_v * (lam0 * tri.v[0].inv_w)
                           + tri.v[1].pos_v * (lam1 * tri.v[1].inv_w)
                           + tri.v[2].pos_v * (lam2 * tri.v[2].inv_w)) * w;
                        Vector3f nrm_v =
                            (tri.v[0].nrm_v * (lam0 * tri.v[0].inv_w)
                           + tri.v[1].nrm_v * (lam1 * tri.v[1].inv_w)
                           + tri.v[2].nrm_v * (lam2 * tri.v[2].inv_w)) * w;
                        // Skip per-pixel normalize() of nrm_v and pos_v:
                        // kernel_geometry_terms() renormalizes N, and the
                        // sign of dot(nrm_v, -pos_v) is invariant under any
                        // positive scaling — saving two sqrts per fragment.

                        // Shade
                        bool front = nrm_v.dot(-pos_v) >= 0.0f;
                        float alpha;
                        Vector3f col = shade(pos_v, nrm_v, front,
                                             tri.mat_id,
                                             pbr_data, phong_data, enamel_data,
                                             scene, &alpha);

                        // Alpha-blend over existing buffer
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

int br_width (const BaseRasterizerState* st) { return st->width;  }
int br_height(const BaseRasterizerState* st) { return st->height; }

} // extern "C"
