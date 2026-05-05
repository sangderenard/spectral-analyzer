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
    std::function<void(int,int)> job;
    int  job_n  = 0;
    int  next   = 0;
    int  active = 0;
    bool quit   = false;

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
            std::function<void(int,int)> fn;
            int s, e;
            {
                std::unique_lock<std::mutex> lk(mx);
                cv_wake.wait(lk, [this]{ return quit || next < job_n; });
                if (quit) return;
                int chunk = std::max(1, (job_n + n - 1) / n);
                s = next; next = std::min(next + chunk, job_n); e = next;
                ++active;
                fn = job;
            }
            if (s < e) fn(s, e);
            {
                std::lock_guard<std::mutex> lk(mx);
                --active;
                if (next >= job_n && active == 0) cv_done.notify_one();
            }
        }
    }

    void run(int total, std::function<void(int,int)> fn) {
        if (total <= 0) return;
        {
            std::unique_lock<std::mutex> lk(mx);
            job = std::move(fn); job_n = total; next = 0; active = 0;
        }
        cv_wake.notify_all();
        std::unique_lock<std::mutex> lk(mx);
        cv_done.wait(lk, [this]{ return next >= job_n && active == 0; });
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
    rast_pool().run(n, [&](int s, int e){ fn(s, e); });
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
    Vector3f light_v       = {0.45f, 0.78f, 0.44f};
    Vector3f scene_rgb     = {1.0f, 1.0f, 1.0f};
    float    scene_indirect = 0.0f;
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
    // Material fetch
    Vector3f albedo   = pbr_data   ? pbr_albedo   (pbr_data,   mat_id) : Vector3f(0.5f,0.5f,0.5f);
    float    rough    = pbr_data   ? pbr_roughness (pbr_data,   mat_id) : 0.5f;
    float    metallic = pbr_data   ? pbr_metallic  (pbr_data,   mat_id) : 0.0f;
    float    ior      = pbr_data   ? pbr_ior       (pbr_data,   mat_id) : 1.5f;
    float    opacity  = pbr_data   ? pbr_opacity   (pbr_data,   mat_id) : 1.0f;
    Vector3f emission = pbr_data   ? pbr_emission  (pbr_data,   mat_id) : Vector3f(0.0f,0.0f,0.0f);

    float    ambient  = phong_data ? ph_ambient    (phong_data, mat_id) : 0.18f;
    float    specstr  = phong_data ? ph_specstr    (phong_data, mat_id) : 0.05f;
    float    shini    = phong_data ? ph_shininess  (phong_data, mat_id) : 32.0f;
    Vector3f inner    = phong_data ? ph_inner      (phong_data, mat_id) : albedo * 0.55f;

    float enam_thick  = enamel_data ? en_thick(enamel_data, mat_id) : 0.0f;

    *alpha_out = opacity;

    // Geometry
    Vector3f N = front_facing ? norm_v : (-norm_v);
    N.normalize();
    Vector3f L = sc.light_v.normalized();
    Vector3f V = (-pos_v).normalized();
    Vector3f H = (L + V).normalized();

    float NdotL = std::max(0.0f, N.dot(L));
    float NdotH = std::max(0.0f, N.dot(H));
    float NdotV = std::max(0.0f, N.dot(V));

    // Base colour (front / back face)
    Vector3f base = front_facing ? albedo : inner;

    // Schlick Fresnel — Fresnel F0 from IOR and metallic
    float ior_f0 = (ior - 1.0f) / (ior + 1.0f);
    ior_f0 *= ior_f0;
    Vector3f F0  = (1.0f - metallic) * Vector3f(ior_f0, ior_f0, ior_f0)
                 + metallic          * albedo;
    float F_scal = schlick(ior_f0, NdotV);
    Vector3f F   = F0 + (Vector3f(1.0f,1.0f,1.0f) - F0) * std::pow(1.0f - NdotV, 5.0f);

    // Phong diffuse + specular
    float diff = NdotL;
    float spec = std::pow(NdotH, std::max(1.0f, shini));

    // Scene-tinted ambient (indirect fill in shadowed regions)
    Vector3f scene_rgb_v = sc.scene_rgb;
    Vector3f amb_light   = ambient * (Vector3f(1.0f,1.0f,1.0f) * 0.45f
                                    + scene_rgb_v * 0.55f);
    float shadow_fill    = sc.scene_indirect * 0.28f * (1.0f - diff);

    // Specular colour: warm tint blended with scene
    Vector3f spec_col = Vector3f(1.0f,0.93f,0.70f) * 0.65f + scene_rgb_v * 0.35f;

    Vector3f col = base.cwiseProduct(amb_light + (0.78f + shadow_fill) * Vector3f::Ones() * diff)
                 + spec_col.cwiseProduct(F) * (specstr * spec);

    // Rim
    float rim = std::pow(1.0f - NdotV, 3.0f);
    col += base * rim * (0.14f + 0.08f * sc.scene_indirect);

    // Emission
    col += emission;

    // Enamel coat
    if (enam_thick > 0.0f) {
        float enam_ior_v = enamel_data ? en_ior  (enamel_data, mat_id) : 1.52f;
        Vector3f enam_c  = enamel_data ? en_color(enamel_data, mat_id) : Vector3f(1.0f,1.0f,1.0f);

        // Thin-film OPD (nm): 2 * n * d * cos(theta_t), simplified as 2*n*d*cosV
        float opd_nm = 2.0f * enam_ior_v * enam_thick * NdotV;
        Vector3f fringe = thin_film_fringe(opd_nm);
        float fringe_w  = std::min(0.45f, enam_thick / 800.0f);
        Vector3f tinted = col.cwiseProduct(enam_c).cwiseProduct(
            Vector3f::Ones() + fringe * fringe_w);

        // Enamel Fresnel gloss bump
        float eF0 = (enam_ior_v - 1.0f) / (enam_ior_v + 1.0f);
        eF0 *= eF0;
        float eF  = schlick(eF0, NdotV);
        tinted   += Vector3f(spec_col) * (eF * 0.28f * spec);

        col = tinted * 0.35f + col * 0.65f;
    }

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
    float* c = st->cbuf.data();
    int n = st->width * st->height;
    for (int i = 0; i < n; ++i) {
        c[4*i+0] = r; c[4*i+1] = g; c[4*i+2] = b; c[4*i+3] = a;
    }
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

void br_set_scene(BaseRasterizerState* st,
                  const float* light_v,
                  const float* scene_rgb,
                  float        scene_indirect) {
    if (light_v)   st->scene.light_v   = Vector3f(light_v[0],   light_v[1],   light_v[2]).normalized();
    if (scene_rgb) st->scene.scene_rgb = Vector3f(scene_rgb[0], scene_rgb[1], scene_rgb[2]);
    st->scene.scene_indirect = scene_indirect;
}

/* ── Project helpers ─────────────────────────────────────────────────────── */

extern "C++" {

static ProjVertex project_vertex(
    const float* vv,        /* view-space [x,y,z,nx,ny,nz] */
    const Matrix4f& proj,   /* projection-only matrix (clip = proj * view) */
    float half_w, float half_h)
{
    ProjVertex pv;
    // View-space position and normal
    pv.pos_v = Vector3f(vv[0], vv[1], vv[2]);
    pv.nrm_v = Vector3f(vv[3], vv[4], vv[5]);
    pv.inv_w = 0.0f;
    pv.ndc_z = 1.0f;
    pv.valid = false;

    // Project: clip = proj * view_pos (w=1 for affine input)
    Vector4f clip = proj * Vector4f(vv[0], vv[1], vv[2], 1.0f);

    if (!std::isfinite(clip.x()) || !std::isfinite(clip.y()) ||
        !std::isfinite(clip.z()) || !std::isfinite(clip.w()) ||
        clip.w() <= 1e-6f) {
        pv.sx = pv.sy = -1e9f;
        return pv;
    }
    float inv_w = 1.0f / clip.w();
    float ndcx  = clip.x() * inv_w;
    float ndcy  = clip.y() * inv_w;
    float ndcz  = clip.z() * inv_w;
    if (!std::isfinite(ndcx) || !std::isfinite(ndcy) || !std::isfinite(ndcz)) {
        pv.sx = pv.sy = -1e9f;
        return pv;
    }

    // NDC → screen pixels (y flipped: NDC +1 = top row)
    pv.sx    = (ndcx + 1.0f) * half_w;
    pv.sy    = (1.0f - ndcy) * half_h;   // flip Y
    pv.inv_w = inv_w;
    pv.ndc_z = ndcz;
    pv.valid = true;
    return pv;
}

} // extern "C++"

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

    // Project all vertices
    st->screen_tris.resize(static_cast<size_t>(n_tris));
    for (int i = 0; i < n_tris; ++i) {
        ScreenTri& st_tri = st->screen_tris[i];
        const float* base = verts_view + i * 18;  // 3 verts × 6 floats
        st_tri.v[0] = project_vertex(base +  0, proj, half_w, half_h);
        st_tri.v[1] = project_vertex(base +  6, proj, half_w, half_h);
        st_tri.v[2] = project_vertex(base + 12, proj, half_w, half_h);
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
    const SceneParams scene  = st->scene;

    parallel_for(n_tiles, [&](int s, int e) {
        for (int tile_id = s; tile_id < e; ++tile_id) {
            const auto& tlist = tile_tris[tile_id];
            if (tlist.empty()) continue;

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
                    for (int px = px0; px < px1; ++px) {
                        float cx = px + 0.5f;
                        float cy = py + 0.5f;

                        // Quick bbox reject
                        if (cx < tri.bbox_x0 || cx > tri.bbox_x1 ||
                            cy < tri.bbox_y0 || cy > tri.bbox_y1) continue;

                        // Edge function test
                        float e0 = ax[0] * cx + ay[0] * cy + ac[0];
                        float e1 = ax[1] * cx + ay[1] * cy + ac[1];
                        float e2 = ax[2] * cx + ay[2] * cy + ac[2];
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
                        nrm_v.normalize();

                        // Shade
                        bool front = nrm_v.dot(-pos_v.normalized()) >= 0.0f;
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
