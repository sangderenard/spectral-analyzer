#define _USE_MATH_DEFINES
#ifndef NOMINMAX
#  define NOMINMAX
#endif
/**
 * ray_tracer.cpp — Complex spectral 3-D ray tracer.
 *
 * Traces acoustic / EM waves in 3-D using Eigen + std::complex<double>.
 *
 * Each ray carries a complex amplitude vector A[b] (one entry per frequency
 * band b).  As the ray travels distance Δr through free space:
 *
 *   A[b] *= exp(j * k[b] * Δr)             — phase accumulation
 *   A[b] *= exp(-atmo_abs[b] * Δr)         — atmospheric decay
 *   A[b] *= 1 / (1 + r_total + Δr * 0.5)  — geometric spreading
 *
 * At each surface hit the amplitude is additionally multiplied by the
 * material's complex reflectance R[b]:
 *
 *   A[b] *= R[b]   where R[b] = refl_re[tri][b] + j * refl_im[tri][b]
 *
 * Directions
 * ----------
 * Each source emits n_rays directions from a Fibonacci sphere (quasi-uniform
 * full-sphere coverage).  Directions are then rotated so that the Fibonacci
 * z-axis aligns with the source's dominant direction, and weighted by a
 * cosine^directivity_power lobe.  If the weight is below 1 % the ray is
 * skipped, which naturally gives a direction-selective emission pattern.
 *
 * Diffuse scatter
 * ---------------
 * At each surface hit a Mersenne-Twister RNG decides whether the reflected
 * direction is specular (Snell) or cosine-weighted Lambertian.  The
 * probability is diffusion[tri].  Both paths share the same complex amplitude
 * update, so the output is spectrally consistent regardless of scatter mode.
 *
 * Output buffer
 * -------------
 * One record per (source, ray, bounce, band) written in arrival order.
 * Layout: RT_FLOATS_PER_SEG = 12 floats — see ray_tracer.h.
 *
 * Acceleration
 * ------------
 * Triangle intersection uses a BVH (axis-aligned bounding volume hierarchy)
 * built at scene-load time.  The inner loop drops from O(n_tris) to O(log
 * n_tris) per bounce, which is the dominant cost at large triangle counts.
 * Propagation is computed once per (band, bounce) rather than twice, and the
 * complex phase factor uses cos/sin directly instead of std::exp(complex).
 */

#include "ray_tracer.h"
#include "triangle_groups.h"
#include "field_grid.h"
#include "thread_pool.h"
#include "optical_handlers.h"
#include "bdpt_record.h"
/* t5_hash_grid.h removed — full brute-force GPU path; no hash needed */

#include <mutex>
#include <thread>
#include <cstring>
#include "ray_pipeline.h"
#include "gl_compute.h"
#include "gpu_diag.h"

/* Compile-time parity checks: SensorRecord and FilmRecord must be exactly the
 * same size as their Python ctypes counterparts (SensorRecord: 48 floats = 192 B,
 * FilmRecord: 64 floats = 256 B).  A mismatch here means the Python layout and
 * the C struct have drifted apart and the SSBO upload will be misread.        */
static_assert(sizeof(SensorRecord) == 48 * sizeof(float),
              "SensorRecord size mismatch — Python SensorRecord is 48 floats");
static_assert(sizeof(FilmRecord) == 64 * sizeof(float),
              "FilmRecord size mismatch — Python FilmRecord is 64 floats");

#include <Eigen/Dense>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using cd   = std::complex<double>;
using V3d  = Eigen::Vector3d;
using VXcd = Eigen::VectorXcd;

/* Portable lock-free add for a plain uint32_t in a shared buffer.
 * std::atomic<uint32_t> is required to be lock-free on all targeted
 * platforms.  The reinterpret_cast is well-defined by the note in
 * [atomics.ref] that std::atomic<T> must have the same size/alignment
 * as T, and is the accepted cross-platform (GCC/Clang/MSVC) idiom when
 * std::atomic_ref (C++20) is not yet available.                            */
static inline void uv_accum_add(uint32_t& slot, uint32_t val) {
    reinterpret_cast<std::atomic<uint32_t>&>(slot)
        .fetch_add(val, std::memory_order_relaxed);
}

static inline float rt_i32_as_f32(int32_t v) {
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

static inline float rt_u32_as_f32(uint32_t v) {
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

static constexpr double TWO_PI     = 2.0 * M_PI;
static constexpr double EPS        = 1e-9;
static constexpr double T_SELF     = 1e-10;
static constexpr int    BVH_LEAF_MAX = 4;   /* triangles per BVH leaf */
static constexpr bool    RT_ENABLE_PROFILE = false;

static thread_local std::string g_rt_alloc_table;

struct RtProfileScope {
    const char* name = nullptr;
    std::chrono::steady_clock::time_point t0;

    explicit RtProfileScope(const char* label)
        : name(label), t0(std::chrono::steady_clock::now())
    {}

    ~RtProfileScope() {
        if (!RT_ENABLE_PROFILE || !name) return;
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[rt-prof] %s %.3f ms\n", name, ms);
    }
};

/* ── Unified material flags (Phase 2: single source of truth) ─────────────── */
#include "mat_flags_generated.h"

/* ── Geometry ──────────────────────────────────────────────────────────────── */

struct Triangle {
    V3d v0;
    V3d edge1;   /* v1 - v0 (precomputed for Möller-Trumbore) */
    V3d edge2;   /* v2 - v0 */
    V3d normal;  /* outward unit normal */
    int mat_idx = 0;   /* index into RayTracerState::mat_buf rows (per-band record stride) */
    /* Boundary-side media relative to geometric normal:
     *   medium_pos_mat_idx = medium on +normal side
     *   medium_neg_mat_idx = medium on -normal side
     * Use -1 for ambient air/vacuum. */
    int medium_pos_mat_idx = -1;
    int medium_neg_mat_idx = -1;
    int flags   = 0;   /* MAT_FLAG_APERTURE_STOP | MAT_FLAG_EMISSIVE | … */
};

/* ── Möller-Trumbore ray-triangle intersection ─────────────────────────────── */

/* Returns true and sets t_out when the ray (orig + t*dir) hits either side
   of the triangle.  Back-face hits are real material interactions; response
   code orients the transport normal after the hit is found. */
static bool ray_triangle_hit(
    const V3d& orig, const V3d& dir,
    const Triangle& tri,
    double& t_out)
{
    V3d h = dir.cross(tri.edge2);
    double a = tri.edge1.dot(h);
    if (std::abs(a) < EPS)
        return false;           /* ray parallel to triangle */

    double f = 1.0 / a;
    V3d  s   = orig - tri.v0;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0)
        return false;

    V3d   q = s.cross(tri.edge1);
    double v = f * dir.dot(q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    double t = f * tri.edge2.dot(q);
    if (t <= T_SELF)
        return false;           /* behind or at origin */

    t_out = t;
    return true;
}

/* ── AABB ───────────────────────────────────────────────────────────────────── */

struct AABB {
    V3d lo{ 1e30,  1e30,  1e30};
    V3d hi{-1e30, -1e30, -1e30};

    void expand(const V3d& p) {
        for (int i = 0; i < 3; ++i) {
            if (p[i] < lo[i]) lo[i] = p[i];
            if (p[i] > hi[i]) hi[i] = p[i];
        }
    }
    void expand(const AABB& o) { expand(o.lo); expand(o.hi); }

    /* Slab test.  inv_dir = 1/dir (IEEE-754 handles ±inf correctly). */
    bool hit(const V3d& orig, const V3d& inv_dir, double t_max) const {
        double tmin = T_SELF;
        for (int i = 0; i < 3; ++i) {
            double a = (lo[i] - orig[i]) * inv_dir[i];
            double b = (hi[i] - orig[i]) * inv_dir[i];
            tmin  = std::max(tmin,  std::min(a, b));
            t_max = std::min(t_max, std::max(a, b));
        }
        return tmin <= t_max;
    }

    int    longest_axis()      const { V3d d=hi-lo; return (d[0]>=d[1]&&d[0]>=d[2])?0:(d[1]>=d[2]?1:2); }
    double centroid(int axis)  const { return (lo[axis] + hi[axis]) * 0.5; }
};

/* ── BVH ────────────────────────────────────────────────────────────────────── */

struct BVHNode {
    AABB aabb;
    int  left      = -1;  /* -1 = leaf node */
    int  right     = -1;
    int  tri_start =  0;  /* valid for leaves: [tri_start, tri_end) into tri_ids */
    int  tri_end   =  0;
};

/* Recursive top-down build using median centroid split on the longest axis.
   `ids` is a mutable index array; `aabbs[i]` is the AABB for triangle i. */
static int bvh_build(
    std::vector<BVHNode>& nodes,
    std::vector<int>&     ids,
    const std::vector<AABB>& aabbs,
    int begin, int end)
{
    int node_idx = static_cast<int>(nodes.size());
    nodes.push_back({});

    AABB box;
    for (int i = begin; i < end; ++i)
        box.expand(aabbs[static_cast<size_t>(ids[i])]);

    if (end - begin <= BVH_LEAF_MAX) {
        nodes[node_idx] = {box, -1, -1, begin, end};
        return node_idx;
    }

    int axis = box.longest_axis();
    int mid  = (begin + end) / 2;
    std::nth_element(
        ids.begin() + begin,
        ids.begin() + mid,
        ids.begin() + end,
        [&](int a, int b) {
            return aabbs[static_cast<size_t>(a)].centroid(axis)
                 < aabbs[static_cast<size_t>(b)].centroid(axis);
        });

    /* Store box before recursive calls that may reallocate `nodes`. */
    nodes[node_idx].aabb = box;
    int left_idx  = bvh_build(nodes, ids, aabbs, begin, mid);
    int right_idx = bvh_build(nodes, ids, aabbs, mid,   end);
    nodes[node_idx].left  = left_idx;
    nodes[node_idx].right = right_idx;
    return node_idx;
}

/* Iterative BVH traversal.  Finds the nearest triangle intersection ≤ t_min.
   Updates t_min and hit_tri in-place; hit_tri is unchanged on no hit. */
static void bvh_query(
    const std::vector<BVHNode>&  nodes,
    const std::vector<int>&      tri_ids,
    const std::vector<Triangle>& tris,
    const V3d& orig, const V3d& dir, const V3d& inv_dir,
    double& t_min, int& hit_tri)
{
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;  /* root is always at index 0 */

    while (sp > 0) {
        const BVHNode& node = nodes[static_cast<size_t>(stack[--sp])];

        if (!node.aabb.hit(orig, inv_dir, t_min))
            continue;

        if (node.left == -1) {
            /* Leaf: test each triangle. */
            for (int i = node.tri_start; i < node.tri_end; ++i) {
                int ti = tri_ids[static_cast<size_t>(i)];
                double t;
                if (ray_triangle_hit(orig, dir, tris[static_cast<size_t>(ti)], t)
                    && t > T_SELF && t < t_min)
                {
                    t_min   = t;
                    hit_tri = ti;
                }
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }
}

/* Camera visibility helpers are defined later, after RayTracerState. */

/* ── Direction sampling ────────────────────────────────────────────────────── */

/* Return the i-th direction from an N-point Fibonacci sphere.
   The sphere is oriented so that the z-axis points along primary_dir. */
static V3d fibonacci_sphere_dir(int i, int N, const V3d& primary_dir)
{
    static const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;  /* golden ratio */

    /* Canonical Fibonacci point on the unit sphere. */
    double cos_t = 1.0 - (2.0 * (i + 0.5)) / static_cast<double>(N);
    double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
    double phi   = TWO_PI * (i / PHI - std::floor(i / PHI));

    V3d v(sin_t * std::cos(phi), sin_t * std::sin(phi), cos_t);

    /* Rodrigues rotation: local z-axis → primary_dir. */
    V3d from(0.0, 0.0, 1.0);
    V3d to = primary_dir.normalized();
    double cos_a = from.dot(to);

    if (cos_a > 1.0 - EPS)
        return v;                     /* already aligned */
    if (cos_a < -(1.0 - EPS))
        return V3d(-v.x(), -v.y(), -v.z());   /* anti-aligned, flip */

    V3d  axis  = from.cross(to);
    double sin_a = axis.norm();
    axis /= sin_a;
    double one_c = 1.0 - cos_a;
    /* Rodrigues: v_rot = v*cos + (axis×v)*sin + axis*(axis·v)*(1-cos) */
    return (v * cos_a
            + axis.cross(v) * sin_a
            + axis * (axis.dot(v) * one_c)).normalized();
}

/* Cosine-weighted random direction in the hemisphere around normal n. */
static V3d cosine_hemisphere(const V3d& n, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r1 = U(rng);
    double r2 = U(rng);

    /* Malley's method: sample disk, project to hemisphere. */
    double r   = std::sqrt(r1);
    double phi = TWO_PI * r2;
    double x   = r * std::cos(phi);
    double y   = r * std::sin(phi);
    double z   = std::sqrt(std::max(0.0, 1.0 - r1));

    /* Build orthonormal frame around n. */
    V3d up = (std::abs(n.x()) < 0.9) ? V3d(1, 0, 0) : V3d(0, 1, 0);
    V3d t  = n.cross(up).normalized();
    V3d b  = n.cross(t);

    return (x * t + y * b + z * n).normalized();
}

/* Sample a hemisphere lobe around axis with PDF proportional to cos(theta)^k.
 * k=0 gives uniform hemisphere; larger k narrows around axis. */
static V3d cosine_power_lobe(const V3d& axis, double k, std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double u1 = U(rng);
    const double u2 = U(rng);
    const double kk = std::max(0.0, k);

    const double cos_t = std::pow(u1, 1.0 / (kk + 1.0));
    const double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
    const double phi   = TWO_PI * u2;

    const double x = sin_t * std::cos(phi);
    const double y = sin_t * std::sin(phi);
    const double z = cos_t;

    const V3d n = axis.normalized();
    const V3d up = (std::abs(n.x()) < 0.9) ? V3d(1, 0, 0) : V3d(0, 1, 0);
    const V3d t = n.cross(up).normalized();
    const V3d b = n.cross(t);
    return (x * t + y * b + z * n).normalized();
}

/* ── Per-worker output chunk ─────────────────────────────────────────────────
 * Holds a thread-local (or call-local) segment buffer and running stats.
 * Using a local chunk rather than writing directly into the caller's buffer
 * makes `count` fully private — no shared reference required — and enables
 * future parallelisation without adding locks around the output path.
 *
 * flush_to(): copies collected segments into the caller-provided output array
 * and credits any excess (over out_cap - already_written) as dropped.
 */
struct RayWorkerChunk {
    std::vector<float> segs;
    RtTraceStats       stats = {0, 0, 0};

    void write_segment(
        const V3d& p0, const V3d& p1,
        int src_id, int bounce, int band,
        cd amp, double path_len)
    {
        segs.resize(segs.size() + RT_FLOATS_PER_SEG);
        float* s = segs.data() + segs.size() - RT_FLOATS_PER_SEG;
        s[0]  = static_cast<float>(p0.x());
        s[1]  = static_cast<float>(p0.y());
        s[2]  = static_cast<float>(p0.z());
        s[3]  = static_cast<float>(p1.x());
        s[4]  = static_cast<float>(p1.y());
        s[5]  = static_cast<float>(p1.z());
        s[6]  = static_cast<float>(src_id);
        s[7]  = static_cast<float>(bounce);
        s[8]  = static_cast<float>(band);
        s[9]  = static_cast<float>(std::abs(amp));
        s[10] = static_cast<float>(std::arg(amp));
        s[11] = static_cast<float>(path_len);
        ++stats.segments_written;
    }

    /* Flush into caller-owned buffer.  Returns number of records copied.
     * Records that don't fit are counted as dropped (not silently lost). */
    int flush_to(float* out, int out_cap, int already_written)
    {
        int available = out_cap - already_written;
        int n_seg     = static_cast<int>(segs.size() / RT_FLOATS_PER_SEG);
        int n_copy    = std::min(n_seg, std::max(0, available));
        if (n_copy > 0)
            std::memcpy(out + static_cast<size_t>(already_written) * RT_FLOATS_PER_SEG,
                        segs.data(),
                        static_cast<size_t>(n_copy) * RT_FLOATS_PER_SEG * sizeof(float));
        int n_drop = n_seg - n_copy;
        stats.segments_dropped += n_drop;
        stats.segments_written -= n_drop;   /* correct: those were not written */
        return n_copy;
    }
};

/* ── Segment writer (legacy thin wrapper used by multiscale path) ────────────
 * New code should use RayWorkerChunk::write_segment instead.
 */
static inline void write_segment(
    float* buf, int& count, int cap,
    const V3d& p0, const V3d& p1,
    int src_id, int bounce, int band,
    cd amp, double path_len,
    RtTraceStats* stats = nullptr)
{
    if (count >= cap) {
        if (stats) ++stats->segments_dropped;
        return;
    }
    float* s = buf + static_cast<size_t>(count) * RT_FLOATS_PER_SEG;
    s[0]  = static_cast<float>(p0.x());
    s[1]  = static_cast<float>(p0.y());
    s[2]  = static_cast<float>(p0.z());
    s[3]  = static_cast<float>(p1.x());
    s[4]  = static_cast<float>(p1.y());
    s[5]  = static_cast<float>(p1.z());
    s[6]  = static_cast<float>(src_id);
    s[7]  = static_cast<float>(bounce);
    s[8]  = static_cast<float>(band);
    s[9]  = static_cast<float>(std::abs(amp));
    s[10] = static_cast<float>(std::arg(amp));
    s[11] = static_cast<float>(path_len);
    ++count;
    if (stats) ++stats->segments_written;
}

/* ── RayTracerState ─────────────────────────────────────────────────────────── */

/* ── Per-scene material spectral cache ───────────────────────────────────────
 * Precomputed lookup tables derived from mat_buf at scene-creation time and
 * rebuilt whenever mat_buf changes.  Replaces per-hit mat_refl_complex /
 * mat_diffusion transcendental calls with direct float reads in bounce loops.
 *
 * Layout:
 *   refl_re / refl_im     — [mat_idx * n_bands + b]
 *   diffusion             — [mat_idx]  (band-0 value)
 *   reactive_shift_hz     — [mat_idx]  (Stokes shift in Hz from band-0 slot 9)
 *   reemit_yield          — [mat_idx]  (band-0 slot 6 value)
 *   reactive_dst_band     — [mat_idx * n_bands + b]  (−1 = no valid downshift)
 */
struct MatSpectralCache {
    int n_mats  = 0;
    int n_bands = 0;
    std::vector<float>  refl_re;           /* [mat * n_bands + b] */
    std::vector<float>  refl_im;
    std::vector<float>  diffusion;         /* [mat]               */
    std::vector<double> reactive_shift_hz; /* [mat]               */
    std::vector<float>  reemit_yield;      /* [mat]               */
    std::vector<int>    reactive_dst_band; /* [mat * n_bands + b] */
};

/* ── Per-scene PBR / enamel / texture-stack surface cache ────────────────────
 * Populated by rt_build_surf_cache() from pbr_chunk (N×16), enamel_chunk (N×8)
 * and tex_stack_chunk (N×16) after ray_tracer_set_surface_chunks().
 *
 * PBR layout (16 floats):  [0..2]=albedo, [3]=roughness, [4]=metallic,
 *   [5]=transmission, [6]=ior, [7]=opacity, [8..10]=emission_rgb, [11..15]=_pad
 * Enamel layout (8 floats): [0]=thickness_nm, [1]=ior_real, [2]=ior_imag,
 *   [3]=roughness, [4..6]=tint_rgb, [7]=_pad
 * TextureStack layout (16 floats): [11]=profile_id (float-encoded uint)
 *   profile_id: 0=standard, 1=emissive, 2=SSS, 3=frosted scatter
 */
struct RtMaterialSurfaceCache {
    int n_mats = 0;
    std::vector<float>    roughness;           /* [mat]    GGX alpha          */
    std::vector<float>    metallic;            /* [mat]    conductor weight   */
    std::vector<float>    transmission;        /* [mat]    bulk transmittance */
    std::vector<float>    opacity;             /* [mat]    1 - transparency   */
    std::vector<float>    albedo;              /* [mat*3]  base RGB           */
    std::vector<float>    f0;                  /* [mat*3]  Schlick F0         */
    std::vector<float>    enamel_thickness_nm; /* [mat]    0 = no coating     */
    std::vector<float>    enamel_ior_real;     /* [mat]                       */
    std::vector<float>    enamel_ior_imag;     /* [mat]                       */
    std::vector<float>    enamel_roughness;    /* [mat]                       */
    std::vector<float>    enamel_tint;         /* [mat*3]                     */
    std::vector<uint32_t> profile_id;          /* [mat]                       */
};

struct RayTracerState {
    std::vector<Triangle>       tris;
    std::vector<BVHNode>        bvh_nodes;
    std::vector<int>            bvh_tri_ids;
    std::vector<double>         tri_areas;   /* area of each triangle (m²) */
    int                         n_bands = 0;
    Eigen::VectorXd             k_real;      /* 2π f_n / c  (wavenumber, ambient) */
    Eigen::VectorXd             atmo_abs;    /* Np/m per band */
    /* ── Unified material buffer (Phase 2 cutover) ────────────────────────
     * Flat (N_mat * MAX_SPECTRAL_BANDS, 12) float32, identical bytes to the
     * GLSL MatBuf SSBO (binding 10).  Each Triangle::mat_idx selects a
     * material; the per-band SpectralBandRecord is read at hit time via the
     * mat_band_record/mat_refl_complex helpers.  Single source of truth
     * shared with material_db.py / mat_flags.py / GLSL shaders. */
    std::vector<float>          mat_buf;
    int                         mat_n_mats = 0;
    MatSpectralCache            mat_cache;     /* precomputed spectral lookups */
    /* ── PBR / enamel / texture-stack chunks (set via ray_tracer_set_surface_chunks) ─ */
    std::vector<float>          pbr_chunk;       /* (mat_n_mats, 16) PBRBaseRecord   */
    std::vector<float>          enamel_chunk;    /* (mat_n_mats,  8) EnamelRecord    */
    std::vector<float>          tex_stack_chunk; /* (mat_n_mats, 16) TextureStack    */
    RtMaterialSurfaceCache      surf_cache;      /* precomputed GGX/enamel lookups   */
    std::vector<RtScaleContext> scale_contexts; /* multi-scale zones, smallest-radius-first */
    double                      speed_m_s = 343.0; /* cached for context k scaling */
    Eigen::VectorXd             freq_hz_vec; /* cached for context k scaling */

    /* ── Stateful ray scheduler ─────────────────────────────────────────────
     * ray_pool[i]        : persistent state for live ray i
     * ray_amp_pool[i*nb + b] : complex amplitude for ray i band b
     * geo_queue          : ray indices currently in the coarse geometric stage
     * ctx_queues[ci]     : ray indices currently inside scale_contexts[ci]
     * live_max_bounces   : set by ray_tracer_spawn, kills ray after N bounces
     * live_min_amplitude : set by ray_tracer_spawn, kills ray when |A| < this */
    std::vector<RtRayState>           ray_pool;
    std::vector<cd>                   ray_amp_pool;   /* [ray_id * n_bands + b] */
    std::vector<int>                  geo_queue;
    std::vector<std::vector<int>>     ctx_queues;     /* one per scale context  */
    int                               live_max_bounces   = 8;
    double                            live_min_amplitude = 0.005;

    /* ── Triangle-group registry (bidirectional integrator support) ─────
     * Owned vectors of group descriptors and per-group triangle-id lists.
     * Filled by ray_tracer_register_tri_group() (declared in
     * triangle_groups.h, implemented in triangle_groups.cpp).  Empty by
     * default so legacy callers see no behaviour change.
     *
     * cum_areas[g] is a CDF over tri_indices[g]; the last entry is the
     * group's total area.  Used for area-weighted emissive sampling.
     */
    /* Per-GID manifold dispatch stats.  One entry per registered tri-group, sized
     * alongside tri_groups.  Keyed by (gid, magic) — magic is set at registration
     * from the first float of the parametric payload, making each entry fully
     * self-describing without a separate look-up.  Counters are incremented by
     * CPU T2 (pipeline_refiner) and the batch trace path for every manifold hit
     * regardless of representation (parametric/MLP/LUT/absorber). */
    struct ManifoldGIDStats {
        float                 magic = 0.0f;  /* payload magic (14949=param, 14948=MLP, 14946-51=LUT) */
        std::atomic<uint64_t> ok{0};         /* rays successfully teleported through this surface    */
        std::atomic<uint64_t> abs{0};        /* rays absorbed (vignetting, TIR, unknown magic)       */
        ManifoldGIDStats() noexcept = default;
        explicit ManifoldGIDStats(float m) noexcept : magic(m) {}
        /* Copy resets counters but preserves magic — safe for vector reallocation
         * at registration time (before any tracing starts). */
        ManifoldGIDStats(const ManifoldGIDStats& o) noexcept : magic(o.magic) {}
        ManifoldGIDStats& operator=(const ManifoldGIDStats& o) noexcept {
            magic = o.magic;
            ok .store(0, std::memory_order_relaxed);
            abs.store(0, std::memory_order_relaxed);
            return *this;
        }
    };
    std::vector<ManifoldGIDStats>      gid_manifold_stats;

    std::vector<TriGroupDesc>          tri_groups;          /* descriptors  */
    std::vector<std::vector<int>>      tri_group_indices;   /* per-group   */
    std::vector<std::vector<double>>   tri_group_cum_areas; /* per-group   */
    /* Group → default material lookup table (per-group fallback).
     * -1 = "no default"; integrator falls back to per-tri MaterialDatabase
     * row.  Populated at register time from desc.default_mat_idx (or, if
     * <0, from the majority material index across the group's tris). */
    std::vector<int>                   tri_group_default_mat;
    /* Optional per-group spectral emission curve (length = n_bands).  Empty
     * inner vector = "no override; use mat_buf emission row." */
    std::vector<std::vector<float>>    tri_group_power_per_band;
    /* Camera sensor descriptor for SENSOR groups with PIXEL_CONE policy.
     * tri_group_has_camera[g] gates use; tri_group_camera[g] holds the
     * deep-copied CameraSensorDesc. */
    std::vector<int>                   tri_group_has_camera;
    std::vector<CameraSensorDesc>      tri_group_camera;
    std::vector<int>                   tri_group_parametric_kind;
    std::vector<std::vector<uint8_t>>  tri_group_parametric_payload;
    std::vector<int>                   tri_param_group_of_tri; /* tri_id -> group_id or -1 */

    /* ── UV integrator image state ──────────────────────────────────────────
     * Each tri group with uv_image_res > 0 gets a (UV_N_HDR_CHANNELS + 5*n_bands)
     * channel flat accumulator.  Layout: channel-major, each channel is res×res
     * uint32 slots.  See ray_tracer.h UV_CH_* constants for semantics.
     * Both the GPU T3 shader and the CPU T3 scatter path write atomically. */
    std::vector<int>                   tri_uv_group_of_tri;   /* tri_id -> uv group_id or -1 */
    std::vector<float>                 tri_uv_data;            /* n_tris * 6: uv0,uv1,uv2 per tri */
    std::vector<int>                   group_uv_res;           /* per-group res (0 = no UV image) */
    std::vector<int>                   group_uv_accum_offset;  /* per-group offset into uv_accum (uint32s) */
    int                                uv_accum_total = 0;     /* total uint32 slots allocated */
    mutable std::vector<uint32_t>      uv_accum_cpu;           /* flat CPU-side accumulator     */

    /* ── BSSRDF per-triangle illumination accumulator ──────────────────────
     * Forward paths that hit a transmissive surface with diffuse_frac > 0
     * accumulate their pre-interaction amplitude here (thread-safe via
     * uv_accum_add).  Backward paths (color_flag==1 / is_backward=true)
     * query this buffer to claim analytical diffuse illumination without
     * Monte Carlo scatter traversal through the diffusing volume.
     *
     * Layout per triangle: stride = 2*tri_illum_nb + 2 uint32 slots:
     *   [2*b + 0] amp_sum[b].real() × 32768  (int32 in uint32 slot)
     *   [2*b + 1] amp_sum[b].imag() × 32768  (int32 in uint32 slot)
     *   [2*nb   ] cos_sum           × 32768  (int32 in uint32 slot)
     *   [2*nb+1 ] count             as uint32 integer
     *
     * Call ray_tracer_init_illum_accum() once after geometry is set.
     * Call ray_tracer_reset_illum_accum() between forward-pass batches. */
    std::vector<uint32_t>              tri_illum_accum;         /* flat uint32 buffer  */
    int                                tri_illum_nb = 0;        /* n_bands used in buf */

    /* Camera-visibility wrappers for image accumulation entry points. */
    int                                camera_vis_mode = RT_CAM_VIS_AS_IS;
    int                                camera_transparency_mode = RT_CAM_TRANSPARENCY_BLOCK;
    int                                camera_depth_cull_enabled = 0;
    double                             camera_depth_cull_m = 0.0;
    uint64_t                           camera_full_march_steps = 0;
    uint64_t                           camera_full_march_context_entries = 0;

    FieldGrid*                         camera_field_grid = nullptr;
    int                                camera_field_grid_owned = 0;
    int                                camera_capture_strikes = 0;
    int                                camera_capture_max_strikes = 0;
    int                                camera_strike_stride_floats = 0;
    std::vector<float>                 camera_strike_rows;

    /* Sensor/film tensor uploads (Python→C++ ingress for camera simulation).
     * Stored verbatim and owned by tracer state; later kernels/GLSL bridges
     * can consume these buffers without touching Python memory. */
    std::vector<float>                 sensor_film_sensor_chunk;
    std::vector<float>                 sensor_film_film_chunk;
    std::vector<int32_t>               sensor_film_active_slots; /* flat [n_slots * 2] */
    int                                sensor_film_sensor_rows = 0;
    int                                sensor_film_sensor_stride = 0;
    int                                sensor_film_film_rows = 0;
    int                                sensor_film_film_stride = 0;
    int                                sensor_film_n_slots = 0;

    int                                profile_pulse_enabled = 0;
    double                             profile_pulse_period_s = 2.0;
    std::atomic<uint64_t>              profile_pulse_seq{0};
    std::atomic<uint64_t>              profile_pulse_next_ns{0};

    /* ── Optical assembly for camera sensor rays (Phase C) ──────────────────── */
    OpticalAssembly*                   optical_assembly = nullptr;
    int                                optical_assembly_owned = 0;
    int                                camera_mode = 0;

    /* ── Event telemetry for optical event tracking (Phase C) ────────────────── */
    struct {
        uint64_t rays_launched = 0;
        uint64_t rays_blocked_by_stop = 0;
        uint64_t rays_hit_lens_surface = 0;
        uint64_t rays_refracted = 0;
        uint64_t rays_reflected = 0;
        uint64_t rays_total_internal_reflection = 0;
        uint64_t rays_entered_wave_region = 0;
        uint64_t rays_exited_wave_region = 0;
        uint64_t rays_deposited_sensor = 0;
        uint64_t rays_out_of_domain = 0;
        uint64_t rays_fell_back_to_full_solve = 0;
        double energy_in = 0.0;
        double energy_out = 0.0;
        double energy_absorbed = 0.0;
        double energy_blocked = 0.0;
        double mean_phase_error = 0.0;
        double mean_focus_error = 0.0;
    } camera_event_telemetry;
};

static inline uint64_t rt_steady_now_ns()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

static inline void rt_maybe_emit_pulse(
    RayTracerState& st,
    const char* phase,
    int src_idx,
    int ray_idx,
    int bounce_idx)
{
    if (!RT_ENABLE_PROFILE) return;
    if (!st.profile_pulse_enabled) return;

    const uint64_t now_ns = rt_steady_now_ns();
    const uint64_t due_ns = st.profile_pulse_next_ns.load(std::memory_order_relaxed);
    if (due_ns != 0 && now_ns < due_ns) return;

    const double period_s = std::max(0.05, st.profile_pulse_period_s);
    const uint64_t next_ns = now_ns + static_cast<uint64_t>(period_s * 1.0e9);
    st.profile_pulse_next_ns.store(next_ns, std::memory_order_relaxed);

    const uint64_t seq = st.profile_pulse_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        std::fprintf(stderr, "[rt-pulse] seq=%llu phase=%s src=%d ray=%d bounce=%d\n",
            static_cast<unsigned long long>(seq),
            phase ? phase : "trace",
            src_idx,
            ray_idx,
            bounce_idx);

    const char* table = ray_tracer_allocation_table(&st);
    if (table && table[0] != '\0') {
        std::fprintf(stderr, "%s\n", table);
    }
    std::fflush(stderr);
}

/* ── MatBuf accessors ─────────────────────────────────────────────────────────
 * Phase 2 unification: per-band physics is read from the flat MatBuf shared
 * with the GLSL backend.  Layout per row (12 float32):
 *   [0] center_hz   [1] bandwidth_hz   [2] reflectance_mag  [3] transmittance
 *   [4] diffuse_frac[5] emission       [6] reemission       [7] ior_real
 *   [8] ior_imag    [9..11] pad
 * Row stride per band: 12; per material: MAX_SPECTRAL_BANDS * 12 = 384.
 *
 * Complex reflectance derivation: amplitude is the authored `reflectance_mag`;
 * phase comes from Fresnel at normal incidence using complex IOR
 *   r_F = (1 - n_complex) / (1 + n_complex),  n_complex = ior_real + i·ior_imag
 *   r_used = mag · exp(i · arg(r_F))
 * This matches the GLSL inline derivation (Phase 2c) for both backends.
 */
static constexpr int MAT_BUF_BAND_STRIDE = 12;
static constexpr int MAT_BUF_MAT_STRIDE  = MAT_BUF_BAND_STRIDE * MAX_SPECTRAL_BANDS;

static inline const float* mat_band_record(const RayTracerState& st, int mat_idx, int b) {
    /* Bounds-safe: out-of-range mat_idx or b clamps to material 0 band 0. */
    if (mat_idx < 0 || mat_idx >= st.mat_n_mats) mat_idx = 0;
    if (b < 0 || b >= MAX_SPECTRAL_BANDS) b = 0;
    size_t off = static_cast<size_t>(mat_idx) * MAT_BUF_MAT_STRIDE
               + static_cast<size_t>(b)       * MAT_BUF_BAND_STRIDE;
    if (off + MAT_BUF_BAND_STRIDE > st.mat_buf.size()) {
        static const float zeros[MAT_BUF_BAND_STRIDE] = {0};
        return zeros;
    }
    return st.mat_buf.data() + off;
}

static inline cd mat_refl_complex(const RayTracerState& st, int mat_idx, int b) {
    const float* r = mat_band_record(st, mat_idx, b);
    double mag  = static_cast<double>(r[2]);
    double n_re = static_cast<double>(r[7]);
    double n_im = static_cast<double>(r[8]);
    cd n_complex(n_re, n_im);
    cd one(1.0, 0.0);
    cd r_fresnel = (one - n_complex) / (one + n_complex);
    double phase = (std::abs(r_fresnel) > 1e-12) ? std::arg(r_fresnel) : 0.0;
    return cd(mag * std::cos(phase), mag * std::sin(phase));
}

static inline double mat_diffusion(const RayTracerState& st, int mat_idx, int b = 0) {
    return static_cast<double>(mat_band_record(st, mat_idx, b)[4]);
}

static inline double mat_n_real(const RayTracerState& st, int mat_idx, int b = 0) {
    return static_cast<double>(mat_band_record(st, mat_idx, b)[7]);
}

static inline double mat_n_imag(const RayTracerState& st, int mat_idx, int b = 0) {
    return static_cast<double>(mat_band_record(st, mat_idx, b)[8]);
}

static inline double mat_transmittance(const RayTracerState& st, int mat_idx, int b = 0) {
    return static_cast<double>(mat_band_record(st, mat_idx, b)[3]);
}

static inline double medium_n_real(const RayTracerState& st, int medium_mat_idx, int b = 0) {
    if (medium_mat_idx < 0) return 1.0;
    const double n = mat_n_real(st, medium_mat_idx, b);
    return (n > EPS) ? n : 1.0;
}

static inline bool tri_boundary_media(
    const Triangle& tri,
    bool front_face,
    int& medium_from_mat_idx,
    int& medium_to_mat_idx)
{
    const bool explicit_pair = (tri.medium_pos_mat_idx != tri.medium_neg_mat_idx);
    if (!explicit_pair) return false;

    if (front_face) {
        /* incoming from +normal side, crossing to -normal side */
        medium_from_mat_idx = tri.medium_pos_mat_idx;
        medium_to_mat_idx   = tri.medium_neg_mat_idx;
    } else {
        /* incoming from -normal side, crossing to +normal side */
        medium_from_mat_idx = tri.medium_neg_mat_idx;
        medium_to_mat_idx   = tri.medium_pos_mat_idx;
    }
    return true;
}


/* Per-material Stokes shift (Hz, positive = red-shift). Stored in band-0
 * pad slot [9] of the MatBuf record by `MaterialDatabase.build_mat_buf()`
 * and `ray_tracer_bridge.per_tri_spectral_to_mat_buf(...)`. Returns 0.0 for
 * non-reactive materials. Used together with `MAT_FLAG_REACTIVE` to drive
 * the in-line band-shift performed by the trace loop on fluorescent hits. */
static inline double mat_reactive_shift_hz(const RayTracerState& st, int mat_idx) {
    return static_cast<double>(mat_band_record(st, mat_idx, 0)[9]);
}

/* Apply a single fluorescent re-emission step to a per-band amplitude vector.
 *
 * For each band b the energy is moved to the band whose center frequency is
 * closest to (freq_hz[b] - shift_hz).  Bands without a valid downshift target
 * (those that land below the lowest band) are absorbed (energy lost — matches
 * the GLSL `react_freq = max(packet.x - stokes_shift, 20.0)` floor).  The
 * caller controls the fraction of energy that takes the shifted path via
 * `yield_frac` (typically the reemission coefficient at the hit band); the
 * remainder is left in `amp` unchanged so the primary specular/diffuse bounce
 * still proceeds with the surviving energy.
 *
 * This is the C++-side analogue of GLSL's `append_pending_ray` / PASS_REACTIVE
 * pair: rather than spawning a separate ray, we collapse the secondary into a
 * spectral redistribution carried by the same path.  Less faithful to a full
 * fluorescence kernel, but cheap and consistent with how the C++ tracer
 * already represents per-ray energy as `amp[n_bands]`. */
static inline void apply_reactive_shift(
        VXcd&                       amp,
        const Eigen::VectorXd&      freq_hz,
        double                      shift_hz,
        double                      yield_frac)
{
    const int nb = static_cast<int>(amp.size());
    if (nb <= 1 || shift_hz <= 0.0 || yield_frac <= 0.0) return;
    const double y = std::min(1.0, yield_frac);

    VXcd shifted = VXcd::Zero(nb);
    for (int b = 0; b < nb; ++b) {
        const double f_target = freq_hz[b] - shift_hz;
        if (f_target <= 0.0) continue;
        int best = -1;
        double best_d = 1e300;
        for (int j = 0; j < nb; ++j) {
            const double d = std::abs(freq_hz[j] - f_target);
            if (d < best_d) { best_d = d; best = j; }
        }
        if (best < 0) continue;
        shifted[best] += amp[b] * y;
    }
    for (int b = 0; b < nb; ++b)
        amp[b] = amp[b] * (1.0 - y) + shifted[b];
}

/* ── MatSpectralCache population and fast accessors ─────────────────────── */

/* Populate / refresh the MatSpectralCache from the current mat_buf.
 * Must be called after ray_tracer_create() and after any mat_buf update. */
static void rt_build_mat_cache(RayTracerState& st)
{
    MatSpectralCache& c = st.mat_cache;
    const int nm = st.mat_n_mats;
    const int nb = st.n_bands;
    c.n_mats  = nm;
    c.n_bands = nb;
    c.refl_re.resize(static_cast<size_t>(nm * nb));
    c.refl_im.resize(static_cast<size_t>(nm * nb));
    c.diffusion.assign(static_cast<size_t>(nm), 0.f);
    c.reactive_shift_hz.assign(static_cast<size_t>(nm), 0.0);
    c.reemit_yield.assign(static_cast<size_t>(nm), 0.f);
    c.reactive_dst_band.assign(static_cast<size_t>(nm * nb), -1);

    for (int m = 0; m < nm; ++m) {
        for (int b = 0; b < nb; ++b) {
            cd r = mat_refl_complex(st, m, b);
            c.refl_re[m * nb + b] = static_cast<float>(r.real());
            c.refl_im[m * nb + b] = static_cast<float>(r.imag());
        }
        c.diffusion[m]         = static_cast<float>(mat_diffusion(st, m, 0));
        c.reactive_shift_hz[m] = mat_reactive_shift_hz(st, m);
        c.reemit_yield[m]      = mat_band_record(st, m, 0)[6];
        const double shift_hz  = c.reactive_shift_hz[m];
        if (shift_hz > 0.0 && nb > 1) {
            for (int b = 0; b < nb; ++b) {
                double f_target = st.freq_hz_vec[b] - shift_hz;
                if (f_target <= 0.0) { c.reactive_dst_band[m * nb + b] = -1; continue; }
                int    best  = 0;
                double bd    = std::abs(st.freq_hz_vec[0] - f_target);
                for (int j = 1; j < nb; ++j) {
                    double d = std::abs(st.freq_hz_vec[j] - f_target);
                    if (d < bd) { bd = d; best = j; }
                }
                c.reactive_dst_band[m * nb + b] = best;
            }
        }
    }
}

/* Populate / refresh the RtMaterialSurfaceCache from pbr_chunk / enamel_chunk /
 * tex_stack_chunk.  Safe to call when chunks are empty (produces defaults). */
static void rt_build_surf_cache(RayTracerState& st)
{
    RtMaterialSurfaceCache& c = st.surf_cache;
    const int nm = st.mat_n_mats;
    c.n_mats = nm;
    if (nm <= 0) return;

    c.roughness.assign(nm, 0.5f);
    c.metallic.assign(nm, 0.0f);
    c.transmission.assign(nm, 0.0f);
    c.opacity.assign(nm, 1.0f);
    c.albedo.assign(nm * 3, 0.5f);
    c.f0.assign(nm * 3, 0.04f);
    c.enamel_thickness_nm.assign(nm, 0.0f);
    c.enamel_ior_real.assign(nm, 1.5f);
    c.enamel_ior_imag.assign(nm, 0.0f);
    c.enamel_roughness.assign(nm, 0.0f);
    c.enamel_tint.assign(nm * 3, 1.0f);
    c.profile_id.assign(nm, 0u);

    const bool have_pbr    = (int)st.pbr_chunk.size()       >= nm * 16;
    const bool have_enamel = (int)st.enamel_chunk.size()    >= nm * 8;
    const bool have_tex    = (int)st.tex_stack_chunk.size() >= nm * 16;

    for (int m = 0; m < nm; ++m) {
        if (have_pbr) {
            const float* p = st.pbr_chunk.data() + m * 16;
            c.albedo[m*3+0]   = p[0]; c.albedo[m*3+1] = p[1]; c.albedo[m*3+2] = p[2];
            c.roughness[m]    = p[3];
            c.metallic[m]     = p[4];
            c.transmission[m] = p[5];
            const float ior   = (p[6] > 0.f) ? p[6] : 1.5f;
            c.opacity[m]      = p[7];
            /* Schlick F0 = mix(vec3(ior_f0), albedo, metallic) */
            const float denom = ior + 1.f;
            const float ior_f0 = (ior - 1.f) / denom * ((ior - 1.f) / denom);
            const float ml = c.metallic[m];
            for (int ch = 0; ch < 3; ++ch)
                c.f0[m*3+ch] = ior_f0 * (1.f - ml) + c.albedo[m*3+ch] * ml;
        }
        if (have_enamel) {
            const float* e = st.enamel_chunk.data() + m * 8;
            c.enamel_thickness_nm[m] = e[0];
            c.enamel_ior_real[m]     = e[1];
            c.enamel_ior_imag[m]     = e[2];
            c.enamel_roughness[m]    = e[3];
            c.enamel_tint[m*3+0]     = e[4];
            c.enamel_tint[m*3+1]     = e[5];
            c.enamel_tint[m*3+2]     = e[6];
        }
        if (have_tex) {
            const float* t = st.tex_stack_chunk.data() + m * 16;
            c.profile_id[m] = static_cast<uint32_t>(t[11]);
        }
    }
}

/* Hot-path accessor — avoids mat_band_record bounds check + Fresnel math. */
static inline cd mat_cache_refl(const MatSpectralCache& c, int m, int b)
{
    return cd(c.refl_re[m * c.n_bands + b], c.refl_im[m * c.n_bands + b]);
}

static inline float mat_cache_diffusion(const MatSpectralCache& c, int m)
{
    return c.diffusion[m];
}

static inline float surf_cache_ggx_alpha(const RtMaterialSurfaceCache& c, int m)
{
    if (m < 0 || m >= c.n_mats || c.roughness.empty()) return 0.0f;
    return std::max(0.0f, std::min(1.0f, c.roughness[static_cast<size_t>(m)]));
}

static inline double ggx_D(double alpha, double NoH)
{
    const double a = std::max(1.0e-4, alpha);
    const double a2 = a * a;
    const double d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (M_PI * d * d);
}

static inline double ggx_G1(double alpha, double NoV)
{
    const double a = std::max(1.0e-4, alpha);
    const double a2 = a * a;
    return (NoV > 0.0)
        ? (2.0 * NoV) / (NoV + std::sqrt(a2 + (1.0 - a2) * NoV * NoV))
        : 0.0;
}

static inline double ggx_pdf_solid_angle(const V3d& n, const V3d& in_dir,
                                         const V3d& out_dir, double alpha)
{
    const V3d V = (-in_dir).normalized();
    const V3d L = out_dir.normalized();
    const double NoV = std::max(0.0, n.dot(V));
    const double NoL = std::max(0.0, n.dot(L));
    if (NoV <= 0.0 || NoL <= 0.0) return 0.0;
    V3d H = V + L;
    const double h2 = H.squaredNorm();
    if (h2 <= 1.0e-20) return 0.0;
    H /= std::sqrt(h2);
    const double NoH = std::max(0.0, n.dot(H));
    const double VoH = std::max(0.0, V.dot(H));
    if (NoH <= 0.0 || VoH <= 1.0e-12) return 0.0;
    return ggx_D(alpha, NoH) * ggx_G1(alpha, NoV) / (4.0 * NoV);
}

static inline V3d ggx_sample_reflection(const V3d& n, const V3d& in_dir,
                                        double alpha, double u1, double u2)
{
    const double a = std::max(1.0e-4, alpha);
    const V3d up = (std::abs(n.z()) < 0.999) ? V3d(0.0, 0.0, 1.0) : V3d(1.0, 0.0, 0.0);
    const V3d t = up.cross(n).normalized();
    const V3d b = n.cross(t);
    const V3d V = (-in_dir).normalized();
    const V3d Vlocal(t.dot(V), b.dot(V), n.dot(V));
    if (Vlocal.z() <= 1.0e-8) {
        return (in_dir - 2.0 * in_dir.dot(n) * n).normalized();
    }

    V3d Vh(a * Vlocal.x(), a * Vlocal.y(), Vlocal.z());
    Vh.normalize();
    const double lensq = Vh.x() * Vh.x() + Vh.y() * Vh.y();
    V3d T1 = (lensq > 1.0e-20)
        ? V3d(-Vh.y(), Vh.x(), 0.0) / std::sqrt(lensq)
        : V3d(1.0, 0.0, 0.0);
    const V3d T2 = Vh.cross(T1);

    const double r = std::sqrt(std::max(0.0, u1));
    const double phi = 2.0 * M_PI * u2;
    const double t1 = r * std::cos(phi);
    double t2 = r * std::sin(phi);
    const double s = 0.5 * (1.0 + Vh.z());
    t2 = (1.0 - s) * std::sqrt(std::max(0.0, 1.0 - t1 * t1)) + s * t2;
    V3d Nh = t1 * T1 + t2 * T2
           + std::sqrt(std::max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    V3d Hlocal(a * Nh.x(), a * Nh.y(), std::max(0.0, Nh.z()));
    Hlocal.normalize();
    V3d H = (t * Hlocal.x() + b * Hlocal.y() + n * Hlocal.z()).normalized();
    if (H.dot(V) < 0.0) H = -H;
    return (in_dir - 2.0 * in_dir.dot(H) * H).normalized();
}

/* O(n_bands) reactive shift using precomputed destination band indices. */
static inline void apply_reactive_shift_cached(VXcd& amp, const MatSpectralCache& c, int m)
{
    const int nb = c.n_bands;
    if (nb <= 0) return;
    const double shift_hz = c.reactive_shift_hz[m];
    if (shift_hz <= 0.0) return;
    const float  yf = c.reemit_yield[m];
    if (yf <= 0.f) return;
    const double y    = std::min(1.0, static_cast<double>(yf));
    const int    base = m * nb;
    VXcd shifted = VXcd::Zero(nb);
    for (int b = 0; b < nb; ++b) {
        const int dst = c.reactive_dst_band[base + b];
        if (dst >= 0) shifted[dst] += amp[b] * y;
    }
    for (int b = 0; b < nb; ++b)
        amp[b] = amp[b] * (1.0 - y) + shifted[b];
}

/* Forward declaration for context-dispatch hook used by full-march tracking. */
static inline uint32_t dispatch_scale_context_entry(
    const RayTracerState& st,
    const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len = nullptr);

static inline bool tri_bary_uv(const Triangle& tri, const V3d& p,
                               double& u, double& v)
{
    V3d q = p - tri.v0;
    double d00 = tri.edge1.dot(tri.edge1);
    double d01 = tri.edge1.dot(tri.edge2);
    double d11 = tri.edge2.dot(tri.edge2);
    double d20 = q.dot(tri.edge1);
    double d21 = q.dot(tri.edge2);
    double den = d00 * d11 - d01 * d01;
    if (std::abs(den) <= EPS) {
        u = 0.0; v = 0.0;
        return false;
    }
    u = (d11 * d20 - d01 * d21) / den;
    v = (d00 * d21 - d01 * d20) / den;
    return true;
}

static inline bool tri_param_kind_is_sdf(const int kind);
static inline bool tri_requires_optical_sdf(const Triangle& tri);

static inline bool apply_parametric_surface_point(
    const RayTracerState& st,
    int tri_id,
    const V3d& hit_pos,
    V3d& out_pos,
    V3d& out_normal)
{
    out_pos = hit_pos;
    if (tri_id < 0 || tri_id >= static_cast<int>(st.tris.size()))
        return false;

    const Triangle& tri = st.tris[static_cast<size_t>(tri_id)];
    out_normal = tri.normal;

    if (tri_id >= static_cast<int>(st.tri_param_group_of_tri.size()))
        return false;
    int gid = st.tri_param_group_of_tri[static_cast<size_t>(tri_id)];
    if (gid < 0 || gid >= static_cast<int>(st.tri_group_parametric_kind.size()))
        return false;

    int kind = st.tri_group_parametric_kind[static_cast<size_t>(gid)];
    if (tri_requires_optical_sdf(tri) && !tri_param_kind_is_sdf(kind))
        return false;
    if (kind == TRI_PARAM_SURFACE_NONE)
        return false;

    const auto& payload = st.tri_group_parametric_payload[static_cast<size_t>(gid)];
    double u = 0.0, v = 0.0;
    tri_bary_uv(tri, hit_pos, u, v);
    const V3d t1 = tri.edge1.normalized();
    const V3d t2 = tri.edge2.normalized();

    double delta = 0.0;
    double dzdu = 0.0;
    double dzdv = 0.0;

    if (kind == TRI_PARAM_SURFACE_POLY_BARY) {
        if (payload.size() < sizeof(double) * 6)
            return false;
        const double* c = reinterpret_cast<const double*>(payload.data());
        delta = c[0] + c[1] * u + c[2] * v
              + c[3] * u * u + c[4] * u * v + c[5] * v * v;
        dzdu = c[1] + 2.0 * c[3] * u + c[4] * v;
        dzdv = c[2] + c[4] * u + 2.0 * c[5] * v;
    } else if (kind == TRI_PARAM_SURFACE_SDF_SADDLE) {
        // Payload: float64[2] [amplitude_m, neighborhood_margin_uv].
        const double amp = (payload.size() >= sizeof(double))
            ? *reinterpret_cast<const double*>(payload.data())
            : 2.0e-3;
        const double margin = (payload.size() >= sizeof(double) * 2)
            ? *(reinterpret_cast<const double*>(payload.data()) + 1)
            : 8.0e-2;

        auto eval_saddle = [&](double uu, double vv, double& d, double& duv, double& dvv) {
            const double cu = uu - (1.0 / 3.0);
            const double cv = vv - (1.0 / 3.0);
            d = amp * (cu * cu - cv * cv);
            duv = 2.0 * amp * cu;
            dvv = -2.0 * amp * cv;
        };

        if (margin <= EPS) {
            eval_saddle(u, v, delta, dzdu, dzdv);
        } else {
            // Post-hit neighborhood integration over candidate impact points.
            static const double kOff[9][2] = {
                { 0.0,  0.0},
                {-1.0,  0.0}, { 1.0,  0.0},
                { 0.0, -1.0}, { 0.0,  1.0},
                {-0.70710678118, -0.70710678118},
                { 0.70710678118, -0.70710678118},
                {-0.70710678118,  0.70710678118},
                { 0.70710678118,  0.70710678118},
            };
            double d_acc = 0.0, du_acc = 0.0, dv_acc = 0.0, w_acc = 0.0;
            for (int i = 0; i < 9; ++i) {
                const double ou = kOff[i][0];
                const double ov = kOff[i][1];
                const double uu = u + margin * ou;
                const double vv = v + margin * ov;
                const double r2 = ou * ou + ov * ov;
                const double w = 1.0 / (1.0 + r2);
                double d_i = 0.0, du_i = 0.0, dv_i = 0.0;
                eval_saddle(uu, vv, d_i, du_i, dv_i);
                d_acc += w * d_i;
                du_acc += w * du_i;
                dv_acc += w * dv_i;
                w_acc += w;
            }
            const double inv_w = (w_acc > EPS) ? (1.0 / w_acc) : 1.0;
            delta = d_acc * inv_w;
            dzdu = du_acc * inv_w;
            dzdv = dv_acc * inv_w;
        }
    } else if (kind == TRI_PARAM_SURFACE_SDF_SPHERE) {
        // Payload: float64[2] [radius_m, neighborhood_margin_uv].
        double radius = (payload.size() >= sizeof(double))
            ? *reinterpret_cast<const double*>(payload.data())
            : 0.12;
        const double margin = (payload.size() >= sizeof(double) * 2)
            ? *(reinterpret_cast<const double*>(payload.data()) + 1)
            : 8.0e-2;
        if (std::abs(radius) < EPS) radius = 0.12;
        const double k = 0.5 / radius;

        auto eval_sphere = [&](double uu, double vv, double& d, double& duv, double& dvv) {
            const double cu = uu - (1.0 / 3.0);
            const double cv = vv - (1.0 / 3.0);
            d = k * (cu * cu + cv * cv);
            duv = 2.0 * k * cu;
            dvv = 2.0 * k * cv;
        };

        if (margin <= EPS) {
            eval_sphere(u, v, delta, dzdu, dzdv);
        } else {
            static const double kOff[9][2] = {
                { 0.0,  0.0},
                {-1.0,  0.0}, { 1.0,  0.0},
                { 0.0, -1.0}, { 0.0,  1.0},
                {-0.70710678118, -0.70710678118},
                { 0.70710678118, -0.70710678118},
                {-0.70710678118,  0.70710678118},
                { 0.70710678118,  0.70710678118},
            };
            double d_acc = 0.0, du_acc = 0.0, dv_acc = 0.0, w_acc = 0.0;
            for (int i = 0; i < 9; ++i) {
                const double ou = kOff[i][0];
                const double ov = kOff[i][1];
                const double uu = u + margin * ou;
                const double vv = v + margin * ov;
                const double r2 = ou * ou + ov * ov;
                const double w = 1.0 / (1.0 + r2);
                double d_i = 0.0, du_i = 0.0, dv_i = 0.0;
                eval_sphere(uu, vv, d_i, du_i, dv_i);
                d_acc += w * d_i;
                du_acc += w * du_i;
                dv_acc += w * dv_i;
                w_acc += w;
            }
            const double inv_w = (w_acc > EPS) ? (1.0 / w_acc) : 1.0;
            delta = d_acc * inv_w;
            dzdu = du_acc * inv_w;
            dzdv = dv_acc * inv_w;
        }
    } else if (kind == TRI_PARAM_SURFACE_NEURAL_ASSEMBLY) {
        /* Exact conic surface refinement.
         * Payload is float32; surface geometry in reserved header bytes:
         *   p[11]=ROC  p[12]=conic_k  p[13]=axis_index  p[14]=r_out  p[15]=side
         * axis_index: 0=X(scene), 2=Z(lens-design default). */
        if (payload.size() < 16 * sizeof(float)) return false;
        const float* pf = reinterpret_cast<const float*>(payload.data());
        const double ROC    = (double)pf[11];
        const double k_con  = (double)pf[12];
        const int    ax_i   = (int)pf[13];   /* optical axis index */
        const double r_out  = (double)pf[14];

        /* Transverse component indices. */
        const int ax = (ax_i >= 0 && ax_i <= 2) ? ax_i : 2;
        const int t0 = (ax == 0) ? 1 : 0;
        const int t1 = (ax == 2) ? 1 : 2;

        const double pt0 = hit_pos[t0], pt1 = hit_pos[t1];
        const double r2  = pt0*pt0 + pt1*pt1;
        const double r   = std::sqrt(r2);
        if (r_out > EPS && r > r_out) return false;

        if (std::abs(ROC) < EPS) {
            out_pos    = hit_pos;
            out_normal = tri.normal;
            return true;
        }

        const double c  = 1.0 / ROC;
        const double c2 = c * c;
        const double disc = 1.0 - (1.0 + k_con) * c2 * r2;
        if (disc < EPS) { out_pos = hit_pos; out_normal = tri.normal; return true; }
        const double sq   = std::sqrt(disc);
        delta             = c * r2 / (1.0 + sq);
        double d0 = 0.0, d1 = 0.0;  /* dz/dt0, dz/dt1 */
        if (r > EPS) {
            const double dsdr = c * r / sq;
            d0 = dsdr * (pt0 / r);
            d1 = dsdr * (pt1 / r);
        }
        /* Displace along optical axis; normal from Jacobian. */
        V3d np = hit_pos;
        np[ax] += delta;
        out_pos = np;
        V3d raw_n;
        raw_n[ax] = 1.0;
        raw_n[t0] = -d0;
        raw_n[t1] = -d1;
        if (raw_n.norm() > EPS) raw_n.normalize();
        else raw_n = tri.normal;
        if (raw_n.dot(tri.normal) < 0.0) raw_n = -raw_n;
        out_normal = raw_n;
        return true;
    } else {
        return false;
    }

    /* Jacobian normal for displaced parametric surface:
     *   S(u,v) = P(u,v) + N0 * delta(u,v)
     * with P(u,v)=v0+u*edge1+v*edge2 and constant base normal N0.
     * Then:
     *   Su = edge1 + N0 * d(delta)/du
     *   Sv = edge2 + N0 * d(delta)/dv
     *   N  = normalize(Su x Sv)
     */
    const V3d Su = tri.edge1 + tri.normal * dzdu;
    const V3d Sv = tri.edge2 + tri.normal * dzdv;
    V3d warped_n = Su.cross(Sv);
    if (warped_n.norm() < EPS) {
        /* Degenerate Jacobian fallback to first-order gradient frame. */
        warped_n = (tri.normal - dzdu * t1 - dzdv * t2);
    }
    if (warped_n.norm() < EPS)
        warped_n = tri.normal;
    else
        warped_n.normalize();

    out_normal = warped_n;
    out_pos = hit_pos + warped_n * delta;
    return true;
}

static inline bool tri_material_is_transmissive(const RayTracerState& st, const Triangle& tri, int band = 0)
{
    return (mat_transmittance(st, tri.mat_idx, band) > 1e-6)
        || (std::abs(mat_n_real(st, tri.mat_idx, band) - 1.0) > 1e-6);
}

static inline bool tri_param_kind_is_sdf(const int kind)
{
    return kind == TRI_PARAM_SURFACE_SDF_SADDLE || kind == TRI_PARAM_SURFACE_SDF_SPHERE;
}

/* Return the parametric payload pointer + size for a neural-assembly triangle,
 * or nullptr if the triangle does not belong to a NEURAL_ASSEMBLY group. */
static inline const float* tri_neural_assembly_payload(
    const RayTracerState& st, int tri_id, int* out_bytes)
{
    if (tri_id < 0 || (size_t)tri_id >= st.tri_param_group_of_tri.size()) return nullptr;
    int gid = st.tri_param_group_of_tri[(size_t)tri_id];
    if (gid < 0 || (size_t)gid >= st.tri_group_parametric_kind.size()) return nullptr;
    if (st.tri_group_parametric_kind[(size_t)gid] != TRI_PARAM_SURFACE_NEURAL_ASSEMBLY)
        return nullptr;
    const auto& pb = st.tri_group_parametric_payload[(size_t)gid];
    if (out_bytes) *out_bytes = (int)pb.size();
    return reinterpret_cast<const float*>(pb.data());
}

/* Return the manifold payload for the active parametric-lens tri-group.
 * NEURAL_ASSEMBLY/LUT payloads are intentionally not transport paths right now:
 * they consumed time before the basic reversible camera was complete.  If one
 * is encountered, return an absorber sentinel so it cannot masquerade as a
 * valid optical transform.
 *
 * Also returns a non-null sentinel for empty-payload parametric-lens
 * groups (interior absorbers): check *out_bytes == 0 to distinguish absorbers
 * from teleport payloads.  Returns nullptr when the triangle carries no
 * manifold group at all (SDF_SPHERE, POLY_BARY, or unregistered). */
static inline const float* tri_manifold_payload(
    const RayTracerState& st, int tri_id, int* out_bytes)
{
    if (tri_id < 0 || (size_t)tri_id >= st.tri_param_group_of_tri.size()) return nullptr;
    int gid = st.tri_param_group_of_tri[(size_t)tri_id];
    if (gid < 0 || (size_t)gid >= st.tri_group_parametric_kind.size()) return nullptr;
    const int kind = st.tri_group_parametric_kind[(size_t)gid];
    if (kind == TRI_PARAM_SURFACE_NEURAL_ASSEMBLY) {
        static const float _disabled_neural_sentinel = 0.0f;
        if (out_bytes) *out_bytes = 0;
        return &_disabled_neural_sentinel;
    }
    if (kind != TRI_PARAM_SURFACE_PARAMETRIC_LENS)
        return nullptr;
    const auto& pb = st.tri_group_parametric_payload[(size_t)gid];
    if (out_bytes) *out_bytes = (int)pb.size();
    if (pb.empty()) {
        static const float _absorb_sentinel = 0.0f;
        return &_absorb_sentinel;   /* non-null + out_bytes==0 → absorber */
    }
    return reinterpret_cast<const float*>(pb.data());
}

/* apply_neural_mlp_from_f32 -- deferred MLP transport.
 * Intention: restore only after the transform can report reversible optical
 * metadata: direction mapping, OPL, eta/cosines, Jacobian, and PDFs. */
static inline void apply_neural_mlp_from_f32(
    const RayTracerState& st,
    const float* p, int payload_bytes,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len);

struct ParametricLensTraceInfo {
    double geom_len = 0.0;
    double opl = 0.0;
    double phase_space_jacobian = 1.0;
    double first_cos_incident = 0.0;
    double last_cos_transmitted = 0.0;
    double first_eta_i = 1.0;
    double last_eta_t = 1.0;
    double throughput_multiplier = 1.0;
    int reason = BDPT_OPT_REFRACTION;
    std::vector<BdptOpticalEventRecord> events;
};

static inline bool apply_parametric_lens_from_f32(
    const RayTracerState& st,
    const float* p, int payload_bytes,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len,
    bool is_backward = false,
    ParametricLensTraceInfo* trace_info = nullptr);

static inline void apply_manifold_transfer_from_payload(
    const RayTracerState& st,
    const float* p, int payload_bytes,
    const V3d& ap_hit,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len);

static inline bool tri_requires_optical_sdf(const Triangle& tri)
{
    const int optical_mask =
        MAT_FLAG_REACTIVE |
        MAT_FLAG_EMISSIVE |
        MAT_FLAG_APERTURE_STOP;
    return (tri.flags & optical_mask) != 0;
}

static bool segment_first_hit(
    const RayTracerState& st,
    const V3d& orig,
    const V3d& dir,
    double t_cap,
    double& t_hit,
    int& hit_tri)
{
    t_hit = t_cap;
    hit_tri = -1;

    if (!st.bvh_nodes.empty()) {
        V3d inv_dir = dir.cwiseInverse();
        bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                  orig, dir, inv_dir, t_hit, hit_tri);
    } else {
        for (size_t ti = 0; ti < st.tris.size(); ++ti) {
            double t;
            if (ray_triangle_hit(orig, dir, st.tris[ti], t) && t < t_hit) {
                t_hit = t;
                hit_tri = static_cast<int>(ti);
            }
        }
    }
    return hit_tri >= 0;
}

/* Camera visibility test for image accumulation.
 * Policy is controlled by RayTracerState::camera_vis_mode and related fields. */
static bool camera_visible_to_point(
    RayTracerState& st,
    const V3d& cam_pos,
    const V3d& surf_pos)
{
    if (st.camera_vis_mode == RT_CAM_VIS_AS_IS)
        return true;

    V3d to = surf_pos - cam_pos;
    double dist = to.norm();
    if (dist <= EPS * 200.0)
        return true;

    V3d dir = to / dist;
    V3d orig = cam_pos + dir * (EPS * 100.0);
    double t_cap = dist - (EPS * 200.0);
    if (t_cap <= EPS)
        return true;

    if (st.camera_vis_mode == RT_CAM_VIS_DIRECT_HIT) {
        double t_hit = t_cap;
        int hit_tri = -1;
        if (!segment_first_hit(st, orig, dir, t_cap, t_hit, hit_tri))
            return true;
        return false;
    }

    if (st.camera_vis_mode == RT_CAM_VIS_FULL_MARCH) {
        static constexpr int MAX_MARCH_STEPS = 256;
        double remain = t_cap;
        V3d cur = orig;
        for (int step = 0; step < MAX_MARCH_STEPS && remain > EPS; ++step) {
            st.camera_full_march_steps++;

            double t_hit = remain;
            int hit_tri = -1;
            if (!segment_first_hit(st, cur, dir, remain, t_hit, hit_tri))
                return true;

            V3d hit_pos = cur + t_hit * dir;
            V3d hit_n = st.tris[static_cast<size_t>(hit_tri)].normal;
            apply_parametric_surface_point(st, hit_tri, hit_pos, hit_pos, hit_n);

            if (st.camera_field_grid) {
                float p[3] = {
                    static_cast<float>(hit_pos.x()),
                    static_cast<float>(hit_pos.y()),
                    static_cast<float>(hit_pos.z())
                };
                static thread_local std::vector<float> amp_re;
                static thread_local std::vector<float> amp_im;
                amp_re.assign(static_cast<size_t>(st.n_bands), 1.0f);
                amp_im.assign(static_cast<size_t>(st.n_bands), 0.0f);
                (void)field_grid_inject_amplitude_all_bands(
                    st.camera_field_grid,
                    p,
                    amp_re.data(),
                    amp_im.data(),
                    st.n_bands);
            }

            VXcd track_amp(1);
            track_amp[0] = cd(1.0, 0.0);
            for (const RtScaleContext& ctx : st.scale_contexts) {
                V3d ctr(ctx.center[0], ctx.center[1], ctx.center[2]);
                if ((hit_pos - ctr).squaredNorm() <= ctx.radius * ctx.radius) {
                    V3d p = hit_pos;
                    V3d d = dir;
                    (void)dispatch_scale_context_entry(st, ctx, p, d, track_amp);
                    st.camera_full_march_context_entries++;
                }
            }

            return false;

            const double advance = std::max(EPS * 200.0, (hit_pos - cur).norm() + EPS * 200.0);
            cur += dir * advance;
            remain -= std::min(remain, advance);
        }
        return remain <= EPS;
    }

    return true;
}

/* Returns true when a strike row CAN be appended (strikes enabled + not full).
 * Use as a pre-check before computing visibility solely for strike telemetry. */
static inline bool camera_strikes_will_accept(const RayTracerState& st)
{
    if (!st.camera_capture_strikes) return false;
    const int stride = (st.camera_strike_stride_floats > 0)
                     ? st.camera_strike_stride_floats
                     : (16 + 2 * st.n_bands);
    const int rows_now = static_cast<int>(st.camera_strike_rows.size() / stride);
    return st.camera_capture_max_strikes <= 0 || rows_now < st.camera_capture_max_strikes;
}

static inline void append_camera_strike(
    RayTracerState& st,
    int src_id,
    int bounce,
    int tri_id,
    const V3d& pos,
    const V3d& normal,
    const V3d& incoming_dir,
    double depth,
    double total_path,
    int visible,
    int context_entries,
    const VXcd& amp_prop)
{
    if (!st.camera_capture_strikes) return;
    if (st.camera_strike_stride_floats <= 0)
        st.camera_strike_stride_floats = 16 + 2 * st.n_bands;

    const int stride = st.camera_strike_stride_floats;
    const int rows_now = static_cast<int>(st.camera_strike_rows.size() / stride);
    if (st.camera_capture_max_strikes > 0 && rows_now >= st.camera_capture_max_strikes)
        return;

    const size_t base = st.camera_strike_rows.size();
    st.camera_strike_rows.resize(base + static_cast<size_t>(stride), 0.0f);
    float* row = st.camera_strike_rows.data() + base;

    row[0] = static_cast<float>(pos.x());
    row[1] = static_cast<float>(pos.y());
    row[2] = static_cast<float>(pos.z());
    row[3] = static_cast<float>(normal.x());
    row[4] = static_cast<float>(normal.y());
    row[5] = static_cast<float>(normal.z());
    row[6] = static_cast<float>(incoming_dir.x());
    row[7] = static_cast<float>(incoming_dir.y());
    row[8] = static_cast<float>(incoming_dir.z());
    row[9] = static_cast<float>(tri_id);
    row[10] = static_cast<float>(src_id);
    row[11] = static_cast<float>(bounce);
    row[12] = static_cast<float>(depth);
    row[13] = static_cast<float>(total_path);
    row[14] = static_cast<float>(visible);
    row[15] = static_cast<float>(context_entries);

    for (int b = 0; b < st.n_bands; ++b) {
        row[16 + 2 * b] = static_cast<float>(amp_prop[b].real());
        row[16 + 2 * b + 1] = static_cast<float>(amp_prop[b].imag());
    }
}

static inline void accumulate_field_capture(
    RayTracerState& st,
    const V3d& pos,
    const VXcd& amp_prop)
{
    if (!st.camera_field_grid) return;
    float p[3] = {
        static_cast<float>(pos.x()),
        static_cast<float>(pos.y()),
        static_cast<float>(pos.z())
    };
    static thread_local std::vector<float> amp_re;
    static thread_local std::vector<float> amp_im;
    amp_re.resize(static_cast<size_t>(st.n_bands));
    amp_im.resize(static_cast<size_t>(st.n_bands));
    for (int b = 0; b < st.n_bands; ++b) {
        amp_re[static_cast<size_t>(b)] = static_cast<float>(amp_prop[b].real());
        amp_im[static_cast<size_t>(b)] = static_cast<float>(amp_prop[b].imag());
    }
    (void)field_grid_inject_amplitude_all_bands(
        st.camera_field_grid,
        p,
        amp_re.data(),
        amp_im.data(),
        st.n_bands);
}

/* Deposit field continuously along a segment p0->p1 by sampling points and
 * injecting complex amplitude into the bound field grid. */
static inline void accumulate_field_capture_segment(
    RayTracerState& st,
    const V3d& p0,
    const V3d& p1,
    const VXcd& amp_prop)
{
    if (!st.camera_field_grid) return;
    const double seg_len = (p1 - p0).norm();
    if (!(seg_len > 0.0)) {
        accumulate_field_capture(st, p1, amp_prop);
        return;
    }

    /* Convert complex spectrum once per segment (not once per sample). */
    static thread_local std::vector<float> amp_re;
    static thread_local std::vector<float> amp_im;
    amp_re.resize(static_cast<size_t>(st.n_bands));
    amp_im.resize(static_cast<size_t>(st.n_bands));
    for (int b = 0; b < st.n_bands; ++b) {
        amp_re[static_cast<size_t>(b)] = static_cast<float>(amp_prop[b].real());
        amp_im[static_cast<size_t>(b)] = static_cast<float>(amp_prop[b].imag());
    }

    // Fast regular-grid path: Amanatides-Woo voxel traversal visits crossed
    // voxels once (O(n_voxels_crossed)) instead of fixed-distance sampling.
    int nx = 0, ny = 0, nz = 0;
    const int dims_rc = field_grid_regular_dims(st.camera_field_grid, &nx, &ny, &nz);
    const float* bmin = field_grid_bmin(st.camera_field_grid);
    const float* bmax = field_grid_bmax(st.camera_field_grid);
    if (dims_rc == SK_OK && bmin && bmax && nx > 0 && ny > 0 && nz > 0) {
        const double minx = static_cast<double>(bmin[0]);
        const double miny = static_cast<double>(bmin[1]);
        const double minz = static_cast<double>(bmin[2]);
        const double maxx = static_cast<double>(bmax[0]);
        const double maxy = static_cast<double>(bmax[1]);
        const double maxz = static_cast<double>(bmax[2]);
        const double dx = (maxx - minx) / static_cast<double>(nx);
        const double dy = (maxy - miny) / static_cast<double>(ny);
        const double dz = (maxz - minz) / static_cast<double>(nz);

        if (dx > 0.0 && dy > 0.0 && dz > 0.0) {
            const V3d d = p1 - p0;
            auto world_to_idx = [](double v, double vmin, double dv, int n) -> int {
                int i = static_cast<int>(std::floor((v - vmin) / dv));
                if (i < 0) i = 0;
                if (i >= n) i = n - 1;
                return i;
            };

            int ix = world_to_idx(p0.x(), minx, dx, nx);
            int iy = world_to_idx(p0.y(), miny, dy, ny);
            int iz = world_to_idx(p0.z(), minz, dz, nz);

            const int stepx = (d.x() > 0.0) ? 1 : ((d.x() < 0.0) ? -1 : 0);
            const int stepy = (d.y() > 0.0) ? 1 : ((d.y() < 0.0) ? -1 : 0);
            const int stepz = (d.z() > 0.0) ? 1 : ((d.z() < 0.0) ? -1 : 0);

            const double inf = std::numeric_limits<double>::infinity();
            auto t_delta = [&](double dd, double dv) -> double {
                return (std::abs(dd) > 1.0e-15) ? std::abs(dv / dd) : inf;
            };
            auto next_boundary_t = [&](double p, double vmin, double dv, int i, int step, double dd) -> double {
                if (step == 0 || std::abs(dd) <= 1.0e-15) return inf;
                double boundary = vmin + ((step > 0 ? (i + 1) : i) * dv);
                return (boundary - p) / dd;
            };

            double tx = next_boundary_t(p0.x(), minx, dx, ix, stepx, d.x());
            double ty = next_boundary_t(p0.y(), miny, dy, iy, stepy, d.y());
            double tz = next_boundary_t(p0.z(), minz, dz, iz, stepz, d.z());
            const double dtx = t_delta(d.x(), dx);
            const double dty = t_delta(d.y(), dy);
            const double dtz = t_delta(d.z(), dz);

            double t_prev = 0.0;
            int guard = 0;
            const int guard_max = nx + ny + nz + 1024;
            while (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz && t_prev <= 1.0 && guard < guard_max) {
                double t_next = std::min(tx, std::min(ty, tz));
                if (!std::isfinite(t_next)) t_next = 1.0;
                t_next = std::max(t_prev, std::min(1.0, t_next));
                const double t_mid = 0.5 * (t_prev + t_next);
                const V3d pm = p0 + d * t_mid;
                float pf[3] = {
                    static_cast<float>(pm.x()),
                    static_cast<float>(pm.y()),
                    static_cast<float>(pm.z())
                };
                (void)field_grid_inject_amplitude_all_bands(
                    st.camera_field_grid,
                    pf,
                    amp_re.data(),
                    amp_im.data(),
                    st.n_bands);

                if (t_next >= 1.0) break;
                if (tx <= ty && tx <= tz) {
                    ix += stepx;
                    tx += dtx;
                } else if (ty <= tx && ty <= tz) {
                    iy += stepy;
                    ty += dty;
                } else {
                    iz += stepz;
                    tz += dtz;
                }
                t_prev = t_next;
                ++guard;
            }
            return;
        }
    }

    // Fallback for non-regular grids: bounded arc-length sampling.
    double cell_step = 0.015;
    int n_steps = static_cast<int>(std::ceil(seg_len / cell_step));
    n_steps = std::max(1, std::min(256, n_steps));
    for (int i = 0; i < n_steps; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(n_steps);
        const V3d p = p0 + (p1 - p0) * t;
        float pf[3] = {
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z())
        };
        (void)field_grid_inject_amplitude_all_bands(
            st.camera_field_grid,
            pf,
            amp_re.data(),
            amp_im.data(),
            st.n_bands);
    }
}

static bool accumulate_field_capture_segments_regular_threaded(
    RayTracerState& st,
    const float* hbuf,
    int n_hits,
    int hit_stride,
    int n_bands)
{
    if (!st.camera_field_grid || !hbuf || n_hits <= 0 || n_bands <= 0) return false;

    int nx = 0, ny = 0, nz = 0;
    const int dims_rc = field_grid_regular_dims(st.camera_field_grid, &nx, &ny, &nz);
    const float* bmin = field_grid_bmin(st.camera_field_grid);
    const float* bmax = field_grid_bmax(st.camera_field_grid);
    float* data = field_grid_data_re_im(st.camera_field_grid);
    if (dims_rc != SK_OK || !bmin || !bmax || !data || nx <= 0 || ny <= 0 || nz <= 0)
        return false;

    const int bands = std::min(std::min(n_bands, st.n_bands), 16);
    if (bands <= 0) return true;

    const double minx = static_cast<double>(bmin[0]);
    const double miny = static_cast<double>(bmin[1]);
    const double minz = static_cast<double>(bmin[2]);
    const double maxx = static_cast<double>(bmax[0]);
    const double maxy = static_cast<double>(bmax[1]);
    const double maxz = static_cast<double>(bmax[2]);
    const double dx = (maxx - minx) / static_cast<double>(nx);
    const double dy = (maxy - miny) / static_cast<double>(ny);
    const double dz = (maxz - minz) / static_cast<double>(nz);
    if (!(dx > 0.0 && dy > 0.0 && dz > 0.0)) return false;

    const int64_t n_cells = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) * static_cast<int64_t>(nz);
    static constexpr size_t N_LOCKS = 4096;
    std::vector<std::mutex> locks(N_LOCKS);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int n_threads = std::max(1, std::min<int>((int)hw, (n_hits + 1023) / 1024));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_threads));

    auto world_to_idx = [](double v, double vmin, double dv, int n) -> int {
        int i = static_cast<int>(std::floor((v - vmin) / dv));
        if (i < 0) i = 0;
        if (i >= n) i = n - 1;
        return i;
    };

    auto add_trilinear = [&](double px, double py, double pz, const float* row) {
        const double ux = (px - minx) / (maxx - minx) * static_cast<double>(nx - 1);
        const double uy = (py - miny) / (maxy - miny) * static_cast<double>(ny - 1);
        const double uz = (pz - minz) / (maxz - minz) * static_cast<double>(nz - 1);
        if (ux < 0.0 || uy < 0.0 || uz < 0.0 || ux >= nx || uy >= ny || uz >= nz)
            return;

        const int x0 = static_cast<int>(ux);
        const int y0 = static_cast<int>(uy);
        const int z0 = static_cast<int>(uz);
        const int x1 = (x0 + 1 < nx) ? x0 + 1 : x0;
        const int y1 = (y0 + 1 < ny) ? y0 + 1 : y0;
        const int z1 = (z0 + 1 < nz) ? z0 + 1 : z0;
        const float fx = static_cast<float>(ux - x0);
        const float fy = static_cast<float>(uy - y0);
        const float fz = static_cast<float>(uz - z0);

        auto add_cell = [&](int x, int y, int z, float w) {
            if (w <= 0.0f) return;
            const int64_t cell = (static_cast<int64_t>(z) * ny + y) * nx + x;
            std::lock_guard<std::mutex> lk(locks[static_cast<size_t>(cell) & (N_LOCKS - 1)]);
            for (int b = 0; b < bands; ++b) {
                const int64_t off = 2 * (static_cast<int64_t>(b) * n_cells + cell);
                data[off + 0] += row[26 + b] * w;
                data[off + 1] += row[42 + b] * w;
            }
        };

        add_cell(x0, y0, z0, (1-fx)*(1-fy)*(1-fz));
        add_cell(x1, y0, z0,    fx *(1-fy)*(1-fz));
        add_cell(x0, y1, z0, (1-fx)*   fy *(1-fz));
        add_cell(x1, y1, z0,    fx *   fy *(1-fz));
        add_cell(x0, y0, z1, (1-fx)*(1-fy)*   fz );
        add_cell(x1, y0, z1,    fx *(1-fy)*   fz );
        add_cell(x0, y1, z1, (1-fx)*   fy *   fz );
        add_cell(x1, y1, z1,    fx *   fy *   fz );
    };

    auto worker = [&](int lo, int hi) {
        const double inf = std::numeric_limits<double>::infinity();
        for (int i = lo; i < hi; ++i) {
            const float* row = hbuf + static_cast<size_t>(i) * hit_stride;
            const V3d p0(row[9], row[10], row[11]);
            const V3d p1(row[0], row[1], row[2]);
            const V3d d = p1 - p0;

            int ix = world_to_idx(p0.x(), minx, dx, nx);
            int iy = world_to_idx(p0.y(), miny, dy, ny);
            int iz = world_to_idx(p0.z(), minz, dz, nz);
            const int stepx = (d.x() > 0.0) ? 1 : ((d.x() < 0.0) ? -1 : 0);
            const int stepy = (d.y() > 0.0) ? 1 : ((d.y() < 0.0) ? -1 : 0);
            const int stepz = (d.z() > 0.0) ? 1 : ((d.z() < 0.0) ? -1 : 0);

            auto t_delta = [&](double dd, double dv) -> double {
                return (std::abs(dd) > 1.0e-15) ? std::abs(dv / dd) : inf;
            };
            auto next_boundary_t = [&](double p, double vmin, double dv, int idx, int step, double dd) -> double {
                if (step == 0 || std::abs(dd) <= 1.0e-15) return inf;
                const double boundary = vmin + ((step > 0 ? (idx + 1) : idx) * dv);
                return (boundary - p) / dd;
            };

            double tx = next_boundary_t(p0.x(), minx, dx, ix, stepx, d.x());
            double ty = next_boundary_t(p0.y(), miny, dy, iy, stepy, d.y());
            double tz = next_boundary_t(p0.z(), minz, dz, iz, stepz, d.z());
            const double dtx = t_delta(d.x(), dx);
            const double dty = t_delta(d.y(), dy);
            const double dtz = t_delta(d.z(), dz);

            double t_prev = 0.0;
            int guard = 0;
            const int guard_max = nx + ny + nz + 1024;
            while (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz
                    && t_prev <= 1.0 && guard < guard_max) {
                double t_next = std::min(tx, std::min(ty, tz));
                if (!std::isfinite(t_next)) t_next = 1.0;
                t_next = std::max(t_prev, std::min(1.0, t_next));
                const double t_mid = 0.5 * (t_prev + t_next);
                const V3d pm = p0 + d * t_mid;
                add_trilinear(pm.x(), pm.y(), pm.z(), row);

                if (t_next >= 1.0) break;
                if (tx <= ty && tx <= tz) {
                    ix += stepx; tx += dtx;
                } else if (ty <= tx && ty <= tz) {
                    iy += stepy; ty += dty;
                } else {
                    iz += stepz; tz += dtz;
                }
                t_prev = t_next;
                ++guard;
            }
        }
    };

    for (int ti = 0; ti < n_threads; ++ti) {
        const int lo = (n_hits * ti) / n_threads;
        const int hi = (n_hits * (ti + 1)) / n_threads;
        if (lo < hi) workers.emplace_back(worker, lo, hi);
    }
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    return true;
}

/* ── Bounce result ─────────────────────────────────────────────────────────── */

struct BounceStepResult {
    int      hit_tri;
    uint32_t hit_tri_flags;
    V3d      hit_pos;
    V3d      hit_n_param;
    V3d      hit_n_transport;  /* hit_n_param oriented to face the incoming ray */
    bool     should_continue;
    bool     is_sensor_hit;
    bool     is_emissive_hit;
    bool     was_teleported;   /* true when a parametric lens teleport handled this hit */
    int      sensor_group_id;
    /* BSSRDF analytical contribution (backward path at diffuse-transmissive surface).
     * When has_illum_contrib=true, illum_contrib carries the per-band complex
     * amplitude that the backward ray receives analytically from forward-path
     * illumination accumulated in tri_illum_accum.  Caller should emit this as
     * an EndpointRecord rather than waiting for a stochastic scatter hit. */
    bool     has_illum_contrib;
    VXcd     illum_contrib;
};

static BounceStepResult ray_bounce_step_bdpt(
    RayTracerState& st,
    const int* tri_sensor_group,
    V3d& pos, V3d& dir, VXcd& amp,
    double& path_len,
    int& current_medium_mat_idx,
    uint32_t& interaction_flags,
    int n_bands, double min_amplitude,
    std::mt19937_64& rng,
    VXcd* amp_at_hit = nullptr,
    bool is_backward = false);  /* defined below */

/* ── Forward-path tracer ────────────────────────────────────────────────────── */

/* PerHitFn is called once per (source, bounce) for every valid surface hit.
 * All bands are available in amp_hit simultaneously.
 *
 * Signature:
 *   bool fn(int src_id, int bounce,
 *           const V3d& seg_start,    // ray origin for this segment
 *           const V3d& incoming_dir, // unit direction before the bounce
 *           const BounceStepResult& step,
 *           const VXcd& amp_hit,     // amplitude post-propagation, pre-reflection
 *           double path_start,       // cumulative path at seg_start
 *           double path_at_hit)      // cumulative path at hit point
 *
 * Return false to abort all tracing.
 */
template<typename PerHitFn>
static void trace_rays(
    RayTracerState& st,
    int n_sources, const double* src_pos,
    const double* src_dir, const double* src_directivity,
    int n_rays, int max_bounces, double min_amplitude,
    std::mt19937_64& rng,
    bool& abort,
    PerHitFn&& per_hit)
{
    const int n_bands = st.n_bands;

    if (st.optical_assembly)
        st.camera_event_telemetry.rays_launched += n_sources * n_rays;

    VXcd amp(n_bands), amp_hit(n_bands);

    for (int si = 0; si < n_sources && !abort; ++si) {
        const double* sp = src_pos + si * 3;
        const double* sd = src_dir + si * 3;
        V3d src_p(sp[0], sp[1], sp[2]);
        V3d src_d = V3d(sd[0], sd[1], sd[2]).normalized();
        double dirpow = src_directivity[si];

        for (int ri = 0; ri < n_rays && !abort; ++ri) {
            rt_maybe_emit_pulse(st, "trace_rays", si, ri, -1);
            V3d dir = fibonacci_sphere_dir(ri, n_rays, src_d);

            double cos_a      = dir.dot(src_d);
            double dir_weight = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
            if (dir_weight < 0.01) continue;

            const int ray_band = (n_bands > 0) ? ((si + ri) % n_bands) : 0;
            for (int b = 0; b < n_bands; ++b)
                amp[b] = (b == ray_band) ? cd(dir_weight, 0.0) : cd(0.0, 0.0);

            V3d    pos      = src_p;
            double path_len = 0.0;
            int    current_medium_mat_idx = -1;
            uint32_t interaction_flags    = 0u;

            for (int bounce = 0; bounce < max_bounces && !abort; ++bounce) {
                rt_maybe_emit_pulse(st, "trace_rays", si, ri, bounce);
                const V3d    seg_start  = pos;
                const V3d    incoming   = dir;
                const double path_start = path_len;

                BounceStepResult step = ray_bounce_step_bdpt(
                    st, nullptr, pos, dir, amp, path_len,
                    current_medium_mat_idx, interaction_flags,
                    n_bands, min_amplitude, rng, &amp_hit);

                if (step.hit_tri < 0) break;

                if (!per_hit(si, bounce, seg_start, incoming, step, amp_hit, path_start, path_len)) {
                    abort = true;
                    break;
                }

                if (!step.should_continue) break;
            }
        }
    }
}

/* trace_rays_v2 deleted — its caller (ray_tracer_trace_surface) now uses trace_rays. */
/* This block intentionally left as a stub so the compiler errors on any remaining call sites. */

/* ── C API ──────────────────────────────────────────────────────────────────── */

RayTracerState* ray_tracer_create(
    int           n_tri,
    const double* verts,
    const double* normals,
    const int*    mat_idx,
    const float*  mat_buf,
    int           mat_n_mats,
    int           n_bands,
    const double* freq_hz,
    double        speed_m_s,
    const double* atmo_abs)
{
    auto* st = new (std::nothrow) RayTracerState();
    if (!st) return nullptr;

    st->n_bands  = n_bands;
    st->k_real   = Eigen::VectorXd(n_bands);
    st->atmo_abs = Eigen::VectorXd::Map(atmo_abs, n_bands);
    st->speed_m_s   = speed_m_s;
    st->freq_hz_vec = Eigen::VectorXd::Map(freq_hz, n_bands);

    for (int b = 0; b < n_bands; ++b)
        st->k_real[b] = TWO_PI * freq_hz[b] / speed_m_s;

    /* Phase 2: copy MatBuf wholesale.  Layout matches GLSL binding=10. */
    st->mat_n_mats = mat_n_mats;
    {
        size_t expect = static_cast<size_t>(mat_n_mats) * MAT_BUF_MAT_STRIDE;
        st->mat_buf.assign(mat_buf, mat_buf + expect);
    }

    st->tris.resize(static_cast<size_t>(n_tri));
    st->tri_param_group_of_tri.assign(static_cast<size_t>(n_tri), -1);
    std::vector<AABB> tri_aabbs(static_cast<size_t>(n_tri));

    for (int i = 0; i < n_tri; ++i) {
        Triangle& tri = st->tris[static_cast<size_t>(i)];

        /* Vertices */
        const double* v = verts + static_cast<size_t>(i) * 9;
        V3d v0(v[0], v[1], v[2]);
        V3d v1(v[3], v[4], v[5]);
        V3d v2(v[6], v[7], v[8]);
        tri.v0    = v0;
        tri.edge1 = v1 - v0;
        tri.edge2 = v2 - v0;

        /* Normal */
        const double* n = normals + static_cast<size_t>(i) * 3;
        tri.normal = V3d(n[0], n[1], n[2]).normalized();

        /* Material handle into MatBuf (per-band physics resolved at hit time). */
        tri.mat_idx = mat_idx[i];
        tri.flags = 0;
        const bool mat_transmissive = tri_material_is_transmissive(*st, tri, 0);
        if (mat_transmissive) {
            /* Default boundary convention: +normal side is ambient air,
             * -normal side is the triangle material.  This gives physically
             * directional n1/n2 when mesh normals are outward-oriented. */
            tri.medium_pos_mat_idx = -1;
            tri.medium_neg_mat_idx = tri.mat_idx;
        } else {
            tri.medium_pos_mat_idx = -1;
            tri.medium_neg_mat_idx = -1;
        }
        if (mat_reactive_shift_hz(*st, tri.mat_idx) > 0.0 &&
            mat_band_record(*st, tri.mat_idx, 0)[6] > 0.0f)
            tri.flags |= MAT_FLAG_REACTIVE;
        /* slot [5] = emission in SpectralBandRecord layout */
        if (mat_band_record(*st, tri.mat_idx, 0)[5] > 0.0f)
            tri.flags |= MAT_FLAG_EMISSIVE;

        /* AABB for BVH build */
        tri_aabbs[static_cast<size_t>(i)].expand(v0);
        tri_aabbs[static_cast<size_t>(i)].expand(v1);
        tri_aabbs[static_cast<size_t>(i)].expand(v2);
    }

    /* Triangle areas — used by trace_surface for irradiance normalisation. */
    st->tri_areas.resize(static_cast<size_t>(n_tri));
    for (int i = 0; i < n_tri; ++i) {
        const Triangle& t = st->tris[static_cast<size_t>(i)];
        st->tri_areas[static_cast<size_t>(i)] = 0.5 * t.edge1.cross(t.edge2).norm();
    }

    /* Build BVH */
    if (n_tri > 0) {
        st->bvh_tri_ids.resize(static_cast<size_t>(n_tri));
        for (int i = 0; i < n_tri; ++i)
            st->bvh_tri_ids[static_cast<size_t>(i)] = i;
        st->bvh_nodes.reserve(static_cast<size_t>(n_tri) * 2);
        bvh_build(st->bvh_nodes, st->bvh_tri_ids, tri_aabbs, 0, n_tri);
    }

    rt_build_mat_cache(*st);
    return st;
}

void ray_tracer_destroy(RayTracerState* st)
{
    if (!st) return;
    if (st->camera_field_grid && st->camera_field_grid_owned)
        field_grid_destroy(st->camera_field_grid);
    st->camera_field_grid = nullptr;
    delete st;
}

template <typename T>
static inline void rt_append_vector_row(
    std::ostringstream& oss,
    const char* name,
    const char* purpose,
    const std::vector<T>& v,
    size_t& total_used,
    size_t& total_reserved)
{
    const size_t elem = sizeof(T);
    const size_t used = v.size() * elem;
    const size_t reserved = v.capacity() * elem;
    total_used += used;
    total_reserved += reserved;
    oss << "| " << std::left << std::setw(34) << name
        << " | " << std::left << std::setw(38) << purpose
        << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
        << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(v.data()))
        << std::dec << std::setfill(' ')
        << " | " << std::right << std::setw(12) << used
        << " | " << std::right << std::setw(12) << reserved
        << " |\n";
}

template <typename T>
static inline void rt_append_nested_vector_row(
    std::ostringstream& oss,
    const char* name,
    const char* purpose,
    const std::vector<std::vector<T>>& vv,
    size_t& total_used,
    size_t& total_reserved)
{
    size_t used = vv.size() * sizeof(std::vector<T>);
    size_t reserved = vv.capacity() * sizeof(std::vector<T>);
    for (const auto& row : vv) {
        used += row.size() * sizeof(T);
        reserved += row.capacity() * sizeof(T);
    }
    total_used += used;
    total_reserved += reserved;
    oss << "| " << std::left << std::setw(34) << name
        << " | " << std::left << std::setw(38) << purpose
        << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
        << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(vv.data()))
        << std::dec << std::setfill(' ')
        << " | " << std::right << std::setw(12) << used
        << " | " << std::right << std::setw(12) << reserved
        << " |\n";
}

const char* ray_tracer_allocation_table(const RayTracerState* st)
{
    std::ostringstream oss;
    oss << "ray_tracer_allocation_table\n";
    oss << "| buffer_name                        | purpose                                | address        |   used_bytes | reserved_bytes |\n";
    oss << "|------------------------------------|----------------------------------------|----------------|--------------|----------------|\n";

    if (!st) {
        oss << "| " << std::left << std::setw(34) << "(null state)"
            << " | " << std::left << std::setw(38) << "no tracer allocated"
            << " | 0x000000000000"
            << " | " << std::right << std::setw(12) << 0
            << " | " << std::right << std::setw(12) << 0
            << " |\n";
        g_rt_alloc_table = oss.str();
        return g_rt_alloc_table.c_str();
    }

    size_t total_used = 0;
    size_t total_reserved = 0;

    {
        const size_t used = sizeof(*st);
        const size_t reserved = sizeof(*st);
        total_used += used;
        total_reserved += reserved;
        oss << "| " << std::left << std::setw(34) << "RayTracerState"
            << " | " << std::left << std::setw(38) << "state object"
            << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
            << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st))
            << std::dec << std::setfill(' ')
            << " | " << std::right << std::setw(12) << used
            << " | " << std::right << std::setw(12) << reserved
            << " |\n";
    }

    rt_append_vector_row(oss, "tris", "scene triangle payload", st->tris, total_used, total_reserved);
    rt_append_vector_row(oss, "bvh_nodes", "BVH nodes", st->bvh_nodes, total_used, total_reserved);
    rt_append_vector_row(oss, "bvh_tri_ids", "BVH triangle index list", st->bvh_tri_ids, total_used, total_reserved);
    rt_append_vector_row(oss, "tri_areas", "triangle area cache", st->tri_areas, total_used, total_reserved);
    rt_append_vector_row(oss, "mat_buf", "flat material spectral buffer", st->mat_buf, total_used, total_reserved);
    rt_append_vector_row(oss, "scale_contexts", "registered scale contexts", st->scale_contexts, total_used, total_reserved);

    {
        const size_t used = static_cast<size_t>(st->k_real.size()) * sizeof(double);
        const size_t reserved = used;
        total_used += used;
        total_reserved += reserved;
        oss << "| " << std::left << std::setw(34) << "k_real"
            << " | " << std::left << std::setw(38) << "wavenumber per band"
            << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
            << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st->k_real.data()))
            << std::dec << std::setfill(' ')
            << " | " << std::right << std::setw(12) << used
            << " | " << std::right << std::setw(12) << reserved
            << " |\n";
    }
    {
        const size_t used = static_cast<size_t>(st->atmo_abs.size()) * sizeof(double);
        const size_t reserved = used;
        total_used += used;
        total_reserved += reserved;
        oss << "| " << std::left << std::setw(34) << "atmo_abs"
            << " | " << std::left << std::setw(38) << "absorption per band"
            << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
            << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st->atmo_abs.data()))
            << std::dec << std::setfill(' ')
            << " | " << std::right << std::setw(12) << used
            << " | " << std::right << std::setw(12) << reserved
            << " |\n";
    }
    {
        const size_t used = static_cast<size_t>(st->freq_hz_vec.size()) * sizeof(double);
        const size_t reserved = used;
        total_used += used;
        total_reserved += reserved;
        oss << "| " << std::left << std::setw(34) << "freq_hz_vec"
            << " | " << std::left << std::setw(38) << "frequency grid"
            << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
            << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st->freq_hz_vec.data()))
            << std::dec << std::setfill(' ')
            << " | " << std::right << std::setw(12) << used
            << " | " << std::right << std::setw(12) << reserved
            << " |\n";
    }

    rt_append_vector_row(oss, "ray_pool", "persistent live rays", st->ray_pool, total_used, total_reserved);
    rt_append_vector_row(oss, "ray_amp_pool", "per-ray spectral amplitude", st->ray_amp_pool, total_used, total_reserved);
    rt_append_vector_row(oss, "geo_queue", "geometric-stage ray queue", st->geo_queue, total_used, total_reserved);
    rt_append_nested_vector_row(oss, "ctx_queues", "per-context ray queues", st->ctx_queues, total_used, total_reserved);

    rt_append_vector_row(oss, "tri_groups", "triangle-group descriptors", st->tri_groups, total_used, total_reserved);
    rt_append_nested_vector_row(oss, "tri_group_indices", "tri ids per group", st->tri_group_indices, total_used, total_reserved);
    rt_append_nested_vector_row(oss, "tri_group_cum_areas", "area CDF per group", st->tri_group_cum_areas, total_used, total_reserved);
    rt_append_vector_row(oss, "tri_group_default_mat", "default material per group", st->tri_group_default_mat, total_used, total_reserved);
    rt_append_nested_vector_row(oss, "tri_group_power_per_band", "spectral power override per group", st->tri_group_power_per_band, total_used, total_reserved);
    rt_append_vector_row(oss, "tri_group_has_camera", "sensor-camera flags", st->tri_group_has_camera, total_used, total_reserved);
    rt_append_vector_row(oss, "tri_group_camera", "sensor-camera payload", st->tri_group_camera, total_used, total_reserved);
    rt_append_vector_row(oss, "tri_group_parametric_kind", "parametric surface kind per group", st->tri_group_parametric_kind, total_used, total_reserved);
    rt_append_nested_vector_row(oss, "tri_group_parametric_payload", "parametric payload bytes", st->tri_group_parametric_payload, total_used, total_reserved);
    rt_append_vector_row(oss, "tri_param_group_of_tri", "tri->parametric group map", st->tri_param_group_of_tri, total_used, total_reserved);

    {
        size_t used = 0;
        if (st->camera_field_grid) {
            const int64_t n_cells = field_grid_n_cells_total(st->camera_field_grid);
            if (n_cells > 0 && st->n_bands > 0) {
                used = static_cast<size_t>(n_cells) * static_cast<size_t>(st->n_bands) * sizeof(float) * 2;
            }
        }
        const size_t reserved = used;
        total_used += used;
        total_reserved += reserved;
        oss << "| " << std::left << std::setw(34) << "camera_field_grid.primary"
            << " | " << std::left << std::setw(38) << "field capture complex grid"
            << " | 0x" << std::hex << std::setw(12) << std::setfill('0')
            << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st->camera_field_grid ? field_grid_data_re_im(st->camera_field_grid) : nullptr))
            << std::dec << std::setfill(' ')
            << " | " << std::right << std::setw(12) << used
            << " | " << std::right << std::setw(12) << reserved
            << " |\n";
    }

    rt_append_vector_row(oss, "camera_strike_rows", "captured strike rows", st->camera_strike_rows, total_used, total_reserved);

    rt_append_vector_row(oss, "sensor_film_sensor_chunk", "uploaded sensor tensor", st->sensor_film_sensor_chunk, total_used, total_reserved);
    rt_append_vector_row(oss, "sensor_film_film_chunk", "uploaded film tensor", st->sensor_film_film_chunk, total_used, total_reserved);
    rt_append_vector_row(oss, "sensor_film_active_slots", "active slot pairs", st->sensor_film_active_slots, total_used, total_reserved);

    oss << "|------------------------------------|----------------------------------------|----------------|--------------|----------------|\n";
    oss << "| " << std::left << std::setw(34) << "TOTAL"
        << " | " << std::left << std::setw(38) << "RayTracer-owned allocations"
        << " | " << std::left << std::setw(14) << "-"
        << " | " << std::right << std::setw(12) << total_used
        << " | " << std::right << std::setw(12) << total_reserved
        << " |\n";

    g_rt_alloc_table = oss.str();
    return g_rt_alloc_table.c_str();
}

int ray_tracer_set_profile_pulse(
    RayTracerState* st,
    int             enabled,
    double          period_s)
{
    if (!st) return SK_ERR_NULL_STATE;
    st->profile_pulse_enabled = enabled ? 1 : 0;
    st->profile_pulse_period_s = std::max(0.05, period_s > 0.0 ? period_s : 2.0);
    st->profile_pulse_next_ns.store(0, std::memory_order_relaxed);
    if (st->profile_pulse_enabled) {
        rt_maybe_emit_pulse(*st, "profile_enable", -1, -1, -1);
    }
    return SK_OK;
}

int ray_tracer_trace(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    float*          out_segs,
    int             out_cap,
    int*            out_count)
{
    RtProfileScope scope("ray_tracer_trace");
    if (!st || !out_segs || !out_count) return SK_ERR_NULL_STATE;

    *out_count = 0;
    bool abort = false;

    /* Use a local RayWorkerChunk so `count` is fully private.  The chunk
     * owns its segment buffer; after tracing it is flushed into out_segs
     * with explicit overflow accounting. */
    RayWorkerChunk chunk;

    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    const int n_bands = st->n_bands;
    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, const V3d& seg_start, const V3d& /*incoming*/,
            const BounceStepResult& step, const VXcd& amp_hit,
            double path_start, double /*path_at_hit*/) -> bool
        {
            for (int b = 0; b < n_bands; ++b) {
                if (std::abs(amp_hit[b]) <= min_amplitude) continue;
                chunk.write_segment(seg_start, step.hit_pos, si, bounce, b,
                                    amp_hit[b], path_start);
            }
            return true;
        });

    *out_count = chunk.flush_to(out_segs, out_cap, 0);
    return SK_OK;
}

int ray_tracer_trace_surface(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    float*          out_segs,
    int             out_cap,
    int*            out_count,
    float*          out_direct,    /* n_tri * n_bands, row-major [tri][band] */
    float*          out_indirect)  /* n_tri * n_bands, row-major [tri][band] */
{
    if (!st) return SK_ERR_NULL_STATE;

    const int n_bands = st->n_bands;
    const int n_tri   = static_cast<int>(st->tris.size());
    static constexpr double AREA_EPS = 1e-12;

    bool abort = false;
    RayWorkerChunk chunk;

    if (out_count) *out_count = 0;

    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, const V3d& seg_start, const V3d& incoming_dir,
            const BounceStepResult& step, const VXcd& amp_hit,
            double path_start, double path_at_hit) -> bool
        {
            (void)path_at_hit;
            const int hit_tri = step.hit_tri;
            const V3d& p1 = step.hit_pos;

            if (st->camera_field_grid) {
                accumulate_field_capture_segment(*st, seg_start, p1, amp_hit);
            }
            append_camera_strike(
                *st,
                si,
                bounce,
                hit_tri,
                p1,
                step.hit_n_transport,
                incoming_dir,
                path_at_hit,
                path_at_hit,
                1,
                0,
                amp_hit);

            if (out_segs && out_cap > 0) {
                for (int b = 0; b < n_bands; ++b) {
                    if (std::abs(amp_hit[b]) <= min_amplitude) continue;
                    chunk.write_segment(seg_start, p1, si, bounce, b,
                                        amp_hit[b], path_start);
                }
            }

            /* Irradiance: energy = |A|² * cos(θ_in) / area */
            if (hit_tri >= 0 && hit_tri < n_tri) {
                double area    = st->tri_areas[static_cast<size_t>(hit_tri)];
                double cos_in  = std::max(0.0, -incoming_dir.dot(step.hit_n_transport));
                double inv_area = cos_in / std::max(area, AREA_EPS);

                size_t base = static_cast<size_t>(hit_tri) * n_bands;

                float* dest = (bounce == 0 && out_direct) ? out_direct : out_indirect;
                if (dest) {
                    for (int b = 0; b < n_bands; ++b) {
                        double e = std::norm(amp_hit[b]) * inv_area;
                        dest[base + b] += static_cast<float>(e);
                    }
                }
            }
            return true;
        });

    int written = (out_segs && out_cap > 0) ? chunk.flush_to(out_segs, out_cap, 0) : 0;
    if (out_count) *out_count = written;
    return SK_OK;
}

/* ── ray_tracer_trace_callback ───────────────────────────────────────────── */

int ray_tracer_trace_callback(
    RayTracerState*   st,
    int               n_sources,
    const double*     src_pos,
    const double*     src_dir,
    const double*     src_directivity,
    int               n_rays,
    int               max_bounces,
    double            min_amplitude,
    uint32_t          seed,
    RtSegmentCallback cb,
    void*             user,
    int               flush_records,
    RtTraceStats*     stats)
{
    RtProfileScope scope("ray_tracer_trace_callback");
    if (!st || !cb) return SK_ERR_NULL_STATE;

    const int flush_n = (flush_records > 0) ? flush_records : 4096;
    const int buf_float_cap = flush_n * RT_FLOATS_PER_SEG;

    /* Internal batch buffer — one flush_n-deep slab, recycled each flush. */
    std::vector<float> buf;
    buf.reserve(static_cast<size_t>(buf_float_cap));

    int64_t total_delivered = 0;
    bool    abort      = false;
    bool    cb_stopped = false;

    /* Flush buf → callback; clear buf.  Returns false if cb signals stop. */
    auto do_flush = [&]() -> bool {
        int n = static_cast<int>(buf.size()) / RT_FLOATS_PER_SEG;
        if (n == 0) return true;
        int r = cb(buf.data(), n, user);
        total_delivered += n;
        buf.clear();
        if (r != 0) {
            cb_stopped = true;
            abort = true;
            return false;
        }
        return true;
    };

    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    const int n_bands_cb = st->n_bands;
    trace_rays(
        *st, n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude, rng, abort,
        [&](int si, int bounce, const V3d& seg_start, const V3d& /*incoming*/,
            const BounceStepResult& step, const VXcd& amp_hit,
            double path_start, double /*path_at_hit*/) -> bool
        {
            for (int b = 0; b < n_bands_cb; ++b) {
                if (std::abs(amp_hit[b]) <= min_amplitude) continue;
                size_t old_sz = buf.size();
                buf.resize(old_sz + RT_FLOATS_PER_SEG);
                float* s = buf.data() + old_sz;
                s[0]  = static_cast<float>(seg_start.x());
                s[1]  = static_cast<float>(seg_start.y());
                s[2]  = static_cast<float>(seg_start.z());
                s[3]  = static_cast<float>(step.hit_pos.x());
                s[4]  = static_cast<float>(step.hit_pos.y());
                s[5]  = static_cast<float>(step.hit_pos.z());
                s[6]  = static_cast<float>(si);
                s[7]  = static_cast<float>(bounce);
                s[8]  = static_cast<float>(b);
                s[9]  = static_cast<float>(std::abs(amp_hit[b]));
                s[10] = static_cast<float>(std::arg(amp_hit[b]));
                s[11] = static_cast<float>(path_start);
                if (static_cast<int>(buf.size()) >= buf_float_cap && !do_flush())
                    return false;
            }
            return true;
        });

    /* Final flush for any records that did not fill the batch. */
    if (!cb_stopped)
        do_flush();

    if (stats) {
        stats->segments_written = total_delivered;
        stats->segments_dropped = 0;
        stats->total_bounces    = 0;  /* not tracked in callback path */
    }
    return cb_stopped ? SK_ERR_DIVERGED : SK_OK;
}

int ray_tracer_integrate_ir(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    int             n_receivers,
    const double*   rec_pos,
    const double*   rec_aperture_r,
    double          speed_m_s,
    double          sample_rate,
    int             n_samples,
    float*          out_re,
    float*          out_im)
{
    RtProfileScope scope("ray_tracer_integrate_ir");
    if (!st || !out_re || !out_im) return SK_ERR_NULL_STATE;

    const int   n_bands = st->n_bands;
    const double inv_speed = 1.0 / speed_m_s;

    /* Pre-load receiver positions into V3d for fast distance checks. */
    std::vector<V3d> rpos(static_cast<size_t>(n_receivers));
    for (int ri = 0; ri < n_receivers; ++ri)
        rpos[static_cast<size_t>(ri)] = V3d(rec_pos[ri*3], rec_pos[ri*3+1], rec_pos[ri*3+2]);

    bool abort = false;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    const int n_bands_ir = st->n_bands;
    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int /*bounce*/, const V3d& /*seg_start*/, const V3d& /*incoming*/,
            const BounceStepResult& step, const VXcd& amp_hit,
            double /*path_start*/, double path_at_hit) -> bool
        {
            for (int b = 0; b < n_bands_ir; ++b) {
                const cd new_amp = amp_hit[b];
                if (std::abs(new_amp) <= min_amplitude) continue;

                double delay_s = path_at_hit * inv_speed * sample_rate;
                int    t0      = static_cast<int>(delay_s);
                double frac    = delay_s - t0;

                for (int ri = 0; ri < n_receivers; ++ri) {
                    double dist = (step.hit_pos - rpos[static_cast<size_t>(ri)]).norm();
                    double apr  = rec_aperture_r[ri];
                    if (dist >= apr) continue;

                    double weight = 1.0 - dist / apr;

                    size_t base = (static_cast<size_t>(si) * n_receivers * n_bands_ir * n_samples)
                                + (static_cast<size_t>(ri) * n_bands_ir * n_samples)
                                + (static_cast<size_t>(b)  * n_samples);

                    if (t0 >= 0 && t0 < n_samples) {
                        float w = static_cast<float>(weight * (1.0 - frac));
                        out_re[base + t0] += w * static_cast<float>(new_amp.real());
                        out_im[base + t0] += w * static_cast<float>(new_amp.imag());
                    }
                    int t1 = t0 + 1;
                    if (t1 >= 0 && t1 < n_samples) {
                        float w = static_cast<float>(weight * frac);
                        out_re[base + t1] += w * static_cast<float>(new_amp.real());
                        out_im[base + t1] += w * static_cast<float>(new_amp.imag());
                    }
                }
            }
            return true;
        });

    return SK_OK;
}

static inline uint64_t rt_splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline void append_camera_strike_local(
    const RayTracerState& st,
    std::vector<float>& rows,
    int src_id,
    int bounce,
    int tri_id,
    const V3d& pos,
    const V3d& normal,
    const V3d& incoming_dir,
    double depth,
    double total_path,
    int visible,
    int context_entries,
    const VXcd& amp_prop)
{
    if (!st.camera_capture_strikes) return;
    const int stride = (st.camera_strike_stride_floats > 0)
                     ? st.camera_strike_stride_floats
                     : (16 + 2 * st.n_bands);
    const size_t base = rows.size();
    rows.resize(base + static_cast<size_t>(stride), 0.0f);
    float* row = rows.data() + base;

    row[0] = static_cast<float>(pos.x());
    row[1] = static_cast<float>(pos.y());
    row[2] = static_cast<float>(pos.z());
    row[3] = static_cast<float>(normal.x());
    row[4] = static_cast<float>(normal.y());
    row[5] = static_cast<float>(normal.z());
    row[6] = static_cast<float>(incoming_dir.x());
    row[7] = static_cast<float>(incoming_dir.y());
    row[8] = static_cast<float>(incoming_dir.z());
    row[9] = static_cast<float>(tri_id);
    row[10] = static_cast<float>(src_id);
    row[11] = static_cast<float>(bounce);
    row[12] = static_cast<float>(depth);
    row[13] = static_cast<float>(total_path);
    row[14] = static_cast<float>(visible);
    row[15] = static_cast<float>(context_entries);
    for (int b = 0; b < st.n_bands; ++b) {
        row[16 + 2 * b] = static_cast<float>(amp_prop[b].real());
        row[16 + 2 * b + 1] = static_cast<float>(amp_prop[b].imag());
    }
}

struct RtImageWorkerLocal {
    std::vector<float> image;
    std::vector<float> strikes;
};

static int ray_tracer_integrate_image_parallel_impl(
    RayTracerState* st,
    int n_sources,
    const double* src_pos,
    const double* src_dir,
    const double* src_directivity,
    const int32_t* src_n_rays,
    int n_rays,
    int max_bounces,
    double min_amplitude,
    uint32_t seed,
    const double* cam_pos,
    const double* cam_fwd,
    const double* cam_up,
    double fov_rad,
    int width,
    int height,
    float* out_image)
{
    if (!st || !out_image) return SK_ERR_NULL_STATE;
    if (src_n_rays == nullptr && n_rays <= 0) return SK_OK;

    V3d cam_p(cam_pos[0], cam_pos[1], cam_pos[2]);
    V3d cam_f = V3d(cam_fwd[0], cam_fwd[1], cam_fwd[2]).normalized();
    V3d cam_u_hint(cam_up[0], cam_up[1], cam_up[2]);
    V3d cam_r = cam_f.cross(cam_u_hint);
    if (cam_r.norm() < EPS)
        cam_r = cam_f.cross(V3d(1, 0, 0));
    cam_r.normalize();
    V3d cam_u = cam_r.cross(cam_f);

    const double tan_half_v = std::tan(fov_rad * 0.5);
    const double aspect = static_cast<double>(width) / std::max(1, height);
    const double tan_half_h = tan_half_v * aspect;
    const int n_bands = st->n_bands;
    const bool has_bvh = !st->bvh_nodes.empty();

    std::vector<int> ray_base(static_cast<size_t>(n_sources + 1), 0);
    for (int si = 0; si < n_sources; ++si) {
        const int nrs = src_n_rays ? std::max(0, static_cast<int>(src_n_rays[si]))
                                   : std::max(0, n_rays);
        ray_base[static_cast<size_t>(si + 1)] = ray_base[static_cast<size_t>(si)] + nrs;
    }
    const int total_jobs = ray_base[static_cast<size_t>(n_sources)];
    if (total_jobs <= 0) return SK_OK;

    const int hw = std::max(1u, std::thread::hardware_concurrency());
    const int n_workers = std::max(1, std::min(hw, total_jobs));
    std::vector<RtImageWorkerLocal> locals(static_cast<size_t>(n_workers));
    const size_t image_size = static_cast<size_t>(n_bands) * static_cast<size_t>(width) * static_cast<size_t>(height);
    for (int wi = 0; wi < n_workers; ++wi)
        locals[static_cast<size_t>(wi)].image.assign(image_size, 0.0f);

    std::mutex st_mutex;
    ThreadPool pool(static_cast<size_t>(n_workers));
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<size_t>(n_workers));

    const int chunk = (total_jobs + n_workers - 1) / n_workers;
    for (int wi = 0; wi < n_workers; ++wi) {
        const int lo = wi * chunk;
        const int hi = std::min(total_jobs, lo + chunk);
        if (lo >= hi) continue;

        futures.push_back(pool.enqueue([&, wi, lo, hi]() {
            RtImageWorkerLocal& L = locals[static_cast<size_t>(wi)];
            VXcd amp(n_bands);
            VXcd amp_prop(n_bands);

            for (int job = lo; job < hi; ++job) {
                const int si = static_cast<int>(std::upper_bound(ray_base.begin(), ray_base.end(), job) - ray_base.begin()) - 1;
                const int ri = job - ray_base[static_cast<size_t>(si)];
                const int rays_for_source = ray_base[static_cast<size_t>(si + 1)] - ray_base[static_cast<size_t>(si)];
                if (rays_for_source <= 0) continue;

                const double* sp = src_pos + si * 3;
                const double* sd = src_dir + si * 3;
                V3d src_p(sp[0], sp[1], sp[2]);
                V3d src_d = V3d(sd[0], sd[1], sd[2]).normalized();
                double dirpow = src_directivity[si];

                V3d dir = fibonacci_sphere_dir(ri, rays_for_source, src_d);
                double cos_a = dir.dot(src_d);
                double dir_weight = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
                if (dir_weight < 0.01) continue;

                const uint64_t ray_seed = rt_splitmix64((uint64_t)seed
                    ^ (static_cast<uint64_t>(si) << 32)
                    ^ static_cast<uint64_t>(ri));
                std::mt19937_64 rng(ray_seed);

                const int ray_band = (n_bands > 0) ? ((si + ri) % n_bands) : 0;
                for (int b = 0; b < n_bands; ++b)
                    amp[b] = (b == ray_band) ? cd(dir_weight, 0.0) : cd(0.0, 0.0);

                V3d pos = src_p;
                V3d cur_dir = dir;
                double path_len = 0.0;
                int    current_medium_mat_idx = -1;
                uint32_t interaction_flags    = 0u;

                for (int bounce = 0; bounce < max_bounces; ++bounce) {
                    BounceStepResult step = ray_bounce_step_bdpt(
                        *st, nullptr, pos, cur_dir, amp, path_len,
                        current_medium_mat_idx, interaction_flags,
                        n_bands, min_amplitude, rng, &amp_prop);

                    if (step.hit_tri < 0) break;

                    V3d v = step.hit_pos - cam_p;
                    double depth = v.dot(cam_f);
                    const bool in_front = (depth > EPS);
                    const bool pass_depth_cull = !st->camera_depth_cull_enabled
                                              || depth <= st->camera_depth_cull_m;
                    int px = -1;
                    int py = -1;
                    bool in_frame = false;
                    if (in_front && pass_depth_cull) {
                        double x_img = v.dot(cam_r);
                        double y_img = v.dot(cam_u);
                        double ndc_x = x_img / (depth * tan_half_h);
                        double ndc_y = -y_img / (depth * tan_half_v);
                        px = static_cast<int>((ndc_x + 1.0) * 0.5 * width);
                        py = static_cast<int>((ndc_y + 1.0) * 0.5 * height);
                        in_frame = (px >= 0 && px < width && py >= 0 && py < height);
                    }

                    const bool need_image_vis = in_frame;
                    const bool need_strike_vis = st->camera_capture_strikes;
                    int visible = 0;
                    int context_entries = 0;
                    if (in_front && (need_image_vis || need_strike_vis)) {
                        std::lock_guard<std::mutex> lk(st_mutex);
                        const uint64_t c0 = st->camera_full_march_context_entries;
                        visible = camera_visible_to_point(*st, cam_p, step.hit_pos) ? 1 : 0;
                        context_entries = static_cast<int>(st->camera_full_march_context_entries - c0);
                    }

                    if (st->camera_capture_strikes) {
                        append_camera_strike_local(*st, L.strikes,
                                                   si, bounce, step.hit_tri,
                                                   step.hit_pos, step.hit_n_param, cur_dir,
                                                   depth, path_len,
                                                   visible, context_entries,
                                                   amp_prop);
                    }

                    if (need_image_vis && visible) {
                        for (int b = 0; b < n_bands; ++b) {
                            size_t idx = static_cast<size_t>(b) * (height * width)
                                       + static_cast<size_t>(py) * width
                                       + static_cast<size_t>(px);
                            L.image[idx] += static_cast<float>(std::abs(amp_prop[b]));
                        }
                    }

                    if (!step.should_continue) break;
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    for (const RtImageWorkerLocal& L : locals) {
        for (size_t i = 0; i < image_size; ++i)
            out_image[i] += L.image[i];
    }

    if (st->camera_capture_strikes) {
        if (st->camera_strike_stride_floats <= 0)
            st->camera_strike_stride_floats = 16 + 2 * st->n_bands;
        const int stride = st->camera_strike_stride_floats;
        int rows_now = static_cast<int>(st->camera_strike_rows.size() / static_cast<size_t>(stride));
        for (const RtImageWorkerLocal& L : locals) {
            const int n_rows = static_cast<int>(L.strikes.size() / static_cast<size_t>(stride));
            for (int r = 0; r < n_rows; ++r) {
                if (st->camera_capture_max_strikes > 0 && rows_now >= st->camera_capture_max_strikes)
                    return SK_OK;
                const size_t src_off = static_cast<size_t>(r) * stride;
                const size_t dst_off = st->camera_strike_rows.size();
                st->camera_strike_rows.resize(dst_off + static_cast<size_t>(stride));
                std::memcpy(st->camera_strike_rows.data() + dst_off,
                            L.strikes.data() + src_off,
                            sizeof(float) * static_cast<size_t>(stride));
                rows_now += 1;
            }
        }
    }

    return SK_OK;
}

int ray_tracer_integrate_image(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    const double*   cam_pos,
    const double*   cam_fwd,
    const double*   cam_up,
    double          fov_rad,
    int             width,
    int             height,
    float*          out_image)
{
    RtProfileScope scope("ray_tracer_integrate_image");
    return ray_tracer_integrate_image_parallel_impl(
        st,
        n_sources,
        src_pos,
        src_dir,
        src_directivity,
        nullptr,
        n_rays,
        max_bounces,
        min_amplitude,
        seed,
        cam_pos,
        cam_fwd,
        cam_up,
        fov_rad,
        width,
        height,
        out_image);
}

int ray_tracer_integrate_image_packed(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    const int32_t*  src_n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    const double*   cam_pos,
    const double*   cam_fwd,
    const double*   cam_up,
    double          fov_rad,
    int             width,
    int             height,
    float*          out_image)
{
    RtProfileScope scope("ray_tracer_integrate_image_packed");
    return ray_tracer_integrate_image_parallel_impl(
        st,
        n_sources,
        src_pos,
        src_dir,
        src_directivity,
        src_n_rays,
        0,
        max_bounces,
        min_amplitude,
        seed,
        cam_pos,
        cam_fwd,
        cam_up,
        fov_rad,
        width,
        height,
        out_image);
}

struct RtTraceImageWorkerLocal {
    std::vector<float> image;
    std::vector<float> strikes;
    RayWorkerChunk     chunk;
};

static int ray_tracer_trace_integrate_image_parallel_impl(
    RayTracerState* st,
    int n_sources,
    const double* src_pos,
    const double* src_dir,
    const double* src_directivity,
    int n_rays,
    int max_bounces,
    double min_amplitude,
    uint32_t seed,
    const double* cam_pos,
    const double* cam_fwd,
    const double* cam_up,
    double fov_rad,
    int width,
    int height,
    float* out_image,
    float* out_segs,
    int out_cap,
    int* out_count)
{
    if (!st || !out_image || !out_count) return SK_ERR_NULL_STATE;
    *out_count = 0;
    if (n_rays <= 0 || n_sources <= 0) return SK_OK;

    V3d cam_p(cam_pos[0], cam_pos[1], cam_pos[2]);
    V3d cam_f = V3d(cam_fwd[0], cam_fwd[1], cam_fwd[2]).normalized();
    V3d cam_u_hint(cam_up[0], cam_up[1], cam_up[2]);
    V3d cam_r = cam_f.cross(cam_u_hint);
    if (cam_r.norm() < EPS)
        cam_r = cam_f.cross(V3d(1, 0, 0));
    cam_r.normalize();
    V3d cam_u = cam_r.cross(cam_f);

    const double tan_half_v = std::tan(fov_rad * 0.5);
    const double aspect = static_cast<double>(width) / std::max(1, height);
    const double tan_half_h = tan_half_v * aspect;
    const int n_bands = st->n_bands;
    const bool has_bvh = !st->bvh_nodes.empty();

    const int total_jobs = n_sources * n_rays;
    const int hw = std::max(1u, std::thread::hardware_concurrency());
    const int n_workers = std::max(1, std::min(hw, total_jobs));

    std::vector<RtTraceImageWorkerLocal> locals(static_cast<size_t>(n_workers));
    const size_t image_size = static_cast<size_t>(n_bands)
                            * static_cast<size_t>(width)
                            * static_cast<size_t>(height);
    for (int wi = 0; wi < n_workers; ++wi)
        locals[static_cast<size_t>(wi)].image.assign(image_size, 0.0f);

    std::mutex st_mutex;
    ThreadPool pool(static_cast<size_t>(n_workers));
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<size_t>(n_workers));

    const int chunk = (total_jobs + n_workers - 1) / n_workers;
    for (int wi = 0; wi < n_workers; ++wi) {
        const int lo = wi * chunk;
        const int hi = std::min(total_jobs, lo + chunk);
        if (lo >= hi) continue;

        futures.push_back(pool.enqueue([&, wi, lo, hi]() {
            RtTraceImageWorkerLocal& L = locals[static_cast<size_t>(wi)];
            VXcd amp(n_bands);
            VXcd amp_prop(n_bands);

            for (int job = lo; job < hi; ++job) {
                const int si = job / n_rays;
                const int ri = job - si * n_rays;

                const double* sp = src_pos + si * 3;
                const double* sd = src_dir + si * 3;
                V3d src_p(sp[0], sp[1], sp[2]);
                V3d src_d = V3d(sd[0], sd[1], sd[2]).normalized();
                double dirpow = src_directivity[si];

                V3d dir = fibonacci_sphere_dir(ri, n_rays, src_d);
                double cos_a = dir.dot(src_d);
                double dir_weight = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
                if (dir_weight < 0.01) continue;

                const uint64_t ray_seed = rt_splitmix64((uint64_t)seed
                    ^ (static_cast<uint64_t>(si) << 32)
                    ^ static_cast<uint64_t>(ri));
                std::mt19937_64 rng(ray_seed);

                const int ray_band = (n_bands > 0) ? ((si + ri) % n_bands) : 0;
                for (int b = 0; b < n_bands; ++b)
                    amp[b] = (b == ray_band) ? cd(dir_weight, 0.0) : cd(0.0, 0.0);

                V3d pos = src_p;
                V3d cur_dir = dir;
                double path_len = 0.0;
                int    current_medium_mat_idx = -1;
                uint32_t interaction_flags    = 0u;

                for (int bounce = 0; bounce < max_bounces; ++bounce) {
                    const V3d seg_start = pos;
                    const double path_start = path_len;

                    BounceStepResult step = ray_bounce_step_bdpt(
                        *st, nullptr, pos, cur_dir, amp, path_len,
                        current_medium_mat_idx, interaction_flags,
                        n_bands, min_amplitude, rng, &amp_prop);

                    if (step.hit_tri < 0) break;

                    if (out_segs && out_cap > 0) {
                        for (int b = 0; b < n_bands; ++b) {
                            L.chunk.write_segment(seg_start, step.hit_pos, si, bounce, b,
                                                  amp_prop[b], path_start);
                        }
                    }

                    V3d v = step.hit_pos - cam_p;
                    double depth = v.dot(cam_f);
                    const bool in_front = (depth > EPS);
                    const bool pass_depth_cull = !st->camera_depth_cull_enabled
                                              || depth <= st->camera_depth_cull_m;
                    int px = -1;
                    int py = -1;
                    bool in_frame = false;
                    if (in_front && pass_depth_cull) {
                        double x_img = v.dot(cam_r);
                        double y_img = v.dot(cam_u);
                        double ndc_x = x_img / (depth * tan_half_h);
                        double ndc_y = -y_img / (depth * tan_half_v);
                        px = static_cast<int>((ndc_x + 1.0) * 0.5 * width);
                        py = static_cast<int>((ndc_y + 1.0) * 0.5 * height);
                        in_frame = (px >= 0 && px < width && py >= 0 && py < height);
                    }

                    const bool need_image_vis = in_frame;
                    const bool need_strike_vis = st->camera_capture_strikes;
                    int visible = 0;
                    int context_entries = 0;
                    if (in_front && (need_image_vis || need_strike_vis)) {
                        std::lock_guard<std::mutex> lk(st_mutex);
                        const uint64_t c0 = st->camera_full_march_context_entries;
                        visible = camera_visible_to_point(*st, cam_p, step.hit_pos) ? 1 : 0;
                        context_entries = static_cast<int>(st->camera_full_march_context_entries - c0);
                    }

                    if (st->camera_capture_strikes) {
                        append_camera_strike_local(*st, L.strikes,
                                                   si, bounce, step.hit_tri,
                                                   step.hit_pos, step.hit_n_param, cur_dir,
                                                   depth, path_len,
                                                   visible, context_entries,
                                                   amp_prop);
                    }

                    if (need_image_vis && visible) {
                        for (int b = 0; b < n_bands; ++b) {
                            size_t idx = static_cast<size_t>(b) * (height * width)
                                       + static_cast<size_t>(py) * width
                                       + static_cast<size_t>(px);
                            L.image[idx] += static_cast<float>(std::abs(amp_prop[b]));
                        }
                    }

                    if (!step.should_continue) break;
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    for (const RtTraceImageWorkerLocal& L : locals) {
        for (size_t i = 0; i < image_size; ++i)
            out_image[i] += L.image[i];
    }

    if (st->camera_capture_strikes) {
        if (st->camera_strike_stride_floats <= 0)
            st->camera_strike_stride_floats = 16 + 2 * st->n_bands;
        const int stride = st->camera_strike_stride_floats;
        int rows_now = static_cast<int>(st->camera_strike_rows.size() / static_cast<size_t>(stride));
        for (const RtTraceImageWorkerLocal& L : locals) {
            const int n_rows = static_cast<int>(L.strikes.size() / static_cast<size_t>(stride));
            for (int r = 0; r < n_rows; ++r) {
                if (st->camera_capture_max_strikes > 0 && rows_now >= st->camera_capture_max_strikes)
                    break;
                const size_t src_off = static_cast<size_t>(r) * stride;
                const size_t dst_off = st->camera_strike_rows.size();
                st->camera_strike_rows.resize(dst_off + static_cast<size_t>(stride));
                std::memcpy(st->camera_strike_rows.data() + dst_off,
                            L.strikes.data() + src_off,
                            sizeof(float) * static_cast<size_t>(stride));
                rows_now += 1;
            }
        }
    }

    if (out_segs && out_cap > 0) {
        int written = 0;
        for (RtTraceImageWorkerLocal& L : locals) {
            written += L.chunk.flush_to(out_segs, out_cap, written);
        }
        *out_count = written;
    }
    return SK_OK;
}

int ray_tracer_trace_integrate_image(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    const double*   cam_pos,
    const double*   cam_fwd,
    const double*   cam_up,
    double          fov_rad,
    int             width,
    int             height,
    float*          out_image,
    float*          out_segs,
    int             out_cap,
    int*            out_count)
{
    return ray_tracer_trace_integrate_image_parallel_impl(
        st,
        n_sources,
        src_pos,
        src_dir,
        src_directivity,
        n_rays,
        max_bounces,
        min_amplitude,
        seed,
        cam_pos,
        cam_fwd,
        cam_up,
        fov_rad,
        width,
        height,
        out_image,
        out_segs,
        out_cap,
        out_count);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Multi-scale context system
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Physical optics surface helpers ────────────────────────────────────── */

/**
 * Exact Snell's law refraction in 3-D.
 *
 * dir        : incident direction (normalised, pointing TOWARD the surface)
 * surf_normal: outward surface normal (pointing INTO the incident medium, n1)
 * n1         : IOR of the incident medium
 * n2         : IOR of the transmitted medium
 * refracted  : [out] refracted direction (normalised) — valid only when true
 *
 * Returns false on total internal reflection (sin²θt > 1).
 */
static bool snell_refract(
    const V3d& dir, const V3d& surf_normal,
    double n1, double n2,
    V3d& refracted)
{
    /* cos_i > 0 when the normal opposes the incident ray (standard convention). */
    double cos_i   = -dir.dot(surf_normal);
    double n_ratio = n1 / n2;
    double sin2_t  = n_ratio * n_ratio * (1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) return false;      /* TIR */
    double cos_t  = std::sqrt(1.0 - sin2_t);
    refracted = (n_ratio * dir + (n_ratio * cos_i - cos_t) * surf_normal).normalized();
    return true;
}

/**
 * Fresnel power reflectance for unpolarised light (average of s and p).
 *
 * cos_i : cosine of angle of incidence  (≥ 0)
 * cos_t : cosine of angle of refraction (≥ 0, 0 on TIR → returns 1)
 * n1/n2 : IOR of incident / transmitted media
 *
 * Returns R ∈ [0, 1].  T = 1 − R by energy conservation.
 */
static double fresnel_R(
    double cos_i, double cos_t,
    double n1,    double n2)
{
    if (cos_i < EPS || cos_t < EPS) return 1.0;   /* grazing or TIR */
    double rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    double rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}


/* Returns false when no intersection. */
static bool ray_sphere_intersect(
    const V3d& orig, const V3d& dir,
    const V3d& center, double radius,
    double& t_enter, double& t_exit)
{
    V3d oc = orig - center;
    double b    = oc.dot(dir);
    double c    = oc.squaredNorm() - radius * radius;
    double disc = b * b - c;
    if (disc < 0.0) return false;
    double sq = std::sqrt(disc);
    t_enter = -b - sq;
    t_exit  = -b + sq;
    return t_exit > EPS;
}

/* Write one multiscale segment record (RT_FLOATS_PER_SEG_MS = 14 floats). */
static inline void write_seg_ms(
    float* buf, int& count, int cap,
    const V3d& p0, const V3d& p1,
    int src_id, int bounce, int band,
    cd amp, double path_len,
    int context_id, int scale_type)
{
    if (count >= cap) return;
    float* s = buf + static_cast<size_t>(count) * RT_FLOATS_PER_SEG_MS;
    s[0]  = static_cast<float>(p0.x());
    s[1]  = static_cast<float>(p0.y());
    s[2]  = static_cast<float>(p0.z());
    s[3]  = static_cast<float>(p1.x());
    s[4]  = static_cast<float>(p1.y());
    s[5]  = static_cast<float>(p1.z());
    s[6]  = static_cast<float>(src_id);
    s[7]  = static_cast<float>(bounce);
    s[8]  = static_cast<float>(band);
    s[9]  = static_cast<float>(std::abs(amp));
    s[10] = static_cast<float>(std::arg(amp));
    s[11] = static_cast<float>(path_len);
    s[12] = static_cast<float>(context_id);
    s[13] = static_cast<float>(scale_type);
    ++count;
}

/* Propagate 'amp' in-place through 'dist' metres in 'ctx' (NULL = ambient),
 * then write one segment record per band. */
/* medium_mat_idx >= 0: ray is currently propagating inside this material;
 * the material's per-band ior_real (slot 7) and ior_imag (slot 8) override
 * the context n_real/n_imag for OPL phase and Beer-Lambert extinction.
 * This is the "medium transform" hook for transmissive optical media. */
static inline void propagate_ms(
    VXcd& amp,
    const RayTracerState& st,
    const RtScaleContext* ctx,
    double dist,
    double path_len_start,
    const V3d& p0, const V3d& p1,
    int si, int bounce,
    float* out_segs, int& seg_count, int seg_cap,
    int context_id,
    int medium_mat_idx = -1)
{
    const int    n_bands = st.n_bands;
    const int    scale_t = (ctx && ctx->scale_type == RT_SCALE_WAVE)
                             ? RT_SCALE_WAVE : RT_SCALE_GEOMETRIC;
    const double c       = st.speed_m_s;

    for (int b = 0; b < n_bands; ++b) {
        double f_hz = st.freq_hz_vec[b];
        /* ── Medium transform ──────────────────────────────────────────────
         * Priority: material IOR > scale-context IOR > air (1.0/0.0).
         * When inside a transmissive solid, use the material's per-band
         * ior_real for OPL phase scaling and ior_imag for Beer-Lambert
         * absorption — the two physically distinct wave effects of a medium.
         */
        double n_re, n_im;
        if (medium_mat_idx >= 0) {
            n_re = mat_n_real(st, medium_mat_idx, b);
            n_im = mat_n_imag(st, medium_mat_idx, b);
            if (n_re < 1.0) n_re = 1.0;   /* clamp: never slower than vacuum */
        } else {
            n_re = ctx ? ctx->n_real : 1.0;
            n_im = ctx ? ctx->n_imag : 0.0;
        }
        double k_ctx = TWO_PI * f_hz * n_re / c;
        double alpha = TWO_PI * f_hz * n_im / c + st.atmo_abs[b];

        double kt     = k_ctx * dist;
        double decay  = std::exp(-alpha * dist);
        double spread;
        if (scale_t == RT_SCALE_WAVE) {
            double r_tot = path_len_start + dist;
            spread = 1.0 / (1.0 + r_tot * r_tot);
        } else {
            spread = 1.0 / (1.0 + path_len_start + dist * 0.5);
        }
        cd prop(decay * spread * std::cos(kt),
                decay * spread * std::sin(kt));
        amp[b] *= prop;
        write_seg_ms(out_segs, seg_count, seg_cap,
                     p0, p1, si, bounce, b, amp[b],
                     path_len_start, context_id, scale_t);
    }
}

/* Core multiscale inner loop.
 * HitFn2: (si, bounce, hit_tri, incoming_dir, geom_normal, transport_normal, front_face,
 *           amp_at_surface, p0, hit_pos, total_path) -> bool */
template<typename HitFn2>
static void trace_rays_multiscale(
    const RayTracerState& st,
    int n_sources, const double* src_pos,
    const double* src_dir, const double* src_directivity,
    int n_rays, int max_bounces, double min_amplitude,
    std::mt19937_64& rng,
    float* out_segs, int seg_cap, int& seg_count,
    HitFn2&& hit_fn)
{
    RtProfileScope scope("trace_rays_multiscale");
    const int  n_bands = st.n_bands;
    const bool has_bvh = !st.bvh_nodes.empty();
    const int  n_ctx   = static_cast<int>(st.scale_contexts.size());

    std::uniform_real_distribution<double> U(0.0, 1.0);
    VXcd amp(n_bands);
    VXcd amp_surf(n_bands);

    struct Interval { double t0, t1; int ci; };
    std::vector<Interval> intervals;
    intervals.reserve(static_cast<size_t>(std::max(n_ctx, 1)));

    for (int si = 0; si < n_sources; ++si) {
        V3d src_p(src_pos[si*3], src_pos[si*3+1], src_pos[si*3+2]);
        V3d src_d = V3d(src_dir[si*3], src_dir[si*3+1], src_dir[si*3+2]).normalized();
        double dirpow = src_directivity[si];

        for (int ri = 0; ri < n_rays; ++ri) {
            V3d  dir  = fibonacci_sphere_dir(ri, n_rays, src_d);
            double cos_a      = dir.dot(src_d);
            double dir_weight = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
            if (dir_weight < 0.01) continue;

            const int ray_band = (n_bands > 0) ? ((si + ri) % n_bands) : 0;
            for (int b = 0; b < n_bands; ++b)
                amp[b] = (b == ray_band) ? cd(dir_weight, 0.0) : cd(0.0, 0.0);

            V3d    pos      = src_p;
            double path_len = 0.0;
            V3d    cur_dir  = dir;
            /* current_medium_mat_idx: material the ray is currently propagating
             * through (-1 = air/vacuum).  Updated at every transmissive boundary
             * crossing to enable per-material OPL phase and Beer-Lambert. */
            int current_medium_mat_idx = -1;

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                double t_hit   = 1e18;
                int    hit_tri = -1;

                if (has_bvh) {
                    V3d inv_dir = cur_dir.cwiseInverse();
                    bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                              pos, cur_dir, inv_dir, t_hit, hit_tri);
                } else {
                    for (size_t ti = 0; ti < st.tris.size(); ++ti) {
                        double t;
                        if (ray_triangle_hit(pos, cur_dir, st.tris[ti], t) && t < t_hit) {
                            t_hit   = t;
                            hit_tri = static_cast<int>(ti);
                        }
                    }
                }
                if (hit_tri < 0) break;

                V3d hit_pos = pos + t_hit * cur_dir;

                /* Build context intervals for this segment [0, t_hit]. */
                intervals.clear();
                for (int ci = 0; ci < n_ctx; ++ci) {
                    const RtScaleContext& ctx = st.scale_contexts[static_cast<size_t>(ci)];
                    V3d ctr(ctx.center[0], ctx.center[1], ctx.center[2]);
                    double te, tx;
                    if (ray_sphere_intersect(pos, cur_dir, ctr, ctx.radius, te, tx)) {
                        te = std::max(te, 0.0);
                        tx = std::min(tx, t_hit);
                        if (te < tx - EPS)
                            intervals.push_back({te, tx, ci});
                    }
                }
                std::sort(intervals.begin(), intervals.end(),
                          [](const Interval& a, const Interval& b){ return a.t0 < b.t0; });

                /* Walk segment sub-regions. */
                amp_surf = amp;
                double t_walk    = 0.0;
                double path_here = path_len;

                auto coarse_sub = [&](double ta, double tb) {
                    if (tb - ta < EPS) return;
                    V3d p0s = pos + ta * cur_dir;
                    V3d p1s = pos + tb * cur_dir;
                    propagate_ms(amp_surf, st, nullptr, tb - ta,
                                 path_here, p0s, p1s,
                                 si, bounce, out_segs, seg_count, seg_cap, -1,
                                 current_medium_mat_idx);
                    path_here += tb - ta;
                };
                auto wave_sub = [&](double ta, double tb, int ci) {
                    if (tb - ta < EPS) return;
                    const RtScaleContext& ctx = st.scale_contexts[static_cast<size_t>(ci)];
                    double remaining = tb - ta;
                    double dt_m      = (ctx.dt_m > EPS) ? ctx.dt_m : remaining;
                    int sub_steps = std::max(1,
                        std::min(ctx.n_substeps, static_cast<int>(std::ceil(remaining / dt_m))));
                    double t_sub = ta;
                    for (int ss = 0; ss < sub_steps && remaining > EPS; ++ss) {
                        double step = std::min(dt_m, remaining);
                        V3d p0s = pos + t_sub * cur_dir;
                        V3d p1s = p0s + step * cur_dir;
                        propagate_ms(amp_surf, st, &ctx, step,
                                     path_here, p0s, p1s,
                                     si, bounce, out_segs, seg_count, seg_cap, ci,
                                     current_medium_mat_idx);
                        path_here += step;
                        t_sub     += step;
                        remaining -= step;
                    }
                };

                for (const Interval& iv : intervals) {
                    if (iv.t0 > t_walk + EPS) coarse_sub(t_walk, iv.t0);
                    const RtScaleContext& ctx = st.scale_contexts[static_cast<size_t>(iv.ci)];
                    if (ctx.scale_type == RT_SCALE_WAVE)
                        wave_sub(iv.t0, iv.t1, iv.ci);
                    else
                        coarse_sub(iv.t0, iv.t1);
                    t_walk = iv.t1;
                }
                coarse_sub(t_walk, t_hit);

                double total_path = path_len + t_hit;

                V3d hit_n_geom = st.tris[static_cast<size_t>(hit_tri)].normal;
                apply_parametric_surface_point(st, hit_tri, hit_pos, hit_pos, hit_n_geom);
                V3d hit_n_transport = hit_n_geom;
                if (cur_dir.dot(hit_n_transport) > 0.0) hit_n_transport = -hit_n_transport;
                const bool front_face = (cur_dir.dot(hit_n_geom) < 0.0);

                double max_abs = 0.0;
                for (int b = 0; b < n_bands; ++b) {
                    double a = std::abs(amp_surf[b]);
                    if (a > max_abs) max_abs = a;
                }

                bool cont = hit_fn(si, bounce, hit_tri, cur_dir, hit_n_geom, hit_n_transport, front_face,
                                   amp_surf, pos, hit_pos, total_path);
                if (!cont) goto ms_done;

                const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
                if (max_abs < min_amplitude) break;

                /* ── UV integrator image splat (fires for every hit) ──────── */
                if (!st.tri_uv_group_of_tri.empty()
                        && hit_tri < (int)st.tri_uv_group_of_tri.size()) {
                    const int uv_gid = st.tri_uv_group_of_tri[(size_t)hit_tri];
                    if (uv_gid >= 0 && uv_gid < (int)st.group_uv_res.size()) {
                        const int res    = st.group_uv_res[(size_t)uv_gid];
                        const int offset = st.group_uv_accum_offset[(size_t)uv_gid];
                        if (res > 0 && !st.uv_accum_cpu.empty()) {
                            /* Recover barycentric (bu, bv) via Gram matrix solve. */
                            const V3d dp = hit_pos - tri.v0;
                            const double e1e1 = tri.edge1.dot(tri.edge1);
                            const double e1e2 = tri.edge1.dot(tri.edge2);
                            const double e2e2 = tri.edge2.dot(tri.edge2);
                            const double de1  = dp.dot(tri.edge1);
                            const double de2  = dp.dot(tri.edge2);
                            const double det  = e1e1 * e2e2 - e1e2 * e1e2;
                            double bu = 0.0, bv = 0.0;
                            if (det > 1.0e-20) {
                                bu = (e2e2 * de1 - e1e2 * de2) / det;
                                bv = (e1e1 * de2 - e1e2 * de1) / det;
                            }
                            bu = std::max(0.0, std::min(1.0, bu));
                            bv = std::max(0.0, std::min(1.0 - bu, bv));
                            /* Interpolate UV vertex data for this tri. */
                            const float* uvd = st.tri_uv_data.data() + (size_t)hit_tri * 6;
                            const double u = uvd[0] + bu * (uvd[2] - uvd[0]) + bv * (uvd[4] - uvd[0]);
                            const double v = uvd[1] + bu * (uvd[3] - uvd[1]) + bv * (uvd[5] - uvd[1]);
                            const int ix = std::max(0, std::min(res - 1, (int)(u * res)));
                            const int iy = std::max(0, std::min(res - 1, (int)(v * res)));
                            const int texel  = iy * res + ix;
                            const int n2     = res * res;
                            auto& ac = st.uv_accum_cpu;
                            /* [0] hit count */
                            uv_accum_add(ac[(size_t)(offset + 0 * n2 + texel)], 1u);
                            /* [1] source-ID bitfield */
                            reinterpret_cast<std::atomic<uint32_t>&>(
                                ac[(size_t)(offset + 1 * n2 + texel)])
                                .fetch_or(1u << std::min(si, 31), std::memory_order_relaxed);
                            /* [2-5] bounce histogram */
                            uv_accum_add(ac[(size_t)(offset + (2 + std::min(bounce, 3)) * n2 + texel)], 1u);
                            /* [6-7] tag fields — not tracked in CPU path */
                            /* [8-10] hit normal xyz, signed ×32768 */
                            const V3d& hn = hit_n_geom;
                            auto add_signed = [&](int ch, double v) {
                                const int32_t fp = (int32_t)std::max(-1073741824.0,
                                    std::min(1073741824.0, v * 32768.0));
                                uv_accum_add(ac[(size_t)(offset + ch * n2 + texel)],
                                             static_cast<uint32_t>(fp));
                            };
                            add_signed(8, hn.x());
                            add_signed(9, hn.y());
                            add_signed(10, hn.z());
                            /* [11..11+B-1] per-band magnitude ×65536 */
                            /* [11+B..11+2B-1] per-band amp_re ×32768 (signed) */
                            /* [11+2B..11+3B-1] per-band amp_im ×32768 (signed) */
                            /* [11+3B..11+4B-1] forward magnitude ×65536 */
                            /* [11+4B..11+5B-1] sensor magnitude ×65536 */
                            for (int b = 0; b < n_bands; ++b) {
                                const std::complex<double> a = amp_surf[b];
                                const double mag = std::abs(a);
                                const uint32_t fp_mag = (uint32_t)std::min(
                                    mag * 65536.0, (double)0xFFFFFFFFu);
                                uv_accum_add(ac[(size_t)(offset + (11 + b) * n2 + texel)], fp_mag);
                                add_signed(11 + n_bands + b,     a.real());
                                add_signed(11 + 2 * n_bands + b, a.imag());
                                uv_accum_add(ac[(size_t)(offset + (11 + 3 * n_bands + b) * n2 + texel)], fp_mag);
                            }
                        }
                    }
                }

                /* ── Surface interaction ──────────────────────────────────── */
                const bool mat_transmissive = tri_material_is_transmissive(st, tri);
                bool medium_changed = false;

                if (mat_transmissive) {
                    /* Material-driven refractive boundary from MatBuf IOR/transmittance.
                     * Convention: outside is air (n=1), inside is the material. */
                    double n_mat = mat_n_real(st, tri.mat_idx);
                    if (n_mat <= EPS || std::abs(n_mat - 1.0) < 1e-6) {
                        /* No effective refraction — fall through to opaque path. */
                        for (int b = 0; b < n_bands; ++b)
                            amp[b] = amp_surf[b] * mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                        if (U(rng) < mat_cache_diffusion(st.mat_cache, tri.mat_idx)) {
                            cur_dir = cosine_hemisphere(hit_n_transport, rng);
                        } else {
                            cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n_transport) * hit_n_transport).normalized();
                            if (cur_dir.dot(hit_n_transport) < 0.0)
                                cur_dir = cosine_hemisphere(hit_n_transport, rng);
                        }
                    } else {
                        int medium_from = -1;
                        int medium_to = -1;
                        const bool has_side_pair = tri_boundary_media(tri, front_face, medium_from, medium_to);
                        const bool entering =
                            (current_medium_mat_idx < 0) ? front_face
                                                         : (current_medium_mat_idx != tri.mat_idx);
                        double n1 = has_side_pair
                            ? medium_n_real(st, medium_from)
                            : (entering
                                ? ((current_medium_mat_idx >= 0) ? mat_n_real(st, current_medium_mat_idx) : 1.0)
                                : n_mat);
                        double n2 = has_side_pair
                            ? medium_n_real(st, medium_to)
                            : (entering ? n_mat : 1.0);
                        double cos_i = std::max(0.0, -cur_dir.dot(hit_n_transport));
                        V3d refracted;
                        bool can_refract = snell_refract(cur_dir, hit_n_transport, n1, n2, refracted);

                        if (!can_refract) {
                            /* TIR: perfect specular reflection, apply surface refl. */
                            cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n_transport) * hit_n_transport).normalized();
                            for (int b = 0; b < n_bands; ++b)
                                amp[b] = amp_surf[b] * mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                        } else {
                            double sin2_t = (n1/n2) * (n1/n2) * (1.0 - cos_i * cos_i);
                            double cos_t  = std::sqrt(std::max(0.0, 1.0 - sin2_t));
                            double R      = fresnel_R(cos_i, cos_t, n1, n2);
                            if (U(rng) < R) {
                                /* Probabilistic reflection (Russian roulette, unbiased). */
                                cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n_transport) * hit_n_transport).normalized();
                                for (int b = 0; b < n_bands; ++b)
                                    amp[b] = amp_surf[b] * mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                            } else {
                                /* Transmission: exact Snell direction, no refl scaling.
                                 * MC probability (1-R) handles energy balance. */
                                cur_dir = refracted;
                                for (int b = 0; b < n_bands; ++b)
                                    amp[b] = amp_surf[b];
                                medium_changed = true;
                            }
                        }
                    }

                } else {
                    /* Opaque surface: Lambertian or specular reflection. */
                    for (int b = 0; b < n_bands; ++b)
                        amp[b] = amp_surf[b] * mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                    if (U(rng) < mat_cache_diffusion(st.mat_cache, tri.mat_idx)) {
                        cur_dir = cosine_hemisphere(hit_n_transport, rng);
                    } else {
                        cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n_transport) * hit_n_transport).normalized();
                        if (cur_dir.dot(hit_n_transport) < 0.0)
                            cur_dir = cosine_hemisphere(hit_n_transport, rng);
                    }
                }

                /* ── Medium tracking ─────────────────────────────────────────
                 * Update current_medium_mat_idx after every surface event so
                 * the next segment's propagate_ms uses the correct IOR.
                 * entering: ray moves into the solid → medium = this material.
                 * exiting:  ray leaves back to air → medium = air (-1). */
                if (mat_transmissive && medium_changed)
                {
                    int medium_from = -1;
                    int medium_to = -1;
                    if (tri_boundary_media(tri, front_face, medium_from, medium_to))
                        current_medium_mat_idx = medium_to;
                    else
                        current_medium_mat_idx = front_face ? tri.mat_idx : -1;
                }

                /* ── Reactive (fluorescent) re-emission ─────────────────────
                 * After the elastic surface interaction has updated `amp` and
                 * `cur_dir`, redistribute a fraction of the per-band energy
                 * to the Stokes-shifted destination band.  The yield is the
                 * material's reemission coefficient at band 0 — matches
                 * GLSL's `tri.emissive.w` / reemission packing convention. */
                if (st.mat_cache.reactive_shift_hz[tri.mat_idx] > 0.0 &&
                    st.mat_cache.reemit_yield[tri.mat_idx] > 0.0f)
                    apply_reactive_shift_cached(amp, st.mat_cache, tri.mat_idx);

                pos       = hit_pos;
                path_len += t_hit;
            }
            continue;
ms_done:
            break;
        }
    }
}

/* ── Scale context C API ─────────────────────────────────────────────────── */

int ray_tracer_add_scale_context(RayTracerState* st, RtScaleContext* ctx)
{
    if (!st || !ctx) return SK_ERR_NULL_STATE;
    ctx->context_id = static_cast<int>(st->scale_contexts.size());
    st->scale_contexts.push_back(*ctx);
    /* Keep sorted smallest-radius-first so inner zones take priority. */
    std::sort(st->scale_contexts.begin(), st->scale_contexts.end(),
              [](const RtScaleContext& a, const RtScaleContext& b){
                  return a.radius < b.radius;
              });
    return ctx->context_id;
}

int ray_tracer_clear_scale_contexts(RayTracerState* st)
{
    if (!st) return SK_ERR_NULL_STATE;
    st->scale_contexts.clear();
    return SK_OK;
}

int ray_tracer_trace_multiscale(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    float*          out_segs,
    int             out_cap,
    int*            out_count)
{
    RtProfileScope scope("ray_tracer_trace_multiscale");
    if (!st || !out_segs || !out_count) return SK_ERR_NULL_STATE;
    *out_count = 0;
    int count = 0;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays_multiscale(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude, rng,
        out_segs, out_cap, count,
        [](int, int, int, const V3d&, const V3d&, const V3d&, bool,
           const VXcd&, const V3d&, const V3d&, double) -> bool {
            return true;
        });

    *out_count = count;
    return SK_OK;
}

int ray_tracer_trace_multiscale_surface(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    float*          out_segs,
    int             out_cap,
    int*            out_count,
    float*          out_direct,
    float*          out_indirect)
{
    RtProfileScope scope("ray_tracer_trace_multiscale_surface");
    if (!st) return SK_ERR_NULL_STATE;
    if (out_count) *out_count = 0;

    const int    n_bands = st->n_bands;
    const int    n_tri   = static_cast<int>(st->tris.size());
    static constexpr double AREA_EPS = 1e-12;

    int count = 0;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays_multiscale(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude, rng,
        out_segs, out_cap, count,
        [&](int, int bounce, int hit_tri,
            const V3d& incoming_dir, const V3d& geom_normal,
            const V3d& transport_normal, bool front_face,
            const VXcd& amp_prop,
            const V3d&, const V3d&, double) -> bool
        {
            (void)geom_normal;
            (void)front_face;
            if (hit_tri >= 0 && hit_tri < n_tri) {
                double area     = st->tri_areas[static_cast<size_t>(hit_tri)];
                double cos_in   = std::max(0.0, -incoming_dir.dot(transport_normal));
                double inv_area = cos_in / std::max(area, AREA_EPS);
                size_t base     = static_cast<size_t>(hit_tri) * n_bands;
                float* dest     = (bounce == 0 && out_direct) ? out_direct : out_indirect;
                if (dest) {
                    for (int b = 0; b < n_bands; ++b)
                        dest[base + b] += static_cast<float>(std::norm(amp_prop[b]) * inv_area);
                }
            }
            return true;
        });

    if (out_count) *out_count = count;
    return SK_OK;
}

/* ── Physical optics extension API ───────────────────────────────────────── */

int ray_tracer_set_tri_ior(
    RayTracerState* st,
    int             tri_start,
    int             n_tris,
    int             flags)
{
    /* Phase 2 cutover: IOR/refl come from MatBuf via tri.mat_idx; this entry
     * point now only adjusts material flags (TRANSMISSIVE, APERTURE_STOP, …).
     * Kept for API stability — callers used to pass n_in/n_out + flags here. */
    if (!st) return SK_ERR_NULL_STATE;
    int n_total = static_cast<int>(st->tris.size());
    int end     = std::min(tri_start + n_tris, n_total);
    for (int i = tri_start; i < end; ++i) {
        st->tris[static_cast<size_t>(i)].flags = flags;
    }
    return SK_OK;
}

int ray_tracer_set_tri_boundary_media(
    RayTracerState* st,
    int             tri_start,
    int             n_tris,
    int             medium_pos_mat_idx,
    int             medium_neg_mat_idx)
{
    if (!st) return SK_ERR_NULL_STATE;
    int n_total = static_cast<int>(st->tris.size());
    int end     = std::min(tri_start + n_tris, n_total);
    for (int i = tri_start; i < end; ++i) {
        Triangle& tri = st->tris[static_cast<size_t>(i)];
        tri.medium_pos_mat_idx = medium_pos_mat_idx;
        tri.medium_neg_mat_idx = medium_neg_mat_idx;
    }
    return SK_OK;
}

int ray_tracer_set_camera_visibility(
    RayTracerState* st,
    int             camera_vis_mode,
    int             transparent_mode,
    int             enable_depth_cull,
    double          depth_cull_m)
{
    if (!st) return SK_ERR_NULL_STATE;

    if (camera_vis_mode != RT_CAM_VIS_AS_IS
        && camera_vis_mode != RT_CAM_VIS_DIRECT_HIT
        && camera_vis_mode != RT_CAM_VIS_FULL_MARCH)
        return SK_ERR_DIM_MISMATCH;

    if (transparent_mode != RT_CAM_TRANSPARENCY_BLOCK
        && transparent_mode != RT_CAM_TRANSPARENCY_XRAY)
        return SK_ERR_DIM_MISMATCH;

    st->camera_vis_mode = camera_vis_mode;
    st->camera_transparency_mode = transparent_mode;
    st->camera_depth_cull_enabled = enable_depth_cull ? 1 : 0;
    st->camera_depth_cull_m = (depth_cull_m > 0.0) ? depth_cull_m : 0.0;
    return SK_OK;
}

int ray_tracer_get_camera_visibility_stats(
    const RayTracerState* st,
    uint64_t*             out_steps,
    uint64_t*             out_context_entries)
{
    if (!st || !out_steps || !out_context_entries) return SK_ERR_NULL_STATE;
    *out_steps = st->camera_full_march_steps;
    *out_context_entries = st->camera_full_march_context_entries;
    return SK_OK;
}

int ray_tracer_set_field_capture_grid(
    RayTracerState* st,
    FieldGrid*      grid,
    int             take_ownership,
    int             capture_strikes,
    int             max_strikes,
    int             clear_existing)
{
    if (!st) return SK_ERR_NULL_STATE;

    if (st->camera_field_grid && st->camera_field_grid_owned)
        field_grid_destroy(st->camera_field_grid);

    st->camera_field_grid = grid;
    st->camera_field_grid_owned = (take_ownership && grid) ? 1 : 0;
    st->camera_capture_strikes = capture_strikes ? 1 : 0;
    st->camera_capture_max_strikes = std::max(0, max_strikes);
    st->camera_strike_stride_floats = 16 + 2 * st->n_bands;

    if (clear_existing)
        st->camera_strike_rows.clear();
    return SK_OK;
}

int ray_tracer_clear_field_capture(
    RayTracerState* st,
    int             clear_grid,
    int             clear_strikes)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (clear_grid && st->camera_field_grid) {
        if (st->camera_field_grid_owned)
            field_grid_destroy(st->camera_field_grid);
        st->camera_field_grid = nullptr;
        st->camera_field_grid_owned = 0;
    }
    if (clear_strikes)
        st->camera_strike_rows.clear();
    return SK_OK;
}

int ray_tracer_get_field_capture_layout(
    const RayTracerState* st,
    int*                  out_grid_kind,
    int*                  out_n_bands,
    int64_t*              out_n_cells,
    int*                  out_strike_stride_floats,
    int*                  out_n_strikes)
{
    if (!st || !out_grid_kind || !out_n_bands || !out_n_cells
        || !out_strike_stride_floats || !out_n_strikes)
        return SK_ERR_NULL_STATE;

    *out_grid_kind = st->camera_field_grid ? field_grid_kind(st->camera_field_grid) : -1;
    *out_n_bands = st->n_bands;
    *out_n_cells = st->camera_field_grid ? field_grid_n_cells_total(st->camera_field_grid) : 0;
    *out_strike_stride_floats = st->camera_strike_stride_floats > 0
                             ? st->camera_strike_stride_floats
                             : (16 + 2 * st->n_bands);
    const int stride = *out_strike_stride_floats;
    *out_n_strikes = (stride > 0)
                   ? static_cast<int>(st->camera_strike_rows.size() / static_cast<size_t>(stride))
                   : 0;
    return SK_OK;
}

int ray_tracer_copy_field_capture_grid_reim(
    const RayTracerState* st,
    float*                out_reim,
    int64_t               out_count)
{
    if (!st || !out_reim || !st->camera_field_grid) return SK_ERR_NULL_STATE;
    const int64_t n_cells = field_grid_n_cells_total(st->camera_field_grid);
    const int64_t need = static_cast<int64_t>(st->n_bands) * n_cells * 2;
    if (out_count < need) return SK_ERR_DIM_MISMATCH;
    const float* src = field_grid_data_re_im(st->camera_field_grid);
    if (!src) return SK_ERR_NULL_STATE;
    std::memcpy(out_reim, src, static_cast<size_t>(need) * sizeof(float));
    return SK_OK;
}

int ray_tracer_copy_field_capture_strikes(
    const RayTracerState* st,
    float*                out_rows,
    int                   out_rows_cap,
    int*                  out_rows_written)
{
    if (!st || !out_rows || !out_rows_written) return SK_ERR_NULL_STATE;
    const int stride = st->camera_strike_stride_floats > 0
                     ? st->camera_strike_stride_floats
                     : (16 + 2 * st->n_bands);
    const int n_rows = (stride > 0)
                     ? static_cast<int>(st->camera_strike_rows.size() / static_cast<size_t>(stride))
                     : 0;
    const int n_copy = std::min(out_rows_cap, n_rows);
    if (n_copy > 0) {
        std::memcpy(out_rows, st->camera_strike_rows.data(),
                    static_cast<size_t>(n_copy) * stride * sizeof(float));
    }
    *out_rows_written = n_copy;
    return SK_OK;
}

int ray_tracer_set_sensor_film_ssbo(
    RayTracerState*   st,
    const float*      sensor_chunk,
    int               sensor_rows,
    int               sensor_stride,
    const float*      film_chunk,
    int               film_rows,
    int               film_stride,
    const int32_t*    active_slots,
    int               n_slots)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (sensor_rows <= 0 || sensor_stride <= 0 || film_rows <= 0 || film_stride <= 0)
        return SK_ERR_DIM_MISMATCH;
    if (!sensor_chunk || !film_chunk)
        return SK_ERR_NULL_STATE;
    if (n_slots < 0)
        return SK_ERR_DIM_MISMATCH;
    if (n_slots > 0 && !active_slots)
        return SK_ERR_NULL_STATE;

    const size_t sensor_count = static_cast<size_t>(sensor_rows) * static_cast<size_t>(sensor_stride);
    const size_t film_count = static_cast<size_t>(film_rows) * static_cast<size_t>(film_stride);

    st->sensor_film_sensor_chunk.assign(sensor_chunk, sensor_chunk + sensor_count);
    st->sensor_film_film_chunk.assign(film_chunk, film_chunk + film_count);
    st->sensor_film_sensor_rows = sensor_rows;
    st->sensor_film_sensor_stride = sensor_stride;
    st->sensor_film_film_rows = film_rows;
    st->sensor_film_film_stride = film_stride;

    st->sensor_film_active_slots.clear();
    if (n_slots > 0) {
        const size_t slot_count = static_cast<size_t>(n_slots) * 2u;
        st->sensor_film_active_slots.assign(active_slots, active_slots + slot_count);
    }
    st->sensor_film_n_slots = n_slots;
    return SK_OK;
}

/* Upload PBR / enamel / texture-stack material chunks and rebuild the
 * RtMaterialSurfaceCache.  Each chunk must have exactly mat_n_mats rows.
 * Pass NULL / 0 for any chunk to clear it (defaults are used in the cache). */
int ray_tracer_set_surface_chunks(
    RayTracerState* st,
    const float*    pbr_chunk,
    int             n_pbr,
    const float*    enamel_chunk,
    int             n_enamel,
    const float*    tex_stack_chunk,
    int             n_tex)
{
    if (!st) return SK_ERR_NULL_STATE;

    if (pbr_chunk && n_pbr > 0)
        st->pbr_chunk.assign(pbr_chunk, pbr_chunk + (size_t)n_pbr * 16);
    else
        st->pbr_chunk.clear();

    if (enamel_chunk && n_enamel > 0)
        st->enamel_chunk.assign(enamel_chunk, enamel_chunk + (size_t)n_enamel * 8);
    else
        st->enamel_chunk.clear();

    if (tex_stack_chunk && n_tex > 0)
        st->tex_stack_chunk.assign(tex_stack_chunk, tex_stack_chunk + (size_t)n_tex * 16);
    else
        st->tex_stack_chunk.clear();

    rt_build_surf_cache(*st);
    return SK_OK;
}

static inline void sensor_film_row_to_summary(
    const SensorRecord* sensor_row,
    const FilmRecord*   film_row,
    int slot_id,
    int sensor_id,
    int film_id,
    double photons_mean,
    double electrons_mean,
    double snr_peak,
    double snr_mean,
    SensorFilmSlotSummary& out)
{
    const float qe_peak          = sensor_row ? sensor_row->qe_peak          : 0.0f;
    const float full_well_e      = sensor_row ? sensor_row->full_well_e      : 0.0f;
    const float read_noise_e     = sensor_row ? sensor_row->read_noise_e     : 0.0f;
    const float dark_current_e_s = sensor_row ? sensor_row->dark_current_e_s : 0.0f;
    const float exposure_time_s  = film_row   ? film_row->exposure_time_s    : 0.0f;

    out.slot_id = slot_id;
    out.sensor_id = sensor_id;
    out.film_id = film_id;
    out.qe_peak = qe_peak;
    out.read_noise_e = read_noise_e;
    out.dark_current_e_s = dark_current_e_s;
    out.dark_current_accumulated_e = dark_current_e_s * exposure_time_s;
    out.exposure_time_s = exposure_time_s;
    out.full_well_e = full_well_e;
    out.snr_peak = static_cast<float>(snr_peak);
    out.snr_mean = static_cast<float>(snr_mean);
    out.photons_flux_hz = static_cast<float>(photons_mean);
    out.electrons_flux_hz = static_cast<float>(electrons_mean);
}

static bool project_endpoint_record_to_sensor_pixel(
    const CameraSensorDesc& cam,
    const EndpointRecord&    E,
    int                      n_px,
    int                      n_py,
    int&                     out_px,
    int&                     out_py);

int ray_tracer_reduce_endpoint_records_to_sensor_integral(
    const RayTracerState* st,
    const EndpointRecord* records,
    int                   n_records,
    int                   n_px,
    int                   n_py,
    int                   sensor_group_id,
    double                target_photons_per_pixel,
    double                gain,
    float*                out_photons,
    float*                out_electrons,
    float*                out_snr,
    void*                 out_slot_summaries_void,
    int                   out_slot_summary_cap,
    int*                  out_slot_summary_count,
    int*                  out_endpoint_record_count,
    EndpointReductionTelemetry* out_telemetry)
{
    if (!st || !records || !out_photons || !out_electrons || !out_snr)
        return SK_ERR_NULL_STATE;
    if (n_px <= 0 || n_py <= 0 || sensor_group_id < 0)
        return SK_ERR_DIM_MISMATCH;
    if (out_slot_summary_count)
        *out_slot_summary_count = 0;
    if (out_endpoint_record_count)
        *out_endpoint_record_count = n_records;
    if (out_telemetry) {
        std::memset(out_telemetry, 0, sizeof(*out_telemetry));
        out_telemetry->input_records = n_records;
    }

    const CameraSensorDesc* cam_desc = nullptr;
    if (sensor_group_id >= 0
        && sensor_group_id < static_cast<int>(st->tri_group_has_camera.size())
        && st->tri_group_has_camera[static_cast<size_t>(sensor_group_id)]) {
        cam_desc = &st->tri_group_camera[static_cast<size_t>(sensor_group_id)];
    }
    if (!cam_desc)
        return SK_ERR_DIM_MISMATCH;

    const int n_bands = st->n_bands;
    const size_t pix_count = static_cast<size_t>(n_px) * static_cast<size_t>(n_py);
    const size_t img_count = static_cast<size_t>(n_bands) * pix_count;

    std::vector<cd> sensor_img(img_count, cd(0.0, 0.0));
    uint32_t prev_subpath = 0u;
    bool have_prev_subpath = false;
    for (int i = 0; i < n_records; ++i) {
        const EndpointRecord& E = records[static_cast<size_t>(i)];
        if (out_telemetry) {
            if (!have_prev_subpath) {
                out_telemetry->first_subpath_id = E.subpath_id;
                have_prev_subpath = true;
            } else if (E.subpath_id < prev_subpath) {
                out_telemetry->order_regressions += 1;
            }
            prev_subpath = E.subpath_id;
            out_telemetry->last_subpath_id = E.subpath_id;
        }

        if (static_cast<int>(E.group_id) != sensor_group_id) {
            if (out_telemetry) out_telemetry->drop_wrong_group += 1;
            continue;
        }
        if (out_telemetry) out_telemetry->sensor_group_records += 1;
        const int band = static_cast<int>(E.band_id);
        if (band < 0 || band >= n_bands) {
            if (out_telemetry) out_telemetry->drop_invalid_band += 1;
            continue;
        }
        if (!(E.pdf > 0.0f)) {
            continue;
        }

        int px = -1;
        int py = -1;
        const bool has_stream_id =
            (E.stream_id == (float)BDPT_SIDE_LIGHT) ||
            (E.stream_id == (float)BDPT_SIDE_SENSOR);
        const bool is_pixel_cone = has_stream_id
            ? (E.stream_id == (float)BDPT_SIDE_SENSOR)
            : (E.vertex_index >= 0);
        if (is_pixel_cone) {
            // PIXEL_CONE: subpath_id encodes py * n_px + px directly.
            // Do not project E.pos (scene hit) back onto the sensor plane.
            const int pix = static_cast<int>(E.subpath_id);
            if (pix < 0 || pix >= n_px * n_py) {
                if (out_telemetry) out_telemetry->drop_out_of_bounds_pixel += 1;
                continue;
            }
            px = pix % n_px;
            py = pix / n_px;
            if (out_telemetry) out_telemetry->kept_pixel_cone_records += 1;
        } else {
            // Fallback for forward/area-BDPT sensor hits: same-group endpoint
            // landed on sensor geometry but did not come from PIXEL_CONE pixel
            // sampling, so recover the pixel from hit position.
            if (!project_endpoint_record_to_sensor_pixel(*cam_desc, E, n_px, n_py, px, py)) {
                if (out_telemetry) {
                    out_telemetry->drop_non_pixel_cone += 1;
                    out_telemetry->drop_projection_failed += 1;
                }
                continue;
            }
            if (out_telemetry) out_telemetry->kept_projected_records += 1;
        }

        const size_t idx = static_cast<size_t>(band) * pix_count
                         + static_cast<size_t>(py) * static_cast<size_t>(n_px)
                         + static_cast<size_t>(px);
        sensor_img[idx] += cd(E.amp_re, E.amp_im);
        if (out_telemetry)
            out_telemetry->kept_records += 1;
    }

    std::fill(out_photons, out_photons + pix_count, 0.0f);
    std::fill(out_electrons, out_electrons + pix_count, 0.0f);
    std::fill(out_snr, out_snr + pix_count, 0.0f);

    double mean_intensity = 0.0;
    std::vector<double> intensity(pix_count, 0.0);
    for (size_t py = 0; py < static_cast<size_t>(n_py); ++py) {
        for (size_t px = 0; px < static_cast<size_t>(n_px); ++px) {
            const size_t pix_idx = py * static_cast<size_t>(n_px) + px;
            double sum_b = 0.0;
            for (int b = 0; b < n_bands; ++b) {
                const size_t idx = static_cast<size_t>(b) * pix_count + pix_idx;
                sum_b += std::norm(sensor_img[idx]);
            }
            sum_b *= gain * gain;
            intensity[pix_idx] = sum_b;
            mean_intensity += sum_b;
        }
    }
    mean_intensity /= std::max<size_t>(1, pix_count);

    const double target_photons = std::max(0.0, target_photons_per_pixel);
    for (size_t i = 0; i < pix_count; ++i) {
        const double photon_val = (mean_intensity > 1.0e-20 && target_photons > 0.0)
                                ? (intensity[i] / mean_intensity) * target_photons
                                : intensity[i];
        out_photons[i] = static_cast<float>(photon_val);
    }

    double photons_mean = 0.0;
    for (size_t i = 0; i < pix_count; ++i)
        photons_mean += out_photons[i];
    photons_mean /= static_cast<double>(std::max<size_t>(1, pix_count));

    auto* out_slot_summaries = static_cast<SensorFilmSlotSummary*>(out_slot_summaries_void);
    const int valid_slot_cap = std::max(0, out_slot_summary_cap);
    int slot_written = 0;
    int valid_slots = 0;
    double qe_sum = 0.0;
    double read_noise_sum = 0.0;
    double full_well_sum = 0.0;
    std::vector<float> slot_electrons(pix_count, 0.0f);
    std::vector<float> slot_snr(pix_count, 0.0f);

    for (int slot_idx = 0; slot_idx < st->sensor_film_n_slots; ++slot_idx) {
        if (slot_idx >= static_cast<int>(st->sensor_film_active_slots.size()))
            break;
        const int sensor_id = st->sensor_film_active_slots[static_cast<size_t>(slot_idx) * 2u];
        const int film_id   = st->sensor_film_active_slots[static_cast<size_t>(slot_idx) * 2u + 1u];
        if (sensor_id < 0 || film_id < 0)
            continue;
        if (sensor_id >= st->sensor_film_sensor_rows || film_id >= st->sensor_film_film_rows)
            continue;

        /* Access typed structs via reinterpret_cast — layout is guaranteed identical
         * to Python SensorRecord/FilmRecord (same #pragma pack(push,4), same field order). */
        const SensorRecord* sensor_row = st->sensor_film_sensor_chunk.empty()
            ? nullptr
            : reinterpret_cast<const SensorRecord*>(
                  st->sensor_film_sensor_chunk.data()
                  + static_cast<size_t>(sensor_id) * static_cast<size_t>(st->sensor_film_sensor_stride));
        const FilmRecord* film_row = st->sensor_film_film_chunk.empty()
            ? nullptr
            : reinterpret_cast<const FilmRecord*>(
                  st->sensor_film_film_chunk.data()
                  + static_cast<size_t>(film_id) * static_cast<size_t>(st->sensor_film_film_stride));
        if (!sensor_row || !film_row)
            continue;

        const float qe_peak              = sensor_row->qe_peak;
        const float read_noise_e         = sensor_row->read_noise_e;
        const float dark_current_e_s     = sensor_row->dark_current_e_s;
        const float exposure_time_s      = film_row->exposure_time_s;
        const float full_well_e          = sensor_row->full_well_e;
        const double dark_current_accumulated_e = static_cast<double>(dark_current_e_s) * static_cast<double>(exposure_time_s);

        double slot_snr_peak = 0.0;
        double slot_snr_mean = 0.0;
        double slot_electrons_mean = 0.0;
        for (size_t i = 0; i < pix_count; ++i) {
            const double electrons = out_photons[i] * qe_peak;
            const double noise_variance = (read_noise_e * read_noise_e)
                                        + dark_current_accumulated_e
                                        + electrons;
            const double snr = std::sqrt(std::max(0.0, electrons))
                             / std::sqrt(std::max(noise_variance, 1.0e-10));
            slot_electrons[i] = static_cast<float>(electrons);
            slot_snr[i] = static_cast<float>(snr);
            if (snr > slot_snr_peak)
                slot_snr_peak = snr;
            slot_snr_mean += snr;
            slot_electrons_mean += electrons;
            out_electrons[i] += static_cast<float>(electrons);
            out_snr[i] += static_cast<float>(snr);
        }

        ++valid_slots;
        const double inv_pix = 1.0 / std::max<size_t>(1, pix_count);
        slot_snr_mean *= inv_pix;
        slot_electrons_mean *= inv_pix;
        qe_sum += qe_peak;
        read_noise_sum += read_noise_e;
        full_well_sum += full_well_e;

        if (out_slot_summaries && slot_written < valid_slot_cap) {
            sensor_film_row_to_summary(
                sensor_row, film_row,
                slot_idx, sensor_id, film_id,
                photons_mean,
                slot_electrons_mean,
                slot_snr_peak,
                slot_snr_mean,
                out_slot_summaries[static_cast<size_t>(slot_written)]);
            ++slot_written;
        }
    }

    if (valid_slots > 0) {
        const float inv_slots = 1.0f / static_cast<float>(valid_slots);
        for (size_t i = 0; i < pix_count; ++i) {
            out_electrons[i] *= inv_slots;
            out_snr[i] *= inv_slots;
        }
    }

    if (out_slot_summary_count)
        *out_slot_summary_count = slot_written;
    return SK_OK;
}

static bool project_endpoint_record_to_sensor_pixel(
    const CameraSensorDesc& cam,
    const EndpointRecord&    E,
    int                      n_px,
    int                      n_py,
    int&                     out_px,
    int&                     out_py)
{
    if (n_px <= 0 || n_py <= 0)
        return false;

    V3d cpos(cam.pos[0], cam.pos[1], cam.pos[2]);
    V3d cfwd(cam.fwd[0], cam.fwd[1], cam.fwd[2]);
    V3d cup (cam.up[0],  cam.up[1],  cam.up[2]);
    const double cfwd_norm = cfwd.norm();
    const double cup_norm = cup.norm();
    if (cfwd_norm <= 1.0e-12 || cup_norm <= 1.0e-12)
        return false;
    cfwd /= cfwd_norm;
    cup  /= cup_norm;

    V3d cright = cfwd.cross(cup);
    const double cright_norm = cright.norm();
    if (cright_norm <= 1.0e-12)
        return false;
    cright /= cright_norm;
    cup = cright.cross(cfwd).normalized();

    if (cam.sensor_w_m <= 0.0 || cam.sensor_h_m <= 0.0)
        return false;

    const V3d sensor_origin = cpos
        - 0.5 * cam.sensor_w_m * cright
        - 0.5 * cam.sensor_h_m * cup;
    const V3d hit_pos(E.pos[0], E.pos[1], E.pos[2]);
    const V3d rel = hit_pos - sensor_origin;

    const double pix_w = cam.sensor_w_m / static_cast<double>(std::max(1, n_px));
    const double pix_h = cam.sensor_h_m / static_cast<double>(std::max(1, n_py));
    if (pix_w <= 0.0 || pix_h <= 0.0)
        return false;

    out_px = static_cast<int>(std::floor(rel.dot(cright) / pix_w));
    out_py = static_cast<int>(std::floor(rel.dot(cup) / pix_h));
    return (out_px >= 0 && out_px < n_px && out_py >= 0 && out_py < n_py);
}

int ray_tracer_reduce_endpoint_records_to_rgb_image(
    const RayTracerState* st,
    const EndpointRecord* records,
    int                   n_records,
    int                   n_px,
    int                   n_py,
    int                   sensor_group_id,
    double                gain,
    double                hdr_white_percentile,
    float*                out_rgb_linear,
    float*                out_rgb_tonemapped,
    EndpointReductionTelemetry* out_telemetry)
{
    if (!st || !records || !out_rgb_linear || !out_rgb_tonemapped)
        return SK_ERR_NULL_STATE;
    if (n_px <= 0 || n_py <= 0 || sensor_group_id < 0)
        return SK_ERR_DIM_MISMATCH;

    const int n_bands = st->n_bands;
    const size_t pix_count = static_cast<size_t>(n_px) * static_cast<size_t>(n_py);
    const size_t img_count = static_cast<size_t>(n_bands) * pix_count;

    if (out_telemetry) {
        std::memset(out_telemetry, 0, sizeof(*out_telemetry));
        out_telemetry->input_records = n_records;
    }

    const CameraSensorDesc* cam_desc = nullptr;
    if (sensor_group_id >= 0
        && sensor_group_id < static_cast<int>(st->tri_group_has_camera.size())
        && st->tri_group_has_camera[static_cast<size_t>(sensor_group_id)]) {
        cam_desc = &st->tri_group_camera[static_cast<size_t>(sensor_group_id)];
    }
    if (!cam_desc)
        return SK_ERR_DIM_MISMATCH;

    // Obstacle-1: make endpoint->color a canonical C++ route.
    // We accumulate coherent complex amplitudes per (band,pixel), then project
    // spectral power to RGB directly here instead of delegating color semantics
    // to Python.
    std::vector<cd> sensor_img(img_count, cd(0.0, 0.0));
    uint32_t prev_subpath = 0u;
    bool have_prev_subpath = false;
    for (int i = 0; i < n_records; ++i) {
        const EndpointRecord& E = records[static_cast<size_t>(i)];
        if (out_telemetry) {
            if (!have_prev_subpath) {
                out_telemetry->first_subpath_id = E.subpath_id;
                have_prev_subpath = true;
            } else if (E.subpath_id < prev_subpath) {
                out_telemetry->order_regressions += 1;
            }
            prev_subpath = E.subpath_id;
            out_telemetry->last_subpath_id = E.subpath_id;
        }
        if (static_cast<int>(E.group_id) != sensor_group_id) {
            if (out_telemetry) out_telemetry->drop_wrong_group += 1;
            continue;
        }
        if (out_telemetry) out_telemetry->sensor_group_records += 1;
        const int band = static_cast<int>(E.band_id);
        if (band < 0 || band >= n_bands) {
            if (out_telemetry) out_telemetry->drop_invalid_band += 1;
            continue;
        }
        if (!(E.pdf > 0.0f)) {
            continue;
        }
        int px = -1;
        int py = -1;
        const bool has_stream_id =
            (E.stream_id == (float)BDPT_SIDE_LIGHT) ||
            (E.stream_id == (float)BDPT_SIDE_SENSOR);
        const bool is_pixel_cone = has_stream_id
            ? (E.stream_id == (float)BDPT_SIDE_SENSOR)
            : (E.vertex_index >= 0);
        if (is_pixel_cone) {
            // PIXEL_CONE: subpath_id encodes py * n_px + px directly.
            // Do not project E.pos (scene hit) back onto the sensor plane.
            const int pix = static_cast<int>(E.subpath_id);
            if (pix < 0 || pix >= n_px * n_py) {
                if (out_telemetry) out_telemetry->drop_out_of_bounds_pixel += 1;
                continue;
            }
            px = pix % n_px;
            py = pix / n_px;
            if (out_telemetry) out_telemetry->kept_pixel_cone_records += 1;
        } else {
            // Fallback for forward/area-BDPT sensor hits: project the sensor
            // hit position into the registered camera plane instead of forcing
            // perfect PIXEL_CONE encoding.
            if (!project_endpoint_record_to_sensor_pixel(*cam_desc, E, n_px, n_py, px, py)) {
                if (out_telemetry) {
                    out_telemetry->drop_non_pixel_cone += 1;
                    out_telemetry->drop_projection_failed += 1;
                }
                continue;
            }
            if (out_telemetry) out_telemetry->kept_projected_records += 1;
        }
        const size_t idx = static_cast<size_t>(band) * pix_count
                         + static_cast<size_t>(py) * static_cast<size_t>(n_px)
                         + static_cast<size_t>(px);
        sensor_img[idx] += cd(E.amp_re, E.amp_im);
        if (out_telemetry) out_telemetry->kept_records += 1;
    }

    auto wavelength_nm = [](double freq_hz) -> double {
        static constexpr double C_LIGHT_M_S = 299792458.0;
        return (freq_hz > 0.0) ? (C_LIGHT_M_S / freq_hz) * 1.0e9 : 0.0;
    };
    auto wavelength_to_rgb = [](double wl, double& r, double& g, double& b) {
        r = 0.0; g = 0.0; b = 0.0;
        if (wl >= 380.0 && wl < 440.0) {
            r = -(wl - 440.0) / (440.0 - 380.0); b = 1.0;
        } else if (wl < 490.0) {
            g = (wl - 440.0) / (490.0 - 440.0); b = 1.0;
        } else if (wl < 510.0) {
            g = 1.0; b = -(wl - 510.0) / (510.0 - 490.0);
        } else if (wl < 580.0) {
            r = (wl - 510.0) / (580.0 - 510.0); g = 1.0;
        } else if (wl < 645.0) {
            r = 1.0; g = -(wl - 645.0) / (645.0 - 580.0);
        } else if (wl <= 700.0) {
            r = 1.0;
        }
        double edge = 1.0;
        if (wl >= 380.0 && wl < 420.0)
            edge = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0);
        else if (wl > 645.0 && wl <= 700.0)
            edge = 0.3 + 0.7 * (700.0 - wl) / (700.0 - 645.0);
        r *= edge; g *= edge; b *= edge;
    };

    std::vector<float> rgb_linear(static_cast<size_t>(3) * pix_count, 0.0f);
    double white = 0.0;
    const double gain_sq = gain * gain;
    for (size_t py = 0; py < static_cast<size_t>(n_py); ++py) {
        for (size_t px = 0; px < static_cast<size_t>(n_px); ++px) {
            const size_t pix_idx = py * static_cast<size_t>(n_px) + px;
            double r = 0.0, g = 0.0, b = 0.0;
            for (int band = 0; band < n_bands; ++band) {
                const size_t idx = static_cast<size_t>(band) * pix_count + pix_idx;
                const double p = std::norm(sensor_img[idx]) * gain_sq;
                const double wl = wavelength_nm(st->freq_hz_vec[band]);
                double wr = 0.0, wg = 0.0, wb = 0.0;
                wavelength_to_rgb(std::min(700.0, std::max(380.0, wl)), wr, wg, wb);
                r += p * wr;
                g += p * wg;
                b += p * wb;
            }
            const size_t base = pix_idx * 3u;
            rgb_linear[base + 0] = static_cast<float>(r);
            rgb_linear[base + 1] = static_cast<float>(g);
            rgb_linear[base + 2] = static_cast<float>(b);
            white = std::max(white, std::max(r, std::max(g, b)));
        }
    }

    if (white < 1.0e-8)
        white = 1.0;

    // Obstacle-2: expose explicit tone-map semantics and telemetry at C++ API
    // boundary so preview consumers do not need hidden Python-side transforms.
    const double wp = std::min(100.0, std::max(75.0, hdr_white_percentile));
    const double white_scale = std::max(1.0e-8, white * (wp / 100.0));
    const double log_denom = std::log1p(6.0);

    for (size_t i = 0; i < pix_count; ++i) {
        const size_t base = i * 3u;
        double rgb_tmp[3] = {0.0, 0.0, 0.0};
        for (int c = 0; c < 3; ++c) {
            const float lin = rgb_linear[base + static_cast<size_t>(c)];
            out_rgb_linear[base + static_cast<size_t>(c)] = lin;
            const double x = std::max(0.0, static_cast<double>(lin));
            double y = std::log1p((x / white_scale) * 6.0) / log_denom;
            y = y / (1.0 + 0.18 * y);
            rgb_tmp[c] = std::min(1.0, std::max(0.0, y));
        }
        const double luma = 0.2126 * rgb_tmp[0] + 0.7152 * rgb_tmp[1] + 0.0722 * rgb_tmp[2];
        for (int c = 0; c < 3; ++c) {
            const double y = 0.90 * rgb_tmp[c] + 0.10 * luma;
            out_rgb_tonemapped[base + static_cast<size_t>(c)] = static_cast<float>(std::min(1.0, std::max(0.0, y)));
        }
    }

    return SK_OK;
}

/**
 * Project multiscale segments onto a coherent complex sensor image.
 *
 * Finds every segment that straddles z = sensor_z, maps it to a pixel, and
 * accumulates E = amp · exp(i·phase) into out_re / out_im:
 *   out_re[b][py][px] += amp · cos(phase)
 *   out_im[b][py][px] += amp · sin(phase)
 *
 * Squared modulus out_re² + out_im² is the coherent intensity including
 * interference fringes, Airy rings, speckle, etc.
 *
 * Pixel mapping:
 *   sx ∈ [−sensor_r, +sensor_r]  →  px ∈ [0, sensor_w)
 *   sy ∈ [−sensor_r, +sensor_r]  →  py ∈ [0, sensor_h)
 *   pixel_pitch = 2·sensor_r / max(sensor_w, sensor_h)
 */
int ray_tracer_project_coherent(
    int             n_segs,
    const float*    segs,
    int             n_bands,
    int             sensor_w,
    int             sensor_h,
    double          sensor_z,
    double          sensor_r,
    float*          out_re,
    float*          out_im)
{
    if (!segs || !out_re || !out_im || n_segs <= 0) return SK_ERR_NULL_STATE;
    double pixel_pitch = (2.0 * sensor_r)
                         / static_cast<double>(std::max(sensor_w, sensor_h));

    for (int i = 0; i < n_segs; ++i) {
        const float* s = segs + static_cast<size_t>(i) * RT_FLOATS_PER_SEG_MS;
        double x0 = s[0], y0 = s[1], z0 = s[2];
        double x1 = s[3], y1 = s[4], z1 = s[5];
        int    b   = static_cast<int>(s[8]);
        double amp = s[9], phase = s[10];

        if (b < 0 || b >= n_bands) continue;

        double dz = z1 - z0;
        if (std::abs(dz) < 1e-12) continue;
        double t = (sensor_z - z0) / dz;
        if (t < 0.0 || t > 1.0) continue;

        double sx = x0 + t * (x1 - x0);
        double sy = y0 + t * (y1 - y0);

        int px = static_cast<int>((sx + sensor_r) / pixel_pitch);
        int py = static_cast<int>((sy + sensor_r) / pixel_pitch);
        if (px < 0 || px >= sensor_w) continue;
        if (py < 0 || py >= sensor_h) continue;

        int idx = (b * sensor_h + py) * sensor_w + px;
        out_re[idx] += static_cast<float>(amp * std::cos(phase));
        out_im[idx] += static_cast<float>(amp * std::sin(phase));
    }
    return SK_OK;
}

/* ── Near-field physical aperture simulation ─────────────────────────────── */

/**
 * Point-in-polygon test using the crossing number algorithm.
 * Returns true if (px, py) is inside the polygon defined by n_verts pairs
 * of (x,y) coordinates stored interleaved in poly_xy[i*2], poly_xy[i*2+1].
 */
static bool _point_in_polygon(double px, double py,
                               int n_verts, const float* poly_xy)
{
    if (n_verts < 3) return false;
    int crossings = 0;
    for (int i = 0, j = n_verts - 1; i < n_verts; j = i++) {
        double xi = poly_xy[i * 2],     yi = poly_xy[i * 2 + 1];
        double xj = poly_xy[j * 2],     yj = poly_xy[j * 2 + 1];
        /* Check if the horizontal ray from (px, py) rightward crosses edge j→i. */
        bool straddles = ((yi > py) != (yj > py));
        if (straddles) {
            double x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if (px < x_cross) ++crossings;
        }
    }
    return (crossings & 1) != 0;
}

int ray_tracer_apply_aperture_mask(
    int             n_bands,
    int             w,
    int             h,
    double          field_r,
    int             n_verts,
    const float*    poly_xy,
    float*          re_buf,
    float*          im_buf)
{
    if (!re_buf || !im_buf) return SK_ERR_NULL_STATE;
    if (n_verts < 3 || !poly_xy) return SK_ERR_NULL_STATE;

    double pixel_pitch = (2.0 * field_r) / static_cast<double>(std::max(w, h));

    for (int b = 0; b < n_bands; ++b) {
        size_t band_off = static_cast<size_t>(b) * h * w;
        for (int row = 0; row < h; ++row) {
            /* pixel centre in world space */
            double py_w = -field_r + (row + 0.5) * pixel_pitch;
            for (int col = 0; col < w; ++col) {
                double px_w = -field_r + (col + 0.5) * pixel_pitch;
                if (!_point_in_polygon(px_w, py_w, n_verts, poly_xy)) {
                    size_t idx = band_off + static_cast<size_t>(row) * w + col;
                    re_buf[idx] = 0.0f;
                    im_buf[idx] = 0.0f;
                }
            }
        }
    }
    return SK_OK;
}

/**
 * Exact scalar Rayleigh-Sommerfeld diffraction integral of the first kind.
 *
 * For each output pixel (ox, oy) on the observation plane at distance z_dist
 * from the aperture plane:
 *
 *   U_out(ox, oy) = (dx²/2π) Σ_{ix,iy} U_in(ix, iy)
 *                    × (z_dist / r²) × (ik − 1/r) × exp(ikr)
 *
 * where r = sqrt((ox−ix)² + (oy−iy)² + z_dist²).
 * The dx² factor accounts for the area element dA of each input pixel.
 *
 * This direct-summation form is embarrassingly parallel: each output pixel
 * is independent.  Use the companion GLSL compute shader for GPU execution.
 */
int ray_tracer_rs_propagate(
    int             n_bands,
    int             w,
    int             h,
    double          dx,
    double          z_dist,
    const double*   wavelengths_m,
    const float*    in_re,
    const float*    in_im,
    float*          out_re,
    float*          out_im)
{
    if (!in_re || !in_im || !out_re || !out_im || !wavelengths_m)
        return SK_ERR_NULL_STATE;
    if (z_dist <= 0.0 || dx <= 0.0 || w <= 0 || h <= 0 || n_bands <= 0)
        return SK_ERR_NULL_STATE;

    const size_t npix = static_cast<size_t>(w) * h;
    const double dA   = dx * dx;
    const double inv2pi = 1.0 / (2.0 * M_PI);

    /* Clear output */
    std::memset(out_re, 0, n_bands * npix * sizeof(float));
    std::memset(out_im, 0, n_bands * npix * sizeof(float));

    /* Pixel-centre coordinate arrays (flat x = col, y = row convention) */
    std::vector<double> px_coords(static_cast<size_t>(w));
    std::vector<double> py_coords(static_cast<size_t>(h));
    /* Centre the grid symmetrically: x ∈ [-(w-1)/2, +(w-1)/2] × dx */
    for (int c = 0; c < w; ++c) px_coords[c] = (c - 0.5 * (w - 1)) * dx;
    for (int r = 0; r < h; ++r) py_coords[r] = (r - 0.5 * (h - 1)) * dx;

    const double z2 = z_dist * z_dist;

    for (int b = 0; b < n_bands; ++b) {
        const double k      = 2.0 * M_PI / wavelengths_m[b];
        const size_t boff   = static_cast<size_t>(b) * h * w;
        const float* ire    = in_re  + boff;
        const float* iim    = in_im  + boff;
        float*       ore    = out_re + boff;
        float*       oim    = out_im + boff;

        /* For each output pixel */
        for (int oy = 0; oy < h; ++oy) {
            const double yo = py_coords[static_cast<size_t>(oy)];
            for (int ox = 0; ox < w; ++ox) {
                const double xo = px_coords[static_cast<size_t>(ox)];

                double sum_re = 0.0, sum_im = 0.0;

                /* Integrate over all input pixels */
                for (int iy = 0; iy < h; ++iy) {
                    const double dy = yo - py_coords[static_cast<size_t>(iy)];
                    const double dy2 = dy * dy;
                    for (int ix = 0; ix < w; ++ix) {
                        const float u_re = ire[static_cast<size_t>(iy) * w + ix];
                        const float u_im = iim[static_cast<size_t>(iy) * w + ix];
                        if (u_re == 0.0f && u_im == 0.0f) continue; /* aperture mask zero */

                        const double dx_  = xo - px_coords[static_cast<size_t>(ix)];
                        const double r2   = dx_ * dx_ + dy2 + z2;
                        const double r    = std::sqrt(r2);
                        const double r3   = r2 * r;
                        const double kr   = k * r;

                        /* RS kernel: (z/r²) × (ik − 1/r) × exp(ikr) / (2π)
                         * = (z / r²) × exp(ikr) × (ik − 1/r) / (2π)
                         *
                         * Split into real/imag:
                         *   exp(ikr) = cos(kr) + i·sin(kr)
                         *   (ik − 1/r) = -1/r + ik
                         *   product = (-cos(kr)/r − k·sin(kr)) + i(−sin(kr)/r + k·cos(kr))
                         */
                        const double cos_kr = std::cos(kr);
                        const double sin_kr = std::sin(kr);
                        const double zfac   = z_dist * inv2pi / r2;  /* z/(2π r²) */

                        /* kernel real and imag parts */
                        const double ker_re = zfac * (-cos_kr / r - k * sin_kr);
                        const double ker_im = zfac * (-sin_kr / r + k * cos_kr);

                        /* Multiply kernel by input field (complex × complex):
                         *   (a + ib)(c + id) = (ac − bd) + i(ad + bc)
                         */
                        const double a = u_re, b_v = u_im;
                        const double c = ker_re, d = ker_im;
                        sum_re += dA * (a * c - b_v * d);
                        sum_im += dA * (a * d + b_v * c);
                    }
                }

                ore[static_cast<size_t>(oy) * w + ox] = static_cast<float>(sum_re);
                oim[static_cast<size_t>(oy) * w + ox] = static_cast<float>(sum_im);
            }
        }
    }

    return SK_OK;
}

/* ── Beam Propagation Method — batchwise PDE z-stepper ──────────────────────
 *
 * Solves ∂U/∂z = (i/2k) ∇_T² U  (paraxial Helmholtz PDE) using ADI
 * Crank-Nicolson dimensional splitting.  Steps all n_bands simultaneously.
 *
 * Each call:
 *   1.  Apply carrier phase   U *= exp(ik·dz)   (global z-advance of carrier)
 *   2a. x-sweep (implicit x, explicit done in previous step):
 *         (I − β Lx) U* = (I + β Lx) U
 *       where β = i·dz/(4k·dx²), Lx = 1D 2nd-difference operator
 *   2b. y-sweep (implicit y, explicit x already propagated):
 *         (I − β Ly) U^{n+1} = (I + β Ly) U*
 *
 * Both sweeps use the Thomas algorithm (O(N) tridiagonal solve per row/col).
 * Total cost: O(n_bands × w × h) per call.
 *
 * Boundary: Dirichlet U=0 at all four edges (absorbing frame).
 * ─────────────────────────────────────────────────────────────────────────── */

int ray_tracer_wave_bpm_step(
    int             n_bands,
    int             w,
    int             h,
    double          dx,
    double          dz,
    const double*   wavelengths_m,
    float*          re_buf,
    float*          im_buf)
{
    if (!re_buf || !im_buf || !wavelengths_m)  return SK_ERR_NULL_STATE;
    if (w < 2 || h < 2 || n_bands < 1 || dx <= 0.0 || dz == 0.0)
        return SK_ERR_NULL_STATE;

    const size_t npix = static_cast<size_t>(w) * h;
    const double dx2  = dx * dx;
    const int    maxn = std::max(w, h);

    /* Working field: double-precision complex for accuracy */
    std::vector<cd> field(npix);
    std::vector<cd> tmp(npix);
    /* Thomas algorithm scratch — one row or column at a time */
    std::vector<cd> rhs(maxn);
    std::vector<cd> c_prime(maxn);   /* upper-diagonal sweep coefficients  */
    std::vector<cd> d_prime(maxn);   /* RHS sweep values                   */

    /* Thomas algorithm for the uniform tridiagonal system:
     *   -beta · x[i-1] + (1 + 2·beta) · x[i] - beta · x[i+1] = rhs[i]
     * with Dirichlet x[0] = x[n-1] = 0.
     * Solution is written back into rhs[0..n-1]. */
    auto thomas = [&](int n, const cd& beta) {
        const cd diag = cd(1.0, 0.0) + 2.0 * beta;
        const cd off  = -beta;
        rhs[0]     = cd(0.0, 0.0);   /* absorbing left/top boundary  */
        rhs[n - 1] = cd(0.0, 0.0);   /* absorbing right/bottom boundary */
        /* Forward sweep */
        c_prime[0] = off / diag;
        d_prime[0] = rhs[0] / diag;
        for (int i = 1; i < n; ++i) {
            const cd denom = diag - off * c_prime[i - 1];
            c_prime[i]     = off / denom;
            d_prime[i]     = (rhs[i] - off * d_prime[i - 1]) / denom;
        }
        /* Back substitution */
        rhs[n - 1] = d_prime[n - 1];
        for (int i = n - 2; i >= 0; --i)
            rhs[i] = d_prime[i] - c_prime[i] * rhs[i + 1];
    };

    for (int b = 0; b < n_bands; ++b) {
        const size_t boff = static_cast<size_t>(b) * npix;
        const double lam  = wavelengths_m[b];
        if (lam <= 0.0) continue;
        const double k = 2.0 * M_PI / lam;

        /* Load field (float32 → double complex) */
        for (size_t i = 0; i < npix; ++i)
            field[i] = cd(static_cast<double>(re_buf[boff + i]),
                          static_cast<double>(im_buf[boff + i]));

        /* ── Step 1: carrier phase advance  U *= exp(i k dz) ───────────── */
        const double cos_kdz = std::cos(k * dz);
        const double sin_kdz = std::sin(k * dz);
        for (size_t i = 0; i < npix; ++i) {
            const double re = field[i].real(), im = field[i].imag();
            field[i] = cd(re * cos_kdz - im * sin_kdz,
                          re * sin_kdz + im * cos_kdz);
        }

        /* ── Step 2: ADI diffraction  exp(i dz ∇_T² / 2k) ─────────────── *
         * β = i dz / (4k dx²)  — ADI coupling coefficient                 */
        const cd beta = cd(0.0, dz / (4.0 * k * dx2));

        /* Half-step 2a: implicit in x, row by row ─────────────────────── */
        for (int row = 0; row < h; ++row) {
            const size_t rbase = static_cast<size_t>(row) * w;
            for (int col = 0; col < w; ++col) {
                const cd u  = field[rbase + col];
                const cd uw = (col > 0)     ? field[rbase + col - 1] : cd(0.0, 0.0);
                const cd ue = (col < w - 1) ? field[rbase + col + 1] : cd(0.0, 0.0);
                /* RHS = (I + β Lx) u = u + β(uw + ue − 2u) */
                rhs[col] = u + beta * (uw + ue - 2.0 * u);
            }
            thomas(w, beta);
            for (int col = 0; col < w; ++col)
                tmp[rbase + col] = rhs[col];
        }

        /* Half-step 2b: implicit in y, column by column ────────────────── */
        for (int col = 0; col < w; ++col) {
            for (int row = 0; row < h; ++row) {
                const size_t idx = static_cast<size_t>(row) * w + col;
                const cd u  = tmp[idx];
                const cd un = (row > 0)     ? tmp[idx - w] : cd(0.0, 0.0);
                const cd us = (row < h - 1) ? tmp[idx + w] : cd(0.0, 0.0);
                rhs[row] = u + beta * (un + us - 2.0 * u);
            }
            thomas(h, beta);
            for (int row = 0; row < h; ++row)
                field[static_cast<size_t>(row) * w + col] = rhs[row];
        }

        /* Write back (double complex → float32) */
        for (size_t i = 0; i < npix; ++i) {
            re_buf[boff + i] = static_cast<float>(field[i].real());
            im_buf[boff + i] = static_cast<float>(field[i].imag());
        }
    }

    return SK_OK;
}

/* ── Stateful ray scheduler ──────────────────────────────────────────────────
 *
 * API: ray_tracer_spawn / ray_tracer_step / ray_tracer_clear_rays /
 *      ray_tracer_live_ray_count
 *
 * Design: rays are persistent objects in ray_pool[], identified by integer
 * index.  Their complex amplitudes live in ray_amp_pool[ray_id * n_bands + b].
 * Queues are plain std::vector<int> holding ray indices:
 *   geo_queue       — coarse geometric stage (BVH intersection jumps)
 *   ctx_queues[ci]  — fine context ci (wave sub-stepping or geometric within
 *                     the context sphere)
 *
 * Each call to ray_tracer_step() ticks every non-empty queue once, from
 * coarsest (geo_queue, then largest-radius contexts) to finest (smallest-
 * radius contexts last).  Rays move between queues as they enter or exit
 * context spheres or bounce off surfaces.
 * ─────────────────────────────────────────────────────────────────────────── */

/* Return the finest context index (smallest radius) that contains pos,
 * or -1 if pos is outside all registered context spheres.
 * scale_contexts is sorted smallest-radius-first, so the first match
 * is already the finest. */
static int ctx_for_pos(const RayTracerState& st, const V3d& pos)
{
    const int n = static_cast<int>(st.scale_contexts.size());
    for (int ci = 0; ci < n; ++ci) {
        const RtScaleContext& ctx = st.scale_contexts[static_cast<size_t>(ci)];
        V3d ctr(ctx.center[0], ctx.center[1], ctx.center[2]);
        if ((pos - ctr).squaredNorm() <= ctx.radius * ctx.radius)
            return ci;
    }
    return -1;
}

/* Along the ray (pos, dir), find the nearest context-sphere *entry* in
 * [EPS, t_max).  Skips the context the ray is already inside (current_ci).
 * Sets t_enter_out and returns the context index, or -1 if none. */
static int ctx_nearest_entry(
    const RayTracerState& st,
    const V3d& pos, const V3d& dir,
    int current_ci, double t_max,
    double& t_enter_out)
{
    double best_t  = t_max;
    int    best_ci = -1;
    const int n    = static_cast<int>(st.scale_contexts.size());
    for (int ci = 0; ci < n; ++ci) {
        if (ci == current_ci) continue;
        const RtScaleContext& ctx = st.scale_contexts[static_cast<size_t>(ci)];
        V3d ctr(ctx.center[0], ctx.center[1], ctx.center[2]);
        double te, tx;
        if (ray_sphere_intersect(pos, dir, ctr, ctx.radius, te, tx)) {
            /* te > EPS means we are entering from outside this sphere */
            if (te > EPS && te < best_t) {
                best_t  = te;
                best_ci = ci;
            }
        }
    }
    t_enter_out = best_t;
    return best_ci;
}

/* Apply surface interaction to amp[]/dir, updating both in-place.
 * Returns false if the ray is absorbed (aperture stop or bounce limit). */
static bool apply_surface(
    RayTracerState& st,
    VXcd& amp, V3d& dir,
    const V3d& hit_n_geom,
    int hit_tri, int& bounce,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& U)
{
    if (bounce >= st.live_max_bounces) return false;

    const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
    V3d hit_n_transport = hit_n_geom;
    if (dir.dot(hit_n_transport) > 0.0) hit_n_transport = -hit_n_transport;

    if (tri.flags & MAT_FLAG_APERTURE_STOP) {
        return false;  /* absorbed */
    }

    {
        double n_mat = mat_n_real(st, tri.mat_idx);
        bool refractive = tri_material_is_transmissive(st, tri)
                       && n_mat > EPS && std::abs(n_mat - 1.0) > 1e-6;
        if (refractive) {
            bool entering  = (dir.dot(hit_n_geom) < 0.0);
            double n1      = entering ? 1.0   : n_mat;
            double n2      = entering ? n_mat : 1.0;
            double cos_i   = std::max(0.0, -dir.dot(hit_n_transport));
            V3d refracted;
            bool ok        = snell_refract(dir, hit_n_transport, n1, n2, refracted);
            if (!ok) {
                dir = (dir - 2.0 * dir.dot(hit_n_transport) * hit_n_transport).normalized();
            } else {
                double s2t = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
                double ct  = std::sqrt(std::max(0.0, 1.0 - s2t));
                double R   = fresnel_R(cos_i, ct, n1, n2);
                dir = (U(rng) < R)
                    ? (dir - 2.0 * dir.dot(hit_n_transport) * hit_n_transport).normalized()
                    : refracted;
            }
            for (int b = 0; b < st.n_bands; ++b)
                amp[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
        } else {
            for (int b = 0; b < st.n_bands; ++b)
                amp[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
            if (U(rng) < mat_cache_diffusion(st.mat_cache, tri.mat_idx)) {
                dir = cosine_hemisphere(hit_n_transport, rng);
            } else {
                dir = (dir - 2.0 * dir.dot(hit_n_transport) * hit_n_transport).normalized();
                if (dir.dot(hit_n_transport) < 0.0)
                    dir = cosine_hemisphere(hit_n_transport, rng);
            }
        }
    }
    ++bounce;
    return true;
}

/* Advance a single ray one step using the given context (NULL = geometric).
 * Updates rs and ray_amp_pool in st.  Writes segments.
 * Returns the context_id to re-queue the ray in (-1 = geo, -2 = dead). */
static int scheduler_advance_ray(
    RayTracerState& st,
    int ray_id, int current_ci,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& U,
    float* out_segs, int& seg_count, int seg_cap)
{
    RtRayState& rs = st.ray_pool[static_cast<size_t>(ray_id)];
    if (!rs.alive) return -2;

    const int n_bands = st.n_bands;
    V3d pos(rs.pos[0], rs.pos[1], rs.pos[2]);
    V3d dir(rs.dir[0], rs.dir[1], rs.dir[2]);

    VXcd amp(n_bands);
    const size_t amp_base = static_cast<size_t>(ray_id) * static_cast<size_t>(n_bands);
    for (int b = 0; b < n_bands; ++b)
        amp[b] = st.ray_amp_pool[amp_base + b];

    /* ── Determine step distance ─────────────────────────────────────────── */
    const RtScaleContext* ctx_ptr  = (current_ci >= 0)
        ? &st.scale_contexts[static_cast<size_t>(current_ci)]
        : nullptr;

    /* Maximum step from context physics */
    double dt_max = (ctx_ptr && ctx_ptr->scale_type == RT_SCALE_WAVE
                     && ctx_ptr->dt_m > EPS)
                    ? ctx_ptr->dt_m : 1e18;

    /* Distance to exit the current context sphere (if in one) */
    double t_ctx_exit = 1e18;
    if (current_ci >= 0 && ctx_ptr) {
        V3d ctr(ctx_ptr->center[0], ctx_ptr->center[1], ctx_ptr->center[2]);
        double te, tx;
        if (ray_sphere_intersect(pos, dir, ctr, ctx_ptr->radius, te, tx) && tx > EPS)
            t_ctx_exit = tx;
    }

    /* BVH: nearest surface intersection */
    double t_hit   = 1e18;
    int    hit_tri = -1;
    if (!st.bvh_nodes.empty()) {
        V3d inv_dir = dir.cwiseInverse();
        bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                  pos, dir, inv_dir, t_hit, hit_tri);
    } else {
        for (size_t ti = 0; ti < st.tris.size(); ++ti) {
            double t;
            if (ray_triangle_hit(pos, dir, st.tris[ti], t) && t < t_hit) {
                t_hit   = t;
                hit_tri = static_cast<int>(ti);
            }
        }
    }

    /* Nearest finer context-sphere entry within the current step cap */
    double step_cap = std::min({dt_max, t_ctx_exit, t_hit < 1e17 ? t_hit : 1e18});
    double t_enter;
    int    enter_ci = ctx_nearest_entry(st, pos, dir, current_ci, step_cap, t_enter);

    /* ── Choose what happens this step ──────────────────────────────────── */
    enum { ACT_ENTER_CTX, ACT_HIT_SURFACE, ACT_EXIT_CTX, ACT_CONTINUE } act;
    double step;

    if (enter_ci >= 0 && t_enter < step_cap - EPS) {
        act  = ACT_ENTER_CTX;
        step = t_enter;
    } else if (hit_tri >= 0 && t_hit < std::min(dt_max, t_ctx_exit) - EPS) {
        act  = ACT_HIT_SURFACE;
        step = t_hit;
    } else if (t_ctx_exit < dt_max - EPS) {
        act  = ACT_EXIT_CTX;
        step = t_ctx_exit;
    } else {
        act  = ACT_CONTINUE;
        step = std::min(dt_max, t_ctx_exit);
    }

    if (step < EPS) step = EPS * 10.0;
    if (step > 1e17) {
        /* Ray escaped without hitting anything */
        rs.alive = 0;
        return -2;
    }

    /* ── Propagate amplitude and emit segment ────────────────────────────── */
    V3d p1 = pos + step * dir;
    propagate_ms(amp, st, ctx_ptr, step, rs.path_len,
                 pos, p1, rs.src_id, rs.bounce,
                 out_segs, seg_count, seg_cap, current_ci);
    rs.path_len += step;

    /* Check amplitude floor */
    double max_abs = 0.0;
    for (int b = 0; b < n_bands; ++b) {
        double a = std::abs(amp[b]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < st.live_min_amplitude) {
        rs.alive = 0;
        for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
        return -2;
    }

    /* ── Handle the outcome ──────────────────────────────────────────────── */
    if (act == ACT_ENTER_CTX) {
        /* Transfer to finer context and apply its entry transform immediately.
         * Full-assembly neural payloads may move pos near the baked sensor hit,
         * so re-evaluate the destination context after dispatch. */
        pos = p1;
        const RtScaleContext& enter_ctx = st.scale_contexts[static_cast<size_t>(enter_ci)];
        (void)dispatch_scale_context_entry(st, enter_ctx, pos, dir, amp, &rs.path_len);
        rs.pos[0] = pos.x(); rs.pos[1] = pos.y(); rs.pos[2] = pos.z();
        rs.dir[0] = dir.x(); rs.dir[1] = dir.y(); rs.dir[2] = dir.z();
        for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
        int new_ci = ctx_for_pos(st, pos);
        rs.context_id = new_ci;
        return new_ci;
    }

    if (act == ACT_HIT_SURFACE) {
        V3d hit_n_geom = st.tris[static_cast<size_t>(hit_tri)].normal;
        V3d hit_pos = p1;
        apply_parametric_surface_point(st, hit_tri, hit_pos, hit_pos, hit_n_geom);
        {
            int nb_mfld = 0;
            const float* mpp = tri_manifold_payload(st, hit_tri, &nb_mfld);
            if (mpp) {
                if (nb_mfld < (int)(8 * sizeof(float))) {
                    /* Interior absorber — kill ray. */
                    rs.alive = 0;
                    for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
                    return -2;
                }
                /* pos is set to T1 proxy-mesh hit; for PARAMETRIC the
                 * apply_* function now seeks to the exact s=0 conic entry. */
                pos = hit_pos;
                const float magic = mpp[0];
                bool absorbed = false;
                if (magic == 14949.0f) {
                    absorbed = apply_parametric_lens_from_f32(st, mpp, nb_mfld, pos, dir, amp, &rs.path_len);
                } else if (magic == 14948.0f) {
                    /* Deferred: MLP transport is disabled until the basic
                     * reversible parametric camera path is finished. */
                    absorbed = true;
                } else if (magic == 14946.0f || magic == 14947.0f ||
                           magic == 14950.0f || magic == 14951.0f) {
                    /* Deferred: LUT transport is disabled until it can supply
                     * the same reversible optical data as the parametric path. */
                    absorbed = true;
                } else {
                    absorbed = true;
                }
                if (absorbed) {
                    rs.alive = 0;
                    for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
                    return -2;
                }
                rs.pos[0] = pos.x(); rs.pos[1] = pos.y(); rs.pos[2] = pos.z();
                rs.dir[0] = dir.x(); rs.dir[1] = dir.y(); rs.dir[2] = dir.z();
                for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
                int new_ci = ctx_for_pos(st, pos);
                rs.context_id = new_ci;
                return new_ci;
            }
        }
        bool alive = apply_surface(st, amp, dir, hit_n_geom, hit_tri, rs.bounce, rng, U);
        if (!alive) {
            rs.alive = 0;
            for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
            return -2;
        }
        V3d npos = p1 + dir * (EPS * 200.0);
        rs.pos[0] = npos.x(); rs.pos[1] = npos.y(); rs.pos[2] = npos.z();
        rs.dir[0] = dir.x();  rs.dir[1] = dir.y();  rs.dir[2] = dir.z();
        for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
        int new_ci   = ctx_for_pos(st, npos);
        rs.context_id = new_ci;
        return new_ci;
    }

    if (act == ACT_EXIT_CTX) {
        /* Nudge just past the sphere boundary */
        V3d npos = p1 + dir * (EPS * 10.0);
        rs.pos[0] = npos.x(); rs.pos[1] = npos.y(); rs.pos[2] = npos.z();
        rs.dir[0] = dir.x();  rs.dir[1] = dir.y();  rs.dir[2] = dir.z();
        for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
        int new_ci   = ctx_for_pos(st, npos);
        rs.context_id = new_ci;
        return new_ci;
    }

    /* ACT_CONTINUE — stays in the same context, advances by step */
    rs.pos[0] = p1.x(); rs.pos[1] = p1.y(); rs.pos[2] = p1.z();
    rs.dir[0] = dir.x(); rs.dir[1] = dir.y(); rs.dir[2] = dir.z();
    for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
    rs.context_id = current_ci;
    return current_ci;
}

/* ── Public scheduler API ─────────────────────────────────────────────────── */

int ray_tracer_spawn(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed)
{
    if (!st) return SK_ERR_NULL_STATE;
    st->live_max_bounces   = max_bounces;
    st->live_min_amplitude = min_amplitude;

    /* Ensure ctx_queues has an entry per registered context */
    st->ctx_queues.resize(st->scale_contexts.size());

    const int n_bands = st->n_bands;
    std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0xdeadbeef8badf00dULL);

    for (int si = 0; si < n_sources; ++si) {
        const double* sp = src_pos + si * 3;
        const double* sd = src_dir + si * 3;
        V3d src_p(sp[0], sp[1], sp[2]);
        V3d src_d = V3d(sd[0], sd[1], sd[2]).normalized();
        double dirpow = src_directivity[si];

        for (int ri = 0; ri < n_rays; ++ri) {
            V3d fib_dir    = fibonacci_sphere_dir(ri, n_rays, src_d);
            double cos_a   = fib_dir.dot(src_d);
            double weight  = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
            if (weight < 0.01) continue;

            int ray_id = static_cast<int>(st->ray_pool.size());

            RtRayState rs{};
            rs.pos[0] = src_p.x(); rs.pos[1] = src_p.y(); rs.pos[2] = src_p.z();
            rs.dir[0] = fib_dir.x(); rs.dir[1] = fib_dir.y(); rs.dir[2] = fib_dir.z();
            rs.path_len   = 0.0;
            rs.bounce     = 0;
            rs.src_id     = si;
            rs.alive      = 1;
            rs.rng_state  = rng();

            /* Determine initial context (spawn point may already be in a fine zone) */
            int init_ci   = ctx_for_pos(*st, src_p);
            rs.context_id = init_ci;

            st->ray_pool.push_back(rs);

            /* Allocate amplitude: initial magnitude = directivity weight */
            for (int b = 0; b < n_bands; ++b)
                st->ray_amp_pool.push_back(cd(weight, 0.0));

            if (init_ci >= 0)
                st->ctx_queues[static_cast<size_t>(init_ci)].push_back(ray_id);
            else
                st->geo_queue.push_back(ray_id);
        }
    }
    return SK_OK;
}

int ray_tracer_step(
    RayTracerState* st,
    float*          out_segs,
    int             out_cap,
    int*            out_count,
    int*            n_live_out)
{
    if (!st || !out_segs || !out_count) return SK_ERR_NULL_STATE;
    *out_count = 0;

    const int n_ctx = static_cast<int>(st->scale_contexts.size());
    st->ctx_queues.resize(static_cast<size_t>(n_ctx));

    int seg_count = 0;
    /* Per-step RNG: seed from ray_pool size + seg_count for unpredictability */
    std::mt19937_64 rng(static_cast<uint64_t>(st->ray_pool.size()) * 6364136223846793005ULL
                        + 1442695040888963407ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    /* Helper: re-queue a ray into the appropriate queue based on its context_id.
     * ctx_queue additions go into a temporary buffer to avoid processing a ray
     * twice in the same step if it moves between context queues. */
    std::vector<std::vector<int>> new_ctx(static_cast<size_t>(n_ctx));

    /* ── 1. Geometric (coarsest) queue ──────────────────────────────────── */
    {
        std::vector<int> snapshot;
        snapshot.swap(st->geo_queue);
        for (int ray_id : snapshot) {
            if (!st->ray_pool[static_cast<size_t>(ray_id)].alive) continue;
            int dest = scheduler_advance_ray(*st, ray_id, -1, rng, U,
                                             out_segs, seg_count, out_cap);
            if (dest == -2) continue;          /* dead */
            if (dest == -1)
                st->geo_queue.push_back(ray_id); /* stays in geo */
            else
                new_ctx[static_cast<size_t>(dest)].push_back(ray_id);
        }
    }

    /* ── 2. Fine context queues, coarsest-first (largest radius first) ───── */
    /* scale_contexts is sorted smallest-radius-first, so iterate in reverse */
    for (int ci = n_ctx - 1; ci >= 0; --ci) {
        std::vector<int> snapshot;
        snapshot.swap(st->ctx_queues[static_cast<size_t>(ci)]);
        /* Prepend any rays that arrived in this context from the geo step */
        for (int id : new_ctx[static_cast<size_t>(ci)]) snapshot.push_back(id);
        new_ctx[static_cast<size_t>(ci)].clear();

        for (int ray_id : snapshot) {
            if (!st->ray_pool[static_cast<size_t>(ray_id)].alive) continue;
            int dest = scheduler_advance_ray(*st, ray_id, ci, rng, U,
                                             out_segs, seg_count, out_cap);
            if (dest == -2) continue;          /* dead */
            if (dest == -1)
                st->geo_queue.push_back(ray_id);
            else if (dest == ci)
                st->ctx_queues[static_cast<size_t>(ci)].push_back(ray_id); /* stays */
            else
                new_ctx[static_cast<size_t>(dest)].push_back(ray_id);
        }
    }

    /* Flush any remaining new_ctx entries (rays that entered a finer context
     * during the finest context's own processing — deferred to next step) */
    for (int ci = 0; ci < n_ctx; ++ci) {
        for (int id : new_ctx[static_cast<size_t>(ci)])
            st->ctx_queues[static_cast<size_t>(ci)].push_back(id);
    }

    *out_count = seg_count;
    if (n_live_out) *n_live_out = ray_tracer_live_ray_count(st);
    return SK_OK;
}

int ray_tracer_clear_rays(RayTracerState* st)
{
    if (!st) return SK_ERR_NULL_STATE;
    st->ray_pool.clear();
    st->ray_amp_pool.clear();
    st->geo_queue.clear();
    for (auto& q : st->ctx_queues) q.clear();
    return SK_OK;
}

int ray_tracer_live_ray_count(const RayTracerState* st)
{
    if (!st) return 0;
    int n = static_cast<int>(st->geo_queue.size());
    for (const auto& q : st->ctx_queues)
        n += static_cast<int>(q.size());
    return n;
}

/* ── Triangle-group registry impls (declared in triangle_groups.h) ────── */
extern "C" SK_API int ray_tracer_register_tri_group(
    RayTracerState* st, const TriGroupDesc* desc)
{
    if (!st || !desc || !desc->tri_indices || desc->n_tris <= 0)
        return SK_ERR_NULL_STATE;

    /* Validate triangle indices. */
    const int n_tri = static_cast<int>(st->tris.size());
    for (int i = 0; i < desc->n_tris; ++i) {
        int t = desc->tri_indices[i];
        if (t < 0 || t >= n_tri) return SK_ERR_NULL_STATE;
    }

    /* Deep-copy the descriptor. */
    TriGroupDesc copy = *desc;
    copy.group_id     = static_cast<int>(st->tri_groups.size());
    copy.tri_indices  = nullptr;  /* ownership stays in tri_group_indices */
    /* Pointer-typed optional fields are deep-copied into separate vectors;
     * null them out in the stored descriptor so a stale pointer can never
     * be dereferenced after the caller's buffer goes away. */
    copy.power_W_per_band = nullptr;
    copy.sensor_camera    = nullptr;
    copy.parametric_payload = nullptr;
    st->tri_groups.push_back(copy);

    std::vector<int> idxs(desc->tri_indices, desc->tri_indices + desc->n_tris);

    /* Build cumulative-area CDF for area-weighted emissive sampling. */
    std::vector<double> cdf(idxs.size());
    double accum = 0.0;
    for (size_t i = 0; i < idxs.size(); ++i) {
        double a = (idxs[i] < (int)st->tri_areas.size())
                   ? st->tri_areas[(size_t)idxs[i]] : 1.0;
        accum += a;
        cdf[i] = accum;
    }
    st->tri_group_indices.push_back(std::move(idxs));
    st->tri_group_cum_areas.push_back(std::move(cdf));

    /* Group → default material.  Use desc->default_mat_idx if the caller
     * supplied >= 0; otherwise derive from majority across the group's
     * triangles (handles the common "one group per material" case). */
    int default_mat = desc->default_mat_idx;
    if (default_mat < 0 && !st->tri_group_indices.back().empty()) {
        std::map<int, int> hist;
        for (int t : st->tri_group_indices.back()) {
            if (t >= 0 && t < (int)st->tris.size())
                hist[st->tris[(size_t)t].mat_idx]++;
        }
        int best = -1, best_n = 0;
        for (auto& kv : hist) if (kv.second > best_n) { best = kv.first; best_n = kv.second; }
        default_mat = best;
    }
    st->tri_group_default_mat.push_back(default_mat);

    /* Per-band spectral power override (NULL pointer = no override). */
    std::vector<float> power_curve;
    if (desc->power_W_per_band && desc->n_power_bands > 0) {
        power_curve.assign(desc->power_W_per_band,
                           desc->power_W_per_band + desc->n_power_bands);
    }
    st->tri_group_power_per_band.push_back(std::move(power_curve));

    /* Camera sensor descriptor (deep-copy if present). */
    if (desc->sensor_camera) {
        st->tri_group_has_camera.push_back(1);
        st->tri_group_camera.push_back(*desc->sensor_camera);
    } else {
        st->tri_group_has_camera.push_back(0);
        st->tri_group_camera.push_back(CameraSensorDesc{});
    }

    std::vector<uint8_t> param_payload;
    if (desc->parametric_payload && desc->parametric_payload_bytes > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(desc->parametric_payload);
        param_payload.assign(p, p + desc->parametric_payload_bytes);
    }
    st->tri_group_parametric_kind.push_back(desc->parametric_surface_kind);
    st->tri_group_parametric_payload.push_back(std::move(param_payload));

    if (desc->parametric_surface_kind != TRI_PARAM_SURFACE_NONE) {
        const bool can_overwrite =
            (desc->parametric_surface_kind == TRI_PARAM_SURFACE_NEURAL_ASSEMBLY ||
             desc->parametric_surface_kind == TRI_PARAM_SURFACE_PARAMETRIC_LENS);
        for (int t : st->tri_group_indices.back()) {
            if (t >= 0 && t < (int)st->tri_param_group_of_tri.size()
                && (st->tri_param_group_of_tri[(size_t)t] < 0 || can_overwrite))
                st->tri_param_group_of_tri[(size_t)t] = copy.group_id;
        }
    }

    /* ── UV integrator image registration ──────────────────────────────── */
    st->group_uv_res.push_back(desc->uv_image_res);
    if (desc->uv_image_res > 0) {
        const int res          = desc->uv_image_res;
        const int n_total_tris = (int)st->tris.size();

        /* Ensure per-tri UV arrays cover the full triangle list. */
        if ((int)st->tri_uv_group_of_tri.size() < n_total_tris) {
            st->tri_uv_group_of_tri.resize(n_total_tris, -1);
            st->tri_uv_data.resize(n_total_tris * 6, 0.0f);
        }

        const std::vector<int>& tidx = st->tri_group_indices.back();
        const int n_gtris = (int)tidx.size();

        if (desc->uv_coords && desc->uv_n_coords == n_gtris * 6) {
            /* Explicit UV vertex coordinates provided (n_tris * 6 floats). */
            for (int gi = 0; gi < n_gtris; ++gi) {
                int t = tidx[(size_t)gi];
                if (t < 0 || t >= n_total_tris) continue;
                st->tri_uv_group_of_tri[(size_t)t] = copy.group_id;
                float* dst = st->tri_uv_data.data() + (size_t)t * 6;
                const float* src = desc->uv_coords + gi * 6;
                for (int k = 0; k < 6; ++k) dst[k] = src[k];
            }
        } else {
            /* Auto planar UV: project vertices onto the group's tangent plane,
             * normalise the bounding box in that plane to [0,1]×[0,1].       */
            V3d pn(desc->plane_normal[0], desc->plane_normal[1], desc->plane_normal[2]);
            if (pn.norm() < 1.0e-10) {
                /* No plane_normal: fall back to the geometric normal of first tri. */
                if (!tidx.empty()) {
                    int first = tidx[0];
                    if (first >= 0 && first < n_total_tris)
                        pn = st->tris[(size_t)first].normal;
                }
            }
            pn.normalize();

            /* Orthonormal tangent frame: right × up span the UV plane. */
            V3d world_up = (std::abs(pn.dot(V3d(0, 1, 0))) < 0.9)
                           ? V3d(0, 1, 0) : V3d(0, 0, 1);
            const V3d right = world_up.cross(pn).normalized();
            const V3d up    = pn.cross(right).normalized();

            /* First pass: bounding box in (right, up) coordinates. */
            double umin = 1.0e30, umax = -1.0e30;
            double vmin = 1.0e30, vmax = -1.0e30;
            for (int t : tidx) {
                if (t < 0 || t >= n_total_tris) continue;
                const Triangle& tri = st->tris[(size_t)t];
                const V3d verts[3] = { tri.v0, tri.v0 + tri.edge1, tri.v0 + tri.edge2 };
                for (const V3d& v : verts) {
                    double u = right.dot(v), vv = up.dot(v);
                    umin = std::min(umin, u); umax = std::max(umax, u);
                    vmin = std::min(vmin, vv); vmax = std::max(vmax, vv);
                }
            }
            const double uspan = (umax - umin > 1.0e-12) ? (umax - umin) : 1.0;
            const double vspan = (vmax - vmin > 1.0e-12) ? (vmax - vmin) : 1.0;

            /* Second pass: write normalised UV coords. */
            for (int t : tidx) {
                if (t < 0 || t >= n_total_tris) continue;
                st->tri_uv_group_of_tri[(size_t)t] = copy.group_id;
                const Triangle& tri = st->tris[(size_t)t];
                const V3d verts[3] = { tri.v0, tri.v0 + tri.edge1, tri.v0 + tri.edge2 };
                float* dst = st->tri_uv_data.data() + (size_t)t * 6;
                for (int k = 0; k < 3; ++k) {
                    dst[k * 2 + 0] = (float)((right.dot(verts[k]) - umin) / uspan);
                    dst[k * 2 + 1] = (float)((up.dot(verts[k]) - vmin) / vspan);
                }
            }
        }

        /* Allocate (UV_N_HDR_CHANNELS + 5*n_bands) × res × res uint32 slots. */
        const int uv_n_ch     = UV_N_HDR_CHANNELS + 5 * st->n_bands;
        const int slot_offset = st->uv_accum_total;
        st->group_uv_accum_offset.push_back(slot_offset);
        st->uv_accum_total += uv_n_ch * res * res;
        st->uv_accum_cpu.resize(st->uv_accum_total, 0u);
    } else {
        st->group_uv_accum_offset.push_back(0);
    }

    {
        float m = 0.0f;
        const auto& pp = st->tri_group_parametric_payload.back();
        if (pp.size() >= sizeof(float)) std::memcpy(&m, pp.data(), sizeof(float));
        st->gid_manifold_stats.emplace_back(m);
    }
    return copy.group_id;
}

extern "C" SK_API int ray_tracer_clear_tri_groups(RayTracerState* st)
{
    if (!st) return SK_ERR_NULL_STATE;
    st->tri_groups.clear();
    st->tri_group_indices.clear();
    st->tri_group_cum_areas.clear();
    st->tri_group_default_mat.clear();
    st->tri_group_power_per_band.clear();
    st->tri_group_has_camera.clear();
    st->tri_group_camera.clear();
    st->tri_group_parametric_kind.clear();
    st->tri_group_parametric_payload.clear();
    st->tri_param_group_of_tri.assign(st->tris.size(), -1);
    st->gid_manifold_stats.clear();
    /* UV integrator image state */
    st->group_uv_res.clear();
    st->group_uv_accum_offset.clear();
    st->tri_uv_group_of_tri.assign(st->tris.size(), -1);
    st->tri_uv_data.assign(st->tris.size() * 6, 0.0f);
    st->uv_accum_total = 0;
    st->uv_accum_cpu.clear();
    return SK_OK;
}

extern "C" SK_API int ray_tracer_n_tri_groups(const RayTracerState* st)
{
    if (!st) return 0;
    return static_cast<int>(st->tri_groups.size());
}

extern "C" SK_API int ray_tracer_get_manifold_gid_stats(
    const RayTracerState* st, int gid,
    float* out_magic, uint64_t* out_transmitted, uint64_t* out_absorbed)
{
    if (!st || gid < 0 || (size_t)gid >= st->gid_manifold_stats.size())
        return SK_ERR_NULL_STATE;
    const auto& s = st->gid_manifold_stats[(size_t)gid];
    if (out_magic)       *out_magic       = s.magic;
    if (out_transmitted) *out_transmitted = s.ok .load(std::memory_order_relaxed);
    if (out_absorbed)    *out_absorbed    = s.abs.load(std::memory_order_relaxed);
    return SK_OK;
}

extern "C" SK_API int ray_tracer_get_group_uv_image(
    const RayTracerState* st,
    int    group_id,
    float* out_channels,
    int*   out_res,
    int*   out_n_channels)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (group_id < 0 || group_id >= (int)st->group_uv_res.size())
        return SK_ERR_DIM_MISMATCH;
    const int res     = st->group_uv_res[(size_t)group_id];
    const int n_ch    = UV_N_HDR_CHANNELS + 5 * st->n_bands;
    const int n_texel = res * res;
    if (out_res)        *out_res        = res;
    if (out_n_channels) *out_n_channels = n_ch;
    if (res <= 0) return SK_ERR_DIM_MISMATCH;
    if (!out_channels) return SK_OK;

    const int offset = st->group_uv_accum_offset[(size_t)group_id];
    if (offset + n_ch * n_texel > (int)st->uv_accum_cpu.size())
        return SK_ERR_DIM_MISMATCH;
    const uint32_t* base = st->uv_accum_cpu.data() + offset;

    /* Channels 0-7: raw uint counts / bitfields → cast directly. */
    for (int ch = 0; ch < 8; ++ch)
        for (int i = 0; i < n_texel; ++i)
            out_channels[ch * n_texel + i] = (float)base[ch * n_texel + i];

    /* Channels 8-10: signed normal xyz, ×32768. */
    for (int ch = 8; ch <= 10; ++ch)
        for (int i = 0; i < n_texel; ++i)
            out_channels[ch * n_texel + i] =
                (float)(int32_t)base[ch * n_texel + i] / 32768.0f;

    /* Channels 11..11+B-1: per-band magnitude, unsigned ×65536. */
    const int nb = st->n_bands;
    for (int b = 0; b < nb; ++b)
        for (int i = 0; i < n_texel; ++i)
            out_channels[(11 + b) * n_texel + i] =
                (float)base[(11 + b) * n_texel + i] / 65536.0f;

    /* Channels 11+B..11+3B-1: per-band re/im, signed ×32768. */
    for (int b = 0; b < 2 * nb; ++b)
        for (int i = 0; i < n_texel; ++i)
            out_channels[(11 + nb + b) * n_texel + i] =
                (float)(int32_t)base[(11 + nb + b) * n_texel + i] / 32768.0f;

    /* Channels 11+3B..11+5B-1: forward/sensor magnitudes, unsigned ×65536. */
    for (int b = 0; b < 2 * nb; ++b)
        for (int i = 0; i < n_texel; ++i)
            out_channels[(11 + 3 * nb + b) * n_texel + i] =
                (float)base[(11 + 3 * nb + b) * n_texel + i] / 65536.0f;

    return SK_OK;
}

extern "C" SK_API int ray_tracer_set_group_uv_image(
    RayTracerState* st,
    int group_id,
    const float* channels,
    int res,
    int n_channels)
{
    if (!st || !channels) return SK_ERR_NULL_STATE;
    if (group_id < 0 || group_id >= (int)st->group_uv_res.size())
        return SK_ERR_DIM_MISMATCH;
    const int group_res = st->group_uv_res[(size_t)group_id];
    const int n_ch = UV_N_HDR_CHANNELS + 5 * st->n_bands;
    if (group_res <= 0 || res != group_res || n_channels != n_ch)
        return SK_ERR_DIM_MISMATCH;
    const int n_texel = res * res;
    const int offset = st->group_uv_accum_offset[(size_t)group_id];
    const int n_slot = n_ch * n_texel;
    if (offset < 0 || offset + n_slot > (int)st->uv_accum_cpu.size())
        return SK_ERR_DIM_MISMATCH;

    uint32_t* base = st->uv_accum_cpu.data() + offset;
    auto encode_unsigned = [](float v, double scale) -> uint32_t {
        if (!(v > 0.0f)) return 0u;
        const double x = std::min((double)v * scale, (double)0xFFFFFFFFu);
        return (uint32_t)std::llround(x);
    };
    auto encode_signed = [](float v, double scale) -> uint32_t {
        const double x = std::max(-(double)INT32_MAX,
                                  std::min((double)INT32_MAX, (double)v * scale));
        return (uint32_t)(int32_t)std::llround(x);
    };

    for (int ch = 0; ch < 8; ++ch) {
        for (int i = 0; i < n_texel; ++i)
            base[ch * n_texel + i] = encode_unsigned(channels[ch * n_texel + i], 1.0);
    }
    for (int ch = 8; ch <= 10; ++ch) {
        for (int i = 0; i < n_texel; ++i)
            base[ch * n_texel + i] = encode_signed(channels[ch * n_texel + i], 32768.0);
    }

    const int nb = st->n_bands;
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < n_texel; ++i)
            base[(11 + b) * n_texel + i] =
                encode_unsigned(channels[(11 + b) * n_texel + i], 65536.0);
    }
    for (int b = 0; b < 2 * nb; ++b) {
        for (int i = 0; i < n_texel; ++i)
            base[(11 + nb + b) * n_texel + i] =
                encode_signed(channels[(11 + nb + b) * n_texel + i], 32768.0);
    }
    for (int b = 0; b < 2 * nb; ++b) {
        for (int i = 0; i < n_texel; ++i)
            base[(11 + 3 * nb + b) * n_texel + i] =
                encode_unsigned(channels[(11 + 3 * nb + b) * n_texel + i], 65536.0);
    }
    return SK_OK;
}

extern "C" SK_API int ray_tracer_get_group_uv_summary(
    const RayTracerState* st,
    int group_id,
    RayTracerUvGroupSummary* out_summary)
{
    if (!st || !out_summary) return SK_ERR_NULL_STATE;
    if (group_id < 0 || group_id >= (int)st->group_uv_res.size())
        return SK_ERR_DIM_MISMATCH;

    const int res = st->group_uv_res[(size_t)group_id];
    const int nb = st->n_bands;
    const int n_ch = UV_N_HDR_CHANNELS + 5 * nb;
    const int n_texel = res * res;
    const int offset = st->group_uv_accum_offset[(size_t)group_id];
    const int n_slot = n_ch * n_texel;
    if (res <= 0 || offset < 0 || offset + n_slot > (int)st->uv_accum_cpu.size())
        return SK_ERR_DIM_MISMATCH;

    const uint32_t* base = st->uv_accum_cpu.data() + offset;
    uint64_t nonzero = 0;
    double total_forward = 0.0;
    double total_sensor = 0.0;
    double peak_total = 0.0;
    for (int i = 0; i < n_texel; ++i) {
        const uint32_t hits = base[UV_CH_HIT_COUNT * n_texel + i];
        if (hits != 0u) ++nonzero;
        double total = 0.0;
        for (int b = 0; b < nb; ++b) {
            const double fwd = (double)base[(11 + 3 * nb + b) * n_texel + i] / 65536.0;
            const double sen = (double)base[(11 + 4 * nb + b) * n_texel + i] / 65536.0;
            total_forward += fwd;
            total_sensor += sen;
            total += fwd + sen;
        }
        if (total > peak_total) peak_total = total;
    }

    RayTracerUvGroupSummary s{};
    s.group_id = group_id;
    s.res = res;
    s.n_channels = n_ch;
    s.tri_count = (group_id >= 0 && group_id < (int)st->tri_group_indices.size())
        ? (int)st->tri_group_indices[(size_t)group_id].size()
        : 0;
    s.memory_bytes = (uint64_t)n_slot * (uint64_t)sizeof(uint32_t);
    s.nonzero_texels = nonzero;
    s.total_forward = total_forward;
    s.total_sensor = total_sensor;
    s.peak_total = peak_total;
    *out_summary = s;
    return SK_OK;
}

extern "C" SK_API int ray_tracer_clear_group_uv_accum(RayTracerState* st, int group_id)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (group_id < 0) {
        /* Clear all groups. */
        std::fill(st->uv_accum_cpu.begin(), st->uv_accum_cpu.end(), 0u);
        return SK_OK;
    }
    if (group_id >= (int)st->group_uv_res.size()) return SK_ERR_DIM_MISMATCH;
    const int res    = st->group_uv_res[(size_t)group_id];
    if (res <= 0) return SK_OK;
    const int offset = st->group_uv_accum_offset[(size_t)group_id];
    const int n_slot = (UV_N_HDR_CHANNELS + 5 * st->n_bands) * res * res;
    if (offset + n_slot > (int)st->uv_accum_cpu.size()) return SK_ERR_DIM_MISMATCH;
    std::fill(st->uv_accum_cpu.begin() + offset,
              st->uv_accum_cpu.begin() + offset + n_slot, 0u);
    return SK_OK;
}

/*
 * Emits N rays from each EMISSIVE triangle group (area-weighted sampling
 * with a cosine-hemisphere distribution about the local normal), traces
 * them through the BVH with the same bounce loop the legacy splatting
 * integrator uses (so APERTURE_STOP / TRANSMISSIVE / REACTIVE handling is
 * shared), and writes one EndpointRecord per (subpath × band) every time
 * a ray hits a triangle that belongs to a SENSOR group.
 *
 * Hard rules of the rewrite are enforced HERE:
 *   - Complex amplitude is preserved verbatim (no abs(), no quantize).
 *   - n_bands is the MaterialDatabase value, NOT a layer-collapsed proxy.
 *   - Records carry subpath_id / band_id / group_id / vertex pdf so the
 *     Python display layer can reconstruct gain, phase, MTF, etc.
 */

/* Forward declarations for region-kind dispatch helpers. */
static inline void apply_thin_lens_transform(
    const RtScaleContext& ctx, V3d& pos, V3d& dir);
static inline void apply_thick_lens_wave_transform(
    const RayTracerState& st, const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp);
/* ── Unified bounce step for BDPT forward and backward paths ──────────
 * Encapsulates the complete physics of one bounce: intersection, 
 * attenuation, material response, and state updates. Both forward 
 * (emitter→scene) and backward (sensor→scene) paths call this identically.
 * No physics or material property code is simplified or duplicated.
 * 
 * RETURNS: struct with:
 *   hit_tri: triangle ID (-1 if miss)
 *   hit_pos: world position of hit
 *   hit_n_param: parametric surface normal at hit
 *   should_continue: true if ray should bounce again
 *   is_sensor_hit: true if hit triangle is a sensor endpoint (backward only)
 */
/* ── Causal bounce record: per-hit data for post-trace power propagation ─────
 * Stores structural information from one bounce step in causal order.
 * Both forward and backward paths may accumulate these into a path array,
 * enabling the power sustenance calculation ("how does emissive light
 * sustain its power back to where I came from") to be vectorized externally
 * on the record stream rather than computed inline per-bounce. */
struct BounceRecord {
    int          bounce_index;
    int          hit_tri;
    uint32_t     hit_tri_flags;
    V3d          hit_pos;
    V3d          hit_n_param;
    V3d          incoming_dir;
    uint32_t     interaction_flags;
    bool         is_emissive_hit;
    bool         is_sensor_hit;
    int          sensor_group_id;
};


static BounceStepResult ray_bounce_step_bdpt(
    RayTracerState& st,
    const int* tri_sensor_group,
    V3d& pos,
    V3d& dir,
    VXcd& amp,
    double& path_len,
    int& current_medium_mat_idx,
    uint32_t& interaction_flags,
    int n_bands,
    double min_amplitude,
    std::mt19937_64& rng,
    VXcd* amp_at_hit,
    bool is_backward)
{
    const bool has_bvh = !st.bvh_nodes.empty();
    BounceStepResult res;
    res.hit_tri           = -1;
    res.hit_tri_flags     = 0u;
    res.should_continue   = false;
    res.is_sensor_hit     = false;
    res.is_emissive_hit   = false;
    res.was_teleported    = false;
    res.sensor_group_id   = -1;
    res.has_illum_contrib = false;

    std::uniform_real_distribution<double> U(0.0, 1.0);

    /* ─ Intersection query ─ */
    double t_hit = 1e18;
    if (has_bvh) {
        V3d inv = dir.cwiseInverse();
        bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris, pos, dir, inv, t_hit, res.hit_tri);
    } else {
        for (size_t ti = 0; ti < st.tris.size(); ++ti) {
            double tt;
            if (ray_triangle_hit(pos, dir, st.tris[ti], tt) && tt > T_SELF && tt < t_hit) {
                t_hit = tt;
                res.hit_tri = (int)ti;
            }
        }
    }
    if (res.hit_tri < 0) {
        return res;  /* Miss → terminate */
    }

    /* ─ Compute hit point and attenuation ─ */
    res.hit_pos = pos + t_hit * dir;
    res.hit_n_param = st.tris[(size_t)res.hit_tri].normal;
    apply_parametric_surface_point(st, res.hit_tri, res.hit_pos, res.hit_pos, res.hit_n_param);
    double total_path = path_len + t_hit;
    
    /* ─ Phase + atmospheric attenuation along segment ─ */
    for (int b = 0; b < n_bands; ++b) {
        double n_re = 1.0, n_im = 0.0;
        if (current_medium_mat_idx >= 0) {
            n_re = mat_n_real(st, current_medium_mat_idx, b);
            n_im = mat_n_imag(st, current_medium_mat_idx, b);
            if (n_re < 1.0) n_re = 1.0;
        }
        double k_medium = st.k_real[b] * n_re;
        double alpha = st.atmo_abs[b];
        if (current_medium_mat_idx >= 0) {
            alpha += TWO_PI * st.freq_hz_vec[b] * n_im / st.speed_m_s;
        }
        double atten = std::exp(-alpha * t_hit);
        double spread = is_backward ? 1.0 : 1.0 / (1.0 + total_path);
        amp[b] *= std::polar(atten * spread, -k_medium * t_hit);
    }
    
    /* ─ Field capture ─ */
    if (st.camera_field_grid) {
        accumulate_field_capture_segment(st, pos, res.hit_pos, amp);
    }

    if (amp_at_hit) *amp_at_hit = amp;  /* post-propagation, pre-reflection */

    /* ─ Manifold surface dispatch (MLP / parametric / LUT) ─ */
    {
        int nb_mfld = 0;
        const float* mpp = tri_manifold_payload(st, res.hit_tri, &nb_mfld);
        if (mpp) {
            if (nb_mfld < (int)(8 * sizeof(float))) {
                /* Interior absorber — drop ray. */
                res.hit_tri      = -1;
                res.should_continue = false;
                return res;
            }
            /* pos is set to T1 proxy-mesh hit; for PARAMETRIC the
             * apply_* function now seeks to the exact s=0 conic entry. */
            const float magic = mpp[0];
            pos = res.hit_pos;
            bool absorbed = false;
            if (magic == 14949.0f) {
                absorbed = apply_parametric_lens_from_f32(st, mpp, nb_mfld, pos, dir, amp, &path_len, is_backward);
            } else if (magic == 14948.0f) {
                /* Deferred: MLP transport is disabled until the basic
                 * reversible parametric camera path is finished. */
                absorbed = true;
            } else if (magic == 14946.0f || magic == 14947.0f ||
                       magic == 14950.0f || magic == 14951.0f) {
                /* Deferred: LUT transport is disabled until it can supply
                 * the same reversible optical data as the parametric path. */
                absorbed = true;
            } else {
                absorbed = true;
            }
            {
                const int _gp = (res.hit_tri >= 0 &&
                                 (size_t)res.hit_tri < st.tri_param_group_of_tri.size())
                                ? st.tri_param_group_of_tri[(size_t)res.hit_tri] : -1;
                if (_gp >= 0 && (size_t)_gp < st.gid_manifold_stats.size()) {
                    auto& _ms = st.gid_manifold_stats[(size_t)_gp];
                    if (absorbed) _ms.abs.fetch_add(1, std::memory_order_relaxed);
                    else          _ms.ok .fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (absorbed) {
                res.hit_tri      = -1;
                res.should_continue = false;
                return res;
            }
            res.hit_pos         = pos;
            res.should_continue = true;
            res.was_teleported  = true;
            return res;
        }
    }

    /* ─ Material consequence kernel ────────────────────────────────────────────
     * tri is the authoritative source for all flag queries. Classify the hit
     * surface first; terminal conditions (aperture stop, emissive) return before
     * any surface physics so arrival amplitude is preserved unmodified.
     * Non-terminal surfaces receive full Fresnel/Snell/diffuse/specular physics. */
    const Triangle& tri = st.tris[(size_t)res.hit_tri];
    res.hit_tri_flags = tri.flags;
    V3d hit_n = res.hit_n_param;
    if (dir.dot(hit_n) > 0.0) hit_n = -hit_n;
    res.hit_n_transport = hit_n;

    /* ─ Sensor endpoint (recorded by caller; physics continues unless terminal) ─ */
    if (tri_sensor_group) {
        int sgid = tri_sensor_group[res.hit_tri];
        if (sgid >= 0) {
            res.is_sensor_hit = true;
            res.sensor_group_id = sgid;
        }
    }

    /* Aperture stop: ray terminates unconditionally */
    if (tri.flags & MAT_FLAG_APERTURE_STOP) {
        return res;
    }

    /* Emissive surface: arrival amplitude is the measurement; skip surface
     * response. path_len advanced so callers record the correct total path.
     * Forward path should not reach emissive mid-chain; backward path
     * terminates here — this is the light source the backward ray sought. */
    if (tri.flags & MAT_FLAG_EMISSIVE) {
        res.is_emissive_hit = true;
        path_len = total_path;  /* advance to emissive surface for EndpointRecord */
        return res;
    }

    const bool mat_transmissive = tri_material_is_transmissive(st, tri);
    const float bssrdf_d_frac   = mat_cache_diffusion(st.mat_cache, tri.mat_idx);

    /* ── BSSRDF forward: accumulate pre-interaction amplitude at diffuse-transmissive
     * surfaces so that backward paths can query it analytically.
     * Single-threaded BDPT path: uv_accum_add is atomic but contention is zero. */
    if (!is_backward && mat_transmissive && bssrdf_d_frac > 0.0f
        && !st.tri_illum_accum.empty())
    {
        const int nb_il  = st.tri_illum_nb;
        const int stride = 2 * nb_il + 2;
        const int base   = res.hit_tri * stride;
        if (nb_il > 0 && base >= 0 && base + stride <= (int)st.tri_illum_accum.size()) {
            const double cos_i_il = std::max(0.0, -dir.dot(hit_n));
            auto add_fp_il = [&](int off, double val) {
                const int32_t fp = (int32_t)std::max(-1073741824.0,
                    std::min(1073741824.0, val * 32768.0));
                uv_accum_add(st.tri_illum_accum[(size_t)(base + off)], (uint32_t)fp);
            };
            for (int b = 0; b < std::min(n_bands, nb_il); ++b) {
                add_fp_il(2 * b,     amp[b].real());
                add_fp_il(2 * b + 1, amp[b].imag());
            }
            add_fp_il(2 * nb_il, cos_i_il);
            uv_accum_add(st.tri_illum_accum[(size_t)(base + 2 * nb_il + 1)], 1u);
        }
    }

    /* ─ Material physics: transmission vs. reflection ─ */
    if (mat_transmissive) {
        /* ── BSSRDF backward: analytical illumination claim ────────────────────
         * A backward ray at a diffuse-transmissive surface analytically receives
         * illumination accumulated during the forward pass.  This replaces the
         * futile scatter-volume traversal for the diffuse fraction.
         *
         * Coupling weight = T_fresnel × diffuse_frac × avg_cos_forward / π
         *
         *   T_fresnel    power fraction entering the surface from the backward side
         *   diffuse_frac fraction of that power which was scattered (not refracted)
         *   avg_cos/π    Lambertian normalization for the illuminated surface
         *
         * The backward amplitude is scaled by (1 − diffuse_frac) before the
         * standard Snell/Fresnel block below so the coherent portion is not
         * double-counted. */
        if (is_backward && bssrdf_d_frac > 0.0f && !st.tri_illum_accum.empty()) {
            const int nb_il  = st.tri_illum_nb;
            const int stride = 2 * nb_il + 2;
            const int base   = res.hit_tri * stride;
            if (nb_il > 0 && base >= 0 && base + stride <= (int)st.tri_illum_accum.size()) {
                const int count = static_cast<int>(
                    st.tri_illum_accum[(size_t)(base + stride - 1)]);
                if (count > 0) {
                    const bool fface_bk = (dir.dot(res.hit_n_param) < 0.0);
                    int mfrom_bk = -1, mto_bk = -1;
                    tri_boundary_media(tri, fface_bk, mfrom_bk, mto_bk);
                    const double n1_bk = (mfrom_bk >= 0) ? medium_n_real(st, mfrom_bk)
                        : ((current_medium_mat_idx >= 0)
                           ? mat_n_real(st, current_medium_mat_idx) : 1.0);
                    const double n2_bk = (mto_bk >= 0) ? medium_n_real(st, mto_bk)
                        : (fface_bk ? mat_n_real(st, tri.mat_idx) : 1.0);
                    const double cos_i_bk = std::max(0.0, -dir.dot(hit_n));
                    V3d refr_bk;
                    const bool can_bk = snell_refract(dir, hit_n, n1_bk, n2_bk, refr_bk);
                    double T_bk = 0.0;
                    if (can_bk) {
                        const double sin2_bk  = (n1_bk/n2_bk)*(n1_bk/n2_bk)
                                                * (1.0 - cos_i_bk*cos_i_bk);
                        const double cos_t_bk = std::sqrt(std::max(0.0, 1.0 - sin2_bk));
                        T_bk = 1.0 - fresnel_R(cos_i_bk, cos_t_bk, n1_bk, n2_bk);
                    }
                    if (T_bk > 1e-9) {
                        const double cos_sum_bk = static_cast<double>(
                            static_cast<int32_t>(
                                st.tri_illum_accum[(size_t)(base + 2*nb_il)])) / 32768.0;
                        const double avg_cos_bk = cos_sum_bk / static_cast<double>(count);
                        const double coupling   = T_bk * static_cast<double>(bssrdf_d_frac)
                                                  * avg_cos_bk / M_PI;
                        const double inv_n_bk   = 1.0 / static_cast<double>(count);
                        res.has_illum_contrib   = true;
                        res.illum_contrib       = VXcd::Zero(n_bands);
                        for (int b = 0; b < std::min(n_bands, nb_il); ++b) {
                            const double re_il = static_cast<double>(
                                static_cast<int32_t>(
                                    st.tri_illum_accum[(size_t)(base + 2*b)])) / 32768.0;
                            const double im_il = static_cast<double>(
                                static_cast<int32_t>(
                                    st.tri_illum_accum[(size_t)(base + 2*b+1)])) / 32768.0;
                            const cd illum_avg_bk(re_il * inv_n_bk, im_il * inv_n_bk);
                            res.illum_contrib[b] = amp[b] * coupling * illum_avg_bk;
                        }
                    }
                    /* Scale backward amplitude to coherent fraction only. */
                    const double scale_bk = 1.0 - static_cast<double>(bssrdf_d_frac);
                    for (int b = 0; b < n_bands; ++b) amp[b] *= scale_bk;
                }
            }
        }

        /* Fresnel refraction / TIR for glass surfaces */
        const bool front_face = (dir.dot(res.hit_n_param) < 0.0);
        int medium_from = -1, medium_to = -1;
        const bool has_side_pair = tri_boundary_media(tri, front_face, medium_from, medium_to);
        const double n1 = has_side_pair
            ? medium_n_real(st, medium_from)
            : ((current_medium_mat_idx >= 0) ? mat_n_real(st, current_medium_mat_idx) : 1.0);
        const double n2 = has_side_pair
            ? medium_n_real(st, medium_to)
            : (front_face ? mat_n_real(st, tri.mat_idx) : 1.0);
        const double cos_i = std::max(0.0, -dir.dot(hit_n));
        V3d refracted;
        const bool can_refract = snell_refract(dir, hit_n, n1, n2, refracted);
        if (!can_refract) {
            /* TIR — reflect */
            dir = (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized();
            for (int b = 0; b < n_bands; ++b)
                amp[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
        } else {
            const double sin2_t = (n1 / n2) * (n1 / n2) * (1.0 - cos_i * cos_i);
            const double cos_t = std::sqrt(std::max(0.0, 1.0 - sin2_t));
            const double R = fresnel_R(cos_i, cos_t, n1, n2);
            if (U(rng) < R) {
                /* Reflect (Fresnel) */
                dir = (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized();
                for (int b = 0; b < n_bands; ++b)
                    amp[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
            } else {
                /* Refract through — no amplitude rescale; probability already encodes it */
                dir = refracted;
                if (has_side_pair)
                    current_medium_mat_idx = medium_to;
                else
                    current_medium_mat_idx = (current_medium_mat_idx == tri.mat_idx) ? -1 : tri.mat_idx;
            }
        }
    } else if (U(rng) < mat_cache_diffusion(st.mat_cache, tri.mat_idx)) {
        /* Diffuse scatter */
        dir = cosine_hemisphere(hit_n, rng);
        for (int b = 0; b < n_bands; ++b)
            amp[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
    } else {
        /* Specular reflection */
        dir = (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized();
        if (dir.dot(hit_n) < 0.0) dir = cosine_hemisphere(hit_n, rng);
        for (int b = 0; b < n_bands; ++b)
            amp[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
    }
    
    /* ─ Reactive material shifts ─ */
    if (tri.flags & MAT_FLAG_REACTIVE) {
        apply_reactive_shift_cached(amp, st.mat_cache, tri.mat_idx);
    }
    
    /* ─ Update ray state ─ */
    pos = res.hit_pos;
    path_len = total_path;
    
    /* ─ Region dispatch (scale contexts) ─ */
    for (const RtScaleContext& ctx : st.scale_contexts) {
        V3d c(ctx.center[0], ctx.center[1], ctx.center[2]);
        if ((pos - c).norm() <= ctx.radius) {
            interaction_flags |= dispatch_scale_context_entry(st, ctx, pos, dir, amp, &path_len);
        }
    }
    
    /* ─ Energy threshold ─ */
    double max_abs = 0.0;
    for (int b = 0; b < n_bands; ++b)
        max_abs = std::max(max_abs, std::abs(amp[b]));
    
    res.should_continue = (max_abs >= min_amplitude);
    return res;
}

static int ray_tracer_bidirectional_impl(
    RayTracerState* st,
    int             n_rays_per_emitter,
    const int32_t*  packed_n_rays,
    int             packed_n_emitters,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    EndpointRecord* out_records,
    int             out_cap,
    int*            out_count)
{
    if (!st || !out_records || !out_count) return SK_ERR_NULL_STATE;
    *out_count = 0;
    if (st->tri_groups.empty()) return SK_OK;

    const int n_bands = st->n_bands;
    const bool has_bvh = !st->bvh_nodes.empty();

    std::mt19937_64 rng(static_cast<uint64_t>(seed) * 6364136223846793005ULL
                        + 1442695040888963407ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    VXcd amp(n_bands);

    /* Build sensor-group lookup: tri_id → first sensor group_id (-1 if none). */
    std::vector<int> tri_sensor_group(st->tris.size(), -1);
    for (size_t g = 0; g < st->tri_groups.size(); ++g) {
        if (!(st->tri_groups[g].role_bits & TRI_GROUP_ROLE_SENSOR)) continue;
        const auto& idxs = st->tri_group_indices[g];
        for (int t : idxs) {
            if (t >= 0 && t < (int)tri_sensor_group.size() && tri_sensor_group[t] < 0)
                tri_sensor_group[t] = static_cast<int>(g);
        }
    }

    int rec_count = 0;
    uint32_t subpath_counter = 0;

    int emissive_idx = 0;

    /* For each emissive group … */
    for (size_t g = 0; g < st->tri_groups.size(); ++g) {
        const TriGroupDesc& gd = st->tri_groups[g];
        if (!(gd.role_bits & TRI_GROUP_ROLE_EMISSIVE)) continue;

        int n_rays_this = n_rays_per_emitter;
        if (packed_n_rays && emissive_idx < packed_n_emitters) {
            n_rays_this = std::max(0, static_cast<int>(packed_n_rays[emissive_idx]));
        }
        emissive_idx += 1;
        if (n_rays_this <= 0) {
            continue;
        }

        const auto& idxs = st->tri_group_indices[g];
        const auto& cdf  = st->tri_group_cum_areas[g];
        if (idxs.empty()) continue;
        const double tot_area = cdf.empty() ? 0.0 : cdf.back();
        if (tot_area <= 0.0) continue;

        /* Initial per-band amplitude: unit (1+0j); the calibration loop
         * scales the result via Python.  We are preserving phase, so we
         * cannot pre-scale by emit_W without losing complex coherence
         * across bands. */
        for (int ri = 0; ri < n_rays_this; ++ri) {
            /* 1. Pick a triangle (area-weighted). */
            double r = U(rng) * tot_area;
            int lo = 0, hi = (int)cdf.size() - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (cdf[mid] < r) lo = mid + 1; else hi = mid;
            }
            int tri_id = idxs[lo];
            const Triangle& T = st->tris[tri_id];

            /* 2. Pick a barycentric origin uniformly in the triangle. */
            double s = U(rng), t = U(rng);
            if (s + t > 1.0) { s = 1.0 - s; t = 1.0 - t; }
            V3d origin = T.v0 + s * T.edge1 + t * T.edge2;

            V3d emit_n = T.normal;
            apply_parametric_surface_point(*st, tri_id, origin, origin, emit_n);

            /* 3. Pick an outgoing direction: cosine-hemisphere about normal. */
            V3d dir = cosine_hemisphere(emit_n, rng);

            const int ray_band = (n_bands > 0) ? (static_cast<int>(subpath_counter % static_cast<uint32_t>(n_bands))) : 0;
            for (int b = 0; b < n_bands; ++b)
                amp[b] = (b == ray_band) ? cd(1.0, 0.0) : cd(0.0, 0.0);

            V3d pos = origin + dir * (EPS * 200.0);
            double path_len = 0.0;
            uint32_t my_subpath = subpath_counter++;
            uint32_t interaction_flags = 0u;
            int current_medium_mat_idx = -1; /* tracks which material the ray is inside */

            /* Launch-context dispatch: apply region transforms to emission
             * rays immediately after post-triangulated intercept prep. */
            for (const RtScaleContext& ctx : st->scale_contexts) {
                V3d c(ctx.center[0], ctx.center[1], ctx.center[2]);
                if ((pos - c).norm() <= ctx.radius) {
                    interaction_flags |= dispatch_scale_context_entry(*st, ctx, pos, dir, amp, &path_len);
                }
            }

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                /* ─ Unified bounce physics (shared with backward path) ─ */
                BounceStepResult step = ray_bounce_step_bdpt(
                    *st, tri_sensor_group.data(), pos, dir, amp, path_len,
                    current_medium_mat_idx, interaction_flags, n_bands, min_amplitude, rng,
                    nullptr, false);

                if (step.hit_tri < 0) {
                    break;  /* Miss */
                }
                
                /* ─ Forward-path-specific: legacy visualization callback ─ */
                append_camera_strike(
                    *st,
                    static_cast<int>(g),
                    bounce,
                    step.hit_tri,
                    step.hit_pos,
                    step.hit_n_param,
                    dir,
                    path_len,
                    path_len,
                    1,
                    0,
                    amp);
                
                /* ─ Forward-path-specific: endpoint recording ─ */
                {
                    V3d hit_n = st->tris[step.hit_tri].normal;
                    double cos_theta = std::abs(dir.dot(hit_n));
                    /* Sensor plate hits: full endpoint with sensor group_id so
                     * the reducer can project them to pixels.
                     * All other scene hits: record with group_id=-1 so the
                     * reducer ignores them, but the correlator can project them
                     * backward through the aperture to find fwd-half pairings. */
                    const int32_t rec_gid = step.is_sensor_hit
                        ? static_cast<int32_t>(step.sensor_group_id)
                        : int32_t(-1);
                    const double min_amp2 = min_amplitude * min_amplitude;
                    for (int b = 0; b < n_bands; ++b) {
                        const double ar = amp[b].real();
                        const double ai = amp[b].imag();
                        if (ar * ar + ai * ai <= min_amp2) {
                            continue;
                        }
                        if (rec_count >= out_cap) goto bdpt_done;
                        EndpointRecord& E = out_records[rec_count++];
                        E.subpath_id   = my_subpath;
                        E.band_id      = static_cast<uint32_t>(b);
                        E.group_id     = rec_gid;
                        E.vertex_index = -1 - static_cast<int32_t>(bounce);
                        E.pos[0] = (float)step.hit_pos.x();
                        E.pos[1] = (float)step.hit_pos.y();
                        E.pos[2] = (float)step.hit_pos.z();
                        E.pathlen_m = (float)path_len;
                        E.dir[0] = (float)dir.x();
                        E.dir[1] = (float)dir.y();
                        E.dir[2] = (float)dir.z();
                        E.pdf       = 1.0f / (float)std::max(1, n_rays_this);
                        E.amp_re    = (float)ar;
                        E.amp_im    = (float)ai;
                        E.cos_theta = (float)cos_theta;
                        E.stream_id = (float)BDPT_SIDE_LIGHT;
                    }
                }

                /* ─ Check if ray should continue bouncing ─ */
                if (!step.should_continue) break;
            }
        }
    }

    /* ── Backward pixel-cone pass ─────────────────────────────────────────────
     * For each SENSOR group that carries a CameraSensorDesc, launch one
     * backward ray per (pixel, aperture-sample).  Rays trace from the sensor
     * plane through the aperture into the scene; when a backward ray reaches
     * an emissive surface the contribution is deposited as an EndpointRecord
     * keyed to the originating pixel:
     *   subpath_id  = py * n_px + px
     *   vertex_index = 0   (non-negative → PIXEL_CONE path, not a forward hit)
     *   group_id    = sensor group id  (so the reducer keeps it)
     *   stream_id   = BDPT_SIDE_SENSOR
     * The reducer in reduce_endpoint_records_to_rgb_image routes these records
     * straight to pixels without any position-projection step.              */
    for (size_t g = 0; g < st->tri_groups.size(); ++g) {
        const TriGroupDesc& sgd = st->tri_groups[g];
        if (!(sgd.role_bits & TRI_GROUP_ROLE_SENSOR)) continue;
        if (g >= st->tri_group_has_camera.size() || !st->tri_group_has_camera[g]) continue;

        const CameraSensorDesc& cam = st->tri_group_camera[g];
        const int n_px = cam.n_px;
        const int n_py = cam.n_py;
        const int n_ap = std::max(1, cam.n_aperture_samples);
        if (n_px <= 0 || n_py <= 0) continue;
        if (cam.sensor_w_m <= 0.0 || cam.sensor_h_m <= 0.0) continue;

        /* Build orthonormal camera basis */
        V3d cpos(cam.pos[0], cam.pos[1], cam.pos[2]);
        V3d cfwd(cam.fwd[0], cam.fwd[1], cam.fwd[2]);
        V3d cup (cam.up[0],  cam.up[1],  cam.up[2]);
        if (cfwd.norm() < 1.0e-12 || cup.norm() < 1.0e-12) continue;
        cfwd.normalize();
        cup.normalize();
        V3d cright = cfwd.cross(cup).normalized();
        cup = cright.cross(cfwd).normalized();

        /* Aperture disk centre and radius */
        const double ap_r  = cam.aperture_radius_m;
        const V3d    ap_cen = cpos + cam.focal_m * cfwd;

        const double pix_w = cam.sensor_w_m / n_px;
        const double pix_h = cam.sensor_h_m / n_py;
        const V3d sensor_origin = cpos
            - 0.5 * cam.sensor_w_m * cright
            - 0.5 * cam.sensor_h_m * cup;

        for (int bpy = 0; bpy < n_py; ++bpy) {
            for (int bpx = 0; bpx < n_px; ++bpx) {
                const V3d pix_center = sensor_origin
                    + (bpx + 0.5) * pix_w * cright
                    + (bpy + 0.5) * pix_h * cup;
                const uint32_t pix_id = static_cast<uint32_t>(bpy * n_px + bpx);

                for (int ap = 0; ap < n_ap; ++ap) {
                    /* Uniform disk sample for aperture point */
                    const double r2 = U(rng);
                    const double th = 2.0 * M_PI * U(rng);
                    const double r  = (ap_r > 0.0) ? std::sqrt(r2) * ap_r : 0.0;
                    const V3d ap_pt = ap_cen
                        + r * std::cos(th) * cright
                        + r * std::sin(th) * cup;

                    V3d bdir = ap_pt - pix_center;
                    const double blen = bdir.norm();
                    if (blen < 1.0e-12) continue;
                    bdir /= blen;

                    const int ray_band = (n_bands > 0)
                        ? static_cast<int>(subpath_counter % static_cast<uint32_t>(n_bands))
                        : 0;
                    for (int b = 0; b < n_bands; ++b)
                        amp[b] = (b == ray_band) ? cd(1.0, 0.0) : cd(0.0, 0.0);

                    V3d bpos = pix_center + bdir * (EPS * 200.0);
                    double bpath_len = 0.0;
                    uint32_t bflags = 0u;
                    int bmedium = -1;
                    subpath_counter++;

                    for (int bounce = 0; bounce < max_bounces; ++bounce) {
                        BounceStepResult step = ray_bounce_step_bdpt(
                            *st, tri_sensor_group.data(),
                            bpos, bdir, amp, bpath_len,
                            bmedium, bflags,
                            n_bands, min_amplitude, rng,
                            nullptr, true);  /* is_backward=true */

                        if (step.hit_tri < 0) break;

                        /* Parametric lens teleport: ray has been redirected to the
                         * scene side.  Continue bouncing — the next hit is the
                         * genuine scene vertex we want as the backward endpoint. */
                        if (step.was_teleported) {
                            bpos = step.hit_pos;
                            if (!step.should_continue) break;
                            continue;
                        }

                        /* Record first real scene-side hit as backward endpoint.
                         * Skip hits on the sensor itself (ray leaving the sensor). */
                        if (!step.is_sensor_hit) {
                            const V3d hit_n = st->tris[step.hit_tri].normal;
                            const double cos_theta = std::abs(bdir.dot(hit_n));
                            const double min_amp2 = min_amplitude * min_amplitude;
                            for (int b = 0; b < n_bands; ++b) {
                                const double ar = amp[b].real();
                                const double ai = amp[b].imag();
                                if (ar * ar + ai * ai <= min_amp2) {
                                    continue;
                                }
                                if (rec_count >= out_cap) goto bdpt_done;
                                EndpointRecord& E = out_records[rec_count++];
                                E.subpath_id   = pix_id;
                                E.band_id      = static_cast<uint32_t>(b);
                                E.group_id     = static_cast<int32_t>(g);
                                E.vertex_index = 0;
                                E.pos[0]       = (float)step.hit_pos.x();
                                E.pos[1]       = (float)step.hit_pos.y();
                                E.pos[2]       = (float)step.hit_pos.z();
                                E.pathlen_m    = (float)bpath_len;
                                E.dir[0]       = (float)bdir.x();
                                E.dir[1]       = (float)bdir.y();
                                E.dir[2]       = (float)bdir.z();
                                E.pdf          = 1.0f / static_cast<float>(n_ap);
                                E.amp_re       = (float)ar;
                                E.amp_im       = (float)ai;
                                E.cos_theta    = (float)cos_theta;
                                E.stream_id    = (float)BDPT_SIDE_SENSOR;
                            }
                            break;  /* one scene-side endpoint per backward subpath */
                        }

                        if (!step.should_continue) break;
                    }
                }
            }
        }
    }
bdpt_done:

    *out_count = rec_count;
    return SK_OK;
}

extern "C" SK_API void ray_tracer_attach_optical_assembly(RayTracerState* st, OpticalAssembly* assembly)
{
    if (!st) return;
    st->optical_assembly       = assembly;
    st->optical_assembly_owned = 0; /* never owned; caller manages lifetime */
}

extern "C" SK_API int ray_tracer_bidirectional(
    RayTracerState* st,
    int             n_rays_per_emitter,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    EndpointRecord* out_records,
    int             out_cap,
    int*            out_count)
{
    return ray_tracer_bidirectional_impl(
        st,
        n_rays_per_emitter,
        nullptr,
        0,
        max_bounces,
        min_amplitude,
        seed,
        out_records,
        out_cap,
        out_count);
}

extern "C" SK_API int ray_tracer_bidirectional_packed(
    RayTracerState* st,
    const int32_t*  n_rays_per_emitter,
    int             n_emitters,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    EndpointRecord* out_records,
    int             out_cap,
    int*            out_count)
{
    if (!n_rays_per_emitter || n_emitters <= 0) return SK_ERR_NULL_STATE;
    return ray_tracer_bidirectional_impl(
        st,
        0,
        n_rays_per_emitter,
        n_emitters,
        max_bounces,
        min_amplitude,
        seed,
        out_records,
        out_cap,
        out_count);
}

/* ── BDPT connection step ────────────────────────────────────────────────────
 * For every backward (sensor) vertex b and every sampled forward (light)
 * vertex f in the same spectral band, cast a shadow ray.  If the segment
 * b.pos → f.pos is unoccluded, the contribution
 *   |b.amp| * |f.amp| / dist²
 * is deposited into the pixel encoded in b.subpath_id.
 * max_fwd_samples caps the number of forward vertices tested per backward
 * vertex; set to 0 to use all forward vertices.                            */
extern "C" SK_API int ray_tracer_bdpt_connect(
    const RayTracerState* st,
    const EndpointRecord* records,
    int                   n_records,
    int                   n_px,
    int                   n_py,
    int                   sensor_gid,
    int                   max_fwd_samples,
    uint32_t              seed,
    float*                out_rgb,   /* n_px * n_py * 3, pre-zeroed by caller */
    int                   n_rgb)
{
    if (!st || !records || !out_rgb) return SK_ERR_NULL_STATE;
    if (n_px <= 0 || n_py <= 0)     return SK_ERR_DIM_MISMATCH;

    const int n_pixels = n_px * n_py;
    const int n_bands  = std::max(1, st->n_bands);

    /* Separate fwd / bwd by stream_id. */
    std::vector<const EndpointRecord*> fwd_recs, bwd_recs;
    fwd_recs.reserve(static_cast<size_t>(n_records));
    bwd_recs.reserve(static_cast<size_t>(n_records));
    for (int i = 0; i < n_records; ++i) {
        const EndpointRecord& E = records[i];
        if (E.stream_id < 0.5f) fwd_recs.push_back(&E);
        else                    bwd_recs.push_back(&E);
    }
    if (fwd_recs.empty() || bwd_recs.empty()) return SK_OK;

    /* Band-bucket the forward records. */
    std::vector<std::vector<const EndpointRecord*>> fwd_by_band(
        static_cast<size_t>(n_bands));
    for (const EndpointRecord* f : fwd_recs) {
        const int b = static_cast<int>(f->band_id);
        if (b >= 0 && b < n_bands) fwd_by_band[b].push_back(f);
    }

    /* Wavelength → RGB using CIE-approximate spectral locus. */
    auto band_to_rgb = [&](int b, double& wr, double& wg, double& wb) {
        if (n_bands == 1 || b < 0 || b >= n_bands) { wr = wg = wb = 1.0; return; }
        double wl_nm = 550.0;
        if (b < (int)st->freq_hz_vec.size() && st->freq_hz_vec[b] > 0.0) {
            constexpr double C = 299792458.0;
            wl_nm = std::max(380.0, std::min(700.0, (C / st->freq_hz_vec[b]) * 1.0e9));
        } else {
            double t = static_cast<double>(b) / static_cast<double>(n_bands - 1);
            wl_nm = 380.0 + t * 320.0;
        }
        wr = wg = wb = 0.0;
        if      (wl_nm < 440.0) { wr = -(wl_nm-440.0)/60.0; wb = 1.0; }
        else if (wl_nm < 490.0) { wg =  (wl_nm-440.0)/50.0; wb = 1.0; }
        else if (wl_nm < 510.0) { wg = 1.0; wb = -(wl_nm-510.0)/20.0; }
        else if (wl_nm < 580.0) { wr =  (wl_nm-510.0)/70.0; wg = 1.0; }
        else if (wl_nm < 645.0) { wr = 1.0; wg = -(wl_nm-645.0)/65.0; }
        else                    { wr = 1.0; }
        double edge = 1.0;
        if      (wl_nm < 420.0) edge = 0.3 + 0.7*(wl_nm-380.0)/40.0;
        else if (wl_nm > 645.0) edge = 0.3 + 0.7*(700.0-wl_nm)/55.0;
        wr *= edge; wg *= edge; wb *= edge;
    };

    std::mt19937 rng(seed ^ 0xBD97u);
    std::vector<double> accum(static_cast<size_t>(n_pixels) * 3u, 0.0);

    for (const EndpointRecord* bptr : bwd_recs) {
        const int pid = static_cast<int>(bptr->subpath_id);
        if (pid < 0 || pid >= n_pixels) continue;

        const int band = static_cast<int>(bptr->band_id);
        if (band < 0 || band >= n_bands) continue;

        const V3d b_pos(bptr->pos[0], bptr->pos[1], bptr->pos[2]);
        const double b_amp = static_cast<double>(std::hypot(bptr->amp_re, bptr->amp_im));

        /* Use the same-band forward pool — the backward ray already sampled
         * this wavelength; match spectrally for coherent transport. */
        const auto& fpool = fwd_by_band[band];
        const int n_f = static_cast<int>(fpool.size());
        if (n_f == 0) continue;

        const bool do_sample = (max_fwd_samples > 0 && max_fwd_samples < n_f);
        const int  n_test    = do_sample ? max_fwd_samples : n_f;
        const double inv_n   = 1.0 / static_cast<double>(n_test);

        double wr, wg, wb;
        band_to_rgb(band, wr, wg, wb);

        const size_t base = static_cast<size_t>(pid) * 3u;

        for (int si = 0; si < n_test; ++si) {
            int fi = do_sample
                ? static_cast<int>(rng() % static_cast<uint32_t>(n_f))
                : si;
            const EndpointRecord* fptr = fpool[fi];

            const V3d f_pos(fptr->pos[0], fptr->pos[1], fptr->pos[2]);
            V3d seg = f_pos - b_pos;
            double dist = seg.norm();
            if (dist < EPS * 200.0) continue;
            V3d seg_dir = seg / dist;

            V3d orig   = b_pos + seg_dir * (EPS * 400.0);
            double t_c = dist - EPS * 800.0;
            if (t_c <= 0.0) continue;
            double t_hit = t_c;
            int hit_tri  = -1;
            segment_first_hit(*st, orig, seg_dir, t_c, t_hit, hit_tri);
            if (hit_tri >= 0) continue;

            const double f_amp = static_cast<double>(std::hypot(fptr->amp_re, fptr->amp_im));
            const double w = b_amp * f_amp * inv_n / (dist * dist + 1.0e-6);

            accum[base + 0] += w * wr;
            accum[base + 1] += w * wg;
            accum[base + 2] += w * wb;
        }
    }

    double peak = 0.0;
    for (double v : accum) peak = std::max(peak, v);
    if (peak > 0.0) {
        for (int i = 0; i < n_pixels * 3; ++i)
            out_rgb[i] = static_cast<float>(accum[i] / peak);
    }
    return SK_OK;
}

extern "C" SK_API int ray_tracer_accumulate_endpoint_records_to_field_capture(
    RayTracerState*       st,
    const EndpointRecord* records,
    int                   n_records,
    int                   sensor_group_id,
    int                   include_sensor_group,
    int                   include_non_sensor_groups,
    int*                  out_written_records,
    double*               out_written_power)
{
    if (!st || !records) return SK_ERR_NULL_STATE;
    if (n_records < 0) return SK_ERR_DIM_MISMATCH;
    if (!st->camera_field_grid) return SK_ERR_DIM_MISMATCH;

    const int n_bands = st->n_bands;
    if (n_bands <= 0) return SK_ERR_DIM_MISMATCH;

    int written = 0;
    double power_sum = 0.0;

    for (int i = 0; i < n_records; ++i) {
        const EndpointRecord& E = records[static_cast<size_t>(i)];
        const bool is_sensor = (sensor_group_id >= 0) && (static_cast<int>(E.group_id) == sensor_group_id);
        if (is_sensor && !include_sensor_group) continue;
        if (!is_sensor && !include_non_sensor_groups) continue;

        const int band = static_cast<int>(E.band_id);
        if (band < 0 || band >= n_bands) continue;
        if (!(E.pdf > 0.0f)) continue;

        const float pos[3] = { E.pos[0], E.pos[1], E.pos[2] };
        const float ar = E.amp_re;
        const float ai = E.amp_im;
        const int rc = field_grid_inject_amplitude(
            st->camera_field_grid,
            band,
            pos,
            ar,
            ai);
        if (rc == SK_OK) {
            written += 1;
            const double dr = static_cast<double>(ar);
            const double di = static_cast<double>(ai);
            power_sum += dr * dr + di * di;
        }
    }

    if (out_written_records) *out_written_records = written;
    if (out_written_power) *out_written_power = power_sum;
    return SK_OK;
}

/* ──────────────────────────────────────────────────────────────────────
 * Region-context dispatch helpers.
 *
 * These are the call sites for SCALE_CONTEXT_KIND_* dispatch.  RAY (0) is
 * the default and means "do nothing, continue with normal ray transport."
 * THIN_LENS_TRANSFORM (2) applies an ABCD-style direction tilt about the
 * region centre; the region's `payload` is interpreted as
 *   const double matrix_optics[4] = { f_m, 0, 0, 0 };  // simplest case
 * with f_m = focal length.  Full ABCD support is queued; this lands the
 * dispatch site so the bounce loop can already call it.
 *
 * WAVE_HELMHOLTZ (1), THICK_LENS_WAVE (3), SPLINE_SURFACE (4),
 * NEURAL_SURFACE (5) decodes a float32 transfer-grid payload built by
 * ManifoldEndpoint.build_transfer_grid() and bilinearly interpolates the
 * baked output direction, skipping detailed per-element lens geometry.
 * NEURAL_VOLUMETRIC (6) remains a stub-passthrough.
 * ────────────────────────────────────────────────────────────────────── */

/* apply_manifold_transfer — redirect dir via a baked float32 transfer grid.
 *
 * Payload layout (float32):
 *   V1 header [0..7]: magic=14946, n_u, n_v, u_min, u_max, v_min, v_max, r_ap
 *   V1 cells  [8..]: (n_v × n_u) × 7 floats — out_dx, out_dy, out_dz, opl,
 *                    count, in_dx, in_dy  (row-major [iv, iu])
 *
 *   V2 header [0..11]: magic=14947, n_u, n_v, u_min, u_max, v_min, v_max,
 *                      r_ap, z_sensor, focus_z, z_front_ref, reserved
 *   V2 cells  [12..]: (n_v × n_u) × 9 floats — V1 fields plus sensor_x,
 *                     sensor_y.  V2 teleports the ray to just before the
 *                     baked sensor hit so the next BVH step can strike the
 *                     sensor without traversing the lens geometry again.
 *
 *   V3 header [0..15]: magic=14950, n_u, n_v, n_a, n_b,
 *                      u_min, u_max, v_min, v_max,
 *                      a_min, a_max, b_min, b_max, r_ap, axis_idx, reserved
 *   V3 cells  [16..]: (n_v × n_u × n_b × n_a) × 7 floats — V1 fields, but
 *                     angle-resolved by incoming transverse direction.
 *
 *   V4 header/cells: magic=14951, same as V3 but 9-float cells.  Header[15]
 *                    stores the destination axis coordinate for teleport.
 */
/* kill_ray_amplitudes — zero all band amplitudes so the tracer's min_amplitude
 * check terminates this ray on the next step. */
static inline void kill_ray_amplitudes(VXcd& amp)
{
    for (int b = 0; b < (int)amp.size(); ++b)
        amp[b] = cd(0.0, 0.0);
}

static inline void apply_manifold_transfer(
    const RayTracerState& st,
    const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len)
{
    (void)st; (void)ctx; (void)pos; (void)dir; (void)path_len;
    /* Deferred: LUT/spline/neural manifold transport is disabled while the
     * basic reversible camera path is being completed.  Re-enable only with
     * OPL, Jacobian, eta/cosine, and valid forward/reverse PDF records. */
    kill_ray_amplitudes(amp);
}

/* apply_neural_mlp_from_f32 -- deferred MLP transport.
 * Intention: restore only after the transform can report reversible optical
 * metadata: direction mapping, OPL, eta/cosines, Jacobian, and PDFs. */
static inline void apply_neural_mlp_from_f32(
    const RayTracerState& st,
    const float* p, int payload_bytes,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len)
{
    (void)st; (void)p; (void)payload_bytes; (void)pos; (void)dir; (void)path_len;
    /* Deferred: the MLP surface transport is disabled until the basic camera
     * path has a complete reversible optical contract.  When restored, this
     * must emit the same data as the parametric lens path: direction mapping,
     * OPL, eta/cosines, Jacobian, and valid forward/reverse PDFs. */
    kill_ray_amplitudes(amp);
}

/* ── apply_parametric_lens_from_f32 ─────────────────────────────────────────
 * CPU equivalent of GPU T2 parametric_lens_teleport().
 * Reads CompoundLens.build_gpu_payload() float32 payload (magic 14949),
 * evaluates exact conic intersection + vector Snell at each surface, and
 * teleports pos/dir to the assembly exit.  Returns true if the ray was
 * absorbed (vignetted, TIR, degenerate), false if teleported successfully.
 * Optical axis is scene-X (same convention as CompoundLens). */
#pragma optimize("", off)  /* MSVC ICE workaround: optimizer overflows on this function */
static inline bool apply_parametric_lens_from_f32(
    const RayTracerState& st,
    const float* p, int payload_bytes,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len,
    bool is_backward,
    ParametricLensTraceInfo* trace_info)
{
    ParametricLensTraceInfo local_trace{};
    if (!p) return true;
    if (payload_bytes < (int)(8 * sizeof(float))) return true;
    if (p[0] != 14949.0f) return true;    /* PLENS_MAGIC */

    const int   n_surf  = (int)p[1];
    const float hood_r  = p[2];
    const float hood_xf = p[3];

    if (n_surf < 1 || n_surf > 64) return true;
    if (payload_bytes < (int)((8 + n_surf * 8) * sizeof(float))) return true;
    local_trace.events.reserve(static_cast<size_t>(n_surf));

    /* Lens hood check (only for forward direction). */
    if (!is_backward && hood_r > 0.0f && std::abs(dir[0]) > 1e-12) {
        const double t_hood = (double(hood_xf) - pos[0]) / dir[0];
        if (t_hood < 0.0) {
            const V3d p_hood = pos + t_hood * dir;
            if (std::hypot(p_hood[1], p_hood[2]) > (double)hood_r)
                return true;
        }
    }

    double opl = 0.0;
    V3d    rp  = pos;
    V3d    rd  = dir;

    /* Backward rays traverse surfaces in reverse order with swapped
     * refractive indices (sensor side → scene side). */
    for (int si = 0; si < n_surf; ++si) {
        const int s = is_backward ? (n_surf - 1 - si) : si;
        const int    sb    = 8 + s * 8;
        const double x_v   = (double)p[sb + 0];
        const double R     = (double)p[sb + 1];
        /* Forward: ray travels from n_bf into n_af.
         * Backward: ray travels from n_af into n_bf at this surface. */
        const double n_bf  = is_backward ? (double)p[sb + 3] : (double)p[sb + 2];
        const double n_af  = is_backward ? (double)p[sb + 2] : (double)p[sb + 3];
        const double ap_r  = (double)p[sb + 4];
        const double k     = (double)p[sb + 5];
        const bool is_stop = ((int)p[sb + 6] & 1) != 0;

        /* Seek to exact parametric surface.
         *
         * si=0 (first iteration): entry seek from T1 BVH hit position.
         * The proxy mesh that fired T1 is geometrically close to the parametric
         * surface but not identical.  Allow t ∈ [-0.01, +∞) so the ray can
         * roll back up to 1 cm to reach the exact conic vertex — this corrects
         * for proxy-mesh overshoot without letting a ray that missed entirely
         * sneak in.  For backward rays si=0 maps to s=n_surf-1 (sensor-side
         * surface), so the si-index (not the physical surface index s) must
         * gate this relaxed t_min and the root-selection branch.
         *
         * si>0: the previous Snell step left rp just in front of the next
         * surface; t must be strictly forward (> 2e-7 m). */
        double seg_t = 0.0;
        double seg_opl = 0.0;
        {
            const double t_min = (si == 0) ? -1e-2 : 2e-7;
            double t = 1e30;
            if (std::abs(R) < 1e-12) {
                if (std::abs(rd[0]) < 1e-12) return true;
                t = (x_v - rp[0]) / rd[0];
            } else {
                const double c  = 1.0 / R;
                const double kp = 1.0 + k;
                const double ox = rp[0] - x_v, oy = rp[1], oz = rp[2];
                const double dx = rd[0],  dy = rd[1],  dz = rd[2];
                const double A  = c * (dy*dy + dz*dz + kp*dx*dx);
                const double B  = 2.0 * (c*(oy*dy + oz*dz + kp*ox*dx) - dx);
                const double C  = c * (oy*oy + oz*oz + kp*ox*ox) - 2.0*ox;
                if (std::abs(A) < 1e-14) {
                    if (std::abs(B) < 1e-14) return true;
                    t = -C / B;
                } else {
                    const double disc = B*B - 4.0*A*C;
                    if (disc < 0.0) return true;
                    const double sq   = std::sqrt(disc);
                    const double t1   = (-B - sq) / (2.0 * A);
                    const double t2   = (-B + sq) / (2.0 * A);
                    if (si == 0) {
                        /* Pick the root with smallest |t| that is >= t_min. */
                        const bool v1 = t1 >= t_min, v2 = t2 >= t_min;
                        if (v1 && v2)      t = (std::abs(t1) <= std::abs(t2)) ? t1 : t2;
                        else if (v1)       t = t1;
                        else if (v2)       t = t2;
                        else return true;
                    } else {
                        const double x1 = rp[0] + t1 * dx;
                        const double x2 = rp[0] + t2 * dx;
                        if (t1 > 2e-7 && std::abs(x1 - x_v) <= std::abs(x2 - x_v)) t = t1;
                        else if (t2 > 2e-7) t = t2;
                        else return true;
                    }
                }
            }
            if (t < t_min) return true;
            seg_t = t;
            seg_opl = n_bf * t;
            opl += seg_opl;
            local_trace.geom_len += std::abs(t);
            rp  += t * rd;
        }

        /* Aperture/stop check. */
        const double r_tr = std::hypot(rp[1], rp[2]);
        if (ap_r > 0.0 && r_tr > ap_r) {
            local_trace.reason = BDPT_OPT_APERTURE_CLIP;
            BdptOpticalEventRecord ev{};
            ev.element_index = static_cast<uint16_t>(s);
            ev.reason = BDPT_OPT_APERTURE_CLIP;
            ev.pos[0] = static_cast<float>(rp.x());
            ev.pos[1] = static_cast<float>(rp.y());
            ev.pos[2] = static_cast<float>(rp.z());
            ev.dir_in[0] = static_cast<float>(rd.x());
            ev.dir_in[1] = static_cast<float>(rd.y());
            ev.dir_in[2] = static_cast<float>(rd.z());
            ev.eta_i = static_cast<float>(n_bf);
            ev.eta_t = static_cast<float>(n_af);
            ev.opl = static_cast<float>(seg_opl);
            ev.geom_len = static_cast<float>(std::abs(seg_t));
            ev.aperture_radius = static_cast<float>(ap_r);
            ev.transverse_radius = static_cast<float>(r_tr);
            ev.dist_past_aperture = static_cast<float>(r_tr - ap_r);
            ev.phase_space_jacobian = 1.0f;
            local_trace.events.push_back(ev);
            if (trace_info) *trace_info = local_trace;
            return true;
        }
        if (is_stop) continue;

        /* Exact surface normal (gradient of conic implicit). */
        V3d surf_n;
        if (std::abs(R) < 1e-12) {
            surf_n = V3d(1.0, 0.0, 0.0);
        } else {
            const double c  = 1.0 / R;
            const double kp = 1.0 + k;
            const double dx = rp[0] - x_v;
            surf_n = V3d(-1.0 + kp*c*dx, c*rp[1], c*rp[2]);
            const double nn = surf_n.norm();
            surf_n = (nn > 1e-14) ? surf_n / nn : V3d(1.0, 0.0, 0.0);
        }
        if (rd.dot(surf_n) > 0.0) surf_n = -surf_n;

        /* Vector Snell's law. */
        const double cos_i  = -rd.dot(surf_n);
        const double eta    = n_bf / n_af;
        const double sin2_t = eta * eta * std::max(0.0, 1.0 - cos_i*cos_i);
        BdptOpticalEventRecord ev{};
        ev.element_index = static_cast<uint16_t>(s);
        ev.pos[0] = static_cast<float>(rp.x());
        ev.pos[1] = static_cast<float>(rp.y());
        ev.pos[2] = static_cast<float>(rp.z());
        ev.normal[0] = static_cast<float>(surf_n.x());
        ev.normal[1] = static_cast<float>(surf_n.y());
        ev.normal[2] = static_cast<float>(surf_n.z());
        ev.dir_in[0] = static_cast<float>(rd.x());
        ev.dir_in[1] = static_cast<float>(rd.y());
        ev.dir_in[2] = static_cast<float>(rd.z());
        ev.cos_incident = static_cast<float>(cos_i);
        ev.eta_i = static_cast<float>(n_bf);
        ev.eta_t = static_cast<float>(n_af);
        ev.opl = static_cast<float>(seg_opl);
        ev.geom_len = static_cast<float>(std::abs(seg_t));
        ev.aperture_radius = static_cast<float>(ap_r);
        ev.transverse_radius = static_cast<float>(r_tr);
        if (sin2_t > 1.0) {
            local_trace.reason = BDPT_OPT_TIR;
            ev.reason = BDPT_OPT_TIR;
            ev.fresnel_reflectance = 1.0f;
            ev.transmittance = 0.0f;
            ev.phase_space_jacobian = 1.0f;
            local_trace.events.push_back(ev);
            if (trace_info) *trace_info = local_trace;
            return true;
        }
        const double cos_t  = std::sqrt(1.0 - sin2_t);
        const double Rf     = fresnel_R(cos_i, cos_t, n_bf, n_af);
        const double J      = (cos_i > EPS && cos_t > EPS)
            ? (eta * eta) * (cos_t / cos_i) : 0.0;
        if (local_trace.first_cos_incident <= 0.0) {
            local_trace.first_cos_incident = cos_i;
            local_trace.first_eta_i = n_bf;
        }
        local_trace.last_cos_transmitted = cos_t;
        local_trace.last_eta_t = n_af;
        if (cos_i > EPS && cos_t > EPS)
            local_trace.phase_space_jacobian *= J;
        V3d refr = eta * rd + (eta * cos_i - cos_t) * surf_n;
        const double rn = refr.norm();
        if (rn < 1e-14) return true;
        rd = refr / rn;
        ev.reason = BDPT_OPT_REFRACTION;
        ev.dir_out[0] = static_cast<float>(rd.x());
        ev.dir_out[1] = static_cast<float>(rd.y());
        ev.dir_out[2] = static_cast<float>(rd.z());
        ev.cos_transmitted = static_cast<float>(cos_t);
        ev.fresnel_reflectance = static_cast<float>(Rf);
        ev.transmittance = static_cast<float>(1.0 - Rf);
        ev.throughput_multiplier = static_cast<float>(1.0 - Rf);
        ev.phase_space_jacobian = static_cast<float>(J);
        local_trace.events.push_back(ev);
    }

    /* Accumulate OPL phase on all bands. */
    if (opl > 0.0 && path_len) {
        const double c0 = (st.speed_m_s > EPS) ? st.speed_m_s : 299792458.0;
        for (int b = 0; b < (int)amp.size() && b < st.n_bands; ++b) {
            const double phase = TWO_PI * st.freq_hz_vec[b] * opl / c0;
            amp[b] *= cd(std::cos(phase), std::sin(phase));
        }
        *path_len += opl;
    }
    local_trace.opl = opl;

    pos = rp + rd * (EPS * 200.0);
    dir = rd;
    if (trace_info) *trace_info = local_trace;
    return false;   /* teleported */
}
#pragma optimize("", on)

/* ── apply_manifold_transfer_from_payload ───────────────────────────────────
 * Like apply_manifold_transfer() but reads directly from a raw float payload
 * pointer instead of a RtScaleContext.  The aperture hit point is provided
 * directly (it is the T1/T2 BVH hit position, already on the entry plane). */
static inline void apply_manifold_transfer_from_payload(
    const RayTracerState& st,
    const float* p, int payload_bytes,
    const V3d& ap_hit,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len)
{
    (void)st; (void)p; (void)payload_bytes; (void)ap_hit; (void)pos; (void)dir; (void)path_len;
    /* Deferred: LUT/spline manifold transport is disabled until it can provide
     * reversible optical metadata instead of an opaque teleport. */
    kill_ray_amplitudes(amp);
}

/* ── seek_parametric_entry_s0 ────────────────────────────────────────────────
 * Advance (or roll back) `pos` from the T1 BVH proxy-mesh hit to the exact
 * parametric s=0 entrance surface defined in a PLENS payload (magic 14949).
 * Used before dispatching LUT or MLP transfers so their entry state is
 * anchored on the true parametric geometry, not the proxy mesh.
 *
 * `dir` is left unchanged — LUT/MLP encode the full lens including the first
 * surface refraction; the caller provides the pre-refraction incident direction.
 *
 * Returns false if the payload is invalid or the ray misses / is out of range.
 * On success `pos` is at the exact s=0 surface intersection and `opl_accum`
 * receives n_before * t (air path from proxy hit to parametric surface). */
static inline bool seek_parametric_entry_s0(
    const float* p, int payload_bytes,
    V3d& pos, const V3d& dir, double& opl_accum)
{
    if (!p || payload_bytes < (int)(16 * sizeof(float))) return false;
    /* Works for PLENS (14949) and LUT/MLP payloads that embed a PLENS header at
     * a 16-float offset — for those, surface geometry lives in the same place. */
    if (p[0] != 14949.0f) return false;  /* only PLENS carries explicit surface geometry */

    const int   n_surf = (int)p[1];
    if (n_surf < 1) return false;

    const int    sb   = 8;   /* s=0 surface record start */
    const double x_v  = (double)p[sb + 0];
    const double R    = (double)p[sb + 1];
    const double n_bf = (double)p[sb + 2];
    const double ap_r = (double)p[sb + 4];
    const double k    = (double)p[sb + 5];

    V3d rp = pos;
    const V3d& rd = dir;
    double t = 1e30;

    if (std::abs(R) < 1e-12) {
        if (std::abs(rd[0]) < 1e-12) return false;
        t = (x_v - rp[0]) / rd[0];
    } else {
        const double c  = 1.0 / R;
        const double kp = 1.0 + k;
        const double ox = rp[0] - x_v, oy = rp[1], oz = rp[2];
        const double dx = rd[0],  dy = rd[1],  dz = rd[2];
        const double A  = c * (dy*dy + dz*dz + kp*dx*dx);
        const double B  = 2.0 * (c*(oy*dy + oz*dz + kp*ox*dx) - dx);
        const double C  = c * (oy*oy + oz*oz + kp*ox*ox) - 2.0*ox;
        if (std::abs(A) < 1e-14) {
            if (std::abs(B) < 1e-14) return false;
            t = -C / B;
        } else {
            const double disc = B*B - 4.0*A*C;
            if (disc < 0.0) return false;
            const double sq = std::sqrt(disc);
            const double t1 = (-B - sq) / (2.0 * A);
            const double t2 = (-B + sq) / (2.0 * A);
            const bool   v1 = t1 >= -1e-2, v2 = t2 >= -1e-2;
            if      (v1 && v2) t = (std::abs(t1) <= std::abs(t2)) ? t1 : t2;
            else if (v1)       t = t1;
            else if (v2)       t = t2;
            else return false;
        }
    }
    if (t < -1e-2) return false;

    rp += t * rd;

    /* Aperture check at the exact entry surface. */
    if (ap_r > 0.0 && std::hypot(rp[1], rp[2]) > ap_r) return false;

    opl_accum += n_bf * t;
    pos = rp;
    return true;
}

/* Thin wrapper so the scale-context dispatch path keeps its original signature. */
static inline void apply_neural_mlp_transfer(
    const RayTracerState& st,
    const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len)
{
    apply_neural_mlp_from_f32(
        st,
        static_cast<const float*>(ctx.payload),
        ctx.payload_size_bytes,
        pos, dir, amp, path_len);
}

static inline void apply_thin_lens_transform(
    const RtScaleContext& ctx, V3d& pos, V3d& dir)
{
    /* payload[0]   = focal length f (m); <= 0 → no-op.
     * payload[1..3] = fixed lens axis (unit vector), OPTIONAL.
     *   If present and non-zero, the lens has a fixed optical axis and
     *   the standard thin-lens matrix deflection is used (physically correct).
     *   If absent or zero, the axis auto-aligns to the current ray direction
     *   (legacy behavior, preserved for non-PIXEL_CONE forward rays). */
    if (!ctx.payload) return;
    const double* p = static_cast<const double*>(ctx.payload);
    double f = p[0];
    if (!(f > 0.0)) return;

    V3d centre(ctx.center[0], ctx.center[1], ctx.center[2]);

    bool has_fixed_axis = (ctx.payload_size_bytes >= (int)(4 * sizeof(double)));
    if (has_fixed_axis) {
        V3d axis(p[1], p[2], p[3]);
        double axis_norm = axis.norm();
        has_fixed_axis = (axis_norm > 0.5);
        if (has_fixed_axis) axis /= axis_norm;

        if (has_fixed_axis) {
            /* Fixed-axis thin-lens: ray-transfer matrix deflection.
             *   h     = lateral offset of crossing point from lens axis
             *   slope = perpendicular / axial velocity component
             *   slope_out = slope_in - h / f   (thin-lens paraxial)
             * Implemented geometrically (exact for finite angles):
             *   new_dir_perp = dir_perp - (h / f) * dir_axial
             *   new_dir = normalize(dir_axial * axis + new_dir_perp) */
            V3d rel = pos - centre;
            V3d h = rel - rel.dot(axis) * axis;   /* lateral offset vector */
            double d_axial = dir.dot(axis);
            V3d d_perp = dir - d_axial * axis;
            V3d new_d_perp = d_perp - h * (d_axial / f);
            V3d new_dir = (d_axial * axis + new_d_perp).normalized();
            if (new_dir.squaredNorm() > 1.0e-18) dir = new_dir;
            return;
        }
    }

    /* Legacy auto-align: optical axis = current ray direction.
     * Preserved for backward compatibility; not a physically fixed lens. */
    V3d focal_pt = centre + f * dir;
    V3d new_dir = (focal_pt - pos).normalized();
    if (new_dir.norm() > 1e-9) dir = new_dir;
}

/* apply_thick_lens_wave_transform — geometric thin-lens ray steering only.
 * Wave physics for RT_SCALE_WAVE regions is handled by WaveArena (T4). */
static inline void apply_thick_lens_wave_transform(
    const RayTracerState& /*st*/,
    const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& /*amp*/)
{
    apply_thin_lens_transform(ctx, pos, dir);
}

/* dispatch_scale_context_entry — apply the appropriate geometric or lens
 * transform when a ray segment crosses into a registered scale context.
 * RT_SCALE_WAVE / WAVE_HELMHOLTZ regions are handled by WaveArena in the
 * pipeline (T4); here we only set the flag so downstream consumers know the
 * ray passed through a wave region. */
static inline uint32_t dispatch_scale_context_entry(
    const RayTracerState& st,
    const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp,
    double* path_len)
{
    switch (ctx.context_kind) {
        case SCALE_CONTEXT_KIND_RAY:
            return 0u;
        case SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ:
            /* Real wave propagation done by WaveArena; just flag the entry. */
            return 1u << SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ;
        case SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM:
            apply_thin_lens_transform(ctx, pos, dir);
            return 1u << SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM;
        case SCALE_CONTEXT_KIND_THICK_LENS_WAVE:
            apply_thick_lens_wave_transform(st, ctx, pos, dir, amp);
            return 1u << SCALE_CONTEXT_KIND_THICK_LENS_WAVE;
        case SCALE_CONTEXT_KIND_SPLINE_SURFACE:
            return 1u << SCALE_CONTEXT_KIND_SPLINE_SURFACE;
        case SCALE_CONTEXT_KIND_NEURAL_SURFACE:
            /* Deferred: neural/LUT scale-context transforms are disabled until
             * they carry the same reversible optical contract as the parametric
             * lens path. */
            kill_ray_amplitudes(amp);
            return 1u << SCALE_CONTEXT_KIND_NEURAL_SURFACE;
        case SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC:
            return 1u << SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC;
        default:
            return 0u;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4-stage ray transport pipeline
 *
 * T1 intersector  — BVH query + amplitude propagation + field capture.
 *                   Routes to T4 if hit is inside a RT_SCALE_WAVE arena.
 * T2 refiner      — parametric surface point / normal refinement.
 * T3 material     — Fresnel/Snell/diffuse physics; spawns child RayIntents.
 * T4 wave solver  — ADI-CN BPM march per arena; exits re-enter T1.
 * T5 connector    — staged BDPT connection/MIS over collected side records.
 *
 * Forward and backward paths are identical — BDPT delivers RayIntents only.
 * ═══════════════════════════════════════════════════════════════════════════ */

struct RayPipelineState {
    RayTracerState*      st  = nullptr;
    RayPipelineConfig    cfg;

    PipelineQueue<RayIntent>  Q_intent;
    PipelineQueue<HitRecord>  Q_hit;
    PipelineQueue<RefinedHit> Q_refined;
    PipelineQueue<WaveIntent> Q_wave;
    PipelineQueue<RayRecord>  Q_out;    /* output records drained by the caller */

    /* ── BDPT side-data queues — never mixed into Q_out ──────────────────────
     * Each queue is drained separately via a dedicated pybind function.
     * Emission sites: T1 → Q_bdpt_vertices, T2/lens → Q_bdpt_optical,
     *                 T3 → Q_bdpt_pdfs + Q_bdpt_spectral. */
    PipelineQueue<BdptVertexRecord>         Q_bdpt_vertices;
    PipelineQueue<BdptSpectralWeightRecord> Q_bdpt_spectral;
    PipelineQueue<BdptPdfRecord>            Q_bdpt_pdfs;
    PipelineQueue<BdptOpticalEventRecord>   Q_bdpt_optical;
    PipelineQueue<BdptConnectionRecord>     Q_bdpt_connections;

    /* Records drained from the live stage queues but not yet connected.
     * Connection can be requested while only one side of the bidirectional
     * exposure has arrived; those records must remain available until the
     * matching camera/light side is present. */
    std::mutex                              bdpt_pending_mu;
    std::vector<BdptVertexRecord>           bdpt_pending_vertices;
    std::vector<BdptSpectralWeightRecord>   bdpt_pending_spectral;
    std::vector<BdptPdfRecord>              bdpt_pending_pdfs;
    std::vector<BdptOpticalEventRecord>     bdpt_pending_optical;

    /* ── Sensor-batching light-record stash ───────────────────────────────
     * When bdpt_batching_mode is true, run_bdpt_connection() saves the
     * light-stream records from the first T5 pass here so subsequent sensor
     * batches can reuse the same flash light field without re-firing flash.
     * Cleared by ray_pipeline_end_sensor_batching() at exposure end. */
    bool                                    bdpt_batching_mode{false};
    std::mutex                              bdpt_stash_mu;
    std::vector<BdptVertexRecord>           bdpt_stash_v;
    std::vector<BdptSpectralWeightRecord>   bdpt_stash_sw;
    std::vector<BdptPdfRecord>              bdpt_stash_pdf;
    std::vector<BdptOpticalEventRecord>     bdpt_stash_opt;

    /* Overflow counters — incremented when a side queue is full.
     * Never wraps: saturate at UINT64_MAX. */
    std::atomic<uint64_t>     bdpt_overflow_vertices{0};
    std::atomic<uint64_t>     bdpt_overflow_spectral{0};
    std::atomic<uint64_t>     bdpt_overflow_pdfs{0};
    std::atomic<uint64_t>     bdpt_overflow_optical{0};
    std::atomic<uint64_t>     bdpt_overflow_connections{0};

    /* Bounded push helpers — drop + count when at cap. */
    void push_bdpt_vertex(BdptVertexRecord r) {
        int cap = cfg.bdpt_max_vertices;
        if (cap > 0 && Q_bdpt_vertices.size() >= cap) {
            bdpt_overflow_vertices.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Q_bdpt_vertices.push(std::move(r));
    }
    void push_bdpt_spectral(BdptSpectralWeightRecord r) {
        int cap = cfg.bdpt_max_spectral;
        if (cap > 0 && Q_bdpt_spectral.size() >= cap) {
            bdpt_overflow_spectral.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Q_bdpt_spectral.push(std::move(r));
    }
    void push_bdpt_pdf(BdptPdfRecord r) {
        int cap = cfg.bdpt_max_pdfs;
        if (cap > 0 && Q_bdpt_pdfs.size() >= cap) {
            bdpt_overflow_pdfs.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Q_bdpt_pdfs.push(std::move(r));
    }
    void push_bdpt_optical(BdptOpticalEventRecord r) {
        int cap = cfg.bdpt_max_optical;
        if (cap > 0 && Q_bdpt_optical.size() >= cap) {
            bdpt_overflow_optical.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Q_bdpt_optical.push(std::move(r));
    }
    void push_bdpt_connection(BdptConnectionRecord r) {
        int cap = cfg.bdpt_max_connections;
        if (cap > 0 && Q_bdpt_connections.size() >= cap) {
            bdpt_overflow_connections.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Q_bdpt_connections.push(std::move(r));
    }

    std::atomic<int>          in_flight{0};

    /* Pre-flagged materials: flag[m]=1 means max reflectance < cfg.min_amplitude,
     * so any opaque ray hitting material m will never produce a viable child.
     * T3 fast-path skips direction sampling + reflectance multiply for these. */
    std::vector<uint8_t>      mat_epsilon_flags;

    StageStats                stats[5]; /* [0]=T1 [1]=T2 [2]=T3 [3]=T4 [4]=T5 */
    std::atomic<uint64_t>     gpu_uv_readback_bytes{0};
    std::atomic<uint64_t>     gpu_uv_readback_count{0};
    std::atomic<uint64_t>     gpu_hit_readback_bytes{0};
    std::atomic<uint64_t>     gpu_hit_readback_count{0};

    std::deque<WaveArena>     arenas;   /* deque: never moves elements, safe with mutex */
    std::vector<std::thread>  workers;

    /* ── Sensor image accumulator ─────────────────────────────────────────
     * Three-channel float64 buffer, res×res, layout [ch][iy][iz].
     * ch0 = R  (red-weighted spectral sum, all path types)
     * ch1 = G  (green-weighted)
     * ch2 = B  (blue-weighted)
     * ch3 = near-miss accumulator (debug only, not displayed)
     * Protected by sensor_mu; read by ray_pipeline_get_sensor_image(). */
    mutable std::mutex        sensor_mu;
    int                       sensor_res  = 0;   /* 0 = disabled */
    float                     sensor_px   = 0.0f;
    float                     sensor_half_w = 0.0f;
    float                     sensor_half_h = 0.0f;
    float                     sensor_eps  = 0.008f;
    float                     sensor_target_x = 0.0f;
    float                     sensor_target_r = 0.0f;
    std::vector<double>       sensor_accum;       /* res*res*4 doubles: ch0=R, ch1=G, ch2=B (spectral), ch3=near-miss */
    std::vector<float>        priority_map;       /* res*res sugar-auxin field (1.0 = baseline) */
    mutable double            sensor_peak[4]      = {1e-30, 1e-30, 1e-30, 1e-30}; /* running per-channel max, never decreases */

    /* ── GPU compute dispatch (nullptr = CPU-only) ─────────────────────────── */
    class GlPipelineDispatch;                       /* forward-declared below    */
    GlPipelineDispatch*       gpu_dispatch = nullptr;

    /* GPU dispatch health state (atomic, readable from any thread/Python):
     *   0 = disabled (use_gpu_compute=false)
     *   1 = init failed (GL context creation or shader compile error)
     *   2 = thread spawned, make_current not yet attempted
     *   3 = thread running (make_current succeeded, scene uploaded)
     *   4 = thread failed (make_current failed in worker thread)
     *   5 = thread exited normally (Q_intent exhausted) */
    std::atomic<int>          gpu_dispatch_state{0};

    /* ── Live BDPT diagnostic stats (updated by T3, lock-free) ──────────
     * nearest_d2: running min of squared YZ distance over all fwd/rev pairs
     * best_collinearity: cos(angle between fwd and rev directions) of best pair
     * exact_snaps / near_miss_count: cumulative hit counts */
    std::atomic<uint64_t>     bdpt_nearest_d2_bits{0xFFFFFFFFFFFFFFFFULL}; /* float bits via reinterpret */
    std::atomic<uint64_t>     bdpt_best_colinear_bits{0};
    std::atomic<uint64_t>     bdpt_exact_snaps{0};
    std::atomic<uint64_t>     bdpt_near_miss_count{0};

    /* Running count of camera-stream BDPT vertices — diagnostic only. */
    std::atomic<uint64_t>     bdpt_cam_vertex_count{0};
    std::atomic<uint32_t>     bdpt_next_subpath_id{1u};
    std::atomic<bool>         bdpt_connection_running{false};
    std::atomic<uint64_t>     emitter_world_culled{0};

    /* T5 substage-dispatch latch — first-submission markers only.
     * These record that a family has been submitted this cycle.
     * They have no bearing on tracing completeness. */
    std::atomic<uint32_t>     bdpt_flash_dispatched{0};
    std::atomic<uint32_t>     bdpt_sensor_dispatched{0};
    std::atomic<uint32_t>     bdpt_t5_fired{0};
    std::mutex                bdpt_t5_spawn_mu;
    std::thread               bdpt_t5_worker;

    /* ── KPN-style T5 completion channel ─────────────────────────────────
     * Per-family in-flight ray counts.  Decremented when a ray (and all its
     * bounce descendants) terminates; incremented when children are spawned.
     * When both hit zero after both families have been submitted, a sentinel
     * is pushed to Q_t5_ready.  The T5 worker thread blocks reading that
     * queue — no polling, no timeouts, no ad-hoc condition variables. */
    std::atomic<int64_t>      bdpt_inflight_flash{0};
    std::atomic<int64_t>      bdpt_inflight_sensor{0};
    PipelineQueue<int>         Q_t5_ready;   /* sentinel channel: push 1 → T5 unblocks */
};

/* bdpt_update_inflight — called after every T3 batch with the net per-family
 * change (positive = children spawned exceed parents processed; negative = net
 * terminations).  When both family counters reach zero and both families have
 * been submitted for this cycle, pushes a sentinel to Q_t5_ready so the T5
 * worker unblocks.  Only one sentinel per cycle: bdpt_connection_running acts
 * as the gate so we don't double-fire while T5 is already consuming. */
static void bdpt_update_inflight(RayPipelineState* ps,
                                  int64_t delta_flash,
                                  int64_t delta_sensor)
{
    if (delta_flash == 0 && delta_sensor == 0) return;
    const int64_t nf = ps->bdpt_inflight_flash.fetch_add(delta_flash,
                           std::memory_order_acq_rel) + delta_flash;
    const int64_t ns = ps->bdpt_inflight_sensor.fetch_add(delta_sensor,
                           std::memory_order_acq_rel) + delta_sensor;
    if (nf != 0 || ns != 0) return;

    /* Both counters just hit zero.  Check whether both families have been
     * submitted for a new cycle and T5 is not already running. */
    const uint32_t flash_submitted  = ps->bdpt_flash_dispatched.load(std::memory_order_acquire);
    const uint32_t sensor_submitted = ps->bdpt_sensor_dispatched.load(std::memory_order_acquire);
    const uint32_t t5_fired         = ps->bdpt_t5_fired.load(std::memory_order_acquire);
    if (std::min(flash_submitted, sensor_submitted) <= t5_fired) return;
    if (ps->bdpt_connection_running.load(std::memory_order_acquire)) return;

    fprintf(stderr, "[T5-kpn] both families exhausted — pushing sentinel "
            "(flash_if=%lld sensor_if=%lld flash_sub=%u sensor_sub=%u t5_fired=%u)\n",
            (long long)nf, (long long)ns,
            flash_submitted, sensor_submitted, t5_fired);
    fflush(stderr);
    ps->Q_t5_ready.push(1);
}

/* CIE-approximate spectral locus — forward declaration; definition is near ray_pipeline_join_t5 */
static void band_to_display_rgb(int b, int n_bands,
                                 const Eigen::VectorXd& freq_hz_vec,
                                 double& wr, double& wg, double& wb);

/* ═══════════════════════════════════════════════════════════════════════════
 * GlPipelineDispatch — GPU compute backend for all four pipeline stages.
 *
 * Design:
 *   One GlPipelineDispatch object owns the WGL headless GL 4.3 context and
 *   all shader programs + SSBOs.  A single `gl_dispatch_thread` (spawned in
 *   ray_pipeline_create when use_gpu_compute=true) runs thread_main(), which
 *   pops large batches from the SAME queues as the CPU workers and dispatches
 *   them through T1→T2→T3 on the GPU atomically, then pushes child intents
 *   back to Q_intent for the next generation.  T2 and T3 are GPU-internal;
 *   they do not compete with CPU T2/T3 on Q_hit/Q_refined.
 *
 *   Both CPU workers and the GPU thread compete on Q_intent.  The
 *   StageStats adaptive-batch logic causes the GPU thread (which uses larger
 *   batches) to absorb more work when the queue is deep, and fall back to CPU
 *   when the queue is thin — automatically balancing the two operators.
 *
 *   T4 (wave BPM) runs on the GPU when a WaveIntent is popped from Q_wave.
 *   The CPU pipeline_wave_solver thread is NOT spawned when use_gpu_compute
 *   is set; T4 runs exclusively on GPU in that mode.
 *
 * Per-stage profiling:
 *   After each GPU dispatch the elapsed wall time is reported via
 *   ps.stats[i].record_gpu(n, ns, q_depth), which updates gpu_items_per_sec
 *   and the adaptive gpu_fraction EMA so Python callers can observe load split.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void t5_kpn_worker(RayPipelineState* ps);  /* KPN T5 process — defined after run_bdpt_connection */

class RayPipelineState::GlPipelineDispatch {
public:
    GlComputeContext ctx{};
    bool             ready = false;

    /* Shader programs for each stage */
    GLuint prog_t1 = 0, prog_t2 = 0, prog_t3 = 0, prog_t4 = 0;
    /* T5 BDPT connection pass (t5_full_connect.comp.glsl) — non-fatal if absent */
    GLuint prog_t5 = 0;

    /* T5 persistent SSBOs — allocated on first use, grown as needed */
    GLuint ssbo_t5_light    = 0;  /* binding 0: T5LightVertBuf       (float)  */
    GLuint ssbo_t5_cam      = 0;  /* binding 1: T5CamVertBuf         (float)  */
    GLuint ssbo_t5_pix      = 0;  /* binding 2: T5PixelBuf           (uint)   */
    GLuint ssbo_t5_params   = 0;  /* binding 3: T5ParamsBuf          (40 B)   */
    GLuint ssbo_t5_spectral = 0;  /* binding 4: T5SpectralWeightBuf  (float)  */
    int      cap_t5_light        = 0;  /* floats */
    int      cap_t5_cam          = 0;  /* floats */
    int      cap_t5_pix          = 0;  /* uint32 */
    int      cap_t5_spectral     = 0;  /* floats */
    uint32_t t5_light_batch_size  = 0;  /* 0 = use default 4096  */
    uint32_t t5_cam_batch_size    = 0;  /* 0 = use default 8192  */
    uint32_t t5_sensor_tile_size  = 0;  /* 0 = use default 128   */


    /* ── T5 GPU job queue ─────────────────────────────────────────────── *
     * The BDPT connection thread submits a job and blocks on the future;  *
     * the GPU thread services it between T1/T2/T3 batches.               */
    struct T5Job {
        std::vector<float>    light_verts;      /* flat, T5_LGV_STRIDE each   */
        std::vector<float>    cam_verts;        /* flat, T5_CGV_STRIDE each   */
        std::vector<uint32_t> pixel_accum;      /* unused — service_t5_job allocates per-tile */
        std::vector<float>    spectral_weights; /* n_bands × 3 (wr, wg, wb)   */
        T5GpuParams           params;
        std::promise<std::vector<uint32_t>> promise;
    };
    std::mutex              t5_job_mu;
    std::condition_variable t5_job_cv;
    std::unique_ptr<T5Job>  t5_job_pending;           /* null = no pending job */
    std::atomic<bool>       t5_job_ready{false};

    /* UV blit shader + shared display textures (valid only when WGL context sharing
     * is active — i.e. ps.cfg.gl_display_hglrc != 0 and prog_uv_blit != 0). */
    GLuint prog_uv_blit  = 0;
    static constexpr int UV_TEX_RING = 3;
    std::array<GLuint, UV_TEX_RING> tex_uv_pages = {}; /* shared GL_TEXTURE_2D_ARRAY ring */
    std::array<GLsync, UV_TEX_RING> uv_fences = {};
    int    uv_build_slot = 0;
    int    uv_active_slot = -1;
    int    uv_pending_slot = -1;
    uint64_t uv_generation = 0;
    mutable std::mutex uv_publish_mu;
    int    blit_n_layers = 0;   /* layer count the texture was allocated for */
    int    blit_res      = 0;   /* texel resolution (res×res per layer)                 */
    bool   display_context_shared = false;
    /* Cached uniform locations for uv_blit.comp.glsl */
    struct { GLint n_uv_groups, uv_meta_base, n_bands, uv_blit_mode, rgb_w; }
        uloc_blit = {-1, -1, -1, -1, -1};
    /* Per-band RGB weights uploaded as rgb_w[32] uniform (3 floats × 32 bands max).
     * Set from Python via set_uv_blit_weights().  Default: equal-weight gray. */
    std::array<float, MAX_SPECTRAL_BANDS * 3> blit_rgb_weights = []() {
        std::array<float, MAX_SPECTRAL_BANDS * 3> a{};
        for (int i = 0; i < MAX_SPECTRAL_BANDS * 3; ++i) a[i] = 1.f/3.f;
        return a;
    }();
    int  blit_n_bands_stored = 0;  /* 0 = use st.n_bands at dispatch time */
    int  blit_mode           = 0;  /* 0=combined(fwd+sen), 1=forward, 2=sensor */
    static constexpr GLuint UV_BLIT_IMAGE_UNIT = 0;

    /* Cached uniform locations — populated once after shader link */
    struct { GLint n_intents, n_bands, n_mats, n_arenas, n_tris, u_arenas; }
        uloc_t1 = {-1,-1,-1,-1,-1,-1};
    struct { GLint n_tris, n_groups, n_bands, freq_hz,
                   bdpt_max_optical, bdpt_optical_base; }
        uloc_t2 = {-1,-1,-1,-1,-1,-1};
    struct { GLint n_bands, n_mats, max_children, max_children_per_hit,
                   sensor_res, sensor_pr, sensor_px,
                   n_uv_groups, uv_group_id_base, uv_meta_base, rng_seed,
                   bdpt_max_verts, bdpt_max_spectral, bdpt_max_pdfs,
                   bdpt_count_base, bdpt_spectral_base, bdpt_pdf_base,
                   n_hits; }
        uloc_t3 = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    struct { GLint mode, nx, ny, n_bands, dx, dz, wavelengths; }
        uloc_t4 = {-1,-1,-1,-1,-1,-1,-1};

    /* Per-stage SSBOs allocated once and resized as needed */
    /* T1 inputs */
    GLuint ssbo_intent      = 0; /* RayIntent flat float buffer */
    GLuint ssbo_bvh         = 0; /* BVH nodes */
    GLuint ssbo_tri_id      = 0; /* BVH permutation ints */
    GLuint ssbo_tri_full    = 0; /* triangle geometry */
    GLuint ssbo_mat_band    = 0; /* material band records */
    GLuint ssbo_scene_band  = 0; /* k_real + atmo_abs */
    GLuint ssbo_wave_arena  = 0; /* arena center+radius */
    GLuint ssbo_tri_sensor  = 0; /* per-tri sensor group id */
    /* T1/T2 outputs */
    GLuint ssbo_hit         = 0; /* RefinedHit layout (T1 out / T2 in-place) */
    GLuint ssbo_counter     = 0; /* {hit_count, miss_count, wave_count, cpu_refine} uint */
    /* T2 refinement */
    GLuint ssbo_tri_param   = 0; /* per-tri parametric group id */
    GLuint ssbo_group_kind  = 0;
    GLuint ssbo_group_pay   = 0;
    GLuint ssbo_neural_pay  = 0; /* concatenated neural MLP payloads (float32) */
    /* T3 output: child intents at [0..max_children*INTENT_STRIDE) and
     * terminal records at [max_children*INTENT_STRIDE..) packed in one SSBO */
    GLuint ssbo_child_int   = 0;
    /* T3 meta: [0]=intent_count [1]=terminal_count [2..2+n_mats-1]=eps_flags
     * Only [0] and [1] are zeroed before each dispatch; eps_flags are written
     * at scene-upload time and persist across dispatches. */
    GLuint ssbo_t3_meta     = 0;
    /* BDPT output: single SSBO (verts first, spectral at bdpt_spectral_base,
     * pdfs at bdpt_pdf_base, optical at bdpt_optical_base).
     * Vertex/spectral/pdf counts live in MetaBuf at [2+n_mats..2+n_mats+2];
     * GPU T2 optical count lives in CounterBuf[4].
     * bdpt_sid is embedded in hit[26+2*MAX_SPECTRAL_BANDS] by T1 — no separate id SSBO needed. */
    GLuint ssbo_bdpt_output   = 0;
    int    cap_bdpt_output    = 0;  /* in floats */
    /* UV integrator image: merged_uv int32 SSBO at binding 5 contains:
     *   [0 .. n_tris*6)            UV vertex coords as floatBitsToInt (6 per tri)
     *   [n_tris*6 .. n_tris*7)     per-tri UV group IDs (int; -1=no group)
     *   [n_tris*7 .. )             group metadata (2 ints each: res, accum_offset)
     * UvAccumBuf at binding 6; BdptOutputBuf at binding 7. */
    GLuint ssbo_merged_uv  = 0;  /* binding 5: merged UV coords + group IDs + meta */
    GLuint ssbo_uv_accum   = 0;  /* binding 6: flat uint32 accumulator             */
    /* T4 BPM wave field */
    GLuint ssbo_wave_re     = 0;
    GLuint ssbo_wave_im     = 0;
    GLuint ssbo_bpm_tmp_re  = 0;
    GLuint ssbo_bpm_tmp_im  = 0;
    GLuint ssbo_thomas_cp   = 0;
    GLuint ssbo_thomas_dp   = 0;

    /* Capacities to know when realloc is needed */
    int cap_intents  = 0;
    int cap_hits     = 0;
    int cap_children = 0; /* also gates terminal capacity: buffer = cap*(INTENT_STRIDE+TERMINAL_STRIDE) */
    int cap_wave_pix = 0; /* per band */

    /* Arena data cached for per-dispatch uniform upload (≤16 arenas × 4 floats) */
    std::vector<float> arena_uniform_data;

    /* CPU-side staging buffers — grown as needed, never shrunk.
     * Eliminates large malloc/free pairs per dispatch batch. */
    std::vector<float>     stg_ibuf;          /* intent upload (n*INTENT_STRIDE)             */
    std::vector<float>     stg_cbuf;          /* child intent readback (nc*CHILD_STRIDE)     */
    std::vector<float>     stg_tbuf;          /* terminal readback (nt_term*TERMINAL_STRIDE) */
    std::vector<float>     stg_hbuf;          /* hit readback (n_hits*HIT_STRIDE)            */
    std::vector<float>     stg_bdpt_verts;    /* BdptVertexRecord readback                   */
    std::vector<float>     stg_bdpt_spectral; /* BdptSpectralWeightRecord readback           */
    std::vector<float>     stg_bdpt_pdfs;     /* BdptPdfRecord readback                      */
    std::vector<float>     stg_bdpt_optical;  /* BdptOpticalEventRecord readback             */
    std::vector<RayRecord> stg_strike;   /* STRIKE records for Q_out                   */
    std::vector<RayIntent> stg_children; /* child intents for push_many                */
    std::vector<uint32_t>  stg_uv_acc;  /* UV accumulator readback                    */
    std::vector<VXcd>      stg_amp_recycle; /* Eigen amp allocs harvested from prior batch */
    struct SensorUpdate { int iy, iz; double r, g, b; };
    std::vector<SensorUpdate> stg_sensor_updates; /* precomputed sensor updates, outside lock */
    std::chrono::steady_clock::time_point last_uv_readback_t = std::chrono::steady_clock::now();
    std::atomic<double> uv_readback_interval_s{1.0};

    /* Scene data uploaded once */
    bool scene_uploaded = false;

    /* Number of parametric groups uploaded to GPU — 0 means T2 is a no-op */
    int n_param_groups = 0;

    /* Total float count last uploaded to ssbo_neural_pay; used to skip
     * re-uploading the large MLP weight buffer when payloads haven't changed. */
    size_t neural_pay_floats_uploaded = 0;

    /* ── helpers ──────────────────────────────────────────────────────────── */

    static std::string resolve_shader(const std::string& dir, const std::string& name) {
        if (!dir.empty()) return dir + "/" + name;
        return "csrc/shaders/" + name;
    }

    /* Allocate or grow an SSBO to at least `bytes`.  Returns false on GL error. */
    bool ensure_ssbo(GLuint& id, GLsizeiptr bytes) {
        if (!id) glc_GenBuffers(1, &id);
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glc_BufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return (glGetError() == GL_NO_ERROR);
    }

    void bind_ssbo(GLuint id, GLuint binding) {
        glc_BindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id);
    }

    /* Upload data to an SSBO (auto-grows if needed). */
    void upload_ssbo(GLuint& id, const void* data, GLsizeiptr bytes) {
        if (!id) glc_GenBuffers(1, &id);
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glc_BufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        if (data && bytes > 0)
            glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, data);
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    /* Read back bytes from an SSBO into dst. */
    void readback_ssbo(GLuint id, void* dst, GLsizeiptr bytes) {
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, dst);
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    /* Re-sync parametric SSBOs from current CPU state.
     * Called once from upload_scene_data and again before every T2 dispatch so
     * that clear_tri_groups() + re-registration (e.g. from _ensure_bdpt_plate_sensor_group)
     * is always visible to the GPU without requiring a full scene re-upload.
     * ssbo_neural_pay (large MLP weights) is skipped when the total payload float
     * count hasn't changed, avoiding multi-MB PCIe transfers every batch. */
    void upload_param_groups_data(const RayTracerState& st) {
        const int nt = (int)st.tris.size();
        const int ng = (int)st.tri_group_parametric_kind.size();
        static constexpr int GPS = 16;

        if (!st.tri_param_group_of_tri.empty()) {
            upload_ssbo(ssbo_tri_param, st.tri_param_group_of_tri.data(),
                        (GLsizeiptr)(nt * sizeof(int)));
        } else {
            std::vector<int> pg(nt, -1);
            upload_ssbo(ssbo_tri_param, pg.data(), (GLsizeiptr)(nt * sizeof(int)));
        }

        if (ng > 0) {
            upload_ssbo(ssbo_group_kind, st.tri_group_parametric_kind.data(),
                        (GLsizeiptr)(ng * sizeof(int)));

            /* Build neural payload offset table and measure total size.
             * Both MLP (kind=4) and PARAMETRIC_LENS (kind=5) share NeuralPayBuf
             * so the GPU shader can read either via the same offset mechanism. */
            std::vector<int> neural_pay_off(static_cast<size_t>(ng), -1);
            size_t total_nf = 0;
            for (int gi = 0; gi < ng; ++gi) {
                const int k = st.tri_group_parametric_kind[static_cast<size_t>(gi)];
                if (k == TRI_PARAM_SURFACE_NEURAL_ASSEMBLY ||
                    k == TRI_PARAM_SURFACE_PARAMETRIC_LENS) {
                    const auto& pb = st.tri_group_parametric_payload[static_cast<size_t>(gi)];
                    if (!pb.empty()) {
                        neural_pay_off[static_cast<size_t>(gi)] = static_cast<int>(total_nf);
                        total_nf += pb.size() / sizeof(float);
                    }
                }
            }

            /* Only re-upload the large payload buffer when content changed. */
            if (total_nf != neural_pay_floats_uploaded) {
                std::vector<float> neural_pay_buf;
                neural_pay_buf.reserve(total_nf);
                for (int gi = 0; gi < ng; ++gi) {
                    const int k = st.tri_group_parametric_kind[static_cast<size_t>(gi)];
                    if (k == TRI_PARAM_SURFACE_NEURAL_ASSEMBLY ||
                        k == TRI_PARAM_SURFACE_PARAMETRIC_LENS) {
                        const auto& pb = st.tri_group_parametric_payload[static_cast<size_t>(gi)];
                        if (!pb.empty()) {
                            const float* src = reinterpret_cast<const float*>(pb.data());
                            int nf = static_cast<int>(pb.size() / sizeof(float));
                            neural_pay_buf.insert(neural_pay_buf.end(), src, src + nf);
                        }
                    }
                }
                if (!neural_pay_buf.empty())
                    upload_ssbo(ssbo_neural_pay, neural_pay_buf.data(),
                                (GLsizeiptr)(neural_pay_buf.size() * sizeof(float)));
                else {
                    float stub = 0.f;
                    upload_ssbo(ssbo_neural_pay, &stub, sizeof(float));
                }
                neural_pay_floats_uploaded = total_nf;
            }

            /* Build group_pay: small inline payload or neural offset. */
            std::vector<float> pay(static_cast<size_t>(ng) * GPS, 0.0f);
            for (int gi = 0; gi < ng; ++gi) {
                int kind = st.tri_group_parametric_kind[static_cast<size_t>(gi)];
                const auto& pb = st.tri_group_parametric_payload[static_cast<size_t>(gi)];
                float* dst = pay.data() + gi * GPS;
                if (kind == TRI_PARAM_SURFACE_POLY_BARY && pb.size() >= 6 * sizeof(double)) {
                    const double* src = reinterpret_cast<const double*>(pb.data());
                    for (int i = 0; i < 6; ++i) dst[i] = (float)src[i];
                } else if (kind == TRI_PARAM_SURFACE_SDF_SPHERE && pb.size() >= sizeof(double)) {
                    dst[0] = (float)(*reinterpret_cast<const double*>(pb.data()));
                } else if (kind == TRI_PARAM_SURFACE_NEURAL_ASSEMBLY ||
                           kind == TRI_PARAM_SURFACE_PARAMETRIC_LENS) {
                    int off = neural_pay_off[static_cast<size_t>(gi)];
                    std::memcpy(dst, &off, sizeof(int));
                }
            }
            upload_ssbo(ssbo_group_pay, pay.data(),
                        (GLsizeiptr)(ng * GPS * sizeof(float)));
            n_param_groups = ng;
        } else {
            int   stub_i = 0;    upload_ssbo(ssbo_group_kind,  &stub_i, sizeof(int));
            float stub_f = 0.f;  upload_ssbo(ssbo_group_pay,   &stub_f, sizeof(float));
            if (neural_pay_floats_uploaded != 0) {
                upload_ssbo(ssbo_neural_pay, &stub_f, sizeof(float));
                neural_pay_floats_uploaded = 0;
            }
            n_param_groups = 0;
        }
    }

    /* ── init ──────────────────────────────────────────────────────────────────── */
    bool init(RayPipelineState& ps, const std::string& shader_dir) {
        if (!gl_compute_create_context(&ctx,
                reinterpret_cast<void*>((uintptr_t)ps.cfg.gl_display_hglrc),
                reinterpret_cast<void*>((uintptr_t)ps.cfg.gl_display_hdc))) {
            fprintf(stderr, "[gpu-dispatch] gl_compute_create_context failed\n"); fflush(stderr);
            return false;
        }
        if (!gl_compute_make_current(&ctx)) {
            fprintf(stderr, "[gpu-dispatch] gl_compute_make_current failed\n"); fflush(stderr);
            return false;
        }
        if (!gl_compute_load_procs()) {
            fprintf(stderr, "[gpu-dispatch] gl_compute_load_procs failed\n"); fflush(stderr);
            return false;
        }
        display_context_shared =
            (ps.cfg.gl_display_hglrc != 0 && std::strcmp(ctx.error, "no-share") != 0);

        /* Apply per-stage initial GPU batch sizes from config */
        auto& cfg = ps.cfg;
        if (cfg.gpu_batch_size_t1 > 0) ps.stats[0].batch_sz_gpu.store(cfg.gpu_batch_size_t1, std::memory_order_relaxed);
        if (cfg.gpu_batch_size_t2 > 0) ps.stats[1].batch_sz_gpu.store(cfg.gpu_batch_size_t2, std::memory_order_relaxed);
        if (cfg.gpu_batch_size_t3 > 0) ps.stats[2].batch_sz_gpu.store(cfg.gpu_batch_size_t3, std::memory_order_relaxed);
        if (cfg.gpu_batch_size_t4 > 0) ps.stats[3].batch_sz_gpu.store(cfg.gpu_batch_size_t4, std::memory_order_relaxed);
        if (cfg.gpu_batch_size_t5 > 0) ps.stats[4].batch_sz_gpu.store(cfg.gpu_batch_size_t5, std::memory_order_relaxed);
        if (cfg.t5_light_batch_size  > 0) t5_light_batch_size  = cfg.t5_light_batch_size;
        if (cfg.t5_cam_batch_size    > 0) t5_cam_batch_size    = cfg.t5_cam_batch_size;
        if (cfg.t5_sensor_tile_size  > 0) t5_sensor_tile_size  = cfg.t5_sensor_tile_size;

        /* Apply pinned GPU fractions (if non-zero in config) */
        if (cfg.gpu_fraction_t1 > 0.0f) ps.stats[0].set_gpu_fraction(cfg.gpu_fraction_t1);
        if (cfg.gpu_fraction_t2 > 0.0f) ps.stats[1].set_gpu_fraction(cfg.gpu_fraction_t2);
        if (cfg.gpu_fraction_t3 > 0.0f) ps.stats[2].set_gpu_fraction(cfg.gpu_fraction_t3);
        if (cfg.gpu_fraction_t5 > 0.0f) ps.stats[4].set_gpu_fraction(cfg.gpu_fraction_t5);

        /* Compile all four compute shaders */
        char err[1024];

        /* Helper: read a file from shader_dir into a std::string. */
        auto read_shader_file = [&](const std::string& name, std::string& out) -> bool {
            std::string path = resolve_shader(shader_dir, name);
            FILE* f = nullptr;
#ifdef _MSC_VER
            fopen_s(&f, path.c_str(), "rb");
#else
            f = fopen(path.c_str(), "rb");
#endif
            if (!f) { snprintf(err, sizeof(err), "Cannot open %s", path.c_str()); return false; }
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            out.assign(static_cast<size_t>(sz), '\0');
            fread(&out[0], 1, static_cast<size_t>(sz), f); fclose(f);
            return true;
        };

        auto load = [&](const std::string& name, GLuint& prog) -> bool {
            std::string src;
            if (!read_shader_file(name, src)) return false;
            prog = gl_compute_build_program(src.c_str(), err, sizeof(err));
            return prog != 0;
        };

        /* load_with_preamble: injects bvh_shadow.glsl.inc (with binding #defines
         * prepended) into the shader after its #version line, using the native
         * multi-source glShaderSource path in gl_compute_build_program2. */
        auto load_with_shadow_bvh = [&](const std::string& name, GLuint& prog,
                                        int bvh_bind, int tri_id_bind, int tri_full_bind) -> bool {
            std::string src, inc;
            if (!read_shader_file(name, src)) return false;
            if (!read_shader_file("bvh_shadow.glsl.inc", inc)) {
                /* Missing .inc is non-fatal: fall back to plain load (no shadow test). */
                fprintf(stderr, "[gpu-dispatch] bvh_shadow.glsl.inc not found (%s) — shadow test disabled\n", err);
                fflush(stderr);
                prog = gl_compute_build_program(src.c_str(), err, sizeof(err));
                return prog != 0;
            }
            /* Build the macro-define block that sets the three binding points. */
            char defines[256];
            snprintf(defines, sizeof(defines),
                "#define SHADOW_BVH_BINDING      %d\n"
                "#define SHADOW_TRI_ID_BINDING   %d\n"
                "#define SHADOW_TRI_FULL_BINDING %d\n",
                bvh_bind, tri_id_bind, tri_full_bind);
            std::string preamble = std::string(defines) + inc;
            prog = gl_compute_build_program2(src.c_str(), preamble.c_str(), err, sizeof(err));
            return prog != 0;
        };

        if (!load("ray_bvh_intersect.comp.glsl",   prog_t1)) { snprintf(ctx.error, sizeof(ctx.error), "%s", err); return false; }
        if (!load("ray_refine.comp.glsl",             prog_t2)) { snprintf(ctx.error, sizeof(ctx.error), "%s", err); return false; }
        if (!load("ray_material.comp.glsl",           prog_t3)) { snprintf(ctx.error, sizeof(ctx.error), "%s", err); return false; }
        if (!load("ray_wave_bpm.comp.glsl",           prog_t4)) { snprintf(ctx.error, sizeof(ctx.error), "%s", err); return false; }
        /* T5 full-connect shader — compiled with BVH shadow preamble injected.
         * BvhBuf→binding 5, TriIdBuf→binding 6, TriFullBuf→binding 7.
         * Non-fatal; GPU T5 path disabled if absent or compilation fails. */
        if (!load_with_shadow_bvh("t5_full_connect.comp.glsl", prog_t5, 5, 6, 7)) {
            fprintf(stderr, "[gpu-dispatch] t5_full_connect.comp.glsl not loaded (%s) — GPU T5 disabled\n", err);
            fflush(stderr);
            prog_t5 = 0;
        }

        /* Load UV blit shader — non-fatal, falls back to CPU path silently */
        if (display_context_shared) {
            if (!load("uv_blit.comp.glsl", prog_uv_blit)) {
                fprintf(stderr, "[gpu-dispatch] uv_blit.comp.glsl not loaded (%s) — GPU blit disabled\n", err);
                fflush(stderr);
                prog_uv_blit = 0;
            } else {
                uloc_blit.n_uv_groups  = glc_GetUniformLocation(prog_uv_blit, "n_uv_groups");
                uloc_blit.uv_meta_base = glc_GetUniformLocation(prog_uv_blit, "uv_meta_base");
                uloc_blit.n_bands      = glc_GetUniformLocation(prog_uv_blit, "n_bands");
                uloc_blit.uv_blit_mode = glc_GetUniformLocation(prog_uv_blit, "uv_blit_mode");
                uloc_blit.rgb_w        = glc_GetUniformLocation(prog_uv_blit, "rgb_w");
            }
        }

        /* Cache uniform locations — string lookup done once at init, not per dispatch */
        uloc_t1.n_intents = glc_GetUniformLocation(prog_t1, "n_intents");
        uloc_t1.n_bands   = glc_GetUniformLocation(prog_t1, "n_bands");
        uloc_t1.n_mats    = glc_GetUniformLocation(prog_t1, "n_mats");
        uloc_t1.n_arenas  = glc_GetUniformLocation(prog_t1, "n_arenas");
        uloc_t1.n_tris    = glc_GetUniformLocation(prog_t1, "n_tris");
        uloc_t1.u_arenas  = glc_GetUniformLocation(prog_t1, "u_arenas");

        uloc_t2.n_tris    = glc_GetUniformLocation(prog_t2, "n_tris");
        uloc_t2.n_groups  = glc_GetUniformLocation(prog_t2, "n_groups");
        uloc_t2.n_bands   = glc_GetUniformLocation(prog_t2, "n_bands");
        uloc_t2.freq_hz   = glc_GetUniformLocation(prog_t2, "freq_hz");
        uloc_t2.bdpt_max_optical = glc_GetUniformLocation(prog_t2, "bdpt_max_optical");
        uloc_t2.bdpt_optical_base = glc_GetUniformLocation(prog_t2, "bdpt_optical_base");

        uloc_t3.n_bands              = glc_GetUniformLocation(prog_t3, "n_bands");
        uloc_t3.n_mats               = glc_GetUniformLocation(prog_t3, "n_mats");
        uloc_t3.max_children         = glc_GetUniformLocation(prog_t3, "max_children");
        uloc_t3.max_children_per_hit = glc_GetUniformLocation(prog_t3, "max_children_per_hit");
        uloc_t3.sensor_res           = glc_GetUniformLocation(prog_t3, "sensor_res");
        uloc_t3.sensor_pr            = glc_GetUniformLocation(prog_t3, "sensor_pr");
        uloc_t3.sensor_px            = glc_GetUniformLocation(prog_t3, "sensor_px");
        uloc_t3.n_uv_groups          = glc_GetUniformLocation(prog_t3, "n_uv_groups");
        uloc_t3.uv_group_id_base     = glc_GetUniformLocation(prog_t3, "uv_group_id_base");
        uloc_t3.uv_meta_base         = glc_GetUniformLocation(prog_t3, "uv_meta_base");
        uloc_t3.rng_seed             = glc_GetUniformLocation(prog_t3, "rng_seed");
        uloc_t3.bdpt_max_verts       = glc_GetUniformLocation(prog_t3, "bdpt_max_verts");
        uloc_t3.bdpt_max_spectral    = glc_GetUniformLocation(prog_t3, "bdpt_max_spectral");
        uloc_t3.bdpt_max_pdfs        = glc_GetUniformLocation(prog_t3, "bdpt_max_pdfs");
        uloc_t3.bdpt_count_base      = glc_GetUniformLocation(prog_t3, "bdpt_count_base");
        uloc_t3.bdpt_spectral_base   = glc_GetUniformLocation(prog_t3, "bdpt_spectral_base");
        uloc_t3.bdpt_pdf_base        = glc_GetUniformLocation(prog_t3, "bdpt_pdf_base");
        uloc_t3.n_hits               = glc_GetUniformLocation(prog_t3, "n_hits");
        /* !!DIAG!! Print T3 BDPT uniform locations.  -1 means the GLSL compiler
         * dead-stripped that uniform (it considers it unreachable).  If
         * uloc_max_pdfs=-1 the glUniform upload is a no-op → bdpt_max_pdfs=0
         * in the shader → emit_bdpt_pdf always returns early → pdfs=0 forever. */
        fprintf(stderr,
            "[gpu-t3-uloc] bdpt_max_verts=%d bdpt_max_spectral=%d bdpt_max_pdfs=%d "
            "bdpt_count_base=%d bdpt_spectral_base=%d bdpt_pdf_base=%d\n",
            (int)uloc_t3.bdpt_max_verts,
            (int)uloc_t3.bdpt_max_spectral,
            (int)uloc_t3.bdpt_max_pdfs,
            (int)uloc_t3.bdpt_count_base,
            (int)uloc_t3.bdpt_spectral_base,
            (int)uloc_t3.bdpt_pdf_base);
        fflush(stderr);

        uloc_t4.mode        = glc_GetUniformLocation(prog_t4, "mode");
        uloc_t4.nx          = glc_GetUniformLocation(prog_t4, "nx");
        uloc_t4.ny          = glc_GetUniformLocation(prog_t4, "ny");
        uloc_t4.n_bands     = glc_GetUniformLocation(prog_t4, "n_bands");
        uloc_t4.dx          = glc_GetUniformLocation(prog_t4, "dx");
        uloc_t4.dz          = glc_GetUniformLocation(prog_t4, "dz");
        uloc_t4.wavelengths = glc_GetUniformLocation(prog_t4, "wavelengths");

        ready = true;
        return true;
    }

    /* ── upload_scene_data ──────────────────────────────────────────────── */
    void upload_scene_data(RayPipelineState& ps) {
        const RayTracerState& st = *ps.st;

        /* BVH nodes: 10 floats each */
        {
            const int nn = (int)st.bvh_nodes.size();
            std::vector<float> buf(nn * 10);
            for (int i = 0; i < nn; ++i) {
                const auto& nd = st.bvh_nodes[static_cast<size_t>(i)];
                float* b = buf.data() + i * 10;
                b[0] = (float)nd.aabb.lo.x(); b[1] = (float)nd.aabb.lo.y(); b[2] = (float)nd.aabb.lo.z();
                b[3] = (float)nd.aabb.hi.x(); b[4] = (float)nd.aabb.hi.y(); b[5] = (float)nd.aabb.hi.z();
                b[6] = rt_i32_as_f32((int32_t)nd.left);
                b[7] = rt_i32_as_f32((int32_t)nd.right);
                b[8] = rt_i32_as_f32((int32_t)nd.tri_start);
                b[9] = rt_i32_as_f32((int32_t)nd.tri_end);
            }
            upload_ssbo(ssbo_bvh, buf.data(), (GLsizeiptr)(nn * 10 * sizeof(float)));
        }

        /* BVH permutation indices */
        if (!st.bvh_tri_ids.empty())
            upload_ssbo(ssbo_tri_id, st.bvh_tri_ids.data(),
                        (GLsizeiptr)(st.bvh_tri_ids.size() * sizeof(int)));

        /* Triangle full geometry: 16 floats each.  Layout:
         *   [0..2]  = v0  (origin vertex)
         *   [3..5]  = edge1 = v1-v0
         *   [6..8]  = edge2 = v2-v0
         *   [9..11] = normal
         *   [12]    = flags (int reinterpreted)
         *   [13]    = mat_idx
         *   [14]    = medium_pos_mat_idx
         *   [15]    = medium_neg_mat_idx
         */
        {
            const int nt = (int)st.tris.size();
            std::vector<float> buf(nt * 16);
            for (int i = 0; i < nt; ++i) {
                const auto& tri = st.tris[static_cast<size_t>(i)];
                float* b = buf.data() + i * 16;
                b[0]=(float)tri.v0.x();    b[1]=(float)tri.v0.y();    b[2]=(float)tri.v0.z();
                b[3]=(float)tri.edge1.x(); b[4]=(float)tri.edge1.y(); b[5]=(float)tri.edge1.z();
                b[6]=(float)tri.edge2.x(); b[7]=(float)tri.edge2.y(); b[8]=(float)tri.edge2.z();
                b[9]=(float)tri.normal.x();b[10]=(float)tri.normal.y();b[11]=(float)tri.normal.z();
                b[12] = rt_i32_as_f32((int32_t)tri.flags);
                b[13] = rt_i32_as_f32((int32_t)tri.mat_idx);
                b[14] = rt_i32_as_f32((int32_t)tri.medium_pos_mat_idx);
                b[15] = rt_i32_as_f32((int32_t)tri.medium_neg_mat_idx);
            }
            upload_ssbo(ssbo_tri_full, buf.data(), (GLsizeiptr)(nt * 16 * sizeof(float)));
        }

        /* Material band records: pass mat_buf directly */
        if (!st.mat_buf.empty())
            upload_ssbo(ssbo_mat_band, st.mat_buf.data(),
                        (GLsizeiptr)(st.mat_buf.size() * sizeof(float)));

        /* Scene band buffer: k_real[0..nb-1] | atmo_abs[0..nb-1] */
        {
            const int nb = st.n_bands;
            std::vector<float> sb(nb * 2);
            for (int b = 0; b < nb; ++b) {
                sb[b]      = (float)st.k_real[b];
                sb[nb + b] = (float)st.atmo_abs[b];
            }
            upload_ssbo(ssbo_scene_band, sb.data(), (GLsizeiptr)(nb * 2 * sizeof(float)));
        }

        /* Wave arenas: cache as uniform data (max 16 × 4 floats: xyz center + radius). */
        {
            const int na = std::min((int)ps.arenas.size(), 16);
            arena_uniform_data.assign(na * 4, 0.0f);
            for (int i = 0; i < na; ++i) {
                const auto& a = ps.arenas[static_cast<size_t>(i)];
                arena_uniform_data[i*4]   = (float)a.center.x();
                arena_uniform_data[i*4+1] = (float)a.center.y();
                arena_uniform_data[i*4+2] = (float)a.center.z();
                arena_uniform_data[i*4+3] = (float)a.radius;
            }
        }
        /* TriSensorGroupBuf removed — T1 shader no longer has that binding. */

        /* Parametric group data — initial upload via shared helper.
         * The helper is also called before every T2 dispatch to re-sync after
         * clear_tri_groups() + re-registration (see dispatch_t1_t2_t3). */
        upload_param_groups_data(st);

        /* Counter SSBO: 8 uints — allocate once, zeroed on each dispatch */
        ensure_ssbo(ssbo_counter, 8 * sizeof(uint32_t));

        /* T3 meta SSBO: [0]=intent_count [1]=terminal_count [2..2+nm-1]=eps_flags
         * Allocate as 2+nm uints.  Counters are zeroed before each dispatch;
         * eps_flags are written here and survive across dispatches. */
        {
            const int nm2 = std::max(1, st.mat_n_mats);  /* at least 1 so the SSBO is non-empty */
            /* +3 tail slots: [2+nm2..2+nm2+2] = bdpt_vert_count, bdpt_spectral_count, bdpt_pdf_count.
             * Zeroed before each dispatch; eps_flags [2..2+nm2) persist across dispatches. */
            std::vector<uint32_t> meta(2 + nm2 + 3, 0u);
            const auto& ef = ps.mat_epsilon_flags;
            for (int i = 0; i < nm2 && i < (int)ef.size(); ++i)
                meta[2 + i] = ef[i] ? 1u : 0u;
            upload_ssbo(ssbo_t3_meta, meta.data(),
                        (GLsizeiptr)((2 + nm2 + 3) * sizeof(uint32_t)));
        }

        {
            const auto& root = st.bvh_nodes[0];
            fprintf(stderr,
                "[gpu-upload] scene: %d BVH nodes, %d tris, %d mats, n_bands=%d, n_param_groups=%d\n"
                "[gpu-upload] root AABB: lo=(%.4f,%.4f,%.4f) hi=(%.4f,%.4f,%.4f) left=%d right=%d\n",
                (int)st.bvh_nodes.size(), (int)st.tris.size(),
                st.mat_n_mats, st.n_bands, n_param_groups,
                (float)root.aabb.lo.x(), (float)root.aabb.lo.y(), (float)root.aabb.lo.z(),
                (float)root.aabb.hi.x(), (float)root.aabb.hi.y(), (float)root.aabb.hi.z(),
                root.left, root.right);
            fflush(stderr);
        }

        /* UV integrator image SSBOs.
         * MergedUvBuf (binding 5) layout: [n_tris*6 UV coords as int-bits] ++
         *   [n_tris group-ids (int)] ++ [n_groups*2 meta ints].
         * ssbo_uv_accum (binding 6) is zeroed after each T3 dispatch. */
        {
            const int nt     = (int)st.tris.size();
            const int nt_uv  = (int)st.tri_uv_data.size() / 6;  /* tris with UV data */
            const int ng     = (int)st.group_uv_res.size();
            const size_t n_coord_ints = (size_t)nt * 6;
            const size_t n_id_ints    = (size_t)nt;
            const size_t n_meta_ints  = (size_t)ng * 2;
            const size_t total        = std::max((size_t)1,
                                                 n_coord_ints + n_id_ints + n_meta_ints);
            std::vector<int32_t> merged(total, -1);
            /* Section 1: UV vertex coords as floatBitsToInt, stride 6 per tri */
            for (int i = 0; i < nt_uv && i < nt; ++i) {
                for (int c = 0; c < 6; ++c) {
                    float fv = st.tri_uv_data[static_cast<size_t>(i) * 6 + c];
                    int32_t iv;
                    static_assert(sizeof(float) == sizeof(int32_t), "size mismatch");
                    std::memcpy(&iv, &fv, sizeof(int32_t));
                    merged[static_cast<size_t>(i) * 6 + c] = iv;
                }
            }
            /* Section 2: per-tri UV group IDs */
            for (size_t i = 0; i < st.tri_uv_group_of_tri.size(); ++i)
                merged[n_coord_ints + i] = st.tri_uv_group_of_tri[i];
            /* Section 3: group metadata (res, accum_offset pairs) */
            for (int g = 0; g < ng; ++g) {
                merged[n_coord_ints + n_id_ints + (size_t)g * 2 + 0] = st.group_uv_res[(size_t)g];
                merged[n_coord_ints + n_id_ints + (size_t)g * 2 + 1] = st.group_uv_accum_offset[(size_t)g];
            }
            upload_ssbo(ssbo_merged_uv, merged.data(),
                        (GLsizeiptr)(total * sizeof(int32_t)));

            /* Accumulator (binding 7): allocate to match CPU side; zero-init once. */
            const int n_accum = st.uv_accum_total;
            if (n_accum > 0)
                ensure_ssbo(ssbo_uv_accum, (GLsizeiptr)(n_accum * sizeof(uint32_t)));
            else {
                uint32_t stub_u = 0; ensure_ssbo(ssbo_uv_accum, sizeof(uint32_t));
            }
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_uv_accum);
            {
                const GLsizeiptr clr_sz = (n_accum > 0)
                    ? (GLsizeiptr)(n_accum * sizeof(uint32_t)) : sizeof(uint32_t);
                std::vector<uint32_t> zeros(n_accum > 0 ? n_accum : 1, 0u);
                glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, clr_sz, zeros.data());
            }
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            /* Allocate shared tex_uv_pages ring when WGL context sharing is active.
             * The display samples only a completed generation; the worker writes a
             * different slot so a render frame never reads the texture being built. */
            if (prog_uv_blit != 0 && display_context_shared) {
                const int ng  = (int)st.group_uv_res.size();
                const int res = (ng > 0) ? st.group_uv_res[0] : 512;
                if (ng != blit_n_layers || res != blit_res) {
                    for (GLsync& f : uv_fences) {
                        if (f && glc_DeleteSync) glc_DeleteSync(f);
                        f = nullptr;
                    }
                    for (GLuint& tex : tex_uv_pages) {
                        if (tex) { glDeleteTextures(1, &tex); tex = 0; }
                    }
                    uv_build_slot = 0;
                    uv_active_slot = -1;
                    uv_pending_slot = -1;
                    if (ng > 0) {
                        glGenTextures(UV_TEX_RING, tex_uv_pages.data());
                        for (GLuint tex : tex_uv_pages) {
                            glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
                            glc_TexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA16F, res, res, ng);
                            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        }
                        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
                        blit_n_layers = ng;
                        blit_res      = res;
                    } else {
                        blit_n_layers = 0;
                        blit_res      = 0;
                    }
                }
            }
        }

        scene_uploaded = true;
    }

    /* ── dispatch_uv_blit: GPU-direct accumulator → RGBA16F display texture ─ */
    void dispatch_uv_blit(RayPipelineState& ps) {
        if (!prog_uv_blit || blit_n_layers <= 0) return;
        int slot = -1;
        {
            std::lock_guard<std::mutex> lk(uv_publish_mu);
            if (uv_pending_slot >= 0) return; /* Display has not adopted the prior generation. */
            for (int i = 0; i < UV_TEX_RING; ++i) {
                const int cand = (uv_build_slot + i) % UV_TEX_RING;
                if (cand != uv_active_slot && tex_uv_pages[(size_t)cand] != 0) {
                    slot = cand;
                    break;
                }
            }
            if (slot < 0) return;
            uv_build_slot = (slot + 1) % UV_TEX_RING;
            if (uv_fences[(size_t)slot] && glc_DeleteSync) {
                glc_DeleteSync(uv_fences[(size_t)slot]);
                uv_fences[(size_t)slot] = nullptr;
            }
        }
        const RayTracerState& st = *ps.st;
        const int n_uv_groups  = (int)st.group_uv_res.size();
        /* MergedUvBuf section 3 (group metadata) starts at index n_tris*7.
         * The uv_blit shader reads uv_meta[uv_meta_base + g*2 + {0,1}]. */
        const int n_tris_blit  = (int)st.tris.size();
        const int uv_meta_base = n_tris_blit * 7;
        const int nb           = (blit_n_bands_stored > 0) ? blit_n_bands_stored : st.n_bands;
        const int res          = blit_res;
        const GLuint tex       = tex_uv_pages[(size_t)slot];
        if (!tex) return;

        /* Ensure T3 image stores are visible before we read the SSBO. */
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        /* Bind resources — binding 6 = MergedUvBuf (contains group metadata at section 3) */
        glc_BindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo_uv_accum);
        glc_BindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_merged_uv);
        glc_BindImageTexture(UV_BLIT_IMAGE_UNIT, tex, 0, GL_TRUE, 0,
                             GL_WRITE_ONLY, GL_RGBA16F);

        glc_UseProgram(prog_uv_blit);
        glc_Uniform1i(uloc_blit.n_uv_groups,  n_uv_groups);
        glc_Uniform1i(uloc_blit.uv_meta_base, uv_meta_base);
        glc_Uniform1i(uloc_blit.n_bands,       nb);
        glc_Uniform1i(uloc_blit.uv_blit_mode,  blit_mode);
        glc_Uniform3fv(uloc_blit.rgb_w, std::min(nb, MAX_SPECTRAL_BANDS), blit_rgb_weights.data());

        const unsigned int gx = (unsigned int)((res + 7) / 8);
        const unsigned int gy = (unsigned int)((res + 7) / 8);
        const unsigned int gz = (unsigned int)n_uv_groups;
        GLC_DISPATCH_CHECKED("UV:blit", gx, gy, gz);

        /* Ensure image writes complete before the display context reads the texture. */
        glc_MemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        GLsync fence = glc_FenceSync ? glc_FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0) : nullptr;
        {
            std::lock_guard<std::mutex> lk(uv_publish_mu);
            uv_fences[(size_t)slot] = fence;
            uv_pending_slot = slot;
            ++uv_generation;
        }
    }

    /* Return the latest completed UV texture ID without blocking the display context. */
    uint64_t get_uv_pages_tex_id() {
        if (!display_context_shared) return 0u;
        std::lock_guard<std::mutex> lk(uv_publish_mu);
        if (uv_pending_slot >= 0) {
            GLsync fence = uv_fences[(size_t)uv_pending_slot];
            GLenum rc = GL_ALREADY_SIGNALED;
            if (fence && glc_ClientWaitSync)
                rc = glc_ClientWaitSync(fence, 0, 0);
            if (rc == GL_ALREADY_SIGNALED || rc == GL_CONDITION_SATISFIED) {
                if (fence && glc_DeleteSync)
                    glc_DeleteSync(fence);
                uv_fences[(size_t)uv_pending_slot] = nullptr;
                uv_active_slot = uv_pending_slot;
                uv_pending_slot = -1;
            } else if (rc == GL_WAIT_FAILED) {
                if (fence && glc_DeleteSync)
                    glc_DeleteSync(fence);
                uv_fences[(size_t)uv_pending_slot] = nullptr;
                uv_pending_slot = -1;
            }
        }
        if (uv_active_slot < 0) return 0u;
        return (uint64_t)tex_uv_pages[(size_t)uv_active_slot];
    }

    /* ── dispatch_t1_t2_t3: run one GPU batch through T1→T2→T3 ──────────── */
    int dispatch_t1_t2_t3(RayPipelineState& ps,
                           const std::vector<RayIntent>& batch,
                           int64_t* out_flash_children = nullptr,
                           int64_t* out_sensor_children = nullptr)
    {
        using Clock = std::chrono::high_resolution_clock;
        const RayTracerState& st = *ps.st;
        int       n  = (int)batch.size();  /* mutable: updated to child count each bounce */
        const int nb = st.n_bands;
        const int nm = st.mat_n_mats;
        const int na = (int)ps.arenas.size();
        const int nt = (int)st.tris.size();

        /* ── GPU-resident bounce loop ─────────────────────────────────────────
         * Generation 0: pack batch from CPU → ssbo_intent and upload.
         * Generation k>0: ssbo_intent was swapped to the old ssbo_child_int at
         * the end of the previous bounce — child intents are already GPU-resident
         * in INTENT_STRIDE format, no PCIe transfer between generations.         */
        bool first_gen    = true;
        int  total_n_hits = 0;
        /* BDPT vertex/spectral/pdf counts accumulate across bounces in ssbo_t3_meta.
         * ssbo_counter[4] (optical write pointer) also accumulates.  Track the
         * previous totals so we read only the new delta records each bounce.   */
        int prev_nv = 0, prev_ns = 0, prev_np = 0, prev_no = 0;
        for (;;) {  /* ── bounce loop ── */

        /* ── Ensure intent SSBO is large enough (INTENT_STRIDE = 20 + 2*MAX_SPECTRAL_BANDS floats) */
        static constexpr int INTENT_STRIDE = 20 + 2 * MAX_SPECTRAL_BANDS;
        if (n > cap_intents) {
            cap_intents = n * 2;
            ensure_ssbo(ssbo_intent, (GLsizeiptr)(cap_intents * INTENT_STRIDE * sizeof(float)));
        }

        if (first_gen) {
        /* Pack intents to flat float buffer — persistent staging, no malloc after warmup */
        stg_ibuf.resize((size_t)n * INTENT_STRIDE);
        auto& ibuf = stg_ibuf;
        for (int i = 0; i < n; ++i) {
            const RayIntent& ri = batch[static_cast<size_t>(i)];
            float* row = ibuf.data() + i * INTENT_STRIDE;
            row[0]=(float)ri.pos.x();  row[1]=(float)ri.pos.y();  row[2]=(float)ri.pos.z();
            row[3]=(float)ri.dir.x();  row[4]=(float)ri.dir.y();  row[5]=(float)ri.dir.z();
            row[6]=(float)ri.path_len;
            row[7]  = rt_i32_as_f32((int32_t)ri.medium_mat_idx);
            row[8]  = rt_u32_as_f32((uint32_t)ri.interaction_flags);
            row[9]  = rt_i32_as_f32((int32_t)ri.src_id);
            row[10] = rt_i32_as_f32((int32_t)ri.bounce);
            row[11] = rt_i32_as_f32((int32_t)ri.bounces_left);
            row[12] = (float)ri.min_amplitude;
            uint32_t tag_lo = (uint32_t)(ri.tag & 0xFFFFFFFFULL);
            uint32_t tag_hi = (uint32_t)(ri.tag >> 32);
            row[13] = rt_u32_as_f32(tag_lo);
            row[14] = rt_u32_as_f32(tag_hi);
            row[15] = rt_u32_as_f32((uint32_t)ri.color_flag);
            row[16] = ri.priority;
            row[17] = ri.sensor_origin_y; row[18] = ri.sensor_origin_z;
            row[19] = rt_u32_as_f32((uint32_t)ri.bdpt_subpath_id); /* bdpt_subpath_id in _pad */
            const int bands = std::min(nb, MAX_SPECTRAL_BANDS);
            for (int b = 0; b < bands; ++b) {
                row[20+b] = (float)ri.amp[b].real();
                row[20+MAX_SPECTRAL_BANDS+b] = (float)ri.amp[b].imag();
            }
        }
        /* Upload intents */
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_intent);
        glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(n * INTENT_STRIDE * sizeof(float)), ibuf.data());
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        } /* end first_gen intent upload */
        /* else: ssbo_intent was swapped from old ssbo_child_int; child intents
         * are already GPU-resident in INTENT_STRIDE format — no upload needed. */

        /* ── Ensure hit SSBO (HIT_STRIDE = 27 + 2*MAX_SPECTRAL_BANDS floats; bdpt_sid at [26+2*MAX_SPECTRAL_BANDS]) */
        static constexpr int HIT_STRIDE = 27 + 2 * MAX_SPECTRAL_BANDS;
        int max_hits = n * 4;
        if (max_hits > cap_hits) {
            cap_hits = max_hits * 2;
            ensure_ssbo(ssbo_hit, (GLsizeiptr)(cap_hits * HIT_STRIDE * sizeof(float)));
        }

        /* ── Ensure BDPT output SSBO (single buffer, binding 7) ─────────── *
         * Layout: [verts 0..bdpt_spectral_base_f) ++ [spectral ..bdpt_pdf_base_f) ++ [pdfs ..)
         * Counts live in MetaBuf tail at [2+nm..2+nm+2] (no separate counter SSBO). */
        static constexpr int BDPT_VERTEX_STRIDE_F   = 28;  /* floats per BdptVertexRecord */
        static constexpr int BDPT_SPECTRAL_STRIDE_F =  8;  /* floats per BdptSpectralWeightRecord */
        static constexpr int BDPT_PDF_STRIDE_F      = 12;  /* floats per BdptPdfRecord */
        static constexpr int BDPT_OPTICAL_STRIDE_F  = 28;  /* floats per BdptOpticalEventRecord */
        const int max_bdpt_v = (ps.cfg.bdpt_max_vertices > 0) ? ps.cfg.bdpt_max_vertices : 2000000;
        const int max_bdpt_s = (ps.cfg.bdpt_max_spectral > 0) ? ps.cfg.bdpt_max_spectral : 4000000;
        const int max_bdpt_p = (ps.cfg.bdpt_max_pdfs > 0) ? ps.cfg.bdpt_max_pdfs : 2000000;
        const int max_bdpt_o = (ps.cfg.bdpt_max_optical > 0) ? ps.cfg.bdpt_max_optical : 1000000;
        const int  bdpt_spectral_base_f = max_bdpt_v * BDPT_VERTEX_STRIDE_F;
        const int  bdpt_pdf_base_f      = bdpt_spectral_base_f + max_bdpt_s * BDPT_SPECTRAL_STRIDE_F;
        const int  bdpt_optical_base_f  = bdpt_pdf_base_f + max_bdpt_p * BDPT_PDF_STRIDE_F;
        const int64_t bdpt_total_f      = (int64_t)bdpt_optical_base_f + (int64_t)max_bdpt_o * BDPT_OPTICAL_STRIDE_F;
        if (bdpt_total_f > cap_bdpt_output) {
            cap_bdpt_output = (int)bdpt_total_f;
            ensure_ssbo(ssbo_bdpt_output, (GLsizeiptr)(bdpt_total_f * sizeof(float)));
        }

        /* Zero per-bounce counters.  On first gen: zero all 8 (including [4]=optical
         * write pointer).  On subsequent gens: zero only [0..3] (hit/miss/wave/cpu_refine)
         * — counter[4] must keep accumulating so T2's append offset is preserved.   */
        {
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_counter);
            if (first_gen) {
                uint32_t zeros[8] = {};
                glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8 * sizeof(uint32_t), zeros);
            } else {
                uint32_t z4[4] = {};
                glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * sizeof(uint32_t), z4);
            }
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
        /* Zero BDPT count slots in MetaBuf tail once — they accumulate across bounces. */
        if (first_gen) {
            uint32_t bzeros[3] = {};
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_t3_meta);
            glc_BufferSubData(GL_SHADER_STORAGE_BUFFER,
                              (GLintptr)((2 + nm) * sizeof(uint32_t)),
                              3 * sizeof(uint32_t), bzeros);
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        /* ── T1 dispatch ──────────────────────────────────────────────═ */
        auto t1_start = Clock::now();
        glc_UseProgram(prog_t1);
        bind_ssbo(ssbo_intent,     0); bind_ssbo(ssbo_hit,       1);
        bind_ssbo(ssbo_counter,    2); bind_ssbo(ssbo_bvh,       3);
        bind_ssbo(ssbo_tri_id,     4); bind_ssbo(ssbo_tri_full,  5);
        bind_ssbo(ssbo_mat_band,   6); bind_ssbo(ssbo_scene_band,7);
        glc_Uniform1i(uloc_t1.n_intents, n);
        glc_Uniform1i(uloc_t1.n_bands,   nb);
        glc_Uniform1i(uloc_t1.n_mats,    nm);
        glc_Uniform1i(uloc_t1.n_arenas,  na);
        glc_Uniform1i(uloc_t1.n_tris,    nt);
        if (uloc_t1.u_arenas >= 0 && na > 0)
            glc_Uniform4fv(uloc_t1.u_arenas, na, arena_uniform_data.data());
        GLC_DISPATCH_CHECKED("T1:BVH", (GLuint)((n + 63) / 64), 1, 1);
        /* Shader-storage barrier only — no CPU readback here.
         * T2 and T3 read n_hits from counters[0] in the SSBO so we can
         * dispatch T1→T2→T3 as a single GPU command sequence with only
         * one CPU sync point (after T3).  This halves GPU↔CPU round-trips. */
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        /* ── T2 dispatch (in-place on ssbo_hit) ─────────────────────────═
         * Re-sync parametric SSBOs from current CPU state before every T2 dispatch.
         * This keeps GPU data consistent after clear_tri_groups() + re-registration
         * (e.g. from _ensure_bdpt_plate_sensor_group) which happens while the scene
         * is already uploaded.  ssbo_neural_pay is skipped when payload hasn't changed. */
        upload_param_groups_data(st);

        /* Invocations beyond the actual hit count early-return via the
         * counters[0] guard inside the shader — no CPU sync needed between stages. */
        if (n_param_groups > 0) {
            /* Reset counter[3] (cpu_refine count for SDF_SADDLE fallbacks) */
            {
                uint32_t z = 0;
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_counter);
                glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 3 * sizeof(uint32_t), sizeof(uint32_t), &z);
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            }
            glc_UseProgram(prog_t2);
            bind_ssbo(ssbo_hit,        0); bind_ssbo(ssbo_tri_full,  1);
            bind_ssbo(ssbo_tri_param,  2); bind_ssbo(ssbo_group_kind,3);
            bind_ssbo(ssbo_group_pay,  4); bind_ssbo(ssbo_counter,   5);
            bind_ssbo(ssbo_neural_pay, 6);
            bind_ssbo(ssbo_bdpt_output, 7);
            /* n_hits uniform removed — shader reads counters[0] from SSBO */
            glc_Uniform1i(uloc_t2.n_tris,   nt);
            glc_Uniform1i(uloc_t2.n_groups, n_param_groups);
            glc_Uniform1i(uloc_t2.n_bands,  nb);
            glc_Uniform1i(uloc_t2.bdpt_max_optical, max_bdpt_o);
            glc_Uniform1i(uloc_t2.bdpt_optical_base, bdpt_optical_base_f);
            if (uloc_t2.freq_hz >= 0) {
                float fhz[MAX_SPECTRAL_BANDS] = {};
                const int nbf = std::min(nb, MAX_SPECTRAL_BANDS);
                for (int i = 0; i < nbf; ++i)
                    fhz[i] = (float)st.freq_hz_vec[i];
                glc_Uniform1fv(uloc_t2.freq_hz, MAX_SPECTRAL_BANDS, fhz);
            }
            GLC_DISPATCH_CHECKED("T2:refine", (GLuint)((n + 63) / 64), 1, 1);
            glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        /* ── T3 dispatch ──────────────────────────────────────────────═ */
        /* CounterBuf is absent from T3 (all 8 bindings used); read n_hits now via CPU sync.
         * Cost: one GPU→CPU 4-byte transfer after T1/T2 barriers are already in flight. */
        int n_hits = 0;
        {
            uint32_t cnt0 = 0u;
            readback_ssbo(ssbo_counter, &cnt0, sizeof(uint32_t));
            n_hits = std::min((int)cnt0, cap_hits);
        }
        static constexpr int CHILD_STRIDE    = 20 + 2 * MAX_SPECTRAL_BANDS;  /* INTENT_STRIDE: 20 + 2*MAX_SPECTRAL_BANDS */
        static constexpr int TERMINAL_STRIDE = 26 + 2 * MAX_SPECTRAL_BANDS;  /* 26 + 2*MAX_SPECTRAL_BANDS */
        /* Size child buffer for worst-case (all n intents hit and spawn children). */
        int max_children = n * ps.cfg.max_children + 1;
        if (max_children > cap_children) {
            cap_children = max_children * 2;
            GLsizeiptr combined = (GLsizeiptr)((size_t)cap_children
                                               * (CHILD_STRIDE + TERMINAL_STRIDE)
                                               * sizeof(float));
            ensure_ssbo(ssbo_child_int, combined);
        }
        /* Zero only the counter pair in ssbo_t3_meta; eps_flags at [2+] persist */
        { uint32_t z[2] = {};
          glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_t3_meta);
          glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(uint32_t), z);
          glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0); }

        static uint32_t t3_rng_seed = 0;
        auto t3_start = Clock::now();
        glc_UseProgram(prog_t3);
        /* Bindings match ray_material.comp.glsl (8 slots, 0-7 — GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS=8):
         * 0=RefinedHitBuf  1=OutBuf  2=MetaBuf     3=TriangleBuf
         * 4=MatBandBuf     5=MergedUvBuf  6=UvAccumBuf  7=BdptOutputBuf */
        bind_ssbo(ssbo_hit,         0);
        bind_ssbo(ssbo_child_int,   1);
        bind_ssbo(ssbo_t3_meta,     2);
        bind_ssbo(ssbo_tri_full,    3);
        bind_ssbo(ssbo_mat_band,    4);
        bind_ssbo(ssbo_merged_uv,   5);
        bind_ssbo(ssbo_uv_accum,    6);
        bind_ssbo(ssbo_bdpt_output, 7);
        const int n_uv_groups      = (int)st.group_uv_res.size();
        const int uv_group_id_base = nt * 6;   /* MergedUvBuf section 2: per-tri group IDs */
        const int uv_meta_base     = nt * 7;   /* MergedUvBuf section 3: group metadata */
        const int bdpt_count_base  = 2 + nm;   /* MetaBuf tail: [2+nm..2+nm+2] BDPT counts */
        glc_Uniform1i (uloc_t3.n_bands,              nb);
        glc_Uniform1i (uloc_t3.n_mats,               nm);
        glc_Uniform1i (uloc_t3.max_children,         max_children);
        glc_Uniform1i (uloc_t3.max_children_per_hit, ps.cfg.max_children);
        glc_Uniform1i (uloc_t3.sensor_res,            0);
        glc_Uniform1f (uloc_t3.sensor_pr,             ps.sensor_half_w);
        glc_Uniform1f (uloc_t3.sensor_px,             ps.sensor_px);
        glc_Uniform1i (uloc_t3.n_uv_groups,           n_uv_groups);
        glc_Uniform1i (uloc_t3.uv_group_id_base,      uv_group_id_base);
        glc_Uniform1i (uloc_t3.uv_meta_base,          uv_meta_base);
        glc_Uniform1ui(uloc_t3.rng_seed,              ++t3_rng_seed);
        glc_Uniform1i (uloc_t3.bdpt_max_verts,        max_bdpt_v);
        glc_Uniform1i (uloc_t3.bdpt_max_spectral,     max_bdpt_s);
        glc_Uniform1i (uloc_t3.bdpt_max_pdfs,         max_bdpt_p);
        glc_Uniform1i (uloc_t3.bdpt_count_base,       bdpt_count_base);
        glc_Uniform1i (uloc_t3.bdpt_spectral_base,    bdpt_spectral_base_f);
        glc_Uniform1i (uloc_t3.bdpt_pdf_base,         bdpt_pdf_base_f);
        glc_Uniform1i (uloc_t3.n_hits,                n_hits);
        GLC_DISPATCH_CHECKED("T3:material", (GLuint)((n + 63) / 64), 1, 1);

        /* Single full barrier + readback covers ALL of T1/T2/T3 output in one
         * GPU→CPU sync.  GL_ALL_BARRIER_BITS ensures CPU-side GetBufferSubData
         * visibility of the atomic writes from all three stages. */
        glc_MemoryBarrier(GL_ALL_BARRIER_BITS);

        /* Read full counter block for stats; n_hits was resolved before T3 dispatch. */
        uint32_t counters[8] = {};
        readback_ssbo(ssbo_counter, counters, 8 * sizeof(uint32_t));

        ps.stats[0].record_gpu(n, (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - t1_start).count(), ps.Q_intent.size());
        ps.stats[1].record_gpu(n_hits, 0, n_hits);
        ps.stats[2].record_gpu(n_hits, (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - t3_start).count(), n_hits);

        /* One-shot per-batch diagnostic (stdout so Jupyter sees it).
         * Fires on the first bounce of every batch, rate-limited to every 64th batch. */
        {
            static std::atomic<uint32_t> _diag_batch_ctr{0};
            const uint32_t bc = _diag_batch_ctr.fetch_add(1u, std::memory_order_relaxed);
            if (first_gen && (bc % 64u == 0u)) {
                printf("[gpu-bounce] batch#%u  intents=%d  n_hits=%d  nb=%d  nm=%d  "
                       "max_bdpt_v=%d  max_bdpt_s=%d  max_bdpt_p=%d  "
                       "uloc_max_verts=%d  uloc_max_spectral=%d  uloc_max_pdfs=%d\n",
                       bc, n, n_hits, nb, nm,
                       max_bdpt_v, max_bdpt_s, max_bdpt_p,
                       (int)uloc_t3.bdpt_max_verts,
                       (int)uloc_t3.bdpt_max_spectral,
                       (int)uloc_t3.bdpt_max_pdfs);
                fflush(stdout);
            }
        }

        if (n_hits <= 0) break;

        /* Merge GPU UV accumulator into the CPU-side uv_accum_cpu and re-zero
         * the GPU SSBO.  This is intentionally rate-limited: with physical
         * 512x512 pages and 11+5*bands channels, the flat accumulator can be
         * >1 GB for the default hot set.  Per-dispatch readback makes GPU mode
         * PCIe/CPU-bound; the GPU buffer is allowed to keep accumulating until
         * the next scheduled readback.                                      */
        const auto uv_now = Clock::now();
        const bool do_uv_readback = n_uv_groups > 0 && st.uv_accum_total > 0
            && std::chrono::duration<double>(uv_now - last_uv_readback_t).count() >=
               uv_readback_interval_s.load(std::memory_order_relaxed);
        if (do_uv_readback) {
            last_uv_readback_t = uv_now;
            const GLsizeiptr accum_sz = (GLsizeiptr)((size_t)st.uv_accum_total * sizeof(uint32_t));
            stg_uv_acc.resize(static_cast<size_t>(st.uv_accum_total));
            auto& gpu_acc = stg_uv_acc;
            readback_ssbo(ssbo_uv_accum, gpu_acc.data(), accum_sz);
            ps.gpu_uv_readback_bytes.fetch_add((uint64_t)accum_sz, std::memory_order_relaxed);
            ps.gpu_uv_readback_count.fetch_add(1, std::memory_order_relaxed);
            /* Atomic-add GPU counts into CPU accumulator. */
            for (int i = 0; i < st.uv_accum_total; ++i)
                uv_accum_add(st.uv_accum_cpu[i], gpu_acc[i]);
            /* GPU-direct display blit: write ssbo_uv_accum → tex_uv_pages directly
             * (no CPU round-trip) when WGL context sharing is active. */
            if (prog_uv_blit && blit_n_layers > 0)
                dispatch_uv_blit(ps);

            /* Zero the GPU accumulator for the next dispatch. */
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_uv_accum);
            std::fill(gpu_acc.begin(), gpu_acc.end(), 0u);
            glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, accum_sz, gpu_acc.data());
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        /* Read back meta[0]=intent_count  meta[1]=terminal_count */
        uint32_t t3_meta_rb[2] = {};
        readback_ssbo(ssbo_t3_meta, t3_meta_rb, 2 * sizeof(uint32_t));
        int nc      = (int)std::min(t3_meta_rb[0], (uint32_t)cap_children);
        int nt_term = (int)std::min(t3_meta_rb[1], (uint32_t)cap_children);
        static bool t3_post_diag_once = true;
        const bool t3_post_diag = t3_post_diag_once;
        if (t3_post_diag) {
            t3_post_diag_once = false;
            fprintf(stderr,
                "[gpu-T3-post] n_hits=%d child_count=%u capped_child=%d terminal_count=%u capped_terminal=%d do_field=%d\n",
                n_hits, t3_meta_rb[0], nc, t3_meta_rb[1], nt_term,
                (int)(ps.cfg.gpu_segment_field_capture && ps.st && ps.st->camera_field_grid));
            fflush(stderr);
        }

        /* Child intents stay GPU-resident in ssbo_child_int; ping-pong swap at
         * the end of the bounce loop advances to the next generation without any
         * PCIe transfer.  out_flash_children / out_sensor_children will be set
         * to zero after the loop (no children escaped to Q_intent).           */
        if (t3_post_diag) {
            fprintf(stderr, "[gpu-T3-post] child intents gpu-resident (nc=%d), no requeue\n", nc);
            fflush(stderr);
        }

        /* Accumulate emissive terminal hits onto the CPU sensor image.
         * Terminal record layout (TERMINAL_STRIDE=26+2*MAX_SPECTRAL_BANDS floats):
         *   [16]=color_flag  [23]=sensor_origin_y  [24]=sensor_origin_z
         *   [25]=is_emissive_hit  [26..26+MAX_SPECTRAL_BANDS-1]=amp_re  [26+MAX_SPECTRAL_BANDS..TERMINAL_STRIDE-1]=amp_im */
        if (nt_term > 0 && ps.sensor_res > 0) {
            /* terminals start at max_children*CHILD_STRIDE floats into the combined buffer */
            GLintptr term_off = (GLintptr)((size_t)max_children * CHILD_STRIDE * sizeof(float));
            stg_tbuf.resize((size_t)nt_term * TERMINAL_STRIDE);
            auto& tbuf = stg_tbuf;
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_child_int);
            glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER, term_off,
                                 (GLsizeiptr)((size_t)nt_term * TERMINAL_STRIDE * sizeof(float)),
                                 tbuf.data());
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            const int    res       = ps.sensor_res;
            const float  inv_w     = (float)res / (2.0f * ps.sensor_half_w);
            const float  inv_h     = (float)res / (2.0f * ps.sensor_half_h);
            /* Pre-compute sensor updates outside the lock so get_sensor_image()
             * is not blocked for the entire terminal scan. */
            auto& sensor_updates = stg_sensor_updates;
            sensor_updates.clear();
            sensor_updates.reserve(static_cast<size_t>(nt_term / 4 + 1));
            for (int ti = 0; ti < nt_term; ++ti) {
                const float* rec = tbuf.data() + (size_t)ti * TERMINAL_STRIDE;
                uint32_t is_emissive; memcpy(&is_emissive, rec + 25, 4);
                if (!is_emissive) continue;
                uint32_t cflag; memcpy(&cflag, rec + 16, 4);
                if (cflag != 1u) continue;
                const float soy = rec[23], soz = rec[24];
                const int iy = (int)((soy + ps.sensor_half_w) * inv_w);
                const int iz = (int)((soz + ps.sensor_half_h) * inv_h);
                if (iy < 0 || iy >= res || iz < 0 || iz >= res) continue;
                double cr = 0.0, cg = 0.0, cb = 0.0;
                for (int b = 0; b < nb && b < MAX_SPECTRAL_BANDS; ++b) {
                    const double re = (double)rec[26 + b], im = (double)rec[26 + MAX_SPECTRAL_BANDS + b];
                    const double amp = std::sqrt(re*re + im*im);
                    double wr, wg, wb;
                    band_to_display_rgb(b, nb, st.freq_hz_vec, wr, wg, wb);
                    cr += amp * wr; cg += amp * wg; cb += amp * wb;
                }
                sensor_updates.push_back({iy, iz, cr, cg, cb});
            }
            /* Apply all updates under a short-held lock, tracking peaks
             * incrementally so get_sensor_image() skips the O(res²) scan. */
            if (!sensor_updates.empty()) {
                std::lock_guard<std::mutex> lk(ps.sensor_mu);
                for (const auto& u : sensor_updates) {
                    const size_t px = (size_t)(u.iy * res + u.iz);
                    const double vr = (ps.sensor_accum[0*(size_t)res*res + px] += u.r);
                    const double vg = (ps.sensor_accum[1*(size_t)res*res + px] += u.g);
                    const double vb = (ps.sensor_accum[2*(size_t)res*res + px] += u.b);
                    if (vr > ps.sensor_peak[0]) ps.sensor_peak[0] = vr;
                    if (vg > ps.sensor_peak[1]) ps.sensor_peak[1] = vg;
                    if (vb > ps.sensor_peak[2]) ps.sensor_peak[2] = vb;
                }
            }
        }
        if (t3_post_diag) {
            fprintf(stderr, "[gpu-T3-post] terminal readback done\n");
            fflush(stderr);
        }
        /* ── Post-T3: STRIKE records → Q_out + field grid (one readback) ──
         * ssbo_hit still holds T2-refined positions (T3 reads it, never writes).
         * Combining both purposes into one PCIe transfer halves the readback cost
         * vs doing them separately. */
        {
            const bool do_field = ps.cfg.gpu_segment_field_capture
                                  && ps.st && ps.st->camera_field_grid;
            static constexpr int Q_OUT_VIS_CAP = 4'000'000;
            /* Skip the hit readback entirely when Q_out is already full, field
             * capture is disabled, and there are no parametric groups to count. */
            const int  q_space   = Q_OUT_VIS_CAP - (int)ps.Q_out.size();
            const bool do_stats  = ps.st && !ps.st->gid_manifold_stats.empty();
            if (q_space > 0 || do_field || do_stats) {
                /* Readback window: vis cap for STRIKE, Q_OUT_VIS_CAP for stats
                 * (bounded so stats-only mode stays within the same PCIe budget),
                 * n_hits only for field capture. */
                const int n_vis   = std::max(0, std::min(n_hits, q_space));
                const int n_stats = do_stats ? std::min(n_hits, Q_OUT_VIS_CAP) : 0;
                const int n_rb    = do_field ? n_hits : std::max(n_vis, n_stats);
                stg_hbuf.resize((size_t)n_rb * HIT_STRIDE);
                auto& hbuf = stg_hbuf;
                readback_ssbo(ssbo_hit, hbuf.data(),
                              (GLsizeiptr)((size_t)n_rb * HIT_STRIDE * sizeof(float)));
                ps.gpu_hit_readback_bytes.fetch_add(
                    (uint64_t)((size_t)n_rb * HIT_STRIDE * sizeof(float)),
                    std::memory_order_relaxed);
                ps.gpu_hit_readback_count.fetch_add(1, std::memory_order_relaxed);
                const int bands = std::min(nb, MAX_SPECTRAL_BANDS);
                const bool field_done_bulk = do_field
                    && accumulate_field_capture_segments_regular_threaded(*ps.st, hbuf.data(), n_rb, HIT_STRIDE, nb);
                /* amp_tmp only needed for per-segment field capture fallback */
                VXcd amp_tmp;
                if (do_field && !field_done_bulk) amp_tmp.resize(nb);

                stg_strike.clear();
                stg_strike.reserve(static_cast<size_t>(n_vis));
                auto& strike_batch = stg_strike;

                for (int hi = 0; hi < n_rb; ++hi) {
                    const float* row = hbuf.data() + (size_t)hi * HIT_STRIDE;

                    if (hi < n_vis) {
                        /* Build STRIKE record for the visualization drain loop */
                        RayRecord rec{};
                        rec.kind = RayRecordKind::STRIKE;
                        uint32_t tlo, thi;
                        memcpy(&tlo, row + 21, 4); memcpy(&thi, row + 22, 4);
                        rec.tag  = (uint64_t)tlo | ((uint64_t)thi << 32);
                        memcpy(&rec.src_id, row + 20, 4);
                        memcpy(&rec.bounce, row + 17, 4);
                        rec.seg_start[0] = row[9];  rec.seg_start[1] = row[10]; rec.seg_start[2] = row[11];
                        rec.pos[0]    = row[0];  rec.pos[1]    = row[1];  rec.pos[2]    = row[2];
                        rec.dir[0]    = row[6];  rec.dir[1]    = row[7];  rec.dir[2]    = row[8];
                        rec.normal[0] = row[3];  rec.normal[1] = row[4];  rec.normal[2] = row[5];
                        rec.path_len          = row[12];
                        rec.path_at_seg_start = row[13];
                        memcpy(&rec.hit_tri, row + 14, 4);
                        rec.hit_group_id = (rec.hit_tri >= 0 && ps.st &&
                                            (size_t)rec.hit_tri < ps.st->tri_param_group_of_tri.size())
                                           ? ps.st->tri_param_group_of_tri[(size_t)rec.hit_tri] : -1;
                        memcpy(&rec.mat_idx, row + 15, 4);
                        uint32_t cflag; memcpy(&cflag, row + 16, 4);
                        rec.color_flag = (uint8_t)cflag;
                        rec.n_bands = bands;
                        for (int b = 0; b < bands; ++b) {
                            rec.amp_re[b] = row[26 + b];
                            rec.amp_im[b] = row[26 + MAX_SPECTRAL_BANDS + b];
                        }
                        strike_batch.push_back(std::move(rec));
                    }

                    /* Field grid contribution using T2-refined segment endpoints */
                    if (do_field && !field_done_bulk) {
                        const V3d seg_s(row[9], row[10], row[11]);
                        const V3d hit_p(row[0], row[1],  row[2]);
                        for (int b = 0; b < bands; ++b)
                            amp_tmp[b] = std::complex<double>(row[26 + b], row[26 + MAX_SPECTRAL_BANDS + b]);
                        for (int b = bands; b < nb; ++b)
                            amp_tmp[b] = cd(0.0, 0.0);
                        accumulate_field_capture_segment(*ps.st, seg_s, hit_p, amp_tmp);
                    }

                    /* GPU-T2 per-GID outcome: bit2(4u)=teleported, bit3(8u)=absorbed.
                     * Only NEURAL_ASSEMBLY and PARAMETRIC_LENS set these bits; geometry-
                     * only refiners (POLY_BARY, SDF_SPHERE) leave them clear. */
                    if (ps.st) {
                        uint32_t _cf; std::memcpy(&_cf, row + 16, sizeof(uint32_t));
                        const bool _tp = (_cf & 4u) != 0u;
                        const bool _ab = (_cf & 8u) != 0u;
                        if (_tp || _ab) {
                            int32_t _ht; std::memcpy(&_ht, row + 14, sizeof(int32_t));
                            const int _gp = (_ht >= 0 &&
                                             (size_t)_ht < ps.st->tri_param_group_of_tri.size())
                                            ? ps.st->tri_param_group_of_tri[(size_t)_ht] : -1;
                            if (_gp >= 0 && (size_t)_gp < ps.st->gid_manifold_stats.size()) {
                                auto& _ms = ps.st->gid_manifold_stats[(size_t)_gp];
                                if (_tp) _ms.ok .fetch_add(1, std::memory_order_relaxed);
                                else     _ms.abs.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }

                if (!strike_batch.empty())
                    ps.Q_out.push_many(strike_batch);
            }
        }
        if (t3_post_diag) {
            fprintf(stderr, "[gpu-T3-post] qout/field readback done\n");
            fflush(stderr);
        }

        /* ── BDPT side-data readback: push GPU-emitted records to CPU queues ──
         * Counts in MetaBuf tail [2+nm..2+nm+2] accumulate across bounces; we
         * read only the delta (new records since prev_nv/ns/np/no) each bounce.
         * counter[4] (optical write pointer) similarly accumulates.          */
        {
            uint32_t bdpt_cnts[3] = {};
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_t3_meta);
            glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                 (GLintptr)((2 + nm) * sizeof(uint32_t)),
                                 3 * sizeof(uint32_t), bdpt_cnts);
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            const int total_nv = std::min((int)bdpt_cnts[0], max_bdpt_v);
            const int total_ns = std::min((int)bdpt_cnts[1], max_bdpt_s);
            const int total_np = std::min((int)bdpt_cnts[2], max_bdpt_p);
            const int total_no = std::min((int)counters[4], max_bdpt_o);
            const int nv = total_nv - prev_nv;  /* delta: records added this bounce */
            const int ns = total_ns - prev_ns;
            const int np = total_np - prev_np;
            const int no = total_no - prev_no;
            /* !!DIAG!! raw GPU-emitted BDPT counts (before clamping to cap).
             * bdpt_cnts[2]=raw PDF count from meta[2+nm+2].
             * If raw_pdfs=0 the shader never incremented the PDF counter →
             *   bdpt_max_pdfs uniform is 0 in shader (dead-stripped or upload failed).
             * If raw_pdfs>0 but np=0 the cap is wrong.
             * uloc_t3.bdpt_max_pdfs=-1 means the uniform was not found in the shader. */
            fprintf(stderr,
                "[gpu-bdpt-diag] nm=%d bdpt_count_base=%d "
                "raw_verts=%u raw_spectral=%u raw_pdfs=%u "
                "cap_v=%d cap_s=%d cap_p=%d "
                "uloc_max_verts=%d uloc_max_spectral=%d uloc_max_pdfs=%d "
                "uloc_count_base=%d uloc_spectral_base=%d uloc_pdf_base=%d "
                "max_bdpt_v=%d max_bdpt_s=%d max_bdpt_p=%d "
                "bdpt_spectral_base_f=%d bdpt_pdf_base_f=%d\n",
                nm, bdpt_count_base,
                bdpt_cnts[0], bdpt_cnts[1], bdpt_cnts[2],
                nv, ns, np,
                (int)uloc_t3.bdpt_max_verts,
                (int)uloc_t3.bdpt_max_spectral,
                (int)uloc_t3.bdpt_max_pdfs,
                (int)uloc_t3.bdpt_count_base,
                (int)uloc_t3.bdpt_spectral_base,
                (int)uloc_t3.bdpt_pdf_base,
                max_bdpt_v, max_bdpt_s, max_bdpt_p,
                bdpt_spectral_base_f, bdpt_pdf_base_f);
            fflush(stderr);
            uint64_t batch_cam_count = 0;  /* sensor verts in this batch */

            if (nv > 0) {
                stg_bdpt_verts.resize((size_t)nv * BDPT_VERTEX_STRIDE_F);
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_bdpt_output);
                glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                     (GLintptr)((size_t)prev_nv * BDPT_VERTEX_STRIDE_F * sizeof(float)),
                                     (GLsizeiptr)((size_t)nv * BDPT_VERTEX_STRIDE_F * sizeof(float)),
                                     stg_bdpt_verts.data());
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                uint64_t cam_count = 0;
                for (int i = 0; i < nv; ++i) {
                    const float* row = stg_bdpt_verts.data() + (size_t)i * BDPT_VERTEX_STRIDE_F;
                    BdptVertexRecord vr{};
                    memcpy(&vr.subpath_id,  row +  0, 4);
                    {
                        uint32_t pk = 0u; memcpy(&pk, row + 1, 4);
                        vr.vertex_index  = (uint16_t)(pk >> 16);
                        vr.stream        = (uint8_t)((pk >> 8) & 0xFFu);
                        vr.sample_domain = (uint8_t)(pk & 0xFFu);
                    }
                    memcpy(&vr.flags,      row +  2, 4);
                    /* row[3] = strategy_id = 0 */
                    memcpy(&vr.tri_id,     row +  4, 4);
                    vr.group_id = -1;  /* row[5] written as -1 by shader */
                    memcpy(&vr.mat_idx,    row +  6, 4);
                    vr.pos[0] = row[7];  vr.pos[1] = row[8];  vr.pos[2] = row[9];
                    vr.normal[0] = row[10]; vr.normal[1] = row[11]; vr.normal[2] = row[12];
                    vr.dir_in[0] = row[13]; vr.dir_in[1] = row[14]; vr.dir_in[2] = row[15];
                    /* dir_out = 0 (row[16..18]) */
                    vr.path_len          = row[19];
                    vr.path_at_seg_start = row[20];
                    vr.pdf_fwd           = row[21];
                    vr.pdf_rev           = row[22];
                    vr.pdf_area          = row[23];
                    vr.pdf_solid_angle   = row[24];
                    vr.throughput_scalar = row[25];
                    vr.sensor_origin_y   = row[26];
                    vr.sensor_origin_z   = row[27];
                    ps.push_bdpt_vertex(vr);
                    if (vr.stream == BDPT_SIDE_SENSOR) { ++cam_count; ++batch_cam_count; }
                }
                if (cam_count > 0)
                    ps.bdpt_cam_vertex_count.fetch_add(cam_count, std::memory_order_relaxed);
            }

            if (ns > 0) {
                stg_bdpt_spectral.resize((size_t)ns * BDPT_SPECTRAL_STRIDE_F);
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_bdpt_output);
                glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                     (GLintptr)((bdpt_spectral_base_f + (size_t)prev_ns * BDPT_SPECTRAL_STRIDE_F) * sizeof(float)),
                                     (GLsizeiptr)((size_t)ns * BDPT_SPECTRAL_STRIDE_F * sizeof(float)),
                                     stg_bdpt_spectral.data());
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                for (int i = 0; i < ns; ++i) {
                    const float* row = stg_bdpt_spectral.data() + (size_t)i * BDPT_SPECTRAL_STRIDE_F;
                    BdptSpectralWeightRecord sw{};
                    memcpy(&sw.subpath_id, row + 0, 4);
                    {
                        uint32_t vb = 0u; memcpy(&vb, row + 1, 4);
                        sw.vertex_index = (uint16_t)(vb >> 16);
                        sw.band_id      = (uint16_t)(vb & 0xFFFFu);
                    }
                    sw.beta_re               = row[2];
                    sw.beta_im               = row[3];
                    sw.wavelength_or_center  = row[4];
                    sw.band_pdf              = row[5];
                    sw.sensor_rgb_weight     = row[6];
                    ps.push_bdpt_spectral(sw);
                }
            }

            if (np > 0) {
                stg_bdpt_pdfs.resize((size_t)np * BDPT_PDF_STRIDE_F);
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_bdpt_output);
                glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                     (GLintptr)((bdpt_pdf_base_f + (size_t)prev_np * BDPT_PDF_STRIDE_F) * sizeof(float)),
                                     (GLsizeiptr)((size_t)np * BDPT_PDF_STRIDE_F * sizeof(float)),
                                     stg_bdpt_pdfs.data());
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                for (int i = 0; i < np; ++i) {
                    const float* row = stg_bdpt_pdfs.data() + (size_t)i * BDPT_PDF_STRIDE_F;
                    BdptPdfRecord pr{};
                    memcpy(&pr.subpath_id, row + 0, 4);
                    {
                        uint32_t pk = 0u; memcpy(&pk, row + 1, 4);
                        pr.vertex_index  = (uint16_t)(pk & 0xFFFFu);
                        pr.sample_domain = (uint8_t)((pk >> 16) & 0xFFu);
                        pr.measure       = (uint8_t)((pk >> 24) & 0xFFu);
                    }
                    pr.pdf_fwd         = row[2];
                    pr.pdf_rev         = row[3];
                    pr.pdf_area        = row[4];
                    pr.pdf_solid_angle = row[5];
                    pr.jacobian_det    = row[6];
                    pr.geometry_term   = row[7];
                    memcpy(&pr.flags, row + 8, 4);
                    ps.push_bdpt_pdf(pr);
                }
            }

            if (no > 0) {
                stg_bdpt_optical.resize((size_t)no * BDPT_OPTICAL_STRIDE_F);
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_bdpt_output);
                glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                     (GLintptr)((bdpt_optical_base_f + (size_t)prev_no * BDPT_OPTICAL_STRIDE_F) * sizeof(float)),
                                     (GLsizeiptr)((size_t)no * BDPT_OPTICAL_STRIDE_F * sizeof(float)),
                                     stg_bdpt_optical.data());
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                for (int i = 0; i < no; ++i) {
                    const float* row = stg_bdpt_optical.data() + (size_t)i * BDPT_OPTICAL_STRIDE_F;
                    BdptOpticalEventRecord oe{};
                    memcpy(&oe.subpath_id, row + 0, 4);
                    {
                        uint32_t pk = 0u; memcpy(&pk, row + 1, 4);
                        oe.vertex_index  = (uint16_t)(pk & 0xFFFFu);
                        oe.element_index = (uint16_t)((pk >> 16) & 0xFFFFu);
                    }
                    {
                        uint32_t pk = 0u; memcpy(&pk, row + 2, 4);
                        oe.reason = (uint8_t)(pk & 0xFFu);
                        oe.stream = (uint8_t)((pk >> 8) & 0xFFu);
                        oe.flags  = (uint16_t)((pk >> 16) & 0xFFFFu);
                    }
                    oe.pos[0] = row[3];  oe.pos[1] = row[4];  oe.pos[2] = row[5];
                    oe.normal[0] = row[6]; oe.normal[1] = row[7]; oe.normal[2] = row[8];
                    oe.dir_in[0] = row[9]; oe.dir_in[1] = row[10]; oe.dir_in[2] = row[11];
                    oe.dir_out[0] = row[12]; oe.dir_out[1] = row[13]; oe.dir_out[2] = row[14];
                    oe.cos_incident         = row[15];
                    oe.cos_transmitted      = row[16];
                    oe.eta_i                = row[17];
                    oe.eta_t                = row[18];
                    oe.fresnel_reflectance  = row[19];
                    oe.transmittance        = row[20];
                    oe.throughput_multiplier = row[21];
                    oe.opl                  = row[22];
                    oe.geom_len             = row[23];
                    oe.aperture_radius      = row[24];
                    oe.transverse_radius    = row[25];
                    oe.dist_past_aperture   = row[26];
                    oe.phase_space_jacobian = row[27];
                    ps.push_bdpt_optical(oe);
                }
            }

            /* Data from this batch is now in its queues.  T5 is NOT fired here —
             * firing per T3 generation is too early; all bounce generations must
             * complete first.  T5 fires from the timeout path once in_flight==0. */
            (void)batch_cam_count;
            /* Advance incremental BDPT read positions for the next bounce. */
            prev_nv = total_nv; prev_ns = total_ns; prev_np = total_np; prev_no = total_no;
        }

        total_n_hits += n_hits;

        /* ── Ping-pong: child intents are already GPU-resident in ssbo_child_int
         * (OutBuf head) in INTENT_STRIDE layout \u2014 identical to T1's input format.
         * Swap the SSBO handles so T1 reads them directly on the next bounce;
         * no data is moved across PCIe between generations.                    */
        if (nc <= 0) break;
        std::swap(ssbo_intent, ssbo_child_int);
        cap_intents = cap_children;  /* reflect new ssbo_intent's capacity */
        n = nc;
        first_gen = false;

        }  /* end bounce loop */

        /* Children were fully consumed GPU-resident; none escaped to Q_intent. */
        if (out_flash_children)  *out_flash_children  = 0;
        if (out_sensor_children) *out_sensor_children = 0;
        return total_n_hits;
    }

    /* ── dispatch_t4_step: one ADI-CN BPM step for a WaveArena ────────── */
    void dispatch_t4_step(RayPipelineState& ps, WaveArena& arena) {
        using Clock = std::chrono::high_resolution_clock;
        const int nx     = arena.nx;
        const int ny     = arena.nz;    /* WaveArena uses nz for the second dim */
        const int nb     = arena.n_bands;
        const int n_pix  = nx * ny * nb;

        if (n_pix > cap_wave_pix) {
            cap_wave_pix = n_pix * 2;
            GLsizeiptr psz = (GLsizeiptr)(cap_wave_pix * sizeof(float));
            ensure_ssbo(ssbo_wave_re,    psz); ensure_ssbo(ssbo_wave_im,    psz);
            ensure_ssbo(ssbo_bpm_tmp_re, psz); ensure_ssbo(ssbo_bpm_tmp_im, psz);
            /* Thomas scratch: n_bands * max(nx,ny) * MAX_WAVE_DIM (=1024) vec2 */
            int max_dim = std::max(nx, ny);
            GLsizeiptr tsz = (GLsizeiptr)((size_t)nb * (size_t)max_dim * 1024 * 2 * sizeof(float));
            ensure_ssbo(ssbo_thomas_cp, tsz); ensure_ssbo(ssbo_thomas_dp, tsz);
        }

        /* Upload field */
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_wave_re);
        glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(n_pix * sizeof(float)), arena.re_buf.data());
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_wave_im);
        glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(n_pix * sizeof(float)), arena.im_buf.data());
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        auto t4_start = Clock::now();
        glc_UseProgram(prog_t4);
        auto set_bpm_uniforms = [&](int mode) {
            glc_Uniform1i(uloc_t4.mode,    mode);
            glc_Uniform1i(uloc_t4.nx,      nx);
            glc_Uniform1i(uloc_t4.ny,      ny);
            glc_Uniform1i(uloc_t4.n_bands, nb);
            glc_Uniform1f(uloc_t4.dx,      (float)arena.dx);
            glc_Uniform1f(uloc_t4.dz,      (float)arena.dz);
            /* Upload wavelengths array */
            float wl[MAX_SPECTRAL_BANDS] = {};
            for (int b = 0; b < std::min(nb, MAX_SPECTRAL_BANDS); ++b)
                wl[b] = (float)arena.wavelengths_m[b];
            glc_Uniform1fv(uloc_t4.wavelengths, MAX_SPECTRAL_BANDS, wl);
        };

        bind_ssbo(ssbo_wave_re, 0); bind_ssbo(ssbo_wave_im, 1);
        bind_ssbo(ssbo_bpm_tmp_re, 2); bind_ssbo(ssbo_bpm_tmp_im, 3);
        bind_ssbo(ssbo_thomas_cp, 4); bind_ssbo(ssbo_thomas_dp, 5);

        /* Mode 0: carrier advance */
        set_bpm_uniforms(0);
        GLC_DISPATCH_CHECKED("T4:BPM-m0", (GLuint)n_pix, 1, 1);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        /* Mode 1: horizontal sweep */
        set_bpm_uniforms(1);
        GLC_DISPATCH_CHECKED("T4:BPM-m1", (GLuint)(nb * ny), 1, 1);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        /* Mode 2: vertical sweep */
        set_bpm_uniforms(2);
        GLC_DISPATCH_CHECKED("T4:BPM-m2", (GLuint)(nb * nx), 1, 1);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        ps.stats[3].record_gpu(1, (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - t4_start).count(), ps.Q_wave.size());

        /* Read back updated field */
        readback_ssbo(ssbo_wave_re, arena.re_buf.data(), (GLsizeiptr)(n_pix * sizeof(float)));
        readback_ssbo(ssbo_wave_im, arena.im_buf.data(), (GLsizeiptr)(n_pix * sizeof(float)));
    }

    /* ── thread_main: GPU dispatch loop ──────────────────────────────────── */
    void thread_main(RayPipelineState& ps) {
        if (!gl_compute_make_current(&ctx)) {
            fprintf(stderr, "[gpu-dispatch] thread: make_current failed — GPU thread exiting\n");
            printf("[gpu-dispatch] thread: make_current failed — GPU thread exiting\n");
            fflush(stderr); fflush(stdout);
            ps.gpu_dispatch_state.store(4, std::memory_order_release);
            /* Force both inflight counters to 0 so bdpt_update_inflight fires
             * the T5 sentinel — otherwise join_t5() will block forever because
             * rays will never be processed and the family counts never reach 0. */
            {
                int64_t f = ps.bdpt_inflight_flash.exchange(0, std::memory_order_acq_rel);
                int64_t s = ps.bdpt_inflight_sensor.exchange(0, std::memory_order_acq_rel);
                if (f != 0 || s != 0)
                    bdpt_update_inflight(&ps, -f, -s);
            }
            /* Mark Q_intent done so any blocked pop_batch_timed returns. */
            ps.Q_intent.set_done();
            return;
        }
        if (!scene_uploaded) upload_scene_data(ps);
        ps.gpu_dispatch_state.store(3, std::memory_order_release);
        fprintf(stderr, "[gpu-dispatch] thread: scene uploaded, entering dispatch loop\n");
        fflush(stderr);

        std::vector<RayIntent>  batch;
        std::vector<WaveIntent> wave_batch;

        while (true) {
            /* Timed pop — returns -1 on timeout so we can service a pending
             * T5 GPU job even when Q_intent is momentarily empty (avoids the
             * deadlock where the T5 connection thread blocks on fut.get()
             * while the GPU thread is stuck in an infinite cv_.wait).      */
            batch.clear();
            int gpu_bsz = ps.stats[0].batch_sz_gpu.load(std::memory_order_relaxed);
            int n = ps.Q_intent.pop_batch_timed(batch, gpu_bsz, /*ms=*/5);

            /* Non-blocking drain of Q_wave — never stall here; the CPU T4
             * worker is absent when use_gpu_compute=true so we opportunistically
             * pick up any wave intents that have accumulated.              */
            wave_batch.clear();
            ps.Q_wave.drain(wave_batch, gpu_bsz);

            /* Timeout: Q_intent is empty but pipeline is still running.
             * If in_flight==0, every ray (all bounce generations) has finished
             * for this dispatch cycle.  Fire T5 if both families are ready. */
            if (n < 0) {
                if (t5_job_ready.load(std::memory_order_acquire))
                    service_t5_job();
                /* T5 is now woken by bdpt_update_inflight via Q_t5_ready.
                 * Nothing to poll here. */
                continue;
            }

            /* Exit only when Q_intent is exhausted-and-done (n==0); do one
             * final wave drain before leaving so nothing is orphaned.     */
            if (n == 0) {
                /* Final wave drain */
                ps.Q_wave.drain(wave_batch, gpu_bsz);
                for (auto& wi : wave_batch) {
                    if (wi.arena_id >= 0 && wi.arena_id < (int)ps.arenas.size())
                        dispatch_t4_step(ps, ps.arenas[static_cast<size_t>(wi.arena_id)]);
                    ps.in_flight.fetch_sub(1, std::memory_order_acq_rel);
                }
                /* Service any outstanding T5 job before exiting */
                if (t5_job_ready.load(std::memory_order_acquire))
                    service_t5_job();
                break;
            }

            /* T1→T2→T3 batch */
            {
                /* Count parents by family before dispatch. */
                int64_t flash_parents = 0, sensor_parents = 0;
                for (const auto& ri : batch)
                    if (ri.color_flag == 1) ++sensor_parents; else ++flash_parents;

                int64_t flash_children = 0, sensor_children = 0;
                int n_hits = dispatch_t1_t2_t3(ps, batch,
                                               &flash_children, &sensor_children);
                /* Harvest Eigen amp allocations from the just-processed batch into
                 * the recycling pool.  dispatch_t1_t2_t3 has already read all amp
                 * data into the flat ibuf; the VXcd heap blocks are now idle until
                 * batch.clear() would free them.  Moving them into the pool lets
                 * the next child-intent build reuse them without new malloc calls. */
                {
                    const int bsz = (int)batch.size();
                    if ((int)stg_amp_recycle.size() < bsz) stg_amp_recycle.resize(bsz);
                    for (int i = 0; i < bsz; ++i)
                        stg_amp_recycle[i] = std::move(batch[i].amp);
                }
                for (int i = 0; i < n; ++i)
                    ps.in_flight.fetch_sub(1, std::memory_order_acq_rel);
                /* Net per-family inflight change: children spawned minus parents done. */
                bdpt_update_inflight(&ps,
                                     flash_children  - flash_parents,
                                     sensor_children - sensor_parents);
                (void)n_hits;
            }

            /* T4 wave step per arena */
            for (auto& wi : wave_batch) {
                if (wi.arena_id >= 0 && wi.arena_id < (int)ps.arenas.size())
                    dispatch_t4_step(ps, ps.arenas[static_cast<size_t>(wi.arena_id)]);
                ps.in_flight.fetch_sub(1, std::memory_order_acq_rel);
            }

            /* Service T5 connection job if one arrived during this batch */
            if (t5_job_ready.load(std::memory_order_acquire))
                service_t5_job();
        }
        ps.gpu_dispatch_state.store(5, std::memory_order_release);
    }

    /* ── service_t5_job: runs on the GPU thread when t5_job_ready is set ── *
     * Uploads T5 SSBOs (bindings 0-3), dispatches t5_full_connect.comp.glsl,*
     * barriers, reads back the pixel accum buffer, fulfills the promise.    */
    void service_t5_job() {
        std::unique_ptr<T5Job> job;
        {
            std::lock_guard<std::mutex> lk(t5_job_mu);
            job = std::move(t5_job_pending);
            t5_job_ready.store(false, std::memory_order_release);
        }
        if (!job) return;

        const auto& lv  = job->light_verts;
        const auto& cv  = job->cam_verts;
        const auto& sw  = job->spectral_weights;
        T5GpuParams par = job->params;  /* mutable copy — tile/batch params updated per iteration */

        const int nlv = (int)lv.size();
        const int ncv = (int)cv.size();
        const int nsw = (int)sw.size();

        /* Sensor tile side — used to cap the pixel SSBO allocation. */
        const int STILE_early = t5_sensor_tile_size ? (int)t5_sensor_tile_size : 128;
        const int npi_tile_max = 3 * STILE_early * STILE_early;

        /* ── Ensure / grow SSBOs ──────────────────────────────────────── */
        if (nlv > cap_t5_light) {
            cap_t5_light = nlv * 2;
            ensure_ssbo(ssbo_t5_light, (GLsizeiptr)(cap_t5_light * sizeof(float)));
        }
        if (ncv > cap_t5_cam) {
            cap_t5_cam = ncv * 2;
            ensure_ssbo(ssbo_t5_cam, (GLsizeiptr)(cap_t5_cam * sizeof(float)));
        }
        /* Pixel SSBO is sized to the tile, not the full image, to avoid VRAM exhaustion. */
        if (npi_tile_max > cap_t5_pix) {
            cap_t5_pix = npi_tile_max * 2;
            ensure_ssbo(ssbo_t5_pix, (GLsizeiptr)(cap_t5_pix * sizeof(uint32_t)));
        }
        if (!ssbo_t5_params) {
            ensure_ssbo(ssbo_t5_params, sizeof(T5GpuParams));
        }
        if (nsw > cap_t5_spectral) {
            cap_t5_spectral = std::max(nsw * 2, T5_MAX_GPU_BANDS * 3);
            ensure_ssbo(ssbo_t5_spectral, (GLsizeiptr)(cap_t5_spectral * sizeof(float)));
        }

        /* ── Upload helper ────────────────────────────────────────────── */
        auto upload = [&](GLuint ssbo, const void* data, GLsizeiptr bytes) {
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
            glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, data);
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        };

        /* Light verts and spectral weights are the same for every sensor tile. */
        if (nlv) upload(ssbo_t5_light,    lv.data(), (GLsizeiptr)(nlv * sizeof(float)));
        if (nsw) upload(ssbo_t5_spectral, sw.data(), (GLsizeiptr)(nsw * sizeof(float)));

        /* ── Batched 2-D dispatch with outer sensor-tile loop ─────────────── *
         * The pixel accumulator grows as 3×res² — at large resolutions this   *
         * alone exceeds GPU VRAM and causes driver crashes/TDR.  The sensor-  *
         * tile loop partitions the pixel grid into small tiles so each        *
         * dispatch operates on a compact pixel buffer (3×tw×th).              *
         *                                                                      *
         * For each tile:                                                       *
         *   - Camera subpaths whose sensor origin falls inside the tile are   *
         *     filtered and packed into a tile-local camera vert SSBO.         *
         *   - A fresh zeroed pixel accum (3×tw×th) is uploaded.               *
         *   - The existing light-vertex batch loop runs unchanged as the      *
         *     inner loop (TDR guard: ≤4096 light verts per dispatch).         *
         *   - The tile pixel result is read back and scattered into the full  *
         *     res×res output buffer.                                           *
         *                                                                      *
         * X axis: camera vert tiles  (TILE_C verts per WG column).            *
         * Y axis: light vert tiles within the current batch (TILE_L per WG).  */
        const uint32_t T5_LIGHT_BATCH = t5_light_batch_size ? t5_light_batch_size : 4096u;
        const uint32_t T5_CAM_BATCH   = t5_cam_batch_size   ? t5_cam_batch_size   : 8192u;
        const uint32_t n_light        = par.n_light_verts;
        const uint32_t n_batches      = (n_light + T5_LIGHT_BATCH - 1u) / T5_LIGHT_BATCH;

        const int res       = par.sensor_res;
        const int STILE     = STILE_early;
        const int n_tiles_y = (res + STILE - 1) / STILE;
        const int n_tiles_x = (res + STILE - 1) / STILE;

        glc_UseProgram(prog_t5);
        bind_ssbo(ssbo_t5_light,    0);
        bind_ssbo(ssbo_t5_cam,      1);
        bind_ssbo(ssbo_t5_pix,      2);
        bind_ssbo(ssbo_t5_params,   3);
        bind_ssbo(ssbo_t5_spectral, 4);
        bind_ssbo(ssbo_bvh,         5);
        bind_ssbo(ssbo_tri_id,      6);
        bind_ssbo(ssbo_tri_full,    7);

        fprintf(stderr, "[T5-gpu] starting: n_cam=%u n_light=%u lbatches=%u cbatch_sz=%u res=%d tiles=%dx%d\n",
                par.n_cam_verts, n_light, n_batches, T5_CAM_BATCH, res, n_tiles_x, n_tiles_y);
        fflush(stderr);

        /* Full-resolution result assembled tile by tile. */
        std::vector<uint32_t> result(3u * static_cast<size_t>(res) * res, 0u);
        bool gpu_ok = true;

        /* Pre-compute sensor-to-pixel scale factors (mirrors shader logic). */
        const float inv_wy = (float)res / (2.0f * par.sensor_half_w);
        const float inv_hz = (float)res / (2.0f * par.sensor_half_h);
        const int   cv_stride = (int)T5_CGV_STRIDE;  /* floats per cam vert */

        const int n_tiles_total = n_tiles_y * n_tiles_x;
        int tiles_done = 0;
        const auto t5_wall_start = std::chrono::steady_clock::now();

        for (int ty = 0; ty < n_tiles_y && gpu_ok; ++ty) {
            for (int tx = 0; tx < n_tiles_x && gpu_ok; ++tx) {
                const int tile_y0 = ty * STILE;
                const int tile_x0 = tx * STILE;
                const int tile_h  = std::min(STILE, res - tile_y0);
                const int tile_w  = std::min(STILE, res - tile_x0);

                /* ── Filter camera subpaths whose sensor origin is in this tile ──
                 * Walk the flat cam_verts array subpath by subpath.  A subpath
                 * starts whenever ci (vinfo bits[0..14]) is 0.  All verts in the
                 * subpath share the same sensor_origin_y/z (stored at [13][14]). */
                std::vector<float> tile_cam;
                {
                    const int n_cv_total = ncv / cv_stride;
                    int v = 0;
                    while (v < n_cv_total) {
                        /* Find the end of this subpath: next vert with ci == 0. */
                        int sp_end = v + 1;
                        while (sp_end < n_cv_total) {
                            const float* pe = cv.data() + sp_end * cv_stride;
                            uint32_t vinf;
                            std::memcpy(&vinf, &pe[9], sizeof(uint32_t));
                            if ((vinf & 0x7FFFu) == 0u) break;
                            ++sp_end;
                        }
                        /* Map first vert's sensor origin to a pixel. */
                        const float* p0  = cv.data() + v * cv_stride;
                        const float  soy = p0[13];
                        const float  soz = p0[14];
                        const int    iy  = (int)((soy + par.sensor_half_w) * inv_wy);
                        const int    iz  = (int)((soz + par.sensor_half_h) * inv_hz);
                        if (iy >= tile_y0 && iy < tile_y0 + tile_h &&
                            iz >= tile_x0 && iz < tile_x0 + tile_w) {
                            /* ci values inside the subpath are already 0,1,2,…
                             * so they remain correct when the subpath is appended
                             * contiguously to tile_cam. */
                            const float* src = cv.data() + v * cv_stride;
                            tile_cam.insert(tile_cam.end(), src,
                                            src + (sp_end - v) * cv_stride);
                        }
                        v = sp_end;
                    }
                }

                const uint32_t n_tile_cam = (uint32_t)(tile_cam.size() / cv_stride);
                if (n_tile_cam == 0) {
                    fprintf(stderr, "[T5-gpu tile(%d,%d) %d/%d] skip — 0 cam verts\n",
                            tx, ty, tiles_done + 1, n_tiles_total);
                    fflush(stderr);
                    ++tiles_done;
                    continue;
                }

                const auto tile_t0 = std::chrono::steady_clock::now();

                /* ── Grow SSBOs for this tile if needed ── */
                const int ncv_tile = (int)tile_cam.size();
                if (ncv_tile > cap_t5_cam) {
                    cap_t5_cam = ncv_tile * 2;
                    ensure_ssbo(ssbo_t5_cam, (GLsizeiptr)(cap_t5_cam * sizeof(float)));
                }
                const size_t n_tile_pix = (size_t)tile_w * tile_h;
                const int    npi_tile   = (int)(3u * n_tile_pix);
                if (npi_tile > cap_t5_pix) {
                    cap_t5_pix = npi_tile * 2;
                    ensure_ssbo(ssbo_t5_pix, (GLsizeiptr)(cap_t5_pix * sizeof(uint32_t)));
                }

                /* ── Upload tile camera verts and zero pixel accum ── */
                upload(ssbo_t5_cam, tile_cam.data(),
                       (GLsizeiptr)(tile_cam.size() * sizeof(float)));
                {
                    std::vector<uint32_t> tile_pix_zero(3u * n_tile_pix, 0u);
                    upload(ssbo_t5_pix, tile_pix_zero.data(),
                           (GLsizeiptr)(3u * n_tile_pix * sizeof(uint32_t)));
                }

                /* ── Update params for this tile ── */
                par.n_light_verts = n_light;
                par.tile_y0     = tile_y0;
                par.tile_x0     = tile_x0;
                par.tile_h      = tile_h;
                par.tile_w      = tile_w;

                const uint32_t n_cbatches = (n_tile_cam + T5_CAM_BATCH - 1u) / T5_CAM_BATCH;

                fprintf(stderr, "[T5-gpu tile(%d,%d) %d/%d] cam=%u lbatches=%u cbatches=%u\n",
                        tx, ty, tiles_done + 1, n_tiles_total, n_tile_cam, n_batches, n_cbatches);
                fflush(stderr);

                /* ── Outer camera-batch loop (TDR guard for X-dispatch) ── */
                for (uint32_t c = 0; c < n_cbatches && gpu_ok; ++c) {
                    const uint32_t cam_batch_actual = std::min(T5_CAM_BATCH,
                                                               n_tile_cam - c * T5_CAM_BATCH);
                    par.cam_offset  = c * T5_CAM_BATCH;
                    par.n_cam_verts = par.cam_offset + cam_batch_actual; /* upper bound */

                    const GLuint wg_x = (cam_batch_actual + (uint32_t)T5_TILE_C - 1u)
                                      / (uint32_t)T5_TILE_C;

                    /* ── Inner light-batch loop (TDR guard for Y-dispatch) ── */
                    for (uint32_t b = 0; b < n_batches && gpu_ok; ++b) {
                        par.light_offset     = b * T5_LIGHT_BATCH;
                        par.light_batch_size = std::min(T5_LIGHT_BATCH,
                                                        n_light - par.light_offset);
                        upload(ssbo_t5_params, &par, sizeof(T5GpuParams));
                        const GLuint wg_y = (par.light_batch_size + (uint32_t)T5_TILE_L - 1u)
                                          / (uint32_t)T5_TILE_L;
                        glc_DispatchCompute(wg_x, wg_y, 1u);
                        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                        glFlush();
                        GLenum gl_err = gpu_check_dispatch("T5:connect", wg_x, wg_y, 1u);
                        if (gl_err != GL_NO_ERROR) {
                            fprintf(stderr,
                                    "[T5-gpu] GL error 0x%04x after cbatch %u/%u lbatch %u/%u tile(%d,%d) — TDR? aborting\n",
                                    (unsigned)gl_err, c, n_cbatches, b, n_batches, tx, ty);
                            fflush(stderr);
                            gpu_ok = false;
                            break;
                        }
                        /* Heartbeat every 8 lbatches so a crash can be located. */
                        if ((b & 7u) == 7u) {
                            const auto now = std::chrono::steady_clock::now();
                            const float elapsed = std::chrono::duration<float>(
                                    now - tile_t0).count();
                            fprintf(stderr, "[T5-gpu tile(%d,%d)] cbatch %u/%u lbatch %u/%u  %.1fs\n",
                                    tx, ty, c + 1u, n_cbatches, b + 1u, n_batches, elapsed);
                            fflush(stderr);
                        }
                    }
                }
                if (!gpu_ok) break;

                /* ── Readback tile pixel accum and scatter into full result ── */
                std::vector<uint32_t> tile_pix(3u * n_tile_pix);
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_t5_pix);
                glc_GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                                     (GLsizeiptr)(3u * n_tile_pix * sizeof(uint32_t)),
                                     tile_pix.data());
                glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

                const size_t full_pix = static_cast<size_t>(res) * res;
                for (int row = 0; row < tile_h; ++row) {
                    for (int col = 0; col < tile_w; ++col) {
                        const size_t src = static_cast<size_t>(row) * tile_w + col;
                        const size_t dst = static_cast<size_t>(tile_y0 + row) * res
                                         + (tile_x0 + col);
                        result[dst]                = tile_pix[src];
                        result[dst + full_pix]     = tile_pix[src + n_tile_pix];
                        result[dst + 2u * full_pix]= tile_pix[src + 2u * n_tile_pix];
                    }
                }

                ++tiles_done;
                {
                    const float tile_s = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - tile_t0).count();
                    const float wall_s = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t5_wall_start).count();
                    fprintf(stderr,
                            "[T5-gpu tile(%d,%d) %d/%d] done %.1fs  wall %.1fs\n",
                            tx, ty, tiles_done, n_tiles_total, tile_s, wall_s);
                    fflush(stderr);
                }
            }
        }

        if (gpu_ok)
            fprintf(stderr, "[T5-gpu] done: %d/%d tiles  res=%d  wall %.1fs\n",
                    tiles_done, n_tiles_total, res,
                    std::chrono::duration<float>(
                        std::chrono::steady_clock::now() - t5_wall_start).count());
        else
            fprintf(stderr, "[T5-gpu] aborted after GPU error — returning empty result\n");
        fflush(stderr);

        job->promise.set_value(std::move(result));
    }

    /* ── submit_t5_connect: called from BDPT connection thread ─────────── *
     * Enqueues a T5 GPU job, blocks until the GPU thread fulfills it, and  *
     * returns the pixel accum buffer (3×res² uint32 bit-cast from float).  */
    std::vector<uint32_t> submit_t5_connect(
        std::vector<float>    light_verts,
        std::vector<float>    cam_verts,
        std::vector<uint32_t> pixel_accum,
        std::vector<float>    spectral_weights,
        T5GpuParams           params)
    {
        if (!prog_t5) return {};
        auto job = std::make_unique<T5Job>();
        job->light_verts      = std::move(light_verts);
        job->cam_verts        = std::move(cam_verts);
        job->pixel_accum      = std::move(pixel_accum);
        job->spectral_weights = std::move(spectral_weights);
        job->params           = params;
        std::future<std::vector<uint32_t>> fut = job->promise.get_future();
        {
            std::lock_guard<std::mutex> lk(t5_job_mu);
            t5_job_pending = std::move(job);
            t5_job_ready.store(true, std::memory_order_release);
        }
        t5_job_cv.notify_one();
        return fut.get();
    }
};

/* ── Wave arena helpers ──────────────────────────────────────────────────────────────── */

static void wave_arena_build(
    WaveArena&               arena,
    int                      id,
    const RtScaleContext&    ctx,
    const RayPipelineConfig& cfg,
    const RayTracerState&    st)
{
    arena.id     = id;
    arena.center = V3d(ctx.center[0], ctx.center[1], ctx.center[2]);
    arena.radius = ctx.radius;
    arena.n_real = (ctx.n_real > 0.0) ? ctx.n_real : 1.0;
    arena.dz     = (ctx.dt_m > 0.0) ? ctx.dt_m : cfg.wave_grid_dx_m * 0.5;
    arena.nz     = (ctx.n_substeps > 0) ? ctx.n_substeps
                 : std::max(4, (int)std::ceil(2.0 * ctx.radius / arena.dz));
    arena.dx     = cfg.wave_grid_dx_m;
    arena.nx = arena.ny = std::min(256, std::max(16,
        (int)std::ceil(2.0 * ctx.radius / cfg.wave_grid_dx_m)));

    /* Propagation axis: payload[3..5] if available, else world +z. */
    arena.axis_z = V3d(0.0, 0.0, 1.0);
    if (ctx.payload && ctx.payload_size_bytes >= (int)(6 * sizeof(double))) {
        const double* p = static_cast<const double*>(ctx.payload);
        V3d cand(p[3], p[4], p[5]);
        if (cand.norm() > 0.5) arena.axis_z = cand.normalized();
    }
    V3d up = (std::abs(arena.axis_z.dot(V3d(0,1,0))) < 0.9)
           ? V3d(0,1,0) : V3d(1,0,0);
    arena.axis_x = arena.axis_z.cross(up).normalized();
    arena.axis_y = arena.axis_z.cross(arena.axis_x).normalized();

    arena.n_bands = std::min(st.n_bands, 32);
    for (int b = 0; b < arena.n_bands; ++b) {
        double freq = (b < (int)st.freq_hz_vec.size()) ? st.freq_hz_vec[b] : 1e3;
        arena.wavelengths_m[b] = (st.speed_m_s / freq) / arena.n_real;
    }
    size_t npix = static_cast<size_t>(arena.n_bands) * arena.ny * arena.nx;
    arena.re_buf.assign(npix, 0.0f);
    arena.im_buf.assign(npix, 0.0f);
}

static V3d wave_arena_to_local(const WaveArena& a, const V3d& wp)
{
    V3d off = wp - a.center;
    return V3d(off.dot(a.axis_x), off.dot(a.axis_y), off.dot(a.axis_z));
}

/* Seed the entry plane from a ray: Gaussian envelope centred at entry point. */
static void wave_arena_seed(WaveArena& arena, const RayIntent& ray)
{
    arena.re_buf.assign(arena.re_buf.size(), 0.0f);
    arena.im_buf.assign(arena.im_buf.size(), 0.0f);

    V3d lpos  = wave_arena_to_local(arena, ray.pos);
    double lx = lpos.x(), ly = lpos.y();
    double ox  = (arena.nx * 0.5) * arena.dx;
    double oy  = (arena.ny * 0.5) * arena.dx;
    const double sigma = 3.0 * arena.dx;

    for (int iy = 0; iy < arena.ny; ++iy) {
        for (int ix = 0; ix < arena.nx; ++ix) {
            double xm = ix * arena.dx - ox;
            double ym = iy * arena.dx - oy;
            double r2 = (xm - lx)*(xm - lx) + (ym - ly)*(ym - ly);
            float  env = static_cast<float>(std::exp(-r2 / (2.0 * sigma * sigma)));
            if (env < 1e-9f) continue;
            size_t xy_off = static_cast<size_t>(iy) * arena.nx + ix;
            for (int b = 0; b < arena.n_bands && b < (int)ray.amp.size(); ++b) {
                cd a = ray.amp[b] * static_cast<double>(env);
                size_t idx = static_cast<size_t>(b) * arena.ny * arena.nx + xy_off;
                arena.re_buf[idx] = static_cast<float>(a.real());
                arena.im_buf[idx] = static_cast<float>(a.imag());
            }
        }
    }
}

/* March BPM from entry to exit plane (nz ADI-CN steps). */
static void wave_arena_march(WaveArena& arena)
{
    for (int step = 0; step < arena.nz; ++step)
        ray_tracer_wave_bpm_step(
            arena.n_bands, arena.nx, arena.ny,
            arena.dx, arena.dz,
            arena.wavelengths_m,
            arena.re_buf.data(), arena.im_buf.data());
}

/* Extract exit ray: power-weighted centroid for position, phase-gradient
 * for direction, field value at centroid for per-band amplitude. */
static ChildRay wave_arena_extract(const WaveArena& arena, const RayIntent& src)
{
    const int    nb   = arena.n_bands;
    const int    nx   = arena.nx, ny = arena.ny;
    const size_t npix = static_cast<size_t>(nx * ny);

    double sum_pow = 0.0, cx = 0.0, cy = 0.0;
    for (int b = 0; b < nb; ++b) {
        const float* re = arena.re_buf.data() + static_cast<size_t>(b) * npix;
        const float* im = arena.im_buf.data() + static_cast<size_t>(b) * npix;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                size_t i = static_cast<size_t>(iy * nx + ix);
                double p = (double)re[i]*re[i] + (double)im[i]*im[i];
                sum_pow += p;
                cx += p * ix;
                cy += p * iy;
            }
        }
    }
    int ic_x = (sum_pow > 0.0) ? (int)std::round(cx / sum_pow) : nx / 2;
    int ic_y = (sum_pow > 0.0) ? (int)std::round(cy / sum_pow) : ny / 2;
    ic_x = std::max(1, std::min(nx - 2, ic_x));
    ic_y = std::max(1, std::min(ny - 2, ic_y));

    /* Phase-gradient estimate for exit direction */
    double kx_sum = 0.0, ky_sum = 0.0;
    int    k_count = 0;
    for (int b = 0; b < nb; ++b) {
        const float* re = arena.re_buf.data() + static_cast<size_t>(b) * npix;
        const float* im = arena.im_buf.data() + static_cast<size_t>(b) * npix;
        auto at = [&](int ix, int iy) -> cd {
            return cd(re[iy*nx+ix], im[iy*nx+ix]);
        };
        cd Ux_fwd = at(ic_x+1, ic_y), Ux_bwd = at(ic_x-1, ic_y);
        cd Uy_fwd = at(ic_x, ic_y+1), Uy_bwd = at(ic_x, ic_y-1);
        if (std::abs(Ux_fwd) > 1e-30 && std::abs(Ux_bwd) > 1e-30) {
            kx_sum += std::arg(Ux_fwd * std::conj(Ux_bwd)) / (2.0 * arena.dx);
            ky_sum += std::arg(Uy_fwd * std::conj(Uy_bwd)) / (2.0 * arena.dx);
            ++k_count;
        }
    }
    double kx = (k_count > 0) ? kx_sum / k_count : 0.0;
    double ky = (k_count > 0) ? ky_sum / k_count : 0.0;
    double k0  = (nb > 0 && arena.wavelengths_m[0] > 0.0)
               ? (TWO_PI / arena.wavelengths_m[0]) : 1.0;
    double kz2 = k0*k0 - kx*kx - ky*ky;
    double kz  = (kz2 > 0.0) ? std::sqrt(kz2) : k0;
    V3d exit_dir = (arena.axis_x * (kx/k0)
                  + arena.axis_y * (ky/k0)
                  + arena.axis_z * (kz/k0)).normalized();

    double ox = (nx * 0.5) * arena.dx;
    double oy = (ny * 0.5) * arena.dx;
    V3d exit_pos = arena.center
                 + arena.axis_z * arena.radius
                 + arena.axis_x * (ic_x * arena.dx - ox)
                 + arena.axis_y * (ic_y * arena.dx - oy)
                 + exit_dir * (EPS * 200.0);

    VXcd exit_amp(nb);
    for (int b = 0; b < nb; ++b) {
        const float* re = arena.re_buf.data() + static_cast<size_t>(b) * npix;
        const float* im = arena.im_buf.data() + static_cast<size_t>(b) * npix;
        exit_amp[b] = cd(re[ic_y*nx+ic_x], im[ic_y*nx+ic_x]);
    }

    ChildRay cr;
    cr.intent              = src;
    cr.intent.pos          = exit_pos;
    cr.intent.dir          = exit_dir;
    cr.intent.amp          = exit_amp;
    cr.intent.path_len    += 2.0 * arena.radius;
    cr.intent.bounce      += 1;
    cr.intent.bounces_left = std::max(0, src.bounces_left - 1);
    if (cr.intent.bdpt_vertex < 0xFFFFu) cr.intent.bdpt_vertex += 1u;
    return cr;
}

/* ── In-flight accounting ─────────────────────────────────────────────────── */

/* Update BDPT family counters on the CPU path — mirrors the GPU path's
 * bdpt_update_inflight() but operates one intent at a time.  color_flag==1
 * is sensor (camera), everything else is flash (light). */
static inline void bdpt_family_spawn(RayPipelineState& ps, uint8_t color_flag)
{
    if (color_flag == 1)
        ps.bdpt_inflight_sensor.fetch_add(1, std::memory_order_release);
    else
        ps.bdpt_inflight_flash.fetch_add(1, std::memory_order_release);
}

static void bdpt_family_finish(RayPipelineState& ps, uint8_t color_flag)
{
    int64_t nf, ns;
    if (color_flag == 1) {
        nf = ps.bdpt_inflight_flash.load(std::memory_order_acquire);
        ns = ps.bdpt_inflight_sensor.fetch_add(-1, std::memory_order_acq_rel) - 1;
    } else {
        nf = ps.bdpt_inflight_flash.fetch_add(-1, std::memory_order_acq_rel) - 1;
        ns = ps.bdpt_inflight_sensor.load(std::memory_order_acquire);
    }
    /* Mirror sentinel logic from bdpt_update_inflight — but only trigger when
     * both truly zero, and only from the CPU T3 path (GPU already does this). */
    if (nf != 0 || ns != 0) return;
    const uint32_t flash_sub = ps.bdpt_flash_dispatched.load(std::memory_order_acquire);
    const uint32_t sens_sub  = ps.bdpt_sensor_dispatched.load(std::memory_order_acquire);
    const uint32_t t5_fired  = ps.bdpt_t5_fired.load(std::memory_order_acquire);
    if (std::min(flash_sub, sens_sub) <= t5_fired) return;
    if (ps.bdpt_connection_running.load(std::memory_order_acquire)) return;
    ps.Q_t5_ready.push(1);
}

static void pipeline_finish_ray(RayPipelineState& ps, uint8_t color_flag = 255)
{
    ps.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    if (color_flag != 255)
        bdpt_family_finish(ps, color_flag);
}

static void pipeline_spawn_child(RayPipelineState& ps, RayIntent child)
{
    const uint8_t cf = child.color_flag;
    ++ps.in_flight;
    bdpt_family_spawn(ps, cf);
    ps.Q_intent.push(std::move(child));
}

/* ── T1: intersector ──────────────────────────────────────────────────────── */

static void pipeline_intersector(RayPipelineState& ps)
{
    using Clock = std::chrono::high_resolution_clock;
    RayTracerState& st      = *ps.st;
    const bool      has_bvh = !st.bvh_nodes.empty();
    std::vector<RayIntent> batch;
    std::mt19937 t1_rng(static_cast<uint32_t>(ps.cfg.seed) ^ 0xDEADBEEFu);

    while (true) {
        batch.clear();
        int bsz = ps.stats[0].batch_sz.load(std::memory_order_relaxed);
        float shuf = ps.cfg.intent_queue_shuffle;
        int n;
        if (shuf > 0.0f) {
            n = ps.Q_intent.pop_batch_shuffled(batch, bsz, shuf, t1_rng);
        } else {
            n = ps.Q_intent.pop_batch(batch, bsz);
        }
        if (n == 0) break;

        auto t0 = Clock::now();

        for (auto& intent : batch) {
            const V3d pos0 = intent.pos;
            const V3d dir  = intent.dir;

            double t_hit   = 1e18;
            int    hit_tri = -1;
            if (has_bvh) {
                V3d inv = dir.cwiseInverse();
                bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                          pos0, dir, inv, t_hit, hit_tri);
            } else {
                for (size_t ti = 0; ti < st.tris.size(); ++ti) {
                    double tt;
                    if (ray_triangle_hit(pos0, dir, st.tris[ti], tt) && tt > T_SELF && tt < t_hit) {
                        t_hit = tt;  hit_tri = (int)ti;
                    }
                }
            }

            if (hit_tri < 0) {
                RayRecord rec;
                rec.kind      = RayRecordKind::MISS;
                rec.tag       = intent.tag;
                rec.src_id    = intent.src_id;
                rec.bounce    = intent.bounce;
                rec.pos[0]    = static_cast<float>(pos0.x());
                rec.pos[1]    = static_cast<float>(pos0.y());
                rec.pos[2]    = static_cast<float>(pos0.z());
                rec.dir[0]    = static_cast<float>(dir.x());
                rec.dir[1]    = static_cast<float>(dir.y());
                rec.dir[2]    = static_cast<float>(dir.z());
                rec.path_len  = static_cast<float>(intent.path_len);
                rec.color_flag = intent.color_flag;
                ps.Q_out.push(std::move(rec));
                pipeline_finish_ray(ps);
                continue;
            }

            const V3d hit_pos    = pos0 + t_hit * dir;
            const double tot_len = intent.path_len + t_hit;
            VXcd amp_prop        = intent.amp;
            const int nb         = std::min((int)amp_prop.size(), st.n_bands);

            for (int b = 0; b < nb; ++b) {
                double n_re = 1.0, n_im = 0.0;
                if (intent.medium_mat_idx >= 0) {
                    n_re = mat_n_real(st, intent.medium_mat_idx, b);
                    n_im = mat_n_imag(st, intent.medium_mat_idx, b);
                    if (n_re < 1.0) n_re = 1.0;
                }
                double k_med  = st.k_real[b] * n_re;
                double alpha  = st.atmo_abs[b];
                if (intent.medium_mat_idx >= 0)
                    alpha += TWO_PI * st.freq_hz_vec[b] * n_im / st.speed_m_s;
                double atten  = std::exp(-alpha * t_hit);
                /* Backward (sensor-cast) rays are importance-sampling paths, not
                 * physical power carriers — skip 1/r² spherical spread so they
                 * are not culled by the amplitude threshold before reaching the
                 * scene emitters on the far side of the lens stack. */
                double spread = (intent.color_flag == 1) ? 1.0 : 1.0 / (1.0 + tot_len);
                amp_prop[b] *= std::polar(atten * spread, -k_med * t_hit);
            }

            if (st.camera_field_grid)
                accumulate_field_capture_segment(st, pos0, hit_pos, amp_prop);

            int wave_id = -1;
            for (int ai = 0; ai < (int)ps.arenas.size(); ++ai) {
                if ((hit_pos - ps.arenas[static_cast<size_t>(ai)].center).norm()
                        <= ps.arenas[static_cast<size_t>(ai)].radius) {
                    wave_id = ai;
                    break;
                }
            }

            const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
            HitRecord hr;
            hr.ray               = intent;
            hr.ray.amp           = amp_prop;
            hr.ray.path_len      = tot_len;
            hr.seg_start         = pos0;
            hr.incoming_dir      = dir;
            hr.path_at_seg_start = intent.path_len;
            hr.hit_tri           = hit_tri;
            hr.hit_tri_flags     = tri.flags;
            hr.hit_pos           = hit_pos;
            hr.hit_n             = tri.normal;
            if (dir.dot(hr.hit_n) > 0.0) hr.hit_n = -hr.hit_n;
            hr.t_hit             = t_hit;
            hr.amp_propagated    = amp_prop;

            /* Barycentric coordinates via 2×2 least-squares on the edge basis.
             * Cheap: 6 dot products, 1 divide — exact for the MT parametric. */
            const V3d bq  = hit_pos - tri.v0;
            const double bd11 = tri.edge1.dot(tri.edge1);
            const double bd12 = tri.edge1.dot(tri.edge2);
            const double bd22 = tri.edge2.dot(tri.edge2);
            const double bd1q = tri.edge1.dot(bq);
            const double bd2q = tri.edge2.dot(bq);
            const double bden = bd11 * bd22 - bd12 * bd12 + 1e-30;
            const float  bu   = static_cast<float>((bd22 * bd1q - bd12 * bd2q) / bden);
            const float  bv   = static_cast<float>((bd11 * bd2q - bd12 * bd1q) / bden);

            {
                RayRecord rec;
                rec.kind              = RayRecordKind::STRIKE;
                rec.tag               = intent.tag;
                rec.src_id            = intent.src_id;
                rec.bounce            = intent.bounce;
                rec.seg_start[0]      = static_cast<float>(pos0.x());
                rec.seg_start[1]      = static_cast<float>(pos0.y());
                rec.seg_start[2]      = static_cast<float>(pos0.z());
                rec.pos[0]            = static_cast<float>(hit_pos.x());
                rec.pos[1]            = static_cast<float>(hit_pos.y());
                rec.pos[2]            = static_cast<float>(hit_pos.z());
                rec.dir[0]            = static_cast<float>(dir.x());
                rec.dir[1]            = static_cast<float>(dir.y());
                rec.dir[2]            = static_cast<float>(dir.z());
                rec.normal[0]         = static_cast<float>(hr.hit_n.x());
                rec.normal[1]         = static_cast<float>(hr.hit_n.y());
                rec.normal[2]         = static_cast<float>(hr.hit_n.z());
                rec.path_len          = static_cast<float>(tot_len);
                rec.path_at_seg_start = static_cast<float>(intent.path_len);
                rec.hit_tri           = hit_tri;
                rec.hit_group_id      = (hit_tri >= 0 && (size_t)hit_tri < st.tri_param_group_of_tri.size())
                                        ? st.tri_param_group_of_tri[(size_t)hit_tri] : -1;
                rec.mat_idx           = tri.mat_idx;
                rec.bary_u            = bu;
                rec.bary_v            = bv;
                rec.color_flag        = intent.color_flag;
                const int rnb = std::min(nb, RAY_RECORD_MAX_BANDS);
                rec.n_bands = rnb;
                for (int b = 0; b < rnb; ++b) {
                    rec.amp_re[b] = static_cast<float>(amp_prop[b].real());
                    rec.amp_im[b] = static_cast<float>(amp_prop[b].imag());
                }
                ps.Q_out.push(std::move(rec));
            }

            /* ── BDPT vertex record (T1) ─────────────────────────────────
             * Only emitted for rays stamped with a subpath id at launch.
             * dir_out and PDFs are unknown here; they arrive via the
             * BdptPdfRecord emitted by T3 at the same (subpath_id, vertex). */
            if (intent.bdpt_subpath_id != 0u) {
                const int grp = (hit_tri >= 0 && (size_t)hit_tri < st.tri_param_group_of_tri.size())
                                ? st.tri_param_group_of_tri[(size_t)hit_tri] : -1;
                BdptVertexRecord vr{};
                vr.subpath_id        = intent.bdpt_subpath_id;
                vr.vertex_index      = intent.bdpt_vertex;
                vr.stream            = intent.bdpt_stream;
                vr.sample_domain     = BDPT_DOMAIN_UNKNOWN;
                vr.flags             = tri.flags;
                vr.strategy_id       = intent.bdpt_strategy;
                vr.tri_id            = hit_tri;
                vr.group_id          = grp;
                vr.mat_idx           = tri.mat_idx;
                vr.pos[0]            = static_cast<float>(hit_pos.x());
                vr.pos[1]            = static_cast<float>(hit_pos.y());
                vr.pos[2]            = static_cast<float>(hit_pos.z());
                vr.normal[0]         = static_cast<float>(hr.hit_n.x());
                vr.normal[1]         = static_cast<float>(hr.hit_n.y());
                vr.normal[2]         = static_cast<float>(hr.hit_n.z());
                vr.dir_in[0]         = static_cast<float>(dir.x());
                vr.dir_in[1]         = static_cast<float>(dir.y());
                vr.dir_in[2]         = static_cast<float>(dir.z());
                vr.path_len          = static_cast<float>(tot_len);
                vr.path_at_seg_start = static_cast<float>(intent.path_len);
                vr.sensor_origin_y   = intent.sensor_origin_y;
                vr.sensor_origin_z   = intent.sensor_origin_z;
                ps.push_bdpt_vertex(vr);
                if (intent.bdpt_stream == BDPT_SIDE_SENSOR)
                    ps.bdpt_cam_vertex_count.fetch_add(1, std::memory_order_relaxed);
            }

            /* ── Sensor image accumulator (T1) ───────────────────────────
             * Forward ray (color_flag==0) hitting near the sensor plate →
             * ch0 (irradiance view).  Backward/sensor rays are NOT accumulated
             * here — their correct contribution arrives only when they terminate
             * on an emissive surface (T3 path).  Counting backward hits at the
             * plate from reflections would produce spurious sensor image
             * saturation before any emitter is found. */
            if (ps.sensor_res > 0 && intent.color_flag == 0) {
                const float hx = static_cast<float>(hit_pos.x());
                const float hy = static_cast<float>(hit_pos.y());
                const float hz = static_cast<float>(hit_pos.z());
                if (std::abs(hx - ps.sensor_px) < 0.004f) {
                    const float inv_w = static_cast<float>(ps.sensor_res) / (2.0f * ps.sensor_half_w);
                    const float inv_h = static_cast<float>(ps.sensor_res) / (2.0f * ps.sensor_half_h);
                    const int   iy    = static_cast<int>((hy + ps.sensor_half_w) * inv_w);
                    const int   iz    = static_cast<int>((hz + ps.sensor_half_h) * inv_h);
                    const int   res   = ps.sensor_res;
                    if (iy >= 0 && iy < res && iz >= 0 && iz < res) {
                        double cr = 0.0, cg = 0.0, cb = 0.0;
                        for (int b = 0; b < nb; ++b) {
                            const double re = amp_prop[b].real(), im = amp_prop[b].imag();
                            const double amp = std::sqrt(re*re + im*im);
                            double wr, wg, wb;
                            band_to_display_rgb(b, nb, st.freq_hz_vec, wr, wg, wb);
                            cr += amp * wr; cg += amp * wg; cb += amp * wb;
                        }
                        const size_t px0 = static_cast<size_t>(iy * res + iz);
                        std::lock_guard<std::mutex> lk(ps.sensor_mu);
                        const double nr = (ps.sensor_accum[0*(size_t)res*res + px0] += cr);
                        const double ng = (ps.sensor_accum[1*(size_t)res*res + px0] += cg);
                        const double nb_ = (ps.sensor_accum[2*(size_t)res*res + px0] += cb);
                        if (nr  > ps.sensor_peak[0]) ps.sensor_peak[0] = nr;
                        if (ng  > ps.sensor_peak[1]) ps.sensor_peak[1] = ng;
                        if (nb_ > ps.sensor_peak[2]) ps.sensor_peak[2] = nb_;
                    }
                }
            }

            if (wave_id >= 0) {
                WaveIntent wi;
                wi.ray      = intent;
                wi.ray.amp  = amp_prop;
                wi.ray.path_len = tot_len;
                wi.ray.interaction_flags |= (1u << SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ);
                wi.arena_id = wave_id;
                ps.Q_wave.push(std::move(wi));
            } else {
                ps.Q_hit.push(std::move(hr));
            }
        }

        auto t1 = Clock::now();
        ps.stats[0].record(n,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
            ps.Q_intent.size());
    }
}

/* ── T2: refiner ──────────────────────────────────────────────────────────── */

static void pipeline_refiner(RayPipelineState& ps)
{
    using Clock = std::chrono::high_resolution_clock;
    RayTracerState& st = *ps.st;
    std::vector<HitRecord> batch;

    while (true) {
        batch.clear();
        int bsz = ps.stats[1].batch_sz.load(std::memory_order_relaxed);
        int n   = ps.Q_hit.pop_batch(batch, bsz);
        if (n == 0) break;
        auto t0 = Clock::now();

        std::vector<RefinedHit> refined_batch;
        refined_batch.reserve(static_cast<size_t>(n));
        for (auto& hr : batch) {
            RefinedHit rh;
            rh.base        = hr;
            rh.refined_pos = hr.hit_pos;
            rh.refined_n   = hr.hit_n;
            rh.was_parametric = apply_parametric_surface_point(
                st, hr.hit_tri, hr.hit_pos, rh.refined_pos, rh.refined_n);
            if (hr.incoming_dir.dot(rh.refined_n) > 0.0)
                rh.refined_n = -rh.refined_n;

            /* ── Manifold teleport: parametric lens only
             * Neural MLP and LUT transports are intentionally disabled here.
             * They need a reversible optical contract (Jacobian, eta/cosine,
             * probability measure) before they can re-enter the camera path.
             * Magic number in the payload selects the evaluator:
             *   14949 → parametric exact conic (apply_parametric_lens_from_f32)
             *   14948 / 14946 / 14947 / 14950 / 14951 → deferred absorber
             * Empty payload (out_bytes==0) → interior absorber: terminate ray. */
            {
                int nb_mfld = 0;
                const float* mpp = tri_manifold_payload(st, hr.hit_tri, &nb_mfld);
                if (mpp) {
                    if (nb_mfld < (int)(8 * sizeof(float))) {
                        /* Interior absorber (no payload or stub) — terminate. */
                        pipeline_finish_ray(ps);
                        continue;
                    }
                    V3d   tpos = rh.refined_pos;
                    V3d   tdir = hr.incoming_dir;
                    VXcd  tamp = hr.amp_propagated;
                    double tpl = hr.ray.path_len;
                    /* Save pre-teleport direction and amplitude sum for the
                     * optical event record computed after the lens transfer. */
                    const V3d    dir_before  = tdir;
                    const double amp_mag_pre = [&]() {
                        double s = 0.0;
                        for (int _b = 0; _b < (int)tamp.size(); ++_b)
                            s += std::abs(tamp[_b]);
                        return s;
                    }();
                    bool absorbed = false;
                    ParametricLensTraceInfo lens_trace{};
                    bool lens_trace_valid = false;
                    const float magic = mpp[0];
                    if (magic == 14948.0f) {
                        /* Deferred MLP transport: absorb rather than fabricate
                         * an optical transform without reversible metadata. */
                        absorbed = true;
                    } else if (magic == 14949.0f) {
                        absorbed = apply_parametric_lens_from_f32(
                            st, mpp, nb_mfld, tpos, tdir, tamp, &tpl,
                            hr.ray.bdpt_stream == BDPT_SIDE_SENSOR, &lens_trace);
                        lens_trace_valid = true;
                    } else if (magic == 14946.0f || magic == 14947.0f ||
                               magic == 14950.0f || magic == 14951.0f) {
                        /* Deferred LUT transport: absorb rather than fabricate
                         * an optical transform without reversible metadata. */
                        absorbed = true;
                    } else {
                        absorbed = true;  /* unknown magic → absorb */
                    }
                    const int _gp = (hr.hit_tri >= 0 &&
                                     (size_t)hr.hit_tri < st.tri_param_group_of_tri.size())
                                    ? st.tri_param_group_of_tri[(size_t)hr.hit_tri] : -1;
                    {
                        if (_gp >= 0 && (size_t)_gp < st.gid_manifold_stats.size()) {
                            auto& _ms = st.gid_manifold_stats[(size_t)_gp];
                            if (absorbed) _ms.abs.fetch_add(1, std::memory_order_relaxed);
                            else          _ms.ok .fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    /* Emit optical event records for BDPT-stamped rays.
                     * Parametric lenses report every refractive interface/clip
                     * they traverse; opaque/deferred absorbers get a single
                     * absorption record so connection can invalidate the edge. */
                    if (hr.ray.bdpt_subpath_id != 0u) {
                        double amp_mag_post = 0.0;
                        for (int _b = 0; _b < (int)tamp.size(); ++_b)
                            amp_mag_post += std::abs(tamp[_b]);
                        const float tp_mul = (amp_mag_pre > 1e-30)
                            ? static_cast<float>(amp_mag_post / amp_mag_pre) : 0.0f;
                        lens_trace.throughput_multiplier = tp_mul;
                        if (lens_trace_valid && !lens_trace.events.empty()) {
                            for (auto oer : lens_trace.events) {
                                oer.subpath_id = hr.ray.bdpt_subpath_id;
                                oer.vertex_index = hr.ray.bdpt_vertex;
                                oer.stream = hr.ray.bdpt_stream;
                                if (oer.throughput_multiplier == 0.0f && !absorbed)
                                    oer.throughput_multiplier = tp_mul;
                                ps.push_bdpt_optical(oer);
                            }
                        } else {
                            BdptOpticalEventRecord oer{};
                            oer.subpath_id           = hr.ray.bdpt_subpath_id;
                            oer.vertex_index         = hr.ray.bdpt_vertex;
                            oer.element_index        = static_cast<uint16_t>(_gp >= 0 ? _gp : 0);
                            oer.reason               = BDPT_OPT_ABSORPTION;
                            oer.stream               = hr.ray.bdpt_stream;
                            oer.pos[0]               = static_cast<float>(rh.refined_pos.x());
                            oer.pos[1]               = static_cast<float>(rh.refined_pos.y());
                            oer.pos[2]               = static_cast<float>(rh.refined_pos.z());
                            oer.dir_in[0]            = static_cast<float>(dir_before.x());
                            oer.dir_in[1]            = static_cast<float>(dir_before.y());
                            oer.dir_in[2]            = static_cast<float>(dir_before.z());
                            oer.throughput_multiplier = 0.0f;
                            ps.push_bdpt_optical(oer);
                        }
                    }
                    if (absorbed) {
                        pipeline_finish_ray(ps);
                        continue;
                    }
                    RayIntent ri      = hr.ray;
                    ri.pos            = tpos;
                    ri.dir            = tdir;
                    ri.amp            = std::move(tamp);
                    ri.path_len       = tpl;
                    ri.medium_mat_idx = -1;
                    ps.Q_intent.push(std::move(ri));
                    continue;  /* skip refined_batch; don't touch in_flight */
                }
            }

            refined_batch.push_back(std::move(rh));
        }
        ps.Q_refined.push_many(refined_batch);

        auto t1 = Clock::now();
        ps.stats[1].record(n,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
            ps.Q_hit.size());
    }
}

/* ── T3: material handler ─────────────────────────────────────────────────── */

static void pipeline_material(RayPipelineState& ps, uint64_t rng_seed)
{
    using Clock = std::chrono::high_resolution_clock;
    RayTracerState& st = *ps.st;
    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    /* Build a TERMINAL RayRecord from a HitRecord and push to Q_out. */
    auto push_terminal = [&](const HitRecord& h) {
        RayRecord rec;
        rec.kind              = RayRecordKind::TERMINAL;
        rec.tag               = h.ray.tag;
        rec.src_id            = h.ray.src_id;
        rec.bounce            = h.ray.bounce;
        rec.seg_start[0]      = static_cast<float>(h.seg_start.x());
        rec.seg_start[1]      = static_cast<float>(h.seg_start.y());
        rec.seg_start[2]      = static_cast<float>(h.seg_start.z());
        rec.pos[0]            = static_cast<float>(h.hit_pos.x());
        rec.pos[1]            = static_cast<float>(h.hit_pos.y());
        rec.pos[2]            = static_cast<float>(h.hit_pos.z());
        rec.dir[0]            = static_cast<float>(h.incoming_dir.x());
        rec.dir[1]            = static_cast<float>(h.incoming_dir.y());
        rec.dir[2]            = static_cast<float>(h.incoming_dir.z());
        rec.normal[0]         = static_cast<float>(h.hit_n.x());
        rec.normal[1]         = static_cast<float>(h.hit_n.y());
        rec.normal[2]         = static_cast<float>(h.hit_n.z());
        rec.path_len          = static_cast<float>(h.ray.path_len);
        rec.path_at_seg_start = static_cast<float>(h.path_at_seg_start);
        rec.hit_tri           = h.hit_tri;
        rec.mat_idx           = (h.hit_tri >= 0)
            ? st.tris[static_cast<size_t>(h.hit_tri)].mat_idx : -1;
        rec.color_flag        = h.ray.color_flag;
        const int nb = std::min((int)h.amp_propagated.size(), RAY_RECORD_MAX_BANDS);
        rec.n_bands = nb;
        for (int b = 0; b < nb; ++b) {
            rec.amp_re[b] = static_cast<float>(h.amp_propagated[b].real());
            rec.amp_im[b] = static_cast<float>(h.amp_propagated[b].imag());
        }
        ps.Q_out.push(std::move(rec));
    };

    std::vector<RefinedHit> batch;

    while (true) {
        batch.clear();
        int bsz = ps.stats[2].batch_sz.load(std::memory_order_relaxed);
        int n   = ps.Q_refined.pop_batch(batch, bsz);
        if (n == 0) break;
        auto t0 = Clock::now();

        for (auto& rh : batch) {
            const HitRecord& hr     = rh.base;
            const Triangle&  tri    = st.tris[static_cast<size_t>(hr.hit_tri)];
        const uint8_t    ray_cf  = hr.ray.color_flag;   /* cached for family accounting */
        const int        nb     = std::min((int)hr.amp_propagated.size(), st.n_bands);
        const V3d&       hit_n  = rh.refined_n;
        const V3d&       in_dir = hr.incoming_dir;
        VXcd             amp    = hr.amp_propagated;

        /* ── UV integrator image splat (CPU T3 parity with ray_material.comp) ── */
        if (!st.tri_uv_group_of_tri.empty()
                && hr.hit_tri >= 0
                && hr.hit_tri < (int)st.tri_uv_group_of_tri.size()) {
            const int uv_gid = st.tri_uv_group_of_tri[(size_t)hr.hit_tri];
            if (uv_gid >= 0 && uv_gid < (int)st.group_uv_res.size()) {
                const int res    = st.group_uv_res[(size_t)uv_gid];
                const int offset = st.group_uv_accum_offset[(size_t)uv_gid];
                const int n2     = res * res;
                const int n_slot = (UV_N_HDR_CHANNELS + 5 * st.n_bands) * n2;
                if (res > 0 && offset >= 0
                    && offset + n_slot <= (int)st.uv_accum_cpu.size()) {
                    const V3d dp = rh.refined_pos - tri.v0;
                    const double e1e1 = tri.edge1.dot(tri.edge1);
                    const double e1e2 = tri.edge1.dot(tri.edge2);
                    const double e2e2 = tri.edge2.dot(tri.edge2);
                    const double de1  = dp.dot(tri.edge1);
                    const double de2  = dp.dot(tri.edge2);
                    const double det  = e1e1 * e2e2 - e1e2 * e1e2;
                    double bu = 0.0, bv = 0.0;
                    if (det > 1.0e-20) {
                        bu = (e2e2 * de1 - e1e2 * de2) / det;
                        bv = (e1e1 * de2 - e1e2 * de1) / det;
                    }
                    bu = std::max(0.0, std::min(1.0, bu));
                    bv = std::max(0.0, std::min(1.0 - bu, bv));
                    const float* uvd = st.tri_uv_data.data() + (size_t)hr.hit_tri * 6;
                    const double u = uvd[0] + bu * (uvd[2] - uvd[0]) + bv * (uvd[4] - uvd[0]);
                    const double v = uvd[1] + bu * (uvd[3] - uvd[1]) + bv * (uvd[5] - uvd[1]);
                    const int ix = std::max(0, std::min(res - 1, (int)(u * res)));
                    const int iy = std::max(0, std::min(res - 1, (int)(v * res)));
                    const int texel = iy * res + ix;
                    auto& ac = st.uv_accum_cpu;
                    uv_accum_add(ac[(size_t)(offset + UV_CH_HIT_COUNT * n2 + texel)], 1u);
                    reinterpret_cast<std::atomic<uint32_t>&>(
                        ac[(size_t)(offset + UV_CH_SRC_FLAGS * n2 + texel)])
                        .fetch_or(1u << std::min((int)hr.ray.src_id, 31), std::memory_order_relaxed);
                    uv_accum_add(ac[(size_t)(offset + (UV_CH_BOUNCE_0 + std::min((int)hr.ray.bounce, 3)) * n2 + texel)], 1u);
                    const uint32_t tag_lo = (uint32_t)(hr.ray.tag & 0xFFFFFFFFull);
                    const uint32_t tag_hi = (uint32_t)((hr.ray.tag >> 32) & 0xFFFFFFFFull);
                    reinterpret_cast<std::atomic<uint32_t>&>(ac[(size_t)(offset + UV_CH_TAG_LO * n2 + texel)])
                        .fetch_or(tag_lo, std::memory_order_relaxed);
                    reinterpret_cast<std::atomic<uint32_t>&>(ac[(size_t)(offset + UV_CH_TAG_HI * n2 + texel)])
                        .fetch_or(tag_hi, std::memory_order_relaxed);
                    auto add_signed = [&](int ch, double val) {
                        const int32_t fp = (int32_t)std::max(-1073741824.0,
                            std::min(1073741824.0, val * 32768.0));
                        uv_accum_add(ac[(size_t)(offset + ch * n2 + texel)], (uint32_t)fp);
                    };
                    add_signed(UV_CH_NORMAL_X, hit_n.x());
                    add_signed(UV_CH_NORMAL_Y, hit_n.y());
                    add_signed(UV_CH_NORMAL_Z, hit_n.z());
                    const int split_base = UV_N_HDR_CHANNELS
                        + ((hr.ray.color_flag == 1) ? 4 : 3) * st.n_bands;
                    for (int b = 0; b < nb; ++b) {
                        const std::complex<double> a = amp[b];
                        const double mag = std::abs(a);
                        const uint32_t fp_mag = (uint32_t)std::min(
                            mag * 65536.0, (double)0xFFFFFFFFu);
                        uv_accum_add(ac[(size_t)(offset + (UV_N_HDR_CHANNELS + b) * n2 + texel)], fp_mag);
                        add_signed(UV_N_HDR_CHANNELS + st.n_bands + b,     a.real());
                        add_signed(UV_N_HDR_CHANNELS + 2 * st.n_bands + b, a.imag());
                        uv_accum_add(ac[(size_t)(offset + (split_base + b) * n2 + texel)], fp_mag);
                    }
                }
            }
        }

        /* ── Terminal: aperture stop ── */
        if (tri.flags & MAT_FLAG_APERTURE_STOP) {
            push_terminal(hr);
            pipeline_finish_ray(ps, ray_cf);
            continue;
        }
        /* ── Terminal: emissive ── */
        if (tri.flags & MAT_FLAG_EMISSIVE) {
            HitRecord eh = hr;
            eh.is_emissive_hit = true;
            push_terminal(eh);
            pipeline_finish_ray(ps, ray_cf);
            /* ── Sensor accumulator (T3): backward ray found emissive surface.
             * Use sensor_origin_y/z (set at submit time, propagated through all
             * children) so this works regardless of how many refractions the ray
             * passed through on its way from the sensor to the source. */
            if (ps.sensor_res > 0 && hr.ray.color_flag == 1) {
                const float sy    = hr.ray.sensor_origin_y;
                const float sz    = hr.ray.sensor_origin_z;
                const int   res   = ps.sensor_res;
                const float inv_w = static_cast<float>(res) / (2.0f * ps.sensor_half_w);
                const float inv_h = static_cast<float>(res) / (2.0f * ps.sensor_half_h);
                const int   iy    = static_cast<int>((sy + ps.sensor_half_w) * inv_w);
                const int   iz    = static_cast<int>((sz + ps.sensor_half_h) * inv_h);
                if (iy >= 0 && iy < res && iz >= 0 && iz < res) {
                    /* Physical amplitude carried by this ray at the emissive. */
                    double amp_mag = 0.0;
                    for (int b = 0; b < nb; ++b) {
                        const double re = amp[b].real(), im = amp[b].imag();
                        amp_mag += std::sqrt(re*re + im*im);
                    }
                    /* ch1 = photon-count: each successful sensor→emissive connection
                     * contributes exactly 1.0 regardless of optical attenuation.
                     * This is the single-photon-detector model: every photon that
                     * arrives at an emissive surface is counted once.  Amplitude
                     * variations from Fresnel/Beer are recorded separately in ch2.
                     *
                     * ch2 = physically weighted (amp_mag) for the radiance view. */
                    const int idx1 = 1 * res * res + iy * res + iz;
                    const int idx2 = 2 * res * res + iy * res + iz;
                    std::lock_guard<std::mutex> lk(ps.sensor_mu);
                    const double v1 = (ps.sensor_accum[static_cast<size_t>(idx1)] += 1.0);
                    const double v2 = (ps.sensor_accum[static_cast<size_t>(idx2)] += amp_mag);
                    if (v1 > ps.sensor_peak[1]) ps.sensor_peak[1] = v1;
                    if (v2 > ps.sensor_peak[2]) ps.sensor_peak[2] = v2;
                }
            }
            continue;
        }
        /* ── Budget exhausted ── */
        if (hr.ray.bounces_left <= 0) {
            push_terminal(hr);
            pipeline_finish_ray(ps, ray_cf);
            continue;
        }

        /* Helper: build a child RayIntent from the current hit. */
        auto make_child = [&](const V3d& new_dir, const VXcd& new_amp,
                              int new_medium = -2) -> RayIntent {
            static constexpr double RAY_ORIGIN_EPS = 2.0e-4;
            RayIntent ri      = hr.ray;
            ri.pos            = rh.refined_pos + RAY_ORIGIN_EPS * new_dir.normalized();
            ri.dir            = new_dir;
            ri.amp            = new_amp;
            ri.path_len       = hr.ray.path_len;
            ri.bounce        += 1;
            ri.bounces_left   = hr.ray.bounces_left - 1;
            if (new_medium != -2) ri.medium_mat_idx = new_medium;
            /* Advance BDPT vertex index for the child so all side-data records
             * emitted by T1/T2/T3 can be correlated back to (subpath, vertex). */
            if (ri.bdpt_vertex < 0xFFFFu) ri.bdpt_vertex += 1u;
            return ri;
        };

        /* BDPT scatter record emitter.
         * Called once per spawned child — one BdptPdfRecord and nb
         * BdptSpectralWeightRecords are pushed to their respective side queues.
         * pdf_fwd : forward sampling PDF in the given domain.
         * domain  : one of BDPT_DOMAIN_* constants.
         * delta   : 1 if this is a specular/delta lobe, 0 otherwise. */
        auto emit_scatter = [&](const V3d& dir_out, const VXcd& child_amp,
                                uint8_t domain, float pdf_fwd, float pdf_rev,
                                uint32_t extra_flags = 0u) {
            if (hr.ray.bdpt_subpath_id == 0u) return;
            /* Geometry term at this vertex. */
            const double cos_in  = std::max(0.0, -in_dir.dot(hit_n));
            const double cos_out_v = std::max(0.0, dir_out.dot(hit_n));
            const float  geom   = static_cast<float>(cos_in * cos_out_v);
            /* Scalar throughput at this vertex (ratio of amplitude magnitudes). */
            double amp_in_mag = 0.0, amp_out_mag = 0.0;
            for (int _b = 0; _b < nb; ++_b) {
                amp_in_mag  += std::abs(amp[_b]);
                amp_out_mag += std::abs(child_amp[_b]);
            }
            const float tp = (amp_in_mag > 1e-30)
                ? static_cast<float>(amp_out_mag / amp_in_mag) : 0.0f;
            BdptPdfRecord pr{};
            pr.subpath_id    = hr.ray.bdpt_subpath_id;
            pr.vertex_index  = hr.ray.bdpt_vertex;
            pr.sample_domain = domain;
            pr.measure       = domain;   /* domain and measure align for scatter */
            pr.pdf_fwd       = pdf_fwd;
            pr.pdf_rev       = pdf_rev;
            pr.pdf_solid_angle = pdf_fwd;
            pr.jacobian_det  = 1.0f;
            pr.geometry_term = geom;
            pr.flags         = tri.flags | extra_flags;
            ps.push_bdpt_pdf(pr);
            /* Per-band spectral weights. */
            for (int _b = 0; _b < nb; ++_b) {
                BdptSpectralWeightRecord sw{};
                sw.subpath_id         = hr.ray.bdpt_subpath_id;
                sw.vertex_index       = hr.ray.bdpt_vertex;
                sw.band_id            = static_cast<uint16_t>(_b);
                sw.beta_re            = static_cast<float>(child_amp[_b].real());
                sw.beta_im            = static_cast<float>(child_amp[_b].imag());
                sw.wavelength_or_center = (_b < (int)st.freq_hz_vec.size())
                    ? static_cast<float>(st.freq_hz_vec[static_cast<size_t>(_b)]) : 0.0f;
                sw.band_pdf           = (nb > 0) ? 1.0f / static_cast<float>(nb) : 1.0f;
                sw.sensor_rgb_weight  = tp;
                ps.push_bdpt_spectral(sw);
            }
        };

        const bool mat_transmissive = tri_material_is_transmissive(st, tri);
        const bool front_face       = (in_dir.dot(tri.normal) < 0.0);

        if (mat_transmissive) {
            int medium_from = -1, medium_to = -1;
            const bool has_pair = tri_boundary_media(tri, front_face,
                                                     medium_from, medium_to);
            const double n1 = has_pair ? medium_n_real(st, medium_from)
                : ((hr.ray.medium_mat_idx >= 0)
                   ? mat_n_real(st, hr.ray.medium_mat_idx) : 1.0);
            const double n2 = has_pair ? medium_n_real(st, medium_to)
                : (front_face ? mat_n_real(st, tri.mat_idx) : 1.0);
            const double cos_i = std::max(0.0, -in_dir.dot(hit_n));
            V3d   refracted;
            const bool can_refract = snell_refract(in_dir, hit_n, n1, n2, refracted);

            /* ── BSSRDF forward: accumulate illumination (color_flag==0) ─────── */
            if (hr.ray.color_flag == 0 && !st.tri_illum_accum.empty()) {
                const float d_frac_il = mat_cache_diffusion(st.mat_cache, tri.mat_idx);
                if (d_frac_il > 0.0f) {
                    const int nb_il  = st.tri_illum_nb;
                    const int stride = 2 * nb_il + 2;
                    const int base   = hr.hit_tri * stride;
                    if (nb_il > 0 && base >= 0
                        && base + stride <= (int)st.tri_illum_accum.size()) {
                        auto add_fp_il = [&](int off, double val) {
                            const int32_t fp = (int32_t)std::max(-1073741824.0,
                                std::min(1073741824.0, val * 32768.0));
                            uv_accum_add(st.tri_illum_accum[(size_t)(base + off)],
                                         (uint32_t)fp);
                        };
                        for (int b = 0; b < std::min(nb, nb_il); ++b) {
                            add_fp_il(2 * b,     amp[b].real());
                            add_fp_il(2 * b + 1, amp[b].imag());
                        }
                        add_fp_il(2 * nb_il, cos_i);
                        uv_accum_add(st.tri_illum_accum[(size_t)(base + 2 * nb_il + 1)],
                                     1u);
                    }
                }
            }

            /* ── BSSRDF backward: analytical illumination claim (color_flag==1) ─
             * Query forward-path accumulator and inject the diffuse contribution
             * directly into the sensor pixel accumulator.  The backward amplitude
             * is scaled by (1 − diffuse_frac) so the coherent refracted children
             * spawned below carry only the remaining coherent energy. */
            if (hr.ray.color_flag == 1 && !st.tri_illum_accum.empty()) {
                const float d_frac_bk = mat_cache_diffusion(st.mat_cache, tri.mat_idx);
                if (d_frac_bk > 0.0f) {
                    const int nb_il  = st.tri_illum_nb;
                    const int stride = 2 * nb_il + 2;
                    const int base   = hr.hit_tri * stride;
                    if (nb_il > 0 && base >= 0
                        && base + stride <= (int)st.tri_illum_accum.size()) {
                        const int count = static_cast<int>(
                            st.tri_illum_accum[(size_t)(base + stride - 1)]);
                        if (count > 0) {
                            double T_bk = 0.0;
                            if (can_refract) {
                                const double sin2_bk  = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
                                const double cos_t_bk = std::sqrt(std::max(0.0, 1.0 - sin2_bk));
                                T_bk = 1.0 - fresnel_R(cos_i, cos_t_bk, n1, n2);
                            }
                            if (T_bk > 1e-9) {
                                const double cos_sum_bk = static_cast<double>(
                                    static_cast<int32_t>(
                                        st.tri_illum_accum[(size_t)(base + 2*nb_il)]))
                                    / 32768.0;
                                const double avg_cos_bk = cos_sum_bk / static_cast<double>(count);
                                const double coupling   = T_bk * static_cast<double>(d_frac_bk)
                                                          * avg_cos_bk / M_PI;
                                const double inv_n_bk   = 1.0 / static_cast<double>(count);
                                double contrib_mag = 0.0;
                                for (int b = 0; b < std::min(nb, nb_il); ++b) {
                                    const double re_il = static_cast<double>(
                                        static_cast<int32_t>(
                                            st.tri_illum_accum[(size_t)(base + 2*b)]))
                                        / 32768.0;
                                    const double im_il = static_cast<double>(
                                        static_cast<int32_t>(
                                            st.tri_illum_accum[(size_t)(base + 2*b+1)]))
                                        / 32768.0;
                                    const cd illum_avg_bk(re_il * inv_n_bk, im_il * inv_n_bk);
                                    contrib_mag += std::abs(amp[b] * coupling * illum_avg_bk);
                                }
                                if (contrib_mag > 0.0 && ps.sensor_res > 0) {
                                    const float sy    = hr.ray.sensor_origin_y;
                                    const float sz    = hr.ray.sensor_origin_z;
                                    const int   res_s = ps.sensor_res;
                                    const float inv_w = static_cast<float>(res_s)
                                                        / (2.0f * ps.sensor_half_w);
                                    const float inv_h = static_cast<float>(res_s)
                                                        / (2.0f * ps.sensor_half_h);
                                    const int   iy = static_cast<int>(
                                        (sy + ps.sensor_half_w) * inv_w);
                                    const int   iz = static_cast<int>(
                                        (sz + ps.sensor_half_h) * inv_h);
                                    if (iy >= 0 && iy < res_s && iz >= 0 && iz < res_s) {
                                        const int idx2 = 2 * res_s * res_s + iy * res_s + iz;
                                        std::lock_guard<std::mutex> lk(ps.sensor_mu);
                                        const double nv = (ps.sensor_accum[static_cast<size_t>(idx2)]
                                            += contrib_mag);
                                        if (nv > ps.sensor_peak[2]) ps.sensor_peak[2] = nv;
                                    }
                                }
                                /* Scale down to coherent fraction. */
                                const double scale_bk = 1.0 - static_cast<double>(d_frac_bk);
                                for (int b = 0; b < nb; ++b) amp[b] *= scale_bk;
                            }
                        }
                    }
                }
            }

            if (!can_refract) {
                /* TIR: pure reflection (delta lobe). */
                V3d  rd = (in_dir - 2.0 * in_dir.dot(hit_n) * hit_n).normalized();
                VXcd ra = amp;
                for (int b = 0; b < nb; ++b)
                    ra[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                emit_scatter(rd, ra, BDPT_DOMAIN_SOLID_ANGLE, 1.0f, 1.0f,
                             BDPT_PDF_FLAG_DELTA_SPECULAR);
                pipeline_spawn_child(ps, make_child(rd, ra));
            } else {
                const double sin2_t = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
                const double cos_t  = std::sqrt(std::max(0.0, 1.0 - sin2_t));
                const double R      = fresnel_R(cos_i, cos_t, n1, n2);
                const double R_rev  = fresnel_R(cos_t, cos_i, n2, n1);
                const int    new_med = has_pair ? medium_to
                    : ((hr.ray.medium_mat_idx == tri.mat_idx) ? -1 : tri.mat_idx);

                if (ps.cfg.max_children >= 2) {
                    /* Deterministic split: both lobes launched with √R/√T weights. */
                    {
                        V3d  rd = (in_dir - 2.0*in_dir.dot(hit_n)*hit_n).normalized();
                        VXcd ra = amp;
                        double rs = std::sqrt(R);
                        for (int b = 0; b < nb; ++b)
                            ra[b] *= rs * mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                        emit_scatter(rd, ra, BDPT_DOMAIN_SOLID_ANGLE,
                                     static_cast<float>(R), static_cast<float>(R),
                                     BDPT_PDF_FLAG_DELTA_SPECULAR | BDPT_PDF_FLAG_SPLIT);
                        pipeline_spawn_child(ps, make_child(rd, ra));
                    }
                    {
                        VXcd ta = amp;
                        double ts = std::sqrt(1.0 - R);
                        for (int b = 0; b < nb; ++b)
                            ta[b] *= ts;
                        emit_scatter(refracted, ta, BDPT_DOMAIN_SOLID_ANGLE,
                                     static_cast<float>(1.0 - R),
                                     static_cast<float>(1.0 - R_rev),
                                     BDPT_PDF_FLAG_DELTA_SPECULAR | BDPT_PDF_FLAG_SPLIT);
                        pipeline_spawn_child(ps, make_child(refracted, ta, new_med));
                    }
                } else {
                    /* Russian roulette: one lobe sampled with probability R. */
                    if (U01(rng) < R) {
                        V3d  rd = (in_dir - 2.0*in_dir.dot(hit_n)*hit_n).normalized();
                        VXcd ra = amp;
                        for (int b = 0; b < nb; ++b)
                            ra[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                        emit_scatter(rd, ra, BDPT_DOMAIN_SOLID_ANGLE,
                                     static_cast<float>(R), static_cast<float>(R),
                                     BDPT_PDF_FLAG_DELTA_SPECULAR);
                        pipeline_spawn_child(ps, make_child(rd, ra));
                    } else {
                        VXcd ta = amp;
                        emit_scatter(refracted, ta, BDPT_DOMAIN_SOLID_ANGLE,
                                     static_cast<float>(1.0 - R),
                                     static_cast<float>(1.0 - R_rev),
                                     BDPT_PDF_FLAG_DELTA_SPECULAR);
                        pipeline_spawn_child(ps, make_child(refracted, ta, new_med));
                    }
                }
            }
        } else {
            /* ── Epsilon fast-path: skip direction sampling for fully absorptive materials ── */
            if (!ps.mat_epsilon_flags.empty() &&
                static_cast<size_t>(tri.mat_idx) < ps.mat_epsilon_flags.size() &&
                ps.mat_epsilon_flags[static_cast<size_t>(tri.mat_idx)]) {
                push_terminal(hr);
                pipeline_finish_ray(ps, ray_cf);
                continue;
            }

            /* Opaque: diffuse, GGX microfacet, or mathematically-degenerate mirror. */
            V3d new_dir;
            bool chose_diffuse = false;
            bool chose_ggx = false;
            const double diffuse_p = std::max(0.0, std::min(1.0,
                static_cast<double>(mat_cache_diffusion(st.mat_cache, tri.mat_idx))));
            const double spec_p = std::max(0.0, 1.0 - diffuse_p);
            const double ggx_alpha = static_cast<double>(
                surf_cache_ggx_alpha(st.surf_cache, tri.mat_idx));
            if (U01(rng) < diffuse_p) {
                new_dir = cosine_hemisphere(hit_n, rng);
                chose_diffuse = true;
            } else if (spec_p > 0.0 && ggx_alpha > 1.0e-3) {
                new_dir = ggx_sample_reflection(hit_n, in_dir, ggx_alpha,
                                                U01(rng), U01(rng));
                chose_ggx = (new_dir.dot(hit_n) > 0.0);
            } else {
                new_dir = (in_dir - 2.0 * in_dir.dot(hit_n) * hit_n).normalized();
            }
            if (new_dir.dot(hit_n) <= 0.0) {
                push_terminal(hr);
                pipeline_finish_ray(ps, ray_cf);
                continue;
            }
            VXcd na = amp;
            for (int b = 0; b < nb; ++b)
                na[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
            if (tri.flags & MAT_FLAG_REACTIVE)
                apply_reactive_shift_cached(na, st.mat_cache, tri.mat_idx);

            double max_abs = 0.0;
            for (int b = 0; b < nb; ++b)
                max_abs = std::max(max_abs, std::abs(na[b]));
            if (max_abs >= hr.ray.min_amplitude) {
                /* PDF: cosine hemisphere, GGX microfacet reflection, or delta mirror. */
                const double cos_in_pdf = std::max(0.0, -in_dir.dot(hit_n));
                float scatter_pdf = 0.0f;
                float scatter_pdf_rev = 0.0f;
                uint32_t scatter_flags = 0u;
                if (chose_diffuse) {
                    scatter_pdf = static_cast<float>(
                        diffuse_p * std::max(0.0, new_dir.dot(hit_n)) / M_PI);
                    scatter_pdf_rev = static_cast<float>(diffuse_p * cos_in_pdf / M_PI);
                    scatter_flags = BDPT_PDF_FLAG_DIFFUSE;
                } else if (chose_ggx) {
                    scatter_pdf = static_cast<float>(
                        spec_p * ggx_pdf_solid_angle(hit_n, in_dir, new_dir, ggx_alpha));
                    scatter_pdf_rev = static_cast<float>(
                        spec_p * ggx_pdf_solid_angle(hit_n, -new_dir, -in_dir, ggx_alpha));
                    scatter_flags = BDPT_PDF_FLAG_GGX;
                } else {
                    scatter_pdf = static_cast<float>(spec_p);
                    scatter_pdf_rev = static_cast<float>(spec_p);
                    scatter_flags = BDPT_PDF_FLAG_DELTA_SPECULAR;
                }
                if (!(scatter_pdf > 0.0f) || !(scatter_pdf_rev > 0.0f)) {
                    push_terminal(hr);
                    pipeline_finish_ray(ps, ray_cf);
                    continue;
                }
                emit_scatter(new_dir, na,
                             chose_diffuse ? BDPT_DOMAIN_PROJ_SOLID_ANGLE : BDPT_DOMAIN_SOLID_ANGLE,
                             scatter_pdf, scatter_pdf_rev, scatter_flags);
                pipeline_spawn_child(ps, make_child(new_dir, na));
            } else {
                push_terminal(hr);  /* amplitude extinguished */
            }
        }

        pipeline_finish_ray(ps, ray_cf);
    }  /* end for rh : batch */

        auto t1 = Clock::now();
        ps.stats[2].record(n,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
            ps.Q_refined.size());
    }  /* end while pop_batch */
}

/* ── T4: wave solver ──────────────────────────────────────────────────────── */

static void pipeline_wave_solver(RayPipelineState& ps)
{
    using Clock = std::chrono::high_resolution_clock;
    std::vector<WaveIntent> batch;

    while (true) {
        batch.clear();
        int bsz = ps.stats[3].batch_sz.load(std::memory_order_relaxed);
        int n   = ps.Q_wave.pop_batch(batch, bsz);
        if (n == 0) break;
        auto t0 = Clock::now();

        for (auto& wi : batch) {
            if (wi.arena_id < 0 || wi.arena_id >= (int)ps.arenas.size()) {
                pipeline_finish_ray(ps, wi.ray.color_flag);
                continue;
            }
            WaveArena& arena = ps.arenas[static_cast<size_t>(wi.arena_id)];
            {
                std::lock_guard<std::mutex> lk(arena.mu);
                wave_arena_seed(arena, wi.ray);
                wave_arena_march(arena);
            }
            ChildRay cr = wave_arena_extract(arena, wi.ray);

            {
                RayRecord rec;
                rec.kind       = RayRecordKind::FIELD;
                rec.tag        = wi.ray.tag;
                rec.src_id     = wi.ray.src_id;
                rec.bounce     = wi.ray.bounce;
                rec.arena_id   = wi.arena_id;
                rec.color_flag = wi.ray.color_flag;
                rec.pos[0]   = static_cast<float>(wi.ray.pos.x());
                rec.pos[1]   = static_cast<float>(wi.ray.pos.y());
                rec.pos[2]   = static_cast<float>(wi.ray.pos.z());
                rec.dir[0]   = static_cast<float>(wi.ray.dir.x());
                rec.dir[1]   = static_cast<float>(wi.ray.dir.y());
                rec.dir[2]   = static_cast<float>(wi.ray.dir.z());
                rec.path_len = static_cast<float>(wi.ray.path_len);
                const int nb = std::min((int)cr.intent.amp.size(), RAY_RECORD_MAX_BANDS);
                rec.n_bands  = nb;
                for (int b = 0; b < nb; ++b) {
                    rec.amp_re[b] = static_cast<float>(cr.intent.amp[b].real());
                    rec.amp_im[b] = static_cast<float>(cr.intent.amp[b].imag());
                }
                ps.Q_out.push(std::move(rec));
            }

            double max_abs = 0.0;
            for (int b = 0; b < (int)cr.intent.amp.size(); ++b)
                max_abs = std::max(max_abs, std::abs(cr.intent.amp[b]));

            if (max_abs >= wi.ray.min_amplitude && cr.intent.bounces_left > 0)
                pipeline_spawn_child(ps, std::move(cr.intent));

            pipeline_finish_ray(ps, wi.ray.color_flag);
        }

        auto t1 = Clock::now();
        ps.stats[3].record(n,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
            ps.Q_wave.size());
    }
}

/* ── Public API ───────────────────────────────────────────────────────────── */

/* (Re)scan mat_cache reflectances; flag materials whose max |refl| <
 * cfg.min_amplitude.  T3 uses these flags to skip child spawning immediately,
 * saving BVH + Fresnel work for rays that would die on the next bounce anyway. */
static void pipeline_precompute_epsilon_flags(RayPipelineState& ps)
{
    const MatSpectralCache& c = ps.st->mat_cache;
    const int nm = c.n_mats;
    const int nb = c.n_bands;
    const double threshold = ps.cfg.min_amplitude;
    ps.mat_epsilon_flags.assign(static_cast<size_t>(std::max(nm, 0)), 0u);
    for (int m = 0; m < nm; ++m) {
        double max_refl = 0.0;
        for (int b = 0; b < nb; ++b)
            max_refl = std::max(max_refl, std::abs(mat_cache_refl(c, m, b)));
        if (max_refl < threshold)
            ps.mat_epsilon_flags[static_cast<size_t>(m)] = 1u;
    }
}

RayPipelineState* ray_pipeline_create(
    RayTracerState*          st,
    const RayPipelineConfig* cfg)
{
    if (!st) return nullptr;
    auto* ps  = new RayPipelineState();
    ps->st    = st;
    ps->cfg   = cfg ? *cfg : RayPipelineConfig{};

    for (int i = 0; i < (int)st->scale_contexts.size(); ++i) {
        const RtScaleContext& ctx = st->scale_contexts[static_cast<size_t>(i)];
        if (ctx.scale_type != RT_SCALE_WAVE) continue;
        ps->arenas.emplace_back();
        wave_arena_build(ps->arenas.back(),
                         (int)ps->arenas.size() - 1,
                         ctx, ps->cfg, *st);
    }

    const uint64_t base  = static_cast<uint64_t>(ps->cfg.seed);
    const int      n_hw  = std::max(1, (int)std::thread::hardware_concurrency());
    const int      n_t3  = std::max(1, n_hw / 2);

    /* Pre-compute epsilon material flags before any CPU/GPU worker can consume
     * scene data.  GPU scene upload copies these flags once into ssbo_t3_meta. */
    pipeline_precompute_epsilon_flags(*ps);

    /* When gpu_all_stages is requested we defer CPU T1/T2/T3 workers so the GPU
     * can take all work without competing.  They are added as a fallback below if
     * GPU init fails. */
    const bool defer_cpu = ps->cfg.use_gpu_compute && ps->cfg.gpu_all_stages;

    auto spawn_cpu_stages = [&]() {
        ps->workers.emplace_back([ps](){ pipeline_intersector(*ps); });
        ps->workers.emplace_back([ps](){ pipeline_refiner(*ps); });
        for (int i = 0; i < n_t3; ++i) {
            uint64_t seed_i = base ^ (static_cast<uint64_t>(0xDEADBEEFULL) * static_cast<uint64_t>(i + 1));
            ps->workers.emplace_back([ps, seed_i](){ pipeline_material(*ps, seed_i); });
        }
    };

    if (!defer_cpu) spawn_cpu_stages();

    if (!ps->arenas.empty() && !ps->cfg.use_gpu_compute)
        ps->workers.emplace_back([ps](){ pipeline_wave_solver(*ps); });

    /* GPU dispatch: when gpu_all_stages=false both CPU and GPU workers compete
     * on shared queues; when gpu_all_stages=true only the GPU worker runs T1-T3. */
    if (ps->cfg.use_gpu_compute) {
        ps->gpu_dispatch = new RayPipelineState::GlPipelineDispatch();
        if (ps->gpu_dispatch->init(*ps, ps->cfg.shader_dir)) {
            fprintf(stderr, "[gpu-dispatch] GL 4.3 compute context ready — GPU thread starting%s\n",
                    defer_cpu ? " (exclusive: CPU T1/T2/T3 not spawned)" : "");
            fflush(stderr);
            /* Also print to stdout so Jupyter/notebook users see it */
            printf("[gpu-dispatch] GL 4.3 compute context ready — GPU thread starting%s\n",
                   defer_cpu ? " (exclusive: CPU T1/T2/T3 not spawned)" : "");
            fflush(stdout);
            ps->gpu_dispatch_state.store(2, std::memory_order_release);
            gl_compute_release_current();   /* transfer ownership to the GPU worker thread */
            ps->workers.emplace_back([ps]() {
                ps->gpu_dispatch->thread_main(*ps);
            });
        } else {
            fprintf(stderr, "[gpu-dispatch] GL init FAILED (%s) — falling back to CPU\n",
                    ps->gpu_dispatch->ctx.error[0] ? ps->gpu_dispatch->ctx.error : "unknown error");
            fflush(stderr);
            printf("[gpu-dispatch] GL init FAILED (%s) — falling back to CPU\n",
                   ps->gpu_dispatch->ctx.error[0] ? ps->gpu_dispatch->ctx.error : "unknown error");
            fflush(stdout);
            ps->gpu_dispatch_state.store(1, std::memory_order_release);
            delete ps->gpu_dispatch;
            ps->gpu_dispatch = nullptr;
            /* GPU failed — if we deferred CPU workers above, add them now */
            if (defer_cpu) spawn_cpu_stages();
            /* Restore T4 CPU wave thread */
            if (!ps->arenas.empty())
                ps->workers.emplace_back([ps](){ pipeline_wave_solver(*ps); });
        }
    }

    /* Start the persistent KPN T5 worker.  It blocks on Q_t5_ready until
     * bdpt_update_inflight pushes a sentinel (both families exhausted), then
     * runs ray_pipeline_run_bdpt_connection and loops back to block again. */
    ps->bdpt_t5_worker = std::thread(t5_kpn_worker, ps);

    return ps;
}

void ray_pipeline_destroy(RayPipelineState* ps)
{
    if (!ps) return;
    ps->Q_intent.set_done();
    ps->Q_hit.set_done();
    ps->Q_refined.set_done();
    ps->Q_wave.set_done();
    ps->Q_bdpt_vertices.set_done();
    ps->Q_bdpt_spectral.set_done();
    ps->Q_bdpt_pdfs.set_done();
    ps->Q_bdpt_optical.set_done();
    ps->Q_bdpt_connections.set_done();
    ps->Q_t5_ready.set_done();         /* unblocks the KPN T5 worker so it can exit */
    for (auto& t : ps->workers) if (t.joinable()) t.join();
    if (ps->bdpt_t5_worker.joinable()) ps->bdpt_t5_worker.join();
    if (ps->gpu_dispatch) {
        gl_compute_destroy_context(&ps->gpu_dispatch->ctx);
        delete ps->gpu_dispatch;
        ps->gpu_dispatch = nullptr;
    }
    delete ps;
}

void ray_pipeline_submit(
    RayPipelineState*  ps,
    const RayIntent*   intents,
    int                n_intents)
{
    if (!ps || !intents || n_intents <= 0) return;
    ps->in_flight.fetch_add(n_intents, std::memory_order_relaxed);
    const int    max_q   = ps->cfg.max_intent_queue;
    const double min_amp = ps->cfg.min_amplitude;
    /* Count flash/sensor split before publishing to Q_intent.  Workers can pop
     * and finish an intent before the counter reservation below; incrementing
     * first ensures the family counter never goes negative and the sentinel
     * in bdpt_update_inflight cannot fire prematurely. */
    int64_t delta_flash = 0, delta_sensor = 0;
    for (int i = 0; i < n_intents; ++i) {
        if (intents[i].color_flag == 1) ++delta_sensor; else ++delta_flash;
    }
    if (delta_flash  > 0) ps->bdpt_inflight_flash.fetch_add(delta_flash,  std::memory_order_release);
    if (delta_sensor > 0) ps->bdpt_inflight_sensor.fetch_add(delta_sensor, std::memory_order_release);
    for (int i = 0; i < n_intents; ++i) {
        RayIntent ri = intents[i];
        if (min_amp > ri.min_amplitude) ri.min_amplitude = min_amp;
        if (max_q > 0)
            ps->Q_intent.push_bounded(std::move(ri), max_q);
        else
            ps->Q_intent.push(std::move(ri));
    }
}

int ray_pipeline_submit_emissive_triangles(
    RayPipelineState* ps,
    const int*        tri_ids,
    int               n_tris,
    int               rays_per_tri,
    double            exposure_weight,
    double            emitter_amp_gain,
    int               max_bounces,
    double            min_amplitude,
    double            interaction_target_x,
    double            interaction_target_y,
    double            interaction_target_z,
    double            interaction_target_r,
    uint32_t          seed)
{
    if (!ps || !ps->st || !tri_ids || n_tris <= 0 || rays_per_tri <= 0)
        return 0;

    RayTracerState& st = *ps->st;
    const int nb = std::max(1, st.n_bands);
    const int n_scene_tris = static_cast<int>(st.tris.size());
    const double scale = std::max(0.0, exposure_weight) * std::max(0.0, emitter_amp_gain);
    if (!(scale > 0.0)) return 0;

    static constexpr int SUBMIT_CHUNK = 262144;
    std::vector<RayIntent> batch;
    batch.reserve(SUBMIT_CHUNK);

    std::mt19937_64 rng(static_cast<uint64_t>(seed) * 6364136223846793005ULL
                        + 1442695040888963407ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    /* ── Flash modifier ────────────────────────────────────────────────────
     * Selects which cosine-hemisphere rays are submitted.  All modes use
     * the interaction target as the reference exit geometry.
     *
     * SNOOT: disk test — each emitter point casts a cone to the exit disk.
     *        Axis is per-ray (origin→disk_center) so the cone is always
     *        centred on the target regardless of emitter position.
     * GRID : square egg-crate cells aligned to the emitter face (YZ plane).
     *        Cell size and tube depth define the maximum exit angle per cell.
     * SCRIM: stochastic transmittance — accept with probability = param0. */
    struct FlashModifier {
        FlashModifierType type  = FlashModifierType::NONE;
        V3d  disk_center        = V3d(0, 0, 0);
        double disk_r           = 0.0;
        double grid_cell_r      = 0.0025;   /* half cell width (m) */
        double grid_depth       = 0.025;    /* tube length (m)     */
        double scrim_pass       = 0.5;      /* acceptance prob     */

        bool accept(const V3d& origin, const V3d& dir,
                    std::mt19937_64& rng_) const
        {
            switch (type) {
            default:
            case FlashModifierType::NONE:
                return true;

            case FlashModifierType::SNOOT: {
                /* Per-ray axis: from this emitter point toward the exit disk. */
                const V3d ax = (disk_center - origin);
                const double ax_len = ax.norm();
                if (ax_len < 1e-9) return true;
                const V3d axis = ax / ax_len;
                const double denom = dir.dot(axis);
                if (denom <= 1e-9) return false;
                const double t = ax_len / denom;   /* ax already = disk_center-origin */
                if (t <= 0.0) return false;
                const V3d hit = origin + t * dir;
                return (hit - disk_center).squaredNorm() <= disk_r * disk_r;
            }

            case FlashModifierType::GRID: {
                /* Snap origin to nearest square cell centre in YZ (emitter face). */
                const double cell_diam = 2.0 * grid_cell_r;
                const double cy = std::round(origin.y() / cell_diam) * cell_diam;
                const double cz = std::round(origin.z() / cell_diam) * cell_diam;
                /* Exit aperture: one grid_depth downstream along -X. */
                if (dir.x() >= 0.0) return false;  /* facing away from subject */
                const double t = -grid_depth / dir.x();
                const V3d hit = origin + t * dir;
                return std::abs(hit.y() - cy) <= grid_cell_r &&
                       std::abs(hit.z() - cz) <= grid_cell_r;
            }

            case FlashModifierType::SCRIM: {
                std::uniform_real_distribution<double> U_(0.0, 1.0);
                return U_(rng_) < scrim_pass;
            }
            }
        }
    };

    const bool has_target =
        interaction_target_r > 0.0 &&
        std::isfinite(interaction_target_x) &&
        std::isfinite(interaction_target_y) &&
        std::isfinite(interaction_target_z);

    FlashModifier modifier;
    if (has_target) {
        modifier.type        = ps->cfg.flash_modifier_type;
        modifier.disk_center = V3d(interaction_target_x,
                                   interaction_target_y,
                                   interaction_target_z);
        modifier.disk_r      = interaction_target_r;
        const float p0 = ps->cfg.flash_modifier_param0;
        const float p1 = ps->cfg.flash_modifier_param1;
        /* GRID: param0=cell_mm, param1=depth_mm */
        modifier.grid_cell_r = (p0 > 0.0f ? static_cast<double>(p0) : 5.0f) * 0.5e-3;
        modifier.grid_depth  = (p1 > 0.0f ? static_cast<double>(p1) : 25.0f) * 1e-3;
        /* SCRIM: param0=transmittance */
        modifier.scrim_pass  = (p0 > 0.0f && p0 <= 1.0f)
                               ? static_cast<double>(p0) : 0.5;
    } else {
        modifier.type = FlashModifierType::NONE;
    }

    uint64_t culled = 0;

    auto flush_batch = [&]() {
        if (!batch.empty()) {
            ray_pipeline_submit(ps, batch.data(), static_cast<int>(batch.size()));
            batch.clear();
        }
    };

    int submitted = 0;
    for (int si = 0; si < n_tris; ++si) {
        const int tri_id = tri_ids[si];
        if (tri_id < 0 || tri_id >= n_scene_tris) continue;
        const Triangle& T = st.tris[static_cast<size_t>(tri_id)];
        const int mat = T.mat_idx;
        if (mat < 0 || mat >= st.mat_n_mats) continue;

        const double area = (tri_id < static_cast<int>(st.tri_areas.size()))
            ? std::max(0.0, st.tri_areas[static_cast<size_t>(tri_id)])
            : 0.5 * T.edge1.cross(T.edge2).norm();
        if (!(area > 0.0)) continue;

        std::vector<cd> emit(static_cast<size_t>(nb), cd(0.0, 0.0));
        double emit_sum = 0.0;
        for (int b = 0; b < nb; ++b) {
            const double e = std::max(0.0, static_cast<double>(mat_band_record(st, mat, b)[5]));
            const double a = scale * area * e / static_cast<double>(rays_per_tri);
            emit[static_cast<size_t>(b)] = cd(a, 0.0);
            emit_sum += a;
        }
        if (!(emit_sum > 0.0)) continue;

        V3d n = T.normal;
        if (n.norm() < 1.0e-12)
            n = T.edge1.cross(T.edge2);
        if (n.norm() < 1.0e-12) continue;
        n.normalize();

        for (int r = 0; r < rays_per_tri; ++r) {
            double u = U(rng);
            double v = U(rng);
            if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
            V3d origin = T.v0 + u * T.edge1 + v * T.edge2;
            V3d emit_n = n;
            apply_parametric_surface_point(st, tri_id, origin, origin, emit_n);
            if (emit_n.norm() < 1.0e-12) emit_n = n;
            emit_n.normalize();

            V3d dir = cosine_hemisphere(emit_n, rng);
            if (!modifier.accept(origin, dir, rng)) {
                ++culled;
                continue;
            }
            RayIntent ri{};
            ri.pos = origin + dir * (EPS * 200.0);
            ri.dir = dir;
            ri.amp.resize(nb);
            for (int b = 0; b < nb; ++b)
                ri.amp[b] = emit[static_cast<size_t>(b)];
            ri.src_id = si;
            ri.tag = 0u;
            ri.color_flag = 0u;
            ri.bounces_left = max_bounces;
            ri.min_amplitude = min_amplitude;
            uint32_t sid = ps->bdpt_next_subpath_id.fetch_add(1u, std::memory_order_relaxed);
            if (sid == 0u)
                sid = ps->bdpt_next_subpath_id.fetch_add(1u, std::memory_order_relaxed);
            ri.bdpt_subpath_id = sid;
            ri.bdpt_vertex = 0u;
            ri.bdpt_stream = BDPT_SIDE_LIGHT;
            ri.bdpt_strategy = 0u;
            batch.push_back(std::move(ri));
            ++submitted;
            if (static_cast<int>(batch.size()) >= SUBMIT_CHUNK)
                flush_batch();
        }
    }

    flush_batch();
    if (culled > 0)
        ps->emitter_world_culled.fetch_add(culled, std::memory_order_relaxed);
    return submitted;
}

int ray_pipeline_drain(
    RayPipelineState*        ps,
    std::vector<RayRecord>&  out,
    int                      max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_out.drain(out, max_n);
}

int ray_pipeline_in_flight(const RayPipelineState* ps)
{
    return ps ? (int)ps->in_flight.load(std::memory_order_relaxed) : 0;
}

void ray_pipeline_get_stats(const RayPipelineState* ps, RayPipelineStats* out)
{
    if (!ps || !out) return;
    auto snap = [](const StageStats& s, RayPipelineStats::Stage& d) {
        const uint64_t cpu_ns = s.ns_active.load(std::memory_order_relaxed);
        const uint64_t gpu_ns = s.ns_gpu.load(std::memory_order_relaxed);
        d.throughput      = s.items_per_sec();
        d.processed       = s.n_processed.load(std::memory_order_relaxed);
        d.batch_size      = s.batch_sz.load(std::memory_order_relaxed);
        d.queue_depth     = s.q_depth.load(std::memory_order_relaxed);
        d.gpu_throughput  = s.gpu_items_per_sec();
        d.gpu_processed   = s.n_gpu.load(std::memory_order_relaxed);
        d.gpu_batch_size  = s.batch_sz_gpu.load(std::memory_order_relaxed);
        d.gpu_fraction    = s.gpu_fraction();
        d.cpu_active_ms   = (double)cpu_ns / 1.0e6;
        d.gpu_active_ms   = (double)gpu_ns / 1.0e6;
    };
    snap(ps->stats[0], out->t1);
    snap(ps->stats[1], out->t2);
    snap(ps->stats[2], out->t3);
    snap(ps->stats[3], out->t4);
    snap(ps->stats[4], out->t5);
    out->output_queue_depth = ps->Q_out.size();
    out->in_flight          = ps->in_flight.load(std::memory_order_relaxed);
    out->gpu_uv_readback_bytes = ps->gpu_uv_readback_bytes.load(std::memory_order_relaxed);
    out->gpu_uv_readback_count = ps->gpu_uv_readback_count.load(std::memory_order_relaxed);
    out->gpu_hit_readback_bytes = ps->gpu_hit_readback_bytes.load(std::memory_order_relaxed);
    out->gpu_hit_readback_count = ps->gpu_hit_readback_count.load(std::memory_order_relaxed);
}

void ray_pipeline_set_min_amplitude(RayPipelineState* ps, double eps)
{
    if (!ps) return;
    ps->cfg.min_amplitude = eps;
    pipeline_precompute_epsilon_flags(*ps);
}

void ray_pipeline_precompute_epsilon_flags(RayPipelineState* ps)
{
    if (!ps) return;
    pipeline_precompute_epsilon_flags(*ps);
}

int ray_pipeline_n_bands(const RayPipelineState* ps)
{
    if (!ps || !ps->st) return 0;
    return ps->st->n_bands;
}

int ray_pipeline_tri_mat_idx(const RayPipelineState* ps, int tri_idx)
{
    if (!ps || !ps->st || tri_idx < 0 ||
        static_cast<size_t>(tri_idx) >= ps->st->tris.size()) return -1;
    return ps->st->tris[static_cast<size_t>(tri_idx)].mat_idx;
}

int ray_pipeline_drain_refined(RayPipelineState* ps,
                                std::vector<RefinedHit>& out,
                                int max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_refined.drain(out, max_n);
}

int ray_pipeline_drain_bdpt_vertices(RayPipelineState* ps,
                                      std::vector<BdptVertexRecord>& out,
                                      int max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_bdpt_vertices.drain(out, max_n);
}

int ray_pipeline_drain_bdpt_spectral(RayPipelineState* ps,
                                      std::vector<BdptSpectralWeightRecord>& out,
                                      int max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_bdpt_spectral.drain(out, max_n);
}

int ray_pipeline_drain_bdpt_pdfs(RayPipelineState* ps,
                                  std::vector<BdptPdfRecord>& out,
                                  int max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_bdpt_pdfs.drain(out, max_n);
}

int ray_pipeline_drain_bdpt_optical(RayPipelineState* ps,
                                     std::vector<BdptOpticalEventRecord>& out,
                                     int max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_bdpt_optical.drain(out, max_n);
}

int ray_pipeline_drain_bdpt_connections(RayPipelineState* ps,
                                          std::vector<BdptConnectionRecord>& out,
                                          int max_n)
{
    if (!ps || max_n <= 0) return 0;
    return ps->Q_bdpt_connections.drain(out, max_n);
}

static bool bdpt_find_parametric_stop_target(const RayTracerState& st,
                                             double& out_x,
                                             double& out_radius)
{
    bool have = false;
    double best_r = std::numeric_limits<double>::infinity();
    for (size_t gi = 0; gi < st.tri_group_parametric_kind.size(); ++gi) {
        if (st.tri_group_parametric_kind[gi] != TRI_PARAM_SURFACE_PARAMETRIC_LENS)
            continue;
        const auto& payload = st.tri_group_parametric_payload[gi];
        auto f32 = [&](size_t idx) -> float {
            float v = 0.0f;
            const size_t off = idx * sizeof(float);
            if (off + sizeof(float) <= payload.size())
                std::memcpy(&v, payload.data() + off, sizeof(float));
            return v;
        };
        if (payload.size() < 8u * sizeof(float) || f32(0) != 14949.0f)
            continue;
        const int n_surf = static_cast<int>(f32(1));
        if (n_surf < 1 || n_surf > 64)
            continue;
        if (payload.size() < static_cast<size_t>(8 + n_surf * 8) * sizeof(float))
            continue;

        for (int s = 0; s < n_surf; ++s) {
            const size_t sb = static_cast<size_t>(8 + s * 8);
            const double x  = static_cast<double>(f32(sb + 0));
            const double ap = static_cast<double>(f32(sb + 4));
            const int flags = static_cast<int>(f32(sb + 6));
            if (!(ap > 0.0) || !std::isfinite(ap))
                continue;
            if ((flags & 1) != 0) {
                out_x = x;
                out_radius = ap;
                return true;
            }
            if (ap < best_r) {
                best_r = ap;
                out_x = x;
                out_radius = ap;
                have = true;
            }
        }
    }
    return have;
}

int ray_pipeline_submit_sensor_sweep(RayPipelineState* ps,
                                      int max_bounces,
                                      double min_amplitude,
                                      int max_rays,
                                      int pix_offset,
                                      uint64_t aperture_seed,
                                      int shutter_mode,
                                      double shutter_open,
                                      double shutter_center_u,
                                      double shutter_center_v,
                                      double shutter_softness,
                                      double exposure_weight)
{
    if (!ps || !ps->st || ps->sensor_res <= 0 || ps->sensor_half_w <= 0.0f || ps->sensor_half_h <= 0.0f)
        return 0;

    const int res   = ps->sensor_res;
    const int total = res * res;
    const int start = std::max(0, std::min(pix_offset, total));
    const int limit = (max_rays > 0) ? std::min(max_rays, total - start) : (total - start);
    if (limit <= 0) return 0;

    const double target_x = static_cast<double>(ps->sensor_target_x);
    const double target_r = static_cast<double>(ps->sensor_target_r);
    if (!(target_r > 0.0) || !std::isfinite(target_x))
        return 0;

    /* ── Aperture disk sampling ─────────────────────────────────────────────
     * aperture_seed == 0: all rays aim at the aperture center (backward compat).
     * aperture_seed >  0: Fibonacci / golden-angle quasi-random aperture point.
     *
     * Batch index s maps to a point on the aperture disk via:
     *   radius   = sqrt(van_der_corput_base2(s)) × target_r   (uniform area)
     *   angle    = s × golden_angle                            (Fibonacci spiral)
     *
     * Van der Corput base-2 reverses the bits of s to produce a perfectly
     * stratified sequence in [0,1) without needing to know N in advance.
     * The result covers the aperture disk evenly across any number of batches.
     */
    double ap_y = 0.0, ap_z = 0.0;
    if (aperture_seed > 0 && target_r > 0.0) {
        static constexpr double kGoldenAngle = 2.39996322972865332; /* 2π/φ² */
        uint32_t s = static_cast<uint32_t>(aperture_seed);
        /* Van der Corput base-2 bit reversal */
        s = ((s & 0x55555555u) << 1) | ((s & 0xAAAAAAAAu) >> 1);
        s = ((s & 0x33333333u) << 2) | ((s & 0xCCCCCCCCu) >> 2);
        s = ((s & 0x0F0F0F0Fu) << 4) | ((s & 0xF0F0F0F0u) >> 4);
        s = ((s & 0x00FF00FFu) << 8) | ((s & 0xFF00FF00u) >> 8);
        const double r_frac = static_cast<double>(s) / 4294967296.0; /* [0,1) */
        const double ap_r   = std::sqrt(r_frac) * target_r;
        const double theta  = static_cast<double>(aperture_seed) * kGoldenAngle;
        ap_y = ap_r * std::cos(theta);
        ap_z = ap_r * std::sin(theta);
    }
    const V3d target(target_x, ap_y, ap_z);

    const int n_bands = std::max(1, std::min(ps->st->n_bands, MAX_SPECTRAL_BANDS));
    const double half_w = static_cast<double>(ps->sensor_half_w);
    const double half_h = static_cast<double>(ps->sensor_half_h);
    const double step_w = (2.0 * half_w) / static_cast<double>(res);
    const double step_h = (2.0 * half_h) / static_cast<double>(res);
    const double open_f = std::max(0.0, std::min(1.0, shutter_open));
    const double cu = std::max(0.0, std::min(1.0, shutter_center_u));
    const double cv = std::max(0.0, std::min(1.0, shutter_center_v));
    const double soft = std::max(0.0, shutter_softness);
    const double exposure_w = std::max(0.0, exposure_weight);
    if (exposure_w <= 0.0 || shutter_mode == 1 || open_f <= 0.0)
        return 0;

    auto smooth_gate = [&](double edge) -> double {
        if (soft <= 1.0e-12)
            return edge >= 0.0 ? 1.0 : 0.0;
        const double x = std::max(0.0, std::min(1.0, 0.5 + 0.5 * edge / soft));
        return x * x * (3.0 - 2.0 * x);
    };
    auto shutter_weight_at = [&](double u, double v) -> double {
        if (shutter_mode == 0) return 1.0;
        if (shutter_mode == 2) {
            if (open_f >= 1.0) return 1.0;
            const double r = 0.5 * std::sqrt(open_f);
            const double du = u - cu;
            const double dv = v - cv;
            return smooth_gate(r - std::sqrt(du*du + dv*dv));
        }
        if (shutter_mode == 3) {
            if (open_f >= 1.0) return 1.0;
            return smooth_gate(0.5 * open_f - std::abs(u - cu));
        }
        if (shutter_mode == 4) {
            if (open_f >= 1.0) return 1.0;
            return smooth_gate(0.5 * open_f - std::abs(v - cv));
        }
        return 1.0;
    };
    std::vector<RayIntent> intents;
    intents.reserve(static_cast<size_t>(limit));

    for (int pix = start; pix < total && static_cast<int>(intents.size()) < limit; ++pix) {
        const int iy = pix / res;
        const int iz = pix - iy * res;
        const double y = -half_w + (static_cast<double>(iy) + 0.5) * step_w;
        const double z = -half_h + (static_cast<double>(iz) + 0.5) * step_h;
        const double u = (z + half_h) / std::max(2.0 * half_h, 1.0e-30);
        const double v = (y + half_w) / std::max(2.0 * half_w, 1.0e-30);
        const double shutter_w = shutter_weight_at(u, v);
        if (!(shutter_w > 0.0) || !std::isfinite(shutter_w))
            continue;
        V3d pos(static_cast<double>(ps->sensor_px), y, z);
        V3d dir = target - pos;
        if (dir.squaredNorm() <= 1e-24)
            dir = V3d(-1.0, 0.0, 0.0);
        else
            dir.normalize();

        RayIntent ri;
        ri.pos = pos + dir * (T_SELF * 256.0);
        ri.dir = dir;
        ri.amp.resize(n_bands);
        ri.amp.setConstant(exposure_w * shutter_w);
        ri.src_id = pix;
        ri.tag = static_cast<uint64_t>(pix);
        ri.color_flag = 1u;
        ri.bounce = 0;
        ri.bounces_left = std::max(1, max_bounces);
        ri.min_amplitude = min_amplitude;
        ri.sensor_origin_y = static_cast<float>(y);
        ri.sensor_origin_z = static_cast<float>(z);
        uint32_t sid = ps->bdpt_next_subpath_id.fetch_add(1u, std::memory_order_relaxed);
        sid = 0x80000000u | (sid & 0x7FFFFFFFu);
        if (sid == 0u)
            sid = 0x80000001u;
        ri.bdpt_subpath_id = sid;
        ri.bdpt_vertex = 0u;
        ri.bdpt_stream = BDPT_SIDE_SENSOR;
        ri.bdpt_strategy = 0u;
        intents.push_back(std::move(ri));
    }

    if (intents.empty()) return 0;
    ray_pipeline_submit(ps, intents.data(), static_cast<int>(intents.size()));
    return static_cast<int>(intents.size());
}

void ray_pipeline_get_bdpt_overflow(const RayPipelineState* ps,
                                     uint64_t* out_vertices,
                                     uint64_t* out_spectral,
                                     uint64_t* out_pdfs,
                                     uint64_t* out_optical,
                                     uint64_t* out_connections)
{
    if (!ps) return;
    if (out_vertices) *out_vertices = ps->bdpt_overflow_vertices.load(std::memory_order_relaxed);
    if (out_spectral) *out_spectral = ps->bdpt_overflow_spectral.load(std::memory_order_relaxed);
    if (out_pdfs)     *out_pdfs     = ps->bdpt_overflow_pdfs    .load(std::memory_order_relaxed);
    if (out_optical)  *out_optical  = ps->bdpt_overflow_optical .load(std::memory_order_relaxed);
    if (out_connections) *out_connections = ps->bdpt_overflow_connections.load(std::memory_order_relaxed);
}

/* t5_connection.h provides T5ConnContext, BdptSubpathView, LightVertRef,
 * BdptCandidateScratch, T5ThreadAccum, T5GpuParams, run_t5_allpairs().
 * Must be included here (after RayTracerState, BVHNode, etc. are defined). */
#include "t5_connection.h"

/* ── BDPT connection pass with balance-heuristic MIS ─────────────────────────
 *
 * Drains the BDPT side records, reconstructs full camera/light subpaths, and
 * evaluates all connectable vertex pairs as (s,t) strategies.  This deliberately
 * does not collapse each subpath to a terminal endpoint: every non-delta vertex
 * can be the connection vertex for its side.
 *
 * Current estimator contract:
 *   - Vertex records provide geometry, stream, pixel origin, material flags.
 *   - Spectral records provide per-band beta; scalar throughput is used only
 *     when it was explicitly emitted on the vertex record.
 *   - PDF records provide sampled-direction densities between vertex i and i+1.
 *   - Prefix PDFs are products of valid area-measure edge PDFs; missing PDF
 *     data invalidates downstream strategies instead of pretending unit density.
 *   - Optical records contribute reversible transform Jacobians where present.
 *   - MIS is a balance heuristic over all valid vertex-pair strategies for the
 *     same camera/light subpath pair.
 *
 * Results accumulate into sensor channel 2 at the sensor_origin_y/z carried by
 * the camera subpath. */
void ray_pipeline_run_bdpt_connection(RayPipelineState* ps)
{
    if (!ps || !ps->st || ps->sensor_res <= 0) return;
    bool expected_running = false;
    if (!ps->bdpt_connection_running.compare_exchange_strong(
            expected_running, true,
            std::memory_order_acq_rel,
            std::memory_order_relaxed))
        return;
    struct BdptConnectionRunGuard {
        RayPipelineState* ps;
        ~BdptConnectionRunGuard() {
            ps->bdpt_connection_running.store(false, std::memory_order_release);
        }
    } run_guard{ps};
    auto t5_start = std::chrono::steady_clock::now();
    size_t t5_work_items = 0;
    struct BdptT5StatsGuard {
        RayPipelineState* ps;
        std::chrono::steady_clock::time_point start;
        size_t* work_items;
        bool used_gpu = false;   /* set to true after GPU dispatch is confirmed */
        ~BdptT5StatsGuard() {
            const auto t1 = std::chrono::steady_clock::now();
            const uint64_t ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - start).count());
            const int n = static_cast<int>(std::min<size_t>(
                std::max<size_t>(1, *work_items),
                static_cast<size_t>(std::numeric_limits<int>::max())));
            if (used_gpu)
                ps->stats[4].record_gpu(n, ns, 0);
            else
                ps->stats[4].record(n, ns, 0);
        }
    } t5_stats{ps, t5_start, &t5_work_items};

    std::vector<BdptVertexRecord>         verts_new;
    std::vector<BdptSpectralWeightRecord> sweights_new;
    std::vector<BdptPdfRecord>            pdfs_new;
    std::vector<BdptOpticalEventRecord>   optical_new;
    ray_pipeline_drain_bdpt_vertices(ps, verts_new,   ps->cfg.bdpt_max_vertices > 0 ? ps->cfg.bdpt_max_vertices : 2000000);
    ray_pipeline_drain_bdpt_spectral(ps, sweights_new, ps->cfg.bdpt_max_spectral > 0 ? ps->cfg.bdpt_max_spectral : 4000000);
    ray_pipeline_drain_bdpt_pdfs    (ps, pdfs_new,     ps->cfg.bdpt_max_pdfs     > 0 ? ps->cfg.bdpt_max_pdfs     : 2000000);
    ray_pipeline_drain_bdpt_optical (ps, optical_new,  ps->cfg.bdpt_max_optical  > 0 ? ps->cfg.bdpt_max_optical  : 1000000);
    t5_work_items = verts_new.size() + sweights_new.size() + pdfs_new.size() + optical_new.size();
    fprintf(stderr, "[T5-conn] drained: verts=%zu sweights=%zu pdfs=%zu optical=%zu\n",
            verts_new.size(), sweights_new.size(), pdfs_new.size(), optical_new.size());
    fflush(stderr);

    std::vector<BdptVertexRecord>         verts;
    std::vector<BdptSpectralWeightRecord> sweights;
    std::vector<BdptPdfRecord>            pdfs;
    std::vector<BdptOpticalEventRecord>   optical;
    {
        std::lock_guard<std::mutex> pending_lk(ps->bdpt_pending_mu);
        ps->bdpt_pending_vertices.insert(ps->bdpt_pending_vertices.end(),
                                         verts_new.begin(), verts_new.end());
        ps->bdpt_pending_spectral.insert(ps->bdpt_pending_spectral.end(),
                                         sweights_new.begin(), sweights_new.end());
        ps->bdpt_pending_pdfs.insert(ps->bdpt_pending_pdfs.end(),
                                     pdfs_new.begin(), pdfs_new.end());
        ps->bdpt_pending_optical.insert(ps->bdpt_pending_optical.end(),
                                        optical_new.begin(), optical_new.end());

        bool have_camera = false;
        bool have_light  = false;
        for (const auto& v : ps->bdpt_pending_vertices) {
            if (v.stream == BDPT_SIDE_SENSOR) have_camera = true;
            else if (v.stream == BDPT_SIDE_LIGHT) have_light = true;
            if (have_camera && have_light) break;
        }
        fprintf(stderr, "[T5-conn] pending after merge: total=%zu have_camera=%d have_light=%d\n",
                ps->bdpt_pending_vertices.size(), (int)have_camera, (int)have_light);
        fflush(stderr);
        if (!have_camera || !have_light) {
            fprintf(stderr, "[T5-conn] EARLY RETURN — missing %s%s\n",
                    have_camera ? "" : "camera ", have_light ? "" : "light");
            fflush(stderr);
            /* Advance fired so join_t5() does not hang on a zero-ray-side cycle. */
            ps->bdpt_t5_fired.fetch_add(1u, std::memory_order_release);
            return;
        }

        verts.swap(ps->bdpt_pending_vertices);
        sweights.swap(ps->bdpt_pending_spectral);
        pdfs.swap(ps->bdpt_pending_pdfs);
        optical.swap(ps->bdpt_pending_optical);
        t5_work_items = verts.size() + sweights.size() + pdfs.size() + optical.size();
    }

    /* ── Sensor-batching stash logic ──────────────────────────────────────
     * Stash is populated on the first T5 call (when empty) with all light-
     * stream records.  Subsequent calls prepend the stash so the GPU shader
     * always sees the full flash light field alongside the fresh camera batch.
     * Light-stream records are identified by the absence of the 0x80000000
     * high bit in subpath_id (sensor subpaths set that bit in submit_sensor_sweep). */
    if (ps->bdpt_batching_mode) {
        std::lock_guard<std::mutex> stash_lk(ps->bdpt_stash_mu);
        if (ps->bdpt_stash_v.empty()) {
            /* First batch: save light-side records to stash. */
            for (const auto& v : verts)
                if (v.stream == BDPT_SIDE_LIGHT)
                    ps->bdpt_stash_v.push_back(v);
            for (const auto& sw : sweights)
                if (!(sw.subpath_id & 0x80000000u))
                    ps->bdpt_stash_sw.push_back(sw);
            for (const auto& pr : pdfs)
                if (!(pr.subpath_id & 0x80000000u))
                    ps->bdpt_stash_pdf.push_back(pr);
            for (const auto& oe : optical)
                if (!(oe.subpath_id & 0x80000000u))
                    ps->bdpt_stash_opt.push_back(oe);
            fprintf(stderr, "[T5-stash] saved light records: v=%zu sw=%zu pdf=%zu opt=%zu\n",
                    ps->bdpt_stash_v.size(), ps->bdpt_stash_sw.size(),
                    ps->bdpt_stash_pdf.size(), ps->bdpt_stash_opt.size());
            fflush(stderr);
        } else {
            /* Subsequent batches: prepend stash light records. */
            std::vector<BdptVertexRecord> merged;
            merged.reserve(ps->bdpt_stash_v.size() + verts.size());
            merged.insert(merged.end(), ps->bdpt_stash_v.begin(), ps->bdpt_stash_v.end());
            merged.insert(merged.end(), verts.begin(), verts.end());
            verts = std::move(merged);
            sweights.insert(sweights.begin(), ps->bdpt_stash_sw.begin(), ps->bdpt_stash_sw.end());
            pdfs.insert(pdfs.begin(),     ps->bdpt_stash_pdf.begin(), ps->bdpt_stash_pdf.end());
            optical.insert(optical.begin(), ps->bdpt_stash_opt.begin(), ps->bdpt_stash_opt.end());
        }
    }

    if (verts.empty()) {
        ps->bdpt_t5_fired.fetch_add(1u, std::memory_order_release);
        return;
    }

    /* Build spectral beta LUT: key = subpath_id<<32 | vertex_index<<16 | band_id */
    std::unordered_map<uint64_t, std::complex<float>> beta_lut;
    beta_lut.reserve(sweights.size());
    for (const auto& sw : sweights) {
        const uint64_t k = ((uint64_t)sw.subpath_id << 32)
                         | ((uint64_t)sw.vertex_index << 16)
                         | (uint64_t)sw.band_id;
        beta_lut[k] = {sw.beta_re, sw.beta_im};
    }

    /* Build PDF LUT: key = subpath_id<<32 | vertex_index<<16 | sample_domain */
    std::unordered_map<uint64_t, BdptPdfRecord> pdf_lut;
    pdf_lut.reserve(pdfs.size());
    for (const auto& pr : pdfs) {
        const uint64_t k = ((uint64_t)pr.subpath_id << 32)
                         | ((uint64_t)pr.vertex_index << 16)
                         | (uint64_t)pr.sample_domain;
        pdf_lut[k] = pr;
    }

    std::unordered_map<uint64_t, std::vector<BdptOpticalEventRecord>> optical_lut;
    optical_lut.reserve(optical.size());
    for (const auto& oe : optical) {
        const uint64_t k = ((uint64_t)oe.subpath_id << 32)
                         | ((uint64_t)oe.vertex_index << 16)
                         | (uint64_t)oe.stream;
        optical_lut[k].push_back(oe);
    }

    /* ── Pass parameters ──────────────────────────────────────────────────── */
    const int   res   = ps->sensor_res;
    const float inv_w = static_cast<float>(res) / (2.0f * ps->sensor_half_w);
    const float inv_h = static_cast<float>(res) / (2.0f * ps->sensor_half_h);

    /* Determine n_bands: use authoritative pipeline count when available, fall back
     * to scanning spectral weight records, then T5_MAX_GPU_BANDS. */
    uint16_t max_band = 0;
    for (const auto& sw : sweights)
        if (sw.band_id > max_band) max_band = sw.band_id;
    const int n_bands = (ps->st && ps->st->n_bands > 0)
        ? ps->st->n_bands
        : (!sweights.empty() ? (int)max_band + 1 : T5_MAX_GPU_BANDS);

    const Eigen::VectorXd& t5_freq_hz = ps->st
        ? ps->st->freq_hz_vec : Eigen::VectorXd{};
    const float t5_min_geom = (ps->cfg.t5_min_geom > 0.0f)
        ? ps->cfg.t5_min_geom : 1e-20f;
    std::mutex connection_flush_mu;

    /* ── Build T5ConnContext (all helper methods live here) ────────────────── */
    T5ConnContext ctx{beta_lut, pdf_lut, optical_lut, ps,
                     n_bands, res, inv_w, inv_h, t5_min_geom, t5_freq_hz,
                     connection_flush_mu};

    /* ── Build subpath views ────────────────────────────────────────────── */
    std::unordered_map<uint32_t, BdptSubpathView> cam_paths, light_paths;
    for (const auto& v : verts) {
        auto& paths = (v.stream == BDPT_SIDE_SENSOR) ? cam_paths : light_paths;
        paths[v.subpath_id].v.push_back(&v);
    }

    auto prepare_paths = [&ctx](std::unordered_map<uint32_t, BdptSubpathView>& paths) {
        for (auto& kv : paths) {
            auto& sp = kv.second;
            std::sort(sp.v.begin(), sp.v.end(),
                      [](const BdptVertexRecord* a, const BdptVertexRecord* b) {
                          return a->vertex_index < b->vertex_index;
                      });
            sp.v.erase(std::unique(sp.v.begin(), sp.v.end(),
                      [](const BdptVertexRecord* a, const BdptVertexRecord* b) {
                          return a->vertex_index == b->vertex_index;
                      }), sp.v.end());

            sp.prefix_pdf.assign(sp.v.size(), 1.0);
            sp.prefix_valid.assign(sp.v.size(), 1u);
            double accum = 1.0;
            bool valid = true;
            for (size_t i = 0; i < sp.v.size(); ++i) {
                if (i > 0) {
                    double edge_pdf = 0.0;
                    valid = valid && ctx.edge_pdf_area(*sp.v[i-1], *sp.v[i], edge_pdf);
                    if (valid) accum *= edge_pdf;
                }
                sp.prefix_valid[i] = valid ? 1u : 0u;
                sp.prefix_pdf[i]   = valid ? std::max(accum, 1e-300) : 0.0;
            }
        }
    };

    prepare_paths(cam_paths);
    prepare_paths(light_paths);

    if (cam_paths.empty() || light_paths.empty()) return;

    /* ── Flat cam/light item vectors ────────────────────────────────────── */
    std::vector<const BdptSubpathView*> cam_items;
    std::vector<const BdptSubpathView*> light_items;
    cam_items.reserve(cam_paths.size());
    light_items.reserve(light_paths.size());
    for (const auto& kv : cam_paths)  cam_items.push_back(&kv.second);
    for (const auto& kv : light_paths) light_items.push_back(&kv.second);

    /* ── Per-thread accumulators + ThreadPool ───────────────────────────── */
    const size_t pix = static_cast<size_t>(res) * static_cast<size_t>(res);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t nw = std::max<size_t>(1, std::min<size_t>(cam_items.size(), (size_t)hw));

    std::vector<T5ThreadAccum> accums;
    accums.reserve(nw);
    for (size_t t = 0; t < nw; ++t) accums.emplace_back(pix);

    ThreadPool t5_pool(nw);

    /* ── Per-vertex packing helpers (used for both LGV and CGV) ─────────── */
    /* These lambdas compute the fields required for per-vertex BRDF PDF and  *
     * edge-PDF evaluation on the GPU side (t5_full_connect.comp.glsl).       */
    auto get_diffuse_p = [&](int mat_idx) -> float {
        return (mat_idx >= 0)
            ? mat_cache_diffusion(ctx.ps->st->mat_cache, mat_idx)
            : 0.5f;
    };
    auto get_ggx_alpha = [&](int mat_idx) -> float {
        return (mat_idx >= 0)
            ? surf_cache_ggx_alpha(ctx.ps->st->surf_cache, mat_idx)
            : 0.0f;
    };
    auto get_opt_jacobian = [&](const BdptVertexRecord& v) -> float {
        double j = 1.0;
        ctx.optical_allows_sampling(v, false, j);
        return (j > 0.0 && std::isfinite(j)) ? static_cast<float>(j) : 1.0f;
    };
    /* Forward area PDF: edge v → nxt (forward sampling direction). */
    auto get_edge_fwd = [&](const BdptVertexRecord& v,
                             const BdptVertexRecord& nxt) -> float {
        double pdf = 0.0;
        if (ctx.edge_pdf_area(v, nxt, pdf)) return static_cast<float>(pdf);
        return 0.0f;
    };
    /* Backward area PDF: reverse pdf at v for arriving from nxt back to v. */
    auto get_edge_bwd = [&](const BdptVertexRecord& v,
                             const BdptVertexRecord& nxt) -> float {
        double pdf = 0.0;
        if (ctx.edge_pdf_area_component(v, nxt, v, true, pdf))
            return static_cast<float>(pdf);
        return 0.0f;
    };

    /* ── Collect and pack ALL light verts consecutively per subpath ──────── */
    /* ALL vertices are packed (no pre-filter) so subpath_flat_base =          *
     * flat_idx − vertex_index is valid and the GPU shader can navigate the   *
     * full chain needed for candidate_strategy_density MIS.                  */
    std::vector<LightVertRef> light_verts_flat;
    for (const auto* lsp : light_items) {
        for (uint32_t li = 0; li < static_cast<uint32_t>(lsp->v.size()); ++li)
            light_verts_flat.push_back({lsp->v[li], lsp, li});
    }
    const uint32_t n_lv = static_cast<uint32_t>(light_verts_flat.size());

    std::vector<float> light_packed(static_cast<size_t>(n_lv) * T5_LGV_STRIDE, 0.0f);
    for (uint32_t i = 0; i < n_lv; ++i) {
        const BdptVertexRecord& lr  = *light_verts_flat[i].rec;
        const BdptSubpathView*  lsp = light_verts_flat[i].sp;
        const uint32_t          li  = light_verts_flat[i].li;
        float* p = light_packed.data() + static_cast<size_t>(i) * T5_LGV_STRIDE;
        /* base fields [0..10] — geometry + ids + scalar beta */
        p[0]  = lr.pos[0];    p[1]  = lr.pos[1];    p[2]  = lr.pos[2];
        p[3]  = lr.normal[0]; p[4]  = lr.normal[1]; p[5]  = lr.normal[2];
        p[6]  = lr.throughput_scalar;
        std::memcpy(&p[7], &lr.flags,      sizeof(uint32_t));
        std::memcpy(&p[8], &lr.subpath_id, sizeof(uint32_t));
        /* vinfo: bits[0..15]=packed_index, bits[16..30]=stream, bit[31]=tri_valid
         * Use 'li' (loop index = position in packed array) NOT lr.vertex_index
         * (which is bdpt_vertex = bounce count including optical traversals).
         * GPU shader: light_flat_base = gid_l - li_v, so li_v must be the
         * packed-array position, not the bounce count. */
        const uint32_t vinfo = (uint32_t)li
                             | ((uint32_t)lr.stream << 16)
                             | ((lr.tri_id >= 0 ? 1u : 0u) << 31);
        std::memcpy(&p[9], &vinfo, sizeof(uint32_t));
        p[10] = static_cast<float>(ctx.beta_scalar_val(lr));
        /* pdf_fwd / pdf_rev [11..12] — direct from vertex record */
        p[11] = lr.pdf_fwd;
        p[12] = lr.pdf_rev;
        /* pdf_flags [13] — BdptPdfRecord::flags for delta-specular / diffuse / GGX bits */
        {
            const BdptPdfRecord* pr13 = ctx.pdf_for(lr.subpath_id, lr.vertex_index);
            const uint32_t pf = pr13 ? pr13->flags : 0u;
            std::memcpy(&p[13], &pf, sizeof(uint32_t));
        }
        /* optical_block [14] — set if any blocking optical event (absorb/TIR/clip) */
        {
            uint32_t oblock = 0u;
            const uint64_t ok = ((uint64_t)lr.subpath_id  << 32)
                              | ((uint64_t)lr.vertex_index << 16)
                              | (uint64_t)lr.stream;
            auto oit = ctx.optical_lut.find(ok);
            if (oit != ctx.optical_lut.end())
                for (const auto& oe : oit->second)
                    if (oe.reason == BDPT_OPT_ABSORPTION   ||
                        oe.reason == BDPT_OPT_TIR          ||
                        oe.reason == BDPT_OPT_APERTURE_CLIP ||
                        oe.reason == BDPT_OPT_VIGNETTE_CLIP) { oblock = 1u; break; }
            std::memcpy(&p[14], &oblock, sizeof(uint32_t));
        }
        /* prefix_pdf [15] — cumulative forward-PDF prefix for MIS */
        p[15] = (li < lsp->prefix_pdf.size() && lsp->prefix_valid[li])
              ? static_cast<float>(lsp->prefix_pdf[li]) : 0.0f;
        /* per-band beta magnitudes [LGV_BAND_BASE .. LGV_BAND_BASE+T5_MAX_GPU_BANDS-1] */
        for (int b = 0; b < T5_MAX_GPU_BANDS; ++b) {
            if (b >= ctx.n_bands) { p[LGV_BAND_BASE + b] = 0.0f; continue; }
            const uint64_t k = ((uint64_t)lr.subpath_id  << 32)
                             | ((uint64_t)lr.vertex_index << 16)
                             | (uint64_t)b;
            auto it = ctx.beta_lut.find(k);
            p[LGV_BAND_BASE + b] = (it != ctx.beta_lut.end())
                ? static_cast<float>(std::abs(it->second)) : 0.0f;
        }
        /* BRDF + edge PDF fields [LGV_DIR_IN_X .. LGV_EDGE_BWD_AREA] */
        p[LGV_DIR_IN_X + 0] = lr.dir_in[0];
        p[LGV_DIR_IN_X + 1] = lr.dir_in[1];
        p[LGV_DIR_IN_X + 2] = lr.dir_in[2];
        p[LGV_DIFFUSE_P]     = get_diffuse_p(lr.mat_idx);
        p[LGV_GGX_ALPHA]     = get_ggx_alpha(lr.mat_idx);
        p[LGV_OPT_JACOBIAN]  = get_opt_jacobian(lr);
        if (li + 1u < static_cast<uint32_t>(lsp->v.size())) {
            const BdptVertexRecord& nxt = *lsp->v[li + 1u];
            p[LGV_EDGE_FWD_AREA] = get_edge_fwd(lr, nxt);
            p[LGV_EDGE_BWD_AREA] = get_edge_bwd(lr, nxt);
        }
        /* else: edge_fwd_area / edge_bwd_area stay 0.0f (last vert in subpath) */
    }

    /* ── GPU path: full brute-force dispatch ────────────────────────────── */
    const bool gpu_t5_ok = ps->cfg.use_gpu_compute
                        && ps->gpu_dispatch != nullptr
                        && n_lv > 0
                        && !cam_items.empty();

    std::vector<uint32_t> gpu_pixel_result;

    if (gpu_t5_ok) {
        /* Count and pack ALL camera verts consecutively per subpath.
         * No pre-filter: ALL verts are packed so subpath_flat_base =
         * flat_idx − vertex_index is valid for GPU chain navigation. */
        uint32_t n_cv = 0;
        for (const auto* csp : cam_items)
            n_cv += static_cast<uint32_t>(csp->v.size());

        std::vector<float> cam_packed(static_cast<size_t>(n_cv) * T5_CGV_STRIDE, 0.0f);
        uint32_t ci_flat = 0;
        for (const auto* csp : cam_items) {
            for (size_t ci = 0; ci < csp->v.size(); ++ci) {
                const BdptVertexRecord& c = *csp->v[ci];
                /* Compute per-channel spectral beta (R, G, B) — now handled by GPU
                 * using per-band slots [CGV_BAND_BASE..] and the spectral_weights SSBO.
                 * Slots [10..12] are zeroed (legacy CPU pre-computation removed). */

                float* p = cam_packed.data() + static_cast<size_t>(ci_flat++) * T5_CGV_STRIDE;
                /* base fields [0..14] — geometry + ids + spectral display betas */
                p[0]  = c.pos[0];    p[1]  = c.pos[1];    p[2]  = c.pos[2];
                p[3]  = c.normal[0]; p[4]  = c.normal[1]; p[5]  = c.normal[2];
                p[6]  = c.throughput_scalar;
                std::memcpy(&p[7],  &c.flags,      sizeof(uint32_t));
                std::memcpy(&p[8],  &c.subpath_id, sizeof(uint32_t));
                /* vinfo: bits[0..15]=packed_index, bits[16..30]=stream, bit[31]=tri_valid
                 * IMPORTANT: use 'ci' (the loop index = position in packed array), NOT
                 * c.vertex_index (which is bdpt_vertex = bounce count including optical
                 * lens traversals that emit no BDPT vertex record).  The GPU shader
                 * computes cam_flat_base = gid_c - ci, so ci must equal gid_c - base. */
                {
                    const uint32_t c_vinfo = (uint32_t)ci
                                           | ((uint32_t)c.stream << 16)
                                           | ((c.tri_id >= 0 ? 1u : 0u) << 31);
                    std::memcpy(&p[9], &c_vinfo, sizeof(uint32_t));
                }
                p[10] = 0.0f;   /* was: spectral beta R (GPU now reads from per-band slots) */
                p[11] = 0.0f;   /* was: spectral beta G */
                p[12] = 0.0f;   /* was: spectral beta B */
                p[13] = c.sensor_origin_y;
                p[14] = c.sensor_origin_z;
                /* pdf_fwd / pdf_rev [15..16] — direct from vertex record */
                p[15] = c.pdf_fwd;
                p[16] = c.pdf_rev;
                /* pdf_flags [17] — BdptPdfRecord::flags for delta-specular / diffuse / GGX bits */
                {
                    const BdptPdfRecord* pr17 = ctx.pdf_for(c.subpath_id, c.vertex_index);
                    const uint32_t pf = pr17 ? pr17->flags : 0u;
                    std::memcpy(&p[17], &pf, sizeof(uint32_t));
                }
                /* optical_block [18] — set if any blocking optical event (absorb/TIR/clip) */
                {
                    uint32_t oblock = 0u;
                    const uint64_t ok = ((uint64_t)c.subpath_id  << 32)
                                      | ((uint64_t)c.vertex_index << 16)
                                      | (uint64_t)c.stream;
                    auto oit = ctx.optical_lut.find(ok);
                    if (oit != ctx.optical_lut.end())
                        for (const auto& oe : oit->second)
                            if (oe.reason == BDPT_OPT_ABSORPTION   ||
                                oe.reason == BDPT_OPT_TIR          ||
                                oe.reason == BDPT_OPT_APERTURE_CLIP ||
                                oe.reason == BDPT_OPT_VIGNETTE_CLIP) { oblock = 1u; break; }
                    std::memcpy(&p[18], &oblock, sizeof(uint32_t));
                }
                /* prefix_pdf [19] — cumulative forward-PDF prefix */
                p[19] = (ci < csp->prefix_pdf.size() && csp->prefix_valid[ci])
                      ? static_cast<float>(csp->prefix_pdf[ci]) : 0.0f;
                /* mis_denom_sum [20] — reserved (computed live on GPU) */
                p[20] = 0.0f;
                /* tri_mat_idx [21] — int bits of material index */
                std::memcpy(&p[21], &c.mat_idx, sizeof(int32_t));
                /* per-band beta magnitudes [CGV_BAND_BASE .. CGV_BAND_BASE+T5_MAX_GPU_BANDS-1] */
                for (int b = 0; b < T5_MAX_GPU_BANDS; ++b) {
                    if (b >= ctx.n_bands) { p[CGV_BAND_BASE + b] = 0.0f; continue; }
                    const uint64_t k = ((uint64_t)c.subpath_id  << 32)
                                     | ((uint64_t)c.vertex_index << 16)
                                     | (uint64_t)b;
                    auto it = ctx.beta_lut.find(k);
                    p[CGV_BAND_BASE + b] = (it != ctx.beta_lut.end())
                        ? static_cast<float>(std::abs(it->second)) : 0.0f;
                }
                /* BRDF + edge PDF fields [CGV_DIR_IN_X .. CGV_EDGE_BWD_AREA] */
                p[CGV_DIR_IN_X + 0] = c.dir_in[0];
                p[CGV_DIR_IN_X + 1] = c.dir_in[1];
                p[CGV_DIR_IN_X + 2] = c.dir_in[2];
                p[CGV_DIFFUSE_P]     = get_diffuse_p(c.mat_idx);
                p[CGV_GGX_ALPHA]     = get_ggx_alpha(c.mat_idx);
                p[CGV_OPT_JACOBIAN]  = get_opt_jacobian(c);
                if (ci + 1u < csp->v.size()) {
                    const BdptVertexRecord& nxt = *csp->v[ci + 1u];
                    p[CGV_EDGE_FWD_AREA] = get_edge_fwd(c, nxt);
                    p[CGV_EDGE_BWD_AREA] = get_edge_bwd(c, nxt);
                }
                /* else: edge_fwd_area / edge_bwd_area stay 0.0f (last vert in subpath) */
                /* [46..55] remain 0.0f */
            }
        }

        std::vector<uint32_t> pixel_accum;  /* zeroed per-tile in service_t5_job */

        /* Pre-compute spectral colour weights (one RGB triple per band).
         * The GPU shader reads per-band beta magnitudes from CGV_BAND_BASE slots
         * and multiplies by these weights to get the display RGB contribution. */
        const int nb_gpu = std::max(1, std::min(ctx.n_bands, (int)T5_MAX_GPU_BANDS));
        std::vector<float> spectral_weights(static_cast<size_t>(nb_gpu) * 3);
        for (int b = 0; b < nb_gpu; ++b) {
            double wr, wg, wb;
            band_to_display_rgb(b, ctx.n_bands, ctx.t5_freq_hz, wr, wg, wb);
            spectral_weights[b * 3 + 0] = static_cast<float>(wr);
            spectral_weights[b * 3 + 1] = static_cast<float>(wg);
            spectral_weights[b * 3 + 2] = static_cast<float>(wb);
        }

        T5GpuParams par{};
        par.min_geom      = (ps->cfg.t5_min_geom > 0.0f) ? ps->cfg.t5_min_geom : 1e-8f;
        par.sensor_half_w = ps->sensor_half_w;
        par.sensor_half_h = ps->sensor_half_h;
        par.n_light_verts = n_lv;
        par.n_cam_verts   = ci_flat;
        par.sensor_res    = res;
        par.n_bands       = nb_gpu;

        fprintf(stderr, "[T5-gpu] submitting: n_cam=%u n_light=%u res=%d n_bands=%d sweights=%zu\n",
                ci_flat, n_lv, res, nb_gpu, sweights.size());
        fflush(stderr);

        gpu_pixel_result = ps->gpu_dispatch->submit_t5_connect(
            std::move(light_packed),
            std::move(cam_packed),
            std::move(pixel_accum),
            std::move(spectral_weights),
            par);
        t5_stats.used_gpu = true;  /* GPU dispatch was attempted — count against GPU counter */
    } else {
        /* CPU fallback: full brute-force via ThreadPool */
        fprintf(stderr, "[T5-conn] WARNING: GPU unavailable (use_gpu=%d dispatch=%p n_lv=%u cam=%zu) — running CPU fallback\n",
                (int)ps->cfg.use_gpu_compute, (void*)ps->gpu_dispatch,
                n_lv, cam_items.size());
        fflush(stderr);
        run_t5_allpairs(ctx, cam_items, light_items, accums, t5_pool);
    }

    uint64_t exact_total = 0;
    {
        std::lock_guard<std::mutex> lk(ps->sensor_mu);

        /* ── GPU pixel accum merge ───────────────────────────────────── *
         * The GPU shader packs R[res²] G[res²] B[res²] as uint32 bit-   *
         * casts of float.  Use neutral 1/3:1/3:1/3 weighting (same as  *
         * the shader's channel split) since spectral colourisation is   *
         * deferred to the CPU post-pass.                                */
        if (gpu_t5_ok && !gpu_pixel_result.empty()) {
            const size_t pix2 = static_cast<size_t>(res) * res;
            for (size_t i = 0; i < pix2; ++i) {
                float fr, fg, fb;
                uint32_t ur = gpu_pixel_result[i];
                uint32_t ug = gpu_pixel_result[pix2 + i];
                uint32_t ub = gpu_pixel_result[2*pix2 + i];
                std::memcpy(&fr, &ur, sizeof(float));
                std::memcpy(&fg, &ug, sizeof(float));
                std::memcpy(&fb, &ub, sizeof(float));
                if (fr + fg + fb <= 0.0f) continue;
                const double nr = (ps->sensor_accum[0 * pix2 + i] += (double)fr);
                const double ng = (ps->sensor_accum[1 * pix2 + i] += (double)fg);
                const double nb = (ps->sensor_accum[2 * pix2 + i] += (double)fb);
                if (nr > ps->sensor_peak[0]) ps->sensor_peak[0] = nr;
                if (ng > ps->sensor_peak[1]) ps->sensor_peak[1] = ng;
                if (nb > ps->sensor_peak[2]) ps->sensor_peak[2] = nb;
            }
        }

        for (size_t t = 0; t < nw; ++t) {
            exact_total += accums[t].exact_count;
            for (size_t i = 0; i < pix; ++i) {
                const double r = accums[t].sensor_r[i];
                const double g = accums[t].sensor_g[i];
                const double b = accums[t].sensor_b[i];
                if (r + g + b <= 0.0) continue;
                const double nr = (ps->sensor_accum[0 * pix + i] += r);
                const double ng = (ps->sensor_accum[1 * pix + i] += g);
                const double nb = (ps->sensor_accum[2 * pix + i] += b);
                if (nr > ps->sensor_peak[0]) ps->sensor_peak[0] = nr;
                if (ng > ps->sensor_peak[1]) ps->sensor_peak[1] = ng;
                if (nb > ps->sensor_peak[2]) ps->sensor_peak[2] = nb;
            }
        }
    }

    if (exact_total > 0)
        ps->bdpt_exact_snaps.fetch_add(exact_total, std::memory_order_relaxed);

    /* Keep the finite camera projection packet resident for the current
     * exposure.  Light-side records are consumed by this connection pass, but
     * subsequent light packets still need the same camera-side path family. */
    {
        std::unordered_set<uint32_t> camera_sids;
        camera_sids.reserve(cam_paths.size());
        for (const auto& kv : cam_paths)
            camera_sids.insert(kv.first);

        std::lock_guard<std::mutex> pending_lk(ps->bdpt_pending_mu);
        for (const auto& v : verts)
            if (v.stream == BDPT_SIDE_SENSOR)
                ps->bdpt_pending_vertices.push_back(v);
        for (const auto& sw : sweights)
            if (camera_sids.find(sw.subpath_id) != camera_sids.end())
                ps->bdpt_pending_spectral.push_back(sw);
        for (const auto& pr : pdfs)
            if (camera_sids.find(pr.subpath_id) != camera_sids.end())
                ps->bdpt_pending_pdfs.push_back(pr);
        for (const auto& oe : optical)
            if (camera_sids.find(oe.subpath_id) != camera_sids.end())
                ps->bdpt_pending_optical.push_back(oe);
    }
    /* Increment before the BdptConnectionRunGuard destructs so that
     * _fire_t5_if_ready cannot re-trigger before bdpt_connection_running
     * has been cleared. */
    fprintf(stderr, "[T5-conn] PASS COMPLETE  exact_snaps=%lu near_miss=%lu overflow_conn=%lu  t5_fired→%u\n",
            (unsigned long)ps->bdpt_exact_snaps.load(std::memory_order_relaxed),
            (unsigned long)ps->bdpt_near_miss_count.load(std::memory_order_relaxed),
            (unsigned long)ps->bdpt_overflow_connections.load(std::memory_order_relaxed),
            ps->bdpt_t5_fired.load(std::memory_order_relaxed) + 1u);
    fflush(stderr);
    ps->bdpt_t5_fired.fetch_add(1u, std::memory_order_release);
}

/* t5_kpn_worker — persistent Kahn Process Network node for the T5 connection
 * pass.  Blocks on Q_t5_ready (a single-sentinel FIFO) waiting for the GPU
 * tracing loop to signal that both ray families have fully terminated.  When
 * the sentinel arrives, all BDPT data is already in its queues; drain and
 * connect.  Exits when Q_t5_ready is set_done (pipeline destroy). */
static void t5_kpn_worker(RayPipelineState* ps)
{
    std::vector<int> tok;
    while (true) {
        tok.clear();
        int n = ps->Q_t5_ready.pop_batch(tok, 1);   /* BLOCKS until sentinel */
        if (n == 0) break;                           /* set_done → exit */
        fprintf(stderr, "[T5-kpn] sentinel received — running connection pass\n");
        fflush(stderr);
        ray_pipeline_run_bdpt_connection(ps);
    }
    fprintf(stderr, "[T5-kpn] worker exiting\n");
    fflush(stderr);
}

/* ray_pipeline_signal_flash_dispatched / ray_pipeline_signal_sensor_dispatched
 *
 * These are FIRST-SUBMISSION markers only.  They record that a new batch of
 * flash or sensor rays has been pushed into Q_intent for the current cycle.
 * They carry no information about tracing completeness and never trigger T5.
 *
 * T5 fires exclusively from the GPU dispatch loop's timeout path, when
 * in_flight == 0 (every ray in every bounce generation has terminated) AND
 * both flash and sensor have been submitted ahead of the last t5_fired count.
 * That is the only moment all subpath data is guaranteed to be in its queues. */

void ray_pipeline_signal_flash_dispatched(RayPipelineState* ps)
{
    if (!ps) return;
    const uint32_t new_flash = ps->bdpt_flash_dispatched.fetch_add(1u, std::memory_order_release) + 1u;
    fprintf(stderr, "[T5-latch] flash first-submission #%u  (sensor=%u t5_fired=%u) — tracing not started\n",
            new_flash,
            ps->bdpt_sensor_dispatched.load(std::memory_order_acquire),
            ps->bdpt_t5_fired.load(std::memory_order_acquire));
    fflush(stderr);
}

void ray_pipeline_signal_sensor_dispatched(RayPipelineState* ps)
{
    if (!ps) return;
    const uint32_t new_sensor = ps->bdpt_sensor_dispatched.fetch_add(1u, std::memory_order_release) + 1u;
    fprintf(stderr, "[T5-latch] sensor first-submission #%u  (flash=%u t5_fired=%u) — tracing not started\n",
            new_sensor,
            ps->bdpt_flash_dispatched.load(std::memory_order_acquire),
            ps->bdpt_t5_fired.load(std::memory_order_acquire));
    fflush(stderr);
}

void ray_pipeline_join_t5(RayPipelineState* ps)
{
    if (!ps) return;
    /* Wait until T5 has actually completed a pass for this dispatch cycle.
     *
     * The old implementation only waited for bdpt_connection_running==false,
     * which is also false BEFORE T5 starts.  There is a window where the
     * sentinel has been pushed to Q_t5_ready but t5_kpn_worker hasn't yet
     * called ray_pipeline_run_bdpt_connection (and set bdpt_connection_running).
     * In that window the old code returned immediately — before T5 ran at all.
     *
     * Correct wait: t5_fired must reach min(flash_dispatched, sensor_dispatched)
     * (the number of full cycles that should have completed) AND
     * bdpt_connection_running must be false (not mid-pass). */
    const uint32_t need = std::min(
        ps->bdpt_flash_dispatched.load(std::memory_order_acquire),
        ps->bdpt_sensor_dispatched.load(std::memory_order_acquire));
    if (need == 0) return;  /* no families submitted yet — nothing to join */
    for (;;) {
        /* If the GPU dispatch thread failed to start (make_current error),
         * rays will never be processed and T5 will never fire.  Bail out
         * immediately rather than hanging forever. */
        if (ps->gpu_dispatch_state.load(std::memory_order_acquire) == 4) {
            fprintf(stderr, "[join-t5] GPU thread failed (state=4) — returning without T5\n");
            fflush(stderr);
            return;
        }
        const uint32_t fired   = ps->bdpt_t5_fired.load(std::memory_order_acquire);
        const bool     running = ps->bdpt_connection_running.load(std::memory_order_acquire);
        if (fired >= need && !running) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int ray_pipeline_gpu_dispatch_state(const RayPipelineState* ps)
{
    if (!ps) return 0;
    return ps->gpu_dispatch_state.load(std::memory_order_acquire);
}

/* CIE-approximate spectral locus: band index → linear RGB weights.
 * Uses freq_hz_vec if available; otherwise linearly interpolates 380–700 nm.
 * Weights are in [0,1]; they do not sum to 1 (spectral colour is not white). */
static void band_to_display_rgb(int b, int n_bands,
                                 const Eigen::VectorXd& freq_hz_vec,
                                 double& wr, double& wg, double& wb)
{
    if (n_bands <= 1 || b < 0 || b >= n_bands) { wr = wg = wb = 1.0; return; }
    double wl_nm = 550.0;
    if (b < (int)freq_hz_vec.size() && freq_hz_vec[b] > 0.0) {
        constexpr double C = 299792458.0;
        wl_nm = std::max(380.0, std::min(700.0, (C / freq_hz_vec[b]) * 1.0e9));
    } else {
        const double t = static_cast<double>(b) / static_cast<double>(n_bands - 1);
        wl_nm = 380.0 + t * 320.0;
    }
    wr = wg = wb = 0.0;
    if      (wl_nm < 440.0) { wr = -(wl_nm-440.0)/60.0; wb = 1.0; }
    else if (wl_nm < 490.0) { wg =  (wl_nm-440.0)/50.0; wb = 1.0; }
    else if (wl_nm < 510.0) { wg = 1.0; wb = -(wl_nm-510.0)/20.0; }
    else if (wl_nm < 580.0) { wr =  (wl_nm-510.0)/70.0; wg = 1.0; }
    else if (wl_nm < 645.0) { wr = 1.0; wg = -(wl_nm-645.0)/65.0; }
    else                    { wr = 1.0; }
    double edge = 1.0;
    if      (wl_nm < 420.0) edge = 0.3 + 0.7*(wl_nm-380.0)/40.0;
    else if (wl_nm > 645.0) edge = 0.3 + 0.7*(700.0-wl_nm)/55.0;
    wr *= edge; wg *= edge; wb *= edge;
}

void ray_pipeline_set_t5_min_geom(RayPipelineState* ps, float v)
{
    if (ps) ps->cfg.t5_min_geom = v;
}

void ray_pipeline_set_t5_light_batch_size(RayPipelineState* ps, uint32_t n)
{
    if (!ps) return;
    ps->cfg.t5_light_batch_size = n;
    if (ps->gpu_dispatch) ps->gpu_dispatch->t5_light_batch_size = n;
}

void ray_pipeline_set_t5_cam_batch_size(RayPipelineState* ps, uint32_t n)
{
    if (!ps) return;
    ps->cfg.t5_cam_batch_size = n;
    if (ps->gpu_dispatch) ps->gpu_dispatch->t5_cam_batch_size = n;
}

void ray_pipeline_set_t5_sensor_tile_size(RayPipelineState* ps, uint32_t n)
{
    if (!ps) return;
    ps->cfg.t5_sensor_tile_size = n;
    if (ps->gpu_dispatch) ps->gpu_dispatch->t5_sensor_tile_size = n;
}

void ray_pipeline_begin_sensor_batching(RayPipelineState* ps)
{
    if (!ps) return;
    std::lock_guard<std::mutex> lk(ps->bdpt_stash_mu);
    ps->bdpt_batching_mode = true;
    ps->bdpt_stash_v.clear();
    ps->bdpt_stash_sw.clear();
    ps->bdpt_stash_pdf.clear();
    ps->bdpt_stash_opt.clear();
}

void ray_pipeline_end_sensor_batching(RayPipelineState* ps)
{
    if (!ps) return;
    std::lock_guard<std::mutex> lk(ps->bdpt_stash_mu);
    ps->bdpt_batching_mode = false;
    ps->bdpt_stash_v.clear();
    ps->bdpt_stash_sw.clear();
    ps->bdpt_stash_pdf.clear();
    ps->bdpt_stash_opt.clear();
}

void ray_pipeline_set_force_cpu_t5(RayPipelineState* /*ps*/, bool /*v*/) {}

void ray_pipeline_set_flash_modifier(RayPipelineState* ps, FlashModifierType type,
                                      float param0, float param1)
{
    if (!ps) return;
    ps->cfg.flash_modifier_type   = type;
    ps->cfg.flash_modifier_param0 = param0;
    ps->cfg.flash_modifier_param1 = param1;
}

uint64_t ray_pipeline_get_uv_pages_tex_id(const RayPipelineState* ps)
{
    if (!ps || !ps->gpu_dispatch) return 0;
    return const_cast<RayPipelineState::GlPipelineDispatch*>(ps->gpu_dispatch)->get_uv_pages_tex_id();
}

void ray_pipeline_set_uv_blit_weights(RayPipelineState* ps,
                                       const float* weights,
                                       int n_bands,
                                       int mode)
{
    if (!ps || !ps->gpu_dispatch || !weights || n_bands < 1) return;
    if (n_bands > MAX_SPECTRAL_BANDS) n_bands = MAX_SPECTRAL_BANDS;
    auto* gd = ps->gpu_dispatch;
    std::copy_n(weights, (size_t)n_bands * 3, gd->blit_rgb_weights.data());
    gd->blit_n_bands_stored = n_bands;
    gd->blit_mode = mode;
}

void ray_pipeline_report_display_frame_time(RayPipelineState* ps,
                                            double frame_ms,
                                            double target_ms)
{
    if (!ps || frame_ms <= 0.0) return;
    if (target_ms <= 0.0) target_ms = 16.667;
    const bool spiked = frame_ms > target_ms * 1.15;
    const bool stable = frame_ms < target_ms * 0.80;

    auto shrink_batch = [](StageStats& s) {
        int cur = s.batch_sz_gpu.load(std::memory_order_relaxed);
        /* Keep compute dispatches large enough to amortize GL sync/readback
         * overhead.  A slow interactive frame should not collapse ray tracing
         * into hundreds of 256-ray launches. */
        int next = std::max(16384, cur / 2);
        if (next != cur) s.batch_sz_gpu.store(next, std::memory_order_relaxed);
    };
    auto grow_batch = [](StageStats& s) {
        int cur = s.batch_sz_gpu.load(std::memory_order_relaxed);
        int next = std::min(1048576, cur + std::max(64, cur / 8));
        if (next != cur) s.batch_sz_gpu.store(next, std::memory_order_relaxed);
    };

    if (spiked) {
        for (StageStats& s : ps->stats) shrink_batch(s);
        if (ps->gpu_dispatch) {
            double cur = ps->gpu_dispatch->uv_readback_interval_s.load(std::memory_order_relaxed);
            ps->gpu_dispatch->uv_readback_interval_s.store(std::min(5.0, cur * 1.25),
                                                           std::memory_order_relaxed);
        }
    } else if (stable) {
        for (StageStats& s : ps->stats) grow_batch(s);
        if (ps->gpu_dispatch) {
            double cur = ps->gpu_dispatch->uv_readback_interval_s.load(std::memory_order_relaxed);
            ps->gpu_dispatch->uv_readback_interval_s.store(std::max(0.25, cur * 0.95),
                                                           std::memory_order_relaxed);
        }
    }
}

void ray_pipeline_set_shuffle(RayPipelineState* ps, float shuffle_frac)
{
    if (!ps) return;
    ps->cfg.intent_queue_shuffle = std::max(0.0f, std::min(1.0f, shuffle_frac));
}

void ray_pipeline_configure_sensor_image(
    RayPipelineState* ps,
    float plate_x, float plate_half_w, float plate_half_h,
    int   res,
    float bdpt_eps,
    float target_x,
    float target_r)
{
    if (!ps) return;
    std::lock_guard<std::mutex> lk(ps->sensor_mu);
    ps->sensor_res    = res;
    ps->sensor_px     = plate_x;
    ps->sensor_half_w = (plate_half_w > 0.0f) ? plate_half_w : 0.028f;
    ps->sensor_half_h = (plate_half_h > 0.0f) ? plate_half_h : 0.028f;
    ps->sensor_eps    = (bdpt_eps > 0.0f) ? bdpt_eps : 0.008f;
    ps->sensor_target_x = target_x;
    ps->sensor_target_r = target_r;
    /* 4 channels: ch0=R, ch1=G, ch2=B (spectral, all path types), ch3=near-miss */
    ps->sensor_accum.assign(static_cast<size_t>(res) * res * 4, 0.0);
    ps->priority_map.assign(static_cast<size_t>(res) * res, 1.0f);
    /* Reset running peaks so the new accumulator starts fresh. */
    for (int _c = 0; _c < 4; ++_c) ps->sensor_peak[_c] = 1e-30;
}

void ray_pipeline_get_sensor_image(
    const RayPipelineState* ps,
    float* buf,
    int*   out_res)
{
    if (out_res) *out_res = 0;
    if (!ps || ps->sensor_res <= 0) return;
    std::lock_guard<std::mutex> lk(ps->sensor_mu);
    const int res = ps->sensor_res;
    if (out_res) *out_res = res;
    if (!buf) return;
    const size_t pix = static_cast<size_t>(res) * res;
    /* Peaks are now tracked incrementally at each write to sensor_accum,
     * so no O(res²) scan is needed here. */
    const double* peak = ps->sensor_peak;
    const double inv_log10 = 1.0 / std::log(10.0);
    auto tone = [&](double v, double pk) -> float {
        double n = v / pk;
        return static_cast<float>(std::log1p(n * 9.0) * inv_log10);
    };
    /* Output layout: (res, res, 3) RGB, rows written bottom-first (y-flipped)
     * so the returned NumPy array is already in OpenGL texture order.
     * ch0=R, ch1=G, ch2=B — spectral accumulation via band_to_display_rgb;
     * all path types (forward, backward emissive, BDPT) contribute to each
     * channel weighted by their spectral composition. */
    for (int y = 0; y < res; ++y) {
        const int out_y = res - 1 - y;  /* flip for OpenGL bottom-to-top convention */
        for (int z = 0; z < res; ++z) {
            const size_t src_px = static_cast<size_t>(y * res + z);
            const size_t dst_px = static_cast<size_t>(out_y * res + z);
            buf[dst_px * 3 + 0] = tone(ps->sensor_accum[0 * pix + src_px], peak[0]);  /* R */
            buf[dst_px * 3 + 1] = tone(ps->sensor_accum[1 * pix + src_px], peak[1]);  /* G */
            buf[dst_px * 3 + 2] = tone(ps->sensor_accum[2 * pix + src_px], peak[2]);  /* B */
        }
    }
}

void ray_pipeline_get_priority_map(
    const RayPipelineState* ps,
    float* buf,
    int*   out_res)
{
    if (out_res) *out_res = 0;
    if (!ps || ps->sensor_res <= 0) return;
    std::lock_guard<std::mutex> lk(ps->sensor_mu);
    const int res = ps->sensor_res;
    if (out_res) *out_res = res;
    if (!buf || ps->priority_map.empty()) return;
    std::copy(ps->priority_map.begin(), ps->priority_map.end(), buf);
}

void ray_pipeline_get_bdpt_stats(
    const RayPipelineState* ps,
    float* out_nearest_dist_m,
    float* out_best_collinearity,
    uint64_t* out_exact_snaps,
    uint64_t* out_near_miss_count)
{
    if (!ps) {
        if (out_nearest_dist_m)    *out_nearest_dist_m    = -1.0f;
        if (out_best_collinearity) *out_best_collinearity = 0.0f;
        if (out_exact_snaps)       *out_exact_snaps       = 0;
        if (out_near_miss_count)   *out_near_miss_count   = 0;
        return;
    }
    if (out_nearest_dist_m) {
        uint64_t bits = ps->bdpt_nearest_d2_bits.load(std::memory_order_relaxed);
        if (bits == 0xFFFFFFFFFFFFFFFFULL) {
            *out_nearest_dist_m = -1.0f;
        } else {
            uint32_t b32 = static_cast<uint32_t>(bits & 0xFFFFFFFFULL);
            float d2; std::memcpy(&d2, &b32, sizeof(float));
            *out_nearest_dist_m = std::sqrt(std::max(0.0f, d2));
        }
    }
    if (out_best_collinearity) {
        uint64_t bits = ps->bdpt_best_colinear_bits.load(std::memory_order_relaxed);
        uint32_t b32 = static_cast<uint32_t>(bits & 0xFFFFFFFFULL);
        std::memcpy(out_best_collinearity, &b32, sizeof(float));
    }
    if (out_exact_snaps)     *out_exact_snaps     = ps->bdpt_exact_snaps.load(std::memory_order_relaxed);
    if (out_near_miss_count) *out_near_miss_count = ps->bdpt_near_miss_count.load(std::memory_order_relaxed);
}

void ray_pipeline_get_bdpt_latch_state(
    const RayPipelineState* ps,
    uint32_t* out_flash_dispatched,
    uint32_t* out_sensor_dispatched,
    uint32_t* out_t5_fired,
    int*      out_connection_running)
{
    if (!ps) {
        if (out_flash_dispatched)   *out_flash_dispatched   = 0;
        if (out_sensor_dispatched)  *out_sensor_dispatched  = 0;
        if (out_t5_fired)           *out_t5_fired           = 0;
        if (out_connection_running) *out_connection_running = 0;
        return;
    }
    if (out_flash_dispatched)   *out_flash_dispatched   = ps->bdpt_flash_dispatched.load(std::memory_order_relaxed);
    if (out_sensor_dispatched)  *out_sensor_dispatched  = ps->bdpt_sensor_dispatched.load(std::memory_order_relaxed);
    if (out_t5_fired)           *out_t5_fired           = ps->bdpt_t5_fired.load(std::memory_order_relaxed);
    if (out_connection_running) *out_connection_running = ps->bdpt_connection_running.load(std::memory_order_relaxed) ? 1 : 0;
}

/* ── Surrogate-emitter power update ──────────────────────────────────────── */

extern "C" SK_API int ray_tracer_set_tri_group_power(
    RayTracerState* st,
    int             group_id,
    const float*    power_W_per_band,
    int             n_bands)
{
    if (!st || !power_W_per_band) return SK_ERR_NULL_STATE;
    if (group_id < 0 || group_id >= static_cast<int>(st->tri_groups.size()))
        return SK_ERR_DIM_MISMATCH;
    /* Resize to match and overwrite. */
    auto& pv = st->tri_group_power_per_band[static_cast<size_t>(group_id)];
    pv.assign(power_W_per_band, power_W_per_band + n_bands);
    st->tri_groups[static_cast<size_t>(group_id)].n_power_bands = n_bands;
    st->tri_groups[static_cast<size_t>(group_id)].power_W_per_band = pv.data();
    return SK_OK;
}

/* ── BSSRDF illumination accumulator API ─────────────────────────────────── */

extern "C" SK_API int ray_tracer_init_illum_accum(RayTracerState* st)
{
    if (!st) return SK_ERR_NULL_STATE;
    const int nb     = st->n_bands;
    const int n_tris = static_cast<int>(st->tris.size());
    if (nb <= 0 || n_tris <= 0) return SK_OK;  /* no-op until scene is built */
    const int stride = 2 * nb + 2;
    st->tri_illum_nb = nb;
    st->tri_illum_accum.assign(static_cast<size_t>(n_tris * stride), 0u);
    return SK_OK;
}

extern "C" SK_API int ray_tracer_reset_illum_accum(RayTracerState* st)
{
    if (!st) return SK_ERR_NULL_STATE;
    std::fill(st->tri_illum_accum.begin(), st->tri_illum_accum.end(), 0u);
    return SK_OK;
}

extern "C" SK_API int ray_tracer_export_illum_accum(
    const RayTracerState* st,
    float*                buf,
    int                   buf_floats,
    int*                  out_n_tris,
    int*                  out_stride)
{
    if (!st) return SK_ERR_NULL_STATE;
    const int nb     = st->tri_illum_nb;
    const int stride = (nb > 0) ? (2 * nb + 2) : 0;
    const int n_tris = (stride > 0)
        ? static_cast<int>(st->tri_illum_accum.size() / (size_t)stride) : 0;
    if (out_n_tris) *out_n_tris = n_tris;
    if (out_stride) *out_stride = stride;
    if (!buf) return SK_OK;
    if (buf_floats < n_tris * stride) return SK_ERR_DIM_MISMATCH;
    for (int t = 0; t < n_tris; ++t) {
        float* row       = buf + t * stride;
        const int base   = t * stride;
        for (int b = 0; b < nb; ++b) {
            row[2*b + 0] = static_cast<float>(
                static_cast<double>(
                    static_cast<int32_t>(st->tri_illum_accum[(size_t)(base + 2*b)]))
                / 32768.0);
            row[2*b + 1] = static_cast<float>(
                static_cast<double>(
                    static_cast<int32_t>(st->tri_illum_accum[(size_t)(base + 2*b+1)]))
                / 32768.0);
        }
        row[2*nb + 0] = static_cast<float>(
            static_cast<double>(
                static_cast<int32_t>(st->tri_illum_accum[(size_t)(base + 2*nb)]))
            / 32768.0);
        row[2*nb + 1] = static_cast<float>(
            st->tri_illum_accum[(size_t)(base + 2*nb + 1)]);
    }
    return SK_OK;
}

extern "C" SK_API int ray_tracer_write_tri_illum(
    RayTracerState* st,
    const int*      tri_ids,
    int             n_tris,
    const float*    amp_re,
    const float*    amp_im,
    const float*    cos_avg,
    int             n_bands)
{
    if (!st || !tri_ids || !amp_re || !amp_im || !cos_avg || n_bands <= 0)
        return SK_ERR_NULL_STATE;
    if (st->tri_illum_accum.empty()) return SK_ERR_DIM_MISMATCH;
    const int nb     = st->tri_illum_nb;
    const int stride = 2 * nb + 2;
    const int nb_use = std::min(nb, n_bands);
    const int n_scene_tris = static_cast<int>(st->tris.size());
    /* Fixed-point encode: float → int32 × 32768, stored as uint32 bitcast.
     * Mirrors the existing uv_accum_add / tri_illum_accum encoding. */
    auto fp32 = [](float v) -> uint32_t {
        const int32_t x = static_cast<int32_t>(
            std::max(-1073741824.0f, std::min(1073741824.0f, v * 32768.0f)));
        return static_cast<uint32_t>(x);
    };
    for (int i = 0; i < n_tris; ++i) {
        const int t = tri_ids[i];
        if (t < 0 || t >= n_scene_tris) continue;
        const int base = t * stride;
        if (base + stride > static_cast<int>(st->tri_illum_accum.size())) continue;
        /* Zero existing Monte Carlo data for this triangle, then write BPM values.
         * Use relaxed atomic stores — we run between drain batches (Python side
         * guarantees the forward pipeline is idle at this call site). */
        for (int off = 0; off < stride; ++off)
            reinterpret_cast<std::atomic<uint32_t>&>(
                st->tri_illum_accum[static_cast<size_t>(base + off)])
                .store(0u, std::memory_order_relaxed);
        for (int b = 0; b < nb_use; ++b) {
            const int row_off = i * n_bands + b;
            reinterpret_cast<std::atomic<uint32_t>&>(
                st->tri_illum_accum[static_cast<size_t>(base + 2*b)])
                .store(fp32(amp_re[row_off]), std::memory_order_relaxed);
            reinterpret_cast<std::atomic<uint32_t>&>(
                st->tri_illum_accum[static_cast<size_t>(base + 2*b + 1)])
                .store(fp32(amp_im[row_off]), std::memory_order_relaxed);
        }
        reinterpret_cast<std::atomic<uint32_t>&>(
            st->tri_illum_accum[static_cast<size_t>(base + 2*nb)])
            .store(fp32(cos_avg[i]), std::memory_order_relaxed);
        /* count = 1: the BPM result is treated as a single synthetic forward hit */
        reinterpret_cast<std::atomic<uint32_t>&>(
            st->tri_illum_accum[static_cast<size_t>(base + 2*nb + 1)])
            .store(1u, std::memory_order_relaxed);
    }
    return SK_OK;
}

/* ── Synchronous compatibility wrapper ───────────────────────────────────── */
/* Callers that need blocking semantics: submit, poll until done, drain.
 * Uses a temporary pipeline so it does not interfere with a persistent one. */
int ray_pipeline_trace_sync(
    RayTracerState*          st,
    const RayPipelineConfig* cfg,
    const RayIntent*         intents,
    int                      n_intents,
    std::vector<RayRecord>&  out)
{
    if (!st || !intents || n_intents <= 0) return SK_ERR_NULL_STATE;
    RayPipelineState* ps = ray_pipeline_create(st, cfg);
    if (!ps) return SK_ERR_NULL_STATE;
    ray_pipeline_submit(ps, intents, n_intents);
    while (ray_pipeline_in_flight(ps) > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    ray_pipeline_drain(ps, out, n_intents * 256);
    ray_pipeline_destroy(ps);
    return SK_OK;
}

