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

#include <mutex>
#include <thread>
#include <cstring>
#include "ray_pipeline.h"
#include "gl_compute.h"

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
    MatSpectralCache            mat_cache; /* precomputed hot-path mat lookups */
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

/* Hot-path accessor — avoids mat_band_record bounds check + Fresnel math. */
static inline cd mat_cache_refl(const MatSpectralCache& c, int m, int b)
{
    return cd(c.refl_re[m * c.n_bands + b], c.refl_im[m * c.n_bands + b]);
}

static inline float mat_cache_diffusion(const MatSpectralCache& c, int m)
{
    return c.diffusion[m];
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
    V3d& pos, V3d& dir, VXcd& amp);

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
    int      sensor_group_id;
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
        if (E.vertex_index >= 0) {
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
        if (E.vertex_index >= 0) {
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
        /* Transfer to finer context */
        rs.pos[0] = p1.x(); rs.pos[1] = p1.y(); rs.pos[2] = p1.z();
        rs.context_id = enter_ci;
        for (int b = 0; b < n_bands; ++b) st.ray_amp_pool[amp_base + b] = amp[b];
        return enter_ci;
    }

    if (act == ACT_HIT_SURFACE) {
        V3d hit_n_geom = st.tris[static_cast<size_t>(hit_tri)].normal;
        V3d hit_pos = p1;
        apply_parametric_surface_point(st, hit_tri, hit_pos, hit_pos, hit_n_geom);
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
        for (int t : st->tri_group_indices.back()) {
            if (t >= 0 && t < (int)st->tri_param_group_of_tri.size()
                && st->tri_param_group_of_tri[(size_t)t] < 0)
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
    res.hit_tri         = -1;
    res.hit_tri_flags   = 0u;
    res.should_continue = false;
    res.is_sensor_hit   = false;
    res.is_emissive_hit = false;
    res.sensor_group_id = -1;
    
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
    
    /* ─ Material physics: transmission vs. reflection ─ */
    if (mat_transmissive) {
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
            interaction_flags |= dispatch_scale_context_entry(st, ctx, pos, dir, amp);
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
                    interaction_flags |= dispatch_scale_context_entry(*st, ctx, pos, dir, amp);
                }
            }

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                /* ─ Unified bounce physics (shared with backward path) ─ */
                BounceStepResult step = ray_bounce_step_bdpt(
                    *st, tri_sensor_group.data(), pos, dir, amp, path_len,
                    current_medium_mat_idx, interaction_flags, n_bands, min_amplitude, rng,
                    nullptr, false);
                
                if (step.hit_tri < 0) break;  /* Miss */
                
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
                
                /* ─ Forward-path-specific: sensor endpoint recording ─ */
                if (step.is_sensor_hit) {
                    V3d hit_n = st->tris[step.hit_tri].normal;
                    double cos_theta = std::abs(dir.dot(hit_n));
                    for (int b = 0; b < n_bands; ++b) {
                        if (rec_count >= out_cap) goto bdpt_done;
                        EndpointRecord& E = out_records[rec_count++];
                        E.subpath_id   = my_subpath;
                        E.band_id      = static_cast<uint32_t>(b);
                        E.group_id     = step.sensor_group_id;
                        E.vertex_index = -1;
                        E.pos[0] = (float)step.hit_pos.x();
                        E.pos[1] = (float)step.hit_pos.y();
                        E.pos[2] = (float)step.hit_pos.z();
                        E.pathlen_m = (float)path_len;
                        E.dir[0] = (float)dir.x();
                        E.dir[1] = (float)dir.y();
                        E.dir[2] = (float)dir.z();
                        E.pdf       = 1.0f / (float)std::max(1, n_rays_this);
                        E.amp_re    = (float)amp[b].real();
                        E.amp_im    = (float)amp[b].imag();
                        E.cos_theta = (float)cos_theta;
                        E.stream_id = (float)BDPT_SIDE_LIGHT;
                    }
                }
                
                /* ─ Check if ray should continue bouncing ─ */
                if (!step.should_continue) break;
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
 * NEURAL_SURFACE (5), NEURAL_VOLUMETRIC (6) are stub-passthroughs: they
 * record region entry in EndpointRecord.flags downstream and continue.
 * Wave + neural fill-in lives in field_march.cpp / future neural_eval.cpp.
 * ────────────────────────────────────────────────────────────────────── */
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
    V3d& pos, V3d& dir, VXcd& amp)
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
        case SCALE_CONTEXT_KIND_NEURAL_SURFACE:
        case SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC:
            return 1u << ctx.context_kind;
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

    std::atomic<int>          in_flight{0};

    /* Pre-flagged materials: flag[m]=1 means max reflectance < cfg.min_amplitude,
     * so any opaque ray hitting material m will never produce a viable child.
     * T3 fast-path skips direction sampling + reflectance multiply for these. */
    std::vector<uint8_t>      mat_epsilon_flags;

    StageStats                stats[4]; /* [0]=T1 [1]=T2 [2]=T3 [3]=T4        */
    std::atomic<uint64_t>     gpu_uv_readback_bytes{0};
    std::atomic<uint64_t>     gpu_uv_readback_count{0};
    std::atomic<uint64_t>     gpu_hit_readback_bytes{0};
    std::atomic<uint64_t>     gpu_hit_readback_count{0};

    std::deque<WaveArena>     arenas;   /* deque: never moves elements, safe with mutex */
    std::vector<std::thread>  workers;

    /* ── Sensor image accumulator ─────────────────────────────────────────
     * Three-channel float64 buffer, res×res, layout [ch][iy][iz].
     * ch0 = forward plate hits; ch1 = backward emissive paths; ch2 = BDPT snap.
     * Protected by sensor_mu; read by ray_pipeline_get_sensor_image(). */
    mutable std::mutex        sensor_mu;
    int                       sensor_res  = 0;   /* 0 = disabled */
    float                     sensor_px   = 0.0f;
    float                     sensor_pr   = 0.0f;
    float                     sensor_eps  = 0.008f;
    std::vector<double>       sensor_accum;       /* res*res*4 doubles: ch0=fwd, ch1=bwd-emis, ch2=exact-BDPT, ch3=near-miss */
    std::vector<float>        priority_map;       /* res*res sugar-auxin field (1.0 = baseline) */
    mutable double            sensor_peak[4]      = {1e-30, 1e-30, 1e-30, 1e-30}; /* running per-channel max, never decreases */

    /* ── GPU compute dispatch (nullptr = CPU-only) ─────────────────────────── */
    class GlPipelineDispatch;                       /* forward-declared below    */
    GlPipelineDispatch*       gpu_dispatch = nullptr;

    /* ── Live BDPT diagnostic stats (updated by T3, lock-free) ──────────
     * nearest_d2: running min of squared YZ distance over all fwd/rev pairs
     * best_collinearity: cos(angle between fwd and rev directions) of best pair
     * exact_snaps / near_miss_count: cumulative hit counts */
    std::atomic<uint64_t>     bdpt_nearest_d2_bits{0xFFFFFFFFFFFFFFFFULL}; /* float bits via reinterpret */
    std::atomic<uint64_t>     bdpt_best_colinear_bits{0};
    std::atomic<uint64_t>     bdpt_exact_snaps{0};
    std::atomic<uint64_t>     bdpt_near_miss_count{0};
};

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

class RayPipelineState::GlPipelineDispatch {
public:
    GlComputeContext ctx{};
    bool             ready = false;

    /* Shader programs for each stage */
    GLuint prog_t1 = 0, prog_t2 = 0, prog_t3 = 0, prog_t4 = 0;

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
    struct { GLint n_tris, n_groups; }
        uloc_t2 = {-1,-1};
    struct { GLint n_bands, n_mats, max_children, max_children_per_hit,
                   sensor_res, sensor_pr, sensor_px,
                   n_uv_groups, uv_meta_base, rng_seed; }
        uloc_t3 = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
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
    /* T3 output: child intents at [0..max_children*INTENT_STRIDE) and
     * terminal records at [max_children*INTENT_STRIDE..) packed in one SSBO */
    GLuint ssbo_child_int   = 0;
    /* T3 meta: [0]=intent_count [1]=terminal_count [2..2+n_mats-1]=eps_flags
     * Only [0] and [1] are zeroed before each dispatch; eps_flags are written
     * at scene-upload time and persist across dispatches. */
    GLuint ssbo_t3_meta     = 0;
    /* UV integrator image: per-tri UV data + per-tri group mapping,
     * per-group metadata, and the flat atomic accumulator. */
    GLuint ssbo_tri_uv          = 0;  /* binding 5: n_tris*6 float32 UV vertex coords         */
    GLuint ssbo_tri_uv_and_meta = 0;  /* binding 6: [n_tris group-ids] ++ [n_groups*2 meta]   */
    GLuint ssbo_uv_accum        = 0;  /* binding 7: flat uint32 accumulator                   */
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
    std::vector<float>     stg_ibuf;     /* intent upload (n*INTENT_STRIDE)             */
    std::vector<float>     stg_cbuf;     /* child intent readback (nc*CHILD_STRIDE)     */
    std::vector<float>     stg_tbuf;     /* terminal readback (nt_term*TERMINAL_STRIDE) */
    std::vector<float>     stg_hbuf;     /* hit readback (n_hits*HIT_STRIDE)            */
    std::vector<RayRecord> stg_strike;   /* STRIKE records for Q_out                   */
    std::vector<RayIntent> stg_children; /* child intents for push_many                */
    std::vector<uint32_t>  stg_uv_acc;  /* UV accumulator readback                    */
    std::vector<VXcd>      stg_amp_recycle; /* Eigen amp allocs harvested from prior batch */
    struct SensorUpdate { int iy, iz; double ch1, ch2; };
    std::vector<SensorUpdate> stg_sensor_updates; /* precomputed sensor updates, outside lock */
    std::chrono::steady_clock::time_point last_uv_readback_t = std::chrono::steady_clock::now();
    std::atomic<double> uv_readback_interval_s{1.0};

    /* Scene data uploaded once */
    bool scene_uploaded = false;

    /* Number of parametric groups uploaded to GPU — 0 means T2 is a no-op */
    int n_param_groups = 0;

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

        /* Apply pinned GPU fractions (if non-zero in config) */
        if (cfg.gpu_fraction_t1 > 0.0f) ps.stats[0].set_gpu_fraction(cfg.gpu_fraction_t1);
        if (cfg.gpu_fraction_t2 > 0.0f) ps.stats[1].set_gpu_fraction(cfg.gpu_fraction_t2);
        if (cfg.gpu_fraction_t3 > 0.0f) ps.stats[2].set_gpu_fraction(cfg.gpu_fraction_t3);

        /* Compile all four compute shaders */
        char err[1024];
        auto load = [&](const std::string& name, GLuint& prog) -> bool {
            std::string path = resolve_shader(shader_dir, name);
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) { snprintf(err, sizeof(err), "Cannot open %s", path.c_str()); return false; }
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            std::string src(static_cast<size_t>(sz), '\0');
            fread(&src[0], 1, static_cast<size_t>(sz), f); fclose(f);
            prog = gl_compute_build_program(src.c_str(), err, sizeof(err));
            return prog != 0;
        };

        if (!load("ray_bvh_intersect.comp.glsl", prog_t1)) { ctx.error[0] = '\0'; strncpy(ctx.error, err, sizeof(ctx.error)-1); return false; }
        if (!load("ray_refine.comp.glsl",        prog_t2)) { ctx.error[0] = '\0'; strncpy(ctx.error, err, sizeof(ctx.error)-1); return false; }
        if (!load("ray_material.comp.glsl",      prog_t3)) { ctx.error[0] = '\0'; strncpy(ctx.error, err, sizeof(ctx.error)-1); return false; }
        if (!load("ray_wave_bpm.comp.glsl",      prog_t4)) { ctx.error[0] = '\0'; strncpy(ctx.error, err, sizeof(ctx.error)-1); return false; }

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

        uloc_t3.n_bands              = glc_GetUniformLocation(prog_t3, "n_bands");
        uloc_t3.n_mats               = glc_GetUniformLocation(prog_t3, "n_mats");
        uloc_t3.max_children         = glc_GetUniformLocation(prog_t3, "max_children");
        uloc_t3.max_children_per_hit = glc_GetUniformLocation(prog_t3, "max_children_per_hit");
        uloc_t3.sensor_res           = glc_GetUniformLocation(prog_t3, "sensor_res");
        uloc_t3.sensor_pr            = glc_GetUniformLocation(prog_t3, "sensor_pr");
        uloc_t3.sensor_px            = glc_GetUniformLocation(prog_t3, "sensor_px");
        uloc_t3.n_uv_groups          = glc_GetUniformLocation(prog_t3, "n_uv_groups");
        uloc_t3.uv_meta_base         = glc_GetUniformLocation(prog_t3, "uv_meta_base");
        uloc_t3.rng_seed             = glc_GetUniformLocation(prog_t3, "rng_seed");

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

        /* Parametric group data — tri→group map and per-group kind+payload.
         * CPU stores payloads as double arrays; GPU T2 shader uses float[16]. */
        {
            const int nt = (int)st.tris.size();
            const int ng = (int)st.tri_group_parametric_kind.size();

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
                static constexpr int GPS = 16; /* GROUP_PAYLOAD_STRIDE */
                std::vector<float> pay(static_cast<size_t>(ng) * GPS, 0.0f);
                for (int gi = 0; gi < ng; ++gi) {
                    int kind = st.tri_group_parametric_kind[static_cast<size_t>(gi)];
                    const auto& pb = st.tri_group_parametric_payload[static_cast<size_t>(gi)];
                    float* dst = pay.data() + gi * GPS;
                    if (kind == TRI_PARAM_SURFACE_POLY_BARY
                            && pb.size() >= 6 * sizeof(double)) {
                        const double* src = reinterpret_cast<const double*>(pb.data());
                        for (int i = 0; i < 6; ++i) dst[i] = (float)src[i];
                    } else if (kind == TRI_PARAM_SURFACE_SDF_SPHERE
                            && pb.size() >= sizeof(double)) {
                        dst[0] = (float)(*reinterpret_cast<const double*>(pb.data()));
                    }
                    /* SDF_SADDLE (kind=2): GPU falls back to CPU via counter[3]; no payload needed */
                }
                upload_ssbo(ssbo_group_pay, pay.data(),
                            (GLsizeiptr)(ng * GPS * sizeof(float)));
                n_param_groups = ng;
            } else {
                int stub_i = 0;    upload_ssbo(ssbo_group_kind, &stub_i, sizeof(int));
                float stub_f = 0.f; upload_ssbo(ssbo_group_pay, &stub_f, sizeof(float));
                n_param_groups = 0;
            }
        }

        /* Counter SSBO: 8 uints — allocate once, zeroed on each dispatch */
        ensure_ssbo(ssbo_counter, 8 * sizeof(uint32_t));

        /* T3 meta SSBO: [0]=intent_count [1]=terminal_count [2..2+nm-1]=eps_flags
         * Allocate as 2+nm uints.  Counters are zeroed before each dispatch;
         * eps_flags are written here and survive across dispatches. */
        {
            const int nm2 = std::max(1, st.mat_n_mats);  /* at least 1 so the SSBO is non-empty */
            std::vector<uint32_t> meta(2 + nm2, 0u);
            const auto& ef = ps.mat_epsilon_flags;
            for (int i = 0; i < nm2 && i < (int)ef.size(); ++i)
                meta[2 + i] = ef[i] ? 1u : 0u;
            upload_ssbo(ssbo_t3_meta, meta.data(),
                        (GLsizeiptr)((2 + nm2) * sizeof(uint32_t)));
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
         * These are uploaded once at scene-upload time; ssbo_uv_accum is the
         * only one that changes (GPU atomicAdds during T3) and is readback +
         * zeroed after every T3 dispatch.                                     */
        {
            const int nt_uv = (int)st.tri_uv_data.size() / 6;
            if (nt_uv > 0 && !st.tri_uv_group_of_tri.empty()) {
                upload_ssbo(ssbo_tri_uv, st.tri_uv_data.data(),
                            (GLsizeiptr)(st.tri_uv_data.size() * sizeof(float)));
            } else {
                float stub_f = 0.0f; upload_ssbo(ssbo_tri_uv, &stub_f, sizeof(float));
            }

            /* Merged buffer at binding 6: [n_tris group-ids] ++ [n_groups*2 meta ints].
             * Build it unconditionally so the binding is always valid. */
            {
                const size_t n_id = st.tri_uv_group_of_tri.size();
                const int    ng   = (int)st.group_uv_res.size();
                std::vector<int32_t> merged(std::max((size_t)1, n_id + (size_t)ng * 2), -1);
                for (size_t i = 0; i < n_id; ++i)
                    merged[i] = st.tri_uv_group_of_tri[i];
                for (int g = 0; g < ng; ++g) {
                    merged[n_id + (size_t)g * 2 + 0] = st.group_uv_res[(size_t)g];
                    merged[n_id + (size_t)g * 2 + 1] = st.group_uv_accum_offset[(size_t)g];
                }
                upload_ssbo(ssbo_tri_uv_and_meta, merged.data(),
                            (GLsizeiptr)(merged.size() * sizeof(int32_t)));
            }

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
        const int uv_meta_base = (int)st.tri_uv_group_of_tri.size();
        const int nb           = (blit_n_bands_stored > 0) ? blit_n_bands_stored : st.n_bands;
        const int res          = blit_res;
        const GLuint tex       = tex_uv_pages[(size_t)slot];
        if (!tex) return;

        /* Ensure T3 image stores are visible before we read the SSBO. */
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        /* Bind resources */
        glc_BindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo_uv_accum);
        glc_BindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_tri_uv_and_meta);
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
        glc_DispatchCompute(gx, gy, gz);

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
                           const std::vector<RayIntent>& batch)
    {
        using Clock = std::chrono::high_resolution_clock;
        const RayTracerState& st = *ps.st;
        const int n  = (int)batch.size();
        const int nb = st.n_bands;
        const int nm = st.mat_n_mats;
        const int na = (int)ps.arenas.size();
        const int nt = (int)st.tris.size();

        /* ── Ensure intent SSBO is large enough (INTENT_STRIDE = 52 floats) */
        static constexpr int INTENT_STRIDE = 52;
        if (n > cap_intents) {
            cap_intents = n * 2;
            ensure_ssbo(ssbo_intent, (GLsizeiptr)(cap_intents * INTENT_STRIDE * sizeof(float)));
        }

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
            row[17] = ri.sensor_origin_y; row[18] = ri.sensor_origin_z; row[19] = 0.0f;
            const int bands = std::min(nb, 16);
            for (int b = 0; b < bands; ++b) {
                row[20+b] = (float)ri.amp[b].real();
                row[36+b] = (float)ri.amp[b].imag();
            }
        }
        /* Upload intents */
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_intent);
        glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(n * INTENT_STRIDE * sizeof(float)), ibuf.data());
        glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        /* ── Ensure hit SSBO (HIT_STRIDE = 58 floats) */
        static constexpr int HIT_STRIDE = 58;
        int max_hits = n * 4;
        if (max_hits > cap_hits) {
            cap_hits = max_hits * 2;
            ensure_ssbo(ssbo_hit, (GLsizeiptr)(cap_hits * HIT_STRIDE * sizeof(float)));
        }

        /* Zero counter SSBO */
        {
            uint32_t zeros[8] = {};
            glc_BindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_counter);
            glc_BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8 * sizeof(uint32_t), zeros);
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
        glc_DispatchCompute((GLuint)((n + 63) / 64), 1, 1);
        /* Shader-storage barrier only — no CPU readback here.
         * T2 and T3 read n_hits from counters[0] in the SSBO so we can
         * dispatch T1→T2→T3 as a single GPU command sequence with only
         * one CPU sync point (after T3).  This halves GPU↔CPU round-trips. */
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        /* ── T2 dispatch (in-place on ssbo_hit) ─────────────────────────═
         * Dispatched on the input batch size (worst case = all hits).
         * Invocations beyond the actual hit count early-return via the
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
            /* n_hits uniform removed — shader reads counters[0] from SSBO */
            glc_Uniform1i(uloc_t2.n_tris,   nt);
            glc_Uniform1i(uloc_t2.n_groups, n_param_groups);
            glc_DispatchCompute((GLuint)((n + 63) / 64), 1, 1);
            glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        /* ── T3 dispatch ──────────────────────────────────────────────═ */
        static constexpr int CHILD_STRIDE    = 52;  /* INTENT_STRIDE: 20 + 2*16 */
        static constexpr int TERMINAL_STRIDE = 58;  /* 26 + 2*16 (MAX_BANDS=16) */
        /* Size child buffer for worst-case (all n intents hit and spawn children).
         * Actual n_hits is unknown until after T3; sizing by n is always sufficient
         * since n_hits <= n. */
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
        /* Bindings match ray_material.comp.glsl:
         * 0=RefinedHitBuf  1=OutBuf(intents+terminals)  2=MetaBuf
         * 3=TriangleBuf    4=MatBandBuf  5=TriUvBuf  6=TriUvAndMetaBuf
         * 7=UvAccumBuf     8=CounterBuf (T1 hit count for early-exit guard) */
        bind_ssbo(ssbo_hit,             0);
        bind_ssbo(ssbo_child_int,       1);
        bind_ssbo(ssbo_t3_meta,         2);
        bind_ssbo(ssbo_tri_full,        3);
        bind_ssbo(ssbo_mat_band,        4);
        bind_ssbo(ssbo_tri_uv,          5);
        bind_ssbo(ssbo_tri_uv_and_meta, 6);
        bind_ssbo(ssbo_uv_accum,        7);
        bind_ssbo(ssbo_counter,         8);  /* T1 counters — read by shader for n_hits */
        const int n_uv_groups  = (int)st.group_uv_res.size();
        const int uv_meta_base = (int)st.tri_uv_group_of_tri.size();
        /* n_hits uniform removed — shader reads t1_counters[0] from binding 8 */
        glc_Uniform1i (uloc_t3.n_bands,              nb);
        glc_Uniform1i (uloc_t3.n_mats,               nm);
        glc_Uniform1i (uloc_t3.max_children,         max_children);
        glc_Uniform1i (uloc_t3.max_children_per_hit, ps.cfg.max_children);
        glc_Uniform1i (uloc_t3.sensor_res,            0);
        glc_Uniform1f (uloc_t3.sensor_pr,             ps.sensor_pr);
        glc_Uniform1f (uloc_t3.sensor_px,             ps.sensor_px);
        glc_Uniform1i (uloc_t3.n_uv_groups,           n_uv_groups);
        glc_Uniform1i (uloc_t3.uv_meta_base,          uv_meta_base);
        glc_Uniform1ui(uloc_t3.rng_seed,              ++t3_rng_seed);
        glc_DispatchCompute((GLuint)((n + 63) / 64), 1, 1);

        /* Single full barrier + readback covers ALL of T1/T2/T3 output in one
         * GPU→CPU sync.  GL_ALL_BARRIER_BITS ensures CPU-side GetBufferSubData
         * visibility of the atomic writes from all three stages. */
        glc_MemoryBarrier(GL_ALL_BARRIER_BITS);

        /* Read counters and meta in one pass — the GPU is done after the barrier. */
        uint32_t counters[8] = {};
        readback_ssbo(ssbo_counter, counters, 8 * sizeof(uint32_t));
        int n_hits = std::min((int)counters[0], cap_hits);

        ps.stats[0].record_gpu(n, (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - t1_start).count(), ps.Q_intent.size());
        ps.stats[1].record_gpu(n_hits, 0, n_hits);
        ps.stats[2].record_gpu(n_hits, (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - t3_start).count(), n_hits);

        if (n_hits <= 0) return 0;

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

        /* Re-submit child intents */
        if (nc > 0) {
            stg_cbuf.resize((size_t)nc * CHILD_STRIDE);
            auto& cbuf = stg_cbuf;
            readback_ssbo(ssbo_child_int, cbuf.data(), (GLsizeiptr)(nc * CHILD_STRIDE * sizeof(float)));
            stg_children.clear();
            stg_children.reserve((size_t)nc);
            auto& children = stg_children;
            int amp_pool_idx = 0;
            for (int ci = 0; ci < nc; ++ci) {
                const float* row = cbuf.data() + ci * CHILD_STRIDE;
                RayIntent ri{};
                /* Reuse a pre-allocated Eigen amp from the recycling pool when
                 * available — avoids a heap alloc per child intent after warmup. */
                if (amp_pool_idx < (int)stg_amp_recycle.size()) {
                    ri.amp = std::move(stg_amp_recycle[amp_pool_idx++]);
                    if ((int)ri.amp.size() != nb) ri.amp.resize(nb);
                } else {
                    ri.amp.resize(nb);
                }
                ri.pos = V3d(row[0], row[1], row[2]);
                ri.dir = V3d(row[3], row[4], row[5]);
                ri.path_len = (double)row[6];
                memcpy(&ri.medium_mat_idx,    row+7,  4);
                memcpy(&ri.interaction_flags, row+8,  4);
                memcpy(&ri.src_id,            row+9,  4);
                memcpy(&ri.bounce,            row+10, 4);
                memcpy(&ri.bounces_left,      row+11, 4);
                ri.min_amplitude = (double)row[12];
                uint32_t tlo, thi; memcpy(&tlo, row+13, 4); memcpy(&thi, row+14, 4);
                ri.tag = (uint64_t)tlo | ((uint64_t)thi << 32);
                uint32_t cflag = 0u;
                memcpy(&cflag, row+15, 4);
                ri.color_flag = (uint8_t)cflag;
                ri.priority = row[16];
                ri.sensor_origin_y = row[17]; ri.sensor_origin_z = row[18];
                const int bands = std::min(nb, 16);
                for (int b = 0; b < bands; ++b)
                    ri.amp[b] = std::complex<double>(row[20+b], row[36+b]);
                for (int b = bands; b < nb; ++b)
                    ri.amp[b] = std::complex<double>(0.0, 0.0);
                children.push_back(std::move(ri));
            }
            ps.in_flight.fetch_add((int)children.size(), std::memory_order_relaxed);
            ps.Q_intent.push_many(children);
        }
        if (t3_post_diag) {
            fprintf(stderr, "[gpu-T3-post] child requeue done\n");
            fflush(stderr);
        }

        /* Accumulate emissive terminal hits onto the CPU sensor image.
         * Terminal record layout (TERMINAL_STRIDE=58 floats, MAX_BANDS=16):
         *   [16]=color_flag  [23]=sensor_origin_y  [24]=sensor_origin_z
         *   [25]=is_emissive_hit  [26..41]=amp_re  [42..57]=amp_im        */
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
            const float  inv_r     = (float)res / (2.0f * ps.sensor_pr);
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
                const int iy = (int)((soy + ps.sensor_pr) * inv_r);
                const int iz = (int)((soz + ps.sensor_pr) * inv_r);
                if (iy < 0 || iy >= res || iz < 0 || iz >= res) continue;
                double amp_mag = 0.0;
                for (int b = 0; b < nb && b < 16; ++b) {
                    const double re = (double)rec[26 + b], im = (double)rec[42 + b];
                    amp_mag += std::sqrt(re*re + im*im);
                }
                sensor_updates.push_back({iy, iz, 1.0, amp_mag});
            }
            /* Apply all updates under a short-held lock, tracking peaks
             * incrementally so get_sensor_image() skips the O(res²) scan. */
            if (!sensor_updates.empty()) {
                std::lock_guard<std::mutex> lk(ps.sensor_mu);
                for (const auto& u : sensor_updates) {
                    const double v1 = (ps.sensor_accum[(size_t)(1*res*res + u.iy*res + u.iz)] += u.ch1);
                    const double v2 = (ps.sensor_accum[(size_t)(2*res*res + u.iy*res + u.iz)] += u.ch2);
                    if (v1 > ps.sensor_peak[1]) ps.sensor_peak[1] = v1;
                    if (v2 > ps.sensor_peak[2]) ps.sensor_peak[2] = v2;
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
            static constexpr int Q_OUT_VIS_CAP = 65536;
            /* Skip the hit readback entirely when Q_out is already full and field
             * capture is disabled: the STRIKE records would be discarded anyway,
             * and this avoids a potentially large (10-15 MB) PCIe transfer.   */
            const int q_space = Q_OUT_VIS_CAP - (int)ps.Q_out.size();
            if (q_space > 0 || do_field) {
                /* Only readback the records we'll actually use.  When field
                 * capture is disabled, limit to n_vis (the cap we'll push to
                 * Q_out), saving PCIe bandwidth on excess hits.              */
                const int n_vis = std::max(0, std::min(n_hits, q_space));
                const int n_rb  = do_field ? n_hits : n_vis;
                stg_hbuf.resize((size_t)n_rb * HIT_STRIDE);
                auto& hbuf = stg_hbuf;
                readback_ssbo(ssbo_hit, hbuf.data(),
                              (GLsizeiptr)((size_t)n_rb * HIT_STRIDE * sizeof(float)));
                ps.gpu_hit_readback_bytes.fetch_add(
                    (uint64_t)((size_t)n_rb * HIT_STRIDE * sizeof(float)),
                    std::memory_order_relaxed);
                ps.gpu_hit_readback_count.fetch_add(1, std::memory_order_relaxed);
                const int bands = std::min(nb, 16);
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
                        memcpy(&rec.mat_idx, row + 15, 4);
                        uint32_t cflag; memcpy(&cflag, row + 16, 4);
                        rec.color_flag = (uint8_t)cflag;
                        rec.n_bands = bands;
                        for (int b = 0; b < bands; ++b) {
                            rec.amp_re[b] = row[26 + b];
                            rec.amp_im[b] = row[42 + b];
                        }
                        strike_batch.push_back(std::move(rec));
                    }

                    /* Field grid contribution using T2-refined segment endpoints */
                    if (do_field && !field_done_bulk) {
                        const V3d seg_s(row[9], row[10], row[11]);
                        const V3d hit_p(row[0], row[1],  row[2]);
                        for (int b = 0; b < bands; ++b)
                            amp_tmp[b] = std::complex<double>(row[26 + b], row[42 + b]);
                        for (int b = bands; b < nb; ++b)
                            amp_tmp[b] = cd(0.0, 0.0);
                        accumulate_field_capture_segment(*ps.st, seg_s, hit_p, amp_tmp);
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

        return n_hits;
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
            float wl[16] = {};
            for (int b = 0; b < std::min(nb,16); ++b)
                wl[b] = (float)arena.wavelengths_m[b];
            glc_Uniform1fv(uloc_t4.wavelengths, 16, wl);
        };

        bind_ssbo(ssbo_wave_re, 0); bind_ssbo(ssbo_wave_im, 1);
        bind_ssbo(ssbo_bpm_tmp_re, 2); bind_ssbo(ssbo_bpm_tmp_im, 3);
        bind_ssbo(ssbo_thomas_cp, 4); bind_ssbo(ssbo_thomas_dp, 5);

        /* Mode 0: carrier advance */
        set_bpm_uniforms(0);
        glc_DispatchCompute((GLuint)n_pix, 1, 1);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        /* Mode 1: horizontal sweep */
        set_bpm_uniforms(1);
        glc_DispatchCompute((GLuint)(nb * ny), 1, 1);
        glc_MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        /* Mode 2: vertical sweep */
        set_bpm_uniforms(2);
        glc_DispatchCompute((GLuint)(nb * nx), 1, 1);
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
            fflush(stderr);
            return;
        }
        if (!scene_uploaded) upload_scene_data(ps);
        fprintf(stderr, "[gpu-dispatch] thread: scene uploaded, entering dispatch loop\n");
        fflush(stderr);

        std::vector<RayIntent>  batch;
        std::vector<WaveIntent> wave_batch;

        while (true) {
            /* Block on Q_intent — the primary work source.  Returns 0 only
             * when set_done() has been called AND the queue is empty.      */
            batch.clear();
            int gpu_bsz = ps.stats[0].batch_sz_gpu.load(std::memory_order_relaxed);
            int n = ps.Q_intent.pop_batch(batch, gpu_bsz);

            /* Non-blocking drain of Q_wave — never stall here; the CPU T4
             * worker is absent when use_gpu_compute=true so we opportunistically
             * pick up any wave intents that have accumulated.              */
            wave_batch.clear();
            ps.Q_wave.drain(wave_batch, gpu_bsz);

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
                break;
            }

            /* T1→T2→T3 batch */
            {
                int n_hits = dispatch_t1_t2_t3(ps, batch);
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
                (void)n_hits;
            }

            /* T4 wave step per arena */
            for (auto& wi : wave_batch) {
                if (wi.arena_id >= 0 && wi.arena_id < (int)ps.arenas.size())
                    dispatch_t4_step(ps, ps.arenas[static_cast<size_t>(wi.arena_id)]);
                ps.in_flight.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
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
    return cr;
}

/* ── In-flight accounting ─────────────────────────────────────────────────── */

static void pipeline_finish_ray(RayPipelineState& ps)
{
    ps.in_flight.fetch_sub(1, std::memory_order_acq_rel);
}

static void pipeline_spawn_child(RayPipelineState& ps, RayIntent child)
{
    ++ps.in_flight;
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
                    const float inv_r = static_cast<float>(ps.sensor_res) / (2.0f * ps.sensor_pr);
                    const int   iy    = static_cast<int>((hy + ps.sensor_pr) * inv_r);
                    const int   iz    = static_cast<int>((hz + ps.sensor_pr) * inv_r);
                    const int   res   = ps.sensor_res;
                    if (iy >= 0 && iy < res && iz >= 0 && iz < res) {
                        double amp_mag = 0.0;
                        for (int b = 0; b < nb; ++b) {
                            const double re = amp_prop[b].real(), im = amp_prop[b].imag();
                            amp_mag += std::sqrt(re*re + im*im);
                        }
                        const int idx = 0 * res * res + iy * res + iz;  /* ch0 only */
                        std::lock_guard<std::mutex> lk(ps.sensor_mu);
                        const double nv = (ps.sensor_accum[static_cast<size_t>(idx)] += amp_mag);
                        if (nv > ps.sensor_peak[0]) ps.sensor_peak[0] = nv;
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
            pipeline_finish_ray(ps);
            continue;
        }
        /* ── Terminal: emissive ── */
        if (tri.flags & MAT_FLAG_EMISSIVE) {
            HitRecord eh = hr;
            eh.is_emissive_hit = true;
            push_terminal(eh);
            pipeline_finish_ray(ps);
            /* ── Sensor accumulator (T3): backward ray found emissive surface.
             * Use sensor_origin_y/z (set at submit time, propagated through all
             * children) so this works regardless of how many refractions the ray
             * passed through on its way from the sensor to the source. */
            if (ps.sensor_res > 0 && hr.ray.color_flag == 1) {
                const float sy    = hr.ray.sensor_origin_y;
                const float sz    = hr.ray.sensor_origin_z;
                const int   res   = ps.sensor_res;
                const float inv_r = static_cast<float>(res) / (2.0f * ps.sensor_pr);
                const int   iy    = static_cast<int>((sy + ps.sensor_pr) * inv_r);
                const int   iz    = static_cast<int>((sz + ps.sensor_pr) * inv_r);
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
                    ps.sensor_accum[static_cast<size_t>(idx1)] += 1.0;
                    ps.sensor_accum[static_cast<size_t>(idx2)] += amp_mag;
                }
            }
            continue;
        }
        /* ── Budget exhausted ── */
        if (hr.ray.bounces_left <= 0) {
            push_terminal(hr);
            pipeline_finish_ray(ps);
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
            return ri;
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

            if (!can_refract) {
                V3d  rd = (in_dir - 2.0 * in_dir.dot(hit_n) * hit_n).normalized();
                VXcd ra = amp;
                for (int b = 0; b < nb; ++b)
                    ra[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                pipeline_spawn_child(ps, make_child(rd, ra));
            } else {
                const double sin2_t = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
                const double cos_t  = std::sqrt(std::max(0.0, 1.0 - sin2_t));
                const double R      = fresnel_R(cos_i, cos_t, n1, n2);
                const int    new_med = has_pair ? medium_to
                    : ((hr.ray.medium_mat_idx == tri.mat_idx) ? -1 : tri.mat_idx);

                if (ps.cfg.max_children >= 2) {
                    {
                        V3d  rd = (in_dir - 2.0*in_dir.dot(hit_n)*hit_n).normalized();
                        VXcd ra = amp;
                        double rs = std::sqrt(R);
                        for (int b = 0; b < nb; ++b)
                            ra[b] *= rs * mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                        pipeline_spawn_child(ps, make_child(rd, ra));
                    }
                    {
                        VXcd ta = amp;
                        double ts = std::sqrt(1.0 - R);
                        for (int b = 0; b < nb; ++b)
                            ta[b] *= ts;
                        pipeline_spawn_child(ps, make_child(refracted, ta, new_med));
                    }
                } else {
                    if (U01(rng) < R) {
                        V3d  rd = (in_dir - 2.0*in_dir.dot(hit_n)*hit_n).normalized();
                        VXcd ra = amp;
                        for (int b = 0; b < nb; ++b)
                            ra[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
                        pipeline_spawn_child(ps, make_child(rd, ra));
                    } else {
                        VXcd ta = amp;
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
                pipeline_finish_ray(ps);
                continue;
            }

            /* Opaque: diffuse or specular */
            V3d new_dir;
            if (U01(rng) < mat_cache_diffusion(st.mat_cache, tri.mat_idx))
                new_dir = cosine_hemisphere(hit_n, rng);
            else {
                new_dir = (in_dir - 2.0 * in_dir.dot(hit_n) * hit_n).normalized();
                if (new_dir.dot(hit_n) < 0.0) new_dir = cosine_hemisphere(hit_n, rng);
            }
            VXcd na = amp;
            for (int b = 0; b < nb; ++b)
                na[b] *= mat_cache_refl(st.mat_cache, tri.mat_idx, b);
            if (tri.flags & MAT_FLAG_REACTIVE)
                apply_reactive_shift_cached(na, st.mat_cache, tri.mat_idx);

            double max_abs = 0.0;
            for (int b = 0; b < nb; ++b)
                max_abs = std::max(max_abs, std::abs(na[b]));
            if (max_abs >= hr.ray.min_amplitude)
                pipeline_spawn_child(ps, make_child(new_dir, na));
            else
                push_terminal(hr);  /* amplitude extinguished */
        }

        pipeline_finish_ray(ps);
    }  /* end for rh : batch */

        /* ── BDPT snap pass (T3) ─────────────────────────────────────────
         * For each backward-origin STRIKE in this batch, find forward STRIKE
         * hits in YZ proximity and accumulate on the sensor image:
         *
         *   d <= eps          → exact match → ch2 (blue)  + sugar on priority map
         *   eps < d <= 3*eps  → near-miss   → ch3 (green, provisional), Gaussian
         *
         * ch3 fills in tentative data where convergence is approaching but not
         * yet snapped; it is superseded visually by ch2 as exact matches arrive. */
        if (ps.sensor_res > 0) {
            const int   res    = ps.sensor_res;
            const float eps    = ps.sensor_eps;
            const float eps3   = 3.0f * eps;
            const float inv_r  = static_cast<float>(res) / (2.0f * ps.sensor_pr);
            const float sigma2 = eps * eps;

            struct HitXY {
                float y, z, amp, ss_y, ss_z, ss_x;
                float dir_y, dir_z, dir_x;   /* incoming direction for collinearity */
            };
            std::vector<HitXY> fwd_hits, rev_hits;
            fwd_hits.reserve(batch.size());
            rev_hits.reserve(batch.size());

            for (auto& rh : batch) {
                const HitRecord& hr = rh.base;
                const V3d& hp = hr.hit_pos;
                double amp_mag = 0.0;
                for (int b = 0; b < (int)hr.amp_propagated.size(); ++b) {
                    const double re = hr.amp_propagated[b].real();
                    const double im = hr.amp_propagated[b].imag();
                    amp_mag += std::sqrt(re*re + im*im);
                }
                HitXY h;
                h.y     = static_cast<float>(hp.y());
                h.z     = static_cast<float>(hp.z());
                h.amp   = static_cast<float>(amp_mag);
                h.ss_y  = static_cast<float>(hr.seg_start.y());
                h.ss_z  = static_cast<float>(hr.seg_start.z());
                h.ss_x  = static_cast<float>(hr.seg_start.x());
                h.dir_y = static_cast<float>(hr.incoming_dir.y());
                h.dir_z = static_cast<float>(hr.incoming_dir.z());
                h.dir_x = static_cast<float>(hr.incoming_dir.x());
                if      (hr.ray.color_flag == 0) fwd_hits.push_back(h);
                else if (hr.ray.bounce     == 0) rev_hits.push_back(h);
            }

            /* --- lock-free stats: track nearest pair and best collinearity --- */
            float batch_nearest_d2   = std::numeric_limits<float>::max();
            float batch_best_colinear = -1.0f;
            for (auto& rv : rev_hits) {
                for (auto& fv : fwd_hits) {
                    const float dy = rv.y - fv.y;
                    const float dz = rv.z - fv.z;
                    const float d2 = dy*dy + dz*dz;
                    if (d2 < batch_nearest_d2) batch_nearest_d2 = d2;
                    /* Collinearity: |cos θ| between forward and backward incoming dirs.
                     * A back-to-back pair on the same path → cos θ ≈ -1 → |cos|≈1. */
                    const float fx = fv.dir_x, fy = fv.dir_y, fz = fv.dir_z;
                    const float rx = rv.dir_x, ry = rv.dir_y, rz = rv.dir_z;
                    const float flen = std::sqrt(fx*fx + fy*fy + fz*fz) + 1e-12f;
                    const float rlen = std::sqrt(rx*rx + ry*ry + rz*rz) + 1e-12f;
                    const float colinear = std::abs((fx*rx + fy*ry + fz*rz) / (flen * rlen));
                    if (colinear > batch_best_colinear) batch_best_colinear = colinear;
                }
            }
            if (batch_nearest_d2 < std::numeric_limits<float>::max()) {
                /* Atomic CAS-loop to update running minimum nearest d² */
                uint64_t newbits;
                std::memcpy(&newbits, &batch_nearest_d2, sizeof(float));
                /* Pad to 64 bits for atomic; only lower 32 bits meaningful. */
                newbits &= 0xFFFFFFFFULL;
                for (;;) {
                    uint64_t cur = ps.bdpt_nearest_d2_bits.load(std::memory_order_relaxed);
                    float cur_f; uint32_t cur32 = static_cast<uint32_t>(cur & 0xFFFFFFFFULL);
                    std::memcpy(&cur_f, &cur32, sizeof(float));
                    if (batch_nearest_d2 >= cur_f) break;
                    if (ps.bdpt_nearest_d2_bits.compare_exchange_weak(cur, newbits,
                            std::memory_order_relaxed, std::memory_order_relaxed)) break;
                }
            }
            if (batch_best_colinear > -1.0f) {
                uint64_t newbits;
                std::memcpy(&newbits, &batch_best_colinear, sizeof(float));
                newbits &= 0xFFFFFFFFULL;
                for (;;) {
                    uint64_t cur = ps.bdpt_best_colinear_bits.load(std::memory_order_relaxed);
                    float cur_f; uint32_t cur32 = static_cast<uint32_t>(cur & 0xFFFFFFFFULL);
                    std::memcpy(&cur_f, &cur32, sizeof(float));
                    if (batch_best_colinear <= cur_f) break;
                    if (ps.bdpt_best_colinear_bits.compare_exchange_weak(cur, newbits,
                            std::memory_order_relaxed, std::memory_order_relaxed)) break;
                }
            }

            if (!fwd_hits.empty() && !rev_hits.empty()) {
                std::lock_guard<std::mutex> lk(ps.sensor_mu);

                for (auto& rv : rev_hits) {
                    if (std::abs(rv.ss_x - ps.sensor_px) >= 0.004f) continue;
                    const int iy = static_cast<int>((rv.ss_y + ps.sensor_pr) * inv_r);
                    const int iz = static_cast<int>((rv.ss_z + ps.sensor_pr) * inv_r);
                    if (iy < 0 || iy >= res || iz < 0 || iz >= res) continue;

                    double exact_w = 0.0, nearmi_w = 0.0;
                    for (auto& fv : fwd_hits) {
                        const float dy = rv.y - fv.y;
                        const float dz = rv.z - fv.z;
                        const float d2 = dy*dy + dz*dz;
                        if (d2 > eps3 * eps3) continue;
                        const float wg = fv.amp * std::exp(-d2 / sigma2);
                        if (d2 <= eps * eps) exact_w  += wg;
                        else                 nearmi_w += wg;
                    }

                    if (exact_w > 0.0) {
                        ps.sensor_accum[static_cast<size_t>(2 * res * res + iy * res + iz)]
                            += exact_w * rv.amp;
                        ps.bdpt_exact_snaps.fetch_add(1, std::memory_order_relaxed);
                        /* Sugar splash around exact match pixel. */
                        const float sugar = static_cast<float>(exact_w) * rv.amp * 0.4f;
                        for (int dy2 = -2; dy2 <= 2; ++dy2) {
                            for (int dz2 = -2; dz2 <= 2; ++dz2) {
                                const int ny = iy + dy2, nz = iz + dz2;
                                if (ny < 0 || ny >= res || nz < 0 || nz >= res) continue;
                                ps.priority_map[static_cast<size_t>(ny * res + nz)] +=
                                    sugar * std::exp(-0.5f * static_cast<float>(dy2*dy2 + dz2*dz2));
                            }
                        }
                    }
                    if (nearmi_w > 0.0) {
                        const double nv3 = (ps.sensor_accum[static_cast<size_t>(3 * res * res + iy * res + iz)]
                            += nearmi_w * rv.amp);
                        if (nv3 > ps.sensor_peak[3]) ps.sensor_peak[3] = nv3;
                        ps.bdpt_near_miss_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                /* One diffusion step on priority map (coeffs sum <1 → steady state=1).
                 * 0.90 self + 0.02*4 neighbours + 0.02 baseline injection = 1.0 eq. */
                if (!ps.priority_map.empty()) {
                    std::vector<float> tmp(ps.priority_map);
                    for (int y = 1; y < res - 1; ++y) {
                        for (int z = 1; z < res - 1; ++z) {
                            const float nb = tmp[(y-1)*res+z] + tmp[(y+1)*res+z]
                                           + tmp[y*res+z-1]   + tmp[y*res+z+1];
                            ps.priority_map[static_cast<size_t>(y*res+z)] =
                                0.90f * tmp[y*res+z] + 0.02f * nb + 0.02f;
                        }
                    }
                }
            }
        }

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
                pipeline_finish_ray(ps);
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

            pipeline_finish_ray(ps);
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
            gl_compute_release_current();   /* transfer ownership to the GPU worker thread */
            ps->workers.emplace_back([ps]() {
                ps->gpu_dispatch->thread_main(*ps);
            });
        } else {
            fprintf(stderr, "[gpu-dispatch] GL init FAILED (%s) — falling back to CPU\n",
                    ps->gpu_dispatch->ctx.error[0] ? ps->gpu_dispatch->ctx.error : "unknown error");
            fflush(stderr);
            delete ps->gpu_dispatch;
            ps->gpu_dispatch = nullptr;
            /* GPU failed — if we deferred CPU workers above, add them now */
            if (defer_cpu) spawn_cpu_stages();
            /* Restore T4 CPU wave thread */
            if (!ps->arenas.empty())
                ps->workers.emplace_back([ps](){ pipeline_wave_solver(*ps); });
        }
    }

    return ps;
}

void ray_pipeline_destroy(RayPipelineState* ps)
{
    if (!ps) return;
    ps->Q_intent.set_done();
    ps->Q_hit.set_done();
    ps->Q_refined.set_done();
    ps->Q_wave.set_done();
    for (auto& t : ps->workers) if (t.joinable()) t.join();
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
    for (int i = 0; i < n_intents; ++i) {
        RayIntent ri = intents[i];
        /* Raise per-ray floor to the pipeline minimum (never lower it). */
        if (min_amp > ri.min_amplitude) ri.min_amplitude = min_amp;
        if (max_q > 0)
            ps->Q_intent.push_bounded(std::move(ri), max_q);
        else
            ps->Q_intent.push(std::move(ri));
    }
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
        int next = std::max(256, cur / 2);
        if (next != cur) s.batch_sz_gpu.store(next, std::memory_order_relaxed);
    };
    auto grow_batch = [](StageStats& s) {
        int cur = s.batch_sz_gpu.load(std::memory_order_relaxed);
        int next = std::min(65536, cur + std::max(64, cur / 8));
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
    float plate_x, float plate_r,
    int   res,
    float bdpt_eps)
{
    if (!ps) return;
    std::lock_guard<std::mutex> lk(ps->sensor_mu);
    ps->sensor_res = res;
    ps->sensor_px  = plate_x;
    ps->sensor_pr  = (plate_r > 0.0f) ? plate_r : 0.16f;
    ps->sensor_eps = (bdpt_eps > 0.0f) ? bdpt_eps : 0.008f;
    /* 4 channels: ch0=forward hits, ch1=backward emissive, ch2=exact BDPT, ch3=near-miss */
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
     *   R = ch0  (forward plate hits, irradiance)
     *   G = ch1 + ch3  (sensor photon count: each emissive connection counts 1,
     *         plus provisional near-miss ch3 attenuated where ch2 is strong)
     *   B = ch2 (amplitude-weighted BDPT radiance) */
    for (int y = 0; y < res; ++y) {
        const int out_y = res - 1 - y;  /* flip for OpenGL bottom-to-top convention */
        for (int z = 0; z < res; ++z) {
            const size_t src_px = static_cast<size_t>(y * res + z);
            const size_t dst_px = static_cast<size_t>(out_y * res + z);
            const float ch2_n = tone(ps->sensor_accum[2 * pix + src_px], peak[2]);
            const float attn  = 1.0f - std::min(1.0f, ch2_n);  /* provisional fades as exact grows */
            buf[dst_px * 3 + 0] = tone(ps->sensor_accum[0 * pix + src_px], peak[0]);  /* R */
            buf[dst_px * 3 + 1] = std::min(1.0f,
                tone(ps->sensor_accum[1 * pix + src_px], peak[1]) +
                attn * tone(ps->sensor_accum[3 * pix + src_px], peak[3]));             /* G */
            buf[dst_px * 3 + 2] = ch2_n;                                               /* B */
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

