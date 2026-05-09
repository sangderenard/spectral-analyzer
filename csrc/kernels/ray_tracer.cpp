#define _USE_MATH_DEFINES
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

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <vector>

using cd   = std::complex<double>;
using V3d  = Eigen::Vector3d;
using VXcd = Eigen::VectorXcd;

static constexpr double TWO_PI     = 2.0 * M_PI;
static constexpr double EPS        = 1e-9;
static constexpr int    BVH_LEAF_MAX = 4;   /* triangles per BVH leaf */

/* ── Unified material flags (Phase 2: single source of truth) ─────────────── */
#include "mat_flags_generated.h"

/* ── Geometry ──────────────────────────────────────────────────────────────── */

struct Triangle {
    V3d v0;
    V3d edge1;   /* v1 - v0 (precomputed for Möller-Trumbore) */
    V3d edge2;   /* v2 - v0 */
    V3d normal;  /* outward unit normal */
    int mat_idx = 0;   /* index into RayTracerState::mat_buf rows (per-band record stride) */
    int flags   = 0;   /* MAT_FLAG_TRANSMISSIVE | MAT_FLAG_APERTURE_STOP | … */
};

/* ── Möller-Trumbore ray-triangle intersection ─────────────────────────────── */

/* Returns true and sets t_out (distance > EPS) when the ray (orig + t*dir)
   hits the triangle.  Culls back-faces when the ray originates inside the
   scene (normal-dot-dir could be positive after a diffuse bounce). */
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
    if (t < 1e-6)
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
        double tmin = 1e-6;
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
                    && t < t_min)
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

/* ── Segment writer ─────────────────────────────────────────────────────────── */

static inline void write_segment(
    float* buf, int& count, int cap,
    const V3d& p0, const V3d& p1,
    int src_id, int bounce, int band,
    cd amp, double path_len)
{
    if (count >= cap) return;
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
}

/* ── RayTracerState ─────────────────────────────────────────────────────────── */

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
};

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
    if (kind != TRI_PARAM_SURFACE_POLY_BARY)
        return false;

    const auto& payload = st.tri_group_parametric_payload[static_cast<size_t>(gid)];
    if (payload.size() < sizeof(double) * 6)
        return false;

    const double* c = reinterpret_cast<const double*>(payload.data());
    double u = 0.0, v = 0.0;
    tri_bary_uv(tri, hit_pos, u, v);

    const double delta = c[0] + c[1] * u + c[2] * v
                       + c[3] * u * u + c[4] * u * v + c[5] * v * v;

    const V3d t1 = tri.edge1.normalized();
    const V3d t2 = tri.edge2.normalized();
    const double dzdu = c[1] + 2.0 * c[3] * u + c[4] * v;
    const double dzdv = c[2] + c[4] * u + 2.0 * c[5] * v;

    V3d warped_n = (tri.normal - dzdu * t1 - dzdv * t2).normalized();
    if (warped_n.norm() < EPS)
        warped_n = tri.normal;

    out_normal = warped_n;
    out_pos = hit_pos + warped_n * delta;
    return true;
}

static bool tri_is_transmissive(const RayTracerState& st, int tri_id)
{
    if (tri_id < 0 || tri_id >= static_cast<int>(st.tris.size()))
        return false;
    return (st.tris[static_cast<size_t>(tri_id)].flags & MAT_FLAG_TRANSMISSIVE) != 0;
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
        if (st.camera_transparency_mode == RT_CAM_TRANSPARENCY_XRAY
            && tri_is_transmissive(st, hit_tri))
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
                for (int b = 0; b < st.n_bands; ++b) {
                    (void)field_grid_inject_amplitude(st.camera_field_grid, b, p, 1.0f, 0.0f);
                }
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

            bool transparent = (st.camera_transparency_mode == RT_CAM_TRANSPARENCY_XRAY)
                            && tri_is_transmissive(st, hit_tri);
            if (!transparent)
                return false;

            const double advance = std::max(EPS * 200.0, (hit_pos - cur).norm() + EPS * 200.0);
            cur += dir * advance;
            remain -= std::min(remain, advance);
        }
        return remain <= EPS;
    }

    return true;
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
    for (int b = 0; b < st.n_bands; ++b) {
        (void)field_grid_inject_amplitude(
            st.camera_field_grid, b, p,
            static_cast<float>(amp_prop[b].real()),
            static_cast<float>(amp_prop[b].imag()));
    }
}

/* ── Generic inner ray loop ─────────────────────────────────────────────────── */

/* PerBandFn is called once per (source, bounce, band) for every valid surface hit.
 *
 * Signature:
 *   bool per_band(int src_id, int bounce, int band,
 *                 cd new_amp,        // amplitude AFTER propagation, BEFORE reflection
 *                 const V3d& p0,     // segment start (previous hit / source pos)
 *                 const V3d& p1,     // hit point
 *                 double total_path) // cumulative path length to p1 (metres)
 *
 * Returning false from the callback signals an abort: the inner band loop is
 * cut short, the current ray is abandoned, and the outer source/ray loops stop.
 * Used by ray_tracer_trace to exit early when the segment buffer is full.
 * Integrators should always return true.
 */
template<typename PerBandFn>
static void trace_rays(
    const RayTracerState& st,
    int n_sources, const double* src_pos,
    const double* src_dir, const double* src_directivity,
    int n_rays, int max_bounces, double min_amplitude,
    std::mt19937_64& rng,
    bool& abort,          /* set to true by callback to stop all loops */
    PerBandFn&& per_band)
{
    const int  n_bands = st.n_bands;
    const bool has_bvh = !st.bvh_nodes.empty();

    std::uniform_real_distribution<double> U(0.0, 1.0);
    VXcd amp(n_bands);

    for (int si = 0; si < n_sources && !abort; ++si) {
        const double* sp = src_pos + si * 3;
        const double* sd = src_dir + si * 3;
        V3d src_p(sp[0], sp[1], sp[2]);
        V3d src_d = V3d(sd[0], sd[1], sd[2]).normalized();
        double dirpow = src_directivity[si];

        for (int ri = 0; ri < n_rays && !abort; ++ri) {
            V3d dir = fibonacci_sphere_dir(ri, n_rays, src_d);

            double cos_a      = dir.dot(src_d);
            double dir_weight = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
            if (dir_weight < 0.01) continue;

            for (int b = 0; b < n_bands; ++b)
                amp[b] = cd(dir_weight, 0.0);

            V3d    pos      = src_p;
            double path_len = 0.0;
            V3d    cur_dir  = dir;

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                double t_min   = 1e18;
                int    hit_tri = -1;

                if (has_bvh) {
                    V3d inv_dir = cur_dir.cwiseInverse();
                    bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                              pos, cur_dir, inv_dir, t_min, hit_tri);
                } else {
                    for (size_t ti = 0; ti < st.tris.size(); ++ti) {
                        double t;
                        if (ray_triangle_hit(pos, cur_dir, st.tris[ti], t)
                            && t < t_min)
                        {
                            t_min   = t;
                            hit_tri = static_cast<int>(ti);
                        }
                    }
                }

                if (hit_tri < 0) break;

                V3d    hit_pos    = pos + t_min * cur_dir;
                double total_path = path_len + t_min;
                double spread     = 1.0 / (1.0 + path_len + t_min * 0.5);

                const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
                double max_abs = 0.0;

                for (int b = 0; b < n_bands; ++b) {
                    double kt    = st.k_real[b] * t_min;
                    double decay = std::exp(-st.atmo_abs[b] * t_min) * spread;
                    cd prop(decay * std::cos(kt), decay * std::sin(kt));

                    cd new_amp = amp[b] * prop;

                    if (!per_band(si, bounce, b, new_amp, pos, hit_pos, total_path)) {
                        abort = true;
                        goto next_ray;
                    }

                    amp[b]     = new_amp * mat_refl_complex(st, tri.mat_idx, b);
                    double a   = std::abs(amp[b]);
                    if (a > max_abs) max_abs = a;
                }

                if (max_abs < min_amplitude) break;

                /* Orient the boundary normal against the incoming ray.  Meshes
                 * extracted from scene builders are not guaranteed to have inward
                 * normals for a cavity, and using the raw normal here can offset the
                 * next origin through the wall and kill the ray set after one hit. */
                V3d hit_n = tri.normal;
                if (cur_dir.dot(hit_n) > 0.0)
                    hit_n = -hit_n;

                if (U(rng) < mat_diffusion(st, tri.mat_idx)) {
                    cur_dir = cosine_hemisphere(hit_n, rng);
                } else {
                    cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n) * hit_n).normalized();
                    if (cur_dir.dot(hit_n) < 0.0)
                        cur_dir = cosine_hemisphere(hit_n, rng);
                }

                pos = hit_pos + cur_dir * (EPS * 100.0);

                path_len += t_min;
            }
            next_ray:;
        }
    }
}

/* ── Extended inner ray loop (v2) ──────────────────────────────────────────── */

/* Like trace_rays but passes hit_tri, incoming_dir, surface_normal, and the
 * full per-band amplitude vector (AFTER propagation, BEFORE reflection) to the
 * callback.  This allows callers to accumulate per-triangle irradiance.
 *
 * HitFn signature:
 *   bool hit_fn(int src_id, int bounce,
 *               int hit_tri,
 *               const V3d& incoming_dir,   // unit ray direction before hit
 *               const V3d& surface_normal, // outward-facing normal (corrected)
 *               const VXcd& amp_prop,      // amplitude vector after propagation, before reflection
 *               const V3d& p0,             // segment start
 *               const V3d& p1,             // hit point
 *               double total_path)         // cumulative path length to p1 (m)
 */
template<typename HitFn>
static void trace_rays_v2(
    const RayTracerState& st,
    int n_sources, const double* src_pos,
    const double* src_dir, const double* src_directivity,
    int n_rays, int max_bounces, double min_amplitude,
    std::mt19937_64& rng,
    bool& abort,
    HitFn&& hit_fn)
{
    const int  n_bands = st.n_bands;
    const bool has_bvh = !st.bvh_nodes.empty();

    std::uniform_real_distribution<double> U(0.0, 1.0);
    VXcd amp(n_bands);
    VXcd amp_prop(n_bands);

    for (int si = 0; si < n_sources && !abort; ++si) {
        const double* sp = src_pos + si * 3;
        const double* sd = src_dir + si * 3;
        V3d src_p(sp[0], sp[1], sp[2]);
        V3d src_d = V3d(sd[0], sd[1], sd[2]).normalized();
        double dirpow = src_directivity[si];

        for (int ri = 0; ri < n_rays && !abort; ++ri) {
            V3d dir = fibonacci_sphere_dir(ri, n_rays, src_d);

            double cos_a      = dir.dot(src_d);
            double dir_weight = std::pow(std::max(0.0, (cos_a + 1.0) * 0.5), dirpow);
            if (dir_weight < 0.01) continue;

            for (int b = 0; b < n_bands; ++b)
                amp[b] = cd(dir_weight, 0.0);

            V3d    pos      = src_p;
            double path_len = 0.0;
            V3d    cur_dir  = dir;

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                double t_min   = 1e18;
                int    hit_tri = -1;

                if (has_bvh) {
                    V3d inv_dir = cur_dir.cwiseInverse();
                    bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                              pos, cur_dir, inv_dir, t_min, hit_tri);
                } else {
                    for (size_t ti = 0; ti < st.tris.size(); ++ti) {
                        double t;
                        if (ray_triangle_hit(pos, cur_dir, st.tris[ti], t)
                            && t < t_min)
                        {
                            t_min   = t;
                            hit_tri = static_cast<int>(ti);
                        }
                    }
                }

                if (hit_tri < 0) break;

                V3d    hit_pos    = pos + t_min * cur_dir;
                double total_path = path_len + t_min;
                double spread     = 1.0 / (1.0 + path_len + t_min * 0.5);

                const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];

                /* Surface normal corrected to face the incoming ray. */
                V3d hit_n = tri.normal;
                if (cur_dir.dot(hit_n) > 0.0)
                    hit_n = -hit_n;

                /* Propagate amplitude (no reflection yet). */
                double max_abs = 0.0;
                for (int b = 0; b < n_bands; ++b) {
                    double kt    = st.k_real[b] * t_min;
                    double decay = std::exp(-st.atmo_abs[b] * t_min) * spread;
                    cd prop(decay * std::cos(kt), decay * std::sin(kt));
                    amp_prop[b] = amp[b] * prop;
                    double a    = std::abs(amp_prop[b]);
                    if (a > max_abs) max_abs = a;
                }

                /* Deliver to caller with full context. */
                if (!hit_fn(si, bounce, hit_tri, cur_dir, hit_n, amp_prop,
                            pos, hit_pos, total_path)) {
                    abort = true;
                    goto v2_next_ray;
                }

                /* Apply reflection. */
                for (int b = 0; b < n_bands; ++b)
                    amp[b] = amp_prop[b] * mat_refl_complex(st, tri.mat_idx, b);

                if (max_abs < min_amplitude) break;

                /* Scatter / reflect direction. */
                if (U(rng) < mat_diffusion(st, tri.mat_idx)) {
                    cur_dir = cosine_hemisphere(hit_n, rng);
                } else {
                    cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n) * hit_n).normalized();
                    if (cur_dir.dot(hit_n) < 0.0)
                        cur_dir = cosine_hemisphere(hit_n, rng);
                }

                pos = hit_pos + cur_dir * (EPS * 100.0);
                path_len += t_min;
            }
            v2_next_ray:;
        }
    }
}

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
    if (!st || !out_segs || !out_count) return SK_ERR_NULL_STATE;

    *out_count = 0;
    int count  = 0;
    bool abort = false;

    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, int b, cd new_amp,
            const V3d& p0, const V3d& p1, double total_path) -> bool
        {
            /* Silently drop segments once the vis buffer is full — never abort
               the trace itself, all rays must complete for correct physics. */
            write_segment(out_segs, count, out_cap,
                          p0, p1, si, bounce, b,
                          new_amp, total_path - (p1 - p0).norm());
            return true;
        });

    *out_count = count;
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

    int  count = 0;
    bool abort = false;

    if (out_count) *out_count = 0;

    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays_v2(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, int hit_tri,
            const V3d& incoming_dir, const V3d& surface_normal,
            const VXcd& amp_prop,
            const V3d& p0, const V3d& p1, double total_path) -> bool
        {
            /* Write segment record (vis buffer). */
            if (out_segs && out_cap > 0) {
                for (int b = 0; b < n_bands; ++b) {
                    write_segment(out_segs, count, out_cap,
                                  p0, p1, si, bounce, b,
                                  amp_prop[b],
                                  total_path - (p1 - p0).norm());
                }
            }

            /* Irradiance contribution to triangle surface.
             *
             * energy = |A|² * cos(θ) / area
             *
             * cos(θ) is the angle between the incoming direction and the
             * surface normal.  We already corrected surface_normal to face
             * the incoming ray, so:
             *   cos_in = -dot(incoming_dir, surface_normal)
             * (incoming_dir points AWAY from the source, normal points TOWARD it).
             */
            if (hit_tri >= 0 && hit_tri < n_tri) {
                double area    = st->tri_areas[static_cast<size_t>(hit_tri)];
                double cos_in  = std::max(0.0, -incoming_dir.dot(surface_normal));
                double inv_area = cos_in / std::max(area, AREA_EPS);

                size_t base = static_cast<size_t>(hit_tri) * n_bands;

                float* dest = (bounce == 0 && out_direct) ? out_direct : out_indirect;
                if (dest) {
                    for (int b = 0; b < n_bands; ++b) {
                        double e = std::norm(amp_prop[b]) * inv_area;
                        dest[base + b] += static_cast<float>(e);
                    }
                }
            }
            return true;
        });

    if (out_count) *out_count = count;
    return SK_OK;
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
    if (!st || !out_re || !out_im) return SK_ERR_NULL_STATE;

    const int   n_bands = st->n_bands;
    const double inv_speed = 1.0 / speed_m_s;

    /* Pre-load receiver positions into V3d for fast distance checks. */
    std::vector<V3d> rpos(static_cast<size_t>(n_receivers));
    for (int ri = 0; ri < n_receivers; ++ri)
        rpos[static_cast<size_t>(ri)] = V3d(rec_pos[ri*3], rec_pos[ri*3+1], rec_pos[ri*3+2]);

    bool abort = false;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int /*bounce*/, int b, cd new_amp,
            const V3d& /*p0*/, const V3d& p1, double total_path) -> bool
        {
            /* Delay in samples from source to this hit point. */
            double delay_s  = total_path * inv_speed * sample_rate;
            int    t0       = static_cast<int>(delay_s);
            double frac     = delay_s - t0;  /* for linear interpolation */

            for (int ri = 0; ri < n_receivers; ++ri) {
                double dist = (p1 - rpos[static_cast<size_t>(ri)]).norm();
                double apr  = rec_aperture_r[ri];
                if (dist >= apr) continue;

                /* Linear distance falloff within aperture. */
                double weight = 1.0 - dist / apr;

                size_t base = (static_cast<size_t>(si) * n_receivers * n_bands * n_samples)
                            + (static_cast<size_t>(ri) * n_bands * n_samples)
                            + (static_cast<size_t>(b)  * n_samples);

                /* Splat with linear interpolation across two adjacent bins. */
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
            return true;
        });

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
    if (!st || !out_image) return SK_ERR_NULL_STATE;

    /* Build orthonormal camera frame. */
    V3d cam_p(cam_pos[0], cam_pos[1], cam_pos[2]);
    V3d cam_f = V3d(cam_fwd[0], cam_fwd[1], cam_fwd[2]).normalized();
    V3d cam_u_hint(cam_up[0], cam_up[1], cam_up[2]);

    /* cam_right = fwd × up_hint, then re-derive true up. */
    V3d cam_r = cam_f.cross(cam_u_hint);
    if (cam_r.norm() < EPS)
        cam_r = cam_f.cross(V3d(1, 0, 0));  /* fallback if fwd ≈ up hint */
    cam_r.normalize();
    V3d cam_u = cam_r.cross(cam_f);  /* guaranteed orthogonal */

    const double tan_half_v  = std::tan(fov_rad * 0.5);
    const double aspect      = static_cast<double>(width) / height;
    const double tan_half_h  = tan_half_v * aspect;

    const int n_bands = st->n_bands;
    bool abort = false;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays_v2(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, int hit_tri,
            const V3d& incoming_dir, const V3d& surface_normal,
            const VXcd& amp_prop,
            const V3d& /*p0*/, const V3d& p1, double total_path) -> bool
        {
            V3d hit_pos = p1;
            V3d hit_n = surface_normal;
            if (hit_tri >= 0)
                apply_parametric_surface_point(*st, hit_tri, p1, hit_pos, hit_n);

            V3d v = hit_pos - cam_p;
            double depth = v.dot(cam_f);
            int visible = 0;
            int context_entries = 0;
            if (depth > EPS) {
                const uint64_t c0 = st->camera_full_march_context_entries;
                visible = camera_visible_to_point(*st, cam_p, hit_pos) ? 1 : 0;
                context_entries = static_cast<int>(st->camera_full_march_context_entries - c0);
            }

            if (st->camera_field_grid)
                accumulate_field_capture(*st, hit_pos, amp_prop);
            append_camera_strike(*st, si, bounce, hit_tri, hit_pos, hit_n,
                                 incoming_dir, depth, total_path,
                                 visible, context_entries, amp_prop);

            if (depth <= EPS) return true;
            if (st->camera_depth_cull_enabled && depth > st->camera_depth_cull_m)
                return true;
            if (!visible)
                return true;

            double x_img = v.dot(cam_r);
            double y_img = v.dot(cam_u);
            double ndc_x =  x_img / (depth * tan_half_h);
            double ndc_y = -y_img / (depth * tan_half_v);
            int px = static_cast<int>((ndc_x + 1.0) * 0.5 * width);
            int py = static_cast<int>((ndc_y + 1.0) * 0.5 * height);

            if (px < 0 || px >= width || py < 0 || py >= height)
                return true;

            for (int b = 0; b < n_bands; ++b) {
                size_t idx = static_cast<size_t>(b) * (height * width)
                           + static_cast<size_t>(py) * width
                           + static_cast<size_t>(px);
                out_image[idx] += static_cast<float>(std::abs(amp_prop[b]));
            }
            return true;
        });

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
    if (!st || !out_image || !out_count) return SK_ERR_NULL_STATE;

    int count = 0;
    *out_count = 0;

    V3d cam_p(cam_pos[0], cam_pos[1], cam_pos[2]);
    V3d cam_f = V3d(cam_fwd[0], cam_fwd[1], cam_fwd[2]).normalized();
    V3d cam_u_hint(cam_up[0], cam_up[1], cam_up[2]);

    V3d cam_r = cam_f.cross(cam_u_hint);
    if (cam_r.norm() < EPS)
        cam_r = cam_f.cross(V3d(1, 0, 0));
    cam_r.normalize();
    V3d cam_u = cam_r.cross(cam_f);

    const double tan_half_v = std::tan(fov_rad * 0.5);
    const double aspect     = static_cast<double>(width) / height;
    const double tan_half_h = tan_half_v * aspect;

    bool abort = false;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays_v2(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, int hit_tri,
            const V3d& incoming_dir, const V3d& surface_normal,
            const VXcd& amp_prop,
            const V3d& p0, const V3d& p1, double total_path) -> bool
        {
            if (out_segs && out_cap > 0) {
                for (int b = 0; b < st->n_bands; ++b) {
                    write_segment(out_segs, count, out_cap,
                                  p0, p1, si, bounce, b,
                                  amp_prop[b],
                                  total_path - (p1 - p0).norm());
                }
            }

            V3d hit_pos = p1;
            V3d hit_n = surface_normal;
            if (hit_tri >= 0)
                apply_parametric_surface_point(*st, hit_tri, p1, hit_pos, hit_n);

            V3d v = hit_pos - cam_p;
            double depth = v.dot(cam_f);
            int visible = 0;
            int context_entries = 0;
            if (depth > EPS) {
                const uint64_t c0 = st->camera_full_march_context_entries;
                visible = camera_visible_to_point(*st, cam_p, hit_pos) ? 1 : 0;
                context_entries = static_cast<int>(st->camera_full_march_context_entries - c0);
            }

            if (st->camera_field_grid)
                accumulate_field_capture(*st, hit_pos, amp_prop);
            append_camera_strike(*st, si, bounce, hit_tri, hit_pos, hit_n,
                                 incoming_dir, depth, total_path,
                                 visible, context_entries, amp_prop);

            if (depth <= EPS) return true;
            if (st->camera_depth_cull_enabled && depth > st->camera_depth_cull_m)
                return true;
            if (!visible)
                return true;

            double x_img = v.dot(cam_r);
            double y_img = v.dot(cam_u);
            double ndc_x =  x_img / (depth * tan_half_h);
            double ndc_y = -y_img / (depth * tan_half_v);
            int px = static_cast<int>((ndc_x + 1.0) * 0.5 * width);
            int py = static_cast<int>((ndc_y + 1.0) * 0.5 * height);
            if (px >= 0 && px < width && py >= 0 && py < height) {
                for (int b = 0; b < st->n_bands; ++b) {
                    size_t idx = static_cast<size_t>(b) * (height * width)
                               + static_cast<size_t>(py) * width
                               + static_cast<size_t>(px);
                    out_image[idx] += static_cast<float>(std::abs(amp_prop[b]));
                }
            }
            return true;
        });

    *out_count = count;
    return SK_OK;
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
static inline void propagate_ms(
    VXcd& amp,
    const RayTracerState& st,
    const RtScaleContext* ctx,
    double dist,
    double path_len_start,
    const V3d& p0, const V3d& p1,
    int si, int bounce,
    float* out_segs, int& seg_count, int seg_cap,
    int context_id)
{
    const int    n_bands = st.n_bands;
    const int    scale_t = (ctx && ctx->scale_type == RT_SCALE_WAVE)
                             ? RT_SCALE_WAVE : RT_SCALE_GEOMETRIC;
    const double n_re    = ctx ? ctx->n_real : 1.0;
    const double n_im    = ctx ? ctx->n_imag : 0.0;
    const double c       = st.speed_m_s;

    for (int b = 0; b < n_bands; ++b) {
        double f_hz  = st.freq_hz_vec[b];
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
 * HitFn2: (si, bounce, hit_tri, incoming_dir, surface_normal,
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

            for (int b = 0; b < n_bands; ++b)
                amp[b] = cd(dir_weight, 0.0);

            V3d    pos      = src_p;
            double path_len = 0.0;
            V3d    cur_dir  = dir;

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
                                 si, bounce, out_segs, seg_count, seg_cap, -1);
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
                                     si, bounce, out_segs, seg_count, seg_cap, ci);
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

                V3d hit_n = st.tris[static_cast<size_t>(hit_tri)].normal;
                if (cur_dir.dot(hit_n) > 0.0) hit_n = -hit_n;

                double max_abs = 0.0;
                for (int b = 0; b < n_bands; ++b) {
                    double a = std::abs(amp_surf[b]);
                    if (a > max_abs) max_abs = a;
                }

                bool cont = hit_fn(si, bounce, hit_tri, cur_dir, hit_n,
                                   amp_surf, pos, hit_pos, total_path);
                if (!cont) goto ms_done;

                const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
                if (max_abs < min_amplitude) break;

                /* ── Surface interaction ──────────────────────────────────── */
                if (tri.flags & MAT_FLAG_APERTURE_STOP) {
                    /* Blade material: absorb.  Diffraction is handled
                     * physically — the dense coherent ray field that passes
                     * through the blade gaps is accumulated by project_coherent
                     * and propagated to the sensor by rs_propagate (exact
                     * Rayleigh-Sommerfeld).  No secondary wavelets needed. */
                    break;

                } else if (tri.flags & MAT_FLAG_TRANSMISSIVE) {
                    /* Phase 2: refractive boundary uses MatBuf-derived IOR.
                     * Convention: outside is air (n=1), inside is the material. */
                    double n_mat = mat_n_real(st, tri.mat_idx);
                    if (n_mat <= EPS || std::abs(n_mat - 1.0) < 1e-6) {
                        /* No effective refraction — fall through to opaque path. */
                        for (int b = 0; b < n_bands; ++b)
                            amp[b] = amp_surf[b] * mat_refl_complex(st, tri.mat_idx, b);
                        if (U(rng) < mat_diffusion(st, tri.mat_idx)) {
                            cur_dir = cosine_hemisphere(hit_n, rng);
                        } else {
                            cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n) * hit_n).normalized();
                            if (cur_dir.dot(hit_n) < 0.0)
                                cur_dir = cosine_hemisphere(hit_n, rng);
                        }
                    } else {
                        bool entering = (cur_dir.dot(tri.normal) < 0.0);
                        double n1 = entering ? 1.0   : n_mat;
                        double n2 = entering ? n_mat : 1.0;

                        /* hit_n is already flipped to oppose cur_dir. */
                        double cos_i = std::max(0.0, -cur_dir.dot(hit_n));
                        V3d refracted;
                        bool can_refract = snell_refract(cur_dir, hit_n, n1, n2, refracted);

                        if (!can_refract) {
                            /* TIR: perfect specular reflection, apply surface refl. */
                            cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n) * hit_n).normalized();
                            for (int b = 0; b < n_bands; ++b)
                                amp[b] = amp_surf[b] * mat_refl_complex(st, tri.mat_idx, b);
                        } else {
                            double sin2_t = (n1/n2) * (n1/n2) * (1.0 - cos_i * cos_i);
                            double cos_t  = std::sqrt(std::max(0.0, 1.0 - sin2_t));
                            double R      = fresnel_R(cos_i, cos_t, n1, n2);
                            if (U(rng) < R) {
                                /* Probabilistic reflection (Russian roulette, unbiased). */
                                cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n) * hit_n).normalized();
                                for (int b = 0; b < n_bands; ++b)
                                    amp[b] = amp_surf[b] * mat_refl_complex(st, tri.mat_idx, b);
                            } else {
                                /* Transmission: exact Snell direction, no refl scaling.
                                 * MC probability (1-R) handles energy balance. */
                                cur_dir = refracted;
                                for (int b = 0; b < n_bands; ++b)
                                    amp[b] = amp_surf[b];
                            }
                        }
                    }

                } else {
                    /* Opaque surface: Lambertian or specular reflection. */
                    for (int b = 0; b < n_bands; ++b)
                        amp[b] = amp_surf[b] * mat_refl_complex(st, tri.mat_idx, b);
                    if (U(rng) < mat_diffusion(st, tri.mat_idx)) {
                        cur_dir = cosine_hemisphere(hit_n, rng);
                    } else {
                        cur_dir = (cur_dir - 2.0 * cur_dir.dot(hit_n) * hit_n).normalized();
                        if (cur_dir.dot(hit_n) < 0.0)
                            cur_dir = cosine_hemisphere(hit_n, rng);
                    }
                }

                /* ── Reactive (fluorescent) re-emission ─────────────────────
                 * After the elastic surface interaction has updated `amp` and
                 * `cur_dir`, redistribute a fraction of the per-band energy
                 * to the Stokes-shifted destination band.  The yield is the
                 * material's reemission coefficient at band 0 — matches
                 * GLSL's `tri.emissive.w` / reemission packing convention. */
                if (tri.flags & MAT_FLAG_REACTIVE) {
                    const double shift_hz = mat_reactive_shift_hz(st, tri.mat_idx);
                    if (shift_hz > 0.0) {
                        /* Slot 6 = reemission (per the documented MatBuf
                         * SpectralBandRecord layout in material_db.py). */
                        const double yield_frac = static_cast<double>(
                            mat_band_record(st, tri.mat_idx, 0)[6]);
                        apply_reactive_shift(amp, st.freq_hz_vec,
                                             shift_hz, yield_frac);
                    }
                }

                pos       = hit_pos + cur_dir * (EPS * 200.0);
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
    if (!st || !out_segs || !out_count) return SK_ERR_NULL_STATE;
    *out_count = 0;
    int count = 0;
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    trace_rays_multiscale(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude, rng,
        out_segs, out_cap, count,
        [](int, int, int, const V3d&, const V3d&,
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
            const V3d& incoming_dir, const V3d& surface_normal,
            const VXcd& amp_prop,
            const V3d&, const V3d&, double) -> bool
        {
            if (hit_tri >= 0 && hit_tri < n_tri) {
                double area     = st->tri_areas[static_cast<size_t>(hit_tri)];
                double cos_in   = std::max(0.0, -incoming_dir.dot(surface_normal));
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
    int hit_tri, int& bounce,
    std::mt19937_64& rng,
    std::uniform_real_distribution<double>& U)
{
    if (bounce >= st.live_max_bounces) return false;

    const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
    V3d hit_n = tri.normal;
    if (dir.dot(hit_n) > 0.0) hit_n = -hit_n;

    if (tri.flags & MAT_FLAG_APERTURE_STOP) {
        return false;  /* absorbed */
    }

    {
        double n_mat = mat_n_real(st, tri.mat_idx);
        bool refractive = (tri.flags & MAT_FLAG_TRANSMISSIVE)
                       && n_mat > EPS && std::abs(n_mat - 1.0) > 1e-6;
        if (refractive) {
            bool entering  = (dir.dot(tri.normal) < 0.0);
            double n1      = entering ? 1.0   : n_mat;
            double n2      = entering ? n_mat : 1.0;
            double cos_i   = std::max(0.0, -dir.dot(hit_n));
            V3d refracted;
            bool ok        = snell_refract(dir, hit_n, n1, n2, refracted);
            if (!ok) {
                dir = (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized();
            } else {
                double s2t = (n1/n2)*(n1/n2)*(1.0 - cos_i*cos_i);
                double ct  = std::sqrt(std::max(0.0, 1.0 - s2t));
                double R   = fresnel_R(cos_i, ct, n1, n2);
                dir = (U(rng) < R)
                    ? (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized()
                    : refracted;
            }
            for (int b = 0; b < st.n_bands; ++b)
                amp[b] *= mat_refl_complex(st, tri.mat_idx, b);
        } else {
            for (int b = 0; b < st.n_bands; ++b)
                amp[b] *= mat_refl_complex(st, tri.mat_idx, b);
            if (U(rng) < mat_diffusion(st, tri.mat_idx)) {
                dir = cosine_hemisphere(hit_n, rng);
            } else {
                dir = (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized();
                if (dir.dot(hit_n) < 0.0)
                    dir = cosine_hemisphere(hit_n, rng);
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
        bool alive = apply_surface(st, amp, dir, hit_tri, rs.bounce, rng, U);
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
    return SK_OK;
}

extern "C" SK_API int ray_tracer_n_tri_groups(const RayTracerState* st)
{
    if (!st) return 0;
    return static_cast<int>(st->tri_groups.size());
}

/* ── Bidirectional integrator ───────────────────────────────────────────
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

/* Forward declarations for region-kind dispatch helpers — implementations
 * live just below the bidirectional function for narrative locality. */
static inline void apply_thin_lens_transform(
    const RtScaleContext& ctx, V3d& pos, V3d& dir);
static inline void apply_wave_aperture_transform(
    const RayTracerState& st, const RtScaleContext& ctx,
    const V3d& cross_pos, VXcd& amp);
static inline uint32_t dispatch_scale_context_entry(
    const RayTracerState& st, const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp);

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

    /* For each emissive group … */
    for (size_t g = 0; g < st->tri_groups.size(); ++g) {
        const TriGroupDesc& gd = st->tri_groups[g];
        if (!(gd.role_bits & TRI_GROUP_ROLE_EMISSIVE)) continue;
        const auto& idxs = st->tri_group_indices[g];
        const auto& cdf  = st->tri_group_cum_areas[g];
        if (idxs.empty()) continue;
        const double tot_area = cdf.empty() ? 0.0 : cdf.back();
        if (tot_area <= 0.0) continue;

        /* Initial per-band amplitude: unit (1+0j); the calibration loop
         * scales the result via Python.  We are preserving phase, so we
         * cannot pre-scale by emit_W without losing complex coherence
         * across bands. */
        for (int ri = 0; ri < n_rays_per_emitter; ++ri) {
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

            /* 4. Initial unit complex amplitude per band. */
            for (int b = 0; b < n_bands; ++b) amp[b] = cd(1.0, 0.0);

            V3d pos = origin + dir * (EPS * 200.0);
            double path_len = 0.0;
            uint32_t my_subpath = subpath_counter++;

            for (int bounce = 0; bounce < max_bounces; ++bounce) {
                double t_hit = 1e18; int hit_tri = -1;
                if (has_bvh) {
                    V3d inv = dir.cwiseInverse();
                    bvh_query(st->bvh_nodes, st->bvh_tri_ids, st->tris,
                              pos, dir, inv, t_hit, hit_tri);
                } else {
                    for (size_t ti = 0; ti < st->tris.size(); ++ti) {
                        double tt;
                        if (ray_triangle_hit(pos, dir, st->tris[ti], tt) && tt < t_hit) {
                            t_hit = tt; hit_tri = (int)ti;
                        }
                    }
                }
                if (hit_tri < 0) break;

                V3d hit_pos = pos + t_hit * dir;
                V3d hit_n_param = st->tris[hit_tri].normal;
                apply_parametric_surface_point(*st, hit_tri, hit_pos, hit_pos, hit_n_param);
                double total_path = path_len + t_hit;

                /* Phase + atmospheric attenuation along the segment. */
                for (int b = 0; b < n_bands; ++b) {
                    double k = st->k_real[b];
                    double atten = std::exp(-st->atmo_abs[b] * t_hit);
                    /* Spreading: 1/(1 + r) keeps numerics stable. */
                    double spread = 1.0 / (1.0 + total_path);
                    amp[b] *= std::polar(atten * spread, -k * t_hit);
                }

                /* Sensor capture? */
                int sgid = tri_sensor_group[hit_tri];
                if (sgid >= 0) {
                    V3d hit_n = st->tris[hit_tri].normal;
                    double cos_theta = std::abs(dir.dot(hit_n));
                    for (int b = 0; b < n_bands; ++b) {
                        if (rec_count >= out_cap) goto bdpt_done;
                        EndpointRecord& E = out_records[rec_count++];
                        E.subpath_id   = my_subpath;
                        E.band_id      = static_cast<uint32_t>(b);
                        E.group_id     = sgid;
                        E.vertex_index = -1;
                        E.pos[0] = (float)hit_pos.x();
                        E.pos[1] = (float)hit_pos.y();
                        E.pos[2] = (float)hit_pos.z();
                        E.pathlen_m = (float)total_path;
                        E.dir[0] = (float)dir.x();
                        E.dir[1] = (float)dir.y();
                        E.dir[2] = (float)dir.z();
                        E.pdf       = 1.0f / (float)n_rays_per_emitter;
                        E.amp_re    = (float)amp[b].real();
                        E.amp_im    = (float)amp[b].imag();
                        E.cos_theta = (float)cos_theta;
                        E._pad      = 0.0f;
                    }
                }

                /* Energy threshold. */
                double max_abs = 0.0;
                for (int b = 0; b < n_bands; ++b)
                    max_abs = std::max(max_abs, std::abs(amp[b]));
                if (max_abs < min_amplitude) break;

                /* Surface scatter — same minimum branch as the splatter, so
                 * material flag handling stays consistent.  Reactive +
                 * transmissive logic is intentionally minimal here; the full
                 * BDPT branch fanout is queued behind the branch_factor
                 * extension (see ray_tracer_set_branch_factor below). */
                const Triangle& tri = st->tris[hit_tri];
                V3d hit_n = hit_n_param;
                if (dir.dot(hit_n) > 0.0) hit_n = -hit_n;
                if (tri.flags & MAT_FLAG_APERTURE_STOP) break;
                if (U(rng) < mat_diffusion(*st, tri.mat_idx)) {
                    dir = cosine_hemisphere(hit_n, rng);
                } else {
                    dir = (dir - 2.0 * dir.dot(hit_n) * hit_n).normalized();
                    if (dir.dot(hit_n) < 0.0) dir = cosine_hemisphere(hit_n, rng);
                }
                for (int b = 0; b < n_bands; ++b)
                    amp[b] *= mat_refl_complex(*st, tri.mat_idx, b);

                if (tri.flags & MAT_FLAG_REACTIVE) {
                    double shift_hz = mat_reactive_shift_hz(*st, tri.mat_idx);
                    if (shift_hz > 0.0) {
                        double yf = (double)mat_band_record(*st, tri.mat_idx, 0)[6];
                        apply_reactive_shift(amp, st->freq_hz_vec, shift_hz, yf);
                    }
                }

                pos = hit_pos + dir * (EPS * 200.0);
                path_len = total_path;

                /* Region dispatch — apply any scale_context transforms
                 * whose sphere contains the new ray origin.  Smallest
                 * radius first (the contexts vector is pre-sorted), so
                 * inner zones override outer ones in a single pass. */
                for (const RtScaleContext& ctx : st->scale_contexts) {
                    V3d c(ctx.center[0], ctx.center[1], ctx.center[2]);
                    if ((pos - c).norm() <= ctx.radius) {
                        dispatch_scale_context_entry(*st, ctx, pos, dir, amp);
                    }
                }
            }
        }
    }
bdpt_done:
    /* ──────────────────────────────────────────────────────────────────
     * PIXEL_CONE sensor pass.
     *
     * For every SENSOR group whose sample_policy == TRI_GROUP_SAMPLE_PIXEL_CONE
     * and whose CameraSensorDesc was registered, drive the integrator from
     * the sensor side: deterministic outer loop over (px, py), stochastic
     * inner loop over n_aperture_samples points on the aperture.  This is
     * the camera-simulator path; the symmetric area-BDPT path above is the
     * reference standard and is preserved unchanged.
     *
     * Aperture sampling: stochastic uniform-on-disk per the user spec
     * ("conic distribution, stochastic, no grid jitter pattern").  When an
     * aperture-stop BLOCKER group is registered, rays additionally must
     * pass an explicit BVH/triangle test against the stop's tris, so blade
     * polygon shape is honoured automatically without separate blade math.
     *
     * Vectorisation: outer (sensor_group × pixel) loop is OpenMP-parallel.
     * Inner per-aperture-sample loop runs serially per pixel; each pixel's
     * write region in out_records is computed up-front so threads never
     * race on rec_count.
     *
     * Encoding (hard rule: no new EndpointRecord fields):
     *   subpath_id   = py * n_px + px
     *   vertex_index = aperture sample index
     *   group_id     = sensor group id
     * ────────────────────────────────────────────────────────────────── */
    {
        std::mt19937_64 cone_rng(static_cast<uint64_t>(seed) * 11400714819323198485ULL
                                 + 9876543210123456789ULL);
        for (size_t g = 0; g < st->tri_groups.size(); ++g) {
            const TriGroupDesc& gd = st->tri_groups[g];
            if (!(gd.role_bits & TRI_GROUP_ROLE_SENSOR)) continue;
            if (gd.sample_policy != TRI_GROUP_SAMPLE_PIXEL_CONE) continue;
            if (g >= st->tri_group_has_camera.size() ||
                !st->tri_group_has_camera[g]) continue;
            const CameraSensorDesc& cam = st->tri_group_camera[g];
            if (cam.n_px <= 0 || cam.n_py <= 0 ||
                cam.n_aperture_samples <= 0) continue;

            /* Build orthonormal sensor basis. */
            V3d cpos(cam.pos[0], cam.pos[1], cam.pos[2]);
            V3d cfwd(cam.fwd[0], cam.fwd[1], cam.fwd[2]);
            V3d cup (cam.up[0],  cam.up[1],  cam.up[2]);
            cfwd.normalize();
            V3d cright = cfwd.cross(cup).normalized();
            cup        = cright.cross(cfwd).normalized();
            V3d sensor_origin =
                cpos
                - 0.5 * cam.sensor_w_m * cright
                - 0.5 * cam.sensor_h_m * cup;
            V3d aperture_centre = cpos + cam.focal_m * cfwd;

            /* Optional BLOCKER stop group for blade-shape honoring. */
            const std::vector<int>* stop_idxs = nullptr;
            if (cam.aperture_stop_group_id >= 0 &&
                cam.aperture_stop_group_id < (int)st->tri_group_indices.size())
                stop_idxs = &st->tri_group_indices[cam.aperture_stop_group_id];

            const int n_px = cam.n_px;
            const int n_py = cam.n_py;
            const int n_ap = cam.n_aperture_samples;
            const double pix_w = cam.sensor_w_m / std::max(1, n_px);
            const double pix_h = cam.sensor_h_m / std::max(1, n_py);

            /* Reserve output capacity per pixel: n_ap * n_bands records,
             * but only if we have room.  Truncate at out_cap. */
            const long long total_pixels = (long long)n_px * n_py;
            const long long recs_per_px  = (long long)n_ap * n_bands;
            int sensor_gid = (int)g;

            /* Per-pixel parallel.  Each pixel computes its own jitter
             * stream from (seed, px, py) for determinism + reproducibility. */
            #pragma omp parallel
            {
                std::mt19937_64 trng;
                VXcd amp_local(n_bands);
                #pragma omp for schedule(dynamic, 8)
                for (long long pi = 0; pi < total_pixels; ++pi) {
                    int py = (int)(pi / n_px);
                    int px = (int)(pi - (long long)py * n_px);

                    /* Deterministic per-pixel RNG seed (decoupled from the
                     * shared cone_rng so threads don't race). */
                    uint64_t s = (uint64_t)seed * 0x9E3779B97F4A7C15ULL
                               + (uint64_t)pi * 0xBF58476D1CE4E5B9ULL
                               + 0x94D049BB133111EBULL;
                    trng.seed(s);
                    std::uniform_real_distribution<double> Up(0.0, 1.0);

                    V3d pix_pt = sensor_origin
                               + (px + 0.5) * pix_w * cright
                               + (py + 0.5) * pix_h * cup;

                    for (int ai = 0; ai < n_ap; ++ai) {
                        /* Stochastic uniform-on-disk aperture sample
                         * (concentric mapping from two uniform [0,1) draws —
                         * cheap, no grid pattern). */
                        double u1 = Up(trng), u2 = Up(trng);
                        double r  = std::sqrt(u1) * cam.aperture_radius_m;
                        double th = 2.0 * M_PI * u2;
                        V3d ap_pt = aperture_centre
                                  + r * std::cos(th) * cright
                                  + r * std::sin(th) * cup;

                        V3d dir = (ap_pt - pix_pt).normalized();
                        V3d pos = pix_pt;

                        /* Honour blade shape via BLOCKER tris if registered.
                         * Test the segment pix_pt → ap_pt against stop tris;
                         * if any tri is hit before ap_pt, drop this sample
                         * (blade occluded the ray). */
                        if (stop_idxs) {
                            double seg_len = (ap_pt - pix_pt).norm();
                            bool blocked = false;
                            for (int tri_id : *stop_idxs) {
                                if (tri_id < 0 || tri_id >= (int)st->tris.size()) continue;
                                double tt;
                                if (ray_triangle_hit(pos, dir, st->tris[(size_t)tri_id], tt)
                                    && tt > 1e-6 && tt < seg_len) {
                                    blocked = true; break;
                                }
                            }
                            if (blocked) continue;
                        }

                        /* Step into the scene from the aperture. */
                        pos = ap_pt + dir * (EPS * 200.0);
                        for (int b = 0; b < n_bands; ++b) amp_local[b] = cd(1.0, 0.0);
                        double path_len = (ap_pt - pix_pt).norm();

                        /* Wave-region dispatch at the aperture crossing.
                         * Any registered scale_context whose sphere
                         * contains the aperture sample point is allowed to
                         * transform (pos, dir, amp).  This is the
                         * "capacity to wave transform the aperture" hook:
                         * a WAVE_HELMHOLTZ region wrapping the stop will
                         * apply Fresnel quadratic phase per band; a
                         * THIN_LENS_TRANSFORM region steers the ray.
                         * Iteration is smallest-radius-first (vector is
                         * pre-sorted at registration). */
                        for (const RtScaleContext& ctx : st->scale_contexts) {
                            V3d c(ctx.center[0], ctx.center[1], ctx.center[2]);
                            if ((ap_pt - c).norm() <= ctx.radius) {
                                dispatch_scale_context_entry(*st, ctx, pos, dir, amp_local);
                            }
                        }

                        /* First-hit trace.  We record the first opaque
                         * surface hit; no bounces in PIXEL_CONE mode (the
                         * forward EMISSIVE path provides scene illumination
                         * — sensor rays exist to register *what they see*,
                         * and the bidirectional join is downstream). */
                        double t_hit = 1e18; int hit_tri = -1;
                        if (has_bvh) {
                            V3d inv = dir.cwiseInverse();
                            bvh_query(st->bvh_nodes, st->bvh_tri_ids, st->tris,
                                      pos, dir, inv, t_hit, hit_tri);
                        } else {
                            for (size_t ti = 0; ti < st->tris.size(); ++ti) {
                                double tt;
                                if (ray_triangle_hit(pos, dir, st->tris[ti], tt) && tt < t_hit) {
                                    t_hit = tt; hit_tri = (int)ti;
                                }
                            }
                        }
                        if (hit_tri < 0) continue;
                        V3d hit_pos = pos + t_hit * dir;
                        double total_path = path_len + t_hit;

                        for (int b = 0; b < n_bands; ++b) {
                            double k = st->k_real[b];
                            double atten = std::exp(-st->atmo_abs[b] * t_hit);
                            double spread = 1.0 / (1.0 + total_path);
                            amp_local[b] *= std::polar(atten * spread, -k * t_hit);
                        }

                        V3d hit_n = st->tris[(size_t)hit_tri].normal;
                        double cos_theta = std::abs(dir.dot(hit_n));
                        uint32_t my_subpath = (uint32_t)pi;

                        /* Emit n_bands records — under critical so the
                         * shared rec_count stays consistent.  Records past
                         * out_cap are dropped silently (caller widens cap). */
                        #pragma omp critical(pixel_cone_emit)
                        {
                            for (int b = 0; b < n_bands; ++b) {
                                if (rec_count >= out_cap) break;
                                EndpointRecord& E = out_records[rec_count++];
                                E.subpath_id   = my_subpath;
                                E.band_id      = (uint32_t)b;
                                E.group_id     = sensor_gid;
                                E.vertex_index = ai;
                                E.pos[0] = (float)hit_pos.x();
                                E.pos[1] = (float)hit_pos.y();
                                E.pos[2] = (float)hit_pos.z();
                                E.pathlen_m = (float)total_path;
                                E.dir[0] = (float)dir.x();
                                E.dir[1] = (float)dir.y();
                                E.dir[2] = (float)dir.z();
                                E.pdf       = 1.0f / (float)n_ap;
                                E.amp_re    = (float)amp_local[b].real();
                                E.amp_im    = (float)amp_local[b].imag();
                                E.cos_theta = (float)cos_theta;
                                E._pad      = 0.0f;
                            }
                        }
                    }
                }
            } /* omp parallel */
        }
    }

    *out_count = rec_count;
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
    /* Treat the region as a thin lens centred at ctx.center, optical axis
     * along the *current ray direction* (i.e. the lens auto-aligns to the
     * incoming ray — a deliberate simplification appropriate for "this is
     * a thin-lens region you can opt into" rather than a full optical
     * bench).  payload[0] = focal length f in metres; <=0 = no-op. */
    if (!ctx.payload) return;
    const double* p = static_cast<const double*>(ctx.payload);
    double f = p[0];
    if (!(f > 0.0)) return;

    V3d centre(ctx.center[0], ctx.center[1], ctx.center[2]);
    V3d to_centre = centre - pos;
    /* Project incoming direction relative to lens normal = ray direction.
     * Standard thin-lens rule: a ray through the lens centre passes
     * undeflected; a ray parallel to the axis converges to the focal
     * point at distance f. */
    V3d lateral = to_centre - to_centre.dot(dir) * dir;
    double focal_pt_dist = f;
    V3d focal_pt = centre + focal_pt_dist * dir;
    /* New direction: from current pos toward focal_pt. */
    V3d new_dir = (focal_pt - pos).normalized();
    if (new_dir.norm() > 1e-9) dir = new_dir;
    /* Position is unchanged (thin lens has zero thickness). */
    (void)lateral; /* reserved for off-axis astigmatism extension          */
}

/**
 * apply_wave_aperture_transform — multiply per-band complex amplitude by
 * the wave-optical transfer of a thin aperture.
 *
 * This is the hook the user demanded: "there must be capacity to wave
 * transform the aperture."  When a ray crosses a WAVE_HELMHOLTZ region at
 * lateral position `lateral_offset` from the region centre, this function
 * applies the analytic Fresnel quadratic phase
 *
 *   amp[b] *= exp( -i · k_b · r² / (2 · f_eff) )
 *
 * where r = ‖lateral_offset‖ and f_eff is taken from the region's payload
 * (payload[0] = focal length / Fresnel scale).  When payload[0] <= 0 we
 * fall back to ctx.radius itself so a bare WAVE_HELMHOLTZ region still
 * produces *some* diffraction-style phase shift instead of being inert.
 *
 * Apodisation: an extra real attenuation w(r) = exp(-(r/R)²) is applied
 * so a small aperture suppresses high spatial frequencies smoothly — this
 * is the analytic stand-in for the full split-step solve performed when
 * a FieldGrid is bound (future extension via ctx.payload subtype).
 *
 * No bands are collapsed; the per-band wavenumber st.k_real[b] is honored.
 */
static inline void apply_wave_aperture_transform(
    const RayTracerState& st,
    const RtScaleContext& ctx,
    const V3d&            cross_pos,
    VXcd&                 amp)
{
    V3d centre(ctx.center[0], ctx.center[1], ctx.center[2]);
    V3d off = cross_pos - centre;
    double r2 = off.squaredNorm();
    if (r2 <= 0.0) return;

    double f_eff = ctx.radius;
    if (ctx.payload && ctx.payload_size_bytes >= (int)sizeof(double)) {
        const double* p = static_cast<const double*>(ctx.payload);
        if (p[0] > 0.0) f_eff = p[0];
    }
    if (!(f_eff > 0.0)) return;

    double R = (ctx.radius > 0.0) ? ctx.radius : f_eff;
    double atten = std::exp(-r2 / (R * R));   /* soft-edge apodisation     */

    int nb = (int)amp.size();
    for (int b = 0; b < nb && b < (int)st.k_real.size(); ++b) {
        double k_b   = st.k_real[b];
        double phase = -k_b * r2 / (2.0 * f_eff);
        amp[b] *= std::polar(atten, phase);
    }
}

/**
 * dispatch_scale_context_entry — call site invoked when a ray segment
 * crosses into (or originates inside) a registered scale context.  Switches
 * on ctx.context_kind and applies the appropriate transform to (pos, dir,
 * amp).  Kinds 3..6 are stub-passthroughs but the call site exists in
 * BOTH backends so we never have to retrofit them.
 *
 * Returns the bit pattern to OR into EndpointRecord-style flags so
 * downstream consumers can tell which kinds the ray actually entered.
 */
static inline uint32_t dispatch_scale_context_entry(
    const RayTracerState& st,
    const RtScaleContext& ctx,
    V3d& pos, V3d& dir, VXcd& amp)
{
    switch (ctx.context_kind) {
        case SCALE_CONTEXT_KIND_RAY:
            return 0u;
        case SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ:
            apply_wave_aperture_transform(st, ctx, pos, amp);
            return 1u << SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ;
        case SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM:
            apply_thin_lens_transform(ctx, pos, dir);
            return 1u << SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM;
        case SCALE_CONTEXT_KIND_THICK_LENS_WAVE:
        case SCALE_CONTEXT_KIND_SPLINE_SURFACE:
        case SCALE_CONTEXT_KIND_NEURAL_SURFACE:
        case SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC:
            /* Stub-passthrough: caller records entry; full impl pending. */
            return 1u << ctx.context_kind;
        default:
            return 0u;
    }
}

