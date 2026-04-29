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

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using cd   = std::complex<double>;
using V3d  = Eigen::Vector3d;
using VXcd = Eigen::VectorXcd;

static constexpr double TWO_PI     = 2.0 * M_PI;
static constexpr double EPS        = 1e-9;
static constexpr int    BVH_LEAF_MAX = 4;   /* triangles per BVH leaf */

/* ── Geometry ──────────────────────────────────────────────────────────────── */

struct Triangle {
    V3d v0;
    V3d edge1;   /* v1 - v0 (precomputed for Möller-Trumbore) */
    V3d edge2;   /* v2 - v0 */
    V3d normal;  /* outward unit normal */
    double diffusion;
    VXcd refl;   /* complex reflectance per band */
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
    std::vector<Triangle> tris;
    std::vector<BVHNode>  bvh_nodes;
    std::vector<int>      bvh_tri_ids;
    int                   n_bands = 0;
    Eigen::VectorXd       k_real;      /* 2π f_n / c  (wavenumber) */
    Eigen::VectorXd       atmo_abs;    /* Np/m per band */
};

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

                    amp[b]     = new_amp * tri.refl[b];
                    double a   = std::abs(amp[b]);
                    if (a > max_abs) max_abs = a;
                }

                if (max_abs < min_amplitude) break;

                pos = hit_pos + tri.normal * (EPS * 100.0);

                if (U(rng) < tri.diffusion) {
                    cur_dir = cosine_hemisphere(tri.normal, rng);
                } else {
                    cur_dir = (cur_dir
                               - 2.0 * cur_dir.dot(tri.normal) * tri.normal)
                              .normalized();
                    if (cur_dir.dot(tri.normal) < 0.0)
                        cur_dir = cosine_hemisphere(tri.normal, rng);
                }

                path_len += t_min;
            }
            next_ray:;
        }
    }
}

/* ── C API ──────────────────────────────────────────────────────────────────── */

RayTracerState* ray_tracer_create(
    int           n_tri,
    const double* verts,
    const double* normals,
    const double* refl_re,
    const double* refl_im,
    const double* diffusion,
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

    for (int b = 0; b < n_bands; ++b)
        st->k_real[b] = TWO_PI * freq_hz[b] / speed_m_s;

    st->tris.resize(static_cast<size_t>(n_tri));
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

        tri.diffusion = diffusion[i];

        /* Complex reflectances */
        tri.refl.resize(n_bands);
        const double* rre = refl_re + static_cast<size_t>(i) * n_bands;
        const double* rim = refl_im + static_cast<size_t>(i) * n_bands;
        for (int b = 0; b < n_bands; ++b)
            tri.refl[b] = cd(rre[b], rim[b]);

        /* AABB for BVH build */
        tri_aabbs[static_cast<size_t>(i)].expand(v0);
        tri_aabbs[static_cast<size_t>(i)].expand(v1);
        tri_aabbs[static_cast<size_t>(i)].expand(v2);
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

    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int /*si*/, int /*bounce*/, int b, cd new_amp,
            const V3d& /*p0*/, const V3d& p1, double /*total_path*/) -> bool
        {
            V3d v = p1 - cam_p;
            double depth = v.dot(cam_f);
            if (depth <= EPS) return true;  /* behind camera */

            double x_img = v.dot(cam_r);
            double y_img = v.dot(cam_u);

            /* NDC in [-1, 1] */
            double ndc_x =  x_img / (depth * tan_half_h);
            double ndc_y = -y_img / (depth * tan_half_v);  /* flip Y: row 0 = top */

            /* Pixel coordinates (floor). */
            int px = static_cast<int>((ndc_x + 1.0) * 0.5 * width);
            int py = static_cast<int>((ndc_y + 1.0) * 0.5 * height);

            if (px < 0 || px >= width || py < 0 || py >= height)
                return true;

            size_t idx = static_cast<size_t>(b) * (height * width)
                       + static_cast<size_t>(py) * width
                       + static_cast<size_t>(px);
            out_image[idx] += static_cast<float>(std::abs(new_amp));
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

    trace_rays(
        *st,
        n_sources, src_pos, src_dir, src_directivity,
        n_rays, max_bounces, min_amplitude,
        rng, abort,
        [&](int si, int bounce, int b, cd new_amp,
            const V3d& p0, const V3d& p1, double total_path) -> bool
        {
            if (out_segs && count < out_cap) {
                write_segment(out_segs, count, out_cap,
                              p0, p1, si, bounce, b,
                              new_amp, total_path - (p1 - p0).norm());
            }

            V3d v = p1 - cam_p;
            double depth = v.dot(cam_f);
            if (depth > EPS) {
                double x_img = v.dot(cam_r);
                double y_img = v.dot(cam_u);
                double ndc_x =  x_img / (depth * tan_half_h);
                double ndc_y = -y_img / (depth * tan_half_v);
                int px = static_cast<int>((ndc_x + 1.0) * 0.5 * width);
                int py = static_cast<int>((ndc_y + 1.0) * 0.5 * height);
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    size_t idx = static_cast<size_t>(b) * (height * width)
                               + static_cast<size_t>(py) * width
                               + static_cast<size_t>(px);
                    out_image[idx] += static_cast<float>(std::abs(new_amp));
                }
            }
            return true;
        });

    *out_count = count;
    return SK_OK;
}
