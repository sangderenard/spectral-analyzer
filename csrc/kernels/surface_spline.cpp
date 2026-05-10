/**
 * surface_spline.cpp — Eigen quadratic POLY_BARY surface fitter.
 *
 * See surface_spline.h for the full algorithm description.
 */
#include "surface_spline.h"
#include "thread_pool.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

/* ── small helpers ─────────────────────────────────────────────────────────── */

static constexpr double EPS_SP = 1.0e-12;

/* Return (u,v) barycentric coords of world point p projected onto triangle
 * defined by (v0, edge1=v1-v0, edge2=v2-v0, normal).
 * u,v may be outside [0,1]; caller decides whether to keep or skip. */
static inline bool project_to_bary(
    const Eigen::Vector3d& v0,
    const Eigen::Vector3d& edge1,
    const Eigen::Vector3d& edge2,
    const Eigen::Vector3d& p,
    double& u_out, double& v_out)
{
    Eigen::Vector3d q  = p - v0;
    double d00 = edge1.dot(edge1);
    double d01 = edge1.dot(edge2);
    double d11 = edge2.dot(edge2);
    double d20 = q.dot(edge1);
    double d21 = q.dot(edge2);
    double det = d00 * d11 - d01 * d01;
    if (std::abs(det) < EPS_SP) { u_out = 0.0; v_out = 0.0; return false; }
    u_out = (d11 * d20 - d01 * d21) / det;
    v_out = (d00 * d21 - d01 * d20) / det;
    return true;
}

/* Evaluate the quadratic basis at (u,v) → row vector [1, u, v, u², uv, v²]. */
static inline Eigen::Matrix<double,1,6> basis(double u, double v)
{
    Eigen::Matrix<double,1,6> b;
    b << 1.0, u, v, u*u, u*v, v*v;
    return b;
}

/* ── Neighbour adjacency ───────────────────────────────────────────────────── */

/* Build vertex → list-of-triangle-indices map for 1-ring lookup. */
static std::vector<std::vector<int>> build_vtx_to_tris(
    int n_verts, int n_tris, const int* tri_idx)
{
    std::vector<std::vector<int>> v2t(static_cast<size_t>(n_verts));
    for (int t = 0; t < n_tris; ++t) {
        for (int k = 0; k < 3; ++k) {
            int vi = tri_idx[static_cast<size_t>(t) * 3 + k];
            if (vi >= 0 && vi < n_verts)
                v2t[static_cast<size_t>(vi)].push_back(t);
        }
    }
    return v2t;
}

/* Collect 1-ring neighbours of triangle t (including t itself). */
/* ── Per-triangle spline fit ───────────────────────────────────────────────── */

/* Fit the 6 POLY_BARY coefficients for triangle t.
 * Uses the centroid of each 1-ring neighbour as a data sample.
 * Falls back gracefully when there are < 3 neighbours. */
static void fit_one_triangle(
    int t,
    const double* verts,
    int           n_verts,
    int           n_tris,
    const int*    tri_idx,
    const std::vector<std::vector<int>>& v2t,
    double        ridge_lambda,
    double*       out6)   /* pointer to float64[6] for this triangle */
{
    /* Silence unused parameter warning. */
    (void)n_tris;

    /* Zero out result in case of early return. */
    std::memset(out6, 0, 6 * sizeof(double));

    /* Triangle basis vectors. */
    auto vid = [&](int tri, int k) -> int {
        return tri_idx[static_cast<size_t>(tri) * 3 + k];
    };
    auto V = [&](int vi) -> Eigen::Vector3d {
        const double* p = verts + static_cast<size_t>(vi) * 3;
        return Eigen::Vector3d(p[0], p[1], p[2]);
    };

    int i0 = vid(t, 0), i1 = vid(t, 1), i2 = vid(t, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= n_verts || i1 >= n_verts || i2 >= n_verts) return;

    Eigen::Vector3d v0    = V(i0);
    Eigen::Vector3d edge1 = V(i1) - v0;
    Eigen::Vector3d edge2 = V(i2) - v0;
    Eigen::Vector3d raw_n = edge1.cross(edge2);
    double len_n = raw_n.norm();
    if (len_n < EPS_SP) return;
    Eigen::Vector3d n = raw_n / len_n;

    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Row6 = Eigen::Matrix<double, 1, 6>;

    Mat6 ATA = Mat6::Zero();
    Vec6 ATb = Vec6::Zero();
    auto add_row = [&](const Row6& row, double y) {
        ATA.noalias() += row.transpose() * row;
        ATb.noalias() += row.transpose() * y;
    };

    /* Self-centroid sample: delta = 0 at bary (1/3, 1/3) — anchor the fit. */
    add_row(basis(1.0 / 3.0, 1.0 / 3.0), 0.0);

    /* 1-ring neighbours (small vector + sort/unique avoids hash allocations). */
    std::vector<int> ring;
    ring.reserve(1
        + v2t[static_cast<size_t>(i0)].size()
        + v2t[static_cast<size_t>(i1)].size()
        + v2t[static_cast<size_t>(i2)].size());
    ring.push_back(t);
    for (int k = 0; k < 3; ++k) {
        int vi = vid(t, k);
        const auto& list = v2t[static_cast<size_t>(vi)];
        ring.insert(ring.end(), list.begin(), list.end());
    }
    std::sort(ring.begin(), ring.end());
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());

    for (int nb : ring) {
        if (nb == t) continue;

        /* Centroid of neighbour in world space. */
        int j0 = vid(nb, 0), j1 = vid(nb, 1), j2 = vid(nb, 2);
        if (j0 < 0 || j1 < 0 || j2 < 0 ||
            j0 >= n_verts || j1 >= n_verts || j2 >= n_verts) continue;

        Eigen::Vector3d c = (V(j0) + V(j1) + V(j2)) / 3.0;

        /* Project centroid onto the base triangle's barycentric frame. */
        double u = 0.0, v = 0.0;
        project_to_bary(v0, edge1, edge2, c, u, v);

        /* Cull wildly out-of-range points (large-angle neighbours). */
        if (u < -1.5 || u > 2.5 || v < -1.5 || v > 2.5) continue;

        /* Displacement along the base triangle's normal axis. */
        double delta = (c - v0).dot(n);

        add_row(basis(u, v), delta);
    }

    /* Also add vertex-normal weighted samples: for each vertex of t, sample
     * the average normal of surrounding triangles to steer curvature. */
    for (int k = 0; k < 3; ++k) {
        int vk = vid(t, k);
        const std::vector<int>& nb_tris = v2t[static_cast<size_t>(vk)];
        if (nb_tris.empty()) continue;

        /* Average normal of the 1-ring around this vertex. */
        Eigen::Vector3d avg_n(0.0, 0.0, 0.0);
        for (int nb : nb_tris) {
            int a = vid(nb, 0), b = vid(nb, 1), c_idx = vid(nb, 2);
            if (a < 0 || b < 0 || c_idx < 0 ||
                a >= n_verts || b >= n_verts || c_idx >= n_verts) continue;
            Eigen::Vector3d e1 = V(b)     - V(a);
            Eigen::Vector3d e2 = V(c_idx) - V(a);
            Eigen::Vector3d fn = e1.cross(e2);
            double flen = fn.norm();
            if (flen > EPS_SP)
                avg_n += fn / flen;
        }
        if (avg_n.norm() < EPS_SP) continue;
        avg_n.normalize();

        /* Barycentric coords of the k-th vertex. */
        double u_vtx = (k == 1) ? 1.0 : 0.0;
        double v_vtx = (k == 2) ? 1.0 : 0.0;

        /* The gradient of delta along u and v at this vertex is constrained
         * by the tangent component of (avg_n - n).  This is a soft linearised
         * normal-matching constraint added as an additional row.
         * Tangent deviation dN = avg_n - dot(avg_n,n)*n (tangent plane component).
         * The normal-axis displacement gradient:
         *   d(delta)/du ≈ dN · edge1  (since u moves along edge1)
         *   d(delta)/dv ≈ dN · edge2
         * encoded as: [0, 1, 0, 2*u, v, 0] c = dN·edge1  at (u,v)
         *             [0, 0, 1, 0, u, 2*v] c = dN·edge2  at (u,v)
         */
        Eigen::Vector3d dN = avg_n - avg_n.dot(n) * n;
        double du_rhs = dN.dot(edge1);
        double dv_rhs = dN.dot(edge2);

        double u = u_vtx, v = v_vtx;

        Row6 du_row, dv_row;
        du_row << 0.0, 1.0, 0.0, 2.0*u,     v, 0.0;
        dv_row << 0.0, 0.0, 1.0,     0.0,    u, 2.0*v;

        add_row(du_row, du_rhs);
        add_row(dv_row, dv_rhs);
    }

    /* Optional Tikhonov ridge on the curvature terms (columns 3-5). */
    if (ridge_lambda > 0.0) {
        for (int j = 3; j < 6; ++j)
            ATA(j, j) += ridge_lambda;
    }

    /* Solve fixed-size normal equations; QR fallback for rank-deficient cases. */
    Eigen::LDLT<Mat6> ldlt(ATA);
    Vec6 coeffs = Vec6::Zero();
    if (ldlt.info() == Eigen::Success) {
        coeffs = ldlt.solve(ATb);
    } else {
        coeffs = ATA.colPivHouseholderQr().solve(ATb);
    }
    for (int j = 0; j < 6; ++j) {
        double v = coeffs(j);
        out6[j] = std::isfinite(v) ? v : 0.0;
    }
}

/* ── Public C API ─────────────────────────────────────────────────────────── */

extern "C" SS_API int surface_spline_fit(
    int            n_verts,
    const double*  verts_xyz,
    int            n_tris,
    const int*     tri_indices,
    const int*     tri_subset,
    int            n_subset,
    double         ridge_lambda,
    int            n_threads,
    double*        out_coeffs)
{
    if (!verts_xyz || !tri_indices || !out_coeffs) return SK_ERR_NULL_STATE;
    if (n_verts <= 0 || n_tris <= 0)               return SK_ERR_NULL_STATE;

    /* Zero-fill entire output so non-subset triangles are safe to read. */
    std::memset(out_coeffs, 0, static_cast<size_t>(n_tris) * 6 * sizeof(double));

    /* Build adjacency once (shared across all threads; read-only after). */
    const auto v2t = build_vtx_to_tris(n_verts, n_tris, tri_indices);

    /* Decide which triangle indices to process. */
    std::vector<int> work_list;
    if (tri_subset && n_subset > 0) {
        work_list.assign(tri_subset, tri_subset + n_subset);
    } else {
        work_list.resize(static_cast<size_t>(n_tris));
        for (int i = 0; i < n_tris; ++i) work_list[static_cast<size_t>(i)] = i;
    }
    const size_t n_work = work_list.size();

    /* Parallelise across the work list. */
    ThreadPool pool(static_cast<size_t>(n_threads > 0 ? n_threads : 0));
    ThreadPool::parallel_for(pool, size_t(0), n_work, [&](size_t wi) {
        int t = work_list[wi];
        if (t < 0 || t >= n_tris) return;
        fit_one_triangle(
            t,
            verts_xyz, n_verts, n_tris, tri_indices,
            v2t,
            ridge_lambda,
            out_coeffs + static_cast<ptrdiff_t>(t) * 6);
    });

    return SK_OK;
}

extern "C" SS_API int surface_spline_eval_normal(
    double         u,
    double         v,
    const double*  coeffs_6,
    const double*  verts_xyz,
    const int*     tri_row,
    double*        out_normal)
{
    if (!coeffs_6 || !verts_xyz || !tri_row || !out_normal) return SK_ERR_NULL_STATE;

    /* Reconstruct triangle basis. */
    auto V3 = [&](int vi) -> Eigen::Vector3d {
        const double* p = verts_xyz + static_cast<ptrdiff_t>(vi) * 3;
        return Eigen::Vector3d(p[0], p[1], p[2]);
    };
    Eigen::Vector3d v0    = V3(tri_row[0]);
    Eigen::Vector3d edge1 = V3(tri_row[1]) - v0;
    Eigen::Vector3d edge2 = V3(tri_row[2]) - v0;
    Eigen::Vector3d raw_n = edge1.cross(edge2);
    double len_n = raw_n.norm();
    if (len_n < EPS_SP) {
        out_normal[0] = out_normal[2] = 0.0; out_normal[1] = 1.0;
        return SK_OK;
    }
    Eigen::Vector3d n = raw_n / len_n;

    /* Gradient of displacement. */
    /* c = [c0, cu, cv, cuu, cuv, cvv]
     * d(delta)/du = cu + 2*cuu*u + cuv*v
     * d(delta)/dv = cv + cuv*u   + 2*cvv*v        */
    const double* c  = coeffs_6;
    double dzdu = c[1] + 2.0 * c[3] * u + c[4] * v;
    double dzdv = c[2] + c[4] * u        + 2.0 * c[5] * v;

    Eigen::Vector3d t1 = edge1.normalized();
    Eigen::Vector3d t2 = edge2.normalized();

    /* Perturbed normal = normalise(n - dzdu*t1 - dzdv*t2). */
    Eigen::Vector3d pn = n - dzdu * t1 - dzdv * t2;
    double plen = pn.norm();
    if (plen < EPS_SP) pn = n;
    else               pn /= plen;

    out_normal[0] = pn.x();
    out_normal[1] = pn.y();
    out_normal[2] = pn.z();
    return SK_OK;
}
