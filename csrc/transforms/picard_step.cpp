/**
 * picard_step.cpp — Picard fixed-point solver for one non-linear SCC.
 *
 * Mirrors CyclicTensorBlock._apply_once() + the Picard loop in step(),
 * executing entirely in C with no Python GIL or PyTorch overhead per
 * iteration.  The kernel is dispatch-eligible only when every edge and node
 * transform in the SCC is in the C registry (ct_apply).
 *
 * Edge model
 * ----------
 * Each compiled edge carries a single pre-multiplied complex weight
 * (product of edge.weight, analog_complex_delay, src_mask, dst_mask,
 * activity_mask, and crosstalk_weight) plus a CTransformID + knee for
 * its saturation_policy.  The weight folding is done in c_transforms.py
 * before the state is handed to this kernel.
 *
 * Node model
 * ----------
 * Each node carries a CTransformID + knee applied to the accumulated
 * incoming sum after all edge contributions have been added.
 *
 * Algorithm per step (interp_mode == "none" only)
 * -----------------------------------------------
 *   Given warm-start z and external source src:
 *   for k in 0..max_iter-1:
 *       z_new[dst] = src[dst]
 *                  + sum_over_edges(sat(weight * z[src_idx]))  per dst
 *       z_new[n]   = node_transform(z_new[n])
 *       if max|z_new - z| < tol: break
 *       z = z_new
 *   return z_new
 */

#include "serial_kernel.h"
#include "transforms_api.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>

using cd = std::complex<double>;

/* ── Transform implementations (definition of ct_apply) ─────────────────── */

void ct_apply(CTransformID id, double knee, double* buf, int n)
{
    const double k = (knee > 1e-30) ? knee : 1.0;
    for (int i = 0; i < n; ++i)
    {
        double re  = buf[2 * i];
        double im  = buf[2 * i + 1];
        double mag = std::sqrt(re * re + im * im);

        switch (id)
        {
        case CT_IDENTITY:
            break;

        case CT_TANH:
            if (mag > 1e-300)
            {
                double s = k * std::tanh(mag / k) / mag;
                buf[2 * i]     = re * s;
                buf[2 * i + 1] = im * s;
            }
            break;

        case CT_SOFTCLIP:
            if (mag > 1e-300)
            {
                /* f(mag) = k*mag / (k + mag) */
                double s = k / (k + mag);
                buf[2 * i]     = re * s;
                buf[2 * i + 1] = im * s;
            }
            break;

        case CT_HARDCLIP:
            if (mag > k)
            {
                double s = k / mag;
                buf[2 * i]     = re * s;
                buf[2 * i + 1] = im * s;
            }
            break;
        }
    }
}

/* ── State struct ─────────────────────────────────────────────────────────── */

struct PicardEdge {
    int          src_idx;
    int          dst_idx;
    cd           weight;
    CTransformID sat_id;
    double       knee;
};

struct PicardSCCState {
    int N;
    std::vector<PicardEdge>   edges;
    std::vector<CTransformID> node_tf;
    std::vector<double>       node_knee;
    int    max_iter;
    double tol;
    /* Diagnostics */
    int last_iters;
};

/* ── Public C API ─────────────────────────────────────────────────────────── */

extern "C" {

PicardSCCState* picard_scc_create(
    int N,
    int n_edges,
    const int*    src_idxs,
    const int*    dst_idxs,
    const double* edge_w_re,
    const double* edge_w_im,
    const int*    edge_sat_ids,
    const double* edge_knees,
    const int*    node_tf_ids,
    const double* node_knees,
    int    max_iterations,
    double convergence_tol)
{
    auto* st = new (std::nothrow) PicardSCCState;
    if (!st) return nullptr;

    st->N        = N;
    st->max_iter = max_iterations;
    st->tol      = convergence_tol;
    st->last_iters = 0;

    st->edges.resize(n_edges);
    for (int i = 0; i < n_edges; ++i)
        st->edges[i] = { src_idxs[i], dst_idxs[i],
                         cd(edge_w_re[i], edge_w_im[i]),
                         static_cast<CTransformID>(edge_sat_ids[i]),
                         edge_knees[i] };

    st->node_tf.resize(N);
    st->node_knee.resize(N);
    for (int i = 0; i < N; ++i) {
        st->node_tf[i]   = static_cast<CTransformID>(node_tf_ids[i]);
        st->node_knee[i] = node_knees[i];
    }
    return st;
}

void picard_scc_destroy(PicardSCCState* st)
{
    delete st;
}

/**
 * Run one Picard fixed-point step.
 *
 * src:    (N,) interleaved complex128 — external input to this SCC.
 * z_init: (N,) interleaved complex128 — warm-start guess.
 * z_out:  (N,) interleaved complex128 — converged output (may alias z_init).
 *
 * Returns SK_OK on convergence or when max_iter is exhausted without
 * divergence.  SK_ERR_DIVERGED is reserved for future NaN/Inf detection.
 */
int picard_scc_step(
    PicardSCCState* st,
    const double*   src,
    const double*   z_init,
    double*         z_out)
{
    if (!st) return SK_ERR_NULL_STATE;
    const int N = st->N;

    std::vector<double> z_cur(2 * N);
    std::vector<double> z_new(2 * N);
    std::memcpy(z_cur.data(), z_init, 2 * N * sizeof(double));

    int iter = 0;
    for (; iter < st->max_iter; ++iter)
    {
        /* Seed z_new from external source */
        std::memcpy(z_new.data(), src, 2 * N * sizeof(double));

        /* Accumulate edge contributions */
        for (const auto& e : st->edges)
        {
            cd z_src(z_cur[2 * e.src_idx], z_cur[2 * e.src_idx + 1]);
            cd contrib = e.weight * z_src;

            double buf[2] = { contrib.real(), contrib.imag() };
            ct_apply(e.sat_id, e.knee, buf, 1);
            z_new[2 * e.dst_idx]     += buf[0];
            z_new[2 * e.dst_idx + 1] += buf[1];
        }

        /* Per-node transform */
        for (int n = 0; n < N; ++n)
            ct_apply(st->node_tf[n], st->node_knee[n], z_new.data() + 2 * n, 1);

        /* Convergence: max|z_new - z_cur|_inf */
        double residual = 0.0;
        for (int n = 0; n < N; ++n)
        {
            double dr = z_new[2*n]   - z_cur[2*n];
            double di = z_new[2*n+1] - z_cur[2*n+1];
            double d  = std::sqrt(dr*dr + di*di);
            if (d > residual) residual = d;
        }

        std::swap(z_cur, z_new);
        if (residual < st->tol) { ++iter; break; }
    }

    st->last_iters = iter;
    std::memcpy(z_out, z_cur.data(), 2 * N * sizeof(double));
    return SK_OK;
}

void picard_scc_diagnostics(const PicardSCCState* st, int* last_iters_out)
{
    if (st && last_iters_out) *last_iters_out = st->last_iters;
}

} /* extern "C" */
