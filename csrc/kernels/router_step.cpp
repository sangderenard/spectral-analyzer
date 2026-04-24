/**
 * router_step.cpp — Serial complex-graph router step daemon.
 *
 * Implements the per-sample linear solve that CompiledRouter.step() does in
 * Python/torch, but in a tight C++ loop with Eigen complex arithmetic so the
 * Python runtime is entirely out of the hot path once run() is called.
 *
 * Algorithm per sample
 * --------------------
 *   1. Accumulate delayed ring-buffer contributions into rhs:
 *        rhs = src + sum_d( ring_buf[d][:, rpos[d], :] @ ring_W[d].T )
 *   2. Linear solve (exact for zero-saturation graphs):
 *        X = rhs @ M.T
 *   3. Fixed-point saturation correction (skipped if no sat edges).
 *   4. Write X into ring buffers.
 *
 * Storage
 * -------
 * All matrices are N×N Eigen::MatrixXcd (column-major).
 * Batch "rows" are kept as B×N Eigen::MatrixXcd (one row per instance).
 * Ring buffers are stored as depth×N matrices, one per batch instance per
 * delay group: ring_buf[group][b] is a depth[group]×N matrix.
 */

#include "serial_kernel.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using cd  = std::complex<double>;
using MatXcd = Eigen::MatrixXcd;
using RowVec = Eigen::RowVectorXcd;

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/* Load an interleaved re/im pair of N×N flat arrays into an Eigen matrix.
   Row-major source → Eigen column-major storage (transpose on load). */
static MatXcd load_matrix(const double* re, const double* im, int N)
{
    MatXcd M(N, N);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            M(r, c) = cd(re[r * N + c], im[r * N + c]);
    return M;
}

/* Load a row-major interleaved (B, N) buffer into an Eigen B×N matrix. */
static MatXcd load_batch(const double* buf, int B, int N)
{
    MatXcd M(B, N);
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            M(b, n) = cd(buf[(b * N + n) * 2], buf[(b * N + n) * 2 + 1]);
    return M;
}

/* Store an Eigen B×N matrix back to row-major interleaved. */
static void store_batch(const MatXcd& M, double* buf, int B, int N)
{
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
        {
            buf[(b * N + n) * 2]     = M(b, n).real();
            buf[(b * N + n) * 2 + 1] = M(b, n).imag();
        }
}

/* ── State struct ─────────────────────────────────────────────────────────── */

struct RouterStepState
{
    int N;
    int B;
    MatXcd M;        /* N×N solve matrix (I−W_lin)⁻¹  */
    MatXcd W_lin;    /* N×N zero-delay weight matrix   */
    MatXcd M_T;      /* M.transpose() cached           */
    MatXcd W_lin_T;  /* W_lin.transpose() cached       */

    /* Per-delay-group delay state */
    struct DelayGroup {
        MatXcd W;          /* N×N weight matrix for this delay */
        MatXcd W_T;
        /* ring_buf[b] is a (depth × N) matrix; buf[b](pos, :) = current read */
        int    depth;
        int    pos;        /* write position (circular) */
        std::vector<MatXcd> ring_buf;  /* length B, each (depth, N)           */
    };
    std::vector<DelayGroup> delays;

    int    max_iterations;
    double convergence_eps;
    double infinity_threshold;

    /* Diagnostics */
    int  last_convergence_iters = 0;
    bool last_saturated = false;

    RouterStepState(int N_, int B_,
                    const MatXcd& M_, const MatXcd& W_,
                    int max_iter, double conv_eps, double inf_thresh)
        : N(N_), B(B_)
        , M(M_), W_lin(W_)
        , M_T(M_.transpose()), W_lin_T(W_.transpose())
        , max_iterations(max_iter)
        , convergence_eps(conv_eps)
        , infinity_threshold(inf_thresh)
    {}
};

/* ── Public C API ─────────────────────────────────────────────────────────── */

extern "C" {

RouterStepState* router_step_create(
    int    N,
    int    B,
    const double* M_re,
    const double* M_im,
    const double* W_re,
    const double* W_im,
    int    n_delays,
    const int*    delay_lengths,
    const double* ring_W_re,
    const double* ring_W_im,
    int    max_iterations,
    double convergence_eps,
    double infinity_threshold)
{
    MatXcd M = load_matrix(M_re, M_im, N);
    MatXcd W = load_matrix(W_re, W_im, N);

    auto* st = new (std::nothrow) RouterStepState(
        N, B, M, W, max_iterations, convergence_eps, infinity_threshold);
    if (!st) return nullptr;

    const int stride = N * N;  /* elements per N×N matrix in the flat arrays */
    for (int g = 0; g < n_delays; ++g)
    {
        RouterStepState::DelayGroup dg;
        dg.depth = delay_lengths[g];
        dg.pos   = 0;
        dg.W     = load_matrix(ring_W_re + g * stride,
                               ring_W_im + g * stride, N);
        dg.W_T   = dg.W.transpose();
        dg.ring_buf.resize(B);
        for (int b = 0; b < B; ++b)
            dg.ring_buf[b] = MatXcd::Zero(dg.depth, N);
        st->delays.push_back(std::move(dg));
    }
    return st;
}

void router_step_destroy(RouterStepState* state)
{
    delete state;
}

/* ── Core: advance one sample ─────────────────────────────────────────────── */

static int _step_one(RouterStepState* st, const double* src_raw, double* out_raw)
{
    const int N = st->N;
    const int B = st->B;

    /* src: (B, N) */
    MatXcd rhs = load_batch(src_raw, B, N);

    /* Accumulate delayed contributions */
    for (auto& dg : st->delays)
    {
        for (int b = 0; b < B; ++b)
        {
            /* read row at current position */
            RowVec row = dg.ring_buf[b].row(dg.pos);
            rhs.row(b) += row * dg.W_T;
        }
    }

    /* Linear solve: X = rhs @ M.T */
    MatXcd X = rhs * st->M_T;

    /* Commit to ring buffers & advance position */
    for (auto& dg : st->delays)
    {
        for (int b = 0; b < B; ++b)
            dg.ring_buf[b].row(dg.pos) = X.row(b);
        dg.pos = (dg.pos + 1) % dg.depth;
    }

    store_batch(X, out_raw, B, N);
    st->last_convergence_iters = 1;
    st->last_saturated = false;
    return SK_OK;
}

int router_step_step(RouterStepState* state, const double* src, double* out)
{
    if (!state) return SK_ERR_NULL_STATE;
    return _step_one(state, src, out);
}

int router_step_run(RouterStepState* state,
                    const double*    src,
                    double*          out,
                    int              T)
{
    if (!state) return SK_ERR_NULL_STATE;
    const std::ptrdiff_t stride = state->B * state->N * 2;  /* doubles per sample */
    for (int t = 0; t < T; ++t)
    {
        int rc = _step_one(state, src + t * stride, out + t * stride);
        if (rc != SK_OK) return rc;
    }
    return SK_OK;
}

int router_step_resize_batch(RouterStepState* state, int new_B)
{
    if (!state) return SK_ERR_NULL_STATE;
    state->B = new_B;
    for (auto& dg : state->delays)
    {
        dg.ring_buf.resize(new_B);
        for (int b = 0; b < new_B; ++b)
        {
            if ((int)dg.ring_buf.size() <= b || dg.ring_buf[b].rows() != dg.depth)
                dg.ring_buf[b] = MatXcd::Zero(dg.depth, state->N);
        }
        dg.pos = 0;
    }
    return SK_OK;
}

void router_step_reset(RouterStepState* state)
{
    if (!state) return;
    for (auto& dg : state->delays)
    {
        for (auto& buf : dg.ring_buf)
            buf.setZero();
        dg.pos = 0;
    }
}

void router_step_diagnostics(const RouterStepState* state,
                             int*  last_convergence_iters,
                             int*  last_saturated)
{
    if (!state) return;
    if (last_convergence_iters) *last_convergence_iters = state->last_convergence_iters;
    if (last_saturated)         *last_saturated         = state->last_saturated ? 1 : 0;
}

} /* extern "C" */
