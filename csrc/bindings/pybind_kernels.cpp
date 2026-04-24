/**
 * pybind_kernels.cpp — thin pybind11 wrapper around the serial kernel C API.
 *
 * Exposes RouterStepState as a Python class whose methods accept raw torch
 * Tensors (complex128, contiguous) and pass data_ptr() directly to the C
 * functions with zero copy.
 *
 * Also exposes PicardSCC for C-dispatch of the Picard fixed-point loop over
 * non-linear SCCs whose transforms are all in the C registry.
 */

#include "serial_kernel.h"
#include "transforms_api.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

/* ── Forward declarations for PicardSCC (defined in transforms/picard_step.cpp) */

struct PicardSCCState;

extern "C" {
PicardSCCState* picard_scc_create(
    int N, int n_edges,
    const int* src_idxs, const int* dst_idxs,
    const double* edge_w_re, const double* edge_w_im,
    const int* edge_sat_ids, const double* edge_knees,
    const int* node_tf_ids, const double* node_knees,
    int max_iterations, double convergence_tol);
void picard_scc_destroy(PicardSCCState* st);
int  picard_scc_step(PicardSCCState* st, const double* src,
                     const double* z_init, double* z_out);
void picard_scc_diagnostics(const PicardSCCState* st, int* last_iters_out);
}

/* ── helpers ─────────────────────────────────────────────────────────────── */

/* Raise on SK_* error codes. */
static void check(int rc, const char* where)
{
    if (rc == SK_OK) return;
    std::string msg = std::string(where) + ": ";
    switch (rc) {
    case SK_ERR_NULL_STATE:    msg += "null state handle";   break;
    case SK_ERR_DIM_MISMATCH:  msg += "dimension mismatch";  break;
    case SK_ERR_DIVERGED:      msg += "solver diverged";     break;
    default:                   msg += "unknown error " + std::to_string(rc);
    }
    throw std::runtime_error(msg);
}

/* Extract a read-only double pointer from a numpy array or buffer. */
static const double* ro_ptr(py::buffer b, const char* name)
{
    auto info = b.request();
    if (info.format != py::format_descriptor<double>::format())
        throw std::runtime_error(
            std::string(name) + ": expected float64 buffer");
    return static_cast<const double*>(info.ptr);
}

static double* rw_ptr(py::buffer b, const char* name)
{
    auto info = b.request();
    if (info.format != py::format_descriptor<double>::format())
        throw std::runtime_error(
            std::string(name) + ": expected float64 buffer");
    return static_cast<double*>(info.ptr);
}

/* ── Python class wrapping RouterStepState ──────────────────────────────── */

struct PyRouterStep
{
    RouterStepState* handle = nullptr;
    int N = 0;
    int B = 0;

    PyRouterStep(int N, int B,
                 py::buffer M_re,   py::buffer M_im,
                 py::buffer W_re,   py::buffer W_im,
                 py::list   delay_lengths,
                 py::buffer ring_W_re, py::buffer ring_W_im,
                 int max_iterations  = 64,
                 double conv_eps     = 1e-10,
                 double inf_thresh   = 1e6)
        : N(N), B(B)
    {
        std::vector<int> dlens;
        dlens.reserve(py::len(delay_lengths));
        for (auto& item : delay_lengths)
            dlens.push_back(item.cast<int>());

        handle = router_step_create(
            N, B,
            ro_ptr(M_re, "M_re"), ro_ptr(M_im, "M_im"),
            ro_ptr(W_re, "W_re"), ro_ptr(W_im, "W_im"),
            (int)dlens.size(), dlens.data(),
            ro_ptr(ring_W_re, "ring_W_re"), ro_ptr(ring_W_im, "ring_W_im"),
            max_iterations, conv_eps, inf_thresh);
        if (!handle)
            throw std::runtime_error("router_step_create: allocation failed");
    }

    ~PyRouterStep()
    {
        router_step_destroy(handle);
        handle = nullptr;
    }

    /* step(src: np.ndarray) -> np.ndarray
     * src: float64 array of shape (B, N, 2) or (B, N*2) — interleaved.
     * Returns same shape/dtype.
     */
    py::array_t<double> step(py::buffer src_buf)
    {
        auto src_info = src_buf.request();
        py::array_t<double> out(src_info.size);
        check(router_step_step(handle,
            static_cast<const double*>(src_info.ptr),
            static_cast<double*>(out.mutable_data())),
            "router_step_step");
        out.resize({B, N, 2});
        return out;
    }

    /* run(src: np.ndarray) -> np.ndarray
     * src: float64 array of shape (T, B, N, 2) or flat equivalent.
     * T is inferred from the total element count.
     */
    py::array_t<double> run(py::buffer src_buf)
    {
        auto src_info = src_buf.request();
        int T = (int)(src_info.size / (B * N * 2));
        if (T <= 0)
            throw std::runtime_error("router_step_run: empty input buffer");
        py::array_t<double> out(src_info.size);
        check(router_step_run(handle,
            static_cast<const double*>(src_info.ptr),
            static_cast<double*>(out.mutable_data()),
            T),
            "router_step_run");
        out.resize({T, B, N, 2});
        return out;
    }

    void resize_batch(int new_B)
    {
        check(router_step_resize_batch(handle, new_B), "resize_batch");
        B = new_B;
    }

    void reset() { router_step_reset(handle); }

    py::dict diagnostics() const
    {
        int iters = 0, sat = 0;
        router_step_diagnostics(handle, &iters, &sat);
        py::dict d;
        d["convergence_iters"] = iters;
        d["saturated"]         = (bool)sat;
        return d;
    }
};

/* ── PicardSCC wrapper ───────────────────────────────────────────────────── */

struct PyPicardSCC
{
    PicardSCCState* handle = nullptr;
    int N = 0;

    PyPicardSCC(int N_,
                py::array_t<int>    src_idxs,
                py::array_t<int>    dst_idxs,
                py::array_t<double> edge_w_re,
                py::array_t<double> edge_w_im,
                py::array_t<int>    edge_sat_ids,
                py::array_t<double> edge_knees,
                py::array_t<int>    node_tf_ids,
                py::array_t<double> node_knees,
                int    max_iterations,
                double convergence_tol)
        : N(N_)
    {
        auto si  = src_idxs.request();
        auto di  = dst_idxs.request();
        auto wre = edge_w_re.request();
        auto wim = edge_w_im.request();
        auto esi = edge_sat_ids.request();
        auto ek  = edge_knees.request();
        auto nti = node_tf_ids.request();
        auto nk  = node_knees.request();

        int n_edges = static_cast<int>(si.size);

        handle = picard_scc_create(
            N_, n_edges,
            static_cast<const int*>(si.ptr),
            static_cast<const int*>(di.ptr),
            static_cast<const double*>(wre.ptr),
            static_cast<const double*>(wim.ptr),
            static_cast<const int*>(esi.ptr),
            static_cast<const double*>(ek.ptr),
            static_cast<const int*>(nti.ptr),
            static_cast<const double*>(nk.ptr),
            max_iterations, convergence_tol);

        if (!handle)
            throw std::runtime_error("picard_scc_create: allocation failed");
    }

    ~PyPicardSCC() { picard_scc_destroy(handle); handle = nullptr; }

    /* step(src, z_init) -> z_out
     * src, z_init: float64 arrays of 2*N doubles (interleaved complex128).
     * Returns a new float64 array of 2*N doubles.
     */
    py::array_t<double> step(py::buffer src_buf, py::buffer z_init_buf)
    {
        auto si = src_buf.request();
        auto zi = z_init_buf.request();
        if (si.format != py::format_descriptor<double>::format() ||
            zi.format != py::format_descriptor<double>::format())
            throw std::runtime_error("PicardSCC.step: expected float64 buffers");
        if (si.size != 2 * N || zi.size != 2 * N)
            throw std::runtime_error("PicardSCC.step: buffer size mismatch");

        py::array_t<double> out(2 * N);
        int rc = picard_scc_step(handle,
                                 static_cast<const double*>(si.ptr),
                                 static_cast<const double*>(zi.ptr),
                                 static_cast<double*>(out.mutable_data()));
        if (rc != 0)
            throw std::runtime_error("picard_scc_step failed: rc=" + std::to_string(rc));
        return out;
    }

    py::dict diagnostics() const
    {
        int iters = 0;
        picard_scc_diagnostics(handle, &iters);
        py::dict d;
        d["last_iters"] = iters;
        return d;
    }
};

/* ── Module definition ───────────────────────────────────────────────────── */

PYBIND11_MODULE(_spectral_kernels, m)
{
    m.doc() = "Spectral-analyzer C serial kernel extensions";

    py::class_<PyRouterStep>(m, "RouterStep",
        R"doc(
Stateful per-sample complex-graph router daemon.

Compile once from pre-computed matrices (M, W_lin, ring_W per delay group),
then call step() per sample or run() for a whole chunk.  No Python overhead
inside the hot loop.

Parameters
----------
N : int
    Number of graph nodes.
B : int
    Batch size (independent instances sharing M).
M_re, M_im : buffer (N, N) float64
    Real and imaginary parts of the solve matrix (I − W_lin)⁻¹.
W_re, W_im : buffer (N, N) float64
    Real and imaginary parts of the zero-delay weight matrix W_lin.
delay_lengths : list[int]
    Sample depth for each ring-buffer group.
ring_W_re, ring_W_im : buffer (n_delays, N, N) float64
    Weight matrices for each delay group, row-major.
max_iterations : int, default 64
    Iteration cap for saturating-edge fixed-point solve.
convergence_eps : float, default 1e-10
convergence_eps : float, default 1e6
)doc")
        .def(py::init<int,int,
                      py::buffer, py::buffer,
                      py::buffer, py::buffer,
                      py::list,
                      py::buffer, py::buffer,
                      int, double, double>(),
             py::arg("N"), py::arg("B"),
             py::arg("M_re"),   py::arg("M_im"),
             py::arg("W_re"),   py::arg("W_im"),
             py::arg("delay_lengths"),
             py::arg("ring_W_re"), py::arg("ring_W_im"),
             py::arg("max_iterations")   = 64,
             py::arg("convergence_eps")  = 1e-10,
             py::arg("infinity_threshold") = 1e6)
        .def("step",         &PyRouterStep::step,         py::arg("src"))
        .def("run",          &PyRouterStep::run,          py::arg("src"))
        .def("resize_batch", &PyRouterStep::resize_batch, py::arg("new_B"))
        .def("reset",        &PyRouterStep::reset)
        .def("diagnostics",  &PyRouterStep::diagnostics)
        .def_readonly("N", &PyRouterStep::N)
        .def_readonly("B", &PyRouterStep::B);

    py::class_<PyPicardSCC>(m, "PicardSCC",
        R"doc(
C-side Picard fixed-point solver for one non-linear SCC.

Compile once from pre-computed edge weights and transform IDs, then call
step() per solver tick.  The full Picard iteration runs in C with no Python
GIL or PyTorch overhead.  Only valid when all transforms in the SCC are in
the C registry (CTransformID).

Parameters
----------
N : int
    Number of nodes in this SCC.
src_idxs, dst_idxs : int32 arrays of length n_edges
    Source and destination node indices for each edge.
edge_w_re, edge_w_im : float64 arrays of length n_edges
    Pre-multiplied complex edge weights (weight * masks * acd + crosstalk).
edge_sat_ids : int32 array of length n_edges
    CTransformID per edge (saturation policy).
edge_knees : float64 array of length n_edges
    Knee parameter per edge.
node_tf_ids : int32 array of length N
    CTransformID per node.
node_knees : float64 array of length N
    Knee parameter per node.
max_iterations : int
    Picard iteration cap.
convergence_tol : float
    |z_new - z|_inf threshold for convergence.
)doc")
        .def(py::init<int,
                      py::array_t<int>,    py::array_t<int>,
                      py::array_t<double>, py::array_t<double>,
                      py::array_t<int>,    py::array_t<double>,
                      py::array_t<int>,    py::array_t<double>,
                      int, double>(),
             py::arg("N"),
             py::arg("src_idxs"),    py::arg("dst_idxs"),
             py::arg("edge_w_re"),   py::arg("edge_w_im"),
             py::arg("edge_sat_ids"), py::arg("edge_knees"),
             py::arg("node_tf_ids"), py::arg("node_knees"),
             py::arg("max_iterations")  = 64,
             py::arg("convergence_tol") = 1e-10)
        .def("step",        &PyPicardSCC::step,        py::arg("src"), py::arg("z_init"))
        .def("diagnostics", &PyPicardSCC::diagnostics)
        .def_readonly("N",  &PyPicardSCC::N);
}
