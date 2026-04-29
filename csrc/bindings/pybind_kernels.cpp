/**
 * pybind_kernels.cpp — thin pybind11 wrapper around the serial kernel C API.
 *
 * Exposes RouterStepState as a Python class whose methods accept raw torch
 * Tensors (complex128, contiguous) and pass data_ptr() directly to the C
 * functions with zero copy.
 *
 * Also exposes PicardSCC for C-dispatch of the Picard fixed-point loop over
 * non-linear SCCs whose transforms are all in the C registry.
 *
 * Also exposes RayTracer for the complex spectral 3-D ray tracer.
 */

#include "serial_kernel.h"
#include "transforms_api.h"
#include "ray_tracer.h"
#include "rt_field_solver.h"
#include "acoustic_fdtd.h"
#include "acoustic_coevolver.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

/* ── RayTracer wrapper ───────────────────────────────────────────────────── */

struct PyRayTracer
{
    RayTracerState* handle  = nullptr;
    int             _n_bands = 0;
    int             _n_tris  = 0;

    PyRayTracer(int                     n_tri,
                py::array_t<double>     verts,
                py::array_t<double>     normals,
                py::array_t<double>     refl_re,
                py::array_t<double>     refl_im,
                py::array_t<double>     diffusion,
                py::array_t<double>     freq_hz,
                double                  speed_m_s,
                py::array_t<double>     atmo_abs)
    {
        auto iv  = verts    .request();
        auto in_ = normals  .request();
        auto ire = refl_re  .request();
        auto iim = refl_im  .request();
        auto id  = diffusion.request();
        auto ifh = freq_hz  .request();
        auto iaa = atmo_abs .request();

        _n_bands = static_cast<int>(ifh.size);
        _n_tris  = n_tri;

        handle = ray_tracer_create(
            n_tri,
            static_cast<const double*>(iv .ptr),
            static_cast<const double*>(in_.ptr),
            static_cast<const double*>(ire.ptr),
            static_cast<const double*>(iim.ptr),
            static_cast<const double*>(id .ptr),
            _n_bands,
            static_cast<const double*>(ifh.ptr),
            speed_m_s,
            static_cast<const double*>(iaa.ptr));

        if (!handle)
            throw std::runtime_error("ray_tracer_create: allocation failed");
    }

    ~PyRayTracer() { ray_tracer_destroy(handle); handle = nullptr; }

    /**
     * trace(src_pos, src_dir, src_directivity,
     *       n_rays, max_bounces, min_amplitude, seed, out_cap)
     *       -> np.ndarray float32, shape (N_segs, 12)
     *
     * Columns: [x0,y0,z0, x1,y1,z1, src_id, bounce, band, amp, phase, path_len]
     */
    py::array_t<float> trace(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        int     n_rays        = 256,
        int     max_bounces   = 8,
        double  min_amplitude = 0.005,
        uint32_t seed         = 42,
        int     out_cap       = -1)
    {
        auto ip  = src_pos        .request();
        auto id_ = src_dir        .request();
        auto idv = src_directivity.request();

        int n_sources = static_cast<int>(idv.size);

        /* Default capacity: fixed vis budget, never scales with n_rays. */
        if (out_cap <= 0)
            out_cap = 2'000'000;

        /* Allocate flat float32 output buffer. */
        py::array_t<float> out(
            static_cast<py::ssize_t>(out_cap) * RT_FLOATS_PER_SEG);

        int n_written = 0;
        int rc = ray_tracer_trace(
            handle,
            n_sources,
            static_cast<const double*>(ip .ptr),
            static_cast<const double*>(id_.ptr),
            static_cast<const double*>(idv.ptr),
            n_rays, max_bounces, min_amplitude,
            seed,
            out.mutable_data(),
            out_cap,
            &n_written);

        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_trace failed: rc=" + std::to_string(rc));

        /* Trim to actual count and reshape to (N, 12). */
        out.resize({static_cast<py::ssize_t>(n_written),
                    static_cast<py::ssize_t>(RT_FLOATS_PER_SEG)});
        return out;
    }

    int n_bands() const { return _n_bands; }
    int n_tris()  const {
        /* The handle carries the geometry; expose triangle count so Python can
         * pre-allocate flux buffers of the right size. */
        if (!handle) return 0;
        /* RayTracerState is opaque — query via the tri_areas size.
         * We cache it at construction time. */
        return _n_tris;
    }

    /**
     * trace_surface(src_pos, src_dir, src_directivity,
     *               n_rays, max_bounces, min_amplitude, seed, out_cap)
     *   -> dict with keys:
     *       'segs'     : float32 (N_segs, 12) — same layout as trace()
     *       'direct'   : float32 (n_tri, n_bands) — direct irradiance
     *       'indirect' : float32 (n_tri, n_bands) — indirect irradiance
     *
     * Runs a full ray trace and simultaneously accumulates per-triangle
     * irradiance (|A|² * cos_in / area) split by bounce depth.
     * Caller should zero-initialise if accumulating across multiple calls.
     */
    py::dict trace_surface(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        int      n_rays        = 256,
        int      max_bounces   = 8,
        double   min_amplitude = 0.005,
        uint32_t seed          = 42,
        int      out_cap       = -1)
    {
        auto ip  = src_pos        .request();
        auto id_ = src_dir        .request();
        auto idv = src_directivity.request();

        int n_sources = static_cast<int>(idv.size);

        if (out_cap <= 0)
            out_cap = 2'000'000;

        /* Segment buffer. */
        py::array_t<float> segs(
            static_cast<py::ssize_t>(out_cap) * RT_FLOATS_PER_SEG);

        /* Per-triangle flux buffers. */
        py::array_t<float> direct(
            {static_cast<py::ssize_t>(_n_tris),
             static_cast<py::ssize_t>(_n_bands)});
        py::array_t<float> indirect(
            {static_cast<py::ssize_t>(_n_tris),
             static_cast<py::ssize_t>(_n_bands)});

        std::fill(direct  .mutable_data(),
                  direct  .mutable_data() + direct  .size(), 0.0f);
        std::fill(indirect.mutable_data(),
                  indirect.mutable_data() + indirect.size(), 0.0f);

        int n_written = 0;
        int rc;
        {
            py::gil_scoped_release release;
            rc = ray_tracer_trace_surface(
                handle,
                n_sources,
                static_cast<const double*>(ip .ptr),
                static_cast<const double*>(id_.ptr),
                static_cast<const double*>(idv.ptr),
                n_rays, max_bounces, min_amplitude, seed,
                segs.mutable_data(), out_cap, &n_written,
                direct  .mutable_data(),
                indirect.mutable_data());
        }

        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_trace_surface failed: rc=" + std::to_string(rc));

        segs.resize({static_cast<py::ssize_t>(n_written),
                     static_cast<py::ssize_t>(RT_FLOATS_PER_SEG)});

        py::dict result;
        result["segs"]     = segs;
        result["direct"]   = direct;
        result["indirect"] = indirect;
        return result;
    }

    /**
     * integrate_ir(src_pos, src_dir, src_directivity,
     *              rec_pos, rec_aperture_r,
     *              speed_m_s, sample_rate, n_samples,
     *              n_rays, max_bounces, min_amplitude, seed)
     *   -> (out_re, out_im) each float32 (n_src, n_rec, n_bands, n_samples)
     *
     * Accumulates ray arrivals at point receivers into time-domain impulse
     * responses.  Hit points within rec_aperture_r of each receiver are
     * binned by path delay with linear interpolation and distance falloff.
     * Caller is responsible for initialising with zeros (not done here).
     */
    py::tuple integrate_ir(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        py::array_t<double> rec_pos,
        py::array_t<double> rec_aperture_r,
        double   speed_m_s    = 343.0,
        double   sample_rate  = 44100.0,
        int      n_samples    = 4096,
        int      n_rays       = 512,
        int      max_bounces  = 12,
        double   min_amplitude = 0.001,
        uint32_t seed         = 42)
    {
        auto ip   = src_pos        .request();
        auto id_  = src_dir        .request();
        auto idv  = src_directivity.request();
        auto irp  = rec_pos        .request();
        auto irap = rec_aperture_r .request();

        int n_sources   = static_cast<int>(idv.size);
        int n_receivers = static_cast<int>(irap.size);

        py::array_t<float> out_re(
            {n_sources, n_receivers, _n_bands, n_samples});
        py::array_t<float> out_im(
            {n_sources, n_receivers, _n_bands, n_samples});

        /* Zero-fill output buffers — integrators accumulate additively. */
        std::fill(out_re.mutable_data(),
                  out_re.mutable_data() + out_re.size(), 0.0f);
        std::fill(out_im.mutable_data(),
                  out_im.mutable_data() + out_im.size(), 0.0f);

        int rc = ray_tracer_integrate_ir(
            handle,
            n_sources,
            static_cast<const double*>(ip  .ptr),
            static_cast<const double*>(id_ .ptr),
            static_cast<const double*>(idv .ptr),
            n_rays, max_bounces, min_amplitude, seed,
            n_receivers,
            static_cast<const double*>(irp .ptr),
            static_cast<const double*>(irap.ptr),
            speed_m_s, sample_rate, n_samples,
            out_re.mutable_data(),
            out_im.mutable_data());

        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_integrate_ir failed: rc=" + std::to_string(rc));

        return py::make_tuple(out_re, out_im);
    }

    /**
     * integrate_image(src_pos, src_dir, src_directivity,
     *                 cam_pos, cam_fwd, cam_up, fov_rad,
     *                 width, height,
     *                 n_rays, max_bounces, min_amplitude, seed)
     *   -> float32 (n_bands, height, width)
     *
     * Projects each ray hit onto a pinhole camera and accumulates
     * per-band amplitude magnitude |A[b]| into pixels.
     */
    py::array_t<float> integrate_image(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        py::array_t<double> cam_pos,
        py::array_t<double> cam_fwd,
        py::array_t<double> cam_up,
        double   fov_rad      = 1.0,
        int      width        = 512,
        int      height       = 384,
        int      n_rays       = 512,
        int      max_bounces  = 12,
        double   min_amplitude = 0.001,
        uint32_t seed         = 42)
    {
        auto ip  = src_pos        .request();
        auto id_ = src_dir        .request();
        auto idv = src_directivity.request();
        auto icp = cam_pos        .request();
        auto icf = cam_fwd        .request();
        auto icu = cam_up         .request();

        int n_sources = static_cast<int>(idv.size);
        (void)n_sources;

        py::array_t<float> out_image({_n_bands, height, width});
        std::fill(out_image.mutable_data(),
                  out_image.mutable_data() + out_image.size(), 0.0f);

        int rc = ray_tracer_integrate_image(
            handle,
            static_cast<int>(idv.size),
            static_cast<const double*>(ip .ptr),
            static_cast<const double*>(id_.ptr),
            static_cast<const double*>(idv.ptr),
            n_rays, max_bounces, min_amplitude, seed,
            static_cast<const double*>(icp.ptr),
            static_cast<const double*>(icf.ptr),
            static_cast<const double*>(icu.ptr),
            fov_rad, width, height,
            out_image.mutable_data());

        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_integrate_image failed: rc=" + std::to_string(rc));

        return out_image;
    }

    void integrate_image_into(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        py::array_t<double> cam_pos,
        py::array_t<double> cam_fwd,
        py::array_t<double> cam_up,
        py::array_t<float>  out_image,
        double   fov_rad       = 1.0,
        int      n_rays        = 512,
        int      max_bounces   = 12,
        double   min_amplitude = 0.001,
        uint32_t seed          = 42)
    {
        auto ip  = src_pos        .request();
        auto id_ = src_dir        .request();
        auto idv = src_directivity.request();
        auto icp = cam_pos        .request();
        auto icf = cam_fwd        .request();
        auto icu = cam_up         .request();
        auto oi  = out_image      .request();

        if (oi.ndim != 3 || oi.shape[0] != _n_bands)
            throw std::runtime_error("out_image must be float32 shape (n_bands, height, width)");
        int height = static_cast<int>(oi.shape[1]);
        int width  = static_cast<int>(oi.shape[2]);

        int rc;
        {
            py::gil_scoped_release release;
            rc = ray_tracer_integrate_image(
                handle,
                static_cast<int>(idv.size),
                static_cast<const double*>(ip .ptr),
                static_cast<const double*>(id_.ptr),
                static_cast<const double*>(idv.ptr),
                n_rays, max_bounces, min_amplitude, seed,
                static_cast<const double*>(icp.ptr),
                static_cast<const double*>(icf.ptr),
                static_cast<const double*>(icu.ptr),
                fov_rad, width, height,
                static_cast<float*>(oi.ptr));
        }

        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_integrate_image failed: rc=" + std::to_string(rc));
    }

    int trace_integrate_image_into(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        py::array_t<double> cam_pos,
        py::array_t<double> cam_fwd,
        py::array_t<double> cam_up,
        py::array_t<float>  out_image,
        py::array_t<float>  out_segs,
        double   fov_rad       = 1.0,
        int      n_rays        = 512,
        int      max_bounces   = 12,
        double   min_amplitude = 0.001,
        uint32_t seed          = 42)
    {
        auto ip  = src_pos        .request();
        auto id_ = src_dir        .request();
        auto idv = src_directivity.request();
        auto icp = cam_pos        .request();
        auto icf = cam_fwd        .request();
        auto icu = cam_up         .request();
        auto oi  = out_image      .request();
        auto os  = out_segs       .request();

        if (oi.ndim != 3 || oi.shape[0] != _n_bands)
            throw std::runtime_error("out_image must be float32 shape (n_bands, height, width)");
        if (os.ndim != 2 || os.shape[1] != RT_FLOATS_PER_SEG)
            throw std::runtime_error("out_segs must be float32 shape (capacity, 12)");

        int height = static_cast<int>(oi.shape[1]);
        int width  = static_cast<int>(oi.shape[2]);
        int out_cap = static_cast<int>(os.shape[0]);
        int n_written = 0;

        int rc;
        {
            py::gil_scoped_release release;
            rc = ray_tracer_trace_integrate_image(
                handle,
                static_cast<int>(idv.size),
                static_cast<const double*>(ip .ptr),
                static_cast<const double*>(id_.ptr),
                static_cast<const double*>(idv.ptr),
                n_rays, max_bounces, min_amplitude, seed,
                static_cast<const double*>(icp.ptr),
                static_cast<const double*>(icf.ptr),
                static_cast<const double*>(icu.ptr),
                fov_rad, width, height,
                static_cast<float*>(oi.ptr),
                static_cast<float*>(os.ptr),
                out_cap,
                &n_written);
        }

        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_trace_integrate_image failed: rc=" + std::to_string(rc));
        return n_written;
    }
};

/* ── FieldSolver wrapper ─────────────────────────────────────────────────── */

struct PyFieldSolver
{
    RtsFieldState* handle  = nullptr;
    int            _n_src  = 0;
    int            _n_rec  = 0;
    int            _n_bands = 0;
    RtsMode        _mode   = RTS_ACOUSTIC;

    /* ctor(scene_dict, receivers_list, freq_hz, speed_m_s, mode_str)
     *
     * scene_dict keys (all numpy arrays unless noted):
     *   verts      : float64 (n_tri, 3, 3) or (n_tri, 9)
     *   normals    : float64 (n_tri, 3)
     *   mat_idx    : int32   (n_tri,)
     *   n_mats     : int
     *   mat_n_re   : float64 (n_mats, n_bands)
     *   mat_n_im   : float64 (n_mats, n_bands)
     *   mat_diffusion : float64 (n_mats,)
     *   medium_n_re : float
     *   medium_n_im : float
     *
     * receivers_list: list of dicts with keys:
     *   pos, axis : float64 (3,)
     *   polar_type : int (RtsPolarType)
     *   aperture_r : float
     *   pol_s, pol_p : float64 (3,) — EM only, defaults to zero
     */
    PyFieldSolver(py::dict              scene_dict,
                  py::list              receivers_list,
                  py::array_t<double>   freq_hz_arr,
                  double                speed_m_s,
                  std::string           mode_str)
    {
        _mode    = (mode_str == "em") ? RTS_EM : RTS_ACOUSTIC;
        auto fhi = freq_hz_arr.request();
        _n_bands = static_cast<int>(fhi.size);
        _n_rec   = static_cast<int>(py::len(receivers_list));

        /* ── unpack scene ── */
        auto verts_arr  = scene_dict["verts"]       .cast<py::array_t<double>>();
        auto norms_arr  = scene_dict["normals"]      .cast<py::array_t<double>>();
        auto midx_arr   = scene_dict["mat_idx"]      .cast<py::array_t<int32_t>>();
        auto mre_arr    = scene_dict["mat_n_re"]     .cast<py::array_t<double>>();
        auto mim_arr    = scene_dict["mat_n_im"]     .cast<py::array_t<double>>();
        auto mdiff_arr  = scene_dict["mat_diffusion"].cast<py::array_t<double>>();
        int  n_mats     = scene_dict["n_mats"]       .cast<int>();
        double med_re   = scene_dict["medium_n_re"]  .cast<double>();
        double med_im   = scene_dict["medium_n_im"]  .cast<double>();

        auto vi  = verts_arr .request();
        auto ni  = norms_arr .request();
        auto mi  = midx_arr  .request();
        auto mri = mre_arr   .request();
        auto mii = mim_arr   .request();
        auto mdi = mdiff_arr .request();

        int n_tri = static_cast<int>(mi.size);

        RtsScene scene{};
        scene.n_tri        = n_tri;
        scene.verts        = static_cast<const double*>(vi .ptr);
        scene.normals      = static_cast<const double*>(ni .ptr);
        scene.mat_idx      = static_cast<const int*>   (mi .ptr);
        scene.n_mats       = n_mats;
        scene.n_bands      = _n_bands;
        scene.mat_n_re     = static_cast<const double*>(mri.ptr);
        scene.mat_n_im     = static_cast<const double*>(mii.ptr);
        scene.mat_diffusion= static_cast<const double*>(mdi.ptr);
        scene.medium_n_re  = med_re;
        scene.medium_n_im  = med_im;

        /* ── unpack receivers ── */
        _receivers.resize(_n_rec);
        for (int i = 0; i < _n_rec; ++i) {
            py::dict rd = receivers_list[i].cast<py::dict>();
            RtsReceiver& r = _receivers[i];

            auto copy3 = [&](const char* key, double* dst) {
                auto arr = rd[key].cast<py::array_t<double>>();
                auto info = arr.request();
                auto* src = static_cast<const double*>(info.ptr);
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            };

            copy3("pos",  r.pos);
            copy3("axis", r.axis);
            r.polar_type = static_cast<RtsPolarType>(rd["polar_type"].cast<int>());
            r.aperture_r = rd.contains("aperture_r") ? rd["aperture_r"].cast<double>() : 0.0;

            if (rd.contains("pol_s")) copy3("pol_s", r.pol_s);
            else r.pol_s[0] = r.pol_s[1] = r.pol_s[2] = 0.0;
            if (rd.contains("pol_p")) copy3("pol_p", r.pol_p);
            else r.pol_p[0] = r.pol_p[1] = r.pol_p[2] = 0.0;
        }

        handle = rts_create(&scene,
                            static_cast<const double*>(fhi.ptr),
                            speed_m_s,
                            _mode);
        if (!handle)
            throw std::runtime_error("rts_create: allocation failed");
    }

    ~PyFieldSolver() { rts_destroy(handle); handle = nullptr; }

    /* ── internal solve helper ── */
    void _run_solve(
        py::array_t<double> src_pos_arr,
        py::array_t<double> src_dir_arr,
        py::array_t<double> src_directivity_arr,
        py::array_t<double> src_pol_re_arr,   /* EM: (n_src, 3); acoustic: ignored */
        py::array_t<double> src_pol_im_arr,
        int     n_rays,
        int     max_bounces,
        double  min_amplitude,
        uint32_t seed,
        double  schroeder_hz,
        double* out_re,
        double* out_im)
    {
        auto ip  = src_pos_arr        .request();
        auto id  = src_dir_arr        .request();
        auto idv = src_directivity_arr.request();

        _n_src = static_cast<int>(idv.size);

        const double* pol_re_ptr = nullptr;
        const double* pol_im_ptr = nullptr;
        std::vector<double> zero_pol;
        if (_mode == RTS_EM) {
            pol_re_ptr = static_cast<const double*>(src_pol_re_arr.request().ptr);
            pol_im_ptr = static_cast<const double*>(src_pol_im_arr.request().ptr);
        } else {
            /* acoustic: pass zeroed placeholder so rts_solve doesn't get NULL */
            zero_pol.assign(_n_src * 3, 0.0);
            pol_re_ptr = zero_pol.data();
            pol_im_ptr = zero_pol.data();
        }

        int rc = rts_solve(
            handle,
            _n_src,
            static_cast<const double*>(ip .ptr),
            static_cast<const double*>(id .ptr),
            static_cast<const double*>(idv.ptr),
            pol_re_ptr,
            pol_im_ptr,
            _n_rec,
            _receivers.data(),
            n_rays, max_bounces, min_amplitude, seed, schroeder_hz,
            out_re, out_im);

        if (rc != SK_OK)
            throw std::runtime_error("rts_solve failed: rc=" + std::to_string(rc));
    }

    /**
     * solve_acoustic(...) → complex128 ndarray, shape (n_src, n_rec, n_bands)
     *
     * Returns H[source, receiver, band] as a numpy complex128 array.
     * The complex value is the coherent pressure transfer function.
     */
    py::array_t<std::complex<double>> solve_acoustic(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        int      n_rays        = 512,
        int      max_bounces   = 8,
        double   min_amplitude = 0.001,
        uint32_t seed          = 42,
        double   schroeder_hz  = 0.0)
    {
        if (_mode != RTS_ACOUSTIC)
            throw std::runtime_error("solve_acoustic called on EM solver; use solve_em");

        /* Temporary -- n_src is set inside _run_solve */
        auto ip = src_pos.request();
        int n_src_pre = static_cast<int>(ip.size / 3);
        size_t total  = static_cast<size_t>(n_src_pre) * _n_rec * _n_bands;

        std::vector<double> re(total, 0.0), im(total, 0.0);
        py::array_t<double> dummy_pol({n_src_pre * 3}); /* zeros */
        dummy_pol.mutable_data()[0]; // touch to allocate

        _run_solve(src_pos, src_dir, src_directivity,
                   dummy_pol, dummy_pol,
                   n_rays, max_bounces, min_amplitude, seed, schroeder_hz,
                   re.data(), im.data());

        /* Pack into complex128 */
        py::array_t<std::complex<double>> out(
            {_n_src, _n_rec, _n_bands});
        auto buf = out.mutable_unchecked<3>();
        for (int si = 0; si < _n_src; ++si)
            for (int ri = 0; ri < _n_rec; ++ri)
                for (int b = 0; b < _n_bands; ++b) {
                    size_t idx = RTS_IDX_ACOUSTIC(si, ri, b, _n_rec, _n_bands);
                    buf(si, ri, b) = {re[idx], im[idx]};
                }
        return out;
    }

    /**
     * solve_em(...) → complex128 ndarray, shape (n_src, n_rec, n_bands, 2, 2)
     *
     * Returns the Jones transfer matrix H[src, rec, band, p_out, p_in].
     * p_out/p_in indices: 0 = s-polarisation, 1 = p-polarisation.
     */
    py::array_t<std::complex<double>> solve_em(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        py::array_t<double> src_pol_re,
        py::array_t<double> src_pol_im,
        int      n_rays        = 512,
        int      max_bounces   = 8,
        double   min_amplitude = 0.001,
        uint32_t seed          = 42,
        double   schroeder_hz  = 0.0)
    {
        if (_mode != RTS_EM)
            throw std::runtime_error("solve_em called on acoustic solver; use solve_acoustic");

        auto ip = src_pos.request();
        int n_src_pre = static_cast<int>(ip.size / 3);
        size_t total  = static_cast<size_t>(n_src_pre) * _n_rec * _n_bands * 4;

        std::vector<double> re(total, 0.0), im(total, 0.0);

        _run_solve(src_pos, src_dir, src_directivity,
                   src_pol_re, src_pol_im,
                   n_rays, max_bounces, min_amplitude, seed, schroeder_hz,
                   re.data(), im.data());

        py::array_t<std::complex<double>> out(
            {_n_src, _n_rec, _n_bands, 2, 2});
        auto buf = out.mutable_unchecked<5>();
        for (int si = 0; si < _n_src; ++si)
            for (int ri = 0; ri < _n_rec; ++ri)
                for (int b = 0; b < _n_bands; ++b)
                    for (int po = 0; po < 2; ++po)
                        for (int pi = 0; pi < 2; ++pi) {
                            size_t idx = RTS_IDX_EM(si, ri, b, po, pi, _n_rec, _n_bands);
                            buf(si, ri, b, po, pi) = {re[idx], im[idx]};
                        }
        return out;
    }

    int n_bands()    const { return _n_bands; }
    int n_receivers()const { return _n_rec; }
    std::string mode()const { return (_mode == RTS_EM) ? "em" : "acoustic"; }

private:
    std::vector<RtsReceiver> _receivers;
};

/* ── Python wrapper for AcousticFDTD ─────────────────────────────────────── */

struct PyAcousticFDTD
{
    AcousticFDTDState* handle = nullptr;
    int Nx = 0, Ny = 0, Nz = 0;

    PyAcousticFDTD(int Nx, int Ny, int Nz,
                   float dx, float c, float rho_air,
                   py::array_t<uint8_t> cell_type_arr,
                   int plate_iz,
                   py::array_t<uint8_t> plate_active_arr,
                   float plate_mass_density,
                   float plate_stiffness_D,
                   float plate_alpha_M,
                   float plate_beta_K,
                   int n_pml)
        : Nx(Nx), Ny(Ny), Nz(Nz)
    {
        auto cti = cell_type_arr  .request();
        auto pai = plate_active_arr.request();
        if ((int)cti.size != Nx*Ny*Nz)
            throw std::runtime_error("cell_type: expected Nx*Ny*Nz elements");
        if ((int)pai.size != Nx*Ny)
            throw std::runtime_error("plate_active: expected Nx*Ny elements");

        handle = fdtd_create(
            Nx, Ny, Nz, dx, c, rho_air,
            static_cast<const uint8_t*>(cti.ptr),
            plate_iz,
            static_cast<const uint8_t*>(pai.ptr),
            plate_mass_density, plate_stiffness_D,
            plate_alpha_M, plate_beta_K, n_pml);
        if (!handle)
            throw std::runtime_error("fdtd_create: allocation failed");
    }

    ~PyAcousticFDTD() { fdtd_destroy(handle); handle = nullptr; }

    void set_bridge_sources(py::array_t<int>   idx_arr,
                            py::array_t<float> wgt_arr)
    {
        auto ii = idx_arr.request();
        auto wi = wgt_arr.request();
        if (ii.size != wi.size)
            throw std::runtime_error("cell_indices and weights must have the same length");
        int rc = fdtd_set_bridge_sources(
            handle,
            (int)ii.size,
            static_cast<const int*>  (ii.ptr),
            static_cast<const float*>(wi.ptr));
        if (rc != FDTD_OK)
            throw std::runtime_error("fdtd_set_bridge_sources error: " + std::to_string(rc));
    }

    void set_face_fractions(py::array_t<float> vx_frac,
                            py::array_t<float> vy_frac,
                            py::array_t<float> vz_frac)
    {
        auto vx = vx_frac.request();
        auto vy = vy_frac.request();
        auto vz = vz_frac.request();
        int rc = fdtd_set_face_fractions(
            handle,
            static_cast<const float*>(vx.ptr),
            static_cast<const float*>(vy.ptr),
            static_cast<const float*>(vz.ptr));
        if (rc != FDTD_OK)
            throw std::runtime_error("fdtd_set_face_fractions error: " + std::to_string(rc));
    }

    void inject_bridge(float signal_val, float signal_ddt, float force_scale)
    {
        int rc = fdtd_inject_bridge(handle, signal_val, signal_ddt, force_scale);
        if (rc != FDTD_OK)
            throw std::runtime_error(
                "fdtd_inject_bridge is disabled: direct pressure bridge injection "
                "bypasses plate impedance; use AcousticCoEvolver structural coupling");
    }

    int step(int n_steps)
    {
        int rc = fdtd_step(handle, n_steps);
        if (rc == FDTD_ERR_UNSTABLE)
            throw std::runtime_error("AcousticFDTD: pressure diverged (FDTD_ERR_UNSTABLE)");
        return rc;
    }

    void reset() { fdtd_reset(handle); }

    py::array_t<float> get_pressure_field()
    {
        py::array_t<float> out({Nx, Ny, Nz});
        int rc = fdtd_get_pressure_field(
            handle,
            static_cast<float*>(out.mutable_unchecked<3>().mutable_data(0,0,0)),
            Nx * Ny * Nz);
        if (rc != FDTD_OK)
            throw std::runtime_error("fdtd_get_pressure_field error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_plate_displacement()
    {
        py::array_t<float> out({Nx, Ny});
        int rc = fdtd_get_plate_displacement(
            handle,
            static_cast<float*>(out.mutable_unchecked<2>().mutable_data(0,0)),
            Nx * Ny);
        if (rc != FDTD_OK)
            throw std::runtime_error("fdtd_get_plate_displacement error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> sample_pressure(py::array_t<float> rec_xyz_arr)
    {
        auto ri = rec_xyz_arr.request();
        int n_rec = (int)(ri.size / 3);
        py::array_t<float> out({n_rec});
        fdtd_sample_pressure(
            handle, n_rec,
            static_cast<const float*>(ri.ptr),
            static_cast<float*>(out.mutable_unchecked<1>().mutable_data(0)));
        return out;
    }

    /** sample_velocity(rec_xyz) → (vx, vy, vz) tuple of float32 (n_rec,) arrays */
    py::tuple sample_velocity(py::array_t<float> rec_xyz_arr)
    {
        auto ri = rec_xyz_arr.request();
        int n_rec = (int)(ri.size / 3);
        py::array_t<float> vx({n_rec}), vy({n_rec}), vz({n_rec});
        fdtd_sample_velocity(handle, n_rec,
            static_cast<const float*>(ri.ptr),
            static_cast<float*>(vx.mutable_unchecked<1>().mutable_data(0)),
            static_cast<float*>(vy.mutable_unchecked<1>().mutable_data(0)),
            static_cast<float*>(vz.mutable_unchecked<1>().mutable_data(0)));
        return py::make_tuple(vx, vy, vz);
    }

    /** get_surface_emission(xyz, normals) → (P, vn) tuple of float32 (n_surf,) arrays */
    py::tuple get_surface_emission(py::array_t<float> xyz_arr,
                                   py::array_t<float> normals_arr)
    {
        auto xi = xyz_arr    .request();
        auto ni = normals_arr.request();
        int n_surf = (int)(xi.size / 3);
        py::array_t<float> P({n_surf}), vn({n_surf});
        fdtd_get_surface_emission(handle, n_surf,
            static_cast<const float*>(xi.ptr),
            static_cast<const float*>(ni.ptr),
            static_cast<float*>(P .mutable_unchecked<1>().mutable_data(0)),
            static_cast<float*>(vn.mutable_unchecked<1>().mutable_data(0)));
        return py::make_tuple(P, vn);
    }

    /** get_velocity_dims() → (Nvx, Nvy, Nvz) */
    py::tuple get_velocity_dims() const
    {
        int Nvx, Nvy, Nvz;
        fdtd_get_velocity_dims(handle, &Nvx, &Nvy, &Nvz);
        return py::make_tuple(Nvx, Nvy, Nvz);
    }

    py::array_t<float> get_velocity_x()
    {
        int Nvx, Nvy, Nvz;
        fdtd_get_velocity_dims(handle, &Nvx, &Nvy, &Nvz);
        py::array_t<float> out({Nvx});
        fdtd_get_velocity_x(handle, out.mutable_unchecked<1>().mutable_data(0), Nvx);
        return out;
    }

    py::array_t<float> get_velocity_y()
    {
        int Nvx, Nvy, Nvz;
        fdtd_get_velocity_dims(handle, &Nvx, &Nvy, &Nvz);
        py::array_t<float> out({Nvy});
        fdtd_get_velocity_y(handle, out.mutable_unchecked<1>().mutable_data(0), Nvy);
        return out;
    }

    py::array_t<float> get_velocity_z()
    {
        int Nvx, Nvy, Nvz;
        fdtd_get_velocity_dims(handle, &Nvx, &Nvy, &Nvz);
        py::array_t<float> out({Nvz});
        fdtd_get_velocity_z(handle, out.mutable_unchecked<1>().mutable_data(0), Nvz);
        return out;
    }

    int   get_step_count() const { return fdtd_get_step_count(handle); }
    float get_dt()         const { return fdtd_get_dt(handle); }
};

/* ── AcousticCoEvolver error → exception helper ──────────────────────────── */

[[noreturn]] static void _ce_throw(int rc, const char* ctx = "")
{
    std::string prefix = ctx && ctx[0] ? (std::string(ctx) + ": ") : "";
    switch (rc) {
        case CE_ERR_NULL:
            throw std::runtime_error(prefix + "null pointer (CE_ERR_NULL -1)");
        case CE_ERR_DIM:
            throw std::runtime_error(prefix + "invalid dimension (CE_ERR_DIM -2)");
        case CE_ERR_UNSTABLE:
            throw std::runtime_error(prefix + "body FDTD diverged — CFL violated or drive too large (CE_ERR_UNSTABLE -3)");
        case CE_ERR_PARAM:
            throw std::runtime_error(prefix + "invalid parameter (CE_ERR_PARAM -4)");
        case CE_ERR_STRING_IDX:
            throw std::runtime_error(prefix + "string index out of range (CE_ERR_STRING_IDX -5)");
        case CE_ERR_BUSY:
            throw std::runtime_error(prefix + "async job is already running (CE_ERR_BUSY -6)");
        case CE_ERR_FDTD:
            throw std::runtime_error(prefix + "FDTD subsystem error (CE_ERR_FDTD -7)");
        case CE_ERR_FDTD_BRIDGE:
            throw std::runtime_error(prefix +
                "FDTD bridge injection failed — fdtd_inject_bridge_drive returned non-OK; "
                "check that st->fdtd is initialised and bridge cell count matches (CE_ERR_FDTD_BRIDGE -8)");
        case CE_ERR_FDTD_STEP:
            throw std::runtime_error(prefix +
                "fdtd_step returned a non-unstable error — likely NULL FDTD state or bad grid dims (CE_ERR_FDTD_STEP -9)");
        case CE_ERR_FDTD_MIC_P:
            throw std::runtime_error(prefix +
                "FDTD pressure sampling failed — check mic index/weight arrays and FDTD grid size (CE_ERR_FDTD_MIC_P -10)");
        case CE_ERR_FDTD_MIC_V:
            throw std::runtime_error(prefix +
                "FDTD velocity sampling failed — check mic velocity index/weight arrays (CE_ERR_FDTD_MIC_V -11)");
        default:
            throw std::runtime_error(prefix + "unknown error code " + std::to_string(rc));
    }
}

/* ── Python wrapper for AcousticCoEvolver ────────────────────────────────── */

struct PyAcousticCoEvolver
{
    AcousticCoEvolverState* handle  = nullptr;
    int _n_strings  = 0;
    int _n_pickups  = 0;
    int _n_mics     = 0;
    int _Nx = 0, _Ny = 0, _Nz = 0;

    /**
     * Constructor — accepts plain Python dicts/lists so the caller does not
     * need to build ctypes structs.
     *
     * string_defs : list of dicts, each with:
     *   path_xyz       : float32 array (n_segs+1, 3)
     *   tension_N      : float
     *   linear_mass_kgm: float
     *   damping        : float
     *   stiffness_EI   : float, optional
     *   axial_stiffness_N : float, optional EA axial stiffness
     *
     * pickup_defs : list of dicts, each with:
     *   type           : int (0=SINGLE_COIL, 1=HUMBUCKER, 2=PIEZO)
     *   pos            : float (3,)
     *   axis           : float (3,)
     *   pole_sigma     : float
     *   coil_spacing   : float  (humbucker)
     *   sensitivity    : float  (piezo)
     *   string_mask    : int
     *
     * mic_defs : list of dicts, each with:
     *   pos  : float (3,)
     *   gain : float
     *
     * body : dict with all CoEvolverBodyDef fields:
     *   Nx, Ny, Nz, dx, c, rho_air
     *   cell_type         : uint8 array (Nx·Ny·Nz)
     *   plate_iz          : int
     *   plate_active      : uint8 array (Nx·Ny)
     *   plate_mass_density: float
     *   plate_stiffness_D : float
     *   n_pml             : int
     *   bridge_src_xyz    : float32 array (n_bridge_src, 3)
     *   bridge_sigma      : float
     */
    PyAcousticCoEvolver(py::list              string_defs_list,
                        py::list              pickup_defs_list,
                        py::list              mic_defs_list,
                        py::dict              body_dict,
                        float                 sample_rate,
                        int                   modal_stride = 16,
                        float                 force_scale  = 1.0f)
    {
        /* ── Strings ── */
        _n_strings = (int)py::len(string_defs_list);
        std::vector<CoEvolverStringDef>    sdefs(_n_strings);
        std::vector<std::vector<float>>    path_bufs(_n_strings);

        for (int si = 0; si < _n_strings; ++si) {
            py::dict d = string_defs_list[si].cast<py::dict>();
            auto path_arr = d["path_xyz"].cast<py::array_t<float>>();
            auto pi = path_arr.request();
            path_bufs[si].assign(
                static_cast<const float*>(pi.ptr),
                static_cast<const float*>(pi.ptr) + pi.size);
            sdefs[si].path_xyz        = path_bufs[si].data();
            sdefs[si].n_segs          = (int)(pi.size / 3) - 1;
            sdefs[si].tension_N       = d["tension_N"].cast<float>();
            sdefs[si].linear_mass_kgm = d["linear_mass_kgm"].cast<float>();
            sdefs[si].damping         = d["damping"].cast<float>();
            sdefs[si].stiffness_EI    = d.contains("stiffness_EI")   ? d["stiffness_EI"].cast<float>()   : 0.0f;
            sdefs[si].axial_stiffness_N =
                d.contains("axial_stiffness_N") ? d["axial_stiffness_N"].cast<float>() : 0.0f;
            sdefs[si].neck_freq_hz    = d.contains("neck_freq_hz")    ? d["neck_freq_hz"].cast<float>()    : 0.0f;
            sdefs[si].neck_mass_kg    = d.contains("neck_mass_kg")    ? d["neck_mass_kg"].cast<float>()    : 0.0f;
            sdefs[si].neck_Q          = d.contains("neck_Q")          ? d["neck_Q"].cast<float>()          : 0.0f;
        }

        /* ── Pickups ── */
        _n_pickups = (int)py::len(pickup_defs_list);
        std::vector<CoEvolverPickupDef> pdefs(_n_pickups);
        for (int pi = 0; pi < _n_pickups; ++pi) {
            py::dict d = pickup_defs_list[pi].cast<py::dict>();
            pdefs[pi].type        = d["type"]       .cast<int>();
            pdefs[pi].pole_sigma  = d["pole_sigma"]  .cast<float>();
            pdefs[pi].coil_spacing= d.contains("coil_spacing") ? d["coil_spacing"].cast<float>() : 0.018f;
            pdefs[pi].sensitivity = d.contains("sensitivity")  ? d["sensitivity"] .cast<float>() : 1.0f;
            pdefs[pi].string_mask = d.contains("string_mask")  ? (uint32_t)d["string_mask"].cast<int>() : 0xFFFFFFFFu;
            auto copy3f = [&](const char* key, float* dst) {
                auto arr = d[key].cast<py::array_t<float>>();
                auto info = arr.request();
                auto* src = static_cast<const float*>(info.ptr);
                dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2];
            };
            copy3f("pos",  pdefs[pi].pos);
            copy3f("axis", pdefs[pi].axis);
        }

        /* ── Mics ── */
        _n_mics = (int)py::len(mic_defs_list);
        std::vector<CoEvolverMicDef> mdefs(_n_mics);
        for (int mi = 0; mi < _n_mics; ++mi) {
            py::dict d = mic_defs_list[mi].cast<py::dict>();
            auto arr = d["pos"].cast<py::array_t<float>>();
            auto info = arr.request();
            auto* src = static_cast<const float*>(info.ptr);
            mdefs[mi].pos[0]=src[0]; mdefs[mi].pos[1]=src[1]; mdefs[mi].pos[2]=src[2];
            mdefs[mi].gain    = d.contains("gain")    ? d["gain"]   .cast<float>() : 1.0f;
            mdefs[mi].polar_a = d.contains("polar_a") ? d["polar_a"].cast<float>() : 1.0f;
            mdefs[mi].polar_b = d.contains("polar_b") ? d["polar_b"].cast<float>() : 0.0f;
            if (d.contains("axis")) {
                auto aa = d["axis"].cast<py::array_t<float>>();
                auto ai = aa.request();
                auto* ap = static_cast<const float*>(ai.ptr);
                mdefs[mi].axis[0]=ap[0]; mdefs[mi].axis[1]=ap[1]; mdefs[mi].axis[2]=ap[2];
            } else {
                mdefs[mi].axis[0]=0.0f; mdefs[mi].axis[1]=0.0f; mdefs[mi].axis[2]=1.0f;
            }
        }

        /* ── Body ── */
        _Nx = body_dict["Nx"].cast<int>();
        _Ny = body_dict["Ny"].cast<int>();
        _Nz = body_dict["Nz"].cast<int>();

        auto ct_arr  = body_dict["cell_type"]   .cast<py::array_t<uint8_t>>();
        auto pa_arr  = body_dict["plate_active"] .cast<py::array_t<uint8_t>>();
        auto bsrc_arr= body_dict["bridge_src_xyz"].cast<py::array_t<float>>();
        auto cti = ct_arr  .request();
        auto pai = pa_arr  .request();
        auto bsi = bsrc_arr.request();
        int n_bridge_src = (int)(bsi.size / 3);

        CoEvolverBodyDef body{};
        body.Nx                 = _Nx;
        body.Ny                 = _Ny;
        body.Nz                 = _Nz;
        body.dx                 = body_dict["dx"]               .cast<float>();
        body.origin[0]          = body_dict.contains("gx_min")   ? body_dict["gx_min"].cast<float>() : 0.0f;
        body.origin[1]          = body_dict.contains("gy_min")   ? body_dict["gy_min"].cast<float>() : 0.0f;
        body.origin[2]          = body_dict.contains("gz_min")   ? body_dict["gz_min"].cast<float>() : 0.0f;
        body.c                  = body_dict.contains("c")        ? body_dict["c"]               .cast<float>() : 343.0f;
        body.rho_air            = body_dict.contains("rho_air")  ? body_dict["rho_air"]          .cast<float>() : 1.21f;
        body.cell_type          = static_cast<const uint8_t*>(cti.ptr);
        body.plate_iz           = body_dict["plate_iz"]          .cast<int>();
        body.plate_active       = static_cast<const uint8_t*>(pai.ptr);
        body.plate_mass_density = body_dict.contains("plate_mass_density") ? body_dict["plate_mass_density"].cast<float>() : 7.0f;
        body.plate_stiffness_D  = body_dict.contains("plate_stiffness_D")  ? body_dict["plate_stiffness_D"] .cast<float>() : 0.45f;
        body.plate_alpha_M      = body_dict.contains("plate_alpha_M")       ? body_dict["plate_alpha_M"]      .cast<float>() : 2.0f;
        body.plate_beta_K       = body_dict.contains("plate_beta_K")        ? body_dict["plate_beta_K"]       .cast<float>() : 1e-5f;
        body.n_pml              = body_dict.contains("n_pml")               ? body_dict["n_pml"]             .cast<int>()   : 10;
        body.n_bridge_src       = n_bridge_src;
        body.bridge_src_xyz     = static_cast<const float*>(bsi.ptr);
        body.bridge_sigma       = body_dict.contains("bridge_sigma") ? body_dict["bridge_sigma"].cast<float>() : 0.01f;

        handle = coevolver_create(
            _n_strings, sdefs.empty()  ? nullptr : sdefs.data(),
            _n_pickups, pdefs.empty()  ? nullptr : pdefs.data(),
            _n_mics,    mdefs.empty()  ? nullptr : mdefs.data(),
            &body,
            sample_rate, modal_stride, force_scale);

        if (!handle)
            throw std::runtime_error("coevolver_create: allocation failed");
    }

    ~PyAcousticCoEvolver() { coevolver_destroy(handle); handle = nullptr; }

    /* step(n_samples) → CE_OK=0 or raises on instability.
     * GIL is released for the duration of the C++ physics call. */
    int step(int n_samples)
    {
        int rc;
        {
            py::gil_scoped_release release;
            rc = coevolver_step(handle, n_samples);
        }
        if (rc != CE_OK) _ce_throw(rc, "step");
        return rc;
    }

    /**
     * step_block_with_drive(drive_blocks, force_pos_norm) → float32 (n_samples,)
     *
     * drive_blocks : list of float32 arrays, one per string, each length n_samples.
     *                Pass an empty list or None to drive with silence.
     * force_pos_norm : fractional string position for force injection (0–1).
     *
     * Injects the per-string drive, advances physics, and returns mic 0 output —
     * all in one call with the GIL released for the C++ work.
     */
    py::array_t<float> step_block_with_drive(
        py::object drive_blocks_obj,
        float      force_pos_norm,
        int        n_samples)
    {
        /* Unpack list of float32 arrays → vector of raw pointers. */
        std::vector<py::array_t<float>> arrays;
        std::vector<const float*>       ptrs;

        if (!drive_blocks_obj.is_none()) {
            py::list lst = drive_blocks_obj.cast<py::list>();
            for (auto item : lst) {
                arrays.push_back(item.cast<py::array_t<float>>());
                auto info = arrays.back().request();
                ptrs.push_back(static_cast<const float*>(info.ptr));
            }
        }

        int n_str = (int)ptrs.size();
        const float* const* blocks = n_str > 0 ? ptrs.data() : nullptr;

        py::array_t<float> out({n_samples});
        float* out_ptr = out.mutable_unchecked<1>().mutable_data(0);

        int rc;
        {
            py::gil_scoped_release release;
            rc = coevolver_step_block_with_drive(
                handle, blocks, n_str > 0 ? n_str : _n_strings,
                n_samples, force_pos_norm, out_ptr);
        }
        if (rc == CE_ERR_PARAM)
            throw std::runtime_error(
                "step_block_with_drive: n_strings mismatch — "
                "got " + std::to_string(n_str) +
                ", expected " + std::to_string(_n_strings) +
                " (CE_ERR_PARAM -4)");
        if (rc != CE_OK) _ce_throw(rc, "step_block_with_drive");
        return out;
    }

    void pluck_string(int string_idx, float position_norm, float amplitude)
    {
        int rc = coevolver_pluck_string(handle, string_idx, position_norm, amplitude);
        if (rc != CE_OK)
            throw std::runtime_error("coevolver_pluck_string error: " + std::to_string(rc));
    }

    void schedule_pluck(int onset_sample, int string_idx,
                        float position_norm, float amplitude)
    {
        int rc = coevolver_schedule_pluck(handle, onset_sample, string_idx,
                                          position_norm, amplitude);
        if (rc != CE_OK)
            throw std::runtime_error("coevolver_schedule_pluck error: " + std::to_string(rc));
    }

    void clear_pluck_schedule()
    {
        coevolver_clear_pluck_schedule(handle);
    }

    void inject_string_force(int string_idx,
                             py::array_t<float> force_arr,
                             float pos_norm)
    {
        auto fi = force_arr.request();
        int rc = coevolver_inject_string_force(
            handle, string_idx,
            (int)fi.size,
            static_cast<const float*>(fi.ptr),
            pos_norm);
        if (rc != CE_OK)
            throw std::runtime_error("coevolver_inject_string_force error: " + std::to_string(rc));
    }

    py::array_t<float> get_pickup_output(int pickup_idx, int n_samples)
    {
        py::array_t<float> out({n_samples});
        int rc = coevolver_get_pickup_output(handle, pickup_idx,
            out.mutable_unchecked<1>().mutable_data(0), n_samples);
        if (rc != CE_OK)
            throw std::runtime_error("get_pickup_output error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_mic_output(int mic_idx, int n_samples)
    {
        py::array_t<float> out({n_samples});
        int rc = coevolver_get_mic_output(handle, mic_idx,
            out.mutable_unchecked<1>().mutable_data(0), n_samples);
        if (rc != CE_OK)
            throw std::runtime_error("get_mic_output error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_pressure_field()
    {
        py::array_t<float> out({_Nx, _Ny, _Nz});
        int rc = coevolver_get_pressure_field(handle,
            out.mutable_unchecked<3>().mutable_data(0,0,0),
            _Nx * _Ny * _Nz);
        if (rc != CE_OK)
            throw std::runtime_error("get_pressure_field error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_plate_displacement()
    {
        py::array_t<float> out({_Nx, _Ny});
        int rc = coevolver_get_plate_displacement(handle,
            out.mutable_unchecked<2>().mutable_data(0,0),
            _Nx * _Ny);
        if (rc != CE_OK)
            throw std::runtime_error("get_plate_displacement error: " + std::to_string(rc));
        return out;
    }

    py::tuple get_modal_amplitudes(int string_idx, int n_modes = 32)
    {
        py::array_t<float> re({n_modes}), im({n_modes});
        int rc = coevolver_get_modal_amplitudes(handle, string_idx,
            re.mutable_unchecked<1>().mutable_data(0),
            im.mutable_unchecked<1>().mutable_data(0),
            n_modes);
        if (rc != CE_OK)
            throw std::runtime_error("get_modal_amplitudes error: " + std::to_string(rc));
        return py::make_tuple(re, im);
    }

    py::array_t<float> sample_pressure(py::array_t<float> rec_xyz_arr)
    {
        auto ri = rec_xyz_arr.request();
        int n_rec = (int)(ri.size / 3);
        py::array_t<float> out({n_rec});
        int rc = coevolver_sample_pressure(handle, n_rec,
            static_cast<const float*>(ri.ptr),
            out.mutable_unchecked<1>().mutable_data(0));
        if (rc != CE_OK)
            throw std::runtime_error("sample_pressure error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_string_velocity(int string_idx)
    {
        /* Get n_segs from the handle query */
        int n_strings = coevolver_get_n_strings(handle);
        if (string_idx < 0 || string_idx >= n_strings)
            throw std::runtime_error("string index out of range");
        /* We don't have a direct n_segs query; allocate generously and trim */
        /* Use a reasonable upper bound — caller knows their own n_segs */
        int max_segs = 512;
        py::array_t<float> out({max_segs, 3});
        int rc = coevolver_get_string_velocity(handle, string_idx,
            out.mutable_unchecked<2>().mutable_data(0,0),
            max_segs * 3);
        if (rc == CE_ERR_DIM) {
            /* Try with a larger buffer — caller must reshape */
            throw std::runtime_error(
                "get_string_velocity: out_len mismatch. "
                "Pass n_segs*3 explicitly via get_string_velocity_n.");
        }
        if (rc != CE_OK)
            throw std::runtime_error("get_string_velocity error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_string_velocity_n(int string_idx, int n_segs)
    {
        py::array_t<float> out({n_segs, 3});
        int rc = coevolver_get_string_velocity(handle, string_idx,
            out.mutable_unchecked<2>().mutable_data(0,0),
            n_segs * 3);
        if (rc != CE_OK)
            throw std::runtime_error("get_string_velocity error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_string_displacement_n(int string_idx, int n_segs)
    {
        py::array_t<float> out({n_segs, 3});
        int rc = coevolver_get_string_displacement(handle, string_idx,
            out.mutable_unchecked<2>().mutable_data(0,0),
            n_segs * 3);
        if (rc != CE_OK)
            throw std::runtime_error("get_string_displacement error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_string_position_n(int string_idx, int n_nodes)
    {
        py::array_t<float> out({n_nodes, 3});
        int rc = coevolver_get_string_position(handle, string_idx,
            out.mutable_unchecked<2>().mutable_data(0,0),
            n_nodes * 3);
        if (rc != CE_OK)
            throw std::runtime_error("get_string_position error: " + std::to_string(rc));
        return out;
    }

    /** get_surface_emission(xyz, normals) → (P, vn) tuple of float32 (n_surf,) */
    py::tuple get_surface_emission(py::array_t<float> xyz_arr,
                                   py::array_t<float> normals_arr)
    {
        auto xi = xyz_arr    .request();
        auto ni = normals_arr.request();
        int n_surf = (int)(xi.size / 3);
        py::array_t<float> P({n_surf}), vn({n_surf});
        int rc = coevolver_get_surface_emission(handle, n_surf,
            static_cast<const float*>(xi.ptr),
            static_cast<const float*>(ni.ptr),
            static_cast<float*>(P .mutable_unchecked<1>().mutable_data(0)),
            static_cast<float*>(vn.mutable_unchecked<1>().mutable_data(0)));
        if (rc != CE_OK)
            throw std::runtime_error("get_surface_emission error: " + std::to_string(rc));
        return py::make_tuple(P, vn);
    }

    /** sample_velocity(rec_xyz) → (vx, vy, vz) tuple of float32 (n_rec,) */
    py::tuple sample_velocity_world(py::array_t<float> rec_xyz_arr)
    {
        auto ri = rec_xyz_arr.request();
        int n_rec = (int)(ri.size / 3);
        py::array_t<float> vx({n_rec}), vy({n_rec}), vz({n_rec});
        int rc = coevolver_sample_velocity(handle, n_rec,
            static_cast<const float*>(ri.ptr),
            static_cast<float*>(vx.mutable_unchecked<1>().mutable_data(0)),
            static_cast<float*>(vy.mutable_unchecked<1>().mutable_data(0)),
            static_cast<float*>(vz.mutable_unchecked<1>().mutable_data(0)));
        if (rc != CE_OK)
            throw std::runtime_error("sample_velocity error: " + std::to_string(rc));
        return py::make_tuple(vx, vy, vz);
    }

    void reset() { coevolver_reset(handle); }

    void set_face_fractions(py::array_t<float> vx_frac,
                            py::array_t<float> vy_frac,
                            py::array_t<float> vz_frac)
    {
        auto vx = vx_frac.request();
        auto vy = vy_frac.request();
        auto vz = vz_frac.request();
        int rc = coevolver_set_face_fractions(
            handle,
            static_cast<const float*>(vx.ptr),
            static_cast<const float*>(vy.ptr),
            static_cast<const float*>(vz.ptr));
        if (rc != CE_OK)
            throw std::runtime_error("coevolver_set_face_fractions error: " + std::to_string(rc));
    }

    void set_damping_scale(float scale)
    {
        coevolver_set_damping_scale(handle, scale);
    }

    /* ── Async step ─────────────────────────────────────────────────────── */

    /* Launch background render job.  Returns immediately.
     * Raises if a job is already running. */
    void step_async(int n_samples)
    {
        int rc = coevolver_step_async(handle, n_samples);
        if (rc == CE_ERR_PARAM)
            throw std::runtime_error("AcousticCoEvolver: async job already running");
        if (rc != CE_OK)
            throw std::runtime_error("coevolver_step_async error: " + std::to_string(rc));
    }

    /* Non-blocking progress snapshot.  Returns a dict:
     *   samples_done  : int    — audio samples completed
     *   samples_total : int    — total requested
     *   pct           : float  — 0.0 – 100.0
     *   status        : int    — CE_STATUS_* integer
     *   status_str    : str    — "idle" | "running" | "done" | "cancelled" | "error"
     *   error_code    : int    — CE_OK=0, or error int if status=="error"
     */
    py::dict get_progress() const
    {
        CoEvolverProgress p{};
        coevolver_get_progress(handle, &p);
        const char* label = "unknown";
        switch (p.status) {
            case CE_STATUS_IDLE:      label = "idle";      break;
            case CE_STATUS_RUNNING:   label = "running";   break;
            case CE_STATUS_DONE:      label = "done";      break;
            case CE_STATUS_CANCELLED: label = "cancelled"; break;
            case CE_STATUS_ERROR:     label = "error";     break;
        }
        py::dict d;
        d["samples_done"]  = p.samples_done;
        d["samples_total"] = p.samples_total;
        d["pct"]           = p.samples_total > 0
                             ? 100.0f * (float)p.samples_done / (float)p.samples_total
                             : 0.0f;
        d["status"]        = p.status;
        d["status_str"]    = label;
        d["error_code"]    = p.error_code;
        return d;
    }

    /* Block until done; returns final error code (0 = ok). */
    int wait()
    {
        return coevolver_wait(handle);
    }

    /* Signal cancel, block until the worker exits. */
    void cancel()
    {
        coevolver_cancel(handle);
    }

    /* Non-blocking: 1 if running, 0 otherwise. */
    int is_running() const
    {
        return coevolver_is_running(handle);
    }

    int   n_strings()  const { return coevolver_get_n_strings(handle); }
    int   n_pickups()  const { return coevolver_get_n_pickups(handle); }
    int   n_mics()     const { return coevolver_get_n_mics   (handle); }
    float dt_audio()   const { return coevolver_get_dt_audio (handle); }
    float dt_fdtd()    const { return coevolver_get_dt_fdtd  (handle); }
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

    py::class_<PyRayTracer>(m, "RayTracer",
        R"doc(
Complex spectral 3-D ray tracer.

Build once from triangulated room geometry and frequency bands, then call
trace() to get a float32 segment buffer ready for OpenGL upload.

Parameters (constructor)
------------------------
n_tri : int
    Number of triangles.
verts : float64 array (n_tri, 3, 3)
    Triangle vertices, row-major.
normals : float64 array (n_tri, 3)
    Outward unit normals, row-major.
refl_re, refl_im : float64 array (n_tri, n_bands)
    Complex reflectance per triangle per frequency band.
diffusion : float64 array (n_tri,)
    Diffuse scatter fraction per triangle [0, 1].
freq_hz : float64 array (n_bands,)
    Centre frequencies in Hz.
speed_m_s : float64
    Propagation speed in m/s (default 343.0 for air at 20 °C).
atmo_abs : float64 array (n_bands,)
    Atmospheric absorption in Neper/m per band.
)doc")
        .def(py::init<int,
                      py::array_t<double>,  /* verts      */
                      py::array_t<double>,  /* normals    */
                      py::array_t<double>,  /* refl_re    */
                      py::array_t<double>,  /* refl_im    */
                      py::array_t<double>,  /* diffusion  */
                      py::array_t<double>,  /* freq_hz    */
                      double,               /* speed_m_s  */
                      py::array_t<double>   /* atmo_abs   */
                      >(),
             py::arg("n_tri"),
             py::arg("verts"),
             py::arg("normals"),
             py::arg("refl_re"),
             py::arg("refl_im"),
             py::arg("diffusion"),
             py::arg("freq_hz"),
             py::arg("speed_m_s") = 343.0,
             py::arg("atmo_abs"))
        .def("trace",    &PyRayTracer::trace,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("n_rays")        = 256,
             py::arg("max_bounces")   = 8,
             py::arg("min_amplitude") = 0.005,
             py::arg("seed")          = 42,
             py::arg("out_cap")       = -1,
             R"doc(
Trace rays and return segment buffer.

Returns
-------
np.ndarray, float32, shape (N_segs, 12)
    Columns: x0,y0,z0, x1,y1,z1, src_id, bounce, band, amplitude, phase, path_len
)doc")
        .def("integrate_ir", &PyRayTracer::integrate_ir,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("rec_pos"),
             py::arg("rec_aperture_r"),
             py::arg("speed_m_s")     = 343.0,
             py::arg("sample_rate")   = 44100.0,
             py::arg("n_samples")     = 4096,
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 12,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             R"doc(
Accumulate impulse responses at point receivers.

Returns
-------
(out_re, out_im) : each float32, shape (n_src, n_rec, n_bands, n_samples)
    Real and imaginary parts of the per-band impulse response.
    Combine as out_re + 1j * out_im for complex IR, or sum |·| across bands.

Parameters
----------
rec_pos        : float64 (n_rec, 3) — receiver positions in metres.
rec_aperture_r : float64 (n_rec,)   — capture sphere radii in metres.
                 Larger values improve energy capture but reduce spatial precision.
speed_m_s      : wave speed for converting path length to delay time.
sample_rate    : IR sample rate in Hz.
n_samples      : length of each IR.
)doc")
        .def("integrate_image", &PyRayTracer::integrate_image,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("cam_pos"),
             py::arg("cam_fwd"),
             py::arg("cam_up"),
             py::arg("fov_rad")       = 1.0,
             py::arg("width")         = 512,
             py::arg("height")        = 384,
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 12,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             R"doc(
Render a 2-D acoustic energy image from a pinhole camera viewpoint.

Each ray hit point is projected onto the image plane and its per-band
amplitude |A[b]| is accumulated into the nearest pixel.  The resulting
image maps acoustic energy density onto the camera's view of the scene.

Returns
-------
np.ndarray, float32, shape (n_bands, height, width)
    Accumulated |amplitude| per pixel per band.
    Sum or average across bands for a scalar energy image.

Parameters
----------
cam_pos : float64 (3,) — camera world position.
cam_fwd : float64 (3,) — camera look direction (normalised internally).
cam_up  : float64 (3,) — up hint (orthogonalised to fwd internally).
fov_rad : full vertical field of view in radians (default 1.0 ≈ 57°).
)doc")
        .def("integrate_image_into", &PyRayTracer::integrate_image_into,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("cam_pos"),
             py::arg("cam_fwd"),
             py::arg("cam_up"),
             py::arg("out_image"),
             py::arg("fov_rad")       = 1.0,
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 12,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             R"doc(
Accumulate a camera energy image into a caller-owned float32 buffer.

The output buffer is not cleared. Reuse the same (n_bands, height, width)
array across massive ray batches, clearing only when you want to erase history.
)doc")
        .def("trace_integrate_image_into", &PyRayTracer::trace_integrate_image_into,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("cam_pos"),
             py::arg("cam_fwd"),
             py::arg("cam_up"),
             py::arg("out_image"),
             py::arg("out_segs"),
             py::arg("fov_rad")       = 1.0,
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 12,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             R"doc(
Trace and integrate in one hot C loop.

out_image is accumulated in place. out_segs is a reusable float32
(capacity, 12) segment buffer; if it fills, integration continues and the
return value reports how many segment records were written.
)doc")
        .def_property_readonly("n_bands", &PyRayTracer::n_bands)
        .def_property_readonly("n_tris",  &PyRayTracer::n_tris,
             "Number of triangles in the scene.")
        .def("trace_surface", &PyRayTracer::trace_surface,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("n_rays")         = 256,
             py::arg("max_bounces")    = 8,
             py::arg("min_amplitude")  = 0.005,
             py::arg("seed")           = 42,
             py::arg("out_cap")        = -1,
R"doc(
Trace rays and accumulate per-triangle irradiance.

Returns a dict with:
  'segs'     : float32 (N_segs, 12) — segment records (same layout as trace())
  'direct'   : float32 (n_tri, n_bands) — direct-illumination irradiance per triangle
  'indirect' : float32 (n_tri, n_bands) — reflected irradiance per triangle
)doc");

    py::class_<PyFieldSolver>(m, "FieldSolver",
        R"doc(
Unified acoustic / EM complex field solver.

Computes the coherent complex transfer function H[source → receiver, freq]
by tracing rays through a triangulated scene and summing contributions at
each receiver.  Supports acoustic scalar pressure and EM vector E-field
(full Jones matrix).  A hybrid modal + ray model is used below the
Schroeder frequency.

Parameters (constructor)
------------------------
scene : dict
    verts        : float64 array (n_tri, 9) — flattened triangle vertices
    normals      : float64 array (n_tri, 3) — outward unit normals
    mat_idx      : int32   array (n_tri,)   — per-triangle material index
    n_mats       : int                      — number of materials
    mat_n_re     : float64 array (n_mats, n_bands) — Re(Z/Z_air) or Re(ñ)
    mat_n_im     : float64 array (n_mats, n_bands) — Im part
    mat_diffusion: float64 array (n_mats,)         — diffuse scatter fraction
    medium_n_re  : float — medium real refractive index (1.0 for air)
    medium_n_im  : float — medium imaginary index (0.0 for air)
receivers : list of dicts
    pos          : float64 (3,)  — receiver centre
    axis         : float64 (3,)  — polar axis toward source
    polar_type   : int (0=OMNI, 1=CARDIOID, 2=FIGURE8, 3=HYPERCARDIOID, 4=APERTURE)
    aperture_r   : float — aperture disc radius (APERTURE only)
    pol_s        : float64 (3,) — s-pol detector axis (EM only)
    pol_p        : float64 (3,) — p-pol detector axis (EM only)
freq_hz   : float64 array (n_bands,) — centre frequencies
speed_m_s : float — wave speed (343.0 for air)
mode      : str — "acoustic" or "em"
)doc")
        .def(py::init<py::dict, py::list, py::array_t<double>, double, std::string>(),
             py::arg("scene"),
             py::arg("receivers"),
             py::arg("freq_hz"),
             py::arg("speed_m_s") = 343.0,
             py::arg("mode")      = "acoustic")
        .def("solve_acoustic", &PyFieldSolver::solve_acoustic,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 8,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             py::arg("schroeder_hz")  = 0.0,
             R"doc(
Compute acoustic transfer matrix H[src, rec, band].

Returns
-------
np.ndarray, complex128, shape (n_src, n_rec, n_bands)
    H[si, ri, b] = complex pressure at receiver ri from source si at band b.
)doc")
        .def("solve_em", &PyFieldSolver::solve_em,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("src_pol_re"),
             py::arg("src_pol_im"),
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 8,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             py::arg("schroeder_hz")  = 0.0,
             R"doc(
Compute EM Jones transfer matrix H[src, rec, band, p_out, p_in].

Returns
-------
np.ndarray, complex128, shape (n_src, n_rec, n_bands, 2, 2)
    H[si, ri, b, po, pi] = Jones matrix entry.
    p_out/p_in: 0 = s-polarisation, 1 = p-polarisation.
)doc")
        .def_property_readonly("n_bands",     &PyFieldSolver::n_bands)
        .def_property_readonly("n_receivers", &PyFieldSolver::n_receivers)
        .def_property_readonly("mode",        &PyFieldSolver::mode);

    /* ── AcousticFDTD ──────────────────────────────────────────────────────── */

    py::class_<PyAcousticFDTD>(m, "AcousticFDTD",
        R"doc(
3-D acoustic FDTD solver with coupled Kirchhoff plate and PML absorbing boundaries.

Bridge excitation must be supplied as structural plate load through
AcousticCoEvolver or fdtd_inject_bridge_drive.  The older direct pressure
inject_bridge(signal, derivative, scale) path is disabled because it bypasses
the plate impedance.

The full 3-D pressure field is returned by get_pressure_field() for
volumetric GL rendering, replacing the coarse nearest-centroid surface
illumination with a spatially continuous acoustic field.

Parameters (constructor)
------------------------
Nx, Ny, Nz         : int — grid dimensions
dx                 : float — cell size in metres
c                  : float — speed of sound (m/s)
rho_air            : float — air density (kg/m³)
cell_type          : uint8 array (Nx·Ny·Nz) — FDTD_AIR=0 FDTD_WALL=1 FDTD_PLATE=2 FDTD_PML=3
plate_iz           : int — Z index of the soundboard
plate_active       : uint8 array (Nx·Ny) — 1=inside guitar outline
plate_mass_density : float — ρ_s·h (kg/m²), typical spruce top: 7.0
plate_stiffness_D  : float — bending stiffness (N·m), typical 3mm spruce: 0.45
n_pml              : int — PML absorbing layer thickness in cells
)doc")
        .def(py::init<int,int,int, float,float,float,
                      py::array_t<uint8_t>,
                      int,
                      py::array_t<uint8_t>,
                      float,float,float,float,int>(),
             py::arg("Nx"), py::arg("Ny"), py::arg("Nz"),
             py::arg("dx"),
             py::arg("c")       = 343.0f,
             py::arg("rho_air") = 1.21f,
             py::arg("cell_type"),
             py::arg("plate_iz"),
             py::arg("plate_active"),
             py::arg("plate_mass_density") = 7.0f,
             py::arg("plate_stiffness_D")  = 0.45f,
             py::arg("plate_alpha_M")      = 2.0f,
             py::arg("plate_beta_K")       = 1e-5f,
             py::arg("n_pml")              = 10)
        .def("set_bridge_sources", &PyAcousticFDTD::set_bridge_sources,
             py::arg("cell_indices"), py::arg("weights"),
             "Register bridge source cells and their Gaussian kernel weights.")
        .def("set_face_fractions", &PyAcousticFDTD::set_face_fractions,
             py::arg("vx_frac"), py::arg("vy_frac"), py::arg("vz_frac"),
             "Set per-face open-area fractions (float32 arrays) for cut-cell body boundary. "
             "Sizes: vx (Nx-1)*Ny*Nz, vy Nx*(Ny-1)*Nz, vz Nx*Ny*(Nz-1).")
        .def("inject_bridge", &PyAcousticFDTD::inject_bridge,
             py::arg("signal_val"), py::arg("signal_ddt"),
             py::arg("force_scale") = 1.0f,
             "Disabled legacy direct-pressure bridge injection; use AcousticCoEvolver.")
        .def("step", &PyAcousticFDTD::step,
             py::arg("n_steps") = 1,
             "Advance FDTD by n_steps time steps.  Returns FDTD_OK=0 or error code.")
        .def("reset",  &PyAcousticFDTD::reset,  "Zero all pressure and plate fields.")
        .def("get_pressure_field",      &PyAcousticFDTD::get_pressure_field,
             "Return full 3-D pressure field as float32 array, shape (Nx,Ny,Nz).")
        .def("get_plate_displacement",  &PyAcousticFDTD::get_plate_displacement,
             "Return 2-D Kirchhoff plate displacement w(x,y), shape (Nx,Ny), metres.")
        .def("sample_pressure",         &PyAcousticFDTD::sample_pressure,
             py::arg("rec_xyz"),
             "Trilinear interpolation at grid-space positions (n_rec,3) → (n_rec,) float32.")
        .def("sample_velocity", &PyAcousticFDTD::sample_velocity,
             py::arg("rec_xyz"),
             R"doc(Sample particle velocity at grid-space positions.

Returns tuple (vx, vy, vz) of float32 arrays, each (n_rec,), units m/s.
Uses staggered trilinear interpolation from the live leapfrog velocity fields.
rec_xyz: float32 (n_rec, 3) in grid coordinates.
)doc")
        .def("get_surface_emission", &PyAcousticFDTD::get_surface_emission,
             py::arg("xyz"), py::arg("normals"),
             R"doc(Sample Kirchhoff–Helmholtz boundary data on a surface.

Returns (P, vn) tuple:
  P  : float32 (n_surf,) — pressure in Pa
  vn : float32 (n_surf,) — outward normal velocity in m/s

xyz     : float32 (n_surf, 3) — grid-space positions of surface points
normals : float32 (n_surf, 3) — outward unit normals at each point

Use this to couple the instrument's acoustic emission into a room simulator.
)doc")
        .def("get_velocity_dims", &PyAcousticFDTD::get_velocity_dims,
             "Return (Nvx, Nvy, Nvz) — sizes of the three staggered velocity arrays.")
        .def("get_velocity_x", &PyAcousticFDTD::get_velocity_x,
             "Staggered Vx field: float32 (Nx-1)*Ny*Nz, index k+Nz*(j+Ny*i).")
        .def("get_velocity_y", &PyAcousticFDTD::get_velocity_y,
             "Staggered Vy field: float32 Nx*(Ny-1)*Nz, index k+Nz*(j+(Ny-1)*i).")
        .def("get_velocity_z", &PyAcousticFDTD::get_velocity_z,
             "Staggered Vz field: float32 Nx*Ny*(Nz-1), index k+(Nz-1)*(j+Ny*i).")
        .def_property_readonly("dt",         &PyAcousticFDTD::get_dt)
        .def_property_readonly("step_count", &PyAcousticFDTD::get_step_count)
        .def_property_readonly("Nx",         [](const PyAcousticFDTD& o){ return o.Nx; })
        .def_property_readonly("Ny",         [](const PyAcousticFDTD& o){ return o.Ny; })
        .def_property_readonly("Nz",         [](const PyAcousticFDTD& o){ return o.Nz; });

    /* ── AcousticCoEvolver ───────────────────────────────────────────────────── */

    py::class_<PyAcousticCoEvolver>(m, "AcousticCoEvolver",
        R"doc(
Unified C co-evolution engine for physically accurate instrument simulation.

Runs the complete instrument physics loop with no Python round-trips per sample:
  1. 1-D string FDTD (leapfrog, two polarisations, arbitrary 3-D procession)
  2. Lazy modal projection for spectral analysis
  3. Bridge velocity sum → acoustic body FDTD injection (derivative-coupling)
  4. Kirchhoff plate + 3-D acoustic FDTD sub-steps (CFL-stable)
  5. Pickup integration: single-coil / humbucker (B-kernel) / piezo (slope)
  6. Mic sampling from FDTD pressure field
  7. Plate displacement → string saddle BC (two-way coupling)

Suitable for acoustic guitars (full acoustic output) and electric guitars
(near-silent body; mic output captures the physically correct tiny pressure
while pickup output captures string velocity via B-kernel integration).

Parameters (constructor)
------------------------
string_defs : list of dicts
    Each dict: path_xyz (float32 (n_segs+1,3)), tension_N, linear_mass_kgm,
    damping, optional stiffness_EI and axial_stiffness_N
pickup_defs : list of dicts
    Each dict: type (0=SINGLE_COIL,1=HUMBUCKER,2=PIEZO), pos (3,), axis (3,),
    pole_sigma, coil_spacing, sensitivity, string_mask (int bitmask)
mic_defs : list of dicts
    Each dict: pos (3,), gain
body : dict
    Nx, Ny, Nz, dx, c, rho_air, cell_type (uint8 Nx*Ny*Nz),
    plate_iz, plate_active (uint8 Nx*Ny),
    plate_mass_density, plate_stiffness_D, n_pml,
    bridge_src_xyz (float32 (n_bridge,3)), bridge_sigma
sample_rate   : float — audio sample rate (Hz)
modal_stride  : int   — steps between modal projection updates (default 16)
force_scale   : float — deprecated compatibility parameter; physical coupling uses explicit impedances
)doc")
        .def(py::init<py::list, py::list, py::list, py::dict, float, int, float>(),
             py::arg("string_defs"),
             py::arg("pickup_defs"),
             py::arg("mic_defs"),
             py::arg("body"),
             py::arg("sample_rate"),
             py::arg("modal_stride") = 16,
             py::arg("force_scale")  = 1.0f)
        .def("step", &PyAcousticCoEvolver::step,
             py::arg("n_samples") = 1,
             "Advance the co-evolver by n_samples audio samples. GIL is released during C++ work.")
        .def("step_block_with_drive", &PyAcousticCoEvolver::step_block_with_drive,
             py::arg("drive_blocks"),
             py::arg("force_pos_norm") = 0.15f,
             py::arg("n_samples"),
             R"doc(
Inject per-string drive blocks, step physics, return mic 0 output.

This is the primary real-time body-drive API.  Equivalent to calling
inject_string_force for each string then step(), but avoids the Python
round-trips between inject and step, and releases the GIL for the full
C++ physics pass.

Parameters
----------
drive_blocks : list of float32 arrays, one per string, each (n_samples,).
               Pass None or [] to advance with silence (no injection).
force_pos_norm : fractional string position for force injection (0=nut, 1=saddle).
n_samples : block length.

Returns
-------
np.ndarray, float32, shape (n_samples,) — mic 0 output for this block.
)doc")
        .def("pluck_string", &PyAcousticCoEvolver::pluck_string,
             py::arg("string_idx"), py::arg("position_norm"), py::arg("amplitude"),
             "Excite a string with a raised-cosine pluck at position_norm (0=nut, 1=saddle).")
        .def("schedule_pluck", &PyAcousticCoEvolver::schedule_pluck,
             py::arg("onset_sample"), py::arg("string_idx"),
             py::arg("position_norm"), py::arg("amplitude"),
             "Pre-register a pluck for the next step_async call; onset_sample is relative to "
             "the start of that call.  Calls must be in non-decreasing onset order.")
        .def("clear_pluck_schedule", &PyAcousticCoEvolver::clear_pluck_schedule,
             "Clear all pre-registered pluck events.")
        .def("inject_string_force", &PyAcousticCoEvolver::inject_string_force,
             py::arg("string_idx"), py::arg("force_buf"), py::arg("force_pos_norm") = 0.1f,
             "Inject an external per-sample force sequence (float32) into a string.")
        .def("get_pickup_output", &PyAcousticCoEvolver::get_pickup_output,
             py::arg("pickup_idx"), py::arg("n_samples"),
             "Return the last n_samples from pickup pickup_idx as float32 array.")
        .def("get_mic_output", &PyAcousticCoEvolver::get_mic_output,
             py::arg("mic_idx"), py::arg("n_samples"),
             "Return the last n_samples from mic mic_idx as float32 array.")
        .def("get_pressure_field", &PyAcousticCoEvolver::get_pressure_field,
             "Return full 3-D pressure field, float32 shape (Nx,Ny,Nz).")
        .def("get_plate_displacement", &PyAcousticCoEvolver::get_plate_displacement,
             "Return 2-D plate displacement w(x,y), float32 shape (Nx,Ny).")
        .def("get_modal_amplitudes", &PyAcousticCoEvolver::get_modal_amplitudes,
             py::arg("string_idx"), py::arg("n_modes") = 32,
             "Return (re, im) tuple of float32 arrays, each (n_modes,).")
        .def("sample_pressure", &PyAcousticCoEvolver::sample_pressure,
             py::arg("rec_xyz"),
             "Sample pressure at world-space positions (n_rec,3) → (n_rec,) float32.")
        .def("get_string_velocity", &PyAcousticCoEvolver::get_string_velocity,
             py::arg("string_idx"),
             "Return world-space transverse velocity (n_segs, 3) for a string (auto-size).")
        .def("get_string_velocity_n", &PyAcousticCoEvolver::get_string_velocity_n,
             py::arg("string_idx"), py::arg("n_segs"),
             "Return world-space transverse velocity (n_segs, 3), explicit n_segs.")
        .def("get_string_displacement_n", &PyAcousticCoEvolver::get_string_displacement_n,
             py::arg("string_idx"), py::arg("n_segs"),
             "Return world-space transverse displacement (n_segs, 3) in metres, explicit n_segs.")
        .def("get_string_position_n", &PyAcousticCoEvolver::get_string_position_n,
             py::arg("string_idx"), py::arg("n_nodes"),
             "Return absolute world-space string node positions (n_nodes, 3), explicit n_nodes.")
        .def("get_surface_emission", &PyAcousticCoEvolver::get_surface_emission,
             py::arg("xyz"), py::arg("normals"),
             R"doc(Sample Kirchhoff-Helmholtz boundary data for room simulation coupling.

Returns (P, vn):
  P  : float32 (n_surf,) — pressure in Pa
  vn : float32 (n_surf,) — outward normal velocity in m/s

xyz     : float32 (n_surf, 3) — world-space surface point positions (metres)
normals : float32 (n_surf, 3) — outward unit normals at each surface point

P and v_n together fully characterise the radiated sound field outside the
surface (Kirchhoff-Helmholtz integral boundary condition).  Feed directly
into a room acoustic simulator as the instrument source.
)doc")
        .def("sample_velocity", &PyAcousticCoEvolver::sample_velocity_world,
             py::arg("rec_xyz"),
             R"doc(Sample particle velocity at world-space positions.

Returns (vx, vy, vz) tuple of float32 (n_rec,) arrays, units m/s.
rec_xyz: float32 (n_rec, 3) in world coordinates (metres).
Uses staggered trilinear interpolation — phase-exact, no approximation.
)doc")
        .def("reset", &PyAcousticCoEvolver::reset,
             "Reset all string, plate, pressure, and output ring fields to zero.")
        .def("set_face_fractions", &PyAcousticCoEvolver::set_face_fractions,
             py::arg("vx_frac"), py::arg("vy_frac"), py::arg("vz_frac"),
             "Set per-face open-area fractions for cut-cell body boundary. "
             "Delegates to fdtd_set_face_fractions on the internal FDTD.")
        .def("set_damping_scale", &PyAcousticCoEvolver::set_damping_scale,
             py::arg("scale"),
             "Set global string damping multiplier (>1.0 overdamps for warm-up; 1.0 = physical).")
        .def("step_async", &PyAcousticCoEvolver::step_async,
             py::arg("n_samples"),
             "Launch a background thread to advance by n_samples. Returns immediately.")
        .def("get_progress", &PyAcousticCoEvolver::get_progress,
             "Non-blocking progress snapshot: dict with samples_done, samples_total, pct, "
             "status, status_str ('idle'|'running'|'done'|'cancelled'|'error'), error_code.")
        .def("wait", &PyAcousticCoEvolver::wait,
             "Block until the current async job finishes. Returns 0 on success.")
        .def("cancel", &PyAcousticCoEvolver::cancel,
             "Signal cancellation and block until the worker thread exits.")
        .def("is_running", &PyAcousticCoEvolver::is_running,
             "1 if an async job is currently running, 0 otherwise.")
        .def_property_readonly("n_strings", &PyAcousticCoEvolver::n_strings)
        .def_property_readonly("n_pickups", &PyAcousticCoEvolver::n_pickups)
        .def_property_readonly("n_mics",    &PyAcousticCoEvolver::n_mics)
        .def_property_readonly("dt_audio",  &PyAcousticCoEvolver::dt_audio)
        .def_property_readonly("dt_fdtd",   &PyAcousticCoEvolver::dt_fdtd);

    /* ── Constants ──────────────────────────────────────────────────────────── */
    m.attr("PICKUP_SINGLE_COIL") = PICKUP_SINGLE_COIL;
    m.attr("PICKUP_HUMBUCKER")   = PICKUP_HUMBUCKER;
    m.attr("PICKUP_PIEZO")       = PICKUP_PIEZO;
    m.attr("FDTD_AIR")   = (int)FDTD_AIR;
    m.attr("FDTD_WALL")  = (int)FDTD_WALL;
    m.attr("FDTD_PLATE") = (int)FDTD_PLATE;
    m.attr("FDTD_PML")   = (int)FDTD_PML;
    /* Async worker status constants */
    m.attr("CE_STATUS_IDLE")      = CE_STATUS_IDLE;
    m.attr("CE_STATUS_RUNNING")   = CE_STATUS_RUNNING;
    m.attr("CE_STATUS_DONE")      = CE_STATUS_DONE;
    m.attr("CE_STATUS_CANCELLED") = CE_STATUS_CANCELLED;
    m.attr("CE_STATUS_ERROR")     = CE_STATUS_ERROR;
}
