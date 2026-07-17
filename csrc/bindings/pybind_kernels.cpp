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
#include "field_grid.h"
#include "rt_field_solver.h"
#include "acoustic_amr.h"
#include "acoustic_fdtd.h"
#include "acoustic_coevolver.h"
#include "acoustic_pressure_backend.h"
#include "doc_renderer.h"
#include "base_rasterizer.h"
#include "emitter_angle_kernel.h"
#include "tile_overlap.h"
#include "player_clip.h"
#include "surface_spline.h"
#include "lens_optics.h"
#include "optical_handlers.h"
#include "exposure_backend.h"
#include "ray_pipeline.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

static py::tuple estimate_lens_camera_jacobians_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> payload,
    py::array_t<double, py::array::c_style | py::array::forcecast> film_origins,
    py::array_t<double, py::array::c_style | py::array::forcecast> aperture_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> base_exit_origins,
    py::array_t<double, py::array::c_style | py::array::forcecast> base_exit_dirs,
    double plate_radius,
    double aperture_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> tb,
    py::array_t<double, py::array::c_style | py::array::forcecast> tc,
    int n_threads)
{
    auto p = payload.request();
    auto fo = film_origins.request();
    auto ap = aperture_points.request();
    auto bo = base_exit_origins.request();
    auto bd = base_exit_dirs.request();
    auto tbv = tb.request();
    auto tcv = tc.request();
    if (p.ndim != 1) throw std::runtime_error("payload must be 1-D float32");
    if (fo.ndim != 2 || ap.ndim != 2 || bo.ndim != 2 || bd.ndim != 2 ||
        fo.shape[1] != 3 || ap.shape[1] != 3 || bo.shape[1] != 3 || bd.shape[1] != 3)
        throw std::runtime_error("ray arrays must have shape (N, 3)");
    if (fo.shape[0] != ap.shape[0] || fo.shape[0] != bo.shape[0] || fo.shape[0] != bd.shape[0])
        throw std::runtime_error("ray arrays must have the same N");
    if (tbv.size != 3 || tcv.size != 3)
        throw std::runtime_error("tb and tc must have length 3");
    const int n = static_cast<int>(fo.shape[0]);
    py::array_t<float> ap_jac(n);
    py::array_t<float> phase_jac(n);
    float* ap_jac_ptr = static_cast<float*>(ap_jac.mutable_data());
    float* phase_jac_ptr = static_cast<float*>(phase_jac.mutable_data());
    {
        py::gil_scoped_release release;
        const int rc = lens_optics_estimate_camera_jacobians(
            static_cast<const float*>(p.ptr),
            static_cast<int>(p.size),
            static_cast<const double*>(fo.ptr),
            static_cast<const double*>(ap.ptr),
            static_cast<const double*>(bo.ptr),
            static_cast<const double*>(bd.ptr),
            n,
            plate_radius,
            aperture_radius,
            static_cast<const double*>(tbv.ptr),
            static_cast<const double*>(tcv.ptr),
            n_threads,
            ap_jac_ptr,
            phase_jac_ptr);
        if (rc != 0) throw std::runtime_error("lens_optics_estimate_camera_jacobians failed");
    }
    return py::make_tuple(ap_jac, phase_jac);
}

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

struct FaceRecordKey
{
    int64_t lo0 = 0;
    int64_t hi0 = 0;
    int64_t lo1 = 0;
    int64_t hi1 = 0;
    int32_t cell = 0;
};

static bool operator<(const FaceRecordKey& a, const FaceRecordKey& b)
{
    return std::tie(a.lo0, a.hi0, a.lo1, a.hi1, a.cell)
         < std::tie(b.lo0, b.hi0, b.lo1, b.hi1, b.cell);
}

static py::dict build_amr_faces_sorted_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> centers_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> half_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> types_arr)
{
    auto centers = centers_arr.request();
    auto half = half_arr.request();
    auto types = types_arr.request();
    if (centers.ndim != 2 || centers.shape[1] != 3)
        throw std::runtime_error("build_amr_faces_sorted_cpp: centers must have shape (N,3)");
    if (half.ndim != 2 || half.shape[0] != centers.shape[0] || half.shape[1] != 3)
        throw std::runtime_error("build_amr_faces_sorted_cpp: half_sizes must have shape (N,3)");
    if (types.ndim != 1 || types.shape[0] != centers.shape[0])
        throw std::runtime_error("build_amr_faces_sorted_cpp: types must have shape (N,)");

    const int64_t N64 = centers.shape[0];
    if (N64 > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("build_amr_faces_sorted_cpp: too many cells for int32 face indices");
    const int N = static_cast<int>(N64);
    const double* C = static_cast<const double*>(centers.ptr);
    const double* H = static_cast<const double*>(half.ptr);
    const uint8_t* T = static_cast<const uint8_t*>(types.ptr);

    std::vector<int32_t> neg;
    std::vector<int32_t> pos;
    std::vector<uint8_t> axis;
    std::vector<double> area;
    std::vector<double> open_fraction;
    std::vector<double> distance;
    std::vector<std::array<int64_t, 3>> lo_key(N);
    std::vector<std::array<int64_t, 3>> hi_key(N);

    {
        py::gil_scoped_release release;

        double key_scale = std::numeric_limits<double>::infinity();
        for (int i = 0; i < N; ++i) {
            for (int ax = 0; ax < 3; ++ax) {
                const double w = 2.0 * H[i * 3 + ax];
                if (std::isfinite(w) && w > 0.0 && w < key_scale)
                    key_scale = w;
            }
        }
        if (!std::isfinite(key_scale) || key_scale <= 0.0)
            throw std::runtime_error("build_amr_faces_sorted_cpp: positive cell sizes required");

        for (int i = 0; i < N; ++i) {
            for (int ax = 0; ax < 3; ++ax) {
                lo_key[i][ax] = static_cast<int64_t>(
                    std::llround((C[i * 3 + ax] - H[i * 3 + ax]) / key_scale));
                hi_key[i][ax] = static_cast<int64_t>(
                    std::llround((C[i * 3 + ax] + H[i * 3 + ax]) / key_scale));
            }
        }

        neg.reserve(static_cast<size_t>(N) * 3);
        pos.reserve(static_cast<size_t>(N) * 3);
        axis.reserve(static_cast<size_t>(N) * 3);
        area.reserve(static_cast<size_t>(N) * 3);
        open_fraction.reserve(static_cast<size_t>(N) * 3);
        distance.reserve(static_cast<size_t>(N) * 3);

        for (int ax = 0; ax < 3; ++ax) {
            const int ax1 = (ax + 1) % 3;
            const int ax2 = (ax + 2) % 3;
            std::unordered_map<int64_t, std::vector<FaceRecordKey>> hi_planes;
            std::unordered_map<int64_t, std::vector<FaceRecordKey>> lo_planes;
            hi_planes.reserve(static_cast<size_t>(N));
            lo_planes.reserve(static_cast<size_t>(N));

            for (int i = 0; i < N; ++i) {
                FaceRecordKey rec{
                    lo_key[i][ax1], hi_key[i][ax1],
                    lo_key[i][ax2], hi_key[i][ax2],
                    static_cast<int32_t>(i)
                };
                hi_planes[hi_key[i][ax]].push_back(rec);
                lo_planes[lo_key[i][ax]].push_back(rec);
            }

            for (auto& plane_pair : hi_planes) {
                auto lo_it = lo_planes.find(plane_pair.first);
                if (lo_it == lo_planes.end())
                    continue;

                auto& hi_records = plane_pair.second;
                auto& lo_records = lo_it->second;
                std::sort(hi_records.begin(), hi_records.end(),
                          [](const FaceRecordKey& a, const FaceRecordKey& b) {
                              return std::tie(a.lo0, a.hi0, a.lo1, a.hi1, a.cell)
                                   < std::tie(b.lo0, b.hi0, b.lo1, b.hi1, b.cell);
                          });
                std::vector<FaceRecordKey> lo_start_sorted = lo_records;
                std::vector<FaceRecordKey> lo_end_sorted = lo_records;
                std::sort(lo_start_sorted.begin(), lo_start_sorted.end(),
                          [](const FaceRecordKey& a, const FaceRecordKey& b) {
                              return std::tie(a.lo0, a.hi0, a.lo1, a.hi1, a.cell)
                                   < std::tie(b.lo0, b.hi0, b.lo1, b.hi1, b.cell);
                          });
                std::sort(lo_end_sorted.begin(), lo_end_sorted.end(),
                          [](const FaceRecordKey& a, const FaceRecordKey& b) {
                              return std::tie(a.hi0, a.lo0, a.lo1, a.hi1, a.cell)
                                   < std::tie(b.hi0, b.lo0, b.lo1, b.hi1, b.cell);
                          });

                std::set<FaceRecordKey> active;
                size_t start_ptr = 0;
                size_t end_ptr = 0;
                const size_t n_lo = lo_records.size();

                for (const auto& h : hi_records) {
                    if (h.hi0 <= h.lo0 || h.hi1 <= h.lo1)
                        continue;
                    while (end_ptr < n_lo && lo_end_sorted[end_ptr].hi0 <= h.lo0) {
                        active.erase(lo_end_sorted[end_ptr]);
                        ++end_ptr;
                    }
                    while (start_ptr < n_lo && lo_start_sorted[start_ptr].lo0 < h.hi0) {
                        const auto& rec = lo_start_sorted[start_ptr];
                        if (rec.hi0 > h.lo0)
                            active.insert(rec);
                        ++start_ptr;
                    }

                    for (const auto& l : active) {
                        if (h.cell == l.cell)
                            continue;
                        if (l.hi0 <= h.lo0 || l.lo0 >= h.hi0)
                            continue;
                        const int64_t ov1_lo = std::max(h.lo1, l.lo1);
                        const int64_t ov1_hi = std::min(h.hi1, l.hi1);
                        if (ov1_hi <= ov1_lo)
                            continue;
                        const int64_t ov0_lo = std::max(h.lo0, l.lo0);
                        const int64_t ov0_hi = std::min(h.hi0, l.hi0);
                        if (ov0_hi <= ov0_lo)
                            continue;

                        const double a = static_cast<double>(ov0_hi - ov0_lo)
                                       * static_cast<double>(ov1_hi - ov1_lo)
                                       * key_scale * key_scale;
                        const double d = std::abs(C[l.cell * 3 + ax] - C[h.cell * 3 + ax]);
                        if (!(a > 0.0 && d > 0.0))
                            continue;
                        const double op = (T[h.cell] == 1 || T[l.cell] == 1) ? 0.0 : 1.0;
                        neg.push_back(h.cell);
                        pos.push_back(l.cell);
                        axis.push_back(static_cast<uint8_t>(ax));
                        area.push_back(a);
                        open_fraction.push_back(op);
                        distance.push_back(d);
                    }
                }
            }
        }
    }

    py::dict out;
    out["neg"] = py::array_t<int32_t>(neg.size(), neg.data());
    out["pos"] = py::array_t<int32_t>(pos.size(), pos.data());
    out["axis"] = py::array_t<uint8_t>(axis.size(), axis.data());
    out["area"] = py::array_t<double>(area.size(), area.data());
    out["open_fraction"] = py::array_t<double>(open_fraction.size(), open_fraction.data());
    out["distance"] = py::array_t<double>(distance.size(), distance.data());
    return out;
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

/* Declared in ray_tracer.cpp — not in the public header, called here only. */
extern int ray_pipeline_trace_sync(
    RayTracerState*          st,
    const RayPipelineConfig* cfg,
    const RayIntent*         intents,
    int                      n_intents,
    std::vector<RayRecord>&  out);

struct PyRayTracer
{
    RayTracerState*   handle   = nullptr;
    int               _n_bands = 0;
    int               _n_tris  = 0;
    std::vector<double> _freq_hz;
    int               _camera_vis_mode    = RT_CAM_VIS_AS_IS;
    int               _transparent_mode   = RT_CAM_TRANSPARENCY_BLOCK;
    bool              _depth_cull_enabled  = false;
    double            _depth_cull_m        = 0.0;

    /* Persistent pipeline — created on first submit_rays(), destroyed with tracer. */
    RayPipelineState* _pipeline   = nullptr;
    std::mutex        _pipeline_mu;
    double            _default_min_amplitude = 1e-6;  /* raised before first create */
    int               _max_intent_queue      = 0;     /* 0 = unbounded             */

    /* Sensor image parameters — cached so configure_sensor_image() can be
     * called before the pipeline is created (lazy init on first submit_rays). */
    float _sensor_plate_x      = 0.0f;
    float _sensor_plate_half_w = 0.0f;
    float _sensor_plate_half_h = 0.0f;
    int   _sensor_res          = 0;
    float _sensor_bdpt_eps     = 0.008f;
    float _sensor_target_x     = 0.0f;
    float _sensor_target_y     = 0.0f;
    float _sensor_target_z     = 0.0f;
    float _sensor_target_r     = 0.0f;
    int   _sensor_target_mode  = 0;
    bool  _sensor_pose_configured = false;
    std::array<float, 3> _sensor_center = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> _sensor_right = {0.0f, 1.0f, 0.0f};
    std::array<float, 3> _sensor_up = {0.0f, 0.0f, 1.0f};
    std::array<float, 3> _aperture_center = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> _aperture_right = {0.0f, 1.0f, 0.0f};
    std::array<float, 3> _aperture_up = {0.0f, 0.0f, 1.0f};
    std::atomic<uint32_t> _bdpt_subpath_counter{1u};

    /* T5 connection config */
    float    _t5_min_geom          = 1e-8f;
    uint32_t _t5_light_batch_size  = 0;
    uint32_t _t5_cam_batch_size    = 0;
    uint32_t _t5_sensor_tile_size  = 0;
    uint64_t _t5_pair_budget       = 200000000ull;  /* per-pass connect budget (tunable via set_t5_pair_budget) */
    float    _t5_backlog_share_base    = 0.15f;
    float    _t5_backlog_share_max     = 0.60f;
    int      _t5_backlog_max_snapshots = 8;
    uint64_t _t5_backlog_max_bytes     = 6ull << 30;
    uint64_t _bdpt_record_float_budget = 1100000000ull;
    bool     _t5_profile           = false;
    bool     _vcm_enabled          = true;
    float    _vcm_merge_radius_m   = 0.002f;
    float    _vcm_radius_alpha     = 0.7f;


    /* Flash modifier config */
    int   _flash_modifier_type   = static_cast<int>(FlashModifierType::SNOOT);
    float _flash_modifier_param0 = 0.0f;
    float _flash_modifier_param1 = 0.0f;

    /* GPU compute config — stored before pipeline creation so first submit_rays
     * can enable the GPU backend.  Ignored after the pipeline is created. */
    bool        _use_gpu_compute = false;
    bool        _force_cpu_t5 = false;
    bool        _gpu_all_stages  = false;
    bool        _gpu_skip_record_readback = false;
    std::string _shader_dir;
    uint64_t    _gl_display_hglrc = 0;  /* Pygame display HGLRC for WGL object sharing */
    uint64_t    _gl_display_hdc   = 0;  /* Pygame display HDC for pixel-format matching */
    std::array<float, MAX_SPECTRAL_BANDS * 3> _uv_blit_weights{};
    int         _uv_blit_n_bands = 0;
    int         _uv_blit_mode = 0;
    bool        _sensor_mipmap_enabled = false;
    uint32_t    _sensor_mipmap_max_nodes = 0;
    uint32_t    _sensor_mipmap_max_depth = 0;
    uint32_t    _sensor_mipmap_samples_per_epoch = 0;
    bool        _sensor_priority_network_enabled = false;
    std::array<float, SENSOR_PRIORITY_NETWORK_PARAMS>
                _sensor_priority_network_params{};
    std::vector<float> _sensor_requested_priority_map;
    uint32_t _sensor_requested_priority_res = 0;
    std::vector<float> _sensor_restore_rgb;
    std::vector<float> _sensor_restore_weight;
    std::vector<uint32_t> _sensor_dirty_sites;
    uint32_t _sensor_restore_res = 0;

    RayPipelineState* _get_pipeline(int max_children = 2, int seed = 42) {
        std::lock_guard<std::mutex> lk(_pipeline_mu);
        if (!_pipeline) {
            RayPipelineConfig cfg;
            cfg.max_children        = max_children;
            cfg.seed                = seed;
            cfg.min_amplitude       = _default_min_amplitude;
            cfg.max_intent_queue    = _max_intent_queue;
            cfg.use_gpu_compute             = _use_gpu_compute;
            cfg.force_cpu_t5                = _force_cpu_t5;
            cfg.gpu_all_stages              = _gpu_all_stages;
            cfg.gpu_skip_record_readback    = _gpu_skip_record_readback;
            cfg.sensor_mipmap_enabled = _sensor_mipmap_enabled;
            cfg.sensor_mipmap_max_nodes = _sensor_mipmap_max_nodes;
            cfg.sensor_mipmap_max_depth = _sensor_mipmap_max_depth;
            cfg.sensor_mipmap_samples_per_epoch = _sensor_mipmap_samples_per_epoch;
            cfg.sensor_priority_network_enabled = _sensor_priority_network_enabled;
            cfg.sensor_priority_network_params = _sensor_priority_network_params;
            cfg.sensor_requested_priority_map = _sensor_requested_priority_map;
            cfg.sensor_requested_priority_res = _sensor_requested_priority_res;
            cfg.sensor_restore_rgb = _sensor_restore_rgb;
            cfg.sensor_restore_weight = _sensor_restore_weight;
            cfg.sensor_dirty_sites = _sensor_dirty_sites;
            cfg.sensor_restore_res = _sensor_restore_res;
            cfg.shader_dir                  = _shader_dir;
            cfg.gl_display_hglrc        = _gl_display_hglrc;
            cfg.gl_display_hdc          = _gl_display_hdc;
            cfg.t5_min_geom             = _t5_min_geom;
            cfg.t5_light_batch_size     = _t5_light_batch_size;
            cfg.t5_cam_batch_size       = _t5_cam_batch_size;
            cfg.t5_sensor_tile_size     = _t5_sensor_tile_size;
            cfg.t5_pair_budget          = _t5_pair_budget;
            cfg.t5_backlog_share_base    = _t5_backlog_share_base;
            cfg.t5_backlog_share_max     = _t5_backlog_share_max;
            cfg.t5_backlog_max_snapshots = _t5_backlog_max_snapshots;
            cfg.t5_backlog_max_bytes     = _t5_backlog_max_bytes;
            cfg.bdpt_record_float_budget = _bdpt_record_float_budget;
            cfg.t5_profile              = _t5_profile;
            cfg.vcm_enabled             = _vcm_enabled;
            cfg.vcm_merge_radius_m      = _vcm_merge_radius_m;
            cfg.vcm_radius_alpha        = _vcm_radius_alpha;
            cfg.flash_modifier_type     = static_cast<FlashModifierType>(_flash_modifier_type);
            cfg.flash_modifier_param0   = _flash_modifier_param0;
            cfg.flash_modifier_param1   = _flash_modifier_param1;
            _pipeline = ray_pipeline_create(handle, &cfg);
            if (!_pipeline)
                throw std::runtime_error("ray_pipeline_create failed");
            /* Apply sensor image config that may have been set before pipeline existed. */
            if (_sensor_res > 0)
                ray_pipeline_configure_sensor_image(
                    _pipeline, _sensor_plate_x, _sensor_plate_half_w, _sensor_plate_half_h,
                    _sensor_res, _sensor_bdpt_eps,
                    _sensor_target_x, _sensor_target_r,
                    _sensor_target_y, _sensor_target_z, _sensor_target_mode);
            if (_sensor_pose_configured)
                ray_pipeline_configure_sensor_pose(
                    _pipeline, _sensor_center.data(), _sensor_right.data(), _sensor_up.data(),
                    _aperture_center.data(), _aperture_right.data(), _aperture_up.data());
            if (_uv_blit_n_bands > 0)
                ray_pipeline_set_uv_blit_weights(
                    _pipeline, _uv_blit_weights.data(),
                    _uv_blit_n_bands, _uv_blit_mode);

        }
        return _pipeline;
    }

    PyRayTracer(int                     n_tri,
                py::array_t<double>     verts,
                py::array_t<double>     normals,
                py::array_t<int>        mat_idx,
                py::array_t<float>      mat_buf,
                int                     mat_n_mats,
                py::array_t<double>     freq_hz,
                double                  speed_m_s,
                py::array_t<double>     atmo_abs)
    {
        auto iv  = verts    .request();
        auto in_ = normals  .request();
        auto imi = mat_idx  .request();
        auto imb = mat_buf  .request();
        auto ifh = freq_hz  .request();
        auto iaa = atmo_abs .request();

        _n_bands = static_cast<int>(ifh.size);
        _n_tris  = n_tri;
        _freq_hz.assign(static_cast<const double*>(ifh.ptr),
                static_cast<const double*>(ifh.ptr) + _n_bands);

        handle = ray_tracer_create(
            n_tri,
            static_cast<const double*>(iv .ptr),
            static_cast<const double*>(in_.ptr),
            static_cast<const int*>   (imi.ptr),
            static_cast<const float*> (imb.ptr),
            mat_n_mats,
            _n_bands,
            static_cast<const double*>(ifh.ptr),
            speed_m_s,
            static_cast<const double*>(iaa.ptr));

        if (!handle)
            throw std::runtime_error("ray_tracer_create: allocation failed");
    }

    ~PyRayTracer() {
        if (_pipeline) { ray_pipeline_destroy(_pipeline); _pipeline = nullptr; }
        ray_tracer_destroy(handle); handle = nullptr;
    }

    py::dict build_frequency_sidecar(
        py::array_t<float> segs_arr,
        int band_col = 8) const
    {
        auto s = segs_arr.request();
        if (s.ndim != 2)
            throw std::invalid_argument("segs must be shape (N, M)");
        const int n_rows = static_cast<int>(s.shape[0]);
        const int n_cols = static_cast<int>(s.shape[1]);
        if (band_col < 0 || band_col >= n_cols)
            throw std::invalid_argument("band_col is out of range for segs columns");

        py::array_t<int32_t> band_id({n_rows});
        py::array_t<double>  frequency_hz({n_rows});
        py::array_t<double>  wavelength_nm({n_rows});

        const float* src = static_cast<const float*>(s.ptr);
        int32_t* out_band = band_id.mutable_data();
        double* out_f = frequency_hz.mutable_data();
        double* out_wl = wavelength_nm.mutable_data();

        for (int i = 0; i < n_rows; ++i) {
            const float raw_b = src[static_cast<size_t>(i) * static_cast<size_t>(n_cols) + static_cast<size_t>(band_col)];
            int b = static_cast<int>(std::llround(static_cast<double>(raw_b)));
            if (b < 0 || b >= _n_bands) b = -1;
            out_band[i] = static_cast<int32_t>(b);
            if (b >= 0) {
                const double f = _freq_hz[static_cast<size_t>(b)];
                out_f[i] = f;
                out_wl[i] = (f > 0.0) ? (299792458.0 / f) * 1.0e9 : 0.0;
            } else {
                out_f[i] = 0.0;
                out_wl[i] = 0.0;
            }
        }

        py::dict out;
        out["band_id"] = band_id;
        out["frequency_hz"] = frequency_hz;
        out["wavelength_nm"] = wavelength_nm;
        return out;
    }

    py::dict spectral_bands_to_rgb(
        py::array_t<float, py::array::c_style | py::array::forcecast> image_bhw,
        double gain = 1.0,
        double hdr_white_percentile = 99.8) const
    {
        auto ib = image_bhw.request();
        if (ib.ndim != 3)
            throw std::invalid_argument("image_bhw must be shape (n_bands, H, W)");
        const int n_bands = static_cast<int>(ib.shape[0]);
        const int H = static_cast<int>(ib.shape[1]);
        const int W = static_cast<int>(ib.shape[2]);
        if (n_bands != _n_bands)
            throw std::invalid_argument("image_bhw first dimension must match tracer n_bands");

        py::array::ShapeContainer shape = {
            static_cast<py::ssize_t>(H),
            static_cast<py::ssize_t>(W),
            static_cast<py::ssize_t>(3)
        };
        py::array_t<float> rgb_linear(shape);
        py::array_t<float> rgb_tonemapped(shape);

        const float* src = static_cast<const float*>(ib.ptr);
        const size_t pix_count = static_cast<size_t>(H) * static_cast<size_t>(W);
        const double gain_sq = gain * gain;

        auto wavelength_nm = [](double freq_hz) -> double {
            return (freq_hz > 0.0) ? (299792458.0 / freq_hz) * 1.0e9 : 0.0;
        };
        auto wavelength_to_rgb = [](double wl, double& r, double& g, double& b) {
            r = 0.0;
            g = 0.0;
            b = 0.0;
            if (wl >= 380.0 && wl < 440.0) {
                r = -(wl - 440.0) / (440.0 - 380.0);
                b = 1.0;
            } else if (wl < 490.0) {
                g = (wl - 440.0) / (490.0 - 440.0);
                b = 1.0;
            } else if (wl < 510.0) {
                g = 1.0;
                b = -(wl - 510.0) / (510.0 - 490.0);
            } else if (wl < 580.0) {
                r = (wl - 510.0) / (580.0 - 510.0);
                g = 1.0;
            } else if (wl < 645.0) {
                r = 1.0;
                g = -(wl - 645.0) / (645.0 - 580.0);
            } else if (wl <= 700.0) {
                r = 1.0;
            }
            double edge = 1.0;
            if (wl >= 380.0 && wl < 420.0)
                edge = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0);
            else if (wl > 645.0 && wl <= 700.0)
                edge = 0.3 + 0.7 * (700.0 - wl) / (700.0 - 645.0);
            r *= edge;
            g *= edge;
            b *= edge;
        };

        std::vector<double> wr(static_cast<size_t>(_n_bands), 0.0);
        std::vector<double> wg(static_cast<size_t>(_n_bands), 0.0);
        std::vector<double> wb(static_cast<size_t>(_n_bands), 0.0);
        for (int b = 0; b < _n_bands; ++b) {
            const double wl = wavelength_nm(_freq_hz[static_cast<size_t>(b)]);
            wavelength_to_rgb(
                std::min(700.0, std::max(380.0, wl)),
                wr[static_cast<size_t>(b)],
                wg[static_cast<size_t>(b)],
                wb[static_cast<size_t>(b)]);
        }

        float* out_lin = rgb_linear.mutable_data();
        double white = 0.0;
        for (size_t pix = 0; pix < pix_count; ++pix) {
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            for (int band = 0; band < _n_bands; ++band) {
                const size_t idx = static_cast<size_t>(band) * pix_count + pix;
                const double p = std::max(0.0, static_cast<double>(src[idx])) * gain_sq;
                r += p * wr[static_cast<size_t>(band)];
                g += p * wg[static_cast<size_t>(band)];
                b += p * wb[static_cast<size_t>(band)];
            }
            const size_t base = pix * 3u;
            out_lin[base + 0] = static_cast<float>(r);
            out_lin[base + 1] = static_cast<float>(g);
            out_lin[base + 2] = static_cast<float>(b);
            white = std::max(white, std::max(r, std::max(g, b)));
        }

        // Robust highlight control: derive a global white point from luminance
        // percentile, then apply Reinhard in normalized space.  This prevents
        // large hot regions from washing out the entire frame.
        std::vector<double> lum;
        lum.resize(pix_count);
        double min_positive_lum = std::numeric_limits<double>::infinity();
        double max_lum = 0.0;
        for (size_t pix = 0; pix < pix_count; ++pix) {
            const size_t base = pix * 3u;
            const double r = std::max(0.0, static_cast<double>(out_lin[base + 0]));
            const double g = std::max(0.0, static_cast<double>(out_lin[base + 1]));
            const double b = std::max(0.0, static_cast<double>(out_lin[base + 2]));
            lum[pix] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            if (lum[pix] > 0.0) {
                min_positive_lum = std::min(min_positive_lum, lum[pix]);
                max_lum = std::max(max_lum, lum[pix]);
            }
        }
        const double pct = std::min(100.0, std::max(0.0, hdr_white_percentile));
        const size_t rank = static_cast<size_t>(std::llround((pct * 0.01) * static_cast<double>(pix_count - 1)));
        std::nth_element(lum.begin(), lum.begin() + rank, lum.end());
        double white_scale = std::max(lum[rank], max_lum);
        if (!(white_scale > 0.0)) white_scale = 1.0;
        if (!std::isfinite(min_positive_lum)) min_positive_lum = white_scale;
        const double black_scale = std::max(1.0e-30, min_positive_lum);
        white_scale = std::max(white_scale, black_scale * 1.000001);

        const float* lin = rgb_linear.data();
        float* out_tm = rgb_tonemapped.mutable_data();
        const double log_denom = std::max(1.0e-12, std::log1p(white_scale / black_scale));
        for (size_t pix = 0; pix < pix_count; ++pix) {
            const size_t base = pix * 3u;
            const double lum_pix = 0.2126 * std::max(0.0, static_cast<double>(lin[base + 0]))
                                 + 0.7152 * std::max(0.0, static_cast<double>(lin[base + 1]))
                                 + 0.0722 * std::max(0.0, static_cast<double>(lin[base + 2]));
            if (!(lum_pix > 0.0)) {
                out_tm[base + 0] = 0.0f;
                out_tm[base + 1] = 0.0f;
                out_tm[base + 2] = 0.0f;
                continue;
            }
            const double y_raw = std::log1p(lum_pix / black_scale) / log_denom;
            const double y = std::max(0.04, y_raw);
            const double scale = std::min(1.0, std::max(0.0, y)) / std::max(lum_pix, 1.0e-30);
            double rgb_tmp[3] = {0.0, 0.0, 0.0};
            for (int c = 0; c < 3; ++c) {
                const double x = std::max(0.0, static_cast<double>(lin[base + static_cast<size_t>(c)]));
                double v = std::min(1.0, x * scale);
                /* Mild post-curve regularization: rounded highlight shoulder
                 * and slight saturation damping after the nonzero-visible map.
                * This keeps tiny signals visible without letting hot bands
                 * turn the preview into hard clipped neon blocks. */
                v = v / (1.0 + 0.18 * v);
                rgb_tmp[c] = std::min(1.0, std::max(0.0, v));
            }
            const double luma = 0.2126 * rgb_tmp[0] + 0.7152 * rgb_tmp[1] + 0.0722 * rgb_tmp[2];
            for (int c = 0; c < 3; ++c) {
                const double v = 0.90 * rgb_tmp[c] + 0.10 * luma;
                out_tm[base + static_cast<size_t>(c)] = static_cast<float>(std::min(1.0, std::max(0.0, v)));
            }
        }

        py::dict out;
        out["rgb_linear"] = rgb_linear;
        out["rgb_tonemapped"] = rgb_tonemapped;
        return out;
    }

    py::dict rasterize_tri_flux_uv(
        py::array_t<float, py::array::c_style | py::array::forcecast> tri_flux,
        py::array_t<int, py::array::c_style | py::array::forcecast> tri_indices,
        py::array_t<float, py::array::c_style | py::array::forcecast> tri_uv,
        int atlas_h,
        int atlas_w,
        double gain = 1.0,
        double hdr_white_percentile = 100.0) const
    {
        auto tf = tri_flux.request();
        auto ti = tri_indices.request();
        auto tu = tri_uv.request();
        if (tf.ndim != 2 || static_cast<int>(tf.shape[0]) != _n_tris || static_cast<int>(tf.shape[1]) != _n_bands)
            throw std::invalid_argument("tri_flux must be shape (n_tri, n_bands)");
        if (ti.ndim != 1)
            throw std::invalid_argument("tri_indices must be shape (n_selected,)");
        if (tu.ndim != 3 || tu.shape[1] != 3 || tu.shape[2] != 2 || tu.shape[0] != ti.shape[0])
            throw std::invalid_argument("tri_uv must be shape (n_selected, 3, 2)");
        if (atlas_h <= 0 || atlas_w <= 0)
            throw std::invalid_argument("atlas_h and atlas_w must be positive");

        py::array_t<float> atlas({
            static_cast<py::ssize_t>(_n_bands),
            static_cast<py::ssize_t>(atlas_h),
            static_cast<py::ssize_t>(atlas_w)
        });
        float* out = atlas.mutable_data();
        std::fill(out, out + atlas.size(), 0.0f);

        const float* flux = static_cast<const float*>(tf.ptr);
        const int* indices = static_cast<const int*>(ti.ptr);
        const float* uv = static_cast<const float*>(tu.ptr);
        const int n_sel = static_cast<int>(ti.shape[0]);
        const size_t pix_count = static_cast<size_t>(atlas_h) * static_cast<size_t>(atlas_w);

        auto edge = [](double ax, double ay, double bx, double by, double px, double py) -> double {
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
        };

        for (int i = 0; i < n_sel; ++i) {
            const int tri = indices[i];
            if (tri < 0 || tri >= _n_tris) continue;
            const size_t ubase = static_cast<size_t>(i) * 6u;
            const double u0 = std::min(1.0, std::max(0.0, static_cast<double>(uv[ubase + 0u])));
            const double v0 = std::min(1.0, std::max(0.0, static_cast<double>(uv[ubase + 1u])));
            const double u1 = std::min(1.0, std::max(0.0, static_cast<double>(uv[ubase + 2u])));
            const double v1 = std::min(1.0, std::max(0.0, static_cast<double>(uv[ubase + 3u])));
            const double u2 = std::min(1.0, std::max(0.0, static_cast<double>(uv[ubase + 4u])));
            const double v2 = std::min(1.0, std::max(0.0, static_cast<double>(uv[ubase + 5u])));

            const double x0 = u0 * static_cast<double>(atlas_w - 1);
            const double y0 = (1.0 - v0) * static_cast<double>(atlas_h - 1);
            const double x1 = u1 * static_cast<double>(atlas_w - 1);
            const double y1 = (1.0 - v1) * static_cast<double>(atlas_h - 1);
            const double x2 = u2 * static_cast<double>(atlas_w - 1);
            const double y2 = (1.0 - v2) * static_cast<double>(atlas_h - 1);

            const double area2 = edge(x0, y0, x1, y1, x2, y2);
            if (std::abs(area2) < 1.0e-9) continue;
            const int xmin = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
            const int xmax = std::min(atlas_w - 1, static_cast<int>(std::ceil (std::max({x0, x1, x2}))));
            const int ymin = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
            const int ymax = std::min(atlas_h - 1, static_cast<int>(std::ceil (std::max({y0, y1, y2}))));
            const double inv_pixels = 1.0 / std::max(1.0, 0.5 * std::abs(area2));

            for (int y = ymin; y <= ymax; ++y) {
                const double py = static_cast<double>(y) + 0.5;
                for (int x = xmin; x <= xmax; ++x) {
                    const double px = static_cast<double>(x) + 0.5;
                    const double w0 = edge(x1, y1, x2, y2, px, py);
                    const double w1 = edge(x2, y2, x0, y0, px, py);
                    const double w2 = edge(x0, y0, x1, y1, px, py);
                    if ((area2 > 0.0 && (w0 < 0.0 || w1 < 0.0 || w2 < 0.0)) ||
                        (area2 < 0.0 && (w0 > 0.0 || w1 > 0.0 || w2 > 0.0))) {
                        continue;
                    }
                    const size_t pix = static_cast<size_t>(y) * static_cast<size_t>(atlas_w) + static_cast<size_t>(x);
                    const size_t fbase = static_cast<size_t>(tri) * static_cast<size_t>(_n_bands);
                    for (int b = 0; b < _n_bands; ++b) {
                        out[static_cast<size_t>(b) * pix_count + pix] +=
                            static_cast<float>(static_cast<double>(flux[fbase + static_cast<size_t>(b)]) * inv_pixels);
                    }
                }
            }
        }

        py::dict rgb = spectral_bands_to_rgb(atlas, gain, hdr_white_percentile);
        rgb["spectral_atlas"] = atlas;
        return rgb;
    }

    py::tuple compose_lens_bench_views(
        py::array_t<float, py::array::c_style | py::array::forcecast> tri_flux,
        py::array_t<double, py::array::c_style | py::array::forcecast> tri_centroids,
        int view_h,
        int view_w,
        double x_min,
        double x_max,
        double view_radius,
        int field_nx,
        int field_ny,
        int field_nz,
        double field_gain = 2.5,
        double surface_gain = 1.0,
        double hdr_white_percentile = 99.8) const
    {
        auto tf = tri_flux.request();
        auto tc = tri_centroids.request();
        if (tf.ndim != 2)
            throw std::invalid_argument("tri_flux must be shape (n_tri, n_bands)");
        if (tc.ndim != 2 || tc.shape[1] != 3)
            throw std::invalid_argument("tri_centroids must be shape (n_tri, 3)");
        if (tf.shape[0] != tc.shape[0])
            throw std::invalid_argument("tri_flux and tri_centroids must have matching n_tri");
        if (static_cast<int>(tf.shape[1]) != _n_bands)
            throw std::invalid_argument("tri_flux second dimension must match tracer n_bands");
        if (view_h <= 0 || view_w <= 0)
            throw std::invalid_argument("view_h and view_w must be positive");
        if (field_nx <= 0 || field_ny <= 0 || field_nz <= 0)
            throw std::invalid_argument("field grid dimensions must be positive");

        const int n_tri = static_cast<int>(tf.shape[0]);
        const float* tri_flux_ptr = static_cast<const float*>(tf.ptr);
        const double* tri_centroids_ptr = static_cast<const double*>(tc.ptr);

        py::array_t<float> surf_top_bhw({
            static_cast<py::ssize_t>(_n_bands),
            static_cast<py::ssize_t>(view_h),
            static_cast<py::ssize_t>(view_w)
        });
        py::array_t<float> surf_side_bhw({
            static_cast<py::ssize_t>(_n_bands),
            static_cast<py::ssize_t>(view_h),
            static_cast<py::ssize_t>(view_w)
        });
        float* surf_top_ptr = surf_top_bhw.mutable_data();
        float* surf_side_ptr = surf_side_bhw.mutable_data();
        const size_t surf_count = static_cast<size_t>(_n_bands) * static_cast<size_t>(view_h) * static_cast<size_t>(view_w);
        std::fill(surf_top_ptr, surf_top_ptr + surf_count, 0.0f);
        std::fill(surf_side_ptr, surf_side_ptr + surf_count, 0.0f);

        const double x_span = std::max(1.0e-12, x_max - x_min);
        const double v_span = std::max(1.0e-12, 2.0 * view_radius);

        std::vector<int> ix(static_cast<size_t>(n_tri));
        std::vector<int> iy_top(static_cast<size_t>(n_tri));
        std::vector<int> iy_side(static_cast<size_t>(n_tri));
        for (int i = 0; i < n_tri; ++i) {
            const size_t base = static_cast<size_t>(i) * 3u;
            const double x = tri_centroids_ptr[base + 0u];
            const double y = tri_centroids_ptr[base + 1u];
            const double z = tri_centroids_ptr[base + 2u];

            int px = static_cast<int>(std::llround(((x - x_min) / x_span) * static_cast<double>(view_w - 1)));
            int py_top = static_cast<int>(std::llround((0.5 - (y / v_span)) * static_cast<double>(view_h - 1)));
            int py_side = static_cast<int>(std::llround((0.5 - (z / v_span)) * static_cast<double>(view_h - 1)));

            px = std::max(0, std::min(view_w - 1, px));
            py_top = std::max(0, std::min(view_h - 1, py_top));
            py_side = std::max(0, std::min(view_h - 1, py_side));

            ix[static_cast<size_t>(i)] = px;
            iy_top[static_cast<size_t>(i)] = py_top;
            iy_side[static_cast<size_t>(i)] = py_side;
        }

        for (int b = 0; b < _n_bands; ++b) {
            const size_t b_off_surf = static_cast<size_t>(b) * static_cast<size_t>(view_h) * static_cast<size_t>(view_w);
            for (int i = 0; i < n_tri; ++i) {
                const float v = tri_flux_ptr[static_cast<size_t>(i) * static_cast<size_t>(_n_bands) + static_cast<size_t>(b)];
                const size_t idx_top = b_off_surf + static_cast<size_t>(iy_top[static_cast<size_t>(i)]) * static_cast<size_t>(view_w) + static_cast<size_t>(ix[static_cast<size_t>(i)]);
                const size_t idx_side = b_off_surf + static_cast<size_t>(iy_side[static_cast<size_t>(i)]) * static_cast<size_t>(view_w) + static_cast<size_t>(ix[static_cast<size_t>(i)]);
                surf_top_ptr[idx_top] += v;
                surf_side_ptr[idx_side] += v;
            }
        }

        py::array_t<float> grid_reim = get_field_capture_grid_reim();
        auto gr = grid_reim.request();
        if (gr.ndim != 3 || static_cast<int>(gr.shape[0]) != _n_bands || static_cast<int>(gr.shape[2]) != 2)
            throw std::runtime_error("field capture grid has unexpected shape");
        const size_t n_vox = static_cast<size_t>(field_nx) * static_cast<size_t>(field_ny) * static_cast<size_t>(field_nz);
        if (static_cast<size_t>(gr.shape[1]) != n_vox)
            throw std::runtime_error("field capture grid size does not match provided field_nx/ny/nz");
        const float* grid_ptr = static_cast<const float*>(gr.ptr);

        py::array_t<float> field_top_bhw({
            static_cast<py::ssize_t>(_n_bands),
            static_cast<py::ssize_t>(view_h),
            static_cast<py::ssize_t>(view_w)
        });
        py::array_t<float> field_side_bhw({
            static_cast<py::ssize_t>(_n_bands),
            static_cast<py::ssize_t>(view_h),
            static_cast<py::ssize_t>(view_w)
        });
        float* field_top_ptr = field_top_bhw.mutable_data();
        float* field_side_ptr = field_side_bhw.mutable_data();
        std::fill(field_top_ptr, field_top_ptr + surf_count, 0.0f);
        std::fill(field_side_ptr, field_side_ptr + surf_count, 0.0f);

        std::vector<int> gx(static_cast<size_t>(view_w), 0);
        std::vector<int> gy(static_cast<size_t>(view_h), 0);
        std::vector<int> gz(static_cast<size_t>(view_h), 0);
        for (int x = 0; x < view_w; ++x) {
            gx[static_cast<size_t>(x)] = (view_w > 1)
                ? static_cast<int>(std::llround((static_cast<double>(x) * static_cast<double>(field_nx - 1)) / static_cast<double>(view_w - 1)))
                : 0;
        }
        for (int y = 0; y < view_h; ++y) {
            gy[static_cast<size_t>(y)] = (view_h > 1)
                ? static_cast<int>(std::llround((static_cast<double>(y) * static_cast<double>(field_ny - 1)) / static_cast<double>(view_h - 1)))
                : 0;
            gz[static_cast<size_t>(y)] = (view_h > 1)
                ? static_cast<int>(std::llround((static_cast<double>(y) * static_cast<double>(field_nz - 1)) / static_cast<double>(view_h - 1)))
                : 0;
        }

        std::vector<float> top_raw(static_cast<size_t>(field_ny) * static_cast<size_t>(field_nx), 0.0f);
        std::vector<float> side_raw(static_cast<size_t>(field_nz) * static_cast<size_t>(field_nx), 0.0f);

        for (int b = 0; b < _n_bands; ++b) {
            std::fill(top_raw.begin(), top_raw.end(), 0.0f);
            std::fill(side_raw.begin(), side_raw.end(), 0.0f);

            const size_t b_off_vox = static_cast<size_t>(b) * n_vox;
            for (int x = 0; x < field_nx; ++x) {
                for (int y = 0; y < field_ny; ++y) {
                    float sum_top = 0.0f;
                    for (int z = 0; z < field_nz; ++z) {
                        const size_t v = (static_cast<size_t>(z) * static_cast<size_t>(field_ny) + static_cast<size_t>(y)) * static_cast<size_t>(field_nx) + static_cast<size_t>(x);
                        const size_t reim_base = (b_off_vox + v) * 2u;
                        const float re = grid_ptr[reim_base + 0u];
                        const float im = grid_ptr[reim_base + 1u];
                        const float amp = std::sqrt(std::max(0.0f, re * re + im * im));
                        sum_top += amp;
                        side_raw[static_cast<size_t>(z) * static_cast<size_t>(field_nx) + static_cast<size_t>(x)] += amp;
                    }
                    top_raw[static_cast<size_t>(y) * static_cast<size_t>(field_nx) + static_cast<size_t>(x)] = sum_top;
                }
            }

            const size_t b_off_img = static_cast<size_t>(b) * static_cast<size_t>(view_h) * static_cast<size_t>(view_w);
            for (int y = 0; y < view_h; ++y) {
                const int sy = std::max(0, std::min(field_ny - 1, gy[static_cast<size_t>(y)]));
                const int sz = std::max(0, std::min(field_nz - 1, gz[static_cast<size_t>(y)]));
                for (int x = 0; x < view_w; ++x) {
                    const int sx = std::max(0, std::min(field_nx - 1, gx[static_cast<size_t>(x)]));
                    const size_t idx = b_off_img + static_cast<size_t>(y) * static_cast<size_t>(view_w) + static_cast<size_t>(x);
                    field_top_ptr[idx] = top_raw[static_cast<size_t>(sy) * static_cast<size_t>(field_nx) + static_cast<size_t>(sx)];
                    field_side_ptr[idx] = side_raw[static_cast<size_t>(sz) * static_cast<size_t>(field_nx) + static_cast<size_t>(sx)];
                }
            }
        }

        // surface_gain / field_gain are forwarded as pre-tonemap gain (gain_sq = gain*gain
        // inside spectral_bands_to_rgb).  This lifts dim transmitted signals before Reinhard
        // so they are proportionally visible rather than crushed by a global white point.
        py::dict surf_top_rgb_d  = spectral_bands_to_rgb(surf_top_bhw,  surface_gain, hdr_white_percentile);
        py::dict surf_side_rgb_d = spectral_bands_to_rgb(surf_side_bhw, surface_gain, hdr_white_percentile);
        py::dict field_top_rgb_d  = spectral_bands_to_rgb(field_top_bhw,  field_gain, hdr_white_percentile);
        py::dict field_side_rgb_d = spectral_bands_to_rgb(field_side_bhw, field_gain, hdr_white_percentile);

        py::array_t<float> surf_top_rgb = surf_top_rgb_d["rgb_tonemapped"].cast<py::array_t<float>>();
        py::array_t<float> surf_side_rgb = surf_side_rgb_d["rgb_tonemapped"].cast<py::array_t<float>>();
        py::array_t<float> field_top_rgb = field_top_rgb_d["rgb_tonemapped"].cast<py::array_t<float>>();
        py::array_t<float> field_side_rgb = field_side_rgb_d["rgb_tonemapped"].cast<py::array_t<float>>();

        py::array_t<float> top_rgb({
            static_cast<py::ssize_t>(view_h),
            static_cast<py::ssize_t>(view_w),
            static_cast<py::ssize_t>(3)
        });
        py::array_t<float> side_rgb({
            static_cast<py::ssize_t>(view_h),
            static_cast<py::ssize_t>(view_w),
            static_cast<py::ssize_t>(3)
        });

        const float* surf_top_rgb_ptr = surf_top_rgb.data();
        const float* surf_side_rgb_ptr = surf_side_rgb.data();
        const float* field_top_rgb_ptr = field_top_rgb.data();
        const float* field_side_rgb_ptr = field_side_rgb.data();
        float* top_rgb_ptr = top_rgb.mutable_data();
        float* side_rgb_ptr = side_rgb.mutable_data();
        const size_t rgb_count = static_cast<size_t>(view_h) * static_cast<size_t>(view_w) * 3u;

        // Gains are already baked into each tonemapped buffer; just add and clamp.
        for (size_t i = 0; i < rgb_count; ++i) {
            top_rgb_ptr[i]  = std::min(1.0f, surf_top_rgb_ptr[i]  + field_top_rgb_ptr[i]);
            side_rgb_ptr[i] = std::min(1.0f, surf_side_rgb_ptr[i] + field_side_rgb_ptr[i]);
        }

        return py::make_tuple(top_rgb, side_rgb);
    }

    py::str allocation_table() const
    {
        const char* t = ray_tracer_allocation_table(handle);
        return py::str(t ? t : "");
    }

    void set_profile_pulse(bool enabled = true, double period_s = 2.0)
    {
        int rc = ray_tracer_set_profile_pulse(
            handle,
            enabled ? 1 : 0,
            period_s);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_set_profile_pulse failed: rc=" + std::to_string(rc));
    }

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

    py::dict trace_with_frequency_sidecar(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        int     n_rays        = 256,
        int     max_bounces   = 8,
        double  min_amplitude = 0.005,
        uint32_t seed         = 42,
        int     out_cap       = -1)
    {
        py::array_t<float> segs = trace(
            src_pos,
            src_dir,
            src_directivity,
            n_rays,
            max_bounces,
            min_amplitude,
            seed,
            out_cap);
        py::dict out;
        out["segs"] = segs;
        out["freq_sidecar"] = build_frequency_sidecar(segs, 8);
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

    py::dict trace_surface_with_frequency_sidecar(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        int      n_rays        = 256,
        int      max_bounces   = 8,
        double   min_amplitude = 0.005,
        uint32_t seed          = 42,
        int      out_cap       = -1)
    {
        py::dict result = trace_surface(
            src_pos,
            src_dir,
            src_directivity,
            n_rays,
            max_bounces,
            min_amplitude,
            seed,
            out_cap);
        py::array_t<float> segs = result["segs"].cast<py::array_t<float>>();
        result["freq_sidecar"] = build_frequency_sidecar(segs, 8);
        return result;
    }

    /**
     * trace_pipeline(origins, directions, amplitudes,
     *                max_bounces, min_amplitude, seed, max_children)
     *   -> dict with key:
     *       'segs' : float32 (N_segs, 12)
     *
     * Runs the 4-stage threaded ray pipeline (T1 intersector, T2 refiner,
     * T3 material, T4 wave solver).  origins and directions are float64 (N,3).
     * amplitudes is complex128 (N, n_bands); if omitted, unit amplitude is used.
     * src_ids is int32 (N,) tagging each ray's source; if omitted, 0 for all.
     * tags is uint64 (N,) free slot carried through all stages; 0 if omitted.
     *
     * Segment layout is identical to trace_surface: 12 floats per segment.
     *   [0..2]  seg_start   [3..5]  hit_pos   [6] src_id   [7] bounce
     *   [8] freq_band   [9] |amplitude|   [10] phase   [11] path_at_seg_start
     */
    py::dict trace_pipeline(
        py::array_t<double>  origins,
        py::array_t<double>  directions,
        py::object           amplitudes  = py::none(),
        py::object           src_ids     = py::none(),
        py::object           tags        = py::none(),
        int                  max_bounces    = 8,
        double               min_amplitude  = 1e-6,
        int                  seed           = 42,
        int                  max_children   = 2,
        int                  out_cap        = 4'000'000)
    {
        auto io = origins   .request();
        auto id = directions.request();
        if (io.ndim != 2 || io.shape[1] != 3)
            throw std::invalid_argument("origins must be (N,3)");
        if (id.ndim != 2 || id.shape[1] != 3)
            throw std::invalid_argument("directions must be (N,3)");
        const int n_rays = static_cast<int>(io.shape[0]);
        if (id.shape[0] != n_rays)
            throw std::invalid_argument("origins and directions must have the same length");

        /* Optional amplitude array (N, n_bands) complex128 */
        py::array_t<std::complex<double>> amp_arr;
        bool has_amp = !amplitudes.is_none();
        if (has_amp) {
            amp_arr = amplitudes.cast<py::array_t<std::complex<double>>>();
            auto ia = amp_arr.request();
            if (ia.ndim != 2 || ia.shape[0] != n_rays)
                throw std::invalid_argument("amplitudes must be (N, n_bands) complex128");
        }

        /* Optional src_ids (N,) int32 */
        py::array_t<int32_t> sid_arr;
        bool has_sid = !src_ids.is_none();
        if (has_sid) {
            sid_arr = src_ids.cast<py::array_t<int32_t>>();
            if (sid_arr.request().size != n_rays)
                throw std::invalid_argument("src_ids must be length N");
        }

        /* Optional tags (N,) uint64 */
        py::array_t<uint64_t> tag_arr;
        bool has_tag = !tags.is_none();
        if (has_tag) {
            tag_arr = tags.cast<py::array_t<uint64_t>>();
            if (tag_arr.request().size != n_rays)
                throw std::invalid_argument("tags must be length N");
        }

        /* Build RayIntents */
        const double* op  = static_cast<const double*>(io.ptr);
        const double* dp  = static_cast<const double*>(id.ptr);
        const std::complex<double>* ap = has_amp
            ? static_cast<const std::complex<double>*>(amp_arr.request().ptr)
            : nullptr;
        const int32_t*  sp = has_sid
            ? static_cast<const int32_t*>(sid_arr.request().ptr) : nullptr;
        const uint64_t* tp = has_tag
            ? static_cast<const uint64_t*>(tag_arr.request().ptr) : nullptr;
        const int amp_bands = has_amp
            ? static_cast<int>(amp_arr.request().shape[1]) : _n_bands;

        std::vector<RayIntent> intents(n_rays);
        for (int i = 0; i < n_rays; ++i) {
            RayIntent& ri    = intents[i];
            ri.pos           = Eigen::Vector3d(op[i*3+0], op[i*3+1], op[i*3+2]);
            ri.dir           = Eigen::Vector3d(dp[i*3+0], dp[i*3+1], dp[i*3+2]).normalized();
            ri.amp.resize(amp_bands);
            if (ap) {
                for (int b = 0; b < amp_bands; ++b)
                    ri.amp[b] = ap[i*amp_bands + b];
            } else {
                ri.amp.setOnes();
            }
            ri.src_id        = sp ? sp[i] : i;
            ri.tag           = tp ? tp[i] : 0u;
            ri.bounces_left  = max_bounces;
            ri.min_amplitude = min_amplitude;
        }

        /* Per-triangle irradiance buffers (|amp|² per hit, direct vs indirect). */
        const int n_tris = _n_tris;
        const int n_bands_buf = amp_bands;
        py::array_t<float> direct(
            {static_cast<py::ssize_t>(n_tris), static_cast<py::ssize_t>(n_bands_buf)});
        py::array_t<float> indirect(
            {static_cast<py::ssize_t>(n_tris), static_cast<py::ssize_t>(n_bands_buf)});
        std::fill(direct  .mutable_data(), direct  .mutable_data() + direct  .size(), 0.0f);
        std::fill(indirect.mutable_data(), indirect.mutable_data() + indirect.size(), 0.0f);
        float* direct_p   = direct  .mutable_data();
        float* indirect_p = indirect.mutable_data();

        std::vector<float> seg_buf;
        seg_buf.reserve(static_cast<size_t>(std::min(n_rays * 4, out_cap))
                        * RT_FLOATS_PER_SEG);

        /* Run synchronous trace via persistent-machine API.
         * GIL released for the whole trace. */
        std::vector<RayRecord> records;
        records.reserve(static_cast<size_t>(std::min(n_rays * 8, out_cap)));
        {
            py::gil_scoped_release release;
            RayPipelineConfig cfg;
            cfg.max_children = max_children;
            cfg.seed         = seed;
            ray_pipeline_trace_sync(handle, &cfg, intents.data(), n_rays, records);
        }

        /* Pack RayRecord stream into irradiance buffers and seg_buf. */
        for (const auto& rec : records) {
            if (rec.kind != RayRecordKind::STRIKE) continue;
            const int tri = rec.hit_tri;
            if (tri >= 0 && tri < n_tris) {
                float* flux = (rec.bounce == 0 ? direct_p : indirect_p)
                              + static_cast<ptrdiff_t>(tri) * n_bands_buf;
                for (int b = 0; b < rec.n_bands && b < n_bands_buf; ++b)
                    flux[b] += rec.amp_re[b]*rec.amp_re[b] + rec.amp_im[b]*rec.amp_im[b];
            }
            if (static_cast<int>(seg_buf.size() / RT_FLOATS_PER_SEG) >= out_cap) continue;
            for (int b = 0; b < rec.n_bands; ++b) {
                float amp_abs = std::hypot(rec.amp_re[b], rec.amp_im[b]);
                float amp_arg = std::atan2(rec.amp_im[b], rec.amp_re[b]);
                seg_buf.push_back(rec.seg_start[0]);
                seg_buf.push_back(rec.seg_start[1]);
                seg_buf.push_back(rec.seg_start[2]);
                seg_buf.push_back(rec.pos[0]);
                seg_buf.push_back(rec.pos[1]);
                seg_buf.push_back(rec.pos[2]);
                seg_buf.push_back(static_cast<float>(rec.src_id));
                seg_buf.push_back(static_cast<float>(rec.bounce));
                seg_buf.push_back(static_cast<float>(b));
                seg_buf.push_back(amp_abs);
                seg_buf.push_back(amp_arg);
                seg_buf.push_back(rec.path_at_seg_start);
            }
        }

        const int n_segs = static_cast<int>(seg_buf.size()) / RT_FLOATS_PER_SEG;
        py::array_t<float> segs({static_cast<py::ssize_t>(n_segs),
                                  static_cast<py::ssize_t>(RT_FLOATS_PER_SEG)});
        if (n_segs > 0)
            std::memcpy(segs.mutable_data(), seg_buf.data(),
                        seg_buf.size() * sizeof(float));

        py::dict result;
        result["segs"]     = segs;
        result["direct"]   = direct;
        result["indirect"] = indirect;
        return result;
    }

    /* ── Persistent-machine async API ───────────────────────────────────── */

    void ensure_pipeline(
        int                  max_children  = 2,
        int                  seed          = 42,
        double               min_amplitude = 1e-6,
        bool                 use_gpu_compute = false,
        bool                 gpu_all_stages  = false,
        std::string          shader_dir      = "")
    {
        {
            std::lock_guard<std::mutex> lk(_pipeline_mu);
            if (!_pipeline) {
                _default_min_amplitude = min_amplitude;
                _use_gpu_compute = use_gpu_compute;
                _gpu_all_stages  = gpu_all_stages;
                if (!shader_dir.empty()) _shader_dir = shader_dir;
            }
        }
        (void)_get_pipeline(max_children, seed);
    }

    /*
     * submit_rays(origins, directions, amplitudes, src_ids, tags,
     *             max_bounces, min_amplitude, max_children, seed)
     *
     * Non-blocking: pushes ray intents into the persistent T1 input queue
     * and returns immediately.  The pipeline chews them as fast as it can.
     * Call drain_records() to collect output.
     */
    void submit_rays(
        py::array_t<double>  origins,
        py::array_t<double>  directions,
        py::object           amplitudes    = py::none(),
        py::object           src_ids       = py::none(),
        py::object           tags          = py::none(),
        py::object           color_flags   = py::none(),
        int                  max_bounces   = 8,
        double               min_amplitude = 1e-6,
        int                  max_children  = 2,
        int                  seed          = 42,
        bool                 use_gpu_compute = false,
        bool                 gpu_all_stages  = false,
        std::string          shader_dir      = "")
    {
        auto io = origins   .request();
        auto id = directions.request();
        if (io.ndim != 2 || io.shape[1] != 3)
            throw std::invalid_argument("origins must be (N,3)");
        if (id.ndim != 2 || id.shape[1] != 3)
            throw std::invalid_argument("directions must be (N,3)");
        const int n_rays = static_cast<int>(io.shape[0]);

        py::array_t<std::complex<double>> amp_arr;
        bool has_amp = !amplitudes.is_none();
        if (has_amp) {
            amp_arr = amplitudes.cast<py::array_t<std::complex<double>>>();
            if (amp_arr.request().shape[0] != n_rays)
                throw std::invalid_argument("amplitudes row count must match origins");
        }
        py::array_t<int32_t> sid_arr;
        bool has_sid = !src_ids.is_none();
        if (has_sid) {
            sid_arr = src_ids.cast<py::array_t<int32_t>>();
            if (sid_arr.request().size != n_rays)
                throw std::invalid_argument("src_ids length must match origins");
        }
        py::array_t<uint64_t> tag_arr;
        bool has_tag = !tags.is_none();
        if (has_tag) {
            tag_arr = tags.cast<py::array_t<uint64_t>>();
            if (tag_arr.request().size != n_rays)
                throw std::invalid_argument("tags length must match origins");
        }
        py::array_t<uint8_t> cflag_arr;
        bool has_cflag = !color_flags.is_none();
        if (has_cflag) {
            cflag_arr = color_flags.cast<py::array_t<uint8_t>>();
            if (cflag_arr.request().size != n_rays)
                throw std::invalid_argument("color_flags length must match origins");
        }

        const double* op = static_cast<const double*>(io.ptr);
        const double* dp = static_cast<const double*>(id.ptr);
        const int amp_bands = has_amp
            ? static_cast<int>(amp_arr.request().shape[1]) : _n_bands;
        const std::complex<double>* ap = has_amp
            ? static_cast<const std::complex<double>*>(amp_arr.request().ptr) : nullptr;
        const int32_t*  sp = has_sid
            ? static_cast<const int32_t*>(sid_arr.request().ptr) : nullptr;
        const uint64_t* tp = has_tag
            ? static_cast<const uint64_t*>(tag_arr.request().ptr) : nullptr;
        const uint8_t*  cp = has_cflag
            ? static_cast<const uint8_t*>(cflag_arr.request().ptr) : nullptr;

        std::vector<RayIntent> intents(n_rays);
        for (int i = 0; i < n_rays; ++i) {
            RayIntent& ri    = intents[i];
            ri.pos           = Eigen::Vector3d(op[i*3+0], op[i*3+1], op[i*3+2]);
            ri.dir           = Eigen::Vector3d(dp[i*3+0], dp[i*3+1], dp[i*3+2]).normalized();
            ri.amp.resize(amp_bands);
            if (ap) {
                for (int b = 0; b < amp_bands; ++b)
                    ri.amp[b] = ap[i*amp_bands + b];
            } else {
                ri.amp.setOnes();
            }
            ri.src_id        = sp ? sp[i] : i;
            ri.tag           = tp ? tp[i] : static_cast<uint64_t>(i);
            ri.color_flag    = cp ? cp[i] : 0u;
            const uint8_t film_channel = static_cast<uint8_t>((ri.tag >> 60) & 0x3u);
            if (((ri.color_flag & 1u) != 0u) && film_channel == 1u)
                ri.priority = 1.0e9f;
            ri.bounces_left  = max_bounces;
            ri.min_amplitude = min_amplitude;
            uint32_t bdpt_sid = _bdpt_subpath_counter.fetch_add(1u, std::memory_order_relaxed);
            if (bdpt_sid == 0u)
                bdpt_sid = _bdpt_subpath_counter.fetch_add(1u, std::memory_order_relaxed);
            ri.bdpt_subpath_id = bdpt_sid;
            ri.bdpt_vertex     = 0u;
            ri.bdpt_stream     = ((ri.color_flag & 1u) != 0u) ? BDPT_SIDE_SENSOR : BDPT_SIDE_LIGHT;
            ri.bdpt_strategy   = 0u;
            /* For backward (sensor-cast) rays, record the pixel origin so it
             * can be used to splat the sensor image after any number of bounces. */
            if ((ri.color_flag & 1u) != 0u) {
                ri.sensor_origin_y = static_cast<float>(op[i*3+1]);
                ri.sensor_origin_z = static_cast<float>(op[i*3+2]);
            }
        }

        /* Store GPU config before first pipeline creation (no-op if already created). */
        {
            std::lock_guard<std::mutex> lk(_pipeline_mu);
            if (!_pipeline) {
                _default_min_amplitude = min_amplitude;
                _use_gpu_compute = use_gpu_compute;
                _gpu_all_stages  = gpu_all_stages;
                if (!shader_dir.empty()) _shader_dir = shader_dir;
            }
        }
        RayPipelineState* ps = _get_pipeline(max_children, seed);
        {
            py::gil_scoped_release release;
            ray_pipeline_submit(ps, intents.data(), n_rays);
        }
    }

    int submit_emissive_triangles(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> tri_ids,
        int                  rays_per_tri,
        double               exposure_weight = 1.0,
        double               emitter_amp_gain = 1.0,
        int                  max_bounces   = 8,
        double               min_amplitude = 1e-6,
        int                  max_children  = 2,
        int                  seed          = 42,
        bool                 use_gpu_compute = false,
        bool                 gpu_all_stages  = false,
        std::string          shader_dir      = "",
        double               interaction_target_x = 0.0,
        double               interaction_target_y = 0.0,
        double               interaction_target_z = 0.0,
        double               interaction_target_r = 0.0)
    {
        auto ids = tri_ids.request();
        if (ids.ndim != 1)
            throw std::invalid_argument("tri_ids must be a 1-D int32 array");
        if (rays_per_tri <= 0 || ids.size <= 0)
            return 0;

        {
            std::lock_guard<std::mutex> lk(_pipeline_mu);
            if (!_pipeline) {
                _default_min_amplitude = min_amplitude;
                _use_gpu_compute = use_gpu_compute;
                _gpu_all_stages  = gpu_all_stages;
                if (!shader_dir.empty()) _shader_dir = shader_dir;
            }
        }
        RayPipelineState* ps = _get_pipeline(max_children, seed);
        py::gil_scoped_release release;
        return ray_pipeline_submit_emissive_triangles(
            ps,
            static_cast<const int*>(ids.ptr),
            static_cast<int>(ids.size),
            rays_per_tri,
            exposure_weight,
            emitter_amp_gain,
            max_bounces,
            min_amplitude,
            interaction_target_x,
            interaction_target_y,
            interaction_target_z,
            interaction_target_r,
            static_cast<uint32_t>(seed));
    }

    /*
     * drain_records(max_n=50000)
     *
     * Non-blocking: pops up to max_n completed records from the output queue.
     * Returns a dict of numpy arrays — empty arrays if nothing is ready.
     *
     * Array keys (all shape (N,) unless noted):
     *   kind         uint8   0=STRIKE 1=TERMINAL 2=MISS 3=FIELD
     *   tag          uint64
     *   src_id       int32
     *   bounce       int32
     *   seg_start    float32 (N,3)  ray origin entering this segment
     *   pos          float32 (N,3)  hit position or last position
     *   dir          float32 (N,3)  incoming ray direction
     *   normal       float32 (N,3)  surface normal (STRIKE)
     *   path_len     float32
     *   path_at_seg  float32
     *   hit_tri      int32
     *   mat_idx      int32
     *   arena_id     int32   (FIELD)
     *   is_sensor    bool
     *   sensor_gid   int32
     *   sensor_origin_y float32
     *   sensor_origin_z float32
     *   amp_re       float32 (N, n_bands)
     *   amp_im       float32 (N, n_bands)
     */
    py::dict drain_records(int max_n = 50000)
    {
        const int nb = _n_bands;
        std::vector<RayRecord> recs;

        if (_pipeline) {
            recs.reserve(std::min(max_n, 4096));
            py::gil_scoped_release release;
            ray_pipeline_drain(_pipeline, recs, max_n);
        }

        const int N = static_cast<int>(recs.size());

        py::array_t<uint8_t>  kind_arr(N);
        py::array_t<uint64_t> tag_arr(N);
        py::array_t<int32_t>  src_id_arr(N);
        py::array_t<int32_t>  bounce_arr(N);
        py::array_t<float>    seg_start_arr({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<float>    pos_arr      ({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<float>    dir_arr      ({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<float>    normal_arr   ({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<float>    path_len_arr(N);
        py::array_t<float>    path_seg_arr(N);
        py::array_t<int32_t>  hit_tri_arr(N);
        py::array_t<int32_t>  mat_idx_arr(N);
        py::array_t<int32_t>  arena_id_arr(N);
        py::array_t<bool>     is_sensor_arr(N);
        py::array_t<int32_t>  sensor_gid_arr(N);
        py::array_t<float>    bary_u_arr(N);
        py::array_t<float>    bary_v_arr(N);
        py::array_t<float>    sensor_origin_y_arr(N);
        py::array_t<float>    sensor_origin_z_arr(N);
        py::array_t<uint8_t>  color_flag_arr(N);
        py::array_t<float>    amp_re_arr({(py::ssize_t)N, (py::ssize_t)nb});
        py::array_t<float>    amp_im_arr({(py::ssize_t)N, (py::ssize_t)nb});

        if (N > 0) {
            auto* k   = kind_arr      .mutable_data();
            auto* tg  = tag_arr       .mutable_data();
            auto* si  = src_id_arr    .mutable_data();
            auto* bo  = bounce_arr    .mutable_data();
            auto* ss  = seg_start_arr .mutable_data();
            auto* po  = pos_arr       .mutable_data();
            auto* di  = dir_arr       .mutable_data();
            auto* no  = normal_arr    .mutable_data();
            auto* pl  = path_len_arr  .mutable_data();
            auto* ps_ = path_seg_arr  .mutable_data();
            auto* ht  = hit_tri_arr   .mutable_data();
            auto* mi  = mat_idx_arr   .mutable_data();
            auto* ai  = arena_id_arr  .mutable_data();
            auto* isn = is_sensor_arr .mutable_data();
            auto* sg  = sensor_gid_arr.mutable_data();
            auto* bu  = bary_u_arr    .mutable_data();
            auto* bv  = bary_v_arr    .mutable_data();
            auto* soy = sensor_origin_y_arr.mutable_data();
            auto* soz = sensor_origin_z_arr.mutable_data();
            auto* cf  = color_flag_arr.mutable_data();
            auto* re  = amp_re_arr    .mutable_data();
            auto* im  = amp_im_arr    .mutable_data();

            for (int i = 0; i < N; ++i) {
                const RayRecord& r = recs[i];
                k[i]  = static_cast<uint8_t>(r.kind);
                tg[i] = r.tag;
                si[i] = r.src_id;
                bo[i] = r.bounce;
                ss[i*3+0] = r.seg_start[0]; ss[i*3+1] = r.seg_start[1]; ss[i*3+2] = r.seg_start[2];
                po[i*3+0] = r.pos[0];       po[i*3+1] = r.pos[1];       po[i*3+2] = r.pos[2];
                di[i*3+0] = r.dir[0];       di[i*3+1] = r.dir[1];       di[i*3+2] = r.dir[2];
                no[i*3+0] = r.normal[0];    no[i*3+1] = r.normal[1];    no[i*3+2] = r.normal[2];
                pl[i]  = r.path_len;
                ps_[i] = r.path_at_seg_start;
                ht[i]  = r.hit_tri;
                mi[i]  = r.mat_idx;
                ai[i]  = r.arena_id;
                isn[i] = r.is_sensor;
                sg[i]  = r.sensor_group_id;
                bu[i]  = r.bary_u;
                bv[i]  = r.bary_v;
                soy[i] = r.sensor_origin_y;
                soz[i] = r.sensor_origin_z;
                cf[i]  = r.color_flag;
                const int cap = std::min((int)r.n_bands, nb);
                for (int b = 0; b < cap; ++b) {
                    re[i*nb + b] = r.amp_re[b];
                    im[i*nb + b] = r.amp_im[b];
                }
                for (int b = cap; b < nb; ++b) {
                    re[i*nb + b] = 0.f;
                    im[i*nb + b] = 0.f;
                }
            }
        }

        py::dict out;
        out["kind"]       = kind_arr;
        out["tag"]        = tag_arr;
        out["src_id"]     = src_id_arr;
        out["bounce"]     = bounce_arr;
        out["seg_start"]  = seg_start_arr;
        out["pos"]        = pos_arr;
        out["dir"]        = dir_arr;
        out["normal"]     = normal_arr;
        out["path_len"]   = path_len_arr;
        out["path_at_seg"] = path_seg_arr;
        out["hit_tri"]    = hit_tri_arr;
        out["mat_idx"]    = mat_idx_arr;
        out["arena_id"]   = arena_id_arr;
        out["is_sensor"]  = is_sensor_arr;
        out["sensor_gid"] = sensor_gid_arr;
        out["bary_u"]      = bary_u_arr;
        out["bary_v"]      = bary_v_arr;
        out["sensor_origin_y"] = sensor_origin_y_arr;
        out["sensor_origin_z"] = sensor_origin_z_arr;
        out["color_flag"]  = color_flag_arr;
        out["amp_re"]      = amp_re_arr;
        out["amp_im"]     = amp_im_arr;
        return out;
    }

    /* Slim drain: returns only the 7 arrays needed by the voxel accumulator.
     * Allocates ~5× less memory than drain_records() per call.
     * Keys: kind (uint8), tag (uint64), src_id (int32), bounce (int32), seg_start (N,3 f32),
     *       pos (N,3 f32), color_flag (uint8), hit_tri (int32), mat_idx (int32),
     *       sensor_origin_y/z (float32), amp_re (N,n_bands f32), amp_im. */
    py::dict drain_records_slim(int max_n = 50000)
    {
        const int nb = _n_bands;
        std::vector<RayRecord> recs;
        if (_pipeline) {
            recs.reserve(std::min(max_n, 4096));
            py::gil_scoped_release release;
            ray_pipeline_drain(_pipeline, recs, max_n);
        }
        const int N = static_cast<int>(recs.size());

        py::array_t<uint8_t> kind_arr(N);
        py::array_t<uint64_t> tag_arr(N);
        py::array_t<int32_t> src_id_arr(N);
        py::array_t<int32_t> bounce_arr(N);
        py::array_t<float>   seg_start_arr({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<float>   pos_arr      ({(py::ssize_t)N, (py::ssize_t)3});
        py::array_t<uint8_t> color_flag_arr(N);
        py::array_t<int32_t> hit_tri_arr(N);
        py::array_t<int32_t> hit_group_id_arr(N);
        py::array_t<int32_t> mat_idx_arr(N);
        py::array_t<float>   sensor_origin_y_arr(N);
        py::array_t<float>   sensor_origin_z_arr(N);
        py::array_t<float>   amp_re_arr({(py::ssize_t)N, (py::ssize_t)nb});
        py::array_t<float>   amp_im_arr({(py::ssize_t)N, (py::ssize_t)nb});

        if (N > 0) {
            auto* k  = kind_arr       .mutable_data();
            auto* tg = tag_arr        .mutable_data();
            auto* si = src_id_arr     .mutable_data();
            auto* bo = bounce_arr     .mutable_data();
            auto* ss = seg_start_arr  .mutable_data();
            auto* po = pos_arr        .mutable_data();
            auto* cf = color_flag_arr .mutable_data();
            auto* ht = hit_tri_arr    .mutable_data();
            auto* hg = hit_group_id_arr.mutable_data();
            auto* mi = mat_idx_arr    .mutable_data();
            auto* soy = sensor_origin_y_arr.mutable_data();
            auto* soz = sensor_origin_z_arr.mutable_data();
            auto* re = amp_re_arr     .mutable_data();
            auto* im = amp_im_arr     .mutable_data();
            for (int i = 0; i < N; ++i) {
                const RayRecord& r = recs[i];
                k[i]  = static_cast<uint8_t>(r.kind);
                tg[i] = r.tag;
                si[i] = r.src_id;
                bo[i] = r.bounce;
                std::memcpy(&ss[i*3], r.seg_start, 3*sizeof(float));
                std::memcpy(&po[i*3], r.pos,       3*sizeof(float));
                cf[i] = r.color_flag;
                ht[i] = r.hit_tri;
                hg[i] = r.hit_group_id;
                mi[i] = r.mat_idx;
                soy[i] = r.sensor_origin_y;
                soz[i] = r.sensor_origin_z;
                const int cap = std::min((int)r.n_bands, nb);
                std::memcpy(&re[i*nb], r.amp_re, (size_t)cap * sizeof(float));
                if (cap < nb) std::memset(&re[i*nb+cap], 0, (size_t)(nb-cap) * sizeof(float));
                std::memcpy(&im[i*nb], r.amp_im, (size_t)cap * sizeof(float));
                if (cap < nb) std::memset(&im[i*nb+cap], 0, (size_t)(nb-cap) * sizeof(float));
            }
        }
        py::dict out;
        out["kind"]        = kind_arr;
        out["tag"]         = tag_arr;
        out["src_id"]      = src_id_arr;
        out["bounce"]      = bounce_arr;
        out["seg_start"]   = seg_start_arr;
        out["pos"]         = pos_arr;
        out["color_flag"]  = color_flag_arr;
        out["hit_tri"]     = hit_tri_arr;
        out["hit_group_id"]= hit_group_id_arr;
        out["mat_idx"]     = mat_idx_arr;
        out["sensor_origin_y"] = sensor_origin_y_arr;
        out["sensor_origin_z"] = sensor_origin_z_arr;
        out["amp_re"]      = amp_re_arr;
        out["amp_im"]      = amp_im_arr;
        return out;
    }

    py::array_t<uint8_t> drain_records_raw(int max_n = 50000)
    {
        std::vector<RayRecord> recs;
        if (_pipeline) {
            recs.reserve(std::min(max_n, 4096));
            py::gil_scoped_release release;
            ray_pipeline_drain(_pipeline, recs, max_n);
        }
        const py::ssize_t n = static_cast<py::ssize_t>(recs.size());
        py::array_t<uint8_t> out({n, static_cast<py::ssize_t>(sizeof(RayRecord))});
        if (n > 0) {
            std::memcpy(out.mutable_data(), recs.data(), recs.size() * sizeof(RayRecord));
        }
        return out;
    }

    py::dict drain_records_display(
        int max_n,
        int image_res,
        float view_radius,
        float sensor_half_w,
        float sensor_half_h,
        py::array_t<int8_t, py::array::c_style | py::array::forcecast> tri_kind,
        py::array_t<float, py::array::c_style | py::array::forcecast> rgb_weights)
    {
        std::vector<RayRecord> recs;
        if (_pipeline) {
            recs.reserve(std::min(max_n, 4096));
            py::gil_scoped_release release;
            ray_pipeline_drain(_pipeline, recs, max_n);
        }

        const int N = static_cast<int>(recs.size());
        const int res = std::max(1, image_res);
        const int nb = _n_bands;
        const float vr = std::max(1.0e-9f, view_radius);
        const float half_w = std::max(1.0e-9f, sensor_half_w);
        const float half_h = std::max(1.0e-9f, sensor_half_h);

        auto tk_req = tri_kind.request();
        const int8_t* tk = static_cast<const int8_t*>(tk_req.ptr);
        const int n_tk = static_cast<int>(tk_req.size);

        auto w_req = rgb_weights.request();
        const float* w = static_cast<const float*>(w_req.ptr);
        const int w_rows = (w_req.ndim >= 2) ? static_cast<int>(w_req.shape[0]) : 0;
        const int w_cols = (w_req.ndim >= 2) ? static_cast<int>(w_req.shape[1]) : 0;
        const int use_bands = std::min(nb, w_rows);

        py::array_t<float> forward_arr({res, res, 3});
        py::array_t<float> reverse_arr({res, res, 3});
        py::array_t<float> camera_arr ({res, res, 3});
        float* fwd = forward_arr.mutable_data();
        float* rev = reverse_arr.mutable_data();
        float* cam = camera_arr .mutable_data();
        std::fill(fwd, fwd + static_cast<size_t>(res) * res * 3, 0.0f);
        std::fill(rev, rev + static_cast<size_t>(res) * res * 3, 0.0f);
        std::fill(cam, cam + static_cast<size_t>(res) * res * 3, 0.0f);

        int fwd_strikes = 0;
        int rev_strikes = 0;
        int fwd_lens_hits = 0;
        int fwd_lens_after_bounce = 0;

        auto tri_is_optic = [&](int htri) -> bool {
            if (htri < 0 || n_tk <= 0) return false;
            const int safe = std::min(std::max(htri, 0), n_tk - 1);
            const int8_t k = tk[safe];
            return k == 1 || k == 2 || k == 4; /* LENS/APERTURE/SENSOR from Python constants */
        };
        auto tri_is_lens = [&](int htri) -> bool {
            if (htri < 0 || n_tk <= 0) return false;
            const int safe = std::min(std::max(htri, 0), n_tk - 1);
            return tk[safe] == 1;
        };
        auto add_rgb = [&](float* dst, int flat, const RayRecord& r) {
            float rgb[3] = {0.0f, 0.0f, 0.0f};
            const int cap = std::min({static_cast<int>(r.n_bands), use_bands, RAY_RECORD_MAX_BANDS});
            if (cap > 0 && w_cols >= 3) {
                for (int b = 0; b < cap; ++b) {
                    const float mag = std::sqrt(std::max(0.0f, r.amp_re[b] * r.amp_re[b] + r.amp_im[b] * r.amp_im[b]));
                    rgb[0] += mag * w[b * w_cols + 0];
                    rgb[1] += mag * w[b * w_cols + 1];
                    rgb[2] += mag * w[b * w_cols + 2];
                }
            } else {
                rgb[0] = rgb[1] = rgb[2] = 1.0f;
            }
            const size_t base = static_cast<size_t>(flat) * 3u;
            dst[base + 0] += rgb[0];
            dst[base + 1] += rgb[1];
            dst[base + 2] += rgb[2];
        };

        for (const RayRecord& r : recs) {
            if (r.kind != RayRecordKind::STRIKE)
                continue;
            if (((r.color_flag & 1u) == 0u) && r.hit_tri >= 0) {
                ++fwd_strikes;
                if (tri_is_lens(r.hit_tri)) {
                    ++fwd_lens_hits;
                    if (r.bounce > 0) ++fwd_lens_after_bounce;
                }
            } else if (((r.color_flag & 1u) != 0u) && r.hit_tri >= 0) {
                ++rev_strikes;
            }

            const bool optic = tri_is_optic(r.hit_tri);
            if (!optic) {
                if (std::fabs(r.pos[1]) <= vr && std::fabs(r.pos[2]) <= vr) {
                    const int iy = std::min(std::max(static_cast<int>(((r.pos[1] + vr) / (2.0f * vr)) * res), 0), res - 1);
                    const int iz = std::min(std::max(static_cast<int>(((r.pos[2] + vr) / (2.0f * vr)) * res), 0), res - 1);
                    const int flat = iy * res + iz;
                    if ((r.color_flag & 1u) == 0u) {
                        add_rgb(fwd, flat, r);
                    } else if ((r.color_flag & 1u) != 0u) {
                        add_rgb(rev, flat, r);
                    }
                }
            }

            if (((r.color_flag & 1u) != 0u) && !optic && r.hit_tri >= 0 &&
                std::fabs(r.sensor_origin_y) <= half_w &&
                std::fabs(r.sensor_origin_z) <= half_h) {
                const int iy = std::min(std::max(static_cast<int>(((r.sensor_origin_y + half_w) / (2.0f * half_w)) * res), 0), res - 1);
                const int iz = std::min(std::max(static_cast<int>(((r.sensor_origin_z + half_h) / (2.0f * half_h)) * res), 0), res - 1);
                add_rgb(cam, iy * res + iz, r);
            }
        }

        py::dict out;
        out["n"] = N;
        out["forward"] = forward_arr;
        out["reverse"] = reverse_arr;
        out["camera"] = camera_arr;
        out["fwd_strikes"] = fwd_strikes;
        out["rev_strikes"] = rev_strikes;
        out["fwd_lens_hits"] = fwd_lens_hits;
        out["fwd_lens_after_bounce"] = fwd_lens_after_bounce;
        return out;
    }

    /* Set pipeline-wide amplitude floor and rebuild material epsilon flags.
     * Safe to call before or after the pipeline is created. */
    void set_min_amplitude(double eps) {
        _default_min_amplitude = eps;
        if (_pipeline)
            ray_pipeline_set_min_amplitude(_pipeline, eps);
    }

    /* (Re)scan material reflectances; flag fully-absorptive ones for fast T3 kill.
     * Pipeline must exist (call after first submit_rays). */
    void precompute_epsilon_material_flags() {
        if (_pipeline)
            ray_pipeline_precompute_epsilon_flags(_pipeline);
    }

    /* Set the maximum depth of the intent queue (backpressure bound).
     * 0 = unbounded.  Must be set before the pipeline is first created. */
    void set_max_intent_queue(int n) {
        _max_intent_queue = n;
    }

    /* Shuffle lever: 0 = strict FIFO, 1 = fully random window into Q_intent.
     * Can be changed at any time; takes effect on the next T1 pop. */
    void set_intent_shuffle(float frac) {
        if (_pipeline)
            ray_pipeline_set_shuffle(_pipeline, frac);
    }

    /* Configure sensor image accumulator.  Call before submitting rays.
     * plate_x, plate_r: world-space sensor plane position and disc radius.
     * res: pixel grid side length (res×res).
     * bdpt_eps: YZ proximity threshold in metres for BDPT snap. */
    void configure_sensor_image(float plate_x, float plate_half_w, float plate_half_h,
                                int res, float bdpt_eps,
                                float target_x = 0.0f, float target_r = 0.0f,
                                float target_y = 0.0f, float target_z = 0.0f,
                                int target_mode = 0) {
        /* Cache params unconditionally — pipeline may not exist yet (it is
         * created lazily on the first submit_rays call).  _get_pipeline will
         * apply these stored values when it constructs the pipeline. */
        _sensor_plate_x      = plate_x;
        _sensor_plate_half_w = plate_half_w;
        _sensor_plate_half_h = plate_half_h;
        _sensor_res          = res;
        _sensor_bdpt_eps     = bdpt_eps;
        _sensor_target_x     = target_x;
        _sensor_target_y     = target_y;
        _sensor_target_z     = target_z;
        _sensor_target_r     = target_r;
        _sensor_target_mode  = (target_mode == 1) ? 1 : 0;
        if (_pipeline)
            ray_pipeline_configure_sensor_image(_pipeline, plate_x, plate_half_w, plate_half_h,
                                                res, bdpt_eps,
                                                target_x, target_r,
                                                target_y, target_z, target_mode);
    }

    void configure_sensor_pose(
        py::array_t<double, py::array::c_style | py::array::forcecast> sensor_center,
        py::array_t<double, py::array::c_style | py::array::forcecast> sensor_right,
        py::array_t<double, py::array::c_style | py::array::forcecast> sensor_up,
        py::array_t<double, py::array::c_style | py::array::forcecast> aperture_center,
        py::array_t<double, py::array::c_style | py::array::forcecast> aperture_right,
        py::array_t<double, py::array::c_style | py::array::forcecast> aperture_up) {
        auto copy_vec3 = [](const auto& input, const char* name) {
            if (input.ndim() != 1 || input.shape(0) != 3)
                throw std::invalid_argument(std::string(name) + " must have shape (3,)");
            std::array<float, 3> result{};
            const double* values = input.data();
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(values[axis]))
                    throw std::invalid_argument(std::string(name) + " must be finite");
                result[axis] = static_cast<float>(values[axis]);
            }
            return result;
        };
        auto normalize_pair = [](std::array<float, 3>& right,
                                 std::array<float, 3>& up,
                                 const char* name) {
            auto norm = [](const std::array<float, 3>& value) {
                return std::sqrt(value[0]*value[0] + value[1]*value[1] + value[2]*value[2]);
            };
            const float nr = norm(right);
            const float nu = norm(up);
            if (!(nr > 1.0e-6f) || !(nu > 1.0e-6f))
                throw std::invalid_argument(std::string(name) + " axes must be non-zero");
            for (int axis = 0; axis < 3; ++axis) {
                right[axis] /= nr;
                up[axis] /= nu;
            }
            const float dot = right[0]*up[0] + right[1]*up[1] + right[2]*up[2];
            if (std::abs(dot) > 1.0e-4f)
                throw std::invalid_argument(std::string(name) + " right/up axes must be orthogonal");
        };
        _sensor_center = copy_vec3(sensor_center, "sensor_center");
        _sensor_right = copy_vec3(sensor_right, "sensor_right");
        _sensor_up = copy_vec3(sensor_up, "sensor_up");
        _aperture_center = copy_vec3(aperture_center, "aperture_center");
        _aperture_right = copy_vec3(aperture_right, "aperture_right");
        _aperture_up = copy_vec3(aperture_up, "aperture_up");
        normalize_pair(_sensor_right, _sensor_up, "sensor");
        normalize_pair(_aperture_right, _aperture_up, "aperture");
        _sensor_pose_configured = true;
        if (_pipeline)
            ray_pipeline_configure_sensor_pose(
                _pipeline, _sensor_center.data(), _sensor_right.data(), _sensor_up.data(),
                _aperture_center.data(), _aperture_right.data(), _aperture_up.data());
    }

    void configure_sensor_mipmap(uint32_t max_nodes = 65536u,
                                 uint32_t maximum_depth = 8u,
                                 uint32_t samples_per_epoch = 9u) {
        if (_pipeline)
            throw std::runtime_error(
                "configure_sensor_mipmap must be called before pipeline creation");
        if (max_nodes < 10u)
            throw std::invalid_argument("sensor mipmap requires at least 10 nodes");
        if (maximum_depth == 0u || maximum_depth > 20u)
            throw std::invalid_argument("sensor mipmap maximum_depth must be in [1, 20]");
        if (samples_per_epoch == 0u)
            throw std::invalid_argument("sensor mipmap samples_per_epoch must be positive");
        _sensor_mipmap_enabled = true;
        _sensor_mipmap_max_nodes = max_nodes;
        _sensor_mipmap_max_depth = maximum_depth;
        _sensor_mipmap_samples_per_epoch = samples_per_epoch;
    }

    void configure_sensor_priority_network(
        py::array_t<float, py::array::c_style | py::array::forcecast> parameters) {
        if (_pipeline)
            throw std::runtime_error(
                "configure_sensor_priority_network must be called before pipeline creation");
        auto p = parameters.request();
        if (p.ndim != 1 || p.size != SENSOR_PRIORITY_NETWORK_PARAMS)
            throw std::invalid_argument("priority network requires exactly 305 float parameters");
        std::memcpy(_sensor_priority_network_params.data(), p.ptr,
                    SENSOR_PRIORITY_NETWORK_PARAMS * sizeof(float));
        for (float value : _sensor_priority_network_params)
            if (!std::isfinite(value))
                throw std::invalid_argument("priority network parameters must be finite");
        _sensor_priority_network_enabled = true;
    }

    void configure_sensor_requested_priority_map(
        py::array_t<float, py::array::c_style | py::array::forcecast> values) {
        if (_pipeline)
            throw std::runtime_error(
                "requested priority map must be configured before pipeline creation");
        auto map = values.request();
        if (map.ndim != 2 || map.shape[0] <= 0 || map.shape[0] != map.shape[1])
            throw std::invalid_argument("requested priority map must be a non-empty square array");
        const float* source = static_cast<const float*>(map.ptr);
        _sensor_requested_priority_map.assign(source, source + map.size);
        for (float value : _sensor_requested_priority_map)
            if (!std::isfinite(value) || value < 0.0f)
                throw std::invalid_argument(
                    "requested priority map values must be finite and non-negative");
        _sensor_requested_priority_res = static_cast<uint32_t>(map.shape[0]);
    }

    void configure_sensor_delta_restore(
        py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
        py::array_t<float, py::array::c_style | py::array::forcecast> weight,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> dirty_sites) {
        if (_pipeline)
            throw std::runtime_error(
                "sensor delta restore must be configured before pipeline creation");
        auto image = rgb.request();
        auto exposure = weight.request();
        auto dirty = dirty_sites.request();
        if (image.ndim != 3 || image.shape[0] <= 0
                || image.shape[0] != image.shape[1] || image.shape[2] != 3)
            throw std::invalid_argument("restore RGB must be square HxWx3");
        if (exposure.ndim != 2 || exposure.shape[0] != image.shape[0]
                || exposure.shape[1] != image.shape[1])
            throw std::invalid_argument("restore weight must match RGB");
        if (dirty.ndim != 1)
            throw std::invalid_argument("dirty sensor sites must be a flat index array");
        const uint32_t res = static_cast<uint32_t>(image.shape[0]);
        const size_t pixels = static_cast<size_t>(res) * res;
        const float* source = static_cast<const float*>(image.ptr);
        _sensor_restore_rgb.assign(3u * pixels, 0.0f);
        for (size_t pixel = 0; pixel < pixels; ++pixel)
            for (size_t channel = 0; channel < 3u; ++channel)
                _sensor_restore_rgb[channel * pixels + pixel] =
                    source[pixel * 3u + channel];
        const float* weight_source = static_cast<const float*>(exposure.ptr);
        _sensor_restore_weight.assign(weight_source, weight_source + pixels);
        const uint32_t* dirty_source = static_cast<const uint32_t*>(dirty.ptr);
        _sensor_dirty_sites.assign(dirty_source, dirty_source + dirty.size);
        std::sort(_sensor_dirty_sites.begin(), _sensor_dirty_sites.end());
        _sensor_dirty_sites.erase(
            std::unique(_sensor_dirty_sites.begin(), _sensor_dirty_sites.end()),
            _sensor_dirty_sites.end());
        for (uint32_t index : _sensor_dirty_sites)
            if (index >= pixels)
                throw std::invalid_argument("dirty sensor site index is out of bounds");
        for (float value : _sensor_restore_rgb)
            if (!std::isfinite(value) || value < 0.0f)
                throw std::invalid_argument("restore RGB must be finite and non-negative");
        for (float value : _sensor_restore_weight)
            if (!std::isfinite(value) || value < 0.0f)
                throw std::invalid_argument("restore weight must be finite and non-negative");
        _sensor_restore_res = res;
    }

    /* ── GPU T3 bridge ─────────────────────────────────────────────────────
     *
     * drain_refined_hits(max_n)
     *   Dequeue up to max_n records from Q_refined and pack them into a flat
     *   float32 numpy array of shape (n, REFINED_HIT_STRIDE) using the layout
     *   defined in ray_material.comp.glsl.
     *
     *   REFINED_HIT_STRIDE = 27 + 2*MAX_BANDS  (MAX_BANDS = 32 -> stride = 91)
     *   row[58] carries bdpt_subpath_id as uintBitsToFloat.
     *
     *   Returns shape (0,) when the queue is empty.
     *
     * submit_intents_flat(buf)
     *   Accepts a float32 numpy array of shape (n, INTENT_STRIDE) — the output
     *   produced by the GPU T3 shader — and feeds each record back into the
     *   pipeline as a new child RayIntent.
     *
     *   INTENT_STRIDE = 20 + 2*MAX_BANDS  (stride = 84)
     *
     * Together these two methods allow Python to intercept the T3 stage and
     * run the GPU shader instead of the C++ worker, or to hybridise. */

    static constexpr int _GPU_MAX_BANDS     = 32;
    static constexpr int _REFINED_HIT_STRIDE = 27 + 2 * _GPU_MAX_BANDS;  /* 59 */
    static constexpr int _INTENT_STRIDE      = 20 + 2 * _GPU_MAX_BANDS;  /* 52 */

    py::array_t<float> drain_refined_hits(int max_n = 256) {
        auto* pl = _pipeline;  /* may be null if no rays yet submitted */
        if (!pl || max_n <= 0)
            return py::array_t<float>(std::vector<py::ssize_t>{0});

        std::vector<RefinedHit> batch;
        /* Non-blocking bulk drain: pop up to max_n in one call. */
        ray_pipeline_drain_refined(pl, batch, max_n);

        const int n   = static_cast<int>(batch.size());
        const int nb  = std::min(ray_pipeline_n_bands(pl), _GPU_MAX_BANDS);
        if (n == 0)
            return py::array_t<float>(std::vector<py::ssize_t>{0});

        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{n, _REFINED_HIT_STRIDE});
        float* p = arr.mutable_data();

        for (int i = 0; i < n; ++i) {
            const RefinedHit& rh = batch[static_cast<size_t>(i)];
            const HitRecord&  hr = rh.base;
            float* row = p + i * _REFINED_HIT_STRIDE;

            /* [0..2] hit_pos */
            row[0]  = static_cast<float>(rh.refined_pos.x());
            row[1]  = static_cast<float>(rh.refined_pos.y());
            row[2]  = static_cast<float>(rh.refined_pos.z());
            /* [3..5] hit_n */
            row[3]  = static_cast<float>(rh.refined_n.x());
            row[4]  = static_cast<float>(rh.refined_n.y());
            row[5]  = static_cast<float>(rh.refined_n.z());
            /* [6..8] incoming_dir */
            row[6]  = static_cast<float>(hr.incoming_dir.x());
            row[7]  = static_cast<float>(hr.incoming_dir.y());
            row[8]  = static_cast<float>(hr.incoming_dir.z());
            /* [9..11] seg_start */
            row[9]  = static_cast<float>(hr.seg_start.x());
            row[10] = static_cast<float>(hr.seg_start.y());
            row[11] = static_cast<float>(hr.seg_start.z());
            /* [12] path_len */
            row[12] = static_cast<float>(hr.ray.path_len);
            /* [13] path_at_seg_start */
            row[13] = static_cast<float>(hr.path_at_seg_start);
            /* [14] hit_tri */
            int hit_tri_i = hr.hit_tri;
            std::memcpy(&row[14], &hit_tri_i, sizeof(float));
            /* [15] mat_idx */
            int mat_idx_i = ray_pipeline_tri_mat_idx(pl, hr.hit_tri);
            std::memcpy(&row[15], &mat_idx_i, sizeof(float));
            /* [16] color_flag */
            uint32_t cflag = hr.ray.color_flag;
            std::memcpy(&row[16], &cflag, sizeof(float));
            /* [17] bounce */
            int bounce_i = hr.ray.bounce;
            std::memcpy(&row[17], &bounce_i, sizeof(float));
            /* [18] bounces_left */
            int bleft_i = hr.ray.bounces_left;
            std::memcpy(&row[18], &bleft_i, sizeof(float));
            /* [19] min_amplitude */
            row[19] = static_cast<float>(hr.ray.min_amplitude);
            /* [20] src_id */
            int src_id_i = hr.ray.src_id;
            std::memcpy(&row[20], &src_id_i, sizeof(float));
            /* [21] tag_lo, [22] tag_hi */
            uint32_t tag_lo = static_cast<uint32_t>(hr.ray.tag & 0xFFFFFFFFULL);
            uint32_t tag_hi = static_cast<uint32_t>(hr.ray.tag >> 32);
            std::memcpy(&row[21], &tag_lo, sizeof(float));
            std::memcpy(&row[22], &tag_hi, sizeof(float));
            /* [23] sensor_origin_y, [24] sensor_origin_z */
            row[23] = hr.ray.sensor_origin_y;
            row[24] = hr.ray.sensor_origin_z;
            /* [25] medium_mat_idx */
            int med_i = hr.ray.medium_mat_idx;
            std::memcpy(&row[25], &med_i, sizeof(float));
            /* [26 .. 26+nb-1] amp_re, [26+MAX_BANDS .. 26+MAX_BANDS+nb-1] amp_im */
            for (int b = 0; b < nb; ++b) {
                row[26 + b]              = static_cast<float>(hr.amp_propagated[b].real());
                row[26 + _GPU_MAX_BANDS + b] = static_cast<float>(hr.amp_propagated[b].imag());
            }
            for (int b = nb; b < _GPU_MAX_BANDS; ++b) {
                row[26 + b] = 0.0f;
                row[26 + _GPU_MAX_BANDS + b] = 0.0f;
            }
            uint32_t bdpt_sid = hr.ray.bdpt_subpath_id;
            std::memcpy(&row[58], &bdpt_sid, sizeof(float));
        }
        return arr;
    }

    /* Feed GPU-processed child intents back into the pipeline.
     * buf: float32 array of shape (n, INTENT_STRIDE=84).
     * Each row is decoded into a RayIntent and pushed to Q_intent. */
    void submit_intents_flat(py::array_t<float, py::array::c_style> buf) {
        auto* pl = _get_pipeline();
        auto info = buf.request();
        if (info.ndim != 2 || info.shape[1] != _INTENT_STRIDE) return;
        const int n  = static_cast<int>(info.shape[0]);
        const int nb = std::min(ray_pipeline_n_bands(pl), _GPU_MAX_BANDS);
        const float* p = static_cast<const float*>(info.ptr);

        for (int i = 0; i < n; ++i) {
            const float* row = p + i * _INTENT_STRIDE;
            RayIntent ri;
            ri.pos = Eigen::Vector3d(row[0], row[1], row[2]);
            ri.dir = Eigen::Vector3d(row[3], row[4], row[5]);
            ri.path_len = static_cast<double>(row[6]);
            std::memcpy(&ri.medium_mat_idx,    &row[7],  sizeof(int));
            uint32_t iflags_u; std::memcpy(&iflags_u, &row[8],  sizeof(uint32_t));
            ri.interaction_flags = iflags_u;
            std::memcpy(&ri.src_id,            &row[9],  sizeof(int));
            std::memcpy(&ri.bounce,            &row[10], sizeof(int));
            std::memcpy(&ri.bounces_left,      &row[11], sizeof(int));
            ri.min_amplitude = static_cast<double>(row[12]);
            uint32_t tag_lo; std::memcpy(&tag_lo, &row[13], sizeof(uint32_t));
            uint32_t tag_hi; std::memcpy(&tag_hi, &row[14], sizeof(uint32_t));
            ri.tag = (static_cast<uint64_t>(tag_hi) << 32) | tag_lo;
            uint32_t cflag; std::memcpy(&cflag, &row[15], sizeof(uint32_t));
            ri.color_flag = static_cast<uint8_t>(cflag & 0xFFu);
            ri.priority = row[16];
            ri.sensor_origin_y = row[17];
            ri.sensor_origin_z = row[18];
            uint32_t bdpt_sid = 0u;
            std::memcpy(&bdpt_sid, &row[19], sizeof(uint32_t));
            ri.bdpt_subpath_id = bdpt_sid;
            ri.bdpt_vertex = (ri.bounce < 0)
                ? 0u
                : static_cast<uint16_t>(std::min(ri.bounce, 0xFFFF));
            ri.bdpt_stream = ((ri.color_flag & 1u) != 0u) ? BDPT_SIDE_SENSOR : BDPT_SIDE_LIGHT;
            ri.bdpt_strategy = 0u;
            ri.amp.resize(nb);
            for (int b = 0; b < nb; ++b)
                ri.amp[b] = std::complex<double>(
                    static_cast<double>(row[20 + b]),
                    static_cast<double>(row[20 + _GPU_MAX_BANDS + b]));
            ray_pipeline_submit(pl, &ri, 1);
        }
    }

    /* Return current sensor image as a (res, res, 3) float32 numpy array.
     * Channels: R=forward plate hits, G=backward emissive+provisional near-miss,
     * B=exact BDPT snap.  Values are log-tone-mapped to [0, 1]. */
    py::array_t<float> get_sensor_image() {
        int res = 0;
        ray_pipeline_get_sensor_image(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<float>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{res, res, 3},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * 3 * sizeof(float)),
                (py::ssize_t)(3 * sizeof(float)),
                (py::ssize_t)(sizeof(float))});
        ray_pipeline_get_sensor_image(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    py::array_t<float> get_sensor_image_linear() {
        int res = 0;
        ray_pipeline_get_sensor_image_linear(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<float>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{res, res, 3},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * 3 * sizeof(float)),
                (py::ssize_t)(3 * sizeof(float)),
                (py::ssize_t)(sizeof(float))});
        ray_pipeline_get_sensor_image_linear(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    py::array_t<uint32_t> get_sensor_epoch_count() {
        int res = 0;
        ray_pipeline_get_sensor_epoch_count(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<uint32_t>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<uint32_t>(
            std::vector<py::ssize_t>{res, res},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * sizeof(uint32_t)),
                (py::ssize_t)(sizeof(uint32_t))});
        ray_pipeline_get_sensor_epoch_count(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    py::array_t<float> get_sensor_image_sum_linear() {
        int res = 0;
        ray_pipeline_get_sensor_image_sum_linear(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<float>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{res, res, 3},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * 3 * sizeof(float)),
                (py::ssize_t)(3 * sizeof(float)),
                (py::ssize_t)(sizeof(float))});
        ray_pipeline_get_sensor_image_sum_linear(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    py::array_t<float> get_sensor_exposure_weight() {
        int res = 0;
        ray_pipeline_get_sensor_exposure_weight(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<float>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{res, res},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * sizeof(float)),
                (py::ssize_t)(sizeof(float))});
        ray_pipeline_get_sensor_exposure_weight(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    py::array_t<float> get_sensor_learned_priority_map() {
        int res = 0;
        ray_pipeline_get_sensor_learned_priority_map(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<float>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{res, res},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * sizeof(float)),
                (py::ssize_t)(sizeof(float))});
        ray_pipeline_get_sensor_learned_priority_map(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    py::dict debug_sensor_mip_subdivide(
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> completed_ids)
    {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        auto ids = completed_ids.request();
        if (ids.ndim != 1 || ids.size <= 0)
            throw std::invalid_argument("completed_ids must be a non-empty uint32 vector");
        std::vector<SensorMipNodeGpu> nodes;
        std::array<uint32_t, 8> control{};
        bool ok = ray_pipeline_debug_sensor_mip_subdivide(
            _pipeline, static_cast<const uint32_t*>(ids.ptr),
            static_cast<int>(ids.size), nodes, control);
        if (!ok)
            throw std::runtime_error("GPU sensor mipmap subdivision failed");
        const py::ssize_t n = static_cast<py::ssize_t>(nodes.size());
        py::array_t<float> bounds({n, py::ssize_t(4)});
        py::array_t<uint32_t> meta({n, py::ssize_t(10)});
        auto b = bounds.mutable_unchecked<2>();
        auto m = meta.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& node = nodes[static_cast<size_t>(i)];
            for (int k = 0; k < 4; ++k) b(i, k) = node.uv_bounds[k];
            m(i, 0) = node.parent_id;
            m(i, 1) = node.first_child_id;
            m(i, 2) = node.level;
            m(i, 3) = node.flags;
            m(i, 4) = node.direct_moment_offset;
            m(i, 5) = node.rollup_moment_offset;
            m(i, 6) = node.direct_sample_count;
            m(i, 7) = node.completed_epochs;
            m(i, 8) = node.child_slot;
            uint32_t priority_bits = 0u;
            std::memcpy(&priority_bits, &node.priority, sizeof(uint32_t));
            m(i, 9) = priority_bits;
        }
        py::array_t<uint32_t> control_array(8);
        std::memcpy(control_array.mutable_data(), control.data(), 8 * sizeof(uint32_t));
        py::dict result;
        result["bounds"] = std::move(bounds);
        result["meta"] = std::move(meta);
        result["control"] = std::move(control_array);
        return result;
    }

    py::dict debug_sensor_mip_samples(uint32_t node_id,
                                      uint32_t sample_begin,
                                      uint32_t sample_count,
                                      uint32_t seed = 0u) {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        std::vector<SensorMipSampleLineageGpu> lineage;
        if (!ray_pipeline_debug_sensor_mip_samples(
                _pipeline, node_id, sample_begin, sample_count, seed, lineage))
            throw std::runtime_error("GPU sensor mipmap sample generation failed");
        const py::ssize_t n = static_cast<py::ssize_t>(lineage.size());
        py::array_t<float> uv({n, py::ssize_t(5)});
        py::array_t<uint32_t> meta({n, py::ssize_t(3)});
        auto u = uv.mutable_unchecked<2>();
        auto m = meta.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& sample = lineage[static_cast<size_t>(i)];
            u(i, 0) = sample.global_u;
            u(i, 1) = sample.global_v;
            u(i, 2) = sample.local_u;
            u(i, 3) = sample.local_v;
            u(i, 4) = sample.estimator_weight;
            m(i, 0) = sample.node_id;
            m(i, 1) = sample.level;
            m(i, 2) = sample.sample_index;
        }
        py::dict result;
        result["uv"] = std::move(uv);
        result["meta"] = std::move(meta);
        return result;
    }

    py::dict debug_sensor_mip_select(uint32_t top_k, uint32_t seed = 0u,
                                     float targeted_fraction = 1.0f) {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        std::vector<SensorMipWorkGpu> work;
        if (targeted_fraction < 0.0f || targeted_fraction > 1.0f)
            throw std::invalid_argument("targeted_fraction must be in [0, 1]");
        if (!ray_pipeline_debug_sensor_mip_select(
                _pipeline, top_k, seed, targeted_fraction, work))
            throw std::runtime_error("GPU sensor mipmap top-k selection failed");
        const py::ssize_t n = static_cast<py::ssize_t>(work.size());
        py::array_t<uint32_t> meta({n, py::ssize_t(7)});
        py::array_t<float> priority(n);
        auto m = meta.mutable_unchecked<2>();
        auto p = priority.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& item = work[static_cast<size_t>(i)];
            m(i, 0) = item.node_id;
            m(i, 1) = item.sample_begin;
            m(i, 2) = item.sample_count;
            m(i, 3) = item.output_offset;
            m(i, 4) = item.seed;
            m(i, 5) = item.flags;
            m(i, 6) = item._pad0;
            p(i) = item.priority;
        }
        py::dict result;
        result["meta"] = std::move(meta);
        result["priority"] = std::move(priority);
        return result;
    }

    py::array_t<float> debug_sensor_mip_score(
        py::array_t<float, py::array::c_style | py::array::forcecast> features,
        std::array<float, 4> weights = {1.0f, 1.0f, 1.0f, 4.0f})
    {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        auto f = features.request();
        if (f.ndim != 2 || f.shape[1] != 4 || f.shape[0] <= 0)
            throw std::invalid_argument("features must have shape (node_count, 4)");
        std::vector<SensorMipPriorityFeaturesGpu> input((size_t)f.shape[0]);
        const float* values = static_cast<const float*>(f.ptr);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i].uncertainty = values[i * 4 + 0];
            input[i].ambiguity = values[i * 4 + 1];
            input[i].learned = values[i * 4 + 2];
            input[i].requested = values[i * 4 + 3];
        }
        std::vector<float> priorities;
        if (!ray_pipeline_debug_sensor_mip_score(
                _pipeline, input.data(), static_cast<uint32_t>(input.size()),
                weights, priorities))
            throw std::runtime_error("GPU sensor mipmap scoring failed");
        py::array_t<float> result(priorities.size());
        std::memcpy(result.mutable_data(), priorities.data(),
                    priorities.size() * sizeof(float));
        return result;
    }

    py::dict debug_sensor_mip_accumulate(
        py::array_t<float, py::array::c_style | py::array::forcecast> spectra)
    {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        auto s = spectra.request();
        if (s.ndim != 2 || s.shape[0] <= 0 || s.shape[1] <= 0)
            throw std::invalid_argument("spectra must have shape (sample_count, n_bands)");
        std::vector<float> sum, sum_sq, weight;
        std::vector<uint32_t> sample_count;
        if (!ray_pipeline_debug_sensor_mip_accumulate(
                _pipeline, static_cast<const float*>(s.ptr),
                static_cast<uint32_t>(s.shape[0]), static_cast<uint32_t>(s.shape[1]),
                sum, sum_sq, weight, sample_count))
            throw std::runtime_error("GPU sensor mipmap accumulation failed");
        const py::ssize_t node_count = static_cast<py::ssize_t>(weight.size());
        const py::ssize_t n_bands = s.shape[1];
        py::array_t<float> sum_array({node_count, n_bands});
        py::array_t<float> sum_sq_array({node_count, n_bands});
        py::array_t<float> weight_array(node_count);
        py::array_t<uint32_t> count_array(node_count);
        std::memcpy(sum_array.mutable_data(), sum.data(), sum.size() * sizeof(float));
        std::memcpy(sum_sq_array.mutable_data(), sum_sq.data(), sum_sq.size() * sizeof(float));
        std::memcpy(weight_array.mutable_data(), weight.data(), weight.size() * sizeof(float));
        std::memcpy(count_array.mutable_data(), sample_count.data(),
                    sample_count.size() * sizeof(uint32_t));
        py::dict result;
        result["sum"] = std::move(sum_array);
        result["sum_sq"] = std::move(sum_sq_array);
        result["weight"] = std::move(weight_array);
        result["sample_count"] = std::move(count_array);
        return result;
    }

    py::dict debug_sensor_mip_rollup() {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        std::vector<float> mean, evidence;
        uint32_t n_bands = 0u;
        if (!ray_pipeline_debug_sensor_mip_rollup(
                _pipeline, mean, evidence, n_bands))
            throw std::runtime_error("GPU sensor mipmap rollup failed");
        const py::ssize_t node_count = static_cast<py::ssize_t>(evidence.size());
        py::array_t<float> mean_array({node_count, (py::ssize_t)n_bands});
        py::array_t<float> evidence_array(node_count);
        std::memcpy(mean_array.mutable_data(), mean.data(), mean.size() * sizeof(float));
        std::memcpy(evidence_array.mutable_data(), evidence.data(),
                    evidence.size() * sizeof(float));
        py::dict result;
        result["mean"] = std::move(mean_array);
        result["evidence"] = std::move(evidence_array);
        return result;
    }

    bool submit_sensor_mip_epoch(uint32_t top_k = 8u,
                                 uint32_t seed = 0u,
                                 uint32_t max_bounces = 8u,
                                 float min_amplitude = 1.0e-12f,
                                 float exposure_weight = 1.0f,
                                 float targeted_fraction = 1.0f) {
        if (!_pipeline)
            throw std::runtime_error("sensor mipmap pipeline has not been created");
        if (targeted_fraction < 0.0f || targeted_fraction > 1.0f)
            throw std::invalid_argument("targeted_fraction must be in [0, 1]");
        return ray_pipeline_submit_sensor_mip_epoch(
            _pipeline, top_k, seed, max_bounces, min_amplitude, exposure_weight,
            targeted_fraction);
    }

    /* Return the sugar-auxin priority map as a float32 (res, res) array.
     * Values >= 1.0; baseline = 1.0; elevated regions recently had BDPT
     * convergence and will receive more backward-ray budget next frame. */
    py::array_t<float> get_priority_map() {
        int res = 0;
        ray_pipeline_get_priority_map(_pipeline, nullptr, &res);
        if (res <= 0 || !_pipeline)
            return py::array_t<float>(std::vector<py::ssize_t>{0});
        auto arr = py::array_t<float>(
            std::vector<py::ssize_t>{res, res},
            std::vector<py::ssize_t>{
                (py::ssize_t)(res * sizeof(float)),
                (py::ssize_t)(sizeof(float))});
        ray_pipeline_get_priority_map(_pipeline, arr.mutable_data(), &res);
        return arr;
    }

    /* Live BDPT diagnostic stats (lock-free snapshot from T3 atomics).
     * Returns a dict with keys:
     *   nearest_dist_m     : float  — √(min YZ d²) in metres, -1 if no pair seen yet
     *   best_collinearity  : float  — |cos θ| of best collinear fwd/rev direction pair [0,1]
     *   exact_snaps        : int    — cumulative exact BDPT pixel hits (ch2)
     *   near_miss_count    : int    — cumulative near-miss pixel hits (ch3)
     */
    py::dict get_bdpt_stats() const {
        float    nd = -1.0f, bc = 0.0f;
        uint64_t es = 0,     nm = 0;
        ray_pipeline_get_bdpt_stats(_pipeline, &nd, &bc, &es, &nm);
        py::dict d;
        d["nearest_dist_m"]    = nd;
        d["best_collinearity"] = bc;
        d["exact_snaps"]       = static_cast<long long>(es);
        d["near_miss_count"]   = static_cast<long long>(nm);
        return d;
    }

    py::dict get_bdpt_latch_state() const {
        uint32_t flash = 0, sensor = 0, fired = 0;
        int running = 0;
        ray_pipeline_get_bdpt_latch_state(_pipeline, &flash, &sensor, &fired, &running);
        py::dict d;
        d["flash_dispatched"]   = static_cast<int>(flash);
        d["sensor_dispatched"]  = static_cast<int>(sensor);
        d["t5_fired"]           = static_cast<int>(fired);
        d["connection_running"] = running != 0;
        return d;
    }

    /* ── BDPT side-data drains ─────────────────────────────────────────────
     *
     * Each function returns a uint8 ndarray of shape (n, sizeof(Record)).
     * Python converts to a structured array via:
     *   records = np.frombuffer(arr.tobytes(), dtype=BDPT_XYZ_DTYPE)
     * or directly for aligned memory:
     *   records = arr.view(dtype=BDPT_XYZ_DTYPE).reshape(-1)
     *
     * Returns shape (0, stride) when the queue is empty. */

    py::array_t<uint8_t> drain_bdpt_vertices(int max_n = 100000) {
        constexpr py::ssize_t stride = static_cast<py::ssize_t>(sizeof(BdptVertexRecord));
        auto* pl = _pipeline;
        if (!pl || max_n <= 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        std::vector<BdptVertexRecord> batch;
        ray_pipeline_drain_bdpt_vertices(pl, batch, max_n);
        const py::ssize_t n = static_cast<py::ssize_t>(batch.size());
        if (n == 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{n, stride});
        std::memcpy(arr.mutable_data(), batch.data(),
                    static_cast<size_t>(n) * sizeof(BdptVertexRecord));
        return arr;
    }

    py::array_t<uint8_t> drain_bdpt_spectral(int max_n = 100000) {
        constexpr py::ssize_t stride = static_cast<py::ssize_t>(sizeof(BdptSpectralWeightRecord));
        auto* pl = _pipeline;
        if (!pl || max_n <= 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        std::vector<BdptSpectralWeightRecord> batch;
        ray_pipeline_drain_bdpt_spectral(pl, batch, max_n);
        const py::ssize_t n = static_cast<py::ssize_t>(batch.size());
        if (n == 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{n, stride});
        std::memcpy(arr.mutable_data(), batch.data(),
                    static_cast<size_t>(n) * sizeof(BdptSpectralWeightRecord));
        return arr;
    }

    py::array_t<uint8_t> drain_bdpt_pdfs(int max_n = 100000) {
        constexpr py::ssize_t stride = static_cast<py::ssize_t>(sizeof(BdptPdfRecord));
        auto* pl = _pipeline;
        if (!pl || max_n <= 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        std::vector<BdptPdfRecord> batch;
        ray_pipeline_drain_bdpt_pdfs(pl, batch, max_n);
        const py::ssize_t n = static_cast<py::ssize_t>(batch.size());
        if (n == 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{n, stride});
        std::memcpy(arr.mutable_data(), batch.data(),
                    static_cast<size_t>(n) * sizeof(BdptPdfRecord));
        return arr;
    }

    py::array_t<uint8_t> drain_bdpt_optical(int max_n = 100000) {
        constexpr py::ssize_t stride = static_cast<py::ssize_t>(sizeof(BdptOpticalEventRecord));
        auto* pl = _pipeline;
        if (!pl || max_n <= 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        std::vector<BdptOpticalEventRecord> batch;
        ray_pipeline_drain_bdpt_optical(pl, batch, max_n);
        const py::ssize_t n = static_cast<py::ssize_t>(batch.size());
        if (n == 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{n, stride});
        std::memcpy(arr.mutable_data(), batch.data(),
                    static_cast<size_t>(n) * sizeof(BdptOpticalEventRecord));
        return arr;
    }

    py::array_t<uint8_t> drain_bdpt_connections(int max_n = 100000) {
        constexpr py::ssize_t stride = static_cast<py::ssize_t>(sizeof(BdptConnectionRecord));
        auto* pl = _pipeline;
        if (!pl || max_n <= 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        std::vector<BdptConnectionRecord> batch;
        ray_pipeline_drain_bdpt_connections(pl, batch, max_n);
        const py::ssize_t n = static_cast<py::ssize_t>(batch.size());
        if (n == 0)
            return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, stride});
        auto arr = py::array_t<uint8_t>(std::vector<py::ssize_t>{n, stride});
        std::memcpy(arr.mutable_data(), batch.data(),
                    static_cast<size_t>(n) * sizeof(BdptConnectionRecord));
        return arr;
    }

    py::dict get_bdpt_overflow() const {
        uint64_t ov = 0, os = 0, op = 0, oo = 0, oc = 0;
        ray_pipeline_get_bdpt_overflow(_pipeline, &ov, &os, &op, &oo, &oc);
        py::dict d;
        d["vertices"]    = static_cast<long long>(ov);
        d["spectral"]    = static_cast<long long>(os);
        d["pdfs"]        = static_cast<long long>(op);
        d["optical"]     = static_cast<long long>(oo);
        d["connections"] = static_cast<long long>(oc);
        return d;
    }

    void run_bdpt_connection() {
        if (_pipeline) {
            py::gil_scoped_release release;
            ray_pipeline_run_bdpt_connection(_pipeline);
        }
    }

    int submit_sensor_sweep(int max_bounces = 8,
                            double min_amplitude = 1e-6,
                            int max_rays = 0,
                            int pix_offset = 0,
                            int max_children = 2,
                            int aperture_samples = 1,
                            int seed = 0,
                            int shutter_mode = 0,
                            double shutter_open = 1.0,
                            double shutter_center_u = 0.5,
                            double shutter_center_v = 0.5,
                            double shutter_softness = 0.0,
                            double exposure_weight = 1.0) {
        RayPipelineState* ps = _get_pipeline(max_children, seed);
        py::gil_scoped_release release;
        return ray_pipeline_submit_sensor_sweep(
            ps, max_bounces, min_amplitude, max_rays,
            pix_offset, aperture_samples, static_cast<uint64_t>(static_cast<unsigned int>(seed)),
            shutter_mode, shutter_open, shutter_center_u, shutter_center_v,
            shutter_softness, exposure_weight);
    }

    void begin_sensor_batching() {
        if (_pipeline) ray_pipeline_begin_sensor_batching(_pipeline);
    }

    void end_sensor_batching() {
        if (_pipeline) ray_pipeline_end_sensor_batching(_pipeline);
    }

    void signal_flash_dispatched() {
        if (_pipeline) ray_pipeline_signal_flash_dispatched(_pipeline);
    }

    void signal_sensor_dispatched() {
        if (_pipeline) ray_pipeline_signal_sensor_dispatched(_pipeline);
    }

    void join_t5() {
        if (_pipeline) {
            py::gil_scoped_release release;
            ray_pipeline_join_t5(_pipeline);
        }
    }

    void set_t5_min_geom(float v) {
        _t5_min_geom = v;
        ray_pipeline_set_t5_min_geom(_pipeline, v);
    }

    void set_t5_light_batch_size(uint32_t n) {
        _t5_light_batch_size = n;
        if (_pipeline) ray_pipeline_set_t5_light_batch_size(_pipeline, n);
    }

    void set_t5_cam_batch_size(uint32_t n) {
        _t5_cam_batch_size = n;
        if (_pipeline) ray_pipeline_set_t5_cam_batch_size(_pipeline, n);
    }

    void set_t5_sensor_tile_size(uint32_t n) {
        _t5_sensor_tile_size = n;
        if (_pipeline) ray_pipeline_set_t5_sensor_tile_size(_pipeline, n);
    }

    void set_t5_pair_budget(uint64_t n) {
        _t5_pair_budget = n;
        if (_pipeline) ray_pipeline_set_t5_pair_budget(_pipeline, n);
    }

    void set_t5_backlog_policy(float share_base, float share_max,
                               int max_snapshots, uint64_t max_bytes) {
        _t5_backlog_share_base    = share_base;
        _t5_backlog_share_max     = share_max;
        _t5_backlog_max_snapshots = max_snapshots;
        _t5_backlog_max_bytes     = max_bytes;
        if (_pipeline)
            ray_pipeline_set_t5_backlog_policy(_pipeline, share_base, share_max,
                                               max_snapshots, max_bytes);
    }

    void set_bdpt_record_float_budget(uint64_t n_floats) {
        _bdpt_record_float_budget = n_floats;
        if (_pipeline)
            ray_pipeline_set_bdpt_record_float_budget(_pipeline, n_floats);
    }

    void set_t5_profile(bool v) {
        _t5_profile = v;
        if (_pipeline) ray_pipeline_set_t5_profile(_pipeline, v);
    }

    void set_vcm(bool enabled, float merge_radius_m, float radius_alpha = 0.7f) {
        if (!(merge_radius_m > 0.0f) || !std::isfinite(merge_radius_m))
            throw std::invalid_argument("VCM merge radius must be finite and > 0 metres");
        if (!(radius_alpha > 0.0f && radius_alpha <= 1.0f) || !std::isfinite(radius_alpha))
            throw std::invalid_argument("VCM radius alpha must be finite and in (0, 1]");
        _vcm_enabled = enabled;
        _vcm_merge_radius_m = merge_radius_m;
        _vcm_radius_alpha = radius_alpha;
        if (_pipeline)
            ray_pipeline_set_vcm(_pipeline, enabled, merge_radius_m, radius_alpha);
    }

    void set_max_children(int n) {
        if (_pipeline) ray_pipeline_set_max_children(_pipeline, std::max(1, n));
    }

    void set_force_cpu_t5(bool v) {
        _force_cpu_t5 = v;
        if (_pipeline) ray_pipeline_set_force_cpu_t5(_pipeline, v);
    }

    void stop_pipeline() {
        std::lock_guard<std::mutex> lk(_pipeline_mu);
        if (_pipeline) {
            ray_pipeline_destroy(_pipeline);
            _pipeline = nullptr;
        }
    }

    void set_flash_modifier(int type_int, float param0, float param1) {
        _flash_modifier_type   = type_int;
        _flash_modifier_param0 = param0;
        _flash_modifier_param1 = param1;
        ray_pipeline_set_flash_modifier(_pipeline,
                                         static_cast<FlashModifierType>(type_int),
                                         param0, param1);
    }

    int in_flight_count() const
    {
        return _pipeline ? ray_pipeline_in_flight(_pipeline) : 0;
    }

    /* Snapshot of pipeline throughput / batch-size / queue-depth for all stages. */
    py::dict pipeline_stats() const
    {
        RayPipelineStats s{};
        if (_pipeline) ray_pipeline_get_stats(_pipeline, &s);

        auto stage_dict = [](const RayPipelineStats::Stage& st) {
            py::dict d;
            d["throughput"]      = st.throughput;
            d["processed"]       = st.processed;
            d["batch_size"]      = st.batch_size;
            d["queue_depth"]     = st.queue_depth;
            d["gpu_throughput"]  = st.gpu_throughput;
            d["gpu_processed"]   = st.gpu_processed;
            d["gpu_batch_size"]  = st.gpu_batch_size;
            d["gpu_fraction"]    = st.gpu_fraction;
            d["cpu_active_ms"]   = st.cpu_active_ms;
            d["gpu_active_ms"]   = st.gpu_active_ms;
            return d;
        };

        py::dict out;
        out["t1"]                 = stage_dict(s.t1);
        out["t2"]                 = stage_dict(s.t2);
        out["t3"]                 = stage_dict(s.t3);
        out["t4"]                 = stage_dict(s.t4);
        out["t5"]                 = stage_dict(s.t5);
        out["output_queue_depth"] = s.output_queue_depth;
        out["intent_queue_depth"] = s.intent_queue_depth;
        out["intent_queue_done"]  = s.intent_queue_done;
        out["in_flight"]          = s.in_flight;
        out["gpu_uv_readback_mb"] = static_cast<double>(s.gpu_uv_readback_bytes) / (1024.0 * 1024.0);
        out["gpu_uv_readback_count"] = static_cast<unsigned long long>(s.gpu_uv_readback_count);
        out["gpu_hit_readback_mb"] = static_cast<double>(s.gpu_hit_readback_bytes) / (1024.0 * 1024.0);
        out["gpu_hit_readback_count"] = static_cast<unsigned long long>(s.gpu_hit_readback_count);
        /* GPU dispatch health:
         *   0=disabled  1=init_failed  2=thread_spawned
         *   3=thread_running  4=thread_failed  5=thread_exited_ok */
        int gds = ray_pipeline_gpu_dispatch_state(_pipeline);
        out["gpu_dispatch_state"] = gds;
        static const char* const _gds_names[] = {
            "disabled", "init_failed", "thread_spawned",
            "thread_running", "thread_failed", "thread_exited_ok"};
        out["gpu_dispatch_status"] = (gds >= 0 && gds <= 5) ? _gds_names[gds] : "unknown";
        out["gpu_ok"] = (gds == 3 || gds == 5);
        return out;
    }

    void report_display_frame_time(double frame_ms, double target_ms = 16.667)
    {
        if (_pipeline)
            ray_pipeline_report_display_frame_time(_pipeline, frame_ms, target_ms);
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

    void integrate_image_into_packed(
        py::array_t<double> src_pos,
        py::array_t<double> src_dir,
        py::array_t<double> src_directivity,
        py::array_t<int32_t> src_n_rays,
        py::array_t<double> cam_pos,
        py::array_t<double> cam_fwd,
        py::array_t<double> cam_up,
        py::array_t<float>  out_image,
        double   fov_rad       = 1.0,
        int      max_bounces   = 12,
        double   min_amplitude = 0.001,
        uint32_t seed          = 42)
    {
        auto ip   = src_pos        .request();
        auto id_  = src_dir        .request();
        auto idv  = src_directivity.request();
        auto inr  = src_n_rays     .request();
        auto icp  = cam_pos        .request();
        auto icf  = cam_fwd        .request();
        auto icu  = cam_up         .request();
        auto oi   = out_image      .request();

        if (oi.ndim != 3 || oi.shape[0] != _n_bands)
            throw std::runtime_error("out_image must be float32 shape (n_bands, height, width)");
        if (inr.ndim != 1)
            throw std::runtime_error("src_n_rays must be int32 shape (n_sources,)");
        if (static_cast<int>(inr.shape[0]) != static_cast<int>(idv.size))
            throw std::runtime_error("src_n_rays length must match n_sources");

        int height = static_cast<int>(oi.shape[1]);
        int width  = static_cast<int>(oi.shape[2]);

        int rc;
        {
            py::gil_scoped_release release;
            rc = ray_tracer_integrate_image_packed(
                handle,
                static_cast<int>(idv.size),
                static_cast<const double*>(ip .ptr),
                static_cast<const double*>(id_.ptr),
                static_cast<const double*>(idv.ptr),
                static_cast<const int32_t*>(inr.ptr),
                max_bounces, min_amplitude, seed,
                static_cast<const double*>(icp.ptr),
                static_cast<const double*>(icf.ptr),
                static_cast<const double*>(icu.ptr),
                fov_rad, width, height,
                static_cast<float*>(oi.ptr));
        }

        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_integrate_image_packed failed: rc=" + std::to_string(rc));
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

    void set_camera_visibility(
        int camera_vis_mode = RT_CAM_VIS_AS_IS,
        int transparent_mode = RT_CAM_TRANSPARENCY_BLOCK,
        bool depth_cull_enabled = false,
        double depth_cull_m = 0.0)
    {
        int rc = ray_tracer_set_camera_visibility(
            handle,
            camera_vis_mode,
            transparent_mode,
            depth_cull_enabled ? 1 : 0,
            depth_cull_m);
        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_set_camera_visibility failed: rc=" + std::to_string(rc));

        _camera_vis_mode = camera_vis_mode;
        _transparent_mode = transparent_mode;
        _depth_cull_enabled = depth_cull_enabled;
        _depth_cull_m = depth_cull_m;
    }

    void set_sensor_film_ssbo(
        py::array_t<float, py::array::c_style | py::array::forcecast> sensor_chunk,
        py::array_t<float, py::array::c_style | py::array::forcecast> film_chunk,
        py::object active_slots_obj = py::none())
    {
        auto s = sensor_chunk.request();
        auto f = film_chunk.request();
        if (s.ndim != 2)
            throw std::runtime_error("sensor_chunk must be float32 shape (rows, stride)");
        if (f.ndim != 2)
            throw std::runtime_error("film_chunk must be float32 shape (rows, stride)");

        std::vector<int32_t> slots_flat;
        int n_slots = 0;
        if (!active_slots_obj.is_none()) {
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> slots_arr;
            try {
                slots_arr = active_slots_obj.cast<
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                auto sb = slots_arr.request();
                if (sb.ndim != 2 || sb.shape[1] != 2)
                    throw std::runtime_error("active_slots must be int32 shape (n_slots, 2)");
                n_slots = static_cast<int>(sb.shape[0]);
                const int32_t* ptr = static_cast<const int32_t*>(sb.ptr);
                slots_flat.assign(ptr, ptr + static_cast<size_t>(n_slots) * 2u);
            } catch (const py::cast_error&) {
                py::sequence seq = active_slots_obj.cast<py::sequence>();
                n_slots = static_cast<int>(py::len(seq));
                slots_flat.reserve(static_cast<size_t>(n_slots) * 2u);
                for (py::handle item : seq) {
                    py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
                    if (py::len(pair) != 2)
                        throw std::runtime_error("active_slots sequence entries must be length-2");
                    slots_flat.push_back(py::cast<int32_t>(pair[0]));
                    slots_flat.push_back(py::cast<int32_t>(pair[1]));
                }
            }
        }

        const int32_t* slots_ptr = slots_flat.empty() ? nullptr : slots_flat.data();
        int rc = ray_tracer_set_sensor_film_ssbo(
            handle,
            static_cast<const float*>(s.ptr),
            static_cast<int>(s.shape[0]),
            static_cast<int>(s.shape[1]),
            static_cast<const float*>(f.ptr),
            static_cast<int>(f.shape[0]),
            static_cast<int>(f.shape[1]),
            slots_ptr,
            n_slots);
        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_set_sensor_film_ssbo failed: rc=" + std::to_string(rc));
    }

    void set_surface_chunks(
        py::array_t<float, py::array::c_style | py::array::forcecast> pbr,
        py::array_t<float, py::array::c_style | py::array::forcecast> enamel,
        py::array_t<float, py::array::c_style | py::array::forcecast> tex_stack)
    {
        auto pb = pbr.request();
        auto en = enamel.request();
        auto tx = tex_stack.request();
        if (pb.ndim != 2 || pb.shape[1] != 16)
            throw std::runtime_error("set_surface_chunks: pbr must be (N,16) float32");
        if (en.ndim != 2 || en.shape[1] != 8)
            throw std::runtime_error("set_surface_chunks: enamel must be (N,8) float32");
        if (tx.ndim != 2 || tx.shape[1] != 16)
            throw std::runtime_error("set_surface_chunks: tex_stack must be (N,16) float32");
        int rc = ray_tracer_set_surface_chunks(
            handle,
            static_cast<const float*>(pb.ptr), static_cast<int>(pb.shape[0]),
            static_cast<const float*>(en.ptr), static_cast<int>(en.shape[0]),
            static_cast<const float*>(tx.ptr), static_cast<int>(tx.shape[0]));
        if (rc != SK_OK)
            throw std::runtime_error(
                "ray_tracer_set_surface_chunks failed: rc=" + std::to_string(rc));
    }

    py::dict get_camera_visibility() const
    {
        py::dict d;
        d["camera_vis_mode"] = py::int_(_camera_vis_mode);
        d["transparent_mode"] = py::int_(_transparent_mode);
        d["depth_cull_enabled"] = py::bool_(_depth_cull_enabled);
        d["depth_cull_m"] = py::float_(_depth_cull_m);

        uint64_t steps = 0, entries = 0;
        if (ray_tracer_get_camera_visibility_stats(handle, &steps, &entries) == SK_OK) {
            d["full_march_steps"] = py::int_(steps);
            d["full_march_context_entries"] = py::int_(entries);
        }
        return d;
    }

    void enable_field_capture_regular(
        int nx, int ny, int nz,
        py::array_t<float, py::array::c_style> bmin,
        py::array_t<float, py::array::c_style> bmax,
        bool capture_strikes = true,
        int max_strikes = 0,
        bool clear_existing = true)
    {
        auto b0 = bmin.request();
        auto b1 = bmax.request();
        if (b0.ndim != 1 || b0.shape[0] != 3 || b1.ndim != 1 || b1.shape[0] != 3)
            throw std::runtime_error("bmin/bmax must be float32 shape (3,)");

        FieldGrid* g = field_grid_create_regular(
            _n_bands, nx, ny, nz,
            static_cast<const float*>(b0.ptr),
            static_cast<const float*>(b1.ptr));
        if (!g) {
            const char* detail = field_grid_last_error();
            std::string msg = "field_grid_create_regular failed";
            if (detail && detail[0] != '\0') {
                msg += ": ";
                msg += detail;
            }
            throw std::runtime_error(msg);
        }

        int rc = ray_tracer_set_field_capture_grid(
            handle, g, 1,
            capture_strikes ? 1 : 0,
            max_strikes,
            clear_existing ? 1 : 0);
        if (rc != SK_OK) {
            field_grid_destroy(g);
            throw std::runtime_error("ray_tracer_set_field_capture_grid failed: rc=" + std::to_string(rc));
        }
    }

    void enable_field_capture_kdtree(
        py::list nodes,
        bool capture_strikes = true,
        int max_strikes = 0,
        bool clear_existing = true)
    {
        std::vector<KdNode> kd;
        kd.reserve(py::len(nodes));
        for (py::handle h : nodes) {
            py::dict d = py::reinterpret_borrow<py::dict>(h);
            KdNode n{};

            auto bm = d["bmin"].cast<py::array_t<float, py::array::c_style>>().unchecked<1>();
            auto bx = d["bmax"].cast<py::array_t<float, py::array::c_style>>().unchecked<1>();
            auto ld = d["leaf_dims"].cast<py::array_t<int32_t, py::array::c_style>>().unchecked<1>();
            for (int i = 0; i < 3; ++i) {
                n.bmin[i] = bm(i);
                n.bmax[i] = bx(i);
                n.leaf_dims[i] = ld(i);
            }
            n.child_lo = d.contains("child_lo") ? d["child_lo"].cast<int32_t>() : -1;
            n.child_hi = d.contains("child_hi") ? d["child_hi"].cast<int32_t>() : -1;
            n.split_axis = d.contains("split_axis") ? d["split_axis"].cast<int32_t>() : -1;
            n.split_pos = d.contains("split_pos") ? d["split_pos"].cast<float>() : 0.0f;
            n.first_data = d.contains("first_data") ? d["first_data"].cast<int64_t>() : -1;
            kd.push_back(n);
        }

        FieldGrid* g = field_grid_create_kdtree(
            _n_bands,
            kd.empty() ? nullptr : kd.data(),
            static_cast<int>(kd.size()));
        if (!g) {
            const char* detail = field_grid_last_error();
            std::string msg = "field_grid_create_kdtree failed";
            if (detail && detail[0] != '\0') {
                msg += ": ";
                msg += detail;
            }
            throw std::runtime_error(msg);
        }

        int rc = ray_tracer_set_field_capture_grid(
            handle, g, 1,
            capture_strikes ? 1 : 0,
            max_strikes,
            clear_existing ? 1 : 0);
        if (rc != SK_OK) {
            field_grid_destroy(g);
            throw std::runtime_error("ray_tracer_set_field_capture_grid failed: rc=" + std::to_string(rc));
        }
    }

    void clear_field_capture(bool clear_grid = true, bool clear_strikes = true)
    {
        int rc = ray_tracer_clear_field_capture(
            handle,
            clear_grid ? 1 : 0,
            clear_strikes ? 1 : 0);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_clear_field_capture failed: rc=" + std::to_string(rc));
    }

    py::dict get_field_capture_meta() const
    {
        int grid_kind = -1, n_bands = 0, stride = 0, n_strikes = 0;
        int64_t n_cells = 0;
        int rc = ray_tracer_get_field_capture_layout(
            handle, &grid_kind, &n_bands, &n_cells, &stride, &n_strikes);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_get_field_capture_layout failed: rc=" + std::to_string(rc));

        py::dict d;
        d["grid_kind"] = py::int_(grid_kind);
        d["n_bands"] = py::int_(n_bands);
        d["n_cells"] = py::int_(n_cells);
        d["strike_stride_floats"] = py::int_(stride);
        d["n_strikes"] = py::int_(n_strikes);
        return d;
    }

    py::array_t<float> get_field_capture_grid_reim() const
    {
        int grid_kind = -1, n_bands = 0, stride = 0, n_strikes = 0;
        int64_t n_cells = 0;
        int rc = ray_tracer_get_field_capture_layout(
            handle, &grid_kind, &n_bands, &n_cells, &stride, &n_strikes);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_get_field_capture_layout failed: rc=" + std::to_string(rc));
        if (grid_kind < 0)
            throw std::runtime_error("no field capture grid bound");

        py::array_t<float> out({n_bands, (int)n_cells, 2});
        rc = ray_tracer_copy_field_capture_grid_reim(
            handle,
            static_cast<float*>(out.request().ptr),
            static_cast<int64_t>(out.size()));
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_copy_field_capture_grid_reim failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_field_capture_strikes() const
    {
        int grid_kind = -1, n_bands = 0, stride = 0, n_strikes = 0;
        int64_t n_cells = 0;
        int rc = ray_tracer_get_field_capture_layout(
            handle, &grid_kind, &n_bands, &n_cells, &stride, &n_strikes);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_get_field_capture_layout failed: rc=" + std::to_string(rc));

        py::array_t<float> out({n_strikes, std::max(1, stride)});
        int written = 0;
        rc = ray_tracer_copy_field_capture_strikes(
            handle,
            static_cast<float*>(out.request().ptr),
            n_strikes,
            &written);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_copy_field_capture_strikes failed: rc=" + std::to_string(rc));
        if (written == n_strikes) return out;

        py::array_t<float> trimmed({written, std::max(1, stride)});
        if (written > 0) {
            std::memcpy(trimmed.request().ptr, out.request().ptr,
                        static_cast<size_t>(written) * stride * sizeof(float));
        }
        return trimmed;
    }

    py::dict accumulate_endpoint_records_to_field_capture(
        py::array_t<float, py::array::c_style | py::array::forcecast> records,
        int sensor_group_id,
        bool include_sensor_group = true,
        bool include_non_sensor_groups = true)
    {
        auto rb = records.request();
        if (rb.ndim != 2 || rb.shape[1] != 16)
            throw std::runtime_error("records must be float32 shape (N, 16)");

        auto recs = py::array_t<float, py::array::c_style | py::array::forcecast>(records);
        auto req = recs.request();
        const int n_records = static_cast<int>(req.shape[0]);
        const EndpointRecord* ptr = reinterpret_cast<const EndpointRecord*>(req.ptr);

        int written = 0;
        double power = 0.0;
        int rc = ray_tracer_accumulate_endpoint_records_to_field_capture(
            handle,
            ptr,
            n_records,
            sensor_group_id,
            include_sensor_group ? 1 : 0,
            include_non_sensor_groups ? 1 : 0,
            &written,
            &power);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_accumulate_endpoint_records_to_field_capture failed: rc=" + std::to_string(rc));

        py::dict out;
        out["written_records"] = py::int_(written);
        out["written_power"] = py::float_(power);
        return out;
    }

    /* ── Scale context API ───────────────────────────────────────────────── */

    int add_scale_context(
        py::array_t<double, py::array::c_style> pos,
        double radius, int scale_type, double dt_m, int n_substeps,
        double n_real, double n_imag,
        int context_kind,
        py::object payload)
    {
        if (!handle) throw std::runtime_error("RayTracer not initialised");
        auto cp = pos.request();
        if (cp.ndim != 1 || cp.shape[0] != 3)
            throw std::invalid_argument("pos must be shape (3,)");
        const double* pd = static_cast<const double*>(cp.ptr);

        RtScaleContext ctx{};
        ctx.center[0]  = pd[0];
        ctx.center[1]  = pd[1];
        ctx.center[2]  = pd[2];
        ctx.radius     = radius;
        ctx.scale_type = scale_type;
        ctx.dt_m       = dt_m;
        ctx.n_substeps = n_substeps;
        ctx.n_real     = n_real;
        ctx.n_imag     = n_imag;
        ctx.context_kind = context_kind;
        /* Payload is caller-owned for now (zero-copy) — Python keeps the
         * underlying bytes alive for the duration of the tracer.  Future:
         * deep-copy when payload_size_bytes > 0 (kind needs it). */
        if (!payload.is_none()) {
            auto buf = py::buffer(payload).request();
            ctx.payload = buf.ptr;
            ctx.payload_size_bytes = (int)(buf.size * buf.itemsize);
        }

        int rc = ray_tracer_add_scale_context(handle, &ctx);
        if (rc < 0)
            throw std::runtime_error("ray_tracer_add_scale_context failed: rc=" + std::to_string(rc));
        return rc; /* returns context_id */
    }

    void clear_scale_contexts()
    {
        if (!handle) throw std::runtime_error("RayTracer not initialised");
        ray_tracer_clear_scale_contexts(handle);
    }

    /* Trace multiscale, write into caller-owned (capacity, 14) float32 buffer.
     * Returns n_written. */
    int trace_multiscale_into(
        py::array_t<double, py::array::c_style> src_pos_arr,
        py::array_t<double, py::array::c_style> src_dir_arr,
        py::array_t<double, py::array::c_style> src_directivity_arr,
        py::array_t<float,  py::array::c_style> out_segs_arr,
        int    n_rays        = 512,
        int    max_bounces   = 12,
        double min_amplitude = 0.001,
        uint32_t seed        = 42)
    {
        if (!handle) throw std::runtime_error("RayTracer not initialised");
        auto isp  = src_pos_arr.request();
        auto isd  = src_dir_arr.request();
        auto isdi = src_directivity_arr.request();
        auto os   = out_segs_arr.request();
        if (isp.ndim != 2 || isp.shape[1] != 3)
            throw std::invalid_argument("src_pos must be shape (N, 3)");
        int n_sources = static_cast<int>(isp.shape[0]);
        if (os.ndim != 2 || os.shape[1] != RT_FLOATS_PER_SEG_MS)
            throw std::invalid_argument("out_segs must be shape (capacity, 14)");
        int out_cap = static_cast<int>(os.shape[0]);
        int n_written = 0;
        int rc = ray_tracer_trace_multiscale(
            handle, n_sources,
            static_cast<const double*>(isp.ptr),
            static_cast<const double*>(isd.ptr),
            static_cast<const double*>(isdi.ptr),
            n_rays, max_bounces, min_amplitude, seed,
            static_cast<float*>(os.ptr), out_cap, &n_written);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_trace_multiscale failed: rc=" + std::to_string(rc));
        return n_written;
    }

    /* Allocate segment buffer, trace, return ndarray (N, 14) float32. */
    py::array_t<float> trace_multiscale(
        py::array_t<double, py::array::c_style> src_pos_arr,
        py::array_t<double, py::array::c_style> src_dir_arr,
        py::array_t<double, py::array::c_style> src_directivity_arr,
        int    n_rays        = 512,
        int    max_bounces   = 12,
        double min_amplitude = 0.001,
        uint32_t seed        = 42,
        int    out_cap       = 65536)
    {
        if (!handle) throw std::runtime_error("RayTracer not initialised");
        auto isp  = src_pos_arr.request();
        auto isd  = src_dir_arr.request();
        auto isdi = src_directivity_arr.request();
        if (isp.ndim != 2 || isp.shape[1] != 3)
            throw std::invalid_argument("src_pos must be shape (N, 3)");
        int n_sources = static_cast<int>(isp.shape[0]);

        std::vector<float> buf(static_cast<size_t>(out_cap) * RT_FLOATS_PER_SEG_MS);
        int n_written = 0;
        int rc = ray_tracer_trace_multiscale(
            handle, n_sources,
            static_cast<const double*>(isp.ptr),
            static_cast<const double*>(isd.ptr),
            static_cast<const double*>(isdi.ptr),
            n_rays, max_bounces, min_amplitude, seed,
            buf.data(), out_cap, &n_written);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_trace_multiscale failed: rc=" + std::to_string(rc));

        py::array_t<float> result({n_written, RT_FLOATS_PER_SEG_MS});
        std::memcpy(result.mutable_data(), buf.data(),
                    static_cast<size_t>(n_written) * RT_FLOATS_PER_SEG_MS * sizeof(float));
        return result;
    }

    py::dict trace_multiscale_with_frequency_sidecar(
        py::array_t<double, py::array::c_style> src_pos_arr,
        py::array_t<double, py::array::c_style> src_dir_arr,
        py::array_t<double, py::array::c_style> src_directivity_arr,
        int    n_rays        = 512,
        int    max_bounces   = 12,
        double min_amplitude = 0.001,
        uint32_t seed        = 42,
        int    out_cap       = 65536)
    {
        py::array_t<float> segs = trace_multiscale(
            src_pos_arr,
            src_dir_arr,
            src_directivity_arr,
            n_rays,
            max_bounces,
            min_amplitude,
            seed,
            out_cap);
        py::dict out;
        out["segs"] = segs;
        out["freq_sidecar"] = build_frequency_sidecar(segs, 8);
        return out;
    }

    /* ── Physical optics extension methods ──────────────────────────────── */

    /** Set semantic surface flags for a range of triangles.
     *  Per-triangle physics (refl, IOR, diffusion, transmission) lives in the MatBuf
     *  addressed by tri.mat_idx (set at construction).  This entry point
     *  only adjusts non-material flags such as MAT_FLAG_APERTURE_STOP.
     */
    void set_tri_ior(int tri_start, int n_tris, int flags)
    {
        if (!handle) throw std::runtime_error("RayTracer not initialised");
        int rc = ray_tracer_set_tri_ior(handle, tri_start, n_tris, flags);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_set_tri_ior failed: rc=" + std::to_string(rc));
    }

    /** Set directional boundary media for a range of triangles.
     *  medium_pos_mat_idx = medium on +normal side
     *  medium_neg_mat_idx = medium on -normal side
     *  Use -1 for ambient air/vacuum.
     */
    void set_tri_boundary_media(
        int tri_start,
        int n_tris,
        int medium_pos_mat_idx,
        int medium_neg_mat_idx)
    {
        if (!handle) throw std::runtime_error("RayTracer not initialised");
        int rc = ray_tracer_set_tri_boundary_media(
            handle,
            tri_start,
            n_tris,
            medium_pos_mat_idx,
            medium_neg_mat_idx);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_set_tri_boundary_media failed: rc=" + std::to_string(rc));
    }

    /** Project a (N, 14) float32 multiscale segment array onto a coherent
     *  complex sensor image.  Returns (out_re, out_im) each shape
     *  (n_bands, sensor_h, sensor_w) float32, accumulative (NOT zeroed here).
     *
     *  segs_arr  : (N, 14) float32 multiscale segment buffer
     *  n_bands   : number of frequency bands
     *  sensor_w/h: sensor resolution in pixels
     *  sensor_z  : z-coordinate of sensor plane (metres)
     *  sensor_r  : half-width of sensor square in metres
     *  out_re/im : (n_bands, sensor_h, sensor_w) float32 — MUST be provided
     *              by caller and are accumulated into (not cleared).
     */
    void project_coherent(
        py::array_t<float, py::array::c_style> segs_arr,
        int    n_bands,
        int    sensor_w,
        int    sensor_h,
        double sensor_z,
        double sensor_r,
        py::array_t<float, py::array::c_style> out_re_arr,
        py::array_t<float, py::array::c_style> out_im_arr)
    {
        auto si  = segs_arr.request();
        auto rei = out_re_arr.request();
        auto imi = out_im_arr.request();
        if (si.ndim != 2 || si.shape[1] != RT_FLOATS_PER_SEG_MS)
            throw std::invalid_argument("segs must be shape (N, 14)");
        int n_segs = static_cast<int>(si.shape[0]);
        int rc = ray_tracer_project_coherent(
            n_segs,
            static_cast<const float*>(si.ptr),
            n_bands, sensor_w, sensor_h,
            sensor_z, sensor_r,
            static_cast<float*>(rei.ptr),
            static_cast<float*>(imi.ptr));
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_project_coherent failed: rc=" + std::to_string(rc));
    }

    /* ── Near-field aperture mask ───────────────────────────────────────── */

    void apply_aperture_mask(
        int n_bands,
        int w,
        int h,
        double field_r,
        py::array_t<float, py::array::c_style> poly_xy_arr,
        py::array_t<float, py::array::c_style> re_arr,
        py::array_t<float, py::array::c_style> im_arr)
    {
        auto pi  = poly_xy_arr.request();
        auto rei = re_arr.request();
        auto imi = im_arr.request();
        int n_verts = static_cast<int>(pi.size / 2);
        int rc = ray_tracer_apply_aperture_mask(
            n_bands, w, h, field_r,
            n_verts,
            static_cast<const float*>(pi.ptr),
            static_cast<float*>(rei.ptr),
            static_cast<float*>(imi.ptr));
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_apply_aperture_mask failed: rc=" + std::to_string(rc));
    }

    /* ── Rayleigh-Sommerfeld propagator ─────────────────────────────────── */

    py::tuple rs_propagate(
        int n_bands,
        int w,
        int h,
        double dx,
        double z_dist,
        py::array_t<double, py::array::c_style> wavelengths_arr,
        py::array_t<float,  py::array::c_style> in_re_arr,
        py::array_t<float,  py::array::c_style> in_im_arr)
    {
        auto wi  = wavelengths_arr.request();
        auto iri = in_re_arr.request();
        auto iii = in_im_arr.request();
        const size_t npix = static_cast<size_t>(n_bands) * w * h;
        py::array_t<float> out_re(npix);
        py::array_t<float> out_im(npix);
        int rc = ray_tracer_rs_propagate(
            n_bands, w, h, dx, z_dist,
            static_cast<const double*>(wi .ptr),
            static_cast<const float* >(iri.ptr),
            static_cast<const float* >(iii.ptr),
            static_cast<float*>(out_re.mutable_data()),
            static_cast<float*>(out_im.mutable_data()));
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_rs_propagate failed: rc=" + std::to_string(rc));
        /* Reshape to (n_bands, h, w) */
        out_re.resize({n_bands, h, w});
        out_im.resize({n_bands, h, w});
        return py::make_tuple(out_re, out_im);
    }

    /* ── BPM wave-PDE z-stepper (in-place, batchwise) ──────────────────── */

    void wave_bpm_step(
        int n_bands,
        int w,
        int h,
        double dx,
        double dz,
        py::array_t<double, py::array::c_style> wavelengths_arr,
        py::array_t<float,  py::array::c_style> re_arr,
        py::array_t<float,  py::array::c_style> im_arr)
    {
        auto wi  = wavelengths_arr.request();
        auto rei = re_arr.request();
        auto imi = im_arr.request();
        int rc = ray_tracer_wave_bpm_step(
            n_bands, w, h, dx, dz,
            static_cast<const double*>(wi.ptr),
            static_cast<float*>(rei.ptr),
            static_cast<float*>(imi.ptr));
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_wave_bpm_step failed: rc=" + std::to_string(rc));
    }

    /* ── Stateful scheduler ─────────────────────────────────────────────── */

    void spawn(
        py::array_t<double, py::array::c_style> src_pos_arr,
        py::array_t<double, py::array::c_style> src_dir_arr,
        py::array_t<double, py::array::c_style> src_dir_power_arr,
        int     n_rays,
        int     max_bounces   = 8,
        double  min_amplitude = 0.005,
        uint32_t seed         = 0)
    {
        auto pp = src_pos_arr.request();
        auto dp = src_dir_arr.request();
        auto ep = src_dir_power_arr.request();
        int n_sources = static_cast<int>(pp.shape[0]);
        int rc = ray_tracer_spawn(
            handle,
            n_sources,
            static_cast<const double*>(pp.ptr),
            static_cast<const double*>(dp.ptr),
            static_cast<const double*>(ep.ptr),
            n_rays, max_bounces, min_amplitude, seed);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_spawn failed: rc=" + std::to_string(rc));
    }

    py::tuple step(
        py::array_t<float, py::array::c_style> seg_buf_arr)
    {
        auto si   = seg_buf_arr.request();
        int  cap  = static_cast<int>(si.size / RT_FLOATS_PER_SEG_MS);
        int  n_written = 0;
        int  n_live    = 0;
        int rc = ray_tracer_step(
            handle,
            static_cast<float*>(si.ptr),
            cap, &n_written, &n_live);
        if (rc != SK_OK)
            throw std::runtime_error("ray_tracer_step failed: rc=" + std::to_string(rc));
        return py::make_tuple(n_written, n_live);
    }

    void clear_rays()
    {
        if (ray_tracer_clear_rays(handle) != SK_OK)
            throw std::runtime_error("ray_tracer_clear_rays failed");
    }

    int live_ray_count() const
    {
        return ray_tracer_live_ray_count(handle);
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
    /* AMR viz scatter grid (set only for from_amr_desc path) */
    int    _n_cells_amr = 0;
    double _bmin[3]     = {};
    double _bmax[3]     = {};

    PyAcousticCoEvolver() = default;  /* for from_amr_desc factory */

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

    /**
     * from_amr_desc(amr_desc, string_defs, pickup_defs, mic_defs,
     *               sample_rate, modal_stride) → PyAcousticCoEvolver
     *
     * AMR-backed constructor.  amr_desc is the dict returned by
     * build_amr_coevolver_descriptor() in acoustic_amr.py.
     * string/pickup/mic_defs use the same schema as the uniform constructor.
     */
    static PyAcousticCoEvolver* from_amr_desc(
        py::dict   amr_dict,
        py::list   string_defs_list,
        py::list   pickup_defs_list,
        py::list   mic_defs_list,
        float      sample_rate,
        int        modal_stride = 16)
    {
        /* ── Strings ── */
        int n_strings = (int)py::len(string_defs_list);
        std::vector<CoEvolverStringDef>    sdefs(n_strings);
        std::vector<std::vector<float>>    path_bufs(n_strings);
        for (int si = 0; si < n_strings; ++si) {
            py::dict d = string_defs_list[si].cast<py::dict>();
            auto path_arr = d["path_xyz"].cast<py::array_t<float>>();
            auto pi = path_arr.request();
            path_bufs[si].assign(
                static_cast<const float*>(pi.ptr),
                static_cast<const float*>(pi.ptr) + pi.size);
            sdefs[si].path_xyz           = path_bufs[si].data();
            sdefs[si].n_segs             = (int)(pi.size / 3) - 1;
            sdefs[si].tension_N          = d["tension_N"].cast<float>();
            sdefs[si].linear_mass_kgm    = d["linear_mass_kgm"].cast<float>();
            sdefs[si].damping            = d["damping"].cast<float>();
            sdefs[si].stiffness_EI       = d.contains("stiffness_EI")       ? d["stiffness_EI"].cast<float>()       : 0.0f;
            sdefs[si].axial_stiffness_N  = d.contains("axial_stiffness_N")  ? d["axial_stiffness_N"].cast<float>()  : 0.0f;
            sdefs[si].neck_freq_hz       = d.contains("neck_freq_hz")       ? d["neck_freq_hz"].cast<float>()       : 0.0f;
            sdefs[si].neck_mass_kg       = d.contains("neck_mass_kg")       ? d["neck_mass_kg"].cast<float>()       : 0.0f;
            sdefs[si].neck_Q             = d.contains("neck_Q")             ? d["neck_Q"].cast<float>()             : 0.0f;
        }

        /* ── Pickups ── */
        int n_pickups = (int)py::len(pickup_defs_list);
        std::vector<CoEvolverPickupDef> pdefs(n_pickups);
        auto copy3f = [](py::dict& d, const char* key, float* dst) {
            auto arr  = d[key].cast<py::array_t<float>>();
            auto info = arr.request();
            auto* src = static_cast<const float*>(info.ptr);
            dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2];
        };
        for (int pi2 = 0; pi2 < n_pickups; ++pi2) {
            py::dict d = pickup_defs_list[pi2].cast<py::dict>();
            pdefs[pi2].type         = d["type"].cast<int>();
            pdefs[pi2].pole_sigma   = d["pole_sigma"].cast<float>();
            pdefs[pi2].coil_spacing = d.contains("coil_spacing") ? d["coil_spacing"].cast<float>() : 0.018f;
            pdefs[pi2].sensitivity  = d.contains("sensitivity")  ? d["sensitivity"].cast<float>()  : 1.0f;
            pdefs[pi2].string_mask  = d.contains("string_mask")  ? (uint32_t)d["string_mask"].cast<int>() : 0xFFFFFFFFu;
            copy3f(d, "pos",  pdefs[pi2].pos);
            copy3f(d, "axis", pdefs[pi2].axis);
        }

        /* ── Mics ── */
        int n_mics = (int)py::len(mic_defs_list);
        std::vector<CoEvolverMicDef> mdefs(n_mics);
        for (int mi = 0; mi < n_mics; ++mi) {
            py::dict d = mic_defs_list[mi].cast<py::dict>();
            auto arr  = d["pos"].cast<py::array_t<float>>();
            auto info = arr.request();
            auto* src = static_cast<const float*>(info.ptr);
            mdefs[mi].pos[0]=src[0]; mdefs[mi].pos[1]=src[1]; mdefs[mi].pos[2]=src[2];
            mdefs[mi].gain    = d.contains("gain")    ? d["gain"].cast<float>()    : 1.0f;
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

        /* ── AMR descriptor ── */
        /* Keep numpy arrays alive for the duration of the call via py::array_t refs. */
        auto cc_arr   = amr_dict["cell_centers"]          .cast<py::array_t<double>>();
        auto cv_arr   = amr_dict["cell_volumes"]           .cast<py::array_t<double>>();
        auto ov_arr   = amr_dict["open_volume_frac"]       .cast<py::array_t<double>>();
        auto ct_arr   = amr_dict["cell_types"]             .cast<py::array_t<uint8_t>>();
        auto cl_arr   = amr_dict["cell_levels"]            .cast<py::array_t<int16_t>>();
        auto fn_arr   = amr_dict["face_cell_neg"]          .cast<py::array_t<int32_t>>();
        auto fp_arr   = amr_dict["face_cell_pos"]          .cast<py::array_t<int32_t>>();
        auto fa_arr   = amr_dict["face_area"]              .cast<py::array_t<double>>();
        auto fo_arr   = amr_dict["face_open_frac"]         .cast<py::array_t<double>>();
        auto fd_arr   = amr_dict["face_distance"]          .cast<py::array_t<double>>();
        auto pa_arr   = amr_dict["plate_active"]           .cast<py::array_t<uint8_t>>();
        auto pafi_arr = amr_dict["plate_active_flat_idx"]  .cast<py::array_t<int32_t>>();
        auto fas_arr  = amr_dict["plate_face_above_starts"].cast<py::array_t<int32_t>>();
        auto fai_arr  = amr_dict["plate_face_above_idx"]   .cast<py::array_t<int32_t>>();
        auto faw_arr  = amr_dict["plate_face_above_wgt"]   .cast<py::array_t<float>>();
        auto fbs_arr  = amr_dict["plate_face_below_starts"].cast<py::array_t<int32_t>>();
        auto fbi_arr  = amr_dict["plate_face_below_idx"]   .cast<py::array_t<int32_t>>();
        auto fbw_arr  = amr_dict["plate_face_below_wgt"]   .cast<py::array_t<float>>();
        auto pca_arr  = amr_dict["plate_cell_above"]       .cast<py::array_t<int32_t>>();
        auto pcb_arr  = amr_dict["plate_cell_below"]       .cast<py::array_t<int32_t>>();
        auto bpi_arr  = amr_dict["bridge_plate_idx"]       .cast<py::array_t<int32_t>>();
        auto bpw_arr  = amr_dict["bridge_plate_wgt"]       .cast<py::array_t<float>>();
        auto npi_arr  = amr_dict["neck_plate_idx"]         .cast<py::array_t<int32_t>>();
        auto npw_arr  = amr_dict["neck_plate_wgt"]         .cast<py::array_t<float>>();

        auto bmin_arr = amr_dict["bounds_min"].cast<py::array_t<double>>();
        auto bmax_arr = amr_dict["bounds_max"].cast<py::array_t<double>>();
        auto porg_arr = amr_dict["plate_origin"].cast<py::array_t<float>>();
        auto bmin_p   = static_cast<const double*>(bmin_arr.request().ptr);
        auto bmax_p   = static_cast<const double*>(bmax_arr.request().ptr);
        auto porg_p   = static_cast<const float*>(porg_arr.request().ptr);

        AMRCoevolverDescriptor desc{};
        desc.n_cells           = amr_dict["n_cells"].cast<int>();
        desc.cell_centers      = static_cast<const double*>(cc_arr.request().ptr);
        desc.cell_volumes      = static_cast<const double*>(cv_arr.request().ptr);
        desc.open_volume_frac  = static_cast<const double*>(ov_arr.request().ptr);
        desc.cell_types        = static_cast<const uint8_t*>(ct_arr.request().ptr);
        desc.cell_levels       = static_cast<const int16_t*>(cl_arr.request().ptr);
        desc.n_faces           = amr_dict["n_faces"].cast<int>();
        desc.face_cell_neg     = static_cast<const int32_t*>(fn_arr.request().ptr);
        desc.face_cell_pos     = static_cast<const int32_t*>(fp_arr.request().ptr);
        desc.face_area         = static_cast<const double*>(fa_arr.request().ptr);
        desc.face_open_frac    = static_cast<const double*>(fo_arr.request().ptr);
        desc.face_distance     = static_cast<const double*>(fd_arr.request().ptr);
        desc.c                 = amr_dict["c"].cast<double>();
        desc.rho_air           = amr_dict["rho_air"].cast<double>();
        desc.min_dx            = amr_dict["min_dx"].cast<double>();
        desc.bounds_min[0] = bmin_p[0]; desc.bounds_min[1] = bmin_p[1]; desc.bounds_min[2] = bmin_p[2];
        desc.bounds_max[0] = bmax_p[0]; desc.bounds_max[1] = bmax_p[1]; desc.bounds_max[2] = bmax_p[2];
        desc.plate_Nx              = amr_dict["plate_Nx"].cast<int>();
        desc.plate_Ny              = amr_dict["plate_Ny"].cast<int>();
        desc.plate_dx              = amr_dict["plate_dx"].cast<float>();
        desc.plate_origin[0] = porg_p[0]; desc.plate_origin[1] = porg_p[1]; desc.plate_origin[2] = porg_p[2];
        desc.plate_active          = static_cast<const uint8_t*>(pa_arr.request().ptr);
        desc.plate_mass_density    = amr_dict["plate_mass_density"].cast<float>();
        desc.plate_stiffness_D     = amr_dict["plate_stiffness_D"].cast<float>();
        desc.plate_alpha_M         = amr_dict["plate_alpha_M"].cast<float>();
        desc.plate_beta_K          = amr_dict["plate_beta_K"].cast<float>();
        desc.n_active_plate        = amr_dict["n_active_plate"].cast<int>();
        desc.plate_active_flat_idx  = static_cast<const int32_t*>(pafi_arr.request().ptr);
        desc.plate_face_above_starts= static_cast<const int32_t*>(fas_arr.request().ptr);
        desc.plate_face_above_idx   = static_cast<const int32_t*>(fai_arr.request().ptr);
        desc.plate_face_above_wgt   = static_cast<const float*>(faw_arr.request().ptr);
        desc.plate_face_below_starts= static_cast<const int32_t*>(fbs_arr.request().ptr);
        desc.plate_face_below_idx   = static_cast<const int32_t*>(fbi_arr.request().ptr);
        desc.plate_face_below_wgt   = static_cast<const float*>(fbw_arr.request().ptr);
        desc.plate_cell_above       = static_cast<const int32_t*>(pca_arr.request().ptr);
        desc.plate_cell_below       = static_cast<const int32_t*>(pcb_arr.request().ptr);
        desc.n_bridge_plate         = amr_dict["n_bridge_plate"].cast<int>();
        desc.bridge_plate_idx       = static_cast<const int32_t*>(bpi_arr.request().ptr);
        desc.bridge_plate_wgt       = static_cast<const float*>(bpw_arr.request().ptr);
        desc.n_neck_plate           = amr_dict["n_neck_plate"].cast<int>();
        desc.neck_plate_idx         = static_cast<const int32_t*>(npi_arr.request().ptr);
        desc.neck_plate_wgt         = static_cast<const float*>(npw_arr.request().ptr);
        desc.soundhole_cx           = amr_dict["soundhole_cx"].cast<double>();
        desc.soundhole_cy           = amr_dict["soundhole_cy"].cast<double>();
        desc.soundhole_radius       = amr_dict["soundhole_radius"].cast<double>();
        desc.border_mode            = amr_dict.contains("border_mode")         ? amr_dict["border_mode"].cast<int>()           : 0;
        desc.border_sigma_order     = amr_dict.contains("border_sigma_order")  ? amr_dict["border_sigma_order"].cast<float>()  : 3.0f;
        desc.border_R_reflection    = amr_dict.contains("border_R_reflection") ? amr_dict["border_R_reflection"].cast<float>() : 0.0f;
        desc.border_Z_match         = amr_dict.contains("border_Z_match")      ? amr_dict["border_Z_match"].cast<float>()      : 0.0f;
        desc.n_pml                  = amr_dict.contains("n_pml")               ? amr_dict["n_pml"].cast<int>()                 : 0;
        desc.gradient_order         = amr_dict["gradient_order"].cast<int>();

        auto* self = new PyAcousticCoEvolver();
        self->_n_strings = n_strings;
        self->_n_pickups = n_pickups;
        self->_n_mics    = n_mics;
        {
            py::gil_scoped_release release;
            self->handle = coevolver_create_amr(
                n_strings, sdefs.empty()  ? nullptr : sdefs.data(),
                n_pickups, pdefs.empty()  ? nullptr : pdefs.data(),
                n_mics,    mdefs.empty()  ? nullptr : mdefs.data(),
                &desc,
                sample_rate, modal_stride);
        }
        if (!self->handle) {
            const char* err_code  = amr_pressure_backend_last_error_code();
            const char* err_msg   = amr_pressure_backend_last_error_message();
            const char* amr_stage = coevolver_create_amr_last_error();
            std::string detail = "coevolver_create_amr failed";
            if (err_code && err_code[0]) {
                detail += " [";
                detail += err_code;
                detail += "]";
            }
            if (err_msg && err_msg[0]) {
                detail += ": ";
                detail += err_msg;
            }
            if (amr_stage && amr_stage[0]) {
                detail += " (stage: ";
                detail += amr_stage;
                detail += ")";
            }
            delete self;
            throw std::runtime_error(detail);
        }
        /* Compute visualization grid for the AMR pressure scatter path.
         * viz_dx = max(min_dx * 4, 0.005) keeps the texture small (< 256³). */
        {
            double viz_dx = std::max(desc.min_dx * 4.0, 0.005);
            auto clamp256 = [](int v) { return std::max(1, std::min(v, 256)); };
            self->_Nx = clamp256((int)std::round((desc.bounds_max[0] - desc.bounds_min[0]) / viz_dx) + 1);
            self->_Ny = clamp256((int)std::round((desc.bounds_max[1] - desc.bounds_min[1]) / viz_dx) + 1);
            self->_Nz = clamp256((int)std::round((desc.bounds_max[2] - desc.bounds_min[2]) / viz_dx) + 1);
            self->_n_cells_amr = desc.n_cells;
            self->_bmin[0] = desc.bounds_min[0];
            self->_bmin[1] = desc.bounds_min[1];
            self->_bmin[2] = desc.bounds_min[2];
            self->_bmax[0] = desc.bounds_max[0];
            self->_bmax[1] = desc.bounds_max[1];
            self->_bmax[2] = desc.bounds_max[2];
        }
        return self;
    }


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

    void pluck_string(int string_idx, float position_norm, float amplitude,
                      int n_duration_samples = 0)
    {
        int rc = coevolver_pluck_string(handle, string_idx, position_norm,
                                        amplitude, n_duration_samples);
        if (rc != CE_OK)
            throw std::runtime_error("coevolver_pluck_string error: " + std::to_string(rc));
    }

    void schedule_pluck(int onset_sample, int string_idx,
                        float position_norm, float amplitude,
                        int n_duration_samples = 0)
    {
        int rc = coevolver_schedule_pluck(handle, onset_sample, string_idx,
                                          position_norm, amplitude,
                                          n_duration_samples);
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
        int rc;
        py::array_t<float> out({_Nx, _Ny, _Nz});
        if (_n_cells_amr > 0) {
            /* AMR path: scatter cell pressures onto the viz uniform grid.
             * Cost is O(n_cells) here vs O(n_cells × fragments × steps) in the
             * brute-force TBO shader — always use the scatter route. */
            rc = coevolver_get_pressure_field_uniform(
                handle,
                _Nx, _Ny, _Nz,
                _bmin, _bmax,
                out.mutable_unchecked<3>().mutable_data(0, 0, 0),
                _Nx * _Ny * _Nz);
        } else {
            rc = coevolver_get_pressure_field(handle,
                out.mutable_unchecked<3>().mutable_data(0, 0, 0),
                _Nx * _Ny * _Nz);
        }
        if (rc != CE_OK)
            throw std::runtime_error("get_pressure_field error: " + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_amr_cell_centers()
    {
        if (_n_cells_amr <= 0)
            throw std::runtime_error("get_amr_cell_centers: not an AMR coevolver");
        py::array_t<float> out({_n_cells_amr, 3});
        int rc = coevolver_get_amr_cell_centers(
            handle,
            out.mutable_unchecked<2>().mutable_data(0, 0),
            _n_cells_amr * 3);
        if (rc != CE_OK)
            throw std::runtime_error("get_amr_cell_centers error: " + std::to_string(rc));
        return out;
    }

    py::tuple get_plate_dims()
    {
        int plate_Nx = 0;
        int plate_Ny = 0;
        int plate_count = 0;
        int rc = coevolver_get_plate_dims(handle, &plate_Nx, &plate_Ny, &plate_count);
        if (rc != CE_OK)
            throw std::runtime_error("get_plate_dims error: " + std::to_string(rc));
        return py::make_tuple(plate_Nx, plate_Ny, plate_count);
    }

    py::array_t<float> get_plate_displacement()
    {
        int plate_Nx = 0;
        int plate_Ny = 0;
        int plate_count = 0;
        int rc = coevolver_get_plate_dims(handle, &plate_Nx, &plate_Ny, &plate_count);
        if (rc != CE_OK)
            throw std::runtime_error("get_plate_dims error: " + std::to_string(rc));
        if (plate_Nx <= 0 || plate_Ny <= 0 || plate_count != plate_Nx * plate_Ny)
            throw std::runtime_error(
                "get_plate_dims returned invalid dimensions: " +
                std::to_string(plate_Nx) + "x" + std::to_string(plate_Ny) +
                " count=" + std::to_string(plate_count));

        py::array_t<float> out({plate_Nx, plate_Ny});
        rc = coevolver_get_plate_displacement(handle,
            out.mutable_unchecked<2>().mutable_data(0,0),
            plate_count);
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

/* ── Python wrapper for AcousticAMR ─────────────────────────────────────── */

struct PyAcousticAMR
{
    AcousticAMRState* handle = nullptr;
    int n_cells = 0;
    int n_faces = 0;

    PyAcousticAMR(py::array_t<double>  cell_centers_arr,
                  py::array_t<double>  cell_volumes_arr,
                  py::array_t<double>  open_volume_frac_arr,
                  py::array_t<uint8_t> cell_types_arr,
                  py::array_t<int32_t> face_cell_neg_arr,
                  py::array_t<int32_t> face_cell_pos_arr,
                  py::array_t<double>  face_area_arr,
                  py::array_t<double>  face_open_frac_arr,
                  py::array_t<double>  face_distance_arr,
                  double c,
                  double rho_air,
                  double min_dx,
                  int gradient_order)
    {
        auto cc = cell_centers_arr.request();
        auto cv = cell_volumes_arr.request();
        auto ov = open_volume_frac_arr.request();
        auto ct = cell_types_arr.request();
        auto fn = face_cell_neg_arr.request();
        auto fp = face_cell_pos_arr.request();
        auto fa = face_area_arr.request();
        auto fo = face_open_frac_arr.request();
        auto fd = face_distance_arr.request();

        if (cc.ndim != 2 || cc.shape[1] != 3)
            throw std::runtime_error("cell_centers must have shape (n_cells, 3)");
        n_cells = static_cast<int>(cc.shape[0]);
        if (cv.size != n_cells || ov.size != n_cells || ct.size != n_cells)
            throw std::runtime_error("cell arrays must have n_cells elements");
        n_faces = static_cast<int>(fn.size);
        if (fp.size != n_faces || fa.size != n_faces || fo.size != n_faces || fd.size != n_faces)
            throw std::runtime_error("face arrays must have n_faces elements");

        handle = amr_create(
            n_cells,
            static_cast<const double*>(cc.ptr),
            static_cast<const double*>(cv.ptr),
            static_cast<const double*>(ov.ptr),
            static_cast<const uint8_t*>(ct.ptr),
            n_faces,
            static_cast<const int32_t*>(fn.ptr),
            static_cast<const int32_t*>(fp.ptr),
            static_cast<const double*>(fa.ptr),
            static_cast<const double*>(fo.ptr),
            static_cast<const double*>(fd.ptr),
            c, rho_air, min_dx, gradient_order);
        if (!handle)
            throw std::runtime_error("amr_create failed: invalid AMR topology or allocation failure");
    }

    ~PyAcousticAMR() { amr_destroy(handle); handle = nullptr; }

    int step(int n_steps = 1)
    {
        int rc = amr_step(handle, n_steps);
        if (rc != SK_OK) throw std::runtime_error("amr_step failed: rc=" + std::to_string(rc));
        return rc;
    }

    void reset()
    {
        int rc = amr_reset(handle);
        if (rc != SK_OK) throw std::runtime_error("amr_reset failed: rc=" + std::to_string(rc));
    }

    void inject_pressure_nearest(py::array_t<double> xyz_arr, float value)
    {
        auto xi = xyz_arr.request();
        if (xi.size != 3) throw std::runtime_error("xyz must have 3 elements");
        int rc = amr_inject_pressure_nearest(
            handle, static_cast<const double*>(xi.ptr), value);
        if (rc != SK_OK)
            throw std::runtime_error("amr_inject_pressure_nearest failed: rc=" + std::to_string(rc));
    }

    py::array_t<float> get_pressure() const
    {
        py::array_t<float> out({n_cells});
        int rc = amr_get_pressure(handle, out.mutable_data(), n_cells);
        if (rc != SK_OK) throw std::runtime_error("amr_get_pressure failed: rc=" + std::to_string(rc));
        return out;
    }

    py::array_t<float> get_velocity() const
    {
        py::array_t<float> out({n_faces});
        int rc = amr_get_velocity(handle, out.mutable_data(), n_faces);
        if (rc != SK_OK) throw std::runtime_error("amr_get_velocity failed: rc=" + std::to_string(rc));
        return out;
    }

    double get_dt() const { return amr_get_dt(handle); }
    int get_step_count() const { return amr_get_step_count(handle); }
};

/* ── Module definition ───────────────────────────────────────────────────── */

PYBIND11_MODULE(_spectral_kernels, m)
{
    m.doc() = "Spectral-analyzer C serial kernel extensions";

    m.def("estimate_lens_camera_jacobians",
          &estimate_lens_camera_jacobians_py,
          py::arg("payload"),
          py::arg("film_origins"),
          py::arg("aperture_points"),
          py::arg("base_exit_origins"),
          py::arg("base_exit_dirs"),
          py::arg("plate_radius"),
          py::arg("aperture_radius"),
          py::arg("tb"),
          py::arg("tc"),
          py::arg("n_threads") = 0,
          R"doc(
Threaded C++/Eigen finite-difference camera transfer Jacobians.

Returns (aperture_to_solid_angle_jac, phase_space_jac) for a batch of
parametric compound-lens camera samples using the compact PLENS payload.
)doc");

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
    [REMOVED in Phase 2]  Complex reflectance now lives in mat_buf.
diffusion : float64 array (n_tri,)
    [REMOVED in Phase 2]  Diffuse scatter fraction now lives in mat_buf (slot 4).
mat_idx : int32 array (n_tri,)
    Per-triangle material index (row in mat_buf).
mat_buf : float32 array (mat_n_mats * MAX_SPECTRAL_BANDS, 12)
    Flat unified material buffer shared with the GLSL backend.
mat_n_mats : int
    Number of registered materials in mat_buf.
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
                      py::array_t<int>,     /* mat_idx    */
                      py::array_t<float>,   /* mat_buf    */
                      int,                  /* mat_n_mats */
                      py::array_t<double>,  /* freq_hz    */
                      double,               /* speed_m_s  */
                      py::array_t<double>   /* atmo_abs   */
                      >(),
             py::arg("n_tri"),
             py::arg("verts"),
             py::arg("normals"),
             py::arg("mat_idx"),
             py::arg("mat_buf"),
             py::arg("mat_n_mats"),
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
                .def("spectral_bands_to_rgb", &PyRayTracer::spectral_bands_to_rgb,
                         py::arg("image_bhw"),
                         py::arg("gain") = 1.0,
                         py::arg("hdr_white_percentile") = 99.8,
R"doc(
Convert spectral band image data to canonical backend RGB outputs.

image_bhw: float32 (n_bands, H, W)

Returns dict:
    - rgb_linear: float32 (H, W, 3)
    - rgb_tonemapped: float32 (H, W, 3)
)doc")
                .def("rasterize_tri_flux_uv", &PyRayTracer::rasterize_tri_flux_uv,
                     py::arg("tri_flux"),
                     py::arg("tri_indices"),
                     py::arg("tri_uv"),
                     py::arg("atlas_h"),
                     py::arg("atlas_w"),
                     py::arg("gain") = 1.0,
                     py::arg("hdr_white_percentile") = 100.0,
R"doc(
Rasterize per-triangle spectral flux into a UV atlas in the C++ backend.

tri_flux:    float32 (n_tri, n_bands)
tri_indices: int32   (n_selected,)
tri_uv:      float32 (n_selected, 3, 2), per-triangle UV coordinates in [0, 1]

Returns dict:
    - spectral_atlas: float32 (n_bands, atlas_h, atlas_w)
    - rgb_linear: float32 (atlas_h, atlas_w, 3)
    - rgb_tonemapped: float32 (atlas_h, atlas_w, 3)
)doc")
                .def("compose_lens_bench_views", &PyRayTracer::compose_lens_bench_views,
                     py::arg("tri_flux"),
                     py::arg("tri_centroids"),
                     py::arg("view_h"),
                     py::arg("view_w"),
                     py::arg("x_min"),
                     py::arg("x_max"),
                     py::arg("view_radius"),
                     py::arg("field_nx"),
                     py::arg("field_ny"),
                     py::arg("field_nz"),
                     py::arg("field_gain") = 2.5,
                     py::arg("surface_gain") = 1.0,
                     py::arg("hdr_white_percentile") = 99.8,
R"doc(
Compose thick-lens top/side RGB views entirely in C++ backend code.

Returns a tuple: (top_rgb, side_rgb), each float32 (view_h, view_w, 3).
)doc")
        .def("trace_with_frequency_sidecar", &PyRayTracer::trace_with_frequency_sidecar,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("n_rays")        = 256,
             py::arg("max_bounces")   = 8,
             py::arg("min_amplitude") = 0.005,
             py::arg("seed")          = 42,
             py::arg("out_cap")       = -1,
R"doc(
Trace rays and return both segment records and C++-computed frequency sidecar.

Returns a dict with:
  'segs'         : float32 (N_segs, 12)
  'freq_sidecar' : dict with
      'band_id'       int32  (N_segs,)
      'frequency_hz'  float64(N_segs,)
      'wavelength_nm' float64(N_segs,)
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
    .def("integrate_image_into_packed", &PyRayTracer::integrate_image_into_packed,
         py::arg("src_pos"),
         py::arg("src_dir"),
         py::arg("src_directivity"),
         py::arg("src_n_rays"),
         py::arg("cam_pos"),
         py::arg("cam_fwd"),
         py::arg("cam_up"),
         py::arg("out_image"),
         py::arg("fov_rad")       = 1.0,
         py::arg("max_bounces")   = 12,
         py::arg("min_amplitude") = 0.001,
         py::arg("seed")          = 42,
         R"doc(
Accumulate camera energy with per-source packed ray quotas.

src_n_rays is an int32 vector (n_sources,) controlling how many rays each
source emits in this dispatch. This keeps adaptive allocation in one hot C++ loop.
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
                .def("set_camera_visibility", &PyRayTracer::set_camera_visibility,
                         py::arg("camera_vis_mode") = RT_CAM_VIS_AS_IS,
                         py::arg("transparent_mode") = RT_CAM_TRANSPARENCY_BLOCK,
                         py::arg("depth_cull_enabled") = false,
                         py::arg("depth_cull_m") = 0.0,
                         R"doc(
Configure camera-visibility wrappers used by image accumulation APIs.

camera_vis_mode:
    RT_CAM_VIS_AS_IS       -> no camera LOS culling (legacy behaviour)
    RT_CAM_VIS_DIRECT_HIT  -> one-shot cam->hit occlusion check
    RT_CAM_VIS_FULL_MARCH  -> iterative march through transparent media

transparent_mode:
    RT_CAM_TRANSPARENCY_BLOCK -> transparent triangles still block LOS
    RT_CAM_TRANSPARENCY_XRAY  -> transmissive triangles are skipped in LOS
)doc")
                .def("set_sensor_film_ssbo", &PyRayTracer::set_sensor_film_ssbo,
                         py::arg("sensor_chunk"),
                         py::arg("film_chunk"),
                         py::arg("active_slots") = py::none(),
                         R"doc(
Upload sensor/film tensor chunks and active slot pairs into tracer-owned C++ memory.

sensor_chunk : float32 (rows, stride)  — SensorFilmDatabase sensor tensor
film_chunk   : float32 (rows, stride)  — SensorFilmDatabase film tensor
active_slots : int32 (n_slots, 2) or sequence of (sensor_id, film_id)

The tracer deep-copies all inputs; caller buffers can be discarded after return.
)doc")
                .def("set_surface_chunks", &PyRayTracer::set_surface_chunks,
                         py::arg("pbr"),
                         py::arg("enamel"),
                         py::arg("tex_stack"),
                         R"doc(
Upload PBR, enamel, and texture-stack material chunks and rebuild the
RtMaterialSurfaceCache for GGX / thin-film BDPT lobes.

pbr       : float32 (N, 16)  — PBRBaseRecord per material (albedo, roughness,
            metallic, transmission, ior, opacity, emission_rgb)
enamel    : float32 (N,  8)  — EnamelRecord per material (thickness_nm, ior_real,
            ior_imag, roughness, tint_rgb)
tex_stack : float32 (N, 16)  — TextureStackRecord per material; [11] = profile_id
            (0=standard, 1=emissive, 2=SSS, 3=frosted scatter)

All three arrays must have N equal to mat_n_mats used at construction time.
The tracer deep-copies all inputs.  Call after construction and after any
material-database change.
)doc")
                .def("get_camera_visibility", &PyRayTracer::get_camera_visibility,
                         "Return current camera visibility policy as a dict.")
                .def("enable_field_capture_regular", &PyRayTracer::enable_field_capture_regular,
                         py::arg("nx"),
                         py::arg("ny"),
                         py::arg("nz"),
                         py::arg("bmin"),
                         py::arg("bmax"),
                         py::arg("capture_strikes") = true,
                         py::arg("max_strikes") = 0,
                         py::arg("clear_existing") = true,
                         "Bind a regular FieldGrid capture target for full complex spectral accumulation.")
                .def("enable_field_capture_kdtree", &PyRayTracer::enable_field_capture_kdtree,
                         py::arg("nodes"),
                         py::arg("capture_strikes") = true,
                         py::arg("max_strikes") = 0,
                         py::arg("clear_existing") = true,
                         "Bind a KdTree FieldGrid capture target for full complex spectral accumulation.")
                .def("clear_field_capture", &PyRayTracer::clear_field_capture,
                         py::arg("clear_grid") = true,
                         py::arg("clear_strikes") = true)
                .def("allocation_table", &PyRayTracer::allocation_table)
                .def("set_profile_pulse", &PyRayTracer::set_profile_pulse,
                         py::arg("enabled") = true,
                         py::arg("period_s") = 2.0)
                .def("get_field_capture_meta", &PyRayTracer::get_field_capture_meta)
                .def("get_field_capture_grid_reim", &PyRayTracer::get_field_capture_grid_reim)
                .def("get_field_capture_strikes", &PyRayTracer::get_field_capture_strikes)
                .def("accumulate_endpoint_records_to_field_capture", &PyRayTracer::accumulate_endpoint_records_to_field_capture,
                         py::arg("records"),
                         py::arg("sensor_group_id"),
                         py::arg("include_sensor_group") = true,
                         py::arg("include_non_sensor_groups") = true,
R"doc(
Deposit BDPT EndpointRecord amplitudes into the currently bound field-capture grid.

records: float32 (N, 16), endpoint records retained by the live ray pipeline.
Returns dict with:
    - written_records: number of endpoint records injected into field grid.
    - written_power: accumulated spectral power sum (|amp|^2) for injected rows.
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
)doc")
                .def("trace_surface_with_frequency_sidecar",
                         &PyRayTracer::trace_surface_with_frequency_sidecar,
                         py::arg("src_pos"),
                         py::arg("src_dir"),
                         py::arg("src_directivity"),
                         py::arg("n_rays")         = 256,
                         py::arg("max_bounces")    = 8,
                         py::arg("min_amplitude")  = 0.005,
                         py::arg("seed")           = 42,
                         py::arg("out_cap")        = -1,
R"doc(
Trace rays with surface accumulation and include C++-computed frequency sidecar.

Returns a dict with:
    'segs'         : float32 (N_segs, 12)
    'direct'       : float32 (n_tri, n_bands)
    'indirect'     : float32 (n_tri, n_bands)
    'freq_sidecar' : dict with per-segment
            'band_id', 'frequency_hz', 'wavelength_nm'
)doc")
        .def("trace_pipeline", &PyRayTracer::trace_pipeline,
             py::arg("origins"),
             py::arg("directions"),
             py::arg("amplitudes")     = py::none(),
             py::arg("src_ids")        = py::none(),
             py::arg("tags")           = py::none(),
             py::arg("max_bounces")    = 8,
             py::arg("min_amplitude")  = 1e-6,
             py::arg("seed")           = 42,
             py::arg("max_children")   = 2,
             py::arg("out_cap")        = 4'000'000,
R"doc(
Run the 4-stage threaded ray transport pipeline.

origins    : float64 (N, 3) — ray start positions
directions : float64 (N, 3) — ray directions (normalised internally)
amplitudes : complex128 (N, n_bands) — initial amplitude per band; None = unit
src_ids    : int32 (N,) — source tag per ray; None = sequential 0..N-1
tags       : uint64 (N,) — free slot carried through all stages; None = 0
max_bounces, min_amplitude, seed, max_children : pipeline settings
out_cap    : max segment records to accumulate

Returns dict with:
  'segs' : float32 (M, 12) — one row per (intersection, band)
           [0..2] seg_start  [3..5] hit_pos  [6] src_id  [7] bounce
           [8] freq_band  [9] |amplitude|  [10] phase  [11] path_at_seg_start
)doc")
        /* ── Persistent-machine async API ─────────────────────────────── */
        .def("submit_rays", &PyRayTracer::submit_rays,
             py::arg("origins"),
             py::arg("directions"),
             py::arg("amplitudes")      = py::none(),
             py::arg("src_ids")         = py::none(),
             py::arg("tags")            = py::none(),
             py::arg("color_flags")     = py::none(),
             py::arg("max_bounces")     = 8,
             py::arg("min_amplitude")   = 1e-6,
             py::arg("max_children")    = 2,
             py::arg("seed")            = 42,
             py::arg("use_gpu_compute") = false,
             py::arg("gpu_all_stages")  = false,
             py::arg("shader_dir")      = "",
R"doc(Non-blocking submit: push ray intents into the persistent pipeline.
Returns immediately; the pipeline processes them concurrently.
Call drain_records() to collect output records.)doc")
        .def("submit_emissive_triangles", &PyRayTracer::submit_emissive_triangles,
             py::arg("tri_ids"),
             py::arg("rays_per_tri"),
             py::arg("exposure_weight") = 1.0,
             py::arg("emitter_amp_gain") = 1.0,
             py::arg("max_bounces") = 8,
             py::arg("min_amplitude") = 1e-6,
             py::arg("max_children") = 2,
             py::arg("seed") = 42,
             py::arg("use_gpu_compute") = false,
             py::arg("gpu_all_stages") = false,
             py::arg("shader_dir") = "",
             py::arg("interaction_target_x") = 0.0,
             py::arg("interaction_target_y") = 0.0,
             py::arg("interaction_target_z") = 0.0,
             py::arg("interaction_target_r") = 0.0,
R"doc(Non-blocking native forward-light submit.
Builds emissive UV domains from the supplied triangles, fills each domain with
a complete Mortonized UV/angle ray budget, creates RayIntents from mat_buf band emission,
and submits them directly to the persistent pipeline.  The rays_per_tri argument
is kept for ABI compatibility but is interpreted as the per-emissive-domain UV
budget.
The optional
interaction target is a world-space sphere; sampled emitter rays that cannot
intersect it are not launched.)doc")
        .def("ensure_pipeline", &PyRayTracer::ensure_pipeline,
             py::arg("max_children") = 2,
             py::arg("seed") = 42,
             py::arg("min_amplitude") = 1e-6,
             py::arg("use_gpu_compute") = false,
             py::arg("gpu_all_stages") = false,
             py::arg("shader_dir") = "",
R"doc(Force creation of the persistent ray pipeline without submitting rays.
Use this on the display thread when the pipeline needs to share with the
currently-bound OpenGL display context.)doc")
        .def("drain_records", &PyRayTracer::drain_records,
             py::arg("max_n") = 50000,
R"doc(Non-blocking drain: pop up to max_n completed records from the output queue.
Returns a dict of numpy arrays (empty arrays if nothing is ready yet).
kind: 0=STRIKE 1=TERMINAL 2=MISS 3=FIELD)doc")
        .def("in_flight_count", &PyRayTracer::in_flight_count,
R"doc(Return the number of ray paths currently live in the persistent pipeline.
Zero means all previously submitted rays have completed.)doc")
        .def("pipeline_stats", &PyRayTracer::pipeline_stats,
R"doc(Snapshot of pipeline throughput/batch-size/queue-depth for all four stages.
Returns dict with keys t1, t2, t3, t4, t5 (each a dict with throughput, processed,
batch_size, queue_depth) plus output_queue_depth and in_flight.)doc")
        .def("report_display_frame_time", &PyRayTracer::report_display_frame_time,
             py::arg("frame_ms"),
             py::arg("target_ms") = 16.667,
R"doc(Feed display frame timing into the GPU producer governor.
Call once per interactive frame; spikes shrink GPU batch sizes and slow UV
updates, while stable frames cautiously restore throughput.)doc")
        .def("drain_records_slim", &PyRayTracer::drain_records_slim,
             py::arg("max_n") = 50000,
R"doc(Non-blocking slim drain: returns only the 7 arrays needed for voxel accumulation.
~5x less allocation than drain_records(). Keys: kind, bounce, seg_start, pos,
color_flag, hit_tri, hit_group_id, mat_idx, sensor_origin_y/z, amp_re, amp_im.)doc")
        .def("drain_records_raw", &PyRayTracer::drain_records_raw,
             py::arg("max_n") = 50000,
R"doc(Non-blocking raw drain: returns uint8 bytes with shape (N, sizeof(RayRecord)).
Use a ctypes/NumPy structured view on the Python side instead of dict-of-arrays.)doc")
        .def("drain_records_display", &PyRayTracer::drain_records_display,
             py::arg("max_n"),
             py::arg("image_res"),
             py::arg("view_radius"),
             py::arg("sensor_half_w"),
             py::arg("sensor_half_h"),
             py::arg("tri_kind"),
             py::arg("rgb_weights"),
R"doc(Non-blocking display drain: consumes RayRecord batches and returns only small
forward/reverse/camera RGB accumulation deltas plus counters. This avoids
materializing million-row Python record dictionaries for the live display path.)doc")
        .def("set_min_amplitude", &PyRayTracer::set_min_amplitude,
             py::arg("eps"),
R"doc(Set the pipeline-wide amplitude floor.  Rays with per-ray min_amplitude below this
are silently raised.  Also rebuilds material epsilon flags so T3 can fast-path
fully-absorptive surfaces.)doc")
        .def("precompute_epsilon_material_flags", &PyRayTracer::precompute_epsilon_material_flags,
R"doc(Rescan material reflectances and flag those with max|refl| < min_amplitude.
T3 uses these flags to skip direction sampling for guaranteed-dying rays.
Call after the pipeline is first created (i.e. after first submit_rays).)doc")
        .def("set_max_intent_queue", &PyRayTracer::set_max_intent_queue,
             py::arg("n"),
R"doc(Set bounded capacity of the intent queue (0 = unbounded).
When > 0, submit_rays blocks until queue is below this limit (backpressure).
Must be called before the first submit_rays.)doc")
        .def("set_intent_shuffle", &PyRayTracer::set_intent_shuffle,
             py::arg("frac"),
R"doc(Shuffle lever for the T1 intent queue.  0.0 = strict FIFO (default);
1.0 = fully random window — every T1 pop draws from a uniformly random position
in Q_intent so deep-bounce children are interleaved with fresh primary rays,
giving diffusion across both depth and breadth.  Can be changed at any time.\n
Typical values: 0.0 (off), 0.25 (mild), 0.75 (strong).)doc")
        .def("configure_sensor_image",
             &PyRayTracer::configure_sensor_image,
             py::arg("plate_x"), py::arg("plate_half_w"), py::arg("plate_half_h"),
             py::arg("res"), py::arg("bdpt_eps"),
             py::arg("target_x") = 0.0f, py::arg("target_r") = 0.0f,
             py::arg("target_y") = 0.0f, py::arg("target_z") = 0.0f,
             py::arg("target_mode") = 0,
R"doc(Configure sensor-plane image accumulator.
plate_x: world X of the sensor rectangle centre; plate_half_w: half-width (Y axis, metres);
plate_half_h: half-height (Z axis, metres);
res: pixel grid side (res×res); bdpt_eps: YZ proximity threshold for BDPT snap.
target_x/target_r: camera projection target, normally the assembly exit pupil.
target_y/target_z: transverse centre of that target disk.
target_mode: 0 launches toward the target disk; 1 launches away from a virtual target.
Resets accumulator.  Call before submitting rays.)doc")
        .def("configure_sensor_pose",
             &PyRayTracer::configure_sensor_pose,
             py::arg("sensor_center"), py::arg("sensor_right"), py::arg("sensor_up"),
             py::arg("aperture_center"), py::arg("aperture_right"), py::arg("aperture_up"),
R"doc(Configure a physical film-plane pose and fixed aperture plane.
Sensor UV and recursive mip coordinates remain local to sensor_right/sensor_up;
changing the pose changes world-space ray origins and focusing geometry.)doc")
        .def("configure_sensor_mipmap",
             &PyRayTracer::configure_sensor_mipmap,
             py::arg("max_nodes") = 65536u,
             py::arg("maximum_depth") = 8u,
             py::arg("samples_per_epoch") = 9u,
R"doc(Enable sparse recursive 3x3 sensor-mipmap storage before pipeline creation.
Allocates a bounded GPU node pool; scheduling explicitly distinguishes sampling
from 3x3 subdivision, and retained leaves continue independent epochs.)doc")
        .def("configure_sensor_priority_network",
             &PyRayTracer::configure_sensor_priority_network,
             py::arg("parameters"),
R"doc(Install the fixed conv3x3-4x8-ReLU-conv1x1-softplus sensor work-value
network before pipeline creation. Inference runs over resident GPU sensor data.)doc")
        .def("configure_sensor_requested_priority_map",
             &PyRayTracer::configure_sensor_requested_priority_map,
             py::arg("values"),
R"doc(Install a camera/network next-scan UV work request before pipeline creation.
The map remains separate from inferred priority and is consumed by the GPU node scorer.)doc")
        .def("configure_sensor_delta_restore",
             &PyRayTracer::configure_sensor_delta_restore,
             py::arg("rgb"), py::arg("weight"), py::arg("dirty_sites"),
R"doc(Restore a prior square sensor sum/weight and zero only the arbitrary
flat site indices selected for delta regrowth.)doc")
        .def("get_sensor_image",
             &PyRayTracer::get_sensor_image,
R"doc(Return current sensor image as float32 ndarray of shape (res, res, 3).
R=forward plate hits, G=backward emissive+provisional near-miss, B=exact BDPT snap.
Values are log-tone-mapped to [0, 1]. Thread-safe.)doc")
    .def("get_sensor_image_linear",
         &PyRayTracer::get_sensor_image_linear,
R"doc(Return linear sensor RGB as float32 (res, res, 3). Adaptive recursive
exposures are divided by continuous per-bin exposure weights; legacy uniform
exposures retain raw-sum semantics. No display curve is applied.)doc")
        .def("get_sensor_exposure_weight",
             &PyRayTracer::get_sensor_exposure_weight,
R"doc(Return continuous adaptive exposure weights as float32 (res, res).
Orientation matches get_sensor_image_linear(); zero means unexposed.)doc")
        .def("get_sensor_image_sum_linear",
             &PyRayTracer::get_sensor_image_sum_linear,
R"doc(Return retained pre-normalization RGB sums as float32 (res, res, 3).
Pair with get_sensor_exposure_weight() to audit adaptive reconstruction.)doc")
        .def("get_sensor_learned_priority_map",
             &PyRayTracer::get_sensor_learned_priority_map,
R"doc(Return the most recently inferred neural work-value map as float32.)doc")
        .def("get_sensor_epoch_count",
             &PyRayTracer::get_sensor_epoch_count,
R"doc(Return completed primary film-stratum counts as uint32 (res, res).
Orientation matches get_sensor_image_linear(). Counts normalize unequal
regional exposure and are independent of radiance.)doc")
        .def("debug_sensor_mip_subdivide",
             &PyRayTracer::debug_sensor_mip_subdivide,
             py::arg("completed_ids"),
R"doc(Validation-only GPU subdivision bridge. Returns sparse node bounds,
metadata, and control counters after unconditionally splitting completed
nonterminal nodes. Production scheduling remains GPU-resident.)doc")
        .def("debug_sensor_mip_samples",
             &PyRayTracer::debug_sensor_mip_samples,
             py::arg("node_id"), py::arg("sample_begin"), py::arg("sample_count"),
             py::arg("seed") = 0u,
R"doc(Validation-only GPU node sampler. Returns continuous global/local UV,
unit estimator weights, and exact node/level/sequence lineage. A workgroup owns
one node and cannot write samples into a neighbouring node.)doc")
        .def("debug_sensor_mip_select",
             &PyRayTracer::debug_sensor_mip_select,
             py::arg("top_k"), py::arg("seed") = 0u,
             py::arg("targeted_fraction") = 1.0f,
R"doc(Validation-only mixed GPU frontier selection. targeted_fraction reserves
the remainder for least-exposed, seed-rotated Morton coverage. Work flag bit 0
identifies coverage records; the two lanes are deduplicated.)doc")
        .def("debug_sensor_mip_score",
             &PyRayTracer::debug_sensor_mip_score,
             py::arg("features"),
             py::arg("weights"),
R"doc(Validation-only GPU scorer boundary. Feature columns are uncertainty,
ambiguity, learned work value, and explicit request strength. This pass writes
only node priority and has no access to subdivision state transitions.)doc")
        .def("debug_sensor_mip_accumulate",
             &PyRayTracer::debug_sensor_mip_accumulate,
             py::arg("spectra"),
R"doc(Validation-only GPU spectral accumulator. Consumes the most recently
generated GPU sample lineage and updates per-node direct sums, squared sums,
weights, and sample counts without modifying parent evidence.)doc")
        .def("debug_sensor_mip_rollup",
             &PyRayTracer::debug_sensor_mip_rollup,
R"doc(Validation-only reverse-depth GPU rollup. A parent receives a separate
descendant estimate only when all nine child estimates are valid; its direct
coarse evidence remains stored independently.)doc")
        .def("submit_sensor_mip_epoch",
             &PyRayTracer::submit_sensor_mip_epoch,
             py::arg("top_k") = 8u, py::arg("seed") = 0u,
             py::arg("max_bounces") = 8u,
             py::arg("min_amplitude") = 1.0e-12f,
             py::arg("exposure_weight") = 1.0f,
             py::arg("targeted_fraction") = 1.0f,
R"doc(Run one production recursive sensor epoch entirely on the GPU: compact
the frontier, select a deduplicated mixture of targeted and broad-coverage
nodes, generate continuous camera rays, trace all
bounces, combine terminal spectra per primary sample, update direct moments,
subdivide every completed nonterminal node, and roll descendants upward.)doc")
        .def("get_priority_map",
             &PyRayTracer::get_priority_map,
R"doc(Return the sugar-auxin work-priority map as float32 ndarray of shape (res, res).
Values >= 1.0.  Baseline = 1.0; elevated pixels had recent BDPT convergence and
receive more backward-ray budget on the next submit call.  Thread-safe.)doc")
        .def("drain_refined_hits",
             &PyRayTracer::drain_refined_hits,
             py::arg("max_n") = 256,
R"doc(Dequeue up to max_n records from Q_refined and return them as a float32
ndarray of shape (n, 58) using the flat RefinedHit layout defined in
ray_material.comp.glsl.  Returns shape (0,) when the queue is empty.

Used by the GPU T3 bridge (GpuRayMaterialStage) to intercept the material/Fresnel
stage and dispatch it on the GPU instead of the C++ worker threads.)doc")
        .def("submit_intents_flat",
             &PyRayTracer::submit_intents_flat,
             py::arg("buf"),
R"doc(Feed GPU-processed child RayIntents back into the pipeline.
buf: float32 ndarray of shape (n, 52) using the flat RayIntent layout defined
in ray_material.comp.glsl.  Each row is decoded into a RayIntent and pushed
to the T1 intent queue.

Used with drain_refined_hits() to form a round-trip GPU T3 dispatch loop.)doc")
        .def("get_bdpt_stats",
             &PyRayTracer::get_bdpt_stats,
R"doc(Return a dict of live BDPT convergence diagnostics (lock-free snapshot):
  nearest_dist_m     (float)  — sqrt(min YZ d²) in metres of best fwd/rev pair seen.
                                -1.0 if no pairs observed yet.
  best_collinearity  (float)  — |cos θ| between incoming directions at best pair [0,1].
  exact_snaps        (int)    — cumulative exact-match (ch2) pixel accumulations.
  near_miss_count    (int)    — cumulative near-miss (ch3) pixel accumulations.)doc")
        .def("get_bdpt_latch_state",
             &PyRayTracer::get_bdpt_latch_state,
R"doc(Return a dict of the four BDPT pipeline latch counters (lock-free relaxed reads):
  flash_dispatched   (int)   — number of trace_forward() calls completed.
  sensor_dispatched  (int)   — number of submit_sensor_sweep() calls completed.
  t5_fired           (int)   — number of T5 connection passes completed.
  connection_running (bool)  — True while a T5 thread is actively running.
T5 fires when min(flash_dispatched, sensor_dispatched) > t5_fired.)doc")
        .def("drain_bdpt_vertices",
             &PyRayTracer::drain_bdpt_vertices,
             py::arg("max_n") = 100000,
R"doc(Non-blocking drain from the BDPT vertex side-queue.
Returns uint8 ndarray of shape (n, 112).  Convert to structured array via:
  import numpy as np
  raw = tracer.drain_bdpt_vertices()
  verts = np.frombuffer(raw.tobytes(), dtype=BDPT_VERTEX_DTYPE)
Returns shape (0, 112) when the queue is empty.)doc")
        .def("drain_bdpt_spectral",
             &PyRayTracer::drain_bdpt_spectral,
             py::arg("max_n") = 100000,
R"doc(Non-blocking drain from the BDPT spectral-weight side-queue.
Returns uint8 ndarray of shape (n, 32).  One row per (subpath, vertex, band).
Returns shape (0, 32) when the queue is empty.)doc")
        .def("drain_bdpt_pdfs",
             &PyRayTracer::drain_bdpt_pdfs,
             py::arg("max_n") = 100000,
R"doc(Non-blocking drain from the BDPT PDF side-queue.
Returns uint8 ndarray of shape (n, 48).  One row per scatter event.
Returns shape (0, 48) when the queue is empty.)doc")
        .def("drain_bdpt_optical",
             &PyRayTracer::drain_bdpt_optical,
             py::arg("max_n") = 100000,
R"doc(Non-blocking drain from the BDPT optical-event side-queue.
Returns uint8 ndarray of shape (n, 112).  One row per lens/interface event.
Returns shape (0, 112) when the queue is empty.)doc")
        .def("drain_bdpt_connections",
             &PyRayTracer::drain_bdpt_connections,
             py::arg("max_n") = 100000,
R"doc(Non-blocking drain from the BDPT connection side-queue.
Returns uint8 ndarray of shape (n, 80).  One row per attempted MIS connection.
Returns shape (0, 80) when the queue is empty.)doc")
        .def("get_bdpt_overflow",
             &PyRayTracer::get_bdpt_overflow,
R"doc(Cumulative overflow counts for BDPT side queues.
Returns dict with keys: vertices, spectral, pdfs, optical, connections (all int).
A non-zero value means records were silently dropped when the queue was full.
Check this after a run to know whether to increase bdpt_max_* in the pipeline config.)doc")
        .def("run_bdpt_connection",
             &PyRayTracer::run_bdpt_connection,
R"doc(Run BDPT connection pass with balance-heuristic MIS.
Drains Q_bdpt_vertices, Q_bdpt_spectral, Q_bdpt_pdfs accumulated since the
last call, evaluates valid sensor/light vertex-pair strategies, computes
geometry terms and MIS weights, emits BdptConnectionRecord diagnostics, and
accumulates visible contributions into sensor channel 2.
Call once per sensor sweep after the pipeline is idle.)doc")
        .def("submit_sensor_sweep",
             &PyRayTracer::submit_sensor_sweep,
             py::arg("max_bounces") = 8,
             py::arg("min_amplitude") = 1e-6,
             py::arg("max_rays") = 0,
             py::arg("pix_offset") = 0,
             py::arg("max_children") = 2,
             py::arg("aperture_samples") = 1,
             py::arg("seed") = 0,
             py::arg("shutter_mode") = 0,
             py::arg("shutter_open") = 1.0,
             py::arg("shutter_center_u") = 0.5,
             py::arg("shutter_center_v") = 0.5,
             py::arg("shutter_softness") = 0.0,
             py::arg("exposure_weight") = 1.0,
R"doc(Submit a native BDPT sensor-frame sweep into the persistent pipeline.
pix_offset: first tiled-Morton schedule index (0 = full grid from start).
max_rays:   cap on tiled-Morton pixels (0 = all from pix_offset onward).
aperture_samples: deterministic pupil placements interleaved per film sample.
seed: 0 = all rays aim at aperture center (backward compat);
      N > 0 = Fibonacci-spiral aperture sample for batch N (golden-angle quasi-random disk).
shutter_mode: 0=open, 1=closed, 2=iris, 3=sliding_x, 4=sliding_y.)doc")
        .def("signal_flash_dispatched",
             &PyRayTracer::signal_flash_dispatched,
R"doc(Signal that all emissive-triangle (flash) rays for the current exposure
substage have been submitted.  T5 fires once both signal_flash_dispatched and
signal_sensor_dispatched have been called for the same substage.)doc")
        .def("signal_sensor_dispatched",
             &PyRayTracer::signal_sensor_dispatched,
R"doc(Signal that all sensor-sweep rays for the current exposure substage have
been submitted.  Mirrors signal_flash_dispatched.)doc")
        .def("join_t5",
             &PyRayTracer::join_t5,
R"doc(Block (releasing the GIL) until the T5 worker thread finishes.
Call after signal_sensor_dispatched + signal_flash_dispatched to guarantee
sensor_accum ch2 (BDPT radiance) is fully written before get_sensor_image().)doc")
        .def("begin_sensor_batching",
             &PyRayTracer::begin_sensor_batching,
R"doc(Enter sensor-batching mode for the current exposure.
In batching mode run_bdpt_connection / the T5 worker stashes light-stream
records on the first pass and reuses them for all subsequent sensor batches,
so flash rays only need to be fired once.  Clears any prior stash.)doc")
        .def("end_sensor_batching",
             &PyRayTracer::end_sensor_batching,
R"doc(Exit sensor-batching mode and clear the light-record stash.
Call once after the final join_t5() for the last sensor batch.)doc")
        .def("set_t5_min_geom",
             &PyRayTracer::set_t5_min_geom,
             py::arg("threshold"),
R"doc(Set minimum geometry term to evaluate a connection (default 1e-8).
Replaces the legacy 1e-20 floor.)doc")
        .def("stop_pipeline",
             &PyRayTracer::stop_pipeline,
R"doc(Explicitly shut down the C++ pipeline — joins the GPU dispatch thread and
all CPU workers, then releases the WGL context.  Safe to call before bench
destruction so the old context is gone before a new shared context is created.)doc")
        .def("set_t5_light_batch_size",
             &PyRayTracer::set_t5_light_batch_size,
             py::arg("n"),
R"doc(Set the number of light vertices processed per T5 GPU dispatch (default 20480).
Larger values reduce dispatch overhead and improve GPU utilisation; derive from
--t5-vram-mb as n = vram_mb * 1024^2 / 48 (12 floats × 4 bytes per light vert).
Safe to call before or after pipeline creation.)doc")
        .def("set_t5_cam_batch_size",
             &PyRayTracer::set_t5_cam_batch_size,
             py::arg("n"),
R"doc(Set the number of camera vertices processed per T5 GPU dispatch (default 8192).
Splitting the X-dispatch dimension prevents Windows TDR watchdog kills when
n_cam is large. Smaller values reduce per-dispatch GPU time at the cost of
more dispatch overhead. 0 restores the default (8192).
Safe to call before or after pipeline creation.)doc")
        .def("set_t5_sensor_tile_size",
             &PyRayTracer::set_t5_sensor_tile_size,
             py::arg("n"),
R"doc(Set the sensor tile side length for the T5 GPU connection pass (default 512).
The sensor grid is partitioned into n×n tiles; each tile is solved with a
compact pixel accum buffer (3×n² instead of 3×res²), avoiding VRAM exhaustion
at large resolutions.  Smaller tiles reduce peak VRAM at the cost of more tile
overhead; 0 restores the default (128).  Safe to call before or after pipeline creation.)doc")
        .def("set_t5_pair_budget",
             &PyRayTracer::set_t5_pair_budget,
             py::arg("n"),
    R"doc(Set the maximum score-sorted T5 camera-light pairs to process in one native pass.
    0 drains all scored units. Nonzero values process a high-score active prefix and
    leave the remainder explicitly logged as deferred.)doc")
        .def("set_t5_backlog_policy",
             &PyRayTracer::set_t5_backlog_policy,
             py::arg("share_base") = 0.15f,
             py::arg("share_max") = 0.60f,
             py::arg("max_snapshots") = 3,
             py::arg("max_bytes") = (uint64_t)(6ull << 30),
R"doc(Configure the T5 deferred-pair backlog.  Units deferred by the pair budget are
snapshotted (packed vertices + score-ordered unit list) and re-worked by later
passes highest-score-first.  Each pass spends share = min(share_max,
share_base * backlog_age_in_passes) of its pair budget on the backlog, so back
work takes proportionally more time the longer it has been accumulating.
max_snapshots / max_bytes bound the host memory retained; least-valuable
snapshots are evicted (loudly) beyond them.  share_base=0 disables.)doc")
        .def("set_bdpt_record_float_budget",
             &PyRayTracer::set_bdpt_record_float_budget,
             py::arg("n_floats"),
R"doc(Total float budget for the GPU BDPT record SSBO (vertex+spectral+pdf+optical
sections).  Record caps auto-grow from observed overflow high-water marks and
are rebalanced by demand to fit this budget (hard-clamped to INT32_MAX floats
for shader indexing).  1 float = 4 bytes of VRAM; default 1.1e9 ≈ 4.4 GiB.)doc")
        .def("set_t5_profile",
             &PyRayTracer::set_t5_profile,
             py::arg("v"),
R"doc(When True, run staged one-pair T5 shader profiling before the first T5 dispatch
of each pass and print per-stage fence timing / 1s timeout diagnostics.)doc")
        .def("set_vcm",
             &PyRayTracer::set_vcm,
             py::arg("enabled") = true,
             py::arg("merge_radius_m") = 0.002f,
             py::arg("radius_alpha") = 0.7f,
R"doc(Configure spectral vertex connection and merging. The merge radius is in
world metres. Delta surfaces remain non-connectable; merging occurs only when
a light subpath has completed its specular chain and lands on a compatible
non-delta receiver. radius_alpha controls progressive radius reduction.)doc")
        .def("set_max_children",
             &PyRayTracer::set_max_children,
             py::arg("count"),
R"doc(Set the live per-hit child cap for subsequent native batches. This avoids
initialization order coupling with field/display setup.)doc")
        .def("set_force_cpu_t5",
             &PyRayTracer::set_force_cpu_t5,
             py::arg("v"),
R"doc(When True, skip the GPU T5 full-connect shader and fall back to CPU run_t5_allpairs().
Use --no-gpu-t5 to pass this at startup for A/B comparison against the GPU path.)doc")
        .def("set_flash_modifier",
             &PyRayTracer::set_flash_modifier,
             py::arg("type_int"),
             py::arg("param0") = 0.0f,
             py::arg("param1") = 0.0f,
R"doc(Set the flash light modifier applied inside submit_emissive_triangles.
type_int: 0=NONE, 1=SNOOT (default), 2=GRID, 3=SCRIM.
SNOOT  : conic disk test — rays must reach the exit disk at the interaction target.
         No extra params; target_r controls exit-disk radius.
GRID   : egg-crate square cells.  param0=cell_diameter_mm (default 5),
         param1=tube_depth_mm (default 25).
SCRIM  : stochastic attenuator.  param0=transmittance in [0,1] (default 0.5).)doc")
        .def("add_scale_context", &PyRayTracer::add_scale_context,
             py::arg("pos"),
             py::arg("radius"),
             py::arg("scale_type")  = 0,
             py::arg("dt_m")        = 1e-6,
             py::arg("n_substeps")  = 1000,
             py::arg("n_real")      = 1.5,
             py::arg("n_imag")      = 0.0,
             py::arg("context_kind") = 0,
             py::arg("payload")     = py::none(),
             R"doc(
Register a scale-context sphere in the scene.

pos          : (3,) float64 world-space centre (metres)
radius       : trigger radius (metres)
scale_type   : 0 = RT_SCALE_GEOMETRIC (coarse), 1 = RT_SCALE_WAVE (fine/wave)
dt_m         : wave sub-step size (metres); only used when scale_type=1
n_substeps   : safety cap on sub-step count per context crossing
n_real       : real part of medium refractive index inside sphere
n_imag       : imaginary part (extinction coefficient; >0 = absorbing)
context_kind : SCALE_CONTEXT_KIND_* — 0=RAY (default), 1=WAVE_HELMHOLTZ,
               2=THIN_LENS_TRANSFORM, 3=THICK_LENS_WAVE, 4=SPLINE_SURFACE,
               5=NEURAL_SURFACE, 6=NEURAL_VOLUMETRIC.  Selects the
               dispatch path the integrator takes inside the region.
               Stub-passthrough for kinds 3..6 right now.
payload      : optional bytes/array carrying kind-specific parameters.
               Must be kept alive for the lifetime of the tracer.
               THIN_LENS_TRANSFORM expects float64[1] = [focal_length_m].

Returns the assigned context_id integer.
)doc")
        .def("clear_scale_contexts", &PyRayTracer::clear_scale_contexts,
             "Remove all registered scale-context spheres.")
        .def("trace_multiscale_into", &PyRayTracer::trace_multiscale_into,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("out_segs"),
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 12,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             R"doc(
Trace rays using the multiscale kernel, writing into a caller-owned buffer.

out_segs must be a float32 array of shape (capacity, 14).  The first 12
columns match the standard segment layout; columns 12 and 13 carry
context_id and scale_type (0=geometric, 1=wave).

Returns n_written (the number of segment records filled in).  Reuse the
same buffer across trickle iterations; clear with out_segs[:] = 0 when
needed.
)doc")
        .def("trace_multiscale", &PyRayTracer::trace_multiscale,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_directivity"),
             py::arg("n_rays")        = 512,
             py::arg("max_bounces")   = 12,
             py::arg("min_amplitude") = 0.001,
             py::arg("seed")          = 42,
             py::arg("out_cap")       = 65536,
             R"doc(
Trace rays using the multiscale kernel, returning a new float32 array.

Returns ndarray of shape (N, 14) where N <= out_cap.  Columns 0-11 match
the standard 12-float segment layout; columns 12-13 are context_id and
scale_type.
)doc")
                .def("trace_multiscale_with_frequency_sidecar",
                         &PyRayTracer::trace_multiscale_with_frequency_sidecar,
                         py::arg("src_pos"),
                         py::arg("src_dir"),
                         py::arg("src_directivity"),
                         py::arg("n_rays")        = 512,
                         py::arg("max_bounces")   = 12,
                         py::arg("min_amplitude") = 0.001,
                         py::arg("seed")          = 42,
                         py::arg("out_cap")       = 65536,
R"doc(
Trace rays using the multiscale kernel and include C++-computed frequency sidecar.

Returns a dict with:
    'segs'         : float32 (N, 14)
    'freq_sidecar' : dict with per-segment
            'band_id', 'frequency_hz', 'wavelength_nm'
)doc")
        .def("set_tri_ior", &PyRayTracer::set_tri_ior,
             py::arg("tri_start"),
             py::arg("n_tris"),
             py::arg("flags") = 0,
             R"doc(
Set semantic surface flags for a range of triangles.

Per-triangle physics (refl, IOR, diffusion, transmission) lives in the
MatBuf addressed by tri.mat_idx.  This entry point only adjusts non-material
flags.

tri_start : index of the first triangle in the range
n_tris    : number of triangles in the range
flags     : MAT_FLAG_APERTURE_STOP (128) | MAT_FLAG_PICKING_ONLY (256) | …

Transmission/refraction is inferred from the material MatBuf record
(transmittance and ior_real). TIR is handled automatically.

When MAT_FLAG_APERTURE_STOP is set the surface absorbs the ray.
Diffraction is handled by the near-field pipeline (coherent_accumulate +
rs_propagate / wave_bpm_step).
)doc")
                .def("set_tri_boundary_media", &PyRayTracer::set_tri_boundary_media,
                         py::arg("tri_start"),
                         py::arg("n_tris"),
                         py::arg("medium_pos_mat_idx"),
                         py::arg("medium_neg_mat_idx"),
                         R"doc(
Set directional boundary media for a range of triangles.

Each triangle stores media per side of its geometric normal:
    medium_pos_mat_idx : medium on +normal side
    medium_neg_mat_idx : medium on -normal side
Use -1 for ambient air/vacuum.

Refraction then uses crossing direction directly (front-face: + -> -,
back-face: - -> +) instead of material-equality toggles.
)doc")
        .def("project_coherent", &PyRayTracer::project_coherent,
             py::arg("segs"),
             py::arg("n_bands"),
             py::arg("sensor_w"),
             py::arg("sensor_h"),
             py::arg("sensor_z"),
             py::arg("sensor_r"),
             py::arg("out_re"),
             py::arg("out_im"),
             R"doc(
Project a multiscale segment buffer onto a coherent complex sensor image.

For every segment that crosses the plane z = sensor_z the intersection point
is mapped to a pixel and the complex amplitude is accumulated additively:
  out_re[b, py, px] += amp * cos(phase)
  out_im[b, py, px] += amp * sin(phase)

After accumulating many trickle batches, out_re**2 + out_im**2 is the
coherent diffraction-correct intensity image: Airy rings, interference
fringes, speckle, and all other wave effects emerge naturally.

segs      : float32 (N, 14) — multiscale segment buffer from trace_multiscale_into
n_bands   : number of frequency bands
sensor_w/h: sensor resolution in pixels
sensor_z  : z position of sensor plane in world space (metres)
sensor_r  : half-width of sensor (metres); pixels cover [-r, +r]
out_re/im : float32 (n_bands, sensor_h, sensor_w) — caller-allocated,
            caller-zeroed; this method ACCUMULATES into them.
)doc")
        .def("apply_aperture_mask", &PyRayTracer::apply_aperture_mask,
             py::arg("n_bands"),
             py::arg("w"),
             py::arg("h"),
             py::arg("field_r"),
             py::arg("poly_xy"),
             py::arg("re"),
             py::arg("im"),
             R"doc(
Zero field pixels outside a polygon aperture (in-place).

poly_xy : float32 (n_verts, 2) — aperture opening polygon in field coordinates
          where ±field_r maps to ±1 in pixel space.
re, im  : float32 (n_bands, h, w) — complex field, modified in-place.
          Pixels whose centre falls outside the polygon are zeroed in all bands.
)doc")
        .def("rs_propagate", &PyRayTracer::rs_propagate,
             py::arg("n_bands"),
             py::arg("w"),
             py::arg("h"),
             py::arg("dx"),
             py::arg("z_dist"),
             py::arg("wavelengths_m"),
             py::arg("in_re"),
             py::arg("in_im"),
             R"doc(
Exact Rayleigh-Sommerfeld diffraction integral (O(N² × M²) CPU).

Propagates a complex aperture field forward by z_dist in a single integral
step.  Every output pixel receives contributions from every input pixel via
the exact non-paraxial RS kernel:

    K(r) = (z / r²) × (ik − 1/r) × exp(ikr) / (2π)

Returns (out_re, out_im) as float32 arrays of shape (n_bands, h, w).

Use wave_bpm_step for multi-step volume propagation; use rs_propagate for a
single exact jump (e.g. aperture → sensor in one call).
)doc")
        .def("wave_bpm_step", &PyRayTracer::wave_bpm_step,
             py::arg("n_bands"),
             py::arg("w"),
             py::arg("h"),
             py::arg("dx"),
             py::arg("dz"),
             py::arg("wavelengths_m"),
             py::arg("re"),
             py::arg("im"),
             R"doc(
Beam Propagation Method — batchwise PDE z-stepper (in-place).

Advances the complex field U[band][y][x] by one step dz by solving the
paraxial Helmholtz PDE:

    ∂U/∂z = (i/2k) ∇_T² U

using an ADI Crank-Nicolson finite-difference scheme (unconditionally stable,
second-order in dz and dx).  The carrier phase exp(ik dz) is also applied so
both the optical path length and the transverse spreading are correct.

Call repeatedly to build a coherent 3-D near-field volume step-by-step:
each call is one z-slice of the true wave PDE solution.  Bokeh, diffraction
rings, Airy patterns, near-field evanescent tails, and all other wave effects
emerge from the field evolution — no approximations or post-processes.

re, im        : float32 (n_bands, h, w) — field modified in-place.
dx            : pixel pitch in metres (same in x and y).
dz            : propagation step in metres (positive = forward along z).
wavelengths_m : float64 (n_bands,) — wavelength per band in metres.
)doc")
        .def("spawn", &PyRayTracer::spawn,
             py::arg("src_pos"),
             py::arg("src_dir"),
             py::arg("src_dir_power"),
             py::arg("n_rays"),
             py::arg("max_bounces")   = 8,
             py::arg("min_amplitude") = 0.005,
             py::arg("seed")          = 0u,
             R"doc(
Spawn rays from sources into the scheduler's coarse geometric queue.

Rays are sampled by Fibonacci sphere with directivity weighting.  They
persist as live stateful objects until absorbed, escaped, or cleared with
clear_rays().  Calling spawn() again adds more rays without clearing.

src_pos       : float64 (n_sources, 3) — source world positions
src_dir       : float64 (n_sources, 3) — dominant emit directions
src_dir_power : float64 (n_sources,)   — directivity exponent (0 = omni)
n_rays        : rays per source
max_bounces   : kill ray after this many surface reflections
min_amplitude : kill ray when max|A| across all bands falls below this
seed          : RNG seed for directivity sampling
)doc")
        .def("step", &PyRayTracer::step,
             py::arg("seg_buf"),
             R"doc(
Advance all live rays one scheduler step.

Each context queue is ticked once, ordered coarsest-to-finest:
  geometric queue : BVH intersection jump, capped at context-sphere entry
  RT_SCALE_GEOMETRIC contexts : same, confined to context sphere
  RT_SCALE_WAVE contexts : min(dt_m, sphere boundary, surface) step with
                           wave-accurate phase and near-field spreading

Rays that enter a finer context sphere are transferred to that queue.
Rays that exit their context sphere move to the next-coarser queue.
Dead rays are removed.  Segments are written in the 14-float MS format.

seg_buf : float32 (N, 14) pre-allocated output buffer (modified in-place)
Returns : (n_written, n_live) — segments written and rays still alive
)doc")
        .def("clear_rays", &PyRayTracer::clear_rays,
             R"doc(Remove all live rays and free the internal ray pool.)doc")
        .def("live_ray_count", &PyRayTracer::live_ray_count,
             R"doc(Return the number of live rays across all context queues.)doc")
        /* ── Triangle-group registry ────────────────────────────────── */
        .def("register_tri_group",
             [](PyRayTracer& self, uint32_t role_bits, uint32_t sample_policy,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> tri_indices,
                py::object plane_origin, py::object plane_normal,
                double custom_emit_W,
                int default_mat_idx,
                py::object power_W_per_band,
            py::object parametric_surface,
                py::object sensor_camera,
                py::object uv_image) {
                 auto buf = tri_indices.request();
                 if (buf.ndim != 1)
                     throw std::runtime_error("tri_indices must be 1-D int32");
                 TriGroupDesc desc{};
                 desc.group_id     = -1;
                 desc.role_bits    = role_bits;
                 desc.sample_policy= sample_policy;
                 desc.n_tris       = (int)buf.shape[0];
                 desc.tri_indices  = static_cast<const int*>(buf.ptr);
                 if (!plane_origin.is_none()) {
                     auto po = plane_origin.cast<py::array_t<double>>();
                     auto p  = po.unchecked<1>();
                     for (int i = 0; i < 3; ++i) desc.plane_origin[i] = p(i);
                 }
                 if (!plane_normal.is_none()) {
                     auto pn = plane_normal.cast<py::array_t<double>>();
                     auto p  = pn.unchecked<1>();
                     for (int i = 0; i < 3; ++i) desc.plane_normal[i] = p(i);
                 }
                 desc.custom_emit_W   = custom_emit_W;
                 desc.default_mat_idx = default_mat_idx;

                 /* Optional per-band power curve (deep-copied inside the
                  * registrar; the float* needs to stay live only across
                  * the registrar call itself). */
                 py::array_t<float, py::array::c_style | py::array::forcecast> pw_arr;
                 if (!power_W_per_band.is_none()) {
                     pw_arr = power_W_per_band.cast<
                         py::array_t<float, py::array::c_style | py::array::forcecast>>();
                     auto pwb = pw_arr.request();
                     if (pwb.ndim != 1)
                         throw std::runtime_error("power_W_per_band must be 1-D float32");
                     desc.n_power_bands    = (int)pwb.shape[0];
                     desc.power_W_per_band = static_cast<const float*>(pwb.ptr);
                 }

                 /* Optional parametric surface payload.
                  * Expected dict: {"kind": int, "coeffs": float64[N]} where
                  * N depends on kind:
                  *   POLY_BARY   -> 6
                  *   SDF_SADDLE  -> 2 (amplitude_m, neighborhood_margin_uv)
                  *   SDF_SPHERE  -> 2 (radius_m, neighborhood_margin_uv)
                  */
                 py::array_t<double, py::array::c_style | py::array::forcecast> param_arr;
                 py::array_t<float,  py::array::c_style | py::array::forcecast> param_arr_f32;
                 if (!parametric_surface.is_none()) {
                     py::dict d = parametric_surface.cast<py::dict>();
                     desc.parametric_surface_kind =
                         d.contains("kind") ? d["kind"].cast<int>() : TRI_PARAM_SURFACE_NONE;
                     if (d.contains("payload_f32") && desc.parametric_surface_kind != TRI_PARAM_SURFACE_NONE) {
                         /* float32 payload — used by NEURAL_ASSEMBLY; avoids float64 conversion */
                         param_arr_f32 = d["payload_f32"].cast<
                             py::array_t<float, py::array::c_style | py::array::forcecast>>();
                         auto pa = param_arr_f32.request();
                         if (pa.ndim != 1)
                             throw std::runtime_error("parametric_surface.payload_f32 must be 1-D float32");
                         desc.parametric_payload = pa.ptr;
                         desc.parametric_payload_bytes = (int)(pa.size * pa.itemsize);
                     } else if (d.contains("coeffs") && desc.parametric_surface_kind != TRI_PARAM_SURFACE_NONE) {
                         param_arr = d["coeffs"].cast<
                             py::array_t<double, py::array::c_style | py::array::forcecast>>();
                         auto pa = param_arr.request();
                         if (pa.ndim != 1)
                             throw std::runtime_error("parametric_surface.coeffs must be 1-D float64");
                         desc.parametric_payload = pa.ptr;
                         desc.parametric_payload_bytes = (int)(pa.size * pa.itemsize);
                     }
                 }

                 /* Optional CameraSensorDesc dict.  Lifetime: the
                  * registrar deep-copies into the state, so the local
                  * cam_desc here just needs to outlive the call. */
                 CameraSensorDesc cam_desc{};
                 if (!sensor_camera.is_none()) {
                     py::dict d = sensor_camera.cast<py::dict>();
                     auto get3 = [&](const char* key, double* out) {
                         auto a = d[key].cast<py::array_t<double>>().unchecked<1>();
                         for (int i = 0; i < 3; ++i) out[i] = a(i);
                     };
                     get3("pos", cam_desc.pos);
                     get3("fwd", cam_desc.fwd);
                     get3("up",  cam_desc.up);
                     cam_desc.sensor_w_m         = d["sensor_w_m"].cast<double>();
                     cam_desc.sensor_h_m         = d["sensor_h_m"].cast<double>();
                     cam_desc.focal_m            = d["focal_m"].cast<double>();
                     cam_desc.aperture_radius_m  = d["aperture_radius_m"].cast<double>();
                     cam_desc.n_px               = d["n_px"].cast<int>();
                     cam_desc.n_py               = d["n_py"].cast<int>();
                     cam_desc.n_aperture_samples = d["n_aperture_samples"].cast<int>();
                     cam_desc.aperture_stop_group_id =
                         d.contains("aperture_stop_group_id")
                         ? d["aperture_stop_group_id"].cast<int>() : -1;
                     cam_desc.pixel_stream_divisor =
                         d.contains("pixel_stream_divisor")
                         ? d["pixel_stream_divisor"].cast<int>() : 1;
                     cam_desc.pixel_stream_phase =
                         d.contains("pixel_stream_phase")
                         ? d["pixel_stream_phase"].cast<int>() : 0;
                     cam_desc.pixel_stream_phase_from_seed =
                         d.contains("pixel_stream_phase_from_seed")
                         ? d["pixel_stream_phase_from_seed"].cast<int>() : 1;
                     /* Optical mode extensions (all optional, safe to omit). */
                     cam_desc.camera_mode =
                         d.contains("camera_mode")
                         ? d["camera_mode"].cast<int>() : 0;
                     cam_desc.effective_focal_m =
                         d.contains("effective_focal_m")
                         ? d["effective_focal_m"].cast<double>() : 0.0;
                     cam_desc.focus_distance_m =
                         d.contains("focus_distance_m")
                         ? d["focus_distance_m"].cast<double>() : 0.0;
                     if (d.contains("lens_center"))
                         get3("lens_center", cam_desc.lens_center);
                     if (d.contains("lens_fwd"))
                         get3("lens_fwd", cam_desc.lens_fwd);
                     cam_desc.use_optical_handlers =
                         d.contains("use_optical_handlers")
                         ? d["use_optical_handlers"].cast<int>() : 0;
                     desc.sensor_camera = &cam_desc;
                 }

                 /* Optional UV integrator image.
                  * Expected dict: {"res": int} — enables UV accumulation at
                  * the given resolution (res×res texels, 2 channels: count +
                  * amplitude).  An optional "uv_coords" key may supply an
                  * explicit float32 array of shape (n_tris*6,) / (n_tris,3,2)
                  * containing per-vertex UV coordinates.  When absent, planar
                  * projection is used (auto-computed from group geometry). */
                 py::array_t<float, py::array::c_style | py::array::forcecast> uv_arr;
                 if (!uv_image.is_none()) {
                     py::dict ud = uv_image.cast<py::dict>();
                     desc.uv_image_res = ud["res"].cast<int>();
                     if (ud.contains("uv_coords") && !ud["uv_coords"].is_none()) {
                         uv_arr = ud["uv_coords"].cast<
                             py::array_t<float, py::array::c_style | py::array::forcecast>>();
                         auto uvb = uv_arr.request();
                         desc.uv_n_coords = (int)uvb.size;
                         desc.uv_coords   = static_cast<const float*>(uvb.ptr);
                     }
                 }

                 int gid = ray_tracer_register_tri_group(self.handle, &desc);
                 if (gid < 0)
                     throw std::runtime_error(
                         "register_tri_group failed: rc=" + std::to_string(gid));
                 return gid;
             },
             py::arg("role_bits"),
             py::arg("sample_policy") = TRI_GROUP_SAMPLE_AREA,
             py::arg("tri_indices"),
             py::arg("plane_origin") = py::none(),
             py::arg("plane_normal") = py::none(),
             py::arg("custom_emit_W") = 0.0,
             py::arg("default_mat_idx") = -1,
             py::arg("power_W_per_band") = py::none(),
             py::arg("parametric_surface") = py::none(),
             py::arg("sensor_camera") = py::none(),
             py::arg("uv_image") = py::none(),
             R"doc(
Register a triangle group for the bidirectional integrator.

role_bits        : OR of TRI_GROUP_ROLE_EMISSIVE / SENSOR / BLOCKER / VOLUME
sample_policy    : TRI_GROUP_SAMPLE_AREA (default) / UNIFORM / POWER /
                   PIXEL_CONE (SENSOR-only camera-sim drive mode)
tri_indices      : int32 1-D array — triangle indices (subset of any size)
plane_origin/plane_normal : optional virtual-plane metadata for plane-tagged
                   sensor groups
custom_emit_W    : optional override for total emitted power (0 = derive)
default_mat_idx  : per-group fallback material index (-1 = derive from
                   majority across the group's tris)
power_W_per_band : optional float32 (n_bands,) per-band emission spectrum
                   override.  None = use mat_buf emission row.  This is the
                   ONLY emission-curve fallback path — no PBR.
parametric_surface: optional dict for strike/emission parametric override.
                                                 Supported now:
                                                      {"kind": TRI_PARAM_SURFACE_POLY_BARY,
                                                          "coeffs": float64[6]} where coeffs define normal-offset
                                                          polynomial in barycentric (u,v).
                                                      {"kind": TRI_PARAM_SURFACE_SDF_SADDLE,
                                                          "coeffs": float64[2]} where coeffs are
                                                          [amplitude_m, neighborhood_margin_uv].
                                                      {"kind": TRI_PARAM_SURFACE_SDF_SPHERE,
                                                          "coeffs": float64[2]} where coeffs are
                                                          [radius_m, neighborhood_margin_uv].
sensor_camera    : optional dict for SENSOR + PIXEL_CONE groups, with keys
                   pos (3,), fwd (3,), up (3,), sensor_w_m, sensor_h_m,
                   focal_m, aperture_radius_m, n_px, n_py,
                   n_aperture_samples, [aperture_stop_group_id (default -1)],
                   [pixel_stream_divisor (default 1)],
                   [pixel_stream_phase (default 0)],
                   [pixel_stream_phase_from_seed (default 1)].

Returns assigned group_id (>= 0).  Raises on failure.
)doc")
        .def("clear_tri_groups",
             [](PyRayTracer& self) { ray_tracer_clear_tri_groups(self.handle); })
        .def("set_tri_group_power",
             [](PyRayTracer& self, int group_id,
                py::array_t<float, py::array::c_style | py::array::forcecast> power_arr) {
                 auto b = power_arr.request();
                 int rc = ray_tracer_set_tri_group_power(
                     self.handle, group_id,
                     static_cast<const float*>(b.ptr),
                     static_cast<int>(b.size));
                 if (rc != 0)
                     throw std::runtime_error(
                         "set_tri_group_power failed rc=" + std::to_string(rc));
             },
             py::arg("group_id"),
             py::arg("power_W_per_band"),
             R"doc(Update the per-band emission power of an already-registered EMISSIVE group.

Call between tracing passes (never during an active trace).  The primary use
case is the WaveTube surrogate emitter: after BPM solve the exit group's power
is set to the BPM-integrated exit flux ∫|E(x,y)|² dx dy per band.

group_id          : int — group_id returned by register_tri_group()
power_W_per_band  : float32 (n_bands,) — physical power in watts per band
)doc")
        .def("attach_optical_assembly",
             [](PyRayTracer& self, py::object backend) {
                 OpticalAssembly* asmb = nullptr;
                 if (!backend.is_none()) {
                     auto* b = backend.cast<ExposureBackendCpp*>();
                     if (b) asmb = b->assembly;
                 }
                 ray_tracer_attach_optical_assembly(self.handle, asmb);
             },
             py::arg("backend"),
             "Attach an ExposureBackendCpp optical assembly for PIXEL_CONE handler "
             "dispatch.  Pass None to detach.")
        .def("n_tri_groups",
             [](PyRayTracer& self) {
                 return ray_tracer_n_tri_groups(self.handle);
             })
        .def("get_manifold_gid_stats",
             [](PyRayTracer& self, int gid) -> py::dict {
                 py::dict out;
                 out["magic"]       = py::float_(0.0f);
                 out["transmitted"] = py::int_(0);
                 out["absorbed"]    = py::int_(0);
                 float m = 0.0f; uint64_t ok = 0, ab = 0;
                 if (ray_tracer_get_manifold_gid_stats(self.handle, gid, &m, &ok, &ab) == SK_OK) {
                     out["magic"]       = py::float_(m);
                     out["transmitted"] = py::int_(ok);
                     out["absorbed"]    = py::int_(ab);
                 }
                 return out;
             },
             py::arg("gid"),
             "Per-GID manifold dispatch stats: {magic, transmitted, absorbed}. CPU T2 path only.")
        .def("get_all_manifold_stats",
             [](PyRayTracer& self) -> py::list {
                 py::list out;
                 const int n = ray_tracer_n_tri_groups(self.handle);
                 for (int gid = 0; gid < n; ++gid) {
                     float m = 0.0f; uint64_t ok = 0, ab = 0;
                     if (ray_tracer_get_manifold_gid_stats(self.handle, gid, &m, &ok, &ab) != SK_OK)
                         continue;
                     py::dict d;
                     d["gid"]         = py::int_(gid);
                     d["magic"]       = py::float_(m);
                     d["transmitted"] = py::int_(ok);
                     d["absorbed"]    = py::int_(ab);
                     out.append(d);
                 }
                 return out;
             },
             "All per-GID manifold dispatch stats as a list of {gid, magic, transmitted, absorbed}.")
        /* ── UV integrator image readback ─────────────────────────────────── */
        .def("get_group_uv_image",
             [](PyRayTracer& self, int group_id) -> py::dict {
                 int res = 0, n_ch = 0;
                 int rc  = ray_tracer_get_group_uv_image(
                     self.handle, group_id, nullptr, &res, &n_ch);
                 if (rc != 0 || res <= 0 || n_ch <= 0)
                     throw std::runtime_error(
                         "get_group_uv_image: group " + std::to_string(group_id) +
                         " has no UV image or is invalid (rc=" + std::to_string(rc) + ")");
                 py::array_t<float> ch_arr({(py::ssize_t)n_ch,
                                            (py::ssize_t)res,
                                            (py::ssize_t)res});
                 rc = ray_tracer_get_group_uv_image(
                     self.handle, group_id,
                     static_cast<float*>(ch_arr.mutable_data()), &res, &n_ch);
                 if (rc != 0)
                     throw std::runtime_error(
                         "get_group_uv_image failed rc=" + std::to_string(rc));
                 const int nb = self._n_bands;
                 py::dict result;
                 result["channels"]  = ch_arr;
                 result["res"]       = res;
                 result["n_bands"]   = nb;
                 /* Convenience aliases. */
                 result["count"]     = ch_arr[py::int_(0)];
                 result["src_flags"] = ch_arr[py::int_(1)];
                 result["normal"]    = ch_arr[py::slice(8, 11, 1)];
                 /* amp: sum of per-band magnitudes → (res,res) */
                 py::object amp_bands = ch_arr[py::slice(11, 11 + nb, 1)];
                 result["amp"] = amp_bands.attr("sum")(
                     py::arg("axis") = 0);
                 result["band_mag"]    = amp_bands;
                 result["amp_re"]      = ch_arr[py::slice(11 + nb,     11 + 2 * nb, 1)];
                 result["amp_im"]      = ch_arr[py::slice(11 + 2 * nb, 11 + 3 * nb, 1)];
                 result["forward_mag"] = ch_arr[py::slice(11 + 3 * nb, 11 + 4 * nb, 1)];
                 result["sensor_mag"]  = ch_arr[py::slice(11 + 4 * nb, 11 + 5 * nb, 1)];
                 return result;
             },
             py::arg("group_id"),
             R"doc(Return UV accumulator data for a group.

Keys:
  channels   : float32(n_channels, res, res) — all decoded channels
  res        : int
  n_bands    : int
  count      : float32(res,res)     — hit count              [ch 0]
  src_flags  : float32(res,res)     — source-ID bitfield     [ch 1]
  normal     : float32(3,res,res)   — hit-normal xyz sum     [ch 8-10]
  amp        : float32(res,res)     — sum of per-band |amp|  [ch 11..11+B]
  band_mag   : float32(B,res,res)   — total per-band magnitude
  amp_re/im  : float32(B,res,res)   — signed complex amplitude components
  forward_mag: float32(B,res,res)   — forward/emissive contribution only
  sensor_mag : float32(B,res,res)   — sensor/reverse contribution only

Full channel layout and UV_CH_* indices are documented in ray_tracer.h.
)doc")
        .def("set_group_uv_image",
             [](PyRayTracer& self, int group_id,
                py::array_t<float, py::array::c_style | py::array::forcecast> channels) {
                 auto b = channels.request();
                 if (b.ndim != 3)
                     throw std::invalid_argument("channels must be float32 shape (n_channels, res, res)");
                 const int n_ch = (int)b.shape[0];
                 const int res_y = (int)b.shape[1];
                 const int res_x = (int)b.shape[2];
                 if (res_x != res_y)
                     throw std::invalid_argument("channels must have square res x res pages");
                 int rc = ray_tracer_set_group_uv_image(
                     self.handle,
                     group_id,
                     static_cast<const float*>(b.ptr),
                     res_x,
                     n_ch);
                 if (rc != 0)
                     throw std::runtime_error(
                         "set_group_uv_image failed rc=" + std::to_string(rc));
             },
             py::arg("group_id"),
             py::arg("channels"),
             "Replace one group's UV accumulator from decoded float32 channels.")
        .def("clear_group_uv_accum",
             [](PyRayTracer& self, int group_id) {
                 int rc = ray_tracer_clear_group_uv_accum(self.handle, group_id);
                 if (rc != 0)
                     throw std::runtime_error(
                         "clear_group_uv_accum failed rc=" + std::to_string(rc));
             },
             py::arg("group_id") = -1,
             "Zero the UV accumulator for group_id (pass -1 to clear all groups).")
        /* ── BSSRDF illumination accumulator ──────────────────────────────── */
        .def("init_illum_accum",
             [](PyRayTracer& self) {
                 int rc = ray_tracer_init_illum_accum(self.handle);
                 if (rc != 0)
                     throw std::runtime_error(
                         "init_illum_accum failed rc=" + std::to_string(rc));
             },
             R"doc(Size and zero the per-triangle BSSRDF illumination accumulator.

Call once after scene geometry is set (before any tracing).  The accumulator
stores the pre-interaction amplitude of forward paths that hit diffuse-transmissive
surfaces (diffuse_frac > 0).  Backward paths query this to receive their
analytical diffuse illumination contribution without scatter-volume traversal.

Two-pass protocol:
  tracer.init_illum_accum()    # once after geometry is set
  tracer.reset_illum_accum()   # between forward-pass batches
  tracer.trace_forward(...)    # forward paths populate the accumulator
  tracer.trace_backward(...)   # backward paths query the accumulator
  data = tracer.export_illum_accum()  # optional: inspect or feed T4 BPM
)doc")
        .def("reset_illum_accum",
             [](PyRayTracer& self) {
                 int rc = ray_tracer_reset_illum_accum(self.handle);
                 if (rc != 0)
                     throw std::runtime_error(
                         "reset_illum_accum failed rc=" + std::to_string(rc));
             },
             "Zero the illumination accumulator without freeing memory. "
             "Call between forward-pass batches to avoid contaminating backward-ray "
             "lookups with stale data.")
        .def("export_illum_accum",
             [](PyRayTracer& self) -> py::array_t<float> {
                 int n_tris = 0, stride = 0;
                 ray_tracer_export_illum_accum(self.handle, nullptr, 0,
                                               &n_tris, &stride);
                 if (n_tris <= 0 || stride <= 0)
                     return py::array_t<float>({0});
                 py::array_t<float> out({(py::ssize_t)n_tris,
                                         (py::ssize_t)stride});
                 int rc = ray_tracer_export_illum_accum(
                     self.handle,
                     static_cast<float*>(out.mutable_data()),
                     n_tris * stride,
                     &n_tris, &stride);
                 if (rc != 0)
                     throw std::runtime_error(
                         "export_illum_accum failed rc=" + std::to_string(rc));
                 return out;
             },
             R"doc(Return float32 array of shape (n_tris, stride) from the BSSRDF accumulator.

stride = 2 * n_bands + 2.  Layout per triangle row:
  col 2*b + 0   amp_sum[b].real  (sum of forward-path pre-interaction re amplitudes)
  col 2*b + 1   amp_sum[b].imag
  col 2*nb      cos_sum          (sum of |cos θ| incidence weights)
  col 2*nb + 1  count            (number of forward contributions)

Divide re/im by count to get the average forward amplitude at each surface.
cos_sum / count gives avg_cos for the Lambertian coupling weight.

This array can be used to seed a T4 BPM wave-solver run: replace the
Monte Carlo averages with diffraction-correct BPM exit-plane amplitudes
for a more physically accurate BSSRDF contribution.)doc")
        .def("write_tri_illum",
             [](PyRayTracer& self,
                py::array_t<int32_t, py::array::c_style|py::array::forcecast>   tri_ids,
                py::array_t<float,   py::array::c_style|py::array::forcecast>   amp_re,
                py::array_t<float,   py::array::c_style|py::array::forcecast>   amp_im,
                py::array_t<float,   py::array::c_style|py::array::forcecast>   cos_avg) {
                 const int n_tris  = static_cast<int>(tri_ids.shape(0));
                 const int n_bands = (amp_re.ndim() == 2)
                                         ? static_cast<int>(amp_re.shape(1)) : 1;
                 int rc = ray_tracer_write_tri_illum(
                     self.handle,
                     static_cast<const int*>(tri_ids.data()),
                     n_tris,
                     static_cast<const float*>(amp_re.data()),
                     static_cast<const float*>(amp_im.data()),
                     static_cast<const float*>(cos_avg.data()),
                     n_bands);
                 if (rc != 0)
                     throw std::runtime_error(
                         "write_tri_illum failed rc=" + std::to_string(rc));
             },
             py::arg("tri_ids"),
             py::arg("amp_re"),
             py::arg("amp_im"),
             py::arg("cos_avg"),
             R"doc(Write BPM-computed amplitudes into tri_illum_accum for specific triangles.

Replaces any Monte Carlo data at those triangles with the BPM exit field.
Backward rays that subsequently hit those triangles receive diffraction-correct
illumination from the BSSRDF analytical path.

Parameters
----------
tri_ids  : int32  (n_tris,)          triangle indices
amp_re   : float32 (n_tris, n_bands) real part of BPM exit field per triangle
amp_im   : float32 (n_tris, n_bands) imaginary part
cos_avg  : float32 (n_tris,)         mean |cos θ| of incidence; use 1.0 for
                                     normal incidence at a flat diffuser face)doc")
        .def("get_group_uv_summary",
             [](PyRayTracer& self, int group_id) -> py::dict {
                 RayTracerUvGroupSummary s{};
                 int rc = ray_tracer_get_group_uv_summary(self.handle, group_id, &s);
                 if (rc != 0)
                     throw std::runtime_error(
                         "get_group_uv_summary failed rc=" + std::to_string(rc));
                 py::dict out;
                 out["group_id"] = s.group_id;
                 out["res"] = s.res;
                 out["n_channels"] = s.n_channels;
                 out["tri_count"] = s.tri_count;
                 out["memory_bytes"] = py::int_(s.memory_bytes);
                 out["nonzero_texels"] = py::int_(s.nonzero_texels);
                 out["total_forward"] = s.total_forward;
                 out["total_sensor"] = s.total_sensor;
                 out["peak_total"] = s.peak_total;
                 return out;
             },
             py::arg("group_id"),
             "Return cheap telemetry for one UV accumulator group.")
        .def("list_uv_groups",
             [](PyRayTracer& self) -> py::list {
                 py::list groups;
                 const int n = ray_tracer_n_tri_groups(self.handle);
                 for (int gid = 0; gid < n; ++gid) {
                     RayTracerUvGroupSummary s{};
                     int rc = ray_tracer_get_group_uv_summary(self.handle, gid, &s);
                     if (rc != 0) continue;
                     py::dict out;
                     out["group_id"] = s.group_id;
                     out["res"] = s.res;
                     out["n_channels"] = s.n_channels;
                     out["tri_count"] = s.tri_count;
                     out["memory_bytes"] = py::int_(s.memory_bytes);
                     out["nonzero_texels"] = py::int_(s.nonzero_texels);
                     out["total_forward"] = s.total_forward;
                     out["total_sensor"] = s.total_sensor;
                     out["peak_total"] = s.peak_total;
                     groups.append(out);
                 }
                 return groups;
             },
             "List UV accumulator groups with cheap telemetry.")
        /* ── WGL context sharing / GPU-direct UV blit ─────────────────────── */
        .def("set_gpu_skip_record_readback",
             [](PyRayTracer& self, bool skip) {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 self._gpu_skip_record_readback = skip;
                 if (self._pipeline)
                     ray_pipeline_set_skip_record_readback(self._pipeline, skip);
             },
             py::arg("skip"),
             R"doc(Toggle production GPU mode: skip per-bounce terminal, hit-record, and
T5 tile readbacks to CPU.  When True the display image goes dark until a
GPU-resident sensor accumulator is wired.  Safe to call at any time;
takes effect on the next GPU bounce.)doc")
        /* ── Pass E: explicit debug readback taps ──────────────────────────── */
        .def("debug_read_hits",
             [](PyRayTracer& self, int max_n) -> py::array_t<float> {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 if (!self._pipeline) return py::array_t<float>();
                 auto data = ray_pipeline_debug_read_hits(self._pipeline, max_n);
                 if (data.empty()) return py::array_t<float>();
                 py::array_t<float> out({(py::ssize_t)data.size()});
                 std::copy(data.begin(), data.end(), out.mutable_data());
                 return out;
             },
             py::arg("max_n") = 4096,
             R"doc(Bounded readback of the last GPU hit-record buffer (HIT_STRIDE floats each).
Returns a 1-D float32 array of length n_actual × HIT_STRIDE.
Never call from the normal frame loop — for diagnostics only.
Blocks ≤ ~5 ms for the GPU thread to service the request.)doc")
        .def("debug_read_terminals",
             [](PyRayTracer& self, int max_n) -> py::array_t<float> {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 if (!self._pipeline) return py::array_t<float>();
                 auto data = ray_pipeline_debug_read_terminals(self._pipeline, max_n);
                 if (data.empty()) return py::array_t<float>();
                 py::array_t<float> out({(py::ssize_t)data.size()});
                 std::copy(data.begin(), data.end(), out.mutable_data());
                 return out;
             },
             py::arg("max_n") = 4096,
             R"doc(Bounded readback of T3 terminal records (TERMINAL_STRIDE floats each).
Returns a 1-D float32 array of length n_actual × TERMINAL_STRIDE.
Never call from the normal frame loop — for diagnostics only.)doc")
        .def("debug_read_bdpt_vertices",
             [](PyRayTracer& self, int max_n) -> py::array_t<float> {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 if (!self._pipeline) return py::array_t<float>();
                 auto data = ray_pipeline_debug_read_bdpt_vertices(self._pipeline, max_n);
                 if (data.empty()) return py::array_t<float>();
                 py::array_t<float> out({(py::ssize_t)data.size()});
                 std::copy(data.begin(), data.end(), out.mutable_data());
                 return out;
             },
             py::arg("max_n") = 4096,
             R"doc(Bounded readback of BDPT vertex records from the GPU (28 floats each).
Returns a 1-D float32 array of length n_actual × 28.
Never call from the normal frame loop — for diagnostics only.)doc")
        .def("set_gl_display_hglrc",
             [](PyRayTracer& self, uint64_t h) {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 self._gl_display_hglrc = h;
                 if (self._pipeline)
                     fprintf(stderr, "[warn] set_gl_display_hglrc called after pipeline "
                             "created — has no effect on this session\n");
             },
             py::arg("hglrc"),
             "Set display HGLRC for WGL object sharing.  Call before first submit_rays.")
        .def("set_gl_display_hdc",
             [](PyRayTracer& self, uint64_t h) {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 self._gl_display_hdc = h;
                 if (self._pipeline)
                     fprintf(stderr, "[warn] set_gl_display_hdc called after pipeline "
                             "created — has no effect on this session\n");
             },
             py::arg("hdc"),
             "Set display HDC for pixel-format matching.  Call before first submit_rays.")
        .def("get_uv_pages_tex_id",
             [](PyRayTracer& self) -> uint64_t {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 return ray_pipeline_get_uv_pages_tex_id(self._pipeline);
             },
             R"doc(Return the OpenGL texture object ID of the shared tex_uv_pages
TEXTURE_2D_ARRAY (RGBA16F).  Returns 0 if WGL sharing is not active or the
pipeline has not been initialised or no completed generation is ready yet.
When non-zero the texture is already owned by the shared GL namespace and the
fence for that generation has signaled; bind it directly in the display context
for zero-copy UV visualisation.)doc")
        .def("get_field_display_tex_id",
             [](PyRayTracer& self) -> uint64_t {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 return ray_pipeline_get_field_display_tex_id(self._pipeline);
             },
             R"doc(Return the OpenGL texture object ID of the shared field-display
GL_TEXTURE_3D (RGBA32F). Returns 0 if GPU field display is unavailable.)doc")
        .def("request_field_display_clear",
             [](PyRayTracer& self) {
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 ray_pipeline_request_field_display_clear(self._pipeline);
             },
             R"doc(Request clearing the GPU field-display accumulator on the GPU thread.)doc")
        .def("set_uv_blit_weights",
             [](PyRayTracer& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> weights,
                int mode) {
                 auto b = weights.request();
                 if (b.ndim != 2 || b.shape[1] != 3)
                     throw std::invalid_argument(
                         "weights must be float32 shape (n_bands, 3)");
                 const int nb = (int)b.shape[0];
                 if (nb < 1 || nb > MAX_SPECTRAL_BANDS)
                     throw std::invalid_argument("n_bands must be in [1, MAX_SPECTRAL_BANDS]");
                 std::lock_guard<std::mutex> lk(self._pipeline_mu);
                 std::copy_n(static_cast<const float*>(b.ptr),
                             static_cast<size_t>(nb) * 3,
                             self._uv_blit_weights.data());
                 self._uv_blit_n_bands = nb;
                 self._uv_blit_mode = mode;
                 if (self._pipeline)
                     ray_pipeline_set_uv_blit_weights(
                         self._pipeline, self._uv_blit_weights.data(), nb, mode);
             },
             py::arg("weights"),
             py::arg("mode") = 0,
             R"doc(Upload per-band RGB weights for the GPU UV blit shader.
weights : float32 ndarray of shape (n_bands, 3) — each row is [r, g, b] weight
          for mapping one spectral band's magnitude to sRGB.  Typically from
          _wavelength_to_rgb_weights(freq_hz).
mode    : 0 = combined (fwd+sensor), 1 = forward only, 2 = sensor only.)doc")
    ;

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

    /* ── AcousticAMR ──────────────────────────────────────────────────────── */

    py::class_<PyAcousticAMR>(m, "AcousticAMR",
        R"doc(
Topology-driven AMR acoustic pressure stepper.

The grid is supplied as explicit pressure cells and connecting velocity faces.
Numerical stepping is performed in C++/Eigen; Python is only responsible for
constructing and validating the AMR descriptor.
)doc")
        .def(py::init<py::array_t<double>,
                      py::array_t<double>,
                      py::array_t<double>,
                      py::array_t<uint8_t>,
                      py::array_t<int32_t>,
                      py::array_t<int32_t>,
                      py::array_t<double>,
                      py::array_t<double>,
                      py::array_t<double>,
                      double,
                      double,
                      double,
                      int>(),
             py::arg("cell_centers"),
             py::arg("cell_volumes"),
             py::arg("open_volume_fraction"),
             py::arg("cell_types"),
             py::arg("face_cell_neg"),
             py::arg("face_cell_pos"),
             py::arg("face_area"),
             py::arg("face_open_fraction"),
             py::arg("face_distance"),
             py::arg("c"),
             py::arg("rho_air"),
             py::arg("min_dx"),
             py::arg("gradient_order"))
        .def("step", &PyAcousticAMR::step, py::arg("n_steps") = 1)
        .def("reset", &PyAcousticAMR::reset)
        .def("inject_pressure_nearest", &PyAcousticAMR::inject_pressure_nearest,
             py::arg("xyz"), py::arg("value"))
        .def("get_pressure", &PyAcousticAMR::get_pressure)
        .def("get_velocity", &PyAcousticAMR::get_velocity)
        .def_property_readonly("dt", &PyAcousticAMR::get_dt)
        .def_property_readonly("step_count", &PyAcousticAMR::get_step_count)
        .def_property_readonly("n_cells", [](const PyAcousticAMR& o){ return o.n_cells; })
        .def_property_readonly("n_faces", [](const PyAcousticAMR& o){ return o.n_faces; });

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
        .def_static("from_amr_desc",
             &PyAcousticCoEvolver::from_amr_desc,
             py::arg("amr_desc"),
             py::arg("string_defs"),
             py::arg("pickup_defs"),
             py::arg("mic_defs"),
             py::arg("sample_rate"),
             py::arg("modal_stride") = 16,
             R"doc(
Create an AcousticCoEvolver backed by an AMR FDTD pressure domain.

amr_desc is the dict returned by ``build_amr_coevolver_descriptor()`` from
acoustic_amr.py.  string/pickup/mic_defs use the same schema as the
uniform constructor.  GIL is released during the C++ allocation.
)doc")
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
             py::arg("n_duration_samples") = 0,
             "Excite a string with a raised-cosine pluck at position_norm (0=nut, 1=saddle). "
             "n_duration_samples controls the draw length (0 = default ~20 ms).")
        .def("schedule_pluck", &PyAcousticCoEvolver::schedule_pluck,
             py::arg("onset_sample"), py::arg("string_idx"),
             py::arg("position_norm"), py::arg("amplitude"),
             py::arg("n_duration_samples") = 0,
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
             "AMR: float32 (n_cells,). FDTD: float32 (Nx,Ny,Nz).")
        .def("get_amr_cell_centers", &PyAcousticCoEvolver::get_amr_cell_centers,
             "Return AMR cell centres as float32 (n_cells,3). Raises if not AMR.")
        .def("get_plate_dims", &PyAcousticCoEvolver::get_plate_dims,
             "Return (plate_Nx, plate_Ny, plate_count) for the active pressure backend.")
        .def("get_plate_displacement", &PyAcousticCoEvolver::get_plate_displacement,
             "Return 2-D plate displacement w(x,y), float32 shape (plate_Nx,plate_Ny).")
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
    /* Ray-tracer triangle surface flags (Phase 2: unified MAT_FLAG_*). */
    m.attr("MAT_FLAG_EMISSIVE")      = (int)MAT_FLAG_EMISSIVE;
    m.attr("MAT_FLAG_REACTIVE")      = (int)MAT_FLAG_REACTIVE;
    m.attr("MAT_FLAG_ABSORBER")      = (int)MAT_FLAG_ABSORBER;
    m.attr("MAT_FLAG_NO_SHADOW")     = (int)MAT_FLAG_NO_SHADOW;
    m.attr("MAT_FLAG_MANIFOLD")      = (int)MAT_FLAG_MANIFOLD;
    m.attr("MAT_FLAG_PARAMETRIC")    = (int)MAT_FLAG_PARAMETRIC;
    m.attr("MAT_FLAG_TRANSMISSIVE")  = (int)MAT_FLAG_TRANSMISSIVE;
    m.attr("BDPT_PDF_FLAG_SENSOR")   = (int)BDPT_PDF_FLAG_SENSOR;
    m.attr("MAT_FLAG_APERTURE_STOP") = (int)MAT_FLAG_APERTURE_STOP;
    m.attr("MAT_FLAG_PICKING_ONLY")  = (int)MAT_FLAG_PICKING_ONLY;
    m.attr("MAX_SPECTRAL_BANDS")     = (int)MAX_SPECTRAL_BANDS;
    /* Multiscale scale-type constants */
    m.attr("RT_SCALE_GEOMETRIC") = (int)RT_SCALE_GEOMETRIC;
    m.attr("RT_SCALE_WAVE")      = (int)RT_SCALE_WAVE;
    m.attr("RT_CAM_VIS_AS_IS") = (int)RT_CAM_VIS_AS_IS;
    m.attr("RT_CAM_VIS_DIRECT_HIT") = (int)RT_CAM_VIS_DIRECT_HIT;
    m.attr("RT_CAM_VIS_FULL_MARCH") = (int)RT_CAM_VIS_FULL_MARCH;
    m.attr("RT_CAM_TRANSPARENCY_BLOCK") = (int)RT_CAM_TRANSPARENCY_BLOCK;
    m.attr("RT_CAM_TRANSPARENCY_XRAY") = (int)RT_CAM_TRANSPARENCY_XRAY;
    /* Scale-context KIND dispatch enum (mirrors GLSL ScaleContext SSBO).
     * Used by RayTracer.add_scale_context(context_kind=...). */
    m.attr("SCALE_CONTEXT_KIND_RAY")                 = (int)SCALE_CONTEXT_KIND_RAY;
    m.attr("SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ")      = (int)SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ;
    m.attr("SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM") = (int)SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM;
    m.attr("SCALE_CONTEXT_KIND_THICK_LENS_WAVE")     = (int)SCALE_CONTEXT_KIND_THICK_LENS_WAVE;
    m.attr("SCALE_CONTEXT_KIND_SPLINE_SURFACE")      = (int)SCALE_CONTEXT_KIND_SPLINE_SURFACE;
    m.attr("SCALE_CONTEXT_KIND_NEURAL_SURFACE")      = (int)SCALE_CONTEXT_KIND_NEURAL_SURFACE;
    m.attr("SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC")   = (int)SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC;
    /* Triangle-group registry constants (used with register_tri_group). */
    m.attr("TRI_GROUP_ROLE_EMISSIVE")     = (int)TRI_GROUP_ROLE_EMISSIVE;
    m.attr("TRI_GROUP_ROLE_SENSOR")       = (int)TRI_GROUP_ROLE_SENSOR;
    m.attr("TRI_GROUP_ROLE_BLOCKER")      = (int)TRI_GROUP_ROLE_BLOCKER;
    m.attr("TRI_GROUP_ROLE_VOLUME")       = (int)TRI_GROUP_ROLE_VOLUME;
    m.attr("TRI_GROUP_SAMPLE_UNIFORM")    = (int)TRI_GROUP_SAMPLE_UNIFORM;
    m.attr("TRI_GROUP_SAMPLE_AREA")       = (int)TRI_GROUP_SAMPLE_AREA;
    m.attr("TRI_GROUP_SAMPLE_POWER")      = (int)TRI_GROUP_SAMPLE_POWER;
    m.attr("TRI_GROUP_SAMPLE_PIXEL_CONE") = (int)TRI_GROUP_SAMPLE_PIXEL_CONE;
    m.attr("TRI_PARAM_SURFACE_NONE")             = (int)TRI_PARAM_SURFACE_NONE;
    m.attr("TRI_PARAM_SURFACE_POLY_BARY")        = (int)TRI_PARAM_SURFACE_POLY_BARY;
    m.attr("TRI_PARAM_SURFACE_SDF_SADDLE")       = (int)TRI_PARAM_SURFACE_SDF_SADDLE;
    m.attr("TRI_PARAM_SURFACE_SDF_SPHERE")       = (int)TRI_PARAM_SURFACE_SDF_SPHERE;
    m.attr("TRI_PARAM_SURFACE_NEURAL_ASSEMBLY")  = (int)TRI_PARAM_SURFACE_NEURAL_ASSEMBLY;
    m.attr("TRI_PARAM_SURFACE_PARAMETRIC_LENS")  = (int)TRI_PARAM_SURFACE_PARAMETRIC_LENS;

    /* ── Surface spline fitter ──────────────────────────────────────────────── */
    m.def("surface_spline_fit",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> verts,
           py::array_t<int,    py::array::c_style | py::array::forcecast> tris,
           py::object  tri_subset_obj,
           double      ridge_lambda,
           int         n_threads) -> py::array_t<double>
        {
            auto vb = verts.request();
            auto tb = tris.request();
            if (vb.ndim != 2 || vb.shape[1] != 3)
                throw std::runtime_error("surface_spline_fit: verts must be (N,3) float64");
            if (tb.ndim != 2 || tb.shape[1] != 3)
                throw std::runtime_error("surface_spline_fit: tris must be (T,3) int32");

            int n_verts = static_cast<int>(vb.shape[0]);
            int n_tris  = static_cast<int>(tb.shape[0]);

            /* Optional subset. */
            py::array_t<int, py::array::c_style | py::array::forcecast> subset_arr;
            const int* subset_ptr = nullptr;
            int n_subset = 0;
            if (!tri_subset_obj.is_none()) {
                subset_arr = tri_subset_obj.cast<
                    py::array_t<int, py::array::c_style | py::array::forcecast>>();
                auto sb = subset_arr.request();
                subset_ptr = static_cast<const int*>(sb.ptr);
                n_subset   = static_cast<int>(sb.size);
            }

            /* Output: (n_tris, 6) float64. */
            py::array_t<double> out({n_tris, 6});
            int rc = surface_spline_fit(
                n_verts,
                static_cast<const double*>(vb.ptr),
                n_tris,
                static_cast<const int*>(tb.ptr),
                subset_ptr,
                n_subset,
                ridge_lambda,
                n_threads,
                static_cast<double*>(out.request().ptr));
            if (rc != SK_OK)
                throw std::runtime_error("surface_spline_fit failed: rc=" + std::to_string(rc));
            return out;
        },
        py::arg("verts"),
        py::arg("tris"),
        py::arg("tri_subset")    = py::none(),
        py::arg("ridge_lambda") = 0.0,
        py::arg("n_threads")    = 0,
        R"doc(
Fit per-triangle quadratic POLY_BARY displacement coefficients.

For each triangle, a 1-ring neighbourhood is assembled, centroid positions
are projected into barycentric frame, and a least-squares quadratic
displacement polynomial is solved with Eigen.

Parameters
----------
verts       : (N, 3) float64 vertex positions.
tris        : (T, 3) int32  vertex index triples.
tri_subset  : (K,) int32 or None.  If given, only those T-indices are fitted;
              remaining rows in the output are zero.
ridge_lambda: Tikhonov regularisation on curvature terms (default 0 = off).
n_threads   : worker threads (0 = hardware_concurrency).

Returns
-------
(T, 6) float64 — per-triangle [c0, cu, cv, cuu, cuv, cvv].
Pass row t directly as the ``coeffs`` field of TRI_PARAM_SURFACE_POLY_BARY.
)doc");

    m.def("surface_spline_eval_normal",
        [](double u, double v,
           py::array_t<double, py::array::c_style | py::array::forcecast> coeffs_6,
           py::array_t<double, py::array::c_style | py::array::forcecast> verts,
           py::array_t<int,    py::array::c_style | py::array::forcecast> tri_row)
        -> py::array_t<double>
        {
            if (coeffs_6.size() < 6)
                throw std::runtime_error("coeffs_6 must have 6 elements");
            if (tri_row.size() < 3)
                throw std::runtime_error("tri_row must have 3 elements");
            if (verts.request().ndim < 2)
                throw std::runtime_error("verts must be (N,3) float64");

            py::array_t<double> out({3});
            int rc = surface_spline_eval_normal(
                u, v,
                static_cast<const double*>(coeffs_6.request().ptr),
                static_cast<const double*>(verts.request().ptr),
                static_cast<const int*>(tri_row.request().ptr),
                static_cast<double*>(out.request().ptr));
            if (rc != SK_OK)
                throw std::runtime_error("surface_spline_eval_normal failed");
            return out;
        },
        py::arg("u"), py::arg("v"),
        py::arg("coeffs_6"),
        py::arg("verts"),
        py::arg("tri_row"),
        "Evaluate the perturbed unit normal at barycentric (u,v) from a POLY_BARY coefficient block.");

    m.def("amr_create_progress", []() {
        py::dict d;
        d["active"] = amr_get_create_progress_active();
        d["done_faces"] = amr_get_create_progress_done_faces();
        d["total_faces"] = amr_get_create_progress_total_faces();
        return d;
    }, "Return AMR create-time stencil progress counters.");

    m.def("build_amr_faces_sorted",
          &build_amr_faces_sorted_cpp,
          py::arg("centers"),
          py::arg("half_sizes"),
          py::arg("types"),
          "Build AMR face topology with the sorted lattice sweep in C++.");

    /* ── DocRenderer ────────────────────────────────────────────────────────── */

    struct PyDocRenderer {
        DocRendererState* st;
        PyDocRenderer(int w, int h) : st(dr_create(w, h)) {
            if (!st) throw std::runtime_error("dr_create failed");
        }
        ~PyDocRenderer() { if (st) { dr_destroy(st); st = nullptr; } }
    };

    py::class_<PyDocRenderer>(m, "DocRenderer",
        R"doc(
Document-hierarchy texture renderer.

Background worker thread + thread-safe FIFO + pointlessness filter.
Producers call submit_node() each frame; composite() blends all live tiles
into a flat RGBA8 numpy array sized (height, width, 4).
)doc")
        .def(py::init<int, int>(),
            py::arg("width"), py::arg("height"),
            "Create a renderer of the given pixel dimensions.")
        .def("load_glyph_atlas",
            [](PyDocRenderer& self,
               py::array_t<uint8_t> rgba,
               int glyph_w, int glyph_h) {
                auto info = rgba.request();
                if (info.ndim != 3 || info.shape[2] != 4)
                    throw std::runtime_error("load_glyph_atlas: expected (H, W, 4) uint8 array");
                dr_load_glyph_atlas_rgba(self.st,
                    static_cast<const uint8_t*>(info.ptr),
                    (int)info.shape[1], (int)info.shape[0],
                    glyph_w, glyph_h);
            },
            py::arg("rgba"), py::arg("glyph_w") = 8, py::arg("glyph_h") = 12,
            "Upload a pre-rendered (H, W, 4) uint8 glyph atlas.")
        .def("load_primitive_atlas",
            [](PyDocRenderer& self,
               py::array_t<uint8_t> rgba,
               int prim_w, int prim_h, int prim_cols) {
                auto info = rgba.request();
                if (info.ndim != 3 || info.shape[2] != 4)
                    throw std::runtime_error("load_primitive_atlas: expected (H, W, 4) uint8 array");
                dr_load_primitive_atlas_rgba(self.st,
                    static_cast<const uint8_t*>(info.ptr),
                    (int)info.shape[1], (int)info.shape[0],
                    prim_w, prim_h, prim_cols);
            },
            py::arg("rgba"), py::arg("prim_w"), py::arg("prim_h"), py::arg("prim_cols"),
            "Upload a pre-rendered (H, W, 4) uint8 primitive atlas.")
        .def("submit_node",
            [](PyDocRenderer& self,
               uint64_t node_id,
               int x, int y, int w, int h,
               int type,
               const std::string& label,
               const std::string& value_str,
               py::array_t<float> bg_rgba,
               py::array_t<float> fg_rgba,
               py::array_t<float> border_rgba,
               py::array_t<float> accent_rgba,
               float corner_radius,
               float font_scale,
               float value_norm,
               int icon_id,
               int border_px,
               uint64_t parent_id,
               int sibling_order,
               float rotation_angle,
               float polar_cx,
               float polar_cy,
               float polar_r0,
               float polar_a0,
               float polar_r1,
               float polar_a1,
               float polar_r2,
               float polar_a2,
               float polar_r3,
               float polar_a3) {
                DocNodeRect rect{x, y, w, h};
                DocNodePayload p{};
                p.type = static_cast<DocNodeType>(type);
                std::snprintf(p.label,     DR_LABEL_MAX, "%s", label.c_str());
                std::snprintf(p.value_str, DR_VALUE_MAX, "%s", value_str.c_str());
                auto fill4 = [](float* dst, py::array_t<float>& arr) {
                    auto buf = arr.request();
                    auto* data = static_cast<const float*>(buf.ptr);
                    py::ssize_t n = (buf.ndim > 0) ? buf.shape[0] : 0;
                    for (int i = 0; i < 4; ++i) dst[i] = (i < (int)n) ? data[i] : 0.f;
                };
                fill4(p.bg_rgba,     bg_rgba);
                fill4(p.fg_rgba,     fg_rgba);
                fill4(p.border_rgba, border_rgba);
                fill4(p.accent_rgba, accent_rgba);
                p.corner_radius  = corner_radius;
                p.font_scale     = font_scale;
                p.value_norm     = value_norm;
                p.icon_id        = icon_id;
                p.border_px      = border_px;
                p.rotation_angle = rotation_angle;
                p.polar_cx = polar_cx;
                p.polar_cy = polar_cy;
                p.polar_r0 = polar_r0;
                p.polar_a0 = polar_a0;
                p.polar_r1 = polar_r1;
                p.polar_a1 = polar_a1;
                p.polar_r2 = polar_r2;
                p.polar_a2 = polar_a2;
                p.polar_r3 = polar_r3;
                p.polar_a3 = polar_a3;
                dr_submit_node_ex(self.st, node_id, parent_id, sibling_order, rect, &p);
            },
            py::arg("node_id"),
            py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"),
            py::arg("type"),
            py::arg("label")         = "",
            py::arg("value_str")     = "",
            py::arg("bg_rgba"),
            py::arg("fg_rgba"),
            py::arg("border_rgba"),
            py::arg("accent_rgba"),
            py::arg("corner_radius") = 2.0f,
            py::arg("font_scale")    = 1.0f,
            py::arg("value_norm")    = 0.0f,
            py::arg("icon_id")       = -1,
            py::arg("border_px")     = 1,
            py::arg("parent_id")     = 0,
            py::arg("sibling_order") = -1,
            py::arg("rotation_angle") = 0.0f,
            py::arg("polar_cx") = 0.0f,
            py::arg("polar_cy") = 0.0f,
            py::arg("polar_r0") = 0.0f,
            py::arg("polar_a0") = 0.0f,
            py::arg("polar_r1") = 0.0f,
            py::arg("polar_a1") = 0.0f,
            py::arg("polar_r2") = 0.0f,
            py::arg("polar_a2") = 0.0f,
            py::arg("polar_r3") = 0.0f,
            py::arg("polar_a3") = 0.0f,
            "Drop one node payload into the render FIFO.")
        .def("remove_node",
            [](PyDocRenderer& self, uint64_t node_id) {
                DocNodeRect r{0, 0, 0, 0};
                dr_submit_node(self.st, node_id, r, nullptr);
            }, py::arg("node_id"), "Remove a node from the live set.")
        .def("mark_dirty",
            [](PyDocRenderer& self, uint64_t node_id) {
                dr_mark_dirty(self.st, node_id);
            }, py::arg("node_id"), "Force a node to re-render next flush.")
        .def("clear",
            [](PyDocRenderer& self) { dr_clear(self.st); },
            "Remove all nodes and clear the composite buffer.")
        .def("flush",
            [](PyDocRenderer& self) { dr_flush(self.st); },
            "Block until the FIFO is empty and all tiles are rendered.")
        .def("composite",
            [](PyDocRenderer& self) -> py::array_t<uint8_t> {
                int w = dr_width(self.st), h = dr_height(self.st);
                py::array_t<uint8_t> out({h, w, 4});
                dr_composite(self.st, out.mutable_data());
                return out;
            },
            "Composite all live tiles; returns (H, W, 4) uint8 RGBA array.")
        .def_property_readonly("composite_dirty",
            [](PyDocRenderer& self) -> bool {
                return dr_composite_dirty(self.st) != 0;
            },
            "True if composite output has changed since last composite() call.")
        .def_property_readonly("width",
            [](PyDocRenderer& self){ return dr_width(self.st); })
        .def_property_readonly("height",
            [](PyDocRenderer& self){ return dr_height(self.st); })
        .def_property_readonly("node_count",
            [](PyDocRenderer& self){ return dr_node_count(self.st); })
        .def_property_readonly("queue_depth",
            [](PyDocRenderer& self){ return dr_queue_depth(self.st); });

    /* ── BaseRasterizer (software 3D global) ───────────────────────────── */

    struct PyBaseRasterizer {
        BaseRasterizerState* st;
        // Hold material chunks alive for the lifetime of the rasterizer.
        std::vector<float> pbr, phong, enamel, texstack;
        PyBaseRasterizer(int w, int h, int tile)
            : st(br_create(w, h, tile)) {
            if (!st) throw std::runtime_error("br_create failed");
        }
        ~PyBaseRasterizer() { if (st) { br_destroy(st); st = nullptr; } }
    };

    py::class_<PyBaseRasterizer>(m, "BaseRasterizer",
        R"doc(
Tile-parallel software rasterizer (3D-C global default).

Deposits triangles directly into a CPU RGBA framebuffer; readback returns
a (H, W, 4) uint8 numpy array suitable for pygame blit or GL texture upload.
)doc")
        .def(py::init<int, int, int>(),
            py::arg("width"), py::arg("height"), py::arg("tile_size") = 16,
            "Create a software rasterizer of the given pixel dimensions.")
        .def("clear",
            [](PyBaseRasterizer& self, float r, float g, float b, float a) {
                br_clear(self.st, r, g, b, a);
            },
            py::arg("r") = 0.0f, py::arg("g") = 0.0f,
            py::arg("b") = 0.0f, py::arg("a") = 0.0f,
            "Clear the colour buffer (and depth) to the given RGBA.")
        .def("set_pbr_chunk",
            [](PyBaseRasterizer& self, py::array_t<float> data) {
                auto info = data.request();
                if (info.ndim != 2 || info.shape[1] != 16)
                    throw std::runtime_error("set_pbr_chunk: expected (N, 16) float32 array");
                int n = (int)info.shape[0];
                self.pbr.assign(static_cast<const float*>(info.ptr),
                                static_cast<const float*>(info.ptr) + (size_t)n * 16);
                br_set_pbr_chunk(self.st, self.pbr.data(), n);
            },
            py::arg("data"),
            "Upload (N, 16) float32 PBR material records.")
        .def("set_phong_chunk",
            [](PyBaseRasterizer& self, py::array_t<float> data) {
                auto info = data.request();
                if (info.ndim != 2 || info.shape[1] != 8)
                    throw std::runtime_error("set_phong_chunk: expected (N, 8) float32 array");
                int n = (int)info.shape[0];
                self.phong.assign(static_cast<const float*>(info.ptr),
                                  static_cast<const float*>(info.ptr) + (size_t)n * 8);
                br_set_phong_chunk(self.st, self.phong.data(), n);
            },
            py::arg("data"),
            "Upload (N, 8) float32 Phong material records.")
        .def("set_enamel_chunk",
            [](PyBaseRasterizer& self, py::array_t<float> data) {
                auto info = data.request();
                if (info.ndim != 2 || info.shape[1] != 8)
                    throw std::runtime_error("set_enamel_chunk: expected (N, 8) float32 array");
                int n = (int)info.shape[0];
                self.enamel.assign(static_cast<const float*>(info.ptr),
                                   static_cast<const float*>(info.ptr) + (size_t)n * 8);
                br_set_enamel_chunk(self.st, self.enamel.data(), n);
            },
            py::arg("data"),
            "Upload (N, 8) float32 enamel material records.")
        .def("set_texture_stack_chunk",
            [](PyBaseRasterizer& self, py::array_t<float> data) {
                auto info = data.request();
                if (info.ndim != 2 || info.shape[1] != 16)
                    throw std::runtime_error("set_texture_stack_chunk: expected (N, 16) float32 array");
                int n = (int)info.shape[0];
                self.texstack.assign(static_cast<const float*>(info.ptr),
                                     static_cast<const float*>(info.ptr) + (size_t)n * 16);
                br_set_texture_stack_chunk(self.st, self.texstack.data(), n);
            },
            py::arg("data"),
            "Upload (N, 16) float32 cold UV/depth/remit texture-stack records.")
        .def("set_emit_uv_texture_array",
            [](PyBaseRasterizer& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> data) {
                auto info = data.request();
                if (info.ndim != 4 || info.shape[3] != 4)
                    throw std::runtime_error("set_emit_uv_texture_array: expected (layers, height, width, 4) uint8 array");
                br_set_emit_uv_texture_array(self.st,
                    static_cast<const uint8_t*>(info.ptr),
                    (int)info.shape[2],
                    (int)info.shape[1],
                    (int)info.shape[0]);
            },
            py::arg("data"),
            "Upload RGBA8 emission UV texture array as (layers, height, width, 4).")
        .def("set_color_uv_texture_array",
            [](PyBaseRasterizer& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> data) {
                auto info = data.request();
                if (info.ndim != 4 || info.shape[3] != 4)
                    throw std::runtime_error("set_color_uv_texture_array: expected (layers, height, width, 4) uint8 array");
                br_set_color_uv_texture_array(self.st,
                    static_cast<const uint8_t*>(info.ptr),
                    (int)info.shape[2],
                    (int)info.shape[1],
                    (int)info.shape[0]);
            },
            py::arg("data"),
            "Upload RGBA8 color override UV texture array as (layers, height, width, 4).")
        .def("set_depth_uv_texture_array",
            [](PyBaseRasterizer& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> data) {
                auto info = data.request();
                if (info.ndim != 4 || info.shape[3] != 4)
                    throw std::runtime_error("set_depth_uv_texture_array: expected (layers, height, width, 4) uint8 array");
                br_set_depth_uv_texture_array(self.st,
                    static_cast<const uint8_t*>(info.ptr),
                    (int)info.shape[2],
                    (int)info.shape[1],
                    (int)info.shape[0]);
            },
            py::arg("data"),
            "Upload RGBA8 depth/thickness UV texture array as (layers, height, width, 4).")
        .def("set_remit_uv_texture_array",
            [](PyBaseRasterizer& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> data) {
                auto info = data.request();
                if (info.ndim != 4 || info.shape[3] != 4)
                    throw std::runtime_error("set_remit_uv_texture_array: expected (layers, height, width, 4) uint8 array");
                br_set_remit_uv_texture_array(self.st,
                    static_cast<const uint8_t*>(info.ptr),
                    (int)info.shape[2],
                    (int)info.shape[1],
                    (int)info.shape[0]);
            },
            py::arg("data"),
            "Upload RGBA8 simple reemission UV texture array as (layers, height, width, 4).")
        .def("set_scene",
            [](PyBaseRasterizer& self,
               py::array_t<float> light_v,
               py::array_t<float> scene_rgb,
               float scene_indirect) {
                auto lv = light_v.request();
                auto sr = scene_rgb.request();
                if (lv.size < 3 || sr.size < 3)
                    throw std::runtime_error("set_scene: light_v and scene_rgb must have 3 floats");
                br_set_scene(self.st,
                             static_cast<const float*>(lv.ptr),
                             static_cast<const float*>(sr.ptr),
                             scene_indirect);
            },
            py::arg("light_v"), py::arg("scene_rgb"),
            py::arg("scene_indirect") = 0.2f,
            "Set scene illumination (vec3 light dir in view space, vec3 tint, indirect ratio).")
        .def("set_lights",
            [](PyBaseRasterizer& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> dirs,
               py::array_t<float, py::array::c_style | py::array::forcecast> colors,
               py::array_t<float, py::array::c_style | py::array::forcecast> intens) {
                auto d = dirs.request();
                auto c = colors.request();
                auto in = intens.request();
                if (d.ndim != 2 || d.shape[1] != 3)
                    throw std::runtime_error("set_lights: dirs must be (N, 3) float32");
                if (c.ndim != 2 || c.shape[1] != 3 || c.shape[0] != d.shape[0])
                    throw std::runtime_error("set_lights: colors must be (N, 3) float32 with same N as dirs");
                if (in.ndim != 1 || in.shape[0] != d.shape[0])
                    throw std::runtime_error("set_lights: intens must be (N,) float32 with same N as dirs");
                br_set_lights(self.st,
                              (int)d.shape[0],
                              static_cast<const float*>(d.ptr),
                              static_cast<const float*>(c.ptr),
                              static_cast<const float*>(in.ptr));
            },
            py::arg("dirs"), py::arg("colors"), py::arg("intens"),
            "Push N directional lights as parallel arrays (no averaging).")
        .def("set_groups",
            [](PyBaseRasterizer& self,
               py::array_t<int,   py::array::c_style | py::array::forcecast> group_ids,
               py::array_t<int,   py::array::c_style | py::array::forcecast> mat_ids,
               py::array_t<int,   py::array::c_style | py::array::forcecast> tri_offsets,
               py::array_t<int,   py::array::c_style | py::array::forcecast> tri_counts,
               py::array_t<float, py::array::c_style | py::array::forcecast> mvs,
               py::array_t<int,   py::array::c_style | py::array::forcecast> dirty) {
                // Parallel-array form to keep numpy on the host side; the
                // engine assembles BRGroup records and forwards.  Centroid
                // caching, dirty-aware refresh, and emitter→light derivation
                // all happen inside the kernel — the host owns no light
                // state of its own.
                auto gi = group_ids.request();
                auto mi = mat_ids.request();
                auto to = tri_offsets.request();
                auto tc = tri_counts.request();
                auto mv = mvs.request();
                auto dy = dirty.request();
                int n = (int)gi.shape[0];
                if (mi.shape[0] != n || to.shape[0] != n ||
                    tc.shape[0] != n || dy.shape[0] != n)
                    throw std::runtime_error("set_groups: scalar arrays must share length");
                if (mv.ndim != 2 || mv.shape[0] != n || mv.shape[1] != 16)
                    throw std::runtime_error("set_groups: mvs must be (N, 16) float32 column-major");

                std::vector<BRGroup> groups((size_t)n);
                const int* gip = static_cast<const int*>(gi.ptr);
                const int* mip = static_cast<const int*>(mi.ptr);
                const int* top = static_cast<const int*>(to.ptr);
                const int* tcp = static_cast<const int*>(tc.ptr);
                const int* dyp = static_cast<const int*>(dy.ptr);
                const float* mvp_ = static_cast<const float*>(mv.ptr);
                for (int i = 0; i < n; ++i) {
                    BRGroup& g = groups[(size_t)i];
                    g.group_id   = gip[i];
                    g.mat_id     = mip[i];
                    g.tri_offset = top[i];
                    g.tri_count  = tcp[i];
                    g.dirty      = dyp[i];
                    std::memcpy(g.mv, mvp_ + i * 16, sizeof(float) * 16);
                }
                br_set_groups(self.st, n, groups.data());
            },
            py::arg("group_ids"), py::arg("mat_ids"),
            py::arg("tri_offsets"), py::arg("tri_counts"),
            py::arg("mvs"), py::arg("dirty"),
            "Declare object groups for the next render.  See base_rasterizer.h "
            "for the BR_DIRTY_* bitmask semantics.  Cache lives in the kernel "
            "and persists across render() calls.")
        .def("set_max_lights",
            [](PyBaseRasterizer& self, int max_lights) {
                br_set_max_lights(self.st, max_lights);
            },
            py::arg("max_lights"),
            "Cap the number of cluster-lights emitted per frame from the "
            "group cache.  Clamped to [1, MAX_LIGHTS=100].")
        .def("set_specular_enabled",
            [](PyBaseRasterizer& self, bool enabled) {
                br_set_specular_enabled(self.st, enabled ? 1 : 0);
            },
            py::arg("enabled"),
            "Enable/disable Phong specular highlights from emitter-derived lights.")
        .def("set_emission_direct_enabled",
            [](PyBaseRasterizer& self, bool enabled) {
                br_set_emission_direct_enabled(self.st, enabled ? 1 : 0);
            },
            py::arg("enabled"),
            "Enable/disable direct texture-stack emission coupling.")
        .def("set_light_calibration",
            [](PyBaseRasterizer& self, float factor) {
                br_set_light_calibration(self.st, factor);
            },
            py::arg("factor"),
            "Set a global scalar applied to all emitter-derived light intensities.")
        .def("set_cat_ccm_matrix",
            [](PyBaseRasterizer& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> matrix) {
                auto m = matrix.request();
                if (!((m.ndim == 2 && m.shape[0] == 3 && m.shape[1] == 3) ||
                      (m.ndim == 1 && m.shape[0] == 9))) {
                    throw std::runtime_error("set_cat_ccm_matrix: matrix must be shape (3,3) float32");
                }
                const float* p = static_cast<const float*>(m.ptr);
                br_set_cat_ccm_matrix(self.st, p);
            },
            py::arg("matrix"),
            "Set 3x3 CAT/CCM matrix applied to shaded linear RGB.")
        .def("render",
            [](PyBaseRasterizer& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> verts_view,
               py::array_t<int,   py::array::c_style | py::array::forcecast> mat_ids,
               py::array_t<float, py::array::c_style | py::array::forcecast> mvp) {
                auto vv = verts_view.request();
                auto mi = mat_ids.request();
                auto mp = mvp.request();
                if (vv.ndim != 2 || vv.shape[1] != 6)
                    throw std::runtime_error("render: verts_view must be (n_tris*3, 6) float32");
                if (mp.size != 16)
                    throw std::runtime_error("render: mvp must have 16 float32 elements");
                int n_tris = (int)mi.shape[0];
                if (vv.shape[0] != (py::ssize_t)n_tris * 3)
                    throw std::runtime_error("render: verts_view rows must equal n_tris*3");
                br_render(self.st,
                          static_cast<const float*>(vv.ptr),
                          static_cast<const int*>(mi.ptr),
                          n_tris,
                          static_cast<const float*>(mp.ptr));
            },
            py::arg("verts_view"), py::arg("mat_ids"), py::arg("mvp"),
            "Project, bin, and shade n_tris triangles into the framebuffer.")
        .def("render_textured",
            [](PyBaseRasterizer& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> verts_view,
               py::array_t<int,   py::array::c_style | py::array::forcecast> mat_ids,
               py::array_t<float, py::array::c_style | py::array::forcecast> mvp) {
                auto vv = verts_view.request();
                auto mi = mat_ids.request();
                auto mp = mvp.request();
                if (vv.ndim != 2 || vv.shape[1] != 8)
                    throw std::runtime_error("render_textured: verts_view must be (n_tris*3, 8) float32");
                if (mp.size != 16)
                    throw std::runtime_error("render_textured: mvp must have 16 float32 elements");
                int n_tris = (int)mi.shape[0];
                if (vv.shape[0] != (py::ssize_t)n_tris * 3)
                    throw std::runtime_error("render_textured: verts_view rows must equal n_tris*3");
                br_render_textured(self.st,
                          static_cast<const float*>(vv.ptr),
                          static_cast<const int*>(mi.ptr),
                          n_tris,
                          static_cast<const float*>(mp.ptr));
            },
            py::arg("verts_view"), py::arg("mat_ids"), py::arg("mvp"),
            "Project, bin, and shade n_tris textured triangles into the framebuffer.")
        .def("readback_u8",
            [](PyBaseRasterizer& self) -> py::array_t<uint8_t> {
                int w = br_width(self.st), h = br_height(self.st);
                py::array_t<uint8_t> out({h, w, 4});
                br_readback_u8(self.st, out.mutable_data());
                return out;
            },
            "Linear clamp/quantized (H, W, 4) uint8 RGBA readback.")
        .def("readback_f32",
            [](PyBaseRasterizer& self) -> py::array_t<float> {
                int w = br_width(self.st), h = br_height(self.st);
                py::array_t<float> out({h, w, 4});
                br_readback_f32(self.st, out.mutable_data());
                return out;
            },
            "Linear (H, W, 4) float32 RGBA readback.")
        .def("readback_f32_view",
            [](PyBaseRasterizer& self) -> py::array {
                int w = br_width(self.st), h = br_height(self.st);
                const float* ptr = br_readback_f32_ptr(self.st);
                if (!ptr) {
                    throw std::runtime_error("readback_f32_view: null framebuffer pointer");
                }
                return py::array(
                    py::dtype::of<float>(),
                    {h, w, 4},
                    {sizeof(float) * w * 4, sizeof(float) * 4, sizeof(float)},
                    ptr,
                    py::cast(&self, py::return_value_policy::reference)
                );
            },
            "Zero-copy linear (H, W, 4) float32 framebuffer view.")
        .def("light_count",
            [](PyBaseRasterizer& self) {
                return br_light_count(self.st);
            },
            "Number of internally derived emissive area lights from the latest render call.")
        .def("readback_lights",
            [](PyBaseRasterizer& self) {
                int n = br_light_count(self.st);
                py::array_t<float> pos({n, 3});
                py::array_t<float> col({n, 3});
                py::array_t<float> inten({n});
                py::array_t<int> gids({n});
                br_readback_lights(
                    self.st,
                    n > 0 ? static_cast<float*>(pos.mutable_data()) : nullptr,
                    n > 0 ? static_cast<float*>(col.mutable_data()) : nullptr,
                    n > 0 ? static_cast<float*>(inten.mutable_data()) : nullptr,
                    n > 0 ? static_cast<int*>(gids.mutable_data()) : nullptr,
                    n
                );
                py::dict out;
                out["positions"] = pos;
                out["colors"] = col;
                out["intensities"] = inten;
                out["group_ids"] = gids;
                return out;
            },
            "Read internally derived emissive area lights from the latest render call.")
        .def_property_readonly("width",
            [](PyBaseRasterizer& self){ return br_width(self.st); })
        .def_property_readonly("height",
            [](PyBaseRasterizer& self){ return br_height(self.st); });

    /* ── TileOverlap ────────────────────────────────────────────────────────── */

    /* tile_overlap_compute(tile_corners, item_rects) -> dict
     *
     * Exact SAT overlap detection: every (tile, item) pair is tested in
     * parallel across the thread pool.  No discretisation error.
     *
     * Parameters
     * ----------
     * tile_corners : float32 (N, 4, 2)  — 4 world-space corners per tile (metres).
     * item_rects   : float32 (M, 4)     — (x, y, w, h) per item (metres).
     *
     * Returns
     * -------
     * dict with keys:
     *   'status'        int8   (N,)  — TO_CLEAR=1, TO_OCCUPIED=2, TO_COLLISION=3
     *   'overlap_counts' int32  (N,)  — number of items overlapping each tile
     */
    m.def("tile_overlap_compute",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> corners_arr,
           py::array_t<float, py::array::c_style | py::array::forcecast> rects_arr) -> py::dict {

            auto ca = corners_arr.request();
            auto ra = rects_arr.request();

            /* Accept (N,4,2) or (N,8) for corners */
            int n_tiles = 0;
            if (ca.ndim == 3) {
                if (ca.shape[1] != 4 || ca.shape[2] != 2)
                    throw std::runtime_error(
                        "tile_overlap_compute: tile_corners must be (N,4,2) float32");
                n_tiles = (int)ca.shape[0];
            } else if (ca.ndim == 2) {
                if (ca.shape[1] != 8)
                    throw std::runtime_error(
                        "tile_overlap_compute: tile_corners must be (N,8) float32");
                n_tiles = (int)ca.shape[0];
            } else {
                throw std::runtime_error(
                    "tile_overlap_compute: tile_corners must be (N,4,2) or (N,8) float32");
            }

            /* Accept (M,4) for item_rects */
            if (ra.ndim != 2 || ra.shape[1] != 4)
                throw std::runtime_error(
                    "tile_overlap_compute: item_rects must be (M,4) float32");
            int n_items = (int)ra.shape[0];

            const float* corners_ptr = static_cast<const float*>(ca.ptr);
            const float* rects_ptr   = static_cast<const float*>(ra.ptr);

            /* Synthetic sequential tile IDs (0..N-1) */
            std::vector<int> tile_ids(n_tiles);
            for (int i = 0; i < n_tiles; ++i) tile_ids[i] = i;

            std::vector<int> item_ids(n_items);
            for (int i = 0; i < n_items; ++i) item_ids[i] = i;

            TileOverlapState* st = to_create();
            to_set_tiles(st, corners_ptr, tile_ids.data(), n_tiles);
            to_set_items(st, rects_ptr,   item_ids.data(), n_items);

            py::array_t<int8_t> status(n_tiles);
            py::array_t<int>    counts(n_tiles);
            std::vector<int>    echo_ids(n_tiles);

            to_compute(st,
                       status.mutable_data(),
                       counts.mutable_data(),
                       echo_ids.data());
            to_destroy(st);

            py::dict result;
            result["status"]         = status;
            result["overlap_counts"] = counts;
            return result;
        },
        py::arg("tile_corners"),
        py::arg("item_rects"),
        R"doc(
Parallel tile-item overlap detection using the Separating Axis Theorem (SAT).

Exact convex-quad vs AABB intersection test for every (tile, item) pair,
parallelised across the hardware thread pool.  For a bounded floor mesh
(hundreds of tiles, tens of items) the full NxM sweep of dot-product
equations across the mesh is trivially cheap with zero discretisation error.

Parameters
----------
tile_corners : float32 (N, 4, 2) or (N, 8)
    Four world-space corner points per tile, in metres.  Any winding order.
item_rects : float32 (M, 4)
    Axis-aligned item footprints as (x, y, w, h) in world metres.

Returns
-------
dict
    'status'         : int8  (N,) — 1=CLEAR, 2=OCCUPIED (1 item), 3=COLLISION (2+ items)
    'overlap_counts' : int32 (N,) — exact count of items overlapping each tile
)doc");

    m.def("analyze_emitter_rgba8_layers",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgba,
           float active_threshold) {
            auto info = rgba.request();
            if (info.ndim != 4 || info.shape[3] != 4)
                throw std::runtime_error("analyze_emitter_rgba8_layers: expected (layers, height, width, 4) uint8");
            int layers = (int)info.shape[0];
            int height = (int)info.shape[1];
            int width  = (int)info.shape[2];
            py::array_t<float> out({layers, 16});
            int rc = eak_analyze_rgba8_layers(
                static_cast<const uint8_t*>(info.ptr),
                width, height, layers, active_threshold,
                reinterpret_cast<EAKMetrics*>(out.mutable_data()));
            if (rc != EAK_OK)
                throw std::runtime_error("analyze_emitter_rgba8_layers failed: " + std::to_string(rc));
            return out;
        },
        py::arg("rgba_layers"),
        py::arg("active_threshold") = 0.0f,
        R"doc(
Reduce RGBA8 emitter/scrim texture layers into family-of-angles metrics.

Input shape:  (layers, height, width, 4) uint8
Output shape: (layers, 16) float32
Columns:
  0 flux_scale, 1 active_fraction, 2 active_density,
  3 centroid_u, 4 centroid_v, 5 axis_u, 6 axis_v,
  7 spread_major, 8 spread_minor, 9 cone_cos, 10 cone_solid_angle.
)doc");

    /* ── PlayerClipEngine ─────────────────────────────────────────────────── */
    py::class_<PlayerClipEngine>(m, "PlayerClipEngine",
        "Axis-aligned slab player collision engine with double-buffered state.\n\n"
        "Thread-safe read via read_state(); tick() is single-threaded (game loop).")
        .def(py::init<>())
        .def("set_slabs",
            [](PlayerClipEngine& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                auto info = arr.request();
                if (info.ndim != 2 || info.shape[1] != 8)
                    throw std::runtime_error("set_slabs: expected float32 (N, 8) array");
                int n = static_cast<int>(info.shape[0]);
                const float* data = static_cast<const float*>(info.ptr);
                std::vector<PlayerWallSlab> slabs(n);
                for (int i = 0; i < n; ++i) {
                    const float* row = data + i * 8;
                    slabs[i].axis         = static_cast<int>(row[0]);
                    slabs[i].pos          = row[1];
                    slabs[i].normal_sign  = row[2];
                    slabs[i].range_lo[0]  = row[3];
                    slabs[i].range_lo[1]  = row[4];
                    slabs[i].range_hi[0]  = row[5];
                    slabs[i].range_hi[1]  = row[6];
                }
                self.set_slabs(slabs);
            },
            py::arg("slabs"),
            "Set wall slabs from (N, 8) float32 array.\n"
            "Columns: axis, pos, normal_sign, rlo0, rlo1, rhi0, rhi1, reserved.")
        .def("tick",
            [](PlayerClipEngine& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> pos,
               py::array_t<float, py::array::c_style | py::array::forcecast> vel,
               float floor_z, float ceil_z, float radius, float yaw_rad,
               float dt) {
                auto pi = pos.request();
                auto vi = vel.request();
                if (pi.size < 3) throw std::runtime_error("tick: pos must have ≥3 elements");
                if (vi.size < 3) throw std::runtime_error("tick: vel must have ≥3 elements");
                float p[3], v[3];
                std::memcpy(p, pi.ptr, 3 * sizeof(float));
                std::memcpy(v, vi.ptr, 3 * sizeof(float));
                uint32_t flags = self.tick(p, v, floor_z, ceil_z, radius, yaw_rad, dt);
                py::array_t<float> out_pos(3), out_vel(3);
                std::memcpy(out_pos.mutable_data(), p, 3 * sizeof(float));
                std::memcpy(out_vel.mutable_data(), v, 3 * sizeof(float));
                return py::make_tuple(out_pos, out_vel, static_cast<uint32_t>(flags));
            },
            py::arg("pos"), py::arg("vel"),
            py::arg("floor_z") = 0.0f, py::arg("ceil_z") = 4.0f,
            py::arg("radius") = 0.25f, py::arg("yaw_rad") = 0.0f,
            py::arg("dt") = 0.016f,
            "Advance one tick.  Returns (corrected_pos, corrected_vel, clip_flags).")
        .def("configure_state_tensor",
            [](PlayerClipEngine& self, int capacity, int stride) {
                self.configure_state_tensor(capacity, stride);
            },
            py::arg("capacity"),
            py::arg("stride") = PlayerClipEngine::STATE_TENSOR_MIN_STRIDE,
            "Preallocate the owned player state tensor ring buffer.")
        .def("state_tensor_view",
            [](PlayerClipEngine& self) -> py::array {
                int cap = self.state_tensor_capacity();
                int stride = self.state_tensor_stride();
                const float* ptr = self.state_tensor_data();
                if (!ptr || cap <= 0 || stride <= 0) {
                    return py::array(
                        py::dtype::of<float>(),
                        {0, PlayerClipEngine::STATE_TENSOR_MIN_STRIDE},
                        {sizeof(float) * PlayerClipEngine::STATE_TENSOR_MIN_STRIDE, sizeof(float)},
                        nullptr,
                        py::cast(&self, py::return_value_policy::reference)
                    );
                }
                return py::array(
                    py::dtype::of<float>(),
                    {cap, stride},
                    {sizeof(float) * stride, sizeof(float)},
                    const_cast<float*>(ptr),
                    py::cast(&self, py::return_value_policy::reference)
                );
            },
            "Zero-copy float32 ring-buffer view of player state history.")
        .def_property_readonly("state_tensor_cursor",
            [](const PlayerClipEngine& self) {
                return self.state_tensor_cursor();
            })
        .def("read_state",
            [](const PlayerClipEngine& self) {
                PlayerStateFrame f;
                self.read(&f);
                py::dict d;
                d["px"]         = f.px;
                d["py"]         = f.py;
                d["pz"]         = f.pz;
                d["vx"]         = f.vx;
                d["vy"]         = f.vy;
                d["vz"]         = f.vz;
                d["yaw_rad"]    = f.yaw_rad;
                d["radius"]     = f.radius;
                d["floor_z"]    = f.floor_z;
                d["ceil_z"]     = f.ceil_z;
                d["clip_flags"] = static_cast<uint32_t>(f.clip_flags);
                d["generation"] = static_cast<uint32_t>(f.generation);
                return d;
            },
            "Return last published player state as a dict.  Safe to call from any thread.");

    /* ── Optical Assembly (Phase C) ──────────────────────────────────────── */
    /* OpticalHandlerRole enum — maps to optical_assembly.py roles */
    py::enum_<OpticalHandlerRole>(m, "OpticalHandlerRole", py::arithmetic())
        .value("STOP",         OPTICAL_ROLE_STOP)
        .value("MIRROR",       OPTICAL_ROLE_MIRROR)
        .value("REFRACTOR",    OPTICAL_ROLE_REFRACTOR)
        .value("FILTER",       OPTICAL_ROLE_FILTER)
        .value("DIFFUSER",     OPTICAL_ROLE_DIFFUSER)
        .value("SENSOR",       OPTICAL_ROLE_SENSOR)
        .value("WAVE_REGION",  OPTICAL_ROLE_WAVE_REGION)
        .export_values();

    /* OpticalRayState — ray transport state through optical element */
    py::class_<OpticalRayState>(m, "OpticalRayState",
        "Optical ray state: amplitude, phase, wavelength during element traversal.")
        .def(py::init<>())
        .def_readwrite("amplitude_real", &OpticalRayState::amplitude_real)
        .def_readwrite("amplitude_imag", &OpticalRayState::amplitude_imag)
        .def_readwrite("phase_error", &OpticalRayState::phase_error)
        .def_readwrite("wavelength_m", &OpticalRayState::wavelength_m)
        .def_readwrite("cumulative_path_length_m", &OpticalRayState::cumulative_path_length_m)
        .def_readwrite("bounce_count", &OpticalRayState::bounce_count)
        .def_readwrite("is_active", &OpticalRayState::is_active)
        .def_readwrite("hit_sensor", &OpticalRayState::hit_sensor);

    /* OpticalGeometry — surface shape and curvature */
    py::class_<OpticalGeometry>(m, "OpticalGeometry",
        "Optical element geometry: position, diameter, curvature, aspheric terms.")
        .def(py::init<>())
        .def_readwrite("center_z_m", &OpticalGeometry::center_z_m)
        .def_readwrite("diameter_m", &OpticalGeometry::diameter_m)
        .def_readwrite("radius_m", &OpticalGeometry::radius_m)
        .def_readwrite("radius_of_curvature_m", &OpticalGeometry::radius_of_curvature_m)
        .def_readwrite("conic_k", &OpticalGeometry::conic_k)
        .def_readwrite("aspheric_a4", &OpticalGeometry::aspheric_a4)
        .def_readwrite("aspheric_a6", &OpticalGeometry::aspheric_a6)
        .def_readwrite("surface_roughness_m", &OpticalGeometry::surface_roughness_m);

    /* OpticalMaterial — refractive index, absorption, thermal properties */
    py::class_<OpticalMaterial>(m, "OpticalMaterial",
        "Optical material: refractive index, absorption, thermal dependence.")
        .def(py::init<>())
        .def_readwrite("n_real", &OpticalMaterial::n_real)
        .def_readwrite("absorption_coeff_per_m", &OpticalMaterial::absorption_coeff_per_m)
        .def_readwrite("fresnel_r_amplitude", &OpticalMaterial::fresnel_r_amplitude)
        .def_readwrite("fresnel_r_phase", &OpticalMaterial::fresnel_r_phase)
        .def_readwrite("dn_dT_per_K", &OpticalMaterial::dn_dT_per_K);

    /* OpticalHandlerResult — output from handler processing */
    py::class_<OpticalHandlerResult>(m, "OpticalHandlerResult",
        "Result from optical handler: status and event counters.")
        .def(py::init<>())
        .def_readwrite("status", &OpticalHandlerResult::status)
        .def_readwrite("rays_blocked", &OpticalHandlerResult::rays_blocked)
        .def_readwrite("rays_hit_surface", &OpticalHandlerResult::rays_hit_surface)
        .def_readwrite("rays_refracted", &OpticalHandlerResult::rays_refracted)
        .def_readwrite("rays_reflected", &OpticalHandlerResult::rays_reflected)
        .def_readwrite("rays_absorbed", &OpticalHandlerResult::rays_absorbed)
        .def_readwrite("rays_transmitted", &OpticalHandlerResult::rays_transmitted);

    /* ExposureBackendTelemetry — aggregated event and energy counters */
    py::class_<ExposureBackendTelemetry>(m, "ExposureBackendTelemetry",
        "Aggregated telemetry from optical assembly ray transport: event counts and energy.")
        .def(py::init<>())
        .def_readwrite("rays_launched", &ExposureBackendTelemetry::rays_launched)
        .def_readwrite("rays_blocked_by_stop", &ExposureBackendTelemetry::rays_blocked_by_stop)
        .def_readwrite("rays_hit_lens_surface", &ExposureBackendTelemetry::rays_hit_lens_surface)
        .def_readwrite("rays_refracted", &ExposureBackendTelemetry::rays_refracted)
        .def_readwrite("rays_reflected", &ExposureBackendTelemetry::rays_reflected)
        .def_readwrite("rays_total_internal_reflection", &ExposureBackendTelemetry::rays_total_internal_reflection)
        .def_readwrite("rays_entered_wave_region", &ExposureBackendTelemetry::rays_entered_wave_region)
        .def_readwrite("rays_exited_wave_region", &ExposureBackendTelemetry::rays_exited_wave_region)
        .def_readwrite("rays_deposited_sensor", &ExposureBackendTelemetry::rays_deposited_sensor)
        .def_readwrite("rays_out_of_domain", &ExposureBackendTelemetry::rays_out_of_domain)
        .def_readwrite("rays_fell_back_to_full_solve", &ExposureBackendTelemetry::rays_fell_back_to_full_solve)
        .def_readwrite("energy_in", &ExposureBackendTelemetry::energy_in)
        .def_readwrite("energy_out", &ExposureBackendTelemetry::energy_out)
        .def_readwrite("energy_absorbed", &ExposureBackendTelemetry::energy_absorbed)
        .def_readwrite("energy_blocked", &ExposureBackendTelemetry::energy_blocked)
        .def_readwrite("mean_phase_error", &ExposureBackendTelemetry::mean_phase_error)
        .def_readwrite("mean_focus_error", &ExposureBackendTelemetry::mean_focus_error);

    /* ExposureBackendCpp — optical assembly ray transport backend */
    py::class_<ExposureBackendCpp>(m, "ExposureBackendCpp",
        "Backend for optical assembly ray transport: processes rays and aggregates telemetry.")
        .def(py::init<>(),
            "Create backend with no assembly (assembly can be attached in native code).")
        .def_readwrite("assembly", &ExposureBackendCpp::assembly)
        .def_readwrite("telemetry", &ExposureBackendCpp::telemetry)
        .def("create_thin_lens_assembly",
            &ExposureBackendCpp::create_thin_lens_assembly,
            py::arg("focal_length_m"), py::arg("aperture_diameter_m"), py::arg("sensor_distance_m"),
            "Create a simple thin lens optical assembly (aperture + lens + sensor).")
        .def("process_ray",
            &ExposureBackendCpp::process_ray,
            py::arg("ray_in"), py::arg("ray_out"), py::arg("camera_mode"),
            "Process a ray through the optical assembly, return 0 if reaches sensor.")
        .def("get_telemetry",
            &ExposureBackendCpp::get_telemetry,
            "Get accumulated telemetry from ray processing.")
        .def("reset_telemetry",
            &ExposureBackendCpp::reset_telemetry,
            "Reset telemetry counters to zero.");
}
