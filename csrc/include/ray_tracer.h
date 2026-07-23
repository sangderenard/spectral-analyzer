/**
 * ray_tracer.h — C ABI for complex spectral 3D ray tracer.
 *
 * Traces acoustic / EM waves through a triangulated scene in 3D using
 * complex phasors per frequency band.  Each ray carries a complex amplitude
 * vector across all bands simultaneously.
 *
 * Physics
 * -------
 *  - Möller-Trumbore ray-triangle intersection.
 *  - Phase accumulates along each path segment: φ += k_n * Δr,
 *    where k_n = 2π f_n / c.
 *  - Geometric spreading: amplitude scaled by 1/(1 + r_total), so the
 *    1/r far-field behaviour is captured without a singularity at the source.
 *  - Atmospheric absorption: exp(−atmo_abs_n * Δr) per metre per band.
 *  - Reflection: complex reflectance R_n = refl_re_n + j*refl_im_n multiplied
 *    onto the amplitude at each surface hit.
 *  - Diffuse scatter: at each hit, with probability diffusion[tri], the
 *    reflected direction is a cosine-weighted random hemisphere sample;
 *    otherwise specular reflection.
 *  - Ray directions sampled from a Fibonacci sphere for quasi-uniform coverage.
 *
 * Output — segment buffer
 * -----------------------
 * One segment record per (source, ray, bounce, band) that survives the
 * amplitude threshold.  Segments are written in order of emission, then sorted
 * by cumulative path length within each (source, band) group so the caller can
 * animate a wavefront by drawing the first K records.
 *
 * Segment layout: RT_FLOATS_PER_SEG = 12 floats (48 bytes)
 *   [0..2]   start position  (x0, y0, z0)
 *   [3..5]   end position    (x1, y1, z1)
 *   [6]      source_id       (float cast of source index)
 *   [7]      bounce_idx      (float cast of bounce number, 0=direct)
 *   [8]      freq_band       (float cast of band index)
 *   [9]      amplitude       (|complex amplitude| for this band)
 *   [10]     phase           (arg(complex amplitude) in radians)
 *   [11]     path_length     (cumulative path length at segment START, metres)
 */

#pragma once
#include "serial_kernel.h"  /* SK_API, SK_OK, SK_ERR_* */
#include "bdpt_record.h"
#include "optical_handlers.h"  /* OpticalAssembly for attach API */

#include <stdint.h>

#define RT_FLOATS_PER_SEG     12
#define RT_BYTES_PER_SEG      48

/**
 * Per-call output statistics filled by ray_tracer_trace and related exported
 * functions.  Pass a pointer to a zero-initialised struct; the tracer writes
 * to it on return.  Pass NULL to discard stats.
 *
 * segments_written : records actually placed in the output buffer.
 * segments_dropped : records discarded because the buffer was full (overflow).
 * total_bounces    : sum of successful ray bounces across all sources/rays.
 */
typedef struct {
    int64_t segments_written;
    int64_t segments_dropped;
    int64_t total_bounces;
} RtTraceStats;

/* Forward declaration — full definition follows below. */
typedef struct RayTracerState RayTracerState;

/**
 * C function-pointer callback for streaming segment records.
 *
 * @param records    Flat float32 array: n_records × RT_FLOATS_PER_SEG floats.
 * @param n_records  Number of records in this batch.
 * @param user       Caller-provided opaque context pointer.
 * @return           0 to continue tracing; non-zero to request early stop.
 *
 * Thread context: called from the same thread as ray_tracer_trace_callback,
 * single-threaded.  Do not call back into the tracer from within the callback.
 */
typedef int (*RtSegmentCallback)(
    const float* records,
    int          n_records,
    void*        user);

/**
 * Callback-based streaming ray trace.
 *
 * Identical physics to ray_tracer_trace(), but instead of writing into a
 * pre-allocated flat buffer, calls `cb(records, n, user)` whenever the
 * internal flush buffer accumulates flush_records records.  A final flush
 * is performed after the last ray regardless of count.
 *
 * Suitable for scenes where the total segment count is not known in advance
 * or may exceed available RAM — the callback can stream to disk, accumulate
 * into a ChunkedMemorySink, or consume records on-the-fly.
 *
 * @param st            Tracer handle.
 * @param n_sources     Number of sources.
 * @param src_pos       (n_sources, 3) float64 — source positions.
 * @param src_dir       (n_sources, 3) float64 — dominant emit directions.
 * @param src_directivity (n_sources,) float64 — directivity exponent.
 * @param n_rays        Rays per source.
 * @param max_bounces   Maximum reflection bounces.
 * @param min_amplitude Amplitude cutoff.
 * @param seed          RNG seed.
 * @param cb            Callback invoked with each batch of segments.
 * @param user          Opaque pointer forwarded verbatim to cb.
 * @param flush_records Batch size; 0 → default (4096 records).
 * @param stats         [out] Optional stats; may be NULL.
 *
 * @return  SK_OK on success.
 *          SK_ERR_NULL_STATE  if st or cb is NULL.
 *          SK_ERR_DIVERGED    if cb returned non-zero (early termination
 *                             requested); records already delivered are not
 *                             retracted and stats reflect actual work done.
 */
SK_API int ray_tracer_trace_callback(
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
    RtTraceStats*     stats);

/* Multiscale segment format — 14 floats (56 bytes).
 * Extends the base 12-float layout with two extra fields:
 *   [12]  context_id   — index of the scale context (-1 = ambient/coarse)
 *   [13]  scale_type   — RT_SCALE_GEOMETRIC (0) or RT_SCALE_WAVE (1)
 */
#define RT_FLOATS_PER_SEG_MS  14
#define RT_BYTES_PER_SEG_MS   56

/* Scale context type tags (legacy `scale_type`). */
#define RT_SCALE_GEOMETRIC    0   /* coarse BVH geometric optics (cm-scale)     */
#define RT_SCALE_WAVE         1   /* fine wave-optics sub-stepping (µm-scale)   */

/* Camera-visibility policy for image accumulation paths.
 * AS_IS       : no camera LOS cull (legacy behaviour).
 * DIRECT_HIT  : one-shot occlusion test on cam->hit segment.
 * FULL_MARCH  : iterative march through transparent surfaces to find blockers.
 */
#define RT_CAM_VIS_AS_IS        0
#define RT_CAM_VIS_DIRECT_HIT   1
#define RT_CAM_VIS_FULL_MARCH   2

/* How transparent media participates in camera occlusion checks. */
#define RT_CAM_TRANSPARENCY_BLOCK 0
#define RT_CAM_TRANSPARENCY_XRAY  1

/* Scale-context KIND — orthogonal to scale_type, selects the *dispatch path*
 * a ray takes when it enters a region.  RAY (0) is the default and matches
 * legacy behaviour exactly.  Higher kinds hand the ray to specialised
 * handlers (wave solver in field_march.cpp, matrix optics, spline-surface
 * refinement, neural transforms).
 *
 * Parity rule: this enum is mirrored byte-for-byte in the GLSL
 * ScaleContextSSBO (binding 8) — see _GPU_RAY_FIELD_CS / _GPU_SENSOR_CS in
 * demo_pluck_gl.py.  Do not reorder.
 *
 * Stub-passthrough for kinds 3..6 right now: dispatch is wired and the
 * region's `context_kind` reaches the bounce loop, but the implementation is
 * "record entry, continue with default ray transport".  The dispatch *call
 * sites* must land in both backends so we never have to retrofit them.
 */
#define SCALE_CONTEXT_KIND_RAY                 0
#define SCALE_CONTEXT_KIND_WAVE_HELMHOLTZ      1
#define SCALE_CONTEXT_KIND_THIN_LENS_TRANSFORM 2
#define SCALE_CONTEXT_KIND_THICK_LENS_WAVE     3
#define SCALE_CONTEXT_KIND_SPLINE_SURFACE      4
#define SCALE_CONTEXT_KIND_NEURAL_SURFACE      5
#define SCALE_CONTEXT_KIND_NEURAL_VOLUMETRIC   6

/**
 * Scale context descriptor.
 *
 * Registered with ray_tracer_add_scale_context().  When a ray propagation
 * segment intersects a context's sphere (|pt - center| < radius) the tracer
 * automatically switches the propagation kernel for that sub-segment:
 *
 *   RT_SCALE_GEOMETRIC — standard BVH geometric step (default everywhere).
 *   RT_SCALE_WAVE      — sub-step with dt_m step size, wave-accurate phase
 *                        using the context's refractive index (n_real, n_imag).
 *                        Evanescent decay is applied when n_imag > 0.
 *                        Near-field spreading uses 1/(1+r²) instead of 1/(1+r).
 *
 * context_id is assigned sequentially by ray_tracer_add_scale_context and
 * returned in segment field [12] of the MS segment format.
 *
 * The additive fields at the end (context_kind + payload) carry the
 * dispatch-kind enum and an opaque per-region payload (matrix-optics
 * coefficients, spline patch handle, neural-net weight ptr).  Zero-init = RAY.
 */
typedef struct {
    double center[3];     /* world-space centre, metres                      */
    double radius;        /* context trigger radius, metres                  */
    int    scale_type;    /* RT_SCALE_GEOMETRIC or RT_SCALE_WAVE             */
    double dt_m;          /* wave sub-step size in metres (RT_SCALE_WAVE)    */
    int    n_substeps;    /* max sub-steps before forced exit (safety cap)   */
    double n_real;        /* real part of medium refractive index            */
    double n_imag;        /* imaginary part (extinction → absorption per m)  */
    int    context_id;    /* filled by add_scale_context; caller may ignore  */
    /* ── Additive extensions (safe to leave zeroed) ───────────────────── */
    int    context_kind;          /* SCALE_CONTEXT_KIND_*; 0 = legacy RAY    */
    const void* payload;          /* opaque per-region data (kind-specific)  */
    int    payload_size_bytes;    /* deep-copy hint; 0 = caller-owned        */
} RtScaleContext;

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tracer state — build once from scene geometry, trace many times. */
typedef struct RayTracerState RayTracerState;
typedef struct FieldGrid FieldGrid;

/**
 * Build a ray tracer from a triangulated scene.
 *
 * @param n_tri       Number of triangles.
 * @param verts       (n_tri, 3, 3) float64 row-major — triangle vertices.
 * @param normals     (n_tri, 3)    float64 row-major — outward unit normals.
 * @param mat_idx     (n_tri,)         int32 — material index per triangle (row in mat_buf).
 * @param mat_buf     (mat_n_mats * MAX_SPECTRAL_BANDS, 12) float32 row-major —
 *                    flat MatBuf shared with the GLSL backend (see mat_flags.py).
 * @param mat_n_mats  Number of registered materials in mat_buf.
 * @param n_bands     Number of frequency bands.
 * @param freq_hz     (n_bands,) float64 — centre frequencies in Hz.
 * @param speed_m_s   Speed of propagation in m/s (343.0 for air at 20°C).
 * @param atmo_abs    (n_bands,) float64 — atmospheric absorption in Np/m per band.
 * @return            Opaque handle, NULL on allocation failure.
 */
SK_API RayTracerState* ray_tracer_create(
    int             n_tri,
    const double*   verts,
    const double*   normals,
    const int*      mat_idx,
    const float*    mat_buf,
    int             mat_n_mats,
    int             n_bands,
    const double*   freq_hz,
    double          speed_m_s,
    const double*   atmo_abs
);

/** Configure continuous-frequency LUT indirection.
 *
 * lane_lut_index has n_bands entries and is the lookup signal carried by each
 * payload lane. profile_offsets has n_profiles+1 entries into the shared knot
 * and density arrays.  Frequencies are sampled once per root ray and retained
 * through all descendants.  Passing n_profiles=0 disables continuous mode.
 */
SK_API int ray_tracer_set_spectral_luts(
    RayTracerState* st,
    const int* lane_lut_index,
    int n_lanes,
    const int* profile_offsets,
    int n_profiles,
    const double* frequency_knots_hz,
    const double* density,
    int n_knots);

/** Free tracer state.  Safe to call with NULL. */
SK_API void ray_tracer_destroy(RayTracerState* st);

/**
 * Attach (or detach) an OpticalAssembly to the tracer for PIXEL_CONE dispatch.
 *
 * The assembly is NOT owned by the tracer; the caller must ensure it remains
 * valid until the tracer is destroyed or the assembly is detached (pass NULL
 * to detach).  The assembly is invoked per-ray per-band when the registered
 * SENSOR group's CameraSensorDesc.use_optical_handlers is non-zero.
 */
SK_API void ray_tracer_attach_optical_assembly(
    RayTracerState*  st,
    OpticalAssembly* assembly);

/**
 * Return an ASCII table describing RayTracer-owned allocations.
 *
 * Includes persistent buffers with programmatic names, purposes, addresses,
 * used bytes, and reserved bytes. Returned pointer is thread-local storage
 * valid until the next call on the same thread.
 */
SK_API const char* ray_tracer_allocation_table(const RayTracerState* st);

/**
 * Enable/disable periodic native profiling pulses during active tracing.
 *
 * When enabled, the tracer emits progress lines and the allocation table to
 * stderr at roughly period_s cadence from inside active trace loops.
 */
SK_API int ray_tracer_set_profile_pulse(
    RayTracerState* st,
    int             enabled,
    double          period_s
);

/**
 * Trace rays from all sources and write segments into the output buffer.
 *
 * Each source emits n_rays directions sampled from a Fibonacci sphere,
 * weighted by a directivity lobe (cosine^directivity_power).  For each
 * surviving (ray, bounce) pair, one segment record per frequency band is
 * written.
 *
 * @param st               Tracer handle from ray_tracer_create.
 * @param n_sources        Number of sources.
 * @param src_pos          (n_sources, 3) float64 — source positions.
 * @param src_dir          (n_sources, 3) float64 — dominant emit directions.
 * @param src_directivity  (n_sources,)   float64 — directivity exponent (0=omni).
 * @param n_rays           Rays per source (Fibonacci sphere over full sphere).
 * @param max_bounces      Maximum reflection bounces per ray.
 * @param min_amplitude    Stop tracing when max |A| across all bands < this.
 * @param seed             RNG seed for Monte Carlo diffuse scatter.
 * @param out_segs         Output float32 buffer, at least out_cap * RT_FLOATS_PER_SEG
 *                         floats.
 * @param out_cap          Maximum number of segments to write.
 * @param out_count        [out] Number of segments actually written.
 * @return                 SK_OK on success, SK_ERR_NULL_STATE if st/out is NULL.
 */
SK_API int ray_tracer_trace(
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
    int*            out_count
);

/**
 * Integrate ray arrivals into time-domain impulse responses at point receivers.
 *
 * For each ray bounce, any receiver whose aperture sphere contains the hit
 * point captures that arrival.  The complex amplitude (AFTER propagation,
 * BEFORE surface reflection) is binned into the nearest sample bin and
 * accumulated into out_re / out_im with a linear distance falloff.
 *
 * Output layout (row-major):
 *   out_re[si * n_rec * n_bands * n_samples
 *         + ri * n_bands * n_samples
 *         + b  * n_samples
 *         + t]
 * and identically for out_im.  Caller must zero both buffers before calling.
 *
 * @param st               Tracer handle from ray_tracer_create.
 * @param n_sources        Number of sources.
 * @param src_pos          (n_sources, 3) float64 — source positions.
 * @param src_dir          (n_sources, 3) float64 — dominant emit directions.
 * @param src_directivity  (n_sources,)   float64 — directivity exponent (0=omni).
 * @param n_rays           Rays per source.
 * @param max_bounces      Maximum reflection bounces per ray.
 * @param min_amplitude    Ray amplitude cutoff.
 * @param seed             RNG seed.
 * @param n_receivers      Number of point receivers.
 * @param rec_pos          (n_receivers, 3) float64 — receiver positions.
 * @param rec_aperture_r   (n_receivers,)   float64 — capture sphere radii (m).
 * @param speed_m_s        Wave speed in m/s (used to convert path length → delay).
 * @param sample_rate      IR sample rate in Hz.
 * @param n_samples        Number of IR samples per (src, rec, band).
 * @param out_re           float32 (n_sources, n_receivers, n_bands, n_samples) — real part.
 * @param out_im           float32 (n_sources, n_receivers, n_bands, n_samples) — imag part.
 * @return                 SK_OK or SK_ERR_NULL_STATE.
 */
SK_API int ray_tracer_integrate_ir(
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
    float*          out_im
);

/**
 * Integrate ray arrivals into a 2-D energy image from a pinhole camera.
 *
 * Each ray hit point is projected onto the camera image plane.  The
 * per-band amplitude magnitude |A[b]| is splatted additively into the
 * corresponding pixel.  Useful for visualising spatial energy distribution
 * and for offline renders of acoustic / EM field illumination.
 *
 * Output layout: out_image[b * height * width + row * width + col]
 * (Y axis: row 0 = top of image).  Caller must zero the buffer before calling.
 *
 * @param st               Tracer handle.
 * @param n_sources        Number of sources.
 * @param src_pos          (n_sources, 3) float64.
 * @param src_dir          (n_sources, 3) float64.
 * @param src_directivity  (n_sources,)   float64.
 * @param n_rays           Rays per source.
 * @param max_bounces      Maximum reflections.
 * @param min_amplitude    Amplitude cutoff.
 * @param seed             RNG seed.
 * @param cam_pos          (3,) float64 — camera position.
 * @param cam_fwd          (3,) float64 — camera look direction (normalised internally).
 * @param cam_up           (3,) float64 — up hint (orthogonalised internally).
 * @param fov_rad          Full vertical field of view in radians.
 * @param width            Image width in pixels.
 * @param height           Image height in pixels.
 * @param out_image        float32 (n_bands, height, width) — accumulated |amplitude|.
 * @return                 SK_OK or SK_ERR_NULL_STATE.
 */
SK_API int ray_tracer_integrate_image(
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
    float*          out_image
);

/**
 * Packed variant of ray_tracer_integrate_image with per-source ray quotas.
 *
 * src_n_rays is an int32 array of length n_sources. Each source si emits
 * max(src_n_rays[si], 0) rays for this dispatch.
 */
SK_API int ray_tracer_integrate_image_packed(
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
    float*          out_image
);

/**
 * Trace rays and integrate camera energy in the same traversal.
 *
 * out_image is accumulated in place and is never cleared by this function.
 * out_segs may be NULL, and out_cap may be 0, to disable segment capture.
 * If the segment buffer fills, tracing continues and image accumulation
 * continues; out_count reports only the number of segment records stored.
 *
 * This is intended for hot loops that reuse image/segment buffers over many
 * massive ray batches without allocating temporary segment streams.
 */
SK_API int ray_tracer_trace_integrate_image(
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
    int*            out_count
);

/**
 * Configure camera visibility wrappers for image accumulation APIs.
 *
 * These options affect ray_tracer_integrate_image() and
 * ray_tracer_trace_integrate_image() only; bounce transport physics are
 * unchanged.
 *
 * @param camera_vis_mode      RT_CAM_VIS_* policy.
 * @param transparent_mode     RT_CAM_TRANSPARENCY_* policy.
 * @param enable_depth_cull    0 disables depth cull, nonzero enables it.
 * @param depth_cull_m         Positive max camera depth in metres.
 */
SK_API int ray_tracer_set_camera_visibility(
    RayTracerState* st,
    int             camera_vis_mode,
    int             transparent_mode,
    int             enable_depth_cull,
    double          depth_cull_m
);

/**
 * Read full-march field-tracking counters.
 *
 * @param out_steps            Number of full-march segment steps evaluated.
 * @param out_context_entries  Number of scale-context entry hooks evaluated.
 */
SK_API int ray_tracer_get_camera_visibility_stats(
    const RayTracerState* st,
    uint64_t*             out_steps,
    uint64_t*             out_context_entries
);

/**
 * Bind/unbind a full-complex field capture grid for camera/image tracing.
 *
 * When bound, every camera strike can be accumulated as complex spectral data
 * into the grid before any display reduction.
 */
SK_API int ray_tracer_set_field_capture_grid(
    RayTracerState* st,
    FieldGrid*      grid,
    int             take_ownership,
    int             capture_strikes,
    int             max_strikes,
    int             clear_existing
);

/** Clear field-capture buffers (grid and/or strike rows). */
SK_API int ray_tracer_clear_field_capture(
    RayTracerState* st,
    int             clear_grid,
    int             clear_strikes
);

/**
 * Query field-capture layout.
 *
 * strike_stride_floats = 16 + 2*n_bands.
 */
SK_API int ray_tracer_get_field_capture_layout(
    const RayTracerState* st,
    int*                  out_grid_kind,
    int*                  out_n_bands,
    int64_t*              out_n_cells,
    int*                  out_strike_stride_floats,
    int*                  out_n_strikes
);

/** Copy captured complex grid data as interleaved float32 (re,im,...). */
SK_API int ray_tracer_copy_field_capture_grid_reim(
    const RayTracerState* st,
    float*                out_reim,
    int64_t               out_count
);

/** Copy captured strike rows (shape: n_rows × strike_stride_floats). */
SK_API int ray_tracer_copy_field_capture_strikes(
    const RayTracerState* st,
    float*                out_rows,
    int                   out_rows_cap,
    int*                  out_rows_written
);

/**
 * Maximum number of sensor/film slots in one batch upload.
 * Matches Python sensor_film_db.MAX_SENSOR_FILM_SLOTS = 8.
 */
#define MAX_SENSOR_FILM_SLOTS 8

/**
 * Sensor parameters — row layout for the sensor_chunk tensor.
 *
 * Binary-compatible with Python SensorRecord (_pack_=4, 48 floats = 192 bytes,
 * 12 × vec4). Every field group is vec4-aligned for std430 safety.
 *
 * Use reinterpret_cast<const SensorRecord*>(sensor_chunk + sensor_id * stride)
 * in C++ reducers instead of raw float offsets.
 */
#pragma pack(push, 4)
typedef struct SensorRecord {
    /* vec4  0 — optics */
    float focal_mm;
    float f_number;
    float aperture_diam_mm;
    float _optics_pad;
    /* vec4  1 — sensor geometry */
    float sensor_w_mm;
    float sensor_h_mm;
    float pixel_pitch_um;
    float _geom_pad;
    /* vec4  2 — photon transport  (offsets 8-11) */
    float qe_peak;           /* peak quantum efficiency [0,1]       */
    float full_well_e;       /* full-well capacity (electrons)      */
    float read_noise_e;      /* read noise RMS (electrons)          */
    float dark_current_e_s;  /* dark current (e⁻/s)                 */
    /* vec4  3 — digitisation */
    float adc_bits;
    float black_level_adu;
    float white_level_adu;
    float _dither_pad;
    /* vec4  4 — CFA red channel */
    float cfa_r_peak_nm;
    float cfa_r_fwhm_nm;
    float cfa_r_area_frac;
    float _cfa_r_pad;
    /* vec4  5 — CFA green channel */
    float cfa_g_peak_nm;
    float cfa_g_fwhm_nm;
    float cfa_g_area_frac;
    float _cfa_g_pad;
    /* vec4  6 — CFA blue channel */
    float cfa_b_peak_nm;
    float cfa_b_fwhm_nm;
    float cfa_b_area_frac;
    float _cfa_b_pad;
    /* vec4  7 — CFA infrared */
    float cfa_ir_peak_nm;
    float cfa_ir_fwhm_nm;
    float cfa_ir_area_frac;
    float _cfa_ir_pad;
    /* vec4  8 — photon transfer curve (cubic polynomial) */
    float pt_0;
    float pt_1;
    float pt_2;
    float pt_3;
    /* vec4  9 — chromatic aberration */
    float ca_red_x;
    float ca_red_y;
    float ca_blue_x;
    float ca_blue_y;
    /* vec4 10 — lens distortion */
    float dist_k1;
    float dist_k2;
    float dist_k3;
    float focal_length_ratio;
    /* vec4 11 — reserved */
    float _future_0;
    float _future_1;
    float _future_2;
    float _future_3;
} SensorRecord;
#pragma pack(pop)

/**
 * Film parameters — row layout for the film_chunk tensor.
 *
 * Binary-compatible with Python FilmRecord (_pack_=4, 64 floats = 256 bytes,
 * 16 × vec4).
 */
#pragma pack(push, 4)
typedef struct FilmRecord {
    /* vec4  0 — exposure  (offsets 0-3) */
    float iso;
    float exposure_time_s;       /* shutter open time (s)   offset 1 */
    float quantum_efficiency;
    float target_grey_point;
    /* vec4  1-2 — layer 0 tone curve (8 floats) */
    float layer0_shadow_r;
    float layer0_shadow_g;
    float layer0_shadow_b;
    float layer0_light_r;
    float layer0_light_g;
    float layer0_light_b;
    float layer0_shadow_point;
    float layer0_highlight_point;
    /* vec4  3-4 — layer 1 tone curve (8 floats) */
    float layer1_shadow_r;
    float layer1_shadow_g;
    float layer1_shadow_b;
    float layer1_light_r;
    float layer1_light_g;
    float layer1_light_b;
    float layer1_shadow_point;
    float layer1_highlight_point;
    /* vec4  5-13 — layers 2-7 + future (36 floats, matches Python _layer_future_0) */
    float _layer_future_0[36];
    /* vec4 14 — layer configuration */
    float n_layers;
    float _layer_config_1;
    float _layer_config_2;
    float _layer_config_3;
    /* vec4 15 — spectral */
    float peak_sensitivity_nm;
    float spectral_fwhm_nm;
    float _spectral_2;
    float _spectral_3;
} FilmRecord;
#pragma pack(pop)

/**
 * Upload sensor/film tensor chunks and active slot mapping into tracer-owned memory.
 *
 * This is the C++ ingress point used by Python helpers before bidirectional
 * dispatch. Data is copied into RayTracerState, so caller buffers can be
 * released immediately after the call returns.
 *
 * sensor_chunk : float32 [sensor_rows, sensor_stride]  — row i is SensorRecord i
 * film_chunk   : float32 [film_rows,   film_stride]    — row i is FilmRecord i
 * active_slots : int32   [n_slots, 2]  (sensor_id, film_id) pairs
 *
 * sensor_stride must equal sizeof(SensorRecord)/sizeof(float) = 48.
 * film_stride   must equal sizeof(FilmRecord)/sizeof(float)   = 64.
 */
SK_API int ray_tracer_set_sensor_film_ssbo(
    RayTracerState*   st,
    const float*      sensor_chunk,
    int               sensor_rows,
    int               sensor_stride,
    const float*      film_chunk,
    int               film_rows,
    int               film_stride,
    const int32_t*    active_slots,
    int               n_slots
);

/**
 * Upload PBR / enamel / texture-stack material chunks and rebuild the
 * RtMaterialSurfaceCache held inside RayTracerState.
 *
 * pbr_chunk       : float32 (n_pbr,   16) — PBRBaseRecord layout per material
 * enamel_chunk    : float32 (n_enamel,  8) — EnamelRecord layout per material
 * tex_stack_chunk : float32 (n_tex,   16) — TextureStackRecord layout per material
 *
 * Pass NULL / 0 for any chunk to clear it (defaults used in the cache).
 * Returns SK_OK on success, SK_ERR_NULL_STATE if st is NULL.
 */
SK_API int ray_tracer_set_surface_chunks(
    RayTracerState* st,
    const float*    pbr_chunk,
    int             n_pbr,
    const float*    enamel_chunk,
    int             n_enamel,
    const float*    tex_stack_chunk,
    int             n_tex
);

/**
 * Per-slot summary emitted by endpoint-record reduction into sensor/film data.
 *
 * The reduction uses the tracer-owned sensor/film slot tensors previously
 * uploaded via ray_tracer_set_sensor_film_ssbo().  Each summary describes one
 * active slot after the endpoint buffer has been converted into a sensor-plane
 * photon/electron/SNR map.
 */
typedef struct SensorFilmSlotSummary {
    int32_t slot_id;
    int32_t sensor_id;
    int32_t film_id;
    float   qe_peak;
    float   read_noise_e;
    float   dark_current_e_s;
    float   dark_current_accumulated_e;
    float   exposure_time_s;
    float   full_well_e;
    float   snr_peak;
    float   snr_mean;
    float   photons_flux_hz;
    float   electrons_flux_hz;
} SensorFilmSlotSummary;

/**
 * Telemetry for endpoint reduction quality and ordering.
 *
 * This addresses two integration obstacles explicitly:
 * 1) make clamp/filter losses visible (no silent degradation),
 * 2) expose ordering regressions so consumers can reason about ray-order
 *    assumptions before integrating into color/sensor products.
 */
typedef struct EndpointReductionTelemetry {
    int32_t input_records;
    int32_t sensor_group_records;
    int32_t kept_records;
    int32_t kept_pixel_cone_records;
    int32_t kept_projected_records;
    int32_t drop_wrong_group;
    int32_t drop_non_pixel_cone;     /* forward/emission record requiring projection */
    int32_t drop_projection_failed;
    int32_t drop_invalid_band;
    int32_t drop_negative_subpath;
    int32_t drop_out_of_bounds_pixel;
    int32_t order_regressions;
    uint32_t first_subpath_id;
    uint32_t last_subpath_id;
} EndpointReductionTelemetry;

/**
 * Reduce bidirectional EndpointRecord rows into a simulated sensor integral.
 *
 * The input records are the float32 (N, 16) array returned by ray_tracer_bidirectional().
 * Records are filtered by sensor_group_id, coherently accumulated into one
 * complex sensor image, then converted into photons/electrons/SNR maps using
 * the active sensor/film slots stored in RayTracerState.
 *
 * out_photons, out_electrons, and out_snr must each point to a float32
 * buffer of shape (n_py, n_px).  The function zeroes and fills them.
 *
 * If out_slot_summaries is non-NULL, one summary is written per valid active
 * slot, up to out_slot_summary_cap entries.
 */
SK_API int ray_tracer_reduce_endpoint_records_to_sensor_integral(
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
    void*                 out_slot_summaries,
    int                   out_slot_summary_cap,
    int*                  out_slot_summary_count,
    int*                  out_endpoint_record_count,
    EndpointReductionTelemetry* out_telemetry
);

/**
 * Canonical C++ endpoint->RGB image path.
 *
 * This is the C++-owned color integration route from endpoint records to
 * linear RGB and tone-mapped preview RGB, so Python no longer has to own the
 * final spectral->color reduction semantics.
 *
 * out_rgb_linear and out_rgb_tonemapped are float32 (n_py, n_px, 3).
 * hdr_white_percentile follows the same semantic as Python tone mapping.
 */
SK_API int ray_tracer_reduce_endpoint_records_to_rgb_image(
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
    EndpointReductionTelemetry* out_telemetry
);

/**
 * Trace rays and accumulate per-triangle irradiance.
 *
 * This combines the segment capture of ray_tracer_trace with direct
 * accumulation of ray energy into per-triangle flux buffers:
 *
 *   out_direct[tri * n_bands + b]   — irradiance from bounce 0 (direct illumination)
 *   out_indirect[tri * n_bands + b] — irradiance from bounce > 0 (reflected)
 *
 * Energy is defined as |A|² * cos(θ_in) / area, giving units of W/m² per
 * source unit power.  Caller must zero both flux buffers before calling.
 *
 * out_segs and out_cap may be 0/NULL to skip segment capture.
 * out_direct and out_indirect may each be NULL individually to skip that
 * flux accumulation (e.g. pass NULL for out_indirect to get direct only).
 *
 * @param st               Tracer handle from ray_tracer_create.
 * @param n_sources        Number of sources.
 * @param src_pos          (n_sources, 3) float64.
 * @param src_dir          (n_sources, 3) float64.
 * @param src_directivity  (n_sources,)   float64.
 * @param n_rays           Rays per source.
 * @param max_bounces      Maximum reflection bounces.
 * @param min_amplitude    Amplitude cutoff.
 * @param seed             RNG seed.
 * @param out_segs         Segment buffer (may be NULL).
 * @param out_cap          Segment buffer capacity (may be 0).
 * @param out_count        [out] Segments written (may be NULL).
 * @param out_direct       float32 (n_tri, n_bands) — direct irradiance (may be NULL).
 * @param out_indirect     float32 (n_tri, n_bands) — indirect irradiance (may be NULL).
 * @return                 SK_OK or SK_ERR_NULL_STATE if st is NULL.
 */
SK_API int ray_tracer_trace_surface(
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
    float*          out_indirect
);

/* ── Multi-scale context API ───────────────────────────────────────────── */

/**
 * Register a scale context with the tracer.
 *
 * Contexts are tested in registration order.  The first context whose sphere
 * contains a point claims that point.  Overlapping contexts at different
 * scales are resolved by smallest-radius-wins: if a wave context sphere is
 * nested inside a geometric context sphere the wave context takes priority for
 * its sub-volume.
 *
 * Returns the assigned context_id (≥ 0) on success, or SK_ERR_NULL_STATE if
 * st is NULL.  The context_id field in *ctx is updated in-place.
 */
SK_API int ray_tracer_add_scale_context(
    RayTracerState*       st,
    RtScaleContext*       ctx           /* context_id filled in on success */
);

/**
 * Join two registered wave contexts without a ray-domain adapter.
 *
 * Forward fields travel src -> dst and backward fields travel dst -> src.
 * Pipeline construction accepts only identical lane/grid/component layouts,
 * aligned bases, and coincident boundary planes.  It never inserts an
 * implicit resampler.
 */
SK_API int ray_tracer_add_wave_context_link(
    RayTracerState* st,
    int             src_context_id,
    int             dst_context_id);

/**
 * Join two persistent wave contexts through one exact-grid complex interface.
 *
 * coordinate_map is wave_t4::RigidFieldMap encoded as 0..7. Jones arrays are
 * split-complex [n_bands][2][2] row-major values copied into tracer-owned cold
 * storage. The reverse edge uses the reciprocal transpose and inverse rigid
 * map. No interface storage is allocated during T4 transfer.
 */
SK_API int ray_tracer_add_wave_context_interface_link(
    RayTracerState* st,
    int             src_context_id,
    int             dst_context_id,
    int             coordinate_map,
    int             n_bands,
    const float*    jones_re,
    const float*    jones_im);

/** Remove all registered scale contexts from the tracer. */
SK_API int ray_tracer_clear_scale_contexts(RayTracerState* st);

/**
 * Trace rays with automatic multi-scale context transitions.
 *
 * Identical to ray_tracer_trace() in all parameters except the output buffer
 * uses the extended 14-float segment layout (RT_FLOATS_PER_SEG_MS):
 *
 *   [0..11]  same as the base 12-float layout
 *   [12]     context_id    (-1.0f = ambient coarse; ≥ 0 = registered context)
 *   [13]     scale_type    (0.0f = geometric, 1.0f = wave)
 *
 * When a segment's propagation path intersects a wave-scale context sphere,
 * that sub-segment is sub-stepped at context.dt_m with wave-accurate phase
 * (k = 2π f / (c / n_real)), evanescent attenuation (n_imag), and near-field
 * spreading (1/(1+r²)).  Multiple fine segments are written per coarse step.
 *
 * All source/ray/bounce outer loops are the same as ray_tracer_trace.
 * BVH geometry tests are performed at every scale; the sub-stepping only
 * changes how phase and amplitude accumulate between surface hits.
 *
 * @param out_segs   float32 buffer, at least out_cap * RT_FLOATS_PER_SEG_MS floats.
 */
SK_API int ray_tracer_trace_multiscale(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    float*          out_segs,        /* (out_cap, RT_FLOATS_PER_SEG_MS) flat */
    int             out_cap,
    int*            out_count
);

/**
 * Same as ray_tracer_trace_multiscale but also accumulates per-triangle
 * irradiance (same semantics as ray_tracer_trace_surface).
 */
SK_API int ray_tracer_trace_multiscale_surface(
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
    float*          out_indirect
);

/* ── Triangle surface flags ────────────────────────────────────────────── */

/* Triangle flags are only for non-material semantic roles such as aperture
 * stops or picking-only surfaces.  Optical transmission/refraction comes from
 * the material MatBuf record, not from a triangle flag. */
#include "../kernels/mat_flags_generated.h"

/**
 * Set the surface flags for a range of triangles.
 *
 * Per-triangle optical physics (refl, IOR, diffusion, transmittance) lives in
 * the MatBuf addressed by tri.mat_idx.  This entry point only adjusts semantic
 * flags such as MAT_FLAG_APERTURE_STOP.
 *
 * @param flags  bitwise OR of MAT_FLAG_* constants.
 */
SK_API int ray_tracer_set_tri_ior(
    RayTracerState* st,
    int             tri_start,
    int             n_tris,
    int             flags
);

/**
 * Set directional boundary media for a range of triangles.
 *
 * Each triangle stores two media, one per side of its geometric normal:
 *   medium_pos_mat_idx : medium on +normal side
 *   medium_neg_mat_idx : medium on -normal side
 * Use -1 for ambient air/vacuum.
 *
 * Refraction transitions then use side crossing direction directly:
 * front-face hit crosses +normal -> -normal, back-face does the reverse.
 */
SK_API int ray_tracer_set_tri_boundary_media(
    RayTracerState* st,
    int             tri_start,
    int             n_tris,
    int             medium_pos_mat_idx,
    int             medium_neg_mat_idx
);

/**
 * Project a multiscale segment buffer onto a coherent complex sensor image.
 *
 * For every segment that crosses the plane z = sensor_z, the intersection
 * point is mapped to a sensor pixel and the complex amplitude is accumulated
 * as  out_re += A·cos(φ),  out_im += A·sin(φ).  After accumulating many
 * trickle batches the squared magnitude  out_re² + out_im²  gives the
 * coherent intensity image including all interference effects.
 *
 * This is the CPU reference implementation; the GLSL compute shader
 * csrc/shaders/coherent_accumulate.comp.glsl does the same work on the GPU.
 *
 * segs       : (n_segs, RT_FLOATS_PER_SEG_MS) float32
 * n_bands    : number of frequency bands tracked in the segment buffer
 * sensor_w/h : sensor resolution in pixels
 * sensor_z   : z-coordinate of the sensor plane in world space (metres)
 * sensor_r   : half-width of the sensor (metres); pixels cover [-r, +r]²
 * out_re/im  : caller-allocated float32 [n_bands * sensor_h * sensor_w],
 *              must be zeroed by caller before first call (accumulative).
 *
 * Layout: out_re[band * sensor_h * sensor_w + row * sensor_w + col]
 */
SK_API int ray_tracer_project_coherent(
    int             n_segs,
    const float*    segs,
    int             n_bands,
    int             sensor_w,
    int             sensor_h,
    double          sensor_z,
    double          sensor_r,
    float*          out_re,
    float*          out_im
);

/* ── Near-field physical aperture simulation ─────────────────────────────── */

/**
 * Apply a polygonal aperture mask to a complex field buffer in-place.
 *
 * The field is defined on a uniform 2D grid covering [-field_r, +field_r]²
 * in the aperture plane.  Any pixel whose centre falls OUTSIDE the aperture
 * opening polygon is zeroed (re = im = 0).  The polygon is specified as a
 * sequence of 2D (x, y) vertices in metres, treated as a simple closed loop;
 * winding direction does not matter (point-in-polygon test is used).
 *
 * n_bands    : number of frequency bands
 * w, h       : field grid resolution in pixels
 * field_r    : half-width of the field square in metres
 * n_verts    : number of polygon vertices
 * poly_xy    : float32 array [n_verts × 2], (x,y) pairs in metres
 * re_buf/im  : (n_bands, h, w) float32, modified in-place
 *
 * Pixels inside the opening are left unchanged; outside pixels are zeroed.
 * Call this after accumulating the incident field at the aperture plane
 * (e.g. via ray_tracer_project_coherent) and before rs_propagate.
 */
SK_API int ray_tracer_apply_aperture_mask(
    int             n_bands,
    int             w,
    int             h,
    double          field_r,
    int             n_verts,
    const float*    poly_xy,
    float*          re_buf,
    float*          im_buf
);

/**
 * Rayleigh-Sommerfeld near-field diffraction propagator (exact, non-paraxial).
 *
 * Propagates a complex monochromatic field from an input plane (z = 0) to a
 * parallel output plane separated by z_dist metres using the exact scalar
 * Rayleigh-Sommerfeld diffraction integral of the first kind:
 *
 *   U_out(x, y) = (1/2π) ∬ U_in(x', y')
 *                   × (z_dist / r²) × (ik − 1/r) × exp(ikr)
 *                   dx' dy'
 *
 * where r = sqrt((x−x')² + (y−y')² + z_dist²),  k = 2π/λ.
 *
 * This is the physically exact (non-approximated) near-field result.  It
 * reduces to the Kirchhoff formula for large r, and correctly handles
 * evanescent contributions and strongly diverging beams.  There is no
 * paraxial (Fresnel) approximation.
 *
 * Each frequency band is propagated independently with its own wavelength.
 *
 * n_bands        : number of frequency bands
 * w, h           : field grid resolution
 * dx             : pixel pitch in metres (same for both planes)
 * z_dist         : propagation distance (metres, must be > 0)
 * wavelengths_m  : double array [n_bands], wavelength per band in metres
 * in_re / in_im  : (n_bands, h, w) float32 — input complex field
 * out_re / out_im: (n_bands, h, w) float32 — output complex field (zeroed here)
 *
 * Complexity: O(n_bands × w² × h²) — fully parallelisable per output pixel.
 * Use the GLSL compute shader csrc/shaders/rs_propagate.comp.glsl for GPU
 * acceleration (one thread per output pixel per band).
 *
 * The CPU implementation uses all available parallelism via loop nesting;
 * add OpenMP pragmas or call from a thread pool for extra speed.
 */
SK_API int ray_tracer_rs_propagate(
    int             n_bands,
    int             w,
    int             h,
    double          dx,
    double          z_dist,
    const double*   wavelengths_m,
    const float*    in_re,
    const float*    in_im,
    float*          out_re,
    float*          out_im
);

/* ── Stateful ray scheduler ──────────────────────────────────────────────── */

/**
 * Persistent state for one live ray in the context scheduler.
 *
 * amp (complex, per-band) is stored out-of-line inside the tracer's
 * ray_amp_pool — callers do not access it directly.
 */
typedef struct {
    double   pos[3];       /* current world position (metres)                */
    double   dir[3];       /* current unit direction vector                  */
    double   path_len;     /* cumulative path length (metres)                */
    int      bounce;       /* surface reflection count so far                */
    int      src_id;       /* index of the spawning source                   */
    int      context_id;   /* current context (-1 = geometric/coarse)        */
    int      alive;        /* 1 = live; 0 = terminated (absorbed or escaped) */
    uint64_t rng_state;    /* per-ray RNG state for scatter decisions         */
} RtRayState;

/**
 * Spawn rays from sources into the geometric (coarse) context queue.
 *
 * Rays are sampled with Fibonacci-sphere directivity weighting.  Each
 * spawned ray is placed in the finest context sphere that contains the
 * spawn point, or in the geometric (coarse) queue if no context contains it.
 *
 * Calling spawn multiple times accumulates rays without clearing previous
 * ones.  Use ray_tracer_clear_rays() to reset the pool.
 *
 * @param st              Tracer handle.
 * @param n_sources       Number of sources.
 * @param src_pos         (n_sources, 3) float64 source positions.
 * @param src_dir         (n_sources, 3) float64 dominant emit directions.
 * @param src_directivity (n_sources,)   float64 directivity exponents.
 * @param n_rays          Rays per source.
 * @param max_bounces     Maximum surface bounces per ray.
 * @param min_amplitude   Kill ray when max |A| across all bands drops below this.
 * @param seed            RNG seed for directivity sampling.
 * @return SK_OK or SK_ERR_NULL_STATE.
 */
SK_API int ray_tracer_spawn(
    RayTracerState* st,
    int             n_sources,
    const double*   src_pos,
    const double*   src_dir,
    const double*   src_directivity,
    int             n_rays,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed
);

/**
 * Advance all live rays one scheduler step.
 *
 * Each context with live rays is ticked once, ordered from coarsest to
 * finest.  Per-context advancement rules:
 *
 *   Geometric queue (context_id == -1):
 *     Each ray jumps to the nearest BVH surface intersection, capped at
 *     the nearest fine context-sphere entry point.  Rays entering a fine
 *     context sphere are transferred to that context's queue.
 *
 *   RT_SCALE_GEOMETRIC context:
 *     Same as the geometric queue but spatially confined to the context
 *     sphere.  BVH intersection jumps are still used.
 *
 *   RT_SCALE_WAVE context:
 *     Each ray advances min(context.dt_m, t_to_sphere_boundary,
 *     t_to_surface_hit) with wave-accurate phase and near-field spreading.
 *     Multiple rays resident in the same wave context at the same step
 *     form a spatially coherent field — segment amplitudes from such a
 *     batch can be interfered by the caller.
 *
 * Rays that exit their context sphere are moved to the finest containing
 * context, or to the geometric queue if no context contains their new
 * position.  Dead rays are removed.  Segments are written in MS format.
 *
 * @param st          Tracer handle.
 * @param out_segs    float32 buffer, capacity out_cap * RT_FLOATS_PER_SEG_MS floats.
 * @param out_cap     Segment buffer capacity in records.
 * @param out_count   [out] Segment records written this step.
 * @param n_live_out  [out, optional] Live ray count after step (may be NULL).
 * @return SK_OK or SK_ERR_NULL_STATE.
 */
SK_API int ray_tracer_step(
    RayTracerState* st,
    float*          out_segs,
    int             out_cap,
    int*            out_count,
    int*            n_live_out
);

/** Remove all live rays from all context queues and free the ray pool. */
SK_API int ray_tracer_clear_rays(RayTracerState* st);

/** Return the total number of live rays across all queues. */
SK_API int ray_tracer_live_ray_count(const RayTracerState* st);

/* ─── Bidirectional integrator surface ───────────────────────────────── */
/* Triangle-group registry (declared in triangle_groups.h, implemented in
 * ray_tracer.cpp).  Forward-include here so callers only need ray_tracer.h. */
#ifdef __cplusplus
} /* close extern "C" before include */
#endif
#include "triangle_groups.h"
#include "bdpt_record.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Bidirectional path tracing entry point.  Emits ``n_rays_per_emitter`` rays
 * from each registered EMISSIVE TriGroup (area-weighted triangle pick +
 * cosine-hemisphere direction), traces them through the BVH up to
 * ``max_bounces`` bounces, and writes one EndpointRecord per (subpath × band)
 * each time a ray strikes a triangle that is part of a SENSOR TriGroup.
 *
 * Phase, atmospheric attenuation, reactive band-shift, aperture-stop kill,
 * and material reflection are all applied inside the bounce loop — the
 * record carries the FINAL complex amplitude per band at the moment of
 * sensor capture.  No reduction.
 *
 * Returns SK_OK on success, SK_ERR_NULL_STATE on bad arguments.
 * Writes the record count produced into *out_count (clamped to out_cap).
 */
SK_API int ray_tracer_bidirectional(
    RayTracerState* st,
    int             n_rays_per_emitter,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    EndpointRecord* out_records,
    int             out_cap,
    int*            out_count);

/**
 * Packed bidirectional entry with per-emitter ray quotas.
 *
 * n_rays_per_emitter has one entry per registered EMISSIVE TriGroup, in
 * registration order. Each emitter uses max(entry, 0) rays.
 */
SK_API int ray_tracer_bidirectional_packed(
    RayTracerState* st,
    const int32_t*  n_rays_per_emitter,
    int             n_emitters,
    int             max_bounces,
    double          min_amplitude,
    uint32_t        seed,
    EndpointRecord* out_records,
    int             out_cap,
    int*            out_count);

/**
 * BDPT connection step.
 *
 * For every backward (sensor-side) EndpointRecord and a random sample of
 * forward (light-side) EndpointRecords in the same spectral band, cast a
 * shadow ray between the two scene-side vertices.  Unoccluded pairs contribute
 *   |b.amp| * |f.amp| / dist²
 * to the pixel addressed by b.subpath_id.
 *
 * out_rgb must be pre-zeroed by the caller; size = n_px * n_py * 3 floats.
 * max_fwd_samples == 0 tests all forward records (potentially very slow).
 */
SK_API int ray_tracer_bdpt_connect(
    const RayTracerState* st,
    const EndpointRecord* records,
    int                   n_records,
    int                   n_px,
    int                   n_py,
    int                   sensor_gid,
    int                   max_fwd_samples,
    uint32_t              seed,
    float*                out_rgb,
    int                   n_rgb);

/**
 * Deposit EndpointRecord amplitudes into the bound field-capture grid.
 *
 * This wires BDPT endpoint transport into volumetric field capture so sensor
 * visibility and volume activation share the same backend accumulation path.
 *
 * include_sensor_group and include_non_sensor_groups control which records are
 * injected relative to sensor_group_id.
 */
SK_API int ray_tracer_accumulate_endpoint_records_to_field_capture(
    RayTracerState*      st,
    const EndpointRecord* records,
    int                  n_records,
    int                  sensor_group_id,
    int                  include_sensor_group,
    int                  include_non_sensor_groups,
    int*                 out_written_records,
    double*              out_written_power);

/* ── UV integrator image API ─────────────────────────────────────────────
 *
 * Each triangle group with uv_image_res > 0 gets a multi-channel flat
 * accumulator: (UV_N_HDR_CHANNELS + 5*n_bands) uint32 channels, each of
 * size res×res, packed as channel-major (channel 0 first, then channel 1, …).
 * Both the GPU T3 shader and the CPU T3 scatter path write identically.
 *
 * Fixed header channels (indices 0..UV_N_HDR_CHANNELS-1):
 *   UV_CH_HIT_COUNT    [0]  raw hit count (atomicAdd 1)
 *   UV_CH_SRC_FLAGS    [1]  source-ID bitfield (atomicOr 1<<src_id, max 32 sources)
 *   UV_CH_BOUNCE_0     [2]  direct-hit count   (bounce == 0)
 *   UV_CH_BOUNCE_1     [3]  1st-order count    (bounce == 1)
 *   UV_CH_BOUNCE_2     [4]  2nd-order count    (bounce == 2)
 *   UV_CH_BOUNCE_3PLUS [5]  3rd-and-higher     (bounce >= 3)
 *   UV_CH_TAG_LO       [6]  tag_lo OR bitfield
 *   UV_CH_TAG_HI       [7]  tag_hi OR bitfield
 *   UV_CH_NORMAL_X     [8]  hit-normal x sum, signed fixed-pt ×32768
 *   UV_CH_NORMAL_Y     [9]  hit-normal y sum, signed fixed-pt ×32768
 *   UV_CH_NORMAL_Z     [10] hit-normal z sum, signed fixed-pt ×32768
 *
 * Per-band channels (b = 0..n_bands-1):
 *   UV_N_HDR_CHANNELS + b            amplitude magnitude, unsigned ×65536
 *   UV_N_HDR_CHANNELS + n_bands + b  amplitude real part, signed ×32768
 *   UV_N_HDR_CHANNELS + 2*n_bands+b  amplitude imag part, signed ×32768
 *   UV_N_HDR_CHANNELS + 3*n_bands+b  forward/emissive magnitude, unsigned ×65536
 *   UV_N_HDR_CHANNELS + 4*n_bands+b  sensor/reverse magnitude, unsigned ×65536
 *
 * Decode signed channels: value = (int32_t)raw_uint / scale
 *
 * ray_tracer_get_group_uv_image() — decode and copy all channels for group_id.
 *   out_channels must be float32[n_channels * res * res]; layout is
 *   channel-major: out_channels[ch * res*res + texel].
 *   *out_res and *out_n_channels are filled with actual values.
 *
 * ray_tracer_clear_group_uv_accum() — zero accumulator for group_id,
 *   or all groups when group_id < 0.
 */

#define UV_N_HDR_CHANNELS   11
#define UV_CH_HIT_COUNT      0
#define UV_CH_SRC_FLAGS      1
#define UV_CH_BOUNCE_0       2
#define UV_CH_BOUNCE_1       3
#define UV_CH_BOUNCE_2       4
#define UV_CH_BOUNCE_3PLUS   5
#define UV_CH_TAG_LO         6
#define UV_CH_TAG_HI         7
#define UV_CH_NORMAL_X       8
#define UV_CH_NORMAL_Y       9
#define UV_CH_NORMAL_Z      10

SK_API int ray_tracer_get_group_uv_image(
    const RayTracerState* st,
    int                   group_id,
    float*                out_channels,   /* float32[n_channels * res * res] */
    int*                  out_res,
    int*                  out_n_channels);

SK_API int ray_tracer_set_group_uv_image(
    RayTracerState*       st,
    int                   group_id,
    const float*          channels,       /* float32[n_channels * res * res] */
    int                   res,
    int                   n_channels);

typedef struct RayTracerUvGroupSummary {
    int      group_id;
    int      res;
    int      n_channels;
    int      tri_count;
    uint64_t memory_bytes;
    uint64_t nonzero_texels;
    double   total_forward;
    double   total_sensor;
    double   peak_total;
} RayTracerUvGroupSummary;

SK_API int ray_tracer_get_group_uv_summary(
    const RayTracerState*      st,
    int                        group_id,
    RayTracerUvGroupSummary*   out_summary);

SK_API int ray_tracer_clear_group_uv_accum(
    RayTracerState* st,
    int             group_id);

/**
 * Update the per-band emission power of an already-registered EMISSIVE group.
 *
 * This is the primary mechanism for surrogate emitters whose power is
 * determined by an external field solver, neural network, or measured
 * radiance) rather than a fixed material property.  Safe to call between
 * tracing passes; never safe to call concurrently with an active trace.
 *
 * group_id       — as returned by ray_tracer_register_tri_group()
 * power_W_per_band — float32 array of length n_bands; caller-owned
 * n_bands        — must match the tracer's n_bands
 */
SK_API int ray_tracer_set_tri_group_power(
    RayTracerState* st,
    int             group_id,
    const float*    power_W_per_band,
    int             n_bands);

/* ── BSSRDF per-triangle illumination accumulator ────────────────────────────
 * Forward paths that hit a diffuse-transmissive surface (diffuse_frac > 0)
 * accumulate their pre-interaction amplitude here.  Backward paths query this
 * accumulator to analytically claim their diffuse illumination contribution
 * without traversing the scatter volume via Monte Carlo.
 *
 * Two-pass protocol:
 *   1. ray_tracer_init_illum_accum()   — after geometry is set; sizes + zeros.
 *   2. ray_tracer_reset_illum_accum()  — between batches (keeps allocation).
 *   3. Run forward tracing pass.
 *   4. Run backward tracing pass (reads the now-populated accumulator).
 *   5. ray_tracer_export_illum_accum() — optional GPU SSBO / Python inspection.
 *
 * T4 field seeding (optional, wave-domain alternative to Monte Carlo):
 *   After T4 propagates the emitter field through a diffusing volume,
 *   write the exit-plane amplitude directly into the accumulator via
 *   ray_tracer_export_illum_accum() + modify + set_illum_accum() (TBD).
 *   Backward rays then receive diffraction-correct illumination automatically.
 */
SK_API int ray_tracer_init_illum_accum(RayTracerState* st);
SK_API int ray_tracer_reset_illum_accum(RayTracerState* st);
SK_API int ray_tracer_export_illum_accum(
    const RayTracerState* st,
    float*                buf,       /* out: float32[n_tris * stride]; NULL = query */
    int                   buf_floats,
    int*                  out_n_tris,
    int*                  out_stride); /* stride = 2*n_bands + 2 */

/* Write field-computed complex amplitudes directly into tri_illum_accum for
 * the specified triangles, replacing any Monte Carlo data already there.
 * After this call backward rays at those triangles receive diffraction-correct
 * field illumination instead of a Monte Carlo average.
 *
 *   tri_ids  : (n_tris,) int32  — triangle indices
 *   amp_re   : (n_tris, n_bands) float32 row-major — real part of field amplitude
 *   amp_im   : (n_tris, n_bands) float32 row-major — imaginary part
 *   cos_avg  : (n_tris,) float32 — mean |cos θ| of incidence (use 1.0 for normal incidence)
 *   n_bands  : number of spectral bands in amp_re / amp_im
 */
SK_API int ray_tracer_write_tri_illum(
    RayTracerState* st,
    const int*      tri_ids,
    int             n_tris,
    const float*    amp_re,
    const float*    amp_im,
    const float*    cos_avg,
    int             n_bands);

#ifdef __cplusplus
} /* extern "C" */
#endif
