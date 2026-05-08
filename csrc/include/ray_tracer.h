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

#include <stdint.h>

#define RT_FLOATS_PER_SEG     12
#define RT_BYTES_PER_SEG      48

/* Multiscale segment format — 14 floats (56 bytes).
 * Extends the base 12-float layout with two extra fields:
 *   [12]  context_id   — index of the scale context (-1 = ambient/coarse)
 *   [13]  scale_type   — RT_SCALE_GEOMETRIC (0) or RT_SCALE_WAVE (1)
 */
#define RT_FLOATS_PER_SEG_MS  14
#define RT_BYTES_PER_SEG_MS   56

/* Scale context type tags. */
#define RT_SCALE_GEOMETRIC    0   /* coarse BVH geometric optics (cm-scale)     */
#define RT_SCALE_WAVE         1   /* fine wave-optics sub-stepping (µm-scale)   */

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
} RtScaleContext;

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tracer state — build once from scene geometry, trace many times. */
typedef struct RayTracerState RayTracerState;

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

/** Free tracer state.  Safe to call with NULL. */
SK_API void ray_tracer_destroy(RayTracerState* st);

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

/* Phase 2 unification: triangle flags use the unified MAT_FLAG_* set defined
 * in mat_flags.py and emitted into mat_flags_generated.h.  Callers should use
 * MAT_FLAG_TRANSMISSIVE / MAT_FLAG_APERTURE_STOP / etc. directly. */
#include "../kernels/mat_flags_generated.h"

/**
 * Set the surface flags for a range of triangles.
 *
 * Phase 2 cutover: per-triangle IOR/refl now live in the MatBuf addressed by
 * tri.mat_idx (set at create time).  This entry point only adjusts the
 * MAT_FLAG_* bitmask (TRANSMISSIVE, APERTURE_STOP, …).
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

/**
 * Beam Propagation Method (BPM) — batchwise PDE z-stepper.
 *
 * Propagates a complex scalar field U[band][y][x] forward by dz by solving
 * the paraxial Helmholtz PDE:
 *
 *   ∂U/∂z = (i/2k) ∇_T² U
 *
 * using an ADI (Alternating Direction Implicit) Crank-Nicolson finite-
 * difference scheme.  The scheme is unconditionally stable and second-order
 * accurate in both dz and dx.  All bands are stepped in a single batched
 * loop.
 *
 * Each call advances the field by exactly one step dz.  Call repeatedly to
 * propagate through a volume: the field evolves continuously through empty
 * space with correct diffraction, interference, and near-field spreading at
 * every point — this is the genuine PDE solution, not a post-process.
 *
 * The carrier phase exp(ik dz) is applied first so both the fast oscillation
 * and the transverse spreading are correct when building coherent volumes.
 *
 * Boundary: Dirichlet U = 0 at all four grid edges (absorbing frame).
 *
 * @param n_bands       Number of frequency bands.
 * @param w, h          Grid width and height (pixels).
 * @param dx            Pixel pitch in metres (same in x and y).
 * @param dz            Propagation step in metres (positive = forward along z).
 * @param wavelengths_m Double array [n_bands], wavelength per band in metres.
 * @param re_buf        float32 [n_bands * h * w] real part  (modified in-place).
 * @param im_buf        float32 [n_bands * h * w] imag part  (modified in-place).
 * @return SK_OK or SK_ERR_NULL_STATE.
 */
SK_API int ray_tracer_wave_bpm_step(
    int             n_bands,
    int             w,
    int             h,
    double          dx,
    double          dz,
    const double*   wavelengths_m,
    float*          re_buf,
    float*          im_buf
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

#ifdef __cplusplus
} /* extern "C" */
#endif
