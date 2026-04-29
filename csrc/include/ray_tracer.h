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

#define RT_FLOATS_PER_SEG  12
#define RT_BYTES_PER_SEG   48

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
 * @param refl_re     (n_tri, n_bands) float64 row-major — complex reflectance, real.
 * @param refl_im     (n_tri, n_bands) float64 row-major — complex reflectance, imag.
 * @param diffusion   (n_tri,)         float64 — diffuse scatter fraction per tri [0,1].
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
    const double*   refl_re,
    const double*   refl_im,
    const double*   diffusion,
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

#ifdef __cplusplus
} /* extern "C" */
#endif
