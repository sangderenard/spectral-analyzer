/**
 * rt_field_solver.h — Unified acoustic / EM complex field solver.
 *
 * Computes the complex transfer function H[source → receiver, freq] by
 * tracing rays coherently through a triangulated scene and summing their
 * contributions at receiver locations.  Supports:
 *
 *  ACOUSTIC  — scalar complex pressure field.
 *              Surface reflectance from acoustic impedance (oblique-incidence).
 *
 *  EM        — vector electric field with full Jones-matrix propagation.
 *              Surface reflectance from Fresnel equations (angle-dependent,
 *              complex refractive indices).  Each source emits two orthogonal
 *              polarisations simultaneously; the result is the 2×2 Jones
 *              transfer matrix H_ij = received_polarisation_i / input_j.
 *
 * Hybrid modal + ray crossover
 * ----------------------------
 * Below schroeder_hz the room-mode contribution is computed analytically
 * from a rectangular-bounding-box eigenfrequency expansion (fast, closed-form)
 * and blended with the geometric ray solution via a smooth crossover filter.
 * Pass schroeder_hz = 0 to disable modal synthesis (ray only).
 *
 * Output buffer layout
 * --------------------
 * Acoustic:  out[src][rec][band]          — complex double (re/im pair)
 * EM:        out[src][rec][band][p_out][p_in] — 2×2 Jones matrix entries
 *
 * Both are stored as flat arrays of interleaved (re, im) double pairs.
 * The helper macros below give the flat index for each mode.
 *
 * Receiver model
 * --------------
 * Point receiver  (RTS_OMNI … RTS_HYPERCARDIOID): scalar complex sum of all
 *   shadow-ray contributions, each weighted by D(θ_arrival).
 * Aperture receiver (RTS_APERTURE): the aperture disk is sampled at
 *   RTS_AP_SAMPLES points; contributions are averaged over these samples.
 *   A polar weighting D(θ) is applied per sample.  The spatial phase
 *   variation across the aperture (different path lengths to each sample point)
 *   provides a natural aperture low-pass effect.
 */

#pragma once
#include "serial_kernel.h"   /* SK_API, SK_OK, SK_ERR_* */
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------- */

#define RTS_AP_SAMPLES    16    /* aperture spatial sampling points           */
#define RTS_MAX_BANDS     256   /* maximum frequency bands                    */
#define RTS_MAX_MODAL     8192  /* maximum room modes below Schroeder freq    */

/* Flat index into output buffer ------------------------------------------ */
/* Acoustic: H[src, rec, band]  → 2 doubles (re, im)                         */
#define RTS_IDX_ACOUSTIC(si, ri, b, n_rec, n_bands) \
    (((size_t)(si)*(n_rec)+(ri))*(n_bands)+(b))

/* EM: H[src, rec, band, p_out, p_in]  → 2 doubles (re, im) per entry        */
#define RTS_IDX_EM(si, ri, b, p_out, p_in, n_rec, n_bands) \
    ((((size_t)(si)*(n_rec)+(ri))*(n_bands)+(b))*4 + (p_out)*2+(p_in))

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Enumerations
 * ------------------------------------------------------------------------- */

typedef enum {
    RTS_ACOUSTIC = 0,   /* scalar pressure, impedance-based reflectance  */
    RTS_EM       = 1    /* vector E-field, Fresnel reflectance, Jones 2×2*/
} RtsMode;

typedef enum {
    RTS_OMNI          = 0,  /* D = 1                                      */
    RTS_CARDIOID      = 1,  /* D = (1 + cos θ) / 2                        */
    RTS_FIGURE8       = 2,  /* D = |cos θ|                                */
    RTS_HYPERCARDIOID = 3,  /* D = 0.25 + 0.75 cos θ                     */
    RTS_APERTURE      = 4   /* area integral over disk, polar-weighted    */
} RtsPolarType;

/* ---------------------------------------------------------------------------
 * Receiver definition
 * ------------------------------------------------------------------------- */

typedef struct {
    double       pos[3];          /* centre position (world units)         */
    double       axis[3];         /* polar axis (points toward desired src)*/
    RtsPolarType polar_type;
    double       aperture_r;      /* disc radius (RTS_APERTURE only)       */

    /* EM: detector polarisation unit vectors (ignored for acoustic).
     * pol_s / pol_p are two orthogonal unit vectors in the plane ⊥ to axis.
     * H[p_out=0] corresponds to projection onto pol_s,
     * H[p_out=1] corresponds to projection onto pol_p.            */
    double       pol_s[3];        /* s-polarisation detector axis          */
    double       pol_p[3];        /* p-polarisation detector axis          */
} RtsReceiver;

/* ---------------------------------------------------------------------------
 * Scene definition
 * ------------------------------------------------------------------------- */

typedef struct {
    /* Geometry — same layout as ray_tracer_create                          */
    int           n_tri;
    const double* verts;          /* (n_tri, 9)  float64 — flat triangle verts */
    const double* normals;        /* (n_tri, 3)  float64 — outward unit normals */

    /* Per-triangle material index                                          */
    const int*    mat_idx;        /* (n_tri,)    int32                      */

    /* Material table
     * Acoustic: mat_n_re = Re(Z/Z_air), mat_n_im = Im(Z/Z_air)
     * EM:       mat_n_re = Re(ñ),       mat_n_im = Im(ñ)     (absorption > 0) */
    int           n_mats;
    int           n_bands;
    const double* mat_n_re;       /* (n_mats, n_bands) float64 row-major  */
    const double* mat_n_im;
    const double* mat_diffusion;  /* (n_mats,)         float64            */

    /* Propagation medium — typically (1.0, 0.0) for air/vacuum             */
    double        medium_n_re;
    double        medium_n_im;
} RtsScene;

/* ---------------------------------------------------------------------------
 * Opaque solver state
 * ------------------------------------------------------------------------- */

typedef struct RtsFieldState RtsFieldState;

/**
 * Allocate solver from a scene description.
 *
 * @param scene       Scene geometry and materials.
 * @param freq_hz     (n_bands,) float64 — centre frequencies in Hz.
 * @param speed_m_s   Speed of wave propagation in the medium (m/s).
 * @param mode        RTS_ACOUSTIC or RTS_EM.
 * @return            Opaque handle, NULL on failure.
 */
SK_API RtsFieldState* rts_create(
    const RtsScene* scene,
    const double*   freq_hz,
    double          speed_m_s,
    RtsMode         mode
);

SK_API void rts_destroy(RtsFieldState* st);

/**
 * Compute the complex transfer matrix H[src → receiver, freq].
 *
 * @param st               Solver state.
 * @param n_sources        Number of sources.
 * @param src_pos          (n_src, 3)  float64 — source positions.
 * @param src_dir          (n_src, 3)  float64 — dominant emit directions.
 * @param src_directivity  (n_src,)    float64 — directivity exponent.
 * @param src_pol_re       (n_src, 3)  float64 — EM source E-field polarisation
 *                                               real part (ignored for acoustic).
 * @param src_pol_im       (n_src, 3)  float64 — same, imaginary part.
 * @param n_receivers      Number of receivers.
 * @param receivers        Array of RtsReceiver.
 * @param n_rays           Rays per source per (EM: polarisation pair).
 * @param max_bounces      Maximum reflection bounces.
 * @param min_amplitude    Terminate ray when max |A| across bands < this.
 * @param seed             RNG seed for diffuse scatter.
 * @param schroeder_hz     Crossover frequency (Hz); 0 = ray-only.
 * @param out_re           Output real parts — see RTS_IDX_ACOUSTIC / RTS_IDX_EM.
 * @param out_im           Output imaginary parts — same layout as out_re.
 *                         Caller must allocate:
 *                           acoustic: n_src * n_rec * n_bands doubles each
 *                           EM:       n_src * n_rec * n_bands * 4 doubles each
 * @return                 SK_OK on success.
 */
SK_API int rts_solve(
    RtsFieldState*      st,
    int                 n_sources,
    const double*       src_pos,
    const double*       src_dir,
    const double*       src_directivity,
    const double*       src_pol_re,
    const double*       src_pol_im,
    int                 n_receivers,
    const RtsReceiver*  receivers,
    int                 n_rays,
    int                 max_bounces,
    double              min_amplitude,
    uint32_t            seed,
    double              schroeder_hz,
    double*             out_re,
    double*             out_im
);

#ifdef __cplusplus
} /* extern "C" */
#endif
