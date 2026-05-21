/**
 * triangle_groups.h — Triangle-group registry for bidirectional integration.
 *
 * A "triangle group" is any subset of scene triangles tagged with one or more
 * roles.  Groups are the unit of bidirectional bookkeeping:
 *
 *   - EMISSIVE_GROUP : light subpaths originate by sampling triangles in the
 *     group (uniform-by-area) and emitting cosine-weighted directions.
 *   - SENSOR_GROUP   : when a ray hits a triangle that belongs to a sensor
 *     group, the segment is recorded as an EndpointRecord (complex amplitude,
 *     phase, direction, path length) instead of being splatted into an image.
 *
 * Either role can be set on a subset of triangles inside a group (see
 * ``TriGroup::tri_indices`` — group membership is by explicit index list, not
 * by mat_idx, so the same material can be sensor in one group and ordinary
 * surface in another).
 *
 * The registry replaces the old single-image splat with a per-group
 * EndpointRecord SSBO.  Magnitude / phase / RGB visualisation is computed at
 * display time only — storage is always full complex per band.
 *
 * Hard rules (per the integrator-rewrite directive):
 *   - No reduction at deposit time. EndpointRecord stores complex amplitude.
 *   - No abs(), no quantize. Aggregation is the caller's responsibility.
 *   - Metadata round-trips: every emit/deposit carries subpath_id,
 *     bounce_index, group_id so the bidirectional join can pair light and
 *     sensor subpaths exactly.
 */
#pragma once
#include "serial_kernel.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Role bits. Multiple bits may be combined on the same group. */
#define TRI_GROUP_ROLE_EMISSIVE   (1u << 0)  /* light subpath origin              */
#define TRI_GROUP_ROLE_SENSOR     (1u << 1)  /* endpoint deposit on hit           */
#define TRI_GROUP_ROLE_BLOCKER    (1u << 2)  /* contributes to occlusion only     */
#define TRI_GROUP_ROLE_VOLUME     (1u << 3)  /* triangles enclose a material region */

/* Sampling weight policy.
 *
 * For EMISSIVE groups: how a triangle is picked when spawning a light subpath.
 * For SENSOR  groups: how the sensor side of the bidirectional join is
 *   driven.  PIXEL_CONE turns the group into a true camera sensor with a
 *   deterministic per-pixel outer loop and a stochastic inner aperture
 *   sampling — see CameraSensorDesc below.
 */
#define TRI_GROUP_SAMPLE_UNIFORM       0  /* uniform per triangle index         */
#define TRI_GROUP_SAMPLE_AREA          1  /* area-weighted (Lambertian default) */
#define TRI_GROUP_SAMPLE_POWER         2  /* power × area weighted              */
#define TRI_GROUP_SAMPLE_PIXEL_CONE    3  /* SENSOR-only: per-pixel cone scan   */

/* Optional per-group parametric surface model.
 * NONE      : use triangle geometry as-is.
 * POLY_BARY : scalar displacement along the triangle normal using a
 *             polynomial in barycentric (u,v) coordinates.
 * Payload format (float64[6]):
 *   [0]=c0, [1]=cu, [2]=cv, [3]=cuu, [4]=cuv, [5]=cvv
 * with
 *   delta(u,v) = c0 + cu*u + cv*v + cuu*u*u + cuv*u*v + cvv*v*v
 */
#define TRI_PARAM_SURFACE_NONE              0
#define TRI_PARAM_SURFACE_POLY_BARY         1
#define TRI_PARAM_SURFACE_SDF_SADDLE        2
#define TRI_PARAM_SURFACE_SDF_SPHERE        3
/* Neural assembly surface.  Payload = float32 neural MLP payload (magic 14948).
 * Header bytes p[11..16] carry surface geometry and dispatch control:
 *   p[11] = ROC          (radius of curvature, m; 0 = flat)
 *   p[12] = k            (conic constant)
 *   p[13] = axis_index   (0=X scene-axis, 2=Z lens-design default)
 *   p[14] = r_out        (outer radius — hits outside are blocked)
 *   p[15] = side         (0 = entrance/scene-side, 1 = exit/sensor-side)
 *   p[16] = x_field_plane (projection plane for the MLP's "field" input:
 *                          z_entrance for side=0, z_sensor_plane for side=1)
 *
 * Two payloads are registered — one per surface, each with a separate set
 * of MLP weights trained in the appropriate direction (forward / backward).
 * The MLP output plane is always p[6]: z_sensor for side=0, z_entrance for
 * side=1.  Rays are teleported to p[6] with the decoded direction. */
#define TRI_PARAM_SURFACE_NEURAL_ASSEMBLY   4

/* Parametric lens surface.  Payload = float32 compact assembly description
 * (magic 14949) built by CompoundLens.build_gpu_payload().
 *
 * Header  (8 floats):
 *   [0] magic  (14949.0)
 *   [1] n_surfaces  (cast to int)
 *   [2] hood_r_opening  (0 = no hood check)
 *   [3] hood_x_front
 *   [4] hood_x_rim
 *   [5..7] reserved
 *
 * Per surface record  (8 floats each, PLENS_SURF_STRIDE):
 *   [0] x_pos         axial vertex position (scene X, metres)
 *   [1] R_curvature   signed ROC; 0 = flat plane
 *   [2] n_before      IOR on entry side
 *   [3] n_after       IOR on exit side
 *   [4] aperture_r    clear aperture radius
 *   [5] conic_k       conic constant (0=sphere, −1=paraboloid, …)
 *   [6] flags         bit 0 = is_stop (aperture check only, no refraction)
 *   [7] reserved
 *
 * T2 evaluates the exact quadratic conic intersection + vector Snell's law
 * for each surface in one pass, accumulates OPL, then teleports the ray
 * (bit 4) or absorbs (bit 3) using the same protocol as NEURAL_ASSEMBLY.
 * Physical parameters live exclusively in the payload; mesh geometry is
 * only used by T1 for intersection detection. */
#define TRI_PARAM_SURFACE_PARAMETRIC_LENS   5

/* Camera operating modes for CameraSensorDesc.camera_mode.
 *
 * PINHOLE             — one ray per pixel through lens/aperture centre; no disk.
 * APERTURE_CONE       — stochastic disk samples; no lens bending (default).
 * THIN_LENS_GEOMETRIC — focus-plane ray generation: all aperture samples for a
 *                       pixel converge to a single world point on the focus plane.
 */
/* Tier 0 — oracle pinhole reference: 1 ray per pixel, deterministic, clean supervision */
#define CAMERA_MODE_ORACLE_PINHOLE_REFERENCE 0

/* Tier 1 — physical pinhole: tiny aperture, photon-limited, noise-dominated */
#define CAMERA_MODE_PHYSICAL_PINHOLE 1

/* Tier 2 — aperture cone: disk samples, no lens bending (default for simple scenes) */
#define CAMERA_MODE_APERTURE_CONE 2

/* Tier 3 — ideal thin-lens geometric: focus-plane convergence via thin lens */
#define CAMERA_MODE_THIN_LENS_GEOMETRIC 3

/* Tier 4 — element-by-element geometric assembly: real optical surface chain (STUB) */
#define CAMERA_MODE_GEOMETRIC_ASSEMBLY 4

/* Tier 5 — wave-patch transport: diffraction, interference, finite-element waves (STUB) */
#define CAMERA_MODE_WAVE_ASSEMBLY 5

/* Tier 6 — baked transform function: accelerated LUT/spline/neural transport map (STUB) */
#define CAMERA_MODE_BAKED_TRANSFORM 6

/* Legacy alias for API compatibility during transition (will be removed) */
#define CAMERA_MODE_THICK_LENS_WAVE 5

/**
 * Camera sensor descriptor (used when sample_policy == PIXEL_CONE).
 *
 * The camera's sensor plane is one TriGroup of role SENSOR.  This descriptor
 * tells the bidirectional integrator how to *drive* it: deterministic outer
 * loop over (px, py) ∈ [0, n_px) × [0, n_py), inner stochastic loop drawing
 * n_aperture_samples points on the aperture stop.  No grid jitter — purely
 * stochastic so we don't print a sampler pattern into the image.
 *
 * Aperture honoring:
 *   aperture_stop_group_id >= 0  →  rays are tested against that BLOCKER
 *                                   group's triangles; misses are dropped.
 *                                   This is how blade polygon shape gets
 *                                   honored without separate blade math.
 *   aperture_stop_group_id <  0  →  fall back to a circular aperture of
 *                                   radius aperture_radius_m centred on
 *                                   (pos + focal_m * fwd).
 *
 * Pixel grid is derived from the camera basis: pixel (px, py) lives at
 *   sensor_origin + (px+0.5)/n_px * sensor_w * right
 *                 + (py+0.5)/n_py * sensor_h * up
 * with sensor_origin = pos - 0.5*sensor_w*right - 0.5*sensor_h*up.
 *
 * Thin-lens mode (camera_mode == CAMERA_MODE_THIN_LENS_GEOMETRIC):
 *   effective_focal_m  — effective focal length f (lens formula: 1/f = 1/si + 1/so).
 *                        0 = fall back to focal_m (image-side distance).
 *   focus_distance_m   — scene focus distance so from the lens centre.
 *                        0 = auto-compute as so = f*si / (si - f).
 *   lens_center[3]     — lens plane centre in world space; zeros = aperture_centre.
 *   lens_fwd[3]        — fixed lens optical axis (unit vector); zeros = camera fwd.
 */
typedef struct {
    double pos[3];                  /* camera nodal/sensor centre (m)       */
    double fwd[3];                  /* unit forward (sensor → scene)        */
    double up[3];                   /* unit up (image-y axis)               */
    double sensor_w_m;              /* physical sensor width  (m)           */
    double sensor_h_m;              /* physical sensor height (m)           */
    double focal_m;                 /* image-side dist sensor→aperture (m)  */
    double aperture_radius_m;       /* fallback disk radius if no stop grp  */
    int    n_px;                    /* horizontal pixel count               */
    int    n_py;                    /* vertical   pixel count               */
    int    n_aperture_samples;      /* stochastic samples per pixel         */
    int    aperture_stop_group_id;  /* -1 = use disk fallback               */
    int    pixel_stream_divisor;    /* process every Nth pixel (>=1)        */
    int    pixel_stream_phase;      /* stream phase offset [0, N)           */
    int    pixel_stream_phase_from_seed; /* 1: derive phase from batch seed */
    /* ── Optical mode extensions (additive; zero-init = APERTURE_CONE) ── */
    int    camera_mode;             /* CAMERA_MODE_*                        */
    double effective_focal_m;       /* effective focal length; 0 = focal_m  */
    double focus_distance_m;        /* scene focus dist from lens; 0 = auto */
    double lens_center[3];          /* lens plane centre; zeros = ap_centre */
    double lens_fwd[3];             /* fixed lens axis; zeros = cam fwd     */
    int    use_optical_handlers;    /* 1: run st->optical_assembly per ray  */
} CameraSensorDesc;

/**
 * TriGroup descriptor.
 *
 * Owned by the RayTracerState; built via ray_tracer_register_tri_group().
 * The state retains the indices and a precomputed cumulative-area table for
 * O(log N) area-weighted sampling.
 *
 * All "Optional" fields default to 0/NULL/-1 when the caller zero-inits.
 * Adding a field at the END of this struct is safe; reorder = ABI break.
 */
typedef struct {
    int       group_id;          /* assigned by registrar; -1 in caller copy */
    uint32_t  role_bits;         /* TRI_GROUP_ROLE_* OR mask                 */
    uint32_t  sample_policy;     /* TRI_GROUP_SAMPLE_*                       */
    int       n_tris;            /* count                                    */
    const int* tri_indices;      /* (n_tris,) — caller-owned during register */
    /* Optional metadata. Zero/NULL = defaults. */
    double    plane_origin[3];   /* virtual plane origin (for plane-tagged   */
    double    plane_normal[3];   /* sensor groups; ignored otherwise)        */
    double    custom_emit_W;     /* override total emitted power; 0 = derive */
    /* ── Additive extensions (safe to leave zeroed) ────────────────────── */
    int       default_mat_idx;   /* -1 = derive from majority tri material   */
    int       n_power_bands;     /* length of power_W_per_band; 0 = none     */
    const float*    power_W_per_band; /* (n_power_bands,) optional spectrum  */
    const CameraSensorDesc* sensor_camera; /* SENSOR + PIXEL_CONE only       */
    int       parametric_surface_kind;   /* TRI_PARAM_SURFACE_*              */
    int       parametric_payload_bytes;  /* byte size of payload             */
    const void* parametric_payload;      /* optional coeff payload           */
    /* ── UV integrator image (optional) ────────────────────────────────── */
    int          uv_image_res;    /* 0 = no UV accumulation; >0 = enable    */
    int          uv_n_coords;     /* n_tris * 6 floats; 0 = auto planar     */
    const float* uv_coords;       /* (n_tris, 3, 2) float32; NULL = auto    */
} TriGroupDesc;

/* Registrar / accessors (state-mutating side opaquely declared in ray_tracer.h). */
struct RayTracerState;

/**
 * Register a triangle group with the tracer.  Returns the assigned group_id
 * (>= 0) or a negative SK_ERR_* on failure.  The triangle indices are copied
 * into the state; the caller's TriGroupDesc may be discarded after the call.
 */
SK_API int ray_tracer_register_tri_group(
    struct RayTracerState* st,
    const TriGroupDesc*    desc);

/**
 * Drop all registered triangle groups.  Does not free any other state.
 */
SK_API int ray_tracer_clear_tri_groups(struct RayTracerState* st);

/**
 * Number of registered groups.
 */
SK_API int ray_tracer_n_tri_groups(const struct RayTracerState* st);

/**
 * Per-GID manifold dispatch stats (CPU T2 path).
 * Returns SK_OK and fills the out-pointers for group `gid`.
 * out_magic       — payload magic float (14949=param, 14948=MLP, 14946-51=LUT, 0=none)
 * out_transmitted — rays successfully teleported through this surface
 * out_absorbed    — rays absorbed (vignetting, TIR, unknown magic)
 */
SK_API int ray_tracer_get_manifold_gid_stats(
    const struct RayTracerState* st, int gid,
    float*    out_magic,
    uint64_t* out_transmitted,
    uint64_t* out_absorbed);

#ifdef __cplusplus
} /* extern "C" */
#endif
