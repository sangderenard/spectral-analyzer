/**
 * bdpt_record.h — Bidirectional path tracing record formats.
 *
 * Two flat AoS records are exposed here, both designed to be storable as
 * float32 SSBOs (so the GLSL backend can write directly to them with a
 * paired r32f atomic-add for the complex parts) AND directly mappable as
 * NumPy structured arrays on the Python side.
 *
 *   PathVertex     — one record per (subpath, bounce) hit.  Captures the
 *                    geometric and direction state of the path; complex
 *                    amplitude lives in a sibling array (see below).
 *   EndpointRecord — one record per (subpath, band, sensor-hit) crossing.
 *                    Carries the per-band complex amplitude and is the
 *                    canonical storage for "what arrives at the sensor".
 *
 * No reductions: phase is preserved, no abs(), no quantization, no band
 * collapse.  Aggregation into a viewable image is a separate operation
 * applied by the Python display layer.
 */
#pragma once
#include "serial_kernel.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Subpath side discriminator. */
#define BDPT_SIDE_LIGHT     0
#define BDPT_SIDE_SENSOR    1

/* PathVertex layout: 80 bytes, 16-byte aligned for SSBO. */
typedef struct {
    float    pos[3];          /* hit position (m) */
    float    path_length_m;   /* cumulative path length at this vertex */
    float    normal[3];       /* surface normal at hit (or sensor-plane normal) */
    float    pdf_fwd;         /* forward sampling pdf at this vertex */
    float    dir_in[3];       /* incoming direction at hit */
    float    pdf_rev;         /* reverse sampling pdf at this vertex */
    float    dir_out[3];      /* outgoing (post-scatter) direction */
    int32_t  tri_id;          /* hit triangle (-1 = endpoint / virtual plane) */
    int32_t  group_id;        /* TriGroup id (-1 = none) */
    int32_t  bounce_index;    /* 0 = first surface interaction */
    int32_t  side;            /* BDPT_SIDE_LIGHT or BDPT_SIDE_SENSOR */
    uint32_t flags;           /* mat flags at hit */
} PathVertex;

/* Per-band complex amplitude record at a sensor-group hit (or virtual plane). */
typedef struct {
    uint32_t subpath_id;      /* which subpath produced this record */
    uint32_t band_id;         /* which spectral band */
    int32_t  group_id;        /* sensor group that captured it */
    int32_t  vertex_index;    /* index into PathVertex array, -1 if standalone */
    float    pos[3];          /* point on the sensor surface */
    float    pathlen_m;       /* total path length at deposition */
    float    dir[3];          /* arrival direction */
    float    pdf;             /* product of forward sampling pdfs */
    float    amp_re;          /* complex amplitude — real part */
    float    amp_im;          /* complex amplitude — imag part */
    float    cos_theta;       /* |dir · normal|, useful for energy conservation */
    float    stream_id;       /* BDPT_SIDE_LIGHT (0) or BDPT_SIDE_SENSOR (1) */
} EndpointRecord;

/* Ray segment for diagnostic visualization — records each bounce point. */
typedef struct {
    float    p0[3];           /* ray origin (m) */
    float    p1[3];           /* ray endpoint / hit point (m) */
    float    amp_mag;         /* amplitude magnitude at this segment */
    float    mat_idx;         /* material hit (or -1 if no hit) */
    int32_t  bounce_index;    /* 0 = first segment, etc */
    int32_t  pixel_id;        /* which sensor pixel launched this ray */
} RayPathSegment;

#ifdef __cplusplus
static_assert(sizeof(RayPathSegment) == 40, "RayPathSegment layout broken");
#endif

/* Sanity sizes (must match the docs above). */
#ifdef __cplusplus
static_assert(sizeof(PathVertex)     == 80, "PathVertex layout broken");
static_assert(sizeof(EndpointRecord) == 64, "EndpointRecord layout broken");
#endif

/* ── BDPT side-data records ──────────────────────────────────────────────────
 *
 * These are kept strictly separate from the GLSL ray-payload types (RayIntent,
 * RefinedHit, TerminalRecord).  Each record class is emitted into its own
 * PipelineQueue and drained via a dedicated pybind function so the transport
 * channel budget is never affected.
 *
 * Layout rules:
 *   - Every struct is padded to a multiple of 16 bytes.
 *   - alignas(16) is applied in C++ mode so arrays stay cache-line aligned
 *     and SSBO element strides are predictable.
 *   - All structs are directly NumPy-mappable as structured arrays.
 *   - Fixed-size; no pointers, no std::string, no VLAs.
 *   - Static-assert guards for both size and alignment are at the bottom.
 */

/* ── BdptVertexRecord — one row per (subpath, bounce) geometry vertex ──────
 * Raw layout: 104 bytes.  Padded to 112 (7 × 16). */
#ifdef __cplusplus
struct alignas(16) BdptVertexRecord {
#else
typedef struct {
#endif
    uint32_t subpath_id;         /* opaque id assigned at launch */
    uint16_t vertex_index;       /* 0 = first surface interaction */
    uint8_t  stream;             /* BDPT_SIDE_LIGHT or BDPT_SIDE_SENSOR */
    uint8_t  sample_domain;      /* sampling domain at this vertex (see BDPT_DOMAIN_*) */
    uint32_t flags;              /* material/event flags at hit */
    uint32_t strategy_id;        /* which (s,t) strategy produced this subpath */
    int32_t  tri_id;             /* hit triangle (-1 = virtual endpoint) */
    int32_t  group_id;           /* TriGroup id (-1 = none) */
    int32_t  mat_idx;            /* material index (-1 = none) */
    float    pos[3];             /* hit position (m) */
    float    normal[3];          /* surface normal at hit */
    float    dir_in[3];          /* incoming ray direction */
    float    dir_out[3];         /* outgoing (post-scatter) direction */
    float    path_len;           /* cumulative path length at this vertex */
    float    path_at_seg_start;  /* path length at the start of this segment */
    float    pdf_fwd;            /* forward sampling PDF at this vertex */
    float    pdf_rev;            /* reverse sampling PDF at this vertex */
    float    pdf_area;           /* area-measure PDF (converted from solid angle) */
    float    pdf_solid_angle;    /* solid-angle PDF before measure conversion */
    float    throughput_scalar;  /* scalar throughput beta at this vertex */
    float    sensor_origin_y;   /* camera-stream only: sensor launch Y (m); 0 for light stream */
    float    sensor_origin_z;   /* camera-stream only: sensor launch Z (m); 0 for light stream */
#ifdef __cplusplus
};
#else
} BdptVertexRecord;
#endif

/* ── BdptSpectralWeightRecord — per (subpath, vertex, band) complex beta ────
 * Raw layout: 28 bytes.  Padded to 32 (2 × 16). */
#ifdef __cplusplus
struct alignas(16) BdptSpectralWeightRecord {
#else
typedef struct {
#endif
    uint32_t subpath_id;
    uint16_t vertex_index;
    uint16_t band_id;
    float    beta_re;               /* complex throughput β — real part */
    float    beta_im;               /* complex throughput β — imag part */
    float    wavelength_or_center;  /* wavelength (m) or band center frequency (Hz) */
    float    band_pdf;              /* probability of selecting this band */
    float    sensor_rgb_weight;     /* camera sensitivity weight for display */
    uint32_t spectral_sample_id;     /* 0=fixed; otherwise shared cached LUT sample */
#ifdef __cplusplus
};
#else
} BdptSpectralWeightRecord;
#endif

/* ── BdptPdfRecord — per sampling event where measure conversion matters ─────
 * Raw layout: 36 bytes.  Padded to 48 (3 × 16). */
#ifdef __cplusplus
struct alignas(16) BdptPdfRecord {
#else
typedef struct {
#endif
    uint32_t subpath_id;
    uint16_t vertex_index;
    uint8_t  sample_domain;   /* which domain was sampled (see BDPT_DOMAIN_*) */
    uint8_t  measure;         /* PDF measure (see BDPT_MEASURE_*) */
    float    pdf_fwd;
    float    pdf_rev;
    float    pdf_area;
    float    pdf_solid_angle;
    float    jacobian_det;    /* measure-conversion Jacobian (area↔solid-angle) */
    float    geometry_term;   /* |cos θ_0 · cos θ_1| / dist² at connection */
    uint32_t flags;           /* delta/specular/aperture eligibility bits */
    uint8_t  _pad[12];        /* padding to 48 bytes (3 × 16) */
#ifdef __cplusplus
};
#else
} BdptPdfRecord;
#endif

/* ── BdptOpticalEventRecord — per lens/interface/stop event ──────────────────
 * Raw layout: 112 bytes — already a multiple of 16, no padding needed. */
#ifdef __cplusplus
struct alignas(16) BdptOpticalEventRecord {
#else
typedef struct {
#endif
    uint32_t subpath_id;
    uint16_t vertex_index;
    uint16_t element_index;         /* lens element or interface index */
    uint8_t  reason;                /* termination/event reason (see BDPT_OPT_*) */
    uint8_t  stream;                /* BDPT_SIDE_LIGHT or BDPT_SIDE_SENSOR */
    uint16_t flags;
    float    pos[3];                /* interface hit point (m) */
    float    normal[3];             /* interface normal */
    float    dir_in[3];             /* ray direction before interface */
    float    dir_out[3];            /* ray direction after interface (0 if terminated) */
    float    cos_incident;
    float    cos_transmitted;
    float    eta_i;                 /* refractive index of incident medium */
    float    eta_t;                 /* refractive index of transmitted medium */
    float    fresnel_reflectance;   /* Fresnel reflectance [0,1] */
    float    transmittance;         /* Fresnel transmittance [0,1] */
    float    throughput_multiplier; /* net throughput multiplier at this surface */
    float    opl;                   /* optical path length through this element (m) */
    float    geom_len;              /* geometric length through this element (m) */
    float    aperture_radius;       /* clear aperture radius (m) */
    float    transverse_radius;     /* ray transverse radius at this surface (m) */
    float    dist_past_aperture;    /* signed distance past clear aperture (m); >0 = clipped */
    float    phase_space_jacobian;  /* phase-space (étendue) Jacobian at this surface */
#ifdef __cplusplus
};
#else
} BdptOpticalEventRecord;
#endif

/* ── BdptConnectionRecord — per attempted (s,t) connection candidate ─────────
 * Raw layout: 72 bytes.  Padded to 80 (5 × 16). */
#ifdef __cplusplus
struct alignas(16) BdptConnectionRecord {
#else
typedef struct {
#endif
    uint32_t camera_subpath_id;
    uint32_t light_subpath_id;
    uint16_t camera_vertex_index;
    uint16_t light_vertex_index;
    uint16_t strategy_s;          /* number of camera-side vertices in this strategy */
    uint16_t strategy_t;          /* number of light-side vertices in this strategy */
    uint32_t flags;               /* visibility, delta-skip, overflow bits */
    float    p0[3];               /* camera-side connection point (m) */
    float    p1[3];               /* light-side connection point (m) */
    float    dist2;               /* squared distance between p0 and p1 */
    float    cos_camera;          /* |cos θ| at camera-side vertex */
    float    cos_light;           /* |cos θ| at light-side vertex */
    float    geometry_term;       /* |cos_camera · cos_light| / dist2 */
    float    visibility;          /* 1.0 = unoccluded, 0.0 = shadowed */
    float    strategy_pdf;        /* PDF of the chosen (s,t) connection strategy */
    float    mis_weight;          /* MIS weight for this strategy */
    uint8_t  _pad[8];             /* padding to 80 bytes (5 × 16) */
#ifdef __cplusplus
};
#else
} BdptConnectionRecord;
#endif

/* ── Sample domain tags (sample_domain / measure fields) ─────────────────── */
#define BDPT_DOMAIN_UNKNOWN         0u
#define BDPT_DOMAIN_AREA            1u  /* sampled w.r.t. surface area */
#define BDPT_DOMAIN_SOLID_ANGLE     2u  /* sampled w.r.t. solid angle */
#define BDPT_DOMAIN_PROJ_SOLID_ANGLE 3u /* sampled w.r.t. projected solid angle */
#define BDPT_DOMAIN_FILM_AREA       4u  /* sampled on the sensor film plane */
#define BDPT_DOMAIN_APERTURE_AREA   5u  /* sampled on the aperture disc */
#define BDPT_DOMAIN_WAVELENGTH      6u  /* spectral band probability */
#define BDPT_DOMAIN_DISCRETE        7u  /* discrete strategy selection */

/* ── Optical event reason tags (reason field of BdptOpticalEventRecord) ───── */
#define BDPT_OPT_REFRACTION         0u
#define BDPT_OPT_REFLECTION         1u
#define BDPT_OPT_TIR                2u  /* total internal reflection */
#define BDPT_OPT_APERTURE_CLIP      3u  /* ray clipped by aperture stop */
#define BDPT_OPT_VIGNETTE_CLIP      4u  /* ray clipped by vignetting stop */
#define BDPT_OPT_ABSORPTION         5u  /* absorbed at this surface */
#define BDPT_OPT_SENSOR_HIT         6u  /* reached sensor plane */
#define BDPT_OPT_EMISSION           7u  /* emissive surface hit */

/* ── Scatter flags for BdptPdfRecord.flags ────────────────────────────────── */
#define BDPT_PDF_FLAG_DELTA_SPECULAR  (1u << 16)  /* mirror reflection or Snell refraction */
#define BDPT_PDF_FLAG_SPLIT           (1u << 17)  /* deterministic split (max_children >= 2) */
#define BDPT_PDF_FLAG_DIFFUSE         (1u << 18)  /* cosine hemisphere scatter */
#define BDPT_PDF_FLAG_ABSORBED        (1u << 19)  /* path terminated here */
#define BDPT_PDF_FLAG_GGX             (1u << 20)  /* rough microfacet reflection */
#define BDPT_PDF_FLAG_EMISSION        (1u << 21)  /* emissive surface launch */
#define BDPT_PDF_FLAG_SENSOR          (1u << 22)  /* camera measurement endpoint */

/* ── Layout sanity checks ────────────────────────────────────────────────── */
#ifdef __cplusplus
static_assert(sizeof(BdptVertexRecord)         == 112, "BdptVertexRecord layout broken");
static_assert(sizeof(BdptSpectralWeightRecord) == 32,  "BdptSpectralWeightRecord layout broken");
static_assert(sizeof(BdptPdfRecord)            == 48,  "BdptPdfRecord layout broken");
static_assert(sizeof(BdptOpticalEventRecord)   == 112, "BdptOpticalEventRecord layout broken");
static_assert(sizeof(BdptConnectionRecord)     == 80,  "BdptConnectionRecord layout broken");
static_assert(alignof(BdptVertexRecord)         == 16, "BdptVertexRecord alignment broken");
static_assert(alignof(BdptSpectralWeightRecord) == 16, "BdptSpectralWeightRecord alignment broken");
static_assert(alignof(BdptPdfRecord)            == 16, "BdptPdfRecord alignment broken");
static_assert(alignof(BdptOpticalEventRecord)   == 16, "BdptOpticalEventRecord alignment broken");
static_assert(alignof(BdptConnectionRecord)     == 16, "BdptConnectionRecord alignment broken");
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
