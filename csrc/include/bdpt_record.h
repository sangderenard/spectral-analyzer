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
    float    _pad;            /* keep 16-byte alignment */
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

#ifdef __cplusplus
} /* extern "C" */
#endif
