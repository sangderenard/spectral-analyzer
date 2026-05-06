/**
 * emitter_angle_kernel.h — Texture scrim/family-of-angles source analysis.
 *
 * This is not a renderer and does not ray trace.  It reduces an emitter
 * appearance/scrim texture to stable source metrics that basic renderers can
 * consume: output scaling, active surface density, centroid, dominant axis,
 * and equivalent cone closure.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EAK_OK = 0,
    EAK_BAD_ARGUMENT = -1,
};

typedef struct EAKMetrics {
    float flux_scale;          /* sum(luminance*alpha) / pixel_count        */
    float active_fraction;     /* active pixels / pixel_count               */
    float active_density;      /* flux_scale / active_fraction              */
    float centroid_u;          /* weighted centroid in [0,1]                */
    float centroid_v;
    float axis_u;              /* dominant 2D axis in UV space              */
    float axis_v;
    float spread_major;        /* sqrt principal covariance eigenvalue      */
    float spread_minor;
    float cone_cos;            /* equivalent spherical-cap cone cosine      */
    float cone_solid_angle;    /* steradians, assumes hemisphere support    */
    float reserved0;
    float reserved1;
    float reserved2;
    float reserved3;
    float reserved4;
} EAKMetrics;

/* rgba_layers is tightly packed (layers, height, width, 4) uint8. */
int eak_analyze_rgba8_layers(const uint8_t* rgba_layers,
                             int width,
                             int height,
                             int layers,
                             float active_threshold,
                             EAKMetrics* out_metrics);

#ifdef __cplusplus
}
#endif
