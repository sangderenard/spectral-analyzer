#ifndef MAT_FLAGS_GENERATED_H
#define MAT_FLAGS_GENERATED_H
/* auto-generated from mat_flags.py — DO NOT EDIT */

static constexpr unsigned MAT_FLAG_EMISSIVE = 1u;
static constexpr unsigned MAT_FLAG_REACTIVE = 2u;
static constexpr unsigned MAT_FLAG_ABSORBER = 4u;
static constexpr unsigned MAT_FLAG_NO_SHADOW = 8u;
static constexpr unsigned MAT_FLAG_MANIFOLD = 16u;
static constexpr unsigned MAT_FLAG_PARAMETRIC = 32u;
static constexpr unsigned MAT_FLAG_TRANSMISSIVE = 64u;
static constexpr unsigned MAT_FLAG_APERTURE_STOP = 128u;
static constexpr unsigned MAT_FLAG_PICKING_ONLY = 256u;

static constexpr int MAX_SPECTRAL_BANDS = 32;

/* SSBO binding indices (informational; std430 only) */
static constexpr int BINDING_TRI_GEOM = 0;
static constexpr int BINDING_NODE = 1;
static constexpr int BINDING_TRI_ID = 2;
static constexpr int BINDING_BDPT_SOURCE = 5;
static constexpr int BINDING_PROFILE = 7;
static constexpr int BINDING_SCALE_CONTEXT = 8;
static constexpr int BINDING_TRI_SHADE = 9;
static constexpr int BINDING_MAT = 10;
static constexpr int BINDING_PARAM_SURF = 11;
static constexpr int BINDING_BAKE_SURFACE = 12;
static constexpr int BINDING_BAKE_VOLUME = 13;
static constexpr int BINDING_BAKE_LIFESPAN = 14;

#endif /* MAT_FLAGS_GENERATED_H */
