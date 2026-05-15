/**
 * optical_handlers.h — Optical element handlers for camera sensor rays.
 *
 * Implements handler interface for optical elements (STOP, MIRROR, REFRACTOR, etc.)
 * that modify ray state during transport through the optical assembly.
 *
 * Handler responsibilities:
 * 1. Accept ray (position, direction, wavelength, amplitude)
 * 2. Apply element-specific physics (blocking, reflection, refraction, etc.)
 * 3. Record event counts for telemetry (rays_hit_lens, rays_refracted, etc.)
 * 4. Update ray state (position, direction, amplitude, phase)
 * 5. Return handler status (continue, block, terminate)
 *
 * Physics Integration:
 * - All handlers operate in world space (meters, radians)
 * - Rays carry complex amplitude per frequency band
 * - Fresnel effects apply to optical surfaces
 * - Mode-based distinction: oracle (perfect) vs physical (with losses)
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────────────
 * Optical Handler Role Enum
 * ────────────────────────────────────────────────────────────────────────── */

typedef enum {
    OPTICAL_ROLE_STOP           = 0,  /* Aperture stop: blocks rays outside aperture */
    OPTICAL_ROLE_MIRROR         = 1,  /* Reflective surface: specular reflection       */
    OPTICAL_ROLE_REFRACTOR      = 2,  /* Refractive surface: refraction + reflection   */
    OPTICAL_ROLE_FILTER         = 3,  /* Wavelength filter: wavelength-dependent loss  */
    OPTICAL_ROLE_DIFFUSER       = 4,  /* Diffuse surface: Lambertian scatter           */
    OPTICAL_ROLE_SENSOR         = 5,  /* Sensor surface: terminal photon collection     */
    OPTICAL_ROLE_WAVE_REGION    = 6   /* Diffraction zone: wave propagation (Phase D)   */
} OpticalHandlerRole;

/* ──────────────────────────────────────────────────────────────────────────
 * Ray State Structure
 * ────────────────────────────────────────────────────────────────────────── */

typedef struct {
    /* Position and direction (world space) */
    double pos[3];                          /* ray origin (m) */
    double dir[3];                          /* unit ray direction */
    
    /* Amplitude and phase (complex amplitude split into real/imag) */
    double amplitude_real;                  /* real part of amplitude (V/m or sqrt(W)) */
    double amplitude_imag;                  /* imaginary part of amplitude */
    double phase_error;                     /* accumulated phase error (rad) */
    
    /* Wavelength/frequency */
    double wavelength_m;                    /* vacuum wavelength (m) */
    
    /* Depth and status */
    double cumulative_path_length_m;        /* total distance traveled (m) */
    int bounce_count;                       /* number of surface hits */
    bool is_active;                         /* ray still propagating? */
    bool hit_sensor;                        /* ray reached terminal surface? */
} OpticalRayState;

/* ──────────────────────────────────────────────────────────────────────────
 * Optical Element Geometry
 * ────────────────────────────────────────────────────────────────────────── */

typedef struct {
    /* Position */
    double center_z_m;                      /* z-position of element centre (m) */
    
    /* Aperture */
    double diameter_m;                      /* element diameter (m) */
    double radius_m;                        /* element radius (m) */
    
    /* Curvature */
    double radius_of_curvature_m;           /* ROC for spherical/aspheric surface (m) */
    
    /* Aspheric coefficients (if applicable) */
    double conic_k;                         /* conic constant for aspheric surface */
    double aspheric_a4;                     /* 4th-order coefficient */
    double aspheric_a6;                     /* 6th-order coefficient */
    
    /* Surface properties */
    double surface_roughness_m;             /* RMS roughness (m) */
} OpticalGeometry;

/* ──────────────────────────────────────────────────────────────────────────
 * Optical Material Properties
 * ────────────────────────────────────────────────────────────────────────── */

typedef struct {
    /* Refractive index */
    double n_real;                          /* real part (dispersion computed separately) */
    
    /* Absorption */
    double absorption_coeff_per_m;          /* absorption α per meter */
    
    /* Fresnel coefficients (for optical surfaces) */
    double fresnel_r_amplitude;             /* reflectance amplitude |R| */
    double fresnel_r_phase;                 /* reflectance phase (rad) */
    
    /* Thermal properties (for focus-error prediction) */
    double dn_dT_per_K;                     /* dn/dT sensitivity (1/K) */
} OpticalMaterial;

/* ──────────────────────────────────────────────────────────────────────────
 * Optical Handler Interface
 * ────────────────────────────────────────────────────────────────────────── */

typedef struct OpticalHandler OpticalHandler;

typedef struct {
    /* Ray path decision: continue, block, or absorb */
    int status;                             /* 0=continue, 1=blocked, 2=absorbed/terminated */
    
    /* Event counters */
    int rays_blocked;
    int rays_hit_surface;
    int rays_refracted;
    int rays_reflected;
    int rays_absorbed;
    int rays_transmitted;
} OpticalHandlerResult;

/**
 * Process a ray through an optical handler element.
 *
 * @param handler       Opaque handler instance (role-specific).
 * @param ray_in        Input ray state (position, direction, amplitude, wavelength).
 * @param ray_out       Output ray state (modified after handler processing).
 * @param mode          Camera mode (0=oracle, >0=physical modes).
 * @return              Handler result with status and event counts.
 */
typedef OpticalHandlerResult (*OpticalHandlerFunc)(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode);

/**
 * Create a STOP handler (aperture blocking).
 *
 * @param geometry      Aperture geometry (diameter, position).
 * @return              Opaque handler pointer.
 */
OpticalHandler* optical_handler_stop_create(
    const OpticalGeometry* geometry);

/**
 * Create a MIRROR handler (specular reflection).
 *
 * @param geometry      Mirror surface geometry (ROC, aspheric coefficients).
 * @param material      Mirror material (reflectance).
 * @return              Opaque handler pointer.
 */
OpticalHandler* optical_handler_mirror_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material);

/**
 * Create a REFRACTOR handler (lens refraction + reflection).
 *
 * @param geometry      Lens surface geometry (ROC, curvature, conic_k).
 * @param material      Glass material (refractive index, absorption).
 * @return              Opaque handler pointer.
 */
OpticalHandler* optical_handler_refractor_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material);

/**
 * Create a FILTER handler (wavelength-dependent absorption).
 *
 * @param geometry      Filter geometry (thickness, position).
 * @param material      Filter material (absorption spectrum).
 * @return              Opaque handler pointer.
 */
OpticalHandler* optical_handler_filter_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material);

/**
 * Create a DIFFUSER handler (Lambertian scattering).
 *
 * @param geometry      Diffuser geometry (roughness, position).
 * @param material      Diffuser material (albedo, roughness).
 * @return              Opaque handler pointer.
 */
OpticalHandler* optical_handler_diffuser_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material);

/**
 * Create a SENSOR handler (terminal surface for photon collection).
 *
 * @param geometry      Sensor geometry (position, dimensions).
 * @return              Opaque handler pointer.
 */
OpticalHandler* optical_handler_sensor_create(
    const OpticalGeometry*  geometry);

/**
 * Process a ray through a handler.
 *
 * @param handler       Opaque handler pointer.
 * @param ray_in        Input ray state.
 * @param ray_out       Output ray state.
 * @param mode          Camera mode (0=oracle, >0=physical).
 * @return              Handler result with status and event counts.
 */
OpticalHandlerResult optical_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode);

/**
 * Destroy a handler and free resources.
 *
 * @param handler       Opaque handler pointer.
 */
void optical_handler_destroy(OpticalHandler* handler);

/* ──────────────────────────────────────────────────────────────────────────
 * Optical Assembly Container
 * ────────────────────────────────────────────────────────────────────────── */

typedef struct {
    OpticalHandler**        handlers;       /* array of element handlers */
    int                     n_handlers;     /* number of elements */
    int*                    handler_order;  /* z-order processing sequence */
} OpticalAssembly;

/**
 * Create an optical assembly from array of handlers.
 *
 * @param handlers      Array of handler pointers (in z-order).
 * @param n_handlers    Number of handlers.
 * @return              Opaque assembly pointer.
 */
OpticalAssembly* optical_assembly_create(
    OpticalHandler** handlers,
    int              n_handlers);

/**
 * Process a ray through all handlers in the assembly.
 *
 * @param assembly      Opaque assembly pointer.
 * @param ray_in        Input ray state.
 * @param ray_out       Output ray state after all elements.
 * @param mode          Camera mode.
 * @return              Aggregated result (combined event counts).
 */
OpticalHandlerResult optical_assembly_process(
    const OpticalAssembly*  assembly,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode);

/**
 * Destroy an assembly and all handlers.
 *
 * @param assembly      Opaque assembly pointer.
 */
void optical_assembly_destroy(OpticalAssembly* assembly);

#ifdef __cplusplus
}
#endif
