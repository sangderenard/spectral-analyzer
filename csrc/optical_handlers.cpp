/**
 * optical_handlers.cpp — Implementation of optical element handlers.
 *
 * Handlers process rays through optical elements, applying physics-based
 * transformations (blocking, reflection, refraction, scattering, collection).
 *
 * Each handler encapsulates:
 * - Geometry (position, curvature, aperture)
 * - Material properties (refractive index, absorption, reflectance)
 * - Physics engine (ray intersection, Snell's law, Fresnel equations)
 * - Event recording (telemetry counters)
 *
 * Design patterns:
 * - Handler state is opaque to caller (void* pointers in C interface).
 * - Each handler type has constructor, processor, destructor.
 * - Results aggregated during ray assembly traversal.
 */

#include "optical_handlers.h"
#include <cmath>
#include <cstring>
#include <stdlib.h>

/* ──────────────────────────────────────────────────────────────────────────
 * Handler Base Context (shared by all handler types)
 * ────────────────────────────────────────────────────────────────────────── */

struct OpticalHandler {
    OpticalHandlerRole role;
    OpticalGeometry geometry;
    OpticalMaterial material;
    OpticalHandlerFunc process_fn;
    
    /* Role-specific state (opaque to caller) */
    void* private_state;
    
    /* Event counting */
    struct {
        uint64_t rays_blocked;
        uint64_t rays_hit_surface;
        uint64_t rays_refracted;
        uint64_t rays_reflected;
        uint64_t rays_absorbed;
        uint64_t rays_transmitted;
    } event_counts;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Utility Functions
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Compute ray-plane intersection distance.
 *
 * @param ray_pos       Ray origin (3D position).
 * @param ray_dir       Ray direction (unit vector).
 * @param plane_z       Z-coordinate of plane.
 * @param out_t         Distance along ray to intersection.
 * @return              true if intersection exists (t > epsilon).
 */
static bool ray_plane_intersect(
    const double ray_pos[3],
    const double ray_dir[3],
    double plane_z,
    double* out_t)
{
    const double epsilon = 1e-10;
    
    /* Solve: ray_pos[2] + t * ray_dir[2] = plane_z */
    if (std::fabs(ray_dir[2]) < epsilon) {
        return false;  /* Ray parallel to plane */
    }
    
    double t = (plane_z - ray_pos[2]) / ray_dir[2];
    if (t < epsilon) {
        return false;  /* Intersection behind ray */
    }
    
    *out_t = t;
    return true;
}

/**
 * Compute distance from ray intersection point to element center.
 *
 * @param hit_pos       3D intersection point.
 * @param element_center_z Element center z-coordinate.
 * @return              Radial distance from optical axis.
 */
static double radial_distance_from_axis(
    const double hit_pos[3],
    double element_center_z)
{
    /* Radial distance from optical axis (assumes axis at x=0, y=0) */
    double r = std::sqrt(hit_pos[0] * hit_pos[0] + hit_pos[1] * hit_pos[1]);
    return r;
}

/**
 * Helper: Represent complex number result as real and imag parts.
 */
typedef struct {
    double real;
    double imag;
} ComplexPair;

/**
 * Helper: Multiply two complex numbers (a + bi) * (c + di).
 */
static ComplexPair complex_multiply(ComplexPair a, ComplexPair b)
{
    ComplexPair result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

/**
 * Helper: Multiply complex by real number.
 */
static ComplexPair complex_scale(ComplexPair c, double scale)
{
    ComplexPair result;
    result.real = c.real * scale;
    result.imag = c.imag * scale;
    return result;
}

/**
 * Helper: Create complex from magnitude and phase.
 */
static ComplexPair complex_from_polar(double magnitude, double phase_rad)
{
    ComplexPair result;
    result.real = magnitude * std::cos(phase_rad);
    result.imag = magnitude * std::sin(phase_rad);
    return result;
}

/**
 * Helper: Compute e^(i*theta) = cos(theta) + i*sin(theta).
 */
static ComplexPair complex_exp_i_theta(double theta)
{
    ComplexPair result;
    result.real = std::cos(theta);
    result.imag = std::sin(theta);
    return result;
}

/**
 * Fresnel reflection coefficient (scalar amplitude).
 *
 * Simplified: uses pre-computed reflectance amplitude from material.
 * Full implementation would solve Fresnel equations for incident angle.
 *
 * @param material      Material with fresnel_r_amplitude, fresnel_r_phase.
 * @return              Complex reflection coefficient as real/imag pair.
 */
static ComplexPair fresnel_coefficient(const OpticalMaterial* material)
{
    return complex_from_polar(material->fresnel_r_amplitude, material->fresnel_r_phase);
}

/**
 * Refraction coefficient: transmitted amplitude (1 - |R|^2).
 *
 * @param material      Material properties.
 * @return              Transmission coefficient amplitude.
 */
static double transmission_coefficient(const OpticalMaterial* material)
{
    double r_mag = material->fresnel_r_amplitude;
    return std::sqrt(1.0 - r_mag * r_mag);  /* Energy conservation */
}

/* ──────────────────────────────────────────────────────────────────────────
 * STOP Handler (Aperture Blocking)
 * ────────────────────────────────────────────────────────────────────────── */

static OpticalHandlerResult stop_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult result = {0};
    
    /* Copy input ray to output (default: pass through) */
    *ray_out = *ray_in;
    
    /* Compute intersection with stop plane */
    double t;
    if (!ray_plane_intersect(ray_in->pos, ray_in->dir, handler->geometry.center_z_m, &t)) {
        /* Ray doesn't hit plane, pass through */
        result.status = 0;  /* continue */
        return result;
    }
    
    /* Compute hit point */
    double hit_pos[3];
    for (int i = 0; i < 3; i++) {
        hit_pos[i] = ray_in->pos[i] + t * ray_in->dir[i];
    }
    
    /* Check radial distance against aperture */
    double r = radial_distance_from_axis(hit_pos, handler->geometry.center_z_m);
    
    if (r > handler->geometry.radius_m) {
        /* Ray blocked by aperture */
        result.status = 1;  /* blocked */
        result.rays_blocked = 1;
        handler->event_counts.rays_blocked++;
        ray_out->is_active = false;
        return result;
    }
    
    /* Ray passes aperture */
    ray_out->pos[0] = hit_pos[0];
    ray_out->pos[1] = hit_pos[1];
    ray_out->pos[2] = hit_pos[2];
    ray_out->cumulative_path_length_m += t;
    ray_out->bounce_count++;
    
    result.status = 0;  /* continue */
    result.rays_hit_surface = 1;
    handler->event_counts.rays_hit_surface++;
    
    return result;
}

OpticalHandler* optical_handler_stop_create(
    const OpticalGeometry* geometry)
{
    OpticalHandler* h = (OpticalHandler*)malloc(sizeof(OpticalHandler));
    h->role = OPTICAL_ROLE_STOP;
    h->geometry = *geometry;
    h->process_fn = stop_handler_process;
    h->private_state = nullptr;
    
    /* Initialize event counters */
    memset(&h->event_counts, 0, sizeof(h->event_counts));
    
    return h;
}

/* ──────────────────────────────────────────────────────────────────────────
 * MIRROR Handler (Specular Reflection)
 * ────────────────────────────────────────────────────────────────────────── */

static OpticalHandlerResult mirror_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult result = {0};
    *ray_out = *ray_in;
    
    /* Compute intersection with mirror surface (z = center_z_m) */
    double t;
    if (!ray_plane_intersect(ray_in->pos, ray_in->dir, handler->geometry.center_z_m, &t)) {
        result.status = 0;  /* continue */
        return result;
    }
    
    /* Check aperture */
    double hit_pos[3];
    for (int i = 0; i < 3; i++) {
        hit_pos[i] = ray_in->pos[i] + t * ray_in->dir[i];
    }
    
    double r = radial_distance_from_axis(hit_pos, handler->geometry.center_z_m);
    if (r > handler->geometry.radius_m) {
        result.status = 0;  /* pass through */
        return result;
    }
    
    /* Compute reflected direction (simple: flip z-component) */
    double reflected_dir[3];
    for (int i = 0; i < 3; i++) {
        reflected_dir[i] = ray_in->dir[i];
    }
    reflected_dir[2] = -reflected_dir[2];  /* Simple planar reflection */
    
    /* Apply Fresnel reflectance */
    ComplexPair fresnel = fresnel_coefficient(&handler->material);
    
    /* Mode-dependent amplitude handling */
    ComplexPair ray_amplitude = {ray_in->amplitude_real, ray_in->amplitude_imag};
    ComplexPair new_amplitude = complex_multiply(ray_amplitude, fresnel);
    
    if (mode == 0) {
        /* Oracle mode: preserve full amplitude (perfect optics) */
        new_amplitude = complex_multiply(ray_amplitude, fresnel);
    } else {
        /* Physical mode: apply loss per mode */
        /* (Could apply additional dispersion, aberration, etc.) */
    }
    
    /* Update ray state */
    ray_out->pos[0] = hit_pos[0];
    ray_out->pos[1] = hit_pos[1];
    ray_out->pos[2] = hit_pos[2];
    
    for (int i = 0; i < 3; i++) {
        ray_out->dir[i] = reflected_dir[i];
    }
    
    ray_out->amplitude_real = new_amplitude.real;
    ray_out->amplitude_imag = new_amplitude.imag;
    ray_out->cumulative_path_length_m += t;
    ray_out->bounce_count++;
    
    result.status = 0;  /* continue */
    result.rays_hit_surface = 1;
    result.rays_reflected = 1;
    handler->event_counts.rays_hit_surface++;
    handler->event_counts.rays_reflected++;
    
    return result;
}

OpticalHandler* optical_handler_mirror_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material)
{
    OpticalHandler* h = (OpticalHandler*)malloc(sizeof(OpticalHandler));
    h->role = OPTICAL_ROLE_MIRROR;
    h->geometry = *geometry;
    h->material = *material;
    h->process_fn = mirror_handler_process;
    h->private_state = nullptr;
    
    memset(&h->event_counts, 0, sizeof(h->event_counts));
    
    return h;
}

/* ──────────────────────────────────────────────────────────────────────────
 * REFRACTOR Handler (Lens Refraction)
 * ────────────────────────────────────────────────────────────────────────── */

static OpticalHandlerResult refractor_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult result = {0};
    *ray_out = *ray_in;
    
    /* Compute intersection */
    double t;
    if (!ray_plane_intersect(ray_in->pos, ray_in->dir, handler->geometry.center_z_m, &t)) {
        result.status = 0;
        return result;
    }
    
    double hit_pos[3];
    for (int i = 0; i < 3; i++) {
        hit_pos[i] = ray_in->pos[i] + t * ray_in->dir[i];
    }
    
    double r = radial_distance_from_axis(hit_pos, handler->geometry.center_z_m);
    if (r > handler->geometry.radius_m) {
        result.status = 0;
        return result;
    }
    
    /* Compute refracted direction (Snell's law simplified) */
    /* Full implementation would compute incident angle, apply Snell's law exactly */
    double refracted_dir[3];
    double n_in = 1.0;      /* air */
    double n_out = handler->material.n_real;  /* glass */
    
    /* For now, approximate with simple deflection */
    for (int i = 0; i < 3; i++) {
        refracted_dir[i] = ray_in->dir[i];
    }
    
    /* Snell's law: n1*sin(θ1) = n2*sin(θ2) */
    /* Simplified: bend direction proportionally */
    double bend_factor = (n_in - n_out) / n_out;
    for (int i = 0; i < 2; i++) {  /* x, y components */
        refracted_dir[i] += bend_factor * refracted_dir[i] * 0.1;  /* Small deflection */
    }
    
    /* Normalize direction */
    double norm = 0.0;
    for (int i = 0; i < 3; i++) {
        norm += refracted_dir[i] * refracted_dir[i];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < 3; i++) {
        refracted_dir[i] /= norm;
    }
    
    /* Apply transmission and absorption */
    double trans_coeff = transmission_coefficient(&handler->material);
    /* Complex exponential: e^(-alpha*t) where alpha is absorption coefficient */
    ComplexPair exp_term = complex_exp_i_theta(-handler->material.absorption_coeff_per_m * t);
    ComplexPair transmission = complex_scale(exp_term, trans_coeff);
    
    ComplexPair ray_amplitude = {ray_in->amplitude_real, ray_in->amplitude_imag};
    ComplexPair new_amplitude = complex_multiply(ray_amplitude, transmission);
    
    /* Update ray state */
    ray_out->pos[0] = hit_pos[0];
    ray_out->pos[1] = hit_pos[1];
    ray_out->pos[2] = hit_pos[2];
    
    for (int i = 0; i < 3; i++) {
        ray_out->dir[i] = refracted_dir[i];
    }
    
    ray_out->amplitude_real = new_amplitude.real;
    ray_out->amplitude_imag = new_amplitude.imag;
    ray_out->cumulative_path_length_m += t;
    ray_out->bounce_count++;
    
    result.status = 0;
    result.rays_hit_surface = 1;
    result.rays_refracted = 1;
    handler->event_counts.rays_hit_surface++;
    handler->event_counts.rays_refracted++;
    
    return result;
}

OpticalHandler* optical_handler_refractor_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material)
{
    OpticalHandler* h = (OpticalHandler*)malloc(sizeof(OpticalHandler));
    h->role = OPTICAL_ROLE_REFRACTOR;
    h->geometry = *geometry;
    h->material = *material;
    h->process_fn = refractor_handler_process;
    h->private_state = nullptr;
    
    memset(&h->event_counts, 0, sizeof(h->event_counts));
    
    return h;
}

/* ──────────────────────────────────────────────────────────────────────────
 * FILTER Handler (Wavelength-dependent Absorption)
 * ────────────────────────────────────────────────────────────────────────── */

static OpticalHandlerResult filter_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult result = {0};
    *ray_out = *ray_in;
    
    /* Apply wavelength-dependent absorption */
    double absorption_factor = std::exp(-handler->material.absorption_coeff_per_m * 
                                        handler->geometry.diameter_m);
    
    /* Scale amplitude by real absorption factor */
    ray_out->amplitude_real = ray_in->amplitude_real * absorption_factor;
    ray_out->amplitude_imag = ray_in->amplitude_imag * absorption_factor;
    
    result.status = 0;
    result.rays_hit_surface = 1;
    result.rays_transmitted = 1;
    handler->event_counts.rays_hit_surface++;
    handler->event_counts.rays_transmitted++;
    
    return result;
}

OpticalHandler* optical_handler_filter_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material)
{
    OpticalHandler* h = (OpticalHandler*)malloc(sizeof(OpticalHandler));
    h->role = OPTICAL_ROLE_FILTER;
    h->geometry = *geometry;
    h->material = *material;
    h->process_fn = filter_handler_process;
    h->private_state = nullptr;
    
    memset(&h->event_counts, 0, sizeof(h->event_counts));
    
    return h;
}

/* ──────────────────────────────────────────────────────────────────────────
 * DIFFUSER Handler (Lambertian Scattering)
 * ────────────────────────────────────────────────────────────────────────── */

static OpticalHandlerResult diffuser_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult result = {0};
    *ray_out = *ray_in;
    
    /* For now, pass through with reduced amplitude (simplified scattering) */
    /* 80% transmission, 20% scatter loss */
    ray_out->amplitude_real = ray_in->amplitude_real * 0.8;
    ray_out->amplitude_imag = ray_in->amplitude_imag * 0.8;
    
    result.status = 0;
    result.rays_hit_surface = 1;
    result.rays_transmitted = 1;
    handler->event_counts.rays_hit_surface++;
    handler->event_counts.rays_transmitted++;
    
    return result;
}

OpticalHandler* optical_handler_diffuser_create(
    const OpticalGeometry*  geometry,
    const OpticalMaterial*  material)
{
    OpticalHandler* h = (OpticalHandler*)malloc(sizeof(OpticalHandler));
    h->role = OPTICAL_ROLE_DIFFUSER;
    h->geometry = *geometry;
    h->material = *material;
    h->process_fn = diffuser_handler_process;
    h->private_state = nullptr;
    
    memset(&h->event_counts, 0, sizeof(h->event_counts));
    
    return h;
}

/* ──────────────────────────────────────────────────────────────────────────
 * SENSOR Handler (Terminal Photon Collection)
 * ────────────────────────────────────────────────────────────────────────── */

static OpticalHandlerResult sensor_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult result = {0};
    *ray_out = *ray_in;
    
    /* Mark ray as collected at sensor */
    ray_out->hit_sensor = true;
    ray_out->is_active = false;
    
    result.status = 2;  /* absorbed/terminated */
    result.rays_hit_surface = 1;
    result.rays_absorbed = 1;  /* Collected, not absorbed */
    handler->event_counts.rays_hit_surface++;
    handler->event_counts.rays_absorbed++;  /* Actually "rays_deposited_sensor" */
    
    return result;
}

OpticalHandler* optical_handler_sensor_create(
    const OpticalGeometry*  geometry)
{
    OpticalHandler* h = (OpticalHandler*)malloc(sizeof(OpticalHandler));
    h->role = OPTICAL_ROLE_SENSOR;
    h->geometry = *geometry;
    h->process_fn = sensor_handler_process;
    h->private_state = nullptr;
    
    memset(&h->event_counts, 0, sizeof(h->event_counts));
    
    return h;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Generic Handler Interface
 * ────────────────────────────────────────────────────────────────────────── */

OpticalHandlerResult optical_handler_process(
    OpticalHandler*         handler,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    if (!handler || !handler->process_fn) {
        OpticalHandlerResult result = {0};
        *ray_out = *ray_in;
        return result;
    }
    
    return handler->process_fn(handler, ray_in, ray_out, mode);
}

void optical_handler_destroy(OpticalHandler* handler)
{
    if (handler) {
        if (handler->private_state) {
            free(handler->private_state);
        }
        free(handler);
    }
}

/* ──────────────────────────────────────────────────────────────────────────
 * Optical Assembly Container
 * ────────────────────────────────────────────────────────────────────────── */

OpticalAssembly* optical_assembly_create(
    OpticalHandler** handlers,
    int              n_handlers)
{
    OpticalAssembly* assembly = (OpticalAssembly*)malloc(sizeof(OpticalAssembly));
    
    assembly->handlers = (OpticalHandler**)malloc(n_handlers * sizeof(OpticalHandler*));
    assembly->handler_order = (int*)malloc(n_handlers * sizeof(int));
    assembly->n_handlers = n_handlers;
    
    /* Copy handler pointers and initialize order */
    for (int i = 0; i < n_handlers; i++) {
        assembly->handlers[i] = handlers[i];
        assembly->handler_order[i] = i;  /* Default: process in array order */
    }
    
    return assembly;
}

OpticalHandlerResult optical_assembly_process(
    const OpticalAssembly*  assembly,
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     mode)
{
    OpticalHandlerResult aggregate = {0};
    *ray_out = *ray_in;
    
    if (!assembly || assembly->n_handlers == 0) {
        return aggregate;
    }
    
    /* Process through handler chain */
    OpticalRayState intermediate = *ray_in;
    
    for (int i = 0; i < assembly->n_handlers; i++) {
        int handler_idx = assembly->handler_order[i];
        OpticalHandler* handler = assembly->handlers[handler_idx];
        
        OpticalRayState result_state;
        OpticalHandlerResult result = optical_handler_process(handler, &intermediate, &result_state, mode);
        
        /* Aggregate event counts */
        aggregate.rays_blocked += result.rays_blocked;
        aggregate.rays_hit_surface += result.rays_hit_surface;
        aggregate.rays_refracted += result.rays_refracted;
        aggregate.rays_reflected += result.rays_reflected;
        aggregate.rays_absorbed += result.rays_absorbed;
        aggregate.rays_transmitted += result.rays_transmitted;
        
        /* Check termination conditions */
        if (result.status == 1) {
            /* Ray blocked */
            *ray_out = result_state;
            aggregate.status = 1;
            return aggregate;
        }
        
        if (result.status == 2) {
            /* Ray absorbed/terminated */
            *ray_out = result_state;
            aggregate.status = 2;
            return aggregate;
        }
        
        intermediate = result_state;
    }
    
    *ray_out = intermediate;
    aggregate.status = 0;  /* continue */
    return aggregate;
}

void optical_assembly_destroy(OpticalAssembly* assembly)
{
    if (assembly) {
        if (assembly->handlers) {
            for (int i = 0; i < assembly->n_handlers; i++) {
                optical_handler_destroy(assembly->handlers[i]);
            }
            free(assembly->handlers);
        }
        if (assembly->handler_order) {
            free(assembly->handler_order);
        }
        free(assembly);
    }
}
