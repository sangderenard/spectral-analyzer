/**
 * exposure_backend.cpp — Exposure computation backend implementation.
 *
 * Manages optical assembly ray transport and telemetry aggregation.
 */

#include "exposure_backend.h"
#include <cmath>
#include <cstdlib>

/**
 * Helper: Compute magnitude (amplitude) of complex number.
 */
static double complex_magnitude(double real, double imag)
{
    return std::sqrt(real * real + imag * imag);
}

/**
 * ExposureBackendCpp::create_thin_lens_assembly — build a simple thin lens setup.
 */
void ExposureBackendCpp::create_thin_lens_assembly(
    double focal_length_m,
    double aperture_diameter_m,
    double sensor_distance_m)
{
    /* Free old assembly if it exists */
    if (assembly != nullptr) {
        if (assembly->handlers != nullptr) {
            for (int i = 0; i < assembly->n_handlers; ++i) {
                optical_handler_destroy(assembly->handlers[i]);
            }
            free(assembly->handlers);
        }
        if (assembly->handler_order != nullptr) {
            free(assembly->handler_order);
        }
        free(assembly);
        assembly = nullptr;
    }
    
    /* Allocate space for aperture stop, lens, sensor (3 handlers) */
    OpticalHandler** handlers = (OpticalHandler**)malloc(3 * sizeof(OpticalHandler*));
    int* handler_order = (int*)malloc(3 * sizeof(int));
    
    if (handlers == nullptr || handler_order == nullptr) {
        return;  /* Allocation failed */
    }
    
    /* Create aperture stop geometry */
    OpticalGeometry aperture_geom;
    aperture_geom.center_z_m = 0.0;
    aperture_geom.diameter_m = aperture_diameter_m;
    aperture_geom.radius_m = aperture_diameter_m * 0.5;
    aperture_geom.radius_of_curvature_m = 0.0;
    aperture_geom.conic_k = 0.0;
    aperture_geom.aspheric_a4 = 0.0;
    aperture_geom.aspheric_a6 = 0.0;
    aperture_geom.surface_roughness_m = 0.0;
    
    /* Create lens geometry */
    OpticalGeometry lens_geom = aperture_geom;
    lens_geom.center_z_m = 0.0;
    lens_geom.radius_of_curvature_m = focal_length_m;  /* Simplified: spherical approximation */
    
    /* Create lens material (glass, n=1.5) */
    OpticalMaterial lens_material;
    lens_material.n_real = 1.5;
    lens_material.absorption_coeff_per_m = 0.01;  /* 1% absorption per meter */
    lens_material.fresnel_r_amplitude = 0.04;  /* ~4% Fresnel reflectance at normal incidence */
    lens_material.fresnel_r_phase = 0.0;
    lens_material.dn_dT_per_K = 1.0e-4;  /* dn/dT = 1e-4 per Kelvin */
    
    /* Create handlers in z-order: aperture stop, refractor (lens), sensor */
    handlers[0] = optical_handler_stop_create(&aperture_geom);
    handlers[1] = optical_handler_refractor_create(&lens_geom, &lens_material);
    handlers[2] = optical_handler_sensor_create(&lens_geom);
    
    /* Set handler order */
    handler_order[0] = 0;  /* Stop first */
    handler_order[1] = 1;  /* Then lens */
    handler_order[2] = 2;  /* Then sensor */
    
    /* Create assembly */
    assembly = (OpticalAssembly*)malloc(sizeof(OpticalAssembly));
    if (assembly != nullptr) {
        assembly->handlers = handlers;
        assembly->n_handlers = 3;
        assembly->handler_order = handler_order;
    } else {
        /* Allocation failed, clean up */
        for (int i = 0; i < 3; ++i) {
            optical_handler_destroy(handlers[i]);
        }
        free(handlers);
        free(handler_order);
    }
}

/**
 * ExposureBackendCpp::process_ray — process a ray through the optical assembly.
 */
int ExposureBackendCpp::process_ray(
    const OpticalRayState*  ray_in,
    OpticalRayState*        ray_out,
    int                     camera_mode)
{
    if (!assembly) {
        return 1;  /* No assembly, ray blocked */
    }
    
    /* Initialize output ray */
    *ray_out = *ray_in;
    telemetry.rays_launched++;
    
    /* Track input energy */
    double energy_in = complex_magnitude(ray_in->amplitude_real, ray_in->amplitude_imag);
    telemetry.energy_in += energy_in;
    
    /* Process ray through each optical element in sequence */
    for (int h = 0; h < assembly->n_handlers; ++h) {
        OpticalHandlerResult result = optical_handler_process(
            assembly->handlers[h],  /* Handler pointer from array */
            ray_out,                /* Input is previous stage output */
            ray_out,                /* Output overwrites for next stage */
            camera_mode);
        
        /* Accumulate telemetry from this handler */
        telemetry.rays_blocked_by_stop += result.rays_blocked;
        telemetry.rays_hit_lens_surface += result.rays_hit_surface;
        telemetry.rays_refracted += result.rays_refracted;
        telemetry.rays_reflected += result.rays_reflected;
        telemetry.energy_absorbed += result.rays_absorbed * 1e-9;  /* Map absorbed count to energy */
        telemetry.energy_blocked += result.rays_transmitted * 0;    /* Transmitted rays don't block */
    }
    
    /* Track output energy */
    double energy_out = complex_magnitude(ray_out->amplitude_real, ray_out->amplitude_imag);
    telemetry.energy_out += energy_out;
    
    /* Return 0 if ray reaches sensor (not blocked) */
    if (energy_out > 1e-10) {
        telemetry.rays_deposited_sensor++;
        return 0;
    } else {
        telemetry.rays_blocked_by_stop++;
        return 1;
    }
}
