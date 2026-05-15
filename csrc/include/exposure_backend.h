/**
 * exposure_backend.h — Python-facing exposure computation backend for optical assembly.
 *
 * Bridges OpticalAssembly (C++ handlers) with Python ray tracing pipeline.
 * Manages ray transport through optical element chain and collects telemetry.
 */

#pragma once

#include "optical_handlers.h"
#include <vector>
#include <cstdint>
#include <cstring>

#ifdef __cplusplus

/**
 * ExposureBackendTelemetry — aggregated event and energy counters.
 */
typedef struct {
    uint64_t rays_launched;
    uint64_t rays_blocked_by_stop;
    uint64_t rays_hit_lens_surface;
    uint64_t rays_refracted;
    uint64_t rays_reflected;
    uint64_t rays_total_internal_reflection;
    uint64_t rays_entered_wave_region;
    uint64_t rays_exited_wave_region;
    uint64_t rays_deposited_sensor;
    uint64_t rays_out_of_domain;
    uint64_t rays_fell_back_to_full_solve;
    double energy_in;
    double energy_out;
    double energy_absorbed;
    double energy_blocked;
    double mean_phase_error;
    double mean_focus_error;
} ExposureBackendTelemetry;

/**
 * ExposureBackendCpp — C++ class for optical assembly ray transport backend.
 *
 * Holds:
 * - OpticalAssembly: chain of optical handlers (lens, aperture, sensor, etc.)
 * - Ray state during transport (position, direction, amplitude, wavelength)
 * - Accumulated telemetry (events, energy, phase errors)
 */
class ExposureBackendCpp {
public:
    OpticalAssembly* assembly;
    ExposureBackendTelemetry telemetry;
    
    ExposureBackendCpp(OpticalAssembly* asm_ptr = nullptr)
        : assembly(asm_ptr)
    {
        std::memset(&telemetry, 0, sizeof(telemetry));
    }
    
    /**
     * Create a simple thin lens assembly (aperture stop + lens + sensor).
     * Builds handlers internally and stores in this backend.
     */
    void create_thin_lens_assembly(
        double focal_length_m,
        double aperture_diameter_m,
        double sensor_distance_m);
    
    /**
     * Process a single ray through the optical assembly.
     */
    int process_ray(
        const OpticalRayState*  ray_in,
        OpticalRayState*        ray_out,
        int                     camera_mode);
    
    /**
     * Get accumulated telemetry.
     */
    ExposureBackendTelemetry get_telemetry() const {
        return telemetry;
    }
    
    /**
     * Reset telemetry counters.
     */
    void reset_telemetry() {
        std::memset(&telemetry, 0, sizeof(telemetry));
    }
};

#endif  /* __cplusplus */
