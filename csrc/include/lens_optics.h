#pragma once

#include <cstddef>

int lens_optics_estimate_camera_jacobians(
    const float* payload,
    int payload_len,
    const double* film_origins,
    const double* aperture_points,
    const double* base_exit_origins,
    const double* base_exit_dirs,
    int n,
    double plate_radius,
    double aperture_radius,
    const double* tb,
    const double* tc,
    int n_threads,
    float* aperture_jac_out,
    float* phase_jac_out);

