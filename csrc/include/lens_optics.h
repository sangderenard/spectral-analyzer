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

/* Full signed tangent map in canonical transverse coordinates
 * [q_y, q_z, n*d_y, n*d_z].  The output stores N row-major 4x4 matrices,
 * signed determinants, symplectic residuals, and validity bytes. */
int lens_optics_estimate_phase_space_jacobians(
    const float* payload,
    int payload_len,
    const double* origins,
    const double* directions,
    int n,
    int spectral_lane,
    double n_input,
    double n_output,
    double q_step_m,
    double p_step,
    int n_threads,
    double* matrices_out,
    double* determinants_out,
    double* symplectic_residuals_out,
    unsigned char* valid_out);
