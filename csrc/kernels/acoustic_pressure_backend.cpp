/**
 * acoustic_pressure_backend.cpp — Concrete pressure backend implementations.
 *
 * UniformPressureBackend wraps the existing AcousticFDTDState.
 * AMRPressureBackend wraps AcousticAMRState with plate/bridge/mic coupling.
 *
 * Both inherit IPressureBackend and are created in coevolver_create /
 * coevolver_create_amr respectively.  The coevolver's _step_one_sample
 * dispatches through the virtual interface; no physics is duplicated here.
 */

#include "acoustic_pressure_backend.h"
#include "acoustic_fdtd.h"
#include "acoustic_amr.h"
#include "serial_kernel.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>

/* ============================================================
 * UniformPressureBackend
 * ============================================================ */

UniformPressureBackend::UniformPressureBackend(AcousticFDTDState* fdtd,
                                               int Nx, int Ny, int Nz,
                                               float dx, const float origin[3],
                                               float rho_air, float c_sound)
    : fdtd_(fdtd), Nx_(Nx), Ny_(Ny), Nz_(Nz),
      dx_(dx), rho_air_(rho_air), c_sound_(c_sound)
{
    origin_[0] = origin[0];
    origin_[1] = origin[1];
    origin_[2] = origin[2];
}

UniformPressureBackend::~UniformPressureBackend()
{
    fdtd_destroy(fdtd_);
    free(P_idx8_);
    free(P_wgt8_);
    free(Vx_idx8_); free(Vx_wgt8_);
    free(Vy_idx8_); free(Vy_wgt8_);
    free(Vz_idx8_); free(Vz_wgt8_);
    free(mic_P_);
    free(mic_vx_); free(mic_vy_); free(mic_vz_);
    free(mic_xyz_);
}

double UniformPressureBackend::get_dt() const
{
    return fdtd_ ? (double)fdtd_get_dt(fdtd_) : 0.0;
}

int UniformPressureBackend::step(int n_steps)
{
    if (!fdtd_) return SK_ERR_NULL_STATE;
    int rc = fdtd_step(fdtd_, n_steps);
    if (rc == FDTD_ERR_UNSTABLE) return FDTD_ERR_UNSTABLE;
    return (rc == FDTD_OK) ? SK_OK : SK_ERR_DIM_MISMATCH;
}

int UniformPressureBackend::reset()
{
    if (!fdtd_) return SK_ERR_NULL_STATE;
    fdtd_reset(fdtd_);
    return SK_OK;
}

int UniformPressureBackend::inject_bridge_drive(const float* cell_drives,
                                                int n_cells, float force_scale)
{
    if (!fdtd_ || !cell_drives) return SK_ERR_NULL_STATE;
    int rc = fdtd_inject_bridge_drive(fdtd_, cell_drives, force_scale);
    return (rc == FDTD_OK) ? SK_OK : SK_ERR_DIM_MISMATCH;
}

int UniformPressureBackend::inject_plate_force(int n_plate,
                                               const int* plate_indices,
                                               const float* weights,
                                               float total_force_N)
{
    if (!fdtd_) return SK_ERR_NULL_STATE;
    if (n_plate <= 0 || !plate_indices || !weights) return SK_ERR_DIM_MISMATCH;
    int rc = fdtd_inject_plate_force(fdtd_, n_plate, plate_indices,
                                     weights, total_force_N);
    return (rc == FDTD_OK) ? SK_OK : SK_ERR_DIM_MISMATCH;
}

int UniformPressureBackend::sample_plate_displacement_batch(
    int n_points, const int* idx4, const float* wgt4, float* out_w)
{
    if (!fdtd_ || !idx4 || !wgt4 || !out_w) return SK_ERR_NULL_STATE;
    int rc = fdtd_sample_plate_displacement_batch(fdtd_, n_points,
                                                  idx4, wgt4, out_w);
    return (rc == FDTD_OK) ? SK_OK : SK_ERR_DIM_MISMATCH;
}

int UniformPressureBackend::sample_plate_weighted(int n_plate,
                                                  const int* plate_indices,
                                                  const float* weights,
                                                  float* out_w)
{
    if (!fdtd_ || !plate_indices || !weights || !out_w) return SK_ERR_NULL_STATE;
    int rc = fdtd_sample_plate_weighted(fdtd_, n_plate, plate_indices,
                                        weights, out_w);
    return (rc == FDTD_OK) ? SK_OK : SK_ERR_DIM_MISMATCH;
}

int UniformPressureBackend::setup_mic_samplers(int n_mics, const float* pos_xyz,
                                               const float* axis_xyz,
                                               int need_veloc)
{
    if (!fdtd_ || !pos_xyz || !axis_xyz) return SK_ERR_NULL_STATE;
    if (n_mics <= 0) return SK_ERR_DIM_MISMATCH;

    n_mics_   = n_mics;
    need_v_   = need_veloc;

    /* Pack grid-space coordinates */
    mic_xyz_  = (float*)calloc(n_mics * 3, sizeof(float));
    mic_P_    = (float*)calloc(n_mics,     sizeof(float));
    mic_vx_   = (float*)calloc(n_mics,     sizeof(float));
    mic_vy_   = (float*)calloc(n_mics,     sizeof(float));
    mic_vz_   = (float*)calloc(n_mics,     sizeof(float));
    P_idx8_   = (int*)  calloc(n_mics * 8, sizeof(int));
    P_wgt8_   = (float*)calloc(n_mics * 8, sizeof(float));
    if (!mic_xyz_ || !mic_P_ || !mic_vx_ || !mic_vy_ || !mic_vz_ ||
        !P_idx8_ || !P_wgt8_) return SK_ERR_NULL_STATE;

    memcpy(mic_xyz_, pos_xyz, n_mics * 3 * sizeof(float));

    if (fdtd_precompute_pressure_samplers(fdtd_, n_mics, mic_xyz_,
                                          P_idx8_, P_wgt8_) != FDTD_OK)
        return SK_ERR_DIM_MISMATCH;

    if (need_veloc) {
        Vx_idx8_  = (int*)  calloc(n_mics * 8, sizeof(int));
        Vx_wgt8_  = (float*)calloc(n_mics * 8, sizeof(float));
        Vy_idx8_  = (int*)  calloc(n_mics * 8, sizeof(int));
        Vy_wgt8_  = (float*)calloc(n_mics * 8, sizeof(float));
        Vz_idx8_  = (int*)  calloc(n_mics * 8, sizeof(int));
        Vz_wgt8_  = (float*)calloc(n_mics * 8, sizeof(float));
        if (!Vx_idx8_ || !Vx_wgt8_ || !Vy_idx8_ || !Vy_wgt8_ ||
            !Vz_idx8_ || !Vz_wgt8_) return SK_ERR_NULL_STATE;
        if (fdtd_precompute_velocity_samplers(fdtd_, n_mics, mic_xyz_,
                Vx_idx8_, Vx_wgt8_,
                Vy_idx8_, Vy_wgt8_,
                Vz_idx8_, Vz_wgt8_) != FDTD_OK)
            return SK_ERR_DIM_MISMATCH;
    }
    return SK_OK;
}

int UniformPressureBackend::sample_mics(int n_mics, float* out_p,
                                        float* out_vx, float* out_vy,
                                        float* out_vz)
{
    if (!fdtd_ || !out_p) return SK_ERR_NULL_STATE;
    if (n_mics != n_mics_) return SK_ERR_DIM_MISMATCH;

    if (fdtd_sample_pressure_precomputed(fdtd_, n_mics,
            P_idx8_, P_wgt8_, out_p) != FDTD_OK)
        return SK_ERR_DIM_MISMATCH;

    if (need_v_ && out_vx && out_vy && out_vz) {
        if (fdtd_sample_velocity_precomputed(fdtd_, n_mics,
                Vx_idx8_, Vx_wgt8_,
                Vy_idx8_, Vy_wgt8_,
                Vz_idx8_, Vz_wgt8_,
                out_vx, out_vy, out_vz) != FDTD_OK)
            return SK_ERR_DIM_MISMATCH;
    }
    return SK_OK;
}

int UniformPressureBackend::get_pressure_field_size() const
{
    return Nx_ * Ny_ * Nz_;
}

int UniformPressureBackend::get_pressure_field_flat(float* out, int out_len)
{
    if (!fdtd_ || !out) return SK_ERR_NULL_STATE;
    if (out_len != Nx_ * Ny_ * Nz_) return SK_ERR_DIM_MISMATCH;
    return (fdtd_get_pressure_field(fdtd_, out, out_len) == FDTD_OK)
           ? SK_OK : SK_ERR_DIM_MISMATCH;
}

int UniformPressureBackend::get_pressure_field_uniform(int Nx, int Ny, int Nz,
                                                       const double* /*bmin*/,
                                                       const double* /*bmax*/,
                                                       float* out, int out_len)
{
    /* Uniform backend: already a Cartesian grid.  Caller must match dims. */
    if (Nx != Nx_ || Ny != Ny_ || Nz != Nz_) return SK_ERR_DIM_MISMATCH;
    return get_pressure_field_flat(out, out_len);
}

int UniformPressureBackend::get_plate_displacement_size() const
{
    return Nx_ * Ny_;
}

int UniformPressureBackend::get_plate_displacement(float* out, int out_len)
{
    if (!fdtd_ || !out) return SK_ERR_NULL_STATE;
    if (out_len != Nx_ * Ny_) return SK_ERR_DIM_MISMATCH;
    return (fdtd_get_plate_displacement(fdtd_, out, out_len) == FDTD_OK)
           ? SK_OK : SK_ERR_DIM_MISMATCH;
}

/* ============================================================
 * AMRPressureBackend
 * ============================================================ */

/**
 * AMRPressureBackend — wraps AcousticAMRState.
 *
 * At construction, sets up the plate, bridge sources, and mic samplers
 * from the AMRCoevolverDescriptor.  Fails loudly (returns NULL) on any
 * invalid mapping.
 */
class AMRPressureBackend : public IPressureBackend {
public:
    explicit AMRPressureBackend(AcousticAMRState* amr, int n_cells,
                                int plate_Nx, int plate_Ny,
                                const double bmin[3], const double bmax[3])
        : amr_(amr), n_cells_(n_cells),
          plate_Nx_(plate_Nx), plate_Ny_(plate_Ny)
    {
        bmin_[0] = bmin[0]; bmin_[1] = bmin[1]; bmin_[2] = bmin[2];
        bmax_[0] = bmax[0]; bmax_[1] = bmax[1]; bmax_[2] = bmax[2];
    }

    ~AMRPressureBackend() override { amr_destroy(amr_); }

    double get_dt() const override { return amr_get_dt(amr_); }

    int step(int n_steps) override
    {
        int rc = amr_step(amr_, n_steps);
        if (rc == SK_ERR_DIVERGED) return FDTD_ERR_UNSTABLE; /* coevolver expects this value */
        return rc;
    }

    int reset() override { return amr_reset(amr_); }

    int inject_bridge_drive(const float* cell_drives, int n_cells,
                            float force_scale) override
    {
        return amr_inject_bridge_drive(amr_, cell_drives, n_cells, force_scale);
    }

    int inject_plate_force(int n_plate, const int* plate_indices,
                           const float* weights, float total_force_N) override
    {
        return amr_inject_plate_force(amr_, n_plate, plate_indices,
                                      weights, total_force_N);
    }

    int sample_plate_displacement_batch(int n_points, const int* idx4,
                                        const float* wgt4, float* out_w) override
    {
        return amr_sample_plate_displacement_batch(amr_, n_points, idx4, wgt4, out_w);
    }

    int sample_plate_weighted(int n_plate, const int* plate_indices,
                              const float* weights, float* out_w) override
    {
        return amr_sample_plate_weighted(amr_, n_plate, plate_indices,
                                         weights, out_w);
    }

    int setup_mic_samplers(int n_mics, const float* pos_xyz,
                           const float* axis_xyz, int need_veloc) override
    {
        return amr_setup_mic_samplers(amr_, n_mics, pos_xyz, axis_xyz, need_veloc);
    }

    int sample_mics(int n_mics, float* out_p, float* out_vx,
                    float* out_vy, float* out_vz) override
    {
        return amr_sample_mics(amr_, n_mics, out_p, out_vx, out_vy, out_vz);
    }

    int get_pressure_field_size() const override { return n_cells_; }

    int get_pressure_field_flat(float* out, int out_len) override
    {
        return amr_get_pressure(amr_, out, out_len);
    }

    int get_cell_centers(float* out_xyz, int n_cells_3) override
    {
        return amr_get_cell_centers(amr_, out_xyz, n_cells_3);
    }

    int get_pressure_field_uniform(int Nx, int Ny, int Nz,
                                   const double bmin[3], const double bmax[3],
                                   float* out, int out_len) override
    {
        if (out_len != Nx * Ny * Nz) return SK_ERR_DIM_MISMATCH;
        return amr_scatter_pressure_uniform(amr_, Nx, Ny, Nz, bmin, bmax, out);
    }

    int get_plate_displacement_size() const override
    {
        return plate_Nx_ * plate_Ny_;
    }

    int get_plate_displacement(float* out, int out_len) override
    {
        return amr_get_plate_displacement(amr_, out, out_len);
    }

private:
    AcousticAMRState* amr_       = nullptr;
    int               n_cells_   = 0;
    int               plate_Nx_  = 0;
    int               plate_Ny_  = 0;
    double            bmin_[3]   = {};
    double            bmax_[3]   = {};
};

/* ============================================================
 * amr_pressure_backend_create
 * ============================================================ */

IPressureBackend* amr_pressure_backend_create(const AMRCoevolverDescriptor* desc)
{
    if (!desc) return nullptr;

    /* ── Validate required fields ── */
    if (desc->n_cells <= 0 || desc->n_faces <= 0) return nullptr;
    if (!desc->cell_centers || !desc->cell_volumes || !desc->open_volume_frac
        || !desc->cell_types || !desc->cell_levels) return nullptr;
    if (!desc->face_cell_neg || !desc->face_cell_pos || !desc->face_area
        || !desc->face_open_frac || !desc->face_distance) return nullptr;
    if (desc->c <= 0.0 || desc->rho_air <= 0.0 || desc->min_dx <= 0.0)
        return nullptr;
    if (desc->n_bridge_plate <= 0 || !desc->bridge_plate_idx || !desc->bridge_plate_wgt)
        return nullptr;
    if (desc->plate_Nx <= 0 || desc->plate_Ny <= 0 || desc->plate_dx <= 0.0f)
        return nullptr;
    if (!desc->plate_active || !desc->plate_active_flat_idx) return nullptr;
    if (!desc->plate_face_above_starts || !desc->plate_face_below_starts) return nullptr;
    if (!desc->plate_cell_above || !desc->plate_cell_below) return nullptr;
    if (desc->plate_mass_density <= 0.0f || desc->plate_stiffness_D <= 0.0f) return nullptr;

    /* Validate CFL timestep sanity (at least the grid is not degenerate) */
    double dt = 0.77 * desc->min_dx / (desc->c * std::sqrt(3.0));
    if (!(dt > 0.0) || !(dt < 0.1)) return nullptr; /* 100 ms upper sanity */

    /* ── Create base AMR state ── */
    AcousticAMRState* amr = amr_create(
        desc->n_cells,
        desc->cell_centers,
        desc->cell_volumes,
        desc->open_volume_frac,
        desc->cell_types,
        desc->n_faces,
        desc->face_cell_neg,
        desc->face_cell_pos,
        desc->face_area,
        desc->face_open_frac,
        desc->face_distance,
        desc->c,
        desc->rho_air,
        desc->min_dx);
    if (!amr) return nullptr;

    /* ── Attach Kirchhoff plate ── */
    if (amr_setup_plate(
            amr,
            desc->plate_Nx, desc->plate_Ny,
            desc->plate_dx, desc->plate_origin,
            desc->plate_active,
            desc->plate_mass_density, desc->plate_stiffness_D,
            desc->plate_alpha_M, desc->plate_beta_K,
            desc->n_active_plate,
            desc->plate_active_flat_idx,
            desc->plate_face_above_starts,
            desc->plate_face_above_idx,
            desc->plate_face_above_wgt,
            desc->plate_face_below_starts,
            desc->plate_face_below_idx,
            desc->plate_face_below_wgt,
            desc->plate_cell_above,
            desc->plate_cell_below) != SK_OK) {
        amr_destroy(amr);
        return nullptr;
    }

    /* ── Bridge sources ── */
    if (amr_set_bridge_plate_sources(amr, desc->n_bridge_plate,
                                      desc->bridge_plate_idx,
                                      desc->bridge_plate_wgt) != SK_OK) {
        amr_destroy(amr);
        return nullptr;
    }

    /* ── Neck sources (optional) ── */
    if (desc->n_neck_plate > 0 && desc->neck_plate_idx && desc->neck_plate_wgt) {
        if (amr_set_neck_plate_sources(amr, desc->n_neck_plate,
                                        desc->neck_plate_idx,
                                        desc->neck_plate_wgt) != SK_OK) {
            amr_destroy(amr);
            return nullptr;
        }
    }

    /* ── Border / PML condition ────────────────────────────────────────────── */
    if (desc->n_pml > 0) {
        if (amr_set_border_condition(
                amr,
                desc->border_mode,
                desc->border_sigma_order > 0.0f ? desc->border_sigma_order : 3.0f,
                desc->border_R_reflection,
                desc->border_Z_match,
                desc->n_pml,
                desc->bounds_min,
                desc->bounds_max) != SK_OK) {
            amr_destroy(amr);
            return nullptr;
        }
    }

    auto* backend = new (std::nothrow) AMRPressureBackend(
        amr, desc->n_cells, desc->plate_Nx, desc->plate_Ny,
        desc->bounds_min, desc->bounds_max);
    if (!backend) { amr_destroy(amr); return nullptr; }
    return backend;
}
