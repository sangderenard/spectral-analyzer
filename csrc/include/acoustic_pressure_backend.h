/**
 * acoustic_pressure_backend.h — Internal C++ abstract interface for the
 * coevolver's acoustic pressure domain.
 *
 * Two concrete implementations:
 *   UniformPressureBackend  — wraps the existing AcousticFDTDState.
 *   AMRPressureBackend      — wraps AcousticAMRState with Kirchhoff plate coupling.
 *
 * Rules
 * -----
 * - No fallback between backends: construction fails loudly on invalid state.
 * - No Python round-trips per sample: all mic sampler stencils are precomputed.
 * - Plate coupling, bridge impedance, and mic velocity sampling are preserved
 *   in both backends; neither approximates them away.
 * - This header is NOT part of the public C ABI; it is included only by
 *   acoustic_coevolver.cpp.
 */

#pragma once

#include "acoustic_fdtd.h"
#include "acoustic_amr.h"
#include "acoustic_coevolver.h"  /* AMRCoevolverDescriptor (plain-C struct) */
#include <cstdint>

/* ── Abstract backend interface ─────────────────────────────────────────── */

/**
 * IPressureBackend — pure-virtual C++ interface used by AcousticCoEvolverState
 * to dispatch into either the uniform FDTD or the AMR pressure solver.
 *
 * All sampling functions must be called after the corresponding step().
 * setup_mic_samplers() must be called exactly once after construction, before
 * the first sample_mics() call.
 */
struct IPressureBackend {
    virtual ~IPressureBackend() = default;

    /* ── Timing ──────────────────────────────────────────────────────────── */

    /** CFL-stable time step (seconds). */
    virtual double get_dt() const = 0;

    /* ── Stepping ────────────────────────────────────────────────────────── */

    /**
     * Advance the pressure domain by n_steps time steps.
     * Returns 0 on success, FDTD_ERR_UNSTABLE / AMR_ERR_DIVERGED on divergence.
     */
    virtual int step(int n_steps) = 0;

    /**
     * Reset pressure, velocity, and plate fields to zero (keep geometry).
     */
    virtual int reset() = 0;

    /* ── Bridge / structural excitation ─────────────────────────────────── */

    /**
     * Inject per-cell bridge structural forces (Newtons) into the plate.
     *
     * cell_drives[i] is the force at bridge source cell i.  The backend
     * maps these to plate nodes via its internal bridge kernel and accumulates
     * them into the plate's external force buffer.
     *
     * @param cell_drives  [n_cells] float — force per source cell (N).
     * @param n_cells      Number of bridge source cells (must match construction).
     * @param force_scale  Compatibility multiplier; use 1.0 for SI.
     */
    virtual int inject_bridge_drive(const float* cell_drives,
                                    int          n_cells,
                                    float        force_scale) = 0;

    /**
     * Inject a weighted structural force at arbitrary plate nodes.
     *
     * Used for neck/body joint reaction distribution.
     *
     * @param n_plate       Number of plate target nodes.
     * @param plate_indices [n_plate] flat plate index (j + plate_Ny * i).
     * @param weights       [n_plate] normalized weights.
     * @param total_force_N Total force in Newtons (distributed by weights).
     */
    virtual int inject_plate_force(int          n_plate,
                                   const int*   plate_indices,
                                   const float* weights,
                                   float        total_force_N) = 0;

    /**
     * Sample plate displacement via 4-tap bilinear interpolation.
     *
     * Saddle batch read for the string nut-BC update; avoids copying the full
     * plate displacement array each substep.
     *
     * @param n_points  Number of sample points (typically n_strings).
     * @param idx4      [n_points * 4] flat plate indices (precomputed corners).
     * @param wgt4      [n_points * 4] bilinear weights.
     * @param out_w     [n_points] output displacements (metres).
     */
    virtual int sample_plate_displacement_batch(int          n_points,
                                                const int*   idx4,
                                                const float* wgt4,
                                                float*       out_w) = 0;

    /**
     * Sample a weighted-average plate displacement at a set of plate nodes.
     *
     * Used for neck-body structural coupling to read the plate displacement
     * at the neck-block region.
     */
    virtual int sample_plate_weighted(int          n_plate,
                                      const int*   plate_indices,
                                      const float* weights,
                                      float*       out_w) = 0;

    /* ── Microphone sampling ─────────────────────────────────────────────── */

    /**
     * Precompute mic pressure and velocity samplers.
     *
     * Called once at coevolver create time.  Failure is fatal: if any mic
     * position cannot be mapped to a valid AMR pressure cell or velocity face
     * stencil, returns a non-zero error code and the coevolver construction
     * must fail.
     *
     * @param n_mics      Number of microphones.
     * @param pos_xyz     [n_mics * 3] float — world-space positions (metres).
     * @param axis_xyz    [n_mics * 3] float — polar axis unit vectors.
     * @param need_veloc  1 if any mic has polar_b != 0 (needs velocity stencil).
     */
    virtual int setup_mic_samplers(int          n_mics,
                                   const float* pos_xyz,
                                   const float* axis_xyz,
                                   int          need_veloc) = 0;

    /**
     * Sample all microphones using precomputed stencils.
     *
     * Fills [n_mics] pressure and three velocity component arrays.
     * Must be called after step().
     *
     * @param n_mics   Number of microphones (must equal setup count).
     * @param out_p    [n_mics] pressure in Pa.
     * @param out_vx   [n_mics] x-component of particle velocity (m/s).
     * @param out_vy   [n_mics] y-component.
     * @param out_vz   [n_mics] z-component.
     */
    virtual int sample_mics(int    n_mics,
                            float* out_p,
                            float* out_vx,
                            float* out_vy,
                            float* out_vz) = 0;

    /* ── Field export ────────────────────────────────────────────────────── */

    /**
     * Copy the acoustic pressure field into a caller-supplied buffer.
     *
     * For UniformPressureBackend: flat Nx*Ny*Nz array (index k+Nz*(j+Ny*i)).
     * For AMRPressureBackend: flat n_cells array (AMR cell order).
     * The caller queries get_pressure_field_size() to allocate.
     *
     * @return 0 on success, FDTD_ERR_DIM / AMR error on size mismatch.
     */
    virtual int get_pressure_field_size() const = 0;
    virtual int get_pressure_field_flat(float* out, int out_len) = 0;

    /**
     * Copy AMR cell centres into a (n_cells*3) float buffer.
     * Only implemented by AMRPressureBackend; returns SK_ERR_DIM_MISMATCH for
     * UniformPressureBackend (uniform grid has no explicit cell-centre list).
     */
    virtual int get_cell_centers(float* out_xyz, int n_cells_3) {
        (void)out_xyz; (void)n_cells_3; return SK_ERR_DIM_MISMATCH;
    }

    /**
     * Scatter the pressure field onto a uniform (Nx×Ny×Nz) voxel grid.
     *
     * For UniformPressureBackend: reuses the existing flat buffer (the backend
     * already stores a Cartesian grid); Nx, Ny, Nz must match construction dims.
     * For AMRPressureBackend: calls amr_scatter_pressure_uniform().
     *
     * @return 0 on success, non-zero on size/state error.
     */
    virtual int get_pressure_field_uniform(int Nx, int Ny, int Nz,
                                           const double bmin[3],
                                           const double bmax[3],
                                           float* out, int out_len) = 0;

    /**
     * Copy the plate displacement field.
     *
     * For both backends: flat plate_Nx * plate_Ny array (index j + plate_Ny*i).
     *
     * @param out_len Must equal plate_Nx * plate_Ny.
     */
    virtual int get_plate_displacement_size() const = 0;
    virtual int get_plate_displacement(float* out, int out_len) = 0;
};

/* ── UniformPressureBackend ──────────────────────────────────────────────── */

/**
 * Wraps the existing AcousticFDTDState for use through IPressureBackend.
 *
 * Bridge source cells and neck plate cells are registered at construction.
 * Mic pressure/velocity samplers are precomputed via the existing 8-tap
 * trilinear stencil API.
 *
 * This backend stores mic samplers internally (as 8-tap index/weight arrays)
 * and hides the legacy fdtd_precompute_*_samplers / fdtd_sample_*_precomputed
 * pair from the coevolver, which now sees only setup_mic_samplers / sample_mics.
 */
class UniformPressureBackend : public IPressureBackend {
public:
    explicit UniformPressureBackend(AcousticFDTDState* fdtd,
                                    int Nx, int Ny, int Nz,
                                    float dx, const float origin[3],
                                    float rho_air, float c_sound);
    ~UniformPressureBackend() override;

    double get_dt() const override;
    int    step(int n_steps) override;
    int    reset() override;

    int inject_bridge_drive(const float* cell_drives, int n_cells,
                            float force_scale) override;
    int inject_plate_force(int n_plate, const int* plate_indices,
                           const float* weights, float total_force_N) override;
    int sample_plate_displacement_batch(int n_points, const int* idx4,
                                        const float* wgt4, float* out_w) override;
    int sample_plate_weighted(int n_plate, const int* plate_indices,
                              const float* weights, float* out_w) override;

    int setup_mic_samplers(int n_mics, const float* pos_xyz,
                           const float* axis_xyz, int need_veloc) override;
    int sample_mics(int n_mics, float* out_p, float* out_vx,
                    float* out_vy, float* out_vz) override;

    int get_pressure_field_size() const override;
    int get_pressure_field_flat(float* out, int out_len) override;
    int get_pressure_field_uniform(int Nx, int Ny, int Nz,
                                   const double bmin[3], const double bmax[3],
                                   float* out, int out_len) override;
    int get_plate_displacement_size() const override;
    int get_plate_displacement(float* out, int out_len) override;

    /* Access to the underlying FDTD state for bridge kernel registration
     * (called by coevolver_create before constructing this backend). */
    AcousticFDTDState* fdtd() const { return fdtd_; }

private:
    AcousticFDTDState* fdtd_  = nullptr;
    int   Nx_ = 0, Ny_ = 0, Nz_ = 0;
    float dx_   = 0.0f;
    float origin_[3] = {};
    float rho_air_  = 1.21f;
    float c_sound_  = 343.0f;

    /* Precomputed mic samplers (owned here) */
    int    n_mics_   = 0;
    int    need_v_   = 0;
    int*   P_idx8_   = nullptr;  /* [n_mics * 8] pressure stencil flat indices  */
    float* P_wgt8_   = nullptr;  /* [n_mics * 8]                                */
    int*   Vx_idx8_  = nullptr;
    float* Vx_wgt8_  = nullptr;
    int*   Vy_idx8_  = nullptr;
    float* Vy_wgt8_  = nullptr;
    int*   Vz_idx8_  = nullptr;
    float* Vz_wgt8_  = nullptr;
    float* mic_P_    = nullptr;  /* [n_mics] output scratch */
    float* mic_vx_   = nullptr;
    float* mic_vy_   = nullptr;
    float* mic_vz_   = nullptr;
    float* mic_xyz_  = nullptr;  /* [n_mics * 3] grid-space coords */
};

/* ── AMRPressureBackend ──────────────────────────────────────────────────── */

/* AMRCoevolverDescriptor is defined in acoustic_coevolver.h (included above). */

/**
 * Create an AMRPressureBackend from the given descriptor.
 *
 * Validates the descriptor strictly; returns NULL on any failure.
 * The backend copies all descriptor data internally.
 */
IPressureBackend* amr_pressure_backend_create(const AMRCoevolverDescriptor* desc);
