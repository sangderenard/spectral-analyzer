/**
 * acoustic_amr.h — C ABI for topology-driven AMR acoustic pressure stepping.
 *
 * The AMR grid is supplied as explicit pressure cells and connecting velocity
 * faces.  Unlike the legacy uniform FDTD solver, this does not assume
 * rectangular Vx/Vy/Vz arrays.  Each face connects two pressure cells and
 * stores its physical area, open-area fraction, and pressure-center distance.
 */
#pragma once

#include "serial_kernel.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AcousticAMRState AcousticAMRState;

SK_API AcousticAMRState* amr_create(
    int            n_cells,
    const double*  cell_centers,      /* (n_cells, 3) */
    const double*  cell_volumes,      /* (n_cells,) */
    const double*  open_volume_frac,  /* (n_cells,) */
    const uint8_t* cell_types,        /* (n_cells,) 0 AIR, 1 WALL, 2 PLATE, 3 PML */
    int            n_faces,
    const int32_t* face_cell_neg,     /* (n_faces,) */
    const int32_t* face_cell_pos,     /* (n_faces,) */
    const double*  face_area,         /* (n_faces,) */
    const double*  face_open_frac,    /* (n_faces,) */
    const double*  face_distance,     /* (n_faces,) */
    double         c,
    double         rho_air,
    double         min_dx,
    int            gradient_order      /* 2 = face-pair gradient, 8 = Fornberg stencil */
);

/**
 * Last amr_create() failure diagnostics for the current thread.
 *
 * Returns a short stage label and message set when amr_create returns NULL.
 * On success, stage is "ok" and message is empty.
 */
SK_API const char* amr_get_last_create_error_stage(void);
SK_API const char* amr_get_last_create_error_message(void);

/**
 * Cross-thread AMR create progress snapshot.
 *
 * These counters are global (not thread-local) and are intended for UI
 * progress reporting while amr_create() is running in another thread.
 */
SK_API int amr_get_create_progress_active(void);      /* 1 while create is running, else 0 */
SK_API int amr_get_create_progress_done_faces(void);  /* completed stencil faces */
SK_API int amr_get_create_progress_total_faces(void); /* total stencil faces */

SK_API void amr_destroy(AcousticAMRState* st);

SK_API int amr_step(AcousticAMRState* st, int n_steps);
SK_API int amr_reset(AcousticAMRState* st);

SK_API int amr_inject_pressure_nearest(
    AcousticAMRState* st,
    const double* xyz,
    float value);

SK_API int amr_get_pressure(
    const AcousticAMRState* st,
    float* out_pressure,
    int out_count);

/**
 * Copy AMR cell centres into a caller-supplied (n_cells*3) float buffer.
 * Layout: out_xyz[3*i+0..2] = (cx, cy, cz) for cell i.
 * n_cells_3 must be >= n_cells * 3.
 * @return SK_OK, SK_ERR_NULL_STATE, or SK_ERR_DIM_MISMATCH.
 */
SK_API int amr_get_cell_centers(
    const AcousticAMRState* st,
    float* out_xyz,
    int n_cells_3);

SK_API int amr_get_velocity(
    const AcousticAMRState* st,
    float* out_velocity,
    int out_count);

SK_API double amr_get_dt(const AcousticAMRState* st);
SK_API int amr_get_step_count(const AcousticAMRState* st);

/**
 * Scatter the AMR cell pressure field onto a uniform (Nx×Ny×Nz) voxel grid
 * for visualization.  For each AMR cell, the nearest voxel index is computed
 * from the cell centre and the supplied bounding box; the cell's pressure is
 * written to that voxel (last-write wins when multiple cells map to the same
 * voxel).  The output buffer is zeroed before scattering.
 *
 * @param out     Caller-supplied float buffer of length Nx*Ny*Nz, indexed
 *                as out[ix*(Ny*Nz) + iy*Nz + iz].
 * @return        SK_OK, SK_ERR_NULL_STATE, or SK_ERR_DIM_MISMATCH.
 */
SK_API int amr_scatter_pressure_uniform(
    const AcousticAMRState* st,
    int Nx, int Ny, int Nz,
    const double bmin[3], const double bmax[3],
    float* out);

/* ── Border condition modes ────────────────────────────────────────────── */

/** Strong absorption; replicates uniform-FDTD PML (~60 dB round-trip loss). */
#define AMR_BORDER_ANECHOIC      0
/** Partial absorption + partial reflection; R_reflection controls residual. */
#define AMR_BORDER_ROOM_PANEL    1
/** Near-zero reflection; reserved for nested-simulation coupling. */
#define AMR_BORDER_TRANSMISSION  2

/**
 * Set absorbing border condition for all PML-type (type=3) cells.
 *
 * Must be called after amr_create() and before the first amr_step().
 * Safe to call multiple times — overwrites any previous condition.
 *
 * @param mode           AMR_BORDER_ANECHOIC / ROOM_PANEL / TRANSMISSION
 * @param sigma_order    Polynomial grading exponent (3.0 = cubic; 0 uses default 3.0)
 * @param R_reflection   Residual reflected energy fraction [0,1] (ROOM_PANEL)
 * @param border_Z_match Outer-medium impedance for TRANSMISSION; 0 = auto (rho*c)
 * @param n_pml          PML thickness in minimum-dx cells
 * @param bounds_min     Domain bounding box minimum corner [3] (double, metres)
 * @param bounds_max     Domain bounding box maximum corner [3] (double, metres)
 * @return SK_OK, or SK_ERR_NULL_STATE if st/bounds are NULL.
 */
SK_API int amr_set_border_condition(
    AcousticAMRState* st,
    int               mode,
    float             sigma_order,
    float             R_reflection,
    float             border_Z_match,
    int               n_pml,
    const double*     bounds_min,
    const double*     bounds_max
);

/* ── Kirchhoff plate setup ─────────────────────────────────────────────── */

/**
 * Attach a Kirchhoff plate to an existing AcousticAMRState.
 *
 * The plate is a uniform 2-D grid of size plate_Nx × plate_Ny with spacing
 * plate_dx, origin plate_origin[3].  The active node mask plate_active[i*plate_Ny+j]
 * selects which nodes participate in the biharmonic solve and AMR coupling.
 *
 * plate_amr_face_above_starts / _idx / _wgt define the ragged mapping from
 * each active plate node to the AMR face(s) immediately above it (interior
 * cavity side).  Similarly for below (exterior / back-plate side).
 * plate_cell_above / _below give the single AMR cell to use for acoustic
 * pressure sampling on each side.
 *
 * Construction fails (returns SK_ERR_DIM_MISMATCH) if:
 *   - n_active_plate doesn't match the number of ones in plate_active.
 *   - Any face index is out of range.
 *   - Any cell index is out of range.
 *
 * @return SK_OK on success.
 */
SK_API int amr_setup_plate(
    AcousticAMRState*  st,
    int                plate_Nx,
    int                plate_Ny,
    float              plate_dx,
    const float        plate_origin[3],
    const uint8_t*     plate_active,          /* [plate_Nx * plate_Ny] */
    float              plate_rho_h,           /* kg/m² */
    float              plate_D,               /* N·m (bending stiffness) */
    float              plate_alpha_M,
    float              plate_beta_K,
    /* AMR coupling — ragged arrays */
    int                n_active_plate,
    const int32_t*     plate_active_flat_idx, /* [n_active_plate] */
    const int32_t*     face_above_starts,     /* [n_active_plate + 1] */
    const int32_t*     face_above_idx,        /* [face_above_starts[n_active_plate]] */
    const float*       face_above_wgt,
    const int32_t*     face_below_starts,     /* [n_active_plate + 1] */
    const int32_t*     face_below_idx,
    const float*       face_below_wgt,
    const int32_t*     cell_above,            /* [n_active_plate] — may be -1 */
    const int32_t*     cell_below             /* [n_active_plate] — may be -1 */
);

/**
 * Register bridge plate source nodes.
 *
 * n must be > 0.  idx[i] = flat plate index j + plate_Ny * i.
 * wgt[i] must sum to 1 across all i (caller normalizes).
 */
SK_API int amr_set_bridge_plate_sources(
    AcousticAMRState*  st,
    int                n,
    const int32_t*     idx,
    const float*       wgt
);

/**
 * Register neck/body plate source nodes (optional; n may be 0).
 */
SK_API int amr_set_neck_plate_sources(
    AcousticAMRState*  st,
    int                n,
    const int32_t*     idx,
    const float*       wgt
);

/* ── Bridge and structural force injection ─────────────────────────────── */

/**
 * Inject per-cell bridge drive forces into the plate.
 *
 * cell_drives[i] is the structural force at bridge source cell i.
 * The AMR state applies the per-node weight from the bridge kernel and
 * accumulates into plate_ext_force.  Must be called before amr_step().
 *
 * This has the same semantics as fdtd_inject_bridge_drive:
 *   plate_ext_force[idx[i]] += wgt[i] * cell_drives[i] * force_scale
 *
 * n_cells must equal the value passed to amr_set_bridge_plate_sources().
 */
SK_API int amr_inject_bridge_drive(
    AcousticAMRState*  st,
    const float*       cell_drives,
    int                n_cells,
    float              force_scale
);

/**
 * Inject an arbitrary distributed plate force (neck reaction).
 *
 * Accumulates total_force_N × wgt[i] into plate_ext_force[plate_idx[i]]
 * for all i in [0, n).
 */
SK_API int amr_inject_plate_force(
    AcousticAMRState*  st,
    int                n,
    const int32_t*     plate_idx,
    const float*       wgt,
    float              total_force_N
);

/* ── Plate displacement sampling ───────────────────────────────────────── */

/**
 * Batch 4-tap bilinear plate displacement sampler (saddle BC).
 *
 * idx4[p*4 + k], wgt4[p*4 + k] for k in 0..3 are the corners of the
 * bilinear stencil for point p.  out_w[p] receives the interpolated value.
 * Identical calling convention to fdtd_sample_plate_displacement_batch.
 */
SK_API int amr_sample_plate_displacement_batch(
    const AcousticAMRState* st,
    int                     n_points,
    const int*              idx4,
    const float*            wgt4,
    float*                  out_w
);

/**
 * Weighted-average plate displacement at n plate nodes (neck coupling read).
 */
SK_API int amr_sample_plate_weighted(
    const AcousticAMRState* st,
    int                     n,
    const int*              plate_idx,
    const float*            wgt,
    float*                  out_w
);

/**
 * Copy the full plate displacement array.
 * out_len must equal plate_Nx * plate_Ny.
 */
SK_API int amr_get_plate_displacement(
    const AcousticAMRState* st,
    float*                  out,
    int                     out_len
);

/* ── Microphone sampler precomputation and sampling ────────────────────── */

/**
 * Precompute stencils for n_mics microphones at world-space positions.
 *
 * For each mic, builds:
 *   - A weighted pressure stencil over nearby AMR cells (inverse-distance).
 *   - If need_velocity != 0: a velocity stencil over AMR faces projected
 *     onto each mic's axis vector (for directional polar patterns).
 *
 * Construction fails (SK_ERR_DIM_MISMATCH) if any mic position lies more
 * than one cell-diameter outside the AMR domain.
 *
 * pos_xyz  [n_mics * 3] world-space mic positions.
 * axis_xyz [n_mics * 3] unit axis vectors (polar axis for velocity sampling).
 */
SK_API int amr_setup_mic_samplers(
    AcousticAMRState*  st,
    int                n_mics,
    const float*       pos_xyz,
    const float*       axis_xyz,
    int                need_velocity
);

/**
 * Sample all microphones using precomputed stencils.
 *
 * Must be called after amr_step().  Fills:
 *   out_p  [n_mics] — pressure (Pa)
 *   out_vx [n_mics] — x-velocity (m/s)
 *   out_vy [n_mics] — y-velocity
 *   out_vz [n_mics] — z-velocity
 *
 * n_mics must match the count used in amr_setup_mic_samplers().
 */
SK_API int amr_sample_mics(
    const AcousticAMRState* st,
    int                     n_mics,
    float*                  out_p,
    float*                  out_vx,
    float*                  out_vy,
    float*                  out_vz
);

#ifdef __cplusplus
}
#endif
