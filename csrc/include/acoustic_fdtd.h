/**
 * acoustic_fdtd.h — C ABI for the 3-D acoustic FDTD solver.
 *
 * Implements a staggered pressure/particle-velocity acoustic FDTD field on a
 * rectilinear Cartesian grid, coupled to a 2-D Kirchhoff plate model for
 * the instrument soundboard.  Designed to co-evolve with the geometric ray
 * tracer: the FDTD handles wave-domain physics below the Schroeder frequency
 * (standing waves, plate modes, near-field), the ray tracer handles
 * high-frequency geometric reflections.
 *
 * Physics
 * -------
 * Pressure field P(x,y,z,t) and staggered face velocities v are advanced on a
 * uniform grid with cell size dx:
 *
 *   v^{n+1/2} = damp_v · (v^{n−1/2} − dt/rho · grad(P^n))
 *   P^{n+1}   = damp_p · (P^n − rho*c²*dt · div(v^{n+1/2}))
 *
 * where alpha is zero in the interior and increases polynomially in the PML.
 * Solid baffle faces enforce v_n = 0. Moving soundboard faces enforce
 * v_n = plate_velocity, so the acoustic field is driven by actual volume
 * displacement rather than by an arbitrary pressure source.
 *
 * Rigid-wall cells use a Neumann ghost: the Laplacian contribution from any
 * solid-wall face direction is zero (pressure gradient = 0 at rigid wall).
 *
 * Kirchhoff plate (soundboard)
 * ----------------------------
 * A 2-D plate layer at z = plate_iz (the top soundboard) obeys:
 *
 *   ρ_s·h·ẇ̈ + D·∇⁴w = ΔP_air + F_bridge
 *
 * where ΔP_air = P(plate_iz+1) − P(plate_iz−1) is the net acoustic load,
 * F_bridge is the force from bridge saddle injection, and D = E·h³/12(1−ν²)
 * is the bending stiffness.  Discretised with a standard 13-point biharmonic
 * stencil (5th-order accurate), simply-supported boundary at the guitar
 * outline.  Plate velocity dw/dt is imposed as the normal velocity boundary
 * condition on the two air faces adjacent to each active plate cell.
 *
 * Bridge excitation
 * -----------------
 * Bridge excitation is expected to enter through fdtd_inject_bridge_drive():
 * a per-cell structural force in Newtons is converted to a plate load and only
 * the moving Kirchhoff plate couples that energy into the acoustic field.  The
 * older fdtd_inject_bridge(signal, derivative, scale) direct-pressure shortcut
 * is disabled because it bypasses the plate impedance and is not conservative.
 *
 * Volumetric pressure field
 * -------------------------
 * The full 3-D pressure array is exposed via fdtd_get_pressure_field() and
 * can be sliced / rendered directly as a volumetric GL texture for smooth
 * animated visualisation, replacing the coarse nearest-centroid ray
 * accumulation.  Trilinear interpolation is provided for arbitrary receiver
 * point sampling.
 *
 * Co-evolution with the ray tracer
 * ---------------------------------
 * The FDTD and ray tracer share the same geometry and run in parallel:
 *   FDTD  — updates every audio block (sub-stepped to maintain CFL stability),
 *            provides wave-accurate impulse response below ~1.5 kHz.
 *   Tracer — re-traced periodically (e.g., every render pass), provides
 *             geometric high-frequency reflections and late reverberation.
 * The Python bridge blends the two outputs via a crossover filter.
 *
 * Grid conventions
 * ----------------
 * Indices: i ∈ [0, Nx), j ∈ [0, Ny), k ∈ [0, Nz)
 * Flat index: k + Nz*(j + Ny*i)       (innermost = Z, as in body depth)
 * X = lateral (guitar width),  Y = longitudinal (guitar length),
 * Z = depth (back plate → top plate).  The bridge is near z = Nz−2
 * (top plate), the air around the instrument extends beyond the guitar outline.
 *
 * Error codes
 * -----------
 *  0  FDTD_OK
 * -1  FDTD_ERR_NULL
 * -2  FDTD_ERR_DIM
 * -3  FDTD_ERR_UNSTABLE   (CFL violated — dt too large for given dx and c)
 */

#pragma once
#include "serial_kernel.h"   /* SK_API */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Cell type flags ───────────────────────────────────────────────────────── */

/** Air cell — acoustic wave propagates freely. */
#define FDTD_AIR    ((uint8_t)0)
/** Rigid wall — Neumann BC (rigid reflection, zero normal velocity). */
#define FDTD_WALL   ((uint8_t)1)
/** Kirchhoff plate cell — coupled structural-acoustic boundary. */
#define FDTD_PLATE  ((uint8_t)2)
/** PML absorbing layer — same physics as AIR but with strong viscous damping. */
#define FDTD_PML    ((uint8_t)3)

/* ── Error codes ──────────────────────────────────────────────────────────── */

#define FDTD_OK          0
#define FDTD_ERR_NULL   (-1)
#define FDTD_ERR_DIM    (-2)
#define FDTD_ERR_UNSTABLE (-3)

/* ── Opaque state ─────────────────────────────────────────────────────────── */

typedef struct AcousticFDTDState AcousticFDTDState;

/* ── Construction / destruction ───────────────────────────────────────────── */

/**
 * Build an FDTD solver from a voxelised instrument geometry.
 *
 * @param Nx, Ny, Nz           Grid dimensions (see coordinate convention above).
 * @param dx                   Cell size (metres).  Typical: 0.008 m for audio.
 * @param c                    Speed of sound in air (m/s).  Default: 343.0.
 * @param rho_air              Air density (kg/m³).  Default: 1.21.
 * @param cell_type            (Nx·Ny·Nz) uint8 array, flat index k+Nz*(j+Ny*i).
 *                             Values: FDTD_AIR, FDTD_WALL, FDTD_PLATE, FDTD_PML.
 *                             Caller retains ownership; copied internally.
 * @param plate_iz             Z-index of the soundboard (FDTD_PLATE) layer.
 * @param plate_active         (Nx·Ny) uint8 — 1 = plate cell active (inside
 *                             guitar outline), 0 = inactive.  Copied internally.
 * @param plate_mass_density   ρ_s · h  (kg/m²).  Typical spruce top: 7.0.
 * @param plate_stiffness_D    Bending stiffness D = E·h³/12(1−ν²) (N·m).
 *                             Typical 3 mm spruce: 0.45.
 * @param plate_alpha_M        Rayleigh mass-proportional damping rate (s⁻¹).
 *                             Controls low-frequency modal decay. Spruce: 1–3.
 * @param plate_beta_K         Rayleigh stiffness-proportional damping time (s).
 *                             Controls high-frequency modal decay. Spruce: ~1e-5.
 * @param n_pml                Thickness of the PML absorbing layer (cells).
 *                             Typical: 10.  Set 0 to disable PML.
 * @return                     Opaque handle, NULL on failure.
 */
SK_API AcousticFDTDState* fdtd_create(
    int     Nx, int Ny, int Nz,
    float   dx,
    float   c,
    float   rho_air,
    const uint8_t* cell_type,
    int     plate_iz,
    const uint8_t* plate_active,
    float   plate_mass_density,
    float   plate_stiffness_D,
    float   plate_alpha_M,
    float   plate_beta_K,
    int     n_pml
);

/** Free all resources.  Safe to call with NULL. */
SK_API void fdtd_destroy(AcousticFDTDState* st);

/* ── Source injection ─────────────────────────────────────────────────────── */

/**
 * Register the set of bridge source cells and their spatial weights.
 *
 * Called once after fdtd_create; can be re-called to change bridge geometry.
 * The source cells should be FDTD_PLATE or the air cell immediately above the
 * plate (z = plate_iz − 1).
 *
 * @param n_cells      Number of source cells.
 * @param cell_indices (n_cells,) int32 — flat grid indices.
 * @param weights      (n_cells,) float32 — Gaussian kernel weights (need not
 *                     be normalised; normalised internally so sum = 1).
 * @return             FDTD_OK or error code.
 */
SK_API int fdtd_set_bridge_sources(
    AcousticFDTDState* st,
    int          n_cells,
    const int*   cell_indices,
    const float* weights
);

/**
 * Disabled legacy direct-pressure bridge injection.  This API used to convert
 * a waveform and derivative directly into acoustic pressure near the bridge,
 * bypassing the structural plate impedance.  It now returns
 * FDTD_ERR_UNSTABLE so stale callers fail loudly instead of creating
 * non-conservative pressure.
 */
SK_API int fdtd_inject_bridge(
    AcousticFDTDState* st,
    float signal_val,
    float signal_ddt,
    float force_scale
);

/**
 * Per-cell bridge drive injection — routes string tension force through the
 * Kirchhoff plate equation rather than directly into the pressure field.
 *
 * cell_drives[i] (Newtons) is accumulated into plate_ext_force at the
 * plate node corresponding to src_idx[i].  plate_step reads and zeros this
 * buffer, so only the moving plate couples to the acoustic field.  This
 * keeps the bridge→plate→air chain passive and free of non-physical pressure
 * sources that bypass the plate's finite impedance.
 *
 * @param cell_drives  (n_src,) float32 — per-cell bridge force in Newtons.
 * @param force_scale  Compatibility parameter. Use 1.0 for physical units;
 *                     callers should tune the coupled impedances instead of
 *                     using an output gain here.
 * @return             FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_inject_bridge_drive(
    AcousticFDTDState* st,
    const float* cell_drives,
    float        force_scale);

/**
 * Inject a weighted structural force directly into arbitrary plate nodes.
 *
 * total_force_N is distributed by weights over plate_indices (flat j + Ny*i).
 * The implementation converts force to pressure-equivalent surface load by
 * dividing by dx², then plate_step consumes and zeroes it.  This is intended
 * for non-bridge structural couplings such as neck/body joint reactions.
 */
SK_API int fdtd_inject_plate_force(
    AcousticFDTDState* st,
    int          n_plate,
    const int*   plate_indices,
    const float* weights,
    float        total_force_N);

/**
 * Set per-face open-area fractions for cut-cell body boundary modelling.
 *
 * Replaces the staircase wall approximation with sub-cell accuracy at the
 * guitar body boundary.  Each fraction ∈ [0, 1]:
 *   0.0 — fully closed (velocity zeroed in BC pass; default for solid-adjacent)
 *   1.0 — fully open   (interior or fully unoccluded; default for air-air)
 *   (0,1) — partial opening at a cut-cell body boundary
 *
 * The BC pass zeros faces where frac == 0; the pressure divergence scales
 * each V contribution by its face fraction to compute partial flux.
 *
 * Call once after fdtd_create to enable smooth staircase-free body walls.
 * Safe to call again to update geometry without rebuilding the solver.
 * Pass NULL for any component to leave it unchanged.
 *
 * Array sizes: vx_frac (Nx-1)*Ny*Nz, vy_frac Nx*(Ny-1)*Nz, vz_frac Nx*Ny*(Nz-1).
 * Use fdtd_get_velocity_dims() to query these sizes.
 *
 * @return FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_set_face_fractions(
    AcousticFDTDState* st,
    const float* vx_frac,
    const float* vy_frac,
    const float* vz_frac);

/**
 * Sample a weighted average of plate displacement at arbitrary plate nodes.
 */
SK_API int fdtd_sample_plate_weighted(
    const AcousticFDTDState* st,
    int          n_plate,
    const int*   plate_indices,
    const float* weights,
    float*       out_w);

/* ── Time stepping ────────────────────────────────────────────────────────── */

/* ── Precomputed sampler construction ────────────────────────────────────── */

/**
 * Precompute 8-tap trilinear indices and weights for pressure sampling.
 *
 * Call once per set of fixed receiver positions.  The resulting idx8/wgt8
 * arrays can be passed to fdtd_sample_pressure_precomputed each step,
 * eliminating per-sample clamping/floor/fraction recomputation.
 *
 * @param n_rec    Number of receiver positions.
 * @param rec_xyz  (n_rec, 3) float32 in grid coords [0..Nx), [0..Ny), [0..Nz).
 * @param out_idx8 (n_rec * 8) int32 — flat P_curr indices for 8 corners.
 * @param out_wgt8 (n_rec * 8) float32 — trilinear weights.
 * @return         FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_precompute_pressure_samplers(
    const AcousticFDTDState* st,
    int          n_rec,
    const float* rec_xyz,
    int*         out_idx8,
    float*       out_wgt8);

/**
 * Precompute 8-tap staggered-trilinear indices and weights for velocity.
 *
 * One set per velocity component (Vx, Vy, Vz), each accounting for the
 * half-cell offset of the respective staggered grid.
 *
 * @return FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_precompute_velocity_samplers(
    const AcousticFDTDState* st,
    int          n_rec,
    const float* rec_xyz,
    int* out_vx_idx8, float* out_vx_wgt8,
    int* out_vy_idx8, float* out_vy_wgt8,
    int* out_vz_idx8, float* out_vz_wgt8);

/**
 * Sample pressure using precomputed 8-tap index/weight arrays.
 * @return FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_sample_pressure_precomputed(
    const AcousticFDTDState* st,
    int          n_rec,
    const int*   idx8,
    const float* wgt8,
    float*       out_p);

/**
 * Sample velocity using precomputed staggered 8-tap index/weight arrays.
 * @return FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_sample_velocity_precomputed(
    const AcousticFDTDState* st,
    int          n_rec,
    const int* vx_idx8, const float* vx_wgt8,
    const int* vy_idx8, const float* vy_wgt8,
    const int* vz_idx8, const float* vz_wgt8,
    float* out_vx, float* out_vy, float* out_vz);

/**
 * Advance the FDTD simulation by n_steps time steps.
 *
 * One step performs:
 *   1. Kirchhoff plate update — structural dynamics + air loading.
 *   2. Face velocity update from pressure gradients.
 *   3. Solid and moving-plate face boundary conditions.
 *   4. Pressure update from velocity divergence.
 *   5. PML absorption through pressure/velocity damping factors.
 *
 * @return FDTD_OK on success, FDTD_ERR_UNSTABLE if pressure diverges.
 */
SK_API int fdtd_step(AcousticFDTDState* st, int n_steps);

/** Query the current time-step count. */
SK_API int   fdtd_get_step_count(const AcousticFDTDState* st);

/** Query the time step size in seconds (computed from CFL condition at create). */
SK_API float fdtd_get_dt(const AcousticFDTDState* st);

/** Reset pressure and plate fields to zero (keep geometry). */
SK_API int fdtd_reset(AcousticFDTDState* st);

/* ── Field extraction ─────────────────────────────────────────────────────── */

/**
 * Copy the full 3-D pressure field into a caller-supplied buffer.
 *
 * @param out      (Nx·Ny·Nz,) float32, index k+Nz*(j+Ny*i).
 * @param out_len  Must equal Nx*Ny*Nz; checked for safety.
 * @return         FDTD_OK or FDTD_ERR_DIM.
 */
SK_API int fdtd_get_pressure_field(
    const AcousticFDTDState* st,
    float* out,
    int    out_len
);

/**
 * Copy the 2-D Kirchhoff plate displacement field.
 *
 * @param out      (Nx·Ny,) float32, index j+Ny*i.
 * @param out_len  Must equal Nx*Ny.
 * @return         FDTD_OK or FDTD_ERR_DIM.
 */
SK_API int fdtd_get_plate_displacement(
    const AcousticFDTDState* st,
    float* out,
    int    out_len
);

/**
 * Sample plate displacement at n_points locations using precomputed
 * bilinear indices and weights — avoids copying the whole plate array.
 *
 * Call once at create-time to compute saddle_idx4 / saddle_wgt4 per string,
 * then call this each substep instead of fdtd_get_plate_displacement.
 *
 * @param n_points     Number of sample points (typically n_strings).
 * @param saddle_idx4  (n_points * 4) int32 — flat plate indices for the four
 *                     bilinear corners, pre-clamped to [0, N_plate).
 *                     Layout: [i0*Ny+j0, i1*Ny+j0, i0*Ny+j1, i1*Ny+j1] per point.
 * @param saddle_wgt4  (n_points * 4) float32 — bilinear weights.
 *                     Layout: [(1-fi)*(1-fj), fi*(1-fj), (1-fi)*fj, fi*fj] per point.
 * @param out_w        (n_points,) float32 — interpolated plate displacement.
 * @return             FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_sample_plate_displacement_batch(
    const AcousticFDTDState* st,
    int          n_points,
    const int*   saddle_idx4,
    const float* saddle_wgt4,
    float*       out_w
);

/**
 * Sample pressure at arbitrary grid-space positions via trilinear interpolation.
 *
 * @param n_rec    Number of receiver points.
 * @param rec_xyz  (n_rec, 3) float32 in grid coordinates [0..Nx), [0..Ny), [0..Nz).
 *                 Out-of-range coordinates are clamped.
 * @param out_p    (n_rec,) float32 — interpolated pressure at each point.
 * @return         FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_sample_pressure(
    const AcousticFDTDState* st,
    int          n_rec,
    const float* rec_xyz,
    float*       out_p
);

/* ── Particle velocity field ──────────────────────────────────────────────── *
 *
 * The staggered particle velocity is co-evolved with the pressure field via
 * the linearised Euler (momentum) equation:
 *
 *   Vx^{n+1/2}[i+1/2,j,k] = Vx^{n-1/2} · (1−α·dt)
 *                           − (dt/(ρ·dx)) · (P^n[i+1,j,k] − P^n[i,j,k])
 *
 * and analogously for Vy and Vz.  This is the exact leapfrog co-integrator
 * of the 2nd-order pressure scheme — the two representations are equivalent.
 * α is the per-face PML coefficient (average of adjacent cells).
 *
 * Storage layout (staggered Yee grid):
 *   Vx: faces (i+1/2,j,k), i∈[0,Nx-2]  → (Nx-1)·Ny·Nz elements
 *        index = k + Nz·(j + Ny·i)
 *   Vy: faces (i,j+1/2,k), j∈[0,Ny-2]  → Nx·(Ny-1)·Nz elements
 *        index = k + Nz·(j + (Ny-1)·i)
 *   Vz: faces (i,j,k+1/2), k∈[0,Nz-2]  → Nx·Ny·(Nz-1) elements
 *        index = k + (Nz-1)·(j + Ny·i)
 */

/** Return sizes of the three staggered velocity arrays. */
SK_API void fdtd_get_velocity_dims(
    const AcousticFDTDState* st,
    int* Nvx,   /**< out: (Nx-1)*Ny*Nz      */
    int* Nvy,   /**< out: Nx*(Ny-1)*Nz      */
    int* Nvz    /**< out: Nx*Ny*(Nz-1)      */
);

/** Copy staggered Vx field.  out_len must equal (Nx-1)*Ny*Nz. */
SK_API int fdtd_get_velocity_x(
    const AcousticFDTDState* st, float* out, int out_len);

/** Copy staggered Vy field.  out_len must equal Nx*(Ny-1)*Nz. */
SK_API int fdtd_get_velocity_y(
    const AcousticFDTDState* st, float* out, int out_len);

/** Copy staggered Vz field.  out_len must equal Nx*Ny*(Nz-1). */
SK_API int fdtd_get_velocity_z(
    const AcousticFDTDState* st, float* out, int out_len);

/**
 * Sample the particle velocity vector at arbitrary grid-space positions.
 *
 * Each component is interpolated from its staggered grid using proper half-cell
 * offset trilinear interpolation:
 *   vx at (gx, gy, gz) → sample Vx at (gx−0.5, gy, gz)
 *   vy at (gx, gy, gz) → sample Vy at (gx, gy−0.5, gz)
 *   vz at (gx, gy, gz) → sample Vz at (gx, gy, gz−0.5)
 *
 * @param n_rec    Number of receiver positions.
 * @param rec_xyz  (n_rec, 3) float32 — grid-space coordinates.
 * @param out_vx   (n_rec,) float32 — x-component of particle velocity (m/s).
 * @param out_vy   (n_rec,) float32 — y-component.
 * @param out_vz   (n_rec,) float32 — z-component.
 * @return         FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_sample_velocity(
    const AcousticFDTDState* st,
    int          n_rec,
    const float* rec_xyz,
    float*       out_vx,
    float*       out_vy,
    float*       out_vz
);

/**
 * Sample pressure AND normal velocity on a closed surface.
 *
 * This is the Kirchhoff–Helmholtz boundary data needed to couple the
 * instrument's acoustic emission into an external room simulation.
 * Together (P, v_n) on a surface enclosing the source fully characterises
 * the radiated sound field outside that surface.
 *
 * @param n_surf       Number of surface sample points.
 * @param surf_xyz     (n_surf, 3) float32 — grid-space positions of surface points.
 * @param surf_normals (n_surf, 3) float32 — outward unit normal at each point.
 * @param out_P        (n_surf,) float32 — pressure (Pa).
 * @param out_vn       (n_surf,) float32 — normal velocity v·n̂ (m/s).
 * @return             FDTD_OK or FDTD_ERR_NULL.
 */
SK_API int fdtd_get_surface_emission(
    const AcousticFDTDState* st,
    int          n_surf,
    const float* surf_xyz,
    const float* surf_normals,
    float*       out_P,
    float*       out_vn
);

#ifdef __cplusplus
} /* extern "C" */
#endif
