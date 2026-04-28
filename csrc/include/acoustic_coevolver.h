/**
 * acoustic_coevolver.h — Unified C co-evolution engine.
 *
 * Runs the complete instrument physics loop inside a single C module with no
 * Python round-trips per sample:
 *
 *   for each sample:
 *     1. 1D string FDTD for every string   (arbitrary 3-D path, both polarisations)
 *     2. Modal projection (every N_stride steps, lazy update)
 *     3. Bridge velocity sum → FDTD body inject (derivative-coupling model)
 *     4. Kirchhoff plate + 3-D acoustic FDTD sub-steps
 *     5. Pickup integration (single-coil / humbucker / piezo)
 *     6. Mic sampling from FDTD pressure field
 *     7. Plate displacement → string saddle BC (two-way coupling)
 *
 * String model
 * ------------
 * Each string is a 1-D transverse wave along its world-space path (arbitrary
 * 3-D procession), with two polarisations (vertical/lateral) treated
 * independently.  The wave equation per polarisation:
 *
 *   u^{n+1}[i] = 2u^n[i] − u^{n-1}[i]·(1−γ·dt)
 *               + cfl2·(u^n[i+1] − 2u^n[i] + u^n[i-1])
 *
 * where cfl2 = (wave_speed · dt / ds)², γ = damping coefficient (s⁻¹).
 * BCs: nut end u[0] = 0; saddle end u[N-1] = body plate displacement
 * (two-way structural coupling).
 *
 * 3-D procession
 * --------------
 * The string path is supplied as N_segs+1 world-space nodes (x,y,z).  Each
 * segment has a local transverse frame (t̂, n̂, b̂) computed from the path
 * tangent.  String displacement is stored in this frame; the physical velocity
 * field for pickup integration is reconstructed from frame vectors and u/u_prev.
 *
 * Modal tracker
 * -------------
 * For each string, the first N_MODES modes are tracked by projecting u_curr
 * onto sin(kπx/L) shapes.  Complex amplitudes (a_re, a_im) are updated every
 * modal_stride FDTD steps using a running dot-product accumulation.  Exposed
 * for spectral analysis; does not feed back into string dynamics.
 *
 * Pickup types
 * ------------
 * PICKUP_SINGLE_COIL   — Gaussian B-lobe centred on pole, per-string:
 *                         B(s) = exp(−r(s)²/2σ²) where r(s) is distance from
 *                         the pickup pole to string segment s in the pickup
 *                         axis plane.
 *                         Output: V = Σ_s B(s) · v_perp(s) · ds
 *
 * PICKUP_HUMBUCKER     — Two antisymmetric coils separated by coil_spacing:
 *                         B(s) = B_coil1(s) − B_coil2(s)
 *                         Hum cancels: uniform B-field gives zero output.
 *
 * PICKUP_PIEZO         — Saddle slope / force sensor:
 *                         V = sensitivity · T · (u[N-1] − u[N-2]) / ds
 *                         (tension × string slope at saddle, summed over both
 *                         polarisations and all strings whose string_mask bit
 *                         is set.)
 *
 * B-kernels are precomputed at coevolver_create time for every
 * (pickup × string × segment) triplet.
 *
 * Near-silent acoustic (electric guitar)
 * ---------------------------------------
 * Electric guitars have a stiff, heavy body → small but nonzero FDTD pressure.
 * Mic output is obtained by the same fdtd_sample_pressure() path; the caller
 * is free to set plate_mass_density and plate_stiffness_D to electric-guitar
 * values (typical: rho_h = 12.0, D = 2.5) to reduce acoustic output to the
 * physically correct tiny level while keeping the code path identical.
 *
 * Co-evolution with the ray tracer
 * ---------------------------------
 * The acoustic FDTD field (pressure + plate displacement) can be extracted
 * at any time via coevolver_get_pressure_field / get_plate_displacement.
 * Blend with geometric ray-tracer output in Python using the crossover filter
 * defined in acoustic_fdtd_bridge.py.
 *
 * Error codes
 * -----------
 *  0   CE_OK
 * -1   CE_ERR_NULL
 * -2   CE_ERR_DIM
 * -3   CE_ERR_UNSTABLE
 * -4   CE_ERR_PARAM       (invalid parameter)
 * -5   CE_ERR_STRING_IDX  (string index out of range)
 * -6   CE_ERR_BUSY        (async job already running)
 * -7   CE_ERR_FDTD        (generic FDTD error — legacy, prefer the codes below)
 * -8   CE_ERR_FDTD_BRIDGE (fdtd_inject_bridge_drive failed; check st->fdtd and bridge cell count)
 * -9   CE_ERR_FDTD_STEP   (fdtd_step returned a non-unstable error; likely NULL state or bad dims)
 * -10  CE_ERR_FDTD_MIC_P  (fdtd_sample_pressure_precomputed failed; check mic index arrays)
 * -11  CE_ERR_FDTD_MIC_V  (fdtd_sample_velocity_precomputed failed; check mic velocity arrays)
 */

#pragma once
#include "serial_kernel.h"   /* SK_API */
#include "acoustic_fdtd.h"   /* AcousticFDTDState, FDTD_* constants */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes ──────────────────────────────────────────────────────────── */

#define CE_OK            0
#define CE_ERR_NULL     (-1)
#define CE_ERR_DIM      (-2)
#define CE_ERR_UNSTABLE (-3)
#define CE_ERR_PARAM    (-4)
#define CE_ERR_STRING_IDX (-5)
#define CE_ERR_BUSY         (-6)  /**< Call rejected: async job is currently running.               */
#define CE_ERR_FDTD         (-7)  /**< Generic FDTD error — legacy; prefer the codes below.         */
#define CE_ERR_FDTD_BRIDGE  (-8)  /**< fdtd_inject_bridge_drive failed (NULL state or bad dims).    */
#define CE_ERR_FDTD_STEP    (-9)  /**< fdtd_step returned a non-unstable error.                     */
#define CE_ERR_FDTD_MIC_P  (-10)  /**< fdtd_sample_pressure_precomputed failed.                     */
#define CE_ERR_FDTD_MIC_V  (-11)  /**< fdtd_sample_velocity_precomputed failed.                     */

/* ── Async worker status ──────────────────────────────────────────────────── */

#define CE_STATUS_IDLE       0  /**< No job running; ready to accept step_async. */
#define CE_STATUS_RUNNING    1  /**< Background step is in progress.             */
#define CE_STATUS_DONE       2  /**< Last job finished successfully.             */
#define CE_STATUS_CANCELLED  3  /**< Last job was cancelled via coevolver_cancel.*/
#define CE_STATUS_ERROR      4  /**< Last job hit an error; check error_code.   */

/**
 * Progress snapshot returned by coevolver_get_progress.
 *
 * Thread-safe: all fields are read from atomic integers; no lock needed.
 * Poll from any thread (e.g. a Qt timer on the GUI thread) while the worker
 * runs on its own thread.
 */
typedef struct {
    int  samples_done;   /**< Audio samples completed so far.                    */
    int  samples_total;  /**< Total samples requested in the current job.        */
    int  status;         /**< CE_STATUS_* — current worker state.               */
    int  error_code;     /**< CE_OK, or the C error code if status==ERROR.       */
} CoEvolverProgress;

/* ── Pickup type enum ─────────────────────────────────────────────────────── */

/** Standard single-coil magnetic pickup (Strat/Tele-style). */
#define PICKUP_SINGLE_COIL  0
/** Dual-coil humbucker — antisymmetric B-lobe cancels hum. */
#define PICKUP_HUMBUCKER    1
/** Piezo saddle sensor — responds to string slope/force at saddle. */
#define PICKUP_PIEZO        2

/* ── Input descriptor structs ─────────────────────────────────────────────── */

/**
 * Defines a single string's physical properties and world-space path.
 *
 * path_xyz: (n_nodes, 3) float32 array — world-space positions of the string
 *           nodes, from nut (index 0) to saddle (index n_nodes−1).
 *           n_nodes = n_segs + 1.  n_segs is the number of FDTD segments.
 *
 * The segment spacing ds is computed as total arc-length / n_segs; the wave
 * speed c_s = sqrt(tension_N / linear_mass_kgm).
 */
typedef struct {
    int    n_segs;           /**< Number of FDTD segments (interior DOF = n_segs−1). */
    const float* path_xyz;  /**< (n_segs+1, 3) float32 — world-space node positions. */
    float  tension_N;       /**< String tension in Newtons. */
    float  linear_mass_kgm; /**< Linear mass density (kg/m). */
    float  damping;         /**< Damping coefficient γ (s⁻¹). Typical: 5–50. */
    float  stiffness_EI;    /**< Bending stiffness EI (N·m²). 0 = ideal flexible string.
                                 Typical guitar values (wound): 1e-5 – 2e-4 N·m².
                                 Produces physically correct inharmonicity (sharp overtones). */
} CoEvolverStringDef;

/**
 * Defines a pickup transducer.
 *
 * For magnetic pickups (SINGLE_COIL / HUMBUCKER):
 *   pos[3]        — pickup position in world space (pole centre for single-coil;
 *                   midpoint between coils for humbucker).
 *   axis[3]       — pickup axis unit vector (normal to the coil plane, i.e.
 *                   the direction perpendicular to string that couples to flux).
 *   pole_sigma    — Gaussian σ (metres) for the B-lobe.  Typical: 0.003–0.007.
 *   coil_spacing  — coil separation (metres, humbucker only).  Typical: 0.018.
 *
 * For piezo (PIEZO):
 *   sensitivity   — piezo sensitivity (V·m/N).  The slope-force product is
 *                   multiplied by tension_N to get the equivalent force.
 *   string_mask   — bitmask of which strings contribute (bit k = string k).
 */
typedef struct {
    int    type;             /**< PICKUP_SINGLE_COIL / PICKUP_HUMBUCKER / PICKUP_PIEZO */
    float  pos[3];           /**< World position (metres). */
    float  axis[3];          /**< Coil axis unit vector (for magnetic pickups). */
    float  pole_sigma;       /**< B-lobe sigma (metres). */
    float  coil_spacing;     /**< Humbucker coil separation (metres). */
    float  sensitivity;      /**< Piezo sensitivity (V·m/N). */
    uint32_t string_mask;    /**< Bitmask: which strings contribute (all → 0xFFFFFFFF). */
} CoEvolverPickupDef;

/**
 * Defines a microphone sampling position with a parametric polar pattern.
 *
 * The microphone response is:
 *   output = gain · (polar_a · P  +  polar_b · ρ_air · c · v_n)
 * where v_n = dot(v_particle, axis) is the particle velocity component
 * along the microphone axis.
 *
 * Standard pattern presets (polar_a, polar_b):
 *   Omnidirectional :  (1.0,  0.0)
 *   Cardioid         :  (0.5,  0.5)
 *   Supercardioid    :  (0.37, 0.63)
 *   Hypercardioid    :  (0.25, 0.75)
 *   Figure-8         :  (0.0,  1.0)
 *
 * For a purely omnidirectional mic (polar_b = 0), axis is ignored.
 * All patterns are phase-exact: both P and v_n are sampled from the live
 * FDTD fields via staggered trilinear interpolation.
 */
typedef struct {
    float pos[3];      /**< World-space position (metres).                        */
    float axis[3];     /**< Polar axis unit vector (direction of max sensitivity). */
    float polar_a;     /**< Omnidirectional (pressure) weight. Default 1.0.       */
    float polar_b;     /**< Figure-8 (velocity) weight. Default 0.0.              */
    float gain;        /**< Linear output gain (Pa→V or arbitrary). Default 1.0.  */
} CoEvolverMicDef;

/**
 * Parameters for the 3-D acoustic FDTD body domain.
 *
 * cell_type and plate_active arrays are copied internally by coevolver_create.
 * n_pml should be ≥ 8 for adequate absorption; set 0 to disable.
 *
 * bridge_src_xyz  — (n_bridge_src, 3) float32 — world-space positions of the
 *                   bridge saddle injection sites.  Weights are computed as a
 *                   Gaussian kernel centred on each point using bridge_sigma.
 */
typedef struct {
    int    Nx, Ny, Nz;
    float  dx;
    float  origin[3];             /**< World-space centre of grid cell (0,0,0). */
    float  c;                     /**< Speed of sound in air (m/s).  Default 343. */
    float  rho_air;               /**< Air density (kg/m³).  Default 1.21. */
    const uint8_t* cell_type;     /**< (Nx·Ny·Nz,) — copied internally. */
    int    plate_iz;              /**< Z-index of the soundboard layer. */
    const uint8_t* plate_active;  /**< (Nx·Ny,) — copied internally. */
    float  plate_mass_density;    /**< ρ_s·h (kg/m²).  Typical acoustic: 7.0. */
    float  plate_stiffness_D;     /**< Bending stiffness D (N·m).  Typical: 0.45. */
    int    n_pml;                 /**< PML layer thickness in cells.  Typical: 10. */
    /* Bridge injection sites */
    int    n_bridge_src;          /**< Number of bridge saddle points. */
    const float* bridge_src_xyz;  /**< (n_bridge_src, 3) float32 world positions. */
    float  bridge_sigma;          /**< Gaussian σ (metres) for bridge kernel. 0.01. */
} CoEvolverBodyDef;

/* ── Opaque state ─────────────────────────────────────────────────────────── */

typedef struct AcousticCoEvolverState AcousticCoEvolverState;

/* ── Construction / destruction ───────────────────────────────────────────── */

/**
 * Create a co-evolver state.
 *
 * @param n_strings    Number of strings.
 * @param string_defs  (n_strings,) array of string descriptors.
 * @param n_pickups    Number of pickup transducers.
 * @param pickup_defs  (n_pickups,) array of pickup descriptors.
 * @param n_mics       Number of microphone positions.
 * @param mic_defs     (n_mics,) array of mic descriptors.
 * @param body         Body FDTD descriptor.  All fields must be filled.
 * @param sample_rate  Audio sample rate (Hz).  Determines FDTD sub-step count.
 * @param modal_stride Steps between modal projection updates.  Typical: 8–32.
 * @param force_scale  Global bridge force scale (tunes body acoustic level).
 * @return             Opaque handle, NULL on failure.
 */
SK_API AcousticCoEvolverState* coevolver_create(
    int                       n_strings,
    const CoEvolverStringDef* string_defs,
    int                       n_pickups,
    const CoEvolverPickupDef* pickup_defs,
    int                       n_mics,
    const CoEvolverMicDef*    mic_defs,
    const CoEvolverBodyDef*   body,
    float                     sample_rate,
    int                       modal_stride,
    float                     force_scale
);

/** Free all resources.  Safe to call with NULL. */
SK_API void coevolver_destroy(AcousticCoEvolverState* st);

/* ── Excitation ───────────────────────────────────────────────────────────── */

/**
 * Apply an initial pluck displacement to a string.
 *
 * Adds a raised-cosine displacement pulse centred at position_norm (0–1 along
 * string length) to both string displacement fields (u_curr and u_prev), so
 * the leapfrog starts propagating on the next step.
 *
 * @param string_idx    Index of the string (0 = bass E, 5 = treble E for guitar).
 * @param position_norm Fractional position along string (0=nut, 1=saddle).
 * @param amplitude     Peak displacement in metres.  Typical: 1e-3 – 5e-3.
 * @return              CE_OK or CE_ERR_STRING_IDX.
 */
SK_API int coevolver_pluck_string(
    AcousticCoEvolverState* st,
    int   string_idx,
    float position_norm,
    float amplitude
);

/**
 * Inject an external per-string, per-sample force sequence.
 *
 * Used when the string driver is computed externally (e.g. a DriverNode in
 * Python generates a force signal).  The force is applied at force_pos_norm
 * as a distributed injection at the nearest FDTD node.
 *
 * @param string_idx    Which string to drive.
 * @param n_samples     Length of force_buf.
 * @param force_buf     (n_samples,) float32 — external force per sample (N).
 * @param force_pos_norm Fractional position along string for the injection site.
 * @return              CE_OK or error.
 */
SK_API int coevolver_inject_string_force(
    AcousticCoEvolverState* st,
    int          string_idx,
    int          n_samples,
    const float* force_buf,
    float        force_pos_norm
);

/* ── Pluck scheduling ─────────────────────────────────────────────────────── */

/**
 * Pre-register a pluck event for the next coevolver_step / coevolver_step_async
 * run.  Onset sample indices must be non-decreasing across successive calls
 * (i.e. the schedule must be sorted by onset_sample before calling).
 *
 * The C worker fires the pluck immediately before processing the sample at
 * index onset_sample (0-based, relative to the start of the next step call).
 * The schedule is consumed by one step call and is NOT carried over to the next.
 *
 * Call coevolver_clear_pluck_schedule() before rebuilding the schedule for a
 * new render pass.
 *
 * Thread-safety: do NOT call while an async job is running.
 *
 * @param onset_sample  Sample index (0 … n_samples−1) at which to pluck.
 * @param string_idx    String to pluck.
 * @param position_norm Fractional position along string (0=nut, 1=saddle).
 * @param amplitude     Peak displacement in metres.
 * @return              CE_OK or CE_ERR_NULL / CE_ERR_PARAM (bad string_idx).
 */
SK_API int coevolver_schedule_pluck(
    AcousticCoEvolverState* st,
    int   onset_sample,
    int   string_idx,
    float position_norm,
    float amplitude
);

/**
 * Clear all pre-registered pluck events.
 * Call before rebuilding the schedule for a new render pass.
 * Thread-safety: do NOT call while an async job is running.
 */
SK_API void coevolver_clear_pluck_schedule(AcousticCoEvolverState* st);

/* ── Main step ────────────────────────────────────────────────────────────── */

/**
 * Advance the co-evolver by n_samples audio samples.
 *
 * For each audio sample, internally:
 *   1. Compute number of FDTD sub-steps k = ceil(dt_audio / dt_fdtd).
 *   2. For each sub-step:
 *      a. Step all string FDTDs (1-D leapfrog, both polarisations).
 *      b. Accumulate bridge velocity sum from all strings.
 *      c. Inject into body FDTD (derivative coupling model).
 *      d. Step body plate + pressure FDTD.
 *      e. Read plate displacement at saddle → update string saddle BC.
 *   3. After all sub-steps for this audio sample:
 *      a. Integrate pickup signals (B-kernel × string velocity).
 *      b. Sample FDTD pressure at mic positions.
 *      c. If modal_counter % modal_stride == 0: project strings onto modal basis.
 *
 * Outputs are written to the internal pickup_out and mic_out ring buffers,
 * accessible via coevolver_get_pickup_output / coevolver_get_mic_output.
 *
 * @param n_samples  Number of audio samples to advance.
 * @return           CE_OK or CE_ERR_UNSTABLE if body FDTD diverges.
 */
SK_API int coevolver_step(AcousticCoEvolverState* st, int n_samples);

/**
 * Inject per-string drive blocks, advance physics, and return mic output —
 * all in one call, with no Python round-trips between inject and step.
 *
 * This is the primary real-time body-drive API.  The caller supplies one
 * contiguous float32 block per string (each of length n_samples) as the
 * physical excitation signal returned from the driver/voice graph layer.
 * The function:
 *   1. Calls coevolver_inject_string_force(si, string_drive_blocks[si], pos_norm)
 *      for each string i in [0, n_strings).
 *   2. Calls coevolver_step(n_samples) to advance physics.
 *   3. Copies mic output 0 into mic_out_buf (must be at least n_samples floats).
 *
 * @param st                Coevolver state handle.
 * @param string_drive_blocks  Array of n_strings pointers, each pointing to a
 *                             float32 buffer of length n_samples (force in N).
 *                             May be NULL to skip injection (silence drive).
 * @param n_strings         Number of drive blocks provided.  Must equal
 *                          coevolver_get_n_strings(st); CE_ERR_PARAM otherwise.
 * @param n_samples         Block length in audio samples.
 * @param force_pos_norm    Fractional string position for force injection (0–1).
 * @param mic_out_buf       Caller-supplied float32 buffer of length >= n_samples.
 *                          Filled with mic 0 output on CE_OK.
 * @return                  CE_OK, CE_ERR_PARAM, CE_ERR_UNSTABLE, or CE_ERR_DIM.
 */
SK_API int coevolver_step_block_with_drive(
    AcousticCoEvolverState* st,
    const float* const*     string_drive_blocks,
    int                     n_strings,
    int                     n_samples,
    float                   force_pos_norm,
    float*                  mic_out_buf
);

/**
 * Launch a background thread to advance the co-evolver by n_samples.
 *
 * Returns immediately.  The caller can poll progress with
 * coevolver_get_progress() or block until done with coevolver_wait().
 *
 * Returns CE_ERR_PARAM if a job is already running.
 * Returns CE_ERR_NULL  if st is NULL or not initialised.
 */
SK_API int coevolver_step_async(AcousticCoEvolverState* st, int n_samples);

/**
 * Fill *out with the current progress snapshot (thread-safe, non-blocking).
 */
SK_API void coevolver_get_progress(const AcousticCoEvolverState* st,
                                    CoEvolverProgress*            out);

/**
 * Block until the current async job finishes (or has already finished).
 * Returns CE_OK, CE_ERR_UNSTABLE, or the error code from the worker.
 * Safe to call even when no job is running (returns immediately).
 */
SK_API int coevolver_wait(AcousticCoEvolverState* st);

/**
 * Request cancellation of the running async job and block until it exits.
 * Returns CE_OK.  Safe to call when idle.
 */
SK_API int coevolver_cancel(AcousticCoEvolverState* st);

/**
 * Non-blocking poll: 1 if a job is currently running, 0 otherwise.
 */
SK_API int coevolver_is_running(const AcousticCoEvolverState* st);

/* ── Output accessors ─────────────────────────────────────────────────────── */

/**
 * Copy the last n_samples pickup outputs into caller-supplied buffers.
 *
 * @param pickup_idx  Which pickup (0 … n_pickups−1).
 * @param out         (n_samples,) float32 — written with pickup signal.
 * @param n_samples   Samples to retrieve; must be ≤ last coevolver_step count.
 * @return            CE_OK or CE_ERR_NULL / CE_ERR_DIM.
 */
SK_API int coevolver_get_pickup_output(
    const AcousticCoEvolverState* st,
    int    pickup_idx,
    float* out,
    int    n_samples
);

/**
 * Copy the last n_samples mic outputs into caller-supplied buffer.
 *
 * @param mic_idx   Which microphone (0 … n_mics−1).
 * @param out       (n_samples,) float32.
 * @param n_samples Samples to retrieve.
 * @return          CE_OK or CE_ERR_NULL / CE_ERR_DIM.
 */
SK_API int coevolver_get_mic_output(
    const AcousticCoEvolverState* st,
    int    mic_idx,
    float* out,
    int    n_samples
);

/**
 * Copy the full 3-D FDTD pressure field.
 *
 * @param out      (Nx·Ny·Nz,) float32, flat index k+Nz*(j+Ny*i).
 * @param out_len  Must equal Nx*Ny*Nz.
 * @return         CE_OK or CE_ERR_DIM.
 */
SK_API int coevolver_get_pressure_field(
    const AcousticCoEvolverState* st,
    float* out,
    int    out_len
);

/**
 * Copy the 2-D Kirchhoff plate displacement field.
 *
 * @param out      (Nx·Ny,) float32, index j+Ny*i.
 * @param out_len  Must equal Nx*Ny.
 * @return         CE_OK or CE_ERR_DIM.
 */
SK_API int coevolver_get_plate_displacement(
    const AcousticCoEvolverState* st,
    float* out,
    int    out_len
);

/**
 * Copy the complex modal amplitudes for a string.
 *
 * The k-th mode has angular frequency ω_k = k·π·c_s/L.  Amplitudes are
 * updated every modal_stride steps (see coevolver_create).
 *
 * @param string_idx  Which string.
 * @param out_re      (N_MODES,) float32 — real part of modal amplitudes.
 * @param out_im      (N_MODES,) float32 — imaginary part.
 * @param n_modes     Number of modes to retrieve; clamped to internal N_MODES.
 * @return            CE_OK or error.
 */
SK_API int coevolver_get_modal_amplitudes(
    const AcousticCoEvolverState* st,
    int    string_idx,
    float* out_re,
    float* out_im,
    int    n_modes
);

/**
 * Sample the FDTD pressure at arbitrary world-space positions.
 *
 * @param n_rec    Number of receiver positions.
 * @param rec_xyz  (n_rec, 3) float32 — world-space coordinates (metres).
 * @param out_p    (n_rec,) float32 — pressure in Pa at each position.
 * @return         CE_OK or CE_ERR_NULL.
 */
SK_API int coevolver_sample_pressure(
    const AcousticCoEvolverState* st,
    int          n_rec,
    const float* rec_xyz,
    float*       out_p
);

/* ── Per-string transverse velocity field ─────────────────────────────────── */

/**
 * Get the current transverse velocity at each string segment in world space.
 *
 * Returns the 3-D velocity vector (vx, vy, vz) at each segment midpoint,
 * reconstructed from both polarisations and the segment's local frame.
 *
 * @param string_idx  Which string.
 * @param out_v       (n_segs, 3) float32 — velocity vectors in m/s.
 * @param out_len     Must equal n_segs * 3.
 * @return            CE_OK or error.
 */
SK_API int coevolver_get_string_velocity(
    const AcousticCoEvolverState* st,
    int    string_idx,
    float* out_v,
    int    out_len
);

/**
 * Get the current transverse displacement at each string segment in world space.
 *
 * Returns the 3-D displacement vector (dx, dy, dz) at each segment midpoint,
 * reconstructed from both polarisations and the segment's local frame.
 * This is the actual FDTD displacement field u[pol][seg], not a velocity integral.
 *
 * @param string_idx  Which string.
 * @param out_d       (n_segs, 3) float32 — displacement vectors in metres.
 * @param out_len     Must equal n_segs * 3.
 * @return            CE_OK or error.
 */
SK_API int coevolver_get_string_displacement(
    const AcousticCoEvolverState* st,
    int    string_idx,
    float* out_d,
    int    out_len
);

/* ── Queries ──────────────────────────────────────────────────────────────── */

SK_API int   coevolver_get_n_strings (const AcousticCoEvolverState* st);
SK_API int   coevolver_get_n_pickups (const AcousticCoEvolverState* st);
SK_API int   coevolver_get_n_mics    (const AcousticCoEvolverState* st);
SK_API float coevolver_get_dt_audio  (const AcousticCoEvolverState* st);
SK_API float coevolver_get_dt_fdtd   (const AcousticCoEvolverState* st);

/** Reset all string, plate, and pressure fields to zero. */
SK_API int coevolver_reset(AcousticCoEvolverState* st);

/* ── Air envelope / room simulation coupling ──────────────────────────────── */

/**
 * Sample the Kirchhoff–Helmholtz boundary data on a surface surrounding the
 * instrument for use as an acoustic emission source in a room simulation.
 *
 * At each sample point, returns:
 *   P(t)   — instantaneous pressure (Pa)
 *   v_n(t) — particle velocity component along the outward surface normal (m/s)
 *
 * Together (P, v_n) on a closed surface fully characterises the sound field
 * outside that surface; a room simulator can integrate these as Kirchhoff
 * boundary conditions.  The velocity is computed from the live staggered FDTD
 * field — not a pressure-gradient approximation.
 *
 * @param n_surf       Number of surface sample points.
 * @param surf_xyz     (n_surf, 3) float32 — world-space positions (metres).
 * @param surf_normals (n_surf, 3) float32 — outward unit normals at each point.
 * @param out_P        (n_surf,) float32 — pressure in Pa.
 * @param out_vn       (n_surf,) float32 — outward normal velocity in m/s.
 * @return             CE_OK or error.
 */
SK_API int coevolver_get_surface_emission(
    const AcousticCoEvolverState* st,
    int          n_surf,
    const float* surf_xyz,
    const float* surf_normals,
    float*       out_P,
    float*       out_vn
);

/**
 * Sample particle velocity at arbitrary world-space positions.
 *
 * Uses staggered trilinear interpolation, consistent with the pressure field.
 *
 * @param n_rec   Number of positions.
 * @param rec_xyz (n_rec, 3) float32 — world-space coordinates (metres).
 * @param out_vx/vy/vz  (n_rec,) float32 — velocity components (m/s).
 * @return        CE_OK or error.
 */
SK_API int coevolver_sample_velocity(
    const AcousticCoEvolverState* st,
    int          n_rec,
    const float* rec_xyz,
    float*       out_vx,
    float*       out_vy,
    float*       out_vz
);

#ifdef __cplusplus
} /* extern "C" */
#endif
