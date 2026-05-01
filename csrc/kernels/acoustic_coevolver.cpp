/**
 * acoustic_coevolver.cpp — Unified C co-evolution engine.
 *
 * Implements the full per-sample loop:
 *   string FDTD (1-D leapfrog, two polarisations, arbitrary 3-D procession)
 *   → modal projection (lazy, every modal_stride steps)
 *   → string endpoint forces → bridge/top impedance → plate load
 *   → Kirchhoff plate + 3-D acoustic FDTD sub-steps
 *   → saddle BC update (two-way plate → string coupling)
 *   → pickup integration (single-coil / humbucker / piezo)
 *   → mic sampling from FDTD field
 *
 * See acoustic_coevolver.h for physics documentation.
 */

#define _USE_MATH_DEFINES
#include "acoustic_coevolver.h"
#include "acoustic_pressure_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <string>

#include <Eigen/Core>

#include <atomic>
#include <thread>

namespace {
thread_local std::string g_coevolver_amr_create_error;
static void coevolver_amr_set_error(const char* stage) {
    g_coevolver_amr_create_error = stage ? stage : "";
}
} // namespace


/* ── Constants ────────────────────────────────────────────────────────────── */

#define N_MODES       32      /* Modal harmonics tracked per string              */
#define MAX_SUBSTEPS  16      /* Safety cap on FDTD sub-steps per audio sample   */
#define OUTPUT_BUF    4096    /* Ring buffer length for pickup/mic output (samples) */
#define STRING_MAX_SEGMENT_SLOPE 0.25f   /* Small-deflection validity limit. */
#define STRING_MAX_STRETCH_FRAC  0.0015f /* Added arc length / open length.  */

/* ── Pluck event ──────────────────────────────────────────────────────────── */

typedef struct {
    int   onset_sample;
    int   string_idx;
    float position_norm;
    float amplitude;
    int   n_duration_samples;  /* 0 = use default ~20 ms draw */
} PluckEvent;

/* ── Internal string state ────────────────────────────────────────────────── */

typedef struct {
    /* Geometry */
    int     n_segs;           /* Number of FDTD segments                        */
    int     n_nodes;          /* n_segs + 1                                     */
    float*  seg_xyz;          /* (n_nodes, 3) — world-space node positions      */
    float*  seg_tangent;      /* (n_segs,  3) — unit tangent per segment        */
    float*  seg_normal;       /* (n_segs,  3) — unit normal (polarisation 0)    */
    float*  seg_binormal;     /* (n_segs,  3) — unit binormal (polarisation 1)  */
    float   total_length;     /* Arc length in metres                           */
    float   ds;               /* Segment spacing ds = total_length / n_segs     */

    /* Wave parameters */
    float   tension_N;
    float   tension_eff;      /* Tension including geometric stretch (N)        */
    float   axial_stiffness_N;/* Axial EA used for geometric tension (N)        */
    float   linear_mass;
    float   wave_speed;       /* c_s = sqrt(T / mu)                             */
    float   cfl2;             /* (wave_speed * dt_s / ds)^2  ≤ 0.9025           */
    float   gamma;            /* Damping coefficient s^{-1}                     */
    float   dt_s;             /* String FDTD time step (≤ dt_fdtd body step)    */
    int     n_substeps;       /* String steps per body FDTD sub-step            */

    /* Displacement fields — two polarisations (p=0 normal, p=1 binormal) */
    float*  u[2];             /* u_curr[pol][node]  — current step              */
    float*  u_prev[2];        /* u_prev[pol][node]  — previous step             */
    float*  u_tmp[2];         /* scratch for new step                           */

    /* External force injection */
    float*  ext_force[2];     /* (n_nodes,) per-node force accumulator          */

    /* Saddle/bridge coupling */
    float   saddle_w;         /* plate displacement at saddle (m)               */
    float   saddle_w_prev;    /* previous plate displacement at saddle (m)      */
    float   bridge_w;         /* bridge/saddle DOF displacement (m)             */
    float   bridge_w_prev;    /* previous bridge/saddle DOF displacement (m)    */
    float   bridge_mass_eff;  /* per-string effective bridge mass (kg)          */
    float   bridge_stiff_eff; /* bridge-to-plate stiffness (N/m)                */
    float   bridge_damp_eff;  /* bridge-to-plate damping (N*s/m)                */

    /* Legacy per-string neck descriptor fields.  The active model is shared
     * at AcousticCoEvolverState level; these are retained for construction
     * compatibility and reset hygiene. */
    int     has_neck_model;
    float   nut_w;            /* current nut displacement (m)                   */
    float   nut_w_prev;       /* previous nut displacement (m)                  */
    float   neck_mass_eff;    /* effective modal mass (kg)                       */
    float   neck_stiff_eff;   /* effective modal stiffness (N/m)                 */
    float   neck_damp_eff;    /* effective modal damping (N·s/m)                 */

    /* Per-step effective damping — set by _step_one_sample from gamma*gamma_scale */
    float   gamma_eff;

    /* Modal state */
    float   modal_re[N_MODES];   /* Real amplitudes A_k                        */
    float   modal_im[N_MODES];   /* Imaginary amplitudes B_k (quadrature)      */
    float   modal_omega[N_MODES];/* ω_k = k·π·c_s/L                            */
    int     modal_phase;         /* Used for rotating-frame update              */
    float*  modal_basis;         /* (N_MODES × n_nodes) precomputed sin shapes, row-major */

    /* Stiff-string biharm coefficient: (EI/μ)·dt_s²/ds⁴
     * 0 when stiffness_EI == 0 (ideal flexible string) */
    float   biharm_coef;

    /* Time-resolved external force.
     * Filled by coevolver_inject_string_force or coevolver_pluck_string.
     * timed_force_buf[s] is pre-scaled by dt_s²/(mu·ds); distributed evenly
     * across all string substeps within each audio sample by _step_one_sample. */
    float*  timed_force_buf;      /* [n_force_buf] heap, may be NULL             */
    int     n_force_buf;          /* number of valid samples                     */
    int     force_node;           /* injection node index (1..n_nodes-2)        */
    int     force_onset_sample;   /* absolute sample index when buf[0] applies  */
} StringState;

/* ── Internal pickup state ────────────────────────────────────────────────── */

typedef struct {
    int      type;
    uint32_t string_mask;
    float    sensitivity;        /* Piezo only                                  */
    float    axis[3];            /* Coil axis unit vector (magnetic pickups)    */
    /* B_kernel[string_idx][seg_idx] — precomputed magnetic weights            */
    int      n_strings;          /* Mirror of parent state for alloc           */
    int*     n_segs_per_string;  /* (n_strings,)                               */
    float**  Bn_kernel;  /* (n_strings,) → (n_segs,) — B·dot(n̂_seg, coil_axis) */
    float**  Bb_kernel;  /* (n_strings,) → (n_segs,) — B·dot(b̂_seg, coil_axis) */
} PickupState;

/* ── Internal mic state ───────────────────────────────────────────────────── */

typedef struct {
    float grid_xyz[3];   /* Preconverted to grid coordinates                   */
    float axis[3];       /* Polar axis in world space (unit vector)            */
    float polar_a;       /* Omnidirectional (pressure) weight                  */
    float polar_b;       /* Figure-8 (velocity) weight                        */
    float gain;
} MicState;

/* ── Async worker state ───────────────────────────────────────────────────── */

/* Heap-allocated separately so the main struct can still use calloc.
 * All fields read/written by both GUI thread (poll) and worker thread (write). */
struct ThreadState {
    std::thread          worker;
    std::atomic<int>     status          {CE_STATUS_IDLE};
    std::atomic<int>     progress_samples{0};
    std::atomic<int>     progress_total  {0};
    std::atomic<int>     cancel_flag     {0};
    std::atomic<int>     last_error      {CE_OK};
};

/* ── Master state ─────────────────────────────────────────────────────────── */

struct AcousticCoEvolverState {
    /* Subsystems */
    int           n_strings;
    StringState*  strings;       /* (n_strings,)                               */

    int           n_pickups;
    PickupState*  pickups;

    int           n_mics;
    MicState*     mics;

    /* Body acoustic FDTD — owned by this module */
    AcousticFDTDState* fdtd;
    int   Nx, Ny, Nz;
    float dx, origin[3];         /* World origin of grid corner (0,0,0) in metres */
    float rho_air;               /* Cached from body def for mic impedance       */
    float c_sound;               /* Cached speed of sound (m/s)                  */

    /* Body plate index for saddle BC lookup */
    int   plate_iz;              /* Z-index of soundboard layer                */

    /* Bridge injection sites — grid-space indices of FDTD cells near saddle  */
    /* (built in coevolver_create from bridge_src_xyz)                         */
    int*  bridge_cell_idx;       /* Flat FDTD indices                          */
    float* bridge_cell_wgt;
    int   n_bridge_cells;

    /* Shared neck/body structural coupling.  All string nut forces drive one
     * neck bending DOF; the equal/opposite joint reaction is distributed into
     * an upper-bout/neck-block plate kernel. */
    int*   neck_plate_idx;       /* Flat plate indices j + Ny*i                */
    float* neck_plate_wgt;       /* Normalized joint kernel weights            */
    int    n_neck_plate_cells;
    float  neck_w;
    float  neck_w_prev;
    float  neck_plate_w_prev;
    float  neck_mass_eff;
    float  neck_stiff_eff;
    float  neck_damp_eff;
    float  neck_body_stiff_eff;
    float  neck_body_damp_eff;

    /* Timing */
    float sample_rate;
    float dt_audio;              /* 1/sample_rate                              */
    float dt_fdtd;               /* From fdtd_get_dt()                         */
    int   substeps;              /* ceil(dt_audio / dt_fdtd), capped           */

    /* Deprecated compatibility field; kept at 1.0. */
    float force_scale;

    /* Global damping multiplier — applied to all string γ each step.
     * 1.0 = physical; >1.0 for warm-up overdamping. */
    float gamma_scale;

    /* Modal */
    int   modal_stride;
    int   modal_counter;

    /* Output ring buffers  [pickup_idx][sample]  and  [mic_idx][sample]      */
    float** pickup_out;          /* (n_pickups, OUTPUT_BUF)                    */
    float** mic_out;             /* (n_mics,    OUTPUT_BUF)                    */
    int     out_write_pos;       /* Write cursor into ring                     */
    int     samples_written;     /* Total samples produced                     */

    /* Deprecated legacy direct-pressure state. */
    float   bridge_vel_prev;

    /* Scratch: per-sample pickup accumulator */
    float*  pickup_accum;        /* (n_pickups,) float                         */

    /* Saddle batch sampler — precomputed plate bilinear indices/weights.
     * Avoids full-plate memcpy every FDTD substep; set up in coevolver_create. */
    int*    saddle_idx4;         /* [n_strings * 4] flat plate indices          */
    float*  saddle_wgt4;         /* [n_strings * 4] bilinear weights            */
    float*  saddle_w_batch;      /* [n_strings] per-substep gather output       */

    /* Mic batch sampling — contiguous grid_xyz array for a single batch call. */
    float*  mic_rec_xyz;         /* [n_mics * 3] packed grid coords             */
    float*  mic_P;               /* [n_mics] batch pressure output              */
    float*  mic_vx;              /* [n_mics] batch velocity output              */
    float*  mic_vy;
    float*  mic_vz;

    /* Precomputed 8-tap trilinear samplers for mic pressure and velocity.
     * Computed once at create-time; eliminates per-sample floor/clamp/fraction. */
    int*    mic_P_idx8;          /* [n_mics * 8] pressure corner flat indices    */
    float*  mic_P_wgt8;          /* [n_mics * 8] pressure trilinear weights      */
    int*    mic_Vx_idx8;         /* [n_mics * 8] staggered Vx corner indices     */
    float*  mic_Vx_wgt8;
    int*    mic_Vy_idx8;
    float*  mic_Vy_wgt8;
    int*    mic_Vz_idx8;
    float*  mic_Vz_wgt8;

    /* Per-string bridge injection kernel (Issue 11).
     * string_bridge_kernel[si * n_bridge_cells + c] = normalised Gaussian weight
     * from string si's saddle position to bridge FDTD cell c. */
    float*  string_bridge_kernel; /* [n_strings * n_bridge_cells]                */
    float*  bridge_drive;         /* [n_bridge_cells] per-substep scratch        */

    /* Pickup GEMV — packed kernel matrices for magnetic pickups.
     * K0[pi_row * total_seg_flat + global_seg] = Bn[pi][si][s] * ds[si]
     * K1 similarly for Bb.  Piezo pickups use the scalar path unchanged. */
    int     total_seg_flat;      /* sum of n_segs over all strings              */
    int*    string_seg_offsets;  /* [n_strings + 1] segment start offsets       */
    int     n_mag_pickups;       /* number of non-piezo pickups                 */
    int*    mag_pickup_map;      /* [n_mag_pickups] → global pickup index       */
    float*  pickup_K0;           /* [n_mag_pickups * total_seg_flat]            */
    float*  pickup_K1;           /* [n_mag_pickups * total_seg_flat]            */
    float*  string_v0_flat;      /* [total_seg_flat] packed pol-0 velocities    */
    float*  string_v1_flat;      /* [total_seg_flat] packed pol-1 velocities    */
    float*  mag_pickup_tmp;      /* [n_mag_pickups]  scratch for GEMV output    */

    /* Pluck event schedule — pre-registered before step_async, fired by worker */
    PluckEvent* pluck_events;   /* sorted by onset_sample                          */
    int         n_pluck_events;
    int         pluck_capacity;
    int         pluck_cursor;   /* index of next event to fire (reset each step)   */

    /* Current absolute sample index during coevolver_step — read by
     * coevolver_pluck_string to set force_onset_sample on the string. */
    int         current_sample_idx;

    /* Precomputed: 1 if any mic has polar_b != 0 (needs velocity sampling)     */
    int         any_mic_needs_velocity;

    /* AMR pressure backend — set when created via coevolver_create_amr.
     * Exactly one of {fdtd, pressure_backend} is non-NULL per instance.
     * All acoustic dispatch goes through pressure_backend when it is set. */
    IPressureBackend* pressure_backend;

    /* Async worker — heap-allocated separately (C++ construction required) */
    ThreadState* thread_state;
};

/* ── Allocation helpers ───────────────────────────────────────────────────── */

static float*   alloc_f(int n) { return (float*)  calloc(n, sizeof(float));   }
static int*     alloc_i(int n) { return (int*)    calloc(n, sizeof(int));     }
static uint8_t* alloc_u8(int n){ return (uint8_t*)calloc(n, sizeof(uint8_t));}

/* ── Async busy check ─────────────────────────────────────────────────────── */

static int coevolver_busy(const AcousticCoEvolverState* st)
{
    return st && st->thread_state &&
           st->thread_state->status.load(std::memory_order_acquire) == CE_STATUS_RUNNING;
}

/* ── Frame computation (Frenet-Serret, stable) ──────────────────────────── */

/* Build tangent / normal / binormal frame for each segment of a string path.
 * Uses a simple parallel-transport scheme: bootstrap normal from world Y
 * (or Z if tangent is parallel to Y), then parallel-transport. */
static void build_string_frame(StringState* ss)
{
    int N = ss->n_segs;
    /* Compute segment tangents (unnormalised) and accumulate length */
    ss->total_length = 0.0f;
    for (int s = 0; s < N; ++s) {
        float dx = ss->seg_xyz[(s+1)*3+0] - ss->seg_xyz[s*3+0];
        float dy = ss->seg_xyz[(s+1)*3+1] - ss->seg_xyz[s*3+1];
        float dz = ss->seg_xyz[(s+1)*3+2] - ss->seg_xyz[s*3+2];
        float len = sqrtf(dx*dx + dy*dy + dz*dz);
        if (len < 1e-12f) len = 1e-12f;
        ss->total_length += len;
        ss->seg_tangent[s*3+0] = dx / len;
        ss->seg_tangent[s*3+1] = dy / len;
        ss->seg_tangent[s*3+2] = dz / len;
    }
    ss->ds = ss->total_length / (float)N;

    /* Bootstrap: choose reference axis not parallel to first tangent */
    float tx = ss->seg_tangent[0], ty = ss->seg_tangent[1], tz = ss->seg_tangent[2];
    float ref[3];
    if (fabsf(ty) < 0.9f) { ref[0]=0; ref[1]=1; ref[2]=0; }   /* world Y */
    else                   { ref[0]=0; ref[1]=0; ref[2]=1; }   /* world Z */

    /* Normal = ref - (ref·t)t */
    float dot = ref[0]*tx + ref[1]*ty + ref[2]*tz;
    float nx = ref[0] - dot*tx;
    float ny = ref[1] - dot*ty;
    float nz = ref[2] - dot*tz;
    float nl = sqrtf(nx*nx+ny*ny+nz*nz); if (nl<1e-12f) nl=1e-12f;
    nx/=nl; ny/=nl; nz/=nl;

    for (int s = 0; s < N; ++s) {
        float* t = ss->seg_tangent  + s*3;
        float* n = ss->seg_normal   + s*3;
        float* b = ss->seg_binormal + s*3;

        /* Parallel-transport n from previous segment */
        if (s > 0) {
            float* tp = ss->seg_tangent + (s-1)*3;
            /* Rotate n around axis (tp × t) */
            float ax = tp[1]*t[2]-tp[2]*t[1];
            float ay = tp[2]*t[0]-tp[0]*t[2];
            float az = tp[0]*t[1]-tp[1]*t[0];
            float a_len = sqrtf(ax*ax+ay*ay+az*az);
            if (a_len > 1e-8f) {
                float theta = asinf(fminf(a_len, 1.0f));
                float c_ = cosf(theta), s_ = sinf(theta)/a_len;
                ax *= s_; ay *= s_; az *= s_;
                /* Rodrigues rotation: n' = c*n + sin(θ)*(axis×n) + (1-c)*(axis·n)*axis */
                float cross_x = ay*nz - az*ny;
                float cross_y = az*nx - ax*nz;
                float cross_z = ax*ny - ay*nx;
                float adn = ax*nx + ay*ny + az*nz;
                nx = c_*nx + sinf(theta)*cross_x + (1-c_)*adn*ax;
                ny = c_*ny + sinf(theta)*cross_y + (1-c_)*adn*ay;
                nz = c_*nz + sinf(theta)*cross_z + (1-c_)*adn*az;
                /* Re-orthogonalise n against t */
                dot = nx*t[0]+ny*t[1]+nz*t[2];
                nx -= dot*t[0]; ny -= dot*t[1]; nz -= dot*t[2];
                nl = sqrtf(nx*nx+ny*ny+nz*nz); if (nl<1e-12f) nl=1e-12f;
                nx/=nl; ny/=nl; nz/=nl;
            }
        }
        n[0]=nx; n[1]=ny; n[2]=nz;
        /* Binormal = t × n */
        b[0] = t[1]*n[2] - t[2]*n[1];
        b[1] = t[2]*n[0] - t[0]*n[2];
        b[2] = t[0]*n[1] - t[1]*n[0];
    }
}

/* ── String state construction ─────────────────────────────────────────────── */

static StringState* string_create(const CoEvolverStringDef* def, float dt_fdtd)
{
    int N = def->n_segs;
    if (N < 2 || !def->path_xyz || def->tension_N <= 0 || def->linear_mass_kgm <= 0)
        return NULL;

    StringState* ss = (StringState*)calloc(1, sizeof(StringState));
    if (!ss) return NULL;
    ss->n_segs  = N;
    ss->n_nodes = N + 1;

    ss->seg_xyz      = alloc_f((N+1)*3);
    ss->seg_tangent  = alloc_f(N*3);
    ss->seg_normal   = alloc_f(N*3);
    ss->seg_binormal = alloc_f(N*3);
    if (!ss->seg_xyz || !ss->seg_tangent || !ss->seg_normal || !ss->seg_binormal)
        goto fail;

    memcpy(ss->seg_xyz, def->path_xyz, (N+1)*3*sizeof(float));

    ss->tension_N    = def->tension_N;
    ss->tension_eff  = def->tension_N;
    ss->linear_mass  = def->linear_mass_kgm;
    ss->gamma        = def->damping;
    ss->gamma_eff    = def->damping;
    ss->axial_stiffness_N = def->axial_stiffness_N;
    if (ss->axial_stiffness_N <= 0.0f && ss->linear_mass > 0.0f) {
        const float steel_E_over_rho = 2.0e11f / 7850.0f;
        ss->axial_stiffness_N = steel_E_over_rho * ss->linear_mass;
    }

    /* Per-string bridge/top impedance.  The string endpoint is the bridge DOF;
     * the soundboard receives the equal spring/damper reaction from that DOF.
     * These conservative defaults put the bridge/top coupling in the low kHz
     * band with moderate loss, avoiding a rigid, energy-injecting saddle. */
    ss->bridge_mass_eff = 0.00035f;
    {
        const float bridge_freq_hz = 450.0f;
        const float bridge_Q       = 2.5f;
        float omega_b = 2.0f * (float)M_PI * bridge_freq_hz;
        ss->bridge_stiff_eff = ss->bridge_mass_eff * omega_b * omega_b;
        ss->bridge_damp_eff  = ss->bridge_mass_eff * omega_b / bridge_Q;
    }

    /* Neck bending model at nut — optional driven BC */
    if (def->neck_freq_hz > 0.0f && def->neck_mass_kg > 0.0f) {
        ss->has_neck_model = 1;
        ss->neck_mass_eff  = def->neck_mass_kg;
        float omega_n = 2.0f * (float)M_PI * def->neck_freq_hz;
        ss->neck_stiff_eff = def->neck_mass_kg * omega_n * omega_n;
        float Q = (def->neck_Q > 0.0f) ? def->neck_Q : 30.0f;
        ss->neck_damp_eff  = def->neck_mass_kg * omega_n / Q;
    }

    build_string_frame(ss);

    ss->wave_speed = sqrtf(def->tension_N / def->linear_mass_kgm);

    /* Per-string adaptive dt subdivider (stiffness-corrected CFL):
     *
     *   Stiff string wave equation:  u_tt = c_s²·u_xx − (EI/μ)·u_xxxx
     *   Effective wave speed at the Nyquist wavenumber k = π/ds:
     *     c_eff = sqrt( c_s² + (EI/μ)·(π/ds)² )
     *   CFL stability limit:  dt_max = 0.95·ds / c_eff
     *   n_substeps            = ceil(dt_fdtd / dt_max)
     *   dt_s                  = dt_fdtd / n_substeps
     *   biharm_coef           = (EI/μ)·dt_s²/ds⁴   (discrete biharmonic coefficient)
     */
    {
        float EI  = def->stiffness_EI;
        float mu  = def->linear_mass_kgm;
        float c_stiff_sq = (EI > 0.0f && mu > 0.0f)
            ? (EI / mu) * ((float)M_PI / ss->ds) * ((float)M_PI / ss->ds)
            : 0.0f;
        float c_eff = sqrtf(ss->wave_speed * ss->wave_speed + c_stiff_sq);
        float dt_max_string = 0.95f * ss->ds / c_eff;
        ss->n_substeps = (int)ceilf(dt_fdtd / dt_max_string);
        if (ss->n_substeps < 1) ss->n_substeps = 1;
        ss->dt_s = dt_fdtd / (float)ss->n_substeps;
        float cfl = ss->wave_speed * ss->dt_s / ss->ds;
        ss->cfl2  = cfl * cfl;
        ss->biharm_coef = (EI > 0.0f && mu > 0.0f)
            ? (EI / mu) * (ss->dt_s * ss->dt_s) / (ss->ds * ss->ds * ss->ds * ss->ds)
            : 0.0f;
    }

    for (int p = 0; p < 2; ++p) {
        ss->u[p]      = alloc_f(N+1);
        ss->u_prev[p] = alloc_f(N+1);
        ss->u_tmp[p]  = alloc_f(N+1);
        ss->ext_force[p] = alloc_f(N+1);
        if (!ss->u[p] || !ss->u_prev[p] || !ss->u_tmp[p] || !ss->ext_force[p])
            goto fail;
    }

    /* Modal angular frequencies ω_k = k·π·c_s/L */
    for (int k = 0; k < N_MODES; ++k)
        ss->modal_omega[k] = (float)(k+1) * (float)M_PI * ss->wave_speed / ss->total_length;

    /* Precompute sin basis: modal_basis[k*Nn + i] = sin((k+1)·π·i/(Nn-1))
     * Used for vectorised matrix-vector modal projection. */
    ss->modal_basis = alloc_f(N_MODES * ss->n_nodes);
    if (!ss->modal_basis) goto fail;
    {
        int   Nn  = ss->n_nodes;
        float inv = 1.0f / (float)(Nn > 1 ? Nn - 1 : 1);
        for (int k = 0; k < N_MODES; ++k) {
            float kpi = (float)(k + 1) * (float)M_PI;
            for (int i = 0; i < Nn; ++i)
                ss->modal_basis[k * Nn + i] = sinf(kpi * (float)i * inv);
        }
    }

    return ss;
fail:
    /* partial free — we can't call string_destroy before it's registered */
    for (int p=0; p<2; ++p) {
        free(ss->u[p]); free(ss->u_prev[p]); free(ss->u_tmp[p]); free(ss->ext_force[p]);
    }
    free(ss->seg_xyz); free(ss->seg_tangent); free(ss->seg_normal); free(ss->seg_binormal);
    free(ss->modal_basis);
    free(ss);
    return NULL;
}

static void string_destroy(StringState* ss)
{
    if (!ss) return;
    for (int p = 0; p < 2; ++p) {
        free(ss->u[p]); free(ss->u_prev[p]); free(ss->u_tmp[p]); free(ss->ext_force[p]);
    }
    free(ss->seg_xyz); free(ss->seg_tangent); free(ss->seg_normal); free(ss->seg_binormal);
    free(ss->modal_basis);
    free(ss->timed_force_buf);
    free(ss);
}

/* ── 1-D string FDTD step ─────────────────────────────────────────────────── */

static int string_validate_shape(const StringState* ss)
{
    if (!ss || ss->n_nodes < 2 || ss->ds <= 0.0f || ss->total_length <= 0.0f)
        return CE_ERR_UNSTABLE;

    float arc = 0.0f;
    for (int i = 0; i < ss->n_nodes - 1; ++i) {
        float du0 = ss->u[0][i + 1] - ss->u[0][i];
        float du1 = ss->u[1][i + 1] - ss->u[1][i];
        if (!isfinite(du0) || !isfinite(du1)) return CE_ERR_UNSTABLE;

        float slope = sqrtf(du0 * du0 + du1 * du1) / ss->ds;
        if (!isfinite(slope) || slope > STRING_MAX_SEGMENT_SLOPE)
            return CE_ERR_UNSTABLE;

        arc += sqrtf(ss->ds * ss->ds + du0 * du0 + du1 * du1);
    }

    float stretch_frac = (arc - ss->total_length) / ss->total_length;
    if (!isfinite(stretch_frac) || stretch_frac > STRING_MAX_STRETCH_FRAC)
        return CE_ERR_UNSTABLE;

    return CE_OK;
}

static int string_update_geometric_tension(StringState* ss)
{
    if (!ss || ss->n_nodes < 2 || ss->ds <= 0.0f || ss->total_length <= 0.0f)
        return CE_ERR_UNSTABLE;

    using CArr = Eigen::Map<const Eigen::ArrayXf>;
    const int Nseg = ss->n_nodes - 1;
    CArr u0(ss->u[0], ss->n_nodes);
    CArr u1(ss->u[1], ss->n_nodes);

    Eigen::ArrayXf du0 = u0.tail(Nseg) - u0.head(Nseg);
    Eigen::ArrayXf du1 = u1.tail(Nseg) - u1.head(Nseg);
    if (!du0.isFinite().all() || !du1.isFinite().all()) return CE_ERR_UNSTABLE;

    const float ds2 = ss->ds * ss->ds;
    const float arc = (ds2 + du0.square() + du1.square()).sqrt().sum();

    float strain = (arc - ss->total_length) / ss->total_length;
    if (!isfinite(strain)) return CE_ERR_UNSTABLE;
    if (strain < 0.0f) strain = 0.0f;

    ss->tension_eff = ss->tension_N + ss->axial_stiffness_N * strain;
    if (!isfinite(ss->tension_eff) || ss->tension_eff <= 0.0f)
        return CE_ERR_UNSTABLE;

    float cfl = sqrtf(ss->tension_eff / ss->linear_mass) * ss->dt_s / ss->ds;
    if (!isfinite(cfl) || cfl * cfl > 0.9025f)
        return CE_ERR_UNSTABLE;

    return CE_OK;
}

static int string_step_pol(StringState* ss, int pol, float nut_bc_w)
{
    int   N    = ss->n_nodes;
    float cfl  = sqrtf(ss->tension_eff / ss->linear_mass) * ss->dt_s / ss->ds;
    float c2   = cfl * cfl;

    /* Centered damped leapfrog (Crank–Nicolson velocity):
     *   (1 + γdt/2) u^{n+1} = 2u^n − (1 − γdt/2) u^{n-1} + c²Δt²∇²u + dt²F/μ
     * gamma_eff is set by _step_one_sample from gamma * gamma_scale each step. */
    float half_gdt = 0.5f * ss->gamma_eff * ss->dt_s;
    float inv_1pg  = 1.0f / (1.0f + half_gdt);

    using Arr = Eigen::Map<Eigen::ArrayXf>;

    Arr u (ss->u[pol],         N);
    Arr up(ss->u_prev[pol],    N);
    Arr ut(ss->u_tmp[pol],     N);
    Arr fe(ss->ext_force[pol], N);

    /* Boundary conditions */
    ut[N-1] = (pol == 0) ? ss->bridge_w : 0.0f;

    /* Nut BC: rigid (0) or driven by neck bending oscillator (pol=0 only).
     * The neck model is a 1-DOF spring-mass-damper driven by the string's
     * nut slope force: F = T·(u[1] − nut_w)/ds.  Centered leapfrog for neck:
     *   (m/dt² + c/2dt)·w_new = (2m/dt² − k)·w − (m/dt² − c/2dt)·w_prev + F */
    ut[0] = (pol == 0) ? nut_bc_w : 0.0f;

    /* Interior nodes [1 .. N-2]: fully vectorised 3-point wave stencil. */
    if (N > 2) {
        const int M = N - 2;
        auto dst = ut.segment(1, M);

        /* Numerator: 2u − (1−γdt/2)u_prev + c²·lap + fe */
        dst =
              2.0f * u.segment(1, M)
            - (1.0f - half_gdt) * up.segment(1, M)
            + c2 * (u.segment(2, M) - 2.0f * u.segment(1, M) + u.segment(0, M))
            + fe.segment(1, M);

        /* Biharmonic stiffness correction — nodes [2..N-3] (5-point stencil). */
        if (N > 4 && ss->biharm_coef > 0.0f) {
            const int M2 = N - 4;   /* nodes 2..N-3 */
            dst.segment(1, M2) -= ss->biharm_coef * (
                  u.segment(0, M2)
                - 4.0f * u.segment(1, M2)
                + 6.0f * u.segment(2, M2)
                - 4.0f * u.segment(3, M2)
                + u.segment(4, M2));
        }

        /* Divide numerator by (1 + γdt/2) to complete the centered update. */
        dst *= inv_1pg;

        if (!dst.isFinite().all()) return CE_ERR_UNSTABLE;
    }

    /* Pointer swap — Maps above are stack-local, buffers stay valid */
    float* tmp      = ss->u_prev[pol];
    ss->u_prev[pol] = ss->u[pol];
    ss->u[pol]      = ss->u_tmp[pol];
    ss->u_tmp[pol]  = tmp;

    /* ext_force[pol] is not part of the swap — zero it for next step */
    fe.setZero();
    return CE_OK;
}

/* ── Modal projection ─────────────────────────────────────────────────────── */

static void modal_project(StringState* ss)
{
    int   N     = ss->n_nodes;
    float scale = 2.0f / (float)(N > 1 ? N - 1 : 1);

    /* basis: (N_MODES × N) row-major matrix of precomputed sin shapes.
     * One gemv call replaces N_MODES separate dot-product loops and fully
     * uses Eigen's SIMD (+ OpenMP threading for large matrices). */
    using RowMat = Eigen::Map<const Eigen::Matrix<float,
                                Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using CVec   = Eigen::Map<const Eigen::VectorXf>;
    using Vec    = Eigen::Map<Eigen::VectorXf>;

    RowMat basis(ss->modal_basis, N_MODES, N);
    CVec   u_vec  (ss->u[0],      N);
    CVec   up_vec (ss->u_prev[0], N);
    Vec    re(ss->modal_re, N_MODES);
    Vec    im(ss->modal_im, N_MODES);

    re = (basis * u_vec)  * scale;
    im = (basis * up_vec) * scale;
}

/* ── Pickup B-kernel precomputation ──────────────────────────────────────── */

/* Distance from point p to line defined by point a with direction d_unit */
static float point_to_line_dist(const float* p, const float* a, const float* d_unit)
{
    float ax = p[0]-a[0], ay = p[1]-a[1], az = p[2]-a[2];
    float proj = ax*d_unit[0] + ay*d_unit[1] + az*d_unit[2];
    float rx = ax - proj*d_unit[0];
    float ry = ay - proj*d_unit[1];
    float rz = az - proj*d_unit[2];
    return sqrtf(rx*rx+ry*ry+rz*rz);
}

/* Build the Bn and Bb kernels for one magnetic pickup × one string.
 * Bn[s] = B(s) · dot(n̂_seg[s], coil_axis)
 * Bb[s] = B(s) · dot(b̂_seg[s], coil_axis)
 *
 * Precomputing the per-segment axis projection eliminates the AoS frame
 * reconstruction from the inner pickup loop.  The per-sample signal becomes:
 *   V = sum_s ( v0[s]·Bn[s] + v1[s]·Bb[s] ) · ds
 * which is a single pair of Eigen dot products — fully SIMD.
 *
 * Both output arrays are heap-allocated (caller must free). */
static int build_bn_bb_kernels(const CoEvolverPickupDef* def,
                                const StringState*        ss,
                                float**                   Bn_out,
                                float**                   Bb_out)
{
    int N = ss->n_segs;
    float* Bn = alloc_f(N);
    float* Bb = alloc_f(N);
    if (!Bn || !Bb) { free(Bn); free(Bb); return -1; }

    /* Normalise coil axis once */
    float ax = def->axis[0], ay = def->axis[1], az = def->axis[2];
    float alen = sqrtf(ax*ax + ay*ay + az*az);
    if (alen < 1e-9f) { ax = 0.0f; ay = 1.0f; az = 0.0f; }
    else { ax /= alen; ay /= alen; az /= alen; }

    float inv_2s2 = (def->pole_sigma > 0.0f)
                    ? 0.5f / (def->pole_sigma * def->pole_sigma)
                    : 1e12f;  /* degenerate sigma → effectively zero B radius */
    float spacing = def->coil_spacing;
    float axis_unit[3] = { ax, ay, az };   /* already normalised above */

    for (int s = 0; s < N; ++s) {
        float mx = 0.5f * (ss->seg_xyz[s*3+0] + ss->seg_xyz[(s+1)*3+0]);
        float my = 0.5f * (ss->seg_xyz[s*3+1] + ss->seg_xyz[(s+1)*3+1]);
        float mz = 0.5f * (ss->seg_xyz[s*3+2] + ss->seg_xyz[(s+1)*3+2]);
        float m[3] = {mx, my, mz};

        float Bval;
        if (def->type == PICKUP_SINGLE_COIL) {
            float r = point_to_line_dist(m, def->pos, axis_unit);
            Bval = expf(-r*r * inv_2s2);
        } else if (def->type == PICKUP_HUMBUCKER) {
            float pos1[3] = {
                def->pos[0] + 0.5f*spacing*ax,
                def->pos[1] + 0.5f*spacing*ay,
                def->pos[2] + 0.5f*spacing*az
            };
            float pos2[3] = {
                def->pos[0] - 0.5f*spacing*ax,
                def->pos[1] - 0.5f*spacing*ay,
                def->pos[2] - 0.5f*spacing*az
            };
            float r1 = point_to_line_dist(m, pos1, axis_unit);
            float r2 = point_to_line_dist(m, pos2, axis_unit);
            Bval = expf(-r1*r1 * inv_2s2) - expf(-r2*r2 * inv_2s2);
        } else {
            Bval = 0.0f;   /* piezo: no B-kernel */
        }

        /* Project onto local frame components along coil axis */
        const float* n = ss->seg_normal   + s*3;
        const float* b = ss->seg_binormal + s*3;
        Bn[s] = Bval * (n[0]*ax + n[1]*ay + n[2]*az);
        Bb[s] = Bval * (b[0]*ax + b[1]*ay + b[2]*az);
    }

    *Bn_out = Bn;
    *Bb_out = Bb;
    return 0;
}

/* ── World ↔ grid coordinate conversion ────────────────────────────────────── */

/* Convert world position to grid-space float coordinates.
 * Grid origin is at world position st->origin, spacing st->dx. */
static void world_to_grid(const AcousticCoEvolverState* st,
                           const float* w, float* g)
{
    g[0] = (w[0] - st->origin[0]) / st->dx;
    g[1] = (w[1] - st->origin[1]) / st->dx;
    g[2] = (w[2] - st->origin[2]) / st->dx;
}

/* Flat FDTD index from grid-integer coords (with bounds clamp). */
static int grid_idx(const AcousticCoEvolverState* st, int i, int j, int k)
{
    if (i < 0) i = 0;  if (i >= st->Nx) i = st->Nx-1;
    if (j < 0) j = 0;  if (j >= st->Ny) j = st->Ny-1;
    if (k < 0) k = 0;  if (k >= st->Nz) k = st->Nz-1;
    return k + st->Nz * (j + st->Ny * i);
}

/* ── Bridge kernel construction ──────────────────────────────────────────── */

/* Build the set of FDTD cell indices + Gaussian weights for bridge injection,
 * from n_src world-space saddle positions.  Cells within 3σ of any saddle
 * position at z = plate_iz-1 (just below soundboard) are included. */
static int build_bridge_kernel(AcousticCoEvolverState* st,
                                int n_src, const float* src_xyz,
                                float sigma)
{
    /* Collect candidate cells in a 2-D slab at z = plate_iz - 1 */
    int plate_z = st->plate_iz - 1;
    if (plate_z < 0) plate_z = 0;

    int radius_cells = (int)ceilf(3.0f * sigma / st->dx) + 1;
    int max_cells = n_src * (2*radius_cells+1) * (2*radius_cells+1);
    if (max_cells < 1) max_cells = 1;

    int*   idx = alloc_i(max_cells);
    float* wgt = alloc_f(max_cells);
    if (!idx || !wgt) { free(idx); free(wgt); return -1; }

    int count = 0;
    float inv_2s2 = 0.5f / (sigma * sigma);

    for (int s = 0; s < n_src; ++s) {
        /* Convert source world position to grid integer */
        float gx = (src_xyz[s*3+0] - st->origin[0]) / st->dx;
        float gy = (src_xyz[s*3+1] - st->origin[1]) / st->dx;
        int ci0 = (int)roundf(gx);
        int cj0 = (int)roundf(gy);

        for (int di = -radius_cells; di <= radius_cells; ++di) {
            for (int dj = -radius_cells; dj <= radius_cells; ++dj) {
                int ci = ci0 + di;
                int cj = cj0 + dj;
                if (ci < 0 || ci >= st->Nx || cj < 0 || cj >= st->Ny) continue;

                float wx = (float)ci * st->dx + st->origin[0] - src_xyz[s*3+0];
                float wy = (float)cj * st->dx + st->origin[1] - src_xyz[s*3+1];
                float r2 = wx*wx + wy*wy;
                float w  = expf(-r2 * inv_2s2);
                if (w < 1e-4f) continue;

                int flat = grid_idx(st, ci, cj, plate_z);
                /* Check for duplicate index */
                int dup = 0;
                for (int x = 0; x < count; ++x) {
                    if (idx[x] == flat) { wgt[x] += w; dup = 1; break; }
                }
                if (!dup && count < max_cells) {
                    idx[count] = flat;
                    wgt[count] = w;
                    ++count;
                }
            }
        }
    }

    /* Normalise weights */
    float wsum = 0.0f;
    for (int i = 0; i < count; ++i) wsum += wgt[i];
    if (wsum > 1e-12f)
        for (int i = 0; i < count; ++i) wgt[i] /= wsum;

    st->bridge_cell_idx = idx;
    st->bridge_cell_wgt = wgt;
    st->n_bridge_cells  = count;
    return CE_OK;
}

/* Neck/body joint kernel: distribute heel reaction over the upper-bout/top-block
 * region instead of pinning one plate cell.  The centre is inferred from the
 * active plate mask's highest-Y cells, which corresponds to the neck joint side
 * for the generated guitar body. */
static int build_neck_plate_kernel(AcousticCoEvolverState* st,
                                   const uint8_t* plate_active,
                                   float sigma)
{
    if (!st || !plate_active || st->Nx <= 0 || st->Ny <= 0) return CE_ERR_NULL;
    if (sigma <= 0.0f) sigma = 0.030f;

    float y_max = -FLT_MAX;
    for (int i = 0; i < st->Nx; ++i) {
        for (int j = 0; j < st->Ny; ++j) {
            int p = j + st->Ny * i;
            if (!plate_active[p]) continue;
            float y = (float)j * st->dx + st->origin[1];
            if (y > y_max) y_max = y;
        }
    }
    if (y_max == -FLT_MAX) return CE_ERR_PARAM;

    float cx = 0.0f;
    float cy = y_max - 0.012f;
    int radius_cells = (int)ceilf(3.0f * sigma / st->dx) + 1;
    int ci = (int)floorf((cx - st->origin[0]) / st->dx);
    int cj = (int)floorf((cy - st->origin[1]) / st->dx);

    int cap = (2 * radius_cells + 1) * (2 * radius_cells + 1);
    int* idx = alloc_i(cap);
    float* wgt = alloc_f(cap);
    if (!idx || !wgt) {
        free(idx);
        free(wgt);
        return CE_ERR_NULL;
    }

    float inv_2s2 = 0.5f / (sigma * sigma);
    int count = 0;
    float wsum = 0.0f;
    for (int di = -radius_cells; di <= radius_cells; ++di) {
        int i = ci + di;
        if (i < 0 || i >= st->Nx) continue;
        for (int dj = -radius_cells; dj <= radius_cells; ++dj) {
            int j = cj + dj;
            if (j < 0 || j >= st->Ny) continue;
            int p = j + st->Ny * i;
            if (!plate_active[p]) continue;
            float x = (float)i * st->dx + st->origin[0];
            float y = (float)j * st->dx + st->origin[1];
            float dx = x - cx;
            float dy = y - cy;
            float ww = expf(-(dx*dx + dy*dy) * inv_2s2);
            if (ww <= 1e-12f) continue;
            idx[count] = p;
            wgt[count] = ww;
            wsum += ww;
            ++count;
        }
    }
    if (count <= 0 || wsum <= 1e-12f) {
        free(idx);
        free(wgt);
        return CE_ERR_PARAM;
    }
    for (int n = 0; n < count; ++n) wgt[n] /= wsum;

    st->neck_plate_idx = idx;
    st->neck_plate_wgt = wgt;
    st->n_neck_plate_cells = count;
    return CE_OK;
}


/* ── Guitar plate aperture mask ────────────────────────────────────────────── */

/* The Python visualiser and scene builder use the canonical acoustic-guitar
 * soundhole centered at (0, 0.050)m with radius 0.028m.  Some bridge paths
 * deliver a full-outline plate mask to C and only cut the hole for rendering.
 * Carve it here before fdtd_create so the pressure solver sees an aperture,
 * not an invisible display-only disc.  If the body grid does not contain that
 * circle, the copy is unchanged. */
static uint8_t* copy_plate_active_with_canonical_soundhole(const CoEvolverBodyDef* body)
{
    if (!body || !body->plate_active || body->Nx <= 0 || body->Ny <= 0 || body->dx <= 0.0f)
        return NULL;

    const int Np = body->Nx * body->Ny;
    uint8_t* out = alloc_u8(Np);
    if (!out) return NULL;
    memcpy(out, body->plate_active, (size_t)Np * sizeof(uint8_t));

    const float cx = 0.0f;
    const float cy = 0.050f;
    const float r  = 0.028f;
    const float r2 = r * r;
    const float x0 = body->origin[0];
    const float y0 = body->origin[1];
    const float x1 = x0 + (float)(body->Nx - 1) * body->dx;
    const float y1 = y0 + (float)(body->Ny - 1) * body->dx;

    if (cx + r < x0 || cx - r > x1 || cy + r < y0 || cy - r > y1)
        return out;

    int carved = 0;
    for (int i = 0; i < body->Nx; ++i) {
        float x = body->origin[0] + (float)i * body->dx;
        for (int j = 0; j < body->Ny; ++j) {
            float y = body->origin[1] + (float)j * body->dx;
            float dx = x - cx;
            float dy = y - cy;
            if (dx*dx + dy*dy < r2) {
                int pidx = j + body->Ny * i;
                if (out[pidx]) { out[pidx] = 0; ++carved; }
            }
        }
    }
    (void)carved;
    return out;
}

/* ── coevolver_create ─────────────────────────────────────────────────────── */

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
    float                     force_scale)
{
    if (!string_defs || !pickup_defs || !mic_defs || !body)
        return NULL;
    if (n_strings < 0 || n_pickups < 0 || n_mics < 0) return NULL;
    if (sample_rate <= 0 || modal_stride <= 0) return NULL;

    AcousticCoEvolverState* st =
        (AcousticCoEvolverState*)calloc(1, sizeof(AcousticCoEvolverState));
    if (!st) return NULL;

    /* ThreadState has C++ members; allocate and construct separately */
    st->thread_state = new (std::nothrow) ThreadState{};
    if (!st->thread_state) { free(st); return NULL; }

    /* ── Build body FDTD first (gives us dt_fdtd) ── */
    uint8_t* plate_active_for_fdtd =
        copy_plate_active_with_canonical_soundhole(body);
    st->fdtd = fdtd_create(
        body->Nx, body->Ny, body->Nz,
        body->dx, body->c, body->rho_air,
        body->cell_type,
        body->plate_iz,
        plate_active_for_fdtd ? plate_active_for_fdtd : body->plate_active,
        body->plate_mass_density,
        body->plate_stiffness_D,
        body->plate_alpha_M,
        body->plate_beta_K,
        body->n_pml
    );
    free(plate_active_for_fdtd);
    if (!st->fdtd) goto fail;

    st->Nx       = body->Nx;
    st->Ny       = body->Ny;
    st->Nz       = body->Nz;
    st->dx       = body->dx;
    st->rho_air  = body->rho_air;
    st->c_sound  = body->c;
    st->plate_iz = body->plate_iz;
    st->origin[0] = body->origin[0];
    st->origin[1] = body->origin[1];
    st->origin[2] = body->origin[2];

    st->sample_rate  = sample_rate;
    st->dt_audio     = 1.0f / sample_rate;
    st->dt_fdtd      = fdtd_get_dt(st->fdtd);
    (void)force_scale;
    st->force_scale  = 1.0f;
    st->gamma_scale  = 1.0f;
    st->modal_stride = modal_stride;

    /* How many FDTD sub-steps per audio sample */
    float ratio = st->dt_audio / st->dt_fdtd;
    st->substeps = (int)ceilf(ratio);
    if (st->substeps < 1)  st->substeps = 1;
    if (st->substeps > MAX_SUBSTEPS) st->substeps = MAX_SUBSTEPS;

    /* ── Build bridge kernel ── */
    if (body->n_bridge_src > 0 && body->bridge_src_xyz) {
        int rc = build_bridge_kernel(st, body->n_bridge_src,
                                     body->bridge_src_xyz, body->bridge_sigma);
        if (rc != CE_OK) goto fail;
    }

    /* Register bridge sources with FDTD */
    if (st->n_bridge_cells > 0) {
        fdtd_set_bridge_sources(st->fdtd,
            st->n_bridge_cells, st->bridge_cell_idx, st->bridge_cell_wgt);
    }

    /* ── Build strings ── */
    st->n_strings = n_strings;
    st->strings   = (StringState*)calloc(n_strings, sizeof(StringState));
    if (n_strings > 0 && !st->strings) goto fail;

    for (int si = 0; si < n_strings; ++si) {
        StringState* ss = string_create(&string_defs[si], st->dt_fdtd);
        if (!ss) goto fail;
        st->strings[si] = *ss;
        free(ss);   /* struct copied; interior pointers now owned by array entry */
    }

    /* Shared neck mode.  Use the first provided neck descriptor as the physical
     * neck, not one independent oscillator per string.  All strings feed this
     * one DOF through their nut slopes. */
    for (int si = 0; si < n_strings; ++si) {
        if (string_defs[si].neck_freq_hz > 0.0f &&
            string_defs[si].neck_mass_kg > 0.0f) {
            st->neck_mass_eff = string_defs[si].neck_mass_kg;
            float omega_n = 2.0f * (float)M_PI * string_defs[si].neck_freq_hz;
            st->neck_stiff_eff = st->neck_mass_eff * omega_n * omega_n;
            float Q = (string_defs[si].neck_Q > 0.0f) ? string_defs[si].neck_Q : 30.0f;
            st->neck_damp_eff = st->neck_mass_eff * omega_n / Q;

            /* Neck heel/body joint impedance: intentionally stiffer and more
             * damped than the free neck bend so energy returns through the
             * body with realistic loss instead of becoming a rigid clamp. */
            const float joint_freq_hz = 95.0f;
            const float joint_Q = 2.0f;
            float omega_j = 2.0f * (float)M_PI * joint_freq_hz;
            st->neck_body_stiff_eff = st->neck_mass_eff * omega_j * omega_j;
            st->neck_body_damp_eff = st->neck_mass_eff * omega_j / joint_Q;
            break;
        }
    }
    if (st->neck_mass_eff > 0.0f) {
        int rc_neck = build_neck_plate_kernel(st, body->plate_active, 0.030f);
        if (rc_neck != CE_OK) goto fail;
    }

    /* ── Build pickups ── */
    st->n_pickups = n_pickups;
    st->pickups   = (PickupState*)calloc(n_pickups > 0 ? n_pickups : 1,
                                          sizeof(PickupState));
    if (!st->pickups) goto fail;

    for (int pi = 0; pi < n_pickups; ++pi) {
        PickupState* pk = &st->pickups[pi];
        pk->type        = pickup_defs[pi].type;
        pk->string_mask = pickup_defs[pi].string_mask;
        pk->sensitivity = pickup_defs[pi].sensitivity;
        pk->axis[0]     = pickup_defs[pi].axis[0];
        pk->axis[1]     = pickup_defs[pi].axis[1];
        pk->axis[2]     = pickup_defs[pi].axis[2];
        pk->n_strings   = n_strings;

        pk->n_segs_per_string = alloc_i(n_strings);
        pk->Bn_kernel = (float**)calloc(n_strings > 0 ? n_strings : 1, sizeof(float*));
        pk->Bb_kernel = (float**)calloc(n_strings > 0 ? n_strings : 1, sizeof(float*));
        if (!pk->n_segs_per_string || !pk->Bn_kernel || !pk->Bb_kernel) goto fail;

        for (int si = 0; si < n_strings; ++si) {
            pk->n_segs_per_string[si] = st->strings[si].n_segs;
            if (pickup_defs[pi].type != PICKUP_PIEZO) {
                if (build_bn_bb_kernels(&pickup_defs[pi], &st->strings[si],
                                        &pk->Bn_kernel[si], &pk->Bb_kernel[si]) < 0)
                    goto fail;
            }
        }
    }

    /* ── Build mics ── */
    st->n_mics = n_mics;
    st->mics   = (MicState*)calloc(n_mics > 0 ? n_mics : 1, sizeof(MicState));
    if (!st->mics) goto fail;

    for (int mi = 0; mi < n_mics; ++mi) {
        world_to_grid(st, mic_defs[mi].pos, st->mics[mi].grid_xyz);
        st->mics[mi].gain    = mic_defs[mi].gain;
        st->mics[mi].polar_a = mic_defs[mi].polar_a;
        st->mics[mi].polar_b = mic_defs[mi].polar_b;
        /* Store axis in world space — used to sample velocity direction */
        st->mics[mi].axis[0] = mic_defs[mi].axis[0];
        st->mics[mi].axis[1] = mic_defs[mi].axis[1];
        st->mics[mi].axis[2] = mic_defs[mi].axis[2];
    }

    /* ── Output ring buffers ── */
    st->pickup_out = (float**)calloc(n_pickups > 0 ? n_pickups : 1, sizeof(float*));
    st->mic_out    = (float**)calloc(n_mics    > 0 ? n_mics    : 1, sizeof(float*));
    if (!st->pickup_out || !st->mic_out) goto fail;

    for (int pi = 0; pi < n_pickups; ++pi) {
        st->pickup_out[pi] = alloc_f(OUTPUT_BUF);
        if (!st->pickup_out[pi]) goto fail;
    }
    for (int mi = 0; mi < n_mics; ++mi) {
        st->mic_out[mi] = alloc_f(OUTPUT_BUF);
        if (!st->mic_out[mi]) goto fail;
    }

    st->pickup_accum = alloc_f(n_pickups > 0 ? n_pickups : 1);
    if (!st->pickup_accum) goto fail;

    /* ── Precompute saddle bilinear indices & weights ── *
     *                                                     *
     * Saddle position for string si is the last node:     *
     *   seg_xyz[(n_nodes-1)*3 + 0/1] gives world X/Y.    *
     * Convert to grid float, clamp, floor → i0/j0, etc.  *
     * Store four flat plate indices and four weights.     *
     * Used per-substep to replace the full plate memcpy.  */
    if (n_strings > 0) {
        st->saddle_idx4    = alloc_i(n_strings * 4);
        st->saddle_wgt4    = alloc_f(n_strings * 4);
        st->saddle_w_batch = alloc_f(n_strings);
        if (!st->saddle_idx4 || !st->saddle_wgt4 || !st->saddle_w_batch) goto fail;

        int Nx_ = st->Nx, Ny_ = st->Ny;
        for (int si = 0; si < n_strings; ++si) {
            const StringState* ss = &st->strings[si];
            int Nn = ss->n_nodes;
            float sx = ss->seg_xyz[(Nn-1)*3+0];
            float sy = ss->seg_xyz[(Nn-1)*3+1];
            float gi = (sx - st->origin[0]) / st->dx;
            float gj = (sy - st->origin[1]) / st->dx;
            if (gi < 0.0f) gi = 0.0f;  if (gi > (float)(Nx_-1)) gi = (float)(Nx_-1);
            if (gj < 0.0f) gj = 0.0f;  if (gj > (float)(Ny_-1)) gj = (float)(Ny_-1);
            int i0 = (int)gi, i1 = (i0 < Nx_-1) ? i0+1 : i0;
            int j0 = (int)gj, j1 = (j0 < Ny_-1) ? j0+1 : j0;
            float fi = gi - (float)i0;
            float fj = gj - (float)j0;
            /* Plate flat index: j + Ny*i */
            st->saddle_idx4[si*4+0] = j0 + Ny_*i0;
            st->saddle_idx4[si*4+1] = j0 + Ny_*i1;
            st->saddle_idx4[si*4+2] = j1 + Ny_*i0;
            st->saddle_idx4[si*4+3] = j1 + Ny_*i1;
            st->saddle_wgt4[si*4+0] = (1.0f-fi)*(1.0f-fj);
            st->saddle_wgt4[si*4+1] = fi        *(1.0f-fj);
            st->saddle_wgt4[si*4+2] = (1.0f-fi)*fj;
            st->saddle_wgt4[si*4+3] = fi        *fj;
        }
    }

    /* ── Precompute mic batch arrays ── *
     *                                    *
     * Pack all mic grid_xyz into one     *
     * contiguous array for a single      *
     * fdtd_sample_pressure batch call.   */
    /* Precompute whether any mic needs velocity sampling.
     * Must be set BEFORE the mic sampler allocation block below, which
     * gates velocity-array allocation on this flag. */
    st->any_mic_needs_velocity = 0;
    for (int mi = 0; mi < n_mics; ++mi) {
        if (mic_defs[mi].polar_b != 0.0f) {
            st->any_mic_needs_velocity = 1;
            break;
        }
    }

    if (n_mics > 0) {
        st->mic_rec_xyz = alloc_f(n_mics * 3);
        st->mic_P       = alloc_f(n_mics);
        st->mic_vx      = alloc_f(n_mics);
        st->mic_vy      = alloc_f(n_mics);
        st->mic_vz      = alloc_f(n_mics);
        if (!st->mic_rec_xyz || !st->mic_P ||
            !st->mic_vx || !st->mic_vy || !st->mic_vz) goto fail;
        for (int mi = 0; mi < n_mics; ++mi) {
            st->mic_rec_xyz[mi*3+0] = st->mics[mi].grid_xyz[0];
            st->mic_rec_xyz[mi*3+1] = st->mics[mi].grid_xyz[1];
            st->mic_rec_xyz[mi*3+2] = st->mics[mi].grid_xyz[2];
        }

        /* Precompute 8-tap pressure samplers (Issue 7) */
        st->mic_P_idx8 = alloc_i(n_mics * 8);
        st->mic_P_wgt8 = alloc_f(n_mics * 8);
        if (!st->mic_P_idx8 || !st->mic_P_wgt8) goto fail;
        if (fdtd_precompute_pressure_samplers(
                st->fdtd, n_mics, st->mic_rec_xyz,
                st->mic_P_idx8, st->mic_P_wgt8) != FDTD_OK) goto fail;

        /* Precompute 8-tap velocity samplers if any mic needs velocity */
        if (st->any_mic_needs_velocity) {
            st->mic_Vx_idx8 = alloc_i(n_mics * 8);
            st->mic_Vx_wgt8 = alloc_f(n_mics * 8);
            st->mic_Vy_idx8 = alloc_i(n_mics * 8);
            st->mic_Vy_wgt8 = alloc_f(n_mics * 8);
            st->mic_Vz_idx8 = alloc_i(n_mics * 8);
            st->mic_Vz_wgt8 = alloc_f(n_mics * 8);
            if (!st->mic_Vx_idx8 || !st->mic_Vx_wgt8 ||
                !st->mic_Vy_idx8 || !st->mic_Vy_wgt8 ||
                !st->mic_Vz_idx8 || !st->mic_Vz_wgt8) goto fail;
            if (fdtd_precompute_velocity_samplers(
                    st->fdtd, n_mics, st->mic_rec_xyz,
                    st->mic_Vx_idx8, st->mic_Vx_wgt8,
                    st->mic_Vy_idx8, st->mic_Vy_wgt8,
                    st->mic_Vz_idx8, st->mic_Vz_wgt8) != FDTD_OK) goto fail;
        }
    }

    /* ── Precompute pickup GEMV matrices (magnetic pickups only) ── *
     *                                                                *
     * K0[mag_pi_row, global_seg] = Bn[pi][si][s] * ds[si]           *
     * K1[mag_pi_row, global_seg] = Bb[pi][si][s] * ds[si]           *
     *                                                                *
     * Per sample: pack v0/v1 velocities flat, then one GEMV per call.*
     * Piezo pickups are unaffected and use the scalar path below.   */
    {
        /* Segment offset table */
        st->string_seg_offsets = alloc_i(n_strings + 1);
        if (!st->string_seg_offsets) goto fail;
        st->string_seg_offsets[0] = 0;
        for (int si = 0; si < n_strings; ++si)
            st->string_seg_offsets[si+1] =
                st->string_seg_offsets[si] + st->strings[si].n_segs;
        st->total_seg_flat = (n_strings > 0) ? st->string_seg_offsets[n_strings] : 0;

        /* Count magnetic pickups */
        int n_mag = 0;
        for (int pi = 0; pi < n_pickups; ++pi)
            if (pickup_defs[pi].type != PICKUP_PIEZO) ++n_mag;
        st->n_mag_pickups = n_mag;

        if (n_mag > 0 && st->total_seg_flat > 0) {
            st->mag_pickup_map = alloc_i(n_mag);
            st->pickup_K0      = alloc_f(n_mag * st->total_seg_flat);
            st->pickup_K1      = alloc_f(n_mag * st->total_seg_flat);
            st->string_v0_flat = alloc_f(st->total_seg_flat);
            st->string_v1_flat = alloc_f(st->total_seg_flat);
            if (!st->mag_pickup_map || !st->pickup_K0 || !st->pickup_K1 ||
                !st->string_v0_flat || !st->string_v1_flat) goto fail;

            /* calloc zeroed K0/K1 — only fill nonzero (masked) entries. */
            int mi2 = 0;
            for (int pi = 0; pi < n_pickups; ++pi) {
                if (pickup_defs[pi].type == PICKUP_PIEZO) continue;
                st->mag_pickup_map[mi2] = pi;
                float* K0_row = st->pickup_K0 + mi2 * st->total_seg_flat;
                float* K1_row = st->pickup_K1 + mi2 * st->total_seg_flat;
                const PickupState* pk = &st->pickups[pi];
                for (int si = 0; si < n_strings; ++si) {
                    if (!(pk->string_mask & (1u << (unsigned)si))) continue;
                    if (!pk->Bn_kernel[si] || !pk->Bb_kernel[si]) continue;
                    int   off = st->string_seg_offsets[si];
                    float ds  = st->strings[si].ds;
                    int   N   = st->strings[si].n_segs;
                    for (int s = 0; s < N; ++s) {
                        K0_row[off + s] = pk->Bn_kernel[si][s] * ds;
                        K1_row[off + s] = pk->Bb_kernel[si][s] * ds;
                    }
                }
                ++mi2;
            }
        }
    }

    /* Allocate GEMV scratch buffer for magnetic pickup batch multiply. */
    if (st->n_mag_pickups > 0) {
        st->mag_pickup_tmp = alloc_f(st->n_mag_pickups);
        if (!st->mag_pickup_tmp) goto fail;
    }

    /* ── Per-string bridge injection kernels (Issue 11) ── *
     *                                                        *
     * For each string si and each bridge cell c, compute a  *
     * Gaussian weight based on the 2-D distance between the *
     * string's saddle XY position and the cell's XY centre. *
     * Rows are normalised so each string drives the same     *
     * total acoustic power regardless of saddle position.   */
    if (n_strings > 0 && st->n_bridge_cells > 0) {
        int nb = st->n_bridge_cells;
        st->string_bridge_kernel = alloc_f(n_strings * nb);
        st->bridge_drive         = alloc_f(nb);
        if (!st->string_bridge_kernel || !st->bridge_drive) goto fail;

        float sigma    = (body->bridge_sigma > 0.0f) ? body->bridge_sigma : st->dx;
        float inv_2s2  = 0.5f / (sigma * sigma);

        for (int si = 0; si < n_strings; ++si) {
            const StringState* ss = &st->strings[si];
            int Nn = ss->n_nodes;
            /* Saddle world XY = last node */
            float sx = ss->seg_xyz[(Nn-1)*3+0];
            float sy = ss->seg_xyz[(Nn-1)*3+1];

            float* row = st->string_bridge_kernel + si * nb;
            float  wsum = 0.0f;
            for (int c = 0; c < nb; ++c) {
                /* Recover XY world position of bridge cell from flat index */
                int flat = st->bridge_cell_idx[c];
                int k    = flat % st->Nz;
                int tmp  = flat / st->Nz;
                int j    = tmp  % st->Ny;
                int i    = tmp  / st->Ny;
                (void)k;   /* Z coordinate not used for 2-D distance */
                float cx = (float)i * st->dx + st->origin[0];
                float cy = (float)j * st->dx + st->origin[1];
                float dx2 = sx - cx, dy2 = sy - cy;
                float r2 = dx2*dx2 + dy2*dy2;
                row[c] = expf(-r2 * inv_2s2);
                wsum += row[c];
            }
            /* Normalise so row sums to 1 */
            if (wsum > 1e-12f)
                for (int c = 0; c < nb; ++c) row[c] /= wsum;
        }
    }

    return st;

fail:
    coevolver_destroy(st);
    return NULL;
}

/* ── coevolver_create_amr ────────────────────────────────────────────────── */

#define AMR_FAIL(stage) do { coevolver_amr_set_error(stage); goto fail; } while(0)

SK_API AcousticCoEvolverState* coevolver_create_amr(
    int                           n_strings,
    const CoEvolverStringDef*     string_defs,
    int                           n_pickups,
    const CoEvolverPickupDef*     pickup_defs,
    int                           n_mics,
    const CoEvolverMicDef*        mic_defs,
    const AMRCoevolverDescriptor* amr_desc,
    float                         sample_rate,
    int                           modal_stride)
{
    coevolver_amr_set_error("");
    if (n_strings < 0 || n_pickups < 0 || n_mics < 0) {
        coevolver_amr_set_error("invalid_counts");
        return NULL;
    }
    if ((n_strings > 0 && !string_defs) ||
        (n_pickups > 0 && !pickup_defs) ||
        (n_mics    > 0 && !mic_defs) ||
        !amr_desc) {
        coevolver_amr_set_error("invalid_null_args");
        return NULL;
    }
    if (sample_rate <= 0 || modal_stride <= 0) {
        coevolver_amr_set_error("invalid_timing_args");
        return NULL;
    }

    AcousticCoEvolverState* st =
        (AcousticCoEvolverState*)calloc(1, sizeof(AcousticCoEvolverState));
    if (!st) return NULL;

    st->thread_state = new (std::nothrow) ThreadState{};
    if (!st->thread_state) { free(st); return NULL; }

    /* ── Create AMR pressure backend (fails loudly on invalid descriptor) ── */
    st->pressure_backend = amr_pressure_backend_create(amr_desc);
    if (!st->pressure_backend) AMR_FAIL("pressure_backend_create");

    /* Mirror plate grid dims into the coevolver's grid fields so that the
     * saddle bilinear indexing and bridge kernel use the plate coordinate
     * system.  st->Nz=1 signals "no uniform 3-D grid". */
    st->Nx       = amr_desc->plate_Nx;
    st->Ny       = amr_desc->plate_Ny;
    st->Nz       = 1;
    st->dx       = amr_desc->plate_dx;
    st->origin[0] = amr_desc->plate_origin[0];
    st->origin[1] = amr_desc->plate_origin[1];
    st->origin[2] = amr_desc->plate_origin[2];
    st->rho_air  = (float)amr_desc->rho_air;
    st->c_sound  = (float)amr_desc->c;

    st->sample_rate  = sample_rate;
    st->dt_audio     = 1.0f / sample_rate;
    st->dt_fdtd      = (float)st->pressure_backend->get_dt();
    if (st->dt_fdtd <= 0.0f) AMR_FAIL("dt_fdtd_invalid");
    st->force_scale  = 1.0f;
    st->gamma_scale  = 1.0f;
    st->modal_stride = modal_stride;

    {
        float ratio   = st->dt_audio / st->dt_fdtd;
        st->substeps  = (int)ceilf(ratio);
        if (st->substeps < 1)           st->substeps = 1;
        if (st->substeps > MAX_SUBSTEPS) st->substeps = MAX_SUBSTEPS;
    }

    /* ── Bridge plate sources ── *
     *                             *
     * Copy bridge plate indices    *
     * and weights for the bridge   *
     * kernel builder below.        */
    if (amr_desc->n_bridge_plate > 0) {
        int nb = amr_desc->n_bridge_plate;
        st->bridge_cell_idx = alloc_i(nb);
        st->bridge_cell_wgt = alloc_f(nb);
        if (!st->bridge_cell_idx || !st->bridge_cell_wgt) AMR_FAIL("bridge_cells_alloc");
        memcpy(st->bridge_cell_idx, amr_desc->bridge_plate_idx, nb * sizeof(int));
        memcpy(st->bridge_cell_wgt, amr_desc->bridge_plate_wgt, nb * sizeof(float));
        st->n_bridge_cells = nb;
    }

    /* ── Neck plate sources (optional) ── */
    if (amr_desc->n_neck_plate > 0 && amr_desc->neck_plate_idx && amr_desc->neck_plate_wgt) {
        int nn = amr_desc->n_neck_plate;
        st->neck_plate_idx = alloc_i(nn);
        st->neck_plate_wgt = alloc_f(nn);
        if (!st->neck_plate_idx || !st->neck_plate_wgt) AMR_FAIL("neck_cells_alloc");
        memcpy(st->neck_plate_idx, amr_desc->neck_plate_idx, nn * sizeof(int));
        memcpy(st->neck_plate_wgt, amr_desc->neck_plate_wgt, nn * sizeof(float));
        st->n_neck_plate_cells = nn;
    }

    /* ── Strings ── */
    st->n_strings = n_strings;
    st->strings   = (StringState*)calloc(n_strings > 0 ? n_strings : 1,
                                          sizeof(StringState));
    if (!st->strings) AMR_FAIL("strings_alloc");

    for (int si = 0; si < n_strings; ++si) {
        StringState* ss = string_create(&string_defs[si], st->dt_fdtd);
        if (!ss) AMR_FAIL("string_create");
        st->strings[si] = *ss;
        free(ss);
    }

    /* Neck mass/stiffness from first string def that has them */
    for (int si = 0; si < n_strings; ++si) {
        if (string_defs[si].neck_freq_hz > 0.0f &&
            string_defs[si].neck_mass_kg > 0.0f) {
            st->neck_mass_eff = string_defs[si].neck_mass_kg;
            float omega_n = 2.0f * (float)M_PI * string_defs[si].neck_freq_hz;
            st->neck_stiff_eff = st->neck_mass_eff * omega_n * omega_n;
            float Q = (string_defs[si].neck_Q > 0.0f) ? string_defs[si].neck_Q : 30.0f;
            st->neck_damp_eff = st->neck_mass_eff * omega_n / Q;
            const float joint_freq_hz = 95.0f;
            const float joint_Q = 2.0f;
            float omega_j = 2.0f * (float)M_PI * joint_freq_hz;
            st->neck_body_stiff_eff = st->neck_mass_eff * omega_j * omega_j;
            st->neck_body_damp_eff  = st->neck_mass_eff * omega_j / joint_Q;
            break;
        }
    }
    /* If no neck nodes provided, disable neck coupling for AMR */
    if (st->neck_mass_eff > 0.0f && st->n_neck_plate_cells == 0)
        st->neck_mass_eff = 0.0f;

    /* ── Pickups ── */
    st->n_pickups = n_pickups;
    st->pickups   = (PickupState*)calloc(n_pickups > 0 ? n_pickups : 1,
                                          sizeof(PickupState));
    if (!st->pickups) AMR_FAIL("pickups_alloc");

    for (int pi = 0; pi < n_pickups; ++pi) {
        PickupState* pk = &st->pickups[pi];
        pk->type        = pickup_defs[pi].type;
        pk->string_mask = pickup_defs[pi].string_mask;
        pk->sensitivity = pickup_defs[pi].sensitivity;
        pk->axis[0]     = pickup_defs[pi].axis[0];
        pk->axis[1]     = pickup_defs[pi].axis[1];
        pk->axis[2]     = pickup_defs[pi].axis[2];
        pk->n_strings   = n_strings;

        pk->n_segs_per_string = alloc_i(n_strings > 0 ? n_strings : 1);
        pk->Bn_kernel = (float**)calloc(n_strings > 0 ? n_strings : 1, sizeof(float*));
        pk->Bb_kernel = (float**)calloc(n_strings > 0 ? n_strings : 1, sizeof(float*));
        if (!pk->n_segs_per_string || !pk->Bn_kernel || !pk->Bb_kernel) AMR_FAIL("pickup_arrays_alloc");

        for (int si = 0; si < n_strings; ++si) {
            pk->n_segs_per_string[si] = st->strings[si].n_segs;
            if (pickup_defs[pi].type != PICKUP_PIEZO) {
                if (build_bn_bb_kernels(&pickup_defs[pi], &st->strings[si],
                                        &pk->Bn_kernel[si], &pk->Bb_kernel[si]) < 0)
                    AMR_FAIL("pickup_kernel_build");
            }
        }
    }

    /* ── Mics ── */
    st->n_mics = n_mics;
    st->mics   = (MicState*)calloc(n_mics > 0 ? n_mics : 1, sizeof(MicState));
    if (!st->mics) AMR_FAIL("mics_alloc");

    for (int mi = 0; mi < n_mics; ++mi) {
        st->mics[mi].gain    = mic_defs[mi].gain;
        st->mics[mi].polar_a = mic_defs[mi].polar_a;
        st->mics[mi].polar_b = mic_defs[mi].polar_b;
        st->mics[mi].axis[0] = mic_defs[mi].axis[0];
        st->mics[mi].axis[1] = mic_defs[mi].axis[1];
        st->mics[mi].axis[2] = mic_defs[mi].axis[2];
        /* grid_xyz not used in AMR path */
    }

    /* ── Output ring buffers ── */
    st->pickup_out = (float**)calloc(n_pickups > 0 ? n_pickups : 1, sizeof(float*));
    st->mic_out    = (float**)calloc(n_mics    > 0 ? n_mics    : 1, sizeof(float*));
    if (!st->pickup_out || !st->mic_out) AMR_FAIL("output_bufs_alloc");

    for (int pi = 0; pi < n_pickups; ++pi) {
        st->pickup_out[pi] = alloc_f(OUTPUT_BUF);
        if (!st->pickup_out[pi]) AMR_FAIL("pickup_out_alloc");
    }
    for (int mi = 0; mi < n_mics; ++mi) {
        st->mic_out[mi] = alloc_f(OUTPUT_BUF);
        if (!st->mic_out[mi]) AMR_FAIL("mic_out_alloc");
    }

    st->pickup_accum = alloc_f(n_pickups > 0 ? n_pickups : 1);
    if (!st->pickup_accum) AMR_FAIL("pickup_accum_alloc");

    /* ── Saddle bilinear indices (into AMR plate grid) ── *
     *                                                      *
     * The AMR plate is a regular 2-D grid plate_Nx × plate_Ny.
     * Flat index = j + plate_Ny * i, identical to the FDTD   *
     * plate layout.  Reuse st->Nx/Ny set above.              */
    if (n_strings > 0) {
        st->saddle_idx4    = alloc_i(n_strings * 4);
        st->saddle_wgt4    = alloc_f(n_strings * 4);
        st->saddle_w_batch = alloc_f(n_strings);
        if (!st->saddle_idx4 || !st->saddle_wgt4 || !st->saddle_w_batch) AMR_FAIL("saddle_alloc");

        int Nx_ = st->Nx, Ny_ = st->Ny;
        for (int si = 0; si < n_strings; ++si) {
            const StringState* ss = &st->strings[si];
            int Nn = ss->n_nodes;
            float sx = ss->seg_xyz[(Nn-1)*3+0];
            float sy = ss->seg_xyz[(Nn-1)*3+1];
            float gi = (sx - st->origin[0]) / st->dx;
            float gj = (sy - st->origin[1]) / st->dx;
            if (gi < 0.0f) gi = 0.0f;  if (gi > (float)(Nx_-1)) gi = (float)(Nx_-1);
            if (gj < 0.0f) gj = 0.0f;  if (gj > (float)(Ny_-1)) gj = (float)(Ny_-1);
            int i0 = (int)gi, i1 = (i0 < Nx_-1) ? i0+1 : i0;
            int j0 = (int)gj, j1 = (j0 < Ny_-1) ? j0+1 : j0;
            float fi = gi - (float)i0;
            float fj = gj - (float)j0;
            st->saddle_idx4[si*4+0] = j0 + Ny_*i0;
            st->saddle_idx4[si*4+1] = j0 + Ny_*i1;
            st->saddle_idx4[si*4+2] = j1 + Ny_*i0;
            st->saddle_idx4[si*4+3] = j1 + Ny_*i1;
            st->saddle_wgt4[si*4+0] = (1.0f-fi)*(1.0f-fj);
            st->saddle_wgt4[si*4+1] = fi        *(1.0f-fj);
            st->saddle_wgt4[si*4+2] = (1.0f-fi)*fj;
            st->saddle_wgt4[si*4+3] = fi        *fj;
        }
    }

    /* ── Precompute mic samplers via AMR backend ── */
    st->any_mic_needs_velocity = 0;
    for (int mi = 0; mi < n_mics; ++mi) {
        if (mic_defs[mi].polar_b != 0.0f) { st->any_mic_needs_velocity = 1; break; }
    }

    if (n_mics > 0) {
        st->mic_P  = alloc_f(n_mics);
        st->mic_vx = alloc_f(n_mics);
        st->mic_vy = alloc_f(n_mics);
        st->mic_vz = alloc_f(n_mics);
        if (!st->mic_P || !st->mic_vx || !st->mic_vy || !st->mic_vz) AMR_FAIL("mic_arrays_alloc");

        /* Pack world-space mic positions and axes for the backend */
        float* mic_pos_xyz  = alloc_f(n_mics * 3);
        float* mic_axis_xyz = alloc_f(n_mics * 3);
        if (!mic_pos_xyz || !mic_axis_xyz) {
            free(mic_pos_xyz); free(mic_axis_xyz);
            AMR_FAIL("mic_pos_axis_alloc");
        }
        for (int mi = 0; mi < n_mics; ++mi) {
            mic_pos_xyz [mi*3+0] = mic_defs[mi].pos [0];
            mic_pos_xyz [mi*3+1] = mic_defs[mi].pos [1];
            mic_pos_xyz [mi*3+2] = mic_defs[mi].pos [2];
            mic_axis_xyz[mi*3+0] = mic_defs[mi].axis[0];
            mic_axis_xyz[mi*3+1] = mic_defs[mi].axis[1];
            mic_axis_xyz[mi*3+2] = mic_defs[mi].axis[2];
        }
        int rc_mic = st->pressure_backend->setup_mic_samplers(
            n_mics, mic_pos_xyz, mic_axis_xyz, st->any_mic_needs_velocity);
        free(mic_pos_xyz);
        free(mic_axis_xyz);
        if (rc_mic != SK_OK) AMR_FAIL("setup_mic_samplers");
    }

    /* ── Pickup GEMV matrices ── */
    {
        st->string_seg_offsets = alloc_i(n_strings + 1);
        if (!st->string_seg_offsets) AMR_FAIL("seg_offsets_alloc");
        st->string_seg_offsets[0] = 0;
        for (int si = 0; si < n_strings; ++si)
            st->string_seg_offsets[si+1] =
                st->string_seg_offsets[si] + st->strings[si].n_segs;
        st->total_seg_flat = (n_strings > 0) ? st->string_seg_offsets[n_strings] : 0;

        int n_mag = 0;
        for (int pi = 0; pi < n_pickups; ++pi)
            if (pickup_defs[pi].type != PICKUP_PIEZO) ++n_mag;
        st->n_mag_pickups = n_mag;

        if (n_mag > 0 && st->total_seg_flat > 0) {
            st->mag_pickup_map = alloc_i(n_mag);
            st->pickup_K0      = alloc_f(n_mag * st->total_seg_flat);
            st->pickup_K1      = alloc_f(n_mag * st->total_seg_flat);
            st->string_v0_flat = alloc_f(st->total_seg_flat);
            st->string_v1_flat = alloc_f(st->total_seg_flat);
            if (!st->mag_pickup_map || !st->pickup_K0 || !st->pickup_K1 ||
                !st->string_v0_flat || !st->string_v1_flat) AMR_FAIL("gemv_alloc");

            int mi3 = 0;
            for (int pi = 0; pi < n_pickups; ++pi) {
                if (pickup_defs[pi].type == PICKUP_PIEZO) continue;
                st->mag_pickup_map[mi3] = pi;
                float* K0_row = st->pickup_K0 + mi3 * st->total_seg_flat;
                float* K1_row = st->pickup_K1 + mi3 * st->total_seg_flat;
                const PickupState* pk = &st->pickups[pi];
                for (int si = 0; si < n_strings; ++si) {
                    if (!(pk->string_mask & (1u << (unsigned)si))) continue;
                    if (!pk->Bn_kernel[si] || !pk->Bb_kernel[si]) continue;
                    int   off = st->string_seg_offsets[si];
                    float ds  = st->strings[si].ds;
                    int   N   = st->strings[si].n_segs;
                    for (int s = 0; s < N; ++s) {
                        K0_row[off + s] = pk->Bn_kernel[si][s] * ds;
                        K1_row[off + s] = pk->Bb_kernel[si][s] * ds;
                    }
                }
                ++mi3;
            }
        }
    }

    if (st->n_mag_pickups > 0) {
        st->mag_pickup_tmp = alloc_f(st->n_mag_pickups);
        if (!st->mag_pickup_tmp) AMR_FAIL("mag_pickup_tmp_alloc");
    }

    /* ── Per-string bridge injection kernels ── *
     *                                            *
     * Maps each string saddle XY to its nearest   *
     * bridge plate nodes (Gaussian weighting by   *
     * distance in plate 2-D space).               */
    if (n_strings > 0 && st->n_bridge_cells > 0) {
        int nb = st->n_bridge_cells;
        st->string_bridge_kernel = alloc_f(n_strings * nb);
        st->bridge_drive         = alloc_f(nb);
        if (!st->string_bridge_kernel || !st->bridge_drive) AMR_FAIL("bridge_kernel_alloc");

        float sigma   = (amr_desc->plate_dx > 0.0f)
                        ? 2.0f * amr_desc->plate_dx : 0.005f;
        float inv_2s2 = 0.5f / (sigma * sigma);

        for (int si = 0; si < n_strings; ++si) {
            const StringState* ss = &st->strings[si];
            int Nn = ss->n_nodes;
            float saddle_x = ss->seg_xyz[(Nn-1)*3+0];
            float saddle_y = ss->seg_xyz[(Nn-1)*3+1];

            float* row  = st->string_bridge_kernel + si * nb;
            float  wsum = 0.0f;
            for (int c = 0; c < nb; ++c) {
                /* Recover XY world position from flat plate index:
                 * flat = j + plate_Ny * i  → i = flat/Ny, j = flat%Ny */
                int flat = st->bridge_cell_idx[c];
                int i = flat / st->Ny;
                int j = flat % st->Ny;
                float cx = st->origin[0] + (float)i * st->dx;
                float cy = st->origin[1] + (float)j * st->dx;
                float dx2 = saddle_x - cx;
                float dy2 = saddle_y - cy;
                row[c] = expf(-(dx2*dx2 + dy2*dy2) * inv_2s2);
                wsum  += row[c];
            }
            if (wsum > 1e-12f)
                for (int c = 0; c < nb; ++c) row[c] /= wsum;
        }
    }

    return st;

fail:
    coevolver_destroy(st);
    return NULL;
}
#undef AMR_FAIL

SK_API const char* coevolver_create_amr_last_error(void)
{
    return g_coevolver_amr_create_error.c_str();
}

/* ── coevolver_schedule_pluck / coevolver_clear_pluck_schedule ────────────── */

SK_API int coevolver_schedule_pluck(
    AcousticCoEvolverState* st,
    int onset_sample, int string_idx,
    float position_norm, float amplitude,
    int n_duration_samples)
{
    if (!st) return CE_ERR_NULL;
    if (coevolver_busy(st)) return CE_ERR_BUSY;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_PARAM;
    if (onset_sample < 0) onset_sample = 0;  /* clamp negative → fire immediately */

    /* Grow buffer if needed */
    if (st->n_pluck_events >= st->pluck_capacity) {
        int newcap = st->pluck_capacity < 8 ? 8 : st->pluck_capacity * 2;
        PluckEvent* nb = (PluckEvent*)realloc(
            st->pluck_events, newcap * sizeof(PluckEvent));
        if (!nb) return CE_ERR_NULL;
        st->pluck_events  = nb;
        st->pluck_capacity = newcap;
    }
    /* Insertion-sort: maintain events sorted by onset_sample. */
    int idx = st->n_pluck_events++;
    while (idx > 0 && st->pluck_events[idx - 1].onset_sample > onset_sample) {
        st->pluck_events[idx] = st->pluck_events[idx - 1];
        --idx;
    }
    st->pluck_events[idx] = {
        onset_sample, string_idx, position_norm, amplitude, n_duration_samples
    };
    return CE_OK;
}

SK_API void coevolver_clear_pluck_schedule(AcousticCoEvolverState* st)
{
    if (!st || coevolver_busy(st)) return;
    st->n_pluck_events = 0;
    st->pluck_cursor   = 0;
}

/* ── coevolver_destroy ────────────────────────────────────────────────────── */

SK_API void coevolver_destroy(AcousticCoEvolverState* st)
{
    if (!st) return;

    /* Cancel any running async job and join before freeing shared state */
    if (st->thread_state) {
        if (st->thread_state->worker.joinable()) {
            st->thread_state->cancel_flag.store(1);
            st->thread_state->worker.join();
        }
        delete st->thread_state;
        st->thread_state = nullptr;
    }

    fdtd_destroy(st->fdtd);
    if (st->pressure_backend) {
        delete st->pressure_backend;
        st->pressure_backend = nullptr;
    }
    free(st->bridge_cell_idx);
    free(st->bridge_cell_wgt);
    free(st->neck_plate_idx);
    free(st->neck_plate_wgt);

    if (st->strings) {
        for (int si = 0; si < st->n_strings; ++si) {
            StringState* ss = &st->strings[si];
            for (int p = 0; p < 2; ++p) {
                free(ss->u[p]); free(ss->u_prev[p]); free(ss->u_tmp[p]); free(ss->ext_force[p]);
            }
            free(ss->seg_xyz); free(ss->seg_tangent); free(ss->seg_normal); free(ss->seg_binormal);
            free(ss->modal_basis);
            free(ss->timed_force_buf);
        }
        free(st->strings);
    }

    if (st->pickups) {
        for (int pi = 0; pi < st->n_pickups; ++pi) {
            PickupState* pk = &st->pickups[pi];
            if (pk->Bn_kernel) {
                for (int si = 0; si < pk->n_strings; ++si) free(pk->Bn_kernel[si]);
                free(pk->Bn_kernel);
            }
            if (pk->Bb_kernel) {
                for (int si = 0; si < pk->n_strings; ++si) free(pk->Bb_kernel[si]);
                free(pk->Bb_kernel);
            }
            free(pk->n_segs_per_string);
        }
        free(st->pickups);
    }

    free(st->mics);

    if (st->pickup_out) {
        for (int pi = 0; pi < st->n_pickups; ++pi) free(st->pickup_out[pi]);
        free(st->pickup_out);
    }
    if (st->mic_out) {
        for (int mi = 0; mi < st->n_mics; ++mi) free(st->mic_out[mi]);
        free(st->mic_out);
    }
    free(st->pickup_accum);
    free(st->saddle_idx4);
    free(st->saddle_wgt4);
    free(st->saddle_w_batch);
    free(st->mic_rec_xyz);
    free(st->mic_P);
    free(st->mic_vx);
    free(st->mic_vy);
    free(st->mic_vz);
    free(st->mic_P_idx8);
    free(st->mic_P_wgt8);
    free(st->mic_Vx_idx8);
    free(st->mic_Vx_wgt8);
    free(st->mic_Vy_idx8);
    free(st->mic_Vy_wgt8);
    free(st->mic_Vz_idx8);
    free(st->mic_Vz_wgt8);
    free(st->string_bridge_kernel);
    free(st->bridge_drive);
    free(st->string_seg_offsets);
    free(st->mag_pickup_map);
    free(st->pickup_K0);
    free(st->pickup_K1);
    free(st->string_v0_flat);
    free(st->string_v1_flat);
    free(st->mag_pickup_tmp);
    free(st->pluck_events);
    free(st);
}

/* ── Pickup integration ───────────────────────────────────────────────────── */

static float pickup_piezo_signal(const PickupState* pk,
                                  const StringState* ss,
                                  int string_idx)
{
    if (!(pk->string_mask & (1u << (unsigned)string_idx))) return 0.0f;
    int N = ss->n_nodes;
    /* Slope at saddle = (u[N-1] - u[N-2]) / ds */
    float slope0 = (ss->u[0][N-1] - ss->u[0][N-2]) / ss->ds;
    float slope1 = (ss->u[1][N-1] - ss->u[1][N-2]) / ss->ds;
    float slope_mag = sqrtf(slope0*slope0 + slope1*slope1);
    return pk->sensitivity * ss->tension_eff * slope_mag;
}

/* ── coevolver_step ───────────────────────────────────────────────────────── */

/* Advance the co-evolver by exactly one audio sample.
 * sample_abs is the absolute sample counter (st->samples_written + block offset);
 * pass -1 to skip timed-force-buffer injection (used by block_with_drive path). */
static int _step_one_sample(AcousticCoEvolverState* st, int sample_abs)
{
    int K  = st->substeps;
    int nb = st->n_bridge_cells;

    memset(st->pickup_accum, 0, st->n_pickups * sizeof(float));

    /* Apply global damping scale and pre-compute per-substep force for each string.
     * timed_force_buf is indexed by (sample_abs − force_onset_sample) so it
     * aligns correctly regardless of when within a block the pluck was scheduled.
     * Dividing by (K × n_substeps) distributes the audio-sample force evenly
     * across all string substeps, eliminating the impulsive first-substep spike. */
    float per_sub[64] = {};
    for (int si = 0; si < st->n_strings && si < 64; ++si) {
        StringState* ss = &st->strings[si];
        ss->gamma_eff = ss->gamma * st->gamma_scale;
        int buf_idx = sample_abs - ss->force_onset_sample;
        float f_total = (buf_idx >= 0 && buf_idx < ss->n_force_buf)
                        ? ss->timed_force_buf[buf_idx] : 0.0f;
        int total_str_steps = K * ss->n_substeps;
        per_sub[si] = (total_str_steps > 0 && f_total != 0.0f)
                      ? f_total / (float)total_str_steps : 0.0f;
    }

    for (int sub = 0; sub < K; ++sub) {

        float neck_plate_force = 0.0f;
        if (st->neck_mass_eff > 0.0f && st->n_neck_plate_cells > 0) {
            float plate_w = 0.0f;
            if (st->pressure_backend) {
                st->pressure_backend->sample_plate_weighted(
                    st->n_neck_plate_cells,
                    st->neck_plate_idx,
                    st->neck_plate_wgt,
                    &plate_w);
            } else {
                fdtd_sample_plate_weighted(
                    st->fdtd,
                    st->n_neck_plate_cells,
                    st->neck_plate_idx,
                    st->neck_plate_wgt,
                    &plate_w);
            }

            float f_strings = 0.0f;
            for (int si = 0; si < st->n_strings; ++si) {
                StringState* ss = &st->strings[si];
                if (ss->n_nodes > 1)
                    f_strings += ss->tension_eff * (ss->u[0][1] - st->neck_w) / ss->ds;
            }

            float dt_n = st->dt_fdtd;
            if (dt_n <= 0.0f) dt_n = st->dt_audio;
            float inv_dt = 1.0f / dt_n;
            float inv_dt2 = inv_dt * inv_dt;
            float plate_v = (plate_w - st->neck_plate_w_prev) * inv_dt;

            float m = st->neck_mass_eff;
            float c = st->neck_damp_eff + st->neck_body_damp_eff;
            float k = st->neck_stiff_eff + st->neck_body_stiff_eff;
            float denom = m * inv_dt2 + 0.5f * c * inv_dt;
            float rhs = f_strings
                      + st->neck_body_stiff_eff * plate_w
                      + st->neck_body_damp_eff * plate_v
                      + (2.0f * m * inv_dt2 - k) * st->neck_w
                      + (-m * inv_dt2 + 0.5f * c * inv_dt) * st->neck_w_prev;
            float neck_new = (denom > 1e-12f) ? rhs / denom : plate_w;
            if (!isfinite(neck_new) || fabsf(neck_new) > 0.05f)
                return CE_ERR_UNSTABLE;

            st->neck_w_prev = st->neck_w;
            st->neck_w = neck_new;
            float neck_v = (st->neck_w - st->neck_w_prev) * inv_dt;
            neck_plate_force = st->neck_body_stiff_eff * (st->neck_w - plate_w)
                             + st->neck_body_damp_eff * (neck_v - plate_v);
            if (!isfinite(neck_plate_force)) return CE_ERR_UNSTABLE;
            st->neck_plate_w_prev = plate_w;
        }

        /* Step all strings, injecting the sub-dt share of this sample's force
         * immediately before each string substep so every step gets equal drive. */
        for (int si = 0; si < st->n_strings; ++si) {
            StringState* ss = &st->strings[si];
            for (int ss_sub = 0; ss_sub < ss->n_substeps; ++ss_sub) {
                int rc_ten = string_update_geometric_tension(ss);
                if (rc_ten != CE_OK) return rc_ten;
                if (per_sub[si] != 0.0f)
                    ss->ext_force[0][ss->force_node] += per_sub[si];
                float nut_bc = (st->neck_mass_eff > 0.0f) ? st->neck_w : 0.0f;
                int rc_s0 = string_step_pol(ss, 0, nut_bc);
                if (rc_s0 != CE_OK) return rc_s0;
                int rc_s1 = string_step_pol(ss, 1, 0.0f);
                if (rc_s1 != CE_OK) return rc_s1;
                int rc_shape = string_validate_shape(ss);
                if (rc_shape != CE_OK) return rc_shape;
            }
        }

        /* Bridge drive: update a per-string bridge DOF, then distribute the
         * equal spring/damper reaction into the plate.  This keeps coupling
         * symmetric: string slope force drives the bridge, the bridge endpoint
         * drives the string, and the plate sees only the bridge reaction. */
        if (nb > 0 && st->bridge_drive && st->string_bridge_kernel) {
            float plate_force_arr[64]; /* stack; guarded by n_strings <= 64 */
            float dt_b = st->dt_fdtd;
            if (dt_b <= 0.0f) dt_b = st->dt_audio;
            int n_force_strings = st->n_strings < 64 ? st->n_strings : 64;
            for (int si = 0; si < n_force_strings; ++si) {
                StringState* ss = &st->strings[si];
                int Nn = ss->n_nodes;
                float slope0 = (ss->u[0][Nn-2] - ss->bridge_w) / ss->ds;
                if (!isfinite(slope0) || fabsf(slope0) > 1.0f)
                    return CE_ERR_UNSTABLE;
                float f_string = ss->tension_eff * slope0;

                float m = ss->bridge_mass_eff;
                float c = ss->bridge_damp_eff;
                float k = ss->bridge_stiff_eff;
                float inv_dt = 1.0f / dt_b;
                float inv_dt2 = inv_dt * inv_dt;
                float plate_v = (ss->saddle_w - ss->saddle_w_prev) * inv_dt;

                float denom = m * inv_dt2 + 0.5f * c * inv_dt;
                float rhs = f_string
                          + k * ss->saddle_w
                          + c * plate_v
                          + (2.0f * m * inv_dt2 - k) * ss->bridge_w
                          + (-m * inv_dt2 + 0.5f * c * inv_dt) * ss->bridge_w_prev;
                float bridge_new = (denom > 1e-12f) ? rhs / denom : ss->saddle_w;
                if (!isfinite(bridge_new) || fabsf(bridge_new) > 0.05f)
                    return CE_ERR_UNSTABLE;

                ss->bridge_w_prev = ss->bridge_w;
                ss->bridge_w = bridge_new;

                float bridge_v = (ss->bridge_w - ss->bridge_w_prev) * inv_dt;
                float f_plate = k * (ss->bridge_w - ss->saddle_w)
                              + c * (bridge_v - plate_v);
                if (!isfinite(f_plate)) return CE_ERR_UNSTABLE;
                plate_force_arr[si] = f_plate;
                ss->saddle_w_prev = ss->saddle_w;
            }
            using Eigen::Map;
            using Eigen::Matrix;
            using Eigen::Dynamic;
            using Eigen::RowMajor;
            using Eigen::VectorXf;
            Map<const Matrix<float,Dynamic,Dynamic,RowMajor>> Kmat(
                st->string_bridge_kernel, st->n_strings, nb);
            Map<const VectorXf> fs(plate_force_arr, n_force_strings);
            Map<VectorXf> bd(st->bridge_drive, nb);
            bd.noalias() = Kmat.topRows(n_force_strings).transpose() * fs;
        }

        /* Inject per-cell bridge drives into acoustic backend */
        if (nb > 0 && st->bridge_drive) {
            int rc_bdg;
            if (st->pressure_backend) {
                rc_bdg = st->pressure_backend->inject_bridge_drive(
                    st->bridge_drive, nb, 1.0f);
                if (rc_bdg != SK_OK) return CE_ERR_FDTD_BRIDGE;
            } else {
                rc_bdg = fdtd_inject_bridge_drive(
                    st->fdtd, st->bridge_drive, 1.0f);
                if (rc_bdg != FDTD_OK) return CE_ERR_FDTD_BRIDGE;
            }
        }

        if (neck_plate_force != 0.0f && st->n_neck_plate_cells > 0) {
            int rc_neck;
            if (st->pressure_backend) {
                rc_neck = st->pressure_backend->inject_plate_force(
                    st->n_neck_plate_cells,
                    st->neck_plate_idx,
                    st->neck_plate_wgt,
                    neck_plate_force);
                if (rc_neck != SK_OK) return CE_ERR_FDTD_BRIDGE;
            } else {
                rc_neck = fdtd_inject_plate_force(
                    st->fdtd,
                    st->n_neck_plate_cells,
                    st->neck_plate_idx,
                    st->neck_plate_wgt,
                    neck_plate_force);
                if (rc_neck != FDTD_OK) return CE_ERR_FDTD_BRIDGE;
            }
        }

        int rc;
        if (st->pressure_backend) {
            rc = st->pressure_backend->step(1);
            if (rc == FDTD_ERR_UNSTABLE) return CE_ERR_UNSTABLE;
            if (rc != SK_OK)             return CE_ERR_FDTD_STEP;
        } else {
            rc = fdtd_step(st->fdtd, 1);
            if (rc == FDTD_ERR_UNSTABLE) return CE_ERR_UNSTABLE;
            if (rc != FDTD_OK)           return CE_ERR_FDTD_STEP;
        }

        /* Plate → saddle BC: batch gather-dot, no full-plate memcpy. */
        if (st->pressure_backend) {
            st->pressure_backend->sample_plate_displacement_batch(
                st->n_strings,
                st->saddle_idx4,
                st->saddle_wgt4,
                st->saddle_w_batch);
        } else {
            fdtd_sample_plate_displacement_batch(
                st->fdtd,
                st->n_strings,
                st->saddle_idx4,
                st->saddle_wgt4,
                st->saddle_w_batch);
        }
        for (int si = 0; si < st->n_strings; ++si)
            st->strings[si].saddle_w = st->saddle_w_batch[si];

        ++st->modal_counter;
        if (st->modal_counter >= st->modal_stride) {
            st->modal_counter = 0;
            for (int si = 0; si < st->n_strings; ++si)
                modal_project(&st->strings[si]);
        }

    } /* end body sub-steps */

    /* ── Pickup integration ── */

    /* Magnetic pickups: pack velocity flat arrays then one GEMV per pickup row. */
    if (st->n_mag_pickups > 0 && st->total_seg_flat > 0) {
        for (int si = 0; si < st->n_strings; ++si) {
            const StringState* ss  = &st->strings[si];
            int   N    = ss->n_segs;
            int   off  = st->string_seg_offsets[si];
            float inv_dt = 1.0f / ss->dt_s;

            using CArr = Eigen::Map<const Eigen::ArrayXf>;
            CArr u0 (ss->u[0],      N);
            CArr u0p(ss->u_prev[0], N);
            CArr u1 (ss->u[1],      N);
            CArr u1p(ss->u_prev[1], N);
            Eigen::Map<Eigen::ArrayXf> v0f(st->string_v0_flat + off, N);
            Eigen::Map<Eigen::ArrayXf> v1f(st->string_v1_flat + off, N);
            v0f = (u0 - u0p) * inv_dt;
            v1f = (u1 - u1p) * inv_dt;
        }

        using RMat = Eigen::Map<const Eigen::Matrix<float,
                                    Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>;
        using CVec = Eigen::Map<const Eigen::VectorXf>;
        RMat K0(st->pickup_K0, st->n_mag_pickups, st->total_seg_flat);
        RMat K1(st->pickup_K1, st->n_mag_pickups, st->total_seg_flat);
        CVec v0f(st->string_v0_flat, st->total_seg_flat);
        CVec v1f(st->string_v1_flat, st->total_seg_flat);

        Eigen::Map<Eigen::VectorXf> mag_out(st->mag_pickup_tmp, st->n_mag_pickups);
        mag_out.noalias() = K0 * v0f;
        mag_out.noalias() += K1 * v1f;
        for (int mi = 0; mi < st->n_mag_pickups; ++mi)
            st->pickup_accum[st->mag_pickup_map[mi]] = mag_out[mi];
    }

    /* Piezo pickups: scalar path unchanged. */
    for (int pi = 0; pi < st->n_pickups; ++pi) {
        PickupState* pk = &st->pickups[pi];
        if (pk->type != PICKUP_PIEZO) continue;
        float sig = 0.0f;
        for (int si = 0; si < st->n_strings; ++si)
            sig += pickup_piezo_signal(pk, &st->strings[si], si);
        st->pickup_accum[pi] = sig;
    }

    /* ── Mic sampling — precomputed 8-tap gather (Issue 7) ── */
    int out_pos = st->out_write_pos;
    if (st->n_mics > 0) {
        if (st->pressure_backend) {
            int rc_m = st->pressure_backend->sample_mics(
                st->n_mics, st->mic_P, st->mic_vx, st->mic_vy, st->mic_vz);
            if (rc_m != SK_OK) return CE_ERR_FDTD_MIC_P;
        } else {
            int rc_p = fdtd_sample_pressure_precomputed(
                st->fdtd, st->n_mics,
                st->mic_P_idx8, st->mic_P_wgt8,
                st->mic_P);
            if (rc_p != FDTD_OK) return CE_ERR_FDTD_MIC_P;

            if (st->any_mic_needs_velocity) {
                int rc_v = fdtd_sample_velocity_precomputed(
                    st->fdtd, st->n_mics,
                    st->mic_Vx_idx8, st->mic_Vx_wgt8,
                    st->mic_Vy_idx8, st->mic_Vy_wgt8,
                    st->mic_Vz_idx8, st->mic_Vz_wgt8,
                    st->mic_vx, st->mic_vy, st->mic_vz);
                if (rc_v != FDTD_OK) return CE_ERR_FDTD_MIC_V;
            }
        }

        for (int mi = 0; mi < st->n_mics; ++mi) {
            const MicState* ms = &st->mics[mi];
            float sig = st->mic_P[mi] * ms->polar_a;
            if (ms->polar_b != 0.0f) {
                float v_n = st->mic_vx[mi]*ms->axis[0]
                          + st->mic_vy[mi]*ms->axis[1]
                          + st->mic_vz[mi]*ms->axis[2];
                sig += ms->polar_b * (st->rho_air * st->c_sound) * v_n;
            }
            st->mic_out[mi][out_pos] = sig * ms->gain;
        }
    }

    for (int pi = 0; pi < st->n_pickups; ++pi)
        st->pickup_out[pi][out_pos] = st->pickup_accum[pi];

    st->out_write_pos = (out_pos + 1) % OUTPUT_BUF;
    ++st->samples_written;
    return CE_OK;
}

SK_API int coevolver_step(AcousticCoEvolverState* st, int n_samples)
{
    if (!st) return CE_ERR_NULL;
    if (coevolver_busy(st)) return CE_ERR_BUSY;
    if (n_samples <= 0) return CE_OK;

    int base_sample = st->samples_written;
    for (int s = 0; s < n_samples; ++s) {
        int sample_idx = base_sample + s;
        st->current_sample_idx = sample_idx;
        while (st->pluck_cursor < st->n_pluck_events &&
               st->pluck_events[st->pluck_cursor].onset_sample <= sample_idx) {
            PluckEvent* ev = &st->pluck_events[st->pluck_cursor++];
            coevolver_pluck_string(st, ev->string_idx, ev->position_norm,
                                   ev->amplitude, ev->n_duration_samples);
        }
        int rc = _step_one_sample(st, sample_idx);
        if (rc != CE_OK) return rc;
    }
    return CE_OK;
}

/* ── coevolver_step_block_with_drive ─────────────────────────────────────── */

SK_API int coevolver_step_block_with_drive(
    AcousticCoEvolverState* st,
    const float* const*     string_drive_blocks,
    int                     n_strings,
    int                     n_samples,
    float                   force_pos_norm,
    float*                  mic_out_buf)
{
    if (!st || !mic_out_buf)        return CE_ERR_NULL;
    if (coevolver_busy(st))         return CE_ERR_BUSY;
    if (n_strings != st->n_strings) return CE_ERR_PARAM;
    if (n_samples < 0)              return CE_ERR_DIM;
    if (n_samples == 0)             return CE_OK;
    if (st->n_mics < 1)             return CE_ERR_DIM;

    /* Precompute force injection node and scale per string once per block.
     * Eliminates per-sample node/scale recomputation from the hot loop. */
    int   force_node        [64];
    float force_scale_per_str[64];
    for (int si = 0; si < n_strings; ++si) {
        StringState* ss = &st->strings[si];
        int   N   = ss->n_nodes;
        float pos = fmaxf(0.0f, fminf(1.0f, force_pos_norm));
        int   i0  = (int)roundf(pos * (float)(N - 1));
        if (i0 <= 0)   i0 = 1;
        if (i0 >= N-1) i0 = N - 2;
        force_node[si]        = i0;
        force_scale_per_str[si] = ss->dt_s * ss->dt_s / (ss->linear_mass * ss->ds);
    }

    /* Step sample-by-sample.  Direct-drive injection writes straight to
     * ext_force; _step_one_sample is called with -1 so it skips the
     * timed_force_buf path entirely. */
    float* mic0_ring = st->mic_out[0];
    int base_sample = st->samples_written;
    for (int s = 0; s < n_samples; ++s) {
        int sample_idx = base_sample + s;
        /* Fire any pending pluck events at this sample index. */
        while (st->pluck_cursor < st->n_pluck_events &&
               st->pluck_events[st->pluck_cursor].onset_sample <= sample_idx) {
            PluckEvent* ev = &st->pluck_events[st->pluck_cursor++];
            coevolver_pluck_string(st, ev->string_idx,
                                   ev->position_norm, ev->amplitude,
                                   ev->n_duration_samples);
        }

        /* Direct drive: write scaled force straight into ext_force. */
        if (string_drive_blocks) {
            for (int si = 0; si < n_strings; ++si) {
                const float* d = string_drive_blocks[si];
                if (!d) continue;
                float f = d[s];
                if (f != 0.0f)
                    st->strings[si].ext_force[0][force_node[si]] +=
                        f * force_scale_per_str[si];
            }
        }

        int rc = _step_one_sample(st, -1);
        if (rc != CE_OK) return rc;

        int last = (st->out_write_pos - 1 + OUTPUT_BUF) % OUTPUT_BUF;
        mic_out_buf[s] = mic0_ring[last];
    }

    return CE_OK;
}

/* ── Async step ───────────────────────────────────────────────────────────── */

static void _worker_fn(AcousticCoEvolverState* st)
{
    ThreadState* ts = st->thread_state;
    int n = ts->progress_total.load(std::memory_order_relaxed);

    ts->status.store(CE_STATUS_RUNNING, std::memory_order_release);

    int base_sample = st->samples_written;
    for (int s = 0; s < n; ++s) {
        int sample_idx = base_sample + s;
        if (ts->cancel_flag.load(std::memory_order_acquire)) {
            ts->status.store(CE_STATUS_CANCELLED, std::memory_order_release);
            return;
        }
        while (st->pluck_cursor < st->n_pluck_events &&
               st->pluck_events[st->pluck_cursor].onset_sample <= sample_idx) {
            PluckEvent* ev = &st->pluck_events[st->pluck_cursor++];
            coevolver_pluck_string(st, ev->string_idx, ev->position_norm,
                                   ev->amplitude, ev->n_duration_samples);
        }
        int rc = _step_one_sample(st, s);
        ts->progress_samples.fetch_add(1, std::memory_order_relaxed);
        if (rc != CE_OK) {
            ts->last_error.store(rc);
            ts->status.store(CE_STATUS_ERROR, std::memory_order_release);
            return;
        }
    }

    ts->status.store(CE_STATUS_DONE, std::memory_order_release);
}

SK_API int coevolver_step_async(AcousticCoEvolverState* st, int n_samples)
{
    if (!st || !st->thread_state) return CE_ERR_NULL;
    if (n_samples <= 0) return CE_OK;
    ThreadState* ts = st->thread_state;

    if (ts->status.load() == CE_STATUS_RUNNING) return CE_ERR_PARAM;

    /* Join previous thread if it completed but was never joined */
    if (ts->worker.joinable()) ts->worker.join();

    ts->cancel_flag    .store(0,        std::memory_order_relaxed);
    ts->progress_samples.store(0,       std::memory_order_relaxed);
    ts->progress_total  .store(n_samples, std::memory_order_relaxed);
    ts->last_error      .store(CE_OK,   std::memory_order_relaxed);
    ts->status          .store(CE_STATUS_IDLE, std::memory_order_release);

    ts->worker = std::thread(_worker_fn, st);
    return CE_OK;
}

SK_API void coevolver_get_progress(const AcousticCoEvolverState* st,
                                    CoEvolverProgress* out)
{
    if (!st || !st->thread_state || !out) return;
    const ThreadState* ts = st->thread_state;
    out->samples_done  = ts->progress_samples.load(std::memory_order_relaxed);
    out->samples_total = ts->progress_total  .load(std::memory_order_relaxed);
    out->status        = ts->status          .load(std::memory_order_acquire);
    out->error_code    = ts->last_error      .load(std::memory_order_relaxed);
}

SK_API int coevolver_wait(AcousticCoEvolverState* st)
{
    if (!st || !st->thread_state) return CE_ERR_NULL;
    ThreadState* ts = st->thread_state;
    if (ts->worker.joinable()) ts->worker.join();
    return ts->last_error.load(std::memory_order_relaxed);
}

SK_API int coevolver_cancel(AcousticCoEvolverState* st)
{
    if (!st || !st->thread_state) return CE_ERR_NULL;
    ThreadState* ts = st->thread_state;
    ts->cancel_flag.store(1, std::memory_order_release);
    if (ts->worker.joinable()) ts->worker.join();
    return CE_OK;
}

SK_API int coevolver_is_running(const AcousticCoEvolverState* st)
{
    if (!st || !st->thread_state) return 0;
    return st->thread_state->status.load(std::memory_order_acquire) == CE_STATUS_RUNNING
           ? 1 : 0;
}

/* ── coevolver_pluck_string ───────────────────────────────────────────────── */

SK_API int coevolver_pluck_string(
    AcousticCoEvolverState* st,
    int   string_idx,
    float position_norm,
    float amplitude,
    int   n_duration_samples)
{
    if (!st) return CE_ERR_NULL;
    if (coevolver_busy(st)) return CE_ERR_BUSY;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_STRING_IDX;

    StringState* ss = &st->strings[string_idx];
    int N = ss->n_nodes;
    float pos = fmaxf(1e-3f, fminf(1.0f - 1e-3f, position_norm));
    float amp = fmaxf(-0.006f, fminf(0.006f, amplitude));

    int i_p = (int)roundf(pos * (float)(N - 1));
    if (i_p <= 0)   i_p = 1;
    if (i_p >= N-1) i_p = N - 2;

    /* Quasi-static transverse force to achieve triangular displacement 'amp':
     *   F = T * |amp| * (1/left_length + 1/right_length)
     * where left_length and right_length are the arc lengths on each side. */
    float left_len  = fmaxf((float)i_p * ss->ds,       ss->ds);
    float right_len = fmaxf((float)(N-1-i_p) * ss->ds, ss->ds);
    float F_peak    = ss->tension_N * fabsf(amp) * (1.0f / left_len + 1.0f / right_len);
    if (amp < 0.0f) F_peak = -F_peak;

    /* Draw-phase length: caller may specify an explicit sample count for
     * stability tests or slow strum draws; 0 → default ~20 ms. */
    int draw_samples = (n_duration_samples > 0)
        ? n_duration_samples
        : (int)roundf(0.020f * st->sample_rate);
    if (draw_samples < 4) draw_samples = 4;

    float* fbuf = (float*)malloc(draw_samples * sizeof(float));
    if (!fbuf) return CE_ERR_NULL;

    float scale = ss->dt_s * ss->dt_s / (ss->linear_mass * ss->ds);
    for (int s = 0; s < draw_samples; ++s) {
        float t = (float)s / (float)(draw_samples - 1);
        float env = 0.5f * (1.0f - cosf((float)M_PI * t));   /* 0 → 1 */
        fbuf[s] = F_peak * env * scale;
    }

    float* nb = (float*)realloc(ss->timed_force_buf, draw_samples * sizeof(float));
    if (!nb) { free(fbuf); return CE_ERR_NULL; }
    ss->timed_force_buf    = nb;
    memcpy(ss->timed_force_buf, fbuf, draw_samples * sizeof(float));
    free(fbuf);

    ss->n_force_buf        = draw_samples;
    ss->force_node         = i_p;
    ss->force_onset_sample = st->current_sample_idx;
    return CE_OK;
}

/* ── coevolver_inject_string_force ──────────────────────────────────────────── */

SK_API int coevolver_inject_string_force(
    AcousticCoEvolverState* st,
    int          string_idx,
    int          n_samples,
    const float* force_buf,
    float        force_pos_norm)
{
    if (!st || !force_buf) return CE_ERR_NULL;
    if (coevolver_busy(st)) return CE_ERR_BUSY;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_STRING_IDX;
    if (n_samples <= 0) return CE_OK;

    StringState* ss = &st->strings[string_idx];
    int N   = ss->n_nodes;
    float pos = fmaxf(0.0f, fminf(1.0f, force_pos_norm));
    int   i0  = (int)roundf(pos * (float)(N-1));
    if (i0 <= 0) i0 = 1;
    if (i0 >= N-1) i0 = N-2;

    /* Resize the timed force buffer for the new block (Issue 12).
     * Store pre-scaled per-sample force so _step_one_sample just adds it.
     * Scale = dt_s²/(mu·ds) converts force (N) to displacement increment. */
    float* nb = (float*)realloc(ss->timed_force_buf, n_samples * sizeof(float));
    if (!nb) return CE_ERR_NULL;
    ss->timed_force_buf = nb;

    float scale = ss->dt_s * ss->dt_s / (ss->linear_mass * ss->ds);
    for (int s = 0; s < n_samples; ++s)
        ss->timed_force_buf[s] = force_buf[s] * scale;

    ss->n_force_buf        = n_samples;
    ss->force_node         = i0;
    ss->force_onset_sample = st->samples_written;  /* starts at next coevolver_step block */
    return CE_OK;
}

/* ── Output accessors ─────────────────────────────────────────────────────── */

/* Helper: copy n_samples from ring buffer ending at out_write_pos */
static int copy_ring(const float* ring, int write_pos, float* out, int n_samples)
{
    if (n_samples > OUTPUT_BUF) return CE_ERR_DIM;
    int start = (write_pos - n_samples + OUTPUT_BUF) % OUTPUT_BUF;
    for (int i = 0; i < n_samples; ++i)
        out[i] = ring[(start + i) % OUTPUT_BUF];
    return CE_OK;
}

SK_API int coevolver_get_pickup_output(
    const AcousticCoEvolverState* st,
    int pickup_idx, float* out, int n_samples)
{
    if (!st || !out) return CE_ERR_NULL;
    if (pickup_idx < 0 || pickup_idx >= st->n_pickups) return CE_ERR_DIM;
    return copy_ring(st->pickup_out[pickup_idx], st->out_write_pos, out, n_samples);
}

SK_API int coevolver_get_mic_output(
    const AcousticCoEvolverState* st,
    int mic_idx, float* out, int n_samples)
{
    if (!st || !out) return CE_ERR_NULL;
    if (mic_idx < 0 || mic_idx >= st->n_mics) return CE_ERR_DIM;
    return copy_ring(st->mic_out[mic_idx], st->out_write_pos, out, n_samples);
}

SK_API int coevolver_get_pressure_field(
    const AcousticCoEvolverState* st, float* out, int out_len)
{
    if (!st) return CE_ERR_NULL;
    if (st->pressure_backend)
        return st->pressure_backend->get_pressure_field_flat(out, out_len);
    return fdtd_get_pressure_field(st->fdtd, out, out_len);
}

SK_API int coevolver_get_pressure_field_uniform(
    const AcousticCoEvolverState* st,
    int Nx, int Ny, int Nz,
    const double bmin[3], const double bmax[3],
    float* out, int out_len)
{
    if (!st || !out || !bmin || !bmax) return CE_ERR_NULL;
    if (out_len != Nx * Ny * Nz) return CE_ERR_DIM;
    if (st->pressure_backend)
        return st->pressure_backend->get_pressure_field_uniform(
            Nx, Ny, Nz, bmin, bmax, out, out_len);
    /* Legacy uniform FDTD path — bmin/bmax ignored, dims must match exactly */
    return fdtd_get_pressure_field(st->fdtd, out, out_len);
}

SK_API int coevolver_get_amr_cell_centers(
    const AcousticCoEvolverState* st,
    float* out_xyz, int n_cells_3)
{
    if (!st || !out_xyz) return CE_ERR_NULL;
    if (!st->pressure_backend) return CE_ERR_PARAM;
    return st->pressure_backend->get_cell_centers(out_xyz, n_cells_3);
}

SK_API int coevolver_get_plate_displacement(
    const AcousticCoEvolverState* st, float* out, int out_len)
{
    if (!st) return CE_ERR_NULL;
    if (st->pressure_backend)
        return st->pressure_backend->get_plate_displacement(out, out_len);
    return fdtd_get_plate_displacement(st->fdtd, out, out_len);
}

SK_API int coevolver_get_plate_dims(
    const AcousticCoEvolverState* st,
    int* out_Nx, int* out_Ny, int* out_count)
{
    if (!st || !out_Nx || !out_Ny || !out_count) return CE_ERR_NULL;
    if (st->pressure_backend)
        return st->pressure_backend->get_plate_dims(out_Nx, out_Ny, out_count);
    *out_Nx = st->Nx;
    *out_Ny = st->Ny;
    *out_count = st->Nx * st->Ny;
    return CE_OK;
}

SK_API int coevolver_get_modal_amplitudes(
    const AcousticCoEvolverState* st,
    int string_idx, float* out_re, float* out_im, int n_modes)
{
    if (!st || !out_re || !out_im) return CE_ERR_NULL;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_STRING_IDX;
    int m = (n_modes < N_MODES) ? n_modes : N_MODES;
    const StringState* ss = &st->strings[string_idx];
    for (int k = 0; k < m; ++k) {
        out_re[k] = ss->modal_re[k];
        out_im[k] = ss->modal_im[k];
    }
    return CE_OK;
}

SK_API int coevolver_sample_pressure(
    const AcousticCoEvolverState* st,
    int n_rec, const float* rec_xyz, float* out_p)
{
    if (!st || !rec_xyz || !out_p) return CE_ERR_NULL;
    /* AMR backend does not expose a uniform grid; world-space sampling is
     * unsupported through this legacy interface.  Use get_pressure_field()
     * to retrieve the flat cell-indexed field and sample externally. */
    if (st->pressure_backend) return CE_ERR_PARAM;
    /* Convert world coords to grid coords */
    float* grid_coords = (float*)malloc(n_rec * 3 * sizeof(float));
    if (!grid_coords) return CE_ERR_NULL;
    for (int r = 0; r < n_rec; ++r)
        world_to_grid(st, rec_xyz + r*3, grid_coords + r*3);
    int rc = fdtd_sample_pressure(st->fdtd, n_rec, grid_coords, out_p);
    free(grid_coords);
    return (rc == FDTD_OK) ? CE_OK : CE_ERR_NULL;
}

SK_API int coevolver_get_string_velocity(
    const AcousticCoEvolverState* st,
    int string_idx, float* out_v, int out_len)
{
    if (!st || !out_v) return CE_ERR_NULL;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_STRING_IDX;
    const StringState* ss = &st->strings[string_idx];
    if (out_len != ss->n_segs * 3) return CE_ERR_DIM;
    float inv_dt = 1.0f / ss->dt_s;
    for (int s = 0; s < ss->n_segs; ++s) {
        float v0 = (ss->u[0][s] - ss->u_prev[0][s]) * inv_dt;
        float v1 = (ss->u[1][s] - ss->u_prev[1][s]) * inv_dt;
        float* n = ss->seg_normal   + s*3;
        float* b = ss->seg_binormal + s*3;
        out_v[s*3+0] = v0*n[0] + v1*b[0];
        out_v[s*3+1] = v0*n[1] + v1*b[1];
        out_v[s*3+2] = v0*n[2] + v1*b[2];
    }
    return CE_OK;
}

SK_API int coevolver_get_string_displacement(
    const AcousticCoEvolverState* st,
    int string_idx, float* out_d, int out_len)
{
    if (!st || !out_d) return CE_ERR_NULL;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_STRING_IDX;
    const StringState* ss = &st->strings[string_idx];
    if (out_len != ss->n_segs * 3) return CE_ERR_DIM;
    for (int s = 0; s < ss->n_segs; ++s) {
        float d0 = ss->u[0][s];
        float d1 = ss->u[1][s];
        float* n = ss->seg_normal   + s*3;
        float* b = ss->seg_binormal + s*3;
        out_d[s*3+0] = d0*n[0] + d1*b[0];
        out_d[s*3+1] = d0*n[1] + d1*b[1];
        out_d[s*3+2] = d0*n[2] + d1*b[2];
    }
    return CE_OK;
}

SK_API int coevolver_get_string_position(
    const AcousticCoEvolverState* st,
    int string_idx, float* out_xyz, int out_len)
{
    if (!st || !out_xyz) return CE_ERR_NULL;
    if (string_idx < 0 || string_idx >= st->n_strings) return CE_ERR_STRING_IDX;
    const StringState* ss = &st->strings[string_idx];
    if (out_len != ss->n_nodes * 3) return CE_ERR_DIM;
    for (int nidx = 0; nidx < ss->n_nodes; ++nidx) {
        int s = (nidx < ss->n_segs) ? nidx : ss->n_segs - 1;
        float d0 = ss->u[0][nidx];
        float d1 = ss->u[1][nidx];
        const float* base = ss->seg_xyz + nidx * 3;
        const float* n = ss->seg_normal + s * 3;
        const float* b = ss->seg_binormal + s * 3;
        out_xyz[nidx*3+0] = base[0] + d0*n[0] + d1*b[0];
        out_xyz[nidx*3+1] = base[1] + d0*n[1] + d1*b[1];
        out_xyz[nidx*3+2] = base[2] + d0*n[2] + d1*b[2];
    }
    return CE_OK;
}

/* ── Simple queries ───────────────────────────────────────────────────────── */

SK_API int   coevolver_get_n_strings (const AcousticCoEvolverState* st) { return st ? st->n_strings  : 0; }
SK_API int   coevolver_get_n_pickups (const AcousticCoEvolverState* st) { return st ? st->n_pickups  : 0; }
SK_API int   coevolver_get_n_mics    (const AcousticCoEvolverState* st) { return st ? st->n_mics     : 0; }
SK_API float coevolver_get_dt_audio  (const AcousticCoEvolverState* st) { return st ? st->dt_audio   : 0.0f; }
SK_API float coevolver_get_dt_fdtd   (const AcousticCoEvolverState* st) { return st ? st->dt_fdtd    : 0.0f; }

SK_API int coevolver_reset(AcousticCoEvolverState* st)
{
    if (!st) return CE_ERR_NULL;
    if (coevolver_busy(st)) return CE_ERR_BUSY;
    fdtd_reset(st->fdtd);
    for (int si = 0; si < st->n_strings; ++si) {
        StringState* ss = &st->strings[si];
        for (int p = 0; p < 2; ++p) {
            memset(ss->u[p],      0, ss->n_nodes * sizeof(float));
            memset(ss->u_prev[p], 0, ss->n_nodes * sizeof(float));
            memset(ss->ext_force[p], 0, ss->n_nodes * sizeof(float));
        }
        memset(ss->modal_re, 0, N_MODES * sizeof(float));
        memset(ss->modal_im, 0, N_MODES * sizeof(float));
        ss->saddle_w    = 0.0f;
        ss->saddle_w_prev = 0.0f;
        ss->bridge_w    = 0.0f;
        ss->bridge_w_prev = 0.0f;
        ss->nut_w       = 0.0f;
        ss->nut_w_prev  = 0.0f;
        ss->gamma_eff   = ss->gamma;
        ss->n_force_buf = 0;   /* discard any pending timed force */
    }
    st->gamma_scale = 1.0f;
    for (int pi = 0; pi < st->n_pickups; ++pi)
        memset(st->pickup_out[pi], 0, OUTPUT_BUF * sizeof(float));
    for (int mi = 0; mi < st->n_mics; ++mi)
        memset(st->mic_out[mi], 0, OUTPUT_BUF * sizeof(float));
    memset(st->pickup_accum, 0, st->n_pickups * sizeof(float));
    st->out_write_pos   = 0;
    st->samples_written = 0;
    st->bridge_vel_prev = 0.0f;
    st->neck_w = 0.0f;
    st->neck_w_prev = 0.0f;
    st->neck_plate_w_prev = 0.0f;
    st->modal_counter   = 0;
    st->pluck_cursor    = 0;
    return CE_OK;
}

/* ── Cut-cell face fractions passthrough ─────────────────────────────────── */

SK_API int coevolver_set_face_fractions(
    AcousticCoEvolverState* st,
    const float* vx_frac,
    const float* vy_frac,
    const float* vz_frac)
{
    if (!st || !st->fdtd) return CE_ERR_NULL;
    return fdtd_set_face_fractions(st->fdtd, vx_frac, vy_frac, vz_frac);
}

/* ── Damping scale (warm-up overdamping) ─────────────────────────────────── */

SK_API void coevolver_set_damping_scale(AcousticCoEvolverState* st, float scale)
{
    if (!st || coevolver_busy(st)) return;
    st->gamma_scale = (scale > 0.0f) ? scale : 1.0f;
}

/* ── Air envelope / emission surface ─────────────────────────────────────── */

SK_API int coevolver_get_surface_emission(
    const AcousticCoEvolverState* st,
    int n_surf, const float* surf_xyz, const float* surf_normals,
    float* out_P, float* out_vn)
{
    if (!st || !surf_xyz || !surf_normals || !out_P || !out_vn) return CE_ERR_NULL;

    /* Convert surface point world positions to grid space */
    float* grid_xyz = alloc_f(n_surf * 3);
    if (!grid_xyz) return CE_ERR_NULL;
    for (int s = 0; s < n_surf; ++s)
        world_to_grid(st, surf_xyz + s*3, grid_xyz + s*3);

    /* Delegate to FDTD surface emission (uses staggered V field, exact) */
    int rc = fdtd_get_surface_emission(st->fdtd, n_surf,
                                        grid_xyz, surf_normals,
                                        out_P, out_vn);
    free(grid_xyz);
    return (rc == FDTD_OK) ? CE_OK : CE_ERR_NULL;
}

SK_API int coevolver_sample_velocity(
    const AcousticCoEvolverState* st,
    int n_rec, const float* rec_xyz,
    float* out_vx, float* out_vy, float* out_vz)
{
    if (!st || !rec_xyz || !out_vx || !out_vy || !out_vz) return CE_ERR_NULL;

    float* grid_xyz = alloc_f(n_rec * 3);
    if (!grid_xyz) return CE_ERR_NULL;
    for (int r = 0; r < n_rec; ++r)
        world_to_grid(st, rec_xyz + r*3, grid_xyz + r*3);

    int rc = fdtd_sample_velocity(st->fdtd, n_rec, grid_xyz,
                                   out_vx, out_vy, out_vz);
    free(grid_xyz);
    return (rc == FDTD_OK) ? CE_OK : CE_ERR_NULL;
}
