/**
 * acoustic_fdtd.cpp — 3-D acoustic FDTD solver + Kirchhoff plate coupling.
 *
 * See acoustic_fdtd.h for physics documentation.
 */

#include "acoustic_fdtd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <Eigen/Core>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <algorithm>

/* ── Grid index helper ────────────────────────────────────────────────────── */

/* 3-D → flat: innermost axis = Z (body depth), then Y (length), then X (width) */
static inline int IDX3(int Ny, int Nz, int i, int j, int k)
{
    return k + Nz * (j + Ny * i);
}

/* 2-D plate → flat: j is Y (longitudinal), innermost i is X (lateral) */
static inline int IDX2(int Ny, int i, int j)
{
    return j + Ny * i;
}

/* Pressure-domain solidity.  FDTD_PLATE is a real acoustic barrier;
 * inactive plate-mask cells remain air and therefore form apertures
 * such as the soundhole. */
static inline int cell_is_pressure_solid(uint8_t c)
{
    return c == FDTD_WALL || c == FDTD_PLATE;
}

/* ── State structure ──────────────────────────────────────────────────────── */

struct AcousticFDTDState
{
    /* Grid dimensions */
    int  Nx, Ny, Nz;
    int  N;          /* Nx*Ny*Nz  */
    int  N_plate;    /* Nx*Ny     */
    int  plate_iz;   /* Z-index of the soundboard plate layer */

    /* Physical parameters */
    float dx, dt, c, rho_air;
    float C2;         /* (c*dt/dx)^2 — Courant number squared */

    /* Pressure fields — two time levels */
    float* P_curr;    /* P^n                  (N,)        */
    float* P_prev;    /* P^{n-1}              (N,)        */
    float* P_tmp;     /* scratch for new step (N,)        */

    /* Per-cell data */
    uint8_t* cell_type;  /* FDTD_AIR / FDTD_WALL / FDTD_PLATE / FDTD_PML (N,) */
    float*   pml_alpha;  /* per-cell PML viscous damping coefficient            (N,) */

    /* Kirchhoff plate fields — two time levels (Nx*Ny each) */
    float* w_curr;      /* w^n              */
    float* w_prev;      /* w^{n-1}          */
    float* w_tmp;       /* scratch          */
    uint8_t* plate_active; /* 1=inside guitar outline (Nx*Ny) */

    /* Plate physical constants */
    float plate_rho_h;       /* ρ_s·h (kg/m²)         */
    float plate_D;           /* bending stiffness D (N·m) */
    float plate_dt2_rho_h;   /* dt² / plate_rho_h     */
    float plate_biharm;      /* dt²·D / (plate_rho_h · dx⁴) */

    /* Rayleigh structural damping */
    float plate_alpha_M;     /* mass-proportional damping rate (s⁻¹)           */
    float plate_beta_K;      /* stiffness-proportional damping time (s)         */
    float plate_vel_damp;    /* 1 − α_M·dt : velocity-proxy multiplier          */
    float plate_bk_coeff;    /* β_K/dt · plate_biharm : stiffness damp per biharm step */

    /* Bridge source cells */
    int*   src_idx;      /* flat cell indices (n_src,) */
    float* src_wgt;      /* spatial kernel weights, normalised (n_src,) */
    int    n_src;

    /* Staggered particle velocity — co-evolved with pressure via Euler equation.
     * Vx at face (i+1/2,j,k): i∈[0,Nx-2], size (Nx-1)*Ny*Nz, idx k+Nz*(j+Ny*i)
     * Vy at face (i,j+1/2,k): j∈[0,Ny-2], size Nx*(Ny-1)*Nz, idx k+Nz*(j+(Ny-1)*i)
     * Vz at face (i,j,k+1/2): k∈[0,Nz-2], size Nx*Ny*(Nz-1), idx k+(Nz-1)*(j+Ny*i)
     */
    float* Vx;   int Nvx;
    float* Vy;   int Nvy;
    float* Vz;   int Nvz;

    /* Step counter */
    int step_count;

    /* Precomputed pressure damping multipliers (one value per cell, [N]).
     * Eliminates per-cell alpha/denom recomputation in the hot pressure loop. */
    float* P_prev_mul;  /* (1 − α·dt) / (1 + α·dt) */
    float* P_curr_mul;  /* 2           / (1 + α·dt) */
    float* P_lap_mul;   /* C2          / (1 + α·dt) */
    float* P_damp;      /* 1           / (1 + α·dt) — precomputed to eliminate per-cell division in pressure loop */

    /* Precomputed velocity face damping:  1 − 0.5·(α_L + α_R)·dt  per face.
     * Eliminates per-face pml_alpha lookup and multiply in the velocity loops. */
    float* Vx_damp;  /* [(Nx-1)*Ny*Nz] */
    float* Vy_damp;  /* [Nx*(Ny-1)*Nz] */
    float* Vz_damp;  /* [Nx*Ny*(Nz-1)] */

    /* Cut-cell open-area fractions.  Each value ∈ [0, 1]:
     *   0.0 — fully closed (solid wall; velocity zeroed in BC pass)
     *   1.0 — fully open  (interior or fully unoccluded face; default)
     *   (0,1) — partial opening at guitar body boundary
     * Applied in the pressure continuity update as a flux weight and in the
     * velocity BC pass to allow partial flux through cut faces.
     * Populated by fdtd_set_face_fractions() after create.  Defaults to 1.0
     * everywhere so the solver behaves identically to the staircase case when
     * fractions are not set. */
    float* Vx_frac;  /* [(Nx-1)*Ny*Nz] */
    float* Vy_frac;  /* [Nx*(Ny-1)*Nz] */
    float* Vz_frac;  /* [Nx*Ny*(Nz-1)] */

    /* Fast-interior mask: 1 for cells where all 6 neighbours are non-wall,
     * cell is not at domain boundary, and not plate-adjacent.  These cells
     * can be updated with direct index offsets, skipping all lambda / branch
     * overhead in the pressure stencil. */
    uint8_t* fast_interior;   /* [N] */

    /* Plate biharmonic precompute — eliminates W() lambda from plate_step. */
    int*     plate_active_idx;   /* [N_active_plate] flat plate indices         */
    int*     plate_above_idx;    /* [N_active_plate] flat 3-D idx above plate   */
    int*     plate_below_idx;    /* [N_active_plate] flat 3-D idx below plate   */
    int*     plate_biharm_idx13; /* [N_active_plate * 13] stencil flat indices  */
    int      N_active_plate;

    /* External body-force buffer for the plate equation.
     * Written by fdtd_inject_bridge_drive (N/m²), consumed and zeroed by plate_step. */
    float*   plate_ext_force;    /* [N_plate] force per unit area (N/m²)        */

    /* Stability check stride phase — rotated each check to cover all residues */
    int      stability_stride_phase;

    /* Compact pressure-step worklists — eliminate per-cell branching in hot loop.
     *
     * fast runs: contiguous k-sequences of fast-interior cells at fixed (i,j).
     *   run_base[r] = flat index of first cell in run r.
     *   run_len[r]  = number of cells.
     *   All 6 neighbours are non-wall and within bounds; direct offsets apply.
     *
     * slow: non-wall, non-fast, non-plate-adj.  Need Neumann ghost stencil.
     *
     * plate_adj: cells at k == plate_iz±1.  Same stencil options as above, but
     *   also receive the plate-velocity soft source.
     *   plate_adj_fast[n]: 1 = direct 6-point stencil; 0 = Neumann stencil.
     */
    int*    run_base;        /* [n_fast_runs] base flat idx of each fast run     */
    int*    run_len;         /* [n_fast_runs] cell count per run                 */
    int     n_fast_runs;

    int*    slow_idx;        /* [n_slow] flat indices of slow non-wall cells     */
    int     n_slow;

    int*    plate_adj_idx;   /* [n_plate_adj] flat 3-D indices                  */
    int*    plate_adj_pidx;  /* [n_plate_adj] flat plate indices                */
    int8_t* plate_adj_sign;  /* [n_plate_adj] +1 = above plate, -1 = below      */
    int*    plate_adj_fast;  /* [n_plate_adj] 1 = direct stencil, 0 = Neumann   */
    int     n_plate_adj;

    /* Persistent thread pool for parallel FDTD passes.
     * NULL on single-core hardware.  Owned by this state; freed by fdtd_destroy. */
    void*   pool;

    /* Precomputed Vz flat indices for plate velocity injection.
     * Eliminates per-cell integer division + IDX3 + solid-check in the hot BC loop.
     * plate_vz_below_vidx[n] == -1 when no valid Vz face below; same for above. */
    int*    plate_vz_below_vidx;   /* [N_active_plate] */
    int*    plate_vz_above_vidx;   /* [N_active_plate] */

    /* Compact closed-face worklists — replace O(Nvx+Nvy+Nvz) BC scan with
     * O(n_closed_*) unconditional zeroing writes.  Rebuilt by
     * build_closed_face_lists() whenever Vx_frac/Vy_frac/Vz_frac changes. */
    int*    closed_vx;  int n_closed_vx;
    int*    closed_vy;  int n_closed_vy;
    int*    closed_vz;  int n_closed_vz;

    /* Set to true by fdtd_set_face_fractions() when cut-cell fractions with
     * values other than 0 or 1 are present.  False at create time (all fracs
     * are exactly 1.0 for open faces; closed faces are already zeroed by the
     * BC worklist pass, so multiplying by their 0.0 frac is redundant). */
    bool    has_frac;
};

/* ── Allocation helpers ───────────────────────────────────────────────────── */

static float* alloc_float(int n)
{
    return (float*)calloc(n, sizeof(float));
}

static uint8_t* alloc_u8(int n)
{
    return (uint8_t*)calloc(n, sizeof(uint8_t));
}

/* ── PML sigma profile (polynomial grading) ─────────────────────────────── */

/* Returns a per-cell damping rate α (s⁻¹) for cells in the PML layer.
 * d is the depth into the PML (0 at interior edge, n_pml at outer wall).
 * σ_max is chosen so that a round-trip through the n_pml layer gives ~−60 dB
 * attenuation with dt steps:
 *   α_max · dt · n_pml  ≈ 3.45   (→ 60 dB) */
static float pml_sigma(int d, int n_pml, float dt)
{
    if (n_pml <= 0 || d <= 0) return 0.0f;
    float frac = (float)d / (float)n_pml;
    float sigma_max = 3.45f / (dt * (float)n_pml);
    return sigma_max * frac * frac * frac;  /* cubic grading */
}

/* ── Persistent thread pool ──────────────────────────────────────────────── *
 *
 * Generation-counter design: each dispatch() increments gen; workers wake
 * only when gen advances, execute their slice, then signal done.  The main
 * thread blocks in dispatch() until all workers have signalled.
 *
 * Workers are created once (in fdtd_create) and destroyed once (in
 * fdtd_destroy).  No threads are created per step or per sample.
 */
struct FDTDPool {
    int                          n;
    std::vector<std::thread>     workers;
    std::mutex                   mtx;
    std::condition_variable      cv_work;
    std::condition_variable      cv_done;
    std::function<void(int,int)> task;
    int total, chunk, done, gen;
    bool exiting;

    explicit FDTDPool(int nw)
        : n(nw), total(0), chunk(0), done(0), gen(-1), exiting(false)
    {
        workers.reserve(nw);
        for (int id = 0; id < nw; ++id)
            workers.emplace_back(&FDTDPool::loop, this, id);
    }

    ~FDTDPool() {
        { std::lock_guard<std::mutex> g(mtx); exiting = true; }
        cv_work.notify_all();
        for (auto& t : workers) t.join();
    }

    void dispatch(int items, std::function<void(int,int)> fn) {
        {
            std::lock_guard<std::mutex> g(mtx);
            task  = std::move(fn);
            total = items;
            chunk = (items + n - 1) / n;
            done  = 0;
            ++gen;
        }
        cv_work.notify_all();
        std::unique_lock<std::mutex> g(mtx);
        cv_done.wait(g, [this]{ return done == n; });
    }

private:
    void loop(int id) {
        int seen = -1;
        for (;;) {
            std::function<void(int,int)> fn;
            int s, e;
            {
                std::unique_lock<std::mutex> g(mtx);
                cv_work.wait(g, [&]{ return exiting || gen != seen; });
                if (exiting) return;
                seen = gen;
                fn   = task;
                s    = id * chunk;
                e    = std::min(s + chunk, total);
            }
            if (s < e) fn(s, e);
            { std::lock_guard<std::mutex> g(mtx); ++done; }
            cv_done.notify_one();
        }
    }
};

/* ── Closed-face worklist builder ────────────────────────────────────────── *
 * Scans Vx/Vy/Vz_frac once and records the indices of every fully-closed face
 * (frac == 0).  Called from fdtd_create and from fdtd_set_face_fractions so
 * the worklist is always consistent with the current frac arrays. */
static void build_closed_face_lists(AcousticFDTDState* st)
{
    free(st->closed_vx); free(st->closed_vy); free(st->closed_vz);
    st->closed_vx = st->closed_vy = st->closed_vz = NULL;
    st->n_closed_vx = st->n_closed_vy = st->n_closed_vz = 0;

    int ncvx = 0, ncvy = 0, ncvz = 0;
    for (int i = 0; i < st->Nvx; ++i) if (st->Vx_frac[i] == 0.0f) ++ncvx;
    for (int i = 0; i < st->Nvy; ++i) if (st->Vy_frac[i] == 0.0f) ++ncvy;
    for (int i = 0; i < st->Nvz; ++i) if (st->Vz_frac[i] == 0.0f) ++ncvz;

    st->closed_vx = (int*)malloc((ncvx > 0 ? ncvx : 1) * sizeof(int));
    st->closed_vy = (int*)malloc((ncvy > 0 ? ncvy : 1) * sizeof(int));
    st->closed_vz = (int*)malloc((ncvz > 0 ? ncvz : 1) * sizeof(int));
    if (!st->closed_vx || !st->closed_vy || !st->closed_vz) return;

    st->n_closed_vx = ncvx;
    st->n_closed_vy = ncvy;
    st->n_closed_vz = ncvz;

    int ci = 0;
    for (int i = 0; i < st->Nvx; ++i) if (st->Vx_frac[i] == 0.0f) st->closed_vx[ci++] = i;
    ci = 0;
    for (int i = 0; i < st->Nvy; ++i) if (st->Vy_frac[i] == 0.0f) st->closed_vy[ci++] = i;
    ci = 0;
    for (int i = 0; i < st->Nvz; ++i) if (st->Vz_frac[i] == 0.0f) st->closed_vz[ci++] = i;
}

/* ── Construction ─────────────────────────────────────────────────────────── */

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
    int     n_pml)
{
    if (Nx <= 0 || Ny <= 0 || Nz <= 0 || dx <= 0.0f || c <= 0.0f
            || cell_type == NULL || plate_active == NULL)
        return NULL;

    /* CFL-stable time step for 4th-order stencil in 3D.
     * 3D stability limit: CFL_1D ≤ 6/(7·√3) ≈ 0.495.
     * Safety factor 0.77 keeps us below that: 0.77/√3 ≈ 0.445. */
    float dt = 0.77f * dx / (c * 1.732050808f);
    float C2 = (c * dt / dx) * (c * dt / dx);

    int N       = Nx * Ny * Nz;
    int N_plate = Nx * Ny;

    AcousticFDTDState* st = (AcousticFDTDState*)calloc(1, sizeof(AcousticFDTDState));
    if (!st) return NULL;

    st->Nx = Nx;  st->Ny = Ny;  st->Nz = Nz;
    st->N  = N;   st->N_plate = N_plate;
    st->plate_iz = (plate_iz >= 0 && plate_iz < Nz) ? plate_iz : Nz / 2;
    st->dx = dx;  st->dt = dt;  st->c = c;  st->rho_air = rho_air;
    st->C2 = C2;

    /* Pressure fields */
    st->P_curr = alloc_float(N);
    st->P_prev = alloc_float(N);
    st->P_tmp  = alloc_float(N);
    if (!st->P_curr || !st->P_prev || !st->P_tmp) goto fail;

    /* Cell type and PML */
    st->cell_type = alloc_u8(N);
    st->pml_alpha = alloc_float(N);
    if (!st->cell_type || !st->pml_alpha) goto fail;

    memcpy(st->cell_type, cell_type, N * sizeof(uint8_t));

    /* Compute PML damping — one pass over the grid. */
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                /* Distance from each face, in cells */
                int di = (i < n_pml) ? (n_pml - i) : ((i >= Nx - n_pml) ? (i - (Nx - n_pml) + 1) : 0);
                int dj = (j < n_pml) ? (n_pml - j) : ((j >= Ny - n_pml) ? (j - (Ny - n_pml) + 1) : 0);
                int dk = (k < n_pml) ? (n_pml - k) : ((k >= Nz - n_pml) ? (k - (Nz - n_pml) + 1) : 0);
                int d  = (di > dj) ? ((di > dk) ? di : dk) : ((dj > dk) ? dj : dk);

                /* Only mark as PML if the cell itself is not a wall */
                float alpha = pml_sigma(d, n_pml, dt);
                int idx = IDX3(Ny, Nz, i, j, k);
                st->pml_alpha[idx] = alpha;

                /* Override cell type to PML if interior was AIR and d > 0 */
                if (st->cell_type[idx] == FDTD_AIR && alpha > 0.0f)
                    st->cell_type[idx] = FDTD_PML;
            }
        }
    }

    /* Plate fields */
    st->w_curr          = alloc_float(N_plate);
    st->w_prev          = alloc_float(N_plate);
    st->w_tmp           = alloc_float(N_plate);
    st->plate_active    = alloc_u8(N_plate);
    st->plate_ext_force = alloc_float(N_plate);
    if (!st->w_curr || !st->w_prev || !st->w_tmp || !st->plate_active
            || !st->plate_ext_force) goto fail;

    memcpy(st->plate_active, plate_active, N_plate * sizeof(uint8_t));

    /* Enforce the 2-D active-plate mask on the 3-D pressure grid.
     * Active soundboard cells are acoustic barriers at plate_iz; inactive
     * cells are air if the caller marked them as plate, which turns the
     * soundhole into an actual aperture instead of a masked render artifact. */
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int pidx = IDX2(Ny, i, j);
            int gidx = IDX3(Ny, Nz, i, j, st->plate_iz);
            if (st->plate_active[pidx]) {
                if (st->cell_type[gidx] == FDTD_AIR || st->cell_type[gidx] == FDTD_PML)
                    st->cell_type[gidx] = FDTD_PLATE;
            } else if (st->cell_type[gidx] == FDTD_PLATE) {
                st->cell_type[gidx] = FDTD_AIR;
            }
        }
    }

    for (int idx = 0; idx < N; ++idx) {
        if (cell_is_pressure_solid(st->cell_type[idx])) {
            st->P_curr[idx] = 0.0f;
            st->P_prev[idx] = 0.0f;
            st->P_tmp [idx] = 0.0f;
        }
    }

    /* Plate stability constants */
    st->plate_rho_h     = plate_mass_density;
    st->plate_D         = plate_stiffness_D;
    st->plate_dt2_rho_h = (dt * dt) / plate_mass_density;
    {
        float dx4 = dx * dx * dx * dx;
        st->plate_biharm = plate_stiffness_D * (dt * dt) / (plate_mass_density * dx4);
    }

    /* Rayleigh damping derived quantities */
    st->plate_alpha_M  = plate_alpha_M;
    st->plate_beta_K   = plate_beta_K;
    st->plate_vel_damp = 1.0f - plate_alpha_M * dt;
    st->plate_bk_coeff = (dt > 0.0f) ? (plate_beta_K / dt * st->plate_biharm) : 0.0f;

    /* Staggered velocity fields */
    st->Nvx = (Nx-1)*Ny*Nz;
    st->Nvy = Nx*(Ny-1)*Nz;
    st->Nvz = Nx*Ny*(Nz-1);
    st->Vx  = alloc_float(st->Nvx);
    st->Vy  = alloc_float(st->Nvy);
    st->Vz  = alloc_float(st->Nvz);
    if (!st->Vx || !st->Vy || !st->Vz) goto fail;

    st->step_count = 0;

    /* ── Precompute pressure damping arrays ── */
    st->P_prev_mul = alloc_float(N);
    st->P_curr_mul = alloc_float(N);
    st->P_lap_mul  = alloc_float(N);
    if (!st->P_prev_mul || !st->P_curr_mul || !st->P_lap_mul) goto fail;

    st->P_damp = alloc_float(N);
    if (!st->P_damp) goto fail;

    for (int idx = 0; idx < N; ++idx) {
        float alpha = st->pml_alpha[idx];
        float denom = 1.0f + alpha * dt;
        st->P_prev_mul[idx] = (1.0f - alpha * dt) / denom;
        st->P_curr_mul[idx] = 2.0f / denom;
        st->P_lap_mul [idx] = C2   / denom;
        st->P_damp    [idx] = 1.0f / denom;
    }

    /* ── Precompute velocity face damping arrays ── */
    st->Vx_damp = alloc_float(st->Nvx);
    st->Vy_damp = alloc_float(st->Nvy);
    st->Vz_damp = alloc_float(st->Nvz);
    if (!st->Vx_damp || !st->Vy_damp || !st->Vz_damp) goto fail;

    /* Cut-cell face fractions.
     * Default: 0.0 for faces adjacent to solid (WALL/PLATE) cells — these get
     * zeroed in the BC pass.  Exceptions:
     *   - Plate-layer Vz faces (k == plate_iz-1 or plate_iz) are kept at 1.0:
     *     plate injection sets them each step so fracs must not suppress them.
     *   - Interior air-air faces: 1.0 (no solid neighbour).
     * fdtd_set_face_fractions() overrides these for cut-cell body boundaries,
     * setting (0,1) for faces partially covered by the guitar outline. */
    st->Vx_frac = alloc_float(st->Nvx);
    st->Vy_frac = alloc_float(st->Nvy);
    st->Vz_frac = alloc_float(st->Nvz);
    if (!st->Vx_frac || !st->Vy_frac || !st->Vz_frac) goto fail;

    for (int i = 0; i < Nx-1; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k) {
                int vidx = k + Nz*(j + Ny*i);
                int L = IDX3(Ny, Nz, i,   j, k);
                int R = IDX3(Ny, Nz, i+1, j, k);
                st->Vx_frac[vidx] = (cell_is_pressure_solid(st->cell_type[L]) ||
                                     cell_is_pressure_solid(st->cell_type[R])) ? 0.0f : 1.0f;
            }

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny-1; ++j)
            for (int k = 0; k < Nz; ++k) {
                int vidx = k + Nz*(j + (Ny-1)*i);
                int L = IDX3(Ny, Nz, i, j,   k);
                int R = IDX3(Ny, Nz, i, j+1, k);
                st->Vy_frac[vidx] = (cell_is_pressure_solid(st->cell_type[L]) ||
                                     cell_is_pressure_solid(st->cell_type[R])) ? 0.0f : 1.0f;
            }

    {
        const int piz = st->plate_iz;
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz-1; ++k) {
                    int vidx = k + (Nz-1)*(j + Ny*i);
                    int L = IDX3(Ny, Nz, i, j, k);
                    int R = IDX3(Ny, Nz, i, j, k+1);
                    /* Plate-layer Vz faces: plate injection drives these each step;
                     * keep frac=1.0 so the divergence uses the full plate velocity. */
                    if (k == piz-1 || k == piz) {
                        st->Vz_frac[vidx] = 1.0f;
                    } else {
                        st->Vz_frac[vidx] = (cell_is_pressure_solid(st->cell_type[L]) ||
                                             cell_is_pressure_solid(st->cell_type[R])) ? 0.0f : 1.0f;
                    }
                }
    }

    for (int i = 0; i < Nx-1; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k) {
                int vidx = k + Nz*(j + Ny*i);
                int L    = IDX3(Ny, Nz, i,   j, k);
                int R    = IDX3(Ny, Nz, i+1, j, k);
                st->Vx_damp[vidx] = 1.0f - 0.5f * (st->pml_alpha[L] + st->pml_alpha[R]) * dt;
            }

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny-1; ++j)
            for (int k = 0; k < Nz; ++k) {
                int vidx = k + Nz*(j + (Ny-1)*i);
                int L    = IDX3(Ny, Nz, i, j,   k);
                int R    = IDX3(Ny, Nz, i, j+1, k);
                st->Vy_damp[vidx] = 1.0f - 0.5f * (st->pml_alpha[L] + st->pml_alpha[R]) * dt;
            }

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz-1; ++k) {
                int vidx = k + (Nz-1)*(j + Ny*i);
                int L    = IDX3(Ny, Nz, i, j, k);
                int R    = IDX3(Ny, Nz, i, j, k+1);
                st->Vz_damp[vidx] = 1.0f - 0.5f * (st->pml_alpha[L] + st->pml_alpha[R]) * dt;
            }

    /* ── Precompute fast-interior mask ── */
    st->fast_interior = alloc_u8(N);
    if (!st->fast_interior) goto fail;
    {
        /* A cell is fast-interior if:
         *  - not a wall (pressure update required)
         *  - not at grid boundary (all 6 neighbours exist)
         *  - all 6 neighbours are non-wall
         *  - not plate-adjacent (k != plate_iz ± 1) */
        for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
        for (int k = 0; k < Nz; ++k) {
            int idx = IDX3(Ny, Nz, i, j, k);
            if (cell_is_pressure_solid(st->cell_type[idx])) continue;
            if (i == 0 || i == Nx-1 || j == 0 || j == Ny-1
                       || k == 0 || k == Nz-1) continue;
            if (k == plate_iz+1 || k == plate_iz-1) continue;
            int nb6[6] = {
                IDX3(Ny,Nz,i+1,j,k), IDX3(Ny,Nz,i-1,j,k),
                IDX3(Ny,Nz,i,j+1,k), IDX3(Ny,Nz,i,j-1,k),
                IDX3(Ny,Nz,i,j,k+1), IDX3(Ny,Nz,i,j,k-1)
            };
            int ok = 1;
            for (int n = 0; n < 6; ++n)
                if (cell_is_pressure_solid(st->cell_type[nb6[n]])) { ok = 0; break; }
            if (ok) st->fast_interior[idx] = 1;
        }
    }

    /* ── Build compact pressure-step worklists from fast-interior mask ── *
     *                                                                       *
     * Two-pass: count then fill.  Uses st->plate_iz (the validated value).  */
    {
        const uint8_t* ct = st->cell_type;
        const int piz = st->plate_iz;

        /* Count pass */
        int ns = 0, npa = 0, nr = 0;
        for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            int prev_fast = 0;
            for (int k = 0; k < Nz; ++k) {
                int idx = IDX3(Ny, Nz, i, j, k);
                if (cell_is_pressure_solid(ct[idx])) { prev_fast = 0; continue; }
                int is_pa = (k == piz+1 || k == piz-1);
                if (is_pa)                    { ++npa; prev_fast = 0; continue; }
                if (st->fast_interior[idx])   { if (!prev_fast) ++nr; prev_fast = 1; }
                else                          { ++ns; prev_fast = 0; }
            }
        }

        st->run_base       = (int*)   malloc(nr  > 0 ? nr  * sizeof(int)    : sizeof(int));
        st->run_len        = (int*)   malloc(nr  > 0 ? nr  * sizeof(int)    : sizeof(int));
        st->slow_idx       = (int*)   malloc(ns  > 0 ? ns  * sizeof(int)    : sizeof(int));
        st->plate_adj_idx  = (int*)   malloc(npa > 0 ? npa * sizeof(int)    : sizeof(int));
        st->plate_adj_pidx = (int*)   malloc(npa > 0 ? npa * sizeof(int)    : sizeof(int));
        st->plate_adj_sign = (int8_t*)malloc(npa > 0 ? npa * sizeof(int8_t) : sizeof(int8_t));
        st->plate_adj_fast = (int*)   malloc(npa > 0 ? npa * sizeof(int)    : sizeof(int));
        if (!st->run_base || !st->run_len || !st->slow_idx ||
            !st->plate_adj_idx || !st->plate_adj_pidx ||
            !st->plate_adj_sign || !st->plate_adj_fast) goto fail;
        st->n_fast_runs = nr;
        st->n_slow      = ns;
        st->n_plate_adj = npa;

        /* Fill pass */
        int si2 = 0, pai = 0, ri = 0;
        for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            int in_run = 0, rlen = 0;
            for (int k = 0; k < Nz; ++k) {
                int idx = IDX3(Ny, Nz, i, j, k);
                if (cell_is_pressure_solid(ct[idx])) {
                    if (in_run) { st->run_len[ri-1] = rlen; in_run = 0; rlen = 0; }
                    continue;
                }
                int is_pa = (k == piz+1 || k == piz-1);
                if (is_pa) {
                    if (in_run) { st->run_len[ri-1] = rlen; in_run = 0; rlen = 0; }
                    st->plate_adj_idx[pai]  = idx;
                    st->plate_adj_pidx[pai] = IDX2(Ny, i, j);
                    st->plate_adj_sign[pai] = (k == piz+1) ? 1 : -1;
                    /* Direct stencil OK if not at boundary and no wall neighbours */
                    int fpa = (i>0 && i<Nx-1 && j>0 && j<Ny-1 && k>0 && k<Nz-1);
                    if (fpa) {
                        int nb6[6] = { IDX3(Ny,Nz,i+1,j,k), IDX3(Ny,Nz,i-1,j,k),
                                       IDX3(Ny,Nz,i,j+1,k), IDX3(Ny,Nz,i,j-1,k),
                                       IDX3(Ny,Nz,i,j,k+1), IDX3(Ny,Nz,i,j,k-1) };
                        for (int n = 0; n < 6; ++n)
                            if (cell_is_pressure_solid(ct[nb6[n]])) { fpa = 0; break; }
                    }
                    st->plate_adj_fast[pai] = fpa;
                    ++pai;
                    continue;
                }
                if (st->fast_interior[idx]) {
                    if (!in_run) { st->run_base[ri++] = idx; in_run = 1; rlen = 0; }
                    ++rlen;
                } else {
                    if (in_run) { st->run_len[ri-1] = rlen; in_run = 0; rlen = 0; }
                    st->slow_idx[si2++] = idx;
                }
            }
            if (in_run) { st->run_len[ri-1] = rlen; in_run = 0; }
        }
    }

    /* ── Precompute plate biharmonic stencil ── */
    {
        /* Stencil offsets (di, dj) for the 13-point biharmonic */
        static const int DI13[13] = { 0, 1,-1, 0, 0, 1, 1,-1,-1, 2,-2, 0, 0};
        static const int DJ13[13] = { 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 2,-2};

        /* Count active plate cells */
        int N_act = 0;
        for (int p = 0; p < N_plate; ++p)
            if (plate_active[p]) ++N_act;
        st->N_active_plate = N_act;

        st->plate_active_idx   = (int*)malloc(N_act * sizeof(int));
        st->plate_above_idx    = (int*)malloc(N_act * sizeof(int));
        st->plate_below_idx    = (int*)malloc(N_act * sizeof(int));
        st->plate_biharm_idx13 = (int*)malloc(N_act * 13 * sizeof(int));
        if (!st->plate_active_idx || !st->plate_above_idx ||
            !st->plate_below_idx  || !st->plate_biharm_idx13) goto fail;

        int n = 0;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                int pidx = IDX2(Ny, i, j);
                if (!plate_active[pidx]) continue;
                st->plate_active_idx[n] = pidx;

                /* Pressure cell above/below plate */
                st->plate_above_idx[n] = (st->plate_iz + 1 < Nz)
                                         ? IDX3(Ny, Nz, i, j, st->plate_iz + 1) : -1;
                st->plate_below_idx[n] = (st->plate_iz - 1 >= 0)
                                         ? IDX3(Ny, Nz, i, j, st->plate_iz - 1) : -1;

                /* 13-tap biharmonic stencil on the *active plate graph*.
                 * Samples that would leave the guitar outline or enter the soundhole
                 * are reflected to the current cell.  That gives a local zero-slope
                 * free-edge approximation at irregular boundaries instead of imposing
                 * a rectangular-grid ghost reflection unrelated to the actual body. */
                for (int t = 0; t < 13; ++t) {
                    int ii = i + DI13[t];
                    int jj = j + DJ13[t];
                    int use_self = 0;
                    if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny) {
                        use_self = 1;
                    } else {
                        int qidx = IDX2(Ny, ii, jj);
                        if (!plate_active[qidx]) use_self = 1;
                    }
                    if (use_self) { ii = i; jj = j; }
                    st->plate_biharm_idx13[n*13 + t] = IDX2(Ny, ii, jj);
                }
                ++n;
            }
        }
    }

    /* ── Precompute plate Vz injection indices ── *
     * Eliminates pidx/Ny, IDX3, and cell_is_pressure_solid from the hot BC pass. */
    {
        int N_act = st->N_active_plate;
        st->plate_vz_below_vidx = (int*)malloc((N_act > 0 ? N_act : 1) * sizeof(int));
        st->plate_vz_above_vidx = (int*)malloc((N_act > 0 ? N_act : 1) * sizeof(int));
        if (!st->plate_vz_below_vidx || !st->plate_vz_above_vidx) goto fail;
        const int piz = st->plate_iz;
        for (int n = 0; n < N_act; ++n) {
            int pidx = st->plate_active_idx[n];
            int ii   = pidx / Ny;
            int jj   = pidx % Ny;
            if (piz > 0) {
                int b = IDX3(Ny, Nz, ii, jj, piz-1);
                st->plate_vz_below_vidx[n] = cell_is_pressure_solid(st->cell_type[b])
                                              ? -1 : (piz-1) + (Nz-1)*(jj + Ny*ii);
            } else {
                st->plate_vz_below_vidx[n] = -1;
            }
            if (piz < Nz-1) {
                int a = IDX3(Ny, Nz, ii, jj, piz+1);
                st->plate_vz_above_vidx[n] = cell_is_pressure_solid(st->cell_type[a])
                                              ? -1 : piz + (Nz-1)*(jj + Ny*ii);
            } else {
                st->plate_vz_above_vidx[n] = -1;
            }
        }
    }

    /* ── Precompute closed-face BC worklists ── */
    build_closed_face_lists(st);
    if (!st->closed_vx || !st->closed_vy || !st->closed_vz) goto fail;

    {
        int nw = (int)std::thread::hardware_concurrency();
        st->pool = (nw > 1) ? new FDTDPool(nw) : NULL;
    }

    return st;

fail:
    fdtd_destroy(st);
    return NULL;
}

SK_API void fdtd_destroy(AcousticFDTDState* st)
{
    if (!st) return;
    free(st->P_curr);     free(st->P_prev);     free(st->P_tmp);
    free(st->cell_type);  free(st->pml_alpha);
    free(st->P_prev_mul); free(st->P_curr_mul); free(st->P_lap_mul);
    free(st->P_damp);
    free(st->w_curr);     free(st->w_prev);     free(st->w_tmp);
    free(st->plate_active); free(st->plate_ext_force);
    free(st->src_idx);    free(st->src_wgt);
    free(st->Vx);         free(st->Vy);         free(st->Vz);
    free(st->Vx_damp);    free(st->Vy_damp);    free(st->Vz_damp);
    free(st->Vx_frac);    free(st->Vy_frac);    free(st->Vz_frac);
    free(st->fast_interior);
    free(st->plate_active_idx); free(st->plate_above_idx); free(st->plate_below_idx);
    free(st->plate_biharm_idx13);
    free(st->run_base);    free(st->run_len);
    free(st->slow_idx);
    free(st->plate_adj_idx); free(st->plate_adj_pidx);
    free(st->plate_adj_sign); free(st->plate_adj_fast);
    free(st->plate_vz_below_vidx); free(st->plate_vz_above_vidx);
    free(st->closed_vx); free(st->closed_vy); free(st->closed_vz);
    if (st->pool) { delete static_cast<FDTDPool*>(st->pool); st->pool = NULL; }
    free(st);
}

/* ── Bridge source registration ───────────────────────────────────────────── */

SK_API int fdtd_set_bridge_sources(
    AcousticFDTDState* st,
    int          n_cells,
    const int*   cell_indices,
    const float* weights)
{
    if (!st || n_cells <= 0 || !cell_indices || !weights)
        return FDTD_ERR_NULL;

    int*   new_idx = (int*)  malloc(n_cells * sizeof(int));
    float* new_wgt = (float*)malloc(n_cells * sizeof(float));
    if (!new_idx || !new_wgt) {
        free(new_idx);
        free(new_wgt);
        return FDTD_ERR_NULL;
    }

    float wsum = 0.0f;
    for (int i = 0; i < n_cells; ++i) wsum += fabsf(weights[i]);
    float wscale = (wsum > 1e-12f) ? 1.0f / wsum : 1.0f;

    for (int i = 0; i < n_cells; ++i) {
        new_idx[i] = cell_indices[i];
        new_wgt[i] = weights[i] * wscale;
    }

    free(st->src_idx);
    free(st->src_wgt);
    st->src_idx = new_idx;
    st->src_wgt = new_wgt;
    st->n_src   = n_cells;
    return FDTD_OK;
}

/* ── Source injection ─────────────────────────────────────────────────────── */

/* Robust bridge pressure deposition.  Bridge source kernels may be built on
 * the geometric soundboard plane.  After FDTD_PLATE is treated as a real
 * pressure barrier, those geometric cells can be solid.  Never deposit pressure
 * into a solid cell: it will be copied forward by the leapfrog boundary rule and
 * can accumulate until the stability guard trips.  Instead, deposit into the
 * coincident air cell, or split to the nearest face-adjacent air cells. */
static inline int deposit_pressure_open(AcousticFDTDState* st, int idx, float amp)
{
    if (!st || idx < 0 || idx >= st->N || amp == 0.0f) return 0;
    if (!cell_is_pressure_solid(st->cell_type[idx])) {
        st->P_curr[idx] += amp;
        return 1;
    }

    int k   = idx % st->Nz;
    int tmp = idx / st->Nz;
    int j   = tmp % st->Ny;
    int i   = tmp / st->Ny;

    int nb[6];
    int n = 0;
    if (i + 1 < st->Nx) nb[n++] = IDX3(st->Ny, st->Nz, i+1, j,   k);
    if (i - 1 >= 0)     nb[n++] = IDX3(st->Ny, st->Nz, i-1, j,   k);
    if (j + 1 < st->Ny) nb[n++] = IDX3(st->Ny, st->Nz, i,   j+1, k);
    if (j - 1 >= 0)     nb[n++] = IDX3(st->Ny, st->Nz, i,   j-1, k);
    if (k + 1 < st->Nz) nb[n++] = IDX3(st->Ny, st->Nz, i,   j,   k+1);
    if (k - 1 >= 0)     nb[n++] = IDX3(st->Ny, st->Nz, i,   j,   k-1);

    int open_count = 0;
    for (int q = 0; q < n; ++q)
        if (!cell_is_pressure_solid(st->cell_type[nb[q]])) ++open_count;
    if (open_count <= 0) return 0;

    float share = amp / (float)open_count;
    for (int q = 0; q < n; ++q)
        if (!cell_is_pressure_solid(st->cell_type[nb[q]]))
            st->P_curr[nb[q]] += share;
    return open_count;
}

SK_API int fdtd_inject_bridge(
    AcousticFDTDState* st,
    float signal_val,
    float signal_ddt,
    float force_scale)
{
    (void)st;
    (void)signal_val;
    (void)signal_ddt;
    (void)force_scale;
    return FDTD_ERR_UNSTABLE;
}

/* ── 4th-order staggered gradient coefficients (Fornberg) ────────────────── *
 * grad_h(P) at face (i+1/2) = O4_C1*(P[i+1]-P[i]) + O4_C2*(P[i+2]-P[i-1]) */
static const float O4_C1 =  9.0f / 8.0f;
static const float O4_C2 = -1.0f / 24.0f;

/* Per-cell 4th-order divergence of the staggered velocity field.
 * Used by both fast-run and slow worklist paths in pressure_step. */
static inline float compute_div_v(
    const float* Vx, const float* Vy, const float* Vz,
    const float* Vxf, const float* Vyf, const float* Vzf,
    int Nx, int Ny, int Nz,
    int i, int j, int k)
{
#define VXF(ii) (Vx[k + Nz*(j + Ny*(ii))] * Vxf[k + Nz*(j + Ny*(ii))])
#define VYF(jj) (Vy[k + Nz*((jj) + (Ny-1)*i)] * Vyf[k + Nz*((jj) + (Ny-1)*i)])
#define VZF(kk) (Vz[(kk) + (Nz-1)*(j + Ny*i)] * Vzf[(kk) + (Nz-1)*(j + Ny*i)])

    float vxp  = (i < Nx-1) ? VXF(i)   : 0.0f;
    float vxm  = (i > 0)    ? VXF(i-1) : 0.0f;
    float vxpp = (i < Nx-2) ? VXF(i+1) : 0.0f;
    float vxmm = (i > 1)    ? VXF(i-2) : 0.0f;

    float vyp  = (j < Ny-1) ? VYF(j)   : 0.0f;
    float vym  = (j > 0)    ? VYF(j-1) : 0.0f;
    float vypp = (j < Ny-2) ? VYF(j+1) : 0.0f;
    float vymm = (j > 1)    ? VYF(j-2) : 0.0f;

    float vzp  = (k < Nz-1) ? VZF(k)   : 0.0f;
    float vzm  = (k > 0)    ? VZF(k-1) : 0.0f;
    float vzpp = (k < Nz-2) ? VZF(k+1) : 0.0f;
    float vzmm = (k > 1)    ? VZF(k-2) : 0.0f;

#undef VXF
#undef VYF
#undef VZF

    return O4_C1 * ((vxp-vxm) + (vyp-vym) + (vzp-vzm))
         + O4_C2 * ((vxpp-vxmm) + (vypp-vymm) + (vzpp-vzmm));
}

/* Frac-free variant: called when has_frac==false (all open-face fracs are 1.0;
 * closed faces were already zeroed by the BC worklist pass).
 * Removes 12 float multiplications per cell relative to compute_div_v. */
static inline float compute_div_v_nofrac(
    const float* Vx, const float* Vy, const float* Vz,
    int Nx, int Ny, int Nz,
    int i, int j, int k)
{
    float vxp  = (i < Nx-1) ? Vx[k + Nz*(j + Ny* i      )] : 0.0f;
    float vxm  = (i > 0)    ? Vx[k + Nz*(j + Ny*(i-1)   )] : 0.0f;
    float vxpp = (i < Nx-2) ? Vx[k + Nz*(j + Ny*(i+1)   )] : 0.0f;
    float vxmm = (i > 1)    ? Vx[k + Nz*(j + Ny*(i-2)   )] : 0.0f;

    float vyp  = (j < Ny-1) ? Vy[k + Nz*( j    + (Ny-1)*i)] : 0.0f;
    float vym  = (j > 0)    ? Vy[k + Nz*((j-1) + (Ny-1)*i)] : 0.0f;
    float vypp = (j < Ny-2) ? Vy[k + Nz*((j+1) + (Ny-1)*i)] : 0.0f;
    float vymm = (j > 1)    ? Vy[k + Nz*((j-2) + (Ny-1)*i)] : 0.0f;

    float vzp  = (k < Nz-1) ? Vz[ k    + (Nz-1)*(j + Ny*i)] : 0.0f;
    float vzm  = (k > 0)    ? Vz[(k-1) + (Nz-1)*(j + Ny*i)] : 0.0f;
    float vzpp = (k < Nz-2) ? Vz[(k+1) + (Nz-1)*(j + Ny*i)] : 0.0f;
    float vzmm = (k > 1)    ? Vz[(k-2) + (Nz-1)*(j + Ny*i)] : 0.0f;

    return O4_C1 * ((vxp-vxm) + (vyp-vym) + (vzp-vzm))
         + O4_C2 * ((vxpp-vxmm) + (vypp-vymm) + (vzpp-vzmm));
}

/* ── Kirchhoff plate update ───────────────────────────────────────────────── */

/* 13-point biharmonic stencil (∇⁴w) with simply-supported ghost cells.
 * Simply supported: w=0 at boundary → ghost extension w[-1]=−w[1] (odd). */
static float biharmonic(const float* w, int Nx, int Ny, int i, int j)
{
    /* Clamped ghost cell function: odd-reflection across boundary gives w=0 BC. */
    auto W = [&](int ii, int jj) -> float {
        /* clamp to [0, Nx) with odd reflection */
        int ci = ii, cj = jj;
        if (ci < 0)   ci = -ci - 1;   /* reflect: W[-1]=−W[0]=0 since W[0]=0 BC */
        if (ci >= Nx) ci = 2*Nx-1-ci;
        if (cj < 0)   cj = -cj - 1;
        if (cj >= Ny) cj = 2*Ny-1-cj;
        /* clip any remaining out-of-range to boundary (returns 0 for inactive) */
        if (ci < 0) ci = 0;
        if (ci >= Nx) ci = Nx-1;
        if (cj < 0) cj = 0;
        if (cj >= Ny) cj = Ny-1;
        return w[IDX2(Ny, ci, cj)];
    };

    float val =
        20.0f * W(i,   j  )
        - 8.0f * (W(i+1, j  ) + W(i-1, j  ) + W(i,   j+1) + W(i,   j-1))
        + 2.0f * (W(i+1, j+1) + W(i+1, j-1) + W(i-1, j+1) + W(i-1, j-1))
        +        (W(i+2, j  ) + W(i-2, j  ) + W(i,   j+2) + W(i,   j-2));
    return val;
}

static int plate_step(AcousticFDTDState* st)
{
    static const float W13[13] = {20,-8,-8,-8,-8, 2, 2, 2, 2, 1, 1, 1, 1};

    float a  = st->plate_dt2_rho_h;
    float bh = st->plate_biharm;
    const float* w  = st->w_curr;
    const float* wp = st->w_prev;
    float*       wt = st->w_tmp;

    /* Active cells are all written below; inactive positions in wt come from
     * the old w_prev and are never read in any stencil (precomputed stencil
     * redirects inactive neighbours to self), so no clear is needed. */

    for (int n = 0; n < st->N_active_plate; ++n) {
        int pidx = st->plate_active_idx[n];

        /* 13-tap biharmonic via precomputed indices — no lambda, no bounds check. */
        const int* nbrs = st->plate_biharm_idx13 + n * 13;
        float biharm = 0.0f;
        for (int t = 0; t < 13; ++t)
            biharm += W13[t] * w[nbrs[t]];

        /* Acoustic pressure load. */
        float P_above = (st->plate_above_idx[n] >= 0)
                        ? st->P_curr[st->plate_above_idx[n]] : 0.0f;
        float P_below = (st->plate_below_idx[n] >= 0)
                        ? st->P_curr[st->plate_below_idx[n]] : 0.0f;

        /* Bridge body force (N/m²) written by fdtd_inject_bridge_drive.
         * Treated identically to a net pressure load on the plate face — the
         * bridge string drives the plate, not the acoustic field directly. */
        float f_bridge = st->plate_ext_force ? st->plate_ext_force[pidx] : 0.0f;

        /* Rayleigh structural damping.
         * vel_damp = 1 − α_M·dt  replaces the old frequency-independent 0.9990 constant.
         * bk_coeff * (biharm − biharm_prev) adds stiffness-proportional loss:
         *   −β_K/dt · (dt²·D/ρh·dx⁴) · ∇⁴(w − wp)  ≈  −β_K/dt · bh · Δbiharm */
        float biharm_prev = 0.0f;
        for (int t = 0; t < 13; ++t)
            biharm_prev += W13[t] * wp[nbrs[t]];

        float vel_proxy = w[pidx] - wp[pidx];
        float w_new = w[pidx] + st->plate_vel_damp * vel_proxy
                    + a  * ((P_below - P_above) + f_bridge)
                    - bh * biharm
                    - st->plate_bk_coeff * (biharm - biharm_prev);

        if (!isfinite(w_new) || fabsf(w_new) > 0.02f)
            return FDTD_ERR_UNSTABLE;
        wt[pidx] = w_new;
    }

    /* Consume the bridge force buffer so it doesn't accumulate across steps. */
    if (st->plate_ext_force)
        memset(st->plate_ext_force, 0, st->N_plate * sizeof(float));

    float* tmp = st->w_prev;
    st->w_prev = st->w_curr;
    st->w_curr = st->w_tmp;
    st->w_tmp  = tmp;
    return FDTD_OK;
}

/* ── Cut-cell face-fraction injection ────────────────────────────────────────
 *
 * Sets the per-face open-area fractions used by the pressure continuity update
 * and the velocity BC pass.  Call once after fdtd_create; safe to call again
 * to update geometry without rebuilding the full solver.
 *
 * Arrays must have exactly (Nx-1)*Ny*Nz, Nx*(Ny-1)*Nz, Nx*Ny*(Nz-1) elements.
 * Pass NULL for any component to leave it unchanged.
 */
SK_API int fdtd_set_face_fractions(
    AcousticFDTDState* st,
    const float* vx_frac,
    const float* vy_frac,
    const float* vz_frac)
{
    if (!st) return FDTD_ERR_NULL;
    if (vx_frac) memcpy(st->Vx_frac, vx_frac, st->Nvx * sizeof(float));
    if (vy_frac) memcpy(st->Vy_frac, vy_frac, st->Nvy * sizeof(float));
    if (vz_frac) memcpy(st->Vz_frac, vz_frac, st->Nvz * sizeof(float));
    if (vx_frac || vy_frac || vz_frac) st->has_frac = true;
    build_closed_face_lists(st);
    return FDTD_OK;
}

/* ── Main FDTD pressure update ───────────────────────────────────────────── */

static int pressure_step(AcousticFDTDState* st)
{
    int Nx = st->Nx, Ny = st->Ny, Nz = st->Nz;
    float dt = st->dt;
    const float inv_dx = 1.0f / st->dx;
    const uint8_t* ct   = st->cell_type;

    const float* Pn = st->P_curr;
    const float vel_coeff = dt / (st->rho_air * st->dx);

    /* Fornberg 4th-order staggered gradient coefficients — defined at file scope. */

    /* Momentum update on staggered faces.  Vx, Vy, Vz write to disjoint arrays
     * and read only from the shared Pn snapshot, so each component is fully
     * independent.  All three are folded into a single pool dispatch that
     * divides the outer i-loop across workers. */
    FDTDPool* pool = static_cast<FDTDPool*>(st->pool);

    auto run_velocity = [&](int i0, int i1) {
        /* Vx: face (i+1/2, j, k), i ∈ [0, Nx-2] */
        for (int i = i0; i < std::min(i1, Nx-1); ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz; ++k) {
                    int vidx = k + Nz*(j + Ny*i);
                    int L = IDX3(Ny,Nz,i,j,k), R = IDX3(Ny,Nz,i+1,j,k);
                    float grad;
                    if (i > 0 && i < Nx-2) {
                        grad = O4_C1*(Pn[R] - Pn[L])
                             + O4_C2*(Pn[IDX3(Ny,Nz,i+2,j,k)] - Pn[IDX3(Ny,Nz,i-1,j,k)]);
                    } else {
                        grad = Pn[R] - Pn[L];
                    }
                    st->Vx[vidx] = (st->Vx[vidx] - vel_coeff * grad) * st->Vx_damp[vidx];
                }
        /* Vy: face (i, j+1/2, k), j ∈ [0, Ny-2] */
        for (int i = i0; i < i1; ++i)
            for (int j = 0; j < Ny-1; ++j)
                for (int k = 0; k < Nz; ++k) {
                    int vidx = k + Nz*(j + (Ny-1)*i);
                    int L = IDX3(Ny,Nz,i,j,k), R = IDX3(Ny,Nz,i,j+1,k);
                    float grad;
                    if (j > 0 && j < Ny-2) {
                        grad = O4_C1*(Pn[R] - Pn[L])
                             + O4_C2*(Pn[IDX3(Ny,Nz,i,j+2,k)] - Pn[IDX3(Ny,Nz,i,j-1,k)]);
                    } else {
                        grad = Pn[R] - Pn[L];
                    }
                    st->Vy[vidx] = (st->Vy[vidx] - vel_coeff * grad) * st->Vy_damp[vidx];
                }
        /* Vz: face (i, j, k+1/2), k ∈ [0, Nz-2] */
        for (int i = i0; i < i1; ++i)
            for (int j = 0; j < Ny; ++j)
                for (int k = 0; k < Nz-1; ++k) {
                    int vidx = k + (Nz-1)*(j + Ny*i);
                    int L = IDX3(Ny,Nz,i,j,k), R = IDX3(Ny,Nz,i,j,k+1);
                    float grad;
                    if (k > 0 && k < Nz-2) {
                        grad = O4_C1*(Pn[R] - Pn[L])
                             + O4_C2*(Pn[IDX3(Ny,Nz,i,j,k+2)] - Pn[IDX3(Ny,Nz,i,j,k-1)]);
                    } else {
                        grad = Pn[R] - Pn[L];
                    }
                    st->Vz[vidx] = (st->Vz[vidx] - vel_coeff * grad) * st->Vz_damp[vidx];
                }
    };

    if (pool) pool->dispatch(Nx, run_velocity);
    else      run_velocity(0, Nx);

    /* Face BCs: zero all fully-closed faces using precomputed worklist.
     * Replaces a full O(Nvx+Nvy+Nvz) scan with O(n_closed) unconditional writes. */
    for (int r = 0; r < st->n_closed_vx; ++r) st->Vx[st->closed_vx[r]] = 0.0f;
    for (int r = 0; r < st->n_closed_vy; ++r) st->Vy[st->closed_vy[r]] = 0.0f;
    for (int r = 0; r < st->n_closed_vz; ++r) st->Vz[st->closed_vz[r]] = 0.0f;

    /* Plate velocity injection via precomputed Vz indices.
     * Eliminates per-cell integer division, IDX3, and solid-cell checks. */
    for (int n = 0; n < st->N_active_plate; ++n) {
        int pidx   = st->plate_active_idx[n];
        float v_pl = (st->w_curr[pidx] - st->w_prev[pidx]) * (1.0f / dt);
        if (st->plate_vz_below_vidx[n] >= 0) st->Vz[st->plate_vz_below_vidx[n]] = v_pl;
        if (st->plate_vz_above_vidx[n] >= 0) st->Vz[st->plate_vz_above_vidx[n]] = v_pl;
    }

    /* Continuity update: three worklist passes eliminate per-cell solid-cell
     * branching and skip the full-grid memset (solid positions stay zero by
     * invariant; inactive positions are never read).
     *
     * Pass 1 — fast interior runs: contiguous k-sequences where all 6 neighbours
     *   are guaranteed non-wall and in-bounds.  Compiler can hoist the x/y
     *   4th-order guards out of the k loop since i,j are fixed per run.
     * Pass 2 — slow cells: near-boundary or near-wall, individual cells.
     * Pass 3 — plate-adjacent cells: same divergence formula; plate velocity is
     *   already injected into Vz above, so no special treatment is needed here.
     */
    const float bulk_coeff = st->rho_air * st->c * st->c * dt * inv_dx;
    const float* pdamp = st->P_damp;
    float*       Ptmp  = st->P_tmp;
    const float* Vx  = st->Vx,  *Vy  = st->Vy,  *Vz  = st->Vz;
    const float* Vxf = st->Vx_frac, *Vyf = st->Vy_frac, *Vzf = st->Vz_frac;
    const bool   hf  = st->has_frac;

    /* Fast-run loop: branch on hf outside the k-offset loop so the compiler
     * sees a uniform inner loop without a per-cell predicate. */
    auto run_fast_pressure = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            int base = st->run_base[r];
            int len  = st->run_len[r];
            int ij   = base / Nz;
            int i    = ij / Ny;
            int j    = ij % Ny;
            int k0   = base % Nz;
            if (hf) {
                for (int ofs = 0; ofs < len; ++ofs) {
                    int idx = base + ofs;
                    int k   = k0 + ofs;
                    float dv = compute_div_v(Vx,Vy,Vz,Vxf,Vyf,Vzf,Nx,Ny,Nz,i,j,k);
                    Ptmp[idx] = (Pn[idx] - bulk_coeff * dv) * pdamp[idx];
                }
            } else {
                for (int ofs = 0; ofs < len; ++ofs) {
                    int idx = base + ofs;
                    int k   = k0 + ofs;
                    float dv = compute_div_v_nofrac(Vx,Vy,Vz,Nx,Ny,Nz,i,j,k);
                    Ptmp[idx] = (Pn[idx] - bulk_coeff * dv) * pdamp[idx];
                }
            }
        }
    };

    if (pool) pool->dispatch(st->n_fast_runs, run_fast_pressure);
    else      run_fast_pressure(0, st->n_fast_runs);

    for (int s = 0; s < st->n_slow; ++s) {
        int idx = st->slow_idx[s];
        int k   = idx % Nz;
        int ij  = idx / Nz;
        int i   = ij / Ny;
        int j   = ij % Ny;
        float dv = hf ? compute_div_v(Vx,Vy,Vz,Vxf,Vyf,Vzf,Nx,Ny,Nz,i,j,k)
                      : compute_div_v_nofrac(Vx,Vy,Vz,Nx,Ny,Nz,i,j,k);
        Ptmp[idx] = (Pn[idx] - bulk_coeff * dv) * pdamp[idx];
    }

    for (int p = 0; p < st->n_plate_adj; ++p) {
        int idx = st->plate_adj_idx[p];
        int k   = idx % Nz;
        int ij  = idx / Nz;
        int i   = ij / Ny;
        int j   = ij % Ny;
        float dv = hf ? compute_div_v(Vx,Vy,Vz,Vxf,Vyf,Vzf,Nx,Ny,Nz,i,j,k)
                      : compute_div_v_nofrac(Vx,Vy,Vz,Nx,Ny,Nz,i,j,k);
        Ptmp[idx] = (Pn[idx] - bulk_coeff * dv) * pdamp[idx];
    }

    float* tmp = st->P_prev;
    st->P_prev = st->P_curr;
    st->P_curr = st->P_tmp;
    st->P_tmp  = tmp;
    return FDTD_OK;
}

/* ── Stability check ──────────────────────────────────────────────────────── */

static int check_stable(AcousticFDTDState* st)
{
    /* Sample every SAMPLE_STRIDE-th cell, rotating the start offset each call
     * so successive checks cover different residue classes — avoids missing a
     * localized divergence that always falls between two sampled positions. */
    static const int SAMPLE_STRIDE = 1013;  /* prime */
    int start = st->stability_stride_phase;
    st->stability_stride_phase = (start + 1) % SAMPLE_STRIDE;
    for (int i = start; i < st->N; i += SAMPLE_STRIDE) {
        float p = st->P_curr[i];
        if (!isfinite(p) || p > 2e3f || p < -2e3f)
            return FDTD_ERR_UNSTABLE;
    }
    return FDTD_OK;
}

/* ── Public step ──────────────────────────────────────────────────────────── */

SK_API int fdtd_step(AcousticFDTDState* st, int n_steps)
{
    if (!st) return FDTD_ERR_NULL;
    if (n_steps <= 0) return FDTD_OK;

    for (int s = 0; s < n_steps; ++s) {
        int rc_plate = plate_step(st);
        if (rc_plate != FDTD_OK) return rc_plate;
        int rc = pressure_step(st);
        if (rc != FDTD_OK) return rc;
        ++st->step_count;

        /* Stability check every 256 steps. */
        if ((st->step_count & 0xFF) == 0) {
            if (check_stable(st) != FDTD_OK) return FDTD_ERR_UNSTABLE;
        }
    }
    return FDTD_OK;
}

/* ── Query ────────────────────────────────────────────────────────────────── */

SK_API int   fdtd_get_step_count(const AcousticFDTDState* st) { return st ? st->step_count : 0; }
SK_API float fdtd_get_dt        (const AcousticFDTDState* st) { return st ? st->dt         : 0.0f; }

SK_API int fdtd_reset(AcousticFDTDState* st)
{
    if (!st) return FDTD_ERR_NULL;
    memset(st->P_curr,  0, st->N       * sizeof(float));
    memset(st->P_prev,  0, st->N       * sizeof(float));
    memset(st->w_curr,        0, st->N_plate * sizeof(float));
    memset(st->w_prev,        0, st->N_plate * sizeof(float));
    memset(st->plate_ext_force, 0, st->N_plate * sizeof(float));
    memset(st->Vx,      0, st->Nvx     * sizeof(float));
    memset(st->Vy,      0, st->Nvy     * sizeof(float));
    memset(st->Vz,      0, st->Nvz     * sizeof(float));
    st->step_count = 0;
    return FDTD_OK;
}

/* ── Field extraction ─────────────────────────────────────────────────────── */

SK_API int fdtd_get_pressure_field(
    const AcousticFDTDState* st, float* out, int out_len)
{
    if (!st || !out) return FDTD_ERR_NULL;
    if (out_len != st->N) return FDTD_ERR_DIM;
    memcpy(out, st->P_curr, st->N * sizeof(float));
    return FDTD_OK;
}

SK_API int fdtd_get_plate_displacement(
    const AcousticFDTDState* st, float* out, int out_len)
{
    if (!st || !out) return FDTD_ERR_NULL;
    if (out_len != st->N_plate) return FDTD_ERR_DIM;
    memcpy(out, st->w_curr, st->N_plate * sizeof(float));
    return FDTD_OK;
}

SK_API int fdtd_sample_plate_displacement_batch(
    const AcousticFDTDState* st,
    int          n_points,
    const int*   saddle_idx4,
    const float* saddle_wgt4,
    float*       out_w)
{
    if (!st || !saddle_idx4 || !saddle_wgt4 || !out_w) return FDTD_ERR_NULL;
    const float* w       = st->w_curr;
    const int    N_plate = st->N_plate;
    for (int si = 0; si < n_points; ++si) {
        const int*   idx = saddle_idx4 + si * 4;
        const float* wgt = saddle_wgt4 + si * 4;
        /* Four-tap bilinear gather — indices are pre-clamped at create time. */
        out_w[si] = wgt[0] * w[idx[0]]
                  + wgt[1] * w[idx[1]]
                  + wgt[2] * w[idx[2]]
                  + wgt[3] * w[idx[3]];
        (void)N_plate; /* bounds guaranteed at create-time precompute */
    }
    return FDTD_OK;
}

SK_API int fdtd_sample_pressure(
    const AcousticFDTDState* st,
    int          n_rec,
    const float* rec_xyz,
    float*       out_p)
{
    if (!st || !rec_xyz || !out_p) return FDTD_ERR_NULL;

    int Nx = st->Nx, Ny = st->Ny, Nz = st->Nz;

    for (int r = 0; r < n_rec; ++r) {
        float gx = rec_xyz[r * 3 + 0];
        float gy = rec_xyz[r * 3 + 1];
        float gz = rec_xyz[r * 3 + 2];

        /* Clamp to [0, N-1] per axis */
        if (gx < 0.0f) gx = 0.0f;  if (gx > (float)(Nx - 1)) gx = (float)(Nx - 1);
        if (gy < 0.0f) gy = 0.0f;  if (gy > (float)(Ny - 1)) gy = (float)(Ny - 1);
        if (gz < 0.0f) gz = 0.0f;  if (gz > (float)(Nz - 1)) gz = (float)(Nz - 1);

        int i0 = (int)gx,  i1 = (i0 < Nx-1) ? i0+1 : i0;
        int j0 = (int)gy,  j1 = (j0 < Ny-1) ? j0+1 : j0;
        int k0 = (int)gz,  k1 = (k0 < Nz-1) ? k0+1 : k0;

        float fx = gx - (float)i0;
        float fy = gy - (float)j0;
        float fz = gz - (float)k0;

        float p = 0.0f;
        p += (1-fx)*(1-fy)*(1-fz) * st->P_curr[IDX3(Ny,Nz,i0,j0,k0)];
        p += (  fx)*(1-fy)*(1-fz) * st->P_curr[IDX3(Ny,Nz,i1,j0,k0)];
        p += (1-fx)*(  fy)*(1-fz) * st->P_curr[IDX3(Ny,Nz,i0,j1,k0)];
        p += (  fx)*(  fy)*(1-fz) * st->P_curr[IDX3(Ny,Nz,i1,j1,k0)];
        p += (1-fx)*(1-fy)*(  fz) * st->P_curr[IDX3(Ny,Nz,i0,j0,k1)];
        p += (  fx)*(1-fy)*(  fz) * st->P_curr[IDX3(Ny,Nz,i1,j0,k1)];
        p += (1-fx)*(  fy)*(  fz) * st->P_curr[IDX3(Ny,Nz,i0,j1,k1)];
        p += (  fx)*(  fy)*(  fz) * st->P_curr[IDX3(Ny,Nz,i1,j1,k1)];
        out_p[r] = p;
    }
    return FDTD_OK;
}

/* ── Velocity field queries ───────────────────────────────────────────────── */

SK_API void fdtd_get_velocity_dims(
    const AcousticFDTDState* st, int* Nvx, int* Nvy, int* Nvz)
{
    if (!st) { *Nvx=*Nvy=*Nvz=0; return; }
    *Nvx = st->Nvx;
    *Nvy = st->Nvy;
    *Nvz = st->Nvz;
}

SK_API int fdtd_get_velocity_x(const AcousticFDTDState* st, float* out, int out_len)
{
    if (!st || !out) return FDTD_ERR_NULL;
    if (out_len != st->Nvx) return FDTD_ERR_DIM;
    memcpy(out, st->Vx, st->Nvx * sizeof(float));
    return FDTD_OK;
}

SK_API int fdtd_get_velocity_y(const AcousticFDTDState* st, float* out, int out_len)
{
    if (!st || !out) return FDTD_ERR_NULL;
    if (out_len != st->Nvy) return FDTD_ERR_DIM;
    memcpy(out, st->Vy, st->Nvy * sizeof(float));
    return FDTD_OK;
}

SK_API int fdtd_get_velocity_z(const AcousticFDTDState* st, float* out, int out_len)
{
    if (!st || !out) return FDTD_ERR_NULL;
    if (out_len != st->Nvz) return FDTD_ERR_DIM;
    memcpy(out, st->Vz, st->Nvz * sizeof(float));
    return FDTD_OK;
}

/* ── Staggered trilinear interpolation helpers ───────────────────────────── */

/* Sample staggered Vx at physical cell-centred position (gx, gy, gz).
 * Vx is defined at (i+1/2, j, k), so we shift gx by −0.5 for interpolation. */
static float sample_Vx(const AcousticFDTDState* st, float gx, float gy, float gz)
{
    int Nx = st->Nx, Ny = st->Ny, Nz = st->Nz;
    float sx = gx - 0.5f;
    if (sx < 0.0f) sx = 0.0f;
    if (sx > (float)(Nx - 2)) sx = (float)(Nx - 2);
    if (gy < 0.0f) gy = 0.0f;  if (gy > (float)(Ny - 1)) gy = (float)(Ny - 1);
    if (gz < 0.0f) gz = 0.0f;  if (gz > (float)(Nz - 1)) gz = (float)(Nz - 1);

    int i0 = (int)sx, i1 = (i0 < Nx-2) ? i0+1 : i0;
    int j0 = (int)gy, j1 = (j0 < Ny-1) ? j0+1 : j0;
    int k0 = (int)gz, k1 = (k0 < Nz-1) ? k0+1 : k0;
    float fx = sx-(float)i0, fy = gy-(float)j0, fz = gz-(float)k0;

    /* Vx index: k + Nz*(j + Ny*i)  for i ∈ [0, Nx-2] */
#define VXI(i,j,k) ((k) + Nz*((j) + Ny*(i)))
    float v = 0.0f;
    v += (1-fx)*(1-fy)*(1-fz)*st->Vx[VXI(i0,j0,k0)];
    v += (  fx)*(1-fy)*(1-fz)*st->Vx[VXI(i1,j0,k0)];
    v += (1-fx)*(  fy)*(1-fz)*st->Vx[VXI(i0,j1,k0)];
    v += (  fx)*(  fy)*(1-fz)*st->Vx[VXI(i1,j1,k0)];
    v += (1-fx)*(1-fy)*(  fz)*st->Vx[VXI(i0,j0,k1)];
    v += (  fx)*(1-fy)*(  fz)*st->Vx[VXI(i1,j0,k1)];
    v += (1-fx)*(  fy)*(  fz)*st->Vx[VXI(i0,j1,k1)];
    v += (  fx)*(  fy)*(  fz)*st->Vx[VXI(i1,j1,k1)];
#undef VXI
    return v;
}

static float sample_Vy(const AcousticFDTDState* st, float gx, float gy, float gz)
{
    int Nx = st->Nx, Ny = st->Ny, Nz = st->Nz;
    float sy = gy - 0.5f;
    if (gx < 0.0f) gx = 0.0f;  if (gx > (float)(Nx - 1)) gx = (float)(Nx - 1);
    if (sy < 0.0f) sy = 0.0f;
    if (sy > (float)(Ny - 2)) sy = (float)(Ny - 2);
    if (gz < 0.0f) gz = 0.0f;  if (gz > (float)(Nz - 1)) gz = (float)(Nz - 1);

    int i0 = (int)gx, i1 = (i0 < Nx-1) ? i0+1 : i0;
    int j0 = (int)sy, j1 = (j0 < Ny-2) ? j0+1 : j0;
    int k0 = (int)gz, k1 = (k0 < Nz-1) ? k0+1 : k0;
    float fx = gx-(float)i0, fy = sy-(float)j0, fz = gz-(float)k0;

    /* Vy index: k + Nz*(j + (Ny-1)*i)  for j ∈ [0, Ny-2] */
#define VYI(i,j,k) ((k) + Nz*((j) + (Ny-1)*(i)))
    float v = 0.0f;
    v += (1-fx)*(1-fy)*(1-fz)*st->Vy[VYI(i0,j0,k0)];
    v += (  fx)*(1-fy)*(1-fz)*st->Vy[VYI(i1,j0,k0)];
    v += (1-fx)*(  fy)*(1-fz)*st->Vy[VYI(i0,j1,k0)];
    v += (  fx)*(  fy)*(1-fz)*st->Vy[VYI(i1,j1,k0)];
    v += (1-fx)*(1-fy)*(  fz)*st->Vy[VYI(i0,j0,k1)];
    v += (  fx)*(1-fy)*(  fz)*st->Vy[VYI(i1,j0,k1)];
    v += (1-fx)*(  fy)*(  fz)*st->Vy[VYI(i0,j1,k1)];
    v += (  fx)*(  fy)*(  fz)*st->Vy[VYI(i1,j1,k1)];
#undef VYI
    return v;
}

static float sample_Vz(const AcousticFDTDState* st, float gx, float gy, float gz)
{
    int Nx = st->Nx, Ny = st->Ny, Nz = st->Nz;
    float sz = gz - 0.5f;
    if (gx < 0.0f) gx = 0.0f;  if (gx > (float)(Nx - 1)) gx = (float)(Nx - 1);
    if (gy < 0.0f) gy = 0.0f;  if (gy > (float)(Ny - 1)) gy = (float)(Ny - 1);
    if (sz < 0.0f) sz = 0.0f;
    if (sz > (float)(Nz - 2)) sz = (float)(Nz - 2);

    int i0 = (int)gx, i1 = (i0 < Nx-1) ? i0+1 : i0;
    int j0 = (int)gy, j1 = (j0 < Ny-1) ? j0+1 : j0;
    int k0 = (int)sz, k1 = (k0 < Nz-2) ? k0+1 : k0;
    float fx = gx-(float)i0, fy = gy-(float)j0, fz = sz-(float)k0;

    /* Vz index: k + (Nz-1)*(j + Ny*i)  for k ∈ [0, Nz-2] */
#define VZI(i,j,k) ((k) + (Nz-1)*((j) + Ny*(i)))
    float v = 0.0f;
    v += (1-fx)*(1-fy)*(1-fz)*st->Vz[VZI(i0,j0,k0)];
    v += (  fx)*(1-fy)*(1-fz)*st->Vz[VZI(i1,j0,k0)];
    v += (1-fx)*(  fy)*(1-fz)*st->Vz[VZI(i0,j1,k0)];
    v += (  fx)*(  fy)*(1-fz)*st->Vz[VZI(i1,j1,k0)];
    v += (1-fx)*(1-fy)*(  fz)*st->Vz[VZI(i0,j0,k1)];
    v += (  fx)*(1-fy)*(  fz)*st->Vz[VZI(i1,j0,k1)];
    v += (1-fx)*(  fy)*(  fz)*st->Vz[VZI(i0,j1,k1)];
    v += (  fx)*(  fy)*(  fz)*st->Vz[VZI(i1,j1,k1)];
#undef VZI
    return v;
}

SK_API int fdtd_sample_velocity(
    const AcousticFDTDState* st,
    int n_rec, const float* rec_xyz,
    float* out_vx, float* out_vy, float* out_vz)
{
    if (!st || !rec_xyz || !out_vx || !out_vy || !out_vz) return FDTD_ERR_NULL;
    for (int r = 0; r < n_rec; ++r) {
        float gx = rec_xyz[r*3+0];
        float gy = rec_xyz[r*3+1];
        float gz = rec_xyz[r*3+2];
        out_vx[r] = sample_Vx(st, gx, gy, gz);
        out_vy[r] = sample_Vy(st, gx, gy, gz);
        out_vz[r] = sample_Vz(st, gx, gy, gz);
    }
    return FDTD_OK;
}

SK_API int fdtd_get_surface_emission(
    const AcousticFDTDState* st,
    int n_surf, const float* surf_xyz, const float* surf_normals,
    float* out_P, float* out_vn)
{
    if (!st || !surf_xyz || !surf_normals || !out_P || !out_vn) return FDTD_ERR_NULL;
    for (int s = 0; s < n_surf; ++s) {
        float gx = surf_xyz[s*3+0];
        float gy = surf_xyz[s*3+1];
        float gz = surf_xyz[s*3+2];
        /* Pressure at point */
        float rec[3] = {gx, gy, gz};
        fdtd_sample_pressure(st, 1, rec, &out_P[s]);
        /* Normal velocity = v · n̂ */
        float vx = sample_Vx(st, gx, gy, gz);
        float vy = sample_Vy(st, gx, gy, gz);
        float vz = sample_Vz(st, gx, gy, gz);
        float nx = surf_normals[s*3+0];
        float ny = surf_normals[s*3+1];
        float nz = surf_normals[s*3+2];
        out_vn[s] = vx*nx + vy*ny + vz*nz;
    }
    return FDTD_OK;
}

/* ── Precomputed sampler construction ──────────────────────────────────── */

SK_API int fdtd_precompute_pressure_samplers(
    const AcousticFDTDState* st,
    int n_rec, const float* rec_xyz,
    int* out_idx8, float* out_wgt8)
{
    if (!st || !rec_xyz || !out_idx8 || !out_wgt8) return FDTD_ERR_NULL;
    int Nx = st->Nx, Ny = st->Ny, Nz = st->Nz;

    for (int r = 0; r < n_rec; ++r) {
        float gx = rec_xyz[r*3+0], gy = rec_xyz[r*3+1], gz = rec_xyz[r*3+2];
        if (gx < 0.0f) gx = 0.0f;  if (gx > (float)(Nx-1)) gx = (float)(Nx-1);
        if (gy < 0.0f) gy = 0.0f;  if (gy > (float)(Ny-1)) gy = (float)(Ny-1);
        if (gz < 0.0f) gz = 0.0f;  if (gz > (float)(Nz-1)) gz = (float)(Nz-1);
        int i0=(int)gx, i1=(i0<Nx-1)?i0+1:i0;
        int j0=(int)gy, j1=(j0<Ny-1)?j0+1:j0;
        int k0=(int)gz, k1=(k0<Nz-1)?k0+1:k0;
        float fx=gx-(float)i0, fy=gy-(float)j0, fz=gz-(float)k0;
        int*   idx = out_idx8 + r*8;
        float* wgt = out_wgt8 + r*8;
        idx[0]=IDX3(Ny,Nz,i0,j0,k0); wgt[0]=(1-fx)*(1-fy)*(1-fz);
        idx[1]=IDX3(Ny,Nz,i1,j0,k0); wgt[1]=(  fx)*(1-fy)*(1-fz);
        idx[2]=IDX3(Ny,Nz,i0,j1,k0); wgt[2]=(1-fx)*(  fy)*(1-fz);
        idx[3]=IDX3(Ny,Nz,i1,j1,k0); wgt[3]=(  fx)*(  fy)*(1-fz);
        idx[4]=IDX3(Ny,Nz,i0,j0,k1); wgt[4]=(1-fx)*(1-fy)*(  fz);
        idx[5]=IDX3(Ny,Nz,i1,j0,k1); wgt[5]=(  fx)*(1-fy)*(  fz);
        idx[6]=IDX3(Ny,Nz,i0,j1,k1); wgt[6]=(1-fx)*(  fy)*(  fz);
        idx[7]=IDX3(Ny,Nz,i1,j1,k1); wgt[7]=(  fx)*(  fy)*(  fz);
    }
    return FDTD_OK;
}

/* Helper: build one set of 8-tap staggered indices for a velocity component.
 * shift_axis: 0=x, 1=y, 2=z — the axis with the 0.5-cell offset.
 * The V arrays have size (N_dim-1) along the shifted axis. */
static void _precompute_v_samplers_one(
    int Nx, int Ny, int Nz,
    int shift_axis,
    int n_rec, const float* rec_xyz,
    int* out_idx8, float* out_wgt8)
{
    /* V index formula (c=shift axis, others normal):
     *   Vx:  k + Nz*(j + Ny*i),   i in [0,Nx-2]
     *   Vy:  k + Nz*(j + (Ny-1)*i), j in [0,Ny-2]
     *   Vz:  k + (Nz-1)*(j + Ny*i), k in [0,Nz-2] */
    int Nx1 = (shift_axis==0) ? Nx-1 : Nx;
    int Ny1 = (shift_axis==1) ? Ny-1 : Ny;
    int Nz1 = (shift_axis==2) ? Nz-1 : Nz;

    for (int r = 0; r < n_rec; ++r) {
        float gx = rec_xyz[r*3+0] - (shift_axis==0 ? 0.5f : 0.0f);
        float gy = rec_xyz[r*3+1] - (shift_axis==1 ? 0.5f : 0.0f);
        float gz = rec_xyz[r*3+2] - (shift_axis==2 ? 0.5f : 0.0f);
        if (gx < 0.0f) gx = 0.0f;  if (gx > (float)(Nx1-1)) gx = (float)(Nx1-1);
        if (gy < 0.0f) gy = 0.0f;  if (gy > (float)(Ny1-1)) gy = (float)(Ny1-1);
        if (gz < 0.0f) gz = 0.0f;  if (gz > (float)(Nz1-1)) gz = (float)(Nz1-1);
        int i0=(int)gx, i1=(i0<Nx1-1)?i0+1:i0;
        int j0=(int)gy, j1=(j0<Ny1-1)?j0+1:j0;
        int k0=(int)gz, k1=(k0<Nz1-1)?k0+1:k0;
        float fx=gx-(float)i0, fy=gy-(float)j0, fz=gz-(float)k0;

#define VIDX(ia,ja,ka) ((shift_axis==2) \
    ? ((ka) + Nz1*((ja) + Ny*(ia))) \
    : ((shift_axis==1) \
        ? ((ka) + Nz*((ja) + Ny1*(ia))) \
        : ((ka) + Nz*((ja) + Ny*(ia)))))

        int*   idx = out_idx8 + r*8;
        float* wgt = out_wgt8 + r*8;
        idx[0]=VIDX(i0,j0,k0); wgt[0]=(1-fx)*(1-fy)*(1-fz);
        idx[1]=VIDX(i1,j0,k0); wgt[1]=(  fx)*(1-fy)*(1-fz);
        idx[2]=VIDX(i0,j1,k0); wgt[2]=(1-fx)*(  fy)*(1-fz);
        idx[3]=VIDX(i1,j1,k0); wgt[3]=(  fx)*(  fy)*(1-fz);
        idx[4]=VIDX(i0,j0,k1); wgt[4]=(1-fx)*(1-fy)*(  fz);
        idx[5]=VIDX(i1,j0,k1); wgt[5]=(  fx)*(1-fy)*(  fz);
        idx[6]=VIDX(i0,j1,k1); wgt[6]=(1-fx)*(  fy)*(  fz);
        idx[7]=VIDX(i1,j1,k1); wgt[7]=(  fx)*(  fy)*(  fz);
#undef VIDX
    }
}

SK_API int fdtd_precompute_velocity_samplers(
    const AcousticFDTDState* st,
    int n_rec, const float* rec_xyz,
    int* out_vx_idx8, float* out_vx_wgt8,
    int* out_vy_idx8, float* out_vy_wgt8,
    int* out_vz_idx8, float* out_vz_wgt8)
{
    if (!st || !rec_xyz) return FDTD_ERR_NULL;
    if (!out_vx_idx8||!out_vx_wgt8||!out_vy_idx8||!out_vy_wgt8
        ||!out_vz_idx8||!out_vz_wgt8) return FDTD_ERR_NULL;
    _precompute_v_samplers_one(st->Nx,st->Ny,st->Nz,0,n_rec,rec_xyz,out_vx_idx8,out_vx_wgt8);
    _precompute_v_samplers_one(st->Nx,st->Ny,st->Nz,1,n_rec,rec_xyz,out_vy_idx8,out_vy_wgt8);
    _precompute_v_samplers_one(st->Nx,st->Ny,st->Nz,2,n_rec,rec_xyz,out_vz_idx8,out_vz_wgt8);
    return FDTD_OK;
}

SK_API int fdtd_sample_pressure_precomputed(
    const AcousticFDTDState* st,
    int n_rec, const int* idx8, const float* wgt8, float* out_p)
{
    if (!st || !idx8 || !wgt8 || !out_p) return FDTD_ERR_NULL;
    const float* P = st->P_curr;
    for (int r = 0; r < n_rec; ++r) {
        const int*   id = idx8 + r*8;
        const float* w  = wgt8 + r*8;
        out_p[r] = w[0]*P[id[0]] + w[1]*P[id[1]] + w[2]*P[id[2]] + w[3]*P[id[3]]
                 + w[4]*P[id[4]] + w[5]*P[id[5]] + w[6]*P[id[6]] + w[7]*P[id[7]];
    }
    return FDTD_OK;
}

SK_API int fdtd_sample_velocity_precomputed(
    const AcousticFDTDState* st,
    int n_rec,
    const int* vx_idx8, const float* vx_wgt8,
    const int* vy_idx8, const float* vy_wgt8,
    const int* vz_idx8, const float* vz_wgt8,
    float* out_vx, float* out_vy, float* out_vz)
{
    if (!st || !vx_idx8||!vx_wgt8||!vy_idx8||!vy_wgt8||!vz_idx8||!vz_wgt8) return FDTD_ERR_NULL;
    if (!out_vx || !out_vy || !out_vz) return FDTD_ERR_NULL;
    for (int r = 0; r < n_rec; ++r) {
        const int*   xi = vx_idx8 + r*8; const float* xw = vx_wgt8 + r*8;
        const int*   yi = vy_idx8 + r*8; const float* yw = vy_wgt8 + r*8;
        const int*   zi = vz_idx8 + r*8; const float* zw = vz_wgt8 + r*8;
        float vx=0,vy=0,vz=0;
        for(int t=0;t<8;++t){ vx+=xw[t]*st->Vx[xi[t]]; vy+=yw[t]*st->Vy[yi[t]]; vz+=zw[t]*st->Vz[zi[t]]; }
        out_vx[r]=vx; out_vy[r]=vy; out_vz[r]=vz;
    }
    return FDTD_OK;
}

SK_API int fdtd_inject_bridge_drive(
    AcousticFDTDState* st,
    const float* cell_drives,
    float        force_scale)
{
    if (!st) return FDTD_ERR_NULL;
    if (!st->src_idx || st->n_src == 0) return FDTD_OK;  /* no bridge — no-op */
    if (!cell_drives) return FDTD_ERR_NULL;
    if (!st->plate_ext_force) return FDTD_ERR_NULL;

    /* Convert per-bridge-cell force (N) to force-per-area (N/m²) and accumulate
     * into the plate equation's external forcing buffer.  plate_step reads this
     * buffer as an additional pressure-equivalent load and zeroes it afterward,
     * keeping the bridge→plate→air coupling chain strictly causal and passive.
     *
     * src_idx[i] = k + Nz*(j + Ny*i) so src_idx[i]/Nz = j + Ny*i = plate pidx. */
    float inv_dx2 = 1.0f / (st->dx * st->dx);
    for (int i = 0; i < st->n_src; ++i) {
        int pidx = st->src_idx[i] / st->Nz;
        st->plate_ext_force[pidx] += force_scale * cell_drives[i] * inv_dx2;
    }
    return FDTD_OK;
}

SK_API int fdtd_inject_plate_force(
    AcousticFDTDState* st,
    int          n_plate,
    const int*   plate_indices,
    const float* weights,
    float        total_force_N)
{
    if (!st || !plate_indices || !weights) return FDTD_ERR_NULL;
    if (!st->plate_ext_force) return FDTD_ERR_NULL;
    if (n_plate <= 0 || total_force_N == 0.0f) return FDTD_OK;

    float inv_dx2 = 1.0f / (st->dx * st->dx);
    for (int i = 0; i < n_plate; ++i) {
        int pidx = plate_indices[i];
        if (pidx < 0 || pidx >= st->N_plate) continue;
        if (!st->plate_active[pidx]) continue;
        st->plate_ext_force[pidx] += total_force_N * weights[i] * inv_dx2;
    }
    return FDTD_OK;
}

SK_API int fdtd_sample_plate_weighted(
    const AcousticFDTDState* st,
    int          n_plate,
    const int*   plate_indices,
    const float* weights,
    float*       out_w)
{
    if (!st || !plate_indices || !weights || !out_w) return FDTD_ERR_NULL;
    float w = 0.0f;
    for (int i = 0; i < n_plate; ++i) {
        int pidx = plate_indices[i];
        if (pidx < 0 || pidx >= st->N_plate) continue;
        w += st->w_curr[pidx] * weights[i];
    }
    *out_w = w;
    return FDTD_OK;
}
