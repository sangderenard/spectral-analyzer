/**
 * acoustic_amr.cpp — Eigen-backed AMR acoustic pressure stepper.
 */

#include "acoustic_amr.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

/* Per-active-plate-node mapping to AMR faces above and below. */
struct PlateAMRNodeMap {
    std::vector<int>   face_above;      /* AMR face indices above this plate node */
    std::vector<float> area_above;      /* area weights (normalized, sum≈1) */
    std::vector<int>   cell_above_side; /* acoustic cell on AIR side of each face_above[k] */
    int                cell_above;      /* nearest AIR cell above (fallback, may be -1) */
    std::vector<int>   face_below;      /* AMR face indices below */
    std::vector<float> area_below;      /* area weights (normalized, sum≈1) */
    std::vector<int>   cell_below_side; /* acoustic cell on AIR side of each face_below[k] */
    int                cell_below;      /* nearest AIR cell below (fallback, may be -1) */
};

/* Per-mic precomputed pressure stencil (inverse-distance weighted AMR cells). */
struct MicPressureSampler {
    std::vector<int>   cell_idx;
    std::vector<float> cell_wgt;  /* sum to 1 */
};

/* Per-mic precomputed velocity stencil.
 * Each entry stores three Cartesian projection weights:
 *   out_vx += wgt_x[k] * v_face[face_idx[k]]
 *   out_vy += wgt_y[k] * ...
 *   out_vz += wgt_z[k] * ...
 * so the coevolver can compute dot(v, axis) the same way for both backends. */
struct MicVelocitySampler {
    std::vector<int>   face_idx;
    std::vector<float> wgt_x;   /* face_area * face_nx / norm */
    std::vector<float> wgt_y;
    std::vector<float> wgt_z;
};

struct AcousticAMRState {
    /* ── AMR grid ──────────────────────────────────────────────────── */
    int n_cells = 0;
    int n_faces = 0;
    double c = 343.0;
    double rho_air = 1.21;
    double dt = 0.0;
    int step_count = 0;

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> centers;
    Eigen::VectorXd volumes;
    Eigen::VectorXd open_vol;
    Eigen::VectorXi cell_type;
    Eigen::VectorXi face_neg;
    Eigen::VectorXi face_pos;
    Eigen::VectorXd face_area;
    Eigen::VectorXd face_open;
    Eigen::VectorXd face_distance;
    Eigen::VectorXf pressure;
    Eigen::VectorXf velocity;
    std::vector<int> acoustic_cells;

    /* ── Precomputed hot-loop buffers (set in amr_create / amr_setup_plate) ── */
    Eigen::VectorXf face_inv_dist;     /* [n_faces]   1/face_distance (float)  */
    Eigen::VectorXf face_flux_coef;    /* [n_faces]   face_area * face_open     */
    Eigen::VectorXf cell_inv_denom;    /* [n_cells]   1/(vol*open) for acoustic */
    std::vector<int>   wall_plate_cells;  /* indices where type==WALL or PLATE   */
    /* Per-cell face CSR for divergence accumulation (scatter-free, cell-centric) */
    std::vector<int>   csr_cell_starts;  /* [n_cells+1]                           */
    std::vector<int>   csr_face_idx;     /* face index list                       */
    std::vector<float> csr_face_sign;    /* +1 (cell is neg side) / -1 (pos side) */
    /* Reusable step scratch — avoids per-step heap allocation */
    Eigen::VectorXf div_flux_buf;      /* [n_cells] reused each amr_step        */

    /* ── Border condition (PML / absorbing layer) ─────────────────── */
    int   border_mode = 0;             /* AMR_BORDER_* constant                 */
    Eigen::VectorXf border_alpha;      /* per-cell σ (s⁻¹); 0 outside PML zone */
    Eigen::VectorXf face_V_damp;       /* per-face: exp(−σ_face·dt)             */
    Eigen::VectorXf P_damp;            /* per-cell: exp(−σ·dt)                  */
    Eigen::VectorXf P_src_coeff;       /* per-cell: (1−exp(−σ·dt))/(σ·dt)      */

    /* ── 8th-order gradient stencil (built once in amr_create) ──────────── */
    /* For each face f: up to STENCIL_SW cell pairs with Fornberg weights.   *
     * Layout: [f*SW2 .. f*SW2+SW−1] = neg-direction cells (neg1..neg4),    *
     *         [f*SW2+SW .. f*SW2+SW2−1] = pos-direction cells (pos1..pos4). *
     * Cell index −1 = unavailable (coeff is 0); graceful degradation.      */
    static constexpr int STENCIL_SW = 4;  /* half-width: 4 cells each side    */
    std::vector<int32_t> face_s_cells;    /* [n_faces * 2 * STENCIL_SW]        */
    std::vector<float>   face_s_coeff;    /* [n_faces * 2 * STENCIL_SW] (Pa/m) */

    /* ── Kirchhoff plate (optional; present if plate_active is set) ── */
    bool plate_active_flag = false; /* true after amr_setup_plate */
    int  plate_Nx = 0;
    int  plate_Ny = 0;
    float plate_dx = 0.0f;
    float plate_origin[3] = {};
    int   n_plate_nodes = 0;     /* plate_Nx * plate_Ny */

    /* Physical parameters */
    float plate_rho_h   = 0.0f;  /* kg/m² */
    float plate_D       = 0.0f;  /* N·m */
    float plate_alpha_M = 0.0f;
    float plate_beta_K  = 0.0f;

    /* Plate fields: indexed as [i*plate_Ny + j] */
    Eigen::VectorXf plate_w;       /* current displacement */
    Eigen::VectorXf plate_w_prev;  /* previous time-step */
    Eigen::VectorXf plate_ext_force; /* external force accumulator (N) */
    Eigen::VectorXi plate_active_mask; /* 1=active, 0=inactive/boundary */
    Eigen::VectorXf plate_w_new_buf;   /* reused scratch for plate leapfrog    */

    /* Active node list (flat indices) */
    std::vector<int> plate_active_idx;   /* [N_active_plate] flat plate indices */
    std::vector<PlateAMRNodeMap> plate_amr_map; /* [N_active_plate] */

    /* ── Bridge and neck sources ────────────────────────────────────── */
    std::vector<int>   bridge_plate_idx;
    std::vector<float> bridge_plate_wgt;
    int n_bridge_plate = 0;

    std::vector<int>   neck_plate_idx;
    std::vector<float> neck_plate_wgt;
    int n_neck_plate = 0;

    /* ── Mic samplers (optional) ────────────────────────────────────── */
    int n_mics_sampler = 0;
    std::vector<MicPressureSampler>  mic_p_samplers;
    std::vector<MicVelocitySampler>  mic_v_samplers;
};

static int validate_inputs(
    int n_cells,
    const double* cell_centers,
    const double* cell_volumes,
    const double* open_volume_frac,
    const uint8_t* cell_types,
    int n_faces,
    const int32_t* face_cell_neg,
    const int32_t* face_cell_pos,
    const double* face_area,
    const double* face_open_frac,
    const double* face_distance,
    double c,
    double rho_air,
    double min_dx)
{
    if (n_cells <= 0 || n_faces <= 0) return SK_ERR_DIM_MISMATCH;
    if (!cell_centers || !cell_volumes || !open_volume_frac || !cell_types
        || !face_cell_neg || !face_cell_pos || !face_area || !face_open_frac
        || !face_distance) return SK_ERR_NULL_STATE;
    if (c <= 0.0 || rho_air <= 0.0 || min_dx <= 0.0) return SK_ERR_DIM_MISMATCH;
    for (int i = 0; i < n_cells; ++i) {
        if (!(cell_volumes[i] > 0.0)) return SK_ERR_DIM_MISMATCH;
        if (open_volume_frac[i] < 0.0 || open_volume_frac[i] > 1.0) return SK_ERR_DIM_MISMATCH;
        if (cell_types[i] > 3) return SK_ERR_DIM_MISMATCH;
    }
    for (int f = 0; f < n_faces; ++f) {
        if (face_cell_neg[f] < 0 || face_cell_neg[f] >= n_cells) return SK_ERR_DIM_MISMATCH;
        if (face_cell_pos[f] < 0 || face_cell_pos[f] >= n_cells) return SK_ERR_DIM_MISMATCH;
        if (face_cell_neg[f] == face_cell_pos[f]) return SK_ERR_DIM_MISMATCH;
        if (!(face_area[f] > 0.0)) return SK_ERR_DIM_MISMATCH;
        if (face_open_frac[f] < 0.0 || face_open_frac[f] > 1.0) return SK_ERR_DIM_MISMATCH;
        if (!(face_distance[f] > 0.0)) return SK_ERR_DIM_MISMATCH;
    }
    return SK_OK;
}

/* ============================================================
 * 8th-order stencil helpers (called from amr_create)
 * ============================================================ */

/**
 * Fornberg (1988) SIAM Rev. algorithm for 1st-derivative finite-difference
 * weights at arbitrary non-uniform node positions.
 *
 * x[0..N-1]: node positions.
 * xi:        evaluation point (typically 0 = face centre).
 * w[0..N-1]: output weights such that f'(xi) ≈ Σ_k w[k]*f(x[k]).
 *
 * Complexity O(N²); N ≤ 8 here so negligible at build time.
 */
static void fornberg_d1(const float* x, int N, float xi, float* w)
{
    /* c[k*2 + m] = weight for point k, derivative order m (m∈{0,1}). */
    std::vector<float> c((size_t)N * 2, 0.0f);
    auto C = [&](int k, int m) -> float& { return c[(size_t)k * 2 + m]; };

    float c1 = 1.0f;
    C(0, 0) = 1.0f;
    for (int n = 1; n < N; ++n) {
        int   mn = std::min(n, 1);
        float c2 = 1.0f;
        for (int nu = 0; nu < n; ++nu) {
            float c3 = x[n] - x[nu];
            c2 *= c3;
            if (nu == n - 1) {
                for (int m = mn; m >= 1; --m)
                    C(n, m) = c1 / c2 * (m * C(n-1, m-1) - (x[n] - xi) * C(n-1, m));
                C(n, 0) = c1 / c2 * (-(x[n] - xi)) * C(n-1, 0);
            }
            for (int m = mn; m >= 1; --m)
                C(nu, m) = ((x[n] - xi) * C(nu, m) - m * C(nu, m-1)) / c3;
            C(nu, 0) = (x[n] - xi) * C(nu, 0) / c3;
        }
        c1 = c2;
    }
    for (int k = 0; k < N; ++k) w[k] = C(k, 1);
}

/**
 * Walk the AMR face-adjacency graph from seed_cell in direction dir.
 *
 * At each step the function selects, among all faces of the current cell,
 * the one whose outward-from-cell normal has the largest projection onto dir
 * (threshold > 0.5 to reject diagonal hops).  The cell on the other side of
 * that face is the next stencil point.  Terminates at:
 *   - boundary (no qualifying face found)
 *   - WALL or PLATE cell (type 1 or 2)
 *   - max_depth reached
 *
 * out_cells[0] = seed_cell; out_x[k] = signed distance along dir from
 * face_proj (the projection of the face centre on the dir axis).
 *
 * Returns number of cells stored (≥ 1 always).
 */
static int amr_walk_stencil(
    const AcousticAMRState* st,
    int seed_cell, float face_proj,
    const Eigen::Vector3f& dir, int max_depth,
    int32_t* out_cells, float* out_x)
{
    auto proj3 = [&](int ci) -> float {
        return (float)(st->centers(ci,0)*dir[0]
                     + st->centers(ci,1)*dir[1]
                     + st->centers(ci,2)*dir[2]);
    };
    out_cells[0] = seed_cell;
    out_x[0]     = proj3(seed_cell) - face_proj;
    int found = 1, cur = seed_cell;

    while (found < max_depth) {
        int   best_next = -1;
        float best_dot  = 0.5f;   /* require > 45° alignment */

        const int k0 = st->csr_cell_starts[cur];
        const int k1 = st->csr_cell_starts[cur + 1];
        for (int k = k0; k < k1; ++k) {
            int   fk = st->csr_face_idx[k];
            float sg = st->csr_face_sign[k];
            int   fneg = st->face_neg[fk], fpos = st->face_pos[fk];
            /* Outward-from-cur normal = sg * (centers[pos] - centers[neg]).norm */
            float dx_ = (float)(st->centers(fpos,0) - st->centers(fneg,0));
            float dy_ = (float)(st->centers(fpos,1) - st->centers(fneg,1));
            float dz_ = (float)(st->centers(fpos,2) - st->centers(fneg,2));
            float len  = std::sqrt(dx_*dx_ + dy_*dy_ + dz_*dz_);
            if (len < 1e-15f) continue;
            float dot = sg * (dx_*dir[0] + dy_*dir[1] + dz_*dir[2]) / len;
            if (dot > best_dot) {
                int nb = (sg > 0.f) ? fpos : fneg;
                if (nb >= 0 && nb < st->n_cells
                    && st->cell_type[nb] != 1 && st->cell_type[nb] != 2) {
                    best_dot = dot;
                    best_next = nb;
                }
            }
        }
        if (best_next < 0) break;
        out_cells[found] = best_next;
        out_x[found]     = proj3(best_next) - face_proj;
        ++found;
        cur = best_next;
    }
    return found;
}

/**
 * Build the per-face 8th-order (STENCIL_SW=4 pair) Fornberg stencil.
 *
 * For each face f the function:
 *   1. Computes the face-normal direction from neg→pos cell centres.
 *   2. Walks up to STENCIL_SW cells in each direction (neg, pos) following
 *      the AMR adjacency graph.
 *   3. Applies Fornberg's algorithm to all available points to compute
 *      exact 1st-derivative weights at the face centre (ξ=0).
 *   4. Stores cell indices and weights in face_s_cells / face_s_coeff.
 *
 * Graceful degradation: at boundaries or refinement interfaces where fewer
 * than 8 neighbours are available, Fornberg computes the best possible order
 * (7th, 6th, …, 1st) from the available points — no special-case needed.
 *
 * At coarse-fine AMR interfaces the actual non-uniform spacing is used
 * directly, so the weights are physically correct and mass-conservative in
 * the sense that the velocity update is consistent with the pressure gradient
 * to high order.
 */
static void amr_build_stencil(AcousticAMRState* st)
{
    const int SW  = AcousticAMRState::STENCIL_SW;
    const int SW2 = SW * 2;
    const int nf  = st->n_faces;

    st->face_s_cells.assign((size_t)nf * SW2, -1);
    st->face_s_coeff.assign((size_t)nf * SW2, 0.0f);

    int32_t neg_cells[4], pos_cells[4];
    float   neg_x[4],    pos_x[4];

    for (int f = 0; f < nf; ++f) {
        const int cn = st->face_neg[f], cp = st->face_pos[f];
        float dxf = (float)(st->centers(cp,0) - st->centers(cn,0));
        float dyf = (float)(st->centers(cp,1) - st->centers(cn,1));
        float dzf = (float)(st->centers(cp,2) - st->centers(cn,2));
        float len = std::sqrt(dxf*dxf + dyf*dyf + dzf*dzf);
        if (len < 1e-15f) continue;
        Eigen::Vector3f dir(dxf/len, dyf/len, dzf/len);

        auto proj3 = [&](int ci) -> float {
            return (float)(st->centers(ci,0)*dir[0]
                         + st->centers(ci,1)*dir[1]
                         + st->centers(ci,2)*dir[2]);
        };
        float face_proj = 0.5f * (proj3(cn) + proj3(cp));

        int n_neg = amr_walk_stencil(st, cn, face_proj, -dir, SW, neg_cells, neg_x);
        int n_pos = amr_walk_stencil(st, cp, face_proj,  dir, SW, pos_cells, pos_x);

        int total = n_neg + n_pos;
        if (total < 2) continue;   /* degenerate: can't approximate gradient */

        /* Merge into one sorted array: farthest-neg → nearest-neg → nearest-pos → farthest-pos */
        std::vector<float>   xs(total);
        std::vector<int32_t> cs(total);
        for (int k = 0; k < n_neg; ++k) {
            xs[k] = neg_x[n_neg - 1 - k];    /* most-negative first  */
            cs[k] = neg_cells[n_neg - 1 - k];
        }
        for (int k = 0; k < n_pos; ++k) {
            xs[n_neg + k] = pos_x[k];
            cs[n_neg + k] = pos_cells[k];
        }

        std::vector<float> w(total);
        fornberg_d1(xs.data(), total, 0.0f, w.data());

        /* Store: neg slots [f*SW2 .. f*SW2+n_neg-1] hold neg cells
         *         (index 0 = nearest-neg = original face_neg),
         *        pos slots [f*SW2+SW .. f*SW2+SW+n_pos-1] hold pos cells
         *         (index 0 = nearest-pos = original face_pos).
         * We reverse the neg side back to nearest-first for the hot loop. */
        for (int k = 0; k < n_neg; ++k) {
            int slot = f * SW2 + (n_neg - 1 - k);   /* slot 0 = nearest neg */
            st->face_s_cells[slot] = cs[k];
            st->face_s_coeff[slot] = w[k];
        }
        for (int k = 0; k < n_pos; ++k) {
            int slot = f * SW2 + SW + k;             /* slot SW = nearest pos */
            st->face_s_cells[slot] = cs[n_neg + k];
            st->face_s_coeff[slot] = w[n_neg + k];
        }
    }
}

AcousticAMRState* amr_create(
    int            n_cells,
    const double*  cell_centers,
    const double*  cell_volumes,
    const double*  open_volume_frac,
    const uint8_t* cell_types,
    int            n_faces,
    const int32_t* face_cell_neg,
    const int32_t* face_cell_pos,
    const double*  face_area,
    const double*  face_open_frac,
    const double*  face_distance,
    double         c,
    double         rho_air,
    double         min_dx)
{
    if (validate_inputs(n_cells, cell_centers, cell_volumes, open_volume_frac,
                        cell_types, n_faces, face_cell_neg, face_cell_pos,
                        face_area, face_open_frac, face_distance,
                        c, rho_air, min_dx) != SK_OK) {
        return nullptr;
    }

    auto* st = new (std::nothrow) AcousticAMRState();
    if (!st) return nullptr;

    st->n_cells = n_cells;
    st->n_faces = n_faces;
    st->c = c;
    st->rho_air = rho_air;
    st->dt = 0.77 * min_dx / (c * std::sqrt(3.0));

    st->centers.resize(n_cells, 3);
    st->volumes.resize(n_cells);
    st->open_vol.resize(n_cells);
    st->cell_type.resize(n_cells);
    st->face_neg.resize(n_faces);
    st->face_pos.resize(n_faces);
    st->face_area.resize(n_faces);
    st->face_open.resize(n_faces);
    st->face_distance.resize(n_faces);
    st->pressure.setZero(n_cells);
    st->velocity.setZero(n_faces);

    for (int i = 0; i < n_cells; ++i) {
        st->centers(i, 0) = cell_centers[i * 3 + 0];
        st->centers(i, 1) = cell_centers[i * 3 + 1];
        st->centers(i, 2) = cell_centers[i * 3 + 2];
        st->volumes[i] = cell_volumes[i];
        st->open_vol[i] = open_volume_frac[i];
        st->cell_type[i] = static_cast<int>(cell_types[i]);
        if ((cell_types[i] == 0 || cell_types[i] == 3) && open_volume_frac[i] > 0.0)
            st->acoustic_cells.push_back(i);
    }
    for (int f = 0; f < n_faces; ++f) {
        st->face_neg[f] = face_cell_neg[f];
        st->face_pos[f] = face_cell_pos[f];
        st->face_area[f] = face_area[f];
        st->face_open[f] = face_open_frac[f];
        st->face_distance[f] = face_distance[f];
    }
    if (st->acoustic_cells.empty()) {
        delete st;
        return nullptr;
    }

    /* ── Precompute hot-loop buffers ─────────────────────────────────── */
    st->face_inv_dist.resize(n_faces);
    st->face_flux_coef.resize(n_faces);
    for (int f = 0; f < n_faces; ++f) {
        st->face_inv_dist[f]  = 1.0f / static_cast<float>(face_distance[f]);
        st->face_flux_coef[f] = static_cast<float>(face_area[f] * face_open_frac[f]);
    }

    st->cell_inv_denom.setZero(n_cells);
    for (int idx : st->acoustic_cells) {
        double denom = cell_volumes[idx] * std::max(open_volume_frac[idx], 1e-12);
        st->cell_inv_denom[idx] = static_cast<float>(1.0 / denom);
    }

    for (int i = 0; i < n_cells; ++i)
        if (cell_types[i] == 1 || cell_types[i] == 2)
            st->wall_plate_cells.push_back(i);

    /* Per-cell CSR for divergence — each face contributes to exactly 2 cells */
    st->csr_cell_starts.assign(n_cells + 1, 0);
    for (int f = 0; f < n_faces; ++f) {
        st->csr_cell_starts[face_cell_neg[f] + 1]++;
        st->csr_cell_starts[face_cell_pos[f] + 1]++;
    }
    for (int i = 1; i <= n_cells; ++i)
        st->csr_cell_starts[i] += st->csr_cell_starts[i - 1];

    int total_entries = st->csr_cell_starts[n_cells];
    st->csr_face_idx.resize(total_entries);
    st->csr_face_sign.resize(total_entries);
    {
        std::vector<int> fill(n_cells, 0);
        for (int f = 0; f < n_faces; ++f) {
            int a = face_cell_neg[f], b = face_cell_pos[f];
            int ka = st->csr_cell_starts[a] + fill[a]++;
            st->csr_face_idx[ka]  = f;
            st->csr_face_sign[ka] = +1.0f;
            int kb = st->csr_cell_starts[b] + fill[b]++;
            st->csr_face_idx[kb]  = f;
            st->csr_face_sign[kb] = -1.0f;
        }
    }

    st->div_flux_buf.setZero(n_cells);

    /* Border damping — identity (no absorption) until amr_set_border_condition called */
    st->border_alpha.setZero(n_cells);
    st->face_V_damp.setOnes(n_faces);
    st->P_damp.setOnes(n_cells);
    st->P_src_coeff.setOnes(n_cells);

    /* Build 8th-order Fornberg gradient stencil for every face */
    amr_build_stencil(st);

    return st;
}

void amr_destroy(AcousticAMRState* st) { delete st; }

int amr_reset(AcousticAMRState* st)
{
    if (!st) return SK_ERR_NULL_STATE;
    st->pressure.setZero();
    st->velocity.setZero();
    if (st->plate_active_flag) {
        st->plate_w.setZero();
        st->plate_w_prev.setZero();
        st->plate_ext_force.setZero();
    }
    st->step_count = 0;
    return SK_OK;
}

/* ── Internal: 13-point biharmonic stencil on uniform plate grid ──────── */
static float plate_biharmonic(const Eigen::VectorXf& w,
                               int i, int j, int Nx, int Ny, float idx4)
{
    /* Returns (1/dx^4) * discrete biharmonic of w at (i,j).
     * Out-of-domain accesses return 0 (equivalent to w=0 Dirichlet edge). */
    auto W = [&](int ii, int jj) -> float {
        if (ii < 0 || ii >= Nx || jj < 0 || jj >= Ny) return 0.0f;
        return w[ii * Ny + jj];
    };
    float L = W(i-2,j) + W(i+2,j) + W(i,j-2) + W(i,j+2)
            + 2.0f*(W(i-1,j-1) + W(i-1,j+1) + W(i+1,j-1) + W(i+1,j+1))
            - 8.0f*(W(i-1,j) + W(i+1,j) + W(i,j-1) + W(i,j+1))
            + 20.0f*W(i,j);
    return L * idx4;
}

/* ── Internal: one plate sub-step ─────────────────────────────────────── */
static void amr_plate_step(AcousticAMRState* st, float dt)
{
    if (!st->plate_active_flag) return;

    const int   Nx   = st->plate_Nx;
    const int   Ny   = st->plate_Ny;
    const float dx   = st->plate_dx;
    const float dx4  = 1.0f / (dx * dx * dx * dx);
    const float D    = st->plate_D;
    const float rh   = st->plate_rho_h;
    const float aM   = st->plate_alpha_M;
    const float bK   = st->plate_beta_K;
    const float dt2  = dt * dt;
    const float inv_rh = 1.0f / rh;

    /* Step plate displacement using Rayleigh-damped leapfrog:
     *   rho_h * (w_new - 2*w + w_prev) / dt^2
     *     = F_acou + F_ext
     *       - D * (1 + bK/dt) * L4[w]
     *       + D * (bK/dt)     * L4[w_prev]
     *       - aM * rho_h * (w - w_prev) / dt
     *
     * Rearranged for w_new:
     *   w_new = (  (2 + aM*dt) * w - (1 + aM*dt/2)*w_prev   [WRONG; see below]
     *   ...
     * Correct explicit central-difference form:
     *   damp_fwd  = 1 + aM * dt / 2
     *   damp_bwd  = 1 - aM * dt / 2   (from velocity damping at t=n)
     *   w_new = (damp_bwd/damp_fwd) * (2*w - w_prev)
     *         + dt^2 / (rho_h * damp_fwd)
     *           * (F_acou + F_ext - D*(1+bK/dt)*L4w + D*(bK/dt)*L4w_prev)
     */
    const float damp_fwd = 1.0f + 0.5f * aM * dt;
    const float damp_bwd = 1.0f - 0.5f * aM * dt;
    const float coeff_D0 = D * (1.0f + bK / dt);  /* current L4 coefficient */
    const float coeff_Dp = D * (bK / dt);          /* previous L4 coefficient */

    /* Use preallocated scratch buffer — no heap allocation per step */
    Eigen::VectorXf& w_new = st->plate_w_new_buf;
    w_new.setZero();

    int N_active = (int)st->plate_active_idx.size();
    for (int n = 0; n < N_active; ++n) {
        int flat = st->plate_active_idx[n];
        int i    = flat / Ny;
        int j    = flat % Ny;

        /* Acoustic load: face-area-weighted pressure integral over the
         * coupled AMR faces on each side of the plate node.
         * area_above / area_below are normalized weights (sum ≈ 1) so
         * P_above is the area-weighted mean pressure, and F = P_avg * dx². */
        float F_acou = 0.0f;
        const PlateAMRNodeMap& mp = st->plate_amr_map[n];
        float P_above = 0.0f;
        if (!mp.cell_above_side.empty()) {
            for (int k = 0; k < (int)mp.face_above.size(); ++k)
                P_above += mp.area_above[k] * st->pressure[mp.cell_above_side[k]];
        } else if (mp.cell_above >= 0) {
            P_above = st->pressure[mp.cell_above];
        }
        float P_below = 0.0f;
        if (!mp.cell_below_side.empty()) {
            for (int k = 0; k < (int)mp.face_below.size(); ++k)
                P_below += mp.area_below[k] * st->pressure[mp.cell_below_side[k]];
        } else if (mp.cell_below >= 0) {
            P_below = st->pressure[mp.cell_below];
        }
        F_acou = (P_below - P_above) * dx * dx;  /* P_below pushes +z (upward) */

        float F_ext = st->plate_ext_force[flat];

        float L4w  = plate_biharmonic(st->plate_w,      i, j, Nx, Ny, dx4);
        float L4wp = plate_biharmonic(st->plate_w_prev, i, j, Nx, Ny, dx4);

        float w_curr = st->plate_w[flat];
        float w_prev = st->plate_w_prev[flat];

        float rhs = F_acou + F_ext - coeff_D0 * L4w + coeff_Dp * L4wp;
        w_new[flat] = (damp_bwd * (2.0f * w_curr - w_prev)
                       + dt2 * inv_rh * rhs)
                      / damp_fwd;
    }

    /* Commit: w_prev ← w, w ← w_new, clear ext_force
     * Use Eigen scatter for active subset to keep inactive nodes at zero. */
    for (int n = 0; n < N_active; ++n) {
        int flat = st->plate_active_idx[n];
        st->plate_w_prev[flat] = st->plate_w[flat];
        st->plate_w[flat]      = w_new[flat];
    }
    st->plate_ext_force.setZero();
}

/* ── Internal: impose plate velocity BC on AMR faces ─────────────────── */
static void amr_apply_plate_bc(AcousticAMRState* st, float dt)
{
    if (!st->plate_active_flag) return;
    int N_active = (int)st->plate_active_idx.size();
    for (int n = 0; n < N_active; ++n) {
        int flat = st->plate_active_idx[n];
        /* Plate velocity (backward finite difference; w_prev is still the
         * value BEFORE the plate step above — i.e., w at time n).
         * At this call point we are mid-step: plate_w still = w[n],
         * plate_w_prev = w[n-1].  So v_plate = (w[n] - w[n-1]) / dt. */
        float v_plate = (st->plate_w[flat] - st->plate_w_prev[flat]) / dt;
        const PlateAMRNodeMap& mp = st->plate_amr_map[n];
        /* Faces above: positive normal points away from plate (upward).
         * Physical velocity v_plate is set directly; face_flux_coef = A_f*open_f
         * already accounts for area in the divergence accumulation, so no
         * additional area-weight factor belongs here. */
        for (int k = 0; k < (int)mp.face_above.size(); ++k)
            st->velocity[mp.face_above[k]] = v_plate;
        /* Faces below: outward from the plate body = negative direction. */
        for (int k = 0; k < (int)mp.face_below.size(); ++k)
            st->velocity[mp.face_below[k]] = -v_plate;
    }
}

int amr_step(AcousticAMRState* st, int n_steps)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (n_steps < 0) return SK_ERR_DIM_MISMATCH;
    if (n_steps == 0) return SK_OK;

    const float dt   = static_cast<float>(st->dt);
    const float rho  = static_cast<float>(st->rho_air);
    /* Bulk modulus prefactor: rho * c^2 * dt */
    const float bulk = static_cast<float>(st->rho_air * st->c * st->c * st->dt);
    const float dt_over_rho = dt / rho;

    Eigen::VectorXf& div = st->div_flux_buf;  /* reuse preallocated buffer */

    for (int s = 0; s < n_steps; ++s) {
        /* ── 1. Velocity update: 8th-order Fornberg gradient ───────────────── */
        /* grad_p at face f = Σ_k face_s_coeff[f*SW2+k] * p[face_s_cells[f*SW2+k]] *
         * Fornberg weights are precomputed for actual cell-centre positions      *
         * (non-uniform spacing handled exactly, not assumed uniform).            *
         * face_s_cells entries of −1 are padding — skipped (coeff is 0).         */
        constexpr int SW2 = AcousticAMRState::STENCIL_SW * 2;
        const auto* sc = st->face_s_cells.data();
        const auto* sw = st->face_s_coeff.data();
        for (int f = 0; f < st->n_faces; ++f) {
            const int base = f * SW2;
            float grad_p = 0.0f;
            for (int k = 0; k < SW2; ++k) {
                const int ci = sc[base + k];
                if (ci >= 0) grad_p += sw[base + k] * st->pressure[ci];
            }
            st->velocity[f] = (st->velocity[f] - dt_over_rho * grad_p)
                               * st->face_V_damp[f];
        }

        /* ── 2. Plate velocity BC (overrides faces adjacent to plate) ── */
        amr_apply_plate_bc(st, dt);

        /* ── 3. Divergence accumulation: cell-centric CSR (no scatter race) ── */
        div.setZero();
        const int n_cells = st->n_cells;
        for (int c = 0; c < n_cells; ++c) {
            float d = 0.0f;
            const int k0 = st->csr_cell_starts[c];
            const int k1 = st->csr_cell_starts[c + 1];
            for (int k = k0; k < k1; ++k) {
                const int f = st->csr_face_idx[k];
                d += st->csr_face_sign[k] * st->velocity[f] * st->face_flux_coef[f];
            }
            div[c] = d;
        }

        /* ── 4. Pressure update: exact CPML integrating factor ─────────── */
        /* p_new[i] = p[i] * exp(-σ·dt)                                         *
         *          - ρc²·dt · div[i]/V_eff[i] · (1-exp(-σ·dt))/(σ·dt)         *
         * = p * P_damp - (bulk*div*cell_inv_denom) * P_src_coeff              *
         * For non-PML cells: P_damp=1, P_src_coeff=1 → standard leapfrog.    */
        st->pressure = st->pressure.cwiseProduct(st->P_damp)
                     - (bulk * div).cwiseProduct(st->cell_inv_denom)
                                   .cwiseProduct(st->P_src_coeff);

        /* ── 5. Zero wall and plate cells (rigid / structural) ── */
        for (int idx : st->wall_plate_cells)
            st->pressure[idx] = 0.0f;

        /* ── 6. Kirchhoff plate step ── */
        amr_plate_step(st, dt);

        /* ── 7. Stability check ── */
        if (!st->pressure.allFinite())
            return SK_ERR_DIVERGED;
        ++st->step_count;
    }
    return SK_OK;
}

int amr_inject_pressure_nearest(AcousticAMRState* st, const double* xyz, float value)
{
    if (!st || !xyz) return SK_ERR_NULL_STATE;
    if (!std::isfinite(value)) return SK_ERR_DIVERGED;
    Eigen::Vector3d p(xyz[0], xyz[1], xyz[2]);
    double best = 1e300;
    int best_idx = -1;
    for (int idx : st->acoustic_cells) {
        double d2 = (st->centers.row(idx).transpose() - p).squaredNorm();
        if (d2 < best) {
            best = d2;
            best_idx = idx;
        }
    }
    if (best_idx < 0) return SK_ERR_DIM_MISMATCH;
    st->pressure[best_idx] += value;
    return SK_OK;
}

int amr_get_pressure(const AcousticAMRState* st, float* out_pressure, int out_count)
{
    if (!st || !out_pressure) return SK_ERR_NULL_STATE;
    if (out_count != st->n_cells) return SK_ERR_DIM_MISMATCH;
    std::copy(st->pressure.data(), st->pressure.data() + st->n_cells, out_pressure);
    return SK_OK;
}

int amr_get_cell_centers(const AcousticAMRState* st, float* out_xyz, int n_cells_3)
{
    if (!st || !out_xyz) return SK_ERR_NULL_STATE;
    if (n_cells_3 < st->n_cells * 3) return SK_ERR_DIM_MISMATCH;
    for (int i = 0; i < st->n_cells; ++i) {
        out_xyz[3*i+0] = (float)st->centers(i, 0);
        out_xyz[3*i+1] = (float)st->centers(i, 1);
        out_xyz[3*i+2] = (float)st->centers(i, 2);
    }
    return SK_OK;
}

int amr_scatter_pressure_uniform(
    const AcousticAMRState* st,
    int Nx, int Ny, int Nz,
    const double bmin[3], const double bmax[3],
    float* out)
{
    if (!st || !out || !bmin || !bmax) return SK_ERR_NULL_STATE;
    if (Nx <= 0 || Ny <= 0 || Nz <= 0) return SK_ERR_DIM_MISMATCH;

    const int out_len = Nx * Ny * Nz;
    std::fill(out, out + out_len, 0.0f);

    const double rx = (Nx > 1) ? (Nx - 1) / (bmax[0] - bmin[0]) : 0.0;
    const double ry = (Ny > 1) ? (Ny - 1) / (bmax[1] - bmin[1]) : 0.0;
    const double rz = (Nz > 1) ? (Nz - 1) / (bmax[2] - bmin[2]) : 0.0;

    for (int ci = 0; ci < st->n_cells; ++ci) {
        const double cx = st->centers(ci, 0);
        const double cy = st->centers(ci, 1);
        const double cz = st->centers(ci, 2);
        int ix = (int)std::lround((cx - bmin[0]) * rx);
        int iy = (int)std::lround((cy - bmin[1]) * ry);
        int iz = (int)std::lround((cz - bmin[2]) * rz);
        if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny || iz < 0 || iz >= Nz)
            continue;
        out[ix * (Ny * Nz) + iy * Nz + iz] = st->pressure[ci];
    }
    return SK_OK;
}

int amr_get_velocity(const AcousticAMRState* st, float* out_velocity, int out_count)
{
    if (!st || !out_velocity) return SK_ERR_NULL_STATE;
    if (out_count != st->n_faces) return SK_ERR_DIM_MISMATCH;
    std::copy(st->velocity.data(), st->velocity.data() + st->n_faces, out_velocity);
    return SK_OK;
}

double amr_get_dt(const AcousticAMRState* st)
{
    return st ? st->dt : 0.0;
}

int amr_get_step_count(const AcousticAMRState* st)
{
    return st ? st->step_count : 0;
}

/* ============================================================
 * Border condition setup
 * ============================================================ */

int amr_set_border_condition(
    AcousticAMRState* st,
    int               mode,
    float             sigma_order,
    float             R_reflection,
    float             border_Z_match,
    int               n_pml,
    const double*     bounds_min,
    const double*     bounds_max)
{
    if (!st || !bounds_min || !bounds_max) return SK_ERR_NULL_STATE;
    if (n_pml <= 0) return SK_OK;  /* no PML requested — leave identity */

    st->border_mode = mode;

    const float eff_order = (sigma_order > 0.0f) ? sigma_order : 3.0f;
    const float dt_f = static_cast<float>(st->dt);
    const int   nc   = st->n_cells;
    const int   nf   = st->n_faces;

    /* Reproduce the uniform FDTD sigma_max formula exactly. */
    /* sigma_max · dt = 3.45 / n_pml  (matches acoustic_fdtd.cpp) */
    float sigma_max = 3.45f / (dt_f * static_cast<float>(n_pml));

    /* For ROOM_PANEL: scale by (1 − R_reflection) so that cells with     *
     * large R get weaker absorption and let more energy reflect back.    */
    if (mode == AMR_BORDER_ROOM_PANEL) {
        float r = std::max(0.0f, std::min(1.0f, R_reflection));
        sigma_max *= (1.0f - r);
    }

    /* min_dx recovered from CFL: dt = 0.77 * min_dx / (c * sqrt(3)) */
    double min_dx   = st->dt * st->c * std::sqrt(3.0) / 0.77;
    double pml_depth = static_cast<double>(n_pml) * min_dx;

    /* ── Per-cell absorption coefficient ─────────────────────────── */
    st->border_alpha.setZero(nc);
    for (int i = 0; i < nc; ++i) {
        if (st->cell_type[i] != 3) continue;  /* only PML-type cells */

        double cx = st->centers(i, 0);
        double cy = st->centers(i, 1);
        double cz = st->centers(i, 2);

        /* Distance to nearest domain boundary */
        double d = std::min({cx - bounds_min[0], bounds_max[0] - cx,
                             cy - bounds_min[1], bounds_max[1] - cy,
                             cz - bounds_min[2], bounds_max[2] - cz});
        d = std::max(d, 0.0);

        /* Normalised depth from inner PML edge (0 at inner, 1 at outer) */
        float norm = static_cast<float>(
            std::max(0.0, std::min(1.0, 1.0 - d / pml_depth)));
        st->border_alpha[i] = sigma_max * std::pow(norm, eff_order);
    }

    /* ── Per-cell pressure damping (exact CPML integrating factor) ─── */
    /* Exact solution to dp/dt + σ·p = source over one timestep.
     * P_damp       = exp(-σ·dt)
     * P_src_coeff  = (1 - exp(-σ·dt)) / (σ·dt)  →  1 as σ→0 (L'Hôpital)
     *
     * The pressure update becomes:
     *   p_new = p * P_damp - ρc²·dt·∇·v/V_eff · P_src_coeff
     * which recovers the standard leapfrog for non-PML cells (σ=0).         */
    for (int i = 0; i < nc; ++i) {
        float s = st->border_alpha[i] * dt_f;
        st->P_damp[i]      = std::exp(-s);
        st->P_src_coeff[i] = (s < 1e-6f) ? 1.0f - 0.5f * s
                                          : (1.0f - std::exp(-s)) / s;
    }

    /* ── Per-face velocity damping (exact, volume-weighted harmonic mean) ── */
    /* At coarse-fine AMR interfaces the coarser (larger-volume) cell has    *
     * a larger share of the impedance; harmonic mean avoids over-damping    *
     * the small fine cells while matching the coarse PML envelope.          *
     * exp(-σ_face·dt) is exact (never negative, even for large σ·dt).      */
    for (int f = 0; f < nf; ++f) {
        int  ni = st->face_neg[f], pi = st->face_pos[f];
        float vn = static_cast<float>(st->open_vol[ni]);
        float vp = static_cast<float>(st->open_vol[pi]);
        float denom = vn + vp;
        float s_face = (denom > 0.f)
            ? 2.0f * (vp * st->border_alpha[ni] + vn * st->border_alpha[pi]) / denom
            : 0.5f * (st->border_alpha[ni] + st->border_alpha[pi]);
        st->face_V_damp[f] = std::exp(-s_face * dt_f);
    }

    return SK_OK;
}

/* ============================================================
 * Plate setup
 * ============================================================ */

int amr_setup_plate(
    AcousticAMRState*  st,
    int                plate_Nx,
    int                plate_Ny,
    float              plate_dx,
    const float        plate_origin[3],
    const uint8_t*     plate_active,
    float              plate_rho_h,
    float              plate_D,
    float              plate_alpha_M,
    float              plate_beta_K,
    int                n_active_plate,
    const int32_t*     plate_active_flat_idx,
    const int32_t*     face_above_starts,
    const int32_t*     face_above_idx,
    const float*       face_above_wgt,
    const int32_t*     face_below_starts,
    const int32_t*     face_below_idx,
    const float*       face_below_wgt,
    const int32_t*     cell_above,
    const int32_t*     cell_below)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (!plate_active || !plate_active_flat_idx) return SK_ERR_NULL_STATE;
    if (!face_above_starts || !face_below_starts) return SK_ERR_NULL_STATE;
    if (!cell_above || !cell_below) return SK_ERR_NULL_STATE;
    if (plate_Nx <= 0 || plate_Ny <= 0 || plate_dx <= 0.0f) return SK_ERR_DIM_MISMATCH;
    if (plate_rho_h <= 0.0f || plate_D <= 0.0f) return SK_ERR_DIM_MISMATCH;
    if (n_active_plate < 0) return SK_ERR_DIM_MISMATCH;

    /* Verify n_active_plate matches active mask */
    int n_plate_nodes = plate_Nx * plate_Ny;
    int count = 0;
    for (int k = 0; k < n_plate_nodes; ++k)
        if (plate_active[k]) ++count;
    if (count != n_active_plate) return SK_ERR_DIM_MISMATCH;

    /* Validate face/cell index bounds */
    int n_face_above_total = face_above_starts[n_active_plate];
    int n_face_below_total = face_below_starts[n_active_plate];
    for (int k = 0; k < n_face_above_total; ++k)
        if (face_above_idx[k] < 0 || face_above_idx[k] >= st->n_faces)
            return SK_ERR_DIM_MISMATCH;
    for (int k = 0; k < n_face_below_total; ++k)
        if (face_below_idx[k] < 0 || face_below_idx[k] >= st->n_faces)
            return SK_ERR_DIM_MISMATCH;
    for (int n = 0; n < n_active_plate; ++n) {
        if (cell_above[n] != -1 && (cell_above[n] < 0 || cell_above[n] >= st->n_cells))
            return SK_ERR_DIM_MISMATCH;
        if (cell_below[n] != -1 && (cell_below[n] < 0 || cell_below[n] >= st->n_cells))
            return SK_ERR_DIM_MISMATCH;
    }

    st->plate_Nx   = plate_Nx;
    st->plate_Ny   = plate_Ny;
    st->plate_dx   = plate_dx;
    st->plate_origin[0] = plate_origin[0];
    st->plate_origin[1] = plate_origin[1];
    st->plate_origin[2] = plate_origin[2];
    st->n_plate_nodes  = n_plate_nodes;
    st->plate_rho_h   = plate_rho_h;
    st->plate_D       = plate_D;
    st->plate_alpha_M = plate_alpha_M;
    st->plate_beta_K  = plate_beta_K;

    st->plate_w.setZero(n_plate_nodes);
    st->plate_w_prev.setZero(n_plate_nodes);
    st->plate_ext_force.setZero(n_plate_nodes);
    st->plate_w_new_buf.setZero(n_plate_nodes);
    st->plate_active_mask.resize(n_plate_nodes);
    for (int k = 0; k < n_plate_nodes; ++k)
        st->plate_active_mask[k] = plate_active[k] ? 1 : 0;

    /* Build active node list and mappings */
    st->plate_active_idx.assign(plate_active_flat_idx,
                                 plate_active_flat_idx + n_active_plate);
    st->plate_amr_map.resize(n_active_plate);
    for (int n = 0; n < n_active_plate; ++n) {
        PlateAMRNodeMap& mp = st->plate_amr_map[n];
        mp.cell_above = cell_above[n];
        mp.cell_below = cell_below[n];

        int a0 = face_above_starts[n];
        int a1 = face_above_starts[n + 1];
        mp.face_above.assign(face_above_idx + a0, face_above_idx + a1);
        mp.area_above.assign(face_above_wgt + a0, face_above_wgt + a1);
        /* For each above-face, the AIR cell is on the pos side (face_pos[f]).
         * Convention: face_above has neg=PLATE (lower z), pos=AIR (higher z). */
        mp.cell_above_side.resize(mp.face_above.size());
        for (int k = 0; k < (int)mp.face_above.size(); ++k)
            mp.cell_above_side[k] = st->face_pos[mp.face_above[k]];

        int b0 = face_below_starts[n];
        int b1 = face_below_starts[n + 1];
        mp.face_below.assign(face_below_idx + b0, face_below_idx + b1);
        mp.area_below.assign(face_below_wgt + b0, face_below_wgt + b1);
        /* For each below-face, the AIR cell is on the neg side (face_neg[f]).
         * Convention: face_below has neg=AIR (lower z), pos=PLATE (higher z). */
        mp.cell_below_side.resize(mp.face_below.size());
        for (int k = 0; k < (int)mp.face_below.size(); ++k)
            mp.cell_below_side[k] = st->face_neg[mp.face_below[k]];
    }

    /* Issue C: validate that every active plate node is acoustically coupled
     * on both sides.  A node with neither face list nor fallback cell on a
     * given side would produce a silently wrong (zero) acoustic load. */
    for (int n = 0; n < n_active_plate; ++n) {
        const PlateAMRNodeMap& mp = st->plate_amr_map[n];
        if (mp.face_above.empty() && mp.cell_above < 0)
            return SK_ERR_DIM_MISMATCH;
        if (mp.face_below.empty() && mp.cell_below < 0)
            return SK_ERR_DIM_MISMATCH;
    }

    st->plate_active_flag = true;
    return SK_OK;
}

/* ============================================================
 * Bridge and neck source registration
 * ============================================================ */

int amr_set_bridge_plate_sources(AcousticAMRState* st, int n,
                                  const int32_t* idx, const float* wgt)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    if (n <= 0 || !idx || !wgt) return SK_ERR_DIM_MISMATCH;
    for (int i = 0; i < n; ++i)
        if (idx[i] < 0 || idx[i] >= st->n_plate_nodes) return SK_ERR_DIM_MISMATCH;
    st->bridge_plate_idx.assign(idx, idx + n);
    st->bridge_plate_wgt.assign(wgt, wgt + n);
    st->n_bridge_plate = n;
    return SK_OK;
}

int amr_set_neck_plate_sources(AcousticAMRState* st, int n,
                                const int32_t* idx, const float* wgt)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (n == 0) { st->n_neck_plate = 0; return SK_OK; }
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    if (!idx || !wgt) return SK_ERR_DIM_MISMATCH;
    for (int i = 0; i < n; ++i)
        if (idx[i] < 0 || idx[i] >= st->n_plate_nodes) return SK_ERR_DIM_MISMATCH;
    st->neck_plate_idx.assign(idx, idx + n);
    st->neck_plate_wgt.assign(wgt, wgt + n);
    st->n_neck_plate = n;
    return SK_OK;
}

/* ============================================================
 * Force injection
 * ============================================================ */

int amr_inject_bridge_drive(AcousticAMRState* st,
                             const float* cell_drives, int n_cells,
                             float force_scale)
{
    if (!st || !cell_drives) return SK_ERR_NULL_STATE;
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    if (n_cells != st->n_bridge_plate) return SK_ERR_DIM_MISMATCH;
    for (int i = 0; i < n_cells; ++i) {
        int flat = st->bridge_plate_idx[i];
        st->plate_ext_force[flat] += cell_drives[i] * st->bridge_plate_wgt[i] * force_scale;
    }
    return SK_OK;
}

int amr_inject_plate_force(AcousticAMRState* st, int n,
                            const int32_t* plate_idx, const float* wgt,
                            float total_force_N)
{
    if (!st) return SK_ERR_NULL_STATE;
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    if (n <= 0 || !plate_idx || !wgt) return SK_ERR_DIM_MISMATCH;
    for (int i = 0; i < n; ++i) {
        if (plate_idx[i] < 0 || plate_idx[i] >= st->n_plate_nodes)
            return SK_ERR_DIM_MISMATCH;
        st->plate_ext_force[plate_idx[i]] += wgt[i] * total_force_N;
    }
    return SK_OK;
}

/* ============================================================
 * Plate displacement sampling
 * ============================================================ */

int amr_sample_plate_displacement_batch(const AcousticAMRState* st,
                                         int n_points, const int* idx4,
                                         const float* wgt4, float* out_w)
{
    if (!st || !idx4 || !wgt4 || !out_w) return SK_ERR_NULL_STATE;
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    for (int p = 0; p < n_points; ++p) {
        float val = 0.0f;
        for (int k = 0; k < 4; ++k) {
            int flat = idx4[p * 4 + k];
            if (flat < 0 || flat >= st->n_plate_nodes) return SK_ERR_DIM_MISMATCH;
            val += wgt4[p * 4 + k] * st->plate_w[flat];
        }
        out_w[p] = val;
    }
    return SK_OK;
}

int amr_sample_plate_weighted(const AcousticAMRState* st, int n,
                               const int* plate_idx, const float* wgt,
                               float* out_w)
{
    if (!st || !plate_idx || !wgt || !out_w) return SK_ERR_NULL_STATE;
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    float val = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (plate_idx[i] < 0 || plate_idx[i] >= st->n_plate_nodes)
            return SK_ERR_DIM_MISMATCH;
        val += wgt[i] * st->plate_w[plate_idx[i]];
    }
    *out_w = val;
    return SK_OK;
}

int amr_get_plate_displacement(const AcousticAMRState* st, float* out, int out_len)
{
    if (!st || !out) return SK_ERR_NULL_STATE;
    if (!st->plate_active_flag) return SK_ERR_DIM_MISMATCH;
    if (out_len != st->n_plate_nodes) return SK_ERR_DIM_MISMATCH;
    std::copy(st->plate_w.data(), st->plate_w.data() + st->n_plate_nodes, out);
    return SK_OK;
}

/* ============================================================
 * Mic sampler precomputation and sampling
 * ============================================================ */

int amr_setup_mic_samplers(AcousticAMRState* st, int n_mics,
                            const float* pos_xyz, const float* axis_xyz,
                            int need_velocity)
{
    if (!st || !pos_xyz || !axis_xyz) return SK_ERR_NULL_STATE;
    if (n_mics <= 0) return SK_ERR_DIM_MISMATCH;

    st->n_mics_sampler = n_mics;
    st->mic_p_samplers.resize(n_mics);
    st->mic_v_samplers.resize(n_mics);

    /* Search radius: use 2× the mean cell spacing */
    double mean_vol = 0.0;
    for (int idx : st->acoustic_cells)
        mean_vol += st->volumes[idx];
    if (!st->acoustic_cells.empty())
        mean_vol /= (double)st->acoustic_cells.size();
    float search_r = 2.0f * (float)std::cbrt(mean_vol);

    for (int mi = 0; mi < n_mics; ++mi) {
        float mx = pos_xyz[mi * 3 + 0];
        float my = pos_xyz[mi * 3 + 1];
        float mz = pos_xyz[mi * 3 + 2];

        /* ── Pressure sampler: inverse-distance weighted acoustic cells ── */
        MicPressureSampler& ps = st->mic_p_samplers[mi];
        float wsum = 0.0f;
        for (int idx : st->acoustic_cells) {
            float dx = mx - (float)st->centers(idx, 0);
            float dy = my - (float)st->centers(idx, 1);
            float dz = mz - (float)st->centers(idx, 2);
            float r  = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < search_r) {
                float w = (r < 1e-9f) ? 1e9f : 1.0f / r;
                ps.cell_idx.push_back(idx);
                ps.cell_wgt.push_back(w);
                wsum += w;
            }
        }
        if (ps.cell_idx.empty()) return SK_ERR_DIM_MISMATCH;
        /* Normalize */
        float inv_sum = 1.0f / wsum;
        for (float& w : ps.cell_wgt) w *= inv_sum;

        /* ── Velocity sampler: area-weighted Cartesian face decomposition ── */
        MicVelocitySampler& vs = st->mic_v_samplers[mi];
        if (need_velocity) {
            float total_area = 0.0f;
            for (int f = 0; f < st->n_faces; ++f) {
                int a = st->face_neg[f], b = st->face_pos[f];
                float fcx = 0.5f * (float)(st->centers(a,0) + st->centers(b,0));
                float fcy = 0.5f * (float)(st->centers(a,1) + st->centers(b,1));
                float fcz = 0.5f * (float)(st->centers(a,2) + st->centers(b,2));
                float dx2 = fcx - mx, dy2 = fcy - my, dz2 = fcz - mz;
                float r2  = std::sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
                if (r2 >= search_r || st->face_open[f] < 0.5) continue;

                /* Face normal direction (neg→pos, unit vector) */
                float nx = (float)(st->centers(b,0) - st->centers(a,0));
                float ny = (float)(st->centers(b,1) - st->centers(a,1));
                float nz = (float)(st->centers(b,2) - st->centers(a,2));
                float n_len = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (n_len < 1e-12f) continue;
                nx /= n_len; ny /= n_len; nz /= n_len;

                float area_w = (float)(st->face_area[f] * st->face_open[f]);
                vs.face_idx.push_back(f);
                vs.wgt_x.push_back(nx * area_w);
                vs.wgt_y.push_back(ny * area_w);
                vs.wgt_z.push_back(nz * area_w);
                total_area += area_w;
            }
            /* Normalize by total area so a uniform plane wave gives v=1 m/s */
            if (total_area > 1e-20f) {
                float inv_a = 1.0f / total_area;
                for (float& w : vs.wgt_x) w *= inv_a;
                for (float& w : vs.wgt_y) w *= inv_a;
                for (float& w : vs.wgt_z) w *= inv_a;
            }
        }
    }
    return SK_OK;
}

int amr_sample_mics(const AcousticAMRState* st, int n_mics,
                    float* out_p, float* out_vx, float* out_vy, float* out_vz)
{
    if (!st || !out_p) return SK_ERR_NULL_STATE;
    if (n_mics != st->n_mics_sampler) return SK_ERR_DIM_MISMATCH;

    for (int mi = 0; mi < n_mics; ++mi) {
        /* Pressure */
        const MicPressureSampler& ps = st->mic_p_samplers[mi];
        float p = 0.0f;
        for (int k = 0; k < (int)ps.cell_idx.size(); ++k)
            p += ps.cell_wgt[k] * st->pressure[ps.cell_idx[k]];
        out_p[mi] = p;

        /* Velocity (decomposed into Cartesian components) */
        if (out_vx && out_vy && out_vz) {
            const MicVelocitySampler& vs = st->mic_v_samplers[mi];
            float vx = 0.0f, vy = 0.0f, vz = 0.0f;
            for (int k = 0; k < (int)vs.face_idx.size(); ++k) {
                float vf = st->velocity[vs.face_idx[k]];
                vx += vs.wgt_x[k] * vf;
                vy += vs.wgt_y[k] * vf;
                vz += vs.wgt_z[k] * vf;
            }
            out_vx[mi] = vx;
            out_vy[mi] = vy;
            out_vz[mi] = vz;
        }
    }
    return SK_OK;
}
