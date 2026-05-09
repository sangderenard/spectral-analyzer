/**
 * field_march.cpp — Dense complex spectral field grid + stencil marcher.
 *
 * Implements the storage and propagation backbone declared in
 * ``csrc/include/field_grid.h``.  Two grid kinds are live:
 *
 *   FIELD_GRID_REGULAR  — uniform Cartesian, full SoA std::complex<float>.
 *   FIELD_GRID_KDTREE   — explicit per-leaf rectangular chunks (anisotropic
 *                         AMR; replaces the old octree plan).
 *
 * Stencils enumerated in field_grid.h drive the time-stepping/Helmholtz
 * routines.  Rule of the rewrite: NO data reduction.  Storage is complex64,
 * n_bands is preserved, no abs() / no quantize / no band collapse.
 *
 * GLSL parity:  the regular-grid layout is byte-identical to a paired
 * (re, im) r32f SSBO so the compute shader port can write directly to it
 * via two ``imageAtomicAdd``-style emulations (compare-and-swap loop on the
 * float bits).  The k-d variant ships CPU-side first; GPU port is staged.
 */

#include "field_grid.h"
#include "serial_kernel.h"

#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>
#include <vector>

using cd32 = std::complex<float>;

/* ── Concrete grid struct (declared opaquely in field_grid.h) ──────────── */
struct FieldGrid {
    int          kind        = FIELD_GRID_REGULAR;
    int          n_bands     = 0;
    int          dims[3]     = {0, 0, 0};   /* REGULAR only */
    float        bmin[3]     = {0.f, 0.f, 0.f};
    float        bmax[3]     = {0.f, 0.f, 0.f};
    int64_t      n_cells_total = 0;          /* across all leaves */
    /* Storage: n_bands * n_cells_total * sizeof(cd32) bytes.  Indexed for
     * REGULAR as data[band * n_cells_total + (z*ny + y)*nx + x]. */
    cd32*        data        = nullptr;
    /* KDTREE only: a flat node list.  Leaf nodes carry first_data offsets
     * into the contiguous data array. */
    std::vector<KdNode> nodes;
};

/* ── Allocation helpers ─────────────────────────────────────────────────── */
static cd32* _alloc_zero(int64_t n) {
    if (n <= 0) return nullptr;
    cd32* p = static_cast<cd32*>(std::calloc(static_cast<size_t>(n), sizeof(cd32)));
    return p;
}

extern "C" SK_API FieldGrid* field_grid_create_regular(
    int n_bands, int nx, int ny, int nz,
    const float bmin[3], const float bmax[3])
{
    if (n_bands <= 0 || nx <= 0 || ny <= 0 || nz <= 0) return nullptr;
    FieldGrid* g = new (std::nothrow) FieldGrid();
    if (!g) return nullptr;
    g->kind = FIELD_GRID_REGULAR;
    g->n_bands = n_bands;
    g->dims[0] = nx; g->dims[1] = ny; g->dims[2] = nz;
    for (int i = 0; i < 3; ++i) {
        g->bmin[i] = bmin ? bmin[i] : 0.f;
        g->bmax[i] = bmax ? bmax[i] : 1.f;
    }
    g->n_cells_total = static_cast<int64_t>(nx) * ny * nz;
    g->data = _alloc_zero(g->n_cells_total * n_bands);
    if (!g->data) { delete g; return nullptr; }
    return g;
}

extern "C" SK_API FieldGrid* field_grid_create_kdtree(
    int n_bands,
    const KdNode* nodes, int n_nodes)
{
    if (n_bands <= 0 || !nodes || n_nodes <= 0) return nullptr;
    FieldGrid* g = new (std::nothrow) FieldGrid();
    if (!g) return nullptr;
    g->kind = FIELD_GRID_KDTREE;
    g->n_bands = n_bands;
    g->nodes.assign(nodes, nodes + n_nodes);

    /* Compute total leaf cell count and assign first_data offsets. */
    int64_t total = 0;
    for (auto& nd : g->nodes) {
        if (nd.child_lo < 0) {  /* leaf */
            int64_t lc = static_cast<int64_t>(nd.leaf_dims[0])
                       * nd.leaf_dims[1] * nd.leaf_dims[2];
            nd.first_data = total;
            total += lc;
        }
    }
    g->n_cells_total = total;

    /* World AABB = root bounds. */
    if (!g->nodes.empty()) {
        for (int i = 0; i < 3; ++i) {
            g->bmin[i] = g->nodes[0].bmin[i];
            g->bmax[i] = g->nodes[0].bmax[i];
        }
    }
    g->data = _alloc_zero(g->n_cells_total * n_bands);
    if (!g->data) { delete g; return nullptr; }
    return g;
}

extern "C" SK_API void field_grid_destroy(FieldGrid* g) {
    if (!g) return;
    if (g->data) std::free(g->data);
    delete g;
}

/* ── Introspection ──────────────────────────────────────────────────────── */
extern "C" SK_API int  field_grid_kind   (const FieldGrid* g) { return g ? g->kind    : -1; }
extern "C" SK_API int  field_grid_n_bands(const FieldGrid* g) { return g ? g->n_bands : 0; }
extern "C" SK_API int64_t field_grid_n_cells_total(const FieldGrid* g) {
    return g ? g->n_cells_total : 0;
}
extern "C" SK_API const float* field_grid_bmin(const FieldGrid* g) { return g ? g->bmin : nullptr; }
extern "C" SK_API const float* field_grid_bmax(const FieldGrid* g) { return g ? g->bmax : nullptr; }
extern "C" SK_API float* field_grid_data_re_im(FieldGrid* g) {
    return g ? reinterpret_cast<float*>(g->data) : nullptr;
}

/* ── Stencil marcher (regular grid) ─────────────────────────────────────── */
/* These are per-band in-place updates on a copy; caller swaps buffers.
 * Implementations are reference-quality (clear, slow); SIMD/threaded port
 * is a follow-up.  Intentionally NOT hidden behind static_cast in headers
 * because the marcher API is currently only invoked from C++ helpers /
 * pybind, not from another translation unit.
 */

static inline int64_t _idx_reg(const FieldGrid* g, int b, int x, int y, int z) {
    int64_t cell = (static_cast<int64_t>(z) * g->dims[1] + y) * g->dims[0] + x;
    return static_cast<int64_t>(b) * g->n_cells_total + cell;
}

extern "C" SK_API int field_grid_step_helmholtz_regular(
    FieldGrid* g, const StencilSpec* spec, int n_steps)
{
    if (!g || !spec) return SK_ERR_NULL_STATE;
    if (g->kind != FIELD_GRID_REGULAR) return SK_ERR_NULL_STATE;
    const int nx = g->dims[0], ny = g->dims[1], nz = g->dims[2];
    if (nx < 3 || ny < 3 || nz < 3) return SK_OK;  /* nothing to do */

    /* Split-step Helmholtz: ψ_new = (1 - i k²Δt) ψ + i Δt ∇²ψ
     * where ∇² uses the requested stencil (defaults to 7-point).
     * Single working buffer per step; in-place layout-aware update. */
    std::vector<cd32> tmp(static_cast<size_t>(nx) * ny * nz);
    const float dxi2 = 1.f / (spec->dx * spec->dx);
    const float dyi2 = 1.f / (spec->dy * spec->dy);
    const float dzi2 = 1.f / (spec->dz * spec->dz);
    const cd32 i_dt(0.f, spec->dt);
    const cd32 i_k2dt(0.f, -spec->dt * (spec->k_real * spec->k_real
                                       - spec->k_imag * spec->k_imag));
    for (int step = 0; step < n_steps; ++step) {
        for (int b = 0; b < g->n_bands; ++b) {
            cd32* base = g->data + static_cast<int64_t>(b) * g->n_cells_total;
            std::memcpy(tmp.data(), base, sizeof(cd32) * tmp.size());
            for (int z = 1; z < nz - 1; ++z) {
                for (int y = 1; y < ny - 1; ++y) {
                    for (int x = 1; x < nx - 1; ++x) {
                        int64_t i = (static_cast<int64_t>(z) * ny + y) * nx + x;
                        cd32 c = tmp[i];
                        cd32 lap = (tmp[i+1] + tmp[i-1] - 2.f*c) * dxi2
                                 + (tmp[i+nx] + tmp[i-nx] - 2.f*c) * dyi2
                                 + (tmp[i+(int64_t)nx*ny] + tmp[i-(int64_t)nx*ny] - 2.f*c) * dzi2;
                        base[i] = c + i_k2dt * c + i_dt * lap;
                    }
                }
            }
        }
    }
    return SK_OK;
}

/* ── Endpoint accumulator (planeless, per group) ────────────────────────── */
/* These are thin helpers for the bidirectional integrator: callers append
 * EndpointRecord entries into a per-group flat buffer, no reduction. */
extern "C" SK_API int field_grid_inject_amplitude_regular(
    FieldGrid* g, int band,
    const float pos[3],
    float amp_re, float amp_im)
{
    if (!g || !pos || g->kind != FIELD_GRID_REGULAR) return SK_ERR_NULL_STATE;
    if (band < 0 || band >= g->n_bands) return SK_ERR_NULL_STATE;
    /* Trilinear deposit. */
    float u[3];
    for (int i = 0; i < 3; ++i) {
        float span = g->bmax[i] - g->bmin[i];
        if (span <= 0.f) return SK_OK;
        u[i] = (pos[i] - g->bmin[i]) / span * (g->dims[i] - 1);
        if (u[i] < 0 || u[i] >= g->dims[i]) return SK_OK;
    }
    int x0 = static_cast<int>(u[0]); int x1 = (x0 + 1 < g->dims[0]) ? x0 + 1 : x0;
    int y0 = static_cast<int>(u[1]); int y1 = (y0 + 1 < g->dims[1]) ? y0 + 1 : y0;
    int z0 = static_cast<int>(u[2]); int z1 = (z0 + 1 < g->dims[2]) ? z0 + 1 : z0;
    float fx = u[0] - x0, fy = u[1] - y0, fz = u[2] - z0;
    cd32 amp(amp_re, amp_im);
    auto add = [&](int x, int y, int z, float w) {
        if (w <= 0.f) return;
        g->data[_idx_reg(g, band, x, y, z)] += amp * w;
    };
    add(x0, y0, z0, (1-fx)*(1-fy)*(1-fz));
    add(x1, y0, z0,    fx *(1-fy)*(1-fz));
    add(x0, y1, z0, (1-fx)*   fy *(1-fz));
    add(x1, y1, z0,    fx *   fy *(1-fz));
    add(x0, y0, z1, (1-fx)*(1-fy)*   fz );
    add(x1, y0, z1,    fx *(1-fy)*   fz );
    add(x0, y1, z1, (1-fx)*   fy *   fz );
    add(x1, y1, z1,    fx *   fy *   fz );
    return SK_OK;
}

extern "C" SK_API int field_grid_inject_amplitude(
    FieldGrid* g, int band,
    const float pos[3],
    float amp_re, float amp_im)
{
    if (!g || !pos) return SK_ERR_NULL_STATE;
    if (band < 0 || band >= g->n_bands) return SK_ERR_NULL_STATE;

    if (g->kind == FIELD_GRID_REGULAR)
        return field_grid_inject_amplitude_regular(g, band, pos, amp_re, amp_im);

    if (g->kind != FIELD_GRID_KDTREE || g->nodes.empty())
        return SK_ERR_NULL_STATE;

    auto in_aabb = [&](const KdNode& nd) {
        return pos[0] >= nd.bmin[0] && pos[0] <= nd.bmax[0]
            && pos[1] >= nd.bmin[1] && pos[1] <= nd.bmax[1]
            && pos[2] >= nd.bmin[2] && pos[2] <= nd.bmax[2];
    };

    int node_id = 0;
    while (node_id >= 0 && node_id < static_cast<int>(g->nodes.size())) {
        const KdNode& nd = g->nodes[static_cast<size_t>(node_id)];
        if (!in_aabb(nd)) return SK_OK;
        if (nd.child_lo < 0 || nd.child_hi < 0) {
            const int lx = std::max(1, nd.leaf_dims[0]);
            const int ly = std::max(1, nd.leaf_dims[1]);
            const int lz = std::max(1, nd.leaf_dims[2]);
            const float sx = std::max(1.0e-12f, nd.bmax[0] - nd.bmin[0]);
            const float sy = std::max(1.0e-12f, nd.bmax[1] - nd.bmin[1]);
            const float sz = std::max(1.0e-12f, nd.bmax[2] - nd.bmin[2]);

            float ux = (pos[0] - nd.bmin[0]) / sx * (lx - 1);
            float uy = (pos[1] - nd.bmin[1]) / sy * (ly - 1);
            float uz = (pos[2] - nd.bmin[2]) / sz * (lz - 1);
            if (ux < 0.f || uy < 0.f || uz < 0.f || ux >= lx || uy >= ly || uz >= lz)
                return SK_OK;

            int x0 = static_cast<int>(ux), x1 = (x0 + 1 < lx) ? x0 + 1 : x0;
            int y0 = static_cast<int>(uy), y1 = (y0 + 1 < ly) ? y0 + 1 : y0;
            int z0 = static_cast<int>(uz), z1 = (z0 + 1 < lz) ? z0 + 1 : z0;
            float fx = ux - x0, fy = uy - y0, fz = uz - z0;

            cd32 amp(amp_re, amp_im);
            auto add_leaf = [&](int x, int y, int z, float w) {
                if (w <= 0.f) return;
                int64_t local = (static_cast<int64_t>(z) * ly + y) * lx + x;
                int64_t cell = nd.first_data + local;
                g->data[static_cast<int64_t>(band) * g->n_cells_total + cell] += amp * w;
            };

            add_leaf(x0, y0, z0, (1 - fx) * (1 - fy) * (1 - fz));
            add_leaf(x1, y0, z0,      fx  * (1 - fy) * (1 - fz));
            add_leaf(x0, y1, z0, (1 - fx) *     fy  * (1 - fz));
            add_leaf(x1, y1, z0,      fx  *     fy  * (1 - fz));
            add_leaf(x0, y0, z1, (1 - fx) * (1 - fy) *     fz );
            add_leaf(x1, y0, z1,      fx  * (1 - fy) *     fz );
            add_leaf(x0, y1, z1, (1 - fx) *     fy  *     fz );
            add_leaf(x1, y1, z1,      fx  *     fy  *     fz );
            return SK_OK;
        }

        int lo = nd.child_lo;
        int hi = nd.child_hi;
        bool in_lo = (lo >= 0 && lo < static_cast<int>(g->nodes.size()))
                  ? in_aabb(g->nodes[static_cast<size_t>(lo)]) : false;
        bool in_hi = (hi >= 0 && hi < static_cast<int>(g->nodes.size()))
                  ? in_aabb(g->nodes[static_cast<size_t>(hi)]) : false;

        if (in_lo && !in_hi) node_id = lo;
        else if (in_hi && !in_lo) node_id = hi;
        else if (in_lo) node_id = lo;
        else return SK_OK;
    }
    return SK_OK;
}
