/**
 * field_grid.h — Dense complex spectral field on regular or k-d AMR grids.
 *
 * Implements the storage backbone for the "second file" mandated by the
 * integrator-rewrite directive: complete field descriptions on a grid that
 * the marching/volumetric routines (field_march.cpp) can consume.
 *
 * Two grid kinds are supported behind one struct:
 *
 *   FIELD_GRID_REGULAR — dims[3] uniform Cartesian cells, contiguous SoA
 *                        (n_bands, dz, dy, dx) of std::complex<float>.
 *
 *   FIELD_GRID_KDTREE  — explicit k-d tree subdivision.  Each leaf holds
 *                        (n_bands, lz, ly, lx) of std::complex<float> with
 *                        per-leaf dimensions chosen by the splitter.  This
 *                        replaces the old octree AMR plan; k-d gives finer
 *                        anisotropic refinement (e.g. high-res near a focal
 *                        plane along z, coarse laterally), which matches the
 *                        optical use case better than an octree.
 *
 * Stencils are described separately (StencilSpec) so the marcher can be
 * configured per-call without re-allocating the grid.
 *
 * Hard rules:
 *   - Storage is std::complex<float>.  No magnitudes, no quantize.
 *   - n_bands is preserved from the source MaterialDatabase, NOT collapsed
 *     into a smaller "layer" set.
 */
#pragma once
#include "serial_kernel.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Grid kind tag. */
#define FIELD_GRID_REGULAR   0
#define FIELD_GRID_KDTREE    1

/* Stencil enumeration — selects the discrete operator used by field_march. */
#define STENCIL_LAPLACIAN_3PT   0   /* 1D Laplacian, axis-only                */
#define STENCIL_LAPLACIAN_7PT   1   /* 3D 7-point cross stencil              */
#define STENCIL_LAPLACIAN_27PT  2   /* 3D 27-point isotropic stencil         */
#define STENCIL_YEE_TE          3   /* Yee FDTD, transverse-electric        */
#define STENCIL_YEE_TM          4   /* Yee FDTD, transverse-magnetic        */
#define STENCIL_BIHARMONIC_5PT  5   /* 1D biharmonic, axis-only              */
#define STENCIL_HELMHOLTZ_7PT   6   /* (∇² + k²)ψ split-step                 */

typedef struct {
    int   kind;        /* STENCIL_*                                          */
    float dx;          /* metre spacing along x (regular), unused for KD     */
    float dy;
    float dz;
    float dt;          /* time step (seconds)                                */
    float k_real;      /* per-step wavenumber if kind=Helmholtz, else unused */
    float k_imag;
} StencilSpec;

/* k-d tree node — flat array, child indices are absolute into the array.
 * Leaves have first_data >= 0 and child_lo == child_hi == -1.
 */
typedef struct {
    float   bmin[3];
    float   bmax[3];
    int32_t child_lo;        /* -1 if leaf                                  */
    int32_t child_hi;        /* -1 if leaf                                  */
    int32_t split_axis;      /* 0=x,1=y,2=z; -1 if leaf                     */
    float   split_pos;
    int32_t leaf_dims[3];    /* (lx, ly, lz) cells inside this leaf         */
    int64_t first_data;      /* base offset into FieldGrid::data (units:    */
                             /*   complex<float>); -1 if internal node      */
} KdNode;

/**
 * FieldGrid — opaque-ish container.  The C++ side allocates ``data`` as a
 * single contiguous std::complex<float> array; the layout depends on kind.
 *
 * For REGULAR: data has length n_bands * dims[0] * dims[1] * dims[2], stored
 * (band, z, y, x) in row-major order.
 *
 * For KDTREE: data length = n_bands * Σ(leaf_dims.x * leaf_dims.y * leaf_dims.z)
 * across all leaves; each leaf's chunk starts at first_data and is band-major
 * within that chunk.
 */
typedef struct FieldGrid FieldGrid;

/* Construction ----------------------------------------------------------- */
SK_API FieldGrid* field_grid_create_regular(
    int n_bands, int nx, int ny, int nz,
    const float bmin[3], const float bmax[3]);

SK_API FieldGrid* field_grid_create_kdtree(
    int n_bands,
    const KdNode* nodes, int n_nodes);

SK_API void field_grid_destroy(FieldGrid* g);

/* Introspection ---------------------------------------------------------- */
SK_API int   field_grid_kind     (const FieldGrid* g);
SK_API int   field_grid_n_bands  (const FieldGrid* g);
SK_API int64_t field_grid_n_cells_total(const FieldGrid* g);
SK_API const float* field_grid_bmin(const FieldGrid* g);
SK_API const float* field_grid_bmax(const FieldGrid* g);

/**
 * Raw data pointer — std::complex<float> *, length field_grid_n_cells_total
 * × n_bands.  Caller may read/write directly (used by Python NumPy view).
 */
SK_API float* field_grid_data_re_im(FieldGrid* g);  /* interleaved (re,im,re,im,...) */

#ifdef __cplusplus
} /* extern "C" */
#endif
