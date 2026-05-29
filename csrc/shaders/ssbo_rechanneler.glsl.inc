/* ssbo_rechanneler.glsl.inc ─────────────────────────────────────────────────
 * Arbitrary SSBO re-channeler framework  ·  spectral-analyzer GPU pipeline
 *
 * SINGLE-SOURCE COUNTERPART
 * ─────────────────────────
 * This file is the GPU (GLSL) half of the rechanneler.  The authoritative
 * API lives in:
 *
 *   csrc/include/ssbo_rechanneler.h   — C struct/function declarations
 *   csrc/kernels/ssbo_rechanneler.c   — CPU execution + GLSL code generator
 *   csrc/bindings/pybind_rechanneler.cpp — Python (pybind11) bindings
 *
 * The GLSL code generator (rchan_emit_glsl) produces a shader that #defines
 * all RCHAN_ macros and then inlines this file, so any generated shader is
 * a direct transcript of the C descriptor.  Do not add any logic here that
 * is not mirrored in ssbo_rechanneler.c's rchan_execute_cpu() — keep them
 * in lockstep.
 *
 * PURPOSE
 * ───────
 * A parameterised preamble for any compute shader that reshapes, repacks, or
 * fans data between SSBOs within the hard 8-binding limit imposed by
 * GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS = 8 (bindings 0–7, non-negotiable).
 *
 * In the T5 upgrade path this is used as a dedicated pre-pass:
 *
 *   C++ sorts and co-locates vertex + spectral + PDF records by
 *   (subpath_id, vertex_index) key into input staging SSBOs.
 *          ↓
 *   Rechanneler GPU pass reshapes them into extended T5LightVertBuf /
 *   T5CamVertBuf with new, larger strides carrying per-band betas, PDF
 *   scalars, optical flags, and MIS prefix products.
 *          ↓
 *   T5 shader reads tightly-packed per-vertex records with no hash lookup,
 *   no cross-buffer scatter, and full per-band spectral + MIS fidelity.
 *
 * DESIGN RULES
 * ────────────
 *  • All slot indices (n in RCHAN_SRC_F(n,...)) MUST be integer literals at
 *    the call site.  The ## token-paste macros require compile-time constants.
 *  • sum(all RCHAN_SRC_BINDING_N + RCHAN_DST_BINDING_N + RCHAN_DST_ATOMIC_N
 *        + optional RCHAN_SCATTER_BINDING) MUST be ≤ 8.  The framework cannot
 *    enforce this; you will get a GL driver error at link time if violated.
 *  • Source SSBOs are declared readonly float[].
 *  • Dest SSBOs are float[] (regular writes, RCHAN_DST_WF/WU/WI) or uint[]
 *    (RCHAN_DST_ATOMIC_N, required for atomic float-CAS via RCHAN_DST_AF).
 *  • The framework emits layout(local_size_x = RCHAN_LOCAL_SIZE).  Do NOT
 *    redeclare a local-size layout in the including shader.
 *
 * CONFIGURATION — define before #include ────────────────────────────────────
 *
 *   Source slots (readonly float[], up to 8 total):
 *     RCHAN_SRC_BINDING_N    – GL binding point for source slot N  (N = 0..7)
 *     RCHAN_SRC_STRIDE_N     – floats per record (compile-time constant)
 *     RCHAN_SRC_BASE_N       – (optional) name of a uniform uint giving a
 *                              flat-float index offset into the buffer.
 *                              Use for multi-region buffers like BdptOutputBuf
 *                              (verts / spectral / PDF in one flat array at
 *                              different uniform-supplied offsets).
 *                              Defaults to 0u if not defined.
 *
 *   Destination slots — regular writes (coherent float[], up to 8 total):
 *     RCHAN_DST_BINDING_N    – GL binding point for dest slot N    (N = 0..7)
 *     RCHAN_DST_STRIDE_N     – floats per record (compile-time constant)
 *     RCHAN_DST_BASE_N       – (optional) uniform uint flat-float base offset.
 *                              Defaults to 0u if not defined.
 *
 *   Destination slots — atomic writes (coherent uint[], up to 8 total):
 *     RCHAN_DST_ATOMIC_BINDING_N – binding point for atomic dest slot N
 *     RCHAN_DST_ATOMIC_STRIDE_N  – uints per record
 *     RCHAN_DST_ATOMIC_BASE_N    – (optional) uniform uint base offset
 *
 *   Scatter index buffer (optional, uint[], counts toward 8-binding total):
 *     RCHAN_SCATTER_BINDING  – binding point.  Provides RCHAN_SCATTER_IDX(i)
 *                              which returns the destination record index for
 *                              source record i.  When undefined, the identity
 *                              mapping is used (RCHAN_SCATTER_IDX(i) == i).
 *
 *   Dispatch geometry:
 *     RCHAN_LOCAL_SIZE       – workgroup x size (default 64)
 *     RCHAN_PROVIDE_MAIN     – define to emit main() that calls
 *                              rchan_map(rec_i) for each in-range record.
 *
 * ACCESSOR MACROS ──────────────────────────────────────────────────────────
 *
 *   Read from source slot n, record rec, field f (float index within record):
 *     RCHAN_SRC_F(n, rec, f)         → float
 *     RCHAN_SRC_U(n, rec, f)         → uint   (floatBitsToUint reinterpret)
 *     RCHAN_SRC_I(n, rec, f)         → int    (floatBitsToInt  reinterpret)
 *
 *   Write to regular dest slot n:
 *     RCHAN_DST_WF(n, rec, f, v)     – float  write
 *     RCHAN_DST_WU(n, rec, f, v)     – uint   write (stored as float bits)
 *     RCHAN_DST_WI(n, rec, f, v)     – int    write (stored as float bits)
 *
 *   Atomic float-CAS add to atomic dest slot n (uint[] backing):
 *     RCHAN_DST_AF(n, rec, f, v)     – float accumulation, no ext required
 *     RCHAN_DST_AU(n, rec, f, v)     – atomicAdd uint
 *
 *   Bulk copy — cnt consecutive float fields from src→dst at same record:
 *     RCHAN_COPY_RANGE(sn, dn, rec, f0, cnt)
 *
 *   Cross-slot bulk copy — src and dst records may differ (e.g. scatter):
 *     RCHAN_COPY_RANGE_XREC(sn, src_rec, dn, dst_rec, f0, cnt)
 *
 *   Scatter destination index:
 *     RCHAN_SCATTER_IDX(i)   → uint  dest record index for source record i
 *
 * DISPATCH UNIFORMS ────────────────────────────────────────────────────────
 *
 *   uniform uint rchan_n_records;    // records this dispatch covers
 *   uniform uint rchan_record_base;  // global index of the first record
 *                                    // Tile by re-dispatching with new base.
 *
 * MULTI-PASS TILING ────────────────────────────────────────────────────────
 *
 *   For record counts that exceed a single workgroup budget (TDR avoidance):
 *     dispatch_count = ceil(tile_size / RCHAN_LOCAL_SIZE)
 *     for each tile t:
 *         upload rchan_record_base = t * tile_size
 *         upload rchan_n_records   = min(tile_size, total - t*tile_size)
 *         glDispatchCompute(dispatch_count, 1, 1)
 *
 * USER HOOK ────────────────────────────────────────────────────────────────
 *
 *   void rchan_map(uint rec_i);
 *     rec_i is the GLOBAL record index (rchan_record_base + local offset).
 *     Define this function after including this header.
 *     Use RCHAN_SRC_F / RCHAN_DST_WF / etc. inside it.
 *
 * EXAMPLE ──────────────────────────────────────────────────────────────────
 *
 *   // Re-pack T5 light verts: source slot 0 = old BdptVertBuf (stride 28),
 *   //                         source slot 1 = per-vertex PDF staging (stride 8)
 *   //                         dest   slot 0 = new T5LightVertBuf  (stride 40)
 *
 *   #define RCHAN_SRC_BINDING_0  0
 *   #define RCHAN_SRC_STRIDE_0   28
 *   #define RCHAN_SRC_BINDING_1  1
 *   #define RCHAN_SRC_STRIDE_1   8
 *   #define RCHAN_DST_BINDING_0  2
 *   #define RCHAN_DST_STRIDE_0   40
 *   #define RCHAN_LOCAL_SIZE     64
 *   #define RCHAN_PROVIDE_MAIN
 *   #include "ssbo_rechanneler.glsl.inc"
 *
 *   void rchan_map(uint rec_i) {
 *       // copy pos+normal+flags from bdpt vert (fields 7..12) to new vert
 *       RCHAN_COPY_RANGE(0, 0, rec_i, 7, 6);          // pos xyz + normal xyz
 *       RCHAN_DST_WF(0, rec_i, 6, RCHAN_SRC_F(0, rec_i, 25));  // throughput
 *       RCHAN_DST_WU(0, rec_i, 7, RCHAN_SRC_U(0, rec_i,  2));  // flags
 *       // pdf fields from staging buffer
 *       RCHAN_DST_WF(0, rec_i, 12, RCHAN_SRC_F(1, rec_i, 0));  // pdf_fwd
 *       RCHAN_DST_WF(0, rec_i, 13, RCHAN_SRC_F(1, rec_i, 1));  // pdf_rev
 *       // ... etc.
 *   }
 * ─────────────────────────────────────────────────────────────────────────── */

#ifndef SSBO_RECHANNELER_INC
#define SSBO_RECHANNELER_INC

/* ── Local size ─────────────────────────────────────────────────────────── */
#ifndef RCHAN_LOCAL_SIZE
#  define RCHAN_LOCAL_SIZE 64
#endif
layout(local_size_x = RCHAN_LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

/* ── Dispatch uniforms ───────────────────────────────────────────────────── */
uniform uint rchan_n_records;    /* records covered by this dispatch          */
uniform uint rchan_record_base;  /* first global record index for this tile   */

/* ─────────────────────────────────────────────────────────────────────────── *
 * SOURCE SSBO DECLARATIONS — readonly flat float[]                           *
 * ─────────────────────────────────────────────────────────────────────────── */

#ifdef RCHAN_SRC_BINDING_0
#  ifndef RCHAN_SRC_STRIDE_0
#    error "[rechanneler] RCHAN_SRC_STRIDE_0 required when RCHAN_SRC_BINDING_0 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_0) readonly buffer _RchanSrc0 { float rchan_src0[]; };
#  ifndef RCHAN_SRC_BASE_0
#    define RCHAN_SRC_BASE_0 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_1
#  ifndef RCHAN_SRC_STRIDE_1
#    error "[rechanneler] RCHAN_SRC_STRIDE_1 required when RCHAN_SRC_BINDING_1 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_1) readonly buffer _RchanSrc1 { float rchan_src1[]; };
#  ifndef RCHAN_SRC_BASE_1
#    define RCHAN_SRC_BASE_1 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_2
#  ifndef RCHAN_SRC_STRIDE_2
#    error "[rechanneler] RCHAN_SRC_STRIDE_2 required when RCHAN_SRC_BINDING_2 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_2) readonly buffer _RchanSrc2 { float rchan_src2[]; };
#  ifndef RCHAN_SRC_BASE_2
#    define RCHAN_SRC_BASE_2 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_3
#  ifndef RCHAN_SRC_STRIDE_3
#    error "[rechanneler] RCHAN_SRC_STRIDE_3 required when RCHAN_SRC_BINDING_3 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_3) readonly buffer _RchanSrc3 { float rchan_src3[]; };
#  ifndef RCHAN_SRC_BASE_3
#    define RCHAN_SRC_BASE_3 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_4
#  ifndef RCHAN_SRC_STRIDE_4
#    error "[rechanneler] RCHAN_SRC_STRIDE_4 required when RCHAN_SRC_BINDING_4 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_4) readonly buffer _RchanSrc4 { float rchan_src4[]; };
#  ifndef RCHAN_SRC_BASE_4
#    define RCHAN_SRC_BASE_4 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_5
#  ifndef RCHAN_SRC_STRIDE_5
#    error "[rechanneler] RCHAN_SRC_STRIDE_5 required when RCHAN_SRC_BINDING_5 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_5) readonly buffer _RchanSrc5 { float rchan_src5[]; };
#  ifndef RCHAN_SRC_BASE_5
#    define RCHAN_SRC_BASE_5 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_6
#  ifndef RCHAN_SRC_STRIDE_6
#    error "[rechanneler] RCHAN_SRC_STRIDE_6 required when RCHAN_SRC_BINDING_6 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_6) readonly buffer _RchanSrc6 { float rchan_src6[]; };
#  ifndef RCHAN_SRC_BASE_6
#    define RCHAN_SRC_BASE_6 0u
#  endif
#endif

#ifdef RCHAN_SRC_BINDING_7
#  ifndef RCHAN_SRC_STRIDE_7
#    error "[rechanneler] RCHAN_SRC_STRIDE_7 required when RCHAN_SRC_BINDING_7 is set"
#  endif
layout(std430, binding = RCHAN_SRC_BINDING_7) readonly buffer _RchanSrc7 { float rchan_src7[]; };
#  ifndef RCHAN_SRC_BASE_7
#    define RCHAN_SRC_BASE_7 0u
#  endif
#endif

/* ─────────────────────────────────────────────────────────────────────────── *
 * DESTINATION SSBO DECLARATIONS — coherent flat float[] (regular writes)     *
 * ─────────────────────────────────────────────────────────────────────────── */

#ifdef RCHAN_DST_BINDING_0
#  ifndef RCHAN_DST_STRIDE_0
#    error "[rechanneler] RCHAN_DST_STRIDE_0 required when RCHAN_DST_BINDING_0 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_0) coherent buffer _RchanDst0 { float rchan_dst0[]; };
#  ifndef RCHAN_DST_BASE_0
#    define RCHAN_DST_BASE_0 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_1
#  ifndef RCHAN_DST_STRIDE_1
#    error "[rechanneler] RCHAN_DST_STRIDE_1 required when RCHAN_DST_BINDING_1 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_1) coherent buffer _RchanDst1 { float rchan_dst1[]; };
#  ifndef RCHAN_DST_BASE_1
#    define RCHAN_DST_BASE_1 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_2
#  ifndef RCHAN_DST_STRIDE_2
#    error "[rechanneler] RCHAN_DST_STRIDE_2 required when RCHAN_DST_BINDING_2 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_2) coherent buffer _RchanDst2 { float rchan_dst2[]; };
#  ifndef RCHAN_DST_BASE_2
#    define RCHAN_DST_BASE_2 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_3
#  ifndef RCHAN_DST_STRIDE_3
#    error "[rechanneler] RCHAN_DST_STRIDE_3 required when RCHAN_DST_BINDING_3 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_3) coherent buffer _RchanDst3 { float rchan_dst3[]; };
#  ifndef RCHAN_DST_BASE_3
#    define RCHAN_DST_BASE_3 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_4
#  ifndef RCHAN_DST_STRIDE_4
#    error "[rechanneler] RCHAN_DST_STRIDE_4 required when RCHAN_DST_BINDING_4 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_4) coherent buffer _RchanDst4 { float rchan_dst4[]; };
#  ifndef RCHAN_DST_BASE_4
#    define RCHAN_DST_BASE_4 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_5
#  ifndef RCHAN_DST_STRIDE_5
#    error "[rechanneler] RCHAN_DST_STRIDE_5 required when RCHAN_DST_BINDING_5 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_5) coherent buffer _RchanDst5 { float rchan_dst5[]; };
#  ifndef RCHAN_DST_BASE_5
#    define RCHAN_DST_BASE_5 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_6
#  ifndef RCHAN_DST_STRIDE_6
#    error "[rechanneler] RCHAN_DST_STRIDE_6 required when RCHAN_DST_BINDING_6 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_6) coherent buffer _RchanDst6 { float rchan_dst6[]; };
#  ifndef RCHAN_DST_BASE_6
#    define RCHAN_DST_BASE_6 0u
#  endif
#endif

#ifdef RCHAN_DST_BINDING_7
#  ifndef RCHAN_DST_STRIDE_7
#    error "[rechanneler] RCHAN_DST_STRIDE_7 required when RCHAN_DST_BINDING_7 is set"
#  endif
layout(std430, binding = RCHAN_DST_BINDING_7) coherent buffer _RchanDst7 { float rchan_dst7[]; };
#  ifndef RCHAN_DST_BASE_7
#    define RCHAN_DST_BASE_7 0u
#  endif
#endif

/* ─────────────────────────────────────────────────────────────────────────── *
 * ATOMIC DESTINATION SSBO DECLARATIONS — coherent flat uint[]                *
 *                                                                             *
 * Required for float-CAS atomic accumulation (e.g. pixel buffers).           *
 * Use RCHAN_DST_AF / RCHAN_DST_AU with these slots.                          *
 * Regular float writes via RCHAN_DST_WF still work (stored as floatBitsToUint).
 * ─────────────────────────────────────────────────────────────────────────── */

#ifdef RCHAN_DST_ATOMIC_BINDING_0
#  ifndef RCHAN_DST_ATOMIC_STRIDE_0
#    error "[rechanneler] RCHAN_DST_ATOMIC_STRIDE_0 required"
#  endif
layout(std430, binding = RCHAN_DST_ATOMIC_BINDING_0) coherent buffer _RchanDstA0 { uint rchan_adst0[]; };
#  ifndef RCHAN_DST_ATOMIC_BASE_0
#    define RCHAN_DST_ATOMIC_BASE_0 0u
#  endif
#endif

#ifdef RCHAN_DST_ATOMIC_BINDING_1
#  ifndef RCHAN_DST_ATOMIC_STRIDE_1
#    error "[rechanneler] RCHAN_DST_ATOMIC_STRIDE_1 required"
#  endif
layout(std430, binding = RCHAN_DST_ATOMIC_BINDING_1) coherent buffer _RchanDstA1 { uint rchan_adst1[]; };
#  ifndef RCHAN_DST_ATOMIC_BASE_1
#    define RCHAN_DST_ATOMIC_BASE_1 0u
#  endif
#endif

#ifdef RCHAN_DST_ATOMIC_BINDING_2
#  ifndef RCHAN_DST_ATOMIC_STRIDE_2
#    error "[rechanneler] RCHAN_DST_ATOMIC_STRIDE_2 required"
#  endif
layout(std430, binding = RCHAN_DST_ATOMIC_BINDING_2) coherent buffer _RchanDstA2 { uint rchan_adst2[]; };
#  ifndef RCHAN_DST_ATOMIC_BASE_2
#    define RCHAN_DST_ATOMIC_BASE_2 0u
#  endif
#endif

#ifdef RCHAN_DST_ATOMIC_BINDING_3
#  ifndef RCHAN_DST_ATOMIC_STRIDE_3
#    error "[rechanneler] RCHAN_DST_ATOMIC_STRIDE_3 required"
#  endif
layout(std430, binding = RCHAN_DST_ATOMIC_BINDING_3) coherent buffer _RchanDstA3 { uint rchan_adst3[]; };
#  ifndef RCHAN_DST_ATOMIC_BASE_3
#    define RCHAN_DST_ATOMIC_BASE_3 0u
#  endif
#endif

/* ─────────────────────────────────────────────────────────────────────────── *
 * SCATTER INDEX BUFFER (optional) — readonly uint[]                          *
 *                                                                             *
 * Maps source record i to a (potentially different) destination record index.*
 * If not defined, RCHAN_SCATTER_IDX(i) is the identity: same index.          *
 * ─────────────────────────────────────────────────────────────────────────── */

#ifdef RCHAN_SCATTER_BINDING
layout(std430, binding = RCHAN_SCATTER_BINDING) readonly buffer _RchanScatter {
    uint rchan_scatter_idx[];
};
#  define RCHAN_SCATTER_IDX(i)   rchan_scatter_idx[(i)]
#else
#  define RCHAN_SCATTER_IDX(i)   (i)
#endif

/* ─────────────────────────────────────────────────────────────────────────── *
 * FIELD INDEX HELPERS                                                         *
 * ─────────────────────────────────────────────────────────────────────────── */

/* Flat index into a source buffer given record and field */
#define _RSIDX(n, rec, f)  (RCHAN_SRC_BASE_##n  + uint(rec) * uint(RCHAN_SRC_STRIDE_##n)  + uint(f))
/* Flat index into a regular dest buffer */
#define _RDIDX(n, rec, f)  (RCHAN_DST_BASE_##n  + uint(rec) * uint(RCHAN_DST_STRIDE_##n)  + uint(f))
/* Flat index into an atomic dest buffer */
#define _RAIDX(n, rec, f)  (RCHAN_DST_ATOMIC_BASE_##n + uint(rec) * uint(RCHAN_DST_ATOMIC_STRIDE_##n) + uint(f))

/* ─────────────────────────────────────────────────────────────────────────── *
 * READ ACCESSORS (token-paste on literal n)                                  *
 * ─────────────────────────────────────────────────────────────────────────── */

/* Float read */
#define RCHAN_SRC_F(n, rec, f)   (rchan_src##n[_RSIDX(n, rec, f)])
/* Uint reinterpret read */
#define RCHAN_SRC_U(n, rec, f)   floatBitsToUint(RCHAN_SRC_F(n, rec, f))
/* Int reinterpret read */
#define RCHAN_SRC_I(n, rec, f)   floatBitsToInt(RCHAN_SRC_F(n, rec, f))

/* ─────────────────────────────────────────────────────────────────────────── *
 * WRITE ACCESSORS — regular dest (float[])                                   *
 * ─────────────────────────────────────────────────────────────────────────── */

/* Float write */
#define RCHAN_DST_WF(n, rec, f, v)  (rchan_dst##n[_RDIDX(n, rec, f)] = float(v))
/* Uint write stored as float bits */
#define RCHAN_DST_WU(n, rec, f, v)  (rchan_dst##n[_RDIDX(n, rec, f)] = uintBitsToFloat(uint(v)))
/* Int write stored as float bits */
#define RCHAN_DST_WI(n, rec, f, v)  (rchan_dst##n[_RDIDX(n, rec, f)] = intBitsToFloat(int(v)))

/* ─────────────────────────────────────────────────────────────────────────── *
 * WRITE ACCESSORS — atomic dest (uint[])                                     *
 * ─────────────────────────────────────────────────────────────────────────── */

/* Direct uint atomic add */
#define RCHAN_DST_AU(n, rec, f, v)  atomicAdd(rchan_adst##n[_RAIDX(n, rec, f)], uint(v))

/* Atomic float accumulation — CAS spin loop (no extension required).
 * Spins up to RCHAN_CAS_ITERS times.  For high-contention pixels consider
 * reducing contention by sub-tiling writes before calling this.              */
#ifndef RCHAN_CAS_ITERS
#  define RCHAN_CAS_ITERS 128
#endif

/* Per atomic-dest slot: one inlined CAS helper function.
 * Defined only if the corresponding RCHAN_DST_ATOMIC_BINDING_N is declared.
 * Called by RCHAN_DST_AF(n, rec, f, v).                                      */

#ifdef RCHAN_DST_ATOMIC_BINDING_0
void _rchan_af0(uint idx, float val) {
    uint exp = rchan_adst0[idx];
    for (int _ci = 0; _ci < RCHAN_CAS_ITERS; ++_ci) {
        uint des = floatBitsToUint(uintBitsToFloat(exp) + val);
        uint act = atomicCompSwap(rchan_adst0[idx], exp, des);
        if (act == exp) return;
        exp = act;
    }
}
#endif

#ifdef RCHAN_DST_ATOMIC_BINDING_1
void _rchan_af1(uint idx, float val) {
    uint exp = rchan_adst1[idx];
    for (int _ci = 0; _ci < RCHAN_CAS_ITERS; ++_ci) {
        uint des = floatBitsToUint(uintBitsToFloat(exp) + val);
        uint act = atomicCompSwap(rchan_adst1[idx], exp, des);
        if (act == exp) return;
        exp = act;
    }
}
#endif

#ifdef RCHAN_DST_ATOMIC_BINDING_2
void _rchan_af2(uint idx, float val) {
    uint exp = rchan_adst2[idx];
    for (int _ci = 0; _ci < RCHAN_CAS_ITERS; ++_ci) {
        uint des = floatBitsToUint(uintBitsToFloat(exp) + val);
        uint act = atomicCompSwap(rchan_adst2[idx], exp, des);
        if (act == exp) return;
        exp = act;
    }
}
#endif

#ifdef RCHAN_DST_ATOMIC_BINDING_3
void _rchan_af3(uint idx, float val) {
    uint exp = rchan_adst3[idx];
    for (int _ci = 0; _ci < RCHAN_CAS_ITERS; ++_ci) {
        uint des = floatBitsToUint(uintBitsToFloat(exp) + val);
        uint act = atomicCompSwap(rchan_adst3[idx], exp, des);
        if (act == exp) return;
        exp = act;
    }
}
#endif

/* Dispatch macro — resolves to the per-slot CAS helper */
#define RCHAN_DST_AF(n, rec, f, v)  _rchan_af##n(_RAIDX(n, rec, f), float(v))

/* ─────────────────────────────────────────────────────────────────────────── *
 * BULK COPY HELPERS                                                           *
 * ─────────────────────────────────────────────────────────────────────────── */

/* Copy cnt consecutive float fields f0..f0+cnt-1 from src slot sn to dst
 * slot dn, same record index on both sides.
 * cnt should be a compile-time constant for loop unrolling.                  */
#define RCHAN_COPY_RANGE(sn, dn, rec, f0, cnt) \
    do { for (int _rcf = 0; _rcf < int(cnt); ++_rcf) \
        RCHAN_DST_WF(dn, rec, int(f0) + _rcf, RCHAN_SRC_F(sn, rec, int(f0) + _rcf)); \
    } while (false)

/* Cross-record bulk copy — src and dst record indices differ (scatter mode) */
#define RCHAN_COPY_RANGE_XREC(sn, src_rec, dn, dst_rec, f0, cnt) \
    do { for (int _rcf = 0; _rcf < int(cnt); ++_rcf) \
        RCHAN_DST_WF(dn, dst_rec, int(f0) + _rcf, RCHAN_SRC_F(sn, src_rec, int(f0) + _rcf)); \
    } while (false)

/* Copy from src slot sn field fs to dst slot dn field fd, different field
 * offsets, same record.  Useful when layouts differ structurally.            */
#define RCHAN_COPY_FIELD(sn, fs, dn, fd, rec) \
    RCHAN_DST_WF(dn, rec, fd, RCHAN_SRC_F(sn, rec, fs))

/* Copy a contiguous src range to a non-overlapping dst range at a different
 * field offset within the same record.                                        */
#define RCHAN_COPY_RANGE_REMAP(sn, dn, rec, f_src, f_dst, cnt) \
    do { for (int _rcf = 0; _rcf < int(cnt); ++_rcf) \
        RCHAN_DST_WF(dn, rec, int(f_dst) + _rcf, RCHAN_SRC_F(sn, rec, int(f_src) + _rcf)); \
    } while (false)

/* ─────────────────────────────────────────────────────────────────────────── *
 * MULTI-SOURCE FAN-IN HELPER                                                  *
 *                                                                             *
 * Read N float fields from up to 4 source slots (interleaved or blocked)     *
 * and write them contiguously into one destination slot starting at f_dst.   *
 *                                                                             *
 * Usage: fan-in 4 fields from src0 and 4 fields from src1 into dst0:         *
 *   RCHAN_FANIN_2(0, 0, 4,  1, 0, 4,  0, 0, rec)                            *
 *     sn0=0 f_src0=0 cnt0=4, sn1=1 f_src1=0 cnt1=4, dn=0, f_dst=0          *
 * ─────────────────────────────────────────────────────────────────────────── */

#define RCHAN_FANIN_2(sn0, f_src0, cnt0, sn1, f_src1, cnt1, dn, f_dst, rec) \
    do { \
        int _fbase = int(f_dst); \
        for (int _ri = 0; _ri < int(cnt0); ++_ri) \
            RCHAN_DST_WF(dn, rec, _fbase + _ri, RCHAN_SRC_F(sn0, rec, int(f_src0) + _ri)); \
        _fbase += int(cnt0); \
        for (int _ri = 0; _ri < int(cnt1); ++_ri) \
            RCHAN_DST_WF(dn, rec, _fbase + _ri, RCHAN_SRC_F(sn1, rec, int(f_src1) + _ri)); \
    } while (false)

#define RCHAN_FANIN_3(sn0, f0, c0, sn1, f1, c1, sn2, f2, c2, dn, f_dst, rec) \
    do { \
        int _fbase = int(f_dst); \
        for (int _ri = 0; _ri < int(c0); ++_ri) \
            RCHAN_DST_WF(dn, rec, _fbase + _ri, RCHAN_SRC_F(sn0, rec, int(f0) + _ri)); \
        _fbase += int(c0); \
        for (int _ri = 0; _ri < int(c1); ++_ri) \
            RCHAN_DST_WF(dn, rec, _fbase + _ri, RCHAN_SRC_F(sn1, rec, int(f1) + _ri)); \
        _fbase += int(c1); \
        for (int _ri = 0; _ri < int(c2); ++_ri) \
            RCHAN_DST_WF(dn, rec, _fbase + _ri, RCHAN_SRC_F(sn2, rec, int(f2) + _ri)); \
    } while (false)

/* ─────────────────────────────────────────────────────────────────────────── *
 * MULTI-REGION BUFFER HELPERS                                                 *
 *                                                                             *
 * For buffers like BdptOutputBuf where multiple logical arrays share one     *
 * SSBO.  The base offsets are supplied as uniforms (see RCHAN_SRC_BASE_N).   *
 *                                                                             *
 * If your src slot 0 is BdptOutputBuf with three regions:                    *
 *   - vertex region:   base = 0                (or a uniform offset)         *
 *   - spectral region: base = bdpt_spectral_base                             *
 *   - pdf region:      base = bdpt_pdf_base                                  *
 *                                                                             *
 * Bind it THREE times as RCHAN_SRC_BINDING_0/1/2 all pointing to the same   *
 * GL buffer object (glBindBufferBase can alias the same buffer to multiple   *
 * binding points), with different RCHAN_SRC_BASE_N uniforms.                 *
 *                                                                             *
 * Then RCHAN_SRC_F(0, vi, f) reads vertex data,                              *
 *      RCHAN_SRC_F(1, si, f) reads spectral data,                            *
 *      RCHAN_SRC_F(2, pi, f) reads pdf data.                                 *
 *                                                                             *
 * Note: binding the same buffer object to multiple binding points is valid   *
 * per GL 4.3 spec §7.7.  It counts as 3 of your 8 binding slots.            *
 * ─────────────────────────────────────────────────────────────────────────── */

/* ─────────────────────────────────────────────────────────────────────────── *
 * OPTIONAL BUILT-IN MAIN                                                      *
 * ─────────────────────────────────────────────────────────────────────────── */

/* Forward-declare the user hook (must be defined after this include) */
void rchan_map(uint rec_i);

#ifdef RCHAN_PROVIDE_MAIN
void main() {
    /* Global invocation index within this tile */
    uint local_idx = gl_GlobalInvocationID.x;
    if (local_idx >= rchan_n_records) return;

    /* Absolute record index */
    uint rec_i = rchan_record_base + local_idx;

    rchan_map(rec_i);
}
#endif /* RCHAN_PROVIDE_MAIN */

#endif /* SSBO_RECHANNELER_INC */
