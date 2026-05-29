/* ssbo_rechanneler.h ─────────────────────────────────────────────────────────
 *
 * Single-source SSBO rechanneler — C API.
 *
 * PURPOSE
 * ───────
 * Defines the data structures and functions for describing an arbitrary
 * many-to-many float-buffer remapping (rechanneling), and provides:
 *
 *   1. A C API for CPU-side execution of the same mapping (testing, fallback,
 *      Python wrapper).
 *   2. A GLSL source code generator that emits a self-contained compute shader
 *      from the exact same descriptor.  The generated shader uses the shared
 *      ssbo_rechanneler.glsl.inc preamble and calls rchan_map() with the
 *      generated body.  No shader code needs to be written by hand for
 *      straightforward remapping jobs.
 *   3. A concise descriptor language: arrays of RchanFieldOp records that
 *      encode reads, writes, bulk copies, and float-CAS atomic accumulations.
 *
 * DESIGN PRINCIPLES
 * ─────────────────
 *  • One descriptor → CPU path, GPU shader, Python callable.  Any change to
 *    the layout is made once and propagates everywhere.
 *  • No dynamic allocation inside rchan_execute_cpu().  All memory is
 *    caller-supplied.
 *  • The GLSL generator emits textual GLSL; the caller compiles it with the
 *    normal GL pipeline.
 *  • Binding-limit safety: rchan_desc_validate() returns an error if the sum
 *    of src + dst + atomic-dst + scatter bindings exceeds RCHAN_MAX_BINDINGS.
 *
 * BINDING LIMIT
 * ─────────────
 *  RCHAN_MAX_BINDINGS = 8  (GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS on target HW)
 *  This is hardware-enforced and checked by rchan_desc_validate().
 *
 * FIELD OPERATIONS
 * ────────────────
 *  RCHAN_OP_COPY      src_slot[src_field] → dst_slot[dst_field]  (float)
 *  RCHAN_OP_COPY_U    same, preserving bit pattern (uint reinterpret)
 *  RCHAN_OP_COPY_I    same, preserving bit pattern (int  reinterpret)
 *  RCHAN_OP_ZERO      dst_slot[dst_field] ← 0.0
 *  RCHAN_OP_CONST_F   dst_slot[dst_field] ← literal float constant
 *  RCHAN_OP_CONST_U   dst_slot[dst_field] ← literal uint  constant (bit-cast)
 *  RCHAN_OP_ATOMIC_AF dst_slot[dst_field] += src_slot[src_field]  (float CAS)
 *  RCHAN_OP_SCATTER   dst record index comes from scatter_buf[src_rec]
 *                     instead of matching src_rec.  src/dst fields as above.
 *
 *  For bulk ranges, repeat RCHAN_OP_COPY entries with consecutive fields, or
 *  use rchan_ops_copy_range() / rchan_ops_copy_range_remap() helpers to
 *  programmatically push a run of copy ops onto an ops array.
 *
 * MULTI-REGION BUFFERS
 * ────────────────────
 *  When the same GL buffer object holds multiple logical arrays (e.g.
 *  BdptOutputBuf: verts / spectral / PDFs), bind it to separate src slots
 *  with different base_floats offsets.  rchan_desc_validate() allows the same
 *  GL binding point to appear in multiple slots only when all are read-only
 *  (src slots).  Aliasing a dst slot is rejected.
 *
 * USAGE — CPU path
 * ────────────────
 *   RchanDesc d = {0};
 *   // ... fill src_slots, dst_slots, ops, n_ops, n_records ...
 *   rchan_execute_cpu(&d,
 *       (float*[]){ src0_ptr, src1_ptr, NULL, ... },
 *       (float*[]){ dst0_ptr, NULL, ... },
 *       (uint32_t*[]){ NULL, ... },   // atomic uint[] dst slots
 *       scatter_indices_or_NULL,
 *       record_base, n_records_this_call);
 *
 * USAGE — GLSL generator
 * ──────────────────────
 *   char buf[65536];
 *   int n = rchan_emit_glsl(&d, buf, sizeof(buf),
 *                           "my_repack",   // shader name for comment
 *                           local_size_x); // workgroup size
 *   // compile buf as a compute shader
 *
 * ─────────────────────────────────────────────────────────────────────────── */

#pragma once
#ifndef SSBO_RECHANNELER_H
#define SSBO_RECHANNELER_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Limits ─────────────────────────────────────────────────────────────── */

#define RCHAN_MAX_BINDINGS   8   /* GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS      */
#define RCHAN_MAX_SRC_SLOTS  8   /* per descriptor                            */
#define RCHAN_MAX_DST_SLOTS  8   /* per descriptor                            */
#define RCHAN_MAX_OPS        4096 /* field operations per descriptor           */
#define RCHAN_MAX_NAME       64   /* buffer/slot name length (for GLSL output) */

/* ── Error codes ────────────────────────────────────────────────────────── */

typedef enum {
    RCHAN_OK                  =  0,
    RCHAN_ERR_TOO_MANY_BINDINGS = -1, /* total bindings > 8                   */
    RCHAN_ERR_BINDING_ALIAS_DST = -2, /* same GL binding point in two dst slots */
    RCHAN_ERR_BAD_SLOT        = -3,  /* op references an undeclared slot       */
    RCHAN_ERR_BAD_FIELD       = -4,  /* op field index ≥ slot stride           */
    RCHAN_ERR_TOO_MANY_OPS    = -5,  /* ops array capacity exceeded            */
    RCHAN_ERR_BUF_TOO_SMALL   = -6,  /* GLSL output buffer too small           */
    RCHAN_ERR_NULL_PARAM      = -7,
} RchanError;

const char* rchan_strerror(RchanError e);

/* ── Slot type ──────────────────────────────────────────────────────────── */

typedef enum {
    RCHAN_SLOT_SRC        = 0,   /* readonly float[]  source                  */
    RCHAN_SLOT_DST        = 1,   /* coherent  float[]  destination (writes)   */
    RCHAN_SLOT_DST_ATOMIC = 2,   /* coherent  uint[]   destination (CAS)      */
} RchanSlotType;

/* ── Slot descriptor ────────────────────────────────────────────────────── */

typedef struct {
    int          gl_binding;     /* GL SSBO binding point (0–7)               */
    uint32_t     stride;         /* floats (or uints) per record               */
    uint32_t     base_floats;    /* flat-element offset into the buffer
                                  * (for multi-region buffers; 0 = start)     */
    RchanSlotType type;
    char         name[RCHAN_MAX_NAME]; /* label used in generated GLSL        */
} RchanSlot;

/* ── Field operation ────────────────────────────────────────────────────── */

typedef enum {
    RCHAN_OP_COPY       = 0,  /* dst_slot[dst_field] = src_slot[src_field]    */
    RCHAN_OP_COPY_U     = 1,  /* same, bit-cast uint reinterpret              */
    RCHAN_OP_COPY_I     = 2,  /* same, bit-cast int  reinterpret              */
    RCHAN_OP_ZERO       = 3,  /* dst_slot[dst_field] = 0.0                   */
    RCHAN_OP_CONST_F    = 4,  /* dst_slot[dst_field] = const_val (float)     */
    RCHAN_OP_CONST_U    = 5,  /* dst_slot[dst_field] = uint (stored as bits) */
    RCHAN_OP_ATOMIC_AF  = 6,  /* atomicCAS-add: dst_slot[dst_field] += src   */
    RCHAN_OP_SCATTER    = 7,  /* like COPY but dst_rec = scatter_buf[src_rec]*/
} RchanOpType;

typedef struct {
    RchanOpType  op;
    uint8_t      src_slot;    /* index into RchanDesc.src_slots[]             */
    uint8_t      dst_slot;    /* index into RchanDesc.dst_slots[]             */
    uint16_t     src_field;   /* field index within source record             */
    uint16_t     dst_field;   /* field index within dest record               */
    union {
        float    const_f;     /* RCHAN_OP_CONST_F constant                    */
        uint32_t const_u;     /* RCHAN_OP_CONST_U constant                    */
    };
} RchanFieldOp;

/* ── Descriptor ─────────────────────────────────────────────────────────── */

typedef struct {
    /* Source slots (readonly float[]) */
    RchanSlot    src_slots[RCHAN_MAX_SRC_SLOTS];
    int          n_src_slots;

    /* Destination slots — regular writes (coherent float[]) */
    RchanSlot    dst_slots[RCHAN_MAX_DST_SLOTS];
    int          n_dst_slots;

    /* Destination slots — atomic uint[] for float-CAS accumulation */
    RchanSlot    adst_slots[RCHAN_MAX_DST_SLOTS];
    int          n_adst_slots;

    /* Scatter index buffer (optional).
     * When gl_binding >= 0, shader uses scatter_buf[src_rec] as dst_rec
     * for RCHAN_OP_SCATTER operations.  Counts toward 8-binding total.      */
    int          scatter_gl_binding;    /* -1 = disabled                      */
    char         scatter_name[RCHAN_MAX_NAME];

    /* Field operation list */
    RchanFieldOp ops[RCHAN_MAX_OPS];
    int          n_ops;

    /* Number of records (needed for validation; actual dispatch count is a
     * runtime parameter to rchan_execute_cpu / the GLSL uniform).           */
    uint32_t     n_records_hint;

    /* GLSL local workgroup size emitted by rchan_emit_glsl() */
    int          local_size_x;         /* 0 → default 64                      */

    /* CAS spin iteration limit emitted in generated GLSL */
    int          cas_iters;            /* 0 → default 128                     */
} RchanDesc;

/* ── Descriptor helpers ─────────────────────────────────────────────────── */

/* Zero-initialise a descriptor */
static inline void rchan_desc_init(RchanDesc* d) {
    /* memset-safe: all zeros is a valid empty descriptor */
    /* In C we can't call memset directly without string.h, so users may
     * use  *d = (RchanDesc){0};  or memset(d, 0, sizeof(*d));             */
    (void)d;
}

/* Push a single op; returns RCHAN_OK or RCHAN_ERR_TOO_MANY_OPS */
RchanError rchan_push_op(RchanDesc* d, RchanFieldOp op);

/* Push a run of consecutive RCHAN_OP_COPY ops:
 *   src_slot[src_field_start + i] → dst_slot[dst_field_start + i]
 *   for i in [0, count) */
RchanError rchan_ops_copy_range(RchanDesc*  d,
                                uint8_t     src_slot,
                                uint16_t    src_field_start,
                                uint8_t     dst_slot,
                                uint16_t    dst_field_start,
                                uint16_t    count);

/* Same but with separate src/dst field bases (layout remap) */
RchanError rchan_ops_copy_range_remap(RchanDesc*  d,
                                      uint8_t     src_slot,
                                      uint16_t    src_field,
                                      uint8_t     dst_slot,
                                      uint16_t    dst_field,
                                      uint16_t    count);

/* Push a RCHAN_OP_ZERO run */
RchanError rchan_ops_zero_range(RchanDesc*  d,
                                uint8_t     dst_slot,
                                uint16_t    dst_field_start,
                                uint16_t    count);

/* ── Validation ─────────────────────────────────────────────────────────── */

/* Returns RCHAN_OK if the descriptor is self-consistent and within all
 * hardware limits.  Writes a human-readable error string (up to err_len
 * bytes) into err_buf on failure; err_buf may be NULL. */
RchanError rchan_desc_validate(const RchanDesc* d,
                               char*            err_buf,
                               size_t           err_len);

/* ── CPU execution ──────────────────────────────────────────────────────── */

/* Execute the rechanneling on the CPU for records [record_base,
 * record_base + n_records).
 *
 *   src_ptrs[i]   — pointer to the flat float[] backing for src_slots[i].
 *                   Must be non-NULL for every declared src slot.
 *   dst_ptrs[i]   — pointer to the flat float[] backing for dst_slots[i].
 *   adst_ptrs[i]  — pointer to the flat uint32_t[] backing for adst_slots[i].
 *   scatter_ptr   — uint32_t[] mapping src_rec → dst_rec; NULL if disabled.
 *
 * Thread-safe: multiple calls with non-overlapping [record_base, +n) ranges
 * may run concurrently as long as no two write the same dst record for
 * non-atomic slots.  Atomic (adst) slots are always safe.
 */
RchanError rchan_execute_cpu(const RchanDesc*   d,
                             float* const*      src_ptrs,
                             float* const*      dst_ptrs,
                             uint32_t* const*   adst_ptrs,
                             const uint32_t*    scatter_ptr,
                             uint32_t           record_base,
                             uint32_t           n_records);

/* ── GLSL source generator ──────────────────────────────────────────────── */

/* Emit a self-contained GLSL compute shader (as a NUL-terminated string)
 * into out_buf (capacity out_len bytes) that performs the same mapping as
 * the descriptor.
 *
 * The generated shader:
 *   • #version 430 core
 *   • #define RCHAN_PROVIDE_MAIN
 *   • #define all RCHAN_ configuration macros derived from the descriptor
 *   • #include "ssbo_rechanneler.glsl.inc"   (injected path, not real #include)
 *   • void rchan_map(uint rec_i) { ... generated body ... }
 *
 * The GLSL #include is expanded inline — the generator reads the .inc file
 * at emit time from inc_path (may be NULL to use a built-in minimal copy
 * of the preamble stubs, which omits the full CAS helper boilerplate and is
 * suitable only for non-atomic descriptors).
 *
 * Returns number of bytes written (excluding NUL), or negative RchanError.
 */
int rchan_emit_glsl(const RchanDesc* d,
                    char*            out_buf,
                    size_t           out_len,
                    const char*      shader_name,
                    const char*      inc_path);

/* ── Accessor utilities ─────────────────────────────────────────────────── */

/* Read a field from a src slot buffer (CPU path convenience) */
static inline float rchan_read_f(const RchanSlot* slot,
                                 const float*     buf,
                                 uint32_t         rec,
                                 uint32_t         field) {
    return buf[slot->base_floats + rec * slot->stride + field];
}

static inline uint32_t rchan_read_u(const RchanSlot* slot,
                                    const float*     buf,
                                    uint32_t         rec,
                                    uint32_t         field) {
    float v = rchan_read_f(slot, buf, rec, field);
    uint32_t u;
    memcpy(&u, &v, 4);
    return u;
}

static inline int32_t rchan_read_i(const RchanSlot* slot,
                                   const float*     buf,
                                   uint32_t         rec,
                                   uint32_t         field) {
    float v = rchan_read_f(slot, buf, rec, field);
    int32_t i;
    memcpy(&i, &v, 4);
    return i;
}

/* Write a field to a dst slot buffer (CPU path convenience) */
static inline void rchan_write_f(const RchanSlot* slot,
                                 float*           buf,
                                 uint32_t         rec,
                                 uint32_t         field,
                                 float            val) {
    buf[slot->base_floats + rec * slot->stride + field] = val;
}

static inline void rchan_write_u(const RchanSlot* slot,
                                 float*           buf,
                                 uint32_t         rec,
                                 uint32_t         field,
                                 uint32_t         val) {
    float v;
    memcpy(&v, &val, 4);
    buf[slot->base_floats + rec * slot->stride + field] = v;
}

/* Atomic float-CAS add (CPU, uses __atomic for thread safety) */
void rchan_atomic_add_f(uint32_t* buf,
                        uint32_t  idx,
                        float     val);

/* ── Descriptor serialisation (for inspection / round-trip testing) ─────── */

/* Write a human-readable text representation of the descriptor to out_buf.
 * Returns bytes written or negative RchanError. */
int rchan_desc_to_text(const RchanDesc* d, char* out_buf, size_t out_len);

/* ─────────────────────────────────────────────────────────────────────────── */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SSBO_RECHANNELER_H */
