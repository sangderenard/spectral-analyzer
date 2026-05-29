/* ssbo_rechanneler.c ─────────────────────────────────────────────────────────
 *
 * Implementation of the single-source SSBO rechanneler C API.
 *
 * See csrc/include/ssbo_rechanneler.h for full documentation.
 * ─────────────────────────────────────────────────────────────────────────── */

#include "ssbo_rechanneler.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _MSC_VER
#  include <intrin.h>
#endif

/* ── Internal: portable memcpy-based bit reinterpret ─────────────────────── */

static inline uint32_t _f2u(float v)  { uint32_t u; memcpy(&u, &v, 4); return u; }
static inline int32_t  _f2i(float v)  { int32_t  i; memcpy(&i, &v, 4); return i; }
static inline float    _u2f(uint32_t u){ float   v; memcpy(&v, &u, 4); return v; }
static inline float    _i2f(int32_t  i){ float   v; memcpy(&v, &i, 4); return v; }

/* ── Error strings ──────────────────────────────────────────────────────── */

const char* rchan_strerror(RchanError e) {
    switch (e) {
        case RCHAN_OK:                   return "OK";
        case RCHAN_ERR_TOO_MANY_BINDINGS:return "too many SSBO bindings (max 8)";
        case RCHAN_ERR_BINDING_ALIAS_DST:return "same GL binding point in two writable (dst) slots";
        case RCHAN_ERR_BAD_SLOT:         return "op references undeclared slot";
        case RCHAN_ERR_BAD_FIELD:        return "op field index >= slot stride";
        case RCHAN_ERR_TOO_MANY_OPS:     return "ops array capacity exceeded (max 4096)";
        case RCHAN_ERR_BUF_TOO_SMALL:    return "output buffer too small";
        case RCHAN_ERR_NULL_PARAM:       return "required parameter is NULL";
        default:                         return "unknown error";
    }
}

/* ── rchan_push_op ──────────────────────────────────────────────────────── */

RchanError rchan_push_op(RchanDesc* d, RchanFieldOp op) {
    if (!d) return RCHAN_ERR_NULL_PARAM;
    if (d->n_ops >= RCHAN_MAX_OPS) return RCHAN_ERR_TOO_MANY_OPS;
    d->ops[d->n_ops++] = op;
    return RCHAN_OK;
}

/* ── rchan_ops_copy_range ────────────────────────────────────────────────── */

RchanError rchan_ops_copy_range(RchanDesc*  d,
                                uint8_t     src_slot,
                                uint16_t    src_field_start,
                                uint8_t     dst_slot,
                                uint16_t    dst_field_start,
                                uint16_t    count) {
    if (!d) return RCHAN_ERR_NULL_PARAM;
    for (uint16_t i = 0; i < count; ++i) {
        if (d->n_ops >= RCHAN_MAX_OPS) return RCHAN_ERR_TOO_MANY_OPS;
        RchanFieldOp op;
        memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_COPY;
        op.src_slot  = src_slot;
        op.dst_slot  = dst_slot;
        op.src_field = (uint16_t)(src_field_start + i);
        op.dst_field = (uint16_t)(dst_field_start + i);
        d->ops[d->n_ops++] = op;
    }
    return RCHAN_OK;
}

RchanError rchan_ops_copy_range_remap(RchanDesc*  d,
                                      uint8_t     src_slot,
                                      uint16_t    src_field,
                                      uint8_t     dst_slot,
                                      uint16_t    dst_field,
                                      uint16_t    count) {
    /* Same as rchan_ops_copy_range with independent src/dst bases */
    return rchan_ops_copy_range(d, src_slot, src_field,
                                   dst_slot, dst_field, count);
}

RchanError rchan_ops_zero_range(RchanDesc*  d,
                                uint8_t     dst_slot,
                                uint16_t    dst_field_start,
                                uint16_t    count) {
    if (!d) return RCHAN_ERR_NULL_PARAM;
    for (uint16_t i = 0; i < count; ++i) {
        if (d->n_ops >= RCHAN_MAX_OPS) return RCHAN_ERR_TOO_MANY_OPS;
        RchanFieldOp op;
        memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_ZERO;
        op.dst_slot  = dst_slot;
        op.dst_field = (uint16_t)(dst_field_start + i);
        d->ops[d->n_ops++] = op;
    }
    return RCHAN_OK;
}

/* ── rchan_desc_validate ────────────────────────────────────────────────── */

static void _verr(char* buf, size_t len, const char* fmt, ...) {
    if (!buf || len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, len, fmt, ap);
    va_end(ap);
}

RchanError rchan_desc_validate(const RchanDesc* d,
                               char*            err_buf,
                               size_t           err_len) {
    if (!d) { _verr(err_buf, err_len, "NULL descriptor"); return RCHAN_ERR_NULL_PARAM; }

    /* Count total GL binding slots consumed */
    int n_bindings = d->n_src_slots + d->n_dst_slots + d->n_adst_slots;
    if (d->scatter_gl_binding >= 0) n_bindings++;
    if (n_bindings > RCHAN_MAX_BINDINGS) {
        _verr(err_buf, err_len,
              "total bindings %d exceeds hardware limit %d",
              n_bindings, RCHAN_MAX_BINDINGS);
        return RCHAN_ERR_TOO_MANY_BINDINGS;
    }

    /* Check for writable dst binding aliases */
    for (int i = 0; i < d->n_dst_slots; ++i) {
        for (int j = i + 1; j < d->n_dst_slots; ++j) {
            if (d->dst_slots[i].gl_binding == d->dst_slots[j].gl_binding) {
                _verr(err_buf, err_len,
                      "dst_slots[%d] and dst_slots[%d] share GL binding %d",
                      i, j, d->dst_slots[i].gl_binding);
                return RCHAN_ERR_BINDING_ALIAS_DST;
            }
        }
        for (int j = 0; j < d->n_adst_slots; ++j) {
            if (d->dst_slots[i].gl_binding == d->adst_slots[j].gl_binding) {
                _verr(err_buf, err_len,
                      "dst_slots[%d] and adst_slots[%d] share GL binding %d",
                      i, j, d->dst_slots[i].gl_binding);
                return RCHAN_ERR_BINDING_ALIAS_DST;
            }
        }
    }

    /* Validate every op */
    for (int k = 0; k < d->n_ops; ++k) {
        const RchanFieldOp* op = &d->ops[k];

        /* Ops that reference a src slot */
        int needs_src = (op->op == RCHAN_OP_COPY   ||
                         op->op == RCHAN_OP_COPY_U  ||
                         op->op == RCHAN_OP_COPY_I  ||
                         op->op == RCHAN_OP_ATOMIC_AF ||
                         op->op == RCHAN_OP_SCATTER);
        if (needs_src) {
            if (op->src_slot >= (uint8_t)d->n_src_slots) {
                _verr(err_buf, err_len, "op[%d]: src_slot %d out of range (%d declared)",
                      k, op->src_slot, d->n_src_slots);
                return RCHAN_ERR_BAD_SLOT;
            }
            if (op->src_field >= (uint16_t)d->src_slots[op->src_slot].stride) {
                _verr(err_buf, err_len,
                      "op[%d]: src_field %d >= stride %u of src_slot %d",
                      k, op->src_field,
                      d->src_slots[op->src_slot].stride, op->src_slot);
                return RCHAN_ERR_BAD_FIELD;
            }
        }

        /* Ops that write a regular dst slot */
        int needs_dst = (op->op == RCHAN_OP_COPY     ||
                         op->op == RCHAN_OP_COPY_U    ||
                         op->op == RCHAN_OP_COPY_I    ||
                         op->op == RCHAN_OP_ZERO       ||
                         op->op == RCHAN_OP_CONST_F    ||
                         op->op == RCHAN_OP_CONST_U    ||
                         op->op == RCHAN_OP_SCATTER);
        if (needs_dst) {
            if (op->dst_slot >= (uint8_t)d->n_dst_slots) {
                _verr(err_buf, err_len, "op[%d]: dst_slot %d out of range (%d declared)",
                      k, op->dst_slot, d->n_dst_slots);
                return RCHAN_ERR_BAD_SLOT;
            }
            if (op->dst_field >= (uint16_t)d->dst_slots[op->dst_slot].stride) {
                _verr(err_buf, err_len,
                      "op[%d]: dst_field %d >= stride %u of dst_slot %d",
                      k, op->dst_field,
                      d->dst_slots[op->dst_slot].stride, op->dst_slot);
                return RCHAN_ERR_BAD_FIELD;
            }
        }

        /* Atomic dst slot ops */
        if (op->op == RCHAN_OP_ATOMIC_AF) {
            if (op->dst_slot >= (uint8_t)d->n_adst_slots) {
                _verr(err_buf, err_len,
                      "op[%d]: ATOMIC_AF dst_slot %d out of range (%d adst declared)",
                      k, op->dst_slot, d->n_adst_slots);
                return RCHAN_ERR_BAD_SLOT;
            }
            if (op->dst_field >= (uint16_t)d->adst_slots[op->dst_slot].stride) {
                _verr(err_buf, err_len,
                      "op[%d]: dst_field %d >= stride %u of adst_slot %d",
                      k, op->dst_field,
                      d->adst_slots[op->dst_slot].stride, op->dst_slot);
                return RCHAN_ERR_BAD_FIELD;
            }
        }
    }

    return RCHAN_OK;
}

/* ── rchan_atomic_add_f ─────────────────────────────────────────────────── */

void rchan_atomic_add_f(uint32_t* buf, uint32_t idx, float val) {
    /* CPU float-CAS loop — mirrors the GLSL CAS helper exactly */
    uint32_t expected;
    uint32_t* ptr = buf + idx;
#if defined(__GNUC__) || defined(__clang__)
    expected = __atomic_load_n(ptr, __ATOMIC_RELAXED);
    for (int iter = 0; iter < 128; ++iter) {
        float     fexp    = _u2f(expected);
        uint32_t  desired = _f2u(fexp + val);
        if (__atomic_compare_exchange_n(ptr, &expected, desired,
                                        /*weak=*/1,
                                        __ATOMIC_SEQ_CST,
                                        __ATOMIC_RELAXED)) return;
    }
#elif defined(_MSC_VER)
    expected = *(volatile uint32_t*)ptr;
    for (int iter = 0; iter < 128; ++iter) {
        float    fexp    = _u2f(expected);
        uint32_t desired = _f2u(fexp + val);
        uint32_t prev = _InterlockedCompareExchange((volatile long*)ptr,
                                                    (long)desired,
                                                    (long)expected);
        if (prev == expected) return;
        expected = prev;
    }
#else
    /* Fallback: non-atomic (single-threaded use only) */
    *ptr = _f2u(_u2f(*ptr) + val);
#endif
}

/* ── rchan_execute_cpu ──────────────────────────────────────────────────── */

RchanError rchan_execute_cpu(const RchanDesc*   d,
                             float* const*      src_ptrs,
                             float* const*      dst_ptrs,
                             uint32_t* const*   adst_ptrs,
                             const uint32_t*    scatter_ptr,
                             uint32_t           record_base,
                             uint32_t           n_records) {
    if (!d || !src_ptrs) return RCHAN_ERR_NULL_PARAM;
    /* dst_ptrs may be NULL when all ops are atomic (adst-only descriptor) */

    for (uint32_t r = 0; r < n_records; ++r) {
        const uint32_t src_rec = record_base + r;
        for (int k = 0; k < d->n_ops; ++k) {
            const RchanFieldOp* op = &d->ops[k];

            /* Determine dst record index */
            uint32_t dst_rec = src_rec;
            if (op->op == RCHAN_OP_SCATTER) {
                if (!scatter_ptr) continue;  /* no scatter buf, skip */
                dst_rec = scatter_ptr[src_rec];
            }

            switch (op->op) {
                case RCHAN_OP_COPY: {
                    const RchanSlot* ss = &d->src_slots[op->src_slot];
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    float v = src_ptrs[op->src_slot][
                                  ss->base_floats + src_rec * ss->stride + op->src_field];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = v;
                    break;
                }
                case RCHAN_OP_COPY_U: {
                    /* bit-cast: already same byte width, copy as float */
                    const RchanSlot* ss = &d->src_slots[op->src_slot];
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    float v = src_ptrs[op->src_slot][
                                  ss->base_floats + src_rec * ss->stride + op->src_field];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = v;
                    break;
                }
                case RCHAN_OP_COPY_I: {
                    const RchanSlot* ss = &d->src_slots[op->src_slot];
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    float v = src_ptrs[op->src_slot][
                                  ss->base_floats + src_rec * ss->stride + op->src_field];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = v;
                    break;
                }
                case RCHAN_OP_ZERO: {
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = 0.0f;
                    break;
                }
                case RCHAN_OP_CONST_F: {
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = op->const_f;
                    break;
                }
                case RCHAN_OP_CONST_U: {
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = _u2f(op->const_u);
                    break;
                }
                case RCHAN_OP_ATOMIC_AF: {
                    if (!adst_ptrs) break;
                    const RchanSlot* ss  = &d->src_slots[op->src_slot];
                    const RchanSlot* ads = &d->adst_slots[op->dst_slot];
                    float v = src_ptrs[op->src_slot][
                                  ss->base_floats + src_rec * ss->stride + op->src_field];
                    uint32_t idx = ads->base_floats + dst_rec * ads->stride + op->dst_field;
                    rchan_atomic_add_f(adst_ptrs[op->dst_slot], idx, v);
                    break;
                }
                case RCHAN_OP_SCATTER: {
                    const RchanSlot* ss = &d->src_slots[op->src_slot];
                    const RchanSlot* ds = &d->dst_slots[op->dst_slot];
                    float v = src_ptrs[op->src_slot][
                                  ss->base_floats + src_rec * ss->stride + op->src_field];
                    dst_ptrs[op->dst_slot][
                                  ds->base_floats + dst_rec * ds->stride + op->dst_field] = v;
                    break;
                }
            }
        }
    }
    return RCHAN_OK;
}

/* ── rchan_emit_glsl ────────────────────────────────────────────────────── */

/* Internal: append-to-buffer helper */
typedef struct { char* p; size_t cap; size_t used; int overflow; } _GBuf;

static void _gb_printf(_GBuf* g, const char* fmt, ...) {
    if (g->overflow) return;
    size_t remaining = g->cap - g->used;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(g->p + g->used, remaining, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= remaining) { g->overflow = 1; return; }
    g->used += (size_t)n;
}

static const char* _op_name(RchanOpType op) {
    switch (op) {
        case RCHAN_OP_COPY:      return "COPY";
        case RCHAN_OP_COPY_U:    return "COPY_U";
        case RCHAN_OP_COPY_I:    return "COPY_I";
        case RCHAN_OP_ZERO:      return "ZERO";
        case RCHAN_OP_CONST_F:   return "CONST_F";
        case RCHAN_OP_CONST_U:   return "CONST_U";
        case RCHAN_OP_ATOMIC_AF: return "ATOMIC_AF";
        case RCHAN_OP_SCATTER:   return "SCATTER";
        default:                 return "UNKNOWN";
    }
}

int rchan_emit_glsl(const RchanDesc* d,
                    char*            out_buf,
                    size_t           out_len,
                    const char*      shader_name,
                    const char*      inc_path) {
    if (!d || !out_buf || out_len == 0) return RCHAN_ERR_NULL_PARAM;

    _GBuf g = { out_buf, out_len, 0, 0 };

    int local_size = d->local_size_x > 0 ? d->local_size_x : 64;
    int cas_iters  = d->cas_iters  > 0 ? d->cas_iters  : 128;

    /* ── Version and identification ─────────────────────────────────────── */
    _gb_printf(&g, "#version 430 core\n");
    _gb_printf(&g, "/* Auto-generated by rchan_emit_glsl() — shader: %s\n",
               shader_name ? shader_name : "unnamed");
    _gb_printf(&g, " * DO NOT EDIT — regenerate via rchan_emit_glsl(). */\n\n");

    /* ── RCHAN configuration macros ─────────────────────────────────────── */

    /* Source slots */
    for (int i = 0; i < d->n_src_slots; ++i) {
        const RchanSlot* s = &d->src_slots[i];
        _gb_printf(&g, "#define RCHAN_SRC_BINDING_%d  %d\n", i, s->gl_binding);
        _gb_printf(&g, "#define RCHAN_SRC_STRIDE_%d   %uu\n", i, s->stride);
        if (s->base_floats != 0)
            _gb_printf(&g, "#define RCHAN_SRC_BASE_%d     %uu\n", i, s->base_floats);
    }
    _gb_printf(&g, "\n");

    /* Regular dest slots */
    for (int i = 0; i < d->n_dst_slots; ++i) {
        const RchanSlot* s = &d->dst_slots[i];
        _gb_printf(&g, "#define RCHAN_DST_BINDING_%d  %d\n", i, s->gl_binding);
        _gb_printf(&g, "#define RCHAN_DST_STRIDE_%d   %uu\n", i, s->stride);
        if (s->base_floats != 0)
            _gb_printf(&g, "#define RCHAN_DST_BASE_%d     %uu\n", i, s->base_floats);
    }
    _gb_printf(&g, "\n");

    /* Atomic dest slots */
    for (int i = 0; i < d->n_adst_slots; ++i) {
        const RchanSlot* s = &d->adst_slots[i];
        _gb_printf(&g, "#define RCHAN_DST_ATOMIC_BINDING_%d  %d\n", i, s->gl_binding);
        _gb_printf(&g, "#define RCHAN_DST_ATOMIC_STRIDE_%d   %uu\n", i, s->stride);
        if (s->base_floats != 0)
            _gb_printf(&g, "#define RCHAN_DST_ATOMIC_BASE_%d     %uu\n", i, s->base_floats);
    }
    if (d->n_adst_slots > 0)
        _gb_printf(&g, "\n");

    /* Scatter buffer */
    if (d->scatter_gl_binding >= 0)
        _gb_printf(&g, "#define RCHAN_SCATTER_BINDING  %d\n\n", d->scatter_gl_binding);

    /* Dispatch params */
    _gb_printf(&g, "#define RCHAN_LOCAL_SIZE   %d\n", local_size);
    _gb_printf(&g, "#define RCHAN_CAS_ITERS    %d\n", cas_iters);
    _gb_printf(&g, "#define RCHAN_PROVIDE_MAIN\n\n");

    /* ── Include preamble (inline if inc_path provided) ──────────────────── */
    if (inc_path) {
        FILE* f = fopen(inc_path, "r");
        if (f) {
            char line[512];
            while (fgets(line, sizeof(line), f))
                _gb_printf(&g, "%s", line);
            fclose(f);
        } else {
            /* Fall through to stub include comment */
            _gb_printf(&g, "/* WARNING: could not open %s — include manually */\n", inc_path);
            _gb_printf(&g, "#include \"%s\"\n\n", inc_path);
        }
    } else {
        _gb_printf(&g, "/* ssbo_rechanneler.glsl.inc — insert preamble here */\n\n");
    }

    /* ── rchan_map() body ────────────────────────────────────────────────── */
    _gb_printf(&g, "void rchan_map(uint rec_i) {\n");

    /* Track if we need a scatter dst_rec variable */
    int needs_scatter_var = 0;
    for (int k = 0; k < d->n_ops; ++k) {
        if (d->ops[k].op == RCHAN_OP_SCATTER) { needs_scatter_var = 1; break; }
    }
    if (needs_scatter_var)
        _gb_printf(&g, "    uint dst_rec = RCHAN_SCATTER_IDX(rec_i);\n");

    for (int k = 0; k < d->n_ops; ++k) {
        const RchanFieldOp* op = &d->ops[k];
        const char* dst_rec_expr = (op->op == RCHAN_OP_SCATTER) ? "dst_rec" : "rec_i";

        _gb_printf(&g, "    /* op[%d] %s */ ", k, _op_name(op->op));

        switch (op->op) {
            case RCHAN_OP_COPY:
                _gb_printf(&g,
                    "RCHAN_DST_WF(%d, %s, %du, RCHAN_SRC_F(%d, rec_i, %du));\n",
                    op->dst_slot, dst_rec_expr, op->dst_field,
                    op->src_slot, op->src_field);
                break;
            case RCHAN_OP_COPY_U:
                _gb_printf(&g,
                    "RCHAN_DST_WU(%d, %s, %du, RCHAN_SRC_U(%d, rec_i, %du));\n",
                    op->dst_slot, dst_rec_expr, op->dst_field,
                    op->src_slot, op->src_field);
                break;
            case RCHAN_OP_COPY_I:
                _gb_printf(&g,
                    "RCHAN_DST_WI(%d, %s, %du, RCHAN_SRC_I(%d, rec_i, %du));\n",
                    op->dst_slot, dst_rec_expr, op->dst_field,
                    op->src_slot, op->src_field);
                break;
            case RCHAN_OP_ZERO:
                _gb_printf(&g,
                    "RCHAN_DST_WF(%d, rec_i, %du, 0.0);\n",
                    op->dst_slot, op->dst_field);
                break;
            case RCHAN_OP_CONST_F:
                _gb_printf(&g,
                    "RCHAN_DST_WF(%d, rec_i, %du, %.9g);\n",
                    op->dst_slot, op->dst_field, (double)op->const_f);
                break;
            case RCHAN_OP_CONST_U:
                _gb_printf(&g,
                    "RCHAN_DST_WU(%d, rec_i, %du, %uu);\n",
                    op->dst_slot, op->dst_field, op->const_u);
                break;
            case RCHAN_OP_ATOMIC_AF:
                _gb_printf(&g,
                    "RCHAN_DST_AF(%d, rec_i, %du, RCHAN_SRC_F(%d, rec_i, %du));\n",
                    op->dst_slot, op->dst_field,
                    op->src_slot, op->src_field);
                break;
            case RCHAN_OP_SCATTER:
                _gb_printf(&g,
                    "RCHAN_DST_WF(%d, dst_rec, %du, RCHAN_SRC_F(%d, rec_i, %du));\n",
                    op->dst_slot, op->dst_field,
                    op->src_slot, op->src_field);
                break;
        }
    }

    _gb_printf(&g, "}\n");

    if (g.overflow) return RCHAN_ERR_BUF_TOO_SMALL;
    return (int)g.used;
}

/* ── rchan_desc_to_text ──────────────────────────────────────────────────── */

int rchan_desc_to_text(const RchanDesc* d, char* out_buf, size_t out_len) {
    if (!d || !out_buf || out_len == 0) return RCHAN_ERR_NULL_PARAM;
    _GBuf g = { out_buf, out_len, 0, 0 };

    _gb_printf(&g, "RchanDesc {\n");
    _gb_printf(&g, "  n_src_slots  = %d\n", d->n_src_slots);
    for (int i = 0; i < d->n_src_slots; ++i) {
        const RchanSlot* s = &d->src_slots[i];
        _gb_printf(&g, "    src[%d]: binding=%d stride=%u base=%u name=\"%s\"\n",
                   i, s->gl_binding, s->stride, s->base_floats, s->name);
    }
    _gb_printf(&g, "  n_dst_slots  = %d\n", d->n_dst_slots);
    for (int i = 0; i < d->n_dst_slots; ++i) {
        const RchanSlot* s = &d->dst_slots[i];
        _gb_printf(&g, "    dst[%d]: binding=%d stride=%u base=%u name=\"%s\"\n",
                   i, s->gl_binding, s->stride, s->base_floats, s->name);
    }
    _gb_printf(&g, "  n_adst_slots = %d\n", d->n_adst_slots);
    for (int i = 0; i < d->n_adst_slots; ++i) {
        const RchanSlot* s = &d->adst_slots[i];
        _gb_printf(&g, "    adst[%d]: binding=%d stride=%u base=%u name=\"%s\"\n",
                   i, s->gl_binding, s->stride, s->base_floats, s->name);
    }
    _gb_printf(&g, "  scatter_gl_binding = %d\n", d->scatter_gl_binding);
    _gb_printf(&g, "  n_ops        = %d\n", d->n_ops);
    for (int k = 0; k < d->n_ops; ++k) {
        const RchanFieldOp* op = &d->ops[k];
        switch (op->op) {
            case RCHAN_OP_CONST_F:
                _gb_printf(&g, "    op[%3d] %-10s src=%d.%d dst=%d.%d const_f=%.6g\n",
                           k, _op_name(op->op),
                           op->src_slot, op->src_field,
                           op->dst_slot, op->dst_field,
                           (double)op->const_f);
                break;
            case RCHAN_OP_CONST_U:
                _gb_printf(&g, "    op[%3d] %-10s src=%d.%d dst=%d.%d const_u=0x%08x\n",
                           k, _op_name(op->op),
                           op->src_slot, op->src_field,
                           op->dst_slot, op->dst_field,
                           op->const_u);
                break;
            case RCHAN_OP_ZERO:
                _gb_printf(&g, "    op[%3d] %-10s dst=%d.%d\n",
                           k, _op_name(op->op),
                           op->dst_slot, op->dst_field);
                break;
            default:
                _gb_printf(&g, "    op[%3d] %-10s src=%d.%d dst=%d.%d\n",
                           k, _op_name(op->op),
                           op->src_slot, op->src_field,
                           op->dst_slot, op->dst_field);
                break;
        }
    }
    _gb_printf(&g, "}\n");

    if (g.overflow) return RCHAN_ERR_BUF_TOO_SMALL;
    return (int)g.used;
}
