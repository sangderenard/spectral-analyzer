/* pybind_rechanneler.cpp ─────────────────────────────────────────────────────
 *
 * Python bindings for the SSBO rechanneler C API (ssbo_rechanneler.h).
 *
 * Exposes:
 *   ssbo_rechanneler.RchanDesc       — descriptor builder (fluent Python API)
 *   ssbo_rechanneler.RchanSlot       — slot descriptor
 *   ssbo_rechanneler.RchanFieldOp    — single field operation
 *   ssbo_rechanneler.execute_cpu()   — run on CPU numpy arrays (zero-copy)
 *   ssbo_rechanneler.emit_glsl()     — generate GLSL compute shader source
 *   ssbo_rechanneler.validate()      — validate a descriptor
 *   ssbo_rechanneler.to_text()       — human-readable descriptor dump
 *
 * Python usage example:
 *
 *   import ssbo_rechanneler as rc
 *   import numpy as np
 *
 *   d = rc.RchanDesc()
 *   d.add_src(gl_binding=0, stride=28, name="BdptVertBuf")
 *   d.add_src(gl_binding=1, stride=8,  name="BdptPdfStagingBuf")
 *   d.add_dst(gl_binding=2, stride=40, name="T5LightVertBuf")
 *   d.copy_range(src_slot=0, src_field=7, dst_slot=0, dst_field=0, count=6)
 *   d.copy(src_slot=0, src_field=25, dst_slot=0, dst_field=6)   # throughput
 *   d.copy_u(src_slot=0, src_field=2,  dst_slot=0, dst_field=7) # flags
 *   d.copy(src_slot=1, src_field=0,   dst_slot=0, dst_field=12) # pdf_fwd
 *   d.zero_range(dst_slot=0, dst_field=30, count=10)            # pad tail
 *
 *   rc.validate(d)   # raises RuntimeError on any error
 *
 *   glsl_src = rc.emit_glsl(d, name="t5_repack_lgv",
 *                            inc_path="csrc/shaders/ssbo_rechanneler.glsl.inc")
 *   print(glsl_src[:200])
 *
 *   # CPU fallback / unit test
 *   src0 = np.zeros(28 * 1000, dtype=np.float32)
 *   dst0 = np.zeros(40 * 1000, dtype=np.float32)
 *   rc.execute_cpu(d,
 *       src_arrays=[src0, src1],
 *       dst_arrays=[dst0],
 *       adst_arrays=[],
 *       scatter=None,
 *       record_base=0, n_records=1000)
 *
 * ─────────────────────────────────────────────────────────────────────────── */

#include "ssbo_rechanneler.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

/* ── Helper: check RchanError and throw on failure ───────────────────────── */

static void _check(RchanError e) {
    if (e != RCHAN_OK)
        throw std::runtime_error(rchan_strerror(e));
}

/* ── Python-facing wrapper class for RchanDesc ───────────────────────────── */

struct PyRchanDesc {
    RchanDesc d;

    PyRchanDesc() {
        memset(&d, 0, sizeof(d));
        d.scatter_gl_binding = -1;
        d.local_size_x       = 64;
        d.cas_iters          = 128;
    }

    /* ── Slot builders ─────────────────────────────────────────────────── */

    int add_src(int gl_binding, uint32_t stride, uint32_t base_floats = 0,
                const std::string& name = "") {
        if (d.n_src_slots >= RCHAN_MAX_SRC_SLOTS)
            throw std::runtime_error("too many src slots");
        int idx = d.n_src_slots++;
        RchanSlot& s = d.src_slots[idx];
        s.gl_binding  = gl_binding;
        s.stride      = stride;
        s.base_floats = base_floats;
        s.type        = RCHAN_SLOT_SRC;
        snprintf(s.name, RCHAN_MAX_NAME, "%s", name.empty() ? "SrcSlot" : name.c_str());
        return idx;
    }

    int add_dst(int gl_binding, uint32_t stride, uint32_t base_floats = 0,
                const std::string& name = "") {
        if (d.n_dst_slots >= RCHAN_MAX_DST_SLOTS)
            throw std::runtime_error("too many dst slots");
        int idx = d.n_dst_slots++;
        RchanSlot& s = d.dst_slots[idx];
        s.gl_binding  = gl_binding;
        s.stride      = stride;
        s.base_floats = base_floats;
        s.type        = RCHAN_SLOT_DST;
        snprintf(s.name, RCHAN_MAX_NAME, "%s", name.empty() ? "DstSlot" : name.c_str());
        return idx;
    }

    int add_adst(int gl_binding, uint32_t stride, uint32_t base_floats = 0,
                 const std::string& name = "") {
        if (d.n_adst_slots >= RCHAN_MAX_DST_SLOTS)
            throw std::runtime_error("too many adst slots");
        int idx = d.n_adst_slots++;
        RchanSlot& s = d.adst_slots[idx];
        s.gl_binding  = gl_binding;
        s.stride      = stride;
        s.base_floats = base_floats;
        s.type        = RCHAN_SLOT_DST_ATOMIC;
        snprintf(s.name, RCHAN_MAX_NAME, "%s", name.empty() ? "ADstSlot" : name.c_str());
        return idx;
    }

    void set_scatter(int gl_binding, const std::string& name = "") {
        d.scatter_gl_binding = gl_binding;
        snprintf(d.scatter_name, RCHAN_MAX_NAME,
                 "%s", name.empty() ? "ScatterBuf" : name.c_str());
    }

    void set_local_size(int ls)   { d.local_size_x = ls; }
    void set_cas_iters(int iters) { d.cas_iters    = iters; }

    /* ── Op builders ───────────────────────────────────────────────────── */

    void copy(int src_slot, int src_field, int dst_slot, int dst_field) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_COPY;
        op.src_slot  = (uint8_t)src_slot;
        op.dst_slot  = (uint8_t)dst_slot;
        op.src_field = (uint16_t)src_field;
        op.dst_field = (uint16_t)dst_field;
        _check(rchan_push_op(&d, op));
    }

    void copy_u(int src_slot, int src_field, int dst_slot, int dst_field) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_COPY_U;
        op.src_slot  = (uint8_t)src_slot;
        op.dst_slot  = (uint8_t)dst_slot;
        op.src_field = (uint16_t)src_field;
        op.dst_field = (uint16_t)dst_field;
        _check(rchan_push_op(&d, op));
    }

    void copy_i(int src_slot, int src_field, int dst_slot, int dst_field) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_COPY_I;
        op.src_slot  = (uint8_t)src_slot;
        op.dst_slot  = (uint8_t)dst_slot;
        op.src_field = (uint16_t)src_field;
        op.dst_field = (uint16_t)dst_field;
        _check(rchan_push_op(&d, op));
    }

    void zero(int dst_slot, int dst_field) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op       = RCHAN_OP_ZERO;
        op.dst_slot  = (uint8_t)dst_slot;
        op.dst_field = (uint16_t)dst_field;
        _check(rchan_push_op(&d, op));
    }

    void const_f(int dst_slot, int dst_field, float val) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op       = RCHAN_OP_CONST_F;
        op.dst_slot  = (uint8_t)dst_slot;
        op.dst_field = (uint16_t)dst_field;
        op.const_f   = val;
        _check(rchan_push_op(&d, op));
    }

    void const_u(int dst_slot, int dst_field, uint32_t val) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op       = RCHAN_OP_CONST_U;
        op.dst_slot  = (uint8_t)dst_slot;
        op.dst_field = (uint16_t)dst_field;
        op.const_u   = val;
        _check(rchan_push_op(&d, op));
    }

    void atomic_af(int src_slot, int src_field, int dst_slot, int dst_field) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_ATOMIC_AF;
        op.src_slot  = (uint8_t)src_slot;
        op.dst_slot  = (uint8_t)dst_slot;
        op.src_field = (uint16_t)src_field;
        op.dst_field = (uint16_t)dst_field;
        _check(rchan_push_op(&d, op));
    }

    void scatter(int src_slot, int src_field, int dst_slot, int dst_field) {
        RchanFieldOp op; memset(&op, 0, sizeof(op));
        op.op        = RCHAN_OP_SCATTER;
        op.src_slot  = (uint8_t)src_slot;
        op.dst_slot  = (uint8_t)dst_slot;
        op.src_field = (uint16_t)src_field;
        op.dst_field = (uint16_t)dst_field;
        _check(rchan_push_op(&d, op));
    }

    /* Bulk helpers */
    void copy_range(int src_slot, int src_field,
                    int dst_slot, int dst_field, int count) {
        _check(rchan_ops_copy_range(&d,
                                    (uint8_t)src_slot, (uint16_t)src_field,
                                    (uint8_t)dst_slot, (uint16_t)dst_field,
                                    (uint16_t)count));
    }

    void zero_range(int dst_slot, int dst_field, int count) {
        _check(rchan_ops_zero_range(&d,
                                    (uint8_t)dst_slot, (uint16_t)dst_field,
                                    (uint16_t)count));
    }

    /* Introspection */
    int n_ops()       const { return d.n_ops; }
    int n_src_slots() const { return d.n_src_slots; }
    int n_dst_slots() const { return d.n_dst_slots; }
    int n_adst_slots()const { return d.n_adst_slots; }

    std::string to_text() const {
        std::string buf(65536, '\0');
        int n = rchan_desc_to_text(&d, &buf[0], buf.size());
        if (n < 0) throw std::runtime_error(rchan_strerror((RchanError)n));
        buf.resize((size_t)n);
        return buf;
    }

    std::string __repr__() const { return to_text(); }
};

/* ── Module ──────────────────────────────────────────────────────────────── */

PYBIND11_MODULE(ssbo_rechanneler, m) {
    m.doc() = "Single-source SSBO rechanneler — CPU execution + GLSL code generator";

    /* ── RchanDesc wrapper ─────────────────────────────────────────────── */
    py::class_<PyRchanDesc>(m, "RchanDesc",
        "Descriptor for an arbitrary float-buffer remapping.\n"
        "\n"
        "Build up a mapping by calling add_src/add_dst/copy*/zero* etc.,\n"
        "then pass to validate(), emit_glsl(), or execute_cpu().\n")

        .def(py::init<>())

        /* Slot builders */
        .def("add_src",  &PyRchanDesc::add_src,
             py::arg("gl_binding"), py::arg("stride"),
             py::arg("base_floats") = 0, py::arg("name") = "",
             "Declare a readonly source SSBO slot.  Returns slot index.")
        .def("add_dst",  &PyRchanDesc::add_dst,
             py::arg("gl_binding"), py::arg("stride"),
             py::arg("base_floats") = 0, py::arg("name") = "",
             "Declare a coherent (writable) destination SSBO slot.  Returns slot index.")
        .def("add_adst", &PyRchanDesc::add_adst,
             py::arg("gl_binding"), py::arg("stride"),
             py::arg("base_floats") = 0, py::arg("name") = "",
             "Declare an atomic uint[] destination SSBO slot.  Returns slot index.")
        .def("set_scatter", &PyRchanDesc::set_scatter,
             py::arg("gl_binding"), py::arg("name") = "",
             "Set the scatter index buffer (uint[]) GL binding point.")
        .def("set_local_size", &PyRchanDesc::set_local_size,
             py::arg("local_size"), "Set GLSL workgroup local_size_x.")
        .def("set_cas_iters",  &PyRchanDesc::set_cas_iters,
             py::arg("iters"), "Set atomic float-CAS spin iteration limit.")

        /* Single-field op builders */
        .def("copy",      &PyRchanDesc::copy,
             py::arg("src_slot"), py::arg("src_field"),
             py::arg("dst_slot"), py::arg("dst_field"),
             "Copy one float field (src→dst).")
        .def("copy_u",    &PyRchanDesc::copy_u,
             py::arg("src_slot"), py::arg("src_field"),
             py::arg("dst_slot"), py::arg("dst_field"),
             "Copy one field preserving uint bit pattern.")
        .def("copy_i",    &PyRchanDesc::copy_i,
             py::arg("src_slot"), py::arg("src_field"),
             py::arg("dst_slot"), py::arg("dst_field"),
             "Copy one field preserving int bit pattern.")
        .def("zero",      &PyRchanDesc::zero,
             py::arg("dst_slot"), py::arg("dst_field"),
             "Write 0.0 to a dst field.")
        .def("const_f",   &PyRchanDesc::const_f,
             py::arg("dst_slot"), py::arg("dst_field"), py::arg("val"),
             "Write a literal float constant to a dst field.")
        .def("const_u",   &PyRchanDesc::const_u,
             py::arg("dst_slot"), py::arg("dst_field"), py::arg("val"),
             "Write a literal uint constant (bit-cast) to a dst field.")
        .def("atomic_af", &PyRchanDesc::atomic_af,
             py::arg("src_slot"), py::arg("src_field"),
             py::arg("dst_slot"), py::arg("dst_field"),
             "Float-CAS atomic accumulate: adst_slot[dst_field] += src_slot[src_field].")
        .def("scatter",   &PyRchanDesc::scatter,
             py::arg("src_slot"), py::arg("src_field"),
             py::arg("dst_slot"), py::arg("dst_field"),
             "Scattered copy: dst_rec = scatter_buf[src_rec].")

        /* Bulk helpers */
        .def("copy_range", &PyRchanDesc::copy_range,
             py::arg("src_slot"), py::arg("src_field"),
             py::arg("dst_slot"), py::arg("dst_field"), py::arg("count"),
             "Copy count consecutive fields from src to dst (field layout may differ).")
        .def("zero_range", &PyRchanDesc::zero_range,
             py::arg("dst_slot"), py::arg("dst_field"), py::arg("count"),
             "Zero count consecutive dst fields.")

        /* Properties */
        .def_property_readonly("n_ops",       &PyRchanDesc::n_ops)
        .def_property_readonly("n_src_slots", &PyRchanDesc::n_src_slots)
        .def_property_readonly("n_dst_slots", &PyRchanDesc::n_dst_slots)
        .def_property_readonly("n_adst_slots",&PyRchanDesc::n_adst_slots)

        .def("to_text",  &PyRchanDesc::to_text,  "Human-readable descriptor dump.")
        .def("__repr__", &PyRchanDesc::__repr__);

    /* ── validate() ─────────────────────────────────────────────────────── */
    m.def("validate", [](const PyRchanDesc& pd) {
        char err[512] = {0};
        RchanError e = rchan_desc_validate(&pd.d, err, sizeof(err));
        if (e != RCHAN_OK)
            throw std::runtime_error(std::string(err));
    }, py::arg("desc"),
    "Validate a RchanDesc.  Raises RuntimeError describing the first problem found.");

    /* ── emit_glsl() ─────────────────────────────────────────────────────── */
    m.def("emit_glsl", [](const PyRchanDesc& pd,
                          const std::string& shader_name,
                          const std::string& inc_path) -> std::string {
        /* Two-pass: first with a small buffer to get exact size, then allocate. */
        const char* ip = inc_path.empty() ? nullptr : inc_path.c_str();
        /* Estimate size: 1KB header + 64 bytes per op + inc file (up to 32KB) */
        size_t cap = 1024 + (size_t)pd.d.n_ops * 128 + 65536;
        std::string buf(cap, '\0');
        int n = rchan_emit_glsl(&pd.d, &buf[0], cap,
                                shader_name.c_str(), ip);
        if (n == RCHAN_ERR_BUF_TOO_SMALL) {
            cap *= 4;
            buf.assign(cap, '\0');
            n = rchan_emit_glsl(&pd.d, &buf[0], cap, shader_name.c_str(), ip);
        }
        if (n < 0)
            throw std::runtime_error(rchan_strerror((RchanError)n));
        buf.resize((size_t)n);
        return buf;
    }, py::arg("desc"),
       py::arg("name")     = "rechanneler",
       py::arg("inc_path") = "",
    "Generate a GLSL compute shader source string from the descriptor.\n"
    "\n"
    "If inc_path is given, the ssbo_rechanneler.glsl.inc file is inlined into\n"
    "the output so the result is a single self-contained shader string.\n"
    "If inc_path is empty, a #include comment placeholder is emitted instead.");

    /* ── execute_cpu() ───────────────────────────────────────────────────── */
    m.def("execute_cpu", [](const PyRchanDesc&               pd,
                             std::vector<py::array_t<float>>  src_arrays,
                             std::vector<py::array_t<float>>  dst_arrays,
                             std::vector<py::array_t<uint32_t>> adst_arrays,
                             py::object                       scatter_obj,
                             uint32_t                         record_base,
                             uint32_t                         n_records)
    {
        /* Gather raw pointers — require C-contiguous float32 arrays */
        std::vector<float*> src_ptrs(src_arrays.size());
        for (size_t i = 0; i < src_arrays.size(); ++i) {
            auto info = src_arrays[i].request();
            if (info.format != py::format_descriptor<float>::format())
                throw std::runtime_error("src_arrays must be float32");
            src_ptrs[i] = static_cast<float*>(info.ptr);
        }

        std::vector<float*> dst_ptrs(dst_arrays.size());
        for (size_t i = 0; i < dst_arrays.size(); ++i) {
            auto info = dst_arrays[i].request();
            if (info.format != py::format_descriptor<float>::format())
                throw std::runtime_error("dst_arrays must be float32");
            dst_ptrs[i] = static_cast<float*>(info.ptr);
        }

        std::vector<uint32_t*> adst_ptrs(adst_arrays.size());
        for (size_t i = 0; i < adst_arrays.size(); ++i) {
            auto info = adst_arrays[i].request();
            if (info.itemsize != sizeof(uint32_t))
                throw std::runtime_error("adst_arrays must be 4-byte unsigned int (uint32)");
            adst_ptrs[i] = static_cast<uint32_t*>(info.ptr);
        }

        /* Optional scatter array */
        const uint32_t* scatter_ptr = nullptr;
        py::array_t<uint32_t> scatter_arr;
        if (!scatter_obj.is_none()) {
            scatter_arr = scatter_obj.cast<py::array_t<uint32_t>>();
            scatter_ptr = static_cast<const uint32_t*>(scatter_arr.request().ptr);
        }

        RchanError e = rchan_execute_cpu(
            &pd.d,
            src_ptrs.empty()  ? nullptr : src_ptrs.data(),
            dst_ptrs.empty()  ? nullptr : dst_ptrs.data(),
            adst_ptrs.empty() ? nullptr : adst_ptrs.data(),
            scatter_ptr,
            record_base, n_records);
        _check(e);
    },
    py::arg("desc"),
    py::arg("src_arrays"),
    py::arg("dst_arrays"),
    py::arg("adst_arrays")  = std::vector<py::array_t<uint32_t>>{},
    py::arg("scatter")      = py::none(),
    py::arg("record_base")  = 0u,
    py::arg("n_records")    = 0u,
    "Run the rechanneling on the CPU.\n"
    "\n"
    "src_arrays  : list of np.ndarray[float32], one per declared src slot.\n"
    "dst_arrays  : list of np.ndarray[float32], one per declared dst slot.\n"
    "adst_arrays : list of np.ndarray[uint32],  one per declared adst slot.\n"
    "scatter     : optional np.ndarray[uint32] mapping src_rec → dst_rec.\n"
    "record_base : first global record index (for tiled calls).\n"
    "n_records   : how many records to process starting at record_base.\n"
    "\n"
    "All arrays must be C-contiguous.  Writes happen in-place to dst/adst arrays.");

    /* ── to_text() convenience top-level ────────────────────────────────── */
    m.def("to_text", [](const PyRchanDesc& pd) { return pd.to_text(); },
          py::arg("desc"), "Human-readable descriptor dump.");

    /* ── Module-level constants ──────────────────────────────────────────── */
    m.attr("MAX_BINDINGS") = RCHAN_MAX_BINDINGS;
    m.attr("MAX_SRC_SLOTS")= RCHAN_MAX_SRC_SLOTS;
    m.attr("MAX_DST_SLOTS")= RCHAN_MAX_DST_SLOTS;
    m.attr("MAX_OPS")      = RCHAN_MAX_OPS;

    m.attr("OP_COPY")      = (int)RCHAN_OP_COPY;
    m.attr("OP_COPY_U")    = (int)RCHAN_OP_COPY_U;
    m.attr("OP_COPY_I")    = (int)RCHAN_OP_COPY_I;
    m.attr("OP_ZERO")      = (int)RCHAN_OP_ZERO;
    m.attr("OP_CONST_F")   = (int)RCHAN_OP_CONST_F;
    m.attr("OP_CONST_U")   = (int)RCHAN_OP_CONST_U;
    m.attr("OP_ATOMIC_AF") = (int)RCHAN_OP_ATOMIC_AF;
    m.attr("OP_SCATTER")   = (int)RCHAN_OP_SCATTER;
}
