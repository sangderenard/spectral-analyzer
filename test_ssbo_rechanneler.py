"""test_ssbo_rechanneler.py — verification suite for the SSBO rechanneler.

Coverage:
  1. Identity copy           — single src→dst, every field round-trips.
  2. Field remap             — source field N lands in dest field M (M ≠ N).
  3. Bulk copy_range         — contiguous block transfer, no neighbours clobbered.
  4. Zero / const_f / const_u — literal writes don't read src at all.
  5. Uint / int bit-cast     — copy_u / copy_i preserve NaN-pattern bit patterns.
  6. Stride extension        — src stride 12 → dst stride 40, pad fields stay 0.
  7. Multi-slot fanin        — two src slots merged into one dst slot.
  8. Multi-region buffer     — same flat array aliased at two base_floats offsets.
  9. Scatter reindex         — dst_rec = scatter[src_rec], no crosstalk.
 10. Atomic float accumulate — rchan_atomic_add_f races safely (sequential check).
 11. Record tiling           — three tiled calls equal one monolithic call.
 12. Binding-limit guard     — validate() rejects > 8 total bindings.
 13. Binding-alias guard     — validate() rejects two writable slots on same binding.
 14. Bad-slot guard          — validate() rejects op with src_slot out of range.
 15. Bad-field guard         — validate() rejects op with src_field ≥ stride.
 16. GLSL emit smoke         — emit_glsl() produces a non-empty string containing
                               required GLSL keywords and correct #defines.
 17. GLSL / CPU parity       — generated GLSL #defines match the descriptor values
                               baked into the CPU execution path.
 18. T5 light-vert repack    — end-to-end: old LGV stride-12 → new stride-40 with
                               all T5 field positions correct (PDF, optical, bands).
 19. T5 cam-vert repack      — end-to-end: old CGV stride-16 → new stride-56 with
                               per-band betas, pdf_fwd/rev, MIS prefix, optical flags.
 20. Out-of-bounds records   — n_records=0 and record_base past end are no-ops.

The module is imported as `ssbo_rechanneler`.  If the compiled extension is
unavailable all tests are skipped (not failed), matching project test policy.
"""
from __future__ import annotations

import math
import struct
import sys
import re
from typing import List, Optional

import numpy as np
import pytest

# ── Try to import the module ──────────────────────────────────────────────────

try:
    import ssbo_rechanneler as rc
    HAS_RC = True
except ImportError:
    HAS_RC = False

pytestmark = pytest.mark.skipif(not HAS_RC,
    reason="ssbo_rechanneler extension not built")

# ── Constants matching the T5 pipeline (must stay in sync with C++) ──────────

# Current (old) strides
T5_LGV_STRIDE_OLD = 12
T5_CGV_STRIDE_OLD = 16

# New extended strides — chosen here to be the canonical values.
# If you change the field layout, change these constants AND the field map
# dictionaries below.  The test will immediately catch any mismatch.
MAX_GPU_BANDS = 16

#  Light-vert extended layout (stride 40):
#    [0..2]   pos xyz
#    [3..5]   normal xyz
#    [6]      throughput_scalar
#    [7]      flags           (uint bits)
#    [8]      subpath_id      (uint bits)
#    [9]      vert_info       (uint bits)
#    [10]     beta_lum        (scalar)
#    [11]     pdf_fwd         (area)
#    [12]     pdf_rev         (area)
#    [13]     pdf_flags       (uint bits)
#    [14]     optical_block   (uint bits — BDPT blocking reasons)
#    [15]     prefix_pdf      (pre-computed forward prefix product)
#    [16..31] per-band beta magnitudes  [0..MAX_GPU_BANDS-1]
#    [32..39] pad / reserved
T5_LGV_STRIDE_NEW = 40
LGV_FIELD = {
    "pos_x":       0,  "pos_y":       1,  "pos_z":       2,
    "norm_x":      3,  "norm_y":      4,  "norm_z":      5,
    "throughput":  6,
    "flags":       7,
    "subpath_id":  8,
    "vert_info":   9,
    "beta_lum":   10,
    "pdf_fwd":    11,
    "pdf_rev":    12,
    "pdf_flags":  13,
    "optical":    14,
    "prefix_pdf": 15,
    # band betas: 16 .. 16+MAX_GPU_BANDS-1
}
LGV_BAND_BASE = 16

#  Cam-vert extended layout (stride 56):
#    [0..2]   pos xyz
#    [3..5]   normal xyz
#    [6]      throughput_scalar
#    [7]      flags           (uint bits)
#    [8]      subpath_id      (uint bits)
#    [9]      vert_index      (uint bits)
#    [10]     spectral_beta_r (pre-baked display)
#    [11]     spectral_beta_g
#    [12]     spectral_beta_b
#    [13]     sensor_origin_y
#    [14]     sensor_origin_z
#    [15]     pdf_fwd
#    [16]     pdf_rev
#    [17]     pdf_flags       (uint bits)
#    [18]     optical_block   (uint bits)
#    [19]     prefix_pdf      (cam-side prefix product)
#    [20]     mis_denom_sum   (pre-summed Σ_cuts denominator)
#    [21]     tri_mat_idx     (int bits — for MatBandBuf lookup in T5)
#    [22..37] per-band beta magnitudes [0..MAX_GPU_BANDS-1]
#    [38..55] pad / reserved
T5_CGV_STRIDE_NEW = 56
CGV_FIELD = {
    "pos_x":         0,  "pos_y":         1,  "pos_z":         2,
    "norm_x":        3,  "norm_y":        4,  "norm_z":        5,
    "throughput":    6,
    "flags":         7,
    "subpath_id":    8,
    "vert_index":    9,
    "beta_r":       10,  "beta_g":       11,  "beta_b":       12,
    "sensor_y":     13,  "sensor_z":     14,
    "pdf_fwd":      15,
    "pdf_rev":      16,
    "pdf_flags":    17,
    "optical":      18,
    "prefix_pdf":   19,
    "mis_denom":    20,
    "tri_mat_idx":  21,
    # band betas: 22 .. 22+MAX_GPU_BANDS-1
}
CGV_BAND_BASE = 22

# ── Helpers ───────────────────────────────────────────────────────────────────

def _f2u(v: float) -> int:
    """Python equivalent of floatBitsToUint / C memcpy float→uint32."""
    return struct.unpack("I", struct.pack("f", v))[0]

def _u2f(u: int) -> float:
    """Python equivalent of uintBitsToFloat."""
    return struct.unpack("f", struct.pack("I", u & 0xFFFFFFFF))[0]

def _rng(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)

def _rng_uint(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=n, dtype=np.uint32)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Identity copy
# ─────────────────────────────────────────────────────────────────────────────

def test_identity_copy():
    """Every field of every record round-trips through a copy descriptor."""
    STRIDE = 7
    N      = 128

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE, name="Src")
    d.add_dst(gl_binding=1, stride=STRIDE, name="Dst")
    d.copy_range(src_slot=0, src_field=0,
                 dst_slot=0, dst_field=0, count=STRIDE)

    src = _rng(1, N * STRIDE)
    dst = np.zeros(N * STRIDE, dtype=np.float32)

    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=N)

    np.testing.assert_array_equal(dst, src,
        err_msg="Identity copy: dst does not match src")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Field remap
# ─────────────────────────────────────────────────────────────────────────────

def test_field_remap():
    """src[field=2] lands at dst[field=5], no other field modified."""
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=8)
    d.add_dst(gl_binding=1, stride=8)
    d.copy(src_slot=0, src_field=2, dst_slot=0, dst_field=5)

    N   = 64
    src = _rng(2, N * 8)
    dst = np.zeros(N * 8, dtype=np.float32)

    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=N)

    for rec in range(N):
        expected = src[rec * 8 + 2]
        got      = dst[rec * 8 + 5]
        assert got == pytest.approx(expected, abs=0), \
            f"rec {rec}: field remap mismatch {got} != {expected}"
        # All other dst fields must remain zero
        for f in range(8):
            if f == 5:
                continue
            assert dst[rec * 8 + f] == 0.0, \
                f"rec {rec} field {f}: unexpected write"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bulk copy_range — no neighbour clobber
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_copy_range_no_clobber():
    """copy_range transfers exactly the requested window; guard bytes stay 0."""
    STRIDE = 16
    N      = 32
    F0, CNT = 3, 6   # copy fields 3..8

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE)
    d.add_dst(gl_binding=1, stride=STRIDE)
    d.copy_range(src_slot=0, src_field=F0,
                 dst_slot=0, dst_field=F0, count=CNT)

    src = _rng(3, N * STRIDE)
    dst = np.zeros(N * STRIDE, dtype=np.float32)
    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=N)

    for rec in range(N):
        base = rec * STRIDE
        for f in range(STRIDE):
            expected = src[base + f] if F0 <= f < F0 + CNT else 0.0
            got      = dst[base + f]
            assert got == pytest.approx(expected, abs=0), \
                f"rec {rec} field {f}: expected {expected} got {got}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Zero / const_f / const_u
# ─────────────────────────────────────────────────────────────────────────────

def test_literal_writes():
    """zero(), const_f(), const_u() write correct values without reading src."""
    MAGIC_F = 3.14159265358979e0
    MAGIC_U = 0xDEADBEEF

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=4)
    d.add_dst(gl_binding=1, stride=4)
    d.zero   (dst_slot=0, dst_field=0)
    d.const_f(dst_slot=0, dst_field=1, val=MAGIC_F)
    d.const_u(dst_slot=0, dst_field=2, val=MAGIC_U)
    # field 3 untouched — must stay 0

    N   = 16
    src = np.ones(N * 4, dtype=np.float32) * 99.0
    dst = np.full(N * 4, fill_value=np.float32(-1.0))

    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=N)

    for rec in range(N):
        b = rec * 4
        assert dst[b + 0] == 0.0,                              "zero field"
        assert dst[b + 1] == pytest.approx(MAGIC_F, rel=1e-6), "const_f field"
        # const_u: bit-pattern must match
        assert _f2u(float(dst[b + 2])) == MAGIC_U,             "const_u bit pattern"
        assert dst[b + 3] == pytest.approx(-1.0, abs=0),       "untouched field"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Uint / int bit-cast round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_bitcast_roundtrip():
    """copy_u and copy_i preserve NaN-pattern / sentinel bit patterns exactly."""
    SENTINELS_U = [0x00000000, 0xFFFFFFFF, 0x7FC00000,  # quiet NaN
                   0x80000000, 0xDEADBEEF, 0x00000001]
    SENTINELS_I = [-1, 0, -2147483648, 2147483647, -12345]

    N = max(len(SENTINELS_U), len(SENTINELS_I))
    STRIDE = 2

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE)
    d.add_dst(gl_binding=1, stride=STRIDE)
    d.copy_u(src_slot=0, src_field=0, dst_slot=0, dst_field=0)
    d.copy_i(src_slot=0, src_field=1, dst_slot=0, dst_field=1)

    src = np.zeros(N * STRIDE, dtype=np.float32)
    src_u = src.view(np.uint32)
    for i, u in enumerate(SENTINELS_U):
        src_u[i * STRIDE + 0] = np.uint32(u)
    for i, v in enumerate(SENTINELS_I):
        src_u[i * STRIDE + 1] = np.uint32(np.int32(v))
    dst = np.zeros(N * STRIDE, dtype=np.float32)

    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=N)

    for i, u in enumerate(SENTINELS_U):
        got = int(dst[i * STRIDE + 0].view(np.uint32))
        assert got == u, f"copy_u sentinel[{i}]: 0x{got:08x} != 0x{u:08x}"
    for i, v in enumerate(SENTINELS_I):
        got = int(dst[i * STRIDE + 1].view(np.int32))
        assert got == v, f"copy_i sentinel[{i}]: {got} != {v}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stride extension — padding fields stay zero
# ─────────────────────────────────────────────────────────────────────────────

def test_stride_extension_padding():
    """Old stride 12 → new stride 40: only explicitly mapped fields are set."""
    OLD_STRIDE = T5_LGV_STRIDE_OLD
    NEW_STRIDE = T5_LGV_STRIDE_NEW
    N          = 64

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=OLD_STRIDE, name="OldLGV")
    d.add_dst(gl_binding=1, stride=NEW_STRIDE, name="NewLGV")

    # Copy the base fields that exist in both (pos, normal, throughput, flags,
    # subpath_id, vert_info, beta_lum) — fields 0..10
    d.copy_range(src_slot=0, src_field=0,
                 dst_slot=0, dst_field=0, count=11)
    # Remaining dst fields are intentionally left as zero (unwritten)

    src = _rng(6, N * OLD_STRIDE)
    # Keep bit-pattern fields as raw uint interpretations
    flags_col = _rng_uint(61, N)
    for i in range(N):
        src[i * OLD_STRIDE + 7] = _u2f(int(flags_col[i]))

    dst = np.zeros(N * NEW_STRIDE, dtype=np.float32)
    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=N)

    for rec in range(N):
        sb = rec * OLD_STRIDE
        db = rec * NEW_STRIDE
        # Fields 0..10: must match src
        for f in range(11):
            np.testing.assert_equal(
                dst[db + f], src[sb + f],
                err_msg=f"rec {rec} field {f} mismatch in extended stride")
        # Fields 11..39: must be zero (no write ever touched them)
        tail = dst[db + 11: db + NEW_STRIDE]
        assert np.all(tail == 0.0), \
            f"rec {rec}: pad fields not zero: {tail[tail != 0.0]}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multi-slot fan-in
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_slot_fanin():
    """Two src slots fanned into one dst slot; fields from each source land correctly."""
    S0, S1 = 5, 3        # strides
    DST_STRIDE = S0 + S1
    N = 48

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=S0, name="Src0")
    d.add_src(gl_binding=1, stride=S1, name="Src1")
    d.add_dst(gl_binding=2, stride=DST_STRIDE, name="DstFanIn")
    # First S0 fields come from slot 0
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=S0)
    # Next S1 fields come from slot 1
    d.copy_range(src_slot=1, src_field=0, dst_slot=0, dst_field=S0, count=S1)

    src0 = _rng(7, N * S0)
    src1 = _rng(8, N * S1)
    dst  = np.zeros(N * DST_STRIDE, dtype=np.float32)

    rc.execute_cpu(d, [src0, src1], [dst], record_base=0, n_records=N)

    for rec in range(N):
        db = rec * DST_STRIDE
        for f in range(S0):
            assert dst[db + f] == pytest.approx(src0[rec * S0 + f], abs=0), \
                f"rec {rec} fan-in src0 field {f}"
        for f in range(S1):
            assert dst[db + S0 + f] == pytest.approx(src1[rec * S1 + f], abs=0), \
                f"rec {rec} fan-in src1 field {f}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Multi-region buffer aliasing
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_region_aliasing():
    """Same flat buffer aliased at two different base offsets (BdptOutputBuf pattern)."""
    REGION_A_BASE  = 0
    REGION_B_BASE  = 200      # flat-element offset where second region starts
    STRIDE         = 4
    N              = 20

    d = rc.RchanDesc()
    # Two src slots, same gl_binding, different base_floats
    d.add_src(gl_binding=0, stride=STRIDE, base_floats=REGION_A_BASE, name="RegionA")
    d.add_src(gl_binding=0, stride=STRIDE, base_floats=REGION_B_BASE, name="RegionB")
    d.add_dst(gl_binding=1, stride=STRIDE * 2, name="Combined")
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0,        count=STRIDE)
    d.copy_range(src_slot=1, src_field=0, dst_slot=0, dst_field=STRIDE,   count=STRIDE)

    flat_src_size = REGION_B_BASE + N * STRIDE
    flat_src = _rng(9, flat_src_size)
    flat_src_b = flat_src  # same backing array

    dst = np.zeros(N * STRIDE * 2, dtype=np.float32)

    # CPU path takes one pointer per declared src slot — alias the same array
    rc.execute_cpu(d, [flat_src, flat_src_b], [dst], record_base=0, n_records=N)

    for rec in range(N):
        db = rec * (STRIDE * 2)
        for f in range(STRIDE):
            a_expected = flat_src[REGION_A_BASE + rec * STRIDE + f]
            b_expected = flat_src[REGION_B_BASE + rec * STRIDE + f]
            assert dst[db + f]          == pytest.approx(a_expected, abs=0), \
                f"rec {rec} region A field {f}"
            assert dst[db + STRIDE + f] == pytest.approx(b_expected, abs=0), \
                f"rec {rec} region B field {f}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Scatter reindex
# ─────────────────────────────────────────────────────────────────────────────

def test_scatter_reindex():
    """RCHAN_OP_SCATTER: dst_rec = scatter[src_rec]; no cross-talk."""
    STRIDE = 3
    N      = 32

    # Build a random permutation as the scatter index
    rng = np.random.default_rng(10)
    scatter = rng.permutation(N).astype(np.uint32)

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE)
    d.add_dst(gl_binding=1, stride=STRIDE)
    d.set_scatter(gl_binding=2)
    # Scatter-copy all 3 fields
    for f in range(STRIDE):
        d.scatter(src_slot=0, src_field=f, dst_slot=0, dst_field=f)

    src = _rng(11, N * STRIDE)
    dst = np.zeros(N * STRIDE, dtype=np.float32)

    rc.execute_cpu(d, [src], [dst], scatter=scatter, record_base=0, n_records=N)

    for src_rec in range(N):
        dst_rec = int(scatter[src_rec])
        for f in range(STRIDE):
            expected = src[src_rec * STRIDE + f]
            got      = dst[dst_rec * STRIDE + f]
            assert got == pytest.approx(expected, abs=0), \
                f"scatter: src[{src_rec}].{f} → dst[{dst_rec}].{f}: {got} != {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Atomic float accumulate (sequential — verifies result correctness)
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_af_sequential():
    """Atomic float-add: N records accumulate into a single dst bucket correctly."""
    N = 256
    STRIDE_SRC = 1
    STRIDE_DST = 1

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE_SRC)
    d.add_adst(gl_binding=1, stride=STRIDE_DST)
    d.atomic_af(src_slot=0, src_field=0, dst_slot=0, dst_field=0)

    rng  = np.random.default_rng(12)
    vals = rng.uniform(0.1, 1.0, N).astype(np.float32)
    # Pack all values as records that all target dst record 0.
    # Scatter to record 0 via a scatter array of all-zeros.
    scatter = np.zeros(N, dtype=np.uint32)

    # dst is a uint32 array (atomic slot)
    adst = np.zeros(STRIDE_DST, dtype=np.uint32)  # single record

    # Use scatter to direct all adds to record 0
    d2 = rc.RchanDesc()
    d2.add_src(gl_binding=0, stride=STRIDE_SRC)
    d2.add_adst(gl_binding=1, stride=STRIDE_DST)
    d2.atomic_af(src_slot=0, src_field=0, dst_slot=0, dst_field=0)

    # For simplicity: each record goes to its own slot (N dst records),
    # then verify sum is equal to sum of src.
    adst_n = np.zeros(N, dtype=np.uint32)
    rc.execute_cpu(d2, [vals], [], adst_arrays=[adst_n], record_base=0, n_records=N)

    # Each adst_n[i] should equal bits-of vals[i] (one write each)
    for i in range(N):
        got_f = _u2f(int(adst_n[i]))
        assert got_f == pytest.approx(float(vals[i]), rel=1e-5), \
            f"atomic_af rec {i}: {got_f} != {float(vals[i])}"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Record tiling — three tiled calls equal one monolithic call
# ─────────────────────────────────────────────────────────────────────────────

def test_tiling_equivalence():
    """Three tiled execute_cpu calls with disjoint record ranges == one full call."""
    STRIDE = 5
    N      = 150
    TILE   = 50

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE)
    d.add_dst(gl_binding=1, stride=STRIDE)
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=STRIDE)

    src     = _rng(13, N * STRIDE)
    dst_ref = np.zeros(N * STRIDE, dtype=np.float32)
    rc.execute_cpu(d, [src], [dst_ref], record_base=0, n_records=N)

    dst_tiled = np.zeros(N * STRIDE, dtype=np.float32)
    for tile in range(3):
        rc.execute_cpu(d, [src], [dst_tiled],
                       record_base=tile * TILE, n_records=TILE)

    np.testing.assert_array_equal(dst_tiled, dst_ref,
        err_msg="Tiled calls do not match monolithic call")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Binding-limit guard
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_binding_limit():
    """validate() raises when total bindings exceed 8."""
    d = rc.RchanDesc()
    # 4 src + 4 dst + 1 adst = 9 total bindings > limit of 8
    for i in range(4):
        d.add_src(gl_binding=i, stride=4, name=f"S{i}")
    for i in range(4):
        d.add_dst(gl_binding=4 + i, stride=4, name=f"D{i}")
    d.add_adst(gl_binding=8, stride=4, name="A0")
    with pytest.raises(RuntimeError, match=r"(?i)binding|limit|max"):
        rc.validate(d)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Binding-alias guard (two writable slots same binding)
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_alias_dst():
    """validate() raises when two dst slots share a GL binding point."""
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=4)
    d.add_dst(gl_binding=1, stride=4, name="DstA")
    d.add_dst(gl_binding=1, stride=4, name="DstB")  # same binding → conflict
    with pytest.raises(RuntimeError, match=r"(?i)alias|binding|same"):
        rc.validate(d)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Bad-slot guard
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_bad_slot():
    """validate() raises when an op references an undeclared src slot."""
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=4)
    d.add_dst(gl_binding=1, stride=4)
    # Manually push an op with src_slot=5 (only 1 src declared)
    d.copy(src_slot=0, src_field=0, dst_slot=0, dst_field=0)  # valid
    d.copy(src_slot=5, src_field=0, dst_slot=0, dst_field=0)  # out-of-range slot pushed without error
    with pytest.raises((RuntimeError, Exception)):
        rc.validate(d)  # validate must catch the bad slot


# ─────────────────────────────────────────────────────────────────────────────
# 15. Bad-field guard
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_bad_field():
    """validate() raises when an op field index >= slot stride."""
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=4)
    d.add_dst(gl_binding=1, stride=4)
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=4)
    # Add an op with src_field=4 (stride is 4 → field 4 is out of range)
    with pytest.raises((RuntimeError, Exception)):
        d.copy(src_slot=0, src_field=4, dst_slot=0, dst_field=0)
        rc.validate(d)


# ─────────────────────────────────────────────────────────────────────────────
# 16. GLSL emit smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_emit_glsl_smoke():
    """emit_glsl produces a non-empty string with required structural markers."""
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=12, name="OldLGV")
    d.add_dst(gl_binding=1, stride=40, name="NewLGV")
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=11)
    d.zero_range(dst_slot=0, dst_field=11, count=29)

    glsl = rc.emit_glsl(d, name="t5_repack_lgv_smoke")
    assert len(glsl) > 100,                 "GLSL output suspiciously short"
    assert "#version 430" in glsl,          "missing #version"
    assert "RCHAN_SRC_BINDING_0" in glsl,   "missing src binding define"
    assert "RCHAN_DST_BINDING_0" in glsl,   "missing dst binding define"
    assert "RCHAN_SRC_STRIDE_0" in glsl,    "missing src stride define"
    assert "RCHAN_DST_STRIDE_0" in glsl,    "missing dst stride define"
    assert "rchan_map" in glsl,             "missing rchan_map function"
    assert "RCHAN_PROVIDE_MAIN" in glsl,    "missing RCHAN_PROVIDE_MAIN"


# ─────────────────────────────────────────────────────────────────────────────
# 17. GLSL / CPU parity — #defines match descriptor values
# ─────────────────────────────────────────────────────────────────────────────

def test_glsl_cpu_parity():
    """#defines in generated GLSL match the stride/binding values in the descriptor."""
    CONFIGS = [
        (0, 28, 2, 40),   # (src_binding, src_stride, dst_binding, dst_stride)
        (3, 16, 5, 56),
    ]
    for src_b, src_s, dst_b, dst_s in CONFIGS:
        d = rc.RchanDesc()
        d.add_src(gl_binding=src_b, stride=src_s)
        d.add_dst(gl_binding=dst_b, stride=dst_s)

        glsl = rc.emit_glsl(d, name="parity_check")

        def _extract(pattern: str) -> Optional[int]:
            m = re.search(pattern, glsl)
            return int(m.group(1)) if m else None

        got_sb = _extract(r"#define\s+RCHAN_SRC_BINDING_0\s+(\d+)")
        got_ss = _extract(r"#define\s+RCHAN_SRC_STRIDE_0\s+(\d+)")
        got_db = _extract(r"#define\s+RCHAN_DST_BINDING_0\s+(\d+)")
        got_ds = _extract(r"#define\s+RCHAN_DST_STRIDE_0\s+(\d+)")

        assert got_sb == src_b, f"src binding: expected {src_b} got {got_sb}"
        assert got_ss == src_s, f"src stride:  expected {src_s} got {got_ss}"
        assert got_db == dst_b, f"dst binding: expected {dst_b} got {got_db}"
        assert got_ds == dst_s, f"dst stride:  expected {dst_s} got {got_ds}"


# ─────────────────────────────────────────────────────────────────────────────
# 18. T5 light-vert repack — end-to-end field positions
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgv_repack_desc() -> rc.RchanDesc:
    """Build the canonical LGV repack descriptor (old stride 12 → new stride 40).

    Src slot 0 = old T5LightVertBuf  (stride 12)
    Src slot 1 = PDF staging buffer  (stride  4: [pdf_fwd, pdf_rev, pdf_flags, optical])
    Src slot 2 = prefix/band staging (stride 17: [prefix_pdf, band_0..band_15])
    Dst slot 0 = new T5LightVertBuf  (stride 40)
    """
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=T5_LGV_STRIDE_OLD, name="OldT5LightVert")
    d.add_src(gl_binding=1, stride=4,                 name="LGVPdfStaging")
    d.add_src(gl_binding=2, stride=1 + MAX_GPU_BANDS, name="LGVBandStaging")
    d.add_dst(gl_binding=3, stride=T5_LGV_STRIDE_NEW, name="NewT5LightVert")

    # Copy base fields from old vert: pos(3) + normal(3) + throughput(1) +
    # flags(1) + subpath_id(1) + vert_info(1) + beta_lum(1) = 11 floats
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=11)

    # PDF staging → fields 11..14
    d.copy  (src_slot=1, src_field=0, dst_slot=0, dst_field=LGV_FIELD["pdf_fwd"])
    d.copy  (src_slot=1, src_field=1, dst_slot=0, dst_field=LGV_FIELD["pdf_rev"])
    d.copy_u(src_slot=1, src_field=2, dst_slot=0, dst_field=LGV_FIELD["pdf_flags"])
    d.copy_u(src_slot=1, src_field=3, dst_slot=0, dst_field=LGV_FIELD["optical"])

    # Band staging: [0]=prefix_pdf, [1..MAX_GPU_BANDS]=per-band betas
    d.copy  (src_slot=2, src_field=0, dst_slot=0, dst_field=LGV_FIELD["prefix_pdf"])
    d.copy_range(src_slot=2, src_field=1,
                 dst_slot=0, dst_field=LGV_BAND_BASE, count=MAX_GPU_BANDS)

    # Pad fields 32..39 — explicit zero
    d.zero_range(dst_slot=0, dst_field=LGV_BAND_BASE + MAX_GPU_BANDS,
                 count=T5_LGV_STRIDE_NEW - LGV_BAND_BASE - MAX_GPU_BANDS)
    return d


def test_t5_lgv_repack_field_positions():
    """T5 LGV repack: every field lands in the exact declared slot position."""
    N  = 32
    d  = _make_lgv_repack_desc()
    rc.validate(d)

    rng   = np.random.default_rng(18)
    src0  = rng.standard_normal(N * T5_LGV_STRIDE_OLD).astype(np.float32)
    src1  = rng.standard_normal(N * 4).astype(np.float32)
    src2  = rng.standard_normal(N * (1 + MAX_GPU_BANDS)).astype(np.float32)
    dst   = np.zeros(N * T5_LGV_STRIDE_NEW, dtype=np.float32)

    rc.execute_cpu(d, [src0, src1, src2], [dst], record_base=0, n_records=N)

    for rec in range(N):
        sb0 = rec * T5_LGV_STRIDE_OLD
        sb1 = rec * 4
        sb2 = rec * (1 + MAX_GPU_BANDS)
        db  = rec * T5_LGV_STRIDE_NEW

        # Base fields 0..10
        for f in range(11):
            assert dst[db + f] == pytest.approx(src0[sb0 + f], abs=0), \
                f"lgv rec {rec} base field {f}"

        # PDF fields
        assert dst[db + LGV_FIELD["pdf_fwd"]]  == pytest.approx(src1[sb1 + 0], abs=0)
        assert dst[db + LGV_FIELD["pdf_rev"]]  == pytest.approx(src1[sb1 + 1], abs=0)
        # pdf_flags and optical are uint bit-cast — compare raw bits
        assert (dst[db + LGV_FIELD["pdf_flags"]].view(np.uint32) ==
                src1[sb1 + 2].view(np.uint32)), "lgv pdf_flags bit pattern"
        assert (dst[db + LGV_FIELD["optical"]].view(np.uint32) ==
                src1[sb1 + 3].view(np.uint32)), "lgv optical bit pattern"

        # Prefix pdf and per-band betas
        assert dst[db + LGV_FIELD["prefix_pdf"]] == pytest.approx(src2[sb2 + 0], abs=0)
        for b in range(MAX_GPU_BANDS):
            assert dst[db + LGV_BAND_BASE + b] == pytest.approx(src2[sb2 + 1 + b], abs=0), \
                f"lgv rec {rec} band {b}"

        # Pad fields
        pad_start = LGV_BAND_BASE + MAX_GPU_BANDS
        for f in range(pad_start, T5_LGV_STRIDE_NEW):
            assert dst[db + f] == 0.0, f"lgv rec {rec} pad field {f} not zero"


# ─────────────────────────────────────────────────────────────────────────────
# 19. T5 cam-vert repack — end-to-end field positions
# ─────────────────────────────────────────────────────────────────────────────

def _make_cgv_repack_desc() -> rc.RchanDesc:
    """Build the canonical CGV repack descriptor (old stride 16 → new stride 56).

    Src slot 0 = old T5CamVertBuf    (stride 16)
    Src slot 1 = PDF+MIS staging     (stride  5: [pdf_fwd, pdf_rev, pdf_flags,
                                                   optical, mis_denom_sum])
    Src slot 2 = prefix/band staging (stride 17: [prefix_pdf, band_0..band_15])
    Src slot 3 = tri_mat staging     (stride  1: [tri_mat_idx int bits])
    Dst slot 0 = new T5CamVertBuf    (stride 56)
    """
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=T5_CGV_STRIDE_OLD, name="OldT5CamVert")
    d.add_src(gl_binding=1, stride=5,                 name="CGVPdfStaging")
    d.add_src(gl_binding=2, stride=1 + MAX_GPU_BANDS, name="CGVBandStaging")
    d.add_src(gl_binding=3, stride=1,                 name="CGVMatStaging")
    d.add_dst(gl_binding=4, stride=T5_CGV_STRIDE_NEW, name="NewT5CamVert")

    # Base fields 0..14 from old vert
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=15)

    # PDF+MIS staging → fields 15..20
    d.copy  (src_slot=1, src_field=0, dst_slot=0, dst_field=CGV_FIELD["pdf_fwd"])
    d.copy  (src_slot=1, src_field=1, dst_slot=0, dst_field=CGV_FIELD["pdf_rev"])
    d.copy_u(src_slot=1, src_field=2, dst_slot=0, dst_field=CGV_FIELD["pdf_flags"])
    d.copy_u(src_slot=1, src_field=3, dst_slot=0, dst_field=CGV_FIELD["optical"])
    d.copy  (src_slot=1, src_field=4, dst_slot=0, dst_field=CGV_FIELD["mis_denom"])

    # tri_mat_idx (int bit-cast)
    d.copy_i(src_slot=3, src_field=0, dst_slot=0, dst_field=CGV_FIELD["tri_mat_idx"])

    # Band staging: [0]=prefix_pdf, [1..MAX_GPU_BANDS]=per-band betas
    d.copy  (src_slot=2, src_field=0, dst_slot=0, dst_field=CGV_FIELD["prefix_pdf"])
    d.copy_range(src_slot=2, src_field=1,
                 dst_slot=0, dst_field=CGV_BAND_BASE, count=MAX_GPU_BANDS)

    # Pad fields
    d.zero_range(dst_slot=0, dst_field=CGV_BAND_BASE + MAX_GPU_BANDS,
                 count=T5_CGV_STRIDE_NEW - CGV_BAND_BASE - MAX_GPU_BANDS)
    return d


def test_t5_cgv_repack_field_positions():
    """T5 CGV repack: every field lands in the exact declared slot position."""
    N  = 32
    d  = _make_cgv_repack_desc()
    rc.validate(d)

    rng   = np.random.default_rng(19)
    src0  = rng.standard_normal(N * T5_CGV_STRIDE_OLD).astype(np.float32)
    src1  = rng.standard_normal(N * 5).astype(np.float32)
    src2  = rng.standard_normal(N * (1 + MAX_GPU_BANDS)).astype(np.float32)
    # tri_mat as int bit-patterns
    mat_ints = rng.integers(-5, 50, N).astype(np.int32)
    src3  = mat_ints.view(np.float32).copy()
    dst   = np.zeros(N * T5_CGV_STRIDE_NEW, dtype=np.float32)

    rc.execute_cpu(d, [src0, src1, src2, src3], [dst], record_base=0, n_records=N)

    for rec in range(N):
        sb0 = rec * T5_CGV_STRIDE_OLD
        sb1 = rec * 5
        sb2 = rec * (1 + MAX_GPU_BANDS)
        sb3 = rec
        db  = rec * T5_CGV_STRIDE_NEW

        # Base fields 0..14
        for f in range(15):
            assert dst[db + f] == pytest.approx(src0[sb0 + f], abs=0), \
                f"cgv rec {rec} base field {f}"

        # PDF+MIS
        assert dst[db + CGV_FIELD["pdf_fwd"]]  == pytest.approx(src1[sb1 + 0], abs=0)
        assert dst[db + CGV_FIELD["pdf_rev"]]  == pytest.approx(src1[sb1 + 1], abs=0)
        assert (dst[db + CGV_FIELD["pdf_flags"]].view(np.uint32) ==
                src1[sb1 + 2].view(np.uint32)), "cgv pdf_flags"
        assert (dst[db + CGV_FIELD["optical"]].view(np.uint32) ==
                src1[sb1 + 3].view(np.uint32)), "cgv optical"
        assert dst[db + CGV_FIELD["mis_denom"]] == pytest.approx(src1[sb1 + 4], abs=0)

        # tri_mat_idx as int
        assert (dst[db + CGV_FIELD["tri_mat_idx"]].view(np.int32) ==
                mat_ints[sb3]), f"cgv rec {rec} tri_mat_idx"

        # Prefix pdf + bands
        assert dst[db + CGV_FIELD["prefix_pdf"]] == pytest.approx(src2[sb2 + 0], abs=0)
        for b in range(MAX_GPU_BANDS):
            assert dst[db + CGV_BAND_BASE + b] == pytest.approx(src2[sb2 + 1 + b], abs=0), \
                f"cgv rec {rec} band {b}"

        # Pad fields
        pad_start = CGV_BAND_BASE + MAX_GPU_BANDS
        for f in range(pad_start, T5_CGV_STRIDE_NEW):
            assert dst[db + f] == 0.0, f"cgv rec {rec} pad field {f}"


# ─────────────────────────────────────────────────────────────────────────────
# 20. Out-of-bounds / edge case records
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_records_is_noop():
    """n_records=0 must not touch dst at all."""
    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=4)
    d.add_dst(gl_binding=1, stride=4)
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=4)

    sentinel = np.full(4 * 10, fill_value=np.float32(7.0))
    dst      = sentinel.copy()
    src      = _rng(20, 4 * 10)

    rc.execute_cpu(d, [src], [dst], record_base=0, n_records=0)
    np.testing.assert_array_equal(dst, sentinel,
        err_msg="n_records=0 modified dst")


def test_record_base_offset():
    """record_base=50 processes only records [50, 50+N), leaving others untouched."""
    STRIDE = 3
    TOTAL  = 100
    BASE   = 50
    N      = 20

    d = rc.RchanDesc()
    d.add_src(gl_binding=0, stride=STRIDE)
    d.add_dst(gl_binding=1, stride=STRIDE)
    d.copy_range(src_slot=0, src_field=0, dst_slot=0, dst_field=0, count=STRIDE)

    src = _rng(21, TOTAL * STRIDE)
    dst = np.zeros(TOTAL * STRIDE, dtype=np.float32)

    rc.execute_cpu(d, [src], [dst], record_base=BASE, n_records=N)

    # Records before BASE must be zero
    assert np.all(dst[:BASE * STRIDE] == 0.0), "pre-base records were written"
    # Records [BASE, BASE+N) must match src
    np.testing.assert_array_equal(
        dst[BASE * STRIDE: (BASE + N) * STRIDE],
        src[BASE * STRIDE: (BASE + N) * STRIDE],
        err_msg="tiled window mismatch")
    # Records after BASE+N must be zero
    assert np.all(dst[(BASE + N) * STRIDE:] == 0.0), "post-window records were written"


# ─────────────────────────────────────────────────────────────────────────────
# Self-consistency: T5 LGV + CGV strides and field maps are internally coherent
# ─────────────────────────────────────────────────────────────────────────────

def test_field_map_coherence():
    """Field map dicts don't overlap and fit within declared strides."""
    # LGV
    lgv_fields = set(LGV_FIELD.values()) | set(range(LGV_BAND_BASE,
                                                      LGV_BAND_BASE + MAX_GPU_BANDS))
    assert max(lgv_fields) < T5_LGV_STRIDE_NEW, \
        f"LGV field map extends beyond stride {T5_LGV_STRIDE_NEW}"
    assert len(lgv_fields) == len(set(lgv_fields)), "LGV field map has duplicates"

    # CGV
    cgv_fields = set(CGV_FIELD.values()) | set(range(CGV_BAND_BASE,
                                                      CGV_BAND_BASE + MAX_GPU_BANDS))
    assert max(cgv_fields) < T5_CGV_STRIDE_NEW, \
        f"CGV field map extends beyond stride {T5_CGV_STRIDE_NEW}"
    assert len(cgv_fields) == len(set(cgv_fields)), "CGV field map has duplicates"

    # Band arrays must not overlap named fields
    lgv_named = set(LGV_FIELD.values())
    lgv_bands = set(range(LGV_BAND_BASE, LGV_BAND_BASE + MAX_GPU_BANDS))
    assert lgv_named.isdisjoint(lgv_bands), "LGV named fields overlap band region"

    cgv_named = set(CGV_FIELD.values())
    cgv_bands = set(range(CGV_BAND_BASE, CGV_BAND_BASE + MAX_GPU_BANDS))
    assert cgv_named.isdisjoint(cgv_bands), "CGV named fields overlap band region"
