"""mat_flags.py
================
Single source of truth for the unified material-flag bit space and the SSBO
binding indices shared between the GLSL forward/backward compute shaders and
the C++ `_spectral_kernels.RayTracer`.

Why one module
--------------
Previously the bit values lived in three places — a comment block in the GLSL
preamble of `_GPU_RAY_FIELD_CS`, a parallel comment in `_GPU_SENSOR_CS`, and
the `RT_TRI_FLAG_*` enum in `csrc/kernels/ray_tracer.h`.  Drift between them
was a real risk and would have caused silent semantic mismatches between the
two backends.  This module exports:

  * Python ints         — for the Python tri-packer and bit-test code
  * `glsl_preamble()`   — emits the `#define MAT_FLAG_* …` block injected at
                          the top of every shader source string
  * `cpp_header_text()` — emits the `#define MAT_FLAG_* …` block written into
                          a generated C++ header (so the C++ tracer sees the
                          identical bit values without manual mirroring)

Bit space (single union of GLSL + C++)
--------------------------------------
The original GLSL bit space was bits 0..6 (EMISSIVE..TRANSMISSIVE).  The
original C++ bit space was bits 0..1 (TRANSMISSIVE, APERTURE_STOP).  These
are reconciled here: TRANSMISSIVE keeps GLSL's value 64u; APERTURE_STOP gets
a fresh slot (128u) that the C++ side will switch to during the cutover.
"""
from __future__ import annotations

# ── Material flag bits (uint, packed into TriShade.mat_flags) ────────────────

MAT_FLAG_EMISSIVE      = 1
MAT_FLAG_REACTIVE      = 2
MAT_FLAG_ABSORBER      = 4
MAT_FLAG_NO_SHADOW     = 8
MAT_FLAG_MANIFOLD      = 16
MAT_FLAG_PARAMETRIC    = 32   # dispatch to ParamSurf intersector (lens-bake mode)
MAT_FLAG_TRANSMISSIVE  = 64
MAT_FLAG_APERTURE_STOP = 128  # opaque blade — kills ray; diffraction handled downstream
MAT_FLAG_PICKING_ONLY  = 256  # tracer-5 mode: nearest-hit, no spectrum, no bounce

ALL_FLAGS = {
    "MAT_FLAG_EMISSIVE":      MAT_FLAG_EMISSIVE,
    "MAT_FLAG_REACTIVE":      MAT_FLAG_REACTIVE,
    "MAT_FLAG_ABSORBER":      MAT_FLAG_ABSORBER,
    "MAT_FLAG_NO_SHADOW":     MAT_FLAG_NO_SHADOW,
    "MAT_FLAG_MANIFOLD":      MAT_FLAG_MANIFOLD,
    "MAT_FLAG_PARAMETRIC":    MAT_FLAG_PARAMETRIC,
    "MAT_FLAG_TRANSMISSIVE":  MAT_FLAG_TRANSMISSIVE,
    "MAT_FLAG_APERTURE_STOP": MAT_FLAG_APERTURE_STOP,
    "MAT_FLAG_PICKING_ONLY":  MAT_FLAG_PICKING_ONLY,
}

# ── SSBO binding indices (single source of truth for both backends) ──────────
# These are the GL std430 binding points used by _GPU_RAY_FIELD_CS and
# _GPU_SENSOR_CS.  C++ ingest functions take parallel pointer arguments in
# the same order so the layout is one schema.
BINDING_TRI_GEOM        = 0    # TriGeomBuf  : (N_tri, 16) float32 — 64 B/tri
BINDING_NODE            = 1    # NodeBuf     : BVH node SoA
BINDING_TRI_ID          = 2    # TriIdBuf    : BVH triangle index list
BINDING_BDPT_SOURCE     = 5    # SourceBuf   : (N_src, 12) float32 (canonical)
BINDING_PROFILE         = 7    # ProfileBuf  : EmissionProfileDatabase RGB tensor
BINDING_SCALE_CONTEXT   = 8    # ScaleCtxBuf : (N_ctx, 8) float32
BINDING_TRI_SHADE       = 9    # TriShadeBuf : (N_tri, 16) float32 — 64 B/tri
BINDING_MAT             = 10   # MatBuf      : (N_mat, 32, 12) float32 spectral
BINDING_PARAM_SURF      = 11   # ParamSurfBuf: parametric-surface descriptors
BINDING_BAKE_SURFACE    = 12   # BakeSurfBuf : per-tri surface-illumination map
BINDING_BAKE_VOLUME     = 13   # BakeVolBuf  : per-tri volumetric path map
BINDING_BAKE_LIFESPAN   = 14   # BakeLifeBuf : ray-lifespan transform records

# ── Spectral storage cap (compile-time max; runtime n_bands ≤ this) ──────────
# Set to 32 to give 6 cache-lines of complex spectral data per material.
# Runtime n_bands is honored as a power-of-two ≤ MAX_SPECTRAL_BANDS.
MAX_SPECTRAL_BANDS = 32


def glsl_preamble() -> str:
    """Emit the `#define` block that must be the first lines after `#version`
    in any shader that reads MAT_FLAG_* bits or SSBO binding indices.

    The Python tri packer and the GLSL shaders MUST agree on these values;
    by injecting from this single source the two cannot drift.
    """
    lines: list[str] = ["// === auto-generated from mat_flags.py — DO NOT EDIT ==="]
    for name, val in ALL_FLAGS.items():
        lines.append(f"#define {name:<22s} {val}u")
    lines.append(f"#define MAX_SPECTRAL_BANDS     {MAX_SPECTRAL_BANDS}")
    lines.append("// ===================================================")
    return "\n".join(lines) + "\n"


def cpp_header_text(guard: str = "MAT_FLAGS_GENERATED_H") -> str:
    """Emit a C++ header containing the same bit values + binding indices.

    Written to `csrc/kernels/mat_flags_generated.h` by the build step (or by
    a one-shot tool) so the C++ ray tracer never hand-mirrors these values.
    """
    out: list[str] = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "/* auto-generated from mat_flags.py — DO NOT EDIT */",
        "",
    ]
    for name, val in ALL_FLAGS.items():
        out.append(f"static constexpr unsigned {name} = {val}u;")
    out.append("")
    out.append(f"static constexpr int MAX_SPECTRAL_BANDS = {MAX_SPECTRAL_BANDS};")
    out.append("")
    out.append("/* SSBO binding indices (informational; std430 only) */")
    for name, val in [
        ("BINDING_TRI_GEOM",      BINDING_TRI_GEOM),
        ("BINDING_NODE",          BINDING_NODE),
        ("BINDING_TRI_ID",        BINDING_TRI_ID),
        ("BINDING_BDPT_SOURCE",   BINDING_BDPT_SOURCE),
        ("BINDING_PROFILE",       BINDING_PROFILE),
        ("BINDING_SCALE_CONTEXT", BINDING_SCALE_CONTEXT),
        ("BINDING_TRI_SHADE",     BINDING_TRI_SHADE),
        ("BINDING_MAT",           BINDING_MAT),
        ("BINDING_PARAM_SURF",    BINDING_PARAM_SURF),
        ("BINDING_BAKE_SURFACE",  BINDING_BAKE_SURFACE),
        ("BINDING_BAKE_VOLUME",   BINDING_BAKE_VOLUME),
        ("BINDING_BAKE_LIFESPAN", BINDING_BAKE_LIFESPAN),
    ]:
        out.append(f"static constexpr int {name} = {val};")
    out.append("")
    out.append(f"#endif /* {guard} */")
    return "\n".join(out) + "\n"


def write_cpp_header(path: str = "csrc/kernels/mat_flags_generated.h") -> None:
    """Write the generated C++ header to `path` (idempotent)."""
    import os
    text = cpp_header_text()
    try:
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == text:
                return
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    print(glsl_preamble())
    print(cpp_header_text())
