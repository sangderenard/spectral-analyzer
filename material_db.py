"""material_db.py
================
Central material registry with chunked tensor export.

Architecture
------------
Every material in the scene is registered by name once at startup.  At the
end of registration the database renders all materials into **chunked**
(property-group-major) float32 tensors.  These tensors are the ONE source
of truth consulted by the hot graphics loop — they are plain row-indexable
arrays, NOT keyed lookups, NOT runtime profile resolutions.

    pbr_chunk[mat_ids, 8:11]     # ← what a hot loop actually does

All spectral profile resolution (SPD → linear sRGB), all enamel packing,
all derived-shader downsamples are performed once inside build_tensors()
before the first frame.  Nothing in the draw loop ever queries the
EmissionProfileDatabase or any other auxiliary store.

Chunked layout (NOT interleaved / AoS)
---------------------------------------
For N registered materials each tensor is shape (N, stride_floats):

    pbr_chunk         (N, 16)            — full authored PBR record
    texture_stack     (N, 16)            — optional base-shader texture stack
    phong_compat      (N,  8)            — DOWNSTREAM-ONLY Phong sink
    raymat_compat     (N, 16)            — DOWNSTREAM-ONLY raytracer sink
    spectral_chunk    (N, MAX_BANDS, 12) — per-frequency-band properties
    enamel_chunk      (N,  8)            — thin-film enamel coating

The `_compat` chunks are NOT alternative authoring formats; they are
strictly post-conversion prebakes for shader paths that cannot consume the
full PBR record.  Authors never write to them, never consult them, and
never name their fields in YAML.  A shader that only knows Phong samples
phong_compat; a shader that only knows mat11 samples raymat_compat.  Both
were derived from the same PBR + spectral records at startup.

ctypes structures
-----------------
Each chunk row is defined as a ctypes.Structure whose binary layout is
identical whether the buffer is handed to an OpenGL SSBO, a C extension, or
consumed as a numpy view via np.frombuffer.  The corresponding GLSL struct
is documented inline.

GLSL binding sketch (std430)
----------------------------
    // binding = 10
    layout(std430, binding=10) readonly buffer PBRChunk   { float pbr[];      };
    // binding = 11  (phong_compat — downstream-only)
    layout(std430, binding=11) readonly buffer PhongChunk { float phong[];    };
    // binding = 12  (raymat_compat — downstream-only)
    layout(std430, binding=12) readonly buffer RayChunk   { float ray[];      };
    // binding = 13
    layout(std430, binding=13) readonly buffer SpecChunk  { float spectral[]; };
    // binding = 14
    layout(std430, binding=14) readonly buffer EnamChunk  { float enamel[];   };

    // Accessors (PBR example):
    vec3 mat_albedo(int id) {
        int base = id * 16;
        return vec3(pbr[base], pbr[base+1], pbr[base+2]);
    }

Compatibility extraction (post-bake, for legacy callers only)
-------------------------------------------------------------
    db.as_compat_mat11(name)     → 11-float32 array (legacy raytracer feed)
    db.as_compat_phong(name)     → dict with Phong uniform keys
    db.compat_mat11_tensor()     → (N, 11) float32 for all materials
"""
from __future__ import annotations

import ctypes
import struct as _struct
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SPECTRAL_BANDS: int = 8   # bands stored per material in the spectral chunk

# ─────────────────────────────────────────────────────────────────────────────
# ctypes record structures
# Each field group is vec4-aligned (4×float = 16 bytes) for std430 safety.
# _pack_ = 4 prevents the compiler from inserting platform-specific padding.
# ─────────────────────────────────────────────────────────────────────────────

class PBRBaseRecord(ctypes.Structure):
    """16 floats = 64 bytes (4 × vec4).

    GLSL equivalent:
        struct PBRBase {
            vec4 albedo_rough;     // xyz=albedo, w=roughness
            vec4 metal_trans_ior;  // x=metallic, y=transmission, z=ior, w=opacity
            vec4 emission;         // xyz=emission_rgb, w=_pad
            vec4 _reserved;        // flags / texture ids (future)
        };
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0
        ('albedo',       ctypes.c_float * 3),  # linear sRGB [0,1]
        ('roughness',    ctypes.c_float),       # 0=mirror, 1=Lambertian
        # vec4 1
        ('metallic',     ctypes.c_float),       # 0=dielectric, 1=conductor
        ('transmission', ctypes.c_float),       # bulk transmittance [0,1]
        ('ior',          ctypes.c_float),       # index of refraction
        ('opacity',      ctypes.c_float),       # 1=opaque, 0=fully transparent
        # vec4 2
        ('emission',     ctypes.c_float * 3),  # self-emission RGB
        ('_pad0',        ctypes.c_float),
        # vec4 3  (reserved — shader flags, atlas UVs, etc.)
        ('_reserved',    ctypes.c_float * 4),
    ]

assert ctypes.sizeof(PBRBaseRecord) == 16 * 4, "PBRBaseRecord layout broken"


class PhongRecord(ctypes.Structure):
    """8 floats = 32 bytes (2 × vec4).

    GLSL equivalent:
        struct Phong {
            vec4 params;       // x=ambient, y=spec_strength, z=shininess, w=grain
            vec4 inner_color;  // xyz=back-face tint, w=_pad
        };
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0
        ('ambient',       ctypes.c_float),
        ('spec_strength', ctypes.c_float),
        ('shininess',     ctypes.c_float),
        ('grain',         ctypes.c_float),
        # vec4 1
        ('inner_color',   ctypes.c_float * 3),  # back-face / inner surface tint
        ('_pad0',         ctypes.c_float),
    ]

assert ctypes.sizeof(PhongRecord) == 8 * 4, "PhongRecord layout broken"


class RayMatRecord(ctypes.Structure):
    """20 floats = 80 bytes (5 × vec4).

    Extends mat11 with full emissive/flag support for the 8×vec4 Tri format.

    GLSL equivalent:
        struct RayMat {
            vec4 refl_in;    // xyz=reflectivity/diffusion/absorption (front), w=_pad
            vec4 refl_out;   // xyz=reflectivity/diffusion/absorption (back),  w=_pad
            vec4 albedo;     // xyz=surface albedo RGB, w=_pad
            vec4 ior_opac;   // x=ior, y=opacity, z=mat_flags(uint as float), w=_pad
            vec4 emissive;   // xyz=emissive_rgb, w=reactive_shift_hz
        };
    Column map for legacy mat11:
        [0,1,2] = refl_in   → mat11[0:3]
        [4,5,6] = refl_out  → mat11[3:6]
        [8,9,10]= albedo    → mat11[6:9]
        [12]    = ior       → mat11[9]
        [13]    = opacity   → mat11[10]
    mat16 extension:
        [14]    = mat_flags (uint bits stored as float)
        [16,17,18] = emissive_rgb
        [19]    = reactive_shift_hz
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0
        ('refl_in',         ctypes.c_float * 3),   # front: reflectivity, diffusion, absorption
        ('_pad0',           ctypes.c_float),
        # vec4 1
        ('refl_out',        ctypes.c_float * 3),   # back:  reflectivity, diffusion, absorption
        ('_pad1',           ctypes.c_float),
        # vec4 2
        ('albedo',          ctypes.c_float * 3),   # surface albedo RGB
        ('_pad2',           ctypes.c_float),
        # vec4 3
        ('ior',             ctypes.c_float),
        ('opacity',         ctypes.c_float),
        ('mat_flags',       ctypes.c_float),        # uint bits stored as float
        ('_pad3',           ctypes.c_float),
        # vec4 4
        ('emit_profile_idx',  ctypes.c_float),     # float(int) index into EmissionProfileDatabase; -1 = none
        ('remit_profile_idx', ctypes.c_float),     # float(int) index into EmissionProfileDatabase for re-emission; -1 = none
        ('_pad4',             ctypes.c_float),
        ('reactive_shift_hz', ctypes.c_float),     # Stokes shift Hz (0=non-reactive)
    ]

assert ctypes.sizeof(RayMatRecord) == 20 * 4, "RayMatRecord layout broken"


class SpectralBandRecord(ctypes.Structure):
    """12 floats = 48 bytes (3 × vec4) — one frequency band.

    GLSL equivalent:
        struct SpectralBand {
            vec4 freq;    // x=center_hz, y=bandwidth_hz, z=reflectance, w=transmittance
            vec4 diff;    // x=diffuse_frac, y=emission, z=reemission, w=ior_real
            vec4 extra;   // x=ior_imag, yzw=_pad
        };
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0
        ('center_hz',     ctypes.c_float),
        ('bandwidth_hz',  ctypes.c_float),
        ('reflectance',   ctypes.c_float),
        ('transmittance', ctypes.c_float),
        # vec4 1
        ('diffuse_frac',  ctypes.c_float),
        ('emission',      ctypes.c_float),
        ('reemission',    ctypes.c_float),
        ('ior_real',      ctypes.c_float),
        # vec4 2
        ('ior_imag',      ctypes.c_float),
        ('_pad',          ctypes.c_float * 3),
    ]

assert ctypes.sizeof(SpectralBandRecord) == 12 * 4, "SpectralBandRecord layout broken"


class SpectralRecord(ctypes.Structure):
    """MAX_SPECTRAL_BANDS × SpectralBandRecord + 4-int header.

    Total: MAX_BANDS * 48 + 16 bytes.  The n_bands field tells shaders how
    many of the MAX_BANDS slots are populated.
    """
    _pack_ = 4
    _fields_ = [
        ('bands',   SpectralBandRecord * MAX_SPECTRAL_BANDS),
        ('n_bands', ctypes.c_int32),
        ('_pad',    ctypes.c_int32 * 3),
    ]

_SPECTRAL_RECORD_FLOATS = ctypes.sizeof(SpectralRecord) // 4   # total float32 slots
_SPECTRAL_BAND_FLOATS   = MAX_SPECTRAL_BANDS * 12              # pure band data slots


class EnamelRecord(ctypes.Structure):
    """8 floats = 32 bytes (2 × vec4) — thin-film enamel coating.

    GLSL equivalent:
        struct Enamel {
            vec4 film;    // x=thickness_nm, y=ior_real, z=ior_imag, w=roughness
            vec4 color;   // xyz=tint RGB, w=_pad
        };
    thickness_nm == 0  → no enamel coating (shader should fast-path skip).
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0
        ('thickness_nm', ctypes.c_float),   # in nanometres; 0 = disabled
        ('ior_real',     ctypes.c_float),
        ('ior_imag',     ctypes.c_float),
        ('roughness',    ctypes.c_float),
        # vec4 1
        ('color',        ctypes.c_float * 3),   # tint colour [0,1]
        ('_pad0',        ctypes.c_float),
    ]

assert ctypes.sizeof(EnamelRecord) == 8 * 4, "EnamelRecord layout broken"


class TextureStackRecord(ctypes.Structure):
    """16 floats = 64 bytes (4 × vec4) — base-shader texture stack.

    Kept parallel to PBR16 so ordinary shading still fetches a single 64-byte
    PBR row. Texture-enabled paths can opt into this second chunk.

    GLSL/C layout:
        vec4 layers;   // x=emit_uv_layer, y=color_uv_layer,
                       // z=depth_uv_layer, w=remit_uv_layer
        vec4 depth;    // x=depth_scale_mm, y=thickness_scale_mm,
                       // z=depth_bias_mm, w=thickness_bias_mm
        vec4 controls; // x=emit_gain, y=color_blend,
                       // z=direct_lobe_power, w=model_flags_or_indices
        vec4 remit;    // x=remit_gain, y=bulb_radius_mm,
                       // z=remit_decay, w=translucence_gain
    """
    _pack_ = 4
    _fields_ = [
        ('emit_uv_layer',      ctypes.c_float),
        ('color_uv_layer',     ctypes.c_float),
        ('depth_uv_layer',     ctypes.c_float),
        ('remit_uv_layer',     ctypes.c_float),
        ('depth_scale_mm',     ctypes.c_float),
        ('thickness_scale_mm', ctypes.c_float),
        ('depth_bias_mm',      ctypes.c_float),
        ('thickness_bias_mm',  ctypes.c_float),
        ('emit_gain',          ctypes.c_float),
        ('color_blend',        ctypes.c_float),
        ('direct_lobe_power',  ctypes.c_float),
        ('model_flags',        ctypes.c_float),
        ('remit_gain',         ctypes.c_float),
        ('bulb_radius_mm',     ctypes.c_float),
        ('remit_decay',        ctypes.c_float),
        ('translucence_gain',  ctypes.c_float),
    ]


assert ctypes.sizeof(TextureStackRecord) == 16 * 4, "TextureStackRecord layout broken"

# ── Chunk strides in floats ───────────────────────────────────────────────────
PBR_FLOATS      = ctypes.sizeof(PBRBaseRecord)  // 4   # 16
PHONG_FLOATS    = ctypes.sizeof(PhongRecord)    // 4   # 8
RAYMAT_FLOATS   = ctypes.sizeof(RayMatRecord)   // 4   # 20
ENAMEL_FLOATS   = ctypes.sizeof(EnamelRecord)   // 4   # 8
TEXSTACK_FLOATS = ctypes.sizeof(TextureStackRecord) // 4 # 16

# ─────────────────────────────────────────────────────────────────────────────
# Fill helpers: Any (Material object or dict) → record
# ─────────────────────────────────────────────────────────────────────────────

def _fill_pbr(rec: PBRBaseRecord, mat: Any) -> None:
    """Fill from a Material object (spectral_material.Material) or a dict."""
    if hasattr(mat, 'albedo'):
        # Material dataclass path
        a = mat.albedo
        rec.albedo[0], rec.albedo[1], rec.albedo[2] = float(a[0]), float(a[1]), float(a[2])
        rec.roughness    = float(mat.roughness)
        rec.metallic     = float(mat.metallic)
        rec.transmission = float(mat.transmission)
        rec.ior          = float(mat.ior)
        rec.opacity      = max(0.0, 1.0 - float(mat.transmission))
        e = mat.emission_rgb
        rec.emission[0], rec.emission[1], rec.emission[2] = float(e[0]), float(e[1]), float(e[2])
    else:
        # dict path (legacy or inline)
        a = mat.get('albedo', mat.get('albedo_rgb', [0.5, 0.5, 0.5]))
        rec.albedo[0], rec.albedo[1], rec.albedo[2] = float(a[0]), float(a[1]), float(a[2])
        rec.roughness    = float(mat.get('roughness', 0.5))
        rec.metallic     = float(mat.get('metallic',  0.0))
        rec.transmission = float(mat.get('transmission', 0.0))
        rec.ior          = float(mat.get('ior', 1.5))
        rec.opacity      = float(mat.get('opacity', 1.0))
        e = mat.get('emission_rgb', mat.get('emission', [0.0, 0.0, 0.0]))
        rec.emission[0], rec.emission[1], rec.emission[2] = float(e[0]), float(e[1]), float(e[2])


def _fill_pbr_from_mat11_dict(rec: PBRBaseRecord, d: dict) -> None:
    """Fill from a dict that carries mat11-style keys (refl_in, refl_out, albedo_rgb)."""
    alb = d.get('albedo_rgb', [0.5, 0.5, 0.5])
    rec.albedo[0], rec.albedo[1], rec.albedo[2] = float(alb[0]), float(alb[1]), float(alb[2])
    ri = d.get('refl_in', [0.5, 0.0, 0.5])
    # Heuristic back-derivation from mat11:
    # refl=ri[0], diff=ri[1], abso=ri[2] → approximate PBR
    rec.roughness    = float(ri[1])                        # diffusion ≈ roughness
    rec.metallic     = max(0.0, float(ri[0]) - 0.12)       # excess reflectivity → metallic
    rec.transmission = max(0.0, 1.0 - float(d.get('opacity', 1.0)))
    rec.ior          = float(d.get('ior', 1.5))
    rec.opacity      = float(d.get('opacity', 1.0))
    em = d.get('emissive_rgb', [0.0, 0.0, 0.0])
    rec.emission[0], rec.emission[1], rec.emission[2] = float(em[0]), float(em[1]), float(em[2])


def _pbr_to_phong_record(pbr: PBRBaseRecord, mat: Any = None) -> PhongRecord:
    """Derive PhongRecord from a PBRBaseRecord, with optional dict overrides."""
    rec   = PhongRecord()
    rough = pbr.roughness
    metal = pbr.metallic
    rec.ambient       = 0.15 + (1.0 - rough) * 0.05
    rec.spec_strength = metal * 0.85 + (1.0 - metal) * 0.04 * (1.0 - rough)
    rec.shininess     = max(4.0, 2.0 / max(rough ** 2, 0.01))
    rec.grain         = rough * 0.06
    rec.inner_color[0] = pbr.albedo[0] * 0.55
    rec.inner_color[1] = pbr.albedo[1] * 0.55
    rec.inner_color[2] = pbr.albedo[2] * 0.55
    # Honour explicit Phong overrides carried in a dict
    if mat is not None and isinstance(mat, dict):
        if 'ambient'       in mat: rec.ambient       = float(mat['ambient'])
        if 'spec_strength' in mat: rec.spec_strength = float(mat['spec_strength'])
        if 'shininess'     in mat: rec.shininess     = float(mat['shininess'])
        if 'grain'         in mat: rec.grain         = float(mat['grain'])
        ic = mat.get('inner_color', mat.get('albedo_rgb', None))
        if ic is not None:
            rec.inner_color[0] = float(ic[0]) * 0.55
            rec.inner_color[1] = float(ic[1]) * 0.55
            rec.inner_color[2] = float(ic[2]) * 0.55
    return rec


def _pbr_to_ray_record(pbr: PBRBaseRecord) -> RayMatRecord:
    """Derive RayMatRecord from PBRBaseRecord."""
    rec   = RayMatRecord()
    rough = pbr.roughness
    metal = pbr.metallic
    refl  = metal * 0.85 + (1.0 - metal) * (1.0 - rough) * 0.12
    diff  = (1.0 - metal) * rough
    abso  = max(0.0, 1.0 - refl - diff)
    rec.refl_in[0]  = rec.refl_out[0] = refl
    rec.refl_in[1]  = rec.refl_out[1] = diff
    rec.refl_in[2]  = rec.refl_out[2] = abso
    rec.albedo[0]   = pbr.albedo[0]
    rec.albedo[1]   = pbr.albedo[1]
    rec.albedo[2]   = pbr.albedo[2]
    rec.ior         = pbr.ior
    rec.opacity     = pbr.opacity
    # Propagate emissive from PBR record; set MAT_FLAG_EMISSIVE bit if non-zero
    rec.emit_profile_idx  = -1.0   # no emission profile assigned; caller sets via register
    rec.remit_profile_idx = -1.0
    return rec


def _fill_ray_from_mat11_dict(rec: RayMatRecord, d: dict) -> None:
    """Fill RayMatRecord from a mat11-style dict (has refl_in/refl_out keys)."""
    ri = d.get('refl_in',  [0.5, 0.0, 0.5])
    ro = d.get('refl_out', ri)
    al = d.get('albedo_rgb', [0.5, 0.5, 0.5])
    rec.refl_in[0],  rec.refl_in[1],  rec.refl_in[2]  = float(ri[0]), float(ri[1]), float(ri[2])
    rec.refl_out[0], rec.refl_out[1], rec.refl_out[2] = float(ro[0]), float(ro[1]), float(ro[2])
    rec.albedo[0],   rec.albedo[1],   rec.albedo[2]   = float(al[0]), float(al[1]), float(al[2])
    rec.ior     = float(d.get('ior', 1.5))
    rec.opacity = float(d.get('opacity', 1.0))
    rec.mat_flags         = float(d.get('mat_flags', 0.0))
    rec.emit_profile_idx  = float(d.get('emit_profile_idx',  -1.0))
    rec.remit_profile_idx = float(d.get('remit_profile_idx', -1.0))
    rec.reactive_shift_hz = float(d.get('reactive_shift_hz',  0.0))


def _fill_spectral(rec: SpectralRecord, mat: Any) -> None:
    """Fill SpectralRecord from Material.effective_bands() or leave zeroed."""
    if not hasattr(mat, 'effective_bands'):
        rec.n_bands = 0
        return
    bands = mat.effective_bands(n_synthetic=MAX_SPECTRAL_BANDS)[:MAX_SPECTRAL_BANDS]
    rec.n_bands = len(bands)
    for i, b in enumerate(bands):
        br = rec.bands[i]
        br.center_hz     = float(b.center_hz)
        br.bandwidth_hz  = float(b.bandwidth_hz)
        br.reflectance   = float(b.reflectance)
        br.transmittance = float(b.transmittance)
        br.diffuse_frac  = float(b.diffuse_frac)
        br.emission      = float(b.emission)
        br.reemission    = float(b.reemission)
        br.ior_real      = float(b.ior_real)
        br.ior_imag      = float(b.ior_imag)


def _fill_enamel(rec: EnamelRecord, mat: Any) -> None:
    """Fill EnamelRecord from Material.enamel or a dict's 'enamel' key.

    Accepts either:
      * an object with `.enamel` attribute (spectral_material.Material)
      * a dict with an `enamel` sub-dict carrying thickness_nm/thickness_m,
        ior_real, ior_imag, roughness, color_rgb
    Returns a null coating (thickness_nm = 0) when no enamel is declared.
    """
    enamel = None
    if isinstance(mat, dict):
        enamel = mat.get('enamel', None)
    if enamel is None:
        enamel = getattr(mat, 'enamel', None)
    if enamel is None:
        # Null coating — thickness_nm = 0 signals the shader to skip
        rec.thickness_nm = 0.0
        rec.ior_real     = 1.52
        rec.ior_imag     = 0.0
        rec.roughness    = 0.5
        rec.color[0] = rec.color[1] = rec.color[2] = 1.0
        return
    if isinstance(enamel, dict):
        if 'thickness_nm' in enamel:
            rec.thickness_nm = float(enamel['thickness_nm'])
        else:
            rec.thickness_nm = float(enamel.get('thickness_m', 0.0)) * 1.0e9
        rec.ior_real  = float(enamel.get('ior_real', 1.52))
        rec.ior_imag  = float(enamel.get('ior_imag', 0.0))
        rec.roughness = float(enamel.get('roughness', 0.05))
        c = enamel.get('color_rgb', [1.0, 1.0, 1.0])
    else:
        rec.thickness_nm = float(getattr(enamel, 'thickness_m', 0.0)) * 1.0e9
        rec.ior_real     = float(getattr(enamel, 'ior_real', 1.52))
        rec.ior_imag     = float(getattr(enamel, 'ior_imag', 0.0))
        rec.roughness    = float(getattr(enamel, 'roughness', 0.05))
        c = getattr(enamel, 'color_rgb', [1.0, 1.0, 1.0])
    rec.color[0], rec.color[1], rec.color[2] = float(c[0]), float(c[1]), float(c[2])


def _fill_texture_stack(rec: TextureStackRecord, mat: Any) -> None:
    """Fill optional UV texture-stack metadata from dict/object attributes."""
    rec.emit_uv_layer = -1.0
    rec.color_uv_layer = -1.0
    rec.depth_uv_layer = -1.0
    rec.remit_uv_layer = -1.0
    rec.depth_scale_mm = 0.0
    rec.thickness_scale_mm = 0.0
    rec.depth_bias_mm = 0.0
    rec.thickness_bias_mm = 0.0
    rec.emit_gain = 1.0
    rec.color_blend = 1.0
    rec.direct_lobe_power = 32.0
    rec.model_flags = 0.0
    rec.remit_gain = 0.0
    rec.bulb_radius_mm = 0.0
    rec.remit_decay = 0.0
    rec.translucence_gain = 0.0

    if isinstance(mat, dict):
        src = mat.get('texture_stack', mat)
        if not isinstance(src, dict):
            src = mat
        rec.emit_uv_layer = float(src.get('emit_uv_layer', rec.emit_uv_layer))
        rec.color_uv_layer = float(src.get('color_uv_layer', rec.color_uv_layer))
        rec.depth_uv_layer = float(src.get('depth_uv_layer', src.get('depth_layer', rec.depth_uv_layer)))
        rec.remit_uv_layer = float(src.get('remit_uv_layer', src.get('remit_mask_layer', rec.remit_uv_layer)))
        rec.depth_scale_mm = float(src.get('depth_scale_mm', rec.depth_scale_mm))
        rec.thickness_scale_mm = float(src.get('thickness_scale_mm', rec.thickness_scale_mm))
        rec.depth_bias_mm = float(src.get('depth_bias_mm', rec.depth_bias_mm))
        rec.thickness_bias_mm = float(src.get('thickness_bias_mm', rec.thickness_bias_mm))
        rec.emit_gain = float(src.get('emit_gain', rec.emit_gain))
        rec.color_blend = float(src.get('color_blend', rec.color_blend))
        rec.direct_lobe_power = float(src.get('direct_lobe_power', rec.direct_lobe_power))
        rec.model_flags = float(src.get('model_flags', src.get('texture_flags', src.get('flags', rec.model_flags))))
        rec.remit_gain = float(src.get('remit_gain', rec.remit_gain))
        rec.bulb_radius_mm = float(src.get('bulb_radius_mm', rec.bulb_radius_mm))
        rec.remit_decay = float(src.get('remit_decay', rec.remit_decay))
        rec.translucence_gain = float(src.get('translucence_gain', rec.translucence_gain))
        return

    rec.emit_uv_layer = float(getattr(mat, 'emit_uv_layer', rec.emit_uv_layer))
    rec.color_uv_layer = float(getattr(mat, 'color_uv_layer', rec.color_uv_layer))
    rec.depth_uv_layer = float(getattr(mat, 'depth_uv_layer', getattr(mat, 'depth_layer', rec.depth_uv_layer)))
    rec.remit_uv_layer = float(getattr(mat, 'remit_uv_layer', getattr(mat, 'remit_mask_layer', rec.remit_uv_layer)))
    rec.depth_scale_mm = float(getattr(mat, 'depth_scale_mm', rec.depth_scale_mm))
    rec.thickness_scale_mm = float(getattr(mat, 'thickness_scale_mm', rec.thickness_scale_mm))
    rec.depth_bias_mm = float(getattr(mat, 'depth_bias_mm', rec.depth_bias_mm))
    rec.thickness_bias_mm = float(getattr(mat, 'thickness_bias_mm', rec.thickness_bias_mm))
    rec.emit_gain = float(getattr(mat, 'emit_gain', rec.emit_gain))
    rec.color_blend = float(getattr(mat, 'color_blend', rec.color_blend))
    rec.direct_lobe_power = float(getattr(mat, 'direct_lobe_power', rec.direct_lobe_power))
    rec.model_flags = float(getattr(mat, 'model_flags', getattr(mat, 'texture_flags', rec.model_flags)))
    rec.remit_gain = float(getattr(mat, 'remit_gain', rec.remit_gain))
    rec.bulb_radius_mm = float(getattr(mat, 'bulb_radius_mm', rec.bulb_radius_mm))
    rec.remit_decay = float(getattr(mat, 'remit_decay', rec.remit_decay))
    rec.translucence_gain = float(getattr(mat, 'translucence_gain', rec.translucence_gain))


# ─────────────────────────────────────────────────────────────────────────────
# Emission / remission / color profile record types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParametricSpread:
    """Resonant peak: amplitude × Lorentzian(center, Q)."""
    amp:    float = 1.0
    center: float = 0.0
    q:      float = 1.0


@dataclass
class HistogramSpread:
    """Discrete bin distribution."""
    edges:  np.ndarray   # shape (N+1,) bin edges or (N,) bin centers
    values: np.ndarray   # shape (N,) amplitudes


@dataclass
class TextureSpread:
    """Pre-sampled spread stored as a texture handle or asset index."""
    texture_id: int = 0


SpreadData = Union[ParametricSpread, HistogramSpread, TextureSpread]


@dataclass
class SpreadProfile:
    """One spread descriptor per physical dimension."""
    angular:   SpreadData = field(default_factory=lambda: ParametricSpread())
    phase:     SpreadData = field(default_factory=lambda: ParametricSpread())
    frequency: SpreadData = field(default_factory=lambda: ParametricSpread())
    amplitude: SpreadData = field(default_factory=lambda: ParametricSpread())


@dataclass
class EmissionProfile:
    """Physical description of what an emissive surface produces.

    All fields are statistical descriptors of the emitted photons — spectrum,
    phase, polarization, coherence, power.  Angular distribution is a property
    of the optical system downstream, not of the emitter.

    SpreadData fields (``spd``, ``phase``, ``amplitude``) accept any of
    ParametricSpread, HistogramSpread, or TextureSpread.
    """
    # ── Spectral ──────────────────────────────────────────────────────────────
    spd:                     SpreadData = field(default_factory=lambda: ParametricSpread(amp=1.0, center=550.0, q=20.0))
    peak_wavelength_nm:      float      = 550.0   # dominant emission wavelength
    fwhm_nm:                 float      = 30.0    # spectral full-width half-maximum

    # ── Radiometric ───────────────────────────────────────────────────────────
    total_power_W:           float      = 1.0     # total radiated power
    radiance_W_sr_m2:        float      = 0.0     # W/(sr·m²), 0 = derive from power

    # ── Temporal / spatial coherence ─────────────────────────────────────────
    temporal_coherence_length_m:  float = 1.0e-6  # λ²/Δλ  (coherence length)
    spatial_coherence_radius_m:   float = 1.0e-3  # van Cittert-Zernike radius

    # ── Polarization (Stokes vector, normalised so S0=1) ─────────────────────
    stokes_s0:               float      = 1.0     # total intensity (always 1.0)
    stokes_s1:               float      = 0.0     # horizontal (+) vs vertical (-) linear
    stokes_s2:               float      = 0.0     # +45° vs −45° linear
    stokes_s3:               float      = 0.0     # right-circular vs left-circular
    degree_of_polarization:  float      = 0.0     # 0=unpolarised, 1=fully polarised

    # ── Phase statistics ──────────────────────────────────────────────────────
    phase:                   SpreadData = field(default_factory=lambda: ParametricSpread(amp=1.0, center=0.0, q=0.0))
    # q≈0 ⟹ broad/uniform; large q ⟹ narrow/coherent

    # ── Amplitude distribution across emitting surface ────────────────────────
    amplitude:               SpreadData = field(default_factory=lambda: ParametricSpread(amp=1.0, center=0.0, q=1.0))


@dataclass
class RemissionResponseEnvelope:
    """Time-domain impulse response for **frame-delayed** re-emission.

    The remission system is intentionally not a self-consistent solve.  When a
    surface is irradiated at frame ``F``, the resulting re-emission appears in
    frames ``F+1, F+2, ..., F+N`` shaped by this envelope — never in frame
    ``F`` itself.  This avoids any feedback loop: the renderer reads the
    previous frame's stimulus, multiplies by the impulse response coefficients,
    and writes the result into next frame's emission slot.

    Shape source
    ────────────
    The envelope is authored as either:

    * a ``parametric_curve.ParametricCurve`` (or any ``Callable[[float],float]``)
      sampled over normalised t ∈ [0, 1] — yields ``decay_frames`` impulse
      coefficients via :py:meth:`bake`, OR

    * ``curve=None`` ⇒ default exponential decay  ``exp(-3·k/N)``  over
      ``decay_frames`` samples (a clean, physically plausible thermal-style
      relaxation that finishes ≈5% by the last frame).

    All coefficients are baked **once** at registration time and uploaded
    alongside the spectral RGB tensor.  The hot loop only does an FIR
    convolution against a small ring buffer of past stimuli — never a
    callable evaluation.

    Coefficient layout
    ──────────────────
    ``_ir`` (shape ``(decay_frames,)`` float32) is the discrete impulse
    response.  ``_ir[k]`` multiplies the stimulus from ``k+1`` frames ago;
    ``_ir[0]`` is the strongest contribution (most recent stimulus).
    """
    curve:        object  = None      # parametric_curve.ParametricCurve | Callable | None
    decay_frames: int     = 8         # length of the impulse response
    gain:         float   = 1.0       # global scalar applied after sampling
    _ir:          "np.ndarray" = field(default=None, repr=False, compare=False)

    def bake(self) -> "np.ndarray":
        """Sample the curve at ``decay_frames`` evenly-spaced normalised
        t-values and store the result as ``self._ir``.  Returns the impulse
        response (shape ``(decay_frames,)`` float32)."""
        n = max(1, int(self.decay_frames))
        if self.curve is None:
            # Default thermal-style exponential decay
            ks = np.arange(n, dtype=np.float32)
            ir = np.exp(-3.0 * ks / float(n)).astype(np.float32)
        else:
            ts = np.linspace(0.0, 1.0, n, dtype=np.float64)
            try:
                # parametric_curve.ParametricCurve callable interface:
                # accepts a tensor / scalar in [0,1], returns tensor.
                import torch as _t
                t_t = _t.from_numpy(ts)
                vals = self.curve(t_t)
                if hasattr(vals, "detach"):
                    vals = vals.detach().cpu().numpy()
                ir = np.asarray(vals, dtype=np.float32).reshape(-1)
            except Exception:
                # Plain Python callable fallback
                ir = np.asarray([float(self.curve(t)) for t in ts],
                                dtype=np.float32)
            if ir.size != n:
                ir = np.resize(ir, n).astype(np.float32)
        ir = ir.astype(np.float32) * float(self.gain)
        self._ir = ir
        return ir

    @property
    def ir(self) -> "np.ndarray":
        """Cached impulse response.  Bakes lazily on first access."""
        if self._ir is None:
            return self.bake()
        return self._ir


@dataclass
class RemissionProfile:
    """Profile for re-emission; carries batch transform lambdas for structural color.

    ``transforms`` is a list of callables, each with signature::

        f(batch: np.ndarray) -> np.ndarray

    where ``batch`` is a 2-D array (N_hits × N_params).  Transforms are **never**
    applied serially per hit; the caller must coalesce all pending indices first,
    then call each transform across the full batch.

    ``manifold`` is an optional :class:`camera_software.LensManifold`.  When set,
    it is registered as the first transform: the batch columns 0-1 are aperture
    (u,v) and columns 4-6 are the incident ray direction; the manifold produces
    the transformed out_dir (cols 7-9) for the whole batch in one vectorised call.
    Setting this field automatically prepends the manifold's batch callable to
    ``transforms`` via :py:meth:`set_manifold`.

    ``response_envelope`` (optional) makes this a *frame-delayed re-emitter*:
    the engine reads previous-frame stimulus per material, convolves with the
    envelope's impulse response, multiplies by ``spread.frequency``-derived
    sRGB, and writes the result into next frame's ``pbr[mat_id, 8:11]``.  See
    :class:`RemissionFeedbackEngine`.
    """
    spread:               SpreadProfile  = field(default_factory=SpreadProfile)
    transforms:           List[Callable] = field(default_factory=list)
    manifold:             object         = field(default=None)   # LensManifold | None
    response_envelope:    "RemissionResponseEnvelope | None" = None

    def set_manifold(self, manifold) -> None:
        def _manifold_transform(batch: "np.ndarray") -> "np.ndarray":
            import numpy as _np
            uv     = batch[:, :2]          # cols 0-1 : aperture (u, v)
            in_dir = batch[:, 4:7]         # cols 4-6 : incident ray direction
            out_dir = manifold.interpolate_with_opl(uv, in_dir=in_dir)[:, :3]  # (N, 3)
            result = batch.copy()
            result[:, 7:10] = out_dir      # cols 7-9 : replace with transformed out_dir
            return result
        if self.transforms and getattr(self.transforms[0], '_is_manifold', False):
            self.transforms[0] = _manifold_transform
        else:
            self.transforms.insert(0, _manifold_transform)
        _manifold_transform._is_manifold = True


@dataclass
class ColorProfile:
    """Absorption-filtering profile; encodes structural color via spread-based filtering."""
    spread: SpreadProfile = field(default_factory=SpreadProfile)


# ─────────────────────────────────────────────────────────────────────────────
# MaterialDatabase
# ─────────────────────────────────────────────────────────────────────────────

class EmissionProfileDatabase:
    """Registry of named emission/reemission profiles.

    Each profile is an opaque object — the caller supplies whatever parametric
    representation is appropriate (transform function, sampled curve, etc.).
    The database assigns it a stable integer index.  That index is what gets
    packed into the mat16 slot and uploaded to the GPU SSBO so the shader can
    fetch the profile by index at hit time.

    Usage::

        ep_db = EmissionProfileDatabase.instance()
        idx = ep_db.register("glass_fluorescence", my_profile_object)
        # idx is an int; store float(idx) in mat16[12] / mat16[13]
    """

    _instance: Optional["EmissionProfileDatabase"] = None

    def __init__(self) -> None:
        self._profiles: Dict[str, Any] = {}
        self._order:    List[str]      = []
        self._rgb_tensor: np.ndarray   = np.zeros((1, 3), np.float32)

    def _rebake_rgb_tensor(self) -> None:
        """Precompute linear-sRGB for all profiles once (outside render loop).

        This is the only place where profile_to_rgb is evaluated.
        """
        N = max(1, len(self._order))
        out = np.zeros((N, 3), np.float32)
        for i, name in enumerate(self._order):
            out[i] = profile_to_rgb(self._profiles[name])
        self._rgb_tensor = out

    @classmethod
    def instance(cls) -> "EmissionProfileDatabase":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtins()
        return cls._instance

    def _register_builtins(self) -> None:
        """Register standard emission profiles that are available at startup.

        These represent the physical light sources used by built-in materials.
        Indices are assigned in registration order and must stay stable across
        runs — append only, never reorder.  Each entry gets a permanent integer
        index; never reorder, only append.
        """
        # Index 0: basic_led_display
        # Cool monitor backlight, modeled as two additive spectral terms:
        #   1) narrow blue pump around 450 nm,
        #   2) broader cyan phosphor shoulder around 510 nm.
        # This yields a dim blue-tinted output (B > G > R) suitable for
        # non-directional ambient-like screen glow in raster paths.
        self.register("basic_led_display", EmissionProfile(
            spd=[
                ParametricSpread(amp=1.0, center=450.0, q=18.0),
                ParametricSpread(amp=0.30, center=510.0, q=5.0),
            ],
            peak_wavelength_nm=450.0,
            fwhm_nm=25.0,
            total_power_W=0.35,
        ))

        # Index 1: missing_material_hazard
        # Warm amber warning glow used by fallback materials when a requested
        # YAML/material registration fails.  This should be noticeable but not
        # scene-dominating, so keep total_power moderate.
        self.register("missing_material_hazard", EmissionProfile(
            spd=[
                ParametricSpread(amp=1.0, center=590.0, q=12.0),
                ParametricSpread(amp=0.25, center=620.0, q=10.0),
            ],
            peak_wavelength_nm=590.0,
            fwhm_nm=49.0,
            total_power_W=0.22,
        ))
        self._rebake_rgb_tensor()

    def register(self, name: str, profile: Any) -> int:
        """Register a profile object.  Returns its integer index (0-based)."""
        if name not in self._profiles:
            self._order.append(name)
        self._profiles[name] = profile
        # Eager prebake: profile->RGB conversion happens here, never in draw loop.
        self._rebake_rgb_tensor()
        return self._order.index(name)

    def index_of(self, name: str) -> int:
        return self._order.index(name)

    def get(self, index: int) -> Any:
        if index < 0 or index >= len(self._order):
            return None
        return self._profiles[self._order[index]]

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, name: str) -> bool:
        return name in self._profiles

    # ── Tensor export ─────────────────────────────────────────────────────────

    def build_rgb_tensor(self) -> np.ndarray:
        """Return prebaked (max(1,N), 3) float32 linear sRGB profile table.

        No profile conversion is performed here; values are baked at
        registration time via _rebake_rgb_tensor().
        """
        return self._rgb_tensor

    def build_gpu_tensor(self) -> np.ndarray:
        """Return flat float32 array for SSBO upload at binding 7.

        Layout: [N * 20 header floats] followed by [inline noodle data for
        any RemissionProfile whose .manifold is set].

        Per-profile header (stride 20 floats / 5 vec4s):
            [0]  profile_type
                   0 = EmissionProfile
                   1 = RemissionProfile  (spread-based, no manifold)
                   2 = ColorProfile
                   3 = PROFILE_MANIFOLD  (RemissionProfile with .manifold set)
            [1]  noodle_start  — float offset (in floats) from buffer start
                                 where this profile's noodle rows begin.
                                 0 when profile_type != 3.
            [2]  noodle_count  — number of noodle rows (N_rays).
                                 0 when profile_type != 3.
            [3]  _pad
            [4]  angular.spread_type   [5] angular.amp   [6] angular.center  [7] angular.q
            [8]  phase.spread_type     [9] phase.amp    [10] phase.center   [11] phase.q
            [12] frequency.spread_type [13] freq.amp    [14] freq.center    [15] freq.q
            [16] amplitude.spread_type [17] amp.amp     [18] amp.center     [19] amp.q

        Noodle rows (appended after all N headers, stride 12 floats):
            [0,1]     u, v          normalised aperture position [-1,1]
            [2,3]     fu, fv        field-angle factors (tan of deviation)
            [4,5,6]   in_dx/dy/dz   unit input ray direction
            [7,8,9]   out_dx/dy/dz  unit output ray direction
            [10]      opl           optical path length (metres)
            [11]      _pad

        spread_type: 0=ParametricSpread  1=HistogramSpread  2=TextureSpread
        Opaque profiles (not one of the known types) get zeros.
        """
        import numpy as _np
        N = max(1, len(self._order))
        headers = _np.zeros((N, 20), _np.float32)
        noodle_blocks: List[_np.ndarray] = []   # list of (K, 12) float32 arrays
        noodle_float_offset = N * 20            # first noodle row starts here

        for i, name in enumerate(self._order):
            prof = self._profiles[name]
            if isinstance(prof, EmissionProfile):
                headers[i, 0] = 0.0
                _pack_spread_profile(prof.spread, headers[i, 4:])
            elif isinstance(prof, RemissionProfile):
                if prof.manifold is not None:
                    raw = _np.asarray(prof.manifold._data, _np.float32)  # (K, 11)
                    K = len(raw)
                    padded = _np.zeros((K, 12), _np.float32)
                    padded[:, :11] = raw
                    noodle_offset = noodle_float_offset + sum(b.size for b in noodle_blocks)
                    headers[i, 0] = 3.0
                    headers[i, 1] = float(noodle_offset)
                    headers[i, 2] = float(K)
                    headers[i, 3] = 0.0
                    # [4..19] unused for PROFILE_MANIFOLD — noodle rows carry all data
                    noodle_blocks.append(padded.ravel())
                else:
                    headers[i, 0] = 1.0
                    _pack_spread_profile(prof.spread, headers[i, 4:])
            elif isinstance(prof, ColorProfile):
                headers[i, 0] = 2.0
                _pack_spread_profile(prof.spread, headers[i, 4:])

        flat_headers = headers.ravel()
        if noodle_blocks:
            return _np.concatenate([flat_headers] + noodle_blocks).astype(_np.float32)
        return flat_headers

    def build_gl_summary(self) -> np.ndarray:
        """Return (max(1,N), 4) float32: [frequency_center, frequency_amp, _pad, _pad].

        GL path does not use spectral profiles — this provides a minimal
        per-profile float4 stub for any GL shader that needs to index the
        table without consuming full spectral data.
        """
        N = max(1, len(self._order))
        out = np.zeros((N, 4), np.float32)
        for i, name in enumerate(self._order):
            prof = self._profiles[name]
            spread = None
            if isinstance(prof, (EmissionProfile, RemissionProfile, ColorProfile)):
                spread = prof.spread.frequency
            if isinstance(spread, ParametricSpread):
                out[i, 0] = spread.center
                out[i, 1] = spread.amp
            elif isinstance(spread, HistogramSpread):
                vals = np.asarray(spread.values, np.float32)
                out[i, 1] = float(vals.max()) if vals.size else 0.0
        return out


def _pack_spread_to_row(s: "SpreadData", dst: np.ndarray) -> None:
    """Pack one SpreadData into 4 consecutive floats: [type, amp, center, q]."""
    if isinstance(s, ParametricSpread):
        dst[0] = 0.0; dst[1] = s.amp; dst[2] = s.center; dst[3] = s.q
    elif isinstance(s, HistogramSpread):
        vals = np.asarray(s.values, np.float32)
        edges = np.asarray(s.edges, np.float32)
        dst[0] = 1.0
        dst[1] = float(vals.max()) if vals.size else 0.0
        dst[2] = float(edges[len(edges) // 2]) if edges.size else 0.0
        dst[3] = 0.0
    elif isinstance(s, TextureSpread):
        dst[0] = 2.0; dst[1] = float(s.texture_id); dst[2] = 0.0; dst[3] = 0.0


def _pack_spread_profile(spread: "SpreadProfile", dst: np.ndarray) -> None:
    """Pack all 4 SpreadData dimensions into dst[0:16]."""
    _pack_spread_to_row(spread.angular,   dst[0:4])
    _pack_spread_to_row(spread.phase,     dst[4:8])
    _pack_spread_to_row(spread.frequency, dst[8:12])
    _pack_spread_to_row(spread.amplitude, dst[12:16])


# ─────────────────────────────────────────────────────────────────────────────
# Emission-profile → linear sRGB conversion (first principles)
# ─────────────────────────────────────────────────────────────────────────────

# Wavelength grid 380–780 nm, 5 nm steps (81 samples).
_WL_NM: np.ndarray = np.arange(380.0, 785.0, 5.0, dtype=np.float64)

def _cie_cmf() -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """CIE 1931 2° colour-matching functions via Wyman et al. (2013) Gaussians.

    Returns (x̄, ȳ, z̄) each of length 81, sampled on _WL_NM (380..780 nm).
    Negative lobes in x̄ are retained so the chromaticity is physically correct;
    they are clipped only when computing SPD integrals (where a negative CMF
    contribution from the x̄ sub-lobe can produce out-of-gamut values for
    monochromatic primaries near 500 nm, which is physical — sRGB is not convex).
    """
    l = _WL_NM
    def _g(mu: float, sigma: float, amp: float = 1.0) -> np.ndarray:
        return amp * np.exp(-0.5 * ((l - mu) / sigma) ** 2)

    xbar = _g(599.8, 37.9, 1.056) + _g(442.0, 16.0, 0.362) + _g(501.1, 20.4, -0.065)
    ybar = _g(568.8, 46.9, 0.821) + _g(530.9, 16.3, 0.286)
    zbar = _g(437.0, 20.0, 1.217) + _g(459.0, 11.18, 0.681)
    return xbar, ybar, zbar


# Pre-compute once at import time.
_CMF_X, _CMF_Y, _CMF_Z = _cie_cmf()

# Normalisation constant: ∫ ȳ(λ) dλ over 380–780 nm (5 nm step).
# A flat equal-energy illuminant with power = 1 W will produce Y = 1.0.
_CMF_Y_NORM: float = float(np.sum(_CMF_Y) * 5.0)


def _spd_term_to_array(term: Any) -> np.ndarray:
    """Convert one spectral term into an SPD sampled on _WL_NM.

    Supported term forms:
      - ParametricSpread
      - HistogramSpread
      - dict with either:
          {"kind": "parametric", "amp", "center", "q"}
          {"kind": "histogram", "edges", "values"}
        Optional field: "weight" (default 1.0)
    """
    weight = 1.0
    spread = term

    if isinstance(term, dict):
        weight = float(term.get("weight", 1.0))
        kind = str(term.get("kind", "parametric")).lower()
        if kind == "histogram":
            spread = HistogramSpread(
                edges=np.asarray(term.get("edges", []), np.float64),
                values=np.asarray(term.get("values", []), np.float64),
            )
        else:
            spread = ParametricSpread(
                amp=float(term.get("amp", 1.0)),
                center=float(term.get("center", 550.0)),
                q=float(term.get("q", 20.0)),
            )

    if isinstance(spread, ParametricSpread):
        if spread.q > 0.0:
            sigma_nm = max(spread.center, 1.0) / (spread.q * 2.3548)
        else:
            sigma_nm = 500.0
        spd = spread.amp * np.exp(-0.5 * ((_WL_NM - spread.center) / sigma_nm) ** 2)
    elif isinstance(spread, HistogramSpread):
        edges = np.asarray(spread.edges, np.float64)
        vals = np.asarray(spread.values, np.float64)
        if edges.size == vals.size:
            centres = edges
        elif edges.size == vals.size + 1:
            centres = 0.5 * (edges[:-1] + edges[1:])
        else:
            return np.zeros(len(_WL_NM), np.float64)
        spd = np.interp(_WL_NM, centres, vals, left=0.0, right=0.0)
    elif isinstance(spread, TextureSpread):
        spd = np.ones(len(_WL_NM), np.float64)
    else:
        spd = np.ones(len(_WL_NM), np.float64)

    return np.maximum(0.0, spd * weight)


def _spd_from_spread(spread: Any) -> np.ndarray:
    """Build SPD on _WL_NM from a spread or a list of spread contributions.

    This accepts:
      - single ParametricSpread / HistogramSpread / TextureSpread
      - list/tuple of those terms (summed additively)
      - list/tuple of dict terms accepted by _spd_term_to_array
    """
    if isinstance(spread, (list, tuple)):
        acc = np.zeros(len(_WL_NM), np.float64)
        for term in spread:
            acc += _spd_term_to_array(term)
        return np.maximum(0.0, acc)
    return _spd_term_to_array(spread)


def profile_to_rgb(profile: Any) -> np.ndarray:
    """Convert an EmissionProfile (or any registered profile object) to a
    linear sRGB float32 3-vector via CIE 1931 XYZ numerical integration.

        Steps:
            1. Build SPD(λ) from profile.spd (single spread, histogram, or arbitrary
                 additive term list).
      2. Integrate  X = ∫ SPD(λ) x̄(λ) dλ,  Y = ∫ SPD(λ) ȳ(λ) dλ,  etc.
      3. Normalise so that a unit-power equal-energy illuminant gives Y = 1.
      4. Scale by profile.total_power_W.
      5. Apply XYZ → linear sRGB matrix (D65 white point, IEC 61966-2-1).

    The result is HDR-legal (values > 1 for bright emitters) and uses the
    native float32 dtype.  It is NOT gamma-corrected.

    Reflective profile types (RemissionProfile, ColorProfile) are also
    accepted: their ``spread.frequency`` field — which may be a single
    SpreadData or a list of additive contributions — is interpreted as
    the spectral reflectance/transmittance, and integrated against the
    same CMFs with unit total_power.  No HDR scaling is applied; the
    resulting RGB is bounded by what the SPD itself encodes.
    """
    if isinstance(profile, EmissionProfile):
        spd   = _spd_from_spread(profile.spd)
        power = float(profile.total_power_W)
    elif isinstance(profile, (RemissionProfile, ColorProfile)):
        # Reflective: the frequency-axis spread (parametric, histogram, or
        # additive list of either) IS the spectral reflectance.  Integrate
        # under unit illuminant — the user authors the amplitudes to match
        # the desired baked sRGB triple.
        spd   = _spd_from_spread(profile.spread.frequency)
        power = 1.0
    else:
        # Opaque object → flat white, 1 W
        spd   = np.ones(len(_WL_NM), np.float64)
        power = 1.0

    dl = 5.0  # nm per step
    X = float(np.sum(spd * _CMF_X) * dl)
    Y = float(np.sum(spd * _CMF_Y) * dl)
    Z = float(np.sum(spd * _CMF_Z) * dl)

    # Normalise so equal-energy illuminant at power=1 gives (X,Y,Z) scaled by
    # power.  _CMF_Y_NORM = ∫ ȳ dλ so normalised Y = 1 for flat unit-power SPD.
    if _CMF_Y_NORM > 0.0:
        scale = power / _CMF_Y_NORM
    else:
        scale = 0.0
    X *= scale;  Y *= scale;  Z *= scale

    # XYZ → linear sRGB (D65 primaries, IEC 61966-2-1 / sRGB spec matrix).
    R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    # Clamp only negative values — positive HDR is valid for emissives.
    return np.array([max(0.0, R), max(0.0, G), max(0.0, B)], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RemissionFeedbackEngine — frame-delayed re-emission
# ─────────────────────────────────────────────────────────────────────────────

class RemissionFeedbackEngine:
    """Frame-delayed remission driver.

    The engine implements re-emission as a **pure FIR convolution** against
    a per-material stimulus history.  At each frame ``F``:

    1. The renderer (or test harness) computes a scalar "stimulus" per
       remissive material — typically that material's recent screen
       luminance, integrated incident scene power, or any other physically
       motivated proxy for "how much energy did I just absorb".
    2. The engine pushes that stimulus into a ring buffer of length
       ``decay_frames`` for that material (the buffer length is dictated
       by the material's :class:`RemissionResponseEnvelope`).
    3. The engine convolves history × envelope ⇒ a scalar amplitude for
       frame ``F+1``.
    4. That amplitude is multiplied by the material's prebaked
       ``frequency``-derived sRGB triple (already in
       ``EmissionProfileDatabase._rgb_tensor``) ⇒ next-frame emission RGB.
    5. :py:meth:`apply_to_pbr` overwrites ``pbr[mat_id, 8:11]`` for every
       remissive material before the next render call.

    There is **no recursion** and **no solve** — frame ``F+1``'s emission is a
    deterministic linear function of frames ``F-K+1 .. F``'s stimuli, where
    ``K = decay_frames``.  Frame ``F``'s emission can never depend on its own
    irradiance; the one-frame delay is the explicit physical price for
    avoiding the feedback loop.

    This Python implementation is a reference: it is correct and vectorised
    over materials but is not the long-term hot path.  The eventual GLSL/C
    LUT will compute the same FIR per fragment, batched.
    """

    def __init__(self, mat_db: "MaterialDatabase", ep_db: "EmissionProfileDatabase"):
        self._mat_db: "MaterialDatabase" = mat_db
        self._ep_db:  "EmissionProfileDatabase" = ep_db
        # mat_id → impulse response (K_m,) float32
        self._ir:        Dict[int, np.ndarray] = {}
        # mat_id → ring buffer (K_m,) float32 ; idx 0 = most-recent stimulus
        self._hist:      Dict[int, np.ndarray] = {}
        # mat_id → baked sRGB triple (3,) float32  (from ColorProfile derived
        # from the material's RemissionProfile.spread.frequency)
        self._color_rgb: Dict[int, np.ndarray] = {}
        # mat_id → resolved remit_profile_idx into ep_db
        self._mat_to_remit_idx: Dict[int, int] = {}

    # ── configuration ────────────────────────────────────────────────────────

    def configure(self) -> None:
        """Scan the material DB; pick up every material whose dict carries a
        ``remit_profile_idx >= 0`` AND whose referenced ``RemissionProfile``
        has a non-None ``response_envelope``.

        Bakes each impulse response and zeros the stimulus history.  Safe to
        call again after re-registration (state for materials that disappear
        is dropped; new materials get fresh zero buffers).
        """
        t = self._mat_db.build_tensors()
        index = dict(t.get("index", {}))
        ep_rgb = self._ep_db.build_rgb_tensor()                  # (P, 3) float32

        new_ir:        Dict[int, np.ndarray] = {}
        new_hist:      Dict[int, np.ndarray] = {}
        new_color_rgb: Dict[int, np.ndarray] = {}
        new_mat_remit: Dict[int, int]        = {}

        for _name, mat_id in index.items():
            mat = self._mat_db._materials.get(_name)
            if mat is None:
                continue
            # Resolve remit_profile_idx from the underlying material record.
            raw = -1.0
            if isinstance(mat, dict):
                raw = mat.get("remit_profile_idx", -1.0)
            elif hasattr(mat, "remit_profile_idx"):
                raw = getattr(mat, "remit_profile_idx", -1.0)
            try:
                remit_idx = int(float(raw))
            except (TypeError, ValueError):
                remit_idx = -1
            if remit_idx < 0 or remit_idx >= len(self._ep_db):
                continue
            prof = self._ep_db.get(remit_idx)
            if not isinstance(prof, RemissionProfile):
                continue
            env = prof.response_envelope
            if env is None:
                continue
            ir = np.asarray(env.ir, dtype=np.float32).reshape(-1)
            if ir.size == 0:
                continue
            new_ir[mat_id]        = ir
            # Preserve any existing history if buffer length matches; else fresh.
            old_h = self._hist.get(mat_id)
            if old_h is not None and old_h.size == ir.size:
                new_hist[mat_id] = old_h
            else:
                new_hist[mat_id] = np.zeros(ir.size, dtype=np.float32)
            new_color_rgb[mat_id] = np.asarray(ep_rgb[remit_idx], dtype=np.float32)
            new_mat_remit[mat_id] = remit_idx

        self._ir              = new_ir
        self._hist            = new_hist
        self._color_rgb       = new_color_rgb
        self._mat_to_remit_idx= new_mat_remit

    # ── runtime ──────────────────────────────────────────────────────────────

    def remissive_mat_ids(self) -> List[int]:
        """Materials currently driven by this engine, in arbitrary order."""
        return list(self._ir.keys())

    def push_stimulus(self, stimulus_per_mat: Dict[int, float]) -> None:
        """Advance every history buffer one frame and inject new stimuli.

        Materials not present in ``stimulus_per_mat`` get a 0.0 push.
        Materials present but unknown to the engine are silently ignored
        (they have no envelope configured).
        """
        for mat_id, hist in self._hist.items():
            # Right-shift: drop oldest, prepend new (idx 0 = most recent).
            hist[1:] = hist[:-1]
            hist[0]  = float(stimulus_per_mat.get(mat_id, 0.0))

    def next_frame_emission(self) -> Dict[int, np.ndarray]:
        """Return ``{mat_id: rgb (3,) float32}`` to be written into the next
        frame's pbr emission slot.  Computed as ``Σ_k ir[k] * hist[k]``
        scaled by the material's baked sRGB."""
        out: Dict[int, np.ndarray] = {}
        for mat_id, ir in self._ir.items():
            hist = self._hist[mat_id]
            amp  = float(np.dot(ir, hist))
            if amp <= 0.0:
                # Negative or zero amplitude ⇒ nothing to emit; we still
                # write zero so a previously hot material cools down on time.
                out[mat_id] = np.zeros(3, dtype=np.float32)
            else:
                out[mat_id] = (self._color_rgb[mat_id] * amp).astype(np.float32)
        return out

    def apply_to_pbr(self, pbr_chunk: np.ndarray) -> None:
        """In-place: write next-frame emission into ``pbr_chunk[mat_id, 8:11]``
        for every remissive material currently tracked.

        Caller is responsible for re-uploading the modified chunk to the GPU
        SSBO before the next render."""
        for mat_id, rgb in self.next_frame_emission().items():
            if 0 <= mat_id < len(pbr_chunk):
                pbr_chunk[mat_id, 8:11] = rgb


class MaterialDatabase:
    """Central registry of materials.

    Registration  (authoring entry points)
    ---------------------------------------
        db = MaterialDatabase.instance()
        db.register("steel_panel", material_object_or_dict)  # → int index
        db.register_from_mat16("wood_top", mat16_array)       # mat16 ingest

    Both paths produce the SAME post-bake tensors.  The mat16 path exists
    for the YAML loader, which composes a 16-float record from the YAML
    fields and hands it in for ingest — it is NOT a "smaller / alternative"
    registration form, despite the historical name.

    Tensor export  (the only thing the hot loop ever reads)
    --------------------------------------------------------
        tensors = db.build_tensors()
        # tensors['pbr']            : (N, 16)             float32  authoritative
        # tensors['phong_compat']   : (N,  8)             float32  ← downstream-only
        # tensors['texture_stack']  : (N, 16)             float32  optional/cold
        # tensors['raymat_compat']  : (N, 16)             float32  ← downstream-only
        # tensors['spectral']       : (N, MAX_BANDS, 12)  float32
        # tensors['enamel']         : (N,  8)             float32
        # tensors['n_bands']        : (N,)                int32
        # tensors['index']          : dict[name → int]

    The `*_compat` chunks are tiny speed-improving prebakes for shader
    paths that cannot consume the full PBR record (basic Phong rasteriser,
    legacy mat11 raytracer feed).  They are derivatives, not authoring
    surfaces.  They are computed once at build_tensors() and frozen until
    the next material registration.

    GPU upload (example)
    --------------------
        t = db.build_tensors()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, pbr_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     t['pbr'].nbytes, t['pbr'], GL_STATIC_DRAW)

    Compatibility extraction (legacy callers only — not the hot path)
    -----------------------------------------------------------------
        db.as_compat_mat11(name)     → np.ndarray shape (11,) float32
        db.as_compat_phong(name)     → dict with Phong uniform keys
        db.compat_mat11_tensor()     → np.ndarray shape (N, 11) float32
    """

    _instance: Optional["MaterialDatabase"] = None

    def __init__(self) -> None:
        self._materials: Dict[str, Any] = {}   # name → Material or dict
        self._order:     List[str]      = []   # stable insertion order
        self._dirty:     bool           = True
        self._tensors:   Optional[dict] = None

    @classmethod
    def instance(cls) -> "MaterialDatabase":
        """Process-global singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, material: Any) -> int:
        """Register a Material object or dict.  Returns its integer index.

        Registering the same name again updates the material and marks the
        cache dirty.
        """
        if name not in self._materials:
            self._order.append(name)
        self._materials[name] = material
        self._dirty = True
        return self._order.index(name)

    def register_from_mat16(self, name: str, mat16: np.ndarray) -> int:
        """Register from a 16-float authored material vector.

        Despite the historical association with "mat11", this is the
        canonical YAML→DB ingest path — the YAML loader composes a full
        16-float record (refl/diff/abso, albedo, ior, opacity, mat_flags,
        emit_profile_idx, remit_profile_idx, color_profile_idx,
        reactive_shift_hz) and hands it in here.  Shorter input arrays
        are accepted for legacy callers; missing slots default to safe
        sentinels.

        Slot layout when len(mat16) >= 16:
            [11] = mat_flags          (uint bits as float)
            [12] = emit_profile_idx   (-1 = none)
            [13] = remit_profile_idx  (-1 = none)
            [14] = color_profile_idx  (-1 = none)
            [15] = reactive_shift_hz  (Hz)
        """
        m = np.asarray(mat16, np.float32).ravel()
        d: Dict[str, Any] = {
            '_source':   'mat11',
            'refl_in':   m[0:3].tolist()  if len(m) >= 3  else [0.5, 0.0, 0.5],
            'refl_out':  m[3:6].tolist()  if len(m) >= 6  else [0.5, 0.0, 0.5],
            'albedo_rgb':m[6:9].tolist()  if len(m) >= 9  else [0.5, 0.5, 0.5],
            'ior':       float(m[9])      if len(m) >= 10 else 1.5,
            'opacity':   float(m[10])     if len(m) >= 11 else 1.0,
        }
        if len(m) >= 12:
            d['mat_flags'] = float(m[11])
        if len(m) >= 13:
            d['emit_profile_idx'] = float(m[12])
        if len(m) >= 14:
            d['remit_profile_idx'] = float(m[13])
        if len(m) >= 15:
            d['color_profile_idx'] = float(m[14])
        if len(m) >= 16:
            d['reactive_shift_hz'] = float(m[15])
        return self.register(name, d)

    def index_of(self, name: str) -> int:
        return self._order.index(name)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    # ── Tensor build ──────────────────────────────────────────────────────────

    def build_tensors(self) -> dict:
        """Build all chunked tensors.  Cached until a new material is registered.

        Returns a dict of numpy arrays (see class docstring for shapes).
        All arrays are float32 except n_bands (int32) and index (dict).
        """
        if not self._dirty and self._tensors is not None:
            return self._tensors

        N = len(self._order)
        if N == 0:
            self._tensors = {
                'pbr':            np.zeros((0, PBR_FLOATS),    np.float32),
                'phong_compat':   np.zeros((0, PHONG_FLOATS),  np.float32),
                'raymat_compat':  np.zeros((0, RAYMAT_FLOATS), np.float32),
                'spectral':       np.zeros((0, MAX_SPECTRAL_BANDS, 12), np.float32),
                'enamel':         np.zeros((0, ENAMEL_FLOATS), np.float32),
                'texture_stack':  np.zeros((0, TEXSTACK_FLOATS), np.float32),
                'n_bands':        np.zeros(0, np.int32),
                'index':          {},
            }
            self._dirty = False
            return self._tensors

        # Allocate ctypes arrays (one block per chunk, all N rows contiguous)
        pbr_arr  = (PBRBaseRecord    * N)()
        phon_arr = (PhongRecord      * N)()
        ray_arr  = (RayMatRecord     * N)()
        spec_arr = (SpectralRecord   * N)()
        enam_arr = (EnamelRecord     * N)()
        tex_arr  = (TextureStackRecord * N)()

        for i, name in enumerate(self._order):
            mat = self._materials[name]
            is_mat11_dict = isinstance(mat, dict) and mat.get('_source') == 'mat11'

            # ── PBR ──
            if is_mat11_dict:
                _fill_pbr_from_mat11_dict(pbr_arr[i], mat)
            else:
                _fill_pbr(pbr_arr[i], mat)

            # ── Phong (derived + optional dict overrides) ──
            phon_arr[i] = _pbr_to_phong_record(pbr_arr[i], mat if isinstance(mat, dict) else None)

            # ── RayMat ──
            if is_mat11_dict:
                _fill_ray_from_mat11_dict(ray_arr[i], mat)
            else:
                ray_arr[i] = _pbr_to_ray_record(pbr_arr[i])
                # Propagate reactive_shift_hz from Material dataclass if present
                if hasattr(mat, 'reactive_shift_hz') and mat.reactive_shift_hz != 0.0:
                    ray_arr[i].reactive_shift_hz = float(mat.reactive_shift_hz)
                    _fv = _struct.unpack('f', _struct.pack('I',
                          _struct.unpack('I', _struct.pack('f', ray_arr[i].mat_flags))[0]
                          | 2))[0]  # set MAT_FLAG_REACTIVE = 2u
                    ray_arr[i].mat_flags = _fv

            # ── Spectral bands ──
            _fill_spectral(spec_arr[i], mat)

            # ── Enamel ──
            _fill_enamel(enam_arr[i], mat)

            # ── Optional UV texture-stack metadata ──
            _fill_texture_stack(tex_arr[i], mat)

        # ── Extract to numpy via frombuffer (zero-copy for contiguous ctypes) ──
        pbr_np  = np.frombuffer(pbr_arr,  dtype=np.float32).reshape(N, PBR_FLOATS).copy()
        phon_np = np.frombuffer(phon_arr, dtype=np.float32).reshape(N, PHONG_FLOATS).copy()
        ray_np  = np.frombuffer(ray_arr,  dtype=np.float32).reshape(N, RAYMAT_FLOATS).copy()
        enam_np = np.frombuffer(enam_arr, dtype=np.float32).reshape(N, ENAMEL_FLOATS).copy()
        tex_np  = np.frombuffer(tex_arr,  dtype=np.float32).reshape(N, TEXSTACK_FLOATS).copy()

        # ── Resolve emit_profile_idx → PBR emission RGB (vectorized) ──────────
        # Build an integer index vector — one entry per material row — then
        # apply ep_rgb via fancy indexing.  No string operations at runtime.
        ep_db = EmissionProfileDatabase.instance()
        if len(ep_db) > 0:
            ep_rgb = ep_db.build_rgb_tensor()   # (M, 3) float32
            M = len(ep_rgb)

            # Extract emit_profile_idx as int32 array, shape (N,), default -1.
            ep_idxs = np.full(N, -1, dtype=np.int32)
            for i, mat in enumerate(self._materials[n] for n in self._order):
                raw = -1.0
                if isinstance(mat, dict):
                    raw = mat.get('emit_profile_idx', -1.0)
                elif hasattr(mat, 'emit_profile_idx'):
                    raw = mat.emit_profile_idx
                try:
                    ep_idxs[i] = int(float(raw))
                except (TypeError, ValueError):
                    ep_idxs[i] = -1

            # Boolean mask of rows that have a valid profile index.
            valid = (ep_idxs >= 0) & (ep_idxs < M)
            if valid.any():
                pbr_np[valid, 8:11] = ep_rgb[ep_idxs[valid]]

            # ── Resolve color_profile_idx → PBR albedo RGB (vectorised) ──
            # Same lookup table as emission (EPDB stores all profile types);
            # the integer simply selects which prebaked sRGB triple replaces
            # the authored albedo.  This is the "spectral as the only truth"
            # path: any material that names a color profile has its albedo
            # column overwritten with the spectrally-baked value, regardless
            # of what (if anything) was in the YAML's albedo_rgb field.
            cp_idxs = np.full(N, -1, dtype=np.int32)
            for i, mat in enumerate(self._materials[n] for n in self._order):
                raw = -1.0
                if isinstance(mat, dict):
                    raw = mat.get('color_profile_idx', -1.0)
                elif hasattr(mat, 'color_profile_idx'):
                    raw = mat.color_profile_idx
                try:
                    cp_idxs[i] = int(float(raw))
                except (TypeError, ValueError):
                    cp_idxs[i] = -1
            cvalid = (cp_idxs >= 0) & (cp_idxs < M)
            if cvalid.any():
                pbr_np[cvalid, 0:3] = ep_rgb[cp_idxs[cvalid]]

        # SpectralRecord contains mixed int32/float32 — extract in two steps:
        #   first 8×12 floats = band data; n_bands read directly from struct
        spec_raw     = np.frombuffer(spec_arr, dtype=np.float32)
        spec_np      = np.zeros((N, MAX_SPECTRAL_BANDS, 12), np.float32)
        n_bands_np   = np.zeros(N, np.int32)
        for i in range(N):
            base = i * _SPECTRAL_RECORD_FLOATS
            spec_np[i] = spec_raw[base : base + _SPECTRAL_BAND_FLOATS].reshape(MAX_SPECTRAL_BANDS, 12)
            n_bands_np[i] = spec_arr[i].n_bands

        self._tensors = {
            'pbr':            pbr_np,
            'phong_compat':   phon_np,   # downstream-only Phong sink
            'raymat_compat':  ray_np,    # downstream-only raytracer sink
            'spectral':       spec_np,
            'enamel':         enam_np,
            'texture_stack':  tex_np,    # optional cold UV/depth/remit metadata
            'n_bands':        n_bands_np,
            'index':          {name: i for i, name in enumerate(self._order)},
        }
        self._dirty = False
        return self._tensors

    # ── Compatibility extraction helpers (post-bake; not the hot path) ──────

    def as_compat_mat11(self, name: str) -> np.ndarray:
        """Return legacy 11-float32 array for one named material.

        Reads the prebaked raymat_compat chunk — NOT a re-derivation.
        Column mapping:  refl_in[3] | refl_out[3] | albedo[3] | ior | opacity
        """
        t = self.build_tensors()
        idx = t['index'][name]
        r   = t['raymat_compat'][idx]
        # Indices into the 16-float RayMatRecord row (pads are at 3, 7, 11, 14, 15)
        return np.array([
            r[0], r[1], r[2],    # refl_in
            r[4], r[5], r[6],    # refl_out
            r[8], r[9], r[10],   # albedo
            r[12],               # ior
            r[13],               # opacity
        ], np.float32)

    def as_compat_phong(self, name: str) -> dict:
        """Return Phong uniform dict for one named material.

        Reads the prebaked phong_compat chunk — NOT a re-derivation.
        """
        t   = self.build_tensors()
        idx = t['index'][name]
        pbr = t['pbr'][idx]
        ph  = t['phong_compat'][idx]
        return {
            'albedo_rgb':    pbr[0:3].tolist(),
            'ambient':       float(ph[0]),
            'spec_strength': float(ph[1]),
            'shininess':     float(ph[2]),
            'grain':         float(ph[3]),
        }

    def compat_mat11_tensor(self) -> np.ndarray:
        """Return (N, 11) float32 legacy tensor for ALL registered materials.

        Compatible with the existing _normalise_materials() input format.
        Column order:  refl_in[3] | refl_out[3] | albedo[3] | ior | opacity
        """
        t = self.build_tensors()
        r = t['raymat_compat']    # (N, 16)
        return np.ascontiguousarray(np.column_stack([
            r[:, 0:3],    # refl_in
            r[:, 4:7],    # refl_out
            r[:, 8:11],   # albedo
            r[:, 12:13],  # ior
            r[:, 13:14],  # opacity
        ]), dtype=np.float32)

    def names(self) -> List[str]:
        """Ordered list of all registered material names."""
        return list(self._order)


# ── Module-level singletons ───────────────────────────────────────────────────
_EMISSION_DB: EmissionProfileDatabase = EmissionProfileDatabase.instance()
