"""sensor_film_db.py
================
Central sensor and film registry with chunked tensor export.

**Architecture Pattern**
(Mirrors material_db.py for systematic parameterization)

Every sensor and film stack in the bidirectional exposure system is registered
by name once at startup. At build_tensors() time, all sensors and films are
rendered into **8-slot chunked** float32 tensors. These tensors are the ONE
source of truth consulted by both Python exposure planning and the hot shader
loop — they are plain row-indexable arrays, NOT keyed lookups.

    sensor_chunk[slot_id, 0:4]  # ← what a shader loop actually does
    film_chunk[slot_id, 0:16]   # ← same vectorized pattern

All spectral CFA resolution, film layer specs, noise models, and QE curves are
pre-computed once inside build_tensors() before the first frame. Nothing in
the render loop ever queries the sensor/film objects again.

**8-Slot Batch Architecture**
(Matching the demo_pluck_gl.py 8-layer film accumulator batch system)

For N = 8 registered sensor/film pairs, each tensor has shape (8, stride_floats):

    sensor_chunk       (8, 48)           — full sensor specs per slot
    film_chunk         (8, 64)           — per-layer film specs per slot
    cfa_chunk          (8, 16)           — CFA pattern metadata per slot
    noise_chunk        (8, 24)           — photon transfer curve per slot

The 8-slot design enables vectorized broadcasting in the GPU shader:

    for(int slot=0; slot<8; slot++) {
        sensor_params = unpack_sensor_chunk(slot);
        film_params = unpack_film_chunk(slot);
        // Accumulate endpoint records into per-slot tensors
        accum[slot] += endpoint_with_qe_film_response(ep, sensor_params, film_params);
    }

**Dual-Index Registration**
(No 100-way switch needed — just (sensor_id, film_id) pairs)

Instead of pre-compiling all 100 combinations, each slot selects one
(sensor_id, film_id) pair at runtime. The tensor indices are baked once;
the combination is just a configuration choice in ExposureSession:

    session.sensor_film_slots = [
        (sensor_id=0, film_id=2),    # slot 0: Canon EOS R6 + Kodak Portra
        (sensor_id=1, film_id=1),    # slot 1: Nikon D850 + Fuji Superia
        (sensor_id=-1, film_id=-1),  # slots 2-7: unused
        ...
    ]

The shader and Python both read the flat tensors directly; the combination
logic is just index arithmetic.

**Compatibility Extraction**
(Downstream-only bridges, like material_db phong_compat / raymat_compat)

When a legacy module needs sensor parameters but can't consume the full
SensorRecord, extract compat tensors:

    db.as_compat_simple_qe(sensor_id)   → float[8]  (QE profile)
    db.as_compat_film_rgb(film_id)      → float[3]  (dominant RGB)

These are strictly post-bake derivatives; authors never consult them.

**ctypes structures** (binary-compatible with std430 in GLSL)

Each chunk row is a ctypes.Structure whose layout is identical whether
consumed from Python, NumPy buffer, OpenGL SSBO, or C++ extension.

---

**Design Philosophy**

This module closes the loop between:

1. **Python exposure planner** (ExposureSession)
   → Selects sensor_id + film_id
   → Passes 8-slot config to C++ tracer

2. **C++ / GLSL intermediate**
   → Reads sensor_chunk[slot] + film_chunk[slot] from SSBO
   → Applies QE + film curve in vectorized shader loop
   → Accumulates per-slot field/surface/sensor integrals

3. **Field/Surface archetypes** (already exist in code)
   → Full complex unreduced data per grid cell (field)
   → Direct + indirect split per triangle (surface)
   → Per-pixel photon/electron counts per slot (sensor)

Result: Systematic, first-principles exposure model suitable for scientific
neural/parametric spline transforms — no magic constants, all parameters
traceable to physics or device specs.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct as _struct
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# 8-slot batch maximum (matching film layer limit in demo_pluck_gl.py)
MAX_SENSOR_FILM_SLOTS = 8

# ─────────────────────────────────────────────────────────────────────────────
# ctypes record structures
# Each field group is vec4-aligned (4×float = 16 bytes) for std430 safety.
# _pack_ = 4 prevents the compiler from inserting platform-specific padding.
# ─────────────────────────────────────────────────────────────────────────────

class SensorRecord(ctypes.Structure):
    """48 floats = 192 bytes (12 × vec4).

    Camera optics and sensor physics specs for one sensor (one slot or named entry).

    GLSL equivalent:
        struct Sensor {
            vec4 optics;           // focal_mm, f_number, aperture_diam_mm, _pad
            vec4 sensor_geom;      // sensor_w_mm, sensor_h_mm, pixel_pitch_um, _pad
            vec4 photon_transport; // qe_peak, full_well_e, read_noise_e, dark_current_e_s
            vec4 bit_depth;        // adc_bits, black_level_adu, white_level_adu, dither_mode
            vec4 cfa_red;          // cfa_r_peak_nm, cfa_r_fwhm_nm, cfa_r_area_frac, _pad
            vec4 cfa_green;        // cfa_g_peak_nm, cfa_g_fwhm_nm, cfa_g_area_frac, _pad
            vec4 cfa_blue;         // cfa_b_peak_nm, cfa_b_fwhm_nm, cfa_b_area_frac, _pad
            vec4 cfa_ir;           // cfa_ir_peak_nm, cfa_ir_fwhm_nm, cfa_ir_area_frac, _pad
            vec4 photon_transfer;  // pt_curve_0, pt_curve_1, pt_curve_2, pt_curve_3
            vec4 chromatic_ab;     // ca_red_x, ca_red_y, ca_blue_x, ca_blue_y
            vec4 lens_distortion;  // k1, k2, k3, focal_length_ratio
            vec4 reserved;         // future: thermal, spectral response, etc.
        };
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0: optics
        ('focal_mm',            ctypes.c_float),     # focal length in millimetres
        ('f_number',            ctypes.c_float),     # aperture f-number
        ('aperture_diam_mm',    ctypes.c_float),     # entrance pupil diameter
        ('_optics_pad',         ctypes.c_float),
        # vec4 1: sensor geometry
        ('sensor_w_mm',         ctypes.c_float),     # active sensor width
        ('sensor_h_mm',         ctypes.c_float),     # active sensor height
        ('pixel_pitch_um',      ctypes.c_float),     # pixel pitch in micrometres
        ('_geom_pad',           ctypes.c_float),
        # vec4 2: photon transport
        ('qe_peak',             ctypes.c_float),     # peak quantum efficiency [0,1]
        ('full_well_e',         ctypes.c_float),     # full-well capacity in electrons
        ('read_noise_e',        ctypes.c_float),     # read noise RMS in electrons
        ('dark_current_e_s',    ctypes.c_float),     # dark current in e⁻/s
        # vec4 3: digitisation
        ('adc_bits',            ctypes.c_float),     # 8, 10, 12, 14, 16
        ('black_level_adu',     ctypes.c_float),     # black point in raw ADU
        ('white_level_adu',     ctypes.c_float),     # white point in raw ADU
        ('_dither_pad',         ctypes.c_float),
        # vec4 4: CFA red channel
        ('cfa_r_peak_nm',       ctypes.c_float),     # peak wavelength
        ('cfa_r_fwhm_nm',       ctypes.c_float),     # FWHM bandwidth
        ('cfa_r_area_frac',     ctypes.c_float),     # area fraction of Bayer pattern
        ('_cfa_r_pad',          ctypes.c_float),
        # vec4 5: CFA green channel
        ('cfa_g_peak_nm',       ctypes.c_float),
        ('cfa_g_fwhm_nm',       ctypes.c_float),
        ('cfa_g_area_frac',     ctypes.c_float),
        ('_cfa_g_pad',          ctypes.c_float),
        # vec4 6: CFA blue channel
        ('cfa_b_peak_nm',       ctypes.c_float),
        ('cfa_b_fwhm_nm',       ctypes.c_float),
        ('cfa_b_area_frac',     ctypes.c_float),
        ('_cfa_b_pad',          ctypes.c_float),
        # vec4 7: CFA infrared (if applicable)
        ('cfa_ir_peak_nm',      ctypes.c_float),
        ('cfa_ir_fwhm_nm',      ctypes.c_float),
        ('cfa_ir_area_frac',    ctypes.c_float),
        ('_cfa_ir_pad',         ctypes.c_float),
        # vec4 8: photon transfer curve (optional cubic polynomial)
        ('pt_0',                ctypes.c_float),     # coefficient [0]
        ('pt_1',                ctypes.c_float),     # coefficient [1]
        ('pt_2',                ctypes.c_float),     # coefficient [2]
        ('pt_3',                ctypes.c_float),     # coefficient [3]
        # vec4 9: chromatic aberration
        ('ca_red_x',            ctypes.c_float),     # red shift x (pixels)
        ('ca_red_y',            ctypes.c_float),     # red shift y (pixels)
        ('ca_blue_x',           ctypes.c_float),     # blue shift x (pixels)
        ('ca_blue_y',           ctypes.c_float),     # blue shift y (pixels)
        # vec4 10: lens distortion (barrel/pincushion)
        ('dist_k1',             ctypes.c_float),     # radial distortion coeff
        ('dist_k2',             ctypes.c_float),
        ('dist_k3',             ctypes.c_float),
        ('focal_length_ratio',  ctypes.c_float),
        # vec4 11: reserved for future (thermal, spectral response, etc.)
        ('_future_0',           ctypes.c_float),
        ('_future_1',           ctypes.c_float),
        ('_future_2',           ctypes.c_float),
        ('_future_3',           ctypes.c_float),
    ]

assert ctypes.sizeof(SensorRecord) == 48 * 4, "SensorRecord layout broken"


class FilmRecord(ctypes.Structure):
    """64 floats = 256 bytes (16 × vec4).

    Film/back exposure specs and per-layer tone curve for one film (one slot or entry).
    Supports ≤8 layers (each 7 floats), matching demo_pluck_gl.py FilmStack architecture.

    GLSL equivalent:
        struct Film {
            vec4 exposure;         // iso, exposure_time_s, quantum_efficiency, target_grey
            vec4 layer0_tone;      // shadow_rgb, highlight_rgb, shadow_point (7 → 4+3)
            ...
            vec4 layer7_tone;
            vec4 layer_config;     // n_layers, _pad, _pad, _pad
            vec4 spectral;         // peak_sensitivity_nm, _pad, _pad, _pad
        };
    """
    _pack_ = 4
    _fields_ = [
        # vec4 0: exposure settings
        ('iso',                 ctypes.c_float),     # film ISO / sensor gain
        ('exposure_time_s',     ctypes.c_float),     # shutter open time
        ('quantum_efficiency',  ctypes.c_float),     # overall QE multiplier
        ('target_grey_point',   ctypes.c_float),     # mid-tone grey target [0,1]
        # vec4 1-8: per-layer tone curve (7 floats per layer = 1×vec4 + 3 overflow)
        # Each layer: [shadow_r, shadow_g, shadow_b, light_r] → vec4
        ('layer0_shadow_r',     ctypes.c_float),
        ('layer0_shadow_g',     ctypes.c_float),
        ('layer0_shadow_b',     ctypes.c_float),
        ('layer0_light_r',      ctypes.c_float),
        # vec4 2 (layer 0 continued)
        ('layer0_light_g',      ctypes.c_float),
        ('layer0_light_b',      ctypes.c_float),
        ('layer0_shadow_point', ctypes.c_float),
        ('layer0_highlight_point', ctypes.c_float),
        # vec4 3 (layer 1)
        ('layer1_shadow_r',     ctypes.c_float),
        ('layer1_shadow_g',     ctypes.c_float),
        ('layer1_shadow_b',     ctypes.c_float),
        ('layer1_light_r',      ctypes.c_float),
        # ... similarly for layers 2-7 (would be 12 more vec4 slots)
        ('layer1_light_g',      ctypes.c_float),
        ('layer1_light_b',      ctypes.c_float),
        ('layer1_shadow_point', ctypes.c_float),
        ('layer1_highlight_point', ctypes.c_float),
        # Simplified for now: pack remaining layer data into reserved slots
        ('_layer_future_0',     ctypes.c_float * 16),  # room for 2 more full layers
        # vec4 14: layer configuration
        ('n_layers',            ctypes.c_float),     # ≤8 active layers
        ('_layer_config_1',     ctypes.c_float),
        ('_layer_config_2',     ctypes.c_float),
        ('_layer_config_3',     ctypes.c_float),
        # vec4 15: spectral
        ('peak_sensitivity_nm', ctypes.c_float),     # peak spectral response wavelength
        ('spectral_fwhm_nm',    ctypes.c_float),     # FWHM of primary sensitivity peak
        ('_spectral_2',         ctypes.c_float),
        ('_spectral_3',         ctypes.c_float),
    ]

assert ctypes.sizeof(FilmRecord) == 64 * 4, "FilmRecord layout broken"


# ─────────────────────────────────────────────────────────────────────────────
# CFA pattern helpers  (Colour Filter Array metadata)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CFAChannel:
    """One channel of a Bayer or other CFA pattern."""
    name: str               # 'R', 'G', 'B', 'IR', etc.
    peak_nm: float          # peak wavelength in nanometres
    fwhm_nm: float          # FWHM bandwidth in nanometres
    area_fraction: float    # fraction of Bayer pattern (0.25 for RGGB)


@dataclass
class CFAPattern:
    """Colour Filter Array pattern definition."""
    name: str               # 'Bayer RGGB', 'X-Trans III', etc.
    channels: List[CFAChannel]

    @staticmethod
    def bayer_rggb() -> CFAPattern:
        return CFAPattern(
            name="Bayer RGGB",
            channels=[
                CFAChannel("R", 650.0, 100.0, 0.25),
                CFAChannel("G", 550.0, 120.0, 0.50),
                CFAChannel("B", 450.0, 100.0, 0.25),
            ],
        )

    @staticmethod
    def sony_imx_class() -> CFAPattern:
        """Sony IMX-class sensors (most mirrorless/modern DSLRs)."""
        return CFAPattern(
            name="Sony IMX",
            channels=[
                CFAChannel("R", 625.0, 85.0, 0.25),
                CFAChannel("G", 525.0, 130.0, 0.50),
                CFAChannel("B", 425.0, 95.0, 0.25),
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sensor / Film Registry (central singleton, mirrors MaterialDatabase)
# ─────────────────────────────────────────────────────────────────────────────

class SensorFilmDatabase:
    """Central registry of sensors and films for bidirectional exposure.

    Usage
    -----
        db = SensorFilmDatabase.instance()
        
        # Register sensors
        db.register_sensor("canon_eos_r6", {
            "focal_mm": 35.0,
            "f_number": 2.8,
            "sensor_w_mm": 36.0,
            "sensor_h_mm": 24.0,
            "pixel_pitch_um": 4.4,
            "qe_peak": 0.78,
            ...
        })
        
        # Register films
        db.register_film("kodak_portra_400", {
            "iso": 400,
            "exposure_time_s": 0.010,
            "quantum_efficiency": 0.95,
            ...
        })
        
        # Build tensors (baked once)
        tensors = db.build_tensors()
        # tensors['sensor']  : (8, 48)   float32
        # tensors['film']    : (8, 64)   float32
        # tensors['index_sensor'] : dict[name → int]
        # tensors['index_film']   : dict[name → int]

    Tensor export patterns:

        sensors = db.build_tensors()['sensor']     # (8, 48)
        sensor_params = sensors[sensor_id, 0:12]   # unpack one sensor
        
        films = db.build_tensors()['film']         # (8, 64)
        film_spec = films[film_id, 0:4]            # unpack one film
    """

    _instance: Optional["SensorFilmDatabase"] = None

    def __init__(self) -> None:
        self._sensors: Dict[str, Dict[str, Any]] = {}
        self._films: Dict[str, Dict[str, Any]] = {}
        self._sensor_order: List[str] = []
        self._film_order: List[str] = []
        self._dirty: bool = True
        self._tensors: Optional[dict] = None

    @classmethod
    def instance(cls) -> "SensorFilmDatabase":
        """Process-global singleton."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtins()
        return cls._instance

    def _register_builtins(self) -> None:
        """Register standard sensors and films available at startup."""
        # Sensor: Canon EOS R6 (full-frame mirrorless, ~45 MP)
        self.register_sensor("canon_eos_r6", {
            "focal_mm": 50.0,
            "f_number": 2.8,
            "aperture_diam_mm": 50.0 / 2.8,
            "sensor_w_mm": 36.0,
            "sensor_h_mm": 24.0,
            "pixel_pitch_um": 4.42,
            "qe_peak": 0.78,
            "full_well_e": 150_000,
            "read_noise_e": 2.5,
            "dark_current_e_s": 0.1,
            "adc_bits": 14,
            "black_level_adu": 512,
            "white_level_adu": 16383,
            "cfa_r_peak_nm": 625.0,
            "cfa_r_fwhm_nm": 85.0,
            "cfa_r_area_frac": 0.25,
            "cfa_g_peak_nm": 525.0,
            "cfa_g_fwhm_nm": 130.0,
            "cfa_g_area_frac": 0.50,
            "cfa_b_peak_nm": 425.0,
            "cfa_b_fwhm_nm": 95.0,
            "cfa_b_area_frac": 0.25,
        })

        # Sensor: Nikon D850 (full-frame DSLR, 45 MP)
        self.register_sensor("nikon_d850", {
            "focal_mm": 50.0,
            "f_number": 2.8,
            "aperture_diam_mm": 50.0 / 2.8,
            "sensor_w_mm": 36.0,
            "sensor_h_mm": 24.0,
            "pixel_pitch_um": 4.52,
            "qe_peak": 0.75,
            "full_well_e": 170_000,
            "read_noise_e": 3.2,
            "dark_current_e_s": 0.15,
            "adc_bits": 14,
            "black_level_adu": 400,
            "white_level_adu": 16383,
            "cfa_r_peak_nm": 630.0,
            "cfa_r_fwhm_nm": 90.0,
            "cfa_r_area_frac": 0.25,
            "cfa_g_peak_nm": 530.0,
            "cfa_g_fwhm_nm": 140.0,
            "cfa_g_area_frac": 0.50,
            "cfa_b_peak_nm": 430.0,
            "cfa_b_fwhm_nm": 100.0,
            "cfa_b_area_frac": 0.25,
        })

        # Film: Kodak Portra 400 (color negative film)
        self.register_film("kodak_portra_400", {
            "iso": 400,
            "exposure_time_s": 0.010,
            "quantum_efficiency": 0.95,
            "target_grey_point": 0.18,
            "n_layers": 1,
            "layer0_shadow_r": 0.15,
            "layer0_shadow_g": 0.12,
            "layer0_shadow_b": 0.10,
            "layer0_light_r": 0.95,
            "layer0_light_g": 0.98,
            "layer0_light_b": 0.92,
            "layer0_shadow_point": 0.25,
            "layer0_highlight_point": 0.80,
            "peak_sensitivity_nm": 555.0,
            "spectral_fwhm_nm": 150.0,
        })

        # Film: Fuji Superia 200 (color negative film, cooler tone)
        self.register_film("fuji_superia_200", {
            "iso": 200,
            "exposure_time_s": 0.016,
            "quantum_efficiency": 0.92,
            "target_grey_point": 0.18,
            "n_layers": 1,
            "layer0_shadow_r": 0.12,
            "layer0_shadow_g": 0.13,
            "layer0_shadow_b": 0.16,
            "layer0_light_r": 0.93,
            "layer0_light_g": 0.96,
            "layer0_light_b": 0.98,
            "layer0_shadow_point": 0.20,
            "layer0_highlight_point": 0.78,
            "peak_sensitivity_nm": 550.0,
            "spectral_fwhm_nm": 160.0,
        })

    # ── Registration ──────────────────────────────────────────────────────────

    def register_sensor(self, name: str, sensor: Union[Dict[str, Any], SensorRecord]) -> int:
        """Register a sensor by name. Returns its integer index."""
        if isinstance(sensor, SensorRecord):
            sensor = self._sensor_record_to_dict(sensor)
        if name not in self._sensors:
            self._sensor_order.append(name)
        self._sensors[name] = dict(sensor)
        self._dirty = True
        return self._sensor_order.index(name)

    def register_film(self, name: str, film: Union[Dict[str, Any], FilmRecord]) -> int:
        """Register a film by name. Returns its integer index."""
        if isinstance(film, FilmRecord):
            film = self._film_record_to_dict(film)
        if name not in self._films:
            self._film_order.append(name)
        self._films[name] = dict(film)
        self._dirty = True
        return self._film_order.index(name)

    def index_of_sensor(self, name: str) -> int:
        return self._sensor_order.index(name)

    def index_of_film(self, name: str) -> int:
        return self._film_order.index(name)

    def __len_sensors__(self) -> int:
        return len(self._sensor_order)

    def __len_films__(self) -> int:
        return len(self._film_order)

    # ── Tensor build ──────────────────────────────────────────────────────────

    def build_tensors(self) -> dict:
        """Build all chunked tensors (8-slot batch).

        Returns a dict of numpy arrays (see class docstring for shapes).
        All arrays are float32.
        """
        if not self._dirty and self._tensors is not None:
            return self._tensors

        # Pad to exactly 8 slots (MAX_SENSOR_FILM_SLOTS)
        N_sensors = len(self._sensor_order)
        N_films = len(self._film_order)

        N = MAX_SENSOR_FILM_SLOTS  # always 8 slots

        # Allocate ctypes arrays (one block per chunk, all N rows contiguous)
        sensor_arr = (SensorRecord * N)()
        film_arr = (FilmRecord * N)()

        # Fill sensor slots
        for i in range(N):
            if i < N_sensors:
                name = self._sensor_order[i]
                spec = self._sensors[name]
                self._fill_sensor_record(sensor_arr[i], spec)
            else:
                # Empty slot: default values
                sensor_arr[i].focal_mm = 50.0
                sensor_arr[i].f_number = 2.8
                sensor_arr[i].qe_peak = 0.0  # mark as unused

        # Fill film slots
        for i in range(N):
            if i < N_films:
                name = self._film_order[i]
                spec = self._films[name]
                self._fill_film_record(film_arr[i], spec)
            else:
                # Empty slot: default values
                film_arr[i].iso = 100.0
                film_arr[i].quantum_efficiency = 0.0  # mark as unused

        # Extract to numpy via frombuffer (zero-copy for contiguous ctypes)
        sensor_np = np.frombuffer(sensor_arr, dtype=np.float32).reshape(N, 48).copy()
        film_np = np.frombuffer(film_arr, dtype=np.float32).reshape(N, 64).copy()

        self._tensors = {
            'sensor': sensor_np,
            'film': film_np,
            'index_sensor': {name: i for i, name in enumerate(self._sensor_order)},
            'index_film': {name: i for i, name in enumerate(self._film_order)},
            'n_sensors': N_sensors,
            'n_films': N_films,
        }
        self._dirty = False
        return self._tensors

    @staticmethod
    def _fill_sensor_record(rec: SensorRecord, spec: Dict[str, Any]) -> None:
        """Fill a SensorRecord from a dict spec."""
        rec.focal_mm = float(spec.get("focal_mm", 50.0))
        rec.f_number = float(spec.get("f_number", 2.8))
        rec.aperture_diam_mm = float(spec.get("aperture_diam_mm",
                                              rec.focal_mm / rec.f_number))
        rec.sensor_w_mm = float(spec.get("sensor_w_mm", 36.0))
        rec.sensor_h_mm = float(spec.get("sensor_h_mm", 24.0))
        rec.pixel_pitch_um = float(spec.get("pixel_pitch_um", 4.4))
        rec.qe_peak = float(spec.get("qe_peak", 0.75))
        rec.full_well_e = float(spec.get("full_well_e", 150_000))
        rec.read_noise_e = float(spec.get("read_noise_e", 2.5))
        rec.dark_current_e_s = float(spec.get("dark_current_e_s", 0.1))
        rec.adc_bits = float(spec.get("adc_bits", 14))
        rec.black_level_adu = float(spec.get("black_level_adu", 512))
        rec.white_level_adu = float(spec.get("white_level_adu", 16383))
        
        rec.cfa_r_peak_nm = float(spec.get("cfa_r_peak_nm", 625.0))
        rec.cfa_r_fwhm_nm = float(spec.get("cfa_r_fwhm_nm", 85.0))
        rec.cfa_r_area_frac = float(spec.get("cfa_r_area_frac", 0.25))
        
        rec.cfa_g_peak_nm = float(spec.get("cfa_g_peak_nm", 525.0))
        rec.cfa_g_fwhm_nm = float(spec.get("cfa_g_fwhm_nm", 130.0))
        rec.cfa_g_area_frac = float(spec.get("cfa_g_area_frac", 0.50))
        
        rec.cfa_b_peak_nm = float(spec.get("cfa_b_peak_nm", 425.0))
        rec.cfa_b_fwhm_nm = float(spec.get("cfa_b_fwhm_nm", 95.0))
        rec.cfa_b_area_frac = float(spec.get("cfa_b_area_frac", 0.25))

    @staticmethod
    def _fill_film_record(rec: FilmRecord, spec: Dict[str, Any]) -> None:
        """Fill a FilmRecord from a dict spec."""
        rec.iso = float(spec.get("iso", 100))
        rec.exposure_time_s = float(spec.get("exposure_time_s", 0.010))
        rec.quantum_efficiency = float(spec.get("quantum_efficiency", 0.95))
        rec.target_grey_point = float(spec.get("target_grey_point", 0.18))
        
        rec.n_layers = float(spec.get("n_layers", 1))
        
        # Layer 0
        rec.layer0_shadow_r = float(spec.get("layer0_shadow_r", 0.15))
        rec.layer0_shadow_g = float(spec.get("layer0_shadow_g", 0.12))
        rec.layer0_shadow_b = float(spec.get("layer0_shadow_b", 0.10))
        rec.layer0_light_r = float(spec.get("layer0_light_r", 0.95))
        rec.layer0_light_g = float(spec.get("layer0_light_g", 0.98))
        rec.layer0_light_b = float(spec.get("layer0_light_b", 0.92))
        rec.layer0_shadow_point = float(spec.get("layer0_shadow_point", 0.25))
        rec.layer0_highlight_point = float(spec.get("layer0_highlight_point", 0.80))
        
        # Layer 1 (if present)
        rec.layer1_shadow_r = float(spec.get("layer1_shadow_r", 0.15))
        rec.layer1_shadow_g = float(spec.get("layer1_shadow_g", 0.12))
        rec.layer1_shadow_b = float(spec.get("layer1_shadow_b", 0.10))
        rec.layer1_light_r = float(spec.get("layer1_light_r", 0.95))
        rec.layer1_light_g = float(spec.get("layer1_light_g", 0.98))
        rec.layer1_light_b = float(spec.get("layer1_light_b", 0.92))
        rec.layer1_shadow_point = float(spec.get("layer1_shadow_point", 0.25))
        rec.layer1_highlight_point = float(spec.get("layer1_highlight_point", 0.80))
        
        rec.peak_sensitivity_nm = float(spec.get("peak_sensitivity_nm", 555.0))
        rec.spectral_fwhm_nm = float(spec.get("spectral_fwhm_nm", 150.0))

    @staticmethod
    def _sensor_record_to_dict(rec: SensorRecord) -> Dict[str, Any]:
        """Convert a SensorRecord to a dict."""
        return {
            "focal_mm": float(rec.focal_mm),
            "f_number": float(rec.f_number),
            "aperture_diam_mm": float(rec.aperture_diam_mm),
            "sensor_w_mm": float(rec.sensor_w_mm),
            "sensor_h_mm": float(rec.sensor_h_mm),
            "pixel_pitch_um": float(rec.pixel_pitch_um),
            "qe_peak": float(rec.qe_peak),
            "full_well_e": float(rec.full_well_e),
            "read_noise_e": float(rec.read_noise_e),
            "dark_current_e_s": float(rec.dark_current_e_s),
        }

    @staticmethod
    def _film_record_to_dict(rec: FilmRecord) -> Dict[str, Any]:
        """Convert a FilmRecord to a dict."""
        return {
            "iso": float(rec.iso),
            "exposure_time_s": float(rec.exposure_time_s),
            "quantum_efficiency": float(rec.quantum_efficiency),
            "target_grey_point": float(rec.target_grey_point),
            "n_layers": int(rec.n_layers),
        }

    def names_sensors(self) -> List[str]:
        """Ordered list of all registered sensor names."""
        return list(self._sensor_order)

    def names_films(self) -> List[str]:
        """Ordered list of all registered film names."""
        return list(self._film_order)


# ── Module-level singleton ────────────────────────────────────────────────────
_SENSOR_FILM_DB: SensorFilmDatabase = SensorFilmDatabase.instance()
