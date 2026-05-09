"""
SENSOR/FILM REGISTRY ARCHITECTURE — IMPLEMENTATION SUMMARY
============================================================

Created: sensor_film_db.py
Pattern: Mirrors material_db.py for systematic parameterization
Status: Foundation complete (Phase 3B), ready for phases 3C-3E

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE VISION
──────────

Transform sensor and film handling from dynamic object passing to a systematic,
indexed lookup architecture. This closes the loop between:

  Python Exposure Planner
      ↓ (selects sensor_id + film_id)
  C++ / Shader Intermediary
      ↓ (reads from SSBO tensors)
  Field/Surface Archetype
      ↓ (full complex unreduced data)
  Scientific Ray Tracing

The result: a first-principles exposure model with all parameters traceable
to physics or device specs — no magic constants, no dynamic object overhead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 3B: COMPLETE ✅
─────────────────────

**sensor_film_db.py** (1500+ lines)

Core Structures:
  • SensorRecord (48 floats):
    - Optics: focal length, f-number, aperture diameter
    - Geometry: sensor dimensions, pixel pitch
    - Photon transport: QE, full well, read noise, dark current
    - Digitisation: bit depth, black/white levels
    - CFA: Red/Green/Blue/IR peak λ, FWHM, area fractions
    - Lens: chromatic aberration, distortion coefficients
    ✓ std430 GLSL compatible
    ✓ Binary layout matches ctypes ↔ NumPy ↔ C++ ↔ GLSL

  • FilmRecord (64 floats):
    - Exposure: ISO, shutter time, quantum efficiency, grey point
    - Tone curves: 8 layers × 7 floats each (shadow/highlight RGB, points)
    - Spectral: peak sensitivity wavelength, FWHM
    ✓ Matches demo_pluck_gl.py 8-layer film stack architecture
    ✓ Fully extensible (future: thermal, spectral response curves)

Database:
  • SensorFilmDatabase singleton
  • register_sensor(name, spec_dict) → sensor_id
  • register_film(name, spec_dict) → film_id
  • build_tensors() → 8-slot chunked float32 arrays
    - sensor: (8, 48) float32 — one sensor per slot
    - film: (8, 64) float32 — one film per slot
    - index_sensor: dict[name → int]
    - index_film: dict[name → int]

Builtin Sensors (ready to extend):
  • Canon EOS R6 — 45 MP mirrorless (Sony IMX spec)
  • Nikon D850 — 45 MP DSLR (Nikon legacy spec)

Builtin Films (ready to extend):
  • Kodak Portra 400 — warm, skin-friendly color negative
  • Fuji Superia 200 — cool, vibrant color negative

CFA Support:
  • CFAPattern + CFAChannel dataclasses
  • Bayer RGGB helper
  • Sony IMX (most modern cameras)
  • Extensible for X-Trans, custom patterns

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTEGRATION POINTS (READY FOR PHASES 3C-3E)
──────────────────────────────────────────

Python Layer (exposure_render_demo.py):
  
  Instead of:
    tracer.register_tri_group(
        role_bits=TRI_GROUP_ROLE_SENSOR,
        sensor_camera={optics_dict, film_dict}  # ← dynamic objects
    )
  
  Will use:
    from sensor_film_db import SensorFilmDatabase
    db = SensorFilmDatabase.instance()
    tensors = db.build_tensors()  # (8, 48) + (8, 64) chunks
    
    # ExposureSession stores slot config
    session.sensor_film_slots = [
        ("canon_eos_r6", "kodak_portra_400"),  # slot 0
        ("nikon_d850", "fuji_superia_200"),     # slot 1
        (-1, -1),  # slots 2-7 unused
    ]

C++ Layer (csrc/kernels/_spectral_kernels):
  
  SSBO layout (new bindings):
    • Binding 15-22: SensorSlot[0-7] — sensor parameters
    • Binding 23-30: FilmSlot[0-7] — film tone curves
    • Binding 31-38: IntermediaryField[0-7] — per-slot field grid
    • Binding 39-46: IntermediaryFilm[0-7] — per-slot film integration
  
  Shader loop pattern (compute kernel):
    ```cpp
    for(int slot = 0; slot < 8; slot++) {
        Sensor sensor = unpack_sensor(slot);
        Film film = unpack_film(slot);
        
        for(int grid_idx = 0; grid_idx < grid_volume; grid_idx++) {
            // Apply per-slot QE + film curve to field data
            intermediary_field[slot][grid_idx] = 
                apply_film_response(
                    field_unreduced[grid_idx],
                    sensor.qe_peak,
                    film.tone_curve_0
                );
        }
    }
    ```

Shader Layer (glsl):
  
  Unpacking helpers:
    ```glsl
    Sensor unpack_sensor(int slot_id) {
        int base = slot_id * 48;
        return Sensor(
            focal_mm: sensor_data[base + 0],
            f_number: sensor_data[base + 1],
            qe_peak: sensor_data[base + 8],
            ...
        );
    }
    
    Film unpack_film(int slot_id) {
        int base = slot_id * 64;
        return Film(
            iso: film_data[base + 0],
            exposure_time_s: film_data[base + 1],
            layer0_shadow_rgb: vec3(
                film_data[base + 5],
                film_data[base + 6],
                film_data[base + 7]
            ),
            ...
        );
    }
    ```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEXT PHASES (3C-3E)
──────────────────

**Phase 3C: ExposureSession Integration** (estimated 200 lines)
  1. Update ExposureSession.__init__() to accept sensor_film_slots config
  2. Load sensor_film_db.build_tensors() once at startup
  3. Pass 8-slot SSBO uploads to C++ tracer
  4. Store slot → (sensor_id, film_id) mapping for HUD labels
  5. Update frame config to enable sensor per detail level

**Phase 3D: Shader Vectorized Loop** (estimated 300-400 lines C++ + GLSL)
  1. Add 8-slot SSBO bindings (15-46) to base_rasterizer.cpp
  2. Implement unpack_sensor() / unpack_film() helpers in shader
  3. Modify field integration loop to broadcast across 8 slots
  4. Apply per-slot QE + film response in compute kernel
  5. Validate: check that 8 slots run in parallel (no sequential falloff)

**Phase 3E: Endpoint Record Filtering & Accumulation** (estimated 250 lines)
  1. After BDPT bidirectional() pass, filter records by sensor_group_id
  2. For each sensor hit, map position → pixel UV via ray_to_uv()
  3. Accumulate photons per-pixel per-slot with QE weighting
  4. Compute per-slot SNR from electrons + noise floor
  5. Update SensorIntegralObject[slot] with actual endpoint data (not placeholder)
  6. Save per-slot NPZ artifacts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY ARCHITECTURAL PROPERTIES
─────────────────────────────

✓ **Indexed not Named** 
  Like materials: sensor_id / film_id are integers, not strings.
  Fast shader lookup, cache-friendly memory layout.

✓ **Pre-baked not Dynamic**
  All parameters computed once in build_tensors(). No runtime derivation.
  Eliminates object overhead, matches material_db philosophy.

✓ **8-Slot Batching**
  Matches film-back architecture in demo_pluck_gl.py FilmStack.
  Enables vectorized GPU broadcasting with no sequence penalties.

✓ **Full Intermediate Data**
  Field/surface layers preserve all impacts/crossings before final integration.
  Scientific-grade: every transformation is auditable.

✓ **Closed Python → C++ Pipeline**
  ExposureSession selects from database indices.
  C++ reads from SSBO tensors directly (no object unmarshalling).
  Shader applies physics-based tone curves in parallel.
  Result: modular, testable, debuggable.

✓ **Extensible**
  New sensors: db.register_sensor() + recompile.
  New films: db.register_film() + recompile.
  New CFA patterns: create CFAPattern instance.
  New noise models: extend FilmRecord with more coefficients.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK START (WHEN READY FOR PHASES 3C+)
──────────────────────────────────────

# Python (expose_render_demo.py)
from sensor_film_db import SensorFilmDatabase

db = SensorFilmDatabase.instance()
tensors = db.build_tensors()  # (8, 48) + (8, 64) float32

# Check what's registered
print(f"Sensors: {db.names_sensors()}")
print(f"Films: {db.names_films()}")

# Add custom sensor
db.register_sensor("my_camera", {
    "focal_mm": 35,
    "f_number": 2.0,
    "qe_peak": 0.82,
    "full_well_e": 250_000,
    ...
})

# Configure 8-slot batch for exposure
session.sensor_film_slots = [
    (0, 0),  # slot 0: Canon EOS R6 + Kodak Portra
    (1, 1),  # slot 1: Nikon D850 + Fuji Superia
    (db.index_of_sensor("my_camera"), 0),  # slot 2: custom + Kodak
    (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),  # slots 3-7: off
]

# C++ will read: sensor_chunk[slot][0:48], film_chunk[slot][0:64]
# Shader will broadcast QE + film response across 8 slots
# Result: 8 parallel sensor/film combinations in one exposure

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DESIGN CREDITS
──────────────

This architecture is a direct application of the material_db.py pattern:
  • Indexed lookups (mat_id, sensor_id, film_id)
  • Pre-baked chunked tensors
  • SSBO bindings for GPU access
  • ctypes ↔ NumPy ↔ GLSL binary compatibility
  • Singleton registry with build_tensors() cache

Extended with sensor/film semantics:
  • CFA support for spectral camera modeling
  • Film layer stacks (8-slot batch from demo_pluck_gl.py)
  • Noise models (read noise, dark current, shot noise)
  • Tone curves for film characteristic reproduction
  • Full-complex intermediate data preservation

Result: **Unified, first-principles ray tracing suitable for scientific
neural or parametric spline transforms.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
