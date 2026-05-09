# Camera Simulator: Front-Line Development Goals

This is the immediate execution summary for completing the articulated camera simulator path with no fake intermediates.

## Current Status (Affirmed)

- The Python exposure loop is end-to-end runnable, but several camera-sensor paths were previously placeholder-driven.
- C++ field-capture and camera-visibility hooks are real and active.
- Surface spline fitting exists in C++ and is now exported in the rebuilt extension.
- Sensor/film tensor upload is now a real C++ API (`set_sensor_film_ssbo`) and no longer Python-skip-only scaffolding.
- Sensor integral generation has been cut over from a uniform synthetic photon map to endpoint-driven accumulation from BDPT PIXEL_CONE records.

## Completion Goals

1. Remove all fake image-adjacent paths from exposure calibration and sensor statistics.
2. Make C++ the source of truth for camera-sensor state and parametric-surface behavior.
3. Keep Python as an interoperability layer and reference implementation for helper logic.
4. Keep GLSL helper wiring aligned to C++ contracts (same tensor semantics, same config enums, same region/surface dispatch intent).

## Immediate Engineering Priorities

1. **Sensor/Film C++ consumption**
   - Use uploaded sensor/film chunks inside bidirectional camera/sensor accumulation.
   - Replace any remaining host-side synthetic assumptions with per-slot tensor reads.

2. **Multi-slot sensor integration**
   - Expand from current first-active-slot integration to full `MAX_SENSOR_FILM_SLOTS` accumulation.
   - Preserve per-slot outputs and metrics without flattening early.

3. **Parametric spline execution parity**
   - Ensure spline payloads affect both emissive sampling and sensor-facing hit behavior where intended.
   - Add regression checks for `fit_all_tris` and emissive-only modes.

4. **Live progressive exposure view**
   - Move viewer from per-exposure blocking updates to per-batch image refresh.
   - Keep final calibrated outputs unchanged while exposing intermediate accumulation states.

5. **Feature-specific rigorous scenes**
   - Add demanding calibration scenes that fail visibly when a feature is miswired:
     - spectral wedge / narrow-band emitters,
     - aperture-focus chart,
     - field regular-vs-kdtree stress scene,
     - transparency visibility stack,
     - spline-sensitive emissive geometry,
     - particle/scattering interaction scene.

## C++-First, Python/GLSL Interoperability Model

- **C++ kernel/binding layer**
  - Owns physical data contracts and execution semantics.
  - Exposes explicit APIs for sensor/film upload, field capture, camera visibility, spline fitting, and bidirectional records.

- **Python layer**
  - Owns orchestration, test harnessing, diagnostics, and compatibility helpers.
  - Must avoid inventing synthetic physical data once equivalent C++ data exists.

- **GLSL layer**
  - Mirrors proven C++ contracts for real-time paths.
  - Uses Python helpers as roadmap/reference for packing, scheduling, and validation.

## Definition of Done (Phase Gate)

- No hardcoded sensor photon maps in the exposure path.
- No silent API skips for sensor/film upload.
- Endpoint-driven sensor statistics for every active slot.
- Parametric spline path validated in runtime extension and exercised by scene tests.
- Progressive viewer shows real batch accumulation.
- Scene suite demonstrates clear pass/fail behavior for each major camera feature.