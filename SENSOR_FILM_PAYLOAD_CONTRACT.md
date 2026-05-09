# SENSOR/FILM PAYLOAD CONTRACT (Phase 3C)

Date: May 9, 2026

## Purpose
Define Python-C++ boundary contract for sensor/film slot data upload and metadata reporting.

## Schema: Sensor Payload (SSBO Binding 15-22, 8 slots)

Each slot carries exactly one SensorRecord (192 bytes, 48 floats).
Packed as std430 GLSL compatible, layout defined in sensor_film_db.py.

Python type: numpy.ndarray shape (8, 48) dtype float32
C++ type: SensorRecord[8] from _spectral_kernels.h
GLSL type: uniform SensorSlot[8] at binding 15-22

## Schema: Film Payload (SSBO Binding 23-30, 8 slots)

Each slot carries exactly one FilmRecord (256 bytes, 64 floats).
Packed as std430 GLSL compatible, layout defined in sensor_film_db.py.

Python type: numpy.ndarray shape (8, 64) dtype float32
C++ type: FilmRecord[8] from _spectral_kernels.h
GLSL type: uniform FilmSlot[8] at binding 23-30

## Schema: Slot Metadata (Python-side, reported in JSON)

```python
slot_metadata = {
    "slot_id": int,                    # 0-7
    "sensor_id": int,                  # index into SensorFilmDatabase
    "sensor_name": str,                # e.g. "canon_eos_r6"
    "film_id": int,                    # index into SensorFilmDatabase
    "film_name": str,                  # e.g. "kodak_portra_400"
    "active": bool,                    # True if sensor_id >= 0 and film_id >= 0
    "qe_peak": float,                  # quantum efficiency
    "read_noise_e": float,             # electrons RMS
    "dark_current_e_s": float,         # dark current per second
}
```

## Contract: Upload Interface (Python -> C++)

Method: ExposureSession._bind_sensor_film_ssbo(tracer)
Precondition: tracer object exists and has method set_sensor_film_ssbo
Action: upload sensor (8,48) and film (8,64) tensors

```python
tracer.set_sensor_film_ssbo(
    sensor_chunk=tensors['sensor'],      # (8, 48) float32
    film_chunk=tensors['film'],          # (8, 64) float32
    active_slots=active_slot_list        # list of (sensor_id, film_id) for active slots
)
```

## Contract: Endpoint Record Extension (BDPT -> Sensor Integral)

Endpoint records must include:
- slot_id: which sensor/film slot was hit (0-7)
- position: xyz world coordinates
- spectral_amplitude: complex spectral integrand at this endpoint

Current code already has position and spectral_amplitude.
Extend records to include slot_id from tracer.register_tri_group context.

## Contract: Artifact Output

Per active slot, save three NPZ files:
1. {frame:04d}_slot{slot_id:02d}_photons.npz
   - photons_per_pixel: (height, width) float32
   - sensor_id: int
   - film_id: int

2. {frame:04d}_slot{slot_id:02d}_electrons.npz
   - electrons_per_pixel: (height, width) float32
   - qe_peak: float (used in computation)
   - sensor_id: int
   - film_id: int

3. {frame:04d}_slot{slot_id:02d}_snr.npz
   - snr_linear: (height, width) float32
   - read_noise_e: float
   - dark_current_e_s: float
   - exposure_time_s: float (from film)
   - sensor_id: int
   - film_id: int

## Invariants

1. All slot IDs are [0, 8).
2. Inactive slots have sensor_id = film_id = -1.
3. Tensor row N corresponds to slot N.
4. Endpoint accumulation is per-slot (no cross-slot photon bleed).
5. SNR computation: SNR = sqrt(electrons) / sqrt(read_noise^2 + dark_current*t + electrons)
