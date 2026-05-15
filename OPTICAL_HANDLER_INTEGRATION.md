# Phase C: Optical Handler Integration Guide

## Overview

Phase C implements **C++ optical handlers** that process rays through the optical assembly, applying physics-based transformations and recording event telemetry.

**Deliverables:**
- `csrc/include/optical_handlers.h` — Handler interface and role enums ✅
- `csrc/optical_handlers.cpp` — Role-specific handler implementations ✅
- `csrc/bindings/pybind_optical_assembly.cpp` — Python bindings ✅
- Integration with RayTracer kernel (in progress)
- Event telemetry population (in progress)

---

## Architecture

### Handler Chain Processing

Each ray passes through a **z-ordered sequence of handlers**, each applying element-specific physics:

```
Ray (oracle/physical mode)
  ↓
STOP (aperture blocking)
  ↓ [if passes aperture]
REFRACTOR (first lens element)
  ↓ [if transmitted]
MIRROR (reflection surface)
  ↓ [if not absorbed]
FILTER (wavelength-dependent loss)
  ↓ [if passes filter]
DIFFUSER (scattering surface)
  ↓ [if bounced]
SENSOR (terminal collection)
  ↓
Ray collected + Telemetry updated
```

### Ray State Evolution

Each handler receives and modifies **OpticalRayState**:

```cpp
struct OpticalRayState {
    double pos[3];                          // Position (meters)
    double dir[3];                          // Unit direction vector
    double complex amplitude;               // Complex amplitude (V/m)
    double phase_error;                     // Accumulated phase error (rad)
    double wavelength_m;                    // Vacuum wavelength (meters)
    double cumulative_path_length_m;        // Total distance (meters)
    int bounce_count;                       // Surface hits
    bool is_active;                         // Still propagating?
    bool hit_sensor;                        // Reached terminal surface?
};
```

### Handler Result Aggregation

Handlers return **OpticalHandlerResult** with event counts:

```cpp
struct OpticalHandlerResult {
    int status;                             // 0=continue, 1=blocked, 2=absorbed
    int rays_blocked;                       // Blocked by aperture
    int rays_hit_surface;                   // Surface intersection
    int rays_refracted;                     // Snell's law refraction
    int rays_reflected;                     // Specular reflection
    int rays_absorbed;                      // Absorbed (or collected at sensor)
    int rays_transmitted;                   // Passed through filter/diffuser
};
```

Telemetry aggregation updates **CameraEventTelemetry**:

```python
@dataclass
class CameraEventTelemetry:
    rays_launched: int = 0
    rays_blocked_by_stop: int = 0
    rays_hit_lens_surface: int = 0
    rays_refracted: int = 0
    rays_reflected: int = 0
    rays_total_internal_reflection: int = 0
    rays_entered_wave_region: int = 0
    rays_exited_wave_region: int = 0
    rays_deposited_sensor: int = 0
    rays_out_of_domain: int = 0
    rays_fell_back_to_full_solve: int = 0
    energy_in: float = 0.0
    energy_out: float = 0.0
    energy_absorbed: float = 0.0
    energy_blocked: float = 0.0
    mean_phase_error: float = 0.0
    mean_focus_error: float = 0.0
```

---

## Handler Implementations

### STOP Handler (Aperture Blocking)

**Purpose:** Block rays outside aperture diameter

**Physics:**
1. Compute ray-plane intersection at `center_z_m`
2. Check radial distance from optical axis
3. Block if `r > radius_m`

**Event Counts:**
- `rays_hit_surface`: rays hitting aperture plane
- `rays_blocked`: rays outside aperture diameter

**Status:** ✅ Implemented (basic plane intersection)

### REFRACTOR Handler (Lens Refraction)

**Purpose:** Apply Snell's law refraction + Fresnel reflection

**Physics:**
1. Compute ray-plane intersection
2. Apply Snell's law: `n1*sin(θ1) = n2*sin(θ2)`
3. Compute Fresnel reflection coefficient: `R = |R_amp|*exp(j*φ_r)`
4. Transmission = `√(1 - |R|²)`
5. Apply absorption: `exp(-α*distance)`

**Event Counts:**
- `rays_hit_surface`: rays hitting lens surface
- `rays_refracted`: transmitted rays
- `rays_reflected`: Fresnel reflections (partial)

**Status:** ✅ Implemented (simplified; needs full Snell's law)

### MIRROR Handler (Specular Reflection)

**Purpose:** Specular reflection with Fresnel laws

**Physics:**
1. Compute reflection direction (mirror normal method)
2. Apply Fresnel coefficient
3. Update amplitude and direction

**Event Counts:**
- `rays_hit_surface`: rays hitting mirror
- `rays_reflected`: reflected rays

**Status:** ✅ Implemented (planar reflection; needs curved surfaces)

### FILTER Handler (Wavelength Absorption)

**Purpose:** Wavelength-dependent absorption

**Physics:**
1. Apply wavelength-dependent absorption: `exp(-α(λ)*thickness)`
2. Update amplitude

**Event Counts:**
- `rays_hit_surface`: rays passing through filter
- `rays_transmitted`: transmitted rays

**Status:** ✅ Implemented (constant absorption; needs spectral curves)

### DIFFUSER Handler (Lambertian Scattering)

**Purpose:** Lambertian diffuse scattering

**Physics:**
1. Sample scattered direction from hemisphere
2. Apply albedo factor
3. Continue propagation

**Event Counts:**
- `rays_hit_surface`: rays hitting diffuser
- `rays_transmitted`: scattered rays

**Status:** ✅ Implemented (placeholder; needs cosine-weighted sampling)

### SENSOR Handler (Terminal Collection)

**Purpose:** Terminal photon collection

**Physics:**
1. Mark ray as collected
2. Terminate propagation
3. Record photon energy

**Event Counts:**
- `rays_hit_surface`: rays reaching sensor
- `rays_absorbed` (actually deposited at sensor)

**Status:** ✅ Implemented (basic termination)

---

## Integration with RayTracer Kernel

### 1. Update RayTracer Initialization

Modify `ray_tracer_create()` to accept OpticalAssembly:

```cpp
RayTracerState* ray_tracer_create(
    const char*            scene_path,
    const OpticalAssembly* optical_assembly,  // NEW
    int                    camera_mode,       // NEW
    /* ... other parameters ... */
);
```

### 2. Update Ray Transport Loop

In the main ray tracing loop (within `ray_tracer_trace_callback()`):

```cpp
// For each ray:
OpticalRayState ray_state;
// ... initialize from camera sample ...

// Process through optical assembly
OpticalHandlerResult assembly_result = optical_assembly_process(
    state->optical_assembly,
    &ray_state,
    &ray_state_out,
    state->camera_mode
);

// Update telemetry
state->camera_event_telemetry.rays_hit_lens_surface += 
    assembly_result.rays_hit_surface;
state->camera_event_telemetry.rays_refracted += 
    assembly_result.rays_refracted;
state->camera_event_telemetry.rays_blocked_by_stop += 
    assembly_result.rays_blocked;
// ... etc for other counters ...
```

### 3. Export Telemetry to Python

After render completes, populate **CameraEventTelemetry** on ExposureBackend:

```cpp
// In backend completion handler:
backend->camera_event_telemetry = {
    .rays_launched = state->rays_launched,
    .rays_blocked_by_stop = state->event_counts.rays_blocked,
    .rays_hit_lens_surface = state->event_counts.rays_hit_surface,
    .rays_refracted = state->event_counts.rays_refracted,
    .rays_reflected = state->event_counts.rays_reflected,
    // ... etc for all 17 fields ...
};
```

### 4. Mode-Based Ray Treatment

Implement mode-based photon buffer selection:

```cpp
// In camera sensor loop:
if (camera_mode == CAMERA_MODE_ORACLE_PINHOLE_REFERENCE) {
    // Oracle mode: use perfect optics, no losses
    apply_oracle_physics(&ray_state);
    oracle_photons.push_back(ray_state);
} else {
    // Physical mode: apply full physics with losses
    apply_physical_physics(&ray_state);
    physical_photons.push_back(ray_state);
}
```

---

## Python-to-C++ Integration

### 1. Create OpticalAssembly in Python

```python
from optical_assembly import OpticalAssembly, OpticalElement, OpticalElementRole

assembly = OpticalAssembly(
    name="camera_lens_assembly",
    elements=[
        OpticalElement(role=OpticalElementRole.STOP, ...),
        OpticalElement(role=OpticalElementRole.REFRACTOR, ...),
        # ... etc ...
    ]
)
```

### 2. Pass to C++ RayTracer

```python
# exposure_render_demo.py
backend = ExposureBackendCpp(
    optical_assembly=assembly,  # NEW
    camera_mode=int(camera_mode),
    # ... other parameters ...
)
```

### 3. Bind OpticalAssembly to C++ Handler Creation

In pybind wrapper (during backend initialization):

```cpp
// pybind_kernels.cpp
py::class_<ExposureBackendCpp>(m, "ExposureBackendCpp")
    .def("set_optical_assembly", 
        [](ExposureBackendCpp& self, py::object py_assembly) {
            // Convert Python OpticalAssembly to C++ handlers
            auto c_assembly = python_assembly_to_c_handlers(py_assembly);
            self.set_assembly(c_assembly);
        });
```

---

## Event Telemetry Mapping

**C++ Handler Events → Python CameraEventTelemetry:**

| C++ Handler Event | Python Field | Mapping |
|---|---|---|
| STOP.rays_blocked | rays_blocked_by_stop | direct |
| REFRACTOR.rays_hit_surface | rays_hit_lens_surface | sum of refractor hits |
| REFRACTOR.rays_refracted | rays_refracted | direct |
| REFRACTOR.rays_reflected | rays_reflected | direct (partial) |
| MIRROR.rays_reflected | rays_reflected | sum with refractor |
| SENSOR.rays_absorbed | rays_deposited_sensor | direct |
| Assembly.status==blocked | rays_out_of_domain | when blocked |
| Ray amplitude loss | energy_absorbed | integral of |amplitude|² |
| Ray amplitude final | energy_out | integral of |amplitude|² |
| Phase deviation | mean_phase_error | mean phase_error field |

---

## Validation Tests

### Test 1: STOP Handler Blocks Rays

```python
# exposure_render_demo.py with APERTURE_CONE mode
# Expected: rays_blocked_by_stop > 0
# Expected: Some rays should pass through aperture
validate_event_count(result, "rays_blocked_by_stop", min=1, max=total_rays*0.5)
```

### Test 2: REFRACTOR Handler Bends Rays

```python
# THIN_LENS_GEOMETRIC mode
# Expected: rays_refracted > 0
# Expected: Final image converges near focal plane
validate_event_count(result, "rays_refracted", min=total_rays*0.9)
```

### Test 3: Oracle vs Physical Modes Differ

```python
# Compare mode 0 (ORACLE_PINHOLE_REFERENCE) vs mode 3 (THIN_LENS)
# Expected: oracle_photons ≈ physical_photons (nearly identical)
# Expected: energy_out[oracle] > energy_out[physical] (less loss)
assert oracle_energy > physical_energy
```

### Test 4: Telemetry JSON Export

```python
# Check frame_config_summary contains telemetry
import json
summary = json.loads(result.frame_config_summary_json)
assert "optical_event_telemetry" in summary
assert summary["optical_event_telemetry"]["rays_launched"] == total_rays
```

---

## Next Steps (Phase D+)

### Phase D: Wave Propagation Handler
- Implement WAVE_REGION handler for diffraction
- Add Fresnel zone computation
- Integrate with Kirchhoff diffraction integral

### Phase E: Thermal Effects
- Implement `dn_dT` thermal sensitivity
- Track ray focus error due to thermal drift
- Validate focus_error field in telemetry

### Phase F: Performance Optimization
- SIMD vectorization of ray-plane intersections
- Batched Fresnel coefficient computation
- GPU acceleration of handler chains

---

## File Changes Summary

**Created:**
- `csrc/include/optical_handlers.h` (248 lines) — Handler interface
- `csrc/optical_handlers.cpp` (687 lines) — Handler implementations
- `csrc/bindings/pybind_optical_assembly.cpp` (109 lines) — Python bindings
- `OPTICAL_HANDLER_INTEGRATION.md` (this file) — Integration guide

**To Modify:**
- `csrc/include/ray_tracer.h` — Add OpticalAssembly member, handler registry
- `csrc/ray_tracer.cpp` — Integrate handlers into ray loop, telemetry aggregation
- `csrc/bindings/pybind_kernels.cpp` — Export OpticalAssembly, telemetry
- `CMakeLists.txt` — Add optical_handlers.cpp, pybind_optical_assembly.cpp to build

**To Verify:**
- `optical_assembly.py` — Already complete, provides Python side
- `camera_mode_validation.py` — Already complete, provides constraints
- `exposure_render_demo.py` — Pass OpticalAssembly to backend

---

## Success Criteria

- [ ] Handler interface compiles without errors
- [ ] All 7 handler types instantiate successfully
- [ ] OpticalAssembly chains handlers in z-order
- [ ] Event counts aggregate correctly
- [ ] Mode 0 (ORACLE) renders with perfect optics
- [ ] Mode 3 (THIN_LENS) renders with focus convergence
- [ ] Telemetry exports to JSON correctly
- [ ] Oracle vs physical photons differ as expected
- [ ] All integration tests pass
- [ ] Performance overhead <5%

