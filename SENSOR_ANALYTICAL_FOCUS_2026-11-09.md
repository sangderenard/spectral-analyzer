# Analytical sensor focus convention and bounded validation

## Implemented scope

`demo_pluck_gl.py` keeps its existing composed `SensorAccumulator` shader,
spectral/region/history/evidence components, pupil sampling, and transport.
Only the analytical camera focus calculation and three CPU focus clamps changed.
The constructor, `set_camera`, and `set_lens` now use a 0.001 m numerical floor,
matching the GLSL input guard. This is not a physical minimum focusing distance.
The Studio API/UI's custom range and lens-contract validation are separate.
The generic accumulator does not acquire a 10 m upper limit.

## Exact geometry

Let E be `uCamEye` and R, U, F the existing orthonormal camera basis, including
any precompiled gimbal rotation. Let x and y be the shader's horizontal and
vertical pinhole slopes (`fu`, `fv`), including the existing principal-point
shift, field of view, and aspect ratio.

For tilt angles `nod = uLensTilt.x`, `pan = uLensTilt.y`, the established signed
tilt convention is the plane, in camera-relative coordinates:

    z - x_world * tan(pan) - y_world * tan(nod) = d

Here d is the axial plane intercept from E, in metres. It is not distance from
the sensor, from a real front element, or distance along each off-axis ray.
Equivalently its normal is `N = F - R*tan(pan) - U*tan(nod)`, with
`dot(N, P-E) = d`.

The pinhole ray vector is `Q = F + R*fu + U*fv`. Its dot product with N is
`D = 1 - fu*tan(pan) - fv*tan(nod)`. Exact ray/plane intersection therefore gives
`P = E + Q*d/D`. Production retains the existing near-horizon guard and uses:

    guarded_d = max(uFocusDist, 0.001)
    guarded_D = max(D, 0.005)
    P = E + Q * guarded_d / guarded_D

For D >= 0.005 this is the exact plane intersection. The original denominator
guard deliberately bounds near-parallel or backward plane intersections; those
clipped rays must not be described as physically accurate tilt behavior. Tilt
signs, input angle order, and the guard threshold did not change. The 1 mm floor
applies to the axial intercept, not to each tilted ray's intersection depth.

The previous code used `normalize(Q)*d/D`. Without tilt that puts P on a sphere
of radius d and makes its axial depth `d/sqrt(1+fu^2+fv^2)`. For a 90-degree square
view at a corner, a 20 mm input yielded 11.547 mm axial depth. Removing that
normalization corrects the spherical focus error.
The existing pupil samples still originate at E plus an R/U aperture offset and
point to P. With zero aperture, the normalized scene-ray direction is unchanged.

## Boundaries and unchanged paths

- Named physical lens minimum focus remains source/validation data. Allowing a
  custom mathematical focus at 1 mm does not validate a real lens, a real image
  conjugate, mechanical clearance, close-range radiometry, or a lens actuator.
- Callers must supply finite settings. Public API finite/range validation is
  separate from this small numerical safeguard.
- `uUseManifold`'s precomputed-ray branch is unchanged. The separate analytical
  bake in `camera_software/lens_manifold.py` still uses the previous spherical
  construction and 10 mm clamp. Its historical "mirrors GLSL exactly" comment
  no longer holds for this analytical branch. Do not claim cross-path parity or
  enable that bake for Studio as part of this change.
- Because focus semantics changed, Studio job/cache identity must include the
  new axial-focus semantic version; old photographs are not validation of it.
- No native compound-lens renderer, live server, current photograph process,
  aperture sampler, random sequence, or spectral/color path was replaced.

## Verification

CPU command:

    python -m pytest -q tests/test_sensor_analytical_focus.py tests/test_studio_camera_geometry.py

Result on final code: **202 passed in 8.59 seconds**. The new tests execute the actual CPU
entry-point assignments and GLSL expressions without importing the renderer or
allocating GL resources. Independent ray/plane calculations cover 1-19 mm focus,
the floor threshold, larger inputs, translated/rotated camera, off-axis/shifted
pixels, both signed tilt axes, pupil-origin convergence, pinhole invariance, and
finite bounded behavior at the retained tilt horizon guard. `py_compile` and
`git diff --check -- demo_pluck_gl.py` also passed.

Opt-in actual GL command, run only after a fresh process check and GPU slot
coordination, under a 120-second subprocess timeout:

    python tests/sensor_analytical_focus_gpu_check.py

Result on **NVIDIA GeForce RTX 3060**:

- Both full composed triangle and region sensor shaders compiled and linked.
- Each executed 30 cases / 270 actual GLSL camera samples, 540 samples total.
- Inputs: 0.5 mm (clamped), 1 mm, 5 mm, 19 mm, and 10 m focus; zero and nonzero
  tilt; pinhole, disk pupil, and rotated seven-blade pupil; nine sensor positions.
- The maximum plane residual below 20 mm was **1.080e-8 m**. Across all inputs,
  including 10 m, the maximum was **5.697e-6 m** (single-precision GLSL).
- GL context creation, compilation, and sample checks took **0.875 seconds**;
  the bounded Python process including imports and shutdown took **3.969 seconds**.
- The probe freed its owned buffers/programs and exited normally. Fresh process
  checks before and after found no live exposure or check jobs. No production
  process was restarted or interrupted.

Machine-readable receipt with composed shader SHA-256 values and every case:
`build/dot-analytical-focus-20261109/gpu-receipt.json`.

This is real shader compilation and camera-sampling validation, not a new
photograph, full renderer regression, radiometric test, or physical lens proof.
