# Exact lens and aperture wave-region boundary (2026-11-13)

## 2026-11-13 traversal correction

The CPU T2 evaluator now yields at the entrance plane of a physical T4
ScaleContext when a parametric lens stop lies in that arena. Its continuation
retains the original exact-conic payload triangle and next surface index;
after T4, T1 routes it back to T2 without intersecting an interior absorber.
The test below now passes scene-to-sensor and sensor-to-scene with one and eight
spectral lanes. The scene-side reverse ray exits past the first lens surface;
the forward ray reaches sensor-side geometry after the wave crossing.

The optical graph also supplies the stop's authored refractive indices to the
wave arena, including per-lane values through an optional tail on the existing
APTR payload. The native T4 exit records optical path using that medium. A
full-GPU T1/T2/T3 request with this interior stop is rejected at pipeline
creation: the GPU intent ABI does not yet carry the resume state.

The following is the original failure receipt, retained for the history of
the boundary. The remaining field-to-ray-to-field Jones continuation failures
are still recorded in `WAVE_CONTINUATION_ENERGY_AUDIT_2026-11-09.md`.

## Original failing receipt

The focused native acceptance test in `tests/test_wave_t4_pipeline.py` builds
the canonical four-group camera from its manifest, registers its exact conic
transport through `build_tracer`, compiles a physical circular aperture into a
T4 wave arena, and submits a center ray from scene side toward the sensor.
The graph installs and the 62×62 transverse field is allocated, but its
generation remains zero. This is a strict expected failure, not a completed
thick-lens wave exposure.

The native record from this trial puts the initial hit at X=-0.179 m and the
next segment start at X=-0.04637295 m. The aperture's authored center is
X=-0.143031506 m. The exact T2 parametric group traverses its surfaces in one
call and emits a ray after the group. T1's arena-entry test only examines the
segment before a T1 hit, so it never sees the region crossed *inside* that T2
call. T4 processed zero rays. Moving the source farther back did not change
that result. The test uses one 550 nm lane to isolate the boundary and makes no
claim about eight-lane or continuous spectral operation.

The aperture region should stay an authored ScaleContext/T4 arena. Its axis,
radius, thickness, and material payload already come from
`compile_compound_lens_graph` → `wave_context_nodes` → `install_optical_graph`.
The missing work is an ordered handoff inside exact lens traversal at the
physical aperture, preserving direction, amplitude/Jones state, optical path,
subpath identity and PDFs when T4 returns. This must be checked in both scene
to sensor and sensor to scene directions before connecting T5. A successful
arena allocation alone is not transport evidence.
