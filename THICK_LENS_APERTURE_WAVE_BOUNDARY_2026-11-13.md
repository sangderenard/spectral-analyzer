# Exact lens and aperture wave-region boundary (2026-11-13)

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
