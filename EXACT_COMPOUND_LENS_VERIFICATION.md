# Exact Compound-Lens Hot-Path Verification

## Finding

The compound lens is participating in the native GPU camera hot path. The old
log text `thin-lens aperture sampling` described the aperture-domain proposal,
not the propagation model, and has been renamed to `aperture-domain proposal
samples`.

When an optical camera is present, the renderer selects
`CAMERA_MODE_PARAMETRIC_ASSEMBLY`, disables the surrogate scale contexts, and
registers forward and backward `PARAMETRIC_LENS` payloads. The GPU T2 shader
then performs, for every camera path that reaches the assembly:

1. a loop over every surface in the payload;
2. closed-form plane or conic intersection;
3. clear-aperture and stop checks;
4. wavelength-selected refractive-index lookup;
5. exact surface-normal evaluation;
6. vector Snell refraction, Fresnel evaluation, and TIR rejection;
7. optical-path-length accumulation and spectral phase advance;
8. exit-state placement only after the complete loop succeeds.

The proxy triangle mesh is used to enter the exact handler through the BVH; it
does not replace the per-surface conic solve.

## Runtime proof added

The exact GPU handler now increments counters that are read and logged beside
the already-synchronized T5 record counts:

- `lens_paths_attempted`
- `lens_surface_intersections`
- `lens_refractions`
- `lens_stop_passes`
- `lens_aperture_rejections`
- `lens_hood_rejections`
- `lens_tir_events`
- `lens_paths_exited_front`
- `lens_surfaces_mean`, `lens_surfaces_min`, `lens_surfaces_max`
- `lens_absorber_rejections`

These cannot be satisfied by configuration labels: they are incremented inside
the shader's actual conic/Snell loop.

## Bounded verification result

A native GPU-resident BDPT render of the current 110,268-triangle prism scene
registered `n_surfs=9`. Its successive exact-lens telemetry included:

```text
lens_paths_attempted=82
lens_surface_intersections=738
lens_refractions=656
lens_stop_passes=82
lens_paths_exited_front=82
lens_surfaces_mean=9.000
lens_surfaces_min=9
lens_surfaces_max=9
```

Later in the same bounded exposure:

```text
lens_paths_attempted=16405
lens_surface_intersections=147608
lens_refractions=131205
lens_stop_passes=16400
lens_aperture_rejections=3
lens_paths_exited_front=16400
lens_surfaces_mean=9.000
lens_surfaces_min=9
lens_surfaces_max=9
lens_absorber_rejections=2
```

The accounting is internally consistent: every successful camera path crossed
all nine authored surfaces, with eight refractive interfaces and one stop.
Five attempted paths were rejected rather than teleported through a surrogate.

## Remaining high-value A/B test

The counters prove that exact per-surface traversal is executing. A separate
regression should still perturb one internal curvature while holding the
sensor plane fixed, then require a measurable PSF/distortion change. That test
would protect against a future payload-authoring regression even though the
current hot-path ambiguity is resolved.
