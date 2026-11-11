# Internal medium directional observer

`camera_designer/studio_directional_media.py` supplies the optional observer
composition. It does not select a public schema or enable itself. The owner of
the directional job must explicitly select and describe the new recording scope.

Apply `with_internal_medium_recording` after the canonical boundary/region/sphere
and directional shader adaptations, before optional external extinction binding.
It moves only directional recording before both canonical Beer updates. The
legacy scalar field's exterior-only guards and the primary transport remain
unchanged. Boundary refraction, Fresnel reflection, TIR, medium ownership and dye
selection remain the existing engine's responsibility.

For each actual segment, the recorder uses entry packet energy and the installed
medium's constant absorption coefficient. Each crossed recording cell receives
the integral from its entry distance a to exit distance b:

`entry_energy * exp(-sigma*a) * (1-exp(-sigma*(b-a))) / sigma`

The zero-absorption limit is the segment length. A small-optical-depth expansion
avoids cancellation. The coefficient helper is mechanically derived from the
installed canonical transmittance function body, including additive region dye
absorption. The original transmittance code is retained verbatim. There is no
inverse of an underflowed endpoint transmittance and no second material law.

Invalid region ownership (-2) is excluded. Exterior air (-1) has zero material
absorption. Only paths the real tracer already traversed can populate the field.
An empty subject atlas is not proof of a glass-transfer failure and does not
authorize extra atmospheric rays, scattering, gain, or invented interior light.

## Explicit unsupported combination

Internal recording plus `ExtinctionGrid` fails closed. Its correct integral must
combine the material coefficient and every supplied grid coefficient over the
union of their spatial intervals. Multiplying two independently integrated
lengths is incorrect. Existing external-extinction jobs without this optional
internal observer retain their previous path.

## Verification

`python -m pytest tests/test_studio_directional_media.py -q`

The focused tests cover canonical legacy and region helper derivation, both
forward/reactive entry-packet hooks, unchanged primary Beer and scalar guards,
root-sweep and ROI compatibility, clear and strongly absorbing millimetre rods,
strict rejection of changed laws, external-grid refusal, and actual GLSL compile
checks. These CPU/compile gates alone do not prove real ray intersections.

`python tests/studio_directional_media_gpu_check.py`

Run the native proof only in a coordinated free GPU window. It uses a bounded
real glass slab scene through the actual forward tracer, compares scalar bytes
with/without the observer, checks internal Beer track integrals, and measures
the Snell-refracted propagation. It does not validate a production subject or
claim that the subject receives enough source samples for a complete image.

Measured 2026-11-11 on AMD Radeon(TM) Graphics: the ten focused CPU/GLSL tests
passed, and the bounded canonical-tracer proof passed. Clear slab integral was
0.128908 versus analytic 0.127936; the 10%-transmission-per-millimetre slab was
0.027344 versus 0.027516. Differences stayed within the explicit per-cell integer
rounding bound of this legacy-65535 diagnostic recorder. Both runs preserved
scalar output bit-for-bit. The dominant refracted bin measured
[0.341743, 0, 0.939793], compared with Snell [1/3, 0, sqrt(8/9)] within the same
coarse direction quantization. The first probe incorrectly compared an all-path
mean containing actual internal reflections with a single Snell ray; the test
was corrected to inspect the dominant bin. No optical implementation was changed
to satisfy that test. This receipt does not cover live v6 integration or an actual
subject exposure.
