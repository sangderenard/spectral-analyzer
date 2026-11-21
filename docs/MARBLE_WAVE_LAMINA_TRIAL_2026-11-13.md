# Marble wave lamina trial (2026-11-13)

The authored Spectral thick lens T2 and native T4 now pass a focused
sensor-to-scene trial with a 10 mm spherical subject. A ray traverses the
exact compound lens, enters a local T4 exterior patch ending at the subject
surface, and returns to the ray pipeline. The patch is 8 µm across its
transverse half extent and 0.1 mm long along the optical axis. The requested
0.2 µm sample pitch is below half of the shortest 450 nm test wavelength;
the local curvature sag is about 3.2 nm. This trial proves routing and
sample configuration, not a rendered picture or a wave material bounce.
The required production coverage is a wavelength tuned margin around every
subject interface, including both sides of bubbles and rod/dye transitions;
this one exterior patch does not implement that rule.

T4 now accepts a per-lane imaginary index alongside the existing per-lane
real index. During each native angular-spectrum step it applies the
Beer–Lambert field factor `exp(-2π k dz / λ_vacuum)` and records the removed
power in the existing absorbed-power counter. Material attenuation and
numerical border loss now also have separate counters. A one-lane and a three-lane
field trial compare resulting power against the analytical attenuation.
The scalar `n_imag` on a scale context also now affects the field. The
existing aperture material operator remains separate from bulk extinction.

`wave_medium_from_regions` lowers **validated** Studio optical records to
wave indices without RGB inference. Host glass supplies its authored PBR
real index; a glass rod replaces that medium; injection dye adds authored
extinction while retaining host index; an air bubble selects air. It calls
the existing `optical_extinction` law, including its thickness and
concentration, and refuses unresolved rod/dye overlap. The optical records
are illustrative unless calibrated. Wavelengths outside their authored
range remain refused by the existing optical law.

A cold normal-incidence dielectric field-pair compiler now uses the existing
`dielectric_interface` law to produce both reflected and transmitted Jones
products for an authored interface. The bubble test verifies their combined
power is one at each of three wavelengths. This preserves coherent phase
and supplies a real boundary operator for a future local mesh-bounce lamina.
The authored-index pair also ran as two native T4 field links from a host
patch into reflected host and transmitted air patches. Both destinations
advanced, emitted field records, and their seeded powers summed to the
source power within the trial tolerance. This is one bubble interface,
not a full bubble with multiple walls or recombination.

## Boundary still missing

The mesh T3 bounce has not yet admitted a local T4 lamina from its hit,
normal, authored region transition and coherent source identity. The
10 mm subject cannot be one T4 FFT arena: its 256-cell cap would make
approximately 79 µm cells across 20.2 mm. The existing field graph can
branch, but coherent fan-in remains uninstalled. Multiple reflections from
bubble walls therefore do **not** yet recombine into structural color.
The new material-loss counter does not yet identify geometry-induced
structural loss: a structural ledger must still distinguish heat from reflection,
transmission and energy leaving the observed aperture. No structural color,
structural absorption, or full Studio exposure is claimed by this trial.

The next native handoff is T3 hit → authenticated region transition →
wave lamina patch. Studio's GPU scene has exact sphere and authored
rod/dye/bubble boundary classification; the native T3 path does not yet
consume that complete region contract, so source parity is an admission gate.
T4 should return both field products to T1/T5
with coherence and power preserved. Recombination needs the existing graph's
coherent reception gate completed before any interference image is claimed.

Verification: `python -m pytest tests/test_wave_t4_pipeline.py
tests/test_studio_optical_media.py tests/test_optical_transport_graph.py
tests/test_complex_optical_operators.py -q` passed 95 tests, including the
per-lane native attenuation trial, after the native rebuild.

## Subsequent native T3 region work (2026-11-13)

The CPU T1/T3 kernel now accepts Studio's validated region identities and
exact sphere owner records through `RayTracer.set_studio_regions` and
`set_studio_spheres`. It resolves the medium from the first crossing of each
authored boundary, including host, dye, rod, bubble and standalone material
roles. Segment propagation uses the resolved base index and additive dye
extinction. T3 probes both sides of a transmissive hit for its refractive
indices. Exact sphere intersections compete with mesh triangles, and the
owner triangle retains its material and region ID. A separate zero-threshold
membership query prevents the ordinary ray self-hit threshold from erasing
1 pm sided probes; the transport intersection keeps its original self-hit
gate. The micrometre triangle test and an exact sphere inside a host cube
both reach the native T3 pipeline.

`native_t5_scene_inputs` preserves the authored region/sphere records and
`create_native_t5_tracer` installs them with the existing native tracer. The
GPU all-stages path refuses Studio region scenes because its T1/T3 shaders
do not yet consume those records. Axial rod bundles are also refused because
their affine ownership is not represented by this CPU resolver.

This completes **region-side CPU transport admission only**. It does not
instantiate wavelength-tuned T4 patches at arbitrary mesh/sphere hits, cover
all subject interfaces, resolve an authored bundle, or deliver a native Studio
photo. Coherent fan-in remains the next problem **after** the complete T3 to
T4 surface handoff, not part of this CPU region result. The focused rebuilt
native suite (`test_studio_sphere_regions.py`, `test_studio_optical_media.py`,
`test_wave_t4_pipeline.py`, `test_studio_atlas_rest.py`,
`test_native_small_surface_transport.py`, and `test_studio_region_gpu.py`)
passed 73 tests; this is no claim of a rendered exposure.

## Authored interface classification (2026-11-13)

`ray_tracer_resolve_studio_wave_interface` now compares a T3 triangle ID
against the installed Studio region triangle lists. Exact spheres use their
owner triangle. It rejects a point that is not on the claimed triangle or
sphere, resolves the incoming and outgoing media with the existing native
region resolver, and returns the region ID, outward normal, complex indices
for every requested transport frequency, and a local margin sized from the
longest wavelength. The Python binding exposes the same query. Focused tests
cover host, dye, and bubble mesh boundaries and an exact bubble sphere.

This is the authenticated **interface classification** needed by the context
handler. It does not yet create a T4 patch at every hit, encode the local
surface into a field operator, or return reflected/transmitted fields to T1/T5.
Those remain the next transport steps; an interface query alone is not a
wave-traversed subject or a photograph.
