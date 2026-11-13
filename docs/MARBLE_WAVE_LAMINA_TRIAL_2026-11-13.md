# Marble wave lamina trial (2026-11-13)

The authored Spectral thick lens T2 and native T4 now pass a focused
sensor-to-scene trial with a 10 mm spherical subject. A ray traverses the
exact compound lens, enters a local T4 exterior patch ending at the subject
surface, and returns to the ray pipeline. The patch is 8 µm across its
transverse half extent and 0.1 mm long along the optical axis. The requested
0.2 µm sample pitch is below half of the shortest 450 nm test wavelength;
the local curvature sag is about 3.2 nm. This trial proves routing and
sample configuration, not a rendered picture or a wave material bounce.

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
