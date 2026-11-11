# Real optional optical field input, 2026-11-11 UTC

Added `camera_designer/studio_volume_extinction.py` and opt-in seams in the
existing prebuilt forward tracer, progressive sensor, local photo worker,
cooperative sensor continuation and Auto bounce probe. There is no second ray
tracer, fluid solver, fake illumination source or RH-to-fog conversion.

The input is an explicitly declared, finite nonnegative, additional scalar
extinction grid (m^-1), world bounds in metres, z/y/x cell layout, at most
64 cells per axis. GPU SSBO 20 carries those actual values. Exact cell path
lengths feed Beer-Lambert attenuation for forward/reactive rays, legacy volume
samples, actual camera surface paths and source connections. The optional
directional recorder integrates transmittance over its own segment-cell
intervals. Missing input keeps the original shader source and defaults.

The existing Turing SPH reconstruction can now publish through
`ExtinctionGrid.from_density_sampler(fluid.sample_at, bounds, dimensions,
mass_extinction_m2_kg=...)`. It requires an explicit valid optical coefficient.
Independent cross-repository measurement: 0.5163623094558716 m^-1 versus
0.516362301252964 from the direct two-particle kernel sum with a synthetic
test coefficient. Particle position, velocity, mass, density, pressure,
temperature and salinity remain unchanged. Empty particles publish zeros.

## Verification

- Installed render Python: 114 focused tests passed, one skipped, covering optical-grid, native conductor,
  cooperative sensor, sensor evidence, tile-coordinate, worker lifecycle,
  preview status, bounce-probe, directional field and forward publication.
- Actual SPH cross-repository check passed (`tests/studio_sph_extinction_check.py`).
- Changed Python files compile; `git diff --check` passes.
- An additional live-history test fails at `dict(camera)` with camera=None in
  `marble_live_worker.py:193`. Both that file and the failing test are byte-for-
  byte unchanged from this repository's HEAD 89ca6eb. No unrelated fix made.
- The Turing venv lacks PyYAML for some spectral tests; rerunning with the
  installed render Python resolved those environment failures without installs.
- Real-GPU verification passed after the shared tiled exposure completed, on
  AMD Radeon(TM) Graphics. Six complete forward/sensor/region/12-bin recorder
  combinations linked. 544 ray integrals matched an independent plane-event
  oracle with maximum absolute error 2.3295301563308612e-6.
- Native prebuilt/sensor fixture: omitted versus explicit zero grid gave
  bit-identical scalar field, directional record and sensor pixels. A 2 m^-1
  field over a 0.4 m camera path gave 0.4493289291858673 transmission versus
  exp(-0.8)=0.44932896411722156. Isolated direct-light plus camera paths gave
  0.20189644396305084 versus exp(-1.6). Measured directional track-length ratio
  was 0.6883726253147173 versus (1-exp(-0.8))/0.8. Owned volume buffers were
  confirmed released. The script exited before the next worker's GPU tests.
- Auto-probe and cooperative-continuation optional-field seams have CPU/source
  coverage, not an end-to-end cooperative network or Auto exposure GPU test.

## Limits

This is extinction-only, achromatic, additional to existing material
absorption. It adds no scattering albedo, phase function, emission or IOR.
Cell-centre SPH publication is a spatial approximation, not a conservative
mass remap. The local worker accepts a static world-space snapshot; Studio
HTTP/browser admission and live simulation-field refresh are not wired yet.

The acceptance bowl is not complete: actual mesh/SPH contact coupling and
owned liquid free-surface/material interfaces are still required. The exact
Studio mesh bake and native contact path remain the canonical starting
points. No static visual liquid was substituted for simulation.

No changes were committed, deployed or used to interrupt the active exposure.
See `camera_designer/STUDIO_VOLUME_EXTINCTION.md` for the precise API contract.
