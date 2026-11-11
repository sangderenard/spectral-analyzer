# Declared volume extinction in native transport

`_gpu_ray_field_prebuilt(..., volume_extinction=ExtinctionGrid(sigma, bounds))`
accepts actual additional optical coefficients. `sigma` is a finite,
nonnegative float32 array in **z, y, x** order, with each dimension between 1
and 64. Values are achromatic extinction in **inverse metres**, constant
within each cell. `bounds` is `[world_xyz_min_m, world_xyz_max_m]`. Outside
that box the additional coefficient is zero.

Pass `state['volume_extinction']` to `SensorAccumulator(volume_extinction=...)`.
This is a borrowed GPU binding. Destroy camera accumulators before calling
`_destroy_prebuilt(state)`. The prebuilt state owns and releases the field.
The optional binding occupies SSBO 20 and requires at least 21 binding slots.

No declaration means no allocation, no shader specialization and no added
extinction or scattering. Existing legacy medium defaults are unchanged.
An explicit all-zero field is also optically neutral.

## What the field changes

The existing forward, reactive and camera ray paths apply Beer-Lambert
transmittance `exp(-integral(sigma_t ds))`. Existing forward splats and camera
volume samples use attenuation up to their sample position; surface
throughput is attenuated up to the actual hit. Existing source visibility
connections include the field along their connection. This works when the
photo worker's legacy `vol_alpha` is zero too.

The field is integrated by bounded voxel traversal, not by a new density
estimator. For the supplied piecewise-constant coefficients it computes exact
cell path lengths, subject to float32 arithmetic. The optical-depth integral
is independent of the legacy volume-march step count. The optional directional
recorder uses the integral of transmittance along each recorded segment-cell,
not unattenuated entry energy. Its metadata identifies the field and SHA256.

Coefficients are **additional** to existing native material absorption. Do
not double-count a material already represented by the existing medium law.
The caller must publish zeros in excluded regions or supply only the missing
coefficient. This interface does not decide overlapping medium ownership.

## Local photo worker payload

Declare exactly `"volume_extinction": {"schema": "studio.volume-extinction.v1"}`
in the worker JSON. Put both arrays in its existing NPZ payload:

- `volume_extinction_sigma_t_m_inv`: the z/y/x coefficient grid.
- `volume_extinction_bounds_m`: the 2x3 world-space bounds.

Partial or undeclared payloads fail rather than silently inventing or ignoring
an atmosphere. The effective configuration includes the validated grid's
bounds, shape, units, reconstruction and digest. This is the local native
worker contract; it does not yet add a Studio HTTP/browser upload control.
It is one fixed world-space snapshot. Live particle-field publication is a
separate integration step, not implied by moving the scene geometry.

## Relationship to the real fluid and atmosphere systems

Turing's existing `DiscreteFluid.sample_at(points)['rho']` is the continuous
compact-kernel mass density. It can be sampled at these cell centres. To turn
mass density (kg/m3) into extinction requires an explicitly supplied,
material-valid mass extinction coefficient (m2/kg): `sigma_t = rho * kappa_t`.
There is deliberately no default kappa, artificial cloud density, particle
size, relative-humidity threshold or brightness gain here. The symbolic
chamber aerosol law already publishes `beta_ext` in m^-1 and is a direct
coefficient source when its coordinates/units and phase state match.

`ExtinctionGrid.from_density_sampler(fluid.sample_at, bounds_m, dimensions_xyz,
mass_extinction_m2_kg=kappa)` performs that publication through the existing
sampler, in bounded query batches. It requires the optical coefficient
explicitly and verifies density values. It does not step, replace or retune
the simulation. Supply the returned grid to the native ray-field parameter.

Sampling SPH at cell centres and then using constant cells is a spatial
approximation, not the original smooth kernel evaluated continuously along
every ray, and not a conservative mass remap. Resolve small features with an
adequate grid or use a future explicitly validated reconstruction.

Extinction alone removes radiance. It does **not** provide in-scattering,
phase function, emission, a liquid free surface or refractive index. Relative
humidity alone is not fog. A clear glass bowl containing moving liquid still
requires the real mesh/SPH contact coupling, a physically owned free-surface
boundary and the existing material interface law. This seam does not replace
those with visual liquid blobs or a second simulator.

## Checks

- `python -m pytest tests/test_studio_volume_extinction.py -q`
- `python tests/studio_volume_extinction_gpu_check.py` when the shared GPU is
  available: real shader-link, independent analytic ray integrals, native host
  zero-field identity, camera/source Beer-Lambert and directional integrals.
- `python tests/studio_sph_extinction_check.py [path_to_turing]`: actual Turing
  SPH publication versus the independent particle-kernel sum and unchanged
  physical state. The default Turing path is the existing sibling repository.

The GPU check must not run concurrently with a long exposure owned by another
worker. A passing CPU contract test is not evidence of a passing GPU check.

## Canonical material profiles

The optional `medium` extension now lives in the existing material catalog and
`MaterialDatabase`. `ExtinctionGrid.from_material_density_sampler` resolves a
declared profile using explicit transport wavelengths and opt-in acceptance of
its approximation, then calls the same density publication path above. It
preserves material provenance in metadata. The three coarse-dust references
provide an explicitly approximate geometric extinction law; molecular vapor
spectra remain unknown and cannot silently become fog or vacuum. See
[`../docs/20261111-participating-medium-catalog.md`](../docs/20261111-participating-medium-catalog.md).
