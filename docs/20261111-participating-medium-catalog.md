# Canonical vapor and dust material profiles

The existing `configs/materials/*.yaml` catalog now contains three molecular
vapors and three coarse solid-aerosol references. `MaterialDatabase` is still
the registry. `material_medium.py` validates the optional cold `medium`
extension; it does not hold another catalog or simulate particles.

## Profiles and evidence

| Material key | Known physical property | Optical status |
| --- | --- | --- |
| `water_vapor` | H2O; molar mass 0.0180153 kg/mol | Molecular spectrum unparameterized |
| `ethanol_vapor` | C2H6O; molar mass 0.0460684 kg/mol | Molecular spectrum unparameterized |
| `acetone_vapor` | C3H6O; molar mass 0.0580791 kg/mol | Molecular spectrum unparameterized |
| `quartz_dust_coarse` | SiO2; nominal grain density 2650 kg/m3 | Opt-in geometric sphere reference |
| `calcite_dust_coarse` | CaCO3; nominal grain density 2710 kg/m3 | Opt-in geometric sphere reference |
| `halite_dust_coarse` | NaCl; measured grain density 2168 kg/m3 | Opt-in geometric sphere reference |

Molar masses are from NIST Chemistry WebBook. Quartz grain density is from
USACE HEC-RAS. Calcite uses Missouri S&T's specific gravity 2.71 with the
explicit nominal 1000 kg/m3 water conversion. Halite uses the Handbook of
Mineralogy's measured 2.168 g/cm3, converted to SI. These are material
properties, **not** concentrations suspended in air. The YAML records carry
property-specific source URLs and explain every conversion and assumption.

- [NIST water](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185)
- [NIST ethanol](https://webbook.nist.gov/cgi/cbook.cgi?Name=ethanol&cTG=on)
- [NIST acetone](https://webbook.nist.gov/cgi/cbook.cgi?ID=C67641&Type=ANTOINE)
- [HITRAN cross-section definitions](https://hitran.org/docs/cross-sections-definitions/)
- [USACE sediment properties](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.6/model-description/water-and-sediment-properties/sediment-properties)
- [Missouri S&T calcite](https://ese.mst.edu/mineral-collection/calcite/)
- [Handbook of Mineralogy halite](https://www.handbookofmineralogy.org/pdfs/halite.pdf)
- [Extinction-efficiency / geometric-limit reference](https://journals.ametsoc.org/view/journals/atsc/79/11/JAS-D-22-0047.1.xml)

No spectral dataset was imported. Vapor temperature, pressure, species
concentration and absorption/scattering spectra remain required inputs.
Unknown coefficients are null, not an assertion of zero extinction. Water
vapor is not condensed fog. Relative humidity does not create a droplet phase.

## Explicit coarse-dust approximation

Each dust reference declares an **authored**, not measured, 25 micrometre
sphere radius (50 micrometre diameter), monodisperse dry particles, geometric
extinction efficiency Q_ext = 2, and a 400-700 nm evaluation interval. This is
an idealized large-particle reference, not a measured aerosol size distribution
or accuracy guarantee. Real shape, size variation, calcite birefringence and
salt deliquescence are unresolved. It is not a fine-particle Brownian model.

The mass extinction is derived from extinction cross-section divided by grain
mass: kappa_ext = Q_ext*pi*r^2 / (rho_grain*4*pi*r^3/3), in m2/kg. It is about
22.6415 for quartz, 22.1402 for calcite and 27.6753 for halite. No fitted visual
gain, sprite opacity, color or apparent particle size enters this calculation.
All transport wavelengths must be explicitly supplied and inside the declared
interval, and the caller must pass `allow_approximation=True`.

Single-scattering albedo, absorption/scattering split, phase function,
asymmetry, and spectral refractive index remain unknown. Extinction alone
removes radiance; this does not invent in-scattered light, glow or beams.
Surface PBR fields in these six files are neutral editor placeholders only.
The surface spectral evaluator and n/k volume adapter refuse medium profiles
rather than manufacturing a spectrum from the swatch. Existing non-medium
material behavior is unchanged.

## Existing volume-field integration

```python
from canonical_material_documents import read_material_document, native_payload
from material_db import MaterialDatabase
from camera_designer.studio_volume_extinction import ExtinctionGrid

database = MaterialDatabase()
name = "quartz_dust_coarse"
database.register(name, native_payload(read_material_document(name)).payload)
grid = ExtinctionGrid.from_material_density_sampler(
    database, name, species_fluid.sample_at,
    bounds_m, dimensions_xyz,
    wavelengths_m=[650e-9, 550e-9, 450e-9],
    allow_approximation=True,
)
# Pass grid through the existing native volume_extinction parameter.
```

The existing sampler must return `rho` as kilograms of this species per cubic
metre of carrier volume, with coordinates in metres. Do not pass total air
mass density or solid-grain density. The adapter calls the real sampler, then
publishes `sigma_t = rho_species*kappa_ext` in m^-1 into the existing bounded
extinction field. Metadata retains the profile digest, model, explicit
wavelengths, coefficient and approximation. Cell-centre sampling followed by
piecewise-constant ray traversal is not a conservative SPH mass remap.

No declaration means no extra field. Zero species density gives zero extra
extinction. A declared material with unknown optics fails explicitly rather
than silently rendering it as vacuum. The existing legacy atmosphere remains
unchanged. An active grid is additional extinction, so callers must avoid
double-counting an already represented medium.

This is the existing local Python/native transport seam. It does not yet add
Studio HTTP field admission, a live species-density publisher or browser
medium controls. Catalog source export already includes all six definitions.
The derived native surface chunks are editor data, not compiled medium optics.
Brownian debug sprites and their transparency remain a separate visualization
choice. No Brownian dynamics, free surface, glass bowl or mesh/SPH contact law
is introduced by these profiles.

## Verification receipt (2026-11-11)

- 94 targeted spectral tests pass across medium schema, canonical documents,
  derived catalog, extinction grid, Maxwell patch, transport grid, material
  authoring and conductor-surface compatibility.
- 39 directional-recorder, root-sweep and forward-field tests also pass, with
  1 existing skip. No changes were made to those concurrently owned sources.
- 2 Turing material-library export tests pass: 53 source records, 44 compiled
  surface records, 6 medium definitions preserved, 9 prior unresolved named
  color profiles still reported without fallback.
- `python tests/studio_sph_extinction_check.py` uses the real Turing
  `DiscreteFluid.sample_at`: canonical quartz kappa = 22.641509433962263 m2/kg;
  published coefficient 584.5610961914062 m^-1 versus independent kernel sum
  584.5610957580724 m^-1 for its explicit synthetic concentration fixture.
  Sampling leaves simulation state unchanged; empty particles publish zero.
- No GPU shader changes were made for the catalog extension. The prior native
  extinction GPU receipt remains separate; no new GPU or live-deploy result is
  implied. The new export test first failed on an incorrect expected warning
  substring; its assertion was corrected to the actual explicit warning, then
  both export tests passed.

Commands:

```text
python -m pytest tests/test_material_medium.py tests/test_canonical_material_documents.py tests/test_abstract_ui_material_catalog.py tests/test_studio_volume_extinction.py tests/test_maxwell_patch_material_contract.py tests/test_material_transport_grid.py tests/test_studio_material_authoring.py tests/test_studio_conductor_surface.py -q --tb=short
python -m pytest ../turing/tests/test_meowdoku_material_library_export.py -q --tb=short
python tests/studio_sph_extinction_check.py
```
