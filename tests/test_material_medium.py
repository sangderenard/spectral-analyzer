import copy
import math
import numpy as np
import pytest
from canonical_material_documents import read_material_document, native_payload, native_volume_payload, restore_material_document
from material_db import MaterialDatabase
from material_medium import validate_medium_profile, mass_extinction_from_medium, medium_profile_digest
from spectral_material import Material

VAPORS = ('water_vapor', 'ethanol_vapor', 'acetone_vapor')
DUSTS = ('quartz_dust_coarse', 'calcite_dust_coarse', 'halite_dust_coarse')
WAVELENGTHS = [650e-9, 550e-9, 450e-9]


def profile(name):
    return read_material_document(name).definition['medium']


@pytest.mark.parametrize('name', VAPORS + DUSTS)
def test_canonical_source_database_and_material_object_keep_medium(name):
    document = read_material_document(name)
    assert restore_material_document(document.snapshot()).source_bytes == document.source_bytes
    payload = native_payload(document)
    assert 'medium' not in payload.uncompiled_fields  # recognized, not an unknown field
    assert any(field.startswith('medium (cold') for field in payload.uncompiled_fields)
    database = MaterialDatabase()
    index = database.register(name, payload.payload)
    expected = profile(name)
    assert database.medium_profile(name) == expected
    payload.payload['medium']['formula'] = 'caller mutation'
    result = database.medium_profile(name)
    result['optics']['model'] = 'caller mutation'
    assert database.medium_profile(name) == expected
    assert database.build_tensors()['n_bands'][index] == 0
    material = Material.from_dict(document.definition, name=name)
    assert material.to_dict()['medium'] == expected
    assert not material.effective_bands()  # no fabricated PBR-derived spectrum
    database.register(name + '-object', material)
    assert database.build_tensors()['n_bands'][-1] == 0
    with pytest.raises(ValueError, match='volume-field'):
        material.evaluate(5e14)
    with pytest.raises(ValueError, match='surface tracer'):
        material.to_tracer_bands(np.array([5e14]))
    with pytest.raises(ValueError, match='extinction-field'):
        native_volume_payload(document)


def test_reference_units_and_unknown_optics_are_explicit():
    for name, mass in zip(VAPORS, [.0180153, .0460684, .0580791]):
        p = profile(name)
        assert p['phase'] == 'molecular_gas'
        assert p['molar_mass_kg_mol'] == mass
        assert p['particle_material_density_kg_m3'] is None
        with pytest.raises(ValueError, match='unavailable'):
            mass_extinction_from_medium(p, wavelengths_m=WAVELENGTHS, allow_approximation=True)
    for name, density in zip(DUSTS, [2650., 2710., 2168.]):
        p = profile(name)
        assert p['particle_material_density_kg_m3'] == density
        assert p['optics']['particle_radius_m'] == 25e-6
        for unknown in ('single_scattering_albedo', 'asymmetry_g', 'phase_function', 'mass_absorption_m2_kg', 'mass_scattering_m2_kg', 'refractive_index'):
            assert p['optics'][unknown] is None


@pytest.mark.parametrize('name', DUSTS)
def test_extinction_is_cross_section_over_mass_and_requires_approximation_opt_in(name):
    p = profile(name)
    radius = p['optics']['particle_radius_m']
    mass = p['particle_material_density_kg_m3'] * (4 / 3) * math.pi * radius**3
    cross_section = 2 * math.pi * radius**2
    result = mass_extinction_from_medium(p, wavelengths_m=WAVELENGTHS, allow_approximation=True)
    assert result == pytest.approx(cross_section / mass)
    with pytest.raises(ValueError, match='allow_approximation'):
        mass_extinction_from_medium(p, wavelengths_m=WAVELENGTHS)
    with pytest.raises(ValueError, match='outside'):
        mass_extinction_from_medium(p, wavelengths_m=[10e-6], allow_approximation=True)
    with pytest.raises(ValueError, match='declare all'):
        mass_extinction_from_medium(p, wavelengths_m=[], allow_approximation=True)
    # Neither a surface swatch nor debug sprite opacity participates in this law.
    p['visualization'] = {'sprite_opacity': .999, 'particle_size_pixels': 100}
    assert mass_extinction_from_medium(p, wavelengths_m=WAVELENGTHS, allow_approximation=True) == result


@pytest.mark.parametrize('path,value', [
    (('phase',), 'fog'),
    (('particle_material_density_kg_m3',), -1),
    (('particle_material_density_kg_m3',), float('inf')),
    (('optics', 'particle_radius_m'), 0),
    (('optics', 'particle_radius_m'), True),
    (('optics', 'extinction_efficiency'), 9),
    (('optics', 'wavelength_interval_m'), [700e-9, 400e-9]),
    (('optics', 'single_scattering_albedo'), 1.2),
    (('optics', 'asymmetry_g'), -1.2),
    (('optics', 'mass_extinction_m2_kg'), 100),
    (('provenance',), []),
])
def test_invalid_physical_declarations_fail_before_registry_mutation(path, value):
    p = profile('quartz_dust_coarse')
    bad = copy.deepcopy(p)
    target = bad
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    database = MaterialDatabase()
    database.register('reference', {'medium': p})
    with pytest.raises(ValueError):
        database.register('reference', {'medium': bad})
    assert database.names() == ['reference']
    assert database.medium_profile('reference') == p


def test_gas_is_not_spherical_dust_and_unknown_does_not_mean_zero():
    p = profile('water_vapor')
    p['optics']['mass_extinction_m2_kg'] = 0
    with pytest.raises(ValueError, match='unparameterized'):
        validate_medium_profile(p)
    p = profile('quartz_dust_coarse')
    p['phase'] = 'molecular_gas'
    with pytest.raises(ValueError, match='gas'):
        validate_medium_profile(p)


def test_central_catalog_to_actual_extinction_field_uses_species_density():
    from camera_designer.studio_volume_extinction import ExtinctionGrid
    database = MaterialDatabase()
    for name in VAPORS + DUSTS:
        database.register(name, native_payload(read_material_document(name)).payload)
    density = 1e-6  # explicit kg of dust / m3 of carrier, not grain material density
    sampler = lambda points: {'rho': np.full(len(points), density)}
    grid = ExtinctionGrid.from_material_density_sampler(database, 'quartz_dust_coarse', sampler,
        [[0, 0, 0], [1, 1, 1]], [2, 3, 4], wavelengths_m=WAVELENGTHS, allow_approximation=True)
    expected = density * 3 * 2 / (4 * 2650 * 25e-6)
    np.testing.assert_allclose(grid.sigma_t_m_inv, expected)
    provenance = grid.metadata()['material_provenance']
    assert provenance['profile_sha256'] == medium_profile_digest(database.medium_profile('quartz_dust_coarse'))
    assert provenance['material_name'] == 'quartz_dust_coarse'
    assert provenance['wavelengths_m'] == WAVELENGTHS
    provenance['wavelengths_m'][0] = 0
    assert grid.metadata()['material_provenance']['wavelengths_m'] == WAVELENGTHS
    empty = ExtinctionGrid.from_material_density_sampler(database, 'quartz_dust_coarse',
        lambda points: {'rho': np.zeros(len(points))}, [[0]*3, [1]*3], [2]*3,
        wavelengths_m=WAVELENGTHS, allow_approximation=True)
    assert not empty.sigma_t_m_inv.any()
    with pytest.raises(ValueError, match='unavailable'):
        ExtinctionGrid.from_material_density_sampler(database, 'water_vapor', sampler,
            [[0]*3, [1]*3], [2]*3, wavelengths_m=WAVELENGTHS, allow_approximation=True)
    database.register('ordinary', {'albedo': [1, 1, 1]})
    assert database.medium_profile('ordinary') is None
    with pytest.raises(ValueError, match='no declared'):
        ExtinctionGrid.from_material_density_sampler(database, 'ordinary', sampler,
            [[0]*3, [1]*3], [2]*3, wavelengths_m=WAVELENGTHS, allow_approximation=True)


def test_existing_catalog_exposes_new_profiles_without_a_parallel_database():
    from abstract_ui_material_catalog import build_material_catalog
    catalog = build_material_catalog()
    records = {record['identity']: record for record in catalog['records']}
    for name in VAPORS + DUSTS:
        assert records[name]['medium'] == profile(name)
        assert records[name]['surface_role'] == 'editor-placeholder-only'
        assert records[name]['source'] == f'configs/materials/{name}.yaml'
