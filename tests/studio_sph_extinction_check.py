"""Cross-repository check using Turing's real SPH reconstruction, without GL."""
from pathlib import Path
import json
import sys
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(root.parent / 'turing'))
    from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams
    from camera_designer.studio_volume_extinction import ExtinctionGrid
    positions = np.array([[.025, 0, 0], [.075, 0, 0]])
    fluid = DiscreteFluid(positions, None, None, None, FluidParams(particle_mass=.01, smoothing_length=.1))
    before = {key: getattr(fluid, key).copy() for key in ('p', 'v', 'm', 'rho', 'P', 'T', 'S')}
    # Explicit synthetic test calibration. This is NOT a water/material preset.
    coefficient = .02
    grid = ExtinctionGrid.from_density_sampler(fluid.sample_at, [[0, -.05, -.05], [.1, .05, .05]],
                                              [1, 1, 1], mass_extinction_m2_kg=coefficient)
    expected = np.sum(fluid.m * fluid.kernel.W(np.linalg.norm(positions - [.05, 0, 0], axis=1))) * coefficient
    np.testing.assert_allclose(grid.sigma_t_m_inv[0, 0, 0], expected, rtol=2e-7)
    for key, values in before.items():
        np.testing.assert_array_equal(getattr(fluid, key), values)
    empty = DiscreteFluid(np.zeros((0, 3)), None, None, None, FluidParams())
    assert not ExtinctionGrid.from_density_sampler(empty.sample_at, [[0]*3, [1]*3], [2]*3,
                                                  mass_extinction_m2_kg=coefficient).sigma_t_m_inv.any()
    # Use the real canonical profile with the same real SPH reconstruction.
    # This fixture's particles now represent suspended quartz mass only; the
    # kernel sum is mass per carrier volume, not the density of solid quartz.
    from canonical_material_documents import native_payload, read_material_document
    from material_db import MaterialDatabase
    from material_medium import mass_extinction_from_medium
    database = MaterialDatabase()
    material_name = 'quartz_dust_coarse'
    database.register(material_name, native_payload(read_material_document(material_name)).payload)
    wavelengths = [650e-9, 550e-9, 450e-9]
    catalog_coefficient = mass_extinction_from_medium(database.medium_profile(material_name),
        wavelengths_m=wavelengths, allow_approximation=True)
    catalog_grid = ExtinctionGrid.from_material_density_sampler(database, material_name, fluid.sample_at,
        [[0, -.05, -.05], [.1, .05, .05]], [1, 1, 1],
        wavelengths_m=wavelengths, allow_approximation=True)
    expected_catalog = expected / coefficient * catalog_coefficient
    np.testing.assert_allclose(catalog_grid.sigma_t_m_inv[0, 0, 0], expected_catalog, rtol=2e-7)
    for key, values in before.items():
        np.testing.assert_array_equal(getattr(fluid, key), values)
    print(json.dumps(dict(result='PASS', actual_sigma_t_m_inv=float(grid.sigma_t_m_inv[0, 0, 0]),
                          independent_kernel_sum_times_explicit_kappa=float(expected),
                          simulation_state_unchanged=True, empty_particles_zero=True,
                          catalog_material=material_name, catalog_mass_extinction_m2_kg=catalog_coefficient,
                          catalog_sigma_t_m_inv=float(catalog_grid.sigma_t_m_inv[0, 0, 0]),
                          independent_catalog_kernel_sum=float(expected_catalog))))


if __name__ == '__main__':
    main()
