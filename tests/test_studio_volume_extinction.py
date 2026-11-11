import numpy as np
import pytest
from camera_designer.studio_volume_extinction import ExtinctionGrid, SCHEMA, from_payload, transport_shader


def test_explicit_units_layout_copy_and_provenance():
    sigma = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    bounds = np.array([[0, 0, 0], [4, 3, 2]], np.float32)
    grid = ExtinctionGrid(sigma, bounds)
    metadata = grid.metadata()
    assert metadata["dimensions_xyz"] == [4, 3, 2]
    assert metadata["units"] == "m^-1"
    sigma[:] = 0
    bounds[:] = 0
    assert grid.sigma_t_m_inv.sum() == 276
    assert grid.metadata() == metadata
    assert not grid.sigma_t_m_inv.flags.writeable
    assert not grid.bounds_m.flags.writeable
    assert ExtinctionGrid(grid.sigma_t_m_inv * 2, grid.bounds_m).metadata()["sha256"] != metadata["sha256"]


@pytest.mark.parametrize("sigma", [np.ones((1, 2)), np.ones((0, 1, 1)), np.ones((65, 1, 1)),
                                  [[[-1]]], [[[np.nan]]], [[[np.inf]]]])
def test_invalid_coefficients_rejected(sigma):
    with pytest.raises(ValueError):
        ExtinctionGrid(sigma, [[0, 0, 0], [1, 1, 1]])


@pytest.mark.parametrize("bounds", [[[0, 0, 0], [0, 1, 1]], [[0, 0, 0], [-1, 1, 1]],
                                   [[0, 0, 0], [np.inf, 1, 1]], [[0, 0], [1, 1]]])
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        ExtinctionGrid(np.ones((1, 1, 1)), bounds)


def test_absence_is_not_fog_and_partial_declarations_fail():
    assert from_payload({}, {}) is None
    declaration = {"volume_extinction": {"schema": SCHEMA}}
    payload = {"volume_extinction_sigma_t_m_inv": np.ones((2, 3, 4)),
               "volume_extinction_bounds_m": [[0, 0, 0], [1, 1, 1]]}
    assert from_payload(declaration, payload).sigma_t_m_inv.shape == (2, 3, 4)
    for config, data in [({}, payload), (declaration, {}),
                         (declaration, {"volume_extinction_bounds_m": payload["volume_extinction_bounds_m"]}),
                         ({"volume_extinction": {"humidity": 0.8}}, payload)]:
        with pytest.raises(ValueError):
            from_payload(config, data)


def test_density_publication_calls_existing_sampler_at_world_cell_centres():
    calls = []
    def density(points):
        calls.append(points.copy())
        return {"rho": points[:, 0] + 10 * points[:, 1] + 100 * points[:, 2]}
    grid = ExtinctionGrid.from_density_sampler(density, [[0, 0, 0], [4, 3, 2]], [4, 3, 2], mass_extinction_m2_kg=.02)
    expected = np.array([[[x + .5 + 10*(y + .5) + 100*(z + .5) for x in range(4)] for y in range(3)] for z in range(2)])*.02
    np.testing.assert_allclose(grid.sigma_t_m_inv, expected)
    np.testing.assert_array_equal(calls[0][0], [.5, .5, .5])
    np.testing.assert_array_equal(calls[0][-1], [3.5, 2.5, 1.5])
    for coefficient in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            ExtinctionGrid.from_density_sampler(density, [[0]*3, [1]*3], [2]*3, mass_extinction_m2_kg=coefficient)
    with pytest.raises(ValueError):
        ExtinctionGrid.from_density_sampler(lambda points: {"rho": [-1]*len(points)}, [[0]*3, [1]*3], [2]*3, mass_extinction_m2_kg=1)
    with pytest.raises(TypeError):
        ExtinctionGrid.from_density_sampler(density, [[0]*3, [1]*3], [2]*3)


def test_current_canonical_variants_are_specialized_without_mutation():
    import ast
    from pathlib import Path
    from camera_designer.studio_directional_field import recording_shader
    # The shared CI conftest stubs pygame/GL. Read the real shader literals;
    # the GPU gate separately compiles all fully composed normal/region variants.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "demo_pluck_gl.py").read_text(encoding="utf-8"))
    shaders = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    shaders[target.id] = node.value.value
    for name in ("_GPU_RAY_FIELD_CS", "_GPU_SENSOR_CS"):
        original = shaders[name]
        result = transport_shader(original, sensor="SENSOR" in name)
        assert shaders[name] == original
        assert "studio_sigma_t" not in original
        assert "studio_sigma_t" in result
        if "SENSOR" not in name:
            recorded = transport_shader(recording_shader(original))
            assert "studio_volume_transmit(a,a+d*cursor)*studio_volume_path(a+d*cursor,a+d*next).y" in recorded
    with pytest.raises(ValueError):
        transport_shader("#version 430 core\nnot a canonical shader")
