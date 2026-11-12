"""CPU equations and offline assembly only, not device/link/transport proof."""
from pathlib import Path
import runpy
import shutil
import subprocess

import numpy as np
import pytest

from camera_designer.studio_directional_field import recording_shader, ROOT_ESTIMATOR
from camera_designer.studio_directional_media import with_internal_medium_recording
from camera_designer.studio_region_gpu import with_region_media
from camera_designer.studio_volume_extinction import GLSL, transport_shader

HERE = Path(__file__).parent
ROOT = HERE.parent


@pytest.mark.parametrize('region', [False, True])
@pytest.mark.parametrize('root_sweep', [False, True])
def test_joint_composer_keeps_actual_transport_and_entry_packets(region, root_sweep):
    source = runpy.run_path(str(HERE / 'test_studio_directional_media.py'))['canonical_source']()
    if region:
        source = with_region_media(source)
    options = dict(estimator=ROOT_ESTIMATOR, grid_size=64, independent_bounds=True) if root_sweep else {}
    source = transport_shader(recording_shader(source, **options))
    result = with_internal_medium_recording(source)
    assert GLSL in result  # actual transport helper is byte-for-byte identical
    assert result.count('packet.z*=studio_volume_transmit(ro,hp);') == 2
    assert result.count('packet.z*=studio_medium_transmittance(currentMedium,packet.x,best);') == 2
    assert result.count('rand01(rng)') == source.count('rand01(rng)')
    hook = 'studio_record_segment(ro,hp,packet,currentMedium);\n        packet.z*=studio_medium_transmittance(currentMedium,packet.x,best);'
    assert result.count(hook) == 2  # before BOTH original endpoint updates
    assert result.count('float sigma=material_sigma+studio_sigma_t[') == 1
    assert 'entry.x*within.y' in result
    assert 'if(medium < -1)return;' in result
    assert 'if(any(isnan(entry)) || any(isnan(within)) || any(isinf(entry)) || any(isinf(within)))' in result
    assert '(length_m==0.0 && any(notEqual(delta,vec3(0.0)))))' in result
    helper = result.split('vec2 studio_record_volume_path(', 1)[1].split('void studio_record_segment(', 1)[0]
    assert helper.index('any(isnan(delta))') < helper.index('if(!(meters>0.0))')
    assert 'isinf(meters)' in helper and '(meters==0.0 && any(notEqual(delta,vec3(0.0))))' in helper
    assert 'Incomplete traversal is unknown, never a vacuum remainder.' in result
    assert 'studio_record_medium_integral(material_sigma,0.0,meters-last)' in result
    assert source.count('studio_record_volume_path(') == 0


def test_joint_composer_rejects_changed_grid_and_missing_observer_specialization():
    source = runpy.run_path(str(HERE / 'test_studio_directional_media.py'))['canonical_source']()
    recorded = recording_shader(source)
    supplied = transport_shader(recorded)
    with pytest.raises(ValueError, match='exact installed ExtinctionGrid'):
        with_internal_medium_recording(supplied.replace('float sigma=studio_sigma_t[', 'float sigma=2.0*studio_sigma_t['))
    with pytest.raises(ValueError, match='canonical internal-field seam changed'):
        with_internal_medium_recording(recorded + GLSL)
    with pytest.raises(ValueError, match='already installed'):
        with_internal_medium_recording(with_internal_medium_recording(supplied))


@pytest.mark.parametrize('queue', [False, True])
def test_joint_full_native_composition_compiles_offline(tmp_path, queue):
    from camera_designer.studio_scalar_visits import with_scalar_visits
    from camera_designer.studio_scalar_columns import with_scalar_columns
    from camera_designer.studio_forward_responses import copy_forward_responses, with_queued_ray_ancestry
    source = runpy.run_path(str(HERE / 'studio_forward_source_control.py'))['native_control_source'](ROOT, volume_extinction=True)
    if queue:
        source = with_queued_ray_ancestry(source)
    source = with_scalar_columns(with_scalar_visits(source), 3, shared=True)
    source, seams = copy_forward_responses(source, terminal_only=True, shared_scalar_buffer=True,
        precleared_rows=True, queue_ancestry=queue)
    assert sum('studio_volume_transmit' in item['expression'] for item in seams) == 2
    compiler = shutil.which('glslangValidator')
    assert compiler, 'official offline GLSL compiler is required'
    path = tmp_path / 'joint_observer.comp'
    path.write_text(source, encoding='utf-8')
    run = subprocess.run([compiler, str(path)], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr


def test_joint_equation_uses_union_of_recording_and_extinction_intervals():
    oracle = runpy.run_path(str(HERE / 'studio_volume_extinction_gpu_check.py'))['oracle']
    sigma = np.array([[[0., 2., 7.]]])
    bounds = [[0., -1., -1.], [3., 1., 1.]]
    a, b = np.array([-1., 0., 0.]), np.array([4., 0., 0.])
    # These are recorder-cell boundaries, intentionally not grid boundaries.
    cuts = [0., .3, 1.2, 2.8, 3.1, 4.4, 5.]
    for material in (0., 1e-8, .7, 2302.585092994):
        expected = oracle(sigma, bounds, a, b, material_sigma=material)
        pieces = 0.
        for first, last in zip(cuts, cuts[1:]):
            entry = oracle(sigma, bounds, a, a + [first, 0, 0], material_sigma=material)
            within = oracle(sigma, bounds, a + [first, 0, 0], a + [last, 0, 0], material_sigma=material)
            pieces += entry[0] * within[1]
        assert pieces == pytest.approx(expected[1], rel=1e-12, abs=1e-15)
        assert expected[0] == pytest.approx(np.exp(-5*material-9), rel=1e-12)
    # Multiplying independent integrals is dimensionally and numerically wrong.
    material = .7
    joint = oracle(sigma, bounds, a, b, material_sigma=material)[1]
    wrong = oracle(sigma, bounds, a, b)[1] * (-np.expm1(-5*material)/material)
    assert not np.isclose(joint, wrong)


def test_uniform_zero_grid_and_material_limits_include_outside_portions():
    oracle = runpy.run_path(str(HERE / 'studio_volume_extinction_gpu_check.py'))['oracle']
    bounds = [[-1.]*3, [1.]*3]
    for a, b in (([-2., 0., 0.], [2., 0., 0.]), ([2., 0., 0.], [-2., 0., 0.]),
                 ([-2., 1., 0.], [2., 1., 0.]), ([-2.]*3, [2.]*3)):
        distance = np.linalg.norm(np.array(b)-a)
        for material in (0., 1e-8, 2., 1e5):
            actual = oracle(np.zeros((2, 3, 4)), bounds, a, b, material_sigma=material)
            expected = [np.exp(-material*distance), -np.expm1(-material*distance)/material if material else distance]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_runtime_admission_stays_closed_until_delivery_and_device_gates():
    # Preparing an observer does not silently enable an unvalidated host path.
    for name in ('demo_pluck_gl.py', 'camera_designer/marble_preview_worker.py'):
        assert 'internal directional recording with ExtinctionGrid requires joint attenuation integration' in (ROOT / name).read_text(encoding='utf-8')


def test_prebuilt_applies_internal_observer_after_grid_source_selection():
    """Exercise the real host selection statements without importing GL."""
    import ast
    tree = ast.parse((ROOT / 'demo_pluck_gl.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gpu_ray_field_prebuilt')
    internal = [n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'record_directional and light_field_internal_media']
    assert len(internal) == 1
    replacements = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'shader_source' for t in n.targets)
        and isinstance(n.value, ast.Name) and n.value.id == 'volume_shader_source']
    assert len(replacements) == 1
    assert internal[0].lineno > replacements[0].lineno
    source = runpy.run_path(str(HERE / 'test_studio_directional_media.py'))['canonical_source']()
    source = transport_shader(recording_shader(source))
    scope = dict(record_directional=True, light_field_internal_media=True, shader_source=source)
    exec(compile(ast.Module(body=internal, type_ignores=[]), '<actual-prebuilt-observer>', 'exec'), scope)
    assert 'float sigma=material_sigma+studio_sigma_t[' in scope['shader_source']
    assert scope['shader_source'].count('studio_record_segment(ro,hp,packet,currentMedium);') == 2
