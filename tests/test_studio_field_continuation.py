from tests.test_studio_camera_intervals import canonical
import pytest
from camera_designer.studio_field_continuation import separate_field_continuation
from camera_designer.studio_camera_intervals import observe_camera_continuation


def test_default_source_exact_identity():
    source = canonical()
    assert separate_field_continuation(source) is source


def test_independent_paths_reuse_exact_laws_and_rng_calls():
    source = canonical()
    result = separate_field_continuation(source, enabled=True)
    assert result.count('rand01(rng)') == source.count('rand01(rng)')
    assert 'studio_band_frequency(studio_field_band):uLayerCentersHz[studio_field_band]' in result
    assert 'ro=studio_field_origin;rd=studio_field_direction;' in result
    assert result.index('for(int studio_field_band=') < result.index('int currentMedium=studio_origin_medium(ro,rd);')
    assert 'throughput[i] = i==studio_field_band?1.0:0.0;' in result
    assert 'throughput[i]*=studio_medium_transmittance(currentMedium,frequency,surf_t);' in result
    assert 'studio_boundary_step(tri,rd,frequency,rand01(rng),currentMedium,factor)' in result
    assert 'float measured = lum[i];' in result
    assert result.count('float lum[8];') == 1
    assert result.count('int currentMedium=studio_origin_medium') == 1
    assert 'for(int studio_field_band=0;studio_field_band<(uFixed3Transport!=0?3:nl)' in result


def test_observer_composes_and_reports_actual_active_band():
    source = separate_field_continuation(canonical(), enabled=True)
    result = observe_camera_continuation(source, enabled=True)
    assert 'currentMedium,(uFixed3Transport!=0?3:nl),studio_field_band,sci_first_hit,throughput)' in result
    assert 'sci_words[o+40u]=active_band<0?' in result
    assert 'studio_band_frequency(max(active_band,0))' in result
    with pytest.raises(ValueError):
        separate_field_continuation(source, enabled=True)


@pytest.mark.parametrize('regions', [False, True])
@pytest.mark.parametrize('opaque_pbr', [False, True])
def test_actual_specialized_sensor_compiles_offline(tmp_path, regions, opaque_pbr):
    import shutil
    import subprocess
    from camera_designer.spectral_sensor_basis import FIXED3_SPECTRAL_GLSL
    from camera_designer.studio_diffuse_surface import STUDIO_DIFFUSE_GLSL
    from camera_designer.studio_spectral_reflector import STUDIO_REFLECTOR_GLSL
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    from camera_designer.studio_sphere_regions import with_sphere_regions
    from camera_designer.studio_region_gpu import with_region_media, with_region_spheres
    from camera_designer.sensor_history import SENSOR_HISTORY_GLSL
    from camera_designer.studio_sensor_evidence import SENSOR_EVIDENCE_GLSL
    import mat_flags
    names = ['MAT_FLAG_SPECTRAL_REFLECTOR', 'MAT_FLAG_ROUGH_CONDUCTOR', 'MAT_FLAG_NATIVE_CONDUCTOR_NK']
    flags = ''.join(f'#define {name} {getattr(mat_flags,name)}u\n' for name in names)
    flags += f'#define STUDIO_NATIVE_PBR_BINDING {mat_flags.BINDING_STUDIO_NATIVE_PBR}\n'
    source = canonical().replace('__FIXED3_SPECTRAL_HELPERS__', FIXED3_SPECTRAL_GLSL)
    source = source.replace('__STUDIO_BOUNDARY_HELPERS__', STUDIO_DIFFUSE_GLSL + flags + STUDIO_REFLECTOR_GLSL + STUDIO_BOUNDARY_GLSL)
    source = with_region_spheres(with_sphere_regions(with_region_media(source))) if regions else with_sphere_regions(source)
    source = source.replace('__SENSOR_HISTORY_HELPERS__', SENSOR_HISTORY_GLSL).replace('__SENSOR_EVIDENCE_HELPERS__', SENSOR_EVIDENCE_GLSL)
    if opaque_pbr:
        from camera_designer.studio_opaque_pbr import with_opaque_pbr
        source = with_opaque_pbr(source, camera=True)
    source = observe_camera_continuation(separate_field_continuation(source, enabled=True), enabled=True)
    from camera_designer.studio_camera_responses import copy_camera_responses
    from camera_designer.studio_scalar_visits import with_scalar_visit_queries
    source = with_scalar_visit_queries(source)
    source = copy_camera_responses(source, enabled=True)
    path = tmp_path / 'actual_sensor.comp'
    path.write_text(source, encoding='utf-8')
    compiler = shutil.which('glslangValidator')
    assert compiler
    result = subprocess.run([compiler, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
