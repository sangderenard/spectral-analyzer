import pytest
from tests.test_studio_camera_intervals import canonical
from camera_designer.studio_camera_intervals import observe_camera_continuation
@pytest.mark.parametrize('regions', [False, True])
@pytest.mark.parametrize('opaque_pbr', [False, True])
def test_completed_original_sensor_compiles_offline(tmp_path, regions, opaque_pbr):
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
    source = observe_camera_continuation(source, enabled=True)
    from camera_designer.studio_camera_responses import copy_camera_responses
    from camera_designer.studio_scalar_visits import with_scalar_visit_queries
    source = with_scalar_visit_queries(source)
    source = copy_camera_responses(source, enabled=True)
    from camera_designer.studio_completed_samples import with_completed_camera_samples
    source = with_completed_camera_samples(source)
    path = tmp_path / 'actual_sensor.comp'
    path.write_text(source, encoding='utf-8')
    compiler = shutil.which('glslangValidator')
    assert compiler
    result = subprocess.run([compiler, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_completed_copy_has_actual_presence_and_original_film_coefficients():
    from camera_designer.studio_camera_responses import copy_camera_responses
    from camera_designer.studio_completed_samples import with_completed_camera_samples
    source=copy_camera_responses(observe_camera_continuation(canonical(),enabled=True),enabled=True)
    copied=with_completed_camera_samples(source)
    assert copied.count('rand01(rng)')==source.count('rand01(rng)')
    assert 'scs_words[o+11u]=scs_position_written?1u:0u;' in copied
    assert 'if(bounce>uMaxBounces && scs_termination==0u)scs_termination=8u;' in copied
    assert 'if(!scs_has_recipient)for(int j=0;j<8;++j)scs_volume_known[j]=0u;' in copied
    assert 'const bool uCameraIntervalsEnabled=false;' in copied
    assert 'buffer StudioCameraIntervals' not in copied and 'buffer StudioCameraResponses' not in copied
    assert 'scs_nee' not in copied
    assert 'sci_nee_shadow< -1?7u:6u' in copied
    assert 'scs_words[scs_sample_row+120u+uint(i)]=floatBitsToUint(max(measured,0.0))' in copied
    assert copied.count('exp(-0.5 * oct * oct / (width * width))')==source.count('exp(-0.5 * oct * oct / (width * width))')
    assert 'scs_measured_surface+=scs_surface[band]*scs_original_filter;' in copied


def test_completed_dispatch_rejects_before_gl_without_changing_rows():
    from camera_designer.studio_sensor_evidence import StudioResponseEvidence
    owner=StudioResponseEvidence(memory_budget_bytes=10000,reserved_bytes=1000,scene_revision='s',camera_revision='c',consume=lambda *args:None,completed_samples=True)
    with pytest.raises(MemoryError,match='original completed-sample dispatch'):
        owner.prepare(width=128,rows=16,samples=1,path_bands=1,max_bounces=8)
    assert owner.buffer==0 and owner.response_buffer==0
    with pytest.raises(ValueError,match='shared-band'):
        owner.prepare(width=1,rows=1,samples=1,path_bands=3,max_bounces=8)


def test_old_observer_only_constructor_does_not_dereference_absent_information():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8')
    assert 'if light_field_information is not None and light_field_information.completed_samples:' in source
    assert 'if light_field_information.completed_samples:' not in source


def test_native_scalar_reset_callback_shares_exact_canonical_reset_epoch():
    from camera_designer.studio_sensor_evidence import StudioResponseEvidence
    owner=StudioResponseEvidence(memory_budget_bytes=10000,reserved_bytes=0,scene_revision='s',camera_revision='c',consume=lambda *args:None)
    calls=[];owner.native_scalar_reset=lambda:calls.append(owner.identity['field_generation'])
    owner.note_field_reset();owner.note_field_reset()
    assert calls==[1,2] and owner.buffer==0
