import runpy
from pathlib import Path
canonical_source=runpy.run_path(str(Path(__file__).with_name('test_studio_ray_arrival_ledger.py')))['canonical_source']
from camera_designer.studio_ray_arrival_ledger import camera_shader
from camera_designer.studio_nee_connection_ledger import connection_shader
import pytest


def test_nee_observer_preserves_random_and_visibility_queries():
    before=camera_shader(canonical_source('_GPU_SENSOR_CS'))
    after=connection_shader(before)
    for operation in ('rand01(rng)','nearest_hit(p+wi*1e-6,wi,shadow_t)',
                      'nearest_hit(p + wi * 1e-4, wi, shadow_t)','direct_bands=src.dir_kind.xyz*weight;'):
        assert after.count(operation)==before.count(operation)
    assert 'binding=22' in after
    assert 'snl_end(3,vec3(0.0),0.0);return 0.0;' in after
    assert 'snl_beta=vec3(throughput[0],throughput[1],throughput[2])' in after
    with pytest.raises(ValueError):connection_shader(after)


def test_nee_requires_actual_known_visibility_seam():
    source=camera_shader(canonical_source('_GPU_SENSOR_CS'))
    with pytest.raises(ValueError):connection_shader(source.replace('dist-2e-6','dist-3e-6'))
