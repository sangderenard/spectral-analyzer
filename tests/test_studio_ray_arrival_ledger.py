import numpy as np
import pytest
import ast
import re
from pathlib import Path

from camera_designer import studio_ray_arrival_ledger as ledger


def canonical_source(name):
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    from camera_designer.studio_region_gpu import with_region_media
    tree=ast.parse((Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                return with_region_media(node.value.value.replace('__STUDIO_BOUNDARY_HELPERS__',STUDIO_BOUNDARY_GLSL))
    raise AssertionError(name)


def forward_source():
    from camera_designer.studio_directional_field import recording_shader, ROOT_ESTIMATOR
    from camera_designer.studio_directional_media import with_internal_medium_recording
    return with_internal_medium_recording(recording_shader(canonical_source('_GPU_RAY_FIELD_CS'),
        grid_size=64, estimator=ROOT_ESTIMATOR, independent_bounds=True))


def test_forward_composition_is_strict_and_preserves_transport_operations():
    source = forward_source()
    observed = ledger.forward_shader(source)
    for operation in ['rand01(rng)', 'packet.z*=studio_medium_transmittance(currentMedium,packet.x,best);',
                      'splat_segment_multiscale(ro, hp, packet, path_len);',
                      'studio_record_track(q,d,packet,studio_record_medium_integral(studio_record_sigma,cursor,next));']:
        assert observed.count(operation) == source.count(operation)
    assert 'sal_bounce=bounce;sal_begin(ro,rd,best,hit' in observed
    assert 'sal_finish(3,vec3(packet.z),rd,currentMedium);' in observed
    with pytest.raises(ValueError):ledger.forward_shader(observed)
    with pytest.raises(ValueError):ledger.forward_shader(source.replace('if (hit < 0) break;', 'if(hit<0)break;'))


def test_camera_composition_does_not_deposit_forward_energy():
    source = canonical_source('_GPU_SENSOR_CS')
    observed = ledger.camera_shader(source)
    assert observed.count('rand01(rng)') == source.count('rand01(rng)')
    assert 'studio_directional[' not in observed
    assert 'sal_root=0xffffffffu' in observed
    assert 'studio_record_segment' not in observed
    assert 'sal_bounce=bounce;sal_begin(ro,rd,hit_id>=0?surf_t:0.0,hit_id' in observed


def test_explicit_truncation_and_missing_endpoint(tmp_path):
    head = np.zeros(256, '<u4');head[3] = 1
    row = np.zeros(64, '<u4');row[7] = 0xffffffff;row[10] = 1
    event = ledger.event_dict(row)
    assert event['b_m'] is None and event['roi_interval_m'] is None
    receipt = ledger.save_capture(tmp_path, (head,np.zeros((0,64),'<u4'),row[None]), metadata={})
    assert not receipt['roi_records_complete']
    assert not receipt['all_event_records_complete']
    assert receipt['counts']['roi_records_dropped'] == 1


@pytest.mark.parametrize('value',[-1,1.5,1048577,True])
def test_capacity_is_bounded(value):
    with pytest.raises(ValueError):ledger._header(value,0)


def test_pending_counter_binding_and_word_match_canonical_shader():
    source=canonical_source('_GPU_RAY_FIELD_CS')
    match=re.search(r'layout\(std430, binding = (\d+)\) buffer\s+CounterBuf\s*\{([^}]+)\}',source)
    assert match and int(match[1])==ledger.COUNTER_BUFFER_BINDING
    names=re.findall(r'uint\s+(\w+)\s*;',match[2])
    assert names==['seg_count','ray_count','hit_count','record_count','pending_count']
    assert names.index('pending_count')==ledger.PENDING_COUNT_WORD
    pending=re.search(r'layout\(std430, binding = (\d+)\) buffer\s+PendingRayBuf\s*\{([^}]+)\}',source)
    assert pending and int(pending[1])==ledger.PENDING_RAY_BUFFER_BINDING
    assert pending[2].strip()=='PendingRay pending_rays[];'
