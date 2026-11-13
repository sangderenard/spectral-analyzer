"""CPU-only v3 store/publisher fixtures and cross-language validation."""
import copy
import hashlib
import json
import numpy as np
import pytest
from camera_designer.studio_forward_responses import FORWARD_ANCESTRY_CONTRACT
from camera_designer.studio_scene_responses import SceneResponseStore, validate_scene_response_snapshot
from camera_designer.studio_response_lineage import NativeResponseLineage
from camera_designer.studio_scene_response_publish import SceneResponsePublisher


def fixture():
    lineage=NativeResponseLineage(0)
    store=SceneResponseStore(scene_revision='a'*64,memory_budget_bytes=1048576,lineage_provider=lineage.snapshot)
    p=dict(dispatch_id=7,population_index=2,ray_offset=10,ray_count=3,total_rays=13,bounce_limit=4,seed=17,root_sampling={})
    meta=dict(version='evaluated-forward-responses-v2',forward_ancestry_contract=copy.deepcopy(FORWARD_ANCESTRY_CONTRACT),
        scene_revision='a'*64,source_sha256='b'*64,transport_scene_sha256='c'*64,stream_id='queue',sequence=0,
        source_population_sha256='b'*64,population_id=2,dispatch=dict(native_queue=dict(version='native-queued-ancestry-v1',
            generation=7,logical_ray_base=20,capacity=4,source_root_count=5,parent_dispatch=p)))
    rows=np.zeros((2,128),'<u4');rows[:,0]=5;rows[:,1]=[1,2];rows[:,2]=[10,0];rows[:,3]=[0,1]
    rows[:,5]=0xffffffff;rows[:,30]=1;rows[:,112:114]=0xffffffff
    rows[1,101:104]=[7,11,1];rows[1,125]=np.float32(.5).view(np.uint32);rows[1,126:128]=[1,4]
    rows[:,120]=0x7fc12345
    return store,meta,rows


def test_exact_v3_and_publisher(tmp_path):
    s,m,r=fixture();s.consume_forward(m,r);manifest,payload=s.snapshot()
    assert manifest['schema']=='studio-scene-responses-v3'
    assert manifest['sections'][2]['stride']==1024
    assert manifest['forward_ancestry_contract']==FORWARD_ANCESTRY_CONTRACT
    assert manifest['contexts'][0]['dispatch']==m['dispatch']
    np.testing.assert_array_equal(np.frombuffer(payload,'<u4',offset=manifest['sections'][1]['offset']).reshape(-1,128),r)
    assert validate_scene_response_snapshot(manifest,payload)
    receipt=SceneResponsePublisher(tmp_path,'d'*64).publish(s)
    assert receipt['schema']==receipt['metadata']['schema']=='studio-scene-responses-v3'


@pytest.mark.parametrize('offset,value',[(2,4),(3,2),(101,6),(102,13),(103,3),(125,0x7fc00000),(125,0x40000000),(126,0),(127,0xffffffff)])
def test_invalid_parent_rows_do_not_mutate(offset,value):
    s,m,r=fixture();r[1,offset]=value
    with pytest.raises(ValueError):s.consume_forward(m,r)
    assert s.retained_entries==0 and not s.forward_ancestry


@pytest.mark.parametrize('edit',[lambda m:m['forward_ancestry_contract'].update(row_words=128.0),
    lambda m:m['forward_ancestry_contract']['enqueue_causes'].update(extra=3),
    lambda m:m['dispatch']['native_queue'].update(source_root_count=0),
    lambda m:m['dispatch']['native_queue'].update(generation=8),
    lambda m:m['dispatch']['native_queue']['parent_dispatch'].update(root_sampling=[]),
    lambda m:m.update(scene_revision='f'*64)])
def test_bad_metadata_does_not_mutate(edit):
    s,m,r=fixture();edit(m)
    with pytest.raises(ValueError):s.consume_forward(m,r)
    assert not s.forward_ancestry and s.retained_entries==0


def test_versions_never_relabel_retained_rows():
    s,m,r=fixture();legacy=copy.deepcopy(m);legacy['version']='evaluated-forward-responses-v1';legacy.pop('forward_ancestry_contract')
    s.consume_forward(legacy,r[:1]);m['sequence']=1
    with pytest.raises(ValueError):s.consume_forward(m,r)
    assert s.snapshot()[0]['schema']=='studio-scene-responses-v1'
    s,m,r=fixture();s.consume_forward(m,r);legacy['sequence']=1
    with pytest.raises(ValueError):s.consume_forward(legacy,r[:1])
    assert s.snapshot()[0]['schema']=='studio-scene-responses-v3'


def test_snapshot_revalidates_ancestry():
    s,m,r=fixture();s.consume_forward(m,r);manifest,payload=s.snapshot()
    changed=bytearray(payload);words=np.frombuffer(changed,'<u4');words[manifest['sections'][1]['offset']//4+128+101]=99
    manifest['payload_sha256']=hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError):validate_scene_response_snapshot(manifest,changed)


def test_zero_weight_stays_unavailable_attribute():
    s,m,r=fixture();r[1,125]=0;s.consume_forward(m,r)
    assert s.snapshot()[0]['application_ready'] is False

def test_v3_accepts_actual_camera_pairs_without_relabeling():
    s,m,r=fixture();s.consume_forward(m,r)
    a=np.zeros((1,128),'<u4');b=np.zeros((1,128),'<u4')
    a[0,4]=0xffffffff;a[0,6]=3;a[0,11]=0x7fc12345;a[0,40]=7;a[0,126]=0xffffffff
    cm=dict(version='native-camera-intervals-v1',response_schema='native-camera-response-copies-v1',
        complete_dispatch_observation=True,application_ready=False,row_count=1,scene_revision='a'*64,
        transport_scene_sha256='c'*64,source_sha256='b'*64,stream_id='camera',sequence=0)
    s.consume_camera_pairs(cm,a,b)
    manifest,payload=s.snapshot()
    assert manifest['schema']=='studio-scene-responses-v3' and manifest['sections'][2]['count']==1
    assert validate_scene_response_snapshot(manifest,payload)
    np.testing.assert_array_equal(np.frombuffer(payload,'<u4',offset=manifest['sections'][2]['offset']),np.concatenate([a[0],b[0]]))


def test_parent_dispatch_and_contract_are_immutable_retained_copies():
    s,m,r=fixture();s.consume_forward(m,r)
    before,payload=s.snapshot()
    m['dispatch']['native_queue']['generation']=999
    m['dispatch']['native_queue']['parent_dispatch']['root_sampling']['changed']=True
    m['forward_ancestry_contract']['enqueue_causes']['edge_diffraction']=99
    after,again=s.snapshot()
    assert after==before and again==payload



def test_zero_enqueue_capacity_is_valid_only_for_forward_rows():
    s,m,r=fixture();q=m['dispatch']['native_queue'];q['capacity']=0;q['parent_dispatch']['bounce_limit']=1
    s.consume_forward(m,r[:1]);assert s.snapshot()[0]['schema']=='studio-scene-responses-v3'
    s,m,r=fixture();m['dispatch']['native_queue']['capacity']=0
    with pytest.raises(ValueError):s.consume_forward(m,r[1:])


@pytest.mark.parametrize('edit',[
    lambda q:q['parent_dispatch'].update(ray_count=0),
    lambda q:q['parent_dispatch'].update(bounce_limit=0),
    lambda q:q['parent_dispatch'].update(total_rays=12),
    lambda q:q.update(logical_ray_base=0xffffffff)])
def test_parent_dispatch_bounds_are_authentic(edit):
    s,m,r=fixture();edit(m['dispatch']['native_queue'])
    with pytest.raises(ValueError):s.consume_forward(m,r[:1])
    assert s.retained_entries==0
