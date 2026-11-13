"""CPU-only delivery of actual paired buffers, with no invented third row."""
import numpy as np
import pytest

from camera_designer.studio_response_factory import SceneResponseConsumer, camera_record_mode, selected_sensor_reservation
from camera_designer.studio_response_lineage import NativeResponseLineage
from camera_designer.studio_scene_responses import SceneResponseStore, CAMERA_PAIR_CONTRACT, validate_scene_response_snapshot


def metadata(sequence=0, count=1):
    return dict(scene_revision='a'*64, camera_revision='camera', stream_id='paired', sequence=sequence,
        source_sha256='b'*64, transport_scene_sha256='c'*64,
        version='native-camera-intervals-v1', response_schema='native-camera-response-copies-v1',
        complete_dispatch_observation=True, application_ready=False, row_count=count,
        shared_band0_path=True, camera_sample_key=['dispatch_identity','invocation_word15','sample_word2'],
        response_weighting='original_path_throughput_not_availability_premultiplied',
        denominator='original_camera_samples_not_response_rows',
        excludes=['NEE sample lineage','untraced paths','field response evaluation'])


def buffers():
    interval=np.zeros((1,128),'<u4');response=np.zeros((1,128),'<u4')
    interval[0,4:8]=[12,4,3,1]
    interval[0,8:11]=np.array([-2,.5,.5],np.float32).view(np.uint32)
    interval[0,11]=np.float32(5).view(np.uint32)
    interval[0,12:15]=np.array([1,0,0],np.float32).view(np.uint32)
    interval[0,15]=17;interval[0,40]=7
    interval[0,48]=0x80000001;interval[0,80]=0x40000002
    interval[0,115]=9;interval[0,124:128]=[1,8,2,12]
    response[0,0]=0;response[0,1]=0x80000000;response[0,2]=0x7fc12345
    response[0,96]=1;response[0,108]=3;response[0,120]=8
    return interval,response


def store():
    lineage=NativeResponseLineage(0)
    return SceneResponseStore(scene_revision='a'*64,memory_budget_bytes=1024*1024,
        lineage_provider=lambda:lineage.snapshot(),bounds=[[0]*3,[1]*3],dimensions=[32]*3)


def test_exact_versioned_pair_without_fake_attachment_or_coverage():
    s=store();a,b=buffers();s.consume_camera_pairs(metadata(),a,b)
    m,p=s.snapshot();section=m['sections'][2]
    assert m['schema']=='studio-scene-responses-v2'
    assert m['families']['3']['codec']=='native-camera-response-pairs-v2'
    assert m['camera_pair_contract']==CAMERA_PAIR_CONTRACT
    assert section['stride']==1024 and section['count']==1 and len(p)==80+1024
    raw=np.frombuffer(p,'<u4',offset=section['offset'])
    np.testing.assert_array_equal(raw,np.concatenate([a[0],b[0]]))
    index=np.frombuffer(p,'<u4',count=20)
    assert tuple(index[:2])==(3,1) and tuple(index[4:8])==(0,0xffffffff,2,1)
    assert np.all(index[8:12]==0) and np.all(index[16:19]==0xffffffff)
    np.testing.assert_array_equal(index[12:15],a[0,12:15])
    assert m['retention']['partitions']['native_only_retained_entries']==1
    assert not m['application_ready'] and not m['physical_completeness_claim']
    assert m['contexts'][0]['camera_observation']['denominator']==metadata()['denominator']
    assert validate_scene_response_snapshot(m,p)


@pytest.mark.parametrize('edit', [lambda m:m.update(schema='studio-scene-responses-v1'),
    lambda m:m['camera_pair_contract'].update(row_words=384),
    lambda m:m['camera_pair_contract'].update(row_words=256.0),
    lambda m:m['camera_pair_contract'].update(application_ready=0),
    lambda m:m['families']['3'].update(codec='native-camera-response-pairs-v1'),
    lambda m:m['sections'][2].update(stride=1536),
    lambda m:m.update(application_ready=True)])
def test_mixed_or_unproved_contract_fails(edit):
    s=store();s.consume_camera_pairs(metadata(),*buffers());m,p=s.snapshot()
    edit(m)
    with pytest.raises(ValueError):validate_scene_response_snapshot(m,p)


@pytest.mark.parametrize('word,value', [(4,1),(6,1),(7,0),(8,1),(11,1),(12,0),(15,1),(16,0)])
def test_outer_index_cannot_invent_spatial_or_recipient_semantics(word,value):
    import hashlib
    s=store();s.consume_camera_pairs(metadata(),*buffers());m,p=s.snapshot()
    raw=np.frombuffer(p,'<u4').copy();raw[word]=value;p=raw.tobytes()
    m['payload_sha256']=hashlib.sha256(p).hexdigest()
    with pytest.raises(ValueError):validate_scene_response_snapshot(m,p)


@pytest.mark.parametrize('bad', ['short','count','version','incomplete','direction','bands','mask','primitive'])
def test_bad_pair_rejected_before_good_store_mutation(bad):
    s=store();s.consume_camera_pairs(metadata(),*buffers());before=s.snapshot()
    a,b=buffers();m=metadata(1)
    if bad=='short':b=b[:,:127]
    elif bad=='count':m['row_count']=2
    elif bad=='version':m['response_schema']='invented'
    elif bad=='incomplete':m['complete_dispatch_observation']=False
    elif bad=='direction':a[0,12]=0x7fc12345
    elif bad=='bands':a[0,6]=0
    elif bad=='mask':a[0,40]=8
    else:a[0,126]=3
    with pytest.raises(ValueError):s.consume_camera_pairs(m,a,b)
    assert s.snapshot()==before


def test_bounded_pair_retention_and_scene_invalidation_keep_v2_without_claiming_absence():
    s=store();a,b=buffers();n=s.native_only_capacity+3
    s.consume_camera_pairs(metadata(count=n),np.repeat(a,n,axis=0),np.repeat(b,n,axis=0))
    assert s.retained_entries==s.native_only_capacity and s.evicted_native_only_entries==3
    assert not s.snapshot()[0]['physical_completeness_claim']
    s.invalidate_scene('d'*64)
    assert s.snapshot()[0]['schema']=='studio-scene-responses-v2' and s.retained_entries==0


def test_default_v1_and_completed_application_path_stay_unchanged():
    s=store();assert s.snapshot()[0]['schema']=='studio-scene-responses-v1'
    assert camera_record_mode({})=='completed-samples-v1'
    assert camera_record_mode({'light_field_information':'evaluated-responses-v1'})=='completed-samples-v1'
    config={'light_field_information':'evaluated-responses-v1','light_field_camera_records':'interval-responses-v2'}
    assert camera_record_mode(config)=='interval-responses-v2'
    for bad in ({'light_field_camera_records':'interval-responses-v2'},dict(config,light_field_camera_records=True)):
        with pytest.raises(ValueError):camera_record_mode(bad)
    base=selected_sensor_reservation({'camera':{'resolution':32}},np.zeros((1,12),np.float32),retention_bytes=32768)
    pair=selected_sensor_reservation(dict(config,camera={'resolution':32},max_bounces=4),np.zeros((1,12),np.float32),retention_bytes=32768)
    assert pair['camera_copy']==4*(8+32*16*5*512)+16*32*16*5+4096
    assert pair['total']-base['total']==pair['camera_copy']-base['camera_copy']


def test_shared_consumer_does_not_promote_pairs_to_completed_query():
    lineage=NativeResponseLineage(0)
    c=SceneResponseConsumer(memory_budget_bytes=1024*1024,lineage_provider=lambda:lineage.snapshot(),
        bounds=[[0]*3,[1]*3],dimensions=[32]*3)
    c.consume_camera_pairs(metadata(),*buffers())
    assert c.current_camera_metadata is None and c.snapshot()[0]['schema']=='studio-scene-responses-v2'


def test_existing_publication_carries_v2_receipt_and_preserves_payload(tmp_path):
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher
    s=store();s.consume_camera_pairs(metadata(),*buffers())
    pub=SceneResponsePublisher(tmp_path,'e'*64,interval=0)
    first=pub.publish(s,now=1)
    assert first['schema']==first['metadata']['schema']=='studio-scene-responses-v2'
    assert (tmp_path/first['name']).read_bytes()==s.snapshot()[1]
    # Mixed versions can occupy the existing same bounded publication queue.
    second=pub.publish(store(),now=2)
    assert second['schema']=='studio-scene-responses-v1'
    assert pub.publish(s,now=3)['schema']=='studio-scene-responses-v2'


@pytest.mark.parametrize('bad_metadata',[None,[]])
def test_malformed_publication_metadata_refuses_room_without_deleting(tmp_path,bad_metadata):
    import json
    from camera_designer.studio_scene_response_publish import SceneResponsePublisher,sidecar_path
    s=store();s.consume_camera_pairs(metadata(),*buffers())
    pub=SceneResponsePublisher(tmp_path,'e'*64,interval=0)
    receipt=pub.publish(s,now=1);binary=tmp_path/receipt['name']
    receipt['metadata']=bad_metadata
    sidecar_path(binary).write_text(json.dumps(receipt))
    assert pub._room_locked() is False and binary.exists()


def test_actual_evidence_callback_reaches_store_with_identity_and_rejects_partial_reads(monkeypatch):
    import struct
    import sys
    from types import SimpleNamespace
    from camera_designer.studio_sensor_evidence import StudioResponseEvidence
    a,b=buffers();s=store()
    class Readback:
        GL_SHADER_STORAGE_BUFFER=1
        def __init__(self):self.current=0;self.buffers={1:struct.pack('<II',1,0)+a.tobytes(),2:bytes(8)+b.tobytes()}
        def glBindBuffer(self,target,handle):self.current=handle
        def glGetBufferSubData(self,target,offset,size):return self.buffers[self.current][offset:offset+size]
    gl=Readback();monkeypatch.setitem(sys.modules,'OpenGL',SimpleNamespace(GL=gl))
    owner=StudioResponseEvidence(memory_budget_bytes=100000,reserved_bytes=1000,
        scene_revision='a'*64,camera_revision='camera',consume=s.consume_camera_pairs)
    owner.buffer=1;owner.response_buffer=2;owner.capacity=1
    owner.identity['transport_scene_sha256']='c'*64
    dispatch=dict(source_sha256='b'*64,active_pixels=1,samples_per_pixel=1,active_path_bands=1,seed=9)
    owner.finish(dispatch_identity=dispatch)
    m,p=s.snapshot()
    assert m['contexts'][0]['stream_id']==owner.stream_id and m['contexts'][0]['source_sha256']=='b'*64
    assert m['contexts'][0]['dispatch']['seed']==9 and owner.sequence==1
    before=s.snapshot()
    gl.buffers[2]=gl.buffers[2][:-4]
    with pytest.raises(RuntimeError,match='truncated paired readback'):owner.finish(dispatch_identity=dispatch)
    assert owner.sequence==1 and s.snapshot()==before
    gl.buffers[2]=struct.pack('<II',0,1)+b.tobytes()
    with pytest.raises(RuntimeError,match='overflow'):owner.finish(dispatch_identity=dispatch)
    gl.buffers[2]=bytes(8)+b.tobytes()
    with pytest.raises(RuntimeError,match='missing original camera invocations'):
        owner.finish(dispatch_identity=dict(dispatch,active_pixels=2))
    assert owner.sequence==1 and s.snapshot()==before


def test_camera_pair_worker_mode_never_removes_hardware_hold_or_default_completed_path():
    import ast
    from pathlib import Path
    source=(Path(__file__).parents[1]/'camera_designer/marble_preview_worker.py').read_text()
    tree=ast.parse(source)
    held=[node.lineno for node in ast.walk(tree) if isinstance(node,ast.Constant) and
          node.value=='evaluated scene responses execution held: hardware compile resource validation pending']
    imports=[node.lineno for node in ast.walk(tree) if isinstance(node,ast.Import) and
             any(alias.name=='demo_pluck_gl' for alias in node.names)]
    assert len(held)==1 and held[0]<min(imports)
    assert 'completed_samples=not camera_pairs' in source
    assert 'if not camera_pairs:response_consumer.application_builder=build_recorded_application' in source
