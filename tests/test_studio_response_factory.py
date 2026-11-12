import numpy as np
import pytest
from camera_designer.studio_response_factory import make_response_owner,retained_array_bytes

def test_absent_response_flag_does_not_import_gl_or_allocate():
    assert make_response_owner({},None,None,consume=None) is None

def test_count_retained_backing_storage_once():
    a=np.zeros(16,np.float32);b=a[1:]
    assert retained_array_bytes(a,b,{'again':a})==64

def test_explicit_response_requires_synchronous_consumer():
    with pytest.raises(TypeError):make_response_owner({'light_field_information':'evaluated-responses-v1'},{},{},consume=None)


def test_rebind_uses_current_scene_and_camera_after_mutations():
    from types import SimpleNamespace
    from camera_designer.studio_response_factory import rebind_response_owner
    calls=[];owner=SimpleNamespace(rebind_identity=lambda **kw:calls.append(kw))
    accumulator=SimpleNamespace(_light_field_information=owner,_w=256,_h=128,_aspect=2)
    state={'directional_record':{'scene_sha256':'a'*64}}
    rebind_response_owner(accumulator,state,{'eye_m':[0,0,1]})
    state['directional_record']['scene_sha256']='b'*64
    rebind_response_owner(accumulator,state,{'eye_m':[0,0,2]})
    assert calls[0]['scene_revision']=='a'*64 and calls[1]['scene_revision']=='b'*64
    assert calls[0]['camera_revision']!=calls[1]['camera_revision']
    rebind_response_owner(SimpleNamespace(_light_field_information=None),None,None)


def test_selected_live_memory_reservation_before_original_warmup():
    from camera_designer.studio_response_factory import selected_sensor_reservation,publication_working_reserve
    config={'live':{'width':128,'height':72,'history_samples':16,'checkpoint':True}}
    plan=selected_sensor_reservation(config,np.empty((32832,12),np.float32),retention_bytes=24*1024*1024)
    assert plan['history']==2654208 and plan['camera_copy']==3182616
    assert plan['total']==40402968
    state={'dims':(96,96,96),'directional_record':{'bin_count':12,'grid_size':64}}
    assert publication_working_reserve(state)==317718528
    state['directional_record']['progressive']={'sum':object()}
    assert publication_working_reserve(state)==317718528-150994944


def test_scene_adapter_keeps_camera_and_population_separate_from_physical_invalidation(monkeypatch):
    from camera_designer.studio_response_factory import SceneResponseConsumer
    import camera_designer.studio_scene_responses as module
    calls=[]
    class Store:
        def __init__(self,**kw):self.scene_revision=kw['scene_revision'];calls.append(('new',self.scene_revision))
        def invalidate_scene(self,revision):
            if revision!=self.scene_revision:calls.append(('invalidate',revision));self.scene_revision=revision
        def consume_forward(self,m,r):calls.append(('forward',m['population_index']))
        def consume_camera_samples(self,m,r):calls.append(('camera',m['camera_revision']))
    monkeypatch.setattr(module,'SceneResponseStore',Store)
    c=SceneResponseConsumer(memory_budget_bytes=32768,lineage_provider=lambda:{},bounds=None,dimensions=None)
    c({'scene_revision':'physicalA','population_index':0},None)
    c({'scene_revision':'physicalA','population_index':1},None)
    c.consume_camera_samples({'scene_revision':'physicalA','camera_revision':'cameraB'},None)
    assert not any(x[0]=='invalidate' for x in calls)
    c({'scene_revision':'physicalB','population_index':2},None)
    assert calls[-2]==('invalidate','physicalB')


def test_retention_identity_excludes_view_but_keeps_authoritative_fixed_geometry_and_lights():
    from camera_designer.studio_response_factory import retained_scene_identity
    vertices=np.zeros((3,3),np.float32);ids=np.zeros(1,np.int32);pbr=np.zeros((1,16),np.float32)
    config={'camera':{'eye_m':[0,0,1]},'live':{'width':128},'room':{'light_power':7},'emitter_sampling':{'version':1,'seed':42}}
    a=retained_scene_identity(config,vertices,ids,pbr)
    moved={**config,'camera':{'eye_m':[1,0,1]},'live':{'width':256},'emitter_sampling':{'version':1,'seed':43}}
    assert retained_scene_identity(moved,vertices,ids,pbr)==a
    assert retained_scene_identity({**config,'room':{'light_power':8}},vertices,ids,pbr)!=a
    vertices[0,0]=1
    assert retained_scene_identity(config,vertices,ids,pbr)!=a


def test_current_query_never_confuses_program_hash_with_source_population():
    import hashlib
    from types import SimpleNamespace
    from camera_designer.studio_response_factory import current_scene_query_metadata
    roots=np.zeros((1,12),'<f4');rh=hashlib.sha256(roots.tobytes()).hexdigest();sh=hashlib.sha256(b'shader').hexdigest()
    m=dict(scene_revision='fixed',camera_revision='camera',transport_scene_sha256='housingA',source_sha256=sh,source_population_sha256=rh,live_revision=2)
    state=dict(source_buf=roots,response_transport_scene_sha256='housingA',response_camera_owner=SimpleNamespace(identity=dict(m)),response_camera_accumulator=SimpleNamespace(_sensor_shader_source='shader'))
    m['reactive_queue']={'observed':True,'complete':True,'incomplete_observed':False}
    state['response_reactive_queue']=dict(m['reactive_queue'])
    consumer=SimpleNamespace(current_camera_metadata=m)
    assert current_scene_query_metadata(state,consumer,2) is not None
    state['response_reactive_queue']['complete']=False
    assert current_scene_query_metadata(state,consumer,2) is None
    state['response_reactive_queue']['complete']=True
    state['response_transport_scene_sha256']='housingB'
    assert current_scene_query_metadata(state,consumer,2) is None
    state['response_transport_scene_sha256']='housingA';roots[0,0]=1
    assert current_scene_query_metadata(state,consumer,2) is None
