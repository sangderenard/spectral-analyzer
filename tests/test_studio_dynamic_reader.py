"""Real scene packer + live loop; GL transport is recorded, never executed."""
import ast
import copy
import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_designer.camera_frame import sensor_camera_basis
from camera_designer.studio_photo_geometry import prepare_photo_geometry
from camera_designer.studio_dynamic_reader import StudioDynamicReader


def actual_upload():
    source = Path(__file__).resolve().parents[1] / "demo_pluck_gl.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "_update_prebuilt_geometry")
    namespace = {"np":np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
    return namespace[node.name]


@pytest.fixture
def setup(monkeypatch):
    events=[]
    gl=importlib.import_module('OpenGL.GL')
    for name in ('GL_SHADER_STORAGE_BUFFER','GL_STATIC_DRAW'):
        monkeypatch.setattr(gl,name,1,raising=False)
    bound=[None]
    monkeypatch.setattr(gl,'glFinish',lambda:events.append('finish'),raising=False)
    monkeypatch.setattr(gl,'glBindBuffer',lambda target,handle:bound.__setitem__(0,handle),raising=False)
    monkeypatch.setattr(gl,'glBufferData',lambda target,size,data,usage:events.append(('upload',bound[0],np.array(data,copy=True))),raising=False)
    for name in ('glUseProgram','glUniform1i','glUniform3f'):
        monkeypatch.setattr(gl,name,lambda *args:None,raising=False)
    monkeypatch.setattr(gl,'glGetUniformLocation',lambda *args:0,raising=False)
    pbr=np.zeros((1,16),np.float32);pbr[0,:3]=.5;pbr[0,6:8]=[1.5,1]
    vertices=np.array([[[0,0,.18],[.01,0,.18],[0,.01,.18]]],np.float32)
    ids=np.array([0],np.int32)
    static={'vertices':[-.1,-.1,.2,.1,-.1,.2,0,.1,.2],'material_ids':[0],'pbr':pbr.ravel().tolist()}
    camera={'eye_m':[0,0,.07],'target_m':[0,0,.18],'fov_degrees':18,'aperture_radius_m':.002,'focus_distance_m':.11}
    live={'aspect':1,'history_samples':2,'lease_seconds':30,'ray_batch_size':128,'checkpoint':False}
    config={'transport':'fixed:3','sphere_coverage':'mesh','scale_m':1,'object_center_m':[0,0,0],
            'fixtures':static,'camera':camera,'live':live}
    scene,_,_,_=prepare_photo_geometry(config,vertices,ids,pbr)
    field={'source_buf':scene[7].copy(),'ssbo':list(range(100,113)),'prog':1,'_pump_seed':7,'native_pbr_ssbo':118}
    renderer=SimpleNamespace(_build_gpu_bvh=lambda tri:(np.zeros((1,12),np.float32),np.arange(len(tri),dtype=np.uint32)),
        _update_prebuilt_geometry=actual_upload(),_clear_prebuilt=lambda f:events.append('clear-forward'),
        _gpu_pump_prebuilt=lambda f,n_rays:events.append('pump'))
    class Sensor:
        _row_off=0;_w=2;_h=83;_rows_per_frame=16;_pass=0;_history_bytes=1
        def set_scene_geometry(self,**kwargs):events.append(('sensor',kwargs));self.geometry=kwargs
        def clear(self):events.append('clear-sensor');self._row_off=0;self._pass=0
        def set_history(self,*args,**kwargs):pass
        def tick(self,strips):
            events.append(('tick',self._row_off))
            self._row_off+=min(16,self._h-self._row_off)
            if self._row_off==self._h:self._row_off=0;self._pass+=1
    sensor=Sensor()
    reader=StudioDynamicReader(renderer,field,config,scene,ids,pbr)
    eye,right,up,forward=sensor_camera_basis(camera['eye_m'],camera['target_m'])
    view=np.eye(4);view[:3,:3]=[right,up,-forward];view[:3,3]=-view[:3,:3]@eye
    packet={'schema':'turing.studio-dynamic-geometry.v1','revision':1,'step':6,'seconds':1/60,
        'nativeFrame':{'unitM':1,'originM':[0,0,0]},'rayOriginM':[0,0,0],
        'camera':{'view':view.T.ravel().tolist(),'fovDegrees':18,'aspect':1},
        'staticRay':copy.deepcopy(static),'subject':{'ray':{'vertices':vertices.ravel().tolist(),
        'normals':[0,0,1]*3,'material_ids':[0]}}}
    return reader,packet,sensor,events,renderer,field,config


def test_real_packer_upload_and_both_readers_commit_after_finish(setup):
    reader,packet,sensor,events,_,field,_=setup
    packet['subject']['ray']['vertices'][0]=.002
    reader.stage(packet)
    packet['subject']['ray']['vertices'][0]=99
    assert events==[] and reader.ack is None
    ack=reader.commit_boundary(sensor)
    assert events[0]=='finish' and events[-1]=='finish'
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=='upload']==[118,100,101,102,108,109,110,112]
    assert events[-3:-1]==['clear-forward','clear-sensor']
    assert ack['step']==6 and ack['both_readers_installed'] and ack['histories_cleared']
    assert ack['normal_mode']=='geometric-face' and field['_pump_seed']==7
    assert ack['authored_vertex_normals_consumed'] is False
    assert json.loads(reader.active_packet)['subject']['ray']['vertices'][0]==.002
    assert reader.commit_boundary(sensor) is None


def test_boundary_owner_latest_and_invalid_stage_preserve_pending(setup):
    reader,packet,sensor,events,*_=setup
    reader.stage(packet);packet['step']=12;packet['seconds']=2/60;reader.stage(packet)
    with pytest.raises(ValueError,match='stale'):reader.stage(packet)
    bad=copy.deepcopy(packet);bad['step']=18;bad['staticRay']['pbr'][0]=.9
    with pytest.raises(ValueError,match='room/light'):reader.stage(bad)
    sensor._row_off=16
    with pytest.raises(RuntimeError,match='boundary'):reader.commit_boundary(sensor)
    sensor._row_off=0
    errors=[]
    def other_thread():
        try:reader.commit_boundary(sensor)
        except RuntimeError as error:errors.append(str(error))
    thread=threading.Thread(target=other_thread);thread.start();thread.join()
    assert errors and events==[]
    assert reader.commit_boundary(sensor)['step']==12


@pytest.mark.parametrize('change,match',[
    (lambda p:p['camera'].__setitem__('fovDegrees',20),'projection'),
    (lambda p:p['camera']['view'].__setitem__(12,3),'camera'),
    (lambda p:p['subject']['ray'].__setitem__('material_ids',[1]),'identity'),
    (lambda p:p['subject']['ray'].__setitem__('vertices',[0]*9),'drop'),
    (lambda p:p['subject']['ray'].__setitem__('normals',[0]*9),'normals')])
def test_unsupported_updates_refuse_without_reader_mutation(setup,change,match):
    reader,packet,_,events,*_=setup;change(packet)
    with pytest.raises(ValueError,match=match):reader.stage(packet)
    assert events==[] and reader.pending is None


def test_partial_install_failure_poisoned_without_ack_or_more_rays(setup):
    reader,packet,sensor,events,renderer,*_=setup
    reader.stage(packet)
    def fail(**kwargs):raise RuntimeError('sensor install failure')
    sensor.set_scene_geometry=fail
    with pytest.raises(RuntimeError,match='sensor install'):reader.commit_boundary(sensor)
    assert reader.ack is None and reader.failed and 'pump' not in events
    with pytest.raises(RuntimeError,match='failed'):reader.commit_boundary(sensor)
    packet['step']=12
    with pytest.raises(RuntimeError,match='failed'):reader.stage(packet)


def test_new_stage_during_install_cannot_mutate_the_active_packet(setup):
    reader,packet,sensor,events,renderer,*_=setup
    reader.stage(packet)
    upload=renderer._update_prebuilt_geometry
    next_packet=copy.deepcopy(packet);next_packet['step']=12;next_packet['seconds']=2/60
    next_packet['subject']['ray']['vertices'][0]=.003
    def replace(*args,**kwargs):
        upload(*args,**kwargs)
        thread=threading.Thread(target=lambda:reader.stage(next_packet))
        thread.start();thread.join()
        renderer._update_prebuilt_geometry=upload
    renderer._update_prebuilt_geometry=replace
    assert reader.commit_boundary(sensor)['step']==6
    assert json.loads(reader.active_packet)['subject']['ray']['vertices'][0]==0
    assert reader.commit_boundary(sensor)['step']==12
    assert json.loads(reader.active_packet)['subject']['ray']['vertices'][0]==.003


def test_static_authored_normals_retained_without_claiming_reader_support(setup):
    reader,packet,sensor,*_=setup
    packet['staticRay']['normals']=[0,0,1]*3
    reader.stage(packet)
    ack=reader.commit_boundary(sensor)
    assert json.loads(reader.active_packet)['staticRay']['normals']==[0,0,1]*3
    assert ack['authored_vertex_normals_consumed'] is False


def test_actual_live_loop_ack_occurs_at_complete_sweep_before_next_packet(setup,tmp_path,monkeypatch):
    from camera_designer import marble_live_worker as worker
    reader,packet,sensor,events,renderer,field,config=setup
    sensor._row_off=16  # old sweep has five strips left
    reader.stage(packet)
    clock=[0]
    def now():clock[0]+=.3;return clock[0]
    monkeypatch.setattr(worker.time,'monotonic',now)
    published=[]
    result=worker.run_live_sensor(sensor,renderer,field,tmp_path,'fixture',config,0,
        lambda *args,**kwargs:{},lambda path,value:published.append((path.name,copy.deepcopy(value))),
        max_epochs=3,dynamic_scene_reader=reader)
    index=events.index('finish')
    assert [e[1] for e in events[:index] if isinstance(e,tuple) and e[0]=='tick']==[16,32,48,64,80]
    assert any(name=='scene-ack.json' and value['step']==6 for name,value in published)
    assert result['scene_ack']['step']==6
    assert events.index('pump')>index
