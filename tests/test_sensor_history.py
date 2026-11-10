"""Finite ring layout contracts and optional real GPU compute invariants."""
import os
import numpy as np
import pytest
from camera_designer.sensor_history import (SensorHistoryLayout, SENSOR_HISTORY_GLSL,
    MAX_HISTORY_BYTES, check_history_device, replace_history_buffers, check_history_sample_count)


def test_ring_memory_is_bounded_before_allocation():
    assert SensorHistoryLayout(256,256,3,16).nbytes == 14155776
    assert SensorHistoryLayout(512,512,3,32).nbytes == 106954752
    assert SensorHistoryLayout(128,128,3,10920).nbytes <= MAX_HISTORY_BYTES
    assert SensorHistoryLayout(512,512,3,674,True).nbytes == 2145386496
    for args in [(512,512,3,681),(512,512,3,675,True),(128,128,3,10921),(256,256,3,0),(256,256,3,1.5),(0,1,1,1)]:
        with pytest.raises(ValueError): SensorHistoryLayout(*args)


def test_device_limits_and_available_memory_before_allocation():
    layout=SensorHistoryLayout(512,512,3,674,True)
    check_history_device(layout,2**31-1)
    with pytest.raises(ValueError,match="SSBO"):check_history_device(layout,128*1024*1024)
    with pytest.raises(MemoryError,match="headroom"):check_history_device(layout,2**31-1,2**30)
    check_history_device(layout,2**31-1,2**30,2**31)  # reclaim an owned old ring
    check_history_sample_count((1<<53)-2)
    with pytest.raises(OverflowError,match="sample-count limit"):check_history_sample_count((1<<53)-1)


def test_resize_releases_old_before_new_and_failed_allocations_clean_up():
    from types import SimpleNamespace
    class GL:
        GL_MAX_SHADER_STORAGE_BLOCK_SIZE=1;GL_NUM_EXTENSIONS=2;GL_EXTENSIONS=3
        GL_SHADER_STORAGE_BUFFER=4;GL_DYNAMIC_DRAW=5;GL_BUFFER_SIZE=6
        def __init__(self,fail=False):self.events=[];self.id=10;self.size=0;self.fail=fail
        def glGetIntegerv(self,key):return 2**31-1 if key==1 else 0
        def glGetStringi(self,*a):raise AssertionError("no extensions advertised")
        def glFinish(self):self.events.append("finish")
        def glDeleteBuffers(self,n,ids):self.events.append(("delete",tuple(ids)))
        def glGenBuffers(self,n):self.id+=1;self.events.append("allocate");return self.id
        def glBindBuffer(self,*a):pass
        def glBufferData(self,target,size,*a):
            if self.fail:raise RuntimeError("out of memory")
            self.size=size
        def glGetBufferParameteriv(self,*a):return self.size
    owner=lambda:SimpleNamespace(_history_buffers=[7,8],_history_samples=16,_history_bytes=100,_history_checkpoint=False)
    value=owner();gl=GL();layout=SensorHistoryLayout(128,128,3,10914,True)
    replace_history_buffers(value,layout,gl)
    assert gl.events[:3]==["finish",("delete",(7,8)),"allocate"]
    assert value._history_bytes==layout.nbytes and value._history_checkpoint
    value=owner();gl=GL(fail=True)
    with pytest.raises(MemoryError,match="allocation failed"):replace_history_buffers(value,layout,gl)
    assert value._history_buffers==[] and value._history_samples==value._history_bytes==0
    assert gl.events[-1]==("delete",(11,))
    value=owner();gl=GL();gl.glGetIntegerv=lambda key:128*1024*1024 if key==1 else 0
    with pytest.raises(ValueError,match="SSBO"):replace_history_buffers(value,layout,gl)
    assert value._history_buffers==[7,8] and gl.events==[],"reject before releasing the working ring"


@pytest.mark.skipif(os.environ.get("MARBLE_RING_GPU_TEST") != "1", reason="explicit bounded GPU test only")
def test_actual_gpu_ring_wrap_expiry_reset_and_layer_independence():
    import demo_pluck_gl as renderer
    from OpenGL.GL import (glGenBuffers, glBindBuffer, glBufferData, glBindBufferBase,
        glGetBufferSubData, glDeleteBuffers, glDeleteProgram, glUseProgram,
        glUniform1i, glUniform1f, glUniform2i, glGetUniformLocation,
        glDispatchCompute, glMemoryBarrier, glFinish, glGetProgramiv,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, GL_COMPUTE_SHADER,
        GL_SHADER_STORAGE_BARRIER_BIT, GL_LINK_STATUS)
    renderer._ensure_worker_gl_context()
    source = "#version 430 core\nlayout(local_size_x=1) in;\nuniform ivec2 uSensorSize;\n" + SENSOR_HISTORY_GLSL + """
layout(std430,binding=8) buffer Result {float result[];};
uniform float uValue;
void main(){int pixel=int(gl_GlobalInvocationID.x); int layer=int(gl_GlobalInvocationID.y);
result[layer*2+pixel]=sensor_history_value(layer,ivec2(pixel,0),uValue+float(layer*10+pixel));}
"""
    program=renderer._prog((source,GL_COMPUTE_SHADER))
    assert glGetProgramiv(program,GL_LINK_STATUS)
    # Also compile the real full sensor shader so integration is covered.
    actual=renderer._prog((renderer._GPU_SENSOR_CS,GL_COMPUTE_SHADER))
    assert glGetProgramiv(actual,GL_LINK_STATUS)
    buffers=[]
    try:
        for binding,size in ((6,2*3*4*4),(7,2*3*32),(8,2*3*4)):
            buffer=int(glGenBuffers(1));buffers.append(buffer)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer)
            glBufferData(GL_SHADER_STORAGE_BUFFER,size,None,GL_DYNAMIC_DRAW)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,binding,buffer)
        glUseProgram(program)
        glUniform2i(glGetUniformLocation(program,"uSensorSize"),2,1)
        glUniform1i(glGetUniformLocation(program,"uHistorySamples"),4)
        values=[1.,2.,3.,4.,9.,0.,0.,0.,0.,1e20,1.,2.,3.,4.,0.,0.,0.,0.]
        for index,value in enumerate(values):
            glUniform1i(glGetUniformLocation(program,"uHistorySlot"),index%4)
            glUniform1i(glGetUniformLocation(program,"uHistoryCount"),min(index,4))
            glUniform1f(glGetUniformLocation(program,"uValue"),value)
            glDispatchCompute(2,3,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);glFinish()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[-1])
            observed=np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,24),np.float32)
            expected=np.array([np.mean([np.float32(np.float32(v)+np.float32(layer*10+pixel))
                for v in values[max(0,index-3):index+1]],dtype=np.float64)
                for layer in range(3) for pixel in range(2)],np.float32)
            np.testing.assert_array_equal(observed,expected)
        # Reset validity without clearing allocated sample storage.
        glUniform1i(glGetUniformLocation(program,"uHistorySlot"),0)
        glUniform1i(glGetUniformLocation(program,"uHistoryCount"),0)
        glUniform1f(glGetUniformLocation(program,"uValue"),7.)
        glDispatchCompute(2,3,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);glFinish()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[-1])
        np.testing.assert_array_equal(np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,24),np.float32),[7,8,17,18,27,28])
        # Capacity changes start a new generation: new address stride and
        # checkpoint state are logically empty, not mixed with the old ring.
        glUniform1i(glGetUniformLocation(program,"uHistoryCheckpoint"),1)
        glUniform1i(glGetUniformLocation(program,"uHistorySamples"),2)
        for index,value in enumerate([2.,6.,10.,14.]):
            glUniform1i(glGetUniformLocation(program,"uHistorySlot"),index%2)
            glUniform1i(glGetUniformLocation(program,"uHistoryCount"),min(index,2))
            glUniform1f(glGetUniformLocation(program,"uValue"),value)
            glDispatchCompute(2,3,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);glFinish()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[-1])
            expected=np.array([np.mean([2.,6.,10.,14.][:index+1])+layer*10+pixel
                for layer in range(3) for pixel in range(2)],np.float32)
            np.testing.assert_array_equal(np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,24),np.float32),expected)
        # Checkpoint evictions are counted once, so ring+checkpoint matches
        # the ordinary mean of ALL input samples rather than a rolling mean.
        glUniform1i(glGetUniformLocation(program,"uHistoryCheckpoint"),1)
        glUniform1i(glGetUniformLocation(program,"uHistorySamples"),4)
        for index,value in enumerate(values):
            glUniform1i(glGetUniformLocation(program,"uHistorySlot"),index%4)
            glUniform1i(glGetUniformLocation(program,"uHistoryCount"),min(index,4))
            glUniform1f(glGetUniformLocation(program,"uValue"),value)
            glDispatchCompute(2,3,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);glFinish()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[-1])
            observed=np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,24),np.float32)
            expected=np.array([np.mean([np.float32(np.float32(v)+np.float32(layer*10+pixel))
                for v in values[:index+1]],dtype=np.float64)
                for layer in range(3) for pixel in range(2)],np.float32)
            np.testing.assert_array_equal(observed,expected)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[1])
            totals=np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,2*3*32),np.float64).reshape(-1,4)
            np.testing.assert_array_equal(totals[:,3],max(0,index+1-4))
        # A new camera/scene starts with validity=0: checkpoint counts, old
        # sums and compensation must all disappear alongside the ring.
        glUniform1i(glGetUniformLocation(program,"uHistorySlot"),0)
        glUniform1i(glGetUniformLocation(program,"uHistoryCount"),0)
        glUniform1f(glGetUniformLocation(program,"uValue"),7.)
        glDispatchCompute(2,3,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);glFinish()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[-1])
        np.testing.assert_array_equal(np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,24),np.float32),[7,8,17,18,27,28])
    finally:
        glUseProgram(0)
        glDeleteBuffers(len(buffers),buffers)
        glDeleteProgram(program);glDeleteProgram(actual)


def test_live_forward_field_is_fresh_and_fixed_for_each_whole_sweep(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import importlib
    gl = importlib.import_module("OpenGL.GL")
    from camera_designer.marble_live_worker import run_live_sensor
    monkeypatch.setattr(gl,"glFinish",lambda:None,raising=False)
    events=[];field={"_pump_seed":0}
    class Accumulator:
        _w=2;_h=23;_rows_per_frame=16;_row_off=0;_pass=0;_history_bytes=100
        def set_history(self,n,checkpoint=False):self._history_samples=n
        def tick(self,strips):
            assert strips==1
            events.append(("tick",self._pass,self._row_off))
            self._row_off+=min(16,self._h-self._row_off)
            if self._row_off==self._h:self._row_off=0;self._pass+=1
    def clear(f):events.append(("clear",));f["_pump_seed"]=0
    def pump(f,n_rays):events.append(("pump",f["_pump_seed"],n_rays));f["_pump_seed"]+=7
    renderer=SimpleNamespace(_clear_prebuilt=clear,_gpu_pump_prebuilt=pump)
    config={"live":{"history_samples":4,"aspect":2/23,"lease_seconds":30}}
    result=run_live_sensor(Accumulator(),renderer,field,tmp_path,"a"*64,config,0,
        lambda *a,**k:{"digest":"a"*64,"image":"test.png"},lambda *a:None,max_epochs=5)
    assert result["reason"]=="test epoch limit"
    assert len(events)==40
    for sweep in range(10):
        assert events[sweep*4:sweep*4+4]==[("clear",),("pump",sweep*7,8192),("tick",sweep,0),("tick",sweep,16)]


def test_display_preserves_history_but_camera_and_resize_reset_checkpoint(tmp_path,monkeypatch):
    import importlib,json
    from types import SimpleNamespace
    from camera_designer import marble_live_worker as worker
    gl=importlib.import_module("OpenGL.GL")
    monkeypatch.setattr(gl,"glFinish",lambda:None,raising=False)
    now=[0.]
    def monotonic():now[0]+=.3;return now[0]
    monkeypatch.setattr(worker.time,"monotonic",monotonic)
    camera=dict(eye_m=[0,0,.07],target_m=[0,0,.18],fov_degrees=18,
                aperture_radius_m=.002,focus_distance_m=.11)
    live=dict(history_samples=4,aspect=1,lease_seconds=30,checkpoint=True,
              display=dict(tint=0,normalization="auto99",normalization_white=1))
    control=dict(camera=dict(camera),live={**live,"display":{**live["display"],"tint":1}},
                 revision=1,expires_at=worker.time.time()+30)
    path=tmp_path/"live-control.json"
    path.write_text(json.dumps(control))
    resets=[];seen=[]
    class Accumulator:
        _w=2;_h=16;_rows_per_frame=16;_row_off=0;_pass=0;_history_bytes=100
        def set_history(self,n,checkpoint=False):self._history_samples=n
        def set_camera(self,*a,**k):pass
        def set_lens(self,*a,**k):pass
        def clear(self):resets.append(control["revision"]);self._pass=0
        def tick(self,strips):self._pass+=1
    def snapshot(accumulator,*a,**kwargs):
        seen.append((accumulator._pass,kwargs["display_params"]["tint"]))
        if len(seen)==1:control["live"]["history_samples"]=8;control["revision"]=2
        elif len(seen)==2:control["camera"]["eye_m"]=[.01,0,.07];control["revision"]=3
        path.write_text(json.dumps(control))
        return {"image":"test.png"}
    field={"_pump_seed":0}
    renderer=SimpleNamespace(_clear_prebuilt=lambda f:None,_gpu_pump_prebuilt=lambda *a,**k:None)
    result=worker.run_live_sensor(Accumulator(),renderer,field,tmp_path,"a"*64,
        {"live":live,"camera":camera},0,snapshot,lambda *a:None,max_epochs=3)
    assert resets==[2,3],"display-only revision must not clear the ring/checkpoint"
    assert seen==[(4,1),(4,1),(4,1)]
    assert result["live_revision"]==3 and result["checkpoint_enabled"]


def test_live_snapshot_tint_and_normalizer_use_shared_display_pipeline(tmp_path,monkeypatch):
    import importlib
    from types import SimpleNamespace
    from PIL import Image
    from camera_designer.marble_preview_worker import _snapshot
    from camera_designer.marble_color_profile import sensor_to_linear_srgb,render_linear_srgb
    gl=importlib.import_module("OpenGL.GL");bound=[0]
    raw=np.array([[[1.,2.,3.],[2.,4.,1.]],[[3.,1.,2.],[.1,.5,.2]]],np.float32)
    original=raw.copy()
    for name in ("GL_TEXTURE_2D","GL_RGBA","GL_FLOAT"):
        monkeypatch.setattr(gl,name,0,raising=False)
    monkeypatch.setattr(gl,"glFinish",lambda:None,raising=False)
    monkeypatch.setattr(gl,"glBindTexture",lambda target,tex:bound.__setitem__(0,tex),raising=False)
    monkeypatch.setattr(gl,"glGetTexImage",lambda *a:np.repeat(raw[:,:,bound[0]-1,None],4,axis=2).tobytes(),raising=False)
    acc=SimpleNamespace(_textures=[1,2,3],_w=2,_h=2,_history_samples=4)
    params=dict(tint=1.25,normalization="manual",normalization_white=3)
    status=_snapshot(acc,tmp_path,"a"*64,1,1,0,8,4,transport="fixed:3",display_params=params)
    linear,_=sensor_to_linear_srgb(raw)
    expected,white=render_linear_srgb(linear,{"gamma":1,**params})
    pixels=np.asarray(Image.open(tmp_path/status["image"]))
    np.testing.assert_array_equal(pixels[:,:,:3],expected)
    np.testing.assert_array_equal(pixels[:,:,3],np.full((2,2),255,np.uint8))
    np.testing.assert_array_equal(raw,original)
    assert status["live_display"]==params and status["display_white"]==white
    assert status["display_transform"]["normalization"]=="manual"
