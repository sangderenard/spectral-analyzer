"""Bounded GPU DDA checks against independent plane-intersection integration."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def oracle(a,b,lo,hi,energy):
    a,b,lo,hi=[np.asarray(x,float) for x in (a,b,lo,hi)];delta=b-a;length=np.linalg.norm(delta)
    result=np.zeros((32,32,32),np.int64)
    if length==0:return result
    if any(delta[k]==0 and (a[k]<lo[k] or a[k]>=hi[k]) for k in range(3)):return result
    times=[0.,1.]
    for k in range(3):
        if delta[k]!=0:
            times.extend(t for t in (np.linspace(lo[k],hi[k],33)-a[k])/delta[k] if 0<t<1)
    times=sorted(set(times))
    for start,end in zip(times[:-1],times[1:]):
        mid=a+delta*(start+end)*.5;q=np.floor((mid-lo)/(hi-lo)*32).astype(int)
        if np.all(q>=0) and np.all(q<32):result[q[2],q[1],q[0]]+=int(np.rint(energy*(end-start)*length*65535))
    return result


def main():
    import demo_pluck_gl as renderer
    from OpenGL import GL
    from camera_designer.studio_directional_field import directional_glsl,allocate,clear,readback,voxel_volume,word_count
    stencil=int(sys.argv[1]) if len(sys.argv)>1 else 12
    GLSL=directional_glsl(stencil);WORDS=word_count(stencil)
    renderer._ensure_worker_gl_context()
    shader='''#version 430 core
layout(local_size_x=1) in;
uniform vec3 uBoxMin,uBoxMax,uA,uB;
uniform float uEnergy;
const int uFixed3Transport=1;
int studio_band_index(float frequency){return 0;}
'''+GLSL+'\nvoid main(){studio_record_segment(uA,uB,vec4(1,0,uEnergy,0));}'
    program=renderer._prog((shader,GL.GL_COMPUTE_SHADER));assert GL.glGetProgramiv(program,GL.GL_LINK_STATUS)
    record=allocate(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),bounds=([0,0,0],[1,1,1]),stencil=stencil)
    cases=[]
    for axis in range(3):
        a=np.array([.4,.4,.4]);b=a.copy();a[axis]=-.5;b[axis]=1.5
        cases.extend([(a,b),(b,a)])
    cases.extend([([-.5,-.5,-.5],[1.5,1.5,1.5]),([1,1,1],[0,0,0]),
        ([0,.5,.5],[1,.5,.5]),([1,.5,.5],[0,.5,.5]),
        ([0,1,.5],[1,1,.5]),([0,0,.5],[1,0,.5]),
        ([-1,-1,0],[-.5,-.5,1]),([.5,.5,.5],[.5,.5,.5]),
        ([-.2,.1234,.7654],[1.2,.9567,.1234])])
    try:
        for index,(a,b) in enumerate(cases):
            clear(record);GL.glUseProgram(program)
            for name,value in [('uBoxMin',[0,0,0]),('uBoxMax',[1,1,1]),('uA',a),('uB',b)]:GL.glUniform3f(GL.glGetUniformLocation(program,name),*value)
            GL.glUniform1f(GL.glGetUniformLocation(program,'uEnergy'),.25)
            GL.glDispatchCompute(1,1,1);GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
            raw=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,(WORDS+1)*4),np.int32)
            assert raw[-1]==0,(index,'overflow/traversal flag',raw[-1])
            measured=raw[:-1].reshape(32,32,32,stencil,6)[...,:3].sum(axis=(3,4))
            expected=oracle(np.float32(a),np.float32(b),[0,0,0],[1,1,1],.25)
            np.testing.assert_allclose(measured,expected,rtol=0,atol=1,err_msg=f'case {index}')
        # Same normalized path, uniform doubling of world length: track length
        # doubles, voxel volume scales8, so throughput density scales1/4.
        values=[]
        for scale in [1,2]:
            clear(record);record['voxel_volume_m3']=voxel_volume(([0,0,0],[scale]*3))
            for name,value in [('uBoxMin',[0,0,0]),('uBoxMax',[scale]*3),('uA',[0,.5*scale,.5*scale]),('uB',[scale,.5*scale,.5*scale])]:GL.glUniform3f(GL.glGetUniformLocation(program,name),*value)
            GL.glUniform1f(GL.glGetUniformLocation(program,'uEnergy'),1)
            GL.glDispatchCompute(1,1,1);values.append(readback(record))
        np.testing.assert_allclose(values[1][...,:3],values[0][...,:3]/4,rtol=1e-6,atol=1e-5)
        print(json.dumps(dict(result='PASS',renderer=GL.glGetString(GL.GL_RENDERER).decode(),cases=len(cases),stencil=stencil,checks=['axis signs','oblique','clipping','parallel boundary policy','zero and missed segments','bounded DDA termination','independent per-voxel track lengths','voxel-volume density scaling'])))
    finally:
        GL.glDeleteBuffers(1,[record['buffer']]);GL.glDeleteProgram(program)

if __name__=='__main__':main()
