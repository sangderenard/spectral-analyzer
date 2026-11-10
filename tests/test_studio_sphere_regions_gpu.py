"""Opt-in actual production GPU sphere/boundary invariants.

The ordinary pytest process stubs OpenGL. Run this function directly in an idle
GPU slot: python -c "from tests.test_studio_sphere_regions_gpu import test_actual_gpu_sphere_regions as t;t()"
"""
import os
import pytest

@pytest.mark.skipif(os.environ.get("SPECTRAL_ACTUAL_GPU") != "1", reason="requires explicit idle-GPU opt-in and real GL imports")
def test_actual_gpu_sphere_regions():
    from pathlib import Path
    import json, math, sys
    import numpy as np
    import demo_pluck_gl as renderer
    from OpenGL.GL import (GL_COMPUTE_SHADER, GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW, glBindBuffer, glBindBufferBase, glBufferData, glDeleteBuffers, glDeleteProgram, glDispatchCompute, glFinish, glGenBuffers, glGetBufferSubData, glGetUniformLocation, glMemoryBarrier, glUniform1i, glUseProgram)

    probe='''
    layout(std430,binding=13) readonly buffer ProbeIn {vec4 rays[];};
    layout(std430,binding=14) buffer ProbeOut {vec4 results[];};
    uniform int probeCount;
    void main(){int i=int(gl_GlobalInvocationID.x);if(i>=probeCount||gl_GlobalInvocationID.y!=0u)return;
    vec3 ro=rays[2*i].xyz,rd=rays[2*i+1].xyz;int medium=studio_origin_medium(ro,rd);
    results[i*25]=vec4(float(medium),0,0,0);
    for(int j=0;j<8;++j){float distance;int hit=nearest_hit(ro,rd,distance);
    if(hit<0){results[i*25+1+3*j]=vec4(-1);break;}
    vec3 hp=ro+rd*distance;Tri t=studio_hit_tri(hit,hp);float factor;int before=medium;
    bool valid=studio_boundary_step(t,rd,299792458./550e-9,.999999,medium,factor);
    results[i*25+1+3*j]=vec4(float(hit),float(before),float(medium),valid?1:0);
    results[i*25+2+3*j]=vec4(hp,distance);
    results[i*25+3+3*j]=vec4(t.normal.xyz,factor);
    if(!valid)break;ro=hp+rd*1e-6;}}
    '''

    def run(source,scale,with_triangle=False):
        # These are material/media owners only, never surrogate polygon shells.
        tri=np.zeros((3 if with_triangle else 2,3,3),np.float32)
        if with_triangle:tri[2]=np.array([[-.5,-.5,.6],[.5,-.5,.6],[0,.5,.6]])*scale
        g=np.zeros((len(tri),16),np.float32);g[:,:3]=tri[:,0]
        g[:,4:7]=tri[:,1]-tri[:,0];g[:,8:11]=tri[:,2]-tri[:,0]
        g[0,3]=1;g[0,7]=2;g[1,3]=2;g[1,7]=1;g[0,11]=4097
        g[:,14]=1
        shade=np.zeros_like(g)
        if with_triangle:shade[2,7]=1
        mat=np.zeros((32,12),np.float32);mat[0,0]=299792458./550e-9;mat[0,7]=1.5
        regions=np.array([[0,0,0,scale,0,0,0,0],[0,0,0,.25*scale,1,0,0,0]],np.float32)
        nodes,ids=renderer._build_gpu_bvh(tri)
        origins=np.array([[0,0,-2],[0,0,-.5],[0,0,0],[1.01,0,-2],[.99,0,-2],[1,0,-2],[.2,0,-.3],[0,0,.3]],np.float32)*scale
        rays=np.zeros((len(origins)*2,4),np.float32);rays[::2,:3]=origins;rays[1::2,2]=1
        source=source.replace('void main() {','void unused_production_main() {',1)+probe
        program=renderer._prog((source,GL_COMPUTE_SHADER));buffers=[]
        try:
            for binding,data in ((0,g),(1,nodes),(2,ids),(9,shade),(10,mat),(12,regions),(13,rays),(14,np.full((len(origins)*25,4),-99,np.float32))):
                b=int(glGenBuffers(1));buffers.append(b);glBindBuffer(GL_SHADER_STORAGE_BUFFER,b)
                glBufferData(GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL_STATIC_DRAW);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,binding,b)
            glUseProgram(program)
            for key,value in [('uNodeCount',len(nodes)),('uStudioSphereCount',2),('probeCount',len(origins))]:glUniform1i(glGetUniformLocation(program,key),value)
            glDispatchCompute(1,1,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);glFinish()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[-1]);out=np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,len(origins)*25*16),np.float32).reshape(len(origins),25,4).copy()
            assert list(out[:4,0,0])==[-1,0,-1,-1],out[:,0]
            if with_triangle:
                assert out[7,1,0]==2 and out[7,1,3]==0,out[7]
                np.testing.assert_allclose(out[7,2,2],.6*scale,atol=1e-7)
            else:
                for j,pair in enumerate([(-1,0),(0,-1),(-1,0),(0,-1)]):
                    assert tuple(out[0,1+3*j,1:3].astype(int))==pair,out[0]
                    assert out[0,1+3*j,3]==1
                    np.testing.assert_allclose(out[0,2+3*j,2],[-1,-.25,.25,1][j]*scale,atol=2e-6*scale)
                assert out[0,13,0]==-1
            assert out[3,1,0]==-1
            np.testing.assert_allclose(out[4,2,:3]/scale,[.99,0,-math.sqrt(1-.99**2)],atol=3e-6)
            np.testing.assert_allclose(out[4,3,:3],out[4,2,:3]/scale,atol=3e-6)
            np.testing.assert_allclose(out[5,2,:3]/scale,[1,0,0],atol=3e-6)
            assert tuple(out[6,1,1:3].astype(int))==(0,0),'TIR must retain glass'
            assert out[6,3,3]==1
            return out
        finally:glUseProgram(0);glDeleteBuffers(len(buffers),buffers);glDeleteProgram(program)

    renderer._ensure_worker_gl_context()
    receipts=[]
    for scale in (1.,.01):
        for interior in (False,True):
            outputs=[]
            for name,source in [('forward',renderer._GPU_RAY_FIELD_CS),('sensor',renderer._GPU_SENSOR_CS)]:
                out=run(source,scale,interior);outputs.append(out)
                print(f'PASS {name}: scale={scale}, interior={interior}; roots/silhouette/tangent/miss, radial normal, media/TIR, ordinary nearest triangle.',flush=True)
            np.testing.assert_array_equal(outputs[0],outputs[1])
            receipts.append({'scale':scale,'interior':interior,'forward_sensor_equal':True,'events':outputs[0].tolist()})

