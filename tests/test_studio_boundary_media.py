"""Explicit media ownership and optional real GPU glass/air invariants.

Run the GPU function directly on a real context (the repo conftest stubs GL):
python -c "from tests.test_studio_boundary_media import test_actual_gpu_interfaces as t;t()"
"""
import math
import os
import numpy as np
import pytest
from camera_designer.studio_boundary_media import (
    validate_boundary_ranges, install_boundary_pairs, studio_medium_extinction,
    STUDIO_BOUNDARY_GLSL,
)


def test_original_triangle_membership_survives_filtering():
    spec={'version':1,'ranges':[{'start':0,'count':2,'positive':-1,'negative':5},
                               {'start':2,'count':1,'positive':5,'negative':-1}]}
    pairs=validate_boundary_ranges(spec,4,6)
    np.testing.assert_array_equal(pairs,[[-1,5],[-1,5],[5,-1],[-2,-2]])
    geom=np.zeros((3,16),np.float32)
    install_boundary_pairs(geom,pairs,[True,False,True,True],{5:20})
    np.testing.assert_array_equal(geom[:,[3,7]],[[1,22],[22,1],[0,0]])


@pytest.mark.parametrize('spec',[
    {'version':2,'ranges':[]},
    {'version':1,'ranges':[{'start':0,'count':4,'positive':-1,'negative':0}]},
    {'version':1,'ranges':[{'start':0,'count':1,'positive':-1,'negative':100}]},
    {'version':1,'ranges':[{'start':0,'count':1,'positive':-1,'negative':0}]*2},
    {'version':1,'ranges':[{'start':False,'count':1,'positive':-1,'negative':0}]},
])
def test_corrupt_membership_rejected(spec):
    with pytest.raises(ValueError):validate_boundary_ranges(spec,3,1)


def test_equal_index_is_permitted_without_an_artificial_interface():
    spec={'version':1,'ranges':[{'start':0,'count':1,'positive':0,'negative':0}]}
    np.testing.assert_array_equal(validate_boundary_ranges(spec,1,1),[[0,0]])


def test_existing_rgb_tint_becomes_segment_extinction():
    row=np.zeros(16,np.float32);row[:3]=1;row[5]=1;row[6]=1.5
    assert studio_medium_extinction(row,.55)==0
    row[:3]=[1,0,0]
    assert studio_medium_extinction(row,.55)>studio_medium_extinction(row,.65)>0


@pytest.mark.skipif(os.environ.get('MARBLE_BOUNDARY_GPU_TEST')!='1',reason='explicit real GPU context required')
def test_actual_gpu_interfaces():
    import demo_pluck_gl as renderer
    from camera_designer.compound_optics import _snell
    from OpenGL import GL
    renderer._ensure_worker_gl_context()
    # Compile and dispatch exactly the same interface helper used in both complete
    # staged production shaders. Hand-picked xi forces reflection/transmission.
    source='''#version 430 core
    layout(local_size_x=1) in;
    struct TriGeom {vec4 v0;vec4 e1;vec4 e2;vec4 normal;};
    struct Tri {vec4 v0;vec4 e1;vec4 e2;vec4 normal;vec4 mat_in;vec4 mat_out;vec4 albedo;vec4 emissive;};
    layout(std430,binding=0) readonly buffer G {TriGeom geom_tris[];};
    layout(std430,binding=10) readonly buffer M {float mat_buf[];};
    Tri load_tri(int i){Tri t;t.v0=geom_tris[i].v0;t.e1=geom_tris[i].e1;t.normal=geom_tris[i].normal;t.mat_out=vec4(0);return t;}
    int nearest_boundary(vec3 ro,vec3 rd,out float best){best=1e30;return -1;}
    '''+STUDIO_BOUNDARY_GLSL+'''
    layout(std430,binding=1) readonly buffer Inputs {vec4 input_rows[];};
    layout(std430,binding=2) buffer Outputs {vec4 output_rows[];};
    void main(){int i=int(gl_GlobalInvocationID.x);vec4 a=input_rows[3*i],b=input_rows[3*i+1],c=input_rows[3*i+2];
    Tri t;t.v0=vec4(0);t.e1=vec4(0);t.e2=vec4(0);t.mat_in=vec4(0);t.albedo=vec4(0);t.emissive=vec4(0);t.v0.w=b.x+2.;t.e1.w=b.y+2.;t.normal=vec4(0,0,c.x,0);t.mat_out=vec4(0);
    vec3 d=a.xyz;int medium=int(b.z);float factor;
    bool valid=studio_boundary_step(t,d,a.w,b.w,medium,factor);
    output_rows[2*i]=vec4(d,float(medium));
    output_rows[2*i+1]=vec4(factor,studio_medium_transmittance(medium,a.w,.1),valid?1.:0.,0.);
    }'''
    f=299792458./550e-9
    # Positive/negative medium IDs: -1 air, 0 glass. normal +/-Z.
    cases=[(30,-1,0,-1,.999,-1),(30,0,-1,0,.999,-1),(30,0,-1,-1,.999,1),
           (30,-1,0,0,.999,1),(50,0,-1,0,.999,-1),(30,0,-1,0,0.,-1),
           (0,-1,0,-1,.02,-1),(0,-1,0,-1,.05,-1),(30,0,-1,-1,.999,-1)]
    inputs=[]
    for angle,pos,neg,medium,xi,n in cases:
        inputs += [[math.sin(math.radians(angle)),0,math.cos(math.radians(angle)),f],[pos,neg,medium,xi],[n,0,0,0]]
    inputs=np.array(inputs,np.float32)
    mat=np.zeros((32,12),np.float32);mat[0,0]=f;mat[0,7]=1.5;mat[0,8]=2.*550e-9/(4*math.pi)
    program=renderer._prog((source,GL.GL_COMPUTE_SHADER));buffers=[]
    try:
        for binding,data in ((0,np.zeros((1,16),np.float32)),(1,inputs),(2,np.zeros((len(cases)*2,4),np.float32)),(10,mat)):
            b=int(GL.glGenBuffers(1));buffers.append(b);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,b)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,b)
        GL.glUseProgram(program);GL.glDispatchCompute(len(cases),1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[2])
        out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,len(cases)*32),np.float32).reshape(len(cases),2,4)
        expected_media=[0,-1,0,-1,0,0,-1,0,-1]
        for i,(angle,pos,neg,medium,xi,n) in enumerate(cases):
            if i==8:assert out[i,1,2]==0;continue
            assert out[i,1,2]==1
            assert int(out[i,0,3])==expected_media[i],(i,out[i])
            d=inputs[i*3,:3].astype(float);normal=np.array([0,0,n],float)
            ni=1.5 if medium==0 else 1.;nt=1.5 if expected_media[i]==0 else 1.
            if i in (4,5,6):expected=d-2*np.dot(d,normal)*normal;expected_factor=1.
            else:expected=_snell(d/np.linalg.norm(d),normal,ni,nt);expected_factor=(ni/nt)**2
            np.testing.assert_allclose(out[i,0,:3],expected,atol=3e-7)
            np.testing.assert_allclose(out[i,1,0],expected_factor,rtol=3e-7)
            np.testing.assert_allclose(out[i,1,1],math.exp(-.2) if expected_media[i]==0 else 1.,rtol=3e-7)
        print('PASS actual GPU: air/glass/bubble interfaces in both directions; Fresnel forced reflection/transmission; TIR keeps glass; mismatch rejected; air unattenuated; host Beer attenuation. 9 cases.',flush=True)
    finally:
        GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)
