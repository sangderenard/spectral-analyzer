"""Bounded NVIDIA rod-cut gate. Run only without an active user render."""
import math
import numpy as np
import demo_pluck_gl as renderer
from OpenGL import GL
from camera_designer.studio_region_gpu import region_boundary_helpers
from camera_designer.studio_sphere_regions import STUDIO_SPHERE_GLSL
from camera_designer.studio_rod_bundle_gpu import pack_bundle_records


def source():
    prefix='''#version 430 core
    layout(local_size_x=1) in;
    struct TriGeom {vec4 v0;vec4 e1;vec4 e2;vec4 normal;};
    struct Tri {vec4 v0;vec4 e1;vec4 e2;vec4 normal;vec4 mat_in;vec4 mat_out;vec4 albedo;vec4 emissive;};
    layout(std430,binding=0) readonly buffer G {TriGeom geom_tris[];};
    layout(std430,binding=10) readonly buffer M {float mat_buf[];};
    Tri load_tri(int i){Tri t;t.v0=geom_tris[i].v0;t.e1=geom_tris[i].e1;t.e2=geom_tris[i].e2;
    t.normal=geom_tris[i].normal;t.mat_in=vec4(0);t.mat_out=vec4(0);t.albedo=vec4(0);t.emissive=vec4(0);return t;}
    '''
    helper=region_boundary_helpers().replace('Tri b=load_tri(hit);','Tri b=studio_hit_tri(hit,ro+rd*t);')
    sphere=STUDIO_SPHERE_GLSL.replace('bool studio_sphere_hit(', 'float studio_region_epsilon(vec3 p);\nbool studio_sphere_hit(',1).replace('1e-6*ray_length/sphere.w','studio_region_epsilon(ro)*ray_length/sphere.w')
    return prefix+sphere+helper+'''
    int nearest_boundary(vec3 ro,vec3 rd,out float best){
        best=1e30;int hit=-1;
        for(int i=0;i<uStudioSphereCount;++i){int ti=int(studio_spheres[i].owner.x);
            if(studio_region_filter>=0 && studio_region_id(geom_tris[ti].v0.w)!=studio_region_filter)continue;
            float t;if(studio_sphere_hit(i,ro,rd,t)&&t<best){best=t;hit=ti;}}
        return hit;
    }
    layout(std430,binding=1) readonly buffer I {vec4 inputs[];};
    layout(std430,binding=2) buffer O {vec4 outputs[];};
    void main(){int i=int(gl_GlobalInvocationID.x);vec4 a=inputs[i*2],b=inputs[i*2+1];
        vec3 rd=normalize(b.xyz);int medium=studio_origin_medium(a.xyz,rd),before=medium;
        float best=a.w;int cut=studio_partition_nearest(a.xyz,rd,best);float factor=1.;bool ok=false;
        if(cut>=0){Tri t=load_tri(cut);t.normal.xyz=studio_partition_normal;
            studio_region_hit_position=a.xyz+rd*best;
            ok=studio_boundary_step(t,rd,299792458.0/550e-9,b.w,medium,factor);}
        outputs[i*3]=vec4(float(cut),best,float(before),float(medium));
        outputs[i*3+1]=vec4(rd,factor);outputs[i*3+2]=vec4(float(studio_active[0]),ok?1.:0.,0,0);
    }'''


def run(scale):
    renderer._ensure_worker_gl_context()
    center=np.array([0,0,.18],np.float32)
    geom=np.zeros((3,16),np.float32);geom[:,3]=[-1,-6,-10];geom[:,7]=[2,3,4];geom[0,11]=4097
    spheres=np.zeros((3,8),np.float32);spheres[:,:3]=center;spheres[:,3]=np.array([1,.5,.5])*scale;spheres[:,4]=np.arange(3)
    records=np.array([[0,0,0,0],[1,1,0,0],[1,2,0,0]],np.float32)
    bundle={'regions':[1,2],'eye_length':.75,'eye_curve':0.,'twist':0.,'phase':0.,'frame':{'origin':[0,0,0],'axes':np.eye(3).tolist(),'scale':1.}}
    records=pack_bundle_records(records,bundle,scale_m=scale,center_m=center)
    mat=np.zeros((3*32,12),np.float32)
    for i,ior in enumerate([1.52,1.54,1.6]):mat[i*32,0]=299792458/550e-9;mat[i*32,7]=ior
    cases=[]
    for xi,limit in ((.999,.3),(0.,.3),(.999,.05)):
        p=center+np.array([.2,0,.1])*scale
        cases.extend([[*p,limit*scale],[-1,0,0,xi]])
    # Coincident divider must fail explicitly, not return a false clear path.
    cases.extend([[*center,.3*scale],[0,1,0,.999]])
    cases=np.asarray(cases,np.float32);n=len(cases)//2
    program=renderer._prog((source(),GL.GL_COMPUTE_SHADER));buffers=[]
    try:
        for binding,data in ((0,geom),(1,cases),(2,np.zeros((n*3,4),np.float32)),(10,mat),(12,spheres),(15,records)):
            buf=int(GL.glGenBuffers(1));buffers.append(buf);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buf)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,buf)
        GL.glUseProgram(program);GL.glUniform1i(GL.glGetUniformLocation(program,'uStudioRegionCount'),3);GL.glUniform1i(GL.glGetUniformLocation(program,'uStudioSphereCount'),3)
        GL.glDispatchCompute(n,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[2]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,n*48),np.float32).reshape(n,3,4)
        np.testing.assert_array_equal(out[:2,0,0],0)
        np.testing.assert_allclose(out[:2,0,1],.1*scale,atol=1e-7)
        np.testing.assert_array_equal(out[:2,0,2],[1,1]);np.testing.assert_array_equal(out[:2,0,3],[2,1])
        np.testing.assert_allclose(out[0,1,3],(1.54/1.6)**2,rtol=2e-6)
        np.testing.assert_array_equal(out[:2,2,1],1)
        assert out[1,1,2]<-.99
        assert out[2,0,0]==-1
        assert out[3,0,0]==-2
        print(f'PASS rod separator at {scale}m: internal cut, native refraction/radiance factor, reflection, nearest bound, explicit coincident failure.',flush=True)
    finally:
        GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)

if __name__=='__main__':
    for radius in (.025,.001):run(radius)
