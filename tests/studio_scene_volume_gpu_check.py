"""Small explicit GPU gate for dice-like closed triangle optical hosts.

No analytic sphere is present. Run only after checking current user jobs.
"""
import math
import numpy as np
import demo_pluck_gl as renderer
from OpenGL import GL
from camera_designer.studio_region_gpu import region_boundary_helpers


def cube_triangles(radius):
    triangles=[]
    for axis in range(3):
        u=(axis+1)%3;v=(axis+2)%3
        for sign in (-1,1):
            corners=[]
            for a,b in ((-1,-1),(1,-1),(1,1),(-1,1)):
                p=np.zeros(3);p[axis]=sign*radius;p[u]=a*radius;p[v]=b*radius;corners.append(p)
            for indices in ((0,1,2),(0,2,3)):
                tri=np.array([corners[i] for i in indices])
                if np.dot(np.cross(tri[1]-tri[0],tri[2]-tri[0]),tri.mean(axis=0))<0:tri=tri[[0,2,1]]
                triangles.append(tri)
    return np.array(triangles,np.float32)


def run():
    renderer._ensure_worker_gl_context()
    print('GPU',GL.glGetString(GL.GL_RENDERER),flush=True)
    source='''#version 430 core
    layout(local_size_x=1) in;
    struct TriGeom {vec4 v0;vec4 e1;vec4 e2;vec4 normal;};
    struct Tri {vec4 v0;vec4 e1;vec4 e2;vec4 normal;vec4 mat_in;vec4 mat_out;vec4 albedo;vec4 emissive;};
    layout(std430,binding=0) readonly buffer G {TriGeom geom_tris[];};
    layout(std430,binding=10) readonly buffer M {float mat_buf[];};
    Tri load_tri(int i){Tri t;t.v0=geom_tris[i].v0;t.e1=geom_tris[i].e1;t.e2=geom_tris[i].e2;t.normal=geom_tris[i].normal;
    t.mat_in=vec4(0);t.mat_out=vec4(0);t.albedo=vec4(0);t.emissive=vec4(0);return t;}
    '''+region_boundary_helpers()+'''
    int nearest_boundary(vec3 ro,vec3 rd,out float best){
      best=1e30;int hit=-1;
      for(int i=0;i<geom_tris.length();++i){Tri t=load_tri(i);
        if(studio_region_filter>=0 && !studio_matches_region(t.v0.w,t.e1.w,studio_region_filter))continue;
        vec3 p=cross(rd,t.e2.xyz);float det=dot(t.e1.xyz,p);if(abs(det)<1e-16)continue;
        vec3 s=ro-t.v0.xyz;float u=dot(s,p)/det;if(u<0||u>1)continue;
        vec3 q=cross(s,t.e1.xyz);float v=dot(rd,q)/det;if(v<0||u+v>1)continue;
        float d=dot(t.e2.xyz,q)/det;
        if(d>studio_region_epsilon(ro)&&d<best){best=d;hit=i;}
      }return hit;
    }
    layout(std430,binding=1) readonly buffer I {vec4 inputs[];};
    layout(std430,binding=2) buffer O {vec4 outputs[];};
    void main(){int i=int(gl_GlobalInvocationID.x);vec3 ro=inputs[i].xyz,rd=vec3(0,0,1);float f=299792458./550e-9;
      int before=studio_origin_medium(ro,rd),medium=before;float attenuation=studio_medium_transmittance(medium,f,.001);
      float d;int hit=nearest_boundary(ro,rd,d);float factor=1.;bool ok=false;
      if(hit>=0){studio_region_hit_position=ro+rd*d;ok=studio_boundary_step(load_tri(hit),rd,f,.999,medium,factor);}
      outputs[2*i]=vec4(float(before),float(medium),attenuation,ok?1.:0.);
      outputs[2*i+1]=vec4(float(studio_active[0]),studio_medium_transmittance(medium,f,.001),factor,0);
    }'''
    verts=np.concatenate((cube_triangles(.025),cube_triangles(.008)+np.array([0,0,.06],np.float32)))
    geom=np.zeros((24,16),np.float32);geom[:,:3]=verts[:,0];geom[:,4:7]=verts[:,1]-verts[:,0];geom[:,8:11]=verts[:,2]-verts[:,0]
    normals=np.cross(geom[:,4:7],geom[:,8:11]);geom[:,12:15]=normals/np.linalg.norm(normals,axis=1)[:,None]
    geom[:12,3]=-1;geom[12:,3]=1;geom[:12,7]=2;geom[12:,7]=3;geom[0,11]=4097
    cases=np.array([[.001,.002,-.04,0],[.001,.002,.02,0],[.001,.002,.04,0],[.001,.002,.06,0]],np.float32)
    records=np.array([[0,0,0,0],[4,1,0,0]],np.float32)
    mat=np.zeros((64,12),np.float32)
    for i,sigma in enumerate((10,200)):
        mat[i*32,0]=299792458/550e-9;mat[i*32,7]=1.52;mat[i*32,8]=sigma*550e-9/(4*math.pi)
    program=renderer._prog((source,GL.GL_COMPUTE_SHADER));buffers=[]
    try:
        for binding,data in ((0,geom),(1,cases),(2,np.zeros((8,4),np.float32)),(10,mat),(15,records)):
            buf=int(GL.glGenBuffers(1));buffers.append(buf);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buf)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,buf)
        GL.glUseProgram(program);GL.glUniform1i(GL.glGetUniformLocation(program,'uStudioRegionCount'),2)
        GL.glDispatchCompute(4,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[2]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,128),np.float32).reshape(4,2,4)
        np.testing.assert_array_equal(out[:,0,0],[-1,0,-1,1]);np.testing.assert_array_equal(out[:,0,1],[0,-1,1,-1]);np.testing.assert_array_equal(out[:,0,3],1)
        np.testing.assert_allclose(out[:,0,2],np.exp(-.001*np.array([0,10,0,200])),rtol=3e-6)
        np.testing.assert_array_equal(out[:,1,0],[1,0,2,0])
        np.testing.assert_allclose(out[:,1,1],np.exp(-.001*np.array([10,0,200,0])),rtol=3e-6)
        print('PASS 4 separate subject/room GPU cases: subject entry/exit, external volume entry/exit and its own Beer absorption.',flush=True)
    finally:
        GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)

if __name__=='__main__':run()
