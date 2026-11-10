"""Opt-in isolated NVIDIA/GL region gate; run only with an idle GPU."""
import math
import numpy as np
import demo_pluck_gl as renderer
from OpenGL import GL
from camera_designer.studio_region_gpu import region_boundary_helpers
from camera_designer.studio_sphere_regions import STUDIO_SPHERE_GLSL


def run(scale=1.0, bubble_radius=None, translation=(0,0,0), native_materials=False):
    renderer._ensure_worker_gl_context()
    print('GPU',GL.glGetString(GL.GL_RENDERER),flush=True)
    for name,source in [('forward',renderer._GPU_RAY_FIELD_REGION_CS),('camera',renderer._GPU_SENSOR_REGION_CS)]:
        program=renderer._prog((source,GL.GL_COMPUTE_SHADER))
        assert GL.glGetProgramiv(program,GL.GL_LINK_STATUS)
        GL.glDeleteProgram(program)
        print('PASS complete region '+name+' shader linked',flush=True)
    prefix='''#version 430 core
    layout(local_size_x=1) in;
    struct TriGeom {vec4 v0;vec4 e1;vec4 e2;vec4 normal;};
    struct Tri {vec4 v0;vec4 e1;vec4 e2;vec4 normal;vec4 mat_in;vec4 mat_out;vec4 albedo;vec4 emissive;};
    layout(std430,binding=0) readonly buffer G {TriGeom geom_tris[];};
    layout(std430,binding=10) readonly buffer M {float mat_buf[];};
    vec3 last_hit=vec3(0);
    Tri load_tri(int i){Tri t;t.v0=geom_tris[i].v0;t.e1=geom_tris[i].e1;t.e2=geom_tris[i].e2;
    t.normal=geom_tris[i].normal;t.mat_in=vec4(0);t.mat_out=vec4(0);t.albedo=vec4(0);t.emissive=vec4(0);return t;}
    '''
    helper=region_boundary_helpers().replace('Tri b=load_tri(hit);','Tri b=studio_hit_tri(hit,ro+rd*t);')
    sphere_source=STUDIO_SPHERE_GLSL.replace('bool studio_sphere_hit(', 'float studio_region_epsilon(vec3 p);\nbool studio_sphere_hit(',1).replace('1e-6*ray_length/sphere.w','studio_region_epsilon(ro)*ray_length/sphere.w')
    source=prefix+sphere_source+helper+'''
    int nearest_boundary(vec3 ro,vec3 rd,out float best){
      best=1e30;int hit=-1;
      for(int i=0;i<uStudioSphereCount;++i){
        int ti=int(studio_spheres[i].owner.x);
        if(studio_region_filter>=0 && studio_region_id(geom_tris[ti].v0.w)!=studio_region_filter)continue;
        float t;if(studio_sphere_hit(i,ro,rd,t)&&t<best){best=t;hit=ti;}
      }
      last_hit=ro+rd*best;return hit;
    }
    layout(std430,binding=1) readonly buffer I {vec4 inputs[];};
    layout(std430,binding=2) buffer O {vec4 outputs[];};
    void main(){int i=int(gl_GlobalInvocationID.x);vec4 a=inputs[3*i],b=inputs[3*i+1],c=inputs[3*i+2];
      vec3 rd=normalize(b.xyz);float f=299792458.0/550e-9;
      int medium=studio_origin_medium(a.xyz,rd),before=medium;float factor=1.0;bool valid=true;
      float attenuation=studio_medium_transmittance(medium,f,.05);
      if(a.w>0.5){Tri t=studio_hit_tri(int(c.w),c.xyz);studio_region_hit_position=c.xyz;
        valid=studio_boundary_step(t,rd,f,b.w,medium,factor);}
      outputs[3*i]=vec4(float(before),float(medium),attenuation,valid?1.:0.);
      outputs[3*i+1]=vec4(rd,factor);
      outputs[3*i+2]=vec4(float(studio_active[0]),studio_medium_transmittance(medium,f,.05),0,0);
    }'''
    dirs=[0,0,1];cases=[]
    for p in ([0,0,-2],[0,0,-.7],[0,0,-.3],[0,0,0],[-.5,0,0]):cases.extend([[*p,0],[*dirs,.999],[0,0,0,0]])
    cases.extend([[0,0,-.2,1],[0,0,1,.999],[0,0,-.1,3]])
    cases.extend([[0,0,0,1],[0,0,1,.999],[0,0,.1,3]])
    d=np.array([math.sin(math.radians(50)),0,math.cos(math.radians(50))]);p=np.array([0,0,1])-d*.05
    cases.extend([[*p,1],[*d,.999],[0,0,1,0]])
    cases.extend([[0,0,-1.1,1],[0,0,1,0],[0,0,-1,0]])
    cases=np.array(cases,np.float32);n=len(cases)//3
    cases[0::3,:3]*=scale
    cases[2::3,:3]*=scale
    geom=np.zeros((4,16),np.float32);geom[:,3]=[-1,-7,-11,-16];geom[:,7]=[2,3,4,1];geom[0,11]=4097
    spheres=np.array([[0,0,0,1,0,0,0,0],[-.2,0,0,.5,1,0,0,0],[.2,0,0,.5,2,0,0,0],[0,0,0,.1,3,0,0,0]],np.float32)
    spheres[:,:4]*=scale
    if bubble_radius is not None:
        spheres[3,3]=bubble_radius
        cases[5*3+2,2]=-bubble_radius
        cases[6*3+2,2]=bubble_radius
    cases[0::3,:3]+=np.asarray(translation,np.float32)
    cases[2::3,:3]+=np.asarray(translation,np.float32)
    spheres[:,:3]+=np.asarray(translation,np.float32)
    records=np.array([[0,0,0,0],[2,1,0,0],[2,2,0,0],[3,-1,0,0]],np.float32)
    mat=np.zeros((3*32,12),np.float32)
    for i,sigma in enumerate((.1,2,3)):
        mat[i*32,0]=299792458/550e-9;mat[i*32,7]=1.52;mat[i*32,8]=sigma*550e-9/(4*math.pi)
    if native_materials:
        from canonical_material_documents import MaterialDocument, native_payload
        from camera_designer.studio_material_authoring import validate_canonical_volume_optics
        from material_db import MaterialDatabase
        db=MaterialDatabase();canonical={}
        for i,sigma in enumerate((.1,2,3)):
            text=(f"ior: 1.52\nspectral_bands: [{{center_hz: {299792458/550e-9}, bandwidth_hz: 1.0e14, ior_real: 1.52, ior_imag: {sigma*550e-9/(4*math.pi)}}}]\n")
            canonical[i]=native_payload(MaterialDocument(f'gpu_volume_{i}','gpu-test',text.encode()))
            db.register(str(i),canonical[i].payload)
        validate_canonical_volume_optics({'version':1,'source':'native_spectral_bands'},canonical,[0,1,2],
            [{'material':0,'mode':'host'},{'material':1,'mode':'injection_dye'},{'material':2,'mode':'injection_dye'}], [.55])
        mat=np.ascontiguousarray(db.build_mat_buf(freq_hz=np.array([299792458/550e-9])),np.float32)
        print('Using canonical source -> native database -> actual MatBuf n/k',flush=True)
    program=renderer._prog((source,GL.GL_COMPUTE_SHADER));buffers=[]
    try:
        for binding,data in ((0,geom),(1,cases),(2,np.zeros((n*3,4),np.float32)),(10,mat),(12,spheres),(15,records)):
            buf=int(GL.glGenBuffers(1));buffers.append(buf);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buf)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,buf)
        GL.glUseProgram(program)
        GL.glUniform1i(GL.glGetUniformLocation(program,'uStudioRegionCount'),4)
        GL.glUniform1i(GL.glGetUniformLocation(program,'uStudioSphereCount'),4)
        GL.glDispatchCompute(n,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[2]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,n*48),np.float32).reshape(n,3,4)
        np.testing.assert_array_equal(out[:,0,3],1)
        np.testing.assert_array_equal(out[:,0,0],[-1,0,0,-1,0,0,-1,0,-1])
        np.testing.assert_array_equal(out[:,0,1],[-1,0,0,-1,0,-1,0,0,-1])
        np.testing.assert_allclose(out[:5,0,2],np.exp(-.05*np.array([0,.1,5.1,0,2.1])),rtol=3e-6)
        np.testing.assert_array_equal(out[:,2,0],[0,1,7,15,3,15,7,1,0])
        assert out[7,1,2]<0 and out[8,1,2]<0
        print(f'PASS 9 GPU cases at host radius {scale}m: sphere containment, overlap absorption, bubble cuts, restoration, TIR/reflection membership.',flush=True)
    finally:
        GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)

if __name__=='__main__':
    for radius in (1., .025, .001):run(radius)
    # Strict release gate, not xfail: submicron air at actual studio position.
    run(.001, bubble_radius=.4e-6, translation=(0,0,.18))
