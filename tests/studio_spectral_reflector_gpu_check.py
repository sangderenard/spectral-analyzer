"""Explicit short GPU reflector algebra gate; run only when user GPU is idle."""
import numpy as np
from OpenGL import GL
import demo_pluck_gl as renderer
from camera_designer.studio_spectral_reflector import authored_reflector_bands,STUDIO_REFLECTOR_GLSL
from mat_flags import MAT_FLAG_SPECTRAL_REFLECTOR,MAT_FLAG_ROUGH_CONDUCTOR,MAT_FLAG_NATIVE_CONDUCTOR_NK,BINDING_STUDIO_NATIVE_PBR
from material_db import MaterialDatabase

def run():
    renderer._ensure_worker_gl_context()
    db=MaterialDatabase();row=np.zeros(16,np.float32);row[:3]=[1,.72,.29];row[4]=1;row[6]=1.5;row[7]=1
    db.register('artistic_gold_test',{'spectral_bands':authored_reflector_bands(row,[.65,.55,.45])})
    mat=np.ascontiguousarray(db.build_mat_buf(freq_hz=np.array([299792458/.65e-6,299792458/.55e-6,299792458/.45e-6])),np.float32)
    source='''#version 430 core
layout(local_size_x=1) in;
struct Tri {vec4 normal;vec4 emissive;};
layout(std430,binding=10) readonly buffer M {float mat_buf[];};
layout(std430,binding=0) buffer O {vec4 results[];};
'''+f'#define MAT_FLAG_SPECTRAL_REFLECTOR {MAT_FLAG_SPECTRAL_REFLECTOR}u\n#define MAT_FLAG_ROUGH_CONDUCTOR {MAT_FLAG_ROUGH_CONDUCTOR}u\n#define MAT_FLAG_NATIVE_CONDUCTOR_NK {MAT_FLAG_NATIVE_CONDUCTOR_NK}u\n#define STUDIO_NATIVE_PBR_BINDING {BINDING_STUDIO_NATIVE_PBR}\n'+STUDIO_REFLECTOR_GLSL+'''
void main(){int i=int(gl_GlobalInvocationID.x);float wavelengths[3]=float[3](650e-9,550e-9,450e-9);Tri t;t.normal=vec4(0);t.emissive=vec4(0);float f=299792458./wavelengths[i];float normal=studio_reflector_rho(t,f,1.);float grazing=studio_reflector_rho(t,f,0.);float camera=normal/.98;results[i]=vec4(normal,grazing,.98*camera,normal*normal);}
'''
    program=renderer._prog((source,GL.GL_COMPUTE_SHADER));assert GL.glGetProgramiv(program,GL.GL_LINK_STATUS);buffers=[]
    try:
        for binding,data in [(10,mat),(0,np.zeros((3,4),np.float32))]:
            b=int(GL.glGenBuffers(1));buffers.append(b);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,b);GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,b)
        GL.glUseProgram(program);GL.glDispatchCompute(3,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish();GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[1]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,48),np.float32).reshape(3,4)
        assert out[0,0]>out[1,0]>out[2,0];np.testing.assert_allclose(out[:,1],1);np.testing.assert_allclose(out[:,2],out[:,0],rtol=1e-6)
        np.testing.assert_allclose(out[:,3],out[:,0]**2,rtol=1e-6)
        print('PASS GPU normal/grazing spectral reflectance, matched camera roulette expectation and two-bounce attenuation:',out.tolist(),flush=True)
    finally:GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)
if __name__=='__main__':run()
