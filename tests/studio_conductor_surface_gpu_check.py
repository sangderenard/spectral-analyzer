"""Isolated kernel parity, not a production transport integration test."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from OpenGL import GL
import demo_pluck_gl as renderer
from camera_designer.studio_conductor_surface import CONDUCTOR_SURFACE_GLSL,conductor_fresnel,sample_ggx

def run():
 renderer._ensure_worker_gl_context();rng=np.random.default_rng(17)
 u=rng.uniform(.0001,.9999,(1024,2)).astype(np.float32)
 u[:8]=np.array([[0,0],[1,0],[1,1],[1,.5]]*2,np.float32)
 cases=np.zeros((len(u),4),np.float32);cases[:,:2]=u;cases[:,2]=np.tile([0,.1,.4,1],len(u)//4)
 source='''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer I{vec4 cases[];};
layout(std430,binding=1) buffer O{vec4 results[];};
'''+CONDUCTOR_SURFACE_GLSL+'''
void main(){uint i=gl_GlobalInvocationID.x;vec4 a=cases[i];vec3 wo;float vh;float w=studio_sample_ggx(vec3(.6,0,.8),a.z,a.xy,wo,vh);results[i*2]=vec4(wo,w);results[i*2+1]=vec4(vh,studio_conductor_fresnel(.2,3.0,vh),0,0);}
'''
 program=renderer._prog((source,GL.GL_COMPUTE_SHADER));assert GL.glGetProgramiv(program,GL.GL_LINK_STATUS);buffers=[]
 try:
  for binding,data in [(0,cases),(1,np.zeros((len(u)*2,4),np.float32))]:
   b=int(GL.glGenBuffers(1));buffers.append(b);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,b);GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,b)
  GL.glUseProgram(program);GL.glDispatchCompute(len(u)//64,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish();GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[1]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,len(u)*32),np.float32).reshape(-1,2,4)
  for rough in [0,.1,.4,1]:
   mask=np.isclose(cases[:,2],rough);wo,w,vh=sample_ggx([.6,0,.8],float(np.float32(rough)),u[mask])
   np.testing.assert_allclose(out[mask,0,:3],wo,atol=2e-5);np.testing.assert_allclose(out[mask,0,3],w,atol=2e-5)
   np.testing.assert_allclose(out[mask,1,1],conductor_fresnel(.2,3,vh),atol=2e-5)
  print('PASS1024CPU/GPU conductor Fresnel/GGX direction and importance-weight comparisons; no production activation',flush=True)
 finally:GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)
if __name__=='__main__':run()
