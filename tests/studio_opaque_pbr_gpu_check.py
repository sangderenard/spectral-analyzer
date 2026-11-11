"""Bounded actual shared GLSL BSDF furnace/PDF/budget proof."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import demo_pluck_gl as r
    from OpenGL import GL
    from camera_designer.studio_opaque_pbr import with_opaque_pbr
    r._ensure_worker_gl_context()
    source=with_opaque_pbr(r._GPU_RAY_FIELD_CS).replace('void main() {','void original_main() {',1)
    source+=r'''
layout(std430,binding=21) buffer TestResults {vec4 result[];};
uniform float test_diffusion;
uniform float test_energy;
void main(){
 uint id=gl_GlobalInvocationID.x;if(id>=65536u)return;
 uint rng=hash_u(id+197u);
 Tri t;t.mat_out=vec4(.0048,test_diffusion,.0352,1);t.mat_in=vec4(.0048,test_diffusion,.0352,1.5);
 t.normal=vec4(0,0,1,uintBitsToFloat(0u));t.albedo=vec4(.9,.9,.9,0);t.emissive=vec4(0);
 vec3 wi=vec3(0,0,1),wo;float pdf;bool delta;
 bool valid=studio_pbr_sample(t,wi,vec3(0,0,1),vec3(rand01(rng),rand01(rng),rand01(rng)),wo,pdf,delta);
 float weight=valid?studio_pbr_weight(t,wi,wo,vec3(0,0,1),studio_band_frequency(0),pdf,delta):0.0;
 float p1,p2;float f1=valid?studio_pbr_eval(t,wi,wo,vec3(0,0,1),studio_band_frequency(0),p1):0.0;
 float f2=valid?studio_pbr_eval(t,wo,wi,vec3(0,0,1),studio_band_frequency(0),p2):0.0;
 result[id]=vec4(weight*test_energy,valid?pdf:0.0,abs(f1-f2),studio_pbr_opaque(t)?1.0:0.0);
}
'''
    program=r._prog((source,GL.GL_COMPUTE_SHADER));assert GL.glGetProgramiv(program,GL.GL_LINK_STATUS)
    buffers=GL.glGenBuffers(2);pbr=np.zeros(16,np.float32);pbr[:3]=.9;pbr[3]=.96;pbr[6]=1.5;pbr[7]=1
    try:
      GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,int(buffers[0]));GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,pbr.nbytes,pbr,GL.GL_STATIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,18,int(buffers[0]))
      GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,int(buffers[1]));GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,65536*16,None,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,21,int(buffers[1]))
      GL.glUseProgram(program);results=[]
      for d in [1.,.96,0.]:
        base=None
        for energy in [1.,1/32768]:
          GL.glUniform1f(GL.glGetUniformLocation(program,'test_diffusion'),d);GL.glUniform1f(GL.glGetUniformLocation(program,'test_energy'),energy)
          GL.glDispatchCompute(512,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
          GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,int(buffers[1]));a=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,65536*16),np.float32).reshape(-1,4).copy()
          assert np.isfinite(a).all() and np.all(a[:,3]==1) and np.max(a[:,2])<2e-6
          if base is None:base=a[:,0].copy()
          else:np.testing.assert_allclose(a[:,0]/energy,base,rtol=2e-6,atol=1e-9)
          mean=float((a[:,0]/energy).mean());assert 0<=mean<=1
          if d==1:np.testing.assert_allclose(a[:,0]/energy,.9,rtol=2e-6)
          if d==.96:assert .85<mean<.88
          results.append(dict(diffusion=d,source_energy=energy,hemispherical_mean=mean,max_reciprocity_error=float(a[:,2].max())))
      print(json.dumps(dict(result='PASS',renderer=GL.glGetString(GL.GL_RENDERER).decode(),results=results)))
    finally:GL.glDeleteBuffers(2,buffers);GL.glDeleteProgram(program)
if __name__=='__main__':main()
