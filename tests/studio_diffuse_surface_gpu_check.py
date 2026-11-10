"""Small isolated GPU reflectance gate; requires idle user renderer."""
import re
import numpy as np
from OpenGL import GL
import demo_pluck_gl as renderer
from camera_designer.studio_diffuse_surface import STUDIO_DIFFUSE_GLSL
from camera_designer.spectral_sensor_basis import FIXED3_SPECTRAL_GLSL

def run():
    renderer._ensure_worker_gl_context()
    for name,full in [('forward',renderer._GPU_RAY_FIELD_CS),('sensor',renderer._GPU_SENSOR_CS),('region-forward',renderer._GPU_RAY_FIELD_REGION_CS),('region-sensor',renderer._GPU_SENSOR_REGION_CS)]:
        p=renderer._prog((full,GL.GL_COMPUTE_SHADER));assert GL.glGetProgramiv(p,GL.GL_LINK_STATUS);GL.glDeleteProgram(p)
        print('PASS full shader link',name,flush=True)
    # Extract the actual forward assignment and camera continuation block.
    forward=re.search(r'bool diffuse_surface = studio_diffuse_surface\(tri\);\s*float reflectivity   = [^;]+;',renderer._GPU_RAY_FIELD_CS).group()
    a=renderer._GPU_SENSOR_CS.index('float survival = diffuse_surface ?')
    b=renderer._GPU_SENSOR_CS.index('float tp_after',a)
    continuation=renderer._GPU_SENSOR_CS[a:b]
    source="#version 430 core\nlayout(local_size_x=1) in;\n"+FIXED3_SPECTRAL_GLSL+"""
#define MAT_FLAG_EMISSIVE 1u
struct Tri {vec4 mat_in;vec4 mat_out;vec4 normal;vec4 albedo;};
layout(std430,binding=0) readonly buffer I {vec4 colors[];};
layout(std430,binding=1) buffer O {vec4 results[];};
float xi;
float rand01(inout uint seed){return xi;}
float layer_transmit_weight(vec3 rgb,int i){return studio_material_weight(rgb,studio_band_frequency(i));}
"""+STUDIO_DIFFUSE_GLSL+"""
void main(){int id=int(gl_GlobalInvocationID.x);Tri tri;
tri.mat_in=vec4(0,1,0,1.5);tri.mat_out=vec4(0,1,0,1);tri.normal=vec4(0,0,1,0);tri.albedo=colors[id];
vec3 sum=vec3(0);vec3 expected=vec3(0);
for(int band=0;band<3;++band){vec4 packet=vec4(studio_band_frequency(band),0,1,0);vec4 mat=tri.mat_out;
"""+forward+"""
expected[band]=reflectivity;
}
for(int si=0;si<10000;++si){xi=(float(si)+0.5)/10000.;
uint rng=0u;int nl=3;float throughput[3]=float[3](1,1,1);vec3 surf_albedo=tri.albedo.rgb;
bool diffuse_surface=studio_diffuse_surface(tri);float refl=1,abso=0;
// The production block breaks on roulette rejection, so put it in a one-pass loop.
for(int once=0;once<1;++once){
"""+continuation+"""
sum+=vec3(throughput[0],throughput[1],throughput[2]);}
}
results[id*2]=vec4(expected,1);results[id*2+1]=vec4(sum/10000.,1);
}
"""
    colors=np.array([[0,0,0,0],[.012,.012,.013,0],[.18,.18,.18,0],[1,1,1,0],[.8,.03,.2,0]],np.float32)
    buffers=[];program=renderer._prog((source,GL.GL_COMPUTE_SHADER));assert GL.glGetProgramiv(program,GL.GL_LINK_STATUS)
    try:
        for binding,data in [(0,colors),(1,np.zeros((len(colors)*2,4),np.float32))]:
            buf=int(GL.glGenBuffers(1));buffers.append(buf);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buf)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,buf)
        GL.glUseProgram(program);GL.glUniform1i(GL.glGetUniformLocation(program,'uFixed3Transport'),1)
        GL.glDispatchCompute(len(colors),1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,buffers[1]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,len(colors)*32),np.float32).reshape(-1,2,4)
        np.testing.assert_allclose(out[:,0,:3],out[:,1,:3],atol=2e-4,rtol=2e-4)
        np.testing.assert_array_equal(out[0,:,:3],0)
        np.testing.assert_allclose(out[2,:,:3],.18,atol=2e-4)
        np.testing.assert_allclose(out[3,:,:3],1,atol=2e-4)
        print('PASS forward reflectance == expected camera roulette weight: black/felt/gray/white/color',out.tolist(),flush=True)
    finally:
        GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),buffers);GL.glDeleteProgram(program)
if __name__=='__main__':run()
