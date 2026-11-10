"""Bounded real-shader area-source check; no production jobs or requests."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np


def main():
    import demo_pluck_gl as renderer
    from OpenGL import GL as gl
    from camera_designer.studio_native_emitters import native_emissive_sources,area_nee_records
    from ray_tracer_bridge import per_tri_spectral_to_mat_buf
    from mat_flags import MAT_FLAG_EMISSIVE
    renderer._ensure_worker_gl_context()
    programs=[];buffers=[];results=[]
    try:
        for source in (renderer._GPU_SENSOR_CS,renderer._GPU_SENSOR_REGION_CS,renderer._GPU_RAY_FIELD_CS):
            program=renderer._prog((source,gl.GL_COMPUTE_SHADER));programs.append(program)
            assert gl.glGetProgramiv(program,gl.GL_LINK_STATUS)
        probe=renderer._GPU_SENSOR_CS.replace('void main() {','void unused_main() {',1)+r'''
        layout(std430,binding=14) buffer Result {vec4 result[];};
        uniform vec3 testPoint;
        void main(){
            uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*256u;
            uint rng=hash_u(i+17u);float f;vec3 bands;
            sample_direct_light(testPoint,vec3(-1,0,0),rng,f,bands);
            result[i]=vec4(bands,1);
        }
        '''
        program=renderer._prog((probe,gl.GL_COMPUTE_SHADER));programs.append(program)
        assert gl.glGetProgramiv(program,gl.GL_LINK_STATUS)
        triangles=np.array([[[0,-.01,-.01],[0,.01,-.01],[0,0,.01]]],float)
        freq=299792458./np.array([450e-9,550e-9,650e-9]);zero=np.zeros((1,3))
        mids,materials,_=per_tri_spectral_to_mat_buf(zero,zero,zero,freq,emission_bands=np.array([[1.,2.,3.]]))
        geom=np.zeros((1,16),np.float32);geom[:,:3]=triangles[:,0];geom[:,4:7]=triangles[:,1]-triangles[:,0];geom[:,8:11]=triangles[:,2]-triangles[:,0]
        geom[:,12]=1;geom[:,15]=np.array([MAT_FLAG_EMISSIVE],np.uint32).view(np.float32)
        shade=np.zeros((1,16),np.float32);shade[:,14]=mids.astype(np.uint32).view(np.float32)
        scene=(geom,shade,materials,triangles)
        for binding,data in [(5,np.zeros((1,12),np.float32)),(14,np.zeros((65536,4),np.float32))]:
            buf=int(gl.glGenBuffers(1));buffers.append(buf);gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buf)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,gl.GL_STATIC_DRAW);gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,buf)
        gl.glUseProgram(program)
        for key,value in [('uNodeCount',0),('uStudioSphereCount',0),('uNEEAreaSources',1),('uNEEFullSpectrum',1),('uFixed3Transport',1)]:
            gl.glUniform1i(gl.glGetUniformLocation(program,key),value)
        for budget,point,flip in [(120,[1,0,0],False),(1200,[1,0,0],False),(1200,[2,0,0],False),(1200,[1,0,0],True)]:
            sources=native_emissive_sources(scene,[.45,.55,.65],ray_budget=budget)
            records=area_nee_records(sources,*scene[:4])
            if flip:records[:,[3,7,11]]*=-1
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[0]);gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,records.nbytes,records,gl.GL_STATIC_DRAW)
            gl.glUniform1i(gl.glGetUniformLocation(program,'uSourceCount'),len(records));gl.glUniform3f(gl.glGetUniformLocation(program,'testPoint'),*point)
            gl.glDispatchCompute(16,16,1);gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT);gl.glFinish()
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[1]);pixels=np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,65536*16),np.float32).reshape(-1,4)[:,:3].copy()
            delta=records[:,:3].astype(float)-point;d2=(delta*delta).sum(1);wi=delta/np.sqrt(d2[:,None])
            cos_light=np.maximum(0,(records[:,[3,7,11]]*(-wi)).sum(1));cos_surface=np.maximum(0,-wi[:,0])
            reference=(records[:,4:7]*(cos_light*cos_surface/d2*len(records)*records[:,10]/np.pi)[:,None]).mean(0)
            measured=pixels.astype(float).mean(0)
            np.testing.assert_allclose(measured,reference,rtol=2e-5,atol=1e-10)
            results.append(dict(roots=len(records),point=point,backface=flip,reference=reference.tolist(),measured=measured.tolist()))
        np.testing.assert_allclose(results[0]['measured'],results[1]['measured'],rtol=2e-5)
        np.testing.assert_allclose(np.array(results[1]['measured'])/results[2]['measured'],4,rtol=2e-4)
        assert results[3]['measured']==[0,0,0]
        receipt=dict(renderer=gl.glGetString(gl.GL_RENDERER).decode(),shader_links=3,samples_per_case=65536,cases=results,
            limits='Actual area NEE shader, unoccluded fixture; not complete scene or caustic validation')
        output=ROOT/'build/dot-native-area-20261110';output.mkdir(parents=True,exist_ok=True)
        (output/'gpu-check.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
    finally:
        gl.glUseProgram(0)
        if buffers:gl.glDeleteBuffers(len(buffers),buffers)
        for program in programs:gl.glDeleteProgram(program)

if __name__=='__main__':main()
