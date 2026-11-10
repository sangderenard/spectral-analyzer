"""Isolated fixed:3 NEE GPU expectation/variance check; no production requests."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import demo_pluck_gl as r
    from OpenGL import GL as gl
    from camera_designer.spectral_sensor_basis import fixed3_nee_records
    r._ensure_worker_gl_context()
    actual=r._prog((r._GPU_SENSOR_CS,gl.GL_COMPUTE_SHADER))
    assert gl.glGetProgramiv(actual,gl.GL_LINK_STATUS)
    probe=r._GPU_SENSOR_CS.replace('void main() {','void unused_production_main() {',1)+r"""
    layout(std430,binding=14) buffer Results { vec4 result[]; };
    void main() {
        uint index=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*256u;
        uint rng=hash_u(index+17u);float frequency;vec3 bands;
        float direct=sample_direct_light(vec3(0),vec3(0,0,1),rng,frequency,bands);
        vec3 value=vec3(0);
        if(uNEEFullSpectrum!=0)value=bands;
        else value[studio_band_index(frequency)]=direct;
        result[index]=vec4(value,1);
    }
    """
    program=r._prog((probe,gl.GL_COMPUTE_SHADER));assert gl.glGetProgramiv(program,gl.GL_LINK_STATUS)
    rows=np.zeros((300,12),np.float32);rows[:,2]=1;rows[:,3]=rows[:,10]=1
    rows[:,8]=np.tile(2.998e17/np.array([650,550,450]),100)
    outputs=[];buffers=[]
    try:
        for binding,data in [(5,rows),(14,np.zeros((65536,4),np.float32))]:
            buf=int(gl.glGenBuffers(1));buffers.append(buf)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buf);gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,gl.GL_STATIC_DRAW)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,buf)
        gl.glUseProgram(program)
        for key,value in [('uNodeCount',0),('uStudioSphereCount',0),('uSourceCount',300),('uFixed3Transport',1)]:
            gl.glUniform1i(gl.glGetUniformLocation(program,key),value)
        for mode,data in [(0,rows),(1,fixed3_nee_records(rows))]:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[0]);gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,gl.GL_STATIC_DRAW)
            gl.glUniform1i(gl.glGetUniformLocation(program,'uNEEFullSpectrum'),mode)
            gl.glDispatchCompute(16,16,1);gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT);gl.glFinish()
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[1]);out=np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,65536*16),np.float32).reshape(-1,4)[:,:3].copy();outputs.append(out)
        np.testing.assert_allclose(outputs[0].mean(0),50,rtol=.025)
        np.testing.assert_allclose(outputs[1],50,rtol=2e-7)
        assert np.max(np.ptp(outputs[1],axis=0))==0
        receipt={'renderer':gl.glGetString(gl.GL_RENDERER).decode(),'production_sensor_shader_link':'PASS','samples':65536,
            'old_mean':outputs[0].mean(0).tolist(),'new_mean':outputs[1].mean(0).tolist(),
            'old_channel_sd':outputs[0].std(0).tolist(),'new_channel_sd':outputs[1].std(0).tolist(),
            'limits':'isolated unoccluded white point-source estimator; not a full image comparison'}
        target=ROOT/'build/dot-nee-spectral-20261109/gpu-check.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
    finally:
        gl.glUseProgram(0);gl.glDeleteBuffers(len(buffers),buffers);gl.glDeleteProgram(program);gl.glDeleteProgram(actual)
if __name__=='__main__':main()
