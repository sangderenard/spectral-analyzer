"""Opt-in native sensor-moment adapter gate; no production API calls."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import demo_pluck_gl as r
    from OpenGL import GL as gl
    from camera_designer.studio_sensor_evidence import StudioSensorEvidence
    r._ensure_worker_gl_context()
    p=r._prog((r._GPU_SENSOR_CS,gl.GL_COMPUTE_SHADER));assert gl.glGetProgramiv(p,gl.GL_LINK_STATUS);gl.glDeleteProgram(p)
    e=StudioSensorEvidence(2,3,3,16,r._prog)
    try:
        lineage=np.zeros((6,8),np.uint32);lineage[:,0]=np.arange(6);lineage.view(np.float32)[:,6]=1
        samples=[np.arange(18,dtype=np.float32).reshape(6,3),np.arange(18,0,-1,dtype=np.float32).reshape(6,3)]
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,e.buffers[1]);gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,lineage.nbytes,lineage)
        for sample in samples:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,e.buffers[2]);gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,sample.nbytes,sample);e.accumulate(6)
        m=e.read_moments();np.testing.assert_array_equal(m['sum'],samples[0]+samples[1]);np.testing.assert_array_equal(m['sum_sq'],samples[0]**2+samples[1]**2);np.testing.assert_array_equal(m['sample_count'],2);np.testing.assert_array_equal(m['weight'],2)
        e.score_conditional_variation()
        scored=e.read_moments()['conditional_variation_priority']
        stack=np.stack(samples).astype(np.float64)
        expected=np.sqrt(stack.var(axis=0).mean(axis=1)/2)/np.maximum(np.abs(stack.mean(axis=0).mean(axis=1)),1e-6)
        np.testing.assert_allclose(scored,expected,rtol=2e-6,atol=1e-7)
        e.reset();e.score_conditional_variation();m=e.read_moments();
        assert not m['conditional_variation_priority'].any()
        # A zero score with no evidence is explicitly NOT confidence.
        assert not m['sample_count'].any() and not m['sum'].any()
        receipt={'renderer':gl.glGetString(gl.GL_RENDERER).decode(),'existing_native_accumulator':'PASS','existing_native_scorer':'PASS','six_pixels_three_bands_two_samples_including_zero':'PASS','reset':'PASS','bytes':e.nbytes,'limits':'side-channel moments only; no adaptive sampling, early stopping, display confidence or production activation'}
        out=ROOT/'build/dot-nee-spectral-20261109/evidence-gpu-check.json';out.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
    finally:e.destroy()
    assert e.nbytes==0 and not e.buffers
if __name__=='__main__':main()
