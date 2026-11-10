"""Opt-in Studio sample-evidence bridge to the native sensor mipmap ABI.

This is a side channel, not a replacement sampler or a confidence certificate.
Each dense pixel owns a leaf node. Native accumulation retains raw sample
moments, including zeros; no rollup, priority selection or early stopping is
activated. Evidence is since reset, independent of finite display-ring length.
"""
from pathlib import Path
import numpy as np

SENSOR_EVIDENCE_GLSL = r"""
struct StudioSensorLineage {
    uint node_id; uint level; float global_u; float global_v;
    float local_u; float local_v; float estimator_weight; uint sample_index;
};
layout(std430,binding=13) buffer StudioEvidenceSpectra { float studio_evidence_spectra[]; };
layout(std430,binding=14) buffer StudioEvidenceLineage { StudioSensorLineage studio_evidence_lineage[]; };
uniform int uSensorEvidenceEnabled;
uniform uint uSensorEvidenceSampleIndex;
"""


def leaf_nodes(width, height, bands):
    if any(type(v) is not int for v in (width,height,bands)) or not (1<=width<=512 and 1<=height<=512 and bands==3):
        raise ValueError("Studio evidence requires a bounded fixed:3 sensor")
    count=width*height
    nodes=np.zeros((count,16),np.uint32)
    # Exact existing SensorMipNodeGpu layout, 64 bytes per leaf.
    x=np.tile(np.arange(width),height);y=np.repeat(np.arange(height),width)
    nodes[:,:4].view(np.float32)[:]=np.column_stack((x/width,y/height,(x+1)/width,(y+1)/height))
    nodes[:,4:6]=0xffffffff
    nodes[:,8]=np.arange(count,dtype=np.uint32)*bands
    return nodes


class StudioSensorEvidence:
    def __init__(self,width,height,bands,rows,compile_program):
        from OpenGL import GL as gl
        self.width,self.height,self.bands,self.rows=width,height,bands,rows
        self._initial_nodes=leaf_nodes(width,height,bands)
        if type(rows) is not int or rows<1 or rows%16:
            raise ValueError("evidence requires unique 16-row strips")
        self.buffers=[];self.program=0;self.score_program=0;self.nbytes=0
        count=width*height;batch=width*min(rows,height)
        values=[self._initial_nodes,np.zeros((batch,8),np.uint32),np.zeros((batch,bands),np.float32),
            np.zeros(count*bands,np.float32),np.zeros(count*bands,np.float32),np.zeros(count,np.float32),
            np.array([count,0,0,0,0,0,0,0],np.uint32),np.zeros(2,np.uint32)]
        try:
            source=(Path(__file__).resolve().parents[1]/'csrc/shaders/sensor_mip_accumulate.comp.glsl').read_text()
            self.program=compile_program((source,gl.GL_COMPUTE_SHADER))
            if not gl.glGetProgramiv(self.program,gl.GL_LINK_STATUS):
                raise RuntimeError("native sensor-mipmap accumulator did not link")
            score_source=(Path(__file__).resolve().parents[1]/'csrc/shaders/sensor_mip_score.comp.glsl').read_text()
            self.score_program=compile_program((score_source,gl.GL_COMPUTE_SHADER))
            if not gl.glGetProgramiv(self.score_program,gl.GL_LINK_STATUS):
                raise RuntimeError("native sensor-mipmap scorer did not link")
            for data in values:
                b=int(gl.glGenBuffers(1));self.buffers.append(b)
                gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,b)
                gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,gl.GL_DYNAMIC_DRAW)
                if int(gl.glGetBufferParameteriv(gl.GL_SHADER_STORAGE_BUFFER,gl.GL_BUFFER_SIZE))!=data.nbytes:
                    raise MemoryError("sensor evidence allocation incomplete")
                self.nbytes+=data.nbytes
        except Exception:
            self.destroy();raise
        finally:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,0)

    def bind_sensor(self):
        from OpenGL import GL as gl
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,13,self.buffers[2])
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,14,self.buffers[1])

    def accumulate(self,count):
        from OpenGL import GL as gl
        if not 0<count<=self.width*min(self.rows,self.height):
            raise ValueError("invalid evidence strip sample count")
        gl.glUseProgram(self.program)
        for binding,b in enumerate(self.buffers):
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,b)
        for name,value in [('sample_count',count),('n_bands',self.bands),('use_schedule_count',0)]:
            gl.glUniform1ui(gl.glGetUniformLocation(self.program,name),value)
        gl.glDispatchCompute((count+63)//64,1,1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

    def score_conditional_variation(self):
        """Native relative-variation priority, NOT calibrated confidence.

        Unmodified native formula is population variance / sample weight.
        It returns zero for constant/zero/single samples; that does not imply
        convergence. Fixed source-bake bias, correlations and unseen paths
        remain outside this score. No scheduler or stopping rule consumes it.
        Image/learned/requested features are deliberately disabled here.
        """
        from OpenGL import GL as gl
        gl.glUseProgram(self.score_program)
        # Disabled native features never read their buffers. Bind valid storage
        # anyway; sensor_res=0 prevents square planar image ABI assumptions.
        for binding,index in enumerate([0,7,3,4,5,6,7,7,7,7]):
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,self.buffers[index])
        for name,value in [('node_count',self.width*self.height),('n_bands',self.bands),
                ('derive_from_moments',1),('sensor_res',0),('learned_map_enabled',0),
                ('requested_map_enabled',0)]:
            gl.glUniform1ui(gl.glGetUniformLocation(self.score_program,name),value)
        gl.glUniform4f(gl.glGetUniformLocation(self.score_program,'feature_weights'),1,0,0,0)
        gl.glDispatchCompute((self.width*self.height+63)//64,1,1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

    def read_moments(self):
        """Explicit debug readback only. Never called by the live frame loop."""
        from OpenGL import GL as gl
        gl.glFinish();result={}
        for name,index,size,dtype in [('nodes',0,self.width*self.height*16,np.uint32),
                ('sum',3,self.width*self.height*self.bands,np.float32),
                ('sum_sq',4,self.width*self.height*self.bands,np.float32),
                ('weight',5,self.width*self.height,np.float32)]:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.buffers[index])
            result[name]=np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,size*4),dtype).copy()
        nodes=result.pop('nodes').reshape(-1,16)
        result['sample_count']=nodes[:,10].copy()
        result['conditional_variation_priority']=nodes[:,12].copy().view(np.float32)
        result['sum']=result['sum'].reshape(-1,self.bands)
        result['sum_sq']=result['sum_sq'].reshape(-1,self.bands)
        return result

    def reset(self):
        from OpenGL import GL as gl
        for index,data in [(0,self._initial_nodes),(3,np.zeros(self.width*self.height*self.bands,np.float32)),
                (4,np.zeros(self.width*self.height*self.bands,np.float32)),(5,np.zeros(self.width*self.height,np.float32))]:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.buffers[index])
            gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,data.nbytes,data)

    def destroy(self):
        from OpenGL import GL as gl
        if self.buffers:gl.glDeleteBuffers(len(self.buffers),self.buffers)
        if self.program:gl.glDeleteProgram(self.program)
        if self.score_program:gl.glDeleteProgram(self.score_program)
        self.buffers=[];self.program=0;self.score_program=0;self.nbytes=0
