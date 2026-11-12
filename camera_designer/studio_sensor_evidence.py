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
    def __init__(self,width,height,bands,rows,compile_program,*,response_node_count=None,sample_capacity=None):
        from OpenGL import GL as gl
        self.width,self.height,self.bands,self.rows=width,height,bands,rows
        if response_node_count is None:
            self._initial_nodes=leaf_nodes(width,height,bands)
        else:
            if type(response_node_count) is not int or response_node_count < 1 or bands != 3 or width*height != response_node_count:
                raise ValueError('response nodes require explicit bounded native three-band storage')
            self._initial_nodes=np.zeros((response_node_count,16),np.uint32)
            self._initial_nodes[:,4:6]=0xffffffff
            self._initial_nodes[:,8]=np.arange(response_node_count,dtype=np.uint32)*bands
        if type(rows) is not int or rows<1 or rows%16:
            raise ValueError("evidence requires unique 16-row strips")
        self.buffers=[];self.program=0;self.score_program=0;self.nbytes=0
        count=width*height;batch=width*min(rows,height) if sample_capacity is None else sample_capacity
        if type(batch) is not int or batch<1:raise ValueError('invalid native evidence sample capacity')
        self.sample_capacity=batch
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
        if not 0<count<=self.sample_capacity:
            raise ValueError("invalid evidence strip sample count")
        gl.glUseProgram(self.program)
        for binding,b in enumerate(self.buffers):
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,b)
        for name,value in [('sample_count',count),('n_bands',self.bands),('use_schedule_count',0)]:
            gl.glUniform1ui(gl.glGetUniformLocation(self.program,name),value)
        gl.glDispatchCompute((count+63)//64,1,1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

    def accumulate_response_rows(self,lineage,spectra):
        """Upload copied response columns to the unchanged native accumulator.

        Its event-row weight/count is NOT the camera denominator. Caller keeps
        original camera-sample weights separately; event variance is unused.
        """
        from OpenGL import GL as gl
        lineage=np.ascontiguousarray(lineage,dtype=np.uint32)
        spectra=np.ascontiguousarray(spectra,dtype=np.float32)
        if lineage.shape!=(len(spectra),8) or spectra.shape[1:]!=(self.bands,) or len(spectra)>self.sample_capacity:
            raise ValueError('invalid copied response columns')
        for index,data in ((1,lineage),(2,spectra)):
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.buffers[index])
            gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,data.nbytes,data)
        self.accumulate(len(spectra))

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


# Optional pre-integration response rows extend the existing sensor evidence
# side channel; leaf-integrated spectra and their ABI remain unchanged.
from .studio_camera_intervals import CameraIntervalObserver, plan_interval_dispatch, WORDS

class StudioResponseEvidence(CameraIntervalObserver):
    """Explicit, synchronous per-dispatch consumption; no retained history.

    Consumer must process/spool each payload before returning, under its own
    explicit disk/retention policy. No data is silently dropped on overflow.
    """
    def __init__(self, *, memory_budget_bytes, reserved_bytes, scene_revision,
                 camera_revision, consume, scalar_visit_buffer=0, completed_samples=False):
        super().__init__(1, 1, scene_revision=scene_revision, camera_revision=camera_revision)
        if not callable(consume):
            raise TypeError('camera interval stream requires an explicit consumer')
        if type(memory_budget_bytes) is not int or type(reserved_bytes) is not int or memory_budget_bytes <= reserved_bytes or reserved_bytes < 0:
            raise ValueError('explicit total and reserved memory budgets are required')
        self.buffer = 0
        self.response_buffer = 0
        if type(scalar_visit_buffer) is not int or scalar_visit_buffer < 0:
            raise ValueError('scalar visitation buffer must be an existing borrowed handle')
        self.scalar_visit_buffer = scalar_visit_buffer
        self.memory_budget_bytes, self.reserved_bytes = memory_budget_bytes, reserved_bytes
        self.consume = consume
        self.completed_samples=bool(completed_samples)
        import uuid
        self.stream_id=uuid.uuid4().hex
        self.plan = None
        self._allocated_bytes = 0
        self._response_allocated_bytes = 0
        self.sequence = 0
        self._pending_identity=set()
        self.identity['field_generation']=0

    def note_mutation(self, kind):
        if kind not in ('scene','camera'):raise ValueError('unknown response identity mutation')
        self._pending_identity.add(kind)

    def rebind_identity(self, *, scene_revision=None, camera_revision=None):
        for kind,value in (('scene',scene_revision),('camera',camera_revision)):
            if value is not None:
                if not isinstance(value,str) or not value:raise ValueError('nonempty actual revision required')
                self.identity[kind+'_revision']=value
                self._pending_identity.discard(kind)

    def note_field_reset(self):
        self.identity['field_generation']+=1
        reset=getattr(self,'native_scalar_reset',None)
        if reset is not None:
            if not callable(reset):raise TypeError('native scalar reset callback must be callable')
            reset()

    def prepare(self, **dispatch):
        if self._pending_identity:
            raise ValueError('response identity must be rebound after '+','.join(sorted(self._pending_identity)))
        if self.completed_samples:return self._prepare_completed(dispatch)
        plan = plan_interval_dispatch(**dispatch, memory_budget_bytes=self.memory_budget_bytes,
                                      reserved_bytes=self.reserved_bytes, response_copies=True)
        if hasattr(self.consume,'preflight_capacity'):
            self.consume.preflight_capacity(plan['capacity'],live_copy_bytes=plan['working_bytes'])
        from OpenGL import GL
        import struct
        if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)) <= 24:
            raise ValueError('response evidence requires SSBO binding index 24')
        if plan['storage_bytes']>int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)):
            raise MemoryError('response evidence exceeds native SSBO block limit')
        if not self.buffer:
            self.buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer)
        if self._allocated_bytes != plan['storage_bytes']:
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, plan['storage_bytes'], None, GL.GL_DYNAMIC_READ)
            if int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))!=plan['storage_bytes']:
                raise MemoryError('camera interval buffer allocation incomplete')
            self._allocated_bytes = plan['storage_bytes']
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, 8, struct.pack('<II', 0, 0))
        if not self.response_buffer:
            self.response_buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.response_buffer)
        if self._response_allocated_bytes != plan['storage_bytes']:
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, plan['storage_bytes'], None, GL.GL_DYNAMIC_READ)
            if int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))!=plan['storage_bytes']:
                raise MemoryError('camera response buffer allocation incomplete')
            self._response_allocated_bytes=plan['storage_bytes']
        GL.glClearBufferData(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, struct.pack('<I', 0))
        self.capacity = plan['capacity']
        self.identity['required_bytes'] = plan['storage_bytes']
        self.plan = plan
        return plan['rows']

    def bind(self, program):
        if self.completed_samples:
            from OpenGL import GL
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,25,self.buffer)
            GL.glUniform1ui(GL.glGetUniformLocation(program,"uCompletedSampleCapacity"),self.capacity)
            GL.glUniform1i(GL.glGetUniformLocation(program,"uCompletedSamplesEnabled"),1)
            GL.glUniform1i(GL.glGetUniformLocation(program,"uCameraIntervalsEnabled"),0)
            if self.scalar_visit_buffer:
                from .studio_scalar_visits import prepare_sampling
                prepare_sampling(self.scalar_visit_buffer)
            return
        super().bind(program)
        from OpenGL import GL
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 24, self.response_buffer)
        if self.scalar_visit_buffer:
            from .studio_scalar_visits import prepare_sampling
            prepare_sampling(self.scalar_visit_buffer)

    def finish(self, *, dispatch_identity):
        if self.completed_samples:return self._finish_completed(dispatch_identity)
        from OpenGL import GL
        import struct
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer)
        header = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, 8)
        count, overflow = struct.unpack('<II', header)
        minimum=int(dispatch_identity.get('active_pixels',0))*int(dispatch_identity.get('samples_per_pixel',1))*int(dispatch_identity.get('active_path_bands',1))
        if count<minimum:
            raise RuntimeError('camera response copies incomplete: missing original camera invocations')
        if overflow or count > self.capacity:
            raise RuntimeError('camera interval stream incomplete: overflow; no absence inference is valid')
        payload = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 8, count * WORDS * 4) if count else b''
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.response_buffer)
        response_header = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, 8)
        if struct.unpack('<II', response_header)[1]:
            raise RuntimeError('camera response copies incomplete: storage overflow')
        responses = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 8, count * WORDS * 4) if count else b''
        if len(payload)!=count*WORDS*4 or len(responses)!=count*WORDS*4:
            raise RuntimeError('camera response copies incomplete: truncated paired readback')
        metadata = dict(self.identity, stream_id=self.stream_id,source_sha256=dispatch_identity.get('source_sha256'),
                        response_schema='native-camera-response-copies-v1',
                        response_weighting='original_path_throughput_not_availability_premultiplied',
                        camera_sample_key=['dispatch_identity','invocation_word15','sample_word2'],
                        denominator='original_camera_samples_not_response_rows',
                        application_ready=False, recipient_mapping_reference=None, sequence=self.sequence, row_count=count,
                        complete_dispatch_observation=True, dispatch=dispatch_identity,
                        excludes=['NEE sample lineage', 'untraced paths', 'field response evaluation'])
        self.consume(metadata, memoryview(payload), memoryview(responses))
        self.sequence += 1

    def _prepare_completed(self, dispatch):
        # Preserve original dispatch width/strip/samples and shared-band path.
        if dispatch.get('path_bands',1)!=1:
            raise ValueError('completed copies require actual original shared-band camera path')
        width,rows,samples=(int(dispatch[k]) for k in ('width','rows','samples'))
        if min(width,rows,samples)<=0:raise ValueError('invalid original camera dispatch')
        capacity=width*rows*samples;storage=8+capacity*512
        peak=self.reserved_bytes+3*storage+capacity*16+4096
        if peak>self.memory_budget_bytes:raise MemoryError('original completed-sample dispatch exceeds total budget')
        from OpenGL import GL
        import struct
        if int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=25:
            raise ValueError('completed sample copy requires SSBO binding25')
        if storage>int(GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)):
            raise MemoryError('completed sample copy exceeds actual SSBO block limit')
        if not self.buffer:self.buffer=int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,self.buffer)
        if self._allocated_bytes!=storage:
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,storage,None,GL.GL_DYNAMIC_READ)
            if int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))!=storage:
                raise MemoryError('incomplete completed-sample allocation')
            self._allocated_bytes=storage
        GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,8,struct.pack('<II',0,0))
        self.capacity=capacity;self.identity['required_bytes']=storage
        self.plan=dict(capacity=capacity,storage_bytes=storage,total_peak_bytes=peak,rows=rows)
        return rows

    def _finish_completed(self, dispatch):
        from OpenGL import GL
        import struct
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT|GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,self.buffer)
        count,overflow=struct.unpack('<II',GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,8))
        expected=int(dispatch.get('active_pixels',0))*int(dispatch.get('samples_per_pixel',1))
        if overflow or count!=expected or count>self.capacity:
            raise RuntimeError('completed original camera sample copies incomplete; no absence inference')
        payload=GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,8,count*512) if count else b''
        metadata=dict(self.identity,stream_id=self.stream_id,sequence=self.sequence,
            source_sha256=dispatch.get('source_sha256'),dispatch=dispatch,row_count=count,
            response_schema='native-camera-scene-sample-v1',complete_dispatch_observation=True,
            shared_band0_path=True,denominator='original camera sample count',application_ready=False)
        self.consume(metadata,memoryview(payload));self.sequence+=1

    def close(self):
        if self.buffer:
            from OpenGL import GL
            GL.glDeleteBuffers(1, [self.buffer])
            if self.response_buffer:
                GL.glDeleteBuffers(1, [self.response_buffer])
                self.response_buffer = 0
            self.buffer = 0
            self._allocated_bytes = 0
