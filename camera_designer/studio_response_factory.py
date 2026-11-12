"""Explicit response-copy owner construction; no default allocations/history."""
import hashlib,json
import numpy as np
MEMORY_BUDGET_BYTES=512*1024*1024


def camera_record_mode(config):
    """Extra interval evidence is explicit; existing completed samples stay default."""
    mode=config.get('light_field_camera_records')
    if mode is None:return 'completed-samples-v1'
    if mode!='interval-responses-v2' or config.get('light_field_information')!='evaluated-responses-v1':
        raise ValueError('camera interval records require explicit evaluated-response information opt-in')
    return mode


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def retained_array_bytes(*values):
    seen=set();total=0
    def walk(value):
        nonlocal total
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,np.ndarray):
            root=value
            while isinstance(root.base,np.ndarray):root=root.base
            if id(root)!=id(value):walk(root)
            else:total+=root.nbytes
        elif isinstance(value,dict):
            for item in value.values():walk(item)
        elif isinstance(value,(tuple,list)):
            for item in value:walk(item)
    for value in values:walk(value)
    return total


def field_reserved_bytes(state,extra_arrays=()):
    """Actual owned field SSBO sizes plus known native R32UI textures/CPU arrays."""
    from OpenGL import GL
    handles=set(int(h) for h in state.get('ssbo',()) if h)
    for h in (state.get('native_pbr_ssbo'),state.get('scalar_visit_buffer'),(state.get('scalar_columns') or {}).get('buffer'),
              (state.get('directional_record') or {}).get('buffer'),(state.get('volume_extinction') or {}).get('buffer')):
        if h:handles.add(int(h))
    forward=state.get('forward_response_owner')
    if forward is not None and forward.handle:handles.add(int(forward.handle))
    previous=int(GL.glGetIntegerv(GL.GL_SHADER_STORAGE_BUFFER_BINDING));total=0
    try:
        for handle in handles:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
            size=int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))
            if size<0:raise ValueError('invalid native allocation size')
            total+=size
    finally:GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,previous)
    total+=len(state['tex_bands'])*int(np.prod(state['dims']))*4
    return total+retained_array_bytes(state,extra_arrays)


def make_response_owner(config,state,camera,*,consume,extra_arrays=(),sensor_reserved_bytes=0,completed_samples=False,retention_reserved_bytes=0):
    mode=camera_record_mode(config)
    if mode=='interval-responses-v2' and completed_samples:raise ValueError('interval response mode cannot substitute completed samples')
    if config.get('light_field_information') is None:return None
    if config.get('light_field_information')!='evaluated-responses-v1':raise ValueError('unsupported response-copy opt-in')
    if not callable(consume):raise TypeError('synchronous native component consumer is required')
    if type(sensor_reserved_bytes) is not int or sensor_reserved_bytes<0:raise ValueError('actual sensor reservation required')
    from camera_designer.studio_sensor_evidence import StudioResponseEvidence
    reserved=field_reserved_bytes(state,extra_arrays)+sensor_reserved_bytes+retention_reserved_bytes+publication_working_reserve(state)
    if reserved>=MEMORY_BUDGET_BYTES:raise MemoryError('existing allocations exhaust response-copy working-memory budget')
    owner=StudioResponseEvidence(memory_budget_bytes=MEMORY_BUDGET_BYTES,reserved_bytes=reserved,
        scene_revision=state.get('response_identity',{}).get('scene_revision',state['directional_record'].get('source_population_scene_base',state['directional_record']['scene_sha256'])),camera_revision=identity(camera),
        scalar_visit_buffer=state['scalar_visit_buffer'],consume=consume,completed_samples=completed_samples)
    def reset_native_scalar_columns():
        if state.get('scalar_columns') is not None:
            from camera_designer.studio_scalar_columns import clear_view
            clear_view(state['scalar_columns']);state['scalar_clear_epoch']+=1
    owner.native_scalar_reset=reset_native_scalar_columns
    return owner


def rebind_response_owner(accumulator,state,camera):
    owner=getattr(accumulator,'_light_field_information',None)
    if owner is None:return
    scene_revision=state.get('response_identity',{}).get('scene_revision',state['directional_record'].get('source_population_scene_base',state['directional_record']['scene_sha256']))
    camera_revision=identity(dict(camera=camera,sensor_size=[accumulator._w,accumulator._h],aspect=getattr(accumulator,'_aspect',None)))
    owner.rebind_identity(scene_revision=scene_revision,camera_revision=camera_revision)
    if hasattr(owner,'identity'):
        owner.identity['transport_scene_sha256']=state.get('response_transport_scene_sha256')
        owner.identity['reactive_queue']=dict(state.get('response_reactive_queue') or {'observed':False,'complete':False,'reason':'native queue not yet observed'})
        owner.identity['live_revision']=int(state.get('response_live_revision',0))
        owner.identity['source_population_sha256']=hashlib.sha256(np.ascontiguousarray(state['source_buf'],dtype='<f4').tobytes()).hexdigest()
        owner.identity['source_kind']='installed-camera-program'
        owner.identity['transport_program_sha256']=hashlib.sha256(accumulator._sensor_shader_source.encode()).hexdigest()
        scope=state['directional_record'].get('recording_region')
        owner.identity['indexing_bounds']=[scope['bounds_min_m'],scope['bounds_max_m']] if scope else [np.asarray(state['bmin']).tolist(),np.asarray(state['bmax']).tolist()]
        owner.identity['indexing_dimensions']=[state['directional_record']['grid_size']]*3
    if hasattr(owner,'_response_extra_arrays'):
        reserved=field_reserved_bytes(state,state.get('_response_extra_arrays',owner._response_extra_arrays))+sensor_reserved_bytes(accumulator,state)+publication_working_reserve(state)+getattr(owner,'_retention_reserved_bytes',0)
        if reserved>=MEMORY_BUDGET_BYTES:raise MemoryError('existing allocations exhaust response-copy memory budget')
        owner.reserved_bytes=reserved
        forward=state.get('forward_response_owner')
        if forward is not None:
            from camera_designer.studio_forward_responses import dispatch_storage_plan
            forward.plan=dispatch_storage_plan(forward.ray_count,forward.bounce_limit,
                reserved_bytes=reserved-forward.plan['buffer_bytes']+int(getattr(owner,'_allocated_bytes',0)),
                max_ssbo_bytes=forward.plan['buffer_bytes'],budget_bytes=MEMORY_BUDGET_BYTES)
        consumer=getattr(owner.consume,'target',owner.consume)
        if consumer is not None:consumer.reserved_bytes=reserved


def sensor_reserved_bytes(accumulator,state):
    """Actual extra sensor-owned buffers/textures, excluding borrowed field and copy staging."""
    from OpenGL import GL
    borrowed=set(int(h) for h in state.get('ssbo',()) if h)
    borrowed.add(int(state.get('native_pbr_ssbo',0)))
    handles=set(int(h) for h in getattr(accumulator,'_history_buffers',()) if h)
    for name in ('_ssbo_sources_nee','_ssbo_sources_fwd','_ssbo_seg_dummy','_ssbo_counter_fwd'):
        value=getattr(accumulator,name,0)
        if value:handles.add(int(value))
    sample=getattr(accumulator,'_sample_evidence',None)
    if sample is not None:handles.update(int(h) for h in sample.buffers if h)
    previous=int(GL.glGetIntegerv(GL.GL_SHADER_STORAGE_BUFFER_BINDING));total=0
    try:
        for handle in handles-borrowed:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,handle)
            total+=int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER,GL.GL_BUFFER_SIZE))
    finally:GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,previous)
    total+=len(accumulator._textures)*accumulator._w*accumulator._h*16
    # Camera vectors and other retained ndarray state, excluding borrowed arrays
    # already counted with the field, share the same root-storage accounting.
    total+=retained_array_bytes(state,vars(accumulator))-retained_array_bytes(state)
    return total


class ResponseConsumerBridge:
    """Owner exists for shader construction; consumer attaches before first dispatch."""
    def __init__(self):self.target=None
    def __call__(self,*args):
        if self.target is None:raise RuntimeError('native copied component consumer not installed')
        return self.target(*args)
    def preflight_capacity(self,*args,**kwargs):
        if self.target is None:raise RuntimeError('native copied component consumer not installed')
        return self.target.preflight_capacity(*args,**kwargs)


def attach_component_consumer(owner,bridge,accumulator,state,*,film_response,compile_program,publish,extra_arrays=()):
    from camera_designer.studio_copied_components import CopiedComponentConsumer
    reserved=field_reserved_bytes(state,extra_arrays)+sensor_reserved_bytes(accumulator,state)
    if reserved>=MEMORY_BUDGET_BYTES:raise MemoryError('existing allocations exhaust copied component memory budget')
    owner.reserved_bytes=reserved
    def current_state():
        return dict(scene_revision=owner.identity['scene_revision'],camera_revision=owner.identity['camera_revision'],lineage=state['response_lineage'])
    bridge.target=CopiedComponentConsumer(state_provider=current_state,film_response=film_response,compile_program=compile_program,
        publish=publish,memory_budget_bytes=MEMORY_BUDGET_BYTES,reserved_bytes=reserved)
    owner._response_extra_arrays=extra_arrays
    return bridge.target


class SceneResponseConsumer:
    """Shared typed scene store, initialized by the first actual native receipt.

    Reservation is explicit before dispatch. Camera moves/population refreshes
    do not reset storage; only a changed physical scene revision invalidates it.
    """
    def __init__(self, *, memory_budget_bytes, lineage_provider, bounds, dimensions,stencil_size=12):
        if type(memory_budget_bytes) is not int or memory_budget_bytes<32768:
            raise ValueError('bounded scene retention budget required')
        if not callable(lineage_provider):raise TypeError('authoritative lineage provider required')
        self.memory_budget_bytes=memory_budget_bytes;self.lineage_provider=lineage_provider
        self.bounds=bounds;self.dimensions=dimensions;self.stencil_size=stencil_size;self.store=None;self.application_builder=None;self.current_camera_metadata=None

    def _store(self,metadata):
        from camera_designer.studio_scene_responses import SceneResponseStore
        revision=metadata.get('scene_revision')
        if 'indexing_bounds' in metadata:
            self.bounds=metadata['indexing_bounds'];self.dimensions=metadata['indexing_dimensions']
        if not isinstance(revision,str) or not revision:raise ValueError('actual physical scene revision required')
        if self.store is None:
            self.store=SceneResponseStore(scene_revision=revision,memory_budget_bytes=self.memory_budget_bytes,
                lineage_provider=self.lineage_provider,bounds=self.bounds,dimensions=self.dimensions,stencil_size=self.stencil_size)
        else:
            self.store.invalidate_scene(revision)
            if 'indexing_bounds' in metadata:
                self.store.bounds=np.asarray(self.bounds,np.float64);self.store.dimensions=np.asarray(self.dimensions,np.int64)
        return self.store

    def __call__(self,metadata,rows):self._store(metadata).consume_forward(metadata,rows)
    def consume_scalar_cells(self,metadata,cells):self._store(metadata).consume_scalar_cells(metadata,cells)
    def consume_camera_samples(self,metadata,rows):
        self._store(metadata).consume_camera_samples(metadata,rows)
        self.current_camera_metadata=dict(metadata)
    def consume_camera_pairs(self,metadata,intervals,responses):
        self._store(metadata).consume_camera_pairs(metadata,intervals,responses)
        # A paired observation is not a completed sample/current RGB query.
    def snapshot(self):
        if self.store is None:raise ValueError('no actual native scene response received')
        application=self.application_builder(self.store) if self.application_builder is not None else None
        return self.store.snapshot(application=application)


def publication_working_reserve(state):
    """Known simultaneous future atlas/scalar conversion storage, not GPU VRAM."""
    record=state.get('directional_record')
    total=0
    if record is not None:
        from camera_designer.studio_directional_field import word_count
        atlas=word_count(record['bin_count'],record['grid_size'])*4
        total+=atlas
        if not record.get('progressive'):total+=2*atlas
        total+=record['grid_size']**2*record['bin_count']*128 # bounded conversion slab intermediates
    # Existing scalar export converts three-band counts and RGB through double
    # intermediates. Reserve twelve f64 planes per native cell conservatively.
    total+=int(np.prod(state['dims']))*12*8
    return total


def selected_sensor_reservation(config,source_rows,*,retention_bytes):
    """Pre-warmup reservation from selected native dimensions/history layouts."""
    from camera_designer.sensor_history import SensorHistoryLayout
    live=config.get('live') or {}
    resolution=int(config.get('camera',{}).get('resolution',256))
    w=int(live.get('width',resolution));h=int(live.get('height',resolution));bands=3
    if not 1<=w<=512 or not 1<=h<=512:
        raise ValueError('scene-response sensor admission currently supports untiled up to512x512')
    history=SensorHistoryLayout(w,h,bands,int(live['history_samples']),bool(live.get('checkpoint',False))).nbytes if live else 0
    outputs=w*h*bands*16
    source_copies=int(np.asarray(source_rows).nbytes)*4
    readback_work=w*h*bands*8*12
    capacity=w*min(16,h)
    camera_copy=3*(8+capacity*512)+16*capacity+4096
    if camera_record_mode(config)=='interval-responses-v2':
        bounces=config.get('max_bounces',4)
        if type(bounces) is not int or not 1<=bounces<=200:raise ValueError('explicit bounded camera bounce count required')
        capacity*=bounces+1
        camera_copy=4*(8+capacity*512)+16*capacity+4096
    return dict(total=retention_bytes+history+outputs+source_copies+readback_work+camera_copy,
        retention=retention_bytes,history=history,outputs=outputs,source_copies=source_copies,
        readback_work=readback_work,camera_copy=camera_copy,width=w,height=h)


def admit_response_history_change(accumulator,state,samples,checkpoint=False):
    """Reject oversized history before its original allocation, only on opt-in."""
    owner=getattr(accumulator,'_light_field_information',None)
    if owner is None:return
    from camera_designer.sensor_history import SensorHistoryLayout
    target=SensorHistoryLayout(accumulator._w,accumulator._h,len(accumulator._textures),int(samples),bool(checkpoint)).nbytes
    total=field_reserved_bytes(state,state.get('_response_extra_arrays',()))+sensor_reserved_bytes(accumulator,state)
    total+=publication_working_reserve(state)+getattr(owner,'_retention_reserved_bytes',0)
    total+=max(0,target-int(getattr(accumulator,'_history_bytes',0)))
    capacity=accumulator._w*min(accumulator._rows_per_frame,accumulator._h)
    total+=3*(8+capacity*512)+16*capacity+4096
    if total>MEMORY_BUDGET_BYTES:
        raise MemoryError(f'selected history plus response working storage requires {total} bytes; budget {MEMORY_BUDGET_BYTES}')


def retained_scene_identity(config,vertices,material_ids,pbr):
    """Authoritative saved target/fixed-scene inputs, not native namespace guesses.

    Camera housing is NOT deleted from transport. Its exact assembled buffers
    are a separate mandatory compatibility hash on every observation/query.
    """
    view_keys={'camera','camera_poses','orbit_frames','live','live_revision','geometry_sha256','emitter_sampling'}
    fixed={k:v for k,v in config.items() if k not in view_keys}
    fixed['source_generator_version']=(config.get('emitter_sampling') or {}).get('version')
    h=hashlib.sha256(identity(fixed).encode())
    for value,dtype in ((vertices,'<f4'),(material_ids,'<i4'),(pbr,'<f4')):
        a=np.ascontiguousarray(value,dtype=dtype)
        h.update(str(a.shape).encode())
        if a.size:h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def current_scene_query_metadata(state,consumer,live_revision=0):
    """Current context is an exact native observation only while still matching.

    No prior observation is promoted merely because it has the largest sequence.
    """
    metadata=consumer.current_camera_metadata
    owner=state.get('response_camera_owner')
    if metadata is None or owner is None:return None
    for key in ('scene_revision','camera_revision','transport_scene_sha256'):
        if not owner.identity.get(key) or owner.identity[key]!=metadata.get(key):return None
    if owner.identity['transport_scene_sha256']!=state.get('response_transport_scene_sha256'):return None
    current_source=hashlib.sha256(np.ascontiguousarray(state['source_buf'],dtype='<f4').tobytes()).hexdigest()
    if metadata.get('source_population_sha256')!=current_source:return None
    accumulator=state.get('response_camera_accumulator')
    if accumulator is None or metadata.get('source_sha256')!=hashlib.sha256(accumulator._sensor_shader_source.encode()).hexdigest():return None
    if int(metadata.get('live_revision',-1))!=int(live_revision):return None
    current_queue=state.get('response_reactive_queue') or {'observed':False,'complete':False,'reason':'native queue not yet observed'}
    if metadata.get('reactive_queue')!=current_queue:return None
    return dict(metadata,live_revision=int(live_revision))
