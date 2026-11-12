"""Opt-in copies of already evaluated forward packet responses.

No BSDF, optical-path or spectral response is reconstructed here. Original RHS
expressions execute once and original assignments remain the integration step.
The caller lends a bounded response SSBO; ordinary shaders have no sidechannel.
"""
import re
from copy import deepcopy
VERSION='evaluated-forward-responses-v1'
ANCESTRY_VERSION='evaluated-forward-responses-v2'
QUEUE_VERSION='native-queued-ancestry-v1'
QUEUE_ENTRY_CONTRACT='studio-prebuilt-forward-reactive-v2'
QUEUE_VALIDATION_BYTES=65536  # at most512 rows of temporary validation columns
FORWARD_ANCESTRY_CONTRACT=dict(version=QUEUE_VERSION,row_words=128,queue_row_words=24,
    root_word=3,queue_ordinal_word=2,parent_dispatch_word=101,parent_ray_word=102,
    parent_bounce_word=103,inverse_replay_word=125,enqueue_cause_word=126,spawn_triangle_word=127,
    enqueue_causes=dict(edge_diffraction=1,fluorescence=2),attachment='native-parent-dispatch',application_ready=False)
BINDING=21
ROW_WORDS=128
ROW_BYTES=512

HELPERS=r'''
layout(std430,binding=21) coherent buffer StudioResponseCopies { uint studio_response_words[]; };
uniform uint uStudioResponseCapacity;
int studio_response_hit_kind=0,studio_response_hit_owner=-1,studio_response_entry_medium=-1;
vec3 studio_response_entry_ro,studio_response_entry_direction;
vec4 studio_response_entry_packet;
vec4 studio_response_before;
vec3 studio_response_direction;
int studio_response_medium;
uint studio_response_active[32];
void studio_response_begin(vec4 packet,vec3 direction,int medium){
 studio_response_before=packet;studio_response_direction=direction;studio_response_medium=medium;
 ACTIVE_BEFORE
}
void studio_response_copy(uint kind,uint path,uint ray,uint root,int bounce,int hit,vec4 packet,vec3 ro,vec3 hp,vec3 direction,int medium,float factor,uint validity,uint termination){
 uint slot=atomicAdd(studio_response_words[0],1u);
 if(slot>=uStudioResponseCapacity){atomicOr(studio_response_words[1],1u);return;}
 uint base=2u+slot*128u;
 // Complete fixed rows, including unused capacity, have deterministic bytes.
 for(uint i=0u;i<128u;++i)studio_response_words[base+i]=0u;
 for(uint j=0u;j<3u;++j){studio_response_words[base+114u+j]=floatBitsToUint(studio_response_entry_direction[j]);studio_response_words[base+117u+j]=floatBitsToUint(studio_response_entry_ro[j]);}
 for(uint j=0u;j<4u;++j)studio_response_words[base+120u+j]=floatBitsToUint(studio_response_entry_packet[j]);
 studio_response_words[base+124u]=uint(studio_response_entry_medium);
 studio_response_words[base+112u]=uint(studio_response_hit_kind);studio_response_words[base+113u]=uint(studio_response_hit_owner);
 studio_response_words[base]=kind;studio_response_words[base+1u]=path;
 studio_response_words[base+2u]=ray;studio_response_words[base+3u]=root;
 studio_response_words[base+4u]=uint(bounce);studio_response_words[base+5u]=uint(hit);
 studio_response_words[base+6u]=hit>=0?floatBitsToUint(load_tri(hit).emissive.z):0xffffffffu;
 studio_response_words[base+7u]=uint(studio_response_medium);studio_response_words[base+8u]=uint(medium);
 studio_response_words[base+9u]=floatBitsToUint(factor);studio_response_words[base+30u]=validity;studio_response_words[base+31u]=termination;
 for(uint i=0u;i<4u;++i){studio_response_words[base+10u+i]=floatBitsToUint(studio_response_before[i]);studio_response_words[base+14u+i]=floatBitsToUint(packet[i]);}
 for(uint i=0u;i<3u;++i){
  studio_response_words[base+18u+i]=floatBitsToUint(ro[i]);studio_response_words[base+21u+i]=floatBitsToUint(hp[i]);
  studio_response_words[base+24u+i]=floatBitsToUint(studio_response_direction[i]);studio_response_words[base+27u+i]=floatBitsToUint(direction[i]);
 }
 for(uint i=0u;i<32u;++i)studio_response_words[base+32u+i]=studio_response_active[i];
 ACTIVE_AFTER
}
'''


def copy_forward_responses(source, *, terminal_only=False, shared_scalar_buffer=False, precleared_rows=False, queue_ancestry=False):
    """Compose only after canonical shader specialization; returns(source,seams)."""
    if 'buffer StudioResponseCopies' in source or 'void studio_response_copy(' in source:raise ValueError('response copies already installed')
    main='void main() {';reactive='void reactive_main(uint gid, inout uint rng) {'
    if source.count(main)!=1 or source.count(reactive)!=1:raise ValueError('forward response function seams changed')
    spheres='bool studio_sphere_hit(' in source
    partitions='bool studio_partition_hit=false;' in source
    if spheres:
        seams_sphere=[('int nearest_filtered(', 'int srf_selected_sphere=-1;\nint nearest_filtered('),
            ('    int best_tri = -1;','    srf_selected_sphere=-1;int best_tri = -1;'),
            ('{best=ht;best_tri=ti;}','{best=ht;best_tri=ti;srf_selected_sphere=si;}'),
            ('                    best_tri = ti;','                    best_tri = ti;srf_selected_sphere=-1;')]
        for old,new in seams_sphere:
            if source.count(old)!=1:raise ValueError('exact native sphere selection seam changed')
            source=source.replace(old,new,1)
    regions='uint studio_active[32];' in source
    helpers=HELPERS.replace('ACTIVE_BEFORE','for(int i=0;i<32;++i)studio_response_active[i]='+('studio_active[i]' if regions else '0u')+';')
    helpers=helpers.replace('ACTIVE_AFTER','for(uint i=0u;i<32u;++i)studio_response_words[base+64u+i]='+('studio_active[i]' if regions else '0u')+';')
    if queue_ancestry:
        if 'uvec4 studio_parent;' not in source:raise ValueError('native queue ancestry composer required')
        close=helpers.rfind('}')
        helpers=helpers[:close]+''' if(path==2u){
 studio_response_words[base+101u]=studio_queue_parent.x;
 studio_response_words[base+102u]=studio_queue_parent.y;
 studio_response_words[base+103u]=studio_queue_parent.w;
 studio_response_words[base+125u]=studio_queue_sampling.x;
 studio_response_words[base+126u]=studio_queue_sampling.y;
 studio_response_words[base+127u]=studio_queue_spawn_triangle;
 }
'''+helpers[close:]
    head,body=source.split(main,1);forward,tail=body.split(reactive,1)
    seams=[]
    def instrument(body,path,ray,root):
        def copy(kind,factor='1.0',validity='1u',direction='rd',termination='0u'):
            if terminal_only and 1<=kind<=4:
                code=f'studio_response_stage_presence|={1<<(kind-1)}u;studio_response_stage_factors[{kind-1}]=({factor});'
                if kind==1:code+='studio_response_beer_before=studio_response_before;studio_response_beer_after=packet;'
                return code
            return f'studio_response_copy({kind}u,{path}u,{ray},{root},bounce,hit,packet,ro,hp,{direction},currentMedium,{factor},{validity},{termination});'
        terminal_status='isnan(packet.z)||isinf(packet.z)?2u:(packet.z==0.0?1u:'+('packet.z<0.002?3u:0u' if 'packet.z < 0.002' in body else '0u')+')'
        bounce_limit='uMaxBounces' if path==1 else 'min(bounces_rem,uMaxBounces)'
        terminal_status='('+terminal_status+')==0u&&bounce+1>='+bounce_limit+'?8u:('+terminal_status+')'
        terminal_valid='isnan(packet.z)||isinf(packet.z)||packet.z<0.0?0u:1u'
        def multiply(match):
            expr=match.group(1).strip();kind=1 if 'studio_medium_transmittance' in expr else 2
            seams.append(dict(kind=kind,path=path,expression=expr))
            outgoing='outgoing' if 'studio_pbr_weight(' in expr or expr.startswith('weight*studio_reflector_rho(') else 'rd'
            terminal=copy(5,validity=terminal_valid,direction=outgoing,termination=terminal_status) if 'studio_pbr_weight(' in expr else ''
            return '{ studio_response_begin(packet,rd,currentMedium); float studio_copied_factor=('+expr+'); packet.z*=studio_copied_factor; '+copy(kind,'studio_copied_factor',direction=outgoing)+terminal+' }'
        hit_seam='int hit = nearest_hit(ro, rd, best);'
        kind=('(srf_selected_sphere>=0?1:0)' if spheres else '0')
        owner=('srf_selected_sphere' if spheres else '-1')
        if partitions:
            kind='(studio_partition_hit?2:'+kind+')'
            owner='(studio_partition_hit?studio_partition_owner_triangle:'+owner+')'
        if body.count(hit_seam)!=1:raise ValueError('actual forward recipient seam changed')
        body=body.replace(hit_seam,'studio_response_entry_ro=ro;studio_response_entry_direction=rd;studio_response_entry_packet=packet;studio_response_entry_medium=currentMedium;'+hit_seam+'studio_response_hit_kind='+kind+';studio_response_hit_owner='+owner+';',1)
        escape='if (hit < 0) break;'
        if body.count(escape)!=1:raise ValueError('native escape seam changed')
        body=body.replace(escape,'if(hit<0){vec3 hp=ro;studio_response_begin(packet,rd,currentMedium);'+copy(6,termination='5u')+'break;}',1)
        body=body.replace('if(weight<=0.0)break;','if(weight<=0.0){studio_response_begin(packet,rd,currentMedium);'+copy(6,validity='0u',termination='6u')+'break;}')
        body=re.sub(r'packet\.z\s*\*=\s*([^;]+);',multiply,body)
        boundary='if(!studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor)) break;'
        if body.count(boundary)!=1:raise ValueError('actual boundary response seam changed')
        body=body.replace(boundary,'studio_response_begin(packet,rd,currentMedium); bool studio_boundary_accepted=studio_boundary_step(tri,rd,packet.x,rand01(rng),currentMedium,factor); '+copy(3,'studio_boundary_accepted?factor:0.0','studio_boundary_accepted?1u:0u')+' if(!studio_boundary_accepted){'+copy(6,validity='0u',termination='4u')+'break;} '+copy(5,validity=terminal_valid,termination=terminal_status))
        wave='packet = apply_wave_context_physics(ro, rd, best, packet);'
        if wave in body:
            if body.count(wave)!=1:raise ValueError('actual wave-context response seam changed')
            body=body.replace(wave,'studio_response_begin(packet,rd,currentMedium); '+wave+' '+copy(4))
        phase='packet.y += best * packet.x * 0.000021;'
        if body.count(phase)!=2:raise ValueError('terminal common/manifold seams changed')
        body=body.replace(phase,phase+' '+copy(5,validity=terminal_valid,termination=terminal_status))
        absorber='if (('+('mat_flags_bits' if path==1 else 'flags')+' & MAT_FLAG_ABSORBER) != 0u) break;'
        if body.count(absorber)!=1:raise ValueError('native absorber termination seam changed')
        body=body.replace(absorber,absorber.replace('break;','{studio_response_begin(packet,rd,currentMedium);'+copy(6,validity='0u',termination='7u')+'break;}'))
        sample='if(!studio_pbr_sample(tri,incoming,n,vec3(rand01(rng),rand01(rng),rand01(rng)),outgoing,pdf,delta))break;'
        if sample in body:
            body=body.replace(sample,'studio_response_begin(packet,rd,currentMedium);bool studio_sample_accepted=studio_pbr_sample(tri,incoming,n,vec3(rand01(rng),rand01(rng),rand01(rng)),outgoing,pdf,delta);if(!studio_sample_accepted){'+copy(6,validity='0u',termination='6u')+'break;}')
        if terminal_only:
            loop=re.compile(r'(for \(int bounce = 0; bounce < [^\n]+; \+\+bounce\) \{)')
            body,n=loop.subn(r'\1 studio_response_stage_presence=0u;',body)
            if n!=1:raise ValueError('native bounce reset seam changed')
        return body
    forward=instrument(forward,1,'ray_id','uint(_ray_idx)')
    tail=instrument(tail,2,'gid','studio_queue_parent.z' if queue_ancestry else '0xffffffffu')
    if sum(s['kind']==1 for s in seams)!=2:raise ValueError('actual Beer response count changed')
    if terminal_only:
        helpers=helpers.replace('vec4 studio_response_before;','uint studio_response_stage_presence=0u;\nfloat studio_response_stage_factors[4];\nvec4 studio_response_beer_before,studio_response_beer_after;\nvec4 studio_response_before;')
        anchor=' ACTIVE_AFTER'
        # ACTIVE_AFTER was already specialized above: append copies before the
        # helper's final brace without changing actual packet arithmetic.
        close=helpers.rfind('}')
        helpers=helpers[:close]+''' studio_response_words[base+96u]=studio_response_stage_presence;
 for(uint j=0u;j<4u;++j){
  studio_response_words[base+97u+j]=floatBitsToUint((studio_response_stage_presence&(1u<<j))!=0u?studio_response_stage_factors[j]:0.0);
  studio_response_words[base+104u+j]=floatBitsToUint((studio_response_stage_presence&1u)!=0u?studio_response_beer_before[j]:0.0);
  studio_response_words[base+108u+j]=floatBitsToUint((studio_response_stage_presence&1u)!=0u?studio_response_beer_after[j]:0.0);
 }
'''+helpers[close:]
    if precleared_rows:
        zero=' for(uint i=0u;i<128u;++i)studio_response_words[base+i]=0u;'
        if helpers.count(zero)!=1:raise ValueError('row initialization seam changed')
        helpers=helpers.replace(zero,' // Unused words are zeroed by the owner in this exact transient range.',1)
    if shared_scalar_buffer:
        declaration='layout(std430,binding=21) coherent buffer StudioResponseCopies { uint studio_response_words[]; };'
        if declaration not in helpers:raise ValueError('response shared storage seam changed')
        helpers=helpers.replace(declaration,'uniform uint uStudioResponseWordOffset;\n#define studio_response_words studio_scalar_visits')
        helpers=helpers.replace('studio_response_words[0]','studio_response_words[uStudioResponseWordOffset]').replace('studio_response_words[1]','studio_response_words[uStudioResponseWordOffset+1u]').replace('uint base=2u+slot*128u;','uint base=uStudioResponseWordOffset+2u+slot*128u;')
    return head+helpers+'\n'+main+forward+reactive+tail,seams


ROW_BYTES=128*4
HEADER_BYTES=8


def queued_ray_storage_plan(ray_count,bounce_limit,*,max_ssbo_bytes):
    """Bound the existing three edge + one fluorescent enqueue opportunities.

    The last bounce has no remaining child budget. Admission fails instead of
    changing forward jobs/bounces or silently dropping secondary paths.
    """
    if any(type(v) is not int or v<1 for v in (ray_count,bounce_limit,max_ssbo_bytes)):
        raise ValueError('invalid queued-ray storage admission')
    capacity=4*ray_count*(bounce_limit-1)
    size=max(96,capacity*96)
    if capacity>0x7fffffff or size>max_ssbo_bytes:
        raise MemoryError('native queued-ray ancestry exceeds SSBO admission')
    return dict(version=QUEUE_VERSION,capacity=capacity,row_words=24,buffer_bytes=size)


def with_queued_ray_ancestry(source):
    """Opt-in extension of PendingRay, preserving its original first64 bytes.

    All optical/RNG expressions and enqueue order stay native. The dispatcher
    supplies the actual parent token and logical enqueue base; no lineage is
    inferred from geometry or secondary packet values.
    """
    if 'studio_queue_parent' in source:raise ValueError('queue ancestry already installed')
    if 'float studio_record_root_weight=0.0;' not in source:
        raise ValueError('native root-sweep recorder required for copied replay weight')
    if (source.count('spawn_edge_diffraction(hp, rd, packet, bounce,')!=3 or
        source.count('append_pending_ray(')!=4 or
        source.count('int bounces_rem = uMaxBounces - bounce - 1;')!=2 or
        source.count('atomicAdd(pending_count, 1u)')!=1):
        raise ValueError('native enqueue count/bounce bounds changed; storage plan must be re-established')
    def once(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('native queue ancestry seam changed: '+old[:70])
        source=source.replace(old,new,1)
    once('    vec4 normal_triid;   // .xyz = hit normal at spawn, .w = source tri index',
         '    vec4 normal_triid;   // .xyz = hit normal at spawn, .w = source tri index\n'
         '    uvec4 studio_parent; // actual dispatch, parent ray, selected root, bounce\n'
         '    uvec4 studio_sampling; // copied replay weight bits, enqueue cause, reserved0, reserved0')
    once('void append_pending_ray(vec3 origin, vec3 dir, vec4 packet,\n                        int bounces_rem, uint pray_flags,\n                        vec3 hit_normal, int tri_id);',
         'uniform uint uStudioParentDispatch;\nuniform uint uStudioQueueSequenceBase;\n'
         'uvec4 studio_queue_parent=uvec4(0u),studio_queue_sampling=uvec4(0u);\n'
         'uint studio_queue_cause=0u,studio_queue_spawn_triangle=0xffffffffu;\n'
         'void append_pending_ray(vec3 origin, vec3 dir, vec4 packet,\n                        int bounces_rem, uint pray_flags,\n                        vec3 hit_normal, int tri_id);')
    once('    int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));',
         '    int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));\n'
         '    studio_queue_parent=uvec4(uStudioParentDispatch,ray_id,uint(_ray_idx),0u);')
    loop='    float path_len = 0.0;   // accumulated path length for near-field and context spreading\n    for (int bounce = 0; bounce < uMaxBounces; ++bounce) {'
    once(loop,loop+'\n'
         '        studio_queue_parent.w=uint(bounce);')
    once('    append_pending_ray(edge_pt + diffr_dir * 1e-5, diffr_dir, diffr_pkt,',
         '    studio_queue_cause=1u;\n    append_pending_ray(edge_pt + diffr_dir * 1e-5, diffr_dir, diffr_pkt,')
    once('                append_pending_ray(hp + react_dir * 1e-5, react_dir, react_pkt,',
         '                studio_queue_cause=2u;\n                append_pending_ray(hp + react_dir * 1e-5, react_dir, react_pkt,')
    once('    pending_rays[idx].normal_triid = vec4(hit_normal, uintBitsToFloat(uint(tri_id)));',
         '    pending_rays[idx].normal_triid = vec4(hit_normal, uintBitsToFloat(uint(tri_id)));\n'
         '    pending_rays[idx].studio_parent=studio_queue_parent;\n'
         '    pending_rays[idx].studio_sampling=uvec4(floatBitsToUint(studio_record_root_weight),studio_queue_cause,0u,0u);')
    once('    PendingRay pr      = pending_rays[gid];',
         '    PendingRay pr      = pending_rays[gid];\n'
         '    studio_queue_parent=pr.studio_parent;studio_queue_sampling=pr.studio_sampling;\n'
         '    studio_queue_spawn_triangle=floatBitsToUint(pr.normal_triid.w);\n'
         '    studio_record_root_weight=uintBitsToFloat(pr.studio_sampling.x);')
    once('    if (gid >= count) return;', '    if (gid >= count || gid >= uint(uPendingCap)) return;')
    once('        reactive_main(gid, rng);','        reactive_main(ray_id, rng);')
    once('    uint rng = hash_u(ray_id ^ uint(uSeed));',
         '    uint rng = hash_u((ray_id+(uMode==PASS_REACTIVE?uStudioQueueSequenceBase:0u)) ^ uint(uSeed));')
    sensor='''    if (uMode == PASS_SENSOR) {
        sensor_main(ray_id, rng);
        return;
    }'''
    once(sensor,'    if (uMode != PASS_FORWARD && uMode != PASS_REACTIVE) return;')
    return source


def dispatch_storage_plan(ray_count, bounce_limit, *, reserved_bytes, consumer_bytes=0,
                          budget_bytes=512*1024*1024, max_ssbo_bytes=128*1024*1024):
    """Preflight terminal copies only; never bound volume-march history this way.

    Peak includes the GPU buffer and one synchronous immutable readback. The
    consumer must declare its additional staging/retained storage separately.
    Original ray count and bounce schedule are not altered to make admission fit.
    """
    args=(ray_count,bounce_limit,reserved_bytes,consumer_bytes,budget_bytes,max_ssbo_bytes)
    if any(type(x) is not int for x in args) or min(ray_count,bounce_limit,budget_bytes,max_ssbo_bytes)<1 or min(reserved_bytes,consumer_bytes)<0:
        raise ValueError('invalid explicit response storage admission')
    rows=ray_count*bounce_limit
    if rows>0xfffffffe:raise MemoryError('response row counter capacity exceeded')
    buffer_bytes=HEADER_BYTES+rows*ROW_BYTES
    peak=reserved_bytes+consumer_bytes+2*buffer_bytes
    if buffer_bytes>max_ssbo_bytes:
        raise MemoryError(f'terminal response buffer requires {buffer_bytes} bytes; SSBO limit {max_ssbo_bytes}')
    if peak>budget_bytes:
        raise MemoryError(f'terminal response peak requires {peak} bytes; budget {budget_bytes}')
    return dict(capacity_rows=rows,buffer_bytes=buffer_bytes,readback_bytes=buffer_bytes,
                reserved_bytes=reserved_bytes,consumer_bytes=consumer_bytes,peak_bytes=peak)


class ForwardResponseCopies:
    """Transient, synchronous terminal-copy owner. No history or estimator.

    The dispatcher supplies unchanged native work bounds and exact ancestry;
    its consumer must retain typed scene records without interpreting factors
    as energy. This class does not dispatch rays or change RNG/scheduling.
    """
    def __init__(self, *, ray_count, bounce_limit, reserved_bytes, consume,
                 consumer_bytes=0, budget_bytes=512*1024*1024, gl=None,borrowed_buffer=None,offset_bytes=0,queue_ancestry=False):
        if not callable(consume):raise TypeError('synchronous scene response consumer required')
        if type(queue_ancestry) is not bool:raise ValueError('queue ancestry must be an explicit boolean')
        self.queue_ancestry=queue_ancestry
        if gl is None:
            from OpenGL import GL as gl
        self.gl=gl;self.consume=consume;self.handle=0;self.pending=None
        self.borrowed=borrowed_buffer is not None;self.offset_bytes=int(offset_bytes)
        if self.offset_bytes<0 or self.offset_bytes%4:raise ValueError("unaligned auxiliary response offset")
        if int(gl.glGetIntegerv(gl.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<=21:
            raise MemoryError('forward response copies require binding21')
        self.plan=dispatch_storage_plan(ray_count,bounce_limit,reserved_bytes=reserved_bytes,
            consumer_bytes=consumer_bytes+(QUEUE_VALIDATION_BYTES if queue_ancestry else 0),budget_bytes=budget_bytes,
            max_ssbo_bytes=int(gl.glGetIntegerv(gl.GL_MAX_SHADER_STORAGE_BLOCK_SIZE)))
        self.ray_count=ray_count;self.bounce_limit=bounce_limit
        if self.borrowed:
            self.handle=int(borrowed_buffer)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.handle)
            if int(gl.glGetBufferParameteriv(gl.GL_SHADER_STORAGE_BUFFER,gl.GL_BUFFER_SIZE))<self.offset_bytes+self.plan["buffer_bytes"]:
                raise MemoryError("borrowed response storage shorter than admitted layout")
            return
        self.handle=int(gl.glGenBuffers(1))
        try:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.handle)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,self.plan['buffer_bytes'],None,gl.GL_DYNAMIC_DRAW)
            if int(gl.glGetBufferParameteriv(gl.GL_SHADER_STORAGE_BUFFER,gl.GL_BUFFER_SIZE))!=self.plan['buffer_bytes']:
                raise MemoryError('forward response buffer allocation incomplete')
        except BaseException:
            self.close();raise

    def begin(self, program, metadata):
        import numpy as np
        if self.pending is not None:raise RuntimeError('previous response dispatch not consumed')
        required=('scene_revision','population_index','source_sha256','dispatch_id','path_family',
                  'ray_offset','ray_count','bounce_limit','seed','root_sampling')
        if any(k not in metadata for k in required):raise ValueError('exact response dispatch ancestry required')
        if metadata['path_family'] not in ('forward','reactive'):
            raise ValueError('response family must match actual native dispatch')
        if not 0<metadata['ray_count']<=self.ray_count or metadata['bounce_limit']!=self.bounce_limit:
            raise ValueError('native dispatch exceeds preflight; schedule must not be reduced')
        if metadata['path_family']=='reactive':
            expected='copied-native-queue-v1' if self.queue_ancestry else 'unavailable-native-queue'
            if metadata.get('root_identity')!=expected:raise ValueError('reactive root identity does not match queue contract')
        if self.queue_ancestry:
            if metadata.get('forward_ancestry_contract')!=FORWARD_ANCESTRY_CONTRACT:
                raise ValueError('exact queued ancestry contract required')
            validate_queue_metadata(metadata)
        g=self.gl;g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,self.handle)
        # Zero only our transient range once, avoiding a large per-row GLSL
        # initialization loop at every terminal/disposition call site.
        g.glClearBufferSubData(g.GL_SHADER_STORAGE_BUFFER,g.GL_R32UI,self.offset_bytes,self.plan['buffer_bytes'],
            g.GL_RED_INTEGER,g.GL_UNSIGNED_INT,np.zeros(1,np.uint32))
        g.glBindBufferBase(g.GL_SHADER_STORAGE_BUFFER,23 if self.borrowed else 21,self.handle)
        if self.borrowed:g.glUniform1ui(g.glGetUniformLocation(program,'uStudioResponseWordOffset'),self.offset_bytes//4)
        g.glUniform1ui(g.glGetUniformLocation(program,'uStudioResponseCapacity'),self.plan['capacity_rows'])
        self.pending=dict(metadata)

    def finish(self):
        import numpy as np
        if self.pending is None:raise RuntimeError('no actual response dispatch pending')
        g=self.gl;g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT|g.GL_BUFFER_UPDATE_BARRIER_BIT)
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,self.handle)
        header=np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,self.offset_bytes,8),dtype=np.uint32)
        count,overflow=map(int,header)
        if overflow or count>self.plan['capacity_rows']:
            self.pending=None
            raise MemoryError('actual terminal response overflow; no complete publication permitted')
        raw=g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,self.offset_bytes+8,count*ROW_BYTES) if count else b''
        rows=np.frombuffer(raw,dtype=np.uint32).reshape(count,128)
        rows.setflags(write=False)
        metadata=dict(self.pending,version=ANCESTRY_VERSION if self.queue_ancestry else VERSION,row_count=count,row_stride_bytes=ROW_BYTES,
                      row_semantics='terminal packet or native disposition; factors are attributes',
                      independent_sample_count=None)
        if self.queue_ancestry:validate_queue_rows(metadata,rows)
        try:self.consume(metadata,rows)
        finally:self.pending=None
        return count

    def close(self):
        if self.handle:
            if not self.borrowed:self.gl.glDeleteBuffers(1,[self.handle])
            self.handle=0
        self.pending=None


def validate_queue_metadata(metadata):
    q=metadata.get('dispatch',{}).get('native_queue',{})
    if q.get('version')!=QUEUE_VERSION:raise ValueError('actual native queue dispatch required')
    for key in ('generation','logical_ray_base','capacity','source_root_count'):
        if type(q.get(key)) is not int or not 0<=q[key]<=0xffffffff:raise ValueError('invalid native queue '+key)
    parent=q.get('parent_dispatch',{})
    if q['generation']!=parent.get('dispatch_id') or q['source_root_count']<1:
        raise ValueError('native queue parent identity mismatch')
    for key in ('ray_offset','ray_count','total_rays','bounce_limit','population_index','seed'):
        if type(parent.get(key)) is not int or not 0<=parent[key]<=0x7fffffff:raise ValueError('invalid parent dispatch '+key)
    if parent['ray_count']<1 or parent['bounce_limit']<1 or parent['ray_offset']+parent['ray_count']>parent['total_rays']:
        raise ValueError('invalid parent dispatch bounds')
    if q['logical_ray_base']+q['capacity']>0xffffffff:raise ValueError('queue logical ray identity exhausted')
    return q


def validate_queue_rows(metadata,rows):
    """Validate copied identities, never repair them or reconstruct packets."""
    import numpy as np
    q=validate_queue_metadata(metadata);parent=q['parent_dispatch']
    expected=1 if metadata['path_family']=='forward' else 2
    for start in range(0,len(rows),512):
        _validate_queue_row_chunk(metadata,rows[start:start+512],q,parent,expected)


def _validate_queue_row_chunk(metadata,rows,q,parent,expected):
    import numpy as np
    if np.any(rows[:,1]!=expected):raise ValueError('response family disagrees with actual dispatch')
    if expected==1:
        if any(np.any(rows[:,word]) for word in (101,102,103,125,126,127)):
            raise ValueError('forward rows cannot claim queued parent fields')
        return
    replay=rows[:,125].copy().view(np.float32)
    if (np.any(rows[:,101]!=q['generation']) or np.any(rows[:,102]<parent['ray_offset']) or
        np.any(rows[:,102]>=parent['ray_offset']+parent['ray_count']) or
        np.any(rows[:,3]!=rows[:,102]%q['source_root_count']) or
        np.any(rows[:,103]>=parent['bounce_limit']-1) or not np.isfinite(replay).all() or
        np.any(replay<0) or np.any(replay>1) or np.any(~np.isin(rows[:,126],[1,2])) or
        np.any(rows[:,127]>=0x80000000) or np.any(rows[:,2]<metadata['ray_offset']) or
        np.any(rows[:,2]>=metadata['ray_offset']+metadata['ray_count'])):
        raise ValueError('copied reactive ancestry disagrees with actual parent/queue dispatch')


def require_queue_idle(state):
    """Pre-mutation guard for the existing owner; never silently drain/retry."""
    if state.get('response_entry_contract')!=QUEUE_ENTRY_CONTRACT:return
    q=state.get('response_queue_owner')
    if q is None or q.get('phase')!='idle':
        raise RuntimeError('native queue unfinished or uncertain; destroy/recreate the field before reuse')


def abort_native_queue(state):
    if state.get('response_entry_contract')!=QUEUE_ENTRY_CONTRACT:return
    state['response_queue_owner']['phase']='uncertain'
    receipt=state.get('response_reactive_queue')
    if receipt is not None:receipt.update(complete=False,incomplete_observed=True)


def begin_native_queue(state,metadata):
    require_queue_idle(state)
    q=state['response_queue_owner'];g=state['forward_response_owner'].gl
    parent=deepcopy(metadata['dispatch'])
    queue=dict(version=QUEUE_VERSION,generation=metadata['dispatch_id'],logical_ray_base=q['logical_ray_base'],
        capacity=state['_pending_cap'],source_root_count=len(state['source_buf']),parent_dispatch=parent)
    metadata['dispatch']['native_queue']=queue
    metadata['forward_ancestry_contract']=deepcopy(FORWARD_ANCESTRY_CONTRACT)
    validate_queue_metadata(metadata)
    # A previous successful drain reset exactly counter word4. Check before
    # another producer, never reinterpret an unexpected backlog as current work.
    import numpy as np
    g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT|g.GL_BUFFER_UPDATE_BARRIER_BIT)
    g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,state['ssbo'][4])
    attempted=int(np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,16,4),np.uint32)[0])
    if attempted:
        q['phase']='uncertain';raise RuntimeError('unexpected native queue backlog; recreate field')
    g.glUniform1ui(g.glGetUniformLocation(state['prog'],'uStudioParentDispatch'),queue['generation'])
    g.glUniform1ui(g.glGetUniformLocation(state['prog'],'uStudioQueueSequenceBase'),queue['logical_ray_base'])
    q.update(phase='forward',parent=deepcopy(metadata),cursor=0)


def _queue_receipt(state,attempted,dispatched):
    q=state['response_queue_owner']
    result=native_queue_status(attempted,state['_pending_cap'],dispatched=dispatched)
    result.update(observed=True,counter_scope='one actual parent forward dispatch; reset only after consumed drain',
        version=QUEUE_VERSION,generation=q['parent']['dispatch_id'],logical_ray_base=q['logical_ray_base'],
        source_root_identity='copied-native-queue-v1',incomplete_observed=q['phase']=='uncertain',counter_regression=False)
    if result['incomplete_observed']:result['complete']=False
    state['response_reactive_queue']=deepcopy(result)
    return result


def _finish_ancestral_queue(owner,state,restore,attempted):
    """Bounded consumption by the prebuilt owner, using its original mode2 pass.

    No new roots are counted. Any exception poisons this additive field: a
    failed readback/consumer may follow real execution and cannot be replayed.
    """
    import numpy as np
    q=state['response_queue_owner'];g=owner.gl
    if q['phase']!='forward':raise RuntimeError('native queue has no actual parent dispatch')
    parent=q['parent'];stored=min(attempted,state['_pending_cap'])
    try:
        owner.pending['reactive_queue']=_queue_receipt(state,attempted,0)
        owner.pending['dispatch']['reactive_queue']=deepcopy(owner.pending['reactive_queue'])
        forward_count=owner.finish()
        if attempted>state['_pending_cap']:raise MemoryError('native secondary queue overflow; field incomplete')
        q['phase']='reactive'
        require_prebuilt_mode(state,2)
        restore()
        while q['cursor']<stored:
            offset=q['cursor'];count=min(owner.ray_count,stored-offset)
            metadata=deepcopy(parent)
            serial=state['response_dispatch_serial'];state['response_dispatch_serial']+=1
            metadata.update(sequence=serial,dispatch_id=serial,path_family='reactive',ray_offset=offset,
                ray_count=count,seed=31415926,root_identity='copied-native-queue-v1')
            metadata['dispatch'].update(dispatch_id=serial,ray_offset=offset,ray_count=count,seed=31415926,
                path_family='reactive')
            metadata['reactive_queue']=_queue_receipt(state,attempted,offset)
            metadata['dispatch']['reactive_queue']=deepcopy(metadata['reactive_queue'])
            g.glUniform1i(state['locs']['mode'],2)
            g.glUniform1i(state['locs']['seed'],31415926)
            g.glUniform1i(state['locs']['batch_size'],count)
            g.glUniform1i(state['locs']['batch_off'],offset)
            owner.begin(state['prog'],metadata)
            groups=(count+127)//128
            g.glDispatchCompute(min(groups,65535),(groups+65534)//65535,1)
            g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT|g.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            owner.finish()  # synchronous readback + consumer must both succeed
            q['cursor']+=count
            restore()
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,state['ssbo'][4])
        after=int(np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,16,4),np.uint32)[0])
        if after!=attempted:raise RuntimeError('reactive pass changed the single-generation native queue')
        # Only this counter, not scalar/diagnostic histories or response views.
        g.glBufferSubData(g.GL_SHADER_STORAGE_BUFFER,16,4,np.zeros(1,np.uint32))
        if int(np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,16,4),np.uint32)[0]):
            raise RuntimeError('native queue reset not observed; recreate field')
        _queue_receipt(state,attempted,stored)
        q.update(phase='idle',logical_ray_base=q['logical_ray_base']+attempted)
        q.pop('parent',None)
        return forward_count
    except BaseException:
        q['phase']='uncertain'
        _queue_receipt(state,attempted,q['cursor'])
        raise
    finally:
        try:
            restore()
            g.glUniform1i(state['locs']['mode'],0)
            g.glUniform1i(state['locs']['seed'],parent['seed'])
        except BaseException:
            abort_native_queue(state)
            raise


def native_queue_status(attempted, capacity, *, dispatched=0):
    if any(type(v) is not int or v<0 for v in (attempted,capacity,dispatched)):
        raise ValueError('invalid actual queue counters')
    stored=min(attempted,capacity)
    if dispatched>stored:raise ValueError('reactive dispatch exceeds native stored queue')
    return dict(attempted=attempted,capacity=capacity,stored=stored,
                dropped_enqueue=max(0,attempted-capacity),dispatched=dispatched,
                pending_unprocessed=stored-dispatched,
                source_root_identity='unavailable-native-queue',
                complete=attempted<=capacity and stored==dispatched)


def finish_native_dispatch(owner,state,restore):
    """Observe legacy queues; the explicit v2 owner also consumes its queue."""
    import numpy as np
    g=owner.gl
    g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT|g.GL_BUFFER_UPDATE_BARRIER_BIT)
    g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,state['ssbo'][4])
    attempted=int(np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,16,4),np.uint32)[0])
    if state.get('response_entry_contract')==QUEUE_ENTRY_CONTRACT:
        return _finish_ancestral_queue(owner,state,restore,attempted)
    owner.pending['reactive_queue']=record_queue_observation(state,attempted)
    owner.pending.setdefault('dispatch',{})['reactive_queue']=owner.pending['reactive_queue']
    try:return owner.finish()
    finally:restore()


def shader_storage_requirements(source):
    """Conservative declared compute-block count, before hardware link."""
    declarations=re.findall(r'\bbuffer\s+([A-Za-z_]\w*)\s*\{',source)
    return dict(declared_compute_storage_blocks=len(declarations),block_names=declarations,
                minimum_binding_count=(24 if '#define studio_response_words studio_scalar_visits' in source else 26),hardware_link_verified=False)


def admit_shader_storage(source,max_compute_blocks):
    receipt=shader_storage_requirements(source)
    if type(max_compute_blocks) is not int or max_compute_blocks<receipt['declared_compute_storage_blocks']:
        raise MemoryError(f"opt-in response shader requires up to {receipt['declared_compute_storage_blocks']} compute storage blocks; hardware limit {max_compute_blocks}")
    return receipt


PREBUILT_ENTRY_CONTRACT = 'studio-prebuilt-forward-only-v1'

def specialize_prebuilt_forward(source):
    """Remove unreachable entry calls, not any forward body or queued work.

    Only the prebuilt owner may use this source. Its original warmup and pumps
    set mode zero; pending reactive entries remain pending, never dispatched.
    """
    branches = """    if (uMode == PASS_SENSOR) {
        sensor_main(ray_id, rng);
        return;
    }
    if (uMode == PASS_REACTIVE) {
        // gid indexes directly into PendingRayBuf; no source-based ray gen.
        reactive_main(gid, rng);
        return;
    }"""
    if source.count(branches) != 1:
        raise ValueError('prebuilt forward entry dispatch seam changed')
    return source.replace(branches, '    if (uMode != PASS_FORWARD) return; // owner rejects unsupported mode')

def require_prebuilt_mode(state, mode):
    if state.get('response_entry_contract') == PREBUILT_ENTRY_CONTRACT and mode != 0:
        raise ValueError('specialized prebuilt response state supports PASS_FORWARD only; reactive queue remains unprocessed')
    if state.get('response_entry_contract')==QUEUE_ENTRY_CONTRACT and mode not in (0,2):
        raise ValueError('ancestral prebuilt queue supports forward/reactive only')


def record_queue_observation(state, attempted):
    """Copy actual cumulative native counter; a scalar clear does not drain it."""
    result=native_queue_status(attempted,int(state['_pending_cap']))
    previous=state.get('response_reactive_queue')
    regressed=bool(previous and attempted<previous['attempted'])
    incomplete=bool(not result['complete'] or regressed or (previous and previous.get('incomplete_observed',False)))
    result.update(observed=True, counter_scope='native queue allocation lifetime; not scalar epoch',
                  counter_regression=regressed, incomplete_observed=incomplete,
                  complete=result['complete'] and not incomplete)
    state['response_reactive_queue']=dict(result)
    return result
