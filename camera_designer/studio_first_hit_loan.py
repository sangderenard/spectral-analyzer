"""Experimental first-intersection continuation for the composed sensor.

No alternative camera or intersection law. Capture executes the existing
camera launch, and resume executes the same frozen launch and transport.
Not a public network endpoint: callers must validate untrusted query results.
"""
import hashlib
import time
import uuid
import threading
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class VerifiedFirstHits:
    """Trusted validator's canonical records, never constructed from a client."""
    hits: np.ndarray


def loan_shader(source):
    """Wrap the existing sensor at its exact launch/triangle query seams."""
    declaration = '''
layout(std430,binding=16) buffer StudioLoanRays { vec4 studio_loan_rays[]; };
layout(std430,binding=17) readonly buffer StudioLoanHits { vec4 studio_loan_hits[]; };
uniform int uStudioLoanPhase;
bool studio_loan_primary=false;
int studio_loan_slot=0;
'''
    launch='        float lum[8];'
    query='            int   hit_id = nearest_hit(ro, rd, surf_t);'
    traversal='    int stack[96];'
    for marker in (launch, query, traversal):
        if source.count(marker)!=1:
            raise RuntimeError('Composed sensor continuation seam changed: '+marker)
    source=source.replace('int nearest_filtered(', declaration+'\nint nearest_filtered(',1)
    source=source.replace(traversal, '''    if(studio_loan_primary && uStudioLoanPhase==2 && !boundary_only) {
        vec4 candidate=studio_loan_hits[studio_loan_slot];
        if(candidate.z>0.5 && candidate.x<best) {best=candidate.x;best_tri=int(candidate.y);}
        return best_tri; // analytic sphere candidate above remains in the competition
    }
'''+traversal,1)
    source=source.replace(launch, '''        studio_loan_slot=(px.y-uRowOffset)*uSensorSize.x+px.x;
        if(uStudioLoanPhase==1) {
            studio_loan_rays[2*studio_loan_slot]=vec4(ro,1e-6);
            studio_loan_rays[2*studio_loan_slot+1]=vec4(rd,1e30);
            return; // no sensor, history, evidence accumulation or sample advancement
        }
'''+launch,1)
    source=source.replace(query, '''            studio_loan_primary=(bounce==0);
'''+query+'''
            studio_loan_primary=false;
''',1)
    if 'float studio_region_epsilon(vec3 p) {' in source:
        source=source.replace('vec4(ro,1e-6);','vec4(ro,studio_region_epsilon(ro)/max(length(rd),1e-30));',1)
    return source


class FirstHitLoan:
    """One authoritative pending strip, owned by its existing accumulator.

    Expiry invalidates the token, not the sample. Reissue uses the retained
    launch; cancel leaves the same strip available for normal server fallback.
    Explicit region-bvh preserves original server sphere/partition/transport
    and lends only the primary unfiltered triangle candidate query.
    """
    def __init__(self, accumulator, source, compile_program, scene_revision, *, query_variant="plain"):
        from OpenGL import GL as gl
        self.owner_thread=threading.get_ident()
        a=accumulator
        if getattr(a,'_intersection_loan',None) is not None:
            raise RuntimeError('Sensor already has pending intersection work')
        if query_variant not in ('plain','plain-bvh','region-bvh'):raise ValueError('Unsupported query variant')
        if bool(a._n_regions)!=(query_variant=='region-bvh'):raise ValueError('Region geometry requires exact region-bvh query variant')
        if ('float studio_region_epsilon(vec3 p) {' in source)!=(query_variant=='region-bvh'):
            raise ValueError('Composed shader does not match declared query variant')
        self.query_variant=query_variant
        if a._sprinkle or a._samples_per_pixel!=1 or a._sample_evidence is not None:
            raise ValueError('First-hit loan requires unique rows, one sample, no explicit regions or evidence')
        self.count=a._w*min(a._rows_per_frame,a._h-a._row_off)
        triangle_limit=4096 if query_variant=='plain' else 200000
        if not 1<=self.count<=65536 or not 0<=a._n_tris<=triangle_limit:
            raise ValueError('First-hit client ABI ray/triangle bound exceeded')
        if int(gl.glGetIntegerv(gl.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS))<18:
            raise RuntimeError('First-hit loan requires 18 SSBO binding slots')
        self.a=a;self.source=loan_shader(source);self.original=a._prog
        self.program=0;self.buffers=[];self.state='preparing';self.revision=str(scene_revision)
        self.signature=self._signature();self.token=None;self.deadline=0
        self.preparation_profile={};self.freeze_check_seconds=0.;self.freeze_checks=0
        def timed(name,operation):
            start=time.perf_counter()
            try:return operation()
            finally:self.preparation_profile[name]=time.perf_counter()-start
        self.geometry,self.geometry_hash=timed('geometry_readback_pack_seconds',self._geometry)
        from .studio_query_geometry import query_geometry_identity
        self.query_identity=timed('query_identity_seconds',lambda:query_geometry_identity(self.geometry,self.bvh_nodes,self.bvh_ids))
        self.resource_hash=timed('resource_readback_hash_seconds',self._resources)
        try:
            self.program=timed('shader_compile_seconds',lambda:compile_program((self.source,gl.GL_COMPUTE_SHADER)))
            if not gl.glGetProgramiv(self.program,gl.GL_LINK_STATUS):
                raise RuntimeError('First-hit continuation program failed to link')
            for size in (self.count*32,self.count*16):
                b=int(gl.glGenBuffers(1));self.buffers.append(b)
                gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,b)
                gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,size,None,gl.GL_DYNAMIC_DRAW)
            a._intersection_loan=self
            a._prog=self.program
            capture_start=time.perf_counter()
            a._dispatch_once(_loan_phase=1)
            gl.glFinish();self.preparation_profile['capture_gpu_seconds']=time.perf_counter()-capture_start
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.buffers[0])
            self.rays=np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,self.count*32),np.float32).copy().reshape(-1,8)
            if not np.isfinite(self.rays).all():raise RuntimeError('Nonfinite exported rays')
            self.state='pending'
        except Exception:
            self.close();raise
        finally:a._prog=self.original

    def _signature(self):
        a=self.a
        # Mutating this accumulator while a strip is on loan is never allowed.
        keys=('_row_off','_pass','_frame','_eye','_right','_up','_fwd','_fov_tan','_aspect',
              '_aperture_radius','_focus_dist','_tilt_shift','_lens_tilt','_n_blades','_aperture_rot',
              '_rows_per_frame','_history_samples','_history_checkpoint','_max_bounces',
              '_ssbo_tris','_ssbo_nodes','_ssbo_ids','_ssbo_shade','_ssbo_mat','_ssbo_spheres',
              '_ssbo_regions','_ssbo_native_pbr','_n_tris','_n_bvh_nodes','_n_spheres','_n_regions','_w','_h','_vs','_rfs','_rfg',
              '_va','_air_ds','_air_ss','_air_an','_medium_extinction','_tex_bands','film')
        return tuple((k,repr(getattr(a,k,None))) for k in keys)+(('film_layers',repr(a.film.layer_specs())),)

    def _geometry(self):
        from OpenGL import GL as gl
        a=self.a;gl.glFinish()
        def read(b,count):
            if count==0:return np.empty(0,np.float32)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,b)
            return np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,count*4),np.float32).copy()
        geom=read(a._ssbo_tris,a._n_tris*16).reshape(-1,16)
        nodes=read(a._ssbo_nodes,a._n_bvh_nodes*12).reshape(-1,12)
        leaves=nodes[nodes[:,3]<0]
        count=int(np.max(leaves[:,8]+leaves[:,9])) if len(leaves) else 0
        if not 0<=count<=a._n_tris:raise ValueError('Unsupported BVH index cardinality')
        ids=read(a._ssbo_ids,count).view(np.int32)
        if np.any(ids<0) or np.any(ids>=a._n_tris):raise ValueError('Invalid BVH triangle indices')
        # The original stack pushes left then right, so equal-distance ties
        # choose the first candidate in right-first DFS leaf order.
        ordered=[];stack=[0] if len(nodes) else [];visited=set()
        while stack:
            ni=stack.pop()
            if not 0<=ni<len(nodes) or ni in visited:raise ValueError('Unsupported cyclic or invalid BVH')
            visited.add(ni);node=nodes[ni]
            if node[3]<0:
                start,length=int(node[8]),int(node[9])
                if start<0 or length<0 or start+length>len(ids):raise ValueError('Invalid BVH leaf')
                ordered.extend(ids[start:start+length].tolist())
            else:stack.extend([int(node[3]),int(node[7])])
        # Preserve actual BVH candidates and tie order, not upload order.
        triangles=[dict(v0=geom[i,:3].tolist(),e1=geom[i,4:7].tolist(),e2=geom[i,8:11].tolist(),owner=int(i)) for i in dict.fromkeys(ordered)]
        digest=hashlib.sha256(geom.tobytes()+nodes.tobytes()+ids.tobytes()+read(a._ssbo_shade,a._n_tris*16).tobytes()+read(a._ssbo_spheres,a._n_spheres*8).tobytes()).hexdigest()
        self.bvh_nodes=nodes.copy();self.bvh_ids=ids.copy()
        return triangles,digest

    def bind(self,phase):
        self._assert_owner()
        from OpenGL import GL as gl
        for binding,b in zip((16,17),self.buffers):gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,b)
        gl.glUniform1i(gl.glGetUniformLocation(self.program,'uStudioLoanPhase'),phase)

    def _assert_owner(self):
        if threading.get_ident()!=self.owner_thread:
            raise RuntimeError('Ray loan must be used on its owning GL thread')

    def _resources(self):
        """Bounded validation-stage freeze check, including same-handle writes.

        This readback is deliberately expensive and cannot establish savings.
        Production needs exclusive versioned resource ownership instead.
        """
        from OpenGL import GL as gl
        gl.glFinish();a=self.a;digest=hashlib.sha256();remaining=3*1024*1024*1024
        def budget(n):
            nonlocal remaining
            if n<0 or n>remaining:raise ValueError('Resource snapshot exceeds supported 3 GiB scene/history budget')
            remaining-=n
        for b in dict.fromkeys([a._ssbo_mat,a._ssbo_native_pbr,a._ssbo_regions,a._ssbo_sources_original,a._ssbo_sources_nee,*a._history_buffers]):
            if not b:continue
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,b)
            n=int(gl.glGetBufferParameteriv(gl.GL_SHADER_STORAGE_BUFFER,gl.GL_BUFFER_SIZE));budget(n)
            for offset in range(0,n,4*1024*1024):
                digest.update(bytes(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,offset,min(4*1024*1024,n-offset))))
        for target,items,fmt,dtype,channels in ((gl.GL_TEXTURE_3D,a._tex_bands,gl.GL_RED_INTEGER,gl.GL_UNSIGNED_INT,1),(gl.GL_TEXTURE_2D,a._textures,gl.GL_RGBA,gl.GL_FLOAT,4)):
            for t in items:
                gl.glBindTexture(target,t)
                dims=[int(gl.glGetTexLevelParameteriv(target,0,k)) for k in (gl.GL_TEXTURE_WIDTH,gl.GL_TEXTURE_HEIGHT)]
                if target==gl.GL_TEXTURE_3D:dims.append(int(gl.glGetTexLevelParameteriv(target,0,gl.GL_TEXTURE_DEPTH)))
                budget(int(np.prod(dims))*channels*4)
                digest.update(np.asarray(gl.glGetTexImage(target,0,fmt,dtype)).tobytes())
            gl.glBindTexture(target,0)
        return digest.hexdigest()

    def _unchanged(self):
        start=time.perf_counter()
        try:return (self.signature==self._signature() and self._geometry()[1]==self.geometry_hash and self._resources()==self.resource_hash)
        finally:self.freeze_check_seconds+=time.perf_counter()-start;self.freeze_checks+=1

    def scene_descriptor(self):
        """Authoritative packed query geometry, transferable once per identity.

        A browser may reuse an existing resident scene only after matching this
        exact identity, not merely a mesh name or displayed camera revision.
        Camera/ray state remains separately identified on every leased strip.
        """
        self._assert_owner()
        return {**self.query_identity,'revision':self.revision,'geometry_hash':self.geometry_hash,
                'triangles':self.geometry,'bvh_nodes':self.bvh_nodes.tolist(),
                'bvh_ids':self.bvh_ids.tolist(),'bvh_epsilon':1e-7,
                'query_variant':self.query_variant,
                'query_scope':'primary-triangles-server-spheres-partitions-and-transport'}

    def issue(self,seconds=10,*,include_scene=True):
        self._assert_owner()
        if self.state!='pending' or not self._unchanged():
            raise RuntimeError('No unchanged pending strip to lease')
        if self.token is not None and time.monotonic()<self.deadline:
            raise RuntimeError('Strip is already on loan')
        if not 0<seconds<=60:raise ValueError('Lease must be 0..60 seconds')
        self.token=uuid.uuid4().hex;self.deadline=time.monotonic()+seconds
        identity=f'{self.revision}:{self.a._pass}:{self.a._row_off}:{self.a._frame}'
        packet={'parent_dispatch':True,'token':self.token,'revision':self.revision,'sample_prefix':identity,
                'rays':self.rays.copy(),'geometry_hash':self.geometry_hash,'query_geometry_hash':self.query_identity['query_geometry_hash'],
                'kernel':hashlib.sha256(self.source.encode()).hexdigest()}
        if include_scene:packet.update(self.scene_descriptor())
        return packet

    def complete(self,token,revision,hits,validate):
        """validate(request rays, candidate hits) MUST check authoritative geometry.

        Merely checking finiteness/bounds does not establish nearest-hit or miss
        correctness. The cooperative worker supplies CompletedFirstHitVerifier;
        successful verification returns canonical GPU records for continuation.
        """
        from OpenGL import GL as gl
        self._assert_owner()
        if self.state!='pending' or token!=self.token or revision!=self.revision:
            raise ValueError('Stale or duplicate first-hit result')
        if time.monotonic()>=self.deadline:raise ValueError('Expired first-hit result; strip retained')
        if not self._unchanged():raise ValueError('Sensor changed while strip was on loan')
        hits=np.asarray(hits,np.float32)
        if hits.shape!=(self.count,4) or not np.isfinite(hits).all():raise ValueError('Invalid hit array')
        valid=hits[:,2]==1
        if np.any(~np.isin(hits[:,2],[0,1])) or np.any(hits[valid,0]<=self.rays[valid,3]) or np.any(hits[valid,0]>=self.rays[valid,7]) or np.any(hits[valid,1]!=np.floor(hits[valid,1])) or np.any(hits[valid,1]<0) or np.any(hits[valid,1]>=self.a._n_tris):
            raise ValueError('Invalid triangle hit record')
        verified=validate(self.rays.copy(),hits.copy())
        if isinstance(verified,VerifiedFirstHits):
            canonical=np.asarray(verified.hits,np.float32)
            if canonical.shape!=hits.shape or not np.isfinite(canonical).all() or not np.array_equal(canonical[:,2],hits[:,2]):raise ValueError('Invalid authoritative hit records')
            hits=canonical.copy()
        elif verified is not True:raise ValueError('Authoritative result validation failed')
        if time.monotonic()>=self.deadline:raise ValueError('Lease expired during validation; strip retained')
        if not self._unchanged():raise ValueError('Sensor changed during result validation')
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self.buffers[1]);gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,hits.nbytes,hits)
        self.state='committing';self.a._prog=self.program
        try:
            self.a._dispatch_once(_loan_phase=2)
            gl.glFinish()
            if gl.glGetError()!=gl.GL_NO_ERROR:raise RuntimeError('Resume GPU error; outcome uncertain, do not retry')
            self.state='completed'
        except Exception:
            self.state='uncertain';raise
        finally:self.a._prog=self.original
        self.close()

    def close(self):
        from OpenGL import GL as gl
        self._assert_owner()
        # An uncertain deposition must not silently return to the server queue.
        if self.state=='uncertain':return
        if getattr(self.a,'_intersection_loan',None) is self:self.a._intersection_loan=None
        self.a._prog=self.original
        if self.buffers:gl.glDeleteBuffers(len(self.buffers),self.buffers)
        if self.program:gl.glDeleteProgram(self.program)
        self.buffers=[];self.program=0
        if self.state!='completed':self.state='canceled'
