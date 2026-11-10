"""GL-owner-only verification of completed first-hit client batches.

Runs no query on construction or offer. Each call verifies the full returned
strip with the original composed triangle/BVH law (1:1), excluding analytic
sphere competition because the loan consumer retains it on the server.
"""
import time
import numpy as np
from .studio_first_hit_loan import VerifiedFirstHits

def triangle_rounding_bound(rays, records, owners, distances):
    """Conservative float32 determinant forward-error window, in ray t units.

    gamma(32) covers separate or contracted cross/dot/division operations.
    The origin subtraction error is propagated explicitly. Ill-conditioned
    denominator intervals fail closed instead of granting an unlimited window.
    This is validation arithmetic, not a substitute intersection engine.
    """
    result=np.zeros(len(rays),np.float64);eps=np.finfo(np.float32).eps
    gamma=32*eps/(1-32*eps)
    selected=np.flatnonzero(owners>=0)
    if not len(selected):return result
    if any(int(owners[i]) not in records for i in selected):
        result[selected]=np.nan;return result
    tri=[records[int(owners[i])] for i in selected]
    v=np.asarray([t['v0'] for t in tri],np.float64)
    a=np.asarray([t['e1'] for t in tri],np.float64)
    b=np.asarray([t['e2'] for t in tri],np.float64)
    ro=np.asarray(rays[selected,:3],np.float64);rd=np.asarray(rays[selected,4:7],np.float64);s=ro-v
    def cross_abs(x,y):return np.stack((x[:,1]*y[:,2]+x[:,2]*y[:,1],x[:,2]*y[:,0]+x[:,0]*y[:,2],x[:,0]*y[:,1]+x[:,1]*y[:,0]),axis=1)
    den=np.sum(a*np.cross(rd,b),axis=1)
    den_error=gamma*np.sum(np.abs(a)*cross_abs(np.abs(rd),np.abs(b)),axis=1)
    sub_error=eps*(np.abs(ro)+np.abs(v))
    num_error=gamma*np.sum(np.abs(b)*cross_abs(np.abs(s),np.abs(a)),axis=1)+(1+gamma)*np.sum(np.abs(b)*cross_abs(sub_error,np.abs(a)),axis=1)
    denominator=np.abs(den)-den_error
    bound=np.full(len(selected),np.nan)
    safe=denominator>0
    reference=np.zeros(len(selected));reference[safe]=np.sum(b[safe]*np.cross(s[safe],a[safe]),axis=1)/den[safe]
    bound[safe]=(num_error[safe]+np.abs(reference[safe])*den_error[safe])/denominator[safe]
    bound+=4*np.abs(np.spacing(np.asarray(distances[selected],np.float32))).astype(np.float64)
    result[selected]=bound
    return result

class CompletedFirstHitVerifier:
    def __init__(self, loan, source, compile_program):
        from OpenGL import GL as g
        loan._assert_owner();self.loan=loan;self.buffer=0;self.program=0
        self.checked_batches=0;self.checked_rays=0;self.seconds=0.0
        self.records={int(t["owner"]):t for t in loan.geometry}
        if source.count('void main() {')!=1:raise RuntimeError('Sensor main seam changed')
        shader=source.replace('void main() {','void unused_sensor_main() {',1)+'''
layout(std430,binding=16) readonly buffer VerifyInput {vec4 verify_rays[];};
layout(std430,binding=17) buffer VerifyOutput {vec4 verify_hits[];};
uniform int uVerifyCount;uniform int uVerifyWidth;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;
if(x>=uint(uVerifyWidth))return;uint i=y*uint(uVerifyWidth)+x;
if(i>=uint(uVerifyCount))return;float t;
int h=nearest_filtered(verify_rays[2*i].xyz,verify_rays[2*i+1].xyz,t,false);
vec4 supplied=verify_hits[i+uint(uVerifyCount)];float candidate_t=0.0;
int candidate=int(supplied.y);
bool candidate_valid=supplied.z>0.5 && candidate>=0 && candidate<uTriCount && hit_tri(verify_rays[2*i].xyz,verify_rays[2*i+1].xyz,geom_tris[candidate],candidate_t);
verify_hits[i]=vec4(h>=0?t:0.0,float(h),h>=0?1.0:0.0,float(h));
verify_hits[i+uint(uVerifyCount)]=vec4(candidate_valid?candidate_t:0.0,supplied.y,candidate_valid?1.0:0.0,0.0);}
'''
        try:
            self.program=compile_program((shader,g.GL_COMPUTE_SHADER))
            if not g.glGetProgramiv(self.program,g.GL_LINK_STATUS):raise RuntimeError('Completed-result verifier failed to link')
            self.buffer=int(g.glGenBuffers(1));g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,self.buffer)
            g.glBufferData(g.GL_SHADER_STORAGE_BUFFER,loan.count*32,None,g.GL_DYNAMIC_DRAW)
        except Exception:self.close();raise

    def __call__(self,rays,hits):
        from OpenGL import GL as g
        loan=self.loan;loan._assert_owner();a=loan.a
        if loan.state!='pending' or not self.program:raise RuntimeError('Verifier requires retained pending work')
        if np.asarray(hits).shape!=(loan.count,4):return False
        start=time.perf_counter()
        supplied=np.ascontiguousarray(hits,np.float32)
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,self.buffer)
        g.glBufferSubData(g.GL_SHADER_STORAGE_BUFFER,loan.count*16,supplied.nbytes,supplied)
        for slot,b in [(0,a._ssbo_tris),(1,a._ssbo_nodes),(2,a._ssbo_ids),(9,a._ssbo_shade),(16,loan.buffers[0]),(17,self.buffer)]:g.glBindBufferBase(g.GL_SHADER_STORAGE_BUFFER,slot,b)
        g.glUseProgram(self.program)
        for key,value in [('uNodeCount',a._n_bvh_nodes),('uTriCount',a._n_tris),('uStudioSphereCount',0),('uVerifyCount',loan.count),('uVerifyWidth',a._w)]:g.glUniform1i(g.glGetUniformLocation(self.program,key),value)
        rows=(loan.count+a._w-1)//a._w
        g.glDispatchCompute((a._w+15)//16,(rows+15)//16,1);g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT);g.glFinish()
        if g.glGetError()!=g.GL_NO_ERROR:raise RuntimeError('Completed-result validation GPU error; sample retained')
        g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,self.buffer)
        all_hits=np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,0,loan.count*32),np.float32).copy().reshape(-1,4)
        expected=all_hits[:loan.count];candidate=all_hits[loan.count:]
        self.checked_batches+=1;self.checked_rays+=loan.count
        tolerance=triangle_rounding_bound(rays,self.records,expected[:,1],expected[:,0])
        candidate_tolerance=triangle_rounding_bound(rays,self.records,candidate[:,1],candidate[:,0])
        same_id=hits[:,1]==expected[:,1]
        # A different nearest primitive can be legitimate at a float32 shared
        # edge. Prove the supplied primitive intersects using the original GPU
        # law, and that both distances are within the same rounding window.
        # Never accept a merely claimed near-tie, and always resume canonical GPU.
        near_tie=np.isfinite(candidate_tolerance)&(hits[:,2]==1)&(expected[:,2]==1)&(candidate[:,2]==1)&(np.abs(candidate[:,0]-expected[:,0])<=tolerance)&(np.abs(hits[:,0]-candidate[:,0])<=candidate_tolerance)
        valid=bool(np.isfinite(tolerance).all() and np.array_equal(hits[:,2],expected[:,2]) and np.all(same_id|near_tie) and np.all(np.abs(hits[:,0]-expected[:,0])<=tolerance))
        self.verified_near_ties=int((~same_id & near_tie).sum())
        self.last_mismatch={'identity_mismatches':int(np.any(hits[:,1:3]!=expected[:,1:3],axis=1).sum()),'distance_mismatches':int((np.abs(hits[:,0]-expected[:,0])>tolerance).sum()),'max_distance_delta':float(np.max(np.abs(hits[:,0]-expected[:,0])))}
        self.last_expected=expected
        self.seconds+=time.perf_counter()-start
        return VerifiedFirstHits(expected.copy()) if valid else False

    def close(self):
        from OpenGL import GL as g
        self.loan._assert_owner()
        if self.buffer:g.glDeleteBuffers(1,[self.buffer]);self.buffer=0
        if self.program:g.glDeleteProgram(self.program);self.program=0

class AuthorizedParticipantAcceptance:
    """Structural acceptance only. Does not retrace or claim geometric proof."""
    def __init__(self,loan):
        self.loan=loan;self.owners={int(t['owner']) for t in loan.geometry}
        self.checked_batches=0;self.checked_rays=0;self.seconds=0.0
        self.structural_rays=0;self.structural_seconds=0.0
    def __call__(self,rays,hits):
        self.loan._assert_owner();start=time.perf_counter()
        try:
            valid=hits[:,2]==1
            accepted=all(int(x) in self.owners for x in hits[valid,1])
            if accepted:self.structural_rays+=len(rays)
            return bool(accepted)
        finally:self.structural_seconds+=time.perf_counter()-start
    def close(self):pass
