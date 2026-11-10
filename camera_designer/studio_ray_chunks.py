"""Retained disjoint ray chunks; native Nodus chooses hosts and offer order.

This is estimator ownership/assembly, not a scheduler or a work queue.
"""
import time,uuid
import numpy as np

class RetainedRayBatch:
    def __init__(self,request,chunk_size,owners,*,clock=time.monotonic):
        rays=np.asarray(request['rays'],np.float32)
        if rays.ndim!=2 or rays.shape[1]!=8 or not 1<=len(rays)<=4096 or not np.isfinite(rays).all():raise ValueError('Invalid retained ray batch')
        if type(chunk_size) is not int or not 1<=chunk_size<=4096:raise ValueError('Invalid chunk size')
        self.rays=rays.copy();self.clock=clock;self.batch=uuid.uuid4().hex;self.owners=set(owners)
        self.identity={k:request[k] for k in ('revision','sample_prefix','geometry_hash','query_geometry_hash','kernel','query_kernel')}
        self.parts=[{'start':start,'count':min(chunk_size,len(rays)-start),'generation':0,'state':'unoffered'} for start in range(0,len(rays),chunk_size)]
        self.hits=np.empty((len(rays),4),np.float32);self.committed=False
    def lease(self,index,participant,generation,seconds=60):
        """Called only for a native Nodus offer; never selects a participant."""
        part=self.parts[index]
        if self.committed or part['state']=='complete':raise ValueError('Chunk already complete')
        if not isinstance(participant,str) or not participant:raise ValueError('Missing authorized participant')
        if type(generation) is not int or generation<=part['generation'] or not 0<seconds<=60:raise ValueError('Stale generation or invalid lease')
        if part['state']=='pending' and self.clock()<part['deadline']:raise ValueError('Chunk already leased')
        token=uuid.uuid4().hex;part.update(state='pending',participant=participant,generation=generation,token=token,deadline=self.clock()+seconds)
        start,count=part['start'],part['count']
        return {**self.identity,'parent_batch':self.batch,'chunk_index':index,'ray_start':start,'ray_count':count,'generation':generation,'participant':participant,'token':token,'rays':self.rays[start:start+count].copy()}
    def reject(self,index,generation):
        part=self.parts[index]
        if part['state']!='pending' or part['generation']!=generation:raise ValueError('No matching pending chunk')
        part['deadline']=self.clock()
    def accept(self,result,participant):
        index=result.get('chunk_index')
        if type(index) is not int or not 0<=index<len(self.parts):raise ValueError('Unknown chunk')
        part=self.parts[index]
        if self.committed or part['state']!='pending' or self.clock()>=part['deadline']:raise ValueError('Expired or duplicate chunk')
        if participant!=part['participant'] or result.get('participant')!=participant:raise ValueError('Wrong participant')
        expected={**self.identity,'parent_batch':self.batch,'token':part['token'],'generation':part['generation'],'ray_start':part['start'],'ray_count':part['count']}
        if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Chunk identity changed')
        hits=np.asarray(result.get('hits'),np.float32)
        if hits.shape!=(part['count'],4) or not np.isfinite(hits).all():raise ValueError('Malformed chunk results')
        valid=hits[:,2]==1;rays=self.rays[part['start']:part['start']+part['count']]
        if not np.isin(hits[:,2],[0,1]).all() or np.any(hits[valid,1]!=np.floor(hits[valid,1])) or any(int(i) not in self.owners for i in hits[valid,1]) or np.any(hits[valid,0]<=rays[valid,3]) or np.any(hits[valid,0]>=rays[valid,7]):raise ValueError('Invalid hit interval or primitive')
        self.hits[part['start']:part['start']+part['count']]=hits;part['state']='complete'
        return self.complete
    @property
    def complete(self):return all(p['state']=='complete' for p in self.parts)
    def assembled(self):
        if self.committed or not self.complete:raise ValueError('Parent batch is incomplete or already committed')
        return self.hits.copy()
    def mark_committed(self):
        if self.committed or not self.complete:raise ValueError('Parent batch cannot commit')
        self.committed=True
