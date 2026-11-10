import numpy as np
import pytest
from camera_designer.studio_ray_chunks import RetainedRayBatch

def batch():
    clock=[0.];rays=np.tile([0,0,0,.001,0,0,1,100.],(8,1))
    request={k:k for k in ('revision','sample_prefix','geometry_hash','query_geometry_hash','kernel','query_kernel')};request['rays']=rays
    return RetainedRayBatch(request,2,{3},clock=lambda:clock[0]),clock

def result(work):return {**{k:v for k,v in work.items() if k!='rays'},'hits':[[1,3,1,3]]*work['ray_count']}

def test_disjoint_concurrent_chunks_out_of_order_exactly_once():
    b,_=batch();offers=[b.lease(i,'p'+str(i%2),1) for i in range(4)]
    assert [o['ray_start'] for o in offers]==[0,2,4,6]
    with pytest.raises(ValueError):b.assembled()
    for i in (3,1,0,2):b.accept(result(offers[i]),offers[i]['participant'])
    assert b.complete and b.assembled().shape==(8,4)
    with pytest.raises(ValueError):b.accept(result(offers[0]),'p0')
    b.mark_committed()
    with pytest.raises(ValueError):b.mark_committed()

def test_expiry_reoffers_only_unfinished_chunk_without_losing_others():
    b,c=batch();a=b.lease(0,'a',1,1);done=b.lease(1,'b',1,1);b.accept(result(done),'b');c[0]=2
    retry=b.lease(0,'b',2)
    np.testing.assert_array_equal(a['rays'],retry['rays'])
    with pytest.raises(ValueError):b.accept(result(a),'a')
    b.accept(result(retry),'b');assert b.parts[1]['state']=='complete'

def test_participant_identity_and_missing_ray_reject_without_completion():
    b,_=batch();a=b.lease(0,'a',1);r=result(a)
    with pytest.raises(ValueError):b.accept(r,'b')
    r['hits'].pop()
    with pytest.raises(ValueError):b.accept(r,'a')
    assert b.parts[0]['state']=='pending'
