import copy
import numpy as np
import pytest
from camera_designer.studio_scene_responses import SceneResponseStore, validate_scene_response_snapshot
from camera_designer.studio_response_lineage import NativeResponseLineage

def store(budget=1024*1024):
    lineage=NativeResponseLineage(0);lineage.append(1,'subject',0,np.array([0]))
    return SceneResponseStore(scene_revision='physical',memory_budget_bytes=budget,lineage_provider=lambda:lineage.snapshot(),bounds=[[0,0,0],[1,1,1]],dimensions=[32,32,32])

def meta(sequence=0,**kw):
    from camera_designer.film_response import normalize_film_response
    return dict(dict(reactive_queue=dict(observed=True,complete=True,incomplete_observed=False,counter_regression=False,attempted=0,capacity=4,stored=0,dispatched=0,dropped_enqueue=0,pending_unprocessed=0),film_response=normalize_film_response(),scene_revision='physical',stream_id='native',sequence=sequence,source_sha256='source',transport_scene_sha256='transport'),**kw)

def camera():
    a=np.zeros((1,128),dtype='<u4');a[0,2:5]=[0,0,0];a[0,8:11]=np.array([.5,.5,.5],np.float32).view(np.uint32);a[0,11]=1;a[0,12:15]=np.array([0,0,1],np.float32).view(np.uint32);a[0,115]=1;a[0,116:119]=a[0,12:15];a[0,28]=0x7fc12345
    return a

def test_bit_exact_progressive_camera_population_contexts():
    s=store();a=camera();s.consume_camera_samples(meta(camera_revision='a',field_generation=1),a)
    s.consume_camera_samples(meta(1,camera_revision='b',field_generation=2),a)
    m,p=s.snapshot();assert m['retention']['retained_entries']==2
    native=m['sections'][1];assert np.frombuffer(p,dtype='<u4',offset=native['offset'])[28]==0x7fc12345
    assert not m['application_ready'];assert validate_scene_response_snapshot(m,p)
    with pytest.raises(ValueError):s.consume_camera_samples(meta(1),a)
    s.invalidate_scene('changed');assert s.retained_entries==0

def test_empty_and_malformed_batches_do_not_mutate_good_cache():
    s=store();s.consume_camera_samples(meta(),camera());old=s.snapshot()
    s.consume_camera_samples(meta(1),np.zeros((0,128),dtype='<u4'))
    bad=camera();bad[0,8]=0x7fc12345
    with pytest.raises(ValueError):s.consume_camera_samples(meta(2),bad)
    assert s.snapshot()==old

def test_scalar_exact_slabs_loss_only_and_snapshot_replacement():
    s=store();a=np.zeros((2,3,4),dtype='<u4');a[1,2]=[0,0,0,8]
    m=dict(scene_revision='physical',scalar_clear_epoch=1,snapshot_id=1,total_cells=2,cell_offset=0,final_slab=True)
    s.consume_scalar_cells(m,a);manifest,p=s.snapshot();section=manifest['sections'][3]
    assert section['stride']==24 and section['count']==1
    assert list(np.frombuffer(p,dtype='<u4',offset=section['offset']))==[1,2,0,0,0,8]
    assert manifest['scalar_snapshot']['metadata']['omitted_unvisited']==5
    s.consume_scalar_cells(dict(m,snapshot_id=2),a*0);assert s.snapshot()[0]['sections'][3]['count']==0

def test_strict_section_order_and_application_claim():
    s=store();s.consume_camera_samples(meta(),camera());m,p=s.snapshot()
    bad=copy.deepcopy(m);bad['sections'][0],bad['sections'][1]=bad['sections'][1],bad['sections'][0]
    with pytest.raises(ValueError):validate_scene_response_snapshot(bad,p)
    bad=copy.deepcopy(m);bad['application_ready']=True
    with pytest.raises(ValueError):validate_scene_response_snapshot(bad,p)


def test_native_film_component_grouping_uses_original_sample_denominator():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();a=camera();a[0,5:8]=[2,1,3];a[0,28]=0
    a[0,88:91]=3;a[0,96:99]=3
    # Exact evaluated zero is a present sample with real weight.
    dispatch={'camera_uniforms':{'eye':[0,0,2],'aperture_radius':0},'layer_specs':[{}, {}, {}]}
    s.consume_camera_samples(meta(dispatch=dispatch),a)
    b=a.copy();b[0,0]=1;b[0,96:99]=0
    s.consume_camera_samples(meta(1,dispatch=dispatch),b)
    calls=[]
    def native_oracle(nodes,lineage,spectra,compiler):
        calls.append(len(spectra));out=np.zeros((nodes,3),np.float32)
        for node,value in zip(lineage[:,0],spectra):out[node]+=value
        return out
    m,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=native_oracle,current_metadata=meta(dispatch=dispatch))
    assert calls==[3] # two volume samples, one known downstream; no extra bounce rows
    glass=rows[rows[:,6]==2][0];volume=rows[rows[:,6]==0][0]
    assert tuple(glass[7:9])==(1,2) and tuple(volume[7:9])==(2,2)
    assert np.all(glass[9:12].view(np.float32)==0)
    assert len(m['query_contexts'])==1 and not m['ready']
    assert m['emission_ownership'].startswith('included once')
    assert validate_scene_response_snapshot(*s.snapshot(application=(m,rows)))
    assert not np.any(rows[:,6]==1) # downstream opaque hit never receives this full glass path again


def test_unknown_native_components_have_identity_alpha_without_summing():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();a=camera();a[0,5:8]=[1,1,3]
    s.consume_camera_samples(meta(dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}),a)
    def forbidden(*args):raise AssertionError('no available native input')
    _,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=forbidden,current_metadata=meta(dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}))
    assert np.all(rows[:,7]==0) and np.all(rows[:,8]==1)


def test_eviction_cannot_promote_unknown_to_known_black():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}
    missing=camera();missing[0,5:8]=[2,1,3];missing[0,28]=0
    s.consume_camera_samples(meta(dispatch=dispatch),missing)
    first=int(np.flatnonzero(s.used)[0]);s._drop(first)
    zero=missing.copy();zero[0,88:91]=3;zero[0,96:99]=3
    s.consume_camera_samples(meta(1,dispatch=dispatch),zero)
    def forbidden(*args):raise AssertionError('evicted support cannot contribute application energy')
    m,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=forbidden,current_metadata=meta(dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}))
    assert np.all(rows[:,7]==0) and np.all(rows[:,8]==1)
    assert m['eviction_suppressed_support_components']==2
    assert s.evicted_camera_entries==1
    # Tombstones stay bounded and are not reset by source generation/camera refresh.
    assert s._evicted_support.nbytes==max(32,s.capacity*16)
    s.invalidate_scene('new physical scene');assert not np.any(s._evicted_support)


def test_exact_transport_context_prevents_housing_alias_and_selects_current_query():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();a=camera();a[0,5:8]=[1,1,3];a[0,28]=0
    dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}
    s.consume_camera_samples(meta(dispatch=dispatch,transport_scene_sha256='housing-a',camera_revision='cam-a'),a)
    s.consume_camera_samples(meta(1,dispatch=dispatch,transport_scene_sha256='housing-b',camera_revision='cam-b'),a)
    current=meta(1,dispatch=dispatch,transport_scene_sha256='housing-b',camera_revision='cam-b',live_revision=8)
    m,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=lambda *args:None,current_metadata=current)
    assert len(m['query_contexts'])==2 and len(m['current_query_context_ids'])==1
    q=m['query_contexts'][m['current_query_context_ids'][0]]
    assert q['transport']['transport_scene_sha256']=='housing-b'
    assert m['current_live_revision']==8


def test_aperture_origins_remain_provenance_under_same_native_lens_sampler():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();dispatch={'camera_uniforms':{'eye':[0,0,2],'right':[1,0,0],'up':[0,1,0],'forward':[0,0,-1],
        'aperture_radius':0.01,'aperture_blades':6},'layer_specs':[{}, {}, {}]}
    a=camera();a[0,5:8]=[2,1,3];a[0,28]=0
    a[0,16:19]=np.array([0.005,0,2],np.float32).view(np.uint32)
    s.consume_camera_samples(meta(dispatch=dispatch),a)
    b=a.copy();b[0,16:19]=np.array([-0.005,0,2],np.float32).view(np.uint32)
    s.consume_camera_samples(meta(1,dispatch=dispatch),b)
    m,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,current_metadata=meta(dispatch=dispatch))
    assert len(m['query_contexts'])==1 and m['query_contexts'][0]['origin_m']==[0,0,2]
    assert np.all(rows[:,8]==2)
    native=s.native[s.used];assert native[0,16]!=native[1,16]


def test_historical_film_is_not_reinterpreted_by_current_profile():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}
    a=camera();a[0,5:8]=[1,1,3];a[0,28]=0
    s.consume_camera_samples(meta(dispatch=dispatch,film_response={'different':'actual historical film'}),a)
    m,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,current_metadata=meta(dispatch=dispatch))
    assert s.retained_entries==1 and len(rows)==0 and m['current_query_context_ids']==[]


def test_actual_fifo_capacity_eviction_cannot_increase_black_coverage():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}
    rows=np.repeat(camera(),s.surface_capacity+1,axis=0)
    rows[:,0]=np.arange(len(rows),dtype=np.uint32);rows[:,5:8]=[2,1,3];rows[:,28]=0
    rows[:,88:91]=3;rows[:,96:99]=3;rows[0,88:91]=0;rows[0,96:99]=0
    s.consume_camera_samples(meta(dispatch=dispatch),rows)
    assert s.evicted_camera_entries==1 and s.retained_entries==s.surface_capacity
    def forbidden(*args):raise AssertionError('discarded unavailable sample cannot improve coverage')
    m,result=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=forbidden,current_metadata=meta(dispatch=dispatch))
    assert np.all(result[:,7]==0) and np.all(result[:,8]==s.surface_capacity)


def test_unmatched_virtual_and_missing_endpoints_retain_exact_native_rows():
    s=store();rows=np.repeat(camera(),3,axis=0);rows[:,0]=np.arange(3,dtype=np.uint32)
    rows[0,2:5]=[0,2,0]  # Actual virtual partition, no mesh recipient.
    rows[1,2:5]=[0xffffffff,0xffffffff,0xffffffff];rows[1,11]=0
    rows[1,8:11]=0;rows[1,115]=0  # Native no-hit sentinel is not a zero-position hit.
    rows[2,2:5]=[25,0,0]  # Native-only/unmapped surface identity.
    rows[:,15]=7;rows[:,36]=0x80000001;rows[:,72]=0x3f123456
    s.consume_camera_samples(meta(),rows)
    m,p=s.snapshot();section=m['sections'][1]
    copied=np.frombuffer(p,'<u4',count=3*128,offset=section['offset']).reshape(-1,128)
    np.testing.assert_array_equal(copied,rows)
    index=np.frombuffer(p,'<u4',count=3*20).reshape(-1,20)
    assert np.all(index[:,4]==0) and np.all(index[:,5]==0xffffffff)
    assert index[0,6]==2 and index[0,11]==1
    assert index[1,11]==0 and np.all(index[1,16:19]==0xffffffff)
    assert m['retention']['partitions']['native_only_retained_entries']==3
    assert m['retention']['omitted_out_of_scope']==0


def test_native_only_flood_cannot_evict_surface_rows_or_improve_availability():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}
    surface=camera();surface[0,5:8]=[2,1,3];surface[0,28]=0
    s.consume_camera_samples(meta(dispatch=dispatch),surface)
    native=np.repeat(camera(),s.capacity*3,axis=0);native[:,0]=np.arange(len(native),dtype=np.uint32)
    native[:,2:5]=[0,2,0];native[:,5:8]=[2,1,3];native[:,28]=0
    native[:,88:91]=3;native[:,96:99]=3
    s.consume_camera_samples(meta(1,dispatch=dispatch),native)
    assert s.evicted_surface_entries==0 and s.evicted_camera_entries==0
    assert s.evicted_native_only_entries==len(native)-s.native_only_capacity
    np.testing.assert_array_equal(s.native[np.flatnonzero(s.used & (s.index[:,4]==1))[0],:128],surface[0])
    def forbidden(*args):raise AssertionError('native-only rows cannot provide surface energy')
    _,application=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=forbidden,current_metadata=meta(dispatch=dispatch))
    assert len(application)==2 and np.all(application[:,7]==0) and np.all(application[:,8]==1)
    m,_=s.snapshot();assert m['retention']['partitions']['surface_retained_entries']==1


def test_raw_only_context_pressure_preserves_surface_evidence_and_reports_drop():
    s=store();s.consume_camera_samples(meta(),camera())
    before=s.native[s.used].copy();context_bytes=s._context_bytes
    native=camera();native[0,2:5]=[0,2,0]
    # One native context fits alone, but cannot fit beside protected context.
    limit=s.memory_budget_bytes//10
    padding='x'*((limit-context_bytes//2)//4)
    s.consume_camera_samples(meta(1,dispatch={'native_context':padding}),native)
    assert s.native_only_context_discarded_entries==1
    assert s.evicted_surface_entries==0 and s.retained_entries==1
    np.testing.assert_array_equal(s.native[s.used],before)
    assert s._context_bytes==context_bytes
    assert s.snapshot()[0]['retention']['partitions']['native_only_context_discarded_entries']==1
    with pytest.raises(ValueError):s.consume_camera_samples(meta(1),native)


def test_native_endpoint_scope_and_explicit_surface_category_scope_stay_enforced():
    s=store();outside=camera();outside[0,2:5]=[0,2,0]
    outside[0,8:11]=np.array([2,.5,.5],np.float32).view(np.uint32)
    s.consume_camera_samples(meta(),outside)
    assert s.retained_entries==0 and s.omitted_out_of_scope==1
    s.recipient_namespaces=frozenset({2})
    s.consume_camera_samples(meta(1),camera())
    assert s.retained_entries==0 and s.omitted_out_of_scope==2


def test_native_partition_receipt_is_checked_against_exact_raw_identity():
    s=store();native=camera();native[0,2:5]=[0,2,0]
    s.consume_camera_samples(meta(),native);m,p=s.snapshot()
    bad=copy.deepcopy(m);r=bad['retention']['partitions']
    r['native_only_retained_entries']=0;r['surface_retained_entries']=1
    with pytest.raises(ValueError,match='identity mismatch'):validate_scene_response_snapshot(bad,p)


def test_forward_interior_partition_and_escape_payloads_are_not_reinterpreted():
    s=store();rows=np.zeros((2,128),dtype='<u4')
    rows[:,0:2]=[5,1];rows[0,5]=0;rows[1,5]=0xffffffff
    rows[:,7:9]=[4,6];rows[:,30]=1;rows[:,32]=0x80000001;rows[:,64]=0x40000002
    rows[:,18:24]=np.array([.1,.2,.3,.4,.5,.6],np.float32).view(np.uint32)
    rows[:,24:30]=np.array([0,1,0,1,0,0],np.float32).view(np.uint32)
    rows[:,112:114]=[2,0];rows[1,112]=0
    rows[:,104:112]=np.arange(8,dtype=np.float32).view(np.uint32)
    rows[:,114:117]=rows[:,24:27];rows[:,117:120]=rows[:,18:21];rows[:,124]=4
    s.consume_forward(meta(primitive_identity_offsets=[112,113]),rows)
    m,p=s.snapshot();section=m['sections'][1]
    actual=np.frombuffer(p,'<u4',count=256,offset=section['offset']).reshape(-1,128)
    np.testing.assert_array_equal(actual,rows)
    index=np.frombuffer(p,'<u4',count=40).reshape(-1,20)
    assert np.all(index[:,4]==0) and tuple(index[:,11])==(1,0)
    assert np.all(index[1,16:19]==0xffffffff)  # Never turn escape origin into a hit.
    assert s.retained_entries==2 and s.omitted_out_of_scope==0


def test_changed_native_prefix_direction_never_claims_straight_raster_support():
    from camera_designer.studio_scene_application import build_scene_application
    from camera_designer.film_response import normalize_film_response
    s=store();dispatch={'camera_uniforms':{'eye':[0,0,2]},'layer_specs':[{}, {}, {}]}
    a=camera();a[0,5:8]=[2,1,3];a[0,28]=0;a[0,88:91]=3;a[0,96:99]=3
    a[0,116:119]=np.array([1,0,0],np.float32).view(np.uint32)
    s.consume_camera_samples(meta(dispatch=dispatch),a)
    def forbidden(*args):raise AssertionError('direction-changing prefix cannot be applied as a straight raster query')
    m,rows=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=forbidden,current_metadata=meta(dispatch=dispatch))
    assert s.retained_entries==1 and np.all(rows[:,7]==0) and np.all(rows[:,8]==1)
    assert np.array_equal(s.native[s.used][0,116:119],a[0,116:119])


def test_scalar_epoch_discard_is_explicit_and_never_claims_progressive_sum():
    s=store();a=np.zeros((1,3,4),dtype='<u4');a[0,0]=[1,1,17,0]
    metadata=dict(scene_revision='physical',scalar_clear_epoch=1,snapshot_id=1,total_cells=1,cell_offset=0,
        final_slab=True,source_population_sha256='population-a',transport_scene_sha256='transport')
    s.consume_scalar_cells(metadata,a)
    s.consume_scalar_cells(dict(metadata,scalar_clear_epoch=2,snapshot_id=2,source_population_sha256='population-b'),a)
    m,_=s.snapshot();r=m['scalar_snapshot']['retention']
    assert r['discarded_epochs']==1 and r['discarded_entries']==1
    assert r['recent_discards'][0]['source_population_sha256']=='population-a'
    assert r['discarded_identity_digest'] and 'no cross-epoch accumulation' in r['mode']
