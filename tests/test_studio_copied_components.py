import numpy as np
import pytest
from camera_designer.studio_copied_components import CopiedComponentConsumer
from camera_designer.marble_color_profile import normalize_film_response


def fixture():
    response=normalize_film_response()
    specs=[dict(center_hz=2.998e17/c,width_oct=w,gain=1.) for c,w in zip(response['centers_nm'],response['sigmas_oct'])]
    state=dict(scene_revision='scene',camera_revision='camera',lineage=dict(rows=np.array([[1,7,0,-1],[1,8,0,-1]],np.int32)))
    calls=[];published=[]
    def native_stub(node_count,lineage,spectra,compiler):
        # CPU test double of the EXISTING native weighted-sum kernel, never
        # used by production. No GPU context is opened by these tests.
        calls.append((lineage.copy(),spectra.copy()))
        sums=np.zeros((node_count,3),np.float32)
        weights=lineage[:,6].view(np.float32)
        np.add.at(sums,lineage[:,0],spectra*weights[:,None])
        return sums
    consumer=CopiedComponentConsumer(state_provider=lambda:state,film_response=response,
        compile_program=None,publish=published.append,memory_budget_bytes=10_000_000,
        reserved_bytes=0,accumulator=native_stub)
    meta=dict(scene_revision='scene',camera_revision='camera',response_schema='native-camera-response-copies-v1',
        response_weighting='original_path_throughput_not_availability_premultiplied',
        sequence=0,row_count=1,complete_dispatch_observation=True,
        dispatch=dict(source_sha256='source',camera_uniforms={'eye':[0,0,1]},layer_specs=specs))
    return consumer,meta,state,published,calls


def copied(*,value=1.,invocation=0,bounce=0,current=0,first=0,stage=1,known=63,status=0,kind=4):
    row=np.zeros(128,np.uint32);result=np.zeros(128,np.uint32)
    row[4]=current;row[6]=3;row[7]=1;row[15]=invocation;row[3]=bounce
    row[32]=status;row[40]=7;row[41]=stage;row[115]=first;row[124]=0
    result[kind*8:kind*8+3]=np.asarray([value]*3,np.float32).view(np.uint32)
    result[96+kind]=7;result[108+kind]=known
    return row,result


def submit(consumer,meta,*pairs):
    meta=dict(meta,row_count=len(pairs))
    consumer(meta,np.stack([p[0] for p in pairs]).tobytes(),np.stack([p[1] for p in pairs]).tobytes())
    return consumer.latest


def record(product,component):
    return next(r for r in product['records'] if r['component']==component)


def test_empty_support_retains_primary_and_valid_zero_has_weight():
    consumer,meta,*_=fixture()
    unknown=record(submit(consumer,meta,copied(known=0)),1)
    assert unknown['availability']==0 and unknown['linear_rgb']==[0.,0.,0.]
    meta['sequence']=1
    valid=record(submit(consumer,meta,copied(value=0)),1)
    assert valid['available_camera_weight']==1 and valid['original_camera_weight']==2
    assert valid['availability']==.5 and valid['linear_rgb']==[0.,0.,0.]


def test_extra_bounce_rows_never_dilute_original_camera_denominator():
    one,meta,*_=fixture();single=record(submit(one,meta,copied()),1)
    two,meta,*_=fixture();double=record(submit(two,meta,copied(),copied(bounce=1,current=1)),1)
    assert double['original_camera_weight']==1 and double['available_camera_weight']==1
    assert np.allclose(double['linear_rgb'],2*np.array(single['linear_rgb']))


def test_first_glass_recipient_owns_downstream_response_once():
    consumer,meta,*_=fixture()
    product=submit(consumer,meta,copied(stage=2),copied(stage=2,bounce=1,current=1))
    assert len(product['records'])==1
    glass=product['records'][0]
    assert glass['component']==2 and glass['recipient_id']==7 and glass['availability']==1
    assert glass['rgb_is_availability_premultiplied'] is False


def test_partial_spectral_or_unresolved_geometry_never_claims_rgb():
    for kwargs in ({'known':15},{'status':6},{'status':7}):
        consumer,meta,*_=fixture()
        assert record(submit(consumer,meta,copied(**kwargs)),1)['availability']==0


def test_stale_identity_and_duplicate_dispatch_rejected():
    consumer,meta,*_=fixture();submit(consumer,meta,copied())
    with pytest.raises(ValueError,match='duplicate'):submit(consumer,meta,copied())
    with pytest.raises(ValueError,match='stale'):submit(consumer,dict(meta,camera_revision='old'),copied())


def test_camera_mutation_resets_sums_and_original_denominator():
    consumer,meta,*_=fixture();submit(consumer,meta,copied())
    meta['sequence']=1;meta['dispatch']['camera_uniforms']['eye']=[1,0,1]
    r=record(submit(consumer,meta,copied(value=0)),1)
    assert r['original_camera_weight']==1 and r['linear_rgb']==[0.,0.,0.]


def test_unknown_volume_does_not_erase_unrelated_known_reflection():
    consumer,meta,*_=fixture()
    row,response=copied()
    response[96]=7;response[108]=0 # Actual volume response unknown.
    product=submit(consumer,meta,(row,response))
    assert record(product,0)['availability']==0
    assert record(product,1)['availability']==1
    # An unknown AIR ATTENUATION genuinely invalidates its later response.
    consumer,meta,*_=fixture();response[107]=7;response[119]=0
    assert record(submit(consumer,meta,(row,response)),1)['availability']==0


def test_native_emission_copied_once_and_linear_components_remain_signed():
    consumer,meta,_,_,calls=fixture()
    r=record(submit(consumer,meta,copied(kind=1)),1)
    assert r['available_camera_weight']==1
    assert len(calls[0][1])==1
    assert consumer.latest['working_color_space']=='linear sRGB D65'
    assert 'included once' in consumer.latest['emission_ownership']


def test_budget_preflight_and_monotonic_replay_tracking():
    consumer,meta,*_=fixture()
    with pytest.raises(MemoryError):consumer.preflight_capacity(10_000_000,live_copy_bytes=1)
    assert not hasattr(consumer,'sequences')


def test_field_reset_resets_previous_native_sums_and_weights():
    consumer,meta,*_=fixture();submit(consumer,meta,copied())
    meta['sequence']=1;meta['field_generation']=1
    r=record(submit(consumer,meta,copied(value=0)),1)
    assert r['original_camera_weight']==1 and r['linear_rgb']==[0.,0.,0.]


def test_rolling_history_is_explicitly_unsupported_not_silently_cumulative():
    consumer,meta,*_=fixture();meta['dispatch']['history_samples']=4
    with pytest.raises(ValueError,match='rolling-history'):submit(consumer,meta,copied())
