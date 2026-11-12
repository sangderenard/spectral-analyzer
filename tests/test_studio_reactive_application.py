"""CPU-only queue-knownness tests. No compiler, GL import or dispatch."""
import copy
import numpy as np
import pytest
from camera_designer.studio_scene_application import build_scene_application, reactive_queue_complete
from camera_designer.studio_forward_responses import native_queue_status
from camera_designer.film_response import normalize_film_response
from tests.test_studio_scene_responses import store, meta, camera


def observed(attempted=0,capacity=4,**changes):
    return dict(native_queue_status(attempted,capacity),observed=True,incomplete_observed=False,counter_regression=False,**changes)


def setup(queue,disabled=False):
    s=store();row=camera();row[0,5:8]=[2,1,3];row[0,28]=0;row[0,88:91]=3;row[0,96:99]=3
    dispatch=dict(camera_uniforms={'eye':[0,0,2]},layer_specs=[{}, {}, {}],
        volume_controls=dict(master_alpha=0.0 if disabled else 1.0,diffuse_scatter=.35,specular_scatter=.1))
    metadata=meta(dispatch=dispatch,camera_revision='camera',reactive_queue=queue)
    if queue is None:metadata.pop('reactive_queue')
    s.consume_camera_samples(metadata,row)
    slot=int(np.flatnonzero(s.used)[0]);assert s._contexts[int(s.index[slot,2])][0].get('reactive_queue')==queue
    return s,metadata,row.copy()


def evaluate(s,metadata,should_sum):
    calls=[]
    def native_oracle(nodes,lineage,spectra,compiler):
        assert should_sum,'incomplete native work cannot authorize a copied zero'
        calls.append(len(spectra));result=np.zeros((nodes,3),np.float32)
        for key,value in zip(lineage[:,0],spectra):result[key]+=value
        return result
    result=build_scene_application(s,film_response=normalize_film_response(),compile_program=None,
        memory_budget_bytes=1024*1024,accumulator=native_oracle,current_metadata=metadata)
    return (*result,calls)


@pytest.mark.parametrize('queue',[None,{},observed(1),observed(5),
    dict(observed(),incomplete_observed=True),dict(observed(),counter_regression=True),
    dict(observed(),complete=True,pending_unprocessed=1),dict(observed(),stored='0')])
def test_missing_pending_dropped_sticky_or_malformed_queue_cannot_authorize_black(queue):
    s,metadata,original=setup(queue)
    m,rows,calls=evaluate(s,metadata,False)
    assert not calls and np.all(rows[:,7]==0) and np.all(rows[:,8]==1)
    assert m['queue_unavailable_component_samples']==2 and m['ready'] is False
    assert np.array_equal(s.native[s.used][0,:128],original[0])


def test_actual_observed_empty_complete_queue_preserves_known_zero_weight():
    s,metadata,_=setup(observed())
    _,rows,calls=evaluate(s,metadata,True)
    assert calls==[2] and np.all(rows[:,7:9]==1)
    assert np.all(rows[:,9:12].view(np.float32)==0)


def test_disabled_native_volume_independent_of_unprocessed_downstream_work():
    s,metadata,_=setup(observed(1),disabled=True)
    _,rows,calls=evaluate(s,metadata,True)
    assert calls==[1]
    volume=rows[rows[:,6]==0][0];glass=rows[rows[:,6]==2][0]
    assert tuple(volume[7:9])==(1,1) and tuple(glass[7:9])==(0,1)


def test_changed_queue_observation_cannot_select_stale_application_context():
    s,metadata,_=setup(observed())
    current=copy.deepcopy(metadata);current['reactive_queue']=observed(1)
    m,_,_=evaluate(s,current,True)
    assert m['current_query_context_ids']==[]


def test_completed_actual_dispatch_counts_are_valid_but_not_fabricated_completion():
    complete=dict(native_queue_status(3,4,dispatched=3),observed=True)
    assert reactive_queue_complete(complete)
    assert not reactive_queue_complete(dict(complete,dropped_enqueue=1))


def test_copied_queue_evidence_is_not_rewritten_by_later_caller_mutation():
    queue=observed();s,metadata,_=setup(queue)
    queue['pending_unprocessed']=1;queue['complete']=False
    slot=int(np.flatnonzero(s.used)[0]);saved=s._contexts[int(s.index[slot,2])][0]['reactive_queue']
    assert saved['complete'] is True and saved['pending_unprocessed']==0
    m,_,_=evaluate(s,metadata,True)
    assert m['current_query_context_ids']==[]


def test_host_dispatch_records_actual_volume_controls_without_shader_changes():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8')
    assert 'volume_controls=dict(master_alpha=self._va,diffuse_scatter=self._air_ds,specular_scatter=self._air_ss)' in source
