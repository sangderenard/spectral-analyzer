"""Windows status readers can briefly block atomic replacement."""

from camera_designer import marble_preview_worker as worker


def test_atomic_status_retries_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    original = worker.os.replace
    calls = 0

    def replace(source, destination):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(13, "status reader holds destination")
        return original(source, destination)

    monkeypatch.setattr(worker.os, "replace", replace)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    worker._atomic_json(target, {"state": "exposing"})
    assert calls == 3
    assert target.read_text(encoding="utf-8") == '{"state": "exposing"}'


def test_timed_shop_cutoff_precedes_routed_flash_and_sensor_admission():
    frames=[{'t0':0.0,'t1':2.0,'pbr':[0.0]*32,
             'emitter_profiles':{'version':1,'records':[
                 {'material':0,'gain':15.0},{'material':1,'gain':0.0}]}}]
    fixtures={'shop_material_ids':[0,1],
              'trigger_emitters':[{'material':1,'receiver':'shop:1','gain':40.0}]}
    intervals,barrier=worker._timed_exposure_program(frames,fixtures,2.0)
    pre=intervals[0];flash=next(item for item in intervals if item['flash_active'])
    rest=intervals[-1]
    assert pre['t0']==0 and pre['t1']==flash['t0'] and not pre['sensor_integrating']
    assert [record['gain'] for record in pre['emitter_profiles']['records']]==[0.0,0.0]
    assert [record['gain'] for record in flash['emitter_profiles']['records']]==[0.0,40.0]
    assert [record['gain'] for record in rest['emitter_profiles']['records']]==[15.0,0.0]
    assert barrier.timeline.flash_slices==(1,) and barrier.timeline.sensor_slices==(1,2)
    from camera_software.trigger_bus import make_camera_trigger_graph
    graph=make_camera_trigger_graph(['shop:1'],bus_keys=['flash-bus'])
    graph.set_weight('Flash','flash-bus',1.0)
    graph.set_weight('flash-bus','shop:1',1.0)
    routed,_=worker._timed_exposure_program(frames,fixtures,2.0,graph.to_dict(),
        {'shop_cutoff_lead_s':.002,'flash_duration_s':.006})
    assert next(item for item in routed if item['flash_active'])['emitter_profiles']['records'][1]['gain']==40.0
    assert routed[0]['t1']==.002 and routed[1]['t1']==.008
    split=[{**frames[0],'t1':.005},{**frames[0],'t0':.005}]
    _,split_barrier=worker._timed_exposure_program(split,fixtures,2.0)
    assert len(split_barrier.timeline.flash_slices)==2
    assert abs(sum(split_barrier.timeline.slices[i].exposure_slice.flash_weight
                   for i in split_barrier.timeline.flash_slices)-1.0)<1e-12
