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


def test_delayed_amplitude_routes_into_real_exposure_intervals():
    from camera_software.trigger_bus import make_camera_trigger_graph
    frames=[{'t0':0.0,'t1':2.0,'pbr':[0.0]*48,
             'emitter_profiles':{'version':1,'records':[
                 {'material':0,'gain':15.0},{'material':1,'gain':0.0},{'material':2,'gain':6.0}]}}]
    fixtures={'shop_material_ids':[0,1], 'trigger_emitters':[
        {'material':1,'receiver':'shop:flash','gain':40.0},
        {'material':2,'receiver':'studio:key','frame_gain':True}]}
    graph=make_camera_trigger_graph(['shop:flash','studio:key'],bus_keys=['rack:center'])
    graph.set_weight('Flash','rack:center',.5)
    graph.set_delay_s('Flash','rack:center',.005)
    graph.set_weight('rack:center','shop:flash',1)
    graph.set_weight('rack:center','studio:key',.5)
    intervals,barrier=worker._timed_exposure_program(frames,fixtures,2.0,graph.to_dict())
    delayed=next(item for item in intervals if abs(item['t0']-.006)<1e-8)
    assert abs(delayed['t1']-.011)<1e-8
    assert [r['gain'] for r in delayed['emitter_profiles']['records']]==[0.0,20.0,1.5]
    after=next(item for item in intervals if abs(item['t0']-.011)<1e-8)
    assert [r['gain'] for r in after['emitter_profiles']['records']]==[15.0,20.0,1.5]
    finished=next(item for item in intervals if abs(item['t0']-.016)<1e-8)
    assert [r['gain'] for r in finished['emitter_profiles']['records']]==[15.0,0.0,0.0]
    assert abs(sum(barrier.timeline.slices[i].exposure_slice.flash_weight
                   for i in barrier.timeline.flash_slices)-1.0)<1e-12


def test_repeated_generator_modulates_light_on_exposure_timeline():
    import pytest
    from camera_software.trigger_bus import make_camera_trigger_graph
    from camera_software.trigger_generators import GENERATOR_KEYS
    frames=[{'t0':0.0,'t1':1.0,'pbr':[0.0]*32,
             'emitter_profiles':{'version':1,'records':[{'material':0,'gain':8.0}]}}]
    fixtures={'shop_material_ids':[], 'trigger_emitters':[
        {'material':0,'receiver':'studio:0','frame_gain':True}]}
    graph=make_camera_trigger_graph(['studio:0'],generator_keys=GENERATOR_KEYS)
    graph.set_weight('mod:1','studio:0',.5)
    spec={'key':'mod:1','enabled':True,'kind':'ramp_down','frequency_hz':0,
          'phase_deg':0,'tension':1,'attack_ms':0,'decay_ms':0,
          'sustain':1,'hold_ms':10,'release_ms':0,'repeat_count':2,
          'repeat_interval_ms':10,'level':1}
    intervals,_=worker._timed_exposure_program(frames,fixtures,1.0,graph.to_dict(),
                                               trigger_generators=[spec])
    def gain_at(t):
        row=next(row for row in intervals if row['t0']<=t<row['t1'])
        return row['emitter_profiles']['records'][0]['gain']
    assert gain_at(.00205)==pytest.approx(3.6,abs=.01)
    assert gain_at(.01205)==pytest.approx(3.6,abs=.01)
    assert gain_at(.02205)==0
