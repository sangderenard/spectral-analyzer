"""CPU-only live-loop budget, sweep-boundary and dispatch accounting checks."""
import ast
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from camera_designer import marble_live_worker as worker


@pytest.mark.parametrize("size", [None, True, 127, 65537, 8192.0, "8192"])
def test_worker_rejects_invalid_budget_before_gl_or_sensor_access(size, tmp_path):
    with pytest.raises(ValueError, match="ray_batch_size"):
        worker.run_live_sensor(None, None, None, tmp_path, "a" * 64,
            {"live": {"ray_batch_size": size}}, 0, None, None)


def run_loop(tmp_path, monkeypatch, *, height=83, changes=None, epochs=5, batch=8192, pause_at=None, cooperative=False):
    gl = importlib.import_module("OpenGL.GL")
    monkeypatch.setattr(gl, "glFinish", lambda: None, raising=False)
    clock = [0.]
    def monotonic():
        clock[0] += .3
        return clock[0]
    monkeypatch.setattr(worker.time, "monotonic", monotonic)
    camera = dict(eye_m=[0, 0, .07], target_m=[0, 0, .18], fov_degrees=18,
                  aperture_radius_m=.002, focus_distance_m=.11)
    live = dict(history_samples=2, aspect=1, lease_seconds=30, checkpoint=True,
                ray_batch_size=batch, display=dict(tint=0, normalization="auto99"))
    events, statuses = [], []
    field = {"_pump_seed": 0, "batch": None}
    pauses=[]
    def sleep(delay):
        assert delay==.05
        pauses.append((len(events),field['_pump_seed']))
        path=tmp_path/'live-control.json';control=json.loads(path.read_text())
        if len(pauses)==2:
            control.update(scheduler_generation=2,scheduler_state='running')
        if len(pauses)==3:
            control['expires_at']=worker.time.time()+30
        path.write_text(json.dumps(control))
    monkeypatch.setattr(worker.time,'sleep',sleep)
    class Accumulator:
        _w = 2
        _h = height
        _rows_per_frame = 16
        _row_off = 0
        _pass = 0
        _history_bytes = 100
        ticks = 0
        def set_history(self, n, checkpoint=False):
            self._history_samples = n
            self._history_checkpoint = checkpoint
        def set_camera(self, *args, **kwargs):
            pass
        def set_lens(self, *args, **kwargs):
            pass
        def clear(self):
            events.append(("reset", self.ticks, self._row_off, self._pass))
            self._row_off = self._pass = 0
        def tick(self, strips):
            assert strips == 1
            events.append(("tick", self._row_off, field["batch"]))
            self._row_off += min(self._rows_per_frame, self._h - self._row_off)
            if self._row_off == self._h:
                self._row_off = 0
                self._pass += 1
            self.ticks += 1
            if self.ticks==pause_at:
                (tmp_path/'live-control.json').write_text(json.dumps(dict(camera=camera,live=live,
                    revision=0,expires_at=0,scheduler_generation=1,scheduler_state='paused')))
            if changes and self.ticks in changes:
                revision, live_update = changes[self.ticks]
                control = dict(camera=camera, live={**live, **live_update},
                               revision=revision, expires_at=worker.time.time() + 30)
                (tmp_path / "live-control.json").write_text(json.dumps(control))
    def clear(f):
        events.append(("clear_field",))
        f["_pump_seed"] = 0
    def pump(f, n_rays):
        events.append(("pump", f["_pump_seed"], n_rays))
        f["_pump_seed"] += 7
        f["batch"] = n_rays
    def snapshot(accumulator, output, digest, epoch, elapsed, preparation,
                 rays, sensor_sites, **kwargs):
        return dict(image="test.png", forward_ray_jobs=rays, sensor_sites=sensor_sites,
                    display=kwargs["display_params"], pass_at_snapshot=accumulator._pass)
    renderer = SimpleNamespace(_clear_prebuilt=clear, _gpu_pump_prebuilt=pump)
    if cooperative:
        from camera_designer import studio_cooperative_sensor
        class Pump:
            def __init__(self,*args,**kwargs):self.profile={'committed_rays':0}
            def step(self,a):
                sites=min(a._rows_per_frame,a._h-a._row_off)*a._w
                events.append(('client_parent',a._row_off,sites))
                a.tick(1);self.profile['committed_rays']+=sites
            def close(self):events.append(('client_close',))
        monkeypatch.setattr(studio_cooperative_sensor,'CooperativeSensorPump',Pump)
    result = worker.run_live_sensor(Accumulator(), renderer, field, tmp_path, "a" * 64,
        dict(live=live, camera=camera,client_intersections=cooperative), 0, snapshot,
        lambda path, status: statuses.append(dict(status)), max_epochs=epochs)
    if pause_at is not None:
        assert len(pauses)==3 and len(set(pauses))==1, 'yield must do no work even after expired-lease resume command'
    return result, events, statuses


def test_scheduler_yields_at_sweep_boundary_and_resumes_exact_history(tmp_path,monkeypatch):
    result,events,statuses=run_loop(tmp_path,monkeypatch,pause_at=4,epochs=6)
    yields=[s for s in statuses if s['state']=='yielded']
    assert yields and all(s['retained_row']==0 and s['retained_pass']==1 for s in yields)
    assert all(s['gpu_dispatch_yielded'] and not s['gpu_memory_released'] for s in yields)
    assert all(s['client_lease_expired'] for s in yields)
    assert not any(e[0]=='reset' for e in events)
    ticks=[e[1] for e in events if e[0]=='tick']
    assert ticks==([0,16,32,48,64,80]*10)[:len(ticks)]
    pumps=[e[1] for e in events if e[0]=='pump']
    assert pumps==list(range(0,7*len(pumps),7))
    assert result['scheduler_generation']==2


def test_scheduler_generations_reject_stale_resume():
    assert worker._scheduler_control(dict(scheduler_generation=4,scheduler_state='running'),5,'paused')==(5,'paused')
    assert worker._scheduler_control(dict(scheduler_generation=6,scheduler_state='running'),5,'paused')==(6,'running')
    for control in (dict(scheduler_generation=True,scheduler_state='paused'),dict(scheduler_generation=2,scheduler_state='skip')):
        with pytest.raises(ValueError):worker._scheduler_control(control,1,'running')


def test_budget_changes_only_at_sweep_boundary_and_resets_checkpoint(tmp_path, monkeypatch):
    result, events, statuses = run_loop(tmp_path, monkeypatch,
        changes={4: (1, {"ray_batch_size": 16384})})
    ticks = [event for event in events if event[0] == "tick"]
    assert ticks[:6] == [("tick", row, 8192) for row in (0, 16, 32, 48, 64, 80)]
    assert all(event[2] == 16384 for event in ticks[6:])
    assert [event for event in events if event[0] == "reset"] == [("reset", 6, 0, 1)]
    pumps = [event for event in events if event[0] == "pump"]
    assert pumps == [("pump", 0, 8192), ("pump", 7, 16384),
                     ("pump", 14, 16384), ("pump", 21, 16384)]
    assert result["forward_ray_jobs"] == sum(event[2] for event in pumps)
    assert result["sensor_sites"] == 3 * 83 * 2 + 32 * 2
    assert result["ray_batch_size"] == 16384 and result["live_revision"] == 1
    assert result["camera_settings"]["focus_distance_m"] > 0
    assert result["forward_history"] == "fresh 16384-ray field per sensor sweep"
    assert result["checkpoint_enabled"] and result["checkpoint_samples"] == 0
    assert statuses[0]["pass_at_snapshot"] == 0 and statuses[0]["live_revision"] == 0
    assert statuses[1]["pass_at_snapshot"] == 0 and statuses[1]["live_revision"] == 1


def test_latest_pending_budget_wins_before_boundary(tmp_path, monkeypatch):
    result, events, _ = run_loop(tmp_path, monkeypatch, height=129, epochs=6,
        changes={4: (1, {"ray_batch_size": 16384}), 8: (2, {"ray_batch_size": 32768})})
    assert [event for event in events if event[0] == "reset"] == [("reset", 9, 0, 1)]
    assert {event[2] for event in events if event[0] == "pump"} == {8192, 32768}
    assert result["live_revision"] == 2 and result["ray_batch_size"] == 32768


def test_return_to_active_budget_cancels_pending_reset(tmp_path, monkeypatch):
    result, events, _ = run_loop(tmp_path, monkeypatch, height=129, epochs=6,
        changes={4: (1, {"ray_batch_size": 16384}), 8: (2, {"ray_batch_size": 8192})})
    assert not any(event[0] == "reset" for event in events)
    assert result["live_revision"] == 2 and result["ray_batch_size"] == 8192


def test_display_only_change_keeps_ring_and_checkpoint(tmp_path, monkeypatch):
    display = dict(tint=1, normalization="specular", normalization_percentile=95)
    result, events, _ = run_loop(tmp_path, monkeypatch,
        changes={4: (1, {"display": display})})
    assert not any(event[0] == "reset" for event in events)
    assert result["display"] == display and result["live_revision"] == 1
    assert result["checkpoint_samples"] == 1 and result["total_samples"] == 3


@pytest.mark.parametrize("size", [128, 129, 8192, 65536])
def test_real_pump_submits_exact_total_in_bounded_dispatches(size, monkeypatch):
    # Compile the actual small pump function without importing the renderer's
    # app startup. Only GL calls are replaced; no GPU context is created.
    source = Path(__file__).resolve().parents[1] / "demo_pluck_gl.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_gpu_pump_prebuilt")
    from mat_flags import BINDING_STUDIO_NATIVE_PBR
    namespace = {"math": math, "GL_TRUE": 1, 'BINDING_STUDIO_NATIVE_PBR': BINDING_STUDIO_NATIVE_PBR}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    gl = importlib.import_module("OpenGL.GL")
    uniforms, dispatches = {}, []
    for name in ("GL_SHADER_STORAGE_BARRIER_BIT", "GL_SHADER_IMAGE_ACCESS_BARRIER_BIT",
                 "GL_SHADER_STORAGE_BUFFER", "GL_R32UI", "GL_READ_WRITE", "GL_TEXTURE_3D"):
        monkeypatch.setattr(gl, name, 1, raising=False)
    for name in ("glUseProgram", "glBindBufferBase", "glBindImageTexture", "glMemoryBarrier"):
        monkeypatch.setattr(gl, name, lambda *args: None, raising=False)
    monkeypatch.setattr(gl, "glUniform1i", lambda location, value: uniforms.__setitem__(location, value), raising=False)
    monkeypatch.setattr(gl, "glDispatchCompute", lambda x, y, z: dispatches.append(
        (uniforms["batch_off"], uniforms["batch_size"], x * y * 128)), raising=False)
    state = dict(prog=1, tex_bands=[1, 2, 3], ssbo=[1], dispatch_batch=512, _pump_seed=7,
                 n_sources=1, native_pbr_ssbo=2,
                 locs={name: name for name in ("mode", "seed", "total_rays", "batch_size", "batch_off")})
    namespace["_gpu_pump_prebuilt"](state, size)
    assert uniforms["total_rays"] == size and uniforms["seed"] == 7
    assert state["_pump_seed"] == 14
    assert sum(count for _, count, _ in dispatches) == size
    assert [offset for offset, _, _ in dispatches] == list(range(0, size, 512))
    assert all(count <= 512 and count <= launched < count + 128 for _, count, launched in dispatches)


def test_cooperative_live_uses_retained_parent_and_reports_committed_sites(tmp_path,monkeypatch):
    result,events,statuses=run_loop(tmp_path,monkeypatch,cooperative=True,height=83)
    parents=[e for e in events if e[0]=='client_parent']
    assert parents and any(e[2]==6 for e in parents)
    assert events[-1]==('client_close',)
    assert result['client_work_profile']['committed_rays']==sum(e[2] for e in parents)
    assert result['sensor_sites']==sum(e[2] for e in parents)
