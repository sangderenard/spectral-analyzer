"""Bounded, GPU-free coverage of the photo worker's per-job film contract."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from camera_designer.film_response import normalize_film_response
from camera_designer import marble_preview_worker as worker


def test_defaults_are_independent_and_preserve_units():
    expected = {"version": 1, "centers_nm": [650, 550, 450], "sigmas_oct": [.06] * 3}
    response = normalize_film_response()
    assert response == expected
    response["centers_nm"][0] = 400
    assert normalize_film_response() == expected
    bounds = {"version": 1, "centers_nm": [380, 780, 380], "sigmas_oct": [.01, 1, .06]}
    assert normalize_film_response(bounds) == bounds
    assert normalize_film_response(bounds)["centers_nm"] is not bounds["centers_nm"]


@pytest.mark.parametrize("value", [None, False, [], "default", {},
    {"version": 2, "centers_nm": [650, 550, 450], "sigmas_oct": [.06] * 3},
    {"version": True, "centers_nm": [650, 550, 450], "sigmas_oct": [.06] * 3},
    {"version": 1.0, "centers_nm": [650, 550, 450], "sigmas_oct": [.06] * 3},
    {"version": 1, "centers_nm": [650, 550, 450], "sigmas_nm": [20] * 3},
    {"version": 1, "centers_nm": [650, 550, 450], "sigmas_oct": [.06] * 3, "gain": 2},
])
def test_invalid_objects_rejected(value):
    with pytest.raises(ValueError, match="film_response"):
        normalize_film_response(value)


@pytest.mark.parametrize("key", ["centers_nm", "sigmas_oct"])
@pytest.mark.parametrize("values", [None, [], [1, 2], [1, 2, 3, 4], "650,550,450",
    [True, 550, 450], ["650", 550, 450], [float("nan"), 550, 450],
    [float("inf"), 550, 450], [float("-inf"), 550, 450], [10**1000, 550, 450],
])
def test_invalid_triplets_rejected(key, values):
    response = normalize_film_response()
    response[key] = values
    with pytest.raises(ValueError, match="film_response"):
        normalize_film_response(response)


@pytest.mark.parametrize("key,value", [
    ("centers_nm", 379.99), ("centers_nm", 780.01),
    ("sigmas_oct", 0), ("sigmas_oct", .00999), ("sigmas_oct", 1.0001),
])
def test_out_of_bounds_rejected(key, value):
    response = normalize_film_response()
    response[key][1] = value
    with pytest.raises(ValueError, match="film_response"):
        normalize_film_response(response)


def test_actual_renderer_preset_copy_preserves_defaults_and_other_fields():
    # Import the actual coordinator in a clean bounded process, not under
    # conftest's pygame/OpenGL test doubles. Never create a window or context.
    code = r'''
from copy import deepcopy
from camera_designer.film_response import build_film_stack, normalize_film_response
from demo_pluck_gl import FILM_STACKS
preset = FILM_STACKS["em_rgb_fixed3"]
before = deepcopy(preset.layers)
default = build_film_stack(preset, normalize_film_response())
assert default is not preset
assert default.active == preset.active
assert default.layers == before
assert default.layer_specs() == preset.layer_specs()
assert default.transport_wavelengths_nm == (650., 550., 450.)
assert not hasattr(preset, "transport_wavelengths_nm")
response = {"version":1,"centers_nm":[610,530,470],"sigmas_oct":[.2,.15,.1]}
custom = build_film_stack(preset, response)
for i, name in enumerate(("R", "G", "B")):
    expected = deepcopy(before[name])
    expected.response_center_hz = 2.998e14 / (response["centers_nm"][i] / 1000)
    expected.response_width_oct = response["sigmas_oct"][i]
    assert custom.layers[name] == expected
    assert custom.layers[name] is not preset.layers[name]
    assert custom.layer_specs()[i]["center_hz"] == expected.response_center_hz
    assert custom.layer_specs()[i]["width_oct"] == expected.response_width_oct
assert preset.layers == before
assert default.layers == before
print("film-preset-preserved")
'''
    completed = subprocess.run([sys.executable, "-c", code],
                               cwd=Path(__file__).resolve().parents[1],
                               check=False, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "film-preset-preserved" in completed.stdout


@pytest.fixture
def fake_worker_renderer(monkeypatch):
    """Exercise real worker/snapshot code without dispatching a GPU exposure."""
    from camera_designer import studio_photo_geometry
    calls = SimpleNamespace(forward=[], forward_params=[], sensors=[], cleared=[], snapshots=[], appended=[], meshes=[], built=[])
    preset = SimpleNamespace(layers={name: SimpleNamespace(
        response_center_hz=2.998e14 / wavelength, response_width_oct=.06,
    ) for name, wavelength in zip(("R", "G", "B"), (.65, .55, .45))})

    def forward(*args, **kwargs):
        calls.forward.append(kwargs["film_stack"])
        calls.forward_params.append(kwargs)
        return {"prog": 31, "ssbo": list(range(13)), "tex_bands": [1, 2, 3],
                "bmin": np.zeros(3), "bmax": np.ones(3), "dims": (1, 1, 1)}

    class Sensor:
        def __init__(self, **kwargs):
            self._active = True
            self.params = kwargs
            self._w = self._h = 2
            self._textures = [1, 2, 3]
            self.tex = 1
            self.film = None
            calls.sensors.append(self)

        def tick(self, **kwargs):
            assert self.film is calls.forward[-1]

        def destroy(self):
            pass

    renderer = SimpleNamespace(
        FILM_STACKS={"em_rgb_fixed3": preset},
        _ensure_worker_gl_context=lambda: None,
        _build_gpu_bvh=lambda triangles: (np.zeros((1, 4)), np.zeros(1)),
        _gpu_ray_field_prebuilt=forward, SensorAccumulator=Sensor,
        _gpu_pump_prebuilt=lambda *args, **kwargs: None,
        _destroy_prebuilt=lambda state: calls.cleared.append(state),
    )
    monkeypatch.setitem(sys.modules, "demo_pluck_gl", renderer)
    scene = (np.zeros((1, 16)), None, None, np.zeros((1, 3, 3)), np.zeros((0, 8)), None,
             (np.zeros(3), np.ones(3)), [])
    def append(scene, *args, **kwargs):
        calls.appended.append(kwargs)
        calls.meshes.append(args)
        if kwargs.get("return_index_map"):
            return scene, np.arange(len(args[0]))
        return scene

    def build(*args, **kwargs):
        calls.built.append(kwargs)
        return scene

    monkeypatch.setitem(sys.modules, "camera_designer.scene_builder", SimpleNamespace(
        build_gpu_scene=build,
        append_saved_studio_mesh_to_gpu_scene=append,
    ))
    # CPU scene preparation is shared with authenticated admission now.
    monkeypatch.setattr(studio_photo_geometry, "build_gpu_scene", build)
    monkeypatch.setattr(studio_photo_geometry, "append_saved_studio_mesh_to_gpu_scene", append)
    gl = sys.modules["OpenGL.GL"]
    for name in ("GL_TEXTURE_2D", "GL_RGBA", "GL_FLOAT"):
        monkeypatch.setattr(gl, name, 1, raising=False)
    monkeypatch.setattr(gl, "glFinish", lambda: None, raising=False)
    monkeypatch.setattr(gl, "glBindTexture", lambda *args: None, raising=False)
    monkeypatch.setattr(gl, "glGetTexImage", lambda *args:
                        np.ones((2, 2, 4), np.float32).tobytes(), raising=False)
    calls.preset = preset
    return calls


def _job_files(tmp_path, config):
    payload = tmp_path / "mesh.npz"
    np.savez(payload, vertices=np.zeros((1, 3, 3)), material_ids=np.zeros(1), pbr=np.zeros((1, 16)))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return payload, config_path


@pytest.mark.parametrize("budget", [1, 4, 12, 13, 50, 200])
def test_full_bounce_budget_reaches_forward_and_sensor(tmp_path, fake_worker_renderer, budget):
    payload, config = _job_files(tmp_path, {"transport": "fixed:3", "max_bounces": budget})
    status = worker.run(payload, config, tmp_path / "job", seconds=.1, max_epochs=1)
    assert status["state"] == "complete"
    assert fake_worker_renderer.forward_params[0]["max_bounces"] == budget
    assert fake_worker_renderer.sensors[0].params["max_bounces"] == budget


@pytest.mark.parametrize("value", [0, 201, 200.1, 1.5, True, "200", float("inf"), float("nan")])
def test_invalid_bounce_budget_rejected_before_renderer(tmp_path, fake_worker_renderer, value):
    payload, config = _job_files(tmp_path, {"transport": "fixed:3", "max_bounces": value})
    with pytest.raises(ValueError, match="max_bounces"):
        worker.run(payload, config, tmp_path / "job", seconds=.1, max_epochs=1)
    assert not fake_worker_renderer.built and not fake_worker_renderer.forward
    assert not (tmp_path / "job").exists()


def test_default_bounce_budget_remains_four():
    assert worker._bounce_limit() == 4


@pytest.mark.parametrize("explicit", [False, True])
def test_worker_shares_stack_and_persists_response(tmp_path, fake_worker_renderer, explicit):
    calls = fake_worker_renderer
    original = deepcopy(calls.preset.layers)
    response = {"version": 1, "centers_nm": [610, 530, 470], "sigmas_oct": [.2, .15, .1]}
    config = {"transport": "fixed:3"}
    if explicit:
        config["film_response"] = response
    else:
        response = normalize_film_response()
    payload, config_path = _job_files(tmp_path, config)
    output = tmp_path / "job"
    status = worker.run(payload, config_path, output, seconds=.1, max_epochs=1)
    assert status["state"] == "complete"
    assert status["film_response"] == response
    assert json.loads((output / "status.json").read_text())["film_response"] == response
    assert json.loads((output / "effective_config.json").read_text())["film_response"] == response
    assert json.loads(config_path.read_text()) == config
    assert status["digest"] == hashlib.sha256(payload.read_bytes() + json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with np.load(output / status["hdr"], allow_pickle=False) as hdr:
        assert hdr["film_response_version"] == 1
        assert hdr["film_centers_nm"].tolist() == response["centers_nm"]
        assert hdr["film_sigmas_oct"].tolist() == response["sigmas_oct"]
        assert json.loads(str(hdr["geometry_stats_json"])) == status["geometry_stats"]
    film = calls.forward[0]
    assert film.transport_wavelengths_nm == (650., 550., 450.)
    assert calls.appended[0]["spectral_emission_rgb"] is True
    assert calls.sensors[0].film is film
    assert film is not calls.preset
    for index, layer in enumerate(film.layers.values()):
        assert layer.response_center_hz == 2.998e14 / (response["centers_nm"][index] / 1000)
        assert layer.response_width_oct == response["sigmas_oct"][index]
    assert calls.preset.layers == original
    assert len(calls.cleared) == 1


def test_worker_monochrome_unchanged_and_explicit_response_rejected(tmp_path, fake_worker_renderer):
    payload, config = _job_files(tmp_path, {"transport": "fixed:1"})
    status = worker.run(payload, config, tmp_path / "job", seconds=.1, max_epochs=1)
    assert status["state"] == "complete"
    assert "film_response" not in status
    assert fake_worker_renderer.forward == [None]
    assert fake_worker_renderer.appended[0]["spectral_emission_rgb"] is False
    _, config = _job_files(tmp_path, {"transport": "fixed:1", "film_response": normalize_film_response()})
    with pytest.raises(ValueError, match="requires fixed:3"):
        worker.run(payload, config, tmp_path / "bad", seconds=.1, max_epochs=1)
    assert not (tmp_path / "bad").exists()


def test_worker_explicit_null_rejected_before_renderer(tmp_path, fake_worker_renderer):
    payload, config = _job_files(tmp_path, {"transport": "fixed:3", "film_response": None})
    with pytest.raises(ValueError, match="film_response"):
        worker.run(payload, config, tmp_path / "bad", seconds=.1, max_epochs=1)
    assert not fake_worker_renderer.forward


def test_worker_fixture_mesh_uses_same_spectral_emission_mode(tmp_path, fake_worker_renderer):
    fixtures = {
        "vertices": [-.03, -.04, .08, .03, -.04, .08, .03, .04, .08,
                     -.02, -.03, .081, .02, -.03, .081, .02, .03, .081],
        "material_ids": [0, 1],
        "pbr": [.2, .3, .4, .8, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                .8, .7, .1, .5, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    payload, config = _job_files(tmp_path, {
        "transport": "fixed:3",
        "fixtures": fixtures,
    })
    worker.run(payload, config, tmp_path / "job", seconds=.1, max_epochs=1)
    assert len(fake_worker_renderer.appended) == 2
    assert all(call["spectral_emission_rgb"] is True for call in fake_worker_renderer.appended)
    assert len(fake_worker_renderer.built) == 1
    assert fake_worker_renderer.built[0]["scene_object"] is None
    imported_vertices, imported_ids, imported_pbr = fake_worker_renderer.meshes[1]
    np.testing.assert_array_equal(imported_vertices, np.asarray(fixtures["vertices"], np.float32).reshape(-1, 3, 3))
    np.testing.assert_array_equal(imported_ids, fixtures["material_ids"])
    np.testing.assert_array_equal(imported_pbr, np.asarray(fixtures["pbr"], np.float32).reshape(-1, 16))
    assert fake_worker_renderer.appended[1]["scale_m"] == 1
    assert fake_worker_renderer.appended[1]["center_m"] == [0, 0, 0]


def test_orbit_keeps_one_response_and_encoding_metadata(tmp_path, fake_worker_renderer, monkeypatch):
    response = {"version": 1, "centers_nm": [610, 530, 470], "sigmas_oct": [.2, .15, .1]}
    payload, config = _job_files(tmp_path, {"transport": "fixed:3", "orbit_frames": 8,
                                          "film_response": response})
    published = []
    atomic = worker._atomic_json

    def record(path, data):
        if path.name == "status.json":
            published.append(deepcopy(data))
        atomic(path, data)

    def encode(output, frames, resolution, root, on_frame):
        for i in range(1, len(frames) + 1):
            on_frame(i)

    monkeypatch.setattr(worker, "_atomic_json", record)
    monkeypatch.setattr(worker, "_write_orbit_avi", encode)
    status = worker.run(payload, config, tmp_path / "job", seconds=.1,
                        max_epochs=1, avi_writer_root=tmp_path)
    assert status["state"] == "complete"
    assert len(status["frames"]) == 8
    assert {item["state"] for item in published} == {"preparing", "exposing", "encoding", "complete"}
    assert all(item["film_response"] == response for item in published)
    assert all(sensor.film is fake_worker_renderer.forward[0]
               for sensor in fake_worker_renderer.sensors)


def test_cli_error_keeps_same_job_response(tmp_path, monkeypatch):
    response = normalize_film_response()
    payload, config = _job_files(tmp_path, {"transport": "fixed:3", "film_response": response})
    output = tmp_path / "job"
    output.mkdir()
    digest = hashlib.sha256(payload.read_bytes() + json.dumps(
        json.loads(config.read_text()), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    worker._atomic_json(output / "status.json", {"digest": digest, "state": "preparing",
                                                "film_response": response})

    def fail(*args, **kwargs):
        raise RuntimeError("shader did not link")

    monkeypatch.setattr(worker, "run", fail)
    monkeypatch.setattr(sys, "argv", ["worker", "--payload", str(payload), "--config",
                                      str(config), "--out", str(output)])
    with pytest.raises(RuntimeError, match="shader did not link"):
        worker.main()
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "error"
    assert status["film_response"] == response

@pytest.mark.parametrize('failed', [False, True])
def test_auto_mode_chooses_global_limit_and_publishes_limits(tmp_path, fake_worker_renderer, monkeypatch, failed):
    from camera_designer import marble_bounce_probe
    renderer=sys.modules['demo_pluck_gl']
    calls=fake_worker_renderer
    calls.uniforms=[];calls.field_resets=[]
    monkeypatch.setattr(renderer,'_clear_prebuilt',lambda state:calls.field_resets.append(state),raising=False)
    def probe(*args,progress,**kwargs):
        assert kwargs['n_triangles']==1 and kwargs['context_count']==0
        progress(0,4)
        if failed: raise RuntimeError('test incomplete pilot')
        return {'mode':'auto','chosen_depth':21,'render_cap':200,'probe_depth':1000,
                'confidence':'limited sparse pilot','warning':''}
    monkeypatch.setattr(marble_bounce_probe,'run_bounce_probe',probe)
    gl=sys.modules['OpenGL.GL']
    monkeypatch.setattr(gl,'glUseProgram',lambda *args:None,raising=False)
    monkeypatch.setattr(gl,'glGetUniformLocation',lambda program,name:name,raising=False)
    monkeypatch.setattr(gl,'glUniform1i',lambda name,value:calls.uniforms.append((name,value)),raising=False)
    original={'transport':'fixed:3','bounce_mode':'auto','max_bounces':37}
    payload,config=_job_files(tmp_path,original)
    output=tmp_path/'job'
    status=worker.run(payload,config,output,seconds=.1,max_epochs=1)
    chosen=200 if failed else 21
    assert calls.sensors[0].params['max_bounces']==chosen
    assert (b'uMaxBounces',chosen) in calls.uniforms
    assert len(calls.field_resets)==1
    assert status['bounce_diagnostic']['chosen_depth']==chosen
    assert status['bounce_diagnostic']['probe_depth']==1000
    effective=json.loads((output/'effective_config.json').read_text())
    assert effective['max_bounces']==chosen and effective['requested_manual_bounces']==37
    assert json.loads(config.read_text())==original
    if failed: assert 'convergence is unknown' in status['bounce_diagnostic']['warning']
