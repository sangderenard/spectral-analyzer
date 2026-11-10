"""Native housing/sensor pose and atomic live revision contracts (CPU only)."""
import ast
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_designer.camera_frame import sensor_camera_basis, thin_lens_housing_frame
from camera_designer.camera_preset import simple_doublet_preset, thin_lens_housing_preset
from camera_designer.scene_builder import _collect_optical_geometry, build_gpu_scene
from camera_designer.studio_photo_geometry import prepare_photo_geometry, camera_geometry_key


ROOT = Path(__file__).resolve().parents[1]


def extracted_renderer(name, *, method=False):
    tree = ast.parse((ROOT / "demo_pluck_gl.py").read_text(encoding="utf-8"))
    if method:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SensorAccumulator")
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native-camera-test", "exec"), ns)
    return ns[name]


def nearest_hit(tris, origin, direction):
    edge1, edge2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, p)
    valid = abs(det) > 1e-15
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    tvec = origin - tris[:, 0]
    u = np.einsum("ij,ij->i", tvec, p) * inv
    q = np.cross(tvec, edge1)
    v = q @ direction * inv
    t = np.einsum("ij,ij->i", edge2, q) * inv
    valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (t > 1e-8)
    return float(t[valid].min()) if valid.any() else np.inf


@pytest.fixture(scope="module")
def housing():
    preset = thin_lens_housing_preset({"aperture_radius_m": .002, "aperture_blades": 7,
                                       "aperture_rotation_radians": .23})
    local = _collect_optical_geometry(preset, None, [.45, .55, .65])
    return preset, local


@pytest.mark.parametrize("eye,target", [
    ([.1621047866975828, .06854091912429612, .4767308216807413], [0, 0, .18]),
    ([0, 0, .07], [0, 0, .18]), ([.8, -.3, .1], [-.1, .2, -.9]),
    ([0, 1, 0], [0, 0, 0]), ([0, -1, 0], [0, 0, 0]),
])
def test_pose_is_proper_rigid_transform_and_all_forward_pupil_rays_clear(eye, target, housing):
    preset, _ = housing
    frame = thin_lens_housing_frame(preset, eye, target)
    rotation, translation = frame
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=3e-7)
    assert np.linalg.det(rotation) == pytest.approx(1, abs=3e-7)
    geo = _collect_optical_geometry(preset, None, [.45, .55, .65], camera_frame=frame)
    tris, normals = geo[0].reshape(-1, 3, 3), geo[1]
    eye, right, up, fwd = sensor_camera_basis(eye, target)
    projected = (tris - eye) @ fwd
    assert projected.max() == pytest.approx(-preset.aperture_stop.thickness_m, abs=4e-8)
    # Complete half-space proof: every point on every triangle is behind all
    # pupil origins; every forward sensor direction monotonically leaves it.
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        pupil = eye + .002 * (right * np.cos(angle) + up * np.sin(angle))
        assert np.max((tris - pupil) @ fwd) < -9e-5
        for x, y in ((0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)):
            direction = fwd + right * x + up * y
            direction /= np.linalg.norm(direction)
            assert direction @ fwd > 0
        assert np.isinf(nearest_hit(tris, pupil, fwd))
    # Secondary/reflected rays still encounter the actual side and back body.
    for local_origin, local_dir, distance in (
        ([.2, .011, .01], [-1, 0, 0], .2 - preset.body.width / 2),
        ([.011, .013, -.2], [0, 0, 1], .2 + .05 - preset.body.depth),
    ):
        origin = np.asarray(local_origin) @ rotation.T + translation
        direction = np.asarray(local_dir) @ rotation.T
        assert nearest_hit(tris, origin, direction) == pytest.approx(distance, abs=2e-7)
    # Native geometry keeps winding and normals under the same proper frame.
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert np.all(np.isfinite(normals))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=3e-7)
    # The new resolved front shell has positive outward normals and winding.
    at_front = np.all(abs(projected + .00015) < 4e-8, axis=1)
    assert at_front.any()
    assert np.all(cross[at_front] @ fwd > 0)


def test_native_preset_defaults_and_screen_pupil_polygon_are_preserved():
    original = simple_doublet_preset()
    assert len(original.lens_group.elements) == 3
    assert original.aperture_stop.n_blades == 8
    camera = {"aperture_radius_m": .002, "aperture_blades": 7, "aperture_rotation_radians": .23}
    preset = thin_lens_housing_preset(camera)
    assert preset.body.to_dict() == original.body.to_dict()
    assert preset.lens_group.elements == []
    assert preset.sensor.to_dict() == original.sensor.to_dict()
    polygon = preset.aperture_stop.blade_polygon_xy()
    screen = polygon * [-1, 1]
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + .23
    expected = .002 * np.column_stack((np.cos(angles), np.sin(angles)))
    assert all(np.min(np.linalg.norm(screen - point, axis=1)) < 1e-14 for point in expected)


@pytest.mark.parametrize("radius", [1e-5, .002, .02])
def test_small_to_large_pupil_has_actual_open_native_body_port(radius):
    preset = thin_lens_housing_preset({"aperture_radius_m": radius})
    frame = thin_lens_housing_frame(preset, [0, 0, 0], [0, 0, 1])
    geo = _collect_optical_geometry(preset, None, [.55], camera_frame=frame)
    tris = geo[0].reshape(-1, 3, 3)
    # The first actual obstruction through the entrance pupil is the native
    # sensor at 52.55 mm behind the eye, never a sealed front body face.
    assert nearest_hit(tris, np.array([radius * .5, 0, .01]), np.array([0, 0, -1])) == pytest.approx(.06255, abs=1e-9)


def test_native_sensor_uses_the_exact_shared_basis_including_vertical_view():
    method = extracted_renderer("set_camera", method=True)
    for target in ([0, 1, 0], [0, -1, 0], [2, -1, 3], [0, 0, 0]):
        obj = SimpleNamespace()
        method(obj, [0, 0, 0], target, fov_deg=36)
        for actual, expected in zip((obj._eye, obj._right, obj._up, obj._fwd), sensor_camera_basis([0, 0, 0], target)):
            np.testing.assert_array_equal(actual, expected)


def test_world_objects_and_sources_do_not_follow_the_camera_or_marble_scale():
    vertices = np.array([[[0, 0, 0], [.1, 0, 0], [0, .1, 0]]], np.float32)
    ids = np.zeros(1, np.int32)
    pbr = np.zeros((1, 16), np.float32)
    pbr[:, :3] = .8; pbr[:, 6] = 1.5; pbr[:, 7] = 1
    base = {"transport": "fixed:3", "scale_m": .01, "camera": {
        "eye_m": [.2, .1, .4], "target_m": [0, 0, .18], "aperture_radius_m": .002}}
    scenes = []
    for scale, eye in ((.01, [.2, .1, .4]), (.04, [-.4, .8, .6])):
        scene, _, stats, _ = prepare_photo_geometry({**base, "scale_m": scale,
            "camera": {**base["camera"], "eye_m": eye}}, vertices, ids, pbr)
        scenes.append(scene)
        assert stats["camera_geometry"]["body_dimensions_m"] == [.139, .101, .077]
        assert stats["camera_geometry"]["eye_m"] == eye
        np.testing.assert_allclose(scene[3][-1], vertices[0] * scale + [0, 0, .18])
    # Existing checker triangles immediately precede the single saved triangle.
    np.testing.assert_array_equal(scenes[0][3][-19:-1], scenes[1][3][-19:-1])
    np.testing.assert_array_equal(scenes[0][7], scenes[1][7])
    np.testing.assert_array_equal(scenes[0][1], scenes[1][1])
    # Changing field of view/focus/tint does not move native body geometry.
    assert camera_geometry_key(base["camera"], [0, 0, .3]) == camera_geometry_key(
        {**base["camera"], "fov_degrees": 90, "focus_distance_m": 1}, [0, 0, .3])


def test_shared_gpu_geometry_update_replaces_bounds_counts_and_all_owned_buffers(monkeypatch):
    gl = importlib.import_module("OpenGL.GL")
    events = []; bound = [None]
    for name in ("GL_SHADER_STORAGE_BUFFER", "GL_STATIC_DRAW"):
        monkeypatch.setattr(gl, name, 1, raising=False)
    monkeypatch.setattr(gl, "glFinish", lambda: events.append(("finish",)), raising=False)
    monkeypatch.setattr(gl, "glBindBuffer", lambda target, handle: bound.__setitem__(0, handle), raising=False)
    monkeypatch.setattr(gl, "glBufferData", lambda target, size, data, usage: events.append(("upload", bound[0], size)), raising=False)
    monkeypatch.setattr(gl, "glUseProgram", lambda *args: None, raising=False)
    monkeypatch.setattr(gl, "glGetUniformLocation", lambda prog, name: name, raising=False)
    monkeypatch.setattr(gl, "glUniform1i", lambda name, value: events.append((name, value)), raising=False)
    monkeypatch.setattr(gl, "glUniform3f", lambda name, *value: events.append((name, value)), raising=False)
    state = dict(ssbo=list(range(100, 113)), prog=1)
    data = np.zeros((3, 16), np.float32)
    extracted_renderer("_update_prebuilt_geometry")(state, data, data, data,
        np.zeros((5, 12), np.float32), np.arange(3, dtype=np.uint32),
        np.zeros((1, 8), np.float32), ([-4, -3, -2], [4, 5, 6]), np.zeros((2, 8), np.float32))
    assert events[0] == ("finish",)
    assert [e[1] for e in events if e[0] == "upload"] == [100, 101, 102, 108, 109, 110, 112]
    assert (b"uTriCount", 3) in events and (b"uNodeCount", 5) in events
    assert (b"uStudioSphereCount", 2) in events
    obj = SimpleNamespace()
    extracted_renderer("set_scene_geometry", method=True)(obj, n_tris=3, n_bvh_nodes=5,
        bmin=state["bmin"], bmax=state["bmax"], n_spheres=2)
    assert obj._n_bvh_nodes == 5 and obj._n_spheres == 2
    np.testing.assert_array_equal(obj._bmin, [-4, -3, -2])


@pytest.mark.parametrize("display_only", [False, True])
def test_live_geometry_and_view_accept_same_revision_only_at_sweep_boundary(tmp_path, monkeypatch, display_only):
    from camera_designer import marble_live_worker as worker
    gl = importlib.import_module("OpenGL.GL")
    monkeypatch.setattr(gl, "glFinish", lambda: None, raising=False)
    clock = [0.]
    def monotonic():
        clock[0] += .3
        return clock[0]
    monkeypatch.setattr(worker.time, "monotonic", monotonic)
    camera = dict(eye_m=[0, 0, .07], target_m=[0, 0, .18], fov_degrees=18,
                  aperture_radius_m=.002, focus_distance_m=.11)
    new_camera = camera if display_only else {**camera, "eye_m": [.3, .2, .4]}
    live = dict(history_samples=2, aspect=1, lease_seconds=30, checkpoint=False,
                ray_batch_size=128, display={"tint": 0})
    events = []; field = {"_pump_seed": 0, "pose": camera["eye_m"]}
    class Accumulator:
        _w = 2; _h = 83; _rows_per_frame = 16; _row_off = 0; _pass = 0; _history_bytes = 1
        ticks = 0; pose = camera["eye_m"]
        def set_history(self, *a, **kw): pass
        def set_lens(self, *a, **kw): pass
        def set_camera(self, eye, *a, **kw):
            assert field["pose"] == eye
            self.pose = eye
            events.append(("camera", self._row_off))
        def clear(self):
            events.append(("reset", self._row_off))
            self._pass = self._row_off = 0
        def tick(self, strips):
            assert self.pose == field["pose"]
            events.append(("tick", self._row_off, tuple(self.pose)))
            self._row_off += min(16, self._h - self._row_off)
            if self._row_off == self._h: self._row_off = 0; self._pass += 1
            self.ticks += 1
            if self.ticks == 4:
                (tmp_path / "live-control.json").write_text(json.dumps(dict(
                    camera=new_camera, live={**live, "display": {"tint": 1}}, revision=1,
                    expires_at=worker.time.time() + 30)))
    def update(next_camera, acc):
        assert acc._row_off == 0 and acc.ticks == 6
        field["pose"] = next_camera["eye_m"]
        events.append(("geometry", acc._row_off))
    renderer = SimpleNamespace(_clear_prebuilt=lambda f: None,
        _gpu_pump_prebuilt=lambda f, n_rays: events.append(("pump", tuple(f["pose"]))))
    status = worker.run_live_sensor(Accumulator(), renderer, field, tmp_path, "test",
        {"live": live, "camera": camera}, 0, lambda *a, **kw: {}, lambda *a: None,
        max_epochs=3, update_camera_geometry=update)
    assert status["live_revision"] == 1
    if display_only:
        assert not any(e[0] in ("geometry", "camera", "reset") for e in events)
    else:
        i = next(i for i, e in enumerate(events) if e[0] == "geometry")
        assert [e[0] for e in events[i:i + 5]] == ["geometry", "camera", "reset", "pump", "tick"]
        assert len([e for e in events[:i] if e[0] == "tick"]) == 6


@pytest.mark.parametrize("frames", [1, 8])
def test_real_worker_orbits_move_geometry_before_each_sensor_and_keep_still_unchanged(tmp_path, monkeypatch, frames):
    from camera_designer import marble_preview_worker as worker
    from camera_designer import studio_photo_geometry as geometry
    gl = importlib.import_module("OpenGL.GL")
    monkeypatch.setattr(gl, "glFinish", lambda: None, raising=False)
    config = {"transport": "fixed:1", "max_bounces": 1, "orbit_frames": frames,
              "camera": {"eye_m": [.25, .125, .5], "target_m": [0, 0, .25], "resolution": 128}}
    payload = tmp_path / "payload.npz"
    np.savez(payload, vertices=np.zeros((1, 3, 3)), material_ids=np.zeros(1), pbr=np.zeros((1, 16)))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    events = []; state = {}
    def prepare(cfg, *unused):
        eye = np.asarray(cfg["camera"]["eye_m"], np.float32)
        events.append(("prepare", eye.copy()))
        packed = np.zeros((1, 16), np.float32)
        scene = (packed, packed, np.zeros((1, 12)), np.zeros((1, 3, 3)),
                 np.zeros((0, 8)), {}, (eye - .01, eye + .01), np.zeros((1, 12)))
        return scene, np.zeros((0, 8)), {"camera_geometry": cfg["camera"]}, [0, 0, .3]
    monkeypatch.setattr(geometry, "prepare_photo_geometry", prepare)
    def create(*args, **kw):
        state.update(ssbo=list(range(13)), tex_bands=[], bmin=args[7][0], bmax=args[7][1],
                     dims=(96, 96, 96), _pump_seed=0)
        return state
    def update(f, *args):
        f.update(bmin=args[6][0], bmax=args[6][1])
        events.append(("upload",))
    def clear(f):
        events.append(("clear",))
        f["_pump_seed"] = 0
    class Sensor:
        _active = True
        def __init__(self, **kw):
            np.testing.assert_allclose(kw["camera_eye"], state["bmin"] + [.03, .03, .05], atol=6e-8)
            events.append(("sensor", np.asarray(kw["camera_eye"]).copy()))
        def tick(self, **kw): events.append(("tick",))
        def destroy(self): events.append(("destroy_sensor",))
    renderer = SimpleNamespace(_ensure_worker_gl_context=lambda: None,
        _build_gpu_bvh=lambda tri: (np.zeros((1, 12)), np.zeros(1, np.uint32)),
        _gpu_ray_field_prebuilt=create, _update_prebuilt_geometry=update,
        _clear_prebuilt=clear, SensorAccumulator=Sensor,
        _gpu_pump_prebuilt=lambda *a, **kw: events.append(("pump",)),
        _destroy_prebuilt=lambda *a: events.append(("destroy_field",)))
    monkeypatch.setitem(sys.modules, "demo_pluck_gl", renderer)
    monkeypatch.setattr(worker, "_snapshot", lambda *a, **kw: {
        "state": "complete", "frames": list(kw["completed_frames"]) + [{"image": "test.png"}]})
    monkeypatch.setattr(worker, "_write_orbit_avi", lambda *a, **kw: None)
    status = worker.run(payload, config_path, tmp_path / "output", seconds=30,
        max_epochs=1, avi_writer_root=tmp_path)
    assert len(status["frames"]) == frames
    assert len([e for e in events if e[0] == "prepare"]) == frames
    assert len([e for e in events if e[0] == "upload"]) == frames - 1
    assert len([e for e in events if e[0] == "clear"]) == frames - 1
    for i, event in enumerate(events):
        if event[0] == "upload":
            assert [e[0] for e in events[i:i + 5]] == ["upload", "clear", "sensor", "pump", "tick"]
    effective = json.loads((tmp_path / "output/effective_config.json").read_text())
    assert effective["camera_geometry_version"] == geometry.CAMERA_GEOMETRY_VERSION
