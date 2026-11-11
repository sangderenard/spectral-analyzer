"""The existing worker consumes admitted sequences without another renderer."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_designer import marble_preview_worker as worker

BASE = {"resolution": 128, "eye_m": [0, 0, .07], "target_m": [0, 0, .18],
        "fov_degrees": 36., "aperture_radius_m": .002, "focus_distance_m": .11,
        "aperture_blades": 0, "aperture_rotation_radians": 0.}
POSES = [dict(BASE, eye_m=[.02, 0, .07]), dict(BASE, eye_m=[0, .01, .05], target_m=[.01, 0, .17],
         fov_degrees=48, focus_distance_m=.12, aperture_radius_m=.003, aperture_blades=5, aperture_rotation_radians=.2)]


def test_resolved_pose_order_and_optics_are_exact():
    config = {"camera": BASE, "camera_poses": POSES}
    sequence = worker._camera_pose_sequence(config)
    for index, expected in enumerate(POSES):
        assert worker._camera_for_frame(BASE, index, len(sequence), sequence) == expected
    assert worker._camera_pose_sequence({"camera": BASE}) is None
    orbit = worker._camera_for_frame(BASE, 4, 8)
    np.testing.assert_allclose(orbit["eye_m"], [0, 0, .29], atol=1e-7)
    assert orbit["fov_degrees"] == BASE["fov_degrees"]


@pytest.mark.parametrize("poses", [None, [], [{}] * 33, [{}], [{**BASE, "resolution": 256}],
    [{**BASE, "projection": "fisheye"}], [{**BASE, "eye_m": [0, 0, .18]}],
    [{**BASE, "fov_degrees": float("nan")}], [{**BASE, "target_m": [0, 0]}]])
def test_bad_resolved_sequences_fail_before_gpu(poses):
    with pytest.raises(ValueError):
        worker._camera_pose_sequence({"camera": BASE, "camera_poses": poses})


@pytest.mark.parametrize("options", [{"orbit_frames": 8}, {"live": {}}, {"client_intersections": True},
    {"transport": "fixed:3", "oversampling": 2}])
def test_sequence_is_not_an_orbit_tile_or_live_mode(options):
    with pytest.raises(ValueError):
        worker._camera_pose_sequence({"camera": BASE, "camera_poses": POSES, **options})


def test_actual_sensor_uses_each_resolved_pose_and_existing_geometry():
    tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    sensor = copy.deepcopy(next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == "sensor"))
    calls, geometry = [], []
    def accumulator(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(_active=True)
    ns = dict(np=np, json=json, width=128, height=128, camera=BASE, camera_poses=POSES,
        total_frames=2, _camera_for_frame=worker._camera_for_frame, update_camera_geometry=geometry.append,
        _sensor_device_limits=lambda *a: {}, renderer=SimpleNamespace(SensorAccumulator=accumulator),
        state={"ssbo": list(range(16)), "tex_bands": [], "bmin": [], "bmax": [], "dims": (), "native_pbr_ssbo": 20},
        triangles=[], nodes=[], cooperative=False, live=None, max_bounces=4, sources=[], sphere_regions=[],
        film=None, transport="fixed:1", geometry_stats={}, authored_bundle=None, bounce_diagnostic={})
    exec(compile(ast.Module(body=[sensor], type_ignores=[]), "<actual-worker-sensor>", "exec"), ns)
    sensors = [ns["sensor"](index) for index in range(2)]
    assert geometry == POSES
    for call, pose, result in zip(calls, POSES, sensors):
        np.testing.assert_allclose(call["camera_eye"], pose["eye_m"])
        np.testing.assert_allclose(call["camera_target"], pose["target_m"])
        assert call["fov_deg"] == pose["fov_degrees"]
        assert call["focus_dist"] == pose["focus_distance_m"]
        assert call["aperture_radius"] == pose["aperture_radius_m"]
        assert call["n_blades"] == pose["aperture_blades"]
        assert call["aperture_rot"] == pose["aperture_rotation_radians"]
        assert result._frame_camera == pose


def test_actual_snapshots_keep_each_frame_and_camera_receipt(tmp_path, monkeypatch):
    import OpenGL.GL as gl
    for name in ("glFinish", "glBindTexture"):
        monkeypatch.setattr(gl, name, lambda *args: None, raising=False)
    monkeypatch.setattr(gl, "glGetTexImage", lambda *args: np.ones((2, 2, 4), np.float32).tobytes(), raising=False)
    for name in ("GL_TEXTURE_2D", "GL_RGBA", "GL_FLOAT"):
        monkeypatch.setattr(gl, name, 0, raising=False)
    frames = []
    for index, pose in enumerate(POSES):
        accumulator = SimpleNamespace(_w=2, _h=2, tex=1, _frame_camera=pose)
        status = worker._snapshot(accumulator, tmp_path, "digest", index + 1, 1, 0, 1, 4,
            finished=True, frame_index=index, total_frames=2, completed_frames=tuple(frames))
        frames = status["frames"]
    assert status["total_frames"] == 2 and status["state"] == "encoding"
    assert [frame["camera"] for frame in frames] == POSES
    assert [frame["index"] for frame in frames] == [0, 1]
    assert all((tmp_path / frame["image"]).is_file() for frame in frames)


def test_encoding_failure_preserves_finished_receipts_in_main(tmp_path, monkeypatch):
    payload, config_path, output = tmp_path / "scene.npz", tmp_path / "config.json", tmp_path / "out"
    payload.write_bytes(b"scene")
    config = {"camera": BASE, "camera_poses": POSES}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes() + json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    frames = [{"index": i, "image": f"preview-{i:06d}.png", "camera": p} for i, p in enumerate(POSES)]
    def failed_run(*args, **kwargs):
        output.mkdir()
        for frame in frames:
            (output / frame["image"]).write_bytes(b"retained frame")
        worker._atomic_json(output / "status.json", {"digest": digest, "state": "encoding", "frames": frames, "total_frames": 2})
        raise RuntimeError("injected AVI failure")
    monkeypatch.setattr(worker, "run", failed_run)
    monkeypatch.setattr(worker.sys, "argv", ["worker", "--payload", str(payload), "--config", str(config_path), "--out", str(output)])
    with pytest.raises(RuntimeError, match="injected AVI failure"):
        worker.main()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error" and status["frames"] == frames and status["total_frames"] == 2
    assert all((output / frame["image"]).is_file() for frame in frames)

@pytest.mark.parametrize("mode", ["still", "orbit", "poses", "avi-failure"])
def test_real_worker_serial_frame_loop_uses_existing_owner(tmp_path, monkeypatch, mode):
    import sys
    from camera_designer import studio_photo_geometry as geometry
    import OpenGL.GL as gl
    monkeypatch.setattr(gl, "glFinish", lambda: None, raising=False)
    monkeypatch.setattr(worker, "_sensor_device_limits", lambda *args: {})
    config = {"transport": "fixed:1", "max_bounces": 1, "camera": BASE,
              "orbit_frames": 8 if mode == "orbit" else 1}
    if mode in ("poses", "avi-failure"):
        config["camera_poses"] = POSES
    count = 2 if "camera_poses" in config else config["orbit_frames"]
    expected = [worker._camera_for_frame(BASE, i, count, config.get("camera_poses")) for i in range(count)]
    payload, config_path = tmp_path / "scene.npz", tmp_path / "config.json"
    np.savez(payload, vertices=np.zeros((1, 3, 3)), material_ids=np.zeros(1), pbr=np.zeros((1, 16)))
    config_path.write_text(json.dumps(config), encoding="utf-8")
    events, sensors, prepared, encoded = [], [], [], []
    state = {}
    def prepare(cfg, *unused):
        prepared.append(cfg["camera"])
        eye = np.asarray(cfg["camera"]["eye_m"], np.float32)
        packed = np.zeros((1, 16), np.float32)
        scene = (packed, packed, np.zeros((1, 12)), np.zeros((1, 3, 3)),
                 np.zeros((0, 8)), {}, (eye - .01, eye + .01), np.zeros((1, 12)))
        return scene, np.zeros((0, 8)), {"camera_geometry": cfg["camera"]}, [0, 0, .3]
    monkeypatch.setattr(geometry, "prepare_photo_geometry", prepare)
    monkeypatch.setitem(sys.modules, "material_db", SimpleNamespace(MaterialDatabase=SimpleNamespace(
        instance=lambda: SimpleNamespace(build_tensors=lambda: {"pbr": np.zeros((1, 16))}))))
    def create(*args, **kwargs):
        state.update(ssbo=list(range(16)), native_pbr_ssbo=20, tex_bands=[],
                     bmin=args[7][0], bmax=args[7][1], dims=(96, 96, 96), _pump_seed=0)
        events.append("create_field")
        return state
    def update(field, *args, **kwargs):
        field.update(bmin=args[6][0], bmax=args[6][1])
        events.append("update_geometry")
    class Sensor:
        _active = True
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            sensors.append(self)
            np.testing.assert_allclose(kwargs["camera_eye"], state["bmin"] + [.03, .03, .05], atol=6e-8)
        def tick(self, **kwargs):
            events.append("tick")
        def destroy(self):
            events.append("destroy_sensor")
    renderer = SimpleNamespace(_ensure_worker_gl_context=lambda: None,
        _build_gpu_bvh=lambda tri: (np.zeros((1, 12)), np.zeros(1, np.uint32)),
        _gpu_ray_field_prebuilt=create, _update_prebuilt_geometry=update,
        _clear_prebuilt=lambda *args: events.append("clear"), SensorAccumulator=Sensor,
        _gpu_pump_prebuilt=lambda *a, **kw: events.append("pump"),
        _destroy_prebuilt=lambda *args: events.append("destroy_field"))
    monkeypatch.setitem(sys.modules, "demo_pluck_gl", renderer)
    def snapshot(accumulator, output, digest, epoch, *args, **kwargs):
        frame = {"index": kwargs["frame_index"], "image": f"preview-{epoch:06d}.png", "camera": accumulator._frame_camera}
        status = {"state": "encoding", "frames": [*kwargs["completed_frames"], frame], "total_frames": count}
        worker._atomic_json(output / "status.json", status)
        return status
    monkeypatch.setattr(worker, "_snapshot", snapshot)
    def encode(output, frames, resolution, writer_root, **kwargs):
        encoded.extend(frames)
        if mode == "avi-failure":
            raise RuntimeError("injected AVI failure")
    monkeypatch.setattr(worker, "_write_orbit_avi", encode)
    if mode == "avi-failure":
        with pytest.raises(RuntimeError, match="injected AVI failure"):
            worker.run(payload, config_path, tmp_path / "out", seconds=30, max_epochs=1, avi_writer_root=tmp_path)
    else:
        status = worker.run(payload, config_path, tmp_path / "out", seconds=30, max_epochs=1, avi_writer_root=tmp_path)
        assert status["total_frames"] == count and len(status["frames"]) == count
    assert len(sensors) == count and events.count("create_field") == 1
    assert events.count("destroy_sensor") == count and events[-1] == "destroy_field"
    for sensor, pose in zip(sensors, expected):
        assert sensor._frame_camera == pose
        np.testing.assert_allclose(sensor.kwargs["camera_eye"], pose["eye_m"])
        assert sensor.kwargs["fov_deg"] == pose["fov_degrees"]
    assert len(encoded) == (count if count > 1 else 0)
    if mode in ("poses", "avi-failure"):
        assert prepared[0] == POSES[0]
        assert events.count("update_geometry") == 1 and events.count("clear") == 1
