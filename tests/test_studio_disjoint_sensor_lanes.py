"""CPU checks for N independent forward origins and sensor owners per slice."""
import ast
from pathlib import Path

import numpy as np
import pytest

from camera_designer import marble_preview_worker as worker


CAMERA = dict(resolution=128, eye_m=[0, 0, 0], target_m=[0, 0, 1],
              fov_degrees=20, aperture_radius_m=.002, focus_distance_m=1,
              aperture_blades=0, aperture_rotation_radians=0)


def test_sparse_calibration_regions_reduce_existing_linear_snapshot_per_lane():
    image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    spec = {'version': 1, 'width': 4, 'height': 4,
            'regions': {'gray18': [[4, 2], [8, 2]], 'red': [[0, 2]],
                        'blue': [[15, 1]]}}
    result = worker._sparse_sensor_regions(image, spec)
    expected = image.reshape(-1, 3)[[4, 5, 8, 9]].mean(axis=0)
    assert result['regions']['gray18']['mean'] == expected.tolist()
    assert result['regions']['gray18']['pixels'] == 4
    assert result['regions']['red']['mean'] is None
    assert result['regions']['blue']['usable'] is False
    mono = worker._sparse_sensor_regions(image[:, :, 0], spec)
    assert mono['regions']['gray18']['mean'] == [float(expected[0])]


def test_n_sensor_lanes_validate_cameras_and_disjoint_forward_origins():
    cameras = [{**CAMERA, 'eye_m': [i, 0, 0],
                **({'sensor_transport': 'fixed:1'} if i == 1 else {})}
               for i in range(4)]
    assert worker._sensor_lane_cameras({'camera': cameras[0],
                                        'sensor_lanes': cameras}) == cameras
    with pytest.raises(ValueError, match='main camera'):
        worker._sensor_lane_cameras({'camera': cameras[1], 'sensor_lanes': cameras})
    scenes = []
    for i in range(4):
        roots = np.zeros((2, 12), np.float32)
        roots[:, 0] = [2*i, 2*i+1]
        roots[:, 8] = 5e14
        scenes.append((None,) * 7 + (roots,))
    worker._require_disjoint_lane_roots(scenes)
    scenes[3][7][0, 0] = 1
    with pytest.raises(ValueError, match='reuses a forward ray origin'):
        worker._require_disjoint_lane_roots(scenes)


def test_n_lanes_dispatch_and_commit_only_after_all_owners_finish(monkeypatch):
    from camera_designer import studio_native_emitters
    events = []
    class Sensor:
        _pass = 0
        _w = 4
        _h = 3
        def __init__(self, index):
            self.index = index
        def set_exposure_interval_weight(self, weight):
            events.append(('weight', self.index, weight))
        def tick(self, *, strips):
            assert strips == 1
            events.append(('sensor', self.index))
            self._pass += 1
    class Renderer:
        def _clear_prebuilt(self, state):
            events.append(('clear', state['index']))
            state['_pump_seed'] = 0
        def _gpu_pump_prebuilt(self, state, *, n_rays):
            events.append(('forward', state['index'], n_rays))
    class Barrier:
        def record_emission_work_dispatched(self, index, count):
            events.append(('emission', count))
        def confirm_transport_products_materialized(self, index):
            events.append(('materialized',))
        def sensor_may_submit(self, index):
            return ('materialized',) in events
        def record_sensor_probe_work_dispatched(self, index, count):
            events.append(('probes', count))
        def confirm_reception_closed(self, index):
            events.append(('closed',))
        def confirm_detector_integration_committed(self, index):
            events.append(('committed',))
    def install(state, sensor, baseline, current):
        events.append(('install', state['index']))
        return np.zeros((state['index'] + 1, 12), np.float32)
    monkeypatch.setattr(studio_native_emitters, 'install_timed_light_frame', install)
    lanes = [dict(index=i, camera={'id': i}, scene=None,
                  state={'index': i, '_pump_seed': i + 10}, accumulator=Sensor(i))
             for i in range(4)]
    interval = {'t0': 1., 't1': 3., 'sensor_integrating': True}
    rays, sites = worker._dispatch_timed_sensor_lanes(
        lanes, interval, 0, prepare_scene=lambda camera: camera,
        renderer=Renderer(), barrier=Barrier(), sweeps=2,
        finish=lambda: events.append(('finish',)))
    assert (rays, sites) == (10, 96)
    assert [lane['state']['_pump_seed'] for lane in lanes] == [10, 11, 12, 13]
    assert [lane['accumulator']._pass for lane in lanes] == [2] * 4
    assert max(i for i, e in enumerate(events) if e[0] == 'forward') < events.index(('materialized',))
    assert max(i for i, e in enumerate(events) if e[0] == 'sensor') < events.index(('closed',))
    assert events[-1] == ('committed',)


def test_sensor_seed_namespace_keeps_single_sensor_and_uniquely_packs_n_lanes():
    source = (Path(__file__).resolve().parents[1] / 'demo_pluck_gl.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'SensorAccumulator')
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == '_sensor_sample_seed')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'sensor.py', 'exec'), namespace)
    sensor = type('Sensor', (), dict(_tile_sampling=False, _row_off=0,
                                  _pass=0, _frame=0, _sprinkle=False,
                                  _h=64, _rows_per_frame=16))()
    assert namespace[method.name](sensor) == 314159
    seeds=set()
    sensor._sensor_lane_count=20
    for sample_pass in range(10):
        for lane in range(20):
            for strip in range(4):
                sensor._pass=sample_pass;sensor._sensor_lane_index=lane
                sensor._row_off=strip*16
                seeds.add(namespace[method.name](sensor))
    assert len(seeds)==10*20*4


def test_native_seeded_populations_have_disjoint_roots_and_full_energy():
    pytest.importorskip('_spectral_kernels')
    from camera_designer.studio_native_emitters import native_emissive_sources
    from tests.test_studio_native_emitters import fixture
    base = fixture()
    scenes = []
    for seed in range(42, 46):
        roots = native_emissive_sources(base, [.45, .55, .65],
                                        ray_budget=120, seed=seed)
        scenes.append((*base, None, None, None, roots))
        assert roots[:, 10].sum() == pytest.approx(30, rel=3e-6)
    worker._require_disjoint_lane_roots(scenes)
    mono = native_emissive_sources(base, [.5875618], ray_budget=120, seed=46)
    assert len(mono) > 0
    worker._require_disjoint_lane_roots([*scenes, (*base, None, None, None, mono)])


def test_camera_visibility_drops_authored_housing_but_keeps_analytical_sensor():
    from camera_designer.camera_preset import thin_lens_housing_preset
    from camera_designer.camera_frame import thin_lens_housing_frame
    from camera_designer.scene_builder import build_gpu_scene
    camera = dict(eye_m=[0, 0, .07], target_m=[0, 0, .18],
                  aperture_radius_m=.002)
    preset = thin_lens_housing_preset(camera)
    frame = thin_lens_housing_frame(preset, camera['eye_m'], camera['target_m'])
    visible = build_gpu_scene(preset, [], scene_object=None,
        wavelengths_um=[.45, .55, .65], camera_frame=frame)
    hidden = build_gpu_scene(preset, [], scene_object=None,
        wavelengths_um=[.45, .55, .65], camera_frame=frame,
        include_camera_geometry=False)
    assert len(visible[3]) > 10000
    assert len(hidden[3]) < 100
    # The camera itself is still a resolved analytical pose; this flag only
    # controls authored scene geometry supplied to forward and sensor BVHs.
    assert camera['eye_m'] == [0, 0, .07]
