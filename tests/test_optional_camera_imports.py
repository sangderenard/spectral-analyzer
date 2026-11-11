"""Real analytical service imports and geometry packing without optional Torch.

Can run directly under the service interpreter; no pytest, GL context, or render.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("Torch deliberately unavailable", name='torch')
sys.meta_path.insert(0, NoTorch())
import camera_software
assert 'camera_software.sensor_priority_network' not in sys.modules
assert 'camera_software.raytraced_priority_training' not in sys.modules
assert 'SensorWorkValueNet' in dir(camera_software)
try:
    camera_software.not_a_camera_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export must raise AttributeError')
for name in ('SensorWorkValueNet', 'train_from_raytraced_examples'):
    try:
        getattr(camera_software, name)
    except ModuleNotFoundError as exc:
        assert exc.name == 'torch'
    else:
        raise AssertionError('optional ML exports must still require real Torch')
import camera_designer
try:
    camera_designer.NeuralAssemblyMLP()
except ImportError as exc:
    assert 'torch is required' in str(exc)
else:
    raise AssertionError('ML construction must explicitly require Torch')
import camera_designer.marble_preview_worker
import demo_pluck_gl
from camera_designer.studio_photo_geometry import prepare_photo_geometry
import numpy as np
vertices = np.array([[[0,0,0], [.1,0,0], [0,.1,0]]], np.float32)
ids = np.zeros(1, np.int32)
pbr = np.zeros((1,16), np.float32)
pbr[:,:3] = .8
pbr[:,6] = 1.5
pbr[:,7] = 1
scene, spheres, stats, _ = prepare_photo_geometry(
    {'transport':'fixed:3', 'scale_m':.01, 'camera': {
        'eye_m':[.2,.1,.4], 'target_m':[0,0,.18], 'aperture_radius_m':.002}},
    vertices, ids, pbr)
assert len(scene[3]) > 1
np.testing.assert_allclose(scene[3][-1], vertices[0] * .01 + [0,0,.18])
assert stats['camera_geometry']['body_dimensions_m'] == [.139,.101,.077]
assert 'torch' not in sys.modules
print('TORCH_FREE_SERVICE_GEOMETRY_OK')
'''


def test_torch_free_service_imports_and_geometry():
    result = subprocess.run([sys.executable, '-c', PROBE], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TORCH_FREE_SERVICE_GEOMETRY_OK' in result.stdout


if __name__ == '__main__':
    test_torch_free_service_imports_and_geometry()
    print('TORCH_FREE_SERVICE_GEOMETRY_OK')
