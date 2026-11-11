"""CPU contracts executing production tile coordinate/seed expressions, no GL context."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = (Path(__file__).resolve().parents[1] / 'demo_pluck_gl.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
SHADER = next(n.value.value for n in TREE.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == '_GPU_SENSOR_CS' for t in n.targets)
              and isinstance(n.value, ast.Constant))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'SensorAccumulator')


def extracted(name):
    nodes = list(TREE.body) + list(CLASS.body)
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<sensor-tile>', 'exec'), ns)
    return ns[name]


coordinates = extracted('_sensor_tile_coordinates')
seed = extracted('_sensor_sample_seed')


def v2(x):
    return SimpleNamespace(x=x[0], y=x[1])


def sample(local, origin, full, tile_size, *, sample_index=0, pass_index=0, row=0, frame=0):
    # Evaluate the actual production GLSL seed and NDC expressions, with unsigned
    # wrapping and a deterministic RNG stand-in. Identical pre-hash inputs imply
    # identical entire transport streams (including aperture/bounce samples).
    sensor_px = tuple(a + b for a, b in zip(local, origin))
    ns = dict(sensor_px=v2(sensor_px), full_size=v2(full), s=sample_index,
              uSeed=seed(SimpleNamespace(_tile_sampling=True, _sprinkle=False,
                                        _pass=pass_index, _row_off=row, _frame=frame)),
              uint=lambda x: int(x) & 0xffffffff, float=float, W=float(full[0]), H=float(full[1]))
    expression = re.search(r'uint rng = hash_u\((.*?)\);', SHADER, re.S).group(1)
    expression = expression.replace('(tiled_sampling ? 0u : linear_id * 9176u)', '0u')
    expression = re.sub(r'(\d+)u\b', r'\1', expression)
    initial = eval(' '.join(expression.split()), {'__builtins__': {}}, ns) & 0xffffffff
    ns['rng'] = initial
    ns['rand01'] = lambda rng: (rng & 0xffffff) / 16777215.0
    uv = []
    for name in ('pu', 'pv'):
        expr = re.search(r'float ' + name + r'\s*=\s*([^;]+);', SHADER).group(1)
        uv.append(eval(expr, {'__builtins__': {}}, ns))
    aspect = full[0] / full[1]
    return initial, tuple(uv), ((uv[0] - .5) * 2 * aspect, (.5 - uv[1]) * 2)


@pytest.mark.parametrize('full,tile', [((37, 29), (16, 11)), ((11, 37), (4, 16)), ((1, 1), (1, 1))])
def test_disjoint_tiles_and_partial_edges_match_monolithic(full, tile):
    seen = set()
    for oy in range(0, full[1], tile[1]):
        for ox in range(0, full[0], tile[0]):
            size = min(tile[0], full[0] - ox), min(tile[1], full[1] - oy)
            assert coordinates(*size, full, (ox, oy)) == (size, full, (ox, oy))
            for y in range(size[1]):
                for x in range(size[0]):
                    global_px = x + ox, y + oy
                    assert global_px not in seen
                    seen.add(global_px)
                    for pass_index in (0, 3):
                        for sample_index in (0, 2):
                            kwargs = dict(pass_index=pass_index, sample_index=sample_index)
                            assert sample((x, y), (ox, oy), full, size, row=y, frame=100, **kwargs) == sample(
                                global_px, (0, 0), full, full, row=global_px[1], frame=1, **kwargs)
    assert len(seen) == full[0] * full[1]


@pytest.mark.parametrize('args', [
    (0, 2, None, (0, 0)), (2.5, 2, None, (0, 0)), (True, 2, None, (0, 0)),
    (2, 2, (5, 5), (-1, 0)), (2, 2, (5, 5), (4, 0)),
    (2, 2, (5, 5), (0, 4)), (2, 2, (0, 5), (0, 0)),
    (2, 2, (5,), (0, 0)), (2, 2, (5, 5), (0.5, 0)),
    (2, 2, None, (1, 0)), (2, 2, (2**31, 5), (0, 0)),
])
def test_invalid_tiles_fail_before_gpu_allocation(args):
    with pytest.raises(ValueError):
        coordinates(*args)
    init = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assert isinstance(init.body[0], ast.Assign)
    assert init.body[0].value.func.id == '_sensor_tile_coordinates'


def test_legacy_defaults_and_seed_unchanged():
    assert coordinates(31, 19, None, (0, 0)) == ((31, 19), (31, 19), (0, 0))
    obj = SimpleNamespace(_tile_sampling=False, _sprinkle=True, _pass=7, _row_off=16, _frame=31)
    assert seed(obj) == 314159 + 16 * 1009 + 7 * 65537 + 31 * 1000003
    obj._tile_sampling = True
    with pytest.raises(ValueError, match='row dispatches'):
        seed(obj)


def test_global_rays_but_local_storage_and_history_contract():
    assert 'ivec2 sensor_px = px + (tiled_sampling ? uSensorOrigin : ivec2(0))' in SHADER
    assert 'sensor_store(i, px, layer_acc[i]' in SHADER
    assert 'sensor_history_value(i, px, lum)' in SHADER
    assert 'int px_idx = px.x + px.y * uSensorSize.x' in SHADER
    assert 'int sample_slot = (px.y-uRowOffset)*uSensorSize.x+px.x' in SHADER
    assert 'sin(float(sensor_px.x)*127.1 + float(sensor_px.y)*311.7' in SHADER
    assert 'float(self._full_sensor_size[0]) / float(self._full_sensor_size[1])' in SOURCE
    assert "_u2i(b'uFullSensorSize',  0, 0)" in SOURCE


@pytest.mark.parametrize('origin,size', [((0, 0), (1024, 1024)), ((31744, 0), (1024, 1024)),
    ((0, 16384), (1024, 896)), ((31744, 16384), (1024, 896)), ((12345, 6789), (703, 511))])
def test_extreme_resolution_corners_and_arbitrary_roi(origin, size):
    full = (32768, 17280)
    coordinates(*size, full, origin)
    for local in ((0, 0), (size[0] - 1, size[1] - 1), (size[0] // 2, size[1] // 2)):
        global_px = tuple(a + b for a, b in zip(local, origin))
        assert sample(local, origin, full, size, pass_index=11) == sample(
            global_px, (0, 0), full, full, pass_index=11)
