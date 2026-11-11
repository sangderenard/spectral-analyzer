"""Exercise the real worker queue/finally control flow without importing renderer or GL."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_designer.marble_tiles import TileAssembly, tile_plan, tile_windows

ROOT = Path(__file__).resolve().parents[1]


def queue_body():
    tree = ast.parse((ROOT / 'camera_designer/marble_preview_worker.py').read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    queue = copy.deepcopy(next(n for n in run.body if isinstance(n, ast.Try)))
    class Rewriter(ast.NodeTransformer):
        def visit_Return(self, node):
            return ast.copy_location(ast.Assign(targets=[ast.Name(id='result', ctx=ast.Store())], value=node.value), node)
    return compile(ast.fix_missing_locations(Rewriter().visit(ast.Module(body=[queue], type_ignores=[]))), '<actual-worker-queue>', 'exec')


@pytest.mark.parametrize('fail_tile', [None, 1])
def test_actual_tile_queue_lifecycle(tmp_path, monkeypatch, fail_tile):
    import OpenGL.GL as gl
    monkeypatch.setattr(gl, 'glFinish', lambda: None, raising=False)
    plan = tile_plan({'transport': 'fixed:3', 'camera': {'width': 129, 'height': 130},
                      'tiled': {'tile_size': 128}})
    events, statuses, accumulators = [], [], []
    class Accumulator:
        def __init__(self, window):
            self.window = window
            self._w, self._h = window['width'], window['height']
            self._rows_per_frame = 16
            self._row_off = self._pass = 0
            self._samples_per_pixel = 1
        def tick(self, strips):
            assert strips == 1
            self._row_off += min(16, self._h - self._row_off)
            if self._row_off == self._h:
                self._pass += 1
                self._row_off = 0
        def destroy(self):
            events.append(('destroy', self.window))
    def sensor(frame_index, window):
        assert frame_index == 0
        a = Accumulator(window)
        accumulators.append(a)
        return a
    def snapshot(a, output, digest, epoch, *args, finished, **kwargs):
        if fail_tile is not None and len(accumulators) - 1 == fail_tile:
            raise RuntimeError('injected snapshot failure')
        filename = f'exposure-{epoch:06d}.npz'
        if finished:
            assert a._row_off == 0 and a._pass >= 1
            np.savez(output / filename, linear_rgb_exposure=np.ones((a._h, a._w, 3), np.float32) * a._pass)
        return {'state': 'complete' if finished else 'exposing', 'image': 'tile.png',
                'hdr': filename, 'frames': [{'index': 0, 'image': 'tile.png', 'hdr': filename}] if finished else []}
    clock = iter(range(100000))
    assembly = TileAssembly(tmp_path, plan)
    ns = dict(live=None, time=SimpleNamespace(monotonic=lambda: next(clock)), tile_windows=tile_windows,
              tiles=plan, total_frames=1, output=tmp_path, preparation_status={}, sensor=sensor,
              snapshot_seconds=.25, renderer=SimpleNamespace(
                  _gpu_pump_prebuilt=lambda *a, **k: events.append(('pump',)),
                  _destroy_prebuilt=lambda state: events.append(('destroy_field',))),
              state={}, cooperative_pump=None, epoch=0, seconds=1., max_epochs=1,
              _snapshot=snapshot, digest='x', preparation=0., completed_frames=[], gamma=1.,
              film_response=None, transport='fixed:3', assembly=assembly, accumulator=None,
              geometry_stats={}, json=json, _atomic_json=lambda path, value: statuses.append(copy.deepcopy(value)))
    if fail_tile is not None:
        with pytest.raises(RuntimeError, match='injected snapshot failure'):
            exec(queue_body(), ns)
        assert len(accumulators) == 2
    else:
        exec(queue_body(), ns)
        assert [a.window for a in accumulators] == list(tile_windows(plan))
        assert ns['result']['state'] == 'complete'
        assert ns['result']['tiled']['completed_tiles'] == 4
        assert not list(tmp_path.glob('tile-*'))
        with np.load(tmp_path / ns['result']['hdr']) as data:
            np.testing.assert_array_equal(data['linear_rgb_exposure'], np.ones((130, 129, 3)))
        assert all(s.get('image') is None for s in statuses if s.get('state') == 'exposing')
    assert len([e for e in events if e[0] == 'destroy']) == len(accumulators)
    assert events[-1] == ('destroy_field',)
    assert assembly.raw is None
