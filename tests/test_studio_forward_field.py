import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from camera_designer.studio_forward_field import (
    ForwardFieldPublisher, field_to_linear_srgb, validate_receive_light_field,
)


def test_explicit_fixed3_only():
    assert not validate_receive_light_field({})
    assert validate_receive_light_field(dict(receive_light_field=True, transport='fixed:3'))
    for config in [dict(receive_light_field=1), dict(receive_light_field='true'),
                   dict(receive_light_field=True), dict(receive_light_field=True, transport='fixed:1')]:
        with pytest.raises(ValueError):
            validate_receive_light_field(config)


def test_conversion_reuses_line_profile_without_sensor_response():
    from camera_designer.marble_color_profile import sensor_to_linear_srgb, response_matrix
    lines = np.array([[[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]]])
    raw = np.moveaxis(lines * 65535, -1, 0)
    actual = field_to_linear_srgb(raw, (4, 1, 1))
    for response in [None, dict(version=1, centers_nm=[630,540,470], sigmas_oct=[.24,.21,.18])]:
        expected, _ = sensor_to_linear_srgb(lines @ response_matrix(response).T, response)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual[0,0,3], 1, atol=1e-6)
    assert actual.min() < 0  # no hidden clipping or tone mapping
    assert actual.dtype == np.dtype('<f4') and actual.flags.c_contiguous
    np.testing.assert_allclose(field_to_linear_srgb(raw*2, (4,1,1)), 2*actual)


@pytest.mark.parametrize('bad', [np.full((3,1,1,1), np.nan), np.full((3,1,1,1), -1), np.zeros((1,1,1,1))])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):
        field_to_linear_srgb(bad, (1,1,1))


def fixture(monkeypatch):
    from OpenGL import GL
    monkeypatch.setattr(GL, 'glMemoryBarrier', lambda bits: None, raising=False)
    monkeypatch.setattr(GL, 'GL_TEXTURE_UPDATE_BARRIER_BIT', 1, raising=False)
    counts = np.arange(72, dtype=np.float32).reshape(3,2,3,4)*65535
    calls = []
    def readback(textures, dims):
        calls.append((textures, dims))
        return counts
    renderer = SimpleNamespace(_gpu_readback_prebuilt=readback)
    field = dict(tex_bands=[1,2,3], dims=(4,3,2), bmin=[-1,-2,-3], bmax=[1,2,3])
    return renderer, field, counts, calls


def test_immutable_layout_bounds_rate_and_revision(tmp_path, monkeypatch):
    renderer, field, counts, calls = fixture(monkeypatch)
    publisher = ForwardFieldPublisher(tmp_path, 'a'*64)
    first = publisher.snapshot(renderer, field, now=0, live_revision=2, sweep=7, ray_jobs=128)
    content = (tmp_path/first['name']).read_bytes()
    assert re.fullmatch(r'light-field-[0-9a-f]{32}\.bin', first['name'])
    assert first['byte_length'] == 4*3*2*3*4 == len(content)
    assert first['sha256'] == hashlib.sha256(content).hexdigest()
    assert first['bounds_min_m'] == [-1,-2,-3] and first['dimensions'] == [4,3,2]
    np.testing.assert_array_equal(np.frombuffer(content,'<f4').reshape(2,3,4,3), field_to_linear_srgb(counts,(4,3,2)))
    counts *= 2
    assert publisher.snapshot(renderer, field, now=.5) is first and len(calls) == 1
    publisher.invalidate()
    assert publisher.snapshot(renderer, field, now=.9, live_revision=3) is None
    second = publisher.snapshot(renderer, field, now=1, live_revision=3, sweep=8)
    assert second['sequence'] == 2 and second['live_revision'] == 3
    assert first['live_revision'] == 2 and (tmp_path/first['name']).read_bytes() == content


def test_retention_only_deletes_owned_artifacts(tmp_path, monkeypatch):
    renderer, field, _, calls = fixture(monkeypatch)
    unowned = tmp_path/'light-field-unrelated.bin'
    unowned.write_bytes(b'not owned')
    publisher = ForwardFieldPublisher(tmp_path, 'd')
    for n in range(9):
        publisher.snapshot(renderer, field, now=n)
    assert len(calls) == 9 and len(publisher.files) == 4
    assert len(list(tmp_path.glob('light-field-*.bin'))) == 5
    assert unowned.read_bytes() == b'not owned'
    assert not list(tmp_path.glob('*.tmp'))


def test_locked_retention_backpressures_readback(tmp_path, monkeypatch):
    renderer, field, _, calls = fixture(monkeypatch)
    publisher = ForwardFieldPublisher(tmp_path, 'd')
    for n in range(4):
        publisher.snapshot(renderer, field, now=n)
    from pathlib import Path
    def locked(self, **kwargs):
        raise PermissionError('reader lock')
    monkeypatch.setattr(Path, 'unlink', locked)
    old = publisher.latest
    assert publisher.snapshot(renderer, field, now=4) is old
    assert len(calls) == len(publisher.files) == 4


def test_restart_reconciles_and_bounds_old_files(tmp_path, monkeypatch):
    import json
    import os
    renderer, field, _, calls = fixture(monkeypatch)
    first = ForwardFieldPublisher(tmp_path, 'd')
    for n in range(4):
        metadata = first.snapshot(renderer, field, now=n)
        os.utime(tmp_path / metadata['name'], (1700000000+n, 1700000000+n))
    newest = tmp_path / metadata['name']
    oldest = first.files[0]
    (tmp_path/'status.json').write_text(json.dumps({'light_field': {'name': oldest.name}}))
    expected = newest.read_bytes()
    second = ForwardFieldPublisher(tmp_path, 'd')
    assert list(second.files) == list(first.files)
    second.snapshot(renderer, field, now=0)
    assert len(list(tmp_path.glob('light-field-*.bin'))) == 4
    assert newest.read_bytes() == expected and oldest.exists()
    # On the next restart, only the newly advertised artifact is protected.
    (tmp_path/'status.json').write_text(json.dumps({'light_field': second.latest}))
    third = ForwardFieldPublisher(tmp_path, 'd')
    for n in range(6):
        third.snapshot(renderer, field, now=n)
        assert len(list(tmp_path.glob('light-field-*.bin'))) == 4


def test_restart_locked_old_file_prevents_growth(tmp_path, monkeypatch):
    renderer, field, _, calls = fixture(monkeypatch)
    first = ForwardFieldPublisher(tmp_path, 'd')
    for n in range(4):
        first.snapshot(renderer, field, now=n)
    second = ForwardFieldPublisher(tmp_path, 'd')
    from pathlib import Path
    real_unlink = Path.unlink
    locked_path = second.files[0]
    def unlink(path, **kwargs):
        if path == locked_path:
            raise PermissionError('reader lock')
        return real_unlink(path, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    for n in range(3):
        assert second.snapshot(renderer, field, now=n) is None
    assert len(calls) == 4 and len(list(tmp_path.glob('light-field-*.bin'))) == 4


def test_restart_cleans_only_exact_regular_incomplete_files(tmp_path, monkeypatch):
    renderer, field, _, _ = fixture(monkeypatch)
    interrupted = tmp_path/('light-field-'+'a'*32+'.tmp')
    interrupted.write_bytes(b'incomplete')
    unrelated = tmp_path/'light-field-unknown.tmp'
    unrelated.write_bytes(b'unrelated')
    directory = tmp_path/('light-field-'+'b'*32+'.bin')
    directory.mkdir()
    publisher = ForwardFieldPublisher(tmp_path, 'd')
    publisher.snapshot(renderer, field, now=0)
    assert not interrupted.exists() and unrelated.read_bytes() == b'unrelated'
    assert directory.is_dir() and len(publisher.files) == 1


def test_restart_does_not_follow_artifact_symlinks(tmp_path, monkeypatch):
    renderer, field, _, _ = fixture(monkeypatch)
    outside = tmp_path/'unrelated-source.bin'
    outside.write_bytes(b'do not touch')
    link = tmp_path/('light-field-'+'c'*32+'.bin')
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('OS does not permit file symlinks')
    publisher = ForwardFieldPublisher(tmp_path, 'd')
    assert not publisher.files
    for n in range(6):
        publisher.snapshot(renderer, field, now=n)
    assert link.is_symlink() and outside.read_bytes() == b'do not touch'


def test_restart_locked_incomplete_prevents_new_readback(tmp_path, monkeypatch):
    renderer, field, _, calls = fixture(monkeypatch)
    interrupted = tmp_path/('light-field-'+'d'*32+'.tmp')
    interrupted.write_bytes(b'incomplete')
    publisher = ForwardFieldPublisher(tmp_path, 'd')
    from pathlib import Path
    real_unlink = Path.unlink
    def unlink(path, **kwargs):
        if path == interrupted:
            raise PermissionError('reader lock')
        return real_unlink(path, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    assert publisher.snapshot(renderer, field, now=0) is None
    assert not calls and not publisher.files and interrupted.exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_real_live_loop_export_and_revision_boundary(tmp_path, monkeypatch, enabled):
    from tests.test_marble_live_ray_batch import run_loop
    from camera_designer import marble_live_worker as worker
    renderer, fixture_field, counts, calls = fixture(monkeypatch)
    original = worker.run_live_sensor
    def run(*args, **kwargs):
        accumulator, live_renderer, field, output, digest, config, *tail = args
        field.update(fixture_field, n_sources=1)
        live_renderer._gpu_readback_prebuilt = renderer._gpu_readback_prebuilt
        return original(accumulator, live_renderer, field, output, digest,
            {**config, 'transport': 'fixed:3', 'receive_light_field': enabled}, *tail, **kwargs)
    monkeypatch.setattr(worker, 'run_live_sensor', run)
    result, events, statuses = run_loop(tmp_path, monkeypatch, epochs=8,
        changes={4: (1, {'ray_batch_size': 16384})})
    assert [event for event in events if event[0] == 'reset'] == [('reset', 6, 0, 1)]
    if not enabled:
        assert not calls and not list(tmp_path.glob('light-field-*'))
        assert all('light_field' not in status for status in statuses)
    else:
        fields = [s['light_field'] for s in statuses if 'light_field' in s]
        assert fields and calls and result['light_field']['live_revision'] == 1
        for status in statuses:
            if 'light_field' in status:
                assert status['light_field']['live_revision'] == status['live_revision']
                assert status['light_field']['ray_jobs'] == status['ray_batch_size']
        assert len(list(tmp_path.glob('light-field-*.bin'))) <= 4


def test_directional_overflow_preserves_primary_scalar_export(tmp_path,monkeypatch):
    from camera_designer import studio_directional_field as directional
    renderer,field,counts,_=fixture(monkeypatch)
    field.update(directional_record={},source_buf=np.zeros((0,12),np.float32))
    def overflow(*args):raise directional.DirectionalOverflow('injected signed overflow')
    monkeypatch.setattr(directional,'bundle_sections',overflow)
    pub=ForwardFieldPublisher(tmp_path,'d')
    meta=pub.snapshot(renderer,field,now=0)
    assert meta['schema']=='studio.forward-field.v1' and 'directional' not in meta
    assert meta['directional_unavailable']['reason']=='signed-accumulator-overflow'
    np.testing.assert_array_equal(np.fromfile(tmp_path/meta['name'],'<f4').reshape(2,3,4,3),field_to_linear_srgb(counts,(4,3,2)))
    field.pop('directional_record')
    recovered=pub.snapshot(renderer,field,now=1)
    assert 'directional_unavailable' not in recovered
