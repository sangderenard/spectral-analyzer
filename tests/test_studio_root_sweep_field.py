import numpy as np
import pytest
from camera_designer.studio_directional_field import (ROOT_ESTIMATOR,LEGACY_ESTIMATOR,recording_options,
    directional_glsl,recording_shader,word_count,voxel_volume,root_multiplicities,reset_root_accounting,
    begin_root_dispatch,submitted_root_chunk,finish_root_dispatch,abort_root_dispatch,root_normalization,
    DirectionalUnavailable)


def record(n=3):
    r=dict(estimator=ROOT_ESTIMATOR,source_root_count=n)
    reset_root_accounting(r)
    return r


def pump(r,jobs,chunks=None):
    enabled=begin_root_dispatch(r,jobs)
    for chunk in chunks or [jobs]:submitted_root_chunk(r,chunk)
    finish_root_dispatch(r)
    return enabled


def test_uneven_actual_replays_are_corrected_per_root_not_global_ray_count():
    n=4464;jobs=8192;counts=root_multiplicities(n,jobs)
    np.testing.assert_array_equal(counts[:3728],2)
    np.testing.assert_array_equal(counts[3728:],1)
    roots=np.arange(jobs)%n
    corrected=np.bincount(roots,weights=1/counts[roots],minlength=n)
    np.testing.assert_array_equal(corrected,np.ones(n))
    assert not np.allclose(np.bincount(roots,minlength=n)*n/jobs,1)


def test_repeat_sweep_mean_invariance_and_uneven_budgets():
    r=record();root_energy=np.array([.125,.5,.75]);total=np.zeros(3)
    for jobs in [3,7,6,9]:
        counts=root_multiplicities(3,jobs)
        roots=np.arange(jobs)%3
        contribution=np.bincount(roots,weights=root_energy[roots]/counts[roots],minlength=3)
        total+=contribution
        assert pump(r,jobs)
        np.testing.assert_allclose(total/root_normalization(r),root_energy,rtol=1e-15)
    assert r['root_sampling']['recorded_root_jobs']==25
    np.testing.assert_array_equal(r['root_replay_counts'],[9,8,8])


def test_partial_coverage_is_excluded_and_never_false_full_scene():
    r=record(4464)
    assert not pump(r,128)
    with pytest.raises(DirectionalUnavailable,match='incomplete-root-coverage'):root_normalization(r)
    assert r['root_sampling']['excluded_partial_root_jobs']==128
    assert pump(r,8192,[512]*16)
    assert root_normalization(r)==1
    assert not pump(r,128)
    assert root_normalization(r)==1 and r['root_sampling']['excluded_partial_root_jobs']==256


def test_failed_partial_dispatch_requires_clear_and_resets_all_accounting():
    r=record();pump(r,3)
    assert begin_root_dispatch(r,7)
    submitted_root_chunk(r,3)
    with pytest.raises(DirectionalUnavailable,match='incomplete-root-dispatch'):root_normalization(r)
    abort_root_dispatch(r)
    assert not pump(r,3)
    with pytest.raises(DirectionalUnavailable,match='incomplete-root-dispatch'):root_normalization(r)
    reset_root_accounting(r)
    assert r['root_sampling']['recorded_root_jobs']==0 and not r['root_replay_counts'].any()
    assert pump(r,3) and root_normalization(r)==1


def test_incomplete_finish_and_mismatched_chunk_fail_closed():
    r=record();begin_root_dispatch(r,7);submitted_root_chunk(r,3);finish_root_dispatch(r)
    with pytest.raises(DirectionalUnavailable):root_normalization(r)
    reset_root_accounting(r);begin_root_dispatch(r,3)
    with pytest.raises(ValueError):submitted_root_chunk(r,4)
    with pytest.raises(DirectionalUnavailable):root_normalization(r)


@pytest.mark.parametrize('grid',[32,64])
def test_grid_counts_bounds_and_optional_extinction_anchor_unchanged(grid):
    shader=directional_glsl(12,grid,ROOT_ESTIMATOR)
    assert f'iteration<{3*grid+3}' in shader
    assert f'studio_directional[{grid**3*12*6}]' in shader
    assert f'/float({grid})' in shader
    assert 'studio_record_track(q,d,packet,next-cursor);' in shader
    assert word_count(12,grid)*4==grid**3*12*6*4
    assert voxel_volume(([0,0,0],[1,1,1]),grid)==1/grid**3


def test_explicit_opt_in_does_not_change_legacy_defaults():
    assert recording_options()==(LEGACY_ESTIMATOR,32)
    assert recording_options(ROOT_ESTIMATOR)==(ROOT_ESTIMATOR,64)
    with pytest.raises(ValueError):recording_options(LEGACY_ESTIMATOR,64)
    legacy=directional_glsl();assert 'studio_record_root_weight' not in legacy
    assert begin_root_dispatch(dict(estimator=LEGACY_ESTIMATOR),8192) is None


def test_shader_root_identity_and_inverse_replay_hook():
    source=('void splat_spectral(vec3 p, vec4 packet) {}\n'
        'if(currentMedium<0) splat_segment_multiscale(ro, hp, packet, path_len);\n'
        'if(currentMedium<0) splat_segment_spectral(ro, hp, packet);\n'
        '    int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));')
    result=recording_shader(source,12,64,ROOT_ESTIMATOR)
    assert result.count('int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));')==1
    assert 'uTotalRaysPerSource/uSourceCount+(_ray_idx<uTotalRaysPerSource%uSourceCount ? 1 : 0)' in result
    assert 'studio_record_root_weight=1.0/float(replay)' in result
    with pytest.raises(ValueError):recording_shader(source.replace('    int _ray_idx','    int changed'),12,64,ROOT_ESTIMATOR)
