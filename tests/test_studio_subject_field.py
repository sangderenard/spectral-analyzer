import numpy as np
import pytest
from camera_designer.studio_directional_field import subject_field_region,subject_recording_region,recording_shader

def mesh(size=.02):
    return np.array([[[-size/2,-size/2,-size/2],[size/2,size/2,size/2],[0,0,0]]],np.float32)

def test_exact_pitch_padding_and_admission():
    roi=subject_recording_region(mesh(),scale_m=1,center_m=[0,0,0],cell_size_m=.0005)
    assert roi['dimensions']==[64]*3
    assert np.allclose(np.subtract(roi['bounds_max_m'],roi['bounds_min_m']),.032,rtol=0,atol=1e-15)
    assert roi['margin_m']==.001
    for k in range(3):
        assert roi['bounds_min_m'][k]<roi['region']['actual_bounds_min_m'][k]
        assert roi['bounds_max_m'][k]>roi['region']['actual_bounds_max_m'][k]
    with pytest.raises(ValueError,match='maximum is64'):
        subject_recording_region(mesh(.04),scale_m=1,center_m=[0,0,0],cell_size_m=.0005)

def test_zero_margin_still_outer_boundary_guard():
    roi=subject_recording_region(mesh(.005),scale_m=1,center_m=[0,0,0],cell_size_m=.0005,margin_m=0)
    assert roi['dimensions']==[32]*3
    assert roi['region']['effective_margin_m']==.0005

def test_spheres_and_receipt_hash():
    config=dict(scale_m=1,object_center_m=[0,0,0],light_field_cell_size_m=.0005,geometry_sha256='a'*64,sphere_coverage='exact',analytic_spheres={'spheres':[{'center':[0,0,0],'radius':.01}]})
    roi=subject_field_region(mesh(.005),config)
    assert roi['region']['subject_geometry_sha256']=='a'*64
    assert roi['region']['actual_bounds_max_m'][0]>.009
    assert len(roi['region']['world_geometry_sha256'])==64

def test_shader_only_recorder_bounds_changed():
    source='void splat_spectral(vec3 p, vec4 packet) { }\nif(currentMedium<0) splat_segment_multiscale(ro, hp, packet, path_len);\nif(currentMedium==-1) splat_segment_spectral(ro, hp, packet);\n    int _ray_idx = int(ray_id % uint(max(uSourceCount, 1)));\nvec3 original=uBoxMin+uBoxMax;'
    out=recording_shader(source,grid_size=64,estimator='track-length-root-sweep-v1',independent_bounds=True)
    assert 'vec3 original=uBoxMin+uBoxMax;' in out
    assert 'uniform vec3 uStudioRecordMin;' in out
    assert 'studio_record_track(q,d,packet,next-cursor);' in out

@pytest.mark.parametrize("pitch",[0,.000049,.010001,float('nan'),float('inf')])
def test_pitch_range(pitch):
    with pytest.raises(ValueError,match='light_field_cell_size_m'):
        subject_recording_region(mesh(),scale_m=1,center_m=[0,0,0],cell_size_m=pitch)
