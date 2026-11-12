import numpy as np
import pytest
from camera_designer.studio_field_arrivals import encode_arrivals,DTYPE,merge_context,checked_counts

def arrays():
    return [np.zeros((4,3)),np.zeros((4,3,3)),np.zeros((4,3),np.uint32),np.zeros((4,3),np.uint32),np.zeros((4,3),np.uint32),np.zeros((4,8),np.uint32)]

def encode(a,**kw):
    return encode_arrivals(*a,quantization_divisor=16,voxel_volume_m3=.5,populations=2,offset_bytes=64,**kw)

def test_zero_visits_are_not_empty_and_exclusions_survive():
    a=arrays();a[2][1,0]=3;a[3][1,0]=3;a[4][2,2]=1
    blob,meta=encode(a);r=np.frombuffer(blob,DTYPE)
    assert r['index'].tolist()==[1,2]
    assert r['visits'][0].tolist()==[3,0,0]
    assert not r['spectral'].any()
    assert meta['count']==2 and meta['stride_bytes']==120
    assert 'unknown' in meta['missing']

def test_band_moments_normalization_and_no_energy_selection():
    a=arrays();a[0][1]=[16,32,48];a[1][1]=[[16,0,0],[0,-32,0],[0,0,48]];a[2][1]=[1,1,1]
    blob,_=encode(a);r=np.frombuffer(blob,DTYPE)
    np.testing.assert_array_equal(r['spectral'][0],[[1,1,0,0],[2,0,-2,0],[3,0,0,3]])
    with pytest.raises(ValueError,match='no records omitted'):encode(a,payload_limit=183)

def test_counter_overflow_and_inconsistent_evidence_rejected():
    with pytest.raises(ValueError):checked_counts([2**32])
    a=arrays();a[0][0,0]=1
    with pytest.raises(ValueError,match='without arrival'):encode(a)
    a=arrays();a[3][0,0]=1
    with pytest.raises(ValueError,match='exceeds'):encode(a)

def test_context_preserves_mixed_unknown_not_fabricated_owner():
    a=np.array([1,1,1,2,2,8,8,1],np.uint32);b=np.array([4,4,4,3,3,9,9,2],np.uint32)
    assert merge_context(a,b).tolist()==[1,4,5,2,3,8,9,3]

@pytest.mark.parametrize('regions',[False,True])
def test_actual_source_observer_seams_are_opt_in_and_prequantization(regions):
    import runpy
    from pathlib import Path
    canonical_source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']
    from camera_designer.studio_directional_field import recording_shader,ROOT_ESTIMATOR
    from camera_designer.studio_directional_media import with_internal_medium_recording
    from camera_designer.studio_region_gpu import with_region_media
    from camera_designer.studio_field_arrivals import observation_shader
    base=canonical_source()
    if regions:base=with_region_media(base)
    recorded=with_internal_medium_recording(recording_shader(base,grid_size=64,estimator=ROOT_ESTIMATOR,independent_bounds=True,quantization_divisor=16777216))
    assert 'ArrivalCounts' not in recorded and 'ArrivalMoments' not in base
    result=observation_shader(recorded,64,12)
    assert result.count('arrival_kind=1u;studio_record_segment')==1
    assert result.count('arrival_kind=2u;studio_record_segment')==1
    assert result.index('arrival_observe(q,d,packet,meters);')<result.index('int w=int(round(max(scaled,0.0)));')
    assert 'arrival_inc(base+21+band)' in result
    assert 'if(arrival_medium < -1)return;' in result
    assert result.count('packet.z*=studio_medium_transmittance(currentMedium,packet.x,best);')==2


def test_additional_information_requires_explicit_internal_field():
    from camera_designer.studio_forward_field import validate_receive_light_field
    assert validate_receive_light_field({}) is False
    for config in [dict(light_field_information='spectral-arrivals-v1'),dict(light_field_information=True),dict(receive_light_field=True,transport='fixed:3',light_field_information='spectral-arrivals-v1')]:
        with pytest.raises(ValueError):validate_receive_light_field(config)

@pytest.mark.parametrize('regions',[False,True])
def test_arrival_observer_glsl_compiles_without_gpu(tmp_path,regions):
    import runpy,shutil,subprocess
    from pathlib import Path
    from camera_designer.studio_directional_field import recording_shader,ROOT_ESTIMATOR
    from camera_designer.studio_directional_media import with_internal_medium_recording
    from camera_designer.studio_region_gpu import with_region_media
    from camera_designer.studio_field_arrivals import observation_shader
    source=runpy.run_path(str(Path(__file__).with_name('test_studio_directional_media.py')))['canonical_source']()
    from camera_designer.spectral_sensor_basis import FIXED3_SPECTRAL_GLSL
    from camera_designer.studio_diffuse_surface import STUDIO_DIFFUSE_GLSL
    from camera_designer.studio_spectral_reflector import STUDIO_REFLECTOR_GLSL
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    import mat_flags as flags
    prefix=''.join('#define '+key+' '+str(getattr(flags,key))+('u' if key.startswith('MAT_FLAG') else '')+'\n' for key in ('MAT_FLAG_SPECTRAL_REFLECTOR','MAT_FLAG_ROUGH_CONDUCTOR','MAT_FLAG_NATIVE_CONDUCTOR_NK'))
    prefix+='#define STUDIO_NATIVE_PBR_BINDING '+str(flags.BINDING_STUDIO_NATIVE_PBR)+'\n'
    source=source.replace('__FIXED3_SPECTRAL_HELPERS__',FIXED3_SPECTRAL_GLSL).replace(STUDIO_BOUNDARY_GLSL,STUDIO_DIFFUSE_GLSL+prefix+STUDIO_REFLECTOR_GLSL+STUDIO_BOUNDARY_GLSL)
    if regions:source=with_region_media(source)
    source=observation_shader(with_internal_medium_recording(recording_shader(source,grid_size=64,estimator=ROOT_ESTIMATOR,independent_bounds=True,quantization_divisor=16777216)),64,12)
    compiler=shutil.which('glslangValidator');assert compiler
    path=tmp_path/'arrivals.comp';path.write_text(source,encoding='utf-8')
    result=subprocess.run([compiler,str(path)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


def test_full_grid_resource_capacity_and_domain_separation():
    from camera_designer.studio_field_arrival_gpu import resource_plan
    from camera_designer.studio_field_arrivals import merge_domain_rows
    plan=resource_plan(64,12)
    assert plan['primary_records']==64**3*12
    assert plan['record_capacity']>=plan['primary_records']
    assert max(plan['table_chunk_bytes'])<=128*1024**2
    assert plan['gpu_bytes']<512*1024**2
    rows=np.zeros((2,64),np.uint32);rows[:,0]=2;rows[:,1]=7;rows[:,2]=2;rows[1,3]=1
    rows[:,35]=16;rows[:,47]=1
    history=merge_domain_rows({},rows)
    assert len(history)==2 # host and host+active dye never collapse
    merge_domain_rows(history,rows)
    assert all(v[0][0]==32 and v[1][0]==2 for v in history.values())

def test_wire_chunks_keep_exact_dye_domains_and_zero_evidence():
    from camera_designer.studio_field_arrivals import chunk_sections,WIRE_DTYPE,DOMAIN_KEY_DTYPE
    rows=np.zeros((2,64),np.uint32);rows[:,0]=2;rows[:,1]=7;rows[:,2]=2;rows[1,3]=1
    rows[:,47]=1;rows[:,50]=1 # actual visits whose energy rounded tozero
    from camera_designer.studio_field_arrivals import merge_domain_rows
    record=dict(arrival_history=merge_domain_rows({},rows),progressive={'metadata':{'count':1}},quantization_divisor=16,voxel_volume_m3=.5,grid_size=32,bin_count=12)
    sections=chunk_sections(record)
    a=np.frombuffer(b''.join(sections[0]['blocks']),WIRE_DTYPE);d=np.frombuffer(b''.join(sections[1]['blocks']),DOMAIN_KEY_DTYPE)
    assert len(a)==2 and a['index'].tolist()==[7,7]
    assert a['domain_id'].tolist()==[0,1] and d['active'][:,0].tolist()==[0,1]
    assert a['visits'][:,0].tolist()==[1,1] and not a['spectral'].any()
