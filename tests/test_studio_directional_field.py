import numpy as np
import pytest
from camera_designer.studio_directional_field import triangle_metadata, recording_shader, bundle_sections, scene_digest


def test_original_triangle_ids_and_native_flags():
    g=np.zeros((3,16),np.float32);s=np.zeros((3,16),np.float32)
    g[:,15]=np.array([0,64,4],np.uint32).view(np.float32)
    s[:,14]=np.array([2,1,0],np.uint32).view(np.float32)
    s[:,7]=[1,.5,1];s[:,4]=[.2,0,0]
    np.testing.assert_array_equal(triangle_metadata(g,s),[[0,2,0,3],[1,1,64,7],[2,0,4,1]])


def test_bundle_sections_are_contiguous_actual_materials_and_sources(monkeypatch):
    import camera_designer.studio_directional_field as m
    atlas=np.zeros((32,32,32,6,6),'<f4');atlas[0,0,0,0]=[1,2,3,.8,.2,.1]
    monkeypatch.setattr(m,'readback',lambda record:atlas)
    triangles=np.array([[0,0,0,3]],'<u4');materials=np.arange(16,dtype='<f4').reshape(1,16)
    sources=np.arange(36,dtype='<f4').reshape(3,12)
    blob,meta=bundle_sections(dict(triangles=triangles,materials=materials,scene_sha256='a'*64,bin_count=6),sources,48)
    offset=48
    for key,data in [('directional',atlas),('triangles',triangles),('sources',sources),('materials',materials)]:
        section=meta[key]
        assert section['offset_bytes']==offset and section['byte_length']==data.nbytes
        assert blob[offset-48:offset-48+data.nbytes]==data.tobytes()
        offset+=data.nbytes
    assert len(blob)==offset-48
    validity=meta['directional']['validity']
    assert validity==m.partial_estimate_validity()
    assert 'replacement_authorized' not in validity
    assert 'darkness_authorized' not in validity
    assert validity['zero_semantics']=='unknown'
    assert validity['occlusion_certification']=='none'


def test_shader_records_only_actual_segment_directions():
    source=('void splat_spectral(vec3 p, vec4 packet) {\n}\n'
        'if(currentMedium<0) splat_segment_multiscale(ro, hp, packet, path_len);\n'
        'if(currentMedium<0) splat_segment_spectral(ro, hp, packet);\n'
        'splat_spectral(hp, emit_pkt);')
    shader=recording_shader(source,stencil=6)
    assert shader.count('studio_record_segment(ro,hp,packet);')==2
    region=recording_shader(source.replace('currentMedium<0','currentMedium==-1'))
    assert region.count('if(currentMedium==-1) { studio_record_segment(ro,hp,packet);')==2
    assert shader.count('splat_spectral(hp, emit_pkt);')==1
    assert 'for(int iteration=0;iteration<99;++iteration)' in shader
    assert 'atomicOr(studio_directional[1179648],1)' in shader
    for broken in ['not canonical shader',source.replace('if(currentMedium<0) splat_segment_spectral(ro, hp, packet);','')]:
        with pytest.raises(ValueError):recording_shader(broken)


def test_scene_hash_changes_on_actual_materials_and_sources():
    a=np.zeros((1,16),np.float32);s=np.zeros((1,12),np.float32)
    before=scene_digest(a,a,a,s)
    changed=a.copy();changed[0,3]=.5
    assert scene_digest(a,a,changed,s)!=before
    s[0,0]=1
    assert scene_digest(a,a,a,s)!=before


def test_registration_seam_keeps_authored_rows_after_projection_and_degenerate_filter():
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),
        np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},
        (np.array([-1,-1,-1],np.float32),np.array([1,1,1],np.float32)),np.zeros((0,12),np.float32))
    vertices=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,0,1],[0,1,1]],[[0,0,0]]*3],np.float32)
    projected=np.zeros((2,16),np.float32);projected[:,:3]=.8;projected[:,6]=1.5;projected[:,7]=1
    authored=projected.copy();authored[0,8:11]=[.3,.2,.1];authored[1,3]=.7;authored[0,12]=-0.
    lineage=[]
    result=append_saved_studio_mesh_to_gpu_scene(scene,vertices,[0,1,0],projected,
        scale_m=1,center_m=[0,0,0],material_lineage_out=lineage,
        lineage_source=([1,0,0],authored))
    actual_ids=result[1][:,14].copy().view(np.uint32)
    records={(native,row.tobytes()) for native,row in lineage}
    assert records=={(int(actual_ids[0]),authored[1].tobytes()),(int(actual_ids[1]),authored[0].tobytes())}
    assert len(lineage)==2

def test_authored_lineage_is_exact_deduplicated_and_appended(monkeypatch):
    import camera_designer.studio_directional_field as m
    monkeypatch.setattr(m,'readback',lambda record:np.zeros((32,32,32,6,6),'<f4'))
    row=np.arange(16,dtype='<f4');row[12]=-0.
    second=row.copy();second[8]=.1
    record=dict(bin_count=6,triangles=np.array([[0,0,0,3]],'<u4'),materials=np.zeros((1,16),'<f4'),scene_sha256='a'*64,
        material_lineage=[(0,row),(0,row.copy()),(0,second)])
    data,meta=m.bundle_sections(record,np.zeros((0,12),'<f4'),12)
    section=meta['source_pbr16']
    assert section['count']==2 and section['material_ids']==[0,0]
    assert section['byte_length']==128
    assert data[section['offset_bytes']-12:]==row.tobytes()+second.tobytes()


def test_voxel_volume_scaling_and_invalid_bounds():
    from camera_designer.studio_directional_field import voxel_volume
    assert voxel_volume(([0,0,0],[1,1,1]))==1/32**3
    assert voxel_volume(([0,0,0],[2,2,2]))==8/32**3
    with pytest.raises(ValueError):voxel_volume(([1,0,0],[0,1,1]))


@pytest.mark.parametrize('count',[6,12])
def test_explicit_unit_basis_shader_allocation_and_bin_ownership(count):
    from camera_designer.studio_directional_field import stencil_directions,directional_glsl,word_count
    directions=stencil_directions(count)
    assert directions.shape==(count,3)
    np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,atol=2e-16)
    np.testing.assert_array_equal(np.argmax(directions@directions.T,axis=1),np.arange(count))
    assert all(np.any(np.linalg.norm(directions+row,axis=1)<1e-15) for row in directions)
    shader=directional_glsl(count)
    assert f'STUDIO_DIRECTION_BIN_COUNT={count};' in shader
    assert f'studio_directional[{word_count(count)}]' in shader
    assert '__STUDIO_' not in shader
    assert 'if(score>closest)' in shader  # deterministic lowest-index tie
    assert word_count(count)*4==32**3*count*6*4


def test_default12_strict_configuration_and_no_fabricated_direction():
    from camera_designer.studio_directional_field import DEFAULT_STENCIL,validate_stencil,directional_glsl
    from camera_designer.studio_forward_field import validate_receive_light_field
    assert DEFAULT_STENCIL==12
    for value in [True,6.,12.,0,8,24,'12',None]:
        with pytest.raises(ValueError):validate_stencil(value)
        with pytest.raises(ValueError):validate_receive_light_field({'light_field_stencil':value})
    shader=directional_glsl()
    assert 'int(round(float(w)*d.x))' in shader
    assert 'studio_record_track(q,d,packet,next-cursor);' in shader


def test_inverse_rgb_weight_recovers_unclipped_spectral_weight():
    from camera_designer.studio_directional_field import weight_from_linear_rgb
    from camera_designer.marble_color_profile import XYZ_TO_SRGB,_LINE_TO_XYZ
    lines=np.array([[1,0,0],[0,1,0],[0,0,1],[.2,.7,1.3],[0,0,0]],float)
    rgb=lines@(XYZ_TO_SRGB@_LINE_TO_XYZ).T
    assert rgb.min()<0
    np.testing.assert_allclose(rgb@weight_from_linear_rgb(),lines.sum(axis=1),rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('count',[6,12])
def test_v3_metadata_describes_exact_stencil_and_weights(monkeypatch,count):
    import camera_designer.studio_directional_field as m
    monkeypatch.setattr(m,'readback',lambda record:np.zeros((32,32,32,count,6),'<f4'))
    record=dict(bin_count=count,triangles=np.zeros((0,4),'<u4'),materials=np.zeros((1,16),'<f4'),scene_sha256='a'*64)
    blob,meta=m.bundle_sections(record,np.zeros((0,12),'<f4'),0)
    d=meta['directional']
    assert d['stencil']==('axes6-v1' if count==6 else 'icosahedron12-v1')
    assert d['bin_count']==len(d['bins'])==len(d['bin_directions'])==count
    assert d['byte_length']==32**3*count*6*4
    np.testing.assert_array_equal(d['bin_directions'],m.stencil_directions(count))
    np.testing.assert_array_equal(d['weight_from_linear_rgb'],m.weight_from_linear_rgb())


@pytest.mark.parametrize('count',[6,12])
def test_stencil_classification_once_per_segment_preserves_atmosphere_hook(count):
    from camera_designer.studio_directional_field import directional_glsl
    source=directional_glsl(count)
    cell=source.split('void studio_record_track(',1)[1].split('void studio_record_segment(',1)[0]
    segment=source.split('void studio_record_segment(',1)[1]
    assert 'dot(d,STUDIO_DIRECTION_BASIS' not in cell
    assert 'STUDIO_DIRECTION_BIN_COUNT+studio_record_bin' in cell
    assert segment.count('for(int i=1;i<STUDIO_DIRECTION_BIN_COUNT;++i)')==1
    assert segment.index('studio_record_bin=0;')<segment.index('for(int iteration=0;iteration<99;++iteration)')
    assert 'if(score>closest){closest=score;studio_record_bin=i;}' in segment
    assert source.count('studio_record_track(q,d,packet,next-cursor);')==1


def test_slab_decode_is_bit_identical_to_original_full_conversion():
    from camera_designer.studio_directional_field import _decode_directional_slices
    from camera_designer.marble_color_profile import XYZ_TO_SRGB, _LINE_TO_XYZ
    rng=np.random.default_rng(824)
    for dtype in (np.int32,np.float64):
        raw=rng.integers(0,1000000,size=(4,5,6,12,6)).astype(dtype)
        raw[0,0,0,:,:]=0
        raw[...,3:]-=10000
        record={'quantization_divisor':16777216,'voxel_volume_m3':.0005**3}
        divisor=22
        expected=np.zeros(raw.shape,dtype='<f4')
        expected[...,:3]=((raw[...,:3].astype(np.float64)/record['quantization_divisor'])@(XYZ_TO_SRGB@_LINE_TO_XYZ).T)/(record['voxel_volume_m3']*divisor)
        weight=raw[...,:3].astype(np.float64).sum(axis=-1)
        np.divide(raw[...,3:],weight[...,None],out=expected[...,3:],where=weight[...,None]>0)
        norm=np.linalg.norm(expected[...,3:],axis=-1,keepdims=True)
        expected[...,3:]/=np.maximum(norm,1)
        actual=_decode_directional_slices(raw,record,divisor)
        assert actual.tobytes()==expected.tobytes()


def test_streamed_sections_are_byte_identical_without_full_join(monkeypatch):
    import camera_designer.studio_directional_field as m
    monkeypatch.setattr(m,'readback',lambda record:np.arange(2*2*2*6*6,dtype='<f4').reshape(2,2,2,6,6))
    record=dict(bin_count=6,triangles=np.zeros((1,4),'<u4'),materials=np.zeros((1,16),'<f4'),scene_sha256='a'*64)
    source=np.zeros((1,12),'<f4')
    joined,meta=m.bundle_sections(record,source,12)
    parts,stream_meta=m.bundle_sections(record,source,12,streaming=True)
    assert isinstance(parts[0],memoryview)
    assert b''.join(parts)==joined and stream_meta==meta
