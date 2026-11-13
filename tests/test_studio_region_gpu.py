import numpy as np
import pytest
from camera_designer.studio_region_gpu import validate_region_ranges, tag_region_geometry, region_records_from_geometry,validate_region_records


def fixture():
    return {'version':1,'regions':[
      {'start':0,'count':1,'mode':'host','material':0},
      {'start':1,'count':1,'mode':'injection_dye','material':1},
      {'start':2,'count':1,'mode':'air_bubble','material':-1}]}


def test_region_tags_survive_material_remapping():
    spec,owners,pairs=validate_region_ranges(fixture(),3,2)
    g=np.zeros((3,16),np.float32);tag_region_geometry(g,spec,owners,np.ones(3,bool),{0:3,1:8})
    records=region_records_from_geometry(g)
    np.testing.assert_array_equal(records[:,:2],[[0,3],[2,8],[3,-1]])
    np.testing.assert_array_equal(pairs,[[-1,0],[0,1],[0,-1]])
    validate_region_records(records,g,np.zeros((3,16)),9)


def test_region_map_cannot_skip_or_double_assign_geometry():
    v=fixture();v['regions'][1]['start']=0
    with pytest.raises(ValueError,match='overlap'):validate_region_ranges(v,3,2)
    v=fixture();v['regions'].pop()
    with pytest.raises(ValueError,match='cover'):validate_region_ranges(v,3,2)


def test_shader_variants_share_native_optical_laws_and_leave_default_source_untouched():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import demo_pluck_gl as renderer\nfor legacy,source in [(renderer._GPU_RAY_FIELD_CS,renderer._GPU_RAY_FIELD_REGION_CS),\n                      (renderer._GPU_SENSOR_CS,renderer._GPU_SENSOR_REGION_CS)]:\n    assert 'studio_active[32]' not in legacy\n    assert 'studio_active[32]' in source\n    assert source.index('float studio_region_epsilon(vec3 p);') < source.index('bool hit_tri(')\n    assert 'studio_legacy_boundary_step(resolved' in source\n    assert 'studio_classify_regions(studio_region_hit_position+rd*studio_region_epsilon(studio_region_hit_position),rd)' in source\n    assert 'studio_matches_region(geom_tris[ti].v0.w,geom_tris[ti].e1.w,studio_region_filter)' in source\n    assert 'studio_region_hit_position=hp;' in source\n    assert '__SENSOR_HISTORY_HELPERS__' not in source\n    assert '__SENSOR_EVIDENCE_HELPERS__' not in source"],check=True,timeout=20)


def test_scene_append_projects_region_optics_into_native_gpu_records():
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),
           np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},
           (np.full(3,-1,np.float32),np.full(3,1,np.float32)),np.zeros((0,12),np.float32))
    v=np.array([[[-1,-1,0],[1,-1,0],[0,1,0]]]*3,np.float32)
    p=np.zeros((2,16),np.float32);p[:,5]=1;p[:,6]=1.52
    optics={'version':1,'materials':[dict(material=i,wavelengths_nm=[400,700],transmittance=[.7,.7],reference_thickness_mm=1,concentration=1) for i in range(2)]}
    result=append_saved_studio_mesh_to_gpu_scene(scene,v,np.array([0,1,0]),p,scale_m=.01,center_m=[0,0,0],wavelengths_um=[.55],region_media=fixture(),optical_media=optics)
    records=region_records_from_geometry(result[0])
    assert records.shape==(3,4)
    validate_region_records(records,result[0],result[1],len(result[2])//32)
    assert result[0][0,11]==4097 # legacy common geometry marker preserved


def test_native_t5_scene_handoff_keeps_authored_region_identity():
    from camera_designer.studio_t5_scene import native_t5_scene_inputs
    from mat_flags import MAT_FLAG_EMISSIVE, MAX_SPECTRAL_BANDS
    geom=np.zeros((4,16),np.float32)
    shade=np.zeros((4,16),np.float32)
    shade[:,14]=np.arange(4,dtype=np.int32).view(np.float32)
    geom[:,3]=[-1,-7,-12,0]
    geom[:,7]=[2,3,1,0]
    geom[3,15]=np.array([MAT_FLAG_EMISSIVE],np.uint32).view(np.float32)[0]
    materials=np.zeros((4*MAX_SPECTRAL_BANDS,12),np.float32)
    materials[3*MAX_SPECTRAL_BANDS,5]=1
    tri=np.zeros((4,3,3),np.float32)
    tri[:,1,0]=1;tri[:,2,1]=1
    scene=(geom,shade,materials,tri,np.empty((0,8),np.float32),{},
           (np.zeros(3,np.float32),np.ones(3,np.float32)),np.empty((0,12),np.float32))
    native=native_t5_scene_inputs(scene,[.55])
    np.testing.assert_array_equal(native['boundary_tags'],geom[:,[3,7]])
    np.testing.assert_array_equal(native['region_records'][:,:2],[[0,0],[2,1],[3,-1]])
    np.testing.assert_array_equal(native['native_region_media'],[[0,0],[2,1],[3,-1]])
    np.testing.assert_array_equal(native['tri_region_ids'],[0,1,2,-1])
    np.testing.assert_array_equal(native['emitter_tri_ids'],[3])


def test_native_t5_scene_preserves_exact_sphere_owner():
    from camera_designer.studio_t5_scene import native_t5_scene_inputs, create_native_t5_tracer
    from mat_flags import MAT_FLAG_EMISSIVE, MAX_SPECTRAL_BANDS
    geom=np.zeros((2,16),np.float32)
    shade=np.zeros((2,16),np.float32)
    shade[:,14]=np.arange(2,dtype=np.int32).view(np.float32)
    geom[1,15]=np.array([MAT_FLAG_EMISSIVE],np.uint32).view(np.float32)[0]
    materials=np.zeros((2*MAX_SPECTRAL_BANDS,12),np.float32)
    materials[MAX_SPECTRAL_BANDS,5]=1
    tri=np.zeros((2,3,3),np.float32)
    tri[0]=[.01,.02,.03]
    tri[1,1]=[1,0,0];tri[1,2]=[0,1,0]
    sphere=np.array([[.01,.02,.03,.002,0,0,0,0]],np.float32)
    scene=(geom,shade,materials,tri,np.empty((0,8),np.float32),{},
           (np.full(3,-1,np.float32),np.ones(3,np.float32)),np.empty((0,12),np.float32))
    native=native_t5_scene_inputs(scene,[.55],sphere)
    np.testing.assert_array_equal(native['exact_spheres'],sphere[:,:5])
    tracer=create_native_t5_tracer(scene,[.55],sphere)
    try:
        assert tracer is not None
    finally:
        tracer.stop_pipeline()
    with pytest.raises(ValueError,match='owner must be'):
        native_t5_scene_inputs(scene,[.55],np.array([[.01,.02,.03,.002,1,0,0,0]],np.float32))


def test_region_thresholds_are_coordinated_and_scale_relative():
    from camera_designer.studio_region_gpu import with_region_media, with_region_spheres
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    from camera_designer.studio_sphere_regions import STUDIO_SPHERE_GLSL
    snippets = STUDIO_BOUNDARY_GLSL + """
    (!boundary_only || (geom_tris[ti].v0.w>=1.0 && shade_tris[ti].mat_out.w<0.999))
    if (abs(a) < 1e-8) return false;
    if (tt <= 1e-6) return false;
    ro=hp+rd*1e-6;
    ro = hp + rd * (studio_media_enabled()?1e-6:1e-5);
    """
    s=with_region_media(snippets)+STUDIO_SPHERE_GLSL+"""
    if(boundary_only && !(geom_tris[ti].v0.w>=1.0 && shade_tris[ti].mat_out.w<0.999)) continue;
    """
    s += "int nearest_hit(vec3 ro,vec3 rd,out float best) {return nearest_filtered(ro,rd,best,false);}"
    s=with_region_spheres(s)
    assert 'if (tt <= studio_region_epsilon(ro)' in s
    assert 'studio_region_epsilon(ro)*ray_length/sphere.w' in s
    assert 'ro=hp+rd*studio_region_epsilon(hp);' in s
    assert 'length(t.e1.xyz) * length(t.e2.xyz) * length(rd)' in s
    assert 'studio_region_hit_position+rd*1e-6' not in s
    # Numerical budget at production coordinates is well below the previous
    # fixed1um skip; this is not a proof of arbitrary float32 geometry.
    eps=2*np.finfo(np.float32).eps*np.float32(.18)
    assert 0 < eps < .4e-6/4
    for scale in (1., .025, .001, .000001):
        e1=np.array([scale,0,0]);e2=np.array([0,scale,0]);rd=np.array([0,0,1])
        det=abs(np.dot(e1,np.cross(rd,e2)))
        assert det > 2*np.finfo(np.float32).eps*np.linalg.norm(e1)*np.linalg.norm(e2)


def test_additive_dyes_cannot_escape_weak_absorption_regime():
    from camera_designer.studio_optical_media import validate_region_absorption_bound
    record={'wavelengths_nm':[400,800], 'transmittance':[.1,.1],
            'reference_thickness_mm':1., 'concentration':3.}
    # Each individual entry has k<.001; repeated dye volumes accumulate.
    optics={0:{**record,'transmittance':[1.,1.]},1:record}
    regions=[{'mode':'host','material':0}]+[{'mode':'injection_dye','material':1}]*3
    validate_region_absorption_bound(regions[:2],optics)
    with pytest.raises(ValueError,match='combined dye'):
        validate_region_absorption_bound(regions,optics)


def test_bundle_metadata_survives_native_material_records():
    from camera_designer.studio_rod_bundle_gpu import pack_bundle_records, region_record_count
    records=np.array([[0,3,0,0],[1,4,0,0],[1,5,0,0]],np.float32)
    b={'regions':[1,2],'eye_length':.75,'eye_curve':.18,'twist':.9,'phase':.2,
       'frame':{'origin':[.01,0,0],'axes':np.eye(3).tolist(),'scale':.9}}
    packed=pack_bundle_records(records,b,scale_m=.025,center_m=[0,0,.18])
    assert region_record_count(packed)==3 and packed.shape==(8,4)
    np.testing.assert_array_equal(packed[1:3,2],[1,2])
    world=np.array([.01*.025,0,.18,1.])
    np.testing.assert_allclose(packed[3:6]@world,0,atol=1e-6)
    geom=np.zeros((3,16),np.float32);geom[:,3]=[-1,-6,-10];geom[:,7]=[5,6,7]
    out=validate_region_records(packed,geom,np.zeros((3,16)),6)
    np.testing.assert_array_equal(out,packed)
    corrupt=packed.copy();corrupt[2,2]=1
    with pytest.raises(ValueError,match='rod order'):validate_region_records(corrupt,geom,np.zeros((3,16)),6)


def test_camera_geometry_prefix_change_refreshes_bundle_host_index():
    from camera_designer.studio_rod_bundle_gpu import pack_bundle_records
    from camera_designer.studio_region_gpu import refreshed_region_records
    records=np.array([[0,3,0,0],[1,4,0,0],[1,5,0,0]],np.float32)
    bundle={'regions':[1,2],'eye_length':.75,'eye_curve':.18,'twist':.9,'phase':0.,
       'frame':{'origin':[0,0,0],'axes':np.eye(3).tolist(),'scale':1.}}
    previous=pack_bundle_records(records,bundle,scale_m=.025,center_m=[0,0,.18],host_triangle=0)
    geom=np.zeros((5,16),np.float32);geom[2:,3]=[-1,-6,-10];geom[2:,7]=[5,6,7]
    current=refreshed_region_records(geom,previous)
    assert current[7,1]==2 and previous[7,1]==0
    np.testing.assert_array_equal(current[3:7],previous[3:7])
    validate_region_records(current,geom,np.zeros((5,16)),6)


def test_standalone_scene_medium_is_distinct_from_subject_regions():
    from camera_designer.studio_region_gpu import region_records_from_geometry
    geom=np.zeros((3,16),np.float32);geom[:,3]=[-1,1,1];geom[:,7]=[2,3,4]
    records=region_records_from_geometry(geom)
    np.testing.assert_array_equal(records[:,:2],[[0,0],[4,1],[4,2]])
    validate_region_records(records,geom,np.zeros((3,16)),3)
    broken=geom.copy();broken[1,3]=2
    with pytest.raises(ValueError,match='outward air/material'):region_records_from_geometry(broken)
