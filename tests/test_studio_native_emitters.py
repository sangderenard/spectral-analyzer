"""Real native source-component tests; CPU only, no service or GPU."""
import numpy as np
import pytest
pytest.importorskip('_spectral_kernels')
from camera_designer.studio_native_emitters import native_emissive_sources, area_nee_records, reweight_native_sources, install_timed_light_frame, timed_light_buffers
from ray_tracer_bridge import per_tri_spectral_to_mat_buf
from mat_flags import MAT_FLAG_EMISSIVE


def fixture(emission=(1.,2.,3.)):
    triangles=np.array([[[0,0,0],[0,1,0],[0,0,1]],[[1,0,0],[1,3,0],[1,0,3]]],float)
    frequencies=299792458./np.array([450e-9,550e-9,650e-9])
    zeros=np.zeros((2,3))
    mids,materials,_=per_tri_spectral_to_mat_buf(zeros,zeros,zeros,frequencies,
        emission_bands=np.tile(emission,(2,1)))
    geom=np.zeros((2,16),np.float32);geom[:,:3]=triangles[:,0]
    geom[:,4:7]=triangles[:,1]-triangles[:,0];geom[:,8:11]=triangles[:,2]-triangles[:,0]
    geom[:,12]=1;geom[:,15]=np.array([MAT_FLAG_EMISSIVE]*2,np.uint32).view(np.float32)
    shade=np.zeros((2,16),np.float32);shade[:,14]=mids.astype(np.uint32).view(np.float32)
    return (geom,shade,materials,triangles)


@pytest.mark.parametrize('emission',[(1,2,3),(0,2,0)])
def test_native_roots_keep_unequal_triangle_power_and_frequencies(emission):
    scene=fixture(emission);rows=native_emissive_sources(scene,[.45,.55,.65],ray_budget=600)
    assert len(np.unique(rows[:,:3],axis=0))==len(rows)
    np.testing.assert_allclose(np.linalg.norm(rows[:,4:7],axis=1),1,atol=2e-7)
    assert np.all(rows[:,4]>0)
    for tri,area in [(0,.5),(1,4.5)]:
        for band,wl in enumerate([450,550,650]):
            sel=(rows[:,7]==tri+1)&np.isclose(rows[:,8],299792458./(wl*1e-9))
            assert rows[sel,10].sum()==pytest.approx(area*emission[band],rel=3e-6,abs=1e-8)
    camera=area_nee_records(rows,*scene[:4])
    for tri,area in [(0,.5),(1,4.5)]:
        sel=rows[:,7]==tri+1
        np.testing.assert_allclose(camera[sel,4:7],np.tile(emission[::-1],(sel.sum(),1)))
        assert camera[sel,10].sum()==pytest.approx(area,rel=2e-6)
        np.testing.assert_array_equal(camera[sel][:,[3,7,11]],np.tile([1,0,0],(sel.sum(),1)))


def test_dark_geometry_has_no_invented_source():
    assert native_emissive_sources(fixture((0,0,0)),[.45,.55,.65]).shape==(0,12)


def test_slice_light_change_keeps_native_roots_and_reweights_each_band_once():
    scene=fixture((1.,2.,3.))
    roots=native_emissive_sources(scene,[.45,.55,.65],ray_budget=120)
    current=fixture((0.,1.,6.))[2]
    changed=reweight_native_sources(roots,scene[1],scene[2],current)
    np.testing.assert_array_equal(changed[:,:10],roots[:,:10])
    np.testing.assert_array_equal(changed[:,11],roots[:,11])
    np.testing.assert_array_equal(roots[:,10],native_emissive_sources(scene,[.45,.55,.65],ray_budget=120)[:,10])
    for tri,area in ((0,.5),(1,4.5)):
        for band,wl in enumerate((450,550,650)):
            selected=(changed[:,7]==tri+1)&np.isclose(changed[:,8],299792458./(wl*1e-9))
            assert changed[selected,10].sum()==pytest.approx(area*(0.,1.,6.)[band],rel=3e-6,abs=1e-8)
    camera=area_nee_records(changed,scene[0],scene[1],current,scene[3])
    np.testing.assert_allclose(camera[:,4:7],np.tile((6.,1.,0.),(len(camera),1)))


def test_slice_cannot_turn_on_unprepared_spectral_band():
    scene=fixture((0.,2.,0.))
    roots=native_emissive_sources(scene,[.45,.55,.65],ray_budget=60)
    with pytest.raises(ValueError,match='unprepared emitting band'):
        reweight_native_sources(roots,scene[1],scene[2],fixture((1.,2.,0.))[2])


def test_timed_light_frame_installs_same_sources_for_forward_and_camera_without_sensor_clear(monkeypatch):
    from OpenGL import GL
    scene=fixture((1.,2.,3.))
    roots=native_emissive_sources(scene,[.45,.55,.65],ray_budget=120)
    baseline=(*scene,None,None,None,roots)
    current=(*scene[:2],fixture((0.,1.,6.))[2],*scene[3:])
    uploads=[];bound=[None]
    monkeypatch.setattr(GL,'GL_SHADER_STORAGE_BUFFER',1,raising=False)
    monkeypatch.setattr(GL,'GL_STATIC_DRAW',2,raising=False)
    monkeypatch.setattr(GL,'glFinish',lambda:uploads.append(('finish',)),raising=False)
    monkeypatch.setattr(GL,'glBindBuffer',lambda target,handle:bound.__setitem__(0,handle),raising=False)
    monkeypatch.setattr(GL,'glBufferData',lambda target,size,data,usage:uploads.append((bound[0],np.array(data,copy=True))),raising=False)
    class Sensor:
        _row_off=0
        _nee_source_count=len(roots)
        def set_nee_spectral_sources(self,rows,original_buffer,*,area_scene):
            uploads.append(('camera',rows.copy(),original_buffer,area_nee_records(rows,*area_scene)))
        def clear(self):raise AssertionError('timed interval cleared the exposure')
    field={'ssbo':list(range(100,113)),'n_sources':len(roots),'source_buf':roots.copy()}
    installed=install_timed_light_frame(field,Sensor(),baseline,current)
    assert [entry[0] for entry in uploads]==['finish',109,110,105,'camera']
    np.testing.assert_array_equal(field['source_buf'],installed)
    np.testing.assert_array_equal(uploads[-1][1],installed)
    assert uploads[-1][2]==105


def test_real_studio_scene_prepares_dark_light_for_later_timed_emission():
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    subject=np.array([[[0,0,.18],[.01,0,.18],[0,.01,.18]]],np.float32)
    subject_pbr=np.zeros((1,16),np.float32);subject_pbr[0,:3]=.5;subject_pbr[0,6:8]=[1.5,1]
    fixture_pbr=np.zeros(16,np.float32);fixture_pbr[3]=fixture_pbr[7]=1
    fixture={"vertices":[-.1,-.1,.2,.1,-.1,.2,0,.1,.2],"material_ids":[0],
             "pbr":fixture_pbr.tolist(),"emissive_material_ids":[0]}
    config={"transport":"fixed:3","sphere_coverage":"mesh","fixtures":fixture,
            "camera":{"eye_m":[0,0,.07],"target_m":[0,0,.18],"fov_degrees":18,
                      "aperture_radius_m":.002,"focus_distance_m":.11}}
    prepared=fixture_pbr.copy();prepared[8:11]=1
    baseline,_,_,_=prepare_photo_geometry({**config,"fixtures":{**fixture,"pbr":prepared.tolist()}},
        subject,np.array([0],np.int32),subject_pbr)
    actual=fixture_pbr.copy()
    current,_,_,_=prepare_photo_geometry({**config,"fixtures":{**fixture,"pbr":actual.tolist()}},
        subject,np.array([0],np.int32),subject_pbr,prepare_sources=False)
    assert len(baseline[7])>0
    assert np.array_equal(baseline[0][:,:15],current[0][:,:15])
    _,mapped,weighted=timed_light_buffers(baseline,current)
    assert mapped.shape==baseline[2].shape
    assert np.count_nonzero(weighted[:,10])==0
    actual[8:11]=[0,2,0]
    lit,_,_,_=prepare_photo_geometry({**config,"fixtures":{**fixture,"pbr":actual.tolist()}},
        subject,np.array([0],np.int32),subject_pbr,prepare_sources=False)
    assert np.count_nonzero(timed_light_buffers(baseline,lit)[2][:,10])>0


def test_two_fixture_materials_retain_distinct_native_ownership_across_light_changes():
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    subject=np.array([[[0,0,.18],[.01,0,.18],[0,.01,.18]]],np.float32)
    subject_pbr=np.zeros((1,16),np.float32);subject_pbr[0,6:8]=[1.5,1]
    fixture_pbr=np.zeros((2,16),np.float32);fixture_pbr[:,3]=fixture_pbr[:,7]=1
    vertices=[-.1,-.1,.2,0,-.1,.2,-.1,0,.2,.01,-.1,.2,.11,-.1,.2,.01,0,.2]
    fixture={"vertices":vertices,"material_ids":[0,1],"pbr":fixture_pbr.ravel().tolist(),
             "emissive_material_ids":[0,1]}
    config={"transport":"fixed:3","sphere_coverage":"mesh","fixtures":fixture}
    baseline_pbr=fixture_pbr.copy();baseline_pbr[0,8:11]=1.01;baseline_pbr[1,8:11]=1.02
    baseline,_,_,_=prepare_photo_geometry({**config,"fixtures":{**fixture,"pbr":baseline_pbr.ravel().tolist()}},
        subject,np.array([0],np.int32),subject_pbr)
    current_pbr=fixture_pbr.copy();current_pbr[1,8:11]=[0,3,0]
    current,_,_,_=prepare_photo_geometry({**config,"fixtures":{**fixture,"pbr":current_pbr.ravel().tolist()}},
        subject,np.array([0],np.int32),subject_pbr,prepare_sources=False)
    _,_,roots=timed_light_buffers(baseline,current)
    active={int(tri):float(roots[roots[:,7]==tri,10].sum()) for tri in np.unique(roots[:,7])}
    assert len(active)==2 and sorted(value>0 for value in active.values())==[False,True]


def test_sensor_interval_weight_preserves_open_exposure_and_requires_sweep_boundary():
    import ast
    import math
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SensorAccumulator')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='set_exposure_interval_weight')
    namespace={'math':math}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'demo_pluck_gl.py','exec'),namespace)
    sensor=type('Sensor',(),{})()
    sensor._row_off=0;sensor._history_samples=0;sensor._pass=7
    namespace[method.name](sensor,.25)
    assert sensor._exposure_interval_weight==.25 and sensor._pass==7
    assert 'lum *= uExposureIntervalWeight;' in source
    sensor._row_off=16
    with pytest.raises(RuntimeError,match='complete sensor sweep'):
        namespace[method.name](sensor,.5)


def test_named_shop_flash_reuses_prepared_native_roots_at_trigger_gain():
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    from camera_designer.studio_named_emitters import named_emitter
    definition,_=named_emitter('fluorescent_cool_white')
    subject=np.array([[[0,0,.18],[.01,0,.18],[0,.01,.18]]],np.float32)
    subject_pbr=np.zeros((1,16),np.float32);subject_pbr[0,6:8]=[1.5,1]
    pbr=np.zeros(16,np.float32);pbr[3]=pbr[7]=1
    ref={'material':0,'profile':'fluorescent_cool_white','revision':definition['revision'],
         'gain':1.01,'filter':[1,1,1]}
    fixture={'vertices':[-.1,-.1,.2,.1,-.1,.2,0,.1,.2],'material_ids':[0],
             'pbr':pbr.tolist(),'emitter_profiles':{'version':1,'records':[ref]}}
    config={'transport':'fixed:3','sphere_coverage':'mesh','fixtures':fixture}
    baseline,_,_,_=prepare_photo_geometry(config,subject,np.array([0],np.int32),subject_pbr)
    flash_ref={**ref,'gain':40.0}
    flash,_,_,_=prepare_photo_geometry({**config,'fixtures':{**fixture,
        'emitter_profiles':{'version':1,'records':[flash_ref]}}},
        subject,np.array([0],np.int32),subject_pbr,prepare_sources=False)
    _,_,roots=timed_light_buffers(baseline,flash)
    assert len(roots)==len(baseline[7]) and roots[:,10].sum()>baseline[7][:,10].sum()


def test_native_sample_count_changes_precision_not_total_power():
    scene=fixture()
    sums=[]
    for budget in (60,600):
        rows=native_emissive_sources(scene,[.45,.55,.65],ray_budget=budget)
        sums.append(rows[:,10].sum())
    np.testing.assert_allclose(sums,[30,30],rtol=3e-6)


def test_skinny_triangle_guard_uses_native_coordinate_precision():
    geom,shade,materials,triangles=fixture()
    triangles[0]=[[0,0,0],[.865,.65,.18],[.692001,.520002,.144001]]
    geom[0,:3]=triangles[0,0];geom[0,4:7]=triangles[0,1]-triangles[0,0]
    geom[0,8:11]=triangles[0,2]-triangles[0,0]
    normal=np.cross(triangles[0,1]-triangles[0,0],triangles[0,2]-triangles[0,0])
    geom[0,12:15]=normal/np.linalg.norm(normal)
    scene=(geom,shade,materials,triangles)
    rows=native_emissive_sources(scene,[.45,.55,.65],ray_budget=60)
    camera=area_nee_records(rows,*scene)
    area=np.linalg.norm(normal)/2
    assert camera[rows[:,7]==1,10].sum()==pytest.approx(area,rel=2e-6)
