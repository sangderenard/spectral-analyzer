"""Real native source-component tests; CPU only, no service or GPU."""
import numpy as np
import pytest
pytest.importorskip('_spectral_kernels')
from camera_designer.studio_native_emitters import native_emissive_sources, area_nee_records
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
