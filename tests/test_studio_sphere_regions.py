import numpy as np
import pytest

from camera_designer.parametric_surfaces import SphericalSurface
from camera_designer.studio_sphere_regions import (
    STUDIO_SPHERE_GLSL, geometry_statistics, pack_sphere_regions,
    validate_sphere_regions,
)


def octahedron():
    points = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], np.float32)
    faces = []
    for x in (0,1):
        for y in (2,3):
            for z in (4,5):
                tri = points[[x,y,z]].copy()
                if np.dot(np.cross(tri[1]-tri[0],tri[2]-tri[0]),tri[0]) < 0:
                    tri = tri[[0,2,1]]
                faces.append(tri)
    return np.array(faces)


def test_original_kernel_is_used_without_triangle_normal_smoothing():
    kernel = SphericalSurface(R=1., r_max=1.).glsl_intercept_fn('studio_original_unit_sphere', min_t_argument=True)
    assert kernel in STUDIO_SPHERE_GLSL
    assert 'if (tt > min_t)' in kernel
    assert 'if (tt > 1e-6)' in SphericalSurface().glsl_intercept_fn()
    assert 'position-studio_spheres[si].center_radius.xyz' in STUDIO_SPHERE_GLSL


@pytest.mark.parametrize('origin,direction,expected', [
    ([0,0,-2],[0,0,1],1), ([0,0,0],[0,0,1],1),
    ([1,0,-2],[0,0,1],2), ([1.01,0,-2],[0,0,1],np.inf),
    ([.99,0,-2],[0,0,1],2-np.sqrt(1-.99**2)),
])
def test_original_parametric_oracle_handles_full_sphere(origin,direction,expected):
    # Translate world-center zero into the original optical-local sphere
    # centered at (0,0,R). Its exact hit includes the coarse-mesh silhouette.
    surface = SphericalSurface(R=1., r_max=1.)
    t,_,_ = surface.intersect(np.asarray(origin)+[0,0,1],np.asarray(direction))
    if np.isinf(expected):
        assert np.isinf(t)
    else:
        assert t == pytest.approx(expected,abs=2e-14)


def test_pack_replaces_only_owned_shell_and_preserves_medium_material_identity():
    shell = octahedron()
    interior = np.array([[[-.2,-.2,0],[.2,-.2,0],[0,.2,0]]],np.float32)
    verts = np.concatenate([interior,shell])
    ids = np.array([1]+[0]*8)
    pbr = np.zeros((2,16),np.float32)
    metadata = {'version':1,'spheres':[{'start':1,'count':8,'center':[0,0,0],'radius':1}]}
    validated = validate_sphere_regions(metadata,verts,ids,pbr)
    geom = np.zeros((9,16),np.float32);geom[:,:3]=verts[:,0]
    geom[:,4:7]=verts[:,1]-verts[:,0];geom[:,8:11]=verts[:,2]-verts[:,0]
    geom[1:,3]=1;geom[1:,7]=2;geom[0,11]=4097
    shade = np.zeros_like(geom);shade[1:,14]=np.uint32(5).view(np.float32)
    scene = (geom,shade,np.zeros((32,12),np.float32),verts,np.zeros((0,8),np.float32),{},
             (np.array([-1,-1,-1]),np.array([1,1,1])),np.zeros((0,12),np.float32))
    packed,regions = pack_sphere_regions(scene,np.arange(9),0,validated,scale_m=1,center_m=[0,0,0])
    assert len(packed[0]) == 2 and regions[0,4] == 1
    np.testing.assert_array_equal(packed[3][0],interior[0])
    np.testing.assert_array_equal(packed[0][1,[3,7]],[1,2])
    assert packed[0][0,11] == 4097
    assert packed[1][1,14].view(np.uint32) == 5
    assert np.all(packed[3][1] == 0)  # metadata owner is not a second surface
    assert len(geom) == 9  # input remains untouched


def test_region_validation_does_not_infer_identity_or_ignore_material_variation():
    vertices=octahedron();pbr=np.zeros((2,16));ids=np.zeros(8,int)
    metadata={'version':1,'spheres':[{'start':0,'count':8,'center':[0,0,0],'radius':1}]}
    with pytest.raises(ValueError,match='metadata'):
        validate_sphere_regions(None,vertices,ids,pbr)
    ids[3]=1
    with pytest.raises(ValueError,match='uniform material'):
        validate_sphere_regions(metadata,vertices,ids,pbr)
    ids[:]=0;pbr[0,8]=1
    with pytest.raises(ValueError,match='emissive'):
        validate_sphere_regions(metadata,vertices,ids,pbr)
    pbr[:]=0;vertices[0,0,0]+=.01
    with pytest.raises(ValueError,match='do not match'):
        validate_sphere_regions(metadata,vertices,ids,pbr)


def test_budget_counts_actual_triangles_not_analytic_metadata_records():
    stats=geometry_statistics(input_triangles=90000,packed_triangles=1009,sphere_count=9,policy='exact',budget=1000)
    assert stats['effective_triangles']==1000 and stats['analytic_spheres']==9
    with pytest.raises(ValueError,match='Scene needs 1001 triangles'):
        geometry_statistics(input_triangles=90000,packed_triangles=1010,sphere_count=9,policy='exact',budget=1000)
    with pytest.raises(ValueError,match='integer'):
        geometry_statistics(input_triangles=1,packed_triangles=1,sphere_count=0,policy='mesh',budget=True)
