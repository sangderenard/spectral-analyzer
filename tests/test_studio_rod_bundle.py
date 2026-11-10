import math
import numpy as np
import pytest
from camera_designer.studio_rod_bundle import AxialRodBundle


def test_partition_is_spatial_and_only_actual_participants_compete():
    b=AxialRodBundle((4,7),.75,.18,.9)
    y=.3;t=y/.75;c=.18*(1-t*t);a=.9*t
    p=np.array([c+.2*math.cos(a),y,.2*math.sin(a)])
    assert b.owner(p,{4,7})==4
    assert b.owner(p,{7})==7
    assert b.owner(p,{99}) is None
    assert b.owner(p,{7,4})==4


def test_exact_bisector_surface_and_normal_follow_curve_and_twist():
    b=AxialRodBundle((1,2,3,4),.75,.18,.9,.2)
    for y in (-.5,0,.4):
        for angle in b.divider_angles():
            a=.2+.9*y/.75+angle;c=.18*(1-(y/.75)**2)
            p=np.array([c+.3*math.cos(a),y,.3*math.sin(a)])
            value,grad=b.divider_value_gradient(p,angle)
            assert abs(value)<1e-14
            numeric=[]
            for axis in np.eye(3):
                numeric.append((b.divider_value_gradient(p+axis*1e-6,angle)[0]-b.divider_value_gradient(p-axis*1e-6,angle)[0])/2e-6)
            np.testing.assert_allclose(grad,numeric,atol=1e-9)


def test_two_rod_cut_really_changes_owner_with_both_old_volumes_active():
    b=AxialRodBundle((1,2),1.,0.,0.)
    assert b.owner([.1001,0,.1],{1,2})==1
    assert b.owner([.0999,0,.1],{1,2})==2
    # Existing original boundary crossings alone cannot capture this interface.
    assert b.divider_value_gradient([.1,0,.1],math.pi/4)[0]==pytest.approx(0,abs=1e-15)


def test_same_authored_ids_are_required():
    with pytest.raises(ValueError):AxialRodBundle((1,1),1,0,0)


def test_divider_intersection_is_a_real_internal_boundary_hit():
    b=AxialRodBundle((1,2),1.,0.,0.)
    hits=b.divider_intersections([.2,0,.1],[-1,0,0],math.pi/4,0,.2)
    assert len(hits)==1
    assert hits[0][0]==pytest.approx(.1,abs=1e-10)
    assert b.owner(hits[0][1]+[1e-5,0,0],{1,2})==1
    assert b.owner(hits[0][1]-[1e-5,0,0],{1,2})==2


def test_twisted_divider_isolates_multiple_crossings_without_fixed_sampling():
    b=AxialRodBundle((1,2),1.,0.,8.)
    hits=b.divider_intersections([.1,-1,0],[0,1,0],0,0,2)
    expected=[1+k*math.pi/8 for k in range(-2,3)]
    np.testing.assert_allclose([h[0] for h in hits],expected,atol=1e-9)


def test_coincident_ray_fails_explicitly_instead_of_silent_missing_interface():
    b=AxialRodBundle((1,2),1.,0.,0.)
    with pytest.raises(ValueError,match='unresolved|work bound'):
        b.divider_intersections([0,0,0],[1,0,0],0,0,1,max_intervals=100)

def test_divider_gpu_kernel_compiles_offline(tmp_path):
    import shutil, subprocess
    from camera_designer.studio_rod_bundle_gpu import DIVIDER_GLSL
    compiler=shutil.which('glslangValidator')
    if compiler is None:pytest.skip('offline Vulkan GLSL compiler unavailable')
    p=tmp_path/'bundle.comp'
    p.write_text('#version 430 core\nlayout(local_size_x=1) in;\n'+DIVIDER_GLSL+'\nvoid main(){float t;vec3 n;int status=studio_divider_root(vec3(.2,0,.1),vec3(-1,0,0),.785398,vec4(1,0,0,0),0.,.2,1e-7,t,n);}')
    subprocess.run([compiler,'-S','comp',str(p)],check=True,timeout=30)
