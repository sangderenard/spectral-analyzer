import json
import math
import numpy as np
import pytest

from camera_designer.curve_cookie import CurveCookie, text_cookie


def square(a):
    p=[[-a,-a],[a,-a],[a,a],[-a,a],[-a,-a]]
    return [[p[i],p[i+1]] for i in range(4)]


def test_even_odd_hole_all_components_and_inverse():
    c=CurveCookie([square(.01),square(.005)],(.02,.02))
    assert not c.open_at(0,0)
    assert c.open_at(.007,0)
    assert not c.open_at(.015,0)
    assert math.isinf(c.intersect([.007,0,1],[0,0,-1])[0])
    assert c.intersect([0,0,1],[0,0,-1])[0]==1
    assert math.isinf(c.intersect([.03,0,1],[0,0,-1])[0])
    inverse=CurveCookie(c.contours,c.half_size,invert=True)
    assert inverse.open_at(0,0)
    assert not inverse.open_at(.007,0)
    assert math.isinf(inverse.intersect([.03,0,1],[0,0,-1])[0])


def test_quadratic_is_curved_not_control_polygon():
    c=CurveCookie([[[[-1,0],[0,2],[1,0]],[[1,0],[-1,0]]]],(2,2))
    assert c.open_at(0,.75)
    assert not c.open_at(0,1.1)
    assert not c.open_at(.7,.75)
    assert c.open_at(.7,.49)
    assert sum(len(loop) for loop in c.contours)==2


def test_ray_parameter_normals_parallel_and_behind():
    c=CurveCookie([square(.005)],(.02,.02),z_pos=.1)
    t,h,n=c.intersect([.01,0,.3],[0,0,-2])
    assert t==pytest.approx(.1);assert h==pytest.approx([.01,0,.1]);assert np.array_equal(n,[0,0,1])
    assert c.intersect([.01,0,-.3],[0,0,2])[2][2]==-1
    assert math.isinf(c.intersect([.01,0,.3],[0,1,0])[0])
    assert math.isinf(c.intersect([.01,0,.3],[0,0,1])[0])
    with pytest.raises(ValueError):c.intersect([0,0,math.nan],[0,0,1])


def test_cubic_tangent_partition_is_not_a_polyline():
    c=CurveCookie([[[[-1,0],[-1,2],[1,2],[1,0]],[[1,0],[-1,0]]]],(2,2))
    assert c.open_at(0,1.49)
    assert not c.open_at(0,1.51)
    assert c.open_at(.9,.5)
    assert not c.open_at(.9,1.2)
    assert len(c._pieces)==3  # two monotone cubic halves plus the baseline


def test_font_keeps_quadratic_curves_and_exact_source_identity():
    c=text_cookie('MEOW')
    assert any(len(segment)>2 for loop in c.contours for segment in loop)
    assert c.source['curve_conversion']=='none'
    assert len(c.source['font']['source_digest'])==64
    restored=CurveCookie.from_dict(json.loads(json.dumps(c.to_dict())))
    assert restored.to_dict()==c.to_dict()
    assert restored.glsl_intercept_fn('cookie')==c.glsl_intercept_fn('cookie')
    assert 'texture' not in c.glsl_intercept_fn()
    assert 'step<24' in c.glsl_intercept_fn()


@pytest.mark.parametrize('text',['','   ','$x$','line\nbreak','x'*65,'\U0010ffff'])
def test_invalid_or_missing_text_is_not_silently_substituted(text):
    with pytest.raises(ValueError):text_cookie(text)


def test_invalid_geometry_and_schema_refused():
    with pytest.raises(ValueError):CurveCookie([[[[0,0],[1,0]]]],(1,1))
    with pytest.raises(ValueError):CurveCookie([square(1)],(0,1))
    c=CurveCookie([square(1)],(2,2))
    with pytest.raises(ValueError):c.glsl_intercept_fn('bad;code')
    packet=c.to_dict();packet['fill_rule']='nonzero'
    with pytest.raises(ValueError):CurveCookie.from_dict(packet)
