import math
import numpy as np
import pytest
from camera_designer.curve_cookie import CurveCookie, text_cookie
from camera_designer.curve_filter_probe import CurveFilterProbe


def setup_filter(transmittance=(.1,.1,.1), **kwargs):
    p=[[-.005,-.005],[.005,-.005],[.005,.005],[-.005,.005],[-.005,-.005]]
    c=CurveCookie([[[p[i],p[i+1]] for i in range(4)]],(.02,.02))
    record={'material':0,'wavelengths_nm':[400,550,700],'transmittance':list(transmittance),
            'reference_thickness_mm':1.,'concentration':1.}
    pbr=np.zeros((1,11));pbr[0,5]=1.;pbr[0,6]=1.
    return CurveFilterProbe(c,record,pbr,thickness_m=kwargs.pop('thickness_m',.001),**kwargs)


def test_neutral_density_is_spectral_attenuation_not_rgb_mask():
    f=setup_filter()
    for wavelength in (.4,.48,.55,.62,.7):
        blocked=f.trace_normal([.01,0,.1],[0,0,-1],wavelength)
        assert blocked['transmittance']==pytest.approx(.1)
        assert blocked['shared_extinction_k']==pytest.approx(math.log(10)*wavelength*1e-6/(4*math.pi*.001))
        assert f.trace_normal([0,0,.1],[0,0,-1],wavelength)['transmittance']==1
    assert f.trace_normal([.03,0,.1],[0,0,-1],.55)['intersects'] is False


def test_thickness_concentration_and_spectral_dye_are_multiplicative():
    f=setup_filter((.2,.5,.8),thickness_m=.002,text_density=.5,background_density=2)
    for wavelength,T in zip((.4,.55,.7),(.2,.5,.8)):
        assert f.trace_normal([0,0,.1],[0,0,-2],wavelength)['transmittance']==pytest.approx(T)
        assert f.trace_normal([.01,0,.1],[0,0,-2],wavelength)['transmittance']==pytest.approx(T**4)
    # Shared helper interpolates absorption, hence geometric-mean T halfway.
    assert f.trace_normal([0,0,.1],[0,0,-1],.475)['transmittance']==pytest.approx(math.sqrt(.2*.5))


def test_unsupported_angle_spectrum_and_invalid_density_refuse():
    f=setup_filter()
    with pytest.raises(ValueError,match='normal incidence'):f.trace_normal([0,0,.1],[.1,0,-1],.55)
    with pytest.raises(ValueError,match='outside authored'):f.trace_normal([0,0,.1],[0,0,-1],.8)
    with pytest.raises(ValueError):setup_filter(text_density=-1)
    with pytest.raises(ValueError):setup_filter(thickness_m=0)
    assert f.trace_normal([0,0,.1],[0,0,1],.55)['intersects'] is False


def test_real_text_cookie_nd_uses_curve_regions():
    f=setup_filter();f.cookie=text_cookie('MEOW')
    values=[]
    for x in np.linspace(-f.cookie.half_size[0],f.cookie.half_size[0],23):
        for y in np.linspace(-f.cookie.half_size[1],f.cookie.half_size[1],9):
            result=f.trace_normal([x,y,.1],[0,0,-1],.55)
            assert result['transmittance']==pytest.approx(1 if f.cookie.open_at(x,y) else .1)
            values.append(result['transmittance'])
    assert min(values)==pytest.approx(.1) and max(values)==1
