import numpy as np
import pytest
from spectral_library import load_default_library
from material_db import EmissionProfileDatabase,_spd_from_spread,_WL_NM
from camera_designer.studio_profile_absorption import color_profile_transmittance
from camera_designer.studio_optical_media import optical_extinction

def test_original_profile_samples_not_rgb_and_beer_length():
    load_default_library();ep=EmissionProfileDatabase.instance()
    record=color_profile_transmittance(material=3,profile_name='soda_lime_glass',wavelengths_nm=[450,550,650],reference_thickness_mm=2,interpretation='authored_internal_transmittance')
    original=_spd_from_spread(ep.get(ep.index_of('soda_lime_glass')).spread.frequency)
    assert np.allclose(record['transmittance'],np.interp([450,550,650],_WL_NM,original))
    for w,t in zip(record['wavelengths_nm'],record['transmittance']):
        k=optical_extinction(record,w/1000)
        assert np.exp(-4*np.pi*k*.002/(w*1e-9))==pytest.approx(t)
        assert np.exp(-4*np.pi*k*.004/(w*1e-9))==pytest.approx(t*t)

def test_no_automatic_material_meaning_or_reference_length():
    load_default_library()
    kwargs=dict(material=0,profile_name='borosilicate',wavelengths_nm=[450,650],reference_thickness_mm=1,interpretation='surface_color')
    with pytest.raises(ValueError,match='explicit'):color_profile_transmittance(**kwargs)
    kwargs.update(interpretation='authored_internal_transmittance',reference_thickness_mm=0)
    with pytest.raises(ValueError,match='reference thickness'):color_profile_transmittance(**kwargs)
