"""Exact conditional spectral integration of the existing camera NEE estimator."""
import numpy as np
import pytest
from camera_designer.spectral_sensor_basis import fixed3_nee_records


def records():
    rows=np.zeros((6,12),np.float32)
    rows[:4,:3]=[0,1,2];rows[4:,:3]=[3,4,5]
    rows[:,3]=[1,2,3,4,5,6];rows[:,10]=[2,1,1,2,1,2]
    rows[:,8]=2.998e17/np.array([650,550,450,650,450,550])
    rows[:,4:7]=[0,0,1]
    return rows


def test_conditional_expectation_preserves_each_position_and_band():
    original=records();saved=original.copy();packed=fixed3_nee_records(original)
    np.testing.assert_array_equal(original,saved)
    np.testing.assert_array_equal(packed[:,:3],original[:,:3])
    expected=np.array([[10/4,2/4,3/4]]*4+[[0,12/2,5/2]]*2)
    np.testing.assert_allclose(packed[:,4:7],expected)
    # Geometry/shadow weights can be arbitrary functions of position.
    geometry=np.array([.7]*4+[0.2]*2)
    old=np.zeros((6,3));bands=np.array([0,1,2,0,2,1])
    old[np.arange(6),bands]=original[:,3]*original[:,10]
    np.testing.assert_allclose((old*geometry[:,None]).mean(0),
        (packed[:,4:7]*geometry[:,None]).mean(0),rtol=1e-7)
    assert np.all(np.var(packed[:,4:7]*geometry[:,None],axis=0)<=np.var(old*geometry[:,None],axis=0))


def test_white_source_has_no_wavelength_selection_noise():
    rows=np.zeros((300,12),np.float32);rows[:,3]=rows[:,10]=1
    rows[:,8]=np.tile(2.998e17/np.array([650,550,450]),100)
    out=fixed3_nee_records(rows)
    np.testing.assert_allclose(out[:,4:7],1/3,rtol=1e-7)
    assert np.max(np.ptp(out[:,4:7],axis=0))==0


def test_nearby_positions_are_never_merged():
    rows=records();rows[1,0]=1e-9
    out=fixed3_nee_records(rows)
    np.testing.assert_array_equal(out[1,4:7],[0,2,0])


def test_empty_and_invalid_records():
    assert fixed3_nee_records(np.zeros((0,12))).shape==(0,12)
    for bad in [np.zeros((3,11)),np.full((1,12),np.nan)]:
        with pytest.raises(ValueError):fixed3_nee_records(bad)
    bad=records();bad[0,8]=0
    with pytest.raises(ValueError):fixed3_nee_records(bad)
