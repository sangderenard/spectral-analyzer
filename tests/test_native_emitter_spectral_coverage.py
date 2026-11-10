"""Regression for complete native emitter wavelength coverage (CPU, no GL)."""
import time
import numpy as np
import pytest
kernels = pytest.importorskip('_spectral_kernels')
from ray_tracer_bridge import per_tri_spectral_to_mat_buf

from bdpt_records import BDPT_SPECTRAL_RECORD_DTYPE as SPECTRAL_WEIGHT

@pytest.mark.parametrize('bands',[1,3,4,8,32])
def test_native_emitter_visits_every_authored_band(bands):
    # One half-square-metre source, no receiver or sensor. Inspect actual
    # native root records, not rendered RGB or a duplicated sampling formula.
    freq=299_792_458.0/np.linspace(680e-9,440e-9,bands)
    zero=np.zeros((1,bands));emission=np.linspace(.5,1.5,bands)[None,:]
    ids,materials,count=per_tri_spectral_to_mat_buf(zero,zero,zero,freq,emission_bands=emission)
    tr=kernels.RayTracer(1,np.array([[0.,0.,0.,0.,1.,0.,0.,0.,1.]]),
        np.array([[1.,0.,0.]]),ids,materials,count,freq,299_792_458.,np.zeros(bands))
    try:
        tr.set_tri_ior(0,1,int(kernels.MAT_FLAG_EMISSIVE))
        rays=bands*8
        assert tr.submit_emissive_triangles(np.array([0],np.int32),rays,
            max_bounces=1,min_amplitude=0.,max_children=1,seed=17,
            use_gpu_compute=False,gpu_all_stages=False)==rays
        deadline=time.monotonic()+5
        while tr.in_flight_count() and time.monotonic()<deadline:time.sleep(.001)
        assert tr.in_flight_count()==0
        raw=np.asarray(tr.drain_bdpt_spectral(rays*bands*2))
        records=np.frombuffer(raw.tobytes(),dtype=SPECTRAL_WEIGHT)
        roots=records[(records['vertex_index']==0)&((records['subpath_id']>>31)==0)]
        counts=[];totals=[]
        for band in range(bands):
            lane=roots[roots['band_id']==band]
            beta=np.hypot(lane['beta_re'],lane['beta_im'])
            counts.append(np.count_nonzero(beta));totals.append(beta.sum())
        assert counts==[8]*bands, counts
        np.testing.assert_allclose(totals,.5*emission[0],rtol=2e-6,atol=1e-8)
    finally:tr.stop_pipeline()
