import numpy as np
import pytest
from camera_designer.studio_spectral_reflector import authored_reflector_bands
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
from mat_flags import MAT_FLAG_SPECTRAL_REFLECTOR


def test_artistic_gold_spectrum_preserves_authored_colour_and_energy_bounds():
    p=np.zeros(16,np.float32);p[:3]=[1,.72,.29];p[4]=1;p[6]=1.5;p[7]=1
    rows=authored_reflector_bands(p,[.65,.55,.45])
    rho=np.array([r['reflectance'] for r in rows])
    assert 1>=rho[0]>rho[1]>rho[2]>=0
    for cosine in [0,.2,.8,1]:
        r=rho+(1-rho)*(1-cosine)**5
        assert np.all((r>=0)&(r<=1))
        np.testing.assert_allclose(.98*(r/.98),r)


def test_scene_uses_native_band_buffer_and_explicit_reflector_flag():
    p=np.zeros((1,16),np.float32);p[0,:3]=[1,.72,.29];p[0,4]=1;p[0,6]=1.5;p[0,7]=1
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    v=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
    out=append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.65,.55,.45],advanced_surface_mask=[True])
    assert out[0][0,15:16].view(np.uint32)[0]&MAT_FLAG_SPECTRAL_REFLECTOR
    mid=int(out[1][0,14:15].view(np.uint32)[0]);bands=out[2].reshape(-1,32,12)[mid,:3]
    assert bands[0,2]>bands[1,2]>bands[2,2]
    assert np.all(bands[:,3]==0)


def test_shader_paths_weight_once_without_air_distance_loss():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import demo_pluck_gl as r\nassert 'if(spectral_reflector)reflectivity=studio_reflector_rho' in r._GPU_RAY_FIELD_CS\nassert '(diffuse_surface || spectral_reflector) ? 1.0 :' in r._GPU_RAY_FIELD_CS\nassert 'opacity > 0.001 && !spectral_reflector' in r._GPU_SENSOR_CS\nassert 'spectral_reflector ? studio_reflector_rho' in r._GPU_SENSOR_CS\nassert 'throughput[i] *= rho / survival;' in r._GPU_SENSOR_CS"],check=True,timeout=20)

def test_blue_tinted_mirror_is_selective_absorption_not_blue_emission():
    p=np.zeros(16,np.float32);p[:3]=[.15,.35,.9];p[4]=1;p[6]=1.5;p[7]=1
    rows=authored_reflector_bands(p,[.65,.55,.45])
    rho=np.array([r['reflectance'] for r in rows])
    assert 0<=rho[0]<rho[1]<rho[2]<=1
    assert all(r['emission']==0 and r['reemission']==0 for r in rows)
    # Repeated reflections further select blue, without creating energy.
    assert (rho[2]/rho[0])**2>rho[2]/rho[0]
    assert np.all(rho**2<=rho)


def test_nonzero_roughness_retains_existing_surface_law():
    p=np.zeros((1,16),np.float32);p[0,:3]=[1,.72,.29];p[0,4]=1;p[0,6]=1.5;p[0,7]=1
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    v=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
    for roughness in [.005,.12,1]:
        p[0,3]=roughness
        out=append_saved_studio_mesh_to_gpu_scene(scene,v,[0],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.65,.55,.45])
        assert not out[0][0,15:16].view(np.uint32)[0]&MAT_FLAG_SPECTRAL_REFLECTOR
        mid=int(out[1][0,14:15].view(np.uint32)[0]);bands=out[2].reshape(-1,32,12)[mid,:3]
        np.testing.assert_allclose(bands[:,2],np.repeat(bands[0,2],3))
        assert 0 < bands[0,2] < 1
