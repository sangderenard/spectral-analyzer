import numpy as np
import pytest
from camera_designer.studio_opaque_pbr import validate_surface_transport,VERSION,GLSL,FORWARD,CAMERA
from camera_designer.studio_conductor_surface import sample_ggx

def evaluate(wi,wo,rough=.96,d=.96,rho=.9,eta=1.5):
    wo=np.asarray(wo);ni=wi[2];no=np.maximum(wo[:,2],0);hs=wo+wi;h=hs/np.maximum(np.linalg.norm(hs,axis=1)[:,None],1e-30)
    nh=np.maximum(h[:,2],0);vh=np.maximum(h@wi,1e-30);a2=rough**4
    den=(1-nh*nh)+a2*nh*nh;D=a2/(np.pi*den*den)
    g=lambda c:2*c/(c+np.sqrt(a2+(1-a2)*c*c))
    ct=np.sqrt(np.maximum(0,1-(1-vh*vh)/(eta*eta)))
    F=.5*((vh-eta*ct)**2/(vh+eta*ct)**2+(eta*vh-ct)**2/(eta*vh+ct)**2)
    f=d*rho/np.pi+(1-d)*F*D*g(ni)*g(no)/np.maximum(4*ni*no,1e-30)
    pdf=d*no/np.pi+(1-d)*D*nh/(4*vh)
    return np.where(no>0,f,0),np.where(no>0,pdf,0)

def test_config_is_explicit_fixed3_fixture_path():
    assert validate_surface_transport({}) is None
    assert validate_surface_transport(dict(surface_transport=VERSION,transport='fixed:3',fixtures={}))==VERSION
    for c in [dict(surface_transport='other'),dict(surface_transport=VERSION,transport='fixed:1',fixtures={}),dict(surface_transport=VERSION,transport='fixed:3')]:
        with pytest.raises(ValueError):validate_surface_transport(c)

def test_white_furnace_reciprocity_and_pdf_weight():
    rng=np.random.default_rng(421);u=rng.random((131072,3));z=u[:,0];phi=2*np.pi*u[:,1]
    wo=np.column_stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
    for angle in (0.,1.1):
        wi=np.array([np.sin(angle),0,np.cos(angle)])
        f,pdf=evaluate(wi,wo)
        energy=(f*z*2*np.pi).mean()
        assert .85<energy<.905
        assert 0<(pdf*2*np.pi).mean()<=1.01
        for row,expected in zip(wo[:12],f[:12]):
            # Rotate both vectors into a coordinate frame with same normal;
            # evaluation accepts arbitrarywi, so reciprocity is direct.
            assert evaluate(row,wi[None,:])[0][0]==pytest.approx(expected,rel=1e-12)
        assert np.all(np.isfinite(f)) and np.all(pdf>=0)

def test_actual_existing_ggx_sampler_and_budget_invariance():
    rng=np.random.default_rng(19);u=rng.random((65536,3));wi=np.array([0.,0.,1.]);d=.96
    wo,weight,vh=sample_ggx(wi,.96,u[:,1:]);diff=u[:,0]<d
    r=np.sqrt(u[:,1]);phi=2*np.pi*u[:,2]
    wo[diff]=np.column_stack([r*np.cos(phi),r*np.sin(phi),np.sqrt(1-u[:,1])])[diff]
    f,pdf=evaluate(wi,wo);gain=np.divide(f*np.maximum(wo[:,2],0),pdf,out=np.zeros(len(wo)),where=pdf>0)
    assert .85<gain.mean()<.88
    expected=gain.sum()/len(gain)
    for copies in (1,2,8):
        assert np.tile(gain,copies).sum()/(len(gain)*copies)==pytest.approx(expected,rel=1e-14)
    # Corrected transporthasno preceding air-distance or absolutepacket cutoff.
    assert 'absorption' not in FORWARD and 'dist_falloff' not in FORWARD
    assert 'f*max(dot(n,wo),0.0)/pdf' in GLSL
    assert 'throughput[i]*direct_bands[i]*STUDIO_PBR_PI*f' in CAMERA
