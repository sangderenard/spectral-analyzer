"""Bounded real-tracer subject ROI proof; only run in a released GPU window."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import demo_pluck_gl as r
    from OpenGL import GL
    from camera_designer.film_response import build_film_stack,normalize_film_response
    from camera_designer.studio_directional_field import ROOT_ESTIMATOR,readback,DirectionalUnavailable,subject_field_region
    from camera_designer.studio_forward_field import ForwardFieldPublisher
    from camera_designer.marble_color_profile import XYZ_TO_SRGB,_LINE_TO_XYZ
    from camera_designer.studio_volume_extinction import ExtinctionGrid
    r._ensure_worker_gl_context()
    tri=np.array([[[-.01,-.01,0],[.01,-.01,0],[0,.01,0]]],np.float32)
    g=np.zeros((1,16),np.float32);g[:,:3]=tri[:,0];g[:,4:7]=tri[:,1]-tri[:,0];g[:,8:11]=tri[:,2]-tri[:,0];g[:,14]=1
    s=np.zeros((1,16),np.float32);s[:,:4]=[.2,0,1,1.5];s[:,4:8]=[.2,0,1,1];s[:,8:11]=1
    src=np.zeros((3,12),np.float32);src[:,:3]=[0,0,.4];src[:,3]=1;src[:,4:7]=[0,0,-1];src[:,8]=2.998e17/np.array([650,550,450]);src[:,10]=1
    nodes,ids=r._build_gpu_bvh(tri);mat=np.zeros((32,12),np.float32);ctx=np.zeros((0,8),np.float32)
    bounds=([-.5]*3,[.5]*3);pbr=np.zeros((1,16),np.float32);pbr[:,:3]=1;pbr[:,3]=.5;pbr[:,6]=1.5;pbr[:,7]=1
    film=build_film_stack(r.FILM_STACKS['em_rgb_fixed3'],normalize_film_response())
    output=Path(sys.argv[1]) if len(sys.argv)>1 else None
    roi_config=dict(scale_m=1,object_center_m=[0,0,0],light_field_region="subject",light_field_cell_size_m=.0005,light_field_margin_m=.001,geometry_sha256="b"*64)
    roi=subject_field_region(tri,roi_config)
    checks=[]
    for grid in [64]:
      baseline=None
      for enabled in [False,True]:
        state=r._gpu_ray_field_prebuilt(g,s,mat,nodes,ids,ctx,src,bounds,dims=(8,8,8),dispatch_batch=128,max_bounces=1,initial_rays=0,film_stack=film,native_pbr=pbr,
            record_directional=enabled,light_field_estimator=ROOT_ESTIMATOR,light_field_grid_size=grid,light_field_region=roi)
        try:
          r._clear_prebuilt(state);r._gpu_pump_prebuilt(state,n_rays=3);GL.glFinish()
          scalar=r._gpu_readback_prebuilt(state['tex_bands'],state['dims'])
          if not enabled:baseline=scalar;continue
          np.testing.assert_array_equal(scalar,baseline)
          record=state['directional_record'];first=readback(record)
          assert first.shape==(grid,grid,grid,12,6) and np.any(first[...,:3])
          for _ in range(3):r._gpu_pump_prebuilt(state,n_rays=3)
          np.testing.assert_allclose(readback(record),first,rtol=1e-6,atol=1e-6)
          assert record['root_sampling']['completed_root_sweeps']==4
          r._clear_prebuilt(state);r._gpu_pump_prebuilt(state,n_rays=2)
          try:readback(record);raise AssertionError('partial coverage falsely available')
          except DirectionalUnavailable:pass
          r._gpu_pump_prebuilt(state,n_rays=7)
          uneven=readback(record)
          # Integer rounding differs after reciprocal3 versus reciprocal2.
          # Per-band error <=(3*.5+.5) counts, transformed into RGB density.
          bound=2*np.max(np.sum(abs(XYZ_TO_SRGB@_LINE_TO_XYZ),axis=1))/65535/record['voxel_volume_m3']
          np.testing.assert_allclose(uneven[...,:3],first[...,:3],rtol=0,atol=bound+1e-5)
          assert record['root_sampling']['excluded_partial_root_jobs']==2
          assert record['root_sampling']['normalization_divisor']==1
          record['material_lineage']=[(0,pbr[0].copy())];state['geometry_sha256']='b'*64
          if output:
            folder=output/str(grid);folder.mkdir(parents=True,exist_ok=True)
            metadata=ForwardFieldPublisher(folder,'a'*64).snapshot(r,state,now=0,ray_jobs=7)
            assert metadata['schema']=='studio.forward-field.v5'
            assert metadata['bounds_min_m']==[-.5]*3
            assert metadata['directional']['cell_size_m']==.0005
            assert metadata['directional']['bounds_min_m']==roi['bounds_min_m']
            (folder/'status.json').write_text(json.dumps({'state':'complete','digest':'a'*64,'light_field':metadata}))
            (folder/'config.json').write_text(json.dumps(dict(**roi_config,receive_light_field=True,transport='fixed:3',light_field_estimator=ROOT_ESTIMATOR,light_field_grid_size=grid,light_field_stencil=12,light_field_schema='studio.forward-field.v5',light_field_subject_region=roi)))
          scalar_before=r._gpu_readback_prebuilt(state['tex_bands'],state['dims'])
          r._update_prebuilt_geometry(state,g,s,mat,nodes,ids,ctx,bounds,np.zeros((0,8),np.float32),native_pbr=pbr)
          np.testing.assert_array_equal(r._gpu_readback_prebuilt(state['tex_bands'],state['dims']),scalar_before)
          assert record['root_sampling']['completed_root_sweeps']==0
          try:readback(record);raise AssertionError('geometry update kept old root average')
          except DirectionalUnavailable:pass
          checks.append(dict(grid=grid,result='PASS',root_sampling=metadata['directional']['root_sampling'] if output else None))
          del first,uneven
        finally:r._destroy_prebuilt(state)
    # Optional supplied extinction uses the same root weighting and64DDA hook.
    state=r._gpu_ray_field_prebuilt(g,s,mat,nodes,ids,ctx,src,bounds,dims=(8,8,8),dispatch_batch=128,max_bounces=1,initial_rays=0,film_stack=film,native_pbr=pbr,record_directional=True,
        light_field_estimator=ROOT_ESTIMATOR,light_field_grid_size=64,light_field_region=roi,volume_extinction=ExtinctionGrid(np.full((1,1,1),.5,np.float32),np.array(bounds)))
    try:
      r._clear_prebuilt(state);r._gpu_pump_prebuilt(state,n_rays=3);a=readback(state['directional_record'])
      r._gpu_pump_prebuilt(state,n_rays=3);np.testing.assert_allclose(readback(state['directional_record']),a,rtol=1e-6,atol=1e-6)
      assert np.any(a[...,:3]);checks.append(dict(optional_extinction='PASS'))
    finally:r._destroy_prebuilt(state)
    print(json.dumps(dict(renderer=GL.glGetString(GL.GL_RENDERER).decode(),result='PASS',checks=checks)))

if __name__=='__main__':main()
