"""Actual production forward field: direct aperture vs metal-wall reflection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import demo_pluck_gl as r
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene

def plane(x,y0,y1):
    a=np.array([[x,y0,-.5],[x,y1,-.5],[x,y1,.5],[x,y0,.5]],np.float32)
    return a[[[0,1,2],[0,2,3]]]

def run(roughness=0,advanced=False):
 r._ensure_worker_gl_context()
 v=np.concatenate([plane(0,-.5,.5),plane(-.9,-1,2),plane(.9,-1,2)])
 empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
 outputs={};direct_fields={};spread=[]
 for colour,rgb in [('white',[1,1,1]),('gold',[1,.72,.29]),('blue',[.15,.35,.9])]:
  p=np.zeros((2,16),np.float32);p[:,6]=1.5;p[:,7]=1;p[0,:3]=rgb;p[0,3]=roughness;p[0,4]=1
  pack=append_saved_studio_mesh_to_gpu_scene(empty,v,[0,0,1,1,1,1],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.65,.55,.45],advanced_surface_mask=[advanced,advanced,False,False,False,False])
  nodes,ids=r._build_gpu_bvh(v)
  outputs[colour]={}
  for kind,y in [('reflected',0),('direct',.8)]:
   rows=[]
   for w in [.65,.55,.45]:
    src=np.zeros((1,12),np.float32);src[0,:3]=[-.5,y,0];src[0,3]=1;src[0,4:7]=[1,.25,0];src[0,8:11]=[299792458/(w*1e-6),0,1]
    state=r._gpu_ray_field_prebuilt(pack[0],pack[1],pack[2],nodes,ids,pack[4],src,(np.array([-1,-1,-1]),np.array([1,2,1])),dims=(64,64,8),max_bounces=2,initial_rays=128,film_stack=r.FILM_STACKS['em_rgb_fixed3'],native_pbr=__import__('material_db').MaterialDatabase.instance().build_tensors()['pbr'])
    assert state is not None
    try:
     out=r._gpu_readback_prebuilt(state['tex_bands'],(64,64,8))
     # Reflected receiver leg lies x<-.5, y>0.25. Incoming segment cannot enter.
     value=out[:,:,27:40,:15].sum() if kind=='reflected' else out.sum()
     if kind=='reflected' and colour=='white':spread.append(int(np.count_nonzero(out[:,:,:,:15])))
     if kind=='direct':
      if colour=='white':direct_fields[w]=out.copy()
      else:np.testing.assert_array_equal(out,direct_fields[w])
     assert value>0,(colour,kind,w)
     rows.append(float(value))
    finally:r._destroy_prebuilt(state)
   outputs[colour][kind]=rows
 white=outputs['white']
 for colour in ['gold','blue']:
  np.testing.assert_array_equal(outputs[colour]['direct'],white['direct'])
  ratios=np.array(outputs[colour]['reflected'])/white['reflected']
  assert np.all((ratios>0)&(ratios<=1)),ratios
  assert ratios[0]>ratios[1]>ratios[2] if colour=='gold' else ratios[0]<ratios[1]<ratios[2]
  print(colour,'reflected spectral ratio',ratios.tolist(),'direct unchanged',flush=True)
 print('roughness',roughness,'receiver occupied bins',spread,flush=True)
 print('PASS actual forward shader hit -> reflection -> receiver field; no emitter tint',flush=True)
 return spread
if __name__=='__main__':
 a=run(0,True);b=run(.4,True);assert all(y>x for x,y in zip(a,b)),(a,b)
