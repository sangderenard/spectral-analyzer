"""Actual camera shader reflection-to-emitter test; no injected radiance field."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from OpenGL import GL
import demo_pluck_gl as r
from material_db import MaterialDatabase
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene

def quad(z):
 a=np.array([[-2,-2,z],[2,-2,z],[2,2,z],[-2,2,z]],np.float32)
 return a[[[0,1,2],[0,2,3]]]

def run():
 r._ensure_worker_gl_context();results={}
 for roughness in [0,.4]:
  for name,rgb in [('white',[1,1,1]),('gold',[1,.72,.29]),('blue',[.15,.35,.9]),('native_lowk',[1,1,1])]:
   empty=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
   p=np.zeros((2,16),np.float32);p[:,6]=1.5;p[:,7]=1;p[0,:3]=rgb;p[0,3]=roughness;p[0,4]=1;p[1,8:11]=1
   v=np.concatenate([quad(0),quad(.8)])
   authoring=None
   if name=='native_lowk':
    from canonical_material_documents import MaterialDocument
    text=f'metallic: 1\nroughness: {roughness}\nopacity: 1\ntransmission: 0\nior: 1.5\nspectral_bands:\n'
    for w in [.65,.55,.45]:
     text+=f'- center_hz: {299792458/(w*1e-6)}\n  bandwidth_hz: 1e12\n  reflectance: 0.9\n  transmittance: 0\n  ior_real: 1.5\n  ior_imag: 0.01\n'
    doc=MaterialDocument('synthetic_low_k','test',text.encode())
    authoring={'version':1,'records':[{'material':0,'source':doc.snapshot(),'overrides':[]}]}
   pack=append_saved_studio_mesh_to_gpu_scene(empty,v,[0,0,1,1],p,scale_m=1,center_m=[0,0,0],wavelengths_um=[.65,.55,.45],spectral_emission_rgb=True,advanced_surface_mask=[True,True,False,False],material_authoring=authoring)
   nodes,ids=r._build_gpu_bvh(v)
   state=r._gpu_ray_field_prebuilt(pack[0],pack[1],pack[2],nodes,ids,pack[4],np.zeros((0,12),np.float32),([-2,-2,-1],[2,2,1]),dims=(8,8,8),max_bounces=3,initial_rays=0,film_stack=r.FILM_STACKS['em_rgb_fixed3'],native_pbr=MaterialDatabase.instance().build_tensors()['pbr'])
   acc=None
   try:
    if name=='white' and roughness==0:
     try:
      rejected=r.SensorAccumulator(state['ssbo'][0],state['ssbo'][1],state['ssbo'][2],state['ssbo'][9],state['ssbo'][10],len(v),len(nodes),state['tex_bands'],[-2,-2,-1],[2,2,1],(8,8,8),[0,0,.5],[0,0,0],32,32,rows_per_frame=32,fov_deg=20,max_bounces=3,ray_field_scale=0,vol_alpha=0,n_sources=0)
      rejected.destroy();raise AssertionError('missing native PBR accepted')
     except ValueError as error:assert 'canonical native PBR' in str(error)
    acc=r.SensorAccumulator(state['ssbo'][0],state['ssbo'][1],state['ssbo'][2],state['ssbo'][9],state['ssbo'][10],len(v),len(nodes),state['tex_bands'],[-2,-2,-1],[2,2,1],(8,8,8),[0,0,.5],[0,0,0],32,32,rows_per_frame=32,fov_deg=20,max_bounces=3,ray_field_scale=0,vol_alpha=0,n_sources=0,ssbo_native_pbr=state['native_pbr_ssbo'])
    assert acc._active;acc.film=r.FILM_STACKS['em_rgb_fixed3'];acc._sprinkle=False;acc.clear()
    for _ in range(16):acc.tick()
    GL.glFinish();bands=[]
    for tex in acc._textures:
     GL.glBindTexture(GL.GL_TEXTURE_2D,tex);image=np.empty((32,32,4),np.float32);GL.glGetTexImage(GL.GL_TEXTURE_2D,0,GL.GL_RGBA,GL.GL_FLOAT,image)
     assert np.isfinite(image).all();bands.append(float(np.mean(image[:,:,0]/np.maximum(image[:,:,3],1))))
    assert min(bands)>0,(roughness,name,bands);results[roughness,name]=np.array(bands)
   finally:
    if acc:acc.destroy()
    r._destroy_prebuilt(state)
  nk_ratio=results[roughness,'native_lowk']/results[roughness,'white']
  np.testing.assert_allclose(nk_ratio,.04001536,atol=.001)
  print('native low-k uses complex Fresnel, not authored .9 reflectance:',nk_ratio.tolist(),flush=True)
  for name in ['gold','blue']:
   ratio=results[roughness,name]/results[roughness,'white']
   assert np.all((ratio>0)&(ratio<=1.01))
   assert ratio[0]>ratio[1]>ratio[2] if name=='gold' else ratio[0]<ratio[1]<ratio[2]
   print('camera',roughness,name,'relative RGB band response',ratio.tolist(),flush=True)
 print('PASS actual camera -> conductor -> emitter, no field/NEE lighting',flush=True)
if __name__=='__main__':run()
