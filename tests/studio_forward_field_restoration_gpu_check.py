"""Isolated actual SensorAccumulator proof of the two restored field terms."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 import demo_pluck_gl as r
 from OpenGL import GL as gl
 from camera_designer.film_response import build_film_stack,normalize_film_response
 r._ensure_worker_gl_context(); buffers=[]; textures=[]; results=[]
 tri=np.array([[[-2,-2,0],[2,-2,0],[0,2,0]]],np.float32)
 geom=np.zeros((1,16),np.float32);geom[:,:3]=tri[:,0];geom[:,4:7]=tri[:,1]-tri[:,0];geom[:,8:11]=tri[:,2]-tri[:,0];geom[:,14]=1
 shade=np.zeros((1,16),np.float32);shade[:,:4]=[.8,1,0,1.5];shade[:,4:8]=[.8,1,0,1];shade[:,8:11]=1
 nodes,ids=r._build_gpu_bvh(tri)
 def buf(a):
  b=int(gl.glGenBuffers(1));buffers.append(b);gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,b);gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,a.nbytes,a,gl.GL_STATIC_DRAW);return b
 bg,bn,bi,bs,bm,br=[buf(a) for a in [geom,nodes,ids,shade,np.zeros((1,32,12),np.float32),np.zeros((1,4),np.float32)]]
 for i in range(3):
  t=int(gl.glGenTextures(1));textures.append(t);gl.glBindTexture(gl.GL_TEXTURE_3D,t);gl.glTexStorage3D(gl.GL_TEXTURE_3D,1,gl.GL_R32UI,8,8,8)
  for p in [gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER]:gl.glTexParameteri(gl.GL_TEXTURE_3D,p,gl.GL_NEAREST)
  for p in [gl.GL_TEXTURE_WRAP_S,gl.GL_TEXTURE_WRAP_T,gl.GL_TEXTURE_WRAP_R]:gl.glTexParameteri(gl.GL_TEXTURE_3D,p,gl.GL_CLAMP_TO_EDGE)
 try:
  for region in [False,True]:
   acc=r.SensorAccumulator(bg,bn,bi,bs,bm,1,len(nodes),textures,[-.5,-.5,-.5],[.5,.5,.5],(8,8,8),[0,0,1],[0,0,0],16,16,rows_per_frame=16,fov_deg=5,max_bounces=1,ray_field_scale=1,vol_steps=16,n_sources=0,ssbo_regions=br if region else 0,n_regions=int(region))
   try:
    assert acc._active
    acc.film=build_film_stack(r.FILM_STACKS['em_rgb_fixed3'],normalize_film_response());acc._sprinkle=False;acc._max_bounces=0
    acc._nee_area_sources=True
    for mode in ['surface','volume']:
     acc._n_tris=1 if mode=='surface' else 0;acc._n_bvh_nodes=len(nodes) if mode=='surface' else 0;acc._va=0 if mode=='surface' else 1
     outputs=[]
     for value in [0,65535]:
      for t in textures:gl.glClearTexImage(t,0,gl.GL_RED_INTEGER,gl.GL_UNSIGNED_INT,np.array([value],np.uint32))
      gl.glMemoryBarrier(gl.GL_TEXTURE_UPDATE_BARRIER_BIT|gl.GL_TEXTURE_FETCH_BARRIER_BIT)
      acc.clear();acc._frame=0;acc.tick();gl.glFinish()
      bands=[]
      for t in acc._textures:
       gl.glBindTexture(gl.GL_TEXTURE_2D,t);p=np.empty((16,16,4),np.float32);gl.glGetTexImage(gl.GL_TEXTURE_2D,0,gl.GL_RGBA,gl.GL_FLOAT,p);bands.append(p[:,:,0])
      outputs.append(np.stack(bands,axis=-1))
     assert np.isfinite(outputs).all();assert not outputs[0].any(),outputs[0].max();assert np.all(outputs[1]>0),outputs[1].min()
     item=dict(shader='region' if region else 'plain',term=mode,area_mode=True,pixels=256,zero_field_max=float(outputs[0].max()),nonzero_field_min=float(outputs[1].min()),nonzero_field_mean=outputs[1].mean(axis=(0,1)).tolist(),nonzero_field_max=float(outputs[1].max()))
     results.append(item);print(json.dumps(item),flush=True)
   finally:acc.destroy()
  receipt=dict(renderer=gl.glGetString(gl.GL_RENDERER).decode(),actual_production_sensor_main=True,cases=results,renderer_sha256=hashlib.sha256((ROOT/'demo_pluck_gl.py').read_bytes()).hexdigest(),limits='Injected field proves both camera reads, independently, under area mode; does not establish native source-to-field energy scale, convergence, or absence of estimator overlap.')
  (ROOT/'build/dot-native-area-20261110/field-restoration/gpu-field-check.json').write_text(json.dumps(receipt,indent=2));print('PASS both restored field terms in actual plain and region sensor shaders',flush=True)
 finally:
  if textures:gl.glDeleteTextures(textures)
  if buffers:gl.glDeleteBuffers(len(buffers),buffers)
if __name__=='__main__':main()
