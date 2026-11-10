"""Saved stage candidate-only parity export. Not a region render/offload claim."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from camera_designer.studio_photo_geometry import prepare_photo_geometry
import demo_pluck_gl as r
from OpenGL import GL as g
folder=Path(sys.argv[1]);config=json.loads((folder/'config.json').read_text())
with np.load(folder/'payload.npz',allow_pickle=False) as p:
 scene,spheres,stats,_=prepare_photo_geometry(config,p['vertices'],p['material_ids'],p['pbr'])
geom=scene[0];nodes,ids=r._build_gpu_bvh(scene[3]);r._ensure_worker_gl_context();buffers=[]
def buf(binding,data):
 b=int(g.glGenBuffers(1));buffers.append(b);g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,b);g.glBufferData(g.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,g.GL_DYNAMIC_DRAW);g.glBindBufferBase(g.GL_SHADER_STORAGE_BUFFER,binding,b);return b
# Deterministic camera-origin probes through a broad stage window.
eye=np.asarray(config.get('camera',{}).get('eye_m',[0,0,.07]),np.float32)
target=np.asarray(config.get('camera',{}).get('target_m',[0,0,.18]),np.float32)
rays=np.zeros((256,8),np.float32);rays[:,:3]=eye;rays[:,3]=1e-6;rays[:,7]=1e30
for i in range(256):
 p=target+np.array([(i%16-7.5)*.005,(i//16-7.5)*.005,0]);d=p-eye;rays[i,4:7]=d/np.linalg.norm(d)
try:
 for binding,data in [(0,geom),(1,nodes),(2,ids),(9,scene[1]),(16,rays)]:buf(binding,np.ascontiguousarray(data))
 result=buf(18,np.zeros((256,4),np.float32))
 src=r._GPU_SENSOR_CS.replace('void main() {','void unused_sensor_main() {',1)+'''\nlayout(std430,binding=16) readonly buffer InputRays {vec4 rays[];};
layout(std430,binding=18) buffer OutputHits {vec4 answer[];};
void main(){uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*16u;if(i>=256u)return;float t;int h=nearest_filtered(rays[2*i].xyz,rays[2*i+1].xyz,t,false);answer[i]=vec4(h>=0?t:0,float(h),h>=0?1:0,float(h));}'''
 program=r._prog((src,g.GL_COMPUTE_SHADER));assert g.glGetProgramiv(program,g.GL_LINK_STATUS);g.glUseProgram(program)
 for k,v in [('uNodeCount',len(nodes)),('uTriCount',len(geom)),('uStudioSphereCount',0)]:g.glUniform1i(g.glGetUniformLocation(program,k),v)
 g.glDispatchCompute(1,1,1);g.glMemoryBarrier(g.GL_SHADER_STORAGE_BARRIER_BIT);g.glFinish();g.glBindBuffer(g.GL_SHADER_STORAGE_BUFFER,result)
 hits=np.frombuffer(g.glGetBufferSubData(g.GL_SHADER_STORAGE_BUFFER,0,4096),np.float32).copy()
 out=ROOT/'build/dot-first-hit-loan-20261109/stage-candidate-request.json'
 out.write_text(json.dumps(dict(revision='saved-stage-candidate-only-v1',scope='plain-law-triangle-only-not-region-or-sphere-render',rays=rays.ravel().tolist(),triangles=[dict(owner=i,v0=t[:3].tolist(),e1=t[4:7].tolist(),e2=t[8:11].tolist()) for i,t in enumerate(geom)],bvh_nodes=nodes.tolist(),bvh_ids=ids.tolist(),bvh_epsilon=1e-7,expected=hits.tolist())))
 print(json.dumps(dict(path=str(out),triangles=len(geom),nodes=len(nodes),hits=int(hits.reshape(-1,4)[:,2].sum()))));g.glDeleteProgram(program)
finally:g.glDeleteBuffers(len(buffers),buffers)
