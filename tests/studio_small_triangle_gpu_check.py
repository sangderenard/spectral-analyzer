"""Opt-in isolated GPU regression for legacy small triangles; run only idle.
Optional payload.npz argument exercises every uploaded triangle at 10mm scale.
"""
import sys
from pathlib import Path
import runpy
import numpy as np
import demo_pluck_gl as renderer
from OpenGL import GL


def run(payload=None):
    helper=runpy.run_path(str(Path(__file__).with_name('test_studio_small_triangle_precision.py')))
    bodies=helper['hit_functions']()
    if payload:
        with np.load(payload) as data:
            vertices=np.asarray(data['vertices'],np.float32)*.01
    else:
        vertices=np.array([[[0,0,0],[1e-5,0,0],[0,1e-5,0]],[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
    n=len(vertices);geom=np.zeros((n,16),np.float32)
    geom[:,:3]=vertices[:,0];geom[:,4:7]=vertices[:,1]-vertices[:,0];geom[:,8:11]=vertices[:,2]-vertices[:,0]
    renderer._ensure_worker_gl_context()
    print('GPU',GL.glGetString(GL.GL_RENDERER),flush=True)
    prefix='''#version 430 core
layout(local_size_x=64) in;
struct TriGeom {vec4 v0;vec4 e1;vec4 e2;vec4 normal;};
layout(std430,binding=0) readonly buffer G {TriGeom tris[];};
layout(std430,binding=1) buffer O {vec4 result[];};
'''
    main='''
void main(){uint i=gl_GlobalInvocationID.x;if(i>=tris.length())return;TriGeom t=tris[i];
vec3 n=normalize(cross(t.e1.xyz,t.e2.xyz));
vec3 center=t.v0.xyz+(t.e1.xyz+t.e2.xyz)/3.0;
float distance=0.0;bool h=hit_tri(center+n*.001,-n,t,distance);
result[i]=vec4(h?1.0:0.0,distance,0,0);}
'''
    for name,body in zip(('forward','camera'),bodies):
        for variant in ('fixed','old'):
            code=body if variant=='fixed' else body.replace(helper['RELATIVE'],'if (abs(a) < 1e-8) return false;')
            program=renderer._prog((prefix+code+main,GL.GL_COMPUTE_SHADER));bufs=[]
            try:
                for binding,data in ((0,geom),(1,np.zeros((n,4),np.float32))):
                    b=int(GL.glGenBuffers(1));bufs.append(b);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,b)
                    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_DYNAMIC_DRAW)
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,b)
                GL.glUseProgram(program);GL.glDispatchCompute((n+63)//64,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,bufs[1])
                out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,n*16),np.float32).reshape(n,4)
                hits=int(np.sum(out[:,0]));print(name,variant,hits,'/',n,flush=True)
                if variant=='fixed':assert hits==n
                else:assert hits<n
            finally:
                GL.glDeleteBuffers(len(bufs),bufs);GL.glDeleteProgram(program)

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else None)

