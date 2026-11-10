"""Opt-in native GLSL cookie predicate test; no renderer or exposure starts."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pygame
from OpenGL import GL
from camera_designer.curve_cookie import text_cookie


def run():
    cookie=text_cookie('MEOW')
    # Deterministic physical rays, all coordinates quantized to the shader ABI.
    rng=np.random.default_rng(7483)
    uv=rng.uniform(-1.15,1.15,(2048,2))*np.asarray(cookie.half_size)
    origins=np.column_stack((uv,np.full(len(uv),.1),np.zeros(len(uv)))).astype(np.float32)
    directions=np.tile([0,0,-1,0],(len(uv),1)).astype(np.float32)
    expected=np.asarray([np.isfinite(cookie.intersect(o[:3],d[:3])[0]) for o,d in zip(origins,directions)])
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((32,32),pygame.OPENGL|pygame.DOUBLEBUF|pygame.HIDDEN)
    resources=[];shader=None;program=None
    try:
        source='''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Origins {vec4 origins[];};
layout(std430,binding=1) readonly buffer Directions {vec4 directions[];};
layout(std430,binding=2) writeonly buffer Hits {vec4 hits[];};
'''+cookie.glsl_intercept_fn('cookie_hit')+'''
void main(){uint i=gl_GlobalInvocationID.x;if(i>=origins.length())return;
float t;vec3 n;bool hit=cookie_hit(origins[i].xyz,directions[i].xyz,t,n);
hits[i]=vec4(hit?1.0:0.0,t,n.z,0);}
'''
        shader=GL.glCreateShader(GL.GL_COMPUTE_SHADER);GL.glShaderSource(shader,source);GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader,GL.GL_COMPILE_STATUS):raise RuntimeError(GL.glGetShaderInfoLog(shader))
        program=GL.glCreateProgram();GL.glAttachShader(program,shader);GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program,GL.GL_LINK_STATUS):raise RuntimeError(GL.glGetProgramInfoLog(program))
        for binding,data in enumerate((origins,directions,np.zeros_like(origins))):
            b=GL.glGenBuffers(1);resources.append(b);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,b)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,GL.GL_STATIC_DRAW)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,b)
        GL.glUseProgram(program);GL.glDispatchCompute(len(uv)//64,1,1);GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT|GL.GL_SHADER_STORAGE_BARRIER_BIT)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,resources[-1]);raw=GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,origins.nbytes)
        values=np.frombuffer(raw,dtype=np.float32).reshape(-1,4)
        actual=values[:,0]>.5;mismatch=int(np.count_nonzero(actual!=expected))
        assert mismatch==0,(mismatch,np.flatnonzero(actual!=expected)[:10])
        assert np.allclose(values[actual,1],.1) and np.all(values[actual,2]==1)
        assert GL.glGetError()==GL.GL_NO_ERROR
        return {'passed':True,'rays':len(uv),'blocked':int(actual.sum()),'open_or_outside':int((~actual).sum()),
                'mismatches':mismatch,'renderer':GL.glGetString(GL.GL_RENDERER).decode(),
                'curves':sum(map(len,cookie.contours)),'monotone_intervals':len(cookie._pieces),
                'scope':'actual GLSL predicate dispatch, not a production mixed-scene render'}
    finally:
        if resources:GL.glDeleteBuffers(len(resources),resources)
        if program:GL.glDeleteProgram(program)
        if shader:GL.glDeleteShader(shader)
        pygame.display.quit()


if __name__=='__main__':print(json.dumps(run(),indent=2))
