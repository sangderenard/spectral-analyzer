"""Hidden offscreen GL field mapping checks of the real material variants.

Run directly, outside pytest's intentional pygame/OpenGL module stubs.
No live application or exposure is opened, stopped, or modified.
"""
import json
import math
import os
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import numpy as np
import pygame
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader


def display(x):
    x = min(1., max(0., x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14)))
    return 12.92*x if x < 0.0031308 else 1.055*x**(1/2.4)-0.055


def main():
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((8, 8), pygame.OPENGL | pygame.HIDDEN)
    try:
        vertex = """#version 430 core
uniform float uCosine; uniform vec3 uProbeView; uniform vec3 uProbeObject;
out vec3 vNormV; out vec3 vPosV; out vec3 vPosObj;
flat out int vMatId; flat out int vGroupId; flat out int vCullImmune;
out vec2 vUv; out vec3 vVelocityW;
void main() {
  vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2)*2.0-1.0;
  gl_Position=vec4(p,0,1); vNormV=vec3(0,0,1);
  vPosV=uProbeView;
  vPosObj=uProbeObject;vMatId=0;vGroupId=0;vCullImmune=1;vUv=vec2(0);vVelocityW=vec3(0);
}"""
        source=(Path(__file__).resolve().parents[1]/"csrc/shaders/base_material.frag.glsl").read_text(encoding="utf-8")
        programs=[]
        legacy=source
        begin=legacy.index("    // 3-D field volume:")
        end=legacy.index("    // Color compensation equation",begin)
        legacy=legacy[:begin]+"    if (uFieldGain > 0.0) { col += texture(uFieldVolume, vPosObj).rgb * uFieldGain; }\n\n"+legacy[end:]
        for defines in ("", "#define PLUCK_TRANSMISSION 1\n", "#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\n"):
            for frag in (legacy,source):
                frag=frag.replace("#version 430 core\n","#version 430 core\n"+defines,1)
                programs.append(compileProgram(compileShader(vertex,gl.GL_VERTEX_SHADER),compileShader(frag,gl.GL_FRAGMENT_SHADER),validate=False))
        vao=gl.glGenVertexArrays(1);gl.glBindVertexArray(vao)
        fbo=gl.glGenFramebuffers(1);gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,fbo)
        tex=gl.glGenTextures(1);gl.glBindTexture(gl.GL_TEXTURE_2D,tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA32F,1,1,0,gl.GL_RGBA,gl.GL_FLOAT,None)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,gl.GL_COLOR_ATTACHMENT0,gl.GL_TEXTURE_2D,tex,0)
        assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)==gl.GL_FRAMEBUFFER_COMPLETE
        gl.glViewport(0,0,1,1)
        pbr=np.zeros(16,np.float32);pbr[:3]=[.1,.43,.91];pbr[3]=.12;pbr[5:8]=[1,1.52,0]
        phong=np.array([0,1,160,0,0,0,0,0],np.float32)
        tx=np.zeros(16,np.float32);tx[:4]=-1;tx[8]=1
        buffers={}
        for binding,data in [(10,pbr),(11,phong),(14,np.zeros(8,np.float32)),(15,tx),(16,np.zeros(1,np.float32))]:
            buffer=gl.glGenBuffers(1);buffers[binding]=buffer
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffer)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,data.nbytes,data,gl.GL_DYNAMIC_DRAW)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,binding,buffer)

        volume=np.empty((2,2,2,3),np.float32)
        for z in range(2):
            for y in range(2):
                for x in range(2): volume[z,y,x]=[.1+.2*x,.2+.2*y,-.3+.2*z]
        field=gl.glGenTextures(1);gl.glActiveTexture(gl.GL_TEXTURE8);gl.glBindTexture(gl.GL_TEXTURE_3D,field)
        gl.glTexImage3D(gl.GL_TEXTURE_3D,0,gl.GL_RGB32F,2,2,2,0,gl.GL_RGB,gl.GL_FLOAT,volume)
        for axis in (gl.GL_TEXTURE_WRAP_S,gl.GL_TEXTURE_WRAP_T,gl.GL_TEXTURE_WRAP_R):
            gl.glTexParameteri(gl.GL_TEXTURE_3D,axis,gl.GL_CLAMP_TO_EDGE)
        for filt in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
            gl.glTexParameteri(gl.GL_TEXTURE_3D,filt,gl.GL_NEAREST)
        pbr[5:8]=[0,1.52,1];pbr[8:11]=[.4,.4,.4]
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[10]);gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,pbr.nbytes,pbr)
        def draw(program, transform=False, gain=1, view=(0,0,-2), obj=(.25,.25,.25), matrix=None):
            gl.glUseProgram(program)
            loc=lambda name:gl.glGetUniformLocation(program,name)
            gl.glUniform1f(loc("uExposure"),1)
            gl.glUniform1i(loc("uNumLights"),0)
            gl.glUniform1i(loc("uFieldVolume"),8)
            gl.glUniform1f(loc("uFieldGain"),gain)
            gl.glUniform1i(loc("uFieldUseTransform"),int(transform))
            gl.glUniform3f(loc("uProbeView"),*view);gl.glUniform3f(loc("uProbeObject"),*obj)
            if matrix is not None: gl.glUniformMatrix4fv(loc("uFieldFromView"),1,gl.GL_TRUE,np.asarray(matrix,np.float32))
            gl.glDisable(gl.GL_DEPTH_TEST);gl.glDisable(gl.GL_BLEND)
            gl.glClearColor(0,0,0,0);gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDrawArrays(gl.GL_TRIANGLES,0,3)
            pixel=np.frombuffer(gl.glReadPixels(0,0,1,1,gl.GL_RGBA,gl.GL_FLOAT),np.float32).copy()
            assert gl.glGetError()==gl.GL_NO_ERROR
            return pixel
        def expected(rgb): return [display(float(c)) for c in np.maximum(rgb,0)]+[1]
        checks=0
        for old,new in zip(programs[::2],programs[1::2]):
            # Off is bit-identical, including legacy out-of-range clamp behavior.
            for obj in ((.25,.25,.25),(.75,.75,.75),(2,-1,.25)):
                np.testing.assert_array_equal(draw(old,obj=obj),draw(new,obj=obj));checks+=1
            np.testing.assert_array_equal(draw(old,gain=0),draw(new,transform=True,gain=0));checks+=1
            # Nonidentity physical scale/offset composed with inverse camera view.
            world_to_field=np.diag([.5,.25,.125,1.]);world_to_field[:3,3]=[-1,-2,-3]
            view=np.array([[0,-1,0,7],[1,0,0,-3],[0,0,1,4],[0,0,0,1]],float)
            mapping=world_to_field@np.linalg.inv(view)
            for coord in ((.25,.25,.25),(.75,.75,.75),(.75,.25,.75),(0,0,0),(1,1,1)):
                world=np.linalg.inv(world_to_field)@[*coord,1]
                point=(view@world)[:3]
                idx=np.minimum((np.array(coord)*2).astype(int),1)
                rgb=volume[idx[2],idx[1],idx[0]]+.4
                np.testing.assert_allclose(draw(new,True,view=point,matrix=mapping),expected(rgb),atol=3e-5,rtol=0);checks+=1
            for axis in range(3):
                for edge in (-.01,1.01):
                    coord=np.array([.5,.5,.5,1]);coord[axis]=edge
                    point=(view@np.linalg.inv(world_to_field)@coord)[:3]
                    np.testing.assert_allclose(draw(new,True,view=point,matrix=mapping),expected([.4]*3),atol=3e-5,rtol=0);checks+=1
        print(json.dumps({"status":"PASS","gpu":gl.glGetString(gl.GL_RENDERER).decode(),"checks":checks,"variants":3,"signed_rgb_preserved":True}))
    finally:
        pygame.display.quit()


if __name__=="__main__":
    main()
