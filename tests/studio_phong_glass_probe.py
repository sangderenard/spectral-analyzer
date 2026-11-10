"""Hidden offscreen GL comparison of the real legacy/Studio fragment variants.

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
uniform float uCosine;
out vec3 vNormV; out vec3 vPosV; out vec3 vPosObj;
flat out int vMatId; flat out int vGroupId; flat out int vCullImmune;
out vec2 vUv; out vec3 vVelocityW;
void main() {
  vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2)*2.0-1.0;
  gl_Position=vec4(p,0,1); vNormV=vec3(0,0,1);
  vPosV=-2.0*vec3(sqrt(1-uCosine*uCosine),0,uCosine);
  vPosObj=vec3(0);vMatId=0;vGroupId=0;vCullImmune=1;vUv=vec2(0);vVelocityW=vec3(0);
}"""
        source=(Path(__file__).resolve().parents[1]/"csrc/shaders/base_material.frag.glsl").read_text(encoding="utf-8")
        programs=[]
        for enabled in (False,True):
            defines="#define PLUCK_TRANSMISSION 1\n"
            if enabled: defines+="#define PLUCK_PREMULTIPLIED_GLASS 1\n"
            frag=source.replace("#version 430 core\n","#version 430 core\n"+defines,1)
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

        def draw(enabled,transmission=1,opacity=0,ior=1.52,cosine=1,backlight=False):
            pbr[5:8]=[transmission,ior,opacity]
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[10]);gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,pbr.nbytes,pbr)
            program=programs[int(enabled)];gl.glUseProgram(program)
            loc=lambda name:gl.glGetUniformLocation(program,name)
            gl.glUniform1f(loc("uCosine"),cosine);gl.glUniform1f(loc("uExposure"),1)
            gl.glUniform1i(loc("uNumLights"),1);gl.glUniform1i(loc("uLightGroupId[0]"),1)
            gl.glUniform3f(loc("uLightColor[0]"),1,1,1)
            gl.glUniform3f(loc("uLightPos[0]"),-4*math.sqrt(1-cosine*cosine),0,-4*cosine if backlight else 0)
            gl.glUniform1f(loc("uLightIntensity[0]"),4/(1-1/(4*math.pi)))
            gl.glUniform1i(loc("uFieldVolume"),4)
            gl.glDisable(gl.GL_DEPTH_TEST);gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_ONE if enabled else gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glClearColor(.2,.2,.2,1);gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDrawArrays(gl.GL_TRIANGLES,0,3)
            pixel=np.frombuffer(gl.glReadPixels(0,0,1,1,gl.GL_RGBA,gl.GL_FLOAT),np.float32)
            assert gl.glGetError()==gl.GL_NO_ERROR
            return pixel[:3].copy()

        records=[]
        for cosine,opacity in [(1.,0.),(.2,0.),(1.,.4)]:
            old=draw(False,cosine=cosine,opacity=opacity)
            new=draw(True,cosine=cosine,opacity=opacity)
            f0=((1.52-1)/(1.52+1))**2
            fresnel=f0+(1-f0)*(1-cosine)**5
            reflection=fresnel+(1-fresnel)*opacity
            expected=display(reflection)+.2*(1-reflection)
            np.testing.assert_allclose(new,[expected]*3,atol=3e-5,rtol=0)
            if opacity==0: np.testing.assert_allclose(old,[.2]*3,atol=1e-6,rtol=0)
            records.append({"cosine":cosine,"opacity":opacity,"reflection":reflection,"old":old.tolist(),"new":new.tolist(),"expected":expected})
        # A blue dielectric still transmits coloured light at zero opacity.
        transmitted=draw(True,backlight=True)
        f0=((1.52-1)/(1.52+1))**2
        expected=[display(float(c)*(1-f0))+.2*(1-f0) for c in pbr[:3]]
        np.testing.assert_allclose(transmitted,expected,atol=3e-5,rtol=0)
        records.append({"backlight":True,"new":transmitted.tolist(),"expected":expected})
        np.testing.assert_allclose(draw(True,ior=1),[.2]*3,atol=1e-6,rtol=0)
        opaque_old=draw(False,transmission=0,opacity=1)
        opaque_new=draw(True,transmission=0,opacity=1)
        np.testing.assert_array_equal(opaque_old,opaque_new)
        print(json.dumps({"status":"PASS","gpu":gl.glGetString(gl.GL_RENDERER).decode(),"comparisons":records,"opaque_unchanged":opaque_new.tolist()}))
    finally:
        pygame.display.quit()


if __name__=="__main__":
    main()
