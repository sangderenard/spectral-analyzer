"""Hidden offscreen GL checks of the canonical secondary material program.

Run directly, outside pytest's intentional pygame/OpenGL module stubs.
No live application or exposure is opened, stopped, or modified.
"""
import json
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
uniform float uCosine; uniform float uProbeDepth; uniform vec3 uProbeView; uniform vec3 uProbeObject;
out vec3 vNormV; out vec3 vPosV; out vec3 vPosObj;
flat out int vMatId; flat out int vGroupId; flat out int vCullImmune;
out vec2 vUv; out vec3 vVelocityW;
void main() {
  vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2)*2.0-1.0;
  gl_Position=vec4(p,uProbeDepth,1); vNormV=vec3(0,0,1);
  vPosV=uProbeView;
  vPosObj=uProbeObject;vMatId=0;vGroupId=0;vCullImmune=1;vUv=vec2(0);vVelocityW=vec3(0);
}"""
        source=(Path(__file__).resolve().parents[1]/"csrc/shaders/base_material.frag.glsl").read_text(encoding="utf-8")
        programs=[]
        import subprocess
        legacy=subprocess.check_output(["git","show","HEAD:csrc/shaders/base_material.frag.glsl"],cwd=Path(__file__).resolve().parents[1],text=True)
        defines="#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\n"
        for frag, extra in ((legacy,""),(source,""),(source,"#define PLUCK_SECONDARY_FIELD 1\n"),(source,"#define PLUCK_FIELD_PRIMARY 1\n")):
            frag=frag.replace("#version 430 core\n","#version 430 core\n"+defines+extra,1)
            programs.append(compileProgram(compileShader(vertex,gl.GL_VERTEX_SHADER),compileShader(frag,gl.GL_FRAGMENT_SHADER),validate=False))
        vao=gl.glGenVertexArrays(1);gl.glBindVertexArray(vao)
        fbo=gl.glGenFramebuffers(1);gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,fbo)
        tex=gl.glGenTextures(1);gl.glBindTexture(gl.GL_TEXTURE_2D,tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA32F,1,1,0,gl.GL_RGBA,gl.GL_FLOAT,None)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,gl.GL_COLOR_ATTACHMENT0,gl.GL_TEXTURE_2D,tex,0)
        depthbuffer=gl.glGenRenderbuffers(1);gl.glBindRenderbuffer(gl.GL_RENDERBUFFER,depthbuffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER,gl.GL_DEPTH_COMPONENT24,1,1)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER,gl.GL_DEPTH_ATTACHMENT,gl.GL_RENDERBUFFER,depthbuffer)
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

        energy=np.zeros((192,2,2,3),np.float32)
        directions=np.zeros_like(energy)
        textures=[]
        for unit, data in ((11,energy),(12,directions)):
            t=gl.glGenTextures(1);textures.append(t)
            gl.glActiveTexture(gl.GL_TEXTURE0+unit);gl.glBindTexture(gl.GL_TEXTURE_3D,t)
            gl.glTexImage3D(gl.GL_TEXTURE_3D,0,gl.GL_RGB32F,2,2,192,0,gl.GL_RGB,gl.GL_FLOAT,data)
            for axis in (gl.GL_TEXTURE_WRAP_S,gl.GL_TEXTURE_WRAP_T,gl.GL_TEXTURE_WRAP_R):
                gl.glTexParameteri(gl.GL_TEXTURE_3D,axis,gl.GL_CLAMP_TO_EDGE)
            for filt in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_3D,filt,gl.GL_NEAREST)
        masktex=gl.glGenTextures(1)
        def field(propagation=(0,0,-1),rgb=(1,1,1),mask=7,bin=5,gradient=False,columns=1,bins=6,spatial=False,pair=False,grid_depth=32,grid_xy=2):
            energy=np.zeros((((bins+columns-1)//columns)*grid_depth,grid_xy,columns*grid_xy,3),np.float32)
            directions=np.zeros((*energy.shape[:3],4),np.float32)
            z=(bin//columns)*grid_depth;x=(bin%columns)*grid_xy
            energy[z:z+grid_depth,:,x:x+grid_xy]=rgb
            directions[z:z+grid_depth,:,x:x+grid_xy,:3]=propagation
            directions[z:z+grid_depth,:,x:x+grid_xy,3]=1
            if gradient:
                energy[z:z+grid_depth,:,x:x+grid_xy] *= np.arange(1,grid_depth+1,dtype=np.float32)[:,None,None,None]
            if pair:
                energy[:]=0;directions[:]=0
                low=grid_depth//2-1;high=low+1
                energy[z+low,:,x:x+grid_xy]=1;energy[z+high,:,x:x+grid_xy]=3
                directions[z+low,:,x:x+grid_xy]=[0,0,-1,1]
                directions[z+high,:,x:x+grid_xy]=[3,0,0,3]
            for unit,t,data in zip((11,12),textures,(energy,directions)):
                gl.glActiveTexture(gl.GL_TEXTURE0+unit);gl.glBindTexture(gl.GL_TEXTURE_3D,t)
                internal,fmt=(gl.GL_RGB32F,gl.GL_RGB) if unit==11 else (gl.GL_RGBA32F,gl.GL_RGBA)
                gl.glTexImage3D(gl.GL_TEXTURE_3D,0,internal,energy.shape[2],energy.shape[1],energy.shape[0],0,fmt,gl.GL_FLOAT,data)
                for filt in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
                    gl.glTexParameteri(gl.GL_TEXTURE_3D,filt,gl.GL_LINEAR if spatial else gl.GL_NEAREST)
            gl.glActiveTexture(gl.GL_TEXTURE13);gl.glBindTexture(gl.GL_TEXTURE_2D,masktex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_R32UI,1,1,0,gl.GL_RED_INTEGER,gl.GL_UNSIGNED_INT,np.array([mask],np.uint32))
            for filt in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D,filt,gl.GL_NEAREST)
        def draw(program=2,pass_id=1,trans=0,opacity=1,rough=.5,linear=True,override=False,
                 rgb=(1,1,1),prop=(0,0,-1),mask=7,bin=5,gain=1,background=None,rotation=None,depth=0,depth_mode=None,lights=False,replace=False,field_z=.5,gradient=False,backlight=False,columns=1,bins=6,spatial=False,pair=False,grid_depth=32,grid_xy=2,field_x=.5,category=True,field_y=.5,restrict=False):
            field(prop,rgb,mask,bin,gradient,columns,bins,spatial,pair,grid_depth,grid_xy)
            pbr[3]=rough;pbr[5:8]=[trans,1.5,opacity]
            for binding,data in ((10,pbr),(11,phong),(15,tx)):
                gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffers[binding]);gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,data.nbytes,data)
            program=programs[program];gl.glUseProgram(program)
            loc=lambda name:gl.glGetUniformLocation(program,name)
            gl.glUniform1f(loc("uExposure"),1);gl.glUniform1i(loc("uNumLights"),int(lights))
            gl.glUniform1i(loc("uLightGroupId[0]"),1)
            gl.glUniform3f(loc("uLightPos[0]"),0,0,-4 if backlight else 0)
            gl.glUniform3f(loc("uLightColor[0]"),1,1,1)
            gl.glUniform1f(loc("uLightIntensity[0]"),4/(1-1/(4*np.pi)))
            gl.glUniform1i(loc("uFieldReplaceSpecular"),int(replace))
            gl.glUniform1i(loc("uFieldCategoryDisabled"),int(not category))
            gl.glUniform1i(loc("uFieldRestrictToBounds"),int(restrict))
            gl.glUniform1i(loc("uLinearOutput"),int(linear));gl.glUniform1i(loc("uSecondaryPass"),pass_id)
            gl.glUniform1i(loc("uSecondaryPhongSpecular"),int(override));gl.glUniform1i(loc("uEnableSpecular"),1)
            gl.glUniform1i(loc("uFieldVolume"),8);gl.glUniform1i(loc("uFieldRadiance"),11)
            gl.glUniform1i(loc("uFieldDirection"),12);gl.glUniform1i(loc("uFieldMaterialMask"),13)
            gl.glUniform1i(loc("uFieldResolvedColor"),14)
            gl.glUniform1f(loc("uFieldGridDepth"),grid_depth);gl.glUniform1f(loc("uFieldGain"),gain)
            gl.glUniform1i(loc("uFieldBinCount"),bins);gl.glUniform1i(loc("uFieldAtlasColumns"),columns)
            gl.glUniform1i(loc("uFieldSpatialLinear"),int(spatial))
            gl.glUniform3f(loc("uProbeView"),0,0,-2);gl.glUniform3f(loc("uProbeObject"),.5,.5,.5)
            mapping=np.eye(4,dtype=np.float32);mapping[:3,3]=[field_x,field_y,2+field_z]
            gl.glUniformMatrix4fv(loc("uFieldFromView"),1,gl.GL_TRUE,mapping)
            gl.glUniformMatrix3fv(loc("uFieldViewFromWorld"),1,gl.GL_TRUE,np.eye(3,dtype=np.float32) if rotation is None else np.asarray(rotation,np.float32))
            gl.glUniform1f(loc("uProbeDepth"),depth)
            if depth_mode is None:
                gl.glDisable(gl.GL_DEPTH_TEST)
            else:
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(depth_mode == "primary")
                gl.glDepthFunc(gl.GL_LESS if depth_mode == "primary" else gl.GL_EQUAL)
                if depth_mode == "primary":
                    gl.glClearDepth(1);gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            if background is None:
                gl.glDisable(gl.GL_BLEND);gl.glClearColor(0,0,0,0);gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            else:
                gl.glEnable(gl.GL_BLEND);gl.glBlendFunc(gl.GL_ONE,gl.GL_ONE_MINUS_SRC_ALPHA)
                if background is not False:
                    gl.glClearColor(*background);gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDrawArrays(gl.GL_TRIANGLES,0,3)
            pixel=np.frombuffer(gl.glReadPixels(0,0,1,1,gl.GL_RGBA,gl.GL_FLOAT),np.float32).copy()
            assert gl.glGetError()==gl.GL_NO_ERROR
            return pixel
        checks=0
        def equal(actual,expected):
            nonlocal checks
            np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=0);checks+=1
        # Original no-field path is bit-identical to repository HEAD.
        for trans,opacity in ((0,1),(0,.4),(1,0),(1,.6)):
            np.testing.assert_array_equal(draw(0,trans=trans,opacity=opacity,linear=False,gain=0),draw(1,trans=trans,opacity=opacity,linear=False,gain=0));checks+=1
        f0=.04;albedo=pbr[:3].copy()
        # Field replacement is per material: unmatched rows retain direct Phong.
        equal(draw(program=3,lights=True,replace=True,mask=0,gain=0),[*(albedo+f0),1])
        equal(draw(program=3,lights=True,replace=True,mask=2,gain=0),[0,0,0,1])
        equal(draw(program=3,lights=True,replace=False,mask=2,gain=0),[f0,f0,f0,1])
        equal(draw(pass_id=2,lights=True,mask=0,gain=0),[*(albedo+f0),1])
        equal(draw(pass_id=2,lights=True,mask=2,gain=0),[0,0,0,1])
        # Shared material rows may belong to enabled and disabled geometry.
        equal(draw(program=3,lights=True,replace=True,mask=2,gain=0,category=False),[*(albedo+f0),1])
        equal(draw(mask=2,category=False),[0,0,0,0])
        equal(draw(pass_id=2,lights=True,mask=2,category=False),[*(albedo+f0),1])
        equal(draw(pass_id=2,lights=True,backlight=True,prop=(0,0,1),mask=4,trans=1,opacity=0,category=False),[*(albedo*.96),f0])
        # Bounded field ownership applies equally to primary and secondary.
        equal(draw(program=3,lights=True,replace=True,mask=2,restrict=True),[0,0,0,1])
        equal(draw(program=3,lights=True,replace=True,mask=2,field_x=-.01,restrict=False),[0,0,0,1])
        for axis in ('field_x','field_y','field_z'):
            for edge in (-.01,1.01):
                point={axis:edge,'restrict':True}
                equal(draw(program=3,lights=True,replace=True,mask=2,**point),[*(albedo+f0),1])
                equal(draw(mask=2,**point),[0,0,0,0])
                equal(draw(pass_id=2,lights=True,mask=2,**point),[*(albedo+f0),1])
                equal(draw(pass_id=2,lights=True,backlight=True,mask=4,trans=1,opacity=0,**point),[*(albedo*.96),f0])
            for edge in (0.,1.):
                equal(draw(program=3,lights=True,replace=True,mask=2,restrict=True,**{axis:edge}),[0,0,0,1])
        # Same direct source and field source are owned once, not summed twice.
        equal(draw(pass_id=2,lights=True,mask=2),[*(albedo+f0),1])
        equal(draw(pass_id=2,lights=True,mask=2,override=True),[*(albedo+f0),1])
        equal(draw(program=3,lights=True,backlight=True,mask=4,gain=0,trans=1,opacity=0),[0,0,0,f0])
        equal(draw(program=3,lights=True,backlight=True,mask=0,gain=0,trans=1,opacity=0),[*(albedo*.96),f0])
        equal(draw(pass_id=2,lights=True,backlight=True,prop=(0,0,1),mask=4,trans=1,opacity=0),[*(albedo*.96),f0])
        equal(draw(),[*((albedo+f0)),0])
        equal(draw(override=True),[*albedo,0])
        equal(draw(prop=(0,0,1)),[0,0,0,0])
        equal(draw(prop=(0,0,-.5)),[*((albedo+f0)*.5),0])
        equal(draw(rotation=np.diag([1,-1,-1])),[0,0,0,0])
        # Each directional slab works independently (no bleed across slab boundaries).
        for bin in range(6): equal(draw(bin=bin),[*((albedo+f0)),0])
        # Nonconstant production-depth slabs reproduce floor(coord*32),
        # including old compressed-z failures and both outer slab boundaries.
        for bin in (0,1,5):
            for z in (0.,.03,.032,.437,.5,.97,.999,1.):
                cell=min(int(z*32),31)
                equal(draw(bin=bin,field_z=z,gradient=True),[*((albedo+f0)*(cell+1)),0])
        # Twelve angular cells use two X tiles, never an unsupported depth384.
        for bin in range(12):
            equal(draw(bin=bin,columns=2,bins=12),[*(albedo+f0),0])
            equal(draw(bin=bin,columns=2,bins=12,spatial=True),[*(albedo+f0),0])
        for z,expected in ((0.,1.),(.5,16.5),(1.,32.)):
            equal(draw(bin=11,columns=2,bins=12,spatial=True,field_z=z,gradient=True),[*((albedo+f0)*expected),0])
        # Interpolate weighted measured moments, not equally weighted directions.
        equal(draw(bin=11,columns=2,bins=12,spatial=True,pair=True,override=True),[*(albedo*.5),0])
        # A denser 64-cell grid packs twelve bins into three X tiles and four
        # Z slabs, preserving WebGL2's guaranteed maximum 256 per dimension.
        for bin in range(12):
            equal(draw(bin=bin,columns=3,bins=12,grid_depth=64),[*(albedo+f0),0])
            equal(draw(bin=bin,columns=3,bins=12,grid_depth=64,spatial=True),[*(albedo+f0),0])
        # Allocate the actual 192x64x256 atlas and inspect both tile boundaries.
        for x,z,expected in ((0.,0.,1.),(1.,.5,32.5),(1.,1.,64.)):
            equal(draw(bin=11,columns=3,bins=12,grid_depth=64,grid_xy=64,
                       spatial=True,field_x=x,field_z=z,gradient=True),[*((albedo+f0)*expected),0])
        equal(draw(bin=11,columns=3,bins=12,grid_depth=64,spatial=True,pair=True,override=True),[*(albedo*.5),0])
        equal(draw(mask=0),[0,0,0,0]);equal(draw(mask=1),[0,0,0,0])
        equal(draw(pass_id=0),[0,0,0,0])
        equal(draw(pass_id=1,trans=1,opacity=0),[0,0,0,0])
        # Eligibility is ANDed with exact primary opaque visibility, not a mask alone.
        draw(program=1,depth_mode="primary",gain=0)
        equal(draw(depth_mode="equal",depth=.2),[0,0,0,0])
        equal(draw(depth_mode="equal",depth=0),[*(albedo+f0),0])
        equal(draw(depth_mode="equal",depth=0,mask=0),[0,0,0,0])
        # Transparent pass applies Fresnel once, and retains throughput at opacity=0.
        equal(draw(pass_id=2,trans=1,opacity=0),[f0,f0,f0,f0])
        equal(draw(pass_id=2,trans=1,opacity=0,prop=(0,0,1)),[*(albedo*.96),f0])
        equal(draw(pass_id=2,trans=1,opacity=0,mask=2,prop=(0,0,1)),[0,0,0,f0])
        # Host conservatively routes transmissive meshes here even at full opacity.
        equal(draw(pass_id=2,trans=1,opacity=1),[1,1,1,1])
        # Two sorted layers: each contributes once; near layer attenuates far field.
        far=draw(pass_id=2,trans=1,opacity=0,background=(.2,.2,.2,1))
        near=draw(pass_id=2,trans=1,opacity=0,background=False)
        equal(far,[f0+.2*.96]*3+[1]);equal(near,[f0+(f0+.2*.96)*.96]*3+[1])
        # Body scattering uses canonical thickness/translucence and field back-light.
        tx[2]=0;tx[5]=2;tx[15]=1
        depthtex=gl.glGenTextures(1);gl.glActiveTexture(gl.GL_TEXTURE3);gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY,depthtex)
        gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY,0,gl.GL_RGBA32F,1,1,1,0,gl.GL_RGBA,gl.GL_FLOAT,np.array([0,1,1,0],np.float32))
        for filt in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER): gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY,filt,gl.GL_NEAREST)
        for program in programs:
            gl.glUseProgram(program);gl.glUniform1i(gl.glGetUniformLocation(program,"uDepthUv"),3)
        equal(draw(pass_id=0,prop=(0,0,1),mask=1),[*(albedo*.04),0])
        equal(draw(pass_id=0,prop=(0,0,1),mask=2),[0,0,0,0])
        equal(draw(program=3,lights=True,backlight=True,mask=1,gain=0),[0,0,0,1])
        equal(draw(program=3,lights=True,backlight=True,mask=0,gain=0),[*(albedo*.04),1])
        equal(draw(pass_id=2,lights=True,backlight=True,prop=(0,0,1),mask=1),[*(albedo*.04),1])
        equal(draw(program=3,lights=True,backlight=True,mask=1,field_x=-.01,restrict=True),[*(albedo*.04),1])
        equal(draw(pass_id=2,lights=True,backlight=True,mask=1,field_x=1.01,restrict=True),[*(albedo*.04),1])
        # The resolve uses the canonical display function, not additive display RGB.
        resolve=gl.glGenTextures(1);gl.glActiveTexture(gl.GL_TEXTURE14);gl.glBindTexture(gl.GL_TEXTURE_2D,resolve)
        gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA32F,1,1,0,gl.GL_RGBA,gl.GL_FLOAT,np.array([.4,1,2,.7],np.float32))
        equal(draw(pass_id=3),[display(.4),display(1),display(2),.7])
        print(json.dumps({"status":"PASS","gpu":gl.glGetString(gl.GL_RENDERER).decode(),"checks":checks,"legacy_bit_identical":True,"transparent_layers":2}))
    finally:
        pygame.display.quit()

if __name__=="__main__": main()
