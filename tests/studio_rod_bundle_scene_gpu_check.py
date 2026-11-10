"""Full canonical geometry -> native packing/BVH -> NVIDIA separator event gate."""
import json,subprocess
from pathlib import Path
import numpy as np
import demo_pluck_gl as r
from OpenGL import GL
from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
from camera_designer.studio_region_gpu import region_records_from_geometry,validate_region_records
from camera_designer.studio_rod_bundle_gpu import pack_bundle_records


def run(turing_root,count=2,transport_path="camera",oriented=False):
    js="""import {catEyeMarble} from './examples/meowdoku/web/marble_mesh.js';
import {regionMediaFromGeometry} from './examples/meowdoku/web/marble_optical_authoring.js';
const m=catEyeMarble({glass:0,eye:1,accent:2},{kind:'cat_eye',ribbonLayout:'axial',strandCount:2,rings:96,segments:128,eyeCurve:0,ribbonTwist:.9,ribbonPhase:0});
const f=[...m.interior,...m.glass];console.log(JSON.stringify({v:f.map(x=>x.slice(0,3)),ids:f.map(x=>x[3]),regions:regionMediaFromGeometry(m,0,'glass_rod')}));"""
    js=js.replace('strandCount:2',f'strandCount:{count}')
    if oriented:js=js.replace('eyeCurve:0', 'eyeCurve:.18,orientationYaw:.6108652381980153,orientationPitch:-.3665191429188092,orientationRoll:1.1868238913561442')
    data=json.loads(subprocess.run(['node','--input-type=module','-e',js],cwd=turing_root,check=True,capture_output=True,text=True,timeout=30).stdout)
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.full(3,-1,np.float32),np.full(3,1,np.float32)),np.zeros((0,12),np.float32))
    pbr=np.zeros((3,16),np.float32);pbr[:,5]=1;pbr[:,6]=[1.52,1.54,1.6]
    optics={'version':1,'materials':[dict(material=i,wavelengths_nm=[400,700],transmittance=[.99,.99],reference_thickness_mm=1,concentration=1) for i in range(3)]}
    geom,shade,mat,tris,*_=append_saved_studio_mesh_to_gpu_scene(scene,np.asarray(data['v'],np.float32),np.asarray(data['ids']),pbr,scale_m=.025,center_m=[0,0,.18],wavelengths_um=[.55],region_media=data['regions'],optical_media=optics)
    records=pack_bundle_records(region_records_from_geometry(geom),data['regions']['axial_bundle'],scale_m=.025,center_m=[0,0,.18],host_triangle=int(np.flatnonzero(geom[:,3]==-1)[0]))
    validate_region_records(records,geom,shade,len(mat)//32)
    nodes,ids=r._build_gpu_bvh(tris)
    # Keep actual full shader helpers/traversal, replace only the dispatch body.
    full_source=r._GPU_SENSOR_REGION_CS if transport_path=='camera' else r._GPU_RAY_FIELD_REGION_CS
    src=full_source.rsplit('void main()',1)[0]+'''
    layout(std430,binding=11) buffer Proof {vec4 result[];};
    void main(){if(gl_GlobalInvocationID.x>1u || gl_GlobalInvocationID.y>0u)return;
        uint i=gl_GlobalInvocationID.x;vec3 ro=vec3(.0005,0,.18025),rd=vec3(-1,0,0);
        int medium=studio_origin_medium(ro,rd),before=medium;float best;
        int hit=nearest_hit(ro,rd,best);bool iscut=studio_partition_hit;float factor=1.;bool ok=false;
        if(hit>=0){vec3 hp=ro+rd*best;Tri tri=studio_hit_tri(hit,hp);studio_region_hit_position=hp;
            ok=studio_boundary_step(tri,rd,299792458.0/550e-9,i==0u?.999:0.,medium,factor);}
        result[i*3]=vec4(float(hit),best,float(before),float(medium));
        result[i*3+1]=vec4(rd,factor);result[i*3+2]=vec4(iscut?1.:0.,ok?1.:0.,float(studio_active[0]),0.);
    }'''
    bundle=data['regions']['axial_bundle'];frame=bundle['frame']
    local=np.array([bundle['eye_curve']+.02,0,.01])
    world=.025*(np.asarray(frame['origin'])+frame['scale']*(local@np.asarray(frame['axes'])))+np.array([0,0,.18])
    direction=np.array([-1,0,0])@np.asarray(frame['axes'])
    vec=lambda a:'vec3('+','.join(f'{float(x):.12g}' for x in a)+')'
    src=src.replace('vec3(.0005,0,.18025),rd=vec3(-1,0,0)',vec(world)+',rd='+vec(direction))
    r._ensure_worker_gl_context();program=r._prog((src,GL.GL_COMPUTE_SHADER));buffers=[]
    try:
        for binding,arr in ((0,geom),(1,nodes),(2,ids),(9,shade),(10,mat),(11,np.zeros((6,4),np.float32)),(12,np.zeros((1,8),np.float32)),(15,records)):
            b=int(GL.glGenBuffers(1));buffers.append((binding,b));GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,b);GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER,arr.nbytes,arr,GL.GL_STATIC_DRAW);GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER,binding,b)
        GL.glUseProgram(program)
        for name,value in [('uNodeCount',len(nodes)),('uStudioRegionCount',count+1),('uStudioSphereCount',0)]:GL.glUniform1i(GL.glGetUniformLocation(program,name),value)
        GL.glDispatchCompute(1,1,1);GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT);GL.glFinish()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,dict(buffers)[11]);out=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,96),np.float32).reshape(2,3,4)
        np.testing.assert_array_equal(out[:,2,:2],1)
        import math
        before=int(math.floor(math.atan2(.01,.02)/(math.pi/count)+.5))
        after=before+1
        expected=(.02-.01/math.tan((before+.5)*math.pi/count))*.025*frame['scale']
        np.testing.assert_allclose(out[:,0,1],expected,atol=1e-7)
        np.testing.assert_array_equal(out[:,0,2],records[before+1,1])
        np.testing.assert_array_equal(out[:,0,3],[records[after+1,1],records[before+1,1]])
        ni=1.54 if before%2==0 else 1.6;nt=1.54 if after%2==0 else 1.6
        np.testing.assert_allclose(out[0,1,3],(ni/nt)**2,rtol=3e-6)
        print(f'PASS {transport_path} oriented={oriented} {count}-rod canonical {len(tris)} triangle native BVH: new internal cut, transmission and reflection; region IDs/material mappings preserved.',flush=True)
    finally:
        GL.glUseProgram(0);GL.glDeleteBuffers(len(buffers),[b for _,b in buffers]);GL.glDeleteProgram(program)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--turing-root',required=True);a=p.parse_args();
    for path in ('camera','forward'):
        for count in (2,3,4,7):run(Path(a.turing_root),count,path)
        run(Path(a.turing_root),3,path,oriented=True)
