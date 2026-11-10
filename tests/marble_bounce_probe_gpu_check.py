"""Opt-in tiny GPU verification. Run directly, never through pytest's GL stubs.

Must first verify that the production photo/live GPU worker is idle.
"""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from camera_designer.marble_bounce_probe import run_bounce_probe, scatter_sources


def box_scene(scale=1., *, opacity=1., reflectivity=.5, ior=1.5):
    triangles=[]; normals=[]
    for axis in range(3):
        free=[i for i in range(3) if i!=axis]
        for sign in [-1,1]:
            points=np.zeros((4,3),np.float32)
            points[:,axis]=sign*scale
            points[:,free[0]]=[-scale,scale,scale,-scale]
            points[:,free[1]]=[-scale,-scale,scale,scale]
            normal=np.zeros(3);normal[axis]=sign
            for ids in [(0,1,2),(0,2,3)]:
                triangle=points[list(ids)]
                if np.dot(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]),normal)<0:
                    triangle=triangle[[0,2,1]]
                triangles.append(triangle);normals.append(normal)
    triangles=np.array(triangles,np.float32)
    geom=np.zeros((12,16),np.float32)
    geom[:,:3]=triangles[:,0];geom[:,4:7]=triangles[:,1]-triangles[:,0]
    geom[:,8:11]=triangles[:,2]-triangles[:,0];geom[:,12:15]=normals
    shade=np.zeros((12,16),np.float32)
    shade[:,0]=shade[:,4]=reflectivity;shade[:,1]=shade[:,5]=0
    shade[:,3]=ior;shade[:,7]=opacity;shade[:,8:11]=1;shade[:,13]=-1
    return geom,shade,triangles


def explicit_bubble_scene():
    # A real triangulated air cavity inside a closed glass host. Use the
    # production medium-pair packer and MatBuf layout, not a substitute tracer.
    from camera_designer.studio_boundary_media import install_boundary_pairs
    from mat_flags import MAX_SPECTRAL_BANDS
    outer,shade,triangles=box_scene(opacity=0.)
    vertices=.25*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)
    faces=[(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    cavity=vertices[np.array(faces)]
    normals=np.cross(cavity[:,1]-cavity[:,0],cavity[:,2]-cavity[:,0])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    assert np.all(np.sum(normals*cavity.mean(axis=1),axis=1)>0)
    inner=np.zeros((8,16),np.float32);inner[:,:3]=cavity[:,0]
    inner[:,4:7]=cavity[:,1]-cavity[:,0];inner[:,8:11]=cavity[:,2]-cavity[:,0];inner[:,12:15]=normals
    geom=np.concatenate((outer,inner));shade=np.concatenate((shade,np.repeat(shade[:1],8,axis=0)))
    triangles=np.concatenate((triangles,cavity))
    pairs=np.array([[-1,0]]*12+[[0,-1]]*8,np.int32)
    geom=install_boundary_pairs(geom,pairs,np.ones(20,bool),{0:0});geom[0,11]=4097.
    mat=np.zeros((MAX_SPECTRAL_BANDS,12),np.float32)
    for band,wavelength in enumerate([.450,.550,.650]):mat[band,0]=2.998e14/wavelength;mat[band,7]=1.5
    return geom,shade,triangles,mat


def run():
    import demo_pluck_gl as renderer
    from OpenGL import GL as gl
    renderer._ensure_worker_gl_context()
    results=[]
    for label,parameters in [('diffuse',{}),('slow_mirror',{'scale':.001,'reflectivity':.98}),
                             ('trapped_glass',{'opacity':0.,'ior':1.5}),('explicit_air_bubble',{})]:
        if label=='explicit_air_bubble':geom,shade,triangles,mat=explicit_bubble_scene()
        else:
            geom,shade,triangles=box_scene(**parameters)
            mat=np.zeros((1,12),np.float32)
        scale=parameters.get('scale',1.)
        bounds=(np.full(3,-scale),np.full(3,scale))
        nodes,ids=renderer._build_gpu_bvh(triangles)
        source=scatter_sources(bounds,[.450,.550,.650])[0]
        field=renderer._gpu_ray_field_prebuilt(geom,shade,mat,nodes,ids,
            np.zeros((0,8),np.float32),source,bounds,dims=(8,8,8),dispatch_batch=64,
            max_bounces=1,initial_rays=128)
        try:
            def field_bytes():
                gl.glBindTexture(gl.GL_TEXTURE_3D,field['tex_bands'][0])
                return bytes(gl.glGetTexImage(gl.GL_TEXTURE_3D,0,gl.GL_RED_INTEGER,gl.GL_UNSIGNED_INT))
            before=field_bytes()
            result=run_bounce_probe(renderer,field,bounds,[.450,.550,.650],n_triangles=len(triangles),n_nodes=len(nodes))
            assert field_bytes()==before,'probe modified production field'
            actual=np.zeros(1,np.int32)
            gl.glGetUniformiv(field['prog'],gl.glGetUniformLocation(field['prog'],b'uMaxBounces'),actual)
            assert actual[0]==1,'probe changed render depth uniform'
            if label=='diffuse': assert result['chosen_depth']<=16 and not result['warning'],result
            if label=='slow_mirror': assert 100 <= result['chosen_depth'] <= 200 and result['paths_at_render_cap'] > 0,result
            if label in ('trapped_glass','explicit_air_bubble'): assert result['paths_at_probe_limit']>0 and result['chosen_depth']==200,result
            results.append({'scene':label,'field_unchanged':True,'render_uniform_unchanged':True,**result})
            print(json.dumps(results[-1]),flush=True)
        finally:
            renderer._destroy_prebuilt(field)
    target=Path('build/dot-auto-bounces-20261109/gpu-result.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results

if __name__=='__main__':run()
