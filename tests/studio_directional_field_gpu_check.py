"""Bounded real-GPU directional export proof; no active exposure should be running."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import demo_pluck_gl as r
    from OpenGL import GL
    from camera_designer.film_response import build_film_stack, normalize_film_response
    from camera_designer.studio_forward_field import ForwardFieldPublisher
    from camera_designer.studio_directional_field import readback, recording_shader
    import faulthandler
    faulthandler.enable()
    stencil=int(sys.argv[3]) if len(sys.argv)>3 else 12
    r._ensure_worker_gl_context()
    region_program=r._prog((recording_shader(r._GPU_RAY_FIELD_REGION_CS,stencil=stencil),GL.GL_COMPUTE_SHADER))
    assert GL.glGetProgramiv(region_program,GL.GL_LINK_STATUS)
    GL.glDeleteProgram(region_program)
    triangles=np.array([[[-2,-2,0],[2,-2,0],[0,2,0]]],np.float32)
    geom=np.zeros((1,16),np.float32)
    geom[:,:3]=triangles[:,0];geom[:,4:7]=triangles[:,1]-triangles[:,0]
    geom[:,8:11]=triangles[:,2]-triangles[:,0];geom[:,14]=1
    shade=np.zeros((1,16),np.float32);shade[:,:4]=[.2,0,1,1.5];shade[:,4:8]=[.2,0,1,1];shade[:,8:11]=1
    sources=np.zeros((3,12),np.float32);sources[:,:3]=[0,0,.4];sources[:,3]=1
    direction=np.array([.2,.1,-1.],np.float32);direction/=np.linalg.norm(direction)
    sources[:,4:7]=direction;sources[:,8]=2.998e17/np.array([650,550,450]);sources[:,10]=1
    nodes,ids=r._build_gpu_bvh(triangles)
    film=build_film_stack(r.FILM_STACKS['em_rgb_fixed3'],normalize_film_response())
    pbr=np.zeros((1,16),np.float32);pbr[:,:3]=1;pbr[:,3]=.5;pbr[:,6]=1.5;pbr[:,7]=1
    ray_count=int(sys.argv[2]) if len(sys.argv)>2 else 128
    if not 128<=ray_count<=131072:raise ValueError("bounded fixture rays must be128..131072")
    baseline=None
    for enabled in [False,True]:
        state=r._gpu_ray_field_prebuilt(geom,shade,np.zeros((32,12),np.float32),nodes,ids,
            np.zeros((0,8),np.float32),sources,([-.5,-.5,-.5],[.5,.5,.5]),dims=(8,8,8),
            dispatch_batch=128,max_bounces=1,initial_rays=0,film_stack=film,record_directional=enabled,native_pbr=pbr,light_field_stencil=stencil)
        assert state is not None
        try:
            r._clear_prebuilt(state);r._gpu_pump_prebuilt(state,n_rays=ray_count);GL.glFinish()
            scalar=r._gpu_readback_prebuilt(state['tex_bands'],state['dims'])
            if not enabled:baseline=scalar;continue
            state["directional_record"]["material_lineage"]=[(0,pbr[0].copy())]
            np.testing.assert_array_equal(scalar,baseline)
            atlas=readback(state['directional_record'])
            active=np.any(atlas[...,:3]!=0,axis=-1)
            from camera_designer.studio_directional_field import stencil_directions
            expected_bin=int(np.argmax(stencil_directions(stencil)@direction))
            assert active.any() and not np.delete(active,expected_bin,axis=-1).any()
            from studio_directional_tracklength_gpu_check import oracle
            endpoint=sources[0,:3]+direction*(-sources[0,2]/direction[2])
            expected_counts=oracle(sources[0,:3],endpoint,[-.5]*3,[.5]*3,1)
            # Per-component moment rounding plus unit-ball correction is
            # bounded by1.5/minimum segment-voxel integer weight.
            direction_error=1.5/expected_counts[expected_counts>0].min()+1e-6
            np.testing.assert_allclose(atlas[...,3:][active],np.tile(direction,(int(active.sum()),1)),atol=direction_error)
            from contextlib import nullcontext
            destination=Path(sys.argv[1]) if len(sys.argv)>1 else None
            if destination is not None:destination.mkdir(parents=True,exist_ok=True)
            state['geometry_sha256']='b'*64
            with (nullcontext(str(destination)) if destination is not None else tempfile.TemporaryDirectory(prefix='studio-vector-export-')) as directory:
                pub=ForwardFieldPublisher(directory,'a'*64)
                meta=pub.snapshot(r,state,now=0,ray_jobs=ray_count)
                (Path(directory)/'status.json').write_text(json.dumps({'digest':'a'*64,'state':'complete','light_field':meta}))
                (Path(directory)/'config.json').write_text(json.dumps({'geometry_sha256':'b'*64,'receive_light_field':True,'transport':'fixed:3'}))
                data=(Path(directory)/meta['name']).read_bytes()
                assert meta['schema']=='studio.forward-field.v3'
                d=meta['directional'];np.testing.assert_array_equal(np.frombuffer(data,'<f4',count=atlas.size,offset=d['offset_bytes']).reshape(atlas.shape),atlas)
                np.testing.assert_array_equal(readback(state['directional_record']),atlas)
                assert meta['source_pbr16']['offset_bytes']+meta['source_pbr16']['byte_length']==len(data)
                assert meta['source_pbr16']['material_ids']==[0]
                # Force an actual signed atomic overflow in a known occupied
                # line accumulator, then prove it cannot abort primary export.
                from camera_designer.studio_directional_field import WORDS
                occupied=int(np.flatnonzero(active)[0])*6
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,state['directional_record']['buffer'])
                GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,occupied*4,4,np.array([2147483647],np.int32))
                r._gpu_pump_prebuilt(state,n_rays=128);GL.glFinish()
                fallback=pub.snapshot(r,state,now=1,ray_jobs=ray_count+128)
                assert fallback['schema']=='studio.forward-field.v1'
                assert fallback['directional_unavailable']['reason']=='signed-accumulator-overflow'
                from camera_designer.studio_forward_field import field_to_linear_srgb
                np.testing.assert_array_equal(np.fromfile(Path(directory)/fallback['name'],'<f4').reshape(8,8,8,3),
                    field_to_linear_srgb(r._gpu_readback_prebuilt(state['tex_bands'],state['dims']),(8,8,8)))
                r._clear_prebuilt(state)
                assert not readback(state['directional_record']).any()
                recovered=pub.snapshot(r,state,now=2,ray_jobs=0)
                assert recovered['schema']=='studio.forward-field.v3' and 'directional_unavailable' not in recovered
                print(json.dumps(dict(result='PASS',renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                    rays=ray_count,stencil=stencil,recorded_voxel_bins=int(active.sum()),actual_direction=direction.tolist(),
                    bytes=len(data),checks=['canonical scalar unchanged','actual oblique direction recorded','only nearest declared direction bin','exact immutable atlas layout','fresh clear resets moments','forced signed atomic overflow preserves primary scalar export','clear resumes directional v3'],metadata=meta),indent=2))
        finally:r._destroy_prebuilt(state)

if __name__=='__main__':main()
