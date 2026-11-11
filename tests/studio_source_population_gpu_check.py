"""Bounded native source refresh + original NEE/GPU recorder integration proof."""
from pathlib import Path
import sys,json,runpy,types
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import demo_pluck_gl as r
    from OpenGL import GL
    from camera_designer.studio_native_emitters import native_emissive_sources,install_source_population,area_nee_records
    from camera_designer.studio_directional_field import ROOT_ESTIMATOR,commit_source_population,readback
    from camera_designer.studio_forward_field import ForwardFieldPublisher
    from camera_designer.film_response import build_film_stack,normalize_film_response
    internal='--internal' in sys.argv
    scene4=runpy.run_path(str(Path(__file__).with_name('test_studio_native_emitters.py')))['fixture']()
    g,s,mat,tri=scene4;ctx=np.zeros((0,8),np.float32)
    scene=(*scene4,ctx,None,([-1]*3,[4]*3),None)
    settings=dict(version=1,ray_budget=128,seed=42,refresh_each_sweep=True)
    src=native_emissive_sources(scene,[.45,.55,.65],ray_budget=128,seed=42)
    nodes,ids=r._build_gpu_bvh(tri)
    pbr=np.zeros((int(np.max(s[:,14].copy().view(np.uint32)))+1,16),np.float32);pbr[:,3]=.5;pbr[:,6]=1.5;pbr[:,7]=1
    r._ensure_worker_gl_context()
    film=build_film_stack(r.FILM_STACKS['em_rgb_fixed3'],normalize_film_response())
    state=r._gpu_ray_field_prebuilt(g,s,mat,nodes,ids,ctx,src,scene[6],dims=(8,8,8),max_bounces=1,initial_rays=0,film_stack=film,native_pbr=pbr,record_directional=True,light_field_estimator=ROOT_ESTIMATOR,light_field_grid_size=32,directional_quantization_divisor=16777216,light_field_internal_media=internal)
    sensor=types.SimpleNamespace(_nee_source_count=len(src),_ssbo_sources_nee=0)
    sensor.set_nee_spectral_sources=types.MethodType(r.SensorAccumulator.set_nee_spectral_sources,sensor)
    previous=None; hashes=[]
    try:
        for index in range(3):
            rows=install_source_population(r,state,sensor,scene,[.45,.55,.65],settings,index,ray_jobs=len(src))
            if previous is not None:
                assert not np.array_equal(rows[:,4:7],previous[:,4:7])
                np.testing.assert_array_equal(rows[:,10],previous[:,10])
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,state['ssbo'][5])
            np.testing.assert_array_equal(np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,rows.nbytes),np.float32).reshape(-1,12),rows)
            nee=area_nee_records(rows,*scene[:4]);GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,sensor._ssbo_sources_nee)
            np.testing.assert_array_equal(np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,nee.nbytes),np.float32).reshape(-1,12),nee)
            r._clear_prebuilt(state,preserve_directional_history=True);r._gpu_pump_prebuilt(state,n_rays=len(src))
            commit_source_population(state['directional_record'],rows,settings,index)
            hashes.append(state['directional_record']['progressive']['metadata']['latest_sources_sha256'])
            previous=rows.copy()
        record=state['directional_record'];assert record['progressive']['metadata']['count']==3
        assert np.any(readback(record)[...,:3])
        state['geometry_sha256']='b'*64;record['material_lineage']=[(i,row) for i,row in enumerate(pbr)]
        out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
        meta=ForwardFieldPublisher(out,'a'*64).snapshot(r,state,now=0,ray_jobs=len(src)*3)
        assert meta['schema']=='studio.forward-field.v6'
        (out/'status.json').write_text(json.dumps(dict(state='complete',digest='a'*64,light_field=meta)))
        (out/'config.json').write_text(json.dumps(dict(receive_light_field=True,transport='fixed:3',geometry_sha256='b'*64,light_field_estimator=ROOT_ESTIMATOR,light_field_grid_size=32,light_field_stencil=12,light_field_schema='studio.forward-field.v6',emitter_sampling=settings,light_field_internal_media=internal)))
        r._clear_prebuilt(state);assert 'progressive' not in record
        print(json.dumps(dict(result='PASS',renderer=GL.glGetString(GL.GL_RENDERER).decode(),roots=len(src),population_hashes=hashes,coverage=meta['directional']['coverage'])))
    finally:
        if sensor._ssbo_sources_nee:GL.glDeleteBuffers(1,[sensor._ssbo_sources_nee])
        r._destroy_prebuilt(state)
if __name__=='__main__':main()
