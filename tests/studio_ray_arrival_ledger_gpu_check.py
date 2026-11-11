"""Actual full saved scene, every forward bounce; process-local observer only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path);parser.add_argument('cache',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['glass_rod','legacy'],default='glass_rod')
    parser.add_argument('--jobs',type=int,default=65536)
    parser.add_argument('--population',type=int,default=63)
    parser.add_argument('--rng-seed',type=int,default=None)
    parser.add_argument('--source-records',type=Path,default=None)
    args=parser.parse_args()
    from OpenGL import GL
    import demo_pluck_gl as renderer
    from camera_designer.studio_ray_arrival_ledger import (ArrivalLedger,forward_shader,save_capture,event_dict,
        COUNTER_BUFFER_BINDING,PENDING_COUNT_WORD,PENDING_RAY_BUFFER_BINDING)
    from camera_designer.film_response import build_film_stack,normalize_film_response
    from camera_designer.studio_directional_field import ROOT_ESTIMATOR,readback
    config=json.loads((args.job/'config.json').read_text())
    population_index=args.population-int(config['emitter_sampling']['seed'])
    rng_seed=(7*population_index if args.rng_seed is None else args.rng_seed)
    if not 0<args.jobs<=65536 or not 0<=rng_seed<=2147483647:raise ValueError('unbounded diagnostic dispatch or invalid RNG seed')
    scene=dict(np.load(args.cache/f'{args.mode}.npz'))
    if args.source_records is not None:scene['sources']=np.load(args.source_records)
    elif args.population!=63:raise ValueError('the saved scene source records are population63; supply verified source records')
    roi=config['light_field_subject_region']
    bounds=(roi['bounds_min_m'],roi['bounds_max_m'])
    renderer._ensure_worker_gl_context()
    ledger=ArrivalLedger(bounds,roi_capacity=65536,other_capacity=393216)
    native_prog=renderer._prog
    def compile_observed(*pairs):
        compiled=[]
        for source,kind in pairs:
            if 'studio_record_segment(ro,hp,packet,currentMedium);' in source:
                source=forward_shader(source,roi_capacity=ledger.roi_capacity,other_capacity=ledger.other_capacity)
            compiled.append((source,kind))
        program=native_prog(*compiled)
        ledger.bind(program,args.population)
        return program
    renderer._prog=compile_observed
    state=None
    try:
        nodes,ids=renderer._build_gpu_bvh(scene['triangles'])
        film=build_film_stack(renderer.FILM_STACKS['em_rgb_fixed3'],normalize_film_response(config.get('film_response')))
        started=time.monotonic()
        margin=np.array([.02,.02,.04],np.float32)
        state=renderer._gpu_ray_field_prebuilt(scene['geom'],scene['shade'],scene['mat'],nodes,ids,scene['contexts'],scene['sources'],(scene['bmin']-margin,scene['bmax']+margin),
            dims=(96,96,96),dispatch_batch=512,max_bounces=int(config.get('max_bounces',8)),initial_rays=0,film_stack=film,native_pbr=scene['native_pbr'],
            sphere_regions=scene['spheres'],region_records=scene['regions'] if scene['regions'].size else None,
            record_directional=True,light_field_estimator=ROOT_ESTIMATOR,light_field_grid_size=64,light_field_stencil=12,
            light_field_region=roi,light_field_internal_media=True,directional_quantization_divisor=16777216)
        if state is None:raise RuntimeError('instrumented tracer compilation failed')
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,state['ssbo'][COUNTER_BUFFER_BINDING])
        counters_before=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,20),'<u4').copy()
        renderer._clear_prebuilt(state);ledger.clear();ledger.bind(state['prog'],args.population)
        state['_pump_seed']=rng_seed
        renderer._gpu_pump_prebuilt(state,n_rays=args.jobs)
        captured=ledger.read()
        # CounterBuf binding4 word4 is the queue count. PendingRayBuf binding6
        # has NO header; its first words are a ray origin and cannot be counts.
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,state['ssbo'][COUNTER_BUFFER_BINDING])
        counters_after=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,20),'<u4').copy()
        pending_count=int(counters_after[PENDING_COUNT_WORD]);pending_before=int(counters_before[PENDING_COUNT_WORD])
        pending_cap=512*4
        if pending_count:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER,state['ssbo'][PENDING_RAY_BUFFER_BINDING])
            pending=np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER,0,min(pending_count,pending_cap)*64),'<u4').reshape(-1,16).copy()
        else:pending=np.zeros((0,16),'<u4')
        field=readback(state['directional_record'])
        receipt=save_capture(args.output,captured,metadata=dict(mode=args.mode,source_population=args.population,
            source_population_index=population_index,
            source_population_note=f'Saved native source seed{args.population}, bounce RNG seed{rng_seed}. Live prebuilt starts at0 and advances7 per complete population; index{population_index}. No source directions or emitter weights changed. One population, not a replay of all22 populations.',
            config_sha256=hashlib.sha256((args.job/'config.json').read_bytes()).hexdigest(),
            scene_cache_sha256=hashlib.sha256((args.cache/f'{args.mode}.npz').read_bytes()).hexdigest(),
            diagnostic_module_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'camera_designer/studio_ray_arrival_ledger.py').read_bytes()).hexdigest(),
            rng_seed=rng_seed,scalar_grid=[96]*3,dispatch_batch=512,
            source_sha256=hashlib.sha256(scene['sources'].tobytes()).hexdigest(),source_roots=len(scene['sources']),
            ray_jobs=args.jobs,max_bounces=int(config.get('max_bounces',8)),roi=roi,
            ledger_bytes=ledger.nbytes,roi_capacity=ledger.roi_capacity,non_roi_capacity=ledger.other_capacity,
            pending_queue=dict(count_binding=COUNTER_BUFFER_BINDING,count_word=PENDING_COUNT_WORD,before_pump=pending_before,after_pump=pending_count,
                newly_queued=pending_count-pending_before,capacity=pending_cap,truncated=pending_count>pending_cap,
                retained_flags={str(int(k)):int(v) for k,v in zip(*np.unique(pending[:,3],return_counts=True))}),
            full_scene_triangle_count=len(scene['triangles']),analytic_sphere_count=len(scene['spheres']),
            atlas_nonzero_values=int(np.count_nonzero(field)),atlas_max=float(np.max(field)),
            renderer=GL.glGetString(GL.GL_RENDERER).decode(),wall_seconds=time.monotonic()-started,
            shader_source_sha256=hashlib.sha256(renderer._GPU_RAY_FIELD_REGION_CS.encode()).hexdigest()))
        examples=[event_dict(row) for row in captured[1][:5]]
        if not examples:examples=[event_dict(row) for row in captured[2][:5]]
        (args.output/'example-events.json').write_text(json.dumps(examples,indent=2))
        print(json.dumps(receipt),flush=True)
    finally:
        renderer._prog=native_prog
        if state is not None:renderer._destroy_prebuilt(state)
        ledger.close()


if __name__=='__main__':main()
