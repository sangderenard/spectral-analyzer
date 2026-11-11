"""Independent camera-adjoint audit using the unchanged native sensor worker."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=['glass_rod','legacy'],default='glass_rod')
    p.add_argument('--nee',action='store_true');p.add_argument('--baseline-shader',type=Path,default=None);args=p.parse_args()
    import demo_pluck_gl as renderer
    from camera_designer import studio_native_emitters as emit,marble_preview_worker as worker,marble_live_worker as live
    from camera_designer.studio_ray_arrival_ledger import ArrivalLedger,camera_shader,save_capture,event_dict
    config=json.loads((args.job/'config.json').read_text());cfg=copy.deepcopy(config)
    obj=json.loads((args.job.parent.parent/'marble_objects'/f"{config['object_digest']}.json").read_text())
    scene=np.load(args.cache/f'{args.mode}.npz');saved=np.load(args.job/'payload.npz');rows=saved['pbr'].copy()
    if args.mode=='legacy':
        authored=obj['design']['optical_authoring']
        for material,row in zip(authored['materials'],authored['legacy_pbr']):rows[material['material']]=row
        for key in ('region_media','optical_media','boundary_media','boundary_transport'):cfg.pop(key,None)
    # Same actual live camera framing/pitch/resolution; bound to one sweep.
    cfg['live'].update(history_samples=1,checkpoint=False,lease_seconds=180)
    cfg['emitter_sampling']['seed']=63
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'payload.npz',vertices=saved['vertices'],material_ids=saved['material_ids'],pbr=rows)
    (args.output/'config.json').write_text(json.dumps(cfg))
    renderer._ensure_worker_gl_context()
    roi=config['light_field_subject_region']
    ledger=ArrivalLedger((roi['bounds_min_m'],roi['bounds_max_m']),roi_capacity=131072,other_capacity=131072)
    nee=None
    if args.nee:
        from camera_designer.studio_nee_connection_ledger import NeeLedger,connection_shader
        nee=NeeLedger((roi['bounds_min_m'],roi['bounds_max_m']))
    original_prog=renderer._prog;original_sampler=emit.native_emissive_sources;original_live=live.run_live_sensor
    programs=[]
    def observed_prog(*pairs):
        rewritten=[];is_camera=False
        for source,kind in pairs:
            if 'int   hit_id = nearest_hit(ro, rd, surf_t);' in source:
                if args.baseline_shader is not None:source=args.baseline_shader.read_text(encoding='utf-8')
                source=camera_shader(source,roi_capacity=ledger.roi_capacity,other_capacity=ledger.other_capacity);is_camera=True
                if nee is not None:source=connection_shader(source,nee.capacity)
            rewritten.append((source,kind))
        result=original_prog(*rewritten)
        if is_camera:
            ledger.bind(result,63);programs.append(result)
            if nee is not None:nee.bind(result,63)
        return result
    def audit_live(*pos,**kw):
        ledger.clear()
        if nee is not None:nee.clear()
        result=original_live(*pos,**kw)
        acc=pos[0];capture=ledger.read()
        receipt=save_capture(args.output/'camera-adjoint',capture,metadata=dict(mode=args.mode,
            channel='camera-adjoint',forward_energy_atlas_written=False,
            source_sha256=hashlib.sha256(scene['sources'].tobytes()).hexdigest(),
            config_sha256=hashlib.sha256((args.job/'config.json').read_bytes()).hexdigest(),
            diagnostic_config_sha256=hashlib.sha256((args.output/'config.json').read_bytes()).hexdigest(),
            camera=cfg['camera'],resolution=[acc._w,acc._h],passes=acc._pass,row_offset=acc._row_off,
            samples_per_pixel=acc._samples_per_pixel,roi=roi,ledger_bytes=ledger.nbytes,
            meaning='Actual camera main-loop adjoint throughput, starting at1 per band. Not source-emitted ray energy and never added to forward atlas. Direct NEE shadow connections are not captured by this path ledger.'))
        examples=[event_dict(row) for row in capture[1][:12]]
        (args.output/'camera-adjoint'/'example-events.json').write_text(json.dumps(examples,indent=2))
        print(json.dumps(receipt),flush=True)
        if nee is not None:print(json.dumps(nee.save(args.output/'nee-connections')),flush=True)
        return result
    renderer._prog=observed_prog;emit.native_emissive_sources=lambda *a,**k:scene['sources'].copy();live.run_live_sensor=audit_live
    try:
        worker.run(args.output/'payload.npz',args.output/'config.json',args.output/'image',seconds=30,max_epochs=4)
    finally:
        renderer._prog=original_prog;emit.native_emissive_sources=original_sampler;live.run_live_sensor=original_live;ledger.close()
        if nee is not None:nee.close()


if __name__=='__main__':main()
