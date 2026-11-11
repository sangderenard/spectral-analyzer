"""Actual original live sensor, matched saved source population; diagnostic captures."""
from pathlib import Path
import sys,json,copy,hashlib,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    job=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    c=json.loads((job/'config.json').read_text());obj=json.loads((job.parent.parent/'marble_objects'/f"{c['object_digest']}.json").read_text())
    receipt=json.loads((job.parent.parent.parent/'live-receipt.json').read_text())['field'];sec=receipt['sources'];blob=(job/receipt['name']).read_bytes()
    sources=np.frombuffer(blob[sec['offset_bytes']:sec['offset_bytes']+sec['byte_length']],'<f4').reshape(-1,12).copy()
    saved=np.load(job/'payload.npz');vertices=saved['vertices'];ids=saved['material_ids'];pbr=saved['pbr']
    from camera_designer import studio_native_emitters as emit,marble_preview_worker as worker,marble_live_worker as live
    original_sampler=emit.native_emissive_sources;original_live=live.run_live_sensor
    emit.native_emissive_sources=lambda *a,**k:sources.copy()
    def capture(*args,**kwargs):
        result=original_live(*args,**kwargs)
        acc,renderer,field,dest,digest,config,prep,snapshot,atomic=args[:9]
        from OpenGL.GL import glFinish
        glFinish();assert acc._pass==1 and acc._row_off==0,(acc._pass,acc._row_off)
        final=snapshot(acc,dest,digest,999,0,prep,65536,acc._w*acc._h,True,transport='fixed:3',gamma=config.get('gamma',1),film_response=config.get('film_response'),display_params=config['live'].get('display'),publish_status=False)
        from camera_designer.studio_forward_field import ForwardFieldPublisher
        final['light_field']=ForwardFieldPublisher(dest,digest).snapshot(renderer,field,now=100000,ray_jobs=65536,sensor_samples=acc._pass,accumulation='controlled single saved source population')
        final['state']='complete';final['matched_source_sha256']=hashlib.sha256(field['source_buf'].tobytes()).hexdigest();final['sensor_passes']=acc._pass;final['sensor_samples_per_pixel']=acc._samples_per_pixel
        atomic(dest/'status.json',final);return final
    live.run_live_sensor=capture
    try:
        for mode in (('glass_rod',) if '--corrected' in sys.argv else ('legacy','glass_rod')):
            dest=out/mode;dest.mkdir(parents=True,exist_ok=True);cfg=copy.deepcopy(c);rows=pbr.copy()
            if mode=='legacy':
                a=obj['design']['optical_authoring']
                for item,row in zip(a['materials'],a['legacy_pbr']):rows[item['material']]=row
                for key in ('region_media','optical_media','boundary_media','boundary_transport'):cfg.pop(key,None)
            if '--corrected' in sys.argv:cfg['surface_transport']='studio-pbr-mixture-v1'
            cfg['camera']['resolution']=256
            cfg['emitter_sampling']['seed']=receipt['directional']['source_populations']['latest_seed']
            cfg['live']['display']={**cfg['live'].get('display',{}),'normalization':'manual','normalization_white':0.03744349243354245}
            cfg['live'].update(width=256,height=256,aspect=1,history_samples=1,checkpoint=False,lease_seconds=180,ray_batch_size=65536)
            # Native camera path and display settings otherwise unchanged.
            np.savez(dest/'payload.npz',vertices=vertices,material_ids=ids,pbr=rows)
            (dest/'config.json').write_text(json.dumps(cfg))
            started=time.monotonic();status=worker.run(dest/'payload.npz',dest/'config.json',dest/'image',seconds=30,max_epochs=4)
            result=dict(mode=mode,image=str(dest/'image'/status['image']),wall_seconds=time.monotonic()-started,status=status,limits='One matched256-square original-sensor sweep; original saved seed63 population; engine authoring differs as saved; not converged or field-overlay image.')
            (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(mode=mode,image=result['image'],seconds=result['wall_seconds'],coverage=status['light_field']['directional']['coverage'])),flush=True)
    finally:emit.native_emissive_sources=original_sampler;live.run_live_sensor=original_live
if __name__=='__main__':main()
