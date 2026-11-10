"""Bounded isolated worker pause/resume gate using a saved optical fixture."""
import json,shutil,subprocess,sys,time,hashlib
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]/'turing'
    source=root/'build/meowdoku/marble_photos/4453fccc8717c9b771b36c7b13b40066d0a694d3f4753a67ad56c770ad2eebb4'
    out=Path(__file__).resolve().parents[1]/'build/dot-live-yield-20261109';out.mkdir(parents=True,exist_ok=True)
    image=out/'image';image.mkdir(exist_ok=True)
    config=json.loads((source/'config.json').read_text());shutil.copyfile(source/'payload.npz',out/'payload.npz')
    config['camera']['resolution']=128;config['max_bounces']=2;config['bounce_mode']='manual'
    config['live']={'width':64,'height':64,'aspect':1.,'history_samples':2,'checkpoint':True,'lease_seconds':60,'ray_batch_size':128,'display':{}}
    (out/'config.json').write_text(json.dumps(config))
    control={'revision':0,'expires_at':time.time()+60,'camera':config['camera'],'live':config['live'],'scheduler_generation':0,'scheduler_state':'running'}
    def write():
        tmp=image/'live-control.check.tmp';tmp.write_text(json.dumps(control));tmp.replace(image/'live-control.json')
    write()
    def status():
        try:return json.loads((image/'status.json').read_text())
        except (OSError,ValueError):return {}
    def wait(predicate):
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            s=status()
            if predicate(s):return s
            if proc.poll() is not None:raise RuntimeError('Isolated worker ended: '+(out/'worker.log').read_text()[-2500:])
            time.sleep(.01)
        raise TimeoutError('Isolated worker state timed out')
    with (out/'worker.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','camera_designer.marble_preview_worker','--payload',str(out/'payload.npz'),'--config',str(out/'config.json'),'--out',str(image),'--seconds','1','--max-epochs','100000'],stdout=log,stderr=subprocess.STDOUT)
        try:
            first=wait(lambda s:s.get('state')=='live' and s.get('total_samples',0)>0)
            control.update(scheduler_generation=1,scheduler_state='paused');write()
            paused=wait(lambda s:s.get('state')=='yielded' and s.get('scheduler_generation')==1)
            assert paused['retained_pass']>0 and paused['retained_row']==0
            assert paused['gpu_dispatch_yielded'] and not paused['gpu_memory_released']
            retained=(paused['sensor_sites'],paused['forward_ray_jobs'],paused['retained_pass'])
            raw=(image/paused['image']).read_bytes();time.sleep(.2)
            assert status()['state']=='yielded' and (image/paused['image']).read_bytes()==raw
            control.update(scheduler_generation=2,scheduler_state='running',expires_at=0);write()
            waiting=wait(lambda s:s.get('state')=='yielded' and s.get('scheduler_generation')==2 and s.get('client_lease_expired'))
            assert retained==(waiting['sensor_sites'],waiting['forward_ray_jobs'],waiting['retained_pass'])
            control['expires_at']=time.time()+60;write()
            resumed=wait(lambda s:s.get('state')=='live' and s.get('scheduler_generation')==2 and s.get('total_samples',0)>paused['retained_pass'])
            assert resumed['sensor_sites']>paused['sensor_sites'] and resumed['checkpoint_samples']>=paused['retained_pass']-2
            control['expires_at']=0;write();proc.wait(timeout=15);assert proc.returncode==0
            receipt={'real_worker':True,'retained_pass':paused['retained_pass'],'resumed_samples':resumed['total_samples'],'image_stable_while_yielded':True,'expired_resume_retained':True,'same_process':True,'retained_vram':True,'retained_image_sha256':hashlib.sha256(raw).hexdigest(),'limits':'64px,128 forward rays,saved glass-rod fixture, no live server calls'}
            (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
        finally:
            if proc.poll() is None:
                # Only our isolated test worker, never the live service.
                proc.terminate();proc.wait(timeout=10)

if __name__=='__main__':main()
