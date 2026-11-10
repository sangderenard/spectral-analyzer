"""Bounded real worker Live gate from completed optical payloads; no server mutation."""
import copy,json,shutil,subprocess,sys,time
from pathlib import Path
from PIL import Image
import numpy as np

def run(root,out):
    receipts=[]
    for mode,digest in [('glass_rod','4453fccc8717c9b771b36c7b13b40066d0a694d3f4753a67ad56c770ad2eebb4'),('injection_dye','1936c40cea98ccc109381225966476b129970b6e42e75d21bbf955271036df9b')]:
        source=root/'build/meowdoku/marble_photos'/digest;c=json.loads((source/'config.json').read_text());assert c['region_media']['regions'][1]['mode']==mode
        dest=out/mode;dest.mkdir(parents=True,exist_ok=True);image=dest/'image';image.mkdir(exist_ok=True)
        shutil.copyfile(source/'payload.npz',dest/'payload.npz')
        c['camera']['resolution']=128;c['max_bounces']=8;c['bounce_mode']='manual'
        c['live']={'width':64,'height':64,'aspect':1.,'history_samples':2,'checkpoint':True,'lease_seconds':60,'ray_batch_size':128,'display':{}}
        (dest/'config.json').write_text(json.dumps(c))
        cam=copy.deepcopy(c['camera']);cam['eye_m'][0]+=.001
        (image/'live-control.json').write_text(json.dumps({'revision':1,'expires_at':time.time()+90,'camera':cam,'live':c['live']}))
        started=time.monotonic()
        with (dest/'worker.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','camera_designer.marble_preview_worker','--payload',str(dest/'payload.npz'),'--config',str(dest/'config.json'),'--out',str(image),'--seconds','1','--max-epochs','4'],stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode:raise RuntimeError((dest/'worker.log').read_text()[-5000:])
        s=json.loads((image/'status.json').read_text());assert s['live_revision']==1,s
        assert s['image'] and s['sensor_sites']>0 and s['total_samples']>0,s
        assert s['camera_settings']['eye_m']==cam['eye_m']
        if mode=='glass_rod':assert s['rod_bundle_diagnostic']['unresolved_samples']==0
        pixels=np.asarray(Image.open(image/s['image']));assert pixels.max()>0
        receipt={'mode':mode,'wall_seconds':time.monotonic()-started,'status':s,'region_contract_preserved':c['region_media']==json.loads((source/'config.json').read_text())['region_media'],'limits':'64px/128forward rays,4epochs real Live plus camera rebuild; not frame-rate/convergence guarantee.'};(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));receipts.append(receipt)
        print(json.dumps({'mode':mode,'state':s['state'],'revision':s['live_revision'],'samples':s['total_samples'],'diagnostic':s.get('rod_bundle_diagnostic'),'seconds':receipt['wall_seconds']}),flush=True)
    return receipts
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
