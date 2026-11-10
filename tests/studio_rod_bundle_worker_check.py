"""Actual worker image/performance receipt for opt-in axial rod transport."""
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image


def run(turing_root,out):
    js="""import {catEyeMarble} from './examples/meowdoku/web/marble_mesh.js';
import {regionMediaFromGeometry} from './examples/meowdoku/web/marble_optical_authoring.js';
const m=catEyeMarble({glass:0,eye:1,accent:2},{kind:'cat_eye',ribbonLayout:'axial',strandCount:3,rings:96,segments:128,eyeCurve:.18,ribbonTwist:.9,ribbonPhase:0});
const f=[...m.interior,...m.glass];console.log(JSON.stringify({v:f.map(x=>x.slice(0,3)),ids:f.map(x=>x[3]),regions:regionMediaFromGeometry(m,0,'glass_rod')}));"""
    data=json.loads(subprocess.run(['node','--input-type=module','-e',js],cwd=turing_root,check=True,capture_output=True,text=True,timeout=30).stdout)
    out.mkdir(parents=True,exist_ok=True)
    pbr=np.zeros((3,16),np.float32);pbr[:,5]=1;pbr[:,6]=[1.52,1.54,1.6];pbr[:,:3]=1
    payload=out/'payload.npz';np.savez(payload,vertices=np.asarray(data['v'],np.float32),material_ids=np.asarray(data['ids'],np.int32),pbr=pbr)
    optics={'version':1,'materials':[dict(material=i,wavelengths_nm=[400,450,550,650,700],transmittance=t,reference_thickness_mm=1,concentration=1) for i,t in enumerate(([.999]*5,[.8,.8,.2,.05,.05],[.05,.05,.2,.8,.8]))]}
    # One real emissive rectangular surface behind the marble, facing camera.
    f=np.array([[[-.1,-.1,.3],[.1,.1,.3],[.1,-.1,.3]],[[-.1,-.1,.3],[-.1,.1,.3],[.1,.1,.3]]],np.float32)
    fp=np.zeros((1,16),np.float32);fp[0,:3]=1;fp[0,7]=1;fp[0,8:11]=30
    config={'transport':'fixed:3','scale_m':.025,'object_center_m':[0,0,.18],
      'sphere_coverage':'mesh','max_bounces':12,'bounce_mode':'manual',
      'camera':{'resolution':128,'eye_m':[.04,.02,.07],'target_m':[0,0,.18],'fov_degrees':30,'aperture_radius_m':.001},
      'fixtures':{'vertices':f.reshape(-1).tolist(),'material_ids':[0,0],'pbr':fp.reshape(-1).tolist()},
      'region_media':data['regions'],'optical_media':optics}
    cfg=out/'config.json';cfg.write_text(json.dumps(config))
    started=time.monotonic()
    cmd=[sys.executable,'-m','camera_designer.marble_preview_worker','--payload',str(payload),'--config',str(cfg),'--out',str(out/'image'),'--seconds','2','--max-epochs','2','--snapshot-seconds','.25']
    with (out/'worker.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    wall=time.monotonic()-started
    status=json.loads((out/'image/status.json').read_text())
    receipt={'exit_code':result.returncode,'wall_seconds':wall,'status':status,'limits':'Bounded128px two-epoch worker smoke image, not convergence, performance benchmark, calibration or visual parity proof.'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    if result.returncode:raise RuntimeError((out/'worker.log').read_text()[-5000:])
    assert status['state']=='complete'
    assert 'rod_bundle_diagnostic' in status
    path=out/'image'/status['image'];pixels=np.asarray(Image.open(path));assert pixels.size and pixels.max()>0
    print(json.dumps({'wall_seconds':wall,'preparation_seconds':status['preparation_seconds'],'render_seconds':status['elapsed_seconds'],'image':str(path),'diagnostic':status['rod_bundle_diagnostic'],'nonzero_components':int(np.count_nonzero(pixels))},indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--turing-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.turing_root,a.out)
