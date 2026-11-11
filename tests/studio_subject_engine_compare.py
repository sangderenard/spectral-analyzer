"""Controlled saved legacy/glass-rod comparison. CPU preparation only by default."""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    job=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((job/'config.json').read_text())
    obj=json.loads((job.parent.parent/'marble_objects'/f"{config['object_digest']}.json").read_text())
    receipt=json.loads((job.parent.parent.parent/'live-receipt.json').read_text())['field']
    sec=receipt['sources'];data=(job/receipt['name']).read_bytes()
    sources=np.frombuffer(data[sec['offset_bytes']:sec['offset_bytes']+sec['byte_length']],'<f4').reshape(-1,12).copy()
    saved=np.load(job/'payload.npz');vertices=saved['vertices'];ids=saved['material_ids'];pbr=saved['pbr']
    import camera_designer.studio_native_emitters as emitters
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    from camera_designer.studio_region_gpu import region_records_from_geometry
    from camera_designer.studio_rod_bundle_gpu import pack_bundle_records
    from material_db import MaterialDatabase
    original=emitters.native_emissive_sources
    # The experiment holds the exact already-published native population fixed.
    emitters.native_emissive_sources=lambda *a,**k:sources.copy()
    results={};scenes={}
    try:
        for mode in ['glass_rod','legacy']:
            c=copy.deepcopy(config);rows=pbr.copy()
            if mode=='legacy':
                authored=obj['design']['optical_authoring']
                for material,row in zip(authored['materials'],authored['legacy_pbr']):rows[material['material']]=row
                for key in ['region_media','optical_media','boundary_media','boundary_transport']:c.pop(key,None)
            scene,spheres,stats,_=prepare_photo_geometry(c,vertices,ids,rows)
            g,s,mat,tri,ctx,_,bounds,src=scene
            regions=region_records_from_geometry(g)
            if c.get('region_media',{}).get('axial_bundle'):
                regions=pack_bundle_records(regions,c['region_media']['axial_bundle'],scale_m=c['scale_m'],center_m=c['object_center_m'],host_triangle=int(np.flatnonzero(g[:,3]==-1)[0]))
            native_pbr=MaterialDatabase.instance().build_tensors()['pbr'].copy()
            np.savez(out/f'{mode}.npz',geom=g,shade=s,mat=mat,triangles=tri,contexts=ctx,bmin=bounds[0],bmax=bounds[1],sources=src,spheres=spheres,regions=np.zeros((0,4),np.float32) if regions is None else regions,native_pbr=native_pbr)
            scenes[mode]=(g,s,tri,spheres,src)
            results[mode]=dict(triangles=len(tri),sources=len(src),sphere_count=len(spheres),region_float_count=0 if regions is None else int(regions.size),source_sha256=hashlib.sha256(src.tobytes()).hexdigest(),subject_pbr={str(i):rows[i].tolist() for i in [14,41,42]})
        a,b=scenes['glass_rod'],scenes['legacy']
        results['comparison']=dict(triangle_positions_identical=bool(np.array_equal(a[2],b[2])),sphere_geometry_identical=bool(np.array_equal(a[3][:,:4],b[3][:,:4])),source_bytes_identical=a[4].tobytes()==b[4].tobytes(),geometry_channels_changed=np.flatnonzero(np.any(a[0]!=b[0],axis=0)).tolist(),shade_channels_changed=np.flatnonzero(np.any(a[1]!=b[1],axis=0)).tolist())
        (out/'report.json').write_text(json.dumps(results,indent=2));print(json.dumps(results['comparison']))
    finally:emitters.native_emissive_sources=original
if __name__=='__main__':main()
