"""Project explicit room/stage volume contracts onto the shared native buffers."""
import copy
import numpy as np

# Release switch: worker processes import source anew; enable only after review.
ADVANCED_CONDUCTOR_RUNTIME_ENABLED = False


def project_scene_engines(fixtures):
    ids=np.asarray(fixtures['material_ids'],np.int32).copy()
    pbr=np.asarray(fixtures['pbr'],np.float32).reshape(-1,16).copy()
    contract=fixtures.get('material_engines')
    if contract is None:return ids,pbr,None,None
    # Admission validates topology and coverage. Revalidate optical data here
    # before any buffer upload; one private native row per closed component
    # lets the region resolver detect simultaneous independent ownership.
    from .studio_optical_media import validate_optical_media
    volumes=contract['volumes'];required={r['material'] for r in volumes}
    check=pbr.copy()
    for m in required:check[m,5]=1;check[m,7]=0
    optics=validate_optical_media(contract['optical_media'],check,required) if required else {}
    rows=list(pbr);pairs=[];records=[]
    for volume in volumes:
        a,n,m=(volume[k] for k in ('start','count','material'))
        target=len(rows);row=pbr[m].copy();row[5]=1;row[7]=0;rows.append(row)
        ids[a:a+n]=target
        pairs.append(dict(start=a,count=n,positive=-1,negative=target))
        record=copy.deepcopy(optics[m]);record['material']=target;records.append(record)
    return ids,np.asarray(rows,np.float32),({'version':1,'ranges':pairs} if pairs else None),({'version':1,'materials':records} if records else None)


def advanced_surface_triangles(fixtures):
    """Select actual engine ranges, not material-global flags."""
    mask=np.zeros(len(fixtures['material_ids']),bool)
    contract=fixtures.get('material_engines')
    if contract and ADVANCED_CONDUCTOR_RUNTIME_ENABLED:
        for group in contract['groups']:
            if group['engine']=='dielectric':
                for r in group['ranges']:mask[r['start']:r['start']+r['count']]=True
    return mask
