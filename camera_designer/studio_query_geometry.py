"""Canonical byte identity for client intersection geometry, not transport."""
import hashlib,struct
import numpy as np

SCHEMA='studio-query-geometry-v1'

def query_geometry_identity(triangles,nodes,ids,epsilon=1e-7):
    ordered=sorted(triangles,key=lambda t:t['owner'])
    owners=np.asarray([t['owner'] for t in ordered],dtype='<u4')
    coords=np.asarray([t['v0']+t['e1']+t['e2'] for t in ordered],dtype='<f4').reshape(-1,9)
    node_array=np.asarray(nodes,dtype='<f4').reshape(-1,12)
    index_array=np.asarray(ids,dtype='<u4').reshape(-1)
    blobs={'triangles':owners.tobytes()+coords.tobytes(),'bvh_nodes':node_array.tobytes(),'bvh_ids':index_array.tobytes()}
    counts={'triangles':len(owners),'bvh_nodes':len(node_array),'bvh_ids':len(index_array)}
    parts={k:{'sha256':hashlib.sha256(v).hexdigest(),'count':counts[k],'bytes':len(v)} for k,v in blobs.items()}
    prefix=SCHEMA.encode('ascii')+b'\0'+struct.pack('<IIIf',len(owners),len(node_array),len(index_array),epsilon)
    digest=hashlib.sha256(prefix+b''.join(bytes.fromhex(parts[k]['sha256']) for k in ('triangles','bvh_nodes','bvh_ids'))).hexdigest()
    return {'query_geometry_schema':SCHEMA,'query_geometry_hash':digest,'query_geometry_parts':parts}
