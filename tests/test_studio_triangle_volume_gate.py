"""CPU geometry gate; GPU script remains explicit and opt-in."""
import ast
from pathlib import Path
import numpy as np


def test_triangle_gate_cube_is_closed_outward_and_nested():
    # Load only the pure geometry function, without renderer/GL dependencies.
    path=Path(__file__).with_name('studio_triangle_volume_gpu_check.py')
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cube_triangles')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    host=env['cube_triangles'](.025);dye=env['cube_triangles'](.008)
    assert host.shape==dye.shape==(12,3,3)
    for tris in (host,dye):
        normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
        assert np.all(np.sum(normals*tris.mean(axis=1),axis=1)>0)
        edges={}
        for t in tris:
            for a,b in zip(t,np.roll(t,-1,axis=0)):
                key=(tuple(a),tuple(b));edges[key]=edges.get(key,0)+1
        assert all(count==1 and edges.get((b,a))==1 for (a,b),count in edges.items())
    assert np.max(np.abs(dye))<np.max(np.abs(host))
