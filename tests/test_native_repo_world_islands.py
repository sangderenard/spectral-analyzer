import unittest
from native_repo_world_islands import RepoWorldIslands

class RepoWindowTests(unittest.TestCase):
    def test_actual_index_tail_window_and_parent_return(self):
        from src.compiler.repo_world_scan import scan_files,region_identity
        from src.compiler.repo_world_units import UnitIndex,window_address
        from src.compiler.repo_world_unit_world import island_of
        h='a'*40
        files={'a/x.py':b'', 'c/x.py':b'', **{f'b/{i}.py':b'' for i in range(5)}}
        index=UnitIndex(scan_files(h,None,sorted(files),files.__getitem__),capacity=2)
        root,b=region_identity(h,'.'),region_identity(h,'b')
        host=RepoWorldIslands(index,lambda geometry:{'edits':[]})
        address=window_address(root,1)
        loader=host.loader(address)
        initial=loader.current
        initial.workspace['edits'].append('keep')
        source=next(o for o in initial.runtime.openings if o.get('island')==island_of(b))
        t=initial.runtime.scroll_treadmill(source['identity'],4)
        event={k:source.get(k) for k in ('island','pair','href')}
        event.update(source=source['identity'],treadmill=t)
        p=loader.prepare(event,now_ms=0)
        loader.commit(p)
        # Five files, offset 4, window 2: only the last file is materialized.
        self.assertEqual(len(loader.current.geometry.model['regions'][0]['buildings']),1)
        self.assertEqual(loader.current.geometry.model['regions'][0]['identity'],window_address(b,4))
        back=next(o for o in loader.current.runtime.openings if o.get('island')==island_of(root))
        event={k:back.get(k) for k in ('island','pair','href')};event['source']=back['identity']
        self.assertEqual(event['href'],address)
        loader.commit(loader.prepare(event,now_ms=300))
        self.assertIs(loader.current,initial)
        self.assertEqual(loader.current.workspace['edits'],['keep'])

    def test_wrong_declared_window_is_rejected(self):
        from src.compiler.repo_world_scan import scan_files,region_identity
        from src.compiler.repo_world_units import UnitIndex
        from src.compiler.repo_world_unit_world import island_of
        h='b'*40;files={f'a/{i}.py':b'' for i in range(3)}
        index=UnitIndex(scan_files(h,None,sorted(files),files.__getitem__),capacity=2)
        host=RepoWorldIslands(index,lambda geometry:object())
        with self.assertRaises(ValueError):
            host.address(dict(island=island_of(region_identity(h,'a')),treadmill=dict(length=30,window=2,offset=0)))
