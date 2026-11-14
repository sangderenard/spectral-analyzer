import unittest
from types import SimpleNamespace
from native_doorways import DoorRuntime, Aperture
from native_island_loader import NativeIsland, NativeIslandLoader


def island(name, destination, pair='entry'):
    records=[dict(identity=name+'/in',kind='portal',transition='door',port_role='in',pair=pair,island=destination),
             dict(identity=name+'/out',kind='portal',transition='door',port_role='out',pair=pair)]
    frame=Aperture((0,0,1),(0,-1,0),(1,0,0),(0,0,1),1)
    runtime=DoorRuntime(records,{o['identity']:frame for o in records})
    return NativeIsland(SimpleNamespace(model={'identity':name}),{'edited':name},runtime)

class IslandLoaderTests(unittest.TestCase):
    def test_prepare_commit_return_preserves_workspace(self):
        a,b=island('a','b'),island('b','a')
        loader=NativeIslandLoader(a)
        calls=[]
        loader.register('b',lambda event: (calls.append(event) or b))
        event=dict(source='a/in',island='b',pair='entry',href=None)
        prepared=loader.prepare(event,now_ms=10)
        self.assertIs(loader.current,a)
        self.assertFalse(b.runtime.cooldowns)
        self.assertEqual(prepared.arrival.velocity,(0,0,0))
        loader.commit(prepared)
        self.assertIs(loader.current,b)
        a.workspace['edited']='retained'
        back=loader.prepare(dict(source='b/in',island='a',pair='entry',href=None),now_ms=300)
        loader.commit(back)
        self.assertIs(loader.current,a)
        self.assertEqual(a.workspace['edited'],'retained')
        with self.assertRaises(ValueError): loader.commit(prepared)
        self.assertEqual(len(calls),1)

    def test_wrong_destination_unregistered_and_bad_pair_do_not_switch(self):
        a=island('a','b');loader=NativeIslandLoader(a)
        with self.assertRaises(ValueError):
            loader.prepare(dict(source='a/in',island='evil',pair='entry',href=None),now_ms=0)
        with self.assertRaises(KeyError):
            loader.prepare(dict(source='a/in',island='b',pair='entry',href=None),now_ms=0)
        loader.register('b',lambda event:island('b','a',pair='wrong'))
        with self.assertRaises(ValueError):
            loader.prepare(dict(source='a/in',island='b',pair='entry',href=None),now_ms=0)
        self.assertIs(loader.current,a)

    def test_window_declaration_is_passed_without_invented_url(self):
        a=island('a','b');a.runtime.openings[0]['treadmill']=dict(length=7,window=3,offset=0)
        a.runtime.scroll_treadmill('a/in',6)
        loader=NativeIslandLoader(a);seen=[]
        loader.register('b',lambda event:(seen.append(event) or island('b','a')))
        event=dict(source='a/in',island='b',pair='entry',href=None,treadmill=dict(length=7,window=3,offset=6))
        p=loader.prepare(event,now_ms=0)
        self.assertEqual(p.key,('b',6))
        self.assertEqual(seen[0],event)
        event['treadmill']['offset']=3
        with self.assertRaises(ValueError): loader.prepare(event,now_ms=0)



class ProducerIslandTests(unittest.TestCase):
    def test_original_turing_declarations_round_trip(self):
        from src.compiler.nodus_console_world import graph_to_world
        from src.compiler.abstract_ui_doorways import declare_island_door, declare_island_passage
        from nodus_world_geometry import from_world
        from nodus_console_station import workspace_from_graph
        graph={'modules':[{'stable_id':'a','rows':[]}],'edges':[]}
        outside=graph_to_world(graph,identity='outside')
        inside=graph_to_world(graph,identity='inside')
        outside=declare_island_door(outside,'outside/console/a','inside')
        inside=declare_island_passage(inside,inside.regions[0].identity,lot='outside/console/a',outside='outside')
        ga,gb=from_world(outside).scaled(4),from_world(inside).scaled(4)
        wa,wb=workspace_from_graph(graph,geometry=ga),workspace_from_graph(graph,geometry=gb)
        a,b=NativeIsland(ga,wa),NativeIsland(gb,wb)
        loader=NativeIslandLoader(a)
        loader.register('inside',lambda event:b)
        for geometry,destination,now in [(ga,'inside',1),(gb,'outside',500)]:
            source=next(o for o in geometry.openings if o.get('island')==destination)
            event={k:source.get(k) for k in ('island','pair','href')}
            event['source']=source['identity']
            prepared=loader.prepare(event,now_ms=now)
            loader.commit(prepared)
            self.assertEqual(loader.current.identity,destination)
        self.assertIs(loader.current.workspace,wa)

if __name__=='__main__': unittest.main()


class LocalManifestTests(unittest.TestCase):
    def test_relative_documents_and_identity_validation(self):
        import json, tempfile
        from pathlib import Path
        from native_island_loader import read_island_manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'b.json').write_text(json.dumps({'schema':'abstract-ui-div-map-v0','identity':'b','regions':[]}))
            manifest=root/'islands.json'
            manifest.write_text(json.dumps({'b':'b.json'}))
            self.assertEqual(read_island_manifest(manifest)['b']['identity'],'b')
            manifest.write_text(json.dumps({'wrong':'b.json'}))
            with self.assertRaises(ValueError): read_island_manifest(manifest)
            manifest.write_text(json.dumps({'b':'https://example.invalid/b.json'}))
            with self.assertRaises(ValueError): read_island_manifest(manifest)
