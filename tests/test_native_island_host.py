"""Exercise the actual demo main-loop island handoff without a GL context."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import math
import numpy as np

class HostAttachmentTests(unittest.TestCase):
    def block(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'demo_pluck_gl.py').read_text(encoding='utf-8'))
        blocks=[node for node in ast.walk(tree) if isinstance(node,ast.If)
                and ast.unparse(node.test)=='_pending_island_events and _island_loader is not None']
        self.assertEqual(len(blocks),1)
        return compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),'demo island host','exec')

    def context(self, fail=False):
        old_station,global_station,new_station=NS(menu=None),NS(menu=None),NS(menu=None)
        old_ws=NS(room_cfg={'old':True})
        new_ws=NS(room_cfg={'new':True},build_scene_objects=lambda w,h:[new_station])
        destination=NS(workspace=new_ws,runtime=object(),host_state={},identity='new',geometry=object())
        arrival=NS(position=(1,2,3),facing=(0,1,0))
        prepared=NS(destination=destination,arrival=arrival)
        calls=[]
        def prepare(*args,**kwargs):
            if fail: raise ValueError('missing arrival')
            return prepared
        loader=NS(prepare=prepare,commit=lambda p:calls.append(p))
        player=NS(set_world_geometry=lambda geometry:calls.append(geometry),set_room_cfg=lambda cfg:calls.append(cfg),set_door_runtime=lambda *a,**k:calls.append(a),_eye_height=1.5)
        class Room:
            def __init__(self,*args):pass
            def init_gl(self):pass
        context=dict(_pending_island_events=[{}],_island_loader=loader,time=NS(perf_counter=lambda:1),
            _RoomStation=Room,WIN_W=800,WIN_H=600,_HAS_CAMERA_ITEM=False,_nodus_feeds=None,
            player_ctrl=player,_room_ws=old_ws,_room_station=object(),_scene_stations=[old_station],
            duty_stations=[global_station,old_station],cameras=['old'],np=np,math=math)
        return context,old_ws,global_station,new_station,calls

    def test_main_block_attaches_and_preserves_global_station(self):
        env,old,global_station,new,calls=self.context()
        exec(self.block(),env)
        self.assertEqual(env['duty_stations'],[global_station,new])
        self.assertEqual(env['_room_ws'].room_cfg,{'new':True})
        self.assertEqual(env['cameras'],[])
        np.testing.assert_allclose(env['player_ctrl']._walk_pos,(1,2,1.5))
        self.assertEqual(env['player_ctrl']._walk_yaw,90)
        self.assertTrue(calls)

    def test_failed_preparation_keeps_original_world(self):
        env,old,global_station,new,calls=self.context(fail=True)
        stations=list(env['duty_stations'])
        exec(self.block(),env)
        self.assertIs(env['_room_ws'],old)
        self.assertEqual(env['duty_stations'],stations)
        self.assertFalse(calls)
