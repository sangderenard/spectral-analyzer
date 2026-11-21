import unittest
import numpy as np
from player_clip_py import PlayerClipEngine
from living_map_room_station import world_surface_triangles
from nodus_world_geometry import from_document_geometry

class WorldClipTests(unittest.TestCase):
    def test_actual_triangle_sweep_stops_wall_but_not_opening(self):
        host=dict(identity='r',kind='building',center=[0,0],half_extent=[3,3],height=3,
            openings=[dict(side='south',offset=0,width=2,height=2,bottom=0)])
        geometry=from_document_geometry({'identity':'world'},{'boxes':[host]})
        engine=PlayerClipEngine();engine.set_slabs(np.zeros((0,8),np.float32))
        engine.set_surface_field(world_surface_triangles(geometry));engine.enable_surface_clip(True)
        for x,blocked in [(2,True),(0,False)]:
            before=np.array([x,-4,1],np.float32);after=np.array([x,-2,1],np.float32)
            out,flags=engine.tick(after,(after-before),-10,10,.15,0,1,previous_pos=before)
            if blocked:
                self.assertLess(out[1],-3)
                self.assertTrue(flags)
            else:
                np.testing.assert_allclose(out,after)

    def test_rotated_mesh_uses_same_world_coordinates(self):
        host=dict(identity='r',kind='building',center=[10,20],half_extent=[3,2],height=3,
                  placement={'rotation':[0,90,0],'elevation':5},openings=[])
        tris=world_surface_triangles(from_document_geometry({'identity':'w'},{'boxes':[host]}))
        np.testing.assert_allclose(tris.min(axis=(0,1)),[8,17,5],atol=1e-5)
        np.testing.assert_allclose(tris.max(axis=(0,1)),[12,23,8],atol=1e-5)
