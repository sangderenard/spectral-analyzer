import unittest
from living_map_room_station import room_prisms

class RoomMeshTests(unittest.TestCase):
    def test_aperture_is_empty_with_sill_and_lintel(self):
        host=dict(half_extent=[4,3],height=4,floor_height=.1,wall_thickness=.2,
                  openings=[dict(side='south',offset=1,width=2,height=2,bottom=.5)])
        prisms=room_prisms(host)
        def occupied(x,y,z):
            return any(a<=x<=d and b<=y<=e and c<=z<=f for a,b,c,d,e,f in prisms)
        self.assertFalse(occupied(1,-2.9,1.5))
        self.assertTrue(occupied(1,-2.9,.3))
        self.assertTrue(occupied(1,-2.9,3))
        self.assertTrue(occupied(-2,-2.9,1.5))
        self.assertEqual(room_prisms(host,4),[tuple(x*4 for x in p) for p in prisms])

    def test_coincident_portal_does_not_seal_aperture(self):
        opening=dict(side='east',offset=0,width=1,height=2,bottom=0)
        prisms=room_prisms(dict(half_extent=[2,2],height=2,openings=[opening,dict(opening)]))
        self.assertFalse(any(a<1.98<d and b<0<e and c<1<f for a,b,c,d,e,f in prisms))

    def test_derived_world_cannot_overwrite_shared_scene(self):
        from nodus_console_station import workspace_from_graph
        from unittest.mock import patch
        workspace=workspace_from_graph({'modules':[],'edges':[]})
        with patch('room_workspace._dump_yaml') as write:
            self.assertFalse(workspace.save_scene())
            write.assert_not_called()
            workspace.save_scene('explicit-user-scene.yaml')
            write.assert_called_once()
