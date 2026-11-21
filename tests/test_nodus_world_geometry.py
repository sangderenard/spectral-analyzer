import unittest
from nodus_world_geometry import opening_aperture, from_document_geometry, pluck_xyz

class GeometryTests(unittest.TestCase):
    def test_exact_south_geometry_and_z_up(self):
        host = dict(center=[10,20], half_extent=[4,3], wall_thickness=.2,
                    placement={'elevation':2})
        opening = dict(side='south',offset=.5,height=2,width=1,bottom=.25)
        a=opening_aperture(host, opening)
        self.assertEqual(a.center,(10.5,17.1,3.25))
        self.assertEqual(a.normal,(0,-1,0))
        self.assertEqual(a.tangent,(1,0,0))
        self.assertEqual(a.bitangent,(0,0,1))
        self.assertEqual(a.radius,.5)

    def test_rotation_and_minimum_radius(self):
        host = dict(center=[0,0], half_extent=[4,3], wall_thickness=.2,
                    placement={'rotation':[0,90,0]})
        a=opening_aperture(host,dict(side='south',offset=0,height=.1,width=.1))
        self.assertAlmostEqual(a.center[0],2.9)
        self.assertAlmostEqual(a.center[1],0)
        self.assertAlmostEqual(a.normal[0],1)
        self.assertEqual(a.radius,.18)

    def test_document_preserves_declarations_and_box_axes(self):
        opening=dict(identity='entry',kind='portal',side='south',offset=0,height=2,width=1,
                     pair='p',transition='door',port_role='in',dial='d',dial_position=0,dial_positions=2)
        host=dict(identity='room',kind='building',center=[10,20],half_extent=[4,3],height=2,openings=[opening])
        g=from_document_geometry({'identity':'world'},{'boxes':[host]})
        self.assertEqual(g.room_boxes[0]['center'],(10,20,1))
        self.assertEqual(g.room_boxes[0]['half_extent'],(4,3,1))
        self.assertEqual(g.openings[0]['dial'],'d')
        self.assertEqual(g.opening_hosts['entry'],'room')
        self.assertEqual(g.door_runtime().openings,g.openings)
        self.assertEqual(pluck_xyz((1,2,3)),(1,3,2))

if __name__ == '__main__': unittest.main()
