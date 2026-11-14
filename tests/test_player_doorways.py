import unittest
import numpy as np
from native_doorways import Aperture, DoorRuntime
from player_controller import PlayerController

class PlayerDoorTests(unittest.TestCase):
    def test_player_receives_z_up_crossing(self):
        records=[{'identity':'a','pair':'p','kind':'portal','port_role':'in','transition':'door'},
                 {'identity':'b','pair':'p','kind':'portal','port_role':'out','transition':'door'}]
        runtime=DoorRuntime(records,{'a':Aperture((0,0,1.65),(0,1,0),(1,0,0),bitangent=(0,0,1)),
                                    'b':Aperture((0,10,1.65),(0,1,0),(1,0,0),bitangent=(0,0,1))})
        p=PlayerController.__new__(PlayerController)
        p._walk_pos=np.array([0.,-.1,0.]); p._walk_yaw=-90.; p._walk_pitch=0.; p._walk_vel_z=0.; p._eye_height=1.65
        p.set_door_runtime(runtime)
        p._advance_doors(np.array([0.,1.,0.]),.1)
        self.assertGreater(p._walk_pos[1],10.)
        self.assertAlmostEqual(p._walk_pos[2],0.)
        self.assertAlmostEqual(p._walk_yaw,90.)

if __name__=='__main__': unittest.main()
