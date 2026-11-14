"""Door-frame mesh using the existing DutyStation native mesh path."""
import math
import numpy as np
from duty_station import DutyStation, _box_verts

class PortalFrameStation(DutyStation):
    def __init__(self, placed):
        w,h,t=placed.width,placed.height,placed.frame_thickness
        if not all(math.isfinite(v) and v>0 for v in (w,h,t)):
            raise ValueError('portal dimensions must be finite and positive')
        super().__init__({'position':placed.pos.tolist(),'yaw_deg':placed.yaw_deg,
                          'material_bindings':placed.material_bindings,'side_wings':{'enabled':False}})
        self._body_data=np.asarray(
            _box_verts(-w/2-t,-t/2,-h/2,w/2+t,t/2,-h/2+t)+
            _box_verts(-w/2-t,-t/2,-h/2,-w/2,t/2,h/2+t)+
            _box_verts(w/2,-t/2,-h/2,w/2+t,t/2,h/2+t)+
            _box_verts(-w/2-t,-t/2,h/2,w/2+t,t/2,h/2+t),dtype=np.float32).reshape(-1,6)
        for name in ('_screen_data','_wing_data','_wall_data','_marker_grey_data','_marker_screen_data','_planned_tile_data'):
            setattr(self,name,np.zeros((0,6),np.float32))
        self._placed_ref=placed
        self.menu=None
        self.build_gl()
