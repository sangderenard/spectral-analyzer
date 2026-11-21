"""Native realization of Living Map box wall/opening records, using Pluck meshes.

Matches abstract_ui_div_map.wallSegments: floor, boundary intervals, sills and
lintels. No opaque polygon covers the declared aperture. Traversal remains in
DoorRuntime; this module does not grant access or modify source geometry.
"""
import numpy as np
from duty_station import DutyStation, _box_verts


def room_prisms(host, scale=1.0):
    hx, hy = map(float, host['half_extent'])
    height = float(host['height'])
    floor = min(height, float(host.get('floor_height', .035)))
    thickness = max(.01, min(hx, hy, float(host.get('appearance', {}).get('wall_thickness', host.get('wall_thickness', .04)))))
    result = []
    def add(x0,y0,z0,x1,y1,z1):
        if x1>x0 and y1>y0 and z1>z0:
            result.append(tuple(v*scale for v in (x0,y0,z0,x1,y1,z1)))
    if host.get('placement', {}).get('custody') == 'inventory':
        return result
    if host.get('geometry_mode') == 'solid':
        add(-hx,-hy,0,hx,hy,height)
        return result
    add(-hx,-hy,0,hx,hy,floor)
    for side,half,fixed,axis in [('south',hx,-hy+thickness/2,'x'),('north',hx,hy-thickness/2,'x'),('west',hy,-hx+thickness/2,'y'),('east',hy,hx-thickness/2,'y')]:
        def wall(lo,hi,bottom,top):
            if axis=='x': add(lo,fixed-thickness/2,bottom,hi,fixed+thickness/2,top)
            else: add(fixed-thickness/2,lo,bottom,fixed+thickness/2,hi,top)
        intervals=[(-half,half)]
        for opening in host.get('openings', ()):
            if opening['side']!=side: continue
            lo=max(-half,float(opening['offset'])-float(opening['width'])/2)
            hi=min(half,float(opening['offset'])+float(opening['width'])/2)
            if hi<=lo: continue
            remaining=[]
            for a,b in intervals:
                if hi<=a or lo>=b: remaining.append((a,b))
                else:
                    if lo>a: remaining.append((a,lo))
                    if hi<b: remaining.append((hi,b))
            intervals=remaining
            bottom=max(floor,min(height,float(opening.get('bottom',floor))))
            top=min(height,bottom+float(opening['height']))
            wall(lo,hi,floor,bottom)
            wall(lo,hi,top,height)
        for lo,hi in intervals: wall(lo,hi,floor,height)
    return result


class LivingMapRoomStation(DutyStation):
    def __init__(self, placed):
        super().__init__({'position':placed.pos.tolist(),'yaw_deg':placed.yaw_deg,
            'material_bindings':placed.material_bindings,'side_wings':{'enabled':False}})
        self.room_prisms=room_prisms(placed.build_state['source'],placed.build_state['scale'])
        vertices=[v for prism in self.room_prisms for v in _box_verts(*prism)]
        self._body_data=np.asarray(vertices,dtype=np.float32).reshape(-1,6)
        for name in ('_screen_data','_wing_data','_wall_data','_marker_grey_data','_marker_screen_data','_planned_tile_data'):
            setattr(self,name,np.zeros((0,6),np.float32))
        self._placed_ref=placed
        self.menu=None
        self.build_gl()


def world_surface_triangles(geometry):
    """World-space triangles from the SAME prisms the renderer submits."""
    import math
    parts=[]
    for box in geometry.room_boxes:
        source=box['source']
        if box['kind'] in ('global','world-envelope') or source.get('geometry_mode') not in (None,'solid'):
            continue
        yaw=math.radians(box['yaw_deg']);c,s=math.cos(yaw),math.sin(yaw)
        origin=np.array(box['center']);origin[2]-=box['half_extent'][2]
        for prism in room_prisms(source,geometry.display_scale):
            vertices=np.asarray(_box_verts(*prism),dtype=np.float32).reshape(-1,6)[:,:3].copy()
            x,y=vertices[:,0].copy(),vertices[:,1].copy()
            vertices[:,0]=x*c-y*s;vertices[:,1]=x*s+y*c
            parts.append((vertices+origin).reshape(-1,3,3))
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros((0,3,3),np.float32)
