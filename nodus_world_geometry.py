"""Geometry-only bridge from existing Living Map layout to native Pluck.

No runtime compilation or HTML projection. The adapter calls the same
world_div_map_graph/document_geometry producers and mirrors their browser's
portalOpeningTransform. Living Map Y-up -> Pluck Z-up swaps Y and Z (a
reflection); mesh consumers must reverse winding if converting triangles.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from copy import deepcopy
import math
from typing import Any, Mapping
from native_doorways import Aperture, DoorRuntime


def pluck_xyz(value):
    return (float(value[0]), float(value[2]), float(value[1]))


def opening_aperture(host: Mapping[str, Any], opening: Mapping[str, Any]) -> Aperture:
    """Exact portalOpeningTransform + radial chart, before inward-facing flip."""
    thickness = max(.01, float(host.get('appearance', {}).get('wall_thickness')
                              or host.get('wall_thickness') or .04))
    hx, hz = host['half_extent']
    offset = float(opening['offset'])
    side = opening['side']
    if side == 'north':
        local, normal = (offset, hz-thickness*.5), (0, 1)
    elif side == 'south':
        local, normal = (offset, -hz+thickness*.5), (0, -1)
    elif side == 'east':
        local, normal = (hx-thickness*.5, offset), (1, 0)
    elif side == 'west':
        local, normal = (-hx+thickness*.5, offset), (-1, 0)
    else:
        raise ValueError(f'unknown opening side {side}')
    placement = host.get('placement', {})
    rotation = placement.get('rotation')
    yaw = float(rotation[1] if rotation is not None else placement.get('yaw_degrees', 0))
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    def rotate(v):
        return (v[0]*c-v[1]*s, v[0]*s+v[1]*c)
    local, normal = rotate(local), rotate(normal)
    bottom = float(placement.get('elevation') or 0)+float(opening.get('bottom') or 0)
    return Aperture(
        pluck_xyz((host['center'][0]+local[0], bottom+float(opening['height'])*.5, host['center'][1]+local[1])),
        pluck_xyz((normal[0], 0, normal[1])),
        pluck_xyz((-normal[1], 0, normal[0])),
        pluck_xyz((0, 1, 0)), max(.18, min(float(opening['width']), float(opening['height']))*.5))


@dataclass(frozen=True)
class NativeWorldGeometry:
    model: Mapping[str, Any]
    document: Mapping[str, Any]
    openings: tuple[Mapping[str, Any], ...]
    apertures: Mapping[str, Aperture]
    room_boxes: tuple[Mapping[str, Any], ...]
    opening_hosts: Mapping[str, str]
    display_scale: float = 1.0

    def scaled(self, factor: float):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError('world display scale must be finite and positive')
        return replace(self, display_scale=self.display_scale*factor,
            apertures={key:replace(a, center=tuple(v*factor for v in a.center),radius=a.radius*factor)
                       for key,a in self.apertures.items()},
            room_boxes=tuple({**box,'center':tuple(v*factor for v in box['center']),
                              'half_extent':tuple(v*factor for v in box['half_extent'])} for box in self.room_boxes))

    def door_runtime(self) -> DoorRuntime:
        return DoorRuntime(self.openings, self.apertures)

    def placed_portals(self, *, body_material: str):
        """Dimensioned Pluck frame records; traversal belongs to door_runtime.

        Host must explicitly supply a canonical material, never invent one here.
        """
        import numpy as np
        from placed_object import PlacedPortalFrame
        if not body_material:
            raise ValueError('canonical portal body material required')
        result = []
        seen = set()
        for opening in self.openings:
            key = (self.opening_hosts[opening['identity']], opening.get('coincident_with', opening['identity']))
            if key in seen:
                continue
            seen.add(key)
            a = self.apertures[opening['identity']]
            targets = [o for o in self.openings if o.get('pair') == opening.get('pair') and o.get('port_role') == 'out']
            destination = self.opening_hosts[targets[0]['identity']] if targets else str(opening.get('island', ''))
            result.append(PlacedPortalFrame(obj_id=opening['identity'], label='Door', pos=np.array(a.center),
                yaw_deg=math.degrees(math.atan2(a.tangent[1], a.tangent[0])),
                width=float(opening['width'])*self.display_scale,
                height=float(opening['height'])*self.display_scale,
                material_bindings={'body': body_material}, target_room_id=destination))
        return tuple(result)


def from_document_geometry(model: Mapping[str, Any], document: Mapping[str, Any]) -> NativeWorldGeometry:
    """Adapt already-realized boxes, including actual user-edited placements."""
    openings, frames, rooms, hosts = [], {}, [], {}
    for host in document['boxes']:
        placement = host.get('placement', {})
        elevation = float(placement.get('elevation') or 0)
        rotation = placement.get('rotation')
        yaw = float(rotation[1] if rotation is not None else placement.get('yaw_degrees', 0))
        rooms.append({'identity': host['identity'], 'kind': host['kind'],
                      'center': pluck_xyz((host['center'][0], elevation+host['height']*.5, host['center'][1])),
                      'half_extent': (host['half_extent'][0], host['half_extent'][1], host['height']*.5),
                      'yaw_deg': yaw, 'source': deepcopy(host)})
        for opening in host.get('openings', ()):
            if opening.get('kind') != 'portal':
                continue
            record = deepcopy(opening)
            openings.append(record)
            frames[record['identity']] = opening_aperture(host, record)
            hosts[record['identity']] = host['identity']
    return NativeWorldGeometry(deepcopy(model), deepcopy(document), tuple(openings), frames, tuple(rooms), hosts)


def from_world(world) -> NativeWorldGeometry:
    """Turing must be on configured PYTHONPATH; no path mutation or compiler."""
    from src.compiler.abstract_ui_div_map import world_div_map_graph
    from src.compiler.abstract_ui_viewports import document_geometry
    model = world_div_map_graph(world)
    return from_document_geometry(model, document_geometry(model))


def from_nodus_graph(graph: Mapping[str, Any], **kwargs) -> NativeWorldGeometry:
    from src.compiler.nodus_console_world import graph_to_world
    return from_world(graph_to_world(graph, **kwargs))
