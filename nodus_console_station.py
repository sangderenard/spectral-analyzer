"""Native Pluck console for Nodus subgraph UI and explicitly supplied live feeds.

The graph owner pushes snapshots on its stepping thread. This menu never
steps or reads a native graph from the rendering thread, nor creates authority.
"""
from __future__ import annotations
from copy import deepcopy
import json
from threading import Lock
from controls import Panel, KnobSpec, readonly_knob, get_action_registry, enqueue_action, clicked_knob_value
from uuid import uuid4
from dataclasses import fields


def panel_from_record(record):
    """Use Pluck's actual Panel/KnobSpec, including authored nested panels."""
    knobs = []
    allowed = {field.name for field in fields(KnobSpec)}
    for item in record.get('knobs', ()):
        knob = KnobSpec(**{key:value for key,value in item.items() if key in allowed})
        for key in ('control_widget', 'widget_metadata'):
            if key in item: setattr(knob, key, deepcopy(item[key]))
        knobs.append(knob)
    return Panel(str(record['name']), str(record.get('label', '')),
        knobs=knobs,
        panels=[panel_from_record(panel) for panel in record.get('panels', ())],
        payload=deepcopy(record.get('payload', {})))


class NodusConsoleMenu:
    STREAMS = ('metadata', 'api', 'abi', 'profile', 'stats')

    def __init__(self, state):
        self.module_ids = tuple(str(key) for key in state['module_ids'])
        self._lock = Lock()
        self._feeds = {}
        self._values = {}
        self._enabled = set()
        self._page = 0
        self._navigation = None
        self._door_actions = {}
        self._dispatch = None
        self._action_key = 'nodus-console/' + uuid4().hex
        self._subscribe = None
        self._hud_visible = False
        self._doc_ids = {}
        self._doc_action_rects = {}
        self._doc_knob_rects = {}
        self.status = 'Read-only: host controls are not attached'
        self.controls = panel_from_record(state['panel']) if state.get('panel') else Panel(
            'rows', 'Nodus rows', knobs=[readonly_knob('graph', 'Graph', default=json.dumps(state.get('graph', {})))])
        self._rebuild()

    def bind_host(self, *, dispatch=None, subscribe=None):
        """Host callbacks must enforce its own authentication and row contract.

        dispatch(module_ids, action, values); subscribe(module_ids, stream, on).
        Neither callback is deserialized from scene data.
        """
        self._dispatch, self._subscribe = dispatch, subscribe
        if dispatch is not None:
            get_action_registry().register_callable(self._action_key,
                lambda **context: dispatch(context['module_ids'], context['route'], context['values']))
        self.status = 'Host controls attached' if dispatch else 'Read-only'
        self._rebuild()

    def update_stream(self, stream, data):
        if stream not in self.STREAMS:
            raise ValueError(stream)
        with self._lock:
            self._feeds[stream] = deepcopy(data)

    def update_values(self, values):
        with self._lock:
            self._values.update(deepcopy(values))

    def bind_doors(self, runtime):
        self._navigation = runtime
        self._rebuild()

    def _rebuild(self):
        pages = [self.controls, *self.controls.panels]
        self._door_actions = {}
        if self._navigation is not None:
            actions, seen = [], set()
            for opening in self._navigation.openings:
                identity = opening.get('dial') or opening.get('coincident_with', opening['identity'])
                if identity in seen: continue
                seen.add(identity)
                kind = 'dial' if opening.get('dial') else 'treadmill' if opening.get('treadmill') else None
                if kind is None: continue
                for direction in (-1,1):
                    key = 'door_' + str(len(self._door_actions))
                    self._door_actions[key] = (kind,opening,direction)
                    actions.append({'key':key,'label':('PREVIOUS ' if direction<0 else 'NEXT ')+identity})
            if actions: pages.append(Panel('doorways','DOOR DESTINATIONS',payload={'actions':actions}))
        for key in self.STREAMS:
            if key in self._enabled:
                pages.append(Panel('feed_' + key, key.upper(),
                    knobs=[readonly_knob('feed_' + key, 'Latest', default='No sample received')]))
        self._page %= len(pages)
        self.panel_spec = Panel('nodus', 'NODUS CONSOLE',
            knobs=[readonly_knob('status', 'Status', default=self.status)],
            panels=[pages[self._page]], payload={'action_first': True, 'actions': [
                {'key':'page_previous','label':'PREVIOUS PANEL'},
                {'key':'page_next','label':'NEXT PANEL'}, *[
                {'key': 'stream_' + key, 'label': ('HIDE ' if key in self._enabled else 'SHOW ') + key.upper()}
                for key in self.STREAMS]]})

    @property
    def knob_values(self):
        def defaults(panel):
            result = {knob.name: knob.default for knob in panel.knobs}
            for child in panel.panels: result.update(defaults(child))
            return result
        with self._lock:
            values = defaults(self.controls)
            values.update(self._values)
            values['status'] = self.status
            for key in self._enabled:
                values['feed_' + key] = json.dumps(self._feeds[key], sort_keys=True) if key in self._feeds else 'No sample received'
            return values

    def activate(self, route):
        action = route.rsplit('.', 1)[-1]
        if action in self._door_actions:
            kind, opening, direction = self._door_actions[action]
            if kind == 'dial':
                key=opening['dial']
                position=self._navigation.dial_positions.get(key,0)
                self._navigation.select_dial(key,(position+direction)%opening['dial_positions'])
            else:
                key=opening.get('coincident_with',opening['identity']); t=opening['treadmill']
                offset=self._navigation.treadmill_offsets.get(key,t['offset'])
                self._navigation.scroll_treadmill(opening['identity'],max(0,min(t['length']-1,offset+direction*t['window'])))
            return True
        if action in ('page_previous', 'page_next'):
            self._page += 1 if action == 'page_next' else -1
            self._rebuild()
            return True
        if action.startswith('stream_') and action[7:] in self.STREAMS:
            key = action[7:]
            on = key not in self._enabled
            if self._subscribe is not None:
                self._subscribe(self.module_ids, key, on)
            if on: self._enabled.add(key)
            else: self._enabled.discard(key)
            self._rebuild()
            return True
        if self._dispatch is None:
            self.status = 'No authorized host binding for this action'
            return False
        enqueue_action(self._action_key, module_ids=self.module_ids, route=route, values=self.knob_values)
        return True

    def show_hud(self, visible):
        self._hud_visible = bool(visible)

    def submit_doc_channel(self, doc_rdr, win_w, win_h):
        if not self._hud_visible: return
        width, height = min(700, max(320, win_w - 40)), min(650, max(260, win_h - 40))
        self._doc_action_rects.clear()
        self._doc_knob_rects.clear()
        doc_rdr.submit_panel(self.panel_spec,
            ((win_w-width)//2, (win_h-height)//2, width, height),
            node_id_map=self._doc_ids, knob_values=self.knob_values,
            action_rects=self._doc_action_rects, knob_rects=self._doc_knob_rects)

    def handle_event(self, event):
        if not self._hud_visible: return False
        import pygame
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1: return False
        px, py = event.pos
        for route, (x,y,w,h) in self._doc_action_rects.items():
            if x <= px < x+w and y <= py < y+h:
                return self.activate(route)
        for name, info in self._doc_knob_rects.items():
            x,y,w,h = info['rect']
            if x <= px < x+w and y <= py < y+h:
                if self._dispatch is None or info.get('widget') == 'readonly':
                    return False
                current = self.knob_values.get(name, info.get('default'))
                value = clicked_knob_value(info, current, px-x, w)
                if value != current:
                    self.update_values({name: value})
                    return self.activate('setting.' + name)
        return False

def workspace_from_graph(graph, *, consoles=None, config_dir='configs/room_station', geometry=None):
    """Place real DutyStations in Pluck's workspace, never another renderer.

    consoles maps an authored console id to module_ids and optional Panel
    record. With no selection every native table gets one console.
    """
    import numpy as np
    from room_workspace import RoomWorkspace
    from placed_object import PlacedDutyStation
    modules = {str(m['stable_id']): m for m in graph['modules']}
    if graph.get('definition_error'):
        raise ValueError('Stable graph definition required')
    selections = consoles if consoles is not None else {
        key: {'module_ids': [key], 'label': m.get('label', key)} for key, m in modules.items()}
    if consoles:
        selections = dict(consoles)
        included = {str(key) for selection in consoles.values() for key in selection['module_ids']}
        for key,module in modules.items():
            if key not in included:
                selections['unselected/'+key] = {'module_ids':[key],'label':module.get('label',key)}
    workspace = RoomWorkspace.blank(config_dir)
    workspace.require_explicit_scene_path = True  # Derived views must not overwrite shared scene.yaml.
    if geometry is not None:
        workspace.nodus_world_geometry = geometry
        workspace.nodus_door_runtime = geometry.door_runtime()
        for portal in geometry.placed_portals(body_material='basic_paneling'):
            workspace.add_object(portal)
        boxes = [b for b in geometry.room_boxes if b['kind'] != 'global']
        for box in boxes:
            source = box['source']
            if source.get('geometry_mode') not in (None, 'solid'):
                continue  # Other original mesh forms require their own realizer.
            workspace.add_object(PlacedDutyStation(
                obj_id='geometry/'+box['identity'], label=box['identity'],
                pos=np.asarray([box['center'][0],box['center'][1],box['center'][2]-box['half_extent'][2]]),
                yaw_deg=box['yaw_deg'], station_type='living_map_room',
                material_bindings={'body':'basic_paneling'},
                build_state={'source':deepcopy(source),'scale':geometry.display_scale}))

        workspace.room_cfg['dimensions'] = {
            **workspace.room_dims(),
            'width_m':2*max((abs(b['center'][0])+b['half_extent'][0] for b in boxes), default=6),
            'depth_m':max((b['center'][1]+b['half_extent'][1] for b in boxes),default=10),
            'height_m':max((b['center'][2]+b['half_extent'][2] for b in boxes),default=4)}
    for index, (key, selection) in enumerate(selections.items()):
        ids = tuple(str(value) for value in selection['module_ids'])
        if not ids or any(value not in modules for value in ids):
            raise ValueError('Console contains missing module identities')
        state = deepcopy(selection)
        state['module_ids'] = list(ids)
        state['graph'] = {'revision': graph.get('revision'),
                          'modules': [deepcopy(modules[value]) for value in ids]}
        position = [(index % 4) * 3.0 - 4.5, (index // 4) * 3.0, 0.0]
        if geometry is not None:
            box = next(b for b in geometry.room_boxes if b['identity'] == geometry.model['identity']+'/console/'+str(key))
            position = [box['center'][0],box['center'][1],box['center'][2]-box['half_extent'][2]]
            if index == 0:
                workspace.station_cfg['position'] = [position[0],position[1]-3.0,position[2]]
                workspace.station_cfg['yaw_deg'] = 90.0
        workspace.add_object(PlacedDutyStation(
            obj_id='nodus/' + str(key), label=str(selection.get('label', key)),
            pos=np.asarray(position),
            station_type='nodus_console', config_dir=config_dir, build_state=state))
    return workspace
