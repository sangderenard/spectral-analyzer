"""Runtime-faithful event test: synthesize pygame.event into handle_event."""
import os, pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
pygame.display.set_mode((1600, 900))

import demo_pluck_gl as d
from controls import Panel as P


class Cam:
    def __init__(self):
        self.focal_mm = 35.0
        self.focus_m = 1.6
        self.aperture = 0.0
        self.ca = 0.0
        self.tilt_shift = [0.0, 0.0]
        self.software = []
        self.eye = [0, 0, 0]
        self.target = [0, 1, 0]
        self._forced_eye = None


class Layer:
    iso = 1.4


class Film:
    active_layer = Layer()


class Acc:
    half_life = 5.0
    _rows_per_frame = 16
    _samples_per_pixel = 1
    _air_ds = 0.35
    _air_ss = 0.65
    _air_an = 12.0


class R:
    film = Film()
    _sensor_acc = Acc()
    _sensor_fps = 0.0
    _layers = [0] * 9
    def set_ray_tonemap(self, exposure, gamma): pass


class SP:
    values = {
        'sensor_iso': 1.4, 'ray_exposure': 1.0, 'ray_gamma': 1.0,
        'sensor_rate': 16, 'sensor_spp': 1, 'sensor_fps': 0,
        'frame_step': 1, 'mic_gain': 1.0, 'pickup_gain': 1.0,
        'segs': 60, 'plate_th': 128, 'dx': 0.01,
        'air_diff': 0.35, 'air_spec': 0.65, 'air_aniso': 12.0,
        'ray_density': 1.0, 'lens_aperture': 0.0, 'lens_ca': 0.0,
        'film_decay': 5.0, 'sensor_gain': 8.0,
    }


panel = d._PlayerCameraPanel()
cam = Cam(); sp = SP(); rr = R()
panel.attach(cam, sp, rr, player_ctrl=None)
panel._set_hud_mode(panel.HUD_FULL)
print('open?', panel.open)


class MockDoc:
    """Mimics doc_renderer.submit_panel hit-rect population for non-image panels."""
    def submit_panel(self, panel_spec, rect, node_id_map=None, knob_values=None,
                     parent_id=0, sibling_order=-1, action_rects=None, knob_rects=None):
        x, y, w, h = rect
        payload = getattr(panel_spec, 'payload', {}) or {}
        HDR_H = 20; KNOB_H = 40; PAD = 2
        cy = y + HDR_H + 2
        actions = payload.get('actions', []) if isinstance(payload, dict) else []
        if payload.get('action_first') and actions:
            for a in actions:
                if cy + 24 > y + h - PAD:
                    break
                if action_rects is not None:
                    key = a['key'] if isinstance(a, dict) else str(a)
                    action_rects['{}.{}'.format(panel_spec.name, key)] = (x + PAD, cy, w - PAD * 2, 24)
                cy += 24 + PAD
        for knob in (getattr(panel_spec, 'knobs', []) or []):
            if cy + KNOB_H > y + h - PAD:
                break
            if knob_rects is not None:
                kn = getattr(knob, 'name', '')
                knob_rects[kn] = {
                    'rect': (x + PAD, cy, w - PAD * 2, KNOB_H),
                    'widget': str(getattr(knob, 'control_widget', '') or ''),
                    'choices': list(getattr(knob, 'choices', []) or []),
                    'default': getattr(knob, 'default', None),
                    'low': float(getattr(knob, 'low', 0.0)),
                    'high': float(getattr(knob, 'high', 1.0)),
                    'step': float(getattr(knob, 'step', 0.0)),
                    'dtype': str(getattr(knob, 'dtype', 'float')),
                }
            cy += KNOB_H + PAD


WIN_W, WIN_H = 1600, 900
right_rect = (WIN_W - 340, 10, 330, min(WIN_H - 20, 720))
selector_h = 132
camera_rect = (right_rect[0], right_rect[1] + selector_h, right_rect[2],
               max(90, right_rect[3] - selector_h))
visible_rows = max(1, (camera_rect[3] - 30) // 42)

panel.set_doc_viewport(right_rect, visible_rows)
panel.clear_doc_hit_maps()

doc = MockDoc()
ren_actions = [{'key': m.value, 'label': m.value} for m in d.RenderMode]
ren_spec = P('camera_renderer_menu', 'Renderer',
             payload={'action_first': True, 'actions': ren_actions})
doc.submit_panel(ren_spec, (right_rect[0], right_rect[1], right_rect[2], selector_h - 6),
                 action_rects=panel._doc_action_rects, knob_rects=panel._doc_knob_rects)

cam_spec = P('camera_panel', 'Camera', knobs=list(d._PlayerCameraPanel.knobspec()))
visible = panel.doc_visible_knobs(list(cam_spec.knobs))
visible_spec = P('camera_panel', 'Camera', knobs=visible)
doc.submit_panel(visible_spec, camera_rect,
                 action_rects=panel._doc_action_rects, knob_rects=panel._doc_knob_rects)

print('action_rects:', list(panel._doc_action_rects.keys())[:6], 'count', len(panel._doc_action_rects))
print('knob_rects:  ', list(panel._doc_knob_rects.keys())[:6], 'count', len(panel._doc_knob_rects))

first = next(iter(panel._doc_knob_rects))
info = panel._doc_knob_rects[first]
rx, ry, rw, rh = info['rect']
click_x = rx + int(rw * 0.9)
click_y = ry + rh // 2

before = panel._values.get(first)
cam_before = getattr(cam, first, None)
print('before {}: panel={}, cam={}'.format(first, before, cam_before))

ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': (click_x, click_y), 'button': 1})
consumed = panel.handle_event(ev)
print('handle_event consumed:', consumed)
print('after  {}: panel={}, cam={}'.format(first, panel._values.get(first), getattr(cam, first, None)))

assert consumed, 'handle_event did not consume the click'
assert panel._values.get(first) != before, 'panel value did not change'
assert getattr(cam, first, None) != cam_before, 'cam attribute did not change'
print('PASS: knob click via handle_event mutates state')

# Render mode action via handle_event
key = next(iter(panel._doc_action_rects))
ax, ay, aw, ah = panel._doc_action_rects[key]
mode_before = panel.render_mode
ev2 = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': (ax + 5, ay + 5), 'button': 1})
panel.handle_event(ev2)
print('action', key, 'mode_before', mode_before, '->', panel.render_mode)
