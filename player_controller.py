"""player_controller.py
======================
PlayerController — a four-state camera state machine that owns the OpenGL
Camera object used in demo_pluck_gl.

States
------
ORBIT
    Classic orbit camera (existing demo_pluck_gl behavior).
    Left-drag = rotate around target.  Wheel = zoom.  WASD = pan target.
    Tab transitions to WALK.

WALK
    First-person navigation.  Mouse captured (relative mode).
    WASD = strafe/walk.  Gravity + floor collision applied.
    Key-on/off drives continuous motion; no momentum.
    Tab returns to ORBIT.  E near a DutyStation → INTERACT.
    E near a CameraItem → IN_CAMERA.

INTERACT
    Camera smoothly lerps to the station's console view.
    Mouse freed (used by sidebar panel).
    Escape / Q returns to WALK.  sidebar_visible = True in this state.

IN_CAMERA
    Player is operating a physical camera item on an armature.
    Arrow keys pan/tilt the camera (mechanical step-wise: held key = continuous
    rate, release = stop instantly).  +/- zoom (FOV).  E or Escape = exit.
    The viewport tracks the camera's physical orientation and FOV.
    Mouse freed.  Left panel should show camera controls.

Usage
-----
    ctrl = PlayerController(camera, config_dict)

    # in event loop:
    for ev in pygame.event.get():
        if ctrl.handle_event(ev, duty_stations, cameras=cameras):
            continue
        ...

    # per frame:
    dt = clock.tick(60) / 1000.0
    keys = pygame.key.get_pressed()
    ctrl.tick(dt, keys, duty_stations, cameras=cameras)
"""
from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pygame

if TYPE_CHECKING:
    pass   # avoid circular imports; Camera is passed by value


# ─────────────────────────────────────────────────────────────────────────────

class PlayerState(Enum):
    ORBIT     = "orbit"
    WALK      = "walk"
    INTERACT  = "interact"
    IN_CAMERA = "in_camera"


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _parse_key(name: str) -> int:
    """Convert a key name string to a pygame key constant."""
    mapping = {
        "tab":    pygame.K_TAB,
        "escape": pygame.K_ESCAPE,
        "e":      pygame.K_e,
        "q":      pygame.K_q,
        "space":  pygame.K_SPACE,
    }
    return mapping.get(name.lower(), pygame.K_TAB)


# ─────────────────────────────────────────────────────────────────────────────

class PlayerController:
    """Camera state machine.  Call handle_event + tick every frame."""

    def __init__(self, camera, config: dict,
                 lens_loader=None, sensor_loader=None):
        """Create the controller.

        Parameters
        ----------
        camera : Camera
            The GL Camera instance owned by the renderer.
        config : dict
            Parsed player config dict (configs/player/default.yaml).
        lens_loader : callable(name: str) -> LensSpec | None
            Passed as ``LensSpec.load`` from demo_pluck_gl.py.  Used by
            ``_enter_camera`` to install the placed camera's full LensSpec
            onto the GL camera so ``fov_y_rad()`` is physically correct.
        sensor_loader : callable(name: str) -> SensorSpec | None
            Passed as ``SensorSpec.load`` from demo_pluck_gl.py.
        """
        self.camera = camera
        self._cfg   = config
        self._lens_loader   = lens_loader
        self._sensor_loader = sensor_loader

        # ── parse config ──────────────────────────────────────────────────────
        o = config.get("orbit", {})
        self._orb_sensitivity = float(o.get("mouse_sensitivity", 0.35))
        self._orb_zoom_rate   = float(o.get("zoom_rate", 0.04))
        self._to_walk_key     = _parse_key(str(o.get("transition_to_walk_key", "tab")))

        w = config.get("walk", {})
        _sp = w.get("start_pos", [0.0, -1.0, 0.0])
        self._walk_pos        = np.array(_sp, np.float64)
        self._walk_yaw        = float(w.get("start_yaw", 90.0))
        self._walk_pitch      = 0.0
        self._eye_height      = float(w.get("eye_height", 1.65))
        self._move_speed      = float(w.get("move_speed", 2.5))
        self._run_mult        = float(w.get("run_multiplier", 2.2))
        self._walk_sensitivity= float(w.get("mouse_sensitivity", 0.15))
        self._pitch_limit     = float(w.get("pitch_limit_deg", 75.0))
        self._to_orbit_key    = _parse_key(str(w.get("transition_to_orbit_key", "tab")))
        # Gravity + floor collision
        self._gravity         = float(w.get("gravity", 9.81))   # m/s²
        self._floor_z         = float(w.get("floor_z", 0.0))    # world Z of floor
        self._walk_vel_z      = 0.0                              # vertical velocity

        i = config.get("interact", {})
        self._lerp_speed      = float(i.get("lerp_speed", 5.0))
        self._trigger_key     = _parse_key(str(i.get("trigger_key", "e")))
        self._exit_key        = _parse_key(str(i.get("exit_key", "escape")))
        self.hud_hint         = str(i.get("hud_hint", "[ E ]  Interact"))

        # Camera-operate mode config
        c = config.get("camera_operate", {})
        self._cam_pan_rate_deg   = float(c.get("pan_rate_deg_s",    45.0))  # deg/s
        self._cam_tilt_rate_deg  = float(c.get("tilt_rate_deg_s",   30.0))  # deg/s
        self._cam_focal_rate_mm  = float(c.get("focal_rate_mm_s",   20.0))  # mm/s
        self._cam_tilt_min       = float(c.get("tilt_min_deg",      -80.0))
        self._cam_tilt_max       = float(c.get("tilt_max_deg",       80.0))

        initial = config.get("initial_state", "orbit")
        self.state = PlayerState[initial.upper()]

        # runtime
        self._dragging       = False
        self._last_mouse     = (0, 0)
        self._active_station = None   # DutyStation being interacted with
        self._active_camera  = None   # CameraItem being operated
        self._proximity_frac = 0.0   # 0..1 closeness to nearest interactable

        # Camera forced-eye override (set by PlayerController in walk/interact)
        if not hasattr(camera, '_forced_eye'):
            camera._forced_eye = None

        # Saved GL camera optics during IN_CAMERA mode (restored on exit)
        self._saved_lens   = None
        self._saved_sensor = None

        if self.state == PlayerState.WALK:
            self._enter_walk()

    # ── Public queries ────────────────────────────────────────────────────────

    @property
    def sidebar_visible(self) -> bool:
        return self.state in (PlayerState.INTERACT, PlayerState.IN_CAMERA)

    @property
    def proximity_frac(self) -> float:
        """0 = far, 1 = at the station threshold.  Used to fade HUD hint."""
        return self._proximity_frac

    def player_eye(self) -> np.ndarray:
        """World-space eye position (works in all states)."""
        if self.state == PlayerState.ORBIT:
            return self.camera.eye
        if self.state == PlayerState.IN_CAMERA and self._active_camera is not None:
            return np.array(self._active_camera.pos, np.float64)
        eye = self._walk_pos.copy()
        eye[2] += self._eye_height
        return eye

    # ── State transitions ─────────────────────────────────────────────────────

    def _enter_orbit(self):
        self.state = PlayerState.ORBIT
        self.camera._forced_eye = None
        if pygame.mouse.get_visible() is False:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

    def _enter_walk(self):
        self.state = PlayerState.WALK
        # Seed walk position from current camera eye (orbit → walk)
        eye = self.camera.eye.copy()
        self._walk_pos = eye.copy()
        self._walk_pos[2] = max(0.0, eye[2] - self._eye_height)
        # Seed yaw from camera azimuth
        self._walk_yaw   = float(self.camera.az)
        self._walk_pitch = 0.0
        self._active_station = None
        self._update_walk_camera()
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def _enter_interact(self, station):
        self.state = PlayerState.INTERACT
        self._active_station = station
        self._dragging = False
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

    def _enter_camera(self, camera_item) -> None:
        """Enter camera-operate mode for *camera_item*.

        Installs the camera item's full LensSpec and SensorSpec onto the
        GL Camera so ``fov_y_rad()``, distortion, vignetting, and all other
        optical parameters reflect the actual placed camera — not whatever
        preset the GL camera was initialised with.

        The previous lens and sensor are saved and restored on ``_exit_camera``.
        """
        self.state = PlayerState.IN_CAMERA
        self._active_camera = camera_item
        self._dragging = False
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

        # Save current GL camera optics
        if hasattr(self.camera, 'lens'):
            self._saved_lens   = self.camera.lens
        if hasattr(self.camera, 'sensor'):
            self._saved_sensor = self.camera.sensor

        # Install this camera item's LensSpec
        lens_name   = str(getattr(getattr(camera_item, 'placed', camera_item),
                                  'lens_name', 'standard_35mm'))
        sensor_name = str(getattr(getattr(camera_item, 'placed', camera_item),
                                  'sensor_name', 'full_frame_35mm'))
        if self._lens_loader is not None:
            try:
                new_lens = self._lens_loader(lens_name)
                if hasattr(self.camera, 'lens'):
                    self.camera.lens = new_lens
            except Exception:
                pass
        if self._sensor_loader is not None:
            try:
                new_sensor = self._sensor_loader(sensor_name)
                if hasattr(self.camera, 'sensor'):
                    self.camera.sensor = new_sensor
            except Exception:
                pass

        # Seed focal_mm from the camera item's live state
        focal = float(getattr(camera_item, 'focal_mm', 35.0))
        if hasattr(self.camera, 'focal_mm'):
            self.camera.focal_mm = focal
        if hasattr(self.camera, 'lens'):
            self.camera.lens.focal_mm = focal
        # Seed focus_m from lens minimum (if available)
        if hasattr(self.camera, 'focus_m') and hasattr(self.camera, 'lens'):
            self.camera.focus_m = getattr(self.camera.lens, 'min_focus_m', 0.45)

        # Snap viewport
        self._update_camera_view(camera_item)

    def _exit_camera(self) -> None:
        """Leave camera-operate mode and return to WALK.

        Restores the GL Camera's original LensSpec and SensorSpec so orbit
        mode and other views are not permanently affected by IN_CAMERA optics.
        """
        # Restore saved optics
        if self._saved_lens is not None and hasattr(self.camera, 'lens'):
            self.camera.lens = self._saved_lens
            if hasattr(self.camera, 'focal_mm'):
                self.camera.focal_mm = self.camera.lens.focal_mm
        if self._saved_sensor is not None and hasattr(self.camera, 'sensor'):
            self.camera.sensor = self._saved_sensor
        self._saved_lens   = None
        self._saved_sensor = None
        self._active_camera = None
        self._enter_walk()

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev, duty_stations: list,
                     cameras: Optional[list] = None) -> bool:
        """Return True if the event was consumed."""

        if self.state == PlayerState.ORBIT:
            return self._orbit_event(ev)

        if self.state == PlayerState.WALK:
            return self._walk_event(ev, duty_stations, cameras or [])

        if self.state == PlayerState.INTERACT:
            return self._interact_event(ev)

        if self.state == PlayerState.IN_CAMERA:
            return self._camera_event(ev)

        return False

    def _orbit_event(self, ev) -> bool:
        if ev.type == pygame.KEYDOWN:
            if ev.key == self._to_walk_key:
                self._enter_walk()
                return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._dragging   = True
            self._last_mouse = ev.pos
            return False   # let demo handle drag too (don't consume)
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._dragging = False
            return False
        if ev.type == pygame.MOUSEMOTION and self._dragging:
            dx = ev.pos[0] - self._last_mouse[0]
            dy = ev.pos[1] - self._last_mouse[1]
            self.camera.orbit(dx * self._orb_sensitivity,
                              -dy * self._orb_sensitivity)
            self.camera._auto = 0.0
            self._last_mouse = ev.pos
            return True
        if ev.type == pygame.MOUSEWHEEL:
            self.camera.zoom(-ev.y * self._orb_zoom_rate)
            return True
        return False

    def _walk_event(self, ev, duty_stations: list,
                    cameras: list) -> bool:
        if ev.type == pygame.KEYDOWN:
            if ev.key == self._to_orbit_key:
                self._enter_orbit()
                return True
            if ev.key == self._trigger_key:
                # Check cameras first (they are interactable items in the room)
                near_cam = self._nearest_camera(cameras)
                if near_cam is not None:
                    self._enter_camera(near_cam)
                    return True
                near = self._nearest_station(duty_stations)
                if near is not None:
                    self._enter_interact(near)
                    return True
        if ev.type == pygame.MOUSEMOTION:
            # captured relative mouse
            dx, dy = ev.rel
            self._walk_yaw   = (self._walk_yaw   - dx * self._walk_sensitivity) % 360.0
            self._walk_pitch = float(np.clip(
                self._walk_pitch - dy * self._walk_sensitivity,
                -self._pitch_limit, self._pitch_limit))
            self._update_walk_camera()
            return True
        # absorb MOUSEWHEEL so it doesn't orbit/zoom
        if ev.type == pygame.MOUSEWHEEL:
            return True
        return False

    def _interact_event(self, ev) -> bool:
        if ev.type == pygame.KEYDOWN:
            if ev.key in (self._exit_key, pygame.K_q, self._trigger_key):
                self._enter_walk()
                return True
        return False

    def _camera_event(self, ev) -> bool:
        """Event handling in camera-operate mode.

        Arrow keys, +/-, and zoom controls are handled in tick() (held-key
        polling) so they are NOT consumed here.  We only intercept exit.
        """
        if ev.type == pygame.KEYDOWN:
            if ev.key in (self._exit_key, pygame.K_q, self._trigger_key):
                self._exit_camera()
                return True
        return False

    # ── Per-frame tick ────────────────────────────────────────────────────────

    def tick(self, dt: float, keys,
             duty_stations: list,
             cameras: Optional[list] = None):
        if self.state == PlayerState.ORBIT:
            self._tick_orbit(dt, keys)
        elif self.state == PlayerState.WALK:
            self._tick_walk(dt, keys, duty_stations, cameras or [])
        elif self.state == PlayerState.INTERACT:
            self._tick_interact(dt)
        elif self.state == PlayerState.IN_CAMERA:
            self._tick_camera(dt, keys)

    def _tick_orbit(self, dt: float, keys):
        pass   # auto-rotate handled by Camera.tick() in the main loop

    def _tick_walk(self, dt: float, keys, duty_stations: list,
                   cameras: list):
        speed = self._move_speed
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed *= self._run_mult

        yaw_r = math.radians(self._walk_yaw)
        fwd   = np.array([math.cos(yaw_r), math.sin(yaw_r), 0.0])
        right = np.array([math.sin(yaw_r), -math.cos(yaw_r), 0.0])

        move = np.zeros(3)
        if keys[pygame.K_w]: move += fwd
        if keys[pygame.K_s]: move -= fwd
        if keys[pygame.K_d]: move += right
        if keys[pygame.K_a]: move -= right

        n = np.linalg.norm(move)
        if n > 1e-9:
            self._walk_pos += (move / n) * speed * dt

        # Gravity + floor collision
        self._walk_vel_z -= self._gravity * dt
        self._walk_pos[2] += self._walk_vel_z * dt
        if self._walk_pos[2] <= self._floor_z:
            self._walk_pos[2] = self._floor_z
            self._walk_vel_z  = 0.0

        self._update_walk_camera()
        self._update_proximity(duty_stations, cameras)

    def _tick_interact(self, dt: float):
        """Smoothly lerp camera to station console view."""
        if self._active_station is None:
            return
        cfg = getattr(self._active_station, 'interact_camera', None)
        if not isinstance(cfg, dict) or 'eye' not in cfg or 'target' not in cfg:
            pos = np.array(getattr(self._active_station,
                                   'world_position',
                                   getattr(self._active_station, 'pos', [0.0, 0.0, 0.0])),
                           np.float64)
            cfg = {
                'eye': (pos + np.array([0.0, -1.2, 1.4], np.float64)).tolist(),
                'target': (pos + np.array([0.0, 0.0, 1.0], np.float64)).tolist(),
            }
        t   = min(1.0, self._lerp_speed * dt)

        cur_eye = (self.camera._forced_eye if self.camera._forced_eye is not None
                   else self.camera.eye.copy())
        self.camera._forced_eye = cur_eye + t * (np.array(cfg['eye'])    - cur_eye)
        self.camera.target      = (self.camera.target
                                   + t * (np.array(cfg['target']) - self.camera.target))

    def _tick_camera(self, dt: float, keys) -> None:
        """Mechanical armature-style camera controls.

        All motion is proportional to ``dt``.  Held key = continuous movement
        at a fixed rate.  Release = stops immediately.  No momentum.
        """
        cam = self._active_camera
        if cam is None:
            self._enter_walk()
            return

        pan_r   = self._cam_pan_rate_deg
        tilt_r  = self._cam_tilt_rate_deg
        focal_r = self._cam_focal_rate_mm

        if keys[pygame.K_LEFT]:  cam.pan_deg  += pan_r  * dt
        if keys[pygame.K_RIGHT]: cam.pan_deg  -= pan_r  * dt
        if keys[pygame.K_UP]:
            cam.tilt_deg = min(self._cam_tilt_max, cam.tilt_deg + tilt_r * dt)
        if keys[pygame.K_DOWN]:
            cam.tilt_deg = max(self._cam_tilt_min, cam.tilt_deg - tilt_r * dt)

        # Focal length: + increases focal_mm (telephoto), - decreases (wide)
        # Clamped to the lens's physical range stored on the camera item.
        focal_in  = keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]
        focal_out = keys[pygame.K_MINUS]  or keys[pygame.K_KP_MINUS]
        if focal_in or focal_out:
            f_min = float(getattr(cam, 'focal_min_mm', 12.0))
            f_max = float(getattr(cam, 'focal_max_mm', f_min))  # prime = no zoom
            if f_max > f_min:  # only drive if it is a zoom lens
                delta = focal_r * dt * (1 if focal_in else -1)
                cam.focal_mm = float(np.clip(
                    float(getattr(cam, 'focal_mm', f_min)) + delta,
                    f_min, f_max))

        self._update_camera_view(cam)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_walk_camera(self):
        """Push walk state into camera (forced-eye + target)."""
        eye   = self._walk_pos.copy()
        eye[2] += self._eye_height

        yaw_r   = math.radians(self._walk_yaw)
        pitch_r = math.radians(self._walk_pitch)
        look = np.array([
            math.cos(pitch_r) * math.cos(yaw_r),
            math.cos(pitch_r) * math.sin(yaw_r),
            math.sin(pitch_r),
        ])
        self.camera._forced_eye = eye
        self.camera.target      = eye + look

    def _nearest_station(self, duty_stations: list):
        """Return nearest DutyStation within its interaction_radius, or None."""
        eye = self.player_eye()
        best, best_d = None, float('inf')
        for st in duty_stations:
            pos = np.array(getattr(st, 'world_position',
                                   getattr(st, 'pos', [0.0, 0.0, 0.0])), np.float64)
            r = float(getattr(st, 'interaction_radius', 1.0))
            d = float(np.linalg.norm(eye - pos))
            if d < r and d < best_d:
                best, best_d = st, d
        return best

    def _nearest_camera(self, cameras: list):
        """Return nearest CameraItem within its interaction_radius, or None."""
        eye = self.player_eye()
        best, best_d = None, float('inf')
        for c in cameras:
            pos = np.array(getattr(c, 'pos', getattr(c, 'world_position', [0, 0, 0])),
                           np.float64)
            r = float(getattr(c, 'interaction_radius', 1.5))
            d = float(np.linalg.norm(eye - pos))
            if d < r and d < best_d:
                best, best_d = c, d
        return best

    def _update_camera_view(self, cam) -> None:
        """Push the camera item's physical orientation into the GL camera.

        Only eye position, target, and the live ``focal_mm`` are synced here.
        The lens and sensor specs were already installed by ``_enter_camera``
        and are managed by the CameraHudPanel sliders.
        """
        pos    = np.array(cam.pos, np.float64)
        pan_r  = math.radians(float(getattr(cam, 'pan_deg',  0.0)))
        tilt_r = math.radians(float(getattr(cam, 'tilt_deg', 0.0)))
        look = np.array([
            math.cos(tilt_r) * math.cos(pan_r),
            math.cos(tilt_r) * math.sin(pan_r),
            math.sin(tilt_r),
        ])
        self.camera._forced_eye = pos
        self.camera.target      = pos + look
        # Sync live focal length only — all other optical params come from
        # the installed LensSpec/SensorSpec and the CameraHudPanel.
        focal = float(getattr(cam, 'focal_mm', 35.0))
        if hasattr(self.camera, 'focal_mm'):
            self.camera.focal_mm = focal
        if hasattr(self.camera, 'lens'):
            self.camera.lens.focal_mm = focal

    def _update_proximity(self, duty_stations: list,
                          cameras: Optional[list] = None) -> None:
        eye  = self.player_eye()
        best = 0.0
        interactables = list(duty_stations) + list(cameras or [])
        for item in interactables:
            pos = np.array(
                getattr(item, 'world_position',
                        getattr(item, 'pos', [0, 0, 0])), np.float64)
            r = float(getattr(item, 'interaction_radius', 1.0))
            if r > 0:
                d    = float(np.linalg.norm(eye - pos))
                frac = max(0.0, 1.0 - d / r)
                best = max(best, frac)
        self._proximity_frac = best
