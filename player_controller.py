"""player_controller.py
======================
PlayerController — a three-state camera state machine that owns the OpenGL
Camera object used in demo_pluck_gl.

States
------
ORBIT
    Classic orbit camera (existing demo_pluck_gl behavior).
    Left-drag = rotate around target.  Wheel = zoom.  WASD = pan target.
    Tab transitions to WALK.

WALK
    First-person navigation.  Mouse captured (relative mode).
    WASD = strafe/walk on the XY plane.  Shift = run.
    Tab returns to ORBIT.  E near a DutyStation transitions to INTERACT.

INTERACT
    Camera smoothly lerps to the station's console view.
    Mouse freed (used by sidebar panel).
    Escape / Q returns to WALK.  sidebar_visible = True in this state.

Usage
-----
    ctrl = PlayerController(camera, config_dict)

    # in event loop:
    for ev in pygame.event.get():
        if ctrl.handle_event(ev, duty_stations):
            continue
        ...

    # per frame:
    dt = clock.tick(60) / 1000.0
    keys = pygame.key.get_pressed()
    ctrl.tick(dt, keys, duty_stations)
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
    ORBIT    = "orbit"
    WALK     = "walk"
    INTERACT = "interact"


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

    def __init__(self, camera, config: dict):
        self.camera = camera
        self._cfg   = config

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

        i = config.get("interact", {})
        self._lerp_speed      = float(i.get("lerp_speed", 5.0))
        self._trigger_key     = _parse_key(str(i.get("trigger_key", "e")))
        self._exit_key        = _parse_key(str(i.get("exit_key", "escape")))
        self.hud_hint         = str(i.get("hud_hint", "[ E ]  Interact"))

        initial = config.get("initial_state", "orbit")
        self.state = PlayerState[initial.upper()]

        # runtime
        self._dragging      = False
        self._last_mouse    = (0, 0)
        self._active_station= None   # DutyStation being interacted with
        self._proximity_frac= 0.0   # 0..1 closeness to nearest station

        # Camera forced-eye override (set by PlayerController in walk/interact)
        if not hasattr(camera, '_forced_eye'):
            camera._forced_eye = None

        if self.state == PlayerState.WALK:
            self._enter_walk()

    # ── Public queries ────────────────────────────────────────────────────────

    @property
    def sidebar_visible(self) -> bool:
        return self.state == PlayerState.INTERACT

    @property
    def proximity_frac(self) -> float:
        """0 = far, 1 = at the station threshold.  Used to fade HUD hint."""
        return self._proximity_frac

    def player_eye(self) -> np.ndarray:
        """World-space eye position (works in all states)."""
        if self.state == PlayerState.ORBIT:
            return self.camera.eye
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

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_event(self, ev, duty_stations: list) -> bool:
        """Return True if the event was consumed."""

        if self.state == PlayerState.ORBIT:
            return self._orbit_event(ev)

        if self.state == PlayerState.WALK:
            return self._walk_event(ev, duty_stations)

        if self.state == PlayerState.INTERACT:
            return self._interact_event(ev)

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

    def _walk_event(self, ev, duty_stations: list) -> bool:
        if ev.type == pygame.KEYDOWN:
            if ev.key == self._to_orbit_key:
                self._enter_orbit()
                return True
            if ev.key == self._trigger_key:
                near = self._nearest_station(duty_stations)
                if near is not None:
                    self._enter_interact(near)
                    return True
        if ev.type == pygame.MOUSEMOTION:
            # captured relative mouse
            dx, dy = ev.rel
            self._walk_yaw   = (self._walk_yaw   + dx * self._walk_sensitivity) % 360.0
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
            if ev.key in (self._exit_key, pygame.K_q):
                self._enter_walk()
                return True
        return False

    # ── Per-frame tick ────────────────────────────────────────────────────────

    def tick(self, dt: float, keys, duty_stations: list):
        if self.state == PlayerState.ORBIT:
            self._tick_orbit(dt, keys)
        elif self.state == PlayerState.WALK:
            self._tick_walk(dt, keys, duty_stations)
        elif self.state == PlayerState.INTERACT:
            self._tick_interact(dt)

    def _tick_orbit(self, dt: float, keys):
        pass   # auto-rotate handled by Camera.tick() in the main loop

    def _tick_walk(self, dt: float, keys, duty_stations: list):
        speed = self._move_speed
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed *= self._run_mult

        yaw_r = math.radians(self._walk_yaw)
        fwd   = np.array([math.cos(yaw_r), math.sin(yaw_r), 0.0])
        right = np.array([-math.sin(yaw_r), math.cos(yaw_r), 0.0])

        move = np.zeros(3)
        if keys[pygame.K_w]: move += fwd
        if keys[pygame.K_s]: move -= fwd
        if keys[pygame.K_d]: move += right
        if keys[pygame.K_a]: move -= right

        n = np.linalg.norm(move)
        if n > 1e-9:
            self._walk_pos += (move / n) * speed * dt

        self._update_walk_camera()
        self._update_proximity(duty_stations)

    def _tick_interact(self, dt: float):
        """Smoothly lerp camera to station console view."""
        if self._active_station is None:
            return
        cfg = self._active_station.interact_camera
        t   = min(1.0, self._lerp_speed * dt)

        cur_eye = (self.camera._forced_eye if self.camera._forced_eye is not None
                   else self.camera.eye.copy())
        self.camera._forced_eye = cur_eye + t * (np.array(cfg['eye'])    - cur_eye)
        self.camera.target      = (self.camera.target
                                   + t * (np.array(cfg['target']) - self.camera.target))

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
            d = float(np.linalg.norm(eye - np.array(st.world_position)))
            if d < st.interaction_radius and d < best_d:
                best, best_d = st, d
        return best

    def _update_proximity(self, duty_stations: list):
        eye = self.player_eye()
        best = 0.0
        for st in duty_stations:
            d = float(np.linalg.norm(eye - np.array(st.world_position)))
            r = st.interaction_radius
            if r > 0:
                frac = max(0.0, 1.0 - d / r)
                best = max(best, frac)
        self._proximity_frac = best
