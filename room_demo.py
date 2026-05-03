"""room_demo.py
==============
Standalone entry point for the RoomStation system.

Runs a full 3-D room with PlayerController (walk mode), all placed
enclosures, lights, and the HUD overlay — completely independent of
demo_pluck_gl.py.

Usage
-----
    python room_demo.py

Controls
--------
  WASD / arrows  Move (walk mode)
  Mouse           Look
  Scroll          Orbit zoom (orbit mode) / fov zoom
  Shift           Sprint
  Tab             Toggle orbit ↔ walk
  E               Interact (show HUD when near room console)
  Escape          Quit / close HUD
"""
from __future__ import annotations

import math
import sys
import time
import os

import numpy as np

# ── pygame + GL setup ─────────────────────────────────────────────────────────
os.environ.setdefault("SDL_VIDEO_X11_FORCE_EGL", "0")

import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, RESIZABLE,
    K_ESCAPE, K_TAB, K_e, K_w, K_a, K_s, K_d,
    K_UP, K_DOWN, K_LEFT, K_RIGHT, K_LSHIFT, K_RSHIFT,
    KEYDOWN, MOUSEMOTION, MOUSEBUTTONDOWN, MOUSEWHEEL, QUIT, VIDEORESIZE,
)

from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
    GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA,
    glBlendFunc, glClear, glClearColor, glEnable, glViewport,
)

from room_workspace import RoomWorkspace
from room_station   import RoomStation, _perspective, _look_at

WIN_W: int = 1400
WIN_H: int = 900
NEAR:  float = 0.05
FAR:   float = 200.0

# ─────────────────────────────────────────────────────────────────────────────
# Minimal room camera
# ─────────────────────────────────────────────────────────────────────────────

class _RoomCamera:
    """Minimal camera compatible with RoomStation.render_room() call signature.

    Supports orbit mode (mouse-drag rotate, scroll zoom) and a forced-eye
    override used by the player controller in walk mode.
    """

    def __init__(self, target=(0.0, 1.0, 5.0), dist=10.0,
                 elev=20.0, az=0.0, fov_deg=60.0):
        self.target   = np.array(target, np.float64)
        self.dist     = float(dist)
        self.elev     = float(elev)   # degrees above horizon
        self.az       = float(az)     # degrees, 0 = +Z
        self._fov_deg = float(fov_deg)
        self._forced_eye: np.ndarray | None = None  # set by walk controller
        self._forced_fwd: np.ndarray | None = None
        self._update_eye()

    def _update_eye(self):
        elev_r = math.radians(self.elev)
        az_r   = math.radians(self.az)
        self.eye = self.target + self.dist * np.array([
            math.cos(elev_r) * math.sin(az_r),
            math.sin(elev_r),
            math.cos(elev_r) * math.cos(az_r),
        ], np.float64)

    def orbit(self, daz: float, delev: float):
        self.az   = (self.az   + daz)   % 360.0
        self.elev = max(-89.0, min(89.0, self.elev + delev))
        self._update_eye()

    def zoom(self, d: float):
        self.dist = max(0.5, self.dist + d)
        self._update_eye()

    def fov_y_rad(self) -> float:
        return math.radians(self._fov_deg)

    def mv(self, aspect: float = 1.0) -> np.ndarray:
        """Return 4×4 view matrix (float32)."""
        if self._forced_eye is not None and self._forced_fwd is not None:
            eye = self._forced_eye
            tgt = eye + self._forced_fwd
        else:
            eye = self.eye
            tgt = self.target
        return _look_at(eye, tgt)

    def mvp(self, aspect: float, near: float = NEAR,
            far: float = FAR) -> np.ndarray:
        """Return MVP = P @ V (float32 4×4)."""
        P  = _perspective(self.fov_y_rad(), aspect, near, far)
        MV = self.mv(aspect)
        return (P @ MV).astype(np.float32)

    def active_eye(self) -> np.ndarray:
        if self._forced_eye is not None:
            return self._forced_eye
        return self.eye


# ─────────────────────────────────────────────────────────────────────────────
# Minimal walk controller (enough for room_demo; not importing demo_pluck_gl)
# ─────────────────────────────────────────────────────────────────────────────

class _WalkController:
    """Very simple FPS walk controller for room_demo.py.

    Reads physics config from RoomWorkspace and drives _RoomCamera.
    """

    def __init__(self, cam: _RoomCamera, phys: dict):
        walk = phys.get("walk", {})
        self._cam           = cam
        self._move_speed    = float(walk.get("move_speed",      3.0))
        self._run_mult      = float(walk.get("run_multiplier",  2.2))
        self._eye_height    = float(walk.get("eye_height",      1.65))
        self._sens          = float(walk.get("mouse_sensitivity", 0.15))
        self._pitch_limit   = float(walk.get("pitch_limit_deg", 75.0))

        sp = walk.get("start_pos", [0.0, 0.0, 0.8])
        self._pos    = np.array([sp[0], 0.0, sp[2]], np.float64)
        self._yaw    = float(walk.get("start_yaw", 0.0))   # degrees
        self._pitch  = 0.0

        self._walking = True    # False = orbit mode
        self._grab    = False   # mouse captured

        self._update_camera()

    def _update_camera(self):
        yr = math.radians(self._yaw)
        pr = math.radians(self._pitch)
        fwd = np.array([
            math.cos(pr) * math.sin(yr),
            math.sin(pr),
            math.cos(pr) * math.cos(yr),
        ], np.float64)
        eye = self._pos + np.array([0.0, self._eye_height, 0.0])
        self._cam._forced_eye = eye
        self._cam._forced_fwd = fwd

    def toggle_mode(self):
        self._walking = not self._walking
        if not self._walking:
            self._cam._forced_eye = None
            self._cam._forced_fwd = None
        else:
            self._update_camera()

    def is_walking(self):
        return self._walking

    def set_mouse_grab(self, grab: bool):
        self._grab = grab
        pygame.event.set_grab(grab)
        pygame.mouse.set_visible(not grab)

    def handle_event(self, ev) -> bool:
        if ev.type == MOUSEMOTION and self._walking and self._grab:
            dx, dy = ev.rel
            self._yaw   = (self._yaw   + dx * self._sens) % 360.0
            self._pitch = max(-self._pitch_limit,
                             min(self._pitch_limit,
                                 self._pitch - dy * self._sens))
            self._update_camera()
            return True
        if ev.type == MOUSEBUTTONDOWN:
            if ev.button == 1 and self._walking:
                self.set_mouse_grab(True)
                return True
            if ev.button == 3:
                self.set_mouse_grab(False)
                return True
        if ev.type == MOUSEWHEEL and not self._walking:
            self._cam.zoom(-ev.y * 0.5)
            return True
        if ev.type == MOUSEMOTION and not self._walking and ev.buttons[0]:
            self._cam.orbit(ev.rel[0] * 0.4, -ev.rel[1] * 0.4)
            return True
        return False

    def tick(self, dt: float, keys):
        if not self._walking:
            return
        speed = self._move_speed
        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            speed *= self._run_mult

        yr  = math.radians(self._yaw)
        fwd = np.array([math.sin(yr), 0.0, math.cos(yr)], np.float64)
        rgt = np.array([math.cos(yr), 0.0, -math.sin(yr)], np.float64)

        move = np.zeros(3, np.float64)
        if keys[K_w] or keys[K_UP]:    move += fwd
        if keys[K_s] or keys[K_DOWN]:  move -= fwd
        if keys[K_a] or keys[K_LEFT]:  move -= rgt
        if keys[K_d] or keys[K_RIGHT]: move += rgt

        n = np.linalg.norm(move)
        if n > 1e-6:
            self._pos += move / n * speed * dt

        self._update_camera()


# ─────────────────────────────────────────────────────────────────────────────
# Light direction helper
# ─────────────────────────────────────────────────────────────────────────────

def _sun_dir_v(MV: np.ndarray, world_dir=(0.5, 1.0, 0.6)) -> np.ndarray:
    """Transform a world-space light direction into view space."""
    wd = np.array(world_dir, np.float32)
    wd /= np.linalg.norm(wd)
    vd = MV[:3, :3] @ wd
    n  = np.linalg.norm(vd)
    return (vd / n if n > 1e-6 else vd).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(
        (WIN_W, WIN_H), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("Room Demo")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.04, 0.04, 0.06, 1.0)

    # ── Load workspace ──────────────────────────────────────────────────────
    ws = RoomWorkspace.from_yaml("configs/room_station")

    # ── Camera + controller ─────────────────────────────────────────────────
    dims  = ws.room_dims()
    W = float(dims.get("width_m",  12.0))
    D = float(dims.get("depth_m",  10.0))
    H = float(dims.get("height_m",  4.0))

    cam   = _RoomCamera(
        target=(0.0, H * 0.5, D * 0.5),
        dist=max(W, D) * 1.1,
        elev=30.0, az=0.0,
        fov_deg=60.0,
    )
    ctrl  = _WalkController(cam, ws.physics_config())

    # ── Room station ────────────────────────────────────────────────────────
    room_st = RoomStation(ws, WIN_W, WIN_H)
    room_st.init_gl()

    win_w, win_h = WIN_W, WIN_H
    prev_t = time.perf_counter()

    clock = pygame.time.Clock()

    # ── Main loop ───────────────────────────────────────────────────────────
    running = True
    while running:
        now  = time.perf_counter()
        dt   = min(now - prev_t, 0.05)
        prev_t = now

        keys = pygame.key.get_pressed()
        ctrl.tick(dt, keys)

        for ev in pygame.event.get():
            if ev.type == QUIT:
                running = False
                break

            if ev.type == VIDEORESIZE:
                win_w, win_h = ev.w, ev.h
                room_st.update_window_size(win_w, win_h)
                continue

            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    if room_st._hud_visible:
                        room_st.show_hud(False)
                        ctrl.set_mouse_grab(True)
                    else:
                        running = False
                    continue
                if ev.key == K_TAB:
                    ctrl.toggle_mode()
                    if ctrl.is_walking():
                        ctrl.set_mouse_grab(True)
                    else:
                        ctrl.set_mouse_grab(False)
                    continue
                if ev.key == K_e:
                    eye = cam.active_eye()
                    if room_st.player_near(eye):
                        visible = not room_st._hud_visible
                        room_st.show_hud(visible)
                        ctrl.set_mouse_grab(not visible)
                    continue

            if room_st.handle_event(ev):
                continue
            ctrl.handle_event(ev)

        # ── Render ──────────────────────────────────────────────────────────
        glViewport(0, 0, win_w, win_h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = win_w / max(1, win_h)
        MVP    = cam.mvp(aspect)
        MV     = cam.mv(aspect).astype(np.float32)
        lv     = _sun_dir_v(MV)

        room_st.render_room(win_w, win_h, MVP, MV, lv)
        room_st.render_hud(win_w, win_h)

        pygame.display.flip()
        clock.tick(60)

    room_st.destroy_gl()
    pygame.quit()


if __name__ == "__main__":
    main()
