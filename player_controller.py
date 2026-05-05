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
import os
import time
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pygame

try:
    import yaml as _yaml
except Exception:
    _yaml = None

try:
    from ray_tracer_bridge import trace_cone_rays as _trace_focus_cone_rays
    from ray_tracer_bridge import trace_single_ray as _trace_focus_single_ray
except Exception:
    _trace_focus_cone_rays = None
    _trace_focus_single_ray = None

if TYPE_CHECKING:
    pass   # avoid circular imports; Camera is passed by value


_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_yaml(path: str) -> dict:
    if _yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _default_synthesis_recipes() -> dict:
    return {
        "basic_paneling": {
            "label": "Basic Paneling",
            "seconds": 3.0,
            "output": {"material": "basic_paneling", "quantity": 1},
            "inputs": {
                "raw_regolith": 3,
                "binder_resin": 1,
                "fiber_mesh": 1,
                "surface_coat": 1,
            },
        },
        "basic_led_display": {
            "label": "Basic LED Display",
            "seconds": 6.0,
            "output": {"material": "basic_led_display", "quantity": 1},
            "inputs": {
                "silica_sand": 2,
                "copper_trace": 1,
                "emitter_dust": 1,
                "polymer_film": 1,
                "control_chip": 1,
            },
        },
    }


def _load_synthesis_recipes() -> dict:
    data = _load_yaml(os.path.join(_HERE, "configs", "crafting", "recipes.yaml"))
    recipes = data.get("recipes", {}) if isinstance(data, dict) else {}
    if not isinstance(recipes, dict) or not recipes:
        return _default_synthesis_recipes()
    out = {}
    for key, raw in recipes.items():
        if not isinstance(raw, dict):
            continue
        output = dict(raw.get("output", {}) or {})
        material = str(output.get("material", key))
        qty = max(1, int(output.get("quantity", 1)))
        inputs = {
            str(k): max(0, int(v))
            for k, v in dict(raw.get("inputs", {}) or {}).items()
            if int(v) > 0
        }
        out[str(key)] = {
            "label": str(raw.get("label", material)),
            "seconds": max(0.1, float(raw.get("seconds", 1.0))),
            "output": {"material": material, "quantity": qty},
            "inputs": inputs,
            "workbench": str(raw.get("workbench", "personal_synth")),
        }
    return out or _default_synthesis_recipes()


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
        self._ray_pick_max_dist = float(i.get("ray_pick_max_dist_m", 6.0))
        self._focus_ray_engine = str(i.get("focus_ray_engine", "bridge")).strip().lower() or "bridge"
        self._focus_cone_angle_deg = float(i.get("focus_cone_angle_deg", 0.0))
        self._focus_cone_rays = max(1, int(i.get("focus_cone_rays", 1)))
        self._synthesis_recipes = _load_synthesis_recipes()
        self._synthesis_job: dict | None = None

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
        self._focus_target   = None   # interactable centered in player view while disambiguating
        self._backpack: dict[str, int] = {}

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

    @property
    def focus_target(self):
        """Interactable currently under the center-view focus ray, if any."""
        return self._focus_target

    @property
    def backpack(self) -> dict:
        return dict(self._backpack)

    @property
    def synthesis_recipes(self) -> dict:
        return {k: dict(v) for k, v in self._synthesis_recipes.items()}

    def recipe_availability(self, recipe_key: str) -> dict:
        key = str(recipe_key)
        recipe = self._synthesis_recipes.get(key)
        if recipe is None:
            return {"known": False, "craftable": False, "needs": []}
        needs = []
        craftable = True
        for mat, qty in dict(recipe.get("inputs", {}) or {}).items():
            need = max(0, int(qty))
            have = max(0, int(self._backpack.get(mat, 0)))
            ok = have >= need
            craftable = craftable and ok
            needs.append({
                "material": str(mat),
                "need": need,
                "have": have,
                "missing": max(0, need - have),
                "ok": ok,
            })
        return {
            "known": True,
            "craftable": craftable,
            "needs": needs,
        }

    def synthesis_status(self) -> dict:
        job = self._synthesis_job
        if not isinstance(job, dict):
            return {"active": False}
        now = time.perf_counter()
        start = float(job.get("start_s", now))
        end = float(job.get("end_s", now))
        total = max(1e-6, end - start)
        remaining = max(0.0, end - now)
        return {
            "active": True,
            "material": str(job.get("material", "")),
            "label": str(job.get("label", job.get("material", ""))),
            "reserved": dict(job.get("reserved", {}) or {}),
            "progress": float(max(0.0, min(1.0, (now - start) / total))),
            "remaining_s": remaining,
            "duration_s": total,
        }

    def begin_synthesis(self, material_key: str) -> bool:
        key = str(material_key)
        if self._synthesis_job is not None:
            return False
        recipe = self._synthesis_recipes.get(key)
        if recipe is None:
            return False
        availability = self.recipe_availability(key)
        if not bool(availability.get("craftable", False)):
            return False
        now = time.perf_counter()
        delay = max(0.1, float(recipe.get("seconds", 1.0)))
        reserved = {}
        for mat, qty in dict(recipe.get("inputs", {}) or {}).items():
            take = max(0, int(qty))
            if take <= 0:
                continue
            self._backpack[mat] = int(self._backpack.get(mat, 0)) - take
            reserved[str(mat)] = take
        output = dict(recipe.get("output", {}) or {})
        self._synthesis_job = {
            "recipe": key,
            "material": str(output.get("material", key)),
            "label": str(recipe.get("label", key)),
            "quantity": max(1, int(output.get("quantity", 1))),
            "reserved": reserved,
            "start_s": now,
            "end_s": now + delay,
        }
        return True

    def cancel_synthesis(self) -> bool:
        job = self._synthesis_job
        if not isinstance(job, dict):
            return False
        for mat, qty in dict(job.get("reserved", {}) or {}).items():
            self._backpack[str(mat)] = int(self._backpack.get(str(mat), 0)) + max(0, int(qty))
        self._synthesis_job = None
        return True

    def update_synthesis(self) -> None:
        self._tick_synthesis()

    @property
    def focus_ray_engine(self) -> str:
        return self._focus_ray_engine

    def set_focus_ray_engine(self, engine: str) -> None:
        eng = str(engine or "bridge").strip().lower()
        if eng not in {"bridge", "analytic"}:
            eng = "bridge"
        self._focus_ray_engine = eng

    @property
    def focus_pick_max_dist(self) -> float:
        return self._ray_pick_max_dist

    def set_focus_pick_max_dist(self, value: float) -> None:
        self._ray_pick_max_dist = max(0.10, float(value))

    @property
    def focus_cone_angle_deg(self) -> float:
        return self._focus_cone_angle_deg

    def set_focus_cone_angle_deg(self, value: float) -> None:
        self._focus_cone_angle_deg = max(0.0, float(value))

    @property
    def focus_cone_rays(self) -> int:
        return self._focus_cone_rays

    def set_focus_cone_rays(self, value: float) -> None:
        self._focus_cone_rays = max(1, int(round(float(value))))

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
        self._focus_target = None
        self._update_walk_camera()
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def _enter_interact(self, station):
        self.state = PlayerState.INTERACT
        self._active_station = station
        self._focus_target = None
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
        self._focus_target = None
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
        self._focus_target = None
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
                hit = self._center_view_interactable(duty_stations, cameras)
                if hit is not None:
                    if hit in cameras:
                        self._enter_camera(hit)
                    else:
                        if hasattr(hit, "try_pickup_material"):
                            self._try_pickup_material(hit)
                            return True
                        if bool(getattr(hit, "is_unfinished", False)):
                            self._try_deliver_to_station(hit)
                        if bool(getattr(hit, "is_unfinished", False)):
                            return True
                        self._enter_interact(hit)
                    return True
                # Fallback to proximity when center ray does not hit anything actionable.
                near_cam = self._nearest_camera(cameras)
                if near_cam is not None:
                    self._enter_camera(near_cam)
                    return True
                near = self._nearest_station(duty_stations)
                if near is not None:
                    if hasattr(near, "try_pickup_material"):
                        self._try_pickup_material(near)
                        return True
                    if bool(getattr(near, "is_unfinished", False)):
                        self._try_deliver_to_station(near)
                        if bool(getattr(near, "is_unfinished", False)):
                            return True
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

    def _try_pickup_material(self, pile) -> None:
        if pile is None or not hasattr(pile, "try_pickup_material"):
            return
        try:
            pile.try_pickup_material(self._backpack, actor_id="player")
        except Exception:
            return

    def _try_deliver_to_station(self, station) -> None:
        if station is None or not hasattr(station, "try_deliver_materials"):
            return
        try:
            station.try_deliver_materials(self._backpack, actor_id="player")
        except Exception:
            return
        self._persist_station_build_state(station)

    @staticmethod
    def _persist_station_build_state(station) -> None:
        ref = getattr(station, "_placed_ref", None)
        if ref is None:
            return
        try:
            ref.build_state = {
                "unfinished": bool(getattr(station, "is_unfinished", False)),
                "job_order_id": str(getattr(station, "job_order_id", "")),
                "required_materials": dict(getattr(station, "required_materials", {})),
                "delivered_materials": dict(getattr(station, "delivered_materials", {})),
            }
        except Exception:
            pass

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
        self.update_synthesis()
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

    def _tick_synthesis(self) -> None:
        job = self._synthesis_job
        if not isinstance(job, dict):
            return
        if time.perf_counter() < float(job.get("end_s", 0.0)):
            return
        material = str(job.get("material", ""))
        qty = max(1, int(job.get("quantity", 1)))
        if material:
            self._backpack[material] = int(self._backpack.get(material, 0)) + qty
        self._synthesis_job = None

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
        self._update_focus_target(keys, duty_stations, cameras)
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

    def _view_ray(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (origin, direction) for the player's center-view ray."""
        origin = self.player_eye().astype(np.float64)
        tgt = np.asarray(getattr(self.camera, 'target', origin + np.array([1.0, 0.0, 0.0])), np.float64)
        d = tgt - origin
        n = float(np.linalg.norm(d))
        if n <= 1e-9:
            yaw_r = math.radians(self._walk_yaw)
            pitch_r = math.radians(self._walk_pitch)
            d = np.array([
                math.cos(pitch_r) * math.cos(yaw_r),
                math.cos(pitch_r) * math.sin(yaw_r),
                math.sin(pitch_r),
            ], np.float64)
            n = float(np.linalg.norm(d))
        return origin, (d / max(n, 1e-12))

    @staticmethod
    def _ray_triangle_t(ro: np.ndarray, rd: np.ndarray,
                        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                        eps: float = 1e-8) -> float:
        """Moller-Trumbore intersection distance, or +inf when no hit."""
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = np.cross(rd, e2)
        det = float(np.dot(e1, pvec))
        if abs(det) < eps:
            return float('inf')
        inv_det = 1.0 / det
        tvec = ro - v0
        u = float(np.dot(tvec, pvec) * inv_det)
        if u < 0.0 or u > 1.0:
            return float('inf')
        qvec = np.cross(tvec, e1)
        v = float(np.dot(rd, qvec) * inv_det)
        if v < 0.0 or (u + v) > 1.0:
            return float('inf')
        t = float(np.dot(e2, qvec) * inv_det)
        if t <= eps:
            return float('inf')
        return t

    @staticmethod
    def _ray_sphere_t(ro: np.ndarray, rd: np.ndarray, center: np.ndarray, radius: float) -> float:
        """Ray/sphere nearest positive hit distance, or +inf."""
        oc = ro - center
        b = float(np.dot(oc, rd))
        c = float(np.dot(oc, oc) - radius * radius)
        disc = b * b - c
        if disc < 0.0:
            return float('inf')
        s = math.sqrt(disc)
        t0 = -b - s
        t1 = -b + s
        if t0 > 1e-8:
            return t0
        if t1 > 1e-8:
            return t1
        return float('inf')

    @staticmethod
    def _interaction_triangles(item) -> Optional[np.ndarray]:
        """Return world-space triangle soup (N,3,3) for an interactable, if available."""
        fn = getattr(item, 'interaction_triangles_world', None)
        if not callable(fn):
            return None
        try:
            tris = np.asarray(fn(), np.float64)
        except Exception:
            return None
        if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
            return None
        return tris

    def _build_interaction_pick_soup(self, duty_stations: list, cameras: list):
        """Owner-tagged pick primitives for center-ray interaction."""
        owners = list(cameras) + list(duty_stations)
        soup = []
        for owner in owners:
            tris = self._interaction_triangles(owner)
            soup.append((owner, tris))
        return soup

    def _build_interaction_trace_arrays(self, duty_stations: list, cameras: list):
        owners = []
        tri_batches = []
        normal_batches = []
        owner_tags = []
        for owner_idx, (owner, tris) in enumerate(self._build_interaction_pick_soup(duty_stations, cameras)):
            if tris is None or len(tris) == 0:
                continue
            owners.append(owner)
            tri_batches.append(tris)
            e1 = tris[:, 1, :] - tris[:, 0, :]
            e2 = tris[:, 2, :] - tris[:, 0, :]
            norms = np.cross(e1, e2)
            nm = np.linalg.norm(norms, axis=1, keepdims=True)
            norms = norms / np.where(nm > 1e-12, nm, 1.0)
            normal_batches.append(norms)
            owner_tags.append(np.full(len(tris), len(owners) - 1, dtype=np.int32))
        if not tri_batches:
            return None, None, None, owners
        return (
            np.concatenate(tri_batches, axis=0),
            np.concatenate(normal_batches, axis=0),
            np.concatenate(owner_tags, axis=0),
            owners,
        )

    def _center_view_interactable_bridge(self, duty_stations: list, cameras: list):
        """Bridge-backed focus path using owner-tagged no-bounce bridge ray entry points."""
        ro, rd = self._view_ray()
        owner, _hit_t = self._pick_interactable_with_ray_bridge(ro, rd, duty_stations, cameras)
        return owner

    def _pick_interactable_with_ray_bridge(
        self,
        ro: np.ndarray,
        rd: np.ndarray,
        duty_stations: list,
        cameras: list,
    ) -> tuple[object | None, float]:
        """Bridge-backed owner pick for an arbitrary ray direction."""
        if _trace_focus_single_ray is None:
            return self._pick_interactable_with_ray_analytic(ro, rd, duty_stations, cameras)

        verts, normals, owner_tags, owners = self._build_interaction_trace_arrays(duty_stations, cameras)
        if verts is None or owner_tags is None or not owners:
            return self._pick_interactable_with_ray_analytic(ro, rd, duty_stations, cameras)

        max_t = max(0.10, float(self._ray_pick_max_dist))
        cone_rays = max(1, int(self._focus_cone_rays))
        cone_angle_rad = math.radians(max(0.0, float(self._focus_cone_angle_deg)))

        if cone_rays <= 1 or cone_angle_rad <= 1e-12 or _trace_focus_cone_rays is None:
            hit = _trace_focus_single_ray(
                None,
                ro,
                rd,
                max_distance=max_t,
                verts=verts,
                normals=normals,
                owner_tags=owner_tags,
            )
            owner_idx = hit.get('owner_tag', -1)
            if owner_idx is None or int(owner_idx) < 0:
                return None, float('inf')
            return owners[int(owner_idx)], float(hit.get('distance', float('inf')))

        hits = _trace_focus_cone_rays(
            None,
            ro,
            rd,
            cone_angle_rad=cone_angle_rad,
            n_rays=cone_rays,
            max_distance=max_t,
            verts=verts,
            normals=normals,
            owner_tags=owner_tags,
        )
        owner_hit_counts = {}
        owner_best_t = {}
        owner_id_arr = np.asarray(hits.get('owner_tag', []))
        distance_arr = np.asarray(hits.get('distance', []), np.float64)
        hit_mask = np.asarray(hits.get('hit', []), dtype=bool)
        for idx, did_hit in enumerate(hit_mask):
            if not did_hit:
                continue
            owner_idx = int(owner_id_arr[idx])
            if owner_idx < 0 or owner_idx >= len(owners):
                continue
            owner_hit_counts[owner_idx] = owner_hit_counts.get(owner_idx, 0) + 1
            owner_best_t[owner_idx] = min(owner_best_t.get(owner_idx, float('inf')), float(distance_arr[idx]))
        if not owner_hit_counts:
            return None, float('inf')
        best_owner_idx = min(owner_hit_counts, key=lambda idx: (-owner_hit_counts[idx], owner_best_t.get(idx, float('inf'))))
        return owners[best_owner_idx], float(owner_best_t.get(best_owner_idx, float('inf')))

    def _center_view_interactable_analytic(self, duty_stations: list, cameras: list):
        """Analytic center-ray intersection against owner-tagged triangle soup."""
        ro, rd = self._view_ray()
        owner, _hit_t = self._pick_interactable_with_ray_analytic(ro, rd, duty_stations, cameras)
        return owner

    def _pick_interactable_with_ray_analytic(
        self,
        ro: np.ndarray,
        rd: np.ndarray,
        duty_stations: list,
        cameras: list,
    ) -> tuple[object | None, float]:
        """Analytic owner pick for an arbitrary ray direction."""
        max_t = max(0.10, float(self._ray_pick_max_dist))
        best_owner = None
        best_t = float('inf')

        for owner, tris in self._build_interaction_pick_soup(duty_stations, cameras):
            hit_t = float('inf')
            if tris is not None:
                for tri in tris:
                    t = self._ray_triangle_t(ro, rd, tri[0], tri[1], tri[2])
                    if t < hit_t:
                        hit_t = t
            else:
                pos = np.array(getattr(owner, 'world_position', getattr(owner, 'pos', [0.0, 0.0, 0.0])), np.float64)
                r = float(getattr(owner, 'interaction_radius', 1.0))
                hit_t = self._ray_sphere_t(ro, rd, pos, r)

            if hit_t < best_t and hit_t <= max_t:
                best_t = hit_t
                best_owner = owner

        return best_owner, float(best_t)

    def pick_interactable_with_ray(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        duty_stations: list,
        cameras: list,
    ) -> tuple[object | None, float]:
        """Public owner-pick API for arbitrary rays (e.g. HUD mouse hover rays)."""
        ro = np.asarray(ray_origin, np.float64)
        rd = np.asarray(ray_dir, np.float64)
        n = float(np.linalg.norm(rd))
        if n <= 1e-12:
            return None, float('inf')
        rd = rd / n
        if self._focus_ray_engine == "analytic":
            return self._pick_interactable_with_ray_analytic(ro, rd, duty_stations, cameras)
        return self._pick_interactable_with_ray_bridge(ro, rd, duty_stations, cameras)

    def _center_view_interactable(self, duty_stations: list, cameras: list):
        """Ray-pick interactable dead-center in view; None when no actionable hit."""
        if self._focus_ray_engine == "analytic":
            return self._center_view_interactable_analytic(duty_stations, cameras)
        return self._center_view_interactable_bridge(duty_stations, cameras)

    def _update_focus_target(self, keys, duty_stations: list, cameras: list) -> None:
        """When Shift is held, cache the centered interactable for wireframe disambiguation."""
        if not (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
            self._focus_target = None
            return
        self._focus_target = self._center_view_interactable(duty_stations, cameras)

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
