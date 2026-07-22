#!/usr/bin/env python3
"""
Thick Lens / Aperture Interactive Viewer — Real-time parameter tuning with live ray-traced preview.

A pygame/OpenGL viewport with background workers:
  - SceneUpdateWorker: rebuilds mesh geometry when parameters change
  - RayTracerWorker: runs C++ BDPT ray tracer, accumulates spectral radiance
  - UI thread: free-running mouse-driven parameter control (never blocked)

Controls
--------
    Left drag (on lens)     Slide lens along optical axis (x)
    Right drag (on lens)    Adjust lens aperture radius (wheel forward/back alternative)
    Mouse wheel (Shift)     Adjust lens aperture radius
    Mouse wheel (Ctrl)      Adjust lens thickness
    Mouse wheel (Alt)       Adjust lens curvature (radius of curvature)
    Mouse wheel (no mod)    Adjust emitter radius
    
    E                       Toggle emitter visibility
    A                       Toggle aperture stop visibility
    L                       Toggle lens geometry visibility
    R                       Reset to defaults
    Escape                  Quit

    Space                   Pause/resume ray tracing
    C                       Clear accumulated radiance (hard reset)

Display
-------
    Left panel:    Accumulated spectral radiance (false-color RGB)
    Right panel:   Parameter values + statistics
    
Dependencies
------------
    pip install numpy pygame PyOpenGL Pillow scipy
    (Assumes OpenGL backend and C++ ray tracer already built)
"""

from __future__ import annotations

import argparse
import gc
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple
from queue import Queue, Empty

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP,
    MOUSEMOTION, MOUSEWHEEL, OPENGL, QUIT, RESIZABLE, VIDEORESIZE,
    K_SPACE, K_ESCAPE, K_e, K_a, K_l, K_r, K_c, K_o, K_w,
    K_LSHIFT, K_RSHIFT, K_LCTRL, K_RCTRL, K_LALT, K_RALT,
)
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders
from PIL import Image, ImageDraw, ImageFont

from thick_lens_focus_lab import (
    SceneConfig, ForwardCppLensBench, DEFAULT_FREQ_HZ,
    LensConfig, MaterialDatabase, _build_scene_mesh, _scene_lenses, C_LIGHT,
    _compound_lens_from_scene,
    FreeFrequencySidecar, MAX_SPECTRAL_BANDS,
)
import _spectral_kernels as _sk
from camera_designer.lens_assembly import LensAssemblySpec
from camera_software import OpticalDesignSpec


@dataclass
class SceneParameters:
    """Mutable scene configuration for interactive tuning."""
    use_optical_design: bool = True
    zoom: float = 0.40
    focus_distance_m: float = 2.0
    f_number: float = 2.8
    focal_min_m: float = 0.070
    focal_max_m: float = 0.160
    aperture_model: str = "geometry"
    # Lens group
    lens_x_list: list[float] = field(default_factory=lambda: [0.3, 0.8])  # Primary lens positions
    lens_aperture_radius_list: list[float] = field(default_factory=lambda: [0.025, 0.022])
    lens_thickness_list: list[float] = field(default_factory=lambda: [0.010, 0.008])
    lens_curvature_list: list[float] = field(default_factory=lambda: [0.070, 0.060])
    
    # Aperture stop
    aperture_x: float = 1.85
    aperture_radius: float = 0.008
    
    # Emitter
    emitter_radius: float = 0.030
    emitter_intensity: float = 1.0
    
    # Visibility toggles
    show_emitter: bool = True
    show_aperture: bool = True
    show_lenses: bool = True
    show_baffles: bool = True
    
    def copy(self) -> SceneParameters:
        """Deep copy for worker thread use."""
        return SceneParameters(
            use_optical_design=self.use_optical_design,
            zoom=self.zoom,
            focus_distance_m=self.focus_distance_m,
            f_number=self.f_number,
            focal_min_m=self.focal_min_m,
            focal_max_m=self.focal_max_m,
            aperture_model=self.aperture_model,
            lens_x_list=self.lens_x_list.copy(),
            lens_aperture_radius_list=self.lens_aperture_radius_list.copy(),
            lens_thickness_list=self.lens_thickness_list.copy(),
            lens_curvature_list=self.lens_curvature_list.copy(),
            aperture_x=self.aperture_x,
            aperture_radius=self.aperture_radius,
            emitter_radius=self.emitter_radius,
            emitter_intensity=self.emitter_intensity,
            show_emitter=self.show_emitter,
            show_aperture=self.show_aperture,
            show_lenses=self.show_lenses,
            show_baffles=self.show_baffles,
        )


class MouseMode(Enum):
    """Track what the user is dragging."""
    IDLE = 0
    DRAG_LENS = 1        # Left drag on lens: move x
    DRAG_APERTURE = 2    # Left drag on aperture: move x


@dataclass
class SceneUpdateMessage:
    """Work unit: rebuild geometry with new parameters."""
    params: SceneParameters
    seq_id: int           # Sequence number for ordering


@dataclass
class RayTracerMessage:
    """Work unit: run ray tracing on current scene."""
    seq_id: int
    n_samples: int        # Photons per emitter for this batch
    timeout_sec: float = 2.0


@dataclass
class DisplayState:
    """Rendered scene + accumulated radiance."""
    radiance_rgb: Optional[np.ndarray] = None
    accumulated_count: int = 0
    mesh_verts: Optional[np.ndarray] = None
    mesh_normals: Optional[np.ndarray] = None
    mesh_faces: Optional[np.ndarray] = None
    suppress_ids: Optional[np.ndarray] = None
    timestamp: float = 0.0


class SceneUpdateWorker(threading.Thread):
    """Background thread: rebuilds mesh when parameters change."""
    
    def __init__(self, in_queue: Queue, out_queue: Queue):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.running = True
        
    def run(self):
        while self.running:
            try:
                msg: SceneUpdateMessage = self.in_queue.get(timeout=0.5)
            except Empty:
                continue
            
            try:
                # Build scene configuration from parameters
                base_scene = SceneConfig()
                params = msg.params
                base_scene.aperture_model = str(params.aperture_model)
                
                if params.use_optical_design:
                    base_scene.optical_design = OpticalDesignSpec(
                        focal_length_range_m=(float(params.focal_min_m), float(params.focal_max_m)),
                        zoom=float(params.zoom),
                        focus_distance_m=float(max(1.0e-3, params.focus_distance_m)),
                        f_number=float(max(0.2, params.f_number)),
                        entrance_x_m=0.88,
                        sensor_x_m=float(base_scene.image_plate.x),
                        image_radius_m=float(base_scene.image_plate.radius),
                    )
                else:
                    base_scene.optical_design = None
                    base_scene.lens_stack = []
                    for x, aperture_r, thickness, curvature in zip(
                        params.lens_x_list,
                        params.lens_aperture_radius_list,
                        params.lens_thickness_list,
                        params.lens_curvature_list,
                    ):
                        base_scene.lens_stack.append(LensConfig(
                            center_x=float(x),
                            thickness=float(max(1.0e-4, thickness)),
                            aperture_radius=float(max(1.0e-4, aperture_r)),
                            radius_front=float(max(1.0e-4, curvature)),
                            radius_back=float(max(1.0e-4, curvature)),
                            ior=1.52,
                        ))
                    base_scene.exit_pupil_x = float(params.aperture_x)
                    base_scene.exit_pupil_radius = float(max(1.0e-4, params.aperture_radius))
                base_scene.source_radius = float(max(1.0e-4, params.emitter_radius))
                
                # Rebuild mesh geometry
                sidecar = FreeFrequencySidecar.lazy_prepare(len(DEFAULT_FREQ_HZ))
                mesh_tuple = _build_scene_mesh(base_scene, sidecar)
                (
                    verts, normals, mat_idx, tri_arr, db,
                    source_ids, lens_front_ids, lens_back_ids, image_plate_ids,
                    aperture_stop_ids, suppress_ids, lens_surface_groups,
                    object_ids, tube_baffle_ids, camera_barrel_ids,
                    camera_rear_cap_ids, camera_front_cap_ids, camera_frustum_ids,
                    red_probe_ids, subject_group_tri_map,
                ) = mesh_tuple
                optics = _compound_lens_from_scene(base_scene)
                field_pair = optics.profile_field_pair(
                    object_point=(float(base_scene.object_plane.x), 0.0, 0.0),
                    image_point=(float(base_scene.image_plate.x), 0.0, 0.0),
                )
                
                # Package for ray tracer thread
                mesh_data = {
                    "scene": base_scene,
                    "optics": optics,
                    "field_pair": field_pair,
                    "tri_arr": tri_arr,
                    "verts": verts,
                    "normals": normals,
                    "mat_idx": mat_idx,
                    "db": db,
                    "source_ids": source_ids,
                    "lens_front_ids": lens_front_ids,
                    "lens_back_ids": lens_back_ids,
                    "lens_surface_groups": lens_surface_groups,
                    "image_plate_ids": image_plate_ids,
                    "aperture_stop_ids": aperture_stop_ids,
                    "suppress_ids": suppress_ids,
                    "tube_baffle_ids": tube_baffle_ids,
                    "freq_hz": np.ascontiguousarray(np.asarray(DEFAULT_FREQ_HZ, dtype=np.float64)),
                }
                
                print(f"[SceneWorker] seq={msg.seq_id} mesh ready: {tri_arr.shape[0]} tris")
                
                self.out_queue.put({
                    "type": "scene_ready",
                    "seq_id": msg.seq_id,
                    "mesh_data": mesh_data,
                    "timestamp": time.time(),
                })
            except Exception as e:
                print(f"[SceneWorker] Error: {e}", flush=True)
    
    def stop(self):
        self.running = False


class RayTracerWorker(threading.Thread):
    """Background thread: runs C++ ray tracer, accumulates results."""
    
    def __init__(self, in_queue: Queue, out_queue: Queue, scene_result_queue: Queue):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.scene_result_queue = scene_result_queue
        self.running = True
        self.paused = False
        self.current_mesh_data: Optional[dict] = None
        self.tracer: Optional[Any] = None
        self.sensor_gid: int = -1
        
    def run(self):
        seed = 0
        while self.running:
            # Poll for scene updates (non-blocking)
            try:
                scene_msg = self.scene_result_queue.get_nowait()
                if scene_msg.get("type") == "scene_ready":
                    self.current_mesh_data = scene_msg.get("mesh_data")
                    self.tracer = None  # Force rebuild
                    self.sensor_gid = -1
                    print(f"[RayTracerWorker] New scene ready, seq={scene_msg.get('seq_id')}", flush=True)
            except Empty:
                pass
            
            # Poll for ray trace requests
            try:
                msg: RayTracerMessage = self.in_queue.get(timeout=0.2)
            except Empty:
                continue
            
            if self.paused or self.current_mesh_data is None:
                continue
            
            try:
                # Lazy-initialize tracer from mesh data
                if self.tracer is None and self.current_mesh_data:
                    self._initialize_tracer()
                
                if self.tracer is None:
                    continue
                
                # ─── C++ API: BDPT ray tracing ───────────────────────────────
                # Careful, skillful use of the C++ RayTracer interface.
                n_rays = int(max(64, msg.n_samples))
                max_bounces = 12
                min_amplitude = 1.0e-7
                max_records = 1_000_000
                
                print(f"[RayTracerWorker] Running BDPT: {n_rays} rays, seq={msg.seq_id}", flush=True)
                
                # Call C++ bidirectional ray tracer
                records = self.tracer.bidirectional(
                    n_rays,
                    max_bounces,
                    min_amplitude,
                    int(seed % (2**31)),
                    max_records,
                )
                seed += 1
                
                # ─── C++ API: Reduce to RGB ─────────────────────────────────
                # Extract pixel grid size from image plate config
                n_px = int(self.current_mesh_data["scene"].image_plate.pixels)
                
                reduced = self.tracer.reduce_endpoint_records_to_rgb_image(
                    records,
                    n_px,
                    n_px,
                    int(self.sensor_gid),
                    1.0,      # exposure scale
                    100.0,    # tone map exponent
                )
                
                rgb_tonemapped = reduced.get("rgb_tonemapped") if isinstance(reduced, dict) else None
                telemetry = reduced.get("telemetry", {}) if isinstance(reduced, dict) else {}
                
                if rgb_tonemapped is not None:
                    rgb_array = np.asarray(rgb_tonemapped, dtype=np.float32)
                    print(f"[RayTracerWorker] RGB shape: {rgb_array.shape}, records: {int(np.asarray(records).shape[0]) if np.asarray(records).ndim == 2 else 0}", flush=True)
                    
                    self.out_queue.put({
                        "type": "radiance_update",
                        "seq_id": msg.seq_id,
                        "rgb": rgb_array,
                        "radiance_count": msg.n_samples,
                        "telemetry": telemetry,
                        "timestamp": time.time(),
                    })
            
            except Exception as e:
                print(f"[RayTracerWorker] Error: {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    def _initialize_tracer(self):
        """Create _sk.RayTracer from mesh data and register tri groups."""
        if self.current_mesh_data is None:
            return
        
        mesh = self.current_mesh_data
        scene = mesh["scene"]
        tri_arr = mesh["tri_arr"]
        verts = mesh["verts"]
        normals = mesh["normals"]
        mat_idx = mesh["mat_idx"]
        db = mesh["db"]
        source_ids = mesh["source_ids"]
        image_plate_ids = mesh["image_plate_ids"]
        aperture_stop_ids = mesh["aperture_stop_ids"]
        lens_surface_groups = mesh.get("lens_surface_groups", [])
        freq_hz = mesh["freq_hz"]
        
        # ─── C++ API: Create RayTracer ──────────────────────────────────
        mat_buf = db.build_mat_buf(freq_hz=freq_hz).astype(np.float32, copy=False)
        mat_n_mats = int(mat_buf.shape[0] // MAX_SPECTRAL_BANDS)
        
        self.tracer = _sk.RayTracer(
            n_tri=int(tri_arr.shape[0]),
            verts=np.ascontiguousarray(verts, dtype=np.float64),
            normals=np.ascontiguousarray(normals, dtype=np.float64),
            mat_idx=np.ascontiguousarray(mat_idx, dtype=np.int32),
            mat_buf=np.ascontiguousarray(mat_buf, dtype=np.float32),
            mat_n_mats=int(mat_n_mats),
            freq_hz=np.ascontiguousarray(freq_hz, dtype=np.float64),
            speed_m_s=float(C_LIGHT),
            atmo_abs=np.zeros_like(freq_hz, dtype=np.float64),
        )
        
        # ─── C++ API: Register tri groups ──────────────────────────────
        role_emissive = int(getattr(_sk, "TRI_GROUP_ROLE_EMISSIVE", 1))
        role_sensor = int(getattr(_sk, "TRI_GROUP_ROLE_SENSOR", 2))
        role_blocker = int(getattr(_sk, "TRI_GROUP_ROLE_BLOCKER", 4))
        sample_area = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA", 1))
        sample_pixel_cone = int(getattr(_sk, "TRI_GROUP_SAMPLE_PIXEL_CONE", 3))
        
        self.tracer.clear_tri_groups()
        
        # Register source (emissive triangles)
        source_gid = int(self.tracer.register_tri_group(
            role_emissive,
            sample_area,
            np.ascontiguousarray(source_ids, dtype=np.int32),
        ))
        
        # Register aperture stop (blocker)
        aperture_gid = -1
        if int(aperture_stop_ids.size) > 0:
            aperture_gid = int(self.tracer.register_tri_group(
                role_blocker,
                sample_area,
                np.ascontiguousarray(aperture_stop_ids, dtype=np.int32),
            ))

        optics = mesh.get("optics")
        if optics is not None and lens_surface_groups:
            assembly = LensAssemblySpec()
            assembly.set_optics(optics, mode=LensAssemblySpec.MODE_PARAMETRIC)
            assembly.register(
                self.tracer,
                lens_surface_groups,
                tri_arr,
                np.ascontiguousarray(np.mean(tri_arr, axis=1), dtype=np.float64),
                _scene_lenses(scene),
            )
            mesh["lens_assembly"] = assembly
        
        # Register sensor (image plate) with camera descriptor
        plate = scene.image_plate
        lenses = _scene_lenses(scene)
        first_lens = lenses[0] if lenses else None
        last_lens = lenses[-1] if lenses else None
        lens_center_x = 0.5 * (first_lens.x_front + last_lens.x_back) if first_lens and last_lens else plate.x - 0.5
        stop_plane_x = last_lens.center_x if last_lens else plate.x - 0.5
        stop_radius = float(getattr(scene, "exit_pupil_radius", 0.008))
        
        self.sensor_gid = int(self.tracer.register_tri_group(
            role_sensor,
            sample_pixel_cone,
            np.ascontiguousarray(image_plate_ids, dtype=np.int32),
            sensor_camera={
                "pos": np.array([plate.x, 0.0, 0.0], dtype=np.float64),
                "fwd": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
                "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
                "sensor_w_m": float(2.0 * plate.radius),
                "sensor_h_m": float(2.0 * plate.radius),
                "focal_m": float(max(0.05, plate.x - stop_plane_x)),
                "aperture_radius_m": float(stop_radius),
                "n_px": int(plate.pixels),
                "n_py": int(plate.pixels),
                "n_aperture_samples": int(max(1, plate.pixels // 8)),
                "aperture_stop_group_id": int(aperture_gid),
                "effective_focal_m": float(max(1.0e-3, plate.x - lens_center_x)),
                "focus_distance_m": float(max(1.0e-3, lens_center_x - scene.object_plane.x)),
                "lens_center": np.array([lens_center_x, 0.0, 0.0], dtype=np.float64),
                "lens_fwd": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
                "camera_mode": int(2),
            },
        ))
        
        print(f"[RayTracerWorker] Tracer initialized: {tri_arr.shape[0]} tris, {len(freq_hz)} bands, source_gid={source_gid}, sensor_gid={self.sensor_gid}", flush=True)
    
    def set_paused(self, paused: bool):
        self.paused = paused
    
    def stop(self):
        self.running = False


class ThickLensInteractiveViewer:
    """Main interactive lens/aperture viewer."""
    
    def __init__(self, width: int = 1600, height: int = 900):
        pygame.init()
        self.width = width
        self.height = height
        self.display = (width, height)
        
        # OpenGL + pygame setup
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Thick Lens Interactive Viewer")
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Scene state
        self.params = SceneParameters()
        self.display_state = DisplayState()
        
        # Worker threads
        self.scene_update_queue = Queue()
        self.scene_result_queue = Queue()
        self.ray_trace_queue = Queue()
        self.ray_trace_result_queue = Queue()
        
        self.scene_worker = SceneUpdateWorker(self.scene_update_queue, self.scene_result_queue)
        self.ray_worker = RayTracerWorker(self.ray_trace_queue, self.ray_trace_result_queue, self.scene_result_queue)
        self.scene_worker.start()
        self.ray_worker.start()
        
        # Mouse state
        self.mouse_mode = MouseMode.IDLE
        self.mouse_start_pos: Tuple[float, float] = (0.0, 0.0)
        self.mouse_current_pos: Tuple[float, float] = (0.0, 0.0)
        self.mouse_dragged_lens_idx: int = -1
        self.mouse_modifier = None  # shift, ctrl, alt, none
        
        # Timing
        self.clock = pygame.time.Clock()
        self.fps_target = 60
        self.frame_count = 0
        self.seq_id = 0
        
        # Timers
        self.last_scene_update = 0.0
        self.last_ray_trace = 0.0
        
    def handle_events(self):
        """Process all pending events; never block."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_SPACE:
                    self.ray_worker.set_paused(not self.ray_worker.paused)
                elif event.key == K_c:
                    self.display_state.radiance_rgb = None
                    self.display_state.accumulated_count = 0
                elif event.key == K_e:
                    self.params.show_emitter = not self.params.show_emitter
                    self.queue_scene_update()
                elif event.key == K_a:
                    self.params.show_aperture = not self.params.show_aperture
                    self.queue_scene_update()
                elif event.key == K_l:
                    self.params.show_lenses = not self.params.show_lenses
                    self.queue_scene_update()
                elif event.key == K_o:
                    self.params.use_optical_design = not self.params.use_optical_design
                    self.queue_scene_update()
                elif event.key == K_w:
                    self.params.aperture_model = (
                        "wave3d" if self.params.aperture_model != "wave3d" else "geometry"
                    )
                    self.queue_scene_update()
                elif event.key == K_r:
                    self.params = SceneParameters()
                    self.queue_scene_update()
            
            elif event.type == MOUSEBUTTONDOWN:
                self.mouse_start_pos = pygame.mouse.get_pos()
                self.mouse_current_pos = self.mouse_start_pos
                if event.button == 1:  # Left click
                    self.mouse_mode = MouseMode.DRAG_LENS
                    # TODO: detect which lens was clicked
                    self.mouse_dragged_lens_idx = 0
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_mode = MouseMode.IDLE
                    self.mouse_dragged_lens_idx = -1
            
            elif event.type == MOUSEMOTION:
                self.mouse_current_pos = pygame.mouse.get_pos()
                self.handle_mouse_drag()
            
            elif event.type == MOUSEWHEEL:
                mods = pygame.key.get_mods()
                if self.params.use_optical_design and mods & (K_LSHIFT | K_RSHIFT):
                    self.params.f_number = max(0.5, self.params.f_number + event.y * 0.1)
                elif self.params.use_optical_design and mods & (K_LCTRL | K_RCTRL):
                    self.params.focus_distance_m = max(0.05, self.params.focus_distance_m + event.y * 0.05)
                elif self.params.use_optical_design and mods & (K_LALT | K_RALT):
                    self.params.focal_max_m = max(
                        self.params.focal_min_m + 0.005,
                        self.params.focal_max_m + event.y * 0.005,
                    )
                elif mods & (K_LSHIFT | K_RSHIFT):
                    # Adjust aperture radius
                    delta = event.y * 0.001
                    self.params.aperture_radius = max(0.001, self.params.aperture_radius + delta)
                elif mods & (K_LCTRL | K_RCTRL):
                    # Adjust lens thickness
                    if self.mouse_dragged_lens_idx >= 0 and self.mouse_dragged_lens_idx < len(self.params.lens_thickness_list):
                        delta = event.y * 0.001
                        self.params.lens_thickness_list[self.mouse_dragged_lens_idx] = max(0.001, 
                            self.params.lens_thickness_list[self.mouse_dragged_lens_idx] + delta)
                elif mods & (K_LALT | K_RALT):
                    # Adjust lens curvature
                    if self.mouse_dragged_lens_idx >= 0 and self.mouse_dragged_lens_idx < len(self.params.lens_curvature_list):
                        delta = event.y * 0.001
                        self.params.lens_curvature_list[self.mouse_dragged_lens_idx] = max(0.001,
                            self.params.lens_curvature_list[self.mouse_dragged_lens_idx] + delta)
                else:
                    # Adjust emitter radius
                    delta = event.y * 0.001
                    self.params.emitter_radius = max(0.001, self.params.emitter_radius + delta)
                
                self.queue_scene_update()
        
        return True
    
    def handle_mouse_drag(self):
        """Handle ongoing mouse drag for lens positioning."""
        if self.mouse_mode == MouseMode.IDLE:
            return
        
        dx = self.mouse_current_pos[0] - self.mouse_start_pos[0]
        
        if self.mouse_mode == MouseMode.DRAG_LENS:
            if self.params.use_optical_design:
                self.params.zoom = float(np.clip(self.params.zoom + dx * 0.0008, 0.0, 1.0))
                self.queue_scene_update()
                self.mouse_start_pos = self.mouse_current_pos
            elif self.mouse_dragged_lens_idx >= 0 and self.mouse_dragged_lens_idx < len(self.params.lens_x_list):
                # Map screen x-drag to scene x-axis change
                # ~100 pixels = 0.01 m along optical axis
                scene_dx = dx * 0.0001
                self.params.lens_x_list[self.mouse_dragged_lens_idx] += scene_dx
                self.queue_scene_update()
                self.mouse_start_pos = self.mouse_current_pos
    
    def queue_scene_update(self):
        """Request a scene rebuild."""
        self.seq_id += 1
        self.scene_update_queue.put(SceneUpdateMessage(
            params=self.params.copy(),
            seq_id=self.seq_id,
        ))
    
    def queue_ray_trace(self):
        """Request a ray tracing batch."""
        self.ray_trace_queue.put(RayTracerMessage(
            seq_id=self.seq_id,
            n_samples=512,
            timeout_sec=2.0,
        ))
    
    def poll_workers(self):
        """Check for completed work from background threads (non-blocking)."""
        # Drain scene result queue
        while True:
            try:
                result = self.scene_result_queue.get_nowait()
                print(f"[Main] Scene ready: seq={result.get('seq_id')}", flush=True)
                # Trigger new ray trace batch
                self.queue_ray_trace()
            except Empty:
                break
        
        # Drain ray trace result queue and accumulate
        while True:
            try:
                result = self.ray_trace_result_queue.get_nowait()
                rgb_batch = result.get("rgb")
                if rgb_batch is not None:
                    # Accumulate RGB with exponential moving average (leak=0.95)
                    leak = 0.95
                    if self.display_state.radiance_rgb is None:
                        self.display_state.radiance_rgb = np.asarray(rgb_batch, dtype=np.float32)
                    else:
                        self.display_state.radiance_rgb = (
                            leak * self.display_state.radiance_rgb +
                            (1.0 - leak) * np.asarray(rgb_batch, dtype=np.float32)
                        )
                    self.display_state.accumulated_count += result.get("radiance_count", 0)
                    print(f"[Main] Radiance accumulated: count={self.display_state.accumulated_count}", flush=True)
            except Empty:
                break
    
    def update(self):
        """Update simulation/accumulation state (non-blocking, per-frame)."""
        self.poll_workers()
        
        # Request new ray tracing batch if workers are idle
        now = time.time()
        if now - self.last_ray_trace > 0.1 and not self.ray_worker.paused:
            self.queue_ray_trace()
            self.last_ray_trace = now
    
    def render(self):
        """Render viewport (spectral radiance + parameters)."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # TODO: render radiance to left panel
        if self.frame_count % 120 == 0:
            print(
                "[interactive-optics]",
                f"mode={'design' if self.params.use_optical_design else 'manual'}",
                f"zoom={self.params.zoom:.3f}",
                f"focus={self.params.focus_distance_m:.2f}m",
                f"f/{self.params.f_number:.2f}",
                f"focal=({self.params.focal_min_m*1e3:.0f},{self.params.focal_max_m*1e3:.0f})mm",
                f"aperture_model={self.params.aperture_model}",
                flush=True,
            )
        
        pygame.display.flip()
    
    def run(self):
        """Main loop: event → update → render, all non-blocking."""
        running = True
        while running:
            # Event loop (never blocks on input)
            running = self.handle_events()
            
            # Update state
            self.update()
            
            # Render
            self.render()
            
            # Maintain frame rate
            self.clock.tick(self.fps_target)
            self.frame_count += 1
            
            if self.frame_count % 60 == 0:
                print(f"[Main] FPS: {self.clock.get_fps():.1f}")
        
        # Cleanup
        self.scene_worker.stop()
        self.ray_worker.stop()
        self.scene_worker.join(timeout=2.0)
        self.ray_worker.join(timeout=2.0)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Interactive Thick Lens Viewer")
    parser.add_argument("--width", type=int, default=1600, help="Window width")
    parser.add_argument("--height", type=int, default=900, help="Window height")
    args = parser.parse_args()
    
    viewer = ThickLensInteractiveViewer(args.width, args.height)
    viewer.run()


if __name__ == "__main__":
    main()
