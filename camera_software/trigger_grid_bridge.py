"""Headless HTTP-image adapter for the existing Pygame camera routing grid.

The RoutingGridView owns hit testing, tabs, scrolling and graph edits. A browser
only forwards pointer events and displays the returned PNG; it never implements
a second mixer or changes the graph schema.
"""
from __future__ import annotations

import base64
from io import BytesIO
import os
from threading import RLock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from PIL import Image

from camera_software.trigger_bus import CAMERA_TRIGGER_PORTS, make_camera_trigger_graph, make_camera_trigger_panel, validate_flash_illumination_graph
from camera_software.trigger_bus import resolve_camera_trigger_bus
from camera_software.exposure_timing import ExposureFrame, ExposureSlice
from routing_engine import RoutingGraph


class CameraTriggerGrid:
    WIDTH = 960
    HEIGHT = 620

    def __init__(self, receivers, buses=(), saved=None):
        self.lock = RLock()
        self.receivers = tuple(receivers)
        self.buses = tuple(buses)
        expected = (*CAMERA_TRIGGER_PORTS, *self.buses, *self.receivers)
        if len(expected) > 160:
            raise ValueError("camera routing grid exceeds 160 named nodes")
        graph = make_camera_trigger_graph(self.receivers, bus_keys=self.buses)
        if saved is not None:
            restored = RoutingGraph.from_dict(saved)
            if tuple(restored.node_keys()) != expected:
                raise ValueError("saved camera routing nodes do not match the current light bus")
            graph = restored
        self.graph = graph
        self.patch, self.view = make_camera_trigger_panel(graph)
        pygame.display.init()
        pygame.font.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1))

    def frame(self):
        with self.lock:
            surface = self.view.render_surface(self.patch, self.WIDTH, self.HEIGHT)
            pixels = pygame.image.tobytes(surface, "RGB")
            output = BytesIO()
            Image.frombytes("RGB", (self.WIDTH, self.HEIGHT), pixels).save(output, format="PNG")
            return {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
                    "graph": self.graph.to_dict(), "width": self.WIDTH, "height": self.HEIGHT}

    def event(self, name, x=0, y=0, *, button=1, wheel_y=0):
        types = {"down": pygame.MOUSEBUTTONDOWN, "up": pygame.MOUSEBUTTONUP,
                 "move": pygame.MOUSEMOTION, "wheel": pygame.MOUSEWHEEL}
        if name not in types or button not in (1, 3):
            raise ValueError("unsupported camera routing pointer event")
        x, y = int(x), int(y)
        if not 0 <= x < self.WIDTH or not 0 <= y < self.HEIGHT or not -20 <= int(wheel_y) <= 20:
            raise ValueError("camera routing pointer is outside its bounded panel")
        with self.lock:
            self.view.render_surface(self.patch, self.WIDTH, self.HEIGHT)
            event = pygame.event.Event(types[name], pos=(x, y), button=button, y=int(wheel_y))
            self.view.handle_event(event, self.patch, 0, 0)
            return self.frame()

    def pulse(self):
        """Resolve one manual 10 ms camera pulse on a 1 ms Studio preview clock."""
        with self.lock:
            maximum_delay = validate_flash_illumination_graph(self.graph)
            duration_ms = max(250, int(maximum_delay * 1000) + 50)
            if duration_ms > 5000:
                raise ValueError("manual trigger path exceeds the 5-second Phong timing window")
            slices = tuple(ExposureSlice(frame_id=0,slice_id=i,t0=i/1000,t1=(i+1)/1000,
                shutter_mode=0,shutter_open=1,shutter_center_u=.5,shutter_center_v=.5,
                shutter_softness=0,exposure_weight=.001,flash_weight=.1 if 1 <= i < 11 else 0,
                sensor_weight=.001 if i>=1 else 0,flash_active=1 <= i < 11,
                sensor_integrating=i>=1,camera_id="studio-manual-flash",causality_id=i)
                for i in range(duration_ms))
            frame=ExposureFrame(frame_id=0,t0=0,t1=duration_ms/1000,slices=slices,camera_id="studio-manual-flash")
            result=resolve_camera_trigger_bus(frame,self.graph,self.receivers,flash_signal_mode="level")
            return {"step_ms":1,"duration_ms":duration_ms,"shop_cutoff_end_ms":11,
                    "signals":{key:[abs(value) for value in result.signals[key]] for key in self.receivers},
                    "graph":self.graph.to_dict()}
