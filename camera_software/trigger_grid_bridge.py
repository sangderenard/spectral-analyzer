"""Headless HTTP-image adapter for the existing Pygame camera routing grid.

The RoutingGridView owns hit testing, tabs, scrolling and graph edits. A browser
only forwards pointer events and displays the returned PNG; it never implements
a second mixer or changes the graph schema.
"""
from __future__ import annotations

import base64
import math
import json
import re
import torch
from io import BytesIO
from threading import RLock

import pygame
from PIL import Image

from camera_software.trigger_bus import CAMERA_TRIGGER_PORTS, make_camera_trigger_graph, make_camera_trigger_panel, validate_flash_illumination_graph
from camera_software.trigger_bus import resolve_camera_trigger_bus
from camera_software.exposure_timing import ExposureFrame, ExposureSlice
from camera_software.trigger_generators import GENERATOR_KEYS, generator_tail_s, validate_generators
from routing_engine import RoutingGraph, TensorPortContract, EdgeTransferSpec
from graph_solver import TensorNode
from network_materializer import compile_nodes, routing_edge_to_tensor_edge


class CameraTriggerGrid:
    WIDTH = 960
    HEIGHT = 620

    def __init__(self, receivers, buses=(), saved=None, generators=None):
        self.lock = RLock()
        self._radio_runtime = None
        self._radio_signature = None
        self.receivers = tuple(receivers)
        self.buses = tuple(buses)
        self.generators = validate_generators(generators)
        radio_keys = tuple(key for key in (saved or {}).get("nodes", [])
                           if isinstance(key, str) and (key == "radio:rms" or re.fullmatch(r"radio:band:([1-9]|[1-9][0-9]|1[01][0-9]|12[0-8])", key)))
        expected = (*CAMERA_TRIGGER_PORTS, *self.buses, *self.receivers, *GENERATOR_KEYS, *radio_keys)
        if len(expected) > 200:
            raise ValueError("camera routing grid exceeds 200 named nodes")
        graph = make_camera_trigger_graph(self.receivers, bus_keys=self.buses, generator_keys=GENERATOR_KEYS)
        if saved is not None:
            restored = RoutingGraph.from_dict(saved)
            if not set(restored.node_keys()) <= set(expected):
                raise ValueError("saved camera routing nodes do not match the current light bus")
            for key in expected:
                if key not in restored.node_keys():
                    if not (key.startswith("filter:studio:") or key in GENERATOR_KEYS):
                        raise ValueError("saved camera routing nodes do not match the current light bus")
                    restored.add_node(key)
            graph = restored
        self.graph = graph
        self.patch, self.view = make_camera_trigger_panel(graph)
        # The grid draws into CPU Surfaces, not a display. Never select an SDL
        # video driver here: this module is imported by the service, whose
        # later OpenGL render workers inherit its environment.
        pygame.font.init()

    def frame(self):
        with self.lock:
            surface = self.view.render_surface(self.patch, self.WIDTH, self.HEIGHT)
            pixels = pygame.image.tobytes(surface, "RGB")
            output = BytesIO()
            Image.frombytes("RGB", (self.WIDTH, self.HEIGHT), pixels).save(output, format="PNG")
            return {"image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
                    "graph": self.graph.to_dict(), "width": self.WIDTH, "height": self.HEIGHT}

    def event(self, name, x=0, y=0, *, button=1, wheel_y=0, shift=False):
        types = {"down": pygame.MOUSEBUTTONDOWN, "up": pygame.MOUSEBUTTONUP,
                 "move": pygame.MOUSEMOTION, "wheel": pygame.MOUSEWHEEL}
        if name not in types or button not in (1, 3) or type(shift) is not bool:
            raise ValueError("unsupported camera routing pointer event")
        x, y = int(x), int(y)
        if not 0 <= x < self.WIDTH or not 0 <= y < self.HEIGHT or not -20 <= int(wheel_y) <= 20:
            raise ValueError("camera routing pointer is outside its bounded panel")
        with self.lock:
            self.view.render_surface(self.patch, self.WIDTH, self.HEIGHT)
            event = pygame.event.Event(types[name], pos=(x, y), button=button, y=int(wheel_y),
                                       mod=pygame.KMOD_SHIFT if shift else 0)
            self.view.handle_event(event, self.patch, 0, 0)
            return self.frame()

    def pulse(self):
        """Resolve one manual 10 ms camera pulse on a 1 ms Studio preview clock."""
        with self.lock:
            maximum_delay = validate_flash_illumination_graph(self.graph)
            duration_ms = max(250, int((maximum_delay+generator_tail_s(self.generators)) * 1000) + 50)
            if duration_ms > 5000:
                raise ValueError("manual trigger path exceeds the 5-second Phong timing window")
            slices = tuple(ExposureSlice(frame_id=0,slice_id=i,t0=i/1000,t1=(i+1)/1000,
                shutter_mode=0,shutter_open=1,shutter_center_u=.5,shutter_center_v=.5,
                shutter_softness=0,exposure_weight=.001,flash_weight=.1 if 1 <= i < 11 else 0,
                sensor_weight=.001 if i>=1 else 0,flash_active=1 <= i < 11,
                sensor_integrating=i>=1,camera_id="studio-manual-flash",causality_id=i)
                for i in range(duration_ms))
            frame=ExposureFrame(frame_id=0,t0=0,t1=duration_ms/1000,slices=slices,camera_id="studio-manual-flash")
            result=resolve_camera_trigger_bus(frame,self.graph,self.receivers,flash_signal_mode="level",generators=self.generators)
            return {"step_ms":1,"duration_ms":duration_ms,"shop_cutoff_end_ms":11,
                    "signals":{key:[abs(value) for value in result.signals[key]] for key in self.receivers},
                    "graph":self.graph.to_dict()}

    def radio(self, snapshot):
        """Patch authenticated Nodus telemetry into the original routing graph.

        NRAD is a sampled level feed, not PCM. No FFT, beat inference, or new
        light source is introduced here. Timing-sensitive delayed wires need a
        timestamped signal stream; refusing them is preferable to pretending
        browser polling is an audio clock.
        """
        with self.lock:
            if snapshot.get("status") != "live":
                self._radio_runtime = None
                self._radio_signature = None
                return {"status": snapshot.get("status", "unavailable"), "active": [], "gains": {}}
            bands = snapshot.get("bands")
            rms = snapshot.get("rms")
            if not isinstance(bands, list) or not 1 <= len(bands) <= 128:
                raise ValueError("radio requires 1..128 measured spectral bands")
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in [rms, *bands]):
                raise ValueError("radio requires finite nonnegative measured levels")
            inputs = {"radio:rms": rms, **{f"radio:band:{i+1}": value for i, value in enumerate(bands)}}
            existing = {key for key in self.graph.node_keys() if key.startswith("radio:")}
            if existing and existing != set(inputs):
                raise ValueError("radio band layout changed; reset the routing graph before reconnecting")
            if len(set(self.graph.node_keys()) | set(inputs)) > 200:
                raise ValueError("camera routing grid exceeds 200 named nodes")
            added = False
            for key in inputs:
                if key not in self.graph.node_keys():
                    if len(self.graph.node_keys()) >= 200:
                        raise ValueError("camera routing grid exceeds 200 named nodes")
                    self.graph.add_node(key)
                    added = True
            sources = {key for key in self.graph.node_keys() if key.startswith("radio:")}
            if sources != set(inputs):
                raise ValueError("radio band layout changed; reset the routing graph before reconnecting")
            validate_flash_illumination_graph(self.graph)
            for edge in self.graph.edges:
                if edge.dst_key in sources:
                    raise ValueError("radio measurement source ports cannot receive patched signals")
                if edge.delay_s or getattr(edge, "delay_samples", 0):
                    raise ValueError("sampled radio preview requires zero-delay wires; use the timestamped audio bus for delays")
                if not math.isfinite(edge.weight) or not math.isfinite(edge.angle_rad):
                    raise ValueError("radio routing gains and phase must be finite")
                if edge.tensor_contract != TensorPortContract() or edge.transfer_spec != EdgeTransferSpec():
                    raise ValueError("radio level preview requires scalar identity transport")
                if edge.saturation or edge.edge_kind not in ("signal", "control"):
                    raise ValueError("radio level preview requires linear signal/control edges")
            if self.graph.param_edges or self.graph.meta_edges or self.graph.feedback.enabled or self.graph.latency_compensation:
                raise ValueError("radio level preview requires the linear routing graph")
            signature = json.dumps(self.graph.to_dict(), sort_keys=True)
            if signature != self._radio_signature:
                self._radio_runtime = compile_nodes(
                    [TensorNode(key) for key in self.graph.node_keys()],
                    [routing_edge_to_tensor_edge(edge) for edge in self.graph.edges],
                    sample_rate=1, output_node_keys=list(self.receivers))
                self._radio_signature = signature
            resolved = self._radio_runtime.process_window(
                {key: torch.tensor([value], dtype=torch.complex128) for key, value in inputs.items()}, n_frames=1)
            if any(not torch.isfinite(value).all() for value in resolved.values()):
                raise ValueError("radio routing resolved a non-finite signal")
            # Reachability only selects overridden receivers. GraphSolver owns
            # the actual transfer, including chained buses and complex phase.
            reached = set(sources)
            while True:
                more = {edge.dst_key for edge in self.graph.edges if edge.src_key in reached and abs(edge.weight) > 1e-12}
                if more <= reached: break
                reached.update(more)
            result = {"status": "live", "active": [key for key in self.receivers if key in reached],
                      "gains": {key: abs(complex(resolved[key].reshape(-1)[0].item())) for key in self.receivers},
                      "frame": snapshot.get("frame"), "transport_contract": self._radio_runtime.transport_contract()}
            if added: result.update(self.frame())
            return result
