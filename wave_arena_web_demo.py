"""Local browser host for the existing native T4 angular-spectrum transform.

The host owns the evolving field buffers. Every propagation step is performed
by RayTracer.t4_angular_spectrum_step, as in wave_transform_visual_demo.py.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from camera_software.physical_aperture import LivePhysicalAperture

from wave_transform_visual_demo import (
    _calibration_tracer,
    _phase_rgba,
    _scalar_rgba,
    _source_field,
)


PAGE = Path(__file__).with_name("wave_arena_web_demo.html")
DEFAULTS = {
    "pattern": "pair",
    "size": 128,
    "boundary": "periodic",
    "absorber_cells": 12,
    "absorber_strength": 8.0,
    "aperture_shape": "none",
    "aperture_opening_fraction": 0.2,
    "aperture_blades": 6,
    "aperture_thickness_um": 20.0,
    "wavelength_nm": 532.0,
    "pitch_um": 1.5,
    "step_um": 25.0,
    "travel_steps": 80,
    "waist_fraction": 0.075,
    "separation_fraction": 0.12,
    "tilt_fraction": 0.28,
    "relative_amplitude": 0.82,
    "relative_phase": 0.72,
}


def _bounded(value: object, key: str, lower: float, upper: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{key} must be in [{lower}, {upper}]")
    return number


def validate_config(payload: dict[str, object]) -> dict[str, float | int | str]:
    if not isinstance(payload, dict) or set(payload) - set(DEFAULTS):
        raise ValueError("unknown arena parameter")
    merged = {**DEFAULTS, **payload}
    pattern = merged["pattern"]
    if pattern not in ("pair", "slits", "ring"):
        raise ValueError("pattern must be pair, slits, or ring")
    size = int(merged["size"])
    if size not in (64, 128, 256, 512):
        raise ValueError("size must be 64, 128, 256, or 512")
    boundary = merged["boundary"]
    if boundary not in ("periodic", "absorbing"):
        raise ValueError("boundary must be periodic or absorbing")
    aperture_shape = merged["aperture_shape"]
    if aperture_shape not in ("none", "iris", "circular"):
        raise ValueError("aperture_shape must be none, iris, or circular")
    blades = int(merged["aperture_blades"])
    if not 3 <= blades <= 16:
        raise ValueError("aperture_blades must be in [3, 16]")
    absorber_cells = int(merged["absorber_cells"])
    if not 0 <= absorber_cells <= size // 4:
        raise ValueError("absorber_cells must fit inside one quarter of the grid")
    travel = int(merged["travel_steps"])
    if travel < 1 or travel > 400:
        raise ValueError("travel_steps must be in [1, 400]")
    return {
        "pattern": pattern,
        "size": size,
        "boundary": boundary,
        "absorber_cells": absorber_cells,
        "absorber_strength": _bounded(merged["absorber_strength"], "absorber_strength", 0, 32),
        "aperture_shape": aperture_shape,
        "aperture_opening_fraction": _bounded(merged["aperture_opening_fraction"], "aperture_opening_fraction", 0.05, 0.4),
        "aperture_blades": blades,
        "aperture_thickness_um": _bounded(merged["aperture_thickness_um"], "aperture_thickness_um", 1, 100),
        "wavelength_nm": _bounded(merged["wavelength_nm"], "wavelength_nm", 380, 900),
        "pitch_um": _bounded(merged["pitch_um"], "pitch_um", 0.1, 5),
        "step_um": _bounded(merged["step_um"], "step_um", 1, 100),
        "travel_steps": travel,
        "waist_fraction": _bounded(merged["waist_fraction"], "waist_fraction", 0.02, 0.25),
        "separation_fraction": _bounded(merged["separation_fraction"], "separation_fraction", 0, 0.3),
        "tilt_fraction": _bounded(merged["tilt_fraction"], "tilt_fraction", 0.1, 1.0),
        "relative_amplitude": _bounded(merged["relative_amplitude"], "relative_amplitude", 0, 1.5),
        "relative_phase": _bounded(merged["relative_phase"], "relative_phase", -math.pi, math.pi),
    }


def _initial_field(config: dict[str, float | int | str], size: int, pitch: float) -> np.ndarray:
    parameters = {
        "waist_fraction": float(config["waist_fraction"]),
        "separation_fraction": float(config["separation_fraction"]),
        "tilt_fraction": float(config["tilt_fraction"]),
        "relative_amplitude": float(config["relative_amplitude"]),
        "relative_phase": float(config["relative_phase"]),
    }
    if config["pattern"] == "pair":
        return _source_field(size, pitch, **parameters)
    coordinate = (np.arange(size, dtype=np.float64) - (size - 1) * 0.5) * pitch
    x, y = np.meshgrid(coordinate, coordinate)
    extent = size * pitch
    waist = extent * parameters["waist_fraction"]
    separation = extent * parameters["separation_fraction"]
    tilt = 2.0 * np.pi / (extent * parameters["tilt_fraction"])
    phase = parameters["relative_phase"]
    strength = parameters["relative_amplitude"]
    if config["pattern"] == "slits":
        slit_width = max(pitch, waist * 0.45)
        envelope = np.exp(-(y / (extent * 0.23)) ** 2)
        left = (np.abs(x + separation) <= slit_width * 0.5) * envelope
        right = strength * (np.abs(x - separation) <= slit_width * 0.5) * envelope
        return left * np.exp(1j * tilt * x) + right * np.exp(1j * (phase - 0.7 * tilt * y))
    radius = np.hypot(x, y)
    annulus = np.exp(-((radius - max(separation, pitch)) / waist) ** 2)
    center = strength * np.exp(-(radius / max(waist * 0.8, pitch)) ** 2)
    return annulus * np.exp(1j * tilt * x) + center * np.exp(1j * phase)


def _png_data_url(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class WaveArena:
    def __init__(self, config: dict[str, object] | None = None) -> None:
        self.configure(config or {})

    def configure(self, payload: dict[str, object]) -> None:
        config = validate_config(payload)
        size = int(config["size"])
        wavelength = float(config["wavelength_nm"]) * 1e-9
        pitch = float(config["pitch_um"]) * 1e-6
        initial = _initial_field(config, size, pitch)
        self.config = config
        self.tracer = _calibration_tracer(wavelength)
        self.wavelengths = np.asarray([wavelength], np.float64)
        self.re = np.ascontiguousarray(initial.real[None], np.float32)
        self.im = np.ascontiguousarray(initial.imag[None], np.float32)
        if config["aperture_shape"] != "none":
            extent = size * pitch
            opening = extent * float(config["aperture_opening_fraction"])
            parameters = dict(
                key="arena-aperture", opening_radius_m=opening,
                assembly_radius_m=extent * 0.45,
                thickness_m=float(config["aperture_thickness_um"]) * 1e-6,
                material_n_real=2.9, material_n_imag=3.0,
            )
            aperture = (
                LivePhysicalAperture.iris(blade_count=int(config["aperture_blades"]), **parameters)
                if config["aperture_shape"] == "iris" else
                LivePhysicalAperture.circular_hole(display_segments=32, **parameters)
            )
            self.tracer.t4_apply_aperture_material(
                1, size, size, pitch, aperture.thickness_m, 1,
                self.wavelengths, aperture.wave_payload(), self.re, self.im,
            )
        self.initial = self.re[0].astype(np.float64) + 1j * self.im[0].astype(np.float64)
        self.initial_peak = float(np.max(np.abs(self.initial) ** 2))
        self.position = 0
        self.direction = 1
        self.generation = 0

    def step(self, count: int = 1) -> None:
        if not 1 <= count <= 8:
            raise ValueError("count must be in [1, 8]")
        for _ in range(count):
            limit = int(self.config["travel_steps"])
            if self.position >= limit:
                self.direction = -1
            elif self.position <= 0:
                self.direction = 1
            self.tracer.t4_angular_spectrum_step(
                1, int(self.config["size"]), int(self.config["size"]),
                float(self.config["pitch_um"]) * 1e-6,
                float(self.config["step_um"]) * 1e-6,
                self.direction, self.wavelengths, self.re, self.im,
            )
            if self.config["boundary"] == "absorbing":
                self.tracer.t4_apply_absorbing_border(
                    1, int(self.config["size"]), int(self.config["size"]),
                    int(self.config["absorber_cells"]),
                    float(self.config["absorber_strength"]), 1.0,
                    self.re, self.im,
                )
            self.position += self.direction
            self.generation += 1

    def snapshot(self) -> dict[str, object]:
        field = self.re[0].astype(np.float64) + 1j * self.im[0].astype(np.float64)
        power = np.abs(field) ** 2
        delta = np.abs(field - self.initial)
        return {
            "config": self.config,
            "generation": self.generation,
            "position": self.position,
            "direction": self.direction,
            "distance_mm": self.position * float(self.config["step_um"]) * 1e-3,
            "field_power": float(np.sum(power)),
            "initial_power": float(np.sum(np.abs(self.initial) ** 2)),
            "panels": {
                "power": _png_data_url(_scalar_rgba(power, relative_peak=self.initial_peak)),
                "phase": _png_data_url(_phase_rgba(field, remove_piston=True)),
                "change": _png_data_url(_scalar_rgba(delta)),
            },
        }


class ArenaServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, ArenaHandler)
        self.arena = WaveArena()
        self.arena_lock = threading.Lock()


class ArenaHandler(BaseHTTPRequestHandler):
    server: ArenaServer

    def _send(self, status: int, body: bytes, mime: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: object) -> None:
        self._send(status, json.dumps(value, allow_nan=False).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/wave_arena_web_demo.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/wave_arena":
            with self.server.arena_lock:
                self._json(200, self.server.arena.snapshot())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/wave_arena":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 4096:
                raise ValueError("request body too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            with self.server.arena_lock:
                operation = payload.pop("op", None)
                if operation == "configure":
                    self.server.arena.configure(payload)
                elif operation == "step":
                    self.server.arena.step(int(payload.get("count", 1)))
                else:
                    raise ValueError("op must be configure or step")
                result = self.server.arena.snapshot()
            self._json(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    with ArenaServer(("127.0.0.1", args.port)) as server:
        print(f"Wave arena: http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
