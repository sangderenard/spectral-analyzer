"""
Bound particle in a spring-supported circular cage.

One particle is excited by the complex input of the hosting state-machine
module.  The cage is a circle with evenly spaced anchor points; each anchor
connects to the particle via a nonlinear spring with linear and cubic terms.

Outputs
- pos_x
- pos_y

Persisted state
- pos_x, pos_y, vel_x, vel_y
"""

from __future__ import annotations

import math
import numpy as np

STATE_VARS = ["pos_x", "pos_y", "vel_x", "vel_y"]
OUTPUT_VARS = ["pos_x", "pos_y"]
ITEM_PREFIX = "particle"

PARAM_SPECS = [
    {"name": "spring_k1", "label": "Spring k1", "dtype": "float",
     "default": 35.0, "low": 0.0, "high": 5000.0, "fmt": ".3f", "is_log": True, "group": "Springs"},
    {"name": "spring_k2", "label": "Spring k2", "dtype": "float",
     "default": 5.0, "low": 0.0, "high": 5000.0, "fmt": ".3f", "is_log": True, "group": "Springs"},
    {"name": "mass", "label": "Mass", "dtype": "float",
     "default": 1.0, "low": 0.01, "high": 100.0, "fmt": ".3f", "is_log": True, "group": "System"},
    {"name": "circle_radius", "label": "Circle Radius", "dtype": "float",
     "default": 1.0, "low": 0.01, "high": 10.0, "fmt": ".3f", "is_log": True, "group": "System"},
    {"name": "rest_length", "label": "Rest Length", "dtype": "float",
     "default": 1.0, "low": 0.0, "high": 10.0, "fmt": ".3f", "group": "System"},
    {"name": "n_bindings", "label": "Bindings", "dtype": "float",
     "default": 12.0, "low": 3.0, "high": 64.0, "fmt": ".0f", "group": "System"},
    {"name": "resonance_coeff", "label": "Res Coeff", "dtype": "float",
     "default": 0.5, "low": 0.0, "high": 4.0, "fmt": ".3f", "group": "Resonance"},
    {"name": "resonance_q", "label": "Res Q", "dtype": "float",
     "default": 1.2, "low": 0.1, "high": 50.0, "fmt": ".3f", "is_log": True, "group": "Resonance"},
    {"name": "resonance_center", "label": "Res Center", "dtype": "float",
     "default": 4.0, "low": 0.01, "high": 2000.0, "fmt": ".3f", "is_log": True, "group": "Resonance"},
]


def item_names(n_items):
    return ["particle"]


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _resonator_coeffs(center_hz: float, q: float, dt: float):
    fs = 1.0 / max(dt, 1e-12)
    w0 = 2.0 * math.pi * max(center_hz, 1e-6) / fs
    alpha = math.sin(w0) / max(2.0 * q, 1e-6)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def step(inputs, state, dt, n_items=1, use_torch=False, **kwargs):
    params = dict(kwargs.get("params", {}) or {})

    spring_k1 = float(params.get("spring_k1", 35.0))
    spring_k2 = float(params.get("spring_k2", 5.0))
    mass = max(float(params.get("mass", 1.0)), 1e-6)
    radius = max(float(params.get("circle_radius", 1.0)), 1e-6)
    rest_length = float(params.get("rest_length", 1.0))
    n_bindings = max(3, int(round(float(params.get("n_bindings", 12.0)))))
    res_coeff = float(params.get("resonance_coeff", 0.5))
    res_q = max(float(params.get("resonance_q", 1.2)), 1e-6)
    res_center = max(float(params.get("resonance_center", 4.0)), 1e-6)

    T = 0
    drive_x = None
    drive_y = None
    for arr in inputs.values():
        a = _to_numpy(arr)
        T = max(T, len(a))
        rx = a.real if np.iscomplexobj(a) else np.asarray(a, dtype=np.float64)
        ry = a.imag if np.iscomplexobj(a) else np.zeros_like(rx)
        drive_x = rx if drive_x is None else drive_x + rx
        drive_y = ry if drive_y is None else drive_y + ry
    if T <= 0:
        T = 1
    if drive_x is None:
        drive_x = np.zeros(T, dtype=np.float64)
        drive_y = np.zeros(T, dtype=np.float64)
    else:
        drive_x = np.asarray(drive_x[:T], dtype=np.float64)
        drive_y = np.asarray(drive_y[:T], dtype=np.float64)

    # Second-order resonant drive coloration per axis.
    b0, b1, b2, a1, a2 = _resonator_coeffs(res_center, res_q, dt)
    def _resonate(sig: np.ndarray) -> np.ndarray:
        y = np.zeros_like(sig, dtype=np.float64)
        x1 = x2 = y1 = y2 = 0.0
        for i, x0 in enumerate(sig):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        return y

    drive_x = drive_x + res_coeff * _resonate(drive_x)
    drive_y = drive_y + res_coeff * _resonate(drive_y)

    item = "particle"
    px = float(state.get(item, {}).get("pos_x", 0.0))
    py = float(state.get(item, {}).get("pos_y", 0.0))
    vx = float(state.get(item, {}).get("vel_x", 0.0))
    vy = float(state.get(item, {}).get("vel_y", 0.0))

    angles = np.linspace(0.0, 2.0 * math.pi, n_bindings, endpoint=False, dtype=np.float64)
    anchors = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    out_px = np.empty(T, dtype=np.float64)
    out_py = np.empty(T, dtype=np.float64)

    for i in range(T):
        p = np.array([px, py], dtype=np.float64)
        delta = anchors - p[None, :]
        dist = np.linalg.norm(delta, axis=1)
        unit = np.zeros_like(delta)
        nz = dist > 1e-9
        unit[nz] = delta[nz] / dist[nz, None]
        ext = dist - rest_length
        spring_mag = spring_k1 * ext + spring_k2 * (ext ** 3)
        spring_force = np.sum(unit * spring_mag[:, None], axis=0)
        force = spring_force + np.array([drive_x[i], drive_y[i]], dtype=np.float64)
        acc = force / mass
        vx += dt * acc[0]
        vy += dt * acc[1]
        px += dt * vx
        py += dt * vy
        out_px[i] = px
        out_py[i] = py

    return {
        "outputs": {
            item: {
                "pos_x": out_px,
                "pos_y": out_py,
            }
        },
        "state": {
            item: {
                "pos_x": float(px),
                "pos_y": float(py),
                "vel_x": float(vx),
                "vel_y": float(vy),
            }
        },
    }
