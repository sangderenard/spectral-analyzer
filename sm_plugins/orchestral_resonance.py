"""
State machine plugin: shared-room orchestral resonance with aperture feedback.

This packages the cavity engine as a deployable stepped state machine for the
analytic driver. All instruments are solved simultaneously inside one shared
geometry, aperture pressures are computed in one batched pass, and passive
feedback is applied on a scheduled iteration loop using cached stream state.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

import numpy as np
import torch

from cavity_engine import (
    AtmosphericSpec,
    CavityAperture,
    CavityPanel,
    CavityReceiver,
    CavityScene,
    CavitySource,
    CavitySourceBand,
    CavityStreamState,
    CircularRoom,
    DiffuseTailSpec,
    MeshRoom,
    PolygonalRoom,
    SceneMaterial,
    init_cavity_stream_state,
    load_obj_mesh_room,
    render_batched_body_steps,
    render_cavity_scene_step,
)

STATE_VARS = ["drive_rms", "aperture_peak", "feedback_peak", "mic_peak"]
OUTPUT_VARS = ["drive", "aperture_pressure", "feedback_pressure", "mic_left", "mic_right"]
ITEM_PREFIX = "inst"
PARAM_SPECS = [
    # ── Geometry ──────────────────────────────────────────────────────────────
    {"name": "room_shape", "label": "Room Shape", "dtype": "choice", "default": "polygon",
     "choices": ["polygon", "circular", "obj_mesh"], "group": "Geometry"},
    {"name": "scene_path", "label": "Scene Path", "dtype": "string", "default": "",
     "group": "Geometry"},
    {"name": "room_radius", "label": "Room Radius", "dtype": "float", "default": 3.4,
     "low": 1.0, "high": 12.0, "fmt": ".2f", "group": "Geometry"},
    {"name": "room_height", "label": "Room Height", "dtype": "float", "default": 3.6,
     "low": 1.5, "high": 12.0, "fmt": ".2f", "group": "Geometry"},
    # ── Receiver / mic array ──────────────────────────────────────────────────
    # Key into mic_arrays registry. Empty = legacy stereo pair at front of room.
    {"name": "receiver_array_key", "label": "Mic Array Preset", "dtype": "string",
     "default": "", "group": "Receivers"},
    # World-space position of the array center (meters, same coord as room).
    {"name": "receiver_pos_x", "label": "Recv Pos X", "dtype": "float", "default": 0.0,
     "low": -20.0, "high": 20.0, "fmt": ".2f", "group": "Receivers"},
    {"name": "receiver_pos_y", "label": "Recv Pos Y", "dtype": "float", "default": 0.0,
     "low": -20.0, "high": 20.0, "fmt": ".2f", "group": "Receivers"},
    {"name": "receiver_pos_z", "label": "Recv Pos Z", "dtype": "float", "default": 1.5,
     "low": 0.0, "high": 5.0, "fmt": ".2f", "group": "Receivers"},
    # Forward direction the array faces (unit vector before normalization).
    {"name": "receiver_fwd_x", "label": "Recv Fwd X", "dtype": "float", "default": 0.0,
     "low": -1.0, "high": 1.0, "fmt": ".3f", "group": "Receivers"},
    {"name": "receiver_fwd_y", "label": "Recv Fwd Y", "dtype": "float", "default": 1.0,
     "low": -1.0, "high": 1.0, "fmt": ".3f", "group": "Receivers"},
    {"name": "receiver_fwd_z", "label": "Recv Fwd Z", "dtype": "float", "default": 0.0,
     "low": -1.0, "high": 1.0, "fmt": ".3f", "group": "Receivers"},
    # ── Performer geometry from placement engine ───────────────────────────────
    # JSON array: [{key, x, y, z, dir_x, dir_y, dir_z, body_type}, ...]
    # Written by analytic_driver placement sync; empty = internal layout.
    {"name": "performer_geometry_json", "label": "Performer Geometry", "dtype": "string",
     "default": "", "group": "Placement"},
    # ── Coupling ──────────────────────────────────────────────────────────────
    {"name": "feedback_iterations", "label": "Feedback Iters", "dtype": "int", "default": 1,
     "low": 0.0, "high": 4.0, "fmt": ".0f", "group": "Coupling"},
    {"name": "feedback_gain", "label": "Feedback Gain", "dtype": "float", "default": 0.16,
     "low": 0.0, "high": 0.6, "fmt": ".3f", "group": "Coupling"},
    {"name": "passive_loss", "label": "Passive Loss", "dtype": "float", "default": 0.48,
     "low": 0.0, "high": 0.98, "fmt": ".3f", "group": "Coupling"},
    # ── Section sympathetic coupling ──────────────────────────────────────────
    # Performers in the same part/section feed each other through sympathetic
    # string-like coupling: each performer hears its neighbours with a delay
    # that combines physical propagation + human reaction lag.
    {"name": "section_coupling_gain", "label": "Section Coupling", "dtype": "float",
     "default": 0.08, "low": 0.0, "high": 0.5, "fmt": ".3f", "group": "Coupling"},
    {"name": "section_reaction_lag_ms", "label": "Reaction Lag ms", "dtype": "float",
     "default": 12.0, "low": 0.0, "high": 80.0, "fmt": ".1f", "group": "Coupling"},
    # ── Projection ────────────────────────────────────────────────────────────
    {"name": "band_split_mode", "label": "Band Split", "dtype": "choice", "default": "fir",
     "choices": ["fir", "fft"], "group": "Projection"},
    {"name": "fir_taps", "label": "FIR Taps", "dtype": "int", "default": 65,
     "low": 5.0, "high": 257.0, "fmt": ".0f", "group": "Projection"},
    {"name": "high_cone_deg", "label": "High Cone", "dtype": "float", "default": 70.0,
     "low": 5.0, "high": 180.0, "fmt": ".1f", "group": "Projection"},
    # ── Room ─────────────────────────────────────────────────────────────────
    {"name": "diffuse_strength", "label": "Diffuse", "dtype": "float", "default": 0.42,
     "low": 0.0, "high": 1.0, "fmt": ".3f", "group": "Room"},
    # ── Atmosphere ────────────────────────────────────────────────────────────
    {"name": "air_db_per_m", "label": "Air Loss", "dtype": "float", "default": 0.01,
     "low": 0.0, "high": 0.5, "fmt": ".4f", "group": "Atmosphere"},
    {"name": "air_highband_db_per_m", "label": "Air Hi Loss", "dtype": "float", "default": 0.02,
     "low": 0.0, "high": 1.0, "fmt": ".4f", "group": "Atmosphere"},
    {"name": "temperature_c", "label": "Temp C", "dtype": "float", "default": 20.0,
     "low": -20.0, "high": 50.0, "fmt": ".1f", "group": "Atmosphere"},
    {"name": "humidity_rel", "label": "Humidity", "dtype": "float", "default": 0.5,
     "low": 0.0, "high": 1.0, "fmt": ".2f", "group": "Atmosphere"},
]


def item_names(n_items: int) -> list[str]:
    return [f"{ITEM_PREFIX}{i}" for i in range(max(1, int(n_items)))]


def _to_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return np.asarray(arr, dtype=np.complex128)
    return np.asarray(arr, dtype=np.float64)


def _np_complex(arr: Any, n: int) -> np.ndarray:
    raw = _to_numpy(arr)
    if raw.ndim == 0:
        raw = np.full((n,), raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64)
    if len(raw) < n:
        raw = np.pad(raw, (0, n - len(raw)), mode="constant")
    raw = raw[:n]
    if np.iscomplexobj(raw):
        return np.asarray(raw, dtype=np.complex128)
    return np.asarray(raw, dtype=np.float64).astype(np.complex128)


def _polygon_vertices(radius: float) -> list[tuple[float, float]]:
    return [
        (-radius * 0.95, -radius * 0.70),
        (radius, -radius * 0.62),
        (radius * 0.88, radius * 0.72),
        (-radius * 0.84, radius * 0.64),
    ]


def _instrument_position(i: int, n_items: int, radius: float) -> tuple[float, float, float]:
    if n_items <= 1:
        return (-radius * 0.25, 0.0, 1.15)
    frac = (i / max(n_items - 1, 1)) - 0.5
    ang = frac * math.pi * 0.85
    r = radius * 0.42
    return (r * math.sin(ang), -r * math.cos(ang) * 0.55, 1.05 + 0.08 * (i % 3))


def _instrument_direction(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    dx = -pos[0]
    dy = max(0.2, 1.25 - pos[1])
    norm = math.hypot(dx, dy) or 1.0
    return (dx / norm, dy / norm, 0.0)


def _parse_performer_geometry(json_str: str) -> list[dict]:
    """Parse performer_geometry_json from placement sync. Returns [] on error."""
    raw = str(json_str or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ── Per-instrument body profiles ──────────────────────────────────────────────
# Each body_type key maps to:
#   bands: list of (low_hz, high_hz, cone_angle_deg, directivity_power, gain)
#   aperture_offset_m: how far the sound aperture is from the source center
#   capture_power: how strongly the aperture couples energy back (feedback)
#   jitter: dict of (field: (mean_offset, spread)) for per-performer randomisation
#
# "direct" = no body resonance, essentially a point source.
# "drum_shell" models a cylindrical bucket with:
#   – sub-200 Hz membrane mode: wide, high gain (the thump)
#   – 200–800 Hz shell ring: moderate directivity from the open top
#   – 800+ Hz attack transient: narrow from the stick strike
_BODY_PROFILES: dict[str, dict] = {
    "direct": {
        "bands": [
            (0.0, 280.0, 180.0, 0.0, 1.0),
            (280.0, 1600.0, 130.0, 1.2, 1.0),
            (1600.0, None, 70.0, 3.0, 1.08),
        ],
        "aperture_offset_m": 0.12,
        "capture_power": 0.4,
    },
    "string_plate": {
        "bands": [
            (0.0, 220.0, 160.0, 0.2, 1.05),   # low: body swell, nearly omni
            (220.0, 1800.0, 100.0, 1.8, 1.12),  # mid: plate resonance, tighter
            (1800.0, None, 60.0, 3.2, 1.15),    # high: bridge/bow attack
        ],
        "aperture_offset_m": 0.18,
        "capture_power": 0.35,
    },
    "reed_box": {
        "bands": [
            (0.0, 260.0, 170.0, 0.1, 0.92),    # low: bore standing wave, wide
            (260.0, 1400.0, 90.0, 2.0, 1.18),   # mid: reed vibration, tight
            (1400.0, None, 55.0, 3.5, 1.20),    # high: reed click, narrow
        ],
        "aperture_offset_m": 0.08,
        "capture_power": 0.45,
    },
    "brass_bell": {
        "bands": [
            (0.0, 300.0, 140.0, 0.4, 0.95),    # low: backpressure, semi-wide
            (300.0, 2200.0, 75.0, 2.5, 1.22),   # mid: bell radiation, focused
            (2200.0, None, 40.0, 4.0, 1.30),    # high: brassiness, very narrow
        ],
        "aperture_offset_m": 0.30,              # bell extends far from player
        "capture_power": 0.28,
    },
    "drum_shell": {
        "bands": [
            (0.0, 200.0, 180.0, 0.0, 1.35),    # sub: membrane fundamental, omni
            (200.0, 800.0, 120.0, 0.8, 1.10),   # low-mid: shell ring
            (800.0, None, 85.0, 1.6, 0.95),     # high: stick attack, moderately wide
        ],
        "aperture_offset_m": 0.22,              # open top of the bucket
        "capture_power": 0.55,                  # strong coupling in the shell
    },
    "pipe_column": {
        "bands": [
            (0.0, 240.0, 180.0, 0.0, 1.0),     # low: column mode, omni
            (240.0, 1200.0, 110.0, 1.5, 1.08),  # mid: standing wave harmonics
            (1200.0, None, 50.0, 3.8, 1.10),    # high: chiff/overblown
        ],
        "aperture_offset_m": 0.10,
        "capture_power": 0.50,
    },
    "voice_body": {
        "bands": [
            (0.0, 250.0, 160.0, 0.15, 0.90),   # low: chest resonance, wide
            (250.0, 2800.0, 95.0, 1.8, 1.25),   # mid: formant region, tight
            (2800.0, None, 65.0, 2.8, 1.08),    # high: sibilance, moderate
        ],
        "aperture_offset_m": 0.06,
        "capture_power": 0.30,
    },
}


def _body_source_bands(body_type: str, high_cone_override: float,
                       jitter_rng: random.Random | None = None) -> tuple[list[CavitySourceBand], float, float]:
    """Return (band_profiles, aperture_offset_m, capture_power) for a body type.

    When *jitter_rng* is provided, slight per-performer variation is applied to
    cone angles (±4°), directivity power (±8%), gain (±3%), and aperture
    offset (±12%) so that two violins in the same section sound subtly different.
    """
    profile = _BODY_PROFILES.get(body_type, _BODY_PROFILES["direct"])
    raw_bands = profile["bands"]
    ap_offset = float(profile["aperture_offset_m"])
    cap_power = float(profile["capture_power"])

    def _jit(val: float, spread: float) -> float:
        if jitter_rng is None:
            return val
        return val * (1.0 + jitter_rng.uniform(-spread, spread))

    bands: list[CavitySourceBand] = []
    for low, high, cone, dp, gain in raw_bands:
        # For the topmost band, allow the high_cone_override to win if it
        # would narrow the pattern further (e.g. from the patch config)
        if high is None and body_type == "direct":
            cone = high_cone_override
        bands.append(CavitySourceBand(
            low_hz=low,
            high_hz=high,
            cone_angle_deg=max(10.0, _jit(cone, 0.04)),
            directivity_power=max(0.0, _jit(dp, 0.08)),
            gain=max(0.1, _jit(gain, 0.03)),
        ))

    ap_offset = max(0.02, _jit(ap_offset, 0.12))
    cap_power = max(0.05, min(0.95, _jit(cap_power, 0.08)))
    return bands, ap_offset, cap_power


# ── Instrument body geometry ─────────────────────────────────────────────────
# Each body type builds a standalone CavityScene whose panels form the physical
# shell of the instrument.  The driver signal enters at the source position
# inside the body, the panels shape radiation via reflection/diffusion/feedback,
# and the aperture captures the output that then feeds the room scene.
#
# Geometry is centred at origin; _build_body_scene translates to world position.

def _ring_panels(
    key_prefix: str,
    n_segments: int,
    radius: float,
    z_lo: float,
    z_hi: float,
    reflectivity: float = 0.82,
    diffusion: float = 0.20,
    absorption: float = 0.12,
    *,
    normal_reflectivity: float | None = None,
    grazing_reflectivity: float | None = None,
    phase_normal: float = 0.0,
    phase_grazing: float = 0.0,
) -> list[CavityPanel]:
    """Create a cylindrical ring of wall panels (shell staves / tube walls)."""
    panels: list[CavityPanel] = []
    z_mid = (z_lo + z_hi) * 0.5
    for i in range(n_segments):
        ang = 2.0 * math.pi * (i / n_segments)
        nx, ny = -math.cos(ang), -math.sin(ang)
        panels.append(CavityPanel(
            key=f"{key_prefix}_seg{i}",
            point=(radius * math.cos(ang), radius * math.sin(ang), z_mid),
            normal=(nx, ny, 0.0),
            reflectivity=reflectivity,
            normal_reflectivity=normal_reflectivity,
            grazing_reflectivity=grazing_reflectivity,
            normal_phase_rad=phase_normal,
            grazing_phase_rad=phase_grazing,
            diffusion=diffusion,
            absorption=absorption,
            is_baffle=True,
        ))
    return panels


def _cap_panel(
    key: str,
    z: float,
    normal_z: float,
    reflectivity: float = 0.70,
    diffusion: float = 0.30,
    absorption: float = 0.18,
    *,
    normal_reflectivity: float | None = None,
    grazing_reflectivity: float | None = None,
    phase_normal: float = 0.0,
    phase_grazing: float = 0.0,
) -> CavityPanel:
    """One flat cap (membrane, soundboard, bell rim, etc.)."""
    return CavityPanel(
        key=key,
        point=(0.0, 0.0, z),
        normal=(0.0, 0.0, normal_z),
        reflectivity=reflectivity,
        normal_reflectivity=normal_reflectivity,
        grazing_reflectivity=grazing_reflectivity,
        normal_phase_rad=phase_normal,
        grazing_phase_rad=phase_grazing,
        diffusion=diffusion,
        absorption=absorption,
        is_baffle=True,
    )


def _body_panels_direct(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """No physical body — empty list means the body scene degenerates to a
    trivial pass-through (skip the body solve entirely)."""
    return []


def _body_panels_drum_shell(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """Cylindrical drum shell: membrane (bottom cap), open top, shell walls.

    The membrane is highly reflective with strong phase rotation (tension), the
    shell walls are moderately reflective wood/metal, and there is no top cap
    (the open head where the stick strikes and sound exits).
    """
    def _j(v: float, s: float) -> float:
        return v * (1.0 + jitter_rng.uniform(-s, s)) if jitter_rng else v

    shell_r = _j(0.18, 0.06)     # ~36 cm diameter
    shell_h = _j(0.30, 0.08)     # ~30 cm deep
    n_staves = 12

    panels = _ring_panels(
        "drum_shell", n_staves, shell_r, 0.0, shell_h,
        reflectivity=0.78, diffusion=0.18, absorption=0.14,
        normal_reflectivity=0.85, grazing_reflectivity=0.65,
        phase_normal=0.10, phase_grazing=0.38,
    )
    # Bottom membrane — high reflectivity, significant phase (tensioned skin)
    panels.append(_cap_panel(
        "drum_membrane", z=0.0, normal_z=1.0,
        reflectivity=0.92, diffusion=0.08, absorption=0.04,
        normal_reflectivity=0.96, grazing_reflectivity=0.80,
        phase_normal=_j(0.62, 0.10), phase_grazing=_j(1.05, 0.10),
    ))
    # No top cap — open head.  A low-reflectivity "lip" ring around the rim
    # captures edge diffraction.
    panels.append(_cap_panel(
        "drum_rim", z=shell_h, normal_z=-1.0,
        reflectivity=0.25, diffusion=0.55, absorption=0.35,
        phase_normal=0.04, phase_grazing=0.18,
    ))
    return panels


def _body_panels_string_plate(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """String instrument body: top plate (soundboard), back plate, ribs.

    Top plate is the primary radiator — high reflectivity, moderate phase.
    Back plate is stiffer, higher absorption.  Ribs connect them.
    """
    def _j(v: float, s: float) -> float:
        return v * (1.0 + jitter_rng.uniform(-s, s)) if jitter_rng else v

    body_half_w = _j(0.17, 0.05)   # ~34 cm wide
    body_h = _j(0.06, 0.08)        # ~6 cm deep (rib height)
    n_rib_segs = 10

    panels = _ring_panels(
        "str_rib", n_rib_segs, body_half_w, 0.0, body_h,
        reflectivity=0.72, diffusion=0.28, absorption=0.16,
        normal_reflectivity=0.78, grazing_reflectivity=0.60,
        phase_normal=0.08, phase_grazing=0.30,
    )
    # Top plate (soundboard) — the primary radiator
    panels.append(_cap_panel(
        "str_top_plate", z=body_h, normal_z=-1.0,
        reflectivity=0.88, diffusion=0.12, absorption=0.06,
        normal_reflectivity=0.94, grazing_reflectivity=0.75,
        phase_normal=_j(0.35, 0.08), phase_grazing=_j(0.72, 0.10),
    ))
    # Back plate — stiffer, less radiative
    panels.append(_cap_panel(
        "str_back_plate", z=0.0, normal_z=1.0,
        reflectivity=0.80, diffusion=0.15, absorption=0.12,
        normal_reflectivity=0.86, grazing_reflectivity=0.68,
        phase_normal=_j(0.18, 0.06), phase_grazing=_j(0.44, 0.08),
    ))
    return panels


def _body_panels_reed_box(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """Reed instrument bore: cylindrical tube, open bell end, closed reed end.

    The bore walls are smooth and highly reflective (standing waves).
    The reed end is nearly closed.  The bell is open with edge diffraction.
    """
    def _j(v: float, s: float) -> float:
        return v * (1.0 + jitter_rng.uniform(-s, s)) if jitter_rng else v

    bore_r = _j(0.012, 0.06)    # ~24 mm diameter
    bore_len = _j(0.60, 0.08)   # ~60 cm long
    n_segs = 8

    panels = _ring_panels(
        "reed_bore", n_segs, bore_r, 0.0, bore_len,
        reflectivity=0.90, diffusion=0.06, absorption=0.04,
        normal_reflectivity=0.95, grazing_reflectivity=0.82,
        phase_normal=0.04, phase_grazing=0.14,
    )
    # Reed end — nearly closed
    panels.append(_cap_panel(
        "reed_mouthpiece", z=bore_len, normal_z=-1.0,
        reflectivity=0.94, diffusion=0.04, absorption=0.02,
        normal_reflectivity=0.97, grazing_reflectivity=0.88,
        phase_normal=_j(0.48, 0.10), phase_grazing=_j(0.85, 0.12),
    ))
    # Bell end — open, diffracting
    panels.append(_cap_panel(
        "reed_bell", z=0.0, normal_z=1.0,
        reflectivity=0.20, diffusion=0.50, absorption=0.40,
        phase_normal=0.02, phase_grazing=0.10,
    ))
    return panels


def _body_panels_brass_bell(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """Brass instrument: tapered bore (modelled as two cylinder sections) + bell.

    Narrow cylindrical bore near the mouthpiece, flaring to a wide bell.
    Mouthpiece end nearly closed; bell is the primary radiator.
    """
    def _j(v: float, s: float) -> float:
        return v * (1.0 + jitter_rng.uniform(-s, s)) if jitter_rng else v

    # Narrow bore section
    bore_r = _j(0.006, 0.06)
    bore_len = _j(0.80, 0.08)
    panels = _ring_panels(
        "brass_bore", 8, bore_r, 0.0, bore_len * 0.6,
        reflectivity=0.93, diffusion=0.04, absorption=0.03,
        normal_reflectivity=0.96, grazing_reflectivity=0.88,
        phase_normal=0.03, phase_grazing=0.10,
    )
    # Flared bell section
    bell_r = _j(0.15, 0.06)
    panels += _ring_panels(
        "brass_bell", 10, bell_r, bore_len * 0.6, bore_len,
        reflectivity=0.82, diffusion=0.22, absorption=0.10,
        normal_reflectivity=0.88, grazing_reflectivity=0.70,
        phase_normal=0.06, phase_grazing=0.24,
    )
    # Mouthpiece — nearly closed
    panels.append(_cap_panel(
        "brass_mpc", z=0.0, normal_z=1.0,
        reflectivity=0.96, diffusion=0.02, absorption=0.02,
        normal_reflectivity=0.98, grazing_reflectivity=0.90,
        phase_normal=_j(0.52, 0.08), phase_grazing=_j(0.90, 0.10),
    ))
    # Bell rim — open radiator
    panels.append(_cap_panel(
        "brass_bell_rim", z=bore_len, normal_z=-1.0,
        reflectivity=0.18, diffusion=0.55, absorption=0.38,
        phase_normal=0.02, phase_grazing=0.08,
    ))
    return panels


def _body_panels_pipe_column(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """Flute / organ pipe: open cylindrical column, both ends partially open."""
    def _j(v: float, s: float) -> float:
        return v * (1.0 + jitter_rng.uniform(-s, s)) if jitter_rng else v

    pipe_r = _j(0.010, 0.06)
    pipe_len = _j(0.55, 0.10)
    panels = _ring_panels(
        "pipe_wall", 8, pipe_r, 0.0, pipe_len,
        reflectivity=0.91, diffusion=0.06, absorption=0.04,
        normal_reflectivity=0.95, grazing_reflectivity=0.84,
        phase_normal=0.03, phase_grazing=0.12,
    )
    # Embouchure end — partially open
    panels.append(_cap_panel(
        "pipe_embouchure", z=pipe_len, normal_z=-1.0,
        reflectivity=0.35, diffusion=0.40, absorption=0.30,
        phase_normal=0.06, phase_grazing=0.20,
    ))
    # Foot end — open
    panels.append(_cap_panel(
        "pipe_foot", z=0.0, normal_z=1.0,
        reflectivity=0.22, diffusion=0.48, absorption=0.38,
        phase_normal=0.03, phase_grazing=0.12,
    ))
    return panels


def _body_panels_voice_body(jitter_rng: random.Random | None = None) -> list[CavityPanel]:
    """Vocal tract: short tapered tube, open mouth, closed glottis."""
    def _j(v: float, s: float) -> float:
        return v * (1.0 + jitter_rng.uniform(-s, s)) if jitter_rng else v

    tract_r = _j(0.018, 0.06)
    tract_len = _j(0.17, 0.08)
    panels = _ring_panels(
        "vocal_tract", 8, tract_r, 0.0, tract_len,
        reflectivity=0.70, diffusion=0.30, absorption=0.20,
        normal_reflectivity=0.76, grazing_reflectivity=0.58,
        phase_normal=0.08, phase_grazing=0.28,
    )
    # Glottis — nearly closed
    panels.append(_cap_panel(
        "glottis", z=0.0, normal_z=1.0,
        reflectivity=0.92, diffusion=0.05, absorption=0.04,
        normal_reflectivity=0.96, grazing_reflectivity=0.84,
        phase_normal=_j(0.55, 0.10), phase_grazing=_j(0.95, 0.12),
    ))
    # Mouth opening
    panels.append(_cap_panel(
        "mouth", z=tract_len, normal_z=-1.0,
        reflectivity=0.28, diffusion=0.45, absorption=0.35,
        phase_normal=0.04, phase_grazing=0.16,
    ))
    return panels


_BODY_PANEL_BUILDERS: dict[str, Any] = {
    "direct":       _body_panels_direct,
    "string_plate": _body_panels_string_plate,
    "reed_box":     _body_panels_reed_box,
    "brass_bell":   _body_panels_brass_bell,
    "drum_shell":   _body_panels_drum_shell,
    "pipe_column":  _body_panels_pipe_column,
    "voice_body":   _body_panels_voice_body,
}


def _build_body_scene(
    body_type: str,
    jitter_rng: random.Random | None = None,
) -> CavityScene | None:
    """Build a standalone CavityScene modelling one instrument body.

    Returns None for "direct" (no body resonance — skip the nested solve).
    The scene is centred at origin; the caller translates world-space.
    Source is at the centre of the body, aperture at the primary opening.
    """
    builder = _BODY_PANEL_BUILDERS.get(body_type, _body_panels_direct)
    panels = builder(jitter_rng=jitter_rng)
    if not panels:
        return None  # "direct" — no body to solve

    profile = _BODY_PROFILES.get(body_type, _BODY_PROFILES["direct"])
    ap_offset = float(profile["aperture_offset_m"])

    # Source at body centre, radiating along +Z (upward / toward aperture)
    source = CavitySource(
        key="body_driver",
        position=(0.0, 0.0, ap_offset * 0.4),
        direction=(0.0, 0.0, 1.0),
        gain=1.0,
        directivity_power=0.5,
        band_profiles=[
            CavitySourceBand(low_hz=0.0, high_hz=None, cone_angle_deg=180.0,
                             directivity_power=0.0, gain=1.0),
        ],
    )
    # Single receiver at the aperture opening to capture body output
    receiver = CavityReceiver(
        key="body_out",
        position=(0.0, 0.0, ap_offset),
        direction=(0.0, 0.0, 1.0),
        directivity_power=0.0,
        polar_pattern="omni",
    )
    # Aperture for feedback inside the body
    aperture = CavityAperture(
        key="body_ap",
        source_key="body_driver",
        position=(0.0, 0.0, ap_offset),
        direction=(0.0, 0.0, 1.0),
        capture_power=float(profile["capture_power"]),
        feedback_gain=0.12,
        passive_loss=0.40,
    )

    # Tiny internal atmosphere — warm, slightly humid (inside the instrument)
    body_atmo = AtmosphericSpec(
        temperature_c=28.0,
        humidity_rel=0.55,
        pressure_kpa=101.325,
        attenuation_db_per_m=0.005,
        high_band_extra_db_per_m=0.008,
    )

    # The body geometry is a polygon room whose baffles ARE the body panels.
    # The polygon room walls are set to a tiny enclosing box that is fully
    # absorptive — only the baffle panels (the actual instrument) matter.
    body_geom = PolygonalRoom(
        vertices_xy=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
        height=1.0,
        closed_floor=False,
        closed_roof=False,
        wall_reflectivity=0.0,
        wall_diffusion=0.0,
        wall_absorption=1.0,
        baffles=panels,
    )
    return CavityScene(
        sources=[source],
        receivers=[receiver],
        geometry=body_geom,
        apertures=[aperture],
        diffuse_tail=DiffuseTailSpec(
            taps_per_panel=4,
            decay_s=0.15,
            delay_spread_s=0.005,
            strength=0.20,
            truncate_at_s=0.3,
        ),
        atmosphere=body_atmo,
        aperture_feedback_iterations=1,
        band_split_mode="fir",
        band_split_fir_taps=33,
    )


def _build_receivers_from_preset(
    array_key: str,
    params: dict[str, Any],
    radius: float,
) -> tuple[list[CavityReceiver], list[CavityPanel]]:
    """Build receivers (and any array baffles) from a mic_arrays preset.
    Falls back to the legacy stereo pair when the key is empty or unknown."""
    try:
        import mic_arrays
        cfg = mic_arrays.get_array(array_key)
    except ImportError:
        cfg = None

    if cfg is None:
        # Legacy stereo pair at front of room
        return (
            [
                CavityReceiver(key="mic_l", position=(-0.12, radius * 0.55, 1.45),
                               direction=(0.0, -1.0, 0.0), directivity_power=1.0,
                               polar_pattern="cardioid"),
                CavityReceiver(key="mic_r", position=(0.12, radius * 0.55, 1.45),
                               direction=(0.0, -1.0, 0.0), directivity_power=1.0,
                               polar_pattern="cardioid"),
            ],
            [],
        )

    px = float(params.get("receiver_pos_x", 0.0))
    py = float(params.get("receiver_pos_y", 0.0))
    pz = float(params.get("receiver_pos_z", 1.5))
    fx = float(params.get("receiver_fwd_x", 0.0))
    fy = float(params.get("receiver_fwd_y", 1.0))
    fz = float(params.get("receiver_fwd_z", 0.0))
    fn = math.sqrt(fx*fx + fy*fy + fz*fz) or 1.0
    fwd = (fx/fn, fy/fn, fz/fn)
    center = (px, py, pz)

    receivers = cfg.to_cavity_receivers(center_position=center, forward_direction=fwd)
    baffles = cfg.to_cavity_baffles(center_position=center, forward_direction=fwd)
    return receivers, baffles


def _build_scene(params: dict[str, Any], n_items: int) -> CavityScene:
    room_shape = str(params.get("room_shape", "polygon"))
    scene_path = str(params.get("scene_path", "") or "").strip()
    radius = max(1.0, float(params.get("room_radius", 3.4)))
    height = max(1.5, float(params.get("room_height", 3.6)))
    high_cone = float(params.get("high_cone_deg", 70.0))
    feedback_gain = max(0.0, float(params.get("feedback_gain", 0.16)))
    passive_loss = min(0.98, max(0.0, float(params.get("passive_loss", 0.48))))
    diffuse_strength = max(0.0, float(params.get("diffuse_strength", 0.42)))

    # ── Sources: use placement performer geometry if provided ─────────────────
    performer_geo = _parse_performer_geometry(str(params.get("performer_geometry_json", "") or ""))
    sources: list[CavitySource] = []
    apertures: list[CavityAperture] = []
    for i, item in enumerate(item_names(n_items)):
        body_type = "direct"
        if i < len(performer_geo):
            pg = performer_geo[i]
            pos = (float(pg.get("x", 0.0)), float(pg.get("y", 0.0)), float(pg.get("z", 1.1)))
            dx, dy, dz = float(pg.get("dir_x", 0.0)), float(pg.get("dir_y", 1.0)), float(pg.get("dir_z", 0.0))
            dn = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
            direction = (dx/dn, dy/dn, dz/dn)
            body_type = str(pg.get("body_type", "direct"))
        else:
            pos = _instrument_position(i, n_items, radius)
            direction = _instrument_direction(pos)

        # Per-performer deterministic jitter RNG seeded by performer key
        _pf_key = pg.get("key", item) if i < len(performer_geo) else item
        _jit_rng = random.Random(hash(_pf_key) & 0xFFFFFFFF)

        band_profiles, ap_offset, cap_power = _body_source_bands(
            body_type, high_cone, jitter_rng=_jit_rng)

        source_key = f"src_{item}"
        sources.append(CavitySource(
            key=source_key,
            position=pos,
            direction=direction,
            gain=1.0,
            directivity_power=band_profiles[1].directivity_power if len(band_profiles) > 1 else 1.2,
            band_profiles=band_profiles,
        ))
        apertures.append(CavityAperture(
            key=f"ap_{item}",
            source_key=source_key,
            position=(pos[0] + ap_offset * direction[0],
                      pos[1] + ap_offset * direction[1],
                      pos[2]),
            direction=direction,
            capture_power=cap_power,
            feedback_gain=feedback_gain,
            passive_loss=passive_loss,
        ))

    # ── Receivers: from mic array preset or legacy stereo pair ────────────────
    array_key = str(params.get("receiver_array_key", "") or "").strip()
    if not array_key:
        array_key = "binaural_standard"
    receivers, array_baffles = _build_receivers_from_preset(array_key, params, radius)

    # If receiver position was not explicitly set, default to front-of-room
    if not any(params.get(k) for k in ("receiver_pos_x", "receiver_pos_y", "receiver_pos_z")):
        # re-build with sensible default position relative to room
        try:
            import mic_arrays
            cfg = mic_arrays.get_array(array_key)
            if cfg is not None:
                default_center = (0.0, radius * 0.55, 1.45)
                default_fwd = (0.0, -1.0, 0.0)
                receivers = cfg.to_cavity_receivers(center_position=default_center, forward_direction=default_fwd)
                array_baffles = cfg.to_cavity_baffles(center_position=default_center, forward_direction=default_fwd)
        except ImportError:
            pass

    room_baffles = [
        CavityPanel(
            key="center_baffle",
            point=(0.0, 0.0, height * 0.42),
            normal=(1.0, 0.0, 0.0),
            reflectivity=0.86,
            normal_reflectivity=0.92,
            grazing_reflectivity=0.72,
            normal_phase_rad=0.14,
            grazing_phase_rad=0.52,
            diffusion=0.48,
            absorption=0.08,
            is_baffle=True,
        )
    ]
    all_baffles = room_baffles + list(array_baffles)

    if room_shape == "obj_mesh" and scene_path:
        geometry: PolygonalRoom | CircularRoom | MeshRoom = load_obj_mesh_room(
            scene_path,
            front_material=SceneMaterial(key="front", reflectivity=0.86, normal_reflectivity=0.92, grazing_reflectivity=0.72,
                                         normal_phase_rad=0.14, grazing_phase_rad=0.52, diffusion=0.48, absorption=0.08),
        )
    elif room_shape == "circular":
        geometry = CircularRoom(
            radius=radius,
            height=height,
            n_segments=24,
            baffles=all_baffles,
        )
    else:
        geometry = PolygonalRoom(
            vertices_xy=_polygon_vertices(radius),
            height=height,
            baffles=all_baffles,
        )
    return CavityScene(
        sources=sources,
        receivers=receivers,
        geometry=geometry,
        apertures=apertures,
        diffuse_tail=DiffuseTailSpec(
            taps_per_panel=8,
            decay_s=1.2,
            delay_spread_s=0.05,
            strength=diffuse_strength,
            truncate_at_s=3.0,
        ),
        band_split_mode=str(params.get("band_split_mode", "fir")),
        band_split_fir_taps=max(5, int(params.get("fir_taps", 65)) | 1),
        aperture_feedback_iterations=max(0, int(params.get("feedback_iterations", 1))),
        atmosphere=AtmosphericSpec(
            temperature_c=float(params.get("temperature_c", 20.0)),
            humidity_rel=max(0.0, min(1.0, float(params.get("humidity_rel", 0.5)))),
            attenuation_db_per_m=max(0.0, float(params.get("air_db_per_m", 0.01))),
            high_band_extra_db_per_m=max(0.0, float(params.get("air_highband_db_per_m", 0.02))),
        ),
    )


def step(inputs, state, dt, n_items=1, use_torch=False, **kwargs):
    params = dict(kwargs.get("params", {}) or {})
    plugin_state = dict(kwargs.get("plugin_state", {}) or {})
    names = item_names(n_items)

    T = 0
    for arr in inputs.values():
        T = max(T, len(_to_numpy(arr)))
    if T <= 0:
        T = 1

    # One simultaneous source drive per instrument. If fewer routed inputs exist
    # than instruments, the remainder stay silent for this step.
    source_signals = np.zeros((len(names), T), dtype=np.complex128)
    for i, _src_key in enumerate(sorted(inputs.keys())[:len(names)]):
        source_signals[i] = _np_complex(inputs[_src_key], T)

    sample_rate = 1.0 / max(float(dt), 1e-9)

    # ── Per-instrument body resonator solve (nested, standalone) ──────────────
    # Each non-"direct" instrument gets its own tiny CavityScene modelling the
    # instrument body.  The driver signal enters, gets shaped by the body panels,
    # and the receiver output replaces the raw driver before it hits the room.
    performer_geo = _parse_performer_geometry(
        str(params.get("performer_geometry_json", "") or ""))
    body_states: dict[str, CavityStreamState] = dict(
        plugin_state.get("body_states") or {})
    body_scenes: dict[str, CavityScene] = dict(
        plugin_state.get("body_scenes") or {})

    # Build / reuse per-performer body scenes (preserves jitter)
    for i, item in enumerate(names):
        body_type = "direct"
        if i < len(performer_geo):
            body_type = str(performer_geo[i].get("body_type", "direct"))
        if body_type == "direct":
            continue
        if item not in body_scenes:
            _pf_key = performer_geo[i].get("key", item) if i < len(performer_geo) else item
            _jit_rng = random.Random(hash(_pf_key) & 0xFFFFFFFF)
            bscene = _build_body_scene(body_type, jitter_rng=_jit_rng)
            if bscene is None:
                continue
            body_scenes[item] = bscene

    # Group non-direct performers by body_type for batched solve
    type_groups: dict[str, list[tuple[int, str]]] = {}  # body_type → [(idx, name), ...]
    for i, item in enumerate(names):
        if item not in body_scenes:
            continue
        bt = "direct"
        if i < len(performer_geo):
            bt = str(performer_geo[i].get("body_type", "direct"))
        if bt == "direct":
            continue
        type_groups.setdefault(bt, []).append((i, item))

    for bt, members in type_groups.items():
        indices = [m[0] for m in members]
        member_names = [m[1] for m in members]
        group_scenes = [body_scenes[nm] for nm in member_names]
        group_states = []
        for nm in member_names:
            st = body_states.get(nm)
            if st is None:
                st = init_cavity_stream_state(group_scenes[0], sample_rate)
            group_states.append(st)

        # Stack drive signals: (B, T)
        group_sigs = torch.from_numpy(
            np.stack([source_signals[idx] for idx in indices])
        ).to(torch.complex128)

        chunk_out, new_states = render_batched_body_steps(
            group_scenes[0], group_scenes, group_sigs, group_states,
            sample_rate=sample_rate, device="cpu")

        # Write back per-performer
        for k, (idx, nm) in enumerate(members):
            body_states[nm] = new_states[k]
            body_out = chunk_out[k].detach().cpu().numpy()
            source_signals[idx, :len(body_out)] = body_out[:T]
        group_scenes = [body_scenes[nm] for nm in member_names]
        group_states = []
        for nm in member_names:
            st = body_states.get(nm)
            if st is None:
                st = init_cavity_stream_state(group_scenes[0], sample_rate)
            group_states.append(st)

        # Stack drive signals: (B, T)
        group_sigs = torch.from_numpy(
            np.stack([source_signals[idx] for idx in indices])
        ).to(torch.complex128)

        chunk_out, new_states = render_batched_body_steps(
            group_scenes[0], group_scenes, group_sigs, group_states,
            sample_rate=sample_rate, device="cpu")

        # Write back per-performer
        for k, (idx, nm) in enumerate(members):
            body_states[nm] = new_states[k]
            body_out = chunk_out[k].detach().cpu().numpy()
            source_signals[idx, :len(body_out)] = body_out[:T]

    # ── Section sympathetic coupling (chirp influence) ────────────────────────
    # Performers in the same part_key form a "clan".  Each member receives a
    # delayed, attenuated mix of its section-mates' signals — modelling the
    # sympathetic string-like mutual influence where nearby players feed on (or
    # fight) each other's chirp texture.  The delay combines physical propagation
    # between performer positions plus a tunable human reaction lag.
    section_coupling_gain = max(0.0, float(params.get("section_coupling_gain", 0.08)))
    section_reaction_lag_ms = max(0.0, float(params.get("section_reaction_lag_ms", 12.0)))

    if section_coupling_gain > 0.0 and len(performer_geo) > 1:
        # Group performer indices by part_key
        clans: dict[str, list[int]] = {}
        for i, pg in enumerate(performer_geo):
            pk = str(pg.get("part_key", ""))
            if pk:
                clans.setdefault(pk, []).append(i)

        for clan_key, members in clans.items():
            if len(members) < 2:
                continue
            # Pre-snapshot: save each member's current signal before mixing
            clan_signals = {i: source_signals[i].copy() for i in members if i < len(names)}

            for i in members:
                if i >= len(names):
                    continue
                pi = performer_geo[i]
                xi, yi = float(pi.get("x", 0.0)), float(pi.get("y", 0.0))
                sympathetic_sum = np.zeros(T, dtype=np.complex128)
                n_mates = 0

                for j in members:
                    if j == i or j >= len(names) or j not in clan_signals:
                        continue
                    pj = performer_geo[j]
                    xj, yj = float(pj.get("x", 0.0)), float(pj.get("y", 0.0))
                    dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                    # Total delay = propagation + reaction lag
                    prop_delay_ms = (dist / 343.0) * 1000.0
                    total_delay_ms = prop_delay_ms + section_reaction_lag_ms
                    delay_samples = int(round(total_delay_ms * 0.001 * sample_rate))

                    # Distance-based attenuation (inverse, soft-clamped)
                    atten = 1.0 / max(1.0, dist * 2.0)

                    mate_sig = clan_signals[j]
                    if delay_samples > 0 and delay_samples < T:
                        shifted = np.zeros(T, dtype=np.complex128)
                        shifted[delay_samples:] = mate_sig[:T - delay_samples]
                        sympathetic_sum += shifted * atten
                    elif delay_samples == 0:
                        sympathetic_sum += mate_sig * atten
                    # delay >= T means the mate's signal is entirely in the future
                    n_mates += 1

                if n_mates > 0:
                    # Normalize by mate count so coupling gain is independent of section size
                    source_signals[i] += sympathetic_sum * (section_coupling_gain / n_mates)

    scene: CavityScene | None = plugin_state.get("room_scene")
    _scene_n = plugin_state.get("room_scene_n_items")
    if scene is None or _scene_n != len(names):
        scene = _build_scene(params, len(names))
    stream_state = plugin_state.get("stream_state")
    if stream_state is None:
        stream_state = init_cavity_stream_state(scene, sample_rate)

    step_result = render_cavity_scene_step(
        scene,
        torch.from_numpy(source_signals),
        sample_rate=sample_rate,
        state=stream_state,
        device="cpu",
    )
    render = step_result.full_result
    next_state = step_result.state
    receiver = render.receiver_signals.detach().cpu().numpy()
    apertures = render.aperture_pressures.detach().cpu().numpy()

    n_recv = receiver.shape[0]
    # mic_left / mic_right always map to receiver[0] / receiver[1] (or mono dup)
    mic_l = receiver[0] if n_recv > 0 else np.zeros(T, dtype=np.complex128)
    mic_r = receiver[1] if n_recv > 1 else mic_l

    outputs: dict[str, dict[str, np.ndarray]] = {}
    scalar_state: dict[str, dict[str, float]] = {}
    for i, item in enumerate(names):
        drive = source_signals[i]
        ap = apertures[i] if i < apertures.shape[0] else np.zeros(T, dtype=np.complex128)
        fb = np.zeros(T, dtype=np.complex128)
        if i < source_signals.shape[0] and i < apertures.shape[0]:
            fb = ap * max(0.0, float(params.get("feedback_gain", 0.16))) * (1.0 - min(0.98, max(0.0, float(params.get("passive_loss", 0.48)))))
        outputs[item] = {
            "drive": drive,
            "aperture_pressure": ap,
            "feedback_pressure": fb,
            "mic_left": mic_l,
            "mic_right": mic_r,
        }
        scalar_state[item] = {
            "drive_rms": float(np.sqrt(np.mean(np.abs(drive) ** 2))) if drive.size else 0.0,
            "aperture_peak": float(np.max(np.abs(ap))) if ap.size else 0.0,
            "feedback_peak": float(np.max(np.abs(fb))) if fb.size else 0.0,
            "mic_peak": float(np.max(np.abs(receiver))) if receiver.size else 0.0,
        }

    log = (
        f"orchestral_resonance step: items={len(names)} "
        f"iters={scene.aperture_feedback_iterations} "
        f"band_split={scene.band_split_mode} "
        f"ap_peak={float(np.max(np.abs(apertures))) if apertures.size else 0.0:.4f}"
    )
    return {
        "outputs": outputs,
        "state": scalar_state,
        "plugin_state": {
            "stream_state": next_state,
            "body_states": body_states,
            "body_scenes": body_scenes,
            "room_scene": scene,
            "room_scene_n_items": len(names),
        },
        "log": log,
    }
