"""
Mic array and receiver preset types.

Local coordinate frame for all arrays:
    +X = right
    +Y = up
    +Z = forward (toward primary sound source / main lobe axis)

Polar patterns use real physics equations, not cosine-power approximations:
    omni          G = 1.0
    subcardioid   G = 0.7  + 0.3·cos θ     (wide, low-proximity)
    cardioid      G = 0.5  + 0.5·cos θ     (industry standard)
    supercardioid G = 0.366 + 0.634·cos θ  (~130° null, tighter)
    hypercardioid G = 0.25 + 0.75·cos θ    (~110° null, tightest lobe)
    figure8       G = |cos θ|               (bidirectional, phase-correct)
    ribbon        G = |cos θ|               (figure-8 + natural HF rolloff note)
    pinnae        G ≈ cardioid              (elevation shaping deferred to interaural module)

The interaural module (iau_* params) owns fine HRTF notching and elevation-
dependent coloring for pinnae receivers. MicArrayConfig.interaural_compatible
flags arrays that should be routed through that module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # late imports inside methods avoid circular deps with cavity_engine

# ─── Polar Pattern Constants ──────────────────────────────────────────────────

POLAR_OMNI          = "omni"
POLAR_SUBCARDIOID   = "subcardioid"
POLAR_CARDIOID      = "cardioid"
POLAR_SUPERCARDIOID = "supercardioid"
POLAR_HYPERCARDIOID = "hypercardioid"
POLAR_FIGURE8       = "figure8"
POLAR_RIBBON        = "ribbon"   # figure-8 + note: ribbon HF rolloff is mic-body physics not modeled here
POLAR_PINNAE        = "pinnae"   # approx cardioid; elevation shaping via interaural module

ALL_POLAR_PATTERNS = [
    POLAR_OMNI, POLAR_SUBCARDIOID, POLAR_CARDIOID, POLAR_SUPERCARDIOID,
    POLAR_HYPERCARDIOID, POLAR_FIGURE8, POLAR_RIBBON, POLAR_PINNAE,
]


def polar_gain(pattern: str, cos_angle: float) -> float:
    """Scalar polar gain for a given pattern. cos_angle = cos(θ), θ from on-axis."""
    c = max(-1.0, min(1.0, float(cos_angle)))
    if pattern == POLAR_OMNI:
        return 1.0
    if pattern == POLAR_SUBCARDIOID:
        return max(0.0, 0.7 + 0.3 * c)
    if pattern == POLAR_CARDIOID:
        return max(0.0, 0.5 + 0.5 * c)
    if pattern == POLAR_SUPERCARDIOID:
        return max(0.0, 0.366 + 0.634 * c)
    if pattern == POLAR_HYPERCARDIOID:
        return max(0.0, 0.25 + 0.75 * c)
    if pattern in (POLAR_FIGURE8, POLAR_RIBBON):
        return abs(c)
    if pattern == POLAR_PINNAE:
        return max(0.0, 0.5 + 0.5 * c)
    return max(0.0, 0.5 + 0.5 * c)


def pattern_to_directivity_power(pattern: str) -> float:
    """Map polar pattern name to approximate cosine-power for solvers that use the power model."""
    return {
        POLAR_OMNI:          0.0,
        POLAR_SUBCARDIOID:   0.4,
        POLAR_CARDIOID:      1.0,
        POLAR_SUPERCARDIOID: 1.8,
        POLAR_HYPERCARDIOID: 2.8,
        POLAR_FIGURE8:       1.0,
        POLAR_RIBBON:        1.0,
        POLAR_PINNAE:        1.2,
    }.get(pattern, 1.0)


# ─── Element & Baffle Types ───────────────────────────────────────────────────

@dataclass
class MicElement:
    """
    One capsule within a mic array, expressed in local array coordinates.

    local_position:  (right, up, forward) offset from array center in meters
    local_direction: unit vector (right, up, forward) — main lobe axis
    rotation_deg:    extra in-plane yaw around local +Y (positive = rotate right)
    """
    key: str
    mic_type: str           # "condenser" | "ribbon" | "dynamic" | "pinnae_sim" | ...
    polar_pattern: str      # one of the POLAR_* constants
    local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_deg: float = 0.0
    sensitivity_db: float = 0.0    # relative to array reference level
    self_noise_db: float = 20.0    # A-weighted capsule noise floor
    max_spl_db: float = 135.0      # SPL at onset of overload
    frequency_response: list[tuple[float, float]] = field(default_factory=list)  # [(hz, db), ...]
    directivity_power_override: float | None = None  # None = use polar_pattern physics


@dataclass
class BaffleSpec:
    """
    An acoustic baffle in array-local coordinates.

    local_position: (right, up, forward) center offset from array center
    local_normal:   (right, up, forward) unit normal — which side the baffle faces
    width_m, height_m: physical extent (informational for rendering; solver uses panel)
    """
    key: str
    local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    width_m: float = 0.18
    height_m: float = 0.23
    reflectivity: float = 0.88
    normal_reflectivity: float = 0.93
    grazing_reflectivity: float = 0.70
    absorption: float = 0.07
    diffusion: float = 0.12
    normal_phase_rad: float = 0.05
    grazing_phase_rad: float = 0.25
    material_key: str = "rigid"


# ─── Array Config ─────────────────────────────────────────────────────────────

@dataclass
class MicArrayConfig:
    """
    Complete mic array descriptor.  Arrays are positioned and oriented at runtime
    via to_cavity_receivers() / to_cavity_baffles().

    interaural_compatible: True for binaural/pinnae arrays that should be routed
        through the interaural module (iau_* params) for HRTF elevation shaping.
        The interaural module owns fine notching; this array owns gross directionality.
    """
    key: str
    label: str
    description: str
    array_pattern: str           # "binaural" | "xy_stereo" | "ortf" | "ab_pair" |
                                 # "decca_tree" | "ribbon_6" | "ambisonic_foa" | ...
    elements: list[MicElement]
    baffles: list[BaffleSpec]
    output_channels: list[str]   # e.g. ["left", "right"] or ["W", "X", "Y", "Z"]
    interaural_compatible: bool = False

    def receiver_keys(self) -> list[str]:
        return [e.key for e in self.elements]

    def to_cavity_receivers(
        self,
        center_position: tuple[float, float, float] = (0.0, 0.0, 1.5),
        forward_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
        up_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> list:
        """Build CavityReceiver list placed at center_position facing forward_direction."""
        from cavity_engine import CavityReceiver
        R, U, F = _build_local_frame(forward_direction, up_direction)
        receivers = []
        for el in self.elements:
            ld = _apply_yaw(el.local_direction, el.rotation_deg)
            world_pos = _local_to_world(el.local_position, center_position, R, U, F)
            world_dir = _rotate_vec(ld, R, U, F)
            dp = el.directivity_power_override
            if dp is None:
                dp = pattern_to_directivity_power(el.polar_pattern)
            receivers.append(CavityReceiver(
                key=el.key,
                position=world_pos,
                direction=world_dir,
                directivity_power=dp,
                polar_pattern=el.polar_pattern,
            ))
        return receivers

    def to_cavity_baffles(
        self,
        center_position: tuple[float, float, float] = (0.0, 0.0, 1.5),
        forward_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
        up_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> list:
        """Build CavityPanel list for array baffles."""
        from cavity_engine import CavityPanel
        R, U, F = _build_local_frame(forward_direction, up_direction)
        panels = []
        for b in self.baffles:
            world_pos = _local_to_world(b.local_position, center_position, R, U, F)
            world_normal = _rotate_vec(b.local_normal, R, U, F)
            panels.append(CavityPanel(
                key=b.key,
                point=world_pos,
                normal=world_normal,
                reflectivity=b.reflectivity,
                normal_reflectivity=b.normal_reflectivity,
                grazing_reflectivity=b.grazing_reflectivity,
                absorption=b.absorption,
                diffusion=b.diffusion,
                normal_phase_rad=b.normal_phase_rad,
                grazing_phase_rad=b.grazing_phase_rad,
                is_baffle=True,
            ))
        return panels


# ─── Array Registry ───────────────────────────────────────────────────────────

_REGISTRY: dict[str, MicArrayConfig] = {}


def register_array(cfg: MicArrayConfig) -> MicArrayConfig:
    _REGISTRY[cfg.key] = cfg
    return cfg


def get_array(key: str) -> MicArrayConfig | None:
    return _REGISTRY.get(key)


def list_arrays() -> list[str]:
    return list(_REGISTRY.keys())


def array_catalog() -> list[dict]:
    return [
        {
            "key": cfg.key,
            "label": cfg.label,
            "description": cfg.description,
            "pattern": cfg.array_pattern,
            "channels": list(cfg.output_channels),
            "elements": len(cfg.elements),
            "baffles": len(cfg.baffles),
            "interaural_compatible": cfg.interaural_compatible,
        }
        for cfg in _REGISTRY.values()
    ]


# ─── Geometry Helpers ─────────────────────────────────────────────────────────

def _build_local_frame(
    forward: tuple[float, float, float],
    up: tuple[float, float, float],
) -> tuple[tuple, tuple, tuple]:
    """Return orthonormal (right, up, forward) frame."""
    fx, fy, fz = forward
    fn = math.sqrt(fx*fx + fy*fy + fz*fz) or 1.0
    fx, fy, fz = fx/fn, fy/fn, fz/fn

    ux, uy, uz = up
    # right = forward × up
    rx = fy*uz - fz*uy
    ry = fz*ux - fx*uz
    rz = fx*uy - fy*ux
    rn = math.sqrt(rx*rx + ry*ry + rz*rz)
    if rn < 1e-9:
        # degenerate: up and forward aligned, pick arbitrary perpendicular
        ux2, uy2, uz2 = (1.0, 0.0, 0.0) if abs(fx) < 0.9 else (0.0, 1.0, 0.0)
        rx = fy*uz2 - fz*uy2
        ry = fz*ux2 - fx*uz2
        rz = fx*uy2 - fy*ux2
        rn = math.sqrt(rx*rx + ry*ry + rz*rz) or 1.0
    rx, ry, rz = rx/rn, ry/rn, rz/rn
    # recompute up = right × forward
    ux = ry*fz - rz*fy
    uy = rz*fx - rx*fz
    uz = rx*fy - ry*fx
    un = math.sqrt(ux*ux + uy*uy + uz*uz) or 1.0
    ux, uy, uz = ux/un, uy/un, uz/un
    return (rx, ry, rz), (ux, uy, uz), (fx, fy, fz)


def _local_to_world(
    local: tuple[float, float, float],
    center: tuple[float, float, float],
    R: tuple, U: tuple, F: tuple,
) -> tuple[float, float, float]:
    lx, ly, lz = local
    cx, cy, cz = center
    return (
        cx + lx*R[0] + ly*U[0] + lz*F[0],
        cy + lx*R[1] + ly*U[1] + lz*F[1],
        cz + lx*R[2] + ly*U[2] + lz*F[2],
    )


def _rotate_vec(
    local: tuple[float, float, float],
    R: tuple, U: tuple, F: tuple,
) -> tuple[float, float, float]:
    lx, ly, lz = local
    wx = lx*R[0] + ly*U[0] + lz*F[0]
    wy = lx*R[1] + ly*U[1] + lz*F[1]
    wz = lx*R[2] + ly*U[2] + lz*F[2]
    n = math.sqrt(wx*wx + wy*wy + wz*wz) or 1.0
    return (wx/n, wy/n, wz/n)


def _apply_yaw(
    direction: tuple[float, float, float],
    yaw_deg: float,
) -> tuple[float, float, float]:
    """Rotate direction around local +Y (up) by yaw_deg degrees."""
    if abs(yaw_deg) < 1e-6:
        return direction
    a = math.radians(yaw_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    dx, dy, dz = direction
    # rotate in XZ plane (around Y)
    return (dx*cos_a + dz*sin_a, dy, -dx*sin_a + dz*cos_a)
