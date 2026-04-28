"""Torch cavity / room renderer with first-class diffusion.

This module is intentionally decoupled from instrument resonators. It renders a
shared geometric cavity given:

- complex driver signals from one or many sources
- a room geometry (polygonal prism or circular/polar room)
- optional baffle panels
- receiver positions / orientations (mono, stereo, or arbitrary arrays)

The render path is hybrid but fully deterministic and fully complex:

1. direct propagation
2. first-order panel reflections via image sources
3. mandatory diffuse tail accumulation

Everything remains in the complex domain until the caller explicitly projects to
real or writes to audio.
"""

from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch


_FloatTensor = torch.Tensor
_ComplexTensor = torch.Tensor


@dataclass
class CavitySourceBand:
    low_hz: float = 0.0
    high_hz: float | None = None
    cone_angle_deg: float = 180.0
    directivity_power: float = 1.0
    gain: float = 1.0


@dataclass
class CavitySource:
    key: str
    position: tuple[float, float, float]
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    directivity_power: float = 1.0
    gain: float = 1.0
    band_profiles: list[CavitySourceBand] = field(default_factory=list)


@dataclass
class CavityReceiver:
    key: str
    position: tuple[float, float, float]
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    directivity_power: float = 0.0
    # Real polar pattern physics — overrides directivity_power when set.
    # Values: "omni" | "subcardioid" | "cardioid" | "supercardioid" |
    #         "hypercardioid" | "figure8" | "ribbon" | "pinnae"
    # "omni" keeps the existing power-law path (directivity_power=0 → flat).
    polar_pattern: str = "omni"


@dataclass
class CavityAperture:
    key: str
    source_key: str
    position: tuple[float, float, float]
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    capture_power: float = 0.0
    feedback_gain: float = 0.18
    feedback_phase_rad: float = 0.0
    passive_loss: float = 0.45


@dataclass
class CavityPanel:
    key: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    reflectivity: float = 0.7
    normal_reflectivity: float | None = None
    grazing_reflectivity: float | None = None
    normal_phase_rad: float = 0.0
    grazing_phase_rad: float = 0.0
    diffusion: float = 0.35
    absorption: float = 0.15
    is_baffle: bool = False
    half_size: float = 1.0
    half_h: float | None = None   # extent along ax1 (height for ring panels); None → half_size
    half_w: float | None = None   # extent along ax2 (arc-tangent for ring panels); None → half_size
    material_mask: np.ndarray | None = field(default=None, repr=False)
    # float32 (H, W) density map: 1.0 = solid material, 0.0 = open/void.
    # Cells with mask == 0 generate no triangles in the ray tracer; intermediate
    # values scale the material properties (reflectivity, diffusion, absorption)
    # continuously, which handles things like bracing density, grain variation,
    # and partial openings within a single panel without needing extra panels.


@dataclass
class SceneMaterial:
    key: str
    reflectivity: float = 0.7
    normal_reflectivity: float | None = None
    grazing_reflectivity: float | None = None
    normal_phase_rad: float = 0.0
    grazing_phase_rad: float = 0.0
    diffusion: float = 0.35
    absorption: float = 0.15


@dataclass
class SceneTriangle:
    key: str
    vertices: tuple[int, int, int]
    material_key: str
    back_material_key: str | None = None
    winding: Literal["ccw", "cw"] = "ccw"
    is_baffle: bool = False


@dataclass
class SceneEdge:
    key: str
    vertices: tuple[int, int]


@dataclass
class MeshRoom:
    vertices: list[tuple[float, float, float]]
    triangles: list[SceneTriangle]
    materials: dict[str, SceneMaterial]
    edges: list[SceneEdge] = field(default_factory=list)


@dataclass
class AtmosphericSpec:
    temperature_c: float = 20.0
    humidity_rel: float = 0.5
    pressure_kpa: float = 101.325
    attenuation_db_per_m: float = 0.01
    high_band_extra_db_per_m: float = 0.02


@dataclass
class PolygonalRoom:
    vertices_xy: list[tuple[float, float]]
    height: float = 3.0
    closed_floor: bool = True
    closed_roof: bool = True
    wall_reflectivity: float = 0.7
    wall_diffusion: float = 0.35
    wall_absorption: float = 0.15
    floor_reflectivity: float = 0.75
    floor_diffusion: float = 0.25
    floor_absorption: float = 0.10
    roof_reflectivity: float = 0.6
    roof_diffusion: float = 0.45
    roof_absorption: float = 0.18
    baffles: list[CavityPanel] = field(default_factory=list)


@dataclass
class CircularRoom:
    radius: float
    height: float = 3.0
    n_segments: int = 24
    closed_floor: bool = True
    closed_roof: bool = True
    wall_reflectivity: float = 0.72
    wall_diffusion: float = 0.40
    wall_absorption: float = 0.14
    floor_reflectivity: float = 0.75
    floor_diffusion: float = 0.25
    floor_absorption: float = 0.10
    roof_reflectivity: float = 0.58
    roof_diffusion: float = 0.50
    roof_absorption: float = 0.20
    baffles: list[CavityPanel] = field(default_factory=list)


@dataclass
class DiffuseTailSpec:
    taps_per_panel: int = 6
    decay_s: float = 1.4
    delay_spread_s: float = 0.08
    strength: float = 0.35
    truncate_at_s: float = 3.5


@dataclass
class CavityScene:
    sources: list[CavitySource]
    receivers: list[CavityReceiver]
    geometry: PolygonalRoom | CircularRoom | MeshRoom
    apertures: list[CavityAperture] = field(default_factory=list)
    diffuse_tail: DiffuseTailSpec = field(default_factory=DiffuseTailSpec)
    atmosphere: AtmosphericSpec = field(default_factory=AtmosphericSpec)
    speed_of_sound_m_s: float = 343.0
    band_split_mode: Literal["fir", "fft"] = "fir"
    band_split_fir_taps: int = 129
    aperture_feedback_iterations: int = 0


@dataclass
class CavityRenderResult:
    receiver_signals: _ComplexTensor
    direct_signals: _ComplexTensor
    specular_signals: _ComplexTensor
    diffuse_signals: _ComplexTensor
    aperture_pressures: _ComplexTensor
    panel_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class CavityStreamState:
    """Persistent overlap state for chunked cavity rendering."""

    overlap: _ComplexTensor
    tail_samples: int
    source_history: _ComplexTensor
    history_samples: int


@dataclass
class CavityStepResult:
    """One streaming solve step plus state update."""

    chunk_output: _ComplexTensor
    state: CavityStreamState
    full_result: CavityRenderResult


def _as_float_tensor(rows: list[tuple[float, float, float]], device: torch.device) -> _FloatTensor:
    if not rows:
        return torch.zeros((0, 3), dtype=torch.float64, device=device)
    return torch.tensor(rows, dtype=torch.float64, device=device)


def _normalize(v: _FloatTensor, eps: float = 1e-9) -> _FloatTensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _polygon_signed_area(vertices_xy: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(vertices_xy)
    for i in range(n):
        x0, y0 = vertices_xy[i]
        x1, y1 = vertices_xy[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def build_polygonal_panels(room: PolygonalRoom) -> list[CavityPanel]:
    verts = room.vertices_xy
    if len(verts) < 3:
        raise ValueError("PolygonalRoom requires at least 3 vertices")
    ccw = _polygon_signed_area(verts) > 0.0
    panels: list[CavityPanel] = []
    for i, (a, b) in enumerate(zip(verts, verts[1:] + verts[:1])):
        x0, y0 = a
        x1, y1 = b
        edge = (x1 - x0, y1 - y0)
        normal_xy = (edge[1], -edge[0]) if ccw else (-edge[1], edge[0])
        nrm = math.hypot(normal_xy[0], normal_xy[1]) or 1.0
        panels.append(CavityPanel(
            key=f"wall_{i}",
            point=(x0, y0, room.height * 0.5),
            normal=(normal_xy[0] / nrm, normal_xy[1] / nrm, 0.0),
            reflectivity=room.wall_reflectivity,
            diffusion=room.wall_diffusion,
            absorption=room.wall_absorption,
        ))
    if room.closed_floor:
        panels.append(CavityPanel(
            key="floor",
            point=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            reflectivity=room.floor_reflectivity,
            diffusion=room.floor_diffusion,
            absorption=room.floor_absorption,
        ))
    if room.closed_roof:
        panels.append(CavityPanel(
            key="roof",
            point=(0.0, 0.0, room.height),
            normal=(0.0, 0.0, -1.0),
            reflectivity=room.roof_reflectivity,
            diffusion=room.roof_diffusion,
            absorption=room.roof_absorption,
        ))
    panels.extend(room.baffles)
    return panels


def build_circular_panels(room: CircularRoom) -> list[CavityPanel]:
    if room.radius <= 0.0:
        raise ValueError("CircularRoom radius must be positive")
    n_segments = max(4, int(room.n_segments))
    panels: list[CavityPanel] = []
    for i in range(n_segments):
        ang = 2.0 * math.pi * (i / n_segments)
        x = room.radius * math.cos(ang)
        y = room.radius * math.sin(ang)
        panels.append(CavityPanel(
            key=f"wall_{i}",
            point=(x, y, room.height * 0.5),
            normal=(-math.cos(ang), -math.sin(ang), 0.0),
            reflectivity=room.wall_reflectivity,
            diffusion=room.wall_diffusion,
            absorption=room.wall_absorption,
        ))
    if room.closed_floor:
        panels.append(CavityPanel(
            key="floor",
            point=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            reflectivity=room.floor_reflectivity,
            diffusion=room.floor_diffusion,
            absorption=room.floor_absorption,
        ))
    if room.closed_roof:
        panels.append(CavityPanel(
            key="roof",
            point=(0.0, 0.0, room.height),
            normal=(0.0, 0.0, -1.0),
            reflectivity=room.roof_reflectivity,
            diffusion=room.roof_diffusion,
            absorption=room.roof_absorption,
        ))
    panels.extend(room.baffles)
    return panels


def _triangle_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    winding: Literal["ccw", "cw"],
) -> tuple[float, float, float]:
    ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    if winding == "cw":
        nx, ny, nz = -nx, -ny, -nz
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / nrm, ny / nrm, nz / nrm)


def _panel_from_material(
    key: str,
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
    material: SceneMaterial,
    *,
    is_baffle: bool = False,
) -> CavityPanel:
    return CavityPanel(
        key=key,
        point=point,
        normal=normal,
        reflectivity=material.reflectivity,
        normal_reflectivity=material.normal_reflectivity,
        grazing_reflectivity=material.grazing_reflectivity,
        normal_phase_rad=material.normal_phase_rad,
        grazing_phase_rad=material.grazing_phase_rad,
        diffusion=material.diffusion,
        absorption=material.absorption,
        is_baffle=is_baffle,
    )


def build_mesh_panels(room: MeshRoom) -> list[CavityPanel]:
    panels: list[CavityPanel] = []
    for tri in room.triangles:
        ia, ib, ic = tri.vertices
        a = room.vertices[ia]
        b = room.vertices[ib]
        c = room.vertices[ic]
        centroid = (
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        )
        normal = _triangle_normal(a, b, c, tri.winding)
        front = room.materials[tri.material_key]
        panels.append(_panel_from_material(
            key=tri.key,
            point=centroid,
            normal=normal,
            material=front,
            is_baffle=tri.is_baffle,
        ))
        back_key = tri.back_material_key or tri.material_key
        back = room.materials[back_key]
        panels.append(_panel_from_material(
            key=f"{tri.key}:back",
            point=centroid,
            normal=(-normal[0], -normal[1], -normal[2]),
            material=back,
            is_baffle=tri.is_baffle,
        ))
    return panels


def build_room_panels(geometry: PolygonalRoom | CircularRoom | MeshRoom) -> list[CavityPanel]:
    if isinstance(geometry, PolygonalRoom):
        return build_polygonal_panels(geometry)
    if isinstance(geometry, CircularRoom):
        return build_circular_panels(geometry)
    if isinstance(geometry, MeshRoom):
        return build_mesh_panels(geometry)
    raise TypeError("Unsupported geometry type")


def _fractional_delay_kernel(
    signal: _ComplexTensor,
    delays_s: _FloatTensor,
    sample_rate: float,
) -> _ComplexTensor:
    """Apply batched fractional delays to one source signal.

    signal:
        shape (t,)
    delays_s:
        shape (...,)
    returns:
        shape (..., t)
    """
    n = signal.shape[-1]
    idx = torch.arange(n, dtype=torch.float64, device=signal.device)
    sample_pos = idx.view(*((1,) * delays_s.ndim), n) - delays_s[..., None] * sample_rate
    lo = torch.floor(sample_pos).to(torch.int64)
    hi = lo + 1
    frac = (sample_pos - lo.to(torch.float64)).to(signal.real.dtype)
    valid_lo = (lo >= 0) & (lo < n)
    valid_hi = (hi >= 0) & (hi < n)
    lo_safe = lo.clamp(0, max(n - 1, 0))
    hi_safe = hi.clamp(0, max(n - 1, 0))
    sig_expand = signal.view(*((1,) * delays_s.ndim), n)
    lo_vals = torch.gather(sig_expand.expand(*delays_s.shape, n), -1, lo_safe)
    hi_vals = torch.gather(sig_expand.expand(*delays_s.shape, n), -1, hi_safe)
    out = torch.where(valid_lo, (1.0 - frac) * lo_vals, torch.zeros_like(lo_vals))
    out = out + torch.where(valid_hi, frac * hi_vals, torch.zeros_like(hi_vals))
    return out


def _source_directivity_gain(
    source_dirs: _FloatTensor,
    source_to_target_dirs: _FloatTensor,
    power: _FloatTensor,
    cone_angle_deg: _FloatTensor | None = None,
) -> _FloatTensor:
    cosang = (source_dirs[:, None, :] * source_to_target_dirs).sum(dim=-1).clamp_min(0.0)
    gain = cosang ** power[:, None]
    if cone_angle_deg is None:
        return gain
    half_angle = 0.5 * torch.deg2rad(cone_angle_deg.clamp_min(1e-6))
    cosine_edge = torch.cos(half_angle)[:, None]
    wide = cone_angle_deg[:, None] >= 180.0
    cone_gain = torch.where(
        wide,
        torch.ones_like(gain),
        ((cosang - cosine_edge).clamp_min(0.0) / (1.0 - cosine_edge).clamp_min(1e-9)),
    )
    return gain * cone_gain


def _receiver_directivity_gain(
    recv_dirs: _FloatTensor,
    incoming_dirs: _FloatTensor,
    power: _FloatTensor,
    patterns: list[str] | None = None,
) -> _FloatTensor:
    # cosang: (n_sources_or_paths, n_receivers)
    cosang = (recv_dirs[None, :, :] * incoming_dirs).sum(dim=-1)
    if not patterns or all(p in ("omni", "") for p in patterns):
        return cosang.clamp_min(0.0) ** power[None, :]
    n_recv = recv_dirs.shape[0]
    gains: list[_FloatTensor] = []
    for i in range(n_recv):
        c = cosang[:, i]
        pat = patterns[i] if i < len(patterns) else "omni"
        if pat == "omni":
            gains.append(torch.ones_like(c))
        elif pat == "subcardioid":
            gains.append((0.7 + 0.3 * c).clamp_min(0.0))
        elif pat == "cardioid":
            gains.append(((1.0 + c) * 0.5).clamp_min(0.0))
        elif pat == "supercardioid":
            gains.append((0.366 + 0.634 * c).clamp_min(0.0))
        elif pat == "hypercardioid":
            gains.append((0.25 + 0.75 * c).clamp_min(0.0))
        elif pat in ("figure8", "ribbon"):
            gains.append(c.abs())
        elif pat == "pinnae":
            # Gross directionality same as cardioid; fine elevation shaping
            # is owned by the interaural module (iau_elevation parameter).
            gains.append(((1.0 + c) * 0.5).clamp_min(0.0))
        else:
            gains.append(c.clamp_min(0.0) ** power[i])
    return torch.stack(gains, dim=1)  # (n_paths, n_receivers)


def _split_source_signal_bands(
    signal: _ComplexTensor,
    band_profiles: list[CavitySourceBand],
    sample_rate: float,
) -> list[tuple[CavitySourceBand, _ComplexTensor]]:
    if not band_profiles:
        return [(CavitySourceBand(low_hz=0.0, high_hz=None, cone_angle_deg=180.0, directivity_power=1.0, gain=1.0), signal)]
    if signal.ndim != 1:
        raise ValueError("signal must have shape (n_samples,)")
    n = signal.shape[0]
    if n == 0:
        return [(band, signal.clone()) for band in band_profiles]
    spec = torch.fft.fft(signal)
    freqs = torch.fft.fftfreq(n, d=1.0 / max(sample_rate, 1e-9)).to(signal.device).abs()
    out: list[tuple[CavitySourceBand, _ComplexTensor]] = []
    for band in band_profiles:
        lo = max(0.0, float(band.low_hz))
        hi = float(band.high_hz) if band.high_hz is not None else None
        mask = freqs >= lo
        if hi is not None:
            mask = mask & (freqs < hi)
        masked = spec * mask.to(spec.dtype)
        out.append((band, torch.fft.ifft(masked)))
    return out


def _bandpass_kernel(
    band: CavitySourceBand,
    sample_rate: float,
    taps: int,
    device: torch.device,
) -> _FloatTensor:
    taps = max(3, int(taps) | 1)
    nyquist = max(sample_rate * 0.5, 1e-9)
    lo = max(0.0, float(band.low_hz)) / nyquist
    hi = 1.0 if band.high_hz is None else min(1.0, max(0.0, float(band.high_hz)) / nyquist)
    center = (taps - 1) * 0.5
    n = torch.arange(taps, dtype=torch.float64, device=device) - center
    window = torch.hann_window(taps, periodic=False, dtype=torch.float64, device=device)
    if hi <= 0.0 or lo >= 1.0 or hi <= lo:
        return torch.zeros((taps,), dtype=torch.float64, device=device)

    def lowpass(norm_cutoff: float) -> _FloatTensor:
        return norm_cutoff * torch.sinc(norm_cutoff * n)

    if lo <= 0.0:
        kernel = lowpass(hi)
        ref_norm = min(0.5 * hi, 0.5)
    elif hi >= 1.0:
        impulse = torch.zeros((taps,), dtype=torch.float64, device=device)
        impulse[int(center)] = 1.0
        kernel = impulse - lowpass(lo)
        ref_norm = min(0.5 * (lo + 1.0), 0.95)
    else:
        kernel = lowpass(hi) - lowpass(lo)
        ref_norm = 0.5 * (lo + hi)
    kernel = kernel * window
    phase_ref = torch.exp(-1j * math.pi * ref_norm * n.to(torch.complex128))
    response = torch.abs((kernel.to(torch.complex128) * phase_ref).sum()).clamp_min(1e-12)
    return kernel / response.to(torch.float64)


def _convolve_same_complex(signal: _ComplexTensor, kernel: _FloatTensor) -> _ComplexTensor:
    if signal.ndim != 1:
        raise ValueError("signal must have shape (n_samples,)")
    taps = int(kernel.shape[0])
    weight = kernel.flip(0).view(1, 1, taps).to(signal.real.dtype)
    real_in = torch.nn.functional.pad(signal.real.view(1, 1, -1), (taps - 1, 0))
    imag_in = torch.nn.functional.pad(signal.imag.view(1, 1, -1), (taps - 1, 0))
    real = torch.nn.functional.conv1d(real_in, weight).view(-1)
    imag = torch.nn.functional.conv1d(imag_in, weight).view(-1)
    return torch.complex(real, imag).to(torch.complex128)


def _split_source_signal_bands_fir(
    signal: _ComplexTensor,
    band_profiles: list[CavitySourceBand],
    sample_rate: float,
    taps: int,
) -> list[tuple[CavitySourceBand, _ComplexTensor]]:
    if not band_profiles:
        return [(CavitySourceBand(low_hz=0.0, high_hz=None, cone_angle_deg=180.0, directivity_power=1.0, gain=1.0), signal)]
    return [
        (band, _convolve_same_complex(signal, _bandpass_kernel(band, sample_rate, taps, signal.device)))
        for band in band_profiles
    ]


def _source_bands_for_scene(
    signal: _ComplexTensor,
    source: CavitySource,
    scene: CavityScene,
    sample_rate: float,
    *,
    filter_history: _ComplexTensor | None = None,
) -> list[tuple[CavitySourceBand, _ComplexTensor]]:
    if scene.band_split_mode == "fft":
        return _split_source_signal_bands(signal, source.band_profiles, sample_rate)
    if filter_history is None or filter_history.numel() == 0:
        return _split_source_signal_bands_fir(signal, source.band_profiles, sample_rate, scene.band_split_fir_taps)
    extended = torch.cat([filter_history.to(signal.device), signal], dim=0)
    return [
        (band, band_sig[-signal.shape[0]:].clone())
        for band, band_sig in _split_source_signal_bands_fir(
            extended,
            source.band_profiles,
            sample_rate,
            scene.band_split_fir_taps,
        )
    ]


def _scene_history_samples(scene: CavityScene) -> int:
    if scene.band_split_mode != "fir":
        return 0
    if not any(source.band_profiles for source in scene.sources):
        return 0
    return max(0, int(max(3, int(scene.band_split_fir_taps) | 1) - 1))


def atmospheric_attenuation(
    atmosphere: AtmosphericSpec,
    distances_m: _FloatTensor,
    *,
    high_band_weight: float = 0.0,
) -> _FloatTensor:
    temp_scale = 1.0 + 0.003 * (float(atmosphere.temperature_c) - 20.0)
    humidity_scale = 1.0 + 0.6 * max(0.0, min(1.0, float(atmosphere.humidity_rel)))
    pressure_scale = max(0.5, float(atmosphere.pressure_kpa) / 101.325)
    db_per_m = max(0.0, float(atmosphere.attenuation_db_per_m))
    db_per_m += max(0.0, float(atmosphere.high_band_extra_db_per_m)) * max(0.0, float(high_band_weight))
    nepers_per_m = (db_per_m / 20.0) * math.log(10.0) * temp_scale * humidity_scale / pressure_scale
    return torch.exp(-nepers_per_m * distances_m.to(torch.float64))


def build_binaural_receivers(
    center_position: tuple[float, float, float],
    forward_direction: tuple[float, float, float] = (0.0, -1.0, 0.0),
    *,
    ear_spacing_m: float = 0.18,
    directivity_power: float = 1.0,
) -> list[CavityReceiver]:
    fx, fy, fz = forward_direction
    right = (fy, -fx, 0.0)
    nrm = math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]) or 1.0
    rx, ry, rz = (right[0] / nrm, right[1] / nrm, right[2] / nrm)
    offset = 0.5 * float(ear_spacing_m)
    cx, cy, cz = center_position
    return [
        CavityReceiver(
            key="ear_l",
            position=(cx - rx * offset, cy - ry * offset, cz - rz * offset),
            direction=forward_direction,
            directivity_power=directivity_power,
        ),
        CavityReceiver(
            key="ear_r",
            position=(cx + rx * offset, cy + ry * offset, cz + rz * offset),
            direction=forward_direction,
            directivity_power=directivity_power,
        ),
    ]


def load_obj_mesh_room(
    path: str,
    *,
    front_material: SceneMaterial | None = None,
    back_material: SceneMaterial | None = None,
) -> MeshRoom:
    verts: list[tuple[float, float, float]] = []
    tris: list[SceneTriangle] = []
    materials = {
        "front": front_material or SceneMaterial(key="front"),
        "back": back_material or (front_material or SceneMaterial(key="front")),
    }
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                verts.append((float(xs), float(ys), float(zs)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idxs) < 3:
                    continue
                for i in range(1, len(idxs) - 1):
                    tris.append(SceneTriangle(
                        key=f"tri_{len(tris)}",
                        vertices=(idxs[0], idxs[i], idxs[i + 1]),
                        material_key="front",
                        back_material_key="back",
                        winding="ccw",
                    ))
    return MeshRoom(vertices=verts, triangles=tris, materials=materials)


def _source_index_by_key(scene: CavityScene) -> dict[str, int]:
    return {source.key: idx for idx, source in enumerate(scene.sources)}


def _get_cached_scene_tensors(
    scene: CavityScene,
    dev: torch.device,
) -> tuple[
    list,           # panels
    _FloatTensor,   # src_pos
    _FloatTensor,   # src_dir
    _FloatTensor,   # panel_points
    _FloatTensor,   # panel_normals
    _FloatTensor,   # panel_reflect
    _FloatTensor,   # panel_normal_reflect
    _FloatTensor,   # panel_grazing_reflect
    _FloatTensor,   # panel_normal_phase
    _FloatTensor,   # panel_grazing_phase
    _FloatTensor,   # panel_diffuse
    _FloatTensor,   # panel_absorb
]:
    """Build panel + source tensors once and cache on the scene object."""
    cache = getattr(scene, "_tensor_cache", None)
    if cache is not None and cache.get("device") == dev:
        c = cache
        return (
            c["panels"], c["src_pos"], c["src_dir"],
            c["panel_points"], c["panel_normals"], c["panel_reflect"],
            c["panel_normal_reflect"], c["panel_grazing_reflect"],
            c["panel_normal_phase"], c["panel_grazing_phase"],
            c["panel_diffuse"], c["panel_absorb"],
        )

    panels = build_room_panels(scene.geometry)
    src_pos = _as_float_tensor([s.position for s in scene.sources], dev)
    src_dir = _normalize(_as_float_tensor([s.direction for s in scene.sources], dev))
    panel_points = _as_float_tensor([p.point for p in panels], dev)
    panel_normals = _normalize(_as_float_tensor([p.normal for p in panels], dev))
    panel_reflect = torch.tensor([p.reflectivity for p in panels], dtype=torch.float64, device=dev)
    panel_normal_reflect = torch.tensor(
        [p.normal_reflectivity if p.normal_reflectivity is not None else -1.0 for p in panels],
        dtype=torch.float64, device=dev,
    )
    panel_grazing_reflect = torch.tensor(
        [p.grazing_reflectivity if p.grazing_reflectivity is not None else -1.0 for p in panels],
        dtype=torch.float64, device=dev,
    )
    panel_normal_phase = torch.tensor([p.normal_phase_rad for p in panels], dtype=torch.float64, device=dev)
    panel_grazing_phase = torch.tensor([p.grazing_phase_rad for p in panels], dtype=torch.float64, device=dev)
    panel_diffuse = torch.tensor([p.diffusion for p in panels], dtype=torch.float64, device=dev)
    panel_absorb = torch.tensor([p.absorption for p in panels], dtype=torch.float64, device=dev)

    cache = {
        "device": dev,
        "panels": panels,
        "src_pos": src_pos, "src_dir": src_dir,
        "panel_points": panel_points, "panel_normals": panel_normals,
        "panel_reflect": panel_reflect,
        "panel_normal_reflect": panel_normal_reflect,
        "panel_grazing_reflect": panel_grazing_reflect,
        "panel_normal_phase": panel_normal_phase,
        "panel_grazing_phase": panel_grazing_phase,
        "panel_diffuse": panel_diffuse,
        "panel_absorb": panel_absorb,
    }
    # Safe to stash on a frozen-ish dataclass via object.__setattr__
    try:
        scene._tensor_cache = cache
    except (AttributeError, TypeError):
        object.__setattr__(scene, "_tensor_cache", cache)
    return (
        panels, src_pos, src_dir,
        panel_points, panel_normals, panel_reflect,
        panel_normal_reflect, panel_grazing_reflect,
        panel_normal_phase, panel_grazing_phase,
        panel_diffuse, panel_absorb,
    )


# ---------------------------------------------------------------------------
# Batched body-cavity solver — same-type bodies, jitter in tensor form
# ---------------------------------------------------------------------------

def _batched_fractional_delay(
    signals: _ComplexTensor,
    delays_s: _FloatTensor,
    sample_rate: float,
) -> _ComplexTensor:
    """Delay B different signals by B-batched delay schedules.

    signals  : (B, T)
    delays_s : (B, ...)     — arbitrary extra dims after the batch dim
    returns  : (B, ..., T)
    """
    B, T = signals.shape
    n_extra = delays_s.ndim - 1
    dev = signals.device

    idx = torch.arange(T, dtype=torch.float64, device=dev)
    idx_shape = [1] * (1 + n_extra) + [T]
    sample_pos = idx.view(*idx_shape) - delays_s[..., None] * sample_rate

    lo = torch.floor(sample_pos).to(torch.int64)
    hi = lo + 1
    frac = (sample_pos - lo.to(torch.float64)).to(signals.real.dtype)
    valid_lo = (lo >= 0) & (lo < T)
    valid_hi = (hi >= 0) & (hi < T)
    lo_safe = lo.clamp(0, max(T - 1, 0))
    hi_safe = hi.clamp(0, max(T - 1, 0))

    sig_shape = [B] + [1] * n_extra + [T]
    sig_expand = signals.view(*sig_shape).expand_as(lo_safe)
    lo_vals = torch.gather(sig_expand, -1, lo_safe)
    hi_vals = torch.gather(sig_expand, -1, hi_safe)
    out = torch.where(valid_lo, (1.0 - frac) * lo_vals, torch.zeros_like(lo_vals))
    out = out + torch.where(valid_hi, frac * hi_vals, torch.zeros_like(hi_vals))
    return out


def render_batched_body_steps(
    ref_scene: CavityScene,
    scenes: list[CavityScene],
    source_signals: _ComplexTensor,
    states: list[CavityStreamState],
    sample_rate: float,
    device: str | torch.device = "cpu",
) -> tuple[_ComplexTensor, list[CavityStreamState]]:
    """Batched cavity solve for B same-type body scenes.

    All scenes must share the same source/receiver/aperture layout and panel
    count.  The per-body jitter lives in the panel tensor values which are
    stacked into a (B, P, ...) batch.

    Parameters
    ----------
    ref_scene   : any one of the scenes (used for source/receiver/aperture config)
    scenes      : list of B body CavityScenes (same type, jitter varies)
    source_signals : (B, T) complex128 — one drive signal per body
    states      : list of B CavityStreamStates
    sample_rate : Hz
    device      : torch device

    Returns
    -------
    chunk_outputs : (B, T) complex128 — receiver signal per body
    new_states    : list of B CavityStreamStates
    """
    dev = torch.device(device)
    B = len(scenes)
    T_chunk = int(source_signals.shape[1])

    # --- Collect stream-state dimensions from the first state ---
    tail_n = int(states[0].tail_samples)
    hist_n = int(states[0].history_samples)

    # --- Pad source signals with tail ---
    chunk_sig = source_signals.to(torch.complex128).to(dev)
    if tail_n > 0:
        pad = torch.zeros((B, tail_n), dtype=torch.complex128, device=dev)
        sig = torch.cat([chunk_sig, pad], dim=1)  # (B, T_ext)
    else:
        sig = chunk_sig
    T_ext = sig.shape[1]

    # --- Stack panel tensors: (B, P, ...) ---
    cached = [_get_cached_scene_tensors(sc, dev) for sc in scenes]
    n_panels = len(cached[0][0])  # panel list length
    # Indices into the cache tuple:
    # 0=panels, 1=src_pos, 2=src_dir, 3=panel_points, 4=panel_normals,
    # 5=panel_reflect, 6=panel_normal_reflect, 7=panel_grazing_reflect,
    # 8=panel_normal_phase, 9=panel_grazing_phase, 10=panel_diffuse, 11=panel_absorb
    b_panel_points   = torch.stack([c[3] for c in cached])   # (B, P, 3)
    b_panel_normals  = torch.stack([c[4] for c in cached])   # (B, P, 3)
    b_panel_reflect  = torch.stack([c[5] for c in cached])   # (B, P)
    b_panel_nrefl    = torch.stack([c[6] for c in cached])   # (B, P)
    b_panel_grefl    = torch.stack([c[7] for c in cached])   # (B, P)
    b_panel_nphase   = torch.stack([c[8] for c in cached])   # (B, P)
    b_panel_gphase   = torch.stack([c[9] for c in cached])   # (B, P)
    b_panel_diffuse  = torch.stack([c[10] for c in cached])  # (B, P)
    b_panel_absorb   = torch.stack([c[11] for c in cached])  # (B, P)

    # --- Shared geometry (same for all bodies of this type) ---
    src_pos = cached[0][1]  # (1, 3) — body has 1 source
    src_dir = cached[0][2]  # (1, 3)
    recv_pos = _as_float_tensor([r.position for r in ref_scene.receivers], dev)  # (1, 3)
    recv_dir = _normalize(_as_float_tensor([r.direction for r in ref_scene.receivers], dev))  # (1, 3)
    recv_pow = torch.tensor([r.directivity_power for r in ref_scene.receivers],
                            dtype=torch.float64, device=dev)  # (1,)
    c = float(ref_scene.speed_of_sound_m_s)
    eps = 1e-9
    diff = ref_scene.diffuse_tail

    # Source-band decomposition is the same for all bodies (same source config)
    ref_source = ref_scene.sources[0]

    # --- Stack source filter histories ---
    if hist_n > 0:
        b_history = torch.stack([st.source_history for st in states]).to(dev)  # (B, 1, hist)
    else:
        b_history = None

    # --- Aperture config (shared) ---
    has_apertures = bool(ref_scene.apertures)
    n_iters = max(0, int(ref_scene.aperture_feedback_iterations)) if has_apertures else 0
    if has_apertures:
        ap_pos = _as_float_tensor([ap.position for ap in ref_scene.apertures], dev)   # (1, 3)
        ap_dir = _normalize(_as_float_tensor([ap.direction for ap in ref_scene.apertures], dev))
        ap_pow = torch.tensor([ap.capture_power for ap in ref_scene.apertures],
                              dtype=torch.float64, device=dev)
        ap = ref_scene.apertures[0]
        fb_gain = max(0.0, float(ap.feedback_gain) * (1.0 - float(ap.passive_loss)))
        fb_phase = float(getattr(ap, "feedback_phase_rad", 0.0))

    # --- Direct path constants (same for all B bodies) ---
    # 1 source at src_pos[0], 1 receiver at recv_pos[0]
    direct_vec = recv_pos[0] - src_pos[0]  # (3,)
    direct_dist = direct_vec.norm().clamp_min(eps)
    direct_delay_s = direct_dist / c
    direct_dir = _normalize(direct_vec.unsqueeze(0))   # (1, 3)
    incoming_dir = _normalize(-direct_vec.unsqueeze(0))  # (1, 3)

    # Specular geometry that varies per body: image source positions
    # sp: (B, P, 3) = src_pos[0] - panel_points
    sp = src_pos[0].unsqueeze(0).unsqueeze(0) - b_panel_points  # (B, P, 3) by broadcast
    plane_dist = (sp * b_panel_normals).sum(dim=-1, keepdim=True)  # (B, P, 1)
    image_pos = src_pos[0].unsqueeze(0).unsqueeze(0) - 2.0 * plane_dist * b_panel_normals  # (B, P, 3)

    # spec_dist: (B, P) — distance from each image source to the single receiver
    spec_dist = (recv_pos[0].unsqueeze(0).unsqueeze(0) - image_pos).norm(dim=-1).clamp_min(eps)  # (B, P)
    spec_delay = spec_dist / c  # (B, P)

    # src_to_panel_dirs: (B, P, 3)
    src_to_panel_dirs = _normalize(image_pos - src_pos[0])  # (B, P, 3)

    # panel_to_recv_dirs: (B, P, 3)
    panel_to_recv_dirs = _normalize(recv_pos[0].unsqueeze(0).unsqueeze(0) - b_panel_points)  # (B, P, 3)

    # Receiver directivity for specular: incoming from image source
    # image_to_recv_dirs: (B, P, 3)
    image_to_recv_dirs = _normalize(image_pos - recv_pos[0])  # (B, P, 3)
    # recv_pow is scalar for body receivers (directivity_power=0 → omni)
    spec_recv_gain = (
        (image_to_recv_dirs * recv_dir[0]).sum(dim=-1).clamp_min(0.0) ** float(recv_pow[0])
    )  # (B, P)

    # Source incidence on panels: (B, P)
    source_incidence = (src_to_panel_dirs * b_panel_normals).sum(dim=-1).abs().clamp(0.0, 1.0)
    # Receiver incidence on panels: (B, P)
    receiver_incidence = (panel_to_recv_dirs * b_panel_normals).sum(dim=-1).abs().clamp(0.0, 1.0)
    incidence_cos = 0.5 * (source_incidence + receiver_incidence)  # (B, P)

    # Panel complex reflection: (B, P)
    blend = incidence_cos.clamp(0.0, 1.0)
    normal_mag = torch.where(b_panel_nrefl >= 0.0, b_panel_nrefl, b_panel_reflect)
    grazing_mag = torch.where(b_panel_grefl >= 0.0, b_panel_grefl, b_panel_reflect)
    magnitude = torch.lerp(grazing_mag, normal_mag, blend)
    phase = torch.lerp(b_panel_gphase, b_panel_nphase, blend)
    panel_coeff = (magnitude * (1.0 - b_panel_absorb)).to(torch.complex128) * torch.exp(
        1j * phase.to(torch.complex128))  # (B, P)

    # Diffuse tail setup
    if n_panels > 0 and diff.taps_per_panel > 0:
        n_taps = diff.taps_per_panel
        tap_idx = torch.arange(n_taps, dtype=torch.float64, device=dev)
        base_phase_vec = 0.61803398875 * tap_idx  # (n_taps,)

        # path_len: (B, P, 1) — src→panel + panel→recv
        src_to_panel_dist = (src_pos[0].unsqueeze(0) - b_panel_points).norm(dim=-1)  # (B, P)
        panel_to_recv_dist = (recv_pos[0].unsqueeze(0).unsqueeze(0) - b_panel_points).norm(dim=-1)  # (B, P)
        diff_path_len = src_to_panel_dist + panel_to_recv_dist  # (B, P)

        diff_base_delay = diff_path_len / c  # (B, P)
        tap_spread = (tap_idx + 1.0) * float(diff.delay_spread_s) / max(n_taps, 1)  # (n_taps,)
        diff_tap_delay = diff_base_delay.unsqueeze(2) + tap_spread[None, None, :]  # (B, P, n_taps)

        diff_tap_gain = (
            float(diff.strength)
            * b_panel_diffuse.unsqueeze(2)          # (B, P, 1)
            * (1.0 - b_panel_absorb.unsqueeze(2))   # (B, P, 1)
            / (1.0 + diff_path_len.unsqueeze(2))    # (B, P, 1)
        )  # (B, P, n_taps)  — still needs band_gain_scalar multiplied later
        diff_atmo = atmospheric_attenuation(ref_scene.atmosphere, diff_path_len.unsqueeze(2))  # (B, P, 1) broadcast

        diff_decay = torch.exp(-diff_tap_delay / max(float(diff.decay_s), 1e-6))  # (B, P, n_taps)

        # recv_cos: (B, P, 1) — directivity toward receiver from each panel
        panel_to_recv_dir_norm = _normalize(b_panel_points - recv_pos[0])  # (B, P, 3) reversed for incoming
        recv_cos = (
            (-panel_to_recv_dir_norm * recv_dir[0]).sum(dim=-1).clamp_min(0.0)
            ** float(recv_pow[0])
        ).unsqueeze(2)  # (B, P, 1)

        # panel_phase: (B, P, n_taps)
        pi_indices = torch.arange(n_panels, dtype=torch.float64, device=dev)
        phase_per_panel = (
            base_phase_vec[None, None, :]
            + b_panel_gphase.unsqueeze(2)
            + 0.5 * (b_panel_nphase.unsqueeze(2) - b_panel_gphase.unsqueeze(2))
            + 0.37 * (pi_indices[None, :, None] + 1.0)
            + 0.13  # si=0 always for body
        )  # (B, P, n_taps)
        diff_panel_phase = torch.exp(1j * phase_per_panel)  # (B, P, n_taps)
        has_diffuse = True
    else:
        has_diffuse = False

    # Aperture geometry if needed (same for all bodies)
    if has_apertures:
        ap_vec = ap_pos[0] - src_pos[0]  # (3,)
        ap_dist = ap_vec.norm().clamp_min(eps)
        ap_delay_s = ap_dist / c
        # Specular: image→aperture distances: (B, P)
        ap_spec_dist = (ap_pos[0].unsqueeze(0).unsqueeze(0) - image_pos).norm(dim=-1).clamp_min(eps)
        ap_spec_delay = ap_spec_dist / c

    # ===== Feedback loop =====
    total_drive = sig.clone()  # (B, T_ext)

    for iter_idx in range(n_iters + 1):
        # Band decomposition (same filter for all bodies, different signals)
        # Use history from the first body to set up bands, but apply per-body
        dummy_history = b_history[:, 0, :] if b_history is not None else None
        # Actually we need to do band decomposition per body since signals differ.
        # But the FILTER COEFFICIENTS are the same — only the signal differs.
        # Use FIR mode: convolve each body's signal with the same filter.
        # For efficiency, get the band profiles once, then apply batched.
        bands_template = _source_bands_for_scene(
            total_drive[0],  # any signal, just to get band structure
            ref_source,
            ref_scene,
            sample_rate=sample_rate,
            filter_history=dummy_history[0] if dummy_history is not None else None,
        )

        recv_accum = torch.zeros((B, T_ext), dtype=torch.complex128, device=dev)
        if has_apertures:
            ap_accum = torch.zeros((B, T_ext), dtype=torch.complex128, device=dev)

        for band_idx, (band, _template_sig) in enumerate(bands_template):
            # Compute band signals for ALL B bodies at once
            # For FIR mode, we need to apply the same filter to each body's signal
            b_band_sigs = torch.stack([
                _source_bands_for_scene(
                    total_drive[b],
                    ref_source,
                    ref_scene,
                    sample_rate=sample_rate,
                    filter_history=dummy_history[b] if dummy_history is not None else None,
                )[band_idx][1]
                for b in range(B)
            ])  # (B, T_ext)

            band_gain_scalar = float(ref_source.gain * band.gain)
            band_power = torch.tensor([band.directivity_power * ref_source.directivity_power],
                                      dtype=torch.float64, device=dev)
            band_cone = torch.tensor([band.cone_angle_deg], dtype=torch.float64, device=dev)
            high_band_weight = (
                0.0 if band.high_hz is not None and float(band.high_hz) <= 500.0
                else (1.0 if float(band.low_hz) >= 1200.0 else 0.35)
            )

            # --- Direct contribution: same delay for all B, different signals ---
            direct_src_gain = _source_directivity_gain(
                src_dir, direct_dir, band_power, cone_angle_deg=band_cone)[0, 0]
            direct_recv_gain = (
                (incoming_dir[0] * recv_dir[0]).sum().clamp_min(0.0) ** float(recv_pow[0])
            )
            direct_gain = (
                band_gain_scalar * direct_src_gain * direct_recv_gain
                / direct_dist
                * atmospheric_attenuation(ref_scene.atmosphere, direct_dist.unsqueeze(0),
                                          high_band_weight=high_band_weight)[0]
            )
            # Delay all B signals by the same direct_delay_s
            direct_delays = direct_delay_s.expand(B)  # (B,)
            direct_delayed = _batched_fractional_delay(b_band_sigs, direct_delays, sample_rate)  # (B, T_ext)
            recv_accum += direct_delayed * direct_gain

            # --- Specular contribution ---
            if n_panels > 0:
                # Source directivity toward each panel: (B, P)
                # src_to_panel_dirs: (B, P, 3), src_dir: (1, 3)
                src_panel_cos = (src_to_panel_dirs * src_dir[0]).sum(dim=-1).clamp_min(0.0)  # (B, P)
                src_panel_gain = src_panel_cos ** float(band_power[0])
                if band_cone[0] < 180.0:
                    half_angle = 0.5 * torch.deg2rad(band_cone[0].clamp_min(1e-6))
                    cosine_edge = torch.cos(half_angle)
                    cone_gain = ((src_panel_cos - cosine_edge).clamp_min(0.0)
                                 / (1.0 - cosine_edge).clamp_min(1e-9))
                    src_panel_gain = src_panel_gain * cone_gain

                spec_gain = (
                    band_gain_scalar
                    * src_panel_gain.to(torch.complex128)
                    * spec_recv_gain.to(torch.complex128)
                    * panel_coeff
                    / spec_dist.to(torch.complex128)
                )  # (B, P) complex
                spec_atmo = atmospheric_attenuation(
                    ref_scene.atmosphere, spec_dist, high_band_weight=high_band_weight
                ).to(torch.complex128)
                spec_gain = spec_gain * spec_atmo

                # Delay: (B, P) delays, (B, T_ext) signals → (B, P, T_ext)
                spec_delayed = _batched_fractional_delay(b_band_sigs, spec_delay, sample_rate)
                # Sum over panels
                recv_accum += (spec_delayed * spec_gain.unsqueeze(-1)).sum(dim=1)  # (B, T_ext)

            # --- Diffuse contribution ---
            if has_diffuse:
                n_taps = diff.taps_per_panel
                # diff_tap_delay: (B, P, n_taps)
                # Flatten to (B, P*n_taps) for fractional delay
                flat_delay = diff_tap_delay.reshape(B, -1)  # (B, P*n_taps)
                flat_delayed = _batched_fractional_delay(
                    b_band_sigs, flat_delay, sample_rate)  # (B, P*n_taps, T_ext)
                delayed_3d = flat_delayed.reshape(B, n_panels, n_taps, T_ext)

                # gain_complex: (B, P, n_taps)
                gain_complex = (
                    band_gain_scalar
                    * diff_tap_gain
                    * diff_atmo
                    * diff_decay
                    * recv_cos
                ).to(torch.complex128)

                contrib = (
                    delayed_3d
                    * gain_complex.unsqueeze(-1)
                    * diff_panel_phase.unsqueeze(-1)
                ).sum(dim=2).sum(dim=1)  # (B, T_ext)
                recv_accum += contrib

            # --- Aperture contribution (if needed) ---
            if has_apertures:
                # Direct to aperture
                ap_direct_gain_scalar = band_gain_scalar * _source_directivity_gain(
                    src_dir, _normalize(ap_pos - src_pos[0]),
                    band_power, cone_angle_deg=band_cone,
                )[0, 0]
                ap_direct_gain = ap_direct_gain_scalar / ap_dist
                ap_direct_gain = ap_direct_gain * atmospheric_attenuation(
                    ref_scene.atmosphere, ap_dist.unsqueeze(0),
                    high_band_weight=high_band_weight)[0]
                ap_cap_pow = float(ap_pow[0])
                ap_direct_recv = (
                    (_normalize(src_pos[0] - ap_pos[0]).unsqueeze(0) * ap_dir[0]).sum().clamp_min(0.0)
                    ** ap_cap_pow
                )
                ap_direct_delays = ap_delay_s.expand(B)
                ap_direct_delayed = _batched_fractional_delay(
                    b_band_sigs, ap_direct_delays, sample_rate)
                ap_accum += ap_direct_delayed * (ap_direct_gain * ap_direct_recv)

                # Specular to aperture
                if n_panels > 0:
                    ap_spec_gain = (
                        band_gain_scalar
                        * src_panel_gain.to(torch.complex128)
                        * panel_coeff
                        / ap_spec_dist.to(torch.complex128)
                    )
                    ap_spec_atmo = atmospheric_attenuation(
                        ref_scene.atmosphere, ap_spec_dist,
                        high_band_weight=high_band_weight).to(torch.complex128)
                    ap_spec_gain = ap_spec_gain * ap_spec_atmo
                    ap_spec_delayed = _batched_fractional_delay(
                        b_band_sigs, ap_spec_delay, sample_rate)
                    ap_accum += (ap_spec_delayed * ap_spec_gain.unsqueeze(-1)).sum(dim=1)

        # --- Aperture feedback ---
        if has_apertures and iter_idx < n_iters:
            feedback = ap_accum * (
                fb_gain * torch.exp(torch.tensor(1j * fb_phase, dtype=torch.complex128, device=dev))
            )
            total_drive = sig + feedback

    # ===== State management (per-body) =====
    chunk_outputs = recv_accum[:, :T_chunk].clone()

    # Add overlap from previous step
    for b in range(B):
        add_n = min(T_chunk, states[b].overlap.shape[1])
        if add_n > 0:
            chunk_outputs[b, :add_n] += states[b].overlap[0, :add_n]

    # Compute new overlaps
    new_states: list[CavityStreamState] = []
    for b in range(B):
        if tail_n > 0:
            new_overlap = recv_accum[b:b+1, T_chunk:T_chunk + tail_n].clone()
            if states[b].overlap.shape[1] > tail_n:
                carry = states[b].overlap[:, tail_n:]
                carry_n = min(carry.shape[1], new_overlap.shape[1])
                new_overlap[:, :carry_n] += carry[:, :carry_n]
        else:
            new_overlap = torch.zeros((1, 0), dtype=torch.complex128, device=dev)

        if hist_n > 0:
            hist_in = torch.cat([states[b].source_history, chunk_sig[b:b+1]], dim=1)
            new_hist = hist_in[:, -hist_n:].clone()
        else:
            new_hist = torch.zeros((1, 0), dtype=torch.complex128, device=dev)

        new_states.append(CavityStreamState(
            overlap=new_overlap,
            tail_samples=tail_n,
            source_history=new_hist,
            history_samples=hist_n,
        ))

    return chunk_outputs, new_states


def _target_field_components(
    scene: CavityScene,
    source_signals: _ComplexTensor,
    target_positions: _FloatTensor,
    target_dirs: _FloatTensor,
    target_powers: _FloatTensor,
    *,
    sample_rate: float,
    source_filter_history: _ComplexTensor | None = None,
    preserve_source_axis: bool = False,
    target_patterns: list[str] | None = None,
) -> tuple[_ComplexTensor, _ComplexTensor, _ComplexTensor]:
    dev = source_signals.device
    sig = source_signals.to(dtype=torch.complex128, device=dev)
    n_sources, n_samples = sig.shape
    n_targets = int(target_positions.shape[0])
    if n_targets == 0:
        empty_shape = (n_sources, 0, n_samples) if preserve_source_axis else (0, n_samples)
        empty = torch.zeros(empty_shape, dtype=torch.complex128, device=dev)
        return empty, empty, empty

    (
        panels, src_pos, src_dir,
        panel_points, panel_normals, panel_reflect,
        panel_normal_reflect, panel_grazing_reflect,
        panel_normal_phase, panel_grazing_phase,
        panel_diffuse, panel_absorb,
    ) = _get_cached_scene_tensors(scene, dev)

    c = float(scene.speed_of_sound_m_s)
    eps = 1e-9

    src_to_target = target_positions[None, :, :] - src_pos[:, None, :]
    direct_dist = src_to_target.norm(dim=-1).clamp_min(eps)
    direct_dirs = _normalize(src_to_target)
    incoming_dirs = _normalize(src_pos[:, None, :] - target_positions[None, :, :])
    direct_delay = direct_dist / c
    target_direct_gain = _receiver_directivity_gain(target_dirs, incoming_dirs, target_powers, target_patterns)

    if panels:
        sp = src_pos[:, None, :] - panel_points[None, :, :]
        plane_dist = (sp * panel_normals[None, :, :]).sum(dim=-1, keepdim=True)
        image_pos = src_pos[:, None, :] - 2.0 * plane_dist * panel_normals[None, :, :]
        spec_dist = (target_positions[None, None, :, :] - image_pos[:, :, None, :]).norm(dim=-1).clamp_min(eps)
        src_to_panel_dirs = _normalize(image_pos - src_pos[:, None, :])
        panel_to_target_dirs = _normalize(target_positions[None, None, :, :] - panel_points[None, :, None, :])
        target_gain = _receiver_directivity_gain(
            target_dirs,
            _normalize(image_pos[:, :, None, :] - target_positions[None, None, :, :]).reshape(-1, n_targets, 3),
            target_powers,
            target_patterns,
        ).reshape(n_sources, len(panels), n_targets)
        spec_delay = spec_dist / c
    else:
        spec_dist = torch.zeros((n_sources, 0, n_targets), dtype=torch.float64, device=dev)
        src_to_panel_dirs = torch.zeros((n_sources, 0, 3), dtype=torch.float64, device=dev)
        panel_to_target_dirs = torch.zeros((1, 0, n_targets, 3), dtype=torch.float64, device=dev)
        target_gain = torch.zeros((n_sources, 0, n_targets), dtype=torch.float64, device=dev)
        spec_delay = torch.zeros((n_sources, 0, n_targets), dtype=torch.float64, device=dev)

    accum_shape = (n_sources, n_targets, n_samples) if preserve_source_axis else (n_targets, n_samples)
    direct_accum = torch.zeros(accum_shape, dtype=torch.complex128, device=dev)
    spec_accum = torch.zeros(accum_shape, dtype=torch.complex128, device=dev)
    diff = scene.diffuse_tail
    diffuse_accum = torch.zeros(accum_shape, dtype=torch.complex128, device=dev)
    tap_idx = torch.arange(diff.taps_per_panel, dtype=torch.float64, device=dev) if panels and diff.taps_per_panel > 0 else None
    base_phase = 0.61803398875 * tap_idx if tap_idx is not None else None

    for si, source in enumerate(scene.sources):
        source_bands = _source_bands_for_scene(
            sig[si],
            source,
            scene,
            sample_rate=sample_rate,
            filter_history=None if source_filter_history is None else source_filter_history[si],
        )
        for band, band_sig in source_bands:
            band_power = torch.tensor([band.directivity_power * source.directivity_power], dtype=torch.float64, device=dev)
            band_cone = torch.tensor([band.cone_angle_deg], dtype=torch.float64, device=dev)
            band_gain_scalar = float(source.gain * band.gain)
            high_band_weight = 0.0 if band.high_hz is not None and float(band.high_hz) <= 500.0 else (1.0 if float(band.low_hz) >= 1200.0 else 0.35)

            direct_gain = band_gain_scalar * _source_directivity_gain(
                src_dir[si:si + 1],
                direct_dirs[si:si + 1],
                band_power,
                cone_angle_deg=band_cone,
            )[0]
            direct_gain = direct_gain * target_direct_gain[si] / direct_dist[si]
            direct_gain = direct_gain * atmospheric_attenuation(scene.atmosphere, direct_dist[si], high_band_weight=high_band_weight)
            delayed = _fractional_delay_kernel(band_sig, direct_delay[si], sample_rate)
            contrib = delayed * direct_gain[:, None]
            if preserve_source_axis:
                direct_accum[si] += contrib
            else:
                direct_accum += contrib

            if panels:
                source_panel_dir_gain = _source_directivity_gain(
                    src_dir[si:si + 1],
                    src_to_panel_dirs[si:si + 1],
                    band_power,
                    cone_angle_deg=band_cone,
                )[0]
                source_incidence = (src_to_panel_dirs[si] * panel_normals).sum(dim=-1).abs().clamp(0.0, 1.0)
                receiver_incidence = (
                    panel_to_target_dirs[0] * panel_normals[:, None, :]
                ).sum(dim=-1).abs().clamp(0.0, 1.0)
                incidence_cos = 0.5 * (source_incidence[:, None] + receiver_incidence)
                panel_coeff = _panel_complex_reflection(
                    panel_reflect,
                    panel_normal_reflect,
                    panel_grazing_reflect,
                    panel_normal_phase,
                    panel_grazing_phase,
                    panel_absorb,
                    incidence_cos[None, :, :],
                )[0]
                spec_gain = (
                    band_gain_scalar
                    * source_panel_dir_gain[:, None].to(torch.complex128)
                    * target_gain[si].to(torch.complex128)
                    * panel_coeff
                    / spec_dist[si].to(torch.complex128)
                )
                spec_gain = spec_gain * atmospheric_attenuation(scene.atmosphere, spec_dist[si], high_band_weight=high_band_weight).to(torch.complex128)
                delayed = _fractional_delay_kernel(band_sig, spec_delay[si], sample_rate)
                contrib = (delayed * spec_gain[..., None]).sum(dim=0)
                if preserve_source_axis:
                    spec_accum[si] += contrib
                else:
                    spec_accum += contrib

                if tap_idx is not None and base_phase is not None:
                    n_panels = len(panels)
                    n_taps = diff.taps_per_panel
                    # path_len: (n_panels, n_targets)
                    src_to_panel_dist = (src_pos[si].unsqueeze(0) - panel_points).norm(dim=-1)  # (n_panels,)
                    panel_to_tgt_dist = (target_positions.unsqueeze(0) - panel_points.unsqueeze(1)).norm(dim=-1)  # (n_panels, n_tgt)
                    path_len_all = src_to_panel_dist.unsqueeze(1) + panel_to_tgt_dist  # (n_panels, n_tgt)

                    base_delay_all = path_len_all / c  # (n_panels, n_tgt)
                    tap_spread = (tap_idx + 1.0) * float(diff.delay_spread_s) / max(n_taps, 1)  # (n_taps,)
                    tap_delay_all = base_delay_all.unsqueeze(2) + tap_spread[None, None, :]  # (n_panels, n_tgt, n_taps)

                    tap_gain_all = (
                        band_gain_scalar
                        * float(diff.strength)
                        * panel_diffuse[:, None, None]
                        * (1.0 - panel_absorb[:, None, None])
                        / (1.0 + path_len_all.unsqueeze(2))
                    )  # (n_panels, n_tgt, n_taps)
                    tap_gain_all = tap_gain_all * atmospheric_attenuation(
                        scene.atmosphere, path_len_all.unsqueeze(2), high_band_weight=high_band_weight)

                    decay_all = torch.exp(-tap_delay_all / max(float(diff.decay_s), 1e-6))

                    # recv_cos: (n_panels, n_tgt)
                    panel_to_recv_dirs = _normalize(panel_points[:, None, :] - target_positions[None, :, :])  # (n_panels, n_tgt, 3)
                    recv_cos_all = (
                        (panel_to_recv_dirs * target_dirs[None, :, :]).sum(dim=-1).clamp_min(0.0)
                        ** target_powers[None, :]
                    ).unsqueeze(2)  # (n_panels, n_tgt, 1)

                    # panel_phase: (n_panels, 1, n_taps)
                    pi_indices = torch.arange(n_panels, dtype=torch.float64, device=dev)
                    phase_per_panel = (
                        base_phase[None, :]
                        + panel_grazing_phase[:, None]
                        + 0.5 * (panel_normal_phase[:, None] - panel_grazing_phase[:, None])
                        + 0.37 * (pi_indices[:, None] + 1.0)
                        + 0.13 * float(si + 1)
                    )  # (n_panels, n_taps)
                    panel_phase_all = torch.exp(1j * phase_per_panel).unsqueeze(1)  # (n_panels, 1, n_taps)

                    # Batched fractional delay: flatten to (n_panels*n_tgt*n_taps,)
                    delayed_all = _fractional_delay_kernel(
                        band_sig,
                        tap_delay_all.reshape(-1),
                        sample_rate,
                    ).reshape(n_panels, n_targets, n_taps, n_samples)

                    # Combine and sum over taps and panels
                    gain_complex = (tap_gain_all * decay_all * recv_cos_all).to(torch.complex128)  # (n_panels, n_tgt, n_taps)
                    contrib = (
                        delayed_all
                        * gain_complex.unsqueeze(-1)
                        * panel_phase_all.unsqueeze(-1)
                    ).sum(dim=2).sum(dim=0)  # (n_tgt, n_samples)
                    if preserve_source_axis:
                        diffuse_accum[si] += contrib
                    else:
                        diffuse_accum += contrib
    return direct_accum, spec_accum, diffuse_accum


def _panel_complex_reflection(
    panel_reflect: _FloatTensor,
    panel_normal_reflect: _FloatTensor,
    panel_grazing_reflect: _FloatTensor,
    panel_normal_phase: _FloatTensor,
    panel_grazing_phase: _FloatTensor,
    panel_absorb: _FloatTensor,
    incidence_cos: _FloatTensor,
) -> _ComplexTensor:
    blend = incidence_cos.clamp(0.0, 1.0)
    normal_mag = torch.where(panel_normal_reflect >= 0.0, panel_normal_reflect, panel_reflect)
    grazing_mag = torch.where(panel_grazing_reflect >= 0.0, panel_grazing_reflect, panel_reflect)
    magnitude = torch.lerp(grazing_mag[None, :, None], normal_mag[None, :, None], blend)
    phase = torch.lerp(panel_grazing_phase[None, :, None], panel_normal_phase[None, :, None], blend)
    return (magnitude * (1.0 - panel_absorb)[None, :, None]).to(torch.complex128) * torch.exp(1j * phase.to(torch.complex128))


def _aperture_pressure_targets(
    scene: CavityScene,
    device: torch.device,
) -> tuple[_FloatTensor, _FloatTensor, _FloatTensor]:
    positions = _as_float_tensor([ap.position for ap in scene.apertures], device)
    directions = _normalize(_as_float_tensor([ap.direction for ap in scene.apertures], device)) if scene.apertures else torch.zeros((0, 3), dtype=torch.float64, device=device)
    powers = torch.tensor([ap.capture_power for ap in scene.apertures], dtype=torch.float64, device=device) if scene.apertures else torch.zeros((0,), dtype=torch.float64, device=device)
    return positions, directions, powers


def _aperture_feedback_signals(
    scene: CavityScene,
    aperture_pressures_by_source: _ComplexTensor,
    *,
    device: torch.device,
) -> _ComplexTensor:
    n_sources = len(scene.sources)
    if not scene.apertures:
        return torch.zeros((n_sources, aperture_pressures_by_source.shape[-1]), dtype=torch.complex128, device=device)
    source_index = _source_index_by_key(scene)
    gains = torch.zeros((len(scene.apertures),), dtype=torch.float64, device=device)
    phases = torch.zeros((len(scene.apertures),), dtype=torch.float64, device=device)
    owner_idx = torch.full((len(scene.apertures),), -1, dtype=torch.int64, device=device)
    for i, ap in enumerate(scene.apertures):
        owner_idx[i] = source_index.get(ap.source_key, -1)
        gains[i] = max(0.0, float(ap.feedback_gain) * (1.0 - float(ap.passive_loss)))
        phases[i] = float(ap.feedback_phase_rad)
    incoming = aperture_pressures_by_source.sum(dim=0)
    valid = owner_idx >= 0
    feedback = torch.zeros((n_sources, aperture_pressures_by_source.shape[-1]), dtype=torch.complex128, device=device)
    if valid.any():
        weighted = incoming[valid] * gains[valid, None].to(torch.complex128) * torch.exp(1j * phases[valid, None].to(torch.complex128))
        feedback.index_add_(0, owner_idx[valid], weighted)
    return feedback


def estimate_cavity_tail_samples(scene: CavityScene, sample_rate: float) -> int:
    """Return a finite overlap horizon for chunked rendering."""

    panels = build_room_panels(scene.geometry)
    source_positions = [s.position for s in scene.sources]
    receiver_positions = [r.position for r in scene.receivers]
    max_distance = 0.0
    for spos in source_positions:
        for rpos in receiver_positions:
            direct = math.dist(spos, rpos)
            max_distance = max(max_distance, direct)
        for panel in panels:
            panel_point = panel.point
            src_to_panel = math.dist(spos, panel_point)
            for rpos in receiver_positions:
                path = src_to_panel + math.dist(panel_point, rpos)
                max_distance = max(max_distance, path)
    direct_spec_s = max_distance / max(scene.speed_of_sound_m_s, 1e-9)
    diff = scene.diffuse_tail
    diffuse_s = (
        diff.taps_per_panel * diff.delay_spread_s
        + max(0.0, diff.truncate_at_s) * max(0.0, diff.decay_s)
    )
    return max(0, int(math.ceil((direct_spec_s + diffuse_s) * sample_rate)))


def init_cavity_stream_state(
    scene: CavityScene,
    sample_rate: float,
    *,
    device: str | torch.device = "cpu",
) -> CavityStreamState:
    receivers = max(1, len(scene.receivers))
    tail_samples = estimate_cavity_tail_samples(scene, sample_rate)
    history_samples = _scene_history_samples(scene)
    n_sources = max(1, len(scene.sources))
    overlap = torch.zeros((receivers, tail_samples), dtype=torch.complex128, device=torch.device(device))
    source_history = torch.zeros((n_sources, history_samples), dtype=torch.complex128, device=torch.device(device))
    return CavityStreamState(
        overlap=overlap,
        tail_samples=tail_samples,
        source_history=source_history,
        history_samples=history_samples,
    )


def render_cavity_scene_step(
    scene: CavityScene,
    source_signals: _ComplexTensor,
    *,
    sample_rate: float,
    state: CavityStreamState | None = None,
    device: str | torch.device | None = None,
) -> CavityStepResult:
    """Render one chunk and preserve overlap for iterative / stepwise solves."""

    if source_signals.ndim != 2:
        raise ValueError("source_signals must have shape (n_sources, n_samples)")
    chunk_n = int(source_signals.shape[1])
    dev = torch.device(device) if device is not None else source_signals.device
    if state is None:
        state = init_cavity_stream_state(scene, sample_rate, device=dev)
    tail_n = int(state.tail_samples)
    hist_n = int(state.history_samples)
    chunk_sig = source_signals.to(torch.complex128).to(dev)
    if tail_n > 0:
        pad = torch.zeros(
            (source_signals.shape[0], tail_n),
            dtype=torch.complex128,
            device=dev,
        )
        extended = torch.cat([chunk_sig, pad], dim=1)
    else:
        extended = chunk_sig

    full = render_cavity_scene(
        scene,
        extended,
        sample_rate=sample_rate,
        device=dev,
        source_filter_history=state.source_history if hist_n > 0 else None,
    )
    chunk_output = full.receiver_signals[:, :chunk_n].clone()
    if state.overlap.numel():
        add_n = min(chunk_n, state.overlap.shape[1])
        chunk_output[:, :add_n] += state.overlap[:, :add_n]

    if tail_n > 0:
        new_overlap = full.receiver_signals[:, chunk_n:chunk_n + tail_n].clone()
        if state.overlap.shape[1] > tail_n:
            carry = state.overlap[:, tail_n:]
            carry_n = min(carry.shape[1], new_overlap.shape[1])
            new_overlap[:, :carry_n] += carry[:, :carry_n]
    else:
        new_overlap = torch.zeros(
            (len(scene.receivers), 0),
            dtype=torch.complex128,
            device=dev,
        )
    if hist_n > 0:
        history_input = torch.cat([state.source_history, chunk_sig], dim=1)
        new_history = history_input[:, -hist_n:].clone()
    else:
        new_history = torch.zeros((source_signals.shape[0], 0), dtype=torch.complex128, device=dev)
    next_state = CavityStreamState(
        overlap=new_overlap,
        tail_samples=tail_n,
        source_history=new_history,
        history_samples=hist_n,
    )
    return CavityStepResult(chunk_output=chunk_output, state=next_state, full_result=full)


def flush_cavity_stream_state(state: CavityStreamState) -> _ComplexTensor:
    """Return the residual tail after the final step."""

    return state.overlap.clone()


def build_test_scene_array(
    geometry_type: Literal["polygon", "circular"] = "polygon",
) -> CavityScene:
    """Return a multi-source, multi-receiver test scene for CLI and tests."""

    sources = [
        CavitySource(
            key="src_a",
            position=(-1.2, -0.4, 1.1),
            direction=(1.0, 0.2, 0.0),
            directivity_power=1.5,
            gain=1.0,
            band_profiles=[
                CavitySourceBand(low_hz=0.0, high_hz=300.0, cone_angle_deg=180.0, directivity_power=0.0, gain=1.0),
                CavitySourceBand(low_hz=300.0, high_hz=1400.0, cone_angle_deg=130.0, directivity_power=1.2, gain=1.0),
                CavitySourceBand(low_hz=1400.0, high_hz=None, cone_angle_deg=70.0, directivity_power=3.0, gain=1.1),
            ],
        ),
        CavitySource(
            key="src_b",
            position=(0.0, 0.8, 1.3),
            direction=(0.0, -1.0, 0.0),
            directivity_power=2.0,
            gain=0.85,
            band_profiles=[
                CavitySourceBand(low_hz=0.0, high_hz=260.0, cone_angle_deg=180.0, directivity_power=0.0, gain=1.0),
                CavitySourceBand(low_hz=260.0, high_hz=1600.0, cone_angle_deg=120.0, directivity_power=1.5, gain=1.0),
                CavitySourceBand(low_hz=1600.0, high_hz=None, cone_angle_deg=60.0, directivity_power=3.5, gain=1.1),
            ],
        ),
        CavitySource(
            key="src_c",
            position=(1.1, -0.7, 1.0),
            direction=(-1.0, 0.3, 0.0),
            directivity_power=1.0,
            gain=0.9,
            band_profiles=[
                CavitySourceBand(low_hz=0.0, high_hz=280.0, cone_angle_deg=180.0, directivity_power=0.0, gain=1.0),
                CavitySourceBand(low_hz=280.0, high_hz=1800.0, cone_angle_deg=140.0, directivity_power=1.0, gain=1.0),
                CavitySourceBand(low_hz=1800.0, high_hz=None, cone_angle_deg=85.0, directivity_power=2.8, gain=1.05),
            ],
        ),
    ]
    receivers = [
        CavityReceiver(key="mic_l", position=(-0.15, 1.6, 1.45), direction=(0.0, -1.0, 0.0), directivity_power=1.0),
        CavityReceiver(key="mic_r", position=(0.15, 1.6, 1.45), direction=(0.0, -1.0, 0.0), directivity_power=1.0),
        CavityReceiver(key="mic_c", position=(0.0, 0.6, 1.25), direction=(0.0, -1.0, 0.0), directivity_power=0.3),
    ]
    apertures = [
        CavityAperture(
            key="ap_src_a",
            source_key="src_a",
            position=(-1.05, -0.35, 1.08),
            direction=(1.0, 0.2, 0.0),
            capture_power=0.5,
            feedback_gain=0.16,
            passive_loss=0.48,
        ),
        CavityAperture(
            key="ap_src_b",
            source_key="src_b",
            position=(0.0, 0.62, 1.26),
            direction=(0.0, -1.0, 0.0),
            capture_power=0.5,
            feedback_gain=0.15,
            passive_loss=0.50,
        ),
        CavityAperture(
            key="ap_src_c",
            source_key="src_c",
            position=(0.95, -0.62, 0.98),
            direction=(-1.0, 0.3, 0.0),
            capture_power=0.5,
            feedback_gain=0.14,
            passive_loss=0.52,
        ),
    ]
    if geometry_type == "circular":
        geometry: PolygonalRoom | CircularRoom = CircularRoom(
            radius=3.4,
            height=3.6,
            n_segments=20,
            baffles=[
                CavityPanel(
                    key="baffle_arc",
                    point=(0.0, 0.0, 1.4),
                    normal=(1.0, 0.0, 0.0),
                    reflectivity=0.82,
                    normal_reflectivity=0.88,
                    grazing_reflectivity=0.68,
                    normal_phase_rad=0.20,
                    grazing_phase_rad=0.65,
                    diffusion=0.55,
                    absorption=0.08,
                    is_baffle=True,
                )
            ],
        )
    else:
        geometry = PolygonalRoom(
            vertices_xy=[(-3.0, -2.2), (3.2, -2.0), (2.8, 2.4), (-2.7, 2.0)],
            height=3.4,
            baffles=[
                CavityPanel(
                    key="baffle_center",
                    point=(0.0, 0.0, 1.5),
                    normal=(1.0, 0.0, 0.0),
                    reflectivity=0.88,
                    normal_reflectivity=0.92,
                    grazing_reflectivity=0.72,
                    normal_phase_rad=0.16,
                    grazing_phase_rad=0.58,
                    diffusion=0.50,
                    absorption=0.06,
                    is_baffle=True,
                )
            ],
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
            strength=0.42,
            truncate_at_s=3.0,
        ),
        aperture_feedback_iterations=1,
    )


def _build_test_source_signals(
    scene: CavityScene,
    *,
    duration_s: float,
    sample_rate: float,
    device: torch.device,
) -> _ComplexTensor:
    n = max(1, int(round(duration_s * sample_rate)))
    t = torch.arange(n, dtype=torch.float64, device=device) / max(sample_rate, 1.0)
    sigs = []
    freqs = [220.0, 330.0, 440.0, 550.0]
    for i, _src in enumerate(scene.sources):
        f = freqs[i % len(freqs)]
        env = torch.exp(-t * (1.5 + 0.4 * i))
        sigs.append((env * torch.exp(1j * (2.0 * math.pi * f * t + 0.25 * i))).to(torch.complex128))
    return torch.stack(sigs, dim=0)


def render_cavity_scene(
    scene: CavityScene,
    source_signals: _ComplexTensor,
    *,
    sample_rate: float,
    device: str | torch.device | None = None,
    source_filter_history: _ComplexTensor | None = None,
) -> CavityRenderResult:
    """Render a shared cavity scene in the complex domain using Torch."""

    dev = torch.device(device) if device is not None else source_signals.device
    sig = source_signals.to(device=dev, dtype=torch.complex128)
    if sig.ndim != 2:
        raise ValueError("source_signals must have shape (n_sources, n_samples)")
    n_sources, n_samples = sig.shape
    if n_sources != len(scene.sources):
        raise ValueError("source_signals first axis must match scene.sources")
    if not scene.receivers:
        raise ValueError("scene must contain at least one receiver")
    if source_filter_history is not None:
        history = source_filter_history.to(device=dev, dtype=torch.complex128)
        if history.ndim != 2 or history.shape[0] != n_sources:
            raise ValueError("source_filter_history must have shape (n_sources, n_history_samples)")
    else:
        history = None
    recv_pos = _as_float_tensor([r.position for r in scene.receivers], dev)
    recv_dir = _normalize(_as_float_tensor([r.direction for r in scene.receivers], dev))
    recv_pow = torch.tensor([r.directivity_power for r in scene.receivers], dtype=torch.float64, device=dev)
    recv_patterns = [getattr(r, "polar_pattern", "omni") for r in scene.receivers]
    total_drive = sig.clone()
    aperture_pressures = torch.zeros((len(scene.apertures), n_samples), dtype=torch.complex128, device=dev)
    source_index = _source_index_by_key(scene)
    n_iters = max(0, int(scene.aperture_feedback_iterations))

    for iter_idx in range(n_iters + 1):
        direct_accum, spec_accum, diffuse_accum = _target_field_components(
            scene,
            total_drive,
            recv_pos,
            recv_dir,
            recv_pow,
            sample_rate=sample_rate,
            source_filter_history=history,
            preserve_source_axis=False,
            target_patterns=recv_patterns,
        )
        if not scene.apertures:
            break
        ap_pos, ap_dir, ap_pow = _aperture_pressure_targets(scene, dev)
        ap_direct, ap_spec, ap_diffuse = _target_field_components(
            scene,
            total_drive,
            ap_pos,
            ap_dir,
            ap_pow,
            sample_rate=sample_rate,
            source_filter_history=history,
            preserve_source_axis=True,
        )
        aperture_by_source = ap_direct + ap_spec + ap_diffuse
        owner_idx = torch.tensor([source_index.get(ap.source_key, -1) for ap in scene.apertures], dtype=torch.int64, device=dev)
        valid = owner_idx >= 0
        if valid.any():
            aperture_by_source[owner_idx[valid], valid, :] = 0.0 + 0.0j
        aperture_pressures = aperture_by_source.sum(dim=0)
        if iter_idx < n_iters:
            feedback = _aperture_feedback_signals(scene, aperture_by_source, device=dev)
            total_drive = sig + feedback

    panels = build_room_panels(scene.geometry)
    total = direct_accum + spec_accum + diffuse_accum
    return CavityRenderResult(
        receiver_signals=total,
        direct_signals=direct_accum,
        specular_signals=spec_accum,
        diffuse_signals=diffuse_accum,
        aperture_pressures=aperture_pressures,
        panel_count=len(panels),
        metadata={
            "receiver_keys": [r.key for r in scene.receivers],
            "aperture_keys": [ap.key for ap in scene.apertures],
            "panel_keys": [p.key for p in panels],
            "geometry_type": type(scene.geometry).__name__,
            "band_limited_directionality": any(source.band_profiles for source in scene.sources),
            "incidence_dependent_reflection": bool(panels),
            "band_split_mode": scene.band_split_mode,
            "aperture_feedback_iterations": n_iters,
        },
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline cavity-engine test renderer")
    p.add_argument("--scene", choices=("polygon", "circular"), default="polygon")
    p.add_argument("--duration", type=float, default=0.5)
    p.add_argument("--sample-rate", type=float, default=2048.0)
    p.add_argument("--chunk-size", type=int, default=0,
                   help="If >0, render in streaming steps of this many samples")
    p.add_argument("--device", default="cpu")
    p.add_argument("--band-split-mode", choices=("fir", "fft"), default="fir")
    p.add_argument("--fir-taps", type=int, default=129)
    p.add_argument("--json", action="store_true",
                   help="Print summary as JSON")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    scene = build_test_scene_array(args.scene)
    scene.band_split_mode = str(args.band_split_mode)
    scene.band_split_fir_taps = int(args.fir_taps)
    source_signals = _build_test_source_signals(
        scene,
        duration_s=float(args.duration),
        sample_rate=float(args.sample_rate),
        device=device,
    )
    if args.chunk_size and args.chunk_size > 0:
        state = init_cavity_stream_state(scene, float(args.sample_rate), device=device)
        chunks: list[torch.Tensor] = []
        for start in range(0, source_signals.shape[1], int(args.chunk_size)):
            chunk = source_signals[:, start:start + int(args.chunk_size)]
            step = render_cavity_scene_step(
                scene,
                chunk,
                sample_rate=float(args.sample_rate),
                state=state,
                device=device,
            )
            state = step.state
            chunks.append(step.chunk_output)
        tail = flush_cavity_stream_state(state)
        rendered = torch.cat(chunks + ([tail] if tail.numel() else []), dim=1)
        direct = None
        specular = None
        diffuse = None
        aperture_pressures = None
        panel_count = len(build_room_panels(scene.geometry))
    else:
        result = render_cavity_scene(
            scene,
            source_signals,
            sample_rate=float(args.sample_rate),
            device=device,
        )
        rendered = result.receiver_signals
        direct = result.direct_signals
        specular = result.specular_signals
        diffuse = result.diffuse_signals
        aperture_pressures = result.aperture_pressures
        panel_count = result.panel_count

    summary = {
        "scene": args.scene,
        "device": str(device),
        "n_sources": len(scene.sources),
        "n_receivers": len(scene.receivers),
        "n_apertures": len(scene.apertures),
        "panel_count": panel_count,
        "sample_rate": float(args.sample_rate),
        "duration_s": float(args.duration),
        "chunk_size": int(args.chunk_size),
        "band_split_mode": scene.band_split_mode,
        "fir_taps": int(scene.band_split_fir_taps),
        "aperture_feedback_iterations": int(scene.aperture_feedback_iterations),
        "receiver_peak": float(torch.max(torch.abs(rendered)).item()) if rendered.numel() else 0.0,
        "direct_peak": (float(torch.max(torch.abs(direct)).item()) if direct is not None and direct.numel() else None),
        "specular_peak": (float(torch.max(torch.abs(specular)).item()) if specular is not None and specular.numel() else None),
        "diffuse_peak": (float(torch.max(torch.abs(diffuse)).item()) if diffuse is not None and diffuse.numel() else None),
        "aperture_pressure_peak": (
            float(torch.max(torch.abs(aperture_pressures)).item())
            if aperture_pressures is not None and aperture_pressures.numel()
            else None
        ),
        "component_peaks_available": direct is not None,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
